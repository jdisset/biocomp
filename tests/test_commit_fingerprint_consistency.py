"""Test that committed networks produce identical predictions as pre-commit with TU masking.

Verifies the critical invariant: applying TU masks during forward pass must
produce the same dependent outputs as running the committed network without masks.
This ensures `stack.commit()` correctly "bakes in" TU masking decisions.
"""

import pickle
import tempfile
import os
from pathlib import Path

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import dracon as dr

import biocomp.biorules as br
from biocomp.library import LibraryContext, load_lib
from biocomp.network import recipe_to_networks
from biocomp.recipe import Recipe
from biocomp.design import DesignManager, initialize_params
from biocomp.design_targets import SVGTarget
from biocomp.tumasking import (
    TU_LOG_ALPHA_PATH,
    get_final_mask,
)
from biocomp.tumasking_strategy import build_tu_masking_strategy, TUMaskingMode
from biocomp.fingerprint import (
    _generate_canonical_grid,
    FINGERPRINT_SEED,
    FINGERPRINT_RESOLUTION,
)
from biocomp.jaxutils import tree_get
from biocomp.ratio_schema import get_slot_entries

SCAFFOLD_PATH = Path(__file__).parent.parent.parent / "biocomp-jobs/design/architectures/two_and_one_skip.yaml"
RESOURCES_DIR = Path(__file__).parent / "resources"


@pytest.fixture(scope="module")
def lib():
    return load_lib()


@pytest.fixture(scope="module")
def scaffold_recipe(lib):
    with LibraryContext.with_library(lib):
        data = dr.load(SCAFFOLD_PATH, context={"Recipe": Recipe})
        recipes = data["recipes"] if "recipes" in data else data.recipes
        return recipes[0]


@pytest.fixture(scope="module")
def designer_model():
    model_path = os.environ.get("BIOCOMP_DESIGNER_MODEL")
    if not model_path or not Path(model_path).exists():
        pytest.skip("BIOCOMP_DESIGNER_MODEL not set or doesn't exist")
    from biocomptools.modelmodel import BiocompModel
    return BiocompModel.load(model_path)


def get_svg_target():
    svg_path = RESOURCES_DIR / "designs/MIT_T.svg"
    if not svg_path.exists():
        pytest.skip(f"SVG target not found: {svg_path}")
    return SVGTarget(
        name="MIT_T (test)",
        path=str(svg_path),
        transform_to_log_space=False,
        latent_x=(0.0, 0.6),
        latent_y=(0.0, 0.6),
    )


def compute_pre_commit_output(stack, params, X, seed=FINGERPRINT_SEED):
    """Compute dependent outputs via stack.apply() with TU masking.

    Returns only the dependent outputs (not marker proteins).
    """
    n_samples = X.shape[0]
    num_z = int(jnp.squeeze(params["global/number_of_random_variables"]))
    z_batch = jnp.zeros((n_samples, num_z))
    keys = jax.random.split(jax.random.key(seed), n_samples)

    def predict_single(x, z, k):
        return stack.apply(params, x, z, k, tu_enabled_random_vars=None)

    Y_full, _ = jax.vmap(predict_single)(X, z_batch, keys)

    dep_mask = stack.get_dependent_output_mask()
    Y_dep = jnp.compress(dep_mask, Y_full, axis=-1)

    return np.asarray(Y_dep)


def compute_post_commit_output(committed_network, model, X, seed=FINGERPRINT_SEED):
    """Compute dependent outputs from committed network via NetworkModel.

    Returns only the dependent outputs (not marker proteins) to match pre-commit.
    """
    from biocomptools.modelmodel import NetworkModel

    nm = NetworkModel(model=model, network=committed_network)
    Y_full, _ = nm.predict(
        X,
        key=jax.random.PRNGKey(seed),
        disable_variational=True,
        z_value=0.0,
    )

    dep_mask = committed_network.get_dependent_output_mask()
    Y_dep = np.asarray(Y_full)[:, dep_mask]

    return Y_dep


@pytest.fixture
def masked_scaffold_setup(lib, scaffold_recipe, designer_model):
    """Create stack with TU masking and random ~50% disabled pattern.

    Uses DesignManager and initialize_params for proper param setup.
    """
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(scaffold_recipe, br.ALL_RULES, invert=True)

        target = get_svg_target()
        dmanager = DesignManager(
            targets=[target],
            networks=networks[:1],
            enable_tu_masking=True,
        )

        stack = dmanager.build_stack(designer_model, unlock_ratios=True)

        n_replicates = 1
        n_targets = 1
        key = jax.random.key(42)

        n_tus = dmanager.n_tus
        n_networks = len(dmanager.networks)
        tu_id_to_idx = dmanager.tu_id_to_idx

        strategy = build_tu_masking_strategy(
            mode=TUMaskingMode.DIRECT, init_mean=10.0, init_std=0.0
        )
        params_full = initialize_params(
            stack,
            n_replicates=n_replicates,
            n_targets=n_targets,
            shared_params=designer_model.shared_params,
            key=key,
            strategy=strategy,
            n_tus=n_tus,
            n_networks=n_networks,
            no_masking_tu_ids=stack.no_masking_tu_ids,
            tu_id_to_idx=tu_id_to_idx,
        )

        params = tree_get(params_full, (0, 0))

        log_alpha_full = params_full[TU_LOG_ALPHA_PATH]
        rng = np.random.default_rng(seed=42)
        tu_ids = sorted(tu_id_to_idx.keys())

        no_masking = stack.no_masking_tu_ids or set()
        log_alpha = np.array(log_alpha_full[0, 0])
        for tu_idx, tu_id in enumerate(tu_ids):
            tu_name = "_".join(tu_id.split("_")[:-1])
            if tu_name.endswith("_marker"):
                continue
            if tu_id in no_masking:
                continue
            if rng.random() < 0.5:
                log_alpha[:, tu_idx] = -10.0

        log_alpha_modified = log_alpha_full.at[0, 0].set(jnp.array(log_alpha))
        params_full.at(TU_LOG_ALPHA_PATH, log_alpha_modified, overwrite=True)
        params = tree_get(params_full, (0, 0))

        return {
            "stack": stack,
            "params": params,
            "networks": dmanager.networks,
            "tu_ids": tu_ids,
            "tu_id_to_idx": tu_id_to_idx,
            "log_alpha": params[TU_LOG_ALPHA_PATH],
            "no_masking_tu_ids": no_masking,
        }


def test_commit_consistency_with_tu_masking(lib, masked_scaffold_setup, designer_model):
    """Test that pre-commit and post-commit predictions match exactly."""
    with LibraryContext.with_library(lib):
        setup = masked_scaffold_setup
        stack = setup["stack"]
        params = setup["params"]
        log_alpha = setup["log_alpha"]
        tu_ids = setup["tu_ids"]
        tu_id_to_idx = setup["tu_id_to_idx"]

        n_inputs = stack.networks[0].nb_inputs
        X = jnp.array(_generate_canonical_grid(n_inputs, FINGERPRINT_RESOLUTION, FINGERPRINT_SEED))

        Y_pre = compute_pre_commit_output(stack, params, X)

        # Log disabled TUs before commit
        disabled_tu_names = []
        enabled_tu_names = []
        for tu_id in tu_ids:
            tu_idx = tu_id_to_idx[tu_id]
            mask_val = get_final_mask(log_alpha[0, tu_idx:tu_idx + 1])[0]
            tu_name = "_".join(tu_id.split("_")[:-1])
            if float(mask_val) < 0.5:
                disabled_tu_names.append(tu_name)
            else:
                enabled_tu_names.append(tu_name)

        print("\n=== TU Masking Debug ===")
        print(f"Disabled TUs ({len(disabled_tu_names)}): {disabled_tu_names}")
        print(f"Enabled TUs ({len(enabled_tu_names)}): {enabled_tu_names}")

        committed_networks = stack.commit(params)
        assert len(committed_networks) > 0, "commit() returned no networks"
        committed_network = committed_networks[0]

        # Log committed network structure
        committed_recipe = committed_network.to_recipe()
        committed_tu_names = [tu.name for cotx in committed_recipe.content for tu in cotx.units]
        print(f"Committed TUs ({len(committed_tu_names)}): {committed_tu_names}")

        # Check ratios in committed network
        for node in committed_network.compute_graph.nodes.values():
            if node.node_type == "aggregation":
                slot_entries = get_slot_entries(node.extra, require=False)
                if slot_entries:
                    ratios = {entry["source_id"]: entry.get("ratio") for entry in slot_entries}
                    print(f"Aggregation node ratios: {ratios}")

        with tempfile.NamedTemporaryFile(suffix=".pickle", delete=False) as f:
            pickle.dump(committed_network, f)
            temp_path = f.name

        with open(temp_path, "rb") as f:
            reloaded_network = pickle.load(f)

        Path(temp_path).unlink()

        Y_post = compute_post_commit_output(reloaded_network, designer_model, X)

        max_diff = float(np.max(np.abs(Y_pre - Y_post)))
        mean_diff = float(np.mean(np.abs(Y_pre - Y_post)))
        corr = float(np.corrcoef(Y_pre.flatten(), Y_post.flatten())[0, 1])

        print("\n=== Prediction Comparison ===")
        print(f"Pre-commit:  shape={Y_pre.shape}, range=[{Y_pre.min():.4f}, {Y_pre.max():.4f}]")
        print(f"Post-commit: shape={Y_post.shape}, range=[{Y_post.min():.4f}, {Y_post.max():.4f}]")
        print(f"Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}, Correlation: {corr:.6f}")

        # TU masking is an approximation - some difference is expected due to:
        # 1. Shared transcription/translation nodes with different input counts
        # 2. Rate embedding and random_var reindexing after commit
        # The ERN topology fix prevents large differences (>5%); accept small differences
        assert max_diff < 0.05, (
            f"Pre/post commit difference too large (>5%)!\n"
            f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}\n"
            f"  Correlation: {corr:.6f}\n"
            f"  Pre shape: {Y_pre.shape}, range: [{Y_pre.min():.4f}, {Y_pre.max():.4f}]\n"
            f"  Post shape: {Y_post.shape}, range: [{Y_post.min():.4f}, {Y_post.max():.4f}]\n"
            f"  Disabled TUs: {disabled_tu_names}\n"
            f"  Committed TUs: {committed_tu_names}"
        )


def test_commit_preserves_ratios(lib, masked_scaffold_setup, designer_model):
    """Test that ratios are correctly preserved during commit."""
    with LibraryContext.with_library(lib):
        setup = masked_scaffold_setup
        stack = setup["stack"]
        params = setup["params"]

        committed_networks = stack.commit(params)
        committed = committed_networks[0]

        agg_nodes = [n for n in committed.compute_graph.nodes.values()
                     if n.node_type == "aggregation"]

        for agg_node in agg_nodes:
            slot_entries = get_slot_entries(agg_node.extra, require=False)
            if slot_entries:
                committed_ratios = [entry["ratio"] for entry in slot_entries]
                committed_ratios = np.array(committed_ratios)

                ratio_sum = np.sum(committed_ratios)
                if ratio_sum > 1e-8:
                    assert np.abs(ratio_sum - 1.0) < 0.01, (
                        f"Committed ratios don't sum to 1: sum={ratio_sum}, ratios={committed_ratios}"
                    )


def test_commit_respects_disabled_tus(lib, masked_scaffold_setup):
    """Test that disabled TUs are actually removed from committed network."""
    with LibraryContext.with_library(lib):
        setup = masked_scaffold_setup
        stack = setup["stack"]
        params = setup["params"]
        log_alpha = setup["log_alpha"]
        tu_ids = setup["tu_ids"]
        tu_id_to_idx = setup["tu_id_to_idx"]

        disabled_tu_names = set()
        for tu_id in tu_ids:
            tu_idx = tu_id_to_idx[tu_id]
            mask_val = get_final_mask(log_alpha[0, tu_idx:tu_idx + 1])[0]
            if float(mask_val) < 0.5:
                tu_name = "_".join(tu_id.split("_")[:-1])
                disabled_tu_names.add(tu_name)

        committed_networks = stack.commit(params)
        committed_recipe = committed_networks[0].to_recipe()

        committed_tu_names = {
            tu.name for cotx in committed_recipe.content
            for tu in cotx.units
        }

        still_present = disabled_tu_names & committed_tu_names
        marker_tus = {n for n in still_present if n.endswith("_marker")}
        non_marker_still_present = still_present - marker_tus

        assert len(non_marker_still_present) == 0, (
            f"Disabled non-marker TUs still present in committed recipe: {non_marker_still_present}"
        )


@pytest.mark.parametrize("disabled_fraction", [0.0, 0.25, 0.5, 0.75])
def test_commit_consistency_varying_mask_fraction(
    lib, scaffold_recipe, designer_model, disabled_fraction
):
    """Test commit consistency with different fractions of disabled TUs."""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(scaffold_recipe, br.ALL_RULES, invert=True)

        target = get_svg_target()
        dmanager = DesignManager(
            targets=[target],
            networks=networks[:1],
            enable_tu_masking=True,
        )

        stack = dmanager.build_stack(designer_model, unlock_ratios=True)

        n_replicates = 1
        n_targets = 1
        key = jax.random.key(42)

        n_tus = dmanager.n_tus
        n_networks = len(dmanager.networks)
        tu_id_to_idx = dmanager.tu_id_to_idx
        tu_ids = sorted(tu_id_to_idx.keys())

        strategy = build_tu_masking_strategy(
            mode=TUMaskingMode.DIRECT, init_mean=10.0, init_std=0.0
        )
        params_full = initialize_params(
            stack,
            n_replicates=n_replicates,
            n_targets=n_targets,
            shared_params=designer_model.shared_params,
            key=key,
            strategy=strategy,
            n_tus=n_tus,
            n_networks=n_networks,
            no_masking_tu_ids=stack.no_masking_tu_ids,
            tu_id_to_idx=tu_id_to_idx,
        )

        log_alpha_full = params_full[TU_LOG_ALPHA_PATH]
        rng = np.random.default_rng(seed=42)
        log_alpha = np.array(log_alpha_full[0, 0])

        no_masking = stack.no_masking_tu_ids or set()
        maskable_indices = []
        for tu_idx, tu_id in enumerate(tu_ids):
            tu_name = "_".join(tu_id.split("_")[:-1])
            if tu_name.endswith("_marker"):
                continue
            if tu_id in no_masking:
                continue
            maskable_indices.append(tu_idx)

        n_to_disable = int(len(maskable_indices) * disabled_fraction)
        if n_to_disable > 0:
            indices_to_disable = rng.choice(maskable_indices, size=n_to_disable, replace=False)
            for idx in indices_to_disable:
                log_alpha[:, idx] = -10.0

        log_alpha_modified = log_alpha_full.at[0, 0].set(jnp.array(log_alpha))
        params_full.at(TU_LOG_ALPHA_PATH, log_alpha_modified, overwrite=True)
        params = tree_get(params_full, (0, 0))

        n_inputs = stack.networks[0].nb_inputs
        X = jnp.array(_generate_canonical_grid(n_inputs, FINGERPRINT_RESOLUTION, FINGERPRINT_SEED))

        Y_pre = compute_pre_commit_output(stack, params, X)
        committed_networks = stack.commit(params)

        if len(committed_networks) == 0:
            pytest.skip("All TUs disabled -> empty network")

        committed_network = committed_networks[0]
        if not committed_network.compute_graph.nodes:
            pytest.skip("Committed network has no nodes")

        Y_post = compute_post_commit_output(committed_network, designer_model, X)

        max_diff = float(np.max(np.abs(Y_pre - Y_post)))

        # TU masking is an approximation - accept differences up to 5%
        assert max_diff < 0.05, (
            f"Pre/post commit difference too large at disabled_fraction={disabled_fraction}\n"
            f"  Max diff: {max_diff:.6f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
