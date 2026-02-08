"""Test that commit() produces predictions matching apply() for latent ratios.

This module tests for a critical bug where the aggregation commit() function
did not match what apply() does for latent ratios, causing:
1. Missing clipping to [ratio_min, ratio_max]
2. Wrong normalization (ratios/min instead of ratios/sum)

This caused logged predictions during design to differ from final committed predictions.
"""

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
from biocomp.design_targets import SVGTarget
from biocomp.nodes.aggregation import _decode_latent_ratios
from biocomp.ratio_schema import get_slot_entries

RESOURCES_DIR = Path(__file__).parent / "resources"


SCAFFOLD_PATH = RESOURCES_DIR / "design/architectures/two_and_one.yaml"


def get_svg_target():
    svg_path = str(RESOURCES_DIR / "designs/MIT_T.svg")
    if not os.path.exists(svg_path):
        pytest.skip(f"SVG target not found: {svg_path}")
    return SVGTarget(
        name="MIT_T (test)",
        path=svg_path,
        transform_to_log_space=False,
        latent_x=(0.0, 0.6),
        latent_y=(0.0, 0.6),
    )


@pytest.fixture
def lib():
    return load_lib()


def load_scaffold_recipe():
    data = dr.load(SCAFFOLD_PATH, context={"Recipe": Recipe})
    if hasattr(data, "recipes") or (hasattr(data, "__getitem__") and "recipes" in data):
        recipes = data["recipes"] if "recipes" in data else data.recipes
        return recipes[0]
    if isinstance(data, Recipe):
        return data
    raise ValueError(f"Unexpected scaffold format: {type(data)}")


def test_latent_ratios_commit_matches_apply(lib):
    """Test that commit() produces same effective ratios as apply() with latent ratios.

    This is a regression test for the bug where commit() did not:
    1. Clip latent-decoded ratios to [ratio_min, ratio_max]
    2. Normalize by sum (instead normalized by min)

    This caused logged predictions during design to differ from final committed predictions.
    """
    if not SCAFFOLD_PATH.exists():
        pytest.skip(f"Scaffold not found: {SCAFFOLD_PATH}")

    from biocomp.design import DesignManager
    from biocomptools.modelmodel import BiocompModel

    model_path = Path(os.environ.get('BIOCOMP_DESIGNER_MODEL', ''))
    if not model_path.exists():
        pytest.skip("Designer model not found at BIOCOMP_DESIGNER_MODEL")
    model = BiocompModel.load(model_path)

    with LibraryContext.with_library(lib):
        scaffold_recipe = load_scaffold_recipe()
        networks = recipe_to_networks(scaffold_recipe, br.ALL_RULES, invert=True)

        target = get_svg_target()
        dmanager = DesignManager(
            targets=[target],
            networks=networks[:1],
            enable_tu_masking=False,
        )

        stack = dmanager.build_stack(model, unlock_ratios=True, use_latent_ratios=True)
        key = jax.random.key(42)
        params = stack.init(key)

        agg_layer = None
        for layer in stack.layers:
            if layer.f_type == "aggregation":
                agg_layer = layer
                break
        assert agg_layer is not None, "No aggregation layer found"

        ns = agg_layer.namespace
        latent_z_path = f"{ns}/latent_z"
        assert latent_z_path in params, "Latent ratios not initialized"

        i = 0
        z = params[f"{ns}/latent_z"][i]
        W1 = params[f"{ns}/latent_W1"][i]
        b1 = params[f"{ns}/latent_b1"][i]
        W2 = params[f"{ns}/latent_W2"][i]
        b2 = params[f"{ns}/latent_b2"][i]
        ratio_min = params[f"{ns}/ratio_min"][i]
        ratio_max = params[f"{ns}/ratio_max"][i]

        raw_ratios = _decode_latent_ratios(z, W1, b1, W2, b2)
        n_outputs = len(ratio_min)
        raw_ratios = raw_ratios[:n_outputs]
        apply_ratios = jnp.clip(raw_ratios, ratio_min, ratio_max)
        apply_ratios = jnp.abs(apply_ratios)
        apply_normalized = apply_ratios / jnp.sum(apply_ratios)

        committed_networks = stack.commit(params)
        committed_net = committed_networks[0]

        committed_node = None
        for n in committed_net.compute_graph.nodes.values():
            if n.node_type == "aggregation":
                committed_node = n
                break
        assert committed_node is not None

        slot_entries = get_slot_entries(committed_node.extra)
        assert len(slot_entries) > 0, "No ratio_schema slots in committed node"
        commit_ratios = np.array([entry["ratio"] for entry in slot_entries])

        apply_normalized = np.array(apply_normalized)
        commit_sum = np.sum(commit_ratios)
        if commit_sum > 1e-8:
            commit_normalized = commit_ratios / commit_sum
        else:
            commit_normalized = commit_ratios

        rel_diff = np.abs(commit_normalized - apply_normalized) / (np.abs(apply_normalized) + 1e-8)
        max_rel_diff = float(np.max(rel_diff))

        assert max_rel_diff < 0.05, (
            f"Ratios differ between apply() and commit():\n"
            f"  apply_normalized: {apply_normalized}\n"
            f"  commit_normalized: {commit_normalized}\n"
            f"  max_rel_diff: {max_rel_diff:.2%}"
        )


def test_design_training_predictions_match_committed_predictions(lib):
    """End-to-end test: predictions during training must match committed network predictions.

    This is the core regression test. It compares:
    - yhat from stack.apply() (what logger shows during training)
    - yhat from committed network via NetworkModel (what final summary shows)

    Both should produce very similar predictions.
    """
    if not SCAFFOLD_PATH.exists():
        pytest.skip(f"Scaffold not found: {SCAFFOLD_PATH}")

    from biocomp.design import DesignManager, DesignConfig, initialize_params
    from biocomptools.modelmodel import BiocompModel, NetworkModel

    model_path = Path(os.environ.get('BIOCOMP_DESIGNER_MODEL', ''))
    if not model_path.exists():
        pytest.skip("Designer model not found at BIOCOMP_DESIGNER_MODEL")
    model = BiocompModel.load(model_path)

    with LibraryContext.with_library(lib):
        scaffold_recipe = load_scaffold_recipe()
        networks = recipe_to_networks(scaffold_recipe, br.ALL_RULES, invert=True)

        target = get_svg_target()
        dmanager = DesignManager(
            targets=[target],
            networks=networks[:1],
            enable_tu_masking=False,
        )
        DesignConfig(
            n_replicates=1,
            n_epochs=1,
            batch_size=64,
            n_batches_per_epoch=5,
            use_latent_ratios=True,
        )

        stack = dmanager.build_stack(model, unlock_ratios=True, use_latent_ratios=True)
        key = jax.random.key(42)
        params_full = initialize_params(
            stack, n_replicates=1, n_targets=1, shared_params=model.shared_params, key=key
        )
        params = jax.tree.map(lambda x: x[0, 0], params_full)

        X_lat, Y_target = target.get_lattice(resolution=(20, 20), seed=0)
        X_lat = jnp.array(X_lat)
        Y_target = np.array(Y_target).ravel()
        n_samples = X_lat.shape[0]

        num_z = int(jnp.squeeze(params["global/number_of_random_variables"]))
        z_batch = jnp.zeros((n_samples, num_z))
        keys = jax.random.split(key, n_samples)

        def predict_single(x, z, k):
            return stack.apply(params, x, z, k, tu_enabled_random_vars=None)

        yhat_apply, _ = jax.vmap(predict_single)(X_lat, z_batch, keys)
        dep_mask = stack.get_dependent_output_mask()
        yhat_apply = jnp.compress(dep_mask, yhat_apply, axis=-1)[:, 0]
        yhat_apply = np.array(yhat_apply).ravel()

        committed_networks = stack.commit(params)
        committed_net = committed_networks[0]

        nm = NetworkModel(model=model, network=committed_net)
        yhat_commit_raw, _ = nm.predict(X_lat, z_value=0.0)
        yhat_commit = np.array(yhat_commit_raw[:, 3]).ravel()

        pred_corr = np.corrcoef(yhat_apply, yhat_commit)[0, 1]
        assert pred_corr > 0.85, (
            f"Raw predictions don't match:\n"
            f"  correlation between apply and commit predictions: {pred_corr:.4f}"
        )
