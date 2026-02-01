"""Tests for hard-pruning functionality in design mode."""

import pytest
import numpy as np
from pathlib import Path
import jax
import dracon as dr

from biocomp.design import DesignConfig, DesignManager, initialize_params
from biocomp.design_pruning import _merge_surviving_params, identify_tus_to_prune, hard_prune_and_rebuild
from biocomp.tumasking import TU_LOG_ALPHA_PATH
from biocomp.parameters import ParameterTree
from biocomp.design_targets import SVGTarget
from biocomp.config import DEFAULT_COMPUTE_CONFIG
from biocomp.datautils import IdentityRescaler
from biocomptools.modelmodel import BiocompModel
from biocomp.library import LibraryContext, load_lib
from biocomp.network import recipe_to_networks
from biocomp.recipe import Recipe
from biocomp.tumasking import extract_tu_ids_from_network
import biocomp.biorules as br
from biocomp.jaxutils import tree_get


class TestHardPruningConstraints:
    """Test that hard-pruning enforces single replicate/target."""

    def test_config_accepts_single_replicate_single_target(self):
        """DesignConfig can be created with hard_pruning_enabled=True."""
        dconf = DesignConfig(
            n_replicates=1,
            n_epochs=1,
            hard_pruning_enabled=True,
        )
        assert dconf.hard_pruning_enabled is True
        assert dconf.n_replicates == 1

    def test_hard_pruning_interval_is_configurable(self):
        """hard_pruning_interval can be set."""
        dconf = DesignConfig(
            n_replicates=1,
            n_epochs=1,
            hard_pruning_enabled=True,
            hard_pruning_interval=250,
        )
        assert dconf.hard_pruning_interval == 250

    def test_hard_pruning_preserve_minimum_is_configurable(self):
        """hard_pruning_preserve_minimum_tus can be set."""
        dconf = DesignConfig(
            n_replicates=1,
            n_epochs=1,
            hard_pruning_enabled=True,
            hard_pruning_preserve_minimum_tus=2,
        )
        assert dconf.hard_pruning_preserve_minimum_tus == 2


class TestMergeSurvivingParams:
    """Test parameter carry-over between pruning cycles."""

    def test_merge_copies_matching_paths(self):
        """Params with matching paths and shapes are copied."""
        old_params = ParameterTree()
        new_params = ParameterTree()

        old_val = np.array([[1.0, 2.0]])
        new_val = np.zeros_like(old_val)

        old_params.at("local/layer_0/ratios", old_val, overwrite=None)
        new_params.at("local/layer_0/ratios", new_val, overwrite=None)

        merged = _merge_surviving_params(old_params, new_params)
        assert np.allclose(merged["local/layer_0/ratios"], old_val)

    def test_merge_skips_tu_log_alpha(self):
        """tu_log_alpha paths are skipped (handled separately)."""
        old_params = ParameterTree()
        new_params = ParameterTree()

        old_val = np.array([[5.0]])
        new_val = np.array([[2.0]])

        old_params.at(TU_LOG_ALPHA_PATH, old_val, overwrite=None)
        new_params.at(TU_LOG_ALPHA_PATH, new_val, overwrite=None)

        merged = _merge_surviving_params(old_params, new_params)
        assert np.allclose(merged[TU_LOG_ALPHA_PATH], new_val)

    def test_merge_skips_shape_mismatch(self):
        """Paths with different shapes are not copied."""
        old_params = ParameterTree()
        new_params = ParameterTree()

        old_val = np.array([[1.0, 2.0, 3.0]])
        new_val = np.array([[0.0, 0.0]])

        old_params.at("local/layer_0/ratios", old_val, overwrite=None)
        new_params.at("local/layer_0/ratios", new_val, overwrite=None)

        merged = _merge_surviving_params(old_params, new_params)
        assert np.allclose(merged["local/layer_0/ratios"], new_val)

    def test_merge_skips_latent_tu_paths(self):
        """latent_tu paths are skipped (handled separately)."""
        old_params = ParameterTree()
        new_params = ParameterTree()

        old_val = np.array([[5.0]])
        new_val = np.array([[2.0]])

        old_params.at("design/latent_tu_z", old_val, overwrite=None)
        new_params.at("design/latent_tu_z", new_val, overwrite=None)

        merged = _merge_surviving_params(old_params, new_params)
        assert np.allclose(merged["design/latent_tu_z"], new_val)

    def test_merge_handles_missing_path_in_new(self):
        """Paths in old but not new are ignored."""
        old_params = ParameterTree()
        new_params = ParameterTree()

        old_val = np.array([[1.0, 2.0]])
        new_val = np.array([[0.0, 0.0]])

        old_params.at("local/layer_0/old_only", old_val, overwrite=None)
        old_params.at("local/layer_0/both", old_val, overwrite=None)
        new_params.at("local/layer_0/both", new_val, overwrite=None)

        merged = _merge_surviving_params(old_params, new_params)
        assert "local/layer_0/old_only" not in merged
        assert np.allclose(merged["local/layer_0/both"], old_val)


class TestDesignConfigHardPruning:
    """Test DesignConfig with hard pruning settings."""

    def test_default_hard_pruning_disabled(self):
        """Hard pruning is disabled by default."""
        dconf = DesignConfig()
        assert dconf.hard_pruning_enabled is False

    def test_hard_pruning_defaults(self):
        """Check default values for hard pruning params."""
        dconf = DesignConfig(hard_pruning_enabled=True)
        assert dconf.hard_pruning_interval == 500
        assert dconf.hard_pruning_ratio_threshold == 0.01
        assert dconf.hard_pruning_preserve_minimum_tus == 1


RESOURCES_DIR = Path(__file__).parent / "resources"
SCAFFOLD_PATH = RESOURCES_DIR / "design/architectures/two_and_one.yaml"


@pytest.fixture
def lib():
    return load_lib()


def _load_scaffold_recipe():
    data = dr.load(SCAFFOLD_PATH, context={"Recipe": Recipe})
    if hasattr(data, "recipes") or (hasattr(data, "__getitem__") and "recipes" in data):
        recipes = data["recipes"] if "recipes" in data else data.recipes
        return recipes[0]
    if isinstance(data, Recipe):
        return data
    raise ValueError(f"Unexpected scaffold format: {type(data)}")


def _make_model():
    return BiocompModel(
        compute_config=DEFAULT_COMPUTE_CONFIG,
        rescaler=IdentityRescaler(),
        shared_params=ParameterTree(),
    )


def _set_low_ratios(params: ParameterTree, stack, threshold: float) -> None:
    updated = False
    assert stack.layers is not None, "Stack must be built before setting ratios"
    for layer in stack.layers:
        if layer.f_type != "aggregation":
            continue
        namespace = layer.namespace
        assert namespace is not None
        ratio_path = f"{namespace}/ratios"
        if ratio_path not in params:
            continue
        ratio_min_path = f"{namespace}/ratio_min"
        ratio_max_path = f"{namespace}/ratio_max"
        ratios = np.asarray(params[ratio_path])
        ratio_min = np.asarray(params[ratio_min_path])
        ratio_max = np.asarray(params[ratio_max_path])
        low_vals = np.where(ratio_min < threshold, ratio_min, ratios)
        low_vals = np.minimum(low_vals, threshold * 0.5)
        keep_slot = 0
        if ratios.shape[1] > keep_slot:
            low_vals[:, keep_slot] = np.maximum(ratio_max[:, keep_slot], threshold * 2)
        params.at(ratio_path, low_vals, overwrite=True)
        updated = updated or bool(np.any(ratio_min < threshold))
    assert updated, "No aggregation ratios below threshold in scaffold"


def test_hard_prune_removes_low_ratios_without_masking(lib):
    if not SCAFFOLD_PATH.exists():
        pytest.skip(f"Scaffold not found: {SCAFFOLD_PATH}")

    model = _make_model()

    with LibraryContext.with_library(lib):
        recipe = _load_scaffold_recipe()
        networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True)

        target = SVGTarget(
            name="MIT_T (prune test)",
            path=RESOURCES_DIR / "designs/MIT_T.svg",
            transform_to_log_space=False,
            latent_x=(0.0, 0.6),
            latent_y=(0.0, 0.6),
        )
        dmanager = DesignManager(
            targets=[target],
            networks=networks[:1],
            enable_tu_masking=False,
        )

        stack = dmanager.build_stack(model, unlock_ratios=True)
        params_full = initialize_params(
            stack,
            n_replicates=1,
            n_targets=1,
            shared_params=model.shared_params,
            key=jax.random.key(0),
            n_tus=0,
            n_networks=len(dmanager.networks),
            no_masking_tu_ids=stack.no_masking_tu_ids,
            tu_id_to_idx=stack.tu_id_to_idx,
        )
        params = tree_get(params_full, (0, 0))
        _set_low_ratios(params, stack, threshold=0.11)

        tus_to_remove = identify_tus_to_prune(
            params,
            stack,
            dmanager,
            ratio_threshold=0.11,
            use_soft_pruning=False,
            preserve_minimum=1,
        )
        assert tus_to_remove[0], "No TUs selected for pruning"

        before = len(extract_tu_ids_from_network(networks[0]))
        dconf = DesignConfig(n_replicates=1, n_epochs=1)
        new_dmanager, _, _ = hard_prune_and_rebuild(
            dmanager,
            dconf,
            model,
            stack,
            params,
            tus_to_remove,
            jax.random.key(1),
        )
        after = len(extract_tu_ids_from_network(new_dmanager.networks[0]))

        assert after < before


def test_identify_tus_to_prune_returns_mapped_ids(lib):
    if not SCAFFOLD_PATH.exists():
        pytest.skip(f"Scaffold not found: {SCAFFOLD_PATH}")

    model = _make_model()

    with LibraryContext.with_library(lib):
        recipe = _load_scaffold_recipe()
        networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True)

        target = SVGTarget(
            name="MIT_T (mapping test)",
            path=RESOURCES_DIR / "designs/MIT_T.svg",
            transform_to_log_space=False,
            latent_x=(0.0, 0.6),
            latent_y=(0.0, 0.6),
        )
        dmanager = DesignManager(
            targets=[target],
            networks=networks[:1],
            enable_tu_masking=False,
        )

        stack = dmanager.build_stack(model, unlock_ratios=True)
        params_full = initialize_params(
            stack,
            n_replicates=1,
            n_targets=1,
            shared_params=model.shared_params,
            key=jax.random.key(2),
            n_tus=0,
            n_networks=len(dmanager.networks),
            no_masking_tu_ids=stack.no_masking_tu_ids,
            tu_id_to_idx=stack.tu_id_to_idx,
        )
        params = tree_get(params_full, (0, 0))
        _set_low_ratios(params, stack, threshold=0.11)

        tus_to_remove = identify_tus_to_prune(
            params,
            stack,
            dmanager,
            ratio_threshold=0.11,
            use_soft_pruning=False,
            preserve_minimum=1,
        )

        stack.ensure_tu_mapping()
        tu_id_set = set(stack.tu_id_to_idx.keys())
        assert tus_to_remove[0].issubset(tu_id_set)
