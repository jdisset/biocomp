"""Tests for hard-pruning functionality in design mode."""

import pytest
import numpy as np
from pathlib import Path
import jax
import jax.numpy as jnp
import dracon as dr
from pydantic import ValidationError

from biocomp.design import DesignConfig, DesignManager, initialize_params
from biocomp.design_pruning import (
    _merge_surviving_params,
    _expand_params_for_merge,
    _store_learned_ratio_inits,
    _collect_ratio_pruning_candidates,
    _compute_hard_pruning_network_keep_count,
    _select_top_network_indices_from_losses,
    identify_tus_to_prune,
    hard_prune_and_rebuild,
    run_with_hard_pruning,
)
from biocomp.tumasking import TU_LOG_ALPHA_PATH
from biocomp.parameters import ParameterTree
from biocomp.design_targets import SVGTarget
from biocomp.config import DEFAULT_COMPUTE_CONFIG
from biocomp.datautils import IdentityRescaler
from biocomp.step_history import StepHistorySnapshot
from biocomptools.modelmodel import BiocompModel
from biocomp.library import LibraryContext, load_lib
from biocomp.network import recipe_to_networks
from biocomp.recipe import Recipe
from biocomp.tumasking import extract_tu_ids_from_network
from biocomp.tumasking_strategy import TUMaskingMode, build_strategy_from_config
from biocomp.ratio_schema import get_slot_entries
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
        assert dconf.hard_pruning_top_percent is None
        assert dconf.hard_pruning_min_networks is None

    def test_hard_pruning_prune_margin_is_configurable(self):
        """Verify prune_margin parameter can be set."""
        dconf = DesignConfig(
            hard_pruning_enabled=True,
            hard_pruning_prune_margin=0.15,
        )
        assert dconf.hard_pruning_prune_margin == 0.15

    def test_hard_pruning_prune_margin_default(self):
        """Verify prune_margin defaults to 0.1."""
        dconf = DesignConfig(hard_pruning_enabled=True)
        assert dconf.hard_pruning_prune_margin == 0.1

    def test_hard_pruning_network_selection_fields_are_configurable(self):
        dconf = DesignConfig(
            hard_pruning_enabled=True,
            hard_pruning_top_percent=25.0,
            hard_pruning_min_networks=3,
        )
        assert dconf.hard_pruning_top_percent == 25.0
        assert dconf.hard_pruning_min_networks == 3

    def test_hard_pruning_top_percent_validation(self):
        with pytest.raises(ValidationError, match="hard_pruning_top_percent"):
            DesignConfig(hard_pruning_enabled=True, hard_pruning_top_percent=0.0)

    def test_hard_pruning_min_networks_validation(self):
        with pytest.raises(ValidationError, match="hard_pruning_min_networks"):
            DesignConfig(hard_pruning_enabled=True, hard_pruning_min_networks=0)


class TestHardPruningTopNetworkSelection:
    def test_keep_count_uses_max_of_percent_and_min(self):
        keep_count = _compute_hard_pruning_network_keep_count(
            n_networks=20,
            top_percent=10.0,
            min_networks=5,
        )
        assert keep_count == 5

    def test_keep_count_returns_none_when_disabled(self):
        keep_count = _compute_hard_pruning_network_keep_count(
            n_networks=20,
            top_percent=None,
            min_networks=None,
        )
        assert keep_count is None

    def test_select_top_network_indices_averages_over_non_network_axes(self):
        losses = np.array([[[0.9, 0.2, 0.6, 0.3]]], dtype=np.float32)
        top_idx = _select_top_network_indices_from_losses(losses, keep_count=2)
        assert top_idx == [1, 3]


def test_run_with_hard_pruning_returns_snapshot_step_history(lib):
    if not SCAFFOLD_PATH.exists():
        pytest.skip(f"Scaffold not found: {SCAFFOLD_PATH}")

    model = _make_model()

    with LibraryContext.with_library(lib):
        recipe = _load_scaffold_recipe()
        networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True)
        target = SVGTarget(
            name="MIT_T (hard prune return type)",
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
        init_stack = dmanager.build_stack(model, unlock_ratios=True)
        model.shared_params = _extract_shared_params(init_stack)
        dconf = DesignConfig(
            n_replicates=1,
            n_epochs=1,
            n_batches_per_epoch=1,
            batch_size=1,
            batches_per_step=1,
            reshuffle_batches=False,
            hard_pruning_enabled=True,
            hard_pruning_interval=1,
        )

        _, loss_history, step_history, _ = run_with_hard_pruning(
            dmanager=dmanager,
            dconf=dconf,
            model=model,
        )

    assert len(loss_history) > 0
    assert isinstance(step_history, StepHistorySnapshot)
    assert "loss" in step_history


def test_run_with_hard_pruning_multiple_replicates(lib):
    """Hard pruning with n_replicates>1 flattens replicates into networks."""
    if not SCAFFOLD_PATH.exists():
        pytest.skip(f"Scaffold not found: {SCAFFOLD_PATH}")

    model = _make_model()

    with LibraryContext.with_library(lib):
        recipe = _load_scaffold_recipe()
        networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True)
        n_original_networks = len(networks[:1])
        n_replicates = 2

        target = SVGTarget(
            name="MIT_T (multi-rep hard prune)",
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
        init_stack = dmanager.build_stack(model, unlock_ratios=True)
        model.shared_params = _extract_shared_params(init_stack)
        dconf = DesignConfig(
            n_replicates=n_replicates,
            n_epochs=1,
            n_batches_per_epoch=1,
            batch_size=1,
            batches_per_step=1,
            reshuffle_batches=False,
            hard_pruning_enabled=True,
            hard_pruning_interval=1,
        )

        params, loss_history, step_history, returned_dmanager = run_with_hard_pruning(
            dmanager=dmanager,
            dconf=dconf,
            model=model,
        )

    assert len(loss_history) > 0
    assert isinstance(step_history, StepHistorySnapshot)

    # Returned dmanager should have flattened networks (n_replicates * original)
    # or fewer if some were pruned
    assert len(returned_dmanager.networks) <= n_replicates * n_original_networks
    assert len(returned_dmanager.networks) >= 1

    # Params should have replicate dim = 1 (flattened)
    for _, val in params.data.iter_leaves():
        if hasattr(val, "shape") and val.ndim >= 2:
            assert val.shape[0] == 1, (
                f"Expected replicate dim = 1 after flattening, got shape {val.shape}"
            )
            break


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


def _extract_shared_params(stack) -> ParameterTree:
    """Get properly initialized shared params from a stack."""
    full = stack.init(jax.random.key(999))
    shared, _ = full.filter_by_tag(["shared"])
    return shared


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
        ratios = np.array(params[ratio_path], copy=True)
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


def test_collect_ratio_pruning_candidates_prunes_only_if_tu_is_weak_everywhere():
    """A TU appearing in multiple slots should be pruned only if all slots are weak."""
    from types import SimpleNamespace

    class _FakeLayer:
        f_type = "aggregation"
        namespace = "local/agg"

        def __init__(self):
            self.nodes = [SimpleNamespace(network_id=0)]

        def get_n_outputs(self):
            return 2

    # Same TU mapped to both slots: one strong, one weak.
    params = ParameterTree()
    params.at("local/agg/ratios", np.array([[0.9, 0.01]], dtype=np.float32), overwrite=None)
    params.at("local/agg/ratio_min", np.array([[0.0, 0.0]], dtype=np.float32), overwrite=None)
    params.at("local/agg/ratio_max", np.array([[1.0, 1.0]], dtype=np.float32), overwrite=None)
    params.at("local/agg/output_tu_indices", np.array([[0, 0]], dtype=np.int32), overwrite=None)

    fake_edge = SimpleNamespace(extra=SimpleNamespace(tu_id=["tuA"]))
    fake_network = SimpleNamespace(compute_graph=SimpleNamespace(edges={0: fake_edge}))
    fake_stack = SimpleNamespace(
        tu_id_to_idx={"tuA": 0},
        layers=[_FakeLayer()],
        networks=[fake_network],
    )

    candidates, all_tu_ids, tu_strengths = _collect_ratio_pruning_candidates(
        params=params,
        stack=fake_stack,
        network_id=0,
        ratio_threshold=0.1,
    )

    assert "tuA" in all_tu_ids
    assert tu_strengths["tuA"] == pytest.approx(0.9)
    assert "tuA" not in candidates


def _find_first_rate_edge(stack, params, rate_name: str):
    if stack.layers is None:
        return None
    for layer in stack.layers:
        if layer.f_type not in ("transcription", "translation"):
            continue
        namespace = layer.namespace
        if namespace is None:
            continue
        rate_path = f"{namespace}/{rate_name}"
        if rate_path not in params:
            continue
        for node_idx, node in enumerate(layer.nodes):
            edges = sorted(node.get_incoming_edges(stack), key=lambda e: e.to_input_slot)
            for input_idx, edge in enumerate(edges):
                tu_ids = edge.extra.get("tu_id", []) if edge.extra else []
                if tu_ids:
                    return namespace, node_idx, input_idx, tu_ids[0]
    return None


def _find_aggregation_member(stack):
    if stack.layers is None:
        return None
    for layer in stack.layers:
        if layer.f_type != "aggregation":
            continue
        namespace = layer.namespace
        if namespace is None:
            continue
        for node_idx, node in enumerate(layer.nodes):
            graph_node = node.get(stack)
            slot_entries = get_slot_entries(graph_node.extra, require=False)
            if slot_entries:
                source_id = str(slot_entries[0]["source_id"])
                return namespace, node_idx, source_id
    return None


def _find_bias_node(stack):
    if stack.layers is None:
        return None
    for layer in stack.layers:
        if layer.f_type not in ("bias", "hard_bias"):
            continue
        namespace = layer.namespace
        if namespace is None:
            continue
        for node_idx, node in enumerate(layer.nodes):
            graph_node = node.get(stack)
            extra = graph_node.extra or {}
            fluo_specs = extra.get("fluo_bias") or extra.get("fluo_bias_data") or {}
            if isinstance(fluo_specs, dict):
                protein = fluo_specs.get("protein")
                if protein:
                    return namespace, node_idx, protein
    return None


def _get_param_value(params: ParameterTree, path: str, node_idx: int, input_idx: int | None = None):
    arr = np.asarray(params[path])
    if input_idx is None:
        return arr[node_idx]
    return arr[node_idx, input_idx]


def _set_param_value(params: ParameterTree, path: str, node_idx: int, value, input_idx: int | None = None):
    arr = np.array(params[path], copy=True)
    if input_idx is None:
        arr[node_idx] = value
    else:
        arr[node_idx, input_idx] = value
    params.at(path, arr, overwrite=True)


def _find_value_in_new_stack(stack, params, tu_id: str, rate_name: str):
    if stack.layers is None:
        return None
    for layer in stack.layers:
        if layer.f_type not in ("transcription", "translation"):
            continue
        namespace = layer.namespace
        if namespace is None:
            continue
        rate_path = f"{namespace}/{rate_name}"
        if rate_path not in params:
            continue
        arr = np.asarray(params[rate_path])
        for node_idx, node in enumerate(layer.nodes):
            edges = sorted(node.get_incoming_edges(stack), key=lambda e: e.to_input_slot)
            for input_idx, edge in enumerate(edges):
                tu_ids = edge.extra.get("tu_id", []) if edge.extra else []
                if tu_id in tu_ids:
                    return arr[node_idx, input_idx]
    return None


def _find_ratio_in_new_stack(stack, params, source_id: str):
    if stack.layers is None:
        return None
    for layer in stack.layers:
        if layer.f_type != "aggregation":
            continue
        namespace = layer.namespace
        if namespace is None:
            continue
        ratio_path = f"{namespace}/ratios"
        if ratio_path not in params:
            continue
        arr = np.asarray(params[ratio_path])
        for node_idx, node in enumerate(layer.nodes):
            graph_node = node.get(stack)
            slot_entries = get_slot_entries(graph_node.extra, require=False)
            source_ids = [str(entry["source_id"]) for entry in slot_entries]
            if source_id in source_ids:
                idx = source_ids.index(source_id)
                return arr[node_idx, idx]
    return None


def _find_bias_in_new_stack(stack, params, protein: str):
    if stack.layers is None:
        return None
    for layer in stack.layers:
        if layer.f_type not in ("bias", "hard_bias"):
            continue
        namespace = layer.namespace
        if namespace is None:
            continue
        raw_path = f"{namespace}/raw_value"
        if raw_path not in params:
            continue
        arr = np.asarray(params[raw_path])
        for node_idx, node in enumerate(layer.nodes):
            graph_node = node.get(stack)
            extra = graph_node.extra or {}
            fluo_specs = extra.get("fluo_bias") or extra.get("fluo_bias_data") or {}
            if isinstance(fluo_specs, dict) and fluo_specs.get("protein") == protein:
                return arr[node_idx]
    return None


def test_expand_params_for_merge_expands_scalars():
    params = ParameterTree()
    params.at("local/x", np.array(3.0), overwrite=None)
    expanded = _expand_params_for_merge(params)
    assert expanded["local/x"].shape == (1, 1)


def test_hard_prune_preserves_semantic_params(lib):
    if not SCAFFOLD_PATH.exists():
        pytest.skip(f"Scaffold not found: {SCAFFOLD_PATH}")

    model = _make_model()

    with LibraryContext.with_library(lib):
        recipe = _load_scaffold_recipe()
        networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True)
        target = SVGTarget(
            name="MIT_T (semantic carryover)",
            path=RESOURCES_DIR / "designs/MIT_T.svg",
            transform_to_log_space=False,
            latent_x=(0.0, 0.6),
            latent_y=(0.0, 0.6),
        )
        dmanager = DesignManager(targets=[target], networks=networks[:1], enable_tu_masking=False)
        stack = dmanager.build_stack(model, unlock_ratios=True)
        params_full = initialize_params(
            stack,
            n_replicates=1,
            n_targets=1,
            shared_params=model.shared_params,
            key=jax.random.key(3),
            n_tus=0,
            n_networks=len(dmanager.networks),
            no_masking_tu_ids=stack.no_masking_tu_ids,
            tu_id_to_idx=stack.tu_id_to_idx,
        )
        params = tree_get(params_full, (0, 0))

        agg_info = _find_aggregation_member(stack)
        rate_info = _find_first_rate_edge(stack, params, "tc_rate") or _find_first_rate_edge(
            stack, params, "tl_rate"
        )
        bias_info = _find_bias_node(stack)

        if agg_info is None or rate_info is None or bias_info is None:
            pytest.skip("Required nodes not found in scaffold")

        agg_ns, agg_node_idx, source_id = agg_info
        rate_ns, rate_node_idx, rate_input_idx, tu_id = rate_info
        bias_ns, bias_node_idx, protein = bias_info

        ratio_path = f"{agg_ns}/ratios"
        rate_name = "tc_rate" if f"{rate_ns}/tc_rate" in params else "tl_rate"
        rate_path = f"{rate_ns}/{rate_name}"
        bias_path = f"{bias_ns}/raw_value"

        ratio_value = 0.234
        rate_value = np.array([0.123])
        bias_value = np.array([0.456])

        ratios = np.array(params[ratio_path], copy=True)
        agg_layer = next(l for l in stack.layers if l.namespace == agg_ns)
        agg_node = agg_layer.nodes[agg_node_idx].get(stack)
        source_ids = [str(entry["source_id"]) for entry in get_slot_entries(agg_node.extra, require=False)]
        if source_id in source_ids:
            source_idx = source_ids.index(source_id)
            ratios[agg_node_idx, source_idx] = ratio_value
        else:
            ratios[agg_node_idx, 0] = ratio_value
        params.at(ratio_path, ratios, overwrite=True)
        _set_param_value(params, rate_path, rate_node_idx, rate_value, input_idx=rate_input_idx)
        _set_param_value(params, bias_path, bias_node_idx, bias_value)

        stack.ensure_tu_mapping()
        tu_ids = list(stack.tu_id_to_idx.keys())
        remove_tu_id = next((t for t in tu_ids if t != tu_id), None)
        if remove_tu_id is None:
            pytest.skip("No removable TU found")
        tus_to_remove = {0: {remove_tu_id}}

        dconf = DesignConfig(n_replicates=1, n_epochs=1)
        new_dmanager, new_stack, new_params = hard_prune_and_rebuild(
            dmanager,
            dconf,
            model,
            stack,
            params,
            tus_to_remove,
            jax.random.key(4),
        )

        new_params_flat = tree_get(new_params, (0, 0))
        new_rate = _find_value_in_new_stack(new_stack, new_params_flat, tu_id, rate_name)
        new_ratio = _find_ratio_in_new_stack(new_stack, new_params_flat, source_id)
        new_bias = _find_bias_in_new_stack(new_stack, new_params_flat, protein)

        assert new_rate is not None
        assert new_ratio is not None
        assert new_bias is not None
        assert np.allclose(new_rate, rate_value, atol=1e-6)
        assert np.allclose(new_ratio, ratio_value, atol=1e-6)
        assert np.allclose(new_bias, bias_value, atol=1e-6)


def test_hard_prune_removes_masked_tus(lib):
    if not SCAFFOLD_PATH.exists():
        pytest.skip(f"Scaffold not found: {SCAFFOLD_PATH}")

    model = _make_model()

    with LibraryContext.with_library(lib):
        recipe = _load_scaffold_recipe()
        networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True)
        target = SVGTarget(
            name="MIT_T (mask prune)",
            path=RESOURCES_DIR / "designs/MIT_T.svg",
            transform_to_log_space=False,
            latent_x=(0.0, 0.6),
            latent_y=(0.0, 0.6),
        )
        dmanager = DesignManager(targets=[target], networks=networks[:1], enable_tu_masking=True)
        stack = dmanager.build_stack(model, unlock_ratios=True)

        dconf = DesignConfig(n_replicates=1, n_epochs=1)
        dconf.tu_masking.mode = TUMaskingMode.DIRECT
        params_full = initialize_params(
            stack,
            n_replicates=1,
            n_targets=1,
            shared_params=model.shared_params,
            key=jax.random.key(5),
            strategy=build_strategy_from_config(dconf),
            n_tus=dmanager.n_tus,
            n_networks=len(dmanager.networks),
            no_masking_tu_ids=stack.no_masking_tu_ids,
            tu_id_to_idx=stack.tu_id_to_idx,
        )
        params = tree_get(params_full, (0, 0))

        stack.ensure_tu_mapping()
        tu_ids = [t for t in stack.tu_id_to_idx.keys() if t not in (stack.no_masking_tu_ids or set())]
        if not tu_ids:
            pytest.skip("No prunable TUs found")
        tu_id = tu_ids[0]
        tu_idx = stack.tu_id_to_idx[tu_id]

        log_alpha = np.array(params[TU_LOG_ALPHA_PATH], copy=True)
        log_alpha[...] = 10.0
        log_alpha[:, tu_idx] = -10.0
        params.at(TU_LOG_ALPHA_PATH, log_alpha, overwrite=True)

        tus_to_remove = identify_tus_to_prune(
            params,
            stack,
            dmanager,
            ratio_threshold=-1.0,
            use_soft_pruning=True,
            preserve_minimum=1,
        )

        assert tu_id in tus_to_remove[0]

        new_dmanager, _, _ = hard_prune_and_rebuild(
            dmanager,
            dconf,
            model,
            stack,
            params,
            tus_to_remove,
            jax.random.key(6),
        )

        remaining = extract_tu_ids_from_network(new_dmanager.networks[0])
        assert tu_id not in remaining


def test_hard_prune_top_network_selection_remaps_tu_log_alpha_rows(lib):
    if not SCAFFOLD_PATH.exists():
        pytest.skip(f"Scaffold not found: {SCAFFOLD_PATH}")

    model = _make_model()

    with LibraryContext.with_library(lib):
        recipe = _load_scaffold_recipe()
        networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True)
        if not networks:
            pytest.skip("No networks found in scaffold")
        net_a = networks[0].model_copy(deep=True)
        net_b = networks[0].model_copy(deep=True)
        net_a.name = f"{net_a.name}_A"
        net_b.name = f"{net_b.name}_B"

        target = SVGTarget(
            name="MIT_T (top network tu remap)",
            path=RESOURCES_DIR / "designs/MIT_T.svg",
            transform_to_log_space=False,
            latent_x=(0.0, 0.6),
            latent_y=(0.0, 0.6),
        )
        dmanager = DesignManager(targets=[target], networks=[net_a, net_b], enable_tu_masking=True)
        stack = dmanager.build_stack(model, unlock_ratios=True)

        dconf = DesignConfig(n_replicates=1, n_epochs=1)
        dconf.tu_masking.mode = TUMaskingMode.DIRECT
        params_full = initialize_params(
            stack,
            n_replicates=1,
            n_targets=1,
            shared_params=model.shared_params,
            key=jax.random.key(12),
            strategy=build_strategy_from_config(dconf),
            n_tus=dmanager.n_tus,
            n_networks=len(dmanager.networks),
            no_masking_tu_ids=stack.no_masking_tu_ids,
            tu_id_to_idx=stack.tu_id_to_idx,
        )
        params = tree_get(params_full, (0, 0))

        old_log_alpha = np.asarray(params[TU_LOG_ALPHA_PATH])
        assert old_log_alpha.ndim == 2
        assert old_log_alpha.shape[0] == 2

        stack.ensure_tu_mapping()
        tus_to_remove = {0: set(), 1: set()}
        new_dmanager, new_stack, new_params_batched = hard_prune_and_rebuild(
            dmanager,
            dconf,
            model,
            stack,
            params,
            tus_to_remove,
            jax.random.key(13),
            keep_network_indices=[1],
        )

        assert len(new_dmanager.networks) == 1

        new_params = tree_get(new_params_batched, (0, 0))
        new_log_alpha = np.asarray(new_params[TU_LOG_ALPHA_PATH])
        assert new_log_alpha.ndim == 2
        assert new_log_alpha.shape[0] == 1

        old_tu_map = stack.tu_id_to_idx or {}
        new_tu_map = new_stack.tu_id_to_idx or {}
        for tu_id, new_idx in new_tu_map.items():
            if tu_id not in old_tu_map:
                continue
            old_idx = old_tu_map[tu_id]
            assert np.allclose(new_log_alpha[0, new_idx], old_log_alpha[1, old_idx], atol=1e-6)


def test_hard_prune_commit_keeps_outputs(lib):
    if not SCAFFOLD_PATH.exists():
        pytest.skip(f"Scaffold not found: {SCAFFOLD_PATH}")

    model = _make_model()

    with LibraryContext.with_library(lib):
        recipe = _load_scaffold_recipe()
        networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True)
        target = SVGTarget(
            name="MIT_T (commit outputs)",
            path=RESOURCES_DIR / "designs/MIT_T.svg",
            transform_to_log_space=False,
            latent_x=(0.0, 0.6),
            latent_y=(0.0, 0.6),
        )
        dmanager = DesignManager(targets=[target], networks=networks[:1], enable_tu_masking=False)
        stack = dmanager.build_stack(model, unlock_ratios=True)

        params_full = initialize_params(
            stack,
            n_replicates=1,
            n_targets=1,
            shared_params=model.shared_params,
            key=jax.random.key(7),
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

        new_dmanager, _, _ = hard_prune_and_rebuild(
            dmanager,
            DesignConfig(n_replicates=1, n_epochs=1),
            model,
            stack,
            params,
            tus_to_remove,
            jax.random.key(8),
        )

        net = new_dmanager.networks[0]
        assert net.compute_graph is not None
        n_outputs = sum(1 for n in net.compute_graph.nodes.values() if n.node_type == "output")
        assert n_outputs >= 1


def _compress_dependent_outputs(stack, y):
    dep_mask = stack.get_dependent_output_mask()
    if dep_mask is None:
        return y
    return np.asarray(y)[..., np.asarray(dep_mask)]


def test_soft_mask_vs_hard_prune_outputs_match(lib):
    if not SCAFFOLD_PATH.exists():
        pytest.skip(f"Scaffold not found: {SCAFFOLD_PATH}")

    model = _make_model()

    with LibraryContext.with_library(lib):
        recipe = _load_scaffold_recipe()
        networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True)
        target = SVGTarget(
            name="MIT_T (soft vs hard)",
            path=RESOURCES_DIR / "designs/MIT_T.svg",
            transform_to_log_space=False,
            latent_x=(0.0, 0.6),
            latent_y=(0.0, 0.6),
        )
        dmanager = DesignManager(targets=[target], networks=networks[:1], enable_tu_masking=True)
        stack = dmanager.build_stack(model, unlock_ratios=True)
        model.shared_params = _extract_shared_params(stack)

        stack.ensure_tu_mapping()
        prunable = [t for t in stack.tu_id_to_idx.keys() if t not in (stack.no_masking_tu_ids or set())]
        if not prunable:
            pytest.skip("No prunable TUs found")
        tu_id = prunable[0]
        tu_idx = stack.tu_id_to_idx[tu_id]

        # Use stack.init() directly for forward pass (avoids vmap shape complexity)
        params = stack.init(jax.random.key(9))
        n_tus = len(stack.tu_id_to_idx)
        n_networks = len(dmanager.networks)
        log_alpha = jnp.full((n_networks, n_tus), 10.0)
        log_alpha = log_alpha.at[:, tu_idx].set(-10.0)
        params.at(TU_LOG_ALPHA_PATH, log_alpha, overwrite=True)

        n_inputs = stack.get_nb_inputs()
        n_z = int(np.asarray(params["global/number_of_random_variables"]).squeeze())
        X = jnp.ones((n_inputs,)) * 0.25
        Z = jnp.zeros((n_z,))
        key = jax.random.key(10)
        y_soft, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=None)

        tus_to_remove = identify_tus_to_prune(
            params,
            stack,
            dmanager,
            ratio_threshold=-1.0,
            use_soft_pruning=True,
            preserve_minimum=1,
        )

        dconf = DesignConfig(n_replicates=1, n_epochs=1)
        new_dmanager, new_stack, new_params_batched = hard_prune_and_rebuild(
            dmanager,
            dconf,
            model,
            stack,
            params,
            tus_to_remove,
            jax.random.key(11),
        )

        if new_stack.get_nb_inputs() != n_inputs:
            pytest.skip("Hard prune changed input count; output comparison not meaningful")

        # Strip (n_rep=1, n_tgt=1) vmap dims and ensure all leaves are JAX arrays
        new_params = tree_get(new_params_batched, (0, 0))
        for path, val in list(new_params.data.iter_leaves()):
            if isinstance(val, np.ndarray):
                new_params.at(path, jnp.asarray(val), overwrite=True)

        n_z_new = int(np.asarray(new_params["global/number_of_random_variables"]).squeeze())

        # Pruning may change random var count; adjust Z accordingly
        Z_new = jnp.zeros((n_z_new,)) if n_z_new != n_z else Z
        y_hard, _ = new_stack.apply(new_params, X, Z_new, key, tu_enabled_random_vars=None)

        y_soft_dep = _compress_dependent_outputs(stack, y_soft)
        y_hard_dep = _compress_dependent_outputs(new_stack, y_hard)

        if y_soft_dep.shape != y_hard_dep.shape:
            pytest.skip("Output shapes differ after prune")

        diff = np.mean(np.abs(y_soft_dep - y_hard_dep))
        denom = np.mean(np.abs(y_soft_dep)) + 1e-8
        rel = diff / denom
        assert rel < 0.1, f"Soft vs hard outputs differ too much (rel={rel:.3f})"


def test_prune_margin_affects_threshold(lib):
    """TUs with prob=0.45 should NOT be pruned with margin=0.1."""
    if not SCAFFOLD_PATH.exists():
        pytest.skip(f"Scaffold not found: {SCAFFOLD_PATH}")

    model = _make_model()

    with LibraryContext.with_library(lib):
        recipe = _load_scaffold_recipe()
        networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True)
        target = SVGTarget(
            name="MIT_T (margin test)",
            path=RESOURCES_DIR / "designs/MIT_T.svg",
            transform_to_log_space=False,
            latent_x=(0.0, 0.6),
            latent_y=(0.0, 0.6),
        )
        dmanager = DesignManager(targets=[target], networks=networks[:1], enable_tu_masking=True)
        stack = dmanager.build_stack(model, unlock_ratios=True)

        dconf = DesignConfig(n_replicates=1, n_epochs=1)
        dconf.tu_masking.mode = TUMaskingMode.DIRECT
        params_full = initialize_params(
            stack,
            n_replicates=1,
            n_targets=1,
            shared_params=model.shared_params,
            key=jax.random.key(100),
            strategy=build_strategy_from_config(dconf),
            n_tus=dmanager.n_tus,
            n_networks=len(dmanager.networks),
            no_masking_tu_ids=stack.no_masking_tu_ids,
            tu_id_to_idx=stack.tu_id_to_idx,
        )
        params = tree_get(params_full, (0, 0))

        stack.ensure_tu_mapping()
        tu_ids = [t for t in stack.tu_id_to_idx.keys() if t not in (stack.no_masking_tu_ids or set())]
        if not tu_ids:
            pytest.skip("No prunable TUs found")
        tu_id = tu_ids[0]
        tu_idx = stack.tu_id_to_idx[tu_id]

        log_alpha_for_prob_045 = np.log(0.45 / 0.55)
        log_alpha = np.array(params[TU_LOG_ALPHA_PATH], copy=True)
        log_alpha[...] = 10.0
        log_alpha[:, tu_idx] = log_alpha_for_prob_045
        params.at(TU_LOG_ALPHA_PATH, log_alpha, overwrite=True)

        prob = float(jax.nn.sigmoid(log_alpha_for_prob_045))
        assert 0.44 < prob < 0.46, f"Test setup: prob should be ~0.45, got {prob}"

        tus_with_margin_0 = identify_tus_to_prune(
            params, stack, dmanager,
            ratio_threshold=-1.0, use_soft_pruning=True, preserve_minimum=1,
            prune_margin=0.0,
        )
        assert tu_id in tus_with_margin_0[0], "With margin=0.0, prob<0.5 should be pruned"

        tus_with_margin_01 = identify_tus_to_prune(
            params, stack, dmanager,
            ratio_threshold=-1.0, use_soft_pruning=True, preserve_minimum=1,
            prune_margin=0.1,
        )
        assert tu_id not in tus_with_margin_01[0], "With margin=0.1, prob=0.45 should NOT be pruned"


def test_store_learned_ratio_inits_sets_init_not_locked(lib):
    """Verify _store_learned_ratio_inits sets ratio_range['init'] but keeps locked=False."""
    if not SCAFFOLD_PATH.exists():
        pytest.skip(f"Scaffold not found: {SCAFFOLD_PATH}")

    model = _make_model()

    with LibraryContext.with_library(lib):
        recipe = _load_scaffold_recipe()
        networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True)
        target = SVGTarget(
            name="MIT_T (ratio init)",
            path=RESOURCES_DIR / "designs/MIT_T.svg",
            transform_to_log_space=False,
            latent_x=(0.0, 0.6),
            latent_y=(0.0, 0.6),
        )
        dmanager = DesignManager(targets=[target], networks=networks[:1], enable_tu_masking=False)
        stack = dmanager.build_stack(model, unlock_ratios=True)
        params_full = initialize_params(
            stack,
            n_replicates=1,
            n_targets=1,
            shared_params=model.shared_params,
            key=jax.random.key(101),
            n_tus=0,
            n_networks=len(dmanager.networks),
            no_masking_tu_ids=stack.no_masking_tu_ids,
            tu_id_to_idx=stack.tu_id_to_idx,
        )
        params = tree_get(params_full, (0, 0))

        agg_info = _find_aggregation_member(stack)
        if agg_info is None:
            pytest.skip("No aggregation node found")

        agg_ns, agg_node_idx, source_id = agg_info
        ratio_path = f"{agg_ns}/ratios"

        test_ratio = 0.789
        ratios = np.array(params[ratio_path], copy=True)
        agg_layer = next(l for l in stack.layers if l.namespace == agg_ns)
        agg_node = agg_layer.nodes[agg_node_idx].get(stack)
        source_ids = [str(entry["source_id"]) for entry in get_slot_entries(agg_node.extra, require=False)]
        if source_id in source_ids:
            source_idx = source_ids.index(source_id)
            ratios[agg_node_idx, source_idx] = test_ratio
        params.at(ratio_path, ratios, overwrite=True)

        _store_learned_ratio_inits(params, stack)

        slot_entries = get_slot_entries(agg_node.extra, require=False)
        updated_entry = next(
            (entry for entry in slot_entries if str(entry.get("source_id")) == source_id),
            None,
        )
        assert updated_entry is not None, f"Source {source_id} not found"
        assert updated_entry.get("locked") is False, "Ratio should NOT be locked"
        assert "ratio_range" in updated_entry, "ratio_range should be set"
        assert updated_entry["ratio_range"].get("init") is not None, "init value should be set"
        assert abs(updated_entry["ratio_range"]["init"] - test_ratio) < 1e-6, (
            f"init should be {test_ratio}, got {updated_entry['ratio_range']['init']}"
        )


def test_store_learned_ratio_inits_uses_schema_slot_order(lib):
    """Regression test: ratio init mapping follows ratio_schema slot order."""
    if not SCAFFOLD_PATH.exists():
        pytest.skip(f"Scaffold not found: {SCAFFOLD_PATH}")

    model = _make_model()

    with LibraryContext.with_library(lib):
        recipe = _load_scaffold_recipe()
        networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True)
        target = SVGTarget(
            name="MIT_T (ratio sorted order)",
            path=RESOURCES_DIR / "designs/MIT_T.svg",
            transform_to_log_space=False,
            latent_x=(0.0, 0.6),
            latent_y=(0.0, 0.6),
        )
        dmanager = DesignManager(targets=[target], networks=networks[:1], enable_tu_masking=False)
        stack = dmanager.build_stack(model, unlock_ratios=True)
        params_full = initialize_params(
            stack,
            n_replicates=1,
            n_targets=1,
            shared_params=model.shared_params,
            key=jax.random.key(102),
            n_tus=0,
            n_networks=len(dmanager.networks),
            no_masking_tu_ids=stack.no_masking_tu_ids,
            tu_id_to_idx=stack.tu_id_to_idx,
        )
        params = tree_get(params_full, (0, 0))

        agg_info = _find_aggregation_member(stack)
        if agg_info is None:
            pytest.skip("No aggregation node found")
        agg_ns, agg_node_idx, _source_id = agg_info
        ratio_path = f"{agg_ns}/ratios"

        agg_layer = next(l for l in stack.layers if l.namespace == agg_ns)
        agg_node = agg_layer.nodes[agg_node_idx].get(stack)
        slot_entries = get_slot_entries(agg_node.extra, require=False)
        if len(slot_entries) < 2:
            pytest.skip("Aggregation node has too few ratio slots")

        ratios = np.array(params[ratio_path], copy=True)
        test_vals = np.linspace(0.11, 0.89, len(slot_entries))
        ratios[agg_node_idx, : len(slot_entries)] = test_vals
        params.at(ratio_path, ratios, overwrite=True)

        _store_learned_ratio_inits(params, stack)

        updated_entries = get_slot_entries(agg_node.extra, require=False)
        for idx, entry in enumerate(updated_entries):
            assert "ratio_range" in entry and entry["ratio_range"].get("init") is not None
            got = float(entry["ratio_range"]["init"])
            expected = float(test_vals[idx])
            assert abs(got - expected) < 1e-6, (
                f"Slot {idx} should get init {expected}, got {got}"
            )


class TestCommitPreservesRatioStates:
    """Tests for the fix: commit_structure must preserve ratio init values."""

    def test_commit_options_for_structure_preserves_ratio_states(self):
        """Verify CommitOptions.for_structure_only() sets preserve_ratio_states=True."""
        from biocomp.stack_commit import CommitOptions

        opts = CommitOptions.for_structure_only()
        assert opts.preserve_ratio_states is True, (
            "for_structure_only() must preserve ratio states to avoid random re-init"
        )

    def test_commit_options_for_final_does_not_preserve(self):
        """Verify CommitOptions.for_final() sets preserve_ratio_states=False."""
        from biocomp.stack_commit import CommitOptions

        opts = CommitOptions.for_final()
        assert opts.preserve_ratio_states is False, (
            "for_final() should not preserve ratio states (collapse to fixed values)"
        )

    def test_structure_commit_preserves_ratio_range_in_graph(self, lib):
        """Verify commit_structure keeps ratio_range dict with init in ratio_schema slots."""
        from biocomp.stack_commit import commit_structure

        if not SCAFFOLD_PATH.exists():
            pytest.skip(f"Scaffold not found: {SCAFFOLD_PATH}")

        model = _make_model()

        with LibraryContext.with_library(lib):
            recipe = _load_scaffold_recipe()
            networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True)
            target = SVGTarget(
                name="MIT_T (commit test)",
                path=RESOURCES_DIR / "designs/MIT_T.svg",
                transform_to_log_space=False,
                latent_x=(0.0, 0.6),
                latent_y=(0.0, 0.6),
            )
            dmanager = DesignManager(
                targets=[target], networks=networks[:1], enable_tu_masking=False
            )
            stack = dmanager.build_stack(model, unlock_ratios=True)
            params_full = initialize_params(
                stack,
                n_replicates=1,
                n_targets=1,
                shared_params=model.shared_params,
                key=jax.random.key(202),
                n_tus=0,
                n_networks=len(dmanager.networks),
                no_masking_tu_ids=stack.no_masking_tu_ids,
                tu_id_to_idx=stack.tu_id_to_idx,
            )
            params = tree_get(params_full, (0, 0))

            # Store learned ratios as init values
            _store_learned_ratio_inits(params, stack)

            # Find an aggregation node to check
            agg_info = _find_aggregation_member(stack)
            if agg_info is None:
                pytest.skip("No aggregation node found")

            agg_ns, agg_node_idx, source_id = agg_info
            agg_layer = next(l for l in stack.layers if l.namespace == agg_ns)
            agg_node = agg_layer.nodes[agg_node_idx].get(stack)

            # Get the init value before commit
            slot_entries_before = get_slot_entries(agg_node.extra, require=False)
            entry_before = next(
                (entry for entry in slot_entries_before if str(entry.get("source_id")) == source_id),
                {},
            )
            init_before = entry_before.get("ratio_range", {}).get("init")
            assert init_before is not None, "Init should be set before commit"

            # Commit with preserve_ratio_states=True (structure-only)
            committed_networks = commit_structure(stack, params, lock_ratios=False)
            assert len(committed_networks) > 0, "Should have committed networks"

            # Check the committed network's aggregation node
            committed_net = committed_networks[0]
            committed_agg = None
            for node in committed_net.compute_graph.nodes.values():
                if node.node_type == "aggregation":
                    entries = get_slot_entries(node.extra, require=False)
                    if any(str(entry.get("source_id")) == source_id for entry in entries):
                        committed_agg = node
                        break

            if committed_agg is None:
                pytest.skip("Source slot not found in committed network (may have been pruned)")

            slot_entries_after = get_slot_entries(committed_agg.extra, require=False)
            entry_after = next(
                (entry for entry in slot_entries_after if str(entry.get("source_id")) == source_id),
                {},
            )

            # THE KEY ASSERTION: ratio_range with init must survive commit
            assert "ratio_range" in entry_after, (
                "ratio_range dict must be preserved after commit_structure. "
                "This is the core fix for the hard-pruning loss regression bug."
            )
            init_after = entry_after["ratio_range"].get("init")
            assert init_after is not None, "init value must be preserved after commit"

    def test_structure_commit_with_lock_ratios_drops_range(self, lib):
        """Verify commit_structure with lock_ratios=True drops ratio_range."""
        from biocomp.stack_commit import commit_structure

        if not SCAFFOLD_PATH.exists():
            pytest.skip(f"Scaffold not found: {SCAFFOLD_PATH}")

        model = _make_model()

        with LibraryContext.with_library(lib):
            recipe = _load_scaffold_recipe()
            networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True)
            target = SVGTarget(
                name="MIT_T (lock test)",
                path=RESOURCES_DIR / "designs/MIT_T.svg",
                transform_to_log_space=False,
                latent_x=(0.0, 0.6),
                latent_y=(0.0, 0.6),
            )
            dmanager = DesignManager(
                targets=[target], networks=networks[:1], enable_tu_masking=False
            )
            stack = dmanager.build_stack(model, unlock_ratios=True)
            params_full = initialize_params(
                stack,
                n_replicates=1,
                n_targets=1,
                shared_params=model.shared_params,
                key=jax.random.key(203),
                n_tus=0,
                n_networks=len(dmanager.networks),
                no_masking_tu_ids=stack.no_masking_tu_ids,
                tu_id_to_idx=stack.tu_id_to_idx,
            )
            params = tree_get(params_full, (0, 0))

            _store_learned_ratio_inits(params, stack)

            # Commit with lock_ratios=True (should drop ratio_range)
            committed_networks = commit_structure(stack, params, lock_ratios=True)
            assert len(committed_networks) > 0

            # Check that ratio_range is dropped
            committed_net = committed_networks[0]
            for node in committed_net.compute_graph.nodes.values():
                if node.node_type == "aggregation":
                    entries = get_slot_entries(node.extra, require=False)
                    for entry in entries:
                        # ratio_range should be None when lock_ratios=True
                        assert entry.get("ratio_range") is None, (
                            "ratio_range should be None with lock_ratios=True, "
                            f"got {entry.get('ratio_range')} for source {entry.get('source_id')}"
                        )
