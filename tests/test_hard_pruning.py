"""Tests for hard-pruning functionality in design mode."""

import pytest
import numpy as np
from pathlib import Path
import jax
import jax.numpy as jnp
import dracon as dr

from biocomp.design import DesignConfig, DesignManager, initialize_params
from biocomp.design_pruning import (
    _merge_surviving_params,
    _expand_params_for_merge,
    _store_learned_ratio_inits,
    identify_tus_to_prune,
    hard_prune_and_rebuild,
)
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
from biocomp.tumasking_strategy import TUMaskingMode, build_strategy_from_config
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
            members = graph_node.extra.get("members", {})
            if isinstance(members, dict) and members:
                member_id = sorted(members.keys())[0]
                return namespace, node_idx, member_id
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
    if arr.ndim >= 2 and arr.shape[0] == 1 and arr.shape[1] == 1:
        arr = arr[0, 0]
    if input_idx is None:
        return arr[node_idx]
    return arr[node_idx, input_idx]


def _set_param_value(params: ParameterTree, path: str, node_idx: int, value, input_idx: int | None = None):
    arr = np.asarray(params[path])
    if arr.ndim >= 2 and arr.shape[0] == 1 and arr.shape[1] == 1:
        arr = arr.copy()
        if input_idx is None:
            arr[0, 0, node_idx] = value
        else:
            arr[0, 0, node_idx, input_idx] = value
    else:
        arr = arr.copy()
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
        if arr.ndim >= 2 and arr.shape[0] == 1 and arr.shape[1] == 1:
            arr = arr[0, 0]
        for node_idx, node in enumerate(layer.nodes):
            edges = sorted(node.get_incoming_edges(stack), key=lambda e: e.to_input_slot)
            for input_idx, edge in enumerate(edges):
                tu_ids = edge.extra.get("tu_id", []) if edge.extra else []
                if tu_id in tu_ids:
                    return arr[node_idx, input_idx]
    return None


def _find_ratio_in_new_stack(stack, params, member_id: str):
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
        if arr.ndim >= 2 and arr.shape[0] == 1 and arr.shape[1] == 1:
            arr = arr[0, 0]
        for node_idx, node in enumerate(layer.nodes):
            graph_node = node.get(stack)
            members = graph_node.extra.get("members", {})
            if isinstance(members, dict) and members:
                sorted_ids = sorted(members.keys())
                if member_id in sorted_ids:
                    idx = sorted_ids.index(member_id)
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
        if arr.ndim >= 2 and arr.shape[0] == 1 and arr.shape[1] == 1:
            arr = arr[0, 0]
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

        agg_ns, agg_node_idx, member_id = agg_info
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
        sorted_ids = sorted((agg_node.extra or {}).get("members", {}).keys())
        if member_id in sorted_ids:
            member_idx = sorted_ids.index(member_id)
            ratios[agg_node_idx, member_idx] = ratio_value
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

        new_rate = _find_value_in_new_stack(new_stack, new_params, tu_id, rate_name)
        new_ratio = _find_ratio_in_new_stack(new_stack, new_params, member_id)
        new_bias = _find_bias_in_new_stack(new_stack, new_params, protein)

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

        dconf = DesignConfig(n_replicates=1, n_epochs=1)
        dconf.tu_masking.mode = TUMaskingMode.DIRECT
        params_full = initialize_params(
            stack,
            n_replicates=1,
            n_targets=1,
            shared_params=model.shared_params,
            key=jax.random.key(9),
            strategy=build_strategy_from_config(dconf),
            n_tus=dmanager.n_tus,
            n_networks=len(dmanager.networks),
            no_masking_tu_ids=stack.no_masking_tu_ids,
            tu_id_to_idx=stack.tu_id_to_idx,
        )
        params = tree_get(params_full, (0, 0))

        stack.ensure_tu_mapping()
        prunable = [t for t in stack.tu_id_to_idx.keys() if t not in (stack.no_masking_tu_ids or set())]
        if not prunable:
            pytest.skip("No prunable TUs found")
        tu_id = prunable[0]
        tu_idx = stack.tu_id_to_idx[tu_id]

        log_alpha = np.array(params[TU_LOG_ALPHA_PATH], copy=True)
        log_alpha[...] = 10.0
        log_alpha[:, tu_idx] = -10.0
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

        new_dmanager, new_stack, new_params = hard_prune_and_rebuild(
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

        n_z_new = int(np.asarray(new_params["global/number_of_random_variables"]).squeeze())
        if n_z_new != n_z:
            pytest.skip("Hard prune changed random var count; output comparison not meaningful")

        y_hard, _ = new_stack.apply(new_params, X, Z, key, tu_enabled_random_vars=None)

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

        agg_ns, agg_node_idx, member_id = agg_info
        ratio_path = f"{agg_ns}/ratios"

        test_ratio = 0.789
        ratios = np.array(params[ratio_path], copy=True)
        agg_layer = next(l for l in stack.layers if l.namespace == agg_ns)
        agg_node = agg_layer.nodes[agg_node_idx].get(stack)
        sorted_ids = sorted((agg_node.extra or {}).get("members", {}).keys())
        if member_id in sorted_ids:
            member_idx = sorted_ids.index(member_id)
            ratios[agg_node_idx, member_idx] = test_ratio
        params.at(ratio_path, ratios, overwrite=True)

        _store_learned_ratio_inits(params, stack)

        members = agg_node.extra.get("members", {})
        m = members.get(member_id)
        assert m is not None, f"Member {member_id} not found"
        assert isinstance(m, dict), f"Member {member_id} is not a dict"
        assert m.get("locked") is False, "Ratio should NOT be locked"
        assert "ratio_range" in m, "ratio_range should be set"
        assert m["ratio_range"].get("init") is not None, "init value should be set"
        assert abs(m["ratio_range"]["init"] - test_ratio) < 1e-6, (
            f"init should be {test_ratio}, got {m['ratio_range']['init']}"
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
        """Verify commit_structure keeps ratio_range dict with init in aggregation members."""
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

            agg_ns, agg_node_idx, member_id = agg_info
            agg_layer = next(l for l in stack.layers if l.namespace == agg_ns)
            agg_node = agg_layer.nodes[agg_node_idx].get(stack)

            # Get the init value before commit
            members_before = agg_node.extra.get("members", {})
            m_before = members_before.get(member_id, {})
            init_before = m_before.get("ratio_range", {}).get("init")
            assert init_before is not None, "Init should be set before commit"

            # Commit with preserve_ratio_states=True (structure-only)
            committed_networks = commit_structure(stack, params, lock_ratios=False)
            assert len(committed_networks) > 0, "Should have committed networks"

            # Check the committed network's aggregation node
            committed_net = committed_networks[0]
            committed_agg = None
            for node in committed_net.compute_graph.nodes.values():
                if node.node_type == "aggregation":
                    members = (node.extra or {}).get("members", {})
                    if member_id in members:
                        committed_agg = node
                        break

            if committed_agg is None:
                pytest.skip("Member not found in committed network (may have been pruned)")

            members_after = committed_agg.extra.get("members", {})
            m_after = members_after.get(member_id, {})

            # THE KEY ASSERTION: ratio_range with init must survive commit
            assert "ratio_range" in m_after, (
                "ratio_range dict must be preserved after commit_structure. "
                "This is the core fix for the hard-pruning loss regression bug."
            )
            init_after = m_after["ratio_range"].get("init")
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
                    members = (node.extra or {}).get("members", {})
                    for mid, m in members.items():
                        if isinstance(m, dict):
                            # ratio_range should be None when lock_ratios=True
                            assert m.get("ratio_range") is None, (
                                f"ratio_range should be None with lock_ratios=True, "
                                f"got {m.get('ratio_range')} for member {mid}"
                            )
