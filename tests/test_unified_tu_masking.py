# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Tests for unified TU masking (get_tu_masks) supporting both binary and Hard Concrete modes."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import dracon as dr

from biocomp.tumasking import (
    TU_BINARY_MASK_PATH,
    TU_LOG_ALPHA_PATH,
    TU_ALWAYS_ENABLED,
    get_tu_masks,
    _apply_binary_mask_single,
    _apply_binary_mask_multi,
    _apply_binary_masks,
    build_tu_id_mapping,
    set_binary_tu_mask,
)
from biocomp.network import recipe_to_networks
from biocomp.compute import ComputeStack
from biocomp.recipe import Recipe
from biocomp.config import SIMPLE_NODES_COMPUTE_CONFIG
from biocomp.library import LibraryContext, load_lib
from biocomp.ratio_schema import get_slot_entries
import biocomp.biorules as br
from pathlib import Path

RESOURCES_DIR = Path(__file__).parent / "resources"
SCAFFOLD_PATH = RESOURCES_DIR / "design/architectures/two_and_one.yaml"


@pytest.fixture(scope="module")
def lib():
    return load_lib()


class MockParams(dict):
    def __contains__(self, key):
        return key in self.keys()


# ===================== Unit Tests: Binary Mask Primitives =====================


def test_apply_binary_mask_single_basic():
    mask = jnp.array([1.0, 0.0, 1.0, 0.0])
    assert float(_apply_binary_mask_single(jnp.array(0), mask)) == 1.0
    assert float(_apply_binary_mask_single(jnp.array(1), mask)) == 0.0
    assert float(_apply_binary_mask_single(jnp.array(2), mask)) == 1.0
    assert float(_apply_binary_mask_single(jnp.array(3), mask)) == 0.0


def test_apply_binary_mask_single_always_enabled():
    mask = jnp.array([0.0, 0.0, 0.0])
    assert float(_apply_binary_mask_single(jnp.array(TU_ALWAYS_ENABLED), mask)) == 1.0


def test_apply_binary_mask_multi_or_reduction():
    mask = jnp.array([1.0, 0.0, 1.0, 0.0])
    assert float(_apply_binary_mask_multi(jnp.array([0, 1]), mask)) == 1.0  # 1 OR 0 = 1
    assert float(_apply_binary_mask_multi(jnp.array([1, 3]), mask)) == 0.0  # 0 OR 0 = 0
    assert float(_apply_binary_mask_multi(jnp.array([2, 3]), mask)) == 1.0  # 1 OR 0 = 1


def test_apply_binary_mask_multi_all_padding():
    mask = jnp.array([0.0, 0.0, 0.0])
    assert (
        float(_apply_binary_mask_multi(jnp.array([TU_ALWAYS_ENABLED, TU_ALWAYS_ENABLED]), mask))
        == 1.0
    )


def test_apply_binary_mask_multi_partial_padding():
    mask = jnp.array([1.0, 0.0, 1.0])
    assert float(_apply_binary_mask_multi(jnp.array([0, TU_ALWAYS_ENABLED]), mask)) == 1.0
    assert float(_apply_binary_mask_multi(jnp.array([1, TU_ALWAYS_ENABLED]), mask)) == 0.0


def test_apply_binary_masks_single_tu():
    mask = jnp.array([1.0, 0.0, 1.0, 0.0])
    tu_indices = jnp.array([0, 1, 2, 3])
    result = _apply_binary_masks(tu_indices, mask, is_multi_tu=False)
    np.testing.assert_array_equal(result, jnp.array([1.0, 0.0, 1.0, 0.0]))


def test_apply_binary_masks_multi_tu():
    mask = jnp.array([1.0, 0.0, 1.0, 0.0])
    tu_indices = jnp.array([[0, 1], [1, 3], [2, 3]])
    result = _apply_binary_masks(tu_indices, mask, is_multi_tu=True)
    np.testing.assert_array_equal(result, jnp.array([1.0, 0.0, 1.0]))


def test_apply_binary_masks_shape_validation():
    mask_1d = jnp.array([1.0, 0.0])
    mask_2d = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    tu_1d = jnp.array([0, 1])
    tu_2d = jnp.array([[0, 1], [1, 0]])

    _apply_binary_masks(tu_1d, mask_1d, is_multi_tu=False)
    _apply_binary_masks(tu_2d, mask_1d, is_multi_tu=True)

    with pytest.raises(AssertionError, match="binary_mask must be 1D"):
        _apply_binary_masks(tu_1d, mask_2d, is_multi_tu=False)

    with pytest.raises(AssertionError, match="requires 2D tu_indices"):
        _apply_binary_masks(tu_1d, mask_1d, is_multi_tu=True)

    with pytest.raises(AssertionError, match="requires 1D tu_indices"):
        _apply_binary_masks(tu_2d, mask_1d, is_multi_tu=False)


# ===================== Unit Tests: get_tu_masks Unified Function =====================


def test_get_tu_masks_binary_mode_single():
    params = MockParams({TU_BINARY_MASK_PATH: jnp.array([[1.0, 0.0, 1.0, 0.0]])})
    tu_indices = jnp.array([0, 1, 2, 3])
    result = get_tu_masks(params, tu_indices, 0, is_multi_tu=False)
    np.testing.assert_array_equal(result, jnp.array([1.0, 0.0, 1.0, 0.0]))


def test_get_tu_masks_binary_mode_multi():
    params = MockParams({TU_BINARY_MASK_PATH: jnp.array([[1.0, 0.0, 1.0, 0.0]])})
    tu_indices = jnp.array([[0, 1], [1, 3], [2, 0]])
    result = get_tu_masks(params, tu_indices, 0, is_multi_tu=True)
    np.testing.assert_array_equal(result, jnp.array([1.0, 0.0, 1.0]))


def test_get_tu_masks_binary_mode_2d_mask():
    mask_2d = jnp.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ]
    )
    params = MockParams({TU_BINARY_MASK_PATH: mask_2d})
    tu_indices = jnp.array([0, 1, 2, 3])

    result_net0 = get_tu_masks(params, tu_indices, 0, is_multi_tu=False)
    np.testing.assert_array_equal(result_net0, jnp.array([1.0, 0.0, 1.0, 0.0]))

    result_net1 = get_tu_masks(params, tu_indices, 1, is_multi_tu=False)
    np.testing.assert_array_equal(result_net1, jnp.array([0.0, 1.0, 0.0, 1.0]))


def test_get_tu_masks_binary_mode_requires_network_id_for_2d():
    mask_2d = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    params = MockParams({TU_BINARY_MASK_PATH: mask_2d})
    tu_indices = jnp.array([0, 1])

    result_net0 = get_tu_masks(params, tu_indices, 0, is_multi_tu=False)
    np.testing.assert_array_equal(result_net0, jnp.array([1.0, 0.0]))

    result_net1 = get_tu_masks(params, tu_indices, 1, is_multi_tu=False)
    np.testing.assert_array_equal(result_net1, jnp.array([0.0, 1.0]))


def test_get_tu_masks_log_alpha_mode():
    params = MockParams({TU_LOG_ALPHA_PATH: jnp.array([[5.0, -5.0, 5.0, -5.0]])})
    tu_indices = jnp.array([0, 1, 2, 3])
    result = get_tu_masks(params, tu_indices, 0, is_multi_tu=False)
    assert result[0] > 0.5
    assert result[1] < 0.5
    assert result[2] > 0.5
    assert result[3] < 0.5


def test_get_tu_masks_log_alpha_mode_2d():
    log_alpha_2d = jnp.array(
        [
            [5.0, -5.0],
            [-5.0, 5.0],
        ]
    )
    params = MockParams({TU_LOG_ALPHA_PATH: log_alpha_2d})
    tu_indices = jnp.array([0, 1])

    result_net0 = get_tu_masks(params, tu_indices, 0, is_multi_tu=False)
    assert result_net0[0] > 0.5
    assert result_net0[1] < 0.5

    result_net1 = get_tu_masks(params, tu_indices, 1, is_multi_tu=False)
    assert result_net1[0] < 0.5
    assert result_net1[1] > 0.5


def test_get_tu_masks_disabled_mode():
    params = MockParams({})
    tu_indices = jnp.array([0, 1, 2, 3])
    result = get_tu_masks(params, tu_indices, 0, is_multi_tu=False)
    np.testing.assert_array_equal(result, jnp.ones(4))


def test_get_tu_masks_binary_priority_over_log_alpha():
    params = MockParams(
        {
            TU_BINARY_MASK_PATH: jnp.array([[1.0, 0.0]]),
            TU_LOG_ALPHA_PATH: jnp.array([[-5.0, 5.0]]),
        }
    )
    tu_indices = jnp.array([0, 1])
    result = get_tu_masks(params, tu_indices, 0, is_multi_tu=False)
    np.testing.assert_array_equal(result, jnp.array([1.0, 0.0]))


# ===================== Integration Tests: Binary Mask Mode =====================
# These tests use scope="function" fixtures to avoid JIT caching issues


@pytest.fixture
def binary_mask_stack(lib):
    """Stack pre-initialized for binary mask mode (fresh per test)."""
    with LibraryContext.with_library(lib):
        data = dr.load(SCAFFOLD_PATH, context={"Recipe": Recipe})
        recipes = data["recipes"] if "recipes" in data else data.recipes
        networks = recipe_to_networks(recipes[0], br.ALL_RULES, invert=True)
        tu_ids, tu_id_to_idx = build_tu_id_mapping(networks)
        stack = ComputeStack(networks)
        config = SIMPLE_NODES_COMPUTE_CONFIG.model_copy(deep=True)
        stack.build(config, enable_tu_masking=True)

        key = jax.random.key(42)
        params = stack.init(key)
        n_tus = len(tu_ids)
        n_networks = len(networks)
        params.at(TU_BINARY_MASK_PATH, jnp.ones((n_networks, n_tus)))

        return stack, tu_ids, tu_id_to_idx, params


def test_binary_mask_matches_tu_ids(lib, binary_mask_stack):
    """Verify binary mask indices correspond to correct TU IDs."""
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx, params = binary_mask_stack

        n_tus = len(tu_ids)
        n_networks = len(stack.networks)

        binary_mask = jnp.zeros(n_tus)
        binary_mask = binary_mask.at[0].set(1.0)
        params.at(
            TU_BINARY_MASK_PATH, jnp.broadcast_to(binary_mask, (n_networks, n_tus)), overwrite=True
        )

        first_tu_id = tu_ids[0]

        for net in stack.networks:
            for edge in net.compute_graph.edges.values():
                edge_tu_ids = edge.extra.get("tu_id", []) if edge.extra else []
                has_only_first_tu = edge_tu_ids == [first_tu_id]

                if has_only_first_tu:
                    assert tu_id_to_idx[first_tu_id] == 0


def test_specific_tu_disable_affects_correct_edges(lib, binary_mask_stack):
    """Disabling all TUs vs enabling all TUs should produce different outputs."""
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx, params = binary_mask_stack

        n_tus = len(tu_ids)
        n_networks = len(stack.networks)
        n_inputs = stack.get_nb_inputs()
        n_z = int(params["global/number_of_random_variables"])

        if n_tus == 0:
            pytest.skip("No TUs in network")

        X = jnp.ones((n_inputs,)) * 0.5
        Z = jnp.ones((n_z,)) * 0.5
        key = jax.random.key(42)

        all_enabled = jnp.ones((n_networks, n_tus))
        params.at(TU_BINARY_MASK_PATH, all_enabled, overwrite=True)
        y_all, _ = stack.apply(params, X, Z, key)

        all_disabled = jnp.zeros((n_networks, n_tus))
        params.at(TU_BINARY_MASK_PATH, all_disabled, overwrite=True)
        y_disabled, _ = stack.apply(params, X, Z, key)

        assert not jnp.allclose(y_all, y_disabled, atol=1e-3), (
            f"Disabling ALL TUs should change output.\n"
            f"All enabled: {y_all}\n"
            f"All disabled: {y_disabled}"
        )


def test_all_tus_disabled_near_zero_output(lib, binary_mask_stack):
    """With all TUs disabled via binary mask, output should be near zero."""
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx, params = binary_mask_stack

        n_tus = len(tu_ids)
        n_networks = len(stack.networks)
        n_inputs = stack.get_nb_inputs()
        n_z = int(params["global/number_of_random_variables"])

        X = jnp.ones((n_inputs,)) * 0.5
        Z = jnp.ones((n_z,)) * 0.5
        key = jax.random.key(42)

        all_disabled = jnp.zeros((n_networks, n_tus))
        params.at(TU_BINARY_MASK_PATH, all_disabled, overwrite=True)
        y, _ = stack.apply(params, X, Z, key)

        assert jnp.sum(jnp.abs(y)) < 0.1, f"All TUs disabled should give near-zero output, got {y}"


def test_per_network_binary_mask_independence(lib, binary_mask_stack):
    """Binary masks should be applied independently per network."""
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx, params = binary_mask_stack
        n_networks = len(stack.networks)

        if n_networks < 2:
            pytest.skip("Need at least 2 networks")

        n_tus = len(tu_ids)
        n_inputs = stack.get_nb_inputs()
        n_z = int(params["global/number_of_random_variables"])

        X = jnp.ones((n_inputs,)) * 0.5
        Z = jnp.ones((n_z,)) * 0.5
        key = jax.random.key(42)

        mask_2d = jnp.ones((n_networks, n_tus))
        mask_2d = mask_2d.at[0, :].set(0.0)
        params.at(TU_BINARY_MASK_PATH, mask_2d, overwrite=True)

        y, aux = stack.apply(params, X, Z, key)

        net0_outputs = [float(y[i]) for i in stack.get_output_node_ids_for_network(0)]
        net1_outputs = [float(y[i]) for i in stack.get_output_node_ids_for_network(1)]

        net0_mag = sum(abs(x) for x in net0_outputs)
        net1_mag = sum(abs(x) for x in net1_outputs)

        assert net0_mag < net1_mag, (
            f"Network 0 (all disabled) should have lower magnitude than network 1.\n"
            f"Net 0: {net0_mag}, Net 1: {net1_mag}"
        )


# ===================== Commit Tests: Recipe Pruning =====================


def test_binary_mask_commit_removes_disabled_tus(lib, binary_mask_stack):
    """Commit with binary mask should prune disabled TUs from graph."""
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx, params = binary_mask_stack
        key = jax.random.key(42)

        n_tus = len(tu_ids)
        n_networks = len(stack.networks)

        if n_tus == 0:
            pytest.skip("No TUs in network")

        all_enabled = jnp.ones((n_networks, n_tus))
        params.at(TU_BINARY_MASK_PATH, all_enabled, overwrite=True)
        committed_enabled = stack.commit(params)

        params_disabled = stack.init(key)
        first_disabled = jnp.ones((n_networks, n_tus))
        first_disabled = first_disabled.at[:, 0].set(0.0)
        params_disabled.at(TU_BINARY_MASK_PATH, first_disabled)
        committed_disabled = stack.commit(params_disabled)

        def count_members(networks):
            total = 0
            for net in networks:
                for node in net.compute_graph.nodes.values():
                    if "aggregation" in node.node_type.lower():
                        total += len(get_slot_entries(node.extra, require=False))
            return total

        members_enabled = count_members(committed_enabled)
        members_disabled = count_members(committed_disabled)

        assert members_disabled < members_enabled, (
            f"Disabling TU should reduce member count.\n"
            f"Enabled: {members_enabled}, Disabled: {members_disabled}"
        )


def test_commit_produces_valid_recipes_per_network(lib, binary_mask_stack):
    """Commit should produce valid recipes, with correct network modified."""
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx, params = binary_mask_stack

        n_tus = len(tu_ids)
        n_networks = len(stack.networks)

        if n_tus == 0 or n_networks < 2:
            pytest.skip("Need TUs and at least 2 networks")

        mask_net0_disabled = jnp.ones((n_networks, n_tus))
        mask_net0_disabled = mask_net0_disabled.at[0, :].set(0.0)
        params.at(TU_BINARY_MASK_PATH, mask_net0_disabled, overwrite=True)

        committed = stack.commit(params)

        def count_tu_edges(net):
            count = 0
            for edge in net.compute_graph.edges.values():
                if edge.extra and edge.extra.get("tu_id"):
                    count += 1
            return count

        net0_tu_edges = count_tu_edges(committed[0])
        net1_tu_edges = count_tu_edges(committed[1])

        assert net0_tu_edges < net1_tu_edges or net0_tu_edges == 0, (
            f"Network 0 (all TUs disabled) should have fewer TU edges.\n"
            f"Net 0 TU edges: {net0_tu_edges}, Net 1 TU edges: {net1_tu_edges}"
        )


def test_commit_recipe_has_disabled_tus_removed(lib, binary_mask_stack):
    """Disabled TUs should be removed from the recipe when committed."""
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx, params = binary_mask_stack

        n_tus = len(tu_ids)
        n_networks = len(stack.networks)

        if n_tus == 0:
            pytest.skip("No TUs in network")

        disabled_tu_name = tu_ids[0]
        mask = jnp.ones((n_networks, n_tus))
        mask = mask.at[:, 0].set(0.0)
        params.at(TU_BINARY_MASK_PATH, mask, overwrite=True)

        committed = stack.commit(params)

        for net in committed:
            for edge in net.compute_graph.edges.values():
                edge_tu_ids = edge.extra.get("tu_id", []) if edge.extra else []
                assert disabled_tu_name not in edge_tu_ids, (
                    f"Disabled TU '{disabled_tu_name}' should not appear in committed network edges. "
                    f"Found in edge {edge}"
                )


def test_set_binary_tu_mask_helper(lib, binary_mask_stack):
    """Test set_binary_tu_mask helper function."""
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx, params = binary_mask_stack

        n_networks = len(stack.networks)
        if len(tu_ids) < 2:
            pytest.skip("Need at least 2 TUs")

        set_binary_tu_mask(params, tu_ids, tu_id_to_idx, n_networks, disabled_tus={0: {tu_ids[0]}})

        mask = params[TU_BINARY_MASK_PATH]
        assert mask[0, 0] == 0.0, "First TU of network 0 should be disabled"
        assert mask[0, 1] == 1.0, "Second TU of network 0 should be enabled"
        if n_networks > 1:
            assert mask[1, 0] == 1.0, "First TU of network 1 should still be enabled"


def test_set_binary_tu_mask_enabled_mode(lib, binary_mask_stack):
    """Test set_binary_tu_mask with enabled_tus (whitelist mode)."""
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx, params = binary_mask_stack

        n_networks = len(stack.networks)
        if len(tu_ids) < 2:
            pytest.skip("Need at least 2 TUs")

        set_binary_tu_mask(params, tu_ids, tu_id_to_idx, n_networks, enabled_tus={0: {tu_ids[0]}})

        mask = params[TU_BINARY_MASK_PATH]
        assert mask[0, 0] == 1.0, "First TU should be enabled in network 0"
        assert mask[0, 1] == 0.0, "Second TU should be disabled in network 0"
        if n_networks > 1:
            np.testing.assert_array_equal(mask[1, :], 0.0)


def test_commit_per_network_independence(lib):
    """Verify that commit modifies correct network's recipe, not others."""
    with LibraryContext.with_library(lib):
        data = dr.load(SCAFFOLD_PATH, context={"Recipe": Recipe})
        recipes = data["recipes"] if "recipes" in data else data.recipes
        networks = recipe_to_networks(recipes[0], br.ALL_RULES, invert=True)

        if len(networks) < 2:
            pytest.skip("Need at least 2 networks")

        tu_ids, tu_id_to_idx = build_tu_id_mapping(networks)
        if len(tu_ids) == 0:
            pytest.skip("No TUs")

        n_networks = len(networks)
        n_tus = len(tu_ids)

        stack = ComputeStack(networks)
        config = SIMPLE_NODES_COMPUTE_CONFIG.model_copy(deep=True)
        stack.build(config, enable_tu_masking=True)

        key = jax.random.key(42)
        params = stack.init(key)

        mask_only_net0_disabled = jnp.ones((n_networks, n_tus))
        mask_only_net0_disabled = mask_only_net0_disabled.at[0, 0].set(0.0)
        params.at(TU_BINARY_MASK_PATH, mask_only_net0_disabled)

        committed = stack.commit(params)

        disabled_tu = tu_ids[0]

        def has_tu_in_edges(net, tu_name):
            for edge in net.compute_graph.edges.values():
                edge_tu_ids = edge.extra.get("tu_id", []) if edge.extra else []
                if tu_name in edge_tu_ids:
                    return True
            return False

        net0_has_disabled_tu = has_tu_in_edges(committed[0], disabled_tu)
        net1_has_disabled_tu = has_tu_in_edges(committed[1], disabled_tu)

        assert not net0_has_disabled_tu, (
            f"Network 0 should NOT have disabled TU '{disabled_tu}' in edges after commit"
        )
        assert net1_has_disabled_tu, (
            f"Network 1 SHOULD still have TU '{disabled_tu}' in edges (it wasn't disabled for net 1)"
        )


# ===================== Hard Concrete Mode Tests (Separate Fixture) =====================


@pytest.fixture
def hard_concrete_stack(lib):
    """Stack pre-initialized for Hard Concrete mode (fresh per test)."""
    with LibraryContext.with_library(lib):
        data = dr.load(SCAFFOLD_PATH, context={"Recipe": Recipe})
        recipes = data["recipes"] if "recipes" in data else data.recipes
        networks = recipe_to_networks(recipes[0], br.ALL_RULES, invert=True)
        tu_ids, tu_id_to_idx = build_tu_id_mapping(networks)
        stack = ComputeStack(networks)
        config = SIMPLE_NODES_COMPUTE_CONFIG.model_copy(deep=True)
        stack.build(config, enable_tu_masking=True)

        key = jax.random.key(42)
        params = stack.init(key)
        n_tus = len(tu_ids)
        n_networks = len(networks)
        params.at(TU_LOG_ALPHA_PATH, jnp.full((n_networks, n_tus), 5.0))

        return stack, tu_ids, tu_id_to_idx, params


def test_hard_concrete_backward_compatible(lib, hard_concrete_stack):
    """Existing Hard Concrete mode should work unchanged."""
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx, params = hard_concrete_stack

        n_tus = len(tu_ids)
        n_networks = len(stack.networks)
        n_inputs = stack.get_nb_inputs()
        n_z = int(params["global/number_of_random_variables"])

        X = jnp.ones((n_inputs,)) * 0.5
        Z = jnp.ones((n_z,)) * 0.5
        key = jax.random.key(42)

        tu_uniform = jnp.full((n_networks, n_tus), 0.5)
        y, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=tu_uniform)
        assert not jnp.allclose(y, 0.0, atol=1e-3)


# ===================== Default Mode Tests (No Masking) =====================


@pytest.fixture
def default_mode_stack(lib):
    """Stack without any masking initialized (fresh per test)."""
    with LibraryContext.with_library(lib):
        data = dr.load(SCAFFOLD_PATH, context={"Recipe": Recipe})
        recipes = data["recipes"] if "recipes" in data else data.recipes
        networks = recipe_to_networks(recipes[0], br.ALL_RULES, invert=True)
        tu_ids, tu_id_to_idx = build_tu_id_mapping(networks)
        stack = ComputeStack(networks)
        config = SIMPLE_NODES_COMPUTE_CONFIG.model_copy(deep=True)
        stack.build(config, enable_tu_masking=True)

        key = jax.random.key(42)
        params = stack.init(key)

        return stack, tu_ids, tu_id_to_idx, params


def test_no_masking_backward_compatible(lib, default_mode_stack):
    """When neither mask path exists, all TUs should be enabled."""
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx, params = default_mode_stack

        n_inputs = stack.get_nb_inputs()
        n_z = int(params["global/number_of_random_variables"])

        X = jnp.ones((n_inputs,)) * 0.5
        Z = jnp.ones((n_z,)) * 0.5
        key = jax.random.key(42)

        y, _ = stack.apply(params, X, Z, key)
        assert not jnp.allclose(y, 0.0, atol=1e-3)


# ===================== Edge Cases =====================


def test_empty_tu_indices():
    params = MockParams({TU_BINARY_MASK_PATH: jnp.array([[1.0, 0.0]])})
    tu_indices = jnp.array([], dtype=jnp.int32)
    result = get_tu_masks(params, tu_indices, 0, is_multi_tu=False)
    assert result.shape == (0,)


def test_all_always_enabled_indices():
    params = MockParams({TU_BINARY_MASK_PATH: jnp.array([[0.0, 0.0, 0.0]])})
    tu_indices = jnp.array([TU_ALWAYS_ENABLED, TU_ALWAYS_ENABLED, TU_ALWAYS_ENABLED])
    result = get_tu_masks(params, tu_indices, 0, is_multi_tu=False)
    np.testing.assert_array_equal(result, jnp.ones(3))


def test_mixed_always_enabled_and_disabled():
    params = MockParams({TU_BINARY_MASK_PATH: jnp.array([[0.0, 1.0, 0.0]])})
    tu_indices = jnp.array([TU_ALWAYS_ENABLED, 0, 1, 2])
    result = get_tu_masks(params, tu_indices, 0, is_multi_tu=False)
    np.testing.assert_array_equal(result, jnp.array([1.0, 0.0, 1.0, 0.0]))


def test_binary_mask_jit_compatible():
    @jax.jit
    def jitted_get_masks(binary_mask, tu_indices):
        params = {TU_BINARY_MASK_PATH: binary_mask}
        return get_tu_masks(params, tu_indices, 0, is_multi_tu=False)

    mask = jnp.array([[1.0, 0.0, 1.0]])
    indices = jnp.array([0, 1, 2])
    result = jitted_get_masks(mask, indices)
    np.testing.assert_array_equal(result, mask[0])


def test_binary_mask_vmap_over_networks():
    mask_2d = jnp.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    tu_indices = jnp.array([0, 1, 2])

    def get_masks_for_network(network_id):
        params = {TU_BINARY_MASK_PATH: mask_2d}
        return get_tu_masks(params, tu_indices, network_id, is_multi_tu=False)

    result = jax.vmap(get_masks_for_network)(jnp.arange(3))
    np.testing.assert_array_equal(result, mask_2d)


def test_binary_mask_all_ones():
    params = MockParams({TU_BINARY_MASK_PATH: jnp.ones((1, 10))})
    tu_indices = jnp.arange(10)
    result = get_tu_masks(params, tu_indices, 0, is_multi_tu=False)
    np.testing.assert_array_equal(result, jnp.ones(10))


def test_binary_mask_all_zeros():
    params = MockParams({TU_BINARY_MASK_PATH: jnp.zeros((1, 10))})
    tu_indices = jnp.arange(10)
    result = get_tu_masks(params, tu_indices, 0, is_multi_tu=False)
    np.testing.assert_array_equal(result, jnp.zeros(10))


def test_binary_mask_gradients_stop():
    def loss_fn(mask_2d):
        params = {TU_BINARY_MASK_PATH: mask_2d}
        tu_indices = jnp.array([0, 1, 2])
        masks = get_tu_masks(params, tu_indices, 0, is_multi_tu=False)
        return jnp.sum(masks)

    mask = jnp.array([[1.0, 0.0, 1.0]])
    grads = jax.grad(loss_fn)(mask)
    np.testing.assert_array_equal(grads, jnp.ones((1, 3)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
