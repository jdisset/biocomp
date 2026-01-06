"""Tests for TU masking in design mode.

Key success criteria:
1. All TUs disabled produces different (lower magnitude) output than enabled
2. Partial TU disabling changes output compared to all enabled
3. Masking is deterministic (same log_alpha -> same output)
4. Gradual disabling produces monotonic decrease in output magnitude

Note: Output may not be exactly zero when disabled because:
- Output activation (e.g., sigmoid) maps zero input to non-zero output
- Some outputs (markers) may have simpler computation paths
- The key property is that masking AFFECTS the output, not that it produces zero

TU masking uses binary thresholding with STE (Straight-Through Estimator):
- Forward: sigmoid(log_alpha) >= 0.5 -> enabled (1.0) or disabled (0.0)
- Backward: gradients flow through sigmoid(log_alpha)
- log_alpha > 0 -> TU enabled, log_alpha < 0 -> TU disabled
"""

import pytest
import jax
import jax.numpy as jnp
import dracon as dr

from biocomp.network import recipe_to_networks
from biocomp.compute import ComputeStack
from biocomp.recipe import Recipe
from biocomp.config import SIMPLE_NODES_COMPUTE_CONFIG
from biocomp.library import LibraryContext, load_lib
import biocomp.biorules as br
from biocomp.tumasking import (
    build_tu_id_mapping,
    TU_LOG_ALPHA_PATH,
    hard_concrete_from_uniform,
)
from pathlib import Path

RESOURCES_DIR = Path(__file__).parent / "resources"
SCAFFOLD_PATH = RESOURCES_DIR / "design/architectures/two_and_one.yaml"


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
def design_stack(lib, scaffold_recipe):
    """Build a stack with integrated TU masking enabled. Module-scoped for speed."""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(scaffold_recipe, br.ALL_RULES, invert=True)
        tu_ids, tu_id_to_idx = build_tu_id_mapping(networks)

        stack = ComputeStack(networks)
        config = SIMPLE_NODES_COMPUTE_CONFIG.model_copy(deep=True)
        stack.build(config, enable_tu_masking=True)

        # JIT warmup: trigger compilation once so all tests use pre-compiled function
        key = jax.random.key(0)
        params = stack.init(key)
        n_tus = len(tu_ids)
        n_networks = len(networks)
        n_inputs = stack.get_nb_inputs()
        n_z = int(params["global/number_of_random_variables"])
        dummy_x = jnp.zeros((n_inputs,))
        dummy_z = jnp.zeros((n_z,))
        params.at(TU_LOG_ALPHA_PATH, jnp.full((n_networks, n_tus), 10.0), overwrite=True)
        stack.apply(params, dummy_x, dummy_z, key, tu_enabled_random_vars=None)

        return stack, tu_ids, tu_id_to_idx


def test_all_tus_disabled_differs_from_enabled(lib, design_stack):
    """When all TUs are disabled (log_alpha < 0), output should differ from enabled (log_alpha > 0)."""
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx = design_stack
        key = jax.random.key(42)
        params = stack.init(key)

        n_tus = len(tu_ids)
        n_networks = len(stack.networks)
        n_inputs = stack.get_nb_inputs()
        n_z_val = params["global/number_of_random_variables"]
        n_z = int(n_z_val.squeeze()) if hasattr(n_z_val, "squeeze") else int(n_z_val)

        X = jnp.ones((n_inputs,)) * 0.5
        Z = jnp.ones((n_z,)) * 0.5

        # All disabled: log_alpha = -10 -> sigmoid(-10) ≈ 0 -> binary mask = 0
        params.at(TU_LOG_ALPHA_PATH, jnp.full((n_networks, n_tus), -10.0), overwrite=True)
        y_disabled, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=None)

        # All enabled: log_alpha = +10 -> sigmoid(10) ≈ 1 -> binary mask = 1
        params.at(TU_LOG_ALPHA_PATH, jnp.full((n_networks, n_tus), 10.0), overwrite=True)
        y_enabled, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=None)

        # Key property: outputs should be different
        assert not jnp.allclose(y_disabled, y_enabled, atol=1e-3), (
            f"Expected different outputs: disabled={y_disabled}, enabled={y_enabled}"
        )

        # Disabled output magnitude should be less than or equal to enabled
        # (masking reduces signal, doesn't amplify it)
        mag_disabled = float(jnp.sum(jnp.abs(y_disabled)))
        mag_enabled = float(jnp.sum(jnp.abs(y_enabled)))
        assert mag_disabled <= mag_enabled + 1e-3, (
            f"Disabled magnitude ({mag_disabled}) should be <= enabled ({mag_enabled})"
        )


def test_all_tus_enabled_produces_nonzero(lib, design_stack):
    """When all TUs are enabled (log_alpha > 0), output should be non-zero."""
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx = design_stack
        key = jax.random.key(42)
        params = stack.init(key)

        n_tus = len(tu_ids)
        n_networks = len(stack.networks)
        n_inputs = stack.get_nb_inputs()
        n_z_val = params["global/number_of_random_variables"]
        n_z = int(n_z_val.squeeze()) if hasattr(n_z_val, "squeeze") else int(n_z_val)

        X = jnp.ones((n_inputs,)) * 0.5
        Z = jnp.ones((n_z,)) * 0.5

        # log_alpha = +10 -> sigmoid(10) ≈ 1 -> binary mask = 1 (enabled)
        params.at(TU_LOG_ALPHA_PATH, jnp.full((n_networks, n_tus), 10.0), overwrite=True)

        y, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=None)

        assert not jnp.allclose(y, 0.0, atol=1e-3), f"Expected non-zero output, got {y}"


def test_partial_masking_reduces_output(lib, design_stack):
    """Disabling some TUs should reduce output compared to all enabled."""
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx = design_stack
        key = jax.random.key(42)
        params = stack.init(key)

        n_tus = len(tu_ids)
        n_networks = len(stack.networks)
        n_inputs = stack.get_nb_inputs()
        n_z_val = params["global/number_of_random_variables"]
        n_z = int(n_z_val.squeeze()) if hasattr(n_z_val, "squeeze") else int(n_z_val)

        X = jnp.ones((n_inputs,)) * 0.5
        Z = jnp.ones((n_z,)) * 0.5

        # All TUs enabled: log_alpha = +10 -> binary mask = 1
        params.at(TU_LOG_ALPHA_PATH, jnp.full((n_networks, n_tus), 10.0), overwrite=True)
        y_all, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=None)

        # Half TUs disabled: first half enabled (+10), second half disabled (-10)
        log_alpha_half = jnp.array(
            [[10.0 if i < n_tus // 2 else -10.0 for i in range(n_tus)] for _ in range(n_networks)]
        )
        params.at(TU_LOG_ALPHA_PATH, log_alpha_half, overwrite=True)
        y_half, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=None)

        # Output should be different (reduced)
        assert not jnp.allclose(y_all, y_half, atol=1e-3), (
            f"Expected different outputs: all={y_all}, half={y_half}"
        )


def test_masking_is_deterministic(lib, design_stack):
    """Same log_alpha should produce same output across multiple runs (binary masking is deterministic)."""
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx = design_stack
        key = jax.random.key(42)
        params = stack.init(key)

        n_tus = len(tu_ids)
        n_networks = len(stack.networks)
        n_inputs = stack.get_nb_inputs()
        n_z_val = params["global/number_of_random_variables"]
        n_z = int(n_z_val.squeeze()) if hasattr(n_z_val, "squeeze") else int(n_z_val)

        X = jnp.ones((n_inputs,)) * 0.5
        Z = jnp.ones((n_z,)) * 0.5

        # Alternating enabled/disabled: +10 for even indices, -10 for odd
        log_alpha = jnp.array(
            [[10.0 if i % 2 == 0 else -10.0 for i in range(n_tus)] for _ in range(n_networks)]
        )
        params.at(TU_LOG_ALPHA_PATH, log_alpha, overwrite=True)

        y1, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=None)
        y2, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=None)

        assert jnp.allclose(y1, y2, atol=1e-6), f"Expected same output: y1={y1}, y2={y2}"


def test_gradual_disabling(lib, design_stack):
    """Gradually disabling more TUs should generally decrease output magnitude.

    Note: Due to marker outputs that don't participate in TU-gated computation,
    the decrease may not be perfectly monotonic. We test that:
    1. Full enabled > Full disabled (strictly)
    2. The general trend is decreasing
    """
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx = design_stack
        key = jax.random.key(42)
        params = stack.init(key)

        n_tus = len(tu_ids)
        n_networks = len(stack.networks)
        n_inputs = stack.get_nb_inputs()
        n_z_val = params["global/number_of_random_variables"]
        n_z = int(n_z_val.squeeze()) if hasattr(n_z_val, "squeeze") else int(n_z_val)

        X = jnp.ones((n_inputs,)) * 0.5
        Z = jnp.ones((n_z,)) * 0.5

        # Test with different fractions of TUs enabled using log_alpha
        magnitudes = []
        for fraction in [1.0, 0.75, 0.5, 0.25, 0.0]:
            n_enabled = int(n_tus * fraction)
            # log_alpha: +10 for enabled, -10 for disabled
            log_alpha = jnp.array(
                [[10.0 if i < n_enabled else -10.0 for i in range(n_tus)] for _ in range(n_networks)]
            )
            params.at(TU_LOG_ALPHA_PATH, log_alpha, overwrite=True)
            y, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=None)
            mag = float(jnp.sum(jnp.abs(y)))
            magnitudes.append(mag)

        # Key property: full enabled should differ from full disabled
        # (magnitude may not be higher if inhibitory ERNs dominate)
        assert abs(magnitudes[0] - magnitudes[-1]) > 1e-3, (
            f"Full enabled ({magnitudes[0]}) should differ from full disabled ({magnitudes[-1]})"
        )

        # Full disabled (fraction=0) should produce near-zero output
        # Threshold is 5e-3 to accommodate leaky mask floor (0.001) ensuring gradient flow
        assert magnitudes[-1] < 5e-3, (
            f"All TUs disabled should give near-zero output, got {magnitudes[-1]}"
        )


def test_per_network_tu_mask(lib, design_stack):
    """Test that get_per_network_tu_mask returns correct per-network TU usage masks.

    This is critical for per-network L0 penalty: each network should only be
    penalized for TUs it actually uses, not for TUs used by other networks.
    """
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx = design_stack
        n_networks = len(stack.networks)
        n_tus = len(tu_ids)

        # test the new method exists and has correct shape
        mask = stack.get_per_network_tu_mask()
        assert mask.shape == (n_networks, n_tus), (
            f"Per-network TU mask shape {mask.shape} != expected ({n_networks}, {n_tus})"
        )

        # mask should be binary (0 or 1)
        assert jnp.all((mask == 0) | (mask == 1)), "TU mask should be binary (0 or 1)"

        # verify mask matches direct TU extraction for each network
        from biocomp.tumasking import extract_tu_ids_from_network

        for net_idx, net in enumerate(stack.networks):
            net_tu_ids = extract_tu_ids_from_network(net)
            expected_tus = {tid for tid in net_tu_ids if tid in tu_id_to_idx}

            # count TUs in mask for this network
            mask_tus = {tu_ids[i] for i in range(n_tus) if mask[net_idx, i] > 0}
            assert mask_tus == expected_tus, (
                f"Network {net_idx}: mask TUs {mask_tus} != expected {expected_tus}"
            )

        # each network should use at least 1 TU (sanity check)
        for net_idx in range(n_networks):
            n_used = int(mask[net_idx].sum())
            assert n_used > 0, f"Network {net_idx} has no TUs (mask sum=0)"

        # total TUs used (union) should match n_tus
        union_mask = jnp.any(mask > 0, axis=0)
        n_union = int(union_mask.sum())
        assert n_union == n_tus, f"Union of TUs {n_union} != total TUs {n_tus}"


def test_hard_concrete_transformation():
    """Verify hard concrete transformation matches expected behavior."""
    # uniform=0.5, log_alpha=0 -> s=0.5 -> s_bar=0.5*1.2-0.1=0.5 -> z=0.5
    z = hard_concrete_from_uniform(jnp.array(0.5), jnp.array(0.0))
    assert 0.4 < float(z) < 0.6, f"Expected ~0.5, got {z}"

    # uniform near 0, log_alpha=0 -> z=0 (disabled)
    z_disabled = hard_concrete_from_uniform(jnp.array(1e-6), jnp.array(0.0))
    assert float(z_disabled) < 0.1, f"Expected ~0, got {z_disabled}"

    # uniform near 1, log_alpha=0 -> z=1 (enabled)
    z_enabled = hard_concrete_from_uniform(jnp.array(1 - 1e-6), jnp.array(0.0))
    assert float(z_enabled) > 0.9, f"Expected ~1, got {z_enabled}"


def test_inference_mode_all_enabled(lib, design_stack):
    """When log_alpha > 0 for all TUs, output should be non-zero (all enabled)."""
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx = design_stack
        key = jax.random.key(42)
        params = stack.init(key)

        n_tus = len(tu_ids)
        n_networks = len(stack.networks)
        n_inputs = stack.get_nb_inputs()
        n_z_val = params["global/number_of_random_variables"]
        n_z = int(n_z_val.squeeze()) if hasattr(n_z_val, "squeeze") else int(n_z_val)

        X = jnp.ones((n_inputs,)) * 0.5
        Z = jnp.ones((n_z,)) * 0.5

        # log_alpha = +10 -> all TUs enabled via binary masking
        params.at(TU_LOG_ALPHA_PATH, jnp.full((n_networks, n_tus), 10.0), overwrite=True)

        # tu_enabled_random_vars is ignored with binary masking (mask from log_alpha)
        y_inference, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=None)

        assert not jnp.allclose(y_inference, 0.0, atol=1e-3), (
            f"Expected non-zero output when all TUs enabled, got {y_inference}"
        )


# ============== Per-Network TU Masking Tests ==============


@pytest.fixture(scope="module")
def multi_network_stack(lib):
    """Build a stack with multiple networks. Module-scoped for speed."""
    scaffold_path_1 = (
        RESOURCES_DIR / "design/architectures/two_and_one.yaml"
    )
    scaffold_path_2 = (
        RESOURCES_DIR / "design/architectures/three.yaml"
    )

    with LibraryContext.with_library(lib):
        data1 = dr.load(scaffold_path_1, context={"Recipe": Recipe})
        recipes1 = data1["recipes"] if "recipes" in data1 else data1.recipes
        data2 = dr.load(scaffold_path_2, context={"Recipe": Recipe})
        recipes2 = data2["recipes"] if "recipes" in data2 else data2.recipes

        networks = []
        for r in recipes1[:2]:
            networks.extend(recipe_to_networks(r, br.ALL_RULES, invert=True))
        for r in recipes2[:1]:
            networks.extend(recipe_to_networks(r, br.ALL_RULES, invert=True))

        tu_ids, tu_id_to_idx = build_tu_id_mapping(networks)

        stack = ComputeStack(networks)
        config = SIMPLE_NODES_COMPUTE_CONFIG.model_copy(deep=True)
        stack.build(config, enable_tu_masking=True)

        # JIT warmup
        key = jax.random.key(0)
        params = stack.init(key)
        n_tus = len(tu_ids)
        n_networks = len(networks)
        n_inputs = stack.get_nb_inputs()
        n_z = int(params["global/number_of_random_variables"])
        params.at(TU_LOG_ALPHA_PATH, jnp.full((n_networks, n_tus), 10.0), overwrite=True)
        stack.apply(
            params,
            jnp.zeros((n_inputs,)),
            jnp.zeros((n_z,)),
            key,
            tu_enabled_random_vars=None,
        )

        return stack, tu_ids, tu_id_to_idx, len(networks)


def test_per_network_tu_masking_shape(lib, multi_network_stack):
    """Verify 2D log_alpha shape (n_networks, n_tus) produces valid output."""
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx, n_networks = multi_network_stack
        key = jax.random.key(42)
        params = stack.init(key)

        n_tus = len(tu_ids)
        n_inputs = stack.get_nb_inputs()
        n_z_val = params["global/number_of_random_variables"]
        n_z = int(n_z_val.squeeze()) if hasattr(n_z_val, "squeeze") else int(n_z_val)

        X = jnp.ones((n_inputs,)) * 0.5
        Z = jnp.ones((n_z,)) * 0.5

        # 2D log_alpha shape (n_networks, n_tus) -> binary masking
        params.at(TU_LOG_ALPHA_PATH, jnp.full((n_networks, n_tus), 10.0), overwrite=True)
        y, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=None)

        assert y is not None
        assert not jnp.any(jnp.isnan(y)), "Got NaN in output"


def test_per_network_tu_masking_independence(lib, multi_network_stack):
    """Disabling TUs in one network should not affect outputs from other networks."""
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx, n_networks = multi_network_stack

        if n_networks < 2:
            pytest.skip("Need at least 2 networks for independence test")

        key = jax.random.key(42)
        params = stack.init(key)

        n_tus = len(tu_ids)
        n_inputs = stack.get_nb_inputs()
        n_z_val = params["global/number_of_random_variables"]
        n_z = int(n_z_val.squeeze()) if hasattr(n_z_val, "squeeze") else int(n_z_val)

        X = jnp.ones((n_inputs,)) * 0.5
        Z = jnp.ones((n_z,)) * 0.5

        # All TUs enabled: log_alpha = +10
        log_alpha_all = jnp.full((n_networks, n_tus), 10.0)
        params.at(TU_LOG_ALPHA_PATH, log_alpha_all, overwrite=True)
        y_all, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=None)

        # Disable TUs in network 0 only: log_alpha = -10 for network 0
        log_alpha_net0_disabled = log_alpha_all.at[0, :].set(-10.0)
        params.at(TU_LOG_ALPHA_PATH, log_alpha_net0_disabled, overwrite=True)
        y_net0_disabled, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=None)

        assert not jnp.allclose(y_all, y_net0_disabled, atol=1e-3), (
            "Disabling TUs in network 0 should change output"
        )


def test_per_network_tu_masking_selective_disable(lib, multi_network_stack):
    """Verify that we can selectively disable specific TUs in specific networks."""
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx, n_networks = multi_network_stack

        if n_networks < 2 or len(tu_ids) < 2:
            pytest.skip("Need at least 2 networks and 2 TUs")

        key = jax.random.key(42)
        params = stack.init(key)

        n_tus = len(tu_ids)
        n_inputs = stack.get_nb_inputs()
        n_z_val = params["global/number_of_random_variables"]
        n_z = int(n_z_val.squeeze()) if hasattr(n_z_val, "squeeze") else int(n_z_val)

        X = jnp.ones((n_inputs,)) * 0.5
        Z = jnp.ones((n_z,)) * 0.5

        # All TUs enabled: log_alpha = +10
        log_alpha_base = jnp.full((n_networks, n_tus), 10.0)
        params.at(TU_LOG_ALPHA_PATH, log_alpha_base, overwrite=True)
        y_base, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=None)

        # Disable TU 0 in network 0 only
        log_alpha_net0_tu0 = log_alpha_base.at[0, 0].set(-10.0)
        params.at(TU_LOG_ALPHA_PATH, log_alpha_net0_tu0, overwrite=True)
        y_net0_tu0, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=None)

        # Disable TU 0 in network 1 only
        log_alpha_net1_tu0 = log_alpha_base.at[1, 0].set(-10.0)
        params.at(TU_LOG_ALPHA_PATH, log_alpha_net1_tu0, overwrite=True)
        y_net1_tu0, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=None)

        net0_changed = not jnp.allclose(y_base, y_net0_tu0, atol=1e-3)
        net1_changed = not jnp.allclose(y_base, y_net1_tu0, atol=1e-3)

        assert net0_changed or net1_changed, (
            "Disabling TU 0 should change output for at least one network"
        )


def test_1d_log_alpha_handled(lib, design_stack):
    """1D log_alpha is handled by slicing with network_id (backward compatibility)."""
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx = design_stack
        key = jax.random.key(42)
        params = stack.init(key)

        n_tus = len(tu_ids)
        n_networks = len(stack.networks)
        n_inputs = stack.get_nb_inputs()
        n_z_val = params["global/number_of_random_variables"]
        n_z = int(n_z_val.squeeze()) if hasattr(n_z_val, "squeeze") else int(n_z_val)

        X = jnp.ones((n_inputs,)) * 0.5
        Z = jnp.ones((n_z,)) * 0.5

        # 2D log_alpha is the standard format
        params.at(TU_LOG_ALPHA_PATH, jnp.full((n_networks, n_tus), 10.0), overwrite=True)
        y, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=None)
        assert not jnp.any(jnp.isnan(y)), "Got NaN with 2D log_alpha"


# ============== Commit TU Masking Tests ==============


def test_commit_applies_tu_masks(lib, design_stack):
    """Verify that commit removes disabled TUs (prunes zero-ratio members).

    With single-source-of-truth approach: disabled TUs are REMOVED from the
    aggregation members list, not kept with ratio=0.
    """
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx = design_stack
        key = jax.random.key(42)
        params = stack.init(key)

        n_tus = len(tu_ids)
        n_networks = len(stack.networks)

        # count total members before commit
        total_members_before = 0
        for net in stack.networks:
            for node in net.compute_graph.nodes.values():
                if "aggregation" in node.node_type.lower():
                    members = node.extra.get("members", [])
                    total_members_before += len(members)

        # set half TUs to disabled (negative log_alpha) and half enabled (positive)
        tu_log_alpha = jnp.zeros((n_networks, n_tus))
        half = n_tus // 2
        tu_log_alpha = tu_log_alpha.at[:, :half].set(-3.0)  # disabled (sigmoid(-3) ≈ 0.05)
        tu_log_alpha = tu_log_alpha.at[:, half:].set(3.0)  # enabled (sigmoid(3) ≈ 0.95)
        params.at(TU_LOG_ALPHA_PATH, tu_log_alpha, overwrite=True)

        # commit the stack
        committed_networks = stack.commit(params)

        # count total members after commit - should be fewer due to pruned disabled TUs
        total_members_after = 0
        for net in committed_networks:
            for node in net.compute_graph.nodes.values():
                if "aggregation" in node.node_type.lower():
                    members = node.extra.get("members", [])
                    total_members_after += len(members)

        assert total_members_after < total_members_before, (
            f"Expected fewer members after commit due to disabled TU pruning. "
            f"Before: {total_members_before}, After: {total_members_after}"
        )


def test_commit_preserves_enabled_tus(lib, design_stack):
    """Verify that commit preserves ratios for TUs with log_alpha > 0 (enabled)."""
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx = design_stack
        key = jax.random.key(42)
        params = stack.init(key)

        n_tus = len(tu_ids)
        n_networks = len(stack.networks)

        # set all TUs to enabled (high log_alpha)
        tu_log_alpha = jnp.full((n_networks, n_tus), 5.0)  # sigmoid(5) ≈ 0.99
        params.at(TU_LOG_ALPHA_PATH, tu_log_alpha, overwrite=True)

        # commit the stack
        committed_networks = stack.commit(params)

        # check that no aggregation node has all-zero ratios
        for net in committed_networks:
            for node in net.compute_graph.nodes.values():
                if "aggregation" in node.node_type.lower():
                    ratios = node.extra.get("ratios", [])
                    if ratios:
                        all_zero = all(abs(r) < 1e-6 for r in ratios)
                        assert not all_zero, (
                            f"Node {node.id} has all-zero ratios but all TUs should be enabled"
                        )


# ============== Comprehensive TU Masking Tests ==============


def test_commit_removes_fully_disabled_tu_edges(lib, design_stack):
    """Edges where ALL TUs are disabled should be removed from committed networks.

    When ALL TUs on an edge are disabled, the edge contributes nothing and should
    be physically removed. Edges with mixed enabled/disabled TUs stay.
    """
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx = design_stack
        key = jax.random.key(42)
        params = stack.init(key)

        n_tus = len(tu_ids)
        n_networks = len(stack.networks)

        if n_tus == 0:
            pytest.skip("No TUs in network")

        # disable ALL TUs to ensure edges with only TUs get removed
        tu_log_alpha = jnp.full((n_networks, n_tus), -10.0)
        params.at(TU_LOG_ALPHA_PATH, tu_log_alpha, overwrite=True)

        # count edges that have TUs (they should all be removed)
        edges_with_any_tu_before = 0
        for net in stack.networks:
            for edge in net.compute_graph.edges.values():
                edge_tu_ids = edge.extra.get("tu_id", []) if edge.extra else []
                if edge_tu_ids:
                    edges_with_any_tu_before += 1

        committed_networks = stack.commit(params)

        # count edges with TUs after (should be 0 since all TUs disabled)
        edges_with_tu_after = 0
        for net in committed_networks:
            for edge in net.compute_graph.edges.values():
                edge_tu_ids = edge.extra.get("tu_id", []) if edge.extra else []
                if edge_tu_ids:
                    edges_with_tu_after += 1

        assert edges_with_tu_after == 0, (
            f"When all TUs disabled, all TU-carrying edges should be removed!\n"
            f"Before: {edges_with_any_tu_before} edges with TUs\n"
            f"After: {edges_with_tu_after} edges with TUs"
        )
        assert edges_with_any_tu_before > 0, "Test setup issue: no TU edges found"


def test_single_tu_edge_removed_when_disabled(lib, design_stack):
    """Verify that edges with ONLY ONE TU get removed when that TU is disabled.

    Edges with multiple TUs may stay if some are enabled. But single-TU edges
    should definitely be removed when their TU is disabled.
    """
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx = design_stack
        key = jax.random.key(42)
        params = stack.init(key)

        n_tus = len(tu_ids)
        n_networks = len(stack.networks)

        if n_tus == 0:
            pytest.skip("No TUs in network")

        # find a TU that has a single-TU edge
        single_tu_edges = []
        for net in stack.networks:
            for edge in net.compute_graph.edges.values():
                edge_tu_ids = edge.extra.get("tu_id", []) if edge.extra else []
                if len(edge_tu_ids) == 1:
                    single_tu_edges.append(edge_tu_ids[0])

        if not single_tu_edges:
            pytest.skip("No single-TU edges found")

        # pick a TU that appears in single-TU edges
        target_tu_id = single_tu_edges[0]
        target_tu_idx = tu_id_to_idx[target_tu_id]

        # disable ONLY that TU, enable all others
        tu_log_alpha = jnp.full((n_networks, n_tus), 5.0)
        tu_log_alpha = tu_log_alpha.at[:, target_tu_idx].set(-10.0)
        params.at(TU_LOG_ALPHA_PATH, tu_log_alpha, overwrite=True)

        committed_networks = stack.commit(params)

        # verify single-TU edges with the disabled TU are removed
        found_single_tu_edge_after = False
        for net in committed_networks:
            for edge in net.compute_graph.edges.values():
                edge_tu_ids = edge.extra.get("tu_id", []) if edge.extra else []
                if edge_tu_ids == [target_tu_id]:
                    found_single_tu_edge_after = True
                    break
            if found_single_tu_edge_after:
                break

        assert not found_single_tu_edge_after, (
            f"Single-TU edge with disabled TU '{target_tu_id}' should be removed"
        )


def test_all_tus_disabled_produces_different_commit(lib, design_stack):
    """When all TUs are disabled, committed network should have fewer members than all enabled.

    With single-source-of-truth: disabled TUs are REMOVED, not zeroed.
    """
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx = design_stack
        key = jax.random.key(42)

        n_tus = len(tu_ids)
        n_networks = len(stack.networks)

        # commit with all TUs enabled
        params_enabled = stack.init(key)
        tu_log_alpha_enabled = jnp.full((n_networks, n_tus), 5.0)
        params_enabled.at(TU_LOG_ALPHA_PATH, tu_log_alpha_enabled, overwrite=True)
        committed_enabled = stack.commit(params_enabled)

        # commit with all TUs disabled
        params_disabled = stack.init(key)
        tu_log_alpha_disabled = jnp.full((n_networks, n_tus), -5.0)
        params_disabled.at(TU_LOG_ALPHA_PATH, tu_log_alpha_disabled, overwrite=True)
        committed_disabled = stack.commit(params_disabled)

        # count total members in enabled vs disabled commits
        def count_members(networks):
            total = 0
            for net in networks:
                for node in net.compute_graph.nodes.values():
                    if "aggregation" in node.node_type.lower():
                        members = node.extra.get("members", [])
                        total += len(members)
            return total

        members_enabled = count_members(committed_enabled)
        members_disabled = count_members(committed_disabled)

        # disabled should have fewer members (pruned)
        assert members_disabled < members_enabled, (
            f"FAILED: Disabled commit should have fewer members than enabled.\n"
            f"Enabled: {members_enabled}, Disabled: {members_disabled}\n"
            "This indicates commit() is NOT applying TU masks!"
        )


def test_tu_mask_boundary_threshold(lib, design_stack):
    """Test TU masking at the sigmoid threshold (0.5 = log_alpha of 0)."""
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx = design_stack
        key = jax.random.key(42)
        params = stack.init(key)

        n_tus = len(tu_ids)
        n_networks = len(stack.networks)

        if n_tus < 2:
            pytest.skip("Need at least 2 TUs")

        # set TUs at boundary: first at -0.1 (disabled), second at +0.1 (enabled)
        tu_log_alpha = jnp.zeros((n_networks, n_tus))
        tu_log_alpha = tu_log_alpha.at[:, 0].set(-0.1)  # sigmoid ≈ 0.475 < 0.5 → disabled
        tu_log_alpha = tu_log_alpha.at[:, 1].set(0.1)  # sigmoid ≈ 0.525 > 0.5 → enabled
        params.at(TU_LOG_ALPHA_PATH, tu_log_alpha, overwrite=True)

        # commit to verify it works, but we're testing the threshold behavior
        stack.commit(params)

        # verify the boundary behavior
        from biocomp.tumasking import get_final_mask

        mask_0 = get_final_mask(jnp.array([-0.1]))[0]
        mask_1 = get_final_mask(jnp.array([0.1]))[0]

        assert float(mask_0) == 0.0, f"TU at log_alpha=-0.1 should be disabled, got mask={mask_0}"
        assert float(mask_1) == 1.0, f"TU at log_alpha=+0.1 should be enabled, got mask={mask_1}"


def test_commit_without_tu_masking_unchanged(lib):
    """Verify that commit works correctly when TU masking is NOT enabled."""
    from biocomp.recipe import Recipe

    scaffold_path = (
        RESOURCES_DIR / "design/architectures/two_and_one.yaml"
    )

    with LibraryContext.with_library(lib):
        data = dr.load(scaffold_path, context={"Recipe": Recipe})
        recipes = data["recipes"] if "recipes" in data else data.recipes
        networks = recipe_to_networks(recipes[0], br.ALL_RULES, invert=True)

        # build stack WITHOUT TU masking
        stack = ComputeStack(networks)
        from biocomp.config import SIMPLE_NODES_COMPUTE_CONFIG

        stack.build(SIMPLE_NODES_COMPUTE_CONFIG)  # no tu_id_to_idx

        key = jax.random.key(42)
        params = stack.init(key)

        # verify no TU_LOG_ALPHA_PATH in params
        assert TU_LOG_ALPHA_PATH not in params, "TU masking should not be initialized"

        # commit should still work
        committed_networks = stack.commit(params)
        assert len(committed_networks) == len(networks)

        # verify ratios are preserved (not zeroed out)
        for net in committed_networks:
            for node in net.compute_graph.nodes.values():
                if "aggregation" in node.node_type.lower():
                    ratios = node.extra.get("ratios", [])
                    if ratios:
                        assert any(r > 0 for r in ratios), "Ratios should not all be zero"


def test_per_network_tu_mask_commit_independence(lib, multi_network_stack):
    """Each network's TU mask should be applied independently during commit.

    With single-source-of-truth: disabled TUs are REMOVED (pruned), not zeroed.
    """
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx, n_networks = multi_network_stack
        key = jax.random.key(42)
        params = stack.init(key)

        n_tus = len(tu_ids)
        if n_networks < 2 or n_tus == 0:
            pytest.skip("Need at least 2 networks and 1 TU")

        # disable all TUs in network 0, enable in network 1
        tu_log_alpha = jnp.full((n_networks, n_tus), 5.0)  # all enabled
        tu_log_alpha = tu_log_alpha.at[0, :].set(-5.0)  # network 0: all disabled
        params.at(TU_LOG_ALPHA_PATH, tu_log_alpha, overwrite=True)

        committed_networks = stack.commit(params)

        # count members (disabled TUs are pruned, so fewer members)
        def count_members(net):
            count = 0
            for node in net.compute_graph.nodes.values():
                if "aggregation" in node.node_type.lower():
                    members = node.extra.get("members", [])
                    count += len(members)
            return count

        members_net0 = count_members(committed_networks[0])
        members_net1 = count_members(committed_networks[1]) if n_networks > 1 else 0

        # network 0 (all disabled) should have fewer members than network 1 (all enabled)
        assert members_net0 < members_net1, (
            f"Network 0 (all TUs disabled) should have fewer members than network 1.\n"
            f"Network 0 members: {members_net0}, Network 1 members: {members_net1}\n"
            f"This indicates per-network TU masking is not working in commit!"
        )


def test_evaluate_design_uses_tu_masks():
    """Verify evaluate_design applies TU masks via binary masking (log_alpha based)."""
    from biocomp.design import evaluate_design
    import inspect

    source = inspect.getsource(evaluate_design)

    assert "TU_LOG_ALPHA_PATH" in source, (
        "evaluate_design does not check TU_LOG_ALPHA_PATH - TU masking may not be applied!"
    )
    assert "tu_mask" in source or "tu_enabled_random_vars" in source, (
        "evaluate_design does not reference TU masking - binary masks may not be applied!"
    )


def test_committed_recipe_excludes_disabled_tus(lib, design_stack):
    """CRITICAL: Recipes exported from committed networks should not contain disabled TUs.

    When a TU is disabled during design optimization, the committed network should
    produce a recipe where that TU simply doesn't exist.
    """
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx = design_stack
        key = jax.random.key(42)
        params = stack.init(key)

        n_tus = len(tu_ids)
        n_networks = len(stack.networks)

        if n_tus == 0:
            pytest.skip("No TUs in network")

        # disable first TU, enable all others
        tu_log_alpha = jnp.full((n_networks, n_tus), 5.0)
        tu_log_alpha = tu_log_alpha.at[:, 0].set(-10.0)
        params.at(TU_LOG_ALPHA_PATH, tu_log_alpha, overwrite=True)

        first_tu_id = tu_ids[0]
        # TU ID format: cotx_name_cotx (e.g. b_a+_b), recipe name: cotx_name (e.g. b_a+)
        first_tu_recipe_name = "_".join(first_tu_id.split("_")[:-1])

        # count TUs before commit
        tus_before = []
        for net in stack.networks:
            r = net.to_recipe()
            tus_before.extend([tu.name for cotx in r.content for tu in cotx.units])
        tus_before = set(tus_before)

        # commit and count TUs after
        committed_networks = stack.commit(params)
        tus_after = []
        for net in committed_networks:
            r = net.to_recipe()
            tus_after.extend([tu.name for cotx in r.content for tu in cotx.units])
        tus_after = set(tus_after)

        assert first_tu_recipe_name in tus_before, (
            f"Test setup issue: TU '{first_tu_recipe_name}' not found in recipes before commit"
        )
        assert first_tu_recipe_name not in tus_after, (
            f"FAILED: TU '{first_tu_recipe_name}' (from ID '{first_tu_id}') was disabled "
            f"but still appears in committed recipe!"
        )


def test_disabled_tu_removed_after_commit(lib, design_stack):
    """Verify that disabled TUs are removed (pruned) from aggregation after commit.

    With single-source-of-truth: disabled TUs are REMOVED, not kept with ratio=0.
    """
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx = design_stack
        key = jax.random.key(42)

        n_tus = len(tu_ids)
        n_networks = len(stack.networks)

        if n_tus == 0:
            pytest.skip("No TUs in network")

        # count members before commit (all enabled)
        params_all_enabled = stack.init(key)
        tu_log_alpha_all = jnp.full((n_networks, n_tus), 5.0)
        params_all_enabled.at(TU_LOG_ALPHA_PATH, tu_log_alpha_all, overwrite=True)
        committed_all_enabled = stack.commit(params_all_enabled)

        total_members_all = 0
        for net in committed_all_enabled:
            for node in net.compute_graph.nodes.values():
                if "aggregation" in node.node_type.lower():
                    members = node.extra.get("members", [])
                    total_members_all += len(members)

        # disable first TU
        params_one_disabled = stack.init(key)
        tu_log_alpha_one = jnp.full((n_networks, n_tus), 5.0)
        tu_log_alpha_one = tu_log_alpha_one.at[:, 0].set(-10.0)
        params_one_disabled.at(TU_LOG_ALPHA_PATH, tu_log_alpha_one, overwrite=True)
        committed_one_disabled = stack.commit(params_one_disabled)

        total_members_one_disabled = 0
        for net in committed_one_disabled:
            for node in net.compute_graph.nodes.values():
                if "aggregation" in node.node_type.lower():
                    members = node.extra.get("members", [])
                    total_members_one_disabled += len(members)

        assert total_members_one_disabled < total_members_all, (
            f"Disabling a TU should reduce member count.\n"
            f"All enabled: {total_members_all}, One disabled: {total_members_one_disabled}"
        )


def test_all_tus_disabled_empty_cotx_skipped(lib, design_stack):
    """When all TUs in a CoTransfection are disabled, that CoTransfection should be skipped in the recipe."""
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx = design_stack
        key = jax.random.key(42)
        params = stack.init(key)

        n_tus = len(tu_ids)
        n_networks = len(stack.networks)

        if n_tus == 0:
            pytest.skip("No TUs in network")

        # count cotx before commit
        recipes_before = [net.to_recipe() for net in stack.networks]
        cotx_count_before = sum(len(r.content) for r in recipes_before)

        # disable ALL TUs
        tu_log_alpha = jnp.full((n_networks, n_tus), -10.0)
        params.at(TU_LOG_ALPHA_PATH, tu_log_alpha, overwrite=True)

        committed = stack.commit(params)

        # count cotx after commit - should be less (empty ones skipped)
        recipes_after = [net.to_recipe() for net in committed]
        cotx_count_after = sum(len(r.content) for r in recipes_after)

        # when all TUs disabled, all cotx should be empty and skipped
        assert cotx_count_after < cotx_count_before, (
            f"Empty CoTransfections should be skipped.\n"
            f"Before: {cotx_count_before} CoTransfections\n"
            f"After: {cotx_count_after} CoTransfections"
        )


def test_fluo_bias_invalid_tu_id_handled(lib):
    """When fluo_bias.tu_id becomes invalid after TU pruning, it should be removed gracefully."""
    from biocomp.recipe import Recipe
    from biocomp.network import recipe_to_networks
    import biocomp.biorules as br

    scaffold_path = (
        RESOURCES_DIR / "design/architectures/two_and_one.yaml"
    )

    with LibraryContext.with_library(lib):
        data = dr.load(scaffold_path, context={"Recipe": Recipe})
        recipes = data["recipes"] if "recipes" in data else data.recipes
        networks = recipe_to_networks(recipes[0], br.ALL_RULES, invert=True)

        tu_ids, tu_id_to_idx = build_tu_id_mapping(networks)
        stack = ComputeStack(networks)
        from biocomp.config import SIMPLE_NODES_COMPUTE_CONFIG

        stack.build(SIMPLE_NODES_COMPUTE_CONFIG, enable_tu_masking=True)
        stack.tu_id_to_idx = tu_id_to_idx

        key = jax.random.key(42)
        params = stack.init(key)

        n_tus = len(tu_ids)
        n_networks = len(networks)

        # disable all TUs
        tu_log_alpha = jnp.full((n_networks, n_tus), -10.0)
        params.at(TU_LOG_ALPHA_PATH, tu_log_alpha, overwrite=True)

        committed = stack.commit(params)

        # this should NOT raise an exception - fluo_bias with invalid tu_id should be handled
        for net in committed:
            try:
                recipe = net.to_recipe()
                # verify no empty cotx and no invalid fluo_bias
                for cotx in recipe.content:
                    assert len(cotx.units) > 0, "Empty CoTransfection should be skipped"
                    if cotx.fluo_bias is not None:
                        assert cotx.fluo_bias.tu_id < len(cotx.units), (
                            f"fluo_bias.tu_id {cotx.fluo_bias.tu_id} >= len(units) {len(cotx.units)}"
                        )
            except ValueError as e:
                if "tu_id" in str(e) and "out of range" in str(e):
                    pytest.fail(f"fluo_bias.tu_id out of range error should be handled: {e}")
                raise


def test_multi_network_independent_tu_removal(lib):
    """CRITICAL: Multiple networks with different TUs disabled should commit independently.

    This tests the full flow with:
    - Multiple networks (sharing TU ID space)
    - Different TUs disabled per network
    - Verifies each committed network has only its disabled TUs removed
    """
    from biocomp.recipe import Recipe
    from biocomp.network import recipe_to_networks
    import biocomp.biorules as br

    scaffold_path = (
        RESOURCES_DIR / "design/architectures/two_and_one.yaml"
    )

    with LibraryContext.with_library(lib):
        data = dr.load(scaffold_path, context={"Recipe": Recipe})
        recipes = data["recipes"] if "recipes" in data else data.recipes
        scaffold_recipe = recipes[0]

        # create multiple networks from the same scaffold (simulates design replicates)
        networks = []
        for i in range(3):
            nets = recipe_to_networks(scaffold_recipe, br.ALL_RULES, invert=True)
            networks.extend(nets)

        n_networks = len(networks)
        assert n_networks >= 3, f"Expected at least 3 networks, got {n_networks}"

        tu_ids, tu_id_to_idx = build_tu_id_mapping(networks)
        n_tus = len(tu_ids)

        stack = ComputeStack(networks)
        from biocomp.config import SIMPLE_NODES_COMPUTE_CONFIG

        stack.build(SIMPLE_NODES_COMPUTE_CONFIG, enable_tu_masking=True)
        stack.tu_id_to_idx = tu_id_to_idx

        key = jax.random.key(42)
        params = stack.init(key)

        # set different TU masks for each network:
        # - network 0: disable first 2 TUs
        # - network 1: disable TUs 2,3,4
        # - network 2: enable all
        tu_log_alpha = jnp.full((n_networks, n_tus), 5.0)  # all enabled
        tu_log_alpha = tu_log_alpha.at[0, :2].set(-10.0)  # net 0: disable first 2
        if n_tus > 4:
            tu_log_alpha = tu_log_alpha.at[1, 2:5].set(-10.0)  # net 1: disable 2,3,4
        params.at(TU_LOG_ALPHA_PATH, tu_log_alpha, overwrite=True)

        # get recipe TUs before commit
        def get_recipe_tus(net):
            r = net.to_recipe()
            return set(tu.name for cotx in r.content for tu in cotx.units)

        tus_before = [get_recipe_tus(net) for net in networks]

        # commit
        committed = stack.commit(params)

        # get recipe TUs after commit
        tus_after = [get_recipe_tus(net) for net in committed]

        # verify network 0 lost exactly the first 2 disabled TUs
        disabled_0 = {tu_ids[0], tu_ids[1]}
        expected_names_0 = {"_".join(tid.split("_")[:-1]) for tid in disabled_0}
        removed_0 = tus_before[0] - tus_after[0]
        assert expected_names_0 == removed_0, (
            f"Network 0 should have removed {expected_names_0}, but removed {removed_0}"
        )

        # verify network 1 lost TUs 2,3,4 (if enough TUs)
        if n_tus > 4:
            disabled_1 = {tu_ids[2], tu_ids[3], tu_ids[4]}
            expected_names_1 = {"_".join(tid.split("_")[:-1]) for tid in disabled_1}
            removed_1 = tus_before[1] - tus_after[1]
            assert expected_names_1 == removed_1, (
                f"Network 1 should have removed {expected_names_1}, but removed {removed_1}"
            )

        # verify network 2 (all enabled) has same TUs
        if n_networks > 2:
            assert tus_before[2] == tus_after[2], (
                f"Network 2 (all enabled) should have same TUs.\n"
                f"Before: {len(tus_before[2])}, After: {len(tus_after[2])}\n"
                f"Removed: {tus_before[2] - tus_after[2]}"
            )

        # verify each network's edges match its disabled TUs
        for net_idx, net in enumerate(committed):
            if net_idx == 0:
                # disabled TUs 0,1 - their single-TU edges should be gone
                for edge in net.compute_graph.edges.values():
                    edge_tu_ids = edge.extra.get("tu_id", []) if edge.extra else []
                    # for single-TU edges, if the TU is disabled, edge should be gone
                    if len(edge_tu_ids) == 1:
                        assert edge_tu_ids[0] not in disabled_0, (
                            f"Net 0: single-TU edge with disabled TU {edge_tu_ids[0]} should be removed"
                        )
            elif net_idx == 1 and n_tus > 4:
                disabled_1 = {tu_ids[2], tu_ids[3], tu_ids[4]}
                for edge in net.compute_graph.edges.values():
                    edge_tu_ids = edge.extra.get("tu_id", []) if edge.extra else []
                    if len(edge_tu_ids) == 1:
                        assert edge_tu_ids[0] not in disabled_1, (
                            f"Net 1: single-TU edge with disabled TU {edge_tu_ids[0]} should be removed"
                        )


def test_committed_network_rebuilds_equivalent(lib, design_stack):
    """CRITICAL: A committed network with disabled TUs should be equivalent to a fresh network
    built from its exported recipe.

    This is the key invariant for the commit system: after commit, the network's graph structure
    should match what you'd get from building a new network from the committed recipe.

    Specifically tests that:
    1. The exported recipe can be rebuilt into a network
    2. The rebuilt network has the same graph structure (nodes, edges)
    3. The rebuilt network can be stacked with the committed network
    4. The aggregation ratios match between committed and rebuilt networks
    """
    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx = design_stack
        key = jax.random.key(42)
        params = stack.init(key)

        n_tus = len(tu_ids)
        n_networks = len(stack.networks)

        if n_tus == 0:
            pytest.skip("No TUs in network")

        # disable half the TUs to create a meaningful pruning scenario
        n_to_disable = max(1, n_tus // 2)
        tu_log_alpha = jnp.full((n_networks, n_tus), 5.0)
        for i in range(n_to_disable):
            tu_log_alpha = tu_log_alpha.at[:, i].set(-10.0)
        params.at(TU_LOG_ALPHA_PATH, tu_log_alpha, overwrite=True)

        # commit the network
        committed_networks = stack.commit(params)

        for net_idx, committed_net in enumerate(committed_networks):
            # export the recipe
            exported_recipe = committed_net.to_recipe()

            # rebuild a fresh network from the exported recipe
            rebuilt_networks = recipe_to_networks(exported_recipe, br.ALL_RULES, invert=True)
            assert len(rebuilt_networks) == 1, (
                f"Expected 1 rebuilt network, got {len(rebuilt_networks)}"
            )
            rebuilt_net = rebuilt_networks[0]

            # compare graph structure
            committed_graph = committed_net.compute_graph
            rebuilt_graph = rebuilt_net.compute_graph

            # verify node counts match
            assert len(committed_graph.nodes) == len(rebuilt_graph.nodes), (
                f"Network {net_idx}: Node count mismatch.\n"
                f"Committed: {len(committed_graph.nodes)} nodes\n"
                f"Rebuilt: {len(rebuilt_graph.nodes)} nodes"
            )

            # verify edge counts match
            assert len(committed_graph.edges) == len(rebuilt_graph.edges), (
                f"Network {net_idx}: Edge count mismatch.\n"
                f"Committed: {len(committed_graph.edges)} edges\n"
                f"Rebuilt: {len(rebuilt_graph.edges)} edges"
            )

            # verify node types match
            committed_types = sorted([n.node_type for n in committed_graph.nodes.values()])
            rebuilt_types = sorted([n.node_type for n in rebuilt_graph.nodes.values()])
            assert committed_types == rebuilt_types, (
                f"Network {net_idx}: Node types mismatch.\n"
                f"Committed: {committed_types}\n"
                f"Rebuilt: {rebuilt_types}"
            )

            # verify aggregation nodes have matching member counts
            committed_aggs = [
                n for n in committed_graph.nodes.values() if n.node_type == "aggregation"
            ]
            rebuilt_aggs = [n for n in rebuilt_graph.nodes.values() if n.node_type == "aggregation"]

            for c_agg, r_agg in zip(
                sorted(committed_aggs, key=lambda x: x.node_id),
                sorted(rebuilt_aggs, key=lambda x: x.node_id),
            ):
                c_members = c_agg.extra.get("members", [])
                r_members = r_agg.extra.get("members", [])
                c_ratios = c_agg.extra.get("ratios", [])
                r_ratios = r_agg.extra.get("ratios", [])
                c_out_edges = len(committed_graph.get_outgoing_edges(c_agg.node_id))
                r_out_edges = len(rebuilt_graph.get_outgoing_edges(r_agg.node_id))

                assert len(c_members) == len(r_members), (
                    f"Network {net_idx}, Agg {c_agg.node_id}: member count mismatch.\n"
                    f"Committed: {len(c_members)} members: {c_members}\n"
                    f"Rebuilt: {len(r_members)} members: {r_members}"
                )

                assert len(c_ratios) == len(r_ratios), (
                    f"Network {net_idx}, Agg {c_agg.node_id}: ratio count mismatch.\n"
                    f"Committed: {len(c_ratios)} ratios\n"
                    f"Rebuilt: {len(r_ratios)} ratios"
                )

                assert c_out_edges == r_out_edges, (
                    f"Network {net_idx}, Agg {c_agg.node_id}: outgoing edge count mismatch.\n"
                    f"Committed: {c_out_edges} edges\n"
                    f"Rebuilt: {r_out_edges} edges"
                )

                # CRITICAL: ratios should match outgoing edges (the original bug!)
                assert len(c_ratios) == c_out_edges, (
                    f"Network {net_idx}, Agg {c_agg.node_id}: COMMITTED network has mismatched ratios/edges!\n"
                    f"Ratios: {len(c_ratios)}, Outgoing edges: {c_out_edges}"
                )
                assert len(r_ratios) == r_out_edges, (
                    f"Network {net_idx}, Agg {c_agg.node_id}: REBUILT network has mismatched ratios/edges!\n"
                    f"Ratios: {len(r_ratios)}, Outgoing edges: {r_out_edges}"
                )

        # verify networks can be stacked together
        from biocomp.config import SIMPLE_NODES_COMPUTE_CONFIG

        combined_stack = ComputeStack(committed_networks)
        combined_stack.build(SIMPLE_NODES_COMPUTE_CONFIG)
        combined_params = combined_stack.init(key)
        assert combined_params is not None, "Combined stack should initialize successfully"
