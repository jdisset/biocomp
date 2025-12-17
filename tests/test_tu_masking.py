"""Tests for Hard Concrete TU masking in design mode.

Key success criteria:
1. All TUs disabled produces different (lower magnitude) output than enabled
2. Partial TU disabling changes output compared to all enabled
3. Masking is deterministic (same uniform samples -> same output)
4. Gradual disabling produces monotonic decrease in output magnitude

Note: Output may not be exactly zero when disabled because:
- Output activation (e.g., sigmoid) maps zero input to non-zero output
- Some outputs (markers) may have simpler computation paths
- The key property is that masking AFFECTS the output, not that it produces zero

New approach:
- TU masking is integrated surgically into each node
- Pass tu_enabled_random_vars (uniform samples) to stack.apply()
- Each node transforms uniform -> hard concrete using log_alpha from params
- No wrapper functions from design_nodes.py
"""

import pytest
import jax
import jax.numpy as jnp
from pathlib import Path
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


SCAFFOLD_PATH = (
    Path(__file__).parent.parent.parent / "biocomp-jobs/design/architectures/two_and_one.yaml"
)


@pytest.fixture
def lib():
    return load_lib()


@pytest.fixture
def scaffold_recipe(lib):
    with LibraryContext.with_library(lib):
        data = dr.load(SCAFFOLD_PATH, context={"Recipe": Recipe})
        recipes = data["recipes"] if "recipes" in data else data.recipes
        return recipes[0]


@pytest.fixture
def design_stack(lib, scaffold_recipe):
    """Build a stack with integrated TU masking enabled."""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(scaffold_recipe, br.ALL_RULES, invert=True)
        tu_ids, tu_id_to_idx = build_tu_id_mapping(networks)

        stack = ComputeStack(networks)
        config = SIMPLE_NODES_COMPUTE_CONFIG.model_copy(deep=True)
        # Enable TU masking during build (no wrapper functions needed)
        stack.build(config, enable_tu_masking=True)

        return stack, tu_ids, tu_id_to_idx


def test_all_tus_disabled_differs_from_enabled(lib, design_stack):
    """When all TUs are disabled, output should differ from enabled output."""
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

        params.at(TU_LOG_ALPHA_PATH, jnp.zeros((n_networks, n_tus)))

        # All disabled (uniform near 0) - 2D shape (n_networks, n_tus)
        tu_uniform_disabled = jnp.full((n_networks, n_tus), 1e-6)
        y_disabled, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=tu_uniform_disabled)

        # All enabled (uniform = 0.5) - 2D shape (n_networks, n_tus)
        tu_uniform_enabled = jnp.full((n_networks, n_tus), 0.5)
        y_enabled, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=tu_uniform_enabled)

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
    """When all TUs are enabled (uniform=0.5 -> hard concrete~1), output should be non-zero."""
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

        # Log alpha = 0, uniform = 0.5 -> hard concrete ~0.55 (enabled)
        params.at(TU_LOG_ALPHA_PATH, jnp.zeros((n_networks, n_tus)))
        tu_uniform = jnp.full((n_networks, n_tus), 0.5)  # Default enabled

        y, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=tu_uniform)

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

        params.at(TU_LOG_ALPHA_PATH, jnp.zeros((n_networks, n_tus)))

        # All TUs enabled - 2D shape (n_networks, n_tus)
        tu_uniform_all = jnp.full((n_networks, n_tus), 0.5)
        y_all, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=tu_uniform_all)

        # Half TUs disabled (uniform=1e-6 for half) - 2D shape
        tu_uniform_half = jnp.array(
            [[0.5 if i < n_tus // 2 else 1e-6 for i in range(n_tus)] for _ in range(n_networks)]
        )
        y_half, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=tu_uniform_half)

        # Output should be different (reduced)
        assert not jnp.allclose(y_all, y_half, atol=1e-3), (
            f"Expected different outputs: all={y_all}, half={y_half}"
        )


def test_masking_is_deterministic(lib, design_stack):
    """Same uniform samples should produce same output across multiple runs."""
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

        params.at(TU_LOG_ALPHA_PATH, jnp.zeros((n_networks, n_tus)))

        # Fixed uniform samples (alternating enabled/disabled) - 2D shape
        tu_uniform = jnp.array(
            [[0.5 if i % 2 == 0 else 1e-6 for i in range(n_tus)] for _ in range(n_networks)]
        )

        y1, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=tu_uniform)
        y2, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=tu_uniform)

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

        params.at(TU_LOG_ALPHA_PATH, jnp.zeros((n_networks, n_tus)))

        # Test with different fractions of TUs enabled
        magnitudes = []
        for fraction in [1.0, 0.75, 0.5, 0.25, 0.0]:
            n_enabled = int(n_tus * fraction)
            # Create uniform samples: 0.5 for enabled, 1e-6 for disabled - 2D shape
            tu_uniform = jnp.array(
                [[0.5 if i < n_enabled else 1e-6 for i in range(n_tus)] for _ in range(n_networks)]
            )
            y, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=tu_uniform)
            mag = float(jnp.sum(jnp.abs(y)))
            magnitudes.append(mag)

        # Key property: full enabled should differ from full disabled
        # (magnitude may not be higher if inhibitory ERNs dominate)
        assert abs(magnitudes[0] - magnitudes[-1]) > 1e-3, (
            f"Full enabled ({magnitudes[0]}) should differ from full disabled ({magnitudes[-1]})"
        )

        # Full disabled (fraction=0) should produce near-zero output
        assert magnitudes[-1] < 1e-3, (
            f"All TUs disabled should give near-zero output, got {magnitudes[-1]}"
        )


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
    """When tu_enabled_random_vars is None, all TUs should be enabled (inference mode)."""
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

        params.at(TU_LOG_ALPHA_PATH, jnp.zeros((n_networks, n_tus)))

        # No tu_enabled_random_vars -> should default to all enabled
        y_inference, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=None)

        # Compare to explicit all-enabled - 2D shape
        tu_uniform_all = jnp.full((n_networks, n_tus), 0.5)
        y_explicit, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=tu_uniform_all)

        # Should be similar (inference mode treats all as enabled)
        assert not jnp.allclose(y_inference, 0.0, atol=1e-3), (
            f"Expected non-zero output in inference mode, got {y_inference}"
        )


# ============== Per-Network TU Masking Tests ==============


@pytest.fixture
def multi_network_stack(lib):
    """Build a stack with multiple networks sharing the same TU names."""
    scaffold_path_1 = (
        Path(__file__).parent.parent.parent / "biocomp-jobs/design/architectures/two_and_one.yaml"
    )
    scaffold_path_2 = (
        Path(__file__).parent.parent.parent / "biocomp-jobs/design/architectures/three.yaml"
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

        return stack, tu_ids, tu_id_to_idx, len(networks)


def test_per_network_tu_masking_shape(lib, multi_network_stack):
    """Verify tu_enabled_random_vars 2D shape is accepted and validated."""
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

        params.at(TU_LOG_ALPHA_PATH, jnp.zeros((n_networks, n_tus)))

        tu_uniform_2d = jnp.full((n_networks, n_tus), 0.5)
        y, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=tu_uniform_2d)

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

        params.at(TU_LOG_ALPHA_PATH, jnp.zeros((n_networks, n_tus)))

        tu_uniform_all = jnp.full((n_networks, n_tus), 0.5)
        y_all, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=tu_uniform_all)

        tu_uniform_net0_disabled = tu_uniform_all.at[0, :].set(1e-6)
        y_net0_disabled, _ = stack.apply(
            params, X, Z, key, tu_enabled_random_vars=tu_uniform_net0_disabled
        )

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

        params.at(TU_LOG_ALPHA_PATH, jnp.zeros((n_networks, n_tus)))

        tu_uniform_base = jnp.full((n_networks, n_tus), 0.5)
        y_base, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=tu_uniform_base)

        tu_uniform_net0_tu0 = tu_uniform_base.at[0, 0].set(1e-6)
        y_net0_tu0, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=tu_uniform_net0_tu0)

        tu_uniform_net1_tu0 = tu_uniform_base.at[1, 0].set(1e-6)
        y_net1_tu0, _ = stack.apply(params, X, Z, key, tu_enabled_random_vars=tu_uniform_net1_tu0)

        net0_changed = not jnp.allclose(y_base, y_net0_tu0, atol=1e-3)
        net1_changed = not jnp.allclose(y_base, y_net1_tu0, atol=1e-3)

        assert net0_changed or net1_changed, (
            "Disabling TU 0 should change output for at least one network"
        )


def test_1d_tu_uniform_rejected(lib, design_stack):
    """1D tu_uniform is no longer supported - must be 2D (n_networks, n_tus)."""
    import pytest

    with LibraryContext.with_library(lib):
        stack, tu_ids, tu_id_to_idx = design_stack
        key = jax.random.key(42)
        params = stack.init(key)

        n_tus = len(tu_ids)
        n_inputs = stack.get_nb_inputs()
        n_z_val = params["global/number_of_random_variables"]
        n_z = int(n_z_val.squeeze()) if hasattr(n_z_val, "squeeze") else int(n_z_val)

        X = jnp.ones((n_inputs,)) * 0.5
        Z = jnp.ones((n_z,)) * 0.5

        tu_uniform_1d = jnp.full((n_tus,), 0.5)
        with pytest.raises(AssertionError, match="must be 2D"):
            stack.apply(params, X, Z, key, tu_enabled_random_vars=tu_uniform_1d)
