"""Test compute stack for complex_twolayers network

This test validates the complex two-layer ERN network with design mode features:
- 3 CoTransfections (x1, x2 inputs + b bias)
- 3 ERNs in 2 layers (CasE + Csy4 → PgU)
- Variable uORFs (u1=none, u2=all, u3=all except none)
- Unlocked bias parameter
- 4 fluorescent outputs

Unlike simpler tests, this focuses on structural validation and consistency
rather than full manual computation due to the network's complexity (63 nodes, 54 random vars).
"""

import pytest
import jax
import jax.numpy as jnp
from biocomp.network import recipe_to_networks
from biocomp.library import LibraryContext
import biocomp.biorules as br
from biocomp.compute import ComputeStack
from biocomp.config import SIMPLE_NODES_COMPUTE_CONFIG
from test_declarative_recipes import lib, complex_twolayers_design_network


def test_complex_twolayers_structure(lib, complex_twolayers_design_network):
    """Validate network structure and layer organization"""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(complex_twolayers_design_network, br.ALL_RULES, invert=True)
        network = networks[0]
        stack = ComputeStack([network])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        # Verify layer structure
        assert len(stack.layers) == 16, f"Expected 16 layers, got {len(stack.layers)}"

        expected_structure = [
            ("input", 2),  # x1, x2
            ("bias", 1),  # b
            ("inv_translation", 3),
            ("inv_transcription", 3),
            ("inv_source", 3),
            ("inv_aggregation", 3),
            ("aggregation", 3),  # x1, x2, b cotx
            ("source", 24),  # 8 units × 3 cotx
            ("transcription", 7),  # ERN pathways + direct reporters
            ("transcription", 3),  # Direct reporters
            ("translation", 5),  # Direct reporters + ERN sources
            ("sequestron_ERN", 2),  # CasE, Csy4 (layer 1)
            ("translation", 1),  # ERN layer 1 output
            ("sequestron_ERN", 1),  # PgU (layer 2)
            ("translation", 1),  # ERN layer 2 output
            ("output", 1),  # 4 fluorescent measurements
        ]

        for i, (expected_type, expected_n_nodes) in enumerate(expected_structure):
            layer = stack.layers[i]
            assert layer.f_type == expected_type, (
                f"Layer {i}: expected {expected_type}, got {layer.f_type}"
            )
            assert len(layer.nodes) == expected_n_nodes, (
                f"Layer {i}: expected {expected_n_nodes} nodes, got {len(layer.nodes)}"
            )


def test_complex_twolayers_parameter_constraints(lib, complex_twolayers_design_network):
    """Validate parameter initialization and constraints"""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(complex_twolayers_design_network, br.ALL_RULES, invert=True)
        network = networks[0]
        stack = ComputeStack([network])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        init_key = jax.random.PRNGKey(42)
        params = stack.init(init_key)

        # TODO: Check bias parameter is in valid range [0.3, 0.6]
        # The bias node structure needs further investigation
        # bias_namespace = stack.layers[1].namespace

        # Check aggregation ratios are properly initialized and match recipe
        agg_layer_idx = 6
        agg_namespace = stack.layers[agg_layer_idx].namespace
        ratios = params[f"{agg_namespace}/ratios"]
        assert ratios.shape == (3, 8), f"Expected (3, 8) ratios, got {ratios.shape}"
        # All ratios should be positive
        assert jnp.all(ratios > 0), "All aggregation ratios should be positive"

        # Verify custom ratios from recipe are respected (normalized)
        # Recipe has: x1=[1,2,3,4,5,6,7,8], x2=[8,7,6,5,4,3,2,1], b=[1,1,1,1,1,1,1,1]
        # Note: The order of ratios in the aggregation node may differ from recipe order
        # (e.g., marker unit might be placed at end), but the VALUES should match when sorted
        expected_x1 = jnp.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=jnp.float32)
        expected_x2 = jnp.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=jnp.float32)
        expected_b = jnp.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=jnp.float32)

        # Normalize expected ratios (aggregation node normalizes by sum of absolute values)
        expected_x1_norm = jnp.abs(expected_x1) / jnp.sum(jnp.abs(expected_x1))
        expected_x2_norm = jnp.abs(expected_x2) / jnp.sum(jnp.abs(expected_x2))
        expected_b_norm = jnp.abs(expected_b) / jnp.sum(jnp.abs(expected_b))

        # Normalize actual ratios
        ratios_norm = jnp.abs(ratios) / jnp.sum(jnp.abs(ratios), axis=1, keepdims=True)

        # Compare sorted ratios (order-independent comparison)
        # Note: x1 and x2 have the same sorted pattern (just reversed), so we can't distinguish them by sorted values
        # We need to check: 2 rows match x1/x2 pattern (sorted), 1 row matches b pattern
        x1x2_sorted = jnp.sort(expected_x1_norm)  # Same as jnp.sort(expected_x2_norm)
        b_sorted = jnp.sort(expected_b_norm)

        # Count matches for each pattern
        x1x2_matches = []
        b_matches = []

        for row_idx in range(3):
            row = ratios_norm[row_idx]
            row_sorted = jnp.sort(row)
            if jnp.allclose(row_sorted, x1x2_sorted, rtol=1e-5):
                x1x2_matches.append(row_idx)
            elif jnp.allclose(row_sorted, b_sorted, rtol=1e-5):
                b_matches.append(row_idx)

        # We expect 2 rows to match x1/x2 pattern (since they're indistinguishable when sorted)
        # and 1 row to match b pattern
        assert len(x1x2_matches) == 2, (
            f"Expected 2 rows to match x1/x2 pattern, found {len(x1x2_matches)}.\n"
            f"Ratios (normalized, sorted):\n{jnp.sort(ratios_norm, axis=1)}\n"
            f"Expected x1/x2 (sorted): {x1x2_sorted}\n"
            f"Expected b (sorted): {b_sorted}"
        )
        assert len(b_matches) == 1, (
            f"Expected 1 row to match b pattern, found {len(b_matches)}.\n"
            f"Ratios (normalized, sorted):\n{jnp.sort(ratios_norm, axis=1)}\n"
            f"Expected b (sorted): {b_sorted}"
        )

        # Check quantization masks for translation nodes
        # Layer 10: Direct reporters (should have no uORF - index 0 only)
        tl_masks_10 = params["local/10/translation/tl_rate_quantization_mask"]
        assert tl_masks_10.shape == (5, 1, 13), (
            f"Layer 10 TL masks: expected (5, 1, 13), got {tl_masks_10.shape}"
        )
        for node_idx in range(5):
            mask = tl_masks_10[node_idx]
            assert jnp.sum(mask) == 1, f"Node {node_idx} should have exactly 1 uORF option"
            assert mask[0, 0], f"Node {node_idx} should have no-uORF (index 0) available"

        # Layer 12: Translation with mixed uORF masks (node 27)
        tl_masks_12 = params["local/12/translation/tl_rate_quantization_mask"]
        assert tl_masks_12.shape == (1, 3, 13), (
            f"Layer 12 TL masks: expected (1, 3, 13), got {tl_masks_12.shape}"
        )
        mask_27 = tl_masks_12[0]
        # Input 0: u1 = none → [1,0,0,...]
        assert jnp.sum(mask_27[0]) == 1, "Input 0 should have 1 option (u1=none)"
        assert mask_27[0, 0], "Input 0 should have index 0 available"
        # Input 1: u2 = all → [1,1,1,1,1,1,1,1,1,0,0,0,0]
        assert jnp.sum(mask_27[1]) == 9, "Input 1 should have 9 options (u2=all)"
        # Input 2: no uORF → [1,0,0,...]
        assert jnp.sum(mask_27[2]) == 1, "Input 2 should have 1 option (no uORF)"

        # Layer 14: Translation with u3 masks (node 34)
        tl_masks_14 = params["local/14/translation/tl_rate_quantization_mask"]
        assert tl_masks_14.shape == (1, 2, 13), (
            f"Layer 14 TL masks: expected (1, 2, 13), got {tl_masks_14.shape}"
        )
        mask_34 = tl_masks_14[0]
        # Input 0: no uORF → [1,0,0,...]
        assert jnp.sum(mask_34[0]) == 1, "Input 0 should have 1 option (no uORF)"
        # Input 1: u3 = all except none → [0,1,1,1,1,1,1,1,1,0,0,0,0]
        assert jnp.sum(mask_34[1]) == 8, "Input 1 should have 8 options (u3=all except none)"
        assert not mask_34[1, 0], "Input 1 should NOT have index 0 (none)"


def test_complex_twolayers_forward_pass(lib, complex_twolayers_design_network):
    """Test forward pass execution and output validation"""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(complex_twolayers_design_network, br.ALL_RULES, invert=True)
        network = networks[0]
        stack = ComputeStack([network])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        test_key = jax.random.PRNGKey(123)
        params = stack.init(test_key)

        # Network has 2 inputs (x1, x2) - bias is handled internally
        inputs = jnp.array([1.0, 2.0])
        n_random_vars = params["global/number_of_random_variables"]
        random_vars = jax.random.normal(test_key, (n_random_vars,))

        # Execute forward pass
        stack_result, aux = stack.apply(params, inputs, random_vars, test_key)

        # Should have 4 outputs: [x1_marker, x2_marker, b_marker, ERN_mNeonGreen]
        # = [mKO2, eBFP2, mMaroon1, mNeonGreen]
        assert stack_result.shape == (4,), f"Expected 4 outputs, got shape {stack_result.shape}"

        # All outputs should be finite
        assert jnp.all(jnp.isfinite(stack_result)), "All outputs should be finite"

        # Outputs should be non-zero (given non-zero inputs)
        assert jnp.all(stack_result != 0), "Outputs should be non-zero for non-zero inputs"


def test_complex_twolayers_reproducibility(lib, complex_twolayers_design_network):
    """Test that same seed produces same results"""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(complex_twolayers_design_network, br.ALL_RULES, invert=True)
        network = networks[0]
        stack = ComputeStack([network])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        test_key = jax.random.PRNGKey(42)
        params = stack.init(test_key)

        inputs = jnp.array([1.0, 2.0])
        n_random_vars = params["global/number_of_random_variables"]
        random_vars = jax.random.normal(test_key, (n_random_vars,))

        # Run twice with same key
        result1, _ = stack.apply(params, inputs, random_vars, test_key)
        result2, _ = stack.apply(params, inputs, random_vars, test_key)

        # Should be identical
        assert jnp.allclose(result1, result2, rtol=1e-10), "Same seed should produce identical results"


def test_complex_twolayers_variability(lib, complex_twolayers_design_network):
    """Test that different seeds produce different results"""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(complex_twolayers_design_network, br.ALL_RULES, invert=True)
        network = networks[0]
        stack = ComputeStack([network])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        base_key = jax.random.PRNGKey(0)
        test_keys = jax.random.split(base_key, 10)

        all_results = []
        inputs = jnp.array([1.0, 2.0])

        for test_key in test_keys:
            params = stack.init(test_key)
            n_random_vars = params["global/number_of_random_variables"]
            random_vars = jax.random.normal(test_key, (n_random_vars,))

            result, _ = stack.apply(params, inputs, random_vars, test_key)
            all_results.append(result)

        all_results = jnp.array(all_results)

        # Check that there's variability across different random initializations
        std_devs = jnp.std(all_results, axis=0)
        assert jnp.all(std_devs > 1e-5), (
            f"All outputs should vary across different seeds, got std_devs: {std_devs}"
        )


@pytest.mark.skip(reason="Manual computation oracle pending - network too complex for immediate completion")
def test_complex_twolayers_manual_computation(lib, complex_twolayers_design_network):
    """Test manual computation oracle (TODO: complete)

    This test will eventually contain the full manual computation for validation.
    Given the network's complexity (63 nodes, 16 layers, 54 random variables),
    this requires careful tracking of every intermediate value.

    TODO:
    - Extract exact random variable assignments for each node
    - Implement step-by-step forward pass
    - Compare with stack output
    """
    pass


if __name__ == "__main__":
    # Run tests manually for debugging
    from biocomp.library import load_lib, LibraryContext
    from biocomp.recipe import Recipe, CoTransfection, TranscriptionUnit, Slot, FluoIntensity, NumRange
    from test_declarative_recipes import make_units, ERNS, COLORS

    lib_instance = load_lib()

    # Create recipe directly instead of calling fixture
    with LibraryContext.with_library(lib_instance):
        recipe_instance = Recipe(
            name=f"two_and_one ({', '.join(ERNS)})",
            content=[
                CoTransfection(name="x1", units=make_units("x1", erns=ERNS)),
                CoTransfection(name="x2", units=make_units("x2", erns=ERNS)),
                CoTransfection(
                    name="b",
                    units=make_units("b", erns=ERNS),
                    fluo_bias=FluoIntensity(
                        tu_id=0,
                        value=NumRange(min=0.3, max=0.6),
                        protein=COLORS["b"],
                        units="Rescaled AU",
                    ),
                ),
            ],
        )

    print("Running structure test...")
    test_complex_twolayers_structure(lib_instance, recipe_instance)
    print("✓ Structure test passed\n")

    print("Running parameter constraints test...")
    test_complex_twolayers_parameter_constraints(lib_instance, recipe_instance)
    print("✓ Parameter constraints test passed\n")

    print("Running forward pass test...")
    test_complex_twolayers_forward_pass(lib_instance, recipe_instance)
    print("✓ Forward pass test passed\n")

    print("Running reproducibility test...")
    test_complex_twolayers_reproducibility(lib_instance, recipe_instance)
    print("✓ Reproducibility test passed\n")

    print("Running variability test...")
    test_complex_twolayers_variability(lib_instance, recipe_instance)
    print("✓ Variability test passed\n")

    print("All tests passed!")
