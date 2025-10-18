"""Test compute stack with simplified nodes

This test file validates that the compute stack correctly executes simple node functions
by comparing stack outputs against manually computed expected values.
"""

import pytest
import jax
import jax.numpy as jnp
from biocomp.network import recipe_to_networks
from biocomp.library import LibraryContext
import biocomp.biorules as br
from biocomp.compute import ComputeStack
from biocomp.jaxutils import flat_concat
from biocomp.config import SIMPLE_NODES_COMPUTE_CONFIG
from biocomp.parameters import ParameterTree
from test_declarative_recipes import (
    lib,
    simple_single_reporter,
    simple_two_reporters,
    simple_single_ern,
)


def manual_simple_single_reporter(params: ParameterTree, X, random_vars: jnp.ndarray, key):
    """
    Fully manual computation for simple_single_reporter recipe.

    Network: hEF1a -> eBFP2 (single reporter)
    - Transcription uses hEF1a (only option)
    - Translation uses 00_empty_tc (no uORF, only option)

    Path: input → inv_translation → inv_transcription → inv_source → source → transcription → translation → output

    This computation is completely hardcoded - we manually:
    1. Look up the exact embeddings we know should be used
    2. Compute the noise that gets added to them
    3. Apply the dummy transform formulas
    """
    # Stack splits key by n_nodes for each layer
    layer_key = jax.random.split(key, 1)[0]
    _, _, k_quant = jax.random.split(layer_key, 3)

    # ========== Hardcoded embeddings ==========
    # Transcription: hEF1a is the ONLY promoter in this recipe, mask has only index 0 as True
    # From params, tc embeddings are at index 0 (hEF1a is the only one in DEFAULT_AVAILABLE_TC_RATES)
    qrate_tc = params["shared/quantization/values/tc_rate"][0, 0]  # hEF1a embedding

    # Translation: 00_empty_tc is the ONLY uORF option (eBFP2 has no uORF), mask has only index 0 as True
    # From params, tl embeddings - 00_empty_tc is at index 0 in DEFAULT_AVAILABLE_TL_RATES
    qrate_tl = params["shared/quantization/values/tl_rate"][0, 0]  # 00_empty_tc

    # ========== Layer 0: input ==========
    y0 = X
    assert y0.shape == (1,)
    print(f"Input: {y0}")

    # ========== Layer 1: inv_translation (dummy inverse) ==========
    # Dummy inverse: inner = mean([value, qrate, rv]) - (qrate + rv) / 3
    #                outer = mean([inner×8, rv_extra]) - mean / 9
    concatinput = flat_concat(y0, qrate_tl, random_vars[0])
    print(f"Concat input inv_translation: {concatinput}")
    inner_mean = jnp.mean(concatinput)
    print(f"Inner mean inv_translation: {inner_mean}")
    inner_val = inner_mean - (qrate_tl + random_vars[0]) / 3.0
    print(f"Inner val inv_translation: {inner_val}")
    outer_mean = (inner_val * 8 + random_vars[1]) / 9.0
    print(f"Outer mean inv_translation: {outer_mean}")
    y1 = outer_mean - outer_mean / 9.0  # subtract mean/len where len=9
    print(f"After inv_translation: {y1}")

    # ========== Layer 2: inv_transcription (dummy inverse) ==========
    inner_mean = jnp.mean(jnp.array([y1, qrate_tc, random_vars[2]]))
    inner_val = inner_mean - (qrate_tc + random_vars[2]) / 3.0
    outer_mean = (inner_val * 8 + random_vars[3]) / 9.0
    y2 = outer_mean - outer_mean / 9.0
    print(f"After inv_transcription: {y2}")

    # ========== Layer 3: inv_source (position 0) ==========
    # Divides by 0.9^0 = 1.0 (passthrough)
    y3 = y2

    # ========== Layer 4: source (position 0) ==========
    # Multiplies by 0.9^0 = 1.0 (passthrough)
    y4 = y3

    # ========== Layer 5: transcription (dummy forward) ==========
    # Dummy forward: inner = mean([value, qrate, rv])
    #                outer = mean([inner×8, rv_extra])
    # Uses random_vars[2] and [3] (shared with inverse)
    inner_mean = jnp.mean(jnp.array([y4, qrate_tc, random_vars[2]]))
    y5 = (inner_mean * 8 + random_vars[3]) / 9.0
    print(f"After transcription: {y5}")

    # ========== Layer 6: translation (dummy forward) ==========
    # Uses random_vars[0] and [1] (shared with inverse)
    inner_mean = jnp.mean(jnp.array([y5, qrate_tl, random_vars[0]]))
    y6 = (inner_mean * 8 + random_vars[1]) / 9.0
    print(f"After translation: {y6}")

    # ========== Layer 7: output ==========
    return y6


def jax_json_dumps(obj):
    import json

    def convert(o):
        if isinstance(o, jnp.ndarray):
            return o.tolist()
        raise TypeError

    return json.dumps(obj, default=convert, indent=2)


def manual_simple_two_reporters(params: ParameterTree, X, random_vars: jnp.ndarray, key):
    """
    Fully manual computation for simple_two_reporters recipe.

    Network: 2 reporters (eBFP2 and mMaroon1) aggregated with ratios [0.833, 0.167]
    - Both use hEF1a promoter (only option)
    - Both use 00_empty_tc for translation (no uORF, only option)

    Path for output 0 (eBFP2):
      input → inv_translation[0,2] → inv_transcription[0,2] → inv_source → inv_aggregation(÷0.833)
      → aggregation(×[0.833,0.167]) → [source0, source1]
      → [transcription0[4,5], transcription1[6,7]]
      → [translation0[0,1], translation1[2,3]]
      → output[0,1]
    """
    # Extract embeddings - same for both branches
    qrate_tc = params["shared/quantization/values/tc_rate"][0, 0]
    qrate_tl = params["shared/quantization/values/tl_rate"][0, 0]

    # Aggregation ratios
    ratios = params["local/5/aggregation2x/ratios"][0]  # [0.833, 0.167]

    # ========== Layer 0: input ==========
    y0 = X
    assert y0.shape == (1,)

    # ========== Layer 1: inv_translation (slot 0) ==========
    # Uses random_vars[0] and [1] (shared with translation node 0)
    # Dummy inverse: inner = mean([value, qrate, rv]) - (qrate + rv) / 3
    #                outer = mean([inner×8, rv_extra]) - mean / 9
    concatinput = flat_concat(y0, qrate_tl, random_vars[0])
    inner_mean = jnp.mean(concatinput)
    inner_val = inner_mean - (qrate_tl + random_vars[0]) / 3.0
    outer_mean = (inner_val * 8 + random_vars[1]) / 9.0
    y1 = outer_mean - outer_mean / 9.0

    # ========== Layer 2: inv_transcription (slot 0) ==========
    # Uses random_vars[4] and [5] (shared with transcription node 0)
    inner_mean = jnp.mean(jnp.array([y1, qrate_tc, random_vars[4]]))
    inner_val = inner_mean - (qrate_tc + random_vars[4]) / 3.0
    outer_mean = (inner_val * 8 + random_vars[5]) / 9.0
    y2 = outer_mean - outer_mean / 9.0

    # ========== Layer 3: inv_source (position 0) ==========
    # Divides by 0.9^0 = 1.0 (passthrough)
    y3 = y2

    # ========== Layer 4: inv_aggregation ==========
    # Divides by ratio for slot 0 (0.833)
    y4 = y3 / jnp.abs(ratios[0])

    # ========== Layer 5: aggregation ==========
    # Multiplies by [0.833, 0.167], creates 2 outputs
    y5_0 = jnp.abs(ratios[0]) * y4
    y5_1 = jnp.abs(ratios[1]) * y4

    # ========== Layer 6: source (2 nodes, both position 0) ==========
    # Both multiply by 0.9^0 = 1.0 (passthrough)
    y6_0 = y5_0
    y6_1 = y5_1

    # ========== Layer 7: transcription (2 nodes) ==========
    # Node 0 uses random_vars[4,5], Node 1 uses random_vars[6,7]
    # Dummy forward: inner = mean([value, qrate, rv])
    #                outer = mean([inner×8, rv_extra])

    # Transcription node 0
    inner_mean_0 = jnp.mean(jnp.array([y6_0, qrate_tc, random_vars[4]]))
    y7_0 = (inner_mean_0 * 8 + random_vars[5]) / 9.0

    # Transcription node 1
    inner_mean_1 = jnp.mean(jnp.array([y6_1, qrate_tc, random_vars[6]]))
    y7_1 = (inner_mean_1 * 8 + random_vars[7]) / 9.0

    # ========== Layer 8: translation (2 nodes) ==========
    # Node 0 uses random_vars[0,1], Node 1 uses random_vars[2,3]

    # Translation node 0
    inner_mean_0 = jnp.mean(jnp.array([y7_0, qrate_tl, random_vars[0]]))
    y8_0 = (inner_mean_0 * 8 + random_vars[1]) / 9.0

    # Translation node 1
    inner_mean_1 = jnp.mean(jnp.array([y7_1, qrate_tl, random_vars[2]]))
    y8_1 = (inner_mean_1 * 8 + random_vars[3]) / 9.0

    # ========== Layer 9: output ==========
    return jnp.array([y8_0, y8_1])


def test_simple_single_reporter_computation(lib, simple_single_reporter):
    """Test that compute stack matches manual computation for simple_single_reporter"""
    with LibraryContext.with_library(lib):
        # Build network and stack
        networks = recipe_to_networks(simple_single_reporter, br.ALL_RULES, invert=True)
        stack = ComputeStack([networks[0]])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        # Initialize parameters once
        init_key = jax.random.PRNGKey(42)
        params = stack.init(init_key)

        # ========== Verify quantization masks ==========
        # Transcription: should only allow hEF1a (index 0 in DEFAULT_AVAILABLE_TC_RATES)
        tc_mask = params["local/5/transcription/tc_rate_quantization_mask"][0]  # mask for node 0
        assert tc_mask.shape == (1, 1), f"TC mask shape should be (1, 1), got {tc_mask.shape}"
        assert tc_mask[0, 0], "hEF1a (index 0) should be available"

        # Translation: should only allow 00_empty_tc (index 0 in DEFAULT_AVAILABLE_TL_RATES)
        tl_mask = params["local/6/translation/tl_rate_quantization_mask"][0]  # mask for node 0
        assert tl_mask.shape == (1, 9), f"TL mask shape should be (1, 9), got {tl_mask.shape}"
        assert tl_mask[0, 0], "00_empty_tc (index 0) should be available"
        assert jnp.sum(tl_mask) == 1, (
            f"Only 1 uORF option should be available, got {jnp.sum(tl_mask)}"
        )

        base_key = jax.random.PRNGKey(123)
        test_keys = jax.random.split(base_key, 10)
        all_res = []
        for test_key in test_keys:
            params = stack.init(test_key)

            inputs = jnp.ones((1,))
            n_random_vars = params["global/number_of_random_variables"]
            random_vars = jax.random.normal(test_key, (n_random_vars,))

            stack_result, aux = stack.apply(params, inputs, random_vars, test_key)
            manual_result = manual_simple_single_reporter(params, inputs, random_vars, test_key)

            print(f"Stack result: {stack_result}, Manual result: {manual_result}")
            print(f"Aux:\n{jax_json_dumps(aux)}")

            print(f"Parameters used:\n{params}")

            assert jnp.allclose(stack_result, manual_result, rtol=1e-5), (
                f"Stack output {stack_result} != manual output {manual_result}"
            )

            all_res.append(stack_result)

        std_dev = jnp.std(jnp.array(all_res))
        assert std_dev > 1e-5, "All results should be different with different random keys"


def test_simple_two_reporters_computation(lib, simple_two_reporters):
    """Test that compute stack matches manual computation for simple_two_reporters"""
    with LibraryContext.with_library(lib):
        # Build network and stack
        networks = recipe_to_networks(simple_two_reporters, br.ALL_RULES, invert=True)
        stack = ComputeStack([networks[0]])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        # Initialize parameters once
        init_key = jax.random.PRNGKey(42)
        params = stack.init(init_key)

        # ========== Verify quantization masks ==========
        # Transcription: should only allow hEF1a (index 0)
        tc_mask = params["local/7/transcription/tc_rate_quantization_mask"]
        assert tc_mask.shape == (2, 1, 1), f"TC mask shape should be (2, 1, 1), got {tc_mask.shape}"
        assert tc_mask[0, 0, 0] and tc_mask[1, 0, 0], "hEF1a should be available for both nodes"

        # Translation: should only allow 00_empty_tc (index 0)
        tl_mask = params["local/8/translation/tl_rate_quantization_mask"]
        assert tl_mask.shape == (2, 1, 9), f"TL mask shape should be (2, 1, 9), got {tl_mask.shape}"
        assert tl_mask[0, 0, 0] and tl_mask[1, 0, 0], (
            "00_empty_tc should be available for both nodes"
        )
        assert jnp.sum(tl_mask[0]) == 1 and jnp.sum(tl_mask[1]) == 1, "Only 1 uORF per node"

        # Aggregation ratios should match recipe
        ratios = params["local/5/aggregation2x/ratios"][0]
        assert jnp.allclose(ratios, jnp.array([0.833, 0.167])), (
            f"Ratios should be [0.833, 0.167], got {ratios}"
        )

        base_key = jax.random.PRNGKey(123)
        test_keys = jax.random.split(base_key, 10)
        all_res = []
        for test_key in test_keys:
            params = stack.init(test_key)

            inputs = jnp.ones((1,))
            n_random_vars = params["global/number_of_random_variables"]
            random_vars = jax.random.normal(test_key, (n_random_vars,))

            stack_result, aux = stack.apply(params, inputs, random_vars, test_key)
            manual_result = manual_simple_two_reporters(params, inputs, random_vars, test_key)

            print(f"\nStack result: {stack_result}")
            print(f"Manual result: {manual_result}")
            print(f"Difference: {stack_result - manual_result}")

            assert jnp.allclose(stack_result, manual_result, rtol=1e-5), (
                f"Stack output {stack_result} != manual output {manual_result}"
            )

            all_res.append(stack_result)

        all_res = jnp.array(all_res)
        std_dev = jnp.std(all_res, axis=0)
        assert jnp.all(std_dev > 1e-5), "All results should be different with different random keys"


def manual_simple_single_ern(params: ParameterTree, X, random_vars: jnp.ndarray, key):
    """
    Fully manual computation for simple_single_ern recipe.

    Network: ERN (CasE_target + CasE_source) + mNeonGreen reporter, 3-way aggregation
    - All use hEF1a promoter (only option)
    - All use 00_empty_tc for translation (no uORF, only option)
    - Aggregation ratios: [1/3, 1/3, 1/3]

    This test focuses on the INVERTIBLE path through mNeonGreen (output slot 1).
    The ERN path (output slot 0) is NOT tested since ERN nodes are non-invertible.

    Invertible path:
      input → inv_translation → inv_transcription → inv_source → inv_aggregation(÷1/3, slot 2)
      → aggregation(×[1/3,1/3,1/3]) → source2 → transcription2 → translation2 → output[1]
    """
    # Extract embeddings
    qrate_tc = params["shared/quantization/values/tc_rate"][0, 0]
    qrate_tl = params["shared/quantization/values/tl_rate"][0, 0]

    # Aggregation ratios (3 branches, all 1/3)
    ratios = params["local/5/aggregation3x/ratios"][0]  # [1/3, 1/3, 1/3]

    # ========== Layer 0: input ==========
    y0 = X
    assert y0.shape == (1,)

    # ========== Layer 1: inv_translation ==========
    # Shares random_vars with translation node 1 (layer 8): [5,6]
    concatinput = flat_concat(y0, qrate_tl, random_vars[5])
    inner_mean = jnp.mean(concatinput)
    inner_val = inner_mean - (qrate_tl + random_vars[5]) / 3.0
    outer_mean = (inner_val * 8 + random_vars[6]) / 9.0
    y1 = outer_mean - outer_mean / 9.0

    # ========== Layer 2: inv_transcription ==========
    # Shares random_vars with transcription node 2 (layer 7): [11,12]
    inner_mean = jnp.mean(jnp.array([y1, qrate_tc, random_vars[11]]))
    inner_val = inner_mean - (qrate_tc + random_vars[11]) / 3.0
    outer_mean = (inner_val * 8 + random_vars[12]) / 9.0
    y2 = outer_mean - outer_mean / 9.0

    # ========== Layer 3: inv_source (position 0) ==========
    y3 = y2  # passthrough (÷1.0)

    # ========== Layer 4: inv_aggregation (slot 2) ==========
    y4 = y3 / jnp.abs(ratios[2])

    # ========== Layer 5: aggregation ==========
    # Creates 3 outputs, we follow slot 2
    y5_2 = jnp.abs(ratios[2]) * y4

    # ========== Layer 6: source 2 (position 0) ==========
    y6_2 = y5_2  # passthrough (×1.0)

    # ========== Layer 7: transcription 2 ==========
    # Node 2 uses random_vars[11,12]
    inner_mean = jnp.mean(jnp.array([y6_2, qrate_tc, random_vars[11]]))
    y7_2 = (inner_mean * 8 + random_vars[12]) / 9.0

    # ========== Layer 8: translation (node 1) ==========
    # Node 1 uses random_vars[5,6]
    inner_mean = jnp.mean(jnp.array([y7_2, qrate_tl, random_vars[5]]))
    y8_1 = (inner_mean * 8 + random_vars[6]) / 9.0

    # ========== Layer 11: output (slot 1 = mNeonGreen) ==========
    return y8_1


def test_simple_single_ern_computation(lib, simple_single_ern):
    """Test that compute stack matches manual computation for simple_single_ern (mNeonGreen path)"""
    with LibraryContext.with_library(lib):
        # Build network and stack
        networks = recipe_to_networks(simple_single_ern, br.ALL_RULES, lib=lib, invert=True)
        stack = ComputeStack([networks[0]])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        # Initialize parameters once
        init_key = jax.random.PRNGKey(42)
        params = stack.init(init_key)

        # ========== Verify quantization masks ==========
        # Transcription: 3 nodes, all should only allow hEF1a (index 0)
        tc_mask = params["local/7/transcription/tc_rate_quantization_mask"]
        assert tc_mask.shape == (3, 1, 1), f"TC mask shape should be (3, 1, 1), got {tc_mask.shape}"
        assert all(tc_mask[i, 0, 0] for i in range(3)), "hEF1a should be available for all nodes"

        # Translation: 2 nodes (mNeonGreen + ERN target), should only allow 00_empty_tc (index 0)
        tl_mask = params["local/8/translation/tl_rate_quantization_mask"]
        assert tl_mask.shape == (2, 1, 9), f"TL mask shape should be (2, 1, 9), got {tl_mask.shape}"
        assert tl_mask[0, 0, 0] and tl_mask[1, 0, 0], (
            "00_empty_tc should be available for both nodes"
        )
        assert jnp.sum(tl_mask[0]) == 1 and jnp.sum(tl_mask[1]) == 1, "Only 1 uORF per node"

        # Aggregation ratios should be [1/3, 1/3, 1/3]
        ratios = params["local/5/aggregation3x/ratios"][0]
        expected_ratio = 1.0 / 3.0
        assert jnp.allclose(ratios, jnp.full(3, expected_ratio)), (
            f"Ratios should be [1/3, 1/3, 1/3], got {ratios}"
        )

        base_key = jax.random.PRNGKey(123)
        test_keys = jax.random.split(base_key, 10)
        all_res = []
        for test_key in test_keys:
            params = stack.init(test_key)

            inputs = jnp.ones((1,))
            n_random_vars = params["global/number_of_random_variables"]
            random_vars = jax.random.normal(test_key, (n_random_vars,))

            stack_result, aux = stack.apply(params, inputs, random_vars, test_key)
            manual_result = manual_simple_single_ern(params, inputs, random_vars, test_key)

            # Stack returns 2 outputs [eBFP2_through_ERN, mNeonGreen]
            # We only test output slot 1 (mNeonGreen, the invertible path)
            stack_mNeonGreen = stack_result[1]

            print(f"\nStack mNeonGreen: {stack_mNeonGreen}")
            print(f"Manual mNeonGreen: {manual_result}")
            print(f"Difference: {stack_mNeonGreen - manual_result}")

            assert jnp.allclose(stack_mNeonGreen, manual_result, rtol=1e-5), (
                f"Stack mNeonGreen output {stack_mNeonGreen} != manual output {manual_result}"
            )

            all_res.append(stack_mNeonGreen)

        all_res = jnp.array(all_res)
        std_dev = jnp.std(all_res)
        assert std_dev > 1e-5, "All results should be different with different random keys"
