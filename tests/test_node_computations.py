# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Test compute stack with simplified nodes

This test file validates that the compute stack correctly executes simple node functions
by comparing stack outputs against manually computed expected values.
"""

import jax
import jax.numpy as jnp
from biocomp.network import recipe_to_networks
from biocomp.library import LibraryContext
import biocomp.biorules as br
from biocomp.compute import ComputeStack
from biocomp.config import SIMPLE_NODES_COMPUTE_CONFIG
from biocomp.parameters import ParameterTree
from biocomp.context import CONTEXT_EMBEDDINGS, _codebook_means_path, _codebook_logstdevs_path, _indices_path

# Import test fixtures from centralized module
pytest_plugins = ["test_declarative_recipes"]


def disable_context_params(params: ParameterTree):
    """Remove context codebook so resolve_context_vector returns None.

    This keeps manual dummy_mlp computations unchanged (no context concat path).
    """
    for ce in CONTEXT_EMBEDDINGS:
        for path in [_codebook_means_path(ce.name), _codebook_logstdevs_path(ce.name), _indices_path(ce.name)]:
            if path in params:
                del params.data[path]


def manual_simple_single_reporter(params: ParameterTree, X, random_vars: jnp.ndarray, key):
    """
    Fully manual computation for simple_single_reporter recipe.

    Network: hEF1a -> eBFP2 (single reporter)
    - Transcription uses hEF1a (only option)
    - Translation uses 00_empty_tc (no uORF, only option)

    Path: input -> inv_output -> inv_translation -> inv_transcription -> inv_source -> source -> transcription -> translation -> output

    This computation is completely hardcoded - we manually:
    1. Look up the exact embeddings we know should be used
    2. Compute the noise that gets added to them
    3. Apply the dummy transform formulas
    """
    # Stack splits key by n_nodes for each layer
    layer_key = jax.random.split(key, 1)[0]
    _, _, k_quant = jax.random.split(layer_key, 3)

    qrate_tc = params["shared/quantization/values/tc_rate"][0, 0]
    qrate_tl = params["shared/quantization/values/tl_rate"][0, 0]

    # ========== Layer 0: input ==========
    y0 = X
    assert y0.shape == (1,)
    print(f"Input: {y0}")

    # ========== Layer 1: inv_output (dummy + random_var) ==========
    # dummy_mlp(flat_concat(value, random_var)) = sum of all elements
    y1 = y0[0] + random_vars[0]
    print(f"After inv_output: {y1}")

    # ========== Layer 2: inv_translation (dummy inverse) ==========
    # Corrected inverse: (input - rv_outer) / 8 - qrate - rv_inner
    # Random var indices shifted +1 due to output node claiming ID 0
    y2 = (y1 - random_vars[2]) / 8 - qrate_tl - random_vars[1]
    print(f"After inv_translation: {y2}")

    # ========== Layer 3: inv_transcription (dummy inverse) ==========
    # Corrected inverse: (input - rv_outer) / 8 - qrate - rv_inner
    y3 = (y2 - random_vars[4]) / 8 - qrate_tc - random_vars[3]
    print(f"After inv_transcription: {y3}")

    # ========== Layer 4: inv_source (position 0) ==========
    # Divides by 0.9^0 = 1.0 (passthrough)
    y4 = y3

    # ========== Layer 5: source (position 0) ==========
    # Multiplies by 0.9^0 = 1.0 (passthrough)
    y5 = y4

    # ========== Layer 6: transcription (dummy forward) ==========
    # Dummy forward: inner = sum([value, qrate, rv])
    #                outer = sum([inner×8, rv_extra])
    # Uses random_vars[3] and [4] (shared with inverse, shifted +1)
    inner_sum = jnp.sum(jnp.array([y5, qrate_tc, random_vars[3]]))
    y6 = inner_sum * 8 + random_vars[4]
    print(f"After transcription: {y6}")

    # ========== Layer 7: translation (dummy forward) ==========
    # Uses random_vars[1] and [2] (shared with inverse, shifted +1)
    inner_sum = jnp.sum(jnp.array([y6, qrate_tl, random_vars[1]]))
    y7 = inner_sum * 8 + random_vars[2]
    print(f"After translation: {y7}")

    # ========== Layer 8: output (dummy + random_var) ==========
    # dummy_mlp(flat_concat(y7, random_vars[0])) = y7 + random_vars[0]
    return y7 + random_vars[0]


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
      input -> inv_output -> inv_translation[0,2] -> inv_transcription[0,2] -> inv_source -> inv_aggregation(÷0.833)
      -> aggregation(×[0.833,0.167]) -> [source0, source1]
      -> [transcription0[4,5], transcription1[6,7]]
      -> [translation0[0,1], translation1[2,3]]
      -> output[0,1]
    """
    # Extract embeddings - same for both branches
    qrate_tc = params["shared/quantization/values/tc_rate"][0, 0]
    qrate_tl = params["shared/quantization/values/tl_rate"][0, 0]

    # Aggregation ratios
    ratios = params["local/6/aggregation2x/ratios"][0]  # [0.833, 0.167]

    # ========== Layer 0: input ==========
    y0 = X
    assert y0.shape == (1,)

    # ========== Layer 1: inv_output (dummy + random_var, slot 0) ==========
    # dummy_mlp(flat_concat(value, random_var)) = sum of all elements
    y1 = y0[0] + random_vars[0]

    # ========== Layer 2: inv_translation (slot 0) ==========
    # Uses random_vars[2] and [3] (shared with translation node 0, shifted +2)
    # Corrected inverse: (input - rv_outer) / 8 - qrate - rv_inner
    y2 = (y1 - random_vars[3]) / 8 - qrate_tl - random_vars[2]

    # ========== Layer 3: inv_transcription (slot 0) ==========
    # Uses random_vars[6] and [7] (shared with transcription node 0, shifted +2)
    # Corrected inverse: (input - rv_outer) / 8 - qrate - rv_inner
    y3 = (y2 - random_vars[7]) / 8 - qrate_tc - random_vars[6]

    # ========== Layer 4: inv_source (position 0) ==========
    # Divides by 0.9^0 = 1.0 (passthrough)
    y4 = y3

    # ========== Layer 5: inv_aggregation ==========
    # Divides by ratio for slot 0 (0.833)
    y5 = y4 / jnp.abs(ratios[0])

    # ========== Layer 6: aggregation ==========
    # Multiplies by [0.833, 0.167], creates 2 outputs
    y6_0 = jnp.abs(ratios[0]) * y5
    y6_1 = jnp.abs(ratios[1]) * y5

    # ========== Layer 7: source (2 nodes, both position 0) ==========
    # Both multiply by 0.9^0 = 1.0 (passthrough)
    y7_0 = y6_0
    y7_1 = y6_1

    # ========== Layer 8: transcription (2 nodes) ==========
    # Node 0 uses random_vars[6,7], Node 1 uses random_vars[8,9] (shifted +2)
    # Dummy forward: inner = sum([value, qrate, rv])
    #                outer = sum([inner×8, rv_extra])

    # Transcription node 0
    inner_sum_0 = jnp.sum(jnp.array([y7_0, qrate_tc, random_vars[6]]))
    y8_0 = inner_sum_0 * 8 + random_vars[7]

    # Transcription node 1
    inner_sum_1 = jnp.sum(jnp.array([y7_1, qrate_tc, random_vars[8]]))
    y8_1 = inner_sum_1 * 8 + random_vars[9]

    # ========== Layer 9: translation (2 nodes) ==========
    # Node 0 uses random_vars[2,3], Node 1 uses random_vars[4,5] (shifted +2)

    # Translation node 0
    inner_sum_0 = jnp.sum(jnp.array([y8_0, qrate_tl, random_vars[2]]))
    y9_0 = inner_sum_0 * 8 + random_vars[3]

    # Translation node 1
    inner_sum_1 = jnp.sum(jnp.array([y8_1, qrate_tl, random_vars[4]]))
    y9_1 = inner_sum_1 * 8 + random_vars[5]

    # ========== Layer 10: output (dummy + random_var) ==========
    # Head 0 uses random_vars[0], Head 1 uses random_vars[1]
    return jnp.array([y9_0 + random_vars[0], y9_1 + random_vars[1]])


def find_layer_idx(stack, layer_type):
    """Find the index of a layer by its type"""
    for i, layer in enumerate(stack.layers):
        if layer.f_type == layer_type:
            return i
    raise ValueError(f"Layer type {layer_type} not found in stack")


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
        disable_context_params(params)

        # Find layer indices dynamically (they may change as graph building evolves)
        tc_layer_idx = find_layer_idx(stack, "transcription")
        tl_layer_idx = find_layer_idx(stack, "translation")

        # ========== Verify quantization masks ==========
        # Transcription: should only allow hEF1a (index 0 in DEFAULT_AVAILABLE_TC_RATES)
        tc_mask = params[f"local/{tc_layer_idx}/transcription/tc_rate_quantization_mask"][0]
        assert tc_mask.shape == (1, 1), f"TC mask shape should be (1, 1), got {tc_mask.shape}"
        assert tc_mask[0, 0], "hEF1a (index 0) should be available"

        # Translation: should only allow 00_empty_tc (index 0 in DEFAULT_AVAILABLE_TL_RATES)
        tl_mask = params[f"local/{tl_layer_idx}/translation/tl_rate_quantization_mask"][0]
        assert tl_mask.shape == (1, 13), f"TL mask shape should be (1, 13), got {tl_mask.shape}"
        assert tl_mask[0, 0], "00_empty_tc (index 0) should be available"
        assert jnp.sum(tl_mask) == 1, (
            f"Only 1 uORF option should be available, got {jnp.sum(tl_mask)}"
        )

        base_key = jax.random.PRNGKey(123)
        test_keys = jax.random.split(base_key, 10)
        all_res = []
        for test_key in test_keys:
            params = stack.init(test_key)
            disable_context_params(params)

            inputs = jnp.ones((1,))
            n_random_vars = params["global/number_of_random_variables"]
            random_vars = jax.random.normal(test_key, (n_random_vars,))

            stack_result, aux = stack.apply(params, inputs, random_vars, test_key)
            manual_result = manual_simple_single_reporter(params, inputs, random_vars, test_key)

            print(f"Stack result: {stack_result}, Manual result: {manual_result}")
            print(f"Aux:\n{jax_json_dumps(aux)}")

            print(f"Parameters used:\n{params}")

            # Use slightly relaxed tolerance due to floating point precision with corrected inverse
            assert jnp.allclose(stack_result, manual_result, rtol=2e-5), (
                f"Stack output {stack_result} != manual output {manual_result}"
            )

            all_res.append(stack_result)

        std_dev = jnp.std(jnp.array(all_res))
        # Use relaxed threshold - variability depends on random keys and network structure
        assert std_dev > 1e-6, "All results should be different with different random keys"


def find_layer_idx_containing(stack, type_prefix):
    """Find the index of a layer whose type starts with type_prefix"""
    for i, layer in enumerate(stack.layers):
        if layer.f_type.startswith(type_prefix):
            return i
    raise ValueError(f"Layer type starting with {type_prefix} not found in stack")


def find_aggregation_param_key(params, layer_idx):
    """Find the param key for aggregation layer (e.g., aggregation2x, aggregation3x)"""
    layer_params = params[f"local/{layer_idx}"]
    for key in layer_params.value.keys():
        if key.startswith("aggregation"):
            return key
    raise ValueError(f"No aggregation key found in layer {layer_idx}")


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
        disable_context_params(params)

        # Find layer indices dynamically
        tc_layer_idx = find_layer_idx(stack, "transcription")
        tl_layer_idx = find_layer_idx(stack, "translation")
        agg_layer_idx = find_layer_idx_containing(stack, "aggregation")

        # ========== Verify quantization masks ==========
        # Transcription: should only allow hEF1a (index 0)
        tc_mask = params[f"local/{tc_layer_idx}/transcription/tc_rate_quantization_mask"]
        assert tc_mask.shape == (2, 1, 1), f"TC mask shape should be (2, 1, 1), got {tc_mask.shape}"
        assert tc_mask[0, 0, 0] and tc_mask[1, 0, 0], "hEF1a should be available for both nodes"

        # Translation: should only allow 00_empty_tc (index 0)
        tl_mask = params[f"local/{tl_layer_idx}/translation/tl_rate_quantization_mask"]
        assert tl_mask.shape == (2, 1, 13), (
            f"TL mask shape should be (2, 1, 13), got {tl_mask.shape}"
        )
        assert tl_mask[0, 0, 0] and tl_mask[1, 0, 0], (
            "00_empty_tc should be available for both nodes"
        )
        assert jnp.sum(tl_mask[0]) == 1 and jnp.sum(tl_mask[1]) == 1, "Only 1 uORF per node"

        # Aggregation ratios should match recipe - find param key dynamically
        agg_param_key = find_aggregation_param_key(params, agg_layer_idx)
        ratios = params[f"local/{agg_layer_idx}/{agg_param_key}/ratios"][0]
        assert jnp.allclose(ratios, jnp.array([0.833, 0.167])), (
            f"Ratios should be [0.833, 0.167], got {ratios}"
        )

        base_key = jax.random.PRNGKey(123)
        test_keys = jax.random.split(base_key, 10)
        all_res = []
        for test_key in test_keys:
            params = stack.init(test_key)
            disable_context_params(params)

            inputs = jnp.ones((1,))
            n_random_vars = params["global/number_of_random_variables"]
            random_vars = jax.random.normal(test_key, (n_random_vars,))

            stack_result, aux = stack.apply(params, inputs, random_vars, test_key)
            manual_result = manual_simple_two_reporters(params, inputs, random_vars, test_key)

            print(f"\nStack result: {stack_result}")
            print(f"Manual result: {manual_result}")
            print(f"Difference: {stack_result - manual_result}")

            # Relaxed tolerance: longer chain (output random_var addition) accumulates more float32 error
            assert jnp.allclose(stack_result, manual_result, rtol=2e-4), (
                f"Stack output {stack_result} != manual output {manual_result}"
            )

            all_res.append(stack_result)

        all_res = jnp.array(all_res)
        std_dev = jnp.std(all_res, axis=0)
        # Use relaxed threshold - variability depends on random keys and network structure
        assert jnp.all(std_dev > 1e-6), "All results should be different with different random keys"


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
      input -> inv_output -> inv_translation -> inv_transcription -> inv_source -> inv_aggregation(÷1/3, slot 2)
      -> aggregation(×[1/3,1/3,1/3]) -> source2 -> transcription2 -> translation2 -> output[1]
    """
    # Extract embeddings
    qrate_tc = params["shared/quantization/values/tc_rate"][0, 0]
    qrate_tl = params["shared/quantization/values/tl_rate"][0, 0]

    # Aggregation ratios (3 branches, all 1/3)
    ratios = params["local/6/aggregation3x/ratios"][0]  # [1/3, 1/3, 1/3]

    # ========== Layer 0: input ==========
    y0 = X
    assert y0.shape == (1,)

    # ========== Layer 1: inv_output (dummy + random_var, slot 1 = mNeonGreen) ==========
    # dummy_mlp(flat_concat(value, random_var)) = sum of all elements
    y1 = y0[0] + random_vars[1]

    # ========== Layer 2: inv_translation ==========
    # Shares random_vars with translation node 1 (layer 9): [7,8] (shifted +2)
    # Corrected inverse: (input - rv_outer) / 8 - qrate - rv_inner
    y2 = (y1 - random_vars[8]) / 8 - qrate_tl - random_vars[7]

    # ========== Layer 3: inv_transcription ==========
    # Shares random_vars with transcription node 2 (layer 8): [13,14] (shifted +2)
    # Corrected inverse: (input - rv_outer) / 8 - qrate - rv_inner
    y3 = (y2 - random_vars[14]) / 8 - qrate_tc - random_vars[13]

    # ========== Layer 4: inv_source (position 0) ==========
    y4 = y3  # passthrough (÷1.0)

    # ========== Layer 5: inv_aggregation (slot 2) ==========
    y5 = y4 / jnp.abs(ratios[2])

    # ========== Layer 6: aggregation ==========
    # Creates 3 outputs, we follow slot 2
    y6_2 = jnp.abs(ratios[2]) * y5

    # ========== Layer 7: source 2 (position 0) ==========
    y7_2 = y6_2  # passthrough (×1.0)

    # ========== Layer 8: transcription 2 ==========
    # Node 2 uses random_vars[13,14] (shifted +2)
    inner_sum = jnp.sum(jnp.array([y7_2, qrate_tc, random_vars[13]]))
    y8_2 = inner_sum * 8 + random_vars[14]

    # ========== Layer 9: translation (node 1) ==========
    # Node 1 uses random_vars[7,8] (shifted +2)
    inner_sum = jnp.sum(jnp.array([y8_2, qrate_tl, random_vars[7]]))
    y9_1 = inner_sum * 8 + random_vars[8]

    # ========== Layer 12: output (dummy + random_var, slot 1 = mNeonGreen) ==========
    # Head 1 uses random_vars[1]
    return y9_1 + random_vars[1]


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
        disable_context_params(params)

        # Find layer indices dynamically
        tc_layer_idx = find_layer_idx(stack, "transcription")
        tl_layer_idx = find_layer_idx(stack, "translation")
        agg_layer_idx = find_layer_idx_containing(stack, "aggregation")

        # ========== Verify quantization masks ==========
        # Transcription: 3 nodes, all should only allow hEF1a (index 0)
        tc_mask = params[f"local/{tc_layer_idx}/transcription/tc_rate_quantization_mask"]
        assert tc_mask.shape == (3, 1, 1), f"TC mask shape should be (3, 1, 1), got {tc_mask.shape}"
        assert all(tc_mask[i, 0, 0] for i in range(3)), "hEF1a should be available for all nodes"

        # Translation: 2 nodes (mNeonGreen + ERN target), should only allow 00_empty_tc (index 0)
        tl_mask = params[f"local/{tl_layer_idx}/translation/tl_rate_quantization_mask"]
        assert tl_mask.shape == (2, 1, 13), (
            f"TL mask shape should be (2, 1, 13), got {tl_mask.shape}"
        )
        assert tl_mask[0, 0, 0] and tl_mask[1, 0, 0], (
            "00_empty_tc should be available for both nodes"
        )
        assert jnp.sum(tl_mask[0]) == 1 and jnp.sum(tl_mask[1]) == 1, "Only 1 uORF per node"

        # Aggregation ratios should be [1/3, 1/3, 1/3]
        agg_param_key = find_aggregation_param_key(params, agg_layer_idx)
        ratios = params[f"local/{agg_layer_idx}/{agg_param_key}/ratios"][0]
        expected_ratio = 1.0 / 3.0
        assert jnp.allclose(ratios, jnp.full(3, expected_ratio)), (
            f"Ratios should be [1/3, 1/3, 1/3], got {ratios}"
        )

        base_key = jax.random.PRNGKey(123)
        test_keys = jax.random.split(base_key, 10)
        all_res = []
        for test_key in test_keys:
            params = stack.init(test_key)
            disable_context_params(params)

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

            # Relaxed tolerance: longer chain (output random_var addition) accumulates more float32 error
            assert jnp.allclose(stack_mNeonGreen, manual_result, rtol=2e-4), (
                f"Stack mNeonGreen output {stack_mNeonGreen} != manual output {manual_result}"
            )

            all_res.append(stack_mNeonGreen)

        all_res = jnp.array(all_res)
        std_dev = jnp.std(all_res)
        # Use relaxed threshold - variability depends on random keys and network structure
        assert std_dev > 1e-6, "All results should be different with different random keys"

