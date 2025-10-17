"""Test compute stack with simplified nodes

This test file validates that the compute stack correctly executes simple node functions
by comparing stack outputs against manually computed expected values.
"""

import pytest
import jax
import jax.numpy as jnp
from biocomp.network import recipe_to_networks
from biocomp.library import load_lib, LibraryContext
from biocomp.recipe import Recipe, CoTransfection, TranscriptionUnit, Slot
import biocomp.biorules as br
from biocomp.compute import ComputeStack
from biocomp.jaxutils import flat_concat
from biocomp.config import SIMPLE_NODES_COMPUTE_CONFIG
from biocomp.parameters import ParameterTree
import biocomp.quantization as qz


@pytest.fixture
def lib():
    """Load the parts library"""
    return load_lib()


@pytest.fixture
def simple_single_reporter(lib):
    """Simplest network: 1 TU with just a reporter"""
    with LibraryContext.with_library(lib):
        return Recipe(
            name="simple_single_reporter",
            content=[
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            name="EBFP2_reporter",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="eBFP2"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                    ],
                )
            ],
        )


@pytest.fixture
def simple_two_reporters(lib):
    """Simple network: 2 TUs (two reporters aggregated)"""
    with LibraryContext.with_library(lib):
        return Recipe(
            name="simple_two_reporters",
            content=[
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            name="EBFP2_reporter",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="eBFP2"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                        TranscriptionUnit(
                            name="mMaroon1_reporter",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="mMaroon1"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                    ],
                    ratios=[0.833, 0.167],
                )
            ],
        )


@pytest.fixture
def simple_single_ern(lib):
    """Simple ERN network: ERN + target"""
    with LibraryContext.with_library(lib):
        return Recipe(
            name="simple_single_ern",
            content=[
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            name="CasE_target",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="CasE_rec"),
                                Slot(part="eBFP2"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                        TranscriptionUnit(
                            name="CasE_source",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="CasE"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                    ],
                )
            ],
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
    import jax.random as random

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


def test_simple_single_reporter_computation(lib, simple_single_reporter):
    """Test that compute stack matches manual computation for simple_single_reporter"""
    with LibraryContext.with_library(lib):
        # Build network and stack
        networks = recipe_to_networks(simple_single_reporter, br.ALL_RULES, lib)
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
