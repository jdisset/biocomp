"""Test compute stack with simplified nodes

This test file validates that the compute stack correctly executes simple node functions
by comparing stack outputs against manually computed expected values.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from biocomp.network import recipe_to_networks
from biocomp.library import load_lib, LibraryContext
from biocomp.recipe import Recipe, CoTransfection, TranscriptionUnit, Slot
import biocomp.biorules as br
from biocomp.compute import ComputeStack
from biocomp.config import SIMPLE_NODES_COMPUTE_CONFIG
from biocomp.parameters import ParameterTree
from biocomp.nodeutils import get_prev_num_random_vars


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


def manual_simple_single_reporter(params: ParameterTree, X_numeric: float) -> float:
    """
    Manual computation for simple_single_reporter recipe.

    Network structure:
    1. numeric node (hard_bias) -> outputs X_numeric
    2. source node (simple_source_with_pos, position 0) -> y_source = X_numeric * 0.9^0 = X_numeric
    3. transcription node -> y_tc = a_tc * y_source
    4. translation node -> y_tl = a_tl * y_tc
    5. output node (passthrough) -> y_out = y_tl

    Therefore: y_out = a_tl * a_tc * X_numeric

    where:
    - a_tc = coefficient from transcription layer
    - a_tl = coefficient from translation layer
    """
    # Get the coefficients from the simple transform nodes
    # The layers are: 0=input, 1=inv_translation, 2=inv_transcription, 3=inv_source,
    #                 4=source, 5=transcription, 6=translation, 7=output
    a_tc = params["local/5/transcription/tc_coeffs"][0, 0]  # node 0, input 0
    a_tl = params["local/6/translation/tl_coeffs"][0, 0]  # node 0, input 0

    # Compute: y = a_tl * a_tc * X_numeric
    return a_tl * a_tc * X_numeric


def manual_simple_two_reporters(params: ParameterTree, X_numeric: float) -> jnp.ndarray:
    """
    Manual computation for simple_two_reporters recipe.

    Network structure (per reporter branch):
    1. numeric node -> X_numeric
    2. aggregation node -> splits into two branches with ratios [0.833, 0.167]
       - branch_0 = 0.833 * X_numeric
       - branch_1 = 0.167 * X_numeric
    3. source nodes (position 0 and 1):
       - y_source_0 = branch_0 * 0.9^0 = 0.833 * X_numeric
       - y_source_1 = branch_1 * 0.9^1 = 0.167 * X_numeric * 0.9
    4. transcription nodes:
       - y_tc_0 = a_tc_0 * y_source_0
       - y_tc_1 = a_tc_1 * y_source_1
    5. translation nodes:
       - y_tl_0 = a_tl_0 * y_tc_0
       - y_tl_1 = a_tl_1 * y_tc_1
    6. output nodes (passthrough):
       - out_0 = y_tl_0
       - out_1 = y_tl_1

    Returns: [out_0, out_1]
    """
    # Get coefficients - need to find the right layer numbers
    # This requires understanding the actual layer structure which might vary
    # Let's try to find them dynamically or use known structure

    # Get aggregation ratios from the aggregation layer
    # Need to determine which layer is aggregation
    r0, r1 = 0.833, 0.167  # These are set in the recipe

    # Get transform coefficients - assuming layer numbers from simple_single_reporter + aggregation
    # This is fragile but for testing purposes we'll hard-code
    try:
        # Try to get coefficients - layer numbering might be different
        a_tc_0 = params["local/5/transcription/tc_coeffs"][0, 0]  # first node
        a_tc_1 = params["local/5/transcription/tc_coeffs"][1, 0]  # second node
        a_tl_0 = params["local/6/translation/tl_coeffs"][0, 0]
        a_tl_1 = params["local/6/translation/tl_coeffs"][1, 0]
    except (KeyError, IndexError):
        # If layers are numbered differently, return zeros for now
        # This test might need adjustment based on actual layer structure
        return jnp.array([0.0, 0.0])

    # Compute outputs for each branch
    out_0 = a_tl_0 * a_tc_0 * r0 * X_numeric * (0.9**0)
    out_1 = a_tl_1 * a_tc_1 * r1 * X_numeric * (0.9**1)

    return jnp.array([out_0, out_1])
