"""Test fixtures for declarative network recipes

This module provides centralized test network fixtures used throughout the test suite.
All networks are manually constructed using the declarative API to ensure we have
complete control and understanding of the network structure.
"""

import pytest
from biocomp.network_new import Network
from biocomp.recipe_new import CoTransfection, TranscriptionUnit, Slot
from biocomp.library import load_lib, LibraryContext


@pytest.fixture
def lib():
    """Load the parts library"""
    return load_lib()


# ============================================================================
# FIXTURE: Simple Networks
# ============================================================================

@pytest.fixture
def simple_single_reporter(lib):
    """Simplest possible network: 1 cotx, 1 plasmid, 1 TU (just a reporter)"""
    with LibraryContext.with_library(lib):
        return Network(
            name="simple_single_reporter",
            cotx=[
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
    """Simple network: 1 cotx, 2 plasmids, 2 TUs (two reporters aggregated)

    This corresponds to NoiseFloor_1_5 recipe:
    - EBFP2 reporter (ratio 0.833)
    - mMaroon1 reporter (ratio 0.167)
    """
    with LibraryContext.with_library(lib):
        return Network(
            name="simple_two_reporters",
            cotx=[
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


# ============================================================================
# FIXTURE: ERN Networks (Enzymatic Reaction Network)
# ============================================================================

@pytest.fixture
def simple_single_ern(lib):
    """Simple ERN network: 1 cotx, 1 plasmid, 2 TUs (ERN + target)

    This is the minimal ERN network:
    - One reporter with ERN recognition site (CasE_rec)
    - One ERN source (CasE enzyme)
    """
    with LibraryContext.with_library(lib):
        return Network(
            name="simple_single_ern",
            cotx=[
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


@pytest.fixture
def simple_single_cotx_ERN(lib):
    """From test_biorules: 1 cotx group, 2 plasmids, 4 units with variable uORFs"""
    with LibraryContext.with_library(lib):
        u1 = Slot(part=["1w_uORF", "2x_uORF"], ref_id="U1")
        u2 = Slot(part=[None, "4x_uORF", "3x_uORF"], ref_id="U2")

        return Network(
            name="simple_single_cotx_ERN",
            cotx=[
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                u1,
                                Slot(part="CasE_rec"),
                                Slot(part="eBFP2"),
                                Slot(part="L0.T_4560"),
                            ]
                        ),
                        TranscriptionUnit(
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="CasE"),
                                Slot(part="L0.T_4560"),
                            ]
                        ),
                        TranscriptionUnit(
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                u2,
                                Slot(part="Csy4_rec"),
                                Slot(part="eYFP"),
                                Slot(part="L0.T_4560"),
                            ]
                        ),
                        TranscriptionUnit(
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="Csy4"),
                                Slot(part="L0.T_4560"),
                            ]
                        ),
                    ],
                )
            ],
        )


# ============================================================================
# FIXTURE: Multi-Aggregation Networks
# ============================================================================

@pytest.fixture
def multi_aggregation_ern(lib):
    """From 5p_sum_Csy4_2: Multiple aggregations with ERN networks

    3 cotx groups:
    1. EBFP2 reporter + Csy4 source (ratios 0.857, 0.143)
    2. mKO2 reporter + Csy4 target with mNG (ratios 0.5, 0.5)
    3. mMaroon1 reporter + Csy4 target with mNG (ratios 0.5, 0.5)
    """
    with LibraryContext.with_library(lib):
        return Network(
            name="multi_aggregation_ern",
            cotx=[
                # First cotx: EBFP2 + Csy4 source
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
                            name="Csy4_source",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="Csy4"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                    ],
                    ratios=[0.857, 0.143],
                ),
                # Second cotx: mKO2 + Csy4 target
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            name="mKO2_reporter",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="mKO2"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                        TranscriptionUnit(
                            name="Csy4_target_mNG_1",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="Csy4_rec"),
                                Slot(part="mNeonGreen"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                    ],
                    ratios=[0.5, 0.5],
                ),
                # Third cotx: mMaroon1 + Csy4 target
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            name="mMaroon1_reporter",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="mMaroon1"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                        TranscriptionUnit(
                            name="Csy4_target_mNG_2",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="Csy4_rec"),
                                Slot(part="mNeonGreen"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                    ],
                    ratios=[0.5, 0.5],
                ),
            ],
        )


# ============================================================================
# FIXTURE: Variable Parts Networks (for quantization testing)
# ============================================================================

@pytest.fixture
def variable_uorf_network(lib):
    """Network with variable uORF parts for testing quantization"""
    with LibraryContext.with_library(lib):
        return Network(
            name="variable_uorf",
            cotx=[
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part=["1x_uORF", "2x_uORF", "3x_uORF"]),
                                Slot(part="eBFP2"),
                                Slot(part="L0.T_4560"),
                            ]
                        )
                    ]
                )
            ],
        )


# ============================================================================
# Tests for the fixture structures
# ============================================================================

def test_simple_single_reporter(simple_single_reporter):
    """Test the simplest network structure"""
    net = simple_single_reporter
    assert net.name == "simple_single_reporter"
    assert len(net.cotx) == 1
    assert len(net.cotx[0].units) == 1
    assert len(net.cotx[0].units[0].slots) == 4


def test_simple_two_reporters(simple_two_reporters):
    """Test two-reporter aggregation"""
    net = simple_two_reporters
    assert net.name == "simple_two_reporters"
    assert len(net.cotx) == 1
    assert len(net.cotx[0].units) == 2
    assert net.cotx[0].ratios == [0.833, 0.167]


def test_simple_single_ern(simple_single_ern):
    """Test simple ERN network"""
    net = simple_single_ern
    assert net.name == "simple_single_ern"
    assert len(net.cotx) == 1
    assert len(net.cotx[0].units) == 2
    # One unit has CasE_rec (target), other has CasE (source)
    parts = [slot.part for tu in net.cotx[0].units for slot in tu.slots]
    assert "CasE_rec" in parts
    assert "CasE" in parts


def test_simple_single_cotx_ERN(simple_single_cotx_ERN):
    """Test ERN network with variable uORFs"""
    net = simple_single_cotx_ERN
    assert net.name == "simple_single_cotx_ERN"
    assert len(net.cotx) == 1
    assert len(net.cotx[0].units) == 4
    # Check that we have variable parts
    all_slots = [slot for tu in net.cotx[0].units for slot in tu.slots]
    variable_slots = [s for s in all_slots if isinstance(s.part, list) and len(s.part) > 1]
    assert len(variable_slots) == 2  # u1 and u2


def test_multi_aggregation_ern(multi_aggregation_ern):
    """Test multi-aggregation ERN network"""
    net = multi_aggregation_ern
    assert net.name == "multi_aggregation_ern"
    assert len(net.cotx) == 3
    assert net.cotx[0].ratios == [0.857, 0.143]
    assert net.cotx[1].ratios == [0.5, 0.5]
    assert net.cotx[2].ratios == [0.5, 0.5]


def test_variable_uorf_network(variable_uorf_network):
    """Test network with variable uORF quantization"""
    net = variable_uorf_network
    assert net.name == "variable_uorf"
    tu = net.cotx[0].units[0]
    # Find the uORF slot (it's the one with multiple options)
    uorf_slot = [s for s in tu.slots if isinstance(s.part, list) and "uORF" in str(s.part)][0]
    assert len(uorf_slot.part) == 3
    assert "1x_uORF" in uorf_slot.part
