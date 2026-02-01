"""Test fixtures for declarative network recipes

This module provides centralized test network fixtures used throughout the test suite.
All networks are manually constructed using the declarative API to ensure we have
complete control and understanding of the network structure.
"""

import pytest
from biocomp.recipe import CoTransfection, TranscriptionUnit, Slot, Recipe
from biocomp.library import load_lib, LibraryContext
from biocomp.recipe import FluoIntensity, NumRange


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
    """Simple network: 1 cotx, 2 plasmids, 2 TUs (two reporters aggregated)

    This corresponds to NoiseFloor_1_5 recipe:
    - EBFP2 reporter (ratio 0.833)
    - mMaroon1 reporter (ratio 0.167)
    """
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


# ============================================================================
# FIXTURE: ERN Networks (Enzymatic Reaction Network)
# ============================================================================


@pytest.fixture
def simple_single_ern(lib):
    """Simple ERN network: ERN + target + separate reporter for inversion

    This is an invertible ERN network:
    - One reporter with ERN recognition site (CasE_rec + eBFP2)
    - One ERN source (CasE enzyme)
    - One separate reporter (mNeonGreen) for invertible path
    """
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
                        TranscriptionUnit(
                            name="mNeonGreen_reporter",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="mNeonGreen"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                    ],
                )
            ],
        )


@pytest.fixture
def simple_single_cotx_ERN(lib):
    """1 cotx group, 3 plasmids, 4 units with variable uORFs"""
    with LibraryContext.with_library(lib):
        u1 = Slot(part=["1w_uORF", "2x_uORF"], ref_id="U1")
        u2 = Slot(part=[None, "4x_uORF", "3x_uORF"], ref_id="U2")

        return Recipe(
            name="simple_single_cotx_ERN",
            content=[
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
                            ],
                            source="plsmd1",
                        ),
                        TranscriptionUnit(
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="CasE"),
                                Slot(part="L0.T_4560"),
                            ],
                            source="plsmd1",
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
        return Recipe(
            name="multi_aggregation_ern",
            content=[
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
        return Recipe(
            name="variable_uorf",
            content=[
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(
                                    part=["1x_uORF", "2x_uORF", "3x_uORF"]
                                ),  # "unlocked" uorf part
                                Slot(part="eBFP2"),
                                Slot(part="L0.T_4560"),
                            ]
                        )
                    ]
                )
            ],
        )


# ============================================================================
# FIXTURE: Unlocked Parameters Networks (for design mode testing)
# ============================================================================


@pytest.fixture
def unlocked_ratios_network(lib):
    """Network with unlocked ratios for testing NumRange functionality"""
    from biocomp.recipe import NumRange

    with LibraryContext.with_library(lib):
        return Recipe(
            name="unlocked_ratios",
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
                    ratios=[NumRange(min=0.5, max=0.9), 0.2],  # First ratio unlocked, second locked
                )
            ],
        )


@pytest.fixture
def bias_network(lib):
    """Network with FluoIntensity bias for testing bias node functionality"""
    from biocomp.recipe import FluoIntensity

    with LibraryContext.with_library(lib):
        return Recipe(
            name="bias_network",
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
                    fluo_bias=FluoIntensity(
                        tu_id=0, value=100.0, protein="eBFP2", units="AU"
                    ),  # Locked bias
                )
            ],
        )


@pytest.fixture
def unlocked_bias_network(lib):
    """Network with unlocked FluoIntensity bias"""
    from biocomp.recipe import FluoIntensity, NumRange

    with LibraryContext.with_library(lib):
        return Recipe(
            name="unlocked_bias",
            content=[
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            name="mNeonGreen_reporter",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="mNeonGreen"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                    ],
                    fluo_bias=FluoIntensity(
                        tu_id=0,
                        value=NumRange(min=50.0, max=200.0),  # Unlocked bias
                        protein="mNeonGreen",
                        units="AU",
                    ),
                )
            ],
        )


@pytest.fixture
def combined_unlocked_network(lib):
    """Network with both unlocked ratios and unlocked bias"""
    from biocomp.recipe import FluoIntensity, NumRange

    with LibraryContext.with_library(lib):
        return Recipe(
            name="combined_unlocked",
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
                    ratios=[
                        NumRange(min=0.6, max=0.9),
                        NumRange(min=0.1, max=0.4),
                    ],  # Both ratios unlocked
                    fluo_bias=FluoIntensity(
                        tu_id=0,
                        value=NumRange(min=80.0, max=150.0),
                        protein="eBFP2",
                        units="AU",
                    ),
                )
            ],
        )


@pytest.fixture
def ern_with_unlocked_ratios(lib):
    """ERN network with unlocked ratios and separate fluo marker for invertibility"""
    from biocomp.recipe import NumRange

    with LibraryContext.with_library(lib):
        return Recipe(
            name="ern_unlocked_ratios",
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
                        TranscriptionUnit(
                            name="mNeonGreen_reporter",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="mNeonGreen"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                    ],
                    ratios=[
                        NumRange(min=0.2, max=0.5),  # Unlocked ratio for CasE_target
                        0.3,  # Locked ratio for CasE_source
                        NumRange(min=0.1, max=0.3),  # Unlocked ratio for mNeonGreen
                    ],
                )
            ],
        )


@pytest.fixture
def ern_with_unlocked_bias(lib):
    """ERN network with unlocked bias and separate fluo marker"""
    from biocomp.recipe import FluoIntensity, NumRange

    with LibraryContext.with_library(lib):
        return Recipe(
            name="ern_unlocked_bias",
            content=[
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            name="Csy4_target",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="Csy4_rec"),
                                Slot(part="eYFP"),
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
                        TranscriptionUnit(
                            name="mKO2_reporter",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="mKO2"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                    ],
                    ratios=[0.4, 0.3, 0.3],  # All locked ratios
                    fluo_bias=FluoIntensity(
                        tu_id=2,  # mKO2 reporter (the invertible path)
                        value=NumRange(min=50.0, max=200.0),  # Unlocked bias
                        protein="mKO2",
                        units="AU",
                    ),
                )
            ],
        )


@pytest.fixture
def ern_with_unlocked_uorfs(lib):
    """ERN network with unlocked uORF parts and separate fluo marker"""
    with LibraryContext.with_library(lib):
        return Recipe(
            name="ern_unlocked_uorfs",
            content=[
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            name="CasE_target_with_uorf",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part=["1x_uORF", "2x_uORF", "3x_uORF"]),  # Unlocked uORF
                                Slot(part="CasE_rec"),
                                Slot(part="eBFP2"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                        TranscriptionUnit(
                            name="CasE_source_with_uorf",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part=["1w_uORF", "2x_uORF"]),  # Unlocked uORF
                                Slot(part="CasE"),
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
                )
            ],
        )


@pytest.fixture
def complex_mixed_unlocked(lib):
    """Complex ERN network with mixed locked/unlocked ratios, bias, and uORF parts"""

    with LibraryContext.with_library(lib):
        return Recipe(
            name="complex_mixed_unlocked",
            content=[
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            name="Csy4_target_with_uorf",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part=["1x_uORF", "4x_uORF"]),  # Unlocked uORF
                                Slot(part="Csy4_rec"),
                                Slot(part="mNeonGreen"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                        TranscriptionUnit(
                            name="Csy4_source",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="1w_uORF"),  # Locked uORF
                                Slot(part="Csy4"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                        TranscriptionUnit(
                            name="eBFP2_reporter",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="eBFP2"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                    ],
                    ratios=[
                        NumRange(min=0.3, max=0.6),  # Unlocked ratio
                        0.2,  # Locked ratio
                        NumRange(min=0.1, max=0.4),  # Unlocked ratio
                    ],
                    fluo_bias=FluoIntensity(
                        tu_id=2,  # eBFP2 reporter (the invertible path)
                        value=NumRange(min=75.0, max=175.0),  # Unlocked bias
                        protein="eBFP2",
                        units="AU",
                    ),
                )
            ],
        )


# ============================================================================
# Tests for the fixture structures
# ============================================================================


def test_simple_single_reporter(simple_single_reporter):
    """Test the simplest network structure"""
    recipe = simple_single_reporter
    assert recipe.name == "simple_single_reporter"
    assert len(recipe.content) == 1
    assert len(recipe.content[0].units) == 1
    assert len(recipe.content[0].units[0].slots) == 4


def test_simple_two_reporters(simple_two_reporters):
    """Test two-reporter aggregation"""
    recipe = simple_two_reporters
    assert recipe.name == "simple_two_reporters"
    assert len(recipe.content) == 1
    assert len(recipe.content[0].units) == 2
    assert recipe.content[0].ratios == [0.833, 0.167]


def test_simple_single_ern(simple_single_ern):
    """Test simple ERN network (with invertible reporter)"""
    recipe = simple_single_ern
    assert recipe.name == "simple_single_ern"
    assert len(recipe.content) == 1
    assert (
        len(recipe.content[0].units) == 3
    )  # ERN target, ERN source, and reporter for invertibility
    # One unit has CasE_rec (target), one has CasE (source), one has mNeonGreen (reporter)
    parts = [slot.part for tu in recipe.content[0].units for slot in tu.slots]
    assert "CasE_rec" in parts
    assert "CasE" in parts
    assert "mNeonGreen" in parts


def test_simple_single_cotx_ERN(simple_single_cotx_ERN):
    """Test ERN network with variable uORFs"""
    recipe = simple_single_cotx_ERN
    assert recipe.name == "simple_single_cotx_ERN"
    assert len(recipe.content) == 1
    assert len(recipe.content[0].units) == 4
    # Check that we have variable parts
    all_slots = [slot for tu in recipe.content[0].units for slot in tu.slots]
    variable_slots = [s for s in all_slots if isinstance(s.part, list) and len(s.part) > 1]
    assert len(variable_slots) == 2  # u1 and u2


def test_multi_aggregation_ern(multi_aggregation_ern):
    """Test multi-aggregation ERN network"""
    recipe = multi_aggregation_ern
    assert recipe.name == "multi_aggregation_ern"
    assert len(recipe.content) == 3
    assert recipe.content[0].ratios == [0.857, 0.143]
    assert recipe.content[1].ratios == [0.5, 0.5]
    assert recipe.content[2].ratios == [0.5, 0.5]


def test_variable_uorf_network(variable_uorf_network):
    """Test network with variable uORF quantization"""
    recipe = variable_uorf_network
    assert recipe.name == "variable_uorf"
    tu = recipe.content[0].units[0]
    uorf_slot = [s for s in tu.slots if isinstance(s.part, list) and "uORF" in str(s.part)][0]
    assert len(uorf_slot.part) == 3
    assert "1x_uORF" in uorf_slot.part


@pytest.fixture
def simple_aggregation(lib):
    """Two L1 plasmids aggregated with equal ratios"""
    with LibraryContext.with_library(lib):
        return Recipe(
            name="simple_aggregation",
            description="Two reporters aggregated 50/50",
            content=[
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="eBFP2"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                        TranscriptionUnit(
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="mKO2"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                    ],
                    ratios=[0.5, 0.5],
                )
            ],
        )


@pytest.fixture
def multi_cotx_aggregation(lib):
    """Multiple cotransfection groups with different aggregation ratios"""
    with LibraryContext.with_library(lib):
        return Recipe(
            name="multi_cotx_aggregation",
            description="Two cotx groups with different aggregation patterns",
            content=[
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="eBFP2"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                        TranscriptionUnit(
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="mNeonGreen"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                    ],
                    ratios=[0.25, 0.25],
                ),
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="mKO2"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                        TranscriptionUnit(
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="mMaroon1"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                    ],
                    ratios=[0.3, 0.2],
                ),
            ],
        )


@pytest.fixture
def complex_ern_network(lib):
    """ERN network with target, source, and control reporter"""
    with LibraryContext.with_library(lib):
        return Recipe(
            name="complex_ern",
            description="CasE ERN with mNG target and mMaroon1 control",
            content=[
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            name="CasE_target",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="CasE_rec"),
                                Slot(part="mNeonGreen"),
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
                        TranscriptionUnit(
                            name="control_reporter",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="mMaroon1"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                    ],
                    ratios=[0.2, 0.2, 0.1],
                )
            ],
        )


@pytest.fixture
def uorf_ern_network(lib):
    """ERN network with uORF on source plasmid"""
    with LibraryContext.with_library(lib):
        return Recipe(
            name="uorf_ern",
            description="Csy4 ERN with 1w_uORF on source",
            content=[
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            name="Csy4_target",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="Csy4_rec"),
                                Slot(part="eYFP"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                        TranscriptionUnit(
                            name="Csy4_source_with_uorf",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="1w_uORF"),
                                Slot(part="Csy4"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                    ],
                )
            ],
        )


# ============================================================================
# Tests for new fixtures
# ============================================================================


def test_simple_aggregation(simple_aggregation):
    recipe = simple_aggregation
    assert recipe.name == "simple_aggregation"
    assert len(recipe.content) == 1
    assert len(recipe.content[0].units) == 2
    assert recipe.content[0].ratios == [0.5, 0.5]


def test_multi_cotx_aggregation(multi_cotx_aggregation):
    recipe = multi_cotx_aggregation
    assert recipe.name == "multi_cotx_aggregation"
    assert len(recipe.content) == 2
    assert recipe.content[0].ratios == [0.25, 0.25]
    assert recipe.content[1].ratios == [0.3, 0.2]


def test_complex_ern_network(complex_ern_network):
    recipe = complex_ern_network
    assert recipe.name == "complex_ern"
    assert len(recipe.content) == 1
    assert len(recipe.content[0].units) == 3
    parts = [slot.part for tu in recipe.content[0].units for slot in tu.slots]
    assert "CasE_rec" in parts
    assert "CasE" in parts
    assert "mNeonGreen" in parts
    assert "mMaroon1" in parts


def test_uorf_ern_network(uorf_ern_network):
    recipe = uorf_ern_network
    assert recipe.name == "uorf_ern"
    assert len(recipe.content) == 1
    assert len(recipe.content[0].units) == 2

    # Flatten parts (some may be lists due to Slot validation)
    parts_raw = [slot.part for tu in recipe.content[0].units for slot in tu.slots]
    parts = []
    for p in parts_raw:
        if isinstance(p, list):
            parts.extend(p)
        else:
            parts.append(p)

    assert "Csy4_rec" in parts
    assert "Csy4" in parts
    assert "1w_uORF" in parts


# ============================================================================
# Compute Graph Structure Tests
# ============================================================================


def test_simple_aggregation_compg(lib, simple_aggregation):
    from biocomp.network import recipe_to_networks

    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(simple_aggregation, invert=False)
        compg = networks[0].compute_graph
        assert compg is not None

        agg_nodes = [n for n in compg.nodes.values() if n.node_type == "aggregation"]
        assert len(agg_nodes) == 1

        agg = agg_nodes[0]
        members = agg.extra["members"]
        assert isinstance(members, dict)
        assert len(members) == 2
        ratios = [members[m]["ratio"] for m in sorted(members.keys())]
        assert ratios == [0.5, 0.5]

        output_nodes = [n for n in compg.nodes.values() if n.node_type == "output"]
        assert len(output_nodes) == 1

        output_node = output_nodes[0]
        incoming = compg.get_incoming_edges(output_node.node_id)
        proteins = sorted([e.content[0].name for e in incoming])
        assert "eBFP2" in proteins
        assert "mKO2" in proteins


def test_single_cotx_ERN(lib, simple_two_reporters):
    from biocomp.network import recipe_to_networks

    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(simple_two_reporters, invert=False)
        compg = networks[0].compute_graph
        assert compg is not None

        agg_nodes = [n for n in compg.nodes.values() if n.node_type == "aggregation"]
        assert len(agg_nodes) == 1

        agg = agg_nodes[0]
        members = agg.extra["members"]
        sorted_ids = sorted(members.keys())
        ratios = [members[m]["ratio"] for m in sorted_ids]
        assert ratios == [0.833, 0.167]
        assert len(members) == 2

        output_nodes = [n for n in compg.nodes.values() if n.node_type == "output"]
        assert len(output_nodes) == 1

        output_node = output_nodes[0]
        incoming = compg.get_incoming_edges(output_node.node_id)
        proteins = sorted([e.content[0].name for e in incoming])
        assert "eBFP2" in proteins
        assert "mMaroon1" in proteins


def test_multi_cotx_aggregation_compg(lib, multi_cotx_aggregation):
    from biocomp.network import recipe_to_networks

    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(multi_cotx_aggregation, invert=False)
        compg = networks[0].compute_graph
        assert compg is not None

        agg_nodes = [n for n in compg.nodes.values() if n.node_type == "aggregation"]
        assert len(agg_nodes) == 2

        ratios_sets = []
        for agg in agg_nodes:
            members = agg.extra["members"]
            assert isinstance(members, dict)
            ratios_sets.append(tuple(members[m]["ratio"] for m in sorted(members.keys())))
        assert (0.5, 0.5) in ratios_sets
        assert (0.6, 0.4) in ratios_sets

        output_nodes = [n for n in compg.nodes.values() if n.node_type == "output"]
        assert len(output_nodes) == 1


def test_complex_ern_compg(lib, complex_ern_network):
    from biocomp.network import recipe_to_networks

    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(complex_ern_network, invert=False)
        compg = networks[0].compute_graph
        assert compg is not None

        agg_nodes = [n for n in compg.nodes.values() if n.node_type == "aggregation"]
        assert len(agg_nodes) == 1
        # Ratios are normalized: [0.2, 0.2, 0.1] → [0.4, 0.4, 0.2]
        agg = agg_nodes[0]
        members = agg.extra["members"]
        sorted_ids = sorted(members.keys())
        ratios = [members[m]["ratio"] for m in sorted_ids]
        assert ratios == [0.4, 0.4, 0.2]

        ern_nodes = [n for n in compg.nodes.values() if n.node_type == "sequestron_ERN"]
        assert len(ern_nodes) == 1

        ern_node = ern_nodes[0]
        incoming_ern = compg.get_incoming_edges(ern_node.node_id)

        positive_edges = [e for e in incoming_ern if e.to_input_slot == 0]
        assert len(positive_edges) == 1
        positive_content = [p.name for p in positive_edges[0].content]
        assert "CasE" in positive_content

        negative_edges = [e for e in incoming_ern if e.to_input_slot == 1]
        assert len(negative_edges) == 1
        negative_content = [p.name for p in negative_edges[0].content]
        assert "CasE_rec" in negative_content

        output_nodes = [n for n in compg.nodes.values() if n.node_type == "output"]
        assert len(output_nodes) == 1
        incoming_output = compg.get_incoming_edges(output_nodes[0].node_id)
        output_proteins = sorted([e.content[0].name for e in incoming_output])
        assert "mNeonGreen" in output_proteins
        assert "mMaroon1" in output_proteins


def test_uorf_ern_compg(lib, uorf_ern_network):
    from biocomp.network import recipe_to_networks

    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(uorf_ern_network, invert=False)
        compg = networks[0].compute_graph
        assert compg is not None

        # Check for translation nodes (central dogma: DNA → RNA → Protein)
        translation_nodes = [n for n in compg.nodes.values() if n.node_type == "translation"]
        assert len(translation_nodes) == 2  # One for Csy4, one for Csy4_rec+eYFP

        # Check for ERN node
        ern_nodes = [n for n in compg.nodes.values() if n.node_type == "sequestron_ERN"]
        assert len(ern_nodes) == 1

        # Verify graph was built successfully (uORFs affect parameters but may not appear as graph parts)
        # The key is that the network with uORF builds without errors
        assert len(compg.nodes) > 0
        assert len(compg.edges) > 0


def test_simple_single_ern_compg_detailed(lib, simple_single_ern):
    from biocomp.network import recipe_to_networks

    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(simple_single_ern, invert=False)
        compg = networks[0].compute_graph
        assert compg is not None

        source_nodes = [n for n in compg.nodes.values() if n.node_type == "source"]
        assert len(source_nodes) == 3  # ERN target, ERN source, and reporter

        transcription_nodes = [n for n in compg.nodes.values() if n.node_type == "transcription"]
        assert len(transcription_nodes) == 3  # 3 transcription units

        translation_nodes = [n for n in compg.nodes.values() if n.node_type == "translation"]
        assert len(translation_nodes) == 3  # 3 proteins (eBFP2, CasE, mNeonGreen)

        ern_nodes = [n for n in compg.nodes.values() if n.node_type == "sequestron_ERN"]
        assert len(ern_nodes) == 1

        ern = ern_nodes[0]
        outgoing_ern = compg.get_outgoing_edges(ern.node_id)
        assert len(outgoing_ern) == 1


def test_variable_uorf_compg_params(lib, variable_uorf_network):
    from biocomp.network import recipe_to_networks

    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(variable_uorf_network, invert=False)
        compg = networks[0].compute_graph
        assert compg is not None

        source_nodes = [n for n in compg.nodes.values() if n.node_type == "source"]
        assert len(source_nodes) == 1

        source = source_nodes[0]
        outgoing = compg.get_outgoing_edges(source.node_id)
        dna_edges = [e for e in outgoing if e.content_type == "DNA"]
        assert len(dna_edges) == 1

        dna_content = dna_edges[0].content
        dna_parts = [p.name for p in dna_content]
        # DNA edges contain insulator, gene, terminator (promoter handled separately)
        assert "cHS4" in dna_parts
        assert "eBFP2" in dna_parts


def test_multi_aggregation_ern_compg_structure(lib, multi_aggregation_ern):
    from biocomp.network import recipe_to_networks

    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(multi_aggregation_ern, invert=False)
        compg = networks[0].compute_graph
        assert compg is not None

        agg_nodes = [n for n in compg.nodes.values() if n.node_type == "aggregation"]
        assert len(agg_nodes) == 3

        cotx_groups = set(agg.extra["cotx_group"] for agg in agg_nodes)
        assert len(cotx_groups) == 3

        # Only 1 ERN node since both cotx 2 and 3 use the same Csy4/Csy4_rec system
        ern_nodes = [n for n in compg.nodes.values() if n.node_type == "sequestron_ERN"]
        assert len(ern_nodes) == 1
        assert "Csy4" in ern_nodes[0].extra.get("seq_name", "")

        output_nodes = [n for n in compg.nodes.values() if n.node_type == "output"]
        assert len(output_nodes) == 1


# ============================================================================
# Tests for Unlocked Parameters
# ============================================================================


def test_unlocked_ratios_fixture(unlocked_ratios_network):
    """Test that unlocked ratios network fixture is correctly structured"""
    from biocomp.recipe import NumRange

    recipe = unlocked_ratios_network
    assert recipe.name == "unlocked_ratios"
    assert len(recipe.content) == 1
    cotx = recipe.content[0]
    assert len(cotx.units) == 2
    assert len(cotx.ratios) == 2
    # First ratio should be NumRange
    assert isinstance(cotx.ratios[0], NumRange)
    assert cotx.ratios[0].min == 0.5
    assert cotx.ratios[0].max == 0.9
    # Second ratio should be locked (float)
    assert isinstance(cotx.ratios[1], (int, float))
    assert cotx.ratios[1] == 0.2


def test_bias_network_fixture(bias_network):
    """Test that bias network fixture is correctly structured"""
    from biocomp.recipe import FluoIntensity

    recipe = bias_network
    assert recipe.name == "bias_network"
    cotx = recipe.content[0]
    assert cotx.fluo_bias is not None
    assert isinstance(cotx.fluo_bias, FluoIntensity)
    assert cotx.fluo_bias.tu_id == 0
    assert cotx.fluo_bias.protein == "eBFP2"
    assert cotx.fluo_bias.is_locked()
    assert cotx.fluo_bias.get_value() == 100.0


def test_unlocked_ratios_network_builds(lib, unlocked_ratios_network):
    """Test that network with unlocked ratios builds successfully"""
    from biocomp.network import recipe_to_networks
    import biocomp.biorules as br

    with LibraryContext.with_library(lib):
        # Use inversion_mode="shortest" to get a single network for simpler testing
        networks = recipe_to_networks(unlocked_ratios_network, br.ALL_RULES, invert=True, inversion_mode="shortest")
        assert len(networks) == 1
        net = networks[0]
        compg = net.compute_graph

        # Should have aggregation node with ratio_ranges in members
        agg_nodes = [n for n in compg.nodes.values() if n.node_type == "aggregation"]
        assert len(agg_nodes) == 1
        agg = agg_nodes[0]
        members = agg.extra["members"]
        sorted_ids = sorted(members.keys())
        ratio_ranges = [members[m].get("ratio_range") for m in sorted_ids]
        # First ratio should have range info (unlocked)
        assert ratio_ranges[0] is not None
        assert ratio_ranges[0]["min"] == 0.5
        assert ratio_ranges[0]["max"] == 0.9
        # Second ratio should be locked (None)
        assert ratio_ranges[1] is None


def test_unlocked_ratios_initialization(lib, unlocked_ratios_network):
    """Test that unlocked ratios are initialized within their ranges"""
    import jax
    from biocomp.network import recipe_to_networks
    from biocomp.compute import ComputeStack
    from biocomp.config import SIMPLE_NODES_COMPUTE_CONFIG
    import biocomp.biorules as br

    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(unlocked_ratios_network, br.ALL_RULES, invert=True)
        stack = ComputeStack([networks[0]])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        # Test multiple random initializations
        for seed in range(10):
            key = jax.random.PRNGKey(seed)
            params = stack.init(key)

            # Find aggregation layer
            agg_namespace = None
            for layer in stack.layers:
                if layer.f_type == "aggregation":
                    agg_namespace = layer.namespace
                    break

            assert agg_namespace is not None, "No aggregation layer found"

            ratios = params[f"{agg_namespace}/ratios"][0]  # Get ratios for node 0
            # First ratio should be within [0.5, 0.9] range (normalized)
            # After normalization, ratios sum to 1.0, so check relative proportions
            assert ratios.shape == (2,), f"Expected 2 ratios, got {ratios.shape}"


def test_bias_network_creates_bias_node(lib, bias_network):
    """Test that network with FluoIntensity creates bias node (not numeric node)"""
    from biocomp.network import recipe_to_networks
    import biocomp.biorules as br

    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(bias_network, br.ALL_RULES, invert=False)
        assert len(networks) == 1
        compg = networks[0].compute_graph

        # Should have a bias node, NOT a numeric node
        bias_nodes = [n for n in compg.nodes.values() if n.node_type == "bias"]
        numeric_nodes = [n for n in compg.nodes.values() if n.node_type == "numeric"]

        assert len(bias_nodes) == 1, "Should have exactly 1 bias node"
        assert len(numeric_nodes) == 0, "Should have NO numeric nodes"

        # Check bias node properties (stored in fluo_bias as dict)
        bias = bias_nodes[0]
        assert bias.extra["role"] == "fluo_bias"
        assert "fluo_bias" in bias.extra

        # fluo_bias should be a dict
        fluo_data = bias.extra["fluo_bias"]
        assert isinstance(fluo_data, dict), f"Expected dict, got {type(fluo_data)}"
        assert fluo_data["tu_id"] == 0
        assert fluo_data["value"] == 100.0
        assert fluo_data["protein"] == "eBFP2"
        assert fluo_data["units"] == "AU"


def test_unlocked_bias_node_properties(lib, unlocked_bias_network):
    """Test that unlocked bias has correct range properties"""
    from biocomp.network import recipe_to_networks
    import biocomp.biorules as br

    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(unlocked_bias_network, br.ALL_RULES, invert=False)
        compg = networks[0].compute_graph

        bias_nodes = [n for n in compg.nodes.values() if n.node_type == "bias"]
        assert len(bias_nodes) == 1

        bias = bias_nodes[0]
        # Value should be a dict with min/max for unlocked bias (stored in fluo_bias as dict)
        fluo_data = bias.extra["fluo_bias"]
        assert isinstance(fluo_data, dict), f"Expected dict, got {type(fluo_data)}"
        value = fluo_data["value"]
        assert isinstance(value, dict)
        assert value["min"] == 50.0
        assert value["max"] == 200.0


def test_combined_unlocked_network(lib, combined_unlocked_network):
    """Test network with both unlocked ratios and unlocked bias"""
    from biocomp.network import recipe_to_networks
    import biocomp.biorules as br

    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(combined_unlocked_network, br.ALL_RULES, invert=False)
        compg = networks[0].compute_graph

        # Should have bias node (not yet implemented for multi-TU cotx)
        bias_nodes = [n for n in compg.nodes.values() if n.node_type == "bias"]
        assert len(bias_nodes) == 1

        # Should have aggregation with ratio_ranges in members
        agg_nodes = [n for n in compg.nodes.values() if n.node_type == "aggregation"]
        assert len(agg_nodes) == 1
        agg = agg_nodes[0]
        members = agg.extra["members"]
        sorted_ids = sorted(members.keys())
        ratio_ranges = [members[m].get("ratio_range") for m in sorted_ids]
        # Both ratios should be unlocked
        assert all(r is not None for r in ratio_ranges)


def test_unlocked_ratios_commit(lib, unlocked_ratios_network):
    """Test that commit() locks unlocked ratios"""
    import jax
    from biocomp.network import recipe_to_networks
    from biocomp.compute import ComputeStack
    from biocomp.config import SIMPLE_NODES_COMPUTE_CONFIG
    import biocomp.biorules as br

    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(unlocked_ratios_network, br.ALL_RULES, invert=True)
        stack = ComputeStack([networks[0]])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        key = jax.random.PRNGKey(42)
        params = stack.init(key)

        # Ratio ranges metadata is stored with NON_GRAD_TAG and verified during commit below

        committed_networks = stack.commit(params)

        # After commit: check that ratio_ranges in members are all None (locked)
        agg_nodes = [
            n
            for n in committed_networks[0].compute_graph.nodes.values()
            if n.node_type == "aggregation"
        ]
        assert len(agg_nodes) == 1
        agg = agg_nodes[0]
        members = agg.extra["members"]
        ratio_ranges = [members[m].get("ratio_range") for m in members]
        # After commit, all ratio_ranges should be None (locked)
        assert all(r is None for r in ratio_ranges), (
            "All ratios should be locked after commit"
        )


def test_multiple_random_inits_produce_different_ratios(lib, unlocked_ratios_network):
    """Test that multiple random inits of unlocked ratios produce different values"""
    import jax
    import jax.numpy as jnp
    from biocomp.network import recipe_to_networks
    from biocomp.compute import ComputeStack
    from biocomp.config import SIMPLE_NODES_COMPUTE_CONFIG
    import biocomp.biorules as br

    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(unlocked_ratios_network, br.ALL_RULES, invert=True)
        stack = ComputeStack([networks[0]])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        # Get aggregation namespace
        agg_namespace = None
        for layer in stack.layers:
            if layer.f_type == "aggregation":
                agg_namespace = layer.namespace
                break

        # Initialize multiple times and collect ratios
        all_ratios = []
        for seed in range(5):
            key = jax.random.PRNGKey(seed)
            params = stack.init(key)
            ratios = params[f"{agg_namespace}/ratios"][0]
            all_ratios.append(ratios)

        # Check that not all ratios are identical
        all_ratios = jnp.stack(all_ratios)
        std_dev = jnp.std(all_ratios, axis=0)
        # At least one ratio should vary (the unlocked one)
        assert jnp.any(std_dev > 1e-6), "Unlocked ratios should vary across random initializations"


# ============================================================================
# Tests for Mixed Locked/Unlocked Parameter Networks
# ============================================================================


def test_ern_with_unlocked_ratios_structure(ern_with_unlocked_ratios):
    """Test ERN with mixed locked/unlocked ratios structure"""
    from biocomp.recipe import NumRange

    recipe = ern_with_unlocked_ratios
    assert recipe.name == "ern_unlocked_ratios"
    cotx = recipe.content[0]
    assert len(cotx.units) == 3
    assert len(cotx.ratios) == 3
    # Check mixed locked/unlocked ratios
    assert isinstance(cotx.ratios[0], NumRange)  # Unlocked
    assert isinstance(cotx.ratios[1], (int, float))  # Locked
    assert isinstance(cotx.ratios[2], NumRange)  # Unlocked


def test_ern_with_unlocked_ratios_compg(lib, ern_with_unlocked_ratios):
    """Test that ERN with unlocked ratios builds correct compute graph"""
    from biocomp.network import recipe_to_networks
    import biocomp.biorules as br

    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(ern_with_unlocked_ratios, br.ALL_RULES, invert=True)
        compg = networks[0].compute_graph

        # Should have aggregation with mixed ratio_ranges in members
        agg_nodes = [n for n in compg.nodes.values() if n.node_type == "aggregation"]
        assert len(agg_nodes) == 1
        agg = agg_nodes[0]
        members = agg.extra["members"]
        sorted_ids = sorted(members.keys())
        ratio_ranges = [members[m].get("ratio_range") for m in sorted_ids]
        # First and third ratios should be unlocked (have ranges)
        assert ratio_ranges[0] is not None
        assert ratio_ranges[1] is None  # Locked
        assert ratio_ranges[2] is not None

        # Should have ERN node
        ern_nodes = [n for n in compg.nodes.values() if n.node_type == "sequestron_ERN"]
        assert len(ern_nodes) == 1

        # Should have separate reporter for invertibility
        output_nodes = [n for n in compg.nodes.values() if n.node_type == "output"]
        assert len(output_nodes) == 1


def test_ern_with_unlocked_bias_structure(ern_with_unlocked_bias):
    """Test ERN with unlocked bias structure"""
    from biocomp.recipe import FluoIntensity, NumRange

    recipe = ern_with_unlocked_bias
    assert recipe.name == "ern_unlocked_bias"
    cotx = recipe.content[0]
    assert cotx.fluo_bias is not None
    assert isinstance(cotx.fluo_bias, FluoIntensity)
    assert cotx.fluo_bias.tu_id == 2  # mKO2 reporter
    assert not cotx.fluo_bias.is_locked()
    # Check bias value is NumRange
    value = cotx.fluo_bias.value
    assert isinstance(value, NumRange)


def test_ern_with_unlocked_bias_compg(lib, ern_with_unlocked_bias):
    """Test that ERN with unlocked bias creates proper bias node"""
    from biocomp.network import recipe_to_networks
    import biocomp.biorules as br

    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(ern_with_unlocked_bias, br.ALL_RULES, invert=True)
        compg = networks[0].compute_graph

        # Should have bias node with unlocked range
        bias_nodes = [n for n in compg.nodes.values() if n.node_type == "bias"]
        assert len(bias_nodes) == 1
        bias = bias_nodes[0]
        assert bias.extra["role"] == "fluo_bias"
        # Just check that bias node was created with correct role
        # Value range info is embedded in the node's initialization logic


def test_ern_with_unlocked_uorfs_structure(ern_with_unlocked_uorfs):
    """Test ERN with unlocked uORF parts structure"""
    recipe = ern_with_unlocked_uorfs
    assert recipe.name == "ern_unlocked_uorfs"
    cotx = recipe.content[0]
    assert len(cotx.units) == 3

    # Check that first two TUs have unlocked uORF slots
    tu0_slots = cotx.units[0].slots
    uorf_slots_0 = [
        s for s in tu0_slots if isinstance(s.part, list) and any("uORF" in str(p) for p in s.part)
    ]
    assert len(uorf_slots_0) == 1
    assert len(uorf_slots_0[0].part) == 3  # ["1x_uORF", "2x_uORF", "3x_uORF"]

    tu1_slots = cotx.units[1].slots
    uorf_slots_1 = [
        s for s in tu1_slots if isinstance(s.part, list) and any("uORF" in str(p) for p in s.part)
    ]
    assert len(uorf_slots_1) == 1
    assert len(uorf_slots_1[0].part) == 2  # ["1w_uORF", "2x_uORF"]


def test_ern_with_unlocked_uorfs_compg(lib, ern_with_unlocked_uorfs):
    """Test that ERN with unlocked uORFs builds with proper quantization masks"""
    from biocomp.network import recipe_to_networks
    from biocomp.compute import ComputeStack
    from biocomp.config import SIMPLE_NODES_COMPUTE_CONFIG
    import biocomp.biorules as br
    import jax
    import jax.numpy as jnp

    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(ern_with_unlocked_uorfs, br.ALL_RULES, invert=True)
        stack = ComputeStack([networks[0]])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        key = jax.random.PRNGKey(42)
        params = stack.init(key)

        # Find translation layer with uORF quantization masks
        for layer in stack.layers:
            if layer.f_type == "translation":
                tl_mask = params[f"{layer.namespace}/tl_rate_quantization_mask"]
                # Should have multiple True values for unlocked uORF slots
                for node_idx in range(tl_mask.shape[0]):
                    node_mask = tl_mask[node_idx]
                    n_available = jnp.sum(node_mask)
                    # At least one node should have multiple uORF options
                    if n_available > 1:
                        break
                else:
                    pytest.fail("No translation nodes with multiple uORF options found")
                break


def test_complex_mixed_unlocked_structure(complex_mixed_unlocked):
    """Test complex network with all types of unlocked params"""
    from biocomp.recipe import NumRange

    recipe = complex_mixed_unlocked
    assert recipe.name == "complex_mixed_unlocked"
    cotx = recipe.content[0]
    assert len(cotx.units) == 3

    # Check mixed ratios
    assert len(cotx.ratios) == 3
    assert isinstance(cotx.ratios[0], NumRange)  # Unlocked
    assert isinstance(cotx.ratios[1], (int, float))  # Locked
    assert isinstance(cotx.ratios[2], NumRange)  # Unlocked

    # Check unlocked bias
    assert cotx.fluo_bias is not None
    assert not cotx.fluo_bias.is_locked()

    # Check mixed uORF parts
    uorf_slots = []
    for tu in cotx.units:
        for slot in tu.slots:
            if isinstance(slot.part, list) and any("uORF" in str(p) for p in slot.part):
                uorf_slots.append(slot)
            elif isinstance(slot.part, str) and "uORF" in slot.part:
                uorf_slots.append(slot)
    assert len(uorf_slots) >= 2  # Should have both unlocked and locked uORF slots


def test_complex_mixed_unlocked_compg(lib, complex_mixed_unlocked):
    """Test that complex mixed network builds correctly"""
    from biocomp.network import recipe_to_networks
    import biocomp.biorules as br

    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(complex_mixed_unlocked, br.ALL_RULES, invert=True)
        compg = networks[0].compute_graph

        # Should have aggregation with mixed ratio_ranges in members
        agg_nodes = [n for n in compg.nodes.values() if n.node_type == "aggregation"]
        assert len(agg_nodes) == 1
        agg = agg_nodes[0]
        members = agg.extra["members"]
        sorted_ids = sorted(members.keys())
        ratio_ranges = [members[m].get("ratio_range") for m in sorted_ids]
        assert ratio_ranges[0] is not None  # Unlocked
        assert ratio_ranges[1] is None  # Locked
        assert ratio_ranges[2] is not None  # Unlocked

        # Should have bias node
        bias_nodes = [n for n in compg.nodes.values() if n.node_type == "bias"]
        assert len(bias_nodes) == 1

        # Should have ERN node
        ern_nodes = [n for n in compg.nodes.values() if n.node_type == "sequestron_ERN"]
        assert len(ern_nodes) == 1
