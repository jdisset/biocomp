"""Test fixtures for declarative network recipes

This module provides centralized test network fixtures used throughout the test suite.
All networks are manually constructed using the declarative API to ensure we have
complete control and understanding of the network structure.
"""

import pytest
from biocomp.network_new import Network
from biocomp.recipe_new import CoTransfection, TranscriptionUnit, Slot, Recipe
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
    """Simple ERN network: 1 cotx, 1 plasmid, 2 TUs (ERN + target)

    This is the minimal ERN network:
    - One reporter with ERN recognition site (CasE_rec)
    - One ERN source (CasE enzyme)
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
    """Test simple ERN network"""
    recipe = simple_single_ern
    assert recipe.name == "simple_single_ern"
    assert len(recipe.content) == 1
    assert len(recipe.content[0].units) == 2
    # One unit has CasE_rec (target), other has CasE (source)
    parts = [slot.part for tu in recipe.content[0].units for slot in tu.slots]
    assert "CasE_rec" in parts
    assert "CasE" in parts


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


# ============================================================================
# Additional fixtures from old test_recipe_roundtrip.py
# ============================================================================


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
    from biocomp.network_new import recipe_to_networks
    import biocomp.biorules as br

    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(simple_aggregation, invert=False)
        compg = networks[0].compute_graph

        agg_nodes = [n for n in compg.nodes.values() if n.node_type == "aggregation"]
        assert len(agg_nodes) == 1

        agg = agg_nodes[0]
        assert agg.extra["ratios"] == [0.5, 0.5]
        assert len(agg.extra["members"]) == 2

        output_nodes = [n for n in compg.nodes.values() if n.node_type == "output"]
        assert len(output_nodes) == 1

        output_node = output_nodes[0]
        incoming = compg.get_incoming_edges(output_node.node_id)
        proteins = sorted([e.content[0].name for e in incoming])
        assert "eBFP2" in proteins
        assert "mKO2" in proteins


def test_multi_cotx_aggregation_compg(lib, multi_cotx_aggregation):
    from biocomp.network_new import recipe_to_networks
    import biocomp.biorules as br

    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(multi_cotx_aggregation, invert=False)
        compg = networks[0].compute_graph

        agg_nodes = [n for n in compg.nodes.values() if n.node_type == "aggregation"]
        assert len(agg_nodes) == 2

        ratios_sets = [tuple(agg.extra["ratios"]) for agg in agg_nodes]
        # Ratios are normalized: [0.25, 0.25] → [0.5, 0.5], [0.3, 0.2] → [0.6, 0.4]
        assert (0.5, 0.5) in ratios_sets
        assert (0.6, 0.4) in ratios_sets

        output_nodes = [n for n in compg.nodes.values() if n.node_type == "output"]
        assert len(output_nodes) == 1


def test_complex_ern_compg(lib, complex_ern_network):
    from biocomp.network_new import recipe_to_networks
    import biocomp.biorules as br

    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(complex_ern_network, invert=False)
        compg = networks[0].compute_graph

        agg_nodes = [n for n in compg.nodes.values() if n.node_type == "aggregation"]
        assert len(agg_nodes) == 1
        # Ratios are normalized: [0.2, 0.2, 0.1] → [0.4, 0.4, 0.2]
        assert agg_nodes[0].extra["ratios"] == [0.4, 0.4, 0.2]

        ern_nodes = [n for n in compg.nodes.values() if n.node_type == "sequestron_ERN"]
        assert len(ern_nodes) == 1

        ern_node = ern_nodes[0]
        incoming_ern = compg.get_incoming_edges(ern_node.node_id)

        positive_edges = [e for e in incoming_ern if e.input_slot == 0]
        assert len(positive_edges) == 1
        positive_content = [p.name for p in positive_edges[0].content]
        assert "CasE" in positive_content

        negative_edges = [e for e in incoming_ern if e.input_slot == 1]
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
    from biocomp.network_new import recipe_to_networks
    import biocomp.biorules as br

    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(uorf_ern_network, invert=False)
        compg = networks[0].compute_graph

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
    from biocomp.network_new import recipe_to_networks
    import biocomp.biorules as br

    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(simple_single_ern, invert=False)
        compg = networks[0].compute_graph

        source_nodes = [n for n in compg.nodes.values() if n.node_type == "source"]
        assert len(source_nodes) == 2

        transcription_nodes = [n for n in compg.nodes.values() if n.node_type == "transcription"]
        assert len(transcription_nodes) == 2

        translation_nodes = [n for n in compg.nodes.values() if n.node_type == "translation"]
        assert len(translation_nodes) == 2

        ern_nodes = [n for n in compg.nodes.values() if n.node_type == "sequestron_ERN"]
        assert len(ern_nodes) == 1

        ern = ern_nodes[0]
        outgoing_ern = compg.get_outgoing_edges(ern.node_id)
        assert len(outgoing_ern) == 1


def test_variable_uorf_compg_params(lib, variable_uorf_network):
    from biocomp.network_new import recipe_to_networks
    import biocomp.biorules as br

    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(variable_uorf_network, invert=False)
        compg = networks[0].compute_graph

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
    from biocomp.network_new import recipe_to_networks
    import biocomp.biorules as br

    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(multi_aggregation_ern, invert=False)
        compg = networks[0].compute_graph

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
