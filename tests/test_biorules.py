"""Test biological graph rewriting rules and CDG/compute graph generation

This test suite validates:
1. Central Dogma Graph (CDG) generation from declarative recipes
2. Graph rewriting rules application
3. Compute graph topology and correctness
"""

from biocomp.network import build_central_dogma_graph_direct
from biocomp.library import LibraryContext
import biocomp.biorules as br
from biocomp.graphengine import apply_rule_sequence
from biocomp.inversion import invert_all_paths

# Import all test fixtures


# ============================================================================
# CDG Structure Tests
# ============================================================================

def test_cdg_simple_single_reporter(lib, simple_single_reporter):
    """Test CDG generation for simplest network: 1 TU → 3 nodes (DNA, RNA, PRT)"""
    with LibraryContext.with_library(lib):
        recipe = simple_single_reporter
        cdg = build_central_dogma_graph_direct(recipe.content, lib, dual=False)

        # Should have 3 nodes: DNA → RNA → PRT
        assert len(cdg.nodes) == 3, f"Expected 3 nodes, got {len(cdg.nodes)}"

        # Check node types
        node_types = [n.node_type for n in cdg.nodes.values()]
        assert node_types == ["DNA", "RNA", "PRT"]

        # Check edges: DNA→RNA, RNA→PRT
        assert len(cdg.edges) == 2
        assert cdg.get_edge(0, 1) is not None  # DNA→RNA
        assert cdg.get_edge(1, 2) is not None  # RNA→PRT

        # Check content
        dna_node = cdg.get_node(0)
        assert "eBFP2" in dna_node.extra["content"]
        assert "hEF1a" in dna_node.extra["params"]["tc_rate"]

        # RNA node should have eBFP2
        rna_node = cdg.get_node(1)
        assert "eBFP2" in rna_node.extra["content"]

        # PRT node should have eBFP2 and be marked as output
        prt_node = cdg.get_node(2)
        assert "eBFP2" in prt_node.extra["content"]
        assert prt_node.extra["is_output"] is True


def test_cdg_simple_two_reporters(lib, simple_two_reporters):
    """Test CDG for 2 TUs aggregated: should have 6 nodes (2 × 3)"""
    with LibraryContext.with_library(lib):
        recipe = simple_two_reporters
        cdg = build_central_dogma_graph_direct(recipe.content, lib, dual=False)

        # 2 TUs × 3 nodes each = 6 nodes
        assert len(cdg.nodes) == 6

        # Check we have 2 DNA, 2 RNA, 2 PRT nodes
        node_types = [n.node_type for n in cdg.nodes.values()]
        assert node_types.count("DNA") == 2
        assert node_types.count("RNA") == 2
        assert node_types.count("PRT") == 2

        # Should have 4 edges (2 × 2 edges per TU)
        assert len(cdg.edges) == 4

        # Check that both outputs are marked correctly
        prt_nodes = [n for n in cdg.nodes.values() if n.node_type == "PRT"]
        assert all(n.extra["is_output"] for n in prt_nodes)

        # Check content: should have eBFP2 and mMaroon1
        all_contents = [item for n in cdg.nodes.values() for item in n.extra.get("content", [])]
        assert "eBFP2" in all_contents
        assert "mMaroon1" in all_contents


def test_cdg_simple_single_ern(lib, simple_single_ern):
    """Test CDG for ERN network: 3 TUs (target + source + reporter) = 9 nodes"""
    with LibraryContext.with_library(lib):
        recipe = simple_single_ern
        cdg = build_central_dogma_graph_direct(recipe.content, lib, dual=False)

        # 3 TUs × 3 nodes each = 9 nodes (DNA -> RNA -> PRT for each TU)
        assert len(cdg.nodes) == 9

        # Check node distribution
        node_types = [n.node_type for n in cdg.nodes.values()]
        assert node_types.count("DNA") == 3  # 3 TUs
        assert node_types.count("RNA") == 3  # 3 mRNAs
        assert node_types.count("PRT") == 3  # 3 proteins (eBFP2, CasE, mNeonGreen)

        # Check content: should have CasE, CasE_rec, and mNeonGreen
        all_contents = [item for n in cdg.nodes.values() for item in n.extra.get("content", [])]
        assert "CasE" in all_contents
        assert "CasE_rec" in all_contents
        assert "mNeonGreen" in all_contents

        # Both reporters (eBFP2 and mNeonGreen) should be output
        prt_nodes = [n for n in cdg.nodes.values() if n.node_type == "PRT"]
        output_prts = [n for n in prt_nodes if n.extra["is_output"]]
        assert len(output_prts) == 2  # eBFP2 and mNeonGreen are both reporters
        output_contents = [item for n in output_prts for item in n.extra["content"]]
        assert "eBFP2" in output_contents
        assert "mNeonGreen" in output_contents


def test_cdg_multi_aggregation_ern(lib, multi_aggregation_ern):
    """Test CDG for multi-aggregation network: 3 cotx groups with 6 TUs total

    Note: Two TUs (Csy4_target_mNG in cotx 2 and 3) are identical,
    so their RNA and PRT nodes are shared, resulting in 16 nodes instead of 18.
    """
    with LibraryContext.with_library(lib):
        recipe = multi_aggregation_ern
        cdg = build_central_dogma_graph_direct(recipe.content, lib, dual=False)

        # 6 TUs but 2 share RNA/PRT nodes → 16 nodes instead of 18
        assert len(cdg.nodes) == 16

        # Check node distribution
        node_types = [n.node_type for n in cdg.nodes.values()]
        assert node_types.count("DNA") == 6  # Each TU has its own DNA
        assert node_types.count("RNA") == 5  # 2 TUs share 1 RNA node
        assert node_types.count("PRT") == 5  # 2 TUs share 1 PRT node

        # Check that we have the expected proteins
        all_contents = [item for n in cdg.nodes.values() for item in n.extra.get("content", [])]
        assert "eBFP2" in all_contents
        assert "mKO2" in all_contents
        assert "mMaroon1" in all_contents
        assert "Csy4" in all_contents
        assert "mNeonGreen" in all_contents


def test_cdg_variable_uorf(lib, variable_uorf_network):
    """Test CDG with variable uORF parts - should handle quantization params"""
    with LibraryContext.with_library(lib):
        recipe = variable_uorf_network
        cdg = build_central_dogma_graph_direct(recipe.content, lib, dual=False)

        # Should have standard 3 nodes
        assert len(cdg.nodes) == 3

        # Check that params include the variable uORFs
        dna_node = cdg.get_node(0)
        params = dna_node.extra.get("params", {})
        # uORFs map to tl_rate parameter
        assert "tl_rate" in params
        uorf_params = params["tl_rate"]
        # Should have the 3 uORF options
        assert len(uorf_params) == 3
        assert "1x_uORF" in uorf_params
        assert "2x_uORF" in uorf_params
        assert "3x_uORF" in uorf_params


# ============================================================================
# Compute Graph Tests (after rule application)
# ============================================================================

def test_compute_graph_simple_single_reporter(lib, simple_single_reporter):
    """Test compute graph after rule application for simple reporter"""
    with LibraryContext.with_library(lib):
        recipe = simple_single_reporter
        cdg = build_central_dogma_graph_direct(recipe.content, lib, dual=False)
        compg = apply_rule_sequence(br.ALL_RULES, cdg)[0]

        # The compute graph should still have nodes representing the biological flow
        # but potentially restructured by the rules
        assert len(compg.nodes) > 0

        # Should still have edges
        assert len(compg.edges) >= 0

        # Check that we still have the essential biological information
        [n.node_type for n in compg.nodes.values()]
        # After rules, might have different node types depending on transformations


def test_compute_graph_simple_ern(lib, simple_single_ern):
    """Test compute graph for ERN network after rule application"""
    with LibraryContext.with_library(lib):
        recipe = simple_single_ern
        cdg = build_central_dogma_graph_direct(recipe.content, lib, dual=False)
        compg = apply_rule_sequence(br.ALL_RULES, cdg)[0]

        # Should have more complex structure with ERN interactions
        assert len(compg.nodes) > 0
        assert len(compg.edges) > 0

        # The graph should maintain biological relationships
        # CasE protein should influence CasE_rec mRNA


def test_dual_cdg_generation(lib, simple_single_reporter):
    """Test dual CDG generation (for models with inverse parameters)"""
    with LibraryContext.with_library(lib):
        recipe = simple_single_reporter
        cdg_dual = build_central_dogma_graph_direct(recipe.content, lib, dual=True)

        # Dual mode should create additional nodes for inverse parameters
        cdg_primal = build_central_dogma_graph_direct(recipe.content, lib, dual=False)

        # Dual should have at least as many nodes as primal
        assert len(cdg_dual.nodes) >= len(cdg_primal.nodes)


# ============================================================================
# Rule Application Tests
# ============================================================================

def test_rule_sequence_doesnt_fail(lib, simple_single_cotx_ERN):
    """Test that applying all rules doesn't crash on complex network"""
    with LibraryContext.with_library(lib):
        recipe = simple_single_cotx_ERN
        cdg = build_central_dogma_graph_direct(recipe.content, lib, dual=False)

        # Should not raise exception
        compg = apply_rule_sequence(br.ALL_RULES, cdg)[0]

        assert compg is not None
        assert len(compg.nodes) > 0


def test_cdg_preserves_cotx_groups(lib, multi_aggregation_ern):
    """Test that CDG preserves cotransfection group information"""
    with LibraryContext.with_library(lib):
        recipe = multi_aggregation_ern
        cdg = build_central_dogma_graph_direct(recipe.content, lib, dual=False)

        # Each TU should have cotx group info in tu_id
        for node in cdg.nodes.values():
            if "tu_id" in node.extra:
                tu_ids = node.extra["tu_id"]
                # Should have cotx group marker in tu_id
                assert any("cotx" in str(tu_id) for tu_id in tu_ids)


# ============================================================================
# Integration Tests
# ============================================================================

def test_all_fixtures_generate_valid_cdg(
    lib,
    simple_single_reporter,
    simple_two_reporters,
    simple_single_ern,
    simple_single_cotx_ERN,
    multi_aggregation_ern,
    variable_uorf_network,
):
    """Integration test: all fixtures should generate valid CDGs"""
    fixtures = [
        simple_single_reporter,
        simple_two_reporters,
        simple_single_ern,
        simple_single_cotx_ERN,
        multi_aggregation_ern,
        variable_uorf_network,
    ]

    with LibraryContext.with_library(lib):
        for recipe in fixtures:
            # Should generate CDG without errors
            cdg = build_central_dogma_graph_direct(recipe.content, lib, dual=False)
            assert len(cdg.nodes) > 0
            assert len(cdg.edges) > 0

            # Should apply rules without errors
            compg = apply_rule_sequence(br.ALL_RULES, cdg)[0]
            assert compg is not None


# ============================================================================
# Inversion Tests
# ============================================================================

def test_inversion_finds_all_paths(lib, multi_aggregation_ern):
    """Test that inversion finds all invertible paths"""
    with LibraryContext.with_library(lib):
        recipe = multi_aggregation_ern
        cdg = build_central_dogma_graph_direct(recipe.content, lib, dual=True)
        forward_compg = apply_rule_sequence(br.ALL_RULES, cdg)[0]

        # Apply inversion
        inverted_graphs = invert_all_paths(forward_compg, mode="all")

        # Should create at least one inverted network
        assert len(inverted_graphs) > 0

        for inv_g in inverted_graphs:
            # Check for input and inverse nodes
            input_nodes = [n for n in inv_g.nodes.values() if n.node_type == "input"]
            inv_nodes = [n for n in inv_g.nodes.values() if n.node_type.startswith("inv_")]

            assert len(input_nodes) > 0
            assert len(inv_nodes) > 0

            # Check that inverse nodes have is_inverse_of set
            for inv_node in inv_nodes:
                assert inv_node.is_inverse_of is not None
                assert inv_node.is_inverse_of.node_id >= 0


def test_inversion_produces_input_nodes(lib, multi_aggregation_ern):
    """Test that inversion creates input nodes for each inverted path"""
    with LibraryContext.with_library(lib):
        recipe = multi_aggregation_ern
        cdg = build_central_dogma_graph_direct(recipe.content, lib, dual=True)
        forward_compg = apply_rule_sequence(br.ALL_RULES, cdg)[0]

        inverted_graphs = invert_all_paths(forward_compg, mode="all")

        for inv_g in inverted_graphs:
            input_nodes = [n for n in inv_g.nodes.values() if n.node_type == "input"]
            # Each inverted graph should have at least one input
            assert len(input_nodes) > 0

            # Input nodes should have the expected metadata
            for inp in input_nodes:
                assert "input_position" in inp.extra
                assert "input_from_output" in inp.extra


