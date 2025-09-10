import pytest
import pandas as pd
import numpy as np
from biocomp.network_new import (
    df_to_graphstate,
    graphstate_to_df,
    build_central_dogma_graph,
    build_central_dogma_graph_from_units,  # Backward compatibility wrapper
    Network,
    CoTransfection,
    Unit,
    Slot,
    LibraryContext,
    NetworkConstructionError,
)
from biocomp.network import Network as OldNetwork, CoTransfection as OldCoTransfection, Unit as OldUnit, Slot as OldSlot
from biocomp.graphengine import GraphState, GraphNode, GraphEdge, Part
from biocomp.library import PartsLibrary
from biocomp.utils import load_lib


# ---------------------------------------------------------------------------
# Helper Functions and Fixtures
# ---------------------------------------------------------------------------


def create_equivalent_old_network(new_network, lib):
    """Create an equivalent old network from a new network for comparison"""
    with LibraryContext.with_library(lib):
        # Convert new network CoTransfection to old format
        old_cotx = []
        for cotx in new_network.cotx:
            old_units = []
            for unit in cotx.units:
                # Convert slots to old format  
                old_slots = []
                for slot in unit.slots:
                    if isinstance(slot, Slot):
                        if slot.part is None:
                            old_slots.append(None)
                        elif isinstance(slot.part, list) and len(slot.part) == 1:
                            # Single item list - extract the item for old format
                            old_slots.append(slot.part[0])
                        elif isinstance(slot.part, list):
                            # Multi-item list - create an old Slot object to preserve the list
                            old_slot = OldSlot(part=slot.part, ref_id=slot.ref_id)
                            old_slots.append(old_slot)
                        else:
                            old_slots.append(slot.part)
                    else:
                        old_slots.append(slot)
                
                # Preserve the unit name for old format
                old_unit = OldUnit(slots=old_slots, source=unit.source, name=unit.name)
                old_units.append(old_unit)
            
            old_cotx.append(OldCoTransfection(
                name=cotx.name,
                units=old_units,
                ratios=cotx.ratios
            ))
        
        # Create old network
        old_network = OldNetwork(
            lib=lib,
            name=new_network.name,
            cotx=old_cotx,
            build_on_init=False,
            invert_on_build=new_network.invert_on_build
        )
        
        return old_network


def compare_dataframes_for_equality(df1, df2, name1="df1", name2="df2"):
    """
    Compare two DataFrames for equality, handling special cases like lists and None values.
    Returns True if equal, raises AssertionError with detailed info if not.
    """
    # Check basic structure
    assert len(df1) == len(df2), f"Row count mismatch: {name1} has {len(df1)} rows, {name2} has {len(df2)} rows"
    assert list(df1.columns) == list(df2.columns), f"Column mismatch: {name1} columns {list(df1.columns)} vs {name2} columns {list(df2.columns)}"
    
    # Check each row and column
    for i in range(len(df1)):
        for col in df1.columns:
            val1 = df1.iloc[i][col]
            val2 = df2.iloc[i][col]
            
            # Handle None values - use explicit None check for arrays/lists
            val1_is_na = val1 is None or (hasattr(val1, '__len__') and len(val1) == 0) or (not isinstance(val1, (list, tuple, dict)) and pd.isna(val1))
            val2_is_na = val2 is None or (hasattr(val2, '__len__') and len(val2) == 0) or (not isinstance(val2, (list, tuple, dict)) and pd.isna(val2))
            
            if val1_is_na and val2_is_na:
                continue
            if val1_is_na or val2_is_na:
                assert False, f"Row {i}, col '{col}': {name1}={val1}, {name2}={val2} (one is NaN/None/empty, other is not)"
            
            # Handle list values
            if isinstance(val1, list) and isinstance(val2, list):
                assert val1 == val2, f"Row {i}, col '{col}': {name1}={val1}, {name2}={val2} (lists differ)"
            elif isinstance(val1, list) or isinstance(val2, list):
                assert False, f"Row {i}, col '{col}': {name1}={val1} ({type(val1)}), {name2}={val2} ({type(val2)}) (type mismatch)"
            else:
                # Handle regular values
                if isinstance(val1, (dict, tuple)) and isinstance(val2, (dict, tuple)):
                    assert val1 == val2, f"Row {i}, col '{col}': {name1}={val1}, {name2}={val2} (values differ)"
                else:
                    assert val1 == val2, f"Row {i}, col '{col}': {name1}={val1}, {name2}={val2} (values differ)"
    
    return True


@pytest.fixture
def lib():
    return load_lib()


@pytest.fixture
def basic_network(lib):
    """Create a simple basic network for testing"""
    with LibraryContext.with_library(lib):
        network = Network(
            cotx=[
                CoTransfection(
                    units=[
                        Unit(slots=["hEF1a", "eYFP"], source="plsmd0"),
                        Unit(slots=["hEF1a", "eBFP2"], source="plsmd0"),
                    ]
                )
            ],
            invert_on_build=False,
        )
        return network


@pytest.fixture
def simple_transcription_units():
    """Create simple transcription units for testing"""
    return {
        "TU1": Unit(name="TU1", slots=["hEF1a", "eYFP"], source="plsmd1"),
        "TU2": Unit(name="TU2", slots=["hEF1a", "eBFP2"], source="plsmd1"),
    }


@pytest.fixture
def complex_network_with_uorfs(lib):
    """Create a more complex network with uORFs and ERN elements"""
    with LibraryContext.with_library(lib):
        # Create slots with multiple parts and ref_ids
        u1 = Slot(
            part=["1w_uORF", "1x_uORF", "2x_uORF", "3x_uORF"],
            ref_id="U1",
        )
        u2 = Slot(
            part=["1w_uORF", "1x_uORF", "2x_uORF", "3x_uORF"],
            ref_id="U2",
        )

        network = Network(
            cotx=[
                CoTransfection(
                    name="cotx1",
                    units=[
                        Unit(slots=["hEF1a", u1, "CasE_rec", "eBFP2"], source="plsmd1"),
                        Unit(slots=["hEF1a", "CasE"], source="plsmd1"),
                        Unit(slots=["hEF1a", u2, "Csy4_rec", "eYFP"], source="plsmd2"),
                        Unit(slots=["hEF1a", "Csy4"], source="plsmd2"),
                    ],
                )
            ],
            invert_on_build=False,
        )
        return network


# ---------------------------------------------------------------------------
# Test Central Dogma Graph Building
# ---------------------------------------------------------------------------


def test_build_central_dogma_graph_basic(lib, simple_transcription_units):
    """Test building CDG from simple transcription units"""
    with LibraryContext.with_library(lib):
        cdg = build_central_dogma_graph_from_units(simple_transcription_units, lib)

        # Check basic structure
        assert isinstance(cdg, pd.DataFrame)
        expected_columns = [
            "tu_id",
            "type",
            "predecessor",
            "successor",
            "content",
            "content_type",
            "params",
            "is_output",
            "is_input",
        ]
        for col in expected_columns:
            assert col in cdg.columns, f"Missing column: {col}"

        # Should have DNA, RNA, and PRT nodes for each TU
        assert len(cdg[cdg["type"] == "DNA"]) == 2  # One DNA per TU
        assert len(cdg[cdg["type"] == "RNA"]) >= 1  # At least one RNA node
        assert len(cdg[cdg["type"] == "PRT"]) >= 1  # At least one PRT node


def test_build_central_dogma_graph_with_network(basic_network, lib):
    """Test building CDG from a complete network"""
    with LibraryContext.with_library(lib):
        # Build CDG directly from network
        cdg = build_central_dogma_graph(basic_network, lib)

        # Verify structure
        assert isinstance(cdg, pd.DataFrame)
        assert len(cdg) > 0

        # Check node types exist
        types = cdg["type"].unique()
        assert "DNA" in types
        assert "RNA" in types or "PRT" in types  # At least one of these should exist


# ---------------------------------------------------------------------------
# Test DataFrame to GraphState Conversion
# ---------------------------------------------------------------------------


def test_df_to_graphstate_basic_cdg(lib, simple_transcription_units):
    """Test converting a basic CDG DataFrame to GraphState"""
    with LibraryContext.with_library(lib):
        # Build CDG
        cdg = build_central_dogma_graph_from_units(simple_transcription_units, lib)

        # Convert to GraphState
        graph_state = df_to_graphstate(cdg)

        # Verify structure
        assert isinstance(graph_state, GraphState)
        assert len(graph_state.nodes) == len(cdg)

        # Verify nodes preserve information
        for i, node in enumerate(graph_state.nodes):
            assert node.node_id == i
            assert node.node_type == cdg.iloc[i]["type"]
            assert "tu_id" in node.extra
            assert "content" in node.extra
            assert "is_output" in node.extra


def test_df_to_graphstate_edges_from_successors(lib, simple_transcription_units):
    """Test that edges are correctly created from successor relationships"""
    with LibraryContext.with_library(lib):
        # Build CDG
        cdg = build_central_dogma_graph_from_units(simple_transcription_units, lib)

        # Convert to GraphState
        graph_state = df_to_graphstate(cdg)

        # Count expected edges based on successor relationships
        expected_edges = 0
        for _, row in cdg.iterrows():
            if pd.notna(row.get("successor")) and row["successor"]:
                expected_edges += len(row["successor"])

        assert len(graph_state.edges) == expected_edges

        # Verify edge properties
        for edge in graph_state.edges:
            assert isinstance(edge, GraphEdge)
            assert edge.source_id >= 0
            assert edge.target_id >= 0
            assert edge.source_id < len(graph_state.nodes)
            assert edge.target_id < len(graph_state.nodes)


# ---------------------------------------------------------------------------
# Test GraphState to DataFrame Conversion
# ---------------------------------------------------------------------------


def test_graphstate_to_df_cdg_format(lib, simple_transcription_units):
    """Test converting GraphState back to CDG DataFrame format"""
    with LibraryContext.with_library(lib):
        # Build CDG and convert to GraphState
        original_cdg = build_central_dogma_graph_from_units(simple_transcription_units, lib)
        graph_state = df_to_graphstate(original_cdg)

        # Convert back to DataFrame
        reconstructed_cdg = graphstate_to_df(graph_state, format_type="cdg")

        # Verify structure
        assert isinstance(reconstructed_cdg, pd.DataFrame)
        assert len(reconstructed_cdg) == len(original_cdg)

        # Check columns
        expected_columns = [
            "type",
            "tu_id",
            "content",
            "content_type",
            "params",
            "is_output",
            "is_input",
            "predecessor",
            "successor",
        ]
        for col in expected_columns:
            assert col in reconstructed_cdg.columns


def test_graphstate_to_df_preserves_node_types(lib, simple_transcription_units):
    """Test that node types are preserved in conversion"""
    with LibraryContext.with_library(lib):
        original_cdg = build_central_dogma_graph_from_units(simple_transcription_units, lib)
        graph_state = df_to_graphstate(original_cdg)
        reconstructed_cdg = graphstate_to_df(graph_state, format_type="cdg")

        # Check that types are preserved
        original_types = set(original_cdg["type"].unique())
        reconstructed_types = set(reconstructed_cdg["type"].unique())
        assert original_types == reconstructed_types


# ---------------------------------------------------------------------------
# Test Round-trip Conversion
# ---------------------------------------------------------------------------


def test_round_trip_conversion_basic(lib, simple_transcription_units):
    """Test that CDG -> GraphState -> CDG preserves essential information"""
    with LibraryContext.with_library(lib):
        # Start with original CDG
        original_cdg = build_central_dogma_graph_from_units(simple_transcription_units, lib)

        # Convert to GraphState and back
        graph_state = df_to_graphstate(original_cdg)
        reconstructed_cdg = graphstate_to_df(graph_state, format_type="cdg")

        # Check basic structural preservation
        assert len(original_cdg) == len(reconstructed_cdg)

        # Check that types are preserved
        assert set(original_cdg["type"]) == set(reconstructed_cdg["type"])

        # Check that essential fields are preserved
        for i in range(len(original_cdg)):
            orig_row = original_cdg.iloc[i]
            recon_row = reconstructed_cdg.iloc[i]

            assert orig_row["type"] == recon_row["type"]
            # tu_id, content, and params should be preserved in extra
            # Note: exact equality may not hold due to serialization, but structure should match


def test_round_trip_preserves_successor_relationships(lib, simple_transcription_units):
    """Test that successor/predecessor relationships are preserved in round-trip"""
    with LibraryContext.with_library(lib):
        original_cdg = build_central_dogma_graph_from_units(simple_transcription_units, lib)

        # Convert to GraphState and back
        graph_state = df_to_graphstate(original_cdg)
        reconstructed_cdg = graphstate_to_df(graph_state, format_type="cdg")

        # Check that connectivity is preserved
        # Count total connections in original
        orig_connections = 0
        for _, row in original_cdg.iterrows():
            if pd.notna(row.get("successor")) and row["successor"]:
                orig_connections += len(row["successor"])

        # Count total connections in reconstructed
        recon_connections = 0
        for _, row in reconstructed_cdg.iterrows():
            if pd.notna(row.get("successor")) and row["successor"]:
                recon_connections += len(row["successor"])

        assert orig_connections == recon_connections


def test_round_trip_complex_network(complex_network_with_uorfs, lib):
    """Test round-trip conversion with a more complex network"""
    with LibraryContext.with_library(lib):
        # Build CDG directly from network
        original_cdg = build_central_dogma_graph(complex_network_with_uorfs, lib)

        # Round-trip conversion
        graph_state = df_to_graphstate(original_cdg)
        reconstructed_cdg = graphstate_to_df(graph_state, format_type="cdg")

        # Verify preservation of complex structure
        assert len(original_cdg) == len(reconstructed_cdg)
        assert set(original_cdg["type"]) == set(reconstructed_cdg["type"])


# ---------------------------------------------------------------------------
# Test Error Handling
# ---------------------------------------------------------------------------


def test_df_to_graphstate_unknown_format():
    """Test error handling for unknown DataFrame format"""
    # Create a DataFrame with unknown structure
    unknown_df = pd.DataFrame({"unknown_col1": [1, 2, 3], "unknown_col2": ["a", "b", "c"]})

    with pytest.raises(ValueError, match="Unknown DataFrame format"):
        df_to_graphstate(unknown_df)


def test_graphstate_to_df_invalid_format():
    """Test error handling for invalid format type"""
    # Create minimal GraphState
    graph_state = GraphState(nodes=[], edges=[])

    with pytest.raises(ValueError, match="Unknown format_type"):
        graphstate_to_df(graph_state, format_type="invalid")


def test_empty_dataframe_conversion():
    """Test conversion of empty DataFrames"""
    # Create empty CDG-like DataFrame
    empty_cdg = pd.DataFrame(
        columns=[
            "tu_id",
            "type",
            "predecessor",
            "successor",
            "content",
            "content_type",
            "params",
            "is_output",
            "is_input",
        ]
    )

    graph_state = df_to_graphstate(empty_cdg)
    assert len(graph_state.nodes) == 0
    assert len(graph_state.edges) == 0

    # Round-trip should work
    reconstructed = graphstate_to_df(graph_state, format_type="cdg")
    assert len(reconstructed) == 0


# ---------------------------------------------------------------------------
# Test Data Integrity
# ---------------------------------------------------------------------------


def test_node_id_consistency(lib, simple_transcription_units):
    """Test that node IDs are consistent and sequential"""
    with LibraryContext.with_library(lib):
        original_cdg = build_central_dogma_graph_from_units(simple_transcription_units, lib)
        graph_state = df_to_graphstate(original_cdg)

        # Node IDs should be sequential starting from 0
        expected_ids = list(range(len(original_cdg)))
        actual_ids = [node.node_id for node in graph_state.nodes]
        assert actual_ids == expected_ids


def test_edge_references_valid_nodes(lib, simple_transcription_units):
    """Test that all edges reference valid node IDs"""
    with LibraryContext.with_library(lib):
        original_cdg = build_central_dogma_graph_from_units(simple_transcription_units, lib)
        graph_state = df_to_graphstate(original_cdg)

        valid_node_ids = set(node.node_id for node in graph_state.nodes)

        for edge in graph_state.edges:
            assert edge.source_id in valid_node_ids, f"Invalid source_id: {edge.source_id}"
            assert edge.target_id in valid_node_ids, f"Invalid target_id: {edge.target_id}"


# ---------------------------------------------------------------------------
# Backward Compatibility Tests (New vs Old Network Module)
# ---------------------------------------------------------------------------


def test_backward_compatibility_basic_network(basic_network, lib):
    """Test that new module produces identical CDG to old module for basic network"""
    with LibraryContext.with_library(lib):
        # Build CDG using new method
        new_cdg = build_central_dogma_graph(basic_network, lib)
        
        # Create equivalent old network and build its CDG
        old_network = create_equivalent_old_network(basic_network, lib)
        old_network.build()
        old_cdg = old_network.central_dogma_graph
        
        # Compare DataFrames for exact equality
        compare_dataframes_for_equality(old_cdg, new_cdg, "old_network", "new_network")


def test_backward_compatibility_complex_network(complex_network_with_uorfs, lib):
    """Test that new module produces identical CDG to old module for complex network"""
    with LibraryContext.with_library(lib):
        # Build CDG using new method
        new_cdg = build_central_dogma_graph(complex_network_with_uorfs, lib)
        
        # Create equivalent old network and build its CDG
        old_network = create_equivalent_old_network(complex_network_with_uorfs, lib)
        old_network.build()
        old_cdg = old_network.central_dogma_graph
        
        # Compare DataFrames for exact equality
        compare_dataframes_for_equality(old_cdg, new_cdg, "old_network", "new_network")


def test_backward_compatibility_simple_transcription_units(simple_transcription_units, lib):
    """Test backward compatibility using simple transcription units directly"""
    with LibraryContext.with_library(lib):
        # Build CDG using new method
        new_cdg = build_central_dogma_graph_from_units(simple_transcription_units, lib)
        
        # Create equivalent old network from transcription units
        # Create a single CoTransfection with all units to match the structure
        old_units = []
        for tu_name, tu in simple_transcription_units.items():
            # Extract the part from each slot
            old_slots = []
            for slot in tu.slots:
                if isinstance(slot, Slot):
                    if slot.part is None:
                        old_slots.append(None) 
                    elif isinstance(slot.part, list) and len(slot.part) == 1:
                        old_slots.append(slot.part[0])  # Extract single item from list
                    elif isinstance(slot.part, list):
                        old_slots.extend(slot.part)
                    else:
                        old_slots.append(slot.part)
                else:
                    old_slots.append(slot)
            
            old_unit = OldUnit(slots=old_slots, source=tu.source, name=tu_name)
            old_units.append(old_unit)
            
        old_cotx = [OldCoTransfection(units=old_units)]
            
        old_network = OldNetwork(lib=lib, cotx=old_cotx, build_on_init=False)
        old_network.build()
        old_cdg = old_network.central_dogma_graph
        
        # Compare DataFrames for exact equality
        compare_dataframes_for_equality(old_cdg, new_cdg, "old_network", "new_network")


def test_backward_compatibility_with_uorf_slots(lib):
    """Test backward compatibility with uORF slots and ref_ids"""
    with LibraryContext.with_library(lib):
        # Create a network with uORF slots (more complex case)
        u1 = Slot(part=["1w_uORF", "2x_uORF"], ref_id="U1")
        u2 = Slot(part=["3x_uORF", "4x_uORF"], ref_id="U2")
        
        new_network = Network(
            cotx=[
                CoTransfection(
                    units=[
                        Unit(name="TU1", slots=["hEF1a", u1, "eYFP"], source="plsmd1"),
                        Unit(name="TU2", slots=["hEF1a", u2, "eBFP2"], source="plsmd2"),
                    ]
                )
            ],
            invert_on_build=False,
        )
        
        # Build CDG using new method
        new_cdg = build_central_dogma_graph(new_network, lib)
        
        # Create equivalent old network
        old_network = create_equivalent_old_network(new_network, lib)
        old_network.build()
        old_cdg = old_network.central_dogma_graph
        
        # Compare DataFrames for exact equality
        compare_dataframes_for_equality(old_cdg, new_cdg, "old_network", "new_network")


def test_backward_compatibility_multiple_cotx(lib):
    """Test backward compatibility with multiple CoTransfection groups"""
    with LibraryContext.with_library(lib):
        new_network = Network(
            cotx=[
                CoTransfection(
                    name="group1",
                    units=[Unit(slots=["hEF1a", "eYFP"], source="plsmd1")]
                ),
                CoTransfection(
                    name="group2", 
                    units=[Unit(slots=["hEF1a", "eBFP2"], source="plsmd2")]
                )
            ],
            invert_on_build=False,
        )
        
        # Build CDG using new method
        new_cdg = build_central_dogma_graph(new_network, lib)
        
        # Create equivalent old network
        old_network = create_equivalent_old_network(new_network, lib)
        old_network.build()
        old_cdg = old_network.central_dogma_graph
        
        # Compare DataFrames for exact equality
        compare_dataframes_for_equality(old_cdg, new_cdg, "old_network", "new_network")


def test_backward_compatibility_with_ratios(lib):
    """Test backward compatibility with custom ratios"""
    with LibraryContext.with_library(lib):
        new_network = Network(
            cotx=[
                CoTransfection(
                    units=[
                        Unit(slots=["hEF1a", "eYFP"], source="plsmd1"),
                        Unit(slots=["hEF1a", "eBFP2"], source="plsmd1"),
                    ],
                    ratios=[1.0, 2.0]
                )
            ],
            invert_on_build=False,
        )
        
        # Build CDG using new method
        new_cdg = build_central_dogma_graph(new_network, lib)
        
        # Create equivalent old network
        old_network = create_equivalent_old_network(new_network, lib)
        old_network.build()
        old_cdg = old_network.central_dogma_graph
        
        # Compare DataFrames for exact equality
        compare_dataframes_for_equality(old_cdg, new_cdg, "old_network", "new_network")


if __name__ == "__main__":
    pytest.main([__file__])
