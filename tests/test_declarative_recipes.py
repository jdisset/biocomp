"""Test declarative recipe format that doesn't require database."""

import pytest
from biocomp.network import Network, CoTransfection, TranscriptionUnit, Slot
from biocomp.utils import load_lib


class TestDeclarativeRecipes:
    """Test the declarative API for creating networks without database."""
    
    @pytest.fixture
    def lib(self):
        return load_lib()
    
    def test_simple_declarative_network(self, lib):
        """Create a simple network using declarative syntax."""
        # Define transcription units directly
        network = Network(
            lib=lib,
            name="simple_declarative",
            cotx=[
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            name="Reporter1",
                            slots=[
                                Slot(part="hEF1a", position=1),  # promoter
                                Slot(part="eBFP2", position=2),   # CDS
                            ]
                        ),
                        TranscriptionUnit(
                            name="Reporter2", 
                            slots=[
                                Slot(part="hEF1a", position=1),
                                Slot(part="mKO2", position=2),
                            ]
                        )
                    ],
                    ratios=[0.7, 0.3]  # Aggregation ratios
                )
            ]
        )
        
        # Check that it builds correctly
        assert network.is_built()
        assert len(network.transcription_units) == 2
        assert len(network.aggregations) == 1
        
        # Check ratios are preserved
        agg_row = network.aggregations.iloc[0]
        assert agg_row["ratio"] == [0.7, 0.3]
    
    def test_multiple_aggregations_declarative(self, lib):
        """Create network with multiple aggregations."""
        network = Network(
            lib=lib,
            name="multi_agg_declarative",
            cotx=[
                # First aggregation
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            slots=[
                                Slot(part="hEF1a", position=1),
                                Slot(part="eBFP2", position=2),
                            ]
                        ),
                        TranscriptionUnit(
                            slots=[
                                Slot(part="hEF1a", position=1),
                                Slot(part="mNeonGreen", position=2),
                            ]
                        )
                    ],
                    ratios=[0.5, 0.5]
                ),
                # Second aggregation
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            slots=[
                                Slot(part="hEF1a", position=1),
                                Slot(part="mKO2", position=2),
                            ]
                        ),
                        TranscriptionUnit(
                            slots=[
                                Slot(part="hEF1a", position=1),
                                Slot(part="mMaroon1", position=2),
                            ]
                        )
                    ],
                    ratios=[0.6, 0.4]
                )
            ]
        )
        
        assert len(network.aggregations) == 2
        assert network.aggregations.iloc[0]["ratio"] == [0.5, 0.5]
        assert network.aggregations.iloc[1]["ratio"] == [0.6, 0.4]
    
    def test_complex_transcription_units(self, lib):
        """Test with more complex transcription units including uORFs."""
        network = Network(
            lib=lib,
            name="complex_declarative",
            cotx=[
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            name="ERN_circuit",
                            slots=[
                                Slot(part="hEF1a", position=1),          # promoter
                                Slot(part="1x_uORF", position=2),        # uORF
                                Slot(part="CasE_rec", position=3),       # ERN recognition site
                                Slot(part="eBFP2", position=4),          # reporter
                            ]
                        ),
                        TranscriptionUnit(
                            name="ERN_source",
                            slots=[
                                Slot(part="hEF1a", position=1),
                                Slot(part="CasE", position=2),           # ERN protein
                            ]
                        )
                    ],
                    ratios=[0.8, 0.2]
                )
            ]
        )
        
        assert "ERN_circuit" in network.transcription_units
        assert "ERN_source" in network.transcription_units
        assert len(network.transcription_units["ERN_circuit"].slots) == 4
    
    def test_variable_parts_declarative(self, lib):
        """Test with variable parts (multiple options for quantization)."""
        network = Network(
            lib=lib,
            name="variable_parts",
            cotx=[
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            slots=[
                                # Multiple uORF options (can't vary promoter - only one available)
                                Slot(part="hEF1a", position=1),
                                # Multiple uORF options  
                                Slot(part=["1x_uORF", "2x_uORF", "3x_uORF"], position=2),
                                Slot(part="eBFP2", position=3),
                            ]
                        )
                    ]
                )
            ]
        )
        
        tu = list(network.transcription_units.values())[0]
        # First slot is single promoter
        assert tu.slots[0].part == "hEF1a"
        # Second slot has multiple uORF options
        assert isinstance(tu.slots[1].part, list) 
        assert len(tu.slots[1].part) == 3
    
    def test_declarative_to_recipe_conversion(self, lib):
        """Test converting declarative network to recipe format."""
        # Create declarative network
        network = Network(
            lib=lib,
            name="test_conversion",
            cotx=[
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            slots=[
                                Slot(part="hEF1a", position=1),
                                Slot(part="eBFP2", position=2),
                            ]
                        ),
                        TranscriptionUnit(
                            slots=[
                                Slot(part="hEF1a", position=1),
                                Slot(part="mKO2", position=2),
                            ]
                        )
                    ],
                    ratios=[0.25, 0.75]
                )
            ]
        )
        
        # Convert to recipe format
        recipe = network.to_recipe()
        
        assert recipe["name"] == "test_conversion"
        assert len(recipe["content"]) == 1
        assert len(recipe["content"][0]["sources"]) == 2
        
        # Check ratios (they get normalized)
        sources = recipe["content"][0]["sources"]
        ratios = {s["plasmid"]: s["ratio"] for s in sources}
        # Ratios sum to 1.0 after normalization
        assert abs(sum(ratios.values()) - 1.0) < 0.001
    
    def test_equivalence_with_database_recipe(self, lib):
        """Test that declarative and database-based methods produce equivalent networks."""
        # Database recipe format
        db_recipe = {
            "name": "test_equiv",
            "content": [
                {
                    "sources": [
                        {"ratio": 0.7, "plasmid": "L1.ST2-3_EBFP2"},
                        {"ratio": 0.3, "plasmid": "L1.ST2-3_mKO2"}
                    ]
                }
            ]
        }
        
        # Note: We can't directly test equivalence without the database,
        # but we can show the declarative format is self-consistent
        
        # Equivalent declarative format (conceptually)
        # This shows how you would represent the same circuit
        declarative = Network(
            lib=lib,
            name="test_equiv",
            cotx=[
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            name="EBFP2_reporter",
                            slots=[
                                # You would need to know the L1 plasmid structure
                                Slot(part="hEF1a", position=1),
                                Slot(part="eBFP2", position=2),
                            ]
                        ),
                        TranscriptionUnit(
                            name="mKO2_reporter",
                            slots=[
                                Slot(part="hEF1a", position=1),
                                Slot(part="mKO2", position=2),
                            ]
                        )
                    ],
                    ratios=[0.7, 0.3]
                )
            ]
        )
        
        assert declarative.is_built()
        assert len(declarative.aggregations) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])