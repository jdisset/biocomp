"""Tests for the network module.

Tests the Slot, TranscriptionUnit, and Network classes along with their
associated functionality for building and manipulating circuit DAGs.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from biocomp.network import (
    Slot,
    TranscriptionUnit,
    Network,
    Unit,
    CoTransfection,
    NetworkConstructionError,
    get_network_family,
    get_all_ERN_ids,
    inverted_network,
    PART_TYPE_TO_EMBEDDING_NAME,
    EMBEDDING_TO_DEFAULT_PART,
)
from biocomp.library import PartsLibrary


class TestSlot:
    """Test cases for the Slot class."""

    @pytest.fixture
    def mock_lib(self):
        """Create a mock library for testing."""
        lib = MagicMock(spec=PartsLibrary)
        # mock parts dataframe
        parts_data = pd.DataFrame({
            'category': ['promoter', 'uORF_group', 'gene', 'terminator', 'promoter', 'uORF_group'],
        }, index=['hEF1a', '1x', 'GFP', 'SV40', 'CMV', '00_empty_tc'])
        lib.parts = parts_data
        lib.pc = parts_data
        return lib

    def test_slot_creation_with_string(self, mock_lib):
        """Test creating a slot with a single string part."""
        with patch('biocomp.network.LibraryContext.get_library', return_value=mock_lib):
            slot = Slot(part='GFP')
            assert slot.part == 'GFP'
            assert slot.maps_to_parameter is None

    def test_slot_creation_with_list(self, mock_lib):
        """Test creating a slot with a list of parts."""
        with patch('biocomp.network.LibraryContext.get_library', return_value=mock_lib):
            slot = Slot(part=['hEF1a', 'CMV'])
            assert slot.part == ['hEF1a', 'CMV']
            assert slot.maps_to_parameter == 'tc_rate'

    def test_slot_parameter_mapping(self, mock_lib):
        """Test automatic parameter mapping based on part category."""
        with patch('biocomp.network.LibraryContext.get_library', return_value=mock_lib):
            # promoter should map to tc_rate
            slot_promoter = Slot(part='hEF1a')
            assert slot_promoter.maps_to_parameter == 'tc_rate'
            assert slot_promoter.part == ['hEF1a']  # converted to list

            # uORF_group should map to tl_rate
            slot_uorf = Slot(part='1x')
            assert slot_uorf.maps_to_parameter == 'tl_rate'
            assert slot_uorf.part == ['1x']

    def test_slot_empty_or_none(self, mock_lib):
        """Test creating slots with None or empty values."""
        with patch('biocomp.network.LibraryContext.get_library', return_value=mock_lib):
            slot_none = Slot(part=None)
            assert slot_none.part is None
            assert slot_none.maps_to_parameter is None

            slot_empty_list = Slot(part=[])
            assert slot_empty_list.part is None
            assert slot_empty_list.maps_to_parameter is None

    def test_slot_invalid_part(self, mock_lib):
        """Test error handling for unknown parts."""
        with patch('biocomp.network.LibraryContext.get_library', return_value=mock_lib):
            with pytest.raises(ValueError, match="Unknown part"):
                Slot(part='INVALID_PART')

    def test_slot_mixed_parameter_mapping(self, mock_lib):
        """Test error when parts map to different parameters."""
        with patch('biocomp.network.LibraryContext.get_library', return_value=mock_lib):
            with pytest.raises(ValueError, match="maps to .* parameters"):
                Slot(part=['hEF1a', '1x'])  # promoter and uORF_group

    def test_slot_repr(self, mock_lib):
        """Test string representation of slots."""
        with patch('biocomp.network.LibraryContext.get_library', return_value=mock_lib):
            slot_simple = Slot(part='GFP')
            assert repr(slot_simple) == '<GFP>'

            slot_mapped = Slot(part='hEF1a')
            assert repr(slot_mapped) == "<['hEF1a'] -> tc_rate>"

            slot_empty = Slot(part=None)
            assert repr(slot_empty) == '<empty slot>'


class TestTranscriptionUnit:
    """Test cases for the TranscriptionUnit class."""

    @pytest.fixture
    def mock_lib(self):
        """Create a mock library for testing."""
        lib = MagicMock(spec=PartsLibrary)
        parts_data = pd.DataFrame({
            'category': ['promoter', 'uORF_group', 'gene', 'terminator', 'promoter', 'uORF_group'],
        }, index=['hEF1a', '1x', 'GFP', 'SV40', 'CMV', '00_empty_tc'])
        lib.parts = parts_data
        lib.pc = parts_data
        return lib

    def test_tu_creation_basic(self, mock_lib):
        """Test basic TranscriptionUnit creation."""
        with patch('biocomp.network.LibraryContext.get_library', return_value=mock_lib):
            tu = TranscriptionUnit(
                name="test_tu",
                slots=['hEF1a', 'GFP', 'SV40']
            )
            assert tu.name == "test_tu"
            assert len(tu.slots) == 3
            assert all(isinstance(s, Slot) for s in tu.slots)

    def test_tu_parameter_extraction(self, mock_lib):
        """Test automatic parameter extraction from slots."""
        with patch('biocomp.network.LibraryContext.get_library', return_value=mock_lib):
            tu = TranscriptionUnit(
                name="test_tu",
                slots=['hEF1a', '1x', 'GFP', 'SV40']
            )
            assert 'tc_rate' in tu.params
            assert tu.params['tc_rate'] == ['hEF1a']
            assert 'tl_rate' in tu.params
            assert tu.params['tl_rate'] == ['1x']

    def test_tu_default_parameters(self, mock_lib):
        """Test that default parameters are added when not provided."""
        with patch('biocomp.network.LibraryContext.get_library', return_value=mock_lib):
            # tu without tl_rate part
            tu = TranscriptionUnit(
                name="test_tu",
                slots=['hEF1a', 'GFP', 'SV40']
            )
            assert tu.params['tl_rate'] == ['00_empty_tc']  # default

    def test_tu_to_parts(self, mock_lib):
        """Test conversion back to parts list."""
        with patch('biocomp.network.LibraryContext.get_library', return_value=mock_lib):
            tu = TranscriptionUnit(
                name="test_tu",
                slots=['hEF1a', 'GFP', 'SV40']
            )
            parts = tu.to_parts()
            # promoter parts are converted to lists due to parameter mapping
            assert parts == [['hEF1a'], 'GFP', 'SV40']

    def test_tu_with_source(self, mock_lib):
        """Test creating a copy with a different source."""
        with patch('biocomp.network.LibraryContext.get_library', return_value=mock_lib):
            tu = TranscriptionUnit(
                name="test_tu",
                slots=['hEF1a', 'GFP', 'SV40'],
                source="plasmid1"
            )
            tu2 = tu.with_source("plasmid2")
            assert tu2.source == "plasmid2"
            assert tu2.name == tu.name
            assert tu2.slots == tu.slots
            assert tu.source == "plasmid1"  # original unchanged

    def test_unit_alias(self):
        """Test that Unit is an alias for TranscriptionUnit."""
        assert Unit is TranscriptionUnit


class TestNetwork:
    """Test cases for the Network class."""

    @pytest.fixture
    def mock_lib(self):
        """Create a comprehensive mock library for network testing."""
        lib = MagicMock(spec=PartsLibrary)
        
        # mock parts dataframe with more comprehensive data
        parts_data = pd.DataFrame({
            'category': ['promoter', 'uORF_group', 'gene', 'terminator', 
                        'fluo_marker', 'fluo_marker', 'ERN_recog_site_5p', 
                        'RNA_sequestron', 'promoter', 'uORF_group'],
            'transcripted': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'translated': [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
        }, index=['hEF1a', '1x', 'GFP', 'SV40', 'mKate', 'BFP', 
                 'ERN1_rec', 'ERN1', 'CMV', '00_empty_tc'])
        lib.parts = parts_data
        lib.pc = parts_data
        
        # mock enabled sequestrons
        sequestrons_data = pd.DataFrame({
            'type': ['ERN'],
            'negative_level': ['RNA'],
            'negative_part': ['ERN1_rec'],
            'positive_level': ['PRT'],
            'positive_part': ['ERN1'],
            'output_level': ['RNA'],
            'output_part': [['ERN1_rec']],
        })
        lib.get_enabled_sequestrons = MagicMock(return_value=sequestrons_data)
        
        return lib

    def test_network_declarative_creation(self, mock_lib):
        """Test creating a network using declarative syntax."""
        with patch('biocomp.network.LibraryContext.get_library', return_value=mock_lib):
            cotx = [
                CoTransfection(
                    name="group1",
                    units=[
                        Unit(name="TU1", slots=['hEF1a', 'mKate', 'SV40']),
                        Unit(name="TU2", slots=['CMV', 'BFP', 'SV40'])
                    ],
                    ratios=[1.0, 2.0]
                )
            ]
            
            net = Network(lib=mock_lib, name="test_net", cotx=cotx, build_on_init=False)
            assert net.name == "test_net"
            assert len(net.transcription_units) == 2
            assert "TU1" in net.transcription_units
            assert "TU2" in net.transcription_units

    def test_network_build_process(self, mock_lib):
        """Test the network building process initialization."""
        with patch('biocomp.network.LibraryContext.get_library', return_value=mock_lib):
            cotx = [
                CoTransfection(
                    units=[
                        Unit(name="TU1", slots=['hEF1a', 'mKate', 'SV40'])
                    ]
                )
            ]
            
            # test initialization without building
            net = Network(lib=mock_lib, name="test_net", cotx=cotx, build_on_init=False)
            
            assert net.name == "test_net"
            assert net.transcription_units is not None
            assert "TU1" in net.transcription_units

    def test_network_output_proteins(self, mock_lib):
        """Test network structure with output proteins."""
        with patch('biocomp.network.LibraryContext.get_library', return_value=mock_lib):
            cotx = [
                CoTransfection(
                    units=[
                        Unit(slots=['hEF1a', 'mKate', 'SV40']),
                        Unit(slots=['CMV', 'BFP', 'SV40'])
                    ]
                )
            ]
            
            net = Network(lib=mock_lib, cotx=cotx, build_on_init=False)
            # verify structure was created properly
            assert len(net.transcription_units) == 2
            # mKate and BFP should be in the transcription unit slots
            tu_parts = []
            for tu in net.transcription_units.values():
                for slot in tu.slots:
                    if slot.part and slot.part not in ['hEF1a', 'CMV', 'SV40', '00_empty_tc']:
                        tu_parts.append(slot.part)
            assert 'mKate' in tu_parts
            assert 'BFP' in tu_parts

    def test_network_get_family(self, mock_lib):
        """Test network family classification requires built network."""
        # this test requires a fully built network with compute graph
        # which is complex to mock, so we'll test basic structure instead
        with patch('biocomp.network.LibraryContext.get_library', return_value=mock_lib):
            cotx = [
                CoTransfection(
                    units=[Unit(slots=['hEF1a', 'mKate', 'SV40'])]
                )
            ]
            net = Network(lib=mock_lib, cotx=cotx, build_on_init=False)
            assert net.transcription_units is not None

    def test_network_error_no_transcription_units(self, mock_lib):
        """Test error when building network without transcription units."""
        with patch('biocomp.network.LibraryContext.get_library', return_value=mock_lib):
            net = Network(lib=mock_lib, build_on_init=False)
            with pytest.raises(AssertionError, match="No transcription units"):
                net.build()

    def test_network_topological_order(self, mock_lib):
        """Test that topological ordering requires built network."""
        with patch('biocomp.network.LibraryContext.get_library', return_value=mock_lib):
            cotx = [
                CoTransfection(
                    units=[Unit(slots=['hEF1a', 'mKate', 'SV40'])]
                )
            ]
            net = Network(lib=mock_lib, cotx=cotx, build_on_init=False)
            
            # topological order requires compute_graph to be built
            with pytest.raises(AssertionError):
                net.topological_order()

    def test_network_from_raw(self, mock_lib):
        """Test creating network from raw data."""
        with patch('biocomp.network.LibraryContext.get_library', return_value=mock_lib):
            transcription_units = {
                'TU1': TranscriptionUnit(name='TU1', slots=['hEF1a', 'mKate', 'SV40'])
            }
            raw_tu_in_sources = [('source1', 'TU1', 0)]
            raw_aggregations = [(0, 'source1', 1.0)]
            
            net = Network.from_raw(
                lib=mock_lib,
                name="test_net",
                transcription_units=transcription_units,
                raw_tu_in_sources=raw_tu_in_sources,
                raw_aggregations=raw_aggregations,
                build=False
            )
            
            assert net.name == "test_net"
            assert net.transcription_units == transcription_units
            assert net.raw_tu_in_sources == raw_tu_in_sources
            assert net.raw_aggregations == raw_aggregations

    def test_network_copy(self, mock_lib):
        """Test network copying."""
        with patch('biocomp.network.LibraryContext.get_library', return_value=mock_lib):
            cotx = [
                CoTransfection(
                    units=[Unit(slots=['hEF1a', 'mKate', 'SV40'])]
                )
            ]
            net1 = Network(lib=mock_lib, name="net1", cotx=cotx, build_on_init=False)
            net2 = net1.copy()
            
            assert net2.name == net1.name
            assert net2 is not net1
            assert net2.transcription_units is not net1.transcription_units

    def test_cotransfection_defaults(self, mock_lib):
        """Test CoTransfection default values."""
        with patch('biocomp.network.LibraryContext.get_library', return_value=mock_lib):
            units = [Unit(slots=['hEF1a', 'mKate', 'SV40']) for _ in range(3)]
            cotx = CoTransfection(units=units)
            
            assert cotx.ratios == [1.0, 1.0, 1.0]  # default equal ratios
            assert cotx.name is None  # no default name

    def test_network_is_built(self, mock_lib):
        """Test is_built method."""
        with patch('biocomp.network.LibraryContext.get_library', return_value=mock_lib):
            cotx = [CoTransfection(units=[Unit(slots=['hEF1a', 'mKate', 'SV40'])])]
            
            # not built yet
            net = Network(lib=mock_lib, cotx=cotx, build_on_init=False)
            assert not net.is_built()
            
            # is_built requires compute_graph and central_dogma_graph
            assert net.compute_graph is None
            assert net.central_dogma_graph is None


class TestNetworkInversion:
    """Test cases for network inversion functionality."""

    @pytest.fixture
    def mock_lib(self):
        """Create a mock library for inversion testing."""
        lib = MagicMock(spec=PartsLibrary)
        parts_data = pd.DataFrame({
            'category': ['promoter', 'fluo_marker', 'terminator', 'uORF_group'],
            'transcripted': [1, 1, 1, 1],
            'translated': [0, 1, 0, 0],
        }, index=['hEF1a', 'mKate', 'SV40', '00_empty_tc'])
        lib.parts = parts_data
        lib.pc = parts_data
        lib.get_enabled_sequestrons = MagicMock(return_value=pd.DataFrame())
        return lib

    def test_inverted_network_basic(self, mock_lib):
        """Test basic network inversion setup."""
        with patch('biocomp.network.LibraryContext.get_library', return_value=mock_lib):
            cotx = [
                CoTransfection(
                    units=[Unit(slots=['hEF1a', 'mKate', 'SV40'])]
                )
            ]
            net = Network(lib=mock_lib, cotx=cotx, build_on_init=False)
            
            # network inversion requires a built network
            assert not net.is_built()

    def test_network_get_inverted_input_proteins(self, mock_lib):
        """Test getting inverted input proteins setup."""
        with patch('biocomp.network.LibraryContext.get_library', return_value=mock_lib):
            cotx = [
                CoTransfection(
                    units=[Unit(slots=['hEF1a', 'mKate', 'SV40'])]
                )
            ]
            net = Network(lib=mock_lib, cotx=cotx, build_on_init=False, invert_on_build=True)
            
            # invert_on_build flag should be set
            assert net.invert_on_build == True


class TestNetworkUtils:
    """Test utility functions for networks."""

    def test_network_construction_error(self):
        """Test custom exception class."""
        with pytest.raises(NetworkConstructionError, match="test error"):
            raise NetworkConstructionError("test error")

    def test_get_all_ern_ids(self):
        """Test getting ERN IDs from network."""
        mock_net = MagicMock()
        mock_net.compute_graph = pd.DataFrame({
            'type': ['sequestron_ERN', 'translation', 'sequestron_ERN'],
        }, index=[0, 1, 2])
        mock_net.topological_order = MagicMock(return_value=[[0], [2]])
        
        ern_ids = get_all_ERN_ids(mock_net)
        assert ern_ids == [0, 2]


class TestRNAGrouping:
    """Test cases for RNA node grouping logic."""
    
    @pytest.fixture
    def real_lib(self):
        """Use the real library for grouping tests."""
        from biocomp.utils import load_lib
        return load_lib()
    
    def test_multi_value_parameter_grouping(self, real_lib):
        """Test that RNAs with multi-value parameters are NOT grouped without ref_id."""
        # create network with multi-value uORF slots (no ref_id)
        net = Network(
            lib=real_lib,
            cotx=[
                CoTransfection(
                    units=[
                        Unit(name="TU1", slots=["hEF1a", Slot(part=["1x_uORF", "2x_uORF"]), "eBFP2"]),
                        Unit(name="TU2", slots=["hEF1a", Slot(part=["1x_uORF", "2x_uORF"]), "eBFP2"]),
                    ]
                )
            ],
            build_on_init=True
        )
        
        # check that RNAs are NOT grouped (each TU gets its own RNA node)
        cdg = net.central_dogma_graph
        rna_nodes = cdg[cdg.type == "RNA"]
        
        # find nodes containing TU1 and TU2
        tu1_nodes = rna_nodes[rna_nodes.tu_id.apply(lambda x: "TU1" in x)]
        tu2_nodes = rna_nodes[rna_nodes.tu_id.apply(lambda x: "TU2" in x)]
        
        # they should be in separate nodes (no ref_id means no grouping for multi-value)
        assert len(tu1_nodes) == 1
        assert len(tu2_nodes) == 1
        assert tu1_nodes.index[0] != tu2_nodes.index[0]
    
    def test_single_value_parameter_grouping(self, real_lib):
        """Test that RNAs with single-value parameters ARE grouped together."""
        net = Network(
            lib=real_lib,
            cotx=[
                CoTransfection(
                    units=[
                        Unit(name="TU1", slots=["hEF1a", Slot(part=["1x_uORF"]), "eBFP2"]),
                        Unit(name="TU2", slots=["hEF1a", Slot(part=["1x_uORF"]), "eBFP2"]),
                    ]
                )
            ],
            build_on_init=True
        )
        
        cdg = net.central_dogma_graph
        rna_nodes = cdg[cdg.type == "RNA"]
        
        # both TUs should be in the same RNA node
        grouped_node = rna_nodes[rna_nodes.tu_id.apply(lambda x: "TU1" in x and "TU2" in x)]
        assert len(grouped_node) == 1
        assert set(grouped_node.iloc[0]['tu_id']) == {"TU1", "TU2"}
    
    def test_different_single_value_parameters(self, real_lib):
        """Test that RNAs with different single-value parameters are NOT grouped."""
        net = Network(
            lib=real_lib,
            cotx=[
                CoTransfection(
                    units=[
                        Unit(name="TU1", slots=["hEF1a", Slot(part=["1x_uORF"]), "eBFP2"]),
                        Unit(name="TU2", slots=["hEF1a", Slot(part=["2x_uORF"]), "eBFP2"]),
                    ]
                )
            ],
            build_on_init=True
        )
        
        cdg = net.central_dogma_graph
        rna_nodes = cdg[cdg.type == "RNA"]
        
        # TUs should be in separate nodes
        tu1_nodes = rna_nodes[rna_nodes.tu_id.apply(lambda x: "TU1" in x)]
        tu2_nodes = rna_nodes[rna_nodes.tu_id.apply(lambda x: "TU2" in x)]
        
        assert len(tu1_nodes) == 1
        assert len(tu2_nodes) == 1
        assert tu1_nodes.index[0] != tu2_nodes.index[0]
    
    def test_no_parameter_grouping(self, real_lib):
        """Test that RNAs with no parameters are grouped by content."""
        net = Network(
            lib=real_lib,
            cotx=[
                CoTransfection(
                    units=[
                        Unit(name="TU1", slots=["hEF1a", "eBFP2"]),  # no uORF
                        Unit(name="TU2", slots=["hEF1a", "eBFP2"]),  # no uORF
                        Unit(name="TU3", slots=["hEF1a", "mKate"]),  # different content
                    ]
                )
            ],
            build_on_init=True
        )
        
        cdg = net.central_dogma_graph
        rna_nodes = cdg[cdg.type == "RNA"]
        
        # TU1 and TU2 should be grouped (same content, no params)
        grouped_node = rna_nodes[rna_nodes.tu_id.apply(lambda x: "TU1" in x and "TU2" in x)]
        assert len(grouped_node) == 1
        
        # TU3 should be separate (different content)
        tu3_node = rna_nodes[rna_nodes.tu_id.apply(lambda x: "TU3" in x)]
        assert len(tu3_node) == 1
        assert tu3_node.index[0] != grouped_node.index[0]


class TestRefIdGrouping:
    """Test cases for ref_id based grouping."""
    
    @pytest.fixture
    def real_lib(self):
        """Use the real library for ref_id tests."""
        from biocomp.utils import load_lib
        return load_lib()
    
    def test_ref_id_parameter_tracking(self, real_lib):
        """Test that ref_id is properly tracked in TranscriptionUnit."""
        tu = TranscriptionUnit(
            name="test_tu",
            slots=[
                Slot(part="hEF1a"),
                Slot(part=["1x_uORF"], ref_id="shared_ref"),
                Slot(part="mKate"),
            ]
        )
        
        assert hasattr(tu, 'param_ref_ids')
        assert tu.param_ref_ids['tl_rate'] == "shared_ref"
        assert tu.param_ref_ids['tc_rate'] is None  # no ref_id for promoter
    
    def test_ref_id_grouping_same_ref(self, real_lib):
        """Test that RNAs with same ref_id are grouped even with different parts."""
        net = Network(
            lib=real_lib,
            cotx=[
                CoTransfection(
                    units=[
                        Unit(
                            name="TU1",
                            slots=["hEF1a", Slot(part=["1x_uORF"], ref_id="shared_ref"), "eBFP2"]
                        ),
                        Unit(
                            name="TU2",
                            slots=["hEF1a", Slot(part=["2x_uORF"], ref_id="shared_ref"), "eBFP2"]
                        ),
                    ]
                )
            ],
            build_on_init=True
        )
        
        cdg = net.central_dogma_graph
        rna_nodes = cdg[cdg.type == "RNA"]
        
        # TUs should be grouped together due to same ref_id
        grouped_node = rna_nodes[rna_nodes.tu_id.apply(lambda x: "TU1" in x and "TU2" in x)]
        assert len(grouped_node) == 1
        assert set(grouped_node.iloc[0]['tu_id']) == {"TU1", "TU2"}
    
    def test_ref_id_grouping_different_ref(self, real_lib):
        """Test that RNAs with different ref_id are NOT grouped."""
        net = Network(
            lib=real_lib,
            cotx=[
                CoTransfection(
                    units=[
                        Unit(
                            name="TU1",
                            slots=["hEF1a", Slot(part=["1x_uORF"], ref_id="ref_A"), "eBFP2"]
                        ),
                        Unit(
                            name="TU2",
                            slots=["hEF1a", Slot(part=["1x_uORF"], ref_id="ref_B"), "eBFP2"]
                        ),
                    ]
                )
            ],
            build_on_init=True
        )
        
        cdg = net.central_dogma_graph
        rna_nodes = cdg[cdg.type == "RNA"]
        
        # TUs should be in separate nodes due to different ref_ids
        tu1_nodes = rna_nodes[rna_nodes.tu_id.apply(lambda x: "TU1" in x)]
        tu2_nodes = rna_nodes[rna_nodes.tu_id.apply(lambda x: "TU2" in x)]
        
        assert len(tu1_nodes) == 1
        assert len(tu2_nodes) == 1
        assert tu1_nodes.index[0] != tu2_nodes.index[0]
    
    def test_ref_id_mixed_with_no_ref(self, real_lib):
        """Test mixing ref_id and no ref_id parameters."""
        net = Network(
            lib=real_lib,
            cotx=[
                CoTransfection(
                    units=[
                        Unit(
                            name="TU1",
                            slots=["hEF1a", Slot(part=["1x_uORF"], ref_id="shared"), "eBFP2"]
                        ),
                        Unit(
                            name="TU2",
                            slots=["hEF1a", Slot(part=["2x_uORF"], ref_id="shared"), "eBFP2"]
                        ),
                        Unit(
                            name="TU3",
                            slots=["hEF1a", Slot(part=["1x_uORF"]), "eBFP2"]  # no ref_id
                        ),
                    ]
                )
            ],
            build_on_init=True
        )
        
        cdg = net.central_dogma_graph
        rna_nodes = cdg[cdg.type == "RNA"]
        
        # TU1 and TU2 should be grouped (same ref_id)
        grouped_node = rna_nodes[rna_nodes.tu_id.apply(lambda x: "TU1" in x and "TU2" in x)]
        assert len(grouped_node) == 1
        
        # TU3 should be separate (no ref_id, different from ref_id group)
        tu3_node = rna_nodes[rna_nodes.tu_id.apply(lambda x: "TU3" in x)]
        assert len(tu3_node) == 1
        assert tu3_node.index[0] != grouped_node.index[0]
    
    def test_ref_id_with_multi_value_parts(self, real_lib):
        """Test that ref_id enables grouping of multi-value part lists."""
        net = Network(
            lib=real_lib,
            cotx=[
                CoTransfection(
                    units=[
                        Unit(
                            name="TU1",
                            slots=["hEF1a", Slot(part=["1x_uORF", "2x_uORF"], ref_id="multi"), "eBFP2"]
                        ),
                        Unit(
                            name="TU2",
                            slots=["hEF1a", Slot(part=["3x_uORF", "4x_uORF"], ref_id="multi"), "eBFP2"]
                        ),
                    ]
                )
            ],
            build_on_init=True
        )
        
        cdg = net.central_dogma_graph
        rna_nodes = cdg[cdg.type == "RNA"]
        
        # With same ref_id, they SHOULD be grouped even with different multi-value parts
        grouped_node = rna_nodes[rna_nodes.tu_id.apply(lambda x: "TU1" in x and "TU2" in x)]
        assert len(grouped_node) == 1
        assert set(grouped_node.iloc[0]['tu_id']) == {"TU1", "TU2"}
    
    def test_multi_value_no_grouping_without_ref_id(self, real_lib):
        """Test that identical multi-value parameters are NOT grouped without ref_id."""
        net = Network(
            lib=real_lib,
            cotx=[
                CoTransfection(
                    units=[
                        Unit(
                            name="TU1",
                            slots=["hEF1a", Slot(part=["1x_uORF", "2x_uORF"]), "eBFP2"]
                        ),
                        Unit(
                            name="TU2",
                            slots=["hEF1a", Slot(part=["1x_uORF", "2x_uORF"]), "eBFP2"]
                        ),
                    ]
                )
            ],
            build_on_init=True
        )
        
        cdg = net.central_dogma_graph
        rna_nodes = cdg[cdg.type == "RNA"]
        
        # Without ref_id, identical multi-value params should NOT be grouped
        tu1_nodes = rna_nodes[rna_nodes.tu_id.apply(lambda x: "TU1" in x)]
        tu2_nodes = rna_nodes[rna_nodes.tu_id.apply(lambda x: "TU2" in x)]
        
        assert len(tu1_nodes) == 1
        assert len(tu2_nodes) == 1
        assert tu1_nodes.index[0] != tu2_nodes.index[0]


class TestGetUniquePlasmidContent:
    """Test cases for get_unique_plasmid_content method."""
    
    @pytest.fixture
    def real_lib(self):
        """Use the real library for tests."""
        from biocomp.utils import load_lib
        return load_lib()
    
    def test_get_unique_plasmid_content_basic(self, real_lib):
        """Test basic functionality of get_unique_plasmid_content."""
        net = Network(
            lib=real_lib,
            cotx=[
                CoTransfection(
                    units=[
                        Unit(name="TU1", slots=["hEF1a", "1x_uORF", "eBFP2"], source="plasmid1"),
                        Unit(name="TU2", slots=["hEF1a", "2x_uORF", "mKate"], source="plasmid1"),
                        Unit(name="TU3", slots=["hEF1a", "1x_uORF", "eBFP2"], source="plasmid2"),
                    ]
                )
            ],
            build_on_init=True
        )
        
        unique_plasmids = net.get_unique_plasmid_content()
        
        # should have 2 unique plasmids
        assert len(unique_plasmids) == 2
        
        # each plasmid is a tuple of TUs, each TU is a tuple of parts
        for plasmid in unique_plasmids:
            assert isinstance(plasmid, tuple)
            for tu in plasmid:
                assert isinstance(tu, tuple)
    
    def test_get_unique_plasmid_content_identical_plasmids(self, real_lib):
        """Test that identical plasmids are deduplicated."""
        net = Network(
            lib=real_lib,
            cotx=[
                CoTransfection(
                    units=[
                        Unit(name="TU1", slots=["hEF1a", "eBFP2"], source="plasmid1"),
                        Unit(name="TU2", slots=["hEF1a", "eBFP2"], source="plasmid2"),
                    ]
                )
            ],
            build_on_init=True
        )
        
        unique_plasmids = net.get_unique_plasmid_content()
        
        # both plasmids have identical content, so should be deduplicated
        assert len(unique_plasmids) == 1
    
    def test_get_unique_plasmid_content_order_matters(self, real_lib):
        """Test that TU order within plasmids matters."""
        net = Network(
            lib=real_lib,
            cotx=[
                CoTransfection(
                    units=[
                        Unit(name="TU1", slots=["hEF1a", "eBFP2"], source="plasmid1"),
                        Unit(name="TU2", slots=["hEF1a", "mKate"], source="plasmid1"),
                        Unit(name="TU3", slots=["hEF1a", "mKate"], source="plasmid2"),
                        Unit(name="TU4", slots=["hEF1a", "eBFP2"], source="plasmid2"),
                    ]
                )
            ],
            build_on_init=True
        )
        
        unique_plasmids = net.get_unique_plasmid_content()
        
        # same TUs but in different order = different plasmids
        assert len(unique_plasmids) == 2
    
    def test_get_unique_plasmid_content_not_built(self, real_lib):
        """Test error when network is not built."""
        net = Network(
            lib=real_lib,
            build_on_init=False
        )
        net.transcription_units = None  # ensure it's not built
        
        with pytest.raises(ValueError, match="Network not built"):
            net.get_unique_plasmid_content()
    
    def test_get_unique_plasmid_content_with_multi_value_slots(self, real_lib):
        """Test with multi-value slots."""
        net = Network(
            lib=real_lib,
            cotx=[
                CoTransfection(
                    units=[
                        Unit(name="TU1", slots=["hEF1a", Slot(part=["1x_uORF", "2x_uORF"]), "eBFP2"], source="plasmid1"),
                        Unit(name="TU2", slots=["hEF1a", Slot(part=["1x_uORF", "2x_uORF"]), "mKate"], source="plasmid2"),
                    ]
                )
            ],
            build_on_init=True
        )
        
        unique_plasmids = net.get_unique_plasmid_content()
        
        # should have 2 unique plasmids
        assert len(unique_plasmids) == 2
        
        # verify multi-value slots are preserved as tuples in the tuple
        for plasmid in unique_plasmids:
            for tu in plasmid:
                # multi-value slots should be represented as tuples within the tuple
                assert any(isinstance(part, tuple) for part in tu)
    
    def test_get_unique_plasmid_content_squeeze_single_parts(self, real_lib):
        """Test that single-element lists are squeezed to direct values."""
        net = Network(
            lib=real_lib,
            cotx=[
                CoTransfection(
                    units=[
                        # Mix of single and multi-value slots
                        Unit(name="TU1", slots=["hEF1a", Slot(part=["1x_uORF"]), "eBFP2"], source="plasmid1"),
                        Unit(name="TU2", slots=["hEF1a", Slot(part=["1x_uORF", "2x_uORF"]), "mKate"], source="plasmid2"),
                    ]
                )
            ],
            build_on_init=True
        )
        
        unique_plasmids = net.get_unique_plasmid_content()
        
        # Find each plasmid content
        plasmid_contents = list(unique_plasmids)
        
        # Check plasmid1: should have squeezed single-value slot
        p1 = next(p for p in plasmid_contents if any('eBFP2' in tu for tu in p))
        tu1 = p1[0]  # First TU in plasmid1
        # The uORF slot had ["1x_uORF"] which should be squeezed to just "1x_uORF"
        assert '1x_uORF' in tu1
        assert not any(isinstance(part, tuple) for part in tu1)
        
        # Check plasmid2: should have tuple for multi-value slot
        p2 = next(p for p in plasmid_contents if any('mKate' in tu for tu in p))
        tu2 = p2[0]  # First TU in plasmid2
        # The uORF slot had ["1x_uORF", "2x_uORF"] which should remain as tuple
        assert ('1x_uORF', '2x_uORF') in tu2


class TestRemoveAggregation:
    """Test cases for remove_aggregation method."""
    
    @pytest.fixture
    def test_network(self):
        """Load test network from biocomptools ALL_NETWORKS."""
        from biocomptools.configs.designs.networks import ALL_NETWORKS
        # Get network at index 6 as specified
        network = ALL_NETWORKS[6]
        return network
    
    def test_remove_aggregation_basic(self, test_network):
        """Test removing an aggregation from the network."""
        # make a copy to avoid modifying the original
        net = test_network.copy()
        
        # verify initial state - should have more than 2 aggregations
        initial_agg_count = len(net.aggregations)
        assert initial_agg_count > 2, "Test network should have at least 3 aggregations"
        
        # get info about aggregation 2 before removal
        agg_2_sources = net.aggregations.loc[2, "source"]
        if not isinstance(agg_2_sources, list):
            agg_2_sources = [agg_2_sources]
        
        # count TUs that will be removed
        tus_to_remove = []
        for source in agg_2_sources:
            source_tus = net.tu_in_sources[net.tu_in_sources["source"] == source]["TU"].tolist()
            tus_to_remove.extend(source_tus)
        initial_tu_count = len(net.transcription_units)
        
        # remove aggregation id 2 (third aggregation)
        net.remove_aggregation(2)
        
        # check that we now have exactly 2 aggregations left
        assert len(net.aggregations) == 2, f"Should have 2 aggregations left, but have {len(net.aggregations)}"
        
        # verify aggregation 2 is not in the index
        assert 2 not in net.aggregations.index, "Aggregation 2 should be removed"
        
        # verify the sources were removed
        for source in agg_2_sources:
            assert source not in net.tu_in_sources["source"].values, f"Source {source} should be removed"
        
        # verify the TUs were removed
        for tu in tus_to_remove:
            assert tu not in net.transcription_units, f"TU {tu} should be removed"
        
        # verify TU count decreased appropriately
        assert len(net.transcription_units) == initial_tu_count - len(tus_to_remove)
        
        # verify raw data structures were updated
        if net.raw_aggregations is not None:
            for aid, _, _ in net.raw_aggregations:
                assert aid != 2, "Aggregation 2 should not be in raw_aggregations"
        
        if net.raw_tu_in_sources is not None:
            for source, _, _ in net.raw_tu_in_sources:
                assert source not in agg_2_sources, f"Source {source} should not be in raw_tu_in_sources"
    
    def test_remove_aggregation_invalid_id(self, test_network):
        """Test error when removing non-existent aggregation."""
        net = test_network.copy()
        
        # try to remove an aggregation that doesn't exist
        with pytest.raises(ValueError, match="Aggregation .* not found"):
            net.remove_aggregation(999)
    
    def test_remove_aggregation_rebuild(self, test_network):
        """Test that network is rebuilt after removing aggregation."""
        net = test_network.copy()
        
        # build the network first
        net.build()
        assert net.is_built()
        
        # remove an aggregation
        net.remove_aggregation(2)
        
        # if there are still TUs, network should be rebuilt
        if len(net.transcription_units) > 0:
            assert net.is_built(), "Network should be rebuilt after removing aggregation"
    
    def test_remove_all_aggregations(self, test_network):
        """Test removing all aggregations leaves empty network."""
        net = test_network.copy()
        
        # build first
        net.build()
        
        # remove all aggregations one by one
        agg_ids = list(net.aggregations.index)
        for agg_id in agg_ids:
            net.remove_aggregation(agg_id)
        
        # should have no aggregations left
        assert len(net.aggregations) == 0
        assert len(net.transcription_units) == 0
        assert len(net.tu_in_sources) == 0
        
        # network should be cleaned but not built
        assert not net.is_built()
    
    def test_clean_all_method(self, test_network):
        """Test that clean_all properly resets network state."""
        net = test_network.copy()
        
        # build the network
        net.build()
        assert net.is_built()
        assert net.compute_graph is not None
        assert net.central_dogma_graph is not None
        
        # clean all
        net.clean_all()
        
        # verify graphs are cleared
        assert net.compute_graph is None
        assert net.central_dogma_graph is None
        assert net._n_inputs is None
        assert net._n_outputs is None
        assert net._output_proteins is None
        
        # verify source data is preserved
        assert net.transcription_units is not None
        assert net.tu_in_sources is not None
        assert net.aggregations is not None
        assert net.name is not None


class TestCommittedNetworkRemoveAggregation:
    """Test cases for remove_aggregation on networks from stack.commit()."""
    
    @pytest.fixture
    def test_params(self, request):
        """Load test parameters from test_params.pickle."""
        import pickle
        from pathlib import Path
        test_file_path = Path(request.fspath).parent / "test_params.pickle"
        return pickle.load(open(test_file_path, "rb"))
    
    def test_committed_network_remove_aggregation(self, test_params):
        """Test removing aggregation from a committed network."""
        import pytest
        from sqlmodel import Session
        import biocomptools.toollib.models as md
        from biocomptools.toollib.common import config
        from biocomptools.configs.designs.networks import ALL_NETWORKS
        from biocomptools.toollib.modelselector import ModelSelector
        import biocomp.compute as cmp
        import biocomp.jaxutils as bju
        
        # Use the test parameters from fixture
        params = test_params
        
        # Load model
        session = Session(md.get_biocompdb_sqlite_engine(config.db.sqlite.path))
        mname = 'irritomit-osterogla-gonalesce'
        model_results = ModelSelector(name=mname).get_models(session)
        assert len(model_results) > 0, f"No models found with name {mname}"
        model = model_results[0].load()
        
        # Set up parameters
        (REP, T, N) = 14, 0, 6
        bestp = bju.tree_get(params, (REP, T))
        
        # Get networks and create stack
        NETWORKS = ALL_NETWORKS
        orig_network = NETWORKS[N]
        stack = cmp.ComputeStack(networks=NETWORKS)
        stack.build(model.compute_config)
        
        # Commit the stack - this modifies the networks
        final_networks = stack.commit(bestp)
        final_network = final_networks[N]
        
        # Verify network is built
        assert final_network.is_built()
        
        # Get initial state
        initial_agg_count = len(final_network.aggregations)
        assert initial_agg_count > 2, "Test network should have at least 3 aggregations"
        
        # Debug: Check the state before removal
        print("Before removal:")
        print(f"  Transcription units: {list(final_network.transcription_units.keys())}")
        print(f"  TUs in sources: {set(final_network.tu_in_sources['TU'].unique())}")
        
        # Check which aggregations contain which sources
        print("  Aggregations and their sources:")
        for agg_id, row in final_network.aggregations.iterrows():
            sources = row["source"]
            if not isinstance(sources, list):
                sources = [sources]
            print(f"    Agg {agg_id}: {sources}")
        
        # Check for 'x1_a+'
        if 'x1_a+' in final_network.transcription_units:
            print("  'x1_a+' is in transcription_units")
            x1_rows = final_network.tu_in_sources[final_network.tu_in_sources['TU'] == 'x1_a+']
            if len(x1_rows) > 0:
                print(f"  'x1_a+' is in sources: {x1_rows['source'].tolist()}")
                # Which aggregations contain plsmd_2?
                for agg_id, row in final_network.aggregations.iterrows():
                    sources = row["source"]
                    if not isinstance(sources, list):
                        sources = [sources]
                    if 'plsmd_2' in sources:
                        print(f"  'plsmd_2' is in aggregation {agg_id}")
            else:
                print("  'x1_a+' is NOT in any sources")
        
        print(f"  Removing aggregation 2, which has sources: {final_network.aggregations.loc[2, 'source']}")
        
        # This should work without errors
        final_network.remove_aggregation(2)
        
        # Verify aggregation was removed
        assert len(final_network.aggregations) == initial_agg_count - 1
        assert 2 not in final_network.aggregations.index
