"""Test compute stack for complex_twolayers network

This test validates the complex two-layer ERN network with design mode features:
- 3 CoTransfections (x1, x2 inputs + b bias)
- 3 ERNs in 2 layers (CasE + Csy4 → PgU)
- Variable uORFs (u1=none, u2=all, u3=all except none)
- Unlocked bias parameter
- 4 fluorescent outputs

Tests cover:
- Network structure and layer organization
- uORF slot configuration
- Compute graph topology
- ERN layer hierarchy
- Aggregation ratios
- Parameter constraints
- Quantization masks
- Forward pass computation
- Manual computation oracle
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from collections import Counter
from functools import partial

from biocomp.network import recipe_to_networks
from biocomp.library import LibraryContext, load_lib
import biocomp.biorules as br
from biocomp.compute import ComputeStack
from biocomp.config import SIMPLE_NODES_COMPUTE_CONFIG
from biocomp.recipe import Recipe, CoTransfection, TranscriptionUnit, Slot, FluoIntensity, NumRange
from biocomp.jaxutils import flat_concat
from biocomp.nodes.ern import ERN_DEFAULT_NEG_PARTS


EMBEDDING_SHAPE = (1,)
U1_EXPECTED_MASK = np.zeros(13, dtype=bool)
U1_EXPECTED_MASK[0] = True
U2_EXPECTED_MASK = np.zeros(13, dtype=bool)
U2_EXPECTED_MASK[:9] = True
U3_EXPECTED_MASK = np.zeros(13, dtype=bool)
U3_EXPECTED_MASK[1:9] = True

P = "hEF1a"
T = "L0.T_4560"

ERNS = ["CasE", "Csy4", "PgU"]

UORFS = [
    None,
    "1w_uORF",
    "1x_uORF",
    "2x_uORF",
    "3x_uORF",
    "4x_uORF",
    "5x_uORF",
    "6x_uORF",
    "8x_uORF",
]

COLORS = {
    "x1": "mKO2",
    "x2": "eBFP2",
    "b": "mMaroon1",
    "y": "mNeonGreen",
}

raw_ratios = np.arange(8, dtype=float) + 1
x1ratios = raw_ratios / np.sum(raw_ratios)
x2ratios = raw_ratios[::-1].copy() / np.sum(raw_ratios[::-1])
bratios = np.ones((8,)) / 8

BIAS_FLUO = FluoIntensity(
    tu_id=0,
    value=NumRange(min=0.3, max=0.6),
    protein=COLORS["b"],
    units="Rescaled AU",
)


def make_units(tu_name, erns):
    recs = [f"{ern}_rec" for ern in erns]
    u1 = Slot(part=UORFS[0], ref_id="U1")
    u2 = Slot(part=UORFS, ref_id="U2")
    u3 = Slot(part=UORFS[1:], ref_id="U3")
    return [
        TranscriptionUnit(slots=[P, COLORS[tu_name], T], name=f"{tu_name}_marker", source="themarker"),
        TranscriptionUnit(slots=[P, u1, recs[0], erns[2], T], name=f"{tu_name}_a+", source="03"),
        TranscriptionUnit(slots=[P, erns[0], T], name=f"{tu_name}_a-", source="45"),
        TranscriptionUnit(slots=[P, u2, recs[1], erns[2], T], name=f"{tu_name}_b+", source="haha12"),
        TranscriptionUnit(slots=[P, erns[1], T], name=f"{tu_name}_b-", source="wrong order 78"),
        TranscriptionUnit(slots=[P, u3, recs[2], COLORS["y"], T], name=f"{tu_name}_c+", source="a random id"),
        TranscriptionUnit(slots=[P, erns[2], T], name=f"{tu_name}_c-", source="00aaa"),
        TranscriptionUnit(slots=[P, COLORS["y"], T], name=f"{tu_name}_direct_out", source="direct"),
    ]


@pytest.fixture
def lib():
    return load_lib()


@pytest.fixture
def complex_twolayers_design_network(lib):
    """Complex network with 3 cotx groups and unlocked bias on one group.
    2 erns on first layer, 1 ern on second layer.
    It's the type of network we will use in design mode,
    so it's super important to have very very solid tests around it."""
    with LibraryContext.with_library(lib):
        recipe = Recipe(
            name="two_and_one",
            content=[
                CoTransfection(name="x1", units=make_units("x1", ERNS), ratios=x1ratios.tolist()),
                CoTransfection(name="x2", units=make_units("x2", ERNS), ratios=x2ratios.tolist()),
                CoTransfection(name="b", units=make_units("b", ERNS), fluo_bias=BIAS_FLUO, ratios=bratios.tolist()),
            ],
        )
    return recipe


def complex_twolayers_basic_stack_assertions(network, stack, params):
    aggs = network.compute_graph.get_nodes_by_type("aggregation")
    for a in aggs:
        ag_layer_num, ag_pos = stack.node_map[(0, a.node_id)]
        assert ag_layer_num == 6
        ag_layer = stack.layers[ag_layer_num]
        assert len(ag_layer.nodes) == 3
        assert list(ag_layer.f_out_shapes) == [(1,)] * 8
        assert list(ag_layer.f_input_shapes) == [(1,)]
        assert ag_layer.f_type == "aggregation"
        assert ag_layer.namespace == "local/6/aggregation8x"
        assert params[ag_layer.namespace]["ratios"].shape == (3, 8)
        param_ratios = params[ag_layer.namespace]["ratios"][ag_pos]
        assert np.allclose(a.extra["ratios"], param_ratios)
    return aggs


def complex_twolayers_advanced_stack_assertions(
    network, stack, params, translate, nouorf_emb, test_key, intermediate_values, y_aux
):
    params[stack.layers[8].namespace]
    stack.layers[8].nodes
    stack.layers[8].f_input_shapes
    test_input = [np.ones(1) * 2 * i for i in range(3)]
    tl_out, tl_aux = stack.layers[8].f_apply(
        *test_input, random_vars=np.zeros(100), params=params, node_id=0, key=test_key
    )
    manual_tl_out = translate(flat_concat(*test_input), [nouorf_emb] * 3)
    assert np.allclose(tl_out, manual_tl_out, rtol=1e-5)
    stack.layers[13].f_input_shapes
    ernc_test_input = [intermediate_values["Ec_negative_in"], intermediate_values["Ec_positive_in"]]
    ernc_out, ernc_aux = stack.layers[13].f_apply(
        *ernc_test_input,
        random_vars=np.zeros(100),
        params=params,
        node_id=0,
        key=test_key,
    )
    Ea = intermediate_values["Ea"]
    Eb = intermediate_values["Eb"]
    Ec = intermediate_values["Ec"]
    B_inv = intermediate_values["B_inv"]
    X1_inv = intermediate_values["X1_inv"]
    X2_inv = intermediate_values["X2_inv"]

    stack.layers[15].f_input_shapes
    fl_test_input = np.array([np.ones(1) * i for i in range(4)])
    fl_out, fl_aux = stack.layers[15].f_apply(
        *fl_test_input, random_vars=np.zeros(100), params=params, node_id=0, key=test_key
    )
    assert np.all(fl_out.flatten() == fl_test_input.flatten())

    assert np.all(y_aux["5"]["trace"]["outputs"] == np.array([[B_inv], [X1_inv], [X2_inv]]))
    y_aux["7"]["trace"]["outputs"]
    for sn in stack.layers[7].nodes:
        n = network.compute_graph.get_node(sn.node_id)
        assert (
            intermediate_values[n.extra["name"]]
            == y_aux["7"]["trace"]["outputs"][sn.node_position_in_layer][0]
        )
    y_aux["11"]["trace"]["outputs"]
    assert np.allclose(
        y_aux["11"]["trace"]["outputs"][0][0],
        Eb,
        rtol=1e-5,
    ), f"Eb: {Eb}, aux: {y_aux['11']['trace']['outputs'][0][0]}"
    assert np.allclose(
        y_aux["11"]["trace"]["outputs"][1][0],
        Ea,
        rtol=1e-5,
    ), f"Ea: {Ea}, aux: {y_aux['11']['trace']['outputs'][1][0]}"
    y_aux["13"]["trace"]["outputs"]
    assert np.allclose(
        y_aux["13"]["trace"]["outputs"][0][0],
        Ec,
        rtol=1e-5,
    ), f"Ec: {Ec}, aux: {y_aux['13']['trace']['outputs'][0][0]}"

    assert np.allclose(ernc_out, Ec, rtol=1e-5), f"ernc_out: {ernc_out}, Ec: {Ec}"


def complex_twolayers_topology_assertions(network, stack, params):
    u1_local_emb = None
    u2_local_emb = None
    u3_local_emb = None

    tlnode = network.compute_graph.get_node(27)
    assert tlnode.node_type == "translation"
    tl_layer_num, tl_pos = stack.node_map[(0, 27)]
    assert tl_layer_num == 12
    assert tl_pos == 0
    assert stack.layers is not None
    tl_layer = stack.layers[tl_layer_num]
    assert tl_layer.f_type == "translation"
    assert tl_layer.f_out_shapes == [(1,)]
    assert tl_layer.f_input_shapes == ((1,), (1,), (1,))
    assert tl_layer.namespace == "local/12/translation"
    assert len(tl_layer.nodes) == 1
    assert tl_layer.nodes[0].node_id == 27
    assert tl_layer.nodes[0].layer_number == tl_layer_num
    assert tl_layer.nodes[0].node_position_in_layer == tl_pos
    params[tl_layer.namespace]
    upnodes = network.compute_graph.get_upstream_nodes(tlnode.node_id)
    assert len(upnodes) == 3

    type_counts = Counter([un[0].node_type for un in upnodes])
    assert type_counts["sequestron_ERN"] == 2
    assert type_counts["transcription"] == 1
    E0, E1, E2 = ERNS
    for i, (un, ue) in enumerate(upnodes):
        assert ue.content_type == "RNA"
        if un.node_type == "sequestron_ERN":
            ern_layer_num, ern_pos = stack.node_map[(0, un.node_id)]
            ern_layer = stack.layers[ern_layer_num]
            assert ern_layer.f_type == "sequestron_ERN"
            assert ern_layer.f_out_shapes == [(1,)]
            assert ern_layer.f_input_shapes == ((1,), (1,))
            assert ern_layer.namespace == "local/11/sequestron_ERN"
            assert len(ern_layer.nodes) == 2
            assert ern_layer.nodes[ern_pos].node_id == un.node_id
            ern_affinity = params[ern_layer.namespace]["affinity"][ern_pos]
            assert ern_affinity.shape == (1,)
            if un.extra["seq_name"] == f"ERN::{E0}#{E0}_rec":
                eid = ERN_DEFAULT_NEG_PARTS.index(E0)
                assert un.extra["layer_id"] == 0
                assert params["shared/ERN_5p/affinities"][eid] == ern_affinity
                i_mask = params[tl_layer.namespace]["tl_rate_quantization_mask"][tl_pos, i, :]
                u1_local_emb = params[tl_layer.namespace]["tl_rate"][0, i, :]
                assert u1_local_emb.shape == EMBEDDING_SHAPE
                assert np.array_equal(i_mask, U1_EXPECTED_MASK)
            elif un.extra["seq_name"] == f"ERN::{E1}#{E1}_rec":
                eid = ERN_DEFAULT_NEG_PARTS.index(E1)
                assert un.extra["layer_id"] == 0
                assert params["shared/ERN_5p/affinities"][eid] == ern_affinity
                i_mask = params[tl_layer.namespace]["tl_rate_quantization_mask"][tl_pos, i, :]
                u2_local_emb = params[tl_layer.namespace]["tl_rate"][0, i, :]
                assert u2_local_emb.shape == EMBEDDING_SHAPE
                assert np.array_equal(i_mask, U2_EXPECTED_MASK)
            else:
                raise ValueError("Unexpected ern node")
        elif un.node_type == "transcription":
            i_mask = params[tl_layer.namespace]["tl_rate_quantization_mask"][tl_pos, i, :]
            assert np.array_equal(i_mask, U1_EXPECTED_MASK)

    tlnode2 = network.compute_graph.get_node(34)
    assert tlnode2.node_type == "translation"
    tl_layer_num2, tl_pos2 = stack.node_map[(0, 34)]
    tl_layer2 = stack.layers[tl_layer_num2]
    assert tl_layer_num2 == 14
    assert tl_pos2 == 0
    upnodes2 = network.compute_graph.get_upstream_nodes(tlnode2.node_id)
    assert len(upnodes2) == 2
    downnodes2 = network.compute_graph.get_downstream_nodes(tlnode2.node_id)
    assert len(downnodes2) == 1
    assert downnodes2[0][0].node_type == "output"
    for i, (un, ue) in enumerate(upnodes2):
        assert ue.content_type == "RNA"
        if un.node_type == "sequestron_ERN":
            ern_layer_num, ern_pos = stack.node_map[(0, un.node_id)]
            ern_layer = stack.layers[ern_layer_num]
            assert ern_layer.f_type == "sequestron_ERN"
            assert ern_layer.f_out_shapes == [(1,)]
            assert ern_layer.f_input_shapes == ((1,), (1,))
            assert ern_layer.namespace == "local/13/sequestron_ERN"
            assert len(ern_layer.nodes) == 1
            assert ern_layer.nodes[ern_pos].node_id == un.node_id
            ern_affinity = params[ern_layer.namespace]["affinity"][ern_pos]
            assert ern_affinity.shape == (1,)
            assert un.extra["seq_name"] == f"ERN::{E2}#{E2}_rec"
            assert un.extra["layer_id"] == 1
            eid = ERN_DEFAULT_NEG_PARTS.index(E2)
            assert params["shared/ERN_5p/affinities"][eid] == ern_affinity
            i_mask = params[tl_layer2.namespace]["tl_rate_quantization_mask"][tl_pos2, i, :]
            u3_local_emb = params[tl_layer2.namespace]["tl_rate"][0, i, :]
            assert u3_local_emb.shape == EMBEDDING_SHAPE
            assert np.array_equal(i_mask, U3_EXPECTED_MASK)

    return u1_local_emb, u2_local_emb, u3_local_emb


def test_complex_twolayers_design_network_structure(complex_twolayers_design_network):
    """Test the basic recipe structure"""
    recipe = complex_twolayers_design_network
    assert len(recipe.content) == 3

    assert recipe.content[0].name == "x1"
    assert len(recipe.content[0].units) == 8
    assert recipe.content[0].fluo_bias is None

    assert recipe.content[1].name == "x2"
    assert len(recipe.content[1].units) == 8
    assert recipe.content[1].fluo_bias is None

    assert recipe.content[2].name == "b"
    assert len(recipe.content[2].units) == 8
    assert recipe.content[2].fluo_bias is not None


def test_complex_twolayers_uorf_slots(complex_twolayers_design_network):
    """Test that uORF slots are correctly configured"""
    recipe = complex_twolayers_design_network

    for cotx in recipe.content:
        a_plus = cotx.units[1]
        uorf_slots_a = [s for s in a_plus.slots if s.ref_id == "U1"]
        assert len(uorf_slots_a) == 1
        assert uorf_slots_a[0].part is None or uorf_slots_a[0].part == [None]

        b_plus = cotx.units[3]
        uorf_slots_b = [s for s in b_plus.slots if s.ref_id == "U2"]
        assert len(uorf_slots_b) == 1
        assert isinstance(uorf_slots_b[0].part, list)
        assert None in uorf_slots_b[0].part
        assert "1x_uORF" in uorf_slots_b[0].part

        c_plus = cotx.units[5]
        uorf_slots_c = [s for s in c_plus.slots if s.ref_id == "U3"]
        assert len(uorf_slots_c) == 1
        assert isinstance(uorf_slots_c[0].part, list)
        assert None not in uorf_slots_c[0].part
        assert "1w_uORF" in uorf_slots_c[0].part


def test_complex_twolayers_compg_structure(lib, complex_twolayers_design_network):
    """Test the compute graph structure"""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(complex_twolayers_design_network, br.ALL_RULES, invert=True)
        compg = networks[0].compute_graph

        node_types = Counter(n.node_type for n in compg.nodes.values())

        assert node_types["aggregation"] == 3
        assert node_types["source"] == 24
        assert node_types["sequestron_ERN"] == 3
        assert node_types["output"] == 1
        assert node_types["bias"] == 1
        assert node_types["input"] == 2


def test_complex_twolayers_ern_topology(lib, complex_twolayers_design_network):
    """Test that ERN nodes have correct topology (2 first-layer, 1 second-layer)"""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(complex_twolayers_design_network, br.ALL_RULES, invert=True)
        compg = networks[0].compute_graph
        assert compg is not None

        ern_nodes = [n for n in compg.nodes.values() if n.node_type == "sequestron_ERN"]
        assert len(ern_nodes) == 3

        case_ern = [n for n in ern_nodes if "CasE" in n.extra.get("seq_name", "")][0]
        csy4_ern = [n for n in ern_nodes if "Csy4" in n.extra.get("seq_name", "")][0]
        pgu_ern = [n for n in ern_nodes if "PgU" in n.extra.get("seq_name", "")][0]

        pgu_incoming = compg.get_incoming_edges(pgu_ern.node_id)
        assert len(pgu_incoming) == 2

        case_incoming = compg.get_incoming_edges(case_ern.node_id)
        assert len(case_incoming) == 2

        csy4_incoming = compg.get_incoming_edges(csy4_ern.node_id)
        assert len(csy4_incoming) == 2

        pgu_outgoing = compg.get_outgoing_edges(pgu_ern.node_id)
        assert len(pgu_outgoing) >= 1


def test_complex_twolayers_outputs(lib, complex_twolayers_design_network):
    """Test output structure"""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(complex_twolayers_design_network, br.ALL_RULES, invert=True)
        network = networks[0]
        assert network.compute_graph is not None

        output_proteins = network.get_output_proteins()
        assert output_proteins == ["eBFP2", "mKO2", "mMaroon1", "mNeonGreen"]
        outnodes = network.compute_graph.get_nodes_by_type("output")
        assert len(outnodes) == 1
        outnode = outnodes[0]
        downstream = network.compute_graph.get_downstream_nodes(outnode.node_id)
        assert len(downstream) == 0
        lastedges = network.compute_graph.get_incoming_edges(outnode.node_id)
        assert len(lastedges) == 4
        sorted_edges = sorted(lastedges, key=lambda e: e.to_input_slot)
        sorted_proteins = [e.content[0].name for e in sorted_edges]
        assert sorted_proteins == output_proteins
        assert network.get_dependent_output_mask().tolist() == [False, False, False, True], (
            f"Dependent output mask is {network.get_dependent_output_mask().tolist()}, expected [False, False, False, True]"
        )
        assert network.get_dependent_output_proteins() == ["mNeonGreen"], (
            f"Dependent output proteins are {network.get_dependent_output_proteins()}, expected ['mNeonGreen']"
        )

        assert network.get_bias_proteins() == ["mMaroon1"], (
            f"Bias proteins are {network.get_bias_proteins()}, expected ['mMaroon1']"
        )

        input_proteins = network.get_inverted_input_proteins(include_biases=False)
        assert input_proteins == [COLORS["x1"], COLORS["x2"]]
        assert network.nb_inputs == 2
        input_proteins_wbias = network.get_inverted_input_proteins(include_biases=True)
        assert input_proteins_wbias == [COLORS["x1"], COLORS["x2"], COLORS["b"]]

        input_output_map = network.get_inverted_input_positions(include_biases=False)
        for inp_id, outp_id in input_output_map.items():
            assert input_proteins[inp_id] == output_proteins[outp_id]
        input_output_map_wbias = network.get_inverted_input_positions(include_biases=True)
        for inp_id, outp_id in input_output_map_wbias.items():
            assert input_proteins_wbias[inp_id] == output_proteins[outp_id]

        aggs = network.compute_graph.get_nodes_by_type("aggregation")
        for a in aggs:
            upnodes = network.compute_graph.get_upstream_nodes(a.node_id, recursive=True)
            upnode_types = [n.node_type for n, e in upnodes]
            assert len(upnodes) == 5
            assert upnode_types[:-1] == [
                "inv_aggregation",
                "inv_source",
                "inv_transcription",
                "inv_translation",
            ]
            root_node = upnodes[-1][0]
            if a.extra["cotx_group"] == "x1":
                assert root_node.node_type == "input"
                assert root_node.extra["input_position"] == 0
                assert root_node.extra["input_from_output"] == output_proteins.index(COLORS["x1"])
            elif a.extra["cotx_group"] == "x2":
                assert root_node.node_type == "input"
                assert root_node.extra["input_position"] == 1
                assert root_node.extra["input_from_output"] == output_proteins.index(COLORS["x2"])
            elif a.extra["cotx_group"] == "b":
                assert root_node.node_type == "bias"
                assert root_node.extra["input_from_output"] == output_proteins.index(COLORS["b"])
                assert FluoIntensity(**root_node.extra["fluo_bias"]) == BIAS_FLUO


def test_complex_twolayers_aggregations(lib, complex_twolayers_design_network):
    """Test aggregation nodes have correct ratios"""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(complex_twolayers_design_network, br.ALL_RULES, invert=True)
        network = networks[0]
        assert network.compute_graph is not None
        aggs = network.compute_graph.get_nodes_by_type("aggregation")
        for a in aggs:
            downnodes = network.compute_graph.get_downstream_nodes(a.node_id)
            assert len(downnodes) == 8
            assert all(dn[0].node_type == "source" for dn in downnodes)
            for dnode, dedge in downnodes:
                assert dedge.to_input_slot == 0
                assert len(network.compute_graph.get_upstream_nodes(dnode.node_id)) == 1
            unique_slots = set(dedge.from_output_slot for _, dedge in downnodes)
            assert len(unique_slots) == 8, (
                f"Expected 8 unique output slots for agg, got {unique_slots}"
            )

            for dnode, dedge in downnodes:
                outslot = dedge.from_output_slot
                assert dnode.node_type == "source"
                ratio_expected = dnode.extra["ratio"]
                ratio_actual = a.extra["ratios"][outslot]
                source_id_expected = dnode.extra["source_id"]
                source_id_actual = a.extra["members"][outslot]
                assert ratio_expected == ratio_actual, (
                    f"Downstream source node {dnode.node_id} ratio {ratio_expected} != aggregation ratio {ratio_actual}"
                )
                assert source_id_expected == source_id_actual, (
                    f"Downstream source node {dnode.node_id} source_id {source_id_expected} != aggregation member {source_id_actual}"
                )

            upnodes = network.compute_graph.get_upstream_nodes(a.node_id)
            assert len(upnodes) == 1
            upnode, upedge = upnodes[0]
            assert upnode.node_type == "inv_aggregation"
            assert upnode.extra["original_output_len"] == 8
            orig_outslot = upnode.extra["original_output_slot"]
            assert a.extra["members"][orig_outslot] == "themarker"

            if a.extra["cotx_group"] == "x1":
                ratio_order = a.extra["members"]
                expected_ratios = {
                    "themarker": x1ratios[0],
                    "03": x1ratios[1],
                    "45": x1ratios[2],
                    "haha12": x1ratios[3],
                    "wrong order 78": x1ratios[4],
                    "a random id": x1ratios[5],
                    "00aaa": x1ratios[6],
                    "direct": x1ratios[7],
                }
                for src_id, ratio in zip(ratio_order, a.extra["ratios"]):
                    assert np.isclose(ratio, expected_ratios[src_id]), (
                        f"Aggregation ratio for {src_id} in cotx x1 is {ratio}, expected {expected_ratios[src_id]}"
                    )

            elif a.extra["cotx_group"] == "x2":
                ratio_order = a.extra["members"]
                expected_ratios = {
                    "themarker": x1ratios[7],
                    "03": x1ratios[6],
                    "45": x1ratios[5],
                    "haha12": x1ratios[4],
                    "wrong order 78": x1ratios[3],
                    "a random id": x1ratios[2],
                    "00aaa": x1ratios[1],
                    "direct": x1ratios[0],
                }
                for src_id, ratio in zip(ratio_order, a.extra["ratios"]):
                    assert np.isclose(ratio, expected_ratios[src_id]), (
                        f"Aggregation ratio for {src_id} in cotx x2 is {ratio}, expected {expected_ratios[src_id]}"
                    )
            elif a.extra["cotx_group"] == "b":
                for ratio in a.extra["ratios"]:
                    assert np.isclose(ratio, 0.125), (
                        f"Aggregation ratio for cotx b is {ratio}, expected 0.125"
                    )


def test_complex_twolayers_structure(lib, complex_twolayers_design_network):
    """Validate network structure and layer organization"""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(complex_twolayers_design_network, br.ALL_RULES, invert=True)
        network = networks[0]
        stack = ComputeStack([network])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        assert len(stack.layers) == 16, f"Expected 16 layers, got {len(stack.layers)}"

        expected_structure = [
            ("input", 2),
            ("bias", 1),
            ("inv_translation", 3),
            ("inv_transcription", 3),
            ("inv_source", 3),
            ("inv_aggregation", 3),
            ("aggregation", 3),
            ("source", 24),
            ("transcription", 7),
            ("transcription", 3),
            ("translation", 5),
            ("sequestron_ERN", 2),
            ("translation", 1),
            ("sequestron_ERN", 1),
            ("translation", 1),
            ("output", 1),
        ]

        for i, (expected_type, expected_n_nodes) in enumerate(expected_structure):
            layer = stack.layers[i]
            assert layer.f_type == expected_type, (
                f"Layer {i}: expected {expected_type}, got {layer.f_type}"
            )
            assert len(layer.nodes) == expected_n_nodes, (
                f"Layer {i}: expected {expected_n_nodes} nodes, got {len(layer.nodes)}"
            )


def test_complex_twolayers_parameter_constraints(lib, complex_twolayers_design_network):
    """Validate parameter initialization and constraints"""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(complex_twolayers_design_network, br.ALL_RULES, invert=True)
        network = networks[0]
        stack = ComputeStack([network])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        init_key = jax.random.PRNGKey(42)
        params = stack.init(init_key)

        agg_layer_idx = 6
        agg_namespace = stack.layers[agg_layer_idx].namespace
        ratios = params[f"{agg_namespace}/ratios"]
        assert ratios.shape == (3, 8), f"Expected (3, 8) ratios, got {ratios.shape}"
        assert jnp.all(ratios > 0), "All aggregation ratios should be positive"

        expected_x1 = jnp.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=jnp.float32)
        expected_x2 = jnp.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=jnp.float32)
        expected_b = jnp.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=jnp.float32)

        expected_x1_norm = jnp.abs(expected_x1) / jnp.sum(jnp.abs(expected_x1))
        expected_x2_norm = jnp.abs(expected_x2) / jnp.sum(jnp.abs(expected_x2))
        expected_b_norm = jnp.abs(expected_b) / jnp.sum(jnp.abs(expected_b))

        ratios_norm = jnp.abs(ratios) / jnp.sum(jnp.abs(ratios), axis=1, keepdims=True)

        x1x2_sorted = jnp.sort(expected_x1_norm)
        b_sorted = jnp.sort(expected_b_norm)

        x1x2_matches = []
        b_matches = []

        for row_idx in range(3):
            row = ratios_norm[row_idx]
            row_sorted = jnp.sort(row)
            if jnp.allclose(row_sorted, x1x2_sorted, rtol=1e-5):
                x1x2_matches.append(row_idx)
            elif jnp.allclose(row_sorted, b_sorted, rtol=1e-5):
                b_matches.append(row_idx)

        assert len(x1x2_matches) == 2, (
            f"Expected 2 rows to match x1/x2 pattern, found {len(x1x2_matches)}.\n"
            f"Ratios (normalized, sorted):\n{jnp.sort(ratios_norm, axis=1)}\n"
            f"Expected x1/x2 (sorted): {x1x2_sorted}\n"
            f"Expected b (sorted): {b_sorted}"
        )
        assert len(b_matches) == 1, (
            f"Expected 1 row to match b pattern, found {len(b_matches)}.\n"
            f"Ratios (normalized, sorted):\n{jnp.sort(ratios_norm, axis=1)}\n"
            f"Expected b (sorted): {b_sorted}"
        )

        tl_masks_10 = params["local/10/translation/tl_rate_quantization_mask"]
        assert tl_masks_10.shape == (5, 1, 13), (
            f"Layer 10 TL masks: expected (5, 1, 13), got {tl_masks_10.shape}"
        )
        for node_idx in range(5):
            mask = tl_masks_10[node_idx]
            assert jnp.sum(mask) == 1, f"Node {node_idx} should have exactly 1 uORF option"
            assert mask[0, 0], f"Node {node_idx} should have no-uORF (index 0) available"

        tl_masks_12 = params["local/12/translation/tl_rate_quantization_mask"]
        assert tl_masks_12.shape == (1, 3, 13), (
            f"Layer 12 TL masks: expected (1, 3, 13), got {tl_masks_12.shape}"
        )
        mask_27 = tl_masks_12[0]
        assert jnp.sum(mask_27[0]) == 1, "Input 0 should have 1 option (u1=none)"
        assert mask_27[0, 0], "Input 0 should have index 0 available"
        assert jnp.sum(mask_27[1]) == 9, "Input 1 should have 9 options (u2=all)"
        assert jnp.sum(mask_27[2]) == 1, "Input 2 should have 1 option (no uORF)"

        tl_masks_14 = params["local/14/translation/tl_rate_quantization_mask"]
        assert tl_masks_14.shape == (1, 2, 13), (
            f"Layer 14 TL masks: expected (1, 2, 13), got {tl_masks_14.shape}"
        )
        mask_34 = tl_masks_14[0]
        assert jnp.sum(mask_34[0]) == 1, "Input 0 should have 1 option (no uORF)"
        assert jnp.sum(mask_34[1]) == 8, "Input 1 should have 8 options (u3=all except none)"
        assert not mask_34[1, 0], "Input 1 should NOT have index 0 (none)"


def test_complex_twolayers_forward_pass(lib, complex_twolayers_design_network):
    """Test forward pass execution and output validation"""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(complex_twolayers_design_network, br.ALL_RULES, invert=True)
        network = networks[0]
        stack = ComputeStack([network])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        test_key = jax.random.PRNGKey(123)
        params = stack.init(test_key)

        inputs = jnp.array([1.0, 2.0])
        n_random_vars = params["global/number_of_random_variables"]
        random_vars = jax.random.normal(test_key, (n_random_vars,))

        stack_result, aux = stack.apply(params, inputs, random_vars, test_key)

        assert stack_result.shape == (4,), f"Expected 4 outputs, got shape {stack_result.shape}"

        assert jnp.all(jnp.isfinite(stack_result)), "All outputs should be finite"

        assert jnp.all(stack_result != 0), "Outputs should be non-zero for non-zero inputs"


def test_complex_twolayers_reproducibility(lib, complex_twolayers_design_network):
    """Test that same seed produces same results"""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(complex_twolayers_design_network, br.ALL_RULES, invert=True)
        network = networks[0]
        stack = ComputeStack([network])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        test_key = jax.random.PRNGKey(42)
        params = stack.init(test_key)

        inputs = jnp.array([1.0, 2.0])
        n_random_vars = params["global/number_of_random_variables"]
        random_vars = jax.random.normal(test_key, (n_random_vars,))

        result1, _ = stack.apply(params, inputs, random_vars, test_key)
        result2, _ = stack.apply(params, inputs, random_vars, test_key)

        assert jnp.allclose(result1, result2, rtol=1e-10), "Same seed should produce identical results"


def test_complex_twolayers_variability(lib, complex_twolayers_design_network):
    """Test that different seeds produce different results"""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(complex_twolayers_design_network, br.ALL_RULES, invert=True)
        network = networks[0]
        stack = ComputeStack([network])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        base_key = jax.random.PRNGKey(0)
        test_keys = jax.random.split(base_key, 10)

        all_results = []
        inputs = jnp.array([1.0, 2.0])

        for test_key in test_keys:
            params = stack.init(test_key)
            n_random_vars = params["global/number_of_random_variables"]
            random_vars = jax.random.normal(test_key, (n_random_vars,))

            result, _ = stack.apply(params, inputs, random_vars, test_key)
            all_results.append(result)

        all_results = jnp.array(all_results)

        std_devs = jnp.std(all_results, axis=0)
        assert jnp.all(std_devs > 5e-6), (
            f"All outputs should vary across different seeds, got std_devs: {std_devs}"
        )


def test_complex_twolayers_quantization_masks(lib, complex_twolayers_design_network):
    """Test that quantization masks are correctly set for uORF slots"""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(complex_twolayers_design_network, br.ALL_RULES, invert=True)
        network = networks[0]
        stack = ComputeStack([network])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)
        test_key = jax.random.PRNGKey(423)
        params = stack.init(test_key)

    assert stack.layers is not None
    assert stack.node_map is not None

    complex_twolayers_basic_stack_assertions(network, stack, params)
    u1_local_emb, u2_local_emb, u3_local_emb = complex_twolayers_topology_assertions(
        network, stack, params
    )

    x1cotx = None
    x2cotx = None
    bcotx = None
    for cotx in complex_twolayers_design_network.content:
        if cotx.name == "x1":
            x1cotx = cotx
        elif cotx.name == "x2":
            x2cotx = cotx
        elif cotx.name == "b":
            bcotx = cotx
    assert x1cotx is not None
    assert x2cotx is not None
    assert bcotx is not None

    tl_rate_values = params["shared/quantization/values/tl_rate"]
    assert tl_rate_values.shape == (13, 1)
    tc_rate_values = params["shared/quantization/values/tc_rate"]
    assert tc_rate_values.shape == (1, 1)
    ern_affinities = params["shared/ERN_5p/affinities"]

    def dummy_transform_fwd(value, qrate, rv_inner=0, rv_outer=0):
        value = np.atleast_1d(value).flatten()
        qrate = np.atleast_1d(qrate).flatten()
        if len(qrate) != len(value):
            if len(qrate) == 1:
                qrate = jnp.full_like(value, qrate[0])
            else:
                raise ValueError("qrate length does not match value length")
        rv_inner = np.atleast_1d(rv_inner).flatten()
        inner_array = flat_concat(value, qrate, rv_inner)
        inner_sum = jnp.sum(inner_array)
        inner_out = inner_sum * 8
        outer_input = flat_concat(jnp.array([inner_out]), rv_outer)
        outer_sum = jnp.sum(outer_input)
        return outer_sum

    def dummy_transform_inv(value, qrate, rv_inner=0, rv_outer=0):
        value = np.atleast_1d(value).flatten()
        qrate = np.atleast_1d(qrate).flatten()
        rv_inner = np.atleast_1d(rv_inner).flatten()
        rv_outer = np.atleast_1d(rv_outer).flatten()

        temp = jnp.sum(value) - jnp.sum(rv_outer)

        temp = temp / 8

        result = temp - jnp.sum(qrate) - jnp.sum(rv_inner)

        return result

    def dummy_source_fwd(value, position=0):
        return value * (0.9**position)

    def dummy_source_inv(value, position=0):
        return value / (0.9**position)

    def ern(positive_rna, negative_protein, layer_id):
        input_diff = jnp.sum(positive_rna) - jnp.sum(negative_protein)
        return input_diff * (0.9**layer_id)

    def aggregation_fwd(value, ratios):
        return value * jnp.abs(ratios)

    def aggregation_inv(value, ratio):
        return value / jnp.abs(ratio)

    def quantize(x, mask, values=tl_rate_values):
        masked_values = values[mask]
        return masked_values[jnp.argmin(jnp.abs(masked_values - x))]

    def inverse(x, marker_ratio, qrate):
        invtlx = dummy_transform_inv(x, qrate)
        invtrx = dummy_transform_inv(invtlx, tc_rate_values[0])
        invsrcx = dummy_source_inv(invtrx)
        invaggx = aggregation_inv(invsrcx, ratio=marker_ratio)
        return invaggx

    def fluo(value):
        value = np.atleast_1d(value).flatten()
        return jnp.sum(value)

    def forward(x, marker_ratio, qrate):
        aggx = aggregation_fwd(x, ratios=marker_ratio)
        srcx = dummy_source_fwd(aggx)
        trx = dummy_transform_fwd(srcx, tc_rate_values[0])
        tlx = dummy_transform_fwd(trx, qrate)
        fx = fluo(tlx)
        return fx

    def dummy_ern(ern_name, positive_rna, negative_prt, layer_id=0, rv=0):
        pos_sum = jnp.sum(positive_rna)
        neg_sum = jnp.sum(negative_prt)
        one_hot_layer_id = jax.nn.one_hot(layer_id, 3)
        layer_id_sum = jnp.sum(one_hot_layer_id)
        ern_name_id = ERNS.index(ern_name)
        affinity_val = ern_affinities[ern_name_id]
        return pos_sum + neg_sum + rv + layer_id_sum + jnp.sum(affinity_val)

    translate = dummy_transform_fwd
    transcribe = partial(dummy_transform_fwd, qrate=tc_rate_values[0])

    nouorf_emb = quantize(0.0, U1_EXPECTED_MASK)
    u1_emb = quantize(u1_local_emb, U1_EXPECTED_MASK)
    u2_emb = quantize(u2_local_emb, U2_EXPECTED_MASK)
    u3_emb = quantize(u3_local_emb, U3_EXPECTED_MASK)

    X = jax.random.uniform(test_key, shape=(2, 1))
    X1, X2 = X
    B = params["local/1/bias/raw_value"][0, 0]

    X1_inv = inverse(X1, x1ratios[0], qrate=nouorf_emb)
    X2_inv = inverse(X2, x2ratios[0], qrate=nouorf_emb)
    B_inv = inverse(B, 1 / 8, qrate=nouorf_emb)

    X1_back = forward(X1_inv, x1ratios[0], qrate=nouorf_emb)
    X2_back = forward(X2_inv, x2ratios[0], qrate=nouorf_emb)
    B_back = forward(B_inv, 1 / 8, qrate=nouorf_emb)

    assert jnp.allclose(X1, X1_back, rtol=1e-4), f"X1: {X1}, X1_back: {X1_back}"
    assert jnp.allclose(X2, X2_back, rtol=1e-4), f"X2: {X2}, X2_back: {X2_back}"
    assert jnp.allclose(B, B_back, rtol=1e-4), f"B: {B}, B_back: {B_back}"

    X1_apos = X1_inv * x1cotx.get_tu_ratio("x1_a+")
    X1_aneg = X1_inv * x1cotx.get_tu_ratio("x1_a-")
    X1_bpos = X1_inv * x1cotx.get_tu_ratio("x1_b+")
    X1_bneg = X1_inv * x1cotx.get_tu_ratio("x1_b-")
    X1_cpos = X1_inv * x1cotx.get_tu_ratio("x1_c+")
    X1_cneg = X1_inv * x1cotx.get_tu_ratio("x1_c-")
    X1_out = X1_inv * x1cotx.get_tu_ratio("x1_direct_out")
    X1_marker = X1_inv * x1cotx.get_tu_ratio("x1_marker")
    X2_apos = X2_inv * x2cotx.get_tu_ratio("x2_a+")
    X2_aneg = X2_inv * x2cotx.get_tu_ratio("x2_a-")
    X2_bpos = X2_inv * x2cotx.get_tu_ratio("x2_b+")
    X2_bneg = X2_inv * x2cotx.get_tu_ratio("x2_b-")
    X2_cpos = X2_inv * x2cotx.get_tu_ratio("x2_c+")
    X2_cneg = X2_inv * x2cotx.get_tu_ratio("x2_c-")
    X2_out = X2_inv * x2cotx.get_tu_ratio("x2_direct_out")
    X2_marker = X2_inv * x2cotx.get_tu_ratio("x2_marker")
    B_apos = B_inv * bcotx.get_tu_ratio("b_a+")
    B_aneg = B_inv * bcotx.get_tu_ratio("b_a-")
    B_bpos = B_inv * bcotx.get_tu_ratio("b_b+")
    B_bneg = B_inv * bcotx.get_tu_ratio("b_b-")
    B_cpos = B_inv * bcotx.get_tu_ratio("b_c+")
    B_cneg = B_inv * bcotx.get_tu_ratio("b_c-")
    B_out = B_inv * bcotx.get_tu_ratio("b_direct_out")
    B_marker = B_inv * bcotx.get_tu_ratio("b_marker")

    Ea_negative_in = translate(
        transcribe(flat_concat(X1_aneg, X2_aneg, B_aneg)),
        nouorf_emb,
    )
    Ea_positive_in = transcribe(flat_concat(X1_apos, X2_apos, B_apos))
    Ea = dummy_ern(
        ERNS[0],
        Ea_positive_in,
        Ea_negative_in,
        layer_id=0,
    )

    Eb_negative_in = translate(transcribe(flat_concat(X1_bneg, X2_bneg, B_bneg)), nouorf_emb)
    Eb_positive_in = transcribe(flat_concat(X1_bpos, X2_bpos, B_bpos))
    Eb = dummy_ern(
        ERNS[1],
        Eb_positive_in,
        Eb_negative_in,
        layer_id=0,
    )

    Ec_negative_in = translate(
        flat_concat(Ea, Eb, transcribe(flat_concat(X1_cneg, X2_cneg, B_cneg))),
        [u1_emb, u2_emb, nouorf_emb],
    )
    Ec_positive_in = transcribe(flat_concat(X1_cpos, X2_cpos, B_cpos))
    Ec = dummy_ern(
        ERNS[2],
        Ec_positive_in,
        Ec_negative_in,
        layer_id=1,
    )

    intermediate_values = {
        "x1_a+": X1_apos,
        "x1_a-": X1_aneg,
        "x1_b+": X1_bpos,
        "x1_b-": X1_bneg,
        "x1_c+": X1_cpos,
        "x1_c-": X1_cneg,
        "x1_direct_out": X1_out,
        "x1_marker": X1_marker,
        "x2_a+": X2_apos,
        "x2_a-": X2_aneg,
        "x2_b+": X2_bpos,
        "x2_b-": X2_bneg,
        "x2_c+": X2_cpos,
        "x2_c-": X2_cneg,
        "x2_direct_out": X2_out,
        "x2_marker": X2_marker,
        "b_a+": B_apos,
        "b_a-": B_aneg,
        "b_b+": B_bpos,
        "b_b-": B_bneg,
        "b_c+": B_cpos,
        "b_c-": B_cneg,
        "b_direct_out": B_out,
        "b_marker": B_marker,
        "Ea_negative_in": Ea_negative_in,
        "Ea_positive_in": Ea_positive_in,
        "Ea": Ea,
        "Eb_negative_in": Eb_negative_in,
        "Eb_positive_in": Eb_positive_in,
        "Eb": Eb,
        "Ec_negative_in": Ec_negative_in,
        "Ec_positive_in": Ec_positive_in,
        "Ec": Ec,
        "B_inv": B_inv,
        "X1_inv": X1_inv,
        "X2_inv": X2_inv,
    }

    flat_prt = flat_concat(Ec, transcribe(flat_concat(X1_out, X2_out, B_out)))
    flat_prt_rate = [u3_emb, nouorf_emb]
    manual_ydep = fluo(translate(flat_prt, flat_prt_rate))

    input_proteins = network.get_inverted_input_proteins(include_biases=True)
    pos_in_output = network.get_inverted_input_positions(include_biases=True)
    x1_marker_pos = pos_in_output[input_proteins.index(COLORS["x1"])]
    x2_marker_pos = pos_in_output[input_proteins.index(COLORS["x2"])]
    b_marker_pos = pos_in_output[input_proteins.index(COLORS["b"])]

    out_prots = network.get_output_proteins()

    manual_Y = np.zeros((len(out_prots),))
    manual_Y[x1_marker_pos] = X1_back
    manual_Y[x2_marker_pos] = X2_back
    manual_Y[b_marker_pos] = B_back
    manual_Y[out_prots.index(COLORS["y"])] = manual_ydep

    num_z = params["global/number_of_random_variables"]
    key = jax.random.PRNGKey(1234)
    Z = jnp.zeros((num_z,))
    assert stack.apply is not None
    Ycomp, (y_aux, flat_out) = stack.apply(params, X, Z, key)
    assert len(Ycomp) == len(manual_Y)

    complex_twolayers_advanced_stack_assertions(
        network,
        stack,
        params,
        translate,
        nouorf_emb,
        test_key,
        intermediate_values,
        y_aux,
    )

    assert np.allclose(Ycomp, manual_Y, rtol=1e-4), f"Ycomp: {Ycomp}, manual_Y: {manual_Y}"


if __name__ == "__main__":
    from biocomp.library import load_lib, LibraryContext

    lib_instance = load_lib()

    with LibraryContext.with_library(lib_instance):
        recipe_instance = Recipe(
            name="two_and_one",
            content=[
                CoTransfection(name="x1", units=make_units("x1", ERNS), ratios=x1ratios.tolist()),
                CoTransfection(name="x2", units=make_units("x2", ERNS), ratios=x2ratios.tolist()),
                CoTransfection(
                    name="b",
                    units=make_units("b", ERNS),
                    fluo_bias=BIAS_FLUO,
                    ratios=bratios.tolist(),
                ),
            ],
        )

    print("Running structure test...")
    test_complex_twolayers_structure(lib_instance, recipe_instance)
    print("✓ Structure test passed\n")

    print("Running parameter constraints test...")
    test_complex_twolayers_parameter_constraints(lib_instance, recipe_instance)
    print("✓ Parameter constraints test passed\n")

    print("Running forward pass test...")
    test_complex_twolayers_forward_pass(lib_instance, recipe_instance)
    print("✓ Forward pass test passed\n")

    print("Running reproducibility test...")
    test_complex_twolayers_reproducibility(lib_instance, recipe_instance)
    print("✓ Reproducibility test passed\n")

    print("Running variability test...")
    test_complex_twolayers_variability(lib_instance, recipe_instance)
    print("✓ Variability test passed\n")

    print("All tests passed!")
