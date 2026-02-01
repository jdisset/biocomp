import pytest
import jax

from biocomp.paramintrospect import (
    ParamKind,
    ParamValue,
    InputSlot,
    TUParamGroup,
    NodeParamInfo,
    is_tu_enabled,
    aggregate_by_tu,
    _fmt_value,
    _fmt_inputs,
    format_network_params,
    get_network_param_dict,
    introspect_stack,
    TU_THRESHOLD,
)
from biocomp.library import LibraryContext, load_lib
from biocomp.compute import ComputeStack
from biocomp.config import SIMPLE_NODES_COMPUTE_CONFIG
from biocomp.network import recipe_to_networks
from biocomp.recipe import Recipe, CoTransfection, TranscriptionUnit, Slot, NumRange
import biocomp.biorules as br


P = "hEF1a"
T = "L0.T_4560"
UORFS = [None, "1w_uORF", "1x_uORF", "2x_uORF", "3x_uORF"]


@pytest.fixture
def lib():
    return load_lib()


@pytest.fixture
def simple_recipe(lib):
    u = Slot(part=UORFS, ref_id="U1")
    with LibraryContext.with_library(lib):
        return Recipe(
            name="simple_introspect_test",
            content=[
                CoTransfection(
                    name="test_cotx",
                    units=[
                        TranscriptionUnit(
                            slots=[P, u, "CasE_rec", "mNeonGreen", T], name="output", source="p1"
                        ),
                        TranscriptionUnit(slots=[P, "CasE", T], name="ern", source="p2"),
                        TranscriptionUnit(slots=[P, "mKO2", T], name="marker", source="p3"),
                    ],
                    ratios=[
                        NumRange(min=0.2, max=0.5),
                        NumRange(min=0.3, max=0.6),
                        NumRange(min=0.1, max=0.3),
                    ],
                ),
            ],
        )


@pytest.fixture
def built_stack(lib, simple_recipe):
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(simple_recipe, br.ALL_RULES, invert=True)
        stack = ComputeStack(networks)
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG, enable_tu_masking=True)
        return stack


@pytest.fixture
def initialized_params(built_stack):
    key = jax.random.PRNGKey(42)
    return built_stack.init(key)


class TestIsTUEnabled:
    @pytest.mark.parametrize(
        "prob,expected",
        [
            (0.0, False),
            (0.49, False),
            (0.5, True),
            (0.51, True),
            (1.0, True),
        ],
    )
    def test_threshold_boundary(self, prob, expected):
        assert is_tu_enabled(prob) == expected
        assert is_tu_enabled(prob, TU_THRESHOLD) == expected

    def test_custom_threshold(self):
        assert is_tu_enabled(0.3, threshold=0.2) is True
        assert is_tu_enabled(0.3, threshold=0.4) is False


class TestFormatValue:
    def test_scalar(self):
        assert _fmt_value(0.12345) == "0.123"
        assert _fmt_value(0.12345, precision=2) == "0.12"

    def test_short_list(self):
        result = _fmt_value([0.1, 0.2, 0.3])
        assert "0.100" in result and "0.200" in result and "0.300" in result
        assert "|" in result

    def test_long_list_truncates(self):
        result = _fmt_value([float(i) for i in range(10)])
        assert "total" in result
        assert "10 total" in result


class TestFormatInputs:
    def test_empty(self):
        assert _fmt_inputs([]) == ""

    def test_single_masked(self):
        slots = [InputSlot(slot_idx=0, tu_id="tu1", is_masked=True, source_node="src")]
        result = _fmt_inputs(slots)
        assert "src: MASKED" in result

    def test_single_enabled(self):
        slots = [InputSlot(slot_idx=0, tu_id="tu1", is_masked=False, source_node="src")]
        result = _fmt_inputs(slots)
        assert "src: ON" in result

    def test_multiple_slots(self):
        slots = [
            InputSlot(slot_idx=0, tu_id="tu1", is_masked=False, source_node="neg"),
            InputSlot(slot_idx=1, tu_id="tu2", is_masked=True, source_node="pos"),
        ]
        result = _fmt_inputs(slots)
        assert "neg: ON" in result
        assert "pos: MASKED" in result

    def test_missing_source_uses_slot_fallback(self):
        slots = [InputSlot(slot_idx=3, tu_id=None, is_masked=False, source_node=None)]
        result = _fmt_inputs(slots)
        assert "slot_3: ON" in result


class TestAggregateByTU:
    def test_empty_list(self):
        assert aggregate_by_tu([]) == {}

    def test_single_node_single_tu(self):
        tg = TUParamGroup(tu_id="tu_x", is_enabled=True, prob=0.9)
        info = NodeParamInfo(
            node_type="aggregation",
            node_name="agg_0",
            network_id=0,
            tu_groups=[tg],
        )
        result = aggregate_by_tu([info])
        assert "tu_x" in result
        assert len(result["tu_x"]) == 1
        assert result["tu_x"][0][0] == "aggregation"
        assert result["tu_x"][0][1] is tg

    def test_multiple_nodes_same_tu(self):
        tg1 = TUParamGroup(tu_id="shared_tu", is_enabled=True, prob=0.8)
        tg2 = TUParamGroup(tu_id="shared_tu", is_enabled=True, prob=0.8)
        info1 = NodeParamInfo("aggregation", "agg", 0, tu_groups=[tg1])
        info2 = NodeParamInfo("translation", "tl", 0, tu_groups=[tg2])
        result = aggregate_by_tu([info1, info2])
        assert len(result["shared_tu"]) == 2

    def test_ignores_ungrouped(self):
        info = NodeParamInfo(
            node_type="bias",
            node_name="b",
            network_id=0,
            ungrouped=[ParamValue("bias", ParamKind.BIAS, 0.5)],
        )
        result = aggregate_by_tu([info])
        assert result == {}


class TestNodeParamInfoDataClass:
    def test_defaults(self):
        info = NodeParamInfo(node_type="test", node_name="n", network_id=0)
        assert info.tu_groups == []
        assert info.ungrouped == []
        assert info.ungrouped_inputs == []

    def test_with_all_fields(self):
        pv = ParamValue("rate", ParamKind.RATE, 0.5, bounds=(0.0, 1.0), quantized_to="2x_uORF")
        inp = InputSlot(0, "tu1", False, "source")
        tg = TUParamGroup("tu1", True, 0.9, params=[pv], inputs=[inp])
        info = NodeParamInfo("translation", "tl_1", 1, [tg], [], [])
        assert info.tu_groups[0].params[0].quantized_to == "2x_uORF"


class TestIntrospectStackReal:
    def test_introspect_returns_node_infos(self, built_stack, initialized_params):
        infos = introspect_stack(built_stack, initialized_params, network_id=0)
        assert isinstance(infos, list)
        for info in infos:
            assert isinstance(info, NodeParamInfo)
            assert info.network_id == 0

    def test_introspect_finds_translation_nodes(self, built_stack, initialized_params):
        infos = introspect_stack(built_stack, initialized_params, network_id=0)
        tl_infos = [i for i in infos if i.node_type == "tl"]
        assert len(tl_infos) > 0

    def test_introspect_finds_aggregation_nodes(self, built_stack, initialized_params):
        infos = introspect_stack(built_stack, initialized_params, network_id=0)
        agg_infos = [i for i in infos if i.node_type == "aggregation"]
        assert len(agg_infos) > 0

    def test_unbuilt_stack_raises(self, lib, simple_recipe):
        with LibraryContext.with_library(lib):
            networks = recipe_to_networks(simple_recipe, br.ALL_RULES, invert=True)
            stack = ComputeStack(networks)
            with pytest.raises(AssertionError, match="must be built"):
                introspect_stack(stack, {}, 0)

    def test_invalid_network_id_raises(self, built_stack, initialized_params):
        n_networks = len(built_stack.networks)
        with pytest.raises(AssertionError, match="out of range"):
            introspect_stack(built_stack, initialized_params, n_networks + 5)

        with pytest.raises(AssertionError, match="out of range"):
            introspect_stack(built_stack, initialized_params, -1)

    def test_cotx_group_populated(self, built_stack, initialized_params):
        infos = introspect_stack(built_stack, initialized_params, network_id=0)
        tu_data = aggregate_by_tu(infos)
        cotx_groups_found = set()
        for _tu_id, entries in tu_data.items():
            for _, tg in entries:
                cotx_groups_found.add(tg.cotx_group)
        assert "test_cotx" in cotx_groups_found, (
            f"Expected cotx group 'test_cotx', got: {cotx_groups_found}"
        )


class TestFormatNetworkParamsReal:
    def test_formats_real_network(self, built_stack, initialized_params):
        result = format_network_params(built_stack, initialized_params, network_id=0)
        assert isinstance(result, str)
        assert "Network:" in result
        assert len(result) > 50

    def test_contains_tu_info_or_ungrouped(self, built_stack, initialized_params):
        result = format_network_params(built_stack, initialized_params, network_id=0)
        has_tu = "TU:" in result
        has_ungrouped = any(
            x in result.upper() for x in ["TRANSLATION:", "TRANSCRIPTION:", "AGGREGATION:", "BIAS:"]
        )
        assert has_tu or has_ungrouped or "no introspectable" in result


class TestGetNetworkParamDictReal:
    def test_produces_serializable_dict(self, built_stack, initialized_params):
        result = get_network_param_dict(built_stack, initialized_params, network_id=0)

        assert isinstance(result, dict)
        assert "network_id" in result
        assert "nodes" in result
        assert result["network_id"] == 0
        assert len(result["nodes"]) > 0

    def test_nodes_have_expected_structure(self, built_stack, initialized_params):
        result = get_network_param_dict(built_stack, initialized_params, network_id=0)

        for node in result["nodes"]:
            assert "node_type" in node
            assert "node_name" in node
            assert "tu_groups" in node
            assert "ungrouped" in node
