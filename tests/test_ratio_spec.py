"""Tests for RatioSpec and dict-based ratio syntax."""

import pytest
from biocomp.recipe import (
    RatioSpec,
    NumRange,
    CoTransfection,
    TranscriptionUnit,
    Slot,
    Recipe,
    DEFAULT_RATIO_MIN,
    DEFAULT_RATIO_MAX,
)
from biocomp.library import load_lib, LibraryContext


@pytest.fixture
def lib():
    return load_lib()


# ============================================================================
# RatioSpec Unit Tests
# ============================================================================


class TestRatioSpecBasic:
    def test_locked_ratio(self):
        r = RatioSpec(value=0.5, locked=True)
        assert r.value == 0.5
        assert r.locked is True
        assert r.min is None
        assert r.max is None
        assert r.is_locked()
        assert r.to_num_range() is None

    def test_unlocked_ratio_with_custom_range(self):
        r = RatioSpec(value=0.3, min=0.1, max=0.6)
        assert r.value == 0.3
        assert r.locked is False
        assert r.min == 0.1
        assert r.max == 0.6
        assert not r.is_locked()
        nr = r.to_num_range()
        assert nr is not None
        assert nr.min == 0.1
        assert nr.max == 0.6

    def test_unlocked_ratio_default_range(self):
        r = RatioSpec(value=0.5, locked=False)
        assert r.value == 0.5
        assert r.locked is False
        assert r.min == DEFAULT_RATIO_MIN
        assert r.max == DEFAULT_RATIO_MAX
        assert not r.is_locked()

    def test_unlocked_ratio_partial_range_min_only(self):
        r = RatioSpec(value=0.5, min=0.2)
        assert r.min == 0.2
        assert r.max == DEFAULT_RATIO_MAX

    def test_unlocked_ratio_partial_range_max_only(self):
        r = RatioSpec(value=0.5, max=0.8)
        assert r.min == DEFAULT_RATIO_MIN
        assert r.max == 0.8


class TestRatioSpecValidation:
    def test_locked_with_min_raises(self):
        with pytest.raises(ValueError, match="locked=True is incompatible with min/max"):
            RatioSpec(value=0.5, locked=True, min=0.1)

    def test_locked_with_max_raises(self):
        with pytest.raises(ValueError, match="locked=True is incompatible with min/max"):
            RatioSpec(value=0.5, locked=True, max=0.8)

    def test_locked_with_both_min_max_raises(self):
        with pytest.raises(ValueError, match="locked=True is incompatible with min/max"):
            RatioSpec(value=0.5, locked=True, min=0.1, max=0.8)

    def test_min_greater_than_max_raises(self):
        with pytest.raises(ValueError, match="min.*>.*max"):
            RatioSpec(value=0.5, min=0.8, max=0.2)

    def test_value_below_min_raises(self):
        with pytest.raises(ValueError, match="value.*not in"):
            RatioSpec(value=0.05, min=0.1, max=0.9)

    def test_value_above_max_raises(self):
        with pytest.raises(ValueError, match="value.*not in"):
            RatioSpec(value=0.95, min=0.1, max=0.9)

    def test_value_at_min_boundary_ok(self):
        r = RatioSpec(value=0.1, min=0.1, max=0.9)
        assert r.value == 0.1

    def test_value_at_max_boundary_ok(self):
        r = RatioSpec(value=0.9, min=0.1, max=0.9)
        assert r.value == 0.9


# ============================================================================
# CoTransfection List Syntax Tests (backward compatibility)
# ============================================================================


class TestCoTransfectionListSyntax:
    def test_float_list_all_locked(self, lib):
        with LibraryContext.with_library(lib):
            cotx = CoTransfection(
                units=[
                    TranscriptionUnit(name="TU_A", slots=[Slot(part="eBFP2")]),
                    TranscriptionUnit(name="TU_B", slots=[Slot(part="mKO2")]),
                ],
                ratios=[0.5, 0.3],
            )
            assert cotx.ratios == [0.5, 0.3]
            assert not cotx.has_unlocked_ratios()
            assert cotx.get_locked_ratios() == [0.5, 0.3]
            assert cotx.get_ratio_ranges() == [None, None]

    def test_numrange_list_all_unlocked(self, lib):
        with LibraryContext.with_library(lib):
            cotx = CoTransfection(
                units=[
                    TranscriptionUnit(name="TU_A", slots=[Slot(part="eBFP2")]),
                    TranscriptionUnit(name="TU_B", slots=[Slot(part="mKO2")]),
                ],
                ratios=[NumRange(min=0.3, max=0.7), NumRange(min=0.1, max=0.5)],
            )
            assert cotx.has_unlocked_ratios()
            assert cotx.get_locked_ratios() is None
            ranges = cotx.get_ratio_ranges()
            assert ranges[0].min == 0.3 and ranges[0].max == 0.7
            assert ranges[1].min == 0.1 and ranges[1].max == 0.5

    def test_mixed_list_some_unlocked(self, lib):
        with LibraryContext.with_library(lib):
            cotx = CoTransfection(
                units=[
                    TranscriptionUnit(name="TU_A", slots=[Slot(part="eBFP2")]),
                    TranscriptionUnit(name="TU_B", slots=[Slot(part="mKO2")]),
                    TranscriptionUnit(name="TU_C", slots=[Slot(part="mMaroon1")]),
                ],
                ratios=[NumRange(min=0.2, max=0.6), 0.3, NumRange(min=0.1, max=0.4)],
            )
            assert cotx.has_unlocked_ratios()
            assert cotx.get_locked_ratios() is None
            ranges = cotx.get_ratio_ranges()
            assert ranges[0] is not None
            assert ranges[1] is None
            assert ranges[2] is not None


# ============================================================================
# CoTransfection Dict Syntax Tests (new feature)
# ============================================================================


class TestCoTransfectionDictSyntax:
    def test_dict_with_float_values(self, lib):
        with LibraryContext.with_library(lib):
            cotx = CoTransfection(
                units=[
                    TranscriptionUnit(name="TU_A", slots=[Slot(part="eBFP2")]),
                    TranscriptionUnit(name="TU_B", slots=[Slot(part="mKO2")]),
                ],
                ratios={"TU_A": 0.5, "TU_B": 0.3},
            )
            assert cotx.ratios == [0.5, 0.3]
            assert not cotx.has_unlocked_ratios()
            assert cotx.get_ratio_values() == [0.5, 0.3]

    def test_dict_with_ratio_spec_locked(self, lib):
        with LibraryContext.with_library(lib):
            cotx = CoTransfection(
                units=[
                    TranscriptionUnit(name="TU_A", slots=[Slot(part="eBFP2")]),
                    TranscriptionUnit(name="TU_B", slots=[Slot(part="mKO2")]),
                ],
                ratios={
                    "TU_A": {"value": 0.5, "locked": True},
                    "TU_B": {"value": 0.3, "locked": True},
                },
            )
            assert not cotx.has_unlocked_ratios()
            assert cotx.get_ratio_values() == [0.5, 0.3]
            assert cotx.get_ratio_ranges() == [None, None]

    def test_dict_with_ratio_spec_unlocked(self, lib):
        with LibraryContext.with_library(lib):
            cotx = CoTransfection(
                units=[
                    TranscriptionUnit(name="TU_A", slots=[Slot(part="eBFP2")]),
                    TranscriptionUnit(name="TU_B", slots=[Slot(part="mKO2")]),
                ],
                ratios={
                    "TU_A": {"value": 0.5, "min": 0.2, "max": 0.8},
                    "TU_B": {"value": 0.3, "min": 0.1, "max": 0.5},
                },
            )
            assert cotx.has_unlocked_ratios()
            assert cotx.get_ratio_values() == [0.5, 0.3]
            ranges = cotx.get_ratio_ranges()
            assert ranges[0].min == 0.2 and ranges[0].max == 0.8
            assert ranges[1].min == 0.1 and ranges[1].max == 0.5

    def test_dict_with_mixed_locked_unlocked(self, lib):
        with LibraryContext.with_library(lib):
            cotx = CoTransfection(
                units=[
                    TranscriptionUnit(name="TU_A", slots=[Slot(part="eBFP2")]),
                    TranscriptionUnit(name="TU_B", slots=[Slot(part="mKO2")]),
                    TranscriptionUnit(name="TU_C", slots=[Slot(part="mMaroon1")]),
                ],
                ratios={
                    "TU_A": {"value": 0.5, "min": 0.2, "max": 0.8},
                    "TU_B": {"value": 0.3, "locked": True},
                    "TU_C": 0.2,
                },
            )
            assert cotx.has_unlocked_ratios()
            assert cotx.get_ratio_values() == [0.5, 0.3, 0.2]
            ranges = cotx.get_ratio_ranges()
            assert ranges[0] is not None
            assert ranges[1] is None
            assert ranges[2] is None


class TestCoTransfectionDictValidation:
    def test_missing_tu_name_raises(self, lib):
        with LibraryContext.with_library(lib):
            with pytest.raises(ValueError, match="Ratio dict missing TU"):
                CoTransfection(
                    units=[
                        TranscriptionUnit(name="TU_A", slots=[Slot(part="eBFP2")]),
                        TranscriptionUnit(name="TU_B", slots=[Slot(part="mKO2")]),
                    ],
                    ratios={"TU_A": 0.5},
                )

    def test_extra_tu_name_raises(self, lib):
        with LibraryContext.with_library(lib):
            with pytest.raises(ValueError, match="unknown TU names"):
                CoTransfection(
                    units=[
                        TranscriptionUnit(name="TU_A", slots=[Slot(part="eBFP2")]),
                    ],
                    ratios={"TU_A": 0.5, "TU_X": 0.3},
                )


# ============================================================================
# Equivalence Tests: List vs Dict Syntax
# ============================================================================


class TestEquivalence:
    def test_locked_float_list_equals_dict(self, lib):
        with LibraryContext.with_library(lib):
            units = [
                TranscriptionUnit(name="TU_A", slots=[Slot(part="eBFP2")]),
                TranscriptionUnit(name="TU_B", slots=[Slot(part="mKO2")]),
            ]
            cotx_list = CoTransfection(units=units, ratios=[0.5, 0.3])
            cotx_dict = CoTransfection(units=units, ratios={"TU_A": 0.5, "TU_B": 0.3})

            assert cotx_list.get_ratio_values() == cotx_dict.get_ratio_values()
            assert cotx_list.has_unlocked_ratios() == cotx_dict.has_unlocked_ratios()
            assert cotx_list.get_ratio_ranges() == cotx_dict.get_ratio_ranges()

    def test_numrange_list_equals_dict_spec(self, lib):
        with LibraryContext.with_library(lib):
            units = [
                TranscriptionUnit(name="TU_A", slots=[Slot(part="eBFP2")]),
                TranscriptionUnit(name="TU_B", slots=[Slot(part="mKO2")]),
            ]
            cotx_list = CoTransfection(
                units=units, ratios=[NumRange(min=0.2, max=0.8), NumRange(min=0.1, max=0.5)]
            )
            cotx_dict = CoTransfection(
                units=units,
                ratios={
                    "TU_A": {"value": 0.5, "min": 0.2, "max": 0.8},
                    "TU_B": {"value": 0.3, "min": 0.1, "max": 0.5},
                },
            )

            assert cotx_list.has_unlocked_ratios() == cotx_dict.has_unlocked_ratios()
            list_ranges = cotx_list.get_ratio_ranges()
            dict_ranges = cotx_dict.get_ratio_ranges()
            assert list_ranges[0].min == dict_ranges[0].min
            assert list_ranges[0].max == dict_ranges[0].max
            assert list_ranges[1].min == dict_ranges[1].min
            assert list_ranges[1].max == dict_ranges[1].max


# ============================================================================
# Integration with Network Building
# ============================================================================


class TestNetworkIntegration:
    def test_dict_ratios_build_network(self, lib):
        from biocomp.network import recipe_to_networks
        import biocomp.biorules as br

        with LibraryContext.with_library(lib):
            recipe = Recipe(
                name="test_dict_ratios",
                content=[
                    CoTransfection(
                        units=[
                            TranscriptionUnit(
                                name="TU_A",
                                slots=[
                                    Slot(part="cHS4"),
                                    Slot(part="hEF1a"),
                                    Slot(part="eBFP2"),
                                    Slot(part="L0.T_4560"),
                                ],
                            ),
                            TranscriptionUnit(
                                name="TU_B",
                                slots=[
                                    Slot(part="cHS4"),
                                    Slot(part="hEF1a"),
                                    Slot(part="mKO2"),
                                    Slot(part="L0.T_4560"),
                                ],
                            ),
                        ],
                        ratios={"TU_A": 0.6, "TU_B": 0.4},
                    )
                ],
            )
            networks = recipe_to_networks(recipe, br.ALL_RULES, invert=False)
            assert len(networks) == 1
            compg = networks[0].compute_graph
            agg_nodes = [n for n in compg.nodes.values() if n.node_type == "aggregation"]
            assert len(agg_nodes) == 1
            members = agg_nodes[0].extra["members"]
            sorted_ids = sorted(members.keys())
            ratios = [members[m]["ratio"] for m in sorted_ids]
            assert ratios == [0.6, 0.4]

    def test_locked_ratios_no_ratio_ranges(self, lib):
        from biocomp.network import recipe_to_networks
        import biocomp.biorules as br

        with LibraryContext.with_library(lib):
            recipe = Recipe(
                name="test_locked",
                content=[
                    CoTransfection(
                        units=[
                            TranscriptionUnit(
                                name="TU_A",
                                slots=[
                                    Slot(part="cHS4"),
                                    Slot(part="hEF1a"),
                                    Slot(part="eBFP2"),
                                    Slot(part="L0.T_4560"),
                                ],
                            ),
                            TranscriptionUnit(
                                name="TU_B",
                                slots=[
                                    Slot(part="cHS4"),
                                    Slot(part="hEF1a"),
                                    Slot(part="mKO2"),
                                    Slot(part="L0.T_4560"),
                                ],
                            ),
                        ],
                        ratios={
                            "TU_A": {"value": 0.6, "locked": True},
                            "TU_B": {"value": 0.4, "locked": True},
                        },
                    )
                ],
            )
            networks = recipe_to_networks(recipe, br.ALL_RULES, invert=False)
            compg = networks[0].compute_graph
            agg_nodes = [n for n in compg.nodes.values() if n.node_type == "aggregation"]
            agg = agg_nodes[0]
            members = agg.extra["members"]
            sorted_ids = sorted(members.keys())
            ratios = [members[m]["ratio"] for m in sorted_ids]
            assert ratios == [0.6, 0.4]
            ratio_ranges = [members[m].get("ratio_range") for m in sorted_ids]
            assert all(r is None for r in ratio_ranges)

    def test_unlocked_ratios_have_ratio_ranges(self, lib):
        from biocomp.network import recipe_to_networks
        import biocomp.biorules as br

        with LibraryContext.with_library(lib):
            recipe = Recipe(
                name="test_unlocked",
                content=[
                    CoTransfection(
                        units=[
                            TranscriptionUnit(
                                name="TU_A",
                                slots=[
                                    Slot(part="cHS4"),
                                    Slot(part="hEF1a"),
                                    Slot(part="eBFP2"),
                                    Slot(part="L0.T_4560"),
                                ],
                            ),
                            TranscriptionUnit(
                                name="TU_B",
                                slots=[
                                    Slot(part="cHS4"),
                                    Slot(part="hEF1a"),
                                    Slot(part="mKO2"),
                                    Slot(part="L0.T_4560"),
                                ],
                            ),
                        ],
                        ratios={
                            "TU_A": {"value": 0.6, "min": 0.3, "max": 0.9},
                            "TU_B": {"value": 0.4, "locked": True},
                        },
                    )
                ],
            )
            networks = recipe_to_networks(recipe, br.ALL_RULES, invert=False)
            compg = networks[0].compute_graph
            agg_nodes = [n for n in compg.nodes.values() if n.node_type == "aggregation"]
            agg = agg_nodes[0]
            members = agg.extra["members"]
            sorted_ids = sorted(members.keys())
            ratio_ranges = [members[m].get("ratio_range") for m in sorted_ids]
            assert ratio_ranges[0] is not None
            assert ratio_ranges[0]["min"] == 0.3
            assert ratio_ranges[0]["max"] == 0.9
            assert ratio_ranges[1] is None


# ============================================================================
# Repr and Display
# ============================================================================


class TestRepr:
    def test_ratio_spec_locked_repr(self):
        r = RatioSpec(value=0.5, locked=True)
        assert "locked" in repr(r)
        assert "0.5" in repr(r)

    def test_ratio_spec_unlocked_repr(self):
        r = RatioSpec(value=0.5, min=0.2, max=0.8)
        assert "0.5" in repr(r)
        assert "0.2" in repr(r)
        assert "0.8" in repr(r)


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    def test_single_tu_no_ratios(self, lib):
        with LibraryContext.with_library(lib):
            cotx = CoTransfection(
                units=[TranscriptionUnit(name="TU_A", slots=[Slot(part="eBFP2")])],
            )
            assert cotx.ratios is None
            assert not cotx.has_unlocked_ratios()
            assert cotx.get_ratio_values() == [1.0]

    def test_empty_units_with_ratios_dict_raises(self, lib):
        with LibraryContext.with_library(lib):
            with pytest.raises(ValueError):
                CoTransfection(units=[], ratios={"TU_A": 0.5})

    def test_ratio_precision(self, lib):
        with LibraryContext.with_library(lib):
            cotx = CoTransfection(
                units=[
                    TranscriptionUnit(name="TU_A", slots=[Slot(part="eBFP2")]),
                ],
                ratios=[0.123456789],
            )
            assert cotx.ratios[0] == 0.12346

    def test_dict_ordering_matches_units(self, lib):
        with LibraryContext.with_library(lib):
            cotx = CoTransfection(
                units=[
                    TranscriptionUnit(name="TU_B", slots=[Slot(part="mKO2")]),
                    TranscriptionUnit(name="TU_A", slots=[Slot(part="eBFP2")]),
                ],
                ratios={"TU_A": 0.3, "TU_B": 0.7},
            )
            assert cotx.get_ratio_values() == [0.7, 0.3]
