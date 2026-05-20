# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
import pytest
from pydantic import ValidationError

from biocomp.library import LibraryContext, load_lib
from biocomp.network import recipe_to_networks
from biocomp.recipe import (
    CoTransfection,
    InputAxis,
    Recipe,
    Slot,
    TranscriptionUnit,
)


@pytest.fixture
def lib():
    return load_lib()


def _marker_cotx(name: str, protein: str) -> CoTransfection:
    return CoTransfection(
        name=name,
        units=[
            TranscriptionUnit(
                name=f"{protein}_reporter",
                slots=[
                    Slot(part="cHS4"),
                    Slot(part="hEF1a"),
                    Slot(part=protein),
                    Slot(part="L0.T_4560"),
                ],
            )
        ],
    )


@pytest.fixture
def two_marker_recipe(lib):
    with LibraryContext.with_library(lib):
        yield Recipe(
            name="two",
            content=[_marker_cotx("cotx1", "eBFP2"), _marker_cotx("cotx2", "mKO2")],
        )


class TestParsing:
    def test_list_of_strings(self):
        r = Recipe(name="r", content=[], input_axes=["a", "b"])
        assert r.input_axes == [InputAxis(name="a"), InputAxis(name="b")]

    def test_list_of_dicts(self):
        r = Recipe(name="r", content=[], input_axes=[{"name": "a", "axis": "x"}, {"name": "b"}])
        assert r.input_axes == [InputAxis(name="a", axis="x"), InputAxis(name="b")]

    def test_list_of_objects(self):
        axes = [InputAxis(name="a", axis="y"), InputAxis(name="b", axis="x")]
        r = Recipe(name="r", content=[], input_axes=axes)
        assert r.input_axes == axes

    def test_dict_shorthand_preserves_order(self):
        r = Recipe(name="r", content=[], input_axes={"cotx1": "y", "cotx2": "x"})
        assert [(ax.name, ax.axis) for ax in r.input_axes] == [("cotx1", "y"), ("cotx2", "x")]

    def test_mixed_str_and_dict(self):
        r = Recipe(name="r", content=[], input_axes=["a", {"name": "b", "axis": "x"}])
        assert r.input_axes == [InputAxis(name="a"), InputAxis(name="b", axis="x")]

    def test_none_stays_none(self):
        r = Recipe(name="r", content=[])
        assert r.input_axes is None
        assert not r.has_input_axes()
        assert r.input_order is None
        assert r.axis_mapping is None

    def test_z_axis_allowed(self):
        r = Recipe(name="r", content=[], input_axes={"a": "x", "b": "y", "c": "z"})
        assert r.input_axes[2].axis == "z"

    def test_invalid_axis_label_rejected(self):
        with pytest.raises(ValidationError):
            Recipe(name="r", content=[], input_axes=[{"name": "a", "axis": "w"}])

    def test_unsupported_top_level_type_rejected(self):
        with pytest.raises(ValidationError):
            Recipe(name="r", content=[], input_axes=42)


class TestValidation:
    def test_duplicate_names_rejected(self):
        with pytest.raises((AssertionError, ValidationError), match="duplicate"):
            Recipe(name="r", content=[], input_axes=["a", "a"])

    def test_duplicate_axis_labels_rejected(self):
        with pytest.raises((AssertionError, ValidationError), match="duplicate"):
            Recipe(name="r", content=[], input_axes={"a": "x", "b": "x"})

    def test_partial_axis_labels_ok(self):
        r = Recipe(name="r", content=[], input_axes=[{"name": "a", "axis": "x"}, "b"])
        assert r.input_axes[0].axis == "x"
        assert r.input_axes[1].axis is None


class TestLegacyMigration:
    def test_legacy_input_order_only(self):
        r = Recipe(name="r", content=[], input_order=["a", "b"])
        assert r.input_axes == [InputAxis(name="a"), InputAxis(name="b")]
        assert r.input_order == ["a", "b"]
        assert r.axis_mapping is None

    def test_legacy_axis_mapping_only(self):
        r = Recipe(name="r", content=[], axis_mapping={"cotx1": "x", "cotx2": "y"})
        assert r.input_axes == [
            InputAxis(name="cotx1", axis="x"),
            InputAxis(name="cotx2", axis="y"),
        ]
        assert r.axis_mapping == {"cotx1": "x", "cotx2": "y"}

    def test_legacy_both_fields_merge(self):
        r = Recipe(
            name="r",
            content=[],
            input_order=["eBFP2", "mKO2"],
            axis_mapping={"cotx1": "y", "cotx2": "x"},
        )
        names = [ax.name for ax in r.input_axes]
        assert set(names) == {"eBFP2", "mKO2", "cotx1", "cotx2"}
        cotx1 = next(ax for ax in r.input_axes if ax.name == "cotx1")
        assert cotx1.axis == "y"

    def test_new_field_wins_over_legacy(self):
        r = Recipe(
            name="r",
            content=[],
            input_axes=["only_this"],
            input_order=["ignored"],
            axis_mapping={"also_ignored": "x"},
        )
        assert [ax.name for ax in r.input_axes] == ["only_this"]


class TestPropertiesDerived:
    def test_input_order_property(self):
        r = Recipe(name="r", content=[], input_axes=[{"name": "a", "axis": "x"}, "b"])
        assert r.input_order == ["a", "b"]

    def test_axis_mapping_property_only_labeled(self):
        r = Recipe(name="r", content=[], input_axes=[{"name": "a", "axis": "x"}, "b"])
        assert r.axis_mapping == {"a": "x"}

    def test_axis_mapping_returns_none_when_no_labels(self):
        r = Recipe(name="r", content=[], input_axes=["a", "b"])
        assert r.axis_mapping is None


class TestResolutionAgainstNetwork:
    def test_protein_names_pass_through(self, lib, two_marker_recipe):
        with LibraryContext.with_library(lib):
            net = recipe_to_networks(two_marker_recipe, invert=True)[0]
            two_marker_recipe.input_axes = ["mKO2", "eBFP2"]
            resolved = two_marker_recipe.resolve_input_order(net)
            assert resolved == ["mKO2", "eBFP2"]

    def test_cotx_names_resolve_to_proteins(self, lib, two_marker_recipe):
        with LibraryContext.with_library(lib):
            net = recipe_to_networks(two_marker_recipe, invert=True)[0]
            two_marker_recipe.input_axes = {"cotx2": "x", "cotx1": "y"}
            resolved = two_marker_recipe.resolve_input_order(net)
            assert resolved == ["mKO2", "eBFP2"]

    def test_mixed_names_resolve(self, lib, two_marker_recipe):
        with LibraryContext.with_library(lib):
            net = recipe_to_networks(two_marker_recipe, invert=True)[0]
            two_marker_recipe.input_axes = [
                {"name": "cotx1", "axis": "y"},
                {"name": "mKO2", "axis": "x"},
            ]
            assert two_marker_recipe.resolve_input_order(net) == ["eBFP2", "mKO2"]

    def test_unknown_name_raises(self, lib, two_marker_recipe):
        with LibraryContext.with_library(lib):
            net = recipe_to_networks(two_marker_recipe, invert=True)[0]
            two_marker_recipe.input_axes = ["UNKNOWN_PROTEIN"]
            with pytest.raises(ValueError, match="matches neither"):
                two_marker_recipe.resolve_input_order(net)


class TestRecipeToNetworkFlow:
    def test_input_axes_applied_to_network(self, lib, two_marker_recipe):
        with LibraryContext.with_library(lib):
            two_marker_recipe.input_axes = [
                {"name": "cotx1", "axis": "y"},
                {"name": "cotx2", "axis": "x"},
            ]
            net = recipe_to_networks(two_marker_recipe, invert=True)[0]
            assert net.has_input_axes()
            axes = net.get_input_axes()
            assert [ax.name for ax in axes] == ["eBFP2", "mKO2"]
            assert axes[0].axis == "y"
            assert axes[1].axis == "x"
            assert net.get_inverted_input_proteins() == ["eBFP2", "mKO2"]

    def test_legacy_axis_mapping_yaml_still_works(self, lib, two_marker_recipe):
        with LibraryContext.with_library(lib):
            two_marker_recipe.input_axes = None
            two_marker_recipe.input_axes = {"cotx1": "y", "cotx2": "x"}
            net = recipe_to_networks(two_marker_recipe, invert=True)[0]
            order = net.get_inverted_input_proteins()
            assert order == ["eBFP2", "mKO2"]

    def test_round_trip_through_to_recipe(self, lib, two_marker_recipe):
        with LibraryContext.with_library(lib):
            two_marker_recipe.input_axes = [
                {"name": "mKO2", "axis": "x"},
                {"name": "eBFP2", "axis": "y"},
            ]
            net = recipe_to_networks(two_marker_recipe, invert=True)[0]
            round_tripped = net.to_recipe()
            assert round_tripped.input_axes is not None
            assert [ax.name for ax in round_tripped.input_axes] == ["mKO2", "eBFP2"]
            assert round_tripped.axis_mapping == {"mKO2": "x", "eBFP2": "y"}


class TestInputAxisModel:
    def test_str_parsed_as_name_only(self):
        ax = InputAxis.model_validate("foo")
        assert ax == InputAxis(name="foo", axis=None)

    def test_idempotent_construction(self):
        ax = InputAxis(name="foo", axis="y")
        ax2 = InputAxis.model_validate(ax)
        assert ax2 == ax
