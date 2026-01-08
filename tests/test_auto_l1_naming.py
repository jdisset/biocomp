"""Tests for auto_name_from_l1 feature in Network.to_recipe()

Tests the automatic TU naming based on L1-level construct matching from parts database.
"""

import pytest
from biocomp.library import LibraryContext, load_lib, get_l1_parts
from biocomp.recipe import Recipe, CoTransfection, TranscriptionUnit, Slot, name_transcription_unit
from biocomp.network import recipe_to_networks


@pytest.fixture
def lib():
    """Load the parts library"""
    return load_lib()


def get_sample_l1_from_library(lib):
    """Find an L1 in the library that has all parts resolvable for testing"""
    for l1_name in lib.L1s.index[:50]:
        parts = get_l1_parts(l1_name, lib)
        if parts and len(parts) >= 2 and all(p for p in parts):
            return l1_name, parts
    return None, None


def test_to_recipe_auto_name_from_l1_matches(lib):
    """TUs matching L1 constructs get L1 names"""
    l1_name, l1_parts = get_sample_l1_from_library(lib)
    if l1_name is None:
        pytest.skip("No suitable L1 found in library for testing")

    with LibraryContext.with_library(lib):
        recipe = Recipe(
            name="test_l1_match",
            content=[
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            name="original_name",
                            slots=[Slot(part=p) for p in l1_parts],
                        ),
                    ],
                )
            ],
        )

        networks = recipe_to_networks(recipe)
        exported = networks[0].to_recipe(auto_name_from_l1=True)

        tu = exported.content[0].units[0]
        assert tu.name == l1_name, f"Expected L1 name '{l1_name}', got '{tu.name}'"


def test_to_recipe_auto_name_from_l1_fallback(lib):
    """TUs NOT matching any L1 get generic tu_N names"""
    with LibraryContext.with_library(lib):
        recipe = Recipe(
            name="test_no_l1_match",
            content=[
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            name="custom_tu",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="eBFP2"),
                                Slot(part="mNeonGreen"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                    ],
                )
            ],
        )

        networks = recipe_to_networks(recipe)
        exported = networks[0].to_recipe(auto_name_from_l1=True)

        tu = exported.content[0].units[0]
        assert tu.name == "tu_1", f"Expected 'tu_1', got '{tu.name}'"


def test_to_recipe_default_preserves_names(lib):
    """Default (auto_name_from_l1=False) preserves original TU names"""
    with LibraryContext.with_library(lib):
        recipe = Recipe(
            name="test_preserve_names",
            content=[
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            name="my_custom_name",
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

        networks = recipe_to_networks(recipe)
        exported = networks[0].to_recipe()

        tu = exported.content[0].units[0]
        assert tu.name == "my_custom_name", f"Expected 'my_custom_name', got '{tu.name}'"


def test_to_recipe_auto_name_mixed(lib):
    """Recipe with both matching and non-matching TUs"""
    l1_name, l1_parts = get_sample_l1_from_library(lib)
    if l1_name is None:
        pytest.skip("No suitable L1 found in library for testing")

    with LibraryContext.with_library(lib):
        recipe = Recipe(
            name="test_mixed",
            content=[
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            name="will_match_l1",
                            slots=[Slot(part=p) for p in l1_parts],
                            source="plasmid_A",
                        ),
                        TranscriptionUnit(
                            name="will_not_match",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="eBFP2"),
                                Slot(part="mNeonGreen"),
                                Slot(part="L0.T_4560"),
                            ],
                            source="plasmid_B",
                        ),
                    ],
                )
            ],
        )

        networks = recipe_to_networks(recipe)
        exported = networks[0].to_recipe(auto_name_from_l1=True)

        tus = exported.content[0].units
        assert len(tus) == 2

        l1_matched = [tu for tu in tus if tu.name == l1_name]
        generic_named = [tu for tu in tus if tu.name.startswith("tu_")]

        assert len(l1_matched) == 1, f"Expected 1 L1-matched TU, got {len(l1_matched)}"
        assert len(generic_named) == 1, f"Expected 1 generic-named TU, got {len(generic_named)}"


def test_to_recipe_auto_name_counter_across_cotx(lib):
    """Generic names are unique across all CoTransfection groups"""
    with LibraryContext.with_library(lib):
        recipe = Recipe(
            name="test_counter_across_cotx",
            content=[
                CoTransfection(
                    name="cotx_1",
                    units=[
                        TranscriptionUnit(
                            name="tu_a",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="eBFP2"),
                                Slot(part="mNeonGreen"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                        TranscriptionUnit(
                            name="tu_b",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="mNeonGreen"),
                                Slot(part="mMaroon1"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                    ],
                ),
                CoTransfection(
                    name="cotx_2",
                    units=[
                        TranscriptionUnit(
                            name="tu_c",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="mMaroon1"),
                                Slot(part="eBFP2"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                        TranscriptionUnit(
                            name="tu_d",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="mKate"),
                                Slot(part="eBFP2"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                    ],
                ),
            ],
        )

        networks = recipe_to_networks(recipe)
        exported = networks[0].to_recipe(auto_name_from_l1=True)

        all_tu_names = []
        for cotx in exported.content:
            for tu in cotx.units:
                all_tu_names.append(tu.name)

        assert len(all_tu_names) == len(set(all_tu_names)), (
            f"Duplicate TU names found: {all_tu_names}"
        )

        expected_names = {"tu_1", "tu_2", "tu_3", "tu_4"}
        actual_names = set(all_tu_names)
        assert actual_names == expected_names, f"Expected {expected_names}, got {actual_names}"


def test_to_recipe_auto_name_roundtrip(lib):
    """Exported recipe can be rebuilt and re-exported with same names"""
    with LibraryContext.with_library(lib):
        recipe = Recipe(
            name="test_roundtrip",
            content=[
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            name="original",
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

        networks = recipe_to_networks(recipe)
        exported1 = networks[0].to_recipe(auto_name_from_l1=True)

        networks2 = recipe_to_networks(exported1)
        exported2 = networks2[0].to_recipe(auto_name_from_l1=True)

        assert exported1.content[0].units[0].name == exported2.content[0].units[0].name


def test_name_transcription_unit_no_match(lib):
    """name_transcription_unit returns None for non-matching TUs"""
    with LibraryContext.with_library(lib):
        tu = TranscriptionUnit(
            name="test",
            slots=[
                Slot(part="cHS4"),
                Slot(part="hEF1a"),
                Slot(part="eBFP2"),
                Slot(part="mNeonGreen"),
                Slot(part="L0.T_4560"),
            ],
        )

        result = name_transcription_unit(tu, lib)
        assert result is None, f"Expected None, got '{result}'"


def test_name_transcription_unit_with_match(lib):
    """name_transcription_unit returns L1 name for matching TUs"""
    l1_name, l1_parts = get_sample_l1_from_library(lib)
    if l1_name is None:
        pytest.skip("No suitable L1 found in library for testing")

    with LibraryContext.with_library(lib):
        tu = TranscriptionUnit(
            name="test",
            slots=[Slot(part=p) for p in l1_parts],
        )

        result = name_transcription_unit(tu, lib)
        assert result == l1_name, f"Expected '{l1_name}', got '{result}'"
