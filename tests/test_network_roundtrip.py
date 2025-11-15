from biocomp.library import LibraryContext
from biocomp.recipe import Recipe, CoTransfection, TranscriptionUnit, Slot
from biocomp.network import recipe_to_networks
import pytest
from test_declarative_recipes import (  # noqa: F401
    lib,
    simple_single_reporter,
    simple_two_reporters,
    simple_single_ern,
    simple_single_cotx_ERN,
    multi_aggregation_ern,
    variable_uorf_network,
    simple_aggregation,
    multi_cotx_aggregation,
    complex_ern_network,
    uorf_ern_network,
    unlocked_ratios_network,
    bias_network,
    unlocked_bias_network,
    combined_unlocked_network,
    ern_with_unlocked_ratios,
    ern_with_unlocked_bias,
    ern_with_unlocked_uorfs,
    complex_mixed_unlocked,
)
from test_complex_twolayers_computation import (  # noqa: F401
    complex_twolayers_design_network,
)


def recipe_equals(r1: Recipe, r2: Recipe, check_name: bool = False) -> bool:
    try:
        recipe_equals_assert(r1, r2, check_name=check_name)
        return True
    except AssertionError:
        return False


def cotx_equals(c1: CoTransfection, c2: CoTransfection) -> bool:
    try:
        cotx_equals_assert(c1, c2, path="CoTransfection")
        return True
    except AssertionError:
        return False


def tu_equals(u1: TranscriptionUnit, u2: TranscriptionUnit) -> bool:
    try:
        tu_equals_assert(u1, u2, path="TranscriptionUnit")
        return True
    except AssertionError:
        return False


def recipe_equals_assert(r1: Recipe, r2: Recipe, check_name: bool = False, path: str = "Recipe"):
    """Assert-based recipe comparison with detailed error messages"""
    if check_name:
        assert r1.name == r2.name, f"{path}: Name mismatch ('{r1.name}' vs '{r2.name}')"

    assert len(r1.content) == len(r2.content), (
        f"{path}: CoTransfection count mismatch ({len(r1.content)} vs {len(r2.content)})"
    )

    for i, (c1, c2) in enumerate(zip(r1.content, r2.content)):
        cotx_path = f"{path}/CoTx[{i}:{c1.name if hasattr(c1, 'name') and c1.name else 'unnamed'}]"
        cotx_equals_assert(c1, c2, path=cotx_path)


def cotx_equals_assert(c1: CoTransfection, c2: CoTransfection, path: str):
    """Assert-based CoTransfection comparison with detailed error messages"""
    from biocomp.recipe import NumRange

    assert len(c1.units) == len(c2.units), (
        f"{path}: Unit count mismatch ({len(c1.units)} vs {len(c2.units)})"
    )

    # Default ratios should be based on unique sources, not total units
    def get_unique_source_count(cotx):
        seen = set()
        for tu in cotx.units:
            seen.add(tu.source)
        return len(seen)

    num_sources1 = get_unique_source_count(c1)
    num_sources2 = get_unique_source_count(c2)
    r1 = c1.ratios or [1.0] * num_sources1
    r2 = c2.ratios or [1.0] * num_sources2

    assert len(r1) == len(r2), f"{path}: Ratio count mismatch ({len(r1)} vs {len(r2)})"

    numeric_indices = []
    for i, (ratio1, ratio2) in enumerate(zip(r1, r2)):
        if isinstance(ratio1, NumRange) and isinstance(ratio2, NumRange):
            assert ratio1.min == ratio2.min, (
                f"{path}/Ratio[{i}]: NumRange.min mismatch ({ratio1.min} vs {ratio2.min})"
            )
            assert ratio1.max == ratio2.max, (
                f"{path}/Ratio[{i}]: NumRange.max mismatch ({ratio1.max} vs {ratio2.max})"
            )
        elif isinstance(ratio1, NumRange) or isinstance(ratio2, NumRange):
            type1 = type(ratio1).__name__
            type2 = type(ratio2).__name__
            assert False, f"{path}/Ratio[{i}]: Type mismatch ({type1} vs {type2})"
        else:
            numeric_indices.append(i)

    if c1.fluo_bias is None and c2.fluo_bias is None:
        pass
    elif c1.fluo_bias is not None and c2.fluo_bias is not None:
        fb1, fb2 = c1.fluo_bias, c2.fluo_bias
        assert fb1.tu_id == fb2.tu_id, (
            f"{path}/fluo_bias: tu_id mismatch ({fb1.tu_id} vs {fb2.tu_id})"
        )
        assert fb1.protein == fb2.protein, (
            f"{path}/fluo_bias: protein mismatch ('{fb1.protein}' vs '{fb2.protein}')"
        )
        assert fb1.units == fb2.units, (
            f"{path}/fluo_bias: units mismatch ('{fb1.units}' vs '{fb2.units}')"
        )

        if isinstance(fb1.value, NumRange) and isinstance(fb2.value, NumRange):
            assert fb1.value.min == fb2.value.min, (
                f"{path}/fluo_bias/value: NumRange.min mismatch ({fb1.value.min} vs {fb2.value.min})"
            )
            assert fb1.value.max == fb2.value.max, (
                f"{path}/fluo_bias/value: NumRange.max mismatch ({fb1.value.max} vs {fb2.value.max})"
            )
        elif isinstance(fb1.value, (int, float)) and isinstance(fb2.value, (int, float)):
            assert abs(fb1.value - fb2.value) < 1e-9, (
                f"{path}/fluo_bias/value: Numeric mismatch ({fb1.value} vs {fb2.value})"
            )
        else:
            type1 = type(fb1.value).__name__
            type2 = type(fb2.value).__name__
            assert False, f"{path}/fluo_bias/value: Type mismatch ({type1} vs {type2})"
    else:
        has1 = c1.fluo_bias is not None
        has2 = c2.fluo_bias is not None
        assert False, (
            f"{path}/fluo_bias: Presence mismatch (original={'present' if has1 else 'absent'}, reconstructed={'present' if has2 else 'absent'})"
        )

    if numeric_indices:
        numeric_r1 = [r1[i] for i in numeric_indices]
        numeric_r2 = [r2[i] for i in numeric_indices]
        sum1 = sum(numeric_r1)
        sum2 = sum(numeric_r2)

        for i in numeric_indices:
            norm1 = r1[i] / sum1 if sum1 > 0 else r1[i]
            norm2 = r2[i] / sum2 if sum2 > 0 else r2[i]
            assert abs(norm1 - norm2) < 1e-9, (
                f"{path}/Ratio[{i}]: Normalized ratio mismatch ({norm1:.6f} vs {norm2:.6f})"
            )

    for i, (u1, u2) in enumerate(zip(c1.units, c2.units)):
        tu_path = f"{path}/Unit[{i}:{u1.name if u1.name else 'unnamed'}]"
        tu_equals_assert(u1, u2, path=tu_path)


def _is_empty_slot(slot: Slot) -> bool:
    """Slots with only None values are equivalent to missing slots (defaults apply)"""
    return slot.part is None or (isinstance(slot.part, list) and all(p is None for p in slot.part))


def tu_equals_assert(u1: TranscriptionUnit, u2: TranscriptionUnit, path: str):
    slots1 = [s for s in u1.slots if not _is_empty_slot(s)]
    slots2 = [s for s in u2.slots if not _is_empty_slot(s)]

    assert len(slots1) == len(slots2), (
        f"{path}: Slot count mismatch ({len(slots1)} vs {len(slots2)}) after filtering empty slots"
    )

    for i, (s1, s2) in enumerate(zip(slots1, slots2)):
        slot_path = f"{path}/Slot[{i}]"

        part1 = s1.part[0] if isinstance(s1.part, list) and len(s1.part) == 1 else s1.part
        part2 = s2.part[0] if isinstance(s2.part, list) and len(s2.part) == 1 else s2.part

        if isinstance(part1, list) and isinstance(part2, list):
            assert set(part1) == set(part2), (
                f"{slot_path}: Part list mismatch (original={sorted(part1)}, reconstructed={sorted(part2)})"
            )
        else:
            assert part1 == part2, (
                f"{slot_path}: Part mismatch (original='{part1}', reconstructed='{part2}')"
            )

        ref1 = getattr(s1, "ref_id", None)
        ref2 = getattr(s2, "ref_id", None)
        assert ref1 == ref2, (
            f"{slot_path}: ref_id mismatch (original='{ref1}', reconstructed='{ref2}')"
        )


def _test_roundtrip(lib, recipe: Recipe, invert: bool = False):  # noqa: F811
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(recipe, invert=invert)
        reconstructed = networks[0].to_recipe()
        recipe_equals_assert(recipe, reconstructed, check_name=False, path=f"Recipe[{recipe.name}]")


@pytest.mark.parametrize(
    "fixture_name,invert",
    [
        ("simple_single_reporter", False),
        ("simple_single_reporter", True),
        ("simple_two_reporters", False),
        ("simple_two_reporters", True),
        ("simple_single_ern", False),
        ("simple_single_ern", True),
        ("simple_single_cotx_ERN", False),
        ("simple_single_cotx_ERN", True),
        ("multi_aggregation_ern", False),
        ("multi_aggregation_ern", True),
        ("variable_uorf_network", False),
        ("variable_uorf_network", True),
        ("simple_aggregation", False),
        ("simple_aggregation", True),
        ("multi_cotx_aggregation", False),
        ("multi_cotx_aggregation", True),
        ("complex_ern_network", False),
        ("complex_ern_network", True),
        ("uorf_ern_network", False),
        ("uorf_ern_network", True),
        # Unlocked param recipes now work with updated to_recipe()
        ("unlocked_ratios_network", False),
        ("unlocked_ratios_network", True),
        ("bias_network", False),
        ("bias_network", True),
        ("unlocked_bias_network", False),
        ("unlocked_bias_network", True),
        ("combined_unlocked_network", False),
        ("combined_unlocked_network", True),
        ("ern_with_unlocked_ratios", False),
        ("ern_with_unlocked_ratios", True),
        ("ern_with_unlocked_bias", False),
        ("ern_with_unlocked_bias", True),
        ("ern_with_unlocked_uorfs", False),
        ("ern_with_unlocked_uorfs", True),
        ("complex_mixed_unlocked", False),
        ("complex_mixed_unlocked", True),
        ("complex_twolayers_design_network", False),
        ("complex_twolayers_design_network", True),
    ],
)
def test_fixture_roundtrip(lib, fixture_name, invert, request):  # noqa: F811
    """Parametrized test for all recipe fixtures"""
    recipe = request.getfixturevalue(fixture_name)
    _test_roundtrip(lib, recipe, invert)


def test_ratio_normalization_preserved(lib):  # noqa: F811
    """Test that normalized ratios [0.25, 0.25] -> [0.5, 0.5] are preserved"""
    original = Recipe(
        name="test_ratio_norm",
        content=[
            CoTransfection(
                units=[
                    TranscriptionUnit(
                        slots=[Slot("cHS4"), Slot("hEF1a"), Slot("eBFP2"), Slot("L0.T_4560")]
                    ),
                    TranscriptionUnit(
                        slots=[Slot("cHS4"), Slot("hEF1a"), Slot("mKO2"), Slot("L0.T_4560")]
                    ),
                ],
                ratios=[0.25, 0.25],
            )
        ],
    )
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(original, invert=False)
        reconstructed = networks[0].to_recipe()

        assert len(reconstructed.content) == 1
        assert reconstructed.content[0].ratios == [0.5, 0.5]


def test_single_unit_no_ratio(lib):  # noqa: F811
    """Test that single unit cotransfections have None or [1.0] ratio"""
    original = Recipe(
        name="single_unit",
        content=[
            CoTransfection(
                units=[
                    TranscriptionUnit(
                        slots=[Slot("cHS4"), Slot("hEF1a"), Slot("eBFP2"), Slot("L0.T_4560")]
                    )
                ]
            )
        ],
    )
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(original, invert=False)
        reconstructed = networks[0].to_recipe()

        assert len(reconstructed.content) == 1
        assert reconstructed.content[0].ratios is None or reconstructed.content[0].ratios == [1.0]


def test_metadata_preservation(lib):  # noqa: F811
    """Test that recipe metadata is preserved through roundtrip"""
    original = Recipe(
        name="test_metadata",
        description="Test recipe with metadata",
        metadata={"experiment": "roundtrip_test", "version": 1},
        content=[
            CoTransfection(
                units=[
                    TranscriptionUnit(
                        slots=[Slot("cHS4"), Slot("hEF1a"), Slot("eBFP2"), Slot("L0.T_4560")]
                    )
                ]
            )
        ],
    )
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(original, invert=True)
        networks[0].metadata.update(original.metadata)
        networks[0].metadata["description"] = original.description

        reconstructed = networks[0].to_recipe()

        assert reconstructed.description == original.description
        assert reconstructed.metadata == original.metadata


def test_variable_uorf_options_preserved(lib):  # noqa: F811
    """Test that variable uORF parts are preserved with all options"""
    original = Recipe(
        name="var_uorf",
        content=[
            CoTransfection(
                units=[
                    TranscriptionUnit(
                        slots=[
                            Slot("cHS4"),
                            Slot("hEF1a"),
                            Slot(["1x_uORF", "2x_uORF", "3x_uORF"]),
                            Slot("eBFP2"),
                            Slot("L0.T_4560"),
                        ]
                    )
                ]
            )
        ],
    )
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(original, invert=True)
        reconstructed = networks[0].to_recipe()

        assert recipe_equals(original, reconstructed)

        uorf_slot = None
        for slot in reconstructed.content[0].units[0].slots:
            if isinstance(slot.part, list) and "1x_uORF" in slot.part:
                uorf_slot = slot
                break

        assert uorf_slot is not None
        assert set(uorf_slot.part) == {"1x_uORF", "2x_uORF", "3x_uORF"}


def test_l1_plasmid_roundtrip(lib):  # noqa: F811
    """Test roundtrip with L1 plasmids that expand from library"""
    from biocomp.recipe import dict_to_recipe

    old_format = {
        "name": "l1_test",
        "description": "Test L1 plasmids",
        "content": [
            {
                "sources": [
                    {"ratio": 0.5, "plasmid": "L1.ST2-3_EBFP2"},
                    {"ratio": 0.5, "plasmid": "L1.ST2-3_mKO2"},
                ]
            }
        ],
    }

    with LibraryContext.with_library(lib):
        original = dict_to_recipe(old_format)
        networks = recipe_to_networks(original, invert=False)
        reconstructed = networks[0].to_recipe()

        assert recipe_equals(original, reconstructed)
        assert len(reconstructed.content) == 1
        assert len(reconstructed.content[0].units) == 2
