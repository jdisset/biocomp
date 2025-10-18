from biocomp.library import LibraryContext
from biocomp.recipe import Recipe, CoTransfection, TranscriptionUnit, Slot
from biocomp.network import recipe_to_networks
import biocomp.biorules as br
import pytest

from test_declarative_recipes import (
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
)  # noqa: F401


def recipe_equals(r1: Recipe, r2: Recipe, check_name: bool = False) -> bool:
    if check_name and r1.name != r2.name:
        return False
    if len(r1.content) != len(r2.content):
        return False
    for c1, c2 in zip(r1.content, r2.content):
        if not cotx_equals(c1, c2):
            return False
    return True


def cotx_equals(c1: CoTransfection, c2: CoTransfection) -> bool:
    from biocomp.recipe import NumRange, FluoIntensity

    if len(c1.units) != len(c2.units):
        return False

    r1 = c1.ratios or [1.0] * len(c1.units)
    r2 = c2.ratios or [1.0] * len(c2.units)

    if len(r1) != len(r2):
        return False

    # Handle NumRange in ratios - they should remain unchanged through roundtrip
    # Build lists of which ratios are NumRange vs numeric for normalization
    numeric_indices = []
    for i, (ratio1, ratio2) in enumerate(zip(r1, r2)):
        if isinstance(ratio1, NumRange) and isinstance(ratio2, NumRange):
            # Both are NumRange - check they match
            if ratio1.min != ratio2.min or ratio1.max != ratio2.max:
                return False
        elif isinstance(ratio1, NumRange) or isinstance(ratio2, NumRange):
            # One is NumRange, other isn't - this is a mismatch
            return False
        else:
            # Both are numeric - will check after normalization
            numeric_indices.append(i)

    # Check fluo_bias equality
    if c1.fluo_bias is None and c2.fluo_bias is None:
        pass  # Both None, OK
    elif c1.fluo_bias is not None and c2.fluo_bias is not None:
        fb1, fb2 = c1.fluo_bias, c2.fluo_bias
        if fb1.tu_id != fb2.tu_id or fb1.protein != fb2.protein or fb1.units != fb2.units:
            return False
        # Check value equality (handling NumRange)
        if isinstance(fb1.value, NumRange) and isinstance(fb2.value, NumRange):
            if fb1.value.min != fb2.value.min or fb1.value.max != fb2.value.max:
                return False
        elif isinstance(fb1.value, (int, float)) and isinstance(fb2.value, (int, float)):
            if abs(fb1.value - fb2.value) >= 1e-9:
                return False
        else:
            return False
    else:
        return False  # One has fluo_bias, other doesn't

    # Normalize and compare numeric ratios only
    if numeric_indices:
        # Get only the numeric ratios for normalization
        numeric_r1 = [r1[i] for i in numeric_indices]
        numeric_r2 = [r2[i] for i in numeric_indices]
        sum1 = sum(numeric_r1)
        sum2 = sum(numeric_r2)

        # Compare normalized numeric ratios
        for i in numeric_indices:
            norm1 = r1[i] / sum1 if sum1 > 0 else r1[i]
            norm2 = r2[i] / sum2 if sum2 > 0 else r2[i]
            if abs(norm1 - norm2) >= 1e-9:
                return False

    for u1, u2 in zip(c1.units, c2.units):
        if not tu_equals(u1, u2):
            return False
    return True


def tu_equals(u1: TranscriptionUnit, u2: TranscriptionUnit) -> bool:
    if len(u1.slots) != len(u2.slots):
        return False
    for s1, s2 in zip(u1.slots, u2.slots):
        if s1.part != s2.part:
            return False
    return True


def _test_roundtrip(lib, recipe: Recipe, invert: bool = False):
    """Helper function to test recipe roundtrip conversion"""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(recipe, invert=invert)
        reconstructed = networks[0].to_recipe()
        assert recipe_equals(recipe, reconstructed)


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
    ],
)
def test_fixture_roundtrip(lib, fixture_name, invert, request):
    """Parametrized test for all recipe fixtures"""
    recipe = request.getfixturevalue(fixture_name)
    _test_roundtrip(lib, recipe, invert)


def test_ratio_normalization_preserved(lib):
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


def test_single_unit_no_ratio(lib):
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


def test_metadata_preservation(lib):
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


def test_variable_uorf_options_preserved(lib):
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


def test_l1_plasmid_roundtrip(lib):
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
