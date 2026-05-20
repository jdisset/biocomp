# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
# ruff: noqa: F811
"""Regression tests for uORF / parts extraction in network_info.

Originally motivated by the ConstraintsV2_3 ERN_ERNuORFsum_NxCasE networks in
the production DB, whose extracted `uorf_values` reported [[0, 0]] and
`uorf_names` reported "No uORF" regardless of the 5'UTR part actually
attached (0x-inert, 1x-, 3x-, 8x-uORF). The parts DB itself stored the
correct L0-5.2_Nx-uORF parts -- only the extraction was broken.

Each test builds a Recipe by hand, compiles to a Network, and asserts that
`generate_network_info` produces the correct `uorf_values`, `uorf_names`,
and `all_parts`. Fixtures cover CasE and Pgu conventions plus the "weak"
(1w) variant.
"""

import pytest

from biocomp.library import LibraryContext
from biocomp.network import generate_network_info, recipe_to_networks
from biocomp.recipe import Recipe, CoTransfection, Slot, TranscriptionUnit

from test_declarative_recipes import lib  # noqa: F401,F811  (fixture)


def _single_ern_recipe(lib, ern_uorf: str | None, rec_uorf: str | None, name: str) -> Recipe:
    """Build a minimal 2-input single-ERN recipe with explicit uORF slots.

    `ern_uorf` / `rec_uorf` are either None (no uORF part in that TU's 5'UTR)
    or a library part name such as "1x_uORF", "3x_uORF", "1w_uORF".
    """
    with LibraryContext.with_library(lib):
        ern_slots = [Slot(part="cHS4"), Slot(part="hEF1a")]
        if ern_uorf is not None:
            ern_slots.append(Slot(part=ern_uorf))
        ern_slots += [Slot(part="CasE"), Slot(part="L0.T_4560")]

        rec_slots = [Slot(part="cHS4"), Slot(part="hEF1a")]
        if rec_uorf is not None:
            rec_slots.append(Slot(part=rec_uorf))
        rec_slots += [
            Slot(part="CasE_rec"),
            Slot(part="eBFP2"),
            Slot(part="L0.T_4560"),
        ]

        return Recipe(
            name=name,
            content=[
                CoTransfection(
                    units=[
                        TranscriptionUnit(name="ERN_TU", slots=ern_slots),
                        TranscriptionUnit(name="REC_TU", slots=rec_slots),
                        TranscriptionUnit(
                            name="mNeonGreen_reporter",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="mNeonGreen"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                    ],
                )
            ],
        )


def _build(lib, recipe):
    with LibraryContext.with_library(lib):
        nets = recipe_to_networks(recipe, lib=lib)
        assert len(nets) >= 1, f"recipe {recipe.name} produced no networks"
        return nets[0]


@pytest.mark.parametrize(
    "ern_uorf, rec_uorf, expected",
    [
        (None, None, (0, 0)),
        ("1x_uORF", None, (10, 0)),
        (None, "1x_uORF", (0, 10)),
        ("3x_uORF", None, (30, 0)),
        ("8x_uORF", None, (80, 0)),
        ("1w_uORF", None, (5, 0)),
        ("2x_uORF", "3x_uORF", (20, 30)),
    ],
)
def test_uorf_values_extraction(lib, ern_uorf, rec_uorf, expected):
    """`generate_network_info` must distinguish 0x / 1x / 3x / 8x / 1w uORFs.

    The original bug surfaced here: every uORF variant of
    ConstraintsV2_3_ERN_ERNuORFsum_* collapsed to (0, 0).
    """
    name = f"ern_{ern_uorf or 'None'}_rec_{rec_uorf or 'None'}"
    recipe = _single_ern_recipe(lib, ern_uorf, rec_uorf, name)
    network = _build(lib, recipe)
    info = generate_network_info(network, lib)

    assert len(info["uorf_values"]) == 1, info["uorf_values"]
    got = tuple(info["uorf_values"][0])
    assert got == expected, (
        f"recipe {name}: expected uorf_values {expected}, got {got} (names={info['uorf_names']})"
    )


def test_uorf_names_match_values(lib):
    """`uorf_names` must track `uorf_values` (No uORF / 1x uORF / 3x uORF / ...)."""
    recipe = _single_ern_recipe(lib, "3x_uORF", "1x_uORF", "uorf_names_smoke")
    network = _build(lib, recipe)
    info = generate_network_info(network, lib)
    assert tuple(tuple(v) for v in info["uorf_values"]) == ((30, 10),)
    assert tuple(info["uorf_names"]) == ("CasE ERN: 3x uORF", "CasE REC: 1x uORF")


def test_all_parts_includes_uorf(lib):
    """`all_parts` must include the uORF part category, not silently drop it.

    In the broken extraction, `all_parts` for L1.3x_CasE_ST1-2 returned only
    {cHS4, CasE, L0.T_4560}, missing the 3x_uORF 5'UTR and the hEF1a promoter.
    """
    recipe = _single_ern_recipe(lib, "3x_uORF", None, "all_parts_smoke")
    network = _build(lib, recipe)
    info = generate_network_info(network, lib)

    ern_tu_parts = next(parts for tu_name, parts in info["all_parts"].items() if "ERN" in tu_name)
    part_names = set(ern_tu_parts.keys())

    assert "CasE" in part_names, part_names
    assert "hEF1a" in part_names, f"promoter missing from all_parts: {part_names}"
    assert "3x_uORF" in part_names, f"uORF missing from all_parts: {part_names}"


def test_uorf_values_zero_when_no_ern(lib):
    """Non-ERN networks should still produce sensible (empty) uORF info."""
    with LibraryContext.with_library(lib):
        recipe = Recipe(
            name="no_ern",
            content=[
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            name="reporter",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="eBFP2"),
                                Slot(part="L0.T_4560"),
                            ],
                        )
                    ]
                )
            ],
        )
    network = _build(lib, recipe)
    info = generate_network_info(network, lib)
    assert tuple(info["uorf_values"]) == ()
    assert tuple(info["uorf_names"]) == ()


def _multi_source_ern_recipe(lib, uorfs_per_source: list[str | None], name: str) -> Recipe:
    """Multi-source single-ERN recipe -- multiple CasE source plasmids with
    different uORF counts, each in its own cotx group.

    Matches the ConstraintsV2_3 ERN_ERNuORFsum topology where N CasE source
    plasmids feed a single downstream reporter through a shared CasE_rec TU.
    """
    with LibraryContext.with_library(lib):
        reporters = ["eBFP2", "mKO2", "mMaroon1"]
        cotxs = []
        for i, uorf in enumerate(uorfs_per_source):
            ern_slots = [Slot(part="cHS4"), Slot(part="hEF1a")]
            if uorf is not None:
                ern_slots.append(Slot(part=uorf))
            ern_slots += [Slot(part="CasE"), Slot(part="L0.T_4560")]
            cotx = CoTransfection(
                units=[
                    TranscriptionUnit(name=f"ERN_TU_{i}", slots=ern_slots),
                    TranscriptionUnit(
                        name=f"reporter_{i}",
                        slots=[
                            Slot(part="cHS4"),
                            Slot(part="hEF1a"),
                            Slot(part=reporters[i % len(reporters)]),
                            Slot(part="L0.T_4560"),
                        ],
                    ),
                ]
            )
            cotxs.append(cotx)
        cotxs.append(
            CoTransfection(
                units=[
                    TranscriptionUnit(
                        name="REC_TU",
                        slots=[
                            Slot(part="cHS4"),
                            Slot(part="hEF1a"),
                            Slot(part="CasE_rec"),
                            Slot(part="mNeonGreen"),
                            Slot(part="L0.T_4560"),
                        ],
                    ),
                ]
            )
        )
        return Recipe(name=name, content=cotxs)


def test_multi_source_ern_reports_max_uorf(lib):
    """When multiple source TUs merge into a single ERN input (RNA merge by
    content identity), the extracted uORF value should not silently pick one
    contributor. Returning the MAX captures "some input has uORF" correctly.

    Reproduces the ConstraintsV2_3 ERN_ERNuORFsum_3xCasE case: 0x + 3x CasE
    sources both feed the ERN. Before the fix, the extraction reported (0, 0).
    """
    # "0x" = no uORF slot (None); "3x" = 3x_uORF part.
    recipe = _multi_source_ern_recipe(lib, [None, "3x_uORF"], "multi_0x_3x")
    network = _build(lib, recipe)
    info = generate_network_info(network, lib)
    assert len(info["uorf_values"]) == 1
    assert tuple(info["uorf_values"][0]) == (30, 0), (
        f"expected max uORF=30 across contributors; got {info['uorf_values']}"
    )


def test_multi_source_ern_two_uorfed_contributors(lib):
    """Both contributors have uORFs at different levels; MAX wins."""
    recipe = _multi_source_ern_recipe(lib, ["1x_uORF", "3x_uORF"], "multi_1x_3x")
    network = _build(lib, recipe)
    info = generate_network_info(network, lib)
    assert tuple(info["uorf_values"][0]) == (30, 0), info["uorf_values"]


def test_l2_plasmid_two_tus_per_source_preserve_uorf(lib):
    """L2 plasmids expand into multiple TUs that share a single source node
    in the compute graph (source is the plasmid, TUs are its slots). The
    uORF on one TU must not be clobbered by the other.

    Reproduces the 2023-02-16_Matrix Matrix(R+CasE0x)+(L2CasER1x_Y+B) case
    where plasmid L2-CasER-1xuORF expands into (CasER+eYFP with 1x uORF)
    AND (eBFP2 marker with no uORF), both feeding through the same source.
    Before the fix, keying by source_id clobbered the 1x uORF with 0.
    """
    with LibraryContext.with_library(lib):
        recipe = Recipe(
            name="l2_two_tus",
            content=[
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            name="ERN_TU",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="CasE"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                    ]
                ),
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            name="REC_TU",
                            source="L2_shared",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="1x_uORF"),
                                Slot(part="CasE_rec"),
                                Slot(part="eBFP2"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                        TranscriptionUnit(
                            name="marker_TU",
                            source="L2_shared",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="mKate"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                    ]
                ),
            ],
        )
    network = _build(lib, recipe)
    info = generate_network_info(network, lib)
    # REC side should see 1x uORF = 10, even though the companion marker_TU
    # (sharing the source node) has no uORF.
    assert tuple(tuple(v) for v in info["uorf_values"]) == ((0, 10),), info["uorf_values"]


def test_pgu_style_plasmid_naming(lib):
    """Pgu TUs historically use L1.ST1-2_<N>x_pguCas13[_REAL] naming.

    The critical thing is that the extraction doesn't depend on the plasmid
    NAME but on the actual slots/parts -- this test builds a Pgu-style
    recipe and asserts the uORF value still comes through correctly.
    """
    with LibraryContext.with_library(lib):
        recipe = Recipe(
            name="single_Pgu_3x",
            content=[
                CoTransfection(
                    units=[
                        TranscriptionUnit(
                            name="ERN_TU",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="3x_uORF"),
                                Slot(part="PgU"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                        TranscriptionUnit(
                            name="REC_TU",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="PgU_rec"),
                                Slot(part="eBFP2"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                        TranscriptionUnit(
                            name="mNeonGreen_reporter",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="mNeonGreen"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                    ],
                )
            ],
        )
    network = _build(lib, recipe)
    info = generate_network_info(network, lib)
    assert tuple(tuple(v) for v in info["uorf_values"]) == ((30, 0),), info["uorf_values"]
    assert list(info["ern_names"]) == ["PgU"]
