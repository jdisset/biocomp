"""Tests for TU masking commit behavior using the two_and_one_skip scaffold.

Verifies:
1. Masked TUs are correctly removed from committed recipes
2. ERN handling follows biological semantics (negative/positive side rules)
3. Recognition sites are properly stripped when ERN protein is masked

ERN Masking Rules:
- Negative masked (e.g., *_a-): Remove negative TU, keep positive TU BUT strip its ERN_rec
- Positive masked (e.g., *_a+): Remove BOTH positive AND negative TUs (cascade)
"""

import pytest
import jax
import jax.numpy as jnp
import dracon as dr
from pathlib import Path

import biocomp.biorules as br
from biocomp.compute import ComputeStack
from biocomp.config import SIMPLE_NODES_COMPUTE_CONFIG
from biocomp.library import LibraryContext, load_lib
from biocomp.network import recipe_to_networks
from biocomp.recipe import Recipe
from biocomp.tumasking import build_tu_id_mapping, TU_LOG_ALPHA_PATH


SCAFFOLD_PATH = Path(__file__).parent.parent.parent / "biocomp-jobs/design/architectures/two_and_one_skip.yaml"


@pytest.fixture(scope="module")
def lib():
    return load_lib()


@pytest.fixture(scope="module")
def scaffold_recipe(lib):
    with LibraryContext.with_library(lib):
        data = dr.load(SCAFFOLD_PATH, context={"Recipe": Recipe})
        recipes = data["recipes"] if "recipes" in data else data.recipes
        return recipes[0]


@pytest.fixture
def scaffold_stack(lib, scaffold_recipe):
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(scaffold_recipe, br.ALL_RULES, invert=True)
        tu_ids, tu_id_to_idx = build_tu_id_mapping(networks)

        stack = ComputeStack(networks)
        config = SIMPLE_NODES_COMPUTE_CONFIG.model_copy(deep=True)
        stack.build(config, enable_tu_masking=True)

        return stack, tu_ids, tu_id_to_idx


def get_tu_names_from_recipe(recipe: Recipe) -> set[str]:
    return {tu.name for cotx in recipe.content for tu in cotx.units}


def get_slot_parts(tu) -> list[str]:
    parts = []
    for slot in tu.slots:
        if hasattr(slot, "part"):
            parts.append(slot.part)
        elif hasattr(slot, "name"):
            parts.append(slot.name)
        elif isinstance(slot, str):
            parts.append(slot)
        else:
            parts.append(str(slot))
    return parts


def find_tu_in_recipe(recipe: Recipe, tu_name: str):
    for cotx in recipe.content:
        for tu in cotx.units:
            if tu.name == tu_name:
                return tu
    return None


def create_tu_mask(
    tu_ids: list[str],
    tu_id_to_idx: dict[str, int],
    n_networks: int,
    disabled_patterns: list[str],
) -> jnp.ndarray:
    n_tus = len(tu_ids)
    log_alpha = jnp.full((n_networks, n_tus), 10.0)

    for pattern in disabled_patterns:
        for tu_id in tu_ids:
            tu_name = "_".join(tu_id.split("_")[:-1])
            if tu_name == pattern:
                idx = tu_id_to_idx[tu_id]
                log_alpha = log_alpha.at[:, idx].set(-10.0)

    return log_alpha


def verify_tu_removed(recipe: Recipe, tu_name: str):
    tu_names = get_tu_names_from_recipe(recipe)
    assert tu_name not in tu_names, f"TU '{tu_name}' should be removed but found in recipe"


def verify_tu_present(recipe: Recipe, tu_name: str):
    tu_names = get_tu_names_from_recipe(recipe)
    assert tu_name in tu_names, f"TU '{tu_name}' should be present but not found in recipe"


def verify_ern_rec_stripped(recipe: Recipe, tu_name: str, ern_rec: str):
    tu = find_tu_in_recipe(recipe, tu_name)
    assert tu is not None, f"TU '{tu_name}' not found in recipe for ERN_rec verification"
    slot_parts = get_slot_parts(tu)
    assert ern_rec not in slot_parts, (
        f"TU '{tu_name}' should have '{ern_rec}' stripped but found in slots: {slot_parts}"
    )


def verify_ern_rec_present(recipe: Recipe, tu_name: str, ern_rec: str):
    tu = find_tu_in_recipe(recipe, tu_name)
    assert tu is not None, f"TU '{tu_name}' not found in recipe for ERN_rec verification"
    slot_parts = get_slot_parts(tu)
    assert ern_rec in slot_parts, (
        f"TU '{tu_name}' should have '{ern_rec}' present but not found in slots: {slot_parts}"
    )


def verify_exact_slots(recipe: Recipe, tu_name: str, expected_slots: list[str]):
    tu = find_tu_in_recipe(recipe, tu_name)
    assert tu is not None, f"TU '{tu_name}' not found in recipe for slot verification"
    slot_parts = get_slot_parts(tu)
    assert slot_parts == expected_slots, (
        f"TU '{tu_name}' slot mismatch.\nExpected: {expected_slots}\nActual: {slot_parts}"
    )


class TestDirectOutMasking:
    """Test direct_out TU masking (skip connection bypass).

    NOTE: Marker TUs (x1_marker, x2_marker, b_marker) cannot be disabled in design mode
    because they provide the input proteins for the inverted network structure.
    Disabling markers invalidates the network.
    """

    def test_single_direct_out(self, lib, scaffold_stack):
        with LibraryContext.with_library(lib):
            stack, tu_ids, tu_id_to_idx = scaffold_stack
            params = stack.init(jax.random.key(42))

            log_alpha = create_tu_mask(tu_ids, tu_id_to_idx, len(stack.networks), ["b_direct_out"])
            params.at(TU_LOG_ALPHA_PATH, log_alpha, overwrite=True)

            committed = stack.commit(params)
            recipe = committed[0].to_recipe()

            verify_tu_removed(recipe, "b_direct_out")
            verify_tu_present(recipe, "x1_direct_out")
            verify_tu_present(recipe, "x2_direct_out")
            verify_tu_present(recipe, "b_a+")

    def test_all_direct_outs(self, lib, scaffold_stack):
        with LibraryContext.with_library(lib):
            stack, tu_ids, tu_id_to_idx = scaffold_stack
            params = stack.init(jax.random.key(42))

            log_alpha = create_tu_mask(
                tu_ids, tu_id_to_idx, len(stack.networks),
                ["x1_direct_out", "x2_direct_out", "b_direct_out"]
            )
            params.at(TU_LOG_ALPHA_PATH, log_alpha, overwrite=True)

            committed = stack.commit(params)
            recipe = committed[0].to_recipe()

            verify_tu_removed(recipe, "x1_direct_out")
            verify_tu_removed(recipe, "x2_direct_out")
            verify_tu_removed(recipe, "b_direct_out")
            verify_tu_present(recipe, "x1_marker")
            verify_tu_present(recipe, "x1_a+")


class TestNegativeSideMasking:
    """Test negative side masking (ERN protein disabled).

    NOTE: ERN nodes are SHARED across all co-transfections. To disable an ERN layer
    and trigger ERN_rec stripping, ALL negative TUs for that layer must be disabled
    across ALL co-transfections.
    """

    def test_neg_a_single_cotx(self, lib, scaffold_stack):
        """Disabling one negative TU only removes that TU, ERN stays active."""
        with LibraryContext.with_library(lib):
            stack, tu_ids, tu_id_to_idx = scaffold_stack
            params = stack.init(jax.random.key(42))

            log_alpha = create_tu_mask(tu_ids, tu_id_to_idx, len(stack.networks), ["x1_a-"])
            params.at(TU_LOG_ALPHA_PATH, log_alpha, overwrite=True)

            committed = stack.commit(params)
            recipe = committed[0].to_recipe()

            verify_tu_removed(recipe, "x1_a-")
            verify_tu_present(recipe, "x1_a+")
            verify_ern_rec_present(recipe, "x1_a+", "CasE_rec")

            verify_tu_present(recipe, "x2_a-")
            verify_tu_present(recipe, "x2_a+")
            verify_ern_rec_present(recipe, "x2_a+", "CasE_rec")

    def test_neg_a_all_cotx(self, lib, scaffold_stack):
        """Disabling ALL negative TUs for layer a strips ERN_rec from ALL positive TUs."""
        with LibraryContext.with_library(lib):
            stack, tu_ids, tu_id_to_idx = scaffold_stack
            params = stack.init(jax.random.key(42))

            log_alpha = create_tu_mask(
                tu_ids, tu_id_to_idx, len(stack.networks),
                ["x1_a-", "x2_a-", "b_a-"]
            )
            params.at(TU_LOG_ALPHA_PATH, log_alpha, overwrite=True)

            committed = stack.commit(params)
            recipe = committed[0].to_recipe()

            for cotx_prefix in ["x1", "x2", "b"]:
                verify_tu_removed(recipe, f"{cotx_prefix}_a-")
                verify_tu_present(recipe, f"{cotx_prefix}_a+")
                verify_ern_rec_stripped(recipe, f"{cotx_prefix}_a+", "CasE_rec")

            verify_tu_present(recipe, "x1_b+")
            verify_ern_rec_present(recipe, "x1_b+", "Csy4_rec")

    def test_neg_b_all_cotx(self, lib, scaffold_stack):
        """Disabling ALL layer b negatives does NOT strip Csy4_rec.

        NOTE: Csy4 ERN's negative input comes from BOTH *_b- AND *_a+ TUs
        (because *_a+ produces Csy4 as output). So disabling just *_b- still
        leaves Csy4 protein being produced by *_a+ TUs.
        """
        with LibraryContext.with_library(lib):
            stack, tu_ids, tu_id_to_idx = scaffold_stack
            params = stack.init(jax.random.key(42))

            log_alpha = create_tu_mask(
                tu_ids, tu_id_to_idx, len(stack.networks),
                ["x1_b-", "x2_b-", "b_b-"]
            )
            params.at(TU_LOG_ALPHA_PATH, log_alpha, overwrite=True)

            committed = stack.commit(params)
            recipe = committed[0].to_recipe()

            for cotx_prefix in ["x1", "x2", "b"]:
                verify_tu_removed(recipe, f"{cotx_prefix}_b-")
                verify_tu_present(recipe, f"{cotx_prefix}_b+")
                # Csy4_rec is NOT stripped because *_a+ still produces Csy4
                verify_ern_rec_present(recipe, f"{cotx_prefix}_b+", "Csy4_rec")

            verify_tu_present(recipe, "x1_a-")
            verify_tu_present(recipe, "x1_a+")
            verify_ern_rec_present(recipe, "x1_a+", "CasE_rec")

    def test_neg_all_layers(self, lib, scaffold_stack):
        """Disabling ALL negative TUs for ALL layers.

        NOTE: Csy4_rec is NOT stripped because *_a+ TUs still produce Csy4.
        Only CasE_rec and PgU_rec are stripped (their ERN proteins come from
        *_a- and *_c- respectively, which have no other source).
        """
        with LibraryContext.with_library(lib):
            stack, tu_ids, tu_id_to_idx = scaffold_stack
            params = stack.init(jax.random.key(42))

            disabled = []
            for cotx in ["x1", "x2", "b"]:
                for layer in ["a", "b", "c"]:
                    disabled.append(f"{cotx}_{layer}-")

            log_alpha = create_tu_mask(tu_ids, tu_id_to_idx, len(stack.networks), disabled)
            params.at(TU_LOG_ALPHA_PATH, log_alpha, overwrite=True)

            committed = stack.commit(params)
            recipe = committed[0].to_recipe()

            for cotx in ["x1", "x2", "b"]:
                for layer in ["a", "b", "c"]:
                    verify_tu_removed(recipe, f"{cotx}_{layer}-")
                    verify_tu_present(recipe, f"{cotx}_{layer}+")

            # CasE_rec stripped (only source is *_a- which is disabled)
            verify_ern_rec_stripped(recipe, "x1_a+", "CasE_rec")
            # Csy4_rec NOT stripped (*_a+ still produces Csy4)
            verify_ern_rec_present(recipe, "x1_b+", "Csy4_rec")
            # PgU_rec stripped (only source is *_c- which is disabled)
            verify_ern_rec_stripped(recipe, "x1_c+", "PgU_rec")

            verify_tu_present(recipe, "x1_marker")
            verify_tu_present(recipe, "x1_direct_out")


class TestPositiveSideMasking:
    """Test positive side masking (mRNA with ERN recognition site disabled).

    When a positive TU is disabled, the ERN node detects that its positive input
    has no enabled TUs. This cascades to also disable the corresponding negative TU
    because the ERN node becomes useless without the mRNA to cleave.

    NOTE: The cascade only happens per-cotx. Disabling x1_a+ cascades to x1_a-,
    but does NOT cascade to x2_a- or b_a- (those ERN proteins still serve the
    other co-transfections' positive TUs).
    """

    def test_pos_a_single_cotx(self, lib, scaffold_stack):
        """Disabling one positive TU cascades to disable its matching negative TU."""
        with LibraryContext.with_library(lib):
            stack, tu_ids, tu_id_to_idx = scaffold_stack
            params = stack.init(jax.random.key(42))

            log_alpha = create_tu_mask(tu_ids, tu_id_to_idx, len(stack.networks), ["x1_a+"])
            params.at(TU_LOG_ALPHA_PATH, log_alpha, overwrite=True)

            committed = stack.commit(params)
            recipe = committed[0].to_recipe()

            verify_tu_removed(recipe, "x1_a+")

            verify_tu_present(recipe, "x2_a+")
            verify_tu_present(recipe, "x2_a-")
            verify_tu_present(recipe, "b_a+")
            verify_tu_present(recipe, "b_a-")

    def test_pos_a_all_cotx(self, lib, scaffold_stack):
        """Disabling ALL positive TUs for layer a cascades to ALL negative TUs."""
        with LibraryContext.with_library(lib):
            stack, tu_ids, tu_id_to_idx = scaffold_stack
            params = stack.init(jax.random.key(42))

            log_alpha = create_tu_mask(
                tu_ids, tu_id_to_idx, len(stack.networks),
                ["x1_a+", "x2_a+", "b_a+"]
            )
            params.at(TU_LOG_ALPHA_PATH, log_alpha, overwrite=True)

            committed = stack.commit(params)
            recipe = committed[0].to_recipe()

            for cotx in ["x1", "x2", "b"]:
                verify_tu_removed(recipe, f"{cotx}_a+")
                verify_tu_removed(recipe, f"{cotx}_a-")

            verify_tu_present(recipe, "x1_b+")
            verify_tu_present(recipe, "x1_b-")
            verify_tu_present(recipe, "x1_c+")
            verify_tu_present(recipe, "x1_c-")

    def test_pos_b_single_cotx(self, lib, scaffold_stack):
        """Disabling one positive TU in layer b."""
        with LibraryContext.with_library(lib):
            stack, tu_ids, tu_id_to_idx = scaffold_stack
            params = stack.init(jax.random.key(42))

            log_alpha = create_tu_mask(tu_ids, tu_id_to_idx, len(stack.networks), ["b_b+"])
            params.at(TU_LOG_ALPHA_PATH, log_alpha, overwrite=True)

            committed = stack.commit(params)
            recipe = committed[0].to_recipe()

            verify_tu_removed(recipe, "b_b+")

            verify_tu_present(recipe, "x1_b+")
            verify_tu_present(recipe, "x1_b-")
            verify_tu_present(recipe, "x2_b+")
            verify_tu_present(recipe, "x2_b-")

    def test_pos_all_layers_all_cotx(self, lib, scaffold_stack):
        """Disabling ALL positive TUs for ALL layers cascades to ALL negative TUs."""
        with LibraryContext.with_library(lib):
            stack, tu_ids, tu_id_to_idx = scaffold_stack
            params = stack.init(jax.random.key(42))

            disabled = []
            for cotx in ["x1", "x2", "b"]:
                for layer in ["a", "b", "c"]:
                    disabled.append(f"{cotx}_{layer}+")

            log_alpha = create_tu_mask(tu_ids, tu_id_to_idx, len(stack.networks), disabled)
            params.at(TU_LOG_ALPHA_PATH, log_alpha, overwrite=True)

            committed = stack.commit(params)
            recipe = committed[0].to_recipe()

            for cotx in ["x1", "x2", "b"]:
                for layer in ["a", "b", "c"]:
                    verify_tu_removed(recipe, f"{cotx}_{layer}+")
                    verify_tu_removed(recipe, f"{cotx}_{layer}-")

            verify_tu_present(recipe, "x1_marker")
            verify_tu_present(recipe, "x1_direct_out")
            verify_tu_present(recipe, "x2_marker")
            verify_tu_present(recipe, "b_marker")


class TestMixedScenarios:

    def test_mixed_direct_out_and_neg_single(self, lib, scaffold_stack):
        """Mix: disable direct_out + single negative TU (no ERN_rec strip)."""
        with LibraryContext.with_library(lib):
            stack, tu_ids, tu_id_to_idx = scaffold_stack
            params = stack.init(jax.random.key(42))

            log_alpha = create_tu_mask(
                tu_ids, tu_id_to_idx, len(stack.networks),
                ["x1_direct_out", "x1_a-"]
            )
            params.at(TU_LOG_ALPHA_PATH, log_alpha, overwrite=True)

            committed = stack.commit(params)
            recipe = committed[0].to_recipe()

            verify_tu_removed(recipe, "x1_direct_out")
            verify_tu_removed(recipe, "x1_a-")
            verify_tu_present(recipe, "x1_a+")
            verify_ern_rec_present(recipe, "x1_a+", "CasE_rec")

    def test_mixed_pos_and_neg_different_layers(self, lib, scaffold_stack):
        """Mix: positive cascade + single negative (no ERN_rec strip for partial neg)."""
        with LibraryContext.with_library(lib):
            stack, tu_ids, tu_id_to_idx = scaffold_stack
            params = stack.init(jax.random.key(42))

            log_alpha = create_tu_mask(
                tu_ids, tu_id_to_idx, len(stack.networks),
                ["x1_a+", "b_b-"]
            )
            params.at(TU_LOG_ALPHA_PATH, log_alpha, overwrite=True)

            committed = stack.commit(params)
            recipe = committed[0].to_recipe()

            verify_tu_removed(recipe, "x1_a+")

            verify_tu_removed(recipe, "b_b-")
            verify_tu_present(recipe, "b_b+")
            verify_ern_rec_present(recipe, "b_b+", "Csy4_rec")

    def test_mixed_direct_out_and_pos_all_cotx(self, lib, scaffold_stack):
        """Mix: direct_out + all positives for layer c (cascade removes all c-)."""
        with LibraryContext.with_library(lib):
            stack, tu_ids, tu_id_to_idx = scaffold_stack
            params = stack.init(jax.random.key(42))

            disabled = ["x1_direct_out"]
            for cotx in ["x1", "x2", "b"]:
                disabled.append(f"{cotx}_c+")

            log_alpha = create_tu_mask(tu_ids, tu_id_to_idx, len(stack.networks), disabled)
            params.at(TU_LOG_ALPHA_PATH, log_alpha, overwrite=True)

            committed = stack.commit(params)
            recipe = committed[0].to_recipe()

            verify_tu_removed(recipe, "x1_direct_out")

            for cotx in ["x1", "x2", "b"]:
                verify_tu_removed(recipe, f"{cotx}_c+")
                verify_tu_removed(recipe, f"{cotx}_c-")

            verify_tu_present(recipe, "x2_direct_out")
            verify_tu_present(recipe, "b_direct_out")


class TestCrossLayerScenarios:
    """Test cross-layer ERN masking combinations."""

    def test_cross_neg_all_pos_all(self, lib, scaffold_stack):
        """All negatives for layer a + all positives for layer b.

        NOTE: When all *_b+ are disabled, the Csy4 ERN's positive input is gone.
        This cascades to disable ALL Csy4 sources, which includes BOTH *_b- AND *_a+
        (because *_a+ produces Csy4 as output).

        Combined with disabling all *_a-:
        - *_a+ is removed by the Csy4 cascade (NOT by the *_a- disable)
        - All layer a and b TUs are removed
        """
        with LibraryContext.with_library(lib):
            stack, tu_ids, tu_id_to_idx = scaffold_stack
            params = stack.init(jax.random.key(42))

            disabled = ["x1_a-", "x2_a-", "b_a-", "x1_b+", "x2_b+", "b_b+"]
            log_alpha = create_tu_mask(tu_ids, tu_id_to_idx, len(stack.networks), disabled)
            params.at(TU_LOG_ALPHA_PATH, log_alpha, overwrite=True)

            committed = stack.commit(params)
            recipe = committed[0].to_recipe()

            # All layer a TUs removed: *_a- directly disabled, *_a+ cascade from Csy4
            for cotx in ["x1", "x2", "b"]:
                verify_tu_removed(recipe, f"{cotx}_a-")
                verify_tu_removed(recipe, f"{cotx}_a+")

            # All layer b TUs removed: *_b+ directly disabled, *_b- cascade
            for cotx in ["x1", "x2", "b"]:
                verify_tu_removed(recipe, f"{cotx}_b+")
                verify_tu_removed(recipe, f"{cotx}_b-")

            # Layer c remains intact
            verify_tu_present(recipe, "x1_c+")
            verify_tu_present(recipe, "x1_c-")
            verify_ern_rec_present(recipe, "x1_c+", "PgU_rec")

    def test_partial_pattern(self, lib, scaffold_stack):
        """Mixed partial: one neg (no strip), one pos (cascade), one neg (no strip)."""
        with LibraryContext.with_library(lib):
            stack, tu_ids, tu_id_to_idx = scaffold_stack
            params = stack.init(jax.random.key(42))

            log_alpha = create_tu_mask(
                tu_ids, tu_id_to_idx, len(stack.networks),
                ["x1_a-", "x2_b+", "b_c-"]
            )
            params.at(TU_LOG_ALPHA_PATH, log_alpha, overwrite=True)

            committed = stack.commit(params)
            recipe = committed[0].to_recipe()

            verify_tu_removed(recipe, "x1_a-")
            verify_tu_present(recipe, "x1_a+")
            verify_ern_rec_present(recipe, "x1_a+", "CasE_rec")

            verify_tu_removed(recipe, "x2_b+")

            verify_tu_removed(recipe, "b_c-")
            verify_tu_present(recipe, "b_c+")
            verify_ern_rec_present(recipe, "b_c+", "PgU_rec")


class TestCompleteDisableScenarios:

    def test_all_ern_pairs(self, lib, scaffold_stack):
        """Disable all ERN pairs, leaving only markers and direct_outs."""
        with LibraryContext.with_library(lib):
            stack, tu_ids, tu_id_to_idx = scaffold_stack
            params = stack.init(jax.random.key(42))

            disabled = []
            for cotx in ["x1", "x2", "b"]:
                for layer in ["a", "b", "c"]:
                    disabled.append(f"{cotx}_{layer}+")
                    disabled.append(f"{cotx}_{layer}-")

            log_alpha = create_tu_mask(tu_ids, tu_id_to_idx, len(stack.networks), disabled)
            params.at(TU_LOG_ALPHA_PATH, log_alpha, overwrite=True)

            committed = stack.commit(params)
            recipe = committed[0].to_recipe()

            tu_names = get_tu_names_from_recipe(recipe)
            for cotx in ["x1", "x2", "b"]:
                for layer in ["a", "b", "c"]:
                    assert f"{cotx}_{layer}+" not in tu_names
                    assert f"{cotx}_{layer}-" not in tu_names

            verify_tu_present(recipe, "x1_marker")
            verify_tu_present(recipe, "x2_marker")
            verify_tu_present(recipe, "b_marker")
            verify_tu_present(recipe, "x1_direct_out")
            verify_tu_present(recipe, "x2_direct_out")
            verify_tu_present(recipe, "b_direct_out")

    def test_all_ern_pairs_and_direct_outs(self, lib, scaffold_stack):
        """Disable all ERN pairs AND direct_outs → empty recipe.

        When all output-producing TUs are disabled (ERN pairs + direct outputs),
        only markers remain. But in design mode (inverted network), markers provide
        input proteins - there are no output nodes left. The network becomes invalid
        and returns an empty recipe.
        """
        with LibraryContext.with_library(lib):
            stack, tu_ids, tu_id_to_idx = scaffold_stack
            params = stack.init(jax.random.key(42))

            disabled = []
            for cotx in ["x1", "x2", "b"]:
                for layer in ["a", "b", "c"]:
                    disabled.append(f"{cotx}_{layer}+")
                    disabled.append(f"{cotx}_{layer}-")
                disabled.append(f"{cotx}_direct_out")

            log_alpha = create_tu_mask(tu_ids, tu_id_to_idx, len(stack.networks), disabled)
            params.at(TU_LOG_ALPHA_PATH, log_alpha, overwrite=True)

            committed = stack.commit(params)
            recipe = committed[0].to_recipe()

            tu_names = get_tu_names_from_recipe(recipe)

            # With all outputs disabled, network is invalid → empty recipe
            assert tu_names == set(), (
                f"Expected empty recipe when all output TUs disabled, got: {tu_names}"
            )


class TestEdgeCases:

    def test_enable_all(self, lib, scaffold_stack, scaffold_recipe):
        with LibraryContext.with_library(lib):
            stack, tu_ids, tu_id_to_idx = scaffold_stack
            params = stack.init(jax.random.key(42))

            log_alpha = jnp.full((len(stack.networks), len(tu_ids)), 10.0)
            params.at(TU_LOG_ALPHA_PATH, log_alpha, overwrite=True)

            committed = stack.commit(params)
            recipe = committed[0].to_recipe()

            original_tus = get_tu_names_from_recipe(scaffold_recipe)
            committed_tus = get_tu_names_from_recipe(recipe)

            assert original_tus == committed_tus, (
                f"With all TUs enabled, committed recipe should match original.\n"
                f"Missing: {original_tus - committed_tus}\n"
                f"Extra: {committed_tus - original_tus}"
            )

    def test_committed_recipe_rebuilds(self, lib, scaffold_stack):
        with LibraryContext.with_library(lib):
            stack, tu_ids, tu_id_to_idx = scaffold_stack
            params = stack.init(jax.random.key(42))

            disabled = ["x1_a-", "x2_b+"]
            log_alpha = create_tu_mask(tu_ids, tu_id_to_idx, len(stack.networks), disabled)
            params.at(TU_LOG_ALPHA_PATH, log_alpha, overwrite=True)

            committed = stack.commit(params)
            committed_recipe = committed[0].to_recipe()

            rebuilt_networks = recipe_to_networks(committed_recipe, br.ALL_RULES, invert=True)
            assert len(rebuilt_networks) == 1

            rebuilt_net = rebuilt_networks[0]
            committed_net = committed[0]

            assert len(rebuilt_net.compute_graph.nodes) == len(committed_net.compute_graph.nodes)
            assert len(rebuilt_net.compute_graph.edges) == len(committed_net.compute_graph.edges)


class TestSlotVerification:
    """Verify exact slot contents after ERN_rec stripping.

    NOTE: Must disable ALL negative TUs for an ERN layer to trigger stripping.
    """

    def test_neg_a_all_slots_exact(self, lib, scaffold_stack):
        """Verify slots after all layer a negatives disabled."""
        with LibraryContext.with_library(lib):
            stack, tu_ids, tu_id_to_idx = scaffold_stack
            params = stack.init(jax.random.key(42))

            log_alpha = create_tu_mask(
                tu_ids, tu_id_to_idx, len(stack.networks),
                ["x1_a-", "x2_a-", "b_a-"]
            )
            params.at(TU_LOG_ALPHA_PATH, log_alpha, overwrite=True)

            committed = stack.commit(params)
            recipe = committed[0].to_recipe()

            tu = find_tu_in_recipe(recipe, "x1_a+")
            assert tu is not None
            slot_parts = get_slot_parts(tu)

            assert "hEF1a" in slot_parts, f"Should have promoter hEF1a: {slot_parts}"
            assert "Csy4" in slot_parts, f"Should have output protein Csy4: {slot_parts}"
            assert "L0.T_4560" in slot_parts, f"Should have terminator L0.T_4560: {slot_parts}"
            assert "CasE_rec" not in slot_parts, f"Should NOT have CasE_rec: {slot_parts}"

    def test_neg_b_all_slots_exact(self, lib, scaffold_stack):
        """Verify slots after all layer b negatives disabled.

        NOTE: Csy4_rec is NOT stripped because *_a+ still produces Csy4.
        """
        with LibraryContext.with_library(lib):
            stack, tu_ids, tu_id_to_idx = scaffold_stack
            params = stack.init(jax.random.key(42))

            log_alpha = create_tu_mask(
                tu_ids, tu_id_to_idx, len(stack.networks),
                ["x1_b-", "x2_b-", "b_b-"]
            )
            params.at(TU_LOG_ALPHA_PATH, log_alpha, overwrite=True)

            committed = stack.commit(params)
            recipe = committed[0].to_recipe()

            tu = find_tu_in_recipe(recipe, "x1_b+")
            assert tu is not None
            slot_parts = get_slot_parts(tu)

            assert "hEF1a" in slot_parts
            assert "mNeonGreen" in slot_parts
            assert "L0.T_4560" in slot_parts
            # Csy4_rec is NOT stripped because *_a+ still produces Csy4
            assert "Csy4_rec" in slot_parts, f"Should have Csy4_rec: {slot_parts}"

    def test_csy4_rec_strip_requires_both_sources(self, lib, scaffold_stack):
        """To strip Csy4_rec, must disable BOTH *_b- AND *_a+ (all Csy4 sources).

        In two_and_one_skip, *_a+ TUs produce Csy4 as output protein.
        The Csy4 ERN's negative input comes from BOTH *_b- (direct) AND *_a+ (indirect).
        To fully disable the Csy4 ERN's negative input and strip Csy4_rec,
        ALL sources of Csy4 protein must be disabled.
        """
        with LibraryContext.with_library(lib):
            stack, tu_ids, tu_id_to_idx = scaffold_stack
            params = stack.init(jax.random.key(42))

            # Disable ALL *_b- AND ALL *_a+ (all Csy4 sources)
            disabled = []
            for cotx in ["x1", "x2", "b"]:
                disabled.append(f"{cotx}_b-")
                disabled.append(f"{cotx}_a+")

            log_alpha = create_tu_mask(
                tu_ids, tu_id_to_idx, len(stack.networks), disabled
            )
            params.at(TU_LOG_ALPHA_PATH, log_alpha, overwrite=True)

            committed = stack.commit(params)
            recipe = committed[0].to_recipe()

            # *_b- and *_a+ removed
            for cotx in ["x1", "x2", "b"]:
                verify_tu_removed(recipe, f"{cotx}_b-")
                verify_tu_removed(recipe, f"{cotx}_a+")

            # *_b+ still present BUT now Csy4_rec is stripped!
            for cotx in ["x1", "x2", "b"]:
                verify_tu_present(recipe, f"{cotx}_b+")
                verify_ern_rec_stripped(recipe, f"{cotx}_b+", "Csy4_rec")

            # *_a- still present (CasE producers not affected)
            # But CasE ERN's positive input (*_a+) is gone, so *_a- is cascade-removed
            for cotx in ["x1", "x2", "b"]:
                verify_tu_removed(recipe, f"{cotx}_a-")

            # Layer c remains intact
            verify_tu_present(recipe, "x1_c+")
            verify_tu_present(recipe, "x1_c-")
            verify_ern_rec_present(recipe, "x1_c+", "PgU_rec")


class TestMultiNetworkMasking:

    @pytest.fixture
    def multi_network_stack(self, lib, scaffold_recipe):
        with LibraryContext.with_library(lib):
            networks = []
            for i in range(2):
                nets = recipe_to_networks(scaffold_recipe, br.ALL_RULES, invert=True)
                for net in nets:
                    net.name = f"{net.name}_replica_{i}"
                networks.extend(nets)

            tu_ids, tu_id_to_idx = build_tu_id_mapping(networks)

            stack = ComputeStack(networks)
            config = SIMPLE_NODES_COMPUTE_CONFIG.model_copy(deep=True)
            stack.build(config, enable_tu_masking=True)

            return stack, tu_ids, tu_id_to_idx

    def test_multi_net_different_masks(self, lib, multi_network_stack):
        """Test per-network masking with different patterns.

        Network 0: all *_a- disabled → CasE_rec stripped from *_a+
        Network 1: all *_b+ disabled → cascade removes *_b- AND *_a+ (Csy4 producers)
        """
        with LibraryContext.with_library(lib):
            stack, tu_ids, tu_id_to_idx = multi_network_stack
            n_networks = len(stack.networks)
            n_tus = len(tu_ids)

            assert n_networks >= 2, "Need at least 2 networks for this test"

            params = stack.init(jax.random.key(42))

            log_alpha = jnp.full((n_networks, n_tus), 10.0)

            # Network 0: disable all *_a-
            for tu_id in tu_ids:
                tu_name = "_".join(tu_id.split("_")[:-1])
                if tu_name.endswith("_a-"):
                    idx = tu_id_to_idx[tu_id]
                    log_alpha = log_alpha.at[0, idx].set(-10.0)

            # Network 1: disable all *_b+
            for tu_id in tu_ids:
                tu_name = "_".join(tu_id.split("_")[:-1])
                if tu_name.endswith("_b+"):
                    idx = tu_id_to_idx[tu_id]
                    log_alpha = log_alpha.at[1, idx].set(-10.0)

            params.at(TU_LOG_ALPHA_PATH, log_alpha, overwrite=True)

            committed = stack.commit(params)

            recipe_0 = committed[0].to_recipe()
            recipe_1 = committed[1].to_recipe()

            tus_0 = get_tu_names_from_recipe(recipe_0)
            tus_1 = get_tu_names_from_recipe(recipe_1)

            # Network 0: *_a- removed, *_a+ remains with CasE_rec stripped
            assert "x1_a-" not in tus_0
            assert "x1_a+" in tus_0
            verify_ern_rec_stripped(recipe_0, "x1_a+", "CasE_rec")

            # Network 0: layer b intact
            assert "x1_b+" in tus_0
            assert "x1_b-" in tus_0

            # Network 1: *_b+ disabled → cascade removes *_b- AND *_a+ (Csy4 producers)
            assert "x1_b+" not in tus_1
            assert "x1_b-" not in tus_1
            assert "x1_a+" not in tus_1  # Cascade from Csy4 ERN!

            # Network 1: *_a- still present (not directly disabled, not cascaded)
            assert "x1_a-" in tus_1

    def test_multi_net_cascade_isolation(self, lib, multi_network_stack):
        with LibraryContext.with_library(lib):
            stack, tu_ids, tu_id_to_idx = multi_network_stack
            n_networks = len(stack.networks)
            n_tus = len(tu_ids)

            params = stack.init(jax.random.key(42))

            log_alpha = jnp.full((n_networks, n_tus), 10.0)

            for tu_id in tu_ids:
                tu_name = "_".join(tu_id.split("_")[:-1])
                if tu_name.endswith("_a+"):
                    idx = tu_id_to_idx[tu_id]
                    log_alpha = log_alpha.at[0, idx].set(-10.0)

            params.at(TU_LOG_ALPHA_PATH, log_alpha, overwrite=True)

            committed = stack.commit(params)

            recipe_0 = committed[0].to_recipe()
            recipe_1 = committed[1].to_recipe()

            tus_0 = get_tu_names_from_recipe(recipe_0)
            tus_1 = get_tu_names_from_recipe(recipe_1)

            assert "x1_a+" not in tus_0
            assert "x1_a-" not in tus_0

            assert "x1_a+" in tus_1
            assert "x1_a-" in tus_1
            assert "x1_b+" in tus_1
            assert "x1_b-" in tus_1
