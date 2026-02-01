"""Test ERN node pruning during design commit with TU masking."""

import pytest
import jax
import jax.numpy as jnp
import dracon as dr

import biocomp.biorules as br
from biocomp.library import LibraryContext, load_lib
from biocomp.compute import ComputeStack
from biocomp.config import SIMPLE_NODES_COMPUTE_CONFIG
from biocomp.network import recipe_to_networks
from biocomp.recipe import Recipe
from biocomp.tumasking import TU_LOG_ALPHA_PATH
from pathlib import Path

RESOURCES_DIR = Path(__file__).parent / "resources"
SCAFFOLD_PATH = RESOURCES_DIR / "design/architectures/two_and_one.yaml"


@pytest.fixture
def lib():
    return load_lib()


def load_scaffold_recipe():
    data = dr.load(SCAFFOLD_PATH, context={"Recipe": Recipe})
    if hasattr(data, "recipes") or (hasattr(data, "__getitem__") and "recipes" in data):
        recipes = data["recipes"] if "recipes" in data else data.recipes
        return recipes[0]
    if isinstance(data, Recipe):
        return data
    raise ValueError(f"Unexpected scaffold format: {type(data)}")


def get_ern_nodes(network):
    """Get all ERN nodes from a network."""
    return [n for n in network.compute_graph.nodes.values() if n.node_type == "sequestron_ERN"]


def get_tu_ids_for_network(network):
    """Get all TU IDs from network edges."""
    tu_ids = set()
    for edge in network.compute_graph.edges.values():
        if edge.extra:
            tu_ids.update(edge.extra.get("tu_id", []))
    return tu_ids


def build_tu_id_to_idx(network):
    """Build tu_id_to_idx mapping from network edges."""
    tu_ids = sorted(get_tu_ids_for_network(network))
    return {tu_id: idx for idx, tu_id in enumerate(tu_ids)}


def find_tu_by_pattern(network, pattern: str):
    """Find TU IDs containing the given pattern."""
    tu_ids = get_tu_ids_for_network(network)
    return [tid for tid in tu_ids if pattern in tid]


def get_ern_neg_tu_ids(network) -> set[str]:
    neg_tu_ids: set[str] = set()
    for node in get_ern_nodes(network):
        incoming = list(network.compute_graph.get_incoming_edges(node.node_id))
        for edge in incoming:
            if edge.to_input_slot != 0:
                continue
            if edge.extra:
                neg_tu_ids.update(edge.extra.get("tu_id", []))
    return neg_tu_ids


class TestErnInputStates:
    """Test get_ern_input_states() method."""

    def test_both_enabled(self, lib):
        """ERN with both inputs enabled returns (True, True)."""
        if not SCAFFOLD_PATH.exists():
            pytest.skip("Scaffold not found")

        with LibraryContext.with_library(lib):
            recipe = load_scaffold_recipe()
            networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True)
            network = networks[0]

            ern_nodes = get_ern_nodes(network)
            assert len(ern_nodes) > 0, "Test requires scaffold with ERN nodes"

            tu_id_to_idx = build_tu_id_to_idx(network)
            n_tus = len(tu_id_to_idx)
            tu_log_alpha = jnp.zeros(n_tus) + 10.0  # all enabled

            states = network.get_ern_input_states(tu_log_alpha, tu_id_to_idx)
            assert len(states) == len(ern_nodes)

            for node_id, (neg_enabled, pos_enabled) in states.items():
                assert neg_enabled, f"ERN {node_id} should have neg enabled"
                assert pos_enabled, f"ERN {node_id} should have pos enabled"

    def test_negative_disabled(self, lib):
        """ERN with negative input disabled returns (False, True)."""
        if not SCAFFOLD_PATH.exists():
            pytest.skip("Scaffold not found")

        with LibraryContext.with_library(lib):
            recipe = load_scaffold_recipe()
            networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True)
            network = networks[0]

            tu_id_to_idx = build_tu_id_to_idx(network)
            neg_tu_ids = find_tu_by_pattern(network, "_a-")
            assert len(neg_tu_ids) > 0, "Test requires TUs with pattern _a-"

            n_tus = len(tu_id_to_idx)
            tu_log_alpha = jnp.zeros(n_tus) + 10.0
            for tu_id in neg_tu_ids:
                if tu_id in tu_id_to_idx:
                    tu_log_alpha = tu_log_alpha.at[tu_id_to_idx[tu_id]].set(-10.0)

            states = network.get_ern_input_states(tu_log_alpha, tu_id_to_idx)
            found_disabled_neg = any(not neg for neg, pos in states.values())
            assert found_disabled_neg, "Should find at least one ERN with disabled negative input"


class TestErnCleanup:
    """Test _cleanup_ern_nodes() method."""

    def test_negative_disabled_returns_strip_parts(self, lib):
        """When negative disabled, returns ERN_rec parts to strip."""
        if not SCAFFOLD_PATH.exists():
            pytest.skip("Scaffold not found")

        with LibraryContext.with_library(lib):
            recipe = load_scaffold_recipe()
            networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True)
            network = networks[0]

            tu_id_to_idx = build_tu_id_to_idx(network)
            neg_tu_ids = find_tu_by_pattern(network, "_a-")
            n_tus = len(tu_id_to_idx)
            tu_log_alpha = jnp.zeros(n_tus) + 10.0
            for tu_id in neg_tu_ids:
                if tu_id in tu_id_to_idx:
                    tu_log_alpha = tu_log_alpha.at[tu_id_to_idx[tu_id]].set(-10.0)

            strip_ern_recs = network._cleanup_ern_nodes(tu_log_alpha, tu_id_to_idx)

            assert len(strip_ern_recs) > 0, "Should return ERN_rec parts to strip"
            for _tu_id, part_name in strip_ern_recs:
                assert "_rec" in part_name, f"Should strip recognition sites, got {part_name}"

    def test_positive_disabled_only_cascades_exclusive_neg_tus(self, lib):
        """When positive disabled, only EXCLUSIVE neg TUs are marked for cascade disable.

        Neg TUs that feed multiple ERNs (shared, common case) are NOT cascade-disabled.
        This prevents breaking other ERNs that still need the shared neg TU.
        """
        if not SCAFFOLD_PATH.exists():
            pytest.skip("Scaffold not found")

        with LibraryContext.with_library(lib):
            recipe = load_scaffold_recipe()
            networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True)
            network = networks[0]

            tu_id_to_idx = build_tu_id_to_idx(network)

            # find exclusive neg TUs (safe to cascade)
            exclusive_neg_tus = network.find_exclusive_ern_neg_tus(tu_id_to_idx)

            pos_tu_ids = find_tu_by_pattern(network, "_a+")
            n_tus = len(tu_id_to_idx)
            tu_log_alpha = jnp.zeros(n_tus) + 10.0
            for tu_id in pos_tu_ids:
                if tu_id in tu_id_to_idx:
                    tu_log_alpha = tu_log_alpha.at[tu_id_to_idx[tu_id]].set(-10.0)

            network._cleanup_ern_nodes(tu_log_alpha, tu_id_to_idx)

            additional_disabled = network.metadata.get("_additional_disabled_tus", set())

            # only exclusive neg TUs should be cascade-disabled
            for disabled_tu in additional_disabled:
                assert disabled_tu in exclusive_neg_tus, (
                    f"TU {disabled_tu} was cascade-disabled but is not exclusive "
                    f"(shared neg TUs should NOT be cascade-disabled)"
                )

    def test_find_exclusive_ern_neg_tus(self, lib):
        """Test find_exclusive_ern_neg_tus correctly identifies exclusive vs shared neg TUs."""
        if not SCAFFOLD_PATH.exists():
            pytest.skip("Scaffold not found")

        with LibraryContext.with_library(lib):
            recipe = load_scaffold_recipe()
            networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True)
            network = networks[0]

            tu_id_to_idx = build_tu_id_to_idx(network)
            exclusive_neg_tus = network.find_exclusive_ern_neg_tus(tu_id_to_idx)

            neg_tu_ids = get_ern_neg_tu_ids(network)
            assert exclusive_neg_tus.issubset(neg_tu_ids), (
                "Exclusive neg TUs must be subset of ERN negative sources"
            )

            # verify exclusive TUs are valid TU IDs
            for tu_id in exclusive_neg_tus:
                assert tu_id in tu_id_to_idx, f"Exclusive TU {tu_id} not in tu_id_to_idx"


class TestCommitWithErnPruning:
    """Integration tests for full commit with ERN pruning."""

    def test_commit_with_disabled_negative_produces_valid_network(self, lib):
        """Committed network is valid when ERN's negative input disabled."""
        if not SCAFFOLD_PATH.exists():
            pytest.skip("Scaffold not found")

        with LibraryContext.with_library(lib):
            recipe = load_scaffold_recipe()
            networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True)
            network = networks[0]

            stack = ComputeStack([network])
            stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)
            params = stack.init(jax.random.PRNGKey(42))

            neg_tu_ids = find_tu_by_pattern(network, "_a-")
            if TU_LOG_ALPHA_PATH not in params:
                pytest.skip("No TU masking in params")

            tu_log_alpha = params[TU_LOG_ALPHA_PATH]
            for tu_id in neg_tu_ids:
                if tu_id in stack.tu_id_to_idx:
                    idx = stack.tu_id_to_idx[tu_id]
                    tu_log_alpha = tu_log_alpha.at[0, idx].set(-10.0)
            params = params.set(TU_LOG_ALPHA_PATH, tu_log_alpha)

            committed = stack.commit(params)
            assert len(committed) == 1

            committed_net = committed[0]
            ern_nodes = get_ern_nodes(committed_net)
            for ern in ern_nodes:
                incoming = list(committed_net.compute_graph.get_incoming_edges(ern.node_id))
                assert len(incoming) == 2, f"ERN {ern.node_id} should have 2 inputs after commit"

    def test_commit_recipe_strips_dead_ern_rec(self, lib):
        """Recipe after commit has no dead ERN_rec parts."""
        if not SCAFFOLD_PATH.exists():
            pytest.skip("Scaffold not found")

        with LibraryContext.with_library(lib):
            recipe = load_scaffold_recipe()
            networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True)
            network = networks[0]

            stack = ComputeStack([network])
            stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)
            params = stack.init(jax.random.PRNGKey(42))

            neg_tu_ids = find_tu_by_pattern(network, "_a-")
            if TU_LOG_ALPHA_PATH not in params:
                pytest.skip("No TU masking in params")

            tu_log_alpha = params[TU_LOG_ALPHA_PATH]
            for tu_id in neg_tu_ids:
                if tu_id in stack.tu_id_to_idx:
                    idx = stack.tu_id_to_idx[tu_id]
                    tu_log_alpha = tu_log_alpha.at[0, idx].set(-10.0)
            params = params.set(TU_LOG_ALPHA_PATH, tu_log_alpha)

            committed = stack.commit(params)
            committed_net = committed[0]

            exported = committed_net.to_recipe()

            for cotx in exported.content:
                for tu in cotx.units:
                    if "_a+" in tu.name:
                        slot_parts = [s.part if hasattr(s, "part") else str(s) for s in tu.slots]
                        assert "CasE_rec" not in slot_parts, (
                            f"TU {tu.name} should have CasE_rec stripped"
                        )

    def test_commit_all_enabled_preserves_all_ern_recs(self, lib):
        """When all TUs enabled, ERN_rec parts are preserved."""
        if not SCAFFOLD_PATH.exists():
            pytest.skip("Scaffold not found")

        with LibraryContext.with_library(lib):
            recipe = load_scaffold_recipe()
            networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True)
            network = networks[0]

            stack = ComputeStack([network])
            stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)
            params = stack.init(jax.random.PRNGKey(42))

            committed = stack.commit(params)
            committed_net = committed[0]

            ern_nodes = get_ern_nodes(committed_net)
            assert len(ern_nodes) > 0, "Should preserve ERN nodes when all enabled"

            exported = committed_net.to_recipe()
            ern_recs_found = []
            for cotx in exported.content:
                for tu in cotx.units:
                    for slot in tu.slots:
                        part = slot.part if hasattr(slot, "part") else str(slot)
                        if isinstance(part, str) and "_rec" in part:
                            ern_recs_found.append(part)
            assert len(ern_recs_found) > 0, "Should preserve ERN_rec parts when all enabled"


class TestEdgeCases:
    """Edge case tests for ERN pruning."""

    def test_multiple_networks_different_states(self, lib):
        """Handle multiple networks with different ERN states."""
        if not SCAFFOLD_PATH.exists():
            pytest.skip("Scaffold not found")

        with LibraryContext.with_library(lib):
            recipe = load_scaffold_recipe()
            networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True)
            if len(networks) < 2:
                networks = [networks[0], networks[0].model_copy(deep=True)]
                networks[1].name = networks[0].name + "_copy"

            stack = ComputeStack(networks)
            stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)
            params = stack.init(jax.random.PRNGKey(42))

            if TU_LOG_ALPHA_PATH not in params:
                pytest.skip("No TU masking in params")

            neg_tu_ids = find_tu_by_pattern(networks[0], "_a-")
            tu_log_alpha = params[TU_LOG_ALPHA_PATH]
            for tu_id in neg_tu_ids:
                if tu_id in stack.tu_id_to_idx:
                    idx = stack.tu_id_to_idx[tu_id]
                    tu_log_alpha = tu_log_alpha.at[0, idx].set(-10.0)
            params = params.set(TU_LOG_ALPHA_PATH, tu_log_alpha)

            committed = stack.commit(params)
            assert len(committed) == len(networks)

    def test_no_ern_nodes_noop(self, lib):
        """Networks without ERN nodes pass through unchanged."""
        with LibraryContext.with_library(lib):
            from biocomp.recipe import Recipe, CoTransfection, TranscriptionUnit

            simple_recipe = Recipe(
                name="simple_no_ern",
                content=[
                    CoTransfection(
                        name="cotx1",
                        units=[
                            TranscriptionUnit(name="tu1", slots=["hEF1a", "mNeonGreen"]),
                        ],
                    )
                ],
            )
            networks = recipe_to_networks(simple_recipe, br.ALL_RULES, invert=True)
            network = networks[0]

            ern_nodes = get_ern_nodes(network)
            assert len(ern_nodes) == 0, "Test requires network without ERN nodes"

            n_tus = max(1, len(get_tu_ids_for_network(network)))
            tu_log_alpha = jnp.zeros(n_tus) + 10.0
            tu_id_to_idx = {}

            strip_ern_recs = network._cleanup_ern_nodes(tu_log_alpha, tu_id_to_idx)
            assert len(strip_ern_recs) == 0
