"""Test commit behavior with complex scaffold recipes used in design runs."""

import pytest
import jax
import dracon as dr

import biocomp.biorules as br
from biocomp.library import LibraryContext, load_lib
from biocomp.compute import ComputeStack
from biocomp.config import SIMPLE_NODES_COMPUTE_CONFIG
from biocomp.network import recipe_to_networks
from biocomp.recipe import Recipe, Slot
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


def test_scaffold_commit_collapses_all_slots(lib):
    if not SCAFFOLD_PATH.exists():
        pytest.skip(f"Scaffold not found: {SCAFFOLD_PATH}")

    with LibraryContext.with_library(lib):
        scaffold_recipe = load_scaffold_recipe()
        networks = recipe_to_networks(scaffold_recipe, br.ALL_RULES, invert=True)
        network = networks[0]

        unlocked_before = sum(
            1
            for edge in network.compute_graph.edges.values()
            if edge.content_embedding_names
            and len(edge.content_embedding_names.get("tl_rate", ())) > 1
        )
        assert unlocked_before > 0, "Test requires scaffold with unlocked tl_rate slots"

        stack = ComputeStack([network])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)
        params = stack.init(jax.random.PRNGKey(42))
        committed = stack.commit(params)[0]

        uncommitted = [
            eid
            for eid, edge in committed.compute_graph.edges.items()
            if edge.content_embedding_names
            and len(edge.content_embedding_names.get("tl_rate", ())) > 1
        ]
        assert len(uncommitted) == 0, f"Found {len(uncommitted)} uncommitted tl_rate edges"


def test_scaffold_exported_recipe_no_lists(lib):
    if not SCAFFOLD_PATH.exists():
        pytest.skip(f"Scaffold not found: {SCAFFOLD_PATH}")

    with LibraryContext.with_library(lib):
        scaffold_recipe = load_scaffold_recipe()
        networks = recipe_to_networks(scaffold_recipe, br.ALL_RULES, invert=True)

        stack = ComputeStack([networks[0]])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)
        params = stack.init(jax.random.PRNGKey(42))
        committed = stack.commit(params)[0]
        exported_recipe = committed.to_recipe()

        list_slots = [
            (cotx.name, tu.name)
            for cotx in exported_recipe.content
            for tu in cotx.units
            for slot in tu.slots
            if isinstance(slot, Slot)
            and slot.maps_to_parameter == "tl_rate"
            and isinstance(slot.part, list)
            and len(slot.part) > 1
        ]
        assert len(list_slots) == 0, f"Found {len(list_slots)} TUs with uncommitted tl_rate slots"


def test_multiple_cotransfections_independent_commit(lib):
    if not SCAFFOLD_PATH.exists():
        pytest.skip(f"Scaffold not found: {SCAFFOLD_PATH}")

    with LibraryContext.with_library(lib):
        scaffold_recipe = load_scaffold_recipe()
        networks = recipe_to_networks(scaffold_recipe, br.ALL_RULES, invert=True)

        stack = ComputeStack([networks[0]])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)
        params = stack.init(jax.random.PRNGKey(42))
        committed = stack.commit(params)[0]

        uncommitted = [
            eid
            for eid, edge in committed.compute_graph.edges.items()
            if edge.content_embedding_names
            and len(edge.content_embedding_names.get("tl_rate", ())) > 1
        ]
        assert len(uncommitted) == 0, f"Found {len(uncommitted)} uncommitted edges"


def test_ref_id_linked_slots_have_consistent_values(lib):
    """Verify that edges sharing the same ref_id get the same committed value.

    This tests the core ref_id propagation fix: when committing, edges linked
    via ref_id (e.g., U1, U2 slots across cotransfections) should all receive
    the same optimized value.
    """
    if not SCAFFOLD_PATH.exists():
        pytest.skip(f"Scaffold not found: {SCAFFOLD_PATH}")

    with LibraryContext.with_library(lib):
        scaffold_recipe = load_scaffold_recipe()
        networks = recipe_to_networks(scaffold_recipe, br.ALL_RULES, invert=True)
        network = networks[0]
        graph = network.compute_graph

        # build mapping from ref_id to tu_ids
        ref_id_to_tu_ids = {}
        for node in graph.nodes.values():
            if node.node_type == "source" and node.extra:
                param_ref_ids = node.extra.get("param_ref_ids", {})
                ref_id = param_ref_ids.get("tl_rate")
                if ref_id:
                    tu_name = node.extra.get("name", "")
                    cotx = node.extra.get("cotx_group", "")
                    tu_id = f"{tu_name}_{cotx}"
                    if ref_id not in ref_id_to_tu_ids:
                        ref_id_to_tu_ids[ref_id] = []
                    ref_id_to_tu_ids[ref_id].append(tu_id)

        # need at least one ref_id with multiple tu_ids
        multi_tu_refs = {k: v for k, v in ref_id_to_tu_ids.items() if len(v) > 1}
        assert len(multi_tu_refs) > 0, "Test requires scaffold with shared ref_ids"

        # commit
        stack = ComputeStack([network])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)
        params = stack.init(jax.random.PRNGKey(42))
        committed = stack.commit(params)[0]
        committed_graph = committed.compute_graph

        # verify: for each ref_id, all edges with linked tu_ids have same tl_rate
        for ref_id, tu_ids in multi_tu_refs.items():
            values_for_ref = []
            for edge in committed_graph.edges.values():
                edge_tu_ids = edge.extra.get("tu_id", []) if edge.extra else []
                if any(tu_id in edge_tu_ids for tu_id in tu_ids):
                    tl_rate = edge.content_embedding_names.get("tl_rate")
                    if tl_rate:
                        values_for_ref.append(tl_rate)

            unique_values = set(values_for_ref)
            assert len(unique_values) == 1, (
                f"ref_id '{ref_id}' has inconsistent committed values: {unique_values}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
