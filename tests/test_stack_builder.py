"""Test cases for stack_builder module - topological ordering and graph building.

Focus: Defensive programming tests for edge cases like cycles, disconnected graphs.
"""

import pytest
from biocomp.graphengine import GraphState, GraphNode, GraphEdge, apply_rule_sequence
from biocomp.graphrules import GraphRewritingRule, MatchQuery, PropertyConstraint, DeleteNode
from biocomp.stack_builder import topological_order


def make_graph(nodes_data: list[dict], edges_data: list[tuple]) -> GraphState:
    """Helper to create GraphState from simple node/edge specs.

    nodes_data: list of {"id": int, "type": str, ...}
    edges_data: list of (source_id, target_id) tuples
    """
    nodes = {
        n["id"]: GraphNode(
            node_id=n["id"],
            node_type=n.get("type", "node"),
            extra={k: v for k, v in n.items() if k not in ("id", "type")},
        )
        for n in nodes_data
    }
    edges = {
        (src, tgt, 0, 0): GraphEdge(
            source_id=src, target_id=tgt, from_output_slot=0, to_input_slot=0, content=()
        )
        for src, tgt in edges_data
    }
    return GraphState(nodes=nodes, edges=edges)


def test_topological_order_linear_chain():
    """Linear chain: A -> B -> C -> D should produce batches [[A], [B], [C], [D]]."""
    graph = make_graph(
        [{"id": i, "type": "node"} for i in range(4)],
        [(0, 1), (1, 2), (2, 3)],
    )
    batches = topological_order(graph)
    assert len(batches) == 4
    assert batches[0] == [0]
    assert batches[1] == [1]
    assert batches[2] == [2]
    assert batches[3] == [3]


def test_topological_order_parallel_independent():
    """Independent nodes can be in same batch."""
    graph = make_graph(
        [{"id": 0}, {"id": 1}, {"id": 2}, {"id": 3}],
        [(0, 2), (1, 2), (2, 3)],  # 0,1 -> 2 -> 3
    )
    batches = topological_order(graph)
    # First batch should contain both 0 and 1 (no dependencies)
    assert len(batches) == 3
    assert set(batches[0]) == {0, 1}
    assert batches[1] == [2]
    assert batches[2] == [3]


def test_topological_order_diamond():
    """Diamond pattern: A -> B,C -> D."""
    graph = make_graph(
        [{"id": 0}, {"id": 1}, {"id": 2}, {"id": 3}],
        [(0, 1), (0, 2), (1, 3), (2, 3)],
    )
    batches = topological_order(graph)
    assert len(batches) == 3
    assert batches[0] == [0]
    assert set(batches[1]) == {1, 2}
    assert batches[2] == [3]


def test_topological_order_cycle_raises_error():
    """CRITICAL: Cycles must raise ValueError, not hang or return invalid ordering.

    This test ensures the defensive programming principle 'Fail Fast, Fail Loud'.
    Without this check, a cycle could cause infinite loops or silently produce
    incorrect compute orderings.
    """
    # Simple cycle: A -> B -> C -> A
    graph = make_graph(
        [{"id": 0}, {"id": 1}, {"id": 2}],
        [(0, 1), (1, 2), (2, 0)],  # Creates cycle
    )
    with pytest.raises(ValueError, match="No independent node"):
        topological_order(graph)


def test_topological_order_self_loop_raises_error():
    """Self-loop (A -> A) should also raise ValueError."""
    graph = make_graph(
        [{"id": 0}, {"id": 1}],
        [(0, 0), (0, 1)],  # Self-loop on node 0
    )
    with pytest.raises(ValueError, match="No independent node"):
        topological_order(graph)


def test_topological_order_complex_cycle():
    """Complex graph with embedded cycle should fail."""
    # Graph: 0 -> 1 -> 2 -> 3 -> 1 (cycle through 1-2-3)
    #              └---4
    graph = make_graph(
        [{"id": i} for i in range(5)],
        [(0, 1), (1, 2), (2, 3), (3, 1), (1, 4)],  # Cycle: 1->2->3->1
    )
    with pytest.raises(ValueError, match="No independent node"):
        topological_order(graph)


def test_topological_order_disconnected_graph():
    """Disconnected components should all be processed."""
    # Two disconnected chains: 0->1 and 2->3
    graph = make_graph(
        [{"id": i} for i in range(4)],
        [(0, 1), (2, 3)],
    )
    batches = topological_order(graph)
    # First batch: all nodes with no incoming edges (0 and 2)
    assert set(batches[0]) == {0, 2}
    # Second batch: their children
    assert set(batches[1]) == {1, 3}


def test_topological_order_single_node():
    """Single node with no edges."""
    graph = make_graph([{"id": 0}], [])
    batches = topological_order(graph)
    assert batches == [[0]]


def test_topological_order_empty_graph():
    """Empty graph should return empty batches."""
    graph = GraphState(nodes={}, edges={})
    batches = topological_order(graph)
    assert batches == []


def test_topological_order_deterministic():
    """Same graph should produce same ordering (sorted within batches)."""
    graph = make_graph(
        [{"id": i} for i in range(5)],
        [(0, 3), (1, 3), (2, 3), (3, 4)],
    )
    batches1 = topological_order(graph)
    batches2 = topological_order(graph)
    assert batches1 == batches2
    # First batch should be sorted
    assert batches1[0] == sorted(batches1[0])


# ============================================================================
# Graph Integrity Validation Tests
# ============================================================================


def test_validate_integrity_valid_graph():
    """Valid graph should pass integrity check."""
    graph = make_graph(
        [{"id": 0}, {"id": 1}, {"id": 2}],
        [(0, 1), (1, 2)],
    )
    # Should not raise
    graph.validate_integrity()


def test_validate_integrity_dangling_source():
    """Edge with non-existent source should fail validation."""
    nodes = {
        0: GraphNode(node_id=0, node_type="node"),
        1: GraphNode(node_id=1, node_type="node"),
    }
    edges = {
        (99, 1, 0, 0): GraphEdge(  # source 99 doesn't exist
            source_id=99, target_id=1, from_output_slot=0, to_input_slot=0, content=()
        )
    }
    graph = GraphState(nodes=nodes, edges=edges)

    with pytest.raises(AssertionError, match="Dangling edge.*source_id 99"):
        graph.validate_integrity()


def test_validate_integrity_dangling_target():
    """Edge with non-existent target should fail validation."""
    nodes = {
        0: GraphNode(node_id=0, node_type="node"),
        1: GraphNode(node_id=1, node_type="node"),
    }
    edges = {
        (0, 99, 0, 0): GraphEdge(  # target 99 doesn't exist
            source_id=0, target_id=99, from_output_slot=0, to_input_slot=0, content=()
        )
    }
    graph = GraphState(nodes=nodes, edges=edges)

    with pytest.raises(AssertionError, match="Dangling edge.*target_id 99"):
        graph.validate_integrity()


def test_validate_integrity_empty_graph():
    """Empty graph should pass validation."""
    graph = GraphState(nodes={}, edges={})
    graph.validate_integrity()


def test_apply_rule_sequence_validates_by_default():
    """apply_rule_sequence should validate graph integrity after each rule.

    This test verifies that the validation parameter works - if we somehow
    create a bad graph through rules, it should be caught.
    """
    # Create a valid graph
    graph = make_graph(
        [{"id": 0, "type": "keep"}, {"id": 1, "type": "keep"}, {"id": 2, "type": "remove"}],
        [(0, 1), (1, 2)],  # 0 -> 1 -> 2
    )

    # Rule that deletes node 2 - this should also delete the edge (1, 2)
    # The engine should handle this properly
    rule = GraphRewritingRule(
        name="Delete remove nodes",
        query=MatchQuery(bind={"n": PropertyConstraint(properties={"type": "remove"})}),
        actions=[DeleteNode(node_var="n")],
    )

    # Should succeed because engine properly removes edges to deleted nodes
    result = apply_rule_sequence([rule], graph, validate=True)
    assert len(result) == 1
    assert len(result[0].nodes) == 2
    assert len(result[0].edges) == 1  # Only edge (0, 1) remains
