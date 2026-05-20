# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
import pytest
from biocomp.graphrules import (
    GraphRewritingRule,
    MatchQuery,
    PropertyConstraint,
    EdgeConstraint,
    AddNode,
    AddEdge,
    SetProperties,
    DeleteNode,
    DeleteEdge,
    RewireEdgesFrom,
    RewireEdgesTo,
    EditEdge,
    CopyEdge,
)
from biocomp.graphengine import GraphState, GraphNode, GraphEdge, apply_rule

# ---------------------------------------------------------------------------
# Helper Fixtures and Functions
# ---------------------------------------------------------------------------


def create_graph_state(nodes_data, edges_data):
    nodes = (
        {
            node["id"]: GraphNode(
                node_id=node["id"],
                node_type=node.get("type", "unknown"),
                extra={k: v for k, v in node.items() if k not in ["id", "type"]},
            )
            for node in nodes_data
        }
        if nodes_data
        else {}
    )

    edges_list = (
        [
            GraphEdge(
                source_id=edge["source"],
                target_id=edge["target"],
                from_output_slot=edge.get("from_output_slot", 0),
                to_input_slot=edge.get("to_input_slot", 0),
                content=edge.get("content", ()),
                content_type=edge.get("content_type", None),
            )
            for edge in edges_data
        ]
        if edges_data
        else []
    )

    edges = {(e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e for e in edges_list}
    return GraphState(nodes=nodes, edges=edges)


@pytest.fixture
def simple_graph():
    """A -> B, B -> C"""
    nodes_data = [
        {"id": 0, "type": "promoter", "name": "pA"},
        {"id": 1, "type": "gene", "name": "gB"},
        {"id": 2, "type": "terminator", "name": "tC"},
    ]
    edges_data = [
        {"source": 0, "target": 1},
        {"source": 1, "target": 2},
    ]
    return create_graph_state(nodes_data, edges_data)


def test_match_single_node_by_property(simple_graph):
    rule = GraphRewritingRule(
        name="Find Gene B",
        query=MatchQuery(bind={"g": PropertyConstraint(properties={"type": "gene", "name": "gB"})}),
        actions=[SetProperties(node_var="g", properties={"matched": True})],
    )

    new_graphs = apply_rule(rule, simple_graph)
    assert len(new_graphs) == 1
    new_graph = new_graphs[0]

    # Find the node with id 1
    gene_node = next(node for node in new_graph.nodes.values() if node.node_id == 1)
    assert gene_node.extra["matched"]

    # Check that node 0 is untouched
    promoter_node = next(node for node in new_graph.nodes.values() if node.node_id == 0)
    assert "matched" not in promoter_node.extra


def test_no_match_found(simple_graph):
    rule = GraphRewritingRule(
        name="Find Non-existent Node",
        query=MatchQuery(bind={"x": PropertyConstraint(properties={"type": "nonexistent"})}),
        actions=[AddNode(local_name="new", properties={"type": "new"})],
    )

    new_graphs = apply_rule(rule, simple_graph)
    assert len(new_graphs) == 1
    new_graph = new_graphs[0]

    # Graph should be identical since no match was found
    assert len(new_graph.nodes) == len(simple_graph.nodes)
    assert len(new_graph.edges) == len(simple_graph.edges)

    # Check nodes are the same
    for orig_node, new_node in zip(simple_graph.nodes.values(), new_graph.nodes.values(), strict=False):
        assert orig_node.node_id == new_node.node_id
        assert orig_node.node_type == new_node.node_type
        assert orig_node.extra == new_node.extra


def test_match_connected_nodes(simple_graph):
    rule = GraphRewritingRule(
        name="Find Promoter connected to Gene",
        query=MatchQuery(
            bind={
                "p": PropertyConstraint(properties={"type": "promoter"}),
                "g": PropertyConstraint(properties={"type": "gene"}),
            },
            where_connected=[EdgeConstraint(source_var="p", target_var="g")],
        ),
        actions=[
            SetProperties(node_var="p", properties={"p_matched": True}),
            SetProperties(node_var="g", properties={"g_matched": True}),
        ],
    )

    new_graphs = apply_rule(rule, simple_graph)
    assert len(new_graphs) == 1
    new_graph = new_graphs[0]

    promoter_node = next(node for node in new_graph.nodes.values() if node.node_id == 0)
    gene_node = next(node for node in new_graph.nodes.values() if node.node_id == 1)
    terminator_node = next(node for node in new_graph.nodes.values() if node.node_id == 2)

    assert promoter_node.extra["p_matched"]
    assert gene_node.extra["g_matched"]
    assert "p_matched" not in terminator_node.extra


def test_match_with_negative_constraint(simple_graph):
    # This rule should match node 0 because it has an outgoing edge, but not an incoming one.
    rule = GraphRewritingRule(
        name="Find Root Node",
        query=MatchQuery(
            bind={"root": PropertyConstraint(properties={})},  # Match any node
            where_not_connected=[EdgeConstraint(source_var="any", target_var="root")],
        ),
        actions=[SetProperties(node_var="root", properties={"is_root": True})],
    )

    new_graphs = apply_rule(rule, simple_graph)
    assert len(new_graphs) == 1
    new_graph = new_graphs[0]

    # Node 0 should be marked as root
    root_node = next(node for node in new_graph.nodes.values() if node.node_id == 0)
    assert root_node.extra["is_root"]

    # Other nodes should not be marked as root
    for node in new_graph.nodes.values():
        if node.node_id != 0:
            assert "is_root" not in node.extra


def test_action_add_node(simple_graph):
    rule = GraphRewritingRule(
        name="Add Node for every Gene",
        query=MatchQuery(bind={"g": PropertyConstraint(properties={"type": "gene"})}),
        actions=[
            AddNode(
                local_name="new_node",
                properties={"type": "marker", "linked_to": "{{g.extra['name']}}"},
            )
        ],
    )

    new_graphs = apply_rule(rule, simple_graph)
    assert len(new_graphs) == 1
    new_graph = new_graphs[0]

    assert len(new_graph.nodes) == 4
    marker_nodes = [node for node in new_graph.nodes.values() if node.node_type == "marker"]
    assert len(marker_nodes) == 1
    assert marker_nodes[0].extra["linked_to"] == "gB"  # Test templating


def test_action_add_edge(simple_graph):
    rule = GraphRewritingRule(
        name="Add feedback loop from C to A",
        query=MatchQuery(
            bind={
                "start": PropertyConstraint(properties={"name": "pA"}),
                "end": PropertyConstraint(properties={"name": "tC"}),
            }
        ),
        actions=[AddEdge(source="end", target="start")],
    )

    new_graphs = apply_rule(rule, simple_graph)
    assert len(new_graphs) == 1
    new_graph = new_graphs[0]

    assert len(new_graph.edges) == 3
    feedback_edges = [
        edge for edge in new_graph.edges.values() if edge.source_id == 2 and edge.target_id == 0
    ]
    assert len(feedback_edges) == 1


def test_action_delete_node(simple_graph):
    rule = GraphRewritingRule(
        name="Delete Gene B",
        query=MatchQuery(bind={"g": PropertyConstraint(properties={"name": "gB"})}),
        actions=[DeleteNode(node_var="g")],
    )

    new_graphs = apply_rule(rule, simple_graph)
    assert len(new_graphs) == 1
    new_graph = new_graphs[0]

    assert len(new_graph.nodes) == 2
    node_ids = {node.node_id for node in new_graph.nodes.values()}
    assert 1 not in node_ids
    # Engine should also delete connected edges
    assert len(new_graph.edges) == 0


def test_action_delete_edge(simple_graph):
    rule = GraphRewritingRule(
        name="Delete B->C link",
        query=MatchQuery(
            bind={
                "b": PropertyConstraint(properties={"name": "gB"}),
                "c": PropertyConstraint(properties={"name": "tC"}),
            },
            where_connected=[EdgeConstraint(source_var="b", target_var="c")],
        ),
        actions=[DeleteEdge(source_var="b", target_var="c")],
    )

    new_graphs = apply_rule(rule, simple_graph)
    assert len(new_graphs) == 1
    new_graph = new_graphs[0]

    assert len(new_graph.nodes) == 3  # Nodes are untouched
    assert len(new_graph.edges) == 1
    remaining_edge = list(new_graph.edges.values())[0]
    assert remaining_edge.source_id == 0  # Only A->B edge remains
    assert remaining_edge.target_id == 1


def test_subgraph_replacement(simple_graph):
    # Rule: Find "promoter -> gene" and replace it with a single "expression_cassette" node

    rule = GraphRewritingRule(
        name="Fuse Promoter and Gene",
        query=MatchQuery(
            bind={
                "p": PropertyConstraint(properties={"type": "promoter"}),
                "g": PropertyConstraint(properties={"type": "gene"}),
            },
            where_connected=[EdgeConstraint(source_var="p", target_var="g")],
        ),
        actions=[
            AddNode(
                local_name="cassette",
                properties={
                    "type": "cassette",
                    "name": "{{ p.extra['name'] + '+' + g.extra['name'] }}",
                },
            ),
            # This is a conceptual "rewire" action. The engine needs to implement this.
            # It means: "find all edges that go FROM g, and make them come from cassette instead".
            # For this simple graph, there's one edge from G to the terminator.
            RewireEdgesFrom(old_source_var="g", new_source_var="cassette"),
            DeleteNode(node_var="p"),
            DeleteNode(node_var="g"),
        ],
    )

    new_graphs = apply_rule(rule, simple_graph)
    assert len(new_graphs) == 1
    new_graph = new_graphs[0]

    assert len(new_graph.nodes) == 2  # cassette + terminator
    cassette_nodes = [node for node in new_graph.nodes.values() if node.node_type == "cassette"]
    assert len(cassette_nodes) == 1
    cassette_node = cassette_nodes[0]
    assert cassette_node.extra["name"] == "pA+gB"  # Jinja2 renders complex template as string

    assert len(new_graph.edges) == 1
    final_edge = list(new_graph.edges.values())[0]
    assert final_edge.source_id == cassette_node.node_id
    assert final_edge.target_id == 2  # The terminator node


def test_iterative_rule_application_run_until_stable():
    # Graph: A -> B -> C -> D. Rule: Fuse a parent with its child.
    graph = create_graph_state(
        [{"id": i, "type": "node", "val": i} for i in range(4)],
        [{"source": i, "target": i + 1} for i in range(3)],
    )

    # Rule: find X -> Y and replace with a single node Z with value X+Y,
    # rewiring all of Y's children to be children of Z.
    rule = GraphRewritingRule(
        name="Fuse Parent and Child",
        query=MatchQuery(
            bind={
                "parent": PropertyConstraint(properties={"type": "node"}),
                "child": PropertyConstraint(properties={"type": "node"}),
            },
            where_connected=[EdgeConstraint(source_var="parent", target_var="child")],
        ),
        actions=[
            AddNode(
                local_name="fused",
                properties={
                    "type": "node",
                    "val": "{{ str(parent.extra['val']) + '+' + str(child.extra['val']) }}",
                },  # String concat for simplicity
            ),
            RewireEdgesFrom(old_source_var="child", new_source_var="fused"),
            RewireEdgesTo(old_target_var="parent", new_target_var="fused"),
            DeleteNode(node_var="parent"),
            DeleteNode(node_var="child"),
        ],
        run_until_stable=True,
    )

    # The `apply_rule` function needs to respect `run_until_stable`.
    final_graphs = apply_rule(rule, graph)
    assert len(final_graphs) == 1
    final_graph = final_graphs[0]

    # Expected outcome: With run_until_stable=True, the rule should keep fusing
    # connected nodes until no more matches exist. Starting with chain 0->1->2->3:
    # Iteration 1: (0,1) and (2,3) fuse -> two nodes "0+1" and "2+3" with edge between
    # Iteration 2: The two fused nodes are connected, so they fuse -> one node "0+1+2+3"
    # Stable: No more connected pairs
    assert len(final_graph.nodes) == 1
    node_vals = [n.extra["val"] for n in final_graph.nodes.values()]
    assert node_vals == ["0+1+2+3"]  # All nodes fused into one (string concatenation)
    assert len(final_graph.edges) == 0


def test_yield_strategy_batched_vs_per_match():
    # Graph with two disconnected promoters
    graph = create_graph_state(
        [
            {"id": 0, "type": "promoter", "name": "pA"},
            {"id": 1, "type": "promoter", "name": "pB"},
        ],
        [],
    )

    rule_batched = GraphRewritingRule(
        name="Add markers to all promoters (batched)",
        query=MatchQuery(bind={"p": PropertyConstraint(properties={"type": "promoter"})}),
        actions=[
            AddNode(
                local_name="marker", properties={"type": "marker", "for": "{{p.extra['name']}}"}
            )
        ],
        yield_strategy="batched",
    )

    rule_per_match = GraphRewritingRule(
        name="Add markers to all promoters (per match)",
        query=MatchQuery(bind={"p": PropertyConstraint(properties={"type": "promoter"})}),
        actions=[
            AddNode(
                local_name="marker", properties={"type": "marker", "for": "{{p.extra['name']}}"}
            )
        ],
        yield_strategy="per_match",
    )

    # Batched strategy should return one graph with both markers
    batched_graphs = apply_rule(rule_batched, graph)
    assert len(batched_graphs) == 1
    batched_graph = batched_graphs[0]
    assert len(batched_graph.nodes) == 4  # 2 promoters + 2 markers
    markers = [n for n in batched_graph.nodes.values() if n.node_type == "marker"]
    assert len(markers) == 2

    # Per-match strategy should return two graphs, each with one marker
    per_match_graphs = apply_rule(rule_per_match, graph)
    assert len(per_match_graphs) == 2
    for result_graph in per_match_graphs:
        assert len(result_graph.nodes) == 3  # 2 promoters + 1 marker
        markers = [n for n in result_graph.nodes.values() if n.node_type == "marker"]
        assert len(markers) == 1


def test_empty_graph():
    empty_graph = create_graph_state([], [])

    rule = GraphRewritingRule(
        name="Find anything",
        query=MatchQuery(bind={"x": PropertyConstraint(properties={})}),
        actions=[AddNode(local_name="new", properties={"type": "created"})],
    )

    result = apply_rule(rule, empty_graph)
    assert len(result) == 1
    assert len(result[0].nodes) == 0
    assert len(result[0].edges) == 0


def test_multiple_property_constraints():
    graph = create_graph_state(
        [
            {"id": 0, "type": "gene", "name": "geneA", "active": True},
            {"id": 1, "type": "gene", "name": "geneB", "active": False},
            {"id": 2, "type": "promoter", "name": "promoterA", "active": True},
        ],
        [],
    )

    rule = GraphRewritingRule(
        name="Find active genes only",
        query=MatchQuery(
            bind={"g": PropertyConstraint(properties={"type": "gene", "active": True})}
        ),
        actions=[SetProperties(node_var="g", properties={"marked": True})],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    active_gene = next(n for n in result_graph.nodes.values() if n.node_id == 0)
    inactive_gene = next(n for n in result_graph.nodes.values() if n.node_id == 1)

    assert active_gene.extra["marked"]
    assert "marked" not in inactive_gene.extra


def test_complex_template_expansion():
    graph = create_graph_state(
        [
            {"id": 0, "type": "gene", "name": "BRCA1", "length": 1863},
            {"id": 1, "type": "promoter", "region": "upstream"},
        ],
        [{"source": 1, "target": 0}],
    )

    rule = GraphRewritingRule(
        name="Create complex annotations",
        query=MatchQuery(
            bind={
                "p": PropertyConstraint(properties={"type": "promoter"}),
                "g": PropertyConstraint(properties={"type": "gene"}),
            },
            where_connected=[EdgeConstraint(source_var="p", target_var="g")],
        ),
        actions=[
            AddNode(
                local_name="annotation",
                properties={
                    "type": "annotation",
                    "description": "{{ f\"{p.extra['region']}_controls_{g.extra['name']}_length_{g.extra['length']}\" }}",
                    "promoter_type": "{{p.node_type}}",
                    "target_gene": "{{g.extra['name']}}",
                },
            )
        ],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    annotations = [n for n in result_graph.nodes.values() if n.node_type == "annotation"]
    assert len(annotations) == 1

    annotation = annotations[0]
    assert annotation.extra["description"] == "upstream_controls_BRCA1_length_1863"
    assert annotation.extra["promoter_type"] == "promoter"
    assert annotation.extra["target_gene"] == "BRCA1"


def test_overlapping_matches_deterministic():
    # Graph where same node could be matched by different variables
    graph = create_graph_state([{"id": 0, "type": "gene", "name": "shared"}], [])

    rule = GraphRewritingRule(
        name="Match same node twice",
        query=MatchQuery(
            bind={
                "a": PropertyConstraint(properties={"type": "gene"}),
                "b": PropertyConstraint(properties={"type": "gene"}),
            }
        ),
        actions=[
            SetProperties(node_var="a", properties={"matched_as_a": True}),
            SetProperties(node_var="b", properties={"matched_as_b": True}),
        ],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    # Should have no matches since same node can't be bound to different variables
    gene_node = next(n for n in result_graph.nodes.values() if n.node_id == 0)
    assert "matched_as_a" not in gene_node.extra
    assert "matched_as_b" not in gene_node.extra


def test_action_sequence_with_local_references():
    graph = create_graph_state([{"id": 0, "type": "promoter"}], [])

    rule = GraphRewritingRule(
        name="Create gene and link to promoter",
        query=MatchQuery(bind={"p": PropertyConstraint(properties={"type": "promoter"})}),
        actions=[
            AddNode(local_name="gene", properties={"type": "gene", "name": "new_gene"}),
            AddNode(local_name="terminator", properties={"type": "terminator"}),
            AddEdge(source="p", target="gene"),
            AddEdge(source="gene", target="terminator"),
        ],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    assert len(result_graph.nodes) == 3
    assert len(result_graph.edges) == 2

    # Verify edge connections
    promoter_to_gene = [e for e in result_graph.edges.values() if e.source_id == 0]
    assert len(promoter_to_gene) == 1

    gene_nodes = [n for n in result_graph.nodes.values() if n.node_type == "gene"]
    term_nodes = [n for n in result_graph.nodes.values() if n.node_type == "terminator"]

    gene_to_term = [
        e
        for e in result_graph.edges.values()
        if e.source_id == gene_nodes[0].node_id and e.target_id == term_nodes[0].node_id
    ]
    assert len(gene_to_term) == 1


def test_query_planning_optimization():
    # Create graph with many nodes to test query planning
    nodes_data = [{"id": i, "type": "gene", "category": "A" if i < 5 else "B"} for i in range(100)]
    graph = create_graph_state(nodes_data, [])

    rule = GraphRewritingRule(
        name="Find rare category A genes",
        query=MatchQuery(
            bind={"g": PropertyConstraint(properties={"type": "gene", "category": "A"})}
        ),
        actions=[SetProperties(node_var="g", properties={"found": True})],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    # Should only match 5 category A genes
    found_genes = [n for n in result_graph.nodes.values() if n.extra.get("found")]
    assert len(found_genes) == 5

    # All found genes should be category A
    for gene in found_genes:
        assert gene.extra["category"] == "A"


def test_run_until_stable_complex():
    # Create a chain that should be fully collapsed
    graph = create_graph_state(
        [{"id": i, "type": "intermediate", "value": i} for i in range(5)],
        [{"source": i, "target": i + 1} for i in range(4)],
    )

    rule = GraphRewritingRule(
        name="Merge connected intermediates",
        query=MatchQuery(
            bind={
                "a": PropertyConstraint(properties={"type": "intermediate"}),
                "b": PropertyConstraint(properties={"type": "intermediate"}),
            },
            where_connected=[EdgeConstraint(source_var="a", target_var="b")],
        ),
        actions=[
            AddNode(
                local_name="merged",
                properties={
                    "type": "intermediate",
                    "value": "{{ f\"{a.extra['value']}_{b.extra['value']}\" }}",
                },
            ),
            RewireEdgesTo(old_target_var="a", new_target_var="merged"),
            RewireEdgesFrom(old_source_var="b", new_source_var="merged"),
            DeleteNode(node_var="a"),
            DeleteNode(node_var="b"),
        ],
        run_until_stable=True,
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    # With run_until_stable=True, the rule should keep applying until no matches
    # Starting with chain 0->1->2->3->4, all adjacent nodes get merged repeatedly
    # until only one node remains
    assert len(result_graph.nodes) == 1
    assert len(result_graph.edges) == 0

    node_values = [n.extra["value"] for n in result_graph.nodes.values()]
    assert node_values == ["0_1_2_3_4"]  # All merged into one

    # Verify all nodes are intermediate type
    for node in result_graph.nodes.values():
        assert node.node_type == "intermediate"
        assert "_" in node.extra["value"]  # Should show merging pattern


def test_biological_circuit_transformation():
    # Simulate a biological transformation: DNA -> RNA -> Protein
    graph = create_graph_state(
        [
            {"id": 0, "type": "promoter", "name": "pLac"},
            {"id": 1, "type": "gene", "name": "lacZ", "product": "beta_gal"},
            {"id": 2, "type": "terminator", "name": "T1"},
        ],
        [
            {"source": 0, "target": 1, "content_type": "DNA"},
            {"source": 1, "target": 2, "content_type": "DNA"},
        ],
    )

    rule = GraphRewritingRule(
        name="Expand central dogma",
        query=MatchQuery(bind={"g": PropertyConstraint(properties={"type": "gene"})}),
        actions=[
            AddNode(
                local_name="mrna",
                properties={"type": "RNA", "transcript_of": "{{g.extra['name']}}"},
            ),
            AddNode(
                local_name="protein",
                properties={"type": "protein", "product": "{{g.extra['product']}}"},
            ),
            AddEdge(source="g", target="mrna"),
            AddEdge(source="mrna", target="protein"),
        ],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    # Should have original + RNA + protein
    assert len(result_graph.nodes) == 5

    rna_nodes = [n for n in result_graph.nodes.values() if n.node_type == "RNA"]
    protein_nodes = [n for n in result_graph.nodes.values() if n.node_type == "protein"]

    assert len(rna_nodes) == 1
    assert len(protein_nodes) == 1

    assert rna_nodes[0].extra["transcript_of"] == "lacZ"
    assert protein_nodes[0].extra["product"] == "beta_gal"


def test_no_actions_rule():
    # Rule that only queries but performs no actions (useful for validation)
    graph = create_graph_state([{"id": 0, "type": "gene"}], [])

    rule = GraphRewritingRule(
        name="Query only",
        query=MatchQuery(bind={"g": PropertyConstraint(properties={"type": "gene"})}),
        actions=[],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1

    # Graph should be unchanged
    result_graph = result[0]
    assert len(result_graph.nodes) == 1
    assert list(result_graph.nodes.values())[0].node_type == "gene"


def test_edge_properties_and_content():
    from biocomp.graphengine import Part

    Part(name="uORF1", category="regulatory")
    graph = create_graph_state([{"id": 0, "type": "promoter"}, {"id": 1, "type": "gene"}], [])

    rule = GraphRewritingRule(
        name="Add regulatory edge",
        query=MatchQuery(
            bind={
                "p": PropertyConstraint(properties={"type": "promoter"}),
                "g": PropertyConstraint(properties={"type": "gene"}),
            }
        ),
        actions=[
            # Note: The AddEdge action doesn't currently support content in properties
            # This tests the current limitation and documents expected behavior
            AddEdge(source="p", target="g")
        ],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    assert len(result_graph.edges) == 1
    edge = list(result_graph.edges.values())[0]
    assert edge.source_id == 0
    assert edge.target_id == 1
    assert edge.content == ()  # Default empty content


def test_rule_with_empty_property_constraints():
    graph = create_graph_state(
        [{"id": 0, "type": "gene", "active": True}, {"id": 1, "type": "promoter", "active": False}],
        [],
    )

    rule = GraphRewritingRule(
        name="Match any node",
        query=MatchQuery(bind={"any": PropertyConstraint(properties={})}),
        actions=[SetProperties(node_var="any", properties={"processed": True})],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    # Should match both nodes
    processed_nodes = [n for n in result_graph.nodes.values() if n.extra.get("processed")]
    assert len(processed_nodes) == 2


def test_run_until_stable_batched_non_overlapping():
    """Test that run_until_stable processes all non-overlapping matches in parallel"""
    # Two independent pairs: A->B, C->D
    graph = create_graph_state(
        [
            {"id": 0, "type": "any", "name": "A"},
            {"id": 1, "type": "any", "name": "B"},
            {"id": 2, "type": "any", "name": "C"},
            {"id": 3, "type": "any", "name": "D"},
        ],
        [{"source": 0, "target": 1}, {"source": 2, "target": 3}],
    )

    fuse_rule = GraphRewritingRule(
        name="Fuse connected nodes",
        query=MatchQuery(
            bind={"a": PropertyConstraint(properties={}), "b": PropertyConstraint(properties={})},
            where_connected=[EdgeConstraint(source_var="a", target_var="b")],
        ),
        actions=[
            AddNode(
                local_name="fused",
                properties={
                    "type": "fused",
                    "name": "{{ a.extra['name'] + '+' + b.extra['name'] }}",
                },
            ),
            DeleteNode(node_var="a"),
            DeleteNode(node_var="b"),
        ],
        run_until_stable=True,
    )

    result = apply_rule(fuse_rule, graph)[0]

    assert len(result.nodes) == 2
    node_names = sorted([n.extra["name"] for n in result.nodes.values()])
    assert node_names == ["A+B", "C+D"]  # Jinja2 renders complex template as string


def test_run_until_stable_deterministic_ordering():
    """Test that overlapping matches are processed deterministically by node ID"""
    # Chain A->B->C creates overlapping matches A->B and B->C
    graph = create_graph_state(
        [
            {"id": 0, "type": "any", "name": "A"},
            {"id": 1, "type": "any", "name": "B"},
            {"id": 2, "type": "any", "name": "C"},
        ],
        [{"source": 0, "target": 1}, {"source": 1, "target": 2}],
    )

    fuse_rule = GraphRewritingRule(
        name="Fuse connected nodes",
        query=MatchQuery(
            bind={"a": PropertyConstraint(properties={}), "b": PropertyConstraint(properties={})},
            where_connected=[EdgeConstraint(source_var="a", target_var="b")],
        ),
        actions=[
            AddNode(
                local_name="fused",
                properties={
                    "type": "any",
                    "name": "{{ '(' + a.extra['name'] + '+' + b.extra['name'] + ')' }}",
                },
            ),
            RewireEdgesTo(old_target_var="a", new_target_var="fused"),
            RewireEdgesFrom(old_source_var="b", new_source_var="fused"),
            DeleteNode(node_var="a"),
            DeleteNode(node_var="b"),
        ],
        run_until_stable=True,
    )

    result = apply_rule(fuse_rule, graph)[0]

    # Should always choose A->B first (lower node IDs) then fuse with C
    assert len(result.nodes) == 1
    assert list(result.nodes.values())[0].extra["name"] == "((A+B)+C)"


def test_edge_constraint_none_source_var():
    """Test EdgeConstraint with source_var=None to match edges by target only"""
    graph = create_graph_state(
        [
            {"id": 0, "type": "promoter", "name": "pA"},
            {"id": 1, "type": "gene", "name": "gB"},
            {"id": 2, "type": "terminator", "name": "tC"},
        ],
        [
            {"source": 0, "target": 1, "content_type": "DNA"},
            {"source": 1, "target": 2, "content_type": "RNA"},
            {"source": 2, "target": 1, "content_type": "PRT"},  # feedback loop
        ],
    )

    rule = GraphRewritingRule(
        name="Find all inputs to gene",
        query=MatchQuery(
            bind={"g": PropertyConstraint(properties={"type": "gene"})},
            where_connected=[EdgeConstraint(source_var=None, target_var="g")],
        ),
        actions=[SetProperties(node_var="g", properties={"has_inputs": True})],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    gene_node = next(node for node in result_graph.nodes.values() if node.node_type == "gene")
    assert gene_node.extra["has_inputs"]


def test_edge_constraint_none_target_var():
    """Test EdgeConstraint with target_var=None to match edges by source only"""
    graph = create_graph_state(
        [
            {"id": 0, "type": "promoter", "name": "pA"},
            {"id": 1, "type": "gene", "name": "gB"},
            {"id": 2, "type": "terminator", "name": "tC"},
        ],
        [
            {"source": 0, "target": 1, "content_type": "DNA"},
            {"source": 1, "target": 2, "content_type": "RNA"},
            {"source": 1, "target": 0, "content_type": "PRT"},  # feedback loop
        ],
    )

    rule = GraphRewritingRule(
        name="Find all outputs from gene",
        query=MatchQuery(
            bind={"g": PropertyConstraint(properties={"type": "gene"})},
            where_connected=[EdgeConstraint(source_var="g", target_var=None)],
        ),
        actions=[SetProperties(node_var="g", properties={"has_outputs": True})],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    gene_node = next(node for node in result_graph.nodes.values() if node.node_type == "gene")
    assert gene_node.extra["has_outputs"]


def test_edge_constraint_both_none():
    """Test EdgeConstraint with both source_var=None and target_var=None to match edges by properties only"""
    graph = create_graph_state(
        [
            {"id": 0, "type": "promoter", "name": "pA"},
            {"id": 1, "type": "gene", "name": "gB"},
            {"id": 2, "type": "terminator", "name": "tC"},
        ],
        [
            {"source": 0, "target": 1, "content_type": "DNA"},
            {"source": 1, "target": 2, "content_type": "RNA"},
            {"source": 2, "target": 0, "content_type": "DNA"},  # Same content_type as first edge
        ],
    )

    rule = GraphRewritingRule(
        name="Find DNA edges",
        query=MatchQuery(
            bind={"any_node": PropertyConstraint(properties={})},  # Need at least one node binding
            where_connected=[
                EdgeConstraint(source_var=None, target_var=None, properties={"content_type": "DNA"})
            ],
        ),
        actions=[SetProperties(node_var="any_node", properties={"has_dna_edges": True})],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    # Should match because there are DNA edges in the graph
    marked_nodes = [node for node in result_graph.nodes.values() if node.extra.get("has_dna_edges")]
    assert len(marked_nodes) > 0


def test_edge_constraint_none_with_negative():
    """Test EdgeConstraint with None values in where_not_connected"""
    graph = create_graph_state(
        [
            {"id": 0, "type": "isolated", "name": "A"},
            {"id": 1, "type": "connected", "name": "B"},
            {"id": 2, "type": "connected", "name": "C"},
        ],
        [
            {"source": 1, "target": 2, "content_type": "DNA"},
        ],
    )

    rule = GraphRewritingRule(
        name="Find isolated nodes",
        query=MatchQuery(
            bind={"n": PropertyConstraint(properties={})},
            where_not_connected=[
                EdgeConstraint(source_var="n", target_var=None)
            ],  # No outgoing edges
        ),
        actions=[SetProperties(node_var="n", properties={"is_isolated": True})],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    # Should match nodes 0 and 2 (no outgoing edges)
    isolated_nodes = [node for node in result_graph.nodes.values() if node.extra.get("is_isolated")]
    isolated_ids = {node.node_id for node in isolated_nodes}
    assert isolated_ids == {0, 2}


def test_bind_edges_with_none_values():
    """Test binding edges using EdgeConstraint with None values in bind_edges"""
    graph = create_graph_state(
        [
            {"id": 0, "type": "promoter", "name": "pA"},
            {"id": 1, "type": "gene", "name": "gB"},
            {"id": 2, "type": "terminator", "name": "tC"},
        ],
        [],
    )

    # Add edges manually with content_embedding_names
    from biocomp.graphengine import GraphEdge

    edges = [
        GraphEdge(
            source_id=0,
            target_id=1,
            from_output_slot=0,
            to_input_slot=0,
            content=(),
            content_type="DNA",
            content_embedding_names={"strength": ("strong",)},
        ),
        GraphEdge(
            source_id=1,
            target_id=2,
            from_output_slot=0,
            to_input_slot=0,
            content=(),
            content_type="RNA",
            content_embedding_names={"strength": ("weak",)},
        ),
    ]
    graph.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e for e in edges
    }

    rule = GraphRewritingRule(
        name="Find strong edges and mark their targets",
        query=MatchQuery(
            bind_edges={
                "strong_edge": EdgeConstraint(
                    source_var=None, target_var=None, properties={"strength": ("strong",)}
                )
            },
        ),
        actions=[
            # Use a template to access the edge target
            AddNode(
                local_name="marker",
                properties={"type": "marker", "marks_target": "{{strong_edge.target_id}}"},
            )
        ],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    markers = [node for node in result_graph.nodes.values() if node.node_type == "marker"]
    assert len(markers) == 1
    assert markers[0].extra["marks_target"] == 1  # Target of the strong edge


def test_validation_allows_none_values():
    """Test that validation correctly allows None values in EdgeConstraint"""

    # This should not raise an exception
    valid_query = MatchQuery(
        bind={"n": PropertyConstraint(properties={"type": "gene"})},
        where_connected=[EdgeConstraint(source_var=None, target_var="n")],
        bind_edges={
            "e": EdgeConstraint(source_var=None, target_var=None, properties={"type": "regulatory"})
        },
    )

    # This should also be valid
    valid_query2 = MatchQuery(
        bind={"n": PropertyConstraint(properties={"type": "gene"})},
        where_not_connected=[EdgeConstraint(source_var="n", target_var=None)],
    )

    # Both should create successfully without validation errors
    assert valid_query is not None
    assert valid_query2 is not None


def test_edge_constraint_none_with_properties():
    """Test EdgeConstraint with None endpoint and specific properties"""
    graph = create_graph_state(
        [
            {"id": 0, "type": "node", "name": "A"},
            {"id": 1, "type": "node", "name": "B"},
            {"id": 2, "type": "node", "name": "C"},
        ],
        [],
    )

    # Add edges manually with content_embedding_names to test property matching
    from biocomp.graphengine import GraphEdge

    edges = [
        GraphEdge(
            source_id=0,
            target_id=1,
            from_output_slot=0,
            to_input_slot=0,
            content=(),
            content_type="DNA",
            content_embedding_names={"edge_type": ("regulatory",)},
        ),
        GraphEdge(
            source_id=1,
            target_id=2,
            from_output_slot=0,
            to_input_slot=0,
            content=(),
            content_type="RNA",
            content_embedding_names={"edge_type": ("structural",)},
        ),
        GraphEdge(
            source_id=2,
            target_id=0,
            from_output_slot=0,
            to_input_slot=0,
            content=(),
            content_type="DNA",
            content_embedding_names={"edge_type": ("regulatory",)},
        ),
    ]
    graph.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e for e in edges
    }

    rule = GraphRewritingRule(
        name="Find nodes connected by regulatory edges",
        query=MatchQuery(
            bind={"n": PropertyConstraint(properties={"type": "node"})},
            where_connected=[
                EdgeConstraint(
                    source_var="n", target_var=None, properties={"edge_type": ("regulatory",)}
                )
            ],
        ),
        actions=[SetProperties(node_var="n", properties={"has_regulatory_output": True})],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    # Should match nodes 0 and 2 (they have outgoing regulatory edges)
    regulatory_nodes = [
        node for node in result_graph.nodes.values() if node.extra.get("has_regulatory_output")
    ]
    regulatory_ids = {node.node_id for node in regulatory_nodes}
    assert regulatory_ids == {0, 2}


def test_edge_constraint_contains_basic():
    """Test EdgeConstraint contains functionality with basic part matching"""
    from biocomp.graphengine import Part, GraphEdge

    graph = create_graph_state(
        [
            {"id": 0, "type": "promoter", "name": "pA"},
            {"id": 1, "type": "gene", "name": "gB"},
            {"id": 2, "type": "terminator", "name": "tC"},
        ],
        [],
    )

    # Add edges with specific part content
    part1 = Part(name="uORF1", category="regulatory")
    part2 = Part(name="uORF2", category="regulatory")
    part3 = Part(name="RBS1", category="ribosome_binding")

    edges = [
        GraphEdge(
            source_id=0,
            target_id=1,
            from_output_slot=0,
            to_input_slot=0,
            content=(part1, part2),
            content_type="DNA",
        ),
        GraphEdge(
            source_id=1,
            target_id=2,
            from_output_slot=0,
            to_input_slot=0,
            content=(part3,),
            content_type="RNA",
        ),
        GraphEdge(
            source_id=0,
            target_id=2,
            from_output_slot=0,
            to_input_slot=0,
            content=(part1, part3),
            content_type="DNA",
        ),
    ]
    graph.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e for e in edges
    }

    rule = GraphRewritingRule(
        name="Find edges containing uORF1",
        query=MatchQuery(
            bind={"n": PropertyConstraint(properties={"type": "promoter"})},
            where_connected=[EdgeConstraint(source_var="n", target_var=None, contains=["uORF1"])],
        ),
        actions=[SetProperties(node_var="n", properties={"has_uorf1_edges": True})],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    # Promoter should be marked because it has outgoing edges containing uORF1
    promoter_node = next(
        node for node in result_graph.nodes.values() if node.node_type == "promoter"
    )
    assert promoter_node.extra["has_uorf1_edges"]


def test_edge_constraint_contains_multiple_parts():
    """Test EdgeConstraint contains with multiple required parts"""
    from biocomp.graphengine import Part, GraphEdge

    graph = create_graph_state(
        [
            {"id": 0, "type": "node", "name": "A"},
            {"id": 1, "type": "node", "name": "B"},
            {"id": 2, "type": "node", "name": "C"},
        ],
        [],
    )

    # Create parts
    part1 = Part(name="uORF1", category="regulatory")
    part2 = Part(name="uORF2", category="regulatory")
    part3 = Part(name="RBS1", category="ribosome_binding")

    edges = [
        GraphEdge(
            source_id=0,
            target_id=1,
            from_output_slot=0,
            to_input_slot=0,
            content=(part1, part2, part3),
            content_type="DNA",
        ),  # Has all three
        GraphEdge(
            source_id=1,
            target_id=2,
            from_output_slot=0,
            to_input_slot=0,
            content=(part1, part3),
            content_type="DNA",
        ),  # Missing part2
        GraphEdge(
            source_id=0,
            target_id=2,
            from_output_slot=0,
            to_input_slot=0,
            content=(part2,),
            content_type="DNA",
        ),  # Only has part2
    ]
    graph.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e for e in edges
    }

    rule = GraphRewritingRule(
        name="Find edges containing both uORF1 and RBS1",
        query=MatchQuery(
            bind={"n": PropertyConstraint(properties={"type": "node"})},
            where_connected=[
                EdgeConstraint(
                    source_var="n",
                    target_var=None,
                    contains=["uORF1", "RBS1"],  # Both required
                )
            ],
        ),
        actions=[SetProperties(node_var="n", properties={"has_both_parts": True})],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    # Only nodes with edges containing both parts should be marked
    marked_nodes = [
        node for node in result_graph.nodes.values() if node.extra.get("has_both_parts")
    ]
    marked_ids = {node.node_id for node in marked_nodes}
    assert marked_ids == {
        0,
        1,
    }  # Node 0 has edge with all parts, node 1 has edge with both required


def test_edge_constraint_contains_no_match():
    """Test EdgeConstraint contains when no edges match the requirements"""
    from biocomp.graphengine import Part, GraphEdge

    graph = create_graph_state(
        [
            {"id": 0, "type": "node", "name": "A"},
            {"id": 1, "type": "node", "name": "B"},
        ],
        [],
    )

    part1 = Part(name="RBS1", category="ribosome_binding")

    edges = [
        GraphEdge(
            source_id=0,
            target_id=1,
            from_output_slot=0,
            to_input_slot=0,
            content=(part1,),
            content_type="DNA",
        ),
    ]
    graph.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e for e in edges
    }

    rule = GraphRewritingRule(
        name="Find edges containing non-existent part",
        query=MatchQuery(
            bind={"n": PropertyConstraint(properties={"type": "node"})},
            where_connected=[
                EdgeConstraint(source_var="n", target_var=None, contains=["uORF_nonexistent"])
            ],
        ),
        actions=[SetProperties(node_var="n", properties={"found": True})],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    # No nodes should be marked since no edges contain the required part
    marked_nodes = [node for node in result_graph.nodes.values() if node.extra.get("found")]
    assert len(marked_nodes) == 0


def test_edge_constraint_contains_with_none_endpoints():
    """Test contains functionality combined with None source/target variables"""
    from biocomp.graphengine import Part, GraphEdge

    graph = create_graph_state(
        [
            {"id": 0, "type": "promoter", "name": "pA"},
            {"id": 1, "type": "gene", "name": "gB"},
            {"id": 2, "type": "terminator", "name": "tC"},
        ],
        [],
    )

    part1 = Part(name="uORF1", category="regulatory")
    part2 = Part(name="RBS1", category="ribosome_binding")

    edges = [
        GraphEdge(
            source_id=0,
            target_id=1,
            from_output_slot=0,
            to_input_slot=0,
            content=(part1,),
            content_type="DNA",
        ),
        GraphEdge(
            source_id=1,
            target_id=2,
            from_output_slot=0,
            to_input_slot=0,
            content=(part2,),
            content_type="RNA",
        ),
        GraphEdge(
            source_id=2,
            target_id=0,
            from_output_slot=0,
            to_input_slot=0,
            content=(part1, part2),
            content_type="DNA",
        ),
    ]
    graph.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e for e in edges
    }

    rule = GraphRewritingRule(
        name="Find any edges containing uORF1",
        query=MatchQuery(
            bind={"any_node": PropertyConstraint(properties={})},  # Match any node
            where_connected=[EdgeConstraint(source_var=None, target_var=None, contains=["uORF1"])],
        ),
        actions=[SetProperties(node_var="any_node", properties={"graph_has_uorf1": True})],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    # Should match because there are edges containing uORF1 in the graph
    marked_nodes = [
        node for node in result_graph.nodes.values() if node.extra.get("graph_has_uorf1")
    ]
    assert len(marked_nodes) > 0


def test_edge_constraint_contains_empty_list():
    """Test EdgeConstraint contains with empty list (should match all edges)"""
    from biocomp.graphengine import Part, GraphEdge

    graph = create_graph_state(
        [
            {"id": 0, "type": "node", "name": "A"},
            {"id": 1, "type": "node", "name": "B"},
        ],
        [],
    )

    part1 = Part(name="RBS1", category="ribosome_binding")

    edges = [
        GraphEdge(
            source_id=0,
            target_id=1,
            from_output_slot=0,
            to_input_slot=0,
            content=(part1,),
            content_type="DNA",
        ),
    ]
    graph.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e for e in edges
    }

    rule = GraphRewritingRule(
        name="Find edges with empty contains list",
        query=MatchQuery(
            bind={"n": PropertyConstraint(properties={"type": "node"})},
            where_connected=[
                EdgeConstraint(
                    source_var="n",
                    target_var=None,
                    contains=[],  # Empty list should match all edges
                )
            ],
        ),
        actions=[SetProperties(node_var="n", properties={"has_edges": True})],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    # Node 0 should be marked since it has outgoing edges and empty list matches all
    marked_nodes = [node for node in result_graph.nodes.values() if node.extra.get("has_edges")]
    marked_ids = {node.node_id for node in marked_nodes}
    assert 0 in marked_ids


def test_edge_constraint_contains_bind_edges():
    """Test contains functionality with bind_edges"""
    from biocomp.graphengine import Part, GraphEdge

    graph = create_graph_state(
        [
            {"id": 0, "type": "promoter", "name": "pA"},
            {"id": 1, "type": "gene", "name": "gB"},
        ],
        [],
    )

    part1 = Part(name="uORF1", category="regulatory")
    part2 = Part(name="uORF2", category="regulatory")

    edges = [
        GraphEdge(
            source_id=0,
            target_id=1,
            from_output_slot=0,
            to_input_slot=0,
            content=(part1, part2),
            content_type="DNA",
        ),
    ]
    graph.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e for e in edges
    }

    rule = GraphRewritingRule(
        name="Bind edges containing specific parts",
        query=MatchQuery(
            bind_edges={
                "regulatory_edge": EdgeConstraint(
                    source_var=None, target_var=None, contains=["uORF1"]
                )
            },
        ),
        actions=[
            AddNode(
                local_name="marker",
                properties={
                    "type": "marker",
                    "edge_source": "{{regulatory_edge.source_id}}",
                    "edge_target": "{{regulatory_edge.target_id}}",
                },
            )
        ],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    markers = [node for node in result_graph.nodes.values() if node.node_type == "marker"]
    assert len(markers) == 1
    assert markers[0].extra["edge_source"] == 0
    assert markers[0].extra["edge_target"] == 1


def test_automatic_endpoint_binding_basic():
    """Test basic automatic endpoint binding for bound edges"""
    from biocomp.graphengine import Part, GraphEdge

    graph = create_graph_state(
        [
            {"id": 0, "type": "promoter", "name": "pA"},
            {"id": 1, "type": "gene", "name": "gB"},
            {"id": 2, "type": "terminator", "name": "tC"},
        ],
        [],
    )

    part1 = Part(name="uORF1", category="regulatory")

    edges = [
        GraphEdge(
            source_id=0,
            target_id=1,
            from_output_slot=0,
            to_input_slot=0,
            content=(part1,),
            content_type="DNA",
        ),
    ]
    graph.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e for e in edges
    }

    rule = GraphRewritingRule(
        name="Test automatic endpoint binding",
        query=MatchQuery(
            bind_edges={"test_edge": EdgeConstraint(contains=["uORF1"])},
        ),
        actions=[
            # Use the automatically bound endpoint nodes
            SetProperties(node_var="test_edge_source", properties={"is_source": True}),
            SetProperties(node_var="test_edge_target", properties={"is_target": True}),
        ],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    # Check that the endpoint nodes were correctly identified and modified
    source_nodes = [node for node in result_graph.nodes.values() if node.extra.get("is_source")]
    target_nodes = [node for node in result_graph.nodes.values() if node.extra.get("is_target")]

    assert len(source_nodes) == 1
    assert len(target_nodes) == 1
    assert source_nodes[0].node_id == 0  # Promoter
    assert target_nodes[0].node_id == 1  # Gene


def test_automatic_endpoint_binding_multiple_edges():
    """Test automatic endpoint binding with multiple bound edges"""
    from biocomp.graphengine import Part, GraphEdge

    graph = create_graph_state(
        [
            {"id": 0, "type": "promoter", "name": "pA"},
            {"id": 1, "type": "gene", "name": "gB"},
            {"id": 2, "type": "terminator", "name": "tC"},
        ],
        [],
    )

    part1 = Part(name="uORF1", category="regulatory")
    part2 = Part(name="RBS1", category="ribosome_binding")

    edges = [
        GraphEdge(
            source_id=0,
            target_id=1,
            from_output_slot=0,
            to_input_slot=0,
            content=(part1,),
            content_type="DNA",
        ),
        GraphEdge(
            source_id=1,
            target_id=2,
            from_output_slot=0,
            to_input_slot=0,
            content=(part2,),
            content_type="RNA",
        ),
    ]
    graph.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e for e in edges
    }

    rule = GraphRewritingRule(
        name="Test multiple edge endpoint binding",
        query=MatchQuery(
            bind_edges={
                "dna_edge": EdgeConstraint(contains=["uORF1"]),
                "rna_edge": EdgeConstraint(contains=["RBS1"]),
            },
        ),
        actions=[
            AddNode(
                local_name="marker",
                properties={
                    "type": "marker",
                    "dna_source": "{{dna_edge_source.node_id}}",
                    "dna_target": "{{dna_edge_target.node_id}}",
                    "rna_source": "{{rna_edge_source.node_id}}",
                    "rna_target": "{{rna_edge_target.node_id}}",
                },
            ),
        ],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    markers = [node for node in result_graph.nodes.values() if node.node_type == "marker"]
    assert len(markers) == 1
    marker = markers[0]

    assert marker.extra["dna_source"] == 0  # Promoter to gene
    assert marker.extra["dna_target"] == 1
    assert marker.extra["rna_source"] == 1  # Gene to terminator
    assert marker.extra["rna_target"] == 2


def test_automatic_endpoint_binding_disabled():
    """Test that automatic endpoint binding can be disabled"""
    from biocomp.graphengine import Part, GraphEdge

    graph = create_graph_state(
        [
            {"id": 0, "type": "promoter", "name": "pA"},
            {"id": 1, "type": "gene", "name": "gB"},
        ],
        [],
    )

    part1 = Part(name="uORF1", category="regulatory")

    edges = [
        GraphEdge(
            source_id=0,
            target_id=1,
            from_output_slot=0,
            to_input_slot=0,
            content=(part1,),
            content_type="DNA",
        ),
    ]
    graph.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e for e in edges
    }

    rule = GraphRewritingRule(
        name="Test disabled endpoint binding",
        query=MatchQuery(
            bind_edges={"test_edge": EdgeConstraint(contains=["uORF1"], bind_endpoints=False)},
        ),
        actions=[
            AddNode(local_name="marker", properties={"type": "marker"}),
        ],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result[0]

    # The automatic endpoint nodes should not be available since bind_endpoints=False
    # This test passes if no error is thrown and the rule works without the endpoint bindings


def test_automatic_endpoint_binding_conflict_validation():
    """Test that conflicts between auto-generated names and manual bindings are caught"""

    # This should raise a validation error due to naming conflict
    try:
        GraphRewritingRule(
            name="Test naming conflict",
            query=MatchQuery(
                bind={"test_edge_source": PropertyConstraint(properties={"type": "promoter"})},
                bind_edges={
                    "test_edge": EdgeConstraint(contains=["uORF1"])
                },  # bind_endpoints=True by default
            ),
            actions=[AddNode(local_name="marker", properties={"type": "marker"})],
        )
        raise AssertionError("Should have raised a validation error due to naming conflict")
    except ValueError as e:
        assert "conflicts with manually bound node" in str(e)


def test_automatic_endpoint_binding_with_none_constraints():
    """Test automatic endpoint binding combined with None source/target constraints"""
    from biocomp.graphengine import Part, GraphEdge

    graph = create_graph_state(
        [
            {"id": 0, "type": "promoter", "name": "pA"},
            {"id": 1, "type": "gene", "name": "gB"},
            {"id": 2, "type": "terminator", "name": "tC"},
        ],
        [],
    )

    part1 = Part(name="uORF1", category="regulatory")

    edges = [
        GraphEdge(
            source_id=0,
            target_id=1,
            from_output_slot=0,
            to_input_slot=0,
            content=(part1,),
            content_type="DNA",
        ),
        GraphEdge(
            source_id=1,
            target_id=2,
            from_output_slot=0,
            to_input_slot=0,
            content=(part1,),
            content_type="RNA",
        ),
    ]
    graph.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e for e in edges
    }

    rule = GraphRewritingRule(
        name="Test endpoint binding with None constraints",
        query=MatchQuery(
            bind_edges={
                "any_uorf_edge": EdgeConstraint(
                    source_var=None, target_var=None, contains=["uORF1"]
                )
            },
        ),
        actions=[
            SetProperties(node_var="any_uorf_edge_source", properties={"has_uorf_output": True}),
            SetProperties(node_var="any_uorf_edge_target", properties={"has_uorf_input": True}),
        ],
    )

    result = apply_rule(rule, graph)
    assert len(result) >= 1  # Should match at least one edge
    result_graph = result[0]

    # Check that at least one source and target were marked
    sources = [node for node in result_graph.nodes.values() if node.extra.get("has_uorf_output")]
    targets = [node for node in result_graph.nodes.values() if node.extra.get("has_uorf_input")]

    assert len(sources) > 0
    assert len(targets) > 0


def test_edit_edge_basic():
    """Test basic EditEdge action functionality"""
    from biocomp.graphengine import Part, GraphEdge

    graph = create_graph_state(
        [
            {"id": 0, "type": "promoter", "name": "pA"},
            {"id": 1, "type": "gene", "name": "gB"},
            {"id": 2, "type": "terminator", "name": "tC"},
        ],
        [],
    )

    part1 = Part(name="uORF1", category="regulatory")

    edges = [
        GraphEdge(
            source_id=0,
            target_id=1,
            from_output_slot=0,
            to_input_slot=0,
            content=(part1,),
            content_type="DNA",
        ),
    ]
    graph.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e for e in edges
    }

    rule = GraphRewritingRule(
        name="Edit edge properties",
        query=MatchQuery(
            bind_edges={"test_edge": EdgeConstraint(contains=["uORF1"])},
        ),
        actions=[
            EditEdge(edge_var="test_edge", properties={"edited": True, "new_prop": "test_value"}),
        ],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    # Check that the edge was edited
    assert len(result_graph.edges) == 1
    edge = list(result_graph.edges.values())[0]
    assert edge.extra.get("edited")
    assert edge.extra.get("new_prop") == "test_value"
    # Original endpoints should be preserved
    assert edge.source_id == 0
    assert edge.target_id == 1


def test_edit_edge_change_endpoints():
    """Test EditEdge action changing source and target"""
    from biocomp.graphengine import Part, GraphEdge

    graph = create_graph_state(
        [
            {"id": 0, "type": "promoter", "name": "pA"},
            {"id": 1, "type": "gene", "name": "gB"},
            {"id": 2, "type": "terminator", "name": "tC"},
        ],
        [],
    )

    part1 = Part(name="uORF1", category="regulatory")

    edges = [
        GraphEdge(
            source_id=0,
            target_id=1,
            from_output_slot=0,
            to_input_slot=0,
            content=(part1,),
            content_type="DNA",
        ),
    ]
    graph.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e for e in edges
    }

    rule = GraphRewritingRule(
        name="Rewire edge endpoints",
        query=MatchQuery(
            bind={"terminator": PropertyConstraint(properties={"type": "terminator"})},
            bind_edges={"test_edge": EdgeConstraint(contains=["uORF1"])},
        ),
        actions=[
            EditEdge(
                edge_var="test_edge",
                target_var="terminator",  # Change target from gene to terminator
            ),
        ],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    # Check that the edge was rewired
    assert len(result_graph.edges) == 1
    edge = list(result_graph.edges.values())[0]
    assert edge.source_id == 0  # Original source preserved
    assert edge.target_id == 2  # Target changed to terminator
    # Content should be preserved
    assert len(edge.content) == 1
    assert edge.content[0].name == "uORF1"


def test_edit_edge_change_content():
    """Test EditEdge action changing edge content"""
    from biocomp.graphengine import Part, GraphEdge

    graph = create_graph_state(
        [
            {"id": 0, "type": "promoter", "name": "pA"},
            {"id": 1, "type": "gene", "name": "gB"},
        ],
        [],
    )

    part1 = Part(name="uORF1", category="regulatory")

    edges = [
        GraphEdge(
            source_id=0,
            target_id=1,
            from_output_slot=0,
            to_input_slot=0,
            content=(part1,),
            content_type="DNA",
        ),
    ]
    graph.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e for e in edges
    }

    rule = GraphRewritingRule(
        name="Change edge content",
        query=MatchQuery(
            bind_edges={"test_edge": EdgeConstraint(contains=["uORF1"])},
        ),
        actions=[
            EditEdge(
                edge_var="test_edge",
                content=["RBS1", "uORF2"],  # Replace content
            ),
        ],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    # Check that the edge content was changed
    assert len(result_graph.edges) == 1
    edge = list(result_graph.edges.values())[0]
    content_names = [part.name for part in edge.content]
    assert "RBS1" in content_names
    assert "uORF2" in content_names
    assert "uORF1" not in content_names  # Original content replaced


def test_edit_edge_comprehensive():
    """Test EditEdge action changing all possible fields"""
    from biocomp.graphengine import Part, GraphEdge

    graph = create_graph_state(
        [
            {"id": 0, "type": "promoter", "name": "pA"},
            {"id": 1, "type": "gene", "name": "gB"},
            {"id": 2, "type": "terminator", "name": "tC"},
            {"id": 3, "type": "new_node", "name": "nD"},
        ],
        [],
    )

    part1 = Part(name="uORF1", category="regulatory")

    edges = [
        GraphEdge(
            source_id=0,
            target_id=1,
            from_output_slot=0,
            to_input_slot=0,
            content=(part1,),
            content_type="DNA",
        ),
    ]
    graph.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e for e in edges
    }

    rule = GraphRewritingRule(
        name="Comprehensive edge edit",
        query=MatchQuery(
            bind={
                "terminator": PropertyConstraint(properties={"type": "terminator"}),
                "new_node": PropertyConstraint(properties={"type": "new_node"}),
            },
            bind_edges={"test_edge": EdgeConstraint(contains=["uORF1"])},
        ),
        actions=[
            EditEdge(
                edge_var="test_edge",
                source_var="terminator",  # Change source
                target_var="new_node",  # Change target
                properties={"fully_edited": True, "iteration": 1},  # Add properties
                content=["RBS1", "new_part"],  # Change content
            ),
        ],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    # Check that all aspects of the edge were changed
    assert len(result_graph.edges) == 1
    edge = list(result_graph.edges.values())[0]

    # Check endpoints
    assert edge.source_id == 2  # Terminator
    assert edge.target_id == 3  # New node

    # Check properties
    assert edge.extra.get("fully_edited")
    assert edge.extra.get("iteration") == 1

    # Check content
    content_names = [part.name for part in edge.content]
    assert "RBS1" in content_names
    assert "new_part" in content_names
    assert "uORF1" not in content_names


def test_edit_edge_partial_updates():
    """Test EditEdge action with partial updates (preserving some fields)"""
    from biocomp.graphengine import Part, GraphEdge

    graph = create_graph_state(
        [
            {"id": 0, "type": "promoter", "name": "pA"},
            {"id": 1, "type": "gene", "name": "gB"},
        ],
        [],
    )

    part1 = Part(name="uORF1", category="regulatory")
    part2 = Part(name="RBS1", category="ribosome_binding")

    edges = [
        GraphEdge(
            source_id=0,
            target_id=1,
            from_output_slot=0,
            to_input_slot=0,
            content=(part1, part2),
            content_type="DNA",
            extra={"original_prop": "keep_me"},
        ),
    ]
    graph.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e for e in edges
    }

    rule = GraphRewritingRule(
        name="Partial edge edit",
        query=MatchQuery(
            bind_edges={"test_edge": EdgeConstraint(contains=["uORF1"])},
        ),
        actions=[
            EditEdge(
                edge_var="test_edge",
                properties={"new_prop": "added"},  # Only add properties, keep everything else
            ),
        ],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    # Check that partial updates worked correctly
    assert len(result_graph.edges) == 1
    edge = list(result_graph.edges.values())[0]

    # Endpoints should be preserved
    assert edge.source_id == 0
    assert edge.target_id == 1

    # Content should be preserved
    content_names = [part.name for part in edge.content]
    assert "uORF1" in content_names
    assert "RBS1" in content_names

    # Original properties should be preserved
    assert edge.extra.get("original_prop") == "keep_me"
    # New properties should be added
    assert edge.extra.get("new_prop") == "added"


def test_edit_edge_with_templates():
    """Test EditEdge action using template variables"""
    from biocomp.graphengine import Part, GraphEdge

    graph = create_graph_state(
        [
            {"id": 0, "type": "promoter", "name": "pA", "strength": 0.8},
            {"id": 1, "type": "gene", "name": "gB", "expression": 1.2},
        ],
        [],
    )

    part1 = Part(name="uORF1", category="regulatory")

    edges = [
        GraphEdge(
            source_id=0,
            target_id=1,
            from_output_slot=0,
            to_input_slot=0,
            content=(part1,),
            content_type="DNA",
        ),
    ]
    graph.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e for e in edges
    }

    rule = GraphRewritingRule(
        name="Template-based edge edit",
        query=MatchQuery(
            bind_edges={"test_edge": EdgeConstraint(contains=["uORF1"])},
        ),
        actions=[
            EditEdge(
                edge_var="test_edge",
                properties={
                    "source_name": "{{test_edge_source.extra['name']}}",
                    "target_name": "{{test_edge_target.extra['name']}}",
                    "combined_values": "{{test_edge_source.extra['strength'] + test_edge_target.extra['expression']}}",
                },
            ),
        ],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    # Check that template variables were correctly resolved
    assert len(result_graph.edges) == 1
    edge = list(result_graph.edges.values())[0]

    assert edge.extra.get("source_name") == "pA"
    assert edge.extra.get("target_name") == "gB"
    assert (
        edge.extra.get("combined_values") == 2.0
    )  # Simple expression: Python eval does actual addition


def test_edit_edge_nonexistent_variable():
    """Test EditEdge action with non-existent edge variable should raise error"""
    from biocomp.graphengine import Part, GraphEdge

    graph = create_graph_state(
        [
            {"id": 0, "type": "promoter", "name": "pA"},
            {"id": 1, "type": "gene", "name": "gB"},
        ],
        [],
    )

    part1 = Part(name="uORF1", category="regulatory")

    edges = [
        GraphEdge(
            source_id=0,
            target_id=1,
            from_output_slot=0,
            to_input_slot=0,
            content=(part1,),
            content_type="DNA",
        ),
    ]
    graph.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e for e in edges
    }

    rule = GraphRewritingRule(
        name="Edit nonexistent edge",
        query=MatchQuery(
            bind_edges={"test_edge": EdgeConstraint(contains=["uORF1"])},
        ),
        actions=[
            EditEdge(
                edge_var="nonexistent_edge",  # This edge variable doesn't exist
                properties={"should_fail": True},
            ),
        ],
    )

    # Should raise an error when trying to apply the rule
    try:
        apply_rule(rule, graph)
        raise AssertionError("Should have raised an error for nonexistent edge variable")
    except (KeyError, ValueError):
        pass  # Expected behavior


def test_copy_edge_basic():
    """Test basic CopyEdge action functionality"""
    from biocomp.graphengine import Part, GraphEdge

    graph = create_graph_state(
        [
            {"id": 0, "type": "promoter", "name": "pA"},
            {"id": 1, "type": "gene", "name": "gB"},
            {"id": 2, "type": "terminator", "name": "tC"},
            {"id": 3, "type": "output", "name": "oD"},
        ],
        [],
    )

    part1 = Part(name="uORF1", category="regulatory")

    edges = [
        GraphEdge(
            source_id=0,
            target_id=1,
            from_output_slot=0,
            to_input_slot=0,
            content=(part1,),
            content_type="DNA",
            extra={"strength": 0.8, "original": True},
        ),
    ]
    graph.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e for e in edges
    }

    rule = GraphRewritingRule(
        name="Copy edge to new endpoints",
        query=MatchQuery(
            bind={
                "terminator": PropertyConstraint(properties={"type": "terminator"}),
                "output": PropertyConstraint(properties={"type": "output"}),
            },
            bind_edges={"original_edge": EdgeConstraint(contains=["uORF1"])},
        ),
        actions=[
            CopyEdge(source_edge_var="original_edge", source_var="terminator", target_var="output"),
        ],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    # Should have original edge + new copied edge
    assert len(result_graph.edges) == 2

    # Find the new edge (should be terminator->output)
    new_edge = next(
        edge for edge in result_graph.edges.values() if edge.source_id == 2 and edge.target_id == 3
    )

    # Check that all properties were copied
    assert new_edge.content_type == "DNA"
    assert len(new_edge.content) == 1
    assert new_edge.content[0].name == "uORF1"
    assert new_edge.extra.get("strength") == 0.8
    assert new_edge.extra.get("original")


def test_copy_edge_with_property_overrides():
    """Test CopyEdge with additional and override properties"""
    from biocomp.graphengine import Part, GraphEdge

    graph = create_graph_state(
        [
            {"id": 0, "type": "source", "name": "A"},
            {"id": 1, "type": "target", "name": "B"},
            {"id": 2, "type": "new_source", "name": "C"},
            {"id": 3, "type": "new_target", "name": "D"},
        ],
        [],
    )

    part1 = Part(name="RBS1", category="ribosome_binding")

    edges = [
        GraphEdge(
            source_id=0,
            target_id=1,
            from_output_slot=0,
            to_input_slot=0,
            content=(part1,),
            content_type="RNA",
            extra={"strength": 0.5, "category": "original", "keep_me": True},
        ),
    ]
    graph.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e for e in edges
    }

    rule = GraphRewritingRule(
        name="Copy edge with overrides",
        query=MatchQuery(
            bind={
                "new_src": PropertyConstraint(properties={"type": "new_source"}),
                "new_tgt": PropertyConstraint(properties={"type": "new_target"}),
            },
            bind_edges={"template_edge": EdgeConstraint(contains=["RBS1"])},
        ),
        actions=[
            CopyEdge(
                source_edge_var="template_edge",
                source_var="new_src",
                target_var="new_tgt",
                properties={
                    "strength": 1.2,  # Override existing property
                    "category": "copied",  # Override existing property
                    "is_copy": True,  # Add new property
                    # keep_me should be preserved from original
                },
            ),
        ],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    # Should have original edge + new copied edge
    assert len(result_graph.edges) == 2

    # Find the new edge
    new_edge = next(
        edge for edge in result_graph.edges.values() if edge.source_id == 2 and edge.target_id == 3
    )

    # Check copied properties
    assert new_edge.content_type == "RNA"  # Copied
    assert new_edge.content[0].name == "RBS1"  # Copied

    # Check property overrides and additions
    assert new_edge.extra.get("strength") == 1.2  # Overridden
    assert new_edge.extra.get("category") == "copied"  # Overridden
    assert new_edge.extra.get("is_copy")  # Added
    assert new_edge.extra.get("keep_me")  # Preserved from original


def test_copy_edge_with_content_override():
    """Test CopyEdge with content override"""
    from biocomp.graphengine import Part, GraphEdge

    graph = create_graph_state(
        [
            {"id": 0, "type": "node", "name": "A"},
            {"id": 1, "type": "node", "name": "B"},
            {"id": 2, "type": "node", "name": "C"},
            {"id": 3, "type": "node", "name": "D"},
        ],
        [],
    )

    part1 = Part(name="original_part", category="regulatory")

    edges = [
        GraphEdge(
            source_id=0,
            target_id=1,
            from_output_slot=0,
            to_input_slot=0,
            content=(part1,),
            content_type="DNA",
            extra={"version": 1},
        ),
    ]
    graph.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e for e in edges
    }

    rule = GraphRewritingRule(
        name="Copy edge with new content",
        query=MatchQuery(
            bind={
                "node_c": PropertyConstraint(properties={"name": "C"}),
                "node_d": PropertyConstraint(properties={"name": "D"}),
            },
            bind_edges={"source_edge": EdgeConstraint(source_var=None, target_var=None)},
        ),
        actions=[
            CopyEdge(
                source_edge_var="source_edge",
                source_var="node_c",
                target_var="node_d",
                content=["new_part1", "new_part2"],  # Override content
                content_type="RNA",  # Override content_type
                properties={"version": 2},  # Override property
            ),
        ],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    # Should have original edge + new copied edge
    assert len(result_graph.edges) == 2

    # Find the new edge
    new_edge = next(
        edge for edge in result_graph.edges.values() if edge.source_id == 2 and edge.target_id == 3
    )

    # Check overridden content and content_type
    assert new_edge.content_type == "RNA"  # Overridden
    content_names = [part.name for part in new_edge.content]
    assert "new_part1" in content_names
    assert "new_part2" in content_names
    assert "original_part" not in content_names

    # Check overridden property
    assert new_edge.extra.get("version") == 2


def test_copy_edge_comprehensive():
    """Test CopyEdge with all possible modifications"""
    from biocomp.graphengine import Part, GraphEdge

    graph = create_graph_state(
        [
            {"id": 0, "type": "original_source", "name": "oA"},
            {"id": 1, "type": "original_target", "name": "oB"},
            {"id": 2, "type": "copy_source", "name": "cA"},
            {"id": 3, "type": "copy_target", "name": "cB"},
        ],
        [],
    )

    part1 = Part(name="old_part", category="regulatory")
    part2 = Part(name="helper_part", category="helper")

    edges = [
        GraphEdge(
            source_id=0,
            target_id=1,
            from_output_slot=1,
            to_input_slot=2,
            content=(part1, part2),
            content_type="DNA",
            content_embedding_names={"rate": ("fast",)},
            extra={"strength": 0.3, "category": "original", "keep": "yes"},
        ),
    ]
    graph.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e for e in edges
    }

    rule = GraphRewritingRule(
        name="Comprehensive copy with all modifications",
        query=MatchQuery(
            bind={
                "copy_src": PropertyConstraint(properties={"type": "copy_source"}),
                "copy_tgt": PropertyConstraint(properties={"type": "copy_target"}),
            },
            bind_edges={"template": EdgeConstraint(contains=["old_part"])},
        ),
        actions=[
            CopyEdge(
                source_edge_var="template",
                source_var="copy_src",
                target_var="copy_tgt",
                content=["new_part1", "new_part2", "new_part3"],  # Override content
                content_type="RNA",  # Override content_type
                properties={
                    "strength": 0.9,  # Override
                    "category": "copied",  # Override
                    "modified": True,  # Add new
                    # "keep" should be preserved
                },
            ),
        ],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    # Should have original edge + new copied edge
    assert len(result_graph.edges) == 2

    # Find the new edge
    new_edge = next(
        edge for edge in result_graph.edges.values() if edge.source_id == 2 and edge.target_id == 3
    )

    # Check copied slots and embedding names
    assert new_edge.from_output_slot == 1  # Copied from original
    assert new_edge.to_input_slot == 2  # Copied from original
    assert new_edge.content_embedding_names == {"rate": ("fast",)}  # Copied

    # Check overridden content and content_type
    assert new_edge.content_type == "RNA"
    content_names = [part.name for part in new_edge.content]
    assert "new_part1" in content_names
    assert "new_part2" in content_names
    assert "new_part3" in content_names
    assert len(content_names) == 3

    # Check properties (preserved + overridden + new)
    assert new_edge.extra.get("strength") == 0.9  # Overridden
    assert new_edge.extra.get("category") == "copied"  # Overridden
    assert new_edge.extra.get("modified")  # Added
    assert new_edge.extra.get("keep") == "yes"  # Preserved


def test_copy_edge_with_templates():
    """Test CopyEdge with template variables in properties"""
    from biocomp.graphengine import Part, GraphEdge

    graph = create_graph_state(
        [
            {"id": 0, "type": "source", "name": "srcA", "weight": 0.4},
            {"id": 1, "type": "target", "name": "tgtB", "value": 0.6},
            {"id": 2, "type": "new_source", "name": "newC"},
            {"id": 3, "type": "new_target", "name": "newD"},
        ],
        [],
    )

    part1 = Part(name="connector", category="regulatory")

    edges = [
        GraphEdge(
            source_id=0,
            target_id=1,
            from_output_slot=0,
            to_input_slot=0,
            content=(part1,),
            content_type="DNA",
            extra={"connection_id": "original_123"},
        ),
    ]
    graph.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e for e in edges
    }

    rule = GraphRewritingRule(
        name="Copy edge with template properties",
        query=MatchQuery(
            bind={
                "new_src": PropertyConstraint(properties={"type": "new_source"}),
                "new_tgt": PropertyConstraint(properties={"type": "new_target"}),
            },
            bind_edges={"conn": EdgeConstraint(contains=["connector"])},
        ),
        actions=[
            CopyEdge(
                source_edge_var="conn",
                source_var="new_src",
                target_var="new_tgt",
                properties={
                    "copied_from": '{{ f"{conn.source_id}_to_{conn.target_id}" }}',
                    "original_connection_id": "{{conn.extra['connection_id']}}",
                    "new_connection": "{{ f\"{new_src.extra['name']}_{new_tgt.extra['name']}\" }}",
                },
            ),
        ],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    # Should have original edge + new copied edge
    assert len(result_graph.edges) == 2

    # Find the new edge
    new_edge = next(
        edge for edge in result_graph.edges.values() if edge.source_id == 2 and edge.target_id == 3
    )

    # Check template-expanded properties
    assert new_edge.extra.get("copied_from") == "0_to_1"
    assert new_edge.extra.get("original_connection_id") == "original_123"
    assert new_edge.extra.get("new_connection") == "newC_newD"

    # Check preserved original property
    assert new_edge.extra.get("connection_id") == "original_123"


def test_copy_edge_nonexistent_source_edge():
    """Test CopyEdge with non-existent source edge variable should raise error"""
    from biocomp.graphengine import Part, GraphEdge

    graph = create_graph_state(
        [
            {"id": 0, "type": "source", "name": "A"},
            {"id": 1, "type": "target", "name": "B"},
            {"id": 2, "type": "new_source", "name": "C"},
            {"id": 3, "type": "new_target", "name": "D"},
        ],
        [],
    )

    part1 = Part(name="test_part", category="regulatory")

    edges = [
        GraphEdge(
            source_id=0,
            target_id=1,
            from_output_slot=0,
            to_input_slot=0,
            content=(part1,),
            content_type="DNA",
        ),
    ]
    graph.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e for e in edges
    }

    rule = GraphRewritingRule(
        name="Copy from nonexistent edge",
        query=MatchQuery(
            bind={
                "new_src": PropertyConstraint(properties={"type": "new_source"}),
                "new_tgt": PropertyConstraint(properties={"type": "new_target"}),
            },
            bind_edges={"real_edge": EdgeConstraint(contains=["test_part"])},
        ),
        actions=[
            CopyEdge(
                source_edge_var="nonexistent_edge",  # This edge variable doesn't exist
                source_var="new_src",
                target_var="new_tgt",
            ),
        ],
    )

    # Should raise an error when trying to apply the rule
    try:
        apply_rule(rule, graph)
        raise AssertionError("Should have raised an error for nonexistent source edge variable")
    except ValueError as e:
        assert "not found in match" in str(e)


def test_copy_edge_nonexistent_node_variables():
    """Test CopyEdge with non-existent node variables should raise error"""
    from biocomp.graphengine import Part, GraphEdge

    graph = create_graph_state(
        [
            {"id": 0, "type": "source", "name": "A"},
            {"id": 1, "type": "target", "name": "B"},
        ],
        [],
    )

    part1 = Part(name="test_part", category="regulatory")

    edges = [
        GraphEdge(
            source_id=0,
            target_id=1,
            from_output_slot=0,
            to_input_slot=0,
            content=(part1,),
            content_type="DNA",
        ),
    ]
    graph.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e for e in edges
    }

    rule = GraphRewritingRule(
        name="Copy to nonexistent nodes",
        query=MatchQuery(
            bind_edges={"real_edge": EdgeConstraint(contains=["test_part"])},
        ),
        actions=[
            CopyEdge(
                source_edge_var="real_edge",
                source_var="nonexistent_source",  # This node variable doesn't exist
                target_var="nonexistent_target",  # This node variable doesn't exist
            ),
        ],
    )

    # Should raise an error when trying to apply the rule
    try:
        apply_rule(rule, graph)
        raise AssertionError("Should have raised an error for nonexistent node variables")
    except ValueError as e:
        assert "not found in match" in str(e)


def test_graphs_are_isomorphic_identical():
    """Test that identical graphs are equivalent"""
    from biocomp.graphengine import graphs_are_isomorphic

    # Create identical graphs
    nodes_data = [
        {"id": 0, "type": "promoter", "name": "pA"},
        {"id": 1, "type": "gene", "name": "gB"},
        {"id": 2, "type": "terminator", "name": "tC"},
    ]
    edges_data = [
        {"source": 0, "target": 1, "content_type": "DNA"},
        {"source": 1, "target": 2, "content_type": "RNA"},
    ]

    graph1 = create_graph_state(nodes_data, edges_data)
    graph2 = create_graph_state(nodes_data, edges_data)

    assert graphs_are_isomorphic(graph1, graph2)


def test_graphs_are_isomorphic_different_node_ids():
    """Test that graphs with different node IDs but same topology are equivalent"""
    from biocomp.graphengine import graphs_are_isomorphic

    # Graph 1: nodes 0,1,2
    graph1 = create_graph_state(
        [
            {"id": 0, "type": "promoter", "name": "pA"},
            {"id": 1, "type": "gene", "name": "gB"},
            {"id": 2, "type": "terminator", "name": "tC"},
        ],
        [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
        ],
    )

    # Graph 2: nodes 10,20,30 but same topology
    graph2 = create_graph_state(
        [
            {"id": 10, "type": "promoter", "name": "pA"},
            {"id": 20, "type": "gene", "name": "gB"},
            {"id": 30, "type": "terminator", "name": "tC"},
        ],
        [
            {"source": 10, "target": 20},
            {"source": 20, "target": 30},
        ],
    )

    assert graphs_are_isomorphic(graph1, graph2)


def test_graphs_are_isomorphic_different_node_order():
    """Test that graphs with same nodes/edges in different order are equivalent"""
    from biocomp.graphengine import graphs_are_isomorphic

    # Graph 1: normal order
    graph1 = create_graph_state(
        [
            {"id": 0, "type": "promoter", "name": "pA"},
            {"id": 1, "type": "gene", "name": "gB"},
            {"id": 2, "type": "terminator", "name": "tC"},
        ],
        [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
        ],
    )

    # Graph 2: different order
    graph2 = create_graph_state(
        [
            {"id": 2, "type": "terminator", "name": "tC"},
            {"id": 0, "type": "promoter", "name": "pA"},
            {"id": 1, "type": "gene", "name": "gB"},
        ],
        [
            {"source": 1, "target": 2},
            {"source": 0, "target": 1},
        ],
    )

    assert graphs_are_isomorphic(graph1, graph2)


def test_graphs_are_isomorphic_with_content():
    """Test that graphs with edge content are compared correctly"""
    from biocomp.graphengine import graphs_are_isomorphic, Part, GraphEdge

    part1 = Part(name="uORF1", category="regulatory")
    part2 = Part(name="RBS1", category="ribosome_binding")

    # Create graphs with content
    nodes_data = [{"id": 0, "type": "source"}, {"id": 1, "type": "target"}]

    graph1 = create_graph_state(nodes_data, [])
    graph1.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e
        for e in [
            GraphEdge(
                source_id=0,
                target_id=1,
                from_output_slot=0,
                to_input_slot=0,
                content=(part1, part2),
                content_type="DNA",
            )
        ]
    }

    graph2 = create_graph_state(nodes_data, [])
    graph2.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e
        for e in [
            GraphEdge(
                source_id=0,
                target_id=1,
                from_output_slot=0,
                to_input_slot=0,
                content=(part1, part2),
                content_type="DNA",
            )
        ]
    }

    assert graphs_are_isomorphic(graph1, graph2)


def test_graphs_are_isomorphic_different_content():
    """Test that graphs with different edge content are not equivalent"""
    from biocomp.graphengine import graphs_are_isomorphic, Part, GraphEdge

    part1 = Part(name="uORF1", category="regulatory")
    part2 = Part(name="RBS1", category="ribosome_binding")
    part3 = Part(name="uORF2", category="regulatory")

    nodes_data = [{"id": 0, "type": "source"}, {"id": 1, "type": "target"}]

    graph1 = create_graph_state(nodes_data, [])
    graph1.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e
        for e in [
            GraphEdge(
                source_id=0,
                target_id=1,
                from_output_slot=0,
                to_input_slot=0,
                content=(part1, part2),
                content_type="DNA",
            )
        ]
    }

    graph2 = create_graph_state(nodes_data, [])
    graph2.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e
        for e in [
            GraphEdge(
                source_id=0,
                target_id=1,
                from_output_slot=0,
                to_input_slot=0,
                content=(part1, part3),
                content_type="DNA",
            )  # Different content
        ]
    }

    assert not graphs_are_isomorphic(graph1, graph2)


def test_graphs_are_isomorphic_different_node_types():
    """Test that graphs with different node types are not equivalent"""
    from biocomp.graphengine import graphs_are_isomorphic

    graph1 = create_graph_state(
        [{"id": 0, "type": "promoter"}, {"id": 1, "type": "gene"}], [{"source": 0, "target": 1}]
    )

    graph2 = create_graph_state(
        [{"id": 0, "type": "promoter"}, {"id": 1, "type": "terminator"}],  # Different type
        [{"source": 0, "target": 1}],
    )

    assert not graphs_are_isomorphic(graph1, graph2)


def test_graphs_are_isomorphic_different_topology():
    """Test that graphs with different topology are not equivalent"""
    from biocomp.graphengine import graphs_are_isomorphic

    nodes_data = [
        {"id": 0, "type": "promoter"},
        {"id": 1, "type": "gene"},
        {"id": 2, "type": "terminator"},
    ]

    # Linear topology: 0 -> 1 -> 2
    graph1 = create_graph_state(
        nodes_data, [{"source": 0, "target": 1}, {"source": 1, "target": 2}]
    )

    # Different topology: 0 -> 2, 1 -> 2
    graph2 = create_graph_state(
        nodes_data, [{"source": 0, "target": 2}, {"source": 1, "target": 2}]
    )

    assert not graphs_are_isomorphic(graph1, graph2)


def test_graphs_are_isomorphic_different_sizes():
    """Test that graphs with different sizes are not equivalent"""
    from biocomp.graphengine import graphs_are_isomorphic

    graph1 = create_graph_state(
        [{"id": 0, "type": "promoter"}, {"id": 1, "type": "gene"}], [{"source": 0, "target": 1}]
    )

    graph2 = create_graph_state(
        [{"id": 0, "type": "promoter"}],  # One fewer node
        [],
    )

    assert not graphs_are_isomorphic(graph1, graph2)


def test_graphs_are_isomorphic_with_extra_ignored():
    """Test that extra fields are ignored by default"""
    from biocomp.graphengine import graphs_are_isomorphic

    graph1 = create_graph_state([{"id": 0, "type": "promoter", "strength": 0.8}], [])

    graph2 = create_graph_state(
        [{"id": 0, "type": "promoter", "strength": 0.9}],  # Different extra
        [],
    )

    # Should be equivalent when compare_extra=False (default)
    assert graphs_are_isomorphic(graph1, graph2, compare_extra=False)


def test_graphs_are_isomorphic_with_extra_compared():
    """Test that extra fields are compared when compare_extra=True"""
    from biocomp.graphengine import graphs_are_isomorphic

    graph1 = create_graph_state([{"id": 0, "type": "promoter", "strength": 0.8}], [])

    graph2 = create_graph_state(
        [{"id": 0, "type": "promoter", "strength": 0.9}],  # Different extra
        [],
    )

    # Should NOT be equivalent when compare_extra=True
    assert not graphs_are_isomorphic(graph1, graph2, compare_extra=True)

    # But should be equivalent if extra is the same
    graph3 = create_graph_state(
        [{"id": 0, "type": "promoter", "strength": 0.8}],  # Same extra
        [],
    )

    assert graphs_are_isomorphic(graph1, graph3, compare_extra=True)


def test_graphs_are_isomorphic_content_embedding_names():
    """Test comparison of content_embedding_names"""
    from biocomp.graphengine import graphs_are_isomorphic, Part, GraphEdge

    part1 = Part(name="uORF1", category="regulatory")
    nodes_data = [{"id": 0, "type": "source"}, {"id": 1, "type": "target"}]

    graph1 = create_graph_state(nodes_data, [])
    graph1.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e
        for e in [
            GraphEdge(
                source_id=0,
                target_id=1,
                from_output_slot=0,
                to_input_slot=0,
                content=(part1,),
                content_type="DNA",
                content_embedding_names={"rate": ("fast",)},
            )
        ]
    }

    graph2 = create_graph_state(nodes_data, [])
    graph2.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e
        for e in [
            GraphEdge(
                source_id=0,
                target_id=1,
                from_output_slot=0,
                to_input_slot=0,
                content=(part1,),
                content_type="DNA",
                content_embedding_names={"rate": ("slow",)},
            )  # Different embedding
        ]
    }

    # Should be equivalent when compare_content_embedding_names=False (default)
    assert graphs_are_isomorphic(graph1, graph2, compare_content_embedding_names=False)

    # Should NOT be equivalent when compare_content_embedding_names=True
    assert not graphs_are_isomorphic(graph1, graph2, compare_content_embedding_names=True)


def test_graphs_are_isomorphic_multiple_node_types():
    """Test equivalence with multiple nodes of the same type"""
    from biocomp.graphengine import graphs_are_isomorphic

    # Both graphs have 2 genes and 1 promoter
    graph1 = create_graph_state(
        [
            {"id": 0, "type": "promoter", "name": "pA"},
            {"id": 1, "type": "gene", "name": "gB"},
            {"id": 2, "type": "gene", "name": "gC"},
        ],
        [
            {"source": 0, "target": 1},
            {"source": 0, "target": 2},
        ],
    )

    # Same structure but different IDs and potentially different mapping
    graph2 = create_graph_state(
        [
            {"id": 10, "type": "gene", "name": "gB"},
            {"id": 20, "type": "promoter", "name": "pA"},
            {"id": 30, "type": "gene", "name": "gC"},
        ],
        [
            {"source": 20, "target": 10},
            {"source": 20, "target": 30},
        ],
    )

    assert graphs_are_isomorphic(graph1, graph2)


def test_graphs_are_isomorphic_edge_slots():
    """Test that edge input/output slots are compared"""
    from biocomp.graphengine import graphs_are_isomorphic, GraphEdge

    nodes_data = [{"id": 0, "type": "source"}, {"id": 1, "type": "target"}]

    graph1 = create_graph_state(nodes_data, [])
    graph1.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e
        for e in [
            GraphEdge(source_id=0, target_id=1, from_output_slot=0, to_input_slot=1, content=())
        ]
    }

    graph2 = create_graph_state(nodes_data, [])
    graph2.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e
        for e in [
            GraphEdge(
                source_id=0, target_id=1, from_output_slot=1, to_input_slot=0, content=()
            )  # Different slots
        ]
    }

    assert not graphs_are_isomorphic(graph1, graph2)


def test_graphs_are_isomorphic_complex_mapping():
    """Test complex node mapping with multiple permutations"""
    from biocomp.graphengine import graphs_are_isomorphic

    # Create a graph where multiple nodes have the same type
    # This tests that the algorithm tries all possible mappings
    graph1 = create_graph_state(
        [
            {"id": 0, "type": "gene", "name": "A"},
            {"id": 1, "type": "gene", "name": "B"},
            {"id": 2, "type": "promoter", "name": "P"},
        ],
        [
            {"source": 2, "target": 0},  # P -> A
            {"source": 2, "target": 1},  # P -> B
        ],
    )

    # Same structure but gene names swapped
    graph2 = create_graph_state(
        [
            {"id": 10, "type": "gene", "name": "B"},  # Note: B first
            {"id": 20, "type": "gene", "name": "A"},  # Note: A second
            {"id": 30, "type": "promoter", "name": "P"},
        ],
        [
            {"source": 30, "target": 10},  # P -> B
            {"source": 30, "target": 20},  # P -> A
        ],
    )

    assert graphs_are_isomorphic(graph1, graph2)


def test_graphs_are_isomorphic_empty_graphs():
    """Test that empty graphs are equivalent"""
    from biocomp.graphengine import graphs_are_isomorphic

    graph1 = create_graph_state([], [])
    graph2 = create_graph_state([], [])

    assert graphs_are_isomorphic(graph1, graph2)


def test_graphs_are_isomorphic_single_node():
    """Test equivalence with single node graphs"""
    from biocomp.graphengine import graphs_are_isomorphic

    graph1 = create_graph_state([{"id": 0, "type": "gene"}], [])
    graph2 = create_graph_state([{"id": 100, "type": "gene"}], [])

    assert graphs_are_isomorphic(graph1, graph2)


def test_graphs_are_isomorphic_self_loop():
    """Test equivalence with self-loops"""
    from biocomp.graphengine import graphs_are_isomorphic

    graph1 = create_graph_state(
        [{"id": 0, "type": "gene"}],
        [{"source": 0, "target": 0}],  # Self-loop
    )

    graph2 = create_graph_state(
        [{"id": 5, "type": "gene"}],
        [{"source": 5, "target": 5}],  # Self-loop
    )

    assert graphs_are_isomorphic(graph1, graph2)


def test_graphs_are_isomorphic_with_real_networks():
    """Test equivalence using real network structures similar to biocompiler networks"""
    from biocomp.graphengine import graphs_are_isomorphic, Part, GraphEdge

    # Create a realistic network structure
    part_dna = Part(name="CasE", category="DNA")
    part_rna = Part(name="CasE", category="RNA")
    part_prt = Part(name="CasE", category="PRT")

    # Network 1: source -> transcription -> translation -> output
    graph1 = create_graph_state(
        [
            {"id": 0, "type": "source"},
            {"id": 1, "type": "transcription"},
            {"id": 2, "type": "translation"},
            {"id": 3, "type": "output"},
        ],
        [],
    )
    graph1.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e
        for e in [
            GraphEdge(
                source_id=0,
                target_id=1,
                from_output_slot=0,
                to_input_slot=0,
                content=(part_dna,),
                content_type="DNA",
            ),
            GraphEdge(
                source_id=1,
                target_id=2,
                from_output_slot=0,
                to_input_slot=0,
                content=(part_rna,),
                content_type="RNA",
            ),
            GraphEdge(
                source_id=2,
                target_id=3,
                from_output_slot=0,
                to_input_slot=0,
                content=(part_prt,),
                content_type="PRT",
            ),
        ]
    }

    # Network 2: same structure, different IDs
    graph2 = create_graph_state(
        [
            {"id": 10, "type": "source"},
            {"id": 20, "type": "transcription"},
            {"id": 30, "type": "translation"},
            {"id": 40, "type": "output"},
        ],
        [],
    )
    graph2.edges = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e
        for e in [
            GraphEdge(
                source_id=10,
                target_id=20,
                from_output_slot=0,
                to_input_slot=0,
                content=(part_dna,),
                content_type="DNA",
            ),
            GraphEdge(
                source_id=20,
                target_id=30,
                from_output_slot=0,
                to_input_slot=0,
                content=(part_rna,),
                content_type="RNA",
            ),
            GraphEdge(
                source_id=30,
                target_id=40,
                from_output_slot=0,
                to_input_slot=0,
                content=(part_prt,),
                content_type="PRT",
            ),
        ]
    }

    assert graphs_are_isomorphic(graph1, graph2)


# ============================================================================
# GraphState Getter Method Tests
# ============================================================================


def test_graphstate_get_node():
    """Test get_node() returns correct node or None"""
    graph = create_graph_state([{"id": 0, "type": "input"}, {"id": 1, "type": "output"}], [])

    node = graph.get_node(0)
    assert node is not None
    assert node.node_id == 0
    assert node.node_type == "input"

    assert graph.get_node(999) is None


def test_graphstate_get_edge():
    """Test get_edge() returns correct edge or None"""
    graph = create_graph_state(
        [{"id": 0, "type": "input"}, {"id": 1, "type": "output"}],
        [{"source": 0, "target": 1, "from_output_slot": 0, "to_input_slot": 0}],
    )

    edge = graph.get_edge(0, 1)
    assert edge is not None
    assert edge.source_id == 0
    assert edge.target_id == 1

    assert graph.get_edge(999, 1) is None
    assert graph.get_edge(0, 999) is None
    assert graph.get_edge(0, 1, from_output_slot=1) is None


def test_graphstate_get_outgoing_edges():
    """Test get_outgoing_edges() returns all edges from a node"""
    graph = create_graph_state(
        [{"id": 0, "type": "input"}, {"id": 1, "type": "middle"}, {"id": 2, "type": "output"}],
        [
            {"source": 0, "target": 1},
            {"source": 0, "target": 2},
            {"source": 1, "target": 2},
        ],
    )

    out_edges = graph.get_outgoing_edges(0)
    assert len(out_edges) == 2
    assert all(e.source_id == 0 for e in out_edges)

    out_edges_1 = graph.get_outgoing_edges(1)
    assert len(out_edges_1) == 1
    assert out_edges_1[0].target_id == 2

    assert graph.get_outgoing_edges(999) == []


def test_graphstate_get_incoming_edges():
    """Test get_incoming_edges() returns all edges to a node"""
    graph = create_graph_state(
        [{"id": 0, "type": "input"}, {"id": 1, "type": "middle"}, {"id": 2, "type": "output"}],
        [
            {"source": 0, "target": 1},
            {"source": 0, "target": 2},
            {"source": 1, "target": 2},
        ],
    )

    in_edges = graph.get_incoming_edges(2)
    assert len(in_edges) == 2
    assert all(e.target_id == 2 for e in in_edges)

    in_edges_1 = graph.get_incoming_edges(1)
    assert len(in_edges_1) == 1
    assert in_edges_1[0].source_id == 0

    assert graph.get_incoming_edges(999) == []


def test_graphstate_dict_structure_uniqueness():
    """Test that dict structure enforces unique node IDs and edge combinations"""
    # Node IDs must be unique
    nodes = {
        0: GraphNode(node_id=0, node_type="A"),
        1: GraphNode(node_id=1, node_type="B"),
    }

    # Edge keys enforce uniqueness
    edge1 = GraphEdge(source_id=0, target_id=1, from_output_slot=0, to_input_slot=0, content=())
    edge2 = GraphEdge(source_id=0, target_id=1, from_output_slot=1, to_input_slot=0, content=())

    edges = {
        (0, 1, 0, 0): edge1,
        (0, 1, 1, 0): edge2,
    }

    graph = GraphState(nodes=nodes, edges=edges)

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 2
    assert graph.get_edge(0, 1, from_output_slot=0) == edge1
    assert graph.get_edge(0, 1, from_output_slot=1) == edge2
