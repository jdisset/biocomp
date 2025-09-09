import pytest
import pandas as pd
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
)
from biocomp.graphengine import GraphState, GraphNode, GraphEdge, apply_rule

# ---------------------------------------------------------------------------
# Helper Fixtures and Functions
# ---------------------------------------------------------------------------


def create_graph_state(nodes_data, edges_data):
    nodes = (
        [
            GraphNode(
                node_id=node["id"],
                node_type=node.get("type", "unknown"),
                extra={k: v for k, v in node.items() if k not in ["id", "type"]},
            )
            for node in nodes_data
        ]
        if nodes_data
        else []
    )

    edges = (
        [
            GraphEdge(
                source_id=edge["source"],
                target_id=edge["target"],
                output_slot=edge.get("output_slot", 0),
                input_slot=edge.get("input_slot", 0),
                content=edge.get("content", ()),
                content_type=edge.get("content_type", None),
            )
            for edge in edges_data
        ]
        if edges_data
        else []
    )

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
    gene_node = next(node for node in new_graph.nodes if node.node_id == 1)
    assert gene_node.extra["matched"] == True

    # Check that node 0 is untouched
    promoter_node = next(node for node in new_graph.nodes if node.node_id == 0)
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
    for orig_node, new_node in zip(simple_graph.nodes, new_graph.nodes):
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

    promoter_node = next(node for node in new_graph.nodes if node.node_id == 0)
    gene_node = next(node for node in new_graph.nodes if node.node_id == 1)
    terminator_node = next(node for node in new_graph.nodes if node.node_id == 2)

    assert promoter_node.extra["p_matched"] == True
    assert gene_node.extra["g_matched"] == True
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
    root_node = next(node for node in new_graph.nodes if node.node_id == 0)
    assert root_node.extra["is_root"] == True

    # Other nodes should not be marked as root
    for node in new_graph.nodes:
        if node.node_id != 0:
            assert "is_root" not in node.extra


def test_action_add_node(simple_graph):
    rule = GraphRewritingRule(
        name="Add Node for every Gene",
        query=MatchQuery(bind={"g": PropertyConstraint(properties={"type": "gene"})}),
        actions=[
            AddNode(local_name="new_node", properties={"type": "marker", "linked_to": "{{g.name}}"})
        ],
    )

    new_graphs = apply_rule(rule, simple_graph)
    assert len(new_graphs) == 1
    new_graph = new_graphs[0]

    assert len(new_graph.nodes) == 4
    marker_nodes = [node for node in new_graph.nodes if node.node_type == "marker"]
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
        edge for edge in new_graph.edges if edge.source_id == 2 and edge.target_id == 0
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
    node_ids = {node.node_id for node in new_graph.nodes}
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
    remaining_edge = new_graph.edges[0]
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
                properties={"type": "cassette", "name": "{{p.name}}+{{g.name}}"},
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
    cassette_nodes = [node for node in new_graph.nodes if node.node_type == "cassette"]
    assert len(cassette_nodes) == 1
    cassette_node = cassette_nodes[0]
    assert cassette_node.extra["name"] == "pA+gB"

    assert len(new_graph.edges) == 1
    final_edge = new_graph.edges[0]
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
                    "val": "{{parent.val}}+{{child.val}}",
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

    # Expected outcome: A, B, C, D -> (A+B), C, D -> (A+B+C), D -> (A+B+C+D)
    assert len(final_graph.nodes) == 1
    final_node = final_graph.nodes[0]
    # The exact value depends on the non-deterministic match order.
    # A robust engine would need a strategy (e.g., match lowest ID first).
    # Let's assume it fuses 0+1, then (0+1)+2, then (0+1+2)+3
    assert final_node.extra["val"] == "0+1+2+3"
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
        actions=[AddNode(local_name="marker", properties={"type": "marker", "for": "{{p.name}}"})],
        yield_strategy="batched",
    )

    rule_per_match = GraphRewritingRule(
        name="Add markers to all promoters (per match)",
        query=MatchQuery(bind={"p": PropertyConstraint(properties={"type": "promoter"})}),
        actions=[AddNode(local_name="marker", properties={"type": "marker", "for": "{{p.name}}"})],
        yield_strategy="per_match",
    )

    # Batched strategy should return one graph with both markers
    batched_graphs = apply_rule(rule_batched, graph)
    assert len(batched_graphs) == 1
    batched_graph = batched_graphs[0]
    assert len(batched_graph.nodes) == 4  # 2 promoters + 2 markers
    markers = [n for n in batched_graph.nodes if n.node_type == "marker"]
    assert len(markers) == 2

    # Per-match strategy should return two graphs, each with one marker
    per_match_graphs = apply_rule(rule_per_match, graph)
    assert len(per_match_graphs) == 2
    for result_graph in per_match_graphs:
        assert len(result_graph.nodes) == 3  # 2 promoters + 1 marker
        markers = [n for n in result_graph.nodes if n.node_type == "marker"]
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

    active_gene = next(n for n in result_graph.nodes if n.node_id == 0)
    inactive_gene = next(n for n in result_graph.nodes if n.node_id == 1)

    assert active_gene.extra["marked"] == True
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
                    "description": "{{p.region}}_controls_{{g.name}}_length_{{g.length}}",
                    "promoter_type": "{{p.type}}",
                    "target_gene": "{{g.name}}",
                },
            )
        ],
    )

    result = apply_rule(rule, graph)
    assert len(result) == 1
    result_graph = result[0]

    annotations = [n for n in result_graph.nodes if n.node_type == "annotation"]
    assert len(annotations) == 1

    annotation = annotations[0]
    assert annotation.extra["description"] == "upstream_controls_BRCA1_length_1863"
    assert annotation.extra["promoter_type"] == "promoter"
    assert annotation.extra["target_gene"] == "BRCA1"


def test_jinja2_advanced_template_features():
    """Test advanced Jinja2 template features that weren't possible with manual expansion"""
    graph = create_graph_state(
        [
            {"id": 0, "type": "gene", "name": "gene_A", "active": True, "score": 85},
            {"id": 1, "type": "gene", "name": "gene_B", "active": False, "score": 42},
        ],
        [],
    )

    rule = GraphRewritingRule(
        name="Advanced Jinja2 features",
        query=MatchQuery(bind={"g": PropertyConstraint(properties={"type": "gene"})}),
        actions=[
            AddNode(
                local_name="summary",
                properties={
                    "type": "summary",
                    # Conditional logic
                    "status": "{% if g.active == 'True' %}ACTIVE{% else %}INACTIVE{% endif %}",
                    # String manipulation  
                    "upper_name": "{{g.name.upper()}}",
                    # Arithmetic operations
                    "score_doubled": "{{(g.score|int) * 2}}",
                    # Complex expressions
                    "grade": "{% if (g.score|int) >= 80 %}A{% elif (g.score|int) >= 60 %}B{% else %}C{% endif %}",
                },
            )
        ],
    )

    result = apply_rule(rule, graph)[0]
    summaries = [n for n in result.nodes if n.node_type == "summary"]
    assert len(summaries) == 2

    # Find active gene summary
    active_summary = next(s for s in summaries if s.extra["status"] == "ACTIVE")
    assert active_summary.extra["upper_name"] == "GENE_A"
    assert active_summary.extra["score_doubled"] == "170"
    assert active_summary.extra["grade"] == "A"

    # Find inactive gene summary  
    inactive_summary = next(s for s in summaries if s.extra["status"] == "INACTIVE")
    assert inactive_summary.extra["upper_name"] == "GENE_B"
    assert inactive_summary.extra["score_doubled"] == "84"
    assert inactive_summary.extra["grade"] == "C"  # 42 < 60, so grade C


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
    gene_node = next(n for n in result_graph.nodes if n.node_id == 0)
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
    promoter_to_gene = [e for e in result_graph.edges if e.source_id == 0]
    assert len(promoter_to_gene) == 1

    gene_nodes = [n for n in result_graph.nodes if n.node_type == "gene"]
    term_nodes = [n for n in result_graph.nodes if n.node_type == "terminator"]

    gene_to_term = [
        e
        for e in result_graph.edges
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
    found_genes = [n for n in result_graph.nodes if n.extra.get("found")]
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
                properties={"type": "intermediate", "value": "{{a.value}}_{{b.value}}"},
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

    # Should eventually merge to single node
    assert len(result_graph.nodes) == 1
    assert len(result_graph.edges) == 0

    final_node = result_graph.nodes[0]
    assert final_node.node_type == "intermediate"
    # Value should show the merging pattern
    assert "_" in final_node.extra["value"]


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
            AddNode(local_name="mrna", properties={"type": "RNA", "transcript_of": "{{g.name}}"}),
            AddNode(
                local_name="protein", properties={"type": "protein", "product": "{{g.product}}"}
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

    rna_nodes = [n for n in result_graph.nodes if n.node_type == "RNA"]
    protein_nodes = [n for n in result_graph.nodes if n.node_type == "protein"]

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
    assert result_graph.nodes[0].node_type == "gene"


def test_edge_properties_and_content():
    from biocomp.graphengine import Part

    part = Part(name="uORF1", category="regulatory")
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
    edge = result_graph.edges[0]
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
    processed_nodes = [n for n in result_graph.nodes if n.extra.get("processed")]
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
            AddNode(local_name="fused", properties={"type": "fused", "name": "{{a.name}}+{{b.name}}"}),
            DeleteNode(node_var="a"),
            DeleteNode(node_var="b"),
        ],
        run_until_stable=True,
    )

    result = apply_rule(fuse_rule, graph)[0]
    
    assert len(result.nodes) == 2
    node_names = sorted([n.extra["name"] for n in result.nodes])
    assert node_names == ["A+B", "C+D"]


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
            AddNode(local_name="fused", properties={"type": "any", "name": "({{a.name}}+{{b.name}})"}),
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
    assert result.nodes[0].extra["name"] == "((A+B)+C)"
