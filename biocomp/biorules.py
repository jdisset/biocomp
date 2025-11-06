from biocomp.graphrules import (
    GraphRewritingRule,
    MatchQuery,
    PropertyConstraint,
    EdgeConstraint,
    AddNode,
    AddEdge,
    DeleteNode,
    DeleteEdge,
    EditEdge,
    CopyEdge,
    RewireEdgesFrom,
    SetProperties,
    DeleteProperties,
)


merge_sources_by_id = GraphRewritingRule(
    name="merge_sources_by_id",
    query=MatchQuery(
        bind={
            "source1": PropertyConstraint(properties={"type": "source"}),
            "source2": PropertyConstraint(properties={"type": "source"}),
        },
        where_filter_function="source1.extra.get('source_id') == source2.extra.get('source_id') and source1.node_id != source2.node_id and source1.extra.get('cotx_group') == source2.extra.get('cotx_group')",
    ),
    actions=[
        RewireEdgesFrom(old_source_var="source2", new_source_var="source1"),
        DeleteNode(node_var="source2"),
    ],
    yield_strategy="batched",
)

create_aggregation_nodes = GraphRewritingRule(
    name="create_aggregation_nodes",
    query=MatchQuery(
        bind={
            "source1": PropertyConstraint(properties={"type": "source"}),
            "source2": PropertyConstraint(properties={"type": "source"}),
        },
        # Only create aggregation when there are at least 2 different sources in the same cotx group
        # Use the two sources with the lowest node_ids in the group to ensure only one aggregation per group
        where_filter_function="source1.extra.get('cotx_group') == source2.extra.get('cotx_group') and source1.node_id < source2.node_id",
        # Only match sources that don't have incoming edges (no aggregation created yet for this group)
        where_not_connected=[
            EdgeConstraint(source_var="any", target_var="source1"),
            EdgeConstraint(source_var="any", target_var="source2"),
        ],
    ),
    actions=[
        AddNode(
            local_name="aggregation",
            properties={
                "type": "aggregation",
                "cotx_group": "{{source1.extra.get('cotx_group')}}",
                "ratios": [],
                "members": [],
                "ratio_ranges": [],
                "fluo_bias": "{{ source1.extra.get('fluo_bias') if source1.extra.get('fluo_bias') else None }}",
            },
        ),
    ],
    yield_strategy="batched",
)

connect_sources_to_aggregation = GraphRewritingRule(
    name="connect_sources_to_aggregation",
    query=MatchQuery(
        bind={
            "source": PropertyConstraint(properties={"type": "source"}),
            "aggregation": PropertyConstraint(properties={"type": "aggregation"}),
        },
        # Only connect to aggregation in same cotx group and sources that are not yet connected
        where_filter_function="source.extra.get('cotx_group') == aggregation.extra.get('cotx_group') and source.extra.get('source_id') not in aggregation.extra.get('members', [])",
        where_not_connected=[EdgeConstraint(target_var="source")],
    ),
    actions=[
        AddEdge(
            source="aggregation",
            target="source",
            properties={
                "content_type": None,
                "from_output_slot": "{{ len(aggregation.extra.get('members', [])) }}",
            },
        ),
        SetProperties(
            node_var="aggregation",
            properties={
                "ratios": "{{ aggregation.extra.get('ratios', []) + [source.extra.get('ratio') if source.extra.get('ratio') else 1.0] }}",
                "members": "{{ aggregation.extra.get('members', []) + [source.extra.get('source_id')] }}",
                "ratio_ranges": "{{ (aggregation.extra.get('ratio_ranges') if aggregation.extra.get('ratio_ranges') else []) + ([source.extra.get('ratio_range')] if source.extra.get('ratio_range') else [None]) }}",
            },
        ),
    ],
    yield_strategy="per_match",  # Changed from batched to per_match for deterministic ordering
    run_until_stable=True,
)

merge_aggregators_by_group = GraphRewritingRule(
    name="merge_aggregators_by_group",
    query=MatchQuery(
        bind={
            "agg1": PropertyConstraint(properties={"type": "aggregation"}),
            "agg2": PropertyConstraint(properties={"type": "aggregation"}),
        },
        # Find two different aggregation nodes in the same cotx group
        where_filter_function="agg1.extra.get('cotx_group') == agg2.extra.get('cotx_group') and agg1.node_id < agg2.node_id",
    ),
    actions=[
        RewireEdgesFrom(old_source_var="agg1", new_source_var="agg2"),
        SetProperties(
            node_var="agg2",
            properties={
                "ratios": "{{ agg1.extra.get('ratios', []) + agg2.extra.get('ratios', []) }}",
                "members": "{{ agg1.extra.get('members', []) + agg2.extra.get('members', []) }}",
                "ratio_ranges": "{{ (agg1.extra.get('ratio_ranges') if agg1.extra.get('ratio_ranges') else []) + (agg2.extra.get('ratio_ranges') if agg2.extra.get('ratio_ranges') else []) }}",
            },
        ),
        # Delete the now-redundant agg1
        DeleteNode(node_var="agg1"),
    ],
    run_until_stable=True,
    yield_strategy="batched",
)


sort_aggregation_members = GraphRewritingRule(
    name="sort_aggregation_members",
    query=MatchQuery(
        bind={
            "aggregation": PropertyConstraint(properties={"type": "aggregation"}),
        },
        # Only sort aggregations that have members and need reordering
        where_filter_function="len(aggregation.extra.get('members', [])) > 1 and aggregation.extra.get('members', []) != sorted(aggregation.extra.get('members', []))",
    ),
    actions=[
        SetProperties(
            node_var="aggregation",
            properties={
                # Sort members and reorder ratios and ratio_ranges to match
                "members": "{{ sorted(aggregation.extra.get('members', [])) }}",
                "ratios": "{{ reorder_list(aggregation.extra.get('ratios', []), sorted_with_indices(aggregation.extra.get('members', []))[1]) }}",
                "ratio_ranges": "{{ reorder_list(aggregation.extra.get('ratio_ranges', []), sorted_with_indices(aggregation.extra.get('members', []))[1]) }}",
            },
        ),
    ],
    yield_strategy="batched",
)

fix_edge_slots = GraphRewritingRule(
    name="fix_edge_slots",
    query=MatchQuery(
        bind={
            "aggregation": PropertyConstraint(properties={"type": "aggregation"}),
            "source": PropertyConstraint(properties={"type": "source"}),
        },
        where_connected=[
            EdgeConstraint(
                source_var="aggregation", target_var="source", properties={"content_type": None}
            ),
        ],
        # Only process edges where the source is in the aggregation's members
        where_filter_function="source.extra.get('source_id') in aggregation.extra.get('members', [])",
    ),
    actions=[
        # Delete the existing edge
        DeleteEdge(source_var="aggregation", target_var="source"),
        # Recreate with correct slot
        AddEdge(
            source="aggregation",
            target="source",
            properties={
                "content_type": None,
                "from_output_slot": "{{ 0 if aggregation.extra.get('members', [])[0] == source.extra.get('source_id') else 1 }}",
            },
        ),
    ],
    yield_strategy="batched",
)

add_numeric_nodes = GraphRewritingRule(
    name="add_numeric_nodes",
    query=MatchQuery(
        bind={
            "top_node": PropertyConstraint(properties={}),  # Any node type
        },
        # Match nodes with no incoming edges (top-level)
        where_not_connected=[EdgeConstraint(source_var="any", target_var="top_node")],
        # Only for source and aggregation nodes without fluo_bias (or with string 'None')
        where_filter_function="top_node.node_type in ['source', 'aggregation'] and (top_node.extra.get('fluo_bias') is None or top_node.extra.get('fluo_bias') == 'None')",
    ),
    actions=[
        AddNode(
            local_name="numeric",
            properties={
                "type": "numeric",
                "role": "copy_number",
            },
        ),
        AddEdge(
            source="numeric",
            target="top_node",
            properties={"content_type": None},  # Copy number flow
        ),
    ],
    yield_strategy="batched",
)

add_bias_nodes = GraphRewritingRule(
    name="add_bias_nodes",
    query=MatchQuery(
        bind={
            "top_node": PropertyConstraint(properties={}),  # Any node type
        },
        # Match nodes with no incoming edges (top-level)
        where_not_connected=[EdgeConstraint(source_var="any", target_var="top_node")],
        # For both source and aggregation nodes WITH fluo_bias (but not string 'None')
        where_filter_function="top_node.node_type in ['source', 'aggregation'] and top_node.extra.get('fluo_bias') is not None and top_node.extra.get('fluo_bias') != 'None'",
    ),
    actions=[
        AddNode(
            local_name="bias",
            properties={
                "type": "bias",
                "role": "fluo_bias",
                "fluo_bias": "{{ top_node.extra.get('fluo_bias') }}",
            },
        ),
        AddEdge(
            source="bias",
            target="top_node",
            properties={"content_type": None},  # Bias flow
        ),
        # single source of truth: we remove fluo_bias from top_node
        DeleteProperties(
            node_var="top_node",
            property_keys=["fluo_bias"],
        ),
    ],
    yield_strategy="batched",
)


def make_ern_rule(ern_name="CasE", ern_rec_name="CasE_rec"):
    return GraphRewritingRule(
        name=f"add_{ern_name.lower()}_sequestron",
        query=MatchQuery(
            bind_edges={
                "negative": EdgeConstraint(properties={"content_type": "PRT"}, contains=[ern_name]),
                "positive": EdgeConstraint(
                    properties={"content_type": "RNA"}, contains=[ern_rec_name]
                ),
            },
        ),
        actions=[
            AddNode(
                local_name="sequestron",
                properties={
                    "type": "sequestron_ERN",
                    "seq_name": f"ERN::{ern_name}#{ern_rec_name}",
                },
            ),
            DeleteNode(node_var="negative_target"),  # Auto-bound target node
            # Rewire the PRT edge (negative input): translation → sequestron (to_input_slot=0)
            EditEdge(edge_var="negative", target_var="sequestron", properties={"to_input_slot": 0}),
            CopyEdge(
                source_edge_var="positive", target_var="positive_target", source_var="sequestron"
            ),
            # Rewire the RNA edge (positive input): transcription → sequestron (to_input_slot=1)
            EditEdge(edge_var="positive", target_var="sequestron", properties={"to_input_slot": 1}),
        ],
        yield_strategy="batched",
        run_until_stable=True,
    )


SEQUESTRON_RULES = [
    make_ern_rule(ern_name=e, ern_rec_name=f"{e}_rec") for e in ["Csy4", "CasE", "PgU"]
]


def sort_output_edges(graph):
    """Sort incoming edges to output nodes alphabetically by protein name for deterministic ordering"""
    from biocomp.graphengine import GraphEdge

    output_nodes = [n for n in graph.nodes.values() if n.node_type == "output"]

    if not output_nodes:
        return graph

    for output_node in output_nodes:
        incoming_edges = graph.get_incoming_edges(output_node.node_id)

        # Sort edges by protein name (first part in content)
        sorted_edges = sorted(incoming_edges, key=lambda e: e.content[0].name if e.content else "")

        # Reassign input_slots to match sorted order
        for new_slot, edge in enumerate(sorted_edges):
            # Remove old edge
            old_key = (edge.source_id, edge.target_id, edge.from_output_slot, edge.to_input_slot)
            if old_key in graph.edges:
                del graph.edges[old_key]

            # Add edge with new input_slot
            new_edge = GraphEdge(
                source_id=edge.source_id,
                target_id=edge.target_id,
                from_output_slot=edge.from_output_slot,
                to_input_slot=new_slot,
                content=edge.content,
                content_type=edge.content_type,
                content_embedding_names=edge.content_embedding_names,
                extra=edge.extra,
            )
            new_key = (
                new_edge.source_id,
                new_edge.target_id,
                new_edge.from_output_slot,
                new_edge.to_input_slot,
            )
            graph.edges[new_key] = new_edge

    return graph


ALL_RULES = [
    merge_sources_by_id,
    create_aggregation_nodes,
    connect_sources_to_aggregation,
    merge_aggregators_by_group,
    # sort_aggregation_members,
    add_bias_nodes,  # Add bias nodes for cotx with fluo_bias
    add_numeric_nodes,  # Add numeric (copy number) nodes for regular cotx
    *SEQUESTRON_RULES,
]
