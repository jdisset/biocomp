from biocomp.graphrules import (
    GraphRewritingRule,
    MatchQuery,
    PropertyConstraint,
    EdgeConstraint,
    AddNode,
    AddEdge,
    DeleteNode,
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
        where_filter_function="source1.extra.get('cotx_group') == source2.extra.get('cotx_group') and source1.node_id < source2.node_id",
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
                "members": {},
                "fluo_bias": "{{ source1.extra.get('fluo_bias') if source1.extra.get('fluo_bias') else None }}",
            },
        ),
    ],
    yield_strategy="batched",
)

create_aggregation_nodes_single = GraphRewritingRule(
    name="create_aggregation_nodes_single",
    query=MatchQuery(
        bind={
            "source": PropertyConstraint(properties={"type": "source"}),
        },
        where_not_connected=[EdgeConstraint(source_var="any", target_var="source")],
        where_filter_function="source.extra.get('cotx_group') is not None",
    ),
    actions=[
        AddNode(
            local_name="aggregation",
            properties={
                "type": "aggregation",
                "cotx_group": "{{source.extra.get('cotx_group')}}",
                "members": {},
                "fluo_bias": "{{ source.extra.get('fluo_bias') if source.extra.get('fluo_bias') else None }}",
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
        where_filter_function="source.extra.get('cotx_group') == aggregation.extra.get('cotx_group') and source.extra.get('source_id') not in aggregation.extra.get('members', {})",
        where_not_connected=[EdgeConstraint(target_var="source")],
    ),
    actions=[
        AddEdge(
            source="aggregation",
            target="source",
            properties={
                "content_type": None,
                "from_output_slot": "{{ len(aggregation.extra.get('members', {})) }}",
                "tu_id": "{{ [source.extra.get('name', '') + '_' + source.extra.get('cotx_group', '')] if source.extra.get('name') else [] }}",
            },
        ),
        SetProperties(
            node_var="aggregation",
            properties={
                "members": "{{ {**aggregation.extra.get('members', {}), source.extra.get('source_id'): {'ratio': source.extra.get('ratio', 1.0), 'ratio_range': source.extra.get('ratio_range'), 'locked': source.extra.get('ratio_locked', False)}} }}",
            },
        ),
    ],
    yield_strategy="per_match",
    run_until_stable=True,
)

merge_aggregators_by_group = GraphRewritingRule(
    name="merge_aggregators_by_group",
    query=MatchQuery(
        bind={
            "agg1": PropertyConstraint(properties={"type": "aggregation"}),
            "agg2": PropertyConstraint(properties={"type": "aggregation"}),
        },
        where_filter_function="agg1.extra.get('cotx_group') == agg2.extra.get('cotx_group') and agg1.node_id < agg2.node_id",
    ),
    actions=[
        RewireEdgesFrom(old_source_var="agg1", new_source_var="agg2"),
        SetProperties(
            node_var="agg2",
            properties={
                "members": "{{ {**agg1.extra.get('members', {}), **agg2.extra.get('members', {})} }}",
            },
        ),
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
        where_filter_function="isinstance(aggregation.extra.get('members'), list) and len(aggregation.extra.get('members', [])) > 1",
    ),
    actions=[
        SetProperties(
            node_var="aggregation",
            properties={
                "members": "{{ {m: {'ratio': aggregation.extra.get('ratios', [1.0]*len(aggregation.extra.get('members', [])))[i], 'ratio_range': (aggregation.extra.get('ratio_ranges') or [None]*len(aggregation.extra.get('members', [])))[i], 'locked': (aggregation.extra.get('ratio_locked') or [False]*len(aggregation.extra.get('members', [])))[i]} for i, m in enumerate(aggregation.extra.get('members', []))} }}",
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


def sort_aggregation_edges(graph):
    """Ensure edge from_output_slot matches sorted(members.keys()) order."""
    from biocomp.graphengine import GraphEdge

    for agg_node in (n for n in graph.nodes.values() if n.node_type == "aggregation"):
        members = agg_node.extra.get("members", {})

        if isinstance(members, list):
            ratios = agg_node.extra.get("ratios", [1.0] * len(members))
            ratio_ranges = agg_node.extra.get("ratio_ranges", [None] * len(members))
            ratio_locked = agg_node.extra.get("ratio_locked", [False] * len(members))
            members = {
                m: {"ratio": ratios[i] if i < len(ratios) else 1.0,
                    "ratio_range": ratio_ranges[i] if i < len(ratio_ranges) else None,
                    "locked": ratio_locked[i] if i < len(ratio_locked) else False}
                for i, m in enumerate(members)
            }
            agg_node.extra["members"] = members

        if len(members) <= 1:
            continue

        sorted_ids = sorted(members.keys())

        for edge in graph.get_outgoing_edges(agg_node.node_id):
            target = graph.nodes.get(edge.target_id)
            if target is None or target.node_type != "source":
                continue

            source_id = target.extra.get("source_id")
            if source_id not in sorted_ids:
                continue

            new_slot = sorted_ids.index(source_id)
            if edge.from_output_slot == new_slot:
                continue

            old_key = (edge.source_id, edge.target_id, edge.from_output_slot, edge.to_input_slot)
            if old_key in graph.edges:
                del graph.edges[old_key]

            new_edge = GraphEdge(
                source_id=edge.source_id,
                target_id=edge.target_id,
                from_output_slot=new_slot,
                to_input_slot=edge.to_input_slot,
                content=edge.content,
                content_type=edge.content_type,
                content_embedding_names=edge.content_embedding_names,
                extra=edge.extra,
            )
            graph.edges[(new_edge.source_id, new_edge.target_id, new_edge.from_output_slot, new_edge.to_input_slot)] = new_edge

    return graph


ALL_RULES = [
    merge_sources_by_id,
    create_aggregation_nodes,
    connect_sources_to_aggregation,
    create_aggregation_nodes_single,  # Single-source cotx groups (remaining after above)
    connect_sources_to_aggregation,  # Connect single sources to their new aggregation
    merge_aggregators_by_group,
    add_bias_nodes,  # Add bias nodes for cotx with fluo_bias
    add_numeric_nodes,  # Add numeric (copy number) nodes for regular cotx
    *SEQUESTRON_RULES,
]
