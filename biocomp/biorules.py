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
)


merge_sources_by_id = GraphRewritingRule(
    name="merge_sources_by_id",
    query=MatchQuery(
        bind={
            "source1": PropertyConstraint(properties={"type": "source"}),
            "source2": PropertyConstraint(properties={"type": "source"}),
        },
        where_filter_function="source1.source_id == source2.source_id and source1.node_id != source2.node_id and source1.cotx_group == source2.cotx_group",
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
        where_filter_function="source1.cotx_group == source2.cotx_group and source1.node_id < source2.node_id",
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
                "cotx_group": "{{source1.cotx_group}}",
                "ratios": [],
                "members": [],
                "ratio_ranges": [],
                "fluo_bias": "{{ source1.fluo_bias if source1.fluo_bias else None }}",
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
        # Only connect to aggregation in same cotx group
        where_filter_function="source.cotx_group == aggregation.cotx_group and source.source_id not in aggregation.members",
        where_not_connected=[
            EdgeConstraint(source_var="aggregation", target_var="source"),
            EdgeConstraint(source_var="any", target_var="source", properties={"output_slot": None}),
        ],
    ),
    actions=[
        AddEdge(
            source="aggregation",
            target="source",
            properties={
                "content_type": None,
                "output_slot": "{{ len(aggregation.members) }}",
            },
        ),
        SetProperties(
            node_var="aggregation",
            properties={
                "ratios": "{{ aggregation.ratios + [source.ratio if source.ratio else 1.0] }}",
                "members": "{{ aggregation.members + [source.source_id] }}",
                "ratio_ranges": "{{ (aggregation.ratio_ranges if aggregation.ratio_ranges else []) + ([source.ratio_range] if source.ratio_range else [None]) }}",
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
        where_filter_function="agg1.cotx_group == agg2.cotx_group and agg1.node_id < agg2.node_id",
    ),
    actions=[
        RewireEdgesFrom(old_source_var="agg1", new_source_var="agg2"),
        SetProperties(
            node_var="agg2",
            properties={
                "ratios": "{{ agg1.ratios + agg2.ratios }}",
                "members": "{{ agg1.members + agg2.members }}",
                "ratio_ranges": "{{ (agg1.ratio_ranges if agg1.ratio_ranges else []) + (agg2.ratio_ranges if agg2.ratio_ranges else []) }}",
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
        where_filter_function="len(aggregation.members) > 1 and aggregation.members != sorted(aggregation.members)",
    ),
    actions=[
        SetProperties(
            node_var="aggregation",
            properties={
                # Sort members and reorder ratios and ratio_ranges to match
                "members": "{{ sorted(aggregation.members) }}",
                "ratios": "{{ reorder_list(aggregation.ratios, sorted_with_indices(aggregation.members)[1]) }}",
                "ratio_ranges": "{{ reorder_list(aggregation.ratio_ranges, sorted_with_indices(aggregation.members)[1]) }}",
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
        where_filter_function="source.source_id in aggregation.members",
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
                "output_slot": "{{ 0 if aggregation.members[0] == source.source_id else 1 }}",
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
        where_filter_function="top_node.type in ['source', 'aggregation'] and (not hasattr(top_node._obj.extra, 'get') or top_node._obj.extra.get('fluo_bias') is None or top_node._obj.extra.get('fluo_bias') == 'None')",
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
        where_filter_function="top_node.type in ['source', 'aggregation'] and hasattr(top_node._obj.extra, 'get') and top_node._obj.extra.get('fluo_bias') is not None and top_node._obj.extra.get('fluo_bias') != 'None'",
    ),
    actions=[
        AddNode(
            local_name="bias",
            properties={
                "type": "bias",
                "role": "fluo_bias",
                "tu_id": "{{ top_node.fluo_bias['tu_id'] }}",
                "value": "{{ top_node.fluo_bias['value'] }}",
                "protein": "{{ top_node.fluo_bias['protein'] if top_node.fluo_bias['protein'] else None }}",
                "units": "{{ top_node.fluo_bias['units'] if top_node.fluo_bias['units'] else 'AU' }}",
            },
        ),
        AddEdge(
            source="bias",
            target="top_node",
            properties={"content_type": None},  # Bias flow
        ),
    ],
    yield_strategy="batched",
)


def make_ern_rule(ern_name="CasE", ern_rec_name="CasE_rec"):
    return GraphRewritingRule(
        name=f"add_{ern_name.lower()}_sequestron",
        query=MatchQuery(
            bind_edges={
                "negative": EdgeConstraint(
                    properties={
                        "content_type": "PRT",
                    },
                    contains=[ern_name],
                ),
                "positive": EdgeConstraint(
                    properties={
                        "content_type": "RNA",
                    },
                    contains=[ern_rec_name],
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
            # Rewire the PRT edge (negative input): translation → sequestron (input_slot=0)
            EditEdge(edge_var="negative", target_var="sequestron", properties={"input_slot": 0}),
            CopyEdge(
                source_edge_var="positive", target_var="positive_target", source_var="sequestron"
            ),
            # Rewire the RNA edge (positive input): transcription → sequestron (input_slot=1)
            EditEdge(edge_var="positive", target_var="sequestron", properties={"input_slot": 1}),
        ],
        yield_strategy="batched",
        run_until_stable=True,
    )


SEQUESTRON_RULES = [
    make_ern_rule(ern_name=e, ern_rec_name=f"{e}_rec") for e in ["Csy4", "CasE", "PgU"]
]


# Inversion rules using the cartesian_product_by_key yield strategy
# This finds ALL invertible paths from ALL numeric nodes and creates
# one inverted network per combination of path selections (cartesian product)

invert_chain_with_aggregation = GraphRewritingRule(
    name="invert_chain_with_aggregation",
    query=MatchQuery(
        bind={
            "numeric": PropertyConstraint(properties={"type": "numeric"}),
            "agg": PropertyConstraint(properties={"type": "aggregation"}),
            "source": PropertyConstraint(properties={"type": "source"}),
            "tx": PropertyConstraint(properties={"type": "transcription"}),
            "tl": PropertyConstraint(properties={"type": "translation"}),
            "output": PropertyConstraint(properties={"type": "output"}),
        },
        where_connected=[
            EdgeConstraint(
                source_var="numeric", target_var="agg", properties={"content_type": None}
            ),
            EdgeConstraint(
                source_var="agg", target_var="source", properties={"content_type": None}
            ),
            EdgeConstraint(
                source_var="source", target_var="tx", properties={"content_type": "DNA"}
            ),
            EdgeConstraint(source_var="tx", target_var="tl", properties={"content_type": "RNA"}),
            EdgeConstraint(
                source_var="tl", target_var="output", properties={"content_type": "PRT"}
            ),
        ],
    ),
    actions=[
        AddNode(
            local_name="input",
            properties={
                "type": "input",
                "input_position": "{{__match_index__}}",
                "input_from_output": 0,
            },
        ),
        AddNode(
            local_name="inv_translation",
            properties={
                "type": "inv_translation",
                "is_inverse_of": {
                    "node_id": "{{tl.node_id}}",
                    "output_slot": 0,
                    "output_len": 1,
                },
            },
        ),
        AddNode(
            local_name="inv_transcription",
            properties={
                "type": "inv_transcription",
                "is_inverse_of": {
                    "node_id": "{{tx.node_id}}",
                    "output_slot": 0,
                    "output_len": 1,
                },
            },
        ),
        AddNode(
            local_name="inv_source",
            properties={
                "type": "inv_source",
                "is_inverse_of": {
                    "node_id": "{{source.node_id}}",
                    "output_slot": 0,
                    "output_len": 1,
                },
            },
        ),
        AddNode(
            local_name="inv_aggregation",
            properties={
                "type": "inv_aggregation",
                "is_inverse_of": {
                    "node_id": "{{agg.node_id}}",
                    "output_slot": 0,
                    "output_len": "{{len(agg.members) if agg.members else 1}}",
                },
            },
        ),
        AddEdge(
            source="input", target="inv_translation", properties={"output_slot": 0, "input_slot": 0}
        ),
        AddEdge(
            source="inv_translation",
            target="inv_transcription",
            properties={"output_slot": 0, "input_slot": 0},
        ),
        AddEdge(
            source="inv_transcription",
            target="inv_source",
            properties={"output_slot": 0, "input_slot": 0},
        ),
        AddEdge(
            source="inv_source",
            target="inv_aggregation",
            properties={"output_slot": 0, "input_slot": 0},
        ),
        AddEdge(
            source="inv_aggregation",
            target="source",
            properties={"output_slot": 0, "input_slot": 0},
        ),
        DeleteNode(node_var="numeric"),
    ],
    yield_strategy="cartesian_product_by_key",
    cartesian_product_key="numeric",
)

invert_chain_without_aggregation = GraphRewritingRule(
    name="invert_chain_without_aggregation",
    query=MatchQuery(
        bind={
            "numeric": PropertyConstraint(properties={"type": "numeric"}),
            "source": PropertyConstraint(properties={"type": "source"}),
            "tx": PropertyConstraint(properties={"type": "transcription"}),
            "tl": PropertyConstraint(properties={"type": "translation"}),
            "output": PropertyConstraint(properties={"type": "output"}),
        },
        where_connected=[
            EdgeConstraint(
                source_var="numeric", target_var="source", properties={"content_type": None}
            ),
            EdgeConstraint(
                source_var="source", target_var="tx", properties={"content_type": "DNA"}
            ),
            EdgeConstraint(source_var="tx", target_var="tl", properties={"content_type": "RNA"}),
            EdgeConstraint(
                source_var="tl", target_var="output", properties={"content_type": "PRT"}
            ),
        ],
    ),
    actions=[
        AddNode(
            local_name="input",
            properties={"type": "input", "input_position": 0, "input_from_output": 0},
        ),
        AddNode(
            local_name="inv_translation",
            properties={
                "type": "inv_translation",
                "is_inverse_of": {
                    "node_id": "{{tl.node_id}}",
                    "output_slot": 0,
                    "output_len": 1,
                },
            },
        ),
        AddNode(
            local_name="inv_transcription",
            properties={
                "type": "inv_transcription",
                "is_inverse_of": {
                    "node_id": "{{tx.node_id}}",
                    "output_slot": 0,
                    "output_len": 1,
                },
            },
        ),
        AddNode(
            local_name="inv_source",
            properties={
                "type": "inv_source",
                "is_inverse_of": {
                    "node_id": "{{source.node_id}}",
                    "output_slot": 0,
                    "output_len": 1,
                },
            },
        ),
        AddEdge(
            source="input", target="inv_translation", properties={"output_slot": 0, "input_slot": 0}
        ),
        AddEdge(
            source="inv_translation",
            target="inv_transcription",
            properties={"output_slot": 0, "input_slot": 0},
        ),
        AddEdge(
            source="inv_transcription",
            target="inv_source",
            properties={"output_slot": 0, "input_slot": 0},
        ),
        AddEdge(
            source="inv_source", target="source", properties={"output_slot": 0, "input_slot": 0}
        ),
        DeleteNode(node_var="numeric"),
    ],
    yield_strategy="cartesian_product_by_key",
    cartesian_product_key="numeric",
)

_deprecated_invert_chain_with_aggregation = GraphRewritingRule(
    name="invert_chain_with_aggregation",
    query=MatchQuery(
        bind={
            "numeric": PropertyConstraint(properties={"type": "numeric"}),
            "agg": PropertyConstraint(properties={"type": "aggregation"}),
            "source": PropertyConstraint(properties={"type": "source"}),
            "tx": PropertyConstraint(properties={"type": "transcription"}),
            "tl": PropertyConstraint(properties={"type": "translation"}),
            "output": PropertyConstraint(properties={"type": "output"}),
        },
        where_connected=[
            EdgeConstraint(
                source_var="numeric", target_var="agg", properties={"content_type": None}
            ),
            EdgeConstraint(
                source_var="agg", target_var="source", properties={"content_type": None}
            ),
            EdgeConstraint(
                source_var="source", target_var="tx", properties={"content_type": "DNA"}
            ),
            EdgeConstraint(source_var="tx", target_var="tl", properties={"content_type": "RNA"}),
            EdgeConstraint(
                source_var="tl", target_var="output", properties={"content_type": "PRT"}
            ),
        ],
    ),
    actions=[
        AddNode(
            local_name="input",
            properties={"type": "input", "input_position": 0, "input_from_output": 0},
        ),
        AddNode(
            local_name="inv_translation",
            properties={
                "type": "inv_translation",
                "is_inverse_of": {
                    "node_id": "{{tl.node_id}}",
                    "output_slot": 0,
                    "output_len": 1,
                },
            },
        ),
        AddNode(
            local_name="inv_transcription",
            properties={
                "type": "inv_transcription",
                "is_inverse_of": {
                    "node_id": "{{tx.node_id}}",
                    "output_slot": 0,
                    "output_len": 1,
                },
            },
        ),
        AddNode(
            local_name="inv_source",
            properties={
                "type": "inv_source",
                "is_inverse_of": {
                    "node_id": "{{source.node_id}}",
                    "output_slot": 0,
                    "output_len": 1,
                },
            },
        ),
        AddNode(
            local_name="inv_aggregation",
            properties={
                "type": "inv_aggregation",
                "is_inverse_of": {
                    "node_id": "{{agg.node_id}}",
                    "output_slot": 0,
                    "output_len": "{{len(agg.members) if agg.members else 1}}",
                },
            },
        ),
        AddEdge(
            source="input", target="inv_translation", properties={"output_slot": 0, "input_slot": 0}
        ),
        AddEdge(
            source="inv_translation",
            target="inv_transcription",
            properties={"output_slot": 0, "input_slot": 0},
        ),
        AddEdge(
            source="inv_transcription",
            target="inv_source",
            properties={"output_slot": 0, "input_slot": 0},
        ),
        AddEdge(
            source="inv_source",
            target="inv_aggregation",
            properties={"output_slot": 0, "input_slot": 0},
        ),
        AddEdge(
            source="inv_aggregation",
            target="source",
            properties={"output_slot": 0, "input_slot": 0},
        ),
        DeleteNode(node_var="numeric"),
    ],
    yield_strategy="per_match",
)

INVERSION_RULES = [
    invert_chain_with_aggregation,
    invert_chain_without_aggregation,
]


def sort_output_edges(graph):
    """Sort incoming edges to output nodes alphabetically by protein name for deterministic ordering"""
    from biocomp.graphengine import GraphState, GraphEdge

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
            old_key = (edge.source_id, edge.target_id, edge.output_slot, edge.input_slot)
            if old_key in graph.edges:
                del graph.edges[old_key]

            # Add edge with new input_slot
            new_edge = GraphEdge(
                source_id=edge.source_id,
                target_id=edge.target_id,
                output_slot=edge.output_slot,
                input_slot=new_slot,
                content=edge.content,
                content_type=edge.content_type,
                content_embedding_names=edge.content_embedding_names,
                extra=edge.extra,
            )
            new_key = (
                new_edge.source_id,
                new_edge.target_id,
                new_edge.output_slot,
                new_edge.input_slot,
            )
            graph.edges[new_key] = new_edge

    return graph


ALL_RULES = [
    merge_sources_by_id,
    create_aggregation_nodes,
    connect_sources_to_aggregation,
    merge_aggregators_by_group,
    sort_aggregation_members,
    add_bias_nodes,  # Add bias nodes for cotx with fluo_bias
    add_numeric_nodes,  # Add numeric (copy number) nodes for regular cotx
    *SEQUESTRON_RULES,
]
