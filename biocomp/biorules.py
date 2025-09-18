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
    RewireEdgesTo,
    SetProperties,
)
from biocomp.graphengine import GraphState


# Source merging rule - merge sources with same source_id
merge_sources_by_id = GraphRewritingRule(
    name="merge_sources_by_id",
    query=MatchQuery(
        bind={
            "source1": PropertyConstraint(properties={"type": "source"}),
            "source2": PropertyConstraint(properties={"type": "source"}),
        },
        where_filter_function="source1.source_id == source2.source_id and source1.node_id != source2.node_id",
    ),
    actions=[
        RewireEdgesFrom(old_source_var="source2", new_source_var="source1"),
        DeleteNode(node_var="source2"),
    ],
    yield_strategy="batched",
)

# Aggregation node creation rule - create aggregation for cotransfection groups with multiple sources
create_aggregation_nodes = GraphRewritingRule(
    name="create_aggregation_nodes",
    query=MatchQuery(
        bind={
            "source1": PropertyConstraint(properties={"type": "source"}),
            "source2": PropertyConstraint(properties={"type": "source"}),
        },
        # Only create aggregation when there are at least 2 different sources in the same cotx group
        where_filter_function="source1.cotx_group == source2.cotx_group and source1.node_id != source2.node_id",
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
            },
        ),
    ],
    yield_strategy="batched",
)

# Connect sources to aggregation nodes in the same cotx group
connect_sources_to_aggregation = GraphRewritingRule(
    name="connect_sources_to_aggregation",
    query=MatchQuery(
        bind={
            "source": PropertyConstraint(properties={"type": "source"}),
            "aggregation": PropertyConstraint(properties={"type": "aggregation"}),
        },
        where_filter_function="source.cotx_group == aggregation.cotx_group",
        where_not_connected=[EdgeConstraint(source_var="aggregation", target_var="source")],
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
            },
        ),
    ],
    yield_strategy="batched",
    run_until_stable=True,
)

# Merge aggregation nodes in the same cotransfection group
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
            },
        ),
        # Delete the now-redundant agg1
        DeleteNode(node_var="agg1"),
    ],
    run_until_stable=True,
    yield_strategy="batched",
)

# Numeric node creation rule - add numeric nodes for copy number inputs
add_numeric_nodes = GraphRewritingRule(
    name="add_numeric_nodes",
    query=MatchQuery(
        bind={
            "top_node": PropertyConstraint(properties={}),  # Any node type
        },
        # Match nodes with no incoming edges (top-level)
        where_not_connected=[EdgeConstraint(source_var="any", target_var="top_node")],
        # Only for source and aggregation nodes
        where_filter_function="top_node.type in ['source', 'aggregation']",
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


ALL_RULES = [
    merge_sources_by_id,
    create_aggregation_nodes,
    connect_sources_to_aggregation,
    merge_aggregators_by_group,
    add_numeric_nodes,
    *SEQUESTRON_RULES,
]
