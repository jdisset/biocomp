from typing import (
    Optional,
    Literal,
    Union,
    Dict,
    List,
)
from pydantic import BaseModel
from copy import deepcopy
from jinja2 import Environment, BaseLoader, meta
from biocomp.graphrules import GraphRewritingRule, PropertyConstraint

# comp_g
# Index(['type', 'cdg_input', 'cdg_output', 'input_from', 'output_to',
#        'is_inverse_of', 'extra', 'source_id'],
#       dtype='object')
# cdg
# Index(['tu_id', 'type', 'predecessor', 'successor', 'content', 'content_type',
#        'params', 'is_output', 'is_input'],
#       dtype='object')


NodeType = Union[
    Literal[
        "output",
        "sequestron_ERN",
        "translation",
        "transcription",
        "source",
        "aggregation",
        "inv_aggregation",
        "inv_source",
        "inv_transcription",
        "inv_translation",
        "input",
    ],
    str,
]


class Part(BaseModel):
    name: str
    category: str


class GraphEdge(BaseModel):
    source_id: int
    target_id: int
    output_slot: int
    input_slot: int
    content: tuple[Part, ...]
    content_type: Optional[Literal["DNA", "RNA", "PRT"]] = None
    content_embedding_names: dict[str, tuple[str]] = {}  # 'tl_rate: ('0xUORF', '1xUORF')}


class InverseSpec(BaseModel):
    # inverse nodes always have only one input and one output
    # but we need to store the original output slot id so that we can use it
    # when converting aggregation nodes for example
    node_id: int
    output_slot: int
    output_len: int


class GraphNode(BaseModel):
    node_id: int
    node_type: NodeType
    is_inverse_of: Optional[InverseSpec] = None
    extra: dict = {}


class GraphState(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class GraphBuilder:
    def __init__(self, graph: GraphState):
        self.nodes: Dict[int, GraphNode] = {node.node_id: deepcopy(node) for node in graph.nodes}
        self.edges: List[GraphEdge] = [deepcopy(edge) for edge in graph.edges]
        self.next_id = max(self.nodes.keys(), default=-1) + 1

    def add_node(self, node_type: str, extra: Dict = None) -> int:
        node_id = self.next_id
        self.next_id += 1
        self.nodes[node_id] = GraphNode(node_id=node_id, node_type=node_type, extra=extra or {})
        return node_id

    def delete_node(self, node_id: int):
        if node_id in self.nodes:
            del self.nodes[node_id]
        self.edges = [e for e in self.edges if e.source_id != node_id and e.target_id != node_id]

    def add_edge(self, source_id: int, target_id: int, **properties):
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            output_slot=properties.get("output_slot", 0),
            input_slot=properties.get("input_slot", 0),
            content=properties.get("content", ()),
            content_type=properties.get("content_type"),
        )
        self.edges.append(edge)

    def delete_edge(self, source_id: int, target_id: int):
        self.edges = [
            e for e in self.edges if not (e.source_id == source_id and e.target_id == target_id)
        ]

    def set_node_properties(self, node_id: int, properties: Dict):
        if node_id in self.nodes:
            self.nodes[node_id].extra.update(properties)

    def rewire_edges_from(self, old_source_id: int, new_source_id: int):
        for edge in self.edges:
            if edge.source_id == old_source_id:
                edge.source_id = new_source_id

    def rewire_edges_to(self, old_target_id: int, new_target_id: int):
        for edge in self.edges:
            if edge.target_id == old_target_id:
                edge.target_id = new_target_id

    def build(self) -> GraphState:
        return GraphState(nodes=list(self.nodes.values()), edges=self.edges)


def match_properties(node: GraphNode, constraint: PropertyConstraint) -> bool:
    for key, expected in constraint.properties.items():
        if key == "type":
            if node.node_type != expected:
                return False
        else:
            if node.extra.get(key) != expected:
                return False
    return True


def has_edge_in_graph(
    source_node: GraphNode, target_node: GraphNode, edges: List[GraphEdge]
) -> bool:
    return any(
        edge.source_id == source_node.node_id and edge.target_id == target_node.node_id
        for edge in edges
    )


def find_matches(rule: GraphRewritingRule, target_graph: GraphState) -> List[Dict[str, GraphNode]]:
    node_vars = list(rule.query.bind.keys())
    if not node_vars:
        return []

    # query planning: pre-filter and sort by constraint
    candidates = {}
    for var_name, constraint in rule.query.bind.items():
        candidates[var_name] = [
            node for node in target_graph.nodes if match_properties(node, constraint)
        ]
    sorted_vars = sorted(node_vars, key=lambda v: len(candidates[v]))
    matches = []

    def check_constraints(assignment: Dict[str, GraphNode]) -> bool:
        for edge_constraint in rule.query.where_connected:
            if not has_edge_in_graph(
                assignment[edge_constraint.source_var],
                assignment[edge_constraint.target_var],
                target_graph.edges,
            ):
                return False

        for edge_constraint in rule.query.where_not_connected:
            if edge_constraint.source_var == "any":
                if any(
                    has_edge_in_graph(
                        other_node, assignment[edge_constraint.target_var], target_graph.edges
                    )
                    for other_node in target_graph.nodes
                    if other_node != assignment[edge_constraint.target_var]
                ):
                    return False
            elif edge_constraint.target_var == "any":
                if any(
                    has_edge_in_graph(
                        assignment[edge_constraint.source_var], other_node, target_graph.edges
                    )
                    for other_node in target_graph.nodes
                    if other_node != assignment[edge_constraint.source_var]
                ):
                    return False
            elif has_edge_in_graph(
                assignment[edge_constraint.source_var],
                assignment[edge_constraint.target_var],
                target_graph.edges,
            ):
                return False
        return True

    def backtrack(var_idx: int, assignment: Dict[str, GraphNode]):
        if var_idx == len(sorted_vars):
            if check_constraints(assignment):
                matches.append(assignment.copy())
            return

        var_name = sorted_vars[var_idx]
        for node in candidates[var_name]:
            if node in assignment.values():
                continue
            assignment[var_name] = node
            backtrack(var_idx + 1, assignment)
            del assignment[var_name]

    backtrack(0, {})
    return matches


_jinja_env = Environment(loader=BaseLoader())


class NodeProxy:
    """Proxy object to make GraphNode properties accessible to Jinja2 templates"""

    def __init__(self, node: GraphNode):
        self._node = node

    @property
    def type(self):
        return str(self._node.node_type)

    @property
    def node_id(self):
        return str(self._node.node_id)

    def __getattr__(self, name):
        """Get properties from node.extra dict"""
        return str(self._node.extra.get(name, ""))


def expand_template(template: str, match: Dict[str, GraphNode]) -> str:
    """Expand template string using Jinja2 with GraphNode properties"""
    if not isinstance(template, str):
        return str(template)
    context = {var_name: NodeProxy(node) for var_name, node in match.items()}
    jinja_template = _jinja_env.from_string(template)
    return jinja_template.render(**context)


def apply_actions(
    rule: GraphRewritingRule, matches: List[Dict[str, GraphNode]], target_graph: GraphState
) -> GraphState:
    builder = GraphBuilder(target_graph)
    applied_nodes = set()

    for match in matches:
        # check for overlapping matches (and skip if so)
        match_nodes = set(node.node_id for node in match.values())
        if match_nodes & applied_nodes:
            continue
        applied_nodes.update(match_nodes)

        var_to_node_id = {var: node.node_id for var, node in match.items()}
        local_nodes = {}

        for action in rule.actions:
            if action.action_type == "add_node":
                expanded_props = {}
                node_type = "unknown"
                for key, value in action.properties.items():
                    expanded_value = (
                        expand_template(str(value), match) if isinstance(value, str) else value
                    )
                    if key == "type":
                        node_type = expanded_value
                    else:
                        expanded_props[key] = expanded_value
                node_id = builder.add_node(node_type, expanded_props)
                local_nodes[action.local_name] = node_id

            elif action.action_type == "add_edge":
                source_id = local_nodes.get(action.source, var_to_node_id.get(action.source))
                target_id = local_nodes.get(action.target, var_to_node_id.get(action.target))
                builder.add_edge(source_id, target_id)

            elif action.action_type == "set_properties":
                node_id = local_nodes.get(action.node_var, var_to_node_id.get(action.node_var))
                expanded_props = {}
                for key, value in action.properties.items():
                    expanded_value = (
                        expand_template(str(value), match) if isinstance(value, str) else value
                    )
                    expanded_props[key] = expanded_value
                builder.set_node_properties(node_id, expanded_props)

            elif action.action_type == "delete_node":
                node_id = var_to_node_id[action.node_var]
                builder.delete_node(node_id)

            elif action.action_type == "delete_edge":
                source_id = var_to_node_id[action.source_var]
                target_id = var_to_node_id[action.target_var]
                builder.delete_edge(source_id, target_id)

            elif action.action_type == "rewire_edges_from":
                old_source_id = var_to_node_id[action.old_source_var]
                new_source_id = local_nodes.get(
                    action.new_source_var, var_to_node_id.get(action.new_source_var)
                )
                builder.rewire_edges_from(old_source_id, new_source_id)

            elif action.action_type == "rewire_edges_to":
                old_target_id = var_to_node_id[action.old_target_var]
                new_target_id = local_nodes.get(
                    action.new_target_var, var_to_node_id.get(action.new_target_var)
                )
                builder.rewire_edges_to(old_target_id, new_target_id)

    return builder.build()


def apply_rule(rule: GraphRewritingRule, graph: GraphState, **kw) -> list[GraphState]:
    if rule.run_until_stable:
        current_graph = graph
        while True:
            matches = find_matches(rule, current_graph)
            if not matches:
                break

            def get_match_key(match):
                return tuple(sorted([node.node_id for node in match.values()]))

            sorted_matches = sorted(matches, key=get_match_key)
            next_graph = apply_actions(rule, sorted_matches, current_graph)
            if len(next_graph.nodes) == len(current_graph.nodes) and len(next_graph.edges) == len(
                current_graph.edges
            ):
                break
            current_graph = next_graph
            # current_graph = apply_actions(rule, matches[:1], current_graph)
        return [current_graph]

    matches = find_matches(rule, graph)
    if not matches:
        return [graph]

    if rule.yield_strategy == "batched":
        return [apply_actions(rule, matches, graph)]
    else:  # per_match
        return [apply_actions(rule, [match], graph) for match in matches]
