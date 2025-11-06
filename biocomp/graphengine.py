from typing import Optional, Literal, Union, Dict, List, Any, Set, Tuple
from pydantic import BaseModel
from copy import deepcopy
from itertools import chain
from biocomp.graphrules import GraphRewritingRule, PropertyConstraint, EdgeConstraint
from collections import defaultdict, Counter


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


"""
Some notes on input/output slots (for me, mostly):
- nodes can have multiple inputs and output slots, which are basically "hubs" of unordered connections
- THERE IS NO ORDER GUARANTEE for edges connected to the same input/output slot other than "it's stable and reproducible for a given graph so long as node ids don't change"
- that's fine for some nodes like the transforms inputs (transcription/translation) or the negative and positive "input hubs" of ERNs
- some nodes like aggregation and inputs really do care about their output channel though so for those the edges are connected from a specific output slot
- output node cares about the input channel (because it maps to a consistent protein ordering)
"""


class Part(BaseModel):
    name: str
    category: str


class GraphEdge(BaseModel):
    source_id: int
    target_id: int
    from_output_slot: int
    to_input_slot: int
    content: tuple[Part, ...]
    content_type: Optional[Literal["DNA", "RNA", "PRT"]] = None
    # Embedding choices carried by the edge; allow multiple possible values per key
    content_embedding_names: dict[str, tuple[str, ...]] = {}
    extra: dict = {}


class InverseSpec(BaseModel):
    node_id: int
    output_slot: int
    output_len: int


class GraphNode(BaseModel):
    node_id: int
    node_type: NodeType
    is_inverse_of: Optional[InverseSpec] = None
    extra: dict = {}


class GraphState(BaseModel):
    nodes: dict[int, GraphNode]
    edges: dict[tuple[int, int, int, int], GraphEdge]
    # key for edges is (source_id, target_id, from_output_slot, to_input_slot)

    def get_node(self, node_id: int) -> Optional[GraphNode]:
        return self.nodes.get(node_id)

    def get_nodes_by_type(self, node_type: str) -> List[GraphNode]:
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def get_edge(
        self, source_id: int, target_id: int, from_output_slot: int = 0, to_input_slot: int = 0
    ) -> Optional[GraphEdge]:
        return self.edges.get((source_id, target_id, from_output_slot, to_input_slot))

    def get_outgoing_edges(self, node_id: int) -> list[GraphEdge]:
        """Returns the *sorted* list of outgoing edges from the given node."""
        o_edges = [e for e in self.edges.values() if e.source_id == node_id]
        o_edges.sort(key=lambda e: (e.target_id, e.from_output_slot, e.to_input_slot))
        return o_edges

    def get_incoming_edges(self, node_id: int) -> list[GraphEdge]:
        """Returns the *sorted* list of incoming edges to the given node."""
        i_edges = [e for e in self.edges.values() if e.target_id == node_id]
        i_edges.sort(key=lambda e: (e.source_id, e.from_output_slot, e.to_input_slot))
        return i_edges

    def get_downstream_nodes(self, node_id: int) -> List[Tuple[GraphNode, GraphEdge]]:
        connected = []
        for edge in self.get_outgoing_edges(node_id):
            target_node = self.get_node(edge.target_id)
            if target_node:
                connected.append((target_node, edge))
        return connected

    def get_upstream_nodes(self, node_id: int) -> List[Tuple[GraphNode, GraphEdge]]:
        connected = []
        for edge in self.get_incoming_edges(node_id):
            source_node = self.get_node(edge.source_id)
            if source_node:
                connected.append((source_node, edge))
        return connected

    def get_nb_outgoing_edges(self, node_id: int) -> int:
        return len(self.get_outgoing_edges(node_id))

    def get_nb_outgoing_slots(self, node_id: int) -> int:
        edges = self.get_outgoing_edges(node_id)
        return len(set(e.from_output_slot for e in edges))

    def get_nb_incoming_edges(self, node_id: int) -> int:
        return len(self.get_incoming_edges(node_id))

    def get_nb_incoming_slots(self, node_id: int) -> int:
        edges = self.get_incoming_edges(node_id)
        return len(set(e.to_input_slot for e in edges))


class GraphBuilder:
    def __init__(self, graph: GraphState):
        self.nodes: Dict[int, GraphNode] = {
            node_id: deepcopy(node) for node_id, node in graph.nodes.items()
        }
        self.edges: Dict[tuple[int, int, int, int], GraphEdge] = {
            (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): deepcopy(e)
            for e in graph.edges.values()
        }
        self.next_id = max(self.nodes.keys(), default=-1) + 1

    def add_node(
        self, node_type: str, extra: dict | None = None, is_inverse_of: Optional[InverseSpec] = None
    ) -> int:
        node_id = self.next_id
        self.next_id += 1
        self.nodes[node_id] = GraphNode(
            node_id=node_id, node_type=node_type, extra=extra or {}, is_inverse_of=is_inverse_of
        )
        return node_id

    def delete_node(self, node_id: int):
        if node_id in self.nodes:
            del self.nodes[node_id]
        self.edges = {
            k: e for k, e in self.edges.items() if e.source_id != node_id and e.target_id != node_id
        }

    def add_edge(self, source_id: int, target_id: int, **properties):
        from_output_slot = properties.get("from_output_slot", 0)
        to_input_slot = properties.get("to_input_slot", 0)

        edge_fields = {
            "source_id": source_id,
            "target_id": target_id,
            "from_output_slot": from_output_slot,
            "to_input_slot": to_input_slot,
            "content": properties.get("content", ()),
            "content_type": properties.get("content_type"),
            "content_embedding_names": properties.get("content_embedding_names", {}),
        }

        extra_props = {
            k: v
            for k, v in properties.items()
            if k
            not in {
                "from_output_slot",
                "to_input_slot",
                "content",
                "content_type",
                "content_embedding_names",
            }
        }
        edge_fields["extra"] = extra_props

        edge = GraphEdge(**edge_fields)
        edge_key = (source_id, target_id, from_output_slot, to_input_slot)
        self.edges[edge_key] = edge

    def delete_edge(self, source_id: int, target_id: int):
        self.edges = {
            k: e
            for k, e in self.edges.items()
            if not (e.source_id == source_id and e.target_id == target_id)
        }

    def set_node_properties(self, node_id: int, properties: Dict):
        if node_id in self.nodes:
            for key, value in properties.items():
                if isinstance(value, str):
                    try:
                        evaluated = eval(value)
                        if isinstance(evaluated, (list, dict)):
                            properties[key] = evaluated
                    except (SyntaxError, NameError):
                        pass
            self.nodes[node_id].extra.update(properties)

    def rewire_edges_from(self, old_source_id: int, new_source_id: int):
        to_rewire = [(k, e) for k, e in self.edges.items() if e.source_id == old_source_id]

        for old_key, edge in to_rewire:
            del self.edges[old_key]
            new_edge = GraphEdge(
                source_id=new_source_id,
                target_id=edge.target_id,
                from_output_slot=edge.from_output_slot,
                to_input_slot=edge.to_input_slot,
                content=edge.content,
                content_type=edge.content_type,
                content_embedding_names=edge.content_embedding_names,
                extra=edge.extra,
            )
            new_key = (new_source_id, edge.target_id, edge.from_output_slot, edge.to_input_slot)
            self.edges[new_key] = new_edge

    def rewire_edges_to(self, old_target_id: int, new_target_id: int):
        to_rewire = [(k, e) for k, e in self.edges.items() if e.target_id == old_target_id]

        for old_key, edge in to_rewire:
            del self.edges[old_key]
            new_edge = GraphEdge(
                source_id=edge.source_id,
                target_id=new_target_id,
                from_output_slot=edge.from_output_slot,
                to_input_slot=edge.to_input_slot,
                content=edge.content,
                content_type=edge.content_type,
                content_embedding_names=edge.content_embedding_names,
                extra=edge.extra,
            )
            new_key = (edge.source_id, new_target_id, edge.from_output_slot, edge.to_input_slot)
            self.edges[new_key] = new_edge

    def build(self) -> GraphState:
        return GraphState(nodes=self.nodes, edges=self.edges)


def match_properties_generic(
    obj: Any,
    properties: Dict[str, Any],
    special_cases: Optional[Dict[str, str]] = None,
    fallback_dict: Optional[str] = None,
) -> bool:
    for key, expected in properties.items():
        if special_cases and key in special_cases:
            if getattr(obj, special_cases[key]) != expected:
                return False
        elif key.startswith("content_has_") and hasattr(obj, "content"):
            attr = key[12:]
            if not any(getattr(part, attr, None) == expected for part in obj.content):
                return False
        elif hasattr(obj, key):
            if getattr(obj, key) != expected:
                return False
        elif fallback_dict and hasattr(obj, fallback_dict):
            fallback = getattr(obj, fallback_dict)
            if fallback.get(key) != expected:
                return False
        else:
            return False
    return True


def match_properties(node: GraphNode, constraint: PropertyConstraint) -> bool:
    return match_properties_generic(node, constraint.properties, {"type": "node_type"}, "extra")


def match_edge_properties(edge: GraphEdge, constraint: EdgeConstraint) -> bool:
    if not match_properties_generic(
        edge, constraint.properties, fallback_dict="content_embedding_names"
    ):
        return False

    if constraint.contains is not None:
        edge_part_names = {part.name for part in edge.content}
        required_parts = set(constraint.contains)
        if not required_parts.issubset(edge_part_names):
            return False

    return True


def _extract_node_ids(objects: List[Union[GraphNode, GraphEdge]]) -> set[int]:
    return {obj.node_id for obj in objects if isinstance(obj, GraphNode)}


def _connects_nodes(edge: GraphEdge, source_id: int, target_id: int) -> bool:
    return edge.source_id == source_id and edge.target_id == target_id


def find_edges_matching_constraint(
    edges: List[GraphEdge], constraint: EdgeConstraint, node_assignment: Dict[str, GraphNode]
) -> List[GraphEdge]:
    source_node = node_assignment.get(constraint.source_var) if constraint.source_var else None
    target_node = node_assignment.get(constraint.target_var) if constraint.target_var else None

    if constraint.source_var is not None and source_node is None:
        return []
    if constraint.target_var is not None and target_node is None:
        return []

    return [
        edge
        for edge in edges
        if (source_node is None or edge.source_id == source_node.node_id)
        and (target_node is None or edge.target_id == target_node.node_id)
        and match_edge_properties(edge, constraint)
    ]


def has_edge_in_graph(
    source_node: GraphNode, target_node: GraphNode, edges: List[GraphEdge]
) -> bool:
    return any(_connects_nodes(edge, source_node.node_id, target_node.node_id) for edge in edges)


def find_matches(
    rule: GraphRewritingRule, target_graph: GraphState, debug: bool = False
) -> List[Dict[str, Any]]:
    node_vars = list(rule.query.bind.keys())
    edge_vars = list(rule.query.bind_edges.keys())

    if not node_vars and not edge_vars:
        return []

    node_candidates = {}
    for var_name, constraint in rule.query.bind.items():
        node_candidates[var_name] = [
            node for node in target_graph.nodes.values() if match_properties(node, constraint)
        ]

    sorted_node_vars = sorted(node_vars, key=lambda v: len(node_candidates[v]))
    matches = []

    def check_constraints(assignment: Dict[str, GraphNode]) -> bool:
        def check_edge_exists(constraint, should_exist=True):
            source, target = constraint.source_var, constraint.target_var

            if source is None and target is None:
                matching_edges = [
                    edge
                    for edge in target_graph.edges.values()
                    if match_edge_properties(edge, constraint)
                ]
                return (len(matching_edges) > 0) == should_exist
            elif source is None:
                if target == "any":
                    # this case doesn't make much sense, but handle it gracefully
                    return should_exist
                return (
                    any(
                        edge.target_id == assignment[target].node_id
                        and match_edge_properties(edge, constraint)
                        for edge in target_graph.edges.values()
                    )
                    == should_exist
                )
            elif target is None:
                if source == "any":
                    return should_exist
                return (
                    any(
                        edge.source_id == assignment[source].node_id
                        and match_edge_properties(edge, constraint)
                        for edge in target_graph.edges.values()
                    )
                    == should_exist
                )
            elif source == "any":
                return (
                    any(
                        has_edge_in_graph(n, assignment[target], list(target_graph.edges.values()))
                        for n in target_graph.nodes.values()
                        if n != assignment[target]
                    )
                    == should_exist
                )
            elif target == "any":
                return (
                    any(
                        has_edge_in_graph(assignment[source], n, list(target_graph.edges.values()))
                        for n in target_graph.nodes.values()
                        if n != assignment[source]
                    )
                    == should_exist
                )
            else:
                return (
                    has_edge_in_graph(
                        assignment[source], assignment[target], list(target_graph.edges.values())
                    )
                    == should_exist
                )

        if rule.query.where_filter_function:
            if not eval(rule.query.where_filter_function, {}, assignment):
                return False

        return all(check_edge_exists(c, True) for c in rule.query.where_connected) and all(
            check_edge_exists(c, False) for c in rule.query.where_not_connected
        )

    def backtrack_nodes(var_idx: int, assignment: Dict[str, GraphNode]):
        if var_idx == len(sorted_node_vars):
            if check_constraints(assignment):
                backtrack_edges(0, assignment, {})
            return
        var_name = sorted_node_vars[var_idx]
        for node in node_candidates[var_name]:
            if node in assignment.values():
                continue
            assignment[var_name] = node
            backtrack_nodes(var_idx + 1, assignment)
            del assignment[var_name]

    def backtrack_edges(
        edge_idx: int, node_assignment: Dict[str, GraphNode], edge_assignment: Dict[str, GraphEdge]
    ):
        if edge_idx == len(edge_vars):
            full_assignment = {**node_assignment, **edge_assignment}

            for edge_var, edge in edge_assignment.items():
                edge_constraint = rule.query.bind_edges[edge_var]
                if edge_constraint.bind_endpoints:
                    source_node = target_graph.get_node(edge.source_id)
                    target_node = target_graph.get_node(edge.target_id)

                    full_assignment[f"{edge_var}_source"] = source_node
                    full_assignment[f"{edge_var}_target"] = target_node

            matches.append(full_assignment)
            return
        edge_var = edge_vars[edge_idx]
        edge_constraint = rule.query.bind_edges[edge_var]
        matching_edges = find_edges_matching_constraint(
            list(target_graph.edges.values()), edge_constraint, node_assignment
        )
        for edge in matching_edges:
            if edge in edge_assignment.values():
                continue
            edge_assignment[edge_var] = edge
            backtrack_edges(edge_idx + 1, node_assignment, edge_assignment)
            del edge_assignment[edge_var]

    if node_vars:
        backtrack_nodes(0, {})
    else:
        backtrack_edges(0, {}, {})

    if debug:
        print(f"\n>>> find_matches for rule '{rule.name}' found {len(matches)} match(es).")
        for i, match in enumerate(matches):
            print(f"  - Match #{i}:")
            for var_name, obj in match.items():
                if isinstance(obj, GraphNode):
                    print(
                        f"    {var_name}: Node(id={obj.node_id}, type='{obj.node_type}', extra={obj.extra})"
                    )
                elif isinstance(obj, GraphEdge):
                    print(f"    {var_name}: Edge(source={obj.source_id}, target={obj.target_id})")

    return matches


def sorted_with_indices(items):
    """Sort items and return the sorted list along with original indices."""
    indexed_items = list(enumerate(items))
    indexed_items.sort(key=lambda x: x[1])  # Sort by item value
    return [item for index, item in indexed_items], [index for index, item in indexed_items]


def reorder_list(source_list, indices):
    """Reorder source_list according to the given indices."""
    return [source_list[i] for i in indices]


def find_index(lst, item):
    """Find the index of item in list."""
    return lst.index(item)


def expand_template(template_str: str, match: Dict[str, Union[GraphNode, GraphEdge]]) -> Any:
    """Expand template strings, preserving types for simple expressions.

    Only supports simple expressions of the form "{{ expr }}" where the entire string
    is a single template expression. The expression is evaluated as Python code with
    match variables available. Multi-part string interpolation is NOT supported.

    Examples:
        "{{ node.node_type }}" -> evaluates to the node type
        "{{ len(node.extra.get('members', [])) }}" -> evaluates to integer
        "{{ node.extra.get('value') }}_suffix" -> NOT SUPPORTED (raises ValueError)
    """
    if not isinstance(template_str, str) or "{{" not in template_str:
        return template_str

    stripped = template_str.strip()
    if stripped.startswith("{{") and stripped.endswith("}}") and stripped.count("{{") == 1:
        expr = stripped[2:-2].strip()
        res = eval(expr, {**match})
        if "len" in expr:
            print(
                f"      Evaluated len() in template '{template_str}' -> {res}. Context is {match}"
            )
        return res
    else:
        raise ValueError(f"Unsupported template format: '{template_str}'")


# pyright: reportAttributeAccessIssue=false
def _process_match(
    match: Dict[str, Union[GraphNode, GraphEdge]],
    rule: GraphRewritingRule,
    builder: GraphBuilder,
    match_index: int = 0,
    debug: bool = False,
):
    if debug:
        print("\n--- Processing Match ---")
        for var, obj in match.items():
            if isinstance(obj, GraphNode):
                print(f"  {var}: Node(id={obj.node_id})")

    var_to_node_id = {var: obj.node_id for var, obj in match.items() if isinstance(obj, GraphNode)}
    local_nodes = {}

    # Add __match_index__ to the match context for templates
    match_with_index = dict(match)

    # Create a simple object to hold the index
    class IndexHolder:
        def __init__(self, idx):
            self.value = idx

        def __str__(self):
            return str(self.value)

    match_with_index["__match_index__"] = IndexHolder(match_index)  # type: ignore[assignment]

    def expand_props(props: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for k, v in props.items():
            if isinstance(v, dict):
                expanded = expand_props(v)
            else:
                expanded = expand_template(v, match_with_index)

            if debug:
                print(f"      Template '{v}' -> {expanded} (type: {type(expanded)})")
            result[k] = expanded
        return result

    def get_node_id(var: str) -> Optional[int]:
        return local_nodes.get(var) or var_to_node_id.get(var)

    for action in rule.actions:
        action_type = action.action_type
        if debug:
            print(f"  [Action] Executing '{action_type}'...")

        if action_type == "add_node":
            props = expand_props(action.properties or {})
            node_type = props.pop("type", "unknown")
            is_inverse_of_dict = props.pop("is_inverse_of", None)
            is_inverse_of = None
            if is_inverse_of_dict is not None:
                if isinstance(is_inverse_of_dict, dict):
                    is_inverse_of = InverseSpec(**is_inverse_of_dict)
                else:
                    is_inverse_of = is_inverse_of_dict
            node_id = builder.add_node(
                node_type, {k: v for k, v in props.items()}, is_inverse_of=is_inverse_of
            )
            local_nodes[action.local_name] = node_id
            if debug:
                print(
                    f"    Added Node '{action.local_name}' with ID {node_id} and properties {props}"
                )

        elif action_type == "add_edge":
            source_id, target_id = get_node_id(action.source), get_node_id(action.target)
            if source_id is not None and target_id is not None:
                props = expand_props(action.properties or {})
                builder.add_edge(source_id, target_id, **props)
                if debug:
                    print(
                        f"    Added Edge from '{action.source}' (ID: {source_id}) to '{action.target}' (ID: {target_id}) with properties {props}"
                    )

        elif action_type == "set_properties":
            node_id = get_node_id(action.node_var)
            if node_id is not None:
                props = expand_props(action.properties or {})
                if debug:
                    print(f"    Setting properties on '{action.node_var}' (ID: {node_id}): {props}")
                builder.set_node_properties(node_id, props)

        elif action_type == "delete_node":
            node_id = var_to_node_id[action.node_var]
            if debug:
                print(f"    Deleting Node '{action.node_var}' (ID: {node_id})")
            builder.delete_node(node_id)

        elif action_type == "delete_edge":
            source_id, target_id = (
                var_to_node_id[action.source_var],  # type: ignore[index]
                var_to_node_id[action.target_var],  # type: ignore[index]
            )
            if debug:
                print(
                    f"    Deleting Edge from '{action.source_var}' (ID: {source_id}) to '{action.target_var}' (ID: {target_id})"
                )
            builder.delete_edge(source_id, target_id)

        elif action_type == "rewire_edges_from":
            old_id = var_to_node_id[action.old_source_var]
            new_id = get_node_id(action.new_source_var)
            if new_id is not None:
                if debug:
                    print(
                        f"    Rewiring edges FROM '{action.old_source_var}' (ID: {old_id}) TO '{action.new_source_var}' (ID: {new_id})"
                    )
                builder.rewire_edges_from(old_id, new_id)

        elif action_type == "rewire_edges_to":
            old_id = var_to_node_id[action.old_target_var]
            new_id = get_node_id(action.new_target_var)
            if new_id is not None:
                if debug:
                    print(
                        f"    Rewiring edges TO '{action.old_target_var}' (ID: {old_id}) TO '{action.new_target_var}' (ID: {new_id})"
                    )
                builder.rewire_edges_to(old_id, new_id)

        elif action_type == "edit_edge":
            # Get the bound edge from the match
            if action.edge_var not in match:
                raise ValueError(f"Edge variable '{action.edge_var}' not found in match")
            edge = match[action.edge_var]
            if isinstance(edge, GraphEdge):
                new_source_id = edge.source_id
                new_target_id = edge.target_id

                if action.source_var is not None:
                    source_id = get_node_id(action.source_var)
                    if source_id is not None:
                        new_source_id = source_id

                if action.target_var is not None:
                    target_id = get_node_id(action.target_var)
                    if target_id is not None:
                        new_target_id = target_id

                new_extra = dict(edge.extra)
                new_output_slot = edge.from_output_slot
                new_input_slot = edge.to_input_slot
                if action.properties is not None:
                    expanded_props = expand_props(action.properties)
                    if "from_output_slot" in expanded_props:
                        new_output_slot = expanded_props.pop("from_output_slot")
                    if "to_input_slot" in expanded_props:
                        new_input_slot = expanded_props.pop("to_input_slot")
                    new_extra.update(expanded_props)

                new_content = edge.content
                if action.content is not None:
                    from biocomp.graphengine import Part

                    new_content = tuple(
                        Part(name=name, category="modified") for name in action.content
                    )

                if debug:
                    print(
                        f"    Editing Edge '{action.edge_var}': {edge.source_id}->{edge.target_id} to {new_source_id}->{new_target_id}"
                    )

                builder.delete_edge(edge.source_id, edge.target_id)

                builder.add_edge(
                    source_id=new_source_id,
                    target_id=new_target_id,
                    from_output_slot=new_output_slot,
                    to_input_slot=new_input_slot,
                    content=new_content,
                    content_type=edge.content_type,
                    content_embedding_names=edge.content_embedding_names,
                    **new_extra,
                )

        elif action_type == "copy_edge":
            if action.source_edge_var not in match:
                raise ValueError(
                    f"Source edge variable '{action.source_edge_var}' not found in match"
                )
            source_edge = match[action.source_edge_var]
            if isinstance(source_edge, GraphEdge):
                new_source_id = get_node_id(action.source_var)  # type: ignore[arg-type]
                new_target_id = get_node_id(action.target_var)  # type: ignore[arg-type]

                if new_source_id is None:
                    raise ValueError(
                        f"Source node variable '{action.source_var}' not found in match"
                    )
                if new_target_id is None:
                    raise ValueError(
                        f"Target node variable '{action.target_var}' not found in match"
                    )

                copied_extra = dict(source_edge.extra)

                if action.properties is not None:
                    expanded_props = expand_props(action.properties)
                    copied_extra.update(expanded_props)

                new_content = source_edge.content
                if action.content is not None:
                    from biocomp.graphengine import Part

                    new_content = tuple(
                        Part(name=name, category="copied") for name in action.content
                    )

                new_content_type = source_edge.content_type
                if action.content_type is not None:
                    new_content_type = action.content_type

                if debug:
                    print(
                        f"    Copying Edge '{action.source_edge_var}': {source_edge.source_id}->{source_edge.target_id} to {new_source_id}->{new_target_id}"
                    )

                builder.add_edge(
                    source_id=new_source_id,
                    target_id=new_target_id,
                    from_output_slot=source_edge.from_output_slot,
                    to_input_slot=source_edge.to_input_slot,
                    content=new_content,
                    content_type=new_content_type,
                    content_embedding_names=source_edge.content_embedding_names,
                    **copied_extra,
                )


def apply_actions(
    rule: GraphRewritingRule,
    matches: List[Dict[str, Union[GraphNode, GraphEdge]]],
    target_graph: GraphState,
    debug: bool = False,
) -> GraphState:
    builder = GraphBuilder(target_graph)
    applied_nodes = set()

    if debug and matches:
        _print_graph_summary(target_graph, "Graph State Before Actions")

    for match_idx, match in enumerate(matches):
        match_nodes = _extract_node_ids(list(match.values()))
        if match_nodes & applied_nodes:
            if debug:
                print(
                    f"Skipping match involving already processed nodes: {match_nodes & applied_nodes}"
                )
            continue
        applied_nodes.update(match_nodes)
        _process_match(match, rule, builder, match_index=match_idx, debug=debug)

    final_graph = builder.build()
    if debug and matches:
        _print_graph_summary(final_graph, "Graph State After Actions")

    return final_graph


def _print_graph_summary(graph: GraphState, message: str):
    print("\n" + "=" * 20 + f" {message} " + "=" * 20)
    print(f"Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}")
    source_nodes = [n for n in graph.nodes.values() if n.node_type == "source"]
    if source_nodes:
        print(f"Source Nodes ({len(source_nodes)}):")
        for node in source_nodes:
            source_id = node.extra.get("source_id", "N/A")
            tu_ids = node.extra.get("tu_id", "N/A")
            print(f"  - Node ID: {node.node_id}, source_id: '{source_id}', tu_ids: {tu_ids}")
    else:
        print("No 'source' nodes found.")
    print("=" * (42 + len(message)))


def apply_rule(
    rule: GraphRewritingRule, graph: GraphState, debug: bool = False, **kw
) -> list[GraphState]:
    if debug:
        print(f"\n\n{'#' * 25} APPLYING RULE: {rule.name.upper()} {'#' * 25}")
        if rule.run_until_stable:
            print("Mode: run_until_stable")
        _print_graph_summary(graph, "Initial Graph State")

    if rule.run_until_stable:
        current_graph = graph
        iteration = 1
        while True:
            if debug:
                print(f"\n--- Stable Iteration {iteration} ---")

            matches = find_matches(rule, current_graph, debug=debug)
            if not matches:
                if debug:
                    print("No more matches found. Rule is stable.")
                break

            next_graph = apply_actions(rule, matches, current_graph, debug=debug)

            if len(next_graph.nodes) == len(current_graph.nodes) and len(next_graph.edges) == len(
                current_graph.edges
            ):
                if debug:
                    print("Graph state is unchanged. Rule is stable.")
                break
            current_graph = next_graph
            iteration += 1

        final_graph = current_graph
        if debug:
            _print_graph_summary(final_graph, "Final Graph State After Stability")
        return [final_graph]

    matches = find_matches(rule, graph, debug=debug)
    if not matches:
        if debug:
            print("No matches found for this rule.")
        return [graph]

    if rule.yield_strategy == "batched":
        final_graph = apply_actions(rule, matches, graph, debug=debug)
        if debug:
            _print_graph_summary(final_graph, "Final Graph State")
        return [final_graph]
    elif rule.yield_strategy == "per_match":
        results = []
        for i, match in enumerate(matches):
            if debug:
                print(f"\n--- Applying rule per_match for Match #{i} ---")
            results.append(apply_actions(rule, [match], graph, debug=debug))
        return results
    elif rule.yield_strategy == "cartesian_product_by_key":
        if rule.cartesian_product_key is None:
            raise ValueError("cartesian_product_by_key requires cartesian_product_key to be set")

        from itertools import product
        from collections import defaultdict

        groups = defaultdict(list)
        for match in matches:
            key_obj = match.get(rule.cartesian_product_key)
            if key_obj is None:
                continue
            if isinstance(key_obj, GraphNode):
                key = key_obj.node_id
            else:
                key = str(key_obj)
            groups[key].append(match)

        if debug:
            print(
                f"\nGrouped {len(matches)} matches into {len(groups)} groups by key '{rule.cartesian_product_key}'"
            )
            for key, group_matches in groups.items():
                print(f"  Group {key}: {len(group_matches)} match(es)")

        if not groups:
            return [graph]

        group_keys = sorted(groups.keys())
        match_combinations = list(product(*[groups[k] for k in group_keys]))

        if debug:
            print(f"\nCartesian product produced {len(match_combinations)} combination(s)")

        results = []
        for i, combo in enumerate(match_combinations):
            if debug:
                print(f"\n--- Applying combination #{i + 1} with {len(combo)} matches ---")
            results.append(apply_actions(rule, list(combo), graph, debug=debug))

        return results
    else:
        raise ValueError(f"Unknown yield_strategy: {rule.yield_strategy}")


def apply_rule_sequence(
    rules: list[GraphRewritingRule],
    graphs: Union[GraphState, list[GraphState]],
    debug: bool = False,
) -> list[GraphState]:
    """Apply a sequence of rules to a list of graphs, returning all resulting graphs."""
    if not isinstance(graphs, list):
        graphs = [graphs]

    current_graphs = graphs
    for rule in rules:
        current_graphs = list(
            chain.from_iterable(apply_rule(rule, g, debug=debug) for g in current_graphs)
        )

    return current_graphs


def _make_hashable(obj):
    if isinstance(obj, list):
        return tuple(_make_hashable(item) for item in obj)
    elif isinstance(obj, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
    else:
        return obj


def graphs_are_isomorphic(
    graph1,
    graph2,
    compare_extra: bool = False,
    compare_content_embedding_names: bool = False,
    unordered_outgoing_types: set[str] | None = None,
    unordered_incoming_types: set[str] | None = None,
) -> bool:
    if len(graph1.nodes) != len(graph2.nodes) or len(graph1.edges) != len(graph2.edges):
        return False
    hash1 = _get_graph_canonical_hash(
        graph1,
        compare_extra,
        compare_content_embedding_names,
        unordered_outgoing_types,
        unordered_incoming_types,
    )
    hash2 = _get_graph_canonical_hash(
        graph2,
        compare_extra,
        compare_content_embedding_names,
        unordered_outgoing_types,
        unordered_incoming_types,
    )
    return hash1 == hash2


def _get_graph_canonical_hash(
    graph,
    compare_extra,
    compare_content_embedding_names,
    unordered_outgoing_types: set[str] | None = None,
    unordered_incoming_types: set[str] | None = None,
    iterations=5,
):
    node_hashes, _ = _get_canonical_invariants_for_diffing(
        graph,
        compare_extra,
        compare_content_embedding_names,
        unordered_outgoing_types,
        unordered_incoming_types,
        iterations,
    )
    return hash(tuple(sorted(node_hashes.values())))


def _get_canonical_edge_tuple(
    edge,
    compare_extra,
    compare_content_embedding_names,
    ignore_input_slot=False,
    ignore_output_slot=False,
):
    content_sig = tuple(sorted(p.name for p in edge.content)) if edge.content else ()
    parts = [
        edge.content_type,
        content_sig,
    ]
    if not ignore_output_slot:
        parts.append(edge.from_output_slot)
    if not ignore_input_slot:
        parts.append(edge.to_input_slot)
    if compare_content_embedding_names and edge.content_embedding_names:
        hashable_names = tuple(
            sorted((k, _make_hashable(v)) for k, v in edge.content_embedding_names.items())
        )
        parts.append(hashable_names)
    if compare_extra and edge.extra:
        hashable_extra = tuple(sorted((k, _make_hashable(v)) for k, v in edge.extra.items()))
        parts.append(hashable_extra)
    return tuple(parts)


def get_isomorphism_diff(
    graph1,
    graph2,
    compare_extra: bool = False,
    compare_content_embedding_names: bool = False,
    unordered_outgoing_types: set[str] | None = None,
    unordered_incoming_types: set[str] | None = None,
) -> Optional[str]:
    if len(graph1.nodes) != len(graph2.nodes):
        return (
            f"Graphs have a different number of nodes ({len(graph1.nodes)} vs {len(graph2.nodes)})."
        )
    if len(graph1.edges) != len(graph2.edges):
        return (
            f"Graphs have a different number of edges ({len(graph1.edges)} vs {len(graph2.edges)})."
        )
    if not graph1.nodes and not graph2.nodes:
        return None

    node_hashes1, descriptions1 = _get_canonical_invariants_for_diffing(
        graph1,
        compare_extra,
        compare_content_embedding_names,
        unordered_outgoing_types,
        unordered_incoming_types,
    )
    node_hashes2, descriptions2 = _get_canonical_invariants_for_diffing(
        graph2,
        compare_extra,
        compare_content_embedding_names,
        unordered_outgoing_types,
        unordered_incoming_types,
    )

    counts1 = Counter(node_hashes1.values())
    counts2 = Counter(node_hashes2.values())

    if counts1 == counts2:
        return None

    diff_lines = ["Graphs are not isomorphic. Differences in structural roles found:"]

    g1_minus_g2 = counts1 - counts2
    for h, count in sorted(g1_minus_g2.items(), key=lambda item: str(item[1])):
        desc = descriptions1.get(h, "Unknown Hash")
        plural = "s" if count > 1 else ""
        diff_lines.append(f"- Graph 1 has {count} extra instance{plural} of: {desc}")

    g2_minus_g1 = counts2 - counts1
    for h, count in sorted(g2_minus_g1.items(), key=lambda item: str(item[1])):
        desc = descriptions2.get(h, "Unknown Hash")
        plural = "s" if count > 1 else ""
        diff_lines.append(f"+ Graph 2 has {count} extra instance{plural} of: {desc}")

    return "\n".join(diff_lines)


def _get_canonical_invariants_for_diffing(
    graph,
    compare_extra: bool,
    compare_content_embedding_names: bool,
    unordered_outgoing_types: Optional[Set[str]] = None,
    unordered_incoming_types: Optional[Set[str]] = None,
    iterations=5,
) -> Tuple[Dict[Any, int], Dict[int, str]]:
    unordered_out = unordered_outgoing_types or set()
    unordered_in = unordered_incoming_types or set()

    out_edges = defaultdict(list)
    in_edges = defaultdict(list)
    for edge in graph.edges.values():
        out_edges[edge.source_id].append(edge)
        in_edges[edge.target_id].append(edge)

    nodes_by_id = graph.nodes
    node_hashes = {}

    for node in graph.nodes.values():
        parts = [node.node_type, node.is_inverse_of]
        if compare_extra and node.extra:
            hashable_extra = tuple(sorted((k, _make_hashable(v)) for k, v in node.extra.items()))
            parts.append(hashable_extra)
        node_hashes[node.node_id] = hash(tuple(parts))

    for _ in range(iterations):
        new_hashes = {}
        for node_id, current_hash in node_hashes.items():
            node = nodes_by_id[node_id]
            ignore_output_slots = node.node_type in unordered_out
            ignore_input_slots = node.node_type in unordered_in

            in_signatures = []
            for edge in in_edges.get(node_id, []):
                source_node = nodes_by_id[edge.source_id]
                should_ignore_output = source_node.node_type in unordered_out
                edge_tuple = _get_canonical_edge_tuple(
                    edge,
                    compare_extra,
                    compare_content_embedding_names,
                    ignore_input_slot=ignore_input_slots,
                    ignore_output_slot=should_ignore_output,
                )
                in_signatures.append((node_hashes[edge.source_id], edge_tuple))

            out_signatures = []
            for edge in out_edges.get(node_id, []):
                target_node = nodes_by_id[edge.target_id]
                should_ignore_input = target_node.node_type in unordered_in
                edge_tuple = _get_canonical_edge_tuple(
                    edge,
                    compare_extra,
                    compare_content_embedding_names,
                    ignore_input_slot=should_ignore_input,
                    ignore_output_slot=ignore_output_slots,
                )
                out_signatures.append((node_hashes[edge.target_id], edge_tuple))

            in_signatures.sort()
            out_signatures.sort()
            combined_signature = (current_hash, tuple(in_signatures), tuple(out_signatures))
            new_hashes[node_id] = hash(combined_signature)

        if node_hashes == new_hashes:
            break
        node_hashes = new_hashes

    hash_to_description = {}
    hash_to_node_id = {h: node_id for node_id, h in node_hashes.items()}

    for h, node_id in hash_to_node_id.items():
        node = nodes_by_id[node_id]

        in_neighbor_types = sorted(
            [nodes_by_id[edge.source_id].node_type for edge in in_edges.get(node_id, [])]
        )
        out_neighbor_types = sorted(
            [nodes_by_id[edge.target_id].node_type for edge in out_edges.get(node_id, [])]
        )

        in_counts = Counter(in_neighbor_types)
        out_counts = Counter(out_neighbor_types)

        in_desc = f"from {dict(in_counts)}" if in_counts else "from nowhere"
        out_desc = f"to {dict(out_counts)}" if out_counts else "to nowhere"

        desc = f"Node(type='{node.node_type}', receives {in_desc}, sends {out_desc})"
        hash_to_description[h] = desc

    return node_hashes, hash_to_description


##────────────────────────────────────────────────────────────────────────────}}}
