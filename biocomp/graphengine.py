from typing import Optional, Literal, Union, Dict, List, Any
from pydantic import BaseModel
from copy import deepcopy
from itertools import chain
from jinja2 import Environment, BaseLoader
from biocomp.graphrules import GraphRewritingRule, PropertyConstraint, EdgeConstraint


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
        edge_fields = {
            "source_id": source_id,
            "target_id": target_id,
            "output_slot": properties.get("output_slot", 0),
            "input_slot": properties.get("input_slot", 0),
            "content": properties.get("content", ()),
            "content_type": properties.get("content_type"),
            "content_embedding_names": properties.get("content_embedding_names", {}),
        }

        extra_props = {
            k: v
            for k, v in properties.items()
            if k
            not in {
                "output_slot",
                "input_slot",
                "content",
                "content_type",
                "content_embedding_names",
            }
        }
        edge_fields["extra"] = extra_props

        edge = GraphEdge(**edge_fields)
        self.edges.append(edge)

    def delete_edge(self, source_id: int, target_id: int):
        self.edges = [
            e for e in self.edges if not (e.source_id == source_id and e.target_id == target_id)
        ]

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

    def rewire_edges(self, old_id: int, new_id: int, attr: str):
        for edge in self.edges:
            if getattr(edge, attr) == old_id:
                setattr(edge, attr, new_id)

    def rewire_edges_from(self, old_source_id: int, new_source_id: int):
        self.rewire_edges(old_source_id, new_source_id, "source_id")

    def rewire_edges_to(self, old_target_id: int, new_target_id: int):
        self.rewire_edges(old_target_id, new_target_id, "target_id")

    def build(self) -> GraphState:
        return GraphState(nodes=list(self.nodes.values()), edges=self.edges)


def match_properties_generic(
    obj: Any,
    properties: Dict[str, Any],
    special_cases: Dict[str, str] = None,
    fallback_dict: str = None,
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
) -> List[Dict[str, any]]:
    node_vars = list(rule.query.bind.keys())
    edge_vars = list(rule.query.bind_edges.keys())

    if not node_vars and not edge_vars:
        return []

    node_candidates = {}
    for var_name, constraint in rule.query.bind.items():
        node_candidates[var_name] = [
            node for node in target_graph.nodes if match_properties(node, constraint)
        ]

    sorted_node_vars = sorted(node_vars, key=lambda v: len(node_candidates[v]))
    matches = []

    def check_constraints(assignment: Dict[str, GraphNode]) -> bool:
        def check_edge_exists(constraint, should_exist=True):
            source, target = constraint.source_var, constraint.target_var

            if source is None and target is None:
                matching_edges = [
                    edge for edge in target_graph.edges if match_edge_properties(edge, constraint)
                ]
                return (len(matching_edges) > 0) == should_exist
            elif source is None:
                if target == "any":
                    # this case doesn't make much sense, but handle it gracefully
                    return True == should_exist
                return (
                    any(
                        edge.target_id == assignment[target].node_id
                        and match_edge_properties(edge, constraint)
                        for edge in target_graph.edges
                    )
                    == should_exist
                )
            elif target is None:
                if source == "any":
                    return True == should_exist
                return (
                    any(
                        edge.source_id == assignment[source].node_id
                        and match_edge_properties(edge, constraint)
                        for edge in target_graph.edges
                    )
                    == should_exist
                )
            elif source == "any":
                return (
                    any(
                        has_edge_in_graph(n, assignment[target], target_graph.edges)
                        for n in target_graph.nodes
                        if n != assignment[target]
                    )
                    == should_exist
                )
            elif target == "any":
                return (
                    any(
                        has_edge_in_graph(assignment[source], n, target_graph.edges)
                        for n in target_graph.nodes
                        if n != assignment[source]
                    )
                    == should_exist
                )
            else:
                return (
                    has_edge_in_graph(assignment[source], assignment[target], target_graph.edges)
                    == should_exist
                )

        if rule.query.where_filter_function:
            context = {var: NodeProxy(node) for var, node in assignment.items()}
            if not eval(rule.query.where_filter_function, {}, context):
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

            # add automatic endpoint bindings for edges with bind_endpoints=True
            for edge_var, edge in edge_assignment.items():
                edge_constraint = rule.query.bind_edges[edge_var]
                if edge_constraint.bind_endpoints:
                    source_node = next(
                        node for node in target_graph.nodes if node.node_id == edge.source_id
                    )
                    target_node = next(
                        node for node in target_graph.nodes if node.node_id == edge.target_id
                    )

                    full_assignment[f"{edge_var}_source"] = source_node
                    full_assignment[f"{edge_var}_target"] = target_node

            matches.append(full_assignment)
            return
        edge_var = edge_vars[edge_idx]
        edge_constraint = rule.query.bind_edges[edge_var]
        matching_edges = find_edges_matching_constraint(
            target_graph.edges, edge_constraint, node_assignment
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


_jinja_env = Environment(loader=BaseLoader())
_jinja_env.globals["len"] = len


class TemplateProxy:
    def __init__(self, obj: Any, attr_map: Dict[str, str] = None, fallback_dict: str = None):
        self._obj, self._attr_map, self._fallback = obj, attr_map or {}, fallback_dict

    def __getattr__(self, name):
        if hasattr(self._obj, name):
            return getattr(self._obj, name)
        if name in self._attr_map and hasattr(self._obj, self._attr_map[name]):
            return getattr(self._obj, self._attr_map[name])
        if self._fallback and hasattr(self._obj, self._fallback):
            fallback_data = getattr(self._obj, self._fallback)
            if isinstance(fallback_data, dict) and name in fallback_data:
                return fallback_data.get(name)
        return None


class NodeProxy(TemplateProxy):
    def __init__(self, node: GraphNode):
        super().__init__(node, {"type": "node_type"}, "extra")


class EdgeProxy(TemplateProxy):
    def __init__(self, edge: GraphEdge):
        super().__init__(edge, fallback_dict="content_embedding_names")

    @property
    def content(self):
        return [part.name for part in self._obj.content]


def expand_template(template_str: str, match: Dict[str, Union[GraphNode, GraphEdge]]) -> Any:
    if not isinstance(template_str, str) or "{{" not in template_str:
        return template_str

    if "+" in template_str and all(
        p.strip().split(".")[0] in match
        for p in template_str.replace("{{", "").replace("}}", "").split("+")
    ):
        parts = [p.strip() for p in template_str.replace("{{", "").replace("}}", "").split("+")]
        result = []
        for part_str in parts:
            var, attr = part_str.split(".")
            proxy = (
                NodeProxy(match[var])
                if isinstance(match[var], GraphNode)
                else EdgeProxy(match[var])
            )
            # First try to get raw value directly from the object
            raw_value = getattr(proxy._obj, attr, None)
            if raw_value is not None:
                val = raw_value
            else:
                # Fall back to proxy (which might do string conversion)
                val = getattr(proxy, attr)

            # If the value is a string that looks like a list, try to parse it
            if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
                try:
                    parsed_val = eval(val)
                    if isinstance(parsed_val, list):
                        val = parsed_val
                except:
                    pass

            if isinstance(val, list):
                result.extend(val)
            else:
                result.append(val)
        return result

    # Check if template is a simple variable access like "{{var.attr}}"
    # If so, return the value directly without string conversion
    simple_var_pattern = template_str.strip()
    if simple_var_pattern.startswith("{{") and simple_var_pattern.endswith("}}"):
        var_expr = simple_var_pattern[2:-2].strip()
        if "." in var_expr and var_expr.count(".") == 1:
            var_name, attr_name = var_expr.split(".", 1)
            if var_name in match:
                proxy = (
                    NodeProxy(match[var_name])
                    if isinstance(match[var_name], GraphNode)
                    else EdgeProxy(match[var_name])
                )
                raw_value = getattr(proxy._obj, attr_name, None)
                if raw_value is not None:
                    return raw_value

    context = {
        var_name: NodeProxy(obj) if isinstance(obj, GraphNode) else EdgeProxy(obj)
        for var_name, obj in match.items()
    }
    jinja_template = _jinja_env.from_string(template_str)
    rendered = jinja_template.render(**context)
    return rendered


def _process_match(
    match: Dict[str, Union[GraphNode, GraphEdge]],
    rule: GraphRewritingRule,
    builder: GraphBuilder,
    debug: bool = False,
):
    if debug:
        print(f"\n--- Processing Match ---")
        for var, obj in match.items():
            if isinstance(obj, GraphNode):
                print(f"  {var}: Node(id={obj.node_id})")

    var_to_node_id = {var: obj.node_id for var, obj in match.items() if isinstance(obj, GraphNode)}
    local_nodes = {}

    def expand_props(props: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for k, v in props.items():
            expanded = expand_template(v, match)
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
            props = expand_props(action.properties)
            node_id = builder.add_node(
                props.pop("type", "unknown"), {k: v for k, v in props.items() if k != "type"}
            )
            local_nodes[action.local_name] = node_id
            if debug:
                print(
                    f"    Added Node '{action.local_name}' with ID {node_id} and properties {props}"
                )

        elif action_type == "add_edge":
            source_id, target_id = get_node_id(action.source), get_node_id(action.target)
            if source_id is not None and target_id is not None:
                props = expand_props(action.properties)
                builder.add_edge(source_id, target_id, **props)
                if debug:
                    print(
                        f"    Added Edge from '{action.source}' (ID: {source_id}) to '{action.target}' (ID: {target_id}) with properties {props}"
                    )

        elif action_type == "set_properties":
            node_id = get_node_id(action.node_var)
            if node_id is not None:
                props = expand_props(action.properties)
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
                var_to_node_id[action.source_var],
                var_to_node_id[action.target_var],
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

                # Prepare new extra properties (preserve existing ones)
                new_extra = dict(edge.extra)
                new_output_slot = edge.output_slot
                new_input_slot = edge.input_slot
                if action.properties is not None:
                    expanded_props = expand_props(action.properties)
                    if "output_slot" in expanded_props:
                        new_output_slot = expanded_props.pop("output_slot")
                    if "input_slot" in expanded_props:
                        new_input_slot = expanded_props.pop("input_slot")
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
                    output_slot=new_output_slot,
                    input_slot=new_input_slot,
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
                new_source_id = get_node_id(action.source_var)
                new_target_id = get_node_id(action.target_var)

                if new_source_id is None:
                    raise ValueError(
                        f"Source node variable '{action.source_var}' not found in match"
                    )
                if new_target_id is None:
                    raise ValueError(
                        f"Target node variable '{action.target_var}' not found in match"
                    )

                copied_extra = dict(source_edge.extra)

                # Add/override with new properties if specified
                if action.properties is not None:
                    expanded_props = expand_props(action.properties)
                    copied_extra.update(expanded_props)

                # Use copied content or override with new content
                new_content = source_edge.content
                if action.content is not None:
                    from biocomp.graphengine import Part

                    new_content = tuple(
                        Part(name=name, category="copied") for name in action.content
                    )

                # use copied content_type or override with new content_type
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
                    output_slot=source_edge.output_slot,
                    input_slot=source_edge.input_slot,
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

    for match in matches:
        match_nodes = _extract_node_ids(list(match.values()))
        if match_nodes & applied_nodes:
            if debug:
                print(
                    f"Skipping match involving already processed nodes: {match_nodes & applied_nodes}"
                )
            continue
        applied_nodes.update(match_nodes)
        _process_match(match, rule, builder, debug=debug)

    final_graph = builder.build()
    if debug and matches:
        _print_graph_summary(final_graph, "Graph State After Actions")

    return final_graph


def _print_graph_summary(graph: GraphState, message: str):
    print("\n" + "=" * 20 + f" {message} " + "=" * 20)
    print(f"Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}")
    source_nodes = [n for n in graph.nodes if n.node_type == "source"]
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
    else:  # per_match
        results = []
        for i, match in enumerate(matches):
            if debug:
                print(f"\n--- Applying rule per_match for Match #{i} ---")
            results.append(apply_actions(rule, [match], graph, debug=debug))
        return results


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
        # Apply the current rule to every graph produced by the previous step.
        # The result is a flattened list of all new graphs.
        current_graphs = list(
            chain.from_iterable(apply_rule(rule, g, debug=debug) for g in current_graphs)
        )

    return current_graphs


## {{{                 --     graph isomorphism     --


def graphs_are_isomorphic(
    graph1,
    graph2,
    compare_extra: bool = False,  # whether to compare node.extra and edge.extra fields (default: False)
    compare_content_embedding_names: bool = False,  # whether to compare edge.content_embedding_names (default: False)
) -> bool:
    """
    Graph isomorphism check using iterative canonical hashing (Weisfeiler-Lehman).
    """
    if len(graph1.nodes) != len(graph2.nodes) or len(graph1.edges) != len(graph2.edges):
        return False

    return _get_graph_canonical_hash(
        graph1, compare_extra, compare_content_embedding_names
    ) == _get_graph_canonical_hash(graph2, compare_extra, compare_content_embedding_names)


def _get_graph_canonical_hash(graph, compare_extra, compare_content_embedding_names, iterations=5):
    from collections import defaultdict

    out_edges = defaultdict(list)
    in_edges = defaultdict(list)
    for edge in graph.edges:
        out_edges[edge.source_id].append(edge)
        in_edges[edge.target_id].append(edge)

    node_hashes = {}
    for node in graph.nodes:
        parts = [node.node_type, node.is_inverse_of]
        if compare_extra and node.extra:
            parts.append(tuple(sorted(node.extra.items())))
        node_hashes[node.node_id] = hash(tuple(parts))

    for _ in range(iterations):
        new_hashes = {}
        for node_id, current_hash in node_hashes.items():
            in_signatures = []
            for edge in in_edges[node_id]:
                edge_tuple = _get_canonical_edge_tuple(
                    edge, compare_extra, compare_content_embedding_names
                )
                in_signatures.append((node_hashes[edge.source_id], edge_tuple))

            out_signatures = []
            for edge in out_edges[node_id]:
                edge_tuple = _get_canonical_edge_tuple(
                    edge, compare_extra, compare_content_embedding_names
                )
                out_signatures.append((node_hashes[edge.target_id], edge_tuple))

            in_signatures.sort()
            out_signatures.sort()

            combined_signature = (current_hash, tuple(in_signatures), tuple(out_signatures))
            new_hashes[node_id] = hash(combined_signature)

        node_hashes = new_hashes

    final_graph_hash = hash(tuple(sorted(node_hashes.values())))
    return final_graph_hash


def _get_canonical_edge_tuple(edge, compare_extra, compare_content_embedding_names):
    content_sig = tuple(sorted((p.name, p.category) for p in edge.content)) if edge.content else ()

    parts = [
        edge.content_type,
        content_sig,
        edge.output_slot,
        edge.input_slot,
    ]

    if compare_content_embedding_names and edge.content_embedding_names:
        parts.append(tuple(sorted(edge.content_embedding_names.items())))
    if compare_extra and edge.extra:
        parts.append(tuple(sorted(edge.extra.items())))

    return tuple(parts)


##────────────────────────────────────────────────────────────────────────────}}
