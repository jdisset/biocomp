from itertools import product
from biocomp.graphengine import GraphState, GraphBuilder, InverseSpec


def invert_all_paths(graph: GraphState, mode: str = "shortest") -> list[GraphState]:
    numeric_nodes = [n for n in graph.nodes.values() if n.node_type == "numeric"]
    if not numeric_nodes:
        return [graph]

    nodes_by_id = graph.nodes
    outgoing = {n.node_id: [] for n in graph.nodes.values()}
    incoming_count = {n.node_id: 0 for n in graph.nodes.values()}

    for edge in graph.edges.values():
        outgoing.setdefault(edge.source_id, []).append((edge.target_id, edge.output_slot, edge.input_slot))
        incoming_count[edge.target_id] += 1

    invertible_types = {"numeric", "aggregation", "source", "transcription", "translation"}
    is_invertible = lambda nid: nodes_by_id[nid].node_type in invertible_types and incoming_count[nid] <= 1

    def find_paths(start_id):
        paths = []
        def dfs(nid, path, visited):
            node = nodes_by_id.get(nid)
            if not node:
                return
            if node.node_type == "output":
                if path:
                    for e in graph.edges.values():
                        if e.source_id == path[-1][0] and e.target_id == nid:
                            paths.append(path + [(nid, e.input_slot)])
                            break
                return
            if nid not in visited and is_invertible(nid):
                for next_id, out_slot, _ in outgoing.get(nid, []):
                    dfs(next_id, path + [(nid, out_slot)], visited | {nid})
        dfs(start_id, [], set())
        return paths

    inv_paths = {n.node_id: find_paths(n.node_id) for n in numeric_nodes}
    inv_paths = {nid: paths for nid, paths in inv_paths.items() if paths}

    if not inv_paths:
        return [graph]

    inversions = (
        [{nid: min(paths, key=len) for nid, paths in inv_paths.items()}] if mode == "shortest"
        else [dict(zip(inv_paths.keys(), combo)) for combo in product(*inv_paths.values())]
    )

    results = []
    for path_combo in inversions:
        builder = GraphBuilder(graph)
        for numeric_id, path in path_combo.items():
            if len(path) <= 1:
                continue
            builder.delete_node(numeric_id)
            prev_id = path[1][0]

            for node_id, slot in path[1:]:
                node = nodes_by_id[node_id]
                if node.node_type == "output":
                    inp_id = builder.add_node("input", extra={"input_from_output": slot, "input_position": len([n for n in builder.nodes.values() if n.node_type == "input"])})
                    builder.add_edge(inp_id, prev_id, output_slot=0, input_slot=0)
                    break

                inv_id = builder.add_node(
                    f"inv_{node.node_type}",
                    extra={},
                    is_inverse_of=InverseSpec(node_id=node_id, output_slot=slot, output_len=len([e for e in graph.edges.values() if e.source_id == node_id]))
                )
                builder.add_edge(inv_id, prev_id, output_slot=0, input_slot=0)
                prev_id = inv_id

        results.append(builder.build())

    return results
