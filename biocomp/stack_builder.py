"""Build-time functions for ComputeStack assembly.

This module contains all the complex graph traversal and layer building algorithms
that are used only during the one-time build process of a ComputeStack.
"""

from __future__ import annotations
from copy import deepcopy
from collections import deque
from queue import PriorityQueue
from typing import TYPE_CHECKING, Optional

from biocomp.graphengine import GraphState
from biocomp import utils as ut

if TYPE_CHECKING:
    from .compute import ComputeStack, StackLayer
    from .network import Network


def topological_order(graph: GraphState) -> list[list[int]]:
    """Returns a list of lists of node IDs from the graph,
    where each node of a sublist can be computed independently of the others,
    but each sublist must be computed in order."""
    visited = set()
    batches = []
    while len(visited) < len(graph.nodes):
        independent = [
            node_id
            for node_id, node in graph.nodes.items()
            if node_id not in visited
            and all(edge.source_id in visited for edge in graph.get_incoming_edges(node_id))
        ]
        if not independent:
            remaining = set(graph.nodes.keys()) - visited
            msg = f"No independent node. Remaining:{remaining}. Visited:{visited}"
            raise ValueError(msg)
        visited.update(independent)
        # sort for deterministic ordering when nodes are topologically equivalent
        batches.append(sorted(independent))
    return batches


def make_all_topo_nodes(networks: list[Network]) -> list[list[list[tuple]]]:
    """Returns topological ordering for all networks as (NodeKey, batch_order, type_signature) tuples"""
    from .compute import StackNode

    return [
        [
            [
                (
                    StackNode(net_id, node_id),
                    b_id,
                    StackNode.generate_type_signature(net.compute_graph, node_id),
                )
                for node_id in node_batch
            ]
            for b_id, node_batch in enumerate(topological_order(net.compute_graph))
        ]
        for net_id, net in enumerate(networks)
    ]


def get_networks_current_batch_number(
    networks: list[Network], type_dict: dict[str, list[tuple]]
) -> list[Optional[int]]:
    """Determines the current (minimum) batch number for each network.
    The batch number is the order in which a node should be computed (topological order of a network).
    type_dict maps node types to the list of all (NodeKey, batch_order) tuples for nodes not yet in the stack.

    Returns a list of the current batch number for each network. If a network has no current batch,
    it means it has no nodes left to be computed and its batch number value will be None.
    """
    MAXINT = 2**60
    current_batches = [MAXINT for _ in networks]
    for nodes_with_batch in type_dict.values():
        for key, batch_order in nodes_with_batch:
            assert batch_order < MAXINT, "Node has no batch order"
            current_batches[key.network_id] = min(current_batches[key.network_id], batch_order)
    current_batches = [None if b == MAXINT else b for b in current_batches]
    return current_batches


def make_layer_from_current_batches(
    current_batches: list[Optional[int]], type_dict: dict[str, list[tuple]], t: str
) -> tuple[StackLayer, dict[str, list[tuple]]]:
    """
    Creates a ComputeLayer from the nodes of type t that have a batch_order <= current_batches[network_id]
    Returns a ComputeLayer and a new type_dict without the nodes that were used in the layer
    """
    from .compute import StackLayer

    layer_keys = []
    new_type_dict = deepcopy(type_dict)
    new_type_dict[t] = []
    used = 0
    for key, batch_order in type_dict[t]:
        current_batch = current_batches[key.network_id]
        if current_batch is not None and batch_order <= current_batch:
            layer_keys.append(key)
            used += 1
        else:
            new_type_dict[t].append((key, batch_order))

    assert used > 0, f"used {used} nodes of type {t} to make layer"
    return StackLayer(nodes=layer_keys), new_type_dict


def make_smallest_stack_bfs(
    networks: list[Network],
    initial_layers: list[StackLayer],
    type_dict: dict[str, list[tuple]],
    max_t: int = 1,
) -> list[StackLayer]:
    """Build stack using breadth-first search"""
    from .compute import ComputeStack

    stack = ComputeStack(networks, initial_layers)
    bfs_queue = deque([(stack, type_dict, [], 0)])
    iteration = 0

    while bfs_queue:
        iteration += 1

        current_stack, current_type_dict, path, depth = bfs_queue.popleft()

        current_batches = get_networks_current_batch_number(networks, current_type_dict)

        if all(b is None for b in current_batches):  # no nodes left to compute
            return current_stack.layers

        # possible_next_types is a list of types that are candidates for the next layer,
        # i.e they contain nodes that that have a batch_order == current_batches[network_id]
        # for at least one network
        possible_next_types = []
        for t, nodes_with_batch in current_type_dict.items():
            can_be_computed = [
                (key, b) for key, b in nodes_with_batch if b == current_batches[key.network_id]
            ]
            if can_be_computed:
                possible_next_types.append((t, len(can_be_computed)))

        if max_t is not None:
            # sort by decreasing number of nodes
            possible_next_types = sorted(possible_next_types, key=lambda x: x[1], reverse=True)
            possible_next_types = possible_next_types[:max_t]

        assert possible_next_types, "No possible next type"

        # we try every possible type for the next layer
        for t, n in possible_next_types:
            l, new_type_dict = make_layer_from_current_batches(
                current_batches, current_type_dict, t
            )
            # l is a ComputeLayer, new_type_dict is a dict[str, list[(NodeKey, int)]]
            # without the nodes that were used in the layer
            node_diff = len(current_type_dict[t]) - len(new_type_dict[t])
            substack = ComputeStack(current_stack.networks, [l])
            path_entry = f"{t}: picked {node_diff} nodes, {len(possible_next_types)} possible types"

            assert n == node_diff, f"{n} != {node_diff}"

            bfs_queue.append((substack, new_type_dict, path + [path_entry], depth + 1))

    # If we reach here, we didn't find a solution
    raise RuntimeError("No solution found")


def make_smallest_stack_dfs(
    networks: list[Network],
    initial_layers: list[StackLayer],
    type_dict: dict[str, list[tuple]],
    path=None,
    depth=0,
    max_depth=70,
    max_t=1,
) -> list[StackLayer]:
    """Build stack using depth-first search"""
    from .compute import ComputeStack

    if path is None:
        path = []

    stack = ComputeStack(networks, initial_layers)

    # current_batches is a list of the current batch number for each network in the stack.
    current_batches = get_networks_current_batch_number(networks, type_dict)

    if all(b is None for b in current_batches):  # no nodes left to compute
        return stack.layers

    # possible_next_types is a list of types that are candidates for the next layer, i.e they contain
    # nodes that that have a batch_order == current_batches[network_id] for at least one network
    possible_next_types = []
    for t, nodes_with_batch in type_dict.items():
        can_be_computed = [
            (key, b) for key, b in nodes_with_batch if b == current_batches[key.network_id]
        ]
        if can_be_computed:
            possible_next_types.append((t, len(can_be_computed)))

    if max_t is not None:
        # we're basically doing beam search here, by only keeping the max_t types with the most nodes
        possible_next_types = sorted(possible_next_types, key=lambda x: x[1], reverse=True)
        possible_next_types = possible_next_types[:max_t]

    assert possible_next_types, "No possible next type"
    candidate_stacks = []
    # we try every possible type for the next layer
    for t, _ in possible_next_types:
        l, new_type_dict = make_layer_from_current_batches(current_batches, type_dict, t)
        # l is a ComputeLayer, new_type_dict is a dict[str, list[(NodeKey, int)]]
        # without the nodes that were used in the layer
        node_diff = len(type_dict[t]) - len(new_type_dict[t])
        substack_layers = [l]
        path_entry = f"{t}: picked {node_diff} nodes, {len(possible_next_types)} possible types"
        candidate_stacks.append(
            make_smallest_stack_dfs(
                networks,
                substack_layers,
                new_type_dict,
                path + [path_entry],
                depth + 1,
                max_depth,
                max_t,
            )
        )

    assert candidate_stacks, "No candidate stack"

    if depth >= max_depth:
        # raise a detailed error
        msg = f"Max depth reached: {max_depth}\n"
        msg += f"Current stack:\n{stack}\n"
        msg += f"Current type_dict:\n{type_dict}\n"
        msg += f"Current batches:\n{current_batches}\n"
        msg += f"Possible next types:\n{possible_next_types}\n"
        msg += f"Candidate stacks:\n{candidate_stacks}\n"
        raise RuntimeError(msg)

    # and we only keep the smallest stack
    minstack_layers = min(candidate_stacks, key=lambda s: len(s))

    # extend current stack with minstack
    return stack.extend(ComputeStack(networks, minstack_layers)).layers


def heuristic(type_dict: dict[str, list[tuple]]) -> float:
    """Heuristic function for A* search"""
    total_nodes_left = sum(len(nodes_with_batch) for nodes_with_batch in type_dict.values())
    if not total_nodes_left:
        return 0

    min_nodes_left = float("inf")
    for t, nodes_with_batch in type_dict.items():
        nodes_left = len(nodes_with_batch)
        min_nodes_left = min(min_nodes_left, nodes_left)

    return min_nodes_left


def make_smallest_stack_astar(
    networks: list[Network],
    initial_layers: list[StackLayer],
    type_dict: dict[str, list[tuple]],
    path=None,
    max_t=2,
) -> list[StackLayer]:
    """Build stack using A* search"""
    from .compute import ComputeStack

    if path is None:
        path = []

    # Initial state
    stack = ComputeStack(networks, initial_layers)
    start_node = (0, (stack, type_dict, path))

    # Priority queue for A* search
    queue = PriorityQueue()
    queue.put(start_node)

    while not queue.empty():
        _, (current_stack, current_type_dict, current_path) = queue.get()

        current_batches = get_networks_current_batch_number(networks, current_type_dict)

        if all(b is None for b in current_batches):  # no nodes left to compute
            return current_stack.layers

        possible_next_types = []
        for t, nodes_with_batch in current_type_dict.items():
            can_be_computed = [
                (key, b) for key, b in nodes_with_batch if b == current_batches[key.network_id]
            ]
            if can_be_computed:
                possible_next_types.append((t, len(can_be_computed)))
                break

        if max_t is not None:
            possible_next_types = sorted(possible_next_types, key=lambda x: x[1], reverse=True)
            possible_next_types = possible_next_types[:max_t]

        assert possible_next_types, "No possible next type"

        # we try every possible type for the next layer
        for t, _ in possible_next_types:
            l, new_type_dict = make_layer_from_current_batches(
                current_batches, current_type_dict, t
            )
            node_diff = len(current_type_dict[t]) - len(new_type_dict[t])
            substack = ComputeStack(current_stack.networks, [l])
            new_path = current_path + [
                f"{t}: picked {node_diff} nodes, {len(possible_next_types)} possible types"
            ]

            # Calculate the priority for A* search
            cost_so_far = len(substack.layers) if substack.layers else 0
            estimated_cost = heuristic(new_type_dict)
            priority = cost_so_far + estimated_cost

            queue.put((priority, (substack, new_type_dict, new_path)))

    raise ValueError("No solution found")


def build_layers(networks: list[Network], stack: ComputeStack, **kwargs) -> list[StackLayer]:
    """Main entry point for building layers from networks.

    Args:
        networks: List of Network objects to build stack from
        stack: ComputeStack instance to set as reference in layers
        **kwargs: Additional arguments passed to the search algorithm

    Returns:
        List of ComputeLayer objects with stack references set
    """
    # flatten all (NodeKey, batch_order, type_signature) tuples from all networks
    # make_all_topo_nodes returns list[list[list[tuple]]] - we need to flatten twice
    n_list = ut.flatten_single(ut.flatten_single(make_all_topo_nodes(networks)))
    type_dict = {}
    for key, batch_order, type_signature in n_list:
        type_dict.setdefault(type_signature, []).append((key, batch_order))

    # build layers using DFS
    layers = make_smallest_stack_dfs(networks, [], type_dict, **kwargs)

    # set stack reference in all layers
    if layers:
        for layer in layers:
            layer.stack = stack

    return layers
