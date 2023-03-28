### {{{                          --     imports     --
from contextlib import contextmanager
from time import sleep
from collections import defaultdict
import biocomp as bc
from biocomp import datautils as du
from biocomp import nodes as nd
from jax.scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import scriptutils as su
from biocomp import utils as ut
from pathlib import Path
import json5
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import gaussian_kde
import jax
import optax
from jax import jit, vmap, value_and_grad
from jax.tree_util import Partial as partial
from tqdm import tqdm
import biocomp.defaults as bdf
import pandas as pd
from dataclasses import dataclass
from rich import print as pprint
from biocomp import defaults as bdf
from copy import deepcopy


@contextmanager
def timer(name=None):
    from time import perf_counter

    t = perf_counter()
    yield
    if name is not None:
        print(f"\n{name}: {perf_counter() - t:.2f} seconds")
    else:
        print(f"\nElapsed time: {perf_counter() - t:.2f} seconds")


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                      --     loading some xp     --
config = bdf.DEFAULT_CONFIG
lib = su.load_lib()
matrix_xp = su.load_xp('2023-02-16_Matrix', lib, data_path='./data/calibrated_data')
dman = du.DataManager.from_xps([matrix_xp], config, inverse='all')
names = [m.node_namespace for m in dman.get_models()]
key = jax.random.PRNGKey(0)
##────────────────────────────────────────────────────────────────────────────}}}


@dataclass
class VirtualNode:
    network_id: int
    node_id: int
    node_type: str
    input_from: list[tuple[int, int, int]]  # (network_id, node_id, port_id)
    output_shape: tuple[int]
    batch_order: int = 0  # only used for sorting and debugging

    @classmethod
    def from_node(
        cls, network_id: int, node_id: int, compute_df: pd.DataFrame, batch_order: int = 0
    ):
        assert isinstance(network_id, int)
        assert isinstance(node_id, int)
        assert isinstance(compute_df, pd.DataFrame)
        node = compute_df.loc[node_id]
        return cls(
            network_id=network_id,
            node_id=node_id,
            node_type=node['type'],
            input_from=node['input_from'],
            output_shape=len(node['output_to']),
            batch_order=batch_order,
        )

    def __repr__(self):
        out = f'{self.network_id}/{self.node_id}-{self.node_type} ({self.batch_order})'
        # \u2193{self.input_from}│'
        return f'{out}'


@dataclass
class ComputeLayer:
    # each layer is parrallelized (all nodes are of the same type)
    nodes: list[VirtualNode]

    def __repr__(self):
        # just print the list
        return self.nodes.__repr__()


@dataclass
class ComputeStack:
    networks: list[bc.Network]
    layers: list[ComputeLayer]

    def add_layer(self, layer: ComputeLayer):
        self.layers.append(layer)
        return self

    def copy(self):
        # we only deepcopy the layers, not the networks
        return ComputeStack(self.networks, deepcopy(self.layers))

    def add_substack(self, substack):
        assert self.networks == substack.networks
        self.layers.extend(substack.layers)
        return self

    def __repr__(self):
        # layers with line breaks
        return '\n'.join([l.__repr__() for l in self.layers])


models = dman.get_models()
m0, m1 = models[0], models[10]
n1, n2 = m0.network, m1.network

print()
pprint(n1.compute_graph)
pprint(n1.compute_graph.columns)


def topological_order(graph: pd.DataFrame):
    """Returns a list of lists of compute nodes from the network,
    where each node of a sublist can be computed independently of the others,
    but each sublist must be computed in order."""
    visited = set()
    batches = []
    while len(visited) < len(graph):
        independent = [
            i
            for i, row in graph.iterrows()
            if (not row['input_from'] or all([x[0] in visited for x in row['input_from']]))
            and i not in visited
        ]
        if not independent:
            msg = f'No independent node. Remaining:{set(graph.index) - visited}. Visited:{visited}'
            raise ValueError(msg)
        visited.update(independent)
        batches.append(independent)
    return batches


networks = [n1, n2]
topo_vnodes = [
    [
        [VirtualNode.from_node(net_id, node_id, net.compute_graph, b_id) for node_id in node_bunch]
        for b_id, node_bunch in enumerate(topological_order(net.compute_graph))
    ]
    for net_id, net in enumerate(networks)
]

pprint(topo_vnodes)


def flatten(x):
    if isinstance(x, list):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


def get_current_batches(stack: ComputeStack, type_dict: dict[str, list[VirtualNode]]):
    # we find the minimum n.batch_order for each network
    current_batches = [None for _ in stack.networks]
    for t, nodes in type_dict.items():
        for n in nodes:
            if current_batches[n.network_id] is None:
                current_batches[n.network_id] = n.batch_order
            else:
                current_batches[n.network_id] = min(
                    current_batches[n.network_id], n.batch_order
                )
    return current_batches



def get_available_next_layer_types(current_batches, type_dict: dict[str, list[VirtualNode]]):
    # a type is available if there are any node with batch_order == current_batches[network_id]
    return [
        t
        for t, nodes in type_dict.items()
        if any(n.batch_order == current_batches[n.network_id] for n in nodes)
    ]


def make_layer(current_batches, type_dict: dict[str, list[VirtualNode]], t: str):
    node_list = type_dict[t]
    layer_nodes = []
    new_type_dict = deepcopy(type_dict)
    for n in node_list:
        if n.batch_order == current_batches[n.network_id]:
            layer_nodes.append(n)
            new_type_dict[t].remove(n)
    return ComputeLayer(layer_nodes), new_type_dict


best_n_layers = None

def make_all_stacks(stack: ComputeStack, type_dict: dict[str, list[VirtualNode]]):
    global best_n_layers
    current_batches = get_current_batches(stack, type_dict)
    if all(b is None for b in current_batches):
        nlayers = len(stack.layers)
        best_n_layers = nlayers if best_n_layers is None else min(best_n_layers, nlayers)
        return stack
    if best_n_layers is None or len(stack.layers) < best_n_layers:
        next_types = get_available_next_layer_types(current_batches, type_dict)
        res = []
        for t in next_types:
            l, new_type_dict = make_layer(current_batches, type_dict, t)
            s = stack.copy().add_layer(l)
            res.append(make_all_stacks(s, new_type_dict))
        return res


def make_smallest_stack(stack: ComputeStack, type_dict: dict[str, list[VirtualNode]]):
    current_batches = get_current_batches(stack, type_dict)
    if all(b is None for b in current_batches):
        return stack
    next_types = get_available_next_layer_types(current_batches, type_dict)
    res = []
    for t in next_types:
        l, new_type_dict = make_layer(current_batches, type_dict, t)
        substack = ComputeStack(stack.networks, [l])
        res.append(make_smallest_stack(substack, new_type_dict))
    # we want to append the stack with the smallest number of layers
    minstack = min(res, key=lambda s: len(s.layers))
    return stack.add_substack(minstack)




all_networks = [m.network for m in models]
topo_vnodes = [
    [
        [VirtualNode.from_node(net_id, node_id, net.compute_graph, b_id) for node_id in node_bunch]
        for b_id, node_bunch in enumerate(topological_order(net.compute_graph))
    ]
    for net_id, net in enumerate(all_networks)
]
n_list = flatten(topo_vnodes)
type_dict = {}
for n in n_list:
    type_dict.setdefault(n.node_type, []).append(n)


# with timer('computing all stacks'):
    # stacks = flatten(make_all_stacks(ComputeStack(all_networks, []), type_dict))

with timer('computing smallest stacks'):
    stack = make_smallest_stack(ComputeStack(all_networks, []), type_dict)


networks

pprint(len(stack.layers))
nbnodes = sum(len(l.nodes) for l in stack.layers)


