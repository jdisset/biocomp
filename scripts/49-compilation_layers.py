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
            batch_order=batch_order,
        )

    def __repr__(self):
        out = f'{self.network_id}/{self.node_id}-{self.node_type} ({self.batch_order})'
        # \u2193{self.input_from}│'
        return f'{out}'

    def __hash__(self):
        return hash((self.network_id, self.node_id, self.node_type, self.batch_order))


@dataclass
class ComputeLayer:
    # each layer is parrallelized (all nodes are of the same type)
    nodes: list[VirtualNode]

    # function to compute
    f_name: str = None
    f_out_shape: tuple[int] = (1,)

    def output_shape(self):
        return (len(self.nodes), *self.f_out_shape)

    def __repr__(self):
        # just print the list
        return self.nodes.__repr__()

    def __hash__(self):
        return hash(tuple(self.nodes))

    def check(self):
        # first check: every nodes of a lyer has the same type AND there's no duplicate
        assert len(set(n.node_type for n in self.nodes)) == 1
        assert len(set(n for n in self.nodes)) == len(self.nodes)
        # check that they have the same output and input shapes
        assert len(set(n.output_shape for n in self.nodes)) == 1
        assert len(set(len(n.input_from) for n in self.nodes)) == 1


@dataclass
class ComputeStack:
    networks: list[bc.Network]
    layers: list[ComputeLayer]

    layers_start_index: list[int] = None
    output_shape: tuple[int] = None
    node_map: dict[tuple[int, int], tuple[int, int]] = None

    def add_layer(self, layer: ComputeLayer):
        self.layers.append(layer)
        return self

    def build_map(self):
        # build a dict of {(net_id, node_id): (layer_id, node_position)}
        self.node_map = {}
        for l_id, l in enumerate(self.layers):
            for n_id, n in enumerate(l.nodes):
                self.node_map[(n.network_id, n.node_id)] = (l_id, n_id)

        # build info about the output shape
        # the output of a compute stack is a flat array of all the flattened outputs of all the nodes
        # in a layer, outputs are flattened in the same order as the nodes in the layer
        start_id = 0
        self.layers_start_index = []
        for l in self.layers:
            self.layers_start_index.append(start_id)
            start_id += np.prod(l.output_shape())  # flattened output shape of the layer
        self.output_shape = (start_id,)
        self.check()

    def get_output_index(self, network_id, node_id):
        """Return the first index of the output of a node in the flattened output array"""
        assert self.node_map is not None
        l_id, n_id = self.node_map[(network_id, node_id)]

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

    def __hash__(self):
        return hash((tuple(self.networks), tuple(self.layers)))

    def check(self):
        for l in self.layers:
            l.check()
        assert self.layers[0].nodes[0].node_type == 'input'
        for net_id in range(len(self.networks)):
            prev = -1
            for l in self.layers:
                for n in l.nodes:
                    if n.network_id == net_id:
                        assert (
                            n.batch_order >= prev
                        ), f'wrong batch order ({n.batch_order} < {prev} for {n})'
                        prev = n.batch_order


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


def flatten(x):
    if isinstance(x, list):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


def make_all_topo_vnodes(networks: list[bc.Network]):
    return [
        [
            [
                VirtualNode.from_node(net_id, node_id, net.compute_graph, b_id)
                for node_id in node_bunch
            ]
            for b_id, node_bunch in enumerate(topological_order(net.compute_graph))
        ]
        for net_id, net in enumerate(networks)
    ]


def get_current_batches(stack: ComputeStack, type_dict: dict[str, list[VirtualNode]]):
    MAXINT = 2**60
    current_batches = [MAXINT for _ in stack.networks]
    for t, nodes in type_dict.items():
        for n in nodes:
            assert n.batch_order < MAXINT
            current_batches[n.network_id] = min(current_batches[n.network_id], n.batch_order)
    current_batches = [None if b == MAXINT else b for b in current_batches]
    return current_batches


def get_available_next_layer_types(current_batches, type_dict: dict[str, list[VirtualNode]]):
    # a type is available if there are any node with batch_order == current_batches[network_id]
    return [
        t
        for t, nodes in type_dict.items()
        if any(n.batch_order <= current_batches[n.network_id] for n in nodes)
    ]


def make_layer(current_batches, type_dict: dict[str, list[VirtualNode]], t: str):
    layer_nodes = []
    new_type_dict = deepcopy(type_dict)
    new_type_dict[t] = []
    for n in type_dict[t]:
        if n.batch_order <= current_batches[n.network_id]:
            layer_nodes.append(n)
        else:
            new_type_dict[t].append(n)
    return ComputeLayer(layer_nodes), new_type_dict


def make_all_stacks(stack: ComputeStack, type_dict: dict[str, list[VirtualNode]]):
    current_batches = get_current_batches(stack, type_dict)
    if all(b is None for b in current_batches):
        return stack
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
    # we append the stack with the smallest number of layers
    minstack = min(res, key=lambda s: len(s.layers))
    return stack.add_substack(minstack)


def build_stack(networks: list[bc.Network]):
    n_list = flatten(make_all_topo_vnodes(networks))
    type_dict = {}
    for n in n_list:
        type_dict.setdefault(n.node_type, []).append(n)
    with timer('Building smallest compute stack'):
        stack = make_smallest_stack(ComputeStack(networks, []), type_dict)
    stack.check()
    stack.build_map()
    return stack


print('done')
##

models = dman.get_models()
all_networks = [m.network for m in models]

stack = build_stack(all_networks)
pprint(stack.layers)
pprint(f'Reduced {sum(len(l.nodes) for l in stack.layers)} nodes to {len(stack.layers)} layers')

##

layer = stack.layers[2]
input_indices = np.array([stack.get_input_indices(n) for n in layer.nodes])
# shape = (n_nodes, n_inputs, out_dim


input_indices.shape

test_out = np.zeros(stack.output_shape)
test_out.shape

test_out[input_indices].shape


def get_layer_base_function(layer: ComputeLayer):
    pass

##

# let's see if we can gather all inputs for this layer
def make_layer_f(stack, layer_id):
    layer = stack.layers[layer_id]
    base_f = get_layer_base_function(layer)

    # We use a big flattened array for all outputs,
    # either one that gets updated at every layer
    # or a (n_layers, stack_output_shape) array where we store the outputs of each layer

    # a node takes as input:
    # - *input_values (same number for all nodes in the layer)
    # - quantile
    # - rng_key

    # for now, let's use one big array for all outputs that is the concatenation of all outputs
    # and has a shape of (n_total_nodes, max_output_size)
    # I guess we could completely flatten but let's not make things too complicated for now

    n_inputs = len(layer.nodes[0].input_nodes)
    input_start_indices = [
        np.array([stack.get_input_start_indices(n, i) for n in layer.nodes])
        for i in range(n_inputs)
    ]
    all_input_shapes = [
        np.array([stack.get_input_shapes(n, i) for n in layer.nodes]) for i in range(n_inputs)
    ]
    # check that a same input has the same shape accross all nodes of the layer
    assert np.all(np.all(s == s[0]) for s in all_input_shapes)
    input_shapes = [s[0] for s in all_input_shapes]
    input_lengths = [np.prod(s) for s in input_shapes]

    # now we should be able to rebuild the inputs for each node

    @partial(jit, static_argnums=(1,))
    def get_inputs(all_outputs, input_slot=0):
        start_indices = input_start_indices[input_slot]
        length = input_lengths[input_slot]
        shape = input_shapes[input_slot]
        def get_input(n):
            return all_outputs[start_indices[n] : start_indices[n] + length].reshape(shape)
        return vmap(get_input)(np.arange(len(layer.nodes)))


    return get_inputs


f = make_layer_f(stack, 2)


##













































