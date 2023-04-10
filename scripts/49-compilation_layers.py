### {{{                          --     imports     --
from contextlib import contextmanager
from time import sleep
from collections import defaultdict
import biocomp
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

import biocomp.new_nodes as bcn

from jax.config import config
config.update("jax_debug_nans", False)
config.update("jax_check_tracer_leaks", False)
# config.update('jax_disable_jit', True)


@contextmanager
def timer(name=None):
    from time import perf_counter

    t = perf_counter()
    yield
    if name is not None:
        print(f"\n{name}: {perf_counter() - t:.2f} seconds")
    else:
        print(f"\nElapsed time: {perf_counter() - t:.2f} seconds")

import jax.profiler
jax.profiler.start_server(9999)

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                      --     loading some xp     --
config = bdf.DEFAULT_CONFIG
lib = su.load_lib()
matrix_xp = su.load_xp('2023-02-16_Matrix', lib, data_path='./data/calibrated_data')
dman = du.DataManager.from_xps([matrix_xp], config, inverse='all')
names = [m.node_namespace for m in dman.get_models()]
key = jax.random.PRNGKey(0)
##────────────────────────────────────────────────────────────────────────────}}}

### {{{                     --     compute functions     --


@dataclass
class VirtualNode:
    network: bc.Network
    network_id: int
    type_signature: str
    compute_node_id: int  # id in the network
    node_id: int = None
    batch_order: int = 0  # only used for sorting and debugging

    @classmethod
    def from_node(
        cls, network_id: int, network: bc.Network, compute_node_id: int, batch_order: int = 0
    ):
        node = network.compute_graph.loc[compute_node_id]
        type = node['type']
        n_inputs = len(node['input_from'])
        n_outputs = len(node['output_to'])
        type_signature = f'{type}_{n_inputs}_{n_outputs}'
        return cls(
            network_id=network_id,
            network=network,
            compute_node_id=compute_node_id,
            type_signature=type_signature,
            batch_order=batch_order,
        )

    def get_compute_node(self):
        return self.network.compute_graph.loc[self.compute_node_id]

    def get_inverse_vnode(self, stack):
        cnode = self.get_compute_node()
        assert cnode.is_inverse_of is not None, 'Node is not an inverse'
        inv = stack.get_vnode_from_net_and_compute_id(self.network_id, cnode.is_inverse_of)
        assert inv is not None, 'Inverse not found'
        return inv

    def __repr__(self):
        out = f'{self.network_id}/{self.compute_node_id}/{self.node_id}-{self.type_signature}'
        return f'{out}'

    def __hash__(self):
        return hash((self.network_id, self.node_id, self.type_signature, self.batch_order))

    def __deepcopy__(self, memo):
        # copy everything except the network (just a reference)
        new_obj = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_obj
        for k, v in self.__dict__.items():
            if k == 'network':
                setattr(new_obj, k, v)
            else:
                setattr(new_obj, k, deepcopy(v, memo))
        return new_obj


@dataclass
class ComputeLayer:
    # each layer is parrallelized (all nodes are of the same type)
    nodes: list[VirtualNode]

    layer_id: int = None

    # information about the function to apply
    f_type: str = None
    f_out_shapes: list[tuple[int]] = None
    f_input_shapes: list[tuple[int]] = None

    f_prepare = None
    f_apply = None

    is_built = False

    def setup(self, config: bcn.ConfigManager, stack):

        self.check()

        first_node = self.nodes[0].get_compute_node()
        self.f_type = first_node.type


        if self.f_type == 'input':
            self.f_out_shapes = [(1,)]
            self.f_input_shapes = [(1,)]
            self.is_built = True
            return

        # get the shapes of the inputs. We'll collect all the inputs for each node
        # to make sure they are all the same
        node_inputs = []  # list of list of (net_id, compute_node_id, slot_id)
        for n in self.nodes:
            ninp = n.get_compute_node().input_from
            node_inputs.append([(n.network_id, *i) for i in ninp])

        # get the shapes of the inputs
        all_input_shapes = []  # list of list of shapes
        for n_inp in node_inputs:
            input_shapes = []
            for input_net_id, input_compute_node_id, input_slot_id in n_inp:
                input_layer_id, _ = stack.node_map[(input_net_id, input_compute_node_id)]
                assert input_layer_id < self.layer_id, 'Input node is in a later layer'
                assert stack.layers[input_layer_id].is_built, 'Input layer is not built'
                input_layer_output_shapes = stack.layers[input_layer_id].f_out_shapes
                assert input_slot_id < len(
                    input_layer_output_shapes
                ), f'Input slot {input_slot_id} is out of range'
                input_shapes.append(input_layer_output_shapes[input_slot_id])
            all_input_shapes.append(tuple(input_shapes))
        # they should all be the same
        assert len(set(all_input_shapes)) == 1
        self.f_input_shapes = all_input_shapes[0]

        n_outputs = len(first_node.output_to)

        impl = config.get(self.f_type)(
            input_shapes=self.f_input_shapes, n_outputs=n_outputs, stack=stack
        )
        self.f_prepare, self.f_apply, self.f_out_shapes = impl
        self.is_built = True

    def flattened_output_shape(self):
        return int(len(self.nodes) * np.sum([np.prod(s) for s in self.f_out_shapes]))

    def __repr__(self):
        return self.nodes.__repr__()

    def __hash__(self):
        return hash(tuple(self.nodes))

    def check(self):
        assert len(set(n.type_signature for n in self.nodes)) == 1


@dataclass
class ComputeStack:
    networks: list[bc.Network]
    layers: list[ComputeLayer]

    layers_start_index: list[int] = None
    output_shape: tuple[int] = None
    node_map: dict[tuple[int, int], tuple[int, int]] = None

    total_nb_of_outputs: int = None
    total_nb_of_inputs: int = None
    max_nb_of_outputs_per_network: int = None

    shared_store: dict = None  # shared store for all the nodes.
    # can be used to store things like the name of the parts for some quantized parameters
    # as they can't be stored in params (no strings allowed)

    init = None
    all_built = False
    number_of_nodes = 0

    def add_layer(self, layer: ComputeLayer):
        self.layers.append(layer)
        return self

    def build_map(self):
        allbuilt = True

        self.total_nb_of_outputs = 0
        self.total_nb_of_inputs = 0
        self.max_nb_of_outputs_per_network = 0
        for n in self.networks:
            nbout = int(n.get_nb_outputs())
            self.total_nb_of_inputs += n.get_nb_inputs()
            self.total_nb_of_outputs += nbout
            self.max_nb_of_outputs_per_network = max(self.max_nb_of_outputs_per_network, nbout)

        # build a dict of {(net_id, node_id): (layer_id, node_position)}
        node_id = 0
        self.node_map = {}
        for l_id, l in enumerate(self.layers):
            l.layer_id = l_id
            if l.is_built:
                for n_id, n in enumerate(l.nodes):
                    self.node_map[(n.network_id, n.compute_node_id)] = (l_id, n_id)
                    n.node_id = node_id
                    node_id += 1
            else:
                allbuilt = False
                break

        self.number_of_nodes = node_id
        # build info about the output shape
        # the output of a compute stack is a flat array of all the flattened outputs of all the nodes
        # in a layer, outputs are flattened in the same order as the nodes in the layer
        if allbuilt:
            start_id = 0
            self.layers_start_index = []
            for l in self.layers:
                self.layers_start_index.append(start_id)
                start_id += int(l.flattened_output_shape())
            self.check()
            self.output_shape = (int(start_id),)

        self.all_built = allbuilt

    def get_network_global_output_id(self, network_id: int, output_id: int):
        """Considering every network's outputs ordered by network id,
        returns the global id of the given output for the given network.
        Useful to convert quantile ids from a network to the global quantile ids"""
        assert network_id < len(self.networks)
        return sum(n.get_nb_outputs() for n in self.networks[:network_id]) + output_id

    def get_vnode_from_net_and_compute_id(
        self, network_id: int, compute_node_id: int
    ) -> VirtualNode:
        layer_id, node_loc = self.node_map[(network_id, compute_node_id)]
        return self.layers[layer_id].nodes[node_loc]

    def get_network_output_indices(self, network_id: int):
        """Returns the start index and shape of the output of the given network"""
        output_node = self.networks[
            network_id
        ].get_output_compute_node()  # a row from the compute df
        node_id = output_node.name
        layer_id, node_loc = self.node_map[(network_id, node_id)]
        out_shape = self.layers[layer_id].f_out_shapes
        start_index = self.layers_start_index[layer_id] + node_loc * np.sum(
            [np.prod(s) for s in out_shape]
        )
        return int(start_index), out_shape

    def get_node_input_start_index(self, node: VirtualNode, input_slot: int) -> int:
        """Returns the start index of the input #input_slot for the given node
        in the flattened output array"""
        assert self.node_map is not None, 'call build_map() first'

        input_compute_node_id, input_compute_node_outslot = node.get_compute_node().input_from[
            input_slot
        ]

        input_layer_id, input_node_layer_loc = self.node_map[
            (node.network_id, input_compute_node_id)
        ]
        this_node_layer_id, _ = self.node_map[(node.network_id, node.compute_node_id)]

        assert input_layer_id < this_node_layer_id, 'input layer must be before this layer'

        this_layer = self.layers[this_node_layer_id]
        input_layer = self.layers[input_layer_id]

        this_input_shapes = this_layer.f_input_shapes
        input_layer_start = self.layers_start_index[input_layer_id]

        assert this_input_shapes[input_slot] == input_layer.f_out_shapes[input_compute_node_outslot]

        flat_out_size = int(np.sum([np.prod(s) for s in input_layer.f_out_shapes]))
        node_start = input_layer_start + input_node_layer_loc * flat_out_size

        flat_output_shape_till_input = np.sum(
            [np.prod(s) for s in input_layer.f_out_shapes[:input_compute_node_outslot]]
        )
        outslot_start = node_start + flat_output_shape_till_input

        return int(outslot_start)

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
            for n in l.nodes:
                assert id(n.network) == id(self.networks[n.network_id])
        assert self.layers[0].nodes[0].get_compute_node().type == 'input'
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
            [VirtualNode.from_node(net_id, net, node_id, b_id) for node_id in node_batch]
            for b_id, node_batch in enumerate(topological_order(net.compute_graph))
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
        type_dict.setdefault(n.type_signature, []).append(n)
    with timer('Building smallest compute stack'):
        stack = make_smallest_stack(ComputeStack(networks, []), type_dict)
    return stack


def make_layer_input_getters(stack, layer_id):

    layer = stack.layers[layer_id]
    assert layer.is_built, 'Layer not built'
    input_shapes = layer.f_input_shapes  # list of tuples, one for each input
    input_lengths = [int(np.prod(s)) for s in input_shapes]  # flattened length of each input

    # input layer is a special case as it doesn't take values from the stack output
    if layer.f_type == 'input':
        assert len(input_shapes) == 1  # each node has one "input"
        assert layer_id == 0  # input layer is the first layer
        assert input_shapes == layer.f_out_shapes
        # input indices of the input layer are just the indices of the nodes
        input_start_indices = np.array([[n.node_id * input_lengths[0] for n in layer.nodes]])
    else:
        input_start_indices = np.array(
            [
                [stack.get_node_input_start_index(n, i) for n in layer.nodes]
                for i in range(len(input_shapes))
            ]
        )

    # We use a big flattened array for all outputs,
    # now we should be able to build a function that returns all inputs for a given layer
    # from the big stack array

    def generate_get_inputs(input_slot):
        starts = input_start_indices[input_slot]
        shape = input_shapes[input_slot]
        length = input_lengths[input_slot]
        indices = np.array([np.arange(st, st + length) for st in starts])

        # We can either dynamically slice the big output array at start:length
        # or directly index it at indices... I'm not sure which is faster
        # but I guess it's better to use dynamic slicing for large inputs?
        DYN_SLICE = False
        DYN_SLICE_THRESHOLD = 20
        if length > DYN_SLICE_THRESHOLD:
            DYN_SLICE = True

        def get_inputs_dyn(all_outputs):
            def dyn_slice(start):
                return jax.lax.dynamic_slice(all_outputs, (start,), (length,)).reshape(shape)

            return vmap(dyn_slice)(starts)

        def get_inputs_idx(all_outputs):
            def slice(idx):
                return all_outputs[idx].reshape(shape)

            return vmap(slice)(indices)

        return get_inputs_dyn if DYN_SLICE else get_inputs_idx

    get_inputs = [generate_get_inputs(i) for i in range(len(input_shapes))]

    return get_inputs


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                 --     tests for param functions     --

from biocomp.new_nodes import set_param, get_param, init_param_if_needed


def test_set_and_get_param():
    params = {}

    # Test setting and getting a single parameter at the root level
    set_param(params, 'a', np.array([1, 2, 3]), node_id=1)
    assert np.array_equal(get_param(params, 'a', node_id=1), np.array([1, 2, 3]))

    # Test setting and getting a single parameter with a custom node_id
    set_param(params, 'a', np.array([4, 5, 6]), node_id=5)
    assert np.array_equal(get_param(params, 'a', node_id=5), np.array([4, 5, 6]))

    # Test setting and getting a parameter with a 3-level deep base_path
    base_path = ['level1', 'level2', 'level3']
    set_param(params, 'b', np.array([1, 1, 1]), base_path=base_path)
    assert np.array_equal(get_param(params, 'b', base_path=base_path), np.array([1, 1, 1]))


def test_init_param_if_needed():
    params = {}

    # Test initializing a parameter with a default value
    init_param_if_needed(params, 'c', lambda: np.array([0, 0, 0]))
    assert np.array_equal(get_param(params, 'c'), np.array([0, 0, 0]))

    # Test initializing a parameter with a custom node_id
    init_param_if_needed(params, 'c', lambda: np.array([1, 1, 1]), node_id=10)
    assert np.array_equal(get_param(params, 'c', node_id=10), np.array([1, 1, 1]))

    # Test initializing a parameter with a 3-level deep base_path
    base_path = ['init', 'level2', 'level3']
    init_param_if_needed(params, 'd', lambda: np.array([-1, -1, -1]), base_path=base_path)
    assert np.array_equal(get_param(params, 'd', base_path=base_path), np.array([-1, -1, -1]))

    # Test not initializing a parameter if it's already set
    set_param(params, 'e', np.array([3, 3, 3]))
    init_param_if_needed(params, 'e', lambda: np.array([4, 4, 4]))
    assert np.array_equal(get_param(params, 'e'), np.array([3, 3, 3]))



##────────────────────────────────────────────────────────────────────────────}}}

### {{{                        --     make stack     --

models = dman.get_models()
m0 = models[0]
m0.network.compute_graph
all_networks = [m.network for m in models][:]


def make_stack(all_networks):

    stack = build_stack(all_networks)
    pprint(f'Reduced {sum(len(l.nodes) for l in stack.layers)} nodes to {len(stack.layers)} layers')

    # build a compute stack

    config = bcn.DEFAULT_NODE_CONFIG
    # let's make shared_config a property of the stack

    # first we need to store all quantizable parameter names
    stack.shared_store = {}
    store = stack.shared_store
    quantization_params = {}
    for n_id, n in enumerate(all_networks):
        for qn, qv in bcn.get_all_possible_quantization_params(n).items():
            quantization_params[qn] = set(qv) | quantization_params.get(qn, set())
    quantization_params = {k: sorted(v) for k, v in quantization_params.items()}
    for pname, pv in quantization_params.items():
        ut.at_path(store, ut.QNAME_PATH + f"/{pname}", pv)


    # setup each layer
    for layer in stack.layers:
        stack.build_map()
        layer.setup(config, stack=stack)
    stack.build_map()

    # now what would be the init function:
    def init(rng_key):
        params = {}
        for layer in tqdm(stack.layers):
            assert layer.is_built, 'Layer not built'
            if layer.f_prepare is not None:
                layer.f_prepare(params, vnodelist=layer.nodes, key=rng_key)
        return params

    stack.init = init

    input_getters = [make_layer_input_getters(stack, l_id) for l_id in range(len(stack.layers))]

    output_shape = stack.output_shape  # we use one big flat output array
    assert len(output_shape) == 1, 'Only flat output arrays are supported for now'

    node_ids = [jnp.array([n.node_id for n in l.nodes]) for l in stack.layers]

    def compute(params, inputs, quantiles, key):

        out = jnp.full(output_shape, np.nan)
        input_indices = jnp.arange(len(inputs))
        out = out.at[input_indices].set(inputs)

        for lid in range(1, len(stack.layers)):
            layer_start_index = stack.layers_start_index[lid]
            input_shapes = stack.layers[lid].f_input_shapes
            n_inputs = len(input_shapes)
            assert n_inputs == len(input_getters[lid]), 'Mismatch in number of inputs'
            layer_inputs = [input_getters[lid][i](out) for i in range(n_inputs)]
            assert all(
                len(li) == len(layer_inputs[0]) for li in layer_inputs
            ), 'Mismatch in input lengths'


            def f(node_id, key, *inputs):
                return stack.layers[lid].f_apply(
                    *inputs, params=params, quantiles=quantiles, node_id=node_id, key=key
                )

            keys = jax.random.split(key, len(layer_inputs[0]))

            out_layer = vmap(f)(node_ids[lid], keys, *layer_inputs)
            out_flat = out_layer.reshape(-1)

            out = out.at[layer_start_index : layer_start_index + out_flat.shape[0]].set(out_flat)

        return out

    stack.compute = compute

    return stack

s = make_stack(all_networks)
with timer('init'):
    params = s.init(jax.random.PRNGKey(0))




##────────────────────────────────────────────────────────────────────────────}}}##


def is_equal(a1, a2):
    if isinstance(a1, jnp.ndarray):
        return jnp.all(a1 == a2)
    else:
        return a1 == a2

def find_keys_diff(d1, d2):
    # prints the keys in d1 and d2 that have different values
    # d1 and d2 are nested
    changed = []
    for k, v in d1.items():
        assert k in d2, f'{k} not in d2'
        if isinstance(v, dict):
            c = find_keys_diff(v, d2[k])
            changed.extend(c)
        else:
            if not is_equal(v, d2[k]):
                name = str(k)
                changed.append(name)
    return changed

def contains_nan(d):
    if isinstance(d, jnp.ndarray):
        return jnp.any(jnp.isnan(d))
    for k, v in d.items():
        if isinstance(v, dict):
            if contains_nan(v):
                return True
        else:
            if jnp.any(jnp.isnan(v)):
                return True
    return False



# from jax.config import config
# config.update("jax_debug_nans", True)

quantiles = jnp.arange(s.total_nb_of_outputs).astype(jnp.float32) / s.total_nb_of_outputs
inputs = jnp.arange(s.total_nb_of_inputs).astype(jnp.float32)
k = jax.random.PRNGKey(0)

batches = jax.random.uniform(k, (64,128, s.total_nb_of_inputs)).astype(jnp.float32)
vmapped_compute = jax.vmap(s.compute, in_axes=(None, 0, None, None))

def simple_loss(dynparams, staticparams, batch, quantiles, key):
    allparams = ut.assemble_params(dynparams, staticparams)
    out = vmapped_compute(allparams, batch, quantiles, key)
    return jnp.mean(out)

lr = 0.01


def scannable_loss(carry, batch):
    allparams, key = carry
    dynamic, static = ut.split_params(allparams, ['/__static__','/node'])
    loss, grads = jax.value_and_grad(simple_loss)(dynamic, static, batch, quantiles, key)
    key, subkey = jax.random.split(key)
    dynamic = jax.tree_map(lambda p, g: p - lr * g, dynamic, grads)
    allparams = ut.assemble_params(dynamic, static)
    return (allparams, key), loss

params.keys()

# pprint(params['shared'])

@jit
def train(params, batches, key):
    key, subkey = jax.random.split(key)
    (params, key), loss = jax.lax.scan(scannable_loss, (params, key), batches)
    return params, key, loss


with timer('Compiling'):
    train(params, batches, k)


for i in range(2):
    with timer(f'Epoch {i}'):
        params, k, losses = train(params, batches, k)

# benchmarking:
# 10 first networks, 128x128: 
# -----------------------------------
# float indexing + all grad -> 3.6s per epoch, first epoch 19.2s
# float indexing + only shared grad -> 3.8s per epoch (???)
# integer indexing + only shared grad -> 3.4s per epoch
# direct indexing + only shared grad -> 3.3s per epoch
# TO TRY:
# flat param dictionnary: just use '/' as separator
# 10 nets, 64x128 with new default nn:
# direct indexing + split grad + nested at_path -> 3s per epoch
# direct indexing + split grad + flat at_path -> 2.9s per epoch




# TODO:
# - split static and dynamic params, use integer indexing as much as possible but in a
#   parameter path that's not differentiated against! that should accelerate things
# - check correctness vs previous model
# - benchmark in simple forward pass
# if it's less than 5x slower, we're good, 
# - let's migrate everything!
# - write a new training loop
# if not:
# - profile with jax to see what's slow (probably the indexing?)
# - benchmark vs the switch version of previous model?
# runtime speed improvement ideas: 
# - add option to write param directly (without key)
##
ploss = partial(simple_loss, params, quantiles=quantiles, key=k)
vploss = jit(vmap(ploss))


# with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
  # # Run the operations to be profiled
  # y = vploss(N_random_inputs)
  # y.block_until_ready()
