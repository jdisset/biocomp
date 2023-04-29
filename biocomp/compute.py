### {{{                          --     imports     --
from copy import deepcopy
from dataclasses import dataclass
from queue import PriorityQueue

from collections import deque
from typing import Tuple
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import vmap
from jax.tree_util import Partial as partial
import json
import inspect

from rich import print as pprint

from . import nodes as nd
from .network import Network
from . import utils as ut

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                      --     Config manager     --


def unwrap_partial_function(implementation):
    if hasattr(implementation, 'func') and hasattr(implementation, 'keywords'):
        partial_args = implementation.keywords
        implementation = implementation.func
    else:
        partial_args = {}
    return implementation, partial_args


class ComputeConfigManager:
    def __init__(self):
        self.config = {'functions': {}}

    def set(self, key, implementation, **kwargs):
        implementation, partial_args = unwrap_partial_function(implementation)
        kwargs.update(partial_args)

        signature = inspect.signature(implementation)
        parameters = {}
        for name, param in signature.parameters.items():
            if name in kwargs:
                parameters[name] = kwargs[name]
            elif param.default != inspect.Parameter.empty:
                parameters[name] = param.default

        sc = parameters.pop('stack', None)
        assert sc is None, 'stack is a reserved parameter name'

        self.config['functions'][key] = {
            'implementation': implementation.__name__,
            'parameters': parameters,
        }

    def get(self, key):
        assert key in self.config['functions'], f'No function named {key}'
        func_data = self.config['functions'][key]
        # take from node module
        node_m = nd
        implementation = getattr(node_m, func_data['implementation'])
        params = func_data['parameters']
        return partial(implementation, **params)

    def export(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.config, f)

    def get_config(self):
        return self.config

    def load_file(self, filename):
        with open(filename, 'r') as f:
            self.config = json.load(f)

    # access functions through [] operator:
    def __getitem__(self, key):
        return self.get(key)

    @classmethod
    def from_file(cls, filename):
        ccm = cls()
        ccm.load_file(filename)
        return ccm

    def load(self, config):
        self.config = config


from . import nodes

DEFAULT_COMPUTE_CONFIG = ComputeConfigManager()
DEFAULT_COMPUTE_CONFIG.set('transcription', nodes.transcription)
DEFAULT_COMPUTE_CONFIG.set('translation', nodes.translation)
DEFAULT_COMPUTE_CONFIG.set('inv_transcription', nodes.inv_transcription)
DEFAULT_COMPUTE_CONFIG.set('inv_translation', nodes.inv_translation)
DEFAULT_COMPUTE_CONFIG.set('sequestron_ERN', nodes.ERN5p)
DEFAULT_COMPUTE_CONFIG.set('sequestron_ERN3p', nodes.ERN3p)
DEFAULT_COMPUTE_CONFIG.set('source', nodes.source)
DEFAULT_COMPUTE_CONFIG.set('inv_source', nodes.inv_source)
DEFAULT_COMPUTE_CONFIG.set('numeric', nodes.numeric)
DEFAULT_COMPUTE_CONFIG.set('inv_numeric', nodes.inv_numeric)
DEFAULT_COMPUTE_CONFIG.set('aggregation', nodes.aggregation)
DEFAULT_COMPUTE_CONFIG.set('inv_aggregation', nodes.inv_aggregation)
DEFAULT_COMPUTE_CONFIG.set('output', nodes.grouped_output)
DEFAULT_COMPUTE_CONFIG.set('deadend', nodes.single_passthrough)

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                       --     Virtual Node     --


@dataclass
class VirtualNode:
    """A virtual node does match with a compute node in the network, but it is
    used to represent a node in the stack, with a unique id, and a type signature
    that depends on the topology of the network. It also has a batch_order, which
    is used to sort the nodes in the stack, so that the nodes that need to be computed
    first are at the top of the stack."""

    network: Network = None
    network_id: int = None
    type_signature: str = None  # type of node, and number of inputs and outputs
    compute_node_id: int = None  # id of the compute node in the network
    node_id: int = None  # unique id for the node in the stack
    batch_order: int = 0  # only used for sorting and debugging

    @staticmethod
    def generate_type_signature(network, compute_node_id):
        node = network.compute_graph.loc[compute_node_id]
        type = node['type']
        n_inputs = len(node['input_from'])
        n_outputs = len(node['output_to'])
        return f'{type}_{n_inputs}_{n_outputs}'

    @classmethod
    def from_node(
        cls, network_id: int, network: Network, compute_node_id: int, batch_order: int = 0
    ):
        type_signature = VirtualNode.generate_type_signature(network, compute_node_id)
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
        out = f'{self.network_id}/{self.compute_node_id}/{self.node_id if self.node_id is not None else self.batch_order}-{self.type_signature}'
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


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                       --     Compute Layer     --
@dataclass
class ComputeLayer:
    nodes: list[VirtualNode]
    layer_id: int = None

    # information about the function to apply
    f_type: str = None
    f_out_shapes: list[tuple[int]] = None
    f_input_shapes: list[tuple[int]] = None

    f_prepare = None
    f_apply = None

    is_built = False

    def setup(self, config: ComputeConfigManager, stack):

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
        first_node = self.nodes[0].get_compute_node()
        ftype = first_node.type
        return f'Layer {self.layer_id} ({ftype}) with {len(self.nodes)} nodes'

    def __hash__(self):
        return hash(tuple(self.nodes))

    def check(self):
        assert len(set(n.type_signature for n in self.nodes)) == 1


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                       --     Compute Stack     --


@dataclass
class ComputeStack:
    networks: list[Network]
    layers: list[ComputeLayer] = None

    layers_start_index: list[int] = None
    output_shape: tuple[int] = None
    node_map: dict[tuple[int, int], tuple[int, int]] = None

    total_nb_of_outputs: int = None
    total_nb_of_inputs: int = None
    max_nb_of_outputs_per_network: int = None

    shared_store: dict = None  # shared store for all the nodes.
    # can be used to store things like the name of the parts for some quantized parameters
    # as they can't be stored in params (no strings allowed)

    is_assembled = False
    is_built = False
    number_of_nodes = 0

    init = None
    apply = None

    ### {{{                     --     public interface     --
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

    def get_network_global_output_id(self, network_id: int, output_id: int = 0):
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

    def init(self, rng_key):
        assert self.is_built, 'Stack not built'
        params = {}
        for l_id, layer in enumerate(self.layers):
            assert layer.is_built, 'Layer not built'
            assert l_id == layer.layer_id, 'Layer id mismatch'
            rng_key, _ = jax.random.split(rng_key)
            ut.logger.info(
                f'Initializing {len(layer.nodes)} nodes in layer {l_id}/{len(self.layers)}'
            )
            if layer.f_prepare is not None:
                layer.f_prepare(params, vnodelist=layer.nodes, key=rng_key)
        return params

    def copy(self):
        # we only deepcopy the layers, not the networks
        return ComputeStack(self.networks, deepcopy(self.layers))

    def __repr__(self):
        # layers with line breaks
        return '\n'.join([l.__repr__() for l in self.layers])

    def __hash__(self):
        return hash((tuple(self.networks), tuple(self.layers)))

    def build(self, config: ComputeConfigManager, **kwargs):
        with ut.timer('Building compute stack'):
            self.config = config
            with ut.timer('assembling stack'):
                self._assemble_stack(**kwargs)
            self._refresh()
            self._init_quantization_params()
            with ut.timer('building layers'):
                for layer in self.layers:
                    layer.setup(config, stack=self)
                    self._refresh()
            self._generate_apply_method()
            self.check()
            self.is_built = True

    def __call__(self, *args, **kwargs):
        if not self.is_built:
            raise ValueError('Compute stack is not built, can\'t call it')
        res, _ = self.apply(*args, **kwargs)
        return res

    def each_node(self):
        for layer in self.layers:
            for node in layer.nodes:
                yield node

    def make_subset(self, network_ids, pre_init=True):
        """Returns a new stack with only the networks with the given ids"""
        assert self.is_built, 'Stack not built'
        s = ComputeStack([self.networks[i] for i in network_ids])
        s.build(self.config)
        s.shared_store = self.shared_store

        old_node_id = []
        for i, n in enumerate(s.each_node()):
            assert n.node_id == i
            old_net_id = network_ids[n.network_id]
            l_id, n_pos = self.node_map[(old_net_id, n.compute_node_id)]
            old_node_id.append(self.layers[l_id].nodes[n_pos].node_id)

        old_node_id = jnp.array(old_node_id)

        captured_static_p = {}
        if pre_init:
            # generate valid static parameters
            _, captured_static_p = ut.split_params(s.init(jax.random.PRNGKey(0)), [ut.STATIC_PATH])

        def get_param_subset(params):
            static_p = captured_static_p
            if not pre_init:
                # new init to generate valid static parameters
                _, static_p = ut.split_params(s.init(jax.random.PRNGKey(0)), [ut.STATIC_PATH])

            # grab a copy of the shared parameters
            copy_p = deepcopy(params)
            _, shared_p = ut.split_params(copy_p, [ut.SHARED_PATH])

            # truncated node parameters
            _, node_p = ut.split_params(copy_p, [ut.NODE_PATH])
            node_p = jax.tree_map(lambda x: jnp.array(x)[old_node_id], node_p)

            return ut.merge_dicts(static_p, shared_p, node_p)

        return s, get_param_subset


    @staticmethod
    def use_shared_params(base_params, other_params):
        """Returns a copy of the base_params with the shared parameters
        replaced by the ones in other_params"""
        # grab a copy of the shared parameters
        my_p = deepcopy(base_params)
        other_p = deepcopy(other_params)

        _, shared_p = ut.split_params(other_p, [ut.SHARED_PATH])
        _, static_p = ut.split_params(my_p, [ut.STATIC_PATH])
        _, node_p = ut.split_params(my_p, [ut.NODE_PATH])

        my_p = ut.merge_dicts(static_p, shared_p, node_p)
        return my_p

    ##────────────────────────────────────────────────────────────────────────────}}}

    ### {{{                       --    internal utils     --

    def add_layer(self, layer: ComputeLayer):
        self.layers.append(layer)
        return self

    def extend(self, substack):
        assert self.networks == substack.networks
        self.layers.extend(substack.layers)
        return self

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

    def get_all_nodes(self):
        return [n for l in self.layers for n in l.nodes]


    def get_node_input_start_index(self, node: VirtualNode, input_slot: int) -> int:
        """Returns the start index of the input #input_slot for the given node
        in the flattened output array"""
        assert self.node_map is not None

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

    def get_node_output_start_index(self, node: VirtualNode, output_slot: int) -> int:
        """Returns the start index of the output #output_slot for the given node"""
        assert self.node_map is not None

        this_node_layer_id, this_node_pos = self.node_map[(node.network_id, node.compute_node_id)]
        this_layer = self.layers[this_node_layer_id]
        assert len(this_layer.f_out_shapes) > output_slot

        this_layer_start = self.layers_start_index[this_node_layer_id]
        flat_out_size = int(np.sum([np.prod(s) for s in this_layer.f_out_shapes]))

        node_start = this_layer_start + this_node_pos * flat_out_size
        out_shape_till_output = np.sum([np.prod(s) for s in this_layer.f_out_shapes[:output_slot]])

        return int(node_start + out_shape_till_output)

    @staticmethod
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

    @staticmethod
    def make_all_topo_vnodes(networks: list[Network]):
        """Topological_order for all networks"""
        return [
            [
                [VirtualNode.from_node(net_id, net, node_id, b_id) for node_id in node_batch]
                for b_id, node_batch in enumerate(ComputeStack.topological_order(net.compute_graph))
            ]
            for net_id, net in enumerate(networks)
        ]

    @staticmethod
    def get_networks_current_batch_number(stack, type_dict: dict[str, list[VirtualNode]]):
        """Determines the current (minimum) batch number for each network in the stack.
        The batch number is the order in which a node should be computed (topological order of a network).
        type_dict maps node types to the list of all VirtualNode of that type, for all the nodes not yet in the stack.

        Returns a list of the current batch number for each network. If a network has no current batch,
        it means it has no nodes left to be computed and its batch number value will be None.
        """
        MAXINT = 2**60
        current_batches = [MAXINT for _ in stack.networks]
        for nodes in type_dict.values():
            for n in nodes:
                assert n.batch_order < MAXINT
                current_batches[n.network_id] = min(current_batches[n.network_id], n.batch_order)
        current_batches = [None if b == MAXINT else b for b in current_batches]
        return current_batches

    ##────────────────────────────────────────────────────────────────────────────}}}

    ### {{{                         --     building     --{{{

    @staticmethod
    def make_smallest_stack(stack, type_dict: dict[str, list[VirtualNode]], max_t=1):
        # Initialize the BFS queue with the initial state
        bfs_queue = deque([(stack, type_dict, [], 0)])
        iteration = 0

        while bfs_queue:
            iteration += 1

            current_stack, current_type_dict, path, depth = bfs_queue.popleft()

            # current_batches is a list of the current batch number for each network in the stack.
            current_batches = ComputeStack.get_networks_current_batch_number(
                current_stack, current_type_dict
            )
            # pprint(f'current_batches: {current_batches}, depth: {depth}, path: {path}')

            if all(b is None for b in current_batches):  # no nodes left to compute
                return current_stack

            # possible_next_types is a list of types that are candidates for the next layer, i.e they contain
            # nodes that that have a batch_order == current_batches[network_id] for at least one network
            possible_next_types = []
            for t, nodes in current_type_dict.items():
                for n in nodes:
                    can_be_computed = [
                        n for n in nodes if n.batch_order == current_batches[n.network_id]
                    ]
                    if can_be_computed:
                        possible_next_types.append((t, len(can_be_computed)))
                        break

            total_nodes_left = sum([len(nodes) for nodes in current_type_dict.values()])
            # pprint(f'total_nodes_left: {total_nodes_left}')

            if max_t is not None:
                # sort by decreasing number of nodes
                possible_next_types = sorted(possible_next_types, key=lambda x: x[1], reverse=True)
                possible_next_types = possible_next_types[:max_t]

            assert possible_next_types, "No possible next type"
            # pprint(f'possible_next_types: {possible_next_types}')

            # we try every possible type for the next layer
            for t, n in possible_next_types:
                l, new_type_dict = ComputeStack.make_layer_from_current_batches(
                    current_batches, current_type_dict, t
                )
                # l is a ComputeLayer, new_type_dict is a dict[str, list[VirtualNode]]
                # without the nodes that were used in the layer
                node_diff = len(current_type_dict[t]) - len(new_type_dict[t])
                substack = ComputeStack(current_stack.networks, [l])
                path_entry = (
                    f'{t}: picked {node_diff} nodes, {len(possible_next_types)} possible types'
                )

                assert n == node_diff, f'{n} != {node_diff}'

                bfs_queue.append((substack, new_type_dict, path + [path_entry], depth + 1))
                # pprint(f'bfs_queue size: {len(bfs_queue)}')

        # If we reach here, we didn't find a solution
        raise RuntimeError("No solution found")

    @staticmethod
    def make_smallest_stack_dfs(
        stack, type_dict: dict[str, list[VirtualNode]], path=None, depth=0, max_depth=70, max_t=1
    ):

        if path == None:
            path = []

        # current_batches is a list of the current batch number for each network in the stack.
        current_batches = ComputeStack.get_networks_current_batch_number(stack, type_dict)

        if all(b is None for b in current_batches):  # no nodes left to compute
            return stack

        # possible_next_types is a list of types that are candidates for the next layer, i.e they contain
        # nodes that that have a batch_order == current_batches[network_id] for at least one network
        possible_next_types = []
        for t, nodes in type_dict.items():
            for n in nodes:
                can_be_computed = [
                    n for n in nodes if n.batch_order == current_batches[n.network_id]
                ]
                if can_be_computed:
                    possible_next_types.append((t, len(can_be_computed)))
                    break

        # total_nodes_left = sum([len(nodes) for nodes in type_dict.values()])
        # pprint(f'total_nodes_left: {total_nodes_left}')

        if max_t is not None:
            # we're basically doing beam search here, by only keeping the max_t types with the most nodes
            possible_next_types = sorted(possible_next_types, key=lambda x: x[1], reverse=True)
            possible_next_types = possible_next_types[:max_t]

        assert possible_next_types, "No possible next type"
        candidate_stacks = []
        # we try every possible type for the next layer
        for t, _ in possible_next_types:
            l, new_type_dict = ComputeStack.make_layer_from_current_batches(
                current_batches, type_dict, t
            )
            # l is a ComputeLayer, new_type_dict is a dict[str, list[VirtualNode]]
            # without the nodes that were used in the layer
            node_diff = len(type_dict[t]) - len(new_type_dict[t])
            substack = ComputeStack(stack.networks, [l])
            path_entry = f'{t}: picked {node_diff} nodes, {len(possible_next_types)} possible types'
            candidate_stacks.append(
                ComputeStack.make_smallest_stack_dfs(
                    substack, new_type_dict, path + [path_entry], depth + 1, max_depth, max_t
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
        minstack = min(candidate_stacks, key=lambda s: len(s.layers))

        return stack.extend(minstack)

    @staticmethod
    def heuristic(type_dict):
        total_nodes_left = sum(len(nodes) for nodes in type_dict.values())
        if not total_nodes_left:
            return 0

        min_nodes_left = float('inf')
        for t, nodes in type_dict.items():
            nodes_left = sum(1 for n in nodes if n.batch_order is not None)
            min_nodes_left = min(min_nodes_left, nodes_left)

        return min_nodes_left

    @staticmethod
    def make_smallest_stack_astar(
        stack, type_dict: dict[str, list[VirtualNode]], path=None, max_t=2
    ):
        if path == None:
            path = []

        # Initial state
        start_node = (0, (stack, type_dict, path))

        # Priority queue for A* search
        queue = PriorityQueue()
        queue.put(start_node)

        while not queue.empty():
            _, (current_stack, current_type_dict, current_path) = queue.get()

            current_batches = ComputeStack.get_networks_current_batch_number(
                current_stack, current_type_dict
            )

            if all(b is None for b in current_batches):  # no nodes left to compute
                return current_stack

            possible_next_types = []
            for t, nodes in current_type_dict.items():
                for n in nodes:
                    can_be_computed = [
                        n for n in nodes if n.batch_order == current_batches[n.network_id]
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
                l, new_type_dict = ComputeStack.make_layer_from_current_batches(
                    current_batches, current_type_dict, t
                )
                node_diff = len(current_type_dict[t]) - len(new_type_dict[t])
                substack = ComputeStack(current_stack.networks, [l])
                new_path = current_path + [
                    f'{t}: picked {node_diff} nodes, {len(possible_next_types)} possible types'
                ]

                # Calculate the priority for A* search
                cost_so_far = len(substack.layers)
                estimated_cost = ComputeStack.heuristic(new_type_dict)
                priority = cost_so_far + estimated_cost

                queue.put((priority, (substack, new_type_dict, new_path)))

        raise ValueError("No solution found")

    @staticmethod
    def make_layer_from_current_batches(
        current_batches, type_dict: dict[str, list[VirtualNode]], t: str
    ):
        """
        Creates a ComputeLayer from the nodes of type t that have a batch_order <= current_batches[network_id]
        Returns a ComputeLayer and a new type_dict
        without the nodes that were used in the layer"""
        layer_nodes = []
        new_type_dict = deepcopy(type_dict)
        new_type_dict[t] = []
        used = 0
        for n in type_dict[t]:
            if n.batch_order <= current_batches[n.network_id]:
                layer_nodes.append(deepcopy(n))
                used += 1
            else:
                new_type_dict[t].append(deepcopy(n))

        assert used > 0, f'used {used} nodes of type {t} to make layer'
        return ComputeLayer(layer_nodes), new_type_dict

    def _assemble_stack(self, **kwargs):
        n_list = ut.flatten(ComputeStack.make_all_topo_vnodes(self.networks))
        type_dict = {}
        for n in n_list:
            type_dict.setdefault(n.type_signature, []).append(n)
        with ut.timer('make_smallest_stack'):
            minstack = ComputeStack.make_smallest_stack_dfs(
                ComputeStack(self.networks, []), type_dict, **kwargs
            )
            ut.logger.info(f'Final stack size: {len(minstack.layers)}')
        self.layers = minstack.layers

    def _make_layer_input_getters(self, layer_id):
        """Returns a list of functions that return the input values for each node in the given layer
        from the flattened output array"""

        layer = self.layers[layer_id]
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
                    [self.get_node_input_start_index(n, i) for n in layer.nodes]
                    for i in range(len(input_shapes))
                ]
            )

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

    def _refresh(self):
        """Refreshes all the meta information, indexing and mapping of the stack"""
        allbuilt = True
        self.total_nb_of_outputs = 0
        self.total_nb_of_inputs = 0
        self.max_nb_of_outputs_per_network = 0
        for n in self.networks:
            nbout = int(n.get_nb_outputs())
            self.total_nb_of_inputs += n.get_nb_inputs()
            self.total_nb_of_outputs += nbout
            self.max_nb_of_outputs_per_network = max(self.max_nb_of_outputs_per_network, nbout)

        # build a dict of {(net_id, compute_node_id): (layer_id, node_position)}
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

        self.is_assembled = allbuilt

    def _init_quantization_params(self):
        # first we need to store all quantizable parameter names
        self.shared_store = {}
        quantization_params = {}
        for n in self.networks:
            for qn, qv in nd.get_all_possible_quantization_params(n).items():
                quantization_params[qn] = set(qv) | quantization_params.get(qn, set())
        quantization_params = {k: sorted(v) for k, v in quantization_params.items()}
        for pname, pv in quantization_params.items():
            ut.at_path(self.shared_store, ut.QNAME_PATH + f"/{pname}", pv)

    def _generate_apply_method(self, get_grads_for=('translation', 'transcription', 'output')):

        input_getters_f = [self._make_layer_input_getters(l_id) for l_id in range(len(self.layers))]
        node_ids = [jnp.array([n.node_id for n in l.nodes]) for l in self.layers]

        out_indices_and_shapes = [
            self.get_network_output_indices(n_id) for n_id in range(len(self.networks))
        ]

        output_indices = []
        for i, shapes in out_indices_and_shapes:
            assert all([s == (1,) for s in shapes])  # only 1d outputs
            output_indices.append(np.arange(i, i + len(shapes)))
        output_indices = np.concatenate(output_indices)

        w_grads = [l.f_type in get_grads_for for l in self.layers]

        def apply(params, inputs, quantiles, key):
            assert len(inputs) == self.total_nb_of_inputs, 'Mismatch in number of inputs'

            out = inputs.reshape(-1)
            grads = jnp.array([])

            for lid in range(1, len(self.layers)):

                assert out.shape[0] == self.layers_start_index[lid]

                n_inputs = len(self.layers[lid].f_input_shapes)
                layer_inputs = [input_getters_f[lid][i](out) for i in range(n_inputs)]
                keys = jax.random.split(key, len(layer_inputs[0]))

                l_apply = self.layers[lid].f_apply

                def f(node_id, key, *inputs):
                    res = l_apply(
                        *inputs, params=params, quantiles=quantiles, node_id=node_id, key=key
                    )
                    if w_grads[lid]:
                        grad = jax.jacfwd(l_apply, argnums=list(range(n_inputs)))(
                            *inputs, params=params, quantiles=quantiles, node_id=node_id, key=key
                        )
                        grad = jnp.concatenate([g.reshape(-1) for g in grad])
                    else:
                        grad = jnp.array([])
                    return res, grad

                def vmapped(*inputs):
                    return vmap(f)(node_ids[lid], keys, *inputs)

                layer_out, layer_grad = vmapped(*layer_inputs)

                out = jnp.concatenate([out, layer_out.reshape(-1)])
                grads = jnp.concatenate([grads, layer_grad.reshape(-1)])

            return out[output_indices], grads

        self.apply = apply


##────────────────────────────────────────────────────────────────────────────}}}

# }}}
##────────────────────────────────────────────────────────────────────────────}}}
