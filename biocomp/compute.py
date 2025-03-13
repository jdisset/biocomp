### {{{                          --     imports     --
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from queue import PriorityQueue

from collections import deque
from typing import Tuple, List, Dict, Callable, Optional, Union, Any
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import vmap
from jax.tree_util import Partial as partial


from . import nodes as nd
from .network import Network
from . import utils as ut
from biocomp.utils import ArbitraryModel, EncodedPartialFunction
from .parameters import ParameterTree
from . import nodes

from biocomp.logging_config import get_logger

logger = get_logger(__name__)


PRNGKey = Union[jnp.ndarray, np.ndarray, int]
NdArray = Union[jnp.ndarray, np.ndarray]

##────────────────────────────────────────────────────────────────────────────}}}

## {{{                      --     Config    --{{{


class ComputeConfig(ArbitraryModel):
    """
    A ComputeConfig is a set of implementations for the different types of nodes
    that can be found in a network, i.e. a dictionary of {node_name -> function}.
    It also contains extra information that can be
    used by the implementations to store and share information across nodes.
    """

    node_functions: Optional[Dict[str, EncodedPartialFunction]] = None
    extra: Optional[Dict[str, Any]] = None

    def get_node_implementation(self, node_name: str, module_name: str = nd.__name__):
        if self.node_functions is None:
            raise ValueError("No node implementations in this config")
        if node_name not in self.node_functions:
            raise ValueError(f"No node implementation for {node_name}")

        return self.node_functions[node_name].get_impl(extra_module_names=[module_name])


DEFAULT_COMPUTE_CONFIG = ComputeConfig.model_validate(
    {
        "node_functions": {
            "transcription": nodes.transcription,
            "translation": nodes.translation,
            "inv_transcription": nodes.inv_transcription,
            "inv_translation": nodes.inv_translation,
            "sequestron_ERN": nodes.ERN5p,
            "source": nodes.source_new,
            "inv_source": nodes.inv_source_new,
            "bias": nodes.bias,
            "numeric": nodes.bias,
            "aggregation": nodes.aggregation,
            "inv_aggregation": nodes.inv_aggregation,
            "output": nodes.grouped_output,
            "deadend": nodes.single_passthrough,
        }
    }
)


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                       --     Virtual Node     --


@dataclass
class VirtualNode:
    """A virtual node does match with a compute node in the network, but it is
    used to represent a node in the stack, with a unique id, and a type signature
    that depends on the topology of the network. It also has a batch_order, which
    is used to sort the nodes in the stack, so that the nodes that need to be computed
    first are at the top of the stack."""

    network: Optional[Network] = None
    network_id: Optional[int] = None
    type_signature: Optional[str] = None  # type of node, and number of inputs and outputs
    compute_node_id: Optional[int] = None  # id of the compute node in the network
    node_id: Optional[int] = None  # unique id for the node in the stack
    batch_order: Optional[int] = 0  # only used for sorting and debugging

    @staticmethod
    def generate_type_signature(network: Network, compute_node_id: int) -> str:
        assert network.compute_graph is not None, "No compute graph"
        ntype = network.compute_graph.at[compute_node_id, "type"]
        n_inputs = len(network.compute_graph.at[compute_node_id, "input_from"])
        n_outputs = len(network.compute_graph.at[compute_node_id, "output_to"])
        return f"{ntype}_{n_inputs}_{n_outputs}"

    @classmethod
    def from_node(
        cls, network_id: int, network: Network, compute_node_id: int, batch_order: int = 0
    ) -> VirtualNode:
        type_signature = VirtualNode.generate_type_signature(network, compute_node_id)
        return cls(
            network_id=network_id,
            network=network,
            compute_node_id=compute_node_id,
            type_signature=type_signature,
            batch_order=batch_order,
        )

    def set_compute_node_column(self, column_name: str, value: Any):
        assert self.network is not None, "No network"
        assert self.compute_node_id is not None, "No compute node id"
        assert self.network.compute_graph is not None, "No compute graph"
        self.network.compute_graph.at[self.compute_node_id, column_name] = value

    def get_compute_node(self, column_name: Optional[str] = None) -> Optional[Any]:
        if self.network is None:
            return None
        assert self.compute_node_id is not None, "No compute node id"
        assert self.network.compute_graph is not None, "No compute graph"
        if column_name is None:
            return self.network.compute_graph.loc[self.compute_node_id]
        else:
            return self.network.compute_graph.at[self.compute_node_id, column_name]

    def get_inverse_node(self, stack: ComputeStack) -> VirtualNode:
        is_inverse_of = self.get_compute_node("is_inverse_of")
        assert isinstance(is_inverse_of, int), "Node is not an inverse"
        assert is_inverse_of is not None, "Node is not an inverse"
        assert self.network_id is not None, "No network id"
        inv = stack.get_node_from_net_and_compute_id(self.network_id, is_inverse_of)
        assert inv is not None, "Inverse not found"
        return inv

    def get_layer_and_local_id(self, stack):
        if stack is None:
            return None, None
        return stack.node_map[(self.network_id, self.compute_node_id)]

    def __repr__(self):
        out = f"{self.network_id}/{self.compute_node_id}/{self.node_id if self.node_id is not None else self.batch_order}-{self.type_signature}"
        return f"{out}"

    def __hash__(self):
        return hash((self.network_id, self.node_id, self.type_signature, self.batch_order))

    def __deepcopy__(self, memo):
        # copy everything except the network (just a reference)
        new_obj = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_obj
        for k, v in self.__dict__.items():
            if k == "network":
                setattr(new_obj, k, v)
            else:
                setattr(new_obj, k, deepcopy(v, memo))
        return new_obj


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                       --     Compute Layer     --
NodeInput = Tuple[int, int, int]  #  (net_id, compute_node_id, slot_id)


@dataclass
class ComputeLayer:
    nodes: List[VirtualNode]
    layer_id: Optional[int] = None

    # information about the function to apply
    f_type: Optional[str] = None
    f_out_shapes: Optional[List[Tuple[int]]] = None
    f_input_shapes: Optional[List[Tuple[int]]] = None

    f_prepare: Optional[Callable] = None
    f_apply: Optional[Callable] = None
    f_commit: Optional[Callable] = None

    is_built: bool = False

    def setup(self, config: ComputeConfig, stack: ComputeStack):
        self.check()

        self.f_type = self.nodes[0].get_compute_node("type")

        if self.f_type == "input":
            self.f_out_shapes = [(1,)]
            self.f_input_shapes = [(1,)]
            self.is_built = True
            return

        # get the shapes of the inputs. We'll collect all the inputs for each node
        # to make sure they are all the same
        node_inputs: List[List[NodeInput]] = []
        for n in self.nodes:
            ninp = n.get_compute_node("input_from")
            node_inputs.append([(n.network_id, *i) for i in ninp])

        # get the shapes of the inputs
        all_input_shapes = []  # list of list of shapes
        for n_inp in node_inputs:
            input_shapes = []
            for input_net_id, input_compute_node_id, input_slot_id in n_inp:
                input_layer_id, _ = stack.node_map[(input_net_id, input_compute_node_id)]
                assert input_layer_id < self.layer_id, "Input node is in a later layer"
                assert stack.layers[input_layer_id].is_built, "Input layer is not built"
                input_layer_output_shapes = stack.layers[input_layer_id].f_out_shapes
                assert input_slot_id < len(
                    input_layer_output_shapes
                ), f"Input slot {input_slot_id} is out of range"
                shape = (
                    tuple(input_layer_output_shapes[input_slot_id])
                    if isinstance(input_layer_output_shapes[input_slot_id], list)
                    else input_layer_output_shapes[input_slot_id]
                )
                input_shapes.append(shape)
            all_input_shapes.append(tuple(input_shapes))
        # they should all be the same
        assert len(set(all_input_shapes)) == 1, f"Input shapes are not the same: {all_input_shapes}"
        self.f_input_shapes = all_input_shapes[0]

        n_outputs = self.get_n_outputs()

        impl = config.get_node_implementation(self.f_type)(
            input_shapes=self.f_input_shapes,
            n_outputs=n_outputs,
            stack=stack,
            layer_id=self.layer_id,
        )

        self.f_prepare = impl.prepare
        self.f_apply = impl.apply
        self.f_out_shapes = impl.output_shapes
        self.f_commit = impl.commit
        self.is_built = True

    def get_n_outputs(self):
        output_to = self.nodes[0].get_compute_node("output_to")
        return len(output_to)

    def flattened_output_shape(self):
        return int(len(self.nodes) * np.sum([np.prod(s) for s in self.f_out_shapes]))

    def type_str(self):
        return self.nodes[0].get_compute_node("type")

    def __repr__(self):
        ftype = self.type_str()
        return f"Layer {self.layer_id} ({ftype}) with {len(self.nodes)} nodes"

    def __hash__(self):
        return hash(tuple(self.nodes))

    def check(self):
        assert len(set(n.type_signature for n in self.nodes)) == 1, "Different types in layer"

    def commit(self, params: ParameterTree):
        if self.f_commit is not None:
            self.f_commit(params, self.nodes)


##────────────────────────────────────────────────────────────────────────────}}}


@dataclass
class ComputeStack:
    networks: List[Network]
    layers: Optional[List[ComputeLayer]] = None

    layers_start_index: Optional[List[int]] = None
    output_shape: Optional[Tuple[int]] = None

    # node_map is (network_id, compute_node_id) -> (layer_id, node_loc)
    node_map: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None

    total_nb_of_outputs: Optional[int] = None
    total_nb_of_inputs: Optional[int] = None
    max_nb_of_outputs_per_network: Optional[int] = None

    is_assembled: bool = False
    is_built: bool = False
    number_of_nodes: int = 0

    apply: Optional[Callable] = None

    post_process_callbacks: Optional[List[Callable]] = None

    ### {{{                     --     public interface     --

    def build(self, config: ComputeConfig, **kwargs):
        """
        Split apart all the networks into their constituent nodes
        and put these nodes into ordered layers, maximizing parallelism by ensuring same-type nodes
        (potentially from different networks) are in the same layer.
        Then generates the apply method for the stack, which will handle the parallel execution
        of all nodes in each layers, as well as the correct mapping and chaining of
        their inputs and outputs.
        """
        with ut.timer("Building compute stack", logger):
            self.config = config
            self._assemble_stack(**kwargs)
            self._refresh()
            assert self.layers is not None, "No layers"
            for layer in self.layers:
                layer.setup(config, stack=self)
                self._refresh()
            self._generate_apply_method()
            self.check()
            self.is_built = True

    def init(self, rng_key: PRNGKey) -> ParameterTree:
        """
        Generates a randomly initilized dictionary of parameters for the stack
        """
        assert self.is_built, "Stack not built"
        params = ParameterTree()
        # for l_id, layer in enumerate(self.layers):
        # let's initialize the stacj it in reverse order
        # so that the inverse nodes can reference fwd ones after init
        assert self.layers is not None, "No layers"
        assert self.layers_start_index is not None, "No layers start index"
        for l_id, layer in reversed(list(enumerate(self.layers))):
            assert layer.is_built, "Layer not built"
            assert l_id == layer.layer_id, "Layer id mismatch"
            rng_key, _ = jax.random.split(rng_key)
            logger.debug(
                f"Initializing {len(layer.nodes)} nodes in layer {l_id}/{len(self.layers)}"
            )
            logger.debug(f"Layer type: {layer.f_type}")
            logger.debug(f"Layer input shapes: {layer.f_input_shapes}")
            logger.debug(f"Layer output shapes: {layer.f_out_shapes}")
            if layer.f_prepare is not None:
                try:
                    layer.f_prepare(params, nodelist=layer.nodes, key=rng_key)
                except Exception as e:
                    logger.error(f"Error in layer {l_id} preparation:")
                    logger.error(f"Layer type: {layer.f_type}")
                    logger.error(f"Layer input shapes: {layer.f_input_shapes}")
                    logger.error(f"Layer output shapes: {layer.f_out_shapes}")
                    raise e
        params.tag("local", "local")
        params.tag("shared", "shared")
        # pp_params = self.post_process(params)
        # assert pp_params == params, 'Post process changed params'
        return params

    def get_network_output_indices(self, network_id: int):
        """Returns the start index and shape of the output of the given network in
        the flattened array of all outputs of all nodes in the stack.
        """
        assert self.node_map is not None, "No node map"
        assert self.layers_start_index is not None, "No layers start index"
        assert self.layers is not None, "No layers"
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

    def get_node_from_net_and_compute_id(
        self, network_id: int, compute_node_id: int
    ) -> VirtualNode:
        """Returns the virtual node corresponding to the given network and compute node ids"""
        assert self.node_map is not None, "No node map"
        assert self.layers is not None, "Stack has no layers"
        assert (
            network_id,
            compute_node_id,
        ) in self.node_map, f"Node not found: {network_id}/{compute_node_id}"
        layer_id, node_loc = self.node_map[(network_id, compute_node_id)]
        return self.layers[layer_id].nodes[node_loc]

    def copy(self):
        # we only deepcopy the layers, not the networks
        return ComputeStack(self.networks, deepcopy(self.layers))

    def commit(self, params: ParameterTree):
        for layer in self.layers:
            layer.commit(params)

    def __repr__(self):
        # layers with line breaks
        if self.layers is None:
            return "Empty stack"
        return "\n".join([l.__repr__() for l in self.layers])

    def __hash__(self):
        if self.layers is None:
            return hash(tuple(self.networks))
        return hash((tuple(self.networks), tuple(self.layers)))

    def __call__(self, *args, **kwargs):
        if not self.is_built:
            raise ValueError("Compute stack is not built, can't call it")
        assert self.apply is not None, "No apply method"
        res, _ = self.apply(*args, **kwargs)
        return res

    def each_node(self):
        assert self.layers is not None, "Stack has no layers"
        for layer in self.layers:
            for node in layer.nodes:
                yield node

    def register_post_process(self, callback: Callable):
        if self.post_process_callbacks is None:
            self.post_process_callbacks = []
        self.post_process_callbacks.append(callback)

    @partial(jax.jit, static_argnums=(0,))
    def post_process(self, params: ParameterTree):
        if self.post_process_callbacks is not None:
            for callback in self.post_process_callbacks:
                params = callback(params)
        return params

    ##────────────────────────────────────────────────────────────────────────────}}}

    ### {{{                       --    internal utils     --

    def add_layer(self, layer: ComputeLayer):
        if self.layers is None:
            self.layers = []
        self.layers.append(layer)
        return self

    def extend(self, substack):
        assert self.networks == substack.networks, "Networks don't match"
        if self.layers is None:
            self.layers = []
        self.layers.extend(substack.layers)
        return self

    def check(self):
        for l in self.layers:
            l.check()
            for n in l.nodes:
                assert id(n.network) == id(self.networks[n.network_id]), "Network mismatch"
        assert (
            self.layers[0].nodes[0].get_compute_node().type == "input"
        ), f"First node is not input: {self.layers[0].nodes[0]}"
        for net_id in range(len(self.networks)):
            prev = -1
            for l in self.layers:
                for n in l.nodes:
                    if n.network_id == net_id:
                        assert (
                            n.batch_order >= prev
                        ), f"wrong batch order ({n.batch_order} < {prev} for {n})"
                        prev = n.batch_order

    def get_all_nodes(self):
        if self.layers is None:
            return []
        return [n for l in self.layers for n in l.nodes]

    def get_node_input_start_index(self, node: VirtualNode, input_slot: int) -> int:
        """Returns the start index of the input #input_slot for the given node
        in the flattened full-stack output array"""
        assert self.node_map is not None, "No node map"
        if self.layers is None:
            raise ValueError("No layers")

        input_compute_node_id, input_compute_node_outslot = node.get_compute_node("input_from")[
            input_slot
        ]

        input_layer_id, input_node_layer_loc = self.node_map[
            (node.network_id, input_compute_node_id)
        ]
        this_node_layer_id, _ = self.node_map[(node.network_id, node.compute_node_id)]

        assert input_layer_id < this_node_layer_id, "input layer must be before this layer"

        this_layer = self.layers[this_node_layer_id]
        input_layer = self.layers[input_layer_id]

        this_input_shapes = this_layer.f_input_shapes
        input_layer_start = self.layers_start_index[input_layer_id]

        assert (
            this_input_shapes[input_slot] == input_layer.f_out_shapes[input_compute_node_outslot]
        ), f"Shapes don't match: {this_input_shapes[input_slot]} != {input_layer.f_out_shapes[input_compute_node_outslot]}"

        flat_out_size = int(np.sum([np.prod(s) for s in input_layer.f_out_shapes]))
        node_start = input_layer_start + input_node_layer_loc * flat_out_size

        flat_output_shape_till_input = np.sum(
            [np.prod(s) for s in input_layer.f_out_shapes[:input_compute_node_outslot]]
        )
        outslot_start = node_start + flat_output_shape_till_input

        return int(outslot_start)

    def get_node_output_start_index(self, node: VirtualNode, output_slot: int) -> int:
        """Returns the start index of the output #output_slot for the given node
        in the flattened full-stack output array"""
        assert self.node_map is not None, "No node map"

        this_node_layer_id, this_node_pos = self.node_map[(node.network_id, node.compute_node_id)]
        this_layer = self.layers[this_node_layer_id]
        assert len(this_layer.f_out_shapes) > output_slot, "Output slot out of range"

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
                if (not row["input_from"] or all([x[0] in visited for x in row["input_from"]]))
                and i not in visited
            ]
            if not independent:
                msg = f"No independent node. Remaining:{set(graph.index) - visited}. Visited:{visited}"
                raise ValueError(msg)
            visited.update(independent)
            batches.append(independent)
        return batches

    @staticmethod
    def make_all_topo_nodes(networks: List[Network]):
        """Topological_order for all networks"""
        return [
            [
                [VirtualNode.from_node(net_id, net, node_id, b_id) for node_id in node_batch]
                for b_id, node_batch in enumerate(ComputeStack.topological_order(net.compute_graph))
            ]
            for net_id, net in enumerate(networks)
        ]

    @staticmethod
    def get_networks_current_batch_number(
        stack: ComputeStack, type_dict: dict[str, list[VirtualNode]]
    ):
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
                assert n.batch_order < MAXINT, "Node has no batch order"
                current_batches[n.network_id] = min(current_batches[n.network_id], n.batch_order)
        current_batches = [None if b == MAXINT else b for b in current_batches]
        return current_batches

    ##────────────────────────────────────────────────────────────────────────────}}}

    ### {{{                         --     building     --{{{

    @staticmethod
    def make_smallest_stack(
        stack: ComputeStack, type_dict: Dict[str, List[VirtualNode]], max_t: int = 1
    ):
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

            if all(b is None for b in current_batches):  # no nodes left to compute
                return current_stack

            # possible_next_types is a list of types that are candidates for the next layer,
            # i.e they contain nodes that that have a batch_order == current_batches[network_id]
            # for at least one network
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
                # sort by decreasing number of nodes
                possible_next_types = sorted(possible_next_types, key=lambda x: x[1], reverse=True)
                possible_next_types = possible_next_types[:max_t]

            assert possible_next_types, "No possible next type"

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
                    f"{t}: picked {node_diff} nodes, {len(possible_next_types)} possible types"
                )

                assert n == node_diff, f"{n} != {node_diff}"

                bfs_queue.append((substack, new_type_dict, path + [path_entry], depth + 1))

        # If we reach here, we didn't find a solution
        raise RuntimeError("No solution found")

    @staticmethod
    def make_smallest_stack_dfs(
        stack: ComputeStack,
        type_dict: Dict[str, List[VirtualNode]],
        path=None,
        depth=0,
        max_depth=70,
        max_t=1,
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
            path_entry = f"{t}: picked {node_diff} nodes, {len(possible_next_types)} possible types"
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

        min_nodes_left = float("inf")
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
                    f"{t}: picked {node_diff} nodes, {len(possible_next_types)} possible types"
                ]

                # Calculate the priority for A* search
                cost_so_far = len(substack.layers) if substack.layers else 0
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

        assert used > 0, f"used {used} nodes of type {t} to make layer"
        return ComputeLayer(layer_nodes), new_type_dict

    def _assemble_stack(self, **kwargs):
        n_list = ut.flatten(ComputeStack.make_all_topo_nodes(self.networks))
        type_dict = {}
        for n in n_list:
            type_dict.setdefault(n.type_signature, []).append(n)
        minstack = ComputeStack.make_smallest_stack_dfs(
            ComputeStack(self.networks, []), type_dict, **kwargs
        )
        logger.debug(f"Final stack size: {len(minstack.layers) if minstack.layers else 0}")
        self.layers = minstack.layers

    def make_layer_input_getters(self, layer_id: int):
        """Returns a list of input_getter functions that return the input values for each node in the given layer
        from the flattened output array"""
        assert self.layers is not None, "No layers"
        assert len(self.layers) > layer_id, "Layer id out of range"

        layer = self.layers[layer_id]
        assert layer.is_built, "Layer not built"
        input_shapes = layer.f_input_shapes  # list of tuples, one for each input
        input_lengths = [int(np.prod(s)) for s in input_shapes]  # flattened length of each input

        # input layer is a special case as it doesn't take values from the stack output
        if layer.f_type == "input":
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
        assert self.layers is not None, "No layers"
        allbuilt = True
        self.total_nb_of_outputs = 0
        self.total_nb_of_inputs = 0
        self.max_nb_of_outputs_per_network = 0
        for n in self.networks:
            nbout = int(n.nb_outputs)
            self.total_nb_of_inputs += n.nb_inputs
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

    def _generate_apply_method(
        self,
        get_grads_for: List[str] = (
            "translation",
            "transcription",
            "output",
            "source_new",
            "source",
        ),
    ):
        """
        Generates the apply method, which will call the apply of all layers of the stack
        with the correct chaining of input/outputs.

        During an apply pass, we maintain and update a stack output array, which is a flat array
        of all the outputs of all the nodes in the stack. The output array is updated in place
        during the apply pass.

        All nodes in a layer are applied in parallel (using vmap)
        and write their outputs to the stack output array at the correct position.
        The correct inputs to the next layers are fetched from the stack output array
        using the input_getters_f functions.

        get_grads_for is a list of node types for which we want to compute gradients wrt their inputs,
        which is useful to enforce monotonicity of certain node types (e.g. translation, transcription)

        """

        # input_getters_f is a list of functions that return the input values for each node in a layer
        # from the array of all outputs of the stack. Of course this assumes correct ordering, i.e
        # each layer is taking inputs from a layer that has already been applied.

        assert self.layers is not None, "No layers"
        assert self.layers_start_index is not None, "No layers start index"
        assert self.node_map is not None, "No node map"

        input_getters_f = [self.make_layer_input_getters(l_id) for l_id in range(len(self.layers))]

        out_indices_and_shapes = [
            self.get_network_output_indices(n_id) for n_id in range(len(self.networks))
        ]

        output_indices = []
        for i, shapes in out_indices_and_shapes:
            assert all([s == (1,) for s in shapes])  # only 1d outputs
            output_indices.append(np.arange(i, i + len(shapes)))
        output_indices = np.concatenate(output_indices)

        w_grads = [l.f_type in get_grads_for for l in self.layers]

        def apply_impl(
            params: ParameterTree,
            inputs: NdArray,
            quantiles: NdArray,
            key: PRNGKey,
            overwrite_values: Optional[NdArray] = None,
            overwrite_at: Optional[NdArray] = None,
        ) -> Tuple[NdArray, NdArray]:
            """
            The core of the apply method. Jittable. Applies the entire stack.
            Overwrite stuff:
                - added overwrite_* to allow injecting values at specific indices
                - allows feeding whatever values we want to some specific nodes
                - overwrite_values is a 1d array of values to inject
                - overwrite_at is a 1d array of indices where to inject the values
            """

            assert len(inputs) == self.total_nb_of_inputs, "Mismatched number of inputs"
            assert self.layers is not None, "No layers"
            assert self.layers_start_index is not None, "No layers start index"
            assert self.node_map is not None, "No node map"

            running_output = inputs.reshape(-1)
            grads = jnp.array([])

            for lid in range(1, len(self.layers)):  # skip the input layer
                assert running_output.shape[0] == self.layers_start_index[lid]

                n_nodes = len(self.layers[lid].nodes)

                n_inputs = len(self.layers[lid].f_input_shapes)

                if overwrite_values is not None and overwrite_at is not None:
                    assert overwrite_values.ndim == 1 and overwrite_at.ndim == 1
                    assert (
                        overwrite_values.shape[0] == overwrite_at.shape[0]
                    ), f"{overwrite_values.shape[0]=} != {overwrite_at.shape[0]=}"
                    running_output = running_output.at[overwrite_at].set(overwrite_values)

                # fetch the inputs for each node in the layer from the output array
                # (which was filled by the previous layers)
                layer_inputs = [input_getters_f[lid][i](running_output) for i in range(n_inputs)]

                assert len(layer_inputs) == n_inputs

                keys = jax.random.split(key, n_nodes)

                apply_f = self.layers[lid].f_apply

                def node_apply(node_id: ArrayLike, key: PRNGKey, *inputs: ArrayLike):
                    res = apply_f(
                        *inputs, params=params, quantiles=quantiles, node_id=node_id, key=key
                    )
                    if w_grads[lid]:
                        # compute the gradient of the first output with respect to the first input

                        # def first_output(
                        #     first_input, *other_inputs, params, quantiles, node_id, key
                        # ):
                        #     return apply_f(
                        #         first_input,
                        #         *other_inputs,
                        #         params=params,
                        #         quantiles=quantiles,
                        #         node_id=node_id,
                        #         key=key,
                        #     )[0][0]
                        # grad = jax.grad(first_output)(
                        #     inputs[0],
                        #     *inputs[1:],
                        #     params=params,
                        #     quantiles=quantiles,
                        #     node_id=node_id,
                        #     key=key,
                        # )
                        # grad = jnp.concatenate([g.reshape(-1) for g in grad])

                        # or using jax.jacfwd, we can just grab the first output wrt the first input

                        grad = jax.jacfwd(apply_f, argnums=list(range(n_inputs)))(
                            *inputs, params=params, quantiles=quantiles, node_id=node_id, key=key
                        )
                        first_output_grad = grad[0][0]
                        grad = first_output_grad.reshape(-1)

                    else:
                        grad = jnp.array([])
                    return res, grad

                def layer_apply(*inputs):
                    return vmap(node_apply)(jnp.arange(n_nodes), keys, *inputs)

                layer_out, layer_grad = layer_apply(*layer_inputs)
                flattened_layer_output = layer_out.reshape(-1)

                if self.layers[lid].flattened_output_shape() != len(flattened_layer_output):
                    raise ValueError(
                        f"Output shape mismatch: {self.layers[lid].flattened_output_shape()=} != "
                        f"{len(flattened_layer_output)=}"
                    )

                running_output = jnp.concatenate([running_output, flattened_layer_output])
                grads = jnp.concatenate([grads, layer_grad.reshape(-1)])

            return running_output, grads

        def apply(*args, **kwargs):
            # returns only the final outputs (of the last layer) + the gradients + the full trace
            o, g = apply_impl(*args, **kwargs)
            return o[output_indices], (g, o)

        self.apply = apply
        self.output_indices = output_indices


##────────────────────────────────────────────────────────────────────────────}}}
