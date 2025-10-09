### {{{                          --     imports     --
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Optional, Union, Any, TypeVar
import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax.tree_util import Partial as partial
from jax.typing import ArrayLike

from . import nodes as nd
from .network import Network
from . import utils as ut
from biocomp.utils import ArbitraryModel, EncodedPartialFunction
from .parameters import ParameterTree
from . import nodes
from .graphengine import GraphState

from biocomp.logging_config import get_logger

logger = get_logger(__name__)


PRNGKey = Union[jnp.ndarray, np.ndarray, int]
NdArray = Union[jnp.ndarray, np.ndarray]

T = TypeVar("T")


@dataclass(frozen=True)
class StackNode:
    """Lightweight, immutable reference to a specific node within a specific network in the stack."""

    network_id: int  # id of the network in the stack
    node_id: int  # id of the node in the network's compute graph
    layer_number: Optional[int] = None  # what layer in the stack this node is in
    node_position_in_layer: Optional[int] = None  # what position in the layer this node is in

    @staticmethod
    def generate_type_signature(graph: GraphState, node_id: int) -> str:
        """Generate a type signature for a node based on its type and number of inputs/outputs"""
        node = graph.nodes[node_id]
        n_inputs = len(graph.get_incoming_edges(node_id))
        n_outputs = graph.get_nb_outgoing_slots(node_id)
        return f"{node.node_type}_{n_inputs}_{n_outputs}"

    def get(self, stack: "ComputeStack") -> nd.ComputeNode:
        """Get the actual ComputeNode object from the stack"""
        assert stack.networks is not None, "Stack has no networks"
        assert self.network_id < len(stack.networks)
        cg = stack.networks[self.network_id].compute_graph
        assert cg is not None
        return cg.get_node(self.node_id)

    def get_forward_stacknode(self, stack: "ComputeStack") -> nd.StackNode:
        """Get the stack node that this node is an inverse of"""
        node = self.get(stack)
        assert node.is_inverse_of is not None, "Node has no inverse"
        assert stack.networks is not None, "Stack has no networks"
        assert self.network_id < len(stack.networks)
        return stack.get_node_from_net_and_compute_id(self.network_id, node.is_inverse_of.node_id)

    def get_outgoing_edges(self, stack: "ComputeStack") -> list[nd.ComputeEdge]:
        """Get the outgoing edges of the node from the stack"""
        assert stack.networks is not None, "Stack has no networks"
        assert self.network_id < len(stack.networks)
        cg = stack.networks[self.network_id].compute_graph
        assert cg is not None
        return cg.get_outgoing_edges(self.node_id)

    def get_incoming_edges(self, stack: "ComputeStack") -> list[nd.ComputeEdge]:
        """Get the incoming edges of the node from the stack"""
        assert stack.networks is not None, "Stack has no networks"
        assert self.network_id < len(stack.networks)
        cg = stack.networks[self.network_id].compute_graph
        assert cg is not None
        return cg.get_incoming_edges(self.node_id)

    def get_nb_outputs(self, stack: "ComputeStack") -> int:
        assert stack.networks is not None, "Stack has no networks"
        assert self.network_id < len(stack.networks)
        cg = stack.networks[self.network_id].compute_graph
        assert cg is not None
        # return number of unique output slots (not edges, as one slot can have multiple edges)
        return cg.get_nb_outgoing_slots(self.node_id)


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                      --     Config    --{{{


class ComputeConfig(ArbitraryModel):
    """
    A ComputeConfig is a set of implementations for the different types of nodes
    that can be found in a network, i.e. a dictionary of {node_name -> function}.
    It also contains extra information that can be
    used by the implementations to store and share information across nodes.
    """

    node_functions: Optional[dict[str, EncodedPartialFunction]] = None
    extra: Optional[dict[str, Any]] = None

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
            "source": nodes.source_with_pos,
            "inv_source": nodes.inv_source_with_pos,
            "bias": nodes.hard_bias,
            "numeric": nodes.hard_bias,
            "aggregation": nodes.aggregation,
            "inv_aggregation": nodes.inv_aggregation,
            "output": nodes.grouped_output,
            "deadend": nodes.single_passthrough,
        }
    }
)


##────────────────────────────────────────────────────────────────────────────}}}

##────────────────────────────────────────────────────────────────────────────}}}

## {{{                       --     Compute Layer     --
NodeInput = tuple[int, int, int]  #  (net_id, compute_node_id, slot_id)


@dataclass
class StackLayer:
    nodes: list[StackNode]
    stack: Optional["ComputeStack"] = None
    layer_id: Optional[int] = None

    # information about the function to apply
    f_type: Optional[str] = None
    f_out_shapes: Optional[list[tuple[int]]] = None
    f_input_shapes: Optional[list[tuple[int]]] = None

    f_prepare: Optional[Callable] = None
    f_apply: Optional[Callable] = None
    f_commit: Optional[Callable] = None

    is_built: bool = False

    def setup(self, config: ComputeConfig, stack: ComputeStack):
        self.check()

        first_key = self.nodes[0]
        first_graph = stack.networks[first_key.network_id].compute_graph
        self.f_type = first_graph.nodes[first_key.node_id].node_type

        if self.f_type == "input":
            self.f_out_shapes = [(1,)]
            self.f_input_shapes = [(1,)]
            self.is_built = True
            return

        # get the shapes of the inputs. We'll collect all the inputs for each node
        # to make sure they are all the same
        node_inputs: list[list[NodeInput]] = []
        for key in self.nodes:
            graph = stack.networks[key.network_id].compute_graph
            incoming_edges = graph.get_incoming_edges(key.node_id)
            # sort by input_slot to match old behavior
            incoming_edges_sorted = sorted(incoming_edges, key=lambda e: e.input_slot)
            ninp = [(edge.source_id, edge.output_slot) for edge in incoming_edges_sorted]
            node_inputs.append([(key.network_id, *i) for i in ninp])

        # get the shapes of the inputs
        all_input_shapes = []  # list of list of shapes
        for n_inp in node_inputs:
            input_shapes = []
            for input_net_id, input_compute_node_id, input_slot_id in n_inp:
                input_layer_id, _ = stack.node_map[(input_net_id, input_compute_node_id)]
                assert input_layer_id < self.layer_id, "Input node is in a later layer"
                assert stack.layers[input_layer_id].is_built, "Input layer is not built"
                input_layer_output_shapes = stack.layers[input_layer_id].f_out_shapes
                assert input_slot_id < len(input_layer_output_shapes), (
                    f"Input slot {input_slot_id} is out of range"
                )
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

    def get_n_outputs(self) -> int:
        """Get the number of outputs for nodes in this layer (all nodes in a layer have the same type signature)"""
        first_key = self.nodes[0]
        graph = self.stack.networks[first_key.network_id].compute_graph
        return len(graph.get_outgoing_edges(first_key.node_id))

    def flattened_output_shape(self) -> int:
        return int(len(self.nodes) * np.sum([np.prod(s) for s in self.f_out_shapes]))

    def type_str(self) -> str:
        first_key = self.nodes[0]
        return (
            self.stack.networks[first_key.network_id]
            .compute_graph.nodes[first_key.node_id]
            .node_type
        )

    def __repr__(self):
        ftype = self.type_str()
        return f"Layer {self.layer_id} ({ftype}) with {len(self.nodes)} nodes"

    def __hash__(self):
        return hash(tuple(self.nodes))

    def check(self):
        """Ensure all nodes in the layer have the same type signature"""
        type_sigs = {
            StackNode.generate_type_signature(
                self.stack.networks[key.network_id].compute_graph, key.node_id
            )
            for key in self.nodes
        }
        assert len(type_sigs) == 1, f"Different types in layer: {type_sigs}"

    def commit(self, params: ParameterTree, stack: "ComputeStack", **kwargs):
        if self.f_commit is not None:
            self.f_commit(params, self.nodes, stack, **kwargs)


##────────────────────────────────────────────────────────────────────────────}}}


@dataclass
class ComputeStack:
    networks: list[Network]
    layers: Optional[list[StackLayer]] = None

    layers_start_index: Optional[list[int]] = None
    output_shape: Optional[tuple[int]] = None

    # node_map is (network_id, compute_node_id) -> (layer_id, node_loc)
    node_map: Optional[dict[tuple[int, int], tuple[int, int]]] = None

    total_nb_of_outputs: Optional[int] = None
    total_nb_of_inputs: Optional[int] = None
    max_nb_of_outputs_per_network: Optional[int] = None

    is_assembled: bool = False
    is_built: bool = False
    number_of_nodes: int = 0

    apply: Optional[Callable] = None

    post_process_callbacks: Optional[list[Callable]] = None

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
        Generates a randomly initilized parameter tree for the stack
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
                    logger.exception(e)
                    raise e
        params.tag("local", "local")
        params.tag("shared", "shared")

        params.at(
            "global/dependent_output_mask", self.get_dependent_output_mask(), tags=[nd.NON_GRAD_TAG]
        )

        # pp_params = self.post_process(params)
        # assert pp_params == params, 'Post process changed params'
        return params

    def get_nb_networks(self) -> int:
        return len(self.networks)

    def get_nb_outputs(self) -> int:
        """
        Returns the total number of outputs in the stack.
        This is the sum of the number of outputs of all networks.
        """
        if self.total_nb_of_outputs is None:
            self.total_nb_of_outputs = sum(n.nb_outputs for n in self.networks)
        return self.total_nb_of_outputs

    def get_nb_inputs(self) -> int:
        """
        Returns the total number of inputs in the stack.
        This is the sum of the number of inputs of all networks.
        """
        if self.total_nb_of_inputs is None:
            self.total_nb_of_inputs = sum(n.nb_inputs for n in self.networks)
        return self.total_nb_of_inputs

    def get_dependent_output_mask(self):
        """
        Get a mask that indicates which outputs are dependent on the inputs.
        """
        m = np.concatenate([n.get_dependent_output_mask() for n in self.networks])
        assert m.shape == (sum(n.nb_outputs for n in self.networks),)
        return m

    def get_nb_dependent_outputs(self) -> int:
        return np.sum(self.get_dependent_output_mask())

    def get_network_output_indices(self, network_id: int):
        """Returns the start index and shape of the output of the given network in
        the flattened array of all outputs of all nodes in the stack.
        """
        assert self.node_map is not None, "No node map"
        assert self.layers_start_index is not None, "No layers start index"
        assert self.layers is not None, "No layers"
        output_node = self.networks[network_id].get_output_compute_node()
        node_id = output_node.node_id
        layer_id, node_loc = self.node_map[(network_id, node_id)]
        out_shape = self.layers[layer_id].f_out_shapes
        start_index = self.layers_start_index[layer_id] + node_loc * np.sum(
            [np.prod(s) for s in out_shape]
        )
        return int(start_index), out_shape

    def get_node_from_net_and_compute_id(self, network_id: int, compute_node_id: int) -> StackNode:
        """Returns the NodeKey corresponding to the given network and compute node ids"""
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

    def commit(self, params: ParameterTree, **kwargs):
        # create copies of all networks
        network_copies = [deepcopy(net) for net in self.networks]

        # create a temporary stack with network copies for commit operations
        temp_stack = ComputeStack(network_copies, self.layers)
        temp_stack.node_map = self.node_map

        # run commit on all layers (will modify the network copies)
        for layer in self.layers:
            layer.commit(params, stack=temp_stack, **kwargs)

        # return the modified network copies
        return network_copies

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

    def split_stack_outputs_per_network(
        self, yhat: T, max_samples: Optional[int] = None
    ) -> list[T]:
        """
        split a stacked output into per-network outputs

        returns:
            list of per-network output arrays
        """

        network_outputs = []
        output_start_id = 0

        for i, _ in enumerate(self.networks):
            # get output shapes for this network
            _, output_shapes = self.get_network_output_indices(i)
            assert isinstance(output_shapes, list)

            # process each output
            outputs = []
            for output_shape in output_shapes:
                nout = np.prod(output_shape)
                output = yhat[:, output_start_id : output_start_id + nout].reshape(
                    -1, *output_shape
                )
                outputs.append(output)
                output_start_id += nout

            if not all(output.shape == outputs[0].shape for output in outputs):
                raise ValueError(
                    f"Outputs have different shapes: {[output.shape for output in outputs]}"
                )

            if not len(outputs):
                continue

            network_output = (jnp if "jax" in type(outputs[0]).__module__ else np).concatenate(
                outputs, axis=1
            )
            network_outputs.append(network_output)

        return network_outputs

    ##────────────────────────────────────────────────────────────────────────────}}}

    ### {{{                       --    internal utils     --

    def add_layer(self, layer: StackLayer):
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
        """Validate the stack structure"""
        for l in self.layers:
            l.check()
            # verify all node keys reference valid networks and nodes
            for key in l.nodes:
                assert key.network_id < len(self.networks), f"Invalid network_id: {key.network_id}"
                assert key.node_id in self.networks[key.network_id].compute_graph.nodes, (
                    f"Invalid node_id {key.node_id} in network {key.network_id}"
                )
        # verify first layer is input
        first_key = self.layers[0].nodes[0]
        first_node = self.networks[first_key.network_id].compute_graph.nodes[first_key.node_id]
        assert first_node.node_type == "input", f"First node is not input: {first_node.node_type}"

    def get_all_node_keys(self) -> list[StackNode]:
        """Get all NodeKeys from all layers"""
        if self.layers is None:
            return []
        return [key for layer in self.layers for key in layer.nodes]

    def get_node_input_start_index(self, node_key: StackNode, input_slot: int) -> int:
        """Returns the start index of the input #input_slot for the given node
        in the flattened full-stack output array"""
        assert self.node_map is not None, "No node map"
        if self.layers is None:
            raise ValueError("No layers")

        # get incoming edges for this node
        graph = self.networks[node_key.network_id].compute_graph
        incoming_edges = sorted(
            graph.get_incoming_edges(node_key.node_id), key=lambda e: e.input_slot
        )

        input_edge = incoming_edges[input_slot]
        input_compute_node_id = input_edge.source_id
        input_compute_node_outslot = input_edge.output_slot

        input_layer_id, input_node_layer_loc = self.node_map[
            (node_key.network_id, input_compute_node_id)
        ]
        this_node_layer_id, _ = self.node_map[(node_key.network_id, node_key.node_id)]

        assert input_layer_id < this_node_layer_id, "input layer must be before this layer"

        this_layer = self.layers[this_node_layer_id]
        input_layer = self.layers[input_layer_id]

        this_input_shapes = this_layer.f_input_shapes
        input_layer_start = self.layers_start_index[input_layer_id]

        assert (
            this_input_shapes[input_slot] == input_layer.f_out_shapes[input_compute_node_outslot]
        ), (
            f"Shapes don't match: {this_input_shapes[input_slot]} != {input_layer.f_out_shapes[input_compute_node_outslot]}"
        )

        flat_out_size = int(np.sum([np.prod(s) for s in input_layer.f_out_shapes]))
        node_start = input_layer_start + input_node_layer_loc * flat_out_size

        flat_output_shape_till_input = np.sum(
            [np.prod(s) for s in input_layer.f_out_shapes[:input_compute_node_outslot]]
        )
        outslot_start = node_start + flat_output_shape_till_input

        return int(outslot_start)

    def get_node_output_start_index(self, node_key: StackNode, output_slot: int) -> int:
        """Returns the start index of the output #output_slot for the given node
        in the flattened full-stack output array"""
        assert self.node_map is not None, "No node map"

        this_node_layer_id, this_node_pos = self.node_map[(node_key.network_id, node_key.node_id)]
        this_layer = self.layers[this_node_layer_id]
        assert len(this_layer.f_out_shapes) > output_slot, "Output slot out of range"

        this_layer_start = self.layers_start_index[this_node_layer_id]
        flat_out_size = int(np.sum([np.prod(s) for s in this_layer.f_out_shapes]))

        node_start = this_layer_start + this_node_pos * flat_out_size
        out_shape_till_output = np.sum([np.prod(s) for s in this_layer.f_out_shapes[:output_slot]])

        return int(node_start + out_shape_till_output)

    ##────────────────────────────────────────────────────────────────────────────}}}

    ### {{{                         --     building     --{{{

    def _assemble_stack(self, **kwargs):
        from . import stack_builder

        self.layers = stack_builder.build_layers(self.networks, self, **kwargs)
        logger.debug(f"Final stack size: {len(self.layers) if self.layers else 0}")

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
            # we need to get the node_id from the node_map
            input_start_indices = np.array(
                [
                    [
                        self.node_map[(key.network_id, key.node_id)][0]
                        * input_lengths[0]  # layer_id * length
                        for key in layer.nodes
                    ]
                ]
            )
        else:
            input_start_indices = np.array(
                [
                    [self.get_node_input_start_index(key, i) for key in layer.nodes]
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
                # update StackNode objects with layer_id
                updated_nodes = []
                for n_id, key in enumerate(l.nodes):
                    # create new StackNode with layer_id filled in
                    updated_key = StackNode(
                        network_id=key.network_id,
                        node_id=key.node_id,
                        layer_number=l_id,
                        node_position_in_layer=n_id,
                    )
                    updated_nodes.append(updated_key)
                    self.node_map[(key.network_id, key.node_id)] = (l_id, n_id)
                    node_id += 1
                l.nodes = updated_nodes
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
        get_grads_for: list[str] = [
            "translation",
            "transcription",
            "output",
            "source_new",
            "source",
        ],
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
        ) -> tuple[NdArray, NdArray]:
            """
            The core of the apply method. Jittable. Applies the entire stack.
            Overwrite stuff:
                - added overwrite_* to allow injecting values at specific indices
                - allows feeding whatever values we want to some specific nodes
                - overwrite_values is a 1d array of values to inject
                - overwrite_at is a 1d array of indices where to inject the values
            """

            if len(inputs) != self.total_nb_of_inputs:
                raise ValueError(
                    f"When applying the stack, received inputs of shape {inputs.shape} "
                    f"but expected a total of {self.total_nb_of_inputs} inputs."
                )
            assert self.layers is not None, "No layers"
            assert self.layers_start_index is not None, "No layers start index"
            assert self.node_map is not None, "No node map"

            running_output = inputs.reshape(-1)
            stack_aux = {}
            stack_grad_wrt_inputs = jnp.array([])

            for lid in range(1, len(self.layers)):  # skip the input layer
                assert running_output.shape[0] == self.layers_start_index[lid]

                n_nodes = len(self.layers[lid].nodes)
                n_inputs = len(self.layers[lid].f_input_shapes)

                if overwrite_values is not None and overwrite_at is not None:
                    assert overwrite_values.ndim == 1 and overwrite_at.ndim == 1
                    assert overwrite_values.shape[0] == overwrite_at.shape[0], (
                        f"{overwrite_values.shape[0]=} != {overwrite_at.shape[0]=}"
                    )
                    running_output = running_output.at[overwrite_at].set(overwrite_values)

                # fetch the inputs for each node in the layer from the output array
                # (which was filled by the previous layers)
                layer_inputs = [input_getters_f[lid][i](running_output) for i in range(n_inputs)]

                assert len(layer_inputs) == n_inputs

                keys = jax.random.split(key, n_nodes)

                apply_f = self.layers[lid].f_apply

                def single_out_apply(*node_args, **node_kwargs):
                    out, _aux = apply_f(*node_args, **node_kwargs)
                    return out  # drop the aux

                def node_apply(node_id: ArrayLike, key: PRNGKey, *inputs: ArrayLike):
                    res, node_aux = apply_f(
                        *inputs, params=params, quantiles=quantiles, node_id=node_id, key=key
                    )
                    if w_grads[lid]:
                        # using jax.jacfwd, we can just grab the first output wrt the first input
                        grad = jax.jacfwd(single_out_apply, argnums=list(range(n_inputs)))(
                            *inputs, params=params, quantiles=quantiles, node_id=node_id, key=key
                        )
                        first_output_grad = grad[0]
                        grad = first_output_grad.reshape(-1)

                    else:
                        grad = jnp.array([])

                    node_aux["grads_wrt_inputs"] = grad  # store the gradient in the aux dict

                    return res, node_aux

                def layer_apply(*inputs):
                    return vmap(node_apply)(jnp.arange(n_nodes), keys, *inputs)

                layer_out, layer_aux = layer_apply(*layer_inputs)
                layer_grad_wrt_inputs = layer_aux.get("grads_wrt_inputs", jnp.array([])).ravel()
                stack_grad_wrt_inputs = jnp.concatenate(
                    [stack_grad_wrt_inputs, layer_grad_wrt_inputs]
                )

                stack_aux[f"{lid}"] = {
                    "layer_aux": layer_aux,
                    "trace": {
                        "inputs": layer_inputs,
                        "outputs": layer_out,
                    },
                }

                flattened_layer_output = layer_out.ravel()

                if self.layers[lid].flattened_output_shape() != len(flattened_layer_output):
                    raise ValueError(
                        f"Output shape mismatch: {self.layers[lid].flattened_output_shape()=} != "
                        f"{len(flattened_layer_output)=} for layer {lid} ({self.layers[lid].f_type})"
                        f" with {n_nodes} nodes. {inputs.shape=}"
                    )

                running_output = jnp.concatenate([running_output, flattened_layer_output])

            stack_aux["grads_wrt_inputs"] = stack_grad_wrt_inputs.ravel()

            return running_output, stack_aux

        def apply(*args, **kwargs):
            # returns only the final outputs (of the last layer) + the gradients + the full trace
            o, aux = apply_impl(*args, **kwargs)
            return o[output_indices], (aux, o)

        self.apply = apply
        self.output_indices = output_indices


##────────────────────────────────────────────────────────────────────────────}}}
