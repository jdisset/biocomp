### {{{                          --     imports     --
from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar, Callable, Optional, Union, Any, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax.tree_util import Partial as partial
from jax.typing import ArrayLike

from .network import Network
from . import utils as ut
from biocomp.utils import ArbitraryModel, EncodedPartialFunction
from .parameters import ParameterTree
from .graphengine import GraphState

from biocomp.logging_config import get_logger
from biocomp.graphengine import GraphNode, GraphEdge, is_inverse_node_type
from biocomp.tracing import (
    is_design_debug_enabled,
    save_debug_state,
    trace_scope,
    summarize_stack,
    summarize_params,
    should_save_full_objects,
    snapshot_full_stack,
)
import dracon as dr


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
        """Generate a type signature for a node based on its type and number of inputs/outputs.

        Uses get_max_output_slot() to determine required outputs, not unique slot count.
        This handles sparse slots correctly (e.g., slots 0,7 requires 8 outputs, not 2).
        """
        node = graph.nodes[node_id]
        n_inputs = len(graph.get_incoming_edges(node_id))
        n_outputs = graph.get_max_output_slot(node_id)
        return f"{node.node_type}_{n_inputs}_{n_outputs}"

    def _get_compute_graph(self, stack: "ComputeStack") -> GraphState:
        assert stack.networks is not None, "Stack has no networks"
        assert self.network_id < len(stack.networks), (
            f"network_id {self.network_id} >= {len(stack.networks)}"
        )
        cg = stack.networks[self.network_id].compute_graph
        assert cg is not None, f"compute_graph is None for network {self.network_id}"
        return cg

    def get(self, stack: "ComputeStack") -> GraphNode:
        n = self._get_compute_graph(stack).get_node(self.node_id)
        assert n is not None, f"Node {self.node_id} not found in network {self.network_id}"
        return n

    def get_forward_stacknode(self, stack: "ComputeStack") -> "StackNode | None":
        node = self.get(stack)
        if node.is_inverse_of is None:
            return None
        return stack.get_node_from_net_and_compute_id(
            self.network_id, node.is_inverse_of.node_id, allow_missing=True
        )

    def get_outgoing_edges(self, stack: "ComputeStack") -> list[GraphEdge]:
        return self._get_compute_graph(stack).get_outgoing_edges(self.node_id)

    def get_incoming_edges(self, stack: "ComputeStack") -> list[GraphEdge]:
        return self._get_compute_graph(stack).get_incoming_edges(self.node_id)

    def get_nb_outputs(self, stack: "ComputeStack") -> int:
        """Get required number of outputs based on max slot index, not unique slot count."""
        return self._get_compute_graph(stack).get_max_output_slot(self.node_id)


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

    OUTPUT_WEIGHT_PATH: ClassVar[str] = "shared/NN/grouped_output/l0/w"

    def get_node_implementation(self, node_name: str, module_name: str = "biocomp.nodes"):
        if self.node_functions is None:
            raise ValueError("No node implementations in this config")
        if node_name not in self.node_functions:
            raise ValueError(f"No node implementation for {node_name}")

        return self.node_functions[node_name].get_impl(extra_module_names=[module_name])

    def backfill_from_defaults(self) -> None:
        """Fill missing node functions from DEFAULT_COMPUTE_CONFIG (for older pickled models)."""
        if self.node_functions is None or DEFAULT_COMPUTE_CONFIG.node_functions is None:
            return
        for name, impl in DEFAULT_COMPUTE_CONFIG.node_functions.items():
            if name not in self.node_functions:
                self.node_functions[name] = impl
                logger.info(f"Backfilled missing node function: {name}")

    def detect_output_compat(self, shared_params: "ParameterTree") -> None:
        """Detect whether the model was trained with random_var in output MLP."""
        if self.OUTPUT_WEIGHT_PATH not in shared_params:
            return
        w_shape = shared_params[self.OUTPUT_WEIGHT_PATH].shape
        if self.extra is None:
            self.extra = {}
        self.extra["output_has_random_var"] = w_shape[0] != 1
        if w_shape[0] == 1:
            logger.info("Model output MLP trained without random_var (backward compat)")

    def apply_transform_out_dim(self) -> int:
        """Apply SSOT `transform_out_dim` to forward transform nodes only."""
        if self.node_functions is None:
            raise ValueError("No node implementations in this config")

        if self.extra is None:
            self.extra = {}

        raw_out_dim = self.extra.get("transform_out_dim", 1)
        if isinstance(raw_out_dim, bool):
            raise ValueError(f"transform_out_dim must be an int >= 1, got {raw_out_dim!r}")
        try:
            out_dim = int(raw_out_dim)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"transform_out_dim must be an int >= 1, got {raw_out_dim!r}") from exc
        if out_dim < 1:
            raise ValueError(f"transform_out_dim must be an int >= 1, got {out_dim}")

        self.extra["transform_out_dim"] = out_dim

        for node_name in ("transcription", "translation", "sequestron_ERN"):
            if node_name in self.node_functions:
                self.node_functions[node_name].set_missing_kwargs({"out_dim": out_dim})

        return out_dim


##────────────────────────────────────────────────────────────────────────────}}}

##────────────────────────────────────────────────────────────────────────────}}}

## {{{                       --     Compute Layer     --
NodeInput = tuple[int, int, int]  #  (net_id, compute_node_id, slot_id)


def _compute_layer_namespace(layer_id: int, node_type: str, n_outputs: int) -> str:
    """Computes the canonical parameter namespace for a layer based on its properties."""
    type_suffix = ""
    if node_type in ["aggregation", "source"]:
        type_suffix = f"{n_outputs}x"
    elif is_inverse_node_type(node_type):
        type_suffix = ""

    layer_name = f"{node_type}{type_suffix}"
    return f"local/{layer_id}/{layer_name}"


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
    f_introspect: Optional[Callable] = None

    # parameter namespace for this layer (e.g., "local/5/aggregation_2x")
    namespace: Optional[str] = None

    is_built: bool = False

    def setup(self, config: ComputeConfig, stack: "ComputeStack"):
        self.check()

        first_key = self.nodes[0]
        first_graph = stack.networks[first_key.network_id].compute_graph
        self.f_type = first_graph.nodes[first_key.node_id].node_type

        if self.f_type == "input":
            self.f_out_shapes = [(1,)]
            self.f_input_shapes = [(1,)]
            self.namespace = f"local/{self.layer_id}/input"  # input layer namespace
            self.is_built = True
            return

        # get the shapes of the inputs. We'll collect all the inputs for each node
        # to make sure they are all the same
        node_inputs: list[list[NodeInput]] = []
        for key in self.nodes:
            graph = stack.networks[key.network_id].compute_graph
            incoming_edges = graph.get_incoming_edges(key.node_id)
            # sort by input_slot to match old behavior
            incoming_edges_sorted = sorted(incoming_edges, key=lambda e: e.to_input_slot)
            ninp = [(edge.source_id, edge.from_output_slot) for edge in incoming_edges_sorted]
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
                    f"Input slot {input_slot_id} is out of range for layer {input_layer_id} "
                    f"with {len(input_layer_output_shapes)} outputs. "
                    f"Source node {input_compute_node_id} in network {input_net_id} "
                    f"({stack.networks[input_net_id].name}). "
                    f"Edge from source node to this layer's node."
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

        # Compute namespace for this layer
        self.namespace = _compute_layer_namespace(self.layer_id, self.f_type, n_outputs)

        impl = config.get_node_implementation(self.f_type)(
            input_shapes=self.f_input_shapes,
            n_outputs=n_outputs,
            stack=stack,
            namespace=self.namespace,
        )

        self.f_prepare = impl.prepare
        self.f_apply = impl.apply
        self.f_out_shapes = impl.output_shapes
        self.f_commit = impl.commit
        self.f_introspect = impl.introspect
        self.is_built = True

    def get_n_outputs(self) -> int:
        """Get the number of outputs for nodes in this layer (all nodes in a layer have the same type signature).

        Uses get_max_output_slot() to determine required outputs, matching signature generation.
        This handles sparse slots correctly (e.g., slots 0,7 requires 8 outputs).
        """
        first_key = self.nodes[0]
        graph = self.stack.networks[first_key.network_id].compute_graph
        return graph.get_max_output_slot(first_key.node_id)

    def flattened_output_shape(self) -> int:
        return int(len(self.nodes) * np.sum([np.prod(s) for s in self.f_out_shapes]))

    def type_str(self) -> str:
        if self.stack is None:
            return "<unbound>"
        first_key = self.nodes[0]
        return (
            self.stack.networks[first_key.network_id]
            .compute_graph.nodes[first_key.node_id]
            .node_type
        )

    def __repr__(self):
        ftype = self.type_str()
        layer_id = self.layer_id if self.layer_id is not None else "?"
        return f"Layer {layer_id} ({ftype}) with {len(self.nodes)} nodes"

    def __hash__(self):
        return hash(
            (
                tuple(self.nodes),
                self.f_type,
                self.namespace,
                self.f_input_shapes,
                tuple(self.f_out_shapes) if self.f_out_shapes else None,
            )
        )

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

    # TU masking support (for design mode)
    tu_id_to_idx: Optional[dict[str, int]] = None
    n_tus: int = 0
    inverse_tu_ids: Optional[set[str]] = None  # TUs feeding inverse nodes (never disabled)
    no_masking_tu_ids: Optional[set[str]] = None  # TUs with no_masking=True in recipe

    def get_node(self, network_id: int, node_id: int):
        """Look up a node from compute_graph by network_id and node_id."""
        assert 0 <= network_id < len(self.networks), f"network_id {network_id} out of range"
        graph = self.networks[network_id].compute_graph
        return graph.nodes.get(node_id)

    ### {{{                     --     public interface     --

    def build(
        self,
        config: ComputeConfig,
        enable_tu_masking: bool = False,
        auto_lock_topology_tus: bool = True,
        **kwargs,
    ):
        """
        Split apart all the networks into their constituent nodes
        and put these nodes into ordered layers, maximizing parallelism by ensuring same-type nodes
        (potentially from different networks) are in the same layer.
        Then generates the apply method for the stack, which will handle the parallel execution
        of all nodes in each layers, as well as the correct mapping and chaining of
        their inputs and outputs.

        Args:
            config: ComputeConfig with node implementations
            enable_tu_masking: If True, build TU-to-index mapping for design mode TU masking
            auto_lock_topology_tus: If True, auto-detect TUs whose masking would change topology
        """
        with trace_scope("stack_build", component="stack") as scope:
            scope.event(
                "build_start",
                "Beginning stack build",
                {
                    "n_networks": len(self.networks),
                    "network_names": [n.name for n in self.networks],
                    "enable_tu_masking": enable_tu_masking,
                    "auto_lock_topology_tus": auto_lock_topology_tus,
                },
            )

            with ut.timer("Building compute stack", logger):
                self.config = config.model_copy(deep=True)
                transform_out_dim = self.config.apply_transform_out_dim()
                scope.event(
                    "transform_out_dim",
                    "Applied transform output dimension",
                    {
                        "transform_out_dim": transform_out_dim,
                    },
                )

                if enable_tu_masking:
                    self._build_tu_mapping(auto_lock_topology_tus=auto_lock_topology_tus)
                    scope.event(
                        "tu_mapping_complete",
                        "TU mapping built",
                        {
                            "n_tus": self.n_tus,
                            "n_inverse_tus": len(self.inverse_tu_ids) if self.inverse_tu_ids else 0,
                            "n_no_masking_tus": len(self.no_masking_tu_ids)
                            if self.no_masking_tu_ids
                            else 0,
                        },
                    )
                    if self.tu_id_to_idx:
                        scope.snapshot("tu_id_to_idx", dict(self.tu_id_to_idx))

                self._assemble_stack(**kwargs)
                scope.event(
                    "assemble_complete",
                    "Stack assembled",
                    {
                        "n_layers": len(self.layers) if self.layers else 0,
                    },
                )

                self._refresh()
                assert self.layers is not None, "No layers"

                for layer_idx, layer in enumerate(self.layers):
                    layer.setup(self.config, stack=self)
                    scope.event(
                        "layer_setup",
                        f"Layer {layer_idx} setup complete",
                        {
                            "layer_id": layer_idx,
                            "layer_type": layer.f_type,
                            "n_nodes": len(layer.nodes),
                            "namespace": layer.namespace,
                        },
                    )
                    self._refresh()

                self._generate_apply_method()
                self.check()
                self.is_built = True

            scope.event(
                "build_complete",
                "Stack build complete",
                {
                    "n_layers": len(self.layers),
                    "total_nodes": self.number_of_nodes,
                    "output_shape": self.output_shape,
                },
            )
            scope.snapshot("stack_summary", summarize_stack(self))
            if should_save_full_objects():
                scope.snapshot("stack_full", snapshot_full_stack(self))

    def _build_tu_mapping(self, auto_lock_topology_tus: bool = True):
        """Build TU ID to index mapping for all networks."""
        from biocomp.tumasking import build_tu_id_mapping_excluding_inverse

        with trace_scope("build_tu_mapping", component="stack") as scope:
            sorted_tu_ids, tu_id_to_idx, inverse_tu_ids, no_masking_tu_ids = (
                build_tu_id_mapping_excluding_inverse(self.networks)
            )

            scope.event(
                "initial_mapping",
                "Initial TU mapping from networks",
                {
                    "n_sorted_tu_ids": len(sorted_tu_ids),
                    "n_inverse_tu_ids": len(inverse_tu_ids),
                    "n_no_masking_tu_ids": len(no_masking_tu_ids),
                },
            )

            if auto_lock_topology_tus:
                from biocomp.network import find_topology_changing_tus

                for net_idx, net in enumerate(self.networks):
                    topology_tus = find_topology_changing_tus(net)
                    if topology_tus:
                        scope.event(
                            "topology_tus_found",
                            "Topology-changing TUs identified",
                            {
                                "network_idx": net_idx,
                                "network_name": net.name,
                                "topology_tu_ids": list(topology_tus),
                            },
                        )
                    no_masking_tu_ids.update(topology_tus)

            # inverse TUs must never be disabled - they're essential for inversion path
            no_masking_tu_ids.update(inverse_tu_ids)

            self.tu_id_to_idx = tu_id_to_idx
            self.n_tus = len(sorted_tu_ids)
            self.inverse_tu_ids = inverse_tu_ids
            self.no_masking_tu_ids = no_masking_tu_ids

            scope.event(
                "tu_mapping_finalized",
                "TU mapping finalized",
                {
                    "n_tus": self.n_tus,
                    "n_inverse_tus": len(inverse_tu_ids),
                    "n_no_masking_tus": len(no_masking_tu_ids),
                    "inverse_tu_ids": list(inverse_tu_ids),
                    "no_masking_tu_ids": list(no_masking_tu_ids),
                },
            )
            scope.snapshot(
                "tu_mapping",
                {
                    "tu_id_to_idx": dict(tu_id_to_idx),
                    "inverse_tu_ids": list(inverse_tu_ids),
                    "no_masking_tu_ids": list(no_masking_tu_ids),
                },
            )

        logger.debug(
            f"Built TU mapping: {self.n_tus} TUs, {len(inverse_tu_ids)} inverse, {len(no_masking_tu_ids)} no_masking"
        )

    def ensure_tu_mapping(self, auto_lock_topology_tus: bool = True) -> dict[str, int]:
        """Ensure TU ID mapping exists, building it if missing."""
        if getattr(self, "tu_id_to_idx", None) is None:
            self._build_tu_mapping(auto_lock_topology_tus=auto_lock_topology_tus)
        assert self.tu_id_to_idx is not None, "TU mapping not built"
        return self.tu_id_to_idx

    def get_per_network_tu_mask(self) -> jnp.ndarray:
        """Returns (n_networks, n_tus) mask: 1 if network uses TU, 0 otherwise.

        This enables per-network L0 penalty: each network is only penalized for
        TUs it actually uses, not for TUs used by other networks.
        """
        from biocomp.tumasking import extract_tu_ids_from_network

        assert self.is_built, "Stack must be built before getting TU masks"
        assert hasattr(self, "tu_id_to_idx"), "TU mapping not built"
        assert self.n_tus > 0, "No TUs in stack"

        mask = np.zeros((len(self.networks), self.n_tus), dtype=np.float32)
        for net_idx, net in enumerate(self.networks):
            net_tu_ids = extract_tu_ids_from_network(net)
            for tu_id in net_tu_ids:
                if tu_id in self.tu_id_to_idx:
                    tu_idx = self.tu_id_to_idx[tu_id]
                    mask[net_idx, tu_idx] = 1.0
        return jnp.array(mask)

    def init(self, rng_key: PRNGKey) -> ParameterTree:
        """
        Generates a randomly initilized parameter tree for the stack
        """
        with trace_scope("stack_init", component="stack") as scope:
            assert self.is_built, "Stack not built"
            params = ParameterTree()
            # for l_id, layer in enumerate(self.layers):
            # let's initialize the stacj it in reverse order
            # so that the inverse nodes can reference fwd ones after init
            assert self.layers is not None, "No layers"
            assert self.layers_start_index is not None, "No layers start index"

            scope.event(
                "init_start",
                "Beginning parameter initialization",
                {
                    "n_layers": len(self.layers),
                    "init_order": "reverse",  # inverse nodes can reference fwd ones
                },
            )

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
                        scope.event(
                            "layer_init",
                            f"Layer {l_id} initialized",
                            {
                                "layer_id": l_id,
                                "layer_type": layer.f_type,
                                "n_nodes": len(layer.nodes),
                                "namespace": layer.namespace,
                            },
                        )
                    except Exception as e:
                        scope.event(
                            "layer_init_error",
                            f"Layer {l_id} init failed",
                            {
                                "layer_id": l_id,
                                "layer_type": layer.f_type,
                                "error": str(e),
                            },
                        )
                        logger.error(f"Error in layer {l_id} preparation:")
                        logger.error(f"Layer type: {layer.f_type}")
                        logger.error(f"Layer input shapes: {layer.f_input_shapes}")
                        logger.error(f"Layer output shapes: {layer.f_out_shapes}")
                        logger.exception(e)
                        raise e

            # Ensure every non-input layer has node_key_ids.
            # Layers whose prepare() already allocated them are skipped (overwrite=None).
            from biocomp.nodeutils import add_node_key_ids as _add_nk

            for layer in self.layers:
                if layer.f_type == "input":
                    continue
                ns = layer.namespace
                if f"{ns}/node_key_id" not in params:
                    _add_nk(params, len(layer.nodes), ns)

            params.tag("local", "local")
            params.tag("shared", "shared")

            from biocomp.nodeutils import NON_GRAD_TAG

            params.at(
                "global/dependent_output_mask",
                self.get_dependent_output_mask(),
                tags=[NON_GRAD_TAG],
            )

            scope.event(
                "init_complete",
                "Parameter initialization complete",
                {
                    "n_param_paths": len(list(params.data.iter_leaves())),
                    "tagnames": params.tagnames,
                },
            )
            scope.snapshot("params_summary", summarize_params(params))
            # Note: Full params snapshot moved to caller (e.g., initialize_params)
            # where values are concrete (outside vmap context)

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

    def get_layer_namespace(self, layer_id: int) -> str:
        """
        Returns the canonical parameter namespace for a given layer.
        This is the single source of truth for local parameter paths.
        """
        assert self.layers is not None, "Stack layers are not built"
        assert 0 <= layer_id < len(self.layers), f"Invalid layer_id: {layer_id}"
        layer = self.layers[layer_id]
        assert layer.namespace is not None, f"Layer {layer_id} has no namespace"
        return layer.namespace

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

    def get_node_from_net_and_compute_id(
        self, network_id: int, compute_node_id: int, allow_missing: bool = False
    ) -> StackNode | None:
        """Returns the NodeKey corresponding to the given network and compute node ids"""
        assert self.node_map is not None, "No node map"
        assert self.layers is not None, "Stack has no layers"
        key = (network_id, compute_node_id)
        if key not in self.node_map:
            if allow_missing:
                return None
            raise AssertionError(f"Node not found: {network_id}/{compute_node_id}")
        layer_id, node_loc = self.node_map[key]
        return self.layers[layer_id].nodes[node_loc]

    def copy(self):
        # we only deepcopy the layers, not the networks
        return ComputeStack(self.networks, deepcopy(self.layers))

    def commit(self, params: ParameterTree, collapse_to_part=True, **kwargs):
        """Commit trained parameters to networks, optionally collapsing embeddings.

        This method delegates to commit_networks() in stack_commit.py for the actual
        implementation. It handles:
        1. Deep copying networks
        2. Running node-level commits
        3. Pruning disabled TUs (via log_alpha or binary_mask)
        4. Rebuilding networks via recipe roundtrip for clean structure

        Args:
            params: Trained parameter tree
            collapse_to_part: If True, collapse embeddings to discrete parts
            **kwargs: Additional arguments passed to layer commits

        Returns:
            List of committed networks
        """
        from .stack_commit import commit_networks, CommitOptions

        # Handle backward compatibility: lock_ratios=True means preserve_ratio_states=False
        lock_ratios = kwargs.get("lock_ratios", True)
        options = CommitOptions(
            prune_tus=True,
            collapse_to_part=collapse_to_part,
            preserve_ratio_states=not lock_ratios,
            roundtrip_rebuild=True,
            preserve_input_order=True,
            max_rebuild_workers=8,
        )

        final_networks, _ = commit_networks(
            self.networks,
            self.layers,
            params,
            options,
            tu_id_to_idx=getattr(self, "tu_id_to_idx", None),
            node_map=self.node_map,
        )

        for i, net in enumerate(final_networks):
            self._assert_valid_edge_slots(net, f"committed network[{i}]")

        return final_networks

    def _assert_valid_edge_slots(self, network, context: str):
        """Assert all edges reference valid slot indices on their source/target nodes."""
        cg = network.compute_graph
        for eid, edge in cg.edges.items():
            src = cg.nodes.get(edge.source_id)
            tgt = cg.nodes.get(edge.target_id)

            assert src is not None, (
                f"{context}: Edge {eid} references missing source node {edge.source_id}"
            )
            assert tgt is not None, (
                f"{context}: Edge {eid} references missing target node {edge.target_id}"
            )

            # aggregation outputs are determined by ratio_schema slot count
            if src.node_type == "aggregation":
                from biocomp.ratio_schema import slot_count

                n_outputs = slot_count(src.extra)
                assert edge.from_output_slot < n_outputs, (
                    f"{context}: Edge {eid} from aggregation[{edge.source_id}] has "
                    f"from_output_slot={edge.from_output_slot} but aggregation only has "
                    f"{n_outputs} outputs (ratio_schema)"
                )

    def __repr__(self):
        # layers with line breaks
        if self.layers is None:
            return "Empty stack"
        return "\n".join([l.__repr__() for l in self.layers])

    def __hash__(self):
        parts: list = [tuple(self.networks)]
        if self.layers is not None:
            parts.append(tuple(self.layers))
        if self.config is not None:
            import json

            parts.append(
                hash(
                    json.dumps(
                        self.config.model_dump(mode="json"),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
            )
        parts.append(self.n_tus)
        if self.tu_id_to_idx:
            parts.append(tuple(sorted(self.tu_id_to_idx.items())))
        return hash(tuple(parts))

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
        first_key = self.layers[0].nodes[0]
        first_node = self.networks[first_key.network_id].compute_graph.nodes[first_key.node_id]
        # "numeric" is valid for networks without inputs (e.g., after TU pruning removes all input nodes)
        assert first_node.node_type in ("input", "bias", "numeric"), (
            f"First node must be input, bias, or numeric: {first_node.node_type}"
        )

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
            graph.get_incoming_edges(node_key.node_id), key=lambda e: e.to_input_slot
        )

        input_edge = incoming_edges[input_slot]
        input_compute_node_id = input_edge.source_id
        input_compute_node_outslot = input_edge.from_output_slot

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

    ### {{{                         --     building     --

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
            # input indices of the input layer are determined by node position
            # node_map returns (layer_id, node_pos)
            input_start_indices = np.array(
                [
                    [
                        self.node_map[(key.network_id, key.node_id)][1]
                        * input_lengths[0]  # node_pos * length
                        for key in layer.nodes
                    ]
                ]
            )
            flat_indices = input_start_indices.flatten()
            n_unique = len(np.unique(flat_indices))
            assert n_unique == len(flat_indices), (
                f"Input layer has duplicate indices! indices={flat_indices}, "
                f"expected {len(flat_indices)} unique values but got {n_unique}. "
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

            # Use dynamic slicing which works with JAX tracing
            def get_inputs_dyn(all_outputs):
                def dyn_slice(start):
                    return jax.lax.dynamic_slice(all_outputs, (start,), (length,)).reshape(shape)

                return vmap(dyn_slice)(starts)

            return get_inputs_dyn

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

        # Debug: dump input node order for axis alignment analysis
        if is_design_debug_enabled() and allbuilt and self.networks:
            self._dump_input_node_order()

    def _dump_input_node_order(self):
        """Dump input node ordering for each network to help debug axis alignment issues.

        During design optimization, X columns are passed positionally to network inputs.
        This dump records which stack position each network's input nodes occupy,
        which determines the X column -> network input mapping.
        """
        # Lazy import to avoid circular dependency (design imports compute)
        from biocomp.design import get_design_debug_output_dir

        for net_idx, network in enumerate(self.networks):
            # Find all input-type nodes in this network
            input_nodes_info = []
            for (n_idx, node_id), (layer_id, node_pos) in self.node_map.items():
                if n_idx != net_idx:
                    continue
                node = network.compute_graph.nodes.get(node_id)
                if node is None:
                    continue
                if node.node_type in ("input", "bias"):
                    protein_name = node.extra.get("protein_name", node.extra.get("name", "unknown"))
                    input_nodes_info.append(
                        {
                            "node_id": node_id,
                            "node_type": node.node_type,
                            "layer_id": layer_id,
                            "node_pos_in_layer": node_pos,
                            "protein_name": protein_name,
                        }
                    )

            # Sort by layer position to get the actual input order
            input_nodes_info.sort(key=lambda x: (x["layer_id"], x["node_pos_in_layer"]))

            save_debug_state(
                "ComputeStack_input_order",
                {"input_nodes": input_nodes_info},
                {
                    "network_idx": net_idx,
                    "network_name": getattr(network, "name", f"network_{net_idx}"),
                    "n_input_nodes": len(input_nodes_info),
                    "input_proteins_in_order": [n["protein_name"] for n in input_nodes_info],
                },
                output_dir=get_design_debug_output_dir(),
                mode="design",
            )

    def _generate_apply_method(
        self,
        get_grads_for: list[str] = None,
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

        if get_grads_for is None:
            get_grads_for = []
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

        # Pre-build namespace list for fold_in key lookup
        layer_namespaces = [l.namespace for l in self.layers]

        # Pre-build network_ids arrays for each layer (node_position -> network_id)
        # Used for per-network TU masking
        n_networks_total = len(self.networks)
        layer_network_ids = []
        for l in self.layers:
            network_ids = jnp.array([n.network_id for n in l.nodes], dtype=jnp.int32)
            # Verify all network_ids are valid
            for nid in network_ids:
                assert 0 <= nid < n_networks_total, (
                    f"Invalid network_id {nid} in layer {l.layer_id}, expected 0..{n_networks_total - 1}"
                )
            layer_network_ids.append(network_ids)

        def apply_impl(
            params: ParameterTree,
            inputs: NdArray,
            random_vars: NdArray,
            key: PRNGKey,
            overwrite_values: Optional[NdArray] = None,
            overwrite_at: Optional[NdArray] = None,
            tu_enabled_random_vars: Optional[NdArray] = None,
        ) -> tuple[NdArray, NdArray]:
            """
            The core of the apply method. Jittable. Applies the entire stack.
            Overwrite stuff:
                - added overwrite_* to allow injecting values at specific indices
                - allows feeding whatever values we want to some specific nodes
                - overwrite_values is a 1d array of values to inject
                - overwrite_at is a 1d array of indices where to inject the values

            TU masking (for design mode):
                - tu_enabled_random_vars: uniform samples, shape (n_networks, n_tus)
                - During design: pass fresh uniform samples each forward pass
                - During inference: pass None (all enabled)
            """

            if len(inputs) != self.total_nb_of_inputs:
                raise ValueError(
                    f"When applying the stack, received inputs of shape {inputs.shape} "
                    f"but expected a total of {self.total_nb_of_inputs} inputs."
                )
            assert self.layers is not None, "No layers"
            assert self.layers_start_index is not None, "No layers start index"
            assert self.node_map is not None, "No node map"

            if tu_enabled_random_vars is not None:
                assert tu_enabled_random_vars.ndim == 2, (
                    f"tu_enabled_random_vars must be 2D (n_networks, n_tus), got {tu_enabled_random_vars.ndim}D. "
                    "Per-network TU masking is required."
                )
                assert tu_enabled_random_vars.shape[0] == len(self.networks), (
                    f"tu_enabled_random_vars.shape[0]={tu_enabled_random_vars.shape[0]} != n_networks={len(self.networks)}"
                )

            # Check if layer 0 is the input layer
            layer0_is_input = self.layers[0].f_type == "input"
            if layer0_is_input:
                # input layer: outputs are the inputs themselves
                running_output = inputs.reshape(-1)
                start_layer = 1
            else:
                # no input layer (e.g., bias-only network): start with empty output
                running_output = jnp.array([])
                start_layer = 0

            stack_aux = {}
            stack_grad_wrt_inputs = jnp.array([])

            # Derive a single base key for fold_in — each node derives its own
            # key deterministically via jax.random.fold_in(base_node_key, node_key_id).
            # Forward/inverse pairs share the same node_key_id (via ArrayRef),
            # so they get identical keys regardless of layer ordering.
            key, base_node_key = jax.random.split(key)

            for lid in range(start_layer, len(self.layers)):
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

                apply_f = self.layers[lid].f_apply
                net_ids = layer_network_ids[lid]  # shape (n_nodes,)

                def single_out_apply(*node_args, **node_kwargs):
                    out, _aux = apply_f(*node_args, **node_kwargs)
                    return out  # drop the aux

                def node_apply(node_id: ArrayLike, network_id: ArrayLike, *inputs: ArrayLike):
                    # Derive per-node key from globally unique node_key_id
                    nk_id = params[f"{layer_namespaces[lid]}/node_key_id"][node_id]
                    node_key = jax.random.fold_in(base_node_key, nk_id)

                    node_tu_uniform = (
                        tu_enabled_random_vars[network_id]
                        if tu_enabled_random_vars is not None
                        else None
                    )
                    res, node_aux = apply_f(
                        *inputs,
                        params=params,
                        random_vars=random_vars,
                        node_id=node_id,
                        key=node_key,
                        tu_enabled_random_vars=node_tu_uniform,
                        network_id=network_id,
                    )
                    if w_grads[lid]:
                        # using jax.jacfwd, we can just grab the first output wrt the first input
                        grad = jax.jacfwd(single_out_apply, argnums=list(range(n_inputs)))(
                            *inputs,
                            params=params,
                            random_vars=random_vars,
                            node_id=node_id,
                            key=node_key,
                            tu_enabled_random_vars=node_tu_uniform,
                            network_id=network_id,
                        )
                        first_output_grad = grad[0]
                        grad = first_output_grad.reshape(-1)

                    else:
                        grad = jnp.array([])

                    node_aux["grads_wrt_inputs"] = grad  # store the gradient in the aux dict

                    return res, node_aux

                def layer_apply(*inputs):
                    return vmap(node_apply)(jnp.arange(n_nodes), net_ids, *inputs)

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

## {{{                  --     Default Configuration     --

DEFAULT_COMPUTE_CONFIG = ComputeConfig.model_validate(
    dr.load("pkg:biocomp:config/default_compute_config.yaml")
)

##────────────────────────────────────────────────────────────────────────────}}}
