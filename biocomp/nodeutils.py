from .library import PartsLibrary as PartsLibrary
from dataclasses import dataclass
from typing import Callable, Optional
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from biocomp.parameters import ArrayRef, ParameterTree
from biocomp.compute import StackNode, ComputeStack

PRNGKey = ArrayLike
NDArray = np.ndarray | jnp.ndarray


ResultAndAux = tuple[NDArray, dict]
NodeID = int


@dataclass
class LayerInstance:
    prepare: Callable[[ParameterTree, list[StackNode], PRNGKey], None]
    apply: Callable[[NDArray, NDArray, ParameterTree, NodeID, NDArray], ResultAndAux]
    output_shapes: list[tuple[int]]
    commit: Optional[Callable[[ParameterTree, list[StackNode], ComputeStack], None]] = None
    introspect: Optional[
        Callable[[ParameterTree, list[StackNode], ComputeStack, int, bool], list]
    ] = None

    def __post_init__(self):
        assert all(isinstance(shape, tuple) for shape in self.output_shapes), (
            f"Invalid output shapes: {self.output_shapes}"
        )
        assert all(all(isinstance(dim, int) for dim in shape) for shape in self.output_shapes), (
            f"Non-integer dimensions in output shapes: {self.output_shapes}"
        )


NON_GRAD_TAG = "non_grad"

GLOBAL_PATH_NUMBER_OF_RANDOM_VARIABLES = "global/number_of_random_variables"


def get_prev_num_random_vars(params: ParameterTree):
    try:
        return params[GLOBAL_PATH_NUMBER_OF_RANDOM_VARIABLES]
    except KeyError:
        return 0


def add_random_var_ids(params: ParameterTree, num_nodes: int, num_per_node, namespace: str):
    """
    Adds random_var variable IDs to the parameters. The random_var variable is just a random variable
    used for generation (ideally the node learns a quantile function,
    and this is the random variable fed to that function).
    It updates (or creates) the following parameters:
        - global/number_of_random_variables -> int, total number of random_var variables (across all neural functions aka nodes)
        - local/{layer_name}/random_variable_id -> id array of shape (num_nodes, num_per_node)
    Then a node can access its random_var variable IDs by simply indexing the vector of random_var variables (Z) with these ids

    :param params: The parameters tree to update.
    :param num_nodes: The number of nodes for which to add random_var variable IDs.
    :param num_per_node: The number of random_var variables per node.
    :param layer_name: The name (possibly subpath) of the layer to which these random_var variables belong.

    """

    prev_num_random_vars = get_prev_num_random_vars(params)
    new_num_random_vars = prev_num_random_vars + num_nodes * num_per_node
    random_var_ids = jnp.arange(prev_num_random_vars, new_num_random_vars).reshape(
        (num_nodes, num_per_node)
    )
    params.at(
        f"{namespace}/random_variable_id",
        random_var_ids,
        tags=[NON_GRAD_TAG],
        overwrite=None,
    )

    params.at(
        GLOBAL_PATH_NUMBER_OF_RANDOM_VARIABLES,
        new_num_random_vars,
        tags=[NON_GRAD_TAG],
        overwrite=True,
    )


def reference_forward_random_var_ids(stack, params, nodelist, inv_namespace):
    # check if all forward nodes exist - skip if any are missing (pruned network)
    all_forward_exist = all(node.get_forward_stacknode(stack) is not None for node in nodelist)
    if not all_forward_exist:
        # pruned network with missing forward nodes - skip reference setup
        # the inv_* nodes will use default random variable handling
        return

    ref = ArrayRef(params.data)
    for node in nodelist:
        fwd_node = node.get_forward_stacknode(stack)
        fwd_namespace = stack.get_layer_namespace(fwd_node.layer_number)
        ref.push_back(f"{fwd_namespace}/random_variable_id", fwd_node.node_position_in_layer)

    params.at(f"{inv_namespace}/random_variable_id", ref, overwrite=None)


def add_tu_input_mapping(
    params: ParameterTree,
    stack: ComputeStack,
    nodelist: list[StackNode],
    namespace: str,
):
    """Add TU index mapping for inputs at {namespace}/input_tu_indices. -1 = always enabled."""
    if stack.tu_id_to_idx is None:
        return

    from biocomp.tumasking import build_input_tu_indices

    tu_indices = build_input_tu_indices(stack, nodelist, stack.tu_id_to_idx)
    params.at(
        f"{namespace}/input_tu_indices",
        tu_indices,
        tags=[NON_GRAD_TAG],
        overwrite=None,
    )


def add_tu_output_mapping(
    params: ParameterTree,
    stack: ComputeStack,
    nodelist: list[StackNode],
    namespace: str,
    n_outputs: int,
):
    """Add TU index mapping for outputs at {namespace}/output_tu_indices. -1 = always enabled."""
    if stack.tu_id_to_idx is None:
        return

    from biocomp.tumasking import build_output_tu_indices

    tu_indices = build_output_tu_indices(stack, nodelist, stack.tu_id_to_idx, n_outputs)
    params.at(
        f"{namespace}/output_tu_indices",
        tu_indices,
        tags=[NON_GRAD_TAG],
        overwrite=None,
    )


def add_node_network_ids(
    params: ParameterTree,
    nodelist: list[StackNode],
    namespace: str,
    stack: "ComputeStack | None" = None,
):
    """Add network_id for each node at {namespace}/node_network_ids.

    Args:
        params: Parameter tree to store the mapping
        nodelist: List of StackNodes in this layer (order must match param indexing)
        namespace: Layer namespace (e.g., "local/3/aggregation")
        stack: Optional ComputeStack for validation (recommended)
    """
    assert len(nodelist) > 0, f"Empty nodelist for {namespace}"

    network_ids = []
    for i, node in enumerate(nodelist):
        nid = node.network_id
        assert isinstance(nid, int) and nid >= 0, (
            f"Invalid network_id {nid} for node {i} in {namespace}"
        )
        if stack is not None:
            assert nid < len(stack.networks), (
                f"network_id {nid} >= n_networks {len(stack.networks)} for node {i} in {namespace}"
            )
        network_ids.append(nid)

    network_ids_arr = jnp.array(network_ids, dtype=jnp.int32)
    assert network_ids_arr.shape[0] == len(nodelist), (
        f"network_ids shape {network_ids_arr.shape} != nodelist len {len(nodelist)}"
    )

    params.at(
        f"{namespace}/node_network_ids",
        network_ids_arr,
        tags=[NON_GRAD_TAG],
        overwrite=None,
    )


def empty_prepare(*_, **__):
    pass
