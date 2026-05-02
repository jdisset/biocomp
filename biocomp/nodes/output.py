from biocomp.compute import StackNode, ComputeStack
from jax.tree_util import Partial as partial
from jax.typing import ArrayLike
import jax.numpy as jnp
from jax import vmap
import numpy as np
from biocomp.parameters import ParameterTree, init_if_needed
from biocomp.jaxutils import flat_concat
from biocomp.nodeutils import (
    LayerInstance,
    add_tu_input_mapping,
    add_node_network_ids,
    add_random_var_ids,
    add_node_key_ids,
    reference_forward_key_ids,
    NON_GRAD_TAG,
)
from typing import Optional

from biocomp.neuralutils import (
    ACTIVATION_FUNCTIONS,
    INITIALIZERS,
    DEFAULT_ACTIVATION,
    DEFAULT_OUT_ACTIVATION,
    DEFAULT_INITIALIZER,
    dense_mlp,
    dummy_mlp,
)

PRNGKey = ArrayLike
NDArray = np.ndarray | jnp.ndarray

### {{{                    --     output (fluorescence) node     --


def grouped_output(
    input_shapes: list[tuple[int, ...]],
    n_outputs: int,  # unused
    stack: ComputeStack,  # unused
    namespace: str,
    wsize: int = 64,
    depth: int = 4,
    bias_offset: float = 0.0,
    inner_activation_name: str = DEFAULT_ACTIVATION,
    outer_activation_name: str = DEFAULT_OUT_ACTIVATION,
    initializer_name: str = DEFAULT_INITIALIZER,
    dummy: bool = False,  # disable neural + residual, for testing
    **_,
):
    del n_outputs
    # stack is used for TU mapping in prepare()

    assert all(shape == input_shapes[0] for shape in input_shapes)
    inner_activation = ACTIVATION_FUNCTIONS[inner_activation_name]
    outer_activation = ACTIVATION_FUNCTIONS[outer_activation_name]
    initializer = INITIALIZERS[initializer_name]

    mlp = dummy_mlp if dummy else dense_mlp

    # Check if model was trained with random_var in output MLP (set by build_stack)
    _use_random_var = (
        stack.config is not None
        and stack.config.extra is not None
        and stack.config.extra.get("output_has_random_var", True)
    )

    def MLP_head(x, random_var, rng_key, params):
        inp = flat_concat(x, random_var) if _use_random_var else flat_concat(x)
        return mlp(
            inp,
            wsize,
            1,
            depth,
            param_f=partial(init_if_needed, params, base_path="shared"),
            initializer=initializer,
            bias_offset=bias_offset,
            key=rng_key,
            name="NN/grouped_output",
            activation=inner_activation,
        )

    def prepare(params: ParameterTree, nodelist: list[StackNode], key: PRNGKey):
        MLP_head(x=np.zeros(input_shapes[0]), random_var=np.zeros(()), rng_key=key, params=params)
        add_random_var_ids(params, len(nodelist), len(input_shapes), namespace)
        add_node_key_ids(params, len(nodelist), namespace)
        add_tu_input_mapping(params, stack, nodelist, namespace)
        add_node_network_ids(params, nodelist, namespace)

    def apply(
        *inputs: ArrayLike,
        random_vars: NDArray,
        params: ParameterTree,
        node_id: ArrayLike,
        key,
        tu_enabled_random_vars: Optional[ArrayLike] = None,
        network_id: Optional[ArrayLike] = None,
        **_kwargs,
    ) -> tuple[ArrayLike, dict]:
        inputs_arr = jnp.array(inputs)
        assert len(inputs_arr) == len(input_shapes)

        rvid = params[f"{namespace}/random_variable_id"][node_id]
        random_var = random_vars[rvid]

        input_tu_indices_path = f"{namespace}/input_tu_indices"
        if input_tu_indices_path in params:
            from biocomp.tumasking import get_tu_masks

            tu_indices = params[input_tu_indices_path][node_id]
            input_masks = get_tu_masks(params, tu_indices, network_id, is_multi_tu=True)
        else:
            input_masks = jnp.ones(len(input_shapes))

        res = vmap(lambda x, rv: MLP_head(x, rv, rng_key=key, params=params))(
            inputs_arr, random_var
        )

        masks_reshaped = input_masks.reshape(-1, *([1] * len(input_shapes[0])))
        masked_inputs = inputs_arr * masks_reshaped
        masked_res = res * input_masks.reshape(-1, 1)
        masked_inputs_scalar = jnp.sum(
            masked_inputs,
            axis=tuple(range(1, masked_inputs.ndim)),
        ).reshape(-1, 1)

        if dummy:
            pre = masked_res
            output = masked_res
        else:
            pre = 0.5 * masked_res + 0.5 * masked_inputs_scalar
            output = outer_activation(pre)

        return output, {
            "mlp_outputs": res,
            "pre_activation": pre,
            "n_inputs": len(inputs_arr),
            "input_values": inputs_arr,
            "input_masks": input_masks,
            "input_scalar_residual": masked_inputs_scalar,
            "random_var": random_var,
        }

    output_shape = [(1,)] * len(input_shapes)

    return LayerInstance(prepare, apply, output_shape)


##────────────────────────────────────────────────────────────────────────────}}}##


### {{{                    --     inv_output (inverse fluorescence) node     --


def inv_output(
    input_shapes: list[tuple[int, ...]],
    n_outputs: int,
    stack: ComputeStack,
    namespace: str,
    wsize: int = 64,
    depth: int = 4,
    bias_offset: float = 0.0,
    inner_activation_name: str = DEFAULT_ACTIVATION,
    outer_activation_name: str = DEFAULT_OUT_ACTIVATION,
    initializer_name: str = DEFAULT_INITIALIZER,
    dummy: bool = False,
    **_,
):
    assert len(input_shapes) == 1, f"inv_output should have 1 input, got {len(input_shapes)}"
    assert n_outputs == 1, f"inv_output should have 1 output, got {n_outputs}"

    inner_activation = ACTIVATION_FUNCTIONS[inner_activation_name]
    outer_activation = ACTIVATION_FUNCTIONS[outer_activation_name]
    initializer = INITIALIZERS[initializer_name]

    mlp = dummy_mlp if dummy else dense_mlp

    def MLP_head(x, random_var, rng_key, params):
        return mlp(
            flat_concat(x, random_var),
            wsize,
            1,
            depth,
            param_f=partial(init_if_needed, params, base_path="shared"),
            initializer=initializer,
            bias_offset=bias_offset,
            key=rng_key,
            name="NN/inv_output",
            activation=inner_activation,
        )

    def prepare(params: ParameterTree, nodelist: list[StackNode], key: PRNGKey):
        MLP_head(x=np.zeros(input_shapes[0]), random_var=np.zeros(()), rng_key=key, params=params)
        # Resolve each inverse-output node's forward random_variable_id at its
        # specific output_slot, eagerly, into a plain 1-D int array. The
        # forward output layers run earlier in the reverse-order init loop, so
        # their random_variable_id arrays already exist in params here.
        #
        # Why eager rather than an ArrayRef across forward layers:
        #
        # 1. Heterogeneous slot widths. Forward output layers bucket networks
        #    by signature, so num_per_node varies (e.g. 3 vs 4 input edges)
        #    across forward output layers in the same dataset. A row-style
        #    ArrayRef requires uniform inner shape. A scalar (fwd_pos, slot)
        #    ArrayRef survives heterogeneity but produces a multi-index gather
        #    in the JIT'd apply graph.
        #
        # 2. The multi-index gather pattern, fused with the rest of the apply
        #    pipeline on heterogeneous-output datasets, has hit XLA runtime
        #    dispatch failures (`Function copy_gather_fusion.* not found`) on
        #    JAX 0.6.0 / Apple Silicon. Pre-resolving to a plain int array
        #    bypasses the ArrayRef gather entirely.
        #
        # The ID values are non-grad and allocated exactly once by the forward
        # output layer's prepare(), so eager resolution is safe.
        rvids_resolved = []
        for node in nodelist:
            fwd_node = node.get_forward_stacknode(stack)
            fwd_namespace = stack.get_layer_namespace(fwd_node.layer_number)
            slot = node.get(stack).is_inverse_of.output_slot
            fwd_rvid_arr = params.data[f"{fwd_namespace}/random_variable_id"]
            rvids_resolved.append(int(fwd_rvid_arr[fwd_node.node_position_in_layer, slot]))
        params.at(
            f"{namespace}/random_variable_id",
            jnp.array(rvids_resolved, dtype=jnp.int32),
            tags=[NON_GRAD_TAG],
        )
        reference_forward_key_ids(stack, params, nodelist, namespace)
        add_node_network_ids(params, nodelist, namespace)

    def apply(
        value: ArrayLike,
        *,
        random_vars: NDArray,
        params: ParameterTree,
        node_id: ArrayLike,
        key,
        **_kwargs,
    ) -> tuple[ArrayLike, dict]:
        assert value.shape == input_shapes[0], (
            f"inv_output: expected {input_shapes[0]}, got {value.shape}"
        )

        rvid = params[f"{namespace}/random_variable_id"][node_id]
        random_var = random_vars[rvid]

        mlp_out = MLP_head(value, random_var, key, params)

        if dummy:
            result = mlp_out
        else:
            pre = 0.5 * mlp_out + 0.5 * value
            result = outer_activation(pre)

        return result, {
            "mlp_output": mlp_out,
            "input_value": value,
            "random_var": random_var,
        }

    output_shape = list(input_shapes)

    return LayerInstance(prepare, apply, output_shape)


##────────────────────────────────────────────────────────────────────────────}}}##
