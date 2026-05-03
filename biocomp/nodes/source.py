from biocomp.jaxutils import flat_concat
from biocomp.compute import StackNode, ComputeStack
import jax
from jax.tree_util import Partial as partial
from jax.typing import ArrayLike
import jax.numpy as jnp
import numpy as np
from biocomp.parameters import ParameterTree, init_if_needed

from biocomp.nodeutils import (
    LayerInstance,
    add_random_var_ids,
    add_node_key_ids,
    add_node_network_ids,
    NON_GRAD_TAG,
    reference_forward_random_var_ids,
    reference_forward_key_ids,
    empty_prepare,
)
from typing import Optional

from .passthrough import single_passthrough
from biocomp.utils import get_logger
from biocomp.neuralutils import (
    ACTIVATION_FUNCTIONS,
    INITIALIZERS,
    DEFAULT_ACTIVATION,
    DEFAULT_OUT_ACTIVATION,
    DEFAULT_INITIALIZER,
    dense_mlp,
)


from biocomp.context import total_context_dim

PRNGKey = ArrayLike
NDArray = np.ndarray | jnp.ndarray

logger = get_logger(__name__)


# source node is just an L2 plasmid, i.e an aggregation that has a fixed ratio of 1:1
# we make it a multi-output node so that it's compatible with the aggregation node but
# really we're just duplicating the input so we could also just use a passthrough node
# or skip the node altogether (for a future version with an optimizer)


# For now, input_shapes will always be [(1,)]
def source(input_shapes: list[tuple[int]], n_outputs: int, **_) -> LayerInstance:
    assert len(input_shapes) == 1, f"A source node should have 1 input, got {len(input_shapes)}"

    def apply(value: ArrayLike, *_, **__) -> tuple[ArrayLike, dict]:
        result = jnp.repeat(value, n_outputs, axis=0)
        return result, {"input_value": value, "n_outputs": n_outputs, "output_shape": result.shape}

    output_shapes = list(input_shapes) * n_outputs

    return LayerInstance(empty_prepare, apply, output_shapes)


# inverse of source is just a passthrough, as it's only inverted when only one output and one input
def inv_source(*args, **kwargs):
    return single_passthrough(*args, **kwargs)


def source_with_pos(
    input_shapes: list[tuple[int]],
    n_outputs: int,
    stack: ComputeStack,
    namespace: str,
    max_L1s: int = 5,
    hidden_s=64,
    depth=3,
    inner_activation_name: str = DEFAULT_ACTIVATION,
    outer_activation_name: str = DEFAULT_OUT_ACTIVATION,
    initializer_name: str = DEFAULT_INITIALIZER,
    bias_offset=0.0,
) -> LayerInstance:
    """Source node with position encoding. Idea is that each Transcription Unit position in the plasmid might have different yields"""
    del stack  # unused

    assert len(input_shapes) == 1, f"A source node should have 1 input, got {len(input_shapes)}"

    inner_activation = ACTIVATION_FUNCTIONS[inner_activation_name]
    outer_activation = ACTIVATION_FUNCTIONS[outer_activation_name]
    initializer = INITIALIZERS[initializer_name]

    def prepare(params: ParameterTree, nodelist: list[StackNode], key, **_):
        add_random_var_ids(params, len(nodelist), len(input_shapes), namespace)
        add_node_key_ids(params, len(nodelist), namespace)
        add_node_network_ids(params, nodelist, namespace)
        _ctx_dim = total_context_dim()
        _dummy_ctx = np.zeros(_ctx_dim) if _ctx_dim > 0 else None
        MLP_head(np.zeros((2 + len(input_shapes),)), params, key, context=_dummy_ctx)

    def MLP_head(vals, params, key, context=None):
        return dense_mlp(
            vals,
            hidden_s,
            1,
            depth=depth,
            activation=inner_activation,
            initializer=initializer,
            bias_offset=bias_offset,
            key=key,
            param_f=partial(init_if_needed, params, base_path="shared"),
            name="NN/source_w_pos",
            context=context,
        )

    def apply(
        value: ArrayLike,
        random_vars: NDArray,
        params: ParameterTree,
        node_id: ArrayLike,
        key,
        tu_enabled_random_vars: Optional[ArrayLike] = None,
        network_id: Optional[ArrayLike] = None,
        **_kwargs,
    ) -> tuple[ArrayLike, dict]:
        context_vector = _kwargs.get("context_vector")
        qid = params[f"{namespace}/random_variable_id"][node_id]
        random_var = random_vars[qid]

        # process each output position
        positions = np.arange(max_L1s)[:n_outputs] / max_L1s
        ans = jax.vmap(
            lambda position: MLP_head(
                flat_concat(value, position, random_var), params, key, context=context_vector
            )
        )(positions)

        # add skip connection and apply activation
        res = 0.5 * ans + 0.5 * jnp.broadcast_to(value, ans.shape)
        activated = outer_activation(res)
        assert activated.shape == (n_outputs, *input_shapes[0]), (
            f"In source_with_pos: {activated.shape} != {(n_outputs, *input_shapes[0])}"
        )
        return activated, {
            "positions": positions,
            "random_var": random_var,
            "pre_activation": res,
            "mlp_output": ans,
            "n_outputs": n_outputs,
        }

    output_shapes = list(input_shapes) * n_outputs

    return LayerInstance(prepare, apply, output_shapes)


def inv_source_with_pos(
    input_shapes: list[tuple[int]],
    n_outputs: int,
    stack: ComputeStack,
    namespace: str,
    max_L1s: int = 5,
    hidden_s=64,
    depth=3,
    inner_activation_name: str = DEFAULT_ACTIVATION,
    outer_activation_name: str = DEFAULT_OUT_ACTIVATION,
    initializer_name: str = DEFAULT_INITIALIZER,
    bias_offset=0.0,
) -> LayerInstance:
    # inverse source is 1->1, inverting a specific position's transformation
    assert len(input_shapes) == 1, f"Inverse source should have 1 input, got {len(input_shapes)}"
    assert n_outputs == 1, f"Inverse source should have 1 output, got {n_outputs}"

    inner_activation = ACTIVATION_FUNCTIONS[inner_activation_name]
    outer_activation = ACTIVATION_FUNCTIONS[outer_activation_name]
    initializer = INITIALIZERS[initializer_name]

    def prepare(params: ParameterTree, nodelist: list[StackNode], key, **_):
        reference_forward_random_var_ids(stack, params, nodelist, namespace)
        reference_forward_key_ids(stack, params, nodelist, namespace)

        # store the original position for each inverse node
        positions = jnp.array(
            [n.get(stack).is_inverse_of.output_slot for n in nodelist], dtype=jnp.int32
        )

        # store positions as a parameter (non-gradient)
        params.at(
            f"{namespace}/original_positions",
            positions,
            tags=[NON_GRAD_TAG],
            overwrite=None,
        )

        # inputs are: value (1) + position (1) + random_var (1) = 3 total
        _ctx_dim = total_context_dim()
        _dummy_ctx = np.zeros(_ctx_dim) if _ctx_dim > 0 else None
        MLP_head(np.zeros((3,)), params, key, context=_dummy_ctx)

    def MLP_head(vals, params, key, context=None):
        """
        MLP for inverse transformation.
        Uses separate parameters from forward node (different namespace).
        """
        return dense_mlp(
            vals,
            hidden_s,
            output_s=1,  # output dimension is 1
            depth=depth,
            activation=inner_activation,
            initializer=initializer,
            bias_offset=bias_offset,
            key=key,
            param_f=partial(init_if_needed, params, base_path="shared"),
            name="NN/inv_source_w_pos",
            context=context,
        )

    def apply(
        value: NDArray,
        random_vars: NDArray,
        params: ParameterTree,
        node_id: ArrayLike,
        key,
        tu_enabled_random_vars: Optional[ArrayLike] = None,
        network_id: Optional[ArrayLike] = None,
        **_kwargs,
    ) -> tuple[ArrayLike, dict]:
        context_vector = _kwargs.get("context_vector")
        assert value.shape == input_shapes[0], f"Invalid input shape {value.shape}"

        qid = params.at(f"{namespace}/random_variable_id")[node_id]
        random_var = random_vars[qid][0]

        # get the original position this node is inverting
        original_position = params.at(f"{namespace}/original_positions")[node_id]
        normalized_position = original_position / max_L1s

        # flatten value if needed (ensure it's 1D for concatenation)
        if value.ndim == 0:
            value_flat = value.reshape((1,))
        else:
            value_flat = value.flatten()

        # apply inverse transformation for this specific position
        mlp_input = flat_concat(value_flat, normalized_position, random_var)
        mlp_out = MLP_head(mlp_input, params, key, context=context_vector)

        # add skip connection and apply activation
        mlp_out_reshaped = mlp_out.reshape(value.shape)
        pre_activation = 0.5 * mlp_out_reshaped + 0.5 * value
        result = outer_activation(pre_activation)

        return result, {
            "original_position": original_position,
            "normalized_position": normalized_position,
            "random_var": random_var,
            "mlp_input": mlp_input,
            "mlp_output": mlp_out_reshaped,
            "pre_activation": pre_activation,
        }

    output_shapes = input_shapes  # 1 input shape -> 1 output shape

    return LayerInstance(prepare, apply, output_shapes)


def simple_source_with_pos(
    input_shapes: list[tuple[int]],
    n_outputs: int,
    stack: ComputeStack,
    namespace: str,
    max_L1s: int = 5,
    **_,
) -> LayerInstance:
    """Simple source node: y_p = x * 0.9^p where p is the position"""
    del stack  # unused

    assert len(input_shapes) == 1, f"A source node should have 1 input, got {len(input_shapes)}"

    def apply(value: ArrayLike, **_) -> tuple[ArrayLike, dict]:
        # compute position scaling factors: 0.9^p for p in [0, n_outputs)
        positions = np.arange(n_outputs)
        scale_factors = 0.9**positions

        # apply scaling to each output position
        result = jnp.array([value * scale for scale in scale_factors])

        return result, {"positions": positions, "scale_factors": scale_factors}

    output_shapes = list(input_shapes) * n_outputs

    return LayerInstance(empty_prepare, apply, output_shapes)


def simple_inv_source_with_pos(
    input_shapes: list[tuple[int]],
    n_outputs: int,
    stack: ComputeStack,
    namespace: str,
    max_L1s: int = 5,
    **_,
) -> LayerInstance:
    """Inverse of simple source: x = y_p / 0.9^p"""
    assert len(input_shapes) == 1, f"Inverse source should have 1 input, got {len(input_shapes)}"
    assert n_outputs == 1, f"Inverse source should have 1 output, got {n_outputs}"

    def prepare(params: ParameterTree, nodelist: list[StackNode], key, **_):
        # store the original position for each inverse node
        positions = jnp.array(
            [n.get(stack).is_inverse_of.output_slot for n in nodelist], dtype=jnp.int32
        )
        params.at(f"{namespace}/original_positions", positions, tags=[NON_GRAD_TAG])

    def apply(
        value: ArrayLike, params: ParameterTree, node_id: ArrayLike, **_
    ) -> tuple[ArrayLike, dict]:
        # get the original position this node is inverting
        original_position = params.at(f"{namespace}/original_positions")[node_id]
        scale_factor = 0.9**original_position

        # invert: x = y_p / 0.9^p
        result = value / scale_factor

        return result, {"original_position": original_position, "scale_factor": scale_factor}

    output_shapes = input_shapes

    return LayerInstance(prepare, apply, output_shapes)
