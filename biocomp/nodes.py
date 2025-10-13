from __future__ import annotations
from .library import PartsLibrary as PartsLibrary
import jax
from jax import vmap
from jax.tree_util import Partial as partial
import jax.numpy as jnp
import numpy as np

from .utils import get_logger
from .jaxutils import flat_concat
from . import quantization as qz

from .parameters import ArrayRef, ParameterTree, init_if_needed, make_view, get_param
from .neuralutils import (
    ACTIVATION_FUNCTIONS,
    INITIALIZERS,
    DEFAULT_ACTIVATION,
    DEFAULT_OUT_ACTIVATION,
    DEFAULT_INITIALIZER,
    dense_mlp,
    uniform_initializer,
)

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .compute import StackNode, ComputeStack

from jax.typing import ArrayLike
from typing import Callable, Tuple, List, Dict
from dataclasses import dataclass

PRNGKey = ArrayLike

logger = get_logger(__name__)

NON_GRAD_TAG = "non_grad"


# =========================== Utils ===========================
@dataclass
class LayerInstance:
    prepare: Callable
    apply: Callable  # Returns tuple of (result, aux_dict)
    output_shapes: List[Tuple[int]]
    commit: Optional[Callable] = None

    def __post_init__(self):
        assert all(isinstance(shape, tuple) for shape in self.output_shapes), (
            f"Invalid output shapes: {self.output_shapes}"
        )
        assert all(all(isinstance(dim, int) for dim in shape) for shape in self.output_shapes), (
            f"Non-integer dimensions in output shapes: {self.output_shapes}"
        )




##────────────────────────────────────────────────────────────────────────────}}}

### {{{                 --     quantile variable helpers     --

# TODO: "quantile variable" is a bit confusing, and my tired brain often confuses it with quantization stuff when reading a bit too fast.
# Maybe should be renamed to "{random/sample/latent} variable" or something like that, or just Z?

GLOBAL_PATH_NUMBER_OF_QUANTILE_VARIABLES = "global/number_of_quantile_variables"


def get_prev_num_quantile_vars(params: ParameterTree):
    try:
        return params[GLOBAL_PATH_NUMBER_OF_QUANTILE_VARIABLES]
    except KeyError:
        return 0


def add_quantile_var_ids(params: ParameterTree, num_nodes: int, num_per_node, namespace: str):
    """
    Adds quantile variable IDs to the parameters. The quantile variable is just a random variable
    used for generation (ideally the node learns a quantile function,
    and this is the quantile variable fed to that function).
    It updates (or creates) the following parameters:
        - global/number_of_quantile_variables -> int, total number of quantile variables (across all neural functions aka nodes)
        - local/{layer_name}/quantile_variable_id -> id array of shape (num_nodes, num_per_node)
    Then a node can access its quantile variable IDs by simply indexing the vector of quantile variables (Z) with these ids

    :param params: The parameters tree to update.
    :param num_nodes: The number of nodes for which to add quantile variable IDs.
    :param num_per_node: The number of quantile variables per node.
    :param layer_name: The name (possibly subpath) of the layer to which these quantile variables belong.

    """

    prev_num_quantile_vars = get_prev_num_quantile_vars(params)
    new_num_quantile_vars = prev_num_quantile_vars + num_nodes * num_per_node
    quantile_var_ids = jnp.arange(prev_num_quantile_vars, new_num_quantile_vars).reshape(
        (num_nodes, num_per_node)
    )
    params.at(
        f"{namespace}/quantile_variable_id",  # NO "local/" prefix needed
        quantile_var_ids,
        tags=[NON_GRAD_TAG],
        overwrite=False,
    )

    params.at(
        GLOBAL_PATH_NUMBER_OF_QUANTILE_VARIABLES,
        new_num_quantile_vars,
        tags=[NON_GRAD_TAG],
        overwrite=True,
    )


##────────────────────────────────────────────────────────────────────────────}}}

# =========================== Simple Nodes ===========================
### {{{                   --     passthrough, source, numeric    --

# When we create a compute layer, we pass the shape of all inputs as a list of tuples.
# Indeed, a node can have several inputs, and each input can have a different shape.
# The node constructor must then return the apply function, and the shape of the outputs.
# There can also be multiple outputs, each of which can have a different shape.
# we also pass the numper of outputs, which is useful for the source node.

# one question is whether we shouls allow for multiple outputs with different shapes.
# I don't think it's necessary for now, but at the same time it's not a big deal to allow it.
# So I guess yes, we should allow it. that means that we output a tuple of arrays


# Signatures:
# prepare (params, node, key)
# apply (*values:ArrayLike, quantiles:ArrayLike, params:ParameterTree, node_id:ArrayLike, key)


def empty_prepare(*_, **__):
    pass


# input_shapes is a list of shape tuples, one for each input
def single_passthrough(input_shapes: List[Tuple[int]], *_, **__) -> LayerInstance:
    assert len(input_shapes) == 1, f"Passthrough expects 1 input, got {len(input_shapes)}"

    def apply(value: ArrayLike, **___) -> Tuple[ArrayLike, Dict]:
        return value, {"input_shape": value.shape}

    output_shapes = input_shapes

    return LayerInstance(empty_prepare, apply, output_shapes)


# source node is just an L2 plasmid, i.e an aggregation that has a fixed ratio of 1:1
# we make it a multi-output node so that it's compatible with the aggregation node but
# really we're just duplicating the input so we could also just use a passthrough node
# or skip the node altogether (for a future version with an optimizer)


# For now, input_shapes will always be [(1,)]
def source(input_shapes: List[Tuple[int]], n_outputs: int, **_) -> LayerInstance:
    assert len(input_shapes) == 1, f"A source node should have 1 input, got {len(input_shapes)}"

    def apply(value: ArrayLike, *_, **__) -> Tuple[ArrayLike, Dict]:
        result = jnp.repeat(value, n_outputs, axis=0)
        return result, {"input_value": value, "n_outputs": n_outputs, "output_shape": result.shape}

    output_shapes = list(input_shapes) * n_outputs

    return LayerInstance(empty_prepare, apply, output_shapes)


# inverse of source is just a passthrough, as it's only inverted when only one output and one input
def inv_source(*args, **kwargs):
    return single_passthrough(*args, **kwargs)


def source_with_pos(
    input_shapes: List[Tuple[int]],
    n_outputs: int,
    layer_id: int,
    stack: ComputeStack,
    namespace: str,
    max_L1s: int = 5,
    hidden_s=64,
    depth=3,
    inner_activation_name: str = DEFAULT_ACTIVATION,
    outer_activation_name: str = DEFAULT_OUT_ACTIVATION,
    initializer_name: str = DEFAULT_INITIALIZER,
    bias_offset=0.0,
    **_,
) -> LayerInstance:
    """Source node with position encoding. Idea is that each Transcription Unit position in the plasmid might have different yields"""

    assert len(input_shapes) == 1, f"A source node should have 1 input, got {len(input_shapes)}"

    inner_activation = ACTIVATION_FUNCTIONS[inner_activation_name]
    outer_activation = ACTIVATION_FUNCTIONS[outer_activation_name]
    initializer = INITIALIZERS[initializer_name]

    def prepare(params: ParameterTree, nodelist: List[StackNode], key, **_):
        add_quantile_var_ids(params, len(nodelist), len(input_shapes), namespace)
        params.at(
            f"{namespace}/input_shapes",
            jnp.array(input_shapes, dtype=jnp.int32),
            tags=[NON_GRAD_TAG],
        )
        MLP_head(np.zeros((2 + len(input_shapes),)), params, key)

    def MLP_head(vals, params, key):
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
        )

    def apply(
        value: ArrayLike,
        quantiles: ArrayLike,
        params: ParameterTree,
        node_id: ArrayLike,
        key,
    ) -> Tuple[ArrayLike, Dict]:
        qid = params[f"{namespace}/quantile_variable_id"][node_id]
        quantile = quantiles[qid]

        # process each output position
        positions = np.arange(max_L1s)[:n_outputs] / max_L1s
        ans = jax.vmap(
            lambda position: MLP_head(flat_concat(value, position, quantile), params, key)
        )(positions)

        # add skip connection and apply activation
        res = 0.5 * ans + 0.5 * jnp.broadcast_to(value, ans.shape)
        activated = outer_activation(res)
        assert activated.shape == (n_outputs, *input_shapes[0]), (
            f"In source_with_pos: {activated.shape} != {(n_outputs, *input_shapes[0])}"
        )
        return activated, {
            "positions": positions,
            "quantile": quantile,
            "pre_activation": res,
            "mlp_output": ans,
            "n_outputs": n_outputs,
        }

    output_shapes = list(input_shapes) * n_outputs

    return LayerInstance(prepare, apply, output_shapes)


def inv_source_with_pos(
    input_shapes: List[Tuple[int]],
    n_outputs: int,
    stack: ComputeStack,
    layer_id: int,
    namespace: str,
    max_L1s: int = 5,
    hidden_s=64,
    depth=3,
    inner_activation_name: str = DEFAULT_ACTIVATION,
    outer_activation_name: str = DEFAULT_OUT_ACTIVATION,
    initializer_name: str = DEFAULT_INITIALIZER,
    bias_offset=0.0,
    **_,
) -> LayerInstance:
    # inverse source is 1->1, inverting a specific position's transformation
    assert len(input_shapes) == 1, f"Inverse source should have 1 input, got {len(input_shapes)}"
    assert n_outputs == 1, f"Inverse source should have 1 output, got {n_outputs}"

    inner_activation = ACTIVATION_FUNCTIONS[inner_activation_name]
    outer_activation = ACTIVATION_FUNCTIONS[outer_activation_name]
    initializer = INITIALIZERS[initializer_name]

    def prepare(params: ParameterTree, nodelist: List[StackNode], key, **_):
        # add quantile variables - one per node
        add_quantile_var_ids(params, len(nodelist), 1, namespace)

        assert stack is not None, "Stack must be provided for inverse source node."

        # store the original position for each inverse node
        positions = []
        for node in nodelist:
            original = node.get(stack).is_inverse_of
            assert original is not None, "Inverse node must have a forward node."
            assert original.output_slot < original.output_len, (
                f"Original slot {original.output_slot} out of bounds for output length {original.output_len}"
            )

            positions.append(original.output_slot)
        positions = jnp.array(positions, dtype=jnp.int32)

        # store positions as a parameter (non-gradient)
        params.at(
            f"{namespace}/original_positions",
            positions,
            tags=[NON_GRAD_TAG],
            overwrite=None,
        )

        # initialize the inverse MLP with dummy inputs
        # inputs are: value (1) + position (1) + quantile (1) = 3 total
        dummy_input = np.zeros((3,))
        MLP_head(dummy_input, params, key)

    def MLP_head(vals, params, key):
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
        )

    def apply(
        value: ArrayLike,
        quantiles: ArrayLike,
        params: ParameterTree,
        node_id: ArrayLike,
        key,
    ) -> Tuple[ArrayLike, Dict]:
        assert value.shape == input_shapes[0], f"Invalid input shape {value.shape}"

        qid = params.at(f"{namespace}/quantile_variable_id")[node_id]
        quantile = quantiles[qid][0]

        # get the original position this node is inverting
        original_position = params.at(f"{namespace}/original_positions")[node_id]
        normalized_position = original_position / max_L1s

        # flatten value if needed (ensure it's 1D for concatenation)
        if value.ndim == 0:
            value_flat = value.reshape((1,))
        else:
            value_flat = value.flatten()

        # apply inverse transformation for this specific position
        mlp_input = flat_concat(value_flat, normalized_position, quantile)
        mlp_out = MLP_head(mlp_input, params, key)

        # add skip connection and apply activation
        mlp_out_reshaped = mlp_out.reshape(value.shape)
        pre_activation = 0.5 * mlp_out_reshaped + 0.5 * value
        result = outer_activation(pre_activation)

        return result, {
            "original_position": original_position,
            "normalized_position": normalized_position,
            "quantile": quantile,
            "mlp_input": mlp_input,
            "mlp_output": mlp_out_reshaped,
            "pre_activation": pre_activation,
        }

    output_shapes = input_shapes  # 1 input shape -> 1 output shape

    return LayerInstance(prepare, apply, output_shapes)


def hard_bias(
    input_shapes: List[Tuple[int]],
    n_outputs: int,
    layer_id: int,
    stack,
    namespace: str,
    valid_range: Tuple[float, float] = (0.0, 0.8),
    shape: Tuple[int] = (1,),
    init_value: Optional[float] = 0.5,
    random_init: bool = False,
    **_,
) -> LayerInstance:
    assert n_outputs == 1, f"Bias node should have 1 output, got {n_outputs}"
    assert len(input_shapes) == 0

    def clamp_to_range(value: ArrayLike):
        # hard clamp to valid_range. scale is ignored
        return jnp.clip(value, valid_range[0], valid_range[1])

    def prepare(params: ParameterTree, nodelist: List[StackNode], key, **_):
        raw_values = []

        for node in nodelist:
            extra = node.get(stack).extra
            if extra and not random_init:
                # try to use values from extra dict
                if "raw_value" in extra:
                    raw_values.append(
                        jnp.array(extra["raw_value"], dtype=jnp.float32).reshape(shape)
                    )
                elif "bias_value" in extra:
                    # use bias_value directly as raw_value (will be clamped in apply)
                    raw_values.append(
                        jnp.array(extra["bias_value"], dtype=jnp.float32).reshape(shape)
                    )
                else:
                    # no valid values in extra, use init_value or random
                    if init_value is not None:
                        raw_values.append(jnp.full(shape, init_value, dtype=jnp.float32))
                    else:
                        raw_values.append(
                            jax.random.uniform(
                                key, shape, minval=valid_range[0], maxval=valid_range[1]
                            )
                        )
            else:
                # random init requested or no extra dict
                if init_value is not None and not random_init:
                    raw_values.append(jnp.full(shape, init_value, dtype=jnp.float32))
                else:
                    raw_values.append(
                        jax.random.uniform(key, shape, minval=valid_range[0], maxval=valid_range[1])
                    )

        params[f"{namespace}/raw_value"] = jnp.stack(raw_values)

    def apply(*_, params: ParameterTree, node_id: ArrayLike, **__) -> Tuple[ArrayLike, Dict]:
        raw_bias_value = params[f"{namespace}/raw_value"][node_id]
        bias_value = clamp_to_range(raw_bias_value)
        return bias_value, {
            "raw_bias_value": raw_bias_value,
            "bias_value": bias_value,
        }

    def commit(params: ParameterTree, nodelist: List[StackNode], **_):
        for i, n in enumerate(nodelist):
            newextra = {}
            bias_value = clamp_to_range(params[f"{namespace}/raw_value"][i])
            newextra["bias_value"] = bias_value
            n.get(stack).extra.update(newextra)

    output_shapes = [tuple(shape)]  # single output shape

    return LayerInstance(prepare, apply, output_shapes, commit=commit)


def bias(
    input_shapes: List[Tuple[int]],
    n_outputs: int,
    layer_id: int,
    stack,
    namespace: str,
    valid_range: Tuple[float, float] = (0.0, 0.6),
    shape: Tuple[int] = (1,),
    random_init: bool = False,
    **_,
) -> LayerInstance:
    """A bias node that outputs a learnable (softer) bias value within a specified valid range."""
    assert n_outputs == 1, f"Bias node should have 1 output, got {n_outputs}"
    assert len(input_shapes) == 0

    def clamp_to_range(value: ArrayLike, scale: ArrayLike):
        # scaled sigmoid function to clamp value to valid_range. smaller scale = sharper transition
        s = jax.nn.sigmoid(scale) + 0.001
        scaled_sigmoid = jax.nn.sigmoid(value / s)
        return scaled_sigmoid * (valid_range[1] - valid_range[0]) + valid_range[0]

    def inverse_clamp(value: ArrayLike, scale: ArrayLike):
        # inverse of clamp_to_range, to get raw_value from bias_value and scale
        s = jax.nn.sigmoid(scale) + 0.001
        y = (value - valid_range[0]) / (valid_range[1] - valid_range[0])
        y = jnp.clip(y, 1e-5, 1 - 1e-5)
        return -s * jnp.log((1 / y) - 1)

    def prepare(params: ParameterTree, nodelist: List[StackNode], key, **_):
        raw_values = []
        scales = []

        for node in nodelist:
            extra = node.get(stack).extra
            if extra and not random_init:
                # try to use values from extra dict
                if "raw_value" in extra and "scale" in extra:
                    raw_values.append(
                        jnp.array(extra["raw_value"], dtype=jnp.float32).reshape(shape)
                    )
                    scales.append(jnp.array(extra["scale"], dtype=jnp.float32))
                    bias_value = clamp_to_range(raw_values[-1], scales[-1])
                    if "bias_value" in extra:
                        assert jnp.allclose(
                            bias_value,
                            jnp.array(extra["bias_value"], dtype=jnp.float32).reshape(shape),
                            atol=1e-4,
                        ), "Inconsistent bias_value in extra dict"
                elif "bias_value" in extra:
                    scale = extra.get("scale", 0.0)
                    scales.append(jnp.array(scale, dtype=jnp.float32))
                    bias_v = jnp.array(extra["bias_value"], dtype=jnp.float32).reshape(shape)
                    raw_v = inverse_clamp(bias_v, scales[-1])
                    raw_values.append(raw_v)
                else:
                    # no valid values in extra, use random init for this node
                    raw_values.append(jax.random.uniform(key, shape, minval=-1, maxval=1))
                    scales.append(jnp.array(0.0, dtype=jnp.float32))
            else:
                # random init requested or no extra dict
                raw_values.append(jax.random.uniform(key, shape, minval=-1, maxval=1))
                scales.append(jnp.array(0.0, dtype=jnp.float32))

        params[f"{namespace}/raw_value"] = jnp.stack(raw_values)
        params[f"{namespace}/scale"] = jnp.stack(scales)

    def apply(*_, params: ParameterTree, node_id: ArrayLike, **__) -> Tuple[ArrayLike, Dict]:
        raw_bias_value = params[f"{namespace}/raw_value"][node_id]
        scale = params[f"{namespace}/scale"][node_id]
        bias_value = clamp_to_range(raw_bias_value, scale)

        return bias_value, {
            "raw_bias_value": raw_bias_value,
            "bias_value": bias_value,
            "scale": scale,
        }

    def commit(params: ParameterTree, nodelist: List[StackNode], **_):
        for i, n in enumerate(nodelist):
            updt = {
                "scale": params[f"{namespace}/scale"][i],
                "raw_value": params[f"{namespace}/raw_value"][i],
            }
            updt["bias_value"] = clamp_to_range(updt["raw_value"], updt["scale"])
            n.get(stack).extra.update(updt)

    output_shapes = [tuple(shape)]  # single output shape

    return LayerInstance(prepare, apply, output_shapes, commit=commit)


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                   --     aggregation node   --


def aggregation(
    input_shapes: List[Tuple[int]],
    n_outputs: int,
    layer_id: int,
    stack: ComputeStack,
    namespace: str,
    random_init: bool = False,
    **_,
) -> LayerInstance:
    assert len(input_shapes) == 1, f"Aggregation expects 1 input, got {len(input_shapes)}"
    pname = "ratios"

    def prepare(params: ParameterTree, nodelist: List[StackNode], key: PRNGKey, **_):
        ratios = []
        for node in nodelist:
            extra = node.get(stack).extra
            if "ratios" in extra and not random_init:
                assert len(extra["ratios"]) == n_outputs
                ratio_v = jnp.array(extra["ratios"], dtype=jnp.float32)
            else:
                ratio_v = jax.random.uniform(key, (n_outputs,), minval=0.05, maxval=1.0)
            ratios.append(ratio_v)

        ratios = jnp.stack(ratios)
        assert ratios.shape == (len(nodelist), n_outputs), f"Invalid ratio shape {ratios.shape}"
        params[f"{namespace}/{pname}"] = ratios

    def apply(
        input: ArrayLike,
        quantiles: ArrayLike,
        params: ParameterTree,
        node_id: ArrayLike,
        key: PRNGKey,
    ) -> Tuple[ArrayLike, Dict]:
        assert input.shape == input_shapes[0], f"Invalid input shape {input.shape}"
        ratios = params[f"{namespace}/{pname}"][node_id][:n_outputs]
        abs_ratios = jnp.abs(jnp.array(ratios))
        result = abs_ratios * input
        return result, {"ratios": ratios, "abs_ratios": abs_ratios, "n_outputs": n_outputs}

    def commit(params: ParameterTree, nodelist: List[StackNode], **_):
        for i, n in enumerate(nodelist):
            updt = {}
            ratios = params[f"{namespace}/{pname}"][i]

            # normalize absolute ratios so that the minimum is 1
            ratios_array = jnp.abs(jnp.array(ratios))  # use absolute values like in apply
            # find the minimum non-zero ratio
            positive_ratios = ratios_array[ratios_array > 0]
            min_ratio = jnp.min(positive_ratios) if len(positive_ratios) > 0 else 1.0
            min_ratio = jnp.maximum(min_ratio, 1e-9)  # avoid division by zero
            normalized_ratios = ratios_array / min_ratio

            # update extra dict
            updt["ratios"] = normalized_ratios.tolist()[:n_outputs]
            n.get(stack).extra.update(updt)

    output_shape = input_shapes * n_outputs

    return LayerInstance(prepare, apply, output_shape, commit)


def inv_aggregation(
    input_shapes: List[Tuple[int]],
    n_outputs: int,
    stack: ComputeStack,
    layer_id: int,
    namespace: str,
    **_,
) -> LayerInstance:
    # an inverse aggregation node always has 1 input and 1 output
    assert len(input_shapes) == 1, f"inverse_Aggregation expects 1 input, got {len(input_shapes)}"
    assert n_outputs == 1, f"inverse_Aggregation expects 1 output, got {n_outputs}"

    def prepare(params: ParameterTree, nodelist: List[StackNode], **_):
        if stack is not None:
            ref = ArrayRef(params.data)
            for node in nodelist:
                extra = node.get(stack).extra
                assert extra["original_output_slot"] < extra["original_output_len"]
                original_slot = extra["original_output_slot"]

                fwd_node = node.get_forward_stacknode(stack)
                fwd_namespace = stack.get_layer_namespace(fwd_node.layer_number)
                ref.push_back(
                    f"{fwd_namespace}/ratios", (fwd_node.node_position_in_layer, original_slot)
                )

            params.at(f"{namespace}/ratios", ref, overwrite=None)

    EPSILON = 1e-9

    def apply(
        input: ArrayLike,
        quantiles: ArrayLike,
        params: ParameterTree,
        node_id: ArrayLike,
        key,
    ) -> Tuple[ArrayLike, Dict]:
        ratio = jnp.abs(params[f"{namespace}/ratios"][node_id])
        clamped_ratio = jnp.maximum(ratio, EPSILON)
        result = input / clamped_ratio

        return result, {"ratio": ratio, "clamped_ratio": clamped_ratio, "epsilon": EPSILON}

    output_shape = input_shapes
    return LayerInstance(prepare, apply, output_shape)


##────────────────────────────────────────────────────────────────────────────}}}


# =========================== Neural Nodes ===========================
### {{{                   --     transform node (tc, tl)     --
def transform_nn(
    input_shapes: List[Tuple[int]],
    n_outputs: int,
    stack: ComputeStack,
    layer_id: int,
    namespace: str,
    transform_name: str,
    quantization_names: List[str],  # ordered list. ex: ['1xuorf', '2xuorf', ...]
    outer_wsize: int = 64,
    outer_depth: int = 4,
    inner_wsize: int = 64,
    inner_depth: int = 3,
    inner_outsize: int = 8,
    rate_dim: int = 1,
    is_inverse: bool = False,
    inner_activation_name: str = DEFAULT_ACTIVATION,
    outer_activation_name: str = DEFAULT_OUT_ACTIVATION,
    initializer_name: str = DEFAULT_INITIALIZER,
    bias_offset: float = 0.0,
    alpha_init: float = 0.5,
    beta_init: float = 0.5,
    **_,
):
    # TODO: make sure incoming edges order is deterministic

    assert n_outputs == 1, f"NN transform only supports 1 output, got {n_outputs}"
    if is_inverse and len(input_shapes) != 1:
        raise ValueError(f"Inverse {transform_name} should have 1 input, got {len(input_shapes)}")

    if not all(s == input_shapes[0] for s in input_shapes):
        raise ValueError(
            f"All inputs of a transformation should have the same shape, got {input_shapes}"
        )

    inner_activation = ACTIVATION_FUNCTIONS[inner_activation_name]
    outer_activation = ACTIVATION_FUNCTIONS[outer_activation_name]
    initializer = INITIALIZERS[initializer_name]

    layer_name = namespace.split("/")[-1]  # extract layer name from namespace

    rate_shape = (len(input_shapes), rate_dim)
    rate_name = f"{transform_name}_rate"  # _x{len(input_shapes)}'
    shared_layer_name = f"{'inv' if is_inverse else 'fwd'}_{transform_name}"

    quantization_values_path = f"shared/quantization/values/{rate_name}"
    mask_name = f"{rate_name}_quantization_mask"
    quantization_mask_path = f"{namespace}/{mask_name}"

    logstdevs_path = f"shared/quantization/logstdevs/{rate_name}"
    count_array_path = f"shared/quantization/counts/{rate_name}"

    def inner(params, value: ArrayLike, quantile, rate_embedding: ArrayLike, key: PRNGKey):
        """For a single source, computes a latent output from the concatenation of
        the rate embedding and the source value.
        All of these outputs will then be summed up and passed through a final layer.
        """

        if value.ndim == 0:
            value = value.reshape((1,))
        if rate_embedding.ndim == 0:
            rate_embedding = rate_embedding.reshape((1,))

        assert value.ndim == 1, f"In {transform_name}: {value.ndim} != 1: {value}"
        assert rate_embedding.ndim == 1

        inputs = flat_concat(value, rate_embedding, quantile)

        out = inner_activation(
            dense_mlp(
                inputs,
                inner_wsize,
                inner_outsize,
                depth=inner_depth,
                activation=inner_activation,
                initializer=initializer,
                bias_offset=bias_offset,
                key=key,
                param_f=partial(init_if_needed, params, base_path="shared"),
                name=f"NN/{shared_layer_name}/inner",
            )
        )

        assert out.shape == (inner_outsize,)

        return out

    def prepare(params: ParameterTree, nodelist: List[StackNode], key: PRNGKey):
        key0, key1 = jax.random.split(key, 2)
        n_nodes = len(nodelist)

        # --------- quantization
        # First, initializing quantization values for the rates (if not already done)
        # qnames is a list of names for the rate values available in this stack (1xuORf, ...)
        try:
            qvalues = params[quantization_values_path]
        except KeyError:
            if rate_dim <= 1:
                qvalues = jnp.linspace(-1, 1, len(quantization_names) * rate_dim).reshape(
                    (len(quantization_names), rate_dim)
                )
            else:
                qvalues = jax.random.normal(key0, (len(quantization_names), rate_dim))
            params[quantization_values_path] = qvalues
        # Now initialize logstdevs in the same way
        try:
            logstdevs = params[logstdevs_path]
        except KeyError:
            logstdevs = jnp.zeros((len(quantization_names), rate_dim)) - 3
            params[logstdevs_path] = logstdevs

        assert qvalues.shape == (len(quantization_names), rate_dim)

        init_if_needed(
            params,
            f"shared/{shared_layer_name}/residual_alpha",
            init_f=lambda: jnp.array(alpha_init),
        )
        init_if_needed(
            params, f"shared/{shared_layer_name}/residual_beta", init_f=lambda: jnp.array(beta_init)
        )

        if not is_inverse:  # forward node
            # We initialize quantization masks for these nodes.
            # Quantization masks are used to select which qvalues are accessible to each node.
            qmasks = [
                qz.get_quantization_mask(quantization_names, rate_name, node, stack)
                for node in nodelist
            ]
            for m in qmasks:
                assert m.shape == (len(input_shapes), len(quantization_names)), (
                    f"Invalid quantization mask shape {m.shape} for node in layer {layer_name}, expected {(len(input_shapes), len(quantization_names))}"
                )

            params.at(f"{quantization_mask_path}", np.array(qmasks), tags=[NON_GRAD_TAG])
            logger.debug(
                f"quantization mask for {layer_name}:\n{quantization_mask_str(quantization_names, qmasks)}"
            )
            try:
                params.at(
                    count_array_path,
                    np.array(qmasks).sum(axis=(0, 1)) + params.at(count_array_path),
                    overwrite=True,
                    tags=[NON_GRAD_TAG],
                )
            except KeyError:
                params.at(
                    count_array_path,
                    np.array(qmasks).sum(axis=(0, 1)),
                    tags=[NON_GRAD_TAG],
                )

            # And we also initialize the quantized rates
            params[f"{namespace}/{rate_name}"] = jax.random.uniform(key1, (n_nodes, *rate_shape))

        else:
            # For inverse nodes, we will use a view (a subtree of ArrayRef that mirrors the original subtree)
            # of both the quantized rates and the quantization masks of the corresponding forward nodes,
            # since they should be shared between the forward and inverse nodes.
            def get_fwd(node):
                fwd_node = node.get_forward_stacknode(stack)
                fwd_namespace = stack.get_layer_namespace(fwd_node.layer_number)
                return fwd_namespace, fwd_node.node_position_in_layer

            fwd_paths, fwd_loc = zip(*[get_fwd(node) for node in nodelist])

            # make view will create 2 subtrees of ArrayRef, one for the rates and one for the masks
            # that point to the same underlying data as the forward nodes
            make_view(params, namespace, fwd_paths, fwd_loc, leaves=[rate_name, mask_name])
            params.tag(f"{namespace}/{mask_name}", [NON_GRAD_TAG])

        # --------- quantile var
        add_quantile_var_ids(params, len(nodelist), len(input_shapes) + 1, namespace)

        fake_vals = [np.zeros(s) for s in input_shapes]

        apply(
            *fake_vals,
            quantiles=np.zeros(get_prev_num_quantile_vars(params) + 1),
            params=params,
            node_id=0,
            key=key1,
        )

    def outer(inner_out: ArrayLike, params, key: PRNGKey):
        return outer_activation(
            dense_mlp(
                inner_out,
                outer_wsize,
                1,
                depth=outer_depth,
                param_f=partial(init_if_needed, params, base_path="shared"),
                initializer=initializer,
                bias_offset=bias_offset,
                key=key,
                name=f"NN/{shared_layer_name}/outer",
                activation=inner_activation,
            )
        )

    def apply(
        *values: ArrayLike,
        quantiles: ArrayLike,
        params: ParameterTree,
        node_id: ArrayLike,
        key: PRNGKey,
    ) -> Tuple[ArrayLike, Dict]:
        k1, k2, k3 = jax.random.split(key, 3)

        qid = params[f"{namespace}/quantile_variable_id"][node_id]
        quantile = quantiles[qid]

        val = jnp.array(values)

        rates = params[f"{namespace}/{rate_name}"][node_id]

        try:
            assert val.shape == (len(input_shapes), *input_shapes[0])
            assert rates.shape == (len(input_shapes), rate_dim)
            assert quantile.shape == (len(input_shapes) + 1,)
        except AssertionError as e:
            logger.error("Shape assertion failed in transform_nn apply:")
            logger.error(f"  val.shape: {val.shape}")
            logger.error(f"  expected val.shape: {(len(input_shapes), *input_shapes[0])}")
            logger.error(f"  rates.shape: {rates.shape}")
            logger.error(f"  expected rates.shape: {(len(input_shapes), rate_dim)}")
            logger.error(f"  quantile.shape: {quantile.shape}")
            logger.error(f"  expected quantile.shape: {(len(input_shapes) + 1,)}")
            raise e

        qrates, qaux = qz.get_variational_quantized(
            rates,
            params,
            quantization_values_path,
            quantization_mask_path,
            logstdevs_path,
            node_id,
            k3,
        )

        # first we apply the inner head to all inputs and sum them:
        inner_keys = jax.random.split(k1, val.shape[0])
        inner_out = sum(
            inner(params, value=v, quantile=quantile[i], rate_embedding=r, key=k)
            for i, (v, r, k) in enumerate(zip(val, qrates, inner_keys))
        )
        inner_out = flat_concat(inner_out, quantile[len(input_shapes)])

        assert inner_out.shape == (inner_outsize + 1,)

        # then we apply a final outer layer to the summed output:
        ans = outer(inner_out, params, k2)

        # residual connection
        input_mean = jnp.mean(val, axis=0)
        alpha = params[f"shared/{shared_layer_name}/residual_alpha"]
        beta = params[f"shared/{shared_layer_name}/residual_beta"]
        # apply softmax normalization to alpha and beta
        alpha_norm = jnp.exp(alpha) / (jnp.exp(alpha) + jnp.exp(beta))
        beta_norm = jnp.exp(beta) / (jnp.exp(alpha) + jnp.exp(beta))
        final_output = alpha_norm * input_mean + beta_norm * ans

        return final_output, {
            "quantile": quantile,
            "rates": rates,
            "quantized_rates": qrates,
            "inner_output": inner_out,
            "outer_output": ans,
            "input_mean": input_mean,
            "alpha_norm": alpha_norm,
            "beta_norm": beta_norm,
            "is_inverse": is_inverse,
            "n_inputs": len(input_shapes),
            **qaux,
        }

    def commit(params: ParameterTree, nodelist: List[StackNode], **_):
        for node_id, node in enumerate(nodelist):
            rates = params[f"{namespace}/{rate_name}"][node_id]
            resolved_parameter_names = qz.get_quantized_part_names(
                rates,
                params,
                quantization_names,
                quantization_values_path,
                quantization_mask_path,
                node_id,
            )
            i_edges = node.get_incoming_edges(stack)
            assert len(i_edges) == len(resolved_parameter_names), (
                f"Number of incoming edges {len(i_edges)} does not match number of resolved rate names {len(resolved_parameter_names)}"
                f" for node {node} in namespace {namespace}"
            )
            for e, pname in zip(i_edges, resolved_parameter_names):
                e.content_embedding_names[rate_name] = (pname,)

    output_shape = [(1,)]

    return LayerInstance(prepare, apply, output_shape, commit=commit)


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                         --     ERN node     --


def sequestron_ERN(
    input_shapes: List[Tuple[int, ...]],
    n_outputs: int,
    stack: ComputeStack,
    layer_id: int,
    namespace: str,
    affinity_names: List[str],  # ordered list of available affinity names (case, csy4, etc..)
    affinity_dim: int = 1,
    wsize: int = 128,
    depth: int = 4,
    out_dim: int = 1,
    subtype: str = "5p",
    inner_activation_name: str = DEFAULT_ACTIVATION,
    outer_activation_name: str = DEFAULT_OUT_ACTIVATION,
    initializer_name: str = DEFAULT_INITIALIZER,
    bias_offset: float = 0.0,
    use_ern_layer_id: bool = False,
    max_ern_layers: int = 4,  # for one-hot encoding size
    alpha_init: float = 0.5,  # initial value for input residual
    beta_init: float = 0.5,  # initial value for network output
    **_,
) -> LayerInstance:
    inner_activation = ACTIVATION_FUNCTIONS[inner_activation_name]
    outer_activation = ACTIVATION_FUNCTIONS[outer_activation_name]
    initializer = INITIALIZERS[initializer_name]

    # ERN have 2 inputs of same size
    assert len(input_shapes) == 2
    assert input_shapes[0] == input_shapes[1], (
        f"ERN inputs must have same shape, got {input_shapes}"
    )
    assert n_outputs == 1, f"ERN only supports 1 output, got {n_outputs}"

    shared_layer_name = f"ERN_{subtype}"
    local_layer_name = namespace.split("/")[-1]  # extract layer name from namespace

    def MLP(
        neg: ArrayLike,
        pos: ArrayLike,
        affinity: ArrayLike,
        quantile: ArrayLike,
        param_f: Callable,
        key: PRNGKey,
        layer_id_onehot: ArrayLike = np.empty((0,)),
    ):
        if use_ern_layer_id:
            input_values = flat_concat(neg, pos, affinity, layer_id_onehot, quantile)
            assert layer_id_onehot.shape == (max_ern_layers,), (
                f"ERN layer_id_onehot should be of size {max_ern_layers}, got {len(layer_id_onehot)}"
            )
        else:
            input_values = flat_concat(neg, pos, affinity, quantile)

        res = dense_mlp(
            input_values,
            wsize,
            out_dim,
            depth,
            param_f=param_f,
            initializer=initializer,
            bias_offset=bias_offset,
            key=key,
            name=f"NN/ERN_{subtype}",
            activation=inner_activation,
        )

        # add residual connections
        neg_mean = jnp.mean(neg)
        pos_mean = jnp.mean(pos)
        alpha = param_f(f"{shared_layer_name}/residual_alpha", init_f=lambda: jnp.array(alpha_init))
        beta = param_f(f"{shared_layer_name}/residual_beta", init_f=lambda: jnp.array(beta_init))
        # apply softmax normalization to alpha and beta
        alpha = jnp.exp(alpha) / (jnp.exp(alpha) + jnp.exp(beta))
        beta = jnp.exp(beta) / (jnp.exp(alpha) + jnp.exp(beta))
        return alpha * (pos_mean - neg_mean) + beta * res

    def prepare(params: ParameterTree, nodelist: List[StackNode], key: PRNGKey):
        # --------- quantile var
        add_quantile_var_ids(params, len(nodelist), 1, local_layer_name)

        init_if_needed(
            params,
            f"shared/{shared_layer_name}/affinities",
            init_f=uniform_initializer(key, (len(affinity_names), affinity_dim)),
        )

        # for now the ERN node does'nt use the more complex quantization,
        # we just have one affinity value per ERN type (case, csy4, etc..)
        # and store one reference to the affinity value per node.

        # very important to use ArrayRef so that we don't copy the data which
        # would be catastrophic as it would create one new affinity value per node
        ref = ArrayRef(params.data)

        # store node layer ids if enabled
        seq_layer_ids = []

        for node in nodelist:
            # handle affinity value for this node
            comp_node = node.get(stack)
            seq_name = comp_node.extra["seq_name"]
            if seq_name not in affinity_names:
                raise ValueError(f"Unknown affinity name {seq_name}. Available: {affinity_names}")
            affinity_id = affinity_names.index(seq_name)
            ref.push_back(f"shared/{shared_layer_name}/affinities", affinity_id)

            # collect node layer ids if enabled
            if use_ern_layer_id:
                assert "layer_id" in comp_node.extra, (
                    f"ERN layer_id enabled but no layer_id found in extra dict of node {node}"
                )
                node_layer_id = comp_node.extra["layer_id"]
                assert 0 <= node_layer_id < max_ern_layers, (
                    f"Invalid ERN layer_id {node_layer_id} for node {node}, should be in [0, {max_ern_layers})"
                )
                seq_layer_ids.append(node_layer_id)

        params.at(f"{namespace}/affinity", ref, overwrite=None)

        # store node layer ids as a param array with non_grad tag if enabled
        if use_ern_layer_id:
            seqlayerid_arr = jnp.array(seq_layer_ids)
            assert seqlayerid_arr.shape == (len(nodelist),), (
                f"ERN node layer IDs should have shape ({(len(nodelist),)}), got {seqlayerid_arr.shape}"
            )
            params.at(
                f"{namespace}/node_layer_ids",
                seqlayerid_arr,
                tags=[NON_GRAD_TAG],
            )
            logger.debug(f"Node layer IDs for {local_layer_name}:\n{seqlayerid_arr}")

        # initialize MLP with dummy inputs
        # include dummy one-hot layer id if needed
        layer_id_onehot = jnp.zeros(max_ern_layers) if use_ern_layer_id else np.empty((0,))

        MLP(
            *[np.zeros(shape) for shape in input_shapes],
            affinity=np.zeros((affinity_dim,)),
            quantile=0,
            param_f=partial(init_if_needed, params, base_path="shared"),
            key=key,
            layer_id_onehot=layer_id_onehot,
        )

    def apply(
        *values: ArrayLike,
        quantiles: ArrayLike,
        params: ParameterTree,
        node_id: ArrayLike,
        key,
    ) -> Tuple[ArrayLike, Dict]:
        assert len(values) == len(input_shapes)

        affinity = params[f"{namespace}/affinity"][node_id]
        assert affinity.shape == (affinity_dim,)

        qid = params[f"{namespace}/quantile_variable_id"][node_id]

        # create one-hot encoded layer_id if enabled
        layer_id_onehot = jnp.empty((0,))  # default empty if not using layer_id
        if use_ern_layer_id:
            node_layer_id = params[f"{namespace}/node_layer_ids"][node_id]
            layer_id_onehot = jax.nn.one_hot(node_layer_id, max_ern_layers)

        result = MLP(
            *values,
            affinity=affinity,
            quantile=quantiles[qid],
            param_f=partial(get_param, params, base_path="shared"),
            key=key,
            layer_id_onehot=layer_id_onehot,
        )

        # calculate input difference for debug
        neg_val, pos_val = values
        input_diff = jnp.mean(pos_val) - jnp.mean(neg_val)

        aux_dict = {
            "affinity": affinity,
            "quantile": quantiles[qid],
            "node_layer_id": node_layer_id if use_ern_layer_id else None,
            "layer_id_onehot": layer_id_onehot,
            "neg_input": neg_val,
            "pos_input": pos_val,
            "input_diff": input_diff,
        }

        if use_ern_layer_id:
            aux_dict["node_layer_id"] = params[f"{namespace}/node_layer_ids"][node_id]

        return outer_activation(result), aux_dict

    output_shape = [(1,)]

    return LayerInstance(prepare, apply, output_shape)


##────────────────────────────────────────────────────────────────────────────}}}


### {{{                    --     output (fluorescence) node     --
def grouped_output(
    input_shapes: List[Tuple[int, ...]],
    n_outputs: int,  # unused
    stack: ComputeStack,
    layer_id: int,
    namespace: str,
    wsize: int = 64,
    depth: int = 4,
    bias_offset: float = 0.0,
    inner_activation_name: str = DEFAULT_ACTIVATION,
    outer_activation_name: str = DEFAULT_OUT_ACTIVATION,
    initializer_name: str = DEFAULT_INITIALIZER,
    **_,
):
    del n_outputs

    assert all(shape == input_shapes[0] for shape in input_shapes)
    inner_activation = ACTIVATION_FUNCTIONS[inner_activation_name]
    outer_activation = ACTIVATION_FUNCTIONS[outer_activation_name]
    initializer = INITIALIZERS[initializer_name]

    layer_name = namespace.split("/")[-1]  # extract layer name from namespace

    def MLP_head(x, q, rng_key, params):
        return dense_mlp(
            flat_concat(x, q),
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

    def prepare(params: ParameterTree, nodelist: List[StackNode], key: PRNGKey):
        # --------- quantile var
        add_quantile_var_ids(params, len(nodelist), len(input_shapes), layer_name)

        # --------- shared MLP layers
        MLP_head(x=np.zeros(input_shapes[0]), q=np.zeros((1,)), rng_key=key, params=params)

    def apply(
        *inputs: ArrayLike,
        quantiles: ArrayLike,
        params: ParameterTree,
        node_id: ArrayLike,
        key,
    ) -> Tuple[ArrayLike, Dict]:
        inputs_arr = jnp.array(inputs)

        assert len(inputs_arr) == len(input_shapes)

        qid = params[f"{namespace}/quantile_variable_id"][node_id]
        quantiles_for_node = quantiles[qid]
        res = vmap(
            partial(MLP_head, rng_key=key, params=params),
        )(inputs_arr, quantiles_for_node)

        pre = 0.5 * res + 0.5 * inputs_arr
        output = outer_activation(pre)

        return output, {
            "quantiles": quantiles_for_node,
            "mlp_outputs": res,
            "pre_activation": pre,
            "n_inputs": len(inputs_arr),
            "input_values": inputs_arr,
        }

    output_shape = [(1,)] * len(input_shapes)

    return LayerInstance(prepare, apply, output_shape)


##────────────────────────────────────────────────────────────────────────────}}}##

### {{{                    --     defaults & aliases     --
DEFAULT_AVAILABLE_TC_RATES = ["hEF1a"]

DEFAULT_AVAILABLE_TL_RATES = [
    "00_empty_tc",
    "1w_uORF",
    "1x_uORF",
    "2x_uORF",
    "3x_uORF",
    "4x_uORF",
    "5x_uORF",
    "6x_uORF",
    "8x_uORF",
    "9x_uORF",
    "10x_uORF",
    "11x_uORF",
    "12x_uORF",
]

ERN_DEFAULT_NEG_PARTS = ["CasE", "Csy4", "PgU"]
ERN_DEFAULT_POS_PARTS = [["CasE_rec"], ["Csy4_rec"], ["PgU_rec"]]
DEFAULT_AVAILABLE_5P_AFFINITIES = []
for i, positive_part in enumerate(ERN_DEFAULT_NEG_PARTS):
    for negative_part in ERN_DEFAULT_POS_PARTS[i]:
        DEFAULT_AVAILABLE_5P_AFFINITIES.append(f"ERN::{positive_part}#{negative_part}")


transcription = partial(
    transform_nn, transform_name="tc", quantization_names=DEFAULT_AVAILABLE_TC_RATES
)
translation = partial(
    transform_nn, transform_name="tl", quantization_names=DEFAULT_AVAILABLE_TL_RATES
)

inv_transcription = partial(
    transform_nn,
    transform_name="tc",
    is_inverse=True,
    quantization_names=DEFAULT_AVAILABLE_TC_RATES,
)
inv_translation = partial(
    transform_nn,
    transform_name="tl",
    is_inverse=True,
    quantization_names=DEFAULT_AVAILABLE_TL_RATES,
)

# source_with_pos used to be called "source_new" so we keep the alias for compatibility
source_new = source_with_pos
inv_source_new = inv_source_with_pos

ERN5p = partial(sequestron_ERN, subtype="5p", affinity_names=DEFAULT_AVAILABLE_5P_AFFINITIES)

##────────────────────────────────────────────────────────────────────────────}}}
