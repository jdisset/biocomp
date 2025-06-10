from __future__ import annotations
from .library import PartsLibrary as PartsLibrary
import jax
from jax import vmap
from jax.tree_util import Partial as partial
import jax.numpy as jnp
import numpy as np
import jax.nn

from .utils import get_logger
from .jaxutils import flat_concat
from . import quantization as qz

from .parameters import ArrayRef, ParameterTree, init_if_needed, make_view, get_param

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .compute import ComputeNode, ComputeStack

from jax.typing import ArrayLike
from typing import Callable, Tuple, List
from dataclasses import dataclass

PRNGKey = ArrayLike

logger = get_logger(__name__)


# =========================== Utils ===========================
### {{{                 --     misc    --
def generate_layer_name(stack, layer_id, name):
    if stack is None:
        return f"{layer_id}/{name}"
    else:
        n_nodes = len(stack.layers[layer_id].nodes)
        n_layers = len(stack.layers)
        return f"{layer_id}/{name} ({n_nodes})"


def quantization_mask_str(names, mask) -> str:
    col_width = max(len(name) for name in names)
    result = " " * 5
    for i, name in enumerate(names):
        result += f"{name:^{col_width}} "
    result += "\n"
    for i, row in enumerate(mask):
        result += f"{i:<4}|"
        for val in row[0]:
            result += f"{'X' if val else ' ':^{col_width}}|"
        result += "\n"
    return result


@dataclass
class LayerInstance:
    prepare: Callable
    apply: Callable
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


def get_prev_num_quantile_vars(params: ParameterTree):
    try:
        return params["global/number_of_quantile_variables"]
    except:
        return 0


def add_quantile_var_ids(params: ParameterTree, num_nodes: int, num_per_node, layer_name: str):
    """
    Updates:
        - global/number_of_quantile_variables
        - local/{layer_name}/quantile_variable_id -> id array of shape (num_nodes, num_per_node)
    """

    prev_num_quantile_vars = get_prev_num_quantile_vars(params)
    new_num_quantile_vars = prev_num_quantile_vars + num_nodes * num_per_node
    quantile_var_ids = jnp.arange(prev_num_quantile_vars, new_num_quantile_vars).reshape(
        (num_nodes, num_per_node)
    )
    params.at(
        f"local/{layer_name}/quantile_variable_id",
        quantile_var_ids,
        tags=["non_grad"],
        overwrite=None,
    )
    params.at(
        "global/number_of_quantile_variables",
        new_num_quantile_vars,
        tags=["non_grad"],
        overwrite=True,
    )


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                    --     neural utils     --


def uniform_initializer(rng, shape=(), minval=0, maxval=1):
    def init():
        return jax.random.uniform(
            key=rng, shape=shape, minval=minval, maxval=maxval, dtype=jnp.float32
        )

    return init


def glorot_normal(rng, shape):
    def init():
        return jax.nn.initializers.glorot_normal()(rng, shape)

    return init


def glorot_uniform(rng, shape):
    def init():
        return jax.nn.initializers.glorot_uniform()(rng, shape)

    return init


def he_normal(rng, shape):
    def init():
        return jax.nn.initializers.he_normal()(rng, shape)

    return init


def he_uniform(rng, shape):
    def init():
        return jax.nn.initializers.he_uniform()(rng, shape)

    return init


def leaky_relu(x, alpha=0.2):
    return jax.nn.leaky_relu(x, negative_slope=alpha)


def sigmoid(x):
    return jax.nn.sigmoid(x)


ACTIVATION_FUNCTIONS = {
    "leaky_relu": leaky_relu,
    "relu": jax.nn.relu,
    "elu": jax.nn.elu,
    "selu": jax.nn.selu,
    "tanh": jax.nn.tanh,
    "gelu": jax.nn.gelu,
    "softplus": jax.nn.softplus,
    "sigmoid": sigmoid,
    "none": lambda x: x,
}

INITIALIZERS = {
    "uniform": uniform_initializer,
    "glorot_normal": glorot_normal,
    "glorot_uniform": glorot_uniform,
    "he_normal": he_normal,
    "he_uniform": he_uniform,
}

DEFAULT_ACTIVATION = "leaky_relu"
DEFAULT_OUT_ACTIVATION = "sigmoid"
DEFAULT_INITIALIZER = "he"


def dense_layer(
    input_values: ArrayLike,
    output_size: ArrayLike,
    param_f: Callable,
    initializer: Callable,
    bias_offset,
    key: PRNGKey,
    name: str,
):
    assert len(input_values.shape) == 1, f"In {name}: input_values should be a 1D array."
    input_size = 1 if input_values.shape == () else input_values.shape[0]

    w = param_f(f"{name}/w", init_f=initializer(key, (input_size, output_size)))
    b = param_f(f"{name}/b", init_f=lambda: np.zeros((output_size,)) + bias_offset)

    assert input_values.shape == (input_size,), (
        f"In {name}: {input_values.shape} != {(input_size,)}"
    )
    assert w.shape == (
        input_size,
        output_size,
    ), f"In {name}: {w.shape} != {(input_size, output_size)}"
    assert b.shape == (output_size,), f"In {name}: {b.shape} != {(output_size,)}"

    assert w.shape == (
        input_size,
        output_size,
    ), f"In {name}: {w.shape} != {(input_size, output_size)}"

    res = jnp.dot(input_values, w) + b
    assert res.shape == (output_size,), f"In {name}: {res.shape} != {(output_size,)}"
    return res


def dense_multilevel(
    input_values: ArrayLike,
    hidden_s: int,
    output_s: int,
    depth: int,
    param_f: Callable[[str, Callable], ArrayLike],
    initializer: Callable,
    bias_offset,
    key: PRNGKey,
    name: str,
    activation: Callable[[ArrayLike], ArrayLike],
):
    assert len(input_values.shape) == 1, f"In {name}: input_values should be a 1D array."
    assert isinstance(depth, int) and depth >= 1, (
        f"In {name}: depth should be an integer greater than or equal to 1."
    )
    assert isinstance(hidden_s, int) and hidden_s > 0, (
        f"In {name}: hidden_s should be a positive integer."
    )
    assert isinstance(output_s, int) and output_s > 0, (
        f"In {name}: output_s should be a positive integer."
    )

    res = input_values
    keys = jax.random.split(key, depth)
    for i in range(depth - 1):
        res = activation(
            dense_layer(
                res,
                hidden_s,
                param_f,
                initializer,
                bias_offset,
                keys[i],
                f"{name}/l{i}",
            )
        )
        assert res.shape == (hidden_s,), f"In {name}: {res.shape} != {(hidden_s,)}"

    res = dense_layer(
        res, output_s, param_f, initializer, bias_offset, keys[-1], f"{name}/l{depth - 1}"
    )
    assert res.shape == (output_s,), f"In {name}: {res.shape} != {(output_s,)}"
    return res


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

    def apply(value: ArrayLike, **___) -> ArrayLike:
        return value

    output_shapes = input_shapes

    return LayerInstance(empty_prepare, apply, output_shapes)


# source node is just an L2 plasmid, i.e an aggregation that has a fixed ratio of 1:1
# we make it a multi-output node so that it's compatible with the aggregation node but
# really we're just duplicating the input so we could also just use a passthrough node
# or skip the node altogether (for a future version with an optimizer)


# For now, input_shapes will always be [(1,)]
def source(input_shapes: List[Tuple[int]], n_outputs: int, **_) -> LayerInstance:
    assert len(input_shapes) == 1, f"A source node should have 1 input, got {len(input_shapes)}"

    def apply(value: ArrayLike, *_, **__) -> ArrayLike:
        return jnp.repeat(value, n_outputs, axis=0)

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
    max_L1s: int = 5,
    hidden_s=64,
    depth=3,
    inner_activation_name: str = DEFAULT_ACTIVATION,
    outer_activation_name: str = DEFAULT_OUT_ACTIVATION,
    initializer_name: str = DEFAULT_INITIALIZER,
    bias_offset=0.0,
    **_,
) -> LayerInstance:
    assert len(input_shapes) == 1, f"A source node should have 1 input, got {len(input_shapes)}"

    inner_activation = ACTIVATION_FUNCTIONS[inner_activation_name]
    outer_activation = ACTIVATION_FUNCTIONS[outer_activation_name]
    initializer = INITIALIZERS[initializer_name]

    local_layer_name = generate_layer_name(stack, layer_id, f"source{n_outputs}x")
    namespace = f"local/{local_layer_name}"
    pname = "shapes"

    def prepare(params: ParameterTree, nodelist: List[ComputeNode], key, **_):
        add_quantile_var_ids(params, len(nodelist), len(input_shapes), local_layer_name)
        params[f"{namespace}/{pname}"] = input_shapes
        MLP_head(np.zeros((2 + len(input_shapes),)), params, key)

    def MLP_head(vals, params, key):
        return dense_multilevel(
            vals,
            hidden_s,
            1,
            depth=depth,
            activation=inner_activation,
            initializer=initializer,
            bias_offset=bias_offset,
            key=key,
            param_f=partial(init_if_needed, params, base_path="shared"),
            name="NN/source",
        )

    def apply(
        value: ArrayLike,
        quantiles: ArrayLike,
        params: ParameterTree,
        node_id: ArrayLike,
        key,
    ) -> ArrayLike:
        qid = params[f"{namespace}/quantile_variable_id"][node_id]
        quantile = quantiles[qid]
        ans = jax.vmap(
            lambda position: MLP_head(flat_concat(value, position, quantile), params, key)
        )(np.arange(max_L1s)[:n_outputs] / max_L1s)
        res = ans + jnp.broadcast_to(value, ans.shape)
        return outer_activation(res)

    output_shapes = list(input_shapes) * n_outputs

    return LayerInstance(prepare, apply, output_shapes)


def inv_source_with_pos(
    input_shapes: List[Tuple[int]],
    n_outputs: int,
    stack: ComputeStack,
    layer_id: int,
    max_L1s: int = 5,
    hidden_s=64,
    depth=3,
    inner_activation_name: str = DEFAULT_ACTIVATION,
    outer_activation_name: str = DEFAULT_OUT_ACTIVATION,
    initializer_name: str = DEFAULT_INITIALIZER,
    bias_offset=0.0,
    **_,
) -> LayerInstance:
    local_layer_name = generate_layer_name(stack, layer_id, "inverse_source")
    namespace = f"local/{local_layer_name}"
    pname = "shapes"

    inner_activation = ACTIVATION_FUNCTIONS[inner_activation_name]
    outer_activation = ACTIVATION_FUNCTIONS[outer_activation_name]
    initializer = INITIALIZERS[initializer_name]

    def prepare(params: ParameterTree, nodelist: List[ComputeNode], key, **_):
        add_quantile_var_ids(params, len(nodelist), len(input_shapes), local_layer_name)
        if stack is not None:
            ref = ArrayRef(params.data)
            for node in nodelist:
                extra = node.get_compute_node("extra")
                assert extra["original_output_slot"] < extra["original_output_len"]
                original_slot = extra["original_output_slot"]

                fwd_node = node.get_inverse_node(stack)
                fwd_layer, fwd_loc = fwd_node.get_layer_and_local_id(stack)
                fwd_n_output = stack.layers[fwd_layer].get_n_outputs()
                fwd_namespace = (
                    f"local/{generate_layer_name(stack, fwd_layer, f'source_{fwd_n_output}x')}"
                )
                ref.push_back(f"{fwd_namespace}/{pname}", (fwd_loc, original_slot))

            params[f"{namespace}/{pname}"] = ref
            # params.at(f'{namespace}/{pname}', ref, overwrite=None)
        MLP_head(np.zeros((2 + len(input_shapes),)), params, key)

    def MLP_head(vals, params, key, hidden_s=hidden_s, depth=depth):
        return dense_multilevel(
            vals,
            hidden_s,
            1,
            depth=depth,
            activation=inner_activation,
            initializer=initializer,
            bias_offset=bias_offset,
            key=key,
            param_f=partial(init_if_needed, params, base_path="shared"),
            name="NN/source",
        )

    def apply(
        value: ArrayLike,
        quantiles: ArrayLike,
        params: ParameterTree,
        node_id: ArrayLike,
        key,
    ) -> ArrayLike:
        qid = params[f"{namespace}/quantile_variable_id"][node_id]
        quantile = quantiles[qid]
        ans = jax.vmap(
            lambda position: MLP_head(flat_concat(value, position, quantile), params, key)
        )(np.arange(max_L1s)[:n_outputs] / max_L1s)
        return outer_activation(ans + value.reshape(ans.shape))

    output_shapes = list(input_shapes) * n_outputs

    return LayerInstance(prepare, apply, output_shapes)


def bias(
    input_shapes: List[Tuple[int]],
    n_outputs: int,
    layer_id: int,
    stack,
    shape: Tuple[int] = (1,),
    **_,
) -> LayerInstance:
    assert n_outputs == 1, f"Bias node should have 1 output, got {n_outputs}"
    assert len(input_shapes) == 0

    local_layer_name = generate_layer_name(stack, layer_id, "bias")
    namespace = f"local/{local_layer_name}"

    def prepare(params: ParameterTree, nodelist: List[ComputeNode], key, **_):
        params[f"{namespace}/value"] = jax.random.uniform(key, (len(nodelist), *shape))

    def apply(*_, params: ParameterTree, node_id: ArrayLike, **__) -> ArrayLike:
        return params[f"{namespace}/value"][node_id]

    def commit(params: ParameterTree, nodelist: List[ComputeNode], **_):
        for i, n in enumerate(nodelist):
            extra = n.get_compute_node("extra") or {}
            extra["bias_value"] = params[f"{namespace}/value"][i]
            n.set_compute_node_column("extra", extra)

    output_shapes = [shape]

    return LayerInstance(prepare, apply, output_shapes, commit=commit)


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                   --     aggregation node   --


def aggregation(
    input_shapes: List[Tuple[int]],
    n_outputs: int,
    layer_id: int,
    stack: ComputeStack = None,
    **_,
) -> LayerInstance:
    assert len(input_shapes) == 1, f"Aggregation expects 1 input, got {len(input_shapes)}"

    local_layer_name = generate_layer_name(stack, layer_id, f"aggregation_{n_outputs}x")
    namespace = f"local/{local_layer_name}"
    pname = f"ratios"

    def prepare(params: ParameterTree, nodelist: List[ComputeNode], key: PRNGKey, **_):
        ratios = []
        for i, node in enumerate(nodelist):
            extra = node.get_compute_node("extra")
            if "ratios" in extra:
                assert len(extra["ratios"]) == n_outputs
                ratio_v = jnp.array(extra["ratios"], dtype=jnp.float32)
            else:
                ratio_v = jax.random.uniform(key, (n_outputs,))
            # pad to max_outputs if necessary
            ratios.append(ratio_v)

        ratios = jnp.stack(ratios)
        assert ratios.shape == (len(nodelist), n_outputs), f"Invalid ratio shape {ratios.shape}"
        params[f"{namespace}/{pname}"] = ratios

        def normalize_ratios_cb(params: ParameterTree, **__):
            current_ratios = params[f"{namespace}/{pname}"]
            assert current_ratios.shape == (len(nodelist), n_outputs)
            max_ratios = jnp.maximum(jnp.max(current_ratios, axis=1), 1e-9)
            normed_ratios = current_ratios / max_ratios[:, None]
            return params.tree_set_at(f"{namespace}/{pname}", jnp.clip(normed_ratios, 0, 1))

        stack.register_post_process(normalize_ratios_cb)

    def apply(
        input: ArrayLike,
        quantiles: ArrayLike,
        params: ParameterTree,
        node_id: ArrayLike,
        key: PRNGKey,
    ) -> ArrayLike:
        assert input.shape == input_shapes[0], f"Invalid input shape {input.shape}"
        ratios = params[f"{namespace}/{pname}"][node_id][:n_outputs]
        return jnp.abs(jnp.array(ratios)) * input

    def commit(params: ParameterTree, nodelist: List[ComputeNode], **_):
        for i, n in enumerate(nodelist):
            extra = n.get_compute_node("extra") or {}
            extra["ratios"] = params[f"{namespace}/{pname}"][i]
            n.set_compute_node_column("extra", extra)

    output_shape = input_shapes * n_outputs

    return LayerInstance(prepare, apply, output_shape, commit)


def inv_aggregation(
    input_shapes: List[Tuple[int]],
    n_outputs: int,
    stack: ComputeStack,
    layer_id: int,
    **_,
) -> LayerInstance:
    # an inverse aggregation node always has 1 input and 1 output
    assert len(input_shapes) == 1, f"inverse_Aggregation expects 1 input, got {len(input_shapes)}"
    assert n_outputs == 1, f"inverse_Aggregation expects 1 output, got {n_outputs}"

    local_layer_name = generate_layer_name(stack, layer_id, f"inverse_aggregation")
    namespace = f"local/{local_layer_name}"

    def prepare(params: ParameterTree, nodelist: List[ComputeNode], **_):
        if stack is not None:
            ref = ArrayRef(params.data)
            for node in nodelist:
                extra = node.get_compute_node("extra")
                assert extra["original_output_slot"] < extra["original_output_len"]
                original_slot = extra["original_output_slot"]

                fwd_node = node.get_inverse_node(stack)
                fwd_layer, fwd_loc = fwd_node.get_layer_and_local_id(stack)
                fwd_n_output = stack.layers[fwd_layer].get_n_outputs()
                fwd_namespace = (
                    f"local/{generate_layer_name(stack, fwd_layer, f'aggregation_{fwd_n_output}x')}"
                )
                ref.push_back(f"{fwd_namespace}/ratios", (fwd_loc, original_slot))

            params.at(f"{namespace}/ratios", ref, overwrite=None)

    EPSILON = 1e-9

    def apply(
        input: ArrayLike,
        quantiles: ArrayLike,
        params: ParameterTree,
        node_id: ArrayLike,
        key,
    ) -> ArrayLike:
        if stack is not None:
            ratio = jnp.abs(params[f"{namespace}/ratios"][node_id])
        else:
            ratio = jnp.ones((1,))

        return input / jnp.maximum(ratio, EPSILON)

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
    **_,
):
    logger.debug("Initializing transform_nn node:")
    logger.debug(f"  transform_name: {transform_name}")
    logger.debug(f"  input_shapes: {input_shapes}")
    logger.debug(f"  n_outputs: {n_outputs}")
    logger.debug(f"  is_inverse: {is_inverse}")
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

    def make_layer_name(l_id, is_inv):
        return generate_layer_name(stack, l_id, f"{'inverse_' if is_inv else ''}{transform_name}")

    rate_shape = (len(input_shapes), rate_dim)
    rate_name = f"{transform_name}_rate"  # _x{len(input_shapes)}'
    shared_layer_name = f"{'inv' if is_inverse else 'fwd'}_{transform_name}"
    layer_name = make_layer_name(layer_id, is_inverse)

    quantization_values_path = f"shared/quantization/values/{rate_name}"
    mask_name = f"{rate_name}_quantization_mask"
    quantization_mask_path = f"local/{layer_name}/{mask_name}"

    logstdevs_path = f"shared/quantization/logstdevs/{rate_name}"
    count_array_path = f"shared/quantization/counts/{rate_name}"

    def inner(params, value: ArrayLike, quantile, rate_embedding: ArrayLike, key: PRNGKey):
        """For a single source, computes a latent output from the concatenation of
        the rate embedding and the source value.
        All of these outputs will then be summed up and passed through a final layer.
        """
        logger.debug("Inner function inputs:")
        logger.debug(f"  value shape: {value.shape if hasattr(value, 'shape') else 'scalar'}")
        logger.debug(
            f"  rate_embedding shape: {rate_embedding.shape if hasattr(rate_embedding, 'shape') else 'scalar'}"
        )
        logger.debug(
            f"  quantile shape: {quantile.shape if hasattr(quantile, 'shape') else 'scalar'}"
        )

        if value.ndim == 0:
            value = value.reshape((1,))
        if rate_embedding.ndim == 0:
            rate_embedding = rate_embedding.reshape((1,))

        assert value.ndim == 1, f"In {transform_name}: {value.ndim} != 1: {value}"
        assert rate_embedding.ndim == 1

        inputs = flat_concat(value, rate_embedding, quantile)
        logger.debug(f"  concatenated inputs shape: {inputs.shape}")

        out = inner_activation(
            dense_multilevel(
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

    def prepare(params: ParameterTree, nodelist: List[ComputeNode], key: PRNGKey):
        key0, key1 = jax.random.split(key, 2)
        n_nodes = len(nodelist)

        # --------- quantization
        # First, initializing quantization values for the rates (if not already done)
        # qnames is a list of names for the rate values available in this stack (1xuORf, ...)
        try:
            qvalues = 1000 * params[quantization_values_path]
        except KeyError:
            qvalues = jax.random.normal(key0, (len(quantization_names), rate_dim)) / 1000
            params[quantization_values_path] = qvalues
        # Now initialize logstdevs in the same way
        try:
            logstdevs = params[logstdevs_path]
        except KeyError:
            logstdevs = jnp.zeros((len(quantization_names), rate_dim)) - 4
            params[logstdevs_path] = logstdevs

        assert qvalues.shape == (len(quantization_names), rate_dim)

        if not is_inverse:  # forward node
            # We initialize quantization masks for these nodes.
            # Quantization masks are used to select which qvalues are accessible to each node.
            qmasks = [
                qz.get_quantization_mask(
                    quantization_names, rate_name, node, masks_per_node=len(input_shapes)
                )
                for node in nodelist
            ]
            params.at(f"{quantization_mask_path}", np.array(qmasks), tags=["non_grad"])
            logger.debug(
                f"quantization mask for {layer_name}:\n{quantization_mask_str(quantization_names, qmasks)}"
            )
            try:
                params.at(
                    count_array_path,
                    np.array(qmasks).sum(axis=(0, 1)) + params.at(count_array_path),
                    overwrite=True,
                    tags=["non_grad"],
                )
            except KeyError:
                params.at(
                    count_array_path,
                    np.array(qmasks).sum(axis=(0, 1)),
                    tags=["non_grad"],
                )

            # And we also initialize the quantized rates
            params[f"local/{layer_name}/{rate_name}"] = jax.random.uniform(
                key1, (n_nodes, *rate_shape)
            )

        else:
            # For inverse nodes, we will use a view (a subtree of ArrayRef that mirrors the original subtree)
            # of both the quantized rates and the quantization masks of the corresponding forward nodes,
            # since they should be shared between the forward and inverse nodes.
            def get_fwd(node):
                fwd_node = node.get_inverse_node(stack)
                fwd_layer_id, fwd_loc = fwd_node.get_layer_and_local_id(stack)
                fwd_layer_name = make_layer_name(fwd_layer_id, is_inv=False)
                return f"local/{fwd_layer_name}", fwd_loc

            fwd_paths, fwd_loc = zip(*[get_fwd(node) for node in nodelist])

            # make view will create 2 subtrees of ArrayRef, one for the rates and one for the masks
            # that point to the same underlying data as the forward nodes
            make_view(
                params, f"local/{layer_name}", fwd_paths, fwd_loc, leaves=[rate_name, mask_name]
            )
            params.tag(f"local/{layer_name}/{mask_name}", ["non_grad"])

        # --------- quantile var
        add_quantile_var_ids(params, len(nodelist), len(input_shapes) + 1, layer_name)

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
            dense_multilevel(
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
    ):
        logger.debug(f"Apply function inputs:")
        logger.debug(
            f"  values shapes: {[v.shape if hasattr(v, 'shape') else 'scalar' for v in values]}"
        )
        logger.debug(
            f"  quantiles shape: {quantiles.shape if hasattr(quantiles, 'shape') else 'scalar'}"
        )
        logger.debug(f"  node_id: {node_id}")

        k1, k2, k3 = jax.random.split(key, 3)

        qid = params[f"local/{layer_name}/quantile_variable_id"][node_id]
        quantile = quantiles[qid]
        logger.debug(f"  quantile shape after indexing: {quantile.shape}")

        val = jnp.array(values)
        logger.debug(f"  values array shape: {val.shape}")

        rates = params[f"local/{layer_name}/{rate_name}"][node_id]
        logger.debug(f"  rates shape: {rates.shape}")

        try:
            assert val.shape == (len(input_shapes), *input_shapes[0])
            assert rates.shape == (len(input_shapes), rate_dim)
            assert quantile.shape == (len(input_shapes) + 1,)
        except AssertionError as e:
            logger.error(f"Shape assertion failed in transform_nn apply:")
            logger.error(f"  val.shape: {val.shape}")
            logger.error(f"  expected val.shape: {(len(input_shapes), *input_shapes[0])}")
            logger.error(f"  rates.shape: {rates.shape}")
            logger.error(f"  expected rates.shape: {(len(input_shapes), rate_dim)}")
            logger.error(f"  quantile.shape: {quantile.shape}")
            logger.error(f"  expected quantile.shape: {(len(input_shapes) + 1,)}")
            raise e
        qrates = qz.get_variational_quantized(
            rates,
            params,
            quantization_values_path,
            quantization_mask_path,
            logstdevs_path,
            node_id,
            k3,
        )

        # first we apply the inner stack to all inputs and sum them:
        inner_keys = jax.random.split(k1, val.shape[0])
        inner_out = sum(
            inner(params, value=v, quantile=quantile[i], rate_embedding=r, key=k)
            for i, (v, r, k) in enumerate(zip(val, qrates, inner_keys))
        )
        inner_out = flat_concat(inner_out, quantile[len(input_shapes)])

        assert inner_out.shape == (inner_outsize + 1,)

        # then we apply a final outer layer to the summed output:
        ans = outer(inner_out, params, k2)

        # return ans + val.reshape(ans.shape)

        return jnp.sum(ans + val, axis=0)  # skip connection

    def commit(params: ParameterTree, nodelist: List[ComputeNode], **_):
        for node_id, node in enumerate(nodelist):
            rates = params[f"local/{layer_name}/{rate_name}"][node_id]
            resolved_parameter_names = qz.get_quantized_rate_names(
                rates,
                params,
                quantization_names,
                quantization_values_path,
                quantization_mask_path,
                node_id,
            )
            extra = node.get_compute_node("extra") or {}
            extra["resolved_parameter_names"] = resolved_parameter_names
            node.set_compute_node_column("extra", extra)
            qz.collapse_quantized_parameter(node, rate_name, resolved_parameter_names)

    output_shape = [(1,)]

    return LayerInstance(prepare, apply, output_shape, commit=commit)


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                         --     ERN node     --


def sequestron_ERN(
    input_shapes: List[Tuple[int, ...]],
    n_outputs: int,
    stack: ComputeStack,
    layer_id: int,
    affinity_names: List[str],
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
    local_layer_name = generate_layer_name(stack, layer_id, f"ERN_{subtype}")

    def MLP(
        neg: ArrayLike,
        pos: ArrayLike,
        affinity: ArrayLike,
        quantile: ArrayLike,
        param_f: Callable,
        key: PRNGKey,
        layer_id_onehot: ArrayLike,
    ):
        res = dense_multilevel(
            flat_concat(neg, pos, affinity, layer_id_onehot, quantile),
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

        return outer_activation(res)

    def prepare(params: ParameterTree, nodelist: List[ComputeNode], key: PRNGKey):
        # --------- quantile var
        add_quantile_var_ids(params, len(nodelist), 1, local_layer_name)

        init_if_needed(
            params,
            f"shared/{shared_layer_name}/affinities",
            init_f=uniform_initializer(key, (len(affinity_names), affinity_dim)),
        )

        # store affinity references
        ref = ArrayRef(params.data)

        # store node layer ids if enabled
        seq_layer_ids = []

        for node in nodelist:
            # handle affinity value for this node
            extra = node.get_compute_node("extra")
            seq_name = extra["seq_name"]  # ex: 'CasE5p'
            if seq_name not in affinity_names:
                raise ValueError(f"Unknown affinity name {seq_name}. Available: {affinity_names}")
            affinity_id = affinity_names.index(seq_name)
            ref.push_back(f"shared/{shared_layer_name}/affinities", affinity_id)

            # collect node layer ids if enabled
            if use_ern_layer_id:
                # get layer_id from node extra info, default to 0 if not present
                node_layer_id = min(extra.get("layer_id", 0), max_ern_layers - 1)
                seq_layer_ids.append(node_layer_id)
                logger.debug(f"Node {node} layer ID: {node_layer_id}")

        params.at(f"local/{local_layer_name}/affinity", ref, overwrite=None)

        # store node layer ids as a param array with non_grad tag if enabled
        if use_ern_layer_id:
            params.at(
                f"local/{local_layer_name}/node_layer_ids",
                jnp.array(seq_layer_ids),
                tags=["non_grad"],
            )

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
    ):
        assert len(values) == len(input_shapes)

        affinity = params[f"local/{local_layer_name}/affinity"][node_id]
        assert affinity.shape == (affinity_dim,)

        qid = params[f"local/{local_layer_name}/quantile_variable_id"][node_id]

        # create one-hot encoded layer_id if enabled
        layer_id_onehot = jnp.empty((0,))
        if use_ern_layer_id:
            node_layer_id = params[f"local/{local_layer_name}/node_layer_ids"][node_id]
            layer_id_onehot = jnp.zeros(max_ern_layers)
            layer_id_onehot = layer_id_onehot.at[node_layer_id].set(1.0)

        return MLP(
            *values,
            affinity=affinity,
            quantile=quantiles[qid],
            param_f=partial(get_param, params, base_path="shared"),
            key=key,
            layer_id_onehot=layer_id_onehot,
        )

    output_shape = [(1,)]

    return LayerInstance(prepare, apply, output_shape)


##────────────────────────────────────────────────────────────────────────────}}}


### {{{                    --     output (fluorescence) node     --
def grouped_output(
    input_shapes: List[Tuple[int, ...]],
    n_outputs: int,  # unused
    stack: ComputeStack,
    layer_id: int,
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

    layer_name = generate_layer_name(stack, layer_id, "grouped_output")

    def MLP_head(x, q, rng_key, params):
        return dense_multilevel(
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

    def prepare(params: ParameterTree, nodelist: List[ComputeNode], key: PRNGKey):
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
    ):
        inputs = jnp.asarray(inputs)

        assert len(inputs) == len(input_shapes)
        quantiles_per_node = len(input_shapes)

        qid = params[f"local/{layer_name}/quantile_variable_id"][node_id]
        res = vmap(
            partial(MLP_head, rng_key=key, params=params),
        )(inputs, quantiles[qid])

        ans = outer_activation(res)
        return ans + inputs

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

ERN5p = partial(sequestron_ERN, subtype="5p", affinity_names=DEFAULT_AVAILABLE_5P_AFFINITIES)

##────────────────────────────────────────────────────────────────────────────}}}
