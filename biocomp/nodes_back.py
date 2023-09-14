from .library import PartsLibrary as PartsLibrary
import jax
import inspect
import json
import importlib
from jax import vmap, jit, grad
from jax.tree_util import Partial as partial
import jax.numpy as jnp
import numpy as np
from . import utils as ut

from .utils import check

from .parameters import (
    get_param,
    set_param,
    init_param_if_needed,
    get_quantized,
    generate_quantization_masks,
    register_quantile_variable_ids,
    get_quantile_variables,
)

from tqdm import tqdm


# TODO:
# to make inits much faster, I should implement something like this:
# set_row(params, "inv_aggregation:original_output_slot", outslots)
# or, if I find the time to implement a proper paramtree
# params.set_row("inv_aggregation_original_output_slot", outslots)
#
# About the paramtree, I think I should implement it as a class
# and allow arbitrary namespaces - static being the only default.
# and maybe a tag system to allow for easy filtering:
# params.set_param("local", "inv_aggregation_original_output_slot", tags=[nograd])
#
# nogradparams = params.filter(tags=["nograd"])
# localparams = params.filter(namespaces=["local"])
# nogradlocalparams = params.filter(tags=["nograd"], namespaces=["local"])
# inverse logic (select all params that are not tagged "nograd"):
# gradparams = params.filter(tags=["nograd"], inverse=True)

### {{{                   --     general compute implementations    --

# When we create a compute layer, we pass the shape of all inputs as a list of tuples.
# Indeed, a node can have several inputs, and each input can have a different shape.
# The node constructor must then return the apply function, and the shape of the outputs.
# There can also be multiple outputs, each of which can have a different shape.
# we also pass the numper of outputs, which is useful for the source node.

# one question is whether we shouls allow for multiple outputs with different shapes.
# I don't think it's necessary for now, but at the same time it's not a big deal to allow it.
# So I guess yes, we should allow it. that means that we output a tuple of arrays


# Signatures:
# prepare (params, vnode, key)
# apply (values, quantiles, params, node_id, key)


def empty_prepare(*_, **__):
    pass


# input_shapes is a list of shape tuples, one for each input
def single_passthrough(input_shapes, *_, **__):

    assert len(input_shapes) == 1, f'Passthrough expects 1 input, got {len(input_shapes)}'

    def apply(value, **___):
        return value

    output_shapes = input_shapes

    return empty_prepare, apply, output_shapes


# source node is just an L2 plasmid, i.e an aggregation that has a fixed ratio of 1:1
# we make it a multi-output node so that it's compatible with the aggregation node but
# really we're just duplicating the input so we could also just use a passthrough node
# or skip the node altogether (for a future version with an optimizer)
def source(input_shapes, n_outputs, **_):
    assert len(input_shapes) == 1, f'A source node should have 1 input, got {len(input_shapes)}'

    def apply(value, *_, **__):
        return jnp.repeat(value, n_outputs, axis=0)

    output_shapes = input_shapes * n_outputs

    return empty_prepare, apply, output_shapes


# inverse of source is just a passthrough, as it's only inverted when only one output and one input
def inv_source(*args, **kwargs):
    return single_passthrough(*args, **kwargs)


def numeric(input_shapes, shape, **__):

    assert len(input_shapes) == 0

    def prepare(params, vnodelist, key, **_):
        maxid = max([vnode.node_id for vnode in vnodelist])
        for vnode in vnodelist:
            init_val = ut.continuous_initializer(key, shape)()
            set_param(
                params,
                "numeric:value",
                init_val,
                node_id=vnode.node_id,
                number_of_nodes_at_least=maxid + 1,
            )

    def apply(v, q, params, node_id, k):
        return get_param(params, "numeric:value", node_id=node_id)

    output_shapes = [shape]

    return prepare, apply, output_shapes


# inverse of numeric is just a pass-through
def inv_numeric(*args, **kwargs):
    return single_passthrough(*args, **kwargs)


def aggregation(input_shapes, n_outputs, stack=None, normalize=False, **_):

    assert len(input_shapes) == 1, f'Aggregation expects 1 input, got {len(input_shapes)}'

    pname = f"aggregation:ratios"

    max_agg_size = 0

    if stack is not None:
        for vnode in stack.get_all_nodes():
            cnode = vnode.get_compute_node()
            if cnode.type == 'aggregation':
                max_agg_size = max(max_agg_size, len(cnode.output_to))

    if max_agg_size < n_outputs:
        raise ValueError(f'Aggregation expects at most {max_agg_size} outputs, got {n_outputs}')

    def prepare(params, vnodelist, key, **_):
        maxid = max([vnode.node_id for vnode in vnodelist])
        for vnode in vnodelist:
            extra = vnode.get_compute_node().extra
            if 'ratios' in extra:
                assert len(extra['ratios']) == n_outputs
                ratio_v = jnp.array(extra['ratios'], dtype=jnp.float32)
            else:
                ratio_v = jax.random.uniform(key, (n_outputs,))
            # pad to max_outputs if necessary
            ratio_v = jnp.pad(ratio_v, (0, max_agg_size - n_outputs), constant_values=0.0)
            set_param(
                params, pname, ratio_v, node_id=vnode.node_id, number_of_nodes_at_least=maxid + 1
            )

    def apply(input, quantiles, params, node_id, key):
        assert input.shape == input_shapes[0], f'Invalid input shape {input.shape}'
        ratios = get_param(params, pname, node_id)[:n_outputs]
        if normalize:
            ratios = ratios / jnp.maximum(jnp.sum(ratios), 1e-12)
        return jnp.array(ratios) * input

    output_shape = input_shapes * n_outputs

    return prepare, apply, output_shape


def inv_aggregation(
    input_shapes,
    n_outputs,
    stack,
    # layer_id,
    normalize=False,
    **_,
):

    # an inverse aggregation node always has 1 input and 1 output
    assert len(input_shapes) == 1, f'inverse_Aggregation expects 1 input, got {len(input_shapes)}'
    assert n_outputs == 1, f'inverse_Aggregation expects 1 output, got {n_outputs}'

    EPSILON = 1e-12

    def prepare(params, vnodelist, **_):
        maxid = max([vnode.node_id for vnode in vnodelist])

        for vnode in vnodelist:
            cnode = vnode.get_compute_node()

            extra = cnode.extra
            assert 'original_output_len' in extra
            assert 'original_output_slot' in extra
            assert extra['original_output_len'] > 0
            assert extra['original_output_slot'] < extra['original_output_len']

            set_param(
                params,
                "inv_aggregation:original_output_slot",
                np.asarray(extra['original_output_slot']),
                node_id=vnode.node_id,
                base_path=ut.STATIC_PATH,
                number_of_nodes_at_least=maxid + 1,
            )

            if stack is not None:
                inv_vnode = vnode.get_inverse_vnode(stack)
                set_param(
                    params,
                    "inv_aggregation:inv_node_id",
                    np.asarray(inv_vnode.node_id),
                    node_id=vnode.node_id,
                    base_path=ut.STATIC_PATH,
                    number_of_nodes_at_least=maxid + 1,
                )

    def apply(inp, quantiles, params, node_id, key):
        original_output_slot = get_param(
            params, "inv_aggregation:original_output_slot", node_id, base_path=ut.STATIC_PATH
        ).astype(jnp.int32)

        if stack is not None:
            inv_id = get_param(
                params, "inv_aggregation:inv_node_id", node_id, base_path=ut.STATIC_PATH
            ).astype(jnp.int32)
            ratios = get_param(params, "aggregation:ratios", inv_id)
        else:
            ratios = jnp.ones((n_outputs,))

        if normalize:
            ratios = ratios / jnp.maximum(jnp.sum(ratios), EPSILON)

        ratio = ratios[original_output_slot]
        return jnp.where(ratio > EPSILON, inp / ratio, 0.0)

    output_shape = input_shapes
    return prepare, apply, output_shape


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                    --     neural utils     --


def leaky_relu(x, alpha=0.2):
    return jnp.where(x > 0, x, alpha * x)


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


ACTIVATION_FUNCTIONS = {
    'leaky_relu': leaky_relu,
    'sigmoid': sigmoid,
    'none': lambda x: x,
}

DEFAULT_ACTIVATION = 'leaky_relu'
DEFAULT_OUT_ACTIVATION = 'sigmoid'


def dense_layer(input_values, output_size, param_f, key, name):
    assert len(input_values.shape) == 1, f"In {name}: input_values should be a 1D array."
    input_size = 1 if input_values.shape == () else input_values.shape[0]

    w = param_f(f'{name}_w', init=ut.he_initializer(key, (input_size, output_size)))
    b = param_f(f'{name}_b', init=lambda: np.zeros((output_size,)))

    assert input_values.shape == (
        input_size,
    ), f'In {name}: {input_values.shape} != {(input_size,)}'
    assert w.shape == (
        input_size,
        output_size,
    ), f'In {name}: {w.shape} != {(input_size, output_size)}'
    assert b.shape == (output_size,), f'In {name}: {b.shape} != {(output_size,)}'

    assert w.shape == (
        input_size,
        output_size,
    ), f'In {name}: {w.shape} != {(input_size, output_size)}'

    res = jnp.dot(input_values, w) + b
    assert res.shape == (output_size,), f'In {name}: {res.shape} != {(output_size,)}'
    return res


def dense_multilevel(
    input_values,
    hidden_s,
    output_s,
    depth,
    param_f,
    key,
    name,
    activation,
):
    assert len(input_values.shape) == 1, f"In {name}: input_values should be a 1D array."
    assert (
        isinstance(depth, int) and depth >= 1
    ), f"In {name}: depth should be an integer greater than or equal to 1."
    assert (
        isinstance(hidden_s, int) and hidden_s > 0
    ), f"In {name}: hidden_s should be a positive integer."
    assert (
        isinstance(output_s, int) and output_s > 0
    ), f"In {name}: output_s should be a positive integer."

    res = input_values
    keys = jax.random.split(key, depth)
    for i in range(depth - 1):
        res = activation(dense_layer(res, hidden_s, param_f, keys[i], f'{name}_{i}'))
        assert res.shape == (hidden_s,), f'In {name}: {res.shape} != {(hidden_s,)}'

    res = dense_layer(res, output_s, param_f, keys[-1], f'{name}_{depth - 1}')
    assert res.shape == (output_s,), f'In {name}: {res.shape} != {(output_s,)}'
    return res


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                       --     neural nodes     --

# about the transcription or translation rates:
# At first I thought it's enough to have 1 continuous param
# for either transcription or translation. Having these params be continuous
# and quantized to the specific values for each part made some biological sense and
# allows the design phase to be more flexible with real numbers optim that
# then get quantized to the specific values.
# However it becomes annoying when we have several parts that affect the same parameter...
# Such an example is an ERN recognition site that affects translation rate.
#
# One solution is to have a sperate "dimension" (a separate parameter really)
# for each part that affects the same parameter. This way we don't have to learn a specific
# value for each combination, and we can resolve the inverse problem during design phase
# (given a value, find the parts that produce it)
#
# However, if we take the ERN recog site, even then that's pretty
# annoying in design mode because you can't simply change this param independently,
# it has to match with whatever ERN node is being used.
# I think a more general way to condition any node with some discreate signature of the
# handled transcription units is better, but probably outside of the scope of
# this first version.
# Another similar idea could be to output some kind of latent vector from certain
# nodes (like the ERN one for example) that could condition the nodes downstream, "signaling"
# to them that the translation should be dampened because it's a specific ERN.
# Right now the ERN node should be able to learn a little bit of how it has to modify
# its output


def transform_nn(
    input_shapes,
    n_outputs,
    stack,
    # layer_id,
    transform_name,
    outer_wsize=64,
    outer_depth=4,
    inner_wsize=64,
    inner_depth=3,
    inner_outsize=8,
    rate_dim=1,
    tr_namespace='',
    quantization_names: list[str] = None,  # ordered list. ex: ['1xuorf', '2xuorf', ...]
    inner_activation_name=DEFAULT_ACTIVATION,
    outer_activation_name=DEFAULT_OUT_ACTIVATION,
    **_,
):

    local_path = ut.NODE_PATH #/ f'layer_{layer_id}'
    assert quantization_names is not None, 'quantization_names should be provided'

    inner_activation = ACTIVATION_FUNCTIONS[inner_activation_name]
    outer_activation = ACTIVATION_FUNCTIONS[outer_activation_name]

    assert n_outputs == 1, f'NN transform only supports 1 output, got {n_outputs}'
    rate_name = f'{transform_name}_rate'

    # we separate between node_impl and shared_impl for performance during prepare
    # only node_impl has to be called for each node_id

    def __node_impl(*values, key, param_f, params, node_id):
        val = jnp.array(values)
        rshape = (val.shape[0], rate_dim)
        # first grab the continuous values for the rates, specific to this node
        individual_rate_name = f'{rate_name}_x{rshape[0]}'
        rates = param_f(
            individual_rate_name,
            init=ut.continuous_initializer(key, rshape),
            node_id=node_id,
            base_path=local_path,
        )
        # then quantize them
        qrates = get_quantized(rates, node_id=node_id, params=params, param_name=rate_name)
        return val, qrates

    def __shared_impl(val, rates, quantile, key, param_f):

        assert val.shape[0] == rates.shape[0]
        k1, k2 = jax.random.split(key, 2)

        def inner(value, rate_embeding, key):
            """For a single source, computes a latent output from the concatenation of
            the rate embedding and the source value.
            All of these outputs will then be summed up and passed through a final layer.
            """

            # TODO idea: to give more flexibility, we could add the index of the
            # value as this might allow clever padding of the sum
            # we'd then need to make sure that the index is unique for each
            # while, probably, being random (to avoid any "preferred" order)

            if value.ndim == 0:
                value = value.reshape((1,))
            if rate_embeding.ndim == 0:
                rate_embeding = rate_embeding.reshape((1,))

            assert value.ndim == 1, f'In {transform_name}: {value.ndim} != 1: {value}'
            assert rate_embeding.ndim == 1

            inputs = ut.flat_concat(value, rate_embeding, quantile)

            out = inner_activation(
                dense_multilevel(
                    inputs,
                    inner_wsize,
                    inner_outsize,
                    depth=inner_depth,
                    activation=inner_activation,
                    key=key,
                    param_f=partial(param_f, node_id=0, base_path=ut.SHARED_PATH),
                    name=f'{tr_namespace}{transform_name}_inner',
                )
            )

            assert out.shape == (inner_outsize,)

            return out

        # first we apply the inner stack to all inputs and sum them:

        inner_keys = jax.random.split(k1, val.shape[0])
        inner_out = sum(inner(v, r, k) for v, r, k in zip(val, rates, inner_keys))
        inner_out = ut.flat_concat(inner_out, quantile)

        assert inner_out.shape == (inner_outsize + 1,)

        # then we apply a final outer layer to the summed output:
        return outer_activation(
            dense_multilevel(
                inner_out,
                outer_wsize,
                1,
                depth=outer_depth,
                param_f=partial(param_f, node_id=0, base_path=ut.SHARED_PATH),
                key=k2,
                name=f'{tr_namespace}{transform_name}_outer',
                activation=inner_activation,
            )
        )

    def prepare(params, vnodelist, key):
        # during prepare, we call _impl with dummy inputs + a param_function that
        # creates the parameters on the fly if they don't exist yet

        # qnames is a list of names for the rate values available in this stack (1xuORf, ...)
        # they all get an initial value that the rates will be quantized to
        # it's a shared parameter
        init = ut.continuous_initializer(key, (len(quantization_names), rate_dim))
        init_param_if_needed(params, rate_name, init=init, base_path=ut.QVALS_PATH, node_id=0)

        maxid = max([vnode.node_id for vnode in vnodelist])

        for local_id, vnode in enumerate(vnodelist):
            register_quantile_variable_ids(params, vnode, stack)
            generate_quantization_masks(
                quantization_names,
                params,
                rate_name,
                vnode,
                number_of_nodes_at_least=maxid + 1,
            )
            key, _ = jax.random.split(key)
            val, rates = __node_impl(
                *[np.zeros(shape) for shape in input_shapes],
                key=key,
                param_f=partial(init_param_if_needed, params, number_of_nodes_at_least=maxid + 1),
                params=params,
                node_id=vnode.node_id,
            )

        __shared_impl(
            val,
            rates,
            quantile=0,
            key=key,
            param_f=partial(init_param_if_needed, params),
        )

    def apply(*values, quantiles, params, node_id, key):
        assert len(values) == len(input_shapes)
        param_f = partial(get_param, params)  # read-only
        val, rates = __node_impl(*values, key=key, param_f=param_f, params=params, node_id=node_id)
        quantile = get_quantile_variables(params, node_id, quantiles, 1)
        return __shared_impl(val, rates, quantile, key, param_f)

    output_shape = [(1,)]

    return prepare, apply, output_shape


def sequestron_ERN(
    input_shapes,
    n_outputs,
    stack,
    # layer_id,
    affinity_dim=1,
    wsize=128,
    depth=4,
    out_dim=1,
    subtype='5p',
    affinity_names=None,
    inner_activation_name=DEFAULT_ACTIVATION,
    outer_activation_name=DEFAULT_OUT_ACTIVATION,
    **_,
):

    assert affinity_names is not None, 'affinity_names must be specified'

    inner_activation = ACTIVATION_FUNCTIONS[inner_activation_name]
    outer_activation = ACTIVATION_FUNCTIONS[outer_activation_name]

    # ERN have 2 inputs of same size
    assert len(input_shapes) == 2
    assert (
        input_shapes[0] == input_shapes[1]
    ), f'ERN inputs must have same shape, got {input_shapes}'
    assert n_outputs == 1, f'ERN only supports 1 output, got {n_outputs}'

    ERN_AFFINITY_ID_NAME = f'ERN{subtype}_affinity_id'
    ERN_AFFINITY_VALUE_PARAM = f'ERN{subtype}_affinity_value'

    def __impl(neg, pos, quantile, rng_key, param_f, affinity_id):
        affinity = param_f(
            ERN_AFFINITY_VALUE_PARAM,
            init=ut.continuous_initializer(rng_key, (affinity_dim,)),
            base_path=ut.SHARED_PATH,
            node_id=affinity_id,
            number_of_nodes_at_least=len(affinity_names),
        )

        res = dense_multilevel(
            ut.flat_concat(neg, pos, affinity, quantile),
            wsize,
            out_dim,
            depth,
            param_f=partial(param_f, node_id=0, base_path=ut.SHARED_PATH),
            key=rng_key,
            name=f'ERN_{subtype}',
            activation=inner_activation,
        )

        return outer_activation(jnp.squeeze(res))

    def prepare(params, vnodelist, key):
        maxid = max([vnode.node_id for vnode in vnodelist])
        for vnode in vnodelist:
            register_quantile_variable_ids(params, vnode, stack)
            # we need to know which affinity value to use for this node
            assert 'seq_name' in vnode.get_compute_node().extra
            seq_name = vnode.get_compute_node().extra['seq_name']  # ex: 'CasE5p'
            if seq_name not in affinity_names:
                raise ValueError(f'Unknown affinity name {seq_name}. Available: {affinity_names}')
            affinity_id = affinity_names.index(seq_name)

            # affinity_id is the index at which the affinity value is stored
            # in the array of all affinity values. We store this index so that
            # we can retrieve the correct value during apply (vectorized on all node_ids)
            set_param(
                params,
                ERN_AFFINITY_ID_NAME,
                affinity_id,
                node_id=vnode.node_id,
                number_of_nodes_at_least=maxid + 1,
                base_path=ut.STATIC_PATH,
            )

        __impl(
            *[np.zeros(shape) for shape in input_shapes],
            quantile=0,
            rng_key=key,
            param_f=partial(init_param_if_needed, params),
            affinity_id=affinity_id,
        )

    def apply(*values, quantiles, params, node_id, key):
        assert len(values) == len(input_shapes)
        affinity_id = get_param(
            params, ERN_AFFINITY_ID_NAME, node_id=node_id, base_path=ut.STATIC_PATH
        )
        quantile = get_quantile_variables(params, node_id, quantiles, 1)
        return __impl(
            *values,
            quantile=quantile,
            rng_key=key,
            param_f=partial(get_param, params),
            affinity_id=affinity_id,
        )

    output_shape = [(1,)]

    return prepare, apply, output_shape


from rich import print as pprint


def grouped_output(
    input_shapes,
    n_outputs,
    stack,
    # layer_id,
    wsize=64,
    depth=4,
    inner_activation_name=DEFAULT_ACTIVATION,
    outer_activation_name=DEFAULT_OUT_ACTIVATION,
    **_,
):

    inner_activation = ACTIVATION_FUNCTIONS[inner_activation_name]
    outer_activation = ACTIVATION_FUNCTIONS[outer_activation_name]

    def __impl(*inputs, quantiles, rng_key, param_f, **_):
        assert quantiles.shape == (len(inputs),)
        assert len(inputs) == len(input_shapes)
        # grouped output is actually simply the same output function aplied
        # to each input, with a different quantile value
        res = vmap(
            lambda x, q: dense_multilevel(
                ut.flat_concat(x, q),
                wsize,
                1,
                depth,
                param_f=partial(param_f, node_id=0, base_path=ut.SHARED_PATH),
                key=rng_key,
                name='grouped_output',
                activation=inner_activation,
            )
        )(jnp.asarray(inputs), quantiles)
        return outer_activation(res)

    def prepare(params, vnodelist, key):
        for vnode in vnodelist:
            register_quantile_variable_ids(params, vnode, stack)

        __impl(
            *[np.zeros(shape) for shape in input_shapes],
            quantiles=np.zeros((len(input_shapes),)),
            rng_key=key,
            param_f=partial(init_param_if_needed, params),
        )

    def apply(*inputs, quantiles, params, node_id, key):
        q = get_quantile_variables(params, node_id, quantiles, len(inputs))
        return __impl(*inputs, quantiles=q, rng_key=key, param_f=partial(get_param, params))

    output_shape = [(1,)] * len(input_shapes)

    return prepare, apply, output_shape


DEFAULT_AVAILABLE_TC_RATES = ['hEF1a']

DEFAULT_AVAILABLE_TL_RATES = [
    '00_empty_tc',
    '1w_uORF',
    '1x_uORF',
    '2x_uORF',
    '3x_uORF',
    '4x_uORF',
    '5x_uORF',
    '6x_uORF',
    '8x_uORF',
    '9x_uORF',
    '10x_uORF',
    '11x_uORF',
    '12x_uORF',
]

ERN_DEFAULT_NEG_PARTS = ['CasE', 'Csy4', 'PgU']
ERN_DEFAULT_POS_PARTS = [['CasE_rec'], ['Csy4_rec'], ['PgU_rec']]
DEFAULT_AVAILABLE_5P_AFFINITIES = []
for i, positive_part in enumerate(ERN_DEFAULT_NEG_PARTS):
    for negative_part in ERN_DEFAULT_POS_PARTS[i]:
        DEFAULT_AVAILABLE_5P_AFFINITIES.append(f'ERN::{positive_part}#{negative_part}')


transcription = partial(
    transform_nn, transform_name='tc', quantization_names=DEFAULT_AVAILABLE_TC_RATES
)
translation = partial(
    transform_nn, transform_name='tl', quantization_names=DEFAULT_AVAILABLE_TL_RATES
)
inv_transcription = partial(
    transform_nn,
    transform_name='tc',
    tr_namespace='inv_',
    quantization_names=DEFAULT_AVAILABLE_TC_RATES,
)
inv_translation = partial(
    transform_nn,
    transform_name='tl',
    tr_namespace='inv_',
    quantization_names=DEFAULT_AVAILABLE_TL_RATES,
)

ERN5p = partial(sequestron_ERN, subtype='5p', affinity_names=DEFAULT_AVAILABLE_5P_AFFINITIES)


##────────────────────────────────────────────────────────────────────────────}}}
