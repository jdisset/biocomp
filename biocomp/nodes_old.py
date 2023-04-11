from .library import PartsLibrary as PartsLibrary
import jax
from jax.tree_util import Partial as partial
import jax.numpy as jnp
from . import utils as ut

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                           --     utils     --
# ···············································································

def quantize(x, possible_values):
    if len(possible_values) == 0:
        return x
    if len(possible_values) == 1:
        return possible_values[0]
    else:
        return quantize_impl(x, possible_values)

def quantize_masked(x, possible_values, mask):
    if len(possible_values) == 0:
        return x
    if len(possible_values) == 1:
        return possible_values[0]
    else:
        return quantize_masked_impl(x, possible_values, mask)


def quantize_impl(x, arr):
    zero = x - jax.lax.stop_gradient(x) # for straight-through gradient
    return zero + jax.lax.stop_gradient(arr[jnp.argmin(jnp.abs(arr - x))])


def quantize_masked_impl(x, arr, mask):
    zero = x - jax.lax.stop_gradient(x) # for straight-through gradient
    dist = jnp.where(mask, jnp.abs(arr - x), jnp.inf)
    return zero + jax.lax.stop_gradient(arr[jnp.argmin(dist)])

@jax.custom_jvp
def round_to_int(x):
    zero = x - jax.lax.stop_gradient(x) # for straight-through gradient
    return zero + jax.lax.stop_gradient(jnp.round(x))

#TODO: 
# could also use some kind of softmin to make things 2 way differentiable 
# between both the quantizee and quantizers I guess. Not useful for now though
# as we'll only use it either with only one quantize value (when learning from data)
# or with fixed quantization values already (in compilation mode)


# old version without stop_gradient
# @partial(jax.custom_jvp, nondiff_argnums=(1,))
# def quantize_impl(x, arr):
# return arr[jnp.argmin(jnp.abs(arr - x))]
# # we define the derivative of the quantize function as if it was just the identity function (x -> x)
# @quantize_impl.defjvp
# def quantize_impl_jvp(_, x, x_tang):
# (x,) = x
# (x_dot,) = x_tang
# return x, x_dot




BC_EPSILON = 1e-9
BC_MAX_FLOAT = float('inf')

DEFAULT_COMPUTE_NODES_DICT = {}
DEFAULT_INVERSE_NODES_DICT = {}


def compnode(f):
    DEFAULT_COMPUTE_NODES_DICT[f.__name__] = f
    return f


def inv_compnode(fwd_name):
    def inv(f):
        DEFAULT_COMPUTE_NODES_DICT[f.__name__] = f
        DEFAULT_INVERSE_NODES_DICT[fwd_name] = f.__name__
        return f

    return inv


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     Default Compute Nodes     --
# ···············································································

# translation and transcription are the same, except for the parameters
def _transform(get_param, get_quantized, transform_name, **_):
    def apply(*values, rng_key):
        val = jnp.array(values)
        k0, k1 = jax.random.split(rng_key, 2)

        rate_name = f'{transform_name}_rate'
        deg_param_name = f'{transform_name}_deg'

        rates = get_quantized(
            rate_name,
            get_param(rate_name, init=ut.continuous_initializer(k0, val.shape)),
            mode='input_edges',
        )

        deg_rate = get_param(deg_param_name, init=ut.continuous_initializer(k1), shared=True)
        res = jnp.dot(rates, val) / deg_rate
        return res

    return apply


def _inverse_transform(get_param, get_quantized, transform_name, **_):
    def apply(value, rng_key):
        # inverse can only work if there's only one input edge
        assert value.shape == (), f'Expected scalar value, got {value.shape}'
        k0, k1 = jax.random.split(rng_key, 2)

        rate_name = f'{transform_name}_rate'
        deg_param_name = f'{transform_name}_deg'

        rate = get_quantized(
            rate_name,
            get_param(rate_name, init=ut.continuous_initializer(k0, (1,))),
            mode='input_edges',
        )[0]
        deg = get_param(deg_param_name, init=ut.continuous_initializer(k1), shared=True)

        res = value * deg / rate
        return res

    return apply


@compnode
def transcription(get_param, get_quantized, **_):
    return _transform(get_param, get_quantized, 'tc')


@inv_compnode(fwd_name='transcription')
def inv_transcription(get_param, get_quantized, **_):
    return _inverse_transform(get_param, get_quantized, 'tc')


@compnode
def translation(get_param, get_quantized, **_):
    return _transform(get_param, get_quantized, 'tl')


@inv_compnode(fwd_name='translation')
def inv_translation(get_param, get_quantized, **_):
    return _inverse_transform(get_param, get_quantized, 'tl')


@compnode
def sequestron_ERN(get_param, get_quantized, **_):
    def apply(neg, pos, **_):
        return jnp.maximum(pos - neg, 0.0)
    return apply

@compnode
def sequestron_ERN3p(get_param, get_quantized, **_):
    def apply(neg, pos, **_):
        return jnp.maximum(pos - neg, 0.0)
    return apply


@compnode
def ERN_with_affinity(get_param, get_quantized, seq_name, **_):
    def apply(neg, pos, rng_key, **_):
        param_name = f'{seq_name}::affinity'
        affinity = get_param(param_name, init=ut.continuous_initializer(rng_key), shared=True)
        return jnp.maximum(pos - neg * affinity, 0.0)

    return apply


@compnode
def sequestron_RCB(get_param, get_quantized, **_):
    EPSILON = 1e-12

    def apply(neg, pos, **_):
        return pos / (neg + pos + EPSILON)

    return apply


@compnode
def source(get_param, get_quantized, n_outputs, **_):
    def apply(inp, **_):
        return jnp.ones(n_outputs) * inp
    return apply


@compnode
def deadend(*_, **__):
    def apply(value, **_):
        return value

    return apply


@compnode
def output(*_, **__):
    def apply(*value, **_):
        return jnp.array(value)

    return apply


# inverse of source is just a pass-through
@inv_compnode(fwd_name='source')
def inv_source(*_, **__):
    def apply(value, **_):
        return value

    return apply


@compnode
def numeric(get_param, get_quantized, **_):
    def apply(rng_key):
        res = get_param("value", init=ut.continuous_initializer(rng_key))
        return res

    return apply


# inverse of numeric is just a pass-through
@inv_compnode(fwd_name='numeric')
def inv_numeric(*_, **__):
    def apply(value, **_):
        return value

    return apply

# 

# aggregations split a single input in ratios (defined by parameters)
@compnode
def aggregation(get_param, get_quantized, n_outputs, normalize=False, **kwargs):
    def apply(inp, quantile, rng_key):
        #TODO: use the quantile to compute some noise around the ratio
        if 'ratios' in kwargs:
            ratios = get_param(
                "ratios", overwrite_with=jnp.array(kwargs['ratios'], dtype=jnp.float32)
            )
        else:
            ratios = get_param("ratios", init=ut.continuous_initializer(rng_key, (n_outputs,)))

        assert ratios.shape == (n_outputs,)

        # ratios = ratios / jnp.maximum(jnp.sum(ratios), 1e-12)
        return jnp.array(ratios) * inp

    return apply


@inv_compnode(fwd_name='aggregation')
def inv_aggregation(get_param, get_quantized, original_output_len, original_output_slot, **_):
    assert original_output_len > 0
    assert original_output_slot < original_output_len

    def apply(inp, quantile, rng_key):
        ratios = get_param("ratios", init=ut.continuous_initializer(rng_key, (original_output_len,)))
        # ratios = ratios / jnp.maximum(jnp.sum(ratios), 1e-12)
        return inp / ratios[original_output_slot]
    return apply


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


DEFAULT_ACTIVATION = jax.nn.leaky_relu
DEFAULT_OUT_ACTIVATION = jax.nn.sigmoid


def dense_layer(input_values, output_size, get_param, key, name):
    input_size = 1 if input_values.shape == () else input_values.shape[0]
    w = get_param(f'{name}_w', init=ut.he_initializer(key, (input_size, output_size)), shared=True)
    b = get_param(f'{name}_b', init=lambda: jnp.zeros((output_size,)), shared=True)

    assert input_values.shape == (input_size,), f'In {name}: {input_values.shape} != {(input_size,)}'
    assert w.shape == (input_size, output_size), f'In {name}: {w.shape} != {(input_size, output_size)}'
    assert b.shape == (output_size,), f'In {name}: {b.shape} != {(output_size,)}'

    res = jnp.dot(input_values, w) + b
    return res.squeeze()


def dense_multilevel(
    input_values,
    hidden_s,
    output_s,
    depth,
    get_param,
    key,
    name,
    activation,
):
    res = input_values
    keys = jax.random.split(key, depth)
    for i in range(depth - 1):
        res = activation(dense_layer(res, hidden_s, get_param, keys[i], f'{name}_{i}'))

    return dense_layer(res, output_s, get_param, keys[-1], f'{name}_{depth - 1}')


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                       --     nn compute nodes     --
# ···············································································


def transform_nn(
    get_param,
    get_quantized,
    transform_name,
    outer_wsize=64,
    outer_depth=2,
    inner_wsize=32,
    inner_depth=2,
    inner_out=4,
    rate_dim=1,
    tr_namespace='',
    inner_activation=DEFAULT_ACTIVATION,
    outer_activation=DEFAULT_OUT_ACTIVATION,
    **_,
):
    def inner(value, rate_embeding, quantile, key):
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

        assert quantile.ndim == 1
        assert value.ndim == 1
        assert rate_embeding.ndim == 1

        inputs = ut.flat_concat(value, rate_embeding, quantile)

        out = inner_activation(
            dense_multilevel(
                inputs,
                inner_wsize,
                inner_out,
                depth=inner_depth,
                get_param=get_param,
                key=key,
                name=f'{tr_namespace}{transform_name}_inner',
                activation=inner_activation,
            )
        )

        assert out.shape == (inner_out,)

        return out

    def apply(*values, quantile, rng_key):

        k0, k1, k2 = jax.random.split(rng_key, 3)
        val = jnp.array(values)

        rate_name = f'{transform_name}_rate'
        rate_shape = (val.shape[0], rate_dim)
        rates = get_quantized(
            rate_name,
            get_param(rate_name, init=ut.continuous_initializer(k0, rate_shape)),
            mode='input_edges',
        )

        assert val.shape[0] == rates.shape[0]

        # first we apply a simple inner layer to all inputs and sum them:
        inner_out = jnp.sum(
            jax.vmap(inner, in_axes=(0, 0, None, None))(val, rates, quantile, k1), axis=0
        )

        inner_out = ut.flat_concat(inner_out, quantile)

        # then we apply a final outer layer to the summed output:
        return outer_activation(
            dense_multilevel(
                inner_out,
                outer_wsize,
                1,
                depth=outer_depth,
                get_param=get_param,
                key=k2,
                name=f'{tr_namespace}{transform_name}_outer',
                activation=inner_activation,
            )
        )

    return apply


def sequestron_ERN(
    get_param,
    get_quantized,
    seq_name,
    affinity_dim=1,
    wsize=128,
    depth=3,
    out_dim=1,
    subtype='5p',
    inner_activation=DEFAULT_ACTIVATION,
    outer_activation=DEFAULT_OUT_ACTIVATION,
    **_,
):
    def apply(neg, pos, quantile, rng_key, **_):
        param_name = f'{seq_name}::affinity_{subtype}'
        affinity = get_param(
            param_name, init=ut.continuous_initializer(rng_key, (affinity_dim,)), shared=True
        )
        res = dense_multilevel(
            ut.flat_concat(neg, pos, affinity, quantile),
            wsize,
            out_dim,
            depth,
            get_param,
            rng_key,
            f'ERN_{subtype}',
            activation=inner_activation,
        )
        return outer_activation(jnp.squeeze(res))

    return apply


transcription = partial(transform_nn, transform_name='tc')
translation = partial(transform_nn, transform_name='tl')
inv_transcription = partial(transform_nn, transform_name='tc', tr_namespace='inv_')
inv_translation = partial(transform_nn, transform_name='tl', tr_namespace='inv_')

ERN5p = partial(sequestron_ERN, subtype='5p')
ERN3p = partial(sequestron_ERN, subtype='3p')


def output(
    get_param,
    get_quantized,
    wsize=64,
    depth=3,
    inner_activation=DEFAULT_ACTIVATION,
    outer_activation=DEFAULT_OUT_ACTIVATION,
    **_,
):
    def apply(*value, quantile, rng_key, **_):

        value = jnp.array(value)
        assert value.shape[0] == quantile.shape[0]
        assert quantile.ndim == 1

        res = jnp.array(
            [
                dense_multilevel(
                    ut.flat_concat(x, q),
                    wsize,
                    1,
                    depth,
                    get_param,
                    rng_key,
                    'out',
                    activation=inner_activation,
                )
                for x, q in zip(value, quantile)
            ]
        )
        return outer_activation(res)

    return apply


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

