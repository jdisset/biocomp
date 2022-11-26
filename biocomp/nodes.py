from .library import PartsLibrary as PartsLibrary
import jax
from jax.tree_util import Partial as partial
import jax.numpy as jnp
from . import utils as ut

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                           --     utils     --
# ···············································································

DEFAULT_MIN_RATE = 0.0
DEFAULT_MAX_RATE = 10.0


def continuous_initializer(rng, shape=(), minval=DEFAULT_MIN_RATE, maxval=DEFAULT_MAX_RATE):
    def init():
        res = jax.random.uniform(
            key=rng, shape=shape, minval=minval, maxval=maxval, dtype=jnp.float32
        )
        return res

    return init


def glorot_initializer(rng, shape):
    def init():
        return jax.nn.initializers.glorot_normal()(rng, shape)

    return init


def quantize(x, possible_values):
    if len(possible_values) == 0:
        return x
    if len(possible_values) == 1:
        return possible_values[0]
    else:
        return quantize_impl(x, possible_values)


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def quantize_impl(x, arr):
    return arr[jnp.argmin(jnp.abs(arr - x))]


# we define the derivative of the quantize function as if it was just the identity function (x -> x)
@quantize_impl.defjvp
def quantize_impl_jvp(_, x, x_tang):
    (x,) = x
    (x_dot,) = x_tang
    return x, x_dot


@jax.custom_jvp
def round_to_int(x):
    return jnp.round(x)


# we define the derivative of the quantize function as if it was just the identity function (x -> x)
@round_to_int.defjvp
def round_to_int_jvp(x, x_tang):
    (x,) = x
    (x_dot,) = x_tang
    return x, x_dot


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
            get_param(rate_name, init=continuous_initializer(k0, val.shape)),
            mode='input_edges',
        )
        deg_rate = get_param(deg_param_name, init=continuous_initializer(k1), shared=True)
        # print(t'Calling {transform_name} with rates {rates} and deg_rate {deg_rate} and rng_key {rng_key}')
        res = jnp.dot(rates, val) / deg_rate
        # print(f'values: {values}')
        # print(f'val: {val}')
        # print(f'rates: {rates}')
        # print(f'deg_rate: {deg_rate}')
        # print(f'res: {res}')
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
            get_param(rate_name, init=continuous_initializer(k0, (1,))),
            mode='input_edges',
        )[0]
        deg = get_param(deg_param_name, init=continuous_initializer(k1), shared=True)

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
def ERN_with_affinity(get_param, get_quantized, seq_name, **_):
    def apply(neg, pos, rng_key, **_):
        param_name = f'{seq_name}::affinity'
        affinity = get_param(param_name, init=continuous_initializer(rng_key), shared=True)
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
        res = get_param("value", init=continuous_initializer(rng_key))
        return res

    return apply


# inverse of numeric is just a pass-through
@inv_compnode(fwd_name='numeric')
def inv_numeric(*_, **__):
    def apply(value, **_):
        return value

    return apply


# aggregations split a single input in ratios (defined by parameters)
@compnode
def aggregation(get_param, get_quantized, n_outputs, **kwargs):
    def apply(inp, rng_key):

        if 'ratios' in kwargs:
            ratios = get_param(
                "ratios", overwrite_with=jnp.array(kwargs['ratios'], dtype=jnp.float32)
            )
        else:
            ratios = get_param("ratios", init=continuous_initializer(rng_key, (n_outputs,)))

        assert ratios.shape == (n_outputs,)

        ratios = ratios / jnp.maximum(jnp.sum(ratios), 1e-12)
        return jnp.array(ratios) * inp

    return apply


@inv_compnode(fwd_name='aggregation')
def inv_aggregation(get_param, get_quantized, original_output_len, original_output_slot, **_):
    assert original_output_len > 0
    assert original_output_slot < original_output_len

    def apply(inp, rng_key):
        ratios = get_param("ratios", init=continuous_initializer(rng_key, (original_output_len,)))
        ratios = ratios / jnp.maximum(jnp.sum(ratios), 1e-12)
        return inp / ratios[original_output_slot]

    return apply


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                    --     nn based --
# ···············································································


def nn_dense(input_values, output_size, get_param, key, name):
    input_size = 1 if input_values.shape == () else input_values.shape[0]
    w = get_param(f'{name}_w', init=glorot_initializer(key, (input_size, output_size)), shared=True)
    b = get_param(f'{name}_b', init=lambda: jnp.zeros((output_size,)), shared=True)
    res = jnp.dot(input_values, w) + b
    return res.squeeze()


def nn_dense_multilevel(input_values, hidden_s, output_s, depth, get_param, key, name, activation):
    res = input_values
    keys = jax.random.split(key, depth)
    for i in range(depth - 1):
        res = activation(nn_dense(res, hidden_s, get_param, keys[i], f'{name}_{i}'))
    return nn_dense(res, output_s, get_param, keys[-1], f'{name}_{depth - 1}')


def transform_w_dense_layer(get_param, get_quantized, transform_name, wsize=128, depth=2, **_):
    app = _transform(get_param, get_quantized, transform_name, **_)

    def apply(*values, rng_key):
        k1, k2 = jax.random.split(rng_key, 2)
        res = app(*values, rng_key=k1)
        return jax.nn.relu(
            nn_dense_multilevel(res, wsize, 1, depth, get_param, k2, transform_name, jax.nn.relu)
        )

    return apply


def inv_transform_w_dense_layer(get_param, get_quantized, transform_name, wsize=128, depth=2, **_):
    def apply(value, rng_key):
        k0, k1, k2 = jax.random.split(rng_key, 3)

        rate_name = f'{transform_name}_rate'
        deg_param_name = f'{transform_name}_deg'

        rate = get_quantized(
            rate_name,
            get_param(rate_name, init=continuous_initializer(k0, (1,))),
            mode='input_edges',
        )[0]
        deg = get_param(deg_param_name, init=continuous_initializer(k1), shared=True)

        res = jax.nn.sigmoid(
            nn_dense_multilevel(
                jnp.array([value, rate, deg]).squeeze(),
                wsize,
                1,
                depth,
                get_param,
                k2,
                f'inv_{transform_name}',
                jax.nn.relu,
            )
        )
        return jax.nn.relu(res)

    return apply


def transcription_nn(get_param, get_quantized, **_):
    return transform_w_dense_layer(get_param, get_quantized, 'tc', **_)


def inverse_transcription_nn(get_param, get_quantized, **_):
    return inv_transform_w_dense_layer(get_param, get_quantized, 'tc', **_)


def translation_nn(get_param, get_quantized, **_):
    return transform_w_dense_layer(get_param, get_quantized, 'tl', **_)


def inverse_translation_nn(get_param, get_quantized, **_):
    return inv_transform_w_dense_layer(get_param, get_quantized, 'tl', **_)


def ERN_nn_multi(get_param, get_quantized, seq_name, **_):
    def apply(neg, pos, rng_key, **_):
        param_name = f'{seq_name}::affinity'
        affinity = get_param(param_name, init=continuous_initializer(rng_key), shared=True)
        res = nn_dense_multilevel(
            jnp.array([neg, pos, affinity]).squeeze(),
            256,
            1,
            3,
            get_param,
            rng_key,
            'ERN',
            jax.nn.relu,
        )
        return jax.nn.relu(jnp.squeeze(res))

    return apply


def output_nn(get_param, get_quantized, **_):
    def apply(*value, rng_key, **_):
        res = jnp.array(
            [
                nn_dense_multilevel(x.squeeze(), 128, 1, 2, get_param, rng_key, 'out', jax.nn.relu)
                for x in value
            ]
        )
        return jax.nn.relu(res)

    return apply


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
