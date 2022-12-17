import jax
from jax.tree_util import Partial as partial
from . import utils as ut
from . import nodes as nodes
import jax.numpy as jnp

DEFAULT_ACTIVATION = jax.nn.leaky_relu


def dense_layer(input_values, output_size, get_param, key, name):
    input_size = 1 if input_values.shape == () else input_values.shape[0]
    w = get_param(
        f'{name}_w', init=ut.glorot_initializer(key, (input_size, output_size)), shared=True
    )
    b = get_param(f'{name}_b', init=lambda: jnp.zeros((output_size,)), shared=True)
    res = jnp.dot(input_values, w) + b
    return res.squeeze()


def dense_multilevel(input_values, hidden_s, output_s, depth, get_param, key, name, activation):
    res = input_values
    keys = jax.random.split(key, depth)
    for i in range(depth - 1):
        res = activation(dense_layer(res, hidden_s, get_param, keys[i], f'{name}_{i}'))
    return dense_layer(res, output_s, get_param, keys[-1], f'{name}_{depth - 1}')


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                       --     compute nodes     --
# ···············································································


def transform_nn(
    get_param,
    get_quantized,
    transform_name,
    outer_wsize=64,
    outer_depth=2,
    outer_activation=DEFAULT_ACTIVATION,
    inner_wsize=32,
    inner_depth=2,
    inner_out=4,
    inner_activation=DEFAULT_ACTIVATION,
    rate_dim=1,
    tr_namespace='',
    **_,
):
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
        inputs = jnp.concatenate([value, rate_embeding], axis=-1)

        return inner_activation(
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

    def apply(*values, rng_key):

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
        inner_out = jnp.sum(jax.vmap(inner, in_axes=(0, 0, None))(val, rates, k1), axis=0)

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
                activation=outer_activation,
            )
        )

    return apply


def sequestron_ERN(
    get_param, get_quantized, seq_name, affinity_dim=1, wsize=128, depth=3, out_dim=1, subtype='5p', **_
):
    def apply(neg, pos, rng_key, **_):
        param_name = f'{seq_name}::affinity_{subtype}'
        affinity = get_param(
            param_name, init=ut.continuous_initializer(rng_key, (affinity_dim,)), shared=True
        )
        res = dense_multilevel(
            jnp.concatenate([jnp.array([neg, pos]), affinity], axis=-1),
            wsize,
            out_dim,
            depth,
            get_param,
            rng_key,
            f'ERN_{subtype}',
            DEFAULT_ACTIVATION,
        )
        return DEFAULT_ACTIVATION(jnp.squeeze(res))
    return apply


transcription = partial(transform_nn, transform_name='tc')
translation = partial(transform_nn, transform_name='tl')
inv_transcription = partial(transform_nn, transform_name='tc', tr_namespace='inv_')
inv_translation = partial(transform_nn, transform_name='tl', tr_namespace='inv_')

ERN5p = partial(sequestron_ERN, subtype='5p')
ERN3p = partial(sequestron_ERN, subtype='3p')

def output(get_param, get_quantized, **_):
    def apply(*value, rng_key, **_):
        res = jnp.array(
            [
                dense_multilevel(
                    x.squeeze(), 128, 1, 2, get_param, rng_key, 'out', DEFAULT_ACTIVATION
                )
                for x in value
            ]
        )
        return DEFAULT_ACTIVATION(res)

    return apply


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
