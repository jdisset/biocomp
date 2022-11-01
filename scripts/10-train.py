from jax.config import config as jax_config

# jax_config.update("jax_debug_nans", True)

import biocomp as bc
import biocomp.compute as bcc
import numpy as np
from functools import partial
import biocomp.utils as bu
import scriptutils as ut
import jax
import jax.numpy as jnp
import random

random.seed()

lib = ut.load_lib()
xp = ut.load_xp('20221012A_massCtrls', lib)


# models = xp.get_models()

# models = xp.get_models(node_remap=config['node_remap'])
# # model is a dict. Let's get one
# model = models.items().__iter__().__next__()[1]
# cg = model.network.compute_graph
# p = model.init(jax.random.PRNGKey(cfg['rng_key']))
# inp = jnp.array([1.0, 2.0])
# model.apply(p, inp, rng_key=jax.random.PRNGKey(0))
# model.collect_all_results(p, inp, rng_key=jax.random.PRNGKey(0))


# lowered = jax.jit(model.apply).lower(p, inp, rng_key=jax.random.PRNGKey(0))
# compiled = lowered.compile()
# print(compiled.as_text())
# # print nb of lines
# print(len(compiled.as_text().splitlines()))
# compiled.cost_analysis()
# compiled.memory_analysis()
# %timeit compiled(p, inp, rng_key=jax.random.PRNGKey(0))

# bc.train.train_xp(xp, cfg, wandb_project="biocomp_20221012A_massCtrls")
# a should contain integers increasing 10 by 10 in the first dimension, and 1 in the second dimension.
# shape (30, 2)

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                        --     helpers   --
# ···············································································
def glorot_initializer(rng_key, shape):
    def init():
        return jax.nn.initializers.glorot_normal()(rng_key, shape)

    return init


params = {}
get_p = partial(
    bcc.get_param,
    params,
    node_id=1,
)
get_q = partial(
    bcc.get_quantized,
    params,
    quantize_fun=bcc.quantize,
    node_id=1,
)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


def nn_dense(input_values, hidden_size, output_size, get_param, key, name):
    k1, k2 = jax.random.split(key, 2)
    input_size = input_values.shape[-1]
    w1 = get_param(
        f'{name}_w1', init=glorot_initializer(k1, (input_size, hidden_size)), shared=True
    )
    b1 = get_param(f'{name}_b1', init=lambda: jnp.zeros((hidden_size,)), shared=True)
    w2 = get_param(
        f'{name}_w2', init=glorot_initializer(k2, (hidden_size, output_size)), shared=True
    )
    b2 = get_param(f'{name}_b2', init=lambda: jnp.zeros((output_size,)), shared=True)
    return jnp.dot(jax.nn.sigmoid(jnp.dot(input_values, w1) + b1), w2) + b2


# y = nn_dense(jnp.array([1.0]), 2, get_p, jax.random.PRNGKey(1), 'test')


def nn_dense_multilevel(input_values, hidden_size, output_size, depth, get_param, key, name):
    # similar to n_dense, but instead of a single layer, we have a stack of layers (depth)
    # each layer is a dense layer, but the input to each layer is the output of the previous layer
    res = input_values
    keys = jax.random.split(key, depth)
    for i in range(depth - 1):
        res = nn_dense(res, hidden_size, hidden_size, get_param, keys[i], f'{name}_{i}')
    return nn_dense(res, hidden_size, output_size, get_param, keys[-1], f'{name}_{depth - 1}')


# def transform_w_dense_layer(get_param, get_quantized, wsize, transform_name, deg_param_name, **_):
# app = bcc._transform(get_param, get_quantized, transform_name, deg_param_name, **_)
# def apply(*values, rng_key):
# k1, k2 = jax.random.split(rng_key, 2)
# res = app(*values, rng_key=k1)
# return nn_dense(res, wsize, get_param, k2, transform_name)
# return apply


@bcc.compnode
def ERN_nn_multi(get_param, get_quantized, seq_name, **_):
    def apply(neg, pos, rng_key, **_):
        param_name = f'{seq_name}::affinity'
        affinity = get_param(param_name, init=continuous_initializer(rng_key), shared=True)
        res = nn_dense_multilevel(
            jnp.array([neg, pos]).squeeze(), 32, 1, 2, get_param, rng_key, seq_name
        )
        return jnp.squeeze(res) + affinity

    return apply


##


def transform_hill(get_param, get_quantized, transform_name, **_):
    def apply(*values, rng_key):
        keys = jax.random.split(rng_key, 4)

        # "$$ [out] = V_{specie} (\\frac{[value]}{[value] + K_{A_{specie}}})^{n_{specie}} / (\\mu_{specie}) $$",

        val = jnp.array(values)

        V_name = f'{transform_name}_rate'
        V_raw = get_quantized(
            V_name,
            get_param(
                V_name,
                init=bcc.continuous_initializer(keys[0], (1,)),
                clip_to=(bcc.BC_EPSILON, None),
            ),
            mode='input_edges',
        )

        # we compute the actual V_species as a weighted average of the V_species_raw
        # (which depend on input edges (i.e. which promoter or uORF is used))
        V = jnp.average(V_raw, weights=val+bcc.BC_EPSILON)

        K_A = get_param(
            f'{transform_name}_K',
            init=bcc.continuous_initializer(keys[1]),
            shared=True,
            clip_to=(0, None),
        )
        n = get_param(
            f'{transform_name}_n',
            init=bcc.continuous_initializer(keys[2]),
            shared=True,
            clip_to=(bcc.BC_EPSILON, None),
        )
        mu = get_param(
            f'{transform_name}_mu',
            init=bcc.continuous_initializer(keys[3]),
            shared=True,
            clip_to=(bcc.BC_EPSILON, None),
        )

        value = jnp.sum(val)

        return V * (value / (value + K_A)) ** n / mu

    return apply


def inverse_transform_hill(get_param, get_quantized, transform_name, **_):
    def apply(value, rng_key):
        keys = jax.random.split(rng_key, 4)
        assert value.shape == ()

        # [out]^* = \\frac{\\alpha K_{A_{specie}}}{1 - \\alpha}
        # where $\\alpha$ is:\n",
        # \\alpha = \\sqrt[n_{specie}]{\\frac{[in]}{V_{specie}} (\\mu_{specie})}

        V_name = f'{transform_name}_rate'
        V = get_quantized(
            V_name,
            get_param(
                V_name,
                init=bcc.continuous_initializer(keys[0], (1,)),
                clip_to=(bcc.BC_EPSILON, None),
            ),
            mode='input_edges',
        )[0]

        K_A = get_param(
            f'{transform_name}_K',
            init=bcc.continuous_initializer(keys[1]),
            shared=True,
            clip_to=(0, None),
        )
        n = get_param(
            f'{transform_name}_n',
            init=bcc.continuous_initializer(keys[2]),
            shared=True,
            clip_to=(bcc.BC_EPSILON, None),
        )
        mu = get_param(
            f'{transform_name}_mu',
            init=bcc.continuous_initializer(keys[3]),
            shared=True,
            clip_to=(bcc.BC_EPSILON, None),
        )

        alpha = jnp.power(value / V * mu, 1 / n)
        return alpha * K_A / (1 - alpha)

    return apply


# TODO: add bounds to the parameters

# def get_qu(_, v, **__):
# return v
# f = transform_hill(get_p, get_qu, 'test')
# inv_f = inverse_transform_hill(get_p, get_qu, 'test')
# y = f(0.093000, rng_key=jax.random.PRNGKey(1))
# inv_f(y, rng_key=jax.random.PRNGKey(1))


@bcc.compnode
def transcription_hill(get_param, get_quantized, **_):
    return transform_hill(get_param, get_quantized, 'tc', **_)


@bcc.compnode
def translation_hill(get_param, get_quantized, **_):
    return transform_hill(get_param, get_quantized, 'tl', **_)


@bcc.inv_compnode(fwd_name='transcription_hill')
def inverse_transcription_hill(get_param, get_quantized, **_):
    return inverse_transform_hill(get_param, get_quantized, 'tc', **_)


@bcc.inv_compnode(fwd_name='translation_hill')
def inverse_translation_hill(get_param, get_quantized, **_):
    return inverse_transform_hill(get_param, get_quantized, 'tl', **_)


cfg = {
    "learning_rate": 0.001,
    "compile_training": True,
    "node_remap": {
        "sequestron_ERN": "ERN_with_affinity",
        "transcription": "transcription_hill",
        "translation": "translation_hill",
    },
    "rng_key": random.randint(0, 1e9),
}


bc.train.train_xp(xp, cfg, wandb_project="biocomp_20221012A_massCtrls_v2")
# bc.train.train_xp(xp, cfg)

