from jax.config import config as jax_config

import biocomp as bc
import biocomp.compute as bcc
import scriptutils as ut
import jax
import jax.numpy as jnp
import random

random.seed()

lib = ut.load_lib()
xp = ut.load_xp('20221012A_massCtrls', lib)

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                        --     helpers   --
# ···············································································
def glorot_initializer(rng_key, shape):
    def init():
        return jax.nn.initializers.glorot_normal()(rng_key, shape)

    return init


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     hill transforms     --
# ···············································································


def transform_hill(get_param, get_quantized, transform_name, **_):
    def apply(*values, rng_key):
        keys = jax.random.split(rng_key, 4)

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
        V = jnp.average(V_raw, weights=val + bcc.BC_EPSILON)

        K_A = get_param(
            f'{transform_name}_K',
            init=bcc.continuous_initializer(keys[1]),
            shared=True,
            clip_to=(bcc.BC_EPSILON, None),
        )

        n = get_param(
            f'{transform_name}_n',
            init=bcc.continuous_initializer(keys[2], minval=0.25, maxval=3.0),
            shared=True,
            clip_to=(0.2, 4.0),
        )

        mu = get_param(
            f'{transform_name}_mu',
            init=bcc.continuous_initializer(keys[3]),
            shared=True,
            clip_to=(bcc.BC_EPSILON, None),
        )

        value = jax.nn.relu(jnp.sum(val))
        v_k = jnp.maximum(value + K_A, bcc.BC_EPSILON)

        v_over_vk = value / v_k

        pown = jnp.power(v_over_vk, n)

        res = V * pown / mu
        return res

    return apply


def inverse_transform_hill(get_param, get_quantized, transform_name, **_):
    def apply(value, rng_key):
        keys = jax.random.split(rng_key, 4)
        assert value.shape == ()

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
            clip_to=(bcc.BC_EPSILON, None),
        )

        n = get_param(
            f'{transform_name}_n',
            init=bcc.continuous_initializer(keys[2], minval=0.25, maxval=3.0),
            shared=True,
            clip_to=(0.2, 4.0),
        )

        mu = get_param(
            f'{transform_name}_mu',
            init=bcc.continuous_initializer(keys[3]),
            shared=True,
            clip_to=(bcc.BC_EPSILON, None),
        )

        alpha = jnp.power(jax.nn.relu(value) / V * mu, 1 / n)
        res = alpha * K_A / jnp.maximum(1 - alpha, bcc.BC_EPSILON)
        return res

    return apply


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


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                    --     nn based transforms     --
# ···············································································


def nn_dense(input_values, output_size, get_param, key, name):
    input_size = 1 if input_values.shape == () else input_values.shape[0]
    # print(f'nn_dense: {name} {input_size} -> {output_size}')
    # print(f'input_values: {input_values}')
    w = get_param(
        f'{name}_w', init=bcc.glorot_initializer(key, (input_size, output_size)), shared=True
    )
    b = get_param(f'{name}_b', init=lambda: jnp.zeros((output_size,)), shared=True)
    res = jnp.dot(input_values, w) + b
    # print(f'res: {res}')
    return res.squeeze()


def nn_dense_multilevel(input_values, hidden_s, output_s, depth, get_param, key, name, activation):
    res = input_values
    keys = jax.random.split(key, depth)
    for i in range(depth - 1):
        res = activation(nn_dense(res, hidden_s, get_param, keys[i], f'{name}_{i}'))
    return nn_dense(res, output_s, get_param, keys[-1], f'{name}_{depth - 1}')


# params = {}
# get_p = partial(
# bcc.get_param,
# params,
# constraints={},
# node_id=1,
# )

# y = nn_dense(jnp.array([1.0]), 2, get_p, jax.random.PRNGKey(1), 'test')
# yy = nn_dense_multilevel(jnp.array([1.0]), 2, 2, 2, get_p, jax.random.PRNGKey(1), 'test', jax.nn.relu)


def transform_w_dense_layer(get_param, get_quantized, transform_name, wsize=32, depth=2, **_):
    app = bcc._transform(get_param, get_quantized, transform_name, f'{transform_name}_deg', **_)

    def apply(*values, rng_key):
        k1, k2 = jax.random.split(rng_key, 2)
        res = app(*values, rng_key=k1)
        return jax.nn.sigmoid(
            nn_dense_multilevel(res, wsize, 1, depth, get_param, k2, transform_name, jax.nn.relu)
        )

    return apply


def inv_transform_w_dense_layer(get_param, get_quantized, transform_name, wsize=32, depth=2, **_):
    app = bcc._inverse_transform(
        get_param, get_quantized, transform_name, f'{transform_name}_deg', **_
    )

    def apply(value, rng_key):
        k1, k2 = jax.random.split(rng_key, 2)
        res = jax.nn.sigmoid(
            nn_dense_multilevel(
                value, wsize, 1, depth, get_param, k2, f'inv_{transform_name}', jax.nn.relu
            )
        )
        return app(res, rng_key=k1)

    return apply


@bcc.compnode
def transcription_nn(get_param, get_quantized, **_):
    return transform_w_dense_layer(get_param, get_quantized, 'tc', **_)


@bcc.inv_compnode(fwd_name='transcription_nn')
def inverse_transcription_nn(get_param, get_quantized, **_):
    return inv_transform_w_dense_layer(get_param, get_quantized, 'tc', **_)


@bcc.compnode
def translation_nn(get_param, get_quantized, **_):
    return transform_w_dense_layer(get_param, get_quantized, 'tl', **_)


@bcc.inv_compnode(fwd_name='translation_nn')
def inverse_translation_nn(get_param, get_quantized, **_):
    return inv_transform_w_dense_layer(get_param, get_quantized, 'tl', **_)


@bcc.compnode
def ERN_nn_multi(get_param, get_quantized, seq_name, **_):
    def apply(neg, pos, rng_key, **_):
        param_name = f'{seq_name}::affinity'
        affinity = get_param(param_name, init=bcc.continuous_initializer(rng_key), shared=True)
        res = nn_dense_multilevel(
            jnp.array([neg, pos, affinity]).squeeze(), 32, 1, 2, get_param, rng_key, seq_name
        )
        return jax.nn.relu(jnp.squeeze(res))

    return apply


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────



cfg = {
    "learning_rate": 0.001,
    "compile_training": True,
    "node_remap": {
        "sequestron_ERN": "ERN_with_affinity",
        # "transcription": "transcription_nn",
        # "inv_transcription": "inverse_transcription_nn",
        # "translation": "translation_nn",
        # "inv_translation": "inverse_translation_nn",
    },
    "balance_bin_resolution": 0.5,
    "balance_threshold_quantile": 0.4,
    "balance_threshold_min": 40,
    "batch_size": 256,
    "rng_key": random.randint(0, 1e12),
}

# params = {}
# get_p = partial(
# bcc.get_param,
# params,
# constraints={},
# node_id=1,
# )

# def get_qu(_, v, **__):
# return v

# f = transform_hill(get_p, get_qu, 'test')
# inv_f = inverse_transform_hill(get_p, get_qu, 'test')
# params
# y = f(-0.093000, rng_key=jax.random.PRNGKey(1))
# inv_f(y, rng_key=jax.random.PRNGKey(1))

# ftc = transcription_hill(get_p, get_qu)
# inv_ftc = inverse_transcription_hill(get_p, get_qu)
# ftl = translation_hill(get_p, get_qu)
# inv_ftl = inverse_translation_hill(get_p, get_qu)

# ytc = ftc(0.1093000, rng_key=jax.random.PRNGKey(1))
# inv_ftc(ytc, rng_key=jax.random.PRNGKey(1))
# ytl = ftl(0.093000, rng_key=jax.random.PRNGKey(1))
# inv_ftl(ytl, rng_key=jax.random.PRNGKey(1))

# models = xp.get_models(node_remap=cfg['node_remap'])
# model = models.items().__iter__().__next__()[1]
# p, c = model.init(jax.random.PRNGKey(cfg['rng_key']))

# X, Y = xp.get_XY(models)

# inp = jnp.array([1.0, 2.0])
# model.collect_all_results(p, inp, rng_key=jax.random.PRNGKey(0))

# model.network.name
# ut.plot_networks([model.network], [f'../__out/{model.network.name}_dbg.pdf'])

# bc.train.train_xp(xp, cfg, wandb_project="biocomp_20221012A_massCtrls_v4")
bc.train.train_xp(xp, cfg)


