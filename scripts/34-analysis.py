### {{{                          --     imports     --
import datetime
import biocomp as bc
import matplotlib.pyplot as plt
import numpy as np
import time
from functools import partial
import biocomp.utils as bu
import scriptutils as ut
import jax
from jax import jit, vmap, grad, value_and_grad
import jax.numpy as jnp
import biocomp.datautils as du
import optax
from pathlib import Path
from tqdm import tqdm
import biocomp.nodes as bn
import biocomp.compute as bcc
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [7.0, 7.0]
plt.rcParams['figure.dpi'] = 200

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                        --     node config     --
T_SIZE = 64
T_DEPTH = 3
I_SIZE = 64
I_DEPTH = 2
I_OUT = 8
ERN_SIZE = 128
ERN_DEPTH = 3
MEFL_SIZE = 64
MEFL_DEPTH = 3

node_impl = dict(
    bc.nodes.DEFAULT_COMPUTE_NODES_DICT,
    **{
        'output': partial(bc.nn.output, wsize=MEFL_SIZE, depth=MEFL_DEPTH),
        'transcription': partial(
            bc.nn.transcription,
            outer_wsize=T_SIZE,
            outer_depth=T_DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
            inner_out=I_OUT,
        ),
        'translation': partial(
            bc.nn.translation,
            outer_wsize=T_SIZE,
            outer_depth=T_DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
            inner_out=I_OUT,
        ),
        'inv_transcription': partial(
            bc.nn.inv_transcription,
            outer_wsize=T_SIZE,
            outer_depth=T_DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
            inner_out=I_OUT,
        ),
        'inv_translation': partial(
            bc.nn.inv_translation,
            outer_wsize=T_SIZE,
            outer_depth=T_DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
            inner_out=I_OUT,
        ),
        'sequestron_ERN': partial(bc.nn.ERN5p, wsize=ERN_SIZE, depth=ERN_DEPTH),
        'sequestron_ERN3p': partial(bc.nn.ERN3p, wsize=ERN_SIZE, depth=ERN_DEPTH),
    },
)
##────────────────────────────────────────────────────────────────────────────}}}

lib = ut.load_lib()
uorf_xp = ut.load_xp('2022-11-10_uORFs_and_company', lib)
ern_xp = ut.load_xp('20220501-GW-l1vsl2', lib)


config = {
    **bc.train.DEFAULT_CFG,
    **{
        'node_impl': node_impl,
        'epochs': 30,
        # "rng_key": np.random.randint(0, 2 ** 32),
        "rng_key": 1,
    },
}
##

nlines = {}
dman = du.DataManager.from_xps([uorf_xp, ern_xp], config, inverse='all')
for subset in tqdm([list(range(i)) for i in range(1, 4)]):
    # dman.set_subset([0, 47])
    dman.set_subset(subset)

    key = jax.random.PRNGKey(config['rng_key'])
    xbatches, ybatches = dman.get_batches(key)  # (B,M,N,F) shape
    zbatches = jax.random.uniform(key, ybatches.shape)
    models = dman.get_jitted_models()

    # --- init
    params, constraints = {}, {}
    optimizer = bc.train.get_optimizer(config)
    for m, k in zip(models, jax.random.split(key, len(models))):
        params, constraints = m.init(k, pre_params=params, pre_constraints=constraints)
    dynamic, _ = bu.split_params(params, config['static_params'])
    opt_state = optimizer.init(dynamic)

    nbatches, nmodels = config['n_batches'], len(models)
    assert nbatches == xbatches.shape[0] == ybatches.shape[0]

    x_start = np.cumsum([m.n_inputs for m in models])[:-1]
    y_start = np.cumsum([m.n_outputs for m in models])[:-1]


    jnp.split(xbatches[0][0], x_start, axis=0)

    def apply_models(params, x, z, key):
        keys = jax.random.split(key, nmodels)
        xs = jnp.split(x, x_start)
        zs = jnp.split(z, y_start)
        yhat = [m(params, xx, zz, k) for m, xx, zz, k in zip(models, xs, zs, keys)]
        return jnp.concatenate(yhat, axis=0)

    vmap_of_for = vmap(apply_models, in_axes=(None, 0, 0, 0))
    def vfloss(params, X, Y, Z, key):
        keys = jax.random.split(key, X.shape[0])
        yhat = vmap_of_for(params, X, Z, keys)
        error = yhat - Y
        return jnp.mean(error**2)
    def for_of_vmap(params, X, Z, key):
        keys = jax.random.split(key, X.shape[0] * nmodels)
        keys = keys.reshape((nmodels, X.shape[0], 2))
        xs = jnp.split(X, x_start, axis=1)
        zs = jnp.split(Z, y_start, axis=1)
        yhat = [
            vmap(m, in_axes=(None, 0, 0, 0))(params, xx, zz, k)
            for m, xx, zz, k in zip(models, xs, zs, keys)
        ]
        return jnp.concatenate(yhat, axis=1)
    def fvloss(params, X, Y, Z, key):
        yhat = for_of_vmap(params, X, Z, key)
        error = yhat - Y
        return jnp.mean(error**2)

    assert vfloss(params, xbatches[0], ybatches[0], zbatches[0], key) == fvloss(
        params, xbatches[0], ybatches[0], zbatches[0], key
    )

    vfloss(params, xbatches[0], ybatches[0], zbatches[0], key)
    fvloss(params, xbatches[0], ybatches[0], zbatches[0], key)

    xbatches.shape

    # s = jax.make_jaxpr(apply_models)(params, xbatches[0][0], zbatches[0][0], key).pretty_print()
    # nlines.setdefault('apply_models', {})
    # nlines[len(subset)] = len(s.splitlines())
    # nlines[len(subset)] = len(s.splitlines())
    # print(len(subset), nlines[len(subset)])

    s_vf = jax.make_jaxpr(vfloss)(params, xbatches[0], ybatches[0], zbatches[0], key).pretty_print()
    len(s_vf.splitlines())

    nlines.setdefault('vf', {})
    nlines['vf'][len(subset)] = len(s_vf.splitlines())

    s_fv = jax.make_jaxpr(fvloss)(params, xbatches[0], ybatches[0], zbatches[0], key).pretty_print()
    len(s_fv.splitlines())
    nlines.setdefault('fv', {})
    nlines['fv'][len(subset)] = len(s_fv.splitlines())

    print(len(subset), nlines['vf'][len(subset)], nlines['fv'][len(subset)])

##

plt.plot(list(nlines.keys()), list(nlines.values()))
plt.show()
