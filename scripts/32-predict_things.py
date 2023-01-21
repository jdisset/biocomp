# ─────────────────────────────────────────────────────────────────────────────
#                                     SETUP
# ───────────────────────────────────── ▼ ─────────────────────────────────────
### {{{                          --     imports     --
import biocomp as bc
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import biocomp.utils as bu
import scriptutils as ut
import jax
from jax import jit, vmap, grad, value_and_grad
import jax.numpy as jnp
import biocomp.datautils as du
import optax
from tqdm import tqdm
import biocomp.nodes as bn
import biocomp.compute as bcc
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10.0, 10.0]
plt.rcParams['figure.dpi'] = 300

##────────────────────────────────────────────────────────────────────────────}}}
### {{{                          --     config     --
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
cfg = {
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "rng_key": np.random.randint(0, 2**32),
    # "rng_key": 11325,
    "epochs": 10,
    "compile_training": True,
    "batch_size": 5000,
    "n_batches": 1,
    "norm_factor": 1e3,
    "kde_bw_method": 0.1,
    "density_quantile_threshold": 0.1,
    "node_impl": node_impl,
}
cfg["batch_size"] = 3
cfg["n_batches"] = 2
cfg["kde_bw_method"] = 0.05
cfg["log_factor"] = 5e2
cfg["max_value"] = 1e7
cfg["density_quantile_threshold"] = 0.1

##────────────────────────────────────────────────────────────────────────────}}}

### {{{ --     create a composite training dataset from uOrfs + ERN data     --

lib = ut.load_lib()

uorf_xp = ut.load_xp('2022-11-10_uORFs_and_company', lib)
ern_xp = ut.load_xp('20220501-GW-l1vsl2', lib)

uorf_models, uorf_samples = uorf_xp.build_models(node_impl=cfg['node_impl'], inverse='all')
ern_models, ern_samples = ern_xp.build_models(node_impl=cfg['node_impl'], inverse='all')
models = uorf_models + ern_models

uorf_X, uorf_Y = uorf_xp.get_XY(uorf_models, uorf_samples)
ern_X, ern_Y = ern_xp.get_XY(ern_models, ern_samples)
raw_X, raw_Y = uorf_X + ern_X, uorf_Y + ern_Y

dman = du.DataManager(raw_X, raw_Y, models, cfg)

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                 --     show data for each model     --
rng = jax.random.PRNGKey(0)

from pathlib import Path

save_dir = Path('~/Desktop/32-predict_things').expanduser()

figs, axes = zip(*[du.mkfig(1, 2) for _ in range(len(dman.kdes))])

ut.plot_networks(
    [model.network for model in dman.models], axes=[ax[0] for ax in axes], show_title=False
)

for i, (model, X, Y, kde, fig, ax) in tqdm(
    list(enumerate(zip(dman.models, dman.X, dman.Y, dman.kdes, figs, axes)))
):
    subsample = du.optimal_density_subsample(X, kde, rng, quantile_threshold=0.1)
    x, y = X[subsample], Y[subsample]
    if x.shape[1] == 1:
        du.smooth_1d(x, y, model, dman.rescale, ax[1])
    else:
        du.smooth_2d(x, y, model, dman.rescale, ax[1])
    fig.suptitle(f'{model.network.name} \n(after density-based resampling of {x.shape[0]} points)')
    # save to desktop
    sdir = save_dir / 'data2' / f'{model.network.name}_{i}.png'
    sdir.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(sdir)
    plt.close(fig)


##────────────────────────────────────────────────────────────────────────────}}}


key = jax.random.PRNGKey(0)
xbatches, ybatches = dman.get_batches(key)  # (B,M,N,F) shape

##

key = jax.random.PRNGKey(cfg['rng_key'])

config = {**bc.train.DEFAULT_CFG, **cfg}
optimizer = optax.adam(learning_rate=cfg['learning_rate'])

ikeys = jax.random.split(key, len(models))

params, constraints = {}, {}
uorf_models, uorf_samples = uorf_xp.build_models(node_impl=config['node_impl'], inverse='all')
ern_models, ern_samples = ern_xp.build_models(node_impl=config['node_impl'], inverse='all')
models = uorf_models + ern_models
for m, k in zip(models, ikeys):
    params, constraints = m.init(k, pre_params=params, pre_constraints=constraints)

dynamic, _ = bu.split_params(params, config['static_params'])
opt_state = optimizer.init(dynamic)

zbatches = jax.random.uniform(key, ybatches.shape)
nmodels = len(models)

# model = models[0]
# x = xbatches[0, 0, :]
# y = ybatches[0, 0, :]
# z = zbatches[0, 0, :]
# X = xbatches[0, :]
# Y = ybatches[0, :]
# Z = zbatches[0, :]


def huber_quantile_loss(e, q, delta=0.1):
    return jnp.where(
        jnp.abs(e) <= delta, 0.5 * e**2, delta * (jnp.abs(e) - 0.5 * delta)
    ) * jnp.where(e < 0, q, (1.0 - q))


def model_loss(model, params, x, y, z, key):
    assert x.ndim == y.ndim == z.ndim == 2
    keys = jax.random.split(key, x.shape[0])
    yhat = vmap(model, in_axes=(None, 0, 0, 0))(params, x, z, keys)
    error = yhat - y[:, : model.n_outputs]
    return jnp.mean(huber_quantile_loss(error, z[:, : model.n_outputs]))


def loss_func(dynamic, static, X, Y, Z, key):
    # Z is the quantile
    # shape = (N_MODELS, BATCH_SIZE, FEATURES)
    assert len(X) == len(Y) == len(Z) == nmodels
    assert X.shape[1] == Y.shape[1] == Z.shape[1]
    assert X.shape[2] == (Y.shape[2] - 1) == (Z.shape[2] - 1)

    params = bu.assemble_params(dynamic, static)

    K = jax.random.split(key, nmodels)
    res = jnp.array(
        [model_loss(m, params, x, y, z, k) for m, x, y, z, k in zip(models, X, Y, Z, K)]
    )
    assert res.shape == (nmodels,)

    return res.mean()

def flatten_tree(g):
        leaves = jax.tree_util.tree_leaves(g)
        return jnp.concatenate([l.flatten() for l in leaves])

def training_step(params, opt_state, x, y, z, key):
    dynamic, static = bu.split_params(params, [['node']])
    loss, grads = value_and_grad(loss_func)(dynamic, static, x, y, z, key)
    updates, opt_state = optimizer.update(grads, opt_state, dynamic)

    magnitude = jnp.linalg.norm(flatten_tree(updates))

    dynamic = optax.apply_updates(dynamic, updates)
    dynamic = bu.apply_constraints(dynamic, constraints)
    params = bu.assemble_params(dynamic, static)

    res = {
        'params': params,
        'loss': loss,
        'grad': grads,
        'opt': opt_state,
        'magnitude': magnitude,
    }
    return res

nbatches = xbatches.shape[0]

@bu.progress_scan(nbatches, message='Training model')
def scannable_step(carry, i_x_y_z_k):
    params, opt_state = carry
    i, x, y, z, k = i_x_y_z_k
    updt = training_step(params, opt_state, x, y, z, k)
    params, opt_state = updt['params'], updt['opt']
    return (params, opt_state), updt

def epoch_step(start_params, start_opt_state, epoch_key):
    batch_keys = jax.random.split(epoch_key, nbatches)
    (final_params, final_opt_state), epoch_history = jax.lax.scan(
        scannable_step,
        (start_params, start_opt_state),
        (jnp.arange(nbatches), xbatches, ybatches, zbatches, batch_keys),
    )
    return final_params, final_opt_state, epoch_history

step = epoch_step

if cfg['compile_training']:
    import time
    print('Compiling training step')
    t0 = time.time()
    step = jit(step)
    lowered = step.lower(params, opt_state, key)
    compiled = lowered.compile()
    step = compiled
    print(f'Compiled in {time.time() - t0:.2f}s')

##
def console_log(epoch, cfg, epoch_history=None, **_):
    if epoch_history is not None:
        loss = np.array(epoch_history['loss'])
        avg = np.mean(loss)
        std = np.std(loss)
        lmin, lmax = jnp.min(loss), jnp.max(loss)
        print(
            f'[{epoch}/{cfg["epochs"]}] loss: {avg:.3f} ± {std:.3f} [min {lmin:.3f}, max {lmax:.3f}]'
        )

loggers = None

if loggers is None:
    loggers = []

print('Initial logger calls')
for _, l in loggers:
    l(0, config)



print('Beginning training')

for i, epoch_key in enumerate(jax.random.split(key, config['epochs']), 1):
    params, opt_state, epoch_history = step(params, opt_state, epoch_key)
    print(f'Epoch {i} complete')
    for t, l in loggers:
        if i % t == 0 or i == cfg['epochs']:
            l(i, config, epoch_history=epoch_history, nbatches=nbatches)

epoch_history['loss']

