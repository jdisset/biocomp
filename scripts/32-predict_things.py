# ─────────────────────────────────────────────────────────────────────────────
#                                     SETUP
# ───────────────────────────────────── ▼ ─────────────────────────────────────
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
    "epochs": 150,
    "compile_training": True,
    "node_impl": node_impl,
}
cfg["batch_size"] = 32
cfg["n_batches"] = 1000
cfg["kde_bw_method"] = 0.05
cfg["log_factor"] = 1e3
cfg["max_value"] = 1e7
cfg["density_quantile_threshold"] = 0.1

##────────────────────────────────────────────────────────────────────────────}}}

### {{{ --     create a composite training dataset from uOrfs + ERN data     --

lib = ut.load_lib()

uorf_xp = ut.load_xp('2022-11-10_uORFs_and_company', lib)
ern_xp = ut.load_xp('20220501-GW-l1vsl2', lib)

uorf_models, uorf_samples = uorf_xp.build_models(node_impl=cfg['node_impl'], inverse='all')
ern_models, ern_samples = ern_xp.build_models(node_impl=cfg['node_impl'], inverse='all')
combined_models = uorf_models + ern_models

uorf_X, uorf_Y = uorf_xp.get_XY(uorf_models, uorf_samples)
ern_X, ern_Y = ern_xp.get_XY(ern_models, ern_samples)
raw_X, raw_Y = uorf_X + ern_X, uorf_Y + ern_Y


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                --     summary model plot functions     --
def model_plot(model: bc.ComputeGraphModel, X, Y, rescaler, ax, kde=None, **kw):
    ninputs = model.n_inputs
    noutputs = model.n_outputs
    x, y = X[:, :ninputs], Y[:, :noutputs]
    if kde is not None:
        rng = jax.random.PRNGKey(0)
        subsample = du.optimal_density_subsample(x, kde, rng, quantile_threshold=0.1)
        x, y = x[subsample], y[subsample]

    if ninputs == 1:
        du.smooth_1d(x, y, model, rescaler, ax, **kw)
    elif ninputs == 2:
        du.smooth_2d(x, y, model, rescaler, ax, **kw)
    elif ninputs == 3:
        du.smooth_3d(x, y, model, rescaler, ax, **kw)


def eval_model_plot(
    model: bc.ComputeGraphModel,
    params,
    rescaler,
    ax,
    npoints=50000,
    key=jax.random.PRNGKey(0),
    jitted=None,
    **kw,
):

    k_i, k_q = jax.random.split(key)
    inputs = jax.random.uniform(k_i, (npoints, model.n_inputs))
    quantiles = jax.random.uniform(k_q, (npoints, model.n_outputs))
    keys = jax.random.split(key, npoints)
    jm = jitted or model
    results = vmap(jm, in_axes=(None, 0, 0, 0))(params, inputs, quantiles, keys)
    model_plot(model, inputs, results, rescaler, ax, **kw)


def report(params, dman, id, suptitle=''):
    fig, ax = du.mkfig(1, 2, size=(4, 4))
    model = dman.get_models()[id]
    mX = dman.get_X()[id]
    mY = dman.get_Y()[id]
    model_plot(model, mX, mY, dman.rescale, ax[0], kde=dman.get_kdes()[id])
    eval_model_plot(model, params, dman.rescale, ax[1], jitted=dman.get_jitted_models()[id])
    ax[0].set_title(f'Original data (mean)')
    ax[1].set_title(f'Predicted (mean)')
    fig.suptitle(f'{suptitle} {model.node_namespace}')
    return fig, ax


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                     --     logging functions     --
import wandb as wb

project = 'quantile_v0'
project = None
log_grads_and_params_to_wandb = False


current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


@partial(jit, static_argnums=(1,))
def compstats(v, smooth_win=1):
    medians = vmap(jnp.median)(v)
    mins = vmap(jnp.min)(v)
    maxs = vmap(jnp.max)(v)
    p20s = vmap(lambda x: jnp.percentile(x, 20))(v)
    p80s = vmap(lambda x: jnp.percentile(x, 80))(v)
    if smooth_win > 1:
        medians = jnp.convolve(medians, jnp.ones(smooth_win) / smooth_win, mode='same')
        p80s = jnp.convolve(p80s, jnp.ones(smooth_win) / smooth_win, mode='same')
        p20s = jnp.convolve(p20s, jnp.ones(smooth_win) / smooth_win, mode='same')
        maxs = jnp.convolve(maxs, jnp.ones(smooth_win) / smooth_win, mode='same')
        mins = jnp.convolve(mins, jnp.ones(smooth_win) / smooth_win, mode='same')
    return medians, p20s, p80s, mins, maxs


def get_epoch_stats(epoch_data, smooth_win=1):
    stats = {'grad': {}, 'params': {}}
    for k, v in epoch_data['grad']['shared'].items():
        stats['grad'][k] = compstats(v)
    for k, v in epoch_data['params']['shared'].items():
        stats['params'][k] = compstats(v)
    return stats


def local_save(epoch, cfg, epoch_history=None, save_dir=None, full_save=False, **_):
    assert save_dir is not None
    if epoch_history is None:
        return
    t0 = time.time()

    if full_save:
        full_save_until_epoch = full_save if isinstance(full_save, int) else 2
        if epoch <= full_save_until_epoch:
            du.save(epoch_history, f'{save_dir}/epoch_{epoch}_full.pkl')

    stats = get_epoch_stats(epoch_history)
    params = bu.tree_get(epoch_history['params'], nbatches - 1)
    loss = np.array(epoch_history['loss'])
    avg_loss = np.mean(loss)
    stats['loss'] = loss

    du.save(stats, f'{save_dir}/epoch_{epoch}_stats.pkl')

    # first we rename the old params
    for f in Path(save_dir).glob('latest_params_*.pkl'):
        f.rename(f'{save_dir}/old_{f.name}')

    # then we save the new ones
    du.save(params, f'{save_dir}/latest_params_({avg_loss:.5f}).pkl')

    # then we delete the old one
    for f in Path(save_dir).glob('old_latest_params_*.pkl'):
        f.unlink()

    print(f"Saving epoch to disk took {time.time() - t0:.2f}s")


def wandb_plot_pred(epoch, cfg, dman, epoch_history=None, **_):
    if epoch_history is None:
        return

    t0 = time.time()
    params = bu.tree_get(epoch_history['params'], nbatches - 1)
    pred = []
    models = dman.get_models()
    try:
        for i in range(len(dman.get_models())):
            fig, ax = report(params, dman, i)
            fig.set_dpi(10)
            pred.append(wb.Image(fig, caption=f'{models[i].node_namespace}'))
            plt.close(fig)
    except Exception as e:
        # raise e
        print(e)
        print("Failed to plot predictions")
    wb.log({'Evaluations': pred})
    print(f'Done logging prediction plots for epoch {epoch} in {time.time() - t0:.2f}s')


def wandb_log_epoch(epoch, cfg, epoch_history=None, **_):
    if epoch_history is not None:
        # measure time now:
        t0 = time.time()
        losses = np.array(epoch_history['loss'])
        for loss in losses:
            wb.log({'loss': loss})
        del losses
        print(f"Logging epoch {epoch} to wandb took {time.time() - t0:.2f}s")


def console_log(epoch, cfg, epoch_history=None, **_):
    if epoch_history is not None:
        loss = np.array(epoch_history['loss'])
        avg = np.mean(loss)
        std = np.std(loss)
        lmin, lmax = jnp.min(loss), jnp.max(loss)
        print(
            f'[{epoch}/{cfg["epochs"]}] loss: {avg:.4f} ± {std:.4f} [min {lmin:.4f}, max {lmax:.4f}]'
        )


##────────────────────────────────────────────────────────────────────────────}}}

config = {**bc.train.DEFAULT_CFG, **cfg}

if project is not None:
    wb.init(config=config, project=project, entity="jdisset", reinit=True)

save_dir = f'../__out/{project}/{current_date}' if project is None else f'../__out/{wb.run.name}'

dman = du.DataManager(raw_X, raw_Y, combined_models, config)
dman.set_subset([0, 22, 47, 10])


if project is not None:
    loggers = [
        (1, console_log),
        (100, partial(local_save, save_dir=save_dir)),
        (1, wandb_log_epoch),
        (10, partial(wandb_plot_pred, dman=dman)),
    ]
else:
    loggers = [
        (1, console_log),
    ]

key = jax.random.PRNGKey(config['rng_key'])
xbatches, ybatches = dman.get_batches(key)  # (B,M,N,F) shape

zbatches = jax.random.uniform(key, ybatches.shape)
nbatches = xbatches.shape[0]

models = dman.get_models()
nmodels = len(models)

assert xbatches.shape[2] == sum([m.n_inputs for m in models])
assert ybatches.shape[2] == sum([m.n_outputs for m in models])
##

x_start = np.cumsum([0] + [m.n_inputs for m in models])[:-1]
x_end = np.array([m.n_inputs for m in models]) + x_start


def apply_models(params, x, z, key):
    keys = jax.random.split(key, nmodels)
    y = [m(params, x[s:e], z[s:e], k) for m, s, e, k in zip(models, x_start, x_end, keys)]
    return jnp.concatenate(y, axis=0)


optimizer = optax.adam(learning_rate=config['learning_rate'])

params, constraints = {}, {}

for m, k in zip(models, jax.random.split(key, len(models))):
    params, constraints = m.init(k, pre_params=params, pre_constraints=constraints)

dynamic, _ = bu.split_params(params, config['static_params'])
opt_state = optimizer.init(dynamic)


def huber_quantile_loss(e, q, delta=0.1):
    return jnp.where(
        jnp.abs(e) <= delta, 0.5 * e**2, delta * (jnp.abs(e) - 0.5 * delta)
    ) * jnp.where(e < 0, q, (1.0 - q))


def loss_func(dynamic, static, X, Y, Z, key):
    # Z is the quantile
    # shape = (BATCH_SIZE, FEATURES*MODELS)
    assert X.ndim == Y.ndim == Z.ndim == 2
    assert X.shape[0] == Y.shape[0] == Z.shape[0]
    assert X.shape[1] == sum([m.n_inputs for m in models])
    assert Y.shape[1] == Z.shape[1] == sum([m.n_outputs for m in models])

    params = bu.assemble_params(dynamic, static)

    keys = jax.random.split(key, X.shape[0])
    yhat = vmap(apply_models, in_axes=(None, 0, 0, 0))(params, X, Z, keys)
    assert yhat.shape == Y.shape

    error = yhat - Y
    return jnp.mean(huber_quantile_loss(error, Z))


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


@bu.progress_scan(nbatches, message='Training model')
def scannable_step(carry, i_x_y_z_k):
    params, opt_state = carry
    i, x, y, z, k = i_x_y_z_k
    updt = training_step(params, opt_state, x, y, z, k)
    params, opt_state = updt['params'], updt['opt']
    return (params, opt_state), updt


# if cfg['compile_training']:
# import time
# print('Compiling training step')
# t0 = time.time()
# scannable_step = jit(scannable_step)
# batch_keys = jax.random.split(key, nbatches)
# ixyzk = (jnp.arange(nbatches)[0], xbatches[0], ybatches[0], zbatches[0], batch_keys[0])
# lowered = scannable_step.lower((params, opt_state), ixyzk)
# scannable_step = lowered.compile()
# print(f'Compiled in {time.time() - t0:.2f}s')


def epoch_step(start_params, start_opt_state, epoch_key):
    batch_keys = jax.random.split(epoch_key, nbatches)
    (final_params, final_opt_state), epoch_history = jax.lax.scan(
        scannable_step,
        (start_params, start_opt_state),
        (jnp.arange(nbatches), xbatches, ybatches, zbatches, batch_keys),
    )
    return final_params, final_opt_state, epoch_history


if config['compile_training']:
    epoch_step = jit(epoch_step)
    # print('Compiling epoch_step')
    # t0 = time.time()
    # lowered = epoch_step.lower(params, opt_state, key)
    # epoch_step = lowered.compile()
    # print(f'Compiled in {time.time() - t0:.2f}s')

if loggers is None:
    loggers = []

print('Initial logger calls')
for _, l in loggers:
    l(0, config)

print('Beginning training')

config['epochs'] = 15

for i, epoch_key in enumerate(jax.random.split(key, config['epochs']), 1):
    params, opt_state, epoch_history = epoch_step(params, opt_state, epoch_key)
    for t, l in loggers:
        if i % t == 0 or i == cfg['epochs']:
            l(i, config, epoch_history=epoch_history, nbatches=nbatches)


# TODO:
# - add recog sites as translation rate modifiers
# - store parameters somewhere easily retrievable
# - plot individual nodes
# - predict uORFs on the ERN side

### {{{                          --     archive     --

# ###  {{{                 --     show data for each model     --
# rng = jax.random.PRNGKey(0)

# from pathlib import Path

# save_dir = Path('~/Desktop/32-predict_things').expanduser()

# figs, axes = zip(*[du.mkfig(1, 2) for _ in range(len(dman.kdes))])

# ut.plot_networks(
# [model.network for model in dman.models], axes=[ax[0] for ax in axes], show_title=False
# )

# for i, (model, X, Y, kde, fig, ax) in tqdm(
# list(enumerate(zip(dman.models, dman.X, dman.Y, dman.kdes, figs, axes)))
# ):
# subsample = du.optimal_density_subsample(X, kde, rng, quantile_threshold=0.1)
# x, y = X[subsample], Y[subsample]
# if x.shape[1] == 1:
# du.smooth_1d(x, y, model, dman.rescale, ax[1])
# else:
# du.smooth_2d(x, y, model, dman.rescale, ax[1])
# fig.suptitle(f'{model.network.name} \n(after density-based resampling of {x.shape[0]} points)')
# # save to desktop
# sdir = save_dir / 'data2' / f'{model.network.name}_{i}.png'
# sdir.parent.mkdir(parents=True, exist_ok=True)
# fig.savefig(sdir)
# plt.close(fig)


##────────────────────────────────────────────────────────────────────────────}}}


##────────────────────────────────────────────────────────────────────────────}}}
