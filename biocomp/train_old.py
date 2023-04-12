## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     import and init     --
# ···············································································
from jax.tree_util import Partial as partial
import jax
from datetime import datetime
import jax.numpy as jnp
from jax import jit, vmap, grad, value_and_grad
from pathlib import Path
from jax.tree_util import Partial as partial
import json
from tqdm import tqdm
from typing import Callable
import pandas as pd
import optax
import matplotlib.pyplot as plt
import numpy as np
from . import datautils as du
from . import utils as ut
from . import nodes as nodes
from . import nodes_old as nodes_old
from . import defaults as dft
# from .compute import ComputeGraphModel
import wandb as wb
import os
import time

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     loss functions     --
# ···············································································


def mse_loss(y, y_hat, n_outputs=None):
    if n_outputs is None:
        n_outputs = y.shape[1]
    assert y_hat.ndim == 2 and y.ndim == 2
    return jnp.mean((y[:, :n_outputs] - y_hat[:, :n_outputs]) ** 2)


def huber_quantile_loss(e, q, delta=0.1):
    return jnp.where(
        jnp.abs(e) <= delta, 0.5 * e**2, delta * (jnp.abs(e) - 0.5 * delta)
    ) * jnp.where(e < 0, q, (1.0 - q))


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     loggers     --
# ···············································································


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

    params = ut.tree_get(epoch_history['params'], -1)
    loss = np.array(epoch_history['loss'])
    avg_loss = np.mean(loss)
    # stats = get_epoch_stats(epoch_history)
    # stats['loss'] = loss
    # du.save(stats, f'{save_dir}/epoch_{epoch}_stats.pkl')

    # first we rename the old params
    for f in Path(save_dir).glob('latest_params.pkl'):
        f.rename(f'{save_dir}/old_params.pkl')

    # then we save the new ones
    du.save(params, f'{save_dir}/latest_params.pkl')

    # then we delete the old one
    for f in Path(save_dir).glob('old_params.pkl'):
        f.unlink()

    print(f"Saving epoch to disk took {time.time() - t0:.2f}s")


def wandb_plot_pred(epoch, cfg, dman, epoch_history=None, **_):
    if epoch_history is None:
        return

    t0 = time.time()
    params = ut.tree_get(epoch_history['params'], -1)
    pred = []
    models = dman.get_models()
    try:
        for i in range(len(dman.get_models())):
            fig, ax = du.report(params, dman, i)
            # fig.set_dpi(10)
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
        losses = np.array(epoch_history['loss'])
        for loss in losses:
            wb.log({'loss': loss})
        wb.log({'epoch_time': epoch_history['epoch_time']})


def console_log(epoch, cfg, epoch_history=None, **_):
    if epoch_history is not None and len(epoch_history['loss']) > 0:
        loss = np.array(epoch_history['loss'])
        avg = np.mean(loss)
        std = np.std(loss)
        lmin, lmax = jnp.min(loss), jnp.max(loss)
        print(
            f'[{epoch}/{cfg["epochs"]}] loss: {avg:.4f} ± {std:.4f} [min {lmin:.4f}, max {lmax:.4f}] in {epoch_history["epoch_time"]:.2f}s'
        )


def log_w_replicates(history, epoch, cfg, **_):
    loss = history['loss'][-1]
    losses = {
        'mean': jnp.mean(loss),
        'std': jnp.std(loss),
        'min': jnp.min(loss),
        'max': jnp.max(loss),
    }
    print(
        f'epoch: {epoch}, loss: \n - mean: {losses["mean"]:.3f}, std: {losses["std"]:.3f}, min: {losses["min"]:.3f}, max: {losses["max"]:.3f}'
    )


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


def get_optimizer(cfg):
    optimizers = {
        'sgd': optax.sgd(learning_rate=cfg['learning_rate']),
        'adam': optax.adam(learning_rate=cfg['learning_rate']),
        'adamw': optax.adamw(learning_rate=cfg['learning_rate'], weight_decay=cfg['adam_w_decay']),
        'amsgrad': optax.amsgrad(learning_rate=cfg['learning_rate']),
    }
    assert (
        cfg['optimizer'] in optimizers.keys()
    ), f"Optimizer {cfg['optimizer']} not available. Available optimizers are {optimizers.keys()}"
    optimizer = optimizers[cfg['optimizer']]
    return optimizer


def setup_wandb_logging(project, dman, config):
    import wandb as wb

    wb.init(config=config, project=project, entity="jdisset", reinit=True)
    save_dir = Path(wb.run.dir)
    loggers = [
        (1, console_log),
        (100, partial(local_save, save_dir=save_dir)),
        (1, wandb_log_epoch),
        (100, partial(wandb_plot_pred, dman=dman)),
    ]
    return loggers


def start(dman: du.DataManager, cfg, loggers=None):
    config = {**dft.DEFAULT_CONFIG, **cfg}

    key = jax.random.PRNGKey(config['rng_key'])

    models = dman.get_models()
    nmodels = len(models)

    # --- init
    # params
    params, constraints = {}, {}
    optimizer = get_optimizer(config)
    for m, k in zip(models, jax.random.split(key, len(models))):
        params, constraints = m.init(k, pre_params=params, pre_constraints=constraints)
    dynamic, _ = ut.split_params(params, config['static_params'])
    opt_state = optimizer.init(dynamic)

    # batches
    xbatches, ybatches = dman.get_batches(key)  # (B,M,N,F) shape
    total_batches = config['n_batches']
    assert total_batches == xbatches.shape[0] == ybatches.shape[0]
    nbatches_per_epoch = total_batches // config['n_epochs_per_batch_rotation']

    # --- loss and updates
    x_start = np.cumsum([m.n_inputs for m in models])[:-1]
    y_start = np.cumsum([m.n_outputs for m in models])[:-1]

    def apply_models_and_grad(params, x, z, key):
        k1, k2 = jax.random.split(key)
        keys = jax.random.split(k1, nmodels)
        xs = jnp.split(x, x_start)
        zs = jnp.split(z, y_start)
        res = [
            m.apply_and_negative_grad(params, xx, zz, k)
            for m, xx, zz, k in zip(models, xs, zs, keys)
        ]
        keys = jax.random.split(k2, nmodels)
        just_for_grads = [
            m.apply_and_negative_grad(
                params, xx, zz, k, override_w_uniform=['transcription', 'translation', 'output']
            )
            for m, xx, zz, k in zip(models, xs, zs, keys)
        ]
        yhat, negative_grads_x = zip(*res)
        _, negative_grads_rdm = zip(*just_for_grads)
        return jnp.concatenate(yhat, axis=0), jnp.sum(
            ut.flat_concat(*[negative_grads_x, negative_grads_rdm])
        )

    def loss_func(dynamic, static, X, Y, Z, key):
        assert X.ndim == Y.ndim == Z.ndim == 2
        assert X.shape[0] == Y.shape[0] == Z.shape[0]
        assert X.shape[1] == sum([m.n_inputs for m in models])
        assert Y.shape[1] == Z.shape[1] == sum([m.n_outputs for m in models])

        params = ut.assemble_params(dynamic, static)

        keys = jax.random.split(key, X.shape[0])
        yhat, grads = vmap(apply_models_and_grad, in_axes=(None, 0, 0, 0))(params, X, Z, keys)
        assert yhat.shape == Y.shape
        assert grads.shape == (X.shape[0],)

        error = yhat - Y
        qantile_loss = jnp.mean(
            huber_quantile_loss(error, Z, delta=config['huber_quantile_loss_delta'])
        )

        negative_grad_penalty = config['negative_grad_penalty'] * jnp.mean(
            jnp.where(grads < 0, -grads, 0)
        )

        return qantile_loss + negative_grad_penalty

    def training_step(params, opt_state, x, y, z, key):
        dynamic, static = ut.split_params(params, [['node']])
        loss, grads = value_and_grad(loss_func)(dynamic, static, x, y, z, key)
        updates, opt_state = optimizer.update(grads, opt_state, dynamic)

        dynamic = optax.apply_updates(dynamic, updates)
        dynamic = ut.apply_constraints(dynamic, constraints)
        params = ut.assemble_params(dynamic, static)

        res = {
            'params': params,
            'loss': loss,
            'grad': grads,
            'opt': opt_state,
        }
        return res

    @ut.progress_scan(nbatches_per_epoch, message='Training model')
    def scannable_step(carry, i_x_y_z_k):
        params, opt_state = carry
        i, x, y, z, k = i_x_y_z_k
        updt = training_step(params, opt_state, x, y, z, k)
        params, opt_state = updt['params'], updt['opt']
        return (params, opt_state), updt

    @jit
    def epoch_step(start_params, start_opt_state, epoch_key, xbs, ybs):
        zbatches = jax.random.uniform(epoch_key, ybs.shape)
        batch_keys = jax.random.split(epoch_key, nbatches_per_epoch)
        (final_params, final_opt_state), epoch_history = jax.lax.scan(
            scannable_step,
            (start_params, start_opt_state),
            (jnp.arange(nbatches_per_epoch), xbs, ybs, zbatches, batch_keys),
        )
        return final_params, final_opt_state, epoch_history

    if loggers is None:
        loggers = [
            (1, console_log),
        ]

    print('Initial logger calls')
    for _, l in loggers:
        l(0, config)

    print(f'Begin training for {config["epochs"]} epochs')

    for i, epoch_key in enumerate(jax.random.split(key, config['epochs']), 1):
        t0 = time.time()

        batch_rotation = i % config['n_epochs_per_batch_rotation']
        start_idx = batch_rotation * nbatches_per_epoch
        end_idx = start_idx + nbatches_per_epoch
        params, opt_state, epoch_history = epoch_step(
            params, opt_state, epoch_key, xbatches[start_idx:end_idx], ybatches[start_idx:end_idx]
        )
        epoch_history['epoch_time'] = time.time() - t0
        for t, l in loggers:
            if i % t == 0 or i == cfg['epochs']:
                l(i, config, epoch_history=epoch_history, nbatches=nbatches_per_epoch)
