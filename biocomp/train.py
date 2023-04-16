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
import joblib
from joblib import Parallel, delayed
from . import datautils as du
from . import utils as ut
from . import nodes as nodes
from . import nodes_old as nodes_old
from . import defaults as dft

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
    if 'grad' in epoch_data:
        for k, v in epoch_data['grad']['shared'].items():
            stats['grad'][k] = compstats(v)
    if 'params' in epoch_data:
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

    # params = ut.tree_get(epoch_history['params'], -1)
    params = epoch_history['latest_params']
    loss = np.array(epoch_history['loss'])

    # first we rename the old params
    for f in Path(save_dir).glob('latest_params.pkl'):
        f.rename(f'{save_dir}/old_params.pkl')

    # then we save the new ones
    du.save(params, f'{save_dir}/latest_params.pkl')

    # then we delete the old one
    for f in Path(save_dir).glob('old_params.pkl'):
        f.unlink()

    ut.logger.info(f"Saving epoch to disk took {time.time() - t0:.2f}s")


def wandb_plot_pred(epoch, cfg, dman, epoch_history=None, **_):
    if epoch_history is None:
        return

    t0 = time.time()
    params = epoch_history['latest_params']
    pred = []
    networks = dman.get_networks()
    try:
        for i, net in enumerate(networks):
            fig, ax = du.report(params, dman, i)
            pred.append(wb.Image(fig, caption=f'{net.name}'))
            plt.close(fig)
    except Exception as e:
        ut.logger.warning(f"Failed to plot predictions: {e}")
        # print a stack trace
        import traceback
        traceback.print_exc()

    wb.log({'Evaluations': pred})
    ut.logger.info(f'Done logging prediction plots for epoch {epoch} in {time.time() - t0:.2f}s')


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
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        fmt = lambda x: f'{x:.1e}' if x < 1e-3 or x > 1e3 else f'{x:.3f}'
        ut.logger.info(
            f"""[{epoch}/{cfg["epochs"]}] \
        loss: {fmt(avg)} ± {fmt(std)} [min {fmt(lmin)}, max {fmt(lmax)}] in \
        {epoch_history["epoch_time"]:.2f}s"""
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

    learning_rate = cfg['learning_rate']

    if 'schedule' in cfg:
        if cfg['schedule'] == 'cosine':
            steps_per_epoch = cfg['steps_per_epoch']
            max_learning_rate = cfg['learning_rate']
            warmup_steps = cfg['warmup_epochs'] * steps_per_epoch
            decay_steps = cfg['decay_epochs'] * steps_per_epoch
            end_learning_rate = cfg['end_learning_rate']
            learning_rate = optax.warmup_cosine_decay_schedule(
                init_value=1e-9,
                peak_value=max_learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
                end_value=end_learning_rate,
            )
        elif cfg['schedule'] == 'constant':
            learning_rate = cfg['learning_rate']
        else:
            raise ValueError(f"Unknown learning rate schedule {cfg['schedule']}")

    optimizers = {
        'sgd': optax.sgd(learning_rate=learning_rate),
        'adamw': optax.adamw(learning_rate=learning_rate, weight_decay=cfg['adam_w_decay']),
        'adam': optax.adam(learning_rate=learning_rate),
        'amsgrad': optax.amsgrad(learning_rate=learning_rate),
    }
    assert (
        cfg['optimizer'] in optimizers.keys()
    ), f"Optimizer {cfg['optimizer']} not available. Available optimizers are {optimizers.keys()}"
    optimizer = optimizers[cfg['optimizer']]

    return optimizer


def setup_wandb_logging(
    project,
    dman,
    training_config,
    compute_config,
    plot_period=1,
    params_save_period=100,
    entity='jdisset',
    **kw,
):
    import wandb as wb

    full_config = {**training_config, **compute_config.get_config()}

    wb.init(config=full_config, project=project, entity=entity, **kw)
    save_dir = Path(wb.run.dir)
    loggers = [
        (1, console_log),
        (params_save_period, partial(local_save, save_dir=save_dir)),
        (1, wandb_log_epoch),
        (plot_period, partial(wandb_plot_pred, dman=dman)),
    ]
    return loggers


def get_memory(config):
    cache_folder = config.get("cache_folder", None)
    if cache_folder:
        cache_folder = Path(cache_folder).expanduser()
        cache_folder.mkdir(parents=True, exist_ok=True)
        return joblib.Memory(cache_folder, verbose=0)
    else:
        return joblib.Memory(None, verbose=0)


def start(dman: du.DataManager, training_config, compute_config, loggers=None, key=None):

    if key is None:
        key = jax.random.PRNGKey(training_config['rng_key'])
    else:
        ut.logger.info(f"Using key {key} for training")

    # --- cached init & batches generation
    memory = get_memory(training_config)

    @memory.cache
    def init_stack(dman, key):
        stack = dman.build_compute_stack(compute_config)
        with ut.timer('Stack initialization'):
            params = stack.init(key)
        return stack, params

    @memory.cache
    def generate_batches(dman, key):
        with ut.timer('Generating batches'):
            xbatches, ybatches = dman.get_batches(key)  # (B,M,N,F) shape
        return xbatches, ybatches

    stack, params = init_stack(dman, key)
    xbatches, ybatches = generate_batches(dman, key)
    optimizer = get_optimizer(training_config)
    dynamic, _ = ut.split_params(params, training_config['static_params'])
    opt_state = optimizer.init(dynamic)
    total_batches = training_config['n_batches']
    assert total_batches == xbatches.shape[0] == ybatches.shape[0]
    steps_per_epoch = max(1, int(training_config['steps_per_epoch']))

    # --- loss & update functions

    vmapped_compute = jax.vmap(stack.apply, in_axes=(None, 0, 0, 0))

    def loss_func(dynamic, static, X, Y, Z, key):
        assert X.ndim == Y.ndim == Z.ndim == 2
        assert X.shape[0] == Y.shape[0] == Z.shape[0]
        assert X.shape[1] == sum([n.get_nb_inputs() for n in stack.networks])
        assert Y.shape[1] == Z.shape[1] == sum([n.get_nb_outputs() for n in stack.networks])

        params = ut.assemble_params(dynamic, static)
        keys = jax.random.split(key, X.shape[0])

        yhat, grads = vmapped_compute(params, X, Z, keys)
        assert yhat.shape == Y.shape

        error = yhat - Y
        quantile_loss = jnp.mean(
            huber_quantile_loss(error, Z, delta=training_config['huber_quantile_loss_delta'])
        )

        # grads is the concatenated and flattened jacobian of
        # translate, transcript, and output nodes wrt their inputs
        # they should be monotonically increasing so we add a loss term
        negative_grads = jnp.mean(jnp.where(grads < 0, -grads, 0))

        return quantile_loss + training_config['negative_grad_penalty'] * negative_grads

    def training_step(params, opt_state, x, y, z, key):
        dynamic, static = ut.split_params(params, training_config['static_params'])
        loss, grads = value_and_grad(loss_func)(dynamic, static, x, y, z, key)
        updates, opt_state = optimizer.update(grads, opt_state, dynamic)

        dynamic = optax.apply_updates(dynamic, updates)
        params = ut.assemble_params(dynamic, static)

        res = {
            'params': params,
            'loss': loss,
            'grad': grads,
            'opt': opt_state,
        }
        return res

    keep_in_history = training_config.get('keep_in_history', ['loss'])

    @ut.progress_scan(steps_per_epoch, message='Training model')
    def scannable_step(carry, i_x_y_z_k):
        params, opt_state = carry
        i, x, y, z, k = i_x_y_z_k
        updt = training_step(params, opt_state, x, y, z, k)
        params, opt_state = updt['params'], updt['opt']
        history = {k: updt[k] for k in keep_in_history}
        return (params, opt_state), history

    @jit
    def epoch_step(start_params, start_opt_state, epoch_key, xbs, ybs):
        zbatches = jax.random.uniform(epoch_key, ybs.shape)
        batch_keys = jax.random.split(epoch_key, steps_per_epoch)
        (final_params, final_opt_state), epoch_history = jax.lax.scan(
            scannable_step,
            (start_params, start_opt_state),
            (jnp.arange(steps_per_epoch), xbs, ybs, zbatches, batch_keys),
        )
        return final_params, final_opt_state, epoch_history

    # --- main training loop

    if loggers is None:
        loggers = [(1, console_log)]

    for _, l in loggers:
        l(0, training_config)

    ut.logger.info(f'Begin training for {training_config["epochs"]} epochs')

    def get_slice(a, start, end):
        offset = start // a.shape[0]
        start = start % a.shape[0]
        end = end - offset * a.shape[0]
        if end > a.shape[0]:  # loop around
            return jnp.concatenate([a[start:], get_slice(a, 0, end - a.shape[0])])
        else:
            return a[start:end]

    for i, epoch_key in enumerate(jax.random.split(key, training_config['epochs']), 1):
        t0 = time.time()
        xb = get_slice(xbatches, i * steps_per_epoch, (i + 1) * steps_per_epoch)
        yb = get_slice(ybatches, i * steps_per_epoch, (i + 1) * steps_per_epoch)
        params, opt_state, epoch_history = epoch_step(params, opt_state, epoch_key, xb, yb)
        epoch_history['epoch_time'] = time.time() - t0
        epoch_history['latest_params'] = params

        for t, l in loggers:
            if i % t == 0 or i == training_config['epochs']:
                l(i, training_config, epoch_history=epoch_history, nbatches=steps_per_epoch)

    return params, epoch_history
