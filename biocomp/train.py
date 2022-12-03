## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     import and init     --
# ···············································································
from jax.tree_util import Partial as partial
import jax
from datetime import datetime
import jax.numpy as jnp
from jax import jit, vmap, grad, value_and_grad
from pathlib import Path
import json5
import json
import sqlite3
from tqdm import tqdm
from typing import Callable, Dict, List, Optional, Tuple, Union
import pandas as pd
import optax
import matplotlib.pyplot as plt
from rich.console import Console
from rich.progress import track

import numpy as np
from .recipe import XP, import_recipes_to_sql
from .network import Network, inverted_network
from . import datautils as du
from . import utils as ut
from .compute import ComputeGraphModel
import wandb as wb
import os

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


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     default config     --
# ···············································································

DEFAULT_CFG = {
    "optimizer": "adam",
    "learning_rate": 0.001,
    "adam_w_decay": 0.0001,
    "loss_function": mse_loss,
    "rng_key": 42,
    "epochs": 10000,
    "n_replicates": 1,
    "compile_training": True,
    "batch_size": 128,  # per whole batch, i.e the sum of each xp's batch size
    "norm_factor": 1e6,
    "balance_bin_resolution": 0.5,
    "balance_threshold_quantile": 0.4,
    "balance_threshold_min": 40,
    "log_rate": 1,
    "node_impl": {},
    "plot_rate": 100,
    "save_rate": 100,
    "static_params": [['node']],
}

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                           --     utils   --
# ···············································································
# to serialize the node functions used into a savable config,
# we need to be able to serialize partials and functions.


def serialize_partial_or_function(field):
    if isinstance(field, partial):
        return {
            'function': field.func.__name__,
            'kwargs': field.keywords,
        }
    elif callable(field):
        return {
            'function': field.__name__,
        }
    else:
        return field


def prep_data(models, Y_raw, cfg=DEFAULT_CFG):
    X, Y = du.balance_each_dataset(
        models,
        Y_raw,
        bin_resolution=cfg['balance_bin_resolution'],
        threshold_quantile=cfg['balance_threshold_quantile'],
        threshold_min=cfg['balance_threshold_min'],
    )
    norm_factor = cfg["norm_factor"]
    X = jax.tree_map(lambda x: x / norm_factor, X)
    Y = jax.tree_map(lambda x: x / norm_factor, Y)
    return X, Y


def batch(X, Y, batch_size, n_batches=None):
    n = X.shape[0]
    if n_batches is None:
        n_batches = n // batch_size
    # using sampling with replacement
    for i in range(n_batches):
        idx = np.random.choice(n, size=batch_size, replace=True)
        yield X[idx], Y[idx]

@jax.jit
def unstack_tree(t):
    n = jax.tree_util.tree_leaves(t)[0].shape[0]
    return [jax.tree_map(lambda x: x[i], t) for i in range(n)]

def get_best_params(history, smooth_window=10):
    # find the lowest loss time point (and which replicate)
    loss = np.array(history['loss'])
    from scipy.ndimage import gaussian_filter1d
    loss_smooth = gaussian_filter1d(loss, sigma=smooth_window, axis=0)
    best_t, best_replicate = np.unravel_index(loss_smooth.argmin(), loss_smooth.shape)
    p = unstack_tree(history['params'][best_t])[best_replicate]
    return p, (best_t, best_replicate)



#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     loggers     --
# ···············································································


def wandb_log(loss, params, iter_num, cfg, **_):

    wb.log({'loss': loss}, step=iter_num)
    wb.log({'shared_params': params['shared']}, step=iter_num)

    # jitted_models = {
    # s: jit(jax.vmap(partial(m, rng_key=jax.random.PRNGKey(0)), in_axes=(None, 0)))
    # for s, m in models.items()
    # }

    wb.init(config=cfg, project=wandb_project, entity="jdisset", reinit=True)

    def log_plots():
        gtruth = []
        pred = []
        for sample, f in jitted_models.items():
            model = models[sample]
            y_hat = f(params, x[sample])
            out_proteins = model.get_output_proteins()
            in_proteins = model.get_inverted_input_proteins()
            stats, bins = du.binstats(Y[sample], out_proteins, in_proteins, resolution=0.5)
            stats_hat, bins_hat = du.binstats(y_hat, out_proteins, in_proteins, resolution=0.5)
            fig, _ = du.heatmap(
                stats,
                bins,
                figscale=0.6,
                stat_columns=['mean'],
                z_protein=(set(out_proteins) - set(in_proteins)).pop(),
                lims={'mean': (1e-5, 100)},
                title=f'{model.network.name} ground truth',
                subtitle=f'{len(y)} data points',
                show=False,
            )
            gtruth.append(wb.Image(fig, caption=f'{model.network.name} ground truth'))
            fig_hat, _ = du.heatmap(
                stats_hat,
                bins_hat,
                figscale=0.6,
                stat_columns=['mean'],
                z_protein=(set(out_proteins) - set(in_proteins)).pop(),
                lims={'mean': (1e-5, 100)},
                title=f'{model.network.name} predicted',
                subtitle=f'{len(y)} data points',
                show=False,
            )
            pred.append(wb.Image(fig_hat, caption=f'{model.network.name} predicted'))

        wb.log({'ground_truth': gtruth}, step=iter_num)
        wb.log({'predictions': pred}, step=iter_num)
        plt.close('all')

    if iter_num == cfg['epochs'] or iter_num % cfg['plot_rate'] == 0 or iter_num == 0:
        log_plots()


def console_log(loss, params, epoch, cfg, **_):
    print(f'[{epoch}/{cfg["epochs"]}] loss: {loss:.5f}')


def save_log(loss, params, iter_num, cfg, save_path=None, loss_history=None, **_):
    save_path = cfg.get('save_path', save_path)
    assert save_path is not None, 'save_path must be specified'

    if iter_num == 0:
        print(f'Saving results to {save_path}')
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / 'config.json', 'w') as f:
            json.dump(cfg, f)

    du.save(params, f'{save_path}/params_epoch-{iter_num}.pkl', overwrite=True)
    if loss_history is not None:
        du.save(loss_history, f'{save_path}/loss_history.pkl', overwrite=True)


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

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                       --     train from xp     --
# ···············································································


def train_xp(xp, config=DEFAULT_CFG, **kwargs):
    console = Console()
    cfg = {**DEFAULT_CFG, **config}
    # ser = json.dumps(config, indent=4, default=serialize_partial_or_function)
    rng_key = jax.random.PRNGKey(cfg['rng_key'])

    console.print(f'Starting training with config: {cfg}')
    times = ut.TimeStore(console)

    models = xp.get_models(node_impl=config['node_impl'])

    _, Y = xp.get_XY(models)

    t = times.start('Balancing data', True)
    X, Y = prep_data(models, Y, cfg)
    t.stop_print()

    t = times.start('Generating batches', True)
    individual_batch_sizes = cfg['batch_size'] // len(models)
    x_batches, y_batches = du.make_batches_uniform_sampling(
        Y, individual_batch_sizes, rng_key, models
    )
    t.stop_print()
    console.print(f'x_batches shape: {x_batches.shape}, y_batches shape: {y_batches.shape}')

    return train_models(models, Y, config=config, **kwargs)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{               --     train models (no replicates)     --
#···············································································
def train_models(
    models: list[ComputeGraphModel],
    x_batches: np.ndarray,
    y_batches: np.ndarray,
    config: dict = DEFAULT_CFG,
    loggers: dict[int, Callable] = None,
):

    cfg = {**DEFAULT_CFG, **config}

    optimizers = {
        'sgd': optax.sgd(learning_rate=cfg['learning_rate']),
        'adam': optax.adam(learning_rate=cfg['learning_rate']),
        'adamw': optax.adamw(learning_rate=cfg['learning_rate'], weight_decay=cfg['adam_w_decay']),
    }
    assert (
        cfg['optimizer'] in optimizers.keys()
    ), f"Optimizer {cfg['optimizer']} not available. Available optimizers are {optimizers.keys()}"
    optimizer = optimizers[cfg['optimizer']]

    loss_f = cfg['loss_function']
    assert callable(loss_f), f"loss_f must be callable, not {type(loss_f)}"

    key = jax.random.PRNGKey(cfg['rng_key'])
    ikeys = jax.random.split(key, len(models))
    params, constraints = {}, {}

    print('Initializing parameters')
    for m, k in zip(models, ikeys):
        params, constraints = m.init(k, pre_params=params, pre_constraints=constraints)

    dynamic, _ = split_params(params, cfg['static_params'])
    opt_state = optimizer.init(dynamic)
    params_history, loss_history, grad_history, opt_history, loss = [], [], [], [], float('inf')

    ## ───────────────────────────────────── ▼ ─────────────────────────────────────
    # {{{                       --     training step     --
    # ···············································································

    def loss_func(dynamic, static, X, Y, rng_key):
        nmodels = len(models)
        assert len(X) == nmodels, f"Expected {nmodels} models, got {X.shape}"
        assert len(Y) == nmodels
        params = assemble_params(dynamic, static)

        K = jax.random.split(rng_key, nmodels)

        res = jnp.array(
            [
                loss_f(vmap(partial(m, params, rng_key=k))(x), y, m.n_outputs)
                for m, x, y, k in zip(models, X, Y, K)
            ]
        ).mean()

        return res

    def training_step(params, opt_state, key, x, y):
        dynamic, static = split_params(params, [['node']])
        loss, grads = jax.value_and_grad(loss_func)(dynamic, static, x, y, key)
        updates, opt_state = optimizer.update(grads, opt_state, dynamic)

        dynamic = optax.apply_updates(dynamic, updates)
        dynamic = ut.apply_constraints(dynamic, constraints)
        params = assemble_params(dynamic, static)

        return params, opt_state, grads, loss

    step = training_step

    if cfg['compile_training']:
        import time

        print('Compiling training step')
        t0 = time.time()
        step = jit(step)
        lowered = step.lower(params, opt_state, k, x_batches[0], y_batches[0])
        compiled = lowered.compile()
        step = compiled
        print(f'Compiled in {time.time() - t0:.2f}s')

    #                                                                            }}}
    ## ─────────────────────────────────────────────────────────────────────────────


    if loggers is None:
        loggers = {}

    for l in loggers.values():
        l(loss, params, 0, cfg)

    print('Beginning training')
    for i, k in enumerate(jax.random.split(key, cfg['epochs']), 1):
        for x, y in zip(x_batches, y_batches):
            params, opt_state, grads, loss = step(params, opt_state, k, x, y)
            loss_history.append(loss)
            params_history.append(params)
            grad_history.append(grads)
            opt_history.append(opt_state)

        for t, l in loggers.items():
            if i % t == 0 or i == cfg['epochs']:
                l(loss, params, i, cfg, loss_history=loss_history, params_history=params_history)

    return params_history, loss_history, grad_history, opt_history


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{          --     train single model but with replicates     --
#···············································································
def train_model(model, x, y, config, loggers=None):
    cfg = {**DEFAULT_CFG, **config}
    loggers = loggers or {}
    optimizer = optax.sgd(learning_rate=cfg['learning_rate'])
    key = jax.random.PRNGKey(cfg['rng_key'])
    repl_keys = jax.random.split(key, cfg['n_replicates'])

    params, constraints = jax.vmap(model.init)(repl_keys)
    dynamic, _ = ut.split_params(params, cfg['static_params'])
    opt_states = optimizer.init(dynamic)

    history = {
        'params': [params],
        'opt': [opt_states],
        'grad': [],
        'loss': [],
    }


    def training_step(params, opt_states, x, y):
        def loss_func(dynamic, static, x, y):
            params = ut.assemble_params(dynamic, static)
            y_hat = jax.vmap(partial(model, params, rng_key=key))(x)
            assert y_hat.shape == y.shape
            return jnp.mean((y - y_hat) ** 2)

        dynamic, static = ut.split_params(params, cfg['static_params'])

        loss, grads = jax.vmap(jax.value_and_grad(loss_func), in_axes=(0, 0, None, None))(dynamic, static, x, y)
        updates, opt_states = optimizer.update(grads, opt_states, dynamic)

        dynamic = optax.apply_updates(dynamic, updates)
        # dynamic = ut.apply_constraints(dynamic, constraints)
        params = ut.assemble_params(dynamic, static)

        res = {
            'params': params,
            'loss': loss,
            'grad': grads,
            'opt': opt_states,
        }
        return res

    step = jax.jit(training_step)

    print('Beginning training')

    n_batches = cfg.get('n_batches', x.shape[0] // cfg['batch_size'])

    for i, k in enumerate(jax.random.split(key, cfg['epochs']), 1):

        for x_batch, y_batch in batch(x, y, cfg['batch_size'], n_batches):
            updt = step(history['params'][-1], history['opt'][-1], x_batch, y_batch)
            history = {k: v + [updt[k]] for k, v in history.items()}

        for t, l in loggers.items():
            if i % t == 0 or i == cfg['epochs']:
                l(history, i, cfg)
    return history


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────






