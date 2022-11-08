## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     import and init     --
# ···············································································
from jax.tree_util import Partial as partial
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, value_and_grad
from pathlib import Path
import json5
import json
import sqlite3
from tqdm import tqdm
import pandas as pd
import optax

import numpy as np
from .recipe import XP, import_recipes_to_sql
from .network import Network, inverted_network
from . import datautils as du
from . import utils as ut
import wandb as wb
import os

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


def mse_loss(y, y_hat, n_outputs=None):
    if n_outputs is None:
        n_outputs = y.shape[1]
    return jnp.mean((y[:, :n_outputs] - y_hat[:, :n_outputs]) ** 2)


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{           --      training of a single model from data     --
# ···············································································


def wandb_update(loss, params, iter_num):
    wb.log({'loss': loss}, step=iter_num)
    wb.log({'params': params}, step=iter_num)


def train_single_model(model, X, Y, cfg, loss_f=mse_loss, wandb=None):
    optimizer = optax.chain(
        optax.adaptive_grad_clip(cfg['clipping']),
        optax.adamw(learning_rate=cfg['learning_rate'], weight_decay=cfg['adam_w_decay']),
    )

    def loss_func(params, x, y, rng_key):
        m = partial(model, params, rng_key=rng_key)
        y_hat = vmap(m)(x).squeeze()
        return loss_f(y, y_hat)

    def logger_update(loss, params, iter_num):
        if wandb:
            wandb_update(loss, params, iter_num)
        else:
            print(f'[{iter_num}/{cfg["epochs"]}] loss: {loss}')

    def training_step(params, opt_state, key, x, y):
        loss, grads = jax.value_and_grad(loss_func)(params, x, y, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, grads, loss

    key = jax.random.PRNGKey(cfg['rng_key'])
    initkeys = jax.random.split(key, cfg['n_replicates'])

    params = vmap(model.init)(initkeys)
    opt_states = vmap(optimizer.init)(params)

    step = jit(vmap(partial(training_step, x=X, y=Y)))

    if wandb:
        wb.init(config=cfg, project=wandb, entity="jdisset", reinit=True)

    params_history = []
    loss_history = []

    for i, k in enumerate(jax.random.split(key, cfg['epochs'])):
        keys = jax.random.split(k, cfg['n_replicates'])
        params, opt_state, grads, loss = step(params, opt_state, keys)
        loss_history.append(loss)
        params_history.append(params)
        if i == cfg['epochs'] or i % cfg['log_rate'] == 0 or i == 0:
            logger_update(loss, params, i)

    return params_history, np.array(loss_history)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CFG = {
    "node_remap": {},
    "optimizer": "sgd",
    "learning_rate": 0.001,
    "adam_w_decay": 0.0001,
    "loss_function": "mse",
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
    "plot_rate": 100,
    "save_rate": 100,
}

DEFAULT_LOSS_FUNCTIONS = {
    "mse": mse_loss,
}


def train_xp(xp, config=DEFAULT_CFG, **kwargs):
    cfg = {**DEFAULT_CFG, **config}
    models = xp.get_models(node_remap=config['node_remap'])
    _, Y = xp.get_XY(models)
    return train_inverted_bunch(models, Y, config=config, **kwargs)


def train_inverted_bunch(
    models,
    Y_raw,
    config=DEFAULT_CFG,
    loss_f=mse_loss,
    wandb_project=None,
    loss_dict=DEFAULT_LOSS_FUNCTIONS,
    save_path='./training_results/',
):
    import matplotlib.pyplot as plt
    from rich.console import Console
    from rich.progress import track

    console = Console()

    cfg = {**DEFAULT_CFG, **config}
    rng_key = jax.random.PRNGKey(cfg['rng_key'])

    console.print(f'Starting training with config: {cfg}')

    times = ut.TimeStore(console)
    ppt = times.start('Preprocessing')

    t = times.start('Balancing data', True)

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

    t.stop_print()

    t = times.start('Generating batches', True)
    individual_batch_sizes = cfg['batch_size'] // len(models)
    x_batches, y_batches = du.make_batches_uniform_sampling(
        Y, individual_batch_sizes, rng_key, models
    )
    t.stop_print()

    console.print(f'x_batches shape: {x_batches.shape}, y_batches shape: {y_batches.shape}')

    optimizers = {
        'sgd': optax.sgd(learning_rate=cfg['learning_rate']),
        'adam': optax.adam(learning_rate=cfg['learning_rate']),
        'adamw': optax.adamw(learning_rate=cfg['learning_rate'], weight_decay=cfg['adam_w_decay']),
    }

    assert (
        cfg['optimizer'] in optimizers.keys()
    ), f"Optimizer {cfg['optimizer']} not available. Available optimizers are {optimizers.keys()}"
    optimizer = optimizers[cfg['optimizer']]

    loss_f = loss_dict[cfg['loss_function']]

    key = jax.random.PRNGKey(cfg['rng_key'])
    ikeys = jax.random.split(key, len(models))

    params = {}
    constraints = {}

    t = times.start('Initializing parameters', False)
    for s, m, k in track(
        list(zip(models.keys(), models.values(), ikeys)), description='Initializing params'
    ):
        params, constraints = m.init(k, pre_params=params, pre_constraints=constraints)
    t.stop_print()

    F = list(models.values())
    nmodels = len(F)
    n_outputs = [y.shape[1] for y in Y.values()]

    ## ───────────────────────────────────── ▼ ─────────────────────────────────────
    # {{{                       --     training step     --
    # ···············································································
    def split_params(params, static_paths):
        """Split params into static and dynamic parts."""
        # any path that is not in static_paths is dynamic
        dynamic = params.copy()
        static = {}
        for path in static_paths:
            ut.at_path(static, path, ut.at_path(dynamic, path))
            ut.delete_path(dynamic, path)

        return dynamic, static

    def assemble_params(dynamic, static):
        """Assemble params from static and dynamic parts."""
        res = ut.updated_dict(dynamic, static)
        return res

    def loss_func(dynamic, static, X, Y, rng_key):
        assert len(X) == nmodels, f"Expected {nmodels} models, got {X.shape}"
        assert len(Y) == nmodels
        params = assemble_params(dynamic, static)

        K = jax.random.split(rng_key, nmodels)

        res = jnp.array(
            [
                loss_f(vmap(partial(f, params, rng_key=k))(x), y, n_out)
                for f, x, y, k, n_out in zip(F, X, Y, K, n_outputs)
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

    #                                                                            }}}
    ## ─────────────────────────────────────────────────────────────────────────────

    ## ───────────────────────────────────── ▼ ─────────────────────────────────────
    # {{{                      --     logging methods     --
    # ···············································································

    jitted_models = {s: jit(jax.vmap(partial(m, rng_key=jax.random.PRNGKey(0)), in_axes=(None, 0))) for s, m in models.items()}

    def wandb_update(loss, params, iter_num):
        wb.log({'loss': loss}, step=iter_num)
        wb.log({'shared_params': params['shared']}, step=iter_num)

        def log_plots():
            gtruth = []
            pred = []
            for sample, f in jitted_models.items():
                model = models[sample]
                y_hat = f(params, X[sample])
                out_proteins = model.get_output_proteins()
                in_proteins = model.get_inverted_input_proteins()
                stats, bins = du.binstats(Y[sample], out_proteins, in_proteins, resolution=0.5)
                stats_hat, bins_hat = du.binstats(
                    y_hat, out_proteins, in_proteins, resolution=0.5
                )
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

    def logger_update(loss, params, iter_num):
        if wandb_project is not None:
            wandb_update(loss, params, iter_num)
        print(f'[{iter_num}/{cfg["epochs"]}] loss: {loss}')

    #                                                                            }}}
    ## ─────────────────────────────────────────────────────────────────────────────

    dynamic, _ = split_params(params, [['node']])
    opt_state = optimizer.init(dynamic)
    params_history = []
    loss_history = []

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if wandb_project is not None:
        t = times.start('Initializing wandb', True)
        wb.init(config=cfg, project=wandb_project, entity="jdisset", reinit=True)
        t.stop_print()
    else:
        console.print('No wandb project specified.')

    if save_path is None:
        console.print('Not saving results.')
    else:
        save_path = Path(save_path) / timestamp
        console.print(f'Saving results to {save_path}')
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / 'config.json', 'w') as f:
            json.dump(cfg, f)

    step = training_step

    if cfg['compile_training']:
        step = jit(step)
        t = times.start('Compiling training step', True)
        lowered = step.lower(params, opt_state, k, x_batches[0], y_batches[0])
        console.print(f'Lowered...')
        compiled = lowered.compile()
        step = compiled
        t.stop_print()

    loss = float('inf')
    ppt.stop_print()
    console.print('Training...')

    for i, k in enumerate(jax.random.split(key, cfg['epochs'])):

        for x, y in track(list(zip(x_batches, y_batches)), description=f'Epoch {i}'):
            params, opt_state, grads, loss = step(params, opt_state, k, x, y)

        if i == cfg['epochs'] or i % cfg['log_rate'] == 0 or i == 0:
            loss_history.append(loss)
            params_history.append(params)
            logger_update(loss, params, i)
            if i == cfg['epochs'] or i % cfg['save_rate'] == 0 or i == 0:
                if save_path is not None:
                    du.save(params, f'{save_path}/params_{timestamp}_epoch-{i}.pkl', overwrite=True)
                    du.save(
                        loss_history, f'{save_path}/loss_history_{timestamp}.pkl', overwrite=True
                    )
