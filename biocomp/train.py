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
import wandb as wb
import os

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{           --      training of a single model from data     --
# ···············································································


def mse_loss(y, y_hat):
    return jnp.mean((y - y_hat) ** 2)


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

    "epochs": 100000,
    "n_replicates": 1,

    "n_batches": 1,
    "norm_factor": 1e6,
    "balance_bin_resolution": 0.5,
    "balance_threshold_quantile": 0.4,
    "balance_threshold_min": 20,

    "log_rate": 100,
    "plot_rate": 10000,
    "save_rate": 10000,
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
    import rich

    cfg = {**DEFAULT_CFG, **config}
    rich.print(f'Starting training with config: {cfg}')

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

    for s, m, k in zip(models.keys(), models.values(), ikeys):
        params = m.init(k, pre_params=params, node_namespace=s)

    def apply_model(params, x, key, name):
        m = partial(models[name], params, node_namespace=name, rng_key=key)
        return vmap(m)(x[name]).squeeze()

    def loss_func(params, x, y, rng_key):
        nmodels = len(models)
        ikeys = jax.random.split(rng_key, nmodels)
        return jnp.array(
            [
                loss_f(apply_model(params, x, k, sample), y[sample])
                for sample, k in zip(models.keys(), ikeys)
            ]
        ).mean()

    def make_batches(X, Y, n_batches):
        x_ = {s: np.array_split(X[s], n_batches) for s in X.keys()}
        x_batches = [{s: x_[s][i] for s in x_.keys()} for i in range(n_batches)]
        y_ = {s: np.array_split(Y[s], n_batches) for s in Y.keys()}
        y_batches = [{s: y_[s][i] for s in y_.keys()} for i in range(n_batches)]
        return x_batches, y_batches

    x_batches, y_batches = make_batches(X, Y, cfg['n_batches'])


    def training_step(params, opt_state, key, x, y):
        print('compiling training step...')
        loss, grads = jax.value_and_grad(loss_func)(params, x, y, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)

        # don't update node parameters
        updates['node'] = jax.tree_map(lambda x: jnp.zeros_like(x), updates['node'])

        params = optax.apply_updates(params, updates)
        return params, opt_state, grads, loss

    def wandb_update(loss, params, iter_num):
        wb.log({'loss': loss}, step=iter_num)
        wb.log({'shared_params': params['shared']}, step=iter_num)

        def log_plots():
            if iter_num == 0:
                gtruth = []
                for sample, model in models.items():
                    y_hat = apply_model(params, X, jax.random.PRNGKey(0), sample)
                    out_proteins = model.get_output_proteins()
                    in_proteins = model.get_inverted_input_proteins()
                    stats, bins = du.binstats(Y[sample], out_proteins, in_proteins, resolution=0.5)
                    fig, ax = du.heatmap(
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

                wb.log({'ground_truth': gtruth}, step=iter_num)

            pred = []
            for sample, model in models.items():
                y_hat = apply_model(params, X, jax.random.PRNGKey(0), sample)
                out_proteins = model.get_output_proteins()
                in_proteins = model.get_inverted_input_proteins()
                stats_hat, bins_hat = du.binstats(y_hat, out_proteins, in_proteins, resolution=0.5)
                fig_hat, ax_hat = du.heatmap(
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

            wb.log({'predictions': pred}, step=iter_num)

        if iter_num == cfg['epochs'] or iter_num % cfg['plot_rate'] == 0 or iter_num == 0:
            log_plots()

    def logger_update(loss, params, iter_num):
        if wandb_project is not None:
            wandb_update(loss, params, iter_num)
        print(f'[{iter_num}/{cfg["epochs"]}] loss: {loss}')


    opt_state = optimizer.init(params)
    params_history = []
    loss_history = []

    step = jit(training_step)

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if wandb_project is not None:
        wb.init(config=cfg, project=wandb_project, entity="jdisset", reinit=True)
    else:
        print('No wandb project specified.')

    if save_path is None:
        print('Not saving results.')
    else:
        save_path = Path(save_path) / timestamp
        print(f'Saving results to {save_path}')
        save_path.mkdir(parents=True, exist_ok=True)
        # save config:
        with open(save_path / 'config.json', 'w') as f:
            json.dump(cfg, f)


    for i, k in enumerate(jax.random.split(key, cfg['epochs'])):
        for x, y in list(zip(x_batches, y_batches)):
            params, opt_state, grads, loss = step(params, opt_state, k, x, y)

        if i == cfg['epochs'] or i % cfg['log_rate'] == 0 or i == 0:
            loss_history.append(loss)
            params_history.append(params)

            logger_update(loss, params, i)
            if i == cfg['epochs'] or i % cfg['save_rate'] == 0 or i == 0:
                if save_path is not None:
                    du.save(params, f'{save_path}/params_{timestamp}_epoch-{i}.pkl', overwrite=True)
                    du.save(loss_history, f'{save_path}/loss_history_{timestamp}.pkl', overwrite=True)

