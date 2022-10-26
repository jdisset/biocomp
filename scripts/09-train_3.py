## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                 --     import and init     --
# ···············································································
import streamlit as st

st.set_page_config(layout='wide')

from jax.tree_util import Partial as partial
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, value_and_grad
import biocomp as bc
import biocomp.utils as bu
import scriptutils as ut
import datautils as du
from pathlib import Path
import json5
import json
import sqlite3
from tqdm import tqdm
import pandas as pd
import numpy as np
import logging

# set to debug
logging.basicConfig(level=logging.DEBUG)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     get lib     --
# ···············································································


def get_lib():
    return ut.getLibFromGoogleSheet()


lib = get_lib()
# ut.save(lib, '/tmp/lib.pickle', overwrite=True)

# lib = ut.load('/tmp/lib.pickle')


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

# selected_recipe = st.sidebar.selectbox("Select a recipe", list(networks.keys()))


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     load xp     --
# ···············································································


base_path = Path("/Users/jeandisset/Dropbox (MIT)/Biocomp/")
xp_path = base_path / "Experiments"
recipe_path = base_path / "Recipes"

xpnames = [x.name for x in xp_path.iterdir() if x.is_dir()]
# xps = {x: bc.XP(x, xp_path, recipe_path, lib) for x in xpnames}

georgXP = '20221012A_massCtrls'
xp = bc.XP(georgXP, xp_path, recipe_path, lib)

# charles xp:
# - put csv in a data folder
# - dots should be escaped in field names
# - having parts named identical to L0s is weird and potentially confusing?
# - L1 Phic31 has only eYFP?

# all recipe paths:
# allrecipes = list(recipe_path.glob('**/*.json5'))
# dbconn = sqlite3.connect(":memory:")
# bc.import_recipes_to_sql(allrecipes, dbconn, lib)

# ut.plot_cdg([attNG], ['../__out/attNG_cdg.pdf'])
# ut.plot_networks([attNG], ['../__out/attNG_network.pdf'])
# ut.plot_networks([attNG]*3, ['../__out/attNG_network.pdf']*3)

# let's plot all networks from the xp

# outfiles = [f'../__out/{xp.name}_{r}.pdf' for r, n in xp.networks.items()]
# ut.plot_networks(xp.networks.values(), outfiles)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

import optax
from matplotlib import pyplot as plt

node_remap = {"sequestron_ERN": "ERN_with_affinity"}

models = xp.get_models(node_remap=node_remap)
X, Y = xp.get_XY(models)
X, Y = du.balance_each_dataset(models, Y)

# divide both X and Y by a constant factor

import io
import pickle



import wandb as wb

cfg = {
    "learning_rate": 0.01,
    "adam_w_decay": 0.0001,
    "n_replicates": 1,
    "initial_param_scaling": 0.01,
    "normalize_data": False,
    "epochs": 10000,
    "log_rate": 1,
    "n_batches": 1,
    "save_rate": 100,
    "rng_key": 1421,
    "norm_factor": 1e6,
}

norm_factor = cfg["norm_factor"]
X = jax.tree_map(lambda x: x / norm_factor, X)
Y = jax.tree_map(lambda x: x / norm_factor, Y)

m = models.values().__iter__().__next__()
m.network.compute_graph.loc[27].extra

optimizer = optax.chain(
    # optax.adamw(learning_rate=cfg['learning_rate'], weight_decay=cfg['adam_w_decay']),
    optax.sgd(learning_rate=cfg['learning_rate']),
)

key = jax.random.PRNGKey(cfg['rng_key'])
ikeys = jax.random.split(key, len(models))
params = {}

for s, m, k in zip(models.keys(), models.values(), ikeys):
    params = m.init(k, pre_params=params, node_namespace=s)


def mse_loss(y, y_hat):
    return jnp.mean((y - y_hat) ** 2)


loss_f = mse_loss


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


def wandb_update(loss, params, iter_num):
    wb.log({'loss': loss}, step=iter_num)
    wb.log({'shared_params': params['shared']}, step=iter_num)

    def log_plots():
        for sample, model in models.items():
            y_hat = apply_model(params, X, jax.random.PRNGKey(0), sample)
            out_proteins = model.get_output_proteins()
            in_proteins = model.get_inverted_input_proteins()

            y = Y[sample]
            stats, bins = du.binstats(y, out_proteins, in_proteins, resolution=0.5)
            fig, ax = du.heatmap(
                    stats,
                    bins,
                    figscale=0.6,
                    stat_columns=['mean'],
                    z_protein=(set(out_proteins) - set(in_proteins)).pop(),
                    lims={'mean': (1e-5, 100)},
                    title=f'{model.network.name}',
                    subtitle=f'{len(y)} data points',
                    show = False
                )

            stats_hat, bins_hat = du.binstats(y_hat, out_proteins, in_proteins, resolution=0.5)
            fig_hat, ax_hat = du.heatmap(
                    stats_hat,
                    bins_hat,
                    figscale=0.6,
                    stat_columns=['mean'],
                    z_protein=(set(out_proteins) - set(in_proteins)).pop(),
                    lims={'mean': (1e-5, 100)},
                    title=f'{model.network.name}',
                    subtitle=f'{len(y)} data points',
                    show = False
                )

            wb.log({f'{sample}_data': wb.Image(fig)}, step=iter_num)
            wb.log({f'{sample}_predicted': wb.Image(fig_hat)}, step=iter_num)

    i = iter_num
    if i == cfg['epochs'] or i % cfg['save_rate'] == 0 or i == 0:
        log_plots()




def logger_update(loss, params, iter_num):
    wandb_update(loss, params, iter_num)
    print(f'[{iter_num}/{cfg["epochs"]}] loss: {loss}')


def training_step(params, opt_state, key, x, y):
    print('compiling training step...')
    loss, grads = jax.value_and_grad(loss_func)(params, x, y, key)
    updates, opt_state = optimizer.update(grads, opt_state, params)

    # don't update node parameters
    updates['node'] = jax.tree_map(lambda x: jnp.zeros_like(x), updates['node'])

    params = optax.apply_updates(params, updates)
    return params, opt_state, grads, loss


def make_batches(X, Y, n_batches):
    x_ = {s: np.array_split(X[s], n_batches) for s in X.keys()}
    x_batches = [{s: x_[s][i] for s in x_.keys()} for i in range(n_batches)]
    y_ = {s: np.array_split(Y[s], n_batches) for s in Y.keys()}
    y_batches = [{s: y_[s][i] for s in y_.keys()} for i in range(n_batches)]
    return x_batches, y_batches



opt_state = optimizer.init(params)
params_history = []
loss_history = []


x_batches, y_batches = make_batches(X, Y, cfg['n_batches'])

step = jit(training_step)

# step = training_step

from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

wb.init(config=cfg, project='georg_data_2', entity="jdisset", reinit=True)

for i, k in enumerate(jax.random.split(key, cfg['epochs'])):
    for x, y in list(zip(x_batches, y_batches)):
        params, opt_state, grads, loss = step(params, opt_state, k, x, y)
    loss_history.append(loss)
    params_history.append(params)
    if i == cfg['epochs'] or i % cfg['log_rate'] == 0 or i == 0:
        logger_update(loss, params, i)
        if i == cfg['epochs'] or i % cfg['save_rate'] == 0 or i == 0:
            ut.save(
                params_history, f'../__out/{xp.name}_{timestamp}_{i}_params.pkl', overwrite=True
            )
            ut.save(loss_history, f'../__out/{xp.name}_{timestamp}_{i}_loss.pkl', overwrite=True)

print('done')

# # open params:
# params = ut.load('../__out/20221012A_massCtrls_20221026_024328_1800_params.pickle')

# ##
# params = params[-1]
# for sample, model in models.items():
    # y_hat = apply_model(params, X, jax.random.PRNGKey(0), sample)
    # out_proteins = model.get_output_proteins()
    # in_proteins = model.get_inverted_input_proteins()
    # y = Y[sample]
    # stats, bins = du.binstats(y, out_proteins, in_proteins, resolution=0.5)
    # fig, ax = du.heatmap(
            # stats,
            # bins,
            # figscale=0.6,
            # stat_columns=['mean'],
            # z_protein=(set(out_proteins) - set(in_proteins)).pop(),
            # lims={'mean': (1e0, 1e8)},
            # title=f'{model.network.name} data',
            # subtitle=f'{len(y)} data points',
            # show = False
        # )
    # stats_hat, bins_hat = du.binstats(y_hat, out_proteins, in_proteins, resolution=0.5)
    # fig_hat, ax_hat = du.heatmap(
            # stats_hat,
            # bins_hat,
            # figscale=0.6,
            # stat_columns=['mean'],
            # z_protein=(set(out_proteins) - set(in_proteins)).pop(),
            # lims={'mean': (1e0, 1e8)},
            # title=f'{model.network.name} predicted',
            # subtitle=f'{len(y)} data points',
            # show = False
        # )

# ##
# sample = models.keys().__iter__().__next__()
# sample
# model = models[sample]
# x = X[sample]
# y_hat = apply_model(params, X, jax.random.PRNGKey(0), sample)

# # model.apply(params, x[0], key, node_namespace=sample)
# yh, res = model.collect_all_results(params, x[0], key, node_namespace=sample)


# # convert to numpy
# p = jax.tree_map(lambda x: x.tolist(), params)
# json.dumps(p)


# networks = list(xp.inv_networks.values())
# filenames = [f'../__out/nets/inv_{n.name}.pdf' for n in networks]
# ut.plot_networks(networks, filenames)

