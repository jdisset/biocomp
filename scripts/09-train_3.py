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
#···············································································




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

models = xp.get_models()
X, Y = xp.get_XY(models)
X, Y = du.balance_each_dataset(models, Y)

import wandb as wb

cfg = {
    "learning_rate": 0.005,
    "adam_w_decay": 0.001,
    "clipping": 0.001,
    "n_replicates": 1,
    "initial_param_scaling": 0.01,
    "normalize_data": False,
    "epochs": 10000,
    "log_rate": 1,
    "save_rate": 100,
    "rng_key": 42131,
}


optimizer = optax.chain(
    # optax.adaptive_grad_clip(cfg['clipping']),
    optax.adamw(learning_rate=cfg['learning_rate'], weight_decay=cfg['adam_w_decay']),
)

key = jax.random.PRNGKey(cfg['rng_key'])
ikeys = jax.random.split(key, len(models))
params = {}

for s, m, k in zip(models.keys(), models.values(), ikeys):
    params = m.init(k, pre_params=params, node_namespace=s)


def mse_loss(y, y_hat):
    return jnp.mean((y - y_hat)**2)

loss_f = mse_loss

# jitted_models = {k: jit(v) for k, v in models.items()}

def apply_model(params, x, key, name):
    m = partial(models[name], params, node_namespace=name, rng_key=key)
    return vmap(m)(x[name]).squeeze()


@jit
def loss_func(params, x, y, rng_key):
    print('loss func')
    nmodels = len(models)
    ikeys = jax.random.split(rng_key, nmodels)
    losses = 0
    for sample, model, k in zip(models.keys(), models.values(), ikeys):
        y_hat = apply_model(params, x, k, sample)
        losses += loss_f(y[sample], y_hat)

    return losses / nmodels

def wandb_update(loss, params, iter_num):
    wb.log({'loss': loss}, step=iter_num)
    wb.log({'params': params}, step=iter_num)

def logger_update(loss, params, iter_num):
    wandb_update(loss, params, iter_num)
    print(f'[{iter_num}/{cfg["epochs"]}] loss: {loss}')


def training_step(params, opt_state, key, x, y):
    loss, grads = jax.value_and_grad(loss_func)(params, x, y, key)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, grads, loss

opt_state = optimizer.init(params)

params_history = []
loss_history = []


def make_batches(X, Y, n_batches):
    x_ = {s: np.array_split(X[s], n_batches) for s in X.keys()}
    x_batches = [{s: x_[s][i] for s in x_.keys()} for i in range(n_batches)]
    y_ = {s: np.array_split(Y[s], n_batches) for s in Y.keys()}
    y_batches = [{s: y_[s][i] for s in y_.keys()} for i in range(n_batches)]
    return x_batches, y_batches

n_batches = 2
x_batches, y_batches = make_batches(X, Y, n_batches)

# step = jit(training_step)
step = training_step

from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

wb.init(config=cfg, project='georg_data_2', entity="jdisset", reinit=True)

for i, k in enumerate(jax.random.split(key, cfg['epochs'])):
    for x, y in tqdm(list(zip(x_batches, y_batches))):
        params, opt_state, grads, loss = step(params, opt_state, k, x, y)
    loss_history.append(loss)
    params_history.append(params)
    if i == cfg['epochs'] or i % cfg['log_rate'] == 0 or i == 0:
        logger_update(loss, params, i)
        if i == cfg['epochs'] or i % cfg['save_rate'] == 0 or i == 0:
            ut.save(params_history, f'../__out/{xp.name}_{timestamp}_{i}_params.pkl', overwrite=True)
            ut.save(loss_history, f'../__out/{xp.name}_{timestamp}_{i}_loss.pkl', overwrite=True)

print('done')

# params_history
# TODO:
# 1 - fix the very slow compilation of the training step. Right now it takes forever to compile (because of the for loop that has all the apply_model calls)
# AND it recompiles for every different batch size. The batch size should be relatively easy to fix if we make sure that the batch size is the same for all samples. (not even sure that'll actually fix it)
# then there has to be a better way to handle all the models jitting

# 2 - need a way to switch implementations:
# we want to be able to switch both which compute node is being used (sequestron_ERN_v1, v2, sequestron_ERN_affinity, etc) for a same layout AND also use different sequestrons entirely (that might actually have different input and output species)
# there might be a way to do both with the same mechanism or similar mechanisms. 
# something to enable/disable sequestrons from the lib, and a possible remapping/renaming of the sequestron_node. 
# Ex: 
# lib.disable_sequestron_type('sequestron_ERN') # or using a list of names to enable?
# lib.enable_sequestron_type('sequestron_ERN_dna')
# and then at the network (or model level? network is probably better), provide a node_remap dict to the build function. Ex: n = Network(..., node_remap={'sequestron_ERN': 'sequestron_ERN_v2'})


