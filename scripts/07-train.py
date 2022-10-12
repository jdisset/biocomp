## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     import and init     --
# ···············································································
# %load_ext autoreload
# %autoreload 2

import streamlit as st

st.set_page_config(layout='wide')

from jax.tree_util import Partial as partial
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, value_and_grad
import biocomp as bc
import scriptutils as ut
from pathlib import Path
import json5
import sqlite3
from tqdm import tqdm
import pandas as pd


lib = ut.getLibFromGoogleSheet()

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     load xp     --
# ···············································································

base_path = Path("/Users/jeandisset/Dropbox (MIT)/Biocomp/")
base_xp_path = base_path / "Experiments"
base_recipe_path = base_path / "Recipes"

experiments = [x.name for x in base_xp_path.iterdir() if x.is_dir()]

xp = experiments[0]
xp
xpfile = base_xp_path / xp / f"{xp}.xp.json5"


xp = XP(xpfile)

xp

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                       --     build models     --
# ···············································································
# the goal is to load every sample file (and every corresponding recipe), generate the model from the recipe,
# and train all the intrinsic parameters.

# Recipe files are in base_recipe_path/recipe_name.recipe.json5

recipe_names = [s['recipe'] for s in xp.samples]
unique_recipe_names = list(set(recipe_names))
dbpath = "./test.db"
if Path(dbpath).exists():
    Path(dbpath).unlink()

dbconn = sqlite3.connect(dbpath)
bc.import_recipes_to_sql(
    [base_recipe_path / f"{r}.recipe.json5" for r in unique_recipe_names], dbconn, lib
)
networks = {recipename: bc.Network(lib, recipename, dbconn) for recipename in unique_recipe_names}
inv_networks = {k: bc.inverted_network(v) for k, v in networks.items()}

# network = networks[recipe_names[0]]
# inv_network = bc.inverted_network(network)


models = [bc.ComputeGraphModel(inv_networks[r]) for r in recipe_names]
for m in models:
    m.build()

# recipe_names: ['CasE_CoTXall', 'NW-B+pGW0010', 'L2_pGW0042+CasE-R', 'L2all_pGW42+10']
# n = networks['CasE_CoTXall']
# inv_n = bc.inverted_network(n)
# inv_n.compute_graph

# selected_recipe = st.sidebar.selectbox("Select a recipe", list(networks.keys()))
# network = inv_networks[selected_recipe]
# ut.h2(f'Recipe {selected_recipe}')
# ut.drawComputeGraph(network.compute_graph, cdg=network.central_dogma_graph)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     load data     --
# ···············································································

# # load the data files
datafiles = [base_xp_path / xp.name / 'data' / f"{s['name']}.{xp.name}.csv" for s in xp.samples]
df_data = [pd.read_csv(f) for f in tqdm(datafiles, "loading data files")]

# we want to reorder data columns to match the model's output
out_prots = [model.get_output_proteins() for model in models]
out_channels = [[xp.color_names[k] for k in out_prot] for out_prot in out_prots]

Y = [jnp.array(d[channels]) for d, channels in zip(df_data, out_channels)]
X = [model.get_input_from_output(d) for model, d in zip(models, Y)]


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

import optax
import wandb as wb

cfg = ut.ddict(
    {
        "learning_rate": 0.001,
        "adam_w_decay": 0.01,
        "clipping": 0.001,
        "n_models": 10,
        "initial_param_scaling": 0.01,
        "epochs": 1000,
        "log_rate": 50,
        "rng_key": 1,
    }
)

optimizer = optax.chain(
    optax.adaptive_grad_clip(cfg['clipping']),
    optax.adamw(learning_rate=cfg['learning_rate'], weight_decay=cfg['adam_w_decay']),
)

model = models[0]


def loss_func(params, x, y, rng_key):
    m = partial(model, params, rng_key=rng_key)
    y_hat = vmap(m)(x).squeeze()
    return jnp.sum(jnp.mean((y - y_hat) ** 2))


@jit
def training_step(params, x, y, opt_state, key):
    loss, grads = jax.value_and_grad(loss_func)(params, x, y, key)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, grads, loss


def wandb_update(loss, params, iter_num):
    wb.log({'loss': loss}, step=iter_num)
    wb.log({'params': params}, step=iter_num)


normalizer = 1.0

x = X[0] / normalizer
y = Y[0] / normalizer

key = jax.random.PRNGKey(cfg['rng_key'])
initkeys = jax.random.split(key, cfg['n_models'])
params = [model.init(key) for key in initkeys]
opt_states = [optimizer.init(p) for p in params]


params_history = []
loss_history = []


# wb.init(config=cfg, project="biocomp_000", entity="jdisset", reinit=True)

vmap_step = vmap(partial(training_step, x=x, y=y))

epochkeys = jax.random.split(key, cfg['epochs'])
for i, k in tqdm(enumerate(epochkeys), total=cfg['epochs']):
    # params, opt_state, grads, loss = training_step(params, x, y, opt_state, key)
    keys = jax.random.split(k, cfg['n_models'])
    params, opt_state, grads, loss = vmap_step(params, opt_states, keys)
    loss_history.append(loss)
    params_history.append(params)
    # if i == cfg['epochs'] or i % cfg['log_rate'] == 0 or i == 0:
    # wandb_update(loss, params, i)


import json

# save params_history to a json file:
with open('params_history.json', 'w') as f:
    json.dump(params_history, f)
