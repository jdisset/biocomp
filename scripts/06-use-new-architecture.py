## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     import and init     --
# ···············································································
# %load_ext autoreload
# %autoreload 2

from scipy.special import ellipkinc
import streamlit as st

st.set_page_config(layout='wide')

import pandas as pd
import numpy as np
import jax.numpy as jnp
import sqlite3
import os

from collections import defaultdict
import jax
from jax import jit, vmap, grad
import scriptutils as ut
import biocomp.utils as bu
from functools import partial
import biocomp as bc
import json
from rich import print
from pathlib import Path

# l = ut.load("../biocomp/test_data/all_sheets.pickle")
# lib = bc.PartsLibrary(l.parts, l.L0s, l.L1s, l.L2s, l.categories, l.sequestrons, l.sequestron_types)
lib = ut.getLibFromGoogleSheet()

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

print(lib)

# TODO:

# write all the compute functions using the prototypes
# in script-05 and commit all that to the compute module.

# then try things here!
# - create network from recipe
# - build model, try to compute things
# - load xp data, train model
# - ...
# - profit

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{               --     load recipe and build network     --
# ···············································································
recipe_path = "../biocomp/test_data/recipe00.json5"
dbpath = "./test.db"
if Path(dbpath).exists():
    os.remove(dbpath)

dbconn = sqlite3.connect(dbpath)
bc.import_recipes_to_sql([recipe_path], dbconn, lib)

c = dbconn.cursor()
c.execute("SELECT name FROM recipes")
recipes = [r[0] for r in c.fetchall()]
recipe_name = st.sidebar.selectbox("Recipe", recipes)
recipe_name = recipes[0]

network = bc.Network(lib, recipe_name, dbconn)
# ut.h2(f'Recipe {recipe_name}')
# ut.drawComputeGraph(network.compute_graph, cdg=network.central_dogma_graph)
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

# model = bc.ComputeGraphModel(network)
# model.build()

# rng_key = jax.random.PRNGKey(0)
# params = model.init(rng_key)
# model(params, [], rng_key)

# def sum_model(params, inputs, rng_key):
    # return jnp.sum(model(params, inputs, rng_key))

# jit(model)(params, [], rng_key)

# g = grad(sum_model)(params, [], rng_key)
# print(g)


inv_network = bc.inverter(network)
ut.h2(f'With inverse path prepended')
ut.drawComputeGraph(inv_network.compute_graph, cdg=inv_network.central_dogma_graph)



