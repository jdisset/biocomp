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

lib = ut.getLibFromGoogleSheet()

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

base_path = Path("/Users/jeandisset/Dropbox (MIT)/Biocomp/")
xp_path = base_path / "Experiments"
recipe_path = base_path / "Recipes"
experiments = [x.name for x in xp_path.iterdir() if x.is_dir()]

xp = bc.XP(experiments[0], xp_path, recipe_path, lib)
models = xp.get_models(inverse=True)
X, Y = xp.get_XY(models)


selected_sample = st.sidebar.selectbox("Select a sample", [s['name'] for s in xp.samples])


cfg = {
    "learning_rate": 0.001,
    "adam_w_decay": 0.01,
    "clipping": 0.001,
    "n_replicates": 10,
    "initial_param_scaling": 0.01,
    "normalize_data": False,
    "epochs": 100,
    "log_rate": 50,
    "rng_key": 1,
}

sample = 'CoTX-All'
sample = selected_sample
model = models[sample]
x, y = X[sample], Y[sample]


# def grnGraph(gdf, key=None, func=ut._component_func):
gdf = model.network.central_dogma_graph
nodes = [{'id': f'{i}', 'type': n.type, 'data': n.to_dict()} for i, n in gdf.iterrows()]
edges = [
    {'id': f'{i}', 'source': f'{i}', 'target': f'{n.successor}'}
    for i, n in gdf.iterrows()
    if n.successor
]
tnodes = [bc.ut.updated_dict(n, {'data': {'id': n['id']}}) for n in nodes]

import streamlit.components.v1 as components
cfunc = ut.components.declare_component("ned_component", url="http://localhost:1234")
# cfunc(nodes=tnodes, edges=edges, output_type='GRN', key=None)

ut.drawComputeGraph(model.network.compute_graph, cdg=model.network.central_dogma_graph)
