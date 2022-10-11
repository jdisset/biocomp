## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     import and init     --
# ···············································································
# %load_ext autoreload
# %autoreload 2

import streamlit as st

st.set_page_config(layout='wide')

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

from rich import print
from pprint import pprint


lib = ut.getLibFromGoogleSheet()

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     load xp     --
#···············································································

base_path = Path("/Users/jeandisset/Dropbox (MIT)/Biocomp/")
base_xp_path = base_path / "Experiments"
base_recipe_path = base_path / "Recipes"

experiments = [x.name for x in base_xp_path.iterdir() if x.is_dir()]

xp = experiments[0]
xpfile = base_xp_path / xp / f"{xp}.xp.json5"


class XP:
    # an XP contains a set of samples.
    # Each sample implements one recipe, and resulted in one data file.

    def __init__(self, filename):
        self.samples: dict
        self.name: str
        self.filename = filename
        with open(filename) as f:
            xpobj = json5.load(f)
            for k, v in xpobj.items():
                setattr(self, k, v)

    def __str__(self):
        # add borders:
        res = '-' * 18 + f'  XP {self.name}  ' + '-' * 18 + '\n'
        for k, v in self.__dict__.items():
            if isinstance(v, dict):
                res += f"* {k}:\n"
                for kk, vv in v.items():
                    res += f"    {kk}: {vv}\n"
            elif isinstance(v, list):
                res += f"* {k}:\n"
                for vv in v:
                    res += f"    {vv}\n"
            else:
                res += f"* {k}: {v}\n"

        return res

    def __repr__(self):
        return self.__str__()


xp = XP(xpfile)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                       --     build models     --
#···············································································
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


selected_recipe = st.sidebar.selectbox("Select a recipe", list(networks.keys()))
network = inv_networks[selected_recipe]
ut.h2(f'Recipe {selected_recipe}')
ut.drawComputeGraph(network.compute_graph, cdg=network.central_dogma_graph)


models = [bc.ComputeGraphModel(inv_networks[r]) for r in recipe_names]
for m in models:
    m.build()

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────



##

# # load the data files
# datafiles = [base_xp_path / xp.name / 'data'/ f"{s['name']}.{xp.name}.csv" for s in xp.samples]
# df_data = [pd.read_csv(f) for f in tqdm(datafiles, "loading data files")]

# # we want to reorder data columns to match the model's output
# out_prots = [model.get_output_proteins() for model in models]
# out_channels = [[xp.color_names[k] for k in out_prot] for out_prot in out_prots]

# Y = [jnp.array(d[channels]) for d, channels in zip(df_data, out_channels)]
# X = [model.get_input_from_output(d) for model, d in zip(models, Y)]

# model = models[0]
# model.flat_batches
# model.network.compute_graph
# x = X[0]
# rng_key = jax.random.PRNGKey(0)
# params = model.init(rng_key)

# model.network.compute_graph
# params

# WHY IS THERE NO INVERSE AGGREGATION?

# models[0](X[0][0])







