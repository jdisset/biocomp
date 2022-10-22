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


# def get_lib():
    # return ut.getLibFromGoogleSheet()
# lib = get_lib()
# ut.save(lib, '/tmp/lib.pickle')

lib = ut.load('/tmp/lib.pickle')


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

# selected_recipe = st.sidebar.selectbox("Select a recipe", list(networks.keys()))


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     load xp     --
#···············································································


def get_xps(xp_path):
    xpnames = [x.name for x in xp_path.iterdir() if x.is_dir()]
    return {x: bc.XP(x, xp_path, recipe_path, lib) for x in xpnames}


base_path = Path("/Users/jeandisset/Dropbox (MIT)/Biocomp/")
xp_path = base_path / "Experiments"
recipe_path = base_path / "Recipes"

xps = get_xps(xp_path)
selected_experiment = list(xps.keys())[1]
xp = xps[selected_experiment]


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

# ut.drawComputeGraph(network.compute_graph, cdg = network.central_dogma_graph)



models = xp.get_models()
X, Y = xp.get_XY(models)


x = list(X.values())[0]
y = list(Y.values())[0]
model = list(models.values())[0]

out_proteins = model.get_output_proteins()
za = out_proteins.index('eYFP')
xa, ya = out_proteins.index('eBFP'), out_proteins.index('mKate')

stats, bins = du.binstats(y, out_proteins)
stats, bins = du.binstats(y, out_proteins, ['mKate', 'eBFP'], resolution=0.5)


du.heatmap(
        stats,
        bins,
        figscale=0.7,
        stat_columns=['mean','count'],
        axis_names=[out_proteins[xa], out_proteins[ya], out_proteins[za]],
        title=f'{model.network.name} unbalanced',
        subtitle=f'{len(data)} points',
        # filename=f'../__out/unbalanced_{sample}.png',
    )


stat_columns = ['mean']
nstats = len(stat_columns)
df = stats[stats['count'] > 1]

xy_axis = [n[1] for n in df.index.names]
z_axis = [c[0] for c in df.columns if c[1] and c[0] not in xy_axis][0]
stat = stat_columns[0]
Z = np.full((len(bins[xy_axis[0]]), len(bins[xy_axis[1]])), np.nan)
statcol = 'count' if stat == 'count' else (z_axis, stat)

for coords, value in df[statcol].items():
    Z[coords] = value











































# now all the data, individually. We build a new X and Y, X_balanced and Y_balanced
X_balanced = {}
Y_balanced = {}
nbins = 20

for sample, model in models.items():
    print('-' * 80)
    data = np.array(Y[sample])
    out_proteins = model.get_output_proteins()
    za = out_proteins.index('eYFP')
    xa, ya = out_proteins.index('eBFP'), out_proteins.index('mKate')
    stats, bins = du.binstats(data, [xa, ya], za, nbins=nbins)
    Y_bal = du.balance_per_bin(data, stats, threshold_quantile=0.4, threshold_min=20)
    X_bal = model.get_input_from_output(Y_bal)
    X_balanced[sample] = X_bal
    Y_balanced[sample] = Y_bal
    # plot heatmap
    # before:
    print(f'before: {sample}')
    du.heatmap(
        stats,
        bins,
        figscale=0.7,
        axis_names=[out_proteins[xa], out_proteins[ya], out_proteins[za]],
        title=f'{sample} unbalanced',
        subtitle=f'{len(data)} points',
        # filename=f'../__out/unbalanced_{sample}.png',
    )
    # after:
    print(f'after: {sample}')
    bdf, bbins = du.binstats(Y_bal, [xa, ya], za, nbins=nbins)
    chg = np.mean(np.abs(bdf['mean'] - stats['mean'])) / np.std(stats['mean'])

    du.heatmap(
        bdf,
        bbins,
        figscale=0.7,
        axis_names=[out_proteins[xa], out_proteins[ya], out_proteins[za]],
        title=f'{sample} balanced',
        subtitle=f'{len(Y_bal)} points ({len(Y_bal)/len(data)*100:.1f}% of original) | changed bin means by {chg*100.0:.1f}% of std',
        # filename=f'../__out/balanced_{sample}.png',
    )

