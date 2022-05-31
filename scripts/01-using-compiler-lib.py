# %load_ext autoreload
# %autoreload 2
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     imports     --
# ···············································································
import streamlit as st

st.set_page_config(layout='wide')

import biocomp as bc
import pandas as pd
import numpy as np
import scriptutils as ut
from rich import print
import jax
import jax.numpy as jnp
from tqdm import tqdm
from functools import partial

lib = ut.getStState('lib', ut.getLibFromGoogleSheet)

print(lib)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                       --     build graphs     --
# ···············································································

# l1_DNAs = [
# ['hEF1a', 'PhiC31RDF', 'CasE_recog_5p'],
# ['hEF1a', 'Csy4'],
# ['hEF1a', 'PhiC31RDF', 'CasE_recog_5p'],
# ['hEF1a', 'Csy4'],
# # biases:
# ['hEF1a', 'CasE'],
# ['hEF1a', 'PhiC31RDF', 'Csy4_recog_5p'],
# ['hEF1a', 'PhiC31'],
# # output
# ['hEF1a', 'attP', 'NeonGreen', 'attB'],
# ]
# inputs = {0: 0, 1: 0, 2: 1, 3: 1}

l1_DNAs = [
    ['hEF1a', 'PhiC31', 'Csy4_recog_5p'],
    ['hEF1a', 'CasE'],
    ['hEF1a', 'PhiC31', 'Csy4_recog_5p'],
    ['hEF1a', 'CasE'],
    # biases:
    ['hEF1a', 'Csy4'],
    ['hEF1a', 'CasE_recog_5p', 'PhiC31'],
    ['hEF1a', 'PhiC31RDF'],
    # output
    # ['hEF1a', 'attP', 'NeonGreen', 'attB'],
    ['hEF1a', 'attL', 'NeonGreen', 'attR'],
]
inputs = {0: 0, 1: 0, 2: 1, 3: 1}

# l1_DNAs = [
# # plasmid 0
# ['hEF1a', 'CasE'],
# ['hEF1a', 'RFP'],
# # plasmid 1
# ['hEF1a', 'NeonGreen', 'CasE_recog_5p'],
# ['hEF1a', 'BFP'],
# ]
# inputs = {0: 0, 1: 0, 2: 1, 3: 1}

# l1_DNAs = [
# # plasmid 0
# ['hEF1a', 'CasE'],
# ['hEF1a', 'RFP'],
# # plasmid 1
# ['hEF1a', 'NeonGreen', 'CasE_recog_5p'],
# ['hEF1a', 'BFP'],
# ]
# inputs = {0: 0, 1: 0, 2: 1, 3: 1}

cdg = bc.buildCentralDogmaGraph(lib, l1_DNAs, inputs)
ut.h3('Central dogma graph:')
ut.grnGraph(cdg)
compg = bc.buildComputeGraph(lib, cdg)

st.write(l1_DNAs)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


ut.h3('Compute graph:')
ut.drawComputeGraph(compg)


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                   --     getting training data     --
# ···············································································
import matplotlib.image as mpimg

target = mpimg.imread('../data/band_pass_dec.png')[::-1]

OUTPUT_LVL = 0.5
N_SAMPLES = 3000
samples = []
key = jax.random.PRNGKey(42)
X = jax.random.uniform(key=key, shape=(N_SAMPLES, 2)) * jnp.array(target.shape[:2])
y_true = jnp.array(target[X.astype(int)[:, 1], X.astype(int)[:, 0], 0] * OUTPUT_LVL).reshape(-1, 1)
X = X / jnp.array(target.shape[:2])

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     Training     --
# ···············································································

TRAIN = True
if TRAIN:
    model = bc.ComputeGraphModel.fromDataframe(compg)
    losses, stacked_params = model.train(
        key, X, y_true, n_init=100, n_steps=1000, learning_rate=0.003
    )

    best_run = np.argmin(losses[:, -1])
    best_loss = losses[best_run]
    stacked_best = bc.ut.get_pytree(stacked_params, best_run)
    best_params = bc.ut.get_pytree(stacked_best, len(best_loss))
    best_params_history = bc.ut.param_unstack(stacked_best, len(best_loss) + 1)

    compg_history = [
        model.toDataframe(p) for p in tqdm(best_params_history, "Unpacking parameters history")
    ]

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     Plotting     --
# ···············································································


ut.h3('Compute graph after training:')
# ut.drawComputeGraph(compg_history[-1])

col1, col2, col3 = st.columns(3)
with col1:
    ut.drawComputeGraph(compg_history[-1])
with col2:
    st.pyplot(ut.plotModelOutput(model, best_params))
with col3:
    st.pyplot(ut.plotBestLoss(best_loss, losses, 'Best loss history'))

##
# import nest_asyncio
# nest_asyncio.apply()
# ut.screenCaptures(partial(ut.drawComputeGraph, height=2000), compg_history[::5], out_dir_path='../__out/testm', height=2000, width=1500, n_batches=5)

# ut.trainingMovie(model, compg_history, best_params_history, best_loss, losses, outdir='../__out/movie_03', step=10)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

