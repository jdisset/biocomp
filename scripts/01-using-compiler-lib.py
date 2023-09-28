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
import time
from rich import print
import biocomp.utils as bu
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
    ['hEF1a', 'attL', 'NeonGreen', 'attR'],
]
inputs = {0: 0, 1: 0, 2: 1, 3: 1}


cdg = bc.buildCentralDogmaGraph(lib, l1_DNAs, inputs)
compg = bc.buildComputeGraph(lib, cdg)

col1, col2 = st.columns([70, 30])
with col1:
    ut.h3('Central dogma graph:')
    ut.grnGraph(cdg)
with col2:
    ut.h3('Compute graph:')
    ut.drawComputeGraph(compg)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                   --     getting training data     --
# ···············································································
import matplotlib.image as mpimg

target = mpimg.imread('../data/band_pass_dec.png')[::-1]

OUTPUT_LVL = 0.5
N_SAMPLES = 2000

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


class StreamlitProgress:
    def __init__(self, num_samples, message):
        st.write(message)
        self.start = time.time()
        self.total = num_samples
        self.current = 0
        self.bar = st.progress(0.0)

    def update(self, count):
        self.current += count
        self.bar.progress(self.current / self.total)

    def close(self):
        self.bar.progress(1.0)
        st.success(f'Trained in {(time.time() - self.start):.1f}s')


trained = False
if st.button('Start training'):
    st.write('Starting training loop...')
    trained = True
    model = bc.ComputeGraphModel.fromDataframe(compg)
    losses, stacked_params = model.train(
        key, X, y_true, n_init=100, n_steps=2000, learning_rate=0.1, progress_type=StreamlitProgress
    )
    run = np.argmin(losses[:, -1])
    best_loss = losses[run]
    params_history = bu.param_unstack(bu.get_pytree(stacked_params, run), len(best_loss) + 1)
    compg_history = [model.toDataframe(p) for p in tqdm(params_history, "Collecting params")]


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     Plotting     --
# ···············································································


if trained:
    ut.h3('Compute graph after training:')

    col1, col2 = st.columns([70, 30])
    with col1:
        ut.drawComputeGraph(compg_history[-1])
    with col2:
        st.pyplot(ut.plotModelOutput(model, params_history[-1]))
        st.pyplot(ut.plotBestLoss(best_loss, losses, vmax=0.15))

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{               --     experimenting with gradients     --
# ···············································································

# from jax import grad
# import matplotlib.pyplot as plt


# def wx(w, x):
# return w * x


# @partial(jax.custom_jvp, nondiff_argnums=(1,))
# def quantize(x, arr):
# return arr[jnp.argmin(jnp.abs(arr - x))]


# # we define the derivative of the quantize function as if it was unquantized (i.e x -> x)
# @quantize.defjvp
# def quantize_jvp(_, x, x_tang):
# (x,) = x
# (x_dot,) = x_tang
# return x, x_dot


# arr = jnp.array([1, 3, 6.3, 9.0])


# @jax.jit
# def wx_q(w, x):
# return quantize(w, arr) * x


# def plotF(F, title=''):
# fig, a = plt.subplots(1, 1, figsize=(10, 10))
# pc, *_ = ut.plotFuncOutput(lambda X: F(*X), a, xrange=(0, 10), yrange=(0, 10))
# cax = a.inset_axes([1.04, 0.2, 0.05, 0.6], transform=a.transAxes)
# fig.colorbar(pc, ax=a, cax=cax)
# fig.suptitle(title)
# plt.show()


# # plotF(wx_qcustom)
# # plotF(grad(wx_qcustom, argnums=1))

# plotF(jax.jit(wx_q), 'original: wx')
# plotF(grad(wx_q, argnums=0), 'original: dwx / dw')
# plotF(grad(wx_q, argnums=1), 'original: dwx / dx')


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{             --     experimenting with stochasticity     --
# ···············································································

# from jax import jit, grad


# def plotF_prob(Fp, key, title=''):
# meshres = (200, 200)
# rngs = jax.random.split(key, meshres[0] * meshres[1])
# fig, a = plt.subplots(1, 1, figsize=(10, 10))

# cpt = 0

# def F(xtup):
# w, x = xtup
# nonlocal cpt
# cpt += 1
# return Fp(w, x, rngs[cpt])

# pc, XX, YY, ZZ = ut.plotFuncOutput(F, a, xrange=(0, 10), yrange=(0, 10), meshres=meshres)
# cax = a.inset_axes([1.04, 0.2, 0.05, 0.6], transform=a.transAxes)
# fig.colorbar(pc, ax=a, cax=cax)
# fig.suptitle(title)
# plt.show()
# return ZZ


# def wx(w, x, key):
# return w * x


# def wx_prob(w, x, key):
# (noise,) = jax.random.multivariate_normal(key, mean=jnp.array([10.0]), cov=jnp.array([[0.5]]))
# return (w + noise) * x


# determ = plotF_prob(wx, key)
# prob = plotF_prob(wx_prob, key)
# g_determ = plotF_prob(grad(wx), key)
# g_prob = plotF_prob(grad(wx_prob), key)

# np.max(np.abs(determ - prob))
# np.max(np.abs(g_determ - g_prob))


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
