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
from copy import deepcopy

lib = ut.getStState('lib', ut.getLibFromGoogleSheet)

lib2 = deepcopy(lib)
print(lib)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     adding new ERNs     --
# ···············································································
lib = deepcopy(lib2)


def add_fake_ERN(lib, name):
    recog_name = f'{name}_recog_5p'
    seq = {
        'type': 'ERN',
        'negative_part': name,
        'positive_part': recog_name,
        'output_part': f'["{recog_name}"]',
        'parameter_values': {},
    }
    lib.addSequestron(seq)
    lib.addPart(recog_name, 'ERN_recog_site_5p')
    lib.addPart(name, 'ERN')


for i in range(20):
    add_fake_ERN(lib, f'ERN_{i}')


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                       --     N + 1 graphs     --
# ···············································································


def num_to_bits(num, n):
    return [num >> i & 1 for i in range(n)]


# ERN nodes:
N = 8


def make_recog(i, sign):
    rcb = 'PhiC31' if sign == 0 else 'PhiC31RDF'
    return ['hEF1a', rcb, f'ERN_{i}_recog_5p']


def generate_all_l1s(N):
    result = []
    ERNs = [['hEF1a', f'ERN_{i}'] for i in range(N)]
    output = [['hEF1a', 'PhiC31'], ['hEF1a', 'PhiC31RDF'], ['hEF1a', 'attL', 'NeonGreen', 'attR']]
    for i in range((2**N)):
        bits = num_to_bits(i, N)
        recogs = [make_recog(i, sign) for i, sign in enumerate(bits)]
        result.append([*ERNs, *recogs] * 3 + [*output])
    return result


l1s = generate_all_l1s(N)
l1s
# bourrin:
inputs = {**{l: 0 for l in range(N * 2)}, **{l: 1 for l in range(N * 2, N * 4)}}


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                       --     build graphs     --
# ···············································································

id = int('11100000', 2)
cdg = bc.buildCentralDogmaGraph(lib, l1s[id], inputs)
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

target = mpimg.imread('../data/double_triangle.png')[::-1]

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
    best_run = np.argmin(losses[:, -1])
    best_loss = losses[best_run]
    params_history = bu.param_unstack(bu.get_pytree(stacked_params, best_run), len(best_loss) + 1)
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
