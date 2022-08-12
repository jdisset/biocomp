# %load_ext autoreload
# %autoreload 2

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     imports     --
# ···············································································
import streamlit as st
from dataclasses import dataclass

st.set_page_config(layout='wide')

import random
import biocomp as bc
import pandas as pd
import numpy as np
import scriptutils as ut
import time
from rich import print
import biocomp.utils as bu
import jax
from jax import jit, vmap, grad
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

inputs = {}

cdg = bc.buildCentralDogmaGraph(lib, l1_DNAs, inputs)
compg = bc.buildComputeGraph(lib, cdg)

# col1, col2 = st.columns([70, 30])
# with col1:
# ut.h3('Central dogma graph:')
# ut.grnGraph(cdg)
# with col2:
# ut.h3('Compute graph:')
# ut.drawComputeGraph(compg)

cdg
compg

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                   --     constructs    --
# ···············································································


def any_promoter(lib, **_):
    all_promoters = lib.pc[lib.pc.category == 'promoter'].index.tolist()
    return all_promoters

def any_uorf(lib, **_):
    all_uORFs = lib.pc[lib.pc.category == 'uORF'].index.tolist()
    return all_uORFs + [None]

# picks a randmo ern_rec, and ensure that it is not for an ERN that's already in the L1
def random_ERN_rec(lib, rdm_key, l1, **_):
    all_sequestrons = lib.sequestrons[lib.sequestrons.type == 'ERN']
    already_in_l1 = []
    for s in l1.parts_slots:
        if s.is_resolved and s.part in all_sequestrons['negative_part'].values:
            already_in_l1.append(s.part)
    possible_recog = all_sequestrons[~all_sequestrons['negative_part'].isin(already_in_l1)][
        'positive_part'
    ].values.tolist()

    if already_in_l1:
        possible_recog = possible_recog + [None]

    return possible_recog[jax.random.randint(rdm_key, (1,), 0, len(possible_recog))[0]]


# picks a random ern, and ensure that it is not for an ERN_rec that's already in the L1
def random_ERN(lib, rdm_key, l1, **_):
    all_sequestrons = lib.sequestrons[lib.sequestrons.type == 'ERN']
    already_in_l1 = []
    for s in l1.parts_slots:
        if s.is_resolved and s.part in all_sequestrons['positive_part'].values:
            already_in_l1.append(s.part)
    possible_ern = all_sequestrons[~all_sequestrons['positive_part'].isin(already_in_l1)][
        'negative_part'
    ].values.tolist()

    if already_in_l1:
        possible_ern = possible_ern + [None]

    return possible_ern[jax.random.randint(rdm_key, (1,), 0, len(possible_ern))[0]]


def random_seed():
    return random.randint(0, 2**32)


L1s = [bc.L1([bc.Part(p) for p in parts]) for parts in l1_DNAs]

# map L1 to parameter values for each node

ERN_template = bc.L1(
    [
        bc.Slot(any_promoter),
        bc.Slot(any_uorf),
        bc.Slot(random_ERN_rec),
        bc.Slot(random_ERN),
        bc.Part('NeonGreen'),
    ]
)
ERN_template.resolve_all_slots(lib, random_seed=3)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
