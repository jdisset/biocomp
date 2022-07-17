# %load_ext autoreload
# %autoreload 2
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     imports     --
# ···············································································
import streamlit as st
from dataclasses import dataclass

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

cdg
compg

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


# XP = {
# 'name': 'testXP',
# 'tubes': [
# {'name': 'tube1', 'aggregates': [
# {'ratio':0.5, 'L1s':[{'name': 'l1_A', 'L0s':['hEF1a', 'PhiC31', 'Csy4_recog_5p']}]},
# {'ratio':0.5, 'L1s':[{'name': 'l1_B', 'L0s':['hEF1a', 'CasE']}]},
# ]},
# {'name': 'tube2', 'aggregates': [
# {'ratio':0.5, 'L1s':[{'name': 'l1_C', 'L0s':['hEF1a', 'PhiC31', 'Csy4_recog_5p']}]},
# {'ratio':0.5, 'L1s':[{'name': 'l1_D', 'L0s':['hEF1a', 'CasE']}]},
# ]},
# ]
# }
# this format is actually the compiler's output format.

## le'ts tink about the input format

# we need to be able to declare templates for the L1s
# an L1 is composed of L0s, but most specifically, we usually will have:
# - a promoter
# - uOrfs
# some recognition site or a recombinase
# a gene
# a "closing" recombinase if there was an opening one

# some thins map directly to a parameter that we can use gradient descent on
# i.e promoters -> transcription_rate and uOrfs -> translation_rate
# the rest will change the nature and topology of the computation tree, which means
# they should be treated as different circuits that we can explore through gradient-free methods


# a L1 templates define some available slots for L0s, the acceptable L0s for these slots
# and the possible mapping to parameters with the possible values. We thus need an L0 template type as well
# Actually, L0 templates are not types but functions that take a library and return a L0

print(cdg)

lib.pc[lib.pc.category == 'promoter']

# parameters:


# hmm what if we have a parameter that depends on more than one slot?
# we should have a list of parameter values, and a function that resolves an L1 template
# from the given parameter values.
# So there's 2 kinds of resolve:
# for a non-parameter slot, we just have one pass, before we start the learning process,
# that will pick one slot
# for a parameter slot, we need a few things:
# - We can't collapse the slot to a single part before training.
#   so we maintain the slot as a list of possible parts (or a function?)
# - We need a function that, from a fully pre-training resolved L1,
#   returns a "constrain" function that will be used to constrain the parameter values
#   (could be just a discretizer or a step function)
# - Them we need a post-training resolve function that will take the parameter values
#   and return a L1 with only one part (or zero, I guess) per slot.


# constraint examples: recombinases should be matching, recognition sites cannot be from the same ERN being expressed, etc...
# for complex constraints, I guess a solution would be to use a CSP solver. But that's some heavy stuff.
# for our simple case, up until a certain point, we can just write functions that are relative to a certain slot,
# and that will take into account the other slots that have already been decided in order to limit the possibilities of the current one.
# the order in which each slot is decided is potentially random. This way we can also make this a generator that creates solutions one by one.
# now, we need to take into account these special slots that map to a parameter. Instead of resolving to a single part/value, they should return
# a discretization function.


# In summary:
# We describe constructs in terms of slots. At the end of a training, we will output fully resolved slots,
# which means that they will all contain a single part (or be empty).
# Now we have to be able to describe the possible content of slots as a function of the other slots in this L1,
# and even as a function of other L1s. This way we can enforce constraints such as:
# - have a fwd recombinase if already a bwd, don't add the ern_recog for the same ERN in a given L1, ... <- L1 level constraints
# - generate at least one corresponding ern_recog for each ERN in the pool of L1s, ... <- Full circuit level constraints
# There is a special type of slots that cannot be fully resolved before training, but merely filtered: the parameter slots.
# These are slots that will directly influence the parameters that will be learned. Fixing them to a single part would make no sense,
# so they need to be left in a special state that contains all the possible parts for this slot (which can be reduced depending on the other slots).
# Then, we will need a function that will take a partially resolved L1 and return the parameter constraints to be used during training.
# Example: we can declare a promoter slot. The first resolve will pick a list of valid promoters from the library.
# (We could also imagine picking a special type of infinitely tunable promoter. We don't have that yet but the future-proofing is nice.)
# Then, as each of the slots in this L1s have been resolved once, we feed the L1 to the parameter constraints function. This function
# will see that we have a slot that contains a promoter list, and will return a function that constrains "transcription_rate" to be one of the values
# matching the list of promoters.
# After training, once we have the final parameter value, we can just fully resolve the slot to the matching promoter.
# This way it should even be doable to write constraint generation functions that depends on mutiple slots,
# but it might get tricky as some parameter values might not be independant. But we don't have that for now.


# ok, let's start implementing

class Slot:
    def __init__(self, f):
        self.f = f
        self.value = 0
        self.is_resolved = False

    def resolve(self, *args):
        self.value = self.f(*args)
        # a slot is resolved when we have a single string value (or None, i.e empty)
        if isinstance(self.value, str) or self.value is None:
            self.is_resolved = True

def any_promoter(lib):
    # promoters are tied to a differentiable parameter so we wont resolve the slot to a single value
    all_promoters = lib.pc[lib.pc.category == 'promoter'].index.tolist()
    return all_promoters

def any_uorf(lib):
    all_uORFs = lib.pc[lib.pc.category == 'uORF'].index.tolist()
    return all_uORFs

def pick_from_category(category, lib, rdm_key, other_slots = [], can_be_empty = False):
    parts = lib.pc[lib.pc.category == category].index.tolist()
    picked_id = jax.random.randint(rdm_key, (1,), 0, len(parts))[0]
    return parts[picked_id]


l1 = [Slot(any_promoter), Slot(any_uorf), Slot(partial(pick_from_category, 'ERN'))]

rdm_key = jax.random.PRNGKey(0)
pick_from_category('ERN', lib, rdm_key, current_state = None)


l1[0].resolve(lib)
l1[0].is_resolved
l1[1].resolve(lib)
l1[1].is_resolved
l1[2].resolve(lib, rdm_key)
l1[2].is_resolved

[(s.value, s.is_resolved) for s in l1]

# extract parameter constraints



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

cdg
compg

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────







