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
# {{{                        --      slot things     --
# ···············································································


class Slot:
    def __init__(self, f):
        self.f = f  # resolve function, that can take the L1 as argument for example, or a dictionary of param values
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


def pick_from_category(category, lib, rdm_key, other_slots=[], can_be_empty=False):
    parts = lib.pc[lib.pc.category == category].index.tolist()
    picked_id = jax.random.randint(rdm_key, (1,), 0, len(parts))[0]
    return parts[picked_id]


rdm_key = jax.random.PRNGKey(0)
pick_from_category('ERN', lib, rdm_key)

part_type_to_parameter_name = {'promoter': 'tx_rate', 'uORF': 'tl_rate', 'degron': 'rna_deg_rate'}


class L1:
    def __init__(self):
        self.quantize_functions = {}  # param -> quantize function
        self.slots = []

    def resolve_from_params(self, params):
        # for each unresolved slot, resolve it with the given params
        for slot in self.slots:
            if not slot.is_resolved:
                slot.resolve(params)  # will attempt resolve from params


# A slot can contain:
# - A: a function that takes the L1 as argument and can resolve the slot
# - B: a function that will return a quantize_function
# or maybe a list of parts ?
# example quantize function dictionary (per L1)


def q(shared_params, node_params):
    quantized_rates_ids = ["hEF1a", "..."]
    possible_rates = shared_params["tx_rate"][quantized_rates_ids]
    return possible_rates


# or maybe just return the values for quantization?
# but if one can be continuous? well just return none and have quantize handle that


def q(shared_params, node_params):
    quantized_rates_ids = ["hEF1a", "..."]
    possible_rates = shared_params["tx_rate"][quantized_rates_ids]
    return bc.quantize(node_params["tx_rate"], possible_rates)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                   --     let's start with L1s     --
# ···············································································


lib
part_type_to_parameter_name = {'promoter': 'tx_rate', 'uORF': 'tl_rate', 'degron': 'rna_deg_rate'}

def mapped_parameter(lib, part_name, category_to_param=part_type_to_parameter_name):
    if part_name is not None:
        if part_name in lib.pc.index:
            category = lib.pc.loc[part_name, 'category']
            if category in category_to_param:
                return category_to_param[category]
        else:
            raise ValueError(f'Unknown part: {part_name}')
    return None

class Slot:
    def __init__(self, f):
        self.resolve_function = f
        self.part = None  # list means multiple parts that should map to a single parameter. Otherwise single string
        self.maps_to_parameter = None
        self.is_resolved = False

    def resolve(self, *args, **kwargs):
        if not self.is_resolved:
            self.part = self.resolve_function(*args, **kwargs)
            if self.part == [] or self.part == [None]:
                self.part = None
            if isinstance(self.part, list):
                mapped = [mapped_parameter(lib, p) for p in self.part if p is not None]
                if len(mapped) != 1:
                    raise ValueError(f'{self.part} maps to {len(mapped)} parameters ({mapped})')
                self.maps_to_parameter = mapped[0]
            else:
                self.maps_to_parameter = mapped_parameter(lib, self.part)
            if self.maps_to_parameter is not None and not isinstance(self.part, list):
                self.part = [self.part]
            self.is_resolved = True

    def resolved(self, *args, **kwargs):
        self.resolve(*args, **kwargs)
        return self

    def __repr__(self):
        if self.is_resolved:
            if self.maps_to_parameter is None:
                if self.part is None:
                    return '<empty slot>'
                else:
                    return f'<{self.part}>'
            return f'<{self.part} -> {self.maps_to_parameter}>'
        else:
            return f'<slot(unresolved, {self.resolve_function})>'


def any_promoter(lib, **_):
    all_promoters = lib.pc[lib.pc.category == 'promoter'].index.tolist()
    return all_promoters + [None]


def any_uorf(lib, **_):
    all_uORFs = lib.pc[lib.pc.category == 'uORF'].index.tolist()
    return all_uORFs + [None]

def random_ERN_rec(lib, rdm_key, **_):
    all_rec = lib.pc[lib.pc.category == 'ERN_recog_site_5p'].index.tolist() + [None]
    return all_rec[jax.random.randint(rdm_key, (1,), 0, len(all_rec))[0]]


def FixedPart(name):
    return Slot(lambda: name).resolved()


# For an L1, biologists need and use named slots
# and named L0s (a list of parts) but we just flatten everything,
# so an L1 just contains a list of parts.
class L1:
    def __init__(self, slots):
        self.name = ''
        self.parts_slots = slots
        self.quantize_functions = {}

    def resolve_all_slots(self, lib, random_seed=1, random_order=False):
        rdm = jax.random.PRNGKey(random_seed)
        allrdm = jax.random.split(rdm, len(self.parts_slots))
        order = list(range(len(self.parts_slots)))
        if random_order:
            order = jax.random.permutation(rdm_key, len(self.parts_slots))
        for i, r in zip(order, allrdm):
            if not self.parts_slots[i].is_resolved:
                print(self.parts_slots[i])
                self.parts_slots[i].resolve(lib, l1=self, rdm_key=r)

    def is_resolved(self):
        return all(s.is_resolved for s in self.parts_slots)

    def __repr__(self):
        return f'L1({self.parts_slots})'

L1s = [L1([FixedPart(p) for p in parts]) for parts in l1_DNAs]

# map L1 to parameter values for each node

# ERN_template = L1([Slot(any_promoter), Slot(any_uorf), Slot(any_ERN_rec), FixedPart('NeonGreen')])
ERN_template = L1([Slot(any_promoter), Slot(any_uorf), Slot(random_ERN_rec), FixedPart('NeonGreen')])
ERN_template.resolve_all_slots(lib, random_seed=1)
ERN_template

ERN_template = L1([Slot(any_promoter), Slot(any_uorf), Slot(random_ERN_rec), FixedPart('NeonGreen')])
ERN_template.resolve_all_slots(lib, random_seed=2)
ERN_template

ERN_template = L1([Slot(any_promoter), Slot(any_uorf), Slot(random_ERN_rec), FixedPart('NeonGreen')])
ERN_template.resolve_all_slots(lib, random_seed=3)
ERN_template

ERN_template = L1([Slot(any_promoter), Slot(any_uorf), Slot(random_ERN_rec), Slot(random_ERN), FixedPart('NeonGreen')])

#                                                                            }}}
