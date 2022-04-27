## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     imports     --
#···············································································
import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
import sys
import os

try: print(__file__)
except NameError: __file__ = ''
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir+'/../')

import utils as ut
from types import SimpleNamespace

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random



def md(t):
    return st.markdown(t)

def h1(t):
    return md(f'# {t}')

def h2(t):
    return md(f'## {t}')

def h3(t):
    return md(f'### {t}')

def h4(t):
    return md(f'#### {t}')

def b():
    return md('---')

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{             --     Loading library from google sheet     --
#···············································································
SHEET_KEY = '1K_2bt90E-Wk-A9PYGXGbKDJy-olojKtksy1jxCQAzME'

if 'lib' not in st.session_state:
    lib = ut.getAllGoogleSheets(SHEET_KEY)
    lib = SimpleNamespace(**lib)
    md(f'Loaded library with {len(lib.__dict__)} tables: '+', '.join(lib.__dict__.keys()))
    st.session_state.lib = lib
else:
    lib = st.session_state.lib

col1, col2, col3 = st.columns(3)
col1.markdown('### Parts')
col1.write(lib.parts)
col2.markdown('### Parts categories')
col2.write(lib.categories)
col3.markdown('### Sequestrons')
col3.write(lib.sequestrons)
b()
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     Define a few L1s     --
#···············································································

p = lib.parts
c = lib.categories
pc = pd.merge(p,c, left_on='category', right_index=True, how='left')

def getRna(dna):
    d = pc.loc[dna]
    return tuple(d[d.transcripted == 1].index)


def getPrt(dna):
    d = pc.loc[dna]
    return tuple(d[d.translated == 1].index)

l1_DNAs=[['hEF1a','NeonGreen','CasE_recog_5p'],['hEF1a','CasE'],['hEF1a', 'Csy4'],['hEF1a','NeonGreen','Csy4_recog_5p'],['hEF1a','NeonGreen','CasE_recog_5p']]
l1 = [{'dna':tuple(d), 'rna':getRna(d), 'prt':getPrt(d)} for d in l1_DNAs]
l1df = pd.DataFrame(l1)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{       --     We need to merge nodes with identical content     --
#···············································································
dna_df= pd.DataFrame({'l1_id':[[x] for x in l1df.index], 
        'type':'dna', 'successor':None} )

rna_df= pd.DataFrame({
        'l1_id':list(l1df.reset_index().groupby(by='rna').agg(list)['index']),
        'type':'rna', 'successor':None} )

prt_df= pd.DataFrame({
        'l1_id':list(l1df.reset_index().groupby(by='prt').agg(list)['index']), 
        'type':'prt','successor':None} )

gdf = pd.concat([dna_df,rna_df,prt_df]).reset_index(drop=True)

for i,r in gdf[gdf.type=='rna'].iterrows():
    gdf.loc[r.l1_id, 'successor'] = i

for i,r in gdf[gdf.type=='prt'].iterrows():
    gdf.loc[gdf.loc[r.l1_id].successor, 'successor'] = i

gdf['predecessor'] = [list() for _ in range(len(gdf))]

for i,r in gdf.iterrows():
    if r.successor is not None:
        gdf.loc[r.successor]['predecessor'] += [i]
gdf.loc[~gdf.predecessor.astype(bool), 'predecessor'] = None

# add content of each node
gdf['content'] = gdf.apply(lambda x: l1df.loc[x.l1_id].iloc[0][x.type], axis=1)
gdf['content_type'] = gdf.apply(lambda x: tuple([lib.parts.loc[p][0] for p in x.content]), axis=1)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                --     defining outputs of the GRN     --
#···············································································
outputs = ['NeonGreen']

def containsOutput(l,outputs):
    for o in outputs:
        if o in l:
            return True
    return False

gdf['is_output'] = False
gdf.loc[gdf.type == 'prt', 'is_output'] = gdf.loc[gdf.type == 'prt'].l1_id.apply(
                lambda x: containsOutput(l1df.loc[x].prt.tolist()[0],outputs))

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


  # { id: "1", type: "input", data: { label: "Input Node" }, position: { x: 250, y: 25 } },

nodes = [ {'id':f'{i}', 'type':n.type, 'data':{'content':n.content, 'content_type':n.content_type}} for i,n in gdf.iterrows()]
edges = [ { 'id': f'{i}', 'source': f'{i}', 'target': f'{n.successor}'} for i,n in gdf.iterrows() if n.successor]


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                     --     plotting the GRN     --
#···············································································

import streamlit.components.v1 as components
_component_func = components.declare_component("ned_component", 
        url="http://localhost:3001")

other_component_func = components.declare_component("other_component", 
        url="http://localhost:3001")

def updated_dict(d1, d2):
    res = {}
    for key, val in d1.items():
        if type(val) == dict:
            if key in d2 and type(d2[key] == dict):
                res[key] = updated_dict(d1[key], d2[key])
        else:
            if key in d2:
                res[key] = d2[key]
            else:
                res[key] = d1[key]
    for key, val in d2.items():
        if not key in d1:
            res[key] = val
    return res


def grnGraph(nodes, edges, key=None):
    tnodes = [updated_dict(n,{'data':{'id':n['id']}}) for n in nodes]
    _component_func(nodes=tnodes,edges=edges,output_type='GRN',key=key)

def dnaOutput(nodes, key=None):
    tnodes = [updated_dict(n,{'data':{'id':n['id']}}) for n in nodes if n['type'] == 'dna']
    _component_func(nodes=tnodes,output_type='DNA',key=key)

# h3('Gene expression graph')
# grnGraph(nodes, edges)

h3('DNA constructs')
dnaOutput(nodes)
b()


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────



## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                       --     compute graph     --
#···············································································
# Here we generate a dataframe that contains all the available functions for our current library
seqs = lib.sequestrons.merge(lib.sequestron_types, left_on='type', right_index=True)
seqs

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


