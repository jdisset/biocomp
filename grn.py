## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     imports     --
#···············································································
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

import torch
import torchmodules as tm
import utils as ut
from types import SimpleNamespace

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

%load_ext autoreload
%autoreload 2

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{             --     Loading library from google sheet     --
#···············································································
SHEET_KEY = '1K_2bt90E-Wk-A9PYGXGbKDJy-olojKtksy1jxCQAzME'
lib = ut.getAllGoogleSheets(SHEET_KEY)
lib = SimpleNamespace(**lib)
print(f'Loaded library with {len(lib.__dict__)} tables: '+', '.join(lib.__dict__.keys()))
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

# ---- LIBRARY ENCODING AND GRN REPRESENTATION
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     Define a few L1s     --
#···············································································

p = lib.parts
c = lib.categories
pc = pd.merge(p,c, left_on='category', right_index=True, how='left')

def getRna(dna):
    d = pc.loc[dna]
    return frozenset(d[d.transcripted == 1].index)

def getPrt(dna):
    d = pc.loc[dna]
    return frozenset(d[d.translated == 1].index)

l1_DNAs=[['hEF1a','NeonGreen','CasE_recog_5p'],['hEF1a','CasE'],['hEF1a', 'Csy4'],['hEF1a','NeonGreen','Csy4_recog_5p'],['hEF1a','NeonGreen','CasE_recog_5p']]
l1 = [{'dna':frozenset(d), 'rna':getRna(d), 'prt':getPrt(d)} for d in l1_DNAs]
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

gdf
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

# - STUDY OF THE GRN -> SN DIRECTION
# ---- BASIS OF COMPUTATIONAL GRAPH GENERATION WITH PYTORCH
# We start by generating a Compute Graph from a list of manually curated L1s

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{            --     generating compute graph from GRN     --
#···············································································
# we only need things that are connected to the output
# so we generate the computation graph starting from the bottom

# Example of Compute Graph:
cg_ex = [{ 'type':'ERNSequestron', 'output_to':[1], 'input_from':[[1,2],[3]] },
    { 'type':'RtoP', 'output_to':None, 'input_from':[0] }]

# The Gene Expression Graph describes each nodes by their type (dna, rna, prt), their successor nodes ids, and their predecessor nodes ids.
# The Compute Graph is also represented in a dataframe (named cdf). It is just 1 level of abstraction above the Gene Expression graph (stored in the gdf dataframe). 
# Each row of the Compute Graph dataframe describes a node with:
# - function_name: ("ERNSeq", "transcription", "translation") taken from the function description dataframe (named fdf)
# - input_from: a list of compute nodes ids whose output we will feed into the function
# - output_to: the list of compute nodes this node is outputing the result of the function to
# - gene_nodes: the id of the gdf rows this compute node contains

# fdf is a dataframe containing the list of available functions, and it applies to a library

# Here we generate a dataframe that contains all the available functions for our current library
seqs = lib.sequestrons.merge(lib.sequestron_types, left_on='type', right_index=True)
gdf

## 
# now let's generate the compute graph that matches gdf. There should actually be only one. Building it is simple: we start from the output node, and we go up. At every level, we check if there's an actual sequestration node. If yes we add it to the CG. And we also need to not forget to add the translation / transcription nodes as required.

# in the compute graph, each node is a computation (that "mostly" matches with a grn edge)

preds = gdf[gdf.is_output].predecessor.tolist()[0]
columns = ['function','inputs','outputs','params']

cur = {}
preds

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


# ---- GENERATING SEQUESTRONS NETWORKS FROM LIB

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{            --     Add ERN-based sequestron reaction    --
#···············································································


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                --     Convert GRN to pytorch (SN with constraints)    --
#···············································································

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                   --     Train on example data     --
#···············································································

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                   --     Output final circuit     --
#···············································································

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

# - STUDY OF THE SN -> GRN DIRECTION
# ---- CONVERTING A SEQUESTRON TO A GENERIC GRN








## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     Archive     --
#···············································································
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{           --     Load Library of Parts into DataFrames     --
#···············································································

# import sqlite3
# import pandas as pd
# cnx = sqlite3.connect('data/database.db')
# part_df = pd.read_sql_query("SELECT * FROM part", cnx)
# category_df = pd.read_sql_query("SELECT * FROM category", cnx)
# lib = part_df.merge(category_df.rename({'name': 'category'}, axis=1))
# lib = lib.set_index('name')

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────



#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


