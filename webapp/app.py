## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{            --     imports, basic tools, general config    --
#···············································································
# -- config

PRINT_DATABASE = True
SHEET_KEY = '1K_2bt90E-Wk-A9PYGXGbKDJy-olojKtksy1jxCQAzME'

def is_interactive():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return True # Terminal running IPython
        if not hasattr(sys, 'ps1'):
            return True;
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

# -- imports
import streamlit as st
st.set_page_config(layout="wide")

# if is_interactive():
    # %load_ext autoreload
    # %autoreload 2

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
import sys
import os
from st_aggrid import AgGrid

try: print(__file__)
except NameError: __file__ = ''
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir+'/../')

import utils as ut
from types import SimpleNamespace

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

from rich.logging import RichHandler


# -- streamlit utils

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

def ag(df):
    rowH = 29
    AgGrid(df.reset_index(), fit_columns_on_grid_load=True, theme='light', height=(len(df)+1)*rowH)

# -- custom streamlit components
import streamlit.components.v1 as components
if not is_interactive():
    _component_func = components.declare_component("ned_component", url="http://localhost:3001")

def grnGraph(nodes, edges, key=None):
    tnodes = [ut.updated_dict(n,{'data':{'id':n['id']}}) for n in nodes]
    if not is_interactive():
        _component_func(nodes=tnodes,edges=edges,output_type='GRN',key=key)

def computeGraph(nodes, edges, key=None):
    tnodes = [ut.updated_dict(n,{'data':{'id':n['id']}}) for n in nodes]
    if not is_interactive():
        _component_func(nodes=tnodes,edges=edges,output_type='COMPUTE',key=key)

def dnaOutput(nodes, key=None):
    tnodes = [ut.updated_dict(n,{'data':{'id':n['id']}}) for n in nodes if n['type'] == 'dna']
    if not is_interactive():
        _component_func(nodes=tnodes,output_type='DNA',key=key)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{             --     Loading library from google sheet     --
#···············································································

if 'lib' not in st.session_state:
    lib = ut.getAllGoogleSheets(SHEET_KEY)
    lib = SimpleNamespace(**lib)
    md(f'Loaded library with {len(lib.__dict__)} tables: '+', '.join(lib.__dict__.keys()))
    st.session_state.lib = lib
else:
    lib = st.session_state.lib

if PRINT_DATABASE:
    col1, col2, col3 = st.columns(3)
    with col1:
        md('### Parts')
        ag(lib.parts)
    with col2:
        md('### Parts categories')
        ag(lib.categories)
    with col3:
        md('### Sequestrons')
        ag(lib.sequestrons)
    b()
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                --     Manually defining a few L1s     --
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
l1 = [{'DNA':tuple(d), 'RNA':getRna(d), 'PRT':getPrt(d)} for d in l1_DNAs]
l1df = pd.DataFrame(l1)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{    --     Constructing the Gene Expression Graph from the L1s    --
#···············································································

# ------------------------------------------------------------------------------
# gdf is the dataframe that contains all the information required to plot the gene
# expression graph, which is just a graph where each node represents a 
# DNA, RNA or PRT molecule
# ------------------------------------------------------------------------------

# To build it, we first create the dataframes containing the dna, rna and prt contents
# deduced from the l1 constructs. We also merge rna and prt nodes with identical content:
dna_df= pd.DataFrame({'l1_id':[[x] for x in l1df.index], 
        'type':'DNA', 'successor':None} )
rna_df= pd.DataFrame({
        'l1_id':list(l1df.reset_index().groupby(by='RNA').agg(list)['index']),
        'type':'RNA', 'successor':None} )
prt_df= pd.DataFrame({
        'l1_id':list(l1df.reset_index().groupby(by='PRT').agg(list)['index']), 
        'type':'PRT','successor':None} )

# Then concatenate them:
gdf = pd.concat([dna_df,rna_df,prt_df]).reset_index(drop=True)

# Add successor and predecessor information:
for i,r in gdf[gdf.type=='RNA'].iterrows():
    gdf.loc[r.l1_id, 'successor'] = i
for i,r in gdf[gdf.type=='PRT'].iterrows():
    gdf.loc[gdf.loc[r.l1_id].successor, 'successor'] = i

gdf['predecessor'] = [list() for _ in range(len(gdf))]
for i,r in gdf.iterrows():
    if r.successor is not None:
        gdf.loc[r.successor]['predecessor'] += [i]
gdf.loc[~gdf.predecessor.astype(bool), 'predecessor'] = None


# We explicit the part content of each node:
gdf['content'] = gdf.apply(lambda x: l1df.loc[x.l1_id].iloc[0][x.type], axis=1)
gdf['content_type'] = gdf.apply(lambda x: tuple([lib.parts.loc[p][0] for p in x.content]), axis=1)

# And finally add information about useful output of the whole graph:
outputs = ['NeonGreen']

def containsOutput(l,outputs):
    for o in outputs:
        if o in l:
            return True
    return False

gdf['is_output'] = False
gdf.loc[gdf.type == 'PRT', 'is_output'] = gdf.loc[gdf.type == 'PRT'].l1_id.apply(
                lambda x: containsOutput(l1df.loc[x].PRT.tolist()[0],outputs))

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{      --     Plotting the Gene Expression Graph    --
#···············································································

def gdfToGraph(gdf):
    nodes = [ {'id':f'{i}', 'type':n.type, 'data':{'content':n.content, 'content_type':n.content_type}} for i,n in gdf.iterrows()]
    edges = [ { 'id': f'{i}', 'source': f'{i}', 'target': f'{n.successor}'} for i,n in gdf.iterrows() if n.successor]
    return (nodes, edges)

nodes, edges = gdfToGraph(gdf)

h3('Gene expression graph')
grnGraph(nodes, edges)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                 --     Consctruct Compute Graph     --
#···············································································
import logging
from rich.pretty import pprint

b()
h4('Gdf')
ag(gdf)

# Here we generate a dataframe that contains all the available functions for our current library
seqs = lib.sequestrons.merge(lib.sequestron_types, left_on='type', right_index=True)
seqs = ut.decode_json(seqs, ['output_part', 'output_category'])

# make sure that a list has at least i elements and then assign val to the ith element
def set_list_item(lst, i, val):
    if len(lst) <= i:
        lst.extend([None] * (i - len(lst) + 1))
    lst[i] = val

class ComputeNode:
    def __init__(self, id, type, gdf_input, gdf_output):
        self.id = id
        self.type = type
        self.gdf_input = gdf_input
        self.gdf_output = gdf_output
        self.input_from = []
        self.output_to = []


    def removeOutput(self, other):
        other.input_from.remove(self.id)
        for i in range(len(self.output_to)):
            if self.output_to[i][0]==other.id:
                self.output_to.pop(i)
                break



    def toDict(self):
        return {
            "id": self.id,
            "type": self.type,
            "gdf_input": self.gdf_input,
            "gdf_output": self.gdf_output,
            "input_from": self.input_from,
            "output_to": self.output_to
        }

    def __str__(self):
        return str(self.toDict())

    def __repr__(self):
        return str(self.toDict())


unique_id = 0
def uniqueId():
    global unique_id
    unique_id += 1
    return unique_id - 1

def isOutputOf(gdf_input_node, compute_nodes):
    res = [other for other in compute_nodes if gdf_input_node == other.gdf_output]
    return res 

def getNode(nodes, id):
    for node in nodes:
        if node.id == id:
            return node
    raise Exception("Node not found")


# removeShortcuts removes indirect links in the Compute graph, 
# turning it from a directed acyclic graph to a tree.
def removeShortcuts(nodes, root_id):
    labels = {}
    for node in nodes:
        labels[node.id] = 1
    S = set()
    S.add(root_id)
    while len(S) > 0:
        N = getNode(nodes, S.pop())
        w = labels[N.id] + 1
        for d in N.input_from:
            if labels[d] < w:
                labels[d] = w
                S.add(d)
    # remove all edges which connect nodes whose labels differ by more than 1.
    for node in nodes:
        for d in node.input_from:
            if labels[node.id] + 1 < labels[d]:
                getNode(nodes, d).removeOutput(node)


newnodes = []
output_gene_nodes = gdf[gdf.is_output]
onode = ComputeNode(uniqueId(),'out',[],None)
for i, r in output_gene_nodes.iterrows():
    onode.gdf_input += [i]
newnodes.append(onode)

# first we add the sequestron nodes with a list of their gdf input nodes
for _,r in seqs.iterrows():
    nlvl = gdf[gdf.type == r.negative_level]
    nparts = nlvl[nlvl.content.apply(lambda x: r.negative_part in x)]
    plvl = gdf[gdf.type == r.positive_level]
    pparts = plvl[plvl.content.apply(lambda x: r.positive_part in x)]
    olvl = gdf[gdf.type == r.output_level]
    oparts = olvl[olvl.content.apply(lambda x: ut.isSubset(r.output_part,x))]
    if (len(nparts) > 0 and len(pparts) > 0):
        assert(len(pparts) == 1)
        assert(len(nparts) == 1)
        cnode = ComputeNode(uniqueId(), f'sequestron_{r.type}', [nparts.index[0],pparts.index[0]], oparts.index[0])
        newnodes.append(cnode)



# then for each input node, we need to go back up to the original DNA using translation
# and transcription nodes, making sure along the way to connect if it is actually also part 
# of a sequestron node
cg = []
pprint(newnodes)

while newnodes:
    n = newnodes.pop()
    print('-'*120)
    print('-'*120)
    pprint(n)
    if n.type != 'constant':
        # for every gene input of this compute node
        for i, n_inp in enumerate(n.gdf_input):
            others = isOutputOf(n_inp, cg + newnodes)
            print(f'  n_inp = {n_inp}, isOutput = {others}')
            for other in others:
                # set_list_item(n.input_from,i,other.id)
                n.input_from += [other.id]
                other.output_to += [(n.id, i)]
            if not others:
                gn = gdf.loc[n_inp] # input gene
                nid = uniqueId()
                ntype = {'PRT':'translation','RNA':'transcription','DNA':'constant'}[gn.type]
                newn = ComputeNode(nid, ntype, gn.predecessor, n_inp)
                newn.input_from = []
                newn.output_to = [(n.id,i)]
                newnodes.append(newn)
                n.input_from += [nid]
                print('    created new node:',end =" ")
                pprint(newn)
    cg += [n]

removeShortcuts(cg,0)

cdf = pd.DataFrame([n.toDict() for n in cg]).set_index('id').sort_index()
h4('Compute nodes:')
ag(cdf.astype(str))
pprint(cdf)
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                 --     Drawing the Compute Graph     --
#···············································································

nodes = [ {'id':str(i), 'type':n.type, 'data':{'type':n.type}} for i,n in cdf.iterrows()]
edges = [ { 'id': f'edge_{uniqueId()}', 'source': str(i), 'target': str(o), 'targetHandle':str(h)} for i,n in cdf.iterrows() if n.output_to for o,h in n.output_to ]

computeGraph(nodes, edges)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     Archives     --
#···············································································
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{       --     Constructing the Compute Graph from the GEG     --
#···············································································
# b()
# h4('Gdf')
# ag(gdf)

# # Here we generate a dataframe that contains all the available functions for our current library
# seqs = lib.sequestrons.merge(lib.sequestron_types, left_on='type', right_index=True)
# seqs = ut.decode_json(seqs, ['output_part', 'output_category'])



# unique_id = 0
# def uniqueId():
    # global unique_id
    # unique_id += 1
    # return unique_id - 1

# # check if input_id is 
# def isOutputOf(gdf_input_node, compute_nodes):
    # res = [other for other in compute_nodes if gdf_input_node == other['gdf_output']]
    # return res 

# # make sure that a list has at least i elements and then assign val to the ith element
# def set_list_item(lst, i, val):
    # if len(lst) <= i:
        # lst.extend([None] * (i - len(lst) + 1))
    # lst[i] = val

# newnodes = []
# # first we add the sequestron nodes with a list of their gdf input nodes
# for _,r in seqs.iterrows():
    # nlvl = gdf[gdf.type == r.negative_level]
    # nparts = nlvl[nlvl.content.apply(lambda x: r.negative_part in x)]
    # plvl = gdf[gdf.type == r.positive_level]
    # pparts = plvl[plvl.content.apply(lambda x: r.positive_part in x)]
    # olvl = gdf[gdf.type == r.output_level]
    # oparts = olvl[olvl.content.apply(lambda x: ut.isSubset(r.output_part,x))]
    # if (len(nparts) > 0 and len(pparts) > 0):
        # assert(len(pparts) == 1)
        # assert(len(nparts) == 1)
        # cnode = { 'id': uniqueId(), 'type' : f'sequestron_{r.type}', 
                # 'gdf_input':[nparts.index[0],pparts.index[0]], 
                # 'gdf_output':[oparts.index[0]], 
                # 'input_from':[], 'output_to':[] }
        # newnodes.append(cnode)

# # we now add the output through a special output compute node
# # which just assembles all the output proteins into one output vector

# output_gene_nodes = gdf[gdf.is_output]
# onode = { 'id': uniqueId(), 'type' : 'out', 'gdf_input':[], 'gdf_output':None, 'output_to':None, 'input_from':[] }
# for i, r in output_gene_nodes.iterrows():
    # onode['gdf_input'] += [i]
# newnodes.append(onode)

# # then for each input node, we need to go back up to the original DNA using translation
# # and transcription nodes, making sure along the way to connect if it is actually also part 
# # of a sequestron node
# cg = []
# while newnodes:
    # n = newnodes.pop()
    # if n['type'] != 'constant':
        # # for every gene input of this compute node
        # for i, n_inp in enumerate(n['gdf_input']):
            # gn = gdf.loc[n_inp] # input gene
            # other = isOutputOf(n_inp, cg + newnodes)
            # print(f'For node {n["id"]}, gdf_input {n_inp} isOutput res = {other}')
            # if other: # if this node's input is some other compute node's output
                # set_list_item(n['input_from'],i,other[0]['id'])
                # other[0]['output_to'] += [(n['id'], i)]
            # else:
                # nid = uniqueId()
                # ntype = {'PRT':'translation','RNA':'transcription','DNA':'constant'}[gn.type]
                # newn = {'id': nid, 'type':ntype, 
                        # 'gdf_input':gn.predecessor, 'gdf_output':n_inp,
                        # 'output_to':[(n['id'],i)], 'input_from':[]}
                # newnodes.append(newn)
                # n['input_from'] += [nid]
    # cg += [n]

# cdf = pd.DataFrame(cg).set_index('id').sort_index()
# h4('Compute nodes:')
# ag(cdf.astype(str))

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
