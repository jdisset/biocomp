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
    def filterType(n):
        if n['type'] == 'input':
            n['type'] = 'in'
        if n['type'] == 'output':
            n['type'] = 'out'
        return n

    tnodes = [ut.updated_dict(filterType(n),{'data':{'id':n['id']}}) for n in nodes]
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

# l1_DNAs=[['hEF1a','NeonGreen','CasE_recog_5p'],['hEF1a','CasE'],['hEF1a', 'Csy4'],['hEF1a','NeonGreen','Csy4_recog_5p'],['hEF1a','NeonGreen','CasE_recog_5p']]

l1_DNAs= [
        ['hEF1a', 'CasE'],
        ['hEF1a', 'Csy4'],
        ['hEF1a', 'PhiC31', 'Csy4_recog_5p'],
        ['hEF1a', 'PhiC31RDF','CasE_recog_5p'],
        ['hEF1a', 'attP', 'NeonGreen', 'attB']]

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


# We explicitly describe the part content of each node:
gdf['content'] = gdf.apply(lambda x: l1df.loc[x.l1_id].iloc[0][x.type], axis=1)
gdf['content_type'] = gdf.apply(lambda x: tuple([lib.parts.loc[p][0] for p in x.content]), axis=1)

# And finally add information about the output of the whole graph:
outputs = ['NeonGreen']

def containsOutput(l,outputs):
    for o in outputs:
        if o in l:
            return True
    return False

gdf['is_output'] = False
gdf.loc[gdf.type == 'PRT', 'is_output'] = gdf.loc[gdf.type == 'PRT'].l1_id.apply(
                lambda x: containsOutput(l1df.loc[x].PRT.tolist()[0],outputs))

gdf['is_input'] = None
gdf.is_input.iloc[0] = 0
gdf.is_input.iloc[1] = 1

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{      --     Plotting the Gene Expression Graph    --
#···············································································

def gdfToGraph(gdf):
    nodes = [ {'id':f'{i}', 'type':n.type, 'data':n.to_dict()} for i,n in gdf.iterrows()]
    edges = [ { 'id': f'{i}', 'source': f'{i}', 'target': f'{n.successor}'} for i,n in gdf.iterrows() if n.successor]
    return (nodes, edges)

nodes, edges = gdfToGraph(gdf)

h3('Gene expression graph')
ag(gdf)
grnGraph(nodes, edges)
b()

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{             --     Consctructing the Compute Graph     --
#···············································································
import logging
from rich.pretty import pprint


# Here we generate a dataframe that contains all the available functions for our current library
seqs = lib.sequestrons.merge(lib.sequestron_types, left_on='type', right_index=True)
seqs = ut.decode_json(seqs, ['output_part', 'output_category'])

# make sure that a list has at least i elements and then assign val to the ith element
def set_list_item(lst, i, val):
    if len(lst) <= i:
        lst.extend([None] * (i - len(lst) + 1))
    lst[i] = val

class GraphComputeNode:
    def __init__(self, id, type, gdf_input, gdf_output):
        self.id = id
        self.type = type
        self.gdf_input = gdf_input
        self.gdf_output = gdf_output if gdf_output is not None else -1
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


unique_id = int(0)
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
onode = GraphComputeNode(uniqueId(),'output',[],None)
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
        # if (len(nparts) > 1): pprint(nparts)
        if (len(pparts) > 1): pprint(pparts)
        assert(len(pparts) == 1)
        assert(len(nparts) == 1)
        cnode = GraphComputeNode(uniqueId(), f'sequestron_{r.type}', [int(nparts.index[0]),int(pparts.index[0])], int(oparts.index[0]))
        newnodes.append(cnode)



# then for each input node, we need to go back up to the original DNA using translation
# and transcription nodes, making sure to connect it to relevant sequestron nodes along the way
cg = []

while newnodes:
    n = newnodes.pop()
    if n.type != 'bias':
        # for every gene input of this compute node
        for i, n_inp in enumerate(n.gdf_input):
            others = isOutputOf(n_inp, cg + newnodes)
            print(f'  n_inp = {n_inp}, isOutput = {others}')
            for other in others:
                set_list_item(n.input_from,i,other.id)
                # n.input_from += [other.id]
                other.output_to += [(n.id, i)]
            if not others:
                gn = gdf.loc[n_inp] # input gene
                nid = uniqueId()
                ntype = {'PRT':'translation','RNA':'transcription','DNA':'bias'}[gn.type]
                newn = GraphComputeNode(nid, ntype, gn.predecessor, int(n_inp))
                newn.input_from = []
                newn.output_to = [(n.id,i)]
                newnodes.append(newn)
                n.input_from += [int(nid)]
    cg += [n]

removeShortcuts(cg,0) # turns the graph back into a tree

cdf = pd.DataFrame([n.toDict() for n in cg]).set_index('id').sort_index()

# add input ids
cdf['is_input'] = None
for index, row in cdf.iterrows():
    if row['type'] == 'bias':
        input_id = gdf.at[row['gdf_output'], 'is_input']
        if input_id is not None:
            cdf.at[index, 'type'] = 'input'
            cdf.at[index, 'is_input'] = input_id

pprint(cdf)
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                 --     Drawing the Compute Graph     --
#···············································································
cdf.input_from = cdf.input_from.apply(lambda x: None if x is None else [int(e) for e in x])
cdf.output_to = cdf.output_to.apply(lambda x: None if x is None else [(int(i),h) for i,h in x])


nodes = [ {'id':str(i), 'type':n.type, 'data':n.to_dict()} for i,n in cdf.iterrows()]
edges = [ { 'id': f'edge_{uniqueId()}', 'source': str(i), 'target': str(o), 'targetHandle':str(h)} for i,n in cdf.iterrows() if n.output_to for o,h in n.output_to ]


h3('Compute nodes:')
computeGraph(nodes, edges)
ag(cdf.astype(str))

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{         --     Definition of JAX compute nodes     --
#···············································································

import jax
import jax.numpy as jnp


DEFAULT_RNA_DEG_RATE = 1.0
DEFAULT_PRT_DEG_RATE = 1.0

DEFAULT_MIN_RATE = 0.0
DEFAULT_MAX_RATE = 1.0

DEFAULT_MIN_COPY_N = 0.0
DEFAULT_MAX_COPY_N = 1.0

def rate_init_continuous(rng, n, minval=DEFAULT_MIN_RATE, maxval=DEFAULT_MAX_RATE):
    return jax.random.uniform(key = rng, shape=(n,), minval=minval, maxval=maxval, dtype=jnp.float32)

def copy_n_init(rng, minval=DEFAULT_MIN_COPY_N, maxval=DEFAULT_MAX_COPY_N):
    return jax.random.uniform(key = rng, minval=minval, maxval=maxval, dtype=jnp.float32)


# each node type is a function that returns 2 other functions:
# - init(rng, n_inputs) -> returns the parameters (this node, others)
# - apply(params, X) -> returns the value of the compute node  
#                       X, the inputs, is only useful for the input leaves


CNODE = {}

def debug(f):
    def wrap(*args, **kwargs):
        print(f'{f.__name__} called with args: ')
        pprint([*args])
        pprint({**kwargs})
        return f(*args, **kwargs)
    return wrap

def compnode(f):
    CNODE[f.__name__] = f
    return f

def apply_upstream(params, apply_funs, inputs, **kwargs):
    nbranches = len(apply_funs)
    rng = kwargs.pop('rng', None) # we transmit rngs upstream as some apply functions might need randomness
    rngs = random.split(rng, nbranches) if rng is not None else (None,) * nbranches
    return jnp.array([f(p, inputs, rng=r, **kwargs) for f, p, r in zip(apply_funs, params, rngs)])

def init_upstream(rng, init_funs):
    nbranches = len(init_funs)
    rngs = random.split(rng, nbranches)
    return [init(rng) for init, rng in zip(init_funs, rngs)]


@compnode
def transcription(*branches, deg_rate = DEFAULT_RNA_DEG_RATE):
    nbranches= len(branches)
    init_funs, apply_funs = zip(*branches)
    def init(rng): 
        return (rate_init_continuous(rng, nbranches), init_upstream(rng, init_funs))
    def apply(params, inputs, **kwargs):
        (t_rates, others) = params
        return jnp.dot(apply_upstream(others, apply_funs, inputs, **kwargs), t_rates) / deg_rate
    return init, apply

@compnode
def translation(*branches, deg_rate = DEFAULT_PRT_DEG_RATE):
    return transcription(*branches, deg_rate=deg_rate)

@compnode
def sequestron_ERN(neg, pos):
    def init(rng): 
        return init_upstream(rng, (neg[0], pos[0]))
    def apply(params, inputs, **kwargs):
        res = apply_upstream(params, (neg[1], pos[1]), inputs, **kwargs)
        return jnp.maximum(0, res[1] - res[0])
    return init, apply

@compnode
def sequestron_RECOMBINASE(neg, pos):
    return sequestron_ERN(neg,pos)

@compnode
def bias(*_):
    def init_fun(rng):
        return copy_n_init(rng)
    def apply_fun(copy_n, inputs, **kwargs):
        return copy_n
    return init_fun, apply_fun

@compnode
def input(id):
    def init_fun(rng):
        return copy_n_init(rng)
    def apply_fun(copy_n, inputs, **kwargs):
        return inputs[id] * copy_n
    return init_fun, apply_fun


@compnode
def output(*branches): # simply returns the vector of results from all branches
    init_funs, apply_funs = zip(*branches)
    def init(rng): 
        return init_upstream(rng, init_funs)
    def apply(params, inputs, **kwargs):
        return apply_upstream(params, apply_funs, inputs, **kwargs)
    return init, apply

@compnode
def constant(v=1.0):
    def init_fun(rng):
        return None
    def apply_fun(*args, **kwargs):
        return v
    return init_fun, apply_fun

@compnode
def plus(*branches):
    init_funs, apply_funs = zip(*branches)
    def init(rng): 
        return init_upstream(rng, init_funs)
    def apply(params, inputs, **kwargs):
        return jnp.sum(apply_upstream(params, apply_funs, inputs, **kwargs))
    return init, apply




init, apply = CNODE['output'](
                CNODE['plus'](CNODE['constant'](2),constant(5),input(0)),
                CNODE['input'](1),
                plus(input(2),translation(bias(),input(3)))
                )


rng = jax.random.PRNGKey(10)
p = init(rng)

apply(p,[0,1,2,3])



#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{              --     Building the JAX compute tree     --
#···············································································



outNode = cdf[cdf.type=='output'].iloc[0]

def buildTree(node):
    if node.input_from: # recursive case: any non-input node
        branches = cdf.loc[node.input_from]
        return CNODE[node.type](*[buildTree(b) for _,b in branches.iterrows()])
    return CNODE[node.type](node.is_input) # terminal node


init, compute = buildTree(outNode)

rng = jax.random.PRNGKey(1)
p = init(rng)
pprint(p)
compute(p, [0.2, 0.1])

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────



## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     Archives     --
#···············································································

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
