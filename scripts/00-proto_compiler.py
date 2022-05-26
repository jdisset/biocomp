## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{            --     imports, basic tools, general config    --
# ···············································································
# -- config

PRINT_DATABASE = True
SHEET_KEY = '1K_2bt90E-Wk-A9PYGXGbKDJy-olojKtksy1jxCQAzME'


# -- imports
import streamlit as st

st.set_page_config(layout="wide")

# %load_ext autoreload
# %autoreload 2

import utils as ut

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np
from types import SimpleNamespace
from copy import deepcopy

from tqdm import tqdm

# JAX stuff
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random, lax, tree_map
from jax import tree_util as pytree
from jax.tree_util import Partial as partial
from jax.experimental import host_callback
from jax.example_libraries.optimizers import adam
from time import time

# Rich logging
from rich import print
from rich.pretty import Pretty

import matplotlib.pyplot as plt

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{              --     Streamlit utils and components     --
# ···············································································
from st_aggrid import AgGrid


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
    AgGrid(
        df.reset_index(), fit_columns_on_grid_load=True, theme='light', height=(len(df) + 1) * rowH
    )


# -- custom streamlit components
import streamlit.components.v1 as components

if not ut.is_interactive():
    _component_func = components.declare_component("ned_component", url="http://localhost:1234")
else:
    _component_func = lambda: None


def grnGraph(nodes, edges, key=None, func=_component_func):
    tnodes = [ut.updated_dict(n, {'data': {'id': n['id']}}) for n in nodes]
    return func(nodes=tnodes, edges=edges, output_type='GRN', key=key)  # {{{}}}


def computeGraph(nodes, edges, key=None, func=_component_func, **kwargs):
    def filterType(n):
        if n['type'] == 'input':
            n['type'] = 'in'
        if n['type'] == 'output':
            n['type'] = 'out'
        return n

    tnodes = [filterType(n) for n in nodes]
    return func(nodes=tnodes, edges=edges, output_type='COMPUTE', key=key, **kwargs)


def dnaOutput(nodes, key=None, func=_component_func, **kwargs):
    tnodes = [ut.updated_dict(n, {'data': {'id': n['id']}}) for n in nodes if n['type'] == 'DNA']
    return func(nodes=tnodes, output_type='DNA', key=key, **kwargs)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{             --     Loading library from google sheet     --
# ···············································································

if 'lib' not in st.session_state:
    lib = ut.getAllGoogleSheets(SHEET_KEY)
    lib = SimpleNamespace(**lib)
    md(f'Loaded library with {len(lib.__dict__)} tables: ' + ', '.join(lib.__dict__.keys()))
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
# ···············································································

p = lib.parts
c = lib.categories
pc = pd.merge(p, c, left_on='category', right_index=True, how='left')


def getRna(dna):
    d = pc.loc[dna]
    return tuple(d[d.transcripted == 1].index)


def getPrt(dna):
    d = pc.loc[dna]
    return tuple(d[d.translated == 1].index)


# l1_DNAs=[['hEF1a','NeonGreen','CasE_recog_5p'],['hEF1a','CasE'],['hEF1a', 'Csy4'],['hEF1a','NeonGreen','Csy4_recog_5p'],['hEF1a','NeonGreen','CasE_recog_5p']]

l1_DNAs = [
    ['hEF1a', 'PhiC31RDF', 'CasE_recog_5p'],
    ['hEF1a', 'Csy4'],
    ['hEF1a', 'PhiC31RDF', 'CasE_recog_5p'],
    ['hEF1a', 'Csy4'],
    # biases:
    ['hEF1a', 'CasE'],
    ['hEF1a', 'PhiC31RDF', 'Csy4_recog_5p'],
    ['hEF1a', 'PhiC31'],
    # output
    ['hEF1a', 'attP', 'NeonGreen', 'attB'],
]

# l1_DNAs= [
# ['hEF1a', 'CasE'],
# ['hEF1a', 'NeonGreen','CasE_recog_5p']]

# l1_DNAs= [['hEF1a', 'NeonGreen']]

l1 = [{'DNA': tuple(d), 'RNA': getRna(d), 'PRT': getPrt(d)} for d in l1_DNAs]
l1df = pd.DataFrame(l1)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{    --     Constructing the Gene Expression Graph from the L1s    --
# ···············································································

# ------------------------------------------------------------------------------
# gdf is the dataframe that contains all the information required to plot the gene
# expression graph, which is just a graph where each node represents a
# DNA, RNA or PRT molecule
# ------------------------------------------------------------------------------

# To build it, we first create the dataframes containing the dna, rna and prt contents
# deduced from the l1 constructs. We also merge rna and prt nodes with identical content:
dna_df = pd.DataFrame({'l1_id': [[x] for x in l1df.index], 'type': 'DNA', 'successor': None})
rna_df = pd.DataFrame(
    {
        'l1_id': list(l1df.reset_index().groupby(by='RNA').agg(list)['index']),
        'type': 'RNA',
        'successor': None,
    }
)
prt_df = pd.DataFrame(
    {
        'l1_id': list(l1df.reset_index().groupby(by='PRT').agg(list)['index']),
        'type': 'PRT',
        'successor': None,
    }
)

# Then concatenate them:
gdf = pd.concat([dna_df, rna_df, prt_df]).reset_index(drop=True)

# Add successor and predecessor information:
for i, r in gdf[gdf.type == 'RNA'].iterrows():
    gdf.loc[r.l1_id, 'successor'] = i
for i, r in gdf[gdf.type == 'PRT'].iterrows():
    gdf.loc[gdf.loc[r.l1_id].successor, 'successor'] = i

gdf['predecessor'] = [list() for _ in range(len(gdf))]
for i, r in gdf.iterrows():
    if r.successor is not None:
        gdf.loc[r.successor]['predecessor'] += [i]
gdf.loc[~gdf.predecessor.astype(bool), 'predecessor'] = None


# We explicitly describe the part content of each node:
gdf['content'] = gdf.apply(lambda x: l1df.loc[x.l1_id].iloc[0][x.type], axis=1)
gdf['content_type'] = gdf.apply(lambda x: tuple([lib.parts.loc[p][0] for p in x.content]), axis=1)

# And finally add information about the output of the whole graph:
outputs = ['NeonGreen']


def containsOutput(l, outputs):
    for o in outputs:
        if o in l:
            return True
    return False


gdf['is_output'] = False
gdf.loc[gdf.type == 'PRT', 'is_output'] = gdf.loc[gdf.type == 'PRT'].l1_id.apply(
    lambda x: containsOutput(l1df.loc[x].PRT.tolist()[0], outputs)
)

gdf['is_input'] = None
gdf.is_input.iloc[0] = 0
gdf.is_input.iloc[1] = 0
gdf.is_input.iloc[2] = 1
gdf.is_input.iloc[3] = 1

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{      --     Plotting the Gene Expression Graph    --
# ···············································································


def gdfToGraph(gdf):
    nodes = [{'id': f'{i}', 'type': n.type, 'data': n.to_dict()} for i, n in gdf.iterrows()]
    edges = [
        {'id': f'{i}', 'source': f'{i}', 'target': f'{n.successor}'}
        for i, n in gdf.iterrows()
        if n.successor
    ]
    return (nodes, edges)


nodes, edges = gdfToGraph(gdf)

h3('Gene expression graph')
ag(gdf)
dnaOutput(nodes, initexpanded=True)
grnGraph(nodes, edges)
b()

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{             --     Consctructing the Compute Graph     --
# ···············································································

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
            if self.output_to[i][0] == other.id:
                self.output_to.pop(i)
                break

    def toDict(self):
        return {
            "id": self.id,
            "type": self.type,
            "gdf_input": self.gdf_input,
            "gdf_output": self.gdf_output,
            "input_from": self.input_from,
            "output_to": self.output_to,
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
onode = GraphComputeNode(uniqueId(), 'output', [], None)
for i, r in output_gene_nodes.iterrows():
    onode.gdf_input += [i]
newnodes.append(onode)

# first we add the sequestron nodes with a list of their gdf input nodes
for _, r in seqs.iterrows():
    nlvl = gdf[gdf.type == r.negative_level]
    nparts = nlvl[nlvl.content.apply(lambda x: r.negative_part in x)]
    plvl = gdf[gdf.type == r.positive_level]
    pparts = plvl[plvl.content.apply(lambda x: r.positive_part in x)]
    olvl = gdf[gdf.type == r.output_level]
    oparts = olvl[olvl.content.apply(lambda x: ut.isSubset(r.output_part, x))]
    if len(nparts) > 0 and len(pparts) > 0:
        # if (len(nparts) > 1): print(nparts)
        if len(pparts) > 1:
            print(pparts)
        assert len(pparts) == 1
        assert len(nparts) == 1
        cnode = GraphComputeNode(
            uniqueId(),
            f'sequestron_{r.type}',
            [int(nparts.index[0]), int(pparts.index[0])],
            int(oparts.index[0]),
        )
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
                # set_list_item(n.input_from,i,other.id)
                n.input_from += [other.id]
                other.output_to += [(n.id, i)]
            if not others:
                gn = gdf.loc[n_inp]  # input gene
                nid = uniqueId()
                ntype = {'PRT': 'translation', 'RNA': 'transcription', 'DNA': 'bias'}[gn.type]
                newn = GraphComputeNode(nid, ntype, gn.predecessor, int(n_inp))
                newn.input_from = []
                newn.output_to = [(n.id, i)]
                newnodes.append(newn)
                n.input_from += [int(nid)]
    cg += [n]

removeShortcuts(cg, 0)  # turns the graph back into a tree

cdf = pd.DataFrame([n.toDict() for n in cg]).set_index('id').sort_index()

# add input ids
cdf['is_input'] = None
for index, row in cdf.iterrows():
    if row['type'] == 'bias':
        input_id = gdf.at[row['gdf_output'], 'is_input']
        if input_id is not None:
            cdf.at[index, 'type'] = 'input'
            cdf.at[index, 'is_input'] = int(input_id)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                 --     Drawing the Compute Graph     --
# ···············································································
import json


def np_converter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()


def make_json_compatible(o):
    return json.loads(json.dumps(o, default=np_converter))


def draw_compute_graph(df, func=None, **kwargs):
    nodes = [
        {'id': str(i), 'type': n.type, 'data': ut.updated_dict(n.to_dict(), {'id': i})}
        for i, n in df.iterrows()
    ]
    edges = [
        {
            'id': f'edge_{uniqueId()}',
            'source': str(i),
            'target': str(o),
            'targetHandle': str(h),
            'data': {
                'srcdata': df.loc[i].to_dict(),
                'tgtdata': df.loc[o].to_dict(),
                'tgthandle': str(h),
            },
        }
        for i, n in df.iterrows()
        if n.output_to
        for o, h in n.output_to
    ]
    if func is None:
        return computeGraph(make_json_compatible(nodes), make_json_compatible(edges), **kwargs)
    else:
        return computeGraph(
            make_json_compatible(nodes), make_json_compatible(edges), func=func, **kwargs
        )


h3('Compute nodes:')
draw_compute_graph(cdf)
ag(cdf.astype(str))

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{         --     Definition of JAX compute nodes     --
# ···············································································

DEFAULT_RNA_DEG_RATE = 1.0
DEFAULT_PRT_DEG_RATE = 1.0

DEFAULT_MIN_RATE = 0.0
DEFAULT_MAX_RATE = 1.0

DEFAULT_MIN_COPY_N = 0.0
DEFAULT_MAX_COPY_N = 1.0


def rate_init_continuous(rng, n, minval=DEFAULT_MIN_RATE, maxval=DEFAULT_MAX_RATE):
    return jax.random.uniform(key=rng, shape=(n,), minval=minval, maxval=maxval, dtype=jnp.float32)


def copy_n_init(rng, minval=DEFAULT_MIN_COPY_N, maxval=DEFAULT_MAX_COPY_N):
    return jax.random.uniform(key=rng, minval=minval, maxval=maxval, dtype=jnp.float32)


# each node type is a function that returns 2 other functions:
# - init(rng, n_inputs) -> returns the parameters (this node, others)
# - apply(params, X) -> returns the value of the compute node
#                       X, the inputs, is only useful for the input leaves
# - assign(params, dic) -> assign the param values to the dict[key] node
# TODO: add constrain functions to add parameter constraints?


CNODE = {}


def debug(f):
    def wrap(*args, **kwargs):
        print(f'{f.__name__} called with args: ')
        print([*args])
        print({**kwargs})
        return f(*args, **kwargs)

    return wrap


def compnode(f):
    CNODE[f.__name__] = f
    return f


def init_upstream(rng, init_funs):
    nbranches = len(init_funs)
    rngs = random.split(rng, nbranches)
    return [init(rng) for init, rng in zip(init_funs, rngs)]


def apply_upstream(params, apply_funs, inputs, **kwargs):
    nbranches = len(apply_funs)
    rng = kwargs.pop(
        'rng', None
    )  # we transmit rngs upstream as some apply functions might need randomness
    rngs = random.split(rng, nbranches) if rng is not None else (None,) * nbranches
    return jnp.array([f(p, inputs, rng=r, **kwargs) for f, p, r in zip(apply_funs, params, rngs)])


def assign_upstream(params, assign_funs, D):
    for f, p in zip(assign_funs, params):
        f(p, D)


def constrain_upstream(params, constrain_funs):
    return [constrain(p) for constrain, p in zip(constrain_funs, params)]


@compnode
def transcription(*branches, deg_rate=DEFAULT_RNA_DEG_RATE, nid=None):
    nbranches = len(branches)
    init_funs, apply_funs, assign_funs, constrain_funs = zip(*branches)

    def init(rng):
        return (rate_init_continuous(rng, nbranches), init_upstream(rng, init_funs))

    def constrain(params):
        t_rates, others = params
        t_rates = jnp.clip(t_rates, 0.0, 1.0)
        return (t_rates, constrain_upstream(others, constrain_funs))

    def apply(params, inputs, **kwargs):
        t_rates, others = params
        return jnp.dot(apply_upstream(others, apply_funs, inputs, **kwargs), t_rates) / deg_rate

    def assign(params, D):
        t_rates, others = params
        if nid is not None:
            D[nid] = {'tr_rates': np.array(t_rates)}
        assign_upstream(others, assign_funs, D)

    return init, constrain, apply, assign


@compnode
def translation(*branches, deg_rate=DEFAULT_PRT_DEG_RATE, **kwargs):
    return transcription(*branches, deg_rate=deg_rate, **kwargs)


@compnode
def sequestron_ERN(neg, pos, nid=None):

    ini, con, app, ass = zip(neg, pos)

    def init(rng):
        return init_upstream(rng, ini)

    def constrain(params):
        return constrain_upstream(params, con)

    def apply(params, inputs, **kwargs):
        res = apply_upstream(params, app, inputs, **kwargs)
        return jnp.maximum(0, res[1] - res[0])

    def assign(params, D):
        assign_upstream(params, ass, D)

    return init, constrain, apply, assign


@compnode
def sequestron_RECOMBINASE(neg, pos, **kwargs):
    return sequestron_ERN(neg, pos, **kwargs)


@compnode
def bias(*_, nid=None):
    def init(rng):
        return copy_n_init(rng)

    def constrain(copy_n):
        copy_n = jnp.clip(copy_n, 0.0, 1.0)
        return copy_n

    def apply(copy_n, inputs, **kwargs):
        return copy_n

    def assign(copy_n, D):
        if nid is not None:
            D[nid] = {'copy_number': float(copy_n)}

    return init, constrain, apply, assign


@compnode
def input(id, nid=None):
    def init(rng):
        return copy_n_init(rng)

    def constrain(copy_n):
        copy_n = jnp.clip(copy_n, 0.0, 1.0)
        return copy_n

    def apply(copy_n, inputs, **kwargs):
        return inputs[id] * copy_n

    def assign(copy_n, D):
        if nid is not None:
            D[nid] = {'copy_number': float(copy_n)}

    return init, constrain, apply, assign


@compnode
def output(*branches, nid=None):  # simply returns the vector of results from all branches
    init_funs, constrain_funs, apply_funs, assign_funs = zip(*branches)

    def init(rng):
        return init_upstream(rng, init_funs)

    def constrain(params):
        return constrain_upstream(params, constrain_funs)

    def apply(params, inputs, **kwargs):
        return apply_upstream(params, apply_funs, inputs, **kwargs)

    def assign(params, D):
        assign_upstream(params, assign_funs, D)

    return init, constrain, apply, assign


def buildTree(cdf):
    outNode = cdf[cdf.type == 'output'].iloc[0]

    def buildImpl(node):
        if node.input_from:  # recursive case: any non-input node
            branches = cdf.loc[node.input_from]
            return CNODE[node.type](*[buildImpl(b) for _, b in branches.iterrows()], nid=node.name)
        return CNODE[node.type](node.is_input, nid=node.name)  # terminal node

    init_tree, constrain_fun, apply_fun, assign = buildImpl(outNode)
    return (init_tree, constrain_fun, jit(apply_fun), assign)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     Training functions --
# ···············································································
from jax import value_and_grad


def base_step(i, state, dlossfunc, get_params, update, model, x, y_true):
    params = get_params(state)
    loss, g = dlossfunc(params, model, x, y_true)
    return (update(i, g, state), loss)


def mseloss(params, model, x, y_true):
    y_preds = vmap(pytree.Partial(model, params))(x)
    return jnp.mean(jnp.power(y_preds - y_true, 2))


dmseloss = value_and_grad(mseloss)


def make_training_start(params_initializer, state_initializer, stepfunc, n_steps):
    # print("Compiling..")
    @ut.tqdm_scan(n_steps, 'Training model')
    def scannable_step(previous_state, iteration):
        new_state, loss = stepfunc(iteration, previous_state)
        return new_state, (loss, previous_state)

    def train_one_start(key):
        params = params_initializer(key)
        initial_state = state_initializer(params)
        final_state, states_and_losses_history = lax.scan(
            scannable_step, initial_state, np.arange(n_steps)
        )
        losses, sthists = states_and_losses_history
        return (losses, ut.tree_append(sthists, final_state))

    return train_one_start


# training parameters
N_INITIALIZATIONS = 20
N_TRAINING_STEPS = 500
LEARNING_RATE = 1e-2


def trainComputeGraph(
    cdf,
    key,
    X,
    y_true,
    learning_rate=LEARNING_RATE,
    n_init=N_INITIALIZATIONS,
    n_steps=N_TRAINING_STEPS,
):
    initialization_keys = random.split(key, n_init)

    # generate compute tree functions from dataframe
    init_params, constrain, model, assign = buildTree(cdf)

    # compiled training functions
    opt_init, update, get_params = adam(step_size=learning_rate)  # optimizer
    step = jit(
        partial(
            base_step,
            get_params=get_params,
            dlossfunc=dmseloss,
            update=update,
            model=model,
            x=X,
            y_true=y_true,
        )
    )
    train_fun = make_training_start(init_params, opt_init, step, n_steps)

    # actual training "loop"
    start = time()
    loss_state_histories = vmap(train_fun)(initialization_keys)
    end = time()
    print('Trained in', end - start)

    losses, stacked_states = loss_state_histories
    return (model, get_params(stacked_states), losses, assign)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                   --     getting training data     --
# ···············································································
import matplotlib.image as mpimg

target = mpimg.imread('../data/band_pass_dec.png')[::-1]

# plt.imshow(target, origin='lower')
# plt.show()


TARGET_OUTPUT = 0.5
N_SAMPLES = 3000
samples = []
key = jax.random.PRNGKey(42)
X = jax.random.uniform(key=key, shape=(N_SAMPLES, 2)) * jnp.array(target.shape[:2])
y_true = jnp.array(target[X.astype(int)[:, 1], X.astype(int)[:, 0], 0] * TARGET_OUTPUT).reshape(
    -1, 1
)
X = X / jnp.array(target.shape[:2])

# plt.scatter(X[:,0], X[:,1], c=y_true)
# plt.show()

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     Training     --
# ···············································································
# training data


model, stacked_params, losses, assign_f = trainComputeGraph(
    cdf, key, X, y_true, n_init=20, n_steps=200, learning_rate=0.003
)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{             --     Adding trained parameters to CDF    --
# ···············································································

best_run = np.argmin(losses[:, -1])
best_losses = losses[best_run]
stacked_best = ut.get_pytree(stacked_params, best_run)
best_params = ut.get_pytree(stacked_best, len(best_losses))
best_hist = ut.param_unstack(stacked_best, len(best_losses) + 1)


def assign_params_to_dataframe(params, df, assign_f):
    D = {}
    assign_f(params, D)
    df['parameters'] = None
    for row_id in D:
        df.loc[row_id, 'parameters'] = [deepcopy(D[row_id])]


hist_dfs = [cdf.copy() for _ in best_hist]
for h, p in zip(hist_dfs, best_hist):
    assign_params_to_dataframe(p, h, assign_f)

# h3('Compute nodes after training:')
# draw_compute_graph(hist_dfs[-1])
# ag(hist_dfs[-1].astype(str))

# print(best_params)


def plotBestLoss(best, others, outfile=None):
    fig, a = plt.subplots(1, 1, figsize=(6, 5))
    for l in others:
        a.plot(l, color="#aaaaaa", linewidth=1)
    a.plot(best, color="red", linewidth=2.5)
    if outfile is not None:
        fig.savefig(outfile, dpi=100)
        plt.close()
    else:
        plt.show()


# for i,l in tqdm(list(enumerate([best_losses[:n] for n in range(0,len(best_losses),10)]))):
# plotBestLoss(l,losses,f'./losses2/{i}.png')


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                 --     decision boundaries plots     --
# ···············································································

from p_tqdm import p_umap, p_map

figsize = 10
alp = 1
bgalp = 0.65
MESHRES = 500
scatsize = 6 * figsize

blist = np.array([[0.073, 0.292, 0.419], [0.844, 0.1, 0.111]])
flist = [[0.0, 0.493, 0.579], [0.896, 0.866, 0.806], [0.844, 0.1, 0.111]]
bcmap = ListedColormap(blist)
cmap = LinearSegmentedColormap.from_list("", flist)

XX, YY = np.meshgrid(np.linspace(0, 1, MESHRES), np.linspace(0, 1, MESHRES), indexing='xy')
coords = np.column_stack((XX.ravel(), YY.ravel()))

mseloss(best_params, model, X, y_true)

plt.rcParams["axes.grid"] = False


def drawOnlyPred(a, fig, XX, YY, ZZ, cmap='Reds'):
    pc = a.pcolormesh(XX, YY, ZZ, cmap=cmap, shading='auto', vmin=0, vmax=0.6)
    a.contour(XX, YY, ZZ, [0.5], colors='black', linewidths=1, alpha=0.65)
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.xaxis.set_ticks([])
    a.yaxis.set_ticks([])
    a.set_aspect('equal')
    a.set_xlabel('predicted')
    cax = a.inset_axes([1.04, 0.2, 0.05, 0.6], transform=a.transAxes)
    fig.colorbar(pc, ax=a, cax=cax)


def savePred(i, p):
    ZZ = vmap(pytree.Partial(model, p))(coords).reshape(XX.shape)
    fig, a = plt.subplots(1, 1, figsize=(12, 10))
    drawOnlyPred(a, fig, XX, YY, ZZ)
    fig.savefig(f'./predict2/{i}.png', dpi=120)
    plt.close()


# p_umap(savePred, *zip(*list(enumerate(best_hist[::400]))))

for i, p in tqdm(list(enumerate(best_hist[::10]))):
    savePred(i, p)

# print()

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     screencaptures     --
# ···············································································
# import nest_asyncio
# nest_asyncio.apply()
# ut.screenCaptures(partial(draw_compute_graph, height=2000), hist_dfs[::10], out_dir_path='./outbiased2', height=2000, width=1500)
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     Archives     --
# ···············································································

# fl,t = jax.tree_util.tree_flatten(final_params)
# print([jnp.append(a,b) if b.ndim == 0 else jnp.vstack((a,b)) for a,b in zip(fl,op)])
# merged = [jnp.concatenate([a,jnp.array([b])]) for a,b in zip(fl,op)]
# print([jnp.vstack((a,b)) for a,b in zip(fl,op)])
# jnp.append(jnp.array([1.0,2.0]), jnp.array(1.0))
# jnp.concatenate((jnp.array([[[1.0]]]), jnp.array([[1.0]])))
# print(jax.tree_util.tree_unflatten(t,fl))

# def add_parameters_to_dataframe(D, P):
# for row_id in D:
# for param_name in D[row_id]:
# if param_name not in P.columns:
# P[param_name] = None
# P.loc[row_id, param_name] = D[row_id][param_name]

# make sure node ids are int
# cdf.input_from = cdf.input_from.apply(lambda x: None if x is None else [int(e) for e in x])
# cdf.output_to = cdf.output_to.apply(lambda x: None if x is None else [(int(i),h) for i,h in x])
# cdf.gdf_output = cdf.gdf_output.apply(lambda x: int(x))

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
