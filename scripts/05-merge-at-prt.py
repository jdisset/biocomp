## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     import and init     --
# ···············································································
# %load_ext autoreload
# %autoreload 2

import streamlit as st

st.set_page_config(layout='wide')

import pandas as pd
import numpy as np
import jax.numpy as jnp
import sqlite3
import os

from collections import defaultdict
import jax
from jax import jit, vmap, grad
import scriptutils as ut
import biocomp.utils as bu
from functools import partial
import biocomp as bc
import json
from rich import print

l = ut.load("all_sheets.pickle")
lib = bc.PartsLibrary(l.parts, l.L0s, l.L1s, l.L2s, l.categories, l.sequestrons, l.sequestron_types)
series_obj = json.load(open("../XP/example_xpfile.json"))

series = bc.xp_series_from_json(series_obj, lib)


# series['L2_all'].build_central_dogma_graph(lib)
#
# Hierarchy of nodes: Aggregations -> sources (aka plasmid) -> transcription units -> rest of graph.

# TODO:
# - rewrite compute graph / cdg so that unresolved nodes are considered different (and thus not merged)
# - link conditionning of node parameters to content of the edge, not TU_id (so basically link it to cdg_node_id)

# If we dont care about conditionning the values of tr_rates, deg_rates, etc, on the TUs
# we can keep the merging algorithm "as is" and just add a rule that says that unresolved nodes
# are always considered different.
# ACTUALLY this should also work with conditionning...
# Things that are merged TRULY are the same in terms of content, so they should have the exact same values.
# YES THIS IS THE WAY, let's just add a rule that we don't merge things that contain unresolved slots.
# Then, the conditionning should be actually written so that it works with content

# previous thinking:
# -------------------------------
# THE GOAL HERE IS TO REWRITE AND COMBINE CENTRAL DOGMA GRAPH + COMPUTE GRAPH
# it stems from the need to handle case where an edge is from 2 different TUs.
# We actually have a wider range of effective weights because a single edge is actually a combination of
# multiple TUs. Example:
# a same ERN recog produced by 2 separate TUs... After the ERN node, we have a single edge, but the translation rate should actually be the combination of 2 nodes...
# However, if we want to be able to condition some values on the TUs (such as degradation rates),
# we have to keep one edge per TU.
# But how does that work for an ERN node for example? An ERN node will apply to multiple edges...
# We could consider that there are several ERN nodes in parallell, as many as there are edges.
# A decent way of dealing with this is to actually not merge anything until the PRT level.

# Let's try to write a function that produces the compute graph


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                            --     cdg     --
# ···············································································

network = series['L2_pGW0042+CasE-R']
# these functions also return a boolean that indicates if there are still some parameters at this level
# (preventing the nodes from being merged)
def getDna(tu):
    content, params = [], []
    for s in tu.slots:
        assert s.is_resolved
        if not isinstance(s.part, list):
            content.append(s.part)
        else:
            params += s.part
    return content, params


def getRna(tu, lib):
    dna, params = getDna(tu)
    d = lib.pc.loc[dna]
    p = lib.pc.loc[params]
    return (tuple(d[d.transcripted == 1].index), tuple(p[p.transcripted == 1].index))


def getPrt(tu, lib):
    dna, params = getDna(tu)
    d = lib.pc.loc[dna]
    p = lib.pc.loc[params]
    return (tuple(d[d.translated == 1].index), tuple(p[p.translated == 1].index))


# from resolved TUs, build compute graph
tu = []
for tuid, t in zip(network.tuids, network.transcription_units):
    dna, dna_params = tuple(tuple(l) for l in getDna(t))
    rna, rna_params = getRna(t, lib)
    prt, prt_params = getPrt(t, lib)
    tu.append(
        {
            'name': tuid,
            'DNA': dna,
            'DNA_params': dna_params,
            'RNA': rna,
            'RNA_params': rna_params,
            'PRT': prt,
            'PRT_params': prt_params,
        }
    )
tudf = pd.DataFrame(tu)

ntu = network.transcription_units
assert ntu[0].slots[1].is_resolved == True

# transcription units are never grouped
dna_df = pd.DataFrame({'tu_id': [[x] for x in tudf.name], 'type': 'DNA'})
rna_noparams_df = pd.DataFrame(  # we group RNA with same content if they don't have a parameter
    {
        'tu_id': list(tudf[tudf['RNA_params'] == ()].groupby(by='RNA').agg(list).name),
        'type': 'RNA',
    }
)
rna_params_df = pd.DataFrame(  # no grouping even if RNA content was identical...
    {
        'tu_id': list(tudf[tudf['RNA_params'] != ()].name),
        'type': 'RNA',
    }
)
prt_noparams_df = pd.DataFrame(  # we group PRT with same content if they don't have a parameter
    {
        'tu_id': list(tudf[tudf['PRT_params'] == ()].groupby(by='RNA').agg(list).name),
        'type': 'PRT',
    }
)
prt_params_df = pd.DataFrame(  # no grouping even if content was identical...
    {
        'tu_id': list(tudf[tudf['PRT_params'] != ()].name),
        'type': 'PRT',
    }
)
tudf.set_index('name', inplace=True)


# Then concatenate them:
cdg = pd.concat(
    [dna_df, rna_params_df, rna_noparams_df, prt_params_df, prt_noparams_df]
).reset_index(drop=True)
cdg['predecessor'] = None
cdg['successor'] = None

# there's probably a better, faster, more pandas way of doing this...
for i, r in cdg[cdg.type == 'DNA'].iterrows():
    cdg.loc[i, 'successor'] = []
    for ii, rr in cdg[cdg.type == 'RNA'].iterrows():
        for tuid in r.tu_id:
            if tuid in rr.tu_id:
                cdg.loc[i, 'successor'].append(ii)
for i, r in cdg[cdg.type == 'RNA'].iterrows():
    cdg.loc[i, 'successor'] = []
    for ii, rr in cdg[cdg.type == 'PRT'].iterrows():
        for tuid in r.tu_id:
            if tuid in rr.tu_id:
                cdg.loc[i, 'successor'].append(ii)

cdg['predecessor'] = [list() for _ in range(len(cdg))]
for i, r in cdg.iterrows():
    if r.successor is not None:
        for s in r.successor:
            cdg.loc[s]['predecessor'] += [i]
cdg.loc[~cdg.predecessor.astype(bool), 'predecessor'] = None

# We explicitly describe the part content of each node:
cdg['content'] = cdg.apply(lambda x: tudf.loc[x.tu_id[0]][x.type], axis=1)
cdg['content_type'] = cdg.apply(lambda x: tuple([lib.parts.loc[p][0] for p in x.content]), axis=1)


# And finally add information about the output of the whole graph:
# by default outputs are all parts whose category is fluo_marker
manual_outputs = []
outputs = manual_outputs + lib.parts[lib.parts['category'] == 'fluo_marker'].index.tolist()


def containsOutput(l, outputs):
    for o in outputs:
        if o in l:
            return True
    return False


cdg['is_output'] = False
cdg.loc[cdg.type == 'PRT', 'is_output'] = cdg.loc[cdg.type == 'PRT'].tu_id.apply(
    lambda x: containsOutput(tudf.loc[x].PRT.tolist()[0], outputs)
)
cdg['is_input'] = None
# for k, v in inputDict.items():
# cdg.loc[k, 'is_input'] = v

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                       --     compute graph     --
# ···············································································


class GraphComputeNode:
    def __init__(self, id, type, cdg_input, cdg_output):
        self.id = id
        self.type = type
        self.cdg_input = cdg_input
        self.cdg_output = cdg_output if cdg_output is not None else -1
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
            "cdg_input": self.cdg_input,
            "cdg_output": self.cdg_output,
            "input_from": self.input_from,
            "output_to": self.output_to,
        }

    def __str__(self):
        return str(self.toDict())

    def __repr__(self):
        return str(self.toDict())


uidGen = bu.uniqueIdGenerator()


def isOutputOf(cdg_input_node, compute_nodes):
    res = [other for other in compute_nodes if cdg_input_node == other.cdg_output]
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

# we start building the compute graph from the output:
output_gene_nodes = cdg[cdg.is_output]

onode = GraphComputeNode(uidGen(), 'output', [], None)
for i, r in output_gene_nodes.iterrows():
    onode.cdg_input += [i]
newnodes.append(onode)

# then we add the sequestron nodes with an associated list of their cdg input nodes
for _, r in lib.seqs.iterrows():
    nlvl = cdg[cdg.type == r.negative_level]
    nparts = nlvl[nlvl.content.apply(lambda x: r.negative_part in x)]
    plvl = cdg[cdg.type == r.positive_level]
    pparts = plvl[plvl.content.apply(lambda x: r.positive_part in x)]
    olvl = cdg[cdg.type == r.output_level]
    oparts = olvl[olvl.content.apply(lambda x: bu.isSubset(r.output_part, x))]
    if len(nparts) > 0 and len(pparts) > 0 and len(oparts.index) > 0:
        assert len(pparts) == 1
        assert len(nparts) == 1
        try:
            cnode = GraphComputeNode(
                uidGen(),
                f'sequestron_{r.type}',
                [int(nparts.index[0]), int(pparts.index[0])],
                int(oparts.index[0]),
            )
            newnodes.append(cnode)
        except:
            print('nparts', nparts)
            print('pparts', nparts)
            print('olvl', olvl)
            print('oparts', nparts)
            print('outlevel', r.output_level)
            print('outpart', r.output_part)

# then for each input node, we need to go back up to the original transcription unit using translation
# and transcription nodes, making sure to connect it to relevant sequestron nodes along the way
cg = []

# at first, each TU has a corresponding source, but we later merge sources (for TUs that are on a same plasmid)
while newnodes:
    n = newnodes.pop()
    if n.type != 'source':
        # for every gene input of this compute node
        for i, n_inp in enumerate(n.cdg_input):
            others = isOutputOf(n_inp, cg + newnodes)
            for other in others:
                # set_list_item(n.input_from,i,other.id)
                n.input_from += [other.id]
                other.output_to += [(n.id, i)]
            if not others:
                gn = cdg.loc[n_inp]  # input gene
                nid = uidGen()
                ntype = {'PRT': 'translation', 'RNA': 'transcription', 'DNA': 'source'}[gn.type]
                newn = GraphComputeNode(nid, ntype, gn.predecessor, int(n_inp))
                newn.input_from = []
                newn.output_to = [(n.id, i)]
                newnodes.append(newn)
                n.input_from += [int(nid)]
    cg += [n]

removeShortcuts(cg, 0)  # turns the graph back into a tree
cdf = pd.DataFrame([n.toDict() for n in cg]).set_index('id').sort_index()

# merge sources (i.e aggregate TUs into plasmids)
c = network.dbconnection.cursor()
c.execute(
    """SELECT tis.source,tis.TU,position  FROM TU_in_source tis, source_in_aggregation sia, aggregations a 
   WHERE tis.source = sia.source AND sia.aggregation = a.id AND a.tube = ?""",
    (network.name,),
)
tu_in_sources = pd.DataFrame([t for t in c.fetchall()]).sort_values(2)
sources_tuids = cdg.loc[cdf[cdf.type == 'source'].cdg_output].tu_id.apply(lambda x: x[0])
tmpdf = pd.DataFrame(
    {'compute_id': cdf[cdf.type == 'source'].index, 'tuid': sources_tuids}
).set_index('compute_id')

cdf['source_id'] = None
cdf['extra'] = None

sources = {}  # plasmid name -> list of compute nodes ids
for i, r in tu_in_sources.groupby(0).agg(list).iterrows():
    # order matters.
    group = []
    for t in r[1]:
        group.append(tmpdf[tmpdf.tuid == t].index[0])
    sources[i] = group

print('sources', sources)

for k, v in sources.items():
    nid = uidGen()
    newsource = GraphComputeNode(nid, 'source', None, [cdf.loc[vv].cdg_output for vv in v])
    newsource.output_to = [cdf.loc[vv].output_to[0] for vv in v]
    print('newsource', newsource)
    # and update input_from of these nodes too 
    cdf.loc[[o[0] for o in newsource.output_to], 'input_from'] = [nid]*len(newsource.output_to)
    cdf = cdf.append(pd.DataFrame([newsource.toDict()]).set_index('id')).drop(v)
    cdf.loc[nid, 'source_id'] = k

# turn every input_from that's a single int into a list
cdf.input_from = cdf.input_from.apply(lambda x: [x] if isinstance(x, int) else x)

c.execute(
    """SELECT a.id,  a.qtty, a.tube, sia.source, sia.ratio FROM aggregations a, source_in_aggregation sia
    WHERE a.id = sia.aggregation AND a.tube = ?""",
    (network.name,),
)

# adding the aggregation nodes
aggregations = (
    pd.DataFrame([t for t in c.fetchall()], columns=['id', 'qtty', 'tube', 'source', 'ratio'])
    .groupby('id')
    .agg(list)
)

for i, r in aggregations.iterrows():
    if len(r.source) > 1:
        nid = uidGen()
        newaggregation = GraphComputeNode(nid, 'aggregation', None, r.source)
        # find the compute node id through the source_id column
        newaggregation.output_to = [(cdf[cdf.source_id == s].index[0], 0) for s in r.source]
        # add the input_from to the cooresponding sources
        for s in r.source:
            cdf.loc[cdf.source_id == s, 'input_from'] = [[nid]]
        # For aggregations, we will store a dictionnary with the name of the aggregation and the ratio of each source
        tmp = pd.DataFrame([newaggregation.toDict()]).set_index('id')
        tmp['extra'] = [{'id': i, 'qtty': np.sum(r.qtty), 'ratios': r.ratio}]
        cdf = cdf.append(tmp)
    else:
        # no need for an aggregation node if there is only one source
        cdf.loc[cdf.source_id == r.source[0], 'extra'] = [{'qtty': np.sum(r.qtty)}]

cdf.source_id = cdf.source_id.apply(lambda x: str(x) if not pd.isnull(x) else None)
cdf

# now we add numeric nodes (can be constant or inputs). They will mostly be used for copy numbers.
# Let's start by adding 1 constant per source or aggregation that's "at the top", i.e its input_from is empty.
topnodes = cdf[cdf.input_from.apply(len) == 0]
for i, r in topnodes.iterrows():
    nid = uidGen()
    newnode = GraphComputeNode(nid, 'numeric', None, 1)
    newnode.output_to = [(i, 0)]
    tmp = pd.DataFrame([newnode.toDict()]).set_index('id')
    assert 'qtty' in r.extra
    tmp['extra'] = [
        {'is_input': False, 'is_constant': True, 'role': 'copy_number', 'value': r.extra['qtty']}
    ]
    cdf = cdf.append(tmp)
    # don't forget to add the new node as input_from to the top node
    cdf.loc[i, 'input_from'] = [[nid]]

cdf = cdf.replace({np.nan: None})

# reconstruct all input_froms from the output_to:
# output_to is a list of (target_node_id, input_position) tuples
# we want to turn input_from into a list of (source_node_id, output_position) tuples
for i, r in cdf.iterrows():
    for p, o in enumerate(r.output_to):
        cdf.loc[o[0], 'input_from'][o[1]] = (i, p)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

# ut.grnGraph(cdg)
# remove any nan from cdf and replace by None:
ut.drawComputeGraph(cdf, cdg=cdg)


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{               --     template + generator for TUs     --
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
    for s in l1.slots:
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
    for s in l1.slots:
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


# map L1 to parameter values for each node

ERN_template = bc.TranscriptionUnit(
    [
        bc.Slot(any_promoter),
        bc.Slot(any_uorf),
        bc.Slot(random_ERN_rec),
        bc.Slot(random_ERN),
        bc.Part('NeonGreen'),
    ]
)
ERN_template.resolve_all_slots(lib, random_seed=3)


ERN_template

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

cdf

# TODO:
# [ ] Complete the compute graph construction from the XP file
# -> [ ] Switch to a dependency-based approach
# -> [ ] Replace Bias and Input nodes by a Numeric one
# -> [ ] Add a Source node. Basically a no-op splitter?
# -> [ ] Add an Aggregation node.
# -> [ ] Add noise distribution to all nodes?
# Maybe / TBD depending on how fixed vs trainable parameters are handled:
# -> [ ] Pass quantizers and param_accessor functors to all node creator.
# -> [ ] Handle inputs at the param level: we can just set the inputs to be fixed parameters in the param dictionnary.
#        Example: we know that numeric node #012 is an input. Therefore we can just:
#                 - set params['local'][12]['value'] to be a non-trainable param
#                 - set the value of the input just before calling compute.
#        Problem: might be slow? instead of being able to use the same dict for each computations, we need copies??


# [ ] Train
# -> [ ] Add a way to specify params that are fixed vs trainable before traning,
#        and aggregate them in a transparent dictionnary that will be passed to the compute graph
#        Probably should just split into 2 dictionnaries given to the train method (1st is differentiated against, 2nd is fixed).
#        Then do a merge of the 2 before passing them to the CG. Q: will Jax be ok to compile that?
# -> [ ] Invertible path addition to the compute graph:
#    -> [ ] ensure that each numeric node is tied to an invertible path.
#    -> [ ] add the inverse path to the compute graph (fluo -> invpath -> numeric -> fwdpath -> fluo)
# -> [ ] Parse data file (start with Georgss) and load into dataframe
# -> [ ] write training loop. Loss = L2 (fluo_out_from_full_gaph, fluo_out_measured)


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{           --     Construction of the compute function     --
# ···············································································


# We have a directed acyclic graph of nodes, and a dictionnary of parameters.
# We used to be able to treat the whole compute graph as a simple tree. However adding split
# and merge nodes makes it a DAG. We need to be able to compute the graph in topological order.
# Let's treat the whole thing as a dependency graph and generate the ordered list of nodes batches to compute.

cdf

def getBatchSequence(cdf):
    """Return a list of lists of nodes, where each node of a sublist can be computed independently of the others,
    but each sublist must be computed in order."""
    visited = set()
    batches = []
    while len(visited) < len(cdf):
        independent = [
            i
            for i, row in cdf.iterrows()
            if (not row['input_from'] or all([x[0] in visited for x in row['input_from']]))
            and i not in visited
        ]
        if not independent:
            raise ValueError('Graph is not acyclic')
        visited.update(independent)
        batches.append(independent)
    return batches


batches = getBatchSequence(cdf)

flat_batches = [item for sublist in batches for item in sublist]

def getParam(params, name, init, shared=False, nodeid=None):
    nid = nodeid if not shared else 'shared'
    if nid not in params:
        params[nid] = {}
    if name not in params[nid]:
        params[nid][name] = init()
    return params[nid][name]

def constant(params, value):
    return [value]

def add(getParam, *values, **kwargs):
    cte = getParam('cte', shared=True, init=lambda: 42.0)
    return jnp.array([jnp.sum(jnp.array(values)), cte])

getfn = defaultdict(lambda: add, {'add': add})

def generate():

    def comp(params):
        results = {}
        for nid in flat_batches:
            getp = partial(getParam, params, nodeid=nid)
            node = cdf.loc[nid]
            if node.type == 'numeric':
                results[nid] = constant(params, 1.0)
            else:
                results[nid] = getfn[node.type](getp, *[results[n][p] for n,p in node.input_from])
        return results[flat_batches[-1]]

    def init():
        params = {}
        comp(params)
        return params

    return init, comp

init, model = generate()

params = init()
print(params)
model(params)

jit(model)(params)

ut.print_jaxpr(model,params)
ut.print_xla(model,params)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
