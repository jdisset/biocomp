## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     import and init     --
# ···············································································
# %load_ext autoreload
# %autoreload 2

import streamlit as st

st.set_page_config(layout='wide')

import pandas as pd
import numpy as np
import sqlite3
import os

import scriptutils as ut
import biocomp.utils as bu
from functools import partial
import biocomp as bc
import json
from rich import print

l = ut.load("all_sheets.pickle")
lib = bc.PartsLibrary(l.parts, l.L0s, l.L1s, l.L2s, l.categories, l.sequestrons, l.sequestron_types)
series_obj = json.load(open("example_xpfile.json"))

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

network = series['L2_all']

def getDna(tu):
    return [s.part for s in tu.slots if s.is_resolved and not isinstance(s.part, list)]

def getRna(tu, lib):
    dna = getDna(tu)
    d = lib.pc.loc[dna]
    return tuple(d[d.transcripted == 1].index)

def getPrt(tu, lib):
    dna = getDna(tu)
    d = lib.pc.loc[dna]
    return tuple(d[d.translated == 1].index)


tu = [
    {
        'name': tuid,
        'DNA': getDna(t),
        'RNA': getRna(t, lib),
        'PRT': getPrt(t, lib),
    }
    for tuid, t in zip(network.tuids, network.transcription_units)
]
tudf = pd.DataFrame(tu)

newtudf = tudf.copy()
newtudf['type'] = newtudf.DNA.apply(lambda x: "DNA")
newtudf['content'] = newtudf.DNA


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

print(tudf)

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
    print(oparts)
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

while newnodes:
    n = newnodes.pop()
    if n.type != 'bias':
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

# at this point all transcription units are independent.
# Let's merge the L2s


cdf = pd.DataFrame([n.toDict() for n in cg]).set_index('id').sort_index()

# add input ids
cdf['is_input'] = None
for index, row in cdf.iterrows():
    if row['type'] == 'bias':
        input_id = cdg.at[row['cdg_output'], 'is_input']
        if input_id is not None:
            cdf.at[index, 'type'] = 'input'
            cdf.at[index, 'is_input'] = int(input_id)
