from . import utils as ut
from .library import PartsLibrary as PartsLibrary
import pandas as pd

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{             --     Build the Compute Graph     --
# ···············································································

# Here we generate a dataframe that contains all the available functions for our current library


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


def buildComputeGraph(lib, cdg):

    uidGen = ut.uniqueIdGenerator()

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
    output_gene_nodes = cdg[cdg.is_output]
    onode = GraphComputeNode(uidGen(), 'output', [], None)
    for i, r in output_gene_nodes.iterrows():
        onode.cdg_input += [i]
    newnodes.append(onode)

    # first we add the sequestron nodes with a list of their cdg input nodes
    for _, r in lib.seqs.iterrows():
        nlvl = cdg[cdg.type == r.negative_level]
        nparts = nlvl[nlvl.content.apply(lambda x: r.negative_part in x)]
        plvl = cdg[cdg.type == r.positive_level]
        pparts = plvl[plvl.content.apply(lambda x: r.positive_part in x)]
        olvl = cdg[cdg.type == r.output_level]
        oparts = olvl[olvl.content.apply(lambda x: ut.isSubset(r.output_part, x))]
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

    # then for each input node, we need to go back up to the original DNA using translation
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
            input_id = cdg.at[row['cdg_output'], 'is_input']
            if input_id is not None:
                cdf.at[index, 'type'] = 'input'
                cdf.at[index, 'is_input'] = int(input_id)

    return cdf


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

