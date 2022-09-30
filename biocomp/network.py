'from' .library import PartsLibrary as PartsLibrary
import jax
import numpy as np
import pandas as pd
from . import utils as ut
import sqlite3

part_type_to_parameter_name = {'promoter': 'tc_rate', 'uORF': 'tl_rate'}
parameter_to_default_part = {'tl_rate': 'empty_tc'} 


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                       --     base classes     --
# ···············································································
class Slot:
    def __init__(self, f):
        self.resolve_function = f
        self.part = None  # list means multiple parts that should map to a single parameter. Otherwise single string
        self.maps_to_parameter = None
        self.is_resolved = False

    def resolve(self, lib, *args, **kwargs):
        if not self.is_resolved:
            self.part = self.resolve_function(lib, *args, **kwargs)
            if self.part == [] or self.part == [None]:
                self.part = None
            if isinstance(self.part, list):
                mapped = [self.__mapped_parameter(lib, p) for p in self.part if p is not None]
                if len(mapped) != 1:
                    raise ValueError(f'{self.part} maps to {len(mapped)} parameters ({mapped})')
                self.maps_to_parameter = mapped[0]
            else:
                self.maps_to_parameter = self.__mapped_parameter(lib, self.part)
            if self.maps_to_parameter is not None and not isinstance(self.part, list):
                self.part = [self.part]
            self.is_resolved = True

    def __mapped_parameter(self, lib, part_name):

        if part_name is not None:
            if part_name in lib.pc.index:
                category = lib.pc.loc[part_name, 'category']
                if category in part_type_to_parameter_name:
                    return part_type_to_parameter_name[category]
            else:
                raise ValueError(f'Unknown part: {part_name}')
        return None

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


# util for a slot that resolves to a single part
def Part(name):
    return Slot(lambda *_, **__: name)


# transcription unit: 1 per L1, multiple per L2
class TranscriptionUnit:
    def __init__(self, slots):
        self.name = ''
        self.slots = slots
        self.params = {}
        self.is_resolved = False

    def resolve_all_slots(self, lib, random_seed=1, random_order=True):
        rdm = jax.random.PRNGKey(random_seed)
        allrdm = jax.random.split(rdm, len(self.slots))
        order = list(range(len(self.slots)))
        if random_order:
            order = jax.random.permutation(rdm, len(self.slots))
        for i, r in zip(order, allrdm):
            if not self.slots[i].is_resolved:
                self.slots[i].resolve(lib, l1=self, rdm_key=r)

        self.__get_parameters()

        assert all(s.is_resolved for s in self.slots)

    def __get_parameters(self):
        for s in self.slots:
            assert s.is_resolved
            if s.maps_to_parameter is not None:
                assert s.maps_to_parameter not in self.params
                self.params[s.maps_to_parameter] = s.part
        # then for each param that is not in the slots, add it with default value
        for _, p in part_type_to_parameter_name.items():
            if p not in self.params:
                self.params[p] = parameter_to_default_part[p]

    def __repr__(self):
        return f'L1({self.slots})'


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


def transcription_unit_from_L1(l1id, lib):
    l0_cols = ["insulator", "promoter", "5'UTR", "gene", "3'UTR", "terminator"]
    L0s = lib.L1s.loc[l1id][l0_cols].tolist()
    part_cols = [f'part_{i}' for i in range(1, 7)]
    parts = []
    for l in L0s:
        parts += [p for p in lib.L0s.loc[l][part_cols].tolist() if p]
    tu = TranscriptionUnit([Part(p) for p in parts])
    tu.resolve_all_slots(lib)
    return tu


# main class: a network of interacting transcription units
class Network:
    def __init__(self, lib, recipe_name, recipe_db, custom_outputs=None, build=True):
        self.lib = lib
        self.name = recipe_name
        self.db = recipe_db
        self.db.commit()
        self.custom_outputs = custom_outputs
        self.central_dogma_graph = None
        self.compute_graph = None
        if build:
            self.build()

    def build(self):
        c = self.db.cursor()
        # first let's check that there is a recipe with this name
        c.execute('SELECT * FROM recipes WHERE name=?', (self.name,))
        assert c.fetchone() is not None, f'No recipe named {self.name} in database'
        c.execute(
            """SELECT TU FROM TU_in_source tis, source_in_aggregation sia, aggregations a
           WHERE tis.source = sia.source AND sia.aggregation = a.id AND a.recipe = ?""",
            (self.name,),
        )
        self.transcription_units = {
            tu[0]: transcription_unit_from_L1(tu[0], self.lib) for tu in c.fetchall()
        }
        assert len(self.transcription_units) > 0, f'No transcription units in recipe {self.name}'
        self.__build_central_dogma_graph(self.custom_outputs)
        self.__build_compute_graph()

    def is_built(self):
        return self.compute_graph is not None and self.central_dogma_graph is not None

    ## ───────────────────────────────────── ▼ ─────────────────────────────────────
    # {{{                           --     utils     --
    # ···············································································
    def __getDna(self, tu):
        content = []
        for s in tu.slots:
            assert s.is_resolved
            if s.maps_to_parameter is None:
                content.append(s.part)
        return content, tu.params

    def __getDownstream(self, tu, transform):
        dna_content, dna_params = self.__getDna(tu)
        d = self.lib.pc.loc[dna_content]
        content = tuple(d[d[transform] == 1].index)
        rna_params = {}
        for param_name, parts in dna_params.items():
            p = self.lib.pc.loc[parts]
            if p[transform].sum() > 0:
                assert p[transform].sum() == len(p), f'Part {parts} is not {transform}. p: \n{p}, sum: {p[transform].sum()}, len: {len(p)}'
                rna_params[param_name] = tuple(p.index)
        return content, rna_params

    def __getRna(self, tu):
        return self.__getDownstream(tu, transform='transcripted')

    def __getPrt(self, tu):
        return self.__getDownstream(tu, transform='translated')

    def __isOutputOf(self, cdg_input_node, compute_nodes):
        res = [other for other in compute_nodes if cdg_input_node == other.cdg_output]
        return res

    def __getNode(self, nodes, id):
        for node in nodes:
            if node.id == id:
                return node
        raise Exception("Node not found")

    # removeShortcuts removes indirect links in the Compute graph,
    # turning it from a directed acyclic graph to a tree.
    def __removeShortcuts(self, nodes, root_id):
        labels = {}
        for node in nodes:
            labels[node.id] = 1
        S = set()
        S.add(root_id)
        while len(S) > 0:
            N = self.__getNode(nodes, S.pop())
            w = labels[N.id] + 1
            for d in N.input_from:
                if labels[d] < w:
                    labels[d] = w
                    S.add(d)
        # remove all edges which connect nodes whose labels differ by more than 1.
        for node in nodes:
            for d in node.input_from:
                if labels[node.id] + 1 < labels[d]:
                    self.__getNode(nodes, d).removeOutput(node)

    class GraphComputeNode:
        # a simple convenience one-off class to store the information about a node
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

    #                                                                            }}}
    ## ─────────────────────────────────────────────────────────────────────────────

    ## ───────────────────────────────────── ▼ ─────────────────────────────────────
    # {{{                 --     build central dogma graph     --
    # ···············································································

    def __build_central_dogma_graph(self, custom_outputs=None):
        tu = []
        for tuid, t in self.transcription_units.items():
            dna, dna_params = self.__getDna(t)
            rna, rna_params = self.__getRna(t)
            prt, prt_params = self.__getPrt(t)
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
        assert len(tu) > 0
        tudf = pd.DataFrame(tu)

        # transcription units are never grouped
        dna_df = pd.DataFrame({'tu_id': [[x] for x in tudf['name']], 'type': 'DNA'})
        rna_noparams_df = (
            pd.DataFrame(  # we group RNA with same content if they don't have a parameter
                {
                    'tu_id': list(
                        tudf[tudf['RNA_params'].map(len) == 0].groupby(by='RNA').agg(list).name
                    ),
                    'type': 'RNA',
                }
            )
        )
        rna_params_df = pd.DataFrame(  # no grouping even if RNA content was identical...
            {
                'tu_id': list(tudf[tudf['RNA_params'].map(len) > 0].name),
                'type': 'RNA',
            }
        )
        prt_noparams_df = (
            pd.DataFrame(  # we group PRT with same content if they don't have a parameter
                {
                    'tu_id': list(
                        tudf[tudf['PRT_params'].map(len) == 0].groupby(by='RNA').agg(list).name
                    ),
                    'type': 'PRT',
                }
            )
        )
        prt_params_df = pd.DataFrame(  # no grouping even if content was identical...
            {
                'tu_id': list(tudf[tudf['PRT_params'].map(len) > 0].name),
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
        cdg['content_type'] = cdg.apply(
            lambda x: tuple([self.lib.parts.loc[p][0] for p in x.content]), axis=1
        )
        # and add the available paras with their possible parts
        cdg['params'] = cdg.apply(lambda x: tudf.loc[x.tu_id[0]][x.type + '_params'], axis=1)

        # And finally add information about the output of the whole graph:
        # by default outputs are all parts whose category is fluo_marker
        outputs = (custom_outputs if custom_outputs is not None else []) + self.lib.parts[
            self.lib.parts['category'] == 'fluo_marker'
        ].index.tolist()

        containsOutput = lambda l, outputs: any([o in l for o in outputs])
        cdg['is_output'] = False
        cdg.loc[cdg.type == 'PRT', 'is_output'] = cdg.loc[cdg.type == 'PRT'].tu_id.apply(
            lambda x: containsOutput(tudf.loc[x].PRT.tolist()[0], outputs)
        )
        cdg['is_input'] = None
        self.central_dogma_graph = cdg

    #                                                                            }}}
    ## ─────────────────────────────────────────────────────────────────────────────

    ## ───────────────────────────────────── ▼ ─────────────────────────────────────
    # {{{      --     build compute graph (from central dogma graph)     --
    # ···············································································

    # a lot of the code below is super verbose and could be simplified and optimized
    # but it's definitely not a priority right now

    def __mergeSources(self, cdf, uidGen):
        assert self.central_dogma_graph is not None
        # merge TUs that are from a same source into a single node
        c = self.db.cursor()
        c.execute(
            """SELECT tis.source,tis.TU,position FROM TU_in_source tis, source_in_aggregation sia, aggregations a 
           WHERE tis.source = sia.source AND sia.aggregation = a.id AND a.recipe = ?""",
            (self.name,),
        )
        tu_in_sources = pd.DataFrame([t for t in c.fetchall()]).sort_values(2)
        sources_tuids = self.central_dogma_graph.loc[
            cdf[cdf.type == 'source'].cdg_output
        ].tu_id.apply(lambda x: x[0])
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

        for k, v in sources.items():
            nid = uidGen()
            newsource = self.GraphComputeNode(
                nid, 'source', None, [cdf.loc[vv].cdg_output for vv in v]
            )
            newsource.output_to = [cdf.loc[vv].output_to[0] for vv in v]
            # and update input_from of these nodes too
            cdf.loc[[o[0] for o in newsource.output_to], 'input_from'] = [nid] * len(
                newsource.output_to
            )
            cdf = cdf.append(pd.DataFrame([newsource.toDict()]).set_index('id')).drop(v)
            cdf.loc[nid, 'source_id'] = k

        # turn every input_from that's a single int into a list
        cdf.input_from = cdf.input_from.apply(lambda x: [x] if isinstance(x, int) else x)
        return cdf

    def __addAggregations(self, cdf, uidGen):
        c = self.db.cursor()
        c.execute(
            """SELECT a.id, a.recipe, sia.source, sia.ratio FROM aggregations a, source_in_aggregation sia
            WHERE a.id = sia.aggregation AND a.recipe = ?""",
            (self.name,),
        )

        # adding the aggregation nodes
        aggregations = (
            pd.DataFrame([t for t in c.fetchall()], columns=['id', 'recipe', 'source', 'ratio'])
            .groupby('id')
            .agg(list)
        )

        for i, r in aggregations.iterrows():
            if len(r.source) > 1:
                nid = uidGen()
                newaggregation = self.GraphComputeNode(nid, 'aggregation', None, r.source)
                # find the compute node id through the source_id column
                newaggregation.output_to = [(cdf[cdf.source_id == s].index[0], 0) for s in r.source]
                # add the input_from to the cooresponding sources
                for s in r.source:
                    cdf.loc[cdf.source_id == s, 'input_from'] = [[nid]]
                # For aggregations, we will store a dictionnary with the name of the aggregation and the ratio of each source
                tmp = pd.DataFrame([newaggregation.toDict()]).set_index('id')
                tmp['extra'] = [{'id': i, 'qtty': np.sum(r.ratio), 'ratios': r.ratio}]
                cdf = cdf.append(tmp)
            else:
                # no need for an aggregation node if there is only one source
                cdf.loc[cdf.source_id == r.source[0], 'extra'] = [{'qtty': np.sum(r.ratio)}]

        return cdf

    def __buildRawGraph(self, uidGen):
        cdg = self.central_dogma_graph
        assert cdg is not None
        assert isinstance(cdg, pd.DataFrame)
        newnodes = []

        # we start building the compute graph from the output:
        output_gene_nodes = cdg[cdg.is_output]

        onode = self.GraphComputeNode(uidGen(), 'output', [], None)
        for i, r in output_gene_nodes.iterrows():
            onode.cdg_input += [i]
        newnodes.append(onode)

        # then we add the sequestron nodes with an associated list of their cdg input nodes
        for _, r in self.lib.seqs.iterrows():
            nlvl = cdg[cdg.type == r.negative_level]
            nparts = nlvl[nlvl.content.apply(lambda x: r.negative_part in x)]
            plvl = cdg[cdg.type == r.positive_level]
            pparts = plvl[plvl.content.apply(lambda x: r.positive_part in x)]
            olvl = cdg[cdg.type == r.output_level]
            oparts = olvl[olvl.content.apply(lambda x: ut.isSubset(r.output_part, x))]
            if len(nparts) > 0 and len(pparts) > 0 and len(oparts.index) > 0:
                assert len(pparts) == 1
                assert len(nparts) == 1
                try:
                    cnode = self.GraphComputeNode(
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
                    others = self.__isOutputOf(n_inp, cg + newnodes)
                    for other in others:
                        # set_list_item(n.input_from,i,other.id)
                        n.input_from += [other.id]
                        other.output_to += [(n.id, i)]
                    if not others:
                        gn = cdg.loc[n_inp]  # input gene
                        nid = uidGen()
                        ntype = {'PRT': 'translation', 'RNA': 'transcription', 'DNA': 'source'}[
                            gn.type
                        ]
                        newn = self.GraphComputeNode(nid, ntype, gn.predecessor, int(n_inp))
                        newn.input_from = []
                        newn.output_to = [(n.id, i)]
                        newnodes.append(newn)
                        n.input_from += [int(nid)]
            cg += [n]
        return cg

    def __addNumericNodes(self, cdf, uidGen):
        # we add 1 numeric node per source or aggregation that's "at the top",
        # i.e its input_from is empty.
        topnodes = cdf[cdf.input_from.apply(len) == 0]
        for i, r in topnodes.iterrows():
            nid = uidGen()
            newnode = self.GraphComputeNode(nid, 'numeric', None, 1)
            newnode.output_to = [(i, 0)]
            tmp = pd.DataFrame([newnode.toDict()]).set_index('id')
            extra = {
                'role': 'copy_number',
            }
            if 'qtty' in r.extra:
                extra['qtty'] = r.extra['qtty']
            tmp['extra'] = [extra]
            cdf = cdf.append(tmp)
            # don't forget to add the new node as input_from to the top node
            cdf.loc[i, 'input_from'] = [[nid]]
        return cdf

    def __build_compute_graph(self):
        assert self.central_dogma_graph is not None, 'central dogma graph not built yet'

        uidGen = ut.uniqueIdGenerator()

        # build the graph of interacting nodes, without any optimization,
        # basically the dual of the central dogma graph:
        cg = self.__buildRawGraph(uidGen)
        # remove shortcuts and cycles:
        self.__removeShortcuts(cg, 0)

        # convert to dataframe
        cdf = pd.DataFrame([n.toDict() for n in cg]).set_index('id').sort_index()

        cdf = self.__mergeSources(cdf, uidGen)  # merge TUs with same source
        cdf = self.__addAggregations(cdf, uidGen)  # add aggregation nodes
        cdf = self.__addNumericNodes(cdf, uidGen)  # now add numeric nodes (constant or inuts)

        assert cdf is not None

        # quick clean up / sanity check
        cdf.source_id = cdf.source_id.apply(lambda x: str(x) if not pd.isnull(x) else None)
        cdf = cdf.replace({np.nan: None})

        # reconstruct all input_froms from the output_to:
        for i, r in cdf.iterrows():
            for p, o in enumerate(r.output_to):
                cdf.loc[o[0], 'input_from'][o[1]] = (i, p)

        self.compute_graph = cdf

    #                                                                            }}}
    ## ─────────────────────────────────────────────────────────────────────────────
