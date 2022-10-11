from .library import PartsLibrary as PartsLibrary
import jax
import numpy as np
import pandas as pd
from . import utils as ut
from .compute import INVERSE_NODES_DICT as INVERSE_NODES_DICT
from typing import Callable, List, Dict, Tuple, Iterable
import copy


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


class GraphComputeNode:
    # a simple convenience one-off class to store the information about a node
    def __init__(self, id, type, cdg_input, cdg_output):
        self.id = id
        self.type = type
        self.cdg_input = cdg_input
        self.cdg_output = cdg_output if cdg_output is not None else -1
        self.input_from = []
        self.output_to = []
        self.extra = {}

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
            "extra": self.extra,
        }

    def __str__(self):
        return str(self.toDict())

    def __repr__(self):
        return str(self.toDict())


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
                self.params[p] = [parameter_to_default_part[p]]

    def __repr__(self):
        return f'L1({self.slots})'


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


class hashabledict(dict):
    def __hash__(self):
        # general idea is return hash(tuple(sorted(self.items())))
        # but we need to turn the values (list in our case) into tuples
        return hash(tuple(sorted((k, tuple(v)) for k, v in self.items())))


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
        return (
            self.compute_graph is not None
            and self.central_dogma_graph is not None
            and self.transcription_units is not None
        )

    ## ───────────────────────────────────── ▼ ─────────────────────────────────────
    # {{{                           --     utils     --
    # ···············································································
    def __getDna(self, tu: TranscriptionUnit) -> Tuple[List[str], Dict[str, List[str]]]:
        content = []
        for s in tu.slots:
            assert s.is_resolved
            if s.maps_to_parameter is None:
                content.append(s.part)
        return content, tu.params

    def __getDownstream(self, tu: TranscriptionUnit, transform: str):
        dna_content, dna_params = self.__getDna(tu)
        d = self.lib.pc.loc[dna_content]
        content = tuple(d[d[transform] == 1].index)
        params = {}
        for param_name, parts in dna_params.items():
            p = self.lib.pc.loc[parts]
            if p[transform].sum() > 0:
                assert p[transform].sum() == len(p)
                params[param_name] = list(p.index)
        return content, params

    def __getRna(self, tu: TranscriptionUnit):
        return self.__getDownstream(tu, transform='transcripted')

    def __getPrt(self, tu: TranscriptionUnit):
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
                    'DNA_params_hashable': hashabledict(dna_params),
                    'RNA': rna,
                    'RNA_params': rna_params,
                    'RNA_params_hashable': hashabledict(rna_params),
                    'PRT': prt,
                    'PRT_params': prt_params,
                    'PRT_params_hashable': hashabledict(prt_params),
                }
            )
        assert len(tu) > 0
        tudf = pd.DataFrame(tu)

        # transcription units are never grouped
        dna_df = pd.DataFrame({'tu_id': [[x] for x in tudf['name']], 'type': 'DNA'})

        def only_one_value_per_param(params: Dict[str, List[str]]) -> bool:
            for _, parts in params.items():
                if len(parts) > 1:
                    return False
            return True

        rna_tuids_noparams = list(
            tudf[tudf['RNA_params'].map(len) == 0].groupby(by='RNA').agg(list).name
        )

        rna_tuids_oneparamvalue = (
            tudf[tudf['RNA_params'].map(len) > 0]
            .groupby(by='RNA')
            .filter(lambda x: only_one_value_per_param(x['RNA_params']))
            .groupby(by=['RNA', 'RNA_params_hashable'])
            .agg(list)
        )
        rna_tuids_oneparamvalue = (
            [] if rna_tuids_oneparamvalue.empty else list(rna_tuids_oneparamvalue.name)
        )

        rna_tuids_manyparamvalues = list(tudf[tudf['RNA_params'].map(len) > 1].name)
        rna_tuids = rna_tuids_noparams + rna_tuids_oneparamvalue + rna_tuids_manyparamvalues
        rna_df = pd.DataFrame({'tu_id': rna_tuids, 'type': 'RNA'})

        prt_tuids_noparams = list(
            tudf[tudf['PRT_params'].map(len) == 0].groupby(by='PRT').agg(list).name
        )
        # we group PRT with same content and same parameters if they have a single parameters
        prt_tuids_oneparamvalue = (
            tudf[tudf['PRT_params'].map(len) > 0]
            .groupby(by='PRT')
            .filter(lambda x: only_one_value_per_param(x['PRT_params']))
            .groupby(by=['PRT', 'PRT_params_hashable'])
            .agg(list)
        )
        prt_tuids_oneparamvalue = (
            [] if prt_tuids_oneparamvalue.empty else list(prt_tuids_oneparamvalue.name)
        )
        prt_tuids_manyparamvalues = list(tudf[tudf['PRT_params'].map(len) > 1].name)
        prt_tuids = prt_tuids_noparams + prt_tuids_oneparamvalue + prt_tuids_manyparamvalues
        prt_df = pd.DataFrame({'tu_id': prt_tuids, 'type': 'PRT'})

        tudf.set_index('name', inplace=True)

        # Then concatenate them:
        cdg = pd.concat([dna_df, rna_df, prt_df], sort=False).reset_index(drop=True)
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
        try:
            cdg['content'] = cdg.apply(lambda x: tudf.loc[x.tu_id[0]][x.type], axis=1)
            cdg['content_type'] = cdg.apply(
                lambda x: tuple([self.lib.parts.loc[p][0] for p in x.content]), axis=1
            )
        except Exception as e:
            msg = f'Error while building central dogma graph. Error: {e}'
            msg += f'\ntudf: \n{tudf}'
            msg += f'\n\ncdg: \n{cdg}'
            raise Exception(msg)

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
        # merge TUs that are from a same source into a single source node (aka plasmid)
        c = self.db.cursor()
        c.execute(
            """SELECT tis.source,tis.TU,position FROM TU_in_source tis, source_in_aggregation sia, aggregations a
           WHERE tis.source = sia.source AND sia.aggregation = a.id AND a.recipe = ?""",
            (self.name,),
        )
        tu_in_sources = pd.DataFrame(
            [t for t in c.fetchall()], columns=['source', 'TU', 'position']
        )
        tu_in_sources.sort_values(by='position', inplace=True)

        sources_tuids = self.central_dogma_graph.loc[
            cdf[cdf.type == 'source'].cdg_output
        ].tu_id.apply(lambda x: x[0])

        tmpdf = pd.DataFrame(
            {'compute_id': cdf[cdf.type == 'source'].index, 'tuid': sources_tuids}
        ).set_index('compute_id')

        cdf['source_id'] = None
        cdf['extra'] = None

        sources = {}  # plasmid name -> list of compute nodes ids
        for i, r in tu_in_sources.groupby('source').agg(list).iterrows():
            # order matters.
            group = []
            for t in r['TU']:
                group.append(tmpdf[tmpdf.tuid == t].index[0])
            sources[i] = group

        for k, v in sources.items():
            nid = uidGen()
            newsource = GraphComputeNode(nid, 'source', None, [cdf.loc[vv].cdg_output for vv in v])
            newsource.output_to = [cdf.loc[vv].output_to[0] for vv in v]
            # and update input_from of these nodes too
            cdf.loc[[o[0] for o in newsource.output_to], 'input_from'] = [nid] * len(
                newsource.output_to
            )
            cdf = pd.concat([cdf, pd.DataFrame([newsource.toDict()]).set_index('id')]).drop(v)

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
                newaggregation = GraphComputeNode(nid, 'aggregation', None, r.source)
                # find the compute node id through the source_id column
                newaggregation.output_to = [(cdf[cdf.source_id == s].index[0], 0) for s in r.source]
                # add the input_from to the cooresponding sources
                for s in r.source:
                    cdf.loc[cdf.source_id == s, 'input_from'] = [[nid]]
                # For aggregations, we will store a dictionnary with the name of the aggregation and the ratio of each source
                tmp = pd.DataFrame([newaggregation.toDict()]).set_index('id')
                tmp['extra'] = [{'id': i, 'qtty': np.sum(r.ratio), 'ratios': r.ratio}]
                cdf = pd.concat([cdf, tmp])

            else:
                # no need for an aggregation node if there is only one source
                cdf.loc[cdf.source_id == r.source[0], 'extra'] = [{'qtty': np.sum(r.ratio)}]

        return cdf

    def __buildRawGraph(self, uidGen: Callable[[], int]) -> List[GraphComputeNode]:
        cdg = self.central_dogma_graph
        assert cdg is not None
        assert isinstance(cdg, pd.DataFrame)
        newnodes = []

        # we start building the compute graph from the output:
        output_gene_nodes = cdg[cdg.is_output]

        onode = GraphComputeNode(uidGen(), 'output', [], None)
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
                        newn = GraphComputeNode(nid, ntype, gn.predecessor, int(n_inp))
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
            newnode = GraphComputeNode(nid, 'numeric', None, 1)
            newnode.output_to = [(i, 0)]
            tmp = pd.DataFrame([newnode.toDict()]).set_index('id')
            extra = {
                'role': 'copy_number',
            }
            if 'qtty' in r.extra:
                extra['qtty'] = r.extra['qtty']
            tmp['extra'] = [extra]
            cdf = pd.concat([cdf, tmp])
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

        self.compute_graph = cdf
        self.cleanup()

    #                                                                            }}}
    ## ─────────────────────────────────────────────────────────────────────────────

    def cleanup(self):
        if self.compute_graph is not None:
            self.compute_graph.source_id = self.compute_graph.source_id.apply(
                lambda x: str(x) if not pd.isnull(x) else None
            )
            self.compute_graph = self.compute_graph.replace({np.nan: None})
            self.compute_graph.cdg_input = self.compute_graph.cdg_input.apply(
                lambda x: [int(x)] if isinstance(x, int) else x
            )

            # reconstruct all input_froms from the output_to:

            # first make sure input_from is of the right size
            for i, r in self.compute_graph.iterrows():
                output_to_me = self.compute_graph[
                    self.compute_graph.output_to.apply(lambda x: i in [y[0] for y in x])
                ]
                self.compute_graph.loc[i, 'input_from'] = [None] * len(output_to_me)

            # then fill it
            for i, r in self.compute_graph.iterrows():
                for p, o in enumerate(r.output_to):
                    self.compute_graph.loc[o[0], 'input_from'][o[1]] = (i, p)


    def copy(self):
        N = Network(self.lib, self.name, self.db, custom_outputs=self.custom_outputs, build=False)
        N.transcription_units = self.transcription_units.copy()
        N.central_dogma_graph = self.central_dogma_graph.copy()
        N.compute_graph = self.compute_graph.copy()
        return N


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                       --     the inverter     --
# ···············································································
# When training the intrinsic parameters of the model (aka the simulation part),
# we actually don't really have data about the numeric values involved (which would
# logically be the inputs of the network, as they most likely represent copy numbers).
# Indeed, training data is only a list of output fluorescence.
# The model needs copy numbers, or at least some number correlated to copy numbers,
# in order to compute the output.
# One solution is to make sure there's always an invertible path that links some component of
# the output to the copy numbers.
# When traning from xp data, we prepend an inverter module to the network,
# and we use (part of) the output as both input and target.

# The plan:
# -> define the inverse version of some compute nodes
# -> take a model, and find all invertible path from copy numbers to output
# -> prepend the invertible paths to the model, define inputs from output


def get_invertible_paths(network, start_node_id, inverse_dict):
    def _is_invertible(node):
        invertible = node.type in inverse_dict and len(node['input_from']) <= 1
        return invertible

    paths = []
    # we want ALL paths from start_node_id to output nodes that consist of invertible nodes
    def _get_invertible_paths(network, start_node_id, path, visited):
        nonlocal paths
        node = network.compute_graph.loc[start_node_id]

        # we need to know how we are connected to the output node
        # i.e we store the path as a list of (node_id, output_id) tuples except for the last node
        # (the output node), where we store (output_id, input_id) instead

        assert node.type != 'output'

        if start_node_id in visited or not _is_invertible(node):
            return []
        visited.add(start_node_id)

        for o, (n, i) in enumerate(node['output_to']):
            if network.compute_graph.loc[n].type == 'output':
                paths.append(path + [(n, i)])
            else:
                _get_invertible_paths(network, n, path + [(n, o)], visited)

    _get_invertible_paths(network, start_node_id, [], set())

    return paths


def inverted_network(network: Network, nodes: str = 'auto', inverse_dict=INVERSE_NODES_DICT):
    # inverse_dict: node_type -> inverse_node_type
    if nodes == 'auto':
        # we assume all numeric nodes should be linked to an inverted path
        start_nodes = network.compute_graph[
            network.compute_graph['type'] == 'numeric'
        ].index.tolist()
    elif isinstance(nodes, str) or not isinstance(nodes, Iterable):
        raise ValueError(f"Unrecognized node mode: {nodes}. Use 'auto' or a list of node ids.")
    else:
        start_nodes = nodes
    invertible_paths = {n: get_invertible_paths(network, n, inverse_dict) for n in start_nodes}
    new_network = network.copy()

    uidGen = ut.uniqueIdGenerator(start=new_network.compute_graph.index.max() + 1)

    # we pick the shortest path for each node
    paths = {n: min(invertible_paths[n], key=len) for n in start_nodes}

    inputpos = 0
    for start_n, path in paths.items():
        # we start by replacing the start node by the first node of the path
        new_network.compute_graph.loc[start_n, 'type'] = inverse_dict[
            new_network.compute_graph.loc[start_n, 'type']
        ]
        prev = start_n

        for i, (node_id, slot) in enumerate(
            path[1:]
        ):  # slot is output_id for nodes, input_id for output
            original_node = new_network.compute_graph.loc[node_id]  # the non inverted node
            n_type = original_node['type']
            nid = uidGen()

            if n_type == 'output':  # special case when we reach the output
                assert i == len(path) - 2, 'output node should be the last node in the path'
                # we add an input node
                in_n = GraphComputeNode(nid, 'input', None, None)
                in_n.output_to = [(prev, 0)]
                in_n.input_from = []
                in_n.extra = {'input_from_output': slot, 'input_position': inputpos}
                inputpos += 1
                new_network.compute_graph = pd.concat(
                    [new_network.compute_graph, pd.DataFrame([in_n.toDict()]).set_index('id')]
                )

                break

            # General case, create a new node and prepend to prev
            cdg_in = new_network.compute_graph.loc[prev, 'cdg_output']
            if isinstance(cdg_in, list):
                cdg_in = cdg_in[slot]
            new_n = GraphComputeNode(
                nid,
                inverse_dict[n_type],
                # get same cdg input / output as original node
                cdg_in,
                original_node.cdg_output,
            )
            new_n.output_to = [(prev, 0)]
            # inverse node always have only one input and one output
            # but we need to store the original output slot id in the extra field
            # so that we can use it when converting aggregation nodes for example
            # (where we convert a single input / multi output node to a single input / single output nodes
            # but we need to know which path, i.e slot, to use)
            new_n.extra = {
                'original_output_slot': slot,
                'is_inverse_of': node_id,
            }
            # set prev input_from to new nodes
            new_network.compute_graph.loc[prev, 'input_from'] = [(nid, 0)]
            new_network.compute_graph = pd.concat(
                [new_network.compute_graph, pd.DataFrame([new_n.toDict()]).set_index('id')]
            )

            prev = nid

    new_network.cleanup()

    return new_network


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
