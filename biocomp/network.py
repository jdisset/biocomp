from .library import PartsLibrary as PartsLibrary
import jax
import numpy as np
import pandas as pd
from . import utils as ut
from typing import Callable, List, Dict, Tuple, Iterable, Optional, cast
from itertools import product


part_type_to_parameter_name = {'promoter': 'tc_rate', 'uORF_group': 'tl_rate'}
parameter_to_default_part = {'tl_rate': 'empty_tc', 'tc_rate': 'empty'}


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                   --     general network utils     --
# ···············································································


def fuse_consecutive(cg: pd.DataFrame, types_to_fuse: Tuple[str, str], new_type: str):
    """Fuse 2 consecutive nodes in a graph when they are of the types specified in types_to_fuse"""
    assert len(types_to_fuse) == 2
    has_fused = True
    while has_fused:
        has_fused = False
        for first_id, first in cg[cg['type'] == types_to_fuse[0]].iterrows():
            second = [o[0] for o in first['output_to'] if cg.loc[o[0]]['type'] == types_to_fuse[1]]
            if len(second) > 0:
                second = cg.loc[second[0]]
                new_node = first.copy()
                new_node.update(
                    {
                        'type': new_type,
                        'output_to': second['output_to'],
                        'cdg_output': second['cdg_output'],
                        'extra': {
                            p: second['extra'][p] for p in second['extra'] if p.count('input') == 0
                        }.update(
                            {p: first['extra'][p] for p in first['extra'] if p.count('output') == 0}
                        ),
                    }
                )
                cg.loc[first_id] = new_node
                # then we also need to update the input_from of the nodes that were connected to second
                for i, to in enumerate(second['output_to']):
                    cg.loc[to[0]]['input_from'][to[1]] = (first_id, i)
                cg.drop(second.name, inplace=True)
                has_fused = True
                break


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                       --     base classes     --
# ···············································································
class Slot:
    def __init__(self, lib, part):
        self.part = part
        self.maps_to_parameter = None
        if self.part == [] or self.part == [None]:
            self.part = None
        if isinstance(self.part, list):
            mapped = list(
                set([self.__mapped_parameter(lib, p) for p in self.part if p is not None])
            )
            if len(mapped) != 1:
                raise ValueError(f'{self.part} maps to {len(mapped)} parameters ({mapped})')
            self.maps_to_parameter = mapped[0]
        else:
            self.maps_to_parameter = self.__mapped_parameter(lib, self.part)
        if self.maps_to_parameter is not None and not isinstance(self.part, list):
            self.part = [self.part]

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
        if self.maps_to_parameter is None:
            if self.part is None:
                return '<empty slot>'
            else:
                return f'<{self.part}>'
        return f'<{self.part} -> {self.maps_to_parameter}>'


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
        self.is_inverse_of = None

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
            "is_inverse_of": self.is_inverse_of,
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
        self.__get_parameters()

    def __get_parameters(self):
        for s in self.slots:
            if s.maps_to_parameter is not None:
                assert s.maps_to_parameter not in self.params
                self.params[s.maps_to_parameter] = s.part
        # then for each param that is not in the slots, add it with default value
        for _, p in part_type_to_parameter_name.items():
            if p not in self.params:
                try:
                    self.params[p] = [parameter_to_default_part[p]]
                except KeyError:
                    msg = f'No default part for parameter {p}'
                    msg += f' (part_type_to_parameter_name: {part_type_to_parameter_name})'
                    msg += f' (parameter_to_default_part: {parameter_to_default_part})'
                    raise ValueError(msg)

    def __repr__(self):
        return f'L1({self.slots})'


# def resolve(self, lib, *args, **kwargs):
# if not self.is_resolved:
# self.part = self.resolve_function(lib, *args, **kwargs)
# if self.part == [] or self.part == [None]:
# self.part = None
# if isinstance(self.part, list):
# mapped = list(set([self.__mapped_parameter(lib, p) for p in self.part if p is not None]))
# if len(mapped) != 1:
# raise ValueError(f'{self.part} maps to {len(mapped)} parameters ({mapped})')
# self.maps_to_parameter = mapped[0]
# else:
# self.maps_to_parameter = self.__mapped_parameter(lib, self.part)
# if self.maps_to_parameter is not None and not isinstance(self.part, list):
# self.part = [self.part]
# self.is_resolved = True


class TranscriptionUnitGenerator:
    def __init__(self, part_generators):
        self.name = ''
        self.part_generators = part_generators

    def generate_all(self, lib, order=None, *args, **kwargs):
        if order is None:
            order = list(range(len(self.part_generators)))

        # for each slot, generate all possible parts.
        # but a slot needs the parts from all previous slots to be generated
        # so we need to do this in order
        def _next(slots, i):
            if i == len(order):
                yield slots
            else:
                g = self.part_generators[order[i]]
                possile_parts = g(lib, slots, *args, **kwargs)
                for p in possile_parts:
                    yield from _next(slots + [Slot(lib, p)], i + 1)

        return _next([], 0)

    # def generate_random(self, lib, random_seed=1, random_order=True):
    # rdm = jax.random.PRNGKey(random_seed)
    # allrdm = jax.random.split(rdm, len(self.slots))
    # order = list(range(len(self.slots)))
    # if random_order:
    # order = jax.random.permutation(rdm, len(self.slots))
    # for i, r in zip(order, allrdm):
    # if not self.slots[i].is_resolved:
    # self.slots[i].resolve(lib, l1=self, rdm_key=r)
    # assert all(s.is_resolved for s in self.slots)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


def transcription_unit_from_L1(l1id, lib):
    l0_cols = ["insulator", "promoter", "5'UTR", "gene", "3'UTR", "terminator"]
    L0s = lib.L1s.loc[l1id][l0_cols].tolist()
    part_cols = [f'part_{i}' for i in range(1, 7)]
    parts = []
    for l in L0s:
        try:
            parts += [p for p in lib.L0s.loc[l][part_cols].tolist() if p]
        except Exception as e:
            msg = f'Error in L0 {l} of L1 {l1id}: {e}'
            msg += f'\npart_cols: {part_cols}'
            msg += f'\nlib.L0s[{l}]: {lib.L0s.loc[l]}'
            msg += f'\nlib.L0s: {lib.L0s}'

            raise RuntimeError(msg)
    tu = TranscriptionUnit([Slot(lib, p) for p in parts])
    return tu


# main class: a network of interacting transcription units
class Network:
    def __init__(self, lib, recipe_name, recipe_db, custom_outputs=None, build=True):
        self.lib = lib
        self.name: str = recipe_name
        self.custom_outputs = custom_outputs
        self.transcription_units: Optional[Dict[str, TranscriptionUnit]] = None
        self.compute_graph: Optional[pd.DataFrame] = None
        self.central_dogma_graph: Optional[pd.DataFrame] = None
        self.aggregations: Optional[pd.DataFrame] = None
        self.tu_inputs: Optional[pd.DataFrame] = None
        self.db = recipe_db
        if recipe_db is not None:
            self.db.commit()
            self.__build_from_db()
        if build:
            self.build()

    ## ───────────────────────────────────── ▼ ─────────────────────────────────────
    # {{{                  --     public tools & utils    --
    # ···············································································

    def get_compute_types(self):
        node_dict = self.compute_graph.groupby('type').apply(lambda x: x.index.to_list()).to_dict()
        return node_dict

    # def get_compute_types_and_extra(self):
    # gb = self.compute_graph.groupby('type')
    # # extra is a dict, so we need to json it
    # gb = gb['extra'].apply(lambda x: x.apply(partial(json.dumps, default=np_converter)).tolist())
    # gb = gb.apply(lambda x: list(set(x)))
    # # we want to build a dict of {type: {extra: [indices]}}
    # # but we have a series of {extra: [indices]}
    # # so we need to build a dict for each extra
    # gb = gb.apply(lambda x: {e: i for e, i in zip(x, gb.index)})
    # return gb.to_dict()

    #                                                                            }}}
    ## ─────────────────────────────────────────────────────────────────────────────

    def __build_from_db(self):
        assert self.db is not None
        c = self.db.cursor()
        # first let's check that there is a recipe with this name
        c.execute('SELECT * FROM recipes WHERE name=?', (self.name,))
        assert (
            c.fetchone() is not None
        ), f'No recipe named {self.name} in database {self.db}. Available recipes: {c.execute("SELECT name FROM recipes").fetchall()}'

        # get the transcription units
        c.execute(
            """SELECT TU, TU || '_' || aggregation as name FROM TU_in_source tis, source_in_aggregation sia, aggregations a
           WHERE tis.source = sia.source AND sia.aggregation = a.id AND a.recipe = ?""",
            (self.name,),
        )
        self.transcription_units = {
            tu[1]: transcription_unit_from_L1(tu[0], self.lib) for i, tu in enumerate(c.fetchall())
        }

        # then get the sources
        c.execute(
            """SELECT tis.source || '_' || aggregation as source, TU || '_' || aggregation as TU, position
            FROM TU_in_source tis, source_in_aggregation sia, aggregations a
           WHERE tis.source = sia.source AND sia.aggregation = a.id AND a.recipe = ?""",
            (self.name,),
        )
        tu_in_sources = pd.DataFrame(
            [t for t in c.fetchall()], columns=['source', 'TU', 'position']
        )
        tu_in_sources.sort_values(by='position', inplace=True)
        self.tu_in_sources = tu_in_sources

        # finally get the aggregations
        c.execute(
            """SELECT a.id, sia.source || '_' || aggregation, sia.ratio FROM aggregations a, source_in_aggregation sia
            WHERE a.id = sia.aggregation AND a.recipe = ?""",
            (self.name,),
        )
        # adding the aggregation nodes
        aggregations = (
            pd.DataFrame([t for t in c.fetchall()], columns=['id', 'source', 'ratio'])
            .groupby('id')
            .agg(list)
        )
        self.aggregations = aggregations

    @classmethod
    def from_dict(cls, lib, name, transcription_units, sources, aggregations, build=True):
        n = cls(lib, name, None, build=False)

        # transcription_units = {TU_name : TU}
        n.transcription_units = transcription_units

        # sources =  {source_name: [TU1, TU2, TU3, ...], ...}
        n.tu_in_sources = pd.DataFrame(
            [
                {'source': s, 'TU': t, 'position': i}
                for s, tuids in sources.items()
                for i, t in enumerate(tuids)
            ]
        )
        n.tu_in_sources.sort_values(by='position', inplace=True)

        # aggregations = [[source1, source2, source3, ...], ...]
        assert n.aggregations is None
        n.aggregations = (
            pd.DataFrame(
                [{'id': i, 'source': s, 'ratio': 1} for i, a in enumerate(aggregations) for s in a]
            )
            .groupby('id')
            .agg(list)
        )

        if build:
            n.__build_central_dogma_graph()
            n.__build_compute_graph()
        return n

    def build(self):

        assert len(self.transcription_units) > 0, f'No transcription units in recipe {self.name}'
        self.__build_central_dogma_graph(self.custom_outputs)
        self.__build_compute_graph()

    def set_inputs(self, input_ids):
        assert self.is_built()
        ut.debug(f'setting inputs to {input_ids}')
        for i, inp_id in enumerate(input_ids):
            self.compute_graph.loc[inp_id, 'type'] = 'input'
            self.compute_graph.loc[inp_id, 'extra'].update({'input_position': i})

    def set_numeric_as_input(self):
        assert self.is_built()
        numeric_nodes = list(self.compute_graph.loc[self.compute_graph['type'] == 'numeric'].index)
        self.set_inputs(numeric_nodes)

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

    def __isOutputedBy(
        self, cdg_input_node: int, compute_nodes: List[GraphComputeNode]
    ) -> List[GraphComputeNode]:
        """returns a list of all the compute nodes that have cdg_input_node as output"""
        return [n for n in compute_nodes if cdg_input_node == n.cdg_output]

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
        tu: List[dict] = []
        assert self.transcription_units is not None

        def make_hashable(x):
            return tuple(sorted((k, tuple(v)) for k, v in x.items()))

        for tuid, t in self.transcription_units.items():
            dna, dna_params = self.__getDna(t)
            rna, rna_params = self.__getRna(t)
            prt, prt_params = self.__getPrt(t)
            tu.append(
                {
                    'name': tuid,
                    'DNA': dna,
                    'DNA_params': dna_params,
                    'DNA_params_hashable': make_hashable(dna_params),
                    'RNA': rna,
                    'RNA_params': rna_params,
                    'RNA_params_hashable': make_hashable(rna_params),
                    'PRT': prt,
                    'PRT_params': prt_params,
                    'PRT_params_hashable': make_hashable(prt_params),
                }
            )
        assert tu is not None
        tudf = pd.DataFrame(tu)

        # transcription units are never grouped
        dna_df = pd.DataFrame({'tu_id': [[x] for x in cast(str, tudf['name'])], 'type': 'DNA'})

        def only_one_value_per_param(params: Dict[str, List[str]]) -> bool:
            return all(len(parts) <= 1 for _, parts in params.items())

        rna_tuids_noparams = list(
            tudf[tudf['RNA_params'].map(len) == 0].groupby(by='RNA').agg(list).name  # type: ignore
        )

        try:
            rna_tuids_oneparamvalue = (
                tudf[tudf['RNA_params'].map(len) > 0]
                .groupby(by='RNA')
                .filter(lambda x: only_one_value_per_param(x['RNA_params']))
                .groupby(by=['RNA', 'RNA_params_hashable'])
                .agg(list)
            )
        except Exception as e:
            msg = f'Error while grouping RNA that have one params: {e}\n'
            msg += f'tudf: \n{tudf}'
            raise Exception(msg)

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

        # connect DNA to RNA through successor list
        for i, r in cdg[cdg.type == 'DNA'].iterrows():
            cdg.loc[i, 'successor'] = []
            for ii, rr in cdg[cdg.type == 'RNA'].iterrows():
                assert (
                    len(r.tu_id) == 1
                ), "a DNA node should have only one value in its tu_id list (1 DNA node per Transcription Unit)"

                if r.tu_id[0] in rr.tu_id:  # if we have an RNA that has the same TU as the DNA
                    cdg.loc[i, 'successor'].append(ii)  # add the RNA to the DNA's successor

        # connect RNA to PRT through successor list
        for i_r, rna in cdg[cdg.type == 'RNA'].iterrows():  # for each RNA
            cdg.loc[i_r, 'successor'] = []
            for i_p, prt in cdg[cdg.type == 'PRT'].iterrows():  # for each PRT
                if set(rna.tu_id).issubset(set(prt.tu_id)):
                    cdg.loc[i_r, 'successor'].append(i_p)  # add the PRT to the RNA's successor

        # now deduce the predecessor lists
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
        # in the compute graph,
        # merge TUs that are from a same source into a single source node (aka plasmid)

        sources_tuids = self.central_dogma_graph.loc[
            cdf[cdf.type == 'source'].cdg_output
        ].tu_id.apply(lambda x: x[0])

        tmpdf = pd.DataFrame(
            {'compute_id': cdf[cdf.type == 'source'].index, 'tuid': sources_tuids}
        ).set_index('compute_id')
        # tmpdf is a mapping between the computegraph ids of every sources and their TUids

        cdf['source_id'] = None
        # cdf['extra'] = None

        sources = {}  # plasmid name -> list of compute nodes ids

        # tu_in_sources contains the list of TUs in each source, sorted by position
        assert self.tu_in_sources is not None

        for i, r in self.tu_in_sources.groupby('source').agg(list).iterrows():
            # but you can have sources in the db that are not in the recipe
            group = []  # group will contain the compute nodes ids of the TUs in the source
            for t in r['TU']:
                try:
                    group.append(tmpdf[tmpdf.tuid == t].index[0])
                except IndexError:
                    msg = f'Error while merging sources. TU {t} not found in tmpdf.'
                    msg += f'\n\ntmpdf: \n{tmpdf}'
                    msg += f'\nsources_tuids: \n{sources_tuids}'
                    msg += f'\ncentral dogma graph: \n{self.central_dogma_graph}'
                    msg += f'\ncdf: \n{cdf}'
                    raise Exception(msg)

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
        assert self.aggregations is not None
        for i, r in self.aggregations.iterrows():
            if len(r.source) > 1:
                nid = uidGen()
                newaggregation = GraphComputeNode(nid, 'aggregation', None, r.source)
                # find the compute node id through the source_id column
                try:
                    newaggregation.output_to = [
                        (cdf[cdf.source_id == s].index[0], 0) for s in r.source
                    ]
                except Exception as e:
                    msg = f'Error while adding aggregation node {nid} to compute graph'
                    msg += f' (recipe {self.name}, aggregation {i}, sources {r.source})'
                    msg += f'\n\naggregations: \n{self.aggregations}\n'
                    msg += f'\n{e}'
                    msg += f'\n{cdf}'
                    raise RuntimeError(msg)
                # problem: some sources have no ids!!

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

        # we start building the compute graph by adding the output:
        output_gene_nodes = cdg[cdg.is_output]
        onode = GraphComputeNode(uidGen(), 'output', [], None)
        for i, r in output_gene_nodes.iterrows():
            onode.cdg_input += [i]
        newnodes.append(onode)

        tu_in_sequestron = set()

        # then we add the sequestron nodes with an associated list of their cdg input nodes
        enabled_sequestrons = self.lib.get_enabled_sequestrons()
        for _, r in enabled_sequestrons.iterrows():
            # sequestrons have 2 input hubs, negative and positive
            nlvl = cdg[cdg.type == r.negative_level]  # negative level (PRT, RNA, DNA)
            nparts = nlvl[nlvl.content.apply(lambda x: r.negative_part in x)]

            plvl = cdg[cdg.type == r.positive_level]  # positive level (PRT, RNA, DNA)
            pparts = plvl[plvl.content.apply(lambda x: r.positive_part in x)]

            olvl = cdg[cdg.type == r.output_level]  # output level (PRT, RNA, DNA)
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
                    # we get a unique name for the  sequestron by concatenating the name of the negative and positive parts
                    cnode.extra = {'seq_name': f'{r.type}::{r.negative_part}#{r.positive_part}'}
                    newnodes.append(cnode)
                    # useful to track which tus are in use
                    tu_in_sequestron.update(oparts['tu_id'].iloc[0])
                    tu_in_sequestron.update(nparts['tu_id'].iloc[0])
                    tu_in_sequestron.update(pparts['tu_id'].iloc[0])
                except Exception as e:
                    msg = f'Error while building compute graph for recipe {self.name}:\n{e}'
                    msg += f'Sequestron {r.type}.\nnparts: {nparts}, pparts: {pparts}, oparts: {oparts}'
                    msg += f'\nolevel: {olvl}, oparts: {oparts}, outlevel: {r.output_level}, outpart: {r.output_part}'
                    raise RuntimeError(msg)

        cg = []

        # right now newnodes contains only the output node and the sequestron nodes
        # we're now going to go upstream from these nodes, adding the relevant translation/transcription
        # nodes until we reach the Transcription Units.

        # Before that, let's also add dead-end paths if there are any.
        # Dead-end paths start with a transcription unit and end with a gene that is not an output.
        # They're practically useless (and we can optimize them out during computation) but they're still
        # part of the graph (and merit being visualized).
        # We add all proteins that are not outputs, and whose tu is not part of any sequestron.
        deadend_nodes = cdg[
            (cdg.is_output == False)
            & (cdg.type == 'PRT')
            & (cdg.tu_id.apply(lambda x: all([xx not in tu_in_sequestron for xx in x])))
        ]
        for i, r in deadend_nodes.iterrows():
            cnode = GraphComputeNode(uidGen(), 'deadend', [i], None)
            newnodes.append(cnode)

        # At first, each TU also has a corresponding source node that we need to add,
        # but we will merge sources later (for TUs that are on a same plasmid)
        while newnodes:
            n: GraphComputeNode = newnodes.pop()
            if n.type != 'source':
                # for every cdg node that is an input of n
                if n.cdg_input is None:
                    msg = f'Error while building compute graph for recipe {self.name}:\n'
                    msg += f'No cdg_input for node {n.id} of type {n.type}:\n{n}'
                    msg += f'Content of its cdg_output node:\n{cdg.loc[n.cdg_output]}'
                    msg += f'CDG:\n{cdg}'
                    raise RuntimeError(msg)
                for i, n_inp in enumerate(n.cdg_input):
                    others = self.__isOutputedBy(
                        n_inp, cg + newnodes
                    )  # list of all other nodes that also output n_inp
                    for other in others:
                        # establish the connection between n and its parents
                        n.input_from += [other.id]
                        other.output_to += [(n.id, i)]
                    if not others:  # if n_inp is not outputed by any node we have already created
                        # then we go up the central dogma and create the matching upstream node
                        gn = cdg.loc[
                            n_inp
                        ]  # the central dogma graph node that is being transformed by this compute node
                        nid = uidGen()
                        # we just need to know what type of transform we have.
                        # for example, if the input cdg node that our compute node expects is a protein,
                        # that means that we need to add a translation node, etc...
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
        ut.debug(f'Adding numeric nodes for {len(topnodes)} top nodes: {topnodes}')
        for i, r in topnodes.iterrows():
            nid = uidGen()
            newnode = GraphComputeNode(nid, 'numeric', None, 1)
            newnode.output_to = [(i, 0)]
            tmp = pd.DataFrame([newnode.toDict()]).set_index('id')
            extra = {
                'role': 'copy_number',
            }
            if r.extra is not None and 'qtty' in r.extra:
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
        self.compute_graph = pd.DataFrame([n.toDict() for n in cg]).set_index('id').sort_index()

        # there should be the same number of sources in the cdf compute graph as DNA nodes in the cdg
        nsources = len(self.compute_graph[self.compute_graph.type == 'source'])
        ndna = len(self.central_dogma_graph[self.central_dogma_graph.type == 'DNA'])
        if nsources != ndna:
            dna_in_compute_graph = self.compute_graph[
                self.compute_graph.type == 'source'
            ].cdg_output.tolist()

            extradna = []
            for i, r in self.central_dogma_graph[self.central_dogma_graph.type == 'DNA'].iterrows():
                if i not in dna_in_compute_graph:
                    extradna += r.tu_id
            msg = f'When building compute graph for recipe {self.name}, '
            msg += f'found {nsources} DNA sources in the graph, but {ndna} DNA nodes total.'
            msg += f'\nExtra DNA nodes: {extradna}'
            raise RuntimeError(msg)

        self.compute_graph = self.__mergeSources(
            self.compute_graph, uidGen
        )  # merge TUs with same source

        self.compute_graph = self.__addAggregations(
            self.compute_graph, uidGen
        )  # add aggregation nodes

        self.compute_graph = self.__addNumericNodes(
            self.compute_graph, uidGen
        )  # now add numeric nodes (constant or inuts)

        self.cleanup()
        self._sanity_check()

    #                                                                            }}}
    ## ─────────────────────────────────────────────────────────────────────────────

    def get_output_proteins(self):
        """Returns the names of the proteins that are outputs of the network"""
        onode = self.compute_graph[self.compute_graph['type'] == 'output']
        assert len(onode) == 1, f'Invalid number of output nodes: {len(onode)}'
        return [
            self.central_dogma_graph.loc[cdg_id]['content'][0]
            for cdg_id in onode.iloc[0]['cdg_input']
        ]

    def get_input_from_output(self, output_arr):
        """Given an array of output values, returns the columns that are inputs of the inverted network,
        properly ordered by input number"""
        # In inverted networks, each input node has,
        # in its extra, 'input_from_output' and 'input_position' (which get_inverted_input_positions uses)
        # We want to transform output_arr by reordering the columns accordingly
        mapping = self.get_inverted_input_positions()
        return output_arr[:, [mapping[i] for i in range(len(mapping))]]

    def get_inverted_input_proteins(self):
        """Returns the names of the proteins that are inputs of the inverted network, ordered"""
        mapping = self.get_inverted_input_positions()
        output_proteins = self.get_output_proteins()
        assert len(mapping) <= len(output_proteins)
        return [output_proteins[mapping[i]] for i in range(len(mapping))]

    def get_inverted_input_positions(self):
        """Returns a mapping from input position to output position"""
        mapping = {}  # input number -> output position
        inputs = self.compute_graph[self.compute_graph['type'] == 'input']
        for _, row in inputs.iterrows():
            assert 'input_position' in row.extra
            assert 'input_from_output' in row.extra
            assert row.extra['input_position'] not in mapping
            mapping[row.extra['input_position']] = row.extra['input_from_output']
        assert set(mapping.keys()) == set(range(len(mapping.keys())))
        assert len(mapping.keys()) == len(set(mapping.values()))
        return mapping

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
                try:
                    self.compute_graph.at[i, 'input_from'] = [None] * len(output_to_me)
                except Exception as e:
                    msg = f'Error cleaning up compute graph: {e}\n'
                    msg += f'Trying to construct input_froms of node {i} from upstream outputs.\n'
                    msg += f'{r}'
                    msg += f'\nDetected upstream outputs:\n{output_to_me}'
                    raise RuntimeError(msg)

            # then fill it
            for i, r in self.compute_graph.iterrows():
                for p, o in enumerate(r.output_to):
                    try:
                        self.compute_graph.at[o[0], 'input_from'][o[1]] = (i, p)
                    except Exception as e:
                        msg = f'Error cleaning up compute graph: {e}\n'
                        msg += f'currently processing {o}.\n'
                        msg += f'\ninput node is:\n{r}'
                        msg += f'\noutput node is:\n{self.compute_graph.loc[o[0]]}'
                        raise RuntimeError(msg)

            # make sure the proper quantile variable is assigned to each node
            self._assign_quantile_variable()

        self._sanity_check()

    def _sanity_check(self):
        # check that all nodes have a unique id
        if self.compute_graph is not None:
            assert len(self.compute_graph.index) == len(
                set(self.compute_graph.index)
            ), 'compute graph has duplicate ids'

            # every source node should have a source_id
            for i, r in self.compute_graph[self.compute_graph.type == 'source'].iterrows():
                if r.source_id is None:
                    msg = (
                        f'In compute graph for recipe {self.name}, source node {i} has no source_id'
                    )
                    msg += f'\n{self.compute_graph}'
                    raise RuntimeError(msg)

        if self.central_dogma_graph is not None:
            assert len(self.central_dogma_graph.index) == len(
                set(self.central_dogma_graph.index)
            ), 'central dogma graph has duplicate ids'

    def copy(self):
        N = Network(self.lib, self.name, None, custom_outputs=self.custom_outputs, build=False)
        N.db = self.db
        N.transcription_units = self.transcription_units.copy()
        N.central_dogma_graph = self.central_dogma_graph.copy()
        N.compute_graph = self.compute_graph.copy()
        N.tu_in_sources = self.tu_in_sources.copy()
        N.aggregations = self.aggregations.copy()
        return N

    def _assign_quantile_variable(self):
        """
        Assigns the correct quantile variable to each node of the compute graph.
        Proceeds by propagating from the output towards the upstream nodes.
        Inverted nodes are assigned the same quantile as their forward node.
        If a node is linked is not linked to a non-inverted one, has only one output
        but is linked downstream to paths that lead to multiple outputs,
        quantile_variable_id is set to [None].
        """
        cg = self.compute_graph

        def propagate_upstream(node, quantile_id, output_id):
            node['extra'].setdefault('quantile_variable_id', [])
            if node.is_inverse_of is not None:
                node['extra']['quantile_variable_id'] = cg.loc[node.is_inverse_of]['extra'].get(
                    'quantile_variable_id', []
                )
            else:
                if len(node['extra']['quantile_variable_id']) <= output_id:
                    # append -1 until the right size
                    node['extra']['quantile_variable_id'].extend(
                        [-1] * (output_id - len(node['extra']['quantile_variable_id']) + 1)
                    )
                if node['extra']['quantile_variable_id'][output_id] == -1:
                    node['extra']['quantile_variable_id'][output_id] = quantile_id
                else:
                    # another node already set the quantile var!
                    # It means we found a node with a single output but linked to multiple downstream
                    # paths. We could append the quantile var but the order would be random. I prefer
                    # to remove the footgun entirely and just not add a quantile variable for this node.
                    # At the time I'm writing this the only case that would happen are for the numeric nodes
                    # or the inputs. They definitely don't need the quantile var.
                    # We change the existing value to None for this special case.
                    node['extra']['quantile_variable_id'][output_id] = None

            if node.input_from:
                for nid, oid in node.input_from:
                    propagate_upstream(cg.loc[nid], quantile_id, oid)

        # first let's remove all "quantile_variable_id" from the extra column:
        for _, node in cg.iterrows():
            node['extra'].pop('quantile_variable_id', None)

        output_node = cg[cg.type == 'output'].iloc[0]
        # add the quantile variable to the output node
        output_node['extra']['quantile_variable_id'] = list(range(len(output_node.input_from)))
        for i, (nid, oid) in enumerate(output_node.input_from):
            propagate_upstream(cg.loc[nid], i, oid)

        # treat the case where we have a "deadend" node, i.e a branch that ends
        # without being connected to the output node. The node type is litteraly "deadend"
        # we'll just assign quantile 0
        deadend_nodes = cg[cg.type == 'deadend'].index
        for node_id in deadend_nodes:
            propagate_upstream(cg.loc[node_id], 0, 0)


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


DEFAULT_INVERSE_DICT = {
    "translation": "inv_translation",
    "transcription": "inv_transcription",
    "numeric": "inv_numeric",
    "aggregation": "inv_aggregation",
    "source": "inv_source",
}


def inverted_network(
    network: Network, nodes: str = 'auto', inverse_dict=DEFAULT_INVERSE_DICT, mode='shortest'
):
    ut.debug(f'Inverting network {network.name}')

    # inverse_dict: node_type -> inverse_node_type

    # First we pick the start nodes. (Numeric nodes by default, or user supplied list)
    # We will then try to find invertible paths that go from each of the start nodes to the output.
    # A path is invertible if each of its nodes has been marked as having an inverted equivalent in the inverse_dict
    # We then prepend an input node + the inverted nodes to the original network, and that's what we
    # call an inverted network.
    if nodes == 'auto':  # numeric nodes as start nodes
        start_nodes = network.compute_graph[
            network.compute_graph['type'] == 'numeric'
        ].index.tolist()
    elif not isinstance(nodes, Iterable):
        raise ValueError(f"Unrecognized node mode: {nodes}. Use 'auto' or a list of node ids.")
    else:  # list of nodes
        start_nodes = nodes

    # we compute a list of invertible paths that link each start nodes to the output
    inv_paths = {n: get_invertible_paths(network, n, inverse_dict) for n in start_nodes}

    # For each start_node, we might have more than one path.
    # In 'shortest' mode, we just pick the shortest one.
    # In the 'all' mode, we want to return every possible combination of paths per start node
    # e.g. if we have 2 start nodes, and 2 paths for each, we want to return 4 paths
    # (the cartesian product of the paths)
    if mode == 'shortest':
        inversions = [{n: min(p, key=len) for n, p in inv_paths.items() if p}]
    elif mode == 'all':
        inversions = [dict(zip(inv_paths.keys(), p)) for p in product(*inv_paths.values())]
    else:
        raise ValueError(f"Unrecognized mode: {mode}. Use 'shortest' or 'all'.")

    new_networks = []
    for paths in inversions:
        inputpos = 0
        new_network = network.copy()
        uidGen = ut.uniqueIdGenerator(start=new_network.compute_graph.index.max() + 1)
        for start_n, path in paths.items():
            # we start by replacing the start node by the first node of the path
            new_network.compute_graph.loc[start_n, 'type'] = inverse_dict[
                new_network.compute_graph.loc[start_n, 'type']
            ]
            prev = start_n

            for i, (node_id, slot) in enumerate(
                path
            ):  # slot is output_id for nodes, input_id for output
                original_node = new_network.compute_graph.loc[node_id]  # the non inverted node
                n_type = original_node['type']
                nid = uidGen()

                if n_type == 'output':  # special case when we reach the output
                    assert i == len(path) - 1, 'output node should be the last node in the path'
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
                    nid, inverse_dict[n_type], cdg_in, original_node.cdg_output
                )
                new_n.output_to = [(prev, 0)]
                # inverse nodes always have only one input and one output
                # but we need to store the original output slot id in the extra field
                # so that we can use it when converting aggregation nodes for example
                # (where we convert a single input / multi output node to a single input / single output node
                # but we need to know which path, i.e slot, to use)
                new_n.is_inverse_of = node_id
                new_n.extra = {
                    'original_output_slot': slot,
                    'original_output_len': len(original_node['output_to']),
                }

                # set prev input_from to new nodes
                new_network.compute_graph.loc[prev, 'input_from'] = [(nid, 0)]
                new_network.compute_graph = pd.concat(
                    [new_network.compute_graph, pd.DataFrame([new_n.toDict()]).set_index('id')]
                )

                prev = nid

        new_network.cleanup()
        new_networks.append(new_network)
    return new_networks


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
