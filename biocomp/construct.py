from .library import PartsLibrary as PartsLibrary
import jax
import numpy as np
import pandas as pd
from . import utils as ut
import os
import sqlite3

part_type_to_parameter_name = {'promoter': 'tx_rate', 'uORF': 'tl_rate'}

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

    def __mapped_parameter(self, lib, part_name, category_to_param=part_type_to_parameter_name):
        if part_name is not None:
            if part_name in lib.pc.index:
                category = lib.pc.loc[part_name, 'category']
                if category in category_to_param:
                    return category_to_param[category]
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

    def __repr__(self):
        return f'L1({self.slots})'


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{               --     helpers to import experiments     --
#···············································································

# into sqlite
def create_db(conn):
    sql = """
    CREATE TABLE IF NOT EXISTS `tubes` (
        name TEXT PRIMARY KEY,
        comment TEXT);

    CREATE TABLE IF NOT EXISTS `aggregations` (
        id INTEGER PRIMARY KEY,
        qtty REAL,
        tube TEXT,
        FOREIGN KEY (tube) REFERENCES tubes(name));

    CREATE TABLE IF NOT EXISTS `sources` (
        name TEXT PRIMARY KEY,
        type TEXT);

    CREATE TABLE IF NOT EXISTS `TU_in_source`(
        source TEXT,
        TU INTEGER,
        position INTEGER,
        FOREIGN KEY(source) REFERENCES sources(name),
        FOREIGN KEY(TU) REFERENCES TUs(id),
        PRIMARY KEY(source, TU));

    CREATE TABLE IF NOT EXISTS `source_in_aggregation`(
        aggregation INTEGER,
        source TEXT,
        ratio REAL,
        FOREIGN KEY (aggregation) REFERENCES aggregations(id),
        FOREIGN KEY (source) REFERENCES sources(name),
        PRIMARY KEY(aggregation, source));
    """
    c = conn.cursor()
    c.executescript(sql)


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


def json_to_sql(xpdict, conn, lib):
    TranscriptionUnits = {}
    c = conn.cursor()
    tubes = xpdict['tubes']
    for t in tubes:
        c.execute("SELECT name FROM tubes WHERE name = ?", (t['name'],))
        if c.fetchone():
            print(f'Warning: tube {t["name"]} already exists in the database')
            break
        c.execute("INSERT INTO tubes VALUES (?, ?)", (t['name'], t['comment']))
        for agg in t['content']:
            ratios = np.array([s['qtty'] for s in agg])
            qtty = float(np.sum(ratios))
            c.execute("INSERT INTO aggregations VALUES (?, ?, ?)", (None, qtty, t['name']))
            aggregation_id = c.lastrowid
            ratios = ratios / qtty
            for (r, s) in zip(ratios, agg):
                type = None
                l1ids = []
                if s['plasmid'] in lib.L1s.index:
                    type = 1
                    l1ids = [lib.L1s.loc[s['plasmid']].name]
                elif s['plasmid'] in lib.L2s.index:
                    type = 2
                    slot_cols = [f'slot_{i}' for i in range(1, 7)]
                    l1ids = [s for s in lib.L2s.loc[s['plasmid']][slot_cols].tolist() if s]
                if type is None:
                    raise Exception("Unknown plasmid")
                c.execute("SELECT name FROM sources WHERE name = ?", (s['plasmid'],))
                if not c.fetchone():
                    c.execute("INSERT INTO sources VALUES (?, ?)", (s['plasmid'], type))
                    for i, l1id in enumerate(l1ids):
                        c.execute(
                            "INSERT INTO TU_in_source VALUES (?, ?, ?)", (s['plasmid'], l1id, i)
                        )

                c.execute(
                    "INSERT INTO source_in_aggregation VALUES (?, ?, ?)",
                    (aggregation_id, s['plasmid'], r),
                )
    conn.commit()
    return TranscriptionUnits

# tubes:[#name, comment]
# aggregations:[#id, qtty, tube*]
# sources:[#name, type];
# TU_in_source:[#source*,#TU*,position]
# source_in_aggregation:[#aggregation*,#source*,ratio]


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     XP class     --
#···············································································

class XP:
    def __init__(self, dbconn, tube_name, lib):
        self.dbconnection = dbconn
        self.name = tube_name
        # select all transcription units in the tube
        c = self.dbconnection.cursor()

        print('tube name:',tube_name)
        c.execute(
            """SELECT TU FROM TU_in_source tis, source_in_aggregation sia, aggregations a 
           WHERE tis.source = sia.source AND sia.aggregation = a.id AND a.tube = ?""",
            (tube_name,),
        )
        # converter them to TranscriptionUnits objects
        self.tuids = [t[0] for t in c.fetchall()]

        self.transcription_units = [transcription_unit_from_L1(t, lib) for t in self.tuids]
        self.cdg = None
        self.outputs = []

    def __getDna(self, tu):
        return [s.part for s in tu.slots if s.is_resolved and not isinstance(s.part, list)]

    def __getRna(self, tu, lib):
        dna = self.__getDna(tu)
        d = lib.pc.loc[dna]
        return tuple(d[d.transcripted == 1].index)

    def __getPrt(self, tu, lib):
        dna = self.__getDna(tu)
        d = lib.pc.loc[dna]
        return tuple(d[d.translated == 1].index)

    def build_central_dogma_graph(self, lib):
        tu = [
            {
                'name': t,
                'DNA': self.__getDna(t),
                'RNA': self.__getRna(t, lib),
                'PRT': self.__getPrt(t, lib),
            }
            for t in self.transcription_units
        ]
        print(tu)

        tudf = pd.DataFrame(tu)

        dna_df = pd.DataFrame(
            {'tu_id': [[x] for x in tudf.name], 'type': 'DNA', 'successor': None}
        )
        rna_df = pd.DataFrame(
            {
                'tu_id': list(tudf.reset_index().groupby(by='RNA').agg(list)['name']),
                'type': 'RNA',
                'successor': None,
            }
        )
        prt_df = pd.DataFrame(
            {
                'tu_id': list(tudf.reset_index().groupby(by='PRT').agg(list)['name']),
                'type': 'PRT',
                'successor': None,
            }
        )

        # Then concatenate them:
        cdg = pd.concat([dna_df, rna_df, prt_df]).reset_index(drop=True)

        # Add successor and predecessor information:
        for _, r in cdg[cdg.type == 'RNA'].iterrows():
            cdg.loc[r.tu_id, 'successor'] = r.name
        for _, r in cdg[cdg.type == 'PRT'].iterrows():
            cdg.loc[cdg.loc[r.tu_id].successor, 'successor'] = i

        cdg['predecessor'] = [list() for _ in range(len(cdg))]
        for i, r in cdg.iterrows():
            if r.successor is not None:
                cdg.loc[r.successor]['predecessor'] += [i]
        cdg.loc[~cdg.predecessor.astype(bool), 'predecessor'] = None

        # We explicitly describe the part content of each node:
        cdg['content'] = cdg.apply(lambda x: tudf.loc[x.tu_id].iloc[0][x.type], axis=1)
        cdg['content_type'] = cdg.apply(
            lambda x: tuple([lib.parts.loc[p][0] for p in x.content]), axis=1
        )

        # And finally add information about the output of the whole graph:
        # by default outputs are all parts whose category is fluo_marker
        self.outputs = lib.parts[lib.parts['category'] == 'fluo_marker'].index.tolist()

        def containsOutput(l, outputs):
            for o in outputs:
                if o in l:
                    return True
            return False

        cdg['is_output'] = False
        cdg.loc[cdg.type == 'PRT', 'is_output'] = cdg.loc[cdg.type == 'PRT'].tu_id.apply(
            lambda x: containsOutput(tudf.loc[x].PRT.tolist()[0], self.outputs)
        )

        cdg['is_input'] = None

        self.cdg = cdg


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

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

def xp_series_from_json(jsonobj, lib):
    conn = sqlite3.connect(':memory:')
    create_db(conn)
    json_to_sql(jsonobj, conn, lib)
    xps = {t['name']: XP(conn, t['name'], lib) for t in jsonobj['tubes']}
    return xps

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     Archive     --
# ···············································································

# class Source:

# def __transcription_unit_from_L1(self, l1id, lib):
# l0_cols = ["insulator", "promoter", "5'UTR", "gene", "3'UTR", "terminator"]
# L0s = lib.L1s.loc[l1id][l0_cols].tolist()
# part_cols = [f'part_{i}' for i in range(1,7)]
# parts = []
# for l in L0s:
# parts += [p for p in lib.L0s.loc[l][part_cols].tolist() if p]
# tu = TranscriptionUnit([Part(p) for p in parts])
# tu.resolve_all_slots(lib)
# return tu

# def __init__(self, ratio, pid, lib):
# self.ratio = ratio
# self.pid = pid
# if self.pid in lib.L1s.index:
# self.level = 1
# self.transcription_units = [self.__transcription_unit_from_L1(self.pid, lib)]
# elif self.pid in lib.L2s.index:
# self.level = 2
# slot_cols=[f'slot_{i}' for i in range(1, 7)]
# l1ids = [s for s in lib.L2s.loc['pGW0010'][slot_cols].tolist() if s]
# self.transcription_units = [self.__transcription_unit_from_L1(l1id, lib) for l1id in l1ids]
# else:
# raise (ValueError(f'Unknown plasmid: {self.pid}'))

# def __repr__(self):
# return f'(ratio={self.ratio:.2f}, id={self.pid}), transcription units: {self.transcription_units}'


# class Aggregation:
# def __init__(self, agobj, lib):
# ratios = np.array([o['qtty'] for o in agobj])
# self.qtty = ratios.sum()
# ratios = ratios / self.qtty
# self.sources = [Source(r, o['plasmid'], lib) for (r, o) in zip(ratios, agobj)]

# def __repr__(self):
# return f'total qtty = {self.qtty}, sources = {self.sources}'


# class Run:
# def __init__(self, obj, lib):
# self.datafile = obj['datafile']
# self.name = obj['name']
# self.aggregations = [Aggregation(o, lib) for o in obj['content']['aggregations']]

# def __repr__(self):
# return f'{self.name}, agg = {self.aggregations}'

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

