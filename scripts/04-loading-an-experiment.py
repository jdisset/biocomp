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
xp = json.load(open("example_xpfile.json"))
xp['tubes']


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

# TODO:
# [x] convert xp to (temp) sql db
# [x] build central dogma from db
# [x] build compute graph
# [x] add content to graph edges
# [ ] add aggregations to the compute graph

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     xp to db     --
# ···············································································
# ACTUALLY THE WHOLE OBJECT COMPOSITION THINGY IS NO BUENO
# Structures from XP down to Sources are just there to hold data, there's no real logic there.
# TranscriptionUnit is the first structure that requires some smarts (with the whole slot resolution thing), and it's
# also the level at which central dogma graph will decompose things.
# Since when building the graphs (central dogma and compute), we'd much rather have a flat data structure,
# it is probably better to store all TranscriptionUnit in a dictionnary (or a dataframe or whatever)
# and then just use a light set of relational tables to express how TUs are structured into Aggregations and Sources

# SO, TODO: turn XP, Aggregation and Source into tables

##

# into sqlite
def create_db(name, remove_existing=False):
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
    if remove_existing and os.path.exists(name):
        os.remove(name)
    conn = sqlite3.connect(name)
    c = conn.cursor()
    c.executescript(sql)
    return conn


TranscriptionUnits = {}


def transcription_unit_from_L1(l1id, lib):
    l0_cols = ["insulator", "promoter", "5'UTR", "gene", "3'UTR", "terminator"]
    L0s = lib.L1s.loc[l1id][l0_cols].tolist()
    part_cols = [f'part_{i}' for i in range(1, 7)]
    parts = []
    for l in L0s:
        parts += [p for p in lib.L0s.loc[l][part_cols].tolist() if p]
    tu = bc.TranscriptionUnit([bc.Part(p) for p in parts])
    tu.resolve_all_slots(lib)
    return tu


def xp_to_db(xp, conn):
    c = conn.cursor()
    tubes = xp['tubes']
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
                        TranscriptionUnits[l1id] = transcription_unit_from_L1(l1id, lib)

                c.execute(
                    "INSERT INTO source_in_aggregation VALUES (?, ?, ?)",
                    (aggregation_id, s['plasmid'], r),
                )
    conn.commit()


conn = create_db(':memory:')
xp_to_db(xp, conn)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

# tubes:[#name, comment]
# aggregations:[#id, qtty, tube*]
# sources:[#name, type];
# TU_in_source:[#source*,#TU*,position]
# source_in_aggregation:[#aggregation*,#source*,ratio]


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                    --     central dogma build     --
#···············································································

# these 3 should be TranscriptionUnit methods
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

def build_central_dogma_graph(tuids, outputs = list()):
    l1 = [
        {
            'DNA': getDna(TranscriptionUnits[t]),
            'RNA': getRna(TranscriptionUnits[t], lib),
            'PRT': getPrt(TranscriptionUnits[t], lib),
        }
        for t in tuids
    ]

    l1df = pd.DataFrame(l1)

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
    cdg = pd.concat([dna_df, rna_df, prt_df]).reset_index(drop=True)

    # Add successor and predecessor information:
    for i, r in cdg[cdg.type == 'RNA'].iterrows():
        cdg.loc[r.l1_id, 'successor'] = i
    for i, r in cdg[cdg.type == 'PRT'].iterrows():
        cdg.loc[cdg.loc[r.l1_id].successor, 'successor'] = i

    cdg['predecessor'] = [list() for _ in range(len(cdg))]
    for i, r in cdg.iterrows():
        if r.successor is not None:
            cdg.loc[r.successor]['predecessor'] += [i]
    cdg.loc[~cdg.predecessor.astype(bool), 'predecessor'] = None

    # We explicitly describe the part content of each node:
    cdg['content'] = cdg.apply(lambda x: l1df.loc[x.l1_id].iloc[0][x.type], axis=1)
    cdg['content_type'] = cdg.apply(lambda x: tuple([lib.parts.loc[p][0] for p in x.content]), axis=1)

    # And finally add information about the output of the whole graph:
    # by default outputs are all parts whose category is fluo_marker
    outputs = outputs + lib.parts[lib.parts['category'] == 'fluo_marker'].index.tolist()

    def containsOutput(l, outputs):
        for o in outputs:
            if o in l:
                return True
        return False

    cdg['is_output'] = False
    cdg.loc[cdg.type == 'PRT', 'is_output'] = cdg.loc[cdg.type == 'PRT'].l1_id.apply(
        lambda x: containsOutput(l1df.loc[x].PRT.tolist()[0], outputs)
    )

    cdg['is_input'] = None
    # for k, v in inputDict.items():
        # cdg.loc[k, 'is_input'] = v

    return cdg

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                --     display all compute graphs     --
#···············································································
c = conn.cursor()
c.execute(
    """SELECT a.tube, tis.TU FROM TU_in_source tis, source_in_aggregation sia, aggregations a
    WHERE tis.source = sia.source AND sia.aggregation = a.id""")
tubedict = {}
for row in c:
    if row[0] not in tubedict:
        tubedict[row[0]] = [row[1]]
    else:
        tubedict[row[0]].append(row[1])


for tubename, tuids in tubedict.items():
    cdg = build_central_dogma_graph(tuids)
    compg = bc.buildComputeGraph(lib, cdg)
    ut.h2(f'Tube {tubename}')
    ut.drawComputeGraph(compg, cdg=cdg, key=f'{tubename}comp')

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


