# %load_ext autoreload
# %autoreload 2

import pandas as pd
import numpy as np

import scriptutils as ut
import biocomp.utils as bu
from functools import partial
import biocomp as bc
import json
from rich import print

l = ut.load("all_sheets.pickle")
lib = bc.PartsLibrary(l.parts, l.L0s, l.L1s, l.L2s, l.categories, l.sequestrons, l.sequestron_types)


xp = json.load(open("example_xpfile.json"))

# remarks for the wet team
# - validate json
# - no caps for any field name (all lowercase, camel case)
# - no named aggregation, just a list of lists in a field named "aggregations"
# - can we make tubes XP instead? Tubes are a very bio-tied concept. From the software side, it makes sense to call a tube an XP, since it's the biggest independant unit. I get that a collection of XP can be related IRL, but maybe there's a better word? Or if we keep XP as a collection of "tubes", maybe there's a better word than tube that makes sense in both the soft and the wet worlds! Could a tube be a "run" or an "xp"? An XP an "assay"? I don't think these terms are perfect..

tubes = xp['tubes']
print(tubes)
print(tubes[2])

# TODO:
# [ ] build central dogma from XP
# [ ] build compute graph
# [ ] add aggregations to the compute graph


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

# create sqlite database in memory
import sqlite3
# delete file if exists
import os
if os.path.exists("test.db"):
    os.remove("test.db")
conn = sqlite3.connect('test.db')
c = conn.cursor()
c.executescript(sql)
conn.close()

TranscriptionUnits = {}

def xp_to_db(xp, dbname):
    conn = sqlite3.connect(dbname)
    c = conn.cursor()
    tubes = xp['tubes']
    for t in tubes:
        c.execute("INSERT INTO tubes VALUES (?, ?)", (t['name'],t['comment']))
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
                    slot_cols=[f'slot_{i}' for i in range(1, 7)]
                    l1ids = [s for s in lib.L2s.loc[s['plasmid']][slot_cols].tolist() if s]
                if type is None:
                    raise Exception("Unknown plasmid")
                c.execute("SELECT name FROM sources WHERE name = ?", (s['plasmid'],))
                if not c.fetchone():
                    c.execute("INSERT INTO sources VALUES (?, ?)", (s['plasmid'], type))
                    for i, l1id in enumerate(l1ids):
                        c.execute("INSERT INTO TU_in_source VALUES (?, ?, ?)", (s['plasmid'], l1id, i))
                c.execute("INSERT INTO source_in_aggregation VALUES (?, ?, ?)", (aggregation_id, s['plasmid'], r))
    conn.commit()
    conn.close()

xp_to_db(xp, "test.db")

# create sqlite database in memory
