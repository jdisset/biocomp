from .library import PartsLibrary as PartsLibrary
from . import utils as ut
import numpy as np
import sqlite3
import json


def create_db(conn):
    sql = """
    CREATE TABLE IF NOT EXISTS `recipes` (
        name TEXT PRIMARY KEY,
        description TEXT,
        notes TEXT);

    CREATE TABLE IF NOT EXISTS `aggregations` (
        id INTEGER PRIMARY KEY,
        recipe TEXT,
        notes TEXT,
        FOREIGN KEY (recipe) REFERENCES recipes(name));

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
        notes TEXT,
        extra TEXT,
        FOREIGN KEY (aggregation) REFERENCES aggregations(id),
        FOREIGN KEY (source) REFERENCES sources(name),
        PRIMARY KEY(aggregation, source));
    """
    c = conn.cursor()
    c.executescript(sql)


def __to_sql(obj, conn, lib):
    c = conn.cursor()
    create_db(conn)
    c.execute("SELECT name FROM recipes WHERE name = ?", (obj['name'],))
    if c.fetchone():
        raise RuntimeError(f'Error while importing recipe {obj["name"]}: already in the database')

    c.execute(
        "INSERT INTO recipes VALUES (?, ?, ?)",
        (
            obj['name'],
            obj['description'] if 'description' in obj else None,
            obj['notes'] if 'notes' in obj else None,
        ),
    )
    for agg in obj['content']:
        ratios = np.array([s['ratio'] for s in agg['sources']])
        qtty = float(np.sum(ratios))
        c.execute(
            "INSERT INTO aggregations VALUES (?, ?, ?)",
            (None, obj['name'], agg['notes'] if 'notes' in agg else None),
        )
        aggregation_id = c.lastrowid
        ratios = ratios / qtty
        for (r, s) in zip(ratios, agg['sources']):
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
                raise RuntimeError(
                    f'Error while importing recipe {obj["name"]}: unknown plasmid {s["plasmid"]}'
                )
            c.execute("SELECT name FROM sources WHERE name = ?", (s['plasmid'],))
            if not c.fetchone():
                c.execute("INSERT INTO sources VALUES (?, ?)", (s['plasmid'], type))
                for i, l1id in enumerate(l1ids):
                    c.execute("INSERT INTO TU_in_source VALUES (?, ?, ?)", (s['plasmid'], l1id, i))
            # we put in "extra" everything other than ratio, plasmid and notes. Serialized to json.
            extra = {k: v for k, v in s.items() if k not in ['ratio', 'plasmid', 'notes']}
            extra_json = json.dumps(extra)
            c.execute(
                "INSERT INTO source_in_aggregation VALUES (?, ?, ?, ?, ?)",
                (aggregation_id, s['plasmid'], r, s['notes'] if 'notes' in s else None, extra_json),
            )
    conn.commit()


def import_recipes_to_sql(recipe_files: list, conn, lib):
    # recipe files are json5 files
    for f in recipe_files:
        xpdict = ut.load_json5(f)
        __to_sql(xpdict, conn, lib)


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                           --     tests     --
# ···············································································
def test_module():
    libpath = "./test_data/all_sheets.pickle"
    l = ut.load(libpath)
    lib = PartsLibrary(l.parts, l.L0s, l.L1s, l.L2s, l.categories, l.sequestrons, l.sequestron_types)
    recipe_path = "./test_data/recipe00.json5"

    conn = sqlite3.connect(":memory:")
    create_db(conn)


    def test_import_recipes_to_sql():
        import_recipes_to_sql([recipe_path], conn, lib)
        c = conn.cursor()
        c.execute("SELECT * FROM recipes")
        # only one recipe
        assert len(c.fetchall()) == 1
        c.execute("SELECT * FROM aggregations")
        # 2 aggregations
        assert len(c.fetchall()) == 2


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
