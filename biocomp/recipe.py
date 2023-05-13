from .library import PartsLibrary as PartsLibrary
from . import utils as ut
from .network import Network, inverted_network
from pathlib import Path
import numpy as np
import jax
import pandas as pd
from tqdm import tqdm
import sqlite3
import json
import json5
from typing import Optional
import logging as log


def escape_name(name):
    return name.replace('-', '_').replace(' ', '_').upper().rstrip('_A')


def escape(names):
    if isinstance(names, str):
        return escape_name(names)
    if isinstance(names, list):
        return [escape_name(name) for name in names]
    if isinstance(names, tuple):
        return tuple([escape_name(name) for name in names])
    if isinstance(names, dict):
        return {escape_name(k): escape_name(v) for k, v in names.items()}
    else:
        return names


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                            --     sql     --
# ···············································································


def create_db(conn):
    sql = """
    CREATE TABLE IF NOT EXISTS `recipes` (
        name TEXT PRIMARY KEY,
        description TEXT,
        notes TEXT,
        extra TEXT);

    CREATE TABLE IF NOT EXISTS `aggregations` (
        id INTEGER PRIMARY KEY,
        recipe TEXT,
        notes TEXT,
        FOREIGN KEY (recipe) REFERENCES recipes(name) ON DELETE CASCADE);

    CREATE TABLE IF NOT EXISTS `sources` (
        name TEXT PRIMARY KEY,
        type TEXT);

    CREATE TABLE IF NOT EXISTS `TU_in_source`(
        source TEXT,
        TU INTEGER,
        position INTEGER,
        FOREIGN KEY(source) REFERENCES sources(name) ON DELETE CASCADE,
        PRIMARY KEY(source, TU));

    CREATE TABLE IF NOT EXISTS `source_in_aggregation`(
        aggregation INTEGER,
        source TEXT,
        ratio REAL,
        notes TEXT,
        extra TEXT,
        FOREIGN KEY (aggregation) REFERENCES aggregations(id) ON DELETE CASCADE,
        FOREIGN KEY (source) REFERENCES sources(name) ON DELETE CASCADE,
        PRIMARY KEY(aggregation, source));

    CREATE TABLE IF NOT EXISTS `XPs` (
        name TEXT PRIMARY KEY,
        flow_date TEXT,
        transfection_date TEXT,
        extra TEXT);

    CREATE TABLE IF NOT EXISTS `recipe_in_XP` (
        XP TEXT,
        recipe TEXT,
        sample_name TEXT,
        sample_notes TEXT,
        FOREIGN KEY (XP) REFERENCES XPs(name) ON DELETE CASCADE,
        FOREIGN KEY (recipe) REFERENCES recipes(name) ON DELETE CASCADE,
        PRIMARY KEY(XP, recipe));

    """
    c = conn.cursor()
    c.executescript(sql)


def recipes_to_sql(recipes: list, conn, lib):
    c = conn.cursor()
    c.execute("PRAGMA foreign_keys = ON;")
    create_db(conn)

    for obj in recipes:
        c.execute("SELECT name FROM recipes WHERE name = ?", (obj['name'],))
        if c.fetchone():
            # already in db so we skip
            log.info(f'Recipe {obj["name"]} already in db, skipping')
            return

        extra = {k: v for k, v in obj.items() if k not in ['name', 'description', 'notes']}
        extra_json = json.dumps(extra)
        c.execute(
            "INSERT INTO recipes VALUES (?, ?, ?, ?)",
            (
                obj['name'],
                obj['description'] if 'description' in obj else None,
                obj['notes'] if 'notes' in obj else None,
                extra_json if extra else None,
            ),
        )
        error_in_recipe = False
        for agg in obj['content']:
            ratios = []
            for s in agg['sources']:
                if 'ratio' in s:
                    ratios.append(s['ratio'])
                else:
                    ratios.append(1.0)
            ratios = np.array(ratios)
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
                    err_msg = f'In recipe {obj["name"]}: unknown plasmid {s["plasmid"]}'
                    ut.logger.error(err_msg)
                    error_in_recipe = True
                    continue  # we still continue to get a list of all errors
                c.execute("SELECT name FROM sources WHERE name = ?", (s['plasmid'],))
                if not c.fetchone():
                    c.execute("INSERT INTO sources VALUES (?, ?)", (s['plasmid'], type))
                    for i, l1id in enumerate(l1ids):
                        c.execute(
                            "INSERT INTO TU_in_source VALUES (?, ?, ?)", (s['plasmid'], l1id, i)
                        )
                # we put in "extra" everything other than ratio, plasmid and notes. Serialized to json.
                extra = {k: v for k, v in s.items() if k not in ['ratio', 'plasmid', 'notes']}
                extra_json = json.dumps(extra)
                c.execute(
                    "INSERT INTO source_in_aggregation VALUES (?, ?, ?, ?, ?)",
                    (
                        aggregation_id,
                        s['plasmid'],
                        r,
                        s['notes'] if 'notes' in s else None,
                        extra_json,
                    ),
                )
        if error_in_recipe:
            ut.logger.error(f'Skipped recipe {obj["name"]} because of import errors')
            c.execute("DELETE FROM recipes WHERE name = ?", (obj['name'],))
            conn.commit()
            # return
    conn.commit()


def xp_to_sql(xps: list, conn):
    c = conn.cursor()
    create_db(conn)
    for obj in xps:
        c.execute("SELECT name FROM XPs WHERE name = ?", (obj['name'],))
        if c.fetchone():
            # already in db so we skip
            ut.logger.debug(f'XP {obj["name"]} already in db, skipping')
            return
        ut.logger.info(f'Adding XP {obj["name"]} to sql db')
        c.execute(
            "INSERT INTO XPs VALUES (?, ?, ?, ?)",
            (
                obj['name'],
                obj['flow_date'] if 'flow_date' in obj else None,
                obj['transfection_date'] if 'transfection_date' in obj else None,
                json.dumps(
                    {
                        k: v
                        for k, v in obj.items()
                        if k not in ['name', 'flow_date', 'transfection_date']
                    }
                ),
            ),
        )
        for s in obj['samples']:
            c.execute(
                "INSERT INTO recipe_in_XP VALUES (?, ?, ?, ?)",
                (obj['name'], s['recipe'], s['name'], s['notes'] if 'notes' in s else None),
            )
    conn.commit()


def import_recipes_to_sql(recipe_files: list, conn, lib):
    # recipe files are json5 files
    recipes = []

    from tqdm import tqdm

    for f in tqdm(recipe_files, desc='Importing recipes'):
        try:
            recipe = ut.load_json5(f)
            ut.logger.debug(f'Importing recipe {recipe["name"]}')
            if not Path(f).name == f'{recipe["name"]}.recipe.json5':
                msg = f'Recipe name vs file name mismatch (declared name: {recipe["name"]})'
                raise RuntimeError(msg)
            recipes.append(recipe)
        except Exception as e:
            raise RuntimeError(f'Error loading recipe {f}: \n{e}')
    recipes_to_sql(recipes, conn, lib)


def network_from_recipe(recipe, lib, db_path=':memory:'):
    dbconn = sqlite3.connect(db_path)
    recipes_to_sql([recipe], dbconn, lib)
    assert recipe['name'] in [r[0] for r in dbconn.execute("SELECT name FROM recipes").fetchall()]
    n = Network(lib, recipe['name'], dbconn)
    return n


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     XP class     --
# ···············································································
from rich.progress import track


class XP:
    # Each sample in an Experiment implements one recipe, and resulted in one data file.
    # The XP object stores, along xp specific infos, all the recipes (in a sqlite database)
    # and data (in a dictionary of sample name -> pandas dataframe)
    # It also provides convenience functions to build the corrersponding networks.

    def __init__(
        self,
        xp_name,
        xp_path,
        recipe_path,
        lib,
        db_path=":memory:",
        inverse='shortest',
        data_path='./data/calibrated_data',
    ):
        log.debug(f'Initializing XP {xp_name}')
        self.xp_path, self.recipe_path = Path(xp_path), Path(recipe_path)
        self.samples: list  # [{name, recipe, notes}]
        self.name: str
        self.dbconn = sqlite3.connect(db_path)
        self.color_names: dict
        self.lib = lib

        # load the xp file
        self.xpfile = xp_path / xp_name / f"{xp_name}.xp.json5"
        with open(self.xpfile) as f:
            try:
                xpobj = json5.load(f)
                xp_to_sql([xpobj], self.dbconn)
                for k, v in xpobj.items():
                    if k == 'color_names':
                        self.color_names = {kk: escape(vv) for kk, vv in v.items()}
                    else:
                        setattr(self, k, v)
            except Exception as e:
                raise RuntimeError(f'Error loading xp file {self.xpfile}: \n{e}')

        if not Path(data_path).is_absolute():
            self.datapath = self.xp_path / self.name / data_path
        else:
            self.datapath = Path(data_path)

        if not self.datapath.exists():
            raise RuntimeError(f'Data path {self.datapath} does not exist')

        # load all the recipes inside
        self.recipe_names = [s['recipe'] for s in self.samples]
        unique_recipe_names = list(set(self.recipe_names))
        log.debug(f'Found {len(unique_recipe_names)} unique recipes')
        import_recipes_to_sql(
            [recipe_path / f"{r}.recipe.json5" for r in unique_recipe_names], self.dbconn, lib
        )
        self.load_raw_data()

    def load_raw_data(self):
        """Load the raw data for each sample in the xp"""
        datafiles = [self.datapath / f"{s['name']}.{self.name}.csv" for s in self.samples]
        df_data: dict[str, pd.DataFrame] = {}
        for s, f in tqdm(
            list(zip(self.samples, datafiles)), desc=f"loading data files for {self.name}"
        ):
            content = pd.read_csv(f, engine="pyarrow")
            assert isinstance(content, pd.DataFrame)  # otherwise type hints won't match
            df_data[s['name']] = content
        self.raw_data = df_data

    def build_networks(self, inverse='shortest'):
        """Build the networks for each sample in the xp,
        returns two lists: (networks, sample names)"""

        # first build each network for each recipe
        unique_recipe_names = list(set(self.recipe_names))
        fwd_networks = {}
        for recipename in tqdm(unique_recipe_names, desc=f'Building networks for xp {self.name}'):
            try:
                log.debug(f'building recipe {recipename}')
                fwd_networks[recipename] = Network(
                    self.lib, recipename, self.dbconn, metadata={'from_xp': self.name}
                )
            except Exception as e:
                raise RuntimeError(
                    f'Error building network for recipe {recipename} in xp {self.name}: \n{e}'
                )

        # now go through the samples and create the correct pairs
        networks = []
        for s in self.samples:
            if not inverse:
                networks.append((fwd_networks[s['recipe']], s['name']))
            else:
                inv_nets = inverted_network(fwd_networks[s['recipe']], mode=inverse)
                for n in inv_nets:
                    networks.append((n, s['name']))

        return tuple(zip(*networks))

    def get_Y(self, networks, sample_names):
        assert self.raw_data is not None
        # we want to reorder data columns to match the network's output
        out_prots = [net.get_output_proteins() for net in networks]
        out_channels = [[self.color_names[k] for k in out_prot] for out_prot in out_prots]
        Y = [
            np.array(self.raw_data[sample][out_chan])
            for sample, out_chan in zip(sample_names, out_channels)
        ]
        return Y

    def get_XY(self, networks, sample_names):
        Y = self.get_Y(networks, sample_names)
        X = [net.get_input_from_output(y) for net, y in zip(networks, Y)]
        return X, Y

    def __str__(self):
        # add borders:
        res = '-' * 18 + f'  XP {self.name}  ' + '-' * 18 + '\n'
        for k, v in self.__dict__.items():
            if isinstance(v, dict):
                res += f"* {k}:\n"
                for kk, vv in v.items():
                    res += f"    {kk}: {vv}\n"
            elif isinstance(v, list):
                res += f"* {k}:\n"
                for vv in v:
                    res += f"    {vv}\n"
            else:
                res += f"* {k}: {v}\n"
        return res

    def __repr__(self):
        return self.__str__()


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                           --     tests     --
# ···············································································
def test_module():
    libpath = "./test_data/all_sheets.pickle"
    l = ut.load(libpath)
    lib = PartsLibrary(
        l.parts, l.L0s, l.L1s, l.L2s, l.categories, l.sequestrons, l.sequestron_types
    )
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
