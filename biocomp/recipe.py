from .library import PartsLibrary as PartsLibrary
from . import utils as ut
from .network import Network, inverted_network
from .compute import ComputeGraphModel
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import sqlite3
import json
import json5
from typing import Optional

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


def __recipe_to_sql(obj, conn, lib):
    c = conn.cursor()
    create_db(conn)
    c.execute("SELECT name FROM recipes WHERE name = ?", (obj['name'],))
    if c.fetchone():
        # already in db so we skip
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
        try:
            recipe = ut.load_json5(f)
            if not Path(f).name == f'{recipe["name"]}.recipe.json5':
                msg = f'Recipe name vs file name mismatch (declared name: {recipe["name"]})'
                raise RuntimeError(msg)
            __recipe_to_sql(recipe, conn, lib)
        except Exception as e:
            raise RuntimeError(f'Error loading recipe {f}: \n{e}')


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     XP class     --
# ···············································································


class XP:
    # Each sample in an Experiment implements one recipe, and resulted in one data file.
    # The XP object stores all the recipes (in a sqlite database), and the corrsponding
    # networks and inverted networks.

    def __init__(self, xp_name, xp_path, recipe_path, lib, db_path=":memory:", inverse=True):
        self.xp_path, self.recipe_path = Path(xp_path), Path(recipe_path)
        self.samples: list  # [{name, recipe, notes}]
        self.name: str
        self.networks: dict[str, Network]  # {sample_name: Network}
        self.inv_networks: Optional[dict[str, Network]]  # {sample_name: Network}
        self.dbconn = None
        self.color_names: dict

        self.xpfile = xp_path / xp_name / f"{xp_name}.xp.json5"
        with open(self.xpfile) as f:
            xpobj = json5.load(f)
            for k, v in xpobj.items():
                setattr(self, k, v)

        self.recipe_names = [s['recipe'] for s in self.samples]
        unique_recipe_names = list(set(self.recipe_names))
        dbconn = sqlite3.connect(db_path)
        import_recipes_to_sql(
            [recipe_path / f"{r}.recipe.json5" for r in unique_recipe_names], dbconn, lib
        )
        self.networks = {
            recipename: Network(lib, recipename, dbconn) for recipename in unique_recipe_names
        }
        if inverse:
            self.inv_networks = {k: inverted_network(v) for k, v in self.networks.items()}
        else:
            self.inv_networks = None

    def get_models(self, inverse=True) -> dict[str, ComputeGraphModel]:
        if inverse:
            assert self.inv_networks is not None
        nets = self.inv_networks if inverse else self.networks
        assert nets
        models = {s['name']: ComputeGraphModel(nets[s['recipe']]) for s in self.samples}
        for s, m in models.items():
            try:
                m.build()
            except Exception as e:
                msg = f'Error building {"inverse" if inverse else ""} model for sample {s}: {e}'
                raise RuntimeError(msg)
        return models

    def get_raw_data(self) -> dict[str, pd.DataFrame]:
        datafiles = [
            self.xp_path / self.name / 'data' / f"{s['name']}.{self.name}.csv" for s in self.samples
        ]
        df_data: dict[str, pd.DataFrame] = {}
        for s, f in tqdm(list(zip(self.samples, datafiles)), f"loading data files for {self.name}"):
            content = pd.read_csv(f)
            assert isinstance(content, pd.DataFrame)  # otherwise type hints won't match
            df_data[s['name']] = content
        return df_data

    def get_XY(self, model_dict) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Returns a dict of {sample_name: (X, Y)} where Y is the reordered data (so that it matches the model's output)"""
        assert self.inv_networks is not None
        # we want to reorder data columns to match the model's output
        df_data = self.get_raw_data()
        out_prots = {sample: model.get_output_proteins() for sample, model in model_dict.items()}
        out_channels = {
            sample: [self.color_names[k] for k in out_prot]
            for sample, out_prot in out_prots.items()
        }
        Y = {
            sample: np.array(df_data[sample][out_channels[sample]]) for sample in model_dict.keys()
        }
        X = {
            sample: model_dict[sample].get_input_from_output(Y[sample])
            for sample in model_dict.keys()
        }
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
