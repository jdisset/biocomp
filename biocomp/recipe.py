from .library import PartsLibrary as PartsLibrary
from . import utils as ut
from .network import Network, inverted_network
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import sqlite3
import hashlib
import json
import json5
from typing import Optional
import logging as log
import traceback


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
    error_log = ''

    for obj in recipes:
        c.execute("SELECT name FROM recipes WHERE name = ?", (obj['name'],))
        if c.fetchone():
            # already in db so we skip
            log.info(f'Recipe {obj["name"]} already in db, skipping')
            return error_log

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
            for r, s in zip(ratios, agg['sources']):
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
                    error_log += err_msg + '\n\n'
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

    conn.commit()
    return error_log


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


def import_recipes_to_sql(recipe_files: list, conn, lib, ignore_errors=False, show_progress=True):
    # recipe files are json5 files
    recipes = []

    error_log = ''

    for f in tqdm(recipe_files, desc='Importing recipes', disable=not show_progress):
        recipe = ut.load_json5(f)
        ut.logger.debug(f'Importing recipe {recipe["name"]}')
        if not Path(f).name == f'{recipe["name"]}.recipe.json5':
            msg = f'File vs recipe name mismatch (recipe: {recipe["name"]}, file: {f})'
            if ignore_errors:
                ut.logger.warning(msg)
                error_log += msg + '\n\n'
            else:
                raise RuntimeError(msg)
        recipes.append(recipe)
    error_log += recipes_to_sql(recipes, conn, lib)
    return error_log


def network_from_recipe(recipe, lib, db_path=':memory:'):
    dbconn = sqlite3.connect(db_path)
    recipes_to_sql([recipe], dbconn, lib)
    assert recipe['name'] in [r[0] for r in dbconn.execute("SELECT name FROM recipes").fetchall()]
    n = Network.from_db(lib, recipe['name'], dbconn)
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

    def resolve_paths_with_priorities(
        paths, file_names, extension='.recipe.json5', throw_error=True, base_path=None
    ):
        """Given a list of base paths ordered by priority, and a list of file names, returns a list of paths
        where each file name is found in the first priority path where it exists"""
        if isinstance(file_names, str):
            file_names = [file_names]
        if not isinstance(paths, list):
            paths = [paths]
        if base_path is None:
            base_path = Path.cwd()
        else:
            base_path = Path(base_path)
        absolute_paths = [Path(p) if Path(p).is_absolute() else base_path / p for p in paths]
        absolute_paths = [p for p in absolute_paths if p.exists()]
        resolved_paths = []
        for f in file_names:
            filename = f + extension
            found = False
            for p in absolute_paths:
                if (p / filename).exists():
                    resolved_paths.append(p / filename)
                    found = True
                    break
            if not found:
                if throw_error:
                    raise RuntimeError(
                        f'Could not find file {filename} in any of the paths {absolute_paths}'
                    )
                else:
                    resolved_paths.append(None)
        return resolved_paths

    def __init__(
        self,
        xp_name,
        xp_path,
        recipe_path,
        lib,
        db_path=None,
        data_path='./data/calibrated_data',
        load_data=True,
        ignore_errors=False,
        show_progress=True,
    ):
        """
        Reads the xp file, and loads both the recipes (into an instance-level sqlite db)
        and the raw data (into a dict of pandas dataframes)
        Important notions:
        - A recipe is the description of a given transfection, and is implemented by a network
        - A sample is the data resulting from a given recipe
        - The XP stores the content of recipes in a sqlite db
        - The XP doesn't store the networks, but provides functions to build them given a recipe_name
        - The XP stores the raw data in a dict of pandas dataframes, indexed by sample name
        """
        log.debug(f'Initializing XP {xp_name}')

        self.lib = lib
        self.recipe_loading_errors = ''
        self.data_loading_errors = ''
        self.network_building_errors = ''
        self.xp_path = Path(xp_path)
        self.show_progress = show_progress
        self.raw_data: dict[str, pd.DataFrame] = {}

        self.db_uri = False
        if db_path is None:
            db_path = f'file:{xp_name}.db?mode=memory'
            self.db_uri = True
        self.db_path = db_path
        self.dbconn = sqlite3.connect(self.db_path, uri=self.db_uri)

        self.xpfile = xp_path / xp_name / f"{xp_name}.xp.json5"
        try:
            self.load_xp_file()
        except Exception as e:
            msg = f'Error loading xp file {self.xpfile}: \n{e}'
            raise RuntimeError(msg) from e

        self.load_recipes(recipe_path, ignore_errors=ignore_errors)
        self.load_samples(data_path, load_data=load_data)

        if self.recipe_loading_errors != '' and not ignore_errors:
            raise RuntimeError(self.recipe_loading_errors)
        if self.data_loading_errors != '' and not ignore_errors:
            raise RuntimeError(self.data_loading_errors)

    def load_xp_file(self):
        required_keys = ['name', 'flow_date', 'transfection_date', 'samples']
        with open(self.xpfile) as f:
            xpobj = json5.load(f)
            xp_to_sql([xpobj], self.dbconn)

            for k in required_keys:
                if k not in xpobj:
                    raise RuntimeError(f'Key {k} not found in xp file {self.xpfile}')

            self.name = xpobj['name']
            self.flow_date = xpobj['flow_date']
            self.transfection_date = xpobj['transfection_date']
            self.samples = {
                s['name']: {k: v for k, v in s.items() if k != 'name'} for s in xpobj['samples']
            }

            # TODO: remove this old stupid color_names thing
            self.color_names = {}
            if 'color_names' in xpobj:
                self.color_names = {kk: escape(vv) for kk, vv in xpobj['color_names'].items()}

            self.extra = {}
            for k, v in xpobj.items():
                if k not in required_keys:
                    self.extra[k] = v

    def load_recipes(self, recipe_path, ignore_errors):
        unique_recipe_names = list(set([s['recipe'] for s in self.samples.values()]))
        log.debug(f'Found {len(unique_recipe_names)} unique recipes')
        recipe_files = XP.resolve_paths_with_priorities(
            recipe_path,
            unique_recipe_names,
            extension='.recipe.json5',
            base_path=self.xp_path / self.name,
            throw_error=False,
        )
        for r, f in zip(unique_recipe_names, recipe_files):
            if f is None:
                self.recipe_loading_errors += f'Could not find recipe file for recipe {r}\n'
        # filter out the recipes that were not found
        recipe_files = [f for f in recipe_files if f is not None]
        unique_recipe_names = [
            n for n, f in zip(unique_recipe_names, recipe_files) if f is not None
        ]
        self.recipes = {n: f for n, f in zip(unique_recipe_names, recipe_files)}
        self.recipe_loading_errors += import_recipes_to_sql(
            recipe_files,
            self.dbconn,
            self.lib,
            ignore_errors=ignore_errors,
            show_progress=self.show_progress,
        )

    def load_samples(self, data_path, load_data=True, ignore_errors=False):
        sample_names = list(self.samples.keys())
        data_files = XP.resolve_paths_with_priorities(
            data_path,
            sample_names,
            extension=f'.{self.name}.csv',
            base_path=self.xp_path / self.name,
            throw_error=False,
        )

        name_to_file = {n: f for n, f in zip(sample_names, data_files) if f is not None}
        for n, s in self.samples.items():
            if name_to_file.get(n, None) is None:
                self.data_loading_errors += f'Could not find data file for sample {n}\n'
            s['data_file'] = name_to_file.get(n, None)

        if load_data:
            self.load_all_raw_data(ignore_errors=ignore_errors)

    def load_raw_data(self, sample_name, proteins=None, ignore_errors=False, force_reload=False):
        """Load the raw data for a given sample in the xp, and store it in a pandas dataframe"""

        # when ignore_errors is true, we return None when there's an error and append the error message to self.data_loading_errors
        def error_handler(msg):
            if ignore_errors:
                self.data_loading_errors += msg + '\n\n'
                ut.logger.warning(msg)
                return None
            else:
                raise RuntimeError(msg)

        if sample_name not in self.samples:
            return error_handler(
                f'Sample {sample_name} not found in xp {self.name}. Available: {self.samples.keys()}'
            )

        s = self.samples[sample_name]
        if 'data_file' not in s:
            return error_handler(f'No data file listed for sample {sample_name} in xp {self.name}')

        if s['data_file'] is None:
            return error_handler(f'No data file listed for sample {sample_name} in xp {self.name}')

        assert s['data_file'] is not None

        f = Path(s['data_file'])
        if not f.exists():
            return error_handler(f'Data file {f} not found for sample {sample_name} in xp {self.name}')

        if sample_name not in self.raw_data or force_reload:
            content = pd.read_csv(f, engine="pyarrow")
            assert isinstance(content, pd.DataFrame)
            self.raw_data[sample_name] = content

        data = self.raw_data[sample_name]

        available_columns = set(data.columns)
        if proteins is None:
            return data
        else:
            remainder = set(proteins) - available_columns
            if len(remainder) > 0:
                return error_handler(
                    f'Proteins {remainder} not found in data for sample {sample_name}. Available: {available_columns}'
                )
            return np.asarray(data[proteins])

    def load_all_raw_data(self, ignore_errors=False, force_reload=True):
        """Load the raw data for each sample in the xp, and store it in a dict [sample name] -> pandas dataframe"""
        self.raw_data: dict[str, pd.DataFrame] = {}
        for s_name, s in tqdm(
            list(self.samples.items()),
            desc=f"loading data files for {self.name}",
            disable=not self.show_progress,
        ):
            self.load_raw_data(s_name, ignore_errors=ignore_errors, force_reload=force_reload)

    def build_network(
        self, recipe_name, inverse='shortest', use_db=None, ignore_errors=False, use_cache=None
    ):
        if str(recipe_name) not in self.recipes:
            raise RuntimeError(
                f'Recipe {recipe_name} not found in xp {self.name}. Cannot build network.'
            )

        dbconn = use_db or self.dbconn

        try:
            ut.logger.debug(f'building network for recipe {recipe_name}')
            fwd_network = Network.from_db(
                self.lib,
                recipe_name,
                dbconn,
                metadata={
                    'from_xp': self.name,
                    'recipe_name': recipe_name,
                    'recipe_file': self.recipes[recipe_name],
                },
                use_cache=use_cache,
            )
        except Exception as e:
            msg = f'Error building network for recipe {recipe_name} in xp {self.name}: \n{e}'
            if ignore_errors:
                # trace = traceback.format_exc()
                # msg += f'\n\n{trace}\n'
                ut.logger.warning(msg)
                self.network_building_errors += msg + '\n\n'
                return None
            else:
                raise RuntimeError(msg) from e

        if not inverse:
            return fwd_network
        else:
            return [n for n in inverted_network(fwd_network, mode=inverse, use_cache=use_cache)]

    def build_networks(
        self,
        inverse='shortest',
        use_db=None,
        ignore_errors=False,
        use_cache=None,
        progress_callback=None,
    ):
        """Build the networks for each sample in the xp,
        returns two lists: (networks, sample names)
        although several networks could in theory share the same sample name
        here we return a list of pairs (network, sample name) to avoid confusion
        """
        self.network_building_errors = ''

        built_networks = {}
        networks = []
        sample_names = []
        for s_name, s in tqdm(
            list(self.samples.items()),
            desc=f"building networks for {self.name}",
            disable=not self.show_progress,
        ):
            recipe_name = s['recipe']
            if recipe_name not in self.recipes:
                if progress_callback is not None:
                    progress_callback(1)
                continue
            if recipe_name not in built_networks:
                built_networks[recipe_name] = self.build_network(
                    recipe_name,
                    inverse=inverse,
                    use_db=use_db,
                    ignore_errors=ignore_errors,
                    use_cache=use_cache,
                )

            nets = built_networks[recipe_name]
            if progress_callback is not None:
                progress_callback(1)
            if nets is None:
                continue
            if isinstance(nets, Network):
                nets = [nets]
            networks += nets
            sample_names += [s_name] * len(nets)
        return networks, sample_names

    def get_Y(self, networks: list[Network], sample_names: list[str], ignore_errors=False):
        """Returns the output data (including cotx markers) for each network and sample"""
        assert len(networks) == len(sample_names)
        # we want to reorder data columns to match the network's output
        output_proteins = [net.get_output_proteins() if net is not None else None for net in networks]
        # if we have a color_names attribute, we use it to alias the protein names
        if hasattr(self, 'color_names'):
            output_proteins = [[self.color_names.get(p, p) for p in prots] for prots in output_proteins]
        return [
            self.load_raw_data(s_name, proteins=p_names, ignore_errors=ignore_errors)
            for s_name, p_names in zip(sample_names, output_proteins)
        ]

    def get_XY(self, networks: list[Network], sample_names: list[str], ignore_errors=False):
        """Returns the X and Y data (the independent and dependent variables) for each network and sample"""
        Y = self.get_Y(networks, sample_names, ignore_errors=ignore_errors)
        X = [net.get_input_from_output(y) for net, y in zip(networks, Y)]
        return X, Y

    def __str__(self):
        # add borders:
        res = '-' * 18 + f'  XP {self.name}  ' + '-' * 18 + '\n'
        for k, v in self.__dict__.items():
            if k == 'dbconn':
                continue
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

    def __hash__(self):
        return hash(self.__str__())


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
