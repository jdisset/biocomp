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
from typing import Union, Optional, Sequence, Iterable, Any, Callable, TypeVar

PathLike = Union[str, Path]


def escape_name(name):
    return name.replace('-', '_').replace(' ', '_').upper()


def escape(names):
    if isinstance(names, str):
        return escape_name(names)
    if isinstance(names, list):
        return [escape_name(name) for name in names]
    if isinstance(names, tuple):
        return tuple([escape_name(name) for name in names])
    if isinstance(names, dict):
        return {escape_name(k): escape_name(v) for k, v in names.items()}
    if isinstance(names, set):
        return {escape_name(name) for name in names}
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


def recipes_to_sql(recipes: list, conn, lib, error_handler=None):
    if error_handler is None:

        def _handler(msg):
            raise RuntimeError(msg)

        error_handler = _handler

    c = conn.cursor()
    c.execute("PRAGMA foreign_keys = ON;")
    create_db(conn)

    for obj in recipes:
        c.execute("SELECT name FROM recipes WHERE name = ?", (obj['name'],))
        if c.fetchone():
            # already in db so we skip
            log.info(f'Recipe {obj["name"]} already in db, skipping')

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
                    error_handler(err_msg)
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


def import_recipes_to_sql(
    recipe_files: list[PathLike] | PathLike,
    conn: sqlite3.Connection,
    lib: PartsLibrary,
    error_handler=None,
    show_progress=True,
) -> list:
    if error_handler is None:

        def _handler(msg):
            raise RuntimeError(msg)

        error_handler = _handler

    # recipe files are json5 files
    recipe_objects = []
    for f in tqdm(recipe_files, desc='Importing recipes', disable=not show_progress):
        recipe = ut.load_json5(f)
        ut.logger.debug(f'Importing recipe {recipe["name"]}')
        if not Path(f).name == f'{recipe["name"]}.recipe.json5':
            error_handler(f'File vs recipe name mismatch (recipe: {recipe["name"]}, file: {f})')
        recipe_objects.append(recipe)
    recipes_to_sql(recipe_objects, conn, lib)
    return recipe_objects


def build_network(
    recipe_name,
    dbconn,
    lib,
    inverse='shortest',
    metadata=None,
    error_handler=None,
    use_cache=None,
):
    if error_handler is None:

        def _handler(msg):
            raise RuntimeError(msg)

        error_handler = _handler

    if metadata is None:
        metadata = {'recipe_name': recipe_name}

    try:
        ut.logger.debug(f'Building network for recipe {recipe_name}')
        fwd_network = Network.from_db(
            lib,
            recipe_name,
            dbconn,
            metadata=metadata,
            use_cache=use_cache,
        )

    except Exception as e:
        error_handler(f'Can\'t build network: {e}')

    if not inverse:
        return fwd_network
    else:
        return [n for n in inverted_network(fwd_network, mode=inverse, use_cache=use_cache)]


def network_from_recipe(
    recipe_path: PathLike, lib: PartsLibrary, db_path=':memory:', metadata=None, **kwargs
):
    dbconn = sqlite3.connect(db_path)
    recipe = import_recipes_to_sql([recipe_path], dbconn, lib)[0]
    if metadata is None:
        metadata = {'recipe_name': recipe['name'], 'recipe_file': recipe_path}
    return build_network(recipe['name'], dbconn, lib, metadata=metadata, **kwargs)


def load_data_file(
    data_file_path: PathLike,
    proteins: Optional[list[str]] = None,
    error_handler: Optional[Callable] = None,
    use_store=None,
    force_reload=False,
):
    if error_handler is None:

        def _handler(msg):
            raise RuntimeError(msg)

        error_handler = _handler

    if use_store is None:
        use_store = {}

    if data_file_path is None:
        return error_handler(f'Data file is null.')

    f = Path(data_file_path)
    if not f.exists():
        return error_handler(f'Data file {f} not found')

    if data_file_path not in use_store or force_reload:
        content = pd.read_csv(f, engine="pyarrow")
        assert isinstance(content, pd.DataFrame)
        use_store[data_file_path] = content

    data = use_store[data_file_path]

    available_columns = set(data.columns)
    if proteins is None:
        return data.to_numpy()
    else:
        remainder = set(proteins) - available_columns
        if len(remainder) > 0:
            return error_handler(
                f'Proteins {remainder} not found in data. Available: {available_columns}'
            )

        return np.asarray(data[proteins])


def get_network_data(
    network: Network,
    data_file_path: PathLike,
    color_aliases: Optional[dict[str, str]] = None,
    **kwargs,
) -> Optional[np.ndarray]:
    # we want to reorder data columns to match the network's output
    out_proteins = escape(network.get_output_proteins())
    if color_aliases is not None:
        aliases = escape(color_aliases)
        out_proteins = [aliases.get(p, p) for p in out_proteins]
    return load_data_file(data_file_path, proteins=out_proteins, **kwargs)


def get_network_XY(
    network: Network,
    data_file_path: PathLike,
    color_aliases: Optional[dict[str, str]] = None,
    **kwargs,
):
    Y = get_network_data(network, data_file_path, color_aliases, **kwargs)
    X = network.get_input_from_output(Y)
    return X, Y


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
        paths: list[PathLike],
        file_names: list[PathLike],
        extension='.recipe.json5',
        throw_error=True,
        base_path=None,
    ) -> list[Optional[Path]]:
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
        xp_name: str,
        xp_path: PathLike,
        recipe_path: PathLike | list[PathLike],
        lib: PartsLibrary,
        db_path=None,
        data_path='./data/calibrated_data',
        load_data=True,
        ignore_errors=False,
        show_progress=True,
        color_aliases=None,
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
            self.load_xp_file(color_aliases)
        except Exception as e:
            msg = f'Error loading xp file {self.xpfile}: \n{e}'
            raise RuntimeError(msg) from e

        self.load_recipes(recipe_path, ignore_errors=ignore_errors)
        self.load_samples(data_path, load_data=load_data)

        if self.recipe_loading_errors != '' and not ignore_errors:
            raise RuntimeError(self.recipe_loading_errors)
        if self.data_loading_errors != '' and not ignore_errors:
            raise RuntimeError(self.data_loading_errors)

    def load_xp_file(self, color_aliases=None):
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
            color_names = {}
            if 'color_names' in xpobj:
                color_names = {kk: escape(vv) for kk, vv in xpobj['color_names'].items()}

            color_aliases = color_aliases or {}
            self.color_aliases = {**color_aliases, **(color_names or {})}

            self.extra = {}
            for k, v in xpobj.items():
                if k not in required_keys:
                    self.extra[k] = v

    # when ignore_errors is true, we return None when there's an error and append the error message to self.data_loading_errors
    def data_error_handler(self, msg, ignore_errors=False):
        if ignore_errors:
            self.data_loading_errors += msg + '\n\n'
            ut.logger.warning(msg)
            return None
        else:
            raise RuntimeError(msg)

    def recipe_error_handler(self, msg, ignore_errors=False):
        if ignore_errors:
            self.recipe_loading_errors += msg + '\n\n'
            ut.logger.warning(msg)
            return None
        else:
            raise RuntimeError(msg)

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

        def error_handler(msg):
            msg = f'Error in xp {self.name}:' + msg
            return self.recipe_error_handler(msg, ignore_errors=ignore_errors)

        import_recipes_to_sql(
            recipe_files,
            self.dbconn,
            self.lib,
            error_handler=error_handler,
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

    def get_sample_data_file(self, sample_name, ignore_errors=False):
        if sample_name not in self.samples:
            return self.data_error_handler(
                f'Sample {sample_name} not found in xp {self.name}. Available: {self.samples.keys()}',
                ignore_errors=ignore_errors,
            )

        s = self.samples[sample_name]
        if 'data_file' not in s:
            return self.data_error_handler(
                f'No data file listed for sample {sample_name} in xp {self.name}',
                ignore_errors=ignore_errors,
            )
        return s['data_file']

    def load_all_raw_data(self, ignore_errors=False, force_reload=True):
        """Load the raw data for each sample in the xp, and store it in a dict [sample name] -> pandas dataframe"""
        self.raw_data: dict[str, pd.DataFrame] = {}
        for sample_name, s in tqdm(
            list(self.samples.items()),
            desc=f"loading data files for {self.name}",
            disable=not self.show_progress,
        ):
            data_file = self.get_sample_data_file(sample_name, ignore_errors)

            def err_handler(msg):
                msg = f'Error for sample {sample_name} in xp {self.name}:' + msg
                return self.data_error_handler(msg, ignore_errors=ignore_errors)

            load_data_file(
                data_file,
                sample_name,
                error_handler=err_handler,
                use_store=self.raw_data,
                force_reload=force_reload,
            )

    def network_error_handler(self, msg, ignore_errors=False):
        if ignore_errors:
            self.network_building_errors += msg + '\n\n'
            ut.logger.warning(msg)
            return None
        else:
            raise RuntimeError(msg)

    def build_networks(
        self,
        inverse='shortest',
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

            def err_handler(msg):
                msg = f'Error for recipe {recipe_name} in xp {self.name}:' + msg
                return self.network_error_handler(msg, ignore_errors=ignore_errors)

            if recipe_name not in built_networks:
                metadata = {
                    'from_xp': self.name,
                    'recipe_name': recipe_name,
                    'recipe_file': self.recipes[recipe_name],
                }

                built_networks[recipe_name] = build_network(
                    recipe_name,
                    inverse=inverse,
                    dbconn=self.dbconn,
                    error_handler=err_handler,
                    use_cache=use_cache,
                    metadata=metadata,
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

    def load_raw_data(self, sample_name, proteins=None, ignore_errors=False, force_reload=False):
        """Load the raw data for a given sample in the xp, and store it in a pandas dataframe"""
        data_file = self.get_sample_data_file(sample_name, ignore_errors)

        def err_handler(msg):
            msg = f'Error for sample {data_file} in xp {self.name}:' + msg
            return self.data_error_handler(msg, ignore_errors=ignore_errors)

        return load_data_file(
            data_file,
            sample_name,
            proteins,
            error_handler=err_handler,
            use_store=self.raw_data,
            force_reload=force_reload,
        )

    def get_Y(
        self,
        networks: list[Network],
        sample_names: list[str],
        ignore_errors=False,
        force_reload=False,
    ):
        """Returns the output data (including cotx markers) for each network and sample"""
        assert len(networks) == len(sample_names)
        Y = []
        for net, sample_name in zip(networks, sample_names):
            if 'recipe_name' in net.metadata:
                assert net.metadata['recipe_name'] == self.samples[sample_name]['recipe']
            data_file = self.get_sample_data_file(sample_name, ignore_errors)

            def err_handler(msg):
                msg = f'Error for sample {sample_name} in xp {self.name}:' + msg
                return self.data_error_handler(msg, ignore_errors=ignore_errors)

            Y.append(
                get_network_data(
                    net,
                    data_file,
                    color_aliases=self.color_aliases,
                    error_handler=err_handler,
                    use_store=self.raw_data,
                    force_reload=force_reload,
                )
            )

        return Y

    def get_XY(
        self,
        networks: list[Network],
        sample_names: list[str],
        ignore_errors=False,
        force_reload=False,
        **_,
    ):
        """Returns the X and Y data (the independent and dependent variables) for each network and sample"""
        Y = self.get_Y(
            networks, sample_names, ignore_errors=ignore_errors, force_reload=force_reload
        )
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
