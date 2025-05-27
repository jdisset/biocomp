from .library import PartsLibrary as PartsLibrary
from . import utils as ut
from .network import Network, inverted_network
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import sqlite3
import json
import logging as log
from typing import Union, Optional, Callable

from biocomp.logging_config import get_logger

logger = get_logger(__name__)
PathLike = Union[str, Path]


# TODO: this whole module is a very old and ugly mess...
# It needs to be heavily refactored and cleaned up
# -> using sqlmodel + pydantic should make things much simpler (wtf are all these nasty sql queries?)
# -> I'm not even sure we really need an XP class...Maybe it should just be about networks + data
# and we rely on the database to add grouping info (like XP, etc)
# anyway... no time for that now...


def escape_name(name):
    return name.replace("-", "_").replace(" ", "_").upper()


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
        c.execute("SELECT name FROM recipes WHERE name = ?", (obj["name"],))
        if c.fetchone():
            # already in db so we skip
            log.info(f"Recipe {obj['name']} already in db, skipping")

        extra = {k: v for k, v in obj.items() if k not in ["name", "description", "notes"]}
        extra_json = json.dumps(extra)
        logger.debug(f"Adding recipe {obj['name']} to temp sql db")
        c.execute(
            "INSERT INTO recipes VALUES (?, ?, ?, ?)",
            (
                obj["name"],
                obj["description"] if "description" in obj else None,
                obj["notes"] if "notes" in obj else None,
                extra_json if extra else None,
            ),
        )
        error_in_recipe = False
        for agg in obj["content"]:
            ratios = []
            for s in agg["sources"]:
                if "ratio" in s:
                    ratios.append(s["ratio"])
                else:
                    ratios.append(1.0)
            ratios = np.array(ratios)
            qtty = float(np.sum(ratios))
            c.execute(
                "INSERT INTO aggregations VALUES (?, ?, ?)",
                (None, obj["name"], agg["notes"] if "notes" in agg else None),
            )
            aggregation_id = c.lastrowid
            ratios = ratios / qtty
            for r, s in zip(ratios, agg["sources"]):
                type = None
                l1ids = []
                if s["plasmid"] in lib.L1s.index:
                    type = 1
                    l1ids = [lib.L1s.loc[s["plasmid"]].name]
                elif s["plasmid"] in lib.L2s.index:
                    type = 2
                    slot_cols = [f"slot_{i}" for i in range(1, 7)]
                    l1ids = [s for s in lib.L2s.loc[s["plasmid"]][slot_cols].tolist() if s]
                if type is None:
                    err_msg = f"In recipe {obj['name']}: unknown plasmid {s['plasmid']}"
                    logger.error(err_msg)
                    error_in_recipe = True
                    error_handler(err_msg)
                    continue  # we still continue to get a list of all errors
                c.execute("SELECT name FROM sources WHERE name = ?", (s["plasmid"],))
                if not c.fetchone():
                    c.execute("INSERT INTO sources VALUES (?, ?)", (s["plasmid"], type))
                    for i, l1id in enumerate(l1ids):
                        c.execute(
                            "INSERT INTO TU_in_source VALUES (?, ?, ?)", (s["plasmid"], l1id, i)
                        )
                # we put in "extra" everything other than ratio, plasmid and notes. Serialized to json.
                extra = {k: v for k, v in s.items() if k not in ["ratio", "plasmid", "notes"]}
                extra_json = json.dumps(extra)
                c.execute(
                    "INSERT INTO source_in_aggregation VALUES (?, ?, ?, ?, ?)",
                    (
                        aggregation_id,
                        s["plasmid"],
                        r,
                        s["notes"] if "notes" in s else None,
                        extra_json,
                    ),
                )
        if error_in_recipe:
            logger.error(f"Skipped recipe {obj['name']} because of import errors")
            c.execute("DELETE FROM recipes WHERE name = ?", (obj["name"],))
            conn.commit()

    conn.commit()


def xp_to_sql(xps: list, conn):
    c = conn.cursor()
    create_db(conn)
    for obj in xps:
        c.execute("SELECT name FROM XPs WHERE name = ?", (obj["name"],))
        if c.fetchone():
            # already in db so we skip
            logger.debug(f"XP {obj['name']} already in db, skipping")
            return
        logger.info(f"Adding XP {obj['name']} to sql db")
        c.execute(
            "INSERT INTO XPs VALUES (?, ?, ?, ?)",
            (
                obj["name"],
                obj["flow_date"] if "flow_date" in obj else None,
                obj["transfection_date"] if "transfection_date" in obj else None,
                json.dumps(
                    {
                        k: v
                        for k, v in obj.items()
                        if k not in ["name", "flow_date", "transfection_date"]
                    }
                ),
            ),
        )
        for s in obj["samples"]:
            c.execute(
                "INSERT INTO recipe_in_XP VALUES (?, ?, ?, ?)",
                (obj["name"], s["recipe"], s["name"], s["notes"] if "notes" in s else None),
            )
    conn.commit()


def import_recipes_to_sql(
    recipe_files: list[PathLike] | PathLike,
    conn: sqlite3.Connection,
    lib: PartsLibrary,
    error_handler=None,
    show_progress=True,
    recipe_objects=None,
) -> list:
    if error_handler is None:

        def _handler(msg):
            raise RuntimeError(msg)

        error_handler = _handler

    # recipe files are json5 files
    recipe_objects = recipe_objects or []

    for f in tqdm(recipe_files, desc="Importing recipes", disable=not show_progress):
        recipe = ut.load_json5(f)
        logger.debug(f"Importing recipe {recipe['name']}")
        if not Path(f).name == f"{recipe['name']}.recipe.json5":
            error_handler(f"File vs recipe name mismatch (recipe: {recipe['name']}, file: {f})")
        recipe_objects.append(recipe)

    recipes_to_sql(recipe_objects, conn, lib, error_handler=error_handler)
    return recipe_objects


def build_network(
    recipe_name,
    dbconn,
    lib,
    inverse="shortest",
    metadata=None,
    error_handler=None,
    use_cache=None,
) -> Network | list[Network]:
    if error_handler is None:

        def _handler1(msg):
            logger.error(f"Error building network for recipe {recipe_name}: {msg}")
            raise RuntimeError(msg)

        error_handler = _handler1
    else:

        def _handler2(msg):
            error_handler(f"Error building network for recipe {recipe_name}: {msg}")

        error_handler = _handler2

    if metadata is None:
        metadata = {"recipe_name": recipe_name}

    try:
        logger.debug(f"Building network for recipe {recipe_name}")
        fwd_network = Network.from_db(
            lib,
            recipe_name,
            dbconn,
            metadata=metadata,
            use_cache=use_cache,
        )
    except Exception as e:
        return error_handler(f"Can't build network: {e}")

    if not inverse:
        return fwd_network
    else:
        return [n for n in inverted_network(fwd_network, mode=inverse, use_cache=use_cache)]


def network_from_recipe(
    recipe_path: Optional[PathLike],
    lib: PartsLibrary,
    db_path=":memory:",
    metadata=None,
    error_handler=None,
    recipe_object: Optional[dict] = None,
    **kwargs,
) -> Network | list[Network]:
    dbconn = sqlite3.connect(db_path)
    if recipe_object is not None:
        # using a recipe object directly
        recipe_objects = [recipe_object]
        assert recipe_path is None, "recipe_path should be None if recipe_object is provided"
        recipe_paths: list[PathLike] = []
    else:
        recipe_objects = None
        assert recipe_path is not None, (
            "recipe_path should not be None if recipe_object is not provided"
        )
        recipe_paths = [recipe_path]

    recipe = import_recipes_to_sql(
        recipe_paths,
        dbconn,
        lib,
        show_progress=False,
        error_handler=error_handler,
        recipe_objects=recipe_objects,
    )[0]

    if metadata is None:
        metadata = {"recipe_name": recipe["name"], "recipe_file": recipe_path}
    return build_network(
        recipe["name"], dbconn, lib, metadata=metadata, error_handler=error_handler, **kwargs
    )


def load_data_file(
    data_file_path: PathLike,
    proteins: Optional[list[str]] = None,
    error_handler: Optional[Callable] = None,
    use_store=None,
    force_reload=False,
):
    if error_handler is None:

        def _handler(msg):
            logger.error(f"Error loading data file {data_file_path}: {msg}")
            raise RuntimeError(msg)

        error_handler = _handler

    if use_store is None:
        use_store = {}

    f = Path(data_file_path)
    if not f.exists():
        return error_handler(f"Data file {f} not found")

    logger.debug(f"Loading data file {f}")

    if data_file_path not in use_store or force_reload:
        ext = f.suffix
        if ext == ".csv":
            content = pd.read_csv(f, engine="pyarrow")
        elif ext == ".parquet":
            content = pd.read_parquet(f)
        else:
            return error_handler(f"Unsupported data file format {ext}")
        assert isinstance(content, pd.DataFrame)
        use_store[data_file_path] = content

    data = use_store[data_file_path]

    res = None
    available_columns = set(data.columns)
    if proteins is None:
        res = data.to_numpy()
    else:
        remainder = set(proteins) - available_columns
        if len(remainder) > 0:
            return error_handler(
                f"""Proteins {remainder} was requested but not found in data. 
Available: {available_columns}
"""
            )

        res = np.asarray(data[proteins])

    if res is None:
        return error_handler(f"Data file {data_file_path} is empty")

    logger.debug(f"Data file {data_file_path} loaded with shape {res.shape}")
    return res


def get_network_data(
    network: Network,
    data_file_path: PathLike,
    color_aliases: Optional[dict[str, str]] = None,
    error_handler: Optional[Callable] = None,
    **kwargs,
) -> Optional[np.ndarray]:
    # we want to reorder data columns to match the network's output

    out_proteins = escape(network.get_output_proteins())
    if color_aliases is not None:
        aliases = escape(color_aliases)
        out_proteins = [aliases.get(p, p) for p in out_proteins]

    if error_handler is None:

        def _handler(msg):
            logger.error(
                f"Error getting data {data_file_path}\nfor network {network.name}\nwith proteins {out_proteins}:\n{msg}"
            )
            raise RuntimeError(msg)

        error_handler = _handler

    return load_data_file(
        data_file_path,
        proteins=out_proteins,
        error_handler=error_handler,
        **kwargs,
    )


def get_network_XY(
    network: Network,
    data_file_path: PathLike,
    color_aliases: Optional[dict[str, str]] = None,
    **kwargs,
):
    Y = get_network_data(network, data_file_path, color_aliases, **kwargs)
    X = network.get_input_from_output(Y)
    return X, Y
