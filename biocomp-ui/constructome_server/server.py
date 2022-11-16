## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     import and load     --
# ···············································································
import sys

sys.path.append('../../scripts/')
from flask_cors import CORS
from flask import Flask, request
from pathlib import Path
import pandas as pd
import json

import sqlite3
import biocomp as bc
import scriptutils as ut
from tqdm import tqdm

import json5

print('Loading data...')
lib = ut.load_lib()
print('Data loaded.')

# create constructome database. If it already exists, delete it.
dbpath = Path('constructome.db')
if dbpath.exists():
    dbpath.unlink()

base_conn = sqlite3.connect(dbpath)


# saving xp to db
xp_path = ut.DEFAULT_XP_PATH
xpnames = [x.name for x in xp_path.iterdir() if x.is_dir()]
xpobjs = [json5.load(open(xp_path / xpname / f"{xpname}.xp.json5")) for xpname in tqdm(xpnames)]
bc.recipe.xp_to_sql(xpobjs, base_conn)

# saving all recipes to db
recipe_path = ut.DEFAULT_RECIPE_PATH
recipenames = [x.name for x in recipe_path.iterdir() if x.is_file()]
recipes = [json5.load(open(recipe_path / recipename)) for recipename in tqdm(recipenames)]
bc.recipe.recipes_to_sql(recipes, base_conn, lib)

base_conn.close()

sql_schema = """
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
        FOREIGN KEY (XP) REFERENCES XPs(id),
        FOREIGN KEY (recipe) REFERENCES recipes(name),
        PRIMARY KEY(XP, recipe));

    """

##


def get_tus(source_id):
    conn = sqlite3.connect(dbpath)
    tus = pd.read_sql_query(
        f"SELECT TU, position FROM TU_in_source WHERE source = ?", conn, params=(source_id,)
    )
    conn.close()
    return tus.to_dict(orient='records')


def get_sources(agg_id):
    conn = sqlite3.connect(dbpath)
    sources = pd.read_sql_query(
        f"SELECT source, ratio, notes, extra FROM source_in_aggregation WHERE aggregation = ?",
        conn,
        params=(agg_id,),
    )
    # parse extra (it's a json string)
    sources['extra'] = sources['extra'].apply(lambda x: json.loads(x))
    sources['tus'] = [get_tus(source_id) for source_id in sources['source']]
    conn.close()
    return sources.to_dict(orient='records')


def get_aggregations(recipe_name):
    conn = sqlite3.connect(dbpath)
    aggregations = pd.read_sql_query(
        f"SELECT id, notes FROM aggregations WHERE recipe = ?", conn, params=(recipe_name,)
    )
    aggregations['sources'] = [get_sources(agg_id) for agg_id in aggregations['id']]
    conn.close()
    return aggregations.to_dict(orient='records')


def get_recipe(recipe_name):
    conn = sqlite3.connect(dbpath)
    recipe = pd.read_sql_query(
        "SELECT name, description, notes, extra FROM recipes WHERE name = ?", conn, params=(recipe_name,)
    )
    recipe['aggregations'] = [get_aggregations(recipe_name)]
    # parse extra (it's a json string)
    recipe['extra'] = recipe['extra'].apply(lambda x: json.loads(x))
    conn.close()
    return recipe.to_dict(orient='records')[0]


def get_xp(xp_name):

    conn = sqlite3.connect(dbpath)
    xp = pd.read_sql_query(
        f"SELECT name, flow_date, transfection_date, extra FROM XPs WHERE name = ?",
        conn,
        params=(xp_name,),
    )
    # parse extra (it's a json string)
    xp['extra'] = xp['extra'].apply(lambda x: json.loads(x))

    assert len(xp) == 1

    conn.close()
    res = xp.to_dict(orient='records')[0]
    # also, we want extra to be expanded so that all its fields are in the top level
    res.update(res['extra'])
    del res['extra']
    return res


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/')
def index():
    return 'Server Works!'


@app.route('/plasmid')
def _plasmids():
    l2s = lib.L2s.reset_index()
    l2s['type'] = 'L2'
    l1s = lib.L1s.reset_index()
    l1s['type'] = 'L1'
    plasmids = pd.concat([l2s, l1s])
    plasmids = plasmids.rename(columns={'id': 'source_id'})
    return plasmids.to_json(orient='records')


# a route for xp. if no id is given, return all xps
@app.route('/xp/<xp_name>')
def _xp(xp_name):
    xp = get_xp(xp_name)
    return json.dumps(xp)

@app.route('/xp')
def _xps():
    conn = sqlite3.connect(dbpath)
    xpnames = pd.read_sql_query("SELECT name FROM XPs", conn)
    conn.close()
    xps = [get_xp(xp_name) for xp_name in xpnames['name']]
    return json.dumps(xps)


@app.route('/recipe/<recipe_name>')
def _recipe(recipe_name):
    recipe = get_recipe(recipe_name)
    return json.dumps(recipe)

@app.route('/recipe')
def _recipes():
    conn = sqlite3.connect(dbpath)
    recipenames = pd.read_sql_query("SELECT name FROM recipes", conn)
    conn.close()
    recipes = [get_recipe(recipe_name) for recipe_name in recipenames['name']]
    return json.dumps(recipes)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port="4321", debug=True, use_reloader=True)
