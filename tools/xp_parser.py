### {{{                          --     imports     --
import sys

sys.path.append('../scripts')
from biocomp import utils as ut
import scriptutils as su
import json
import biocomp.datautils as du
import biocomp.plotutils as pu
import biocomp.train as train
import biocomp.compute as cmp
import biocomp.parameters as pm
import biocomp as bc
import time
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json5

# pretty print from rich
from rich import print as rprint


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                      --     helpers    --
import sqlite3
import pandas as pd


def find_all_keys_recursively(d, key):
    if isinstance(d, dict):
        if key in d:
            yield d[key]
        for k, v in d.items():
            if isinstance(v, dict):
                for result in find_all_keys_recursively(v, key):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in find_all_keys_recursively(d, key):
                        yield result


def get_colors(recipe_name):
    r = get_recipe(recipe_name)
    all_parts = set()
    for p in find_all_keys_recursively(r, 'parts'):
        all_parts.update(p)
    all_parts = list(sorted(all_parts))
    part_types = [lib.parts.loc[p]['category'] for p in all_parts]
    colors = [all_parts[i] for i, t in enumerate(part_types) if t == 'fluo_marker']
    return colors


def get_parts(l1id):
    l0_cols = ["insulator", "promoter", "5'UTR", "gene", "3'UTR", "terminator"]
    l0_cols = ["promoter", "5'UTR", "gene", "3'UTR"]
    # l0_cols = ["5'UTR", "gene", "3'UTR"]
    L0s = lib.L1s.loc[l1id][l0_cols].tolist()
    part_cols = [f'part_{i}' for i in range(1, 7)]
    parts = []
    for l in L0s:
        parts += [p for p in lib.L0s.loc[l][part_cols].tolist() if p]
    return parts


def get_tus(source_id):
    conn = sqlite3.connect(dbpath)
    tus = pd.read_sql_query(
        f"SELECT TU, position FROM TU_in_source WHERE source = ?", conn, params=(source_id,)
    )
    conn.close()
    res = tus.to_dict(orient='records')
    for r in res:
        r['parts'] = get_parts(r['TU'])
    return res


def get_sources(agg_id):
    conn = sqlite3.connect(dbpath)
    sources = pd.read_sql_query(
        f"SELECT sia.*, s.type FROM source_in_aggregation AS sia, sources AS s WHERE aggregation = ? AND s.name = sia.source",
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
        "SELECT name, description, notes, extra FROM recipes WHERE name = ?",
        conn,
        params=(recipe_name,),
    )
    recipe['aggregations'] = [get_aggregations(recipe_name)]
    # parse extra (it's a json string)
    recipe['extra'] = recipe['extra'].apply(lambda x: json.loads(x))
    res = recipe.to_dict(orient='records')[0]
    # we also want to include the list of xp that have this recipe in their samples
    xps = pd.read_sql_query(
        "SELECT XP FROM recipe_in_XP WHERE recipe = ?", conn, params=(recipe_name,)
    )
    xplist = xps['XP'].to_list()
    res['xps'] = xplist
    conn.close()
    return {'data': res}


def has_sample_data(xp_path, xp_name, sample_name):
    data_path = xp_path / 'data'
    if not data_path.exists():
        return False
    return (data_path / f"{sample_name}.{xp_name}.csv").exists()


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

    for s in res['samples']:
        s['has_data'] = has_sample_data(xp_name, s['name'])

    return {'data': res}


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                      --     load parameters     --
training_archive = du.load('../__results/training_archives/20230923_fulltrain_v0.pkl')
shared_parameters = training_archive['parameters']
compute_config = training_archive['compute_config']
training_config = training_archive['training_config']
compute_config.set_impl('bias', bc.nodes.bias)
##────────────────────────────────────────────────────────────────────────────}}}
### {{{          --     delete db if it exists, create new one     --
lib = su.load_lib()
dbpath = Path('allxps.db')
if dbpath.exists():
    dbpath.unlink()

base_conn = sqlite3.connect(dbpath)
##────────────────────────────────────────────────────────────────────────────}}}### {{{                  --     create and populate db     --
bc.recipe.create_db(base_conn)

xp_path = su.DEFAULT_XP_PATH

# list all folders
xp_folders = sorted([f for f in xp_path.iterdir() if f.is_dir()])


xppaths = {}
xpobjs = []
for xp_dir in xp_folders[:]:
    warning_msg = ''
    subfolders = sorted([f for f in xp_dir.iterdir() if f.is_dir()])
    # check if there is {xp_dir.name}.xp.json5
    xp_json = xp_dir / f'{xp_dir.name}.xp.json5'
    if not xp_json.exists():
        print(f'xp.json5 not found in {xp_dir.name}')
        continue

    with open(xp_json, 'r') as f:
        xp = json5.load(f)

    assert (
        xp['name'] not in xppaths
    ), f'xp name ({xp["name"]}) in {xp_dir.name} already exists at {xppaths[xp["name"]]}'

    if xp['name'] != xp_dir.name:
        warning_msg += f'xp name ({xp["name"]}) does not match folder name ({xp_dir.name})\n'

    xppaths[xp['name']] = xp_dir

    xpobjs.append(xp)

    if warning_msg:
        print(warning_msg)

# # saving xp to db
bc.recipe.xp_to_sql(xpobjs, base_conn)

# saving all recipes to db
recipe_path = su.DEFAULT_RECIPE_PATH
recipenames = [
    x.name for x in recipe_path.iterdir() if x.is_file() and x.name.endswith('.recipe.json5')
]

recipes = []
for recipename in tqdm(recipenames):
    try:
        obj = json5.load(open(recipe_path / recipename))
    except Exception as e:
        print(f'error in {recipename}: {e}')
        continue
    recipes.append(obj)

bc.recipe.recipes_to_sql(recipes, base_conn, lib)

##────────────────────────────────────────────────────────────────────────────}}}
### {{{            --     initial xpdf with calibration info     --
def calibration_info(xppath):
    calib_folders = list(xppath.glob('data/calibrated_data*'))
    calib_type = 'no'
    calib_plot = False
    calib_path = None

    if len(calib_folders) > 0:
        if any([f.name == 'calibrated_data_v3' for f in calib_folders]):
            calib_type = 'v3'
            calib_path = Path('./data/calibrated_data_v3')
        else:
            calib_type = 'old'
            calib_path = Path(f'./data/{calib_folders[0].name}')

    # check if there is a calibration plot
    calib_diag_path = xppath / 'data' / 'unmixing_diagnostics'
    if calib_diag_path.exists():
        calib_plot = True

    return calib_type, calib_plot, calib_path


# select all XPs from db
xpq = pd.read_sql_query("SELECT * FROM XPs", base_conn)
xpdicts = []

for xp in xpobjs:
    d = {
        'xpname': xp['name'],
        'transfection_date': xp['transfection_date'],
    }
    calib_type, calib_plot, _ = calibration_info(xppaths[xp['name']])
    d['calibration'] = calib_type
    d['calibration_diagnostics'] = calib_plot
    xpdicts.append(d)

xpdf = pd.DataFrame(xpdicts)


##────────────────────────────────────────────────────────────────────────────}}}
### {{{    --     merge recipe infos, turn xpdf as one row per recipe     --
recipe_in_xp = pd.read_sql_query("SELECT XP, recipe FROM recipe_in_XP", base_conn)
# this gives a list of recipes for each xp. Let's merge it with the xp dataframe
xpdf = xpdf.merge(recipe_in_xp, left_on='xpname', right_on='XP')
# now we can drop the XP column
xpdf.drop(columns=['XP'], inplace=True)
# group by recipe and aggregate the xps in a list
xpdf = xpdf.groupby('recipe').agg(list).reset_index()
# add number of repeats (number of xps)
xpdf['n_repeats'] = xpdf['xpname'].apply(len)
##────────────────────────────────────────────────────────────────────────────}}}
### {{{     --     create loaded_xps dictionary of loaded XP objects     --


def load_xp(xp_name, xppaths):
    base_xp_path = xppaths[xp_name].parent
    print(f'loading {xp_name} from {base_xp_path}')
    calib_type, _, calib_data_path = calibration_info(xppaths[xp_name])
    try:
        xpobj = bc.XP(
            xp_name,
            base_xp_path,
            recipe_path,
            lib,
            data_path=calib_data_path,
        )
    except Exception as e:
        print(f'Error loading xp {xp_name}: {e}')
        return None
    return xpobj


loaded_xps = {}
for xpname in xppaths:
    x = load_xp(xpname, xppaths)
    if x:
        loaded_xps[xpname] = x

print('done')

##────────────────────────────────────────────────────────────────────────────}}}##
### {{{  --     create dmans dictionary of datamanagers (where available)    --
dmans = {}
for xpname, xp in tqdm(loaded_xps.items()):
    dm = None
    try:
        if xp and xp.datapath:
            print(f'loading {xpname}')
            dm = du.DataManager.from_xps([xp], training_config)
    except Exception as e:
        print(f'Error loading xp {xpname}: {e}')
        continue
    if dm:
        dmans[xpname] = dm
print('done')

##────────────────────────────────────────────────────────────────────────────}}}##
### {{{--     add the best_repeat column: id of best XP, based on quality of calibration     --
prefered_calibration_order = ['v3', 'old', 'no']
xpdf['best_repeat'] = 0

for i, row in xpdf.iterrows():
    assert len(row['calibration']) == len(row['xpname'])
    best_xp_id = 0
    # find the best xp for this recipe using prefered_calibration_order on the calibration column
    for calib in prefered_calibration_order:
        if calib in row['calibration']:
            best_xp_id = row['calibration'].index(calib)
            break
    xpdf.loc[i, 'best_repeat'] = best_xp_id
##────────────────────────────────────────────────────────────────────────────}}}
### {{{      --     create networks dictionary of recipe -> network     --

networks = {}

built_networks_from_xp = {}

for i, row in tqdm(xpdf.iterrows()):
    xpname = row['xpname'][row['best_repeat']]
    if xpname in dmans:
        net_names = [n.name for n in dmans[xpname].get_networks()]
        net_id = net_names.index(row['recipe'])
        networks[row['recipe']] = dmans[xpname].get_networks()[net_id]
    elif xpname in loaded_xps:
        if xpname not in built_networks_from_xp:
            nets, _ = loaded_xps[xpname].build_networks(use_db=base_conn, ignore_errors=True)
            built_networks_from_xp[xpname] = nets
        net_names = [n.name for n in built_networks_from_xp[xpname]]
        if row['recipe'] not in net_names:
            print(f'no network built for {row["recipe"]} in {xpname}')
            continue
        net_id = net_names.index(row['recipe'])
        networks[row['recipe']] = built_networks_from_xp[xpname][net_id]

xpdf['network_loaded'] = False
for i, row in xpdf.iterrows():
    if row['recipe'] in networks:
        xpdf.loc[i, 'network_loaded'] = True

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                      --     quantify uorfs     --


def get_uorf_value(param):
    if 'tl_rate' in param:
        u = param['tl_rate'][0].split('_')[0]
        try:
            v = int(u[:-1]) * 10
        except ValueError:
            v = 0
        if u[-1] == 'w':
            v = v - 5
        return v
    else:
        return 0


uorf_dict = {
    0: 'No uORF',
    5: 'weak uORF',
    10: '1x uORF',
    20: '2x uORF',
    30: '3x uORF',
    40: '4x uORF',
    50: '5x uORF',
    60: '6x uORF',
    70: '7x uORF',
    80: '8x uORF',
}


def get_all_ERN_ids(network):
    ERN_ids = network.compute_graph[network.compute_graph['type'] == 'sequestron_ERN'].index.values
    return network.sort_nodes_by_upstream(ERN_ids)


def get_all_ERNs_names(network):
    ERNs = network.compute_graph.loc[get_all_ERN_ids(network)]
    ERN_extras = ERNs['extra'].values
    ERN_names = [e['seq_name'].split('#')[0].split('::')[-1] for e in ERN_extras]
    return ERN_names


def get_all_uorf_values(network):
    cdg = network.central_dogma_graph
    ERNs = network.compute_graph.loc[get_all_ERN_ids(network)]
    ERN_inputs = ERNs['cdg_input'].values
    values = []
    for inp in ERN_inputs:
        cdgin = cdg.loc[inp]
        ern_side = cdg.loc[cdgin.iloc[0].predecessor[0]]
        recog_side = cdgin.iloc[1]
        values.append((get_uorf_value(ern_side.params), get_uorf_value(recog_side.params)))
    return tuple(values)


##────────────────────────────────────────────────────────────────────────────}}}

### {{{        --     add architecture family and sequestron type     --

from typing import List, Callable

def get_ERN_ids(network):
    return network.compute_graph[network.compute_graph['type'] == 'sequestron_ERN'].index.values

def get_RCB_ids(network):
    return network.compute_graph[network.compute_graph['type'].str.startswith('sequestron_R')].index.values

def get_sequestron_ids(network):
    return network.compute_graph[network.compute_graph['type'].str.startswith('sequestron_')].index.values


def make_is_upstream(network):
    def is_upstream(i, j):
        return network.compute_node_is_upstream_of(i, j)
    return is_upstream

def topological_sort(
    node_list: List[int], is_upstream: Callable[[int, int], bool]
) -> List[List[int]]:

    visited = set()
    batches = []
    while len(visited) < len(node_list):
        independent = [
            i
            for i in node_list
            if i not in visited
            and all([j in visited for j in node_list if j != i and is_upstream(j, i)])
        ]

        if not independent:
            raise ValueError('Cycle detected in graph')
        visited.update(independent)
        batches.append(independent)
    return batches


def get_network_family(network):
    erns = get_ERN_ids(network)
    rcbs = get_RCB_ids(network)
    seqs = get_sequestron_ids(network)
    ts = topological_sort(seqs, make_is_upstream(network))

    seqtype = 'none'
    family = 'unknown'
    match (len(erns) > 0, len(rcbs) > 0):
        case (True, True):
            seqtype = 'hybrid'
        case (True, False):
            seqtype = 'ERN'
        case (False, True):
            seqtype = 'RCB'

    match (len(seqs), len(ts)):
        case (0, 0):
            family = 'no device'
        case (1, 1):
            family = 'single'
        case (2, 2):
            family = 'cascade'
        case (2, 1):
            family = 'dual region'
        case (3, 2):
            family = 'bandpass'


    return family, seqtype

xpdf['architecture'] = 'unknown'
xpdf['sequestron_type'] = 'unknown'
for i, row in xpdf.iterrows():
    if row['recipe'] in networks:
        net = networks[row['recipe']]
        arch, seqtype = get_network_family(net)
        xpdf.loc[i, 'architecture'] = arch
        xpdf.loc[i, 'sequestron_type'] = seqtype

##────────────────────────────────────────────────────────────────────────────}}}##

colors = [get_colors(r) for r in xpdf['recipe']]

xpdf['colors'] = colors

for i, row in xpdf.iterrows():
    if row['recipe'] in networks:
        net = networks[row['recipe']]
        prots = net.get_output_proteins()
        xpdf.at[i, 'colors'] = prots

xpdf

##
# save xpdf to csv
xpdf.to_csv('xpdf.csv', index=False)
