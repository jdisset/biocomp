### {{{                          --     imports     --
import sys

from dataclasses import dataclass
from typing import List, Tuple

from biocomp import utils as ut
import json
import biocomp.datautils as du
import biocomp.plotutils as pu
import biocomp.utils as ut
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


def has_sample_data(xp_path, xp_name, sample_name):
    data_path = xp_path / 'data'
    if not data_path.exists():
        return False
    return (data_path / f"{sample_name}.{xp_name}.csv").exists()


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                      --     load parameters     --
training_archive = ut.load('../../__results/training_archives/20230923_fulltrain_v0.pkl')
shared_parameters = training_archive['parameters']
compute_config = training_archive['compute_config']
training_config = training_archive['training_config']
compute_config.set_impl('bias', bc.nodes.bias)
##────────────────────────────────────────────────────────────────────────────}}}
### {{{          --     delete db if it exists, create new one     --
lib = ut.load_lib()
dbpath = Path('allxps.db')
if dbpath.exists():
    dbpath.unlink()

# base_conn=None
# base_conn = sqlite3.connect(dbpath)
# base_conn.close()
##────────────────────────────────────────────────────────────────────────────}}}

### {{{                  --     create and populate db     --
xp_path = ut.DEFAULT_XP_PATH
recipe_paths = ut.DEFAULT_RECIPE_PATH
calib_paths = ['./data/calibrated_data_v3', './data/calibrated_data']

# list all folders
xp_folders = sorted([f for f in xp_path.iterdir() if f.is_dir()])

xppaths = {}
xpobjs = []
xperrors = []
for xp_dir in xp_folders[:]:
    warning_msg = ''
    subfolders = sorted([f for f in xp_dir.iterdir() if f.is_dir()])
    # check if there is {xp_dir.name}.xp.json5
    xp_json = xp_dir / f'{xp_dir.name}.xp.json5'
    if not xp_json.exists():
        print(f'xp.json5 not found in {xp_dir.name}')
        continue

    base_xp_path = xp_dir.parent

    xp = bc.XP(
        xp_dir.name,
        base_xp_path,
        recipe_path=recipe_paths,
        lib=lib,
        # db_path=dbpath,
        data_path=calib_paths,
        load_data=False,
        ignore_errors=True,

    )

    recipe_loading_errors = xp.recipe_loading_errors
    xperrors.append(recipe_loading_errors)
    if recipe_loading_errors:
        warning_msg += f'loading errors: {recipe_loading_errors}\n'

    assert (
        xp.name not in xppaths
    ), f'xp name ({xp.name}) in {xp_dir.name} already exists at {xppaths[xp.name]}'

    if xp.name != xp_dir.name:
        warning_msg += f'xp name ({xp.name}) does not match folder name ({xp_dir.name})\n'

    xppaths[xp.name] = xp_dir
    xpobjs.append(xp)
    if warning_msg:
        print(warning_msg)

loaded_xps = {xp.name: xp for xp in xpobjs if xp}

##────────────────────────────────────────────────────────────────────────────}}}
### {{{            --     initial xpdf with calibration info     --
# reopen db
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
xpdicts = []

for xp in xpobjs:
    d = {
        'xpname': xp.name,
        'transfection_date': xp.transfection_date,
        'recipe_errors': xp.recipe_loading_errors,
        'data_loading_errors': '',
    }
    calib_type, calib_plot, _ = calibration_info(xppaths[xp.name])
    d['calibration'] = calib_type
    d['calibration_diagnostics'] = calib_plot
    if calib_type != 'no':
        d['data_errors']: xp.data_loading_errors
    xpdicts.append(d)

xpdf = pd.DataFrame(xpdicts)

##────────────────────────────────────────────────────────────────────────────}}}

### {{{  --     create dmans dictionary of datamanagers (where available)    --
dmans = {}

for xpname, xp in list(loaded_xps.items())[-6:-7:-1]:
    print(f'loading {xpname}')
    if xp.data_files:
        print(f'loading {xpname}')
        xp.load_raw_data()
        networks, samples = xp.build_networks(ignore_errors=True)
        X, Y = xp.get_XY(networks, samples, ignore_errors=True)
        if xp.network_building_errors:
            print(f'{xp.network_building_errors}')
        if xp.data_loading_errors:
            print(f'{xp.data_loading_errors}')
        assert len(networks) == len(X) == len(Y)
        networks
print('done')
networks

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

self.networks = {}

built_networks_from_xp = {}

for i, row in tqdm(xpdf.iterrows()):
    xpname = row['xpname'][row['best_repeat']]
    if xpname in dmans:
        net_names = [n.name for n in dmans[xpname].get_networks()]
        net_id = net_names.index(row['recipe'])
        self.networks[row['recipe']] = dmans[xpname].get_networks()[net_id]
    elif xpname in loaded_xps:
        if xpname not in built_networks_from_xp:
            nets, _ = loaded_xps[xpname].build_networks(use_db=base_conn, ignore_errors=True)
            built_networks_from_xp[xpname] = nets
        net_names = [n.name for n in built_networks_from_xp[xpname]]
        if row['recipe'] not in net_names:
            print(f'no network built for {row["recipe"]} in {xpname}')
            continue
        net_id = net_names.index(row['recipe'])
        self.networks[row['recipe']] = built_networks_from_xp[xpname][net_id]

xpdf['network_loaded'] = False
for i, row in xpdf.iterrows():
    if row['recipe'] in self.networks:
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
### {{{        --     add archit ecture family and sequestron type     --

from typing import List, Callable


def get_ERN_ids(network):
    return network.compute_graph[network.compute_graph['type'] == 'sequestron_ERN'].index.values


def get_RCB_ids(network):
    return network.compute_graph[
        network.compute_graph['type'].str.startswith('sequestron_R')
    ].index.values


def get_sequestron_ids(network):
    return network.compute_graph[
        network.compute_graph['type'].str.startswith('sequestron_')
    ].index.values


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
    if row['recipe'] in self.networks:
        net = self.networks[row['recipe']]
        arch, seqtype = get_network_family(net)
        xpdf.loc[i, 'architecture'] = arch
        xpdf.loc[i, 'sequestron_type'] = seqtype

##────────────────────────────────────────────────────────────────────────────}}}##
### {{{                     --     list used colors     --
colors = [get_colors(r) for r in xpdf['recipe']]

xpdf['colors'] = colors

for i, row in xpdf.iterrows():
    if row['recipe'] in self.networks:
        net = self.networks[row['recipe']]
        prots = net.get_output_proteins()
        xpdf.at[i, 'colors'] = prots

##────────────────────────────────────────────────────────────────────────────}}}
### {{{                       --     export to csv     --
export_copy = xpdf.copy()
for col in ['xpname', 'calibration', 'colors']:
    export_copy[col] = export_copy[col].apply(lambda x: ','.join(x) if isinstance(x, list) else x)


export_copy.to_csv('xpdf.csv', index=False)

##────────────────────────────────────────────────────────────────────────────}}}##
