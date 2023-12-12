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
lib = ut.load_lib()
##────────────────────────────────────────────────────────────────────────────}}}

# TODO: an xpdf and a networkdf

### {{{                  --     create and populate db     --
xp_path = ut.DEFAULT_XP_PATH
recipe_paths = ut.DEFAULT_RECIPE_PATH
calib_paths = ['./data/calibrated_data_v3', './data/calibrated_data']

# list all folders
xp_folders = sorted([f for f in xp_path.iterdir() if f.is_dir()])

xp_entries = {}
xp_objs = {}
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
    xp_entries[xp.name] = {
        'name': xp.name,
        'transfection_date': xp.transfection_date,
        'path': xp_dir,
        'recipe_errors': xp.recipe_loading_errors,
    }
    xp_objs[xp.name] = xp


xp_entries

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


for xp in xp_entries.values():
    calib_type, calib_plot, _ = calibration_info(xp['path'])
    xp['calibration'] = calib_type
    xp['calibration_diagnostics'] = calib_plot


##────────────────────────────────────────────────────────────────────────────}}}
### {{{  --     create dmans dictionary of datamanagers (where available)    --
dmans = {}
all_networks = []
xp_dmans = {}

for xpname, xp in list(xp_objs.items())[:]:
    print(f'loading {xpname}')
    is_ok = True
    if xp.data_files:
        print(f'loading {xpname}')
        xp.load_raw_data()
        networks, samples = xp.build_networks(ignore_errors=True)
        X, Y = xp.get_XY(networks, samples, ignore_errors=True)
        if xp.network_building_errors:
            is_ok = False
            print(f'{xp.network_building_errors}')
        if xp.data_loading_errors:
            is_ok = False
            print(f'{xp.data_loading_errors}')
        xp_entries[xpname]['network_building_errors'] = xp.network_building_errors
        xp_entries[xpname]['data_loading_errors'] = xp.data_loading_errors
        assert len(networks) == len(X) == len(Y)
        for i, net_entry in enumerate(networks):
            if net_entry:
                net_entry = {
                    'xp': xpname,
                    'network': net_entry,
                    'X': X[i],
                    'Y': Y[i],
                    'sample': samples[i],
                }
                all_networks.append(net_entry)
        if is_ok:
            for x, y, net_entry in zip(X, Y, networks):
                if x.size == 0 or y.size == 0:
                    is_ok = False
                    xp_entries[xpname][
                        'data_loading_errors'
                    ] += f'empty data for network {net_entry.name}\n\n'

        if is_ok:
            xp_dmans[xpname] = du.DataManager(X, Y, networks, data_cfg=training_config)


print('done')
networks

##────────────────────────────────────────────────────────────────────────────}}}##
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


UORF_DICT = {
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


def flatten(l):
    return [item for sublist in l for item in sublist]


# xpdf['architecture'] = 'unknown'
# xpdf['sequestron_type'] = 'unknown'

net_entry = all_networks[0]

for net_entry in all_networks:
    net = net_entry['network']
    arch, seqtype = get_network_family(net)
    net_entry['architecture'] = arch
    net_entry['sequestron_type'] = seqtype
    net_entry['ERN_names'] = ', '.join(get_all_ERNs_names(net))
    net_entry['uorf_values'] = get_all_uorf_values(net)
    net_entry['name'] = net.name
    cdg = net.central_dogma_graph
    genes = flatten(cdg[cdg.type == 'PRT']['content'].tolist())
    net_entry['genes'] = ', '.join(genes)
    net_entry['markers'] = ', '.join(net.get_inverted_input_proteins())
    net_entry['output_proteins'] = ', '.join(net.get_output_proteins())
    net_entry['recipe_file'] = net.metadata['recipe_file']

##────────────────────────────────────────────────────────────────────────────}}}##

### {{{          --     add gene and data plots (if available)     --
local_savedir = Path('~/ResearchMisc/biocomp/').expanduser()
local_savedir.mkdir(parents=True, exist_ok=True)

url_base = 'https://jdisset.com/biocomp'

for network in all_networks:
    net = network['network']
    netname = f'{net.metadata["from_xp"]}.{net.name}'

    geneplotfile = f'jeanplots/{netname}.png'
    network['jeanplot'] = f'=image("{url_base}/{geneplotfile}")'

    dataplotfile = f'dataplots/{netname}.png'
    network['dataplot'] = f'=image("{url_base}/{dataplotfile}")'

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                         --     jeanplot     --

import jeanplot as jp

JP_CONFIG = jp.DEFAULT_CONFIG.copy()
JP_CONFIG['resource_path'] = Path('../jeanplot/resources').resolve()

jeanplot_savedir = Path('~/ResearchMisc/biocomp/jeanplots').expanduser()
jeanplot_savedir.mkdir(parents=True, exist_ok=True)

for i, net_entry in list(enumerate(all_networks))[:]:
    try:
        net = net_entry['network']
        print(f'Plotting {i} / {len(all_networks)} - {net.name}')
        fig, ax = plt.subplots(dpi=300, figsize=(10, 10))
        netscene = jp.NetworkScene(net, position=(0, 0), params=JP_CONFIG)
        netscene.draw(ax)
        netname = f'{net.metadata["from_xp"]}.{net.name}'
        fig.savefig(jeanplot_savedir / f'{netname}.png', bbox_inches='tight')
        plt.show()
        plt.close(fig)
        plt.close('all')
    except Exception as e:
        print(f'Error plotting {net.name}')
        print(e)
        continue

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                         --     plot data     --

dataplot_savedir = Path('~/ResearchMisc/biocomp/dataplots').expanduser()
dataplot_savedir.mkdir(parents=True, exist_ok=True)

xp_to_plot =  ['2023-03-26_MatrixCsy4']

for xp_name, dman in list(xp_dmans.items())[:]:
    if xp_name not in xp_to_plot:
        continue
    print(f'Plotting {xp_name}')
    for i, net in list(enumerate(dman.get_networks()))[:1]:
        print(f'Plotting {i} / {len(dman.get_networks())} - {net.name}')
        # try:
        netname = f'{net.metadata["from_xp"]}.{net.name}'
        filename = f'{netname}.png'
        # if Path(dataplot_savedir / filename).exists():
            # print(f'Skipping {filename} (already exists)')
            # continue
        outputs = net.get_output_proteins()
        inputs = net.get_inverted_input_proteins()
        print(f'outputs: {outputs}')
        params = dict(
            vmin=0,
            vmax=0.7,
            xmin=0,
            xmax=0.85,
            slices=[0.1, 0.3, 0.5],
            knn_method='quantile',
            qu=0.5,
            method='scatter'
        )
        if len(outputs) <= 3:
            fig, ax = pu.mkfig(1, 1, (4, 4), dpi=300)
            pu.network_plot(dman, i, ax=ax, **params)
        else:
            # find the protein in output but not in input
            actual_outputs = [o for o in outputs if o not in inputs]
            assert len(actual_outputs) == 1
            print(f'actual_outputs: {actual_outputs}')
            fig, allaxes = pu.mkfig(3, 3, (3, 3), dpi=300)
            input_order = [0, 1, 2]
            axes = allaxes[0]
            pu.network_plot(dman, i, axes=axes, ax=None, input_order=input_order, **params)
            input_order = [0, 2, 1]
            axes = allaxes[1]
            pu.network_plot(dman, i, axes=axes, ax=None, input_order=input_order, **params)
            input_order = [2, 1, 0]
            axes = allaxes[2]
            pu.network_plot(dman, i, axes=axes, ax=None, input_order=input_order, **params)

        fig.tight_layout()
        fig.savefig(dataplot_savedir / f'{netname}.png', bbox_inches='tight')
        plt.show()
        plt.close(fig)
        plt.close('all')
        # except Exception as e:
            # print(f'Error plotting {net.name}')
            # print(e)
            # continue


##────────────────────────────────────────────────────────────────────────────}}}


xpdf = pd.DataFrame(xp_entries).T
# replace all the *_errors types to string
for col in xpdf.columns:
    if '_errors' in col:
        xpdf[col] = xpdf[col].astype(str)
        xpdf[col] = xpdf[col].apply(lambda x: x.replace('nan', ''))



all_networks

netdf = pd.DataFrame(all_networks)
# netdf = netdf.drop(columns=['network', 'X', 'Y', 'jeanplot'])
netdf = netdf.drop(columns=['network', 'X', 'Y'])
# put name column in front
cols = netdf.columns.tolist()
netdf['id'] = netdf.index
col_order_beginning = ['id', 'name', 'xp', 'architecture']
col_order_end = ['jeanplot', 'dataplot']
col_order = (
    col_order_beginning
    + [c for c in cols if c not in col_order_beginning + col_order_end]
    + col_order_end
)
netdf = netdf[col_order]

# add row id

# export as csv
net_savedir = Path('~/ResearchMisc/biocomp/networkdf').expanduser()
net_savedir.mkdir(parents=True, exist_ok=True)
netdf.to_excel(net_savedir / 'networkdf.xlsx', index=False)

print('done')

##
xpdf.columns
# save xpdf as xls
xpdf.to_excel(net_savedir / 'xpdf.xlsx', index=False)

##
training_config
compute_config
compute_config


##

# better id system
# create deterministic short id for xp and net (netid = f'{xpname}.{i}')
# create a dictionary of xpname -> xp_id
from hashlib import sha1
import base64
for xp in xp_entries.values():
    uid = sha1(xp['name'].encode('utf-8'))
    uid = base64.urlsafe_b64encode(uid.digest())
    print(f'{xp["name"]} -> {uid}')



##

### {{{                          --     traindf     --
import json

this_training = [{
    'training_date': None,
    'training_duration': None,
    'final_loss': None,
    'compute_config': compute_config.dumps(),
    'training_config': json.dumps(training_config, indent=4),
    'training_subset_id': None,
}]

traindf = pd.DataFrame(this_training)
traindf

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                --     manual write to excel sheet     --
import shutil

onedrive = Path(
    '~/Library/CloudStorage/OneDrive-SharedLibraries-MassachusettsInstituteofTechnology/Neuromorphic Biocompiler - Documents'
).expanduser()

xlpath = onedrive / 'BigTable/bigtable.xlsx'

from openpyxl import Workbook
# make if not exists
xlpath.parent.mkdir(parents=True, exist_ok=True)
if xlpath.exists():
    shutil.copy(xlpath, xlpath.parent / f'{xlpath.stem}_backup.xlsx')
else:
    # create file if not exists
    with pd.ExcelWriter(xlpath, engine='openpyxl') as writer:
        xpdf.to_excel(writer, sheet_name='experiments', index=False)
        netdf.to_excel(writer, sheet_name='networks', index=False)
        traindf.to_excel(writer, sheet_name='training_runs', index=False)
##

# Open the file and load the specified sheets
setsdf = pd.read_excel(xlpath, sheet_name='training_runs')
expdf = pd.read_excel(xlpath, sheet_name='experiments')

##
# Dump the 'netdf' dataframe into the "network" sheet
with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
    book = writer.book
    # Remove the existing "network" sheet if it exists
    if 'network' in book.sheetnames:
        del book['network']
    # Write the 'netdf' dataframe to a new "network" sheet
    netdf.to_excel(writer, sheet_name='network', index=False)


##────────────────────────────────────────────────────────────────────────────}}}



### {{{                          --     archive     --

# ##
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive

# gauth = GoogleAuth()
# gauth.LocalWebserverAuth()

# drive = GoogleDrive(gauth)

# ##
# import gspread
# from gspread_dataframe import get_as_dataframe, set_with_dataframe
# from types import SimpleNamespace

# GOOGLE_APP_CREDENTIALS = '/Users/jeandisset/.google/biocomp/key.json'
# SHEET_KEY = '1yAq5x4qKoDzkUGHNA69eH8tJEg1GvbnyJyerb8290-M'

# gspread_client = gspread.service_account(filename=GOOGLE_APP_CREDENTIALS)
# workbook = gspread_client.open_by_key(SHEET_KEY)
# xpsheet = workbook.worksheet('Experiments')
# set_with_dataframe(xpsheet, xpdf, include_index=False, include_column_header=True, resize=True)

# netsheet = workbook.worksheet('Networks')
# netdf.columns
# # reorder columns:
# set_with_dataframe(netsheet, netdf[:], include_index=False, include_column_header=True, resize=True)
# print('done')


##
##────────────────────────────────────────────────────────────────────────────}}}
