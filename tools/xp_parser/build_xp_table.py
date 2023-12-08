### {{{                          --     imports     --
import sys

import openpyxl
import pandas as pd
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
import argparse
import json
from pathlib import Path
from rich import print as rprint

import common as cm


##────────────────────────────────────────────────────────────────────────────}}}
prog = cm.CLIProgram()
### {{{                --     arg declaration and parsing     --

# arguments:
# --database: path to the database file (mandatory)
# --xp_path: path to the experiment files, or empty to use env default
# --mode:
#      'update_from_filesystem': (DEFAULT) update the existing database, prioritizing the filesystem
#      'overwrite': overwrite the existing database
#      'update_from_db': update the existing database, prioritizing the database
# --create: create a new database if it doesn't exist (default: False)

prog.add_argument('--database', type=str, required=True)
prog.add_argument('--mode', type=str, default='update_from_filesystem')
prog.add_argument('--create', action='store_true', default=False)

DEFAULT_CALIB_PATHS = [
    './data/calibrated_data_v3',
    './data/calibrated_data_v2',
    './data/calibrated_data',
]
prog.add_argument('--calib_paths', type=str, nargs='+', default=DEFAULT_CALIB_PATHS)

DEFAULT_CALIB_NAMES = ['v3', 'v2', 'old']
prog.add_argument('--calib_names', type=str, nargs='+', default=DEFAULT_CALIB_NAMES)

DEFAULT_XP_PATH = ut.DEFAULT_XP_PATH
prog.add_argument('--xp_path', type=str, default=DEFAULT_XP_PATH)

DEFAULT_RECIPE_PATH = ut.DEFAULT_RECIPE_PATH
prog.add_argument('--recipe_paths', type=str, nargs='+', default=DEFAULT_RECIPE_PATH)

DEFAULT_XP_CACHE_DIR = './devtmp/cache/xp_objs'
prog.add_argument('--xp_cache_dir', type=str, default=DEFAULT_XP_CACHE_DIR)


DEFAULT_DATA_CONFIG = {
    'network_cache_location': './__cache/network',
    'training_cache_location': './__cache/training',
    'densities_cache_location': './__cache/densities',
    'data_min_value': 500,
    'data_max_value': 100000000.0,
    'data_log_offset': 3000.0,
    'data_log_factor': 100,
    'data_log_poly_threshold': 300,
    'data_log_poly_compression': 0.4,
    'data_sampling_kde_bw_method': 0.02,
    'data_sampling_max_density_samples': 4000,
    'data_sampling_density_quantile_threshold': 0.025,
    'data_sampling_coords_for_density_threshold': 0.15,
}
DEFAULT_DATA_CONFIG_PATH = None
prog.add_argument('--data_config', type=str, default=DEFAULT_DATA_CONFIG_PATH)

prog.parse_args(['--database', 'devtmp/database.xlsx', '--create'])
##────────────────────────────────────────────────────────────────────────────}}}

### {{{                    --     arg postprocessing     --
# get the database path
database_path = Path(prog.database)
if not database_path.exists():
    if not prog.create:
        raise ValueError(f'database file {database_path} does not exist')
    else:
        wb = cm.create_database_file(database_path, ['experiment'])

# check extensiion (it should be an excel file)
if database_path.suffix != '.xlsx':
    raise ValueError(f'database file {database_path} must be an excel file')

prog.xp_path = Path(prog.xp_path)
prog.recipe_paths = [Path(p) for p in prog.recipe_paths]
prog.lib = ut.load_lib()
if prog.data_config is None:
    prog.data_config = DEFAULT_DATA_CONFIG
else:
    import json5

    prog.data_config = json5.load(open(prog.data_config, 'r'))

assert len(prog.calib_paths) == len(prog.calib_names)

##────────────────────────────────────────────────────────────────────────────}}}


xp_entries = {}
xp_objs = {}
# xp_dmans = {}
# dmans = {}
all_recipes = []
all_networks = []

### {{{                  --     list all xps in experiment folder    --
xp_folders = sorted([f for f in prog.xp_path.iterdir() if f.is_dir()])

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
        recipe_path=prog.recipe_paths,
        lib=prog.lib,
        data_path=prog.calib_paths,
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


ut.logger.info(f'found {len(xp_entries)} experiments')

##────────────────────────────────────────────────────────────────────────────}}}
### {{{            --     initial xpdf with calibration info     --


def calibration_info(xppath, calib_paths=prog.calib_paths, calib_names=prog.calib_names):
    # calib_folders = list(xppath.glob('data/calibrated_data*'))
    calib_folders = [xppath / p for p in calib_paths]
    calib_type = 'no'
    calib_plot = False
    calib_path = None

    for calib_folder, calib_name in zip(calib_folders, calib_names):
        if calib_folder.exists():
            calib_type = calib_name
            calib_path = calib_folder
            break

    # check if there is a calibration plot
    calib_diag_path = xppath / 'data' / 'unmixing_diagnostics'
    if calib_diag_path.exists():
        calib_plot = True

    return calib_type, calib_plot, calib_path


for xp in xp_entries.values():
    calib_type, calib_plot, _ = calibration_info(xp['path'])
    xp['calibration_version'] = calib_type
    xp['calibration_diagnostics'] = calib_plot

##────────────────────────────────────────────────────────────────────────────}}}
### {{{  --     create dmans dictionary of datamanagers (where available)    --

print('Creating datamanagers')

# with ut.timer('Creating datamanagers'):
with ut.profiler('./devtmp/profiles/dmans.prof'):
    for xpname, xp in list(xp_objs.items())[:]:
        print(f'loading {xpname}')
        is_ok = True
        if xp.data_files:
            print(f'loading {xpname}')
            xp.load_raw_data()
            networks, samples = xp.build_networks(
                ignore_errors=True, inverse='all', use_cache=prog.data_config['network_cache_location']
            )
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

            # if is_ok:
                # xp_dmans[xpname] = du.DataManager(X, Y, networks, data_cfg=prog.data_config)

print('done')
##────────────────────────────────────────────────────────────────────────────}}}##

### {{{                          --     kde sig     --
import numpy as np

from scipy import stats


def measure(n):
    m1 = np.random.normal(size=n)
    m2 = np.random.normal(scale=0.5, size=n)
    return m1 + m2, m1 - m2


m1, m2 = measure(200)
xmin = m1.min()
xmax = m1.max()
ymin = m2.min()
ymax = m2.max()

X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([m1, m2])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)

##────────────────────────────────────────────────────────────────────────────}}}
### {{{                      --     topology analysis helpers     --


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


def get_uorf_names(uorf_values, ern_names):
    uorf_names = []
    for uorf, ern_name in zip(uorf_values, ern_names):
        ERN_uorf, REC_uorf = uorf
        ERN_uorf = uorf_dict[ERN_uorf]
        REC_uorf = uorf_dict[REC_uorf]
        uorf_names.append((f'{ern_name} ERN: {ERN_uorf}', f'{ern_name} REC: {REC_uorf}'))
    return uorf_names


def get_all_uorf_values(network):
    cdg = network.central_dogma_graph
    ERNs = network.compute_graph.loc[get_all_ERN_ids(network)]
    ERN_names = get_all_ERNs_names(network)
    ERN_inputs = ERNs['cdg_input'].values
    values = []
    for inp in ERN_inputs:
        cdgin = cdg.loc[inp]
        ern_side = cdg.loc[cdgin.iloc[0].predecessor[0]]
        recog_side = cdgin.iloc[1]
        uvals = (get_uorf_value(ern_side.params), get_uorf_value(recog_side.params))
        values.append(uvals)
    names = get_uorf_names(values, ERN_names)
    return tuple(values), tuple(names)


##────────────────────────────────────────────────────────────────────────────}}}
### {{{        --     add architecture family, sequestron type, ...     --

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


local_savedir = Path('~/ResearchMisc/biocomp/').expanduser()
local_savedir.mkdir(parents=True, exist_ok=True)
url_base = 'https://jdisset.com/biocomp'


for net_entry in all_networks:
    net = net_entry['network']
    arch, seqtype = get_network_family(net)
    uorf_vals, uorf_names = get_all_uorf_values(net)
    cdg = net.central_dogma_graph
    genes = flatten(cdg[cdg.type == 'PRT']['content'].tolist())
    full_netname = f'{net.metadata["from_xp"]}.{net.name}'
    geneplotfile = f'jeanplots/{full_netname}.png'
    dataplotfile = f'dataplots/{full_netname}.png'
    new_entry = {
        'xp': net.metadata['from_xp'],
        'name': net.name,
        'sequestron_type': seqtype,
        'architecture': arch,
        'ERN_names': ', '.join(get_all_ERNs_names(net)),
        'uorf_values': ', '.join([str(v) for v in uorf_vals]),
        'uorf_names': ', '.join(flatten(uorf_names)),
        'genes': ', '.join(genes),
        'markers': ', '.join(net.get_inverted_input_proteins()),
        'output_proteins': ', '.join(net.get_output_proteins()),
        'recipe_file': net.metadata['recipe_file'],
        'geneplot': f'=image("{url_base}/{geneplotfile}")',
        'dataplot': f'=image("{url_base}/{dataplotfile}")',
    }
    net_entry.update(new_entry)


##────────────────────────────────────────────────────────────────────────────}}}##

### {{{   --     turn xp_entries and all_networks into xpdf and netdf     --
xpdf = pd.DataFrame(xp_entries).T
# replace all the *_errors types to string
for col in xpdf.columns:
    if '_errors' in col:
        xpdf[col] = xpdf[col].astype(str)
        xpdf[col] = xpdf[col].apply(lambda x: x.replace('nan', ''))


netdf = pd.DataFrame(all_networks)
# netdf = netdf.drop(columns=['network', 'X', 'Y', 'jeanplot'])
netdf = netdf.drop(columns=['network', 'X', 'Y'])
# put name column in front
cols = netdf.columns.tolist()
col_order_beginning = ['name', 'xp', 'architecture']
col_order_end = ['geneplot', 'dataplot']
col_order = (
    col_order_beginning
    + [c for c in cols if c not in col_order_beginning + col_order_end]
    + col_order_end
)
netdf = netdf[col_order]

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                  --     create and update xpdf     --
xpdf = pd.DataFrame(xp_entries).T

xpdf.columns
# replace all the *_errors types to string
for col in xpdf.columns:
    if '_errors' in col:
        xpdf[col] = xpdf[col].astype(str)
        xpdf[col] = xpdf[col].apply(lambda x: x.replace('nan', ''))


# now let's merge with the existing database
dbdf = cm.load_database_table(prog.database, 'experiment')


# try to merge the two tables using the name as key, keeping any extra columns
# and merging on the existing ones according to the mode

if not dbdf.empty:
    priority = 'left' if prog.mode == 'update_from_filesystem' else 'right'
    merged_df = cm.merge_update(xpdf, dbdf, 'name', priority, how='outer', use_right=['id'])
else:
    merged_df = xpdf

# make sure there's an id column and that every row has a unique id
# and put it as the first column
if 'id' not in merged_df.columns:
    merged_df['id'] = np.arange(len(merged_df))
maxid = merged_df['id'].max()
merged_df['id'] = merged_df['id'].fillna(-1).astype(int)
for i, row in merged_df.iterrows():
    if row['id'] == -1:
        maxid += 1
        merged_df.loc[i, 'id'] = maxid

merged_df = cm.reorder_columns_front(merged_df, ['id'])
# use id as index (but keep it as a column)
merged_df.set_index('id', inplace=True, drop=False)

##────────────────────────────────────────────────────────────────────────────}}}

# save the database
cm.save_database_table(merged_df, prog.database, 'experiment')

# minimal styling
workbook = openpyxl.load_workbook(prog.database)
sheet = workbook['experiment']
cm.style_header_row(sheet, '000000', 'EEECEA')
cm.wrap_text_all_cells(sheet)
workbook.save(prog.database)
