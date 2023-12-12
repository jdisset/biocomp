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

# verbosity level
prog.add_argument('--verbose', type=int, default=0)

prog.parse_args(['--database', 'devtmp/database.xlsx', '--create'])
##────────────────────────────────────────────────────────────────────────────}}}
### {{{                    --     arg postprocessing     --

# get the database path
database_path = Path(prog.database)
if not database_path.exists():
    if not prog.create:
        raise ValueError(f'database file {database_path} does not exist')
    else:
        wb = cm.create_database_file(database_path, ['experiment', 'network'])
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

import logging

# loggers = [logging.getLogger(name) for name in sorted(logging.root.manager.loggerDict)]
logging.getLogger('jax').setLevel(logging.WARNING)
# completely silence biocomp's logger (including warning and error messages)
logging.getLogger('biocomp').setLevel(logging.CRITICAL)


# rich console
from rich.console import Console

prog.console = Console()


##────────────────────────────────────────────────────────────────────────────}}}

logger = logging.getLogger('build_xp_table')

xp_entries = {}
xp_objs = {}
all_networks = []

### {{{                  --     list all xps in experiment folder    --
import time
xp_folders = sorted([f for f in prog.xp_path.iterdir() if f.is_dir()])
for xp_dir in tqdm(xp_folders, desc='loading experiments'):
    warning_msg = ''
    subfolders = sorted([f for f in xp_dir.iterdir() if f.is_dir()])
    # check if there is {xp_dir.name}.xp.json5
    xp_json = xp_dir / f'{xp_dir.name}.xp.json5'
    if not xp_json.exists():
        logger.warning(f'no xp.json5 file found in {xp_dir.name}')
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
        show_progress=False,
    )

    recipe_loading_errors = xp.recipe_loading_errors
    xp_entries[xp.name] = {
        'name': xp.name,
        'transfection_date': xp.transfection_date,
        'path': xp_dir,
        'recipe_errors': xp.recipe_loading_errors,
    }
    xp_objs[xp.name] = xp


logger.info(f'found {len(xp_entries)} experiments')

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
### {{{  --     build networks    --
total_samples = sum([len(x.samples) for x in xp_objs.values()])
logger.info(f'Building networks for {total_samples} samples')
progress = tqdm(total=total_samples, desc='Building networks')
for xpname, xp in list(xp_objs.items())[:]:
    is_ok = True
    xp.load_raw_data()
    progress.set_description(f'Building networks for {xpname}')
    networks, sample_names = xp.build_networks(
        ignore_errors=True,
        inverse='all',
        use_cache=prog.data_config['network_cache_location'],
        progress_callback=lambda _: progress.update(1),
    )
    X, Y = xp.get_XY(networks, sample_names, ignore_errors=True)
    if xp.network_building_errors:
        is_ok = False
    if xp.data_loading_errors:
        is_ok = False
    xp_entries[xpname]['network_building_errors'] = xp.network_building_errors
    xp_entries[xpname]['data_loading_errors'] = xp.data_loading_errors
    assert len(networks) == len(X) == len(Y)
    for i, net_entry in enumerate(networks):
        if net_entry:
            net_entry = {
                'xp': xpname,
                'network': net_entry,
                'sample_name': sample_names[i],
                'recipe_name': net_entry.metadata['recipe_name'],
            }
            all_networks.append(net_entry)

    if is_ok:
        logger.info(f'checking data for {xpname}')
        for x, y, net_entry in zip(X, Y, networks):
            if x is None or y is None or x.size == 0 or y.size == 0:
                is_ok = False
                xp_entries[xpname][
                    'data_loadng_errors'
                ] += f'empty data for network {net_entry.name}\n\n'

    
logger.info(f'Done building networks')


##────────────────────────────────────────────────────────────────────────────}}}##
### {{{        --     add architecture family, sequestron type, ...     --
def flatten(l):
    return [item for sublist in l for item in sublist]


local_savedir = Path('~/ResearchMisc/biocomp/').expanduser()
local_savedir.mkdir(parents=True, exist_ok=True)
url_base = 'https://jdisset.com/biocomp'

for net_entry in tqdm(all_networks, desc='Adding network metadata'):
    net = net_entry['network']
    arch, seqtype = ut.get_network_family(net)
    uorf_vals, uorf_names = ut.get_all_uorf_values(net)
    cdg = net.central_dogma_graph
    genes = flatten(cdg[cdg.type == 'PRT']['content'].tolist())
    new_entry = {
        'xp': net.metadata['from_xp'],
        'name': net.name,
        'sequestron_type': seqtype,
        'architecture': arch,
        'ERN_names': ', '.join(ut.get_all_ERNs_names(net)),
        'uorf_values': ', '.join([str(v) for v in uorf_vals]),
        'uorf_names': ', '.join(flatten(uorf_names)),
        'genes': ', '.join(genes),
        'markers': ', '.join(net.get_inverted_input_proteins()),
        'output_proteins': ', '.join(net.get_output_proteins()),
        'recipe_file': net.metadata['recipe_file'],
    }
    net_entry.update(new_entry)


##────────────────────────────────────────────────────────────────────────────}}}##
### {{{                  --     create and update xpdf     --

def merge_update(df, table_name, prog):
    dbdf = cm.load_database_table(prog.database, table_name, create_if_not_exists=True)
    # try to merge the two tables using the name as key, keeping any extra columns
    # and merging on the existing ones according to the mode
    if not dbdf.empty:
        priority = 'left' if prog.mode == 'update_from_filesystem' else 'right'
        merged_df = cm.merge_update(df, dbdf, 'name', priority, how='outer', use_right=['id'])
    else:
        merged_df = df
    return merged_df


def ensure_unique_id(df):
    if 'id' not in df.columns:
        df['id'] = np.arange(len(df))
    maxid = df['id'].max()
    df['id'] = df['id'].fillna(-1).astype(int)
    for i, row in df.iterrows():
        if row['id'] == -1:
            maxid += 1
            df.loc[i, 'id'] = maxid
    df = cm.reorder_columns_front(df, ['id'])
    df.set_index('id', inplace=True, drop=False)
    return df

# minimal styling
def table_style(table_name, prog):
    workbook = openpyxl.load_workbook(prog.database)
    sheet = workbook[table_name]
    cm.style_header_row(sheet, '000000', 'EEECEA')
    cm.wrap_text_all_cells(sheet)
    workbook.save(prog.database)


xpdf = pd.DataFrame(xp_entries).T
# replace all the *_errors types to string
error_cols = sorted([col for col in xpdf.columns if '_errors' in col])
for col in error_cols:
    xpdf[col] = xpdf[col].astype(str)
    xpdf[col] = xpdf[col].apply(lambda x: x.replace('nan', ''))

xpdf = cm.reorder_columns_back(xpdf, error_cols)


merged_xpdf = merge_update(xpdf, 'experiment', prog)
merged_xpdf = ensure_unique_id(merged_xpdf)

cm.save_database_table(merged_xpdf, prog.database, 'experiment')
table_style('experiment', prog)

##────────────────────────────────────────────────────────────────────────────}}}


### {{{                  --     create and update netdf     --

netdf = pd.DataFrame(all_networks)
netdf = netdf.drop(columns=['network'])
netdf = cm.reorder_columns_front(netdf, ['name', 'xp', 'architecture'])
xp_id = merged_xpdf[['id', 'name']].set_index('name')
netdf['xp_id'] = netdf['xp'].apply(lambda x: xp_id.loc[x, 'id'])
merged_netdf = merge_update(netdf, 'network', prog)
merged_netdf = ensure_unique_id(merged_netdf)

cm.save_database_table(merged_netdf, prog.database, 'network')
table_style('network', prog)


print('done')
##────────────────────────────────────────────────────────────────────────────}}}
