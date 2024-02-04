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
from common import (
    DEFAULT_CALIB_PATHS,
    DEFAULT_CALIB_NAMES,
    DEFAULT_XP_PATH,
    DEFAULT_RECIPE_PATH,
    DEFAULT_XP_CACHE_DIR,
    DEFAULT_DATA_CONFIG,
    DEFAULT_DATA_CONFIG_PATH,
)

import logging

##────────────────────────────────────────────────────────────────────────────}}}
prog = cm.CLIProgram()
logger = logging.getLogger('build_xp_table')
logger.setLevel(logging.DEBUG)
logging.getLogger('biocomp').setLevel(logging.ERROR)
import psycopg2
DBCONN = cm.connect_to_db()
### {{{                --     arg declaration and parsing     --

# arguments:

prog.add_argument('--calib_paths', type=str, nargs='+', default=DEFAULT_CALIB_PATHS)
prog.add_argument('--calib_names', type=str, nargs='+', default=DEFAULT_CALIB_NAMES)
prog.add_argument('--xp_path', type=str, default=DEFAULT_XP_PATH)
# --xp_path: path to the experiment files, or empty to use env default
prog.add_argument('--recipe_paths', type=str, nargs='+', default=DEFAULT_RECIPE_PATH)
prog.add_argument('--xp_cache_dir', type=str, default=DEFAULT_XP_CACHE_DIR)
prog.add_argument('--base_dir', type=str, default=Path(DEFAULT_XP_PATH).parent)
prog.add_argument('--data_config', type=str, default=DEFAULT_DATA_CONFIG_PATH)

# verbosity level
prog.add_argument('--verbose', type=int, default=0)
prog.parse_args([])
##────────────────────────────────────────────────────────────────────────────}}}
### {{{                    --     arg postprocessing     --

prog.xp_path = Path(prog.xp_path)
prog.base_dir = Path(prog.base_dir)
prog.recipe_paths = [Path(p) for p in prog.recipe_paths]
prog.lib = ut.load_lib()

if prog.data_config is None:
    prog.data_config = DEFAULT_DATA_CONFIG
else:
    import json5

    prog.data_config = json5.load(open(prog.data_config, 'r'))

assert len(prog.calib_paths) == len(prog.calib_names)

# loggers = [logging.getLogger(name) for name in sorted(logging.root.manager.loggerDict)]
logging.getLogger('jax').setLevel(logging.WARNING)
# completely silence biocomp's logger (including warning and error messages)
logging.getLogger('biocomp').setLevel(logging.CRITICAL)


# rich console
from rich.console import Console

prog.console = Console()


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                  --     list all xps in experiment folder    --

xp_entries = {}
xp_objs = {}

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
        'path': Path(xp_dir).relative_to(prog.base_dir),
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
    xp['has_calibration_diagnostics'] = calib_plot


##────────────────────────────────────────────────────────────────────────────}}}
### {{{  --     build networks    --
all_networks = []

total_samples = sum([len(x.samples) for x in xp_objs.values()])
logger.info(f'Building networks for {total_samples} samples')
progress = tqdm(total=total_samples, desc='Building networks')

for xpname, xp in list(xp_objs.items())[:]:
    is_ok = True
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
            sname = sample_names[i]
            data_file = xp.get_sample_data_file(sname, ignore_errors=True)
            # subtract prog.xp_path from data_file to get the relative path:
            if data_file is not None:
                data_file = Path(data_file).relative_to(prog.base_dir)

            recipe_file = net_entry.metadata['recipe_file']
            if recipe_file is not None:
                recipe_file = Path(recipe_file).relative_to(prog.base_dir)

            net_entry = {
                'xp': xpname,
                'network': net_entry,
                'sample_name': sname,
                'recipe_name': net_entry.metadata['recipe_name'],
                'recipe_file': recipe_file,
                'data_file': data_file,
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
        'ern_names': ', '.join(ut.get_all_ERNs_names(net)),
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


xpdf = pd.DataFrame(xp_entries).T
# replace all the *_errors types to string
error_cols = sorted([col for col in xpdf.columns if '_errors' in col])
for col in error_cols:
    xpdf[col] = xpdf[col].astype(str)
    xpdf[col] = xpdf[col].apply(lambda x: x.replace('nan', ''))

logger.debug(f'Saving experiment table to {prog.database}')

# def commit_to_db(conn, df, table_name, primary_key):
# c = conn.cursor()
# c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
# if not c.fetchone():
# raise ValueError(f'{table_name} table not found in db')
# # check if the table has the right columns
# c.execute(f'PRAGMA table_info({table_name});')
# columns = [col[1] for col in c.fetchall()]
# if not all([col in columns for col in df.columns]):
# missing_cols = [col for col in df.columns if col not in columns]
# raise ValueError(f'{table_name} table is missing columns {missing_cols}')

# # check that our df has unique names
# if len(df[primary_key].unique()) != len(df):
# raise ValueError(f'{table_name} table has non-unique {primary_key} values')

# # insert or ignore, then update
# strdf = df.astype(str)
# # put the name column last
# strdf = cm.reorder_columns_back(strdf, [primary_key])
# colnames = strdf.columns.tolist()
# c.executemany(
# f'INSERT OR IGNORE INTO {table_name} ({", ".join(colnames)}) VALUES ({", ".join(["?"] * len(colnames))});',
# strdf.values.tolist(),
# )

# conn.commit()
# # update
# c.executemany(
# f'UPDATE {table_name} SET {", ".join([col + " = ?" for col in colnames[:-1]])} WHERE name = ?;',
# strdf.values.tolist(),
# )
# conn.commit()
# conn.close()


from psycopg2 import sql


def commit_to_db(conn, df, table_name, primary_key):
    cursor = conn.cursor()
    try:
        # Check if the table has the right columns
        cursor.execute(
            sql.SQL("SELECT column_name FROM information_schema.columns WHERE table_name = %s;"),
            (table_name,),
        )
        columns = [col[0] for col in cursor.fetchall()]
        if not all(col in columns for col in df.columns):
            missing_cols = [col for col in df.columns if col not in columns]
            cursor.close()
            raise ValueError(f'{table_name} table is missing columns {missing_cols}')

        # Check that our df has unique primary key values
        if len(df[primary_key].unique()) != len(df):
            cursor.close()
            raise ValueError(f'{table_name} table has non-unique {primary_key} values')

        # Prepare data for insertion and update
        strdf = df.astype(str)
        colnames = strdf.columns.tolist()
        records = strdf.values.tolist()

        # Insert or ignore
        insert_query = sql.SQL("INSERT INTO {} ({}) VALUES ({}) ON CONFLICT DO NOTHING;").format(
            sql.Identifier(table_name),
            sql.SQL(', ').join(map(sql.Identifier, colnames)),
            sql.SQL(', ').join(sql.Placeholder() * len(colnames)),
        )
        cursor.executemany(insert_query, records)
        conn.commit()

        # Update
        update_cols = sql.SQL(', ').join(
            [
                sql.SQL("{} = {}").format(sql.Identifier(col), sql.Placeholder())
                for col in colnames
                if col != primary_key
            ]
        )
        update_query = sql.SQL("UPDATE {} SET {} WHERE {} = {};").format(
            sql.Identifier(table_name), update_cols, sql.Identifier(primary_key), sql.Placeholder()
        )
        cursor.executemany(update_query, records)
        conn.commit()

    except Exception as e:
        # Rollback the transaction on error
        conn.rollback()
        cursor.close()
        raise e

    else:
        # Commit the transaction if no errors
        conn.commit()

    finally:
        # Always close the cursor
        if cursor is not None:
            cursor.close()

    cursor.close()


commit_to_db(DBCONN, xpdf, 'experiment', 'name')

logger.info(f'Experiment table saved to {prog.database}')

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                  --     create and update netdf     --
netdf = pd.DataFrame(all_networks)
netdf = netdf.drop(columns=['network'])

# compute unique names
unique_names = []
for i, row in netdf.iterrows():
    n = f'{row["recipe_name"]}_{row["xp"]}_{"-".join(row["markers"].split(", "))}'
    unique_names.append(n)
netdf['name'] = unique_names
n_names = len(netdf['name'].unique())
if n_names != len(netdf):
    raise ValueError(f'found {len(netdf)} networks, but {n_names} unique names')

# check that each xp exists in the experiment table
for xp in netdf['xp'].unique():
    if xp not in xpdf.index:
        raise ValueError(f'xp {xp} not found in experiment table')

logger.debug(f'Saving network table to {prog.database}')

commit_to_db(DBCONN, netdf, 'network', 'name')
DBCONN.close()

logger.info(f'Network table saved to {prog.database}')

DBCONN.close()
##────────────────────────────────────────────────────────────────────────────}}}
