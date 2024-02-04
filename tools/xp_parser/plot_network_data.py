### {{{                          --     imports     --
import sys

import urllib.parse
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
import biocomp.plotutils as pu
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

##────────────────────────────────────────────────────────────────────────────}}}

"""
Utils for plotting data in various ways, from the network representation of an experiment.
It can build this network from scratch given a recipe, library and data file, or it can
use a database file to load the network and plot it.
"""

### {{{                   --     constants and config     --
lib = ut.load_lib()
protein_aliases = {'EBFP': 'EBFP2', 'L0.G_MNEONGREEN': 'MNEONGREEN'}
path_prefix = '/Users/jeandisset/Dropbox (MIT)/Biocomp'

DEFAULT_OUTPUT_DIR = Path('./biocomp-static/dataplots').resolve()

BASE_DEFAULT_CONFIG = {
    'xlims': (-0.027, 0.8),
    'ylims': (-0.027, 0.8),
    'log_density': True,
    'size': (4, 4),
    'skip_ticklabel_range': (0.0, 101),
}

DEFAULT_1D_CONFIG = {
    'method': 'histogram',
}

DEFAULT_2D_CONFIG = {
    'method': 'smooth',
}

DEFAULT_3D_CONFIG = {
    'xlims': (-0.027, 0.85),
    'ylims': (-0.027, 0.85),
    'vlims': (-0.027, 0.85),
    'method': 'smooth',
    'slices': (0.1, 0.3, 0.5),
    'radius': 0.11,
    'knn': 500,
    'min_points': 20,
}

##────────────────────────────────────────────────────────────────────────────}}}
### {{{                --     arg declaration and parsing     --
prog = cm.CLIProgram()

# and a network id (or list of ids, or 'all')
prog.add_argument(
    '--network_id', help='network id to plot: int, list of network ids, or "all"', default='all'
)

prog.add_argument('--data_config', type=str, default=DEFAULT_DATA_CONFIG_PATH)
prog.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)

prog.parse_args()

DBCONN = cm.connect_to_db()

netdf = cm.load_table_as_dataframe(DBCONN, 'network')
xpdf = cm.load_table_as_dataframe(DBCONN, 'experiment')

prog.database_mode = True

if prog.args.network_id == 'all':
    net_names = netdf['name'].tolist()
else:
    raise NotImplementedError('Only "all" is supported for network_id right now')

prog.args.output_dir = Path(prog.args.output_dir).resolve()
prog.args.output_dir.mkdir(parents=True, exist_ok=True)

if prog.data_config is None:
    prog.data_config = DEFAULT_DATA_CONFIG
else:
    import json5

    assert Path(prog.data_config).exists()
    prog.data_config = json5.load(open(prog.data_config, 'r'))






##────────────────────────────────────────────────────────────────────────────}}}
### {{{                           --     utils     --

def get_network_nb_inputs(dman, net_id, net_id_to_dman_id):
    assert net_id in net_id_to_dman_id
    dmanid = net_id_to_dman_id[net_id]
    network = dman.get_networks()[dmanid]
    return network.get_nb_inputs()


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                --     load networks and data     --

net_with_data = netdf[netdf['data_file'].notna()]
net_names = net_with_data['name'].tolist()

net_name_to_dman_id, load_errors = {}, {}
networks, Xs, Ys = [], [], []

for net_name in tqdm(list(net_names), desc='Loading networks'):
    net_name_to_dman_id[net_name] = len(networks)
    try:
        network, X, Y = cm.load_network_and_data(netdf, net_name, lib, path_prefix=path_prefix)
        networks.append(network)
        Xs.append(X)
        Ys.append(Y)
    except Exception as e:
        load_errors[net_name] = f'{e.__class__.__name__}: {e}'

dman = du.DataManager(Xs, Ys, networks, data_cfg=prog.data_config)

load_errors

##────────────────────────────────────────────────────────────────────────────}}}
### {{{                       --     plot function     --


def make_network_title(netdf, net_id):
    assert net_id in netdf['name'].values
    net_row = netdf[netdf['name'] == net_id]
    assert len(net_row) == 1
    net_row = net_row.iloc[0]
    title = r"\fontsize{12}{12}\selectfont " + net_row['recipe_name'] + '\n'
    title += r"\fontsize{8}{8}\selectfont from " + net_row['xp'] + '\n'
    return title


def plot_network_data(dman, net_name, net_name_to_dman_id, extra_args=None):
    plot_title = make_network_title(netdf, net_name)
    assert net_name in net_name_to_dman_id
    dmanid = net_name_to_dman_id[net_name]
    network = dman.get_networks()[dmanid]
    n_inputs = network.get_nb_inputs()

    extra_args = extra_args or {}
    plot_config = BASE_DEFAULT_CONFIG

    ax, axes = None, None

    if n_inputs == 1:
        plot_config = ut.updated_dict(plot_config, DEFAULT_1D_CONFIG)
    elif n_inputs == 2:
        plot_config = ut.updated_dict(plot_config, DEFAULT_2D_CONFIG)
    elif n_inputs == 3:
        plot_config = ut.updated_dict(plot_config, DEFAULT_3D_CONFIG)
    else:
        raise NotImplementedError(f'Plotting {n_inputs} inputs is not implemented')
    plot_config = ut.updated_dict(plot_config, extra_args)

    input_order = plot_config.get('input_order', None)
    if 'input_order' in plot_config:
        del plot_config['input_order']

    fig = None
    if n_inputs <= 2:
        fig, ax = pu.mkfig(1, 1, size=plot_config['size'])
        if input_order is None:
            input_order = list(range(n_inputs))
        pu.network_plot(dman, dmanid, ax=ax, input_order=input_order, **plot_config)
    else:
        if 'slices' not in plot_config:
            raise ValueError('You must specify slices for 3D plots')
        if plot_config['method'] == 'smooth':
            nslices = len(plot_config['slices'])
            if input_order is None:
                # we plot every ordering
                fig, axes = pu.mkfig(n_inputs, nslices, size=plot_config['size'])
                for i in range(n_inputs):
                    iorder = list(range(n_inputs))
                    iorder = iorder[i:] + iorder[:i]
                    pu.network_plot(
                        dman, dmanid, axes=axes[i, :], input_order=iorder, **plot_config
                    )
            else:
                fig, axes = pu.mkfig(1, nslices, size=plot_config['size'])
                pu.network_plot(dman, dmanid, axes=axes, input_order=input_order, **plot_config)
    fig.suptitle(plot_title, fontsize=12)
    return fig


##────────────────────────────────────────────────────────────────────────────}}}

encoded_names = {net_name: cm.get_name_hash(net_name) for net_name in net_names}
plot_errors = {}


def plot_and_save(net_name, **kw):
    global dman
    global prog
    global plot_errors
    global load_errors
    global net_name_to_dman_id
    if net_name not in load_errors:
        try:
            fpath = prog.args.output_dir / f'{encoded_names[net_name]}.png'
            # if already exists, skip
            if fpath.exists():
                print(f'File {fpath} already exists, skipping')
                return

            f = plot_network_data(dman, net_name, net_name_to_dman_id, **kw)
            f.savefig(
                fpath,
                bbox_inches='tight',
                pad_inches=0.05,
                dpi=300,
            )
            plt.close(f)
        except Exception as e:
            print(f'Error plotting {net_name}: {e}')
            plot_errors[net_name] = e


for net_name in tqdm(net_names[:]):
    plot_and_save(net_name, extra_args={'method': 'smooth'})


# add the path in the data_plot_path column of the network table, if plot_errors is empty

##
base_dir_url = 'dataplots'
DBCONN = cm.connect_to_db()
with DBCONN.cursor() as cursor:
    try:
        sql = 'UPDATE network SET data_plot = %s WHERE name = %s'
        for net_name in net_names:
            if net_name not in plot_errors and net_name not in load_errors:
                fname = f'{base_dir_url}/{encoded_names[net_name]}.png'
                print(f'Updating network {net_name} with plot path {fname}')
                cursor.execute(sql, (fname, net_name))
            else:
                cursor.execute(sql, (None, net_name))
    except Exception as e:
        print(f'Error updating network table: {e}')
        raise e
    finally:
        DBCONN.commit()

DBCONN.close()


##
# fetch list of network data plots from db:
DBCONN = cm.connect_to_db()
with DBCONN.cursor() as cursor:
    try:
        cursor.execute('SELECT name, data_plot FROM network')
        rows = cursor.fetchall()
    except Exception as e:
        print(f'Error fetching network data plots: {e}')
        raise e
    finally:
        DBCONN.close()

rows = [r for r in rows if r[1] is not None]
##

# check that all files exist
for r in rows:
    fpath = Path('biocomp-static')/r[1]
    if not fpath.exists():
        print(f'File {fpath} does NOT exist!!')
    else:
        print(f'File {fpath} DOES exists')
