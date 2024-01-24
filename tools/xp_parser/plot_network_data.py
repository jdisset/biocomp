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

DEFAULT_OUTPUT_DIR = Path('./data_plots').resolve()

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

# for database mode we need a database file
prog.add_argument('--database', type=str, help='path to database file')
# and a network id (or list of ids, or 'all')
prog.add_argument(
    '--network_id', help='network id to plot: int, list of ints, or "all"', default='all'
)

# for recipe mode we need a recipe file and a data file
prog.add_argument('--recipe_file', type=str, help='path to recipe file')
prog.add_argument('--data_file', type=str, help='path to data file')

prog.add_argument('--data_config', type=str, default=DEFAULT_DATA_CONFIG_PATH)
prog.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)

prog.parse_args(['--database', 'devtmp/database.xlsx'])

if (
    prog.args.database is not None
    and (prog.args.recipe_file is not None or prog.args.data_file is not None)
) or (prog.args.database is None and prog.args.recipe_file is None and prog.args.data_file is None):
    raise ValueError('You must provide EITHER a database file or a recipe file and a data file')

if prog.args.recipe_file is not None or prog.args.data_file is not None:
    raise NotImplementedError('Direct recipe + data not implemented yet, please provide a database')

if prog.args.database is not None:
    netdf = cm.load_database_table(prog.args.database, 'network')
    xpdf = cm.load_database_table(prog.args.database, 'experiment')
    prog.database_mode = True

if prog.args.network_id == 'all':
    net_ids = netdf['id'].tolist()
else:
    net_ids = int(prog.args.network_id)

prog.args.output_dir = Path(prog.args.output_dir).resolve()
# make dir if needed:
prog.args.output_dir.mkdir(parents=True, exist_ok=True)

if prog.data_config is None:
    prog.data_config = DEFAULT_DATA_CONFIG
else:
    import json5

    prog.data_config = json5.load(open(prog.data_config, 'r'))


def get_network_row(netdf, net_id):
    if nid not in netdf['id']:
        raise ValueError(f'Network id {nid} not found in database')
    net_row = netdf[netdf['id'] == net_id]
    if len(net_row) > 1:
        raise ValueError(f'Network id {nid} is not unique in database')
    return net_row.iloc[0]


def get_recipe_and_data_filepaths(data_file, recipe_file, path_prefix=''):
    # check data file present
    if pd.isna(data_file):
        raise ValueError(f'Data file information for network id {nid} is missing')
    data_file = Path(path_prefix) / data_file
    data_file = Path(data_file).resolve()
    if not Path(data_file).exists():
        raise ValueError(f'Data file {data_file} not found')

    # check recipe file present
    if pd.isna(recipe_file):
        raise ValueError(f'Recipe file information for network id {nid} is missing')
    recipe_file = Path(path_prefix) / recipe_file
    recipe_file = Path(recipe_file).resolve()
    if not Path(recipe_file).exists():
        raise ValueError(f'Recipe file {recipe_file} not found')

    return recipe_file, data_file


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                           --     utils     --
def parse_list(input_string):
    if input_string is None:
        return []
    if isinstance(input_string, list):
        return input_string
    # Split the string by comma and then strip whitespaces from each element
    return [element.strip() for element in input_string.split(',')]


def load_network_from_database(
    netdf, net_id, lib, path_prefix='/Users/jeandisset/Dropbox (MIT)/Biocomp'
):
    row = get_network_row(netdf, net_id)
    rfile, dfile = get_recipe_and_data_filepaths(
        row['data_file'], row['recipe_file'], path_prefix=path_prefix
    )
    networks = bc.recipe.network_from_recipe(rfile, lib, inverse='all')
    # we potentially have several networks, one for each possible inversion
    # we can use the markers to select the right one
    markers = [set(bc.recipe.escape(n.get_inverted_input_proteins())) for n in networks]
    # outputs = [set(bc.recipe.escape(networks[0].get_output_proteins())) - m for m in markers]

    if prog.database_mode:
        target_markers = set(parse_list(row['markers']))
        escaped_target_markers = bc.recipe.escape(target_markers)
        network = networks[markers.index(escaped_target_markers)]

    X, Y = bc.recipe.get_network_XY(network, dfile, color_aliases=protein_aliases)
    return network, X, Y


def get_network_nb_inputs(dman, net_id, net_id_to_dman_id):
    assert net_id in net_id_to_dman_id
    dmanid = net_id_to_dman_id[net_id]
    network = dman.get_networks()[dmanid]
    return network.get_nb_inputs()


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                --     load networks and data     --
if prog.network_id == 'all' and prog.database_mode:
    net_with_data = netdf[netdf['data_file'].notna()]
    net_ids = net_with_data['id'].tolist()
else:
    if isinstance(net_ids, int):
        net_ids = [net_ids]
    net_ids = [int(nid) for nid in net_ids]

net_id_to_dman_id, load_errors = {}, {}
networks, Xs, Ys = [], [], []

for nid in tqdm(list(net_ids), desc='Loading networks'):
    net_id_to_dman_id[nid] = len(networks)
    try:
        network, X, Y = load_network_from_database(netdf, nid, lib, path_prefix=path_prefix)
        networks.append(network)
        Xs.append(X)
        Ys.append(Y)
    except Exception as e:
        load_errors[nid] = f'{e.__class__.__name__}: {e}'

dman = du.DataManager(Xs, Ys, networks, data_cfg=prog.data_config)

load_errors

##────────────────────────────────────────────────────────────────────────────}}}
### {{{                       --     plot function     --


def make_network_title(netdf, net_id):
    assert net_id in netdf['id']
    net_row = netdf[netdf['id'] == net_id]
    assert len(net_row) == 1
    net_row = net_row.iloc[0]
    title = r"\fontsize{12}{12}\selectfont " + net_row['recipe_name'] + '\n'
    title += r"\fontsize{8}{8}\selectfont from " + net_row['xp'] + '\n'
    return title


# n_inputs = {nid: get_network_nb_inputs(dman, nid, net_id_to_dman_id) for nid in net_ids}


def plot_network_data(dman, net_id, net_id_to_dman_id, extra_args=None):
    plot_title = make_network_title(netdf, net_id)
    assert net_id in net_id_to_dman_id
    dmanid = net_id_to_dman_id[net_id]
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
                pu.network_plot(
                    dman, dmanid, axes=axes, input_order=input_order, **plot_config
                )
    fig.suptitle(plot_title, fontsize=12)
    return fig


##────────────────────────────────────────────────────────────────────────────}}}

plot_errors = {}
def plot_and_save(net_id, **kw):
    global dman
    global prog
    global plot_errors
    global load_errors
    global net_id_to_dman_id
    if net_id not in load_errors:
        try:
            f = plot_network_data(
                dman,
                net_id,
                net_id_to_dman_id,
                **kw
            )
            fpath = prog.args.output_dir / f'{net_id}.png'
            f.savefig(
                fpath,
                bbox_inches='tight',
                pad_inches=0.05,
                dpi=300,
            )
            plt.close(f)
        except Exception as e:
            print(f'Error plotting {net_id}: {e}')
            plot_errors[net_id] = e

for net_id in tqdm(net_ids[:]):
    plot_and_save(net_id, extra_args={'method': 'smooth'})

