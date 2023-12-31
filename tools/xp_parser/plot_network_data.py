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

lib = ut.load_lib()
protein_aliases = {'EBFP': 'EBFP2', 'L0.G_MNEONGREEN': 'MNEONGREEN'}
path_prefix = '/Users/jeandisset/Dropbox (MIT)/Biocomp'

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


if prog.network_id == 'all' and prog.database_mode:
    net_with_data = netdf[netdf['data_file'].notna()]
    net_ids = net_with_data['id'].tolist()
else:
    if isinstance(net_ids, int):
        net_ids = [net_ids]
    net_ids = [int(nid) for nid in net_ids]

net_ids = net_ids[5:126]

net_id_to_dman_id = {}
networks, Xs, Ys = [], [], []
for nid in tqdm(list(net_ids), desc='Loading networks'):
    net_id_to_dman_id[nid] = len(networks)
    network, X, Y = load_network_from_database(netdf, nid, lib, path_prefix=path_prefix)
    networks.append(network)
    Xs.append(X)
    Ys.append(Y)

dman = du.DataManager(Xs, Ys, networks, data_cfg=prog.data_config)

net_id_to_dman_id

##

BASE_DEFAULT_CONFIG = {
    'xlims': (-.027, 0.8),
    'ylims': (-.027, 0.8),
    'log_density': True,
    'size': (4, 4),
    'skip_ticklabel_range':(0.0,101),
}

DEFAULT_1D_CONFIG = {
    'method': 'histogram',
}

DEFAULT_2D_CONFIG = {
    'method': 'smooth',
}

DEFAULT_3D_CONFIG = {
    'xlims': (-.027, 0.85),
    'ylims': (-.027, 0.85),
    'vlims': (-.027, 0.85),
    'method': 'smooth',
    'slices': (0.1, 0.3, 0.5),
    'radius': 0.11,
    'knn': 500,
    'min_points': 20,
}

def make_network_title(netdf, net_id):
    assert net_id in netdf['id']
    net_row = netdf[netdf['id'] == net_id]
    assert len(net_row) == 1
    net_row = net_row.iloc[0]
    title = r"\fontsize{12}{12}\selectfont " + net_row['recipe_name'] + '\n'
    title += r"\fontsize{8}{8}\selectfont from " + net_row['xp'] + '\n'
    return title

net_id = 172

def get_network_nb_inputs(dman, net_id, net_id_to_dman_id):
    assert net_id in net_id_to_dman_id
    dmanid = net_id_to_dman_id[net_id]
    network = dman.get_networks()[dmanid]
    return network.get_nb_inputs()

n_inputs = {nid: get_network_nb_inputs(dman, nid, net_id_to_dman_id) for nid in net_ids}

n_inputs

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

    if n_inputs <= 2:
        fig, ax = pu.mkfig(1, 1, size=plot_config['size'])
    else:
        if 'slices' not in plot_config:
            raise ValueError('You must specify slices for 3D plots')
        if plot_config['method'] == 'smooth':
            nslices = len(plot_config['slices'])
            fig, axes = pu.mkfig(1, nslices, size=plot_config['size'])

    plot_config = ut.updated_dict(plot_config, extra_args)
    pu.network_plot(dman, dmanid, ax=ax, axes=axes, **plot_config)
    # remove any existing title
    fig.suptitle(plot_title, fontsize=12)

plot_network_data(
    dman,
    net_id,
    net_id_to_dman_id,
    extra_args={'method':'scatter'},
)




### {{{                        --     cube smooth     --
dmanid = net_id_to_dman_id[net_id]
network = dman.get_networks()[dmanid]
x, y = dman.get_X()[dmanid], dman.get_Y()[dmanid]
rescaler = pu.DataRescaler.from_data_manager(dman)
porder, pnames = pu.get_reordered_protein_names(network)

plot_config = BASE_DEFAULT_CONFIG
plot_config = ut.updated_dict(plot_config, DEFAULT_3D_CONFIG)

nslices = len(plot_config['slices'])
# fig, axes = pu.mkfig(1, nslices, size=plot_config['size'])
slices = (0.1, 0.3, 0.5)

slice_images = []

import matplotlib.transforms as mtransforms
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
from tempfile import mkdtemp
from os import path


cmap=pu.DEFAULT_CMAP
bad_color='#EEEEEE00'
res=100
vlims=(-.027, 0.85)
contours=3
kw={}

# ax = axes[0]
fig = plt.figure(figsize=(4, 4), dpi=300)
ax = fig.add_subplot(111, projection='3d')

def style_3d(ax):
    fig = ax.get_figure()
    fig.patch.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # long thin ticks
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.25)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.get_xaxis().tick_bottom()
    # font
    ax.tick_params(axis='both', which='both', labelsize=8)
    ax.tick_params(axis='both', which='major', length=5, width=0.4)
    ax.tick_params(axis='both', which='minor', length=2, width=0.12)
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)
    # tick outside
    ax.tick_params(axis='both', which='both', direction='out')
    # spine color
    ax.spines['bottom'].set_color('#777777')
    ax.spines['left'].set_color('#777777')

style_3d(ax)

for i, s in enumerate(slices):
    xslice = np.asarray([s])

    protein_order, protein_names = pu.get_reordered_protein_names(network, **kw)
    input_order, output_pos = protein_order[:-1], protein_order[-1]
    input_names, output_name = protein_names[:-1], protein_names[-1]
    xy_grid, output_values, opacities = pu.prepare_smooth_2d(
        x, y, network, input_names, input_order, output_pos, res, xlims, xslice, **kw
    )

    cmap = plt.get_cmap(cmap)
    cmap.set_bad(color=bad_color)

    xres = len(np.unique(xy_grid[:, 0]))
    yres = len(np.unique(xy_grid[:, 1]))

    xlims = np.array([xy_grid[:, 0].min(), xy_grid[:, 0].max()])
    ylims = np.array([xy_grid[:, 1].min(), xy_grid[:, 1].max()])

    vmin, vmax = vlims
    vmin = vmin if vmin is not None else np.nanmin(output_values)
    vmax = vmax if vmax is not None else np.nanmax(output_values)

    Z = output_values.reshape((xres, yres))
    opacities = np.ones_like(Z) if opacities is None else opacities.reshape((xres, yres))

    X, Y = np.meshgrid(np.linspace(*xlims, num=xres), np.linspace(*ylims, num=yres))

    Z_coord = np.ones_like(X) * s

    # Create an RGBA color array
    colors = cmap(Z / vmax)
    alpha_multiplier = 1 if (i == 1) else 0.1
    colors[..., -1] *= alpha_multiplier * opacities

    # Plot the surface with the RGBA colors
    ax.plot_surface(X, Y, Z_coord, facecolors=colors, rstride=1, cstride=1)

    # Add contour lines if needed
    # if contours is not None:
        # ax.contour(X, Y, Z, zdir='y', offset=s, levels=contours, linestyles="solid", linewidths=0.25, alpha=alpha_multiplier*0.5)


    ax.invert_zaxis()
    ax.view_init(elev=20, azim=45, vertical_axis='y')
    pu.setup_transformed_axis(ax, xlims, ylims, rescaler)
    # no grid:
    ax.grid(False)


# def setup_transformed_axis(
    # ax, xaxis_lims=None, yaxis_lims=None, rescaler=None, margins=0.05, transform=None, **kw
# ):
    # if xaxis_lims is not None:
        # xaxis_lims = setup_transformed_xaxis(ax, xaxis_lims, rescaler, margins=margins, **kw)
    # if yaxis_lims is not None:
        # yaxis_lims = setup_transformed_yaxis(ax, yaxis_lims, rescaler, margins=margins, **kw)
    # return xaxis_lims, yaxis_lims


##────────────────────────────────────────────────────────────────────────────}}}

