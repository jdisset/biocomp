### {{{                          --     imports     --
import sys
sys.path.append('../scripts')

from biocomp import utils as ut
import scriptutils as su
import biocomp.datautils as du
import biocomp.plotutils as pu
import time
import biocomp.train as train
import biocomp.compute as cmp
import biocomp.parameters as pm
import biocomp as bc
from biocomp.parameters import ParameterTree
from jax.tree_util import Partial as partial
from matplotlib import pyplot as plt
import jax.tree_util as jtu
from pathlib import Path
import jax.numpy as jnp
from copy import deepcopy
import optax
from tqdm import tqdm
import numpy as np
import jax
from jax import jit, grad, vmap, random, value_and_grad
from jax import numpy as jnp
from matplotlib import pyplot as plt

import cProfile


class profiler:
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self.profiler = cProfile.Profile()
        self.profiler.enable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.disable()
        self.profiler.dump_stats(self.filename)


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                      --     load parameters     --
training_archive = du.load('../__results/training_archives/20230923_fulltrain_v0.pkl')
shared_parameters = training_archive['parameters']
compute_config = training_archive['compute_config']
training_config = training_archive['training_config']
compute_config.set_impl('bias', bc.nodes.bias)
##────────────────────────────────────────────────────────────────────────────}}}
### {{{         --     importing, loading, preparing xp data and output directory   --
XPs = ['2023-10-01_Cascades_CCv4', '2023-10-10_Cascades_CCv4_2', '2023-10-16_ConstraintsV2_1_i720']

lib = su.load_lib()
loaded_xps = [
    su.load_xp(
        xp,
        lib,
        data_path='./data/calibrated_data_v3',
        recipe_path=su.DEFAULT_DATA_PATH / 'Experiments' / xp / 'recipes',
    )
    for xp in XPs
]
dman_full = du.DataManager.from_xps(loaded_xps, training_config, inverse='all')

onedrive = Path(
    '~/Library/CloudStorage/OneDrive-SharedLibraries-MassachusettsInstituteofTechnology/'
).expanduser()
plotdir = onedrive / 'Neuromorphic Biocompiler - Documents/Plots'

basedir = plotdir / 'Cascade'
basedir.mkdir(exist_ok=True, parents=True)
##────────────────────────────────────────────────────────────────────────────}}}

### {{{                           --     cmaps     --
import matplotlib.colors as mcolors

cmap = 'Blues'
cmap = 'GnBu'
cmap = 'BuPu'

blues = [
    '#F9F7F5',
    '#EEECEA',
    '#B0CCD6',
    '#6CAFC3',
    '#2974A4',
    '#3B4B90',
    '#3D1277',
    '#22044B',
]

greens = [
    '#F9F7F5',
    '#E2EADA',
    '#CBE4BB',
    '#9DDDAA',
    '#4CCDAB',
    '#30A78F',
    '#1F7D73',
    '#0C5558',
]

reds = [
    '#F5F5F5',
    '#F1E6E5',
    '#F3CFBC',
    '#EF957D',
    '#D3494B',
    '#B00031',
    '#840137',
    '#560140',
]


cmap = mcolors.LinearSegmentedColormap.from_list('cm', blues, N=256)

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


[get_all_ERN_ids(network) for network in dman_full.get_networks()]
[network.name for network in dman_full.get_networks()]
[get_all_ERNs_names(network) for network in dman_full.get_networks()]
[get_all_uorf_values(network) for network in dman_full.get_networks()]
##────────────────────────────────────────────────────────────────────────────}}}

### {{{                  --     heatmaps with x3 slices     --
from matplotlib import rc

rc('text', usetex=True)

# --- parameters
savedir = basedir / 'heatmaps_x3_slices'
slices = [0.1, 0.2, 0.4, 0.5]
vmin = 0.0
vmax = 0.6
xmin = 0.0
xmax = 0.6
protein_aliases = {
    'eBFP': r'$\mathbf{X_1}$ (eBFP2)',
    'mKO2': r'$\mathbf{X_2}$ (mKO2)',
    '1xiRFP720': r'$\mathbf{X_3}$ (iRFP720)',
    'L0.G_mNeonGreen': r'$\mathbf{Y}$ (mNeonGreen)',
}

# plot style definition;
# first, we want ticks inside the plot:

# --- plot
savedir.mkdir(exist_ok=True, parents=True)
n_slices = len(slices)


with ut.timer('plotting'):
    for i in range(len(dman_full.get_networks()))[:1]:
        net = dman_full.get_networks()[i]
        uorf_values = get_all_uorf_values(net)
        ERN_names = get_all_ERNs_names(net)
        net_name = net.name
        if len(ERN_names) != 2:
            continue

        uorf_val = uorf_values[0][1]
        uorf_perc = (uorf_val / 80) * 100
        uorf_name = uorf_dict[uorf_val]

        fig, axes = pu.mkfig(1, len(slices), (2, 2), dpi=200)
        pu.network_plot(
            dman_full,
            i,
            ax=None,
            axes=axes,
            input_order=[0, 1, 2],
            method='smooth',
            slices=slices,
            vmin=vmin,
            vmax=vmax,
            protein_aliases=protein_aliases,
            cmap=cmap,
            xmin=xmin,
            xmax=xmax,
            # contours=[0.1, 0.4],
            contours=None,
        )
        for ax in axes:
            ax.set_title('')

        # as text:
        fig.text(
            0.5,
            1.05,
            f'$ERN_A =$ {ERN_names[0]}, \  $ERN_B =$ {ERN_names[1]}'
            + r', \ $W_{a,b}^{tl} = '
            + f'{uorf_perc:.0f}\%$ ({uorf_name})',
            ha='center',
            va='bottom',
            fontsize=9,
        )

        fig.tight_layout()
        fig.savefig(savedir / f'heatmap_{net_name}.pdf', bbox_inches='tight')


##────────────────────────────────────────────────────────────────────────────}}}
