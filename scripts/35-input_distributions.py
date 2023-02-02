### {{{                          --     imports     --
import datetime
import biocomp as bc
import matplotlib.pyplot as plt
import numpy as np
import time
from functools import partial
import biocomp.utils as bu
import scriptutils as ut
import jax
from jax import jit, vmap, grad, value_and_grad
import jax.numpy as jnp
from jax.scipy.stats import gaussian_kde
import biocomp.datautils as du
import optax
from pathlib import Path
from tqdm import tqdm
import biocomp.nodes as bn
import biocomp.compute as bcc
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [7.0, 7.0]
plt.rcParams['figure.dpi'] = 300

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                         --     load xps     --
lib = ut.load_lib()

xp = ut.load_xp('2023-01-22_CasE_ALLuORFs', lib)
dman = du.DataManager.from_xps([xp])

uorf_xp = ut.load_xp('2022-11-10_uORFs_and_company', lib)
uorf_dman = du.DataManager.from_xps([uorf_xp])

# create dict of dman for data in:
# RESULTS_PRE-BEADS
# RESULTS_PRE-MEFL_AJ01
# RESULTS_PRE-MEFL_AL01
# RESULTS_TASBE_AJ01
# RESULTS_TASBE_AL01

all_dman = {
    'prebeads': du.DataManager.from_xps(
        [ut.load_xp('2023-01-22_CasE_ALLuORFs', lib, data_path='./data/RESULTS_PRE-BEADS')]
    ),
    'premefl_aj01': du.DataManager.from_xps(
        [ut.load_xp('2023-01-22_CasE_ALLuORFs', lib, data_path='./data/RESULTS_PRE-MEFL_AJ01')]
    ),
    'premefl_al01': du.DataManager.from_xps(
        [ut.load_xp('2023-01-22_CasE_ALLuORFs', lib, data_path='./data/RESULTS_PRE-MEFL_AL01')]
    ),
    'tasbe_aj01': du.DataManager.from_xps(
        [ut.load_xp('2023-01-22_CasE_ALLuORFs', lib, data_path='./data/RESULTS_TASBE_AJ01')]
    ),
    'tasbe_al01': du.DataManager.from_xps(
        [ut.load_xp('2023-01-22_CasE_ALLuORFs', lib, data_path='./data/RESULTS_TASBE_AL01')]
    ),
    'ctrls': du.DataManager.from_xps(
        [ut.load_xp('2023-01-22_CasE_ALLuORFs', lib, data_path='./data/RESULTS_PRE-BEADS-Color_Controls')]
    ),

}


ern_xp = ut.load_xp('20220501-GW-l1vsl2', lib)
ern_dman = du.DataManager.from_xps([ern_xp])

mass_xp = ut.load_xp('E20221012A_massCtrls', lib)
mass_dman = du.DataManager.from_xps([mass_xp])

mnames = [m.node_namespace for m in dman.get_models()]
uorf_mnames = [m.node_namespace for m in uorf_dman.get_models()]
ern_mnames = [m.node_namespace for m in ern_dman.get_models()]
mass_mnames = [m.node_namespace for m in mass_dman.get_models()]
# ut.plot_networks([m.network for m in mass_dman.get_models()])

ern_mnames

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                      --     plot functions     --


def remove_axis_and_spines(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def set_size(w, h, axes, fig):
    """w, h: width, height in inches"""
    l = min([ax.figure.subplotpars.left for ax in axes])
    r = max([ax.figure.subplotpars.right for ax in axes])
    t = max([ax.figure.subplotpars.top for ax in axes])
    b = min([ax.figure.subplotpars.bottom for ax in axes])
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    fig.set_size_inches(figw, figh)


def plot_fluo_distributions(dman, mid, xrange=10, res=2000, title=None, fname=None):
    from jax.scipy.stats import gaussian_kde

    model = dman.get_models()[mid]
    rawx = dman.get_raw_X()[mid]
    rawy = dman.get_raw_Y()[mid]
    # ut.plot_networks([model.network])
    input_names = model.get_inverted_input_proteins()
    reordered_input = sorted(input_names)
    output_names = model.get_output_proteins()
    output = list(set(output_names) - set(input_names))
    output_pos = output_names.index(output[0])
    if reordered_input != input_names:
        rawx = rawx[:, [input_names.index(i) for i in reordered_input]]

    rawx = jnp.hstack([rawx, rawy[:, output_pos][:, None]])
    # remove rows where any value is negative
    rawx = rawx[~jnp.any(rawx < 0, axis=1)]
    pnames = reordered_input + output
    types = ['in'] * len(reordered_input) + ['out']

    fig, axes = plt.subplots(1, len(pnames), figsize=(1.25 * len(pnames), 15))

    ticks = np.arange(0, xrange + 1, 1)
    XX = jnp.linspace(0, xrange, res)
    for xid, ax in enumerate(axes):
        color = {'eBFP': '#529edb', 'eYFP': '#fbda73', 'mKate': '#f75a5a', 'NeonGreen': '#33f397'}[
            pnames[xid]
        ]
        logx = jnp.log10(rawx[:, xid])
        kde = gaussian_kde(logx.T, bw_method=0.0075)
        smoothkde = gaussian_kde(logx.T, bw_method=0.1)
        densities = kde(XX.T)
        densities = (densities / densities.max()) * 0.4
        smoothdensities = smoothkde(XX.T)
        smoothdensities = (smoothdensities / smoothdensities.max()) * 0.4
        maxd = XX[smoothdensities.argmax()]
        ax.plot(densities, XX, color='k', alpha=1, lw=0.5)
        ax.plot(-smoothdensities, XX, color='k', alpha=1, lw=0.5)
        # ax.plot(densities, XX, color='k', alpha=1, lw=0.25)
        # ax.plot(-densities, XX, color='k', alpha=1, lw=0.25)
        q1 = jnp.quantile(logx, 0.005)
        q9 = jnp.quantile(logx, 0.995)
        ax.axhline(q1, color=color, alpha=1, lw=1)
        ax.axhline(q9, color=color, alpha=1, lw=1)
        ax.axhspan(q1, q9, color='k', alpha=0.075, lw=0)
        ax.axhline(maxd, color='k', alpha=1, lw=1)
        ax.fill_betweenx(XX, -smoothdensities, 0, color=color, alpha=1, lw=0)
        ax.fill_betweenx(XX, 0, densities, color=color, alpha=1, lw=0)
        ax.axvline(0, color='k', alpha=0.5, lw=0.5, dashes=(10, 10), dash_capstyle='round')
        ax.set_aspect("equal")
        ax.set_ylim(0, xrange)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xlabel(f'{pnames[xid]} [{types[xid]}]', rotation=0, labelpad=20, fontsize=10)
        for t in ticks:
            ax.axhline(
                t,
                xmin=-0.2,
                xmax=1,
                c='#777777',
                linewidth=0.2,
                zorder=0,
                clip_on=False,
                alpha=1,
                dashes=(10, 20),
                dash_capstyle='round',
            )

        remove_axis_and_spines(ax)

    # share x axis, only show on last
    axes[0].set_yticks(ticks)
    # display the real log10 values
    xlabels = [du.scformat.format("{:m}", x) for x in 10**ticks]
    axes[0].set_yticklabels(xlabels)
    axes[0].tick_params(axis='y', which='both', length=0, pad=30)
    # offset labels by 0.5 towards bottom
    # axes[-1].yaxis.set_tick_params(pad=30)
    for tick in axes[0].yaxis.get_major_ticks():
        tick.label.set_fontsize(8)
        tick.label.set_color('grey')

    mname = model.node_namespace
    if title is None:
        title = (f'Fluorescence distributions for \n{mname}\n\n({len(rawx)} points)',)
    fig.suptitle(
        title,
        fontsize=10,
        y=0.85,
        x=0.45,
    )
    if fname is None:
        fname = f'{mname}_fluodist.png'
    savepath = Path(f'~/Desktop/figures/{fname}').expanduser()
    savepath.parent.mkdir(parents=True, exist_ok=True)

    set_size(1.25 * len(pnames), 15, axes, fig)
    fig.savefig(savepath, bbox_inches='tight', dpi=200)


##────────────────────────────────────────────────────────────────────────────}}}

### {{{              --     plotting some xp distributions     --
# plot_fluo_distributions(ern_dman, 0)
# plot_fluo_distributions(mass_dman, 0)
# plot_fluo_distributions(dman, 1)
# plot_fluo_distributions(uorf_dman, 0)
# plot_fluo_distributions(uorf_dman, 8)

for k, v in all_dman.items():
    title = f'Fluorescence distributions \n{k}'
    print(title)
    plot_fluo_distributions(v, 1, title=title, fname=f'{k}_fluodist.png')

x = all_dman['prebeads'].get_raw_X()[1]

##
import pandas as pd

ctrls = ut.DEFAULT_XP_PATH / xpname / 'data/RESULTS_PRE-BEADS-Color_Controls/ALL.2023-01-22_CasE_ALLuORFs.csv'
ctrls = pd.read_csv(ctrls)
##

# plot kde of log data for each channel
fig, axes = du.mkfig(1,3, (6,3))
for i, column in enumerate(ctrls.columns):
    data = ctrls[column]
    data = data.to_numpy()
    data = data[data > 0]
    logdata = np.log10(data)
    kde = gaussian_kde(logdata, bw_method=0.01)
    x = np.linspace(0, 5, 1000)
    ax = axes[i]
    y = kde(x)
    y = y / np.max(y)
    ax.plot(x, y)
    ax.set_title(ctrls.columns[i])
    ax.set_xlabel('log10')
    ax.set_ylabel('density')





##────────────────────────────────────────────────────────────────────────────}}}

import flowio

xpname = '2023-01-22_CasE_ALLuORFs'
beadfcs = ut.DEFAULT_XP_PATH / xpname / 'data/FCS_FILES/2023-01-22_CasE_ALLuORFs_BEADS_AJ01_018.fcs'
fcs_data = flowio.FlowData(beadfcs.as_posix())

# fcs_data.channels : 
# {'1': {'PnN': 'Time'},
# '2': {'PnN': 'FSC-A'},
# '3': {'PnN': 'FSC-H'},
# '4': {'PnN': 'FSC-W'},
# '5': {'PnN': 'SSC-A'},
# '6': {'PnN': 'SSC-H'},
# '7': {'PnN': 'SSC-W'},
# '8': {'PnN': 'Pacific Blue-A'},
# '9': {'PnN': 'AmCyan-A'},
# '10': {'PnN': 'FITC-A'},
# '11': {'PnN': 'PerCP-Cy5-5-A'},
# '12': {'PnN': 'PE-A'},
# '13': {'PnN': 'PE-Texas Red-A'},
# '14': {'PnN': 'APC-A'},
# '15': {'PnN': 'APC-Alexa 700-A'},
# '16': {'PnN': 'APC-Cy7-A'}}


# print all fcs_data member variables
fcs_data.__dict__.keys()
data = np.reshape(fcs_data.events, (-1, fcs_data.channel_count))

fcs_data.channels

data.shape

##

from scipy.signal import find_peaks
fig, axes = du.mkfig(3, 5, (5,3))
axes = axes.flatten()
for i, ax in tqdm(enumerate(axes), total=len(axes)):
    x = np.linspace(0, 6, 1000)
    logdata = data[:, i+1]
    logdata = jnp.log10(logdata[logdata > 0])

    kde = gaussian_kde(logdata, bw_method=0.025)
    y = kde(x)
    y /= y.max()
    ax.plot(x, y)

    # smoothkde = gaussian_kde(logdata, bw_method=0.05)
    # smoothy = smoothkde(x)
    # smoothy = np.log10(1+y*20)
    # smoothy /= smoothy.max()
    # peaks, _ = find_peaks(smoothy, height=0.3)
    # for peak in peaks:
        # ax.axvline(x[peak], color='red', linewidth=1, zorder=0, clip_on=False, alpha=1, dashes=(5, 5))

    ax.set_title(fcs_data.channels[str(i+2)]['PnN'])
    ax.set_xlabel('log10')
    ax.set_ylabel('density')
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1.1)

# increase margins between subplots
fig.subplots_adjust(hspace=0.5, wspace=0.25)
fig.suptitle('FCS bead data (with peak amplification)', fontsize=10, y=0.95, x=0.45)



##
import jax
import jax.numpy as jnp

from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

# sample two point clouds and their weights.
rngs = jax.random.split(jax.random.PRNGKey(0), 4)
n, m, d = 12, 14, 2
x = jax.random.normal(rngs[0], (n,d)) + 1
y = jax.random.uniform(rngs[1], (m,d))
a = jax.random.uniform(rngs[2], (n,))
b = jax.random.uniform(rngs[3], (m,))
a, b = a / jnp.sum(a), b / jnp.sum(b)
# Computes the couplings using the Sinkhorn algorithm.
geom = pointcloud.PointCloud(x, y)
prob = linear_problem.LinearProblem(geom, a, b)

solver = sinkhorn.Sinkhorn()
out = solver(prob)
