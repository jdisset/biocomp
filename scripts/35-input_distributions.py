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

mnames = [m.node_namespace for m in dman.get_models()]
# ut.plot_networks([m.network for m in mass_dman.get_models()])


##────────────────────────────────────────────────────────────────────────────}}}

def remove_axis_and_spines(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_fluo_distributions(dman, mid, xrange=10, res=1500):
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
    pnames = reordered_input + output
    types = ['input'] * len(reordered_input) + ['output']


    fig, axes = plt.subplots(len(pnames), 1, figsize=(15, 1.5*len(pnames)))

    ticks = np.arange(0, xrange + 1, 1)
    XX = jnp.linspace(0, xrange, res)
    for xid, ax in enumerate(axes):
        logx = jnp.log10(rawx[:, xid])
        kde = gaussian_kde(logx.T, bw_method=0.0075)
        densities = kde(XX.T)
        densities = (densities / densities.max()) * 0.4
        ax.plot(XX, densities, color='k', alpha=1, lw=0.25)
        ax.plot(XX, -densities, color='k', alpha=1, lw=0.25)
        color = {'eBFP': '#529edb', 'eYFP': '#fbda73', 'mKate': '#f75a5a', 'NeonGreen': '#33f397'}[
            pnames[xid]
        ]
        q1 = jnp.quantile(logx, 0.005)
        q9 = jnp.quantile(logx, 0.995)
        ax.axvline(q1, color=color, alpha=0.5, lw=1)
        ax.axvline(q9, color=color, alpha=0.5, lw=1)
        ax.axvspan(q1, q9, color='k', alpha=0.05, lw=0)
        ax.fill_between(XX, densities, -densities, color=color, alpha=1, lw=0)
        ax.set_aspect("equal")
        ax.set_xlim(0, xrange)
        ax.set_ylim(-0.5, 0.5)
        ax.set_ylabel(f'{pnames[xid]} [{types[xid]}]', rotation=0, ha='right', va='center')
        for t in ticks:
            ax.axvline(x=t,ymin=-0.3,ymax=1,c='k',linewidth=0.3,zorder=0, clip_on=False, alpha=0.4, dashes=(5, 10), dash_capstyle='round')

        remove_axis_and_spines(ax)

    # share x axis, only show on last
    axes[-1].set_xticks(ticks)
    # display the real log10 values
    xlabels = [du.scformat.format("{:m}", x) for x in 10 ** ticks]
    axes[-1].set_xticklabels(xlabels)
    axes[-1].tick_params(axis='x', which='both', length=0)
    # offset labels by 0.5 towards bottom
    axes[-1].xaxis.set_tick_params(pad=30)
    for tick in axes[-1].xaxis.get_major_ticks():
        tick.label.set_fontsize(8)
        tick.label.set_color('grey')

    mname = model.node_namespace
    fig.suptitle(f'Fluorescence distributions for \n{mname}\n\n({len(rawx)} points)', fontsize=10, y=1.2)

plot_fluo_distributions(dman, 1)
plot_fluo_distributions(dman, 2)







