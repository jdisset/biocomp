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
import flowio
import matplotlib.pyplot as plt
from ott.geometry.pointcloud import PointCloud
from ott.problems.linear.linear_problem import LinearProblem
from ott.solvers.linear.sinkhorn import Sinkhorn
import pandas as pd
import matplotlib as mpl
from typing import Optional, Tuple, List, Dict, Union, Callable

plt.rcParams['figure.figsize'] = [7.0, 7.0]
plt.rcParams['figure.dpi'] = 300

# ────────────────────────────────────────────────────────────────────────────}}}

### {{{                           --     beads     --


def w_function(x, a=0.3, b=0.75, lsteepness=70, rsteepness=500):
    y1 = 1 / (1 + jnp.exp(-lsteepness * (x - a)))
    y2 = 1 / (1 + jnp.exp(-rsteepness * (x - b)))
    return jnp.clip(y1 - y2, 0, 1)


PEAKS_MIN_X = -0.025
PEAKS_MAX_X = 1.1

# observations = logbeads
# beads = logcalib
# resolution = 1000
# bw_method = 0.15
# max_obs = 20000


@partial(jit, static_argnums=(2, 3, 4))
def compute_peaks(observations, beads, resolution=1000, bw_method=0.15, max_obs=20000):

    # First we compute the vote matrix:
    # it's a (CHANNEL, OBSERVATIONS, BEAD) matrix
    # where each channel tells what affinity each observation has to each bead, from the channel's perspective.
    # This is computed using optimal transport (it's the OT matrix)
    # High values mean it's obvious in this channel that the observation should be paired with a certain bead.
    # Low values mean it's not so obvious, usually because the observation is not in the valid range
    # so there's a bunch of points around this one that could be paired with any of the remaining beads
    # This is much more robust than just computing OT for all channels at once

    @partial(vmap, in_axes=(1, 1))
    def vote(s, t):
        return Sinkhorn()(LinearProblem(PointCloud(s[:, None], t[:, None]))).matrix

    if observations.shape[0] > max_obs:
        key = jax.random.PRNGKey(0)
        reorder = jax.random.permutation(key, jnp.arange(max_obs))
        observations = observations[reorder]

    votes = vote(observations, beads)  # (CHANNELS, OBSERVATIONS, BEADS)

    # votes already intrinsically contain a notion of confidence in the pairing, but I want to
    # make it even more explicit by discrediting votes for observations that are clearly out of range.
    # Weight each observation in each channel: 0 = out of range, 1 = in range (+ smooth at the edges)
    weights = vmap(w_function)(observations).T  # (CHANNELS, OBSERVATIONS)
    weighted_votes = votes * weights[:, :, None] + 1e-12  # (CHANNELS, OBSERVATIONS, BEADS)
    vmat = jnp.sum(weighted_votes, axis=0) / jnp.sum(weights, axis=0)[:, None]  # weighted average

    # Use these votes to decide which beads are the most likely for each observation
    # Tried with a softer version of this just in case, but I couldn't see any improvement
    vmat = jnp.argmax(vmat, axis=1)[:, None] == jnp.arange(vmat.shape[1])[None, :]

    # We add some tiny random normal noise to avoid singular matrix errors when computing the KDE
    # on a bead that would have only the exact same value (which can happen when out of range)
    noise_std = (PEAKS_MAX_X - PEAKS_MIN_X) / (resolution * 5)
    obs = observations + jax.random.normal(jax.random.PRNGKey(0), observations.shape) * noise_std

    # Now we can compute the densities for each bead in each channel
    x = jnp.linspace(PEAKS_MIN_X, PEAKS_MAX_X, resolution)
    w_kde = lambda s, w: gaussian_kde(s, weights=w, bw_method=bw_method)(x)
    densities = jit(vmap(vmap(w_kde, in_axes=(None, 1)), in_axes=(1, None)))(obs, vmat)
    densities = densities.transpose(1, 0, 2)  # densities.shape is (BEADS, CHANNELS, RESOLUTION)

    peaks = x[jnp.argmax(densities, axis=2)]  # peaks.shape is (BEADS, CHANNELS)

    return peaks, (densities, vmat)


def plot_bead_peaks_diagnostics(
    peaks, densities, vmat, observations, color_channels, max_obs=20000
):

    if observations.shape[0] > max_obs:
        observations = observations[np.random.choice(observations.shape[0], max_obs, replace=False)]

    mainfig = plt.figure(constrained_layout=True, figsize=(12, 14))
    subfigs = mainfig.subfigures(1, 2, wspace=-0.3, width_ratios=[0.4, 10])
    assignment = jnp.sort(jnp.argmax(vmat, axis=1))[:, None]
    ax = subfigs[0].subplots(1, 1)
    cmap = plt.get_cmap('tab10')
    cmap.set_bad(color='white')
    centroids = [
        (jnp.arange(len(assignment))[assignment[:, 0] == i]).mean() for i in range(vmat.shape[1])
    ]
    ax.imshow(
        assignment,
        aspect='auto',
        cmap=cmap,
        interpolation='none',
        vmin=0,
        vmax=vmat.shape[1],
        origin='lower',
    )
    for i, c in enumerate(centroids):
        ax.text(
            0, c, f'bead {i}', rotation=90, verticalalignment='center', horizontalalignment='center'
        )
    ax.set_ylabel('Observations, sorted by bead assignment')
    ax.set_xticks([])
    ax.set_yticks([0, len(assignment)])

    resolution = densities.shape[2]
    xmin = PEAKS_MIN_X
    xmax = PEAKS_MAX_X
    x = jnp.linspace(xmin, xmax, resolution)
    axes = subfigs[1].subplots(len(color_channels), 1, sharex=True)
    if len(color_channels) == 1:
        axes = [axes]
    weights = vmap(w_function)(x)
    beadcolors = [plt.get_cmap('tab10')(1.0 * i / 10) for i in range(10)]
    for c in range(len(color_channels)):
        ax = axes[c]
        for b in range(peaks.shape[0]):
            dens = densities[b, c]
            dens /= jnp.max(densities[b, c])
            # plot a gradient to show the confidence threshold
            ax.plot(x, dens, label=f'bead {b}', linewidth=1.25, color=beadcolors[b])
            # also plot the actual distribution
            kde = gaussian_kde(observations[:, c], bw_method=0.01)
            d2 = kde(x)
            d2 /= jnp.max(d2) * 1.25
            ax.plot(x, d2, color='k', linewidth=0.75, alpha=1, label='_nolegend_')
            ax.fill_between(x, d2, 0, color='k', alpha=0.01)

            ax.imshow(
                weights[::5][None, :],
                extent=(xmin, xmax, 0, 1.5),
                aspect='auto',
                cmap='Greys_r',
                alpha=0.05,
            )
            # vline at peak
            ax.axvline(peaks[b, c], color='k', linewidth=0.5, dashes=(3, 3))
            # write bead number
            ax.text(peaks[b, c] + 0.01, 0.3 + 0.1 * b, f'{b}', fontsize=8, ha='center', va='center')
            ax.set_ylim(0, 1.2)
            ax.set_yticks([])
            ax.set_xlim(xmin, 1.1)
            title = f'{color_channels[c]}'
            ax.set_ylabel(title)

        axes[0].set_title(
            'Density of observations across channel range (grey background = out of range)'
        )
    subfigs[1].suptitle(
        """
        Bead peak diagnostics\n
        color: density of observations per bead assignment\n
        ----: estimated peak position\n
        dark curve: original distribution of observations in channel
        """
    )


bead_transform = lambda params, x: params['a'] * x + params['b']

bead_init = lambda NCHAN: {
    'a': jnp.ones((NCHAN,)),
    'b': jnp.zeros((NCHAN,)),
}


def remove_axis_and_spines(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
def plot_fluo_distribution(ax, data, res=2000):
    from jax.scipy.stats import gaussian_kde
    xrange = 1.2
    XX = jnp.linspace(-0.05, xrange, res)
    kde = gaussian_kde(data.T, bw_method=0.01)
    smoothkde = gaussian_kde(data.T, bw_method=0.1)
    densities = kde(XX.T)
    ldensities = jnp.log10(1.0 + densities)
    densities = (densities / densities.max()) * 0.4
    ldensities = (ldensities / ldensities.max()) * 0.4
    ax.fill_betweenx(XX, -ldensities, 0, color='k', alpha=0.2, lw=0)
    # ax.fill_betweenx(XX, 0, ldensities, color='k', alpha=0.2, lw=0)
    ax.fill_betweenx(XX, -densities, 0, color='k', alpha=0.2, lw=0)
    # ax.fill_betweenx(XX, 0, densities, color='k', alpha=0.2, lw=0)
    # ax.plot(densities, XX, color='k', alpha=0.5, lw=0.7)
    ax.plot(-densities, XX, color='k', alpha=0.5, lw=0.7)
    # ax.set_aspect("equal")
    ax.set_ylim(-0.05, xrange)
    ax.set_xlim(-0.5, 0.5)
    remove_axis_and_spines(ax)


def plot_dists(
    params, peaks, calib, observations, color_channels, channel_to_unit, max_obs=20000, fname=None
):
    from matplotlib import cm

    if observations.shape[0] > max_obs:
        key = jax.random.PRNGKey(0)
        reorder = jax.random.permutation(key, jnp.arange(max_obs))
        observations = observations[reorder]

    NBEADS, NCHAN = peaks.shape

    fig, axes = du.mkfig(1, NCHAN, (2.2, 12))
    tdata = bead_transform(params, observations)
    tpeaks = bead_transform(params, peaks)
    weights = vmap(w_function)(peaks)
    beadcolors = [plt.get_cmap('tab10')(1.0 * i / 10) for i in range(10)]
    for c in range(NCHAN):
        ax = axes[c]
        error = jnp.average(jnp.abs(tpeaks[1:, c] - calib[1:, c]), weights=weights[1:, c])
        # add beads as horizontal red lines
        for b in range(1, NBEADS):
            w = float(weights[b, c])
            # color by w : from red to green in jet cmap
            alpha = 0.3 + w * 0.7
            # ax.axhline(calib[j, i], color=color, lw=1, dashes=(3, 3))
            ax.axhline(tpeaks[b, c], color='k', lw=1, xmin=0, xmax=0.5)
            ax.text(
                -0.07 - (0.05 * b),
                tpeaks[b, c] + 0.01,
                f'{b}',
                fontsize=8,
                color='k',
                horizontalalignment='center',
                verticalalignment='center',
            )
            ax.axhline(calib[b, c], alpha=alpha, color='k', lw=1, dashes=(3, 3), xmin=0.5, xmax=1)
            ax.text(
                0.45,
                logcalib[b, c] + 0.01,
                f'{b}',
                fontsize=8,
                color='k',
                alpha=alpha,
                horizontalalignment='center',
                verticalalignment='center',
            )
        plot_fluo_distribution(ax, tdata[:, c])
        ax.set_title(f'{color_channels[c]} \n(to {channel_to_unit[color_channels[c]]})', pad=40)

        ax.text(
            0.05,
            1.15,
            f'TARGET',
            fontsize=8,
            color='#999999',
            horizontalalignment='left',
            verticalalignment='center',
        )

        ax.text(
            -0.05,
            1.15,
            f'REAL',
            fontsize=8,
            color='#999999',
            horizontalalignment='right',
            verticalalignment='center',
        )

        cmap = cm.get_cmap('RdYlGn_r')
        color = cmap(error / 0.02)
        ax.text(
            0,
            0.05,
            f'distance: {error:.5f}',
            fontsize=9,
            color=color,
            horizontalalignment='center',
        )

    if fname is not None:
        fig.savefig(fname, dpi=200)
        print('saved', fname)
        plt.close(fig)


def beads_fit(logpeaks, logcalib, num_iter=5000, learning_rate=0.01, ignore_first_bead=True):
    # simple full batch gradient descent

    NCHAN = logpeaks.shape[1]
    params = bead_init(NCHAN)
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    X = jnp.where(jnp.isnan(logpeaks), 0, logpeaks)
    weights = vmap(w_function, in_axes=0)(logpeaks)
    if ignore_first_bead:
        weights = weights.at[0, :].set(0)

    @jax.value_and_grad
    def lossf(params):
        x = bead_transform(params, X)
        avg = jnp.average((x - logcalib) ** 2, weights=weights)
        penalty = -jnp.sum(jnp.clip(params['a'], None, 0))  # penalty where a < 0
        return avg + penalty

    @jit
    def update(params, opt_state):
        loss, grad = lossf(params)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    losses = []
    for i in range(0, num_iter):
        params, opt_state, loss = update(params, opt_state)
        losses.append(loss)

    return params, losses


##────────────────────────────────────────────────────────────────────────────}}}

### {{{             --     some default bead configurations     --
# remember: the first bead is not reliable and should'nt contribute to the calibration
SPHEROTECH_RCP_30_5a = {
    'MEFL': [456, 4648, 14631, 42313, 128924, 381106, 1006897, 2957538, 7435549],
    'MEPE': [196, 2800, 8770, 25174, 74335, 219816, 548646, 1600005, 4255375],
    'MEPTR': [1034, 17518, 53950, 153641, 450901, 1283877, 3254513, 9431807, 24840372],
    'MEPerCP': [605, 4354, 10032, 22473, 51739, 115599, 256091, 562684, 1201350],
    'MEPCY5.5': [180, 2088, 6705, 20441, 66215, 211174, 645020, 2478405, 10603147],
    'MEeF710': [258, 2259, 6862, 20129, 63316, 196680, 609247, 2451473, 11687960],
    'MEPCY7': [88, 534, 1555, 4600, 14826, 47575, 161926, 706536, 3262715],
    'MEPTR': [1142, 17518, 53950, 153641, 450901, 1283877, 3254513, 9431807, 24840372],
    'MEPCY5': [1026, 4354, 10032, 22473, 51739, 115599, 256091, 562684, 1201350],
    'MEPCY5.5': [185, 1999, 6228, 20393, 69124, 220232, 777840, 2521966, 8948283],
    'MEAX700': [480, 6625, 17113, 58590, 199825, 629666, 2289301, 6504723, 17637305],
    'MEPCY7': [25, 457, 1334, 4666, 17500, 58774, 230324, 724800, 2057002],
    'MEAPC': [743, 1170, 1970, 4669, 13757, 36757, 119744, 293242, 638909],
    'MEAX680': [945, 6844, 17166, 56676, 195246, 622426, 2333985, 6617776, 17561028],
    'MEAX700': [495, 6625, 17113, 58590, 199825, 629666, 2289301, 6504723, 17637305],
    'MEAPCCY7': [73, 1385, 3804, 13066, 47512, 151404, 542987, 1305924, 2540123],
    'PacBlue': [979, 4450, 8342, 17587, 38906, 89281, 179989, 408481, 822214],
    'MEAMCY': [1987, 5974, 10513, 21623, 46727, 105630, 213273, 494395, 1072308],
    'MEPO': [148, 391, 753, 1797, 4766, 13937, 39280, 156244, 652221],
    'MEQ605': [1718, 3133, 4774, 8471, 16359, 34465, 71375, 189535, 517591],
    'MEQ655': [1060, 1859, 2858, 5598, 11928, 27542, 66084, 202508, 650000],
    'MEQ705': [840, 1695, 2858, 5598, 11928, 27542, 66084, 202508, 650000],
    'MEBV711': [1345, 1564, 3234, 5516, 12249, 29651, 71051, 197915, 596714],
    'MEQ800': [857, 1358, 2085, 4301, 10037, 23446, 64511, 186279, 644779],
}


FORTESSA_CHANNELS = {
    'Pacific Blue-A': 'PacBlue',
    # 'AmCyan-A': 'MEAMCY',
    'FITC-A': 'MEFL',
    # 'PerCP-Cy5-5-A': 'MEPCY5.5',
    # 'PE-A': 'MEPE',
    'PE-Texas Red-A': 'MEPTR',
    # 'APC-A': 'MEAPC',
    # 'APC-Alexa 700-A': 'MEAX700',
    # 'APC-Cy7-A': 'MEAPCCY7',
}


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                       --     loading data     --


def load_fcs_to_df(fcs_file):
    fcs_data = flowio.FlowData(fcs_file.as_posix())
    channels = [fcs_data.channels[str(i + 1)]['PnN'] for i in range(fcs_data.channel_count)]
    original_data = np.reshape(fcs_data.events, (-1, fcs_data.channel_count))
    df = pd.DataFrame(original_data, columns=channels)
    return df


column_rename = {
    'Pacific_Blue_A': 'Pacific Blue-A',
    'FITC_A': 'FITC-A',
    'PE_Texas_Red_A': 'PE-Texas Red-A',
}


def load_csv_to_df(csv_file):
    df = pd.read_csv(csv_file)
    df = df.rename(columns=column_rename)
    return df


# first, let's grab all the channels by associating the non gated raw files' rows to the gated ones
data_dir = ut.DEFAULT_XP_PATH / '2023-01-22_CasE_ALLuORFs/data/'
fcs = list(Path(data_dir / 'raw_data_gated/').rglob('*.csv'))
fcs = {f.stem: f for f in fcs}
# loaded_fcs = {k: load_fcs_to_df(v) for k, v in fcs.items()}
loaded_fcs = {k: load_csv_to_df(v) for k, v in fcs.items()}


list(loaded_fcs.keys())


##────────────────────────────────────────────────────────────────────────────}}}

### {{{             --     simple controls and preprocessing     --

# IDEA
# it seems some channels (like APC-A for ex) either have a very different start of their validity range,
# or have a fairly non linear behavior for a good while
# Rather than using a non-linear fit of the Spectral Signature thingy,
# I think it'd be smarter to keep the linear model BUT first use the beads to make the channels linear
# (by finding the non-linear mapping between the beads and the channels)!

channel_order = sorted(list(FORTESSA_CHANNELS.keys()))

controls = {
    ('eYFP',): loaded_fcs['eYFP.2023-01-22_CasE_ALLuORFs'][channel_order].copy(),
    ('eBFP',): loaded_fcs['EBFP2.2023-01-22_CasE_ALLuORFs'][channel_order].copy(),
    ('mKate',): loaded_fcs['mKate.2023-01-22_CasE_ALLuORFs'][channel_order].copy(),
    ('eYFP', 'eBFP', 'mKate'): loaded_fcs['ALL.2023-01-22_CasE_ALLuORFs'][channel_order].copy(),
}

bead_fcs = Path(data_dir / 'FCS_FILES/2023-01-22_CasE_ALLuORFs_BEADS_AL01_017.fcs')
beads = load_fcs_to_df(bead_fcs)[channel_order]
beads_reference_values = SPHEROTECH_RCP_30_5a
channel_to_unit = FORTESSA_CHANNELS

controls_order = list(controls.keys())
fluo_proteins = list(set([p for c in controls_order for p in c]))

controls_values, controls_masks = [], []
for m in controls_order:
    vals = controls[m].values
    masks = jnp.tile(jnp.array([p in m for p in fluo_proteins]), (vals.shape[0], 1))
    controls_values.append(vals)
    controls_masks.append(masks)
controls_values = jnp.vstack(controls_values)
controls_masks = jnp.vstack(controls_masks)


def axtransform(x, offset, scale):
    return jnp.log10(x + offset) / scale


def inverse_transform(x, offset, scale):
    return 10 ** (x * scale) - offset


SCALE = 7
OFFSET = 0

# AUTOFLUORESCENCE
blankdf = loaded_fcs['CNTL.2023-01-22_CasE_ALLuORFs'][channel_order].copy()
autofluorescence = np.median(blankdf.values, axis=0)
controls_values = controls_values - autofluorescence

# TODO: try remove instead of clip for values < 1 (NOT FOR BEADS AS IT WOULD BREAK PEAK ASSIGNMENT)
logcontrols = axtransform(jnp.clip(controls_values, 1), OFFSET, SCALE)

logbeads = axtransform(jnp.clip(beads.values, 1), OFFSET, SCALE)
calib_values = jnp.array([beads_reference_values[channel_to_unit[c]] for c in channel_order]).T
logcalib = axtransform(calib_values, OFFSET, SCALE)

calib_values.max()

logbeads.shape
print('Computing peaks assignment')
peaks, (densities, vmat) = compute_peaks(logbeads, logcalib)
print('Plotting peaks assignment')
plot_bead_peaks_diagnostics(peaks, densities, vmat, logbeads, channel_order)
print('Fitting')
beads_params, l = beads_fit(peaks, logcalib)
fig, ax = du.mkfig(1, 1, (5, 5))
ax.plot(l)
ax.set_yscale('log')
plot_dists(beads_params, peaks, logcalib, logbeads, channel_order, channel_to_unit)


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                        --     optim loop     --
from jax.config import config

config.update("jax_debug_nans", True)
# {{{
unit_to_channel = {v: k for k, v in channel_to_unit.items()}
# reorder logdata and controls_masks randomly
unit_to_channel

key = jax.random.PRNGKey(2222)
# reorder = jax.random.permutation(key, jnp.arange(len(logcontrols)))
# we use the beads to calibrate all channels first


Y = logcontrols
masks = controls_masks
weights = vmap(w_function)(Y)
# Y = bead_transform(beads_params, Y)
Y = inverse_transform(Y, OFFSET, SCALE) / 1e3

# # plot Y}}}
# fig, ax = du.mkfig(1, 1, (5, 5))
# # use masks as color. Right now masks is 3 values between 0 and 1 for each observation
# # so we can use it as a RGB color
# # first get only 1000 random points
# ysample = jax.random.permutation(key, jnp.arange(Y.shape[0]))[:5000]
# colors = masks[ysample] * 0.6
# # ax.scatter(Y[ysample, 0], Y[ysample, 1], c=colors, s=3, alpha=0.1, lw=0)
# # plot weights
# weights.shape
# Y.shape
# ax.scatter(Y[ysample, 1], weights[ysample, 1], c=colors, s=3, alpha=0.1, lw=0)
# ax.set_xlabel(channel_order[0])
# ax.set_ylabel(channel_order[1])
# # plot wx
# # ax.set_xscale('log')
# # ax.set_yscale('log')


# standardization to calibrated units
standardize_to = ('MEFL', 'eYFP')  # (channel, protein)
std_chan = (channel_order.index(unit_to_channel[standardize_to[0]]),)
std_prot = (fluo_proteins.index(standardize_to[1]),)
std_prot_mask = jnp.zeros((len(fluo_proteins),), dtype=jnp.bool_)
std_prot_mask = std_prot_mask.at[std_prot].set(True)

num_iter = 500
learning_rate = 0.05
dump_every = 50

params = {
    'S': jax.random.uniform(
        key, shape=(len(fluo_proteins), len(channel_order)), minval=0, maxval=1
    ),  # spectral signature matrix
}


def cosine_similarity(u, v, w):
    uv = jnp.average(u * v, weights=w)
    uu = jnp.average(jnp.square(u), weights=w)
    vv = jnp.average(jnp.square(v), weights=w)
    return 1.0 - uv / jnp.sqrt(uu * vv)


a = jnp.array([1, 0])
b = jnp.array([2.1, 0.1])
cosine_similarity(a, b, jnp.ones_like(a))

ANGLE_W, RATIO_W = 1, 0

ZERO_W, NEG_W = 0, 0


def inner_loss(S, y_i, m_i, w_i):
    MS = m_i @ S
    is_ref_ctrl = jnp.all(m_i == std_prot_mask)

    ww = jnp.ones_like(y_i)
    # ww = w_i
    ratio_error = (
        is_ref_ctrl
        * jnp.average((MS * y_i[std_chan] - y_i) ** 2, weights=ww)
        / jnp.average(y_i**2, weights=ww)
    )
    angle_error = cosine_similarity(y_i, MS, ww) * ~jnp.all(m_i)
    # wherever m_i is 0, we want x to be 0
    sinv = jnp.linalg.inv(S)
    x = y_i @ sinv
    zero_x = jnp.mean(jnp.square(jnp.where(~m_i, x, 0)))

    neg_x = jnp.mean(jnp.square(jnp.clip(S, None, 0)))

    return angle_error * ANGLE_W + ratio_error * RATIO_W + zero_x * ZERO_W + neg_x * NEG_W


@jax.value_and_grad
def lossf(params):
    l = vmap(inner_loss, in_axes=(None, 0, 0, 0))(params['S'], Y, masks, weights)
    return jnp.mean(l)


optimizer = optax.adam(learning_rate=learning_rate)
opt_state = optimizer.init(params)


@jit
def update(params, opt_state):
    # start with full batch gradient descent
    loss, grad = lossf(params)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


all_params = []
losses = []
for iteration in tqdm(range(0, num_iter + 1), desc='Estimating S and A', total=-1, leave=False):
    params, opt_state, loss = update(params, opt_state)
    losses.append(loss)
    if iteration % dump_every == 0:
        all_params.append(params)
        print(f'loss: {loss}')

# plot losses
fig, ax = du.mkfig(1, 1, (2, 2))
ax.plot(losses, lw=1)
ax.set_yscale('log')
best_params = params.copy()

## diagnostics
# first plot S
S = best_params['S']
fig, ax = du.mkfig(1, 1, (5, 5))
for s, p in zip(S, fluo_proteins):
    ax.plot(s, label=p)
ax.legend()
ax.set_xlabel('channel')
ax.set_xticks(range(len(channel_order)))
ax.set_xticklabels(channel_order, rotation=90)

##
reorder = jax.random.permutation(key, jnp.arange(Y.shape[0]))[:20000]
# plot original distributions
fig, axes = du.mkfig(1, len(channel_order), (4, 4))
submasks = masks[reorder]
M = submasks[:, 0] & submasks[:, 1] & submasks[:, 2]
subsample = Y[reorder]
ssample = subsample[M]
for i, c in enumerate(channel_order):
    # hist in log scale
    axes[i].hist(ssample[:, i], bins=70, log=True)
    axes[i].set_title(c)
plt.show()

##
sinv = jnp.linalg.inv(S)
getp = lambda y: y @ (jnp.linalg.inv(S.T @ S) @ S.T)
proteins = vmap(getp)(Y)

subproteins = proteins[reorder]
submasks = masks[reorder]
prots = subproteins[M]

i = 11
ssample[i]
prots[i] @ S
getp(np.array([0, 10, 100]))


# now let's check per mask

channel_order
fluo_proteins

fig, axes = du.mkfig(1, len(fluo_proteins), (4, 4))
for i, p in enumerate(fluo_proteins):
    # hist in log scale
    axes[i].hist(prots[:, i], bins=70, log=True)
    axes[i].set_title(p)
plt.show()


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                          --     archive     --

### {{{                          --     gating?     --
# df = controls[tuple()]
# # join all controls in df
# df = pd.concat(controls.values(), ignore_index=True)

# # keep only the 0.001 - 0.999 quantile in SSC-A and FSC-A
# df = df[(df['SSC-A'] > df['SSC-A'].quantile(0.001)) & (df['SSC-A'] < df['SSC-A'].quantile(0.999))]


# def scatter_density(x, y, ax=None, **kwargs):
# if ax is None:
# fig, ax = plt.subplots(1, 1)
# xy = np.vstack([x, y])
# xy_sub = xy[:, np.random.choice(xy.shape[1], 2000)]
# kde = jit(gaussian_kde)(xy_sub)
# # z = kde(xy)
# contour_res = 250
# xx = np.linspace(x.min(), x.max(), contour_res)
# yy = np.linspace(y.min(), y.max(), contour_res)
# mgrid = np.meshgrid(xx, yy)
# mgrid = np.vstack([mgrid[0].ravel(), mgrid[1].ravel()])
# zz = kde(np.vstack(mgrid)).reshape(contour_res, contour_res)
# # don't display first contour
# ax.contour(xx, yy, zz, 5, colors='k', alpha=0.5, linewidths=0.5)

# dthreshold = np.quantile(z, 0.3)
# # display a background image of the density with green when > 0.1 quantile, red otherwise
# # custom cmap (green to red)
# cmap = mpl.colors.ListedColormap(['red', 'green'])
# bounds = [0, dthreshold, 1]
# norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
# # using pcolormesh
# ax.scatter(x, y, cmap='inferno', **kwargs)
# ax.pcolormesh(xx, yy, zz, cmap=cmap, norm=norm, alpha=0.3)

# # log scales
# return ax


# fig, ax = du.mkfig(1, 1)
# scatter_density(df['SSC-A'], df['FSC-A'], ax=ax, s=0.25, alpha=0.25, linewidths=0)
# ax.set_xlabel('SSC-A')
# ax.set_ylabel('FSC-A')
# # ax.set_xscale('log')
# # ax.set_yscale('log')
# # ax.set_xlim(1e0, 1e7)
# # ax.set_ylim(1e0, 1e7)

# ##
# # simple interactive plot to draw a polygon:
# import matplotlib.pyplot as plt
# from matplotlib.widgets import PolygonSelector
# from matplotlib.path import Path

# plt.switch_backend('QtAgg')
# fig, ax = plt.subplots()
# ax.scatter(df['SSC-A'], df['FSC-A'], s=0.25, alpha=0.25, linewidths=0)
# ax.set_xlabel('SSC-A')
# ax.set_ylabel('FSC-A')


# def onselect(verts):
# path = Path(verts)


# ps = PolygonSelector(ax, onselect, useblit=True, lineprops=dict(color='r', linewidth=2))
# # open in interactive mode in a new window (we're in jupyter so we have to manually specify the backend):
# plt.show()

# def norm_per_P_C(k, s, y, prot=PROT, chan=CHAN):
# # normalize the k matrix so that the sum of each column of k should be equal to
# # y[:, YFPC] when the mask is the single ctrl for prot
# # and
# is_prot = jnp.all(masks == jax.nn.one_hot(prot, p), axis=1)
# normalized = k * y[:, chan] / k.sum(axis=0)
# kk = jnp.where(is_prot, normalized, k)
# s = s / s[prot, chan]
# # assert(jnp.all(~(jnp.diag(res) == 1) == is_prot))
# return s, kk
# fig, ax = du.mkfig(1, 1, (10, 10))
# ax.imshow(Kguess[:100, :100], cmap='viridis')
# for i in range(p):
# mm = jnp.all(masks == jax.nn.one_hot(i, p), axis=1)[:100]
# ismm = jnp.where(mm)[0]
# ax.scatter(ismm, jnp.zeros_like(ismm) - i - 1, marker='s', s=6)
# ax.set_title('Kguess')
# fig.tight_layout()

##────────────────────────────────────────────────────────────────────────────}}}

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                      --     plot functions     --

# plot the spectral signature matrix
def plot_spectral_sig(S, ax):
    ax.imshow(S, cmap='Reds')
    ax.set_xticks(range(S.shape[1]))
    ax.set_yticks(range(S.shape[0]))
    ax.set_title('Spectral signature matrix')
    ax.set_xlabel('Channels')
    ax.set_ylabel('Proteins')


KDE_BW = 0.075


def plot_single_controls_Xdist(X, masks, ax, title=None):
    choice = jax.random.choice(jax.random.PRNGKey(0), X.shape[0], (10000,))
    s_masks = masks[choice]
    s_x = X[choice]
    unique_masks = jax.nn.one_hot(jnp.arange(p), p).astype(jnp.int32)
    for i, mask in enumerate(unique_masks):
        # compute kde
        ids = jnp.all(s_masks == mask, axis=1)
        kde = gaussian_kde(s_x[ids][:, i], bw_method=KDE_BW)
        q = jnp.quantile(X, 0.95)
        x = np.linspace(-0.1*q,q, 1500)
        y = kde(x)
        ax.plot(x, y, label=f'Control {mask}')
        ax.legend()
    if title is not None:
        ax.set_title(title)



def plot_single_controls_Ydist(y, masks, ax, title=None):
    choice = jax.random.choice(jax.random.PRNGKey(0), len(y), (10000,))
    s_masks = masks[choice]
    s_y = y[choice]
    # unique_masks = jax.nn.one_hot(jnp.arange(p), p).astype(jnp.int32)
    # unique_masks = jnp.vstack([unique_masks, jnp.ones((p,))])
    unique_masks = jnp.unique(masks, axis=0).astype(jnp.int32)
    for i, mask in enumerate(unique_masks):
        # compute kde
        ids = jnp.all(s_masks == mask, axis=1)
        q = jnp.quantile(y, 0.95)
        x = np.linspace(-0.1*q, q, 1500)
        d = gaussian_kde(s_y[ids], bw_method=KDE_BW)(x)
        d = d / d.max()
        ax.plot(x, d, label=f'Control {mask}')
        ax.legend()
        # log scale
        # ax.set_yscale('log')
        # ax.set_xscale('log')
    if title is not None:
        ax.set_title(title)


def contrast(s, pow=2, n=2):
    for _ in range(n):
        s = s**pow
        s = s / s.max(axis=1)[..., None]
    return s


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                        --     toy example     --

# we have the following spectral signature matrix, for 4 channels and 3 proteins
p, c = 3,6 
key = jax.random.PRNGKey(1087)
k1, k2, k3, k4 = jax.random.split(key, 4)
Strue = jax.random.uniform(k1, (p, c), minval=0.25, maxval=1)
Strue = contrast(Strue, pow=2, n=3) * jax.random.uniform(k2, (p, 1), minval=0.5, maxval=1)
Strue *= 0.3


# lets generate data for single color and all color controls
NCELLS = 100000
masks = jax.nn.one_hot(jax.random.randint(jax.random.PRNGKey(0), (NCELLS,), 0, p), p)
masks = masks.at[jax.random.bernoulli(jax.random.PRNGKey(0), 1 / (p + 1), (NCELLS,))].set(
    jnp.ones_like(masks[0])
)
XSCALE = 1e3
Xbase = jax.random.gamma(k3, 1, (NCELLS, 1)) * XSCALE
Xtrue = Xbase * masks

# now we generate the observations
Y = Xtrue @ Strue
Y += jax.random.normal(k4, Y.shape) * Y * 0.1

# plot the data
fig, ax = du.mkfig(1, 1, (8, 2))
plot_spectral_sig(Strue, ax)
# plot_single_controls_Xdist(Xtrue, masks, axes[1])
# for i in range(Y.shape[1]):
    # plot_single_controls_Ydist(Y[:, i], masks, axes[2 + i], title=f'Channel {i}')
# fig.tight_layout()

# estimate S
# TODO

masks.shape
Y.shape

# X = jax.random.uniform(jax.random.PRNGKey(0), (Y.shape[0], M.shape[1]), minval=0.1, maxval=1)
##

M = masks

# fluo_proteins
# channel_order


def spectral_signature_estimation(
    Y,
    M,
    max_iterations=10,
    max_n=10000,
    normalize_to_prot_chan=(0, 0),
    jax_seed=0,
):


    if Y.shape[0] > max_n:  # resample Y and M to get only max_n
        choice = jax.random.choice(jax.random.PRNGKey(jax_seed), len(Y), (max_n,))
        Y, M = Y[choice], M[choice]

    # Initialize S with random positive values and K as identity
    S = jax.random.uniform(jax.random.PRNGKey(0), (M.shape[1], Y.shape[1]), minval=0.1, maxval=1)
    S /= S[normalize_to_prot_chan]  # normalize S to the desired protein and channel
    K = jnp.identity(Y.shape[0])  # Identity is a decent start for K as it's supposedly diagonal

    @jit
    def alsq(S, K):  # one iteration of alternating least squares
        # model: Y = KMS
        # TODO regularization
        S = jnp.linalg.pinv(K @ M) @ Y  # find S from K
        S /= S[normalize_to_prot_chan]  # normalize S to the desired protein and channel
        K = Y @ jnp.linalg.pinv(M @ S)  # find K from S
        return S, K

    ynorm = jnp.linalg.norm(Y)
    pbar = tqdm(range(max_iterations), desc='Spectral signature estimation')
    for iter in pbar:
        S, K = alsq(S, K)
        err = jnp.mean((Y - K @ M @ S) ** 2) / ynorm
        pbar.set_description(f'Spectral signature estimation (err={err:.2e})')
        # if err < convergence_threshold and iter > 2:
        # break

    return S, K


jnp.set_printoptions(precision=3, suppress=True)

def spectral_signature_estimation_gd(
    Y,
    M,
    max_iterations=300,
    max_n=500000,
    learning_rate=0.1,
    normalize_to_prot_chan=(0, 0),
    jax_seed=0,
):

    # TODO: weighted Alternating Least Square
    # http://ethen8181.github.io/machine-learning/recsys/1_ALSWR.html
    if Y.shape[0] > max_n:  # resample Y and M to get only max_n
        choice = jax.random.choice(jax.random.PRNGKey(jax_seed), len(Y), (max_n,))
        Y, M = Y[choice], M[choice]




    # initialize with OLS (way faster!)
    S, _ = spectral_signature_estimation(
        Y,
        M,
        max_iterations=10,
        max_n=5000,
        jax_seed=jax_seed,
        normalize_to_prot_chan=normalize_to_prot_chan,
    )

    rescaler = jnp.quantile(Y, 0.9)
    Y = Y / max(rescaler, 1e-6)

    K = Y @ jnp.linalg.pinv(S)
    K = jnp.average(K, axis=1, weights=M)


    optS = optax.amsgrad(learning_rate=learning_rate)
    optK = optax.amsgrad(learning_rate=learning_rate)
    stateS, stateK = optS.init(S), optK.init(K)

    def loss_single_row(yi, ki, mi, S):
        return jnp.mean((ki * mi @ S - yi) ** 2)

    def lS(S, K):
        s = S / S[normalize_to_prot_chan]
        err = vmap(loss_single_row, in_axes=(0, 0, 0, None))(Y, K, M, s)
        return jnp.mean(err)

    def lK(K, S):
        return lS(S, K)


    def half_update(a, b, lossf, opt, state):
        loss, g = jax.value_and_grad(lossf)(a, b)
        update, state = opt.update(g, state, a)
        a = optax.apply_updates(a, update)
        return a, state, loss

    @jit
    def update(s, k, stateS, stateK):
        s, stateS, lossS = half_update(s, k, lS, optS, stateS)
        k, stateK, lossK = half_update(k, s, lK, optK, stateK)
        return s, k, stateS, stateK, lossS + lossK

    losses = []
    pbar = tqdm(range(max_iterations), desc='Spectral signature estimation')
    for i in pbar:
        S, K, stateS, stateK, loss = update(S, K, stateS, stateK)
        losses.append(loss)
        pbar.set_description(f'Spectral signature estimation (loss={loss:.5f})')
        if i < 3:
            print(f'loss = {loss:.7f}')

    S = S / S[normalize_to_prot_chan]

    return S, K, losses


PROT, CHAN = 1, 1

Sguess, K, _ = spectral_signature_estimation_gd(Y, masks, normalize_to_prot_chan=(PROT, CHAN))
# Sguess, K = spectral_signature_estimation(Y, masks, normalize_to_prot_chan=(PROT, CHAN))
Xguess = Y @ jnp.linalg.pinv(Sguess)
Xguess.max()
Xguess.min()

is_prot = jnp.all(masks == jax.nn.one_hot(PROT, p), axis=1)
true_protchan = Y[:, CHAN] * is_prot
guess_protchan = Xguess[:, PROT] * is_prot
err_protchan = jnp.abs(true_protchan - guess_protchan).mean() / true_protchan.mean()
print(f'Prot Xguess error: {100*err_protchan:.1f}%')

# fullprot_guess = jnp.maximum(0, Xguess) * 1
# fullprot_true = jnp.maximum(0, Xtrue)
# ratio = fullprot_guess / fullprot_true
# ratio = jnp.where(jnp.isinf(ratio), jnp.nan, ratio)
# # compute mean of ratio when not nan or inf:
# mratio = jnp.nanvar(ratio)
# print(f'Prot Xguess / Xtrue ratio variance: {mratio:.5f}')



fig, axes = du.mkfig(2, 1, (8, 2))
plot_spectral_sig(Strue, axes[0])
plot_spectral_sig(Sguess, axes[1])
print(f'\n Sguess = \n {Sguess}')
print(f'\n Strue= \n {Strue/ Strue[PROT, CHAN]}')


# fig, axes = du.mkfig(4, 1, (8, 2))
# plot_single_controls_Xdist(Xguess, masks, axes[0], 'Estimated X dist')
# plot_single_controls_Xdist(Xtrue, masks, axes[1], 'Original X dist')
# plot_single_controls_Ydist(Y[:,CHAN], masks, axes[2], 'Y distribution of ref channel')
# fig.tight_layout()

##


def plot_single_controls_Xdist(X, masks, mask, ax, title=None):
    choice = jax.random.choice(jax.random.PRNGKey(0), X.shape[0], (min(10000, X.shape[0]),))
    s_masks = masks[choice]
    s_x = X[choice]
    ids = jnp.all(s_masks == mask, axis=1)
    # plot estimated protein content for this mask
    for prot in range(X.shape[1]):
        # compute kde
        kde = gaussian_kde(s_x[ids, prot], bw_method=0.1)
        q = jnp.quantile(X, 0.95)
        x = np.linspace(-0.1*q,q, 1500)
        y = kde(x)
        y = y / y.max()
        ax.plot(x, y, label=f'Protein {prot}')
        ax.legend()
    if title is not None:
        ax.set_title(title)


fig, axes = du.mkfig(3, 1, (10,3))
unique_masks = jax.nn.one_hot(jnp.arange(p), p).astype(jnp.int32)
mask = unique_masks[PROT]
plot_single_controls_Xdist(Xguess, masks, mask, axes[0], f'Estimated X dist for mask {mask}')
plot_single_controls_Xdist(Xtrue, masks, mask, axes[1], f'Original X dist for mask {mask}')
plot_single_controls_Ydist(Y[:,CHAN], masks, axes[2], 'Y distribution of ref channel')
fig.tight_layout()

# reorder = jax.random.permutation(key, jnp.arange(Y.shape[0]))[:50000]
# # plot original distributions
# fig, axes = du.mkfig(1, len(channel_order), (4, 4))
# submasks = masks[reorder]
# M = submasks[:, 0] & submasks[:, 1] & submasks[:, 2]
# subsample = Y[reorder]
# ssample = subsample[M]
# for i, c in enumerate(channel_order):
    # # hist in log scale
    # axes[i].hist(ssample[:, i], bins=70, log=True)
    # axes[i].set_title(c)
    # plt.show()

# proteins = Y @ jnp.linalg.pinv(Sguess)

# subproteins = proteins[reorder]
# submasks = masks[reorder]
# prots = subproteins[M]

# # channel_order
# # fluo_proteins

# fig, axes = du.mkfig(1, len(fluo_proteins), (4, 4))
# for i, p in enumerate(fluo_proteins):
# # hist in log scale
# axes[i].hist(prots[:, i], bins=70, log=True)
# axes[i].set_title(p)
# plt.show()

# print(f'Cond number of S: {jnp.linalg.cond(Sguess):.2f}')

##────────────────────────────────────────────────────────────────────────────}}}
