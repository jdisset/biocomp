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

plt.rcParams['figure.figsize'] = [7.0, 7.0]
plt.rcParams['figure.dpi'] = 300

# ────────────────────────────────────────────────────────────────────────────}}}

### {{{                     --     calibrated values     --

# remember: the first bead is not reliable and should'nt contribute to the calibration
calibrated_beads_values = {
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

calib_names = list(calibrated_beads_values.keys())
calib_values = jnp.array([calibrated_beads_values[name] for name in calib_names]).T

channels_to_units = {
    'Pacific Blue-A': 'PacBlue',
    'AmCyan-A': 'MEAMCY',
    'FITC-A': 'MEFL',
    'PerCP-Cy5-5-A': 'MEPCY5.5',
    'PE-A': 'MEPE',
    'PE-Texas Red-A': 'MEPTR',
    # 'APC-A': 'MEAPC',
    # 'APC-Alexa 700-A': 'MEAX700',
    # 'APC-Cy7-A': 'MEAPCCY7',
}

xpname = '2023-01-22_CasE_ALLuORFs'
beadfcs = ut.DEFAULT_XP_PATH / xpname / 'data/FCS_FILES/2023-01-22_CasE_ALLuORFs_BEADS_AJ01_018.fcs'
fcs_data = flowio.FlowData(beadfcs.as_posix())

channels = [fcs_data.channels[str(i + 1)]['PnN'] for i in range(fcs_data.channel_count)]

original_data = np.reshape(fcs_data.events, (-1, fcs_data.channel_count))

# now let's just get the values for the channels we have:
color_channels = list(channels_to_units.keys())

calib_values = jnp.array(
    [calibrated_beads_values[channels_to_units[channel]] for channel in color_channels]
).T

calib_values

data = original_data[:, [channels.index(channel) for channel in color_channels]]


def transform(x, offset, scale):
    return jnp.log10(x + offset) / scale


def inverse_transform(x, offset, scale):
    return 10 ** (x * scale) - offset


# OFFSET = 1 - jnp.min(data, axis=0)
SCALE = 7
OFFSET = 0


# we can't remove data where values are < 0 as
# it would mess up the distribution much more than just clipping
# so... we clip
data = jnp.clip(data, 1, None)

logdata = transform(data, OFFSET, SCALE)
logcalib = transform(calib_values, OFFSET, SCALE)

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                --     compute_centroids function     --


def w_function(x, a=0.3, b=0.72, lsteepness=70, rsteepness=400):
    y1 = 1 / (1 + jnp.exp(-lsteepness * (x - a)))
    y2 = 1 / (1 + jnp.exp(-rsteepness * (x - b)))
    return jnp.clip(y1 - y2, 0, 1)


source = logdata
target = logcalib


def vote(s, t):
    ot_prob = LinearProblem(PointCloud(s[:, None], t[:, None]))
    return Sinkhorn()(ot_prob).matrix


vote_vm = jit(vmap(vote, in_axes=(1, 1)))


@jit
def vote_all(source, target):
    geom = PointCloud(source, target)
    ot_prob = LinearProblem(geom)
    return Sinkhorn()(ot_prob).matrix


@jit
def compute_centroids(
    source,
    target,
    thresholds=(0.33, 0.72),
    confidence_threshold_quantile=0.9,
    confidence_threshold_absolute=0.2,
    left_steepness=10,
    right_steepness=100,
):

    # First we compute the "votes" matrix:
    # it's a (CHANNEL x OBSERVATIONS x BEAD) matrix
    # where each channel tells what affinity each observation has to each bead
    # from its perspective. This "affinity" measure is computed using optimal transport.
    # High values mean it's quite obvious that the observation should be paired with the given bead
    # Low values mean it's not so obvious, usually because the observation is not in the valid range
    # so there's a bunch of points around this one that could be paired with any of the remaining beads

    @partial(vmap, in_axes=(1, 1))
    def vote(s, t):
        return Sinkhorn()(LinearProblem(PointCloud(s[:, None], t[:, None]))).matrix

    votes = vote(source, target)

    # votes already intrinsically contain a notion of confidence in the pairing, but I want to
    # make it more explicit by manually "discrediting" votes for observations that are clearly out of range.
    # Weight each observation in each channel:
    # 0 = out of range, 1 = in range (with some smoothness at the edges)
    # same shape as votes
    weights = vmap(w_function, in_axes=(0, None, None, None, None))(
        source, thresholds[0], thresholds[1], left_steepness, right_steepness
    ).T

    weighted_votes = votes * weights[:, :, None]
    wvmat = jnp.sum(weighted_votes, axis=0) / jnp.sum(weights, axis=0)[:, None]

    # threshold the weights
    wvmat_norm = wvmat / jnp.sum(wvmat, axis=1)[:, None]
    conf = jnp.quantile(wvmat_norm, confidence_threshold_quantile, axis=0)
    confidence_threshold = jnp.clip(conf, confidence_threshold_absolute)
    wvmat_th = jnp.where(wvmat_norm > confidence_threshold, wvmat_norm, 0)

    beadcentroids = vmap(jnp.average, in_axes=(None, None, 1))(source, 0, wvmat_th)
    return beadcentroids


centroids = compute_centroids(logdata, logcalib)

##────────────────────────────────────────────────────────────────────────────}}}


### {{{                         --     use peaks     --


@jit
def compute_peaks(observations, beads, resolution=1000, bw_method=0.15):

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

    votes = vote(observations, beads)

    # votes already intrinsically contain a notion of confidence in the pairing, but I want to
    # make it even more explicit by discrediting votes for observations that are clearly out of range.
    # Weight each observation in each channel: 0 = out of range, 1 = in range (+ smooth at the edges)
    weights = vmap(w_function)(observations).T
    weighted_votes = votes * weights[:, :, None] + 1e-12
    vmat = jnp.sum(weighted_votes, axis=0) / jnp.sum(weights, axis=0)[:, None]  # weighted average

    # Use these votes to decide which beads are the most likely for each observation
    # Tried with a softer version of this just in case, but I couldn't see any improvement
    vmat = jnp.argmax(vmat, axis=1)[:, None] == jnp.arange(vmat.shape[1])[None, :]

    # Now we can compute the densities for each bead in each channel
    x = jnp.linspace(0, 1.25, resolution)
    w_kde = lambda s, w: gaussian_kde(s, weights=w, bw_method=bw_method)(x.T)
    densities = jit(vmap(vmap(w_kde, in_axes=(None, 1)), in_axes=(1, None)))(observations, vmat)
    densities = densities.transpose(1, 0, 2) # densities.shape is (BEADS, CHANNELS, RESOLUTION)

    peaks = x[jnp.argmax(densities, axis=2)] # peaks.shape is (BEADS, CHANNELS)

    return peaks, densities


peaks, densities = compute_peaks(logdata, logcalib)


x = jnp.linspace(0, 1.25, 1000)
fig, axes = du.mkfig(len(color_channels), 1, (8, 2))
if len(color_channels) == 1:
    axes = [axes]
weights = vmap(w_function)(x)
for c in range(len(color_channels)):
    ax = axes[c]
    for b in range(peaks.shape[0]):
        dens = densities[b, c]
        dens /= jnp.max(densities[b, c])
        # plot a gradient to show the confidence threshold
        ax.plot(x, dens, label=f'bead {b}', linewidth=1)

        # also plot the actual distribution
        kde = gaussian_kde(logdata[:, c], bw_method=0.01)
        d2 = kde(x)
        d2 /= jnp.max(d2) * 2
        ax.plot(x, d2, color='k', linewidth=0.5, alpha=0.5)
        ax.fill_between(x, d2, 0, color='k', alpha=0.01)

        ax.imshow(
            weights[::5][None, :],
            extent=(0, 1.25, 0, 1.5),
            aspect='auto',
            cmap='Greys_r',
            alpha=0.05,
        )
        # vline at peak
        ax.axvline(peaks[b, c], color='k', linewidth=0.5, dashes=(3, 3))
        # write bead number
        ax.text(peaks[b, c] + 0.01, 0.3 + 0.1 * b, f'{b}', fontsize=8, ha='center', va='center')
        ax.set_ylim(0, 1.2)
        ax.set_xlim(0, 1.1)
        ax.set_title(f'{color_channels[c]}')
fig.tight_layout()

##────────────────────────────────────────────────────────────────────────────}}}


##────────────────────────────────────────────────────────────────────────────}}}


### {{{                      --     plot centroids     --
ax0 = 4
ax1 = 2
# scatter bead centroids and target
fig, ax = du.mkfig(1, 1, (5, 5))
# scater points in black
ax.scatter(logdata[:, ax0], logdata[:, ax1], s=1, alpha=0.05, color='k', linewidth=0)
colors = plt.cm.tab10(np.linspace(0, 1, len(color_channels)))

for i in range(len(color_channels)):
    ax.scatter(
        peaks[i, ax0],
        peaks[i, ax1],
        s=20,
        alpha=0.5,
        color=colors[i],
    )
    ax.scatter(
        centroids[i, ax0],
        centroids[i, ax0],
        s=20,
        alpha=0.5,
        color=colors[i],
        marker='^',
    )

# scatter target, same color (from assignment
for i in range(len(color_channels)):
    ax.scatter(logcalib[i, ax0], logcalib[i, ax1], s=25, alpha=1, marker='x', color=colors[i])

# add labels
ax.set_xlabel(color_channels[ax0])
ax.set_ylabel(color_channels[ax1])

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                   --     fitting and plotting     --
# we could maybe have gone with a simple Ordinary Least Squares,
# but I want to be able to weight the peaks and add non-negative constraints
# and we already have all the machinery for that in the biocomp so why not use it

LEARNING_RATE = 0.01
DUMP_EVERY = 1
N_ITER = 3000

source = peaks
target = logcalib


def loss_mse(params, source, target):
    xx = jnp.where(jnp.isnan(source), 0, source)
    xx = params['a'] * xx + params['b']
    weights = vmap(w_function, in_axes=0)(source)
    weights = jnp.where(jnp.isnan(weights), 0, weights)
    # first bead has 0 weight:
    weights = weights.at[0, :].set(0)
    avg = jnp.average((xx - target) ** 2, weights=weights)
    # we want a to be non-negative so we add a penalty where a < 0
    penalty = -jnp.sum(jnp.clip(params['a'], None, 0))
    return avg + penalty


lossf = jax.value_and_grad(loss_mse)
optimizer = optax.adam(learning_rate=LEARNING_RATE)
params = {'a': jnp.ones((source.shape[1],)), 'b': jnp.zeros((source.shape[1],))}
opt_state = optimizer.init(params)


@jit
def update(params, opt_state, source=source, target=logcalib):
    loss, grad = lossf(params, source, target)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


all_params = []
losses = []
for i in tqdm(list(range(0, N_ITER + 1))):
    params, opt_state, loss = update(params, opt_state)
    losses.append(loss)
    if i % DUMP_EVERY == 0:
        all_params.append(params)

print('final loss', loss)
# plot loss
fig, ax = du.mkfig(1, 1, (5, 5))
ax.plot(losses)
ax.set_yscale('log')

##

### {{{                      --     plot functions     --


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
    ax.fill_betweenx(XX, 0, ldensities, color='k', alpha=0.2, lw=0)
    ax.fill_betweenx(XX, -densities, 0, color='k', alpha=0.25, lw=0)
    ax.fill_betweenx(XX, 0, densities, color='k', alpha=0.25, lw=0)
    ax.plot(densities, XX, color='k', alpha=1, lw=1)
    ax.plot(-densities, XX, color='k', alpha=1, lw=1)
    # ax.set_aspect("equal")
    ax.set_ylim(-0.05, xrange)
    ax.set_xlim(-0.5, 0.5)
    remove_axis_and_spines(ax)


##────────────────────────────────────────────────────────────────────────────}}}
# as kde for each channel:
params = all_params[-1]
from matplotlib import cm
import matplotlib as mpl


def plot_dists(params, fname=None):
    fig, axes = du.mkfig(1, 9, (2, 10))
    tdata = params['a'] * logdata + params['b']
    weights = vmap(w_function)(peaks)
    for i in range(centroids.shape[1]):
        ax = axes[i]
        # add beads as horizontal red lines
        for j in range(1, centroids.shape[0]):
            w = float(weights[j, i])
            # color by w : from red to green in jet cmap
            color = cm.seismic(1.0 - w)
            ax.axhline(logcalib[j, i], color=color, lw=1, dashes=(3, 3))
            ax.text(
                0.45,
                logcalib[j, i] + 0.01,
                f'{j}',
                fontsize=9,
                color=color,
                horizontalalignment='center',
                verticalalignment='center',
            )
        plot_fluo_distribution(ax, tdata[:, i])
        ax.set_title(f'{color_channels[i]} \n(to {channels_to_units[color_channels[i]]})', pad=40)

    if fname is not None:
        fig.savefig(fname, dpi=200)
        print('saved', fname)
        plt.close(fig)


plot_dists(params)
##

basedir = Path('~/Desktop/calib/single2').expanduser()
basedir.mkdir(exist_ok=True)

for fname, params in zip(fnames, selectedparams):
    plot_dists(params, fname=basedir / fname)


##────────────────────────────────────────────────────────────────────────────}}}# TODO


### {{{                            --     old     --
def plot_calib(axpairs, params, layout, fname=None):
    fig, axes = du.mkfig(*layout, (5, 5))
    axes = axes.flatten()
    for i, (ax0, ax1) in enumerate(axpairs):
        ax = axes if len(axpairs) == 1 else axes[i]
        xx = params['a'] * centroids + params['b']
        data_tr = params['a'] * logdata + params['b']
        ax.scatter(data_tr[:, ax0], data_tr[:, ax1], s=0.2, alpha=0.2, color='k', linewidth=0)
        # ax.scatter(xx[1:, ax0], xx[1:, ax1], s=25, alpha=1)
        ax.scatter(logcalib[1:, ax0], logcalib[1:, ax1], s=25, alpha=1, marker='x', color='red')
        # add text "bead 1" etc near the logcalib points
        for i in range(1, centroids.shape[0]):
            # slight offset to avoid overlap, fontsize 8
            ax.text(
                logcalib[i, ax0] + 0.01,
                logcalib[i, ax1],
                f'bead {i}',
                fontsize=8,
                color='red',
            )
        ax.set_xlabel(color_channels[ax0])
        ax.set_ylabel(color_channels[ax1])
        ax.set_xlabel(
            f'{color_channels[ax0]} (calibrated to {channels_to_units[color_channels[ax0]]})'
        )
        ax.set_ylabel(
            f'{color_channels[ax1]} (calibrated to {channels_to_units[color_channels[ax1]]})'
        )
        ax.set_xlim(0, 1.1)
        ax.set_ylim(0, 1.1)

    if fname is not None:
        fig.savefig(fname, dpi=140)
        print('saved', fname)

    plt.close(fig)


plot_calib(axpairs, params, layout=(4, 9))
##


# basedir = Path('~/Desktop/calib').expanduser()
# basedir.mkdir(exist_ok=True)
# logparams = np.logspace(0, np.log10(len(all_params) - 1), 100).astype(int)
# logparams = np.unique(logparams)
# selectedparams = [all_params[i] for i in logparams]
# fnames = [f'calib_step_{i}.png' for i in range(len(selectedparams))]

# generate all pairs of axes
# from itertools import combinations

# axpairs = list(combinations(range(centroids.shape[1]), 2))
# len(axpairs)


# for fname, params in zip(fnames, selectedparams):
# plot_calib(axpairs, params, layout=(4, 9), fname=basedir / fname)


##────────────────────────────────────────────────────────────────────────────}}}##
