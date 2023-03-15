import jax
import optax
from jax import jit, vmap
import jax.numpy as jnp
from jax.tree_util import Partial as partial
from jax.scipy.stats import gaussian_kde

from ott.geometry.pointcloud import PointCloud
from ott.problems.linear.linear_problem import LinearProblem
from ott.solvers.linear.sinkhorn import Sinkhorn

from scipy.interpolate import UnivariateSpline

import time
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import flowio


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
    'Pacific_Blue_A': 'PacBlue',
    'AmCyan_A': 'MEAMCY',
    'FITC_A': 'MEFL',
    'PerCP_Cy5_5_A': 'MEPCY5.5',
    'PE_A': 'MEPE',
    'PE_Texas_Red_A': 'MEPTR',
    'APC_A': 'MEAPC',
    'APC_Alexa_700_A': 'MEAX700',
    'APC_Cy7_A': 'MEAPCCY7',
}

FORTESSA_MAX_V = 2**18 - 1

FORTESSA_OFFSET = 200

##────────────────────────────────────────────────────────────────────────────}}}

### {{{              --     small tools     --
def escape_name(name):
    return name.replace('-', '_').replace(' ', '_').upper().rstrip('_A')


def escape(names):
    if isinstance(names, str):
        return escape_name(names)
    if isinstance(names, list):
        return [escape_name(name) for name in names]
    if isinstance(names, tuple):
        return tuple([escape_name(name) for name in names])
    if isinstance(names, dict):
        return {escape_name(k): escape_name(v) for k, v in names.items()}
    else:
        return names


def astuple(x):
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    return (x,)


# weights observations: 0 if out of range, 1 if in range
def w_function(x, a=0, b=0.99, lsteepness=10, rsteepness=1000):
    y1 = 1 / (1 + jnp.exp(-lsteepness * (x - a)))
    y2 = 1 / (1 + jnp.exp(-rsteepness * (x - b)))
    return jnp.clip(y1 - y2, 0, 1)


def load_fcs_to_df(fcs_file):
    fcs_data = flowio.FlowData(fcs_file.as_posix())
    channels = [fcs_data.channels[str(i + 1)]['PnN'] for i in range(fcs_data.channel_count)]
    original_data = np.reshape(fcs_data.events, (-1, fcs_data.channel_count))
    return pd.DataFrame(original_data, columns=escape(channels))


def load_to_df(file, column_order=None):
    file = Path(file)
    assert file.exists(), f'File {file} does not exist'
    if file.suffix == '.fcs':
        df = load_fcs_to_df(file)
    elif file.suffix == '.csv':
        df = pd.read_csv(file)
        df.columns = escape(list(df.columns))
    else:
        raise ValueError(f'File {file} has unknown suffix {file.suffix}')
    if column_order is not None:
        df = df[escape(column_order)]
    return df


def logtransform(x, scale, offset=0):
    x = jnp.clip(x, 1e-9, None)
    return jnp.log10(x + offset) / scale


def inverse_logtransform(x, scale, offset=0):
    return 10 ** (x * scale) - offset


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                  --     beads related functions     --



CALIBRATION_PEAKS_MIN_X = -0.025
CALIBRATION_PEAKS_MAX_X = 1.025


@partial(jit, static_argnums=(2, 3, 4))
def compute_peaks(
    observations,
    beads,
    resolution=1000,
    bw_method=0.2,
    max_obs=20000,
    peaks_min_x=CALIBRATION_PEAKS_MIN_X,
    peaks_max_x=CALIBRATION_PEAKS_MAX_X,
):
    """Compute coordinates of each bead's peak in all channels"""

    # First we compute the vote matrix:
    # it's a (CHANNEL, OBSERVATIONS, BEAD) matrix
    # where each channel tells what affinity each observation has to each bead, from the channel's perspective.
    # This is computed using optimal transport (it's the OT matrix)
    # High values mean it's obvious in this channel that the observation should be paired with a certain bead.
    # Low values mean it's not so obvious, usually because the observation is not in the valid range
    # so there's a bunch of points around this one that could be paired with any of the remaining beads
    # This is much more robust than just computing OT for all channels at once

    @jit
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
    weights = vmap(w_function)(observations).T  # (CHANNELS, OBSERVATIONS) -> "validity" from range
    weighted_votes = votes * weights[:, :, None] + 1e-12  # (CHANNELS, OBSERVATIONS, BEADS)
    vmat = jnp.sum(weighted_votes, axis=0) / jnp.sum(weights, axis=0)[:, None]  # weighted average

    # Use these votes to decide which beads are the most likely for each observation
    # Tried with a softer version of this just in case, but I couldn't see any improvement
    vmat = jnp.argmax(vmat, axis=1)[:, None] == jnp.arange(vmat.shape[1])[None, :]

    # We add some tiny random normal noise to avoid singular matrix errors when computing the KDE
    # on a bead that would have only the exact same value (which can happen when out of range)
    noise_std = (peaks_max_x - peaks_min_x) / (resolution * 5)
    obs = observations + jax.random.normal(jax.random.PRNGKey(0), observations.shape) * noise_std

    # Now we can compute the densities for each bead in each channel
    x = jnp.linspace(peaks_min_x, peaks_max_x, resolution)
    w_kde = lambda s, w: gaussian_kde(s, weights=w, bw_method=bw_method)(x)
    densities = jit(vmap(vmap(w_kde, in_axes=(None, 1)), in_axes=(1, None)))(obs, vmat)
    densities = densities.transpose(1, 0, 2)  # densities.shape is (BEADS, CHANNELS, RESOLUTION)

    # find the most likely peak positions for each bead using the average intensity weighted by density
    DENSITY_TH = (
        0.25  # min threshold (as fraction of max) for density to be included in the average
    )
    RANGE_TH = (
        0.2  # min weight from the validity range (to avoid giving importance to saturated values)
    )
    normalized_densities = densities / jnp.max(densities, axis=2)[:, :, None]
    w = jnp.where(
        (normalized_densities > DENSITY_TH) & (w_function(x) > RANGE_TH)[None, None, :],
        normalized_densities - DENSITY_TH,
        0,
    )
    w = jnp.where(
        normalized_densities > DENSITY_TH, jnp.maximum(w, 1e-9), 0
    )  # still some tiny weight when OOR
    xx = jnp.tile(x, (densities.shape[0], densities.shape[1], 1))
    peaks = jnp.average(xx, axis=2, weights=w)  # peaks.shape is (BEADS, CHANNELS)

    # let's also estimate a confidence score for each peak
    # we will compute that by how much overlap ther is between each bead's distributions

    return peaks, (densities, vmat)


def beads_fit_affine(logpeaks, logcalib, num_iter=5000, learning_rate=0.01, ignore_first_bead=True):
    # simple full batch gradient descent

    affine_transform = lambda params, x: params['a'] * x + params['b']

    beads_init = lambda NCHAN: {
        'a': jnp.ones((NCHAN,)),
        'b': jnp.zeros((NCHAN,)),
    }

    NCHAN = logpeaks.shape[1]
    params = beads_init(NCHAN)
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    X = jnp.where(jnp.isnan(logpeaks), 0, logpeaks)
    weights = vmap(w_function, in_axes=0)(logpeaks)
    xrange = X.max(axis=0) - X.min(axis=0)
    # ignore top 5% and bottom 5% of values
    weights = jnp.where(
        (X < X.min(axis=0) + xrange * 0.05) | (X > X.max(axis=0) - xrange * 0.05), 0, weights
    )
    if ignore_first_bead:
        weights = weights.at[0, :].set(0)

    @jax.value_and_grad
    def lossf(params):
        x = affine_transform(params, X)
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




def beads_fit_spline(
    logpeaks, logcalib, spline_degree=2, smooth_percent=1, CF=100, ignore_first_bead=True
):
    X = logpeaks * CF  # (NBEADS, NCHAN)
    Y = logcalib * CF
    NBEADS, NCHAN = X.shape

    # it's necessary that the function is monotonous for invertibility
    ydiff = np.diff(Y, axis=0)
    assert np.all(ydiff >= 0)

    # any distance between peaks that is less than MIN_PERCENT_DIST%
    # of the range will be given a weight that's inversely proportional to the distance
    # Peaks so close together probably means the bead data is overlapping in this region
    # and we can't be that confident in the fit
    MIN_PERCENT_DIST = 3
    dist_fwd = np.abs(X[1:] - X[:-1])
    dist_fwd = np.concatenate([dist_fwd[0:1], dist_fwd], axis=0)
    dist_bwd = np.abs(X[:-1] - X[1:])
    dist_bwd = np.concatenate([dist_bwd, dist_bwd[-1:]], axis=0)
    avg_dist_to_neigh = (dist_fwd + dist_bwd) / 2
    w = np.clip(avg_dist_to_neigh / MIN_PERCENT_DIST, 0, 1)
    # check that X is ordered
    order = np.argsort(X, axis=0)
    # detect if there are any beads that are out of order and put their weight to a very small value
    w = np.where(order != np.arange(0, NBEADS)[:, None], 1e-6, w)

    # reorder, but only X
    X = X[order, np.arange(0, NCHAN)]

    if ignore_first_bead:  # almost...
        w[0, :] = 1e-6

    splines = [
        UnivariateSpline(
            X[:, c], Y[:, c], k=spline_degree, s=smooth_percent, check_finite=True, w=w[:, c]
        )
        for c in range(0, NCHAN)
    ]
    inv_splines = [
        UnivariateSpline(
            Y[:, c], X[:, c], k=spline_degree, s=smooth_percent, check_finite=True, w=w[:, c]
        )
        for c in range(0, NCHAN)
    ]

    def make_transform(s):
        def transform(x):
            return s(x * CF) / CF

        return transform

    transforms = [make_transform(s) for s in splines]
    inv_transforms = [make_transform(s) for s in inv_splines]
    return transforms, inv_transforms


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                     --     diagnostic plots     --
def plot_bead_peaks_diagnostics(
    peaks,
    densities,
    vmat,
    observations,
    color_channels,
    max_obs=20000,
    peaks_min_x=CALIBRATION_PEAKS_MIN_X,
    peaks_max_x=CALIBRATION_PEAKS_MAX_X,
):
    """
    Plot diagnostics for the bead peaks computation:
    - An assignment chart showing the proportion of observations assigned to each bead
    - 1 plot per channel showing both bead assignment, peak detection and final standardized densities
    """

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
    x = jnp.linspace(peaks_min_x, peaks_max_x, resolution)
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
                extent=(peaks_min_x, peaks_max_x, 0, 1.5),
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
            ax.set_xlim(peaks_min_x, 1.1)
            title = f'{color_channels[c]}'
            ax.set_ylabel(title)

        axes[0].set_title(
            'Density of observations across channel range (grey background = out of range)'
        )
        # color: density of observations per bead assignment
        # --: estimated peak position\n
        # dark curve: original distribution of observations in channel
    subfigs[1].suptitle("Bead peak diagnostics")


def remove_axis_and_spines(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_fluo_distribution(ax, data, res=2000):
    from jax.scipy.stats import gaussian_kde

    xmax = np.max(data)
    XX = np.linspace(0.0, 1.1 * xmax, res)
    kde = gaussian_kde(data.T, bw_method=0.01)
    densities = kde(XX.T)
    ldensities = np.log10(1.0 + densities)
    densities = (densities / densities.max()) * 0.4
    ldensities = (ldensities / ldensities.max()) * 0.4
    ax.fill_betweenx(XX, -ldensities, 0, color='k', alpha=0.2, lw=0)
    ax.fill_betweenx(XX, -densities, 0, color='k', alpha=0.2, lw=0)
    ax.plot(-densities, XX, color='k', alpha=0.5, lw=0.7)
    ax.set_xlim(-0.5, 0.5)
    remove_axis_and_spines(ax)


def plot_beads_after_correction(
    transform,
    peaks,
    calib,
    observations,
    color_channels,
    channel_to_unit,
    max_obs=20000,
    title=None,
    fname=None,
):
    from matplotlib import cm

    if observations.shape[0] > max_obs:
        key = jax.random.PRNGKey(0)
        reorder = jax.random.permutation(key, np.arange(max_obs))
        observations = observations[reorder]

    NBEADS, NCHAN = peaks.shape
    fig, axes = plt.subplots(1, NCHAN, figsize=(2.2 * NCHAN, 12))
    tdata = jnp.array([t(o) for t, o in zip(transform, observations.T)]).T
    tpeaks = jnp.array([t(p) for t, p in zip(transform, peaks.T)]).T
    for c in range(NCHAN):
        ax = axes[c]
        error = np.average(np.abs(tpeaks[1:, c] - calib[1:, c]))
        # add beads as horizontal lines
        for b in range(0, NBEADS):
            alpha = 0.7
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
                calib[b, c] + 0.01,
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

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()

    if fname is not None:
        fig.savefig(fname, dpi=150)
        print('saved', fname)
        plt.close(fig)


# plot the spectral signature matrix
def plot_spectral_signature(S, ax, prots=None, channels=None):
    ax.imshow(S, cmap='Reds')
    ax.set_xticks(range(S.shape[1]))
    ax.set_yticks(range(S.shape[0]))
    ax.set_title('Spectral signature matrix')
    ax.set_xlabel('Channels')
    ax.set_ylabel('Proteins')
    if channels is not None:
        # rotate
        ax.set_xticklabels(channels, rotation=90)

    if prots is not None:
        ax.set_yticklabels(prots)


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                 --     simple optimization functions   --


def spectral_bleedthrough(Y, M, max_iterations=50, jax_seed=0, verbose=False):
    # Solves with Alternating Least Square using closed form solutions (pinv)

    # model: Y = XM.S
    # Y: observations
    # M: masks (from controls: single color = (1,0,...))
    # S: spectral signature
    # X: estimated quantity of protein.

    normalize = lambda x: x / (jnp.maximum(jnp.max(x, axis=1)[:, None], 1e-12))
    K = jax.random.uniform(jax.random.PRNGKey(jax_seed), (M.shape[0],), minval=0.1)

    @jit
    def alsq(K):  # one iteration of alternating least squares
        S = normalize(jnp.linalg.pinv(K[:, None] * M) @ Y)  # find S from K
        K = Y @ jnp.linalg.pinv(S)  # find K from S
        K = jnp.average(K, axis=1, weights=M)
        K = jnp.maximum(jnp.nan_to_num(K, nan=0), 0)
        return S, K

    Sprev = 0
    for _ in range(max_iterations):
        S, K = alsq(K)
        error = ((Y - (K[:, None] * M) @ S) ** 2).mean() / jnp.linalg.norm(Y)
        if verbose:
            print(f'error: {error:.2e}, update: {jnp.mean((Sprev - S)**2):.2e}')
        Sprev = S

    return S, K


def affine_opt(Ylog, ref_channel):
    # Ylog is a matrix of log-transformed fluorescence values
    # we need to find the best affine mapping to the reference channel
    # Ylog[:, i] = a_i * Ylog[:, refchanid] + b_i
    # so we solve the linear system
    # Ylog[:, i] = A @ [a_i, b_i] (with A = [Ylog[:, refchanid], 1])

    ref = Ylog[:, ref_channel]

    def solve_column(y):
        A = jnp.stack([y, jnp.ones_like(y)], axis=1)
        return jnp.linalg.lstsq(A, ref, rcond=None)[0]

    res = vmap(solve_column)(Ylog.T)
    a, b = res[:, 0], res[:, 1]
    return a, b


def affine_gd(Y, ref_channel, num_iter=1000, learning_rate=0.1, max_N=500000, verbose=False):
    # gradient descent version of the above affine fit

    NCHAN = Y.shape[1]
    if len(Y) > max_N:
        choice = jax.random.choice(jax.random.PRNGKey(0), len(Y), (min(max_N, len(Y)),))
        Y = Y[choice]

    params = {
        'a': jnp.ones((NCHAN,)),
        'b': jnp.zeros((NCHAN,)),
        'c': jnp.zeros((NCHAN,)),
    }

    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    Yref = Y[:, ref_channel]

    @jax.value_and_grad
    def lossf(params):
        yhat = params['a'] * Y + params['b']
        yhat = jnp.maximum(yhat, 0)
        err = (yhat - Yref[:, None]) ** 2
        err = err
        return jnp.mean(err**2)

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
        if verbose:
            print(f'iter {i}: loss = {loss:.2e}')

    a, b = params['a'], params['b']
    return a, b


from tqdm import tqdm


from scipy.interpolate import LSQUnivariateSpline

DEFAULT_CMAP_Q = np.array([0.7, 0.8])  # where to put spline knots
DEFAULT_CLAMP_Q = np.array([0.0001, 0.9999])  # where to clamp values


def cmap_spline(X, Y, n_knots=1, spline_order=3, CF=100, Q=DEFAULT_CMAP_Q):
    X, Y = X * CF, Y * CF
    x_order = np.argsort(X, axis=0)
    Xx, Yx = np.take_along_axis(X, x_order, axis=0), Y[x_order]
    w = 0.25 * np.ones_like(Y)
    X6 = Xx[-6]
    xknots = np.linspace(np.quantile(X, Q[0], axis=0), np.quantile(X, Q[1], axis=0), n_knots).T
    splines = [
        LSQUnivariateSpline(Xx[:, c], Yx[:, c], k=spline_order, t=xknots[c], w=w)
        for c in tqdm(range(X.shape[1]))
    ]
    y_order = np.argsort(Y, axis=0)
    Xy, Yy = X[y_order], Y[y_order]
    yknots = np.linspace(
        np.quantile(Yy, Q[0], axis=0), np.quantile(Yy, Q[-1], axis=0), n_knots, endpoint=True
    ).T
    inv_splines = [
        LSQUnivariateSpline(Yy, Xy[:, c], k=spline_order, t=yknots, w=w) for c in range(X.shape[1])
    ]
    transform = lambda X: np.array([s(x * CF) / CF for s, x in zip(splines, X.T)]).T
    inv_transform = lambda Y: np.array([s(y * CF) / CF for s, y in zip(inv_splines, Y.T)]).T

    return transform, inv_transform


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                     --     calibration class     --

class Calibration:
    """
    The Calibration class handles the calibration of the fluorescence values of a given experiment
    using color controls, blank controls, and some beads of known fluorescence values.
    """

    ### {{{                           --     init     --

    def __init__(
        self,
        color_controls_files: dict[tuple[str], str],
        beads_file: str,
        reference_protein: str = 'mkate',
        reference_channel: str = 'PE_TEXAS_RED',
        beads_mef_values: dict[str, list[float]] = SPHEROTECH_RCP_30_5a,
        channel_to_unit: dict[str, str] = FORTESSA_CHANNELS,
        random_seed: int = 42,
        use_channels: list[str] = None,
        max_value=FORTESSA_MAX_V,
        offset=FORTESSA_OFFSET,
    ):
        """
        Parameters
        ----------
        color_controls_files : dict[tuple[str], str] - dictionary {control protein : path to fcs or csv)
            e.g. {'eYFP': 'path/to/eyfp-control.fcs',
                  'eBFP': 'path/to/ebfp-control.fcs',
                  'all OR allcolor OR allcolors OR full': 'path/to/allcolor-control.fcs'
                  'blank OR empty OR inert OR cntl ': 'path/to/allcolor-control.fcs' }

        beads_file : str - path to the beads control fcs file

        reference_protein : str - name of the reference protein that will be used to align all
                            other proteins quantities to, i.e they will be expressed in the same unit
                            The channel is picked as the brightest channel of the reference protein

        reference_channel : str - name of the reference channel that will be used to convert
                            to standatd MEF units

        beads_mef_vlaues : dict[str, list[float]] - dictionary associating the unit name
            to the list of reference values for the beads in this unit
            e.g : {'MEFL': [456, 4648, 14631, 42313, 128924, 381106, 1006897, 2957538, 7435549], 'MEPE': ...}

        channel_to_unit : dict[str, str] - dictionary associating the channel name to the unit name
            e.g. {'Pacific Blue-A': 'PacBlue', 'AmCyan-A': 'MEAMCY', ...}

        random_seed : int - random seed for any resampling that might be done

        use_channels : list[str] - list of channels to use when computing protein quantity 
                        from channel values (aka "bleedthrough correction"). 
                        None means all available channels

        max_value : int - maximum value of the fluorescence values, used to detect saturation

        offset : int - offset to add to the fluorescence values to avoid negative values

        """
        self.color_controls_files = color_controls_files
        self.beads_file = beads_file
        self.reference_protein = escape(reference_protein)
        self.reference_channel = escape(reference_channel)
        self.beads_mef_values = {escape(k): v for k, v in beads_mef_values.items()}
        self.channel_to_unit = {escape(k): escape(v) for k, v in channel_to_unit.items()}
        self.unit_to_channel = {v: k for k, v in self.channel_to_unit.items()}
        self.random_seed = random_seed
        self.__fitted = False
        self.__channel_order = sorted(list(self.channel_to_unit.keys()))
        # remove channels that are not in the channel_to_unit dict
        if use_channels is not None:
            self.__channel_order = [c for c in self.__channel_order if c in escape(use_channels)]

        self.controls = {
            astuple(escape(k)): load_to_df(v, self.__channel_order)
            for k, v in color_controls_files.items()
        }

        self.__beads_channel_order = sorted(list(self.channel_to_unit.keys()))
        self.beads = load_to_df(self.beads_file, self.__beads_channel_order)

        self.__max_value = max_value
        self.__offset = offset

        VALID_BLANK_NAMES = ['BLANK', 'EMPTY', 'INERT', 'CNTL']
        VALID_ALL_NAMES = ['ALL', 'ALLCOLOR', 'ALLCOLORS', 'FULL']

        controls_order = sorted(list(self.controls.keys()))
        self.__fluo_proteins = sorted(
            list(
                set(
                    [
                        p
                        for c in controls_order
                        for p in c
                        if p not in VALID_BLANK_NAMES + VALID_ALL_NAMES
                    ]
                )
            )
        )

        assert self.reference_protein in self.__fluo_proteins

        controls_values, controls_masks = [], []
        NPROTEINS = len(self.__fluo_proteins)
        has_blank, has_all = False, False
        for m in controls_order:
            vals = self.controls[m].values
            if m in [astuple(escape(n)) for n in VALID_BLANK_NAMES]:
                base_mask = np.zeros((NPROTEINS,))
                has_blank = True
            elif m in [astuple(escape(n)) for n in VALID_ALL_NAMES]:
                base_mask = np.ones((NPROTEINS,))
                has_all = True
            else:
                for p in m:
                    assert p in self.__fluo_proteins, f'Unknown protein {p}'
                base_mask = np.array([p in m for p in self.__fluo_proteins])

            masks = np.tile(base_mask, (vals.shape[0], 1))
            controls_values.append(vals)
            controls_masks.append(masks)

        self.__controls_values = jnp.vstack(controls_values)
        self.__controls_masks = jnp.vstack(controls_masks)
        assert self.__controls_values.shape == self.__controls_masks.shape

        assert has_blank, f'BLANK control not found. Should be named any of {VALID_BLANK_NAMES}'
        assert has_all, f'ALLCOLOR control not found. Should be named any of {VALID_ALL_NAMES}'

    ##────────────────────────────────────────────────────────────────────────────}}}

    def fit(self):
        t0 = time.time()
        print('Estimating autofluorescence...')
        zero_masks = self.__controls_masks.sum(axis=1) == 0
        self.__autofluorescence = np.median(self.__controls_values[zero_masks], axis=0)

        print('Computing bleedthrough matrix...')
        single_masks = self.__controls_masks.sum(axis=1) == 1
        masks = self.__controls_masks[single_masks]
        Y = self.__controls_values[single_masks] - self.__autofluorescence
        self.__bleedthrough_matrix, _ = spectral_bleedthrough(Y, masks)

        print('Computing peaks assignment...')
        refprotid = self.__fluo_proteins.index(self.reference_protein)
        refchanid = jnp.argmax(self.__bleedthrough_matrix[refprotid])
        self.cal_units = self.channel_to_unit[self.__channel_order[refchanid]]

        # normalizing beads values to [0;1] (fo the observations, not necessarily the reference values)
        self.__log_scale_factor = jnp.log10(self.__max_value + self.__offset)
        calib_values = jnp.array(
            [self.beads_mef_values[self.channel_to_unit[c]] for c in self.__beads_channel_order]
        ).T
        self.__log_beads_mef = logtransform(calib_values, self.__log_scale_factor, 0)
        self.__log_beads_data = logtransform(
            self.beads.values, self.__log_scale_factor, self.__offset
        )
        self.__log_beads_peaks, (self.__beads_densities, self.__beads_vmat) = compute_peaks(
            self.__log_beads_data, self.__log_beads_mef
        )

        print('Computing channel calibration to MEF...')
        self.beads_transform, self.beads_inv_transform = beads_fit_spline(
            self.__log_beads_peaks, self.__log_beads_mef
        )

        print('Computing color mapping...')  # yi = xi mi S ;  gi = ai * log(yi) + bi
        all_masks = self.__controls_masks.sum(axis=1) > 1
        fluo_X = self.__controls_values[all_masks] - self.__autofluorescence
        X = fluo_X @ jnp.linalg.pinv(self.__bleedthrough_matrix)
        X += self.__offset  # add offset
        refprotid = self.__fluo_proteins.index(self.reference_protein)
        X = X[jnp.all(X > 1, axis=1)]
        logX = logtransform(X, self.__log_scale_factor, 0)
        self.cmap_transform, self.cmap_inv_transform = cmap_spline(logX, logX[:, refprotid])
        self.clamp_values = jnp.quantile(logX, DEFAULT_CLAMP_Q, axis=0)

        self.__fitted = True
        print('Done fitting in {:.2f} seconds'.format(time.time() - t0))

    def apply_bleedthrough_correction(self,Y):
        assert self.__fitted, 'You must fit the calibration first'
        X = Y @ jnp.linalg.pinv(self.__bleedthrough_matrix)  # apply bleedthrough
        X += self.__offset  # add offset
        X = X[jnp.all(X > 1, axis=1)]
        return X

    def apply_to_array(self, Y):

        assert self.__fitted, 'You must fit the calibration first'
        Y = Y - self.__autofluorescence  # remove autofluorescence
        X = self.apply_bleedthrough_correction(Y)

        logX = logtransform(X, self.__log_scale_factor, 0)
        logX = logX[jnp.all(logX > self.clamp_values[0], axis=1)]
        logX = logX[jnp.all(logX < self.clamp_values[1], axis=1)]
        cmapX = self.cmap_transform(logX)

        refchanid = self.__beads_channel_order.index(self.reference_channel)
        calibratedX = jnp.array(
            [self.beads_transform[refchanid](cmapX[:, i]) for i in range(cmapX.shape[1])]
        ).T

        finalX = inverse_logtransform(calibratedX, self.__log_scale_factor)

        return finalX

    def apply(self, df):
        assert self.__fitted, 'You must fit the calibration first'
        df.columns = escape(list(df.columns))
        df = df[self.__channel_order]
        X = self.apply_to_array(df.values)
        return pd.DataFrame(X, columns=self.__fluo_proteins)

    def plot_beads_diagnostics(self):
        if self.__log_beads_peaks is None:
            raise ValueError('You must fit the calibration first')

        plot_bead_peaks_diagnostics(
            self.__log_beads_peaks,
            self.__beads_densities,
            self.__beads_vmat,
            self.__log_beads_data,
            self.__beads_channel_order,
        )

        plot_beads_after_correction(
            self.beads_transform,
            self.__log_beads_peaks,
            self.__log_beads_mef,
            self.__log_beads_data,
            self.__beads_channel_order,
            self.channel_to_unit,
            title="""Beads-based MEF calibration
            right side shows beads intensities in their respective MEF units
            left side shows the real beads data after calibration\n\n"""
        )

    def plot_color_mapping_diagnostics(self):
        fig, ax = plt.subplots(1, 1)
        # plot transform
        NP = len(self.__fluo_proteins)
        all_masks = self.__controls_masks.sum(axis=1) > 1
        Y = self.__controls_values[all_masks] - self.__autofluorescence
        X = self.apply_bleedthrough_correction(Y)
        logX = logtransform(X, self.__log_scale_factor, 0)
        logX = logX[jnp.all(logX > self.clamp_values[0], axis=1)]
        logX = logX[jnp.all(logX < self.clamp_values[1], axis=1)]
        print(f'clamp values: {self.clamp_values}')
        target = logX[:, self.__fluo_proteins.index(self.reference_protein)]
        for i in range(NP):
            xx = np.linspace(self.clamp_values[0,i], self.clamp_values[1,i], 200)
            print(f'xx range: {xx.min()} {xx.max()}')
            yy = self.cmap_transform(np.tile(xx, (NP,1)).T)[:, i]
            print(f'shapes: {xx.shape} {yy.shape}')
            ax.scatter(logX[:, i], target, s=5, alpha=0.05, marker='.', linewidths=0)
            ax.plot(xx, yy, alpha=1, lw=2.5, color='w')
            ax.plot(xx, yy, label=self.__fluo_proteins[i], alpha=1, lw=1, color='C%d' % i)
        yrange = self.clamp_values.min()*0.9, self.clamp_values.max()*1.05
        ax.set_ylim(yrange)
        ax.set_xlim(yrange)
        ax.legend()
        ax.set_xlabel('source')
        ax.set_ylabel('target')
        ax.set_title(f'Color mapping to {self.reference_protein}')


##────────────────────────────────────────────────────────────────────────────}}}##
