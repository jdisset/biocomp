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


def w_function(x, a=0.3, b=0.72, lsteepness=70, rsteepness=400):
    y1 = 1 / (1 + jnp.exp(-lsteepness * (x - a)))
    y2 = 1 / (1 + jnp.exp(-rsteepness * (x - b)))
    return jnp.clip(y1 - y2, 0, 1)


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
    densities = densities.transpose(1, 0, 2)  # densities.shape is (BEADS, CHANNELS, RESOLUTION)

    peaks = x[jnp.argmax(densities, axis=2)]  # peaks.shape is (BEADS, CHANNELS)

    return peaks, densities


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
    'AmCyan-A': 'MEAMCY',
    'FITC-A': 'MEFL',
    'PerCP-Cy5-5-A': 'MEPCY5.5',
    'PE-A': 'MEPE',
    'PE-Texas Red-A': 'MEPTR',
    'APC-A': 'MEAPC',
    'APC-Alexa 700-A': 'MEAX700',
    'APC-Cy7-A': 'MEAPCCY7',
}


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                     --     calibration class     --

class Calibration:
    """
    The Calibration class handles the calibration of the fluorescence values of a given experiment
    using color controls, blank controls, and some beads of known fluorescence values.
    """

    def __init__(
        self,
        blank: str,
        color_controls: dict[tuple[str], str],
        beads: str,
        beads_reference_values: dict[str, list[float]] = SPHEROTECH_RCP_30_5a,
        channel_to_unit: dict[str, str] = FORTESSA_CHANNELS,
    ):
        """
        Parameters
        ----------
        blank : str - path to the blank control fcs file

        color_controls : dict[tuple[str], str] - dictionary associating a list of the proteins in the control to the path for the fcs control file
            e.g. {('eYFP',): 'path/to/eyfp-control.fcs', ('eBFP',): 'path/to/ebfp-control.fcs', ('eYFP', 'eBFP'): 'path/to/allcolor-control.fcs'}

        beads : str - path to the beads control fcs file

        beads_reference_vlaues : dict[str, list[float]] - dictionary associating the unit name
            to the list of reference values for the beads in this unit
            e.g : {'MEFL': [456, 4648, 14631, 42313, 128924, 381106, 1006897, 2957538, 7435549], 'MEPE': ...}

        channel_to_unit : dict[str, str] - dictionary associating the channel name to the unit name
            e.g. {'Pacific Blue-A': 'PacBlue', 'AmCyan-A': 'MEAMCY', ...}

        """
        self.blank = blank
        self.color_controls = color_controls
        self.beads = beads
        self.beads_reference_values = beads_reference_values
        self.channel_to_unit = channel_to_unit
        self.__autofluorescence = None
        self.__spectral_signature_matrix = None
        self.__fitted = False

    def fit(self, standardize_to_unit='MEFL', **kwargs):
        """
        Fit the calibration parameters to the controls.
        """
        # The general idea is that we can express fluorescence as a
        # linear combination of the spectral signatures of the proteins + autofluorescence.
        # So for each observation Y_i, we have Y_i = X_i @ S + A
        # where X_i is the vector representing the quantity of fluorochrome of each type of shape (n_proteins,)
        # S is the matrix of spectral signatures of shape (n_proteins, n_channels),
        # and A is the vector of autofluorescence of shape (n_channels,).
        # Y_i is the observation, a vector of fluorescence intensities of shape (n_channels,).
        # The goal of this optimization is to find S and A such that Y_i = X_i @ S + A for all i. (i.e. minimize the error)
        # However for each observation, we only have the fluorescence intensities Y_i, and we don't know X_i.
        # The "trick" is that we have many controls where, although we don't know X_I exactly, we know
        # that X_i should be of the form x_i * M where x_i is a scalar and M is a "mask" vector of shape (n_proteins,).
        # For example, if we have a control with only eYFP, then M = [1, 0, 0, ...] and x_i is the amount of eYFP.
        # If we have a control with eYFP and eBFP, then M = [1, 0, 1, ...] and x_i is the amount of eYFP + eBFP.
        # And to estimate the autofluorescence, we can use the blank control, where M = [0, 0, 0, ...]
        # As long as we have enough controls to constrain the problem, we should be able to estimate S and A pretty well.

        # With the controls only, we can estimate *a* value for S and A, but then X will be in arbitrary units.
        # In order to standardize things and constrain X to reproducible units, we can use the beads:
        # we need to fist compute the functions Fstd that convert a channel to it's standardized units (using the beads).
        # Then if we want to standardize to, let's say, MEFL, we need to add another constrain that the
        # value of X_i[MEFL] should be equal to Fstd[MEFL](Y_i[MEFL]).

        # Let's GO

        # First, we prepare all data with the corresponding mask.

        # The blank control

    def apply(
        self, fcs_file: str, fluorescent_proteins: Optional[list[str]] = None, **kwargs
    ) -> pd.DataFrame:
        """
        Apply the calibration to the fcs file. The input file contains raw, but gated, intensity data for all channels.
        The returned dataframe contains the standardized count of fluorescent proteins. If fluorescent_proteins is not None,
        only the count of the specified proteins will be returned (in the same order as in the list), and it will help with the
        precision of the count. If fluorescent_proteins is None, all the proteins will be estimated.

        Parameters
        ----------
        fcs_file : str - path to the fcs file to calibrate

        Returns
        -------
        pd.DataFrame - the calibrated fluorescen
        """
        if not self.__fitted:
            self.fit(**kwargs)

        # apply...



##────────────────────────────────────────────────────────────────────────────}}}##

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


# first, let's grab all the channels by associating the non gated raw files' rows to the gated ones
data_dir = ut.DEFAULT_XP_PATH / '2023-01-22_CasE_ALLuORFs/data/'
fcs = list(Path(data_dir / 'raw_data/').rglob('*.fcs'))
fcs = {f.stem: f for f in fcs}
loaded_fcs = {k: load_fcs_to_df(v) for k, v in fcs.items()}


list(loaded_fcs.keys())


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                          --     gating?     --
# df = controls[tuple()]
# join all controls in df
df = pd.concat(controls.values(), ignore_index=True)

# keep only the 0.001 - 0.999 quantile in SSC-A and FSC-A
df = df[(df['SSC-A'] > df['SSC-A'].quantile(0.001)) & (df['SSC-A'] < df['SSC-A'].quantile(0.999))]

def scatter_density(x, y, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    xy = np.vstack([x, y])
    xy_sub = xy[:, np.random.choice(xy.shape[1], 2000)]
    kde = jit(gaussian_kde)(xy_sub)
    # z = kde(xy)
    contour_res = 250
    xx = np.linspace(x.min(), x.max(), contour_res)
    yy = np.linspace(y.min(), y.max(), contour_res)
    mgrid = np.meshgrid(xx, yy)
    mgrid = np.vstack([mgrid[0].ravel(), mgrid[1].ravel()])
    zz = kde(np.vstack(mgrid)).reshape(contour_res, contour_res)
    # don't display first contour
    ax.contour(xx, yy, zz, 5, colors='k', alpha=0.5, linewidths=0.5)

    dthreshold = np.quantile(z, 0.3)
    # display a background image of the density with green when > 0.1 quantile, red otherwise
    # custom cmap (green to red)
    cmap = mpl.colors.ListedColormap(['red', 'green'])
    bounds = [0, dthreshold, 1]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    # using pcolormesh
    ax.scatter(x, y, cmap='inferno', **kwargs)
    ax.pcolormesh(xx, yy, zz, cmap=cmap, norm=norm, alpha=0.3)

    # log scales
    return ax

fig, ax = du.mkfig(1,1)
scatter_density(df['SSC-A'], df['FSC-A'], ax=ax, s=0.25, alpha=0.25, linewidths=0)
ax.set_xlabel('SSC-A')
ax.set_ylabel('FSC-A')
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlim(1e0, 1e7)
# ax.set_ylim(1e0, 1e7)

##
# simple interactive plot to draw a polygon:
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path

plt.switch_backend('QtAgg')
fig, ax = plt.subplots()
ax.scatter(df['SSC-A'], df['FSC-A'], s=0.25, alpha=0.25, linewidths=0)
ax.set_xlabel('SSC-A')
ax.set_ylabel('FSC-A')

def onselect(verts):
    path = Path(verts)

ps = PolygonSelector(ax, onselect, useblit=True, lineprops=dict(color='r', linewidth=2))
# open in interactive mode in a new window (we're in jupyter so we have to manually specify the backend):
plt.show()




##────────────────────────────────────────────────────────────────────────────}}}

### {{{             --     simple controls and preprocessing     --


channel_order = sorted(list(FORTESSA_CHANNELS.keys()))

controls = {
    tuple(): loaded_fcs['CNTL.2023-01-22_CasE_ALLuORFs'][channel_order].copy(),
    ('eYFP',): loaded_fcs['eYFP.2023-01-22_CasE_ALLuORFs'][channel_order].copy(),
    ('eBFP',): loaded_fcs['EBFP2.2023-01-22_CasE_ALLuORFs'][channel_order].copy(),
    ('mKate',): loaded_fcs['mKate.2023-01-22_CasE_ALLuORFs'][channel_order].copy(),
    ('eYFP', 'eBFP', 'mKate'): loaded_fcs['ALL.2023-01-22_CasE_ALLuORFs'][channel_order].copy(),
}

bead_fcs = Path(data_dir / 'FCS_FILES/2023-01-22_CasE_ALLuORFs_BEADS_AL01_017.fcs')
beads = load_fcs_to_df(bead_fcs)
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


def transform(x, offset, scale):
    return jnp.log10(x + offset) / scale

def inverse_transform(x, offset, scale):
    return 10 ** (x * scale) - offset


# OFFSET = 1 - jnp.min(data, axis=0)
SCALE = 7
OFFSET = 0

# TODO: try clip vs remove
data = jnp.clip(controls_values, 1, None)
logdata = transform(data, OFFSET, SCALE)



##────────────────────────────────────────────────────────────────────────────}}}

### {{{                        --     optim loop     --


unit_to_channel = {v: k for k, v in channel_to_unit.items()}
# reorder logdata and controls_masks randomly

key = jax.random.PRNGKey(0)
reorder = jax.random.permutation(key, jnp.arange(len(logdata)))
Y = logdata[reorder]
masks = controls_masks[reorder]
weights = vmap(w_function)(Y)

standardize_to = ('MEFL', 'eYFP')
standardize_ids = (channel_order.index(unit_to_channel[standardize_to[0]]), fluo_proteins.index(standardize_to[1]))

# TODO: compute std from bead peak assignment and transform


num_iter = 10
learning_rate = 1e-3

params = {
    'S': jax.random.uniform(key, shape=(len(fluo_proteins), len(channel_to_unit)), minval=0, maxval=1),
    'A': jnp.zeros((len(channel_to_unit),)),
}



def loss(A, S, y_i, m_i, w_i, std_ij):
    yhat = m_i @ S + A
    x = yhat / y_i
    mean = jnp.mean(x)
    linear_error = jnp.average((x - mean) ** 2, weights=w_i)
    j = standardize_ids[1]
    standardization_error = w_i[j] * m_i[j] * (mean - std_ij) ** 2
    return linear_error + standardization_error

@jax.value_and_grad
def lossf(params, y, m, w, std):
    """ 
    with N = batch size:
    y: (N, C): observations
    m: (N, P): masks 
    w: (N, C): weights for each observation + channel
    std: (N, P): standardized values for channel standardize_ids[0] and protein standardize_ids[1]
    """
    S, A = params['S'], params['A']
    l = vmap(loss, in_axes=(None, None, 0, 0, 0, 0))(S, A, y, m, w, std)
    return jnp.mean(l)


optimizer = optax.adam(learning_rate=learning_rate)
opt_state = optimizer.init(params)

@jit
def update(params, opt_state):
    loss, grad = lossf(params)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


all_params = []
for i in tqdm(list(range(0, num_iter + 1))):
    params, opt_state, loss = update(params, opt_state)
    if i % dump_every == 0:
        print(loss)
        all_params.append(params)


##────────────────────────────────────────────────────────────────────────────}}}
