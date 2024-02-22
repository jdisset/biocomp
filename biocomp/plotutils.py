# {{{                          --     imports     --
# ···············································································
import jax
import jax.numpy as jnp
from matplotlib import scale as mscale
from jax.tree_util import Partial as partial
from scipy.spatial import cKDTree
from jax import jit, vmap
import numpy as np
from . import utils as ut
from . import datautils as du
from . import compute as cmp
from . import parameters as pm
from .datautils import DataManager
import matplotlib.pyplot as plt
from jax.scipy.stats import gaussian_kde
import matplotlib.ticker as ticker
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.offline as pyo
import plotly.graph_objects as go
import plotly.graph_objs as go
import plotly.graph_objects as go
import plotly.offline as pyo
import numpy as np
import difflib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import string
from labellines import labelLine, labelLines
from jax.typing import ArrayLike
from typing import Tuple
import os

os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'

##────────────────────────────────────────────────────────────────────────────}}}

# ─────────────────────────────────────────────────────────────────────────────
#                           TOOLS & UTILS
# ───────────────────────────────────── ▼ ─────────────────────────────────────

### {{{                   --     DataRescaler wrapper     --


class DataRescaler:
    def __init__(self, fwd_transform, inv_transform):
        self.fwd_transform = fwd_transform
        self.inv_transform = inv_transform

    def __call__(self, x):
        return self.fwd_transform(x)

    def inv(self, x):
        return self.inv_transform(x)

    @classmethod
    def from_data_manager(cls, dm):
        def fwd(x):
            return dm.rescale([x])[0]

        def inv(x):
            return dm.unscale([x])[0]

        return cls(fwd, inv)


class DataManagerRescaler:
    def __init__(self, dm):
        self.dm = dm

    def fwd(self, x):
        return self.dm.rescale([x])[0]

    def inv(self, x):
        return self.dm.unscale([x])[0]

    def __call__(self, x):
        return self.fwd(x)


IDENTITY_RESCALER = DataRescaler(lambda x: x, lambda x: x)

##────────────────────────────────────────────────────────────────────────────}}}
### {{{                   --     log_spline_log scale     --


def powers_of_ten(xmin, xmax, skip_ticklabel_range=None, resolution=1, **_):
    bounds = np.array([xmin, xmax])
    logbounds = np.sign(bounds) * np.floor(
        np.maximum(np.log10(np.maximum(np.abs(bounds), 0.1)), 0)
    ).astype(int)
    if logbounds[0] == logbounds[1]:
        logbounds[1] += 1

    powers = np.arange(logbounds[0], logbounds[1] + 1)

    if skip_ticklabel_range is not None:
        skip_power_low = np.floor(np.log10(max(skip_ticklabel_range[0], 0.1))).astype(int)
        skip_power_high = np.ceil(np.log10(skip_ticklabel_range[1])).astype(int)
        powers = np.delete(
            powers,
            np.where((np.abs(powers) >= skip_power_low) & (np.abs(powers) <= skip_power_high)),
        )

    base_powers = np.power(10, powers)

    increments = np.arange(1, resolution + 1).reshape(-1, 1)
    values = (base_powers * increments).flatten()

    # Filter values to be within the bounds
    values = values[(values >= xmin) & (values <= xmax)]
    return values


def format_powers(x, *_, n_decimals=1):
    abs_x = abs(x)
    if abs_x < 1000:
        if np.abs(x - int(x)) < 1e-3:
            return f'{int(x)}'  # No decimal point
        else:
            return f'{x:.1f}'  # Up to 1 decimal point
    else:
        E = int(np.log10(abs_x))
        if x == int(x):
            return r'${0:.0f}e{1}$'.format(x // 10**E, E)
        else:
            return r'${0:.{2}f}e{1}$'.format(x / 10**E, E, n_decimals)


class PowerFormatter(ticker.Formatter):
    def __init__(self, values, skip_ticklabel_range=None, **_):
        self.values = values
        self.skip_ticklabel_range = skip_ticklabel_range

    def __call__(self, x, pos):
        v = self.values[pos]
        if (
            self.skip_ticklabel_range is not None
            and np.abs(v) < self.skip_ticklabel_range[1]
            and np.abs(v) > self.skip_ticklabel_range[0]
        ):
            return ''
        return format_powers(v, None)


def get_bio_color(name, default='k'):
    colors = {'ebfp': '#529edb', 'eyfp': '#fbda73', 'mkate': '#f75a5a', 'neongreen': '#33f397'}
    colors['fitc'] = colors['neongreen']
    colors['pe_texas_red'] = colors['mkate']
    colors['pacific_blue'] = colors['ebfp']
    closest = difflib.get_close_matches(name.lower(), colors.keys(), n=1)
    if len(closest) == 0:
        color = default
    else:
        color = colors[closest[0]]
    return color


@jit
def loglog(x):
    return jnp.where(x > 1, jnp.log10(x), jnp.where(x < -1, -jnp.log10(-x), 0))


@jit
def inv_loglog(x):
    return jnp.where(x > 0, 10**x, jnp.where(x < 0, -(10**-x), 0))


##────────────────────────────────────────────────────────────────────────────}}}
### {{{               --     get rescaled network ticks and labels     --
def get_reordered_protein_names(network, input_order=None, protein_aliases=None, **_):
    input_names = network.get_inverted_input_proteins()
    output_names = network.get_output_proteins()

    if input_order is not None:
        assert len(input_order) == len(input_names), f'Wrong number of inputs: {input_order}'
        reordered_input_names = [input_names[i] for i in input_order]
        in_order = input_order
    else:
        reordered_input_names = sorted(input_names)
        in_order = [input_names.index(i) for i in reordered_input_names]

    if len(output_names) != (len(input_names) + 1):
        raise ValueError(
            f"""Wrong number of inputs/outputs:
                         {len(input_names)} inputs: {input_names},
                         {len(output_names)} outputs: {output_names}.
                         Expecting networks to have one more output than inputs."""
        )

    output_name = list(set(output_names) - set(input_names))[0]
    output_pos = output_names.index(output_name)

    if protein_aliases is not None:
        reordered_input_names = [protein_aliases.get(n, n) for n in reordered_input_names]
        output_name = protein_aliases.get(output_name, output_name)

    return list(in_order) + [output_pos], list(reordered_input_names) + [output_name]


def network_ticks_and_labels(network, rescaler, xmin=0, xmax=1, **kw):
    unscaled_ticks = np.logspace(0, 12, 13)
    ticks = np.array(rescaler(unscaled_ticks))
    valid_ticks = (ticks <= xmax) & (ticks >= xmin)
    # valid_ticks = np.ones_like(ticks, dtype=bool)
    ticks = ticks[valid_ticks]
    tlabels = [scformat.format("{:m}", x) for x in unscaled_ticks[valid_ticks]]

    secondary_ticks = []

    rpnames = get_reordered_protein_names(network, **kw)

    return *rpnames, ticks, tlabels, secondary_ticks


def setup_transformed_xaxis(ax, xaxis_lims, rescaler, margins=0.05, **kw):
    xlims_tr = np.asarray(xaxis_lims)
    xlims_inv = rescaler.inv(np.asarray(xlims_tr))
    p10 = powers_of_ten(xmin=xlims_inv[0], xmax=xlims_inv[1])
    xlims_margin = xlims_tr + np.array([-1, 1]) * margins * np.diff(xlims_tr)
    ax.set_xlim(xlims_margin)
    ax.set_xticks(rescaler(p10))  # major ticks
    ax.xaxis.set_major_formatter(PowerFormatter(p10, **kw))
    p10_minor = powers_of_ten(xmin=xlims_inv[0], xmax=xlims_inv[1], resolution=10)
    ax.set_xticks(rescaler(p10_minor), minor=True)
    return xlims_inv


def setup_transformed_yaxis(ax, yaxis_lims, rescaler, margins=0.05, **kw):
    ylims_tr = np.asarray(yaxis_lims)
    ylims_inv = rescaler.inv(np.asarray(ylims_tr))
    p10 = powers_of_ten(xmin=ylims_inv[0], xmax=ylims_inv[1])
    ylims_margin = ylims_tr + np.array([-1, 1]) * margins * np.diff(ylims_tr)
    ax.set_ylim(ylims_margin)
    ax.set_yticks(rescaler(p10))
    ax.yaxis.set_major_formatter(PowerFormatter(p10, **kw))
    p10_minor = powers_of_ten(xmin=ylims_inv[0], xmax=ylims_inv[1], resolution=10)
    ax.set_yticks(rescaler(p10_minor), minor=True)
    return ylims_inv


def get_transformed_ticks_and_labels(axis_lims, rescaler, **kw):
    # will return 2 things:
    # - ticks: a dict with 'major' and 'minor' keys, each containing a list of ticks
    #   ex: ticks={'major': [0, 5, 10, 15, 20], 'minor': [2.5, 7.5, 12.5, 17.5]},
    # - labels: a list of (float, str) tuples, each containing a tick and its label

    lims_tr = np.asarray(axis_lims)
    lims_inv = rescaler.inv(np.asarray(lims_tr))
    p10 = powers_of_ten(xmin=lims_inv[0], xmax=lims_inv[1])
    p10_minor = powers_of_ten(xmin=lims_inv[0], xmax=lims_inv[1], resolution=10)
    ticks = {'major': rescaler(p10), 'minor': rescaler(p10_minor)}
    pf = PowerFormatter(p10, **kw)
    labels = [(rescaler(x), pf(x, i)) for i,x in enumerate(p10)]
    return ticks, labels


def setup_transformed_axis(
    ax, xaxis_lims=None, yaxis_lims=None, rescaler=None, margins=0.05, transform=None, **kw
):
    if xaxis_lims is not None:
        xaxis_lims = setup_transformed_xaxis(ax, xaxis_lims, rescaler, margins=margins, **kw)
    if yaxis_lims is not None:
        yaxis_lims = setup_transformed_yaxis(ax, yaxis_lims, rescaler, margins=margins, **kw)
    return xaxis_lims, yaxis_lims


def setup_symlog_xaxis(ax, xaxis_lims, transform, margins=0.05, **kw):
    xlims_tr = transform(np.asarray(xaxis_lims))
    xp10 = powers_of_ten(*xaxis_lims)
    xlims_margin = xlims_tr + np.array([-1, 1]) * margins * np.diff(xlims_tr)
    ax.set_xlim(xlims_margin)
    ax.set_xticks(transform(xp10))
    ax.xaxis.set_major_formatter(PowerFormatter(xp10, **kw))


def setup_symlog_yaxis(ax, yaxis_lims, transform, margins=0.05, **kw):
    ylims_tr = transform(np.asarray(yaxis_lims))
    yp10 = powers_of_ten(*yaxis_lims)
    ylims_margin = ylims_tr + np.array([-1, 1]) * margins * np.diff(ylims_tr)
    ax.set_ylim(ylims_margin)
    ax.set_yticks(transform(yp10))
    ax.yaxis.set_major_formatter(PowerFormatter(yp10, **kw))


def setup_symlog_axis(
    ax, xaxis_lims=None, yaxis_lims=None, linthresh=200, linscale=0.4, margins=0.05, **kw
):
    tr = partial(ut.log_poly_log, threshold=linthresh, compression=linscale)
    invtr = partial(ut.inverse_log_poly_log, threshold=linthresh, compression=linscale)
    xlims_tr, ylims_tr = None, None

    if xaxis_lims is not None:
        setup_symlog_xaxis(ax, xaxis_lims, tr, margins=margins, **kw)

    if yaxis_lims is not None:
        setup_symlog_yaxis(ax, yaxis_lims, tr, margins=margins, **kw)

    return tr, invtr, xlims_tr, ylims_tr


##────────────────────────────────────────────────────────────────────────────}}}
### {{{              --     knn and spatial partitionning    --


@jax.jit
def weighted_quantile(data, weights, qu):
    ix = jnp.argsort(data)
    data = data[ix]
    weights = weights[ix]
    cdf = (jnp.cumsum(weights) - 0.5 * weights) / jnp.sum(weights)
    return jnp.interp(qu, cdf, data)


class NaiveGridSpacePartitioner:
    # TODO: optimize, use jax (currently not jittable and too slow to use)
    def __init__(self, data: np.ndarray, lower: ArrayLike, upper: ArrayLike, binsize: float):
        """Create a uniform grid space partitioner in N dimensions,
        with lower and upper coordinates and a binsize"""
        self.data = data
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.binsize = binsize
        assert len(lower) == len(upper) == data.shape[1]

        # Determine number of bins along each dimension
        self.num_bins = np.ceil((self.upper - self.lower) / self.binsize).astype(int)
        self.grid = {}  # TODO: use dense array instead

        # Store data points in their corresponding grid cells
        for idx, point in enumerate(data):
            bin_indices = self._get_bin(point)
            if bin_indices not in self.grid:
                self.grid[bin_indices] = []
            self.grid[bin_indices].append(idx)

    def _get_bin(self, point: ArrayLike) -> Tuple[int, ...]:
        """Determine which bin a point belongs to."""
        return tuple(((np.array(point) - self.lower) / self.binsize).astype(int))

    def query(
        self, x: ArrayLike, k: int, distance_upper_bound: float
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Query the partitioner for the k nearest neighbors of x,
        within a maximum distance of distance_upper_bound.
        Returns a pair of array: first one is the distances, second one is the indices.
        Pads with np.inf and -1 respectively, if there are less than k neighbors.
        """

        bin_indices = self._get_bin(x)
        neighbors = []

        # Compute the range of bins to search in each dimension
        search_range = int(np.ceil(distance_upper_bound / self.binsize))

        # Iterate over nearby bins
        for offset in np.ndindex(tuple([2 * search_range + 1] * len(bin_indices))):
            target_bin = tuple((np.array(bin_indices) - search_range + np.array(offset)).tolist())
            if target_bin in self.grid:
                neighbors.extend(self.grid[target_bin])

        # Calculate distances and sort neighbors by distance
        distances = np.linalg.norm(self.data[neighbors] - np.array(x), axis=1)
        sorted_indices = np.argsort(distances)

        nearest_indices = [neighbors[i] for i in sorted_indices[:k]]
        nearest_distances = distances[sorted_indices[:k]]

        # Padding if there are less than k neighbors
        while len(nearest_indices) < k:
            nearest_indices.append(-1)
            nearest_distances = np.append(nearest_distances, np.inf)

        return nearest_distances, np.array(nearest_indices)


def gausspdf(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def get_knn(x, tree, knn=500, min_points=20, radius=0.1, **_):
    distances, indices = tree.query(x, k=knn, distance_upper_bound=radius)
    mask = distances == np.inf
    nb_points = (~mask).sum(axis=1)
    weights = gausspdf(distances, 0, radius / 3)
    indices[mask] = 0
    weights[mask] = 0
    weights[nb_points < min_points, :] = np.nan
    return indices, weights


def get_knn_mean(x, y, tree, **kw):
    indices, weights = get_knn(x, tree, **kw)
    avg = np.average(y[indices], axis=1, weights=weights)
    density = np.nansum(weights, axis=1)
    return avg, density


def get_knn_quantile(x, y, tree, qu, **kw):
    indices, weights = get_knn(x, tree, **kw)
    q = jax.vmap(weighted_quantile, in_axes=(0, 0, None))(y[indices], weights, qu)
    density = np.nansum(weights, axis=1)
    return q, density


def get_knn_smooth(xquery, logY, tree, knn=500, min_points=20, knn_method='mean', **kw):
    if knn_method == 'mean':
        Z, p = get_knn_mean(xquery, logY, knn=knn, min_points=min_points, tree=tree, **kw)
    elif knn_method == 'quantile':
        assert 'qu' in kw
        Z, p = get_knn_quantile(xquery, logY, knn=knn, min_points=min_points, tree=tree, **kw)
    else:
        raise ValueError(f'Unknown method {knn_method}')
    return Z, p


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                    --     misc plot styling tools     --
from matplotlib import colors as mcolors

BIOCOMP_BLUES = [
    '#F9F7F5',
    '#EEECEA',
    '#B0CCD6',
    '#6CAFC3',
    '#2974A4',
    '#3B4B90',
    '#3D1277',
    '#22044B',
]

BIOCOMP_GREENS = [
    '#F9F7F5',
    '#E2EADA',
    '#CBE4BB',
    '#9DDDAA',
    '#4CCDAB',
    '#30A78F',
    '#1F7D73',
    '#0C5558',
]

BIOCOMP_REDS = [
    '#F5F5F5',
    '#F1E6E5',
    '#F3CFBC',
    '#EF957D',
    '#D3494B',
    '#B00031',
    '#840137',
    '#560140',
]


DEFAULT_CMAPS = {
    'blues': mcolors.LinearSegmentedColormap.from_list('cm', BIOCOMP_BLUES, N=256),
    'greens': mcolors.LinearSegmentedColormap.from_list('cm', BIOCOMP_GREENS, N=256),
    'reds': mcolors.LinearSegmentedColormap.from_list('cm', BIOCOMP_REDS, N=256),
}

DEFAULT_CMAP = DEFAULT_CMAPS['blues']


def setup_clean_fig(title):
    fig, ax = plt.subplots(1, 1)
    fig.patch.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.suptitle(title)
    return fig, ax


def default_style(ax):
    fig = ax.get_figure()
    fig.patch.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # long thin ticks
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # font
    ax.tick_params(axis='both', which='both', labelsize=8)
    ax.tick_params(axis='both', which='major', length=5, width=0.4)
    ax.tick_params(axis='both', which='minor', length=2, width=0.2)
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)
    # tick outside
    ax.tick_params(axis='both', which='both', direction='out')

    # spine color
    ax.spines['bottom'].set_color('#777777')
    ax.spines['left'].set_color('#777777')


def mkfig(rows=1, cols=1, size=(4, 4), dpi=300, **kw):
    fig, ax = plt.subplots(rows, cols, figsize=(cols * size[0], rows * size[1]), dpi=dpi, **kw)
    if rows == 1 and cols == 1:
        default_style(ax)
    else:
        for a in ax.flatten():
            default_style(a)
    return fig, ax


def remove_spines(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)


def remove_axis_and_spines(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    remove_spines(ax)


class ShortScientificFormatter(string.Formatter):
    def format_field(self, value, format_spec):
        if format_spec == 'm':
            if value < 1000:
                if value == int(value):
                    return super().format_field(int(value), '')
                else:
                    return super().format_field(value, '.1f')
            else:
                if value == int(value):
                    return super().format_field(value, '.0e').replace('e+0', 'e').replace('e+', 'e')
                else:
                    return super().format_field(value, '.1e').replace('e+0', 'e').replace('e+', 'e')
        else:
            return super().format_field(value, format_spec)


scformat = ShortScientificFormatter()


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                        --     misc utils     --
def make_xy_grid(xmin, xmax, ymin=None, ymax=None, xres=100, yres=None):
    ymin = ymin if ymin is not None else xmin
    ymax = ymax if ymax is not None else xmax
    yres = yres if yres is not None else xres
    xx = np.linspace(xmin, xmax, xres)
    yy = np.linspace(ymin, ymax, yres)
    X, Y = np.meshgrid(xx, yy)
    # we want to return as a big array of shape (res**2, 2)
    return np.vstack([X.ravel(), Y.ravel()]).T


def get_web_font(url, font_name):
    import tempfile
    from pathlib import Path
    import urllib
    from rich import print
    from matplotlib import font_manager

    # Create a temporary directory for the font file
    path = Path(tempfile.mkdtemp())

    # URL and downloaded path of the font
    url_font = url
    path_font = path / f"{font_name}.ttf"

    # Download the font to our temporary directory
    urllib.request.urlretrieve(url_font, path_font)
    # Create a Matplotlib Font object from our `.ttf` file
    font = font_manager.FontEntry(fname=str(path_font), name=font_name)

    # Register this object with Matplotlib's ttf list
    font_manager.fontManager.ttflist.append(font)
    return font


##────────────────────────────────────────────────────────────────────────────}}}

# general workflow:

# -- network_data_plot:
# -- -- smooth/scatter/smooth_lines (x, y, network, DataRescaler, **kw)
# -- -- -- smooth_2d:
# -- -- -- -- get labels and ticks

# ─────────────────────────────────────────────────────────────────────────────
#                            PLOTTING FUNCTIONS
# ───────────────────────────────────── ▼ ─────────────────────────────────────


# ---- base functions
### {{{                          --     heatmap     --
def heatmap(
    ax,
    Z,
    vmin=0,
    vmax=1,
    ticks=[],
    ticklabels=[],
    secondticks=[],
    transform=None,
    text='',
    connector=False,
    connector_orientation='bottom',
    contours=3,
    colorbar=True,
    opacities=None,
    get_cbar_ticks=None,
    cmap=DEFAULT_CMAP,
    **_,
):
    cmap.set_bad(color='#EEEEEE')
    trans_data = ax.transData
    if transform is not None:
        trans_data = trans_data + transform
    if opacities is None:
        opacities = np.ones_like(Z)

    if vmin is None:
        vmin = np.nanmin(Z)
    if vmax is None:
        vmax = np.nanmax(Z)

    im = ax.imshow(
        Z.T,
        origin='lower',
        aspect=1,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=trans_data,
        interpolation='none',
        alpha=opacities.T,
    )
    # add contour
    if contours is not None:
        ax.contour(
            Z.T,
            levels=contours,
            linewidths=0.3,
            transform=trans_data,
        )

    x1, x2, y1, y2 = im.get_extent()
    w = x2 - x1
    h = y1 - y2

    ax.plot(
        [x1, x2, x2, x1, x1],
        [y1, y1, y2, y2, y1],
        "-",
        color='#AAAAAA',
        transform=trans_data,
        linewidth=0.2,
    )

    # ticks:
    if len(ticks) > 0:
        # rescale ticks to image coordinates (they are btwn 0 and 1 to start)
        sc_ticks = ticks * Z.shape[0]
        ax.set_xticks(sc_ticks)
        ax.set_xticklabels(ticklabels)
        ax.set_yticks(sc_ticks)
        ax.set_yticklabels(ticklabels)
        # secondticks:
        if len(secondticks) > 0:
            sc_secondticks = secondticks * Z.shape[0]
            ax.set_xticks(sc_secondticks, minor=True)
            ax.set_yticks(sc_secondticks, minor=True)

    # colorbar
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=6)
        # no cbar spines
        for spine in cbar.ax.spines.values():
            spine.set_visible(False)

        if get_cbar_ticks is not None:
            # get ticks every 0.1 decade
            unscaled_ticks = np.geomspace(du.inv_tr(vmin), du.inv_tr(vmax), 5, endpoint=True)
            ticks = np.array(du.tr(unscaled_ticks))
            ticks = ticks[ticks < vmax]
            ticks = ticks[ticks > vmin]
            ticklabels = [scformat.format("{:m}", du.inv_tr(x)) for x in ticks]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(ticklabels)

    return im


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                          --     heatmap     --
def heatmap_new(
    ax,
    xy_grid,
    output_values,
    rescaler=IDENTITY_RESCALER,
    vlims=(None, None),
    contours=3,
    colorbar=True,
    opacities=None,
    axtransform=None,
    cmap=DEFAULT_CMAP,
    bad_color='#EEEEEE00',
    **kw,
):
    cmap = plt.get_cmap(cmap)
    cmap.set_bad(color=bad_color)
    full_transform = ax.transData
    if axtransform is not None:
        full_transform = full_transform + axtransform

    xres = len(np.unique(xy_grid[:, 0]))
    yres = len(np.unique(xy_grid[:, 1]))

    xlims = np.array([xy_grid[:, 0].min(), xy_grid[:, 0].max()])
    ylims = np.array([xy_grid[:, 1].min(), xy_grid[:, 1].max()])
    vmin, vmax = vlims
    vmin = vmin if vmin is not None else np.nanmin(output_values)
    vmax = vmax if vmax is not None else np.nanmax(output_values)

    Z = output_values.reshape((xres, yres)).T

    opacities = np.ones_like(Z) if opacities is None else opacities.reshape((xres, yres)).T
    opacities = np.where(np.isnan(Z), 0, opacities)

    if np.isnan(Z).all():
        Z = np.zeros_like(Z)

    im = ax.imshow(
        Z.T,
        origin='lower',
        aspect=1,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=full_transform,
        interpolation='none',
        alpha=opacities.T,
        extent=[*xlims, *ylims],
    )

    # no borders

    cnt = None
    if not np.isnan(Z).all():
        if contours is not None:
            cnt = ax.contour(
                Z.T,
                levels=contours,
                linewidths=0.25,
                linestyles='solid',
                extent=[*xlims, *ylims],
                transform=full_transform,
                alpha=0.3,
                colors='k',
            )

        setup_transformed_axis(
            ax,
            xaxis_lims=xlims,
            yaxis_lims=xlims,
            rescaler=rescaler,
            margins=0.0,
            **kw,
        )

        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="7%", pad=0.5)
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=6)
            default_style(cbar.ax)
            cbar.ax.tick_params(axis='both', which='both', direction='out', pad=2, labelsize=8)
            for spine in cbar.ax.spines.values():
                spine.set_linewidth(0.2)
            setup_transformed_axis(
                cbar.ax,
                yaxis_lims=[vmin, vmax],
                rescaler=rescaler,
                margins=0.0,
                **kw,
            )

    return im, cnt, vmin, vmax


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                      --     density plot 1D     --
def density_plot_1d(
    x,
    sample_at,
    ax,
    color='k',
    label=None,
    ticks=None,
    ticks_labels=None,
    bw_method=None,
    x2=None,
    show_quantiles=[0.005, 0.995],
    **kw,
):
    if bw_method is None:
        bw_method = 0.01
    left_kde = gaussian_kde(x.T, bw_method=bw_method)
    left_densities = left_kde(sample_at.T)
    if x2 is not None:
        right_kde = gaussian_kde(x2.T, bw_method=bw_method)
        right_densities = right_kde(sample_at.T)
    else:
        x2 = x
        right_kde = left_kde
        right_densities = left_densities

    left_densities = (left_densities / left_densities.max()) * 0.4
    right_densities = (right_densities / right_densities.max()) * 0.4

    ax.plot(-left_densities, sample_at, color='k', alpha=1, lw=0.5)
    ax.plot(right_densities, sample_at, color='k', alpha=1, lw=0.5)

    if show_quantiles is not None:
        maxleft = sample_at[left_densities.argmax()]
        q1 = np.quantile(x, show_quantiles[0])
        q9 = np.quantile(x, show_quantiles[-1])
        ax.plot([-0.5, 0], [q1, q1], color=color, lw=1)
        ax.plot([-0.5, 0], [q9, q9], color=color, lw=1)
        # ax.plot([-0.5, 0], [maxleft, maxleft], color='k', lw=1)
        ax.fill_betweenx([q1, q9], -0.5, 0, color=color, alpha=0.1, lw=0)
        maxright = sample_at[right_densities.argmax()]
        q1 = np.quantile(x2, show_quantiles[0])
        q9 = np.quantile(x2, show_quantiles[-1])
        ax.plot([0, 0.5], [q1, q1], color=color, lw=1)
        ax.plot([0, 0.5], [q9, q9], color=color, lw=1)
        # ax.plot([0, 0.5], [maxright, maxright], color='k', lw=1)
        ax.fill_betweenx([q1, q9], 0, 0.5, color=color, alpha=0.1, lw=0)

    ax.fill_betweenx(sample_at, -left_densities, 0, color=color, alpha=1, lw=0)
    ax.fill_betweenx(sample_at, 0, right_densities, color=color, alpha=1, lw=0)
    ax.axvline(0, color='k', alpha=0.5, lw=0.5, dashes=(10, 10), dash_capstyle='round')
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 0.5)
    remove_axis_and_spines(ax)
    if label is not None:
        ax.set_xlabel(label, rotation=0, labelpad=20, fontsize=10)
    if ticks is not None:
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
        if ticks_labels is not None:
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticks_labels)
            ax.tick_params(axis='y', which='both', length=0, pad=30)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(8)
                tick.label.set_color('grey')


##────────────────────────────────────────────────────────────────────────────}}}


# ---- smooth plots (gaussian neighborhood based)
### {{{                          --     main smooth method (route to 1D, 2D, 3D)    --


def smooth(x, y, network, rescaler, **kw):
    ninputs = network.get_nb_inputs()
    if ninputs == 1:
        smooth_1d(x, y, network, rescaler, **kw)
    elif ninputs == 2:
        smooth_2d(x, y, network, rescaler, **kw)
    elif ninputs == 3:
        smooth_3d(x, y, network, rescaler, **kw)
    else:
        raise NotImplementedError(f'Cannot plot {ninputs} inputs')


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                            --     1D     --
def smooth_1d(
    x,
    y,
    network,
    rescaler,
    ax,
    res=500,
    xmin=0,
    xmax=None,
    quantiles=None,
    quantiles_alpha=0.2,
    color=BIOCOMP_BLUES[4],
    radius=0.075,
    knn=2000,
    min_points=500,
    **kw,
):
    if xmax is None:
        xmax = x.max()

    tree = cKDTree(x)

    protein_order, protein_names = get_reordered_protein_names(network, **kw)
    input_order, output_pos = protein_order[:-1], protein_order[-1]
    input_names, output_name = protein_names[:-1], protein_names[-1]

    assert len(input_names) == 1

    y = y[:, output_pos]
    x = x[:, input_order]

    unscaled_ticks = np.logspace(0, 12, 13)
    ticks = np.array(rescaler(unscaled_ticks))
    ticks = ticks[ticks < xmax]
    tlabels = [
        scformat.format("{:m}", x) if i > 1 else ''
        for i, x in enumerate(unscaled_ticks[: len(ticks)])
    ]

    xquery_max = min(xmax, x.max() - radius)

    xquery = np.linspace(xmin, xquery_max, res).reshape(-1, 1)
    z, _ = get_knn_smooth(
        xquery, y, tree, knn_method='mean', radius=radius, knn=knn, min_points=min_points, **kw
    )
    if len(z) == 0:
        return
    try:
        ax.plot(xquery, z, color=color)
    except ValueError as e:
        ut.logger.warning(f'Could not plot: {e}.\nxx: {xquery}\nz: {z}')
        pass
    try:
        if quantiles is None:
            quantiles = [0.1, 0.9]
        if quantiles != False:
            assert len(quantiles) == 2
            zq1, _ = get_knn_smooth(
                xquery,
                y,
                tree,
                knn_method='quantile',
                qu=quantiles[0],
                radius=radius,
                knn=knn,
                min_points=min_points,
                **kw,
            )
            zq9, _ = get_knn_smooth(
                xquery,
                y,
                tree,
                knn_method='quantile',
                qu=quantiles[1],
                radius=radius,
                knn=knn,
                min_points=min_points,
                **kw,
            )
            ax.fill_between(xquery[:, 0], zq1, zq9, alpha=quantiles_alpha, color=color)
    except ValueError as e:
        ut.logger.warning(f'Could not fill between: {e}.\nzq1: {zq1}\nzq9: {zq9}')
        pass

    xlims = np.array([xmin, xmax])
    setup_transformed_axis(
        ax,
        xaxis_lims=xlims,
        yaxis_lims=xlims,
        rescaler=rescaler,
        margins=0.0,
        **kw,
    )

    ax.set_xlabel(input_names[0])
    ax.set_ylabel(output_name)


##────────────────────────────────────────────────────────────────────────────}}}
### {{{        --     2D     --


def prepare_smooth_2d(
    x,
    y,
    network,
    input_names,
    input_order,
    output_pos,
    res=200,
    xlims=(0, 1),
    xslice=None,
    density_plot=False,
    density_as_alpha=False,
    density_threshold=10,
    use_y_as_x=False,  # if True, use the output of the independent variables as coordinates
    **kw,
):
    xmin, xmax = xlims

    if use_y_as_x:
        output_names = network.get_output_proteins()
        xind = [output_names.index(i) for i in input_names]
        x = y[:, xind]
    else:
        x = x[:, input_order]

    y = y[:, output_pos]

    xy = make_xy_grid(xmin, xmax, xres=res)
    if x.shape[1] > 2:
        assert xslice.shape == (x.shape[1] - 2,)
        xquery = np.concatenate([xy, np.tile(xslice, (xy.shape[0], 1))], axis=1)
    else:
        xquery = xy
    tree = cKDTree(x)
    output_values, density = get_knn_smooth(xquery, y, tree=tree, **kw)
    assert output_values.shape == (xy.shape[0],)
    assert density.shape == (xy.shape[0],)
    opacities = (
        np.ones_like(density)
        if not density_as_alpha
        else np.minimum(density / density_threshold, 1.0)
    )
    opacities = np.where(np.isnan(output_values), 1, opacities)
    if density_plot:
        output_values = density

    return xy, output_values, opacities


def smooth_2d(
    x,
    y,
    network,
    rescaler,
    ax,
    res=200,
    xlims=(0, 1),
    xslice=None,  # should be called zslice, really...
    title=None,
    text_x=0.5,
    text_y=0.9,
    axtransform=None,
    show_slice_title=True,
    **kw,
):
    protein_order, protein_names = get_reordered_protein_names(network, **kw)
    input_order, output_pos = protein_order[:-1], protein_order[-1]
    input_names, output_name = protein_names[:-1], protein_names[-1]
    # remove input_order from kw
    kw.pop('input_order', None)
    xy, output_values, opacities = prepare_smooth_2d(
        x, y, network, input_names, input_order, output_pos, res, xlims, xslice, **kw
    )

    hm = heatmap_new(
        ax, xy, output_values, rescaler, opacities=opacities, axtransform=axtransform, **kw
    )

    ax.set_xlabel(input_names[0])
    ax.set_ylabel(input_names[1])

    full_transform = ax.transData if axtransform is None else ax.transData + axtransform

    if x.shape[1] > 2 and show_slice_title:
        ax.text(
            text_x,
            text_y,
            f'{input_names[2]} $ \\approx $ {format_powers(rescaler.inv(xslice[0]), n_decimals=0)}',
            fontsize=5,
            transform=full_transform,
            ha='center',
            va='bottom',
        )

    # spines only on bottom and left
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ttle = None
    if title is True:
        ttle = f'{network.name}\n{output_name} smoothed mean'
    elif title is not None:
        ttle = title
    if ttle is not None:
        ax.set_title(ttle)

    return hm


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                            --     3D     --
def smooth_3d(
    x, y, network, rescaler, slices=np.linspace(0, 0.65, 4), axes=None, top_ax=None, **kw
):
    assert axes is not None
    if len(axes) != len(slices):
        raise ValueError(
            f'axes and slices must have the same length, got {len(axes)} and {len(slices)}'
        )

    porder, pnames = get_reordered_protein_names(network, **kw)

    for i, s in enumerate(slices):

        def get_cbar_ticks(vmin, vmax):
            (
                in_order,
                in_names,
                out_pos,
                out_name,
                vticks,
                vtlabels,
                secondticks,
            ) = network_ticks_and_labels(network, rescaler, xmin=vmin, xmax=vmax, **kw)
            return vticks, vtlabels

        kw.pop('ax', None)
        smooth_2d(
            x,
            y,
            network,
            rescaler,
            ax=axes[i],
            xslice=np.array([slices[i]]),
            get_cbar_ticks=get_cbar_ticks,
            **kw,
        )

    if top_ax is not None:
        top_ax.set_xlabel(pnames[-2])
        default_style(top_ax)
        top_ax.spines['left'].set_visible(False)

    # resize all axes  so that they are square and fit in the original ax
    for i, a in enumerate(axes):
        if len(a.get_images()) > 0:
            if i > 0:
                a.set_ylabel('')
            if i < len(axes) - 1:
                a.get_images()[0].colorbar.remove()
            else:
                # write the label on the right of the colorbar
                cbarax = a.get_images()[0].colorbar.ax
                cbarax.yaxis.set_label_position('right')
                cbarax.set_ylabel(pnames[-1], fontsize=8)

        a.set_title('')


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                       --     smooth line plots     --
def smooth_line_plot(
    x,
    y,
    network,
    rescaler,
    ax,
    res=200,  # resolution of the plot (linearspace of input_order[0])
    xmin=0,
    xmax=1,
    vlims=(None, None),
    slice_at=None,  # list of values to slice at (for input_order[1:])
    label=None,
    color=None,
    lw=1,
    tree=None,
    marker=None,
    markevery=20,
    markoffset=0,
    with_quantiles=[0.25, 0.75],
    sample_quantiles_at=None,
    **kw,
):
    protein_order, protein_names = get_reordered_protein_names(network, **kw)
    input_order, output_pos = protein_order[:-1], protein_order[-1]
    input_names, output_name = protein_names[:-1], protein_names[-1]

    y = y[:, output_pos]

    if tree is None:
        x = x[:, input_order]
        tree = cKDTree(x)

    xquery = np.linspace(xmin, xmax, res).reshape(-1, 1)
    slice_at = np.array([]) if slice_at is None else np.array(slice_at)
    slice_at = np.array(slice_at)

    if x.shape[1] > 1:
        assert slice_at.shape == (x.shape[1] - 1,)
        xquery = np.concatenate([xquery, np.tile(slice_at, (xquery.shape[0], 1))], axis=1)

    z, _ = get_knn_smooth(xquery, y, tree=tree, **kw)

    ax.plot(xquery[:, 0], z, label=label, color=color, lw=lw, marker=marker, markevery=markevery)
    if with_quantiles is not None:
        if sample_quantiles_at is None:
            # use markevery to sample quantiles
            sample_quantiles_at = xquery[:, 0][::markevery]

        zqlow, _ = get_knn_quantile(xquery, y, qu=with_quantiles[0], tree=tree)
        zqhigh, _ = get_knn_quantile(xquery, y, qu=with_quantiles[1], tree=tree)

        # ax.fill_between(
        # xquery[:, 0],
        # zqlow,
        # zqhigh,
        # alpha=0.25,
        # color=color,
        # lw=0,
        # )

        ax.errorbar(
            sample_quantiles_at,
            zqlow[::markevery],
            yerr=zqhigh[::markevery] - zqlow[::markevery],
            fmt='none',
            color=color,
            alpha=0.3,
            lw=2,
            capsize=5,
            capthick=2,
            elinewidth=0.5,
            marker=marker,
            markevery=markevery,
        )

    ax.set_xlabel(input_names[0])
    ax.set_ylabel(output_name)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    vmin, vmax = vlims
    vmin = vmin if vmin is not None else np.nanmin(z)
    vmax = vmax if vmax is not None else np.nanmax(z)
    vlims = [vmin, vmax]
    return vlims


def smooth_line_slices(
    x,
    y,
    network,
    rescaler,
    slices,
    input_order,
    axes=None,
    ax=None,
    xmin=0,
    xmax=None,
    color_mode='inner_slice',
    markers=['x', 'o', '^', 'v', '<', '>', '1', '2', '3', '4', '8', 'p', 'P', '*', 'h', 'H'],
    **kwargs,
):
    # slices is a list of list of slice values. (max 2 dimensions)
    assert len(slices) <= 2, 'Can only slice maximum 2 dimensions'
    outerslices = slices[1] if len(slices) > 1 else []
    innerslices = slices[0] if len(slices) > 0 else []

    protein_order, protein_names = get_reordered_protein_names(network, input_order=input_order)
    input_order, output_pos = protein_order[:-1], protein_order[-1]
    input_names, output_name = protein_names[:-1], protein_names[-1]

    x = x[:, input_order]
    tree = cKDTree(x)

    xmin = xmin if xmin is not None else x[:, 0].min()
    xmax = xmax if xmax is not None else x[:, 0].max()

    same_ax = axes is None
    if same_ax:
        assert ax is not None
    else:
        assert len(axes) == len(outerslices), 'Number of axes must match number of outer slices'

    color = 'k'
    # cmap = plt.cm.YlGnBu
    cmap = plt.cm.Spectral
    vlims = np.array([np.inf, -np.inf])
    ivlims = np.tile(vlims, (len(outerslices), 1))
    for i, outsl in enumerate(outerslices):
        iax = ax if same_ax else axes[i]
        if color_mode == 'outer_slice':
            color = cmap(i / len(outerslices))
        for j, insl in enumerate(innerslices):
            if color_mode == 'inner_slice':
                color = cmap(j / len(innerslices))
            vl = smooth_line_plot(
                x,
                y,
                network,
                rescaler,
                ax=iax,
                slice_at=[insl, outsl],
                input_order=input_order,
                xmax=x.max(),
                label=f'{input_names[input_order[1]]} ≈ {format_powers(rescaler.inv(insl), n_decimals=0)}',
                tree=tree,
                color=color,
                marker=markers[i],
                **kwargs,
            )
            ivlims[i] = np.array([min(ivlims[i][0], vl[0]), max(ivlims[i][1], vl[1])])

        vlims = np.array([min(vlims[0], ivlims[i][0]), max(vlims[1], ivlims[i][1])])

        # labelLines(iax.get_lines(), zorder=2.5)

    # for i, outsl in enumerate(outerslices):
    # # iax.text(
    # # 0.75,
    # # ivlims[i][1],
    # # f'{input_names[input_order[2]]} ≈ {format_powers(rescaler.inv(outsl), n_decimals=0)}',
    # # transform=iax.transAxes,
    # # ha='center',
    # # va='center',
    # # fontsize=8,
    # # color=color,
    # # )

    if same_ax:
        # add marker legend: one marker per outer slice
        # for i, outsl in enumerate(outerslices):

        setup_transformed_axis(
            iax,
            xaxis_lims=[xmin, xmax],
            yaxis_lims=vlims,
            rescaler=rescaler,
            margins=0.05,
            **kwargs,
        )


##────────────────────────────────────────────────────────────────────────────}}}


# ---- scatter plots


### {{{         --     main scatter method (route to 1D, 2D, 3D)     --
def scatter(x, y, network, *args, **kw):
    ninputs = network.get_nb_inputs()
    if ninputs == 1:
        return scatter_1d(x, y, network, *args, **kw)
    if ninputs == 2:
        return scatter_2d(x, y, network, *args, **kw)
    if ninputs == 3:
        return scatter_3d_interactive(x, y, network, *args, **kw)
    else:
        raise NotImplementedError(f'Cannot scater plot {ninputs} inputs')


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                            --     1D     --
def scatter_1d(
    x,
    y,
    network,
    rescaler,
    ax,
    xmin=0,
    xmax=1,
    title=None,
    max_n=20000,
    s=10,
    alpha=0.1,
    lw=0,
    key=jax.random.PRNGKey(0),
    use_y_as_x=False,
    **kw,
):
    protein_order, protein_names = get_reordered_protein_names(network, **kw)
    input_order, output_pos = protein_order[:-1], protein_order[-1]
    input_names, output_name = protein_names[:-1], protein_names[-1]
    random_order = np.random.permutation(min(max_n, len(x)))

    if use_y_as_x:
        other_pos = 1 - output_pos
        x = y[random_order, other_pos].squeeze()
    else:
        x = x[random_order].squeeze()

    y = y[random_order, output_pos]

    sc = ax.scatter(x, y, s=s, lw=lw, edgecolor='k', alpha=alpha, color='k')

    ax.set_xlabel(input_names[0])
    ax.set_ylabel(output_name)

    xlims = np.array([xmin, xmax])
    setup_transformed_axis(
        ax,
        xaxis_lims=xlims,
        yaxis_lims=xlims,
        rescaler=rescaler,
        margins=0.0,
        **kw,
    )

    ttle = None

    if title is True:
        ttle = f'{network.name}'
    elif title is not None:
        ttle = title
    if ttle is not None:
        ax.set_title(ttle)


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                            --     2D     --
def scatter_2d(
    x,
    y,
    network,
    rescaler,
    ax,
    xmin=0,
    xmax=1,
    title=None,
    key=jax.random.PRNGKey(0),
    size=10,
    colorbar=True,
    lw=0.1,
    cmap=DEFAULT_CMAP,
    xlims=None,
    ylims=None,
    **kw,
):
    protein_order, protein_names = get_reordered_protein_names(network, **kw)
    input_order, output_pos = protein_order[:-1], protein_order[-1]
    input_names, output_name = protein_names[:-1], protein_names[-1]

    random_order = jax.random.permutation(key, len(x))
    y = y[random_order, output_pos]
    x = x[random_order][:, input_order]

    setup_transformed_axis(
        ax,
        xaxis_lims=xlims,
        yaxis_lims=xlims,
        rescaler=rescaler,
        margins=0.0,
        **kw,
    )

    sc = ax.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap, s=size, lw=lw, edgecolor='k')

    ax.set_xlabel(input_names[0])
    ax.set_ylabel(input_names[1])

    # # colorbar
    # if colorbar:
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="4%", pad=0.05)
    # cbar = plt.colorbar(sc, cax=cax)
    # cbar.ax.tick_params(labelsize=6)
    # # no cbar spines
    # for spine in cbar.ax.spines.values():
    # spine.set_visible(False)
    # # use same ticks if present
    # if len(ticks) > 0:
    # valid = ticks >= xmin
    # diff = len(ticks)
    # ticks = ticks[valid]
    # diff -= len(ticks)
    # cbar.set_ticks(ticks)
    # cbar.set_ticklabels(ticklabels[diff:])

    ttle = None

    if title is True:
        ttle = f'{network.name}\n{output_name} smoothed mean'
    elif title is not None:
        ttle = title
    if ttle is not None:
        ax.set_title(ttle)


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                            --     3D     --
def scatter_3d_interactive(
    x,
    y,
    network,
    rescaler,
    xlims=(0, 1),
    title=None,
    key=jax.random.PRNGKey(0),
    size=10,
    colorbar=True,
    lw=0.01,
    filename=None,
    **kw,
):
    xmin, xmax = xlims

    protein_order, protein_names = get_reordered_protein_names(network, **kw)
    input_order, output_pos = protein_order[:-1], protein_order[-1]
    input_names, output_name = protein_names[:-1], protein_names[-1]

    # xlims_tr = np.asarray(xaxis_lims)
    # xlims_inv = rescaler.inv(np.asarray(xlims_tr))
    # p10 = powers_of_ten(xmin=xlims_inv[0], xmax=xlims_inv[1])
    # xlims_margin = xlims_tr + np.array([-1, 1]) * margins * np.diff(xlims_tr)
    # ax.set_xlim(xlims_margin)
    # ax.set_xticks(rescaler(p10))  # major ticks
    # ax.xaxis.set_major_formatter(PowerFormatter(p10, **kw))
    # p10_minor = powers_of_ten(xmin=xlims_inv[0], xmax=xlims_inv[1], resolution=10)
    # ax.set_xticks(rescaler(p10_minor), minor=True)
    # return xlims_inv

    random_order = jax.random.permutation(key, len(x))
    y = y[random_order, output_pos]
    x = x[random_order][:, input_order]

    fig = go.Figure()

    scatter = go.Scatter3d(
        x=x[:, 0],
        y=x[:, 1],
        z=x[:, 2],
        mode='markers',
        marker=dict(
            size=size, color=y, colorscale='YlGnBu', opacity=1, line=dict(color='black', width=lw)
        ),
    )

    fig.add_trace(scatter)

    fig.update_layout(
        scene=dict(
            xaxis_title=input_names[0],
            yaxis_title=input_names[1],
            zaxis_title=input_names[2],
            xaxis=dict(
                showspikes=False, showbackground=False
            ),  # tickvals=ticks, ticktext=ticklabels),
            yaxis=dict(
                showspikes=False, showbackground=False
            ),  # tickvals=ticks, ticktext=ticklabels),
            zaxis=dict(
                showspikes=False, showbackground=False
            ),  # tickvals=ticks, ticktext=ticklabels),
        ),
        width=1000,
        height=800,
    )

    if colorbar:
        cbar_trace = go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode='markers',
            marker=dict(
                size=0,
                cmin=y.min(),
                cmax=y.max(),
                colorscale='YlGnBu',
                showscale=True,
                colorbar=dict(title=output_name),  # tickvals=ticks, ticktext=ticklabels),
            ),
        )

        fig.add_trace(cbar_trace)

    ttle = None
    if title is True:
        ttle = f'{network.name}\n{output_name} smoothed mean'
    elif title is not None:
        ttle = title
    if ttle is not None:
        fig.update_layout(title=ttle)

    if filename is None:
        return pyo.plot(fig, auto_open=True)
    else:
        return pyo.plot(fig, filename=filename, auto_open=False)


def scatter_3d(
    x,
    y,
    network,
    rescaler,
    fig,
    n_views,
    xmin=0,
    xmax=1,
    title=None,
    key=jax.random.PRNGKey(0),
    size=10,
    colorbar=True,
    lw=0.1,
    **kw,
):
    (
        input_order,
        input_names,
        output_pos,
        output_name,
        ticks,
        ticklabels,
        secondticks,
    ) = network_ticks_and_labels(network, rescaler, xmax=xmax, **kw)

    cmap = plt.get_cmap('YlGnBu')
    random_order = jax.random.permutation(key, len(x))
    y = y[random_order, output_pos]
    x = x[random_order][:, input_order]

    azim_values = np.linspace(0, 270, n_views)

    for i, azim in enumerate(azim_values):
        ax = fig.add_subplot(1, n_views, i + 1, projection='3d')
        sc = ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y, cmap=cmap, s=size, lw=lw, edgecolor='k')
        ax.set_xlabel(input_names[0])
        ax.set_ylabel(input_names[1])
        ax.set_zlabel(input_names[2])

        if len(ticks) > 0:
            sc_ticks = ticks
            ax.set_xticks(sc_ticks)
            ax.set_xticklabels(ticklabels)
            ax.set_yticks(sc_ticks)
            ax.set_yticklabels(ticklabels)
            ax.set_zticks(sc_ticks)
            ax.set_zticklabels(ticklabels)

        # if colorbar and i == n_views - 1:  # Only show colorbar on the last plot
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("top", size="4%", pad=0.05)
        # cbar = plt.colorbar(sc, cax=cax)
        # cbar.ax.tick_params(labelsize=6)
        # for spine in cbar.ax.spines.values():
        # spine.set_visible(False)

        # if len(ticks) > 0:
        # valid = ticks >= xmin
        # diff = len(ticks)
        # ticks = ticks[valid]
        # diff -= len(ticks)
        # cbar.set_ticks(ticks)
        # cbar.set_ticklabels(ticklabels[diff:])

        ttle = None

        if title is True:
            ttle = f'{network.name}\n{output_name} smoothed mean'
        elif title is not None:
            ttle = title
        if ttle is not None:
            ax.set_title(ttle)

        # Rotate the axes
        ax.view_init(elev=10, azim=azim)


##────────────────────────────────────────────────────────────────────────────}}}


### {{{                       --     density histogram     --


def histogram(x, y, network, rescaler, ax, **kw):
    ninputs = network.get_nb_inputs()
    if ninputs == 1:
        histogram_plot(x, y, network, rescaler, ax, **kw)
    else:
        raise NotImplementedError(f'Cannot plot {ninputs} inputs')


def histogram_plot(
    X,
    Y,
    network,
    rescaler,
    ax,
    nbins=256,
    xlims=(0, 1),
    ylims=(0, 1),
    vlims=(0.001, None),
    cmap=DEFAULT_CMAPS['blues'],
    noise_smooth=0,
    log_density=True,
    **kw,
):
    assert X.shape[1] == 1

    if isinstance(nbins, int):
        nbins = [nbins, nbins]

    protein_order, protein_names = get_reordered_protein_names(network, **kw)
    input_order, output_pos = protein_order[:-1], protein_order[-1]
    input_names, output_name = protein_names[:-1], protein_names[-1]

    Y = Y[:, output_pos]
    X = X[:, 0]

    xres = np.abs(np.subtract(*xlims)) / nbins[0]
    yres = np.abs(np.subtract(*ylims)) / nbins[1]

    X = X + np.random.normal(size=X.shape) * noise_smooth * xres
    Y = Y + np.random.normal(size=Y.shape) * noise_smooth * yres

    h, xedges, yedges = np.histogram2d(
        X,
        Y,
        bins=nbins,
        density=False,
        range=[xlims, ylims],
    )

    if log_density:
        h = np.log10(h + 1)

    xlims_true_scale, ylims_true_scale = setup_transformed_axis(
        ax,
        xaxis_lims=xlims,
        yaxis_lims=ylims,
        rescaler=rescaler,
        margins=0.0,
        **kw,
    )

    h = h.T  # matplotlib wants it transposed
    ax.imshow(
        h,
        extent=[*xlims, *ylims],
        origin='lower',
        aspect='auto',
        cmap=cmap,
        vmin=vlims[0],
        vmax=vlims[1],
    )

    ax.set_xlabel(input_names[0])
    ax.set_ylabel(output_name)

    # show grid, including minor grid
    ax.grid(color='k', alpha=0.25, linestyle='-', linewidth=0.2, which='major')
    ax.grid(color='k', alpha=0.1, linestyle='-', linewidth=0.1, which='minor')


##────────────────────────────────────────────────────────────────────────────}}}


# ---- specialized plots
### {{{                --     summary model plot functions     --
def network_plot(
    dman: DataManager,
    network_id: int,
    *args,
    kde=None,
    density_quantile_threshold=0.05,
    use_xy=None,
    method='smooth',
    **kw,
):
    network = dman.get_networks()[network_id]
    if use_xy is None:
        x, y = dman.get_X()[network_id], dman.get_Y()[network_id]
    else:
        x, y = use_xy

    if kde is not False:
        if kde is None:
            kde = dman.get_kdes()[network_id]

    rescaler = DataRescaler.from_data_manager(dman)

    return direct_network_plot(
        network,
        x,
        y,
        rescaler,
        *args,
        kde=kde,
        density_quantile_threshold=density_quantile_threshold,
        method=method,
        **kw,
    )


def direct_network_plot(
    network,
    x,
    y,
    rescaler,
    *args,
    kde=None,
    density_quantile_threshold=0.05,
    method='smooth',
    **kw,
):
    if kde is not False:
        rng = jax.random.PRNGKey(0)
        subsample = du.optimal_density_subsample(
            x, kde, rng, quantile_threshold=density_quantile_threshold
        )
        x, y = x[subsample], y[subsample]

    if method == 'smooth':
        return smooth(x, y, network, rescaler, *args, **kw)
    elif method == 'scatter':
        return scatter(x, y, network, rescaler, *args, **kw)
    elif method == 'histogram':
        return histogram(x, y, network, rescaler, *args, **kw)
    elif method == 'smooth_line_slices':
        return smooth_line_slices(x, y, network, rescaler, *args, **kw)


def eval_network_plot(
    params,
    dman,
    id,
    ax,
    npoints_eval=20000,
    quantile_range=[0.2, 0.8],
    key=jax.random.PRNGKey(0),
    xrange_eval=None,
    **kw,
):
    k_i, k_q = jax.random.split(key)
    if xrange_eval is None:
        xrange_eval = np.array([[0, 0], [1, 1]])

    network = dman.get_networks()[id]
    jm = jit(dman.get_individual_compute_stack(id).apply)

    x = jax.random.uniform(
        k_i, (npoints_eval, network.get_nb_inputs()), minval=xrange_eval[0], maxval=xrange_eval[1]
    )
    quantiles = jax.random.uniform(
        k_q, (npoints_eval, network.n_outputs), minval=quantile_range[0], maxval=quantile_range[1]
    )
    keys = jax.random.split(key, npoints_eval)
    y = vmap(jm, in_axes=(None, 0, 0, 0))(params, x, quantiles, keys)

    xmin, xmax = np.min(x, axis=0)[0], np.max(x, axis=0)[0]

    smooth(x, y, network, dman.rescale, ax, xmin=xmin, xmax=xmax, **kw)


def get_stack(dman, net_id, params):
    stack, pf = dman.get_individual_compute_stack(net_id)
    p = pf(params)
    return stack, p


def eval_network_on_grid(
    params,
    network,
    stack,
    ax,
    rescale=du.tr,
    key=jax.random.PRNGKey(0),
    xrange_eval=(0, 1),
    n_repeats=10,
    quantile_range=(0.2, 0.8),
    res=100,
    **kw,
):
    jm = jit(stack.apply)

    (
        input_order,
        input_names,
        output_pos,
        output_name,
        ticks,
        tlabels,
        secondticks,
    ) = network_ticks_and_labels(network, rescale, xmax=xrange_eval[1], **kw)

    k_i, k_q = jax.random.split(key)
    if xrange_eval is None:
        xrange_eval = np.array([0, 1])

    xx = np.linspace(xrange_eval[0], xrange_eval[1], res)
    x = np.array(np.meshgrid(xx, xx)).T.reshape(-1, 2)

    def compute(k):
        quantiles = jax.random.uniform(
            k,
            (len(x), network.get_nb_outputs()),
            minval=quantile_range[0],
            maxval=quantile_range[1],
        )
        keys = jax.random.split(k, len(x))
        y, _ = vmap(jm, in_axes=(None, 0, 0, 0))(params, x, quantiles, keys)
        return y

    keys = jax.random.split(key, n_repeats)
    all_y = vmap(compute)(keys)
    y_mean = np.mean(all_y, axis=0)

    z = y_mean[:, output_pos]
    z = z.reshape(res, res)

    heatmap(ax, z, ticks=ticks, ticklabels=tlabels, **kw)
    ax.set_xlabel(input_names[0])
    ax.set_ylabel(input_names[1])
    remove_spines(ax)


def eval_model_grid(
    params,
    dman,
    id,
    ax,
    **kw,
):
    network = dman.get_networks()[id]
    stack, p = get_stack(dman, id, params)
    return eval_network_on_grid(params, network, stack, ax, rescale=dman.rescale, **kw)


def model_at_x(params, dman: DataManager, id, key=jax.random.PRNGKey(0), quantile=None, **_):
    stack, p = get_stack(dman, id, params)

    x, y = dman.get_X()[id], dman.get_Y()[id]
    keys = jax.random.split(key, x.shape[0])

    if quantile is not None:
        Q = jnp.ones(y.shape) * quantile
    else:
        Q = jax.random.uniform(key, y.shape)

    yhat, _ = jit(vmap(stack.apply, in_axes=(None, 0, 0, 0)))(p, x, Q, keys)

    return x, y, yhat


def plot_model_at_x(params, dman, id, ax, **kw):
    x, _, yhat = model_at_x(params, dman, id, **kw)
    net = dman.get_networks()[id]
    smooth(x, yhat, net, dman.rescale, ax, **kw)


def plot_model_diff(params, dman, id, ax, **kw):
    x, y, yhat = model_at_x(params, dman, id, **kw)
    net = dman.get_networks()[id]
    err = np.abs(y - yhat)
    smooth(x, err, net, dman.rescale, ax, **kw)


def report(params, dman, id, suptitle='', use_x_y_yhat=None, **kw):
    if use_x_y_yhat is not None:
        x, y, yhat = use_x_y_yhat
        assert len(x) == len(y), 'x and y must have the same length'
        assert y.shape == yhat.shape, 'y and yhat must have the same shape'
        ndim = x.shape[1]
        if ndim <= 2:
            fig, ax = mkfig(1, 2, size=(4, 4))
            network_plot(dman, id, ax[0], use_xy=(x, y), kde=False, **kw)
            network_plot(dman, id, ax[1], use_xy=(x, yhat), kde=False, **kw)
            ax[0].set_title(f'Original data (mean)')
            ax[1].set_title(f'Predicted (mean)')
        elif ndim == 3:
            fig, axes = mkfig(2, 4, size=(4, 4))
            contours = np.linspace(0, 0.8, 5)
            top_row_axes = axes[0, :]
            bottom_row_axes = axes[1, :]
            slices = (np.linspace(0.1, 0.8, 4),)
            network_plot(
                dman,
                id,
                ax=None,
                axes=top_row_axes,
                contours=contours,
                slices=np.linspace(0.1, 0.8, 4),
                use_xy=(x, y),
                **kw,
            )
            network_plot(
                dman,
                id,
                ax=None,
                axes=bottom_row_axes,
                contours=contours,
                slices=np.linspace(0.1, 0.8, 4),
                use_xy=(x, yhat),
                **kw,
            )
            for ax in axes.flatten():
                ax.set_title('')
            axes[0, 0].set_title(f'Original data (mean)')
            axes[1, 0].set_title(f'Predicted (mean)')
        else:
            raise ValueError(f'ndim={ndim} not supported')
    else:
        fig, ax = mkfig(1, 2, size=(4, 4))
        network_plot(dman, id, ax[0], **kw)
        plot_model_at_x(params, dman, id, ax[1], **kw)
        ax[0].set_title(f'Original data (mean)')
        ax[1].set_title(f'Predicted (mean)')

    network = dman.get_networks()[id]
    fig.suptitle(f'{suptitle} {network.name}')
    fig.tight_layout()
    return fig


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                  --     Fluo distribution plots     --


def fluo_scatter(
    rawx,
    pnames,
    xmin=0,
    xmax=None,
    title=None,
    types=None,
    fname=None,
    logscale=True,
    alpha=0.1,
    maxn=50000,
    s=2,
    **_,
):
    fig, axes = plt.subplots(1, len(pnames), figsize=(1.25 * len(pnames), 10), sharey=True)

    if len(pnames) == 1:
        axes = [axes]

    if types is None:
        types = [''] * len(pnames)

    if xmin is None:
        xmin = rawx.min()
    if xmax is None:
        xmax = rawx.max()

    X = rawx.copy()
    if len(X) > maxn:
        X = X[np.random.choice(len(X), maxn, replace=False)]

    tr = lambda x: x
    itr = tr
    for xid, ax in enumerate(axes):
        color = get_bio_color(pnames[xid])
        xcoords = np.random.normal(0, 0.1, (X.shape[0],))
        if logscale:
            tr, itr, _, ytr = setup_symlog_axis(ax, None, yaxis_lims=[xmin, xmax])
        else:
            ax.set_ylim(xmin, xmax)
        ax.scatter(xcoords, tr(X[:, xid]), color=color, alpha=alpha, s=s, zorder=10, lw=0)

        remove_spines(ax)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xlabel(f'{pnames[xid]} {types[xid]}', rotation=0, labelpad=20, fontsize=10)
        ax.set_xticks([])

    if title is not None:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if fname is not None:
        fig.savefig(fname)

    return fig, axes


def fluo_densities(
    rawx, pnames, xmin=None, xmax=None, res=1000, title=None, types=None, logscale=False, **kw
):
    fig, axes = plt.subplots(1, len(pnames), figsize=(1.5 * len(pnames), 10))

    if logscale:
        X = loglog(rawx)
    else:
        X = rawx
    xmin = xmin if xmin is not None else np.floor(X.min())
    xmax = xmax if xmax is not None else np.ceil(X.max())

    ticks = np.arange(xmin, xmax + 1, 1)
    sample_at = np.linspace(xmin, xmax, res)

    if logscale:
        ylabels = [scformat.format("{:m}", x) for x in inv_loglog(ticks)]
    else:
        ylabels = [scformat.format("{:m}", x) for x in ticks]

    if types is None:
        types = [''] * len(pnames)
    for xid, ax in enumerate(axes):
        color = get_bio_color(pnames[xid], default='#AAAAAA')
        tlabels = ylabels if xid == 0 else None
        density_plot_1d(
            X[:, xid],
            sample_at,
            ax,
            color=color,
            label=f'{pnames[xid]} {types[xid]}',
            ticks=ticks,
            ticks_labels=tlabels,
            **kw,
        )
        ax.set_ylim(xmin, xmax)
    if title is not None:
        fig.suptitle(
            title,
            fontsize=10,
            y=0.95,
            x=0.45,
        )
    fig.tight_layout()
    return fig, axes


def model_fluo_distributions(dman, model_id, method='scatter', **kwargs):
    model = dman.get_models()[model_id]
    rawx = dman.get_raw_X()[model_id]
    rawy = dman.get_raw_Y()[model_id]
    input_names = model.get_inverted_input_proteins()
    reordered_input = sorted(input_names)
    output_names = model.get_output_proteins()
    output = list(set(output_names) - set(input_names))
    output_pos = output_names.index(output[0])
    if reordered_input != input_names:
        rawx = rawx[:, [input_names.index(i) for i in reordered_input]]
    rawx = np.hstack([rawx, rawy[:, output_pos][:, None]])
    pnames = reordered_input + output
    types = ['[in]'] * len(reordered_input) + ['[out]']
    if method == 'scatter':
        fluo_scatter(rawx, pnames, types=types, **kwargs)
    elif method == 'kde':
        fluo_densities(rawx, pnames, types=types, **kwargs)


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                   --     node functions plots     --
def plot_node(
    node_name,
    shared_parameters,
    compute_config,
    ax,
    median_evals_resolution=200,
    n_random_evals=10000,
    xlims=(0, 1),
    color='k',
    quantized_param_id=0,
):
    tl = compute_config.get_impl(node_name)

    L = tl(input_shapes=[(1,)], n_outputs=1, stack=None, layer_id=0)

    class FakeNode(cmp.VirtualNode):
        def get_compute_node(self, _):
            return None

        def get_inverse_node(self, _):
            return None

        def get_layer_and_local_id(self, _):
            return 0, 0

    key = jax.random.PRNGKey(0)

    p = pm.ParameterTree()
    L.prepare(p, [FakeNode()], key)
    p.tag('local', 'local')
    local, _ = p.filter_by_tag('local')

    qname = None
    qnames = []
    if node_name in ('translation', 'transcription', 'inv_transcription', 'inv_translation'):
        qmaskleaf = None
        for l, v in local.data.iter_leaves():
            if str(l).endswith('quantization_mask'):
                qmaskleaf = l
                break
        qnames = compute_config.config['functions'][node_name]['parameters']['quantization_names']
        base_mask = np.zeros((len(qnames),), dtype=np.bool).reshape(1, 1, -1)
        base_mask[:, :, quantized_param_id] = True
        local[qmaskleaf] = base_mask
        qname = qnames[quantized_param_id]

    pmerged = pm.ParameterTree.merge(shared_parameters, local)

    @jax.jit
    def vapply(xvals, qs, params):
        f = lambda x, q: L.apply(x, quantiles=q, node_id=0, params=params, key=key)
        return jax.vmap(f)(xvals, qs)

    x = np.linspace(*xlims, median_evals_resolution).reshape(-1, 1)
    medianq = np.ones_like(x) * 0.5

    ymedian = vapply(x, medianq, pmerged).flatten()

    n_random_evals = 20000
    randomx = np.random.uniform(0, 1, n_random_evals).reshape(-1, 1)
    randomq = np.random.uniform(0, 1, n_random_evals).reshape(-1, 1)
    yrandom = vapply(randomx, randomq, pmerged).flatten()

    # from qid

    ax.scatter(randomx, yrandom, s=2, c=color, alpha=0.05, linewidth=0)
    ax.plot(x, ymedian, label=qname if qname is not None else '', c=color, ls='--', lw=2)


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                    --     High level helpers     --

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
