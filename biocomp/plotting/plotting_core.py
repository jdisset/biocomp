# {{{                          --     imports     --
# ···············································································
import jax
import jax.numpy as jnp
import copy
from matplotlib import scale as mscale
from functools import partial
from scipy.spatial import cKDTree
from jax import jit, vmap
import numpy as np
from biocomp import utils as ut
from biocomp import datautils as du
from biocomp import compute as cmp
from biocomp.datautils import DataManager
from biocomp.network import Network
import matplotlib.pyplot as plt
from jax.scipy.stats import gaussian_kde
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import difflib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import string
from labellines import labelLine, labelLines
from jax.typing import ArrayLike
from typing import Tuple
import os
from typing import Union, Sequence, List, Tuple, Dict, Any, Optional, Callable
from matplotlib.ticker import ScalarFormatter, NullFormatter, MaxNLocator
from matplotlib import colors as mcolors
from pkg_resources import resource_filename

##────────────────────────────────────────────────────────────────────────────}}}


### {{{                   --     default configuration     --

NumLike = Union[np.ndarray, jnp.ndarray, float, int]
NdArray = Union[np.ndarray, jnp.ndarray]

os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
logger = ut.setup_logger('biocomp.plotutils')
configurable = ut.configurable_decorator('biocomp.plotutils')

os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'

BASE_COLOR_CONFIG = ut.load_config(
    resource_filename('biocomp', 'biocomp_default_config/colors.yaml')
)
cmap_definitions = BASE_COLOR_CONFIG.color_maps or {}
CUSTOM_CMAPS = {
    k: mcolors.LinearSegmentedColormap.from_list(k, v, N=256) for k, v in cmap_definitions.items()
}

# register custom colormaps
for k, v in CUSTOM_CMAPS.items():
    # check if it's already registered
    if k in plt.colormaps():
        plt.colormaps.unregister(k)
    plt.colormaps.register(v, name=k)

DEFAULT_CMAP_NAME = BASE_COLOR_CONFIG.default_color_map or 'viridis'

##────────────────────────────────────────────────────────────────────────────}}}

# ╭─────────────────────────────────────────────╮
# │                TOOLS & UTILS                │
# ╰───────────────────── ⟱ ─────────────────────╯
### {{{                   --     DataRescaler wrapper     --


class DataRescaler:
    def __init__(self, fwd_transform, inv_transform):
        self.fwd_transform = fwd_transform
        self.inv_transform = inv_transform

    def add_kwargs(self, **kw):
        assert callable(self.fwd_transform) and callable(self.inv_transform)
        if kw:
            self.fwd_transform = partial(self.fwd_transform, **kw)
            self.inv_transform = partial(self.inv_transform, **kw)

    def __call__(self, x: NumLike) -> NdArray:
        return self.fwd_transform(x)

    def inv(self, x: NumLike) -> NdArray:
        return self.inv_transform(x)

    @classmethod
    def from_data_manager(cls, dm: DataManager):
        assert isinstance(dm, DataManager)

        def fwd(x):
            return dm.rescale([x])[0]

        def inv(x):
            return dm.unscale([x])[0]

        return cls(fwd, inv)


BIOCOMP_PLOTTING_DEFAULT_RESCALERS = {
    None: DataRescaler(lambda x: x, lambda x: x),
    'identity': DataRescaler(lambda x: x, lambda x: x),
    'log': DataRescaler(lambda x: np.log(x), lambda x: np.exp(x)),
    'log10': DataRescaler(lambda x: np.log10(x), lambda x: 10**x),
}


def get_rescaler(rescaler, **kw):
    if isinstance(rescaler, DataRescaler):
        return rescaler
    if isinstance(rescaler, str):
        if rescaler in BIOCOMP_PLOTTING_DEFAULT_RESCALERS:
            r = copy.deepcopy(BIOCOMP_PLOTTING_DEFAULT_RESCALERS[rescaler])
            r.add_kwargs(**kw)
            return r
        else:
            raise ValueError(f'Unknown rescaler {rescaler}')
    if isinstance(rescaler, (tuple, list)):
        assert len(rescaler) == 2, 'Rescaler must be a tuple of (fwd, inv) functions'
        assert callable(rescaler[0]) and callable(
            rescaler[1]
        ), 'Rescaler must be a tuple of (fwd, inv) functions'
        return DataRescaler(partial(rescaler[0], **kw), partial(rescaler[1], **kw))


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
    x = float(x)
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


##────────────────────────────────────────────────────────────────────────────}}}
### {{{               --     get rescaled network ticks and labels     --
def get_reordered_protein_names(network, input_order=None, protein_aliases=None, **_):
    # TODO: add support for input_order as a list of protein names
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


TickDict = Dict[str, NdArray]
LabelList = List[Tuple[NdArray, str]]


def get_transformed_ticks_and_labels(
    axis_lims: Sequence[float], rescaler: DataRescaler, **kw
) -> Tuple[TickDict, LabelList]:

    # will return 2 things:
    # - ticks: a dict with 'major' and 'minor' keys, each containing a list of ticks
    #   ex: ticks={'major': [0, 5, 10, 15, 20], 'minor': [2.5, 7.5, 12.5, 17.5]},
    # - labels: a list of (float, str) tuples, each containing a tick and its label
    lims_tr = np.asarray(axis_lims)
    lims_inv = rescaler.inv(np.asarray(lims_tr))
    assert isinstance(lims_inv, np.ndarray)
    assert lims_inv.shape == (2,)
    p10 = powers_of_ten(xmin=lims_inv[0], xmax=lims_inv[1])
    p10_minor = powers_of_ten(xmin=lims_inv[0], xmax=lims_inv[1], resolution=10)
    ticks = {'major': rescaler(p10), 'minor': rescaler(p10_minor)}
    pf = PowerFormatter(p10, **kw)
    labels = [(rescaler(x), pf(x, i)) for i, x in enumerate(p10)]
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


def get_knn(x: NdArray, tree: cKDTree, k: int = 500, min_points: int = 20, radius: float = 0.1):
    """Get the k-nearest neighbors of x in the tree,
    and return their indices together with their weights (from a gaussian kernel)."""
    SIGMA_FROM_RADIUS = 1 / 3
    distances, indices = tree.query(x, k=k, distance_upper_bound=radius)
    empty_neighbor_mask = distances == np.inf
    nb_points = (~empty_neighbor_mask).sum(axis=1)
    weights = gausspdf(distances, 0, radius * SIGMA_FROM_RADIUS)
    indices[empty_neighbor_mask] = 0
    weights[empty_neighbor_mask] = 0
    weights[nb_points < min_points, :] = np.nan
    return indices, weights


def get_knn_mean(x, y, tree, **kw):
    """Get the k-nearest neighbors of x in the tree,
    and return their weighted average value together with their density."""
    indices, weights = get_knn(x, tree, **kw)
    assert indices.shape == weights.shape
    normed_w = weights / weights.sum(axis=1)[:, None]
    avg = (y[indices] * normed_w[:, :, None]).sum(axis=1)
    density = np.nansum(weights, axis=1)
    return avg, density


def get_knn_quantile(x, y, tree, qu, **kw):
    indices, weights = get_knn(x, tree, **kw)
    q = jax.vmap(weighted_quantile, in_axes=(0, 0, None))(y[indices], weights, qu)
    density = np.nansum(weights, axis=1)
    return q, density


@configurable
def knn_avg(xquery, logY, tree, k=500, min_points=20, avg_method='mean', **kw):
    if avg_method == 'mean':
        Z, p = get_knn_mean(xquery, logY, k=k, min_points=min_points, tree=tree, **kw)
    elif avg_method == 'quantile':
        assert 'qu' in kw
        Z, p = get_knn_quantile(xquery, logY, k=k, min_points=min_points, tree=tree, **kw)
    else:
        raise ValueError(f'Unknown method {avg_method}')
    return Z, p


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                    --     misc plot styling tools     --


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


@configurable
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


def to_display_units(x, ax):
    """Convert x from data units to display units"""
    ppd = 72.0 / ax.figure.dpi
    trans = ax.transData.transform
    return ((trans((1, x)) - trans((0, 0))) * ppd)[1]


def to_data_units(y_display, ax):
    """Convert y from display units to data units"""
    ppd = 72.0 / ax.figure.dpi
    trans_inv = ax.transData.inverted().transform
    origin = trans_inv((0, 0))
    point_in_data_units = trans_inv((0, y_display / ppd))
    return point_in_data_units[1] - origin[1]


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                       --     network utils     --

from dataclasses import dataclass


@dataclass
class PlotData:
    x: NdArray
    y: NdArray
    input_names: List[str]
    output_name: str
    rescaler: Callable
    metadata: Optional[Dict[str, Any]] = None


def extract_plot_data_from_network(
    network: Network,
    X: NdArray,
    Y: NdArray,
    rescaler: DataRescaler,
    input_order: Optional[Sequence[int]] = None,
    protein_aliases: Optional[Dict[str, str]] = None,
    use_y_as_x: bool = False,
) -> PlotData:

    if input_order is None:
        input_order = np.arange(network.get_nb_inputs())
    if protein_aliases is None:
        protein_aliases = {}

    protein_order, protein_names = get_reordered_protein_names(
        network, input_order, protein_aliases
    )

    input_order, output_pos = protein_order[:-1], protein_order[-1]
    input_names, output_name = protein_names[:-1], protein_names[-1]

    if use_y_as_x:
        output_names = network.get_output_proteins()
        xind = [output_names.index(i) for i in input_names]
        x = Y[:, xind]
    else:
        x = X[:, input_order]

    y = Y[:, output_pos]
    y = Y.reshape(-1, 1)

    assert x.shape[1] == len(input_order)
    assert y.shape[0] == X.shape[0]

    return PlotData(
        x=x,
        y=y,
        input_names=input_names,
        output_name=output_name,
        rescaler=rescaler,
        metadata=network.metadata,
    )


##────────────────────────────────────────────────────────────────────────────}}}


# ╭─────────────────────────────────────────────╮
# │             PLOTTING PRIMITIVES             │
# ╰───────────────────── ⟱ ─────────────────────╯
### {{{                          --     heatmap     --
@configurable
def heatmap(
    ax,
    xy_grid,
    output_values,
    vlims=(None, None),
    contours=3,
    opacities=None,
    axtransform=None,
    cmap=DEFAULT_CMAP_NAME,
    bad_color='#EEEEEE00',
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

    cntrs = None
    if contours is not None:
        cntrs = ax.contour(
            Z.T,
            levels=contours,
            linewidths=0.25,
            linestyles='solid',
            extent=[*xlims, *ylims],
            transform=full_transform,
            alpha=0.3,
            colors='k',
        )

    return im, cntrs


##────────────────────────────────────────────────────────────────────────────}}}
