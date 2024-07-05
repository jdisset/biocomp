# {{{                          --     imports     --
# ···············································································
import jax
from dataclasses import dataclass
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
from biocomp.datautils import DataManager, DataRescaler
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
from typing import Union, Sequence, List, Tuple, Dict, Any, Optional, Callable, TypeAlias
from matplotlib.ticker import ScalarFormatter, NullFormatter, MaxNLocator
from matplotlib import colors as mcolors

from dataclasses import dataclass, field, asdict

from copy import deepcopy

import dracon.loader as dr
##────────────────────────────────────────────────────────────────────────────}}}

logger = ut.setup_logger('biocomp.plotting')
configurable = ut.configurable_decorator('biocomp.plotting')

# ╭─────────────────────────────────────────────╮
# │                TOOLS & UTILS                │
# ╰───────────────────── ⟱ ─────────────────────╯

NdArray: TypeAlias = Union[np.ndarray, jnp.ndarray]
NumLike: TypeAlias = Union[np.ndarray, jnp.ndarray, float, int]

## {{{                   --     default configuration     --

from matplotlib import colors as mcolors

os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
logger = ut.setup_logger('biocomp.plotting')
configurable = ut.configurable_decorator('biocomp.plotting')


BIOCOMP_COLORS = dr.load('pkg:biocomp:config/colors.yaml')
cmap_definitions = BIOCOMP_COLORS['color_maps'] or {}

CUSTOM_CMAPS = {
    k: mcolors.LinearSegmentedColormap.from_list(k, v, N=256)
    for k, v in cmap_definitions.items()
}

# register custom colormaps
for k, v in CUSTOM_CMAPS.items():
    # check if it's already registered
    if k in plt.colormaps():
        plt.colormaps.unregister(k)
    plt.colormaps.register(v, name=k)

DEFAULT_CMAP_NAME = BIOCOMP_COLORS['default_color_map'] or 'viridis'



##────────────────────────────────────────────────────────────────────────────}}}
### {{{                   --     log_spline_log scale     --


def powers_of_ten(xmin, xmax, skip_ticklabel_range=None, resolution=1, **_):
    bounds = np.array([xmin, xmax])
    logbounds = np.sign(bounds) * np.floor(
        np.maximum(np.log10(np.maximum(np.abs(bounds), 0.1)), 0)
    ).astype(int)
    if logbounds[0] == logbounds[1]:
        logbounds[1] += 1

    try:
        powers = np.arange(logbounds[0], logbounds[1] + 1)
    except ValueError:
        powers = np.arange(1)

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

    """
    input_order can be a mix of protein names, protein aliases, integers, and '*'
    - protein names and aliases will be converted to lowercase to find matches
    - integers will be used as indices
    - '*' will be replaced by the missing indices
    """

    input_names = network.get_inverted_input_proteins()
    output_names = network.get_output_proteins()

    lower_input_names = [n.lower() for n in input_names]
    lower_protein_aliases = {k.lower(): v for k, v in protein_aliases.items()} if protein_aliases else {}

    if input_order is not None:
        old_order = deepcopy(input_order)

        if any(isinstance(i, str) for i in old_order):
            input_order = []
            for iname in old_order:
                if isinstance(iname, str):
                    if iname == '*':
                        input_order.append('*')
                    else:
                        iname = iname.lower()
                        if iname in lower_input_names:
                            input_order.append(lower_input_names.index(iname))
                        elif iname in lower_protein_aliases:
                            input_order.append(lower_input_names.index(lower_protein_aliases[iname]))
                        else:
                            raise ValueError(f'Invalid protein name: {iname}')
                else:
                    # should be a regular index
                    assert isinstance(iname, (int, np.integer)), f'Invalid protein index: {iname}'
                    assert iname in range(len(input_names)), f'Invalid protein index: {iname}'
                    input_order.append(iname)

        assert len(input_order) == len(input_names), f'Wrong number of inputs: {input_order}'

        if '*' in input_order:
            missing = set(range(len(input_names))) - set(input_order)
            input_order = [i if i != '*' else missing.pop() for i in input_order]

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
    ticks = np.array(rescaler.fwd(unscaled_ticks))
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
    try:
        ax.set_xlim(xlims_margin)
        ax.set_xticks(rescaler.fwd(p10))  # major ticks
        ax.xaxis.set_major_formatter(PowerFormatter(p10, **kw))
        p10_minor = powers_of_ten(xmin=xlims_inv[0], xmax=xlims_inv[1], resolution=10)
        ax.set_xticks(rescaler.fwd(p10_minor), minor=True)
    except ValueError as e:
        ...

    return xlims_inv


def setup_transformed_yaxis(ax, yaxis_lims, rescaler, margins=0.05, **kw):
    ylims_tr = np.asarray(yaxis_lims)
    ylims_inv = rescaler.inv(np.asarray(ylims_tr))
    p10 = powers_of_ten(xmin=ylims_inv[0], xmax=ylims_inv[1])
    ylims_margin = ylims_tr + np.array([-1, 1]) * margins * np.diff(ylims_tr)
    try:
        ax.set_ylim(ylims_margin)
        ax.set_yticks(rescaler.fwd(p10))
        ax.yaxis.set_major_formatter(PowerFormatter(p10, **kw))
        p10_minor = powers_of_ten(xmin=ylims_inv[0], xmax=ylims_inv[1], resolution=10)
        ax.set_yticks(rescaler.fwd(p10_minor), minor=True)
    except Exception as e:
        ...
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
    ticks = {'major': rescaler.fwd(p10), 'minor': rescaler.fwd(p10_minor)}
    pf = PowerFormatter(p10, **kw)
    labels = [(rescaler.fwd(x), pf(x, i)) for i, x in enumerate(p10)]
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


# ╭─────────────────────────────────────────────╮
# │             PLOTTING PRIMITIVES             │
# ╰───────────────────── ⟱ ─────────────────────╯
## {{{                          --     heatmap     --
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


    print(f'in heatmap, contours={contours}')

    if isinstance(ax, list):
        ax = ax[0]

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
    if contours is not None and contours > 0:
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

