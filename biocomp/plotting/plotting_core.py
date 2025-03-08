# {{{                          --     imports     --
# ···············································································
import jax
import jax.numpy as jnp
from functools import partial

from jax import jit, vmap
import numpy as np
from biocomp import utils as ut
from biocomp.datautils import DataRescaler
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as font_manager
import difflib
from jax.typing import ArrayLike
import os
from typing import Union, Sequence, List, Tuple, Dict, Any, Optional, Callable, TypeAlias, Literal
from matplotlib import colors as mcolors

from copy import deepcopy

import dracon as dr
from biocomp.logging_config import get_logger

logger = get_logger(__name__)

##────────────────────────────────────────────────────────────────────────────}}}

configurable = ut.configurable_decorator("biocomp.plotting")


# ╭─────────────────────────────────────────────╮
# │                TOOLS & UTILS                │
# ╰───────────────────── ⟱ ─────────────────────╯

NdArray: TypeAlias = Union[np.ndarray, jnp.ndarray]
NumLike: TypeAlias = Union[np.ndarray, jnp.ndarray, float, int]

## {{{                   --     default configuration     --

from matplotlib import colors as mcolors

os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"
configurable = ut.configurable_decorator("biocomp.plotting")


BIOCOMP_COLORS = dr.load("pkg:biocomp:config/colors.yaml")
cmap_definitions = BIOCOMP_COLORS["color_maps"] or {}

CUSTOM_CMAPS = {
    k: mcolors.LinearSegmentedColormap.from_list(k, v, N=256) for k, v in cmap_definitions.items()
}

# register custom colormaps
for k, v in CUSTOM_CMAPS.items():
    # check if it's already registered
    if k in plt.colormaps():
        plt.colormaps.unregister(k)
    plt.colormaps.register(v, name=k)

DEFAULT_CMAP_NAME = BIOCOMP_COLORS["default_color_map"] or "viridis"


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                   --     log_spline_log scale     --


def get_bio_color(name, default="k"):
    colors = {"ebfp": "#529edb", "eyfp": "#fbda73", "mkate": "#f75a5a", "neongreen": "#33f397"}
    colors["fitc"] = colors["neongreen"]
    colors["pe_texas_red"] = colors["mkate"]
    colors["pacific_blue"] = colors["ebfp"]
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
    lower_protein_aliases = (
        {k.lower(): v for k, v in protein_aliases.items()} if protein_aliases else {}
    )

    if input_order is not None:
        old_order = deepcopy(input_order)

        if any(isinstance(i, str) for i in old_order):
            input_order = []
            for iname in old_order:
                if isinstance(iname, str):
                    if iname == "*":
                        input_order.append("*")
                    else:
                        iname = iname.lower()
                        if iname in lower_input_names:
                            input_order.append(lower_input_names.index(iname))
                        elif iname in lower_protein_aliases:
                            input_order.append(
                                lower_input_names.index(lower_protein_aliases[iname])
                            )
                        else:
                            raise ValueError(f"Invalid protein name: {iname}")
                else:
                    # should be a regular index
                    assert isinstance(iname, (int, np.integer)), f"Invalid protein index: {iname}"
                    assert iname in range(len(input_names)), f"Invalid protein index: {iname}"
                    input_order.append(iname)

        assert len(input_order) == len(
            input_names
        ), f"Wrong number of inputs: {input_order=}, {input_names=}"

        if "*" in input_order:
            missing = set(range(len(input_names))) - set(input_order)
            input_order = [i if i != "*" else missing.pop() for i in input_order]

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

    if resolution > 1:
        increments = np.arange(2, resolution).reshape(-1, 1)
    else:
        increments = np.array([[1]])

    values = (base_powers * increments).flatten()

    values = values[(values >= xmin) & (values <= xmax)]
    return values


def format_powers(x, *_, n_decimals=1):
    x = float(x)
    abs_x = abs(x)
    if abs_x < 1000:
        if np.abs(x - int(x)) < 1e-3:
            return rf"${int(x)}$"  # No decimal point
        else:
            return rf"${x:.1f}$"  # Up to 1 decimal point
    else:
        E = int(np.log10(abs_x))
        if x == int(x):
            return r"${0:.0f}e{1}$".format(x // 10**E, E)
        else:
            return r"${0:.{2}f}e{1}$".format(x / 10**E, E, n_decimals)


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
            return ""
        return format_powers(v, None)


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
    ticks = {"major": rescaler.fwd(p10), "minor": rescaler.fwd(p10_minor)}
    pf = PowerFormatter(p10, **kw)
    labels = [(rescaler.fwd(x), pf(x, i)) for i, x in enumerate(p10)]

    return ticks, labels


def setup_transformed_axis_generic(
    ax,
    axis_lims,
    rescaler,
    axis="x",  # 'x' or 'y'
    margins=0.0,
    show_minor=False,
    major_tick_length=None,
    major_tick_width=None,
    minor_tick_length=None,
    minor_tick_width=None,
    label_fontsize=None,
    show_labels=True,
    spine_position=None,
    force_spine_only=False,
    **kw,
):
    # Get the appropriate axis object and methods based on axis parameter
    axis_obj = getattr(ax, f"{axis}axis")
    set_lim = getattr(ax, f"set_{axis}lim")
    set_ticks = getattr(ax, f"set_{axis}ticks")

    # Get the appropriate rcParams prefix
    rc_prefix = f"{axis}tick"

    # Determine spine position
    if spine_position is None:
        spine_position = "bottom" if axis == "x" else "left"

    # Handle spine visibility
    if force_spine_only:
        # Special handling for colorbar-like cases
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines[spine_position].set_visible(True)

        if axis == "x":
            ax.xaxis.set_ticks_position(spine_position)
            ax.xaxis.set_label_position(spine_position)
        else:
            ax.yaxis.set_ticks_position(spine_position)
            ax.yaxis.set_label_position(spine_position)

    lims_tr = np.asarray(axis_lims)
    lims_inv = rescaler.inv(np.asarray(lims_tr))
    p10 = powers_of_ten(xmin=lims_inv[0], xmax=lims_inv[1])
    lims_margin = lims_tr + np.array([-1, 1]) * margins * np.diff(lims_tr)

    try:
        set_lim(lims_margin)
        set_ticks(rescaler.fwd(p10))  # major ticks
        axis_obj.set_major_formatter(PowerFormatter(p10, **kw))

        p10_minor = powers_of_ten(xmin=lims_inv[0], xmax=lims_inv[1], resolution=10)
        set_ticks(rescaler.fwd(p10_minor), minor=True)
        if show_minor:
            axis_obj.set_minor_formatter(PowerFormatter(p10_minor, **kw))

        # Set up tick parameters
        if force_spine_only:
            # Special handling for colorbar-like cases
            tick_params_dict = {
                spine_position: True,
                f"label{spine_position}": True,
                "which": "both",
            }

            other_positions = {"top", "bottom", "left", "right"} - {spine_position}
            for pos in other_positions:
                tick_params_dict[pos] = False
                tick_params_dict[f"label{pos}"] = False
                ax.spines[pos].set_visible(True)

            ax.tick_params(axis=axis, **tick_params_dict)
        else:
            spine_name = "bottom" if axis == "x" else "left"
            tick_params_dict = {
                spine_name: plt.rcParams[f"{rc_prefix}.{spine_name}"],
                f"label{spine_name}": plt.rcParams[f"{rc_prefix}.label{spine_name}"],
                "which": "both",
            }
            ax.tick_params(axis=axis, **tick_params_dict)

        # major tick properties
        if major_tick_length is not None or major_tick_width is not None:
            ax.tick_params(
                axis=axis,
                which="major",
                length=major_tick_length
                if major_tick_length is not None
                else plt.rcParams[f"{rc_prefix}.major.size"],
                width=major_tick_width
                if major_tick_width is not None
                else plt.rcParams[f"{rc_prefix}.major.width"],
            )

        # minor tick properties
        if minor_tick_length is not None or minor_tick_width is not None:
            ax.tick_params(
                axis=axis,
                which="minor",
                length=minor_tick_length
                if minor_tick_length is not None
                else plt.rcParams[f"{rc_prefix}.minor.size"],
                width=minor_tick_width
                if minor_tick_width is not None
                else plt.rcParams[f"{rc_prefix}.minor.width"],
            )

        if label_fontsize is not None:
            ax.tick_params(axis=axis, labelsize=label_fontsize)

        if not show_labels:
            if axis == "x":
                ax.set_xticklabels([])
            else:
                ax.set_yticklabels([])

    except ValueError as e:
        logger.error(f"Error setting up {axis}-axis")
        logger.exception(e)

    return lims_inv


@configurable
def setup_xaxis(ax, xaxis_lims, rescaler, **kw):
    return setup_transformed_axis_generic(ax, xaxis_lims, rescaler, axis="x", **kw)


@configurable
def setup_yaxis(ax, yaxis_lims, rescaler, **kw):
    return setup_transformed_axis_generic(ax, yaxis_lims, rescaler, axis="y", **kw)


@configurable
def setup_transformed_axis(
    ax,
    xaxis_lims=None,
    yaxis_lims=None,
    rescaler=None,
    setup_xaxis_params={},
    setup_yaxis_params={},
    **kw,
):
    if xaxis_lims is not None:
        xaxis_lims = setup_xaxis(
            ax,
            xaxis_lims,
            rescaler,
            **setup_xaxis_params,
            **kw,
        )

    if yaxis_lims is not None:
        yaxis_lims = setup_yaxis(
            ax,
            yaxis_lims,
            rescaler,
            **setup_yaxis_params,
            **kw,
        )

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


class SpatialQueryGrid:
    def __init__(
        self,
        data: np.ndarray,
        resolution: int = 200,
    ):
        """Initialize the spatial grid structure"""
        self.__data = data
        self.__resolution = resolution
        # save data to /tmp/datadump.npy
        # np.save("/tmp/datadump.npy", data)

        self.make_grid()
        self.make_query_fn()

    def make_grid(self, bin_capacity: int | Literal["auto"] = "auto") -> None:
        """Create the spatial grid structure. Specifying bin_capacity instead 'auto' allows this method to be jitted."""

        @jax.jit
        def get_bin(point: ArrayLike) -> int:
            nd_pos = self.get_bin_nd(point)
            return jnp.ravel_multi_index(nd_pos, self.__grid_shape, mode="clip")  # type: ignore

        self.__lower = jnp.min(self.__data, axis=0)
        self.__upper = jnp.max(self.__data, axis=0)

        self.__binsize = jnp.min((self.__upper - self.__lower) / self.__resolution)
        self.__grid_shape = tuple(
            np.ceil((self.__upper - self.__lower) / self.__binsize).astype(int)
        )

        n_points = len(self.__data)

        point_indices = jnp.arange(n_points)
        point_positions = vmap(get_bin)(self.__data)
        bin_counts = jnp.bincount(point_positions)
        sorted_point_positions = point_indices[jnp.argsort(point_positions)]

        if bin_capacity == "auto":
            bin_capacity = int(jnp.max(bin_counts))
        else:
            bin_capacity = bin_capacity

        last_elt = jnp.full((1, bin_capacity), -1)  # just so that grid[-1] returns an "empty" bin

        @jax.jit
        def make_grid(bin_counts):
            csum = jnp.cumsum(bin_counts)
            start_idx = jnp.concatenate((jnp.zeros(1), csum[:-1])).astype(int)

            @vmap
            def impl(start_idx, end_idx):
                sortedpos = jax.lax.dynamic_slice(
                    sorted_point_positions, (start_idx,), (bin_capacity,)
                )
                cell_bincount = end_idx - start_idx
                cell_indices = jnp.where(
                    jnp.arange(bin_capacity) < cell_bincount,
                    sortedpos,
                    -1,
                )
                return cell_indices

            grid = impl(start_idx, csum)
            grid = jnp.concatenate((grid, last_elt), axis=0)
            return grid

        self.__grid = make_grid(bin_counts)

    @partial(jax.jit, static_argnums=(0,))
    def get_bin_nd(self, point: ArrayLike) -> ArrayLike:
        """Get n-dimensional bin coordinates for a point"""
        return jnp.floor((jnp.array(point) - self.__lower) / self.__binsize).astype(int)

    def make_query_fn(self) -> None:
        binsize = np.asarray(self.__binsize)
        grid_shape = np.asarray(self.__grid_shape)
        grid = jnp.asarray(self.__grid)
        data = jnp.asarray(self.__data)

        def query_impl(xquery, k, distance_upper_bound):
            """
            Query the spatial grid for k nearest neighbors within radius.

            Args:
                xquery: Query point
                k: Number of neighbors to return
                qradius: Search radius

            Returns:
                Tuple of (distances, indices) to nearest neighbors
            """
            qrange = int(np.ceil(distance_upper_bound / binsize))
            qlower = self.get_bin_nd(xquery) - qrange

            query_bins = jnp.meshgrid(*[jnp.arange(qrange * 2) + ql for ql in qlower])
            query_bins = jnp.stack([q.flatten() for q in query_bins], axis=-1)

            in_bounds = jnp.all((query_bins >= 0) & (query_bins < jnp.array(grid_shape)), axis=1)
            query_ids = jnp.ravel_multi_index(query_bins.T, grid_shape, mode="clip")  # type: ignore
            query_ids = jnp.where(in_bounds, query_ids, -1)
            candidates = grid[query_ids].flatten()
            candidates = jnp.pad(candidates, (0, k), constant_values=-1)
            candidate_positions = data[candidates]

            sqdists = jnp.sum(jnp.square(candidate_positions - xquery), axis=1)

            mask = (sqdists < distance_upper_bound**2) & (candidates != -1)
            sqdists = jnp.where(mask, sqdists, jnp.inf)
            candidates = jnp.where(mask, candidates, -1)
            topk_dist, topk_ids = jax.lax.top_k(-sqdists, k=k)
            topk = candidates[topk_ids]
            topk_dist = jnp.sqrt(-topk_dist)
            topk_dist = jnp.where(topk != -1, topk_dist, jnp.inf)

            return topk_dist, topk

        jitquery = jit(vmap(query_impl, in_axes=(0, None, None)), static_argnums=(1, 2))

        def query(xquery, k, distance_upper_bound):
            return jitquery(xquery, k, distance_upper_bound)

        self.query = query


@partial(jax.jit, static_argnums=(1, 2))
@partial(jax.vmap, in_axes=(None, 0, None, None))
def bfquery(data, xquery, k, distance_upper_bound):
    """
    Brute-force query for k nearest neighbors within radius.
    """
    distances = jnp.sum(jnp.square(data - xquery), axis=1)
    topk_dist, topk_ids = jax.lax.approx_max_k(-distances, k=k, recall_target=0.98)
    topk_dist = -topk_dist
    mask = topk_dist < (distance_upper_bound**2)
    topk_dist = jnp.where(mask, topk_dist, jnp.inf)
    topk_ids = jnp.where(mask, topk_ids, -1)
    topk_dist = jnp.sqrt(topk_dist)
    return topk_dist, topk_ids


@jax.jit
def weighted_quantile(data, weights, qu):
    ix = jnp.argsort(data)
    data = data[ix]
    weights = weights[ix]
    cdf = (jnp.cumsum(weights) - 0.5 * weights) / jnp.sum(weights)
    return jnp.interp(qu, cdf, data)


def gausspdf(x, mu, sigma):
    from scipy.stats import norm

    return norm.pdf(x, loc=mu, scale=sigma)


def jax_gausspdf(x, mu, sigma):
    return jax.scipy.stats.norm.pdf(x, loc=mu, scale=sigma)


def get_gaussian_weighted_knn_nojax(
    x: NdArray,
    tree,
    k: int = 500,  # number of neighbors to consider
    min_points: int = 20,  # minimum number of points to consider a neighborhood. fewer = nan
    radius: float = 0.1,
    sigma_in_radius: float = 3,  # sigma of the gaussian kernel in units of radius
):
    """Get the k-nearest neighbors of x in the tree,
    and return their indices together with their weights (from a gaussian kernel)."""

    distances, indices = tree.query(x, k=k, distance_upper_bound=radius)
    empty_neighbor_mask = distances == np.inf
    nb_points = (~empty_neighbor_mask).sum(axis=1)
    weights = gausspdf(distances, 0, radius / sigma_in_radius)
    indices[empty_neighbor_mask] = 0
    weights[empty_neighbor_mask] = 0
    weights[nb_points < min_points, :] = np.nan

    return indices, weights


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def get_gaussian_weighted_knn_jax(
    x: NdArray,
    data,
    k: int = 500,  # number of neighbors to consider
    min_points: int = 20,  # minimum number of points to consider a neighborhood. fewer = nan
    radius: float = 0.1,
    sigma_in_radius: float = 3,  # sigma of the gaussian kernel in units of radius
    n_devices: int = 1,
):
    """Get the k-nearest neighbors of x in the tree,
    and return their indices together with their weights (from a gaussian kernel)."""

    # pad x to be divisible by n_devices
    n_padding = n_devices - x.shape[0] % n_devices
    padded_x = jnp.pad(x, ((0, n_padding), (0, 0)))
    batches = jnp.asarray(jnp.split(padded_x, n_devices))

    res = jax.pmap(lambda x: bfquery(data, x, k, radius))(batches)

    distances = jnp.vstack(res[0])
    indices = jnp.vstack(res[1])

    # remove padding
    distances = distances[: x.shape[0]]
    indices = indices[: x.shape[0]]

    empty_neighbor_mask = indices == -1
    nb_points = (~empty_neighbor_mask).sum(axis=1)
    weights = jax_gausspdf(distances, 0, radius / sigma_in_radius)
    weights = jnp.where(empty_neighbor_mask, 0, weights)
    nbinferior = nb_points < min_points
    weights = jnp.where(nbinferior[:, None], jnp.nan, weights)

    return indices, weights


def get_gaussian_weighted_knn(x, tree, **kw):
    return get_gaussian_weighted_knn_nojax(x, tree, **kw)
    # if jax.devices()[0].platform == "gpu" or jax.devices()[0].platform == "tpu":
    #     return get_gaussian_weighted_knn_jax(x, tree.data, **kw)
    # elif len(jax.devices()) > 1:
    #     return get_gaussian_weighted_knn_jax(x, tree.data, n_devices=len(jax.devices()), **kw)
    # else:
    #     return get_gaussian_weighted_knn_nojax(x, tree, **kw)


def get_knn_mean(x, y, tree, **kw):
    """Get the k-nearest neighbors of x in the tree,
    and return their weighted average value together with their density."""

    indices, weights = get_gaussian_weighted_knn(x, tree, **kw)

    assert indices.shape == weights.shape
    normed_w = weights / weights.sum(axis=1)[:, None]
    weighted_mean = (y[indices] * normed_w[:, :, None]).sum(axis=1)

    density = np.nansum(weights, axis=1)

    return weighted_mean, density


def get_knn_std(x, y, tree, **kw):
    """
    Get the k-nearest neighbors of x in the tree,
    and return their weighted standard deviation.
    """

    indices, weights = get_gaussian_weighted_knn(x, tree, **kw)
    assert indices.shape == weights.shape
    normed_w = weights / weights.sum(axis=1)[:, None]
    weighted_mean = (y[indices] * normed_w[:, :, None]).sum(axis=1)

    # Compute weighted variance (and then std)
    squared_diff = (y[indices] - weighted_mean[:, None, :]) ** 2
    weighted_squared_diff = squared_diff * normed_w[:, :, None]
    variance = weighted_squared_diff.sum(axis=1)

    return jnp.sqrt(variance)


def get_knn_quantile(x, y, tree, qu, **kw):
    indices, weights = get_gaussian_weighted_knn(x, tree, **kw)
    q = jax.vmap(weighted_quantile, in_axes=(0, 0, None))(y[indices], weights, qu)
    density = np.nansum(weights, axis=1)
    return q, density


@configurable
def knn_avg(xquery, logY, tree, k=500, min_points=20, avg_method="mean", **kw):
    if avg_method == "mean":
        Z, p = get_knn_mean(xquery, logY, k=k, min_points=min_points, tree=tree, **kw)
    elif avg_method == "quantile":
        assert "qu" in kw
        Z, p = get_knn_quantile(xquery, logY, k=k, min_points=min_points, tree=tree, **kw)
    elif avg_method == "std":
        Z = get_knn_std(xquery, logY, k=k, min_points=min_points, tree=tree, **kw)
        p = np.ones
    else:
        raise ValueError(f"Unknown method {avg_method}")
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
    contours_alpha=1,
    contours_color="k",
    contours_linewidth=0.5,
    contours_linestyle="solid",
    contours_print=False,
    opacities=None,
    show_image=True,
    axtransform=None,
    cmap=DEFAULT_CMAP_NAME,
    transparent_below=None,
    transparent_above=None,
    image_interpolation=None,
    opacity=1,
    bad_color="#EEEEEE00",
    clip_to_lowest_contour=False,
):
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
    opacities *= opacity

    if transparent_below is not None:
        opacities = np.where(Z < transparent_below, 0, opacities)

    if transparent_above is not None:
        opacities = np.where(Z > transparent_above, 0, opacities)

    if np.isnan(Z).all():
        Z = np.zeros_like(Z)

    cntrs = None
    clip_cntrs = None
    if contours is not None:
        Z_contour = Z.copy()
        # also set the border to 0
        Z_contour[:, 0] = 0
        Z_contour[:, -1] = 0
        Z_contour[0, :] = 0
        Z_contour[-1, :] = 0

        # Main visible contours (solid lines)
        cntrs = ax.contour(
            Z_contour.T,
            levels=contours if isinstance(contours, (list, np.ndarray)) else contours,
            linewidths=contours_linewidth,
            linestyles=contours_linestyle,
            extent=[*xlims, *ylims],
            alpha=contours_alpha,
            colors=contours_color,
        )

        if clip_to_lowest_contour:
            # set nans to 0, so that contours are not broken
            Z_contour = np.nan_to_num(Z_contour)  # this allows to close contours that are open

            # invisible contours for clipping
            clip_cntrs = ax.contour(
                Z_contour.T,
                levels=cntrs.levels
                if isinstance(cntrs.levels, (list, np.ndarray))
                else [cntrs.levels],
                extent=[*xlims, *ylims],
                alpha=0,
                colors="none",
            )

            # dashed contours around NaN regions
            nan_mask = np.isnan(Z)
            if np.any(nan_mask):
                ax.contour(
                    Z_contour.T,
                    levels=cntrs.levels
                    if isinstance(cntrs.levels, (list, np.ndarray))
                    else [cntrs.levels],
                    extent=[*xlims, *ylims],
                    alpha=0.4,
                    linewidths=contours_linewidth * 0.95,
                    linestyles=[(0, (1, 3))],
                    dash_capstyle="round",
                    colors=contours_color,
                )

        if contours_print:
            ax.clabel(cntrs, inline=True, fontsize=8)

    im = None
    if show_image:
        if clip_to_lowest_contour and cntrs is not None:
            Z = np.nan_to_num(Z)

        im = ax.imshow(
            Z.T,
            origin="lower",
            aspect=1,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation=image_interpolation,
            alpha=opacities.T,
            extent=[*xlims, *ylims],
        )

        if clip_to_lowest_contour and clip_cntrs is not None:
            # Use the invisible solid contours for clipping
            all_paths = clip_cntrs.collections[0].get_paths()
            if len(all_paths) > 0:
                lowest_contour_path = all_paths[0]
                clip_path = mpl.patches.PathPatch(lowest_contour_path, transform=ax.transData)
                im.set_clip_path(clip_path)
            else:  # we "clip" everything out i.e. we delete the image
                im.remove()

    return im, cntrs


##────────────────────────────────────────────────────────────────────────────}}}
