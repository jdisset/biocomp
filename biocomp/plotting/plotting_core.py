# {{{                          --     imports     --
# ···············································································


# TODO: CLEAN UP utils so that there's no jax in it (separate into utils and jax_utils)

from os import getenv
from functools import partial
import numpy as np
from biocomp import utils as ut
from biocomp.datautils import DataRescaler
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import difflib
import os
from typing import Sequence
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


## {{{                   --     default configuration     --


os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"
configurable = ut.configurable_decorator("biocomp.plotting")

BIOCOMP_COLORS = dr.resolve_all_lazy(dr.load("pkg:biocomp:config/biocomp_colors.yaml"))
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


def get_reordered_protein_names(
    network, input_order=None, protein_aliases=None, only_dependent_outputs=True, **_
):
    """Resolve a column convention for X (and the dependent-output positions for Y).

    `input_order` semantics:
      - ``None``: identity. X is assumed to already be in network order
        (see `network.get_inverted_input_proteins()`). No reorder applied.
      - ``"inv"``: reverse network order.
      - list of ints / protein names / aliases / ``"*"``: explicit permutation
        into network order.

    Never silently falls back to alphabetical or any other heuristic — that
    fallback was the root cause of the X-column scrambling bug class
    (see bugs/eval-x-axis-permutation-iRFP720.md). Callers wanting a
    display-order sort must request it explicitly via the protein-name list
    form (typically `recipe.input_order` on the network's recipe).

    Returns a 4-tuple: ``(in_order, output_pos, reordered_input_names, output_name)``.
    `output_pos` and `output_name` are scalars when there's a single dependent
    output; lists otherwise.
    """
    input_names = network.get_inverted_input_proteins()
    output_names = network.get_output_proteins(only_dependent_outputs=only_dependent_outputs)

    lower_input_names = [n.lower() for n in input_names]
    lower_protein_aliases = (
        {k.lower(): v for k, v in protein_aliases.items()} if protein_aliases else {}
    )

    if input_order is None:
        # Identity: X is already in network order.
        in_order = list(range(len(input_names)))
        reordered_input_names = list(input_names)
    elif input_order == "inv":
        in_order = list(range(len(input_names) - 1, -1, -1))
        reordered_input_names = [input_names[i] for i in in_order]
    else:
        old_order = deepcopy(input_order)
        resolved: list = []
        if any(isinstance(i, str) for i in old_order):
            for iname in old_order:
                if isinstance(iname, str):
                    if iname == "*":
                        resolved.append("*")
                    else:
                        iname_low = iname.lower()
                        if iname_low in lower_input_names:
                            resolved.append(lower_input_names.index(iname_low))
                        elif iname_low in lower_protein_aliases:
                            resolved.append(
                                lower_input_names.index(lower_protein_aliases[iname_low])
                            )
                        else:
                            raise ValueError(f"Invalid protein name: {iname}")
                else:
                    assert isinstance(iname, (int, np.integer)), f"Invalid protein index: {iname}"
                    assert iname in range(len(input_names)), f"Invalid protein index: {iname}"
                    resolved.append(int(iname))
        else:
            resolved = [int(i) for i in old_order]

        assert len(resolved) == len(input_names), (
            f"Wrong number of inputs: {resolved=}, {input_names=}"
        )

        if "*" in resolved:
            missing = set(range(len(input_names))) - set(resolved)
            resolved = [i if i != "*" else missing.pop() for i in resolved]

        in_order = resolved
        reordered_input_names = [input_names[i] for i in in_order]

    # output_names already respects only_dependent_outputs from get_output_proteins
    output_name = list(output_names)
    if len(output_name) > 1:
        logger.debug(f"multiple output proteins found: {output_name}")
    all_outputs = network.get_output_proteins(only_dependent_outputs=False)
    output_pos = [all_outputs.index(n) for n in output_name]

    if protein_aliases is not None:
        reordered_input_names = [protein_aliases.get(n, n) for n in reordered_input_names]
        output_name = [protein_aliases.get(n, n) for n in output_name]

    if len(output_pos) == 1:
        return in_order, output_pos[0], reordered_input_names, output_name[0]
    return in_order, output_pos, reordered_input_names, output_name


def network_ticks_and_labels(network, rescaler, xmin=0, xmax=1, **kw):
    from biocomp.plotutils import ShortScientificFormatter

    scformat = ShortScientificFormatter()
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


def get_transformed_ticks_and_labels(axis_lims: Sequence[float], rescaler: DataRescaler, **kw):
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
    show_minor_labels=False,
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
        if show_minor_labels:
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
    setup_xaxis_params=None,
    setup_yaxis_params=None,
    **kw,
):
    if setup_yaxis_params is None:
        setup_yaxis_params = {}
    if setup_xaxis_params is None:
        setup_xaxis_params = {}
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


USE_KNN_JAX = getenv("BC_KNN_USE_JAX", default=False)


def build_tree(x, use_jax=USE_KNN_JAX):
    # filter out nan/inf values before building tree
    import numpy as np

    if not use_jax:
        mask = np.all(np.isfinite(x), axis=1) if x.ndim > 1 else np.isfinite(x)
        x_clean = x[mask]
        if len(x_clean) == 0:
            raise ValueError("No finite data points available for building KD-tree")

    if use_jax:
        import jaxkd as jk
        import jax

        tree = jax.jit(jk.build_tree)(x)
    else:
        from scipy.spatial import KDTree

        tree = KDTree(x_clean)
        # store the mask to use later for filtering other arrays
        tree._finite_mask = mask
        tree._original_x = x
    return tree


def _ball_volume(d: int) -> float:
    # Volume of the unit d-ball
    from scipy.special import gamma

    return (np.pi ** (d / 2.0)) / gamma(d / 2.0 + 1.0)


def per_point_knn_density(tree, X_ref=None, kdensity: int = 50):
    """
    kNN density for the points that define `tree` (points per unit d-volume).
    Works in any dimension d = X_ref.shape[1].
    """
    if X_ref is None:
        X_ref = getattr(tree, "data", None)
        if X_ref is None:
            raise ValueError(
                "Cannot infer reference coordinates for density. "
                "Pass X_ref or use a tree exposing `.data`."
            )

    dists, _ = tree.query(X_ref, k=kdensity + 1)
    rk = dists[:, -1]

    d = X_ref.shape[1]
    Vd = _ball_volume(d)
    rho = kdensity / (Vd * np.maximum(rk, 1e-12) ** d)  # points per unit volume
    return rho


def uniform_resampling(
    X, npoints: int = 1000, kdensity=50, density_floor_q=0.01, density_cap_q=0.99
):
    tree = build_tree(X)
    densities = per_point_knn_density(tree=tree, X_ref=X, kdensity=kdensity)
    density_floor = float(np.quantile(densities, density_floor_q))
    density_cap = float(np.quantile(densities, density_cap_q))
    densities = np.clip(densities, density_floor, density_cap)
    weights = 1.0 / densities
    weights /= weights.sum()
    indices = np.random.choice(np.arange(X.shape[0]), size=npoints, replace=True, p=weights)
    return indices, weights[indices]


@configurable
def knn_stats(
    xquery,
    y=None,
    tree=None,  # KDTree or jaxkd tree
    iw=None,  # tuple of (indices, weights) of the k-nearest neighbors
    k=500,
    min_points=20,
    stats: str | list[str] = "iw",
    use_jax=USE_KNN_JAX,
    weight_by_densities: bool = False,
    kdensity: int = 50,  # k for density pilot if weights_by_densities
    density_power: float = 0.0,  # alpha in dens^{-alpha}; 0 disables
    density_floor_q: float | None = 0.01,
    density_cap_q: float | None = 0.99,  # quantile-based floor/cap if densities are used
    **kw,
):
    if isinstance(stats, str):
        stats = [stats]
    if use_jax:
        from .knn_utils_jax import get_gaussian_weighted_knn, get_knn_mean_and_variance
        from jax import numpy as xnp
    else:
        from .knn_utils_np import get_gaussian_weighted_knn, get_knn_mean_and_variance

        xnp = np

    if tree is None and iw is None:
        tree = build_tree(xquery, use_jax=use_jax)

    if weight_by_densities:
        X_ref = kw.get("X_ref", None)
        densities = per_point_knn_density(tree=tree, X_ref=X_ref, kdensity=kdensity)
        # floor/cap via quantiles (if provided in kw)
        if density_floor_q is not None:
            kw["density_floor"] = float(xnp.quantile(densities, density_floor_q))
        if density_cap_q is not None:
            kw["density_cap"] = float(xnp.quantile(densities, density_cap_q))
        kw["densities"] = densities
        kw["density_power"] = density_power

    iw = iw or get_gaussian_weighted_knn(
        xquery,
        tree,
        k=k,
        min_points=min_points,
        **kw,
    )

    assert iw[0].shape[1] == iw[1].shape[1] == k, (
        f"Wrong shape for indices and weights: {iw[0].shape=}, {iw[1].shape=}, {k=}"
    )
    assert iw[0].shape[0] == xquery.shape[0], (
        f"Wrong shape for indices and weights: {iw[0].shape=}, {iw[1].shape=}, {xquery.shape=}"
    )

    need_mv = {"mean", "variance", "std"} & set(stats)
    mean, var = (  # type: ignore
        get_knn_mean_and_variance(xquery, y, iw=iw, k=k, min_points=min_points, **kw)
        if need_mv
        else (None, None)
    )

    def calc(s):
        if s == "iw":
            return iw
        if s == "density":
            # (left as-is to minimize changes; note this is not a calibrated density)
            return xnp.nansum(iw[1], 1)
        if s == "quantile":
            from .knn_utils_jax import get_knn_quantile

            return get_knn_quantile(xquery, y, iw=iw, k=k, min_points=min_points, **kw)
        if s == "mean":
            return mean
        if s == "variance":
            return var
        if s == "std":
            return xnp.sqrt(var)  # type: ignore
        raise ValueError(f"Unknown stat: {s}")

    res = tuple([calc(s) for s in stats])
    return res[0] if len(res) == 1 else res


def weighted_kde_1d(
    values,
    weights=None,
    *,
    kde_points: int = 80,
    pad_frac: float = 0.15,
    bw_method=None,
):
    """Return weighted 1D KDE as ``(grid, density)`` or ``None`` when ill-posed."""
    from scipy.stats import gaussian_kde

    v = np.asarray(values).ravel()
    if weights is None:
        w = np.ones_like(v, dtype=float)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.shape != v.shape:
            raise ValueError(
                f"weights shape {w.shape} must match values shape {v.shape}"
            )

    finite = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if finite.sum() < 3:
        return None
    v = v[finite]
    w = w[finite]

    if np.unique(v).size < 2:
        return None

    wsum = float(w.sum())
    if not np.isfinite(wsum) or wsum <= 0:
        return None
    w = w / wsum

    try:
        kde = gaussian_kde(v, weights=w, bw_method=bw_method)
    except (np.linalg.LinAlgError, ValueError):
        return None

    v_lo, v_hi = float(v.min()), float(v.max())
    span = max(v_hi - v_lo, 1e-9)
    pad = span * float(pad_frac)
    grid = np.linspace(v_lo - pad, v_hi + pad, int(kde_points))
    density = np.asarray(kde(grid), dtype=float)
    return grid, density


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

        # main visible contours (solid lines)
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

            # get the lowest contour level
            if hasattr(cntrs, "levels") and len(cntrs.levels) > 0:
                lowest_level = cntrs.levels[0]
            else:
                lowest_level = cntrs.levels  # For single level case

            # create a single-level contour specifically for clipping
            clip_cntrs = ax.contour(
                Z_contour.T,
                levels=[lowest_level],  # Just the lowest level
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
            lowest_contour_path = clip_cntrs.get_paths()[0]
            clip_path = mpl.patches.PathPatch(lowest_contour_path, transform=ax.transData)
            im.set_clip_path(clip_path)
            if len(lowest_contour_path.vertices) == 0:
                im.remove()

    return im, cntrs


##────────────────────────────────────────────────────────────────────────────}}}
