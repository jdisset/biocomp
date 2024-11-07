# {{{                          --     imports     --
# ···············································································

import jax.numpy as jnp
from functools import partial
import matplotlib as mpl

import numpy as np
import matplotlib.pyplot as plt
from typing import (
    Union,
    Sequence,
    List,
    Tuple,
    Dict,
    Any,
    Optional,
    TypeVar,
    TypeAlias,
    Literal,
)
from . import plotting_core as pc

from biocomp.plotutils import (
    make_xy_grid,
)


from .plotting_core import (
    DEFAULT_CMAP_NAME,
    setup_transformed_axis,
    get_reordered_protein_names,
    knn_avg,
    get_knn_quantile,
    format_powers,
    heatmap,
)

from scipy.spatial import KDTree

KDtree = partial(KDTree, leafsize=32)

T = TypeVar("T")
ListOrSingle: TypeAlias = Union[List[T], T]
NdArray = Union[np.ndarray, jnp.ndarray]
configurable = pc.configurable

##────────────────────────────────────────────────────────────────────────────}}}


def print_rc_params():
    for key, value in mpl.rcParams.items():
        print(f"{key}: {value}")


# ---- smooth plots (gaussian neighborhood based)


### {{{                            --     1D     --
DEFAULT_MARKER_ROTATION: tuple = (
    "o",
    "x",
    "s",
    "^",
    "*",
    "v",
    "+",
    "<",
    ">",
    "d",
    "p",
    "P",
    "h",
    "H",
)


def make_n_props(n: int, props: Optional[Dict | List]) -> List[Dict]:
    if props is None:
        props = [{}] * n
    elif not isinstance(props, list):
        props = [props] * n
    if len(props) != n:
        raise ValueError(f"props must have length {n}")
    return props


@configurable
def smooth_1d(
    X: NdArray,
    Y: NdArray,
    input_names: Sequence[str],
    output_name: str,
    rescaler,
    ax,
    slices: Optional[NdArray] = None,
    title: Optional[str] = None,
    xtitle: Optional[str] = None,
    ytitle: Optional[str] = None,
    xlims=(0, 1),
    vlims=(0, None),
    draw_xlabel=True,
    draw_ylabel=True,
    res=500,
    show_std=True,
    show_legend=True,
    std_alpha: float = 0.15,
    std_mode: str = "errorbar",  # "errorbar" or "fill"
    n_errorbars: int = 5,
    lineplot_props: Optional[List[Dict] | Dict] = None,
    errorbar_props: Optional[List[Dict] | Dict] = None,
    colors: Optional[List[Any]] = None,
    knn_avg_params: Dict = {},
):
    knn_radius = knn_avg_params.get("radius", 0.075)
    knn_avg_params["radius"] = knn_radius
    knn_avg_params.pop("avg_method", None)

    # remove nans
    nans = np.isnan(X).any(axis=1)
    if nans.any():
        X = X[~nans]
        Y = Y[~nans]

    if slices is not None:
        slices = np.asarray(slices)
    nslices = 1 if slices is None else slices.shape[0]
    n_input = X.shape[1]
    if n_input > 1:
        if slices is None:
            raise ValueError("slices must be provided for multi-dimensional input")
        if slices.shape[1] != n_input - 1:
            raise ValueError(f"slices shape must be (nslices, n_input - 1). Got {slices.shape}")

    lineplot_props = make_n_props(nslices, lineplot_props)
    errorbar_props = make_n_props(nslices, errorbar_props)

    if colors is not None:
        assert len(colors) == nslices
    else:
        colors = plt.get_cmap(DEFAULT_CMAP_NAME)(np.linspace(0.25, 1, nslices))
    assert colors is not None

    tree = KDTree(X)

    xmin, xmax = xlims
    xmax = X[:, 0].max() if xmax is None else xmax
    xmin = X[:, 0].min() if xmin is None else xmin

    xquery_max = min(float(xmax), X[:, 0].max() - knn_radius)
    xquery_min = max(float(xmin), X[:, 0].min() + knn_radius * 0.5)

    xquery = np.linspace(xquery_min, xquery_max, res).reshape(-1, 1)

    minz, maxz = np.inf, -np.inf
    for i in range(nslices):
        query = xquery
        if n_input > 1:
            assert slices is not None and slices.shape[1] == n_input - 1
            query = np.hstack([query, np.tile(slices[i], (query.shape[0], 1))])

        z, _ = knn_avg(query, Y, tree, avg_method="mean", **knn_avg_params)

        minz = min(minz, z.min())
        maxz = max(maxz, z.max())

        legend_label = ""
        for j in range(n_input - 1):
            iname = r"$X_{" + str(j + 2) + r"} \approx $"
            legend_label += f"{iname} {format_powers(rescaler.inv(slices[i][j]), n_decimals=0)}"
            if j < n_input - 2:
                legend_label += ", "

        marker = lineplot_props[i].get(
            "marker", DEFAULT_MARKER_ROTATION[i % len(DEFAULT_MARKER_ROTATION)]
        )

        DEFAULT_LINEPLOT_PROPS = {
            "lw": 1,
            "color": colors[i],
            "label": legend_label,
            "marker": marker,
            # use marker but don't show it, it's only for the legend:
            "markevery": -1,
        }
        lineplot_props[i] = {**DEFAULT_LINEPLOT_PROPS, **lineplot_props[i]}

        ax.plot(xquery, z, **lineplot_props[i])

        if show_std:
            # std instead:
            std, _ = knn_avg(query, Y, tree, avg_method="std", **knn_avg_params)
            minz = min(minz, z.min() - std.max())
            maxz = max(maxz, z.max() + std.max())

            if std_mode == "errorbar":
                n = len(z) // n_errorbars
                # shift proportional to i so that errorbars don't overlap
                shift = i * n // nslices
                qxquery = xquery[shift::n].squeeze()
                qz = z[shift::n].squeeze()
                yerr = std[shift::n].squeeze()

                DEFAULT_ERRORBAR_PROPS = {
                    "fmt": marker,
                    "color": colors[i],
                    "lw": 0,
                    "capsize": 2,
                    "capthick": 0.5,
                    "elinewidth": 0.5,
                    "markevery": 1,
                }
                errorbar_props[i] = {**DEFAULT_ERRORBAR_PROPS, **errorbar_props[i]}

                ax.errorbar(
                    qxquery,
                    qz,
                    yerr=yerr[::n].squeeze(),
                    **errorbar_props[i],
                )
            else:
                assert std_mode == "fill", f"std_mode must be 'errorbar' or 'fill'. Got {std_mode}"
                ax.fill_between(
                    xquery.squeeze(),
                    (z - std).squeeze(),
                    (z + std).squeeze(),
                    alpha=std_alpha,
                    color=colors[i],
                    lw=0,
                )

    minz = minz - 0.02 * (maxz - minz)
    maxz = maxz + 0.02 * (maxz - minz)

    vlims = [minz if vlims[0] is None else vlims[0], maxz if vlims[1] is None else vlims[1]]

    setup_transformed_axis(
        ax,
        xaxis_lims=xlims,
        yaxis_lims=vlims,
        rescaler=rescaler,
        margins=0.0,
    )

    # Show only bottom and left spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    xlabel = input_names[0] if xtitle is None else xtitle
    ylabel = output_name if ytitle is None else ytitle

    if nslices > 1 and show_legend:
        ax.legend(loc="upper right")

    if draw_xlabel:
        ax.set_xlabel(xlabel)
    if draw_ylabel:
        ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)


##────────────────────────────────────────────────────────────────────────────}}}
### {{{        --     2D     --
@configurable
def knn_grid(
    x: NdArray,
    y: NdArray,
    xlims,
    ylims,
    zslice=None,
    is_density_plot=False,
    grid_resolution=200,
    knn_avg_params={},
):
    xmin, xmax = xlims
    ymin, ymax = ylims or xlims
    xy = make_xy_grid(xmin, xmax, xres=grid_resolution, ymin=ymin, ymax=ymax, yres=grid_resolution)
    if x.shape[1] > 2:
        assert zslice is not None
        if zslice.shape != (x.shape[1] - 2,):
            raise ValueError(f"zslice.shape = {zslice.shape} != {x.shape[1] - 2}")
        xquery = np.hstack([xy, [zslice] * xy.shape[0]])
    else:
        xquery = xy

    tree = KDTree(x)
    output_values, density = knn_avg(xquery, y, tree=tree, **knn_avg_params)

    output_values = output_values.squeeze()

    if output_values.shape != (xy.shape[0],):
        raise ValueError(f"output_values.shape = {output_values.shape} != {xy.shape[0]}")
    if density.shape != (xy.shape[0],):
        raise ValueError(f"density.shape = {density.shape} != {xy.shape[0]}")

    if is_density_plot:
        output_values = density

    return xy, output_values


@configurable
def colorbar(
    ax,
    im,
    rescaler,
    vlims=(None, None),
    yslice=None,
    label=None,
    position=(1.1, 0.4),
    size=(0.04, 0.52),
    orientation: Literal["horizontal", "vertical"] = "vertical",
    label_position: Literal["left", "right", "bottom", "top"] = "right",
    label_props: Dict = {},
    show_minor_ticks=False,
    tick_props: Optional[ListOrSingle[Dict]] = None,
    border_width=0.7,
):
    imlims = im.get_clim()
    c_vmin = imlims[0] if vlims[0] is None else vlims[0]
    c_vmax = imlims[1] if vlims[1] is None else vlims[1]

    colorbar_ax = ax.inset_axes(position + size)
    cbar = plt.colorbar(im, cax=colorbar_ax, orientation=orientation)

    DEFAULT_TICK_PROPS = {
        "axis": "both",
        "which": "both",
        "direction": "out",
        "pad": 2,
        "labelsize": 8,
        # thickness:
        "width": 0.7,
    }
    cbar.ax.tick_params(**DEFAULT_TICK_PROPS)

    if tick_props is not None:
        if not isinstance(tick_props, list):
            tick_props = [tick_props]
        for tick_prop in tick_props:
            cbar.ax.tick_params(**tick_prop)

    for spine in cbar.ax.spines.values():
        spine.set_linewidth(border_width)

    mxticks = orientation == "vertical" and show_minor_ticks
    myticks = orientation == "horizontal" and show_minor_ticks

    setup_transformed_axis(
        cbar.ax,
        yaxis_lims=[c_vmin, c_vmax],
        xaxis_lims=[c_vmin, c_vmax],
        margins=0.0,
        rescaler=rescaler,
        show_minor_xticks=mxticks,
        show_minor_yticks=myticks,
    )

    if label is not None:
        if orientation == "vertical":
            if label_position not in ["left", "right"]:
                raise ValueError("Vertical orientation: label_position must be left or righ")
            cbar.ax.yaxis.set_label_position(label_position)  # type: ignore
            cbar.ax.set_ylabel(label, **label_props)
            cbar.ax.tick_params(axis="x", which="both", size=0)
            cbar.ax.set_xticks([])
        else:
            if label_position not in ["bottom", "top"]:
                raise ValueError("Horizontal orientation: label_position must be bottom or top")
            cbar.ax.xaxis.set_label_position(label_position)  # type: ignore
            cbar.ax.set_xlabel(label, **label_props)
            cbar.ax.tick_params(axis="y", which="both", size=0)
            cbar.ax.set_yticks([])

    return cbar


@configurable
def smooth_2d(
    X: NdArray,
    Y: NdArray,
    input_names: Sequence[str],
    output_name: str,
    rescaler,
    ax,
    zslice: Optional[NdArray] = None,
    title: Optional[str] = None,
    xtitle: Optional[str] = None,
    ytitle: Optional[str] = None,
    vtitle: Optional[str] = None,
    xlims=(0, 1),
    ylims=(None, None),
    vlims=(None, None),
    draw_xlabel=True,
    draw_ylabel=True,
    draw_colorbar=True,
    draw_colorbar_label=True,
    show_minor_xticks=False,
    show_minor_yticks=False,
    colorbar_params: Dict = {},
    knn_grid_params: Dict = {},
    heatmap_params: Dict = {},
) -> Tuple:
    ylims = xlims if ylims == (None, None) else ylims

    if isinstance(ax, (list, tuple)):
        ax = ax[0]

    # count any row in x with nan values
    nans = np.isnan(X).any(axis=1)
    if nans.any():
        X = X[~nans]
        Y = Y[~nans]

    zslice = np.asarray(zslice) if zslice is not None else None

    input_coords, output_values = knn_grid(
        X,
        Y,
        xlims,
        ylims,
        **{**knn_grid_params, "zslice": zslice},
    )

    im, cntrs = heatmap(ax, input_coords, output_values, **{**heatmap_params, "vlims": vlims})

    # as latex if xtitle not none
    xlabel = input_names[0] if xtitle is None else xtitle
    ylabel = input_names[1] if ytitle is None else ytitle

    if draw_xlabel:
        ax.set_xlabel(xlabel)
    if draw_ylabel:
        ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    setup_transformed_axis(
        ax,
        xaxis_lims=xlims,
        yaxis_lims=ylims,
        rescaler=rescaler,
        margins=0.0,
        show_minor_xticks=show_minor_xticks,
        show_minor_yticks=show_minor_yticks,
    )

    # spines only on bottom and left
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    vlabel = output_name if vtitle is None else vtitle

    if draw_colorbar:
        vlabel = vlabel if draw_colorbar_label else None
        colorbar(
            ax,
            im,
            rescaler,
            vlims,
            **{**colorbar_params, "label": vlabel},
        )

    return im, cntrs


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
        tree = KDTree(x)

    xquery = np.linspace(xmin, xmax, res).reshape(-1, 1)
    slice_at = np.array([]) if slice_at is None else np.array(slice_at)
    slice_at = np.array(slice_at)

    if x.shape[1] > 1:
        assert slice_at.shape == (x.shape[1] - 1,)
        xquery = np.concatenate([xquery, np.tile(slice_at, (xquery.shape[0], 1))], axis=1)

    z, _ = knn_avg(xquery, y, tree=tree, **kw)

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
            fmt="none",
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
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

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
    color_mode="inner_slice",
    markers=["x", "o", "^", "v", "<", ">", "1", "2", "3", "4", "8", "p", "P", "*", "h", "H"],
    **kwargs,
):
    # slices is a list of list of slice values. (max 2 dimensions)
    assert len(slices) <= 2, "Can only slice maximum 2 dimensions"
    outerslices = slices[1] if len(slices) > 1 else []
    innerslices = slices[0] if len(slices) > 0 else []

    protein_order, protein_names = get_reordered_protein_names(network, input_order=input_order)
    input_order, output_pos = protein_order[:-1], protein_order[-1]
    input_names, output_name = protein_names[:-1], protein_names[-1]

    x = x[:, input_order]
    tree = KDTree(x)

    xmin = xmin if xmin is not None else x[:, 0].min()
    xmax = xmax if xmax is not None else x[:, 0].max()

    same_ax = axes is None
    if same_ax:
        assert ax is not None
    else:
        assert len(axes) == len(outerslices), "Number of axes must match number of outer slices"

    color = "k"
    # cmap = plt.cm.YlGnBu
    cmap = plt.cm.Spectral
    vlims = np.array([np.inf, -np.inf])
    ivlims = np.tile(vlims, (len(outerslices), 1))
    for i, outsl in enumerate(outerslices):
        iax = ax if same_ax else axes[i]
        if color_mode == "outer_slice":
            color = cmap(i / len(outerslices))
        for j, insl in enumerate(innerslices):
            if color_mode == "inner_slice":
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
                label=f"{input_names[input_order[1]]} ≈ {format_powers(rescaler.inv(insl), n_decimals=0)}",
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
