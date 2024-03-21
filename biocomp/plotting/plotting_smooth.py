# {{{                          --     imports     --
# ···············································································
import jax
import jax.numpy as jnp
from matplotlib import scale as mscale
from functools import partial
from scipy.spatial import cKDTree
from jax import jit, vmap
import numpy as np
from biocomp import utils as ut
from biocomp import datautils as du
from biocomp import compute as cmp
from biocomp.datautils import DataManager
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
from . import plotting_core as pc
from .plotting_core import (
    DEFAULT_CMAP_NAME,
    setup_transformed_axis,
    get_reordered_protein_names,
    network_ticks_and_labels,
    make_xy_grid,
    knn_avg,
    get_knn_quantile,
    format_powers,
    apply_style,
    heatmap,
)

NdArray = Union[np.ndarray, jnp.ndarray]
configurable = pc.configurable
##────────────────────────────────────────────────────────────────────────────}}}


# ---- smooth plots (gaussian neighborhood based)
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
    color=plt.get_cmap(DEFAULT_CMAP_NAME)(0.7),
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
    ticks = np.array(rescaler.fwd(unscaled_ticks))
    ticks = ticks[ticks < xmax]
    tlabels = [
        scformat.format("{:m}", x) if i > 1 else ''
        for i, x in enumerate(unscaled_ticks[: len(ticks)])
    ]

    xquery_max = min(xmax, x.max() - radius)

    xquery = np.linspace(xmin, xquery_max, res).reshape(-1, 1)
    z, _ = knn_avg(
        xquery, y, tree, avg_method='mean', radius=radius, k=knn, min_points=min_points, **kw
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
            zq1, _ = knn_avg(
                xquery,
                y,
                tree,
                avg_method='quantile',
                qu=quantiles[0],
                radius=radius,
                k=knn,
                min_points=min_points,
                **kw,
            )
            zq9, _ = knn_avg(
                xquery,
                y,
                tree,
                avg_method='quantile',
                qu=quantiles[1],
                radius=radius,
                k=knn,
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
@configurable
def knn_grid(
    x, y, xlims, ylims, zslice=None, is_density_plot=False, grid_resolution=200, knn_avg_params={}
):

    print(
        f'knn_grid: {x.shape=}, {y.shape=}, {xlims=}, {ylims=}, {zslice=}, {is_density_plot=}, {grid_resolution=}, {knn_avg_params=}'
    )
    # stats on x and y
    print(f'knn_grid: {np.nanmin(x)=}, {np.nanmax(x)=}, {np.nanmean(y)=}, {np.nanmean(y)=}')

    xmin, xmax = xlims
    ymin, ymax = ylims or xlims
    xy = make_xy_grid(xmin, xmax, xres=grid_resolution, ymin=ymin, ymax=ymax, yres=grid_resolution)
    if x.shape[1] > 2:
        assert zslice is not None
        assert zslice.shape == (x.shape[1] - 2,)
        xquery = np.hstack([xy, [zslice] * xy.shape[0]])
    else:
        xquery = xy

    # stats on xquery
    print(f'knn_grid: {xquery.shape=}, {np.nanmin(xquery)=}, {np.nanmax(xquery)=}')


    tree = cKDTree(x)
    output_values, density = knn_avg(xquery, y, tree=tree, **knn_avg_params)

    output_values = output_values.squeeze()

    print(f'knn_grid: {xy.shape=}, {output_values.shape=}, {density.shape=}')
    print(f'knn_grid: {np.nanmin(output_values)=}, {np.nanmax(output_values)=}, {np.nanmin(density)=}, {np.nanmax(density)=}')

    if output_values.shape != (xy.shape[0],):
        raise ValueError(f'output_values.shape = {output_values.shape} != {xy.shape[0]}')
    if density.shape != (xy.shape[0],):
        raise ValueError(f'density.shape = {density.shape} != {xy.shape[0]}')

    if is_density_plot:
        output_values = density

    return xy, output_values


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
    xlims=(0, 1),
    ylims=(None, None),
    vlims=(None, None),
    draw_colorbar=True,
    knn_grid_params: Dict = {},
    heatmap_params: Dict = {},
) -> Tuple:

    ylims = xlims if ylims == (None, None) else ylims

    print(
        f'smooth_2d: {X.shape=}, {Y.shape=}, {input_names=}, {output_name=}, {rescaler=}, {ax=}, {zslice=}, {title=}, {xlims=}, {ylims=}, {vlims=}, {draw_colorbar=}, {knn_grid_params=}, {heatmap_params=}'
    )

    print(f'smooth_2d: {np.nanmin(X)=}, {np.nanmax(X)=}, {np.nanmin(Y)=}, {np.nanmax(Y)=}, {np.nanmean(X)=}, {np.nanmean(Y)=}')

    input_coords, output_values = knn_grid(
        X,
        Y,
        xlims,
        ylims,
        **{**knn_grid_params, 'zslice': zslice},
    )

    im, cntrs = heatmap(ax, input_coords, output_values, **{**heatmap_params, 'vlims': vlims})

    ax.set_xlabel(input_names[0])
    ax.set_ylabel(input_names[1])

    if title is not None:
        ax.set_title(title)

    setup_transformed_axis(
        ax,
        xaxis_lims=xlims,
        yaxis_lims=ylims,
        rescaler=rescaler,
        margins=0.0,
    )

    # spines only on bottom and left
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if draw_colorbar:
        imlims = im.get_clim()
        c_vmin = imlims[0] if vlims[0] is None else vlims[0]
        c_vmax = imlims[1] if vlims[1] is None else vlims[1]
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="7%", pad=0.5)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=6)
        # apply_style(cbar.ax)
        cbar.ax.tick_params(axis='both', which='both', direction='out', pad=2, labelsize=8)
        for spine in cbar.ax.spines.values():
            spine.set_linewidth(0.2)
        setup_transformed_axis(
            cbar.ax,
            yaxis_lims=[c_vmin, c_vmax],
            rescaler=rescaler,
            margins=0.0,
        )
        cbar.ax.set_ylabel(output_name, fontsize=8)

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
        tree = cKDTree(x)

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
