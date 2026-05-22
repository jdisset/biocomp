# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Smooth (KNN-conditioned) plotting facade.

Re-exports 1D / 2D kernels and hosts the legacy network-aware
`smooth_line_plot` / `smooth_line_slices` callers.
"""

import numpy as np
import matplotlib.pyplot as plt

# Pull plotutils into the loader graph first to match the legacy load order:
# plotting_smooth historically imported plotutils at module top, which forced
# plotutils to finish initializing (including its lazy `plotting_3d` import) in
# the same loader frame. Doing it here keeps that contract intact when the
# split sub-modules are imported below.
from biocomp.plotutils import (  # noqa: F401
    IDENTITY_RESCALER as _IDENTITY_RESCALER,
    PlotFunctionResult,
    make_xy_grid,
)

from .plotting_core import (
    build_tree,
    format_powers,
    get_reordered_protein_names,
    knn_stats,
    setup_transformed_axis,
)

from jeanplot.data.grid import (  # noqa: F401
    GridData,
    extract_grid_data,
    grid_data_from_b64,
    grid_data_to_b64,
)
from jeanplot.plots.colorbar import colorbar  # noqa: F401
from jeanplot.plots.smooth_1d import (  # noqa: F401
    DEFAULT_MARKER_ROTATION,
    _annotate_theta,
    _draw_tail_fits,
    _linear_fit_overlay,
    make_n_props,
    smooth_1d,
)
from jeanplot.plots.smooth_2d import (  # noqa: F401
    KnnGradientField,
    gradient_field_2d,
    knn_gradient_grid,
    smooth_2d,
    smooth_grad_magnitude_2d,
)
from jeanplot.plots.smooth_kernel import (  # noqa: F401
    _KNN_GRID_CACHE,
    _KNN_GRID_CACHE_MAX,
    _finite_xy,
    _knn_grid_cache_key,
    _render_smooth_heatmap,
    _resolve_lims,
    _resolve_vlims,
    knn_grid,
)


# ─────────────────────────────────────────────────────────────────────────────
# Legacy smooth line plotting (kept here -- no current external callers)
# ─────────────────────────────────────────────────────────────────────────────


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
    with_quantiles=None,
    sample_quantiles_at=None,
    **kw,
):
    if with_quantiles is None:
        with_quantiles = [0.25, 0.75]
    input_order, output_pos, input_names, output_name = get_reordered_protein_names(
        network, **kw
    )

    y = y[:, output_pos]

    if tree is None:
        x = x[:, input_order]
        tree = build_tree(x)

    xquery = np.linspace(xmin, xmax, res).reshape(-1, 1)
    slice_at = np.array([]) if slice_at is None else np.array(slice_at)
    slice_at = np.array(slice_at)

    if x.shape[1] > 1:
        assert slice_at.shape == (x.shape[1] - 1,)
        xquery = np.concatenate([xquery, np.tile(slice_at, (xquery.shape[0], 1))], axis=1)

    z = knn_stats(xquery, y, tree=tree, stats="mean", **kw)

    ax.plot(xquery[:, 0], z, label=label, color=color, lw=lw, marker=marker, markevery=markevery)
    if with_quantiles is not None:
        if sample_quantiles_at is None:
            sample_quantiles_at = xquery[:, 0][::markevery]

    ax.set_xlabel(input_names[0])
    ax.set_ylabel(output_name)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, 1)

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
    markers=None,
    **kwargs,
):
    # slices is a list of list of slice values. (max 2 dimensions)
    if markers is None:
        markers = ["x", "o", "^", "v", "<", ">", "1", "2", "3", "4", "8", "p", "P", "*", "h", "H"]
    assert len(slices) <= 2, "Can only slice maximum 2 dimensions"
    outerslices = slices[1] if len(slices) > 1 else []
    innerslices = slices[0] if len(slices) > 0 else []

    input_order, _output_pos, input_names, _output_name = get_reordered_protein_names(
        network, input_order=input_order
    )

    x = x[:, input_order]

    xmin = xmin if xmin is not None else x[:, 0].min()
    xmax = xmax if xmax is not None else x[:, 0].max()

    same_ax = axes is None
    if same_ax:
        assert ax is not None
    else:
        assert len(axes) == len(outerslices), "Number of axes must match number of outer slices"

    tree = build_tree(x)

    color = "k"
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

    if same_ax:
        setup_transformed_axis(
            iax,
            xaxis_lims=[xmin, xmax],
            yaxis_lims=vlims,
            rescaler=rescaler,
            margins=0.05,
            **kwargs,
        )


__all__ = [
    # 1D
    "DEFAULT_MARKER_ROTATION",
    "make_n_props",
    "smooth_1d",
    # 2D
    "GridData",
    "KnnGradientField",
    "colorbar",
    "extract_grid_data",
    "gradient_field_2d",
    "grid_data_from_b64",
    "grid_data_to_b64",
    "knn_gradient_grid",
    "knn_grid",
    "smooth_2d",
    "smooth_grad_magnitude_2d",
    "smooth_line_plot",
    "smooth_line_slices",
]
