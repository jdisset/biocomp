# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""2D smooth heatmap, colorbar, KNN grid, and gradient-field plotting."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, TypeAlias, TypeVar

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from biocomp.plotutils import (
    IDENTITY_RESCALER as _IDENTITY_RESCALER,
    PlotFunctionResult,
)

from . import plotting_core as pc
from .plotting_core import (
    heatmap,
    setup_transformed_axis,
)

T = TypeVar("T")
ListOrSingle: TypeAlias = list[T] | T
NdArray = np.ndarray
configurable = pc.configurable


# ─────────────────────────────────────────────────────────────────────────────
# Grid Data container (raw output of smooth_2d, transportable across processes)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GridData:
    """Raw grid data from smooth_2d. values[yi, xi] = value at (x_coords[xi], y_coords[yi])."""

    x_coords: np.ndarray  # (R,)
    y_coords: np.ndarray  # (R,)
    values: np.ndarray  # (R, R)
    xlims: tuple[float, float]
    ylims: tuple[float, float]
    resolution: int
    input_names: list[str]
    output_name: str
    z_value: float | None = None


def extract_grid_data(
    output_values: np.ndarray,
    xlims: tuple[float, float],
    ylims: tuple[float, float],
    resolution: int,
    input_names: Sequence[str],
    output_name: str,
    z_value: float | None = None,
) -> GridData:
    return GridData(
        x_coords=np.linspace(xlims[0], xlims[1], resolution),
        y_coords=np.linspace(ylims[0], ylims[1], resolution),
        values=output_values.reshape(resolution, resolution),
        xlims=tuple(xlims),
        ylims=tuple(ylims),
        resolution=resolution,
        input_names=list(input_names),
        output_name=output_name,
        z_value=z_value,
    )


def grid_data_to_b64(grids: list[GridData]) -> str:
    """Serialize list of GridData to base64-encoded compressed npz."""
    import base64
    import io
    import json

    arrays: dict[str, np.ndarray] = {}
    meta: list[dict[str, Any]] = []
    for i, gd in enumerate(grids):
        p = f"t{i}_"
        arrays[f"{p}x"] = gd.x_coords.astype(np.float32)
        arrays[f"{p}y"] = gd.y_coords.astype(np.float32)
        arrays[f"{p}v"] = gd.values.astype(np.float32)
        meta.append(
            {
                "xlims": [float(gd.xlims[0]), float(gd.xlims[1])],
                "ylims": [float(gd.ylims[0]), float(gd.ylims[1])],
                "resolution": int(gd.resolution),
                "input_names": list(gd.input_names),
                "output_name": str(gd.output_name),
                "z_value": float(gd.z_value) if gd.z_value is not None else None,
            }
        )
    buf = io.BytesIO()
    np.savez_compressed(buf, _meta=np.array(json.dumps(meta)), _n=np.array(len(grids)), **arrays)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def grid_data_from_b64(b64_string: str) -> list[GridData]:
    """Deserialize list of GridData from base64-encoded npz."""
    import base64
    import io
    import json

    data = np.load(io.BytesIO(base64.b64decode(b64_string)), allow_pickle=False)
    n = int(data["_n"])
    meta: list[dict[str, Any]] = json.loads(str(data["_meta"]))
    return [
        GridData(
            x_coords=data[f"t{i}_x"],
            y_coords=data[f"t{i}_y"],
            values=data[f"t{i}_v"],
            xlims=tuple(m["xlims"]),
            ylims=tuple(m["ylims"]),
            resolution=m["resolution"],
            input_names=m["input_names"],
            output_name=m["output_name"],
            z_value=m.get("z_value"),
        )
        for i, m in enumerate(meta[:n])
    ]


def print_rc_params():
    for key, value in mpl.rcParams.items():
        print(f"{key}: {value}")


# ─────────────────────────────────────────────────────────────────────────────
# KNN grid (cached query of smoothed values over a regular lattice)
# ─────────────────────────────────────────────────────────────────────────────


_KNN_GRID_CACHE: dict = {}
_KNN_GRID_CACHE_MAX = 8


def _knn_grid_cache_key(
    x, y, xlims, ylims, zslice, is_density_plot, grid_resolution, knn_stats_params,
    max_centroid_offset_frac=0.0,
    query_mode="grid",
    query_seed=0,
):
    kx = pc.array_content_key(x)
    ky = pc.array_content_key(y)
    if kx is None or ky is None:
        return None
    kz = pc.array_content_key(np.asarray(zslice)) if zslice is not None else None
    return (
        kx, ky,
        tuple(xlims) if xlims is not None else None,
        tuple(ylims) if ylims is not None else None,
        kz,
        bool(is_density_plot),
        int(grid_resolution),
        tuple(sorted((knn_stats_params or {}).items())),
        float(max_centroid_offset_frac),
        str(query_mode),
        int(query_seed),
    )


@configurable
def knn_grid(
    x: NdArray,
    y: NdArray,
    xlims,
    ylims,
    zslice=None,
    is_density_plot=False,
    grid_resolution=200,
    knn_stats_params=None,
    max_centroid_offset_frac: float = 0.0,
    query_mode: Literal["grid", "uniform"] = "grid",
    query_seed: int = 0,
):
    """KNN-smoothed (X, Y) at query points covering xlims × ylims.

    `query_mode="grid"` (default) places `grid_resolution²` query points on
    a regular lattice. `query_mode="uniform"` draws the same number of
    query points uniformly at random -- useful when lattice alignment with
    underlying data structure causes visible aliasing.

    `max_centroid_offset_frac > 0` masks (NaN-fills) cells where the
    Gaussian-weighted centroid of neighbor positions sits more than
    `frac · kernel_sigma` from the cell center -- the direct measure of
    Nadaraya-Watson boundary asymmetry. Recommended values: 0.8-1.5;
    1.0 is a clean cut between interior and one-sided support.
    """
    from biocomp.plotutils import make_xy_grid
    from .plotting_core import build_tree, knn_stats

    if knn_stats_params is None:
        knn_stats_params = {}

    cache_key = _knn_grid_cache_key(
        x, y, xlims, ylims, zslice, is_density_plot, grid_resolution, knn_stats_params,
        max_centroid_offset_frac, query_mode, query_seed,
    )
    if cache_key is not None:
        cached = _KNN_GRID_CACHE.get(cache_key)
        if cached is not None:
            return cached

    mask = np.all(np.isfinite(x), axis=1) if x.ndim > 1 else np.isfinite(x)
    mask = mask & (np.all(np.isfinite(y), axis=1) if y.ndim > 1 else np.isfinite(y))

    if mask.all():
        x_clean, y_clean = x, y
    else:
        x_clean = x[mask]
        y_clean = y[mask]

    xmin, xmax = xlims
    ymin, ymax = ylims or xlims
    if query_mode == "uniform":
        rng = np.random.default_rng(int(query_seed))
        n_query = int(grid_resolution) ** 2
        xy = np.column_stack([
            rng.uniform(xmin, xmax, size=n_query),
            rng.uniform(ymin, ymax, size=n_query),
        ]).astype(np.float64)
    else:
        xy = make_xy_grid(xmin, xmax, xres=grid_resolution, ymin=ymin, ymax=ymax, yres=grid_resolution)

    if len(x_clean) == 0:
        return xy, np.full(xy.shape[0], np.nan)

    if x_clean.shape[1] > 2:
        assert zslice is not None
        if zslice.shape != (x_clean.shape[1] - 2,):
            raise ValueError(f"zslice.shape = {zslice.shape} != {x_clean.shape[1] - 2}")
        xquery = np.hstack([xy, [zslice] * xy.shape[0]])
    else:
        xquery = xy

    tree = build_tree(x_clean)
    primary = "density" if is_density_plot else "mean"
    requested = [primary, "centroid_offset"] if max_centroid_offset_frac > 0.0 else primary
    result = knn_stats(xquery, y_clean, tree=tree, stats=requested, **knn_stats_params)
    if max_centroid_offset_frac > 0.0:
        output_values, offset = result
        output_values = output_values.squeeze()
        radius = float(knn_stats_params.get("radius", 0.1))
        sigma_in_radius = float(knn_stats_params.get("sigma_in_radius", 3.0))
        boundary = np.asarray(offset) > max_centroid_offset_frac * (radius / sigma_in_radius)
        output_values = np.where(boundary, np.nan, output_values)
    else:
        output_values = result.squeeze()

    if output_values.shape != (xy.shape[0],):
        raise ValueError(f"output_values.shape = {output_values.shape} != {xy.shape[0]}")

    if cache_key is not None:
        if len(_KNN_GRID_CACHE) >= _KNN_GRID_CACHE_MAX:
            _KNN_GRID_CACHE.pop(next(iter(_KNN_GRID_CACHE)))
        _KNN_GRID_CACHE[cache_key] = (xy, output_values)
    return xy, output_values


# ─────────────────────────────────────────────────────────────────────────────
# Colorbar
# ─────────────────────────────────────────────────────────────────────────────


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
    tick_position: Literal["left", "right", "bottom", "top"] | None = "right",
    label_props: dict = None,
    tick_props: ListOrSingle[dict] | None = None,
    border_width=0.7,
    setup_transformed_axis_params: dict = None,
    threshold_below=None,
    threshold_above=None,
    alpha_opacity=1.0,
    cax=None,
):
    if setup_transformed_axis_params is None:
        setup_transformed_axis_params = {}
    else:
        setup_transformed_axis_params = dict(setup_transformed_axis_params)
    # The colorbar's scale axis must always show its tick labels -- a colorbar
    # without numbers is useless. Override any inherited show_labels=False that
    # cascaded down via nested_resolve from the heatmap's cell-level config
    # (where show_labels is used to hide ticks on non-edge cells).
    _active_axis_key = "setup_yaxis_params" if orientation == "vertical" else "setup_xaxis_params"
    _sub = setup_transformed_axis_params.get(_active_axis_key) or {}
    if not isinstance(_sub, dict):
        _sub = {}
    else:
        _sub = dict(_sub)
    _sub["show_labels"] = True
    setup_transformed_axis_params[_active_axis_key] = _sub
    if label_props is None:
        label_props = {}
    imlims = im.get_clim()
    c_vmin = imlims[0] if vlims[0] is None else vlims[0]
    c_vmax = imlims[1] if vlims[1] is None else vlims[1]

    colorbar_ax = (
        cax if cax is not None else ax.inset_axes([position[0], position[1], size[0], size[1]])
    )

    if threshold_below is not None or threshold_above is not None:
        from matplotlib.colors import ListedColormap

        cmap = im.get_cmap()
        colors = np.array(cmap(np.linspace(0, 1, 256)))
        values = np.linspace(c_vmin, c_vmax, len(colors))

        alpha_mask = np.ones(len(colors)) * alpha_opacity
        if threshold_below is not None:
            alpha_mask = np.where(values < threshold_below, 0, alpha_mask)
        if threshold_above is not None:
            alpha_mask = np.where(values > threshold_above, 0, alpha_mask)

        colors = np.column_stack([colors[:, :3], alpha_mask])
        threshold_cmap = ListedColormap(colors)

        cbar = plt.colorbar(
            mpl.cm.ScalarMappable(
                norm=mpl.colors.Normalize(vmin=c_vmin, vmax=c_vmax), cmap=threshold_cmap
            ),
            cax=colorbar_ax,
            orientation=orientation,
            aspect=20,
        )
    else:
        cbar = plt.colorbar(im, cax=colorbar_ax, orientation=orientation, aspect=20)

    if tick_position is None:
        tick_position = label_position

    if orientation == "vertical":
        if tick_position == "right":
            colorbar_ax.yaxis.set_ticks_position("right")
            colorbar_ax.tick_params(left=False)
        else:
            colorbar_ax.yaxis.set_ticks_position("left")
            colorbar_ax.tick_params(right=False)
    else:
        if tick_position == "top":
            colorbar_ax.xaxis.set_ticks_position("top")
            colorbar_ax.tick_params(bottom=False)
        else:
            colorbar_ax.xaxis.set_ticks_position("bottom")
            colorbar_ax.tick_params(top=False)

    # Default tick properties
    DEFAULT_TICK_PROPS = {
        "axis": "y" if orientation == "vertical" else "x",
        "which": "both",
        "direction": "out",
        "pad": 2,
        "labelsize": 8,
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

    setup_transformed_axis_params_with_spine = {
        "spine_position": tick_position,  # Use tick_position for the spine
        "force_spine_only": True,
        **setup_transformed_axis_params,
    }

    if orientation == "vertical":
        setup_transformed_axis(
            cbar.ax,
            yaxis_lims=[c_vmin, c_vmax],
            xaxis_lims=None,
            rescaler=rescaler,
            **setup_transformed_axis_params_with_spine,
        )

        if label_position not in ["left", "right"]:
            raise ValueError("Vertical orientation: label_position must be left or right")
        if tick_position not in ["left", "right"]:
            raise ValueError("Vertical orientation: tick_position must be left or right")

        cbar.ax.yaxis.set_label_position(label_position)

        if label is not None:
            cbar.ax.set_ylabel(label, **label_props)

        cbar.ax.tick_params(axis="x", which="both", size=0)
        cbar.ax.set_xticks([])
    else:
        setup_transformed_axis(
            cbar.ax,
            xaxis_lims=[c_vmin, c_vmax],
            yaxis_lims=None,
            rescaler=rescaler,
            **setup_transformed_axis_params_with_spine,
        )

        if label_position not in ["bottom", "top"]:
            raise ValueError("Horizontal orientation: label_position must be bottom or top")
        if tick_position not in ["bottom", "top"]:
            raise ValueError("Horizontal orientation: tick_position must be bottom or top")

        cbar.ax.xaxis.set_label_position(label_position)

        if label is not None:
            cbar.ax.set_xlabel(label, **label_props)

        cbar.ax.tick_params(axis="y", which="both", size=0)
        cbar.ax.set_yticks([])

    return cbar


# ─────────────────────────────────────────────────────────────────────────────
# 2D smooth heatmap + gradient field
# ─────────────────────────────────────────────────────────────────────────────


class KnnGradientField(NamedTuple):
    input_coords: np.ndarray
    gx: np.ndarray
    gy: np.ndarray
    x1_lat: np.ndarray
    x2_lat: np.ndarray
    xlims: list
    ylims: list


def _resolve_lims(X, xlims, ylims):
    dx = [X[:, 0].min(), X[:, 0].max()]
    dy = [X[:, 1].min(), X[:, 1].max()]
    return (
        [dx[0] if xlims[0] is None else xlims[0], dx[1] if xlims[1] is None else xlims[1]],
        [dy[0] if ylims[0] is None else ylims[0], dy[1] if ylims[1] is None else ylims[1]],
    )


def _finite_xy(X, Y):
    m = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y), axis=1)
    return (X, Y) if m.all() else (X[m], Y[m])


def _resolve_vlims(values, vlims, vlim_quantiles, vlim_min_floor, vlim_min_range):
    vlims = list(vlims)
    if vlim_quantiles is not None:
        finite = np.asarray(values)
        finite = finite[np.isfinite(finite)]
        q_lo, q_hi = vlim_quantiles
        if vlims[0] is None and q_lo is not None and finite.size:
            vlims[0] = float(np.quantile(finite, q_lo))
        if vlims[1] is None and q_hi is not None and finite.size:
            vlims[1] = float(np.quantile(finite, q_hi))
    if vlim_min_floor is not None and vlims[0] is not None:
        vlims[0] = float(min(vlims[0], vlim_min_floor))
    if vlim_min_range is not None and vlims[0] is not None and vlims[1] is not None:
        if (vlims[1] - vlims[0]) < vlim_min_range:
            vlims[1] = float(vlims[0] + vlim_min_range)
    return tuple(vlims)


def _render_smooth_heatmap(
    ax, input_coords, output_values,
    input_names, output_name,
    axis_rescaler, value_rescaler,
    xlims, ylims, resolution,
    *,
    title=None, title_kwargs=None,
    xtitle=None, ytitle=None, vtitle=None,
    vlims=(None, None), vlim_quantiles=(0.01, 0.99),
    vlim_min_floor=None, vlim_min_range=None,
    draw_xlabel=True, draw_ylabel=True,
    xaxis_labelpad=None, yaxis_labelpad=None,
    draw_colorbar=True, draw_colorbar_label=True,
    colorbar_params=None, heatmap_params=None,
    setup_transformed_axis_params=None,
):
    heatmap_params = heatmap_params or {}
    colorbar_params = colorbar_params or {}
    setup_transformed_axis_params = setup_transformed_axis_params or {}

    grid_data = extract_grid_data(
        np.asarray(output_values),
        xlims=tuple(xlims), ylims=tuple(ylims),
        resolution=resolution,
        input_names=list(input_names), output_name=output_name,
    )

    vlims = _resolve_vlims(output_values, vlims, vlim_quantiles, vlim_min_floor, vlim_min_range)
    im, cntrs = heatmap(ax, input_coords, output_values, **{**heatmap_params, "vlims": vlims})

    xlabel = input_names[0] if xtitle is None else xtitle
    ylabel = input_names[1] if ytitle is None else ytitle
    if draw_xlabel and xlabel:
        xkw = {"labelpad": xaxis_labelpad} if xaxis_labelpad is not None else {}
        ax.set_xlabel(xlabel, **xkw)
    if draw_ylabel and ylabel:
        ykw = {"labelpad": yaxis_labelpad} if yaxis_labelpad is not None else {}
        ax.set_ylabel(ylabel, **ykw)
    if title is not None:
        ax.set_title(title, **(title_kwargs or {}))

    setup_transformed_axis(
        ax, xaxis_lims=xlims, yaxis_lims=ylims, rescaler=axis_rescaler,
        **setup_transformed_axis_params,
    )

    if draw_colorbar:
        vlabel = (output_name if vtitle is None else vtitle) if draw_colorbar_label else None
        colorbar(ax, im, value_rescaler, vlims, **{**colorbar_params, "label": vlabel})

    return PlotFunctionResult(rendering=(im, cntrs), metadata={"grid_data": [grid_data]})


@configurable
def smooth_2d(
    X: NdArray,
    Y: NdArray,
    input_names: Sequence[str],
    output_name: str,
    rescaler,
    ax,
    zslice: NdArray | None = None,
    title: str | None = None,
    title_kwargs: dict | None = None,
    xtitle: str | None = None,
    ytitle: str | None = None,
    vtitle: str | None = None,
    xlims=(0, 1),
    ylims=(None, None),
    vlims=(None, None),
    vlim_quantiles: tuple[float | None, float | None] | None = (0.01, 0.99),
    vlim_min_floor: float | None = None,
    vlim_min_range: float | None = None,
    draw_xlabel=True,
    draw_ylabel=True,
    xaxis_labelpad=None,
    yaxis_labelpad=None,
    draw_colorbar=True,
    draw_colorbar_label=True,
    colorbar_params: dict = None,
    knn_grid_params: dict = None,
    heatmap_params: dict = None,
    setup_transformed_axis_params: dict = None,
) -> PlotFunctionResult:
    if isinstance(ax, list | tuple):
        ax = ax[0]
    knn_grid_params = dict(knn_grid_params or {})
    xlims, ylims = _resolve_lims(X, xlims, ylims)
    X, Y = _finite_xy(X, Y)
    zslice = np.asarray(zslice) if zslice is not None else None
    resolution = knn_grid_params.get("grid_resolution", 200)
    input_coords, output_values = knn_grid(
        X, Y, xlims, ylims, **{**knn_grid_params, "zslice": zslice},
    )
    return _render_smooth_heatmap(
        ax, input_coords, output_values, input_names, output_name,
        rescaler, rescaler, xlims, ylims, resolution,
        title=title, title_kwargs=title_kwargs,
        xtitle=xtitle, ytitle=ytitle, vtitle=vtitle,
        vlims=vlims, vlim_quantiles=vlim_quantiles,
        vlim_min_floor=vlim_min_floor, vlim_min_range=vlim_min_range,
        draw_xlabel=draw_xlabel, draw_ylabel=draw_ylabel,
        xaxis_labelpad=xaxis_labelpad, yaxis_labelpad=yaxis_labelpad,
        draw_colorbar=draw_colorbar, draw_colorbar_label=draw_colorbar_label,
        colorbar_params=colorbar_params, heatmap_params=heatmap_params,
        setup_transformed_axis_params=setup_transformed_axis_params,
    )


@configurable
def knn_gradient_grid(
    X: NdArray,
    Y: NdArray,
    xlims,
    ylims,
    knn_grid_params: dict | None = None,
    space: Literal["raw", "latent"] = "latent",
    rescaler=None,
) -> KnnGradientField:
    knn_grid_params = dict(knn_grid_params or {})
    resolution = knn_grid_params.get("grid_resolution", 200)
    input_coords, output_values = knn_grid(X, Y, xlims, ylims, **knn_grid_params)
    y_lat = np.asarray(output_values).reshape(resolution, resolution)
    x1_lat = np.linspace(xlims[0], xlims[1], resolution)
    x2_lat = np.linspace(ylims[0], ylims[1], resolution)

    if space == "raw":
        assert rescaler is not None, "rescaler required for space='raw'"
        x1_axis = np.asarray(rescaler.inv(x1_lat[:, None]).squeeze())
        x2_axis = np.asarray(rescaler.inv(x2_lat[:, None]).squeeze())
        y_field = np.asarray(rescaler.inv(y_lat[..., None]).squeeze())
    else:
        x1_axis, x2_axis, y_field = x1_lat, x2_lat, y_lat

    nan_mask = ~np.isfinite(y_field)
    if nan_mask.any() and not nan_mask.all():
        from scipy.ndimage import distance_transform_edt
        _, (ii, jj) = distance_transform_edt(nan_mask, return_indices=True)
        y_filled = y_field[ii, jj]
    else:
        y_filled = y_field

    gy, gx = np.gradient(y_filled, x2_axis, x1_axis)
    gy = np.where(nan_mask, np.nan, gy)
    gx = np.where(nan_mask, np.nan, gx)
    return KnnGradientField(input_coords, gx, gy, x1_lat, x2_lat, xlims, ylims)


@configurable
def smooth_grad_magnitude_2d(
    X: NdArray,
    Y: NdArray,
    input_names: Sequence[str],
    output_name: str,
    rescaler,
    ax,
    space: Literal["raw", "latent"] = "latent",
    title: str | None = None,
    title_kwargs: dict | None = None,
    xtitle: str | None = None,
    ytitle: str | None = None,
    vtitle: str | None = None,
    xlims=(0, 1),
    ylims=(None, None),
    vlims=(None, None),
    vlim_quantiles: tuple[float | None, float | None] | None = (0.0, 0.99),
    vlim_min_floor: float | None = None,
    vlim_min_range: float | None = None,
    draw_xlabel=True,
    draw_ylabel=True,
    xaxis_labelpad=None,
    yaxis_labelpad=None,
    draw_colorbar=True,
    draw_colorbar_label=True,
    colorbar_params: dict = None,
    knn_grid_params: dict = None,
    heatmap_params: dict = None,
    setup_transformed_axis_params: dict = None,
) -> PlotFunctionResult:
    if isinstance(ax, list | tuple):
        ax = ax[0]
    knn_grid_params = dict(knn_grid_params or {})
    xlims, ylims = _resolve_lims(X, xlims, ylims)
    X, Y = _finite_xy(X, Y)
    resolution = knn_grid_params.get("grid_resolution", 200)
    g = knn_gradient_grid(
        X, Y, xlims, ylims,
        knn_grid_params=knn_grid_params, space=space, rescaler=rescaler,
    )
    mag = np.hypot(g.gx, g.gy).ravel()
    return _render_smooth_heatmap(
        ax, g.input_coords, mag, input_names, f"|∇{output_name}|",
        rescaler, _IDENTITY_RESCALER, xlims, ylims, resolution,
        title=title, title_kwargs=title_kwargs,
        xtitle=xtitle, ytitle=ytitle, vtitle=vtitle,
        vlims=vlims, vlim_quantiles=vlim_quantiles,
        vlim_min_floor=vlim_min_floor, vlim_min_range=vlim_min_range,
        draw_xlabel=draw_xlabel, draw_ylabel=draw_ylabel,
        xaxis_labelpad=xaxis_labelpad, yaxis_labelpad=yaxis_labelpad,
        draw_colorbar=draw_colorbar, draw_colorbar_label=draw_colorbar_label,
        colorbar_params=colorbar_params, heatmap_params=heatmap_params,
        setup_transformed_axis_params=setup_transformed_axis_params,
    )


@configurable
def gradient_field_2d(
    X: NdArray,
    Y: NdArray,
    input_names: Sequence[str],
    output_name: str,
    rescaler,
    ax,
    xlims=(0, 1),
    ylims=(None, None),
    knn_grid_params: dict | None = None,
    space: Literal["raw", "latent"] = "latent",
    quiver_resolution: int = 22,
    normalize_arrows: bool = False,
    arrow_scale: float | None = None,
    arrow_width: float = 0.0025,
    quiver_props: dict | None = None,
    color_by: Literal["angle", "magnitude", "deviation_subtraction", "fixed"] = "angle",
    cmap: str = "twilight_shifted",
    fixed_color: str = "#222222",
    zero_dot_threshold: float = 0.05,
    zero_dot_size: float = 6.0,
    zero_dot_props: dict | None = None,
) -> PlotFunctionResult:
    if isinstance(ax, list | tuple):
        ax = ax[0]
    quiver_props = dict(quiver_props or {})
    xlims, ylims = _resolve_lims(X, xlims, ylims)
    X, Y = _finite_xy(X, Y)
    g = knn_gradient_grid(
        X, Y, xlims, ylims,
        knn_grid_params=knn_grid_params, space=space, rescaler=rescaler,
    )

    step_x = max(1, len(g.x1_lat) // quiver_resolution)
    step_y = max(1, len(g.x2_lat) // quiver_resolution)
    xs, ys = np.meshgrid(g.x1_lat[::step_x], g.x2_lat[::step_y])
    u, v = g.gx[::step_y, ::step_x], g.gy[::step_y, ::step_x]
    mag = np.hypot(u, v)
    safe = np.where(mag > 0, mag, 1.0)
    u_n, v_n = u / safe, v / safe
    u_plot, v_plot = (u_n, v_n) if normalize_arrows else (u, v)
    finite = np.isfinite(u_plot) & np.isfinite(v_plot) & (mag > 0)

    if color_by == "angle":
        c = np.arctan2(v_n, u_n)
    elif color_by == "deviation_subtraction":
        ref = np.array([-1.0, 1.0]) / np.sqrt(2.0)
        c = np.degrees(np.arccos(np.clip(u_n * ref[0] + v_n * ref[1], -1.0, 1.0)))
    elif color_by == "magnitude":
        c = mag
    else:
        c = None

    norm = None
    if c is not None and finite.any():
        norm = mpl.colors.Normalize(
            vmin=float(np.nanmin(c[finite])), vmax=float(np.nanmax(c[finite])),
        )

    if zero_dot_threshold > 0 and not normalize_arrows and finite.any():
        ref_mag = float(np.nanmax(mag[finite]))
        near_zero = mag < zero_dot_threshold * ref_mag if ref_mag > 0 else np.zeros_like(mag, dtype=bool)
        arrow_mask = finite & ~near_zero
        dot_mask = finite & near_zero
    else:
        arrow_mask = finite
        dot_mask = np.zeros_like(finite)

    q_kwargs = {
        **dict(
            angles="xy", scale_units="width",
            width=arrow_width, headwidth=4.5, headlength=4.5, headaxislength=4.0,
            pivot="middle", alpha=0.9,
        ),
        **quiver_props,
    }
    if arrow_scale is not None:
        q_kwargs["scale"] = arrow_scale
    elif normalize_arrows:
        q_kwargs.setdefault("scale", 28.0)

    if c is not None:
        q = ax.quiver(xs[arrow_mask], ys[arrow_mask], u_plot[arrow_mask], v_plot[arrow_mask],
                      c[arrow_mask], cmap=cmap, norm=norm, **q_kwargs)
    else:
        q = ax.quiver(xs[arrow_mask], ys[arrow_mask], u_plot[arrow_mask], v_plot[arrow_mask],
                      color=fixed_color, **q_kwargs)

    if dot_mask.any():
        dot_kwargs = {**dict(s=zero_dot_size, linewidths=0, alpha=0.9, marker="o", zorder=q.zorder),
                      **(zero_dot_props or {})}
        if c is not None and "color" not in dot_kwargs and "c" not in dot_kwargs:
            ax.scatter(xs[dot_mask], ys[dot_mask], c=c[dot_mask], cmap=cmap, norm=norm, **dot_kwargs)
        else:
            dot_kwargs.setdefault("color", fixed_color)
            ax.scatter(xs[dot_mask], ys[dot_mask], **dot_kwargs)

    return PlotFunctionResult(
        rendering=q,
        metadata={"gx": g.gx, "gy": g.gy, "x1_lat": g.x1_lat, "x2_lat": g.x2_lat, "space": space},
    )
