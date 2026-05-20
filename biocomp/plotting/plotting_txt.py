# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
import numpy as np
from typing import Any
from collections.abc import Sequence

from .ascii_heatmap import heatmap_with_labels

NdArray = np.ndarray


def _make_xy_grid(
    xmin: float, xmax: float, ymin: float, ymax: float, xres: int, yres: int
) -> NdArray:
    x = np.linspace(xmin, xmax, xres)
    y = np.linspace(ymin, ymax, yres)
    xx, yy = np.meshgrid(x, y)
    return np.stack([xx.flatten(), yy.flatten()], axis=1)


def _knn_grid(
    x: NdArray,
    y: NdArray,
    xlims: tuple[float, float],
    ylims: tuple[float | None, float | None],
    zslice: NdArray | None = None,
    grid_resolution: int = 100,
    knn_stats_params: dict[str, Any] | None = None,
) -> tuple[NdArray, NdArray]:
    from .plotting_core import knn_stats, build_tree

    if knn_stats_params is None:
        knn_stats_params = {"radius": 0.25, "k": 100, "min_points": 1}

    mask = np.all(np.isfinite(x), axis=1) if x.ndim > 1 else np.isfinite(x)
    mask = mask & (np.all(np.isfinite(y), axis=1) if y.ndim > 1 else np.isfinite(y))

    x_clean = x[mask]
    y_clean = y[mask]

    xmin, xmax = xlims
    ymin, ymax = ylims if ylims[0] is not None else xlims

    if len(x_clean) == 0:
        xy = _make_xy_grid(xmin, xmax, ymin, ymax, grid_resolution, grid_resolution)
        return xy, np.full(xy.shape[0], np.nan)

    xy = _make_xy_grid(xmin, xmax, ymin, ymax, grid_resolution, grid_resolution)
    if x_clean.shape[1] > 2:
        assert zslice is not None
        xquery = np.hstack([xy, np.tile(zslice, (xy.shape[0], 1))])
    else:
        xquery = xy

    tree = build_tree(x_clean)
    output_values, _ = knn_stats(
        xquery, y_clean, tree=tree, stats=["mean", "density"], **knn_stats_params
    )

    return xy, output_values.squeeze()


class TextPlotResult:
    def __init__(self, text: str, title: str = "", metadata: dict[str, Any] | None = None):
        self.text = text
        self.title = title
        self.metadata = metadata or {}

    def __str__(self) -> str:
        return self.text


def smooth_1d_txt(
    X: NdArray,
    Y: NdArray,
    input_names: Sequence[str],
    output_name: str,
    rescaler: Any = None,
    ax: Any = None,
    slices: NdArray | None = None,
    title: str | None = None,
    xtitle: str | None = None,
    ytitle: str | None = None,
    xlims: tuple[float, float | None] = (0, 1),
    vlims: tuple[float, float | None] = (0, None),
    res: int = 80,
    height: int = 20,
    knn_stats_params: dict[str, Any] | None = None,
    **_: Any,
) -> TextPlotResult:
    import plotext as plt
    from .plotting_core import knn_stats, build_tree

    if knn_stats_params is None:
        knn_stats_params = {}
    knn_radius = knn_stats_params.get("radius", 0.075)
    knn_stats_params = {**knn_stats_params, "radius": knn_radius}
    knn_stats_params.pop("avg_method", None)

    nans = np.isnan(X).any(axis=1)
    if nans.any():
        X, Y = X[~nans], Y[~nans]

    if slices is not None:
        slices = np.asarray(slices)
    nslices = 1 if slices is None else slices.shape[0]
    n_input = X.shape[1]

    xmin, xmax = xlims
    xmax = X[:, 0].max() if xmax is None else xmax
    xmin = X[:, 0].min() if xmin is None else xmin

    xquery_max = min(float(xmax), X[:, 0].max() - knn_radius)
    xquery_min = max(float(xmin), X[:, 0].min() + knn_radius * 0.5)
    xquery = np.linspace(xquery_min, xquery_max, res).reshape(-1, 1)

    tree = build_tree(X)

    plt.clear_figure()
    plt.plotsize(res, height)

    for i in range(nslices):
        query = xquery
        if n_input > 1 and slices is not None:
            query = np.hstack([query, np.tile(slices[i], (query.shape[0], 1))])

        knn_mean, _ = knn_stats(query, Y, tree=tree, stats=["mean", "variance"], **knn_stats_params)

        label = ""
        if n_input > 1 and slices is not None:
            for j in range(n_input - 1):
                label += f"X{j + 2}≈{slices[i][j]:.2f}"
                if j < n_input - 2:
                    label += ", "

        plt.plot(xquery.squeeze(), knn_mean.squeeze(), label=label if label else None)

    xlabel = input_names[0] if xtitle is None else xtitle
    ylabel = output_name if ytitle is None else ytitle

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)

    if nslices > 1:
        plt.theme("pro")

    result = plt.build()
    return TextPlotResult(result, title=title or "")


def smooth_2d_txt(
    X: NdArray,
    Y: NdArray,
    input_names: Sequence[str],
    output_name: str,
    rescaler: Any = None,
    ax: Any = None,
    zslice: NdArray | None = None,
    title: str | None = None,
    xtitle: str | None = None,
    ytitle: str | None = None,
    vtitle: str | None = None,
    xlims: tuple[float | None, float | None] = (0, 1),
    ylims: tuple[float | None, float | None] = (None, None),
    vlims: tuple[float | None, float | None] = (None, None),
    xres: int = 64,
    yres: int = 32,
    knn_grid_params: dict[str, Any] | None = None,
    **_: Any,
) -> TextPlotResult:
    if knn_grid_params is None:
        knn_grid_params = {}

    data_xlims = [X[:, 0].min(), X[:, 0].max()]
    data_ylims = [X[:, 1].min(), X[:, 1].max()]

    xlims = [
        data_xlims[0] if xlims[0] is None else xlims[0],
        data_xlims[1] if xlims[1] is None else xlims[1],
    ]
    ylims = [
        data_ylims[0] if ylims[0] is None else ylims[0],
        data_ylims[1] if ylims[1] is None else ylims[1],
    ]

    finite_mask = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y), axis=1)
    if not np.all(finite_mask):
        X, Y = X[finite_mask], Y[finite_mask]

    zslice = np.asarray(zslice) if zslice is not None else None

    input_coords, output_values = _knn_grid(
        X,
        Y,
        xlims,
        ylims,
        zslice=zslice,
        grid_resolution=max(xres, yres),
        knn_stats_params=knn_grid_params.get("knn_stats_params", {}),
    )

    grid_size = int(np.sqrt(len(output_values)))
    grid_data = output_values.reshape(grid_size, grid_size)
    grid_data = np.flipud(grid_data)

    vmin = vlims[0] if vlims[0] is not None else float(np.nanmin(output_values))
    vmax = vlims[1] if vlims[1] is not None else float(np.nanmax(output_values))

    xlabel = input_names[0] if xtitle is None else xtitle
    ylabel = input_names[1] if ytitle is None else ytitle

    text = heatmap_with_labels(
        grid_data,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        vmin=vmin,
        vmax=vmax,
        xres=xres,
        yres=yres,
        show_colorbar=True,
    )

    return TextPlotResult(text, title=title or "", metadata={"vmin": vmin, "vmax": vmax})


def smooth_3d_txt(
    X: NdArray,
    Y: NdArray,
    input_names: Sequence[str],
    output_name: str,
    rescaler: Any = None,
    ax: Any = None,
    zslices: NdArray | None = None,
    xlims: tuple[float | None, float | None] = (0, 1),
    ylims: tuple[float | None, float | None] = (None, None),
    zlims: tuple[float | None, float | None] = (None, None),
    vlims: tuple[float | None, float | None] = (None, None),
    xres: int = 48,
    yres: int = 24,
    xtitle: str | None = None,
    ytitle: str | None = None,
    ztitle: str | None = None,
    title: str | None = None,
    smooth_2d_params: dict[str, Any] | None = None,
    knn_grid_params: dict[str, Any] | None = None,
    **_: Any,
) -> TextPlotResult:
    if smooth_2d_params is None:
        smooth_2d_params = {}
    if knn_grid_params is None:
        knn_grid_params = {}

    ylims = xlims if ylims == (None, None) else ylims
    zlims = xlims if zlims == (None, None) else zlims

    if zslices is None:
        zslices = np.array([[0.25, 0.5, 0.75]])
    zslices = np.atleast_2d(zslices)

    all_slice_results = []
    global_vmin, global_vmax = np.inf, -np.inf

    explicit_keys = {
        "xlims",
        "ylims",
        "vlims",
        "xres",
        "yres",
        "xtitle",
        "ytitle",
        "knn_grid_params",
    }
    s2d_params = {k: v for k, v in smooth_2d_params.items() if k not in explicit_keys}

    for slice_group in zslices:
        slice_positions = np.atleast_1d(slice_group)

        for z_pos in slice_positions:
            zslice = np.atleast_1d(z_pos)
            result = smooth_2d_txt(
                X,
                Y,
                input_names[:2],
                output_name,
                rescaler,
                ax=None,
                zslice=zslice,
                xlims=xlims,
                ylims=ylims,
                vlims=vlims,
                xres=xres,
                yres=yres,
                xtitle=xtitle,
                ytitle=ytitle,
                knn_grid_params=knn_grid_params,
                **s2d_params,
            )
            all_slice_results.append((z_pos, result))
            if "vmin" in result.metadata:
                global_vmin = min(global_vmin, result.metadata["vmin"])
                global_vmax = max(global_vmax, result.metadata["vmax"])

    lines = []
    if title:
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")

    z_label = ztitle if ztitle is not None else (input_names[2] if len(input_names) > 2 else "z")

    for z_pos, result in all_slice_results:
        z_display = f"{z_label} = {z_pos:.3f}"
        lines.append(f"─── {z_display} {'─' * (xres - len(z_display) - 5)}")
        lines.append(result.text)
        lines.append("")

    text = "\n".join(lines)
    return TextPlotResult(
        text, title=title or "", metadata={"vmin": global_vmin, "vmax": global_vmax}
    )


TXT_PLOT_FUNCTION_MAP = {
    "biocomp.plotting.plotting_smooth.smooth_1d": smooth_1d_txt,
    "biocomp.plotting.plotting_smooth.smooth_2d": smooth_2d_txt,
    "biocomp.plotting.plotting_3d.smooth_3d": smooth_3d_txt,
    "biocomp.plotutils.smooth": None,
}


def get_txt_plot_function(original_func_name: str):
    if original_func_name in TXT_PLOT_FUNCTION_MAP:
        return TXT_PLOT_FUNCTION_MAP[original_func_name]

    for key, func in TXT_PLOT_FUNCTION_MAP.items():
        if original_func_name.endswith(key.split(".")[-1]):
            return func
    return None
