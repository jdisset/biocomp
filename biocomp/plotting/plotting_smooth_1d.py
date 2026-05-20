# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""1D smooth-curve plotting (KNN-smoothed line + optional head/tail linear fits)."""

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from . import plotting_core as pc
from .plotting_core import (
    DEFAULT_CMAP_NAME,
    build_tree,
    format_powers,
    knn_stats,
    setup_transformed_axis,
)

NdArray = np.ndarray
configurable = pc.configurable


DEFAULT_MARKER_ROTATION: tuple = (
    "o", "x", "s", "^", "*", "v", "+", "<", ">", "d", "p", "P", "h", "H",
)


def make_n_props(n: int, props: dict | list | None) -> list[dict]:
    if props is None:
        props = [{}] * n
    elif not isinstance(props, list):
        props = [props] * n
    if len(props) != n:
        raise ValueError(f"props must have length {n}")
    return props


def _linear_fit_overlay(ax, fit_xq, fit_ym, xd_raw, rescaler, kw):
    x_raw, y_raw = rescaler.inv(fit_xq), rescaler.inv(fit_ym)
    ok = np.isfinite(x_raw) & np.isfinite(y_raw)
    if ok.sum() < 2:
        return None
    a, b = np.polyfit(x_raw[ok], y_raw[ok], 1)
    ax.plot(rescaler.fwd(xd_raw), rescaler.fwd(a * xd_raw + b), **kw)
    return float(a), float(b)


_SLOPE_ANCHOR_DEFAULTS = {
    "head": {"ha": "right", "va": "center", "xytext": (-4, 0), "textcoords": "offset points"},
    "tail": {"ha": "left", "va": "center", "xytext": (4, 0), "textcoords": "offset points"},
}


def _draw_tail_fits(ax, xquery, knn_mean, rescaler, color,
                    head_frac, head_props, tail_frac, tail_props, resolution,
                    slope_fmt=None, slope_props=None):
    # linear fits y_raw = a·x_raw + b on head/tail segments (in raw fluo),
    # extrapolated across the full x range and displayed via rescaler.fwd.
    # returns {"head": (a, b), "tail": (a, b)} for downstream theta annotation.
    fits: dict[str, tuple] = {}
    xq = np.asarray(xquery).reshape(-1)
    ym = np.asarray(knn_mean).reshape(-1)
    finite = np.isfinite(xq) & np.isfinite(ym)
    if finite.sum() < 4:
        return fits
    xq, ym = xq[finite], ym[finite]
    span = xq.max() - xq.min()
    if span <= 0:
        return fits
    fr = rescaler.inv(xq)
    fr = fr[np.isfinite(fr)]
    if fr.size < 2:
        return fits
    lo, hi = float(fr.min()), float(fr.max())
    xd = (np.logspace(np.log10(lo), np.log10(hi), resolution)
          if lo > 0 else np.linspace(lo, hi, resolution))
    base = {"linestyle": "--", "linewidth": 0.5, "alpha": 0.9, "zorder": 3.0, "color": color}
    segments = [
        ("head", head_frac, head_props, lo, lambda f: xq <= xq.min() + f * span),
        ("tail", tail_frac, tail_props, hi, lambda f: xq >= xq.max() - f * span),
    ]
    for tag, frac, props, anchor, sel_fn in segments:
        if not frac or frac <= 0:
            continue
        sel = sel_fn(frac)
        if sel.sum() < 2:
            continue
        ab = _linear_fit_overlay(ax, xq[sel], ym[sel], xd, rescaler, {**base, **(props or {})})
        if ab is None:
            continue
        fits[tag] = ab
        if slope_fmt:
            a, b = ab
            kw = {"xycoords": "data", "fontsize": 7, "color": color,
                  **_SLOPE_ANCHOR_DEFAULTS[tag], **(slope_props or {})}
            ax.annotate(slope_fmt.format(a),
                        (float(rescaler.fwd(anchor)), float(rescaler.fwd(a * anchor + b))),
                        **kw)
    return fits


def _annotate_theta(ax, fits, rescaler, color, props, fmt):
    if "head" not in fits or "tail" not in fits:
        return
    (ah, bh), (at, bt) = fits["head"], fits["tail"]
    if abs(ah - at) < 1e-15:
        return
    # angle between head/tail direction vectors (1, a) in raw fluo space.
    v_h, v_t = np.array([1.0, ah]), np.array([1.0, at])
    cos_t = float(np.clip(np.dot(v_h, v_t) / (np.linalg.norm(v_h) * np.linalg.norm(v_t)), -1.0, 1.0))
    theta_deg = float(np.degrees(np.arccos(cos_t)))
    x_i_raw = (bt - bh) / (ah - at)
    y_i_raw = ah * x_i_raw + bh
    x_i, y_i = float(rescaler.fwd(x_i_raw)), float(rescaler.fwd(y_i_raw))
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if not (xmin <= x_i <= xmax and ymin <= y_i <= ymax):
        return
    kw = {"xycoords": "data", "fontsize": 8, "color": color, "ha": "left", "va": "center"}
    kw.update(props or {})
    ax.annotate(fmt.format(theta_deg), (x_i, y_i), **kw)


@configurable
def smooth_1d(
    X: NdArray,
    Y: NdArray,
    input_names: Sequence[str],
    output_name: str,
    rescaler,
    ax,
    slices: NdArray | None = None,
    title: str | None = None,
    xtitle: str | None = None,
    ytitle: str | None = None,
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
    lineplot_props: list[dict] | dict | None = None,
    errorbar_props: list[dict] | dict | None = None,
    colors: list[Any] | None = None,
    knn_stats_params: dict = None,
    max_centroid_offset_frac: float = 0.0,
    legend_kwargs: dict | None = None,
    head_fit_frac: float = 0.0,
    head_fit_props: list[dict] | dict | None = None,
    tail_fit_frac: float = 0.0,
    tail_fit_props: list[dict] | dict | None = None,
    tail_fit_resolution: int = 50,
    show_theta: bool = True,
    theta_props: dict | None = None,
    theta_fmt: str = "{:.0f}°",
    show_slopes: bool = True,
    slope_props: dict | None = None,
    slope_fmt: str = r"$\alpha={:.2g}$",
):
    if knn_stats_params is None:
        knn_stats_params = {}
    knn_radius = knn_stats_params.get("radius", 0.075)
    knn_stats_params["radius"] = knn_radius
    knn_stats_params.pop("avg_method", None)
    sigma_in_radius = float(knn_stats_params.get("sigma_in_radius", 3.0))
    offset_cutoff = (
        max_centroid_offset_frac * (knn_radius / sigma_in_radius)
        if max_centroid_offset_frac > 0.0
        else None
    )

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
        if slices is None or nslices == 0:
            return
        if slices.ndim != 2 or slices.shape[1] != n_input - 1:
            raise ValueError(f"slices shape must be (nslices, n_input - 1). Got {slices.shape}")

    lineplot_props = make_n_props(nslices, lineplot_props)
    errorbar_props = make_n_props(nslices, errorbar_props)

    if colors is not None:
        assert len(colors) == nslices
    else:
        colors = plt.get_cmap(DEFAULT_CMAP_NAME)(np.linspace(0.25, 1, nslices))
    assert colors is not None

    tree = build_tree(X)

    xmin, xmax = xlims
    xmax = X[:, 0].max() if xmax is None else xmax
    xmin = X[:, 0].min() if xmin is None else xmin

    xquery_max = min(float(xmax), X[:, 0].max() - knn_radius)
    xquery_min = max(float(xmin), X[:, 0].min() + knn_radius * 0.5)

    xquery = np.linspace(xquery_min, xquery_max, res).reshape(-1, 1)

    minz, maxz = np.inf, -np.inf
    slice_fits: list[tuple] = []
    for i in range(nslices):
        query = xquery
        if n_input > 1:
            assert slices is not None and slices.shape[1] == n_input - 1
            query = np.hstack([query, np.tile(slices[i], (query.shape[0], 1))])

        requested = ["mean", "variance"]
        if offset_cutoff is not None:
            requested.append("centroid_offset")
        knn_result = knn_stats(
            query,
            Y,
            tree=tree,
            stats=requested,
            **knn_stats_params,
        )
        if offset_cutoff is not None:
            knn_mean, knn_variance, knn_offset = knn_result
            # knn_mean / knn_variance are (n_query, n_outs); offset is (n_query,) -- reshape to broadcast cleanly.
            boundary = (np.asarray(knn_offset) > offset_cutoff).reshape(-1, 1)
            knn_mean = np.where(boundary, np.nan, knn_mean)
            knn_variance = np.where(boundary, np.nan, knn_variance)
        else:
            knn_mean, knn_variance = knn_result

        minz = min(minz, np.nanmin(knn_mean) if np.isfinite(knn_mean).any() else np.inf)
        maxz = max(maxz, np.nanmax(knn_mean) if np.isfinite(knn_mean).any() else -np.inf)

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

        ax.plot(xquery, knn_mean, **lineplot_props[i])

        if (head_fit_frac and head_fit_frac > 0) or (tail_fit_frac and tail_fit_frac > 0):
            fits = _draw_tail_fits(
                ax, xquery, knn_mean, rescaler, colors[i],
                float(head_fit_frac or 0),
                head_fit_props[i] if isinstance(head_fit_props, list) else head_fit_props,
                float(tail_fit_frac or 0),
                tail_fit_props[i] if isinstance(tail_fit_props, list) else tail_fit_props,
                int(tail_fit_resolution),
                slope_fmt=slope_fmt if show_slopes else None,
                slope_props=slope_props,
            )
            if show_theta:
                slice_fits.append((fits, colors[i]))

        if show_std:
            std = np.sqrt(knn_variance)
            if np.isfinite(std).any():
                minz = min(minz, np.nanmin(knn_mean) - np.nanmax(std))
                maxz = max(maxz, np.nanmax(knn_mean) + np.nanmax(std))

            if std_mode == "errorbar":
                n = len(knn_mean) // n_errorbars
                # shift proportional to i so that errorbars don't overlap
                shift = i * n // nslices
                qxquery = xquery[shift::n].squeeze()
                qz = knn_mean[shift::n].squeeze()
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
                    (knn_mean - std).squeeze(),
                    (knn_mean + std).squeeze(),
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

    for fits, col in slice_fits:
        _annotate_theta(ax, fits, rescaler, col, theta_props, theta_fmt)

    xlabel = input_names[0] if xtitle is None else xtitle
    ylabel = output_name if ytitle is None else ytitle

    if nslices > 1 and show_legend:
        ax.legend(**(legend_kwargs or {}))

    if draw_xlabel and xlabel:
        ax.set_xlabel(xlabel)
    if draw_ylabel and ylabel:
        ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)
