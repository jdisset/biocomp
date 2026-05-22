# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Fluorescence density plot (biology-specific)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from collections.abc import Sequence

from biocomp.datautils import LogPolyLogRescaler
from biocomp.plotting.plotting_core import get_bio_color
from jeanplot.plots.density import density_plot_1d
from jeanplot.plots.ticks import get_transformed_ticks_and_labels


def _finite_1d(col: np.ndarray) -> np.ndarray:
    return col[np.isfinite(col)]


def _fmt_val(v: float) -> str:
    av = abs(v)
    if av == 0:
        return "0"
    if av >= 1e6:
        return f"{v:.1e}"
    if av >= 1e3:
        return f"{v / 1e3:.1f}k"
    if av >= 1:
        return f"{v:.0f}"
    return f"{v:.2g}"


def _annotate_channel_stats(
    ax: Axes,
    raw_col: np.ndarray,
    zero_threshold: float,
    color: str,
    lpl_rescaler: LogPolyLogRescaler | None,
):
    col = _finite_1d(raw_col)
    if len(col) == 0:
        return

    below_pct = np.sum(col < zero_threshold) / len(col) * 100
    lines = [
        f"p90 {_fmt_val(np.percentile(col, 90))}",
        f"{below_pct:.0f}% below",
        f"null-point ({_fmt_val(zero_threshold)})",
    ]
    txt = "\n".join(lines)
    ax.text(
        0.5,
        1.01,
        txt,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.5,
        color="#555555",
        fontfamily="monospace",
        linespacing=1.3,
    )

    if lpl_rescaler is not None:
        zt_display = float(lpl_rescaler.fwd(np.array([zero_threshold]))[0])
    else:
        zt_display = zero_threshold
    ax.axhline(
        zt_display,
        color=color,
        alpha=0.4,
        lw=0.6,
        dashes=(6, 6),
        dash_capstyle="round",
        zorder=5,
    )


def _global_below_pct(rawdata: np.ndarray, n_channels: int, zero_threshold: float) -> float:
    below_all = np.ones(rawdata.shape[0], dtype=bool)
    for i in range(n_channels):
        col = rawdata[:, i]
        finite_mask = np.isfinite(col)
        below_all &= finite_mask & (col < zero_threshold)
    return float(np.sum(below_all) / rawdata.shape[0] * 100)


def fluo_densities(
    rawdata: np.ndarray,
    channel_names: list[str],
    ax: list[Axes] | None = None,
    logscale: bool = True,
    res: int = 3000,
    xmin: float | None = None,
    xmax: float | None = None,
    title: str | None = None,
    bw_method: float = 0.01,
    rawdata2: np.ndarray | None = None,
    show_quantiles: Sequence[float] = (0.01, 0.99),
    figsize_per_channel: tuple[float, float] = (1.5, 10),
    lpl_threshold: float = 200,
    lpl_compression: float = 0.4,
    zero_threshold: float = 4500,
    n_inputs: int | None = None,
) -> tuple[Figure | None, list[Axes]]:
    n_channels = len(channel_names)
    assert rawdata.shape[1] >= n_channels, (
        f"rawdata has {rawdata.shape[1]} columns but {n_channels} channel names given"
    )

    own_fig = ax is None
    if own_fig:
        fig, axes = plt.subplots(
            1,
            n_channels,
            figsize=(figsize_per_channel[0] * n_channels, figsize_per_channel[1]),
        )
        if n_channels == 1:
            axes = [axes]
        else:
            axes = list(axes)
    else:
        axes = list(ax)
        fig = axes[0].get_figure()

    lpl_rescaler = LogPolyLogRescaler(
        poly_region_threshold=lpl_threshold, poly_region_coef=lpl_compression
    )

    def _prep_col(col: np.ndarray) -> np.ndarray:
        col = _finite_1d(col)
        return _finite_1d(lpl_rescaler.fwd(col)) if logscale else col

    cols = [_prep_col(rawdata[:, i]) for i in range(n_channels)]
    cols2 = [_prep_col(rawdata2[:, i]) for i in range(n_channels)] if rawdata2 is not None else None

    all_vals = np.concatenate(cols)

    q_lo_colored = float(np.quantile(all_vals, show_quantiles[0]))
    q_hi_colored = float(np.quantile(all_vals, show_quantiles[-1]))
    colored_range = q_hi_colored - q_lo_colored
    pad = 0.05 * colored_range

    lo = xmin if xmin is not None else float(np.quantile(all_vals, 0.0001)) - pad
    hi = xmax if xmax is not None else float(np.quantile(all_vals, 0.9999)) + pad
    sample_at = np.linspace(lo, hi, res)

    if logscale:
        tick_info, label_info = get_transformed_ticks_and_labels([lo, hi], lpl_rescaler)
        ticks = tick_info["major"]
        minor_ticks = []
        ylabels = [lbl for _, lbl in label_info]
    else:
        ticks = np.linspace(lo, hi, 6)
        minor_ticks = []
        ylabels = [f"{v:.1f}" for v in ticks]

    for i, a in enumerate(axes):
        color = get_bio_color(channel_names[i], default="#AAAAAA")
        tlabels = ylabels if i == 0 else None
        x2_col = cols2[i] if cols2 is not None else None
        density_plot_1d(
            cols[i],
            sample_at,
            a,
            color=color,
            label=channel_names[i],
            ticks=ticks,
            minor_ticks=minor_ticks,
            ticks_labels=tlabels,
            bw_method=bw_method,
            x2=x2_col,
            show_quantiles=show_quantiles,
            is_first=(i == 0),
        )
        a.set_ylim(lo, hi)
        _annotate_channel_stats(
            a,
            rawdata[:, i],
            zero_threshold,
            color,
            lpl_rescaler if logscale else None,
        )

    n_total = rawdata.shape[0]
    global_below = _global_below_pct(rawdata[:, :n_channels], n_channels, zero_threshold)
    n_in = n_inputs if n_inputs is not None else n_channels
    above_null_mask = np.ones(n_total, dtype=bool)
    for i in range(n_in):
        col = rawdata[:, i]
        above_null_mask &= np.isfinite(col) & (col >= zero_threshold)
    pct_above = float(np.sum(above_null_mask) / n_total * 100)
    subtitle = (
        f"{n_total} points:\n"
        f"{pct_above:.1f}% above null-point in all input channels\n"
        f"{global_below:.1f}% below null-point in all channels"
    )
    if fig is not None:
        fig.text(
            0.5,
            0.97,
            subtitle,
            ha="center",
            va="top",
            fontsize=7,
            color="#777777",
            fontfamily="monospace",
        )

    if own_fig and fig is not None:
        fig.tight_layout(rect=[0, 0, 1, 0.87])

    return fig, axes
