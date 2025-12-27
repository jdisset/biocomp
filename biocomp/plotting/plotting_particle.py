# {{{                          --     imports     --
# ···············································································

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_rgba
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator, LogLocator

from . import plotting_core as pc
from .plotting_core import setup_transformed_axis

configurable = pc.configurable

##────────────────────────────────────────────────────────────────────────────}}}


### {{{                   --     helper functions     --


def _resolve_colors(colors, n, cycle=True, cmap_n=8):
    if colors is None:
        colors = "bc_multi"
    if isinstance(colors, str):
        cmap = plt.get_cmap(colors)
        if cycle:
            c = np.linspace(0, 1, cmap_n)
            return ([cmap(ci) for ci in c] * (n // cmap_n + 1))[:n]
        return [cmap(i / max(n - 1, 1)) for i in range(n)]
    return [colors[i % len(colors)] for i in range(n)]


def _get_ticks(y_min, y_max, n_major, n_minor_per_major, is_log):
    if is_log:
        major = LogLocator(base=10, numticks=n_major)
        minor = LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100)
        major_ticks = [t for t in major.tick_values(y_min, y_max) if y_min <= t <= y_max]
        minor_ticks = [
            t
            for t in minor.tick_values(y_min, y_max)
            if y_min <= t <= y_max and t not in major_ticks
        ]
    else:
        major = MaxNLocator(nbins=n_major, steps=[1, 2, 2.5, 5, 10])
        major_ticks = [t for t in major.tick_values(y_min, y_max) if y_min <= t <= y_max]
        minor_ticks = []
        if len(major_ticks) >= 2:
            step = major_ticks[1] - major_ticks[0]
            minor_step = step / (n_minor_per_major + 1)
            t = major_ticks[0] - step
            while t <= y_max + step:
                for j in range(1, n_minor_per_major + 1):
                    mt = t + j * minor_step
                    if y_min <= mt <= y_max and mt not in major_ticks:
                        minor_ticks.append(mt)
                t += step
    return major_ticks, minor_ticks


##────────────────────────────────────────────────────────────────────────────}}}


### {{{                   --     particle_plot     --


@configurable
def particle_plot(
    ax: plt.Axes,
    data: np.ndarray,
    value_names: list[str],
    colors=None,
    rescaler=None,
    ylims=None,
    max_line_extend: int = 40,
    value_spacing: float = 25.0,
    derivative=None,
    line_params: dict = {},
    dot_params: dict = {},
    arrow_params: dict = {},
    vaxis_params: dict = {},
    label_params: dict = {},
    setup_yaxis_params: dict = {},
) -> None:
    """
    Particle plot showing current values, history trails, and trend arrows.

    Args:
        ax: matplotlib axes
        data: (num_variables, num_timepoints) array
        value_names: labels for each variable
        colors: colormap name or list of colors (default: bc_multi)
        rescaler: biocomp DataRescaler for axis transformation
        ylims: (min, max) y-axis limits
        max_line_extend: historical ticks to show
        value_spacing: x-axis spacing between columns
        derivative: explicit derivatives, or None to compute from data
        line_params: {width, style, oldest_alpha, color, halo, halo_width, halo_cmap}
        dot_params: {size}
        arrow_params: {scale, cmap, size, threshold}
        vaxis_params: {color, linewidth, major_tick_length, minor_tick_length,
                       n_major_ticks, n_minor_per_major}
        label_params: {show, rotation, line_color}
        setup_yaxis_params: passed to setup_transformed_axis
    """
    n_vars, n_time = data.shape
    var_colors = _resolve_colors(colors, n_vars)

    lp = {
        "width": 0.5,
        "style": "--",
        "oldest_alpha": 0.2,
        "color": "#ffffffaa",
        "halo": True,
        "halo_width": 4.0,
        "halo_cmap": "bc_blues",
        **line_params,
    }
    dp = {"size": 10, **dot_params}
    ap = {"scale": 8.0, "cmap": "bc_grrd_r", "size": 0.2, "threshold": 0.01, **arrow_params}
    vp = {
        "color": "#ccccccaa",
        "linewidth": 0.7,
        "major_tick_length": 0.1,
        "minor_tick_length": 0.05,
        "n_major_ticks": 6,
        "n_minor_per_major": 4,
        **vaxis_params,
    }
    labp = {"show": True, "rotation": 45, "line_color": "#444444", **label_params}

    if derivative is None:
        derivative = data[:, -1] - data[:, -2] if n_time >= 2 else np.zeros(n_vars)
    derivative = np.nan_to_num(derivative, nan=0.0, posinf=0.0, neginf=0.0)

    x_pos = np.arange(n_vars) * value_spacing
    current = data[:, -1]
    is_log = ax.get_yscale() == "log"

    try:
        cmap = plt.get_cmap(ap["cmap"])
    except ValueError:
        cmap = plt.get_cmap("RdBu_r")
    deriv_max = max(np.nanmax(np.abs(derivative)), 1e-10)
    norm = Normalize(vmin=-deriv_max, vmax=deriv_max)

    # use nanmin/nanmax to handle NaN values in data
    finite_data = data[np.isfinite(data)]
    if finite_data.size == 0:
        # all NaN/Inf - use sensible defaults
        y_min_d, y_max_d = 0.0, 1.0
    else:
        y_min_d, y_max_d = finite_data.min(), finite_data.max()
        if y_min_d == y_max_d:
            # prevent singular transform
            y_min_d, y_max_d = y_min_d - 0.5, y_max_d + 0.5

    if ylims:
        y_min, y_max = ylims
    elif is_log:
        if y_min_d <= 0:
            y_min_d = 1e-10  # prevent log of non-positive
        if y_max_d <= 0:
            y_max_d = 1.0
        log_range = np.log10(y_max_d) - np.log10(max(y_min_d, 1e-10))
        y_min, y_max = y_min_d / (10 ** (0.1 * log_range)), y_max_d * (10 ** (0.15 * log_range))
    else:
        y_range = y_max_d - y_min_d
        y_min, y_max = y_min_d - 0.1 * y_range, y_max_d + 0.15 * y_range

    major_ticks, minor_ticks = _get_ticks(
        y_min, y_max, vp["n_major_ticks"], vp["n_minor_per_major"], is_log
    )

    # batch vertical axes
    vline_segs = [[(x, y_min), (x, y_max)] for x in x_pos]
    ax.add_collection(
        LineCollection(vline_segs, colors=vp["color"], linewidths=vp["linewidth"], zorder=0)
    )

    # batch tick marks (scale lengths relative to spacing)
    major_len = vp["major_tick_length"] * value_spacing
    minor_len = vp["minor_tick_length"] * value_spacing
    tick_segs, tick_lws = [], []
    for x in x_pos:
        for yt in major_ticks:
            tick_segs.append([(x - major_len, yt), (x + major_len, yt)])
            tick_lws.append(vp["linewidth"] * 1.5)
        for yt in minor_ticks:
            tick_segs.append([(x - minor_len, yt), (x + minor_len, yt)])
            tick_lws.append(vp["linewidth"])
    if tick_segs:
        ax.add_collection(
            LineCollection(tick_segs, colors=vp["color"], linewidths=tick_lws, zorder=0)
        )

    n_history = min(max_line_extend, n_time)
    if n_history >= 2:
        halo_cmap = plt.get_cmap(lp["halo_cmap"]) if lp["halo"] else None
        alphas = np.linspace(lp["oldest_alpha"], 1.0, n_history - 1)
        cmap_vals = np.linspace(0, 1, n_history - 1)
        line_c = to_rgba(lp["color"])
        line_colors = np.array([[*line_c[:3], a * line_c[3]] for a in alphas])
        halo_colors = (
            np.array([[*halo_cmap(cv)[:3], a] for cv, a in zip(cmap_vals, alphas)])
            if lp["halo"]
            else None
        )
        x_offsets = np.arange(n_history)[::-1]

        all_halo_segs, all_line_segs = [], []
        for i in range(n_vars):
            hist = data[i, -n_history:]
            x_coords = x_pos[i] - x_offsets
            points = np.column_stack([x_coords, hist]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            all_line_segs.append(segments)
            if lp["halo"]:
                all_halo_segs.append(segments)

        if lp["halo"] and all_halo_segs:
            all_halo_segs = np.concatenate(all_halo_segs)
            all_halo_colors = np.tile(halo_colors, (n_vars, 1))
            ax.add_collection(
                LineCollection(
                    all_halo_segs,
                    colors=all_halo_colors,
                    linewidths=lp["halo_width"],
                    linestyle="-",
                    zorder=3,
                    capstyle="round",
                )
            )

        all_line_segs = np.concatenate(all_line_segs)
        all_line_colors = np.tile(line_colors, (n_vars, 1))
        ax.add_collection(
            LineCollection(
                all_line_segs,
                colors=all_line_colors,
                linewidths=lp["width"],
                linestyle=lp["style"],
                zorder=4,
                capstyle="round",
            )
        )

        dot_color = halo_cmap(1.0) if halo_cmap else var_colors[0]
        ax.scatter(x_pos, current, s=dp["size"], c=[dot_color] * n_vars, zorder=5)
    else:
        ax.scatter(x_pos, current, s=dp["size"], c=var_colors, zorder=5)

    for i in range(n_vars):
        d = derivative[i]
        if np.abs(d) < ap["threshold"]:
            continue
        x, y = x_pos[i], current[i]
        if is_log:
            rel = d / y if y != 0 else 0
            if np.abs(rel) < ap["threshold"]:
                continue
            y_end = y * np.exp(ap["scale"] * rel)
        else:
            y_end = y + np.sign(d) * np.abs(d) * ap["scale"]
        ax.annotate(
            "",
            xy=(x, y_end),
            xytext=(x, y),
            zorder=4.5,
            arrowprops=dict(
                arrowstyle=f"-|>,head_length={ap['size']},head_width={ap['size'] * 0.7}",
                color=cmap(norm(d)),
                lw=lp["width"] * 1.5,
            ),
        )

    if labp["show"]:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(value_names, rotation=labp["rotation"], ha="center")

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_pos[0] - max_line_extend - 1, x_pos[-1] + 1)

    if rescaler is not None:
        setup_transformed_axis(
            ax, yaxis_lims=(y_min, y_max), rescaler=rescaler, setup_yaxis_params=setup_yaxis_params
        )
    else:
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        ax.tick_params(axis="y", which="major", length=5, width=0.8)
        ax.tick_params(axis="y", which="minor", length=2.5, width=0.5)

    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)

    seg_hw = value_spacing * 0.4
    xspine_segs = [[(x - seg_hw, y_min), (x + seg_hw, y_min)] for x in x_pos]
    ax.add_collection(
        LineCollection(
            xspine_segs, colors=labp["line_color"], linewidths=0.8, zorder=10, clip_on=False
        )
    )
    ax.tick_params(axis="x", length=0)


##────────────────────────────────────────────────────────────────────────────}}}
