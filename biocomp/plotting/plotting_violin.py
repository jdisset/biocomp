# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Voxel-conditioned violin plotting (smooth slice-style distribution display)."""

from typing import Literal

import numpy as np
from scipy.optimize import minimize

from biocomp.plotutils import make_xy_grid

from . import plotting_core as pc
from .plotting_core import (
    build_tree,
    knn_stats,
    setup_transformed_axis,
    weighted_kde_1d,
)

configurable = pc.configurable


def _as_xy_arrays(X, Y):
    x = np.asarray(X, dtype=float)
    y = np.asarray(Y, dtype=float)

    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]

    if x.ndim != 2:
        raise ValueError(f"X must be 1D or 2D, got shape {x.shape}")
    if y.ndim != 2:
        raise ValueError(f"Y must be 1D or 2D, got shape {y.shape}")
    if y.shape[1] != 1:
        raise ValueError(f"Only single-output Y is supported for this plot, got shape {y.shape}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and Y must have the same number of rows, got {x.shape[0]} and {y.shape[0]}"
        )

    finite = np.isfinite(x).all(axis=1) & np.isfinite(y[:, 0])
    x = x[finite]
    y = y[finite, 0]
    if len(x) == 0:
        raise ValueError("No finite points remain after filtering")

    return x, y


def _make_voxel_query_grid(X, grid_resolution=8, max_voxels=None):
    dim = X.shape[1]
    if isinstance(grid_resolution, int):
        res = [int(grid_resolution)] * dim
    else:
        res = [int(v) for v in grid_resolution]
        if len(res) != dim:
            raise ValueError(f"grid_resolution length must match input dim {dim}, got {len(res)}")

    # Guard against combinatorial explosion for higher-dimensional inputs.
    if max_voxels is not None:
        total = int(np.prod(res))
        if total > int(max_voxels):
            capped = max(2, int(np.floor(max_voxels ** (1.0 / dim))))
            res = [capped] * dim

    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    if dim == 1:
        return np.linspace(mins[0], maxs[0], res[0])[:, None]
    if dim == 2:
        return make_xy_grid(mins[0], maxs[0], xres=res[0], ymin=mins[1], ymax=maxs[1], yres=res[1])

    axes = [np.linspace(mins[i], maxs[i], res[i]) for i in range(dim)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([m.reshape(-1) for m in mesh])


def _compute_voxel_distributions(
    X,
    Y,
    query_points,
    *,
    knn_stats_params,
):
    tree = build_tree(X, use_jax=False)
    y_mean, iw = knn_stats(
        query_points,
        y=Y[:, None],
        tree=tree,
        stats=["mean", "iw"],
        use_jax=False,
        **knn_stats_params,
    )

    y_mean = np.asarray(y_mean).reshape(-1)
    idx, w = iw
    idx = np.asarray(idx)
    w = np.asarray(w)

    voxel_means = []
    voxel_values = []
    voxel_weights = []
    voxel_counts = []

    for i in range(len(query_points)):
        valid = np.isfinite(w[i]) & (w[i] > 0)
        if not np.any(valid):
            continue

        ii = idx[i, valid].astype(int)
        ww = w[i, valid].astype(float)
        wsum = float(np.sum(ww))
        if wsum <= 0:
            continue
        ww = ww / wsum
        vv = Y[ii]

        ym = float(y_mean[i])
        if not np.isfinite(ym):
            continue
        if not np.isfinite(vv).any():
            continue

        voxel_means.append(ym)
        voxel_values.append(vv)
        voxel_weights.append(ww)
        voxel_counts.append(int(valid.sum()))

    means_arr = np.asarray(voxel_means, dtype=float)
    means_tree = build_tree(means_arr[:, None], use_jax=False) if len(means_arr) > 0 else None

    return {
        "means": means_arr,
        "means_tree": means_tree,
        "values": voxel_values,
        "weights": voxel_weights,
        "counts": np.asarray(voxel_counts, dtype=float),
    }


def _tick_aggregation_from_voxels(
    voxel_data,
    tick,
    *,
    tick_knn_stats_params,
    voxel_weight_mode,
):
    means = voxel_data["means"]
    if len(means) == 0:
        return None

    means_tree = voxel_data.get("means_tree")
    if means_tree is None:
        return None

    idx, w = knn_stats(
        np.asarray([[float(tick)]], dtype=float),
        tree=means_tree,
        stats="iw",
        use_jax=False,
        **tick_knn_stats_params,
    )
    idx = np.asarray(idx[0], dtype=int)
    w = np.asarray(w[0], dtype=float)
    valid = np.isfinite(w) & (w > 0)
    if not np.any(valid):
        return None

    idx = idx[valid]
    tick_weights = w[valid]
    if voxel_weight_mode == "count":
        tick_weights = tick_weights * voxel_data["counts"][idx]

    wsum = float(np.sum(tick_weights))
    if wsum <= 0:
        return None
    tick_weights = tick_weights / wsum

    vals = []
    wts = []
    for i, tw in zip(idx, tick_weights, strict=False):
        vals.append(voxel_data["values"][i])
        wts.append(voxel_data["weights"][i] * tw)

    values = np.concatenate(vals)
    weights = np.concatenate(wts)
    wsum2 = float(np.sum(weights))
    if not np.isfinite(wsum2) or wsum2 <= 0:
        return None
    effective_mean = float(np.sum(values * weights) / wsum2)
    return {
        "values": values,
        "weights": weights,
        "effective_mean": effective_mean,
    }


def _weighted_mean_and_median(values, weights):
    v = np.asarray(values, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    valid = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if valid.sum() < 1:
        return None, None

    v = v[valid]
    w = w[valid]
    wsum = float(np.sum(w))
    if wsum <= 0:
        return None, None
    w = w / wsum

    mean = float(np.sum(v * w))
    order = np.argsort(v)
    vs = v[order]
    ws = w[order]
    cdf = np.cumsum(ws)
    median = float(vs[np.searchsorted(cdf, 0.5, side="left")])
    return mean, median


def _effective_map_from_voxels(
    voxel_data,
    *,
    tick_knn_stats_params,
    voxel_weight_mode,
    q_bounds,
    nq=400,
):
    q0, q1 = float(q_bounds[0]), float(q_bounds[1])
    if not np.isfinite(q0) or not np.isfinite(q1) or q1 <= q0:
        return None, None

    q_dense = np.linspace(q0, q1, int(nq))
    c_dense = np.full_like(q_dense, np.nan, dtype=float)
    for i, q in enumerate(q_dense):
        agg = _tick_aggregation_from_voxels(
            voxel_data,
            float(q),
            tick_knn_stats_params=tick_knn_stats_params,
            voxel_weight_mode=voxel_weight_mode,
        )
        if agg is not None and np.isfinite(agg["effective_mean"]):
            c_dense[i] = float(agg["effective_mean"])

    valid = np.isfinite(c_dense)
    if valid.sum() < 2:
        return None, None

    return q_dense[valid], c_dense[valid]


def _query_ticks_for_effective_centers(
    voxel_data,
    desired_centers,
    *,
    tick_knn_stats_params,
    voxel_weight_mode,
    q_bounds,
    nq=400,
):
    q_map, c_map = _effective_map_from_voxels(
        voxel_data,
        tick_knn_stats_params=tick_knn_stats_params,
        voxel_weight_mode=voxel_weight_mode,
        q_bounds=q_bounds,
        nq=nq,
    )
    if q_map is None or c_map is None:
        return np.full_like(np.asarray(desired_centers, dtype=float), np.nan, dtype=float)

    c = c_map
    q = q_map

    order = np.argsort(c)
    c = c[order]
    q = q[order]
    c_unique, ix = np.unique(c, return_index=True)
    q_unique = q[ix]

    if len(c_unique) < 2:
        return np.full_like(np.asarray(desired_centers, dtype=float), np.nan, dtype=float)

    desired = np.asarray(desired_centers, dtype=float)
    return np.interp(desired, c_unique, q_unique, left=q_unique[0], right=q_unique[-1])


def _query_ticks_for_effective_clusters(
    voxel_data,
    desired_centers,
    *,
    tick_knn_stats_params,
    voxel_weight_mode,
    q_bounds,
    nq=400,
    spacing_weight=0.15,
    smooth_weight=0.05,
    n_restarts=3,
    maxiter=200,
):
    desired = np.asarray(desired_centers, dtype=float)
    n = len(desired)
    if n == 0:
        return desired.copy()

    q0_in, q1_in = float(q_bounds[0]), float(q_bounds[1])
    if not np.isfinite(q0_in) or not np.isfinite(q1_in) or q1_in <= q0_in:
        return np.full_like(desired, np.nan, dtype=float)

    q_map, m_map = _effective_map_from_voxels(
        voxel_data,
        tick_knn_stats_params=tick_knn_stats_params,
        voxel_weight_mode=voxel_weight_mode,
        q_bounds=q_bounds,
        nq=nq,
    )
    if q_map is None or m_map is None:
        return np.full_like(desired, np.nan, dtype=float)

    q0 = float(np.min(q_map))
    q1 = float(np.max(q_map))
    if not np.isfinite(q0) or not np.isfinite(q1) or q1 <= q0:
        return np.full_like(desired, np.nan, dtype=float)

    eps = max(1e-8, (q1 - q0) * 1e-6)

    def _q_to_logits(q):
        gaps = np.diff(np.concatenate([[q0], q, [q1]]))
        gaps = np.maximum(gaps, eps)
        v = np.log(gaps)
        return v - np.mean(v)

    def _logits_to_q(v):
        z = v - np.max(v)
        g = np.exp(np.clip(z, -60, 60))
        g = g / np.sum(g)
        c = np.cumsum(g)
        return q0 + (q1 - q0) * c[:n]

    m_lo, m_hi = float(np.min(m_map)), float(np.max(m_map))
    if not np.isfinite(m_lo) or not np.isfinite(m_hi) or m_hi <= m_lo:
        return np.full_like(desired, np.nan, dtype=float)

    # Map requested centers into reachable effective-mean support without hard clipping
    # collapse at the boundaries.
    if n == 1:
        target = np.asarray([0.5 * (m_lo + m_hi)], dtype=float)
    else:
        d_lo = float(np.min(desired))
        d_hi = float(np.max(desired))
        if np.isfinite(d_lo) and np.isfinite(d_hi) and d_hi > d_lo:
            u = (desired - d_lo) / (d_hi - d_lo)
        else:
            u = (np.arange(n, dtype=float) + 0.5) / n
        u = np.clip(u, 0.0, 1.0)
        target = m_lo + u * (m_hi - m_lo)

    q_init = _query_ticks_for_effective_centers(
        voxel_data,
        target,
        tick_knn_stats_params=tick_knn_stats_params,
        voxel_weight_mode=voxel_weight_mode,
        q_bounds=q_bounds,
        nq=nq,
    )
    if not np.isfinite(q_init).all():
        q_init = np.linspace(q0, q1, n)

    q_init = np.clip(q_init, q0 + eps, q1 - eps)
    q_init = np.maximum.accumulate(q_init)
    for i in range(1, n):
        if q_init[i] <= q_init[i - 1]:
            q_init[i] = min(q1 - eps, q_init[i - 1] + eps)

    if q_init[-1] >= q1:
        q_init = np.linspace(q0 + eps, q1 - eps, n)

    v0 = _q_to_logits(q_init)
    gap_target = (q1 - q0) / (n + 1)

    def objective(v):
        q = _logits_to_q(v)
        m = np.interp(q, q_map, m_map, left=m_map[0], right=m_map[-1])
        data_term = float(np.mean((m - target) ** 2))

        gaps = np.diff(np.concatenate([[q0], q, [q1]]))
        spacing_term = float(np.mean((gaps - gap_target) ** 2))

        if n >= 3:
            curv = np.diff(q, n=2)
            smooth_term = float(np.mean(curv**2))
        else:
            smooth_term = 0.0

        return data_term + float(spacing_weight) * spacing_term + float(smooth_weight) * smooth_term

    best_v = v0.copy()
    best_obj = objective(best_v)

    rng = np.random.default_rng(0)
    n_restarts = max(1, int(n_restarts))
    for r in range(n_restarts):
        if r == 0:
            v_start = v0
        else:
            v_start = v0 + rng.normal(0.0, 0.25, size=v0.shape)
        try:
            res = minimize(
                objective,
                v_start,
                method="L-BFGS-B",
                options={"maxiter": int(maxiter)},
            )
            v_try = res.x if np.isfinite(res.fun) else v_start
        except Exception:
            v_try = v_start

        obj = objective(v_try)
        if np.isfinite(obj) and obj < best_obj:
            best_obj = obj
            best_v = v_try

    return _logits_to_q(best_v)


@configurable
def smooth_voxel_conditioned_violin(
    X,
    Y,
    input_names,
    output_name,
    rescaler,
    ax,
    mode: Literal["single", "split"] = "single",
    title: str | None = None,
    xtitle: str | None = None,
    ytitle: str | None = None,
    xlims=(0.0, 0.7),
    ylims=(0.0, 0.7),
    draw_xlabel=True,
    draw_ylabel=True,
    grid_resolution=64,
    max_voxels=None,
    tick_values=None,
    tick_count=6,
    tick_sigma=0.05,
    cluster_nquery=400,
    cluster_spacing_weight=0.15,
    cluster_smooth_weight=0.05,
    cluster_restarts=3,
    cluster_maxiter=200,
    tick_knn_stats_params=None,
    tick_line=True,
    tick_line_color="#7f7f7f",
    tick_line_alpha=0.35,
    tick_line_width=0.6,
    knn_stats_params=None,
    kde_points=600,
    kde_bw_method=None,
    violin_width=0.035,
    violin_alpha=0.2,
    violin_line_width=0.8,
    show_tick_stats=True,
    tick_stat_bar_frac=0.65,
    tick_stat_line_width=1.1,
    tick_stat_mean_marker_size=18,
    tick_stat_alpha=0.9,
    show_marginal_kde=True,
    marginal_size="12%",
    marginal_pad=0.0,
    marginal_line_width=1.0,
    marginal_fill_alpha=0.2,
    marginal_normalize=True,
    marginal_kde_pad_frac=0.15,
    title_y_with_marginal=1.16,
    show_identity_line=True,
    identity_line_color="#7f7f7f",
    identity_line_style="--",
    identity_line_width=0.9,
    identity_line_alpha=0.8,
    single_color="#222222",
    split_left_color="#222222",
    split_right_color="#777777",
    voxel_weight_mode: Literal["equal", "count"] = "equal",
):
    """Smooth voxel-conditioned violin plot.

    - ``mode='single'``: one source (full violins).
    - ``mode='split'``: two independent sources (left/right half violins).
    - Tick placement is cluster-optimized in effective-mean space (single SSOT mode).
    - By default (`tick_values=None`), ticks are centered in equal x-intervals:
      ``x_i = x0 + (i + 0.5) * (x1 - x0) / tick_count``.
    - Optional outside marginals:
      x-axis: KDE of local mean levels, y-axis: KDE of full output distribution.

    In split mode, pass ``X=(X_left, X_right)``, ``Y=(Y_left, Y_right)``.
    """
    if knn_stats_params is None:
        knn_stats_params = {}
    if tick_knn_stats_params is None:
        tick_knn_stats_params = {
            "k": 1000,
            "radius": 3.0 * float(tick_sigma),
            "min_points": 20,
        }
    else:
        tick_knn_stats_params = dict(tick_knn_stats_params)

    def _stats_params_for(x):
        p = dict(knn_stats_params)
        p.setdefault("k", 1000)
        p.setdefault("radius", 0.1)
        p.setdefault("min_points", 20)
        return p

    if mode == "single":
        x0, y0 = _as_xy_arrays(X, Y)
        q0 = _make_voxel_query_grid(x0, grid_resolution=grid_resolution, max_voxels=max_voxels)
        vox0 = _compute_voxel_distributions(
            x0,
            y0,
            q0,
            knn_stats_params=_stats_params_for(x0),
        )
        sources = [("single", vox0, single_color, y0)]
    elif mode == "split":
        if not isinstance(X, list | tuple) or not isinstance(Y, list | tuple):
            raise ValueError("mode='split' expects X and Y to be 2-tuples/lists")
        if len(X) != 2 or len(Y) != 2:
            raise ValueError("mode='split' expects exactly two sources")

        x_l, y_l = _as_xy_arrays(X[0], Y[0])
        x_r, y_r = _as_xy_arrays(X[1], Y[1])
        q_l = _make_voxel_query_grid(x_l, grid_resolution=grid_resolution, max_voxels=max_voxels)
        q_r = _make_voxel_query_grid(x_r, grid_resolution=grid_resolution, max_voxels=max_voxels)
        vox_l = _compute_voxel_distributions(
            x_l,
            y_l,
            q_l,
            knn_stats_params=_stats_params_for(x_l),
        )
        vox_r = _compute_voxel_distributions(
            x_r,
            y_r,
            q_r,
            knn_stats_params=_stats_params_for(x_r),
        )
        sources = [("left", vox_l, split_left_color, y_l), ("right", vox_r, split_right_color, y_r)]
    else:
        raise ValueError(f"Unknown mode {mode!r}. Expected 'single' or 'split'.")

    all_means = np.concatenate([src[1]["means"] for src in sources if len(src[1]["means"]) > 0])
    if len(all_means) == 0:
        raise ValueError("No valid voxel means found for tick generation")
    means_lo, means_hi = float(all_means.min()), float(all_means.max())

    t0 = means_lo if xlims[0] is None else float(xlims[0])
    t1 = means_hi if xlims[1] is None else float(xlims[1])

    if tick_values is None:
        n_ticks = int(tick_count)
        if n_ticks < 1:
            raise ValueError(f"tick_count must be >= 1, got {tick_count}")
        if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
            raise ValueError(f"Invalid tick range [{t0}, {t1}]")
        step = (t1 - t0) / n_ticks
        base_ticks = t0 + (np.arange(n_ticks, dtype=float) + 0.5) * step
    else:
        base_ticks = np.asarray(tick_values, dtype=float).ravel()

    query_ticks_by_source = {}
    for name, vdata, _color, _yraw in sources:
        query_ticks_by_source[name] = _query_ticks_for_effective_clusters(
            vdata,
            base_ticks,
            tick_knn_stats_params=tick_knn_stats_params,
            voxel_weight_mode=voxel_weight_mode,
            q_bounds=(t0, t1),
            nq=int(cluster_nquery),
            spacing_weight=float(cluster_spacing_weight),
            smooth_weight=float(cluster_smooth_weight),
            n_restarts=int(cluster_restarts),
            maxiter=int(cluster_maxiter),
        )
    display_ticks = base_ticks

    line_ymin = float(ylims[0]) if ylims[0] is not None else float(ax.get_ylim()[0])
    line_ymax = float(ylims[1]) if ylims[1] is not None else float(ax.get_ylim()[1])
    if ylims[0] is not None and ylims[1] is not None:
        ax.set_ylim(line_ymin, line_ymax)

    def _draw_tick_stats(x_draw, agg, color):
        if not show_tick_stats or agg is None:
            return
        mean_y, median_y = _weighted_mean_and_median(agg["values"], agg["weights"])
        if mean_y is None or median_y is None:
            return

        bar_half = float(violin_width) * float(tick_stat_bar_frac) * 0.5
        ax.plot(
            [x_draw - bar_half, x_draw + bar_half],
            [median_y, median_y],
            color=color,
            lw=tick_stat_line_width,
            alpha=tick_stat_alpha,
            zorder=5,
        )
        ax.scatter(
            [x_draw],
            [mean_y],
            s=float(tick_stat_mean_marker_size),
            marker="o",
            color=color,
            alpha=tick_stat_alpha,
            edgecolors="none",
            zorder=6,
        )

    plotted_centers = []
    if mode == "single":
        name, vdata, color, _yraw = sources[0]
        for i, _x_ref in enumerate(display_ticks):
            q = float(query_ticks_by_source[name][i])
            if not np.isfinite(q):
                continue
            agg = _tick_aggregation_from_voxels(
                vdata,
                q,
                tick_knn_stats_params=tick_knn_stats_params,
                voxel_weight_mode=voxel_weight_mode,
            )
            if agg is None:
                continue
            dens = weighted_kde_1d(
                agg["values"],
                agg["weights"],
                kde_points=int(kde_points),
                pad_frac=0.05,
                bw_method=kde_bw_method,
            )
            if dens is None:
                continue
            y_grid, density = dens
            peak = float(np.nanmax(density))
            if not np.isfinite(peak) or peak <= 0:
                continue

            x_draw = float(agg["effective_mean"])
            plotted_centers.append(x_draw)

            if tick_line:
                ax.plot(
                    [x_draw, x_draw],
                    [line_ymin, line_ymax],
                    color=tick_line_color,
                    alpha=tick_line_alpha,
                    lw=tick_line_width,
                    zorder=1,
                )

            half = (density / peak) * float(violin_width)
            ax.fill_betweenx(
                y_grid,
                x_draw - half,
                x_draw + half,
                color=color,
                alpha=violin_alpha,
                lw=0.0,
                zorder=3,
            )
            ax.plot(
                x_draw - half,
                y_grid,
                color=color,
                lw=violin_line_width,
                alpha=1.0,
                zorder=4,
            )
            ax.plot(
                x_draw + half,
                y_grid,
                color=color,
                lw=violin_line_width,
                alpha=1.0,
                zorder=4,
            )
            _draw_tick_stats(x_draw, agg, color)
    else:
        name_l, vleft, color_l, _yl = sources[0]
        name_r, vright, color_r, _yr = sources[1]
        q_l_all = query_ticks_by_source[name_l]
        q_r_all = query_ticks_by_source[name_r]

        for i, _x_ref in enumerate(display_ticks):
            ql = float(q_l_all[i])
            qr = float(q_r_all[i])

            agg_l = (
                _tick_aggregation_from_voxels(
                    vleft,
                    ql,
                    tick_knn_stats_params=tick_knn_stats_params,
                    voxel_weight_mode=voxel_weight_mode,
                )
                if np.isfinite(ql)
                else None
            )
            agg_r = (
                _tick_aggregation_from_voxels(
                    vright,
                    qr,
                    tick_knn_stats_params=tick_knn_stats_params,
                    voxel_weight_mode=voxel_weight_mode,
                )
                if np.isfinite(qr)
                else None
            )
            if agg_l is None and agg_r is None:
                continue

            d_l = (
                weighted_kde_1d(
                    agg_l["values"],
                    agg_l["weights"],
                    kde_points=int(kde_points),
                    pad_frac=0.05,
                    bw_method=kde_bw_method,
                )
                if agg_l is not None
                else None
            )
            d_r = (
                weighted_kde_1d(
                    agg_r["values"],
                    agg_r["weights"],
                    kde_points=int(kde_points),
                    pad_frac=0.05,
                    bw_method=kde_bw_method,
                )
                if agg_r is not None
                else None
            )
            if d_l is None and d_r is None:
                continue

            centers = []
            if agg_l is not None:
                centers.append(float(agg_l["effective_mean"]))
            if agg_r is not None:
                centers.append(float(agg_r["effective_mean"]))
            x_draw = float(np.mean(centers))
            plotted_centers.append(x_draw)

            if tick_line:
                ax.plot(
                    [x_draw, x_draw],
                    [line_ymin, line_ymax],
                    color=tick_line_color,
                    alpha=tick_line_alpha,
                    lw=tick_line_width,
                    zorder=1,
                )

            peak = 0.0
            if d_l is not None:
                peak = max(peak, float(np.nanmax(d_l[1])))
            if d_r is not None:
                peak = max(peak, float(np.nanmax(d_r[1])))
            if peak <= 0:
                continue

            if d_l is not None:
                yl, dl = d_l
                wl = (dl / peak) * float(violin_width)
                ax.fill_betweenx(
                    yl,
                    x_draw - wl,
                    x_draw,
                    color=color_l,
                    alpha=violin_alpha,
                    lw=0.0,
                    zorder=3,
                )
                ax.plot(
                    x_draw - wl,
                    yl,
                    color=color_l,
                    lw=violin_line_width,
                    alpha=1.0,
                    zorder=4,
                )
                _draw_tick_stats(x_draw, agg_l, color_l)

            if d_r is not None:
                yr, dr = d_r
                wr = (dr / peak) * float(violin_width)
                ax.fill_betweenx(
                    yr,
                    x_draw,
                    x_draw + wr,
                    color=color_r,
                    alpha=violin_alpha,
                    lw=0.0,
                    zorder=3,
                )
                ax.plot(
                    x_draw + wr,
                    yr,
                    color=color_r,
                    lw=violin_line_width,
                    alpha=1.0,
                    zorder=4,
                )
                _draw_tick_stats(x_draw, agg_r, color_r)

    axis_ticks = np.asarray(
        plotted_centers if len(plotted_centers) > 0 else display_ticks, dtype=float
    )
    xmin = float(np.nanmin(axis_ticks)) if xlims[0] is None else float(xlims[0])
    xmax = float(np.nanmax(axis_ticks)) if xlims[1] is None else float(xlims[1])
    ymin = float(ylims[0]) if ylims[0] is not None else float(ax.get_ylim()[0])
    ymax = float(ylims[1]) if ylims[1] is not None else float(ax.get_ylim()[1])

    if rescaler is not None:
        setup_transformed_axis(
            ax,
            xaxis_lims=(xmin, xmax),
            yaxis_lims=(ymin, ymax),
            rescaler=rescaler,
            margins=0.0,
        )
    else:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    # Match smooth_* styling: no top/right frame.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_identity_line:
        lo = max(xmin, ymin)
        hi = min(xmax, ymax)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            ax.plot(
                [lo, hi],
                [lo, hi],
                color=identity_line_color,
                linestyle=identity_line_style,
                lw=identity_line_width,
                alpha=identity_line_alpha,
                zorder=2,
            )

    if show_marginal_kde:
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)
        ax_xm = divider.append_axes("top", size=marginal_size, pad=float(marginal_pad))
        ax_ym = divider.append_axes("right", size=marginal_size, pad=float(marginal_pad))

        max_dx = 0.0
        max_dy = 0.0
        for _name, vdata, color, yraw in sources:
            xvals = np.asarray(vdata["means"], dtype=float)
            xw = None
            if voxel_weight_mode == "count":
                xw = np.asarray(vdata["counts"], dtype=float)
            xkde = weighted_kde_1d(
                xvals,
                xw,
                kde_points=int(kde_points),
                pad_frac=float(marginal_kde_pad_frac),
                bw_method=kde_bw_method,
            )
            if xkde is not None:
                xg, xd = xkde
                xd = np.asarray(xd, dtype=float)
                if marginal_normalize:
                    peak = float(np.nanmax(xd))
                    if np.isfinite(peak) and peak > 0:
                        xd = xd / peak
                ax_xm.plot(xg, xd, color=color, lw=marginal_line_width)
                ax_xm.fill_between(xg, 0.0, xd, color=color, alpha=marginal_fill_alpha, lw=0.0)
                max_dx = max(max_dx, float(np.nanmax(xd)))

            yvals = np.asarray(yraw, dtype=float).ravel()
            ykde = weighted_kde_1d(
                yvals,
                None,
                kde_points=int(kde_points),
                pad_frac=float(marginal_kde_pad_frac),
                bw_method=kde_bw_method,
            )
            if ykde is not None:
                yg, yd = ykde
                yd = np.asarray(yd, dtype=float)
                if marginal_normalize:
                    peak = float(np.nanmax(yd))
                    if np.isfinite(peak) and peak > 0:
                        yd = yd / peak
                ax_ym.plot(yd, yg, color=color, lw=marginal_line_width)
                ax_ym.fill_betweenx(yg, 0.0, yd, color=color, alpha=marginal_fill_alpha, lw=0.0)
                max_dy = max(max_dy, float(np.nanmax(yd)))

        ax_xm.set_xlim(ax.get_xlim())
        if max_dx > 0:
            ax_xm.set_ylim(0.0, max_dx * 1.05)

        ax_ym.set_ylim(ax.get_ylim())
        if max_dy > 0:
            ax_ym.set_xlim(0.0, max_dy * 1.05)

        for spine in ["top", "right", "left", "bottom"]:
            ax_xm.spines[spine].set_visible(False)
            ax_ym.spines[spine].set_visible(False)
        ax_xm.tick_params(
            axis="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
        )
        ax_ym.tick_params(
            axis="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
        )
        ax_xm.set_facecolor("none")
        ax_ym.set_facecolor("none")

    xlabel = f"Mean measured {output_name}" if xtitle is None else xtitle
    ylabel = output_name if ytitle is None else ytitle
    if draw_xlabel and xlabel:
        ax.set_xlabel(xlabel)
    if draw_ylabel and ylabel:
        ax.set_ylabel(ylabel)
    if title is not None:
        if show_marginal_kde:
            ax.set_title(title, y=float(title_y_with_marginal))
        else:
            ax.set_title(title)

    return {
        "mode": mode,
        "ticks": np.asarray(display_ticks, dtype=float),
        "query_ticks": {k: np.asarray(v, dtype=float) for k, v in query_ticks_by_source.items()},
        "plotted_x": np.asarray(plotted_centers, dtype=float),
        "n_voxels": {name: int(len(vd["means"])) for name, vd, _color, _yraw in sources},
    }
