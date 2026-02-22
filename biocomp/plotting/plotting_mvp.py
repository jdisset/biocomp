"""Measured vs Predicted scatter plot with density and trendline.

Optional generative-model diagnostics (pass ``model_samples``):
- PIT histogram inset (auto-computed, or pass ``pit_values`` directly)
- Sample-based coverage bands (empirical quantiles from model draws)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray as NdArray

from biocomp.metric_utils import rmse, r_squared
from .plotting_core import configurable, setup_transformed_axis, build_tree, knn_stats

if TYPE_CHECKING:
    from biocomp.datautils import DataRescaler


def _clean_paired(measured: NdArray, predicted: NdArray) -> tuple[NdArray, NdArray]:
    mask = np.isfinite(measured) & np.isfinite(predicted)
    return measured[mask], predicted[mask]


def _axis_lims(measured: NdArray, predicted: NdArray, vlims: tuple, margins: float) -> tuple[float, float]:
    lo = float(vlims[0]) if vlims[0] is not None else float(min(measured.min(), predicted.min()))
    hi = float(vlims[1]) if vlims[1] is not None else float(max(measured.max(), predicted.max()))
    span = hi - lo
    return lo - margins * span, hi + margins * span


# ── PIT histogram inset ──────────────────────────────────────────────────


def _draw_pit_inset(
    ax,
    pit_values: NdArray,
    nbins: int = 20,
    position: tuple[float, float, float, float] = (0.62, 0.03, 0.35, 0.22),
):
    pit = np.asarray(pit_values).ravel()
    pit = pit[np.isfinite(pit)]
    assert len(pit) > 0

    inset = ax.inset_axes(position)
    inset.hist(
        pit, bins=nbins, range=(0, 1), density=True,
        color="steelblue", alpha=0.8, edgecolor="white", lw=0.3,
    )
    inset.axhline(1.0, color="black", ls="--", lw=0.6, alpha=0.7)
    inset.set_xlim(0, 1)
    inset.set_ylim(0, None)
    inset.set_title("PIT", fontsize=6, pad=2)
    inset.tick_params(labelsize=5, length=2, pad=1)
    inset.set_xticks([0, 0.5, 1])
    for spine in inset.spines.values():
        spine.set_linewidth(0.4)


# ── conditional violin whiskers ──────────────────────────────────────────


def _violin_half_kde(
    values: NdArray, weights: NdArray, kde_points: int = 80,
) -> tuple[NdArray, NdArray] | None:
    """Compute weighted KDE, return (y_grid, density) or None on failure."""
    from scipy.stats import gaussian_kde

    w = weights / weights.sum()
    try:
        kde = gaussian_kde(values, weights=w)
    except (np.linalg.LinAlgError, ValueError):
        return None
    y_lo, y_hi = values.min(), values.max()
    pad = (y_hi - y_lo) * 0.15
    y_grid = np.linspace(y_lo - pad, y_hi + pad, kde_points)
    return y_grid, kde(y_grid)


def _draw_violins(
    ax,
    measured: NdArray,
    predicted: NdArray,
    tree_measured,
    tree_predicted,
    knn_kw: dict,
    dense_mask: NdArray,
    eval_x: NdArray,
    n_violins: int = 5,
    violin_width: float | None = None,
    color_right: str = "#1f77b4",
    color_left: str = "#d62728",
    alpha: float = 0.25,
    contour_lw: float = 0.8,
    kde_points: int = 80,
):
    """Mirrored conditional violins.

    Right half: P(predicted | measured ≈ t) — "truth is t, where do predictions land?"
    Left half:  P(measured | predicted ≈ t) — "model says t, what was the truth?"
    """
    dense_x = eval_x[dense_mask]
    if len(dense_x) < 2:
        return
    positions = np.linspace(dense_x[0], dense_x[-1], n_violins + 2)[1:-1]
    if violin_width is None:
        violin_width = (dense_x[-1] - dense_x[0]) / (n_violins + 2) * 0.45

    for t in positions:
        query = np.array([[t]])

        # right half: P(predicted | measured ≈ t)
        iw_m = knn_stats(query, tree=tree_measured, stats="iw", use_jax=False, **knn_kw)
        idx_m, w_m = iw_m[0][0], iw_m[1][0]
        valid_m = np.isfinite(w_m) & (w_m > 0)

        # left half: P(measured | predicted ≈ t)
        iw_p = knn_stats(query, tree=tree_predicted, stats="iw", use_jax=False, **knn_kw)
        idx_p, w_p = iw_p[0][0], iw_p[1][0]
        valid_p = np.isfinite(w_p) & (w_p > 0)

        if valid_m.sum() < 10 or valid_p.sum() < 10:
            continue

        right = _violin_half_kde(predicted[idx_m[valid_m]], w_m[valid_m], kde_points)
        left = _violin_half_kde(measured[idx_p[valid_p]], w_p[valid_p], kde_points)
        if right is None or left is None:
            continue

        # normalize both to same max width
        r_grid, r_dens = right
        l_grid, l_dens = left
        peak = max(r_dens.max(), l_dens.max())
        r_dens = r_dens / peak * violin_width
        l_dens = l_dens / peak * violin_width

        ax.fill_betweenx(r_grid, t, t + r_dens, alpha=alpha, color=color_right, lw=0, zorder=5)
        ax.plot(t + r_dens, r_grid, color=color_right, lw=contour_lw, alpha=0.6, zorder=5)

        ax.fill_betweenx(l_grid, t - l_dens, t, alpha=alpha, color=color_left, lw=0, zorder=5)
        ax.plot(t - l_dens, l_grid, color=color_left, lw=contour_lw, alpha=0.6, zorder=5)


# ── sample-based coverage bands ──────────────────────────────────────────


def _pit_from_samples(predicted: NdArray, model_samples: NdArray) -> NdArray:
    """Empirical PIT: fraction of model samples <= observed value."""
    assert model_samples.ndim == 2 and model_samples.shape[1] == len(predicted)
    return (model_samples <= predicted[None, :]).mean(axis=0)


def _draw_sample_bands(
    ax,
    eval_x: NdArray,
    tree,
    predicted: NdArray,
    model_samples: NdArray,
    bands: list[list[int]],
    knn_kw: dict,
    dense_mask: NdArray,
    smooth_sigma: float = 0.0,
    color: str = "#d62728",
    lw: float = 1.5,
    show_coverage: bool = True,
):
    """Draw quantile bands from model samples, smoothed via knn_stats."""
    assert model_samples.ndim == 2 and model_samples.shape[1] == len(predicted)
    query = eval_x[:, None]
    for band in bands:
        q_lo_pct, q_hi_pct = band[0], band[1]
        q_lo, q_hi = q_lo_pct / 100.0, q_hi_pct / 100.0

        q_lo_vals = np.quantile(model_samples, q_lo, axis=0)
        q_hi_vals = np.quantile(model_samples, q_hi, axis=0)

        y_lo = np.asarray(knn_stats(query, y=q_lo_vals[:, None], tree=tree, stats="mean", use_jax=False, **knn_kw)).ravel()
        y_hi = np.asarray(knn_stats(query, y=q_hi_vals[:, None], tree=tree, stats="mean", use_jax=False, **knn_kw)).ravel()
        y_lo[~dense_mask] = np.nan
        y_hi[~dense_mask] = np.nan
        if smooth_sigma > 0:
            from scipy.ndimage import gaussian_filter1d

            for y in (y_lo, y_hi):
                finite = np.isfinite(y)
                y[finite] = gaussian_filter1d(y[finite], sigma=smooth_sigma)

        ax.plot(eval_x, y_lo, color=color, ls="--", lw=lw, zorder=4)
        ax.plot(eval_x, y_hi, color=color, ls="--", lw=lw, zorder=4)

        if show_coverage:
            within = (predicted >= q_lo_vals) & (predicted <= q_hi_vals)
            coverage = within.mean()
            expected = q_hi - q_lo
            mid_idx = len(eval_x) // 2
            ax.annotate(
                f"{coverage:.0%} in {q_lo_pct}-{q_hi_pct}%\n(expect {expected:.0%})",
                xy=(eval_x[mid_idx], y_hi[mid_idx]), fontsize=6, color=color,
                xytext=(5, 5), textcoords="offset points", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec=color, lw=0.5),
            )


# ── main plot function ───────────────────────────────────────────────────


@configurable
def measured_vs_predicted(
    ax,
    measured: NdArray,
    predicted: NdArray,
    rescaler: DataRescaler | None = None,
    # density heatmap
    show_density: bool = True,
    density_res: int = 200,
    density_cmap: str = "bc_blues",
    density_log: bool = True,
    density_noise_smooth: float = 0.25,
    # trendline (stacked poly)
    show_trendline: bool = True,
    trendline_color: str = "black",
    trendline_lw: float = 1.0,
    trendline_eval_points: int = 200,
    trendline_quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    trendline_degree: int = 1,
    # knn params (density cutoff, violins, bands, residual smoothing)
    knn_stats_params: dict | None = None,
    density_cutoff_q: float = 0.05,
    smooth_sigma: float = 3.0,
    # 1:1 line
    show_identity: bool = True,
    identity_color: str = "grey",
    identity_ls: str = "--",
    identity_lw: float = 1.0,
    # stats
    show_stats: bool = True,
    # axis
    vlims: tuple = (0.0, 0.7),
    margins: float = 0.02,
    xlabel: str = "Measured",
    ylabel: str = "Predicted",
    # ── generative model diagnostics ──
    model_samples: NdArray | None = None,
    model_bands: list[list[int]] | None = None,
    model_band_color: str = "#d62728",
    model_band_lw: float = 1.5,
    show_coverage: bool = True,
    pit_values: NdArray | None = None,
    pit_nbins: int = 20,
    # ── conditional violins ──
    show_violins: bool = False,
    n_violins: int = 5,
    violin_width: float | None = None,
    violin_color_right: str = "#1f77b4",
    violin_color_left: str = "#d62728",
    violin_alpha: float = 0.25,
    # ── residual std profile ──
    residual_ax=None,
    residual_color: str = "black",
    residual_fill_alpha: float = 0.15,
):
    """Render a measured-vs-predicted scatter plot on ``ax``.

    Generative model diagnostics (pass ``model_samples``, shape ``(n_draws, n_points)``):
      - PIT histogram inset (auto-computed from samples, or pass ``pit_values``).
      - Sample-based coverage bands: pointwise quantiles smoothed with knn_stats.
    """
    if model_bands is None:
        model_bands = [[25, 75]]

    measured = np.asarray(measured).ravel()
    predicted = np.asarray(predicted).ravel()
    assert measured.shape == predicted.shape
    measured, predicted = _clean_paired(measured, predicted)
    assert len(measured) > 0, "No finite data points"

    if rescaler is not None:
        measured = np.asarray(rescaler.fwd(measured))
        predicted = np.asarray(rescaler.fwd(predicted))

    lo, hi = _axis_lims(measured, predicted, vlims, margins)
    eval_x = np.linspace(lo, hi, trendline_eval_points)

    knn_stats_params = knn_stats_params or {}
    knn_stats_params.setdefault("radius", 0.1)
    tree = build_tree(measured[:, None], use_jax=False)

    # --- density cutoff: mask eval points in low-density regions ---
    query = eval_x[:, None]
    eval_density = np.asarray(knn_stats(query, tree=tree, stats="density", use_jax=False, **knn_stats_params)).ravel()
    data_density = np.asarray(knn_stats(measured[:, None], tree=tree, stats="density", use_jax=False, **knn_stats_params)).ravel()
    density_threshold = np.quantile(data_density[np.isfinite(data_density)], density_cutoff_q)
    dense_mask = eval_density >= density_threshold

    # --- density heatmap ---
    if show_density:
        nbins = density_res
        if density_noise_smooth > 0:
            res = (hi - lo) / nbins
            m_jitter = measured + np.random.default_rng(0).normal(
                scale=density_noise_smooth * res, size=measured.shape
            )
            p_jitter = predicted + np.random.default_rng(1).normal(
                scale=density_noise_smooth * res, size=predicted.shape
            )
        else:
            m_jitter, p_jitter = measured, predicted

        h, _xedges, _yedges = np.histogram2d(
            m_jitter, p_jitter, bins=nbins, range=[[lo, hi], [lo, hi]], density=False,
        )
        h = np.ma.masked_where(h == 0, h)
        if density_log:
            h = np.log1p(h)

        ax.imshow(
            h.T, extent=[lo, hi, lo, hi], origin="lower", aspect="auto",
            cmap=density_cmap, interpolation="nearest",
        )

    # --- 1:1 reference line ---
    if show_identity:
        ax.plot([lo, hi], [lo, hi], color=identity_color, ls=identity_ls, lw=identity_lw, zorder=2)

    # --- trendline (stacked poly) ---
    if show_trendline:
        import jax.numpy as jnp
        from biocomp.plotting.stacked_poly import fit_stacked_poly_at_quantiles, evaluate_stacked_poly

        quantiles = jnp.array(trendline_quantiles)
        weights = jnp.ones(len(measured))
        params = fit_stacked_poly_at_quantiles(
            jnp.asarray(measured), jnp.asarray(predicted), weights,
            quantiles, degree=trendline_degree,
        )
        trendline_y = np.array(
            evaluate_stacked_poly(jnp.asarray(eval_x), params), copy=True,
        )
        trendline_y[~dense_mask] = np.nan
        ax.plot(eval_x, trendline_y, color=trendline_color, lw=trendline_lw, zorder=3)

    # --- conditional violins ---
    if show_violins:
        tree_predicted = build_tree(predicted[:, None], use_jax=False)
        _draw_violins(
            ax, measured, predicted, tree, tree_predicted, knn_stats_params,
            dense_mask, eval_x, n_violins=n_violins, violin_width=violin_width,
            color_right=violin_color_right, color_left=violin_color_left,
            alpha=violin_alpha,
        )

    # --- generative model diagnostics ---
    if model_samples is not None:
        model_samples = np.asarray(model_samples)
        assert model_samples.ndim == 2 and model_samples.shape[1] == len(measured)

        _draw_sample_bands(
            ax, eval_x, tree, predicted, model_samples,
            bands=model_bands, knn_kw=knn_stats_params, dense_mask=dense_mask,
            smooth_sigma=smooth_sigma,
            color=model_band_color, lw=model_band_lw, show_coverage=show_coverage,
        )

        if pit_values is None:
            pit_values = _pit_from_samples(predicted, model_samples)

    if pit_values is not None:
        _draw_pit_inset(ax, pit_values, nbins=pit_nbins)

    # --- stats annotation ---
    if show_stats:
        _rmse = rmse(measured, predicted)
        _r2 = r_squared(measured, predicted)
        stats_text = f"RMSE = {_rmse:.4f}\nR² = {_r2:.4f}"
        ax.text(
            0.05, 0.95, stats_text, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="grey"),
        )

    # --- residual std profile ---
    if residual_ax is not None:
        from scipy.ndimage import gaussian_filter1d as _g1d

        residuals = (predicted - measured)[:, None]
        res_mean, res_std = knn_stats(
            query, y=residuals, tree=tree, stats=["mean", "std"], use_jax=False, **knn_stats_params,
        )
        res_mean = np.asarray(res_mean).ravel()
        res_std = np.asarray(res_std).ravel()
        res_mean[~dense_mask] = np.nan
        res_std[~dense_mask] = np.nan
        if smooth_sigma > 0:
            for arr in (res_mean, res_std):
                finite = np.isfinite(arr)
                arr[finite] = _g1d(arr[finite], sigma=smooth_sigma)

        residual_ax.fill_between(eval_x, 0, res_std, alpha=residual_fill_alpha, color=residual_color)
        residual_ax.plot(eval_x, res_std, color=residual_color, lw=0.8, label="σ(residual)")
        residual_ax.plot(eval_x, res_mean, color=residual_color, lw=0.8, ls="--", alpha=0.6, label="bias")
        residual_ax.axhline(0, color="grey", lw=0.5, ls=":")
        residual_ax.set_xlim(lo, hi)
        residual_ax.set_ylim(bottom=min(0, float(np.nanmin(res_mean)) * 1.2))
        residual_ax.set_ylabel("Residual", fontsize=7)
        residual_ax.tick_params(labelsize=6)
        residual_ax.legend(fontsize=6, loc="upper right")

    # --- axis setup ---
    if rescaler is not None:
        setup_transformed_axis(ax, xaxis_lims=[lo, hi], yaxis_lims=[lo, hi], rescaler=rescaler)
    else:
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
