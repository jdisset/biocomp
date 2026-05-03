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
from .plotting_core import (
    configurable,
    setup_transformed_axis,
    build_tree,
    knn_stats,
    weighted_kde_1d,
)

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
    return weighted_kde_1d(values, weights, kde_points=kde_points, pad_frac=0.15)


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


def fit_median_trend(
    measured: NdArray,
    predicted: NdArray,
    eval_x: NdArray,
    degree: int = 1,
    quantiles: tuple[float, ...] = (0.4, 0.5, 0.6),
) -> NdArray:
    """Fit a stacked-polynomial conditional median of `predicted` given `measured`.

    Returns the trend evaluated at `eval_x`. Used both for the optional
    display trendline and for the bias-area metric (see `bias_area`).

    The underlying ``fit_stacked_poly_at_quantiles`` requires at least
    two distinct quantiles (it builds Gaussian-weighted segments between
    them), so this helper defaults to a tight band around 0.5 — the
    average across these quantiles approximates the conditional median
    while staying numerically stable.
    """
    import jax.numpy as jnp
    from biocomp.plotting.stacked_poly import (
        evaluate_stacked_poly,
        fit_stacked_poly_at_quantiles,
    )

    q = jnp.array(quantiles)
    weights = jnp.ones(len(measured))
    params = fit_stacked_poly_at_quantiles(
        jnp.asarray(measured), jnp.asarray(predicted), weights,
        q, degree=degree,
    )
    return np.asarray(evaluate_stacked_poly(jnp.asarray(eval_x), params))


def calibration_rms(
    measured: NdArray,
    predicted: NdArray,
    eval_x: NdArray | None = None,
    dense_mask: NdArray | None = None,
    weights: NdArray | None = None,
    degree: int = 1,
    n_eval: int = 200,
    trend_y: NdArray | None = None,
    quantiles: tuple[float, ...] = (0.4, 0.5, 0.6),
) -> float:
    """RMS of `(conditional_median − x)` over the dense region.

    L2 partner of :func:`bias_area`. While `bias_area` is signed (and so
    can cancel positive and negative deviations), `calibration_rms`
    measures the *magnitude* of systematic miscalibration. Together with
    :func:`conditional_spread` and the data noise floor, this gives the
    classic Murphy / Bröcker decomposition of MSE:

        MSE  ≈  cal_rms²  +  spread²  +  noise²

    so `sqrt(cal_rms² + spread²)` is the model-only RMSE (noise excluded).

    Pass `trend_y` to skip the (cached) fit and reuse a precomputed
    conditional-median trend — useful when many metrics share one fit.
    """
    measured, predicted = _clean_paired(np.asarray(measured).ravel(), np.asarray(predicted).ravel())
    if measured.size < max(2, degree + 1):
        return float("nan")
    if eval_x is None:
        eval_x = np.linspace(float(np.min(measured)), float(np.max(measured)), n_eval)
    if trend_y is None:
        trend_y = fit_median_trend(measured, predicted, eval_x, degree=degree, quantiles=quantiles)
    if dense_mask is not None:
        if not np.any(dense_mask):
            return float("nan")
        x_use, y_use = eval_x[dense_mask], trend_y[dense_mask]
        w = weights[dense_mask] if weights is not None else None
    else:
        x_use, y_use = eval_x, trend_y
        w = weights
    diff_sq = (y_use - x_use) ** 2
    if w is not None and np.any(w > 0):
        return float(np.sqrt(np.average(diff_sq, weights=w)))
    return float(np.sqrt(np.mean(diff_sq)))


def conditional_spread(
    measured: NdArray,
    predicted: NdArray,
    degree: int = 1,
    trend_at_measured: NdArray | None = None,
    dense_mask_at_measured: NdArray | None = None,
    quantiles: tuple[float, ...] = (0.4, 0.5, 0.6),
) -> float:
    """RMS of `(predicted − conditional_median(predicted | measured))`.

    Captures the stochastic component of the model's MSE — how much the
    predictions disperse around their conditional median trend. This is
    the natural variance-side companion to :func:`calibration_rms`.

    Pass `trend_at_measured` to reuse a precomputed trend evaluation at
    each input point. `dense_mask_at_measured` filters out tail points
    where the trend is extrapolated, keeping spread comparable to
    `calibration_rms` (which already restricts to the dense region).
    """
    measured, predicted = _clean_paired(np.asarray(measured).ravel(), np.asarray(predicted).ravel())
    if measured.size < max(2, degree + 1):
        return float("nan")
    if trend_at_measured is None:
        trend_at_measured = fit_median_trend(measured, predicted, measured, degree=degree, quantiles=quantiles)
    residuals = predicted - trend_at_measured
    if dense_mask_at_measured is not None:
        residuals = residuals[dense_mask_at_measured]
    if residuals.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(residuals ** 2)))


def crps_empirical(
    observation: NdArray,
    samples: NdArray,
) -> float:
    """Continuous Ranked Probability Score for an empirical sample distribution.

    CRPS(F, y) = E_F|X − y| − ½ E_F|X − X'|

    Estimated unbiasedly per observation by

        crps_i = (1/M) Σ_m |s_im − y_i|  −  (1/(2M(M-1))) Σ_{m,m'} |s_im − s_im'|

    then averaged across observations. Reduces to MAE when the predictive
    distribution is a point mass; smaller is better. Generalises the
    Murphy decomposition to the full conditional distribution: it folds
    calibration AND sharpness into a single proper score.
    """
    obs = np.asarray(observation).ravel()
    samp = np.asarray(samples)
    if samp.ndim == 1:
        samp = samp[:, None]
    if samp.shape[1] != obs.size:
        # accept (n_obs, n_samples) — transpose if needed
        if samp.shape[0] == obs.size:
            samp = samp.T
        else:
            return float("nan")
    n_samples = samp.shape[0]
    if n_samples < 2:
        return float("nan")

    # E_F|X − y|
    mae_term = np.mean(np.abs(samp - obs[None, :]), axis=0)
    # ½ E_F|X − X'|, unbiased estimator using all sample pairs
    diffs = np.abs(samp[:, None, :] - samp[None, :, :])  # (n_samples, n_samples, n_obs)
    pair_term = diffs.sum(axis=(0, 1)) / (2.0 * n_samples * (n_samples - 1))

    crps_per_obs = mae_term - pair_term
    finite = np.isfinite(crps_per_obs)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(crps_per_obs[finite]))


def bias_area(
    measured: NdArray,
    predicted: NdArray,
    eval_x: NdArray | None = None,
    dense_mask: NdArray | None = None,
    degree: int = 1,
    n_eval: int = 200,
    signed: bool = True,
    trend_y: NdArray | None = None,
    quantiles: tuple[float, ...] = (0.4, 0.5, 0.6),
) -> float:
    """Average vertical offset of the conditional-median trend from y=x.

    Mathematically: ``∫ [f(x) - x] dx / Δx`` over the support, where
    ``f(x) = median(predicted | measured = x)`` is fit with a stacked
    polynomial. The result is in y-units and represents the mean
    *systematic* bias of the model — distinct from RMSE, which folds in
    both bias and variance. Switch `signed=False` for the unsigned
    (mean absolute calibration error) variant.

    Interpretation:
      ≈ 0   : the trend tracks y=x → well-calibrated on average
      > 0   : model systematically over-predicts
      < 0   : model systematically under-predicts

    Computed only over the dense region (`dense_mask`) to avoid
    extrapolating into low-density tails. If no dense region exists,
    returns ``nan``.
    """
    measured, predicted = _clean_paired(np.asarray(measured).ravel(), np.asarray(predicted).ravel())
    if measured.size < max(2, degree + 1):
        return float("nan")
    if eval_x is None:
        eval_x = np.linspace(float(np.min(measured)), float(np.max(measured)), n_eval)
    if trend_y is None:
        trend_y = fit_median_trend(measured, predicted, eval_x, degree=degree, quantiles=quantiles)

    if dense_mask is None:
        x_use, y_use = eval_x, trend_y
    else:
        if not np.any(dense_mask):
            return float("nan")
        x_use, y_use = eval_x[dense_mask], trend_y[dense_mask]

    width = float(x_use[-1] - x_use[0])
    if width <= 0:
        return float("nan")
    diff = y_use - x_use
    if not signed:
        diff = np.abs(diff)
    return float(np.trapz(diff, x_use) / width)  # noqa: NPY201


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
    density_res: int = 100,
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
    show_bias: bool = True,                 # signed bias-area (direction)
    bias_signed: bool = True,
    show_calibration_rms: bool = True,      # L2 magnitude of miscalibration
    show_spread: bool = True,               # RMS conditional dispersion around trend
    show_crps: bool = True,                 # only fires when model_samples is set
    extra_metrics: dict | None = None,      # appended to RMSE / R² (e.g. {"NRE": 0.42})
    # noise floor band — shaded envelope along y=x. Pass either an
    # explicit `noise_floor` (y-units) or `noise_nrmse` (dimensionless,
    # converted to y-units via `noise_nrmse * std(measured)`).
    # When `noise_local=True` the band's half-width *fluctuates* with
    # measured value: a knn-smoothed std of the residuals around the
    # median trend, calibrated to integrate to the global `noise_floor`.
    # Set `noise_local=False` to keep the constant-band behaviour.
    noise_floor: float | None = None,
    noise_nrmse: float | None = None,
    noise_local: bool = True,
    noise_local_radius: float = 0.08,
    noise_local_min_points: int = 30,
    noise_color: str = "grey",
    noise_alpha: float = 0.12,
    show_noise_floor: bool = True,
    # axis
    vlims: tuple = (0.0, 0.7),
    margins: float = 0.02,
    xlabel: str = "Measured",
    ylabel: str = "Predicted",
    title: str | None = None,
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
    # ── grid-mean overlay (cube-view "yellow crosses") ──
    # Per-grid-cell (gt_mean, yhat_mean) from the same Gaussian-KNN kernel
    # that feeds grid_nrmse / grid_snr. Off-diagonal crosses = systematic
    # bias structure. Marker size/alpha scale with `grid_weights` (n_eff)
    # so sparse-data corners fade out automatically.
    grid_measured: NdArray | None = None,
    grid_predicted: NdArray | None = None,
    grid_weights: NdArray | None = None,
    grid_min_weight: float = 1.0,
    grid_color: str = "#ffd400",
    grid_marker: str = "+",
    grid_lw: float = 1.0,
    grid_size_min: float = 8.0,
    grid_size_max: float = 60.0,
    grid_alpha_min: float = 0.25,
    grid_alpha_max: float = 0.95,
    grid_zorder: int = 6,
    # "crosses" | "density" | "both"
    grid_render: str = "density",
    grid_density_cmap: str = "bc_greens",
    grid_density_smooth_bins: float = 1.5,
    grid_density_alpha: float = 0.9,
    grid_density_log: bool = True,
    grid_density_levels: tuple[float, ...] = (0.5, 0.75, 0.9),
    grid_density_linewidth: float = 0.9,
    grid_density_fill_alpha: float = 0.1,
    grid_density_color: str | None = "black",
    grid_density_hatches: tuple[str, ...] | None = ("//", "\\\\", ".."),
    grid_density_hatch_linewidth: float = 0.4,
    # ── legend ──
    show_legend: bool = True,
    legend_loc: str = "upper right",
    legend_fontsize: float = 7,
    density_label: str = "raw data",
    grid_density_label: str = "kernel-smoothed",
    identity_label: str = "y = x",
    trendline_label: str = "median trend",
    legend_kwargs: dict | None = None,
    # ── per-pair density weights ──
    # When set, the density heatmap is built with `np.histogram2d(weights=…)`
    # so each (measured, predicted) pair contributes proportionally to its
    # weight rather than uniformly. Used by the noise-floor panel to feed
    # Gaussian-kernel weights through, so close kernel neighbours dominate
    # the density just like the smoothed plot does.
    density_weights: NdArray | None = None,
    # ── delta-vs-kernel marginal strip ──
    # Per-data-point kernel-smoother prediction aligned with `measured` /
    # `predicted`. When set, attaches a marginal strip below the cloud
    # showing 2D density of (measured_latent, model_pred_latent − kernel_pred_latent)
    # — the "extra information" the model contributes beyond the optimal
    # nonparametric predictor at the cube-view bandwidth. Zero-line means
    # the model agrees with the kernel-smoother; spread is structural
    # disagreement.
    kernel_predicted: NdArray | None = None,
    delta_strip_size: str = "20%",
    delta_strip_pad: float = 0.05,
    delta_strip_cmap: str = "bc_blues",
    delta_strip_log: bool = True,
    delta_strip_ylim: tuple[float, float] | None = None,
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
    finite = np.isfinite(measured) & np.isfinite(predicted)
    if density_weights is not None:
        density_weights = np.asarray(density_weights).ravel()
        assert density_weights.shape == measured.shape
        finite &= np.isfinite(density_weights)
    if kernel_predicted is not None:
        kernel_predicted = np.asarray(kernel_predicted).ravel()
        assert kernel_predicted.shape == measured.shape
        finite &= np.isfinite(kernel_predicted)
    if density_weights is not None:
        density_weights = density_weights[finite]
    if kernel_predicted is not None:
        kernel_predicted = kernel_predicted[finite]
    measured = measured[finite]
    predicted = predicted[finite]
    assert len(measured) > 0, "No finite data points"

    if rescaler is not None:
        measured = np.asarray(rescaler.fwd(measured))
        predicted = np.asarray(rescaler.fwd(predicted))
        if kernel_predicted is not None:
            kernel_predicted = np.asarray(rescaler.fwd(kernel_predicted))

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
    # Also mask raw measured points by the same density floor so spread /
    # local-noise estimators don't include extrapolated tails.
    dense_mask_at_measured = data_density >= density_threshold

    # ── Conditional-median trend (single fit, shared everywhere) ────────
    # Compute once with `trendline_quantiles` + `trendline_degree` so the
    # trendline the viewer SEES, the bias / cal_rms / spread metrics in
    # the stats annotation, and the local noise-floor envelope all agree.
    # Helpers (`bias_area`, `calibration_rms`, `conditional_spread`) skip
    # their own fits when `trend_y` / `trend_at_measured` is provided.
    try:
        trend_eval = fit_median_trend(
            measured, predicted, eval_x,
            degree=trendline_degree, quantiles=trendline_quantiles,
        )
        trend_at_measured = fit_median_trend(
            measured, predicted, measured,
            degree=trendline_degree, quantiles=trendline_quantiles,
        )
    except Exception:
        trend_eval = None
        trend_at_measured = None

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
            weights=density_weights,
        )
        h = np.ma.masked_where(h == 0, h)
        if density_log:
            h = np.log1p(h)

        ax.imshow(
            h.T, extent=[lo, hi, lo, hi], origin="lower", aspect="auto",
            cmap=density_cmap, interpolation="nearest",
        )

    # --- noise floor band (irreducible-noise envelope along y=x) ───────
    # Resolves either a direct y-units half-width or a dimensionless nRMSE
    # estimate scaled by the global measured std. Sits below the identity
    # line and density so the scatter / contours stay legible. When
    # `noise_local=True`, the band fluctuates: knn-smoothed local std of
    # (measured − median_trend), calibrated so its mean matches the global
    # noise scale derived above.
    _noise_half = noise_floor
    if (
        _noise_half is None
        and noise_nrmse is not None
        and np.isfinite(noise_nrmse)
        and measured.size
    ):
        _noise_half = float(noise_nrmse * np.std(measured))
    if show_noise_floor and _noise_half is not None and np.isfinite(_noise_half) and _noise_half > 0:
        if noise_local and trend_at_measured is not None:
            try:
                residuals = (measured - trend_at_measured)[:, None]
                # Local std of residuals over the eval grid; gives a
                # smooth σ(x) for free.
                sigma = np.asarray(
                    knn_stats(
                        eval_x[:, None], y=residuals, tree=tree,
                        stats="std", use_jax=False,
                        radius=noise_local_radius, min_points=noise_local_min_points,
                    )
                ).ravel()
                # Calibrate the local profile so its dense-region mean
                # matches the global noise scale (preserves total area).
                cal_mask = dense_mask & np.isfinite(sigma) & (sigma > 0)
                if np.any(cal_mask):
                    scale = _noise_half / float(np.mean(sigma[cal_mask]))
                    sigma_cal = sigma * scale
                    sigma_cal[~np.isfinite(sigma_cal)] = _noise_half
                    ax.fill_between(
                        eval_x, eval_x - sigma_cal, eval_x + sigma_cal,
                        color=noise_color, alpha=noise_alpha, zorder=1,
                        linewidth=0,
                    )
                else:
                    noise_local = False
            except Exception:
                noise_local = False
        if not noise_local:
            band_x = np.array([lo, hi])
            ax.fill_between(
                band_x, band_x - _noise_half, band_x + _noise_half,
                color=noise_color, alpha=noise_alpha, zorder=1,
                linewidth=0,
            )

    # --- 1:1 reference line ---
    if show_identity:
        ax.plot([lo, hi], [lo, hi], color=identity_color, ls=identity_ls, lw=identity_lw, zorder=2)

    # --- trendline (shared with bias / cal_rms / spread) ───────────────
    # Reuses the `trend_eval` fit computed once above; ensures the
    # displayed curve matches the metrics in the stats annotation.
    if show_trendline and trend_eval is not None:
        trendline_y = trend_eval.copy()
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

    # --- stats annotation ─────────────────────────────────────────────
    # Build (key, value, signed) rows in display order, then format with
    # a single monospace column width. Each derived metric is wrapped in
    # try/except so a degenerate inner fit never blocks the rest of the
    # plot.
    if show_stats:
        # Density-weighted bias / cal_rms so the Murphy decomposition
        # `MSE ≈ cal² + spread² + noise²` holds: weighting by data
        # density turns the eval-grid integral into an estimate of
        # `E_x[(f(x)−x)²]`. Without weights, sparse-but-dense-mask-passing
        # tail regions inflate the metric. All three trend-based metrics
        # share `trend_eval` / `trend_at_measured` computed above so the
        # displayed curve and the stats agree.
        bias_kw = dict(eval_x=eval_x, dense_mask=dense_mask, trend_y=trend_eval)
        cal_kw = dict(**bias_kw, weights=eval_density)
        spread_kw = dict(
            trend_at_measured=trend_at_measured,
            dense_mask_at_measured=dense_mask_at_measured,
        )
        derived: list[tuple[bool, str, "object", bool]] = [
            (True, "RMSE", lambda: rmse(measured, predicted), False),
            (True, "R²", lambda: r_squared(measured, predicted), False),
            (
                show_bias,
                "bias" if bias_signed else "|bias|",
                lambda: bias_area(measured, predicted, signed=bias_signed, **bias_kw),
                bias_signed,
            ),
            (show_calibration_rms, "cal_rms", lambda: calibration_rms(measured, predicted, **cal_kw), False),
            (show_spread, "spread", lambda: conditional_spread(measured, predicted, **spread_kw), False),
            (show_crps and model_samples is not None, "CRPS", lambda: crps_empirical(measured, model_samples), False),
        ]
        rows: list[tuple[str, float, bool]] = []
        for enabled, key, compute, signed in derived:
            if not enabled:
                continue
            try:
                v = float(compute())  # type: ignore[operator]
            except Exception:
                v = float("nan")
            if np.isfinite(v):
                rows.append((key, v, signed))

        if extra_metrics:
            for k, v in extra_metrics.items():
                if v is None:
                    continue
                if isinstance(v, float) and not np.isfinite(v):
                    continue
                rows.append((k, v, False))  # type: ignore[arg-type]

        kw = max(len(k) for k, _, _ in rows)
        lines = []
        for k, v, signed in rows:
            if isinstance(v, (int, float)):
                fmt = f"{v:+.4f}" if signed else f"{v:.4f}"
            else:
                fmt = str(v)
            lines.append(f"{k.ljust(kw)} = {fmt}")
        stats_text = "\n".join(lines)
        ax.text(
            0.05, 0.95, stats_text, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="grey"),
        )

    # --- cube-view grid-mean overlay ────────────────────────────────────
    if grid_measured is not None and grid_predicted is not None:
        gm = np.asarray(grid_measured).ravel()
        gp = np.asarray(grid_predicted).ravel()
        if rescaler is not None:
            gm = np.asarray(rescaler.fwd(gm))
            gp = np.asarray(rescaler.fwd(gp))
        gw = (
            np.asarray(grid_weights).ravel()
            if grid_weights is not None
            else np.ones_like(gm)
        )
        ok = (
            np.isfinite(gm) & np.isfinite(gp) & np.isfinite(gw)
            & (gw >= grid_min_weight)
            & (gm >= lo) & (gm <= hi) & (gp >= lo) & (gp <= hi)
        )
        if ok.any():
            gm, gp, gw = gm[ok], gp[ok], gw[ok]

            if grid_render in ("density", "both"):
                h_g, xedges, yedges = np.histogram2d(
                    gm, gp, bins=density_res,
                    range=[[lo, hi], [lo, hi]], density=False,
                    weights=gw,
                )
                if grid_density_smooth_bins > 0:
                    from scipy.ndimage import gaussian_filter
                    h_g = gaussian_filter(h_g, sigma=grid_density_smooth_bins)
                if grid_density_log:
                    h_g = np.log1p(h_g)
                pos = h_g[h_g > 0]
                if pos.size:
                    levels = np.unique(np.quantile(pos, grid_density_levels))
                    if levels.size >= 2:
                        xc = 0.5 * (xedges[:-1] + xedges[1:])
                        yc = 0.5 * (yedges[:-1] + yedges[1:])
                        use_hatch = (
                            grid_density_hatches is not None
                            and len(grid_density_hatches) > 0
                        )
                        # Single string color: RGBA tuples in `colors=`
                        # suppress hatch rendering in some matplotlib versions.
                        if grid_density_color is not None:
                            single_color = grid_density_color
                            line_colors = grid_density_color
                        else:
                            from matplotlib import colormaps
                            from matplotlib.colors import to_hex

                            cmap_obj = colormaps[grid_density_cmap]
                            single_color = to_hex(cmap_obj(0.85))
                            line_colors = cmap_obj(
                                np.linspace(0.55, 0.95, len(levels))
                            )
                        if grid_density_fill_alpha > 0 or use_hatch:
                            import matplotlib as _mpl

                            cf_kwargs = dict(
                                levels=levels,
                                colors=single_color,
                                alpha=grid_density_fill_alpha,
                                zorder=grid_zorder - 1,
                            )
                            if use_hatch:
                                n_bands = len(levels)
                                cf_kwargs["hatches"] = list(
                                    grid_density_hatches[:n_bands]
                                )
                                cf_kwargs["extend"] = "max"
                            with _mpl.rc_context({
                                "hatch.linewidth": grid_density_hatch_linewidth,
                                "hatch.color": single_color,
                            }):
                                ax.contourf(xc, yc, h_g.T, **cf_kwargs)
                        ax.contour(
                            xc, yc, h_g.T, levels=levels,
                            colors=line_colors,
                            linewidths=grid_density_linewidth,
                            alpha=grid_density_alpha,
                            zorder=grid_zorder - 0.5,
                        )

            if grid_render in ("crosses", "both"):
                w_max = float(gw.max()) if gw.size else 1.0
                w_norm = gw / w_max if w_max > 0 else np.ones_like(gw)
                sizes = grid_size_min + (grid_size_max - grid_size_min) * w_norm
                alphas = grid_alpha_min + (grid_alpha_max - grid_alpha_min) * w_norm
                from matplotlib.colors import to_rgba

                base = np.array(to_rgba(grid_color))
                colors = np.tile(base, (len(gm), 1))
                colors[:, 3] = alphas
                ax.scatter(
                    gm, gp, s=sizes, marker=grid_marker, linewidths=grid_lw,
                    c=colors, zorder=grid_zorder,
                )

    # --- legend (proxy artists; `imshow`/`contourf` aren't in legend by default) ─
    if show_legend:
        from matplotlib import colormaps as _cm
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        items: list = []
        if show_density:
            items.append(Patch(facecolor=_cm[density_cmap](0.7), label=density_label))
        has_grid = grid_measured is not None and grid_predicted is not None
        if has_grid and grid_render in ("density", "both"):
            line_legend_color = (
                grid_density_color if grid_density_color is not None
                else _cm[grid_density_cmap](0.85)
            )
            items.append(Line2D(
                [], [], color=line_legend_color,
                lw=grid_density_linewidth, label=grid_density_label,
            ))
        elif has_grid and grid_render == "crosses":
            items.append(Line2D(
                [], [], color=grid_color, marker=grid_marker, ls="",
                mew=grid_lw, ms=8, label=grid_density_label,
            ))
        if show_identity:
            items.append(Line2D(
                [], [], color=identity_color, ls=identity_ls,
                lw=identity_lw, label=identity_label,
            ))
        if show_trendline and trend_eval is not None:
            items.append(Line2D(
                [], [], color=trendline_color, lw=trendline_lw,
                label=trendline_label,
            ))
        if items:
            kw = dict(
                handles=items, loc=legend_loc, fontsize=legend_fontsize,
                framealpha=0.85, edgecolor="grey",
                borderpad=0.4, handlelength=1.6, handletextpad=0.5,
            )
            if legend_kwargs:
                kw.update(legend_kwargs)
            ax.legend(**kw)

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
    if title:
        ax.set_title(title)

    # --- delta-vs-kernel marginal strip ---
    # Per-data-point density of (measured_latent, model − kernel) in latent.
    # Sits below the cloud, sharing the x-axis. Zero-line = model agrees
    # with the kernel-smoother; spread = structural disagreement. By
    # construction `model_rmse² − kernel_rmse² ≈ ∫ delta² dy`, so this
    # strip is a literal visualization of the metric the stats column
    # reports as `excess`.
    if kernel_predicted is not None:
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        delta = predicted - kernel_predicted
        finite_d = np.isfinite(delta) & np.isfinite(measured)
        delta_f = delta[finite_d]
        measured_f = measured[finite_d]
        if delta_f.size > 0:
            divider = make_axes_locatable(ax)
            delta_ax = divider.append_axes(
                "bottom", size=delta_strip_size, pad=delta_strip_pad,
                sharex=ax,
            )
            d_lo = -float(np.nanquantile(np.abs(delta_f), 0.995)) if delta_strip_ylim is None else delta_strip_ylim[0]
            d_hi = -d_lo if delta_strip_ylim is None else delta_strip_ylim[1]
            if not (np.isfinite(d_lo) and np.isfinite(d_hi)) or d_hi <= d_lo:
                d_lo, d_hi = -0.05, 0.05
            nbins_x = density_res
            nbins_y = max(20, density_res // 4)
            h_d, _xe, _ye = np.histogram2d(
                measured_f, delta_f,
                bins=[nbins_x, nbins_y],
                range=[[lo, hi], [d_lo, d_hi]],
                density=False,
            )
            h_d = np.ma.masked_where(h_d == 0, h_d)
            if delta_strip_log:
                h_d = np.log1p(h_d)
            delta_ax.imshow(
                h_d.T, extent=[lo, hi, d_lo, d_hi],
                origin="lower", aspect="auto",
                cmap=delta_strip_cmap, interpolation="nearest",
            )
            delta_ax.axhline(0, color="grey", lw=0.6, ls="--", alpha=0.7)
            delta_ax.set_xlim(lo, hi)
            delta_ax.set_ylim(d_lo, d_hi)
            delta_ax.set_ylabel("model − kernel", fontsize=6)
            delta_ax.tick_params(axis='y', labelsize=6)
            # Hide the cloud's x-axis (now driven by the marginal below it)
            # and let `ax`'s xlabel turn into the marginal's xlabel.
            ax.set_xlabel("")
            ax.tick_params(axis='x', labelbottom=False)
            delta_ax.set_xlabel(xlabel)
            if rescaler is not None:
                # Reapply the transformed-axis tick labels on the new
                # bottom axis so the x-coords show raw fluo values.
                setup_transformed_axis(
                    delta_ax, xaxis_lims=[lo, hi], yaxis_lims=None, rescaler=rescaler,
                )


# ── noise-floor panel ───────────────────────────────────────────────────


@configurable
def noise_floor_panel(ax, **kwargs):
    """Data-only kernel-smoother MVP twin of :func:`measured_vs_predicted`.

    Thin wrapper over ``measured_vs_predicted`` whose only purpose is to
    carry a *different* name so plot-config's ``measured_vs_predicted_params``
    callstack defaults don't bleed into this panel. Auto-shows RMSE and R²
    (kernel-vs-gt) so the user can compare against the model panel directly.
    """
    defaults = {
        "show_stats": True,
        "show_noise_floor": False,
        "show_trendline": False,
        "show_violins": False,
        "show_bias": False,
        "show_calibration_rms": False,
        "show_spread": False,
        "show_crps": False,
        "show_coverage": False,
        "xlabel": "Measured",
        "ylabel": "Kernel-smoothed mean",
    }
    for k, v in defaults.items():
        kwargs.setdefault(k, v)
    return measured_vs_predicted(ax, **kwargs)
