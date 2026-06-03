# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Centralized biocomp metrics: nRMSE, NRE, SNR, grid stats, distributional
objectives, RegressionStats — all the biocomp-domain validation logic.

The elementary kernels (mse, rmse, mae, r_squared, max_error) are re-imported
from jeanplot.stats, the SSOT for the generic formulas. We add the domain layer
(space-mismatch checks, p-value, grids, objectives) on top.

Space conventions:
  - LATENT space: normalized [0, 1] range used by the model
  - RAW space: original fluorescence values (e.g., 1e6)

Grid-based metrics operate in whatever space the inputs are in. For fair comparison
across experiments, use LATENT space. Assertions help detect accidental space mismatches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from scipy import stats as scipy_stats
from pydantic import BaseModel

# SSOT: jeanplot owns the generic NaN-robust kernels; we re-export + build on them.
from jeanplot.stats import (
    mae as mae,
    max_error as max_error,
    mse as mse,
    r_squared as r_squared,
    rmse as rmse,
)


NdArray: TypeAlias = np.ndarray

LATENT_RANGE = (0.0, 1.0)
RAW_FLOOR = 100.0  # if min > this, likely raw space

EPSILON = 1e-9  # for numerical stability in divisions
ROBUST_EPSILON_FRACTION = 0.01  # 1% of data range as minimum denominator


def _to_1d(arr: NdArray) -> NdArray:
    """flatten to 1d, preserving dtype."""
    return np.asarray(arr).ravel()


def _finite_mask(y_true: NdArray, y_pred: NdArray) -> NdArray:
    """mask of positions where both arrays are finite."""
    return np.isfinite(y_true) & np.isfinite(y_pred)


def _validate_same_length(y_true: NdArray, y_pred: NdArray) -> None:
    assert len(y_true) == len(y_pred), (
        f"length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
    )


def _validate_not_empty(arr: NdArray, name: str) -> None:
    assert arr.size > 0, f"{name} is empty"


def _detect_likely_space(arr: NdArray) -> str:
    """heuristic detection of latent vs raw space."""
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return "unknown"
    mn, mx = float(np.min(finite)), float(np.max(finite))
    if mn >= -0.5 and mx <= 1.5:
        return "latent"
    if mn > RAW_FLOOR:
        return "raw"
    return "ambiguous"


def _warn_space_mismatch(y_true: NdArray, y_pred: NdArray) -> None:
    """assert-level warning if spaces look inconsistent."""
    space_true = _detect_likely_space(y_true)
    space_pred = _detect_likely_space(y_pred)
    if space_true != "unknown" and space_pred != "unknown":
        assert space_true == space_pred or "ambiguous" in (space_true, space_pred), (
            f"possible space mismatch: y_true looks {space_true}, y_pred looks {space_pred}. "
            "ensure both are in same space (latent or raw)."
        )


# ─────────────────────────────────────────────────────────────────────────────
# BASIC REGRESSION METRICS
# ─────────────────────────────────────────────────────────────────────────────


def ermse(mse_model: float, mse_floor: float) -> float:
    """excess RMSE: sqrt(max(0, mMSE - kMSE)). Model error above the kernel
    noise floor (kRMSE). Noise-corrected, same units as RMSE. Aggregate over
    groups by averaging in MSE space first, then calling once."""
    return float(np.sqrt(max(0.0, float(mse_model) - float(mse_floor))))


def pearson_r(y_true: NdArray, y_pred: NdArray, *, validate: bool = True) -> tuple[float, float]:
    """pearson correlation coefficient AND p-value ((nan, nan) if <3 valid pairs).
    For just the coefficient, the SSOT is jeanplot.stats.pearson_r."""
    yt, yp = _to_1d(y_true), _to_1d(y_pred)
    if validate:
        _validate_same_length(yt, yp)
    mask = _finite_mask(yt, yp)
    if np.sum(mask) < 3:
        return float("nan"), float("nan")
    r, p = scipy_stats.pearsonr(yt[mask], yp[mask])
    return float(r), float(p)


def percentile_error(
    y_true: NdArray, y_pred: NdArray, percentile: float = 95.0, *, validate: bool = True
) -> float:
    """percentile of absolute errors (e.g., p95 error)."""
    assert 0.0 <= percentile <= 100.0, f"percentile must be in [0, 100], got {percentile}"
    yt, yp = _to_1d(y_true), _to_1d(y_pred)
    if validate:
        _validate_same_length(yt, yp)
    mask = _finite_mask(yt, yp)
    if not np.any(mask):
        return float("nan")
    return float(np.percentile(np.abs(yp[mask] - yt[mask]), percentile))


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION STATS DATACLASS
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RegressionStats:
    """basic regression statistics bundle."""

    mse: float
    rmse: float
    mae: float
    r2: float
    pearson_r: float
    pearson_p: float
    max_error: float
    p95_error: float
    n_samples: int

    @classmethod
    def compute(cls, y_true: NdArray, y_pred: NdArray, *, validate: bool = True) -> RegressionStats:
        """compute all basic regression statistics."""
        yt, yp = _to_1d(y_true), _to_1d(y_pred)
        if validate:
            _validate_same_length(yt, yp)
            _warn_space_mismatch(yt, yp)
        mask = _finite_mask(yt, yp)
        n = int(np.sum(mask))
        if n == 0:
            return cls(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0)
        yt_v, yp_v = yt[mask], yp[mask]
        err = yp_v - yt_v
        abs_err = np.abs(err)
        mse_val = float(np.mean(err**2))
        rmse_val = float(np.sqrt(mse_val))
        mae_val = float(np.mean(abs_err))
        ss_res = np.sum(err**2)
        ss_tot = np.sum((yt_v - np.mean(yt_v)) ** 2)
        r2_val = float(1.0 - ss_res / ss_tot) if ss_tot > EPSILON else float("nan")
        pr, pp = scipy_stats.pearsonr(yt_v, yp_v) if n > 2 else (float("nan"), float("nan"))
        return cls(
            mse=mse_val,
            rmse=rmse_val,
            mae=mae_val,
            r2=r2_val,
            pearson_r=float(pr),
            pearson_p=float(pp),
            max_error=float(np.max(abs_err)),
            p95_error=float(np.percentile(abs_err, 95)),
            n_samples=n,
        )


# ─────────────────────────────────────────────────────────────────────────────
# DISTRIBUTION METRICS
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DistributionStats:
    """distribution statistics for y_true and y_pred."""

    target_mean: float
    target_std: float
    target_min: float
    target_max: float
    pred_mean: float
    pred_std: float
    pred_min: float
    pred_max: float

    @classmethod
    def compute(cls, y_true: NdArray, y_pred: NdArray) -> DistributionStats:
        yt, yp = _to_1d(y_true), _to_1d(y_pred)
        return cls(
            target_mean=float(np.nanmean(yt)),
            target_std=float(np.nanstd(yt)),
            target_min=float(np.nanmin(yt)),
            target_max=float(np.nanmax(yt)),
            pred_mean=float(np.nanmean(yp)),
            pred_std=float(np.nanstd(yp)),
            pred_min=float(np.nanmin(yp)),
            pred_max=float(np.nanmax(yp)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# GRID-BASED METRICS (for KNN-smoothed statistics)
# ─────────────────────────────────────────────────────────────────────────────


def grid_mse(yhat_mean: NdArray, gt_mean: NdArray) -> float:
    """MSE between KNN-smoothed prediction and ground truth means."""
    return float(np.nanmean((yhat_mean - gt_mean) ** 2))


def grid_rmse(yhat_mean: NdArray, gt_mean: NdArray) -> float:
    """RMSE between KNN-smoothed means."""
    return float(np.sqrt(grid_mse(yhat_mean, gt_mean)))


def grid_r_squared(yhat_mean: NdArray, gt_mean: NdArray) -> float:
    """R² for KNN-smoothed means."""
    mse_val = grid_mse(yhat_mean, gt_mean)
    gt_var = np.nanvar(gt_mean)
    if gt_var < EPSILON:
        return float("nan")
    return float(1.0 - mse_val / (gt_var + EPSILON))


def grid_snr(gt_mean: NdArray, local_var: NdArray, *, epsilon: float = EPSILON) -> float:
    """SNR (dB) = 10·log10(var(local means around global mean) / mean(local var))."""
    local_var_safe = np.maximum(local_var, epsilon)
    avg_noise = float(np.nanmean(local_var_safe))
    avg_signal = float(np.nanmean((gt_mean - np.nanmean(gt_mean)) ** 2))
    return float(10.0 * np.log10((avg_signal / avg_noise) + epsilon))


def grid_kl_divergence(
    yhat_mean: NdArray,
    yhat_std: NdArray,
    gt_mean: NdArray,
    gt_std: NdArray,
    *,
    epsilon: float = EPSILON,
) -> tuple[float, float]:
    """KL divergence between prediction and ground-truth Gaussians.

    Returns (mean KL across grid, similarity = mean(exp(-kl)) * 100).
    """
    yhat_std_safe = np.maximum(yhat_std, epsilon)
    gt_std_safe = np.maximum(gt_std, epsilon)
    log_term = np.log(yhat_std_safe / gt_std_safe)
    num_term = gt_std_safe**2 + (gt_mean - yhat_mean) ** 2
    denom_term = 2.0 * yhat_std_safe**2
    kl = np.maximum(log_term + num_term / denom_term - 0.5, 0.0)
    kl_mean = float(np.nanmean(kl))
    kl_sim = float(np.nanmean(np.exp(-kl))) * 100.0
    return kl_mean, kl_sim


def compute_nrmse(
    sq_error: NdArray,
    local_var: NdArray,
    n_eff: NdArray,
    global_var: float,
    global_range: float,
    gt_mean: NdArray,
    *,
    prior_strength: float = 100.0,
    rel_tolerance: float = 0.05,
    abs_tolerance: float = 0.01,
    weight_cap_fraction: float = 0.1,
    k: int = 1024,
) -> float:
    """Bayesian-smoothed nRMSE: error / local_variance (fair across noise levels).

    Interpretation: ~0 perfect, ~1 error = local noise, >1 worse than predicting local mean.
    """
    assert sq_error.shape == local_var.shape == n_eff.shape == gt_mean.shape, (
        f"shape mismatch: sq_error={sq_error.shape}, local_var={local_var.shape}, "
        f"n_eff={n_eff.shape}, gt_mean={gt_mean.shape}"
    )
    robust_eps = max(abs_tolerance, ROBUST_EPSILON_FRACTION * global_range)

    smoothed_var = (local_var * n_eff + global_var * prior_strength) / (n_eff + prior_strength)
    tolerance_var = (rel_tolerance * np.abs(gt_mean) + abs_tolerance) ** 2
    denom_var = smoothed_var + tolerance_var
    safe_denom = np.maximum(np.sqrt(denom_var), robust_eps)

    weight_cap = weight_cap_fraction * k
    capped_weights = np.minimum(n_eff, weight_cap)

    norm_sq_err = sq_error / (safe_denom**2)
    weights_flat = capped_weights.flatten()
    norm_sq_flat = norm_sq_err.flatten()
    mask = np.isfinite(norm_sq_flat) & (weights_flat > 0)

    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.average(norm_sq_flat[mask], weights=weights_flat[mask])))


def compute_nrmse_pointwise(
    gt: NdArray,
    pred: NdArray,
    sigma_local: NdArray,
    *,
    gt_mean_local: NdArray | None = None,
    global_range: float = 1.0,
    rel_tolerance: float = 0.05,
    abs_tolerance: float = 0.01,
) -> float:
    """Per-point nRMSE: each residual normalized by σ_local² + (rel·|μ_local|+abs)².

    `gt_mean_local` defaults to `gt`; pass kernel-smoothed mean for a smoother tolerance.
    """
    if gt_mean_local is None:
        gt_mean_local = gt
    sq_error = (np.asarray(gt) - np.asarray(pred)) ** 2
    robust_eps = max(abs_tolerance, ROBUST_EPSILON_FRACTION * global_range)
    tolerance_var = (rel_tolerance * np.abs(gt_mean_local) + abs_tolerance) ** 2
    denom_var = np.asarray(sigma_local) ** 2 + tolerance_var
    safe_denom = np.maximum(np.sqrt(denom_var), robust_eps)
    norm_sq = sq_error / (safe_denom ** 2)
    finite = np.isfinite(norm_sq)
    if not finite.any():
        return float("nan")
    return float(np.sqrt(np.mean(norm_sq[finite])))


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION OBJECTIVES (for hyperopt)
# ─────────────────────────────────────────────────────────────────────────────


def extract_metric_values(stats: list[dict], key: str, *, positive_only: bool = False) -> NdArray:
    """extract finite metric values from list of stat dicts."""
    vals = [
        s[key]
        for s in stats
        if key in s
        and s[key] is not None
        and np.isfinite(s[key])
        and (not positive_only or s[key] > 0)
    ]
    return np.array(vals) if vals else np.array([])


def objective_mean_rmse(stats: list[dict]) -> float:
    """simple mean RMSE across networks."""
    vals = extract_metric_values(stats, "rmse")
    return float(np.mean(vals)) if len(vals) else float("inf")


def objective_softmax_nrmse(stats: list[dict], alpha: float = 5.0) -> float:
    """soft-max (LogSumExp) of grid_nrmse - focuses on worst performer."""
    vals = extract_metric_values(stats, "grid_nrmse")
    if len(vals) == 0:
        vals = extract_metric_values(stats, "rmse")
        if len(vals) == 0:
            return float("inf")
    mx = np.max(vals)
    return float(mx + np.log(np.sum(np.exp(alpha * (vals - mx)))) / alpha)


def objective_geomean_nrmse(stats: list[dict]) -> float:
    """geometric mean of grid_nrmse - balanced between mean and worst-case."""
    from scipy.stats import gmean

    vals = extract_metric_values(stats, "grid_nrmse", positive_only=True)
    if len(vals) == 0:
        vals = extract_metric_values(stats, "rmse")
        if len(vals) == 0:
            return float("inf")
        return float(np.mean(vals))
    return float(gmean(vals))


def compute_validation_objective(
    stats: list[dict],
    objective: str,
    *,
    softmax_alpha: float = 5.0,
) -> float:
    """compute validation loss; `objective` is one of 'mean_rmse' | 'softmax_nrmse' | 'geomean_nrmse'."""
    objectives = {
        "mean_rmse": lambda: objective_mean_rmse(stats),
        "softmax_nrmse": lambda: objective_softmax_nrmse(stats, softmax_alpha),
        "geomean_nrmse": lambda: objective_geomean_nrmse(stats),
    }
    assert objective in objectives, (
        f"unknown objective '{objective}'. valid: {list(objectives.keys())}"
    )
    return objectives[objective]()


# ─────────────────────────────────────────────────────────────────────────────
# GRID STATS BUNDLE (full computation from KNN-smoothed data)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GridStats:
    """bundle of grid-based metrics from KNN-smoothed statistics."""

    grid_gt_var: float
    grid_mse: float
    grid_rmse: float
    grid_nrmse: float
    grid_snr: float
    grid_kl: float
    grid_kl_similarity: float
    grid_r_squared: float

    @classmethod
    def compute(
        cls,
        yhat_mean: NdArray,
        yhat_std: NdArray,
        gt_mean: NdArray,
        gt_std: NdArray,
        n_eff: NdArray,
        *,
        k: int = 1024,
    ) -> GridStats:
        """compute all grid statistics from KNN-smoothed (yhat_*, gt_*, n_eff)."""
        assert yhat_mean.shape == yhat_std.shape == gt_mean.shape == gt_std.shape, (
            f"shape mismatch in grid stats inputs: yhat_mean={yhat_mean.shape}, yhat_std={yhat_std.shape}, "
            f"gt_mean={gt_mean.shape}, gt_std={gt_std.shape}"
        )

        sq_error = (yhat_mean - gt_mean) ** 2
        local_var = gt_std**2
        global_var = float(np.nanvar(gt_mean))
        global_range = float(np.nanmax(gt_mean) - np.nanmin(gt_mean))

        mse_val = grid_mse(yhat_mean, gt_mean)
        rmse_val = grid_rmse(yhat_mean, gt_mean)
        r2_val = grid_r_squared(yhat_mean, gt_mean)
        snr_val = grid_snr(gt_mean, local_var)
        kl_mean, kl_sim = grid_kl_divergence(yhat_mean, yhat_std, gt_mean, gt_std)
        nrmse_val = compute_nrmse(
            sq_error, local_var, n_eff, global_var, global_range, gt_mean, k=k
        )

        return cls(
            grid_gt_var=global_var,
            grid_mse=mse_val,
            grid_rmse=rmse_val,
            grid_nrmse=nrmse_val,
            grid_snr=snr_val,
            grid_kl=kl_mean,
            grid_kl_similarity=kl_sim,
            grid_r_squared=r2_val,
        )

    def to_dict(self) -> dict:
        """convert to dict for JSON serialization."""
        return {
            "grid_gt_var": self.grid_gt_var,
            "grid_mse": self.grid_mse,
            "grid_rmse": self.grid_rmse,
            "grid_nrmse": self.grid_nrmse,
            "grid_snr": self.grid_snr,
            "grid_kl": self.grid_kl,
            "grid_kl_similarity": self.grid_kl_similarity,
            "grid_r_squared": self.grid_r_squared,
        }


# nre = grid_nrmse / data_nrmse. normalizes for scale + heteroscedasticity.
# k=64 gives ~4x higher noise floor than k=1024
DEFAULT_GRIDSTATS_PARAMS: dict = {
    "hypercube_res": 64,
    "hypercube_min": 0.0,
    "hypercube_max": 0.7,
    "k": 256,
    "radius": 0.1,
    "sigma_in_radius": 3.0,
    "min_points": 20,
    # Density-balanced subsample (same selection logic as training batches)
    # used to evaluate `model_rmse_latent` / `kernel_rmse_latent`. Removes
    # the heavy weighting of dense low-fluo regions (untransfected cells),
    # so the model/kernel ratio reflects predictor quality across the
    # whole cube view rather than a transfection-efficiency proxy.
    "subsample_n": 64000,
    "subsample_knn_k": 64,
    "subsample_density_quantile": 0.025,
}

# ─────────────────────────────────────────────────────────────────────────────
# GRIDSTATS CONFIG MIXIN
# ─────────────────────────────────────────────────────────────────────────────


class GridStatsFields(BaseModel):
    """Mixin providing gridstats configuration fields for Pydantic models.

    This is NOT a BaseModel subclass - it provides field annotations that Pydantic
    will pick up when combined with a BaseModel class via multiple inheritance.

    Use `get_gridstats_params()` to reconstruct the params dict.

    Example:
        class MyPredictor(GridStatsFields, BaseModel):
            enable_gridstats: bool = True

        pred = MyPredictor(gridstats_k=128)
        params = pred.get_gridstats_params()
    """

    gridstats_hypercube_res: int = DEFAULT_GRIDSTATS_PARAMS["hypercube_res"]
    gridstats_hypercube_min: float = DEFAULT_GRIDSTATS_PARAMS["hypercube_min"]
    gridstats_hypercube_max: float = DEFAULT_GRIDSTATS_PARAMS["hypercube_max"]
    gridstats_k: int = DEFAULT_GRIDSTATS_PARAMS["k"]
    gridstats_radius: float = DEFAULT_GRIDSTATS_PARAMS["radius"]
    gridstats_sigma_in_radius: float = DEFAULT_GRIDSTATS_PARAMS["sigma_in_radius"]
    gridstats_min_points: int = DEFAULT_GRIDSTATS_PARAMS["min_points"]
    gridstats_subsample_n: int = DEFAULT_GRIDSTATS_PARAMS["subsample_n"]
    gridstats_subsample_knn_k: int = DEFAULT_GRIDSTATS_PARAMS["subsample_knn_k"]
    gridstats_subsample_density_quantile: float = DEFAULT_GRIDSTATS_PARAMS[
        "subsample_density_quantile"
    ]

    def get_gridstats_params(self) -> dict[str, Any]:
        """Build gridstats params dict from fields."""
        return {
            "hypercube_res": self.gridstats_hypercube_res,
            "hypercube_min": self.gridstats_hypercube_min,
            "hypercube_max": self.gridstats_hypercube_max,
            "k": self.gridstats_k,
            "radius": self.gridstats_radius,
            "sigma_in_radius": self.gridstats_sigma_in_radius,
            "min_points": self.gridstats_min_points,
            "subsample_n": self.gridstats_subsample_n,
            "subsample_knn_k": self.gridstats_subsample_knn_k,
            "subsample_density_quantile": self.gridstats_subsample_density_quantile,
        }


# alias for backwards compatibility
GridStatsConfigMixin = GridStatsFields
