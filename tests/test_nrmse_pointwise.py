"""Tests for `compute_nrmse_pointwise` (subsample-level local-σ nRMSE)."""

from __future__ import annotations

import numpy as np
import pytest

from biocomp.metric_utils import compute_nrmse_pointwise


def test_zero_for_perfect_prediction():
    rng = np.random.default_rng(0)
    gt = rng.uniform(0, 1, 200)
    sigma = rng.uniform(0.1, 0.3, 200)
    assert compute_nrmse_pointwise(gt, gt, sigma, global_range=1.0) == pytest.approx(0.0)


def test_one_when_error_equals_local_sigma():
    """When |gt - pred| = σ_local everywhere AND tolerance ≪ σ, nrmse → 1."""
    sigma = np.full(500, 0.2)
    gt = np.full(500, 0.5)  # constant gt → tolerance term dominated by abs_tol
    pred = gt + sigma
    # With σ=0.2 and tol²≈(0.05*0.5+0.01)² ≈ 0.0012, the denom is ≈ √(0.04+0.0012) ≈ 0.203.
    # nrmse ≈ 0.2 / 0.203 ≈ 0.985.
    val = compute_nrmse_pointwise(gt, pred, sigma, global_range=1.0)
    assert val == pytest.approx(0.985, abs=0.01)


def test_local_sigma_weighting():
    """Identical absolute residuals weigh more in low-σ regions."""
    n = 1000
    gt = np.zeros(n)
    pred = np.full(n, 0.1)  # constant 0.1 absolute error
    sigma_uniform = np.full(n, 0.1)
    sigma_split = np.where(np.arange(n) < n // 2, 0.05, 0.20)
    nrmse_uniform = compute_nrmse_pointwise(gt, pred, sigma_uniform, global_range=1.0)
    nrmse_split = compute_nrmse_pointwise(gt, pred, sigma_split, global_range=1.0)
    # The split-σ case has half the points at σ=0.05 → contribute 4× more to mean(sq_norm).
    # → split nrmse > uniform nrmse.
    assert nrmse_split > nrmse_uniform


def test_finite_when_sigma_zero_thanks_to_tolerance():
    """σ=0 must NOT blow up — the tolerance floor catches it."""
    gt = np.full(100, 0.5)
    pred = gt + 0.01
    sigma = np.zeros(100)
    val = compute_nrmse_pointwise(gt, pred, sigma, global_range=1.0)
    assert np.isfinite(val)
    assert val > 0


def test_handles_nan_residuals():
    gt = np.array([0.1, 0.2, np.nan, 0.4, 0.5])
    pred = np.array([0.1, 0.2, 0.3, np.nan, 0.5])
    sigma = np.full(5, 0.1)
    val = compute_nrmse_pointwise(gt, pred, sigma, global_range=1.0)
    assert val == pytest.approx(0.0, abs=1e-6)  # remaining 3 points are perfect


def test_all_nan_returns_nan():
    gt = np.array([np.nan, np.nan, np.nan])
    pred = np.array([np.nan, np.nan, np.nan])
    sigma = np.full(3, 0.1)
    assert np.isnan(compute_nrmse_pointwise(gt, pred, sigma, global_range=1.0))


def test_multi_output_shape():
    """Per-point arrays may be (N, n_outs); residuals broadcast naturally."""
    rng = np.random.default_rng(0)
    gt = rng.uniform(0, 1, (200, 2))
    pred = gt + rng.normal(0, 0.05, gt.shape)
    sigma = np.full(gt.shape, 0.1)
    val = compute_nrmse_pointwise(gt, pred, sigma, global_range=1.0)
    assert np.isfinite(val) and val > 0


def test_gt_mean_local_default_is_gt():
    """Omitting gt_mean_local must give same answer as passing gt itself."""
    rng = np.random.default_rng(0)
    gt = rng.uniform(0, 1, 200)
    pred = gt + 0.05
    sigma = np.full(200, 0.1)
    a = compute_nrmse_pointwise(gt, pred, sigma)
    b = compute_nrmse_pointwise(gt, pred, sigma, gt_mean_local=gt)
    assert a == pytest.approx(b)
