# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Tests for the density-balanced sampling SSOT helpers in biocomp.datautils."""

import numpy as np
import pytest

from biocomp.datautils import (
    _selection_proba_from_threshold,
    compute_selection_probabilities,
    density_balanced_indices,
    density_balanced_selection_proba,
    sample_batches,
    sample_batches_w_coord_threshold,
)


def test_selection_proba_normalises_to_one():
    densities = np.array([1.0, 2.0, 4.0, 8.0])
    p = _selection_proba_from_threshold(densities, threshold=2.0)
    assert pytest.approx(p.sum(), abs=1e-12) == 1.0
    assert (p >= 0).all()


def test_selection_proba_high_density_damped():
    """Above the threshold, selection probability should fall as 1/density."""
    densities = np.array([1.0, 2.0, 4.0, 8.0])
    p = _selection_proba_from_threshold(densities, threshold=2.0)
    # densities[0] (=1.0) is below threshold -> full mass; the bigger ones are damped.
    # Ratio between consecutive damped points should reflect the density ratio.
    p_norm = p / p[0]
    assert pytest.approx(p_norm[2] / p_norm[3], rel=1e-6) == 2.0  # 1/4 vs 1/8 -> ratio 2


def test_selection_proba_handles_nan_densities():
    densities = np.array([1.0, float("nan"), 3.0, np.inf])
    p = _selection_proba_from_threshold(densities, threshold=1.0)
    assert pytest.approx(p.sum(), abs=1e-12) == 1.0
    assert np.isfinite(p).all()


def test_selection_proba_zero_total_falls_back_uniform():
    densities = np.array([np.inf, np.inf, np.inf])
    p = _selection_proba_from_threshold(densities, threshold=1.0)
    assert pytest.approx(p.sum(), abs=1e-12) == 1.0
    np.testing.assert_allclose(p, [1 / 3, 1 / 3, 1 / 3])


def test_density_balanced_selection_proba_quantile_threshold():
    densities = np.linspace(1.0, 10.0, 100)
    p = density_balanced_selection_proba(densities, density_threshold_quantile=0.05)
    assert pytest.approx(p.sum(), abs=1e-12) == 1.0
    # Lowest densities should get more weight than highest.
    assert p[:5].mean() > p[-5:].mean() * 4


def test_density_balanced_indices_shape_and_replace():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (1000, 2))
    idx = density_balanced_indices(X, n_samples=200, knn_k=32, seed=0)
    assert idx.shape == (200,)
    assert idx.dtype == np.intp
    assert (idx >= 0).all() and (idx < 1000).all()


def test_density_balanced_indices_deterministic_with_seed():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (500, 2)).astype(np.float32)
    idx1 = density_balanced_indices(X, n_samples=100, knn_k=32, seed=42)
    idx2 = density_balanced_indices(X, n_samples=100, knn_k=32, seed=42)
    idx3 = density_balanced_indices(X, n_samples=100, knn_k=32, seed=43)
    np.testing.assert_array_equal(idx1, idx2)
    assert not np.array_equal(idx1, idx3)


def test_density_balanced_indices_accepts_precomputed_densities():
    X = np.zeros((100, 2), dtype=np.float32)  # X is unused when densities provided
    densities = np.linspace(1.0, 10.0, 100)
    idx = density_balanced_indices(
        X, n_samples=50, densities=densities, density_threshold_quantile=0.05, seed=0,
    )
    assert idx.shape == (50,)
    # Low-density points should appear more often than high-density ones.
    counts = np.bincount(idx, minlength=100)
    assert counts[:20].sum() > counts[80:].sum()


def test_density_balanced_indices_accepts_precomputed_proba():
    X = np.zeros((10, 2))
    p = np.zeros(10)
    p[0] = 1.0
    idx = density_balanced_indices(X, n_samples=20, selection_proba=p, seed=0)
    np.testing.assert_array_equal(idx, np.zeros(20, dtype=np.intp))


def test_density_balanced_indices_empty_input():
    idx = density_balanced_indices(np.zeros((0, 2)), n_samples=10)
    assert idx.shape == (0,)
    assert idx.dtype == np.intp


def test_density_balanced_indices_zero_samples():
    X = np.random.rand(100, 2).astype(np.float32)
    idx = density_balanced_indices(X, n_samples=0)
    assert idx.shape == (0,)


def test_density_balanced_indices_handles_exact_duplicates():
    """Exact duplicates inflate kNN density (~1/eps^d) but stay finite, so
    selection_proba rounds to ~0 and duplicates are effectively excluded -- the
    same outcome we'd want from a dedup pass, without the bookkeeping."""
    rng = np.random.default_rng(0)
    n_dup = 50
    X = np.vstack([
        np.tile([[0.5, 0.5]], (n_dup, 1)),
        rng.uniform(0, 0.7, (450, 2)),
    ]).astype(np.float32)
    idx = density_balanced_indices(X, n_samples=300, knn_k=10, seed=0)
    assert idx.shape == (300,)
    assert np.all(np.isfinite(idx))
    # Duplicates should be heavily under-represented (ideally zero).
    duplicates_picked = (idx < n_dup).sum()
    assert duplicates_picked < 5


def test_density_balanced_concentrates_in_low_density():
    """End-to-end: a tight high-density cluster should be down-weighted."""
    rng = np.random.default_rng(0)
    cluster = rng.normal(0.1, 0.01, (8000, 2)).astype(np.float32)
    spread = rng.uniform(0, 1, (2000, 2)).astype(np.float32)
    X = np.vstack([cluster, spread])
    idx = density_balanced_indices(X, n_samples=2000, knn_k=64, seed=0)
    # The cluster occupies 80% of raw points; after balancing it should get
    # far less than 80% of the subsample.
    cluster_mask = idx < 8000
    assert cluster_mask.mean() < 0.5


def test_compute_selection_probabilities_matches_quantile_only():
    densities = np.linspace(1.0, 10.0, 200)
    p1 = density_balanced_selection_proba(densities, 0.05)
    p2 = compute_selection_probabilities(
        np.zeros((200, 2)), densities, 0.05, None, np.zeros((2, 10)), 0.02,
    )
    np.testing.assert_allclose(p1, p2)


def test_sample_batches_uses_density_balanced_indices():
    """Training-side wrapper produces same indices as the SSOT helper."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (500, 2)).astype(np.float32)
    Y = rng.uniform(0, 1, (500, 1)).astype(np.float32)
    densities = rng.uniform(1, 10, 500)

    Xb, Yb = sample_batches((X, Y, 50, 4, densities, 0.05, 7))
    assert Xb.shape == (4, 50, 2)
    assert Yb.shape == (4, 50, 1)

    expected_idx = density_balanced_indices(
        X, n_samples=200, densities=densities, density_threshold_quantile=0.05, seed=7,
    )
    np.testing.assert_array_equal(
        Xb.reshape(-1, 2), X[expected_idx],
    )
    np.testing.assert_array_equal(
        Yb.reshape(-1, 1), Y[expected_idx],
    )


def test_sample_batches_w_coord_threshold_runs():
    from scipy.stats import gaussian_kde

    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (500, 2)).astype(np.float32)
    Y = rng.uniform(0, 1, (500, 1)).astype(np.float32)
    densities = rng.uniform(1, 10, 500)
    kde = gaussian_kde(X.T)
    Xb, Yb = sample_batches_w_coord_threshold(
        X, Y, 50, 4, kde, densities, None,
        density_threshold_quantile=0.05, density_threshold_coords=0.3,
    )
    assert Xb.shape == (4, 50, 2)
    assert Yb.shape == (4, 50, 1)
