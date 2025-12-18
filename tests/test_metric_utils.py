"""Comprehensive tests for biocomp.metric_utils.

Tests cover basic regression metrics, grid-based metrics, validation objectives,
edge cases, and defensive programming assertions.
"""

import numpy as np
import pytest

from biocomp.metric_utils import (
    mse,
    rmse,
    mae,
    r_squared,
    pearson_r,
    max_error,
    percentile_error,
    RegressionStats,
    DistributionStats,
    GridStats,
    grid_mse,
    grid_rmse,
    grid_r_squared,
    grid_snr,
    grid_kl_divergence,
    compute_nrmse,
    noise_relative_error,
    extract_metric_values,
    compute_validation_objective,
)


# ─────────────────────────────────────────────────────────────────────────────
# BASIC REGRESSION METRICS
# ─────────────────────────────────────────────────────────────────────────────


class TestMSE:
    def test_perfect_prediction_is_zero(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.isclose(mse(y, y), 0.0, atol=1e-10)

    @pytest.mark.parametrize("seed", [42, 123, 456, 789])
    def test_non_negative(self, seed):
        np.random.seed(seed)
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)
        result = mse(y_true, y_pred)
        assert result >= 0.0

    def test_symmetric(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        assert np.isclose(mse(y_true, y_pred), mse(y_pred, y_true))

    @pytest.mark.parametrize("scale", [0.5, 1.0, 2.0, 5.0])
    def test_scales_quadratically_with_constant_error(self, scale):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred_base = y_true + 1.0  # constant error of 1
        y_pred_scaled = y_true + scale  # constant error of scale
        mse_base = mse(y_true, y_pred_base)
        mse_scaled = mse(y_true, y_pred_scaled)
        expected_ratio = scale**2
        actual_ratio = mse_scaled / mse_base
        assert np.isclose(actual_ratio, expected_ratio, rtol=1e-5)

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])
        expected = ((0.1**2) + (0.1**2) + (0.1**2)) / 3
        assert np.isclose(mse(y_true, y_pred), expected, rtol=1e-5)


class TestRMSE:
    def test_perfect_prediction_is_zero(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.isclose(rmse(y, y), 0.0, atol=1e-10)

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_equals_sqrt_mse(self, seed):
        np.random.seed(seed)
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)
        assert np.isclose(rmse(y_true, y_pred), np.sqrt(mse(y_true, y_pred)), rtol=1e-10)

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])
        expected = np.sqrt(((0.1**2) + (0.1**2) + (0.1**2)) / 3)
        assert np.isclose(rmse(y_true, y_pred), expected, rtol=1e-5)


class TestMAE:
    def test_perfect_prediction_is_zero(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.isclose(mae(y, y), 0.0, atol=1e-10)

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_non_negative(self, seed):
        np.random.seed(seed)
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)
        assert mae(y_true, y_pred) >= 0.0

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_le_rmse(self, seed):
        np.random.seed(seed)
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)
        assert mae(y_true, y_pred) <= rmse(y_true, y_pred) + 1e-10

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 1.5, 3.5])
        expected = (0.5 + 0.5 + 0.5) / 3
        assert np.isclose(mae(y_true, y_pred), expected, rtol=1e-5)


class TestRSquared:
    def test_perfect_prediction_is_one(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.isclose(r_squared(y, y), 1.0, atol=1e-10)

    def test_mean_prediction_is_zero(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.full_like(y, np.mean(y))
        assert np.isclose(r_squared(y, y_pred), 0.0, atol=1e-10)

    def test_constant_input_is_nan(self):
        y_constant = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        y_pred = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        assert np.isnan(r_squared(y_constant, y_pred))

    def test_known_value(self):
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = y_true + 0.1 * np.random.randn(100)
        ss_res = np.sum((y_pred - y_true) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        expected = 1 - ss_res / ss_tot
        assert np.isclose(r_squared(y_true, y_pred), expected, atol=1e-10)


class TestPearsonR:
    def test_perfect_correlation_is_one(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r, p = pearson_r(y, y)
        assert np.isclose(r, 1.0, atol=1e-10)

    def test_negative_correlation_is_minus_one(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r, p = pearson_r(y, -y)
        assert np.isclose(r, -1.0, atol=1e-10)

    @pytest.mark.parametrize("scale,offset", [(0.5, 10.0), (2.0, -5.0), (1.0, 100.0)])
    def test_invariant_to_linear_transform(self, scale, offset):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_scaled = y * scale + offset
        r_orig, _ = pearson_r(y, y)
        r_scaled, _ = pearson_r(y, y_scaled)
        assert np.isclose(r_orig, r_scaled, atol=1e-10)

    def test_too_few_points_returns_nan(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.0, 2.0])
        r, p = pearson_r(y_true, y_pred)
        assert np.isnan(r)


class TestMaxError:
    def test_perfect_prediction_is_zero(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.isclose(max_error(y, y), 0.0, atol=1e-10)

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_ge_mae(self, seed):
        np.random.seed(seed)
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)
        assert max_error(y_true, y_pred) >= mae(y_true, y_pred) - 1e-10

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.5, 3.0])
        assert np.isclose(max_error(y_true, y_pred), 0.5)


class TestPercentileError:
    def test_perfect_prediction_is_zero(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.isclose(percentile_error(y, y, percentile=95.0), 0.0, atol=1e-10)

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_p95_le_max_error(self, seed):
        np.random.seed(seed)
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)
        assert percentile_error(y_true, y_pred, 95.0) <= max_error(y_true, y_pred) + 1e-10

    def test_invalid_percentile_raises(self):
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(AssertionError):
            percentile_error(y, y, percentile=101.0)
        with pytest.raises(AssertionError):
            percentile_error(y, y, percentile=-1.0)


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION STATS DATACLASS
# ─────────────────────────────────────────────────────────────────────────────


class TestRegressionStats:
    def test_perfect_prediction_stats(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = RegressionStats.compute(y, y)
        assert np.isclose(stats.mse, 0.0, atol=1e-10)
        assert np.isclose(stats.rmse, 0.0, atol=1e-10)
        assert np.isclose(stats.mae, 0.0, atol=1e-10)
        assert np.isclose(stats.r2, 1.0, atol=1e-10)
        assert np.isclose(stats.pearson_r, 1.0, atol=1e-10)

    def test_empty_after_nan_filtering(self):
        y_true = np.array([np.nan, np.nan])
        y_pred = np.array([1.0, 2.0])
        stats = RegressionStats.compute(y_true, y_pred)
        assert stats.n_samples == 0
        assert np.isnan(stats.mse)

    def test_n_samples_correct(self):
        y_true = np.array([1.0, np.nan, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, np.nan, 4.0])
        stats = RegressionStats.compute(y_true, y_pred)
        assert stats.n_samples == 2  # only indices 0 and 3 have both valid


class TestDistributionStats:
    def test_basic_stats(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        stats = DistributionStats.compute(y_true, y_pred)
        assert np.isclose(stats.target_mean, 3.0)
        assert np.isclose(stats.target_min, 1.0)
        assert np.isclose(stats.target_max, 5.0)
        assert np.isclose(stats.pred_mean, 4.0)
        assert np.isclose(stats.pred_min, 2.0)
        assert np.isclose(stats.pred_max, 6.0)


# ─────────────────────────────────────────────────────────────────────────────
# GRID-BASED METRICS
# ─────────────────────────────────────────────────────────────────────────────


class TestGridMSE:
    def test_perfect_is_zero(self):
        y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        assert np.isclose(grid_mse(y, y), 0.0, atol=1e-10)

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_non_negative(self, seed):
        np.random.seed(seed)
        yhat = np.random.rand(100)
        gt = np.random.rand(100)
        assert grid_mse(yhat, gt) >= 0.0


class TestGridRMSE:
    def test_perfect_is_zero(self):
        y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        assert np.isclose(grid_rmse(y, y), 0.0, atol=1e-10)

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_equals_sqrt_mse(self, seed):
        np.random.seed(seed)
        yhat = np.random.rand(100)
        gt = np.random.rand(100)
        assert np.isclose(grid_rmse(yhat, gt), np.sqrt(grid_mse(yhat, gt)), rtol=1e-10)


class TestGridRSquared:
    def test_perfect_is_one(self):
        y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        assert np.isclose(grid_r_squared(y, y), 1.0, atol=1e-10)

    def test_constant_gt_is_nan(self):
        yhat = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        gt = np.array([0.3, 0.3, 0.3, 0.3, 0.3])
        assert np.isnan(grid_r_squared(yhat, gt))


class TestGridSNR:
    def test_finite_for_valid_inputs(self):
        np.random.seed(42)
        gt_mean = np.random.rand(100) * 0.5 + 0.25
        local_var = np.random.rand(100) * 0.01 + 0.001
        result = grid_snr(gt_mean, local_var)
        assert np.isfinite(result)

    def test_increases_with_signal_variance(self):
        local_var = np.ones(100) * 0.01
        gt_mean_low = np.random.rand(100) * 0.1 + 0.45  # small variance
        gt_mean_high = np.random.rand(100) * 0.8 + 0.1  # large variance
        snr_low = grid_snr(gt_mean_low, local_var)
        snr_high = grid_snr(gt_mean_high, local_var)
        assert snr_high > snr_low


class TestGridKL:
    def test_identical_is_near_zero(self):
        mean = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        std = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
        kl_mean, kl_sim = grid_kl_divergence(mean, std, mean, std)
        assert kl_mean < 0.1
        assert kl_sim > 90.0


# ─────────────────────────────────────────────────────────────────────────────
# NRMSE COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeNRMSE:
    def test_perfect_prediction_near_zero(self):
        n = 100
        np.random.seed(42)
        gt_mean = np.random.rand(n, 1) * 0.5 + 0.25
        sq_error = np.zeros((n, 1))
        local_var = np.random.rand(n, 1) * 0.1 + 0.01
        n_eff = np.ones((n, 1)) * 50
        global_var = float(np.var(gt_mean))
        global_range = float(np.ptp(gt_mean))
        result = compute_nrmse(sq_error, local_var, n_eff, global_var, global_range, gt_mean, k=100)
        assert result < 0.1

    def test_increases_with_error(self):
        n = 100
        np.random.seed(42)
        gt_mean = np.random.rand(n, 1) * 0.5 + 0.25
        local_var = np.random.rand(n, 1) * 0.01 + 0.001
        n_eff = np.ones((n, 1)) * 50
        global_var = float(np.var(gt_mean))
        global_range = float(np.ptp(gt_mean))

        sq_error_small = np.ones((n, 1)) * 0.001
        nrmse_small = compute_nrmse(
            sq_error_small, local_var, n_eff, global_var, global_range, gt_mean, k=100
        )

        sq_error_large = np.ones((n, 1)) * 0.1
        nrmse_large = compute_nrmse(
            sq_error_large, local_var, n_eff, global_var, global_range, gt_mean, k=100
        )

        assert nrmse_large > nrmse_small

    def test_shape_mismatch_raises(self):
        n = 100
        np.random.seed(42)
        sq_error = np.random.rand(n, 1)
        local_var = np.random.rand(n + 10, 1)  # wrong shape
        n_eff = np.ones((n, 1)) * 50
        gt_mean = np.random.rand(n, 1)

        with pytest.raises(AssertionError, match="shape mismatch"):
            compute_nrmse(sq_error, local_var, n_eff, 0.1, 0.5, gt_mean, k=100)


# ─────────────────────────────────────────────────────────────────────────────
# NOISE-RELATIVE ERROR
# ─────────────────────────────────────────────────────────────────────────────


class TestNoiseRelativeError:
    @pytest.mark.parametrize(
        "grid_nrmse,data_nrmse,expected",
        [
            (0.5, 0.25, 2.0),
            (0.25, 0.5, 0.5),
            (1.0, 1.0, 1.0),
        ],
    )
    def test_is_ratio(self, grid_nrmse, data_nrmse, expected):
        result = noise_relative_error(grid_nrmse, data_nrmse)
        assert np.isclose(result, expected, rtol=1e-10)

    def test_zero_data_nrmse_is_nan(self):
        assert np.isnan(noise_relative_error(0.5, 0.0))

    def test_negative_data_nrmse_is_nan(self):
        assert np.isnan(noise_relative_error(0.5, -0.1))

    def test_nan_inputs_is_nan(self):
        assert np.isnan(noise_relative_error(np.nan, 0.5))
        assert np.isnan(noise_relative_error(0.5, np.nan))


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION OBJECTIVES
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractMetricValues:
    def test_extract_from_valid_stats(self):
        stats = [
            {"rmse": 0.1, "grid_nrmse": 0.5},
            {"rmse": 0.2, "grid_nrmse": 0.6},
            {"rmse": 0.3},
        ]
        result = extract_metric_values(stats, "rmse")
        assert len(result) == 3
        assert np.allclose(result, [0.1, 0.2, 0.3])

    def test_extract_skips_nan(self):
        stats = [
            {"rmse": 0.1},
            {"rmse": np.nan},
            {"rmse": 0.3},
        ]
        result = extract_metric_values(stats, "rmse")
        assert len(result) == 2
        assert np.allclose(result, [0.1, 0.3])

    def test_extract_positive_only(self):
        stats = [
            {"rmse": -0.1},
            {"rmse": 0.2},
            {"rmse": 0.3},
        ]
        result = extract_metric_values(stats, "rmse", positive_only=True)
        assert len(result) == 2
        assert np.allclose(result, [0.2, 0.3])

    def test_missing_key(self):
        stats = [{"other": 0.1}]
        result = extract_metric_values(stats, "rmse")
        assert len(result) == 0


class TestValidationObjectives:
    def test_mean_rmse(self):
        stats = [{"rmse": 0.1}, {"rmse": 0.2}, {"rmse": 0.3}]
        result = compute_validation_objective(stats, "mean_rmse")
        assert np.isclose(result, 0.2, rtol=1e-10)

    def test_geomean_nrmse(self):
        stats = [
            {"grid_nrmse": 0.1},
            {"grid_nrmse": 0.2},
            {"grid_nrmse": 0.4},
        ]
        result = compute_validation_objective(stats, "geomean_nrmse")
        expected = (0.1 * 0.2 * 0.4) ** (1 / 3)
        assert np.isclose(result, expected, rtol=1e-5)

    def test_softmax_nrmse(self):
        stats = [
            {"grid_nrmse": 0.1},
            {"grid_nrmse": 0.5},
            {"grid_nrmse": 0.3},
        ]
        result = compute_validation_objective(stats, "softmax_nrmse", softmax_alpha=5.0)
        assert result > 0.4  # dominated by max but not exactly

    def test_empty_stats_returns_inf(self):
        result = compute_validation_objective([], "mean_rmse")
        assert result == float("inf")

    def test_unknown_objective_raises(self):
        with pytest.raises(AssertionError, match="unknown objective"):
            compute_validation_objective([{"rmse": 0.1}], "unknown_objective")

    def test_geomean_nre(self):
        stats = [
            {"noise_relative_error": 0.8},
            {"noise_relative_error": 1.0},
            {"noise_relative_error": 1.2},
        ]
        result = compute_validation_objective(stats, "geomean_nre")
        expected = (0.8 * 1.0 * 1.2) ** (1 / 3)
        assert np.isclose(result, expected, rtol=1e-5)

    def test_powermean_nre(self):
        stats = [
            {"noise_relative_error": 1.0},
            {"noise_relative_error": 2.0},
            {"noise_relative_error": 3.0},
        ]
        # power mean with p=2 is RMS
        result = compute_validation_objective(stats, "powermean_nre", powermean_p=2.0)
        expected = np.sqrt(np.mean(np.array([1.0, 2.0, 3.0]) ** 2))
        assert np.isclose(result, expected, rtol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# SPACE DETECTION (defensive programming)
# ─────────────────────────────────────────────────────────────────────────────


class TestSpaceDetection:
    def test_latent_space_values_accepted(self):
        y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        y_pred = np.array([0.15, 0.22, 0.28, 0.42, 0.48])
        result = mse(y_true, y_pred)
        assert np.isfinite(result)

    def test_raw_space_values_accepted(self):
        y_true = np.array([1e5, 2e5, 3e5, 4e5, 5e5])
        y_pred = np.array([1.1e5, 2.2e5, 2.8e5, 4.2e5, 4.8e5])
        result = mse(y_true, y_pred)
        assert np.isfinite(result)

    def test_mixed_space_raises_assertion(self):
        y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # latent
        y_pred = np.array([1e5, 2e5, 3e5, 4e5, 5e5])  # raw
        with pytest.raises(AssertionError, match="space mismatch"):
            mse(y_true, y_pred)


# ─────────────────────────────────────────────────────────────────────────────
# EDGE CASES
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_element_arrays(self):
        y_true = np.array([0.5])
        y_pred = np.array([0.5])
        assert np.isclose(mse(y_true, y_pred), 0.0)
        assert np.isclose(rmse(y_true, y_pred), 0.0)
        assert np.isclose(mae(y_true, y_pred), 0.0)

    def test_all_nan_returns_nan(self):
        y_true = np.array([np.nan, np.nan])
        y_pred = np.array([np.nan, np.nan])
        assert np.isnan(mse(y_true, y_pred))
        assert np.isnan(rmse(y_true, y_pred))
        assert np.isnan(mae(y_true, y_pred))

    def test_mixed_nan_filters_correctly(self):
        y_true = np.array([1.0, np.nan, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        result = mse(y_true, y_pred)
        assert np.isclose(result, 0.0)

    def test_inf_values_filtered(self):
        y_true = np.array([1.0, np.inf, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        result = mse(y_true, y_pred)
        assert np.isfinite(result)

    def test_large_arrays_stable(self):
        np.random.seed(42)
        n = 100000
        y_true = np.random.randn(n)
        y_pred = y_true + np.random.randn(n) * 0.1
        result = mse(y_true, y_pred)
        assert np.isfinite(result)
        assert result > 0


class TestLengthMismatch:
    def test_different_lengths_raises(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0])
        with pytest.raises(AssertionError, match="length mismatch"):
            mse(y_true, y_pred)


# ─────────────────────────────────────────────────────────────────────────────
# GRIDSTATS DATACLASS
# ─────────────────────────────────────────────────────────────────────────────


class TestGridStatsDataclass:
    def test_compute_returns_all_fields(self):
        np.random.seed(42)
        n = 100
        yhat_mean = np.random.rand(n)
        yhat_std = np.random.rand(n) * 0.1 + 0.01
        gt_mean = np.random.rand(n)
        gt_std = np.random.rand(n) * 0.1 + 0.01
        n_eff = np.ones(n) * 50

        stats = GridStats.compute(yhat_mean, yhat_std, gt_mean, gt_std, n_eff, k=100)

        assert np.isfinite(stats.grid_mse)
        assert np.isfinite(stats.grid_rmse)
        assert np.isfinite(stats.grid_nrmse)
        assert np.isfinite(stats.grid_snr)
        assert np.isfinite(stats.grid_kl)
        assert np.isfinite(stats.grid_kl_similarity)
        assert np.isfinite(stats.grid_r_squared)

    def test_to_dict(self):
        np.random.seed(42)
        n = 100
        yhat_mean = np.random.rand(n)
        yhat_std = np.random.rand(n) * 0.1 + 0.01
        gt_mean = np.random.rand(n)
        gt_std = np.random.rand(n) * 0.1 + 0.01
        n_eff = np.ones(n) * 50

        stats = GridStats.compute(yhat_mean, yhat_std, gt_mean, gt_std, n_eff, k=100)
        d = stats.to_dict()

        assert "grid_mse" in d
        assert "grid_rmse" in d
        assert "grid_nrmse" in d
        assert "grid_snr" in d
        assert "grid_kl" in d
        assert "grid_kl_similarity" in d
        assert "grid_r_squared" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
