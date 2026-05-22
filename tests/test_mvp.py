# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Tests for measured_vs_predicted scatter plot rendering."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from scipy import stats

from jeanplot.plots.mvp import (
    measured_vs_predicted,
    _clean_paired,
    _axis_lims,
    _pit_from_samples,
)


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def test_perfect_predictor(ax):
    rng = np.random.default_rng(0)
    measured = rng.uniform(0, 0.6, 500)
    predicted = measured.copy()
    measured_vs_predicted(ax, measured, predicted)
    texts = [t for t in ax.texts if "R²" in t.get_text()]
    assert len(texts) == 1
    assert "1.0000" in texts[0].get_text()


def test_noisy_predictor(ax):
    rng = np.random.default_rng(42)
    measured = rng.uniform(0, 0.6, 1000)
    predicted = 0.8 * measured + 0.1 + rng.normal(0, 0.02, 1000)
    measured_vs_predicted(ax, measured, predicted)
    texts = [t for t in ax.texts if "RMSE" in t.get_text()]
    assert len(texts) == 1


def test_nan_handling(ax):
    measured = np.array([0.1, 0.2, np.nan, 0.4, 0.5])
    predicted = np.array([0.1, np.nan, 0.3, 0.4, 0.5])
    measured_vs_predicted(ax, measured, predicted)


def test_identity_line_present(ax):
    measured = np.linspace(0, 0.6, 100)
    predicted = measured + 0.05
    measured_vs_predicted(
        ax, measured, predicted, show_density=False, show_trendline=False, show_stats=False
    )
    lines = ax.get_lines()
    assert any(np.allclose(l.get_xdata(), l.get_ydata()) for l in lines), "identity line missing"


def test_density_off(ax):
    measured = np.linspace(0, 0.6, 100)
    predicted = measured
    measured_vs_predicted(ax, measured, predicted, show_density=False)
    assert len(ax.images) == 0


def test_density_on(ax):
    rng = np.random.default_rng(0)
    measured = rng.uniform(0, 0.6, 500)
    predicted = measured + rng.normal(0, 0.05, 500)
    measured_vs_predicted(ax, measured, predicted, show_density=True, show_trendline=False)
    assert len(ax.images) == 1


def test_no_stats(ax):
    measured = np.linspace(0, 0.5, 50)
    predicted = measured
    measured_vs_predicted(
        ax, measured, predicted, show_stats=False, show_density=False, show_trendline=False
    )
    assert len(ax.texts) == 0


def test_clean_paired():
    m = np.array([1.0, np.nan, 3.0, np.inf])
    p = np.array([1.0, 2.0, np.nan, 4.0])
    mc, pc = _clean_paired(m, p)
    assert len(mc) == 1
    assert mc[0] == 1.0


def test_axis_lims_auto():
    m = np.array([0.1, 0.5, 0.9])
    p = np.array([0.2, 0.6, 0.8])
    lo, hi = _axis_lims(m, p, (None, None), 0.0)
    assert lo == pytest.approx(0.1)
    assert hi == pytest.approx(0.9)


def test_axis_lims_override():
    m = np.array([0.1, 0.5, 0.9])
    p = np.array([0.2, 0.6, 0.8])
    lo, hi = _axis_lims(m, p, (0.0, 1.0), 0.0)
    assert lo == 0.0
    assert hi == 1.0


def test_residual_panel(ax):
    """Residual std panel draws lines on the provided axis."""
    fig, (main_ax, res_ax) = plt.subplots(2, 1, height_ratios=[3, 1])
    rng = np.random.default_rng(0)
    measured = rng.uniform(0.05, 0.65, 2000)
    predicted = measured + rng.normal(0, 0.03, 2000)
    measured_vs_predicted(
        main_ax, measured, predicted, show_density=False, residual_ax=res_ax,
    )
    assert len(res_ax.lines) >= 2, "residual panel should have std + bias lines"
    assert len(res_ax.collections) >= 1, "residual panel should have fill"
    plt.close(fig)


def test_knn_trendline_near_diagonal(ax):
    """Trendline on y=x data should be near-diagonal (knn_stats under the hood)."""
    rng = np.random.default_rng(0)
    measured = rng.uniform(0.05, 0.65, 2000)
    predicted = measured.copy()
    measured_vs_predicted(
        ax, measured, predicted, show_density=False, show_stats=False, show_identity=False,
    )
    line = ax.get_lines()[0]
    xdata, ydata = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
    # check interior only (edges have few neighbors)
    interior = np.isfinite(ydata) & (xdata >= 0.1) & (xdata <= 0.6)
    assert np.allclose(xdata[interior], ydata[interior], atol=0.02)


# ── sample-based diagnostics ────────────────────────────────────────────


def _make_heteroscedastic(rng, n=5000):
    x = rng.uniform(0.05, 0.6, n)
    sigma = 0.02 + 0.15 * x
    y = x + rng.normal(0, sigma)
    return x, y, sigma


def _draw_samples(rng, x, n_samples=500):
    """Draw from the true heteroscedastic distribution."""
    sigma = 0.02 + 0.15 * x
    return x[None, :] + rng.normal(0, 1, (n_samples, len(x))) * sigma[None, :]


def test_pit_from_samples_calibrated():
    rng = np.random.default_rng(42)
    N = 10_000
    x, y, _ = _make_heteroscedastic(rng, N)
    samples = _draw_samples(rng, x, n_samples=2000)
    pit = _pit_from_samples(y, samples)
    ks_stat, p_value = stats.kstest(pit, "uniform")
    assert p_value > 0.01, f"PIT should be uniform for true model, KS p={p_value:.4f}"


def test_pit_from_samples_misspecified():
    rng = np.random.default_rng(42)
    N = 10_000
    x, y, _ = _make_heteroscedastic(rng, N)
    global_sigma = np.std(y - x)
    wrong_samples = x[None, :] + rng.normal(0, global_sigma, (2000, N))
    pit = _pit_from_samples(y, wrong_samples)
    ks_stat, p_value = stats.kstest(pit, "uniform")
    assert p_value < 0.01, f"PIT should NOT be uniform for wrong model, KS p={p_value:.4f}"


def test_pit_inset_auto_from_samples(ax):
    rng = np.random.default_rng(0)
    x, y, _ = _make_heteroscedastic(rng, 2000)
    samples = _draw_samples(rng, x, 200)
    measured_vs_predicted(ax, x, y, model_samples=samples, show_density=False)
    assert len(ax.child_axes) == 1, "PIT inset should be auto-created from model_samples"


def test_pit_inset_explicit_overrides_samples(ax):
    rng = np.random.default_rng(0)
    x, y, _ = _make_heteroscedastic(rng, 2000)
    samples = _draw_samples(rng, x, 200)
    pit_manual = np.full(len(x), 0.5)
    measured_vs_predicted(
        ax, x, y, model_samples=samples, pit_values=pit_manual, show_density=False,
    )
    assert len(ax.child_axes) == 1


def test_sample_bands_drawn(ax):
    rng = np.random.default_rng(0)
    x, y, _ = _make_heteroscedastic(rng, 5000)
    samples = _draw_samples(rng, x, 500)
    measured_vs_predicted(
        ax, x, y, model_samples=samples,
        show_density=False, model_bands=[[25, 75]],
    )
    dashed = [l for l in ax.get_lines() if l.get_linestyle() == "--" and l.get_color() == "#d62728"]
    assert len(dashed) == 2, f"expected 2 sample band lines, got {len(dashed)}"


def test_sample_coverage_annotation(ax):
    rng = np.random.default_rng(0)
    x, y, _ = _make_heteroscedastic(rng, 5000)
    samples = _draw_samples(rng, x, 500)
    measured_vs_predicted(
        ax, x, y, model_samples=samples,
        show_density=False, show_stats=False, model_bands=[[25, 75]],
    )
    coverage_texts = [t for t in ax.texts if "expect" in t.get_text()]
    assert len(coverage_texts) == 1


def test_sample_coverage_close_to_nominal():
    rng = np.random.default_rng(42)
    x, y, _ = _make_heteroscedastic(rng, 20_000)
    samples = _draw_samples(rng, x, 2000)
    q_lo = np.quantile(samples, 0.25, axis=0)
    q_hi = np.quantile(samples, 0.75, axis=0)
    within = (y >= q_lo) & (y <= q_hi)
    coverage = within.mean()
    assert abs(coverage - 0.50) < 0.03, f"coverage={coverage:.3f}, expected ~0.50"
