"""Tests for ASCII/text plotting functionality."""

import numpy as np
import pytest
from biocomp.plotting.ascii_heatmap import (
    heatmap,
    heatmap_with_labels,
    heatmap_bigram,
    _resample_nearest,
    _resample_mean,
)
from biocomp.plotting.plotting_txt import (
    TextPlotResult,
    smooth_1d_txt,
    smooth_2d_txt,
    get_txt_plot_function,
)


class TestAsciiHeatmap:
    def test_heatmap_basic(self):
        data = np.random.rand(10, 10)
        result = heatmap(data)
        assert isinstance(result, str)
        assert len(result) > 0
        lines = result.split("\n")
        assert len(lines) > 10

    def test_heatmap_with_vmin_vmax(self):
        data = np.array([[0, 0.5], [0.5, 1.0]])
        result = heatmap(data, vmin=0, vmax=1, xres=4, yres=2)
        assert isinstance(result, str)
        assert " " in result or "░" in result or "▒" in result

    def test_heatmap_handles_nan(self):
        data = np.array([[0.0, np.nan], [0.5, 1.0]])
        result = heatmap(data, vmin=0, vmax=1, xres=4, yres=2)
        assert isinstance(result, str)

    def test_heatmap_with_labels(self):
        data = np.random.rand(10, 10)
        result = heatmap_with_labels(
            data,
            title="Test Title",
            xlabel="X Axis",
            ylabel="Y Axis",
            vmin=0,
            vmax=1,
        )
        assert "Test Title" in result
        assert "X Axis" in result
        assert "Y Axis" in result

    def test_heatmap_bigram_mode(self):
        data = np.random.rand(10, 10)
        result = heatmap_bigram(data, xres=32, yres=16)
        assert isinstance(result, str)

    def test_heatmap_colorbar(self):
        data = np.array([[0, 1], [1, 0]])
        result = heatmap(data, vmin=0, vmax=1, show_colorbar=True)
        assert "0" in result
        assert "1" in result

    def test_heatmap_no_colorbar(self):
        data = np.array([[0, 1], [1, 0]])
        result = heatmap(data, vmin=0, vmax=1, show_colorbar=False, xres=4, yres=2)
        lines = [l for l in result.split("\n") if l.strip()]
        assert len(lines) == 2

    def test_resample_nearest(self):
        data = np.arange(16).reshape(4, 4).astype(float)
        resampled = _resample_nearest(data, 2, 2)
        assert resampled.shape == (2, 2)

    def test_resample_mean(self):
        data = np.ones((4, 4))
        resampled = _resample_mean(data, 2, 2)
        assert resampled.shape == (2, 2)
        np.testing.assert_allclose(resampled, 1.0)


class TestTextPlotResult:
    def test_str_representation(self):
        result = TextPlotResult("test text", title="Test")
        assert str(result) == "test text"

    def test_metadata(self):
        result = TextPlotResult("text", title="t", metadata={"vmin": 0, "vmax": 1})
        assert result.metadata["vmin"] == 0
        assert result.metadata["vmax"] == 1


class TestSmooth1dTxt:
    def test_basic_1d_plot(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (100, 1))
        Y = np.sin(X * 2 * np.pi)
        result = smooth_1d_txt(
            X,
            Y,
            input_names=["x"],
            output_name="sin(x)",
            xlims=(0, 1),
            res=40,
            height=10,
        )
        assert isinstance(result, TextPlotResult)
        assert len(result.text) > 0

    def test_1d_larger_dataset(self):
        # tests 1D plot with more data points
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (500, 1))
        Y = np.sin(X * 4 * np.pi) + rng.normal(0, 0.1, X.shape)
        result = smooth_1d_txt(
            X,
            Y,
            input_names=["x"],
            output_name="sin(4πx)",
            xlims=(0, 1),
            res=60,
            height=15,
        )
        assert isinstance(result, TextPlotResult)
        assert len(result.text) > 100


class TestSmooth2dTxt:
    def test_basic_2d_plot(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (200, 2))
        Y = X[:, 0:1] * X[:, 1:2]
        result = smooth_2d_txt(
            X,
            Y,
            input_names=["x", "y"],
            output_name="x*y",
            xlims=(0, 1),
            ylims=(0, 1),
            xres=32,
            yres=16,
        )
        assert isinstance(result, TextPlotResult)
        assert "vmin" in result.metadata
        assert "vmax" in result.metadata

    def test_2d_with_title(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (100, 2))
        Y = rng.uniform(0, 1, (100, 1))
        result = smooth_2d_txt(
            X,
            Y,
            input_names=["a", "b"],
            output_name="c",
            title="Test 2D Plot",
        )
        assert "Test 2D Plot" in result.text


class TestGetTxtPlotFunction:
    def test_exact_match(self):
        func = get_txt_plot_function("biocomp.plotting.plotting_smooth.smooth_1d")
        assert func is smooth_1d_txt

    def test_suffix_match(self):
        func = get_txt_plot_function("some.module.smooth_2d")
        assert func is smooth_2d_txt

    def test_no_match(self):
        func = get_txt_plot_function("nonexistent.function")
        assert func is None


@pytest.mark.parametrize("levels", [5, 8, 16])
def test_heatmap_levels(levels):
    data = np.linspace(0, 1, 100).reshape(10, 10)
    result = heatmap(data, levels=levels, xres=10, yres=10)
    assert isinstance(result, str)


@pytest.mark.parametrize("mode", ["single", "bigram"])
def test_heatmap_modes(mode):
    data = np.random.rand(10, 10)
    result = heatmap(data, mode=mode, xres=20, yres=10)
    assert isinstance(result, str)


def test_heatmap_constant_data():
    data = np.ones((10, 10)) * 0.5
    result = heatmap(data, vmin=0, vmax=1)
    assert isinstance(result, str)


def test_heatmap_edge_values():
    data = np.array([[0.0, 1.0], [1.0, 0.0]])
    result = heatmap(data, vmin=0, vmax=1)
    assert isinstance(result, str)
