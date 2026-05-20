# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Tests for matplotlib-generated SVG contour plots.

These tests verify that SVGs exported from matplotlib (e.g., contourf plots)
are correctly parsed and sampled, particularly handling the pt->px unit conversion.
"""

import os
import numpy as np
import pytest
from pathlib import Path

from biocomp.design_targets import SVGTarget
from biocomp.designutils import _extract_shapes_from_svg, sample_from_svg


TESTS_DIR = Path(__file__).parent
BIOCOMP_ROOT = Path(os.environ.get("BIOCOMP_ROOT", ""))
ERIC_DESIGNS_DIR = BIOCOMP_ROOT / "Designs" / "eric" / "v1"


@pytest.fixture
def cairo_available():
    try:
        import cairosvg  # noqa: F401

        return True
    except ImportError:
        pytest.skip("cairosvg not available")


class TestMatplotlibContourSVGs:
    """Test handling of matplotlib-generated contour SVGs."""

    @pytest.mark.skipif(not ERIC_DESIGNS_DIR.exists(), reason="Eric designs not found")
    def test_bandpass_high_symmetry(self):
        """Bandpass+high pattern should be symmetric around center x-axis."""
        svg_path = ERIC_DESIGNS_DIR / "bandpass+high.svg"
        target = SVGTarget(path=svg_path, name="bandpass_high")
        _, Y_grid = target.get_lattice(resolution=(48, 48), seed=0)

        # Check horizontal symmetry (left vs right halves should be mirror)
        left_half = Y_grid[:, :24]
        right_half = np.flip(Y_grid[:, 24:], axis=1)
        symmetry_diff = np.abs(left_half - right_half).mean()
        assert symmetry_diff < 0.1, f"Pattern not symmetric: mean diff = {symmetry_diff}"

    @pytest.mark.skipif(not ERIC_DESIGNS_DIR.exists(), reason="Eric designs not found")
    def test_bandpass_high_matches_cairo(self, cairo_available):
        """Our sampling should match cairosvg reference render."""
        import cairosvg
        import io
        from PIL import Image

        svg_path = ERIC_DESIGNS_DIR / "bandpass+high.svg"
        target = SVGTarget(path=svg_path, name="bandpass_high", latent_out=(0.0, 1.0))
        _, Y_grid = target.get_lattice(resolution=(48, 48), seed=0)
        Y_flipped = np.flipud(Y_grid)

        # Cairo reference
        png_data = cairosvg.svg2png(url=str(svg_path), output_width=48, output_height=48)
        img = Image.open(io.BytesIO(png_data)).convert("L")
        cairo_arr = 1.0 - (np.array(img) / 255.0)

        diff = np.abs(cairo_arr - Y_flipped)
        assert diff.mean() < 0.02, f"Mean diff {diff.mean():.3f} too high (expected < 0.02)"
        assert diff.max() < 0.25, f"Max diff {diff.max():.3f} too high (expected < 0.25)"

    @pytest.mark.skipif(not ERIC_DESIGNS_DIR.exists(), reason="Eric designs not found")
    def test_pt_to_px_scaling_handled(self):
        """Verify pt->px scaling is correctly handled."""
        svg_path = ERIC_DESIGNS_DIR / "bandpass+high.svg"

        paths, greys, (vx, vy, vw, vh) = _extract_shapes_from_svg(svg_path, max_is_black=True)

        # svgelements should return scaled dimensions (277.2pt * 4/3 = 369.6px)
        # Check that vw/vh match the path bounding boxes
        path_max_x = max(p.get_extents().x1 for p in paths)
        path_max_y = max(p.get_extents().y1 for p in paths)

        # vw/vh should encompass the paths (with some tolerance for paths extending beyond)
        assert vw >= path_max_x * 0.9, f"vw={vw} too small for paths extending to {path_max_x}"
        assert vh >= path_max_y * 0.9, f"vh={vh} too small for paths extending to {path_max_y}"

    @pytest.mark.skipif(not ERIC_DESIGNS_DIR.exists(), reason="Eric designs not found")
    def test_center_has_high_values(self):
        """The center-top of bandpass+high should have the highest values."""
        svg_path = ERIC_DESIGNS_DIR / "bandpass+high.svg"
        target = SVGTarget(path=svg_path, name="bandpass_high")
        _, Y_grid = target.get_lattice(resolution=(48, 48), seed=0)

        # Center of horizontal axis, top portion (bandpass peak + high)
        center_top = Y_grid[40:48, 20:28].mean()

        # Edges should have lower values
        left_edge = Y_grid[:, :5].mean()
        right_edge = Y_grid[:, -5:].mean()
        bottom = Y_grid[:10, :].mean()

        assert center_top > left_edge, "Center-top should be brighter than left edge"
        assert center_top > right_edge, "Center-top should be brighter than right edge"
        assert center_top > bottom, "Center-top should be brighter than bottom"


class TestAllEricDesigns:
    """Validate all eric design SVGs."""

    @pytest.mark.skipif(not ERIC_DESIGNS_DIR.exists(), reason="Eric designs not found")
    @pytest.mark.parametrize(
        "svg_name", ["bandpass+high.svg", "bandpass+low.svg", "high+low.svg"]
    )
    def test_eric_svg_loads_without_error(self, svg_name):
        svg_path = ERIC_DESIGNS_DIR / svg_name
        if not svg_path.exists():
            pytest.skip(f"{svg_name} not found")

        target = SVGTarget(path=svg_path, name=svg_name.replace(".svg", ""))
        _, Y_grid = target.get_lattice(resolution=(32, 32), seed=0)

        # Basic sanity checks
        assert Y_grid.shape == (32, 32)
        assert not np.isnan(Y_grid).any(), "Grid contains NaN values"
        assert Y_grid.min() >= 0.0, f"Grid min {Y_grid.min()} < 0"
        assert Y_grid.max() <= 1.0, f"Grid max {Y_grid.max()} > 1"

    @pytest.mark.skipif(not ERIC_DESIGNS_DIR.exists(), reason="Eric designs not found")
    @pytest.mark.parametrize(
        "svg_name", ["bandpass+high.svg", "bandpass+low.svg", "high+low.svg"]
    )
    def test_eric_svg_matches_cairo(self, svg_name, cairo_available):
        """All eric SVGs should reasonably match cairosvg reference."""
        import cairosvg
        import io
        from PIL import Image

        svg_path = ERIC_DESIGNS_DIR / svg_name
        if not svg_path.exists():
            pytest.skip(f"{svg_name} not found")

        target = SVGTarget(
            path=svg_path, name=svg_name.replace(".svg", ""), latent_out=(0.0, 1.0)
        )
        _, Y_grid = target.get_lattice(resolution=(48, 48), seed=0)
        Y_flipped = np.flipud(Y_grid)

        # Cairo reference
        png_data = cairosvg.svg2png(url=str(svg_path), output_width=48, output_height=48)
        img = Image.open(io.BytesIO(png_data)).convert("L")
        cairo_arr = 1.0 - (np.array(img) / 255.0)

        diff = np.abs(cairo_arr - Y_flipped)
        assert diff.mean() < 0.05, (
            f"{svg_name}: Mean diff {diff.mean():.3f} too high (expected < 0.05)"
        )


class TestSampleFromSvgWithMatplotlib:
    """Test sample_from_svg function with matplotlib-generated SVGs."""

    @pytest.mark.skipif(not ERIC_DESIGNS_DIR.exists(), reason="Eric designs not found")
    def test_sample_from_svg_grid_shape(self):
        """Grid sampling should return correct shapes."""
        svg_path = ERIC_DESIGNS_DIR / "bandpass+high.svg"
        X, Y = sample_from_svg(svg_path, grid=(32, 32))

        assert X.shape == (32 * 32, 2), f"X shape should be (1024, 2), got {X.shape}"
        assert Y.shape == (1, 32, 32), f"Y shape should be (1, 32, 32), got {Y.shape}"

    @pytest.mark.skipif(not ERIC_DESIGNS_DIR.exists(), reason="Eric designs not found")
    def test_sample_from_svg_latent_ranges(self):
        """Latent coordinate ranges should be respected."""
        svg_path = ERIC_DESIGNS_DIR / "bandpass+high.svg"
        X, Y = sample_from_svg(
            svg_path,
            grid=(32, 32),
            latent_x=(0.1, 0.5),
            latent_y=(0.2, 0.6),
            latent_out=(0.1, 0.9),
        )

        assert X[:, 0].min() >= 0.09, f"X min should be >= 0.1, got {X[:, 0].min()}"
        assert X[:, 0].max() <= 0.51, f"X max should be <= 0.5, got {X[:, 0].max()}"
        assert X[:, 1].min() >= 0.19, f"Y min should be >= 0.2, got {X[:, 1].min()}"
        assert X[:, 1].max() <= 0.61, f"Y max should be <= 0.6, got {X[:, 1].max()}"
        assert Y.min() >= 0.09, f"Output min should be >= 0.1, got {Y.min()}"
        assert Y.max() <= 0.91, f"Output max should be <= 0.9, got {Y.max()}"
