"""Comprehensive tests for SVG parsing and sampling."""

import io
import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from biocomp.design_targets import SVGTarget
from biocomp.designutils import (
    _parse_transform,
    _parse_svg_path,
    _extract_shapes_from_svg,
    sample_from_svg,
)


TESTS_DIR = Path(__file__).parent
DESIGNS_DIR = TESTS_DIR / "resources" / "designs"

MIT_DESIGNS_DIR = Path("${BIOCOMP_ROOT}/Designs").expanduser()
if not MIT_DESIGNS_DIR.exists():
    import os

    MIT_DESIGNS_DIR = Path(os.environ.get("BIOCOMP_ROOT", "")) / "Designs"


def render_svg_to_array(svg_path: Path, size: tuple[int, int]) -> np.ndarray:
    """Render SVG to numpy array using cairosvg (reference implementation)."""
    import cairosvg

    png_data = cairosvg.svg2png(
        url=str(svg_path),
        output_width=size[0],
        output_height=size[1],
    )
    img = Image.open(io.BytesIO(png_data)).convert("L")
    return np.array(img) / 255.0


class TestTransformParsing:
    """Test _parse_transform function with various SVG transform strings."""

    def test_empty_transform(self):
        result = _parse_transform("")
        assert np.allclose(result, np.eye(3))

    def test_none_transform(self):
        result = _parse_transform(None)
        assert np.allclose(result, np.eye(3))

    def test_unrecognized_transform(self):
        result = _parse_transform("unknown(1, 2, 3)")
        assert np.allclose(result, np.eye(3))

    def test_rotate_basic(self):
        result = _parse_transform("rotate(90)")
        cos_90, sin_90 = 0.0, 1.0
        expected = np.array([[cos_90, -sin_90, 0], [sin_90, cos_90, 0], [0, 0, 1]])
        assert np.allclose(result, expected, atol=1e-6)

    def test_rotate_with_center(self):
        result = _parse_transform("rotate(90 50 50)")
        assert result.shape == (3, 3)
        pt = np.array([50, 0, 1])
        transformed = result @ pt
        assert np.allclose(transformed[:2], [100, 50], atol=1e-6)

    def test_matrix_transform(self):
        result = _parse_transform("matrix(0.707 0.707 -0.707 0.707 50 20)")
        assert result.shape == (3, 3)
        assert np.isclose(result[0, 0], 0.707, atol=1e-3)
        assert np.isclose(result[0, 1], -0.707, atol=1e-3)
        assert np.isclose(result[0, 2], 50, atol=1e-3)
        assert np.isclose(result[1, 0], 0.707, atol=1e-3)
        assert np.isclose(result[1, 1], 0.707, atol=1e-3)
        assert np.isclose(result[1, 2], 20, atol=1e-3)

    def test_translate_xy(self):
        result = _parse_transform("translate(30, 40)")
        expected = np.array([[1, 0, 30], [0, 1, 40], [0, 0, 1]])
        assert np.allclose(result, expected)

    def test_translate_x_only(self):
        result = _parse_transform("translate(30)")
        expected = np.array([[1, 0, 30], [0, 1, 0], [0, 0, 1]])
        assert np.allclose(result, expected)

    def test_scale_uniform(self):
        result = _parse_transform("scale(2)")
        expected = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
        assert np.allclose(result, expected)

    def test_scale_nonuniform(self):
        result = _parse_transform("scale(2, 0.5)")
        expected = np.array([[2, 0, 0], [0, 0.5, 0], [0, 0, 1]])
        assert np.allclose(result, expected)

    def test_skewx(self):
        result = _parse_transform("skewX(45)")
        assert np.isclose(result[0, 1], np.tan(np.pi / 4), atol=1e-6)
        assert np.isclose(result[1, 0], 0)

    def test_skewy(self):
        result = _parse_transform("skewY(45)")
        assert np.isclose(result[1, 0], np.tan(np.pi / 4), atol=1e-6)
        assert np.isclose(result[0, 1], 0)

    def test_matrix_with_spaces(self):
        result = _parse_transform("matrix( 1 0 0 1 10 20 )")
        expected = np.array([[1, 0, 10], [0, 1, 20], [0, 0, 1]])
        assert np.allclose(result, expected)

    def test_matrix_with_commas(self):
        result = _parse_transform("matrix(1, 0, 0, 1, 10, 20)")
        expected = np.array([[1, 0, 10], [0, 1, 20], [0, 0, 1]])
        assert np.allclose(result, expected)


class TestPathParsing:
    """Test _parse_svg_path function."""

    def test_basic_polygon(self):
        pts = _parse_svg_path("M 0 0 L 10 0 L 10 10 L 0 10")
        assert len(pts) >= 4
        assert pts[0] == (0, 0)

    def test_horizontal_line_H(self):
        pts = _parse_svg_path("M 0 0 H 20")
        assert len(pts) >= 2
        assert pts[-1][0] == 20

    def test_vertical_line_V(self):
        pts = _parse_svg_path("M 0 0 V 30")
        assert len(pts) >= 2
        assert pts[-1][1] == 30

    def test_close_path_Z(self):
        pts = _parse_svg_path("M 0 0 L 10 0 L 10 10 Z")
        assert len(pts) >= 3

    def test_relative_moveto_m(self):
        pts = _parse_svg_path("M 10 10 m 5 5 L 30 30")
        assert len(pts) >= 2
        assert pts[1] == (15, 15)

    def test_relative_lineto_l(self):
        pts = _parse_svg_path("M 10 10 l 10 10")
        assert len(pts) >= 2
        assert pts[1] == (20, 20)

    def test_relative_horizontal_h(self):
        pts = _parse_svg_path("M 10 10 h 15")
        assert len(pts) >= 2
        assert pts[-1] == (25, 10)

    def test_relative_vertical_v(self):
        pts = _parse_svg_path("M 10 10 v 15")
        assert len(pts) >= 2
        assert pts[-1] == (10, 25)

    def test_mixed_commands(self):
        pts = _parse_svg_path("M 0 0 L 10 0 h 5 V 20 l -5 5 Z")
        assert len(pts) >= 4


class TestPathParsingVsReference:
    """Compare our parser with svgpath2mpl reference."""

    @pytest.mark.parametrize(
        "path_d",
        [
            "M 0 0 L 10 10 L 20 0 Z",
            "M 5 5 H 15 V 20 H 5 Z",
            "M 10 10 L 30 10 V 30 H 10 Z",
            "M 0 0 l 10 10 l 10 -10 z",
            "M 50 50 h 20 v 20 h -20 z",
        ],
    )
    def test_path_matches_reference(self, path_d):
        from svgpath2mpl import parse_path as ref_parse

        our_pts = np.array(_parse_svg_path(path_d))
        ref_path = ref_parse(path_d)
        ref_vertices = (
            ref_path.vertices[:-1] if len(ref_path.vertices) > len(our_pts) else ref_path.vertices
        )

        n_match = min(len(our_pts), len(ref_vertices))
        assert n_match >= 3, f"Not enough points parsed: {n_match}"
        for i in range(n_match):
            assert np.allclose(our_pts[i], ref_vertices[i], atol=0.1), (
                f"Mismatch at point {i}: ours={our_pts[i]}, ref={ref_vertices[i]}"
            )


class TestShapeSampling:
    """Test complete SVG → lattice sampling pipeline."""

    def test_rectangle_coverage(self):
        target = SVGTarget(path=DESIGNS_DIR / "test_top_bar.svg", name="top_bar")
        _, Y_grid = target.get_lattice(resolution=(32, 32), seed=0)
        high_fraction = (Y_grid > 0.2).mean()
        assert 0.15 < high_fraction < 0.35, (
            f"Expected ~25% high values, got {high_fraction * 100:.1f}%"
        )

    def test_circle_coverage(self):
        target = SVGTarget(path=DESIGNS_DIR / "test_circles.svg", name="circles")
        _, Y_grid = target.get_lattice(resolution=(64, 64), seed=0)
        high_fraction = (Y_grid > 0.1).mean()
        assert high_fraction > 0.05, (
            f"Expected some circle coverage, got {high_fraction * 100:.1f}%"
        )

    def test_rotated_rectangle_via_matrix(self):
        target = SVGTarget(path=DESIGNS_DIR / "test_matrix_transform.svg", name="matrix")
        _, Y_grid = target.get_lattice(resolution=(64, 64), seed=0)
        high_mask = Y_grid > 0.2
        high_fraction = high_mask.mean()
        assert high_fraction > 0.01, "Rotated rectangle should cover some area"
        rows_with_high = np.any(high_mask, axis=1).sum()
        cols_with_high = np.any(high_mask, axis=0).sum()
        assert rows_with_high > 5, "Rotated rect should span multiple rows (not axis-aligned)"
        assert cols_with_high > 5, "Rotated rect should span multiple cols (not axis-aligned)"

    def test_overlapping_shapes_last_wins(self):
        target = SVGTarget(path=DESIGNS_DIR / "test_overlapping.svg", name="overlapping")
        _, Y_grid = target.get_lattice(resolution=(64, 64), seed=0)
        center_area = Y_grid[8:30, 32:58].mean()
        assert center_area > 0.35, f"Black rect region should be darkest, got {center_area:.3f}"


class TestCairosvgComparison:
    """Compare sampling output with cairosvg rendering."""

    @pytest.fixture
    def cairo_available(self):
        try:
            import cairosvg  # noqa: F401

            return True
        except ImportError:
            pytest.skip("cairosvg not available")

    def test_simple_rect_matches_cairo(self, cairo_available):
        svg_path = DESIGNS_DIR / "test_top_bar.svg"
        resolution = (64, 64)
        target = SVGTarget(path=svg_path, name="top_bar", latent_out=(0.0, 1.0))
        _, Y_grid = target.get_lattice(resolution=resolution, seed=0)
        cairo_render = render_svg_to_array(svg_path, resolution)
        cairo_render = 1.0 - cairo_render
        Y_binary = (Y_grid > 0.5).astype(float)
        cairo_binary = (cairo_render > 0.5).astype(float)
        Y_binary_flipped = np.flipud(Y_binary)
        agreement = (Y_binary_flipped == cairo_binary).mean()
        assert agreement > 0.9, f"Expected >90% agreement with cairo, got {agreement * 100:.1f}%"

    def test_rotated_rect_matches_cairo(self, cairo_available):
        svg_path = DESIGNS_DIR / "test_matrix_transform.svg"
        resolution = (64, 64)
        target = SVGTarget(path=svg_path, name="matrix", latent_out=(0.0, 1.0))
        _, Y_grid = target.get_lattice(resolution=resolution, seed=0)
        cairo_render = render_svg_to_array(svg_path, resolution)
        cairo_render = 1.0 - cairo_render
        our_coverage = (Y_grid > 0.5).mean()
        cairo_coverage = (cairo_render > 0.5).mean()
        coverage_ratio = our_coverage / max(cairo_coverage, 1e-6)
        assert 0.5 < coverage_ratio < 2.0, (
            f"Coverage mismatch: ours={our_coverage:.3f}, cairo={cairo_coverage:.3f}"
        )

    def test_circles_match_cairo(self, cairo_available):
        svg_path = DESIGNS_DIR / "test_circles.svg"
        resolution = (64, 64)
        target = SVGTarget(path=svg_path, name="circles", latent_out=(0.0, 1.0))
        _, Y_grid = target.get_lattice(resolution=resolution, seed=0)
        cairo_render = render_svg_to_array(svg_path, resolution)
        cairo_render = 1.0 - cairo_render
        our_coverage = (Y_grid > 0.3).mean()
        cairo_coverage = (cairo_render > 0.3).mean()
        coverage_ratio = our_coverage / max(cairo_coverage, 1e-6)
        assert 0.5 < coverage_ratio < 2.0, (
            f"Circle coverage mismatch: ours={our_coverage:.3f}, cairo={cairo_coverage:.3f}"
        )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_shape(self):
        target = SVGTarget(path=DESIGNS_DIR / "test_small_shapes.svg", name="small")
        _, Y_grid = target.get_lattice(resolution=(100, 100), seed=0)
        high_pixels = (Y_grid > 0.2).sum()
        assert high_pixels > 0, "At least some small shapes should be detected"

    def test_shape_at_viewbox_edge(self):
        target = SVGTarget(path=DESIGNS_DIR / "test_viewbox_edges.svg", name="edges")
        _, Y_grid = target.get_lattice(resolution=(64, 64), seed=0)
        left_col = Y_grid[:, 0:3].mean()
        right_col = Y_grid[:, -3:].mean()
        assert left_col > 0.2, f"Left edge should have content, got {left_col:.3f}"
        assert right_col > 0.1, f"Right edge should have content, got {right_col:.3f}"

    def test_translate_transform(self):
        target = SVGTarget(path=DESIGNS_DIR / "test_translate.svg", name="translate")
        _, Y_grid = target.get_lattice(resolution=(64, 64), seed=0)
        high_mask = Y_grid > 0.2
        rows_with_high = np.where(np.any(high_mask, axis=1))[0]
        cols_with_high = np.where(np.any(high_mask, axis=0))[0]
        assert len(rows_with_high) > 0, "Should have some high-value rows"
        assert len(cols_with_high) > 0, "Should have some high-value cols"
        center_col = cols_with_high.mean() / Y_grid.shape[1]
        assert 0.2 < center_col < 0.7, (
            f"Translated rect should be offset right, col center={center_col:.2f}"
        )

    def test_scale_transform(self):
        target = SVGTarget(path=DESIGNS_DIR / "test_scale.svg", name="scale")
        _, Y_grid = target.get_lattice(resolution=(64, 64), seed=0)
        high_fraction = (Y_grid > 0.2).mean()
        assert high_fraction > 0.01, "Scaled rectangle should cover some area"

    def test_complex_path(self):
        target = SVGTarget(path=DESIGNS_DIR / "test_complex_path.svg", name="complex")
        _, Y_grid = target.get_lattice(resolution=(64, 64), seed=0)
        high_fraction = (Y_grid > 0.2).mean()
        assert high_fraction > 0.1, "Complex path (step shape) should cover significant area"

    def test_relative_path(self):
        target = SVGTarget(path=DESIGNS_DIR / "test_relative_path.svg", name="relative")
        _, Y_grid = target.get_lattice(resolution=(64, 64), seed=0)
        high_fraction = (Y_grid > 0.2).mean()
        assert high_fraction > 0.1, "Relative path should produce same coverage as absolute"


@pytest.mark.skipif(not MIT_DESIGNS_DIR.exists(), reason="MIT Designs directory not found")
class TestMITDesigns:
    """Validate real MIT design targets."""

    def test_mit_m2_sharp_diagonal_stripes(self):
        svg_path = MIT_DESIGNS_DIR / "MIT_M2_sharp.svg"
        if not svg_path.exists():
            pytest.skip("MIT_M2_sharp.svg not found")
        target = SVGTarget(path=svg_path, name="MIT_M2_sharp")
        _, Y_grid = target.get_lattice(resolution=(64, 64), seed=0)
        high_mask = Y_grid > 0.2
        rows_with_4_transitions = 0
        for row in range(Y_grid.shape[0]):
            row_data = high_mask[row, :]
            transitions = np.abs(np.diff(row_data.astype(int))).sum()
            if transitions >= 4:
                rows_with_4_transitions += 1
        assert rows_with_4_transitions >= 8, (
            f"M shape should have multiple rows with 4+ transitions (diagonal stripes), "
            f"got {rows_with_4_transitions} rows"
        )

    def test_mit_t_sharp_t_structure(self):
        svg_path = MIT_DESIGNS_DIR / "MIT_T_sharp.svg"
        if not svg_path.exists():
            pytest.skip("MIT_T_sharp.svg not found")
        target = SVGTarget(path=svg_path, name="MIT_T_sharp")
        _, Y_grid = target.get_lattice(resolution=(64, 64), seed=0)
        top_row = Y_grid[-1, :]
        assert top_row.mean() > 0.2, "T's horizontal bar should be at top"
        center_col = Y_grid[:, 28:36].mean()
        assert center_col > 0.15, "T's vertical stem should be in center"

    def test_mit_i_sharp_i_structure(self):
        svg_path = MIT_DESIGNS_DIR / "MIT_I_sharp.svg"
        if not svg_path.exists():
            pytest.skip("MIT_I_sharp.svg not found")
        target = SVGTarget(path=svg_path, name="MIT_I_sharp")
        _, Y_grid = target.get_lattice(resolution=(64, 64), seed=0)
        top_row = Y_grid[-1, :].mean()
        bottom_row = Y_grid[0, :].mean()
        center_col = Y_grid[20:44, 28:36].mean()
        assert top_row > 0.15, "I should have top bar"
        assert bottom_row > 0.15, "I should have bottom bar"
        assert center_col > 0.15, "I should have vertical stem"


class TestTransformIntegration:
    """Integration tests for transforms applied to actual SVG elements."""

    def test_matrix_transform_produces_rotated_shape(self):
        svg_path = DESIGNS_DIR / "test_matrix_transform.svg"
        paths, greys, viewbox = _extract_shapes_from_svg(svg_path, max_is_black=True)
        assert len(paths) >= 1, "Should extract at least one path"
        vertices = paths[0].vertices
        xs, ys = vertices[:, 0], vertices[:, 1]
        x_range = xs.max() - xs.min()
        y_range = ys.max() - ys.min()
        aspect = (
            max(x_range, y_range) / min(x_range, y_range)
            if min(x_range, y_range) > 0
            else float("inf")
        )
        assert aspect < 5, (
            f"Rotated rect should have moderate aspect ratio, got {aspect:.2f} "
            f"(x_range={x_range:.2f}, y_range={y_range:.2f})"
        )

    def test_translate_transform_offsets_shape(self):
        svg_path = DESIGNS_DIR / "test_translate.svg"
        paths, greys, viewbox = _extract_shapes_from_svg(svg_path, max_is_black=True)
        assert len(paths) >= 1, "Should extract at least one path"
        vertices = paths[0].vertices
        xs = vertices[:, 0]
        assert xs.min() >= 25, (
            f"Translated rect should start at x>=30 (with tolerance), got min x={xs.min()}"
        )


class TestSampleFromSvgDirectly:
    """Test sample_from_svg function directly."""

    def test_grid_sampling_returns_correct_shape(self):
        svg_path = DESIGNS_DIR / "test_top_bar.svg"
        X, Y = sample_from_svg(svg_path, grid=(32, 32))
        assert X.shape == (32 * 32, 2), f"X shape should be (1024, 2), got {X.shape}"
        assert Y.shape == (1, 32, 32), f"Y shape should be (1, 32, 32), got {Y.shape}"

    def test_uniform_sampling_returns_correct_shape(self):
        svg_path = DESIGNS_DIR / "test_top_bar.svg"
        X, Y = sample_from_svg(svg_path, n=100)
        assert X.shape == (100, 2), f"X shape should be (100, 2), got {X.shape}"
        assert Y.shape == (100, 1), f"Y shape should be (100, 1), got {Y.shape}"

    def test_latent_range_respected(self):
        svg_path = DESIGNS_DIR / "test_top_bar.svg"
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
