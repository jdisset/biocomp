"""Test SVG sampling correctly captures edges of the image."""

import numpy as np
import pytest
from pathlib import Path

from biocomp.design_targets import SVGTarget


TESTS_DIR = Path(__file__).parent
TEST_TOP_BAR_SVG = TESTS_DIR / "resources" / "designs" / "test_top_bar.svg"
MIT_T_SHARP_SVG = Path("${BIOCOMP_ROOT}/Designs/MIT_T_sharp.svg").expanduser()


@pytest.fixture
def top_bar_target():
    """SVG with a black bar at the top (y=0 to y=25 in SVG coords)."""
    return SVGTarget(
        path=TEST_TOP_BAR_SVG,
        name="test_top_bar",
        blur_sigma=0.0,
    )


class TestSVGSamplingEdges:
    """Test that SVG sampling correctly captures content at image edges."""

    def test_top_bar_lattice_last_row_is_high(self, top_bar_target):
        """The SVG has a bar at top (SVG y=0). After get_lattice, last row should be high.

        Grid generation uses descending y_vals (from vh to 0), so:
        - Row 0 corresponds to SVG bottom (y=vh) → background → low value
        - Row -1 corresponds to SVG top (y=0) → bar → high value
        """
        _, Y_grid = top_bar_target.get_lattice(resolution=(32, 32), seed=0)

        top_row_in_grid = Y_grid[-1, :]  # last row = SVG y=0 = top of image
        bottom_row_in_grid = Y_grid[0, :]  # first row = SVG y=vh = bottom of image

        assert top_row_in_grid.mean() > 0.3, (
            f"Top of SVG (bar) should have high values, got mean={top_row_in_grid.mean():.4f}"
        )
        assert bottom_row_in_grid.mean() < 0.1, (
            f"Bottom of SVG (background) should have low values, got mean={bottom_row_in_grid.mean():.4f}"
        )

    def test_top_bar_covers_correct_fraction(self, top_bar_target):
        """The bar covers top 25% of image. ~25% of rows should be high."""
        _, Y_grid = top_bar_target.get_lattice(resolution=(32, 32), seed=0)

        row_means = Y_grid.mean(axis=1)
        high_rows = (row_means > 0.2).sum()
        fraction_high = high_rows / len(row_means)

        assert 0.2 < fraction_high < 0.35, (
            f"Expected ~25% of rows to be high (bar), got {fraction_high*100:.1f}%"
        )

    def test_top_row_not_blank_after_display_flips(self, top_bar_target):
        """Simulate the display pipeline's 3 flipud operations.

        Display pipeline:
        1. Line 712-714: flipud(cached_target_grid)
        2. Line 612: flipud(Y_target_grid)
        3. Line 1064: flipud(target_grid)

        Net effect: 3 flips = 1 flip. Row 0 should show SVG top (the bar).
        """
        _, Y_grid = top_bar_target.get_lattice(resolution=(32, 32), seed=0)

        # Simulate the 3 flipud operations in the display pipeline
        after_flip_1 = np.flipud(Y_grid)
        after_flip_2 = np.flipud(after_flip_1)
        after_flip_3 = np.flipud(after_flip_2)

        # Row 0 is displayed at top of terminal - should be the bar (high value)
        display_top_row = after_flip_3[0, :]

        assert display_top_row.mean() > 0.3, (
            f"Display top row should show the bar (high value), got mean={display_top_row.mean():.4f}"
        )


@pytest.mark.skipif(
    not MIT_T_SHARP_SVG.exists() and not Path("/home/jean/MIT Dropbox/Jean Disset/Biocomp_v2/Designs/MIT_T_sharp.svg").exists(),
    reason="MIT_T_sharp.svg not found"
)
class TestMITTSharpSampling:
    """Test MIT_T_sharp.svg sampling - the T's horizontal bar should reach the top."""

    @pytest.fixture
    def mit_t_sharp_target(self):
        svg_path = MIT_T_SHARP_SVG
        if not svg_path.exists():
            svg_path = Path("/home/jean/MIT Dropbox/Jean Disset/Biocomp_v2/Designs/MIT_T_sharp.svg")
        return SVGTarget(
            path=svg_path,
            name="MIT_T_sharp",
            blur_sigma=0.0,
        )

    def test_horizontal_bar_at_top(self, mit_t_sharp_target):
        """The T's horizontal bar spans full width at top. Last row of grid should have high center values."""
        _, Y_grid = mit_t_sharp_target.get_lattice(resolution=(64, 64), seed=0)

        # Last row = SVG top = horizontal bar
        top_row = Y_grid[-1, :]

        # The horizontal bar spans the full width
        assert top_row.mean() > 0.25, (
            f"Top row (horizontal bar) should have high values, got mean={top_row.mean():.4f}"
        )

    def test_display_top_row_shows_bar(self, mit_t_sharp_target):
        """After 3 flipud operations, display top row should show the horizontal bar."""
        _, Y_grid = mit_t_sharp_target.get_lattice(resolution=(64, 64), seed=0)

        # Simulate display pipeline flips
        display_grid = np.flipud(np.flipud(np.flipud(Y_grid)))
        display_top_row = display_grid[0, :]

        assert display_top_row.mean() > 0.25, (
            f"Display top row should show horizontal bar, got mean={display_top_row.mean():.4f}"
        )
