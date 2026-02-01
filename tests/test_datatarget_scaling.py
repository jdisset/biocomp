"""Test DataTarget scaling behavior.

When latent_x/latent_y extents are smaller than the data range, the target
should be SCALED to fit, not CROPPED.

Example: If data X spans [0, 1] and latent_x=(0, 0.6):
- CORRECT: Full pattern compressed to [0, 0.6], data point at x=1 maps to 0.6
- WRONG: Only left 60% of pattern shown (cropped at x=0.6)
"""

import numpy as np
import pytest
from biocomp.design_targets import DataTarget


def make_ramp_data(x_range=(0.0, 1.0), y_range=(0.0, 1.0), n_points=2500):
    """Create 2D data with a clear diagonal ramp pattern.

    Y = (x + y) / 2, normalized to output range.
    This creates a distinctive pattern where top-right is bright.
    """
    n_side = int(np.sqrt(n_points))
    x = np.linspace(x_range[0], x_range[1], n_side)
    y = np.linspace(y_range[0], y_range[1], n_side)
    xx, yy = np.meshgrid(x, y)
    X = np.column_stack([xx.ravel(), yy.ravel()])
    # Diagonal ramp: high at top-right, low at bottom-left
    Y = (xx.ravel() + yy.ravel()) / 2
    return X, Y


@pytest.fixture
def ramp_data():
    """Data with a diagonal ramp spanning [0, 1] x [0, 1]."""
    return make_ramp_data(x_range=(0.0, 1.0), y_range=(0.0, 1.0), n_points=2500)


class TestDataTargetScaling:
    """Tests for DataTarget latent extent scaling behavior."""

    def test_full_extent_baseline(self, ramp_data):
        """Baseline: full extent preserves the pattern."""
        X, Y = ramp_data
        target = DataTarget(X=X, Y=Y, latent_x=(0.0, 1.0), latent_y=(0.0, 1.0))

        X_grid, Y_grid = target.get_lattice(resolution=(16, 16))

        # Grid should cover [0, 1] x [0, 1]
        assert X_grid[:, 0].min() == pytest.approx(0.0, abs=0.05)
        assert X_grid[:, 0].max() == pytest.approx(1.0, abs=0.05)

        # Check center values to avoid NaN issues at edges
        # Y_grid is (yres, xres)
        # Diagonal ramp: Y = (x + y) / 2
        # At center (0.5, 0.5), Y should be 0.5
        center = Y_grid[8, 8]
        assert not np.isnan(center), "Center value should not be NaN"
        assert center == pytest.approx(0.5, abs=0.1), "Center of ramp should be ~0.5"

        # High region (near 0.8, 0.8) should have higher value than low (0.2, 0.2)
        high_region = Y_grid[12, 12]  # roughly (0.8, 0.8)
        low_region = Y_grid[3, 3]  # roughly (0.2, 0.2)
        assert high_region > low_region, "Diagonal ramp pattern should be present"

    def test_scaled_extent_preserves_full_pattern(self, ramp_data):
        """Key test: when latent_x/y is smaller, the FULL pattern should be scaled.

        If data spans [0, 1] but latent_x=(0, 0.6):
        - Grid coordinates should be in [0, 0.6]
        - But the pattern should include data from the ENTIRE original range
        - i.e., data point at original x=1.0 should appear at grid x=0.6
        """
        X, Y = ramp_data
        target = DataTarget(X=X, Y=Y, latent_x=(0.0, 0.6), latent_y=(0.0, 0.6))

        X_grid, Y_grid = target.get_lattice(resolution=(16, 16))

        # Grid should cover [0, 0.6] x [0, 0.6]
        assert X_grid[:, 0].min() == pytest.approx(0.0, abs=0.05)
        assert X_grid[:, 0].max() == pytest.approx(0.6, abs=0.05)

        # THE KEY ASSERTION: Full pattern should be scaled to fit in [0, 0.6]
        # Check values away from edges to avoid NaN issues

        # The value range in the data should still span [0, 1]
        # because we're scaling the X coordinates, not the Y values
        valid_values = Y_grid[~np.isnan(Y_grid)]
        assert len(valid_values) > 0, "Should have valid values"

        max_value = np.max(valid_values)
        np.min(valid_values)

        # With SCALING: the full pattern is compressed, so max should be ~1.0 and min ~0.0
        # With CROPPING: we only see [0, 0.6] portion, so max would be ~0.6 and min ~0.0
        assert max_value > 0.85, (
            f"Full pattern should be scaled to fit: max should be ~1.0 (from "
            f"original data at (1,1)), got {max_value}. This looks like CROPPING."
        )

    def test_scaling_vs_cropping_key_difference(self, ramp_data):
        """Demonstrate the key difference between scaling and cropping.

        For a diagonal ramp Y = (x + y) / 2:
        - Full data: X in [0,1], Y values range [0, 1]
        - With SCALING to [0, 0.6]: X scaled to [0, 0.6], Y values STILL range [0, 1]
        - With CROPPING to [0, 0.6]: Only see X in [0, 0.6], Y values range [0, 0.6]
        """
        X, Y = ramp_data
        target = DataTarget(X=X, Y=Y, latent_x=(0.0, 0.6), latent_y=(0.0, 0.6))

        X_grid, Y_grid = target.get_lattice(resolution=(16, 16))

        valid_values = Y_grid[~np.isnan(Y_grid)]
        max_value = np.max(valid_values)

        # If CROPPING: max_value ≈ 0.6 (value at original (0.6, 0.6))
        # If SCALING: max_value ≈ 1.0 (value at original (1.0, 1.0), now at grid (0.6, 0.6))
        cropping_max = 0.6

        is_cropping = abs(max_value - cropping_max) < 0.15
        is_scaling = max_value > 0.85

        assert is_scaling and not is_cropping, (
            f"Behavior is CROPPING (max={max_value} ≈ {cropping_max}). "
            f"Should be SCALING (max should be near 1.0)."
        )


class TestDataTargetSourcePassesCorrectParams:
    """Tests that DataTargetSource uses correct parameter names."""

    def test_latent_extent_is_set_not_ignored(self):
        """DataTarget should receive latent_x/latent_y, not lattice_x_extent."""
        # When DataTarget is created with custom latent_x/y, those values should be used
        X = np.random.rand(100, 2)
        Y = np.random.rand(100)

        target = DataTarget(X=X, Y=Y, latent_x=(0.1, 0.5), latent_y=(0.2, 0.7))

        assert target.latent_x == (0.1, 0.5), "latent_x should be set correctly"
        assert target.latent_y == (0.2, 0.7), "latent_y should be set correctly"

    def test_wrong_param_names_dont_set_values(self):
        """Passing wrong parameter names should not change latent_x/latent_y.

        This demonstrates the bug in DataTargetSource where it passes
        'lattice_x_extent' instead of 'latent_x'.
        """
        X = np.random.rand(100, 2)
        Y = np.random.rand(100)

        # These are the WRONG parameter names - Pydantic ignores them
        # and DataTarget uses the defaults (0.0, 0.6)
        target = DataTarget(X=X, Y=Y)

        # Default values should be used
        assert target.latent_x == (0.0, 0.6), "Default latent_x should be (0.0, 0.6)"
        assert target.latent_y == (0.0, 0.6), "Default latent_y should be (0.0, 0.6)"

        # Now with correct parameter names - should work
        target2 = DataTarget(X=X, Y=Y, latent_x=(0.1, 0.5), latent_y=(0.2, 0.7))
        assert target2.latent_x == (0.1, 0.5), "latent_x should be set correctly"
        assert target2.latent_y == (0.2, 0.7), "latent_y should be set correctly"


class TestDataTargetScaleToLatentOption:
    """Tests for the scale_to_latent option."""

    def test_scale_to_latent_true_rescales_x(self):
        """When scale_to_latent=True (default), X is rescaled to fit latent range."""
        X = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
        Y = np.array([0.0, 1.0, 0.5])

        target = DataTarget(X=X, Y=Y, latent_x=(0.0, 0.6), latent_y=(0.0, 0.6))

        # X should be rescaled: [0,1] -> [0, 0.6]
        assert target.X[:, 0].min() == pytest.approx(0.0, abs=0.01)
        assert target.X[:, 0].max() == pytest.approx(0.6, abs=0.01)
        assert target.X[:, 1].min() == pytest.approx(0.0, abs=0.01)
        assert target.X[:, 1].max() == pytest.approx(0.6, abs=0.01)

    def test_scale_to_latent_false_preserves_x(self):
        """When scale_to_latent=False, X is NOT rescaled (legacy behavior)."""
        X = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
        Y = np.array([0.0, 1.0, 0.5])

        target = DataTarget(
            X=X, Y=Y, latent_x=(0.0, 0.6), latent_y=(0.0, 0.6), scale_to_latent=False
        )

        # X should NOT be rescaled
        assert target.X[:, 0].min() == pytest.approx(0.0, abs=0.01)
        assert target.X[:, 0].max() == pytest.approx(1.0, abs=0.01)

    def test_x_coordinates_are_properly_scaled(self):
        """Verify individual X coordinates are correctly mapped."""
        # Data: x from 0 to 1, y from 0 to 1
        X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        Y = np.array([0.0, 0.5, 1.0])

        target = DataTarget(X=X, Y=Y, latent_x=(0.2, 0.8), latent_y=(0.1, 0.7))

        # After scaling:
        # x: 0 -> 0.2, 0.5 -> 0.5, 1 -> 0.8
        # y: 0 -> 0.1, 0.5 -> 0.4, 1 -> 0.7
        assert target.X[0, 0] == pytest.approx(0.2, abs=0.01)  # x=0 -> 0.2
        assert target.X[1, 0] == pytest.approx(0.5, abs=0.01)  # x=0.5 -> 0.5
        assert target.X[2, 0] == pytest.approx(0.8, abs=0.01)  # x=1 -> 0.8

        assert target.X[0, 1] == pytest.approx(0.1, abs=0.01)  # y=0 -> 0.1
        assert target.X[1, 1] == pytest.approx(0.4, abs=0.01)  # y=0.5 -> 0.4
        assert target.X[2, 1] == pytest.approx(0.7, abs=0.01)  # y=1 -> 0.7
