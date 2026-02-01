"""Tests for biocomp.fingerprint - deterministic network fingerprinting."""

import numpy as np

from biocomp.fingerprint import (
    _generate_canonical_grid,
    _hash_output,
    compare_fingerprints,
    FINGERPRINT_RESOLUTION,
    FINGERPRINT_SEED,
    FINGERPRINT_DECIMALS,
)


class TestCanonicalGrid:
    """Tests for _generate_canonical_grid function."""

    def test_1d_shape(self):
        grid = _generate_canonical_grid(1, 21)
        assert grid.shape == (21, 1)

    def test_2d_shape(self):
        grid = _generate_canonical_grid(2, 21)
        assert grid.shape == (441, 2)

    def test_3d_shape(self):
        grid = _generate_canonical_grid(3, 11)
        assert grid.shape == (1331, 3)

    def test_4d_fallback_shape(self):
        grid = _generate_canonical_grid(4, 11)
        assert grid.shape == (1331, 4)

    def test_values_in_range(self):
        for n_inputs in [1, 2, 3, 4]:
            grid = _generate_canonical_grid(n_inputs, 11)
            assert grid.min() >= 0.0
            assert grid.max() <= 1.0

    def test_determinism(self):
        grid1 = _generate_canonical_grid(2, 11, seed=42)
        grid2 = _generate_canonical_grid(2, 11, seed=42)
        np.testing.assert_array_equal(grid1, grid2)

    def test_different_seeds_differ(self):
        grid1 = _generate_canonical_grid(4, 11, seed=42)
        grid2 = _generate_canonical_grid(4, 11, seed=123)
        assert not np.array_equal(grid1, grid2)

    def test_dtype_is_float32(self):
        for n_inputs in [1, 2, 3, 4]:
            grid = _generate_canonical_grid(n_inputs, 11)
            assert grid.dtype == np.float32


class TestHashOutput:
    """Tests for _hash_output function."""

    def test_determinism(self):
        Y = np.array([[0.1234, 0.5678], [0.9012, 0.3456]])
        h1 = _hash_output(Y, decimals=4)
        h2 = _hash_output(Y, decimals=4)
        assert h1 == h2

    def test_length_is_16(self):
        Y = np.random.randn(100)
        h = _hash_output(Y)
        assert len(h) == 16

    def test_hex_format(self):
        Y = np.random.randn(100)
        h = _hash_output(Y)
        assert all(c in "0123456789abcdef" for c in h)

    def test_approximate_tolerance_within(self):
        Y1 = np.array([0.12341])
        Y2 = np.array([0.12342])
        assert _hash_output(Y1, decimals=4) == _hash_output(Y2, decimals=4)

    def test_approximate_tolerance_beyond(self):
        Y1 = np.array([0.1234])
        Y2 = np.array([0.1236])
        assert _hash_output(Y1, decimals=4) != _hash_output(Y2, decimals=4)

    def test_different_arrays_differ(self):
        Y1 = np.array([0.1, 0.2, 0.3])
        Y2 = np.array([0.1, 0.2, 0.4])
        assert _hash_output(Y1, decimals=4) != _hash_output(Y2, decimals=4)

    def test_order_matters(self):
        Y1 = np.array([0.1, 0.2, 0.3])
        Y2 = np.array([0.3, 0.2, 0.1])
        assert _hash_output(Y1, decimals=4) != _hash_output(Y2, decimals=4)


class TestCompareFingerprints:
    """Tests for compare_fingerprints function."""

    def test_matching_returns_true(self):
        fp = "a1b2c3d4e5f67890"
        assert compare_fingerprints(fp, fp) is True

    def test_different_returns_false(self):
        fp1 = "a1b2c3d4e5f67890"
        fp2 = "0987654321fedcba"
        assert compare_fingerprints(fp1, fp2) is False

    def test_with_context(self):
        fp = "a1b2c3d4e5f67890"
        assert compare_fingerprints(fp, fp, context="test") is True


class TestDefaults:
    """Tests for default constant values."""

    def test_resolution_default(self):
        assert FINGERPRINT_RESOLUTION == 21

    def test_seed_default(self):
        assert FINGERPRINT_SEED == 42

    def test_decimals_default(self):
        assert FINGERPRINT_DECIMALS == 4
