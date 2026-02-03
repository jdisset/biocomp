"""Tests for ratio_utils.py - the SSOT for ratio decoding and normalization."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from biocomp.ratio_utils import (
    decode_ratios,
    decode_ratios_numpy,
    normalize_ratios_for_pruning,
    RATIO_PRUNE_THRESHOLD,
)
from biocomp.parameters import ParameterTree


class TestDecodeRatiosNumpy:
    def test_direct_ratios_decoding(self):
        """Direct ratios are read and clipped correctly."""
        params = ParameterTree()
        namespace = "local/0/aggregation_2_1"
        node_idx = 0
        n_outputs = 3

        params.at(f"{namespace}/ratios", np.array([[0.5, 0.3, 0.2, 0.0]]))
        params.at(f"{namespace}/ratio_min", np.array([[0.0, 0.0, 0.0, 0.0]]))
        params.at(f"{namespace}/ratio_max", np.array([[1.0, 1.0, 1.0, 1.0]]))

        result = decode_ratios_numpy(params, namespace, node_idx, n_outputs)
        assert result.shape == (n_outputs,)
        np.testing.assert_array_almost_equal(result, np.array([0.5, 0.3, 0.2]))

    def test_direct_ratios_clipping(self):
        """Direct ratios are clipped to min/max."""
        params = ParameterTree()
        namespace = "local/0/aggregation_2_1"
        node_idx = 0
        n_outputs = 2

        params.at(f"{namespace}/ratios", np.array([[2.0, -1.0]]))
        params.at(f"{namespace}/ratio_min", np.array([[0.1, 0.1]]))
        params.at(f"{namespace}/ratio_max", np.array([[1.0, 1.0]]))

        result = decode_ratios_numpy(params, namespace, node_idx, n_outputs)
        np.testing.assert_array_almost_equal(result, np.array([1.0, 0.1]))

    def test_latent_ratios_decoding(self):
        """Latent MLP ratios are decoded and clipped correctly."""
        params = ParameterTree()
        namespace = "local/0/aggregation_2_1"
        node_idx = 0
        n_outputs = 3
        latent_dim, hidden_dim = 4, 8

        # With zero z and zero weights, output is just b2
        z = np.zeros((1, latent_dim))
        W1 = np.zeros((1, hidden_dim, latent_dim))
        b1 = np.zeros((1, hidden_dim))
        W2 = np.zeros((1, n_outputs + 1, hidden_dim))
        b2 = np.array([[0.5, 0.3, 0.2, 0.0]])

        params.at(f"{namespace}/latent_z", z)
        params.at(f"{namespace}/latent_W1", W1)
        params.at(f"{namespace}/latent_b1", b1)
        params.at(f"{namespace}/latent_W2", W2)
        params.at(f"{namespace}/latent_b2", b2)
        params.at(f"{namespace}/ratio_min", np.array([[0.0, 0.0, 0.0, 0.0]]))
        params.at(f"{namespace}/ratio_max", np.array([[1.0, 1.0, 1.0, 1.0]]))

        result = decode_ratios_numpy(params, namespace, node_idx, n_outputs)
        assert result.shape == (n_outputs,)
        np.testing.assert_array_almost_equal(result, np.array([0.5, 0.3, 0.2]), decimal=5)


class TestDecodeRatios:
    def test_direct_ratios_in_jit(self):
        """Direct ratios work in JIT context."""
        params = ParameterTree()
        namespace = "local/0/aggregation_2_1"
        node_idx = 0
        n_outputs = 3

        params.at(f"{namespace}/ratios", jnp.array([[0.5, 0.3, 0.2, 0.0]]))
        params.at(f"{namespace}/ratio_min", jnp.array([[0.0, 0.0, 0.0, 0.0]]))
        params.at(f"{namespace}/ratio_max", jnp.array([[1.0, 1.0, 1.0, 1.0]]))

        @jax.jit
        def decode(p):
            return decode_ratios(p, namespace, node_idx, n_outputs)

        result = decode(params)
        assert result.shape == (n_outputs,)
        np.testing.assert_array_almost_equal(result, jnp.array([0.5, 0.3, 0.2]))

    def test_latent_ratios_in_jit(self):
        """Latent MLP ratios work in JIT context."""
        params = ParameterTree()
        namespace = "local/0/aggregation_2_1"
        node_idx = 0
        n_outputs = 3
        latent_dim, hidden_dim = 4, 8

        z = jnp.zeros((1, latent_dim))
        W1 = jnp.zeros((1, hidden_dim, latent_dim))
        b1 = jnp.zeros((1, hidden_dim))
        W2 = jnp.zeros((1, n_outputs + 1, hidden_dim))
        b2 = jnp.array([[0.5, 0.3, 0.2, 0.0]])

        params.at(f"{namespace}/latent_z", z)
        params.at(f"{namespace}/latent_W1", W1)
        params.at(f"{namespace}/latent_b1", b1)
        params.at(f"{namespace}/latent_W2", W2)
        params.at(f"{namespace}/latent_b2", b2)
        params.at(f"{namespace}/ratio_min", jnp.array([[0.0, 0.0, 0.0, 0.0]]))
        params.at(f"{namespace}/ratio_max", jnp.array([[1.0, 1.0, 1.0, 1.0]]))

        @jax.jit
        def decode(p):
            return decode_ratios(p, namespace, node_idx, n_outputs)

        result = decode(params)
        assert result.shape == (n_outputs,)
        np.testing.assert_array_almost_equal(result, jnp.array([0.5, 0.3, 0.2]), decimal=5)


class TestNormalizeRatiosForPruning:
    def test_1d_normalization(self):
        """1D ratios are normalized correctly."""
        ratios = jnp.array([0.5, 0.1, 0.25, 0.04])
        result = normalize_ratios_for_pruning(ratios, threshold=0.1)
        assert result.shape == (4,)
        assert float(result[0]) == 1.0  # max value
        assert float(result[3]) == 0.0  # below threshold (0.04/0.5 = 0.08 < 0.1)

    def test_2d_normalization(self):
        """2D ratios are normalized per row."""
        ratios = jnp.array([
            [0.5, 0.1, 0.05],
            [1.0, 0.2, 0.01],
        ])
        result = normalize_ratios_for_pruning(ratios, threshold=0.1)
        assert result.shape == (2, 3)
        assert float(result[0, 0]) == 1.0  # max of row 0
        assert float(result[1, 0]) == 1.0  # max of row 1
        assert float(result[1, 2]) == 0.0  # 0.01/1.0 < 0.1

    def test_scalar_passthrough(self):
        """Scalar values are returned as-is."""
        result = normalize_ratios_for_pruning(jnp.array(0.5))
        assert float(result) == 0.5

    def test_works_with_numpy(self):
        """Function works with numpy arrays."""
        ratios = np.array([0.5, 0.1, 0.04])
        result = normalize_ratios_for_pruning(ratios, threshold=0.1)
        assert isinstance(result, np.ndarray)
        assert float(result[0]) == 1.0
        assert float(result[2]) == 0.0  # 0.04/0.5 = 0.08 < 0.1

    def test_works_with_jax(self):
        """Function works with JAX arrays."""
        ratios = jnp.array([0.5, 0.1, 0.04])
        result = normalize_ratios_for_pruning(ratios, threshold=0.1)
        assert hasattr(result, 'at')  # JAX array
        assert float(result[0]) == 1.0
        assert float(result[2]) == 0.0  # 0.04/0.5 = 0.08 < 0.1

    def test_threshold_constant_matches_config(self):
        """RATIO_PRUNE_THRESHOLD matches config value."""
        from biocomp.config import BIOCOMP_CONSTANTS
        expected = BIOCOMP_CONSTANTS["ratio"]["prune_threshold"]
        assert RATIO_PRUNE_THRESHOLD == expected


class TestBackwardsCompatibility:
    def test_design_import_works(self):
        """Old imports from design.py still work."""
        from biocomp.design import normalize_ratios_prune

        ratios = jnp.array([0.5, 0.1, 0.05])
        result = normalize_ratios_prune(ratios, threshold=0.1)
        assert float(result[0]) == 1.0

    def test_pluggable_opt_import_works(self):
        """pluggable_opt can still import from design."""
        from biocomp.design import normalize_ratios_prune, get_ratio_paths_and_sources, RATIO_PRUNE_THRESHOLD
        assert callable(normalize_ratios_prune)
        assert callable(get_ratio_paths_and_sources)
        assert isinstance(RATIO_PRUNE_THRESHOLD, float)


def test_aggregation_ratio_min_uses_recipe_constant():
    """Verify aggregation.py uses DEFAULT_RATIO_MIN from recipe, not hardcoded 0.05."""
    import importlib
    from biocomp.recipe import DEFAULT_RATIO_MIN

    assert DEFAULT_RATIO_MIN == 0.001, "recipe constant should be 0.001"

    # Force module load (not the function re-exported from nodes/__init__.py)
    aggregation_module = importlib.import_module("biocomp.nodes.aggregation")
    assert hasattr(aggregation_module, 'DEFAULT_RATIO_MIN'), "aggregation should import DEFAULT_RATIO_MIN"
    assert aggregation_module.DEFAULT_RATIO_MIN == DEFAULT_RATIO_MIN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
