# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Tests for latent ratio overparameterization feature.

Tests use real components (no mocks) to verify:
1. Latent decoder MLP produces valid, differentiable outputs
2. DesignConfig fields are correctly added
3. Gradients flow through the full latent -> ratio -> loss path
"""

import jax
import jax.numpy as jnp
import numpy as np

from biocomp.ratio_utils import _decode_latent_ratios
from biocomp.design import DesignConfig


class TestDecodeLatentRatios:
    """Test the MLP decoding function - the core of latent ratio parameterization."""

    def test_decode_produces_finite_output(self):
        z = jnp.zeros(8)
        W1 = jnp.ones((16, 8)) * 0.1
        b1 = jnp.zeros(16)
        W2 = jnp.ones((3, 16)) * 0.1
        b2 = jnp.array([0.3, 0.3, 0.4])
        result = _decode_latent_ratios(z, W1, b1, W2, b2)
        assert jnp.isfinite(result).all()
        assert result.shape == (3,)

    def test_decode_is_differentiable(self):
        z = jnp.zeros(8)
        W1 = jnp.ones((16, 8)) * 0.1
        b1 = jnp.zeros(16)
        W2 = jnp.ones((3, 16)) * 0.1
        b2 = jnp.array([0.3, 0.3, 0.4])

        def loss_fn(z):
            ratios = _decode_latent_ratios(z, W1, b1, W2, b2)
            return jnp.sum(ratios**2)

        grad = jax.grad(loss_fn)(z)
        assert jnp.isfinite(grad).all()
        assert grad.shape == z.shape

    def test_decode_with_zero_weights_uses_bias(self):
        """With zero weights, output should equal bias (no hidden activation contribution)."""
        z = jnp.zeros(8)
        W1 = jnp.zeros((16, 8))
        b1 = jnp.zeros(16)
        W2 = jnp.zeros((3, 16))
        b2 = jnp.array([0.2, 0.3, 0.5])
        result = _decode_latent_ratios(z, W1, b1, W2, b2)
        np.testing.assert_allclose(result, b2, rtol=1e-5)


class TestDesignConfigLatentFields:
    """Test DesignConfig has latent ratio fields with correct defaults."""

    def test_default_latent_ratios_disabled(self):
        config = DesignConfig()
        assert config.use_latent_ratios is False
        assert config.latent_dim == 8
        assert config.latent_hidden_dim == 16

    def test_latent_ratios_can_be_enabled(self):
        config = DesignConfig(use_latent_ratios=True, latent_dim=16, latent_hidden_dim=32)
        assert config.use_latent_ratios is True
        assert config.latent_dim == 16
        assert config.latent_hidden_dim == 32


class TestLatentRatioGradientFlow:
    """Test gradients flow through latent decoder to all parameters."""

    def test_gradients_propagate_to_latent_z(self):
        """Verify non-zero gradients reach z when optimizing toward target ratios."""
        z = jnp.zeros(8)
        W1 = jax.random.normal(jax.random.PRNGKey(0), (16, 8)) * 0.1
        b1 = jnp.zeros(16)
        W2 = jax.random.normal(jax.random.PRNGKey(1), (3, 16)) * 0.1
        b2 = jnp.array([0.3, 0.3, 0.4])
        target = jnp.array([0.5, 0.3, 0.2])

        def loss_fn(z):
            ratios = _decode_latent_ratios(z, W1, b1, W2, b2)
            abs_ratios = jnp.abs(ratios)
            normalized = abs_ratios / jnp.sum(abs_ratios)
            return jnp.sum((normalized - target) ** 2)

        grad = jax.grad(loss_fn)(z)
        assert jnp.isfinite(grad).all()
        assert not jnp.allclose(grad, 0.0), "Gradients should be non-zero"

    def test_gradients_propagate_to_all_decoder_params(self):
        """All decoder params (z, W1, b1, W2, b2) should receive gradients."""
        z = jnp.zeros(8)
        W1 = jax.random.normal(jax.random.PRNGKey(0), (16, 8)) * 0.1
        b1 = jnp.zeros(16)
        W2 = jax.random.normal(jax.random.PRNGKey(1), (3, 16)) * 0.1
        b2 = jnp.array([0.3, 0.3, 0.4])
        target = jnp.array([0.5, 0.3, 0.2])

        def loss_fn(params):
            ratios = _decode_latent_ratios(
                params["z"], params["W1"], params["b1"], params["W2"], params["b2"]
            )
            abs_ratios = jnp.abs(ratios)
            normalized = abs_ratios / jnp.sum(abs_ratios)
            return jnp.sum((normalized - target) ** 2)

        params = {"z": z, "W1": W1, "b1": b1, "W2": W2, "b2": b2}
        grads = jax.grad(loss_fn)(params)

        for name, grad in grads.items():
            assert jnp.isfinite(grad).all(), f"Non-finite grad for {name}"

    def test_optimization_reduces_loss(self):
        """Simple gradient descent should reduce loss toward target."""
        key = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(key, 3)

        z = jax.random.normal(k1, (8,)) * 0.1
        W1 = jax.random.normal(k2, (16, 8)) * 0.1
        b1 = jnp.zeros(16)
        W2 = jax.random.normal(k3, (3, 16)) * 0.1
        b2 = jnp.array([0.3, 0.3, 0.4])
        target = jnp.array([0.6, 0.3, 0.1])

        def loss_fn(z):
            ratios = _decode_latent_ratios(z, W1, b1, W2, b2)
            abs_ratios = jnp.abs(ratios)
            normalized = abs_ratios / jnp.sum(abs_ratios)
            return jnp.sum((normalized - target) ** 2)

        initial_loss = loss_fn(z)
        for _ in range(50):
            grad = jax.grad(loss_fn)(z)
            z = z - 0.1 * grad
        final_loss = loss_fn(z)

        assert final_loss < initial_loss, f"Loss should decrease: {initial_loss:.4f} -> {final_loss:.4f}"
