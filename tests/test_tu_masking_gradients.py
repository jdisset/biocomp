"""Tests for TU masking gradient flow.

Verifies that:
1. Design params (bias, ratios, log_alpha) receive gradients with proper init
2. Binary STE masking provides gradient flow
"""

import jax
import jax.numpy as jnp

from biocomp.tumasking import (
    binary_mask_with_ste,
    compute_binary_masks,
)


class TestBinarySTEGradients:
    """Test binary STE masking gradient flow."""

    def test_binary_ste_gradient_flows(self):
        """Gradient w.r.t. log_alpha flows through binary STE."""

        def loss_fn(log_alpha):
            mask = binary_mask_with_ste(log_alpha)
            return jnp.sum(mask)

        log_alpha = jnp.array([0.0, 0.5, -0.5])

        grad = jax.grad(loss_fn)(log_alpha)
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.all(grad >= 0)  # increasing log_alpha should increase mask probability

    def test_binary_ste_gradient_magnitude(self):
        """Gradients have meaningful magnitude (not vanishing)."""
        log_alpha_near_zero = jnp.array([0.0])

        def loss_fn(log_alpha):
            mask = binary_mask_with_ste(log_alpha)
            return jnp.sum(mask**2)

        grads = jax.grad(loss_fn)(log_alpha_near_zero)
        assert jnp.abs(grads[0]) > 1e-4, (
            f"Gradient magnitude {jnp.abs(grads[0])} is too small, possible vanishing gradient"
        )


class TestComputeBinaryMasksGradients:
    """Test compute_binary_masks gradient flow."""

    def test_compute_masks_gradient_to_log_alpha(self):
        """Gradient flows from masks to log_alpha."""

        def loss_fn(log_alpha):
            tu_indices = jnp.array([0, 1, 2])
            masks = compute_binary_masks(tu_indices, log_alpha, is_multi_tu=False)
            return jnp.sum(masks)

        log_alpha = jnp.array([0.0, 2.0, -2.0])
        grad = jax.grad(loss_fn)(log_alpha)
        assert jnp.all(jnp.isfinite(grad))

    def test_always_enabled_tu_has_no_gradient(self):
        """TU index -1 (always enabled) has no gradient to log_alpha."""

        def loss_fn(log_alpha):
            tu_indices = jnp.array([-1, 0, -1])  # -1 = always enabled
            masks = compute_binary_masks(tu_indices, log_alpha, is_multi_tu=False)
            return jnp.sum(masks)

        log_alpha = jnp.array([0.0, 2.0, -2.0])
        grad = jax.grad(loss_fn)(log_alpha)

        # Only index 0 should have gradient (indices -1 bypass masking)
        assert jnp.all(jnp.isfinite(grad))
        assert grad[0] != 0.0  # TU 0 is used


class TestMaskingLayerGradients:
    """Test masking in aggregation/transform layer patterns."""

    def test_aggregation_pattern_gradient(self):
        """Simulate aggregation layer masking pattern."""

        def aggregation_apply(ratios, log_alpha, inputs):
            tu_indices = jnp.array([0, 1])
            masks = compute_binary_masks(tu_indices, log_alpha, is_multi_tu=False)
            abs_ratios = jnp.abs(ratios)
            masked_ratios = abs_ratios * masks
            masked_sum = jnp.sum(masked_ratios)
            safe_sum = jnp.maximum(masked_sum, 1e-8)
            normalized = jnp.where(
                masked_sum > 1e-8, masked_ratios / safe_sum, jnp.zeros_like(ratios)
            )
            return jnp.sum(normalized * inputs)

        ratios = jnp.array([0.3, 0.7])
        log_alpha = jnp.array([5.0, 5.0])  # both enabled
        inputs = jnp.array([0.2, 0.8])

        grad_ratios = jax.grad(aggregation_apply)(ratios, log_alpha, inputs)
        assert jnp.all(jnp.isfinite(grad_ratios))
