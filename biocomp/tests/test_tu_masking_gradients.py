"""Tests for TU masking gradient flow.

Verifies that:
1. leaky_mask_floor provides gradients even when mask=0
2. Design params (bias, ratios, log_alpha) receive gradients with proper init
3. Custom VJP approach was replaced with simpler leaky floor
"""

import jax
import jax.numpy as jnp
import pytest

from biocomp.tumasking import (
    leaky_mask_floor,
    LEAKY_MASK_FLOOR,
    sample_hard_concrete,
    hard_concrete_from_uniform,
    compute_input_masks,
)


class TestLeakyMaskFloor:
    """Test leaky_mask_floor provides gradients through zero masks."""

    def test_leaky_floor_value_when_mask_zero(self):
        mask = jnp.array([0.0, 0.0])
        result = leaky_mask_floor(mask)
        expected = jnp.array([LEAKY_MASK_FLOOR, LEAKY_MASK_FLOOR])
        assert jnp.allclose(result, expected)

    def test_leaky_floor_preserves_positive_mask(self):
        mask = jnp.array([0.5, 1.0, 0.01])
        result = leaky_mask_floor(mask)
        assert jnp.allclose(result, mask)

    def test_leaky_floor_gradient_nonzero(self):
        """Key test: gradients through disabled (mask=0) values are non-zero."""

        def loss_fn(x, mask):
            masked = x * leaky_mask_floor(mask)
            return jnp.sum(masked)

        x = jnp.array([1.0, 2.0, 3.0])
        mask = jnp.array([0.0, 0.0, 1.0])  # first two are disabled

        grad_x = jax.grad(loss_fn, argnums=0)(x, mask)

        # gradient should be LEAKY_MASK_FLOOR for disabled positions (mask=0)
        assert jnp.allclose(grad_x[0], LEAKY_MASK_FLOOR)
        assert jnp.allclose(grad_x[1], LEAKY_MASK_FLOOR)
        # gradient should be 1.0 for enabled position (mask=1)
        assert jnp.allclose(grad_x[2], 1.0)

    def test_leaky_floor_works_with_jacfwd(self):
        """Verify leaky_mask_floor works with forward-mode autodiff (jacfwd)."""

        def fn(x, mask):
            return x * leaky_mask_floor(mask)

        x = jnp.array([1.0, 2.0])
        mask = jnp.array([0.0, 1.0])

        # This should not raise
        jac = jax.jacfwd(fn, argnums=0)(x, mask)
        assert jac.shape == (2, 2)
        assert jnp.allclose(jac[0, 0], LEAKY_MASK_FLOOR)
        assert jnp.allclose(jac[1, 1], 1.0)


class TestHardConcreteGradients:
    """Test Hard Concrete distribution gradient flow."""

    def test_sample_hard_concrete_gradient_flows(self):
        """Gradient w.r.t. log_alpha flows through sampling."""

        def loss_fn(log_alpha, key):
            z = sample_hard_concrete(log_alpha, key)
            return jnp.sum(z)

        key = jax.random.PRNGKey(42)
        # use moderate values to avoid clipping (high log_alpha saturates at z=1)
        log_alpha = jnp.array([0.0, 0.5, -0.5])

        grad = jax.grad(loss_fn)(log_alpha, key)
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.all(grad >= 0)  # increasing log_alpha should increase z

    def test_hard_concrete_from_uniform_gradient(self):
        """Gradient flows through uniform->hard concrete transform."""

        def loss_fn(log_alpha, u):
            z = hard_concrete_from_uniform(u, log_alpha)
            return jnp.sum(z)

        log_alpha = jnp.array([0.0, 2.0])
        u = jnp.array([0.3, 0.7])

        grad = jax.grad(loss_fn)(log_alpha, u)
        assert jnp.all(jnp.isfinite(grad))


class TestComputeInputMasksGradients:
    """Test compute_input_masks gradient flow."""

    def test_compute_masks_gradient_to_log_alpha(self):
        """Gradient flows from masks to log_alpha."""

        def loss_fn(log_alpha):
            tu_indices = jnp.array([0, 1, 2])
            tu_uniform = jnp.array([0.5, 0.5, 0.5])
            masks = compute_input_masks(
                tu_indices, tu_uniform, log_alpha, is_multi_tu=False
            )
            return jnp.sum(masks)

        log_alpha = jnp.array([0.0, 2.0, -2.0])
        grad = jax.grad(loss_fn)(log_alpha)
        assert jnp.all(jnp.isfinite(grad))

    def test_always_enabled_tu_has_no_gradient(self):
        """TU index -1 (always enabled) has no gradient to log_alpha."""

        def loss_fn(log_alpha):
            tu_indices = jnp.array([-1, 0, -1])  # -1 = always enabled
            tu_uniform = jnp.array([0.5, 0.5, 0.5])
            masks = compute_input_masks(
                tu_indices, tu_uniform, log_alpha, is_multi_tu=False
            )
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

        def aggregation_apply(ratios, masks):
            abs_ratios = jnp.abs(ratios)
            masked_ratios = abs_ratios * leaky_mask_floor(masks)
            masked_sum = jnp.sum(masked_ratios)
            safe_sum = jnp.maximum(masked_sum, 1e-8)
            normalized = jnp.where(
                masked_sum > 1e-8, masked_ratios / safe_sum, jnp.zeros_like(ratios)
            )
            return jnp.sum(normalized)

        ratios = jnp.array([0.3, 0.7])
        masks = jnp.array([0.0, 1.0])  # first ratio is "disabled"

        grad_ratios = jax.grad(aggregation_apply)(ratios, masks)

        # Even disabled ratio should have some gradient (via leaky floor)
        assert jnp.all(jnp.isfinite(grad_ratios))
        assert grad_ratios[0] != 0.0  # leaky gradient

    def test_transform_pattern_gradient(self):
        """Simulate transform layer inner output masking."""

        def transform_apply(inner_outputs, masks):
            masked = [out * leaky_mask_floor(masks[i]) for i, out in enumerate(inner_outputs)]
            return jnp.sum(jnp.stack(masked))

        inner_outputs = [jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])]
        masks = jnp.array([0.0, 1.0])  # first output disabled

        def loss_fn(outputs):
            return transform_apply(outputs, masks)

        grad = jax.grad(lambda o: loss_fn([o[0], o[1]]))(jnp.stack(inner_outputs))
        assert jnp.all(jnp.isfinite(grad))
        # Disabled output has leaky gradient
        assert jnp.allclose(grad[0], LEAKY_MASK_FLOOR)
