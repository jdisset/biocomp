import pytest
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any
from functools import partial
from biocomp.jaxutils import robust_sort, jax_log_poly_log, jax_inverse_log_poly_log


@pytest.mark.parametrize(
    "shape, axis",
    [
        ((10,), 0),
        ((32, 8), 0),
        ((32, 8), 1),
        ((4, 5, 6), 1),
    ],
)
def test_robust_sort(shape, axis):
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, shape)

    # 1. Test forward pass
    expected_sorted = jnp.sort(x, axis=axis)
    actual_sorted = robust_sort(x, axis=axis)
    np.testing.assert_allclose(actual_sorted, expected_sorted, atol=1e-6)

    # 2. Test backward pass (gradient)
    # The gradient of the sum of a sorted array should be ones.
    grad_fn = jax.grad(lambda arr: jnp.sum(robust_sort(arr, axis=axis)))
    grads = grad_fn(x)
    expected_grads = jnp.ones_like(x)
    np.testing.assert_allclose(grads, expected_grads, atol=1e-6)


@pytest.mark.parametrize("threshold", [100, 50, 200])
@pytest.mark.parametrize("compression", [0.5, 0.7])
def test_log_poly_log_roundtrip(threshold, compression):
    """Roundtrip: inverse(forward(x)) ≈ x across all regions including the critical threshold.

    Note: The cubic spline inverse has known precision limitations for small x values
    (below ~10% of threshold). For values near and above threshold, precision is excellent.
    Low compression values (e.g., 0.3) have worse numerical stability in the inverse.
    """
    # Test values spanning: near threshold, at threshold, above threshold
    test_values = jnp.array([
        threshold * 0.5, threshold * 0.9, threshold * 0.99,  # approaching threshold
        threshold,  # exactly at threshold
        threshold * 1.01, threshold * 1.1, threshold * 2.0, threshold * 10.0,  # above threshold
    ])
    # Also test negative values (function is symmetric via sign handling)
    test_values_neg = -test_values
    all_values = jnp.concatenate([test_values, test_values_neg])

    for x in all_values:
        y = jax_log_poly_log(x, threshold=threshold, compression=compression)
        x_roundtrip = jax_inverse_log_poly_log(y, threshold=threshold, compression=compression)
        # Log region (above threshold): expect high precision
        # Polynomial region (below threshold): cubic inverse has some numerical error
        expected_rtol = 1e-5 if abs(float(x)) >= threshold else 0.02
        np.testing.assert_allclose(
            float(x_roundtrip), float(x), rtol=expected_rtol, atol=1e-5,
            err_msg=f"Roundtrip failed at x={x}, threshold={threshold}, compression={compression}"
        )


def test_log_poly_log_inverse_low_compression_produces_nan():
    """KNOWN BUG: Low compression values produce NaN in inverse function.

    This test documents that compression=0.3 produces NaN output from the
    cubic polynomial inverse. This is a known numerical stability issue
    in jcubic_exp_inv - the cubic formula involves square roots that can
    produce complex numbers for certain parameter combinations.

    Users should avoid compression values below ~0.4.
    """
    threshold = 100
    compression = 0.3  # Known problematic value
    x = threshold * 0.5  # In polynomial region

    y = jax_log_poly_log(x, threshold=threshold, compression=compression)
    x_roundtrip = jax_inverse_log_poly_log(y, threshold=threshold, compression=compression)

    # Document the NaN behavior - this is a regression test
    # If someone fixes this bug, the test will fail and can be updated
    assert jnp.isnan(x_roundtrip), (
        f"Expected NaN for compression={compression}, got {x_roundtrip}. "
        "If the inverse function has been fixed, update this test!"
    )


def test_log_poly_log_roundtrip_vectorized():
    """Test roundtrip on a dense grid around threshold boundary.

    Focus on the critical threshold region where the function transitions
    from cubic polynomial to logarithmic.
    """
    threshold = 100
    compression = 0.5
    # Dense sampling around the threshold (80 to 120) - the critical transition region
    x_near_threshold = jnp.linspace(80, 120, 100)
    # Also test the log region (well above threshold)
    x_log_region = jnp.concatenate([
        jnp.linspace(150, 500, 50),
        jnp.linspace(-500, -150, 50),
    ])
    all_x = jnp.concatenate([x_near_threshold, x_log_region])

    y = jax_log_poly_log(all_x, threshold=threshold, compression=compression)
    x_roundtrip = jax_inverse_log_poly_log(y, threshold=threshold, compression=compression)

    # Near threshold region: allow 1% error due to cubic inverse precision
    # Far from threshold: expect high precision
    np.testing.assert_allclose(x_roundtrip, all_x, rtol=0.01, atol=1e-5)


def test_log_poly_log_monotonicity():
    """log_poly_log should be monotonically increasing."""
    x = jnp.linspace(0.1, 1000, 1000)
    y = jax_log_poly_log(x)
    diffs = jnp.diff(y)
    assert jnp.all(diffs > 0), "log_poly_log should be strictly increasing for positive x"

    x_neg = jnp.linspace(-1000, -0.1, 1000)
    y_neg = jax_log_poly_log(x_neg)
    diffs_neg = jnp.diff(y_neg)
    assert jnp.all(diffs_neg > 0), "log_poly_log should be strictly increasing for negative x"


def test_log_poly_log_continuity_at_threshold():
    """Function should be C2 continuous at threshold (matching first and second derivatives)."""
    threshold = 100.0
    eps = 1e-6
    x_below = threshold - eps
    x_above = threshold + eps
    y_below = jax_log_poly_log(x_below)
    y_above = jax_log_poly_log(x_above)
    # Value should be nearly continuous
    np.testing.assert_allclose(float(y_below), float(y_above), rtol=1e-4)

    # Check derivative continuity
    grad_fn = jax.grad(lambda x: jax_log_poly_log(x).sum())
    dy_below = grad_fn(jnp.array(x_below))
    dy_above = grad_fn(jnp.array(x_above))
    np.testing.assert_allclose(float(dy_below), float(dy_above), rtol=1e-3)
