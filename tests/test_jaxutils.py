import pytest
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any
from functools import partial
from biocomp.jaxutils import robust_sort


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
