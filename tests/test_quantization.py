"""Test cases for the quantization system in biocompiler."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any

from biocomp.quantization import (
    quantize_all_values_to_nearest_masked_embeddings,
    get_variational_quantized,
    get_quantized,
    get_nearest_masked_id,
)
from biocomp.parameters import ParameterTree, ArrayRef
from biocomp import jaxutils


@pytest.mark.parametrize(
    "x, possible_values, mask, expected",
    [
        # 1D with all-true mask
        (
            jnp.array([[2.3], [5.7], [8.1]]),
            jnp.array([[1.0], [4.0]]),
            jnp.array([True, True]),
            jnp.array([[1.0], [4.0], [4.0]]),  # nearest values
        ),
        (
            jnp.array([[2.3], [5.7], [9.0]]),
            jnp.array([[1.0], [3.0], [5.5], [7.0], [9.0]]),
            jnp.array([True, True, False, False, True]),
            jnp.array([[3.0], [3.0], [9.0]]),  # nearest from allowed values
        ),
        # 2D cases
        (
            jnp.array([[2.3, 3.7], [8.1, 9.2], [-6.2, 1.1]]),
            jnp.array([[-5, 2], [2, 3], [0, 0], [15, 12], [8, 9]]),
            jnp.array([True, True, False, True, False]),
            jnp.array([[2, 3], [15, 12], [-5.0, 2.0]]),  # nearest from allowed values
        ),
    ],
)
def test_quantize_masked_parametrized(x, possible_values, mask, expected):
    """Test quantize_masked with various input dimensions and mask configurations."""

    result = quantize_all_values_to_nearest_masked_embeddings(x, possible_values, mask)
    assert jnp.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_quantize_masked_straight_through_gradient():
    """Test that gradients pass through unchanged (straight-through estimator)."""

    def loss_fn(x):
        possible_values = jnp.array([[1.0], [3.0]])
        mask = jnp.array([True, True])
        quantized = quantize_all_values_to_nearest_masked_embeddings(x, possible_values, mask)
        return jnp.sum(quantized**2)

    x = jnp.array([[2.2]])
    val, grad = jax.value_and_grad(loss_fn)(x)
    expected_val = 9.0  # (3.0)^2
    expected_grad = np.array([[6.0]])
    assert np.isclose(np.asarray(val), expected_val)
    assert np.allclose(np.asarray(grad), expected_grad)


def test_get_quantized_basic():
    params = ParameterTree()

    qpath = "shared/quantization/values/test_rate"
    params[qpath] = jnp.array([[0.1], [0.5], [1.0]])

    mpath = "local/layer0/test_rate_mask"
    params[mpath] = jnp.array(
        [
            [  # node 0 mask
                [True, False, True],
                [True, True, False],
                [True, False, True],
                [False, False, True],
            ]
        ]
    )

    # values to quantize
    values = np.array([[0.3], [4.0], [0.65], [-2.0]])

    result, aux = get_quantized(values, params, qpath, mpath, node_id=0)

    expected = np.array([[0.1], [0.5], [1.0], [1.0]])

    assert jnp.allclose(result, expected)


def test_get_variational_quantized():
    params = ParameterTree()

    qpath = "shared/quantization/values/test_rate"
    means = jnp.array([[0.1], [0.5], [1.0]])
    params[qpath] = means

    mpath = "local/layer0/test_rate_mask"
    params[mpath] = jnp.array(
        [
            [  # node 0 mask
                [True, False, True],
                [True, True, False],
                [True, False, True],
                [False, False, True],
            ]
        ]
    )

    lpath = "shared/quantization/logstdevs/test_rate"
    logstdevs = jnp.array([[-2.0], [-1.0], [0.0]])
    actual_stdevs = np.exp(logstdevs)
    params[lpath] = logstdevs

    values = np.array([[0.3], [4.0], [0.65], [-2.0]])

    base_key = jax.random.PRNGKey(42)
    N_REPEAT = 5000
    all_keys = jax.random.split(base_key, N_REPEAT)

    @jax.vmap
    def vmapped_get_variational_quantized(key):
        return get_variational_quantized(values, params, qpath, mpath, lpath, node_id=0, key=key)[0]

    result = jax.jit(vmapped_get_variational_quantized)(all_keys)

    expected_ids = np.array([0, 1, 2, 2])
    expected_means = means[expected_ids]
    expected_stds = actual_stdevs[expected_ids]

    mean_result = np.mean(result, axis=0)
    std_result = np.std(result, axis=0)

    print(f"Mean result: {mean_result}, Std result: {std_result}")
    print(f"Expected means: {expected_means}, Expected stds: {expected_stds}")

    assert jnp.allclose(mean_result, expected_means, atol=0.05)
    assert jnp.allclose(std_result, expected_stds, atol=0.05)


def test_get_nearest_masked_id_empty_mask_raises():
    """Empty mask (all False) should raise AssertionError - design is impossible."""
    x = jnp.array([0.5])
    qvalues = jnp.array([[0.1], [0.5], [1.0]])
    empty_mask = jnp.array([False, False, False])

    with pytest.raises(AssertionError, match="no valid options"):
        get_nearest_masked_id(x, qvalues, empty_mask)


def test_get_nearest_masked_id_empty_mask_with_checkify():
    """Empty mask should raise CheckError when checkify is enabled."""
    from jax.experimental import checkify

    x = jnp.array([0.5])
    qvalues = jnp.array([[0.1], [0.5], [1.0]])
    empty_mask = jnp.array([False, False, False])

    jaxutils.set_enable_checks(True)
    try:
        checkified_fn = checkify.checkify(
            lambda: get_nearest_masked_id(x, qvalues, empty_mask),
            errors=checkify.user_checks,
        )
        err, _ = jax.jit(checkified_fn)()
        # err.throw() should raise because mask is empty
        with pytest.raises(checkify.JaxRuntimeError, match="no valid options"):
            err.throw()
    finally:
        jaxutils.set_enable_checks(False)


def test_get_nearest_masked_id_single_valid_option():
    """With only one valid option, should always select it."""
    x = jnp.array([0.5])
    qvalues = jnp.array([[0.1], [0.5], [1.0]])
    single_mask = jnp.array([False, False, True])

    idx = get_nearest_masked_id(x, qvalues, single_mask)
    assert idx == 2
