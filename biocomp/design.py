import jax.numpy as jnp


def single_l2loss(yhat, target_y):  # simplest possible loss (certainly pretty bad)
    assert yhat.shape == target_y.shape, f"Shape mismatch: {yhat.shape} != {target_y.shape}"
    assert yhat.ndim == 1, f"Expected 1D arrays, got {yhat.ndim}D"
    return jnp.mean(jnp.square(yhat - target_y))

