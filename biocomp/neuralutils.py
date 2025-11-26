from __future__ import annotations
import jax
import jax.nn
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from typing import Callable

PRNGKey = ArrayLike


def uniform_initializer(rng, shape=(), minval=0, maxval=1):
    def init():
        return jax.random.uniform(
            key=rng, shape=shape, minval=minval, maxval=maxval, dtype=jnp.float32
        )

    return init


def glorot_normal(rng, shape):
    def init():
        return jax.nn.initializers.glorot_normal()(rng, shape)

    return init


def glorot_uniform(rng, shape):
    def init():
        return jax.nn.initializers.glorot_uniform()(rng, shape)

    return init


def he_normal(rng, shape):
    def init():
        return jax.nn.initializers.he_normal()(rng, shape)

    return init


def he_uniform(rng, shape):
    def init():
        return jax.nn.initializers.he_uniform()(rng, shape)

    return init


def leaky_relu(x, alpha=0.2):
    return jax.nn.leaky_relu(x, negative_slope=alpha)


def sigmoid(x):
    return jax.nn.sigmoid(x)


ACTIVATION_FUNCTIONS = {
    "leaky_relu": leaky_relu,
    "relu": jax.nn.relu,
    "elu": jax.nn.elu,
    "selu": jax.nn.selu,
    "tanh": jax.nn.tanh,
    "gelu": jax.nn.gelu,
    "softplus": jax.nn.softplus,
    "sigmoid": sigmoid,
    "none": lambda x: x,
}

INITIALIZERS = {
    "uniform": uniform_initializer,
    "glorot_normal": glorot_normal,
    "glorot_uniform": glorot_uniform,
    "he_normal": he_normal,
    "he_uniform": he_uniform,
    "xavier_normal": glorot_normal,  # alias
    "xavier_uniform": glorot_uniform,  # alias
}

DEFAULT_ACTIVATION = "leaky_relu"
DEFAULT_OUT_ACTIVATION = "sigmoid"
DEFAULT_INITIALIZER = "he_normal"

def dense_layer(
    input_values: ArrayLike,
    output_size: ArrayLike,
    param_f: Callable,
    initializer: Callable,
    bias_offset,
    key: PRNGKey,
    name: str,
):
    assert len(input_values.shape) == 1, f"In {name}: input_values should be a 1D array."
    input_size = 1 if input_values.shape == () else input_values.shape[0]

    w = param_f(f"{name}/w", init_f=initializer(key, (input_size, output_size)))
    b = param_f(f"{name}/b", init_f=lambda: np.zeros((output_size,)) + bias_offset)

    assert input_values.shape == (input_size,), (
        f"In {name}: {input_values.shape} != {(input_size,)}"
    )
    assert w.shape == (
        input_size,
        output_size,
    ), f"In {name}: {w.shape} != {(input_size, output_size)}"
    assert b.shape == (output_size,), f"In {name}: {b.shape} != {(output_size,)}"

    assert w.shape == (
        input_size,
        output_size,
    ), f"In {name}: {w.shape} != {(input_size, output_size)}"

    res = jnp.dot(input_values, w) + b
    assert res.shape == (output_size,), f"In {name}: {res.shape} != {(output_size,)}"
    return res


def layer_norm(x, param_f: Callable, name: str, axis=-1, epsilon=1e-5, gamma_init=0.1):
    mean = jnp.mean(x, axis=axis, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=axis, keepdims=True)
    xhat = (x - mean) / jnp.sqrt(var + epsilon)
    gamma = param_f(f"{name}/gamma", init_f=lambda: jnp.ones(x.shape[axis]) * gamma_init)
    beta = param_f(f"{name}/beta", init_f=lambda: jnp.zeros(x.shape[axis]))

    return gamma * xhat + beta


def dense_mlp(
    input_values: ArrayLike,
    hidden_s: int,
    output_s: int,
    depth: int,
    param_f: Callable[[str, Callable], ArrayLike],
    initializer: Callable,
    bias_offset,
    key: PRNGKey,
    name: str,
    activation: Callable[[ArrayLike], ArrayLike],
):
    assert len(input_values.shape) == 1, f"In {name}: input_values should be a 1D array."
    assert isinstance(depth, int) and depth >= 1, (
        f"In {name}: depth should be an integer greater than or equal to 1."
    )
    assert isinstance(hidden_s, int) and hidden_s > 0, (
        f"In {name}: hidden_s should be a positive integer."
    )
    assert isinstance(output_s, int) and output_s > 0, (
        f"In {name}: output_s should be a positive integer."
    )

    res = input_values
    keys = jax.random.split(key, depth)
    for i in range(depth - 1):
        pre = dense_layer(res, hidden_s, param_f, initializer, bias_offset, keys[i], f"{name}/l{i}")
        normed = layer_norm(pre, param_f, f"{name}/l{i}/norm")
        res = activation(normed)
        assert res.shape == (hidden_s,), f"In {name}: {res.shape} != {(hidden_s,)}"

    res = dense_layer(
        res, output_s, param_f, initializer, bias_offset, keys[-1], f"{name}/l{depth - 1}"
    )
    assert res.shape == (output_s,), f"In {name}: {res.shape} != {(output_s,)}"
    return res


def dummy_mlp(
    input_values: ArrayLike,
    hidden_s: int,
    output_s: int,
    depth: int,
    param_f: Callable[[str, Callable], ArrayLike],
    initializer: Callable,
    bias_offset,
    key: PRNGKey,
    name: str,
    activation: Callable[[ArrayLike], ArrayLike],
):
    """A dummy non-neural module that just returns the sum of the input repeated to match output size."""

    assert len(input_values.shape) == 1, f"In {name}: input_values should be a 1D array."
    assert isinstance(depth, int) and depth >= 1, (
        f"In {name}: depth should be an integer greater than or equal to 1."
    )
    assert isinstance(hidden_s, int) and hidden_s > 0, (
        f"In {name}: hidden_s should be a positive integer."
    )
    assert isinstance(output_s, int) and output_s > 0, (
        f"In {name}: output_s should be a positive integer."
    )

    sum_val = jnp.sum(input_values) if input_values.shape != () else input_values
    res = jnp.full((output_s,), sum_val)
    assert res.shape == (output_s,), f"In {name}: {res.shape} != {(output_s,)}"
    return res
