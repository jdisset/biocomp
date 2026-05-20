# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from jax.typing import ArrayLike
import jax.numpy as jnp
import numpy as np
from biocomp.nodeutils import (
    LayerInstance,
    empty_prepare,
)

NDArray = np.ndarray | jnp.ndarray


def single_passthrough(input_shapes: list[tuple[int]], *_, **__) -> LayerInstance:
    assert len(input_shapes) == 1, (
        f"Single passthrough node expects 1 input, got {len(input_shapes)}"
    )

    def apply(input: NDArray, **___) -> tuple[ArrayLike, dict]:
        return input, {"input_shape": input.shape}

    output_shapes = input_shapes

    return LayerInstance(empty_prepare, apply, output_shapes)


def multi_passthrough(input_shapes: list[tuple[int]], *_, **__) -> LayerInstance:
    """Passthrough for multiple inputs - just passes them through unchanged"""

    def apply(*inputs: NDArray, **___) -> tuple[ArrayLike, dict]:
        return jnp.array(inputs), {"n_inputs": len(inputs)}

    output_shapes = input_shapes

    return LayerInstance(empty_prepare, apply, output_shapes)
