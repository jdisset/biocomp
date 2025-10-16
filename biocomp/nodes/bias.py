from biocomp.compute import StackNode
import jax
from jax.typing import ArrayLike
from typing import Optional
import jax.numpy as jnp
import numpy as np
from biocomp.parameters import ParameterTree
from biocomp.nodeutils import (
    LayerInstance,
)
from biocomp.utils import get_logger


PRNGKey = ArrayLike
NDArray = np.ndarray | jnp.ndarray

logger = get_logger(__name__)


def hard_bias(
    input_shapes: list[tuple[int]],
    n_outputs: int,
    stack,
    namespace: str,
    valid_range: tuple[float, float] = (0.0, 0.8),
    shape: tuple[int] = (1,),
    init_value: Optional[float] = 0.5,
    random_init: bool = False,
    **_,
) -> LayerInstance:
    assert n_outputs == 1, f"Bias node should have 1 output, got {n_outputs}"
    assert len(input_shapes) == 0

    def clamp_to_range(value: ArrayLike):
        # hard clamp to valid_range. scale is ignored
        return jnp.clip(value, valid_range[0], valid_range[1])

    def prepare(params: ParameterTree, nodelist: list[StackNode], key, **_):
        raw_values = []

        for node in nodelist:
            extra = node.get(stack).extra
            if extra and not random_init:
                # try to use values from extra dict
                if "raw_value" in extra:
                    raw_values.append(
                        jnp.array(extra["raw_value"], dtype=jnp.float32).reshape(shape)
                    )
                elif "bias_value" in extra:
                    # use bias_value directly as raw_value (will be clamped in apply)
                    raw_values.append(
                        jnp.array(extra["bias_value"], dtype=jnp.float32).reshape(shape)
                    )
                else:
                    # no valid values in extra, use init_value or random
                    if init_value is not None:
                        raw_values.append(jnp.full(shape, init_value, dtype=jnp.float32))
                    else:
                        raw_values.append(
                            jax.random.uniform(
                                key, shape, minval=valid_range[0], maxval=valid_range[1]
                            )
                        )
            else:
                # random init requested or no extra dict
                if init_value is not None and not random_init:
                    raw_values.append(jnp.full(shape, init_value, dtype=jnp.float32))
                else:
                    raw_values.append(
                        jax.random.uniform(key, shape, minval=valid_range[0], maxval=valid_range[1])
                    )

        params[f"{namespace}/raw_value"] = jnp.stack(raw_values)

    def apply(*_, params: ParameterTree, node_id: ArrayLike, **__) -> tuple[ArrayLike, dict]:
        raw_bias_value = params[f"{namespace}/raw_value"][node_id]
        bias_value = clamp_to_range(raw_bias_value)
        return bias_value, {
            "raw_bias_value": raw_bias_value,
            "bias_value": bias_value,
        }

    def commit(params: ParameterTree, nodelist: list[StackNode], **_):
        for i, n in enumerate(nodelist):
            newextra = {}
            bias_value = clamp_to_range(params[f"{namespace}/raw_value"][i])
            newextra["bias_value"] = bias_value
            n.get(stack).extra.update(newextra)

    output_shapes = [tuple(shape)]  # single output shape

    return LayerInstance(prepare, apply, output_shapes, commit=commit)


def bias(
    input_shapes: list[tuple[int]],
    n_outputs: int,
    stack,
    namespace: str,
    valid_range: tuple[float, float] = (0.0, 0.6),
    shape: tuple[int] = (1,),
    random_init: bool = False,
    **_,
) -> LayerInstance:
    """A bias node that outputs a learnable (softer) bias value within a specified valid range."""
    assert n_outputs == 1, f"Bias node should have 1 output, got {n_outputs}"
    assert len(input_shapes) == 0

    def clamp_to_range(value: ArrayLike, scale: ArrayLike):
        # scaled sigmoid function to clamp value to valid_range. smaller scale = sharper transition
        s = jax.nn.sigmoid(scale) + 0.001
        scaled_sigmoid = jax.nn.sigmoid(value / s)
        return scaled_sigmoid * (valid_range[1] - valid_range[0]) + valid_range[0]

    def inverse_clamp(value: ArrayLike, scale: ArrayLike):
        # inverse of clamp_to_range, to get raw_value from bias_value and scale
        s = jax.nn.sigmoid(scale) + 0.001
        y = (value - valid_range[0]) / (valid_range[1] - valid_range[0])
        y = jnp.clip(y, 1e-5, 1 - 1e-5)
        return -s * jnp.log((1 / y) - 1)

    def prepare(params: ParameterTree, nodelist: list[StackNode], key, **_):
        raw_values = []
        scales = []

        for node in nodelist:
            extra = node.get(stack).extra
            if extra and not random_init:
                # try to use values from extra dict
                if "raw_value" in extra and "scale" in extra:
                    raw_values.append(
                        jnp.array(extra["raw_value"], dtype=jnp.float32).reshape(shape)
                    )
                    scales.append(jnp.array(extra["scale"], dtype=jnp.float32))
                    bias_value = clamp_to_range(raw_values[-1], scales[-1])
                    if "bias_value" in extra:
                        assert jnp.allclose(
                            bias_value,
                            jnp.array(extra["bias_value"], dtype=jnp.float32).reshape(shape),
                            atol=1e-4,
                        ), "Inconsistent bias_value in extra dict"
                elif "bias_value" in extra:
                    scale = extra.get("scale", 0.0)
                    scales.append(jnp.array(scale, dtype=jnp.float32))
                    bias_v = jnp.array(extra["bias_value"], dtype=jnp.float32).reshape(shape)
                    raw_v = inverse_clamp(bias_v, scales[-1])
                    raw_values.append(raw_v)
                else:
                    # no valid values in extra, use random init for this node
                    raw_values.append(jax.random.uniform(key, shape, minval=-1, maxval=1))
                    scales.append(jnp.array(0.0, dtype=jnp.float32))
            else:
                # random init requested or no extra dict
                raw_values.append(jax.random.uniform(key, shape, minval=-1, maxval=1))
                scales.append(jnp.array(0.0, dtype=jnp.float32))

        params[f"{namespace}/raw_value"] = jnp.stack(raw_values)
        params[f"{namespace}/scale"] = jnp.stack(scales)

    def apply(*_, params: ParameterTree, node_id: ArrayLike, **__) -> tuple[ArrayLike, dict]:
        raw_bias_value = params[f"{namespace}/raw_value"][node_id]
        scale = params[f"{namespace}/scale"][node_id]
        bias_value = clamp_to_range(raw_bias_value, scale)

        return bias_value, {
            "raw_bias_value": raw_bias_value,
            "bias_value": bias_value,
            "scale": scale,
        }

    def commit(params: ParameterTree, nodelist: list[StackNode], **_):
        for i, n in enumerate(nodelist):
            updt = {
                "scale": params[f"{namespace}/scale"][i],
                "raw_value": params[f"{namespace}/raw_value"][i],
            }
            updt["bias_value"] = clamp_to_range(updt["raw_value"], updt["scale"])
            n.get(stack).extra.update(updt)

    output_shapes = [tuple(shape)]  # single output shape

    return LayerInstance(prepare, apply, output_shapes, commit=commit)
