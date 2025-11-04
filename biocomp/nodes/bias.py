from biocomp.compute import StackNode
import jax
from jax.typing import ArrayLike
import jax.numpy as jnp
import numpy as np
from biocomp.parameters import ParameterTree
from biocomp.nodeutils import LayerInstance
from biocomp.utils import get_logger


PRNGKey = ArrayLike
NDArray = np.ndarray | jnp.ndarray

logger = get_logger(__name__)

MIN_FLUO_INTENSITY = 0.0
MAX_FLUO_INTENSITY = 1.0


def hard_bias(
    input_shapes: list[tuple[int]],
    n_outputs: int,
    stack,
    namespace: str,
    shape: tuple[int] = (1,),
    **_,
) -> LayerInstance:
    assert n_outputs == 1, f"Bias node should have 1 output, got {n_outputs}"
    assert len(input_shapes) == 0

    def get_bias_value(params, node_id):
        raw_bias_value = params[f"{namespace}/raw_value"][node_id]
        min_value = params[f"{namespace}/min_value"][node_id]
        max_value = params[f"{namespace}/max_value"][node_id]
        return jnp.clip(raw_bias_value, min_value, max_value)

    def prepare(params: ParameterTree, nodelist: list[StackNode], key, **_):
        raw_values, min_values, max_values = [], [], []

        keys = jax.random.split(key, len(nodelist))

        for node, k in zip(nodelist, keys):
            extra = node.get(stack).extra
            fluo_specs = extra.get("fluo_bias_data")

            if not fluo_specs:
                fluo_specs = {"value": 0.5, "tu_id": 0, "protein": None, "units": "AU"}

            assert isinstance(fluo_specs, dict), (
                f"fluo_bias_data should be a dict, got {type(fluo_specs)}: {repr(fluo_specs)}"
            )

            value = fluo_specs.get("value")
            if isinstance(value, dict) and "min" in value and "max" in value:
                min_v = value.get("min", MIN_FLUO_INTENSITY)
                max_v = value.get("max", MAX_FLUO_INTENSITY)
            else:
                min_v = max_v = float(value)

            raw_v = jax.random.uniform(k, shape, minval=min_v, maxval=max_v)
            raw_values.append(raw_v)
            min_values.append(jnp.asarray(min_v, dtype=jnp.float32))
            max_values.append(jnp.asarray(max_v, dtype=jnp.float32))

        params[f"{namespace}/raw_value"] = jnp.stack(raw_values)
        params[f"{namespace}/min_value"] = jnp.stack(min_values)
        params[f"{namespace}/max_value"] = jnp.stack(max_values)

    def apply(*_, params: ParameterTree, node_id: ArrayLike, **__) -> tuple[ArrayLike, dict]:
        bias_value = get_bias_value(params, node_id)
        raw_bias_value = params[f"{namespace}/raw_value"][node_id]
        return bias_value, {
            "raw_bias_value": raw_bias_value,
            "bias_value": bias_value,
        }

    def commit(params: ParameterTree, nodelist: list[StackNode], **_):
        for i, n in enumerate(nodelist):
            newextra = {}
            bias_value = get_bias_value(params, i)
            newextra["bias_value"] = bias_value
            n.get(stack).extra.update(newextra)

    output_shapes = [tuple(shape)]

    return LayerInstance(prepare, apply, output_shapes, commit=commit)


def bias(
    input_shapes: list[tuple[int]],
    n_outputs: int,
    stack,
    namespace: str,
    shape: tuple[int] = (1,),
    valid_range: tuple[float, float] = (0.0, 1.0),
    **_,
) -> LayerInstance:
    """Learnable bias with sigmoid clamping (softer than hard_bias)"""
    assert n_outputs == 1, f"Bias node should have 1 output, got {n_outputs}"
    assert len(input_shapes) == 0

    def clamp_to_range(value: ArrayLike, scale: ArrayLike):
        s = jax.nn.sigmoid(scale) + 0.001
        scaled_sigmoid = jax.nn.sigmoid(value / s)
        return scaled_sigmoid * (valid_range[1] - valid_range[0]) + valid_range[0]

    def inverse_clamp(value: ArrayLike, scale: ArrayLike):
        s = jax.nn.sigmoid(scale) + 0.001
        y = (value - valid_range[0]) / (valid_range[1] - valid_range[0])
        y = jnp.clip(y, 1e-5, 1 - 1e-5)
        return -s * jnp.log((1 / y) - 1)

    def prepare(params: ParameterTree, nodelist: list[StackNode], key, **_):
        raw_values, scales = [], []

        for node in nodelist:
            extra = node.get(stack).extra
            if extra and "raw_value" in extra and "scale" in extra:
                raw_values.append(jnp.array(extra["raw_value"], dtype=jnp.float32).reshape(shape))
                scales.append(jnp.array(extra["scale"], dtype=jnp.float32))
            elif extra and "bias_value" in extra:
                scale = extra.get("scale", 0.0)
                scales.append(jnp.array(scale, dtype=jnp.float32))
                bias_v = jnp.array(extra["bias_value"], dtype=jnp.float32).reshape(shape)
                raw_values.append(inverse_clamp(bias_v, scales[-1]))
            else:
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

    output_shapes = [tuple(shape)]

    return LayerInstance(prepare, apply, output_shapes, commit=commit)
