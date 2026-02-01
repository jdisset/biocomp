from biocomp.compute import StackNode, ComputeStack
import jax
from jax.typing import ArrayLike
import jax.numpy as jnp
import numpy as np
from biocomp.parameters import ParameterTree
from biocomp.nodeutils import LayerInstance, NON_GRAD_TAG, add_node_network_ids
from biocomp.utils import get_logger


PRNGKey = ArrayLike
NDArray = np.ndarray | jnp.ndarray

logger = get_logger(__name__)

MIN_FLUO_INTENSITY = 0.0
MAX_FLUO_INTENSITY = 1.0

DEFAULT_BIAS_MIN = 0.0
DEFAULT_BIAS_MAX = 0.7


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

        for node, k in zip(nodelist, keys, strict=False):
            extra = node.get(stack).extra
            fluo_specs = extra.get("fluo_bias") or extra.get("fluo_bias_data")

            if not fluo_specs:
                fluo_specs = {"value": 0.5, "tu_id": 0, "protein": None, "units": "AU"}

            assert isinstance(fluo_specs, dict), (
                f"fluo_bias/fluo_bias_data should be a dict, got {type(fluo_specs)}: {repr(fluo_specs)}"
            )

            value = fluo_specs.get("value")
            if isinstance(value, dict) and "min" in value and "max" in value:
                min_v = value.get("min", MIN_FLUO_INTENSITY)
                max_v = value.get("max", MAX_FLUO_INTENSITY)
                if "init" in value:
                    init_v = jnp.full(shape, float(value["init"]), dtype=jnp.float32)
                else:
                    init_v = jax.random.uniform(k, shape, minval=min_v, maxval=max_v)
            else:
                fixed_val = float(value)
                min_v = fixed_val
                max_v = fixed_val
                init_v = jnp.full(shape, fixed_val, dtype=jnp.float32)

            raw_values.append(init_v)
            min_values.append(jnp.asarray(min_v, dtype=jnp.float32))
            max_values.append(jnp.asarray(max_v, dtype=jnp.float32))

        raw_values_arr = jnp.stack(raw_values)
        min_values_arr = jnp.stack(min_values)
        max_values_arr = jnp.stack(max_values)

        all_constrained = jnp.allclose(min_values_arr, max_values_arr)

        if all_constrained:
            params.at(f"{namespace}/raw_value", raw_values_arr, tags=[NON_GRAD_TAG])
        else:
            params[f"{namespace}/raw_value"] = raw_values_arr
        params.at(f"{namespace}/min_value", min_values_arr, tags=[NON_GRAD_TAG])
        params.at(f"{namespace}/max_value", max_values_arr, tags=[NON_GRAD_TAG])
        add_node_network_ids(params, nodelist, namespace)

    def apply(*_, params: ParameterTree, node_id: ArrayLike, **__) -> tuple[ArrayLike, dict]:
        bias_value = get_bias_value(params, node_id)
        raw_bias_value = params[f"{namespace}/raw_value"][node_id]
        return bias_value, {
            "raw_bias_value": raw_bias_value,
            "bias_value": bias_value,
        }

    def commit(params: ParameterTree, nodelist: list[StackNode], stack: ComputeStack, **_):
        for i, n in enumerate(nodelist):
            newextra = {}
            bias_value = get_bias_value(params, i)
            newextra["bias_value"] = bias_value
            n.get(stack).extra.update(newextra)

    output_shapes = [tuple(shape)]

    def introspect(
        params: ParameterTree,
        nodelist: list[StackNode],
        stack: ComputeStack,
        network_id: int,
        local_only: bool = True,
    ) -> list:
        from biocomp.paramintrospect import NodeParamInfo, ParamValue, ParamKind

        result = []
        for node_idx, node in enumerate(nodelist):
            if node.network_id != network_id:
                continue

            extra = node.get(stack).extra
            fluo_specs = extra.get("fluo_bias") or extra.get("fluo_bias_data") or {}
            protein = fluo_specs.get("protein", "unknown")
            node_name = f"bias_{protein}"

            raw_val = float(np.mean(np.asarray(params[f"{namespace}/raw_value"][node_idx])))
            min_val = float(np.mean(np.asarray(params[f"{namespace}/min_value"][node_idx])))
            max_val = float(np.mean(np.asarray(params[f"{namespace}/max_value"][node_idx])))
            bias_val = float(np.clip(raw_val, min_val, max_val))

            is_constrained = abs(min_val - max_val) < 1e-6

            ungrouped = [
                ParamValue(
                    name="bias",
                    kind=ParamKind.BIAS,
                    value=bias_val,
                    bounds=None if is_constrained else (min_val, max_val),
                )
            ]

            result.append(
                NodeParamInfo(
                    node_type="hard_bias",
                    node_name=node_name,
                    network_id=network_id,
                    ungrouped=ungrouped,
                )
            )

        return result

    return LayerInstance(prepare, apply, output_shapes, commit=commit, introspect=introspect)


def bias(
    input_shapes: list[tuple[int]],
    n_outputs: int,
    stack,
    namespace: str,
    shape: tuple[int] = (1,),
    **_,
) -> LayerInstance:
    """Learnable bias with sigmoid clamping (softer than hard_bias)"""
    assert n_outputs == 1, f"Bias node should have 1 output, got {n_outputs}"
    assert len(input_shapes) == 0

    def get_bias_value(params, node_id):
        raw_bias_value = params[f"{namespace}/raw_value"][node_id]
        min_value = params[f"{namespace}/min_value"][node_id]
        max_value = params[f"{namespace}/max_value"][node_id]
        scale = params[f"{namespace}/scale"][node_id]

        s = jax.nn.sigmoid(scale) + 0.001
        scaled_sigmoid = jax.nn.sigmoid(raw_bias_value / s)
        return scaled_sigmoid * (max_value - min_value) + min_value

    def prepare(params: ParameterTree, nodelist: list[StackNode], key, **_):
        raw_values, min_values, max_values, scales = [], [], [], []

        keys = jax.random.split(key, len(nodelist))

        for node, k in zip(nodelist, keys, strict=False):
            extra = node.get(stack).extra
            fluo_specs = extra.get("fluo_bias") or extra.get("fluo_bias_data")

            if not fluo_specs:
                fluo_specs = {"value": 0.5, "tu_id": 0, "protein": None, "units": "AU"}

            assert isinstance(fluo_specs, dict), (
                f"fluo_bias/fluo_bias_data should be a dict, got {type(fluo_specs)}: {repr(fluo_specs)}"
            )

            value = fluo_specs.get("value")
            if isinstance(value, dict) and "min" in value and "max" in value:
                min_v = value.get("min", MIN_FLUO_INTENSITY)
                max_v = value.get("max", MAX_FLUO_INTENSITY)
                if "init" in value:
                    init_output = float(value["init"])
                    if abs(max_v - min_v) < 1e-8:
                        init_v = jnp.full(shape, init_output, dtype=jnp.float32)
                    else:
                        s = 0.501  # sigmoid(0) + 0.001, matches scale init below
                        t = (init_output - min_v) / (max_v - min_v)
                        t = jnp.clip(t, 0.001, 0.999)  # avoid logit singularities
                        raw_v = s * jnp.log(t / (1 - t))  # s * logit(t)
                        init_v = jnp.full(shape, float(raw_v), dtype=jnp.float32)
                else:
                    init_v = jax.random.uniform(k, shape, minval=min_v, maxval=max_v)
            else:
                fixed_val = float(value)
                min_v = fixed_val
                max_v = fixed_val
                init_v = jnp.full(shape, fixed_val, dtype=jnp.float32)

            raw_values.append(init_v)
            min_values.append(jnp.asarray(min_v, dtype=jnp.float32))
            max_values.append(jnp.asarray(max_v, dtype=jnp.float32))
            scales.append(jnp.array(0.0, dtype=jnp.float32))

        raw_values_arr = jnp.stack(raw_values)
        scales_arr = jnp.stack(scales)
        min_values_arr = jnp.stack(min_values)
        max_values_arr = jnp.stack(max_values)

        all_constrained = jnp.allclose(min_values_arr, max_values_arr)

        if all_constrained:
            params.at(f"{namespace}/raw_value", raw_values_arr, tags=[NON_GRAD_TAG])
            params.at(f"{namespace}/scale", scales_arr, tags=[NON_GRAD_TAG])
        else:
            params[f"{namespace}/raw_value"] = raw_values_arr
            params[f"{namespace}/scale"] = scales_arr
        params.at(f"{namespace}/min_value", min_values_arr, tags=[NON_GRAD_TAG])
        params.at(f"{namespace}/max_value", max_values_arr, tags=[NON_GRAD_TAG])
        add_node_network_ids(params, nodelist, namespace)

    def apply(*_, params: ParameterTree, node_id: ArrayLike, **__) -> tuple[ArrayLike, dict]:
        bias_value = get_bias_value(params, node_id)
        raw_bias_value = params[f"{namespace}/raw_value"][node_id]
        scale = params[f"{namespace}/scale"][node_id]
        return bias_value, {
            "raw_bias_value": raw_bias_value,
            "bias_value": bias_value,
            "scale": scale,
        }

    def commit(params: ParameterTree, nodelist: list[StackNode], stack: ComputeStack, **_):
        for i, n in enumerate(nodelist):
            newextra = {}
            bias_value = get_bias_value(params, i)
            newextra["bias_value"] = bias_value
            newextra["scale"] = params[f"{namespace}/scale"][i]
            n.get(stack).extra.update(newextra)

    output_shapes = [tuple(shape)]

    def introspect(
        params: ParameterTree,
        nodelist: list[StackNode],
        stack: ComputeStack,
        network_id: int,
        local_only: bool = True,
    ) -> list:
        from biocomp.paramintrospect import NodeParamInfo, ParamValue, ParamKind

        result = []
        for node_idx, node in enumerate(nodelist):
            if node.network_id != network_id:
                continue

            extra = node.get(stack).extra
            fluo_specs = extra.get("fluo_bias") or extra.get("fluo_bias_data") or {}
            protein = fluo_specs.get("protein", "unknown")
            node_name = f"bias_{protein}"

            bias_val = float(np.mean(np.asarray(get_bias_value(params, node_idx))))
            min_val = float(np.mean(np.asarray(params[f"{namespace}/min_value"][node_idx])))
            max_val = float(np.mean(np.asarray(params[f"{namespace}/max_value"][node_idx])))

            is_constrained = abs(min_val - max_val) < 1e-6

            ungrouped = [
                ParamValue(
                    name="bias",
                    kind=ParamKind.BIAS,
                    value=bias_val,
                    bounds=None if is_constrained else (min_val, max_val),
                )
            ]

            result.append(
                NodeParamInfo(
                    node_type="bias",
                    node_name=node_name,
                    network_id=network_id,
                    ungrouped=ungrouped,
                )
            )

        return result

    return LayerInstance(prepare, apply, output_shapes, commit=commit, introspect=introspect)
