from biocomp.jaxutils import flat_concat
from biocomp.compute import StackNode, ComputeStack
import jax
from jax.tree_util import Partial as partial
from jax.typing import ArrayLike
from typing import Optional
import jax.numpy as jnp
from jax import vmap
import numpy as np
from biocomp.parameters import ArrayRef, ParameterTree, init_if_needed, make_view, get_param
from biocomp.nodeutils import (
    LayerInstance,
    add_random_var_ids,
    NON_GRAD_TAG,
    get_prev_num_random_vars,
    reference_forward_random_var_ids,
    empty_prepare,
)
from biocomp.utils import get_logger
from biocomp.neuralutils import (
    ACTIVATION_FUNCTIONS,
    INITIALIZERS,
    DEFAULT_ACTIVATION,
    DEFAULT_OUT_ACTIVATION,
    DEFAULT_INITIALIZER,
    dense_mlp,
)
import biocomp.quantization as qz


PRNGKey = ArrayLike
NDArray = np.ndarray | jnp.ndarray

logger = get_logger(__name__)


def aggregation(
    input_shapes: list[tuple[int]],
    n_outputs: int,
    stack: ComputeStack,
    namespace: str,
    random_init: bool = False,
    **_,
) -> LayerInstance:
    assert len(input_shapes) == 1, f"Aggregation expects 1 input, got {len(input_shapes)}"
    pname = "ratios"

    def prepare(params: ParameterTree, nodelist: list[StackNode], key: PRNGKey, **_):
        ratios = []
        for node in nodelist:
            extra = node.get(stack).extra
            if "ratios" in extra and not random_init:
                assert len(extra["ratios"]) == n_outputs
                ratio_v = jnp.array(extra["ratios"], dtype=jnp.float32)
            else:
                ratio_v = jax.random.uniform(key, (n_outputs,), minval=0.05, maxval=1.0)
            ratios.append(ratio_v)

        ratios = jnp.stack(ratios)
        assert ratios.shape == (len(nodelist), n_outputs), f"Invalid ratio shape {ratios.shape}"
        params[f"{namespace}/{pname}"] = ratios

    def apply(
        input: NDArray,
        random_vars: NDArray,
        params: ParameterTree,
        node_id: ArrayLike,
        key: PRNGKey,
    ) -> tuple[ArrayLike, dict]:
        assert input.shape == input_shapes[0], f"Invalid input shape {input.shape}"
        ratios = params[f"{namespace}/{pname}"][node_id][:n_outputs]
        abs_ratios = jnp.abs(jnp.array(ratios))
        result = abs_ratios * input
        return result, {"ratios": ratios, "abs_ratios": abs_ratios, "n_outputs": n_outputs}

    def commit(params: ParameterTree, nodelist: list[StackNode], **_):
        for i, n in enumerate(nodelist):
            updt = {}
            ratios = params[f"{namespace}/{pname}"][i]

            # normalize absolute ratios so that the minimum is 1
            ratios_array = jnp.abs(jnp.array(ratios))  # use absolute values like in apply
            # find the minimum non-zero ratio
            positive_ratios = ratios_array[ratios_array > 0]
            min_ratio = jnp.min(positive_ratios) if len(positive_ratios) > 0 else 1.0
            min_ratio = jnp.maximum(min_ratio, 1e-9)  # avoid division by zero
            normalized_ratios = ratios_array / min_ratio

            # update extra dict
            updt["ratios"] = normalized_ratios.tolist()[:n_outputs]
            n.get(stack).extra.update(updt)

    output_shape = input_shapes * n_outputs

    return LayerInstance(prepare, apply, output_shape, commit)


def inv_aggregation(
    input_shapes: list[tuple[int]],
    n_outputs: int,
    stack: ComputeStack,
    namespace: str,
    **_,
) -> LayerInstance:
    # an inverse aggregation node always has 1 input and 1 output
    assert len(input_shapes) == 1, f"inverse_Aggregation expects 1 input, got {len(input_shapes)}"
    assert n_outputs == 1, f"inverse_Aggregation expects 1 output, got {n_outputs}"

    def prepare(params: ParameterTree, nodelist: list[StackNode], **_):
        ref = ArrayRef(params.data)
        for node in nodelist:
            extra = node.get(stack).extra
            assert extra["original_output_slot"] < extra["original_output_len"]
            original_slot = extra["original_output_slot"]

            fwd_node = node.get_forward_stacknode(stack)
            assert fwd_node.layer_number is not None
            fwd_namespace = stack.get_layer_namespace(fwd_node.layer_number)
            ref.push_back(
                f"{fwd_namespace}/ratios", (fwd_node.node_position_in_layer, original_slot)
            )

        params.at(f"{namespace}/ratios", ref, overwrite=None)

    EPSILON = 1e-9

    def apply(
        input: NDArray,
        random_vars: NDArray,
        params: ParameterTree,
        node_id: ArrayLike,
        key,
    ) -> tuple[ArrayLike, dict]:
        ratio = jnp.abs(params[f"{namespace}/ratios"][node_id])
        clamped_ratio = jnp.maximum(ratio, EPSILON)
        result = input / clamped_ratio

        return result, {"ratio": ratio, "clamped_ratio": clamped_ratio, "epsilon": EPSILON}

    output_shape = input_shapes
    return LayerInstance(prepare, apply, output_shape)
