from biocomp.compute import StackNode, ComputeStack
import jax
from jax.typing import ArrayLike
import jax.numpy as jnp
import numpy as np
from biocomp.parameters import ArrayRef, ParameterTree
from biocomp.nodeutils import LayerInstance, add_tu_output_mapping, NON_GRAD_TAG
from biocomp.tumasking import TU_LOG_ALPHA_PATH
from biocomp.utils import get_logger
from typing import Optional


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
    PNAME = "ratios"

    def prepare(params: ParameterTree, nodelist: list[StackNode], key: PRNGKey, **_):
        ratios = []
        ratio_ranges_list = []  # Store range info for each node

        for i, node in enumerate(nodelist):
            extra = node.get(stack).extra
            if "ratios" in extra and not random_init:
                assert len(extra["ratios"]) == n_outputs

                # Check if this node has unlocked ratios (ratio_ranges with non-None values)
                ranges = extra.get("ratio_ranges", [])
                has_unlocked = any(r is not None for r in ranges)

                if ranges and has_unlocked:
                    ratio_ranges_list.append(ranges)

                    # find min of unlocked ranges for locked ratios' default
                    locked_default = 1.0
                    for range_info in ranges:
                        if range_info is not None:
                            locked_default = range_info.get("min", 1.0) or 1.0
                            break  # use first unlocked range's min

                    # sample in absolute space, then normalize
                    absolute_ratios = []
                    for j, range_info in enumerate(ranges):
                        if range_info is not None:
                            min_v = range_info.get("min", 1.0) or 1.0
                            max_v = range_info.get("max", 1.0) or 1.0
                            # sample absolute ratio from range
                            absolute_ratios.append(
                                jax.random.uniform(
                                    jax.random.fold_in(key, i * n_outputs + j),
                                    minval=min_v,
                                    maxval=max_v,
                                )
                            )
                        else:
                            # locked: use min of unlocked range
                            absolute_ratios.append(locked_default)

                    # normalize to sum to 1
                    absolute_ratios = jnp.array(absolute_ratios, dtype=jnp.float32)
                    ratio_v = absolute_ratios / jnp.sum(absolute_ratios)
                else:
                    ratio_v = jnp.array(extra["ratios"], dtype=jnp.float32)
                    ratio_ranges_list.append([None] * n_outputs)  # All locked
            else:
                # Random init
                ratio_v = jax.random.uniform(key, (n_outputs,), minval=0.05, maxval=1.0)
                ratio_ranges_list.append([None] * n_outputs)

            ratios.append(ratio_v)

        ratios = jnp.stack(ratios)
        assert ratios.shape == (len(nodelist), n_outputs), f"Invalid ratio shape {ratios.shape}"
        params[f"{namespace}/{PNAME}"] = ratios

        add_tu_output_mapping(params, stack, nodelist, namespace, n_outputs)

    def apply(
        input: NDArray,
        random_vars: NDArray,
        params: ParameterTree,
        node_id: ArrayLike,
        key: PRNGKey,
        tu_enabled_random_vars: Optional[ArrayLike] = None,
        network_id: Optional[ArrayLike] = None,
        **_kwargs,
    ) -> tuple[ArrayLike, dict]:
        assert input.shape == input_shapes[0], f"Invalid input shape {input.shape}"
        ratios = params[f"{namespace}/{PNAME}"][node_id][:n_outputs]
        abs_ratios = jnp.abs(jnp.array(ratios))

        output_tu_indices_path = f"{namespace}/output_tu_indices"
        if output_tu_indices_path in params:
            from biocomp.tumasking import compute_input_masks

            tu_indices = params[output_tu_indices_path][node_id]
            tu_log_alpha_full = params[TU_LOG_ALPHA_PATH] if TU_LOG_ALPHA_PATH in params else None
            tu_log_alpha = None
            if tu_log_alpha_full is not None:
                if tu_log_alpha_full.ndim > 1 and network_id is not None:
                    tu_log_alpha = tu_log_alpha_full[network_id]
                else:
                    tu_log_alpha = tu_log_alpha_full
            output_masks = compute_input_masks(tu_indices, tu_enabled_random_vars, tu_log_alpha)
        else:
            output_masks = jnp.ones(n_outputs)

        masked_ratios = abs_ratios * output_masks
        masked_sum = jnp.sum(masked_ratios)
        normalized_ratios = jnp.where(
            masked_sum > 1e-8,
            masked_ratios / masked_sum,
            jnp.zeros_like(masked_ratios)
        )
        result = normalized_ratios * input

        return result, {
            "ratios": ratios,
            "abs_ratios": abs_ratios,
            "n_outputs": n_outputs,
            "output_masks": output_masks,
            "masked_ratios": masked_ratios,
            "normalized_ratios": normalized_ratios,
        }

    def commit(params: ParameterTree, nodelist: list[StackNode], stack: ComputeStack = None, **_):
        for i, n in enumerate(nodelist):
            updt = {}
            ratios = params[f"{namespace}/{PNAME}"][i]

            ratios_array = jnp.abs(jnp.array(ratios))
            positive_ratios = ratios_array[ratios_array > 0]
            min_ratio = jnp.min(positive_ratios) if len(positive_ratios) > 0 else 1.0
            min_ratio = jnp.maximum(min_ratio, 1e-9)
            normalized_ratios = ratios_array / min_ratio

            updt["ratios"] = normalized_ratios.tolist()[:n_outputs]

            # After commit, ratios are locked - remove ratio_ranges
            updt["ratio_ranges"] = [None] * n_outputs

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
    assert len(input_shapes) == 1, f"inverse_Aggregation expects 1 input, got {len(input_shapes)}"
    assert n_outputs == 1, f"inverse_Aggregation expects 1 output, got {n_outputs}"

    def prepare(params: ParameterTree, nodelist: list[StackNode], **_):
        ratio_ref = ArrayRef(params.data)
        original_slots = []
        fwd_node_positions = []

        for node in nodelist:
            extra = node.get(stack).extra
            assert extra["original_output_slot"] < extra["original_output_len"]
            original_slot = extra["original_output_slot"]
            original_slots.append(original_slot)

            fwd_node = node.get_forward_stacknode(stack)
            assert fwd_node.layer_number is not None
            node_fwd_ns = stack.get_layer_namespace(fwd_node.layer_number)
            fwd_pos = fwd_node.node_position_in_layer

            ratio_ref.push_back(f"{node_fwd_ns}/ratios", (fwd_pos, original_slot))
            fwd_node_positions.append(fwd_pos)

        params.at(f"{namespace}/ratios", ratio_ref, overwrite=None)
        params.at(f"{namespace}/original_slots", jnp.array(original_slots), tags=[NON_GRAD_TAG])
        params.at(f"{namespace}/fwd_node_positions", jnp.array(fwd_node_positions), tags=[NON_GRAD_TAG])

    DISABLED_THRESHOLD = 1.0 / 120.0

    def apply(
        input: NDArray,
        random_vars: NDArray,
        params: ParameterTree,
        node_id: ArrayLike,
        key,
        tu_enabled_random_vars: Optional[ArrayLike] = None,
        network_id: Optional[ArrayLike] = None,
        **_kwargs,
    ) -> tuple[ArrayLike, dict]:
        original_ratio = jnp.abs(params[f"{namespace}/ratios"][node_id])
        original_slot = params[f"{namespace}/original_slots"][node_id]
        fwd_node_pos = params[f"{namespace}/fwd_node_positions"][node_id]

        ratio_ref = params.data.get_at(f"{namespace}/ratios", get_leaf_value=False).value
        fwd_ratios_path = ratio_ref.paths[0]
        fwd_ns = fwd_ratios_path.rsplit("/ratios", 1)[0]
        all_fwd_ratios = jnp.abs(params[fwd_ratios_path][fwd_node_pos])

        fwd_tu_path = f"{fwd_ns}/output_tu_indices"
        if fwd_tu_path in params and tu_enabled_random_vars is not None:
            from biocomp.tumasking import compute_input_masks
            tu_indices = params[fwd_tu_path][fwd_node_pos]
            tu_log_alpha_full = params[TU_LOG_ALPHA_PATH] if TU_LOG_ALPHA_PATH in params else None
            tu_log_alpha = None
            if tu_log_alpha_full is not None:
                if tu_log_alpha_full.ndim > 1 and network_id is not None:
                    tu_log_alpha = tu_log_alpha_full[network_id]
                else:
                    tu_log_alpha = tu_log_alpha_full
            all_masks = compute_input_masks(tu_indices, tu_enabled_random_vars, tu_log_alpha)
        else:
            all_masks = jnp.ones_like(all_fwd_ratios)

        masked_ratios = all_fwd_ratios * all_masks
        masked_sum = jnp.sum(masked_ratios)
        this_mask = all_masks[original_slot]
        normalized_ratio = jnp.where(
            masked_sum > 1e-8,
            original_ratio * this_mask / masked_sum,
            0.0
        )

        is_enabled = normalized_ratio >= DISABLED_THRESHOLD
        safe_ratio = jnp.maximum(normalized_ratio, DISABLED_THRESHOLD)
        result = jnp.where(is_enabled, input / safe_ratio, 0.0)

        return result, {
            "original_ratio": original_ratio,
            "normalized_ratio": normalized_ratio,
            "is_enabled": is_enabled,
            "masked_sum": masked_sum,
        }

    output_shape = input_shapes
    return LayerInstance(prepare, apply, output_shape)
