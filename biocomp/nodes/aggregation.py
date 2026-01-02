from biocomp.compute import StackNode, ComputeStack
import jax
from jax.typing import ArrayLike
import jax.numpy as jnp
import numpy as np
from biocomp.parameters import ArrayRef, ParameterTree
from biocomp.nodeutils import (
    LayerInstance,
    add_tu_output_mapping,
    add_node_network_ids,
    NON_GRAD_TAG,
)
from biocomp.tumasking import TU_LOG_ALPHA_PATH, TU_BINARY_MASK_PATH
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
                assert len(extra["ratios"]) == n_outputs, (
                    f"Aggregation ratios length mismatch: got {len(extra['ratios'])}, expected {n_outputs}. "
                    f"This usually means the network was committed but graph structure wasn't pruned to match. "
                    f"Members: {extra.get('members', [])}"
                )

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
        add_node_network_ids(params, nodelist, namespace)

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
            from biocomp.tumasking import get_tu_masks

            tu_indices = params[output_tu_indices_path][node_id]
            output_masks = get_tu_masks(
                params, tu_indices, tu_enabled_random_vars, network_id, is_multi_tu=False
            )
        else:
            output_masks = jnp.ones(n_outputs)

        masked_ratios = abs_ratios * output_masks
        masked_sum = jnp.sum(masked_ratios)
        safe_sum = jnp.maximum(masked_sum, 1e-8)  # jnp.where evals both branches
        normalized_ratios = jnp.where(
            masked_sum > 1e-8, masked_ratios / safe_sum, jnp.zeros_like(masked_ratios)
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
        from biocomp.tumasking import get_final_mask, TU_ALWAYS_ENABLED

        output_tu_indices_path = f"{namespace}/output_tu_indices"
        has_hard_concrete = output_tu_indices_path in params and TU_LOG_ALPHA_PATH in params
        has_binary_mask = output_tu_indices_path in params and TU_BINARY_MASK_PATH in params
        has_tu_masking = has_hard_concrete or has_binary_mask

        def get_mask_for_tu(tu_idx: int, network_id: int) -> float:
            if has_binary_mask:
                binary_mask = params[TU_BINARY_MASK_PATH]
                if binary_mask.ndim == 2:
                    binary_mask = binary_mask[network_id]
                return float(binary_mask[tu_idx])
            else:
                tu_log_alpha = params[TU_LOG_ALPHA_PATH]
                if tu_log_alpha.ndim == 2:
                    tu_log_alpha = tu_log_alpha[network_id]
                return float(get_final_mask(tu_log_alpha[tu_idx : tu_idx + 1])[0])

        if has_hard_concrete:
            tu_log_alpha = params[TU_LOG_ALPHA_PATH]
            assert tu_log_alpha.ndim == 2, (
                f"COMMIT BUG: tu_log_alpha must be 2D (n_networks, n_tus), got {tu_log_alpha.ndim}D. "
                f"Shape: {tu_log_alpha.shape}. This likely means params were not sliced for (rep, target)."
            )

        for i, n in enumerate(nodelist):
            updt = {}
            ratios = params[f"{namespace}/{PNAME}"][i]
            ratios_array = jnp.abs(jnp.array(ratios))
            original_ratios = ratios_array.copy()

            n_masked = 0
            if has_tu_masking:
                tu_indices = params[output_tu_indices_path][i]
                network_id = n.network_id

                for j in range(n_outputs):
                    tu_idx = int(tu_indices[j])
                    if tu_idx != TU_ALWAYS_ENABLED:
                        mask = get_mask_for_tu(tu_idx, network_id)
                        assert mask in (0.0, 1.0), f"COMMIT BUG: mask should be binary, got {mask}"
                        if mask == 0.0:
                            n_masked += 1
                        ratios_array = ratios_array.at[j].set(ratios_array[j] * mask)

            positive_ratios = ratios_array[ratios_array > 0]
            min_ratio = jnp.min(positive_ratios) if len(positive_ratios) > 0 else 1.0
            min_ratio = jnp.maximum(min_ratio, 1e-9)
            normalized_ratios = ratios_array / min_ratio

            if has_tu_masking and n_masked > 0:
                n_zeros = sum(1 for r in normalized_ratios.tolist() if abs(r) < 1e-8)
                assert n_zeros >= n_masked, (
                    f"COMMIT BUG: {n_masked} TUs should be masked but only {n_zeros} ratios are zero. "
                    f"Original: {original_ratios.tolist()}, Final: {normalized_ratios.tolist()}"
                )

            extra = n.get(stack).extra
            ratios_list = normalized_ratios.tolist()[:n_outputs]
            updt["ratios"] = ratios_list
            updt["ratio_ranges"] = [None] * len(updt["ratios"])
            extra.update(updt)

    output_shape = input_shapes * n_outputs

    return LayerInstance(prepare, apply, output_shape, commit)


def inv_aggregation(
    input_shapes: list[tuple[int]],
    n_outputs: int,
    stack: ComputeStack,
    namespace: str,
    **_,
) -> LayerInstance:
    assert len(input_shapes) == 1, f"inv_aggregation expects 1 input, got {len(input_shapes)}"
    assert n_outputs == 1, f"inv_aggregation expects 1 output, got {n_outputs}"

    fwd_path_to_idx: dict[str, int] = {}
    fwd_paths_list: list[str] = []

    def prepare(params: ParameterTree, nodelist: list[StackNode], **_):
        nonlocal fwd_path_to_idx, fwd_paths_list
        fwd_path_to_idx = {}
        fwd_paths_list = []

        ratio_ref = ArrayRef(params.data)
        original_slots = []
        fwd_node_positions = []
        fwd_path_indices = []

        for node in nodelist:
            extra = node.get(stack).extra
            assert extra["original_output_slot"] < extra["original_output_len"]
            original_slot = extra["original_output_slot"]
            original_slots.append(original_slot)

            fwd_node = node.get_forward_stacknode(stack)
            assert fwd_node.layer_number is not None
            node_fwd_ns = stack.get_layer_namespace(fwd_node.layer_number)
            fwd_pos = fwd_node.node_position_in_layer
            fwd_path = f"{node_fwd_ns}/ratios"

            if fwd_path not in fwd_path_to_idx:
                fwd_path_to_idx[fwd_path] = len(fwd_paths_list)
                fwd_paths_list.append(fwd_path)
            fwd_path_indices.append(fwd_path_to_idx[fwd_path])

            ratio_ref.push_back(fwd_path, (fwd_pos, original_slot))
            fwd_node_positions.append(fwd_pos)

        params.at(f"{namespace}/ratios", ratio_ref, overwrite=None)
        params.at(f"{namespace}/original_slots", jnp.array(original_slots), tags=[NON_GRAD_TAG])
        params.at(
            f"{namespace}/fwd_node_positions", jnp.array(fwd_node_positions), tags=[NON_GRAD_TAG]
        )
        params.at(f"{namespace}/fwd_path_indices", jnp.array(fwd_path_indices), tags=[NON_GRAD_TAG])

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
        original_slot = params[f"{namespace}/original_slots"][node_id]
        fwd_node_pos = params[f"{namespace}/fwd_node_positions"][node_id]
        fwd_path_idx = params[f"{namespace}/fwd_path_indices"][node_id]

        ratio_ref = params.data.get_at(f"{namespace}/ratios", get_leaf_value=False).value
        original_ratio = jnp.abs(params[f"{namespace}/ratios"][node_id])

        all_masked_sums = []
        all_this_masks = []
        for path in ratio_ref.paths:
            fwd_ratios = jnp.abs(params[path][fwd_node_pos])
            fwd_ns = path.rsplit("/ratios", 1)[0]
            fwd_tu_path = f"{fwd_ns}/output_tu_indices"

            if fwd_tu_path in params and tu_enabled_random_vars is not None:
                from biocomp.tumasking import get_tu_masks

                tu_indices = params[fwd_tu_path][fwd_node_pos]
                masks = get_tu_masks(
                    params, tu_indices, tu_enabled_random_vars, network_id, is_multi_tu=False
                )
            else:
                masks = jnp.ones_like(fwd_ratios)

            all_masked_sums.append(jnp.sum(fwd_ratios * masks))
            slot_idx = jnp.minimum(original_slot, masks.shape[0] - 1)
            all_this_masks.append(masks[slot_idx])

        all_masked_sums = jnp.stack(all_masked_sums)
        all_this_masks = jnp.stack(all_this_masks)
        masked_sum = all_masked_sums[fwd_path_idx]
        this_mask = all_this_masks[fwd_path_idx]

        safe_sum = jnp.maximum(masked_sum, 1e-8)
        masked_ratio = original_ratio * this_mask
        normalized_ratio = jnp.where(masked_sum > 1e-8, masked_ratio / safe_sum, 0.0)

        is_enabled = normalized_ratio >= DISABLED_THRESHOLD
        safe_ratio = jnp.maximum(normalized_ratio, DISABLED_THRESHOLD)
        full_result = input / safe_ratio
        # STE for leaky gradient: forward uses 0 when disabled, backward uses small floor
        DISABLED_RESULT_FLOOR = 0.01
        leaky_result = full_result * DISABLED_RESULT_FLOOR
        result = jnp.where(is_enabled, full_result, leaky_result)
        # Correct to exactly 0 in forward for disabled cases (STE)
        result = result + jax.lax.stop_gradient(jnp.where(is_enabled, 0.0, -leaky_result))

        return result, {
            "original_ratio": original_ratio,
            "normalized_ratio": normalized_ratio,
            "is_enabled": is_enabled,
            "masked_sum": masked_sum,
        }

    output_shape = input_shapes
    return LayerInstance(prepare, apply, output_shape)
