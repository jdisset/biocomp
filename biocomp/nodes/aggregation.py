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


def _decode_latent_ratios(z, W1, b1, W2, b2):
    """Decode latent code to ratios via MLP. No softmax - aggregation handles normalization."""
    h = jax.nn.gelu(W1 @ z + b1)
    return W2 @ h + b2


def aggregation(
    input_shapes: list[tuple[int]],
    n_outputs: int,
    stack: ComputeStack,
    namespace: str,
    random_init: bool = False,
    use_latent_ratios: bool = False,
    latent_dim: int = 16,
    latent_hidden_dim: int = 32,
    **_,
) -> LayerInstance:
    assert len(input_shapes) == 1, f"Aggregation expects 1 input, got {len(input_shapes)}"
    PNAME = "ratios"

    def prepare(params: ParameterTree, nodelist: list[StackNode], key: PRNGKey, **_):
        ratios = []
        ratio_mins = []
        ratio_maxs = []

        for i, node in enumerate(nodelist):
            extra = node.get(stack).extra
            base_ratios = extra.get("ratios", [1.0] * n_outputs)
            ranges = extra.get("ratio_ranges", [None] * n_outputs)

            if len(base_ratios) != n_outputs:
                assert not random_init, (
                    f"Aggregation ratios length mismatch: got {len(base_ratios)}, expected {n_outputs}. "
                    f"This usually means the network was committed but graph structure wasn't pruned to match. "
                    f"Members: {extra.get('members', [])}"
                )
                base_ratios = [1.0] * n_outputs
                ranges = [None] * n_outputs

            has_unlocked = any(r is not None for r in ranges)

            if random_init and not has_unlocked:
                ranges = [{"min": 0.05, "max": 1.0} for _ in range(n_outputs)]
                has_unlocked = True

            if has_unlocked:
                absolute_ratios = []
                node_mins = []
                node_maxs = []
                for j, range_info in enumerate(ranges):
                    base_val = base_ratios[j] if j < len(base_ratios) else 1.0
                    if range_info is not None:
                        min_v = range_info.get("min", 0.05) or 0.05
                        max_v = range_info.get("max", 1.0) or 1.0
                        init_v = range_info.get("init")
                        if init_v is not None:
                            absolute_ratios.append(float(init_v))
                        else:
                            absolute_ratios.append(
                                jax.random.uniform(
                                    jax.random.fold_in(key, i * n_outputs + j),
                                    minval=min_v,
                                    maxval=max_v,
                                )
                            )
                        node_mins.append(min_v)
                        node_maxs.append(max_v)
                    else:
                        absolute_ratios.append(base_val)
                        node_mins.append(base_val)
                        node_maxs.append(base_val)

                absolute_ratios = jnp.array(absolute_ratios, dtype=jnp.float32)
                ratio_v = absolute_ratios / jnp.sum(absolute_ratios)
                ratio_mins.append(jnp.array(node_mins, dtype=jnp.float32))
                ratio_maxs.append(jnp.array(node_maxs, dtype=jnp.float32))
            else:
                ratio_v = jnp.array(base_ratios, dtype=jnp.float32)
                ratio_mins.append(ratio_v.copy())
                ratio_maxs.append(ratio_v.copy())

            ratios.append(ratio_v)

        ratios = jnp.stack(ratios)
        ratio_mins_arr = jnp.stack(ratio_mins)
        ratio_maxs_arr = jnp.stack(ratio_maxs)
        assert ratios.shape == (len(nodelist), n_outputs), f"Invalid ratio shape {ratios.shape}"

        all_constrained = jnp.allclose(ratio_mins_arr, ratio_maxs_arr)

        if all_constrained:
            params.at(f"{namespace}/{PNAME}", ratios, tags=[NON_GRAD_TAG])
        else:
            params[f"{namespace}/{PNAME}"] = ratios

        params.at(f"{namespace}/ratio_min", ratio_mins_arr, tags=[NON_GRAD_TAG])
        params.at(f"{namespace}/ratio_max", ratio_maxs_arr, tags=[NON_GRAD_TAG])

        if use_latent_ratios and not all_constrained:
            n_nodes = len(nodelist)
            latent_z_list = []
            latent_W1_list = []
            latent_b1_list = []
            latent_W2_list = []
            latent_b2_list = []

            for i in range(n_nodes):
                k1, k2, key = jax.random.split(key, 3)
                init_ratios = ratios[i]
                target_ratios = jnp.abs(init_ratios) + 0.1
                target_ratios = target_ratios / jnp.sum(target_ratios)

                z = jax.random.normal(jax.random.fold_in(key, i), (latent_dim,)) * 0.1
                W1 = jax.random.normal(k1, (latent_hidden_dim, latent_dim)) * jnp.sqrt(2.0 / latent_dim)
                b1 = jnp.zeros(latent_hidden_dim)
                W2 = jax.random.normal(k2, (n_outputs, latent_hidden_dim)) * jnp.sqrt(2.0 / latent_hidden_dim) * 0.1
                b2 = target_ratios

                latent_z_list.append(z)
                latent_W1_list.append(W1)
                latent_b1_list.append(b1)
                latent_W2_list.append(W2)
                latent_b2_list.append(b2)

            params[f"{namespace}/latent_z"] = jnp.stack(latent_z_list)
            params[f"{namespace}/latent_W1"] = jnp.stack(latent_W1_list)
            params[f"{namespace}/latent_b1"] = jnp.stack(latent_b1_list)
            params[f"{namespace}/latent_W2"] = jnp.stack(latent_W2_list)
            params[f"{namespace}/latent_b2"] = jnp.stack(latent_b2_list)
            logger.info(f"Latent ratios enabled: {n_nodes} nodes × {latent_dim}d latent → {n_outputs} outputs")

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

        latent_z_path = f"{namespace}/latent_z"
        if latent_z_path in params:
            z = params[latent_z_path][node_id]
            W1 = params[f"{namespace}/latent_W1"][node_id]
            b1 = params[f"{namespace}/latent_b1"][node_id]
            W2 = params[f"{namespace}/latent_W2"][node_id]
            b2 = params[f"{namespace}/latent_b2"][node_id]
            raw_ratios = _decode_latent_ratios(z, W1, b1, W2, b2)[:n_outputs]
            ratio_min = params[f"{namespace}/ratio_min"][node_id][:n_outputs]
            ratio_max = params[f"{namespace}/ratio_max"][node_id][:n_outputs]
            constrained_ratios = jnp.clip(raw_ratios, ratio_min, ratio_max)
        else:
            ratios = params[f"{namespace}/{PNAME}"][node_id][:n_outputs]
            ratio_min = params[f"{namespace}/ratio_min"][node_id][:n_outputs]
            ratio_max = params[f"{namespace}/ratio_max"][node_id][:n_outputs]
            constrained_ratios = jnp.clip(ratios, ratio_min, ratio_max)

        abs_ratios = jnp.abs(constrained_ratios)

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
            "ratios": constrained_ratios,
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
            latent_z_path = f"{namespace}/latent_z"
            if latent_z_path in params:
                z = params[latent_z_path][i]
                W1 = params[f"{namespace}/latent_W1"][i]
                b1 = params[f"{namespace}/latent_b1"][i]
                W2 = params[f"{namespace}/latent_W2"][i]
                b2 = params[f"{namespace}/latent_b2"][i]
                raw_ratios = _decode_latent_ratios(z, W1, b1, W2, b2)[:n_outputs]
                ratio_min = params[f"{namespace}/ratio_min"][i][:n_outputs]
                ratio_max = params[f"{namespace}/ratio_max"][i][:n_outputs]
                ratios = jnp.clip(raw_ratios, ratio_min, ratio_max)
            else:
                ratios = params[f"{namespace}/{PNAME}"][i][:n_outputs]
                ratio_min = params[f"{namespace}/ratio_min"][i][:n_outputs]
                ratio_max = params[f"{namespace}/ratio_max"][i][:n_outputs]
                ratios = jnp.clip(ratios, ratio_min, ratio_max)
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
