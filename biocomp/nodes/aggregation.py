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
from biocomp.utils import get_logger
from biocomp.config import BIOCOMP_CONSTANTS
from typing import Optional
from dataclasses import dataclass, asdict


PRNGKey = ArrayLike
NDArray = np.ndarray | jnp.ndarray

logger = get_logger(__name__)


@dataclass
class AggregationMember:
    ratio: float = 1.0
    ratio_range: Optional[dict] = None
    locked: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AggregationMember":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _migrate_legacy_members(extra: dict) -> dict[str, AggregationMember]:
    members_data = extra.get("members")

    if isinstance(members_data, dict):
        result = {}
        for source_id, val in members_data.items():
            if isinstance(val, AggregationMember):
                result[source_id] = val
            elif isinstance(val, dict):
                result[source_id] = AggregationMember.from_dict(val)
            else:
                raise ValueError(f"Invalid member data for {source_id}: {type(val)}")
        return result

    if isinstance(members_data, list):
        member_ids = members_data
        ratios = extra.get("ratios", [1.0] * len(member_ids))
        ratio_ranges = extra.get("ratio_ranges", [None] * len(member_ids))
        locked = extra.get("ratio_locked", [False] * len(member_ids))

        assert len(ratios) == len(member_ids), (
            f"ratios length {len(ratios)} != members {len(member_ids)}"
        )
        assert len(ratio_ranges) == len(member_ids), (
            f"ratio_ranges length {len(ratio_ranges)} != members {len(member_ids)}"
        )
        assert len(locked) == len(member_ids), (
            f"ratio_locked length {len(locked)} != members {len(member_ids)}"
        )

        return {
            mid: AggregationMember(ratio=r, ratio_range=rr, locked=lk)
            for mid, r, rr, lk in zip(member_ids, ratios, ratio_ranges, locked)
        }

    return {}


def renormalize_members_after_removal(extra: dict, removed_member_ids: set[str]) -> None:
    """Remove members by id and renormalize ratios, preserving member metadata."""
    if not removed_member_ids:
        return
    members = _migrate_legacy_members(extra)
    if not members:
        return
    for member_id in removed_member_ids:
        members.pop(member_id, None)
    if not members:
        extra["members"] = {}
        return
    total = sum(m.ratio for m in members.values())
    if total > 1e-9:
        for member in members.values():
            member.ratio = member.ratio / total
    extra["members"] = {mid: m.to_dict() for mid, m in members.items()}


def _members_to_arrays(
    members: dict[str, AggregationMember],
) -> tuple[list[str], list[float], list[Optional[dict]], list[bool]]:
    sorted_ids = sorted(members.keys())
    return (
        sorted_ids,
        [members[m].ratio for m in sorted_ids],
        [members[m].ratio_range for m in sorted_ids],
        [members[m].locked for m in sorted_ids],
    )


def extract_ratios_from_extra(extra: dict) -> tuple[list[str], list[float]]:
    """Extract member IDs and ratios from aggregation node extra.

    Expects new dict format only. Returns (member_ids, ratios) with deterministic ordering.
    """
    members_data = extra.get("members", {})
    if not isinstance(members_data, dict) or not members_data:
        return [], []
    sorted_ids = sorted(members_data.keys())
    ratios = [
        members_data[m].get("ratio", 1.0) if isinstance(members_data[m], dict) else 1.0
        for m in sorted_ids
    ]
    return sorted_ids, ratios


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

            members = _migrate_legacy_members(extra)
            if members:
                sorted_ids, base_ratios, ranges, locked = _members_to_arrays(members)
                extra["_sorted_member_ids"] = sorted_ids
            else:
                base_ratios = [1.0] * n_outputs
                ranges = [None] * n_outputs
                locked = [False] * n_outputs
                extra["_sorted_member_ids"] = []

            if len(base_ratios) != n_outputs:
                if random_init:
                    raise AssertionError(
                        f"Aggregation ratios length mismatch: got {len(base_ratios)}, expected {n_outputs}. "
                        f"Members: {list(members.keys()) if members else []}"
                    )
                logger.warning(
                    f"Aggregation n_outputs mismatch: {len(base_ratios)} members vs {n_outputs} expected. "
                    f"Adjusting to match (node {i})."
                )
                if len(base_ratios) < n_outputs:
                    base_ratios = base_ratios + [0.0] * (n_outputs - len(base_ratios))
                    ranges = ranges + [None] * (n_outputs - len(ranges))
                    locked = locked + [True] * (n_outputs - len(locked))
                else:
                    base_ratios = base_ratios[:n_outputs]
                    ranges = ranges[:n_outputs]
                    locked = locked[:n_outputs]

            has_unlocked = any(r is not None for r in ranges)

            # random_init only unlocks ratios that are NOT explicitly locked
            if random_init and not has_unlocked:
                any_unlocked = False
                new_ranges = []
                for j in range(n_outputs):
                    is_locked = locked[j] if j < len(locked) else False
                    if is_locked:
                        new_ranges.append(None)  # keep locked
                    else:
                        new_ranges.append({"min": 0.05, "max": 1.0})
                        any_unlocked = True
                ranges = new_ranges
                has_unlocked = any_unlocked

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
                node_mins_arr = jnp.array(node_mins, dtype=jnp.float32)
                node_maxs_arr = jnp.array(node_maxs, dtype=jnp.float32)
                ratio_v = absolute_ratios
                ratio_mins.append(node_mins_arr)
                ratio_maxs.append(node_maxs_arr)
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
                W1 = jax.random.normal(k1, (latent_hidden_dim, latent_dim)) * jnp.sqrt(
                    2.0 / latent_dim
                )
                b1 = jnp.zeros(latent_hidden_dim)
                W2 = (
                    jax.random.normal(k2, (n_outputs, latent_hidden_dim))
                    * jnp.sqrt(2.0 / latent_hidden_dim)
                    * 0.1
                )
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
            logger.info(
                f"Latent ratios enabled: {n_nodes} nodes × {latent_dim}d latent → {n_outputs} outputs"
            )

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
                params, tu_indices, network_id, is_multi_tu=False
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

    def commit(
        params: ParameterTree,
        nodelist: list[StackNode],
        stack: ComputeStack = None,
        lock_ratios: bool = True,
        **_,
    ):
        from biocomp.tumasking import get_final_mask, TU_ALWAYS_ENABLED
        from biocomp.tumasking_strategy import get_full_log_alpha

        output_tu_indices_path = f"{namespace}/output_tu_indices"
        log_alpha_full = get_full_log_alpha(params)
        has_tu_masking = output_tu_indices_path in params and log_alpha_full is not None

        def get_mask_for_tu(tu_idx: int, network_id: int) -> float:
            assert log_alpha_full is not None
            tu_log_alpha = log_alpha_full[network_id]
            return float(get_final_mask(tu_log_alpha[tu_idx : tu_idx + 1])[0])

        if log_alpha_full is not None:
            assert log_alpha_full.ndim == 2, (
                f"COMMIT BUG: params not sliced for (rep, target), got ndim={log_alpha_full.ndim}"
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
            ratio_sum = jnp.sum(positive_ratios) if len(positive_ratios) > 0 else 1.0
            ratio_sum = jnp.maximum(ratio_sum, 1e-9)
            normalized_ratios = ratios_array / ratio_sum

            if has_tu_masking and n_masked > 0:
                n_zeros = sum(1 for r in normalized_ratios.tolist() if abs(r) < 1e-8)
                assert n_zeros >= n_masked, (
                    f"COMMIT BUG: {n_masked} TUs should be masked but only {n_zeros} ratios are zero. "
                    f"Original: {original_ratios.tolist()}, Final: {normalized_ratios.tolist()}"
                )

            ratio_ranges = []
            ratio_locked = []
            for j in range(n_outputs):
                min_v = float(ratio_min[j])
                max_v = float(ratio_max[j])
                if lock_ratios:
                    ratio_ranges.append(None)
                    ratio_locked.append(False)
                elif abs(min_v - max_v) < 1e-8:
                    ratio_ranges.append(None)
                    ratio_locked.append(True)
                else:
                    init_v = float(normalized_ratios[j])
                    if init_v < min_v:
                        init_v = min_v
                    elif init_v > max_v:
                        init_v = max_v
                    ratio_ranges.append({"min": min_v, "max": max_v, "init": init_v})
                    ratio_locked.append(False)

            extra = n.get(stack).extra
            ratios_list = normalized_ratios.tolist()[:n_outputs]
            sorted_ids = extra.get("_sorted_member_ids", [])

            if sorted_ids:
                assert len(ratios_list) == len(sorted_ids), (
                    f"COMMIT BUG: ratios {len(ratios_list)} != members {len(sorted_ids)}"
                )
                # Filter out disabled members (ratio=0 from TU masking)
                updt["members"] = {
                    mid: AggregationMember(
                        ratio=ratios_list[j],
                        ratio_range=ratio_ranges[j],
                        locked=ratio_locked[j],
                    ).to_dict()
                    for j, mid in enumerate(sorted_ids)
                    if abs(ratios_list[j]) > 1e-8
                }
            else:
                updt["ratios"] = ratios_list
                updt["ratio_ranges"] = ratio_ranges
                updt["ratio_locked"] = ratio_locked

            extra.update(updt)

    output_shape = input_shapes * n_outputs

    def introspect(
        params: ParameterTree,
        nodelist: list[StackNode],
        stack: ComputeStack,
        network_id: int,
        local_only: bool = True,
    ) -> list:
        from biocomp.paramintrospect import (
            NodeParamInfo,
            TUParamGroup,
            ParamValue,
            ParamKind,
            get_tu_prob,
            is_tu_enabled,
        )
        from biocomp.tumasking import TU_ALWAYS_ENABLED

        result = []
        for i, node in enumerate(nodelist):
            if node.network_id != network_id:
                continue

            comp_node = node.get(stack)
            extra = comp_node.extra
            sorted_ids = extra.get("_sorted_member_ids", [])
            node_name = extra.get("name", f"agg_{i}")

            source_to_tu_name: dict[str, str] = {}
            graph = stack.networks[network_id].compute_graph
            for src in graph.nodes.values():
                if src.node_type == "source":
                    src_id = src.extra.get("source_id") or src.extra.get("name", "")
                    tu_name = src.extra.get("name", src_id)
                    if src_id and tu_name and src_id != tu_name:
                        source_to_tu_name[src_id] = tu_name

            latent_z_path = f"{namespace}/latent_z"
            if latent_z_path in params:
                z = np.asarray(params[latent_z_path][i])
                W1 = np.asarray(params[f"{namespace}/latent_W1"][i])
                b1 = np.asarray(params[f"{namespace}/latent_b1"][i])
                W2 = np.asarray(params[f"{namespace}/latent_W2"][i])
                b2 = np.asarray(params[f"{namespace}/latent_b2"][i])
                raw_ratios = _decode_latent_ratios(z, W1, b1, W2, b2)[:n_outputs]
                ratio_min = np.asarray(params[f"{namespace}/ratio_min"][i][:n_outputs])
                ratio_max = np.asarray(params[f"{namespace}/ratio_max"][i][:n_outputs])
                ratios = np.clip(raw_ratios, ratio_min, ratio_max)
            else:
                ratios = np.asarray(params[f"{namespace}/{PNAME}"][i][:n_outputs])
                ratio_min = np.asarray(params[f"{namespace}/ratio_min"][i][:n_outputs])
                ratio_max = np.asarray(params[f"{namespace}/ratio_max"][i][:n_outputs])
                ratios = np.clip(ratios, ratio_min, ratio_max)

            output_tu_path = f"{namespace}/output_tu_indices"
            tu_indices = None
            if output_tu_path in params:
                tu_indices = np.asarray(params[output_tu_path][i])

            tu_groups = []
            for j, member_id in enumerate(sorted_ids):
                if j >= len(ratios):
                    break

                ratio_val = float(np.asarray(ratios[j]).item())
                r_min = float(np.asarray(ratio_min[j]).item())
                r_max = float(np.asarray(ratio_max[j]).item())
                is_constrained = abs(r_min - r_max) < 1e-6

                prob = 1.0
                tu_idx = TU_ALWAYS_ENABLED
                if tu_indices is not None and j < len(tu_indices):
                    tu_idx = int(np.asarray(tu_indices[j]).item())
                    if tu_idx >= 0:
                        prob = get_tu_prob(params, network_id, tu_idx)

                pv = ParamValue(
                    name="ratio",
                    kind=ParamKind.RATIO,
                    value=ratio_val,
                    bounds=None if is_constrained else (r_min, r_max),
                )

                display_name = source_to_tu_name.get(member_id, member_id)
                tu_groups.append(
                    TUParamGroup(
                        tu_id=display_name,
                        is_enabled=is_tu_enabled(prob),
                        prob=prob,
                        params=[pv],
                        inputs=[],
                    )
                )

            result.append(
                NodeParamInfo(
                    node_type="aggregation",
                    node_name=node_name,
                    network_id=network_id,
                    tu_groups=tu_groups,
                )
            )

        return result

    return LayerInstance(prepare, apply, output_shape, commit, introspect)


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

    DISABLED_THRESHOLD = BIOCOMP_CONSTANTS["ratio"]["prune_threshold"]

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

        all_masked_sums, all_this_masks, all_this_ratios = [], [], []
        for path in ratio_ref.paths:
            fwd_ns = path.rsplit("/ratios", 1)[0]
            latent_path = f"{fwd_ns}/latent_z"
            if latent_path in params:
                z = params[latent_path][fwd_node_pos]
                W1, b1 = (
                    params[f"{fwd_ns}/latent_W1"][fwd_node_pos],
                    params[f"{fwd_ns}/latent_b1"][fwd_node_pos],
                )
                W2, b2 = (
                    params[f"{fwd_ns}/latent_W2"][fwd_node_pos],
                    params[f"{fwd_ns}/latent_b2"][fwd_node_pos],
                )
                raw = _decode_latent_ratios(z, W1, b1, W2, b2)
                fwd_ratios = jnp.abs(
                    jnp.clip(
                        raw,
                        params[f"{fwd_ns}/ratio_min"][fwd_node_pos],
                        params[f"{fwd_ns}/ratio_max"][fwd_node_pos],
                    )
                )
            else:
                fwd_ratios = jnp.abs(params[path][fwd_node_pos])

            slot_idx = jnp.minimum(original_slot, fwd_ratios.shape[0] - 1)
            all_this_ratios.append(fwd_ratios[slot_idx])

            tu_path = f"{fwd_ns}/output_tu_indices"
            if tu_path in params:
                from biocomp.tumasking import get_tu_masks

                masks = get_tu_masks(
                    params,
                    params[tu_path][fwd_node_pos],
                    network_id,
                    is_multi_tu=False,
                )
            else:
                masks = jnp.ones_like(fwd_ratios)

            all_masked_sums.append(jnp.sum(fwd_ratios * masks))
            all_this_masks.append(masks[slot_idx])

        masked_sum = jnp.stack(all_masked_sums)[fwd_path_idx]
        this_mask = jnp.stack(all_this_masks)[fwd_path_idx]
        original_ratio = jnp.stack(all_this_ratios)[fwd_path_idx]

        safe_sum = jnp.maximum(masked_sum, 1e-8)
        masked_ratio = original_ratio * this_mask
        normalized_ratio = jnp.where(masked_sum > 1e-8, masked_ratio / safe_sum, 0.0)

        is_enabled = normalized_ratio >= DISABLED_THRESHOLD
        safe_ratio = jnp.maximum(normalized_ratio, DISABLED_THRESHOLD)
        full_result = input / safe_ratio
        leaky_result = full_result * 0.01
        result = jnp.where(is_enabled, full_result, leaky_result)
        result = result + jax.lax.stop_gradient(jnp.where(is_enabled, 0.0, -leaky_result))

        return result, {
            "original_ratio": original_ratio,
            "normalized_ratio": normalized_ratio,
            "is_enabled": is_enabled,
            "masked_sum": masked_sum,
        }

    output_shape = input_shapes
    return LayerInstance(prepare, apply, output_shape)
