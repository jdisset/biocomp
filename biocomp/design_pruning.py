from __future__ import annotations

from typing import Callable, TYPE_CHECKING
from copy import deepcopy

import numpy as np
import jax
import jax.numpy as jnp

from .logging_config import get_logger
from .tumasking import TU_LOG_ALPHA_PATH
from .tumasking_strategy import build_strategy_from_config
from .tracing import trace_scope, trace_here

if TYPE_CHECKING:
    from .design import DesignManager, DesignConfig
    from .compute import ComputeStack
    from .parameters import ParameterTree
    from biocomptools.modelmodel import BiocompModel

logger = get_logger(__name__)

def _expand_params_for_merge(params: "ParameterTree") -> "ParameterTree":
    return jax.tree.map(
        lambda x: x.reshape((1, 1) + x.shape)
        if hasattr(x, "ndim") and x.ndim >= 0
        else x,
        params,
    )


def _get_tag_names(params: "ParameterTree", path_str: str) -> list[str] | None:
    if params.tags is None:
        return None
    try:
        tag_flags = params.tags[path_str]
    except KeyError:
        tag_flags = None
    if tag_flags is None:
        return None
    return [name for name, flag in zip(params.tagnames, tag_flags, strict=False) if flag]


def _set_param_value(params: "ParameterTree", path_str: str, value) -> None:
    tag_names = _get_tag_names(params, path_str)
    params.at(path_str, value, overwrite=True, tags=tag_names)


def _ensure_output_tu_indices(params: "ParameterTree", stack: "ComputeStack") -> None:
    if stack.layers is None:
        return
    if stack.tu_id_to_idx is None:
        return

    from biocomp.tumasking import build_output_tu_indices
    from biocomp.nodeutils import NON_GRAD_TAG

    for layer in stack.layers:
        if layer.f_type != "aggregation":
            continue
        assert layer.namespace is not None, "Aggregation layer missing namespace"
        output_tu_path = f"{layer.namespace}/output_tu_indices"
        if output_tu_path in params:
            continue
        n_outputs = layer.get_n_outputs()
        tu_indices = build_output_tu_indices(stack, layer.nodes, stack.tu_id_to_idx, n_outputs)
        params.at(output_tu_path, tu_indices, tags=[NON_GRAD_TAG], overwrite=None)


def _get_node_ratios(
    params: "ParameterTree",
    namespace: str,
    node_idx: int,
    n_outputs: int,
) -> np.ndarray:
    from biocomp.ratio_utils import decode_ratios_numpy

    return decode_ratios_numpy(params, namespace, node_idx, n_outputs)

_DEFAULT_RATIO_MIN = 0.001
_DEFAULT_RATIO_MAX = 10.0


def _store_learned_ratio_inits(
    params: "ParameterTree",
    stack: "ComputeStack",
) -> None:
    """Update aggregation node members with learned ratios as init values.

    Sets ratio_range["init"] to the current learned ratio while keeping
    locked=False. This ensures ratios are INITIALIZED with learned values
    after rebuild but remain OPTIMIZABLE.
    """
    if stack.layers is None:
        return

    for layer in stack.layers:
        if layer.f_type != "aggregation":
            continue
        namespace = layer.namespace
        if namespace is None:
            continue

        n_outputs = layer.get_n_outputs()

        for node_idx, node in enumerate(layer.nodes):
            ratios = _get_node_ratios(params, namespace, node_idx, n_outputs)

            network = stack.networks[node.network_id]
            assert network.compute_graph is not None
            graph_node = network.compute_graph.nodes.get(node.node_id)
            if graph_node is None or graph_node.extra is None:
                continue

            members = graph_node.extra.get("members", {})
            if not isinstance(members, dict):
                continue

            sorted_ids = sorted(members.keys())
            for slot, member_id in enumerate(sorted_ids):
                if slot >= len(ratios):
                    break
                if member_id not in members or not isinstance(members[member_id], dict):
                    continue

                m = members[member_id]
                learned_ratio = float(ratios[slot])

                if m.get("locked", False):
                    continue

                existing_range = m.get("ratio_range") or {}
                m["ratio_range"] = {
                    "min": existing_range.get("min", _DEFAULT_RATIO_MIN),
                    "max": existing_range.get("max", _DEFAULT_RATIO_MAX),
                    "init": learned_ratio,
                }
                m["ratio"] = learned_ratio
                m["locked"] = False


def _collect_ratio_pruning_candidates(
    params: "ParameterTree",
    stack: "ComputeStack",
    network_id: int,
    ratio_threshold: float,
) -> tuple[set[str], set[str], dict[str, float]]:
    from biocomp.tumasking import extract_tu_ids_from_network, build_output_tu_indices

    tu_id_to_idx = stack.tu_id_to_idx or {}
    idx_to_id = {idx: tu_id for tu_id, idx in tu_id_to_idx.items()}

    all_tu_ids = extract_tu_ids_from_network(stack.networks[network_id])
    tu_strengths = {tu_id: 0.0 for tu_id in all_tu_ids}
    candidates: set[str] = set()

    if stack.layers is None:
        return candidates, all_tu_ids, tu_strengths

    for layer in stack.layers:
        if layer.f_type != "aggregation":
            continue
        namespace = layer.namespace
        assert namespace is not None, "Aggregation layer missing namespace"
        n_outputs = layer.get_n_outputs()

        output_tu_path = f"{namespace}/output_tu_indices"
        if output_tu_path in params:
            tu_indices = np.asarray(params[output_tu_path])
        else:
            tu_indices = np.asarray(
                build_output_tu_indices(stack, layer.nodes, tu_id_to_idx, n_outputs)
            )

        for node_idx, node in enumerate(layer.nodes):
            if node.network_id != network_id:
                continue
            ratios = _get_node_ratios(params, namespace, node_idx, n_outputs)
            for slot in range(min(n_outputs, len(ratios))):
                tu_idx = int(tu_indices[node_idx, slot])
                if tu_idx < 0:
                    continue
                tu_id = idx_to_id.get(tu_idx)
                if tu_id is None:
                    continue
                ratio_val = float(np.asarray(ratios[slot]).item())
                strength = abs(ratio_val)
                if strength > tu_strengths.get(tu_id, 0.0):
                    tu_strengths[tu_id] = strength
                if strength < ratio_threshold:
                    candidates.add(tu_id)

    return candidates, all_tu_ids, tu_strengths


def identify_tus_to_prune(
    params: "ParameterTree",
    stack: "ComputeStack",
    dmanager: "DesignManager",
    ratio_threshold: float,
    use_soft_pruning: bool,
    preserve_minimum: int,
    prune_margin: float = 0.1,
    auto_lock_topology_tus: bool = True,
) -> dict[int, set[str]]:
    """Identify TUs to remove for each network based on normalized ratios."""
    from biocomp.tumasking_strategy import get_full_log_alpha

    with trace_scope("identify_tus_to_prune", component="design") as scope:
        scope.event(
            "config",
            "Pruning configuration",
            {
                "ratio_threshold": ratio_threshold,
                "use_soft_pruning": use_soft_pruning,
                "preserve_minimum": preserve_minimum,
                "prune_margin": prune_margin,
                "auto_lock_topology_tus": auto_lock_topology_tus,
                "n_networks": len(stack.networks),
            },
        )
        scope.snapshot("params", jax.device_get(params))

        tus_to_remove: dict[int, set[str]] = {}
        decision_counts = {"mask_below_threshold": 0, "ratio_below_threshold": 0}
        stack.ensure_tu_mapping(auto_lock_topology_tus=auto_lock_topology_tus)
        no_masking_tu_ids = stack.no_masking_tu_ids or set()
        tu_id_to_idx = stack.tu_id_to_idx or {}

        log_alpha_full = get_full_log_alpha(params)
        has_tu_masking = log_alpha_full is not None

        for net_idx in range(len(stack.networks)):
            candidates, all_tu_ids, tu_strengths = _collect_ratio_pruning_candidates(
                params, stack, net_idx, ratio_threshold
            )
            candidates = {tid for tid in candidates if tid not in no_masking_tu_ids}

            if use_soft_pruning and has_tu_masking:
                assert log_alpha_full is not None
                network_log_alpha = log_alpha_full[net_idx]
                if network_log_alpha.ndim > 1:
                    network_log_alpha = network_log_alpha.reshape(-1)

                for tu_id in all_tu_ids:
                    if tu_id in no_masking_tu_ids:
                        scope.decision(
                            "protect_tu",
                            outcome=False,
                            reason="in_no_masking_set",
                            inputs={"tu_id": tu_id, "network": net_idx},
                        )
                        continue
                    if tu_id in tu_id_to_idx:
                        tu_idx = tu_id_to_idx[tu_id]
                        if tu_idx < len(network_log_alpha):
                            prob = float(jax.nn.sigmoid(network_log_alpha[tu_idx]))
                            tu_strengths[tu_id] = max(tu_strengths.get(tu_id, 0.0), prob)
                            if prob < (0.5 - prune_margin):
                                candidates.add(tu_id)
                                decision_counts["mask_below_threshold"] += 1
                                scope.decision(
                                    "prune_tu",
                                    outcome=True,
                                    reason="mask_below_threshold",
                                    inputs={
                                        "tu_id": tu_id,
                                        "network": net_idx,
                                        "prob": prob,
                                    },
                                )

            # Log ratio-based pruning decisions
            for tu_id in candidates:
                if tu_id not in no_masking_tu_ids:
                    strength = tu_strengths.get(tu_id, 0.0)
                    if strength < ratio_threshold:
                        decision_counts["ratio_below_threshold"] += 1
                        scope.decision(
                            "prune_tu",
                            outcome=True,
                            reason="ratio_below_threshold",
                            inputs={
                                "tu_id": tu_id,
                                "network": net_idx,
                                "strength": strength,
                                "threshold": ratio_threshold,
                            },
                        )

            remaining = len(all_tu_ids) - len(candidates)
            if remaining < preserve_minimum:
                n_to_keep = preserve_minimum - remaining
                sorted_by_strength = sorted(candidates, key=lambda x: tu_strengths.get(x, 0.0))
                strongest_to_keep = set(sorted_by_strength[-n_to_keep:]) if n_to_keep > 0 else set()
                for tu_id in strongest_to_keep:
                    scope.decision(
                        "preserve_tu",
                        outcome=False,
                        reason="preserve_minimum",
                        inputs={
                            "tu_id": tu_id,
                            "network": net_idx,
                            "strength": tu_strengths.get(tu_id, 0.0),
                            "n_to_keep": n_to_keep,
                        },
                    )
                candidates = candidates - strongest_to_keep

            tus_to_remove[net_idx] = candidates

        scope.event(
            "result",
            "TU pruning complete",
            {
                "tus_to_remove": {k: list(v) for k, v in tus_to_remove.items()},
                "total_to_remove": sum(len(v) for v in tus_to_remove.values()),
                "decision_counts": decision_counts,
            },
        )

        return tus_to_remove


def _apply_hard_pruning_mask(
    params: "ParameterTree",
    stack: "ComputeStack",
    tus_to_remove: dict[int, set[str]],
    auto_lock_topology_tus: bool = True,
) -> int:
    from biocomp.tumasking import set_binary_tu_mask

    tu_id_to_idx = stack.ensure_tu_mapping(auto_lock_topology_tus=auto_lock_topology_tus)
    no_masking_tu_ids = stack.no_masking_tu_ids or set()

    idx_to_id = {idx: tu_id for tu_id, idx in tu_id_to_idx.items()}
    tu_ids = [idx_to_id[i] for i in range(len(idx_to_id))]

    missing = set()
    disabled_tus: dict[int, set[str]] = {}
    applied_pairs: set[tuple[int, int]] = set()

    for net_idx, tu_ids_in_net in tus_to_remove.items():
        assert 0 <= net_idx < len(stack.networks), (
            f"network_id {net_idx} out of range for {len(stack.networks)} networks"
        )
        for tu_id in tu_ids_in_net:
            if tu_id in no_masking_tu_ids:
                continue
            if tu_id not in tu_id_to_idx:
                missing.add(tu_id)
                continue
            disabled_tus.setdefault(net_idx, set()).add(tu_id)
            applied_pairs.add((net_idx, tu_id_to_idx[tu_id]))

    assert not missing, f"Hard-prune requested unknown TU IDs (sample): {sorted(list(missing))[:5]}"

    set_binary_tu_mask(
        params,
        tu_ids=tu_ids,
        tu_id_to_idx=tu_id_to_idx,
        n_networks=len(stack.networks),
        disabled_tus=disabled_tus,
    )
    _ensure_output_tu_indices(params, stack)

    return len(applied_pairs)


def _merge_surviving_params(
    old_params: "ParameterTree",
    new_params: "ParameterTree",
) -> "ParameterTree":
    """Transfer compatible params from old to new by path + shape matching."""
    from biocomp.parameters import isArrayRef

    skip_patterns = ("tu_log_alpha", "latent_tu", "tu_binary_mask", "protected_tu")

    for path, old_val in old_params.data.iter_leaves():
        path_str = str(path)

        if any(p in path_str for p in skip_patterns):
            continue

        if isArrayRef(old_val):
            continue

        try:
            new_leaf = new_params.data.get_at(path_str, get_leaf_value=False)
        except (KeyError, TypeError):
            continue

        new_val = new_leaf.value

        if isArrayRef(new_val):
            continue

        if not hasattr(old_val, "shape") or not hasattr(new_val, "shape"):
            continue
        if old_val.shape != new_val.shape:
            continue
        if hasattr(old_val, "dtype") and hasattr(new_val, "dtype"):
            old_dtype = old_val.dtype
            new_dtype = new_val.dtype
            new_inexact = np.issubdtype(new_dtype, np.inexact) or np.issubdtype(
                new_dtype, np.complexfloating
            )
            old_inexact = np.issubdtype(old_dtype, np.inexact) or np.issubdtype(
                old_dtype, np.complexfloating
            )
            if new_inexact and not old_inexact:
                continue
            if not new_inexact and old_dtype != new_dtype:
                continue

        tag_names = _get_tag_names(new_params, path_str)
        new_params.at(path_str, old_val, overwrite=True, tags=tag_names)

    return new_params


def _get_member_ids(extra: dict) -> list[str]:
    if "_sorted_member_ids" in extra and extra["_sorted_member_ids"]:
        return list(extra["_sorted_member_ids"])
    members = extra.get("members")
    if isinstance(members, dict) and members:
        return sorted(members.keys())
    return []


def _build_aggregation_ratio_map(
    params: "ParameterTree",
    stack: "ComputeStack",
) -> dict[tuple[int, tuple[str, ...]], np.ndarray]:
    ratio_map: dict[tuple[int, tuple[str, ...]], np.ndarray] = {}
    if stack.layers is None:
        return ratio_map
    for layer in stack.layers:
        if layer.f_type != "aggregation":
            continue
        namespace = layer.namespace
        if namespace is None:
            continue
        ratio_path = f"{namespace}/ratios"
        if ratio_path not in params:
            continue
        ratios = np.asarray(params[ratio_path])
        for node_idx, node in enumerate(layer.nodes):
            graph_node = node.get(stack)
            member_ids = _get_member_ids(graph_node.extra or {})
            if not member_ids:
                continue
            key = (node.network_id, tuple(member_ids))
            if key not in ratio_map:
                ratio_map[key] = ratios[node_idx]
    return ratio_map


def _get_bias_protein(extra: dict) -> str | None:
    fluo_specs = extra.get("fluo_bias") or extra.get("fluo_bias_data") or {}
    if isinstance(fluo_specs, dict):
        protein = fluo_specs.get("protein")
        if protein:
            return protein
    return extra.get("protein_name") or extra.get("name")


def _build_bias_map(
    params: "ParameterTree",
    stack: "ComputeStack",
) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    bias_map: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    if stack.layers is None:
        return bias_map
    for layer in stack.layers:
        if layer.f_type not in ("bias", "hard_bias"):
            continue
        namespace = layer.namespace
        if namespace is None:
            continue
        raw_path = f"{namespace}/raw_value"
        if raw_path not in params:
            continue
        raw_values = np.asarray(params[raw_path])
        min_values = np.asarray(params[f"{namespace}/min_value"])
        max_values = np.asarray(params[f"{namespace}/max_value"])
        scale_path = f"{namespace}/scale"
        scales = np.asarray(params[scale_path]) if scale_path in params else None
        for node_idx, node in enumerate(layer.nodes):
            graph_node = node.get(stack)
            protein = _get_bias_protein(graph_node.extra or {})
            if not protein:
                continue
            key = (node.network_id, protein)
            bias_map[key] = {
                "raw": raw_values[node_idx],
                "min": min_values[node_idx],
                "max": max_values[node_idx],
                "scale": scales[node_idx] if scales is not None else None,
            }
    return bias_map


def _sorted_incoming_edges(node, stack):
    edges = node.get_incoming_edges(stack)
    return sorted(edges, key=lambda e: e.to_input_slot)


def _build_rate_map(
    params: "ParameterTree",
    stack: "ComputeStack",
) -> dict[tuple[int, str, tuple[str, ...]], np.ndarray]:
    rate_map: dict[tuple[int, str, tuple[str, ...]], np.ndarray] = {}
    if stack.layers is None:
        return rate_map
    for layer in stack.layers:
        if layer.f_type not in ("transcription", "translation"):
            continue
        namespace = layer.namespace
        if namespace is None:
            continue
        if f"{namespace}/tc_rate" in params:
            rate_name = "tc_rate"
        elif f"{namespace}/tl_rate" in params:
            rate_name = "tl_rate"
        else:
            continue
        rate_path = f"{namespace}/{rate_name}"
        rates = np.asarray(params[rate_path])
        for node_idx, node in enumerate(layer.nodes):
            edges = _sorted_incoming_edges(node, stack)
            for input_idx, edge in enumerate(edges):
                tu_ids = edge.extra.get("tu_id", []) if edge.extra else []
                if not tu_ids:
                    continue
                key = (node.network_id, rate_name, tuple(sorted(tu_ids)))
                if key not in rate_map:
                    rate_map[key] = rates[node_idx, input_idx]
    return rate_map


def _select_rep_target(array: np.ndarray) -> tuple[np.ndarray, tuple[int, int] | None]:
    if array.ndim >= 2 and array.shape[0] == 1 and array.shape[1] == 1:
        return array, (0, 0)
    return array, None


def _restore_aggregation_ratios(
    ratio_map: dict[tuple[int, tuple[str, ...]], np.ndarray],
    params: "ParameterTree",
    stack: "ComputeStack",
) -> int:
    restored = 0
    if stack.layers is None:
        return restored
    for layer in stack.layers:
        if layer.f_type != "aggregation":
            continue
        namespace = layer.namespace
        if namespace is None:
            continue
        ratio_path = f"{namespace}/ratios"
        if ratio_path not in params:
            continue
        ratio_min_path = f"{namespace}/ratio_min"
        ratio_max_path = f"{namespace}/ratio_max"
        ratios_arr = np.array(params[ratio_path], copy=True)
        ratio_min = np.asarray(params[ratio_min_path]) if ratio_min_path in params else None
        ratio_max = np.asarray(params[ratio_max_path]) if ratio_max_path in params else None
        ratios_arr, rep_tgt = _select_rep_target(ratios_arr)
        for node_idx, node in enumerate(layer.nodes):
            graph_node = node.get(stack)
            member_ids = _get_member_ids(graph_node.extra or {})
            if not member_ids:
                continue
            key = (node.network_id, tuple(member_ids))
            if key not in ratio_map:
                continue
            old_ratios = np.asarray(ratio_map[key])
            if rep_tgt is None:
                target = ratios_arr[node_idx]
                mins = ratio_min[node_idx] if ratio_min is not None else None
                maxs = ratio_max[node_idx] if ratio_max is not None else None
                new_vals = old_ratios[: target.shape[0]]
                if mins is not None and maxs is not None:
                    new_vals = np.clip(new_vals, mins[: target.shape[0]], maxs[: target.shape[0]])
                target[: new_vals.shape[0]] = new_vals
                ratios_arr[node_idx] = target
            else:
                r, t = rep_tgt
                target = ratios_arr[r, t, node_idx]
                mins = ratio_min[r, t, node_idx] if ratio_min is not None else None
                maxs = ratio_max[r, t, node_idx] if ratio_max is not None else None
                new_vals = old_ratios[: target.shape[0]]
                if mins is not None and maxs is not None:
                    new_vals = np.clip(new_vals, mins[: target.shape[0]], maxs[: target.shape[0]])
                target[: new_vals.shape[0]] = new_vals
                ratios_arr[r, t, node_idx] = target
            restored += 1
        _set_param_value(params, ratio_path, ratios_arr)
    return restored


def _restore_bias_values(
    bias_map: dict[tuple[int, str], dict[str, np.ndarray]],
    params: "ParameterTree",
    stack: "ComputeStack",
) -> int:
    restored = 0
    if stack.layers is None:
        return restored
    for layer in stack.layers:
        if layer.f_type not in ("bias", "hard_bias"):
            continue
        namespace = layer.namespace
        if namespace is None:
            continue
        raw_path = f"{namespace}/raw_value"
        if raw_path not in params:
            continue
        raw_values = np.array(params[raw_path], copy=True)
        raw_values, rep_tgt = _select_rep_target(raw_values)
        for node_idx, node in enumerate(layer.nodes):
            graph_node = node.get(stack)
            protein = _get_bias_protein(graph_node.extra or {})
            if not protein:
                continue
            key = (node.network_id, protein)
            if key not in bias_map:
                continue
            raw_val = bias_map[key]["raw"]
            if rep_tgt is None:
                raw_values[node_idx] = raw_val
            else:
                r, t = rep_tgt
                raw_values[r, t, node_idx] = raw_val
            restored += 1
        _set_param_value(params, raw_path, raw_values)
    return restored


def _restore_rate_values(
    rate_map: dict[tuple[int, str, tuple[str, ...]], np.ndarray],
    params: "ParameterTree",
    stack: "ComputeStack",
) -> int:
    restored = 0
    if stack.layers is None:
        return restored
    for layer in stack.layers:
        if layer.f_type not in ("transcription", "translation"):
            continue
        namespace = layer.namespace
        if namespace is None:
            continue
        if f"{namespace}/tc_rate" in params:
            rate_name = "tc_rate"
        elif f"{namespace}/tl_rate" in params:
            rate_name = "tl_rate"
        else:
            continue
        rate_path = f"{namespace}/{rate_name}"
        rates_arr = np.array(params[rate_path], copy=True)
        rates_arr, rep_tgt = _select_rep_target(rates_arr)
        for node_idx, node in enumerate(layer.nodes):
            edges = _sorted_incoming_edges(node, stack)
            for input_idx, edge in enumerate(edges):
                tu_ids = edge.extra.get("tu_id", []) if edge.extra else []
                if not tu_ids:
                    continue
                key = (node.network_id, rate_name, tuple(sorted(tu_ids)))
                if key not in rate_map:
                    continue
                old_rate = np.asarray(rate_map[key])
                if rep_tgt is None:
                    rates_arr[node_idx, input_idx] = old_rate
                else:
                    r, t = rep_tgt
                    rates_arr[r, t, node_idx, input_idx] = old_rate
                restored += 1
        _set_param_value(params, rate_path, rates_arr)
    return restored


def _restore_params_by_semantics(
    old_params: "ParameterTree",
    old_stack: "ComputeStack",
    new_params: "ParameterTree",
    new_stack: "ComputeStack",
) -> dict[str, int]:
    ratio_map = _build_aggregation_ratio_map(old_params, old_stack)
    bias_map = _build_bias_map(old_params, old_stack)
    rate_map = _build_rate_map(old_params, old_stack)

    restored_ratios = _restore_aggregation_ratios(ratio_map, new_params, new_stack)
    restored_bias = _restore_bias_values(bias_map, new_params, new_stack)
    restored_rates = _restore_rate_values(rate_map, new_params, new_stack)

    return {
        "aggregation_nodes_restored": restored_ratios,
        "bias_nodes_restored": restored_bias,
        "rate_edges_restored": restored_rates,
    }

def _remap_tu_log_alpha(
    old_log_alpha: jnp.ndarray,
    old_tu_id_to_idx: dict[str, int],
    new_tu_id_to_idx: dict[str, int],
    init_value: float = 2.0,
) -> jnp.ndarray:
    """Remap tu_log_alpha from old to new TU indexing."""
    n_networks = old_log_alpha.shape[0]
    n_new_tus = len(new_tu_id_to_idx)
    new_log_alpha = jnp.full((n_networks, n_new_tus), init_value)

    old_idx_to_id = {v: k for k, v in old_tu_id_to_idx.items()}

    for old_idx in range(old_log_alpha.shape[-1]):
        tu_id = old_idx_to_id.get(old_idx)
        if tu_id and tu_id in new_tu_id_to_idx:
            new_idx = new_tu_id_to_idx[tu_id]
            new_log_alpha = new_log_alpha.at[:, new_idx].set(old_log_alpha[:, old_idx])

    return new_log_alpha


def _is_valid_network(network) -> bool:
    """Check if a network is valid (has at least one output node)."""
    cg = getattr(network, "compute_graph", None)
    if cg is None or not cg.nodes:
        return False
    return sum(1 for n in cg.nodes.values() if n.node_type == "output") >= 1


def hard_prune_and_rebuild(
    dmanager: "DesignManager",
    dconf: "DesignConfig",
    model: "BiocompModel",
    stack: "ComputeStack",
    params: "ParameterTree",
    tus_to_remove: dict[int, set[str]],
    key: jax.Array,
    lock_ratios: bool = False,
) -> tuple["DesignManager", "ComputeStack", "ParameterTree"]:
    """Execute hard pruning: mark TUs disabled, commit, rebuild."""
    from .design import DesignManager, initialize_params
    from .stack_commit import commit_structure
    from .network import Network

    with trace_scope("hard_prune_and_rebuild", component="commit") as scope:
        scope.event(
            "start",
            "Starting hard prune and rebuild",
            {
                "n_networks": len(stack.networks),
                "tus_to_remove": {k: list(v) for k, v in tus_to_remove.items()},
                "lock_ratios": lock_ratios,
            },
        )
        scope.snapshot("params_before", jax.device_get(params))

        stack.ensure_tu_mapping(auto_lock_topology_tus=dconf.auto_lock_topology_tus)
        old_tu_id_to_idx = stack.tu_id_to_idx or {}
        old_n_tus = len(old_tu_id_to_idx)
        n_networks = len(stack.networks)

        _store_learned_ratio_inits(params, stack)

        params_for_commit = deepcopy(params)

        applied = _apply_hard_pruning_mask(
            params_for_commit,
            stack,
            tus_to_remove,
            auto_lock_topology_tus=dconf.auto_lock_topology_tus,
        )

        committed_networks = commit_structure(stack, params_for_commit, lock_ratios=lock_ratios)

        # Filter degenerate networks
        valid_networks: list[Network] = []
        for i, net in enumerate(committed_networks):
            if _is_valid_network(net):
                valid_networks.append(net)
            else:
                scope.decision(
                    "filter_degenerate",
                    outcome="removed",
                    reason="zero_outputs",
                    inputs={"network_idx": i, "network_name": net.name},
                )
                logger.warning(f"[HARD-PRUNE] Network {i} ({net.name}) became degenerate, removing")

        if not valid_networks:
            raise RuntimeError(
                "All networks became degenerate after pruning. "
                "Try reducing L0 penalty, increasing preserve_minimum, or adjusting ratio_threshold."
            )

        total_removed = sum(len(tus) for tus in tus_to_remove.values())
        logger.info(
            f"[HARD-PRUNE] Committed networks, requested {total_removed} TUs, applied {applied}"
        )
        if len(valid_networks) < len(committed_networks):
            logger.info(
                f"[HARD-PRUNE] Filtered {len(committed_networks) - len(valid_networks)} "
                f"degenerate networks, {len(valid_networks)} remaining"
            )

        new_dmanager = DesignManager(
            targets=dmanager.targets,
            networks=valid_networks,
            sampling=dmanager.sampling,
            enable_tu_masking=dmanager.enable_tu_masking,
        )

        pkey, init_key = jax.random.split(key)
        new_stack = new_dmanager.build_stack(
            model,
            unlock_ratios=not lock_ratios,
            use_latent_ratios=dconf.use_latent_ratios,
            latent_dim=dconf.latent_dim,
            latent_hidden_dim=dconf.latent_hidden_dim,
            auto_lock_topology_tus=dconf.auto_lock_topology_tus,
        )

        new_n_tus = new_dmanager.n_tus if new_dmanager.enable_tu_masking else 0
        new_tu_id_to_idx = new_dmanager.tu_id_to_idx if new_dmanager.enable_tu_masking else {}

        logger.info(f"[HARD-PRUNE] Rebuilt stack: {old_n_tus} -> {new_n_tus} TUs")

        new_params = initialize_params(
            new_stack,
            dconf.n_replicates,
            new_dmanager.n_targets,
            model.shared_params,
            init_key,
            strategy=build_strategy_from_config(dconf),
            n_tus=new_n_tus,
            n_networks=len(new_dmanager.networks),
            no_masking_tu_ids=new_stack.no_masking_tu_ids,
            tu_id_to_idx=new_stack.tu_id_to_idx,
        )

        expanded_old_params = _expand_params_for_merge(params)
        new_params = _merge_surviving_params(expanded_old_params, new_params)
        restore_stats = _restore_params_by_semantics(params, stack, new_params, new_stack)

        if TU_LOG_ALPHA_PATH in params and TU_LOG_ALPHA_PATH in new_params and new_n_tus > 0:
            old_log_alpha = params[TU_LOG_ALPHA_PATH]
            if old_log_alpha.ndim == 4:
                old_2d = old_log_alpha[0, 0]
            elif old_log_alpha.ndim == 2:
                old_2d = old_log_alpha
            else:
                old_2d = old_log_alpha.reshape(n_networks, -1)

            remapped = _remap_tu_log_alpha(old_2d, old_tu_id_to_idx, new_tu_id_to_idx)

            new_log_alpha = new_params[TU_LOG_ALPHA_PATH]
            if new_log_alpha.ndim == 4:
                remapped_4d = jnp.tile(
                    remapped[None, None, :, :],
                    (dconf.n_replicates, new_dmanager.n_targets, 1, 1),
                )
                new_params.at(TU_LOG_ALPHA_PATH, remapped_4d, overwrite=True)
            elif new_log_alpha.ndim == 2:
                new_params.at(TU_LOG_ALPHA_PATH, remapped, overwrite=True)

        scope.event(
            "param_carryover",
            "Hard prune param carryover summary",
            {
                "old_param_leaves": len(list(params.data.iter_leaves())),
                "new_param_leaves": len(list(new_params.data.iter_leaves())),
                "carryover_stats": restore_stats,
            },
        )

        scope.event(
            "complete",
            "Hard prune and rebuild complete",
            {
                "old_n_tus": old_n_tus,
                "new_n_tus": new_n_tus,
                "old_n_networks": n_networks,
                "new_n_networks": len(valid_networks),
            },
        )
        scope.snapshot("params_after", jax.device_get(new_params))

        return new_dmanager, new_stack, new_params


def run_with_hard_pruning(
    dmanager: "DesignManager",
    dconf: "DesignConfig",
    model: "BiocompModel",
    loggers: list[tuple[int, Callable]] | None = None,
    logger_objects: list | None = None,
    async_handler=None,
    lock_ratios: bool = False,
):
    """Design optimization with periodic hard-pruning."""
    from .design import start
    from .design_session import PhaseTimer as _PhaseTimer
    from .design import sample_for_evaluation, evaluate_design

    if dconf.n_replicates != 1:
        raise ValueError(
            f"hard_pruning_enabled=True requires n_replicates=1, got {dconf.n_replicates}. "
            "Run separate single-replicate designs or disable hard pruning."
        )
    if dmanager.n_targets != 1:
        raise ValueError(
            f"hard_pruning_enabled=True requires n_targets=1, got {dmanager.n_targets}. "
            "Run separate single-target designs or disable hard pruning."
        )

    timer = _PhaseTimer()
    logger.info("=" * 60)
    logger.info("DESIGN OPTIMIZATION WITH HARD-PRUNING")
    logger.info("=" * 60)

    _, _, loop_key = jax.random.split(dconf.seed_key, 3)

    steps_per_epoch = max(1, dconf.n_batches_per_epoch // dconf.batches_per_step)
    total_steps = int(dconf.n_epochs * steps_per_epoch)
    steps_per_segment = dconf.hard_pruning_interval
    n_segments = (total_steps + steps_per_segment - 1) // steps_per_segment

    logger.info(
        f"[HARD-PRUNE] Total steps: {total_steps}, interval: {steps_per_segment}, "
        f"segments: {n_segments}"
    )

    trace_here(
        "hard_pruning_config",
        component="design",
        data={
            "total_steps": total_steps,
            "steps_per_segment": steps_per_segment,
            "n_segments": n_segments,
            "n_networks": len(dmanager.networks),
            "n_targets": dmanager.n_targets,
        },
    )

    current_dmanager = dmanager
    accumulated_loss_history: list = []
    accumulated_step_history: list = []
    segment_params: "ParameterTree" | None = None
    current_params: "ParameterTree" | None = None

    for segment_idx in range(n_segments):
        segment_start_step = segment_idx * steps_per_segment
        segment_end_step = min((segment_idx + 1) * steps_per_segment, total_steps)
        segment_steps = segment_end_step - segment_start_step

        if segment_steps <= 0:
            break

        segment_epochs = 1
        segment_batches_per_epoch = segment_steps * dconf.batches_per_step

        new_seed = int(jax.random.key_data(jax.random.fold_in(loop_key, segment_idx))[0]) % (2**31)
        segment_config = dconf.model_copy(
            update={
                "n_epochs": segment_epochs,
                "n_batches_per_epoch": segment_batches_per_epoch,
                "hard_pruning_enabled": False,
                "pluggable_optimizer": None,
                "seed": new_seed,
            }
        )

        logger.info(
            f"[HARD-PRUNE] Segment {segment_idx + 1}/{n_segments}: "
            f"steps {segment_start_step}-{segment_end_step}"
        )

        segment_params, segment_loss_history, segment_step_history = start(
            current_dmanager,
            segment_config,
            model,
            loggers=loggers,
            logger_objects=logger_objects,
            async_handler=async_handler,
            lock_ratios=lock_ratios,
            initial_params=current_params,
        )
        current_params = segment_params

        accumulated_loss_history.extend(segment_loss_history)
        accumulated_step_history.extend(segment_step_history)

        if segment_idx < n_segments - 1:
            timer.start("prune", "[HARD-PRUNE] Identifying TUs to prune...")

            temp_stack = current_dmanager.build_stack(
                model,
                unlock_ratios=not lock_ratios,
                use_latent_ratios=dconf.use_latent_ratios,
                latent_dim=dconf.latent_dim,
                latent_hidden_dim=dconf.latent_hidden_dim,
                auto_lock_topology_tus=dconf.auto_lock_topology_tus,
            )

            from biocomp.jaxutils import tree_get

            single_rep_params = tree_get(segment_params, (0, 0))

            tus_to_remove = identify_tus_to_prune(
                single_rep_params,
                temp_stack,
                current_dmanager,
                ratio_threshold=dconf.hard_pruning_ratio_threshold,
                use_soft_pruning=dconf.enable_tu_masking,
                preserve_minimum=dconf.hard_pruning_preserve_minimum_tus,
                prune_margin=dconf.hard_pruning_prune_margin,
                auto_lock_topology_tus=dconf.auto_lock_topology_tus,
            )

            timer.end("prune")

            total_to_remove = sum(len(tus) for tus in tus_to_remove.values())
            loss_pre_mean: float | None = None
            if total_to_remove > 0:
                with trace_scope("hard_prune_compare", component="prune") as prune_scope:
                    try:
                        compare_key = jax.random.fold_in(loop_key, segment_idx + 2000)
                        xraw, yraw = sample_for_evaluation(
                            current_dmanager,
                            dconf,
                            segment_params,
                            n_eval_samples=256,
                            key=compare_key,
                        )
                        yhat_pre, loss_pre = evaluate_design(
                            current_dmanager,
                            dconf,
                            model,
                            segment_params,
                            xraw,
                            yraw,
                            compare_key,
                            store_predictions=True,
                        )
                        prune_scope.snapshot("xraw", jax.device_get(xraw))
                        prune_scope.snapshot("yraw", jax.device_get(yraw))
                        prune_scope.snapshot("yhat_pre", jax.device_get(yhat_pre))
                        prune_scope.snapshot("loss_pre", jax.device_get(loss_pre))
                        loss_pre_mean = float(np.asarray(loss_pre).mean())
                        prune_scope.event(
                            "pre_prune",
                            "Pre-prune evaluation complete",
                            {
                                "mean_loss_pre": loss_pre_mean,
                                "n_networks": len(current_dmanager.networks),
                                "network_names": [n.name for n in current_dmanager.networks],
                            },
                        )
                    except Exception as e:
                        prune_scope.event(
                            "pre_prune_error",
                            "Pre-prune evaluation failed",
                            {"error": str(e)},
                        )

                prune_key = jax.random.fold_in(loop_key, segment_idx + 1000)
                current_dmanager, _, current_params = hard_prune_and_rebuild(
                    current_dmanager,
                    dconf,
                    model,
                    temp_stack,
                    single_rep_params,
                    tus_to_remove,
                    prune_key,
                    lock_ratios=lock_ratios,
                )
                logger.info(f"[HARD-PRUNE] Removed {total_to_remove} TUs, rebuilt stack")

                with trace_scope("hard_prune_compare", component="prune") as prune_scope:
                    try:
                        compare_key = jax.random.fold_in(loop_key, segment_idx + 2000)
                        xraw, yraw = sample_for_evaluation(
                            current_dmanager,
                            dconf,
                            current_params,
                            n_eval_samples=256,
                            key=compare_key,
                        )
                        yhat_post, loss_post = evaluate_design(
                            current_dmanager,
                            dconf,
                            model,
                            current_params,
                            xraw,
                            yraw,
                            compare_key,
                            store_predictions=True,
                        )
                        prune_scope.snapshot("xraw", jax.device_get(xraw))
                        prune_scope.snapshot("yraw", jax.device_get(yraw))
                        prune_scope.snapshot("yhat_post", jax.device_get(yhat_post))
                        prune_scope.snapshot("loss_post", jax.device_get(loss_post))
                        loss_post_mean = float(np.asarray(loss_post).mean())
                        prune_scope.event(
                            "post_prune",
                            "Post-prune evaluation complete",
                            {
                                "mean_loss_post": loss_post_mean,
                                "n_networks": len(current_dmanager.networks),
                                "network_names": [n.name for n in current_dmanager.networks],
                            },
                        )

                        if loss_pre_mean is not None and loss_pre_mean > 1e-8:
                            loss_increase = (loss_post_mean - loss_pre_mean) / loss_pre_mean
                            if loss_increase > 0.20:
                                logger.warning(
                                    f"[HARD-PRUNE] Loss regression: "
                                    f"{loss_pre_mean:.4f} -> {loss_post_mean:.4f} "
                                    f"({loss_increase:.1%} increase)"
                                )
                                prune_scope.event(
                                    "loss_regression",
                                    "Significant loss increase after pruning",
                                    {
                                        "loss_pre": loss_pre_mean,
                                        "loss_post": loss_post_mean,
                                        "increase_pct": loss_increase * 100,
                                        "tus_removed": total_to_remove,
                                    },
                                )
                                prune_scope.snapshot("current_params", jax.device_get(current_params))
                                prune_scope.snapshot("tus_to_remove", tus_to_remove)
                    except Exception as e:
                        prune_scope.event(
                            "post_prune_error",
                            "Post-prune evaluation failed",
                            {"error": str(e)},
                        )
            else:
                logger.info("[HARD-PRUNE] No TUs to remove, continuing with current structure")

    logger.info("=" * 60)
    logger.info(f"HARD-PRUNING OPTIMIZATION COMPLETE in {timer.total():.2f}s")
    logger.info("=" * 60)

    assert segment_params is not None, "No optimization segments were run"
    return segment_params, accumulated_loss_history, accumulated_step_history, current_dmanager
