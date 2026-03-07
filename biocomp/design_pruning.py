from __future__ import annotations

from typing import TYPE_CHECKING
from copy import deepcopy

import numpy as np
import jax
import jax.numpy as jnp

from .logging_config import get_logger
from .tumasking import TU_LOG_ALPHA_PATH
from .tumasking_strategy import TUMaskingMode, build_strategy_from_config
from .tracing import trace_scope, trace_here
from .logger_dispatch import LoggerDispatch
from .step_history import StepHistorySnapshot
from .ratio_schema import get_slot_entries, set_ratio_schema_slots, source_ids_in_slot_order

if TYPE_CHECKING:
    from .design import DesignManager, DesignConfig
    from .compute import ComputeStack
    from .parameters import ParameterTree
    from biocomptools.modelmodel import BiocompModel

logger = get_logger(__name__)


def _flatten_replicates_into_networks(
    dmanager: "DesignManager",
    n_replicates: int,
) -> "DesignManager":
    """Flatten replicates into the network dimension for hard-pruning mode.

    Deep copies each network for each replicate so they can diverge independently
    during pruning. Deep copies are essential because _store_learned_ratio_inits
    mutates graph_node.extra["ratio_schema"] on network objects.
    """
    original_networks = dmanager.networks
    flattened = []
    for rep_idx in range(n_replicates):
        for net in original_networks:
            copy = net.model_copy(deep=True)
            copy.name = f"{net.name}_rep{rep_idx}"
            flattened.append(copy)
    return dmanager.model_copy(update={"networks": flattened})


def _expand_params_for_merge(params: "ParameterTree") -> "ParameterTree":
    return jax.tree.map(
        lambda x: x.reshape((1, 1) + x.shape) if hasattr(x, "ndim") and x.ndim >= 0 else x,
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
    """Update aggregation ratio_schema with learned ratios as init values.

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

            slot_entries = get_slot_entries(graph_node.extra, require=False)
            if not slot_entries:
                continue

            for slot, entry in enumerate(slot_entries):
                if slot >= len(ratios):
                    break
                learned_ratio = float(ratios[slot])

                if bool(entry.get("locked", False)):
                    continue

                existing_range = entry.get("ratio_range") or {}
                entry["ratio_range"] = {
                    "min": existing_range.get("min", _DEFAULT_RATIO_MIN),
                    "max": existing_range.get("max", _DEFAULT_RATIO_MAX),
                    "init": learned_ratio,
                }
                entry["ratio"] = learned_ratio
                entry["locked"] = False

            set_ratio_schema_slots(graph_node.extra, slot_entries)


def _collect_ratio_pruning_candidates(
    params: "ParameterTree",
    stack: "ComputeStack",
    network_id: int,
    ratio_threshold: float,
) -> tuple[set[str], set[str], dict[str, float]]:
    from biocomp.tumasking import extract_tu_ids_from_network, build_output_tu_indices
    from biocomp.ratio_utils import normalize_ratios_for_pruning

    tu_id_to_idx = stack.tu_id_to_idx or {}
    idx_to_id = {idx: tu_id for tu_id, idx in tu_id_to_idx.items()}

    all_tu_ids = set(extract_tu_ids_from_network(stack.networks[network_id]))
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
            normalized = np.asarray(normalize_ratios_for_pruning(ratios, threshold=0.0))
            for slot in range(min(n_outputs, len(normalized))):
                tu_idx = int(tu_indices[node_idx, slot])
                if tu_idx < 0:
                    continue
                tu_id = idx_to_id.get(tu_idx)
                if tu_id is None:
                    continue
                ratio_val = float(np.asarray(normalized[slot]).item())
                strength = abs(ratio_val)
                if strength > tu_strengths.get(tu_id, 0.0):
                    tu_strengths[tu_id] = strength

    # Prune only if TU is weak everywhere it appears (max strength below threshold).
    candidates = {tu_id for tu_id, strength in tu_strengths.items() if strength < ratio_threshold}

    return candidates, all_tu_ids, tu_strengths


def _compute_hard_pruning_network_keep_count(
    n_networks: int,
    top_percent: float | None,
    min_networks: int | None,
) -> int | None:
    """Compute survivor count for hard-pruning network selection."""
    if top_percent is None and min_networks is None:
        return None

    keep_by_percent = 0
    if top_percent is not None:
        keep_by_percent = int(np.ceil((top_percent / 100.0) * n_networks))

    keep_by_min = min_networks or 0
    keep_count = max(keep_by_percent, keep_by_min)
    keep_count = max(1, keep_count)
    return min(n_networks, keep_count)


def _select_top_network_indices_from_losses(losses: np.ndarray, keep_count: int) -> list[int]:
    """Select top-performing network indices (lowest mean loss)."""
    if keep_count < 1:
        raise ValueError(f"keep_count must be >= 1, got {keep_count}")

    losses_arr = np.asarray(losses)
    if losses_arr.ndim == 0:
        raise ValueError("losses must have at least one dimension")
    if not np.all(np.isfinite(losses_arr)):
        raise ValueError("losses contains non-finite values")

    if losses_arr.ndim == 1:
        per_network_loss = losses_arr
    else:
        reduce_axes = tuple(range(losses_arr.ndim - 1))
        per_network_loss = losses_arr.mean(axis=reduce_axes)
    if per_network_loss.ndim != 1:
        raise ValueError(f"expected per-network 1D losses, got shape {per_network_loss.shape}")

    n_networks = int(per_network_loss.shape[0])
    if keep_count >= n_networks:
        return list(range(n_networks))

    sorted_idx = np.argsort(per_network_loss, kind="stable")
    return [int(i) for i in sorted_idx[:keep_count]]


def _extract_output_tu_ids(network) -> set[str]:
    """Return TU IDs found on any path upstream of output nodes."""
    graph = network.compute_graph
    if graph is None:
        return set()
    output_nodes = graph.get_nodes_by_type("output")
    if not output_nodes:
        return set()

    tu_ids: set[str] = set()
    visited_nodes: set[int] = set()
    queue = [node.node_id for node in output_nodes]

    while queue:
        node_id = queue.pop()
        if node_id in visited_nodes:
            continue
        visited_nodes.add(node_id)
        for edge in graph.get_incoming_edges(node_id):
            if edge.extra:
                tu_ids.update(edge.extra.get("tu_id", []))
            queue.append(edge.source_id)

    return tu_ids


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
            ratio_strengths = dict(tu_strengths)
            mask_strengths: dict[str, float] = {}

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
                            mask_strengths[tu_id] = prob
                            if prob < 0.5:
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

            # Protect confidently-enabled TUs from ratio-only pruning.
            # This prevents low-ratio TUs that are still clearly ON from being
            # removed and causing cascading structural deletions.
            if use_soft_pruning and has_tu_masking:
                for tu_id in list(candidates):
                    prob = mask_strengths.get(tu_id)
                    if prob is None:
                        continue
                    if prob > (0.5 + prune_margin):
                        strength = ratio_strengths.get(tu_id, 0.0)
                        if strength < ratio_threshold:
                            candidates.remove(tu_id)
                            scope.decision(
                                "preserve_tu",
                                outcome=False,
                                reason="mask_confidently_enabled",
                                inputs={
                                    "tu_id": tu_id,
                                    "network": net_idx,
                                    "prob": prob,
                                    "strength": strength,
                                    "threshold": ratio_threshold,
                                },
                            )

            # Log ratio-based pruning decisions
            for tu_id in candidates:
                if tu_id not in no_masking_tu_ids:
                    strength = ratio_strengths.get(tu_id, 0.0)
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
                sorted_by_strength = sorted(
                    candidates,
                    key=lambda x: max(ratio_strengths.get(x, 0.0), mask_strengths.get(x, 0.0)),
                )
                strongest_to_keep = set(sorted_by_strength[-n_to_keep:]) if n_to_keep > 0 else set()
                for tu_id in strongest_to_keep:
                    keep_strength = max(
                        ratio_strengths.get(tu_id, 0.0), mask_strengths.get(tu_id, 0.0)
                    )
                    scope.decision(
                        "preserve_tu",
                        outcome=False,
                        reason="preserve_minimum",
                        inputs={
                            "tu_id": tu_id,
                            "network": net_idx,
                            "strength": keep_strength,
                            "n_to_keep": n_to_keep,
                        },
                    )
                candidates = candidates - strongest_to_keep

            output_tu_ids = _extract_output_tu_ids(stack.networks[net_idx])
            if output_tu_ids:
                remaining_output_tus = output_tu_ids - candidates
                if not remaining_output_tus:
                    prunable_output_tus = output_tu_ids & candidates
                    if prunable_output_tus:
                        keep_tu = max(
                            prunable_output_tus,
                            key=lambda x: max(
                                ratio_strengths.get(x, 0.0), mask_strengths.get(x, 0.0)
                            ),
                        )
                        candidates.remove(keep_tu)
                        scope.decision(
                            "preserve_tu",
                            outcome=False,
                            reason="preserve_output_topology",
                            inputs={
                                "tu_id": keep_tu,
                                "network": net_idx,
                            },
                        )

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

    skip_patterns = (
        "tu_log_alpha",
        "latent_tu",
        "tu_binary_mask",
        "protected_tu",
        "output_tu_indices",
    )

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


def transfer_params_to_new_stack(
    old_params: "ParameterTree",
    old_stack: "ComputeStack",
    new_params: "ParameterTree",
    new_stack: "ComputeStack",
) -> tuple["ParameterTree", dict[str, int]]:
    """Transfer compatible + semantic parameters from old stack/params to new stack/params."""
    expanded_old_params = _expand_params_for_merge(old_params)
    merged_params = _merge_surviving_params(expanded_old_params, new_params)
    restore_stats = _restore_params_by_semantics(old_params, old_stack, merged_params, new_stack)
    return merged_params, restore_stats


def _build_aggregation_ratio_maps(
    params: "ParameterTree",
    stack: "ComputeStack",
) -> tuple[dict[tuple[int, tuple[str, ...]], np.ndarray], dict[tuple[int, str], float]]:
    """Build ratio carry-over maps for exact-node and per-source restoration.

    Exact node map is keyed by full source slot order for 1:1 node matches.
    Per-source map is keyed by source_id for subset matches after TU removal.
    Both maps use SSOT-decoded ratios (handles latent/direct uniformly).
    """
    exact_map: dict[tuple[int, tuple[str, ...]], np.ndarray] = {}
    source_map: dict[tuple[int, str], float] = {}
    if stack.layers is None:
        return exact_map, source_map
    for layer in stack.layers:
        if layer.f_type != "aggregation":
            continue
        namespace = layer.namespace
        if namespace is None:
            continue

        n_outputs = layer.get_n_outputs()
        for node_idx, node in enumerate(layer.nodes):
            graph_node = node.get(stack)
            source_ids = source_ids_in_slot_order(graph_node.extra)
            if not source_ids:
                continue

            decoded = np.asarray(_get_node_ratios(params, namespace, node_idx, n_outputs))
            decoded = decoded[: len(source_ids)]
            if decoded.size == 0:
                continue

            exact_key = (node.network_id, tuple(source_ids))
            exact_map.setdefault(exact_key, decoded.copy())

            for slot, source_id in enumerate(source_ids):
                if slot >= decoded.shape[0]:
                    break
                source_key = (node.network_id, str(source_id))
                ratio_val = float(decoded[slot])
                if source_key not in source_map:
                    source_map[source_key] = ratio_val
                else:
                    # If a source appears multiple times, preserve the strongest magnitude.
                    if abs(ratio_val) > abs(source_map[source_key]):
                        source_map[source_key] = ratio_val
    return exact_map, source_map


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


def _restore_ratio_target(
    target: np.ndarray,
    *,
    network_id: int,
    source_ids: list[str],
    exact_ratio_map: dict[tuple[int, tuple[str, ...]], np.ndarray],
    source_ratio_map: dict[tuple[int, str], float],
    mins: np.ndarray | None,
    maxs: np.ndarray | None,
) -> tuple[np.ndarray, bool]:
    """Restore one aggregation target vector from exact or per-source maps."""
    slot_count = min(target.shape[0], len(source_ids))
    if slot_count == 0:
        return target, False

    exact_key = (network_id, tuple(source_ids))
    exact_vals = exact_ratio_map.get(exact_key)
    if exact_vals is not None:
        restored_vals = np.asarray(exact_vals[:slot_count], dtype=target.dtype)
        if mins is not None and maxs is not None:
            restored_vals = np.clip(restored_vals, mins[:slot_count], maxs[:slot_count])
        target[:slot_count] = restored_vals
        return target, True

    restored = False
    for slot, source_id in enumerate(source_ids[:slot_count]):
        source_key = (network_id, str(source_id))
        if source_key not in source_ratio_map:
            continue
        value = source_ratio_map[source_key]
        if mins is not None and maxs is not None:
            value = float(np.clip(value, mins[slot], maxs[slot]))
        target[slot] = value
        restored = True

    # If this node was restored by per-source fallback (subset match after prune),
    # rescale surviving slots to max=1.0 so ratio semantics remain consistent
    # with pruning/introspection expectations.
    if restored:
        from biocomp.ratio_utils import normalize_ratios_for_pruning

        restored_slice = np.asarray(target[:slot_count], dtype=np.float32)
        restored_slice = np.asarray(normalize_ratios_for_pruning(restored_slice, threshold=0.0))
        if mins is not None and maxs is not None:
            restored_slice = np.clip(restored_slice, mins[:slot_count], maxs[:slot_count])
        target[:slot_count] = restored_slice.astype(target.dtype, copy=False)

    return target, restored


def _restore_aggregation_ratios(
    exact_ratio_map: dict[tuple[int, tuple[str, ...]], np.ndarray],
    source_ratio_map: dict[tuple[int, str], float],
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
            source_ids = source_ids_in_slot_order(graph_node.extra)
            if not source_ids:
                continue
            if rep_tgt is None:
                target = ratios_arr[node_idx]
                mins = ratio_min[node_idx] if ratio_min is not None else None
                maxs = ratio_max[node_idx] if ratio_max is not None else None
                target, did_restore = _restore_ratio_target(
                    target,
                    network_id=node.network_id,
                    source_ids=source_ids,
                    exact_ratio_map=exact_ratio_map,
                    source_ratio_map=source_ratio_map,
                    mins=mins,
                    maxs=maxs,
                )
                ratios_arr[node_idx] = target
            else:
                r, t = rep_tgt
                target = ratios_arr[r, t, node_idx]
                mins = ratio_min[r, t, node_idx] if ratio_min is not None else None
                maxs = ratio_max[r, t, node_idx] if ratio_max is not None else None
                target, did_restore = _restore_ratio_target(
                    target,
                    network_id=node.network_id,
                    source_ids=source_ids,
                    exact_ratio_map=exact_ratio_map,
                    source_ratio_map=source_ratio_map,
                    mins=mins,
                    maxs=maxs,
                )
                ratios_arr[r, t, node_idx] = target
            if did_restore:
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
    exact_ratio_map, source_ratio_map = _build_aggregation_ratio_maps(old_params, old_stack)
    bias_map = _build_bias_map(old_params, old_stack)
    rate_map = _build_rate_map(old_params, old_stack)

    restored_ratios = _restore_aggregation_ratios(
        exact_ratio_map, source_ratio_map, new_params, new_stack
    )
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
    keep_network_indices: list[int] | None = None,
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
                "keep_network_indices": keep_network_indices,
            },
        )
        scope.snapshot("params_before", jax.device_get(params))

        stack.ensure_tu_mapping(auto_lock_topology_tus=dconf.auto_lock_topology_tus)
        old_tu_id_to_idx = stack.tu_id_to_idx or {}
        old_n_tus = len(old_tu_id_to_idx)
        n_networks = len(stack.networks)

        _store_learned_ratio_inits(params, stack)

        def _commit_and_filter(
            removed_tus: dict[int, set[str]],
        ) -> tuple["ParameterTree", list[Network], list[tuple[int, Network]], int]:
            candidate_params = deepcopy(params)
            applied_count = _apply_hard_pruning_mask(
                candidate_params,
                stack,
                removed_tus,
                auto_lock_topology_tus=dconf.auto_lock_topology_tus,
            )
            candidate_networks, commit_report = commit_structure(
                stack, candidate_params, lock_ratios=lock_ratios
            )
            for cr in commit_report.commit_results:
                if cr.status.is_degenerate:
                    logger.debug(
                        f"[HARD-PRUNE] Degenerate commit: {cr.status.value} "
                        f"for {cr.diagnostics.get('network_name', '?')}"
                    )
            valid: list[tuple[int, Network]] = []
            for idx, candidate_net in enumerate(candidate_networks):
                if _is_valid_network(candidate_net):
                    valid.append((idx, candidate_net))
            return candidate_params, candidate_networks, valid, applied_count

        effective_tus_to_remove = {net_idx: set(tus) for net_idx, tus in tus_to_remove.items()}
        params_for_commit, committed_networks, valid_pairs, applied = _commit_and_filter(
            effective_tus_to_remove
        )

        if not valid_pairs:
            logger.warning(
                "[HARD-PRUNE] All networks degenerated after masking; backing off TU removals by ratio strength"
            )
            strength_rankings: dict[int, list[str]] = {}
            for net_idx, removed_tus in effective_tus_to_remove.items():
                _candidates, _all_tus, strengths = _collect_ratio_pruning_candidates(
                    params, stack, net_idx, ratio_threshold=float("inf")
                )
                strength_rankings[net_idx] = sorted(
                    removed_tus,
                    key=lambda tu_id: strengths.get(tu_id, 0.0),
                    reverse=True,
                )

            while not valid_pairs:
                progress = False
                for net_idx, ranking in strength_rankings.items():
                    if not ranking:
                        continue
                    tu_to_restore = ranking.pop(0)
                    if tu_to_restore in effective_tus_to_remove.get(net_idx, set()):
                        effective_tus_to_remove[net_idx].remove(tu_to_restore)
                        progress = True
                        scope.decision(
                            "preserve_tu",
                            outcome=False,
                            reason="degeneracy_backoff",
                            inputs={"network": net_idx, "tu_id": tu_to_restore},
                        )

                if not progress:
                    break

                params_for_commit, committed_networks, valid_pairs, applied = _commit_and_filter(
                    effective_tus_to_remove
                )

        # Filter degenerate networks while preserving original indices
        if valid_pairs:
            for i, net in enumerate(committed_networks):
                if not _is_valid_network(net):
                    scope.decision(
                        "filter_degenerate",
                        outcome="removed",
                        reason="zero_outputs",
                        inputs={"network_idx": i, "network_name": net.name},
                    )
                    logger.warning(
                        f"[HARD-PRUNE] Network {i} ({net.name}) became degenerate, removing"
                    )

        if not valid_pairs:
            raise RuntimeError(
                "All networks became degenerate after pruning. "
                "Try reducing L0 penalty, increasing preserve_minimum, or adjusting ratio_threshold."
            )

        n_after_degenerate = len(valid_pairs)
        if keep_network_indices is not None:
            requested = set(keep_network_indices)
            max_idx = len(committed_networks) - 1
            bad = sorted(i for i in requested if i < 0 or i > max_idx)
            if bad:
                raise ValueError(
                    f"keep_network_indices contains out-of-range indices (max={max_idx}): {bad[:5]}"
                )

            selected_pairs = [(idx, net) for idx, net in valid_pairs if idx in requested]
            if selected_pairs:
                valid_pairs = selected_pairs
            else:
                logger.warning(
                    "[HARD-PRUNE] Requested top-network selection had no non-degenerate survivors; "
                    "skipping network filtering for this cycle."
                )

        kept_old_network_indices = [idx for idx, _ in valid_pairs]
        valid_networks = [net for _, net in valid_pairs]

        requested_removed = sum(len(tus) for tus in tus_to_remove.values())
        total_removed = sum(len(tus) for tus in effective_tus_to_remove.values())
        logger.info(
            f"[HARD-PRUNE] Committed networks, requested {requested_removed} TUs, "
            f"effective removals {total_removed}, applied {applied}"
        )
        if n_after_degenerate < len(committed_networks):
            logger.info(
                f"[HARD-PRUNE] Filtered {len(committed_networks) - n_after_degenerate} "
                f"degenerate networks, {n_after_degenerate} remaining"
            )
        if keep_network_indices is not None:
            logger.info(
                f"[HARD-PRUNE] Top-network filter kept {len(valid_networks)} / "
                f"{len(committed_networks)} committed networks"
            )

        from .design_prune_controller import build_stack_from_dconf

        new_dmanager = DesignManager(
            targets=dmanager.targets,
            networks=valid_networks,
            sampling=dmanager.sampling,
            enable_tu_masking=dmanager.enable_tu_masking,
        )

        pkey, init_key = jax.random.split(key)
        new_stack = build_stack_from_dconf(new_dmanager, dconf, model, lock_ratios=lock_ratios)

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

        new_params, restore_stats = transfer_params_to_new_stack(
            params, stack, new_params, new_stack
        )

        if TU_LOG_ALPHA_PATH in params and TU_LOG_ALPHA_PATH in new_params and new_n_tus > 0:
            old_log_alpha = params[TU_LOG_ALPHA_PATH]
            if old_log_alpha.ndim == 4:
                old_2d = old_log_alpha[0, 0]
            elif old_log_alpha.ndim == 2:
                old_2d = old_log_alpha
            else:
                old_2d = old_log_alpha.reshape(n_networks, -1)
            assert old_2d.ndim == 2, f"Expected 2D TU log-alpha, got shape {old_2d.shape}"
            assert old_2d.shape[0] == len(committed_networks), (
                f"TU log-alpha network axis mismatch before remap: {old_2d.shape[0]} "
                f"vs committed networks {len(committed_networks)}"
            )

            row_idx = jnp.asarray(kept_old_network_indices, dtype=jnp.int32)
            old_2d = jnp.take(old_2d, row_idx, axis=0)
            assert old_2d.shape[0] == len(valid_networks), (
                f"TU log-alpha row selection mismatch: {old_2d.shape[0]} "
                f"vs kept networks {len(valid_networks)}"
            )

            remapped = _remap_tu_log_alpha(old_2d, old_tu_id_to_idx, new_tu_id_to_idx)
            assert remapped.shape[0] == len(valid_networks), (
                f"Remapped TU log-alpha network axis mismatch: {remapped.shape[0]} "
                f"vs kept networks {len(valid_networks)}"
            )

            new_log_alpha = new_params[TU_LOG_ALPHA_PATH]
            if new_log_alpha.ndim == 4:
                remapped_4d = jnp.tile(
                    remapped[None, None, :, :],
                    (dconf.n_replicates, new_dmanager.n_targets, 1, 1),
                )
                assert remapped_4d.shape == new_log_alpha.shape, (
                    f"TU log-alpha shape mismatch: remapped {remapped_4d.shape} "
                    f"vs new {new_log_alpha.shape}"
                )
                new_params.at(TU_LOG_ALPHA_PATH, remapped_4d, overwrite=True)
            elif new_log_alpha.ndim == 2:
                assert remapped.shape == new_log_alpha.shape, (
                    f"TU log-alpha shape mismatch: remapped {remapped.shape} "
                    f"vs new {new_log_alpha.shape}"
                )
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
    dispatch: LoggerDispatch | None = None,
    lock_ratios: bool = False,
) -> tuple["ParameterTree", list, StepHistorySnapshot, "DesignManager"]:
    """Design optimization with periodic hard-pruning."""
    from .design import start
    from .design_session import PhaseTimer as _PhaseTimer

    if dconf.n_replicates > 1:
        logger.info(
            f"[HARD-PRUNE] Flattening {dconf.n_replicates} replicates x "
            f"{len(dmanager.networks)} networks -> "
            f"{dconf.n_replicates * len(dmanager.networks)} flattened networks"
        )
        dmanager = _flatten_replicates_into_networks(dmanager, dconf.n_replicates)
        dconf = dconf.model_copy(update={"n_replicates": 1})

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
            "hard_pruning_top_percent": dconf.hard_pruning_top_percent,
            "hard_pruning_min_networks": dconf.hard_pruning_min_networks,
        },
    )

    current_dmanager = dmanager
    accumulated_loss_history: list = []
    final_step_history: StepHistorySnapshot | None = None
    segment_params: ParameterTree | None = None
    current_params: ParameterTree | None = None

    for segment_idx in range(n_segments):
        segment_start_step = segment_idx * steps_per_segment
        segment_end_step = min((segment_idx + 1) * steps_per_segment, total_steps)
        segment_steps = segment_end_step - segment_start_step
        segment_start_params = current_params
        segment_start_eval_loss: float | None = None
        segment_start_committed_loss: float | None = None
        best_synced_score_fn = None
        best_synced_initial_score: float | None = None

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
        segment_dmanager = current_dmanager

        # Final segment should refine the already-pruned topology with deterministic
        # structure. Keeping TU masking active here can reintroduce mask drift
        # without any further prune step to reconcile it.
        if (
            dconf.hard_pruning_disable_tu_masking_final_segment
            and segment_idx == n_segments - 1
            and segment_config.enable_tu_masking
        ):
            segment_config = segment_config.model_copy(
                update={
                    "tu_masking": segment_config.tu_masking.model_copy(
                        update={"mode": TUMaskingMode.NONE}
                    )
                }
            )
            if hasattr(current_dmanager, "model_copy"):
                segment_dmanager = current_dmanager.model_copy(update={"enable_tu_masking": False})
            else:
                segment_dmanager = deepcopy(current_dmanager)
                segment_dmanager.enable_tu_masking = False
            logger.info("[HARD-PRUNE] Final segment: TU masking disabled")

        logger.info(
            f"[HARD-PRUNE] Segment {segment_idx + 1}/{n_segments}: "
            f"steps {segment_start_step}-{segment_end_step}"
        )

        if segment_idx == n_segments - 1 and segment_start_params is not None:
            from .design_prune_controller import evaluate_segment_snapshot

            guard_key = jax.random.fold_in(loop_key, segment_idx + 3000)
            start_snap = evaluate_segment_snapshot(
                current_dmanager, dconf, model, segment_start_params, guard_key
            )
            segment_start_eval_loss = start_snap.mean_loss
            logger.info(
                "[HARD-PRUNE] Final-segment guard baseline loss: %.4f",
                segment_start_eval_loss,
            )
            if dconf.hard_pruning_commit_aware_final_guard:
                from .design_prune_controller import evaluate_committed_snapshot
                start_committed = evaluate_committed_snapshot(
                    current_dmanager,
                    dconf,
                    model,
                    segment_start_params,
                    guard_key,
                    lock_ratios=lock_ratios,
                )
                segment_start_committed_loss = start_committed.mean_loss
                logger.info(
                    "[HARD-PRUNE] Final-segment committed baseline loss: %.4f",
                    segment_start_committed_loss,
                )
                best_synced_initial_score = segment_start_committed_loss
                checkpoint_every = max(1, int(dconf.hard_pruning_commit_aware_selection_interval))

                def _committed_score_fn(params_at_step, _step_history, local_step):
                    if local_step % checkpoint_every != 0 and local_step != segment_steps:
                        return None
                    eval_step = segment_start_step + local_step
                    score_key = jax.random.fold_in(loop_key, 7000 + eval_step)
                    snap = evaluate_committed_snapshot(
                        current_dmanager,
                        dconf,
                        model,
                        params_at_step,
                        score_key,
                        lock_ratios=lock_ratios,
                    )
                    logger.info(
                        "[HARD-PRUNE] Final-segment committed checkpoint @ step %d: %.4f",
                        eval_step,
                        snap.mean_loss,
                    )
                    return snap.mean_loss

                best_synced_score_fn = _committed_score_fn

        segment_params, segment_loss_history, segment_step_history = start(
            segment_dmanager,
            segment_config,
            model,
            dispatch=dispatch,
            lock_ratios=lock_ratios,
            initial_params=current_params,
            initial_step=segment_start_step,
            select_best_synced_params=(segment_idx == n_segments - 1),
            best_synced_score_fn=best_synced_score_fn,
            best_synced_initial_score=best_synced_initial_score,
        )
        segment_step_history = StepHistorySnapshot.from_raw(segment_step_history)
        current_params = segment_params

        accumulated_loss_history.extend(segment_loss_history)

        if segment_idx == n_segments - 1 and segment_start_params is not None:
            from .design_prune_controller import evaluate_segment_snapshot

            guard_key = jax.random.fold_in(loop_key, segment_idx + 3000)
            end_snap = evaluate_segment_snapshot(
                current_dmanager, dconf, model, current_params, guard_key
            )
            segment_end_eval_loss = end_snap.mean_loss
            logger.info(
                "[HARD-PRUNE] Final-segment guard end loss: %.4f",
                segment_end_eval_loss,
            )

            should_restore = False
            if dconf.hard_pruning_commit_aware_final_guard:
                from .design_prune_controller import evaluate_committed_snapshot
                end_committed = evaluate_committed_snapshot(
                    current_dmanager,
                    dconf,
                    model,
                    current_params,
                    guard_key,
                    lock_ratios=lock_ratios,
                )
                segment_end_committed_loss = end_committed.mean_loss
                logger.info(
                    "[HARD-PRUNE] Final-segment committed end loss: %.4f",
                    segment_end_committed_loss,
                )
                if (
                    segment_start_committed_loss is not None
                    and segment_end_committed_loss > segment_start_committed_loss
                ):
                    logger.warning(
                        "[HARD-PRUNE] Final segment committed regression (%.4f -> %.4f); "
                        "restoring pre-segment parameters",
                        segment_start_committed_loss,
                        segment_end_committed_loss,
                    )
                    should_restore = True
            elif (
                segment_start_eval_loss is not None
                and segment_end_eval_loss > segment_start_eval_loss
            ):
                logger.warning(
                    "[HARD-PRUNE] Final segment regressed (%.4f -> %.4f); restoring "
                    "pre-segment parameters",
                    segment_start_eval_loss,
                    segment_end_eval_loss,
                )
                should_restore = True

            if should_restore:
                current_params = segment_start_params
                segment_params = segment_start_params

        # Ensure step-history params reflect the params that are actually carried
        # forward/returned (including post-segment restoration paths).
        segment_step_history_data = segment_step_history.to_dict()
        segment_step_history_data["latest_params"] = segment_params
        segment_step_history_data["params"] = segment_params
        segment_step_history = StepHistorySnapshot.from_raw(segment_step_history_data)
        final_step_history = segment_step_history

        if segment_idx < n_segments - 1:
            timer.start("prune", "[HARD-PRUNE] Identifying TUs to prune...")

            from .design_prune_controller import build_stack_from_dconf

            temp_stack = build_stack_from_dconf(
                current_dmanager, dconf, model, lock_ratios=lock_ratios
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
            keep_count = _compute_hard_pruning_network_keep_count(
                len(current_dmanager.networks),
                dconf.hard_pruning_top_percent,
                dconf.hard_pruning_min_networks,
            )
            use_top_network_selection = keep_count is not None and keep_count < len(
                current_dmanager.networks
            )
            keep_network_indices: list[int] | None = None
            loss_pre_mean: float | None = None
            if total_to_remove > 0 or use_top_network_selection:
                from .design_prune_controller import (
                    compare_snapshots,
                    evaluate_segment_snapshot,
                )

                with trace_scope("hard_prune_compare", component="prune") as prune_scope:
                    try:
                        compare_key = jax.random.fold_in(loop_key, segment_idx + 2000)
                        pre_snap = evaluate_segment_snapshot(
                            current_dmanager, dconf, model, segment_params, compare_key
                        )
                        prune_scope.snapshot("xraw", jax.device_get(pre_snap.xraw))
                        prune_scope.snapshot("yraw", jax.device_get(pre_snap.yraw))
                        prune_scope.snapshot("yhat_pre", jax.device_get(pre_snap.yhat))
                        prune_scope.snapshot("loss_pre", jax.device_get(pre_snap.loss))
                        loss_pre_mean = pre_snap.mean_loss
                        prune_scope.event(
                            "pre_prune",
                            "Pre-prune evaluation complete",
                            {
                                "mean_loss_pre": loss_pre_mean,
                                "n_networks": len(current_dmanager.networks),
                                "network_names": [n.name for n in current_dmanager.networks],
                            },
                        )
                        if use_top_network_selection:
                            assert keep_count is not None
                            if pre_snap.loss.shape[-1] != len(current_dmanager.networks):
                                raise AssertionError(
                                    "loss_pre last dimension must match number of networks, "
                                    f"got {pre_snap.loss.shape[-1]} vs {len(current_dmanager.networks)}"
                                )
                            keep_network_indices = _select_top_network_indices_from_losses(
                                np.asarray(pre_snap.loss),
                                keep_count,
                            )
                            selected_names = [
                                current_dmanager.networks[i].name for i in keep_network_indices
                            ]
                            logger.info(
                                "[HARD-PRUNE] Selecting top %d/%d networks for next segment",
                                len(keep_network_indices),
                                len(current_dmanager.networks),
                            )
                            prune_scope.event(
                                "top_network_selection",
                                "Selected top-performing networks",
                                {
                                    "keep_count": len(keep_network_indices),
                                    "total_networks": len(current_dmanager.networks),
                                    "keep_indices": keep_network_indices,
                                    "keep_network_names": selected_names,
                                },
                            )
                    except Exception as e:
                        prune_scope.event(
                            "pre_prune_error",
                            "Pre-prune evaluation failed",
                            {"error": str(e)},
                        )
                        raise

                n_networks_before = len(current_dmanager.networks)
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
                    keep_network_indices=keep_network_indices,
                )
                if total_to_remove > 0 and keep_network_indices is not None:
                    logger.info(
                        "[HARD-PRUNE] Removed %d TUs and kept top %d/%d networks, rebuilt stack",
                        total_to_remove,
                        len(current_dmanager.networks),
                        n_networks_before,
                    )
                elif total_to_remove > 0:
                    logger.info(f"[HARD-PRUNE] Removed {total_to_remove} TUs, rebuilt stack")
                elif keep_network_indices is not None:
                    logger.info(
                        "[HARD-PRUNE] Kept top %d/%d networks, rebuilt stack",
                        len(current_dmanager.networks),
                        n_networks_before,
                    )

                with trace_scope("hard_prune_compare", component="prune") as prune_scope:
                    try:
                        compare_key = jax.random.fold_in(loop_key, segment_idx + 2000)
                        post_snap = evaluate_segment_snapshot(
                            current_dmanager, dconf, model, current_params, compare_key
                        )
                        prune_scope.snapshot("xraw", jax.device_get(post_snap.xraw))
                        prune_scope.snapshot("yraw", jax.device_get(post_snap.yraw))
                        prune_scope.snapshot("yhat_post", jax.device_get(post_snap.yhat))
                        prune_scope.snapshot("loss_post", jax.device_get(post_snap.loss))
                        prune_scope.event(
                            "post_prune",
                            "Post-prune evaluation complete",
                            {
                                "mean_loss_post": post_snap.mean_loss,
                                "n_networks": len(current_dmanager.networks),
                                "network_names": [n.name for n in current_dmanager.networks],
                            },
                        )

                        if loss_pre_mean is not None:
                            comparison = compare_snapshots(pre_snap, post_snap)
                            if comparison["is_regression"]:
                                logger.warning(
                                    f"[HARD-PRUNE] Loss regression: "
                                    f"{pre_snap.mean_loss:.4f} -> {post_snap.mean_loss:.4f} "
                                    f"({comparison['increase_pct']:.1f}% increase)"
                                )
                                prune_scope.event(
                                    "loss_regression",
                                    "Significant loss increase after pruning",
                                    {
                                        **comparison,
                                        "tus_removed": total_to_remove,
                                        "top_network_selection": keep_network_indices is not None,
                                    },
                                )
                                prune_scope.snapshot(
                                    "current_params", jax.device_get(current_params)
                                )
                                prune_scope.snapshot("tus_to_remove", tus_to_remove)
                    except Exception as e:
                        prune_scope.event(
                            "post_prune_error",
                            "Post-prune evaluation failed",
                            {"error": str(e)},
                        )
            else:
                logger.info("[HARD-PRUNE] No pruning or top-network filtering applied")

    logger.info("=" * 60)
    logger.info(f"HARD-PRUNING OPTIMIZATION COMPLETE in {timer.total():.2f}s")
    logger.info("=" * 60)

    assert segment_params is not None, "No optimization segments were run"
    assert final_step_history is not None, (
        "No step history produced during hard-pruning optimization"
    )
    return segment_params, accumulated_loss_history, final_step_history, current_dmanager
