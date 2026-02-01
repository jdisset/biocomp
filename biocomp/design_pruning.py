from __future__ import annotations

from typing import Callable, TYPE_CHECKING
from copy import deepcopy

import numpy as np
import jax
import jax.numpy as jnp

from .logging_config import get_logger
from .tumasking import TU_LOG_ALPHA_PATH

if TYPE_CHECKING:
    from .design import DesignManager, DesignConfig
    from .compute import ComputeStack
    from .parameters import ParameterTree
    from biocomptools.modelmodel import BiocompModel

logger = get_logger(__name__)


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
    from biocomp.nodes.aggregation import _decode_latent_ratios

    latent_z_path = f"{namespace}/latent_z"
    if latent_z_path in params:
        z = np.asarray(params[latent_z_path][node_idx])
        W1 = np.asarray(params[f"{namespace}/latent_W1"][node_idx])
        b1 = np.asarray(params[f"{namespace}/latent_b1"][node_idx])
        W2 = np.asarray(params[f"{namespace}/latent_W2"][node_idx])
        b2 = np.asarray(params[f"{namespace}/latent_b2"][node_idx])
        raw_ratios = _decode_latent_ratios(z, W1, b1, W2, b2)[:n_outputs]
        ratio_min = np.asarray(params[f"{namespace}/ratio_min"][node_idx][:n_outputs])
        ratio_max = np.asarray(params[f"{namespace}/ratio_max"][node_idx][:n_outputs])
        ratios = np.clip(raw_ratios, ratio_min, ratio_max)
    else:
        ratios = np.asarray(params[f"{namespace}/ratios"][node_idx][:n_outputs])
        ratio_min = np.asarray(params[f"{namespace}/ratio_min"][node_idx][:n_outputs])
        ratio_max = np.asarray(params[f"{namespace}/ratio_max"][node_idx][:n_outputs])
        ratios = np.clip(ratios, ratio_min, ratio_max)
    return ratios


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
    auto_lock_topology_tus: bool = True,
) -> dict[int, set[str]]:
    """Identify TUs to remove for each network based on normalized ratios."""
    from biocomp.tumasking import get_log_alpha_from_params, LATENT_TU_Z_PATH

    tus_to_remove: dict[int, set[str]] = {}
    stack.ensure_tu_mapping(auto_lock_topology_tus=auto_lock_topology_tus)
    no_masking_tu_ids = stack.no_masking_tu_ids or set()
    tu_id_to_idx = stack.tu_id_to_idx or {}

    has_tu_masking = TU_LOG_ALPHA_PATH in params or LATENT_TU_Z_PATH in params

    for net_idx in range(len(stack.networks)):
        candidates, all_tu_ids, tu_strengths = _collect_ratio_pruning_candidates(
            params, stack, net_idx, ratio_threshold
        )
        candidates = {tid for tid in candidates if tid not in no_masking_tu_ids}

        if use_soft_pruning and has_tu_masking:
            try:
                network_log_alpha = get_log_alpha_from_params(params, net_idx)
                if network_log_alpha.ndim > 1:
                    network_log_alpha = network_log_alpha.reshape(-1)

                for tu_id in all_tu_ids:
                    if tu_id in no_masking_tu_ids:
                        continue
                    if tu_id in tu_id_to_idx:
                        tu_idx = tu_id_to_idx[tu_id]
                        if tu_idx < len(network_log_alpha):
                            prob = float(jax.nn.sigmoid(network_log_alpha[tu_idx]))
                            tu_strengths[tu_id] = max(tu_strengths.get(tu_id, 0.0), prob)
                            if prob < 0.5:
                                candidates.add(tu_id)
            except ValueError:
                pass  # No TU masking params - skip soft pruning

        remaining = len(all_tu_ids) - len(candidates)
        if remaining < preserve_minimum:
            n_to_keep = preserve_minimum - remaining
            sorted_by_strength = sorted(candidates, key=lambda x: tu_strengths.get(x, 0.0))
            strongest_to_keep = set(sorted_by_strength[-n_to_keep:]) if n_to_keep > 0 else set()
            candidates = candidates - strongest_to_keep

        tus_to_remove[net_idx] = candidates

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

    assert not missing, (
        f"Hard-prune requested unknown TU IDs (sample): {sorted(list(missing))[:5]}"
    )

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

        tag_names = None
        if new_params.tags is not None:
            try:
                tag_flags = new_params.tags[path_str]
            except KeyError:
                tag_flags = None
            if tag_flags is not None:
                tag_names = [
                    name
                    for name, flag in zip(new_params.tagnames, tag_flags)
                    if flag
                ]

        new_params.at(path_str, old_val, overwrite=True, tags=tag_names)

    return new_params


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

    stack.ensure_tu_mapping(auto_lock_topology_tus=dconf.auto_lock_topology_tus)
    old_tu_id_to_idx = stack.tu_id_to_idx or {}
    old_n_tus = len(old_tu_id_to_idx)
    n_networks = len(stack.networks)

    params_for_commit = deepcopy(params)

    applied = _apply_hard_pruning_mask(
        params_for_commit,
        stack,
        tus_to_remove,
        auto_lock_topology_tus=dconf.auto_lock_topology_tus,
    )

    if TU_LOG_ALPHA_PATH in params_for_commit and old_tu_id_to_idx:
        tu_log_alpha = params_for_commit[TU_LOG_ALPHA_PATH]
        modified_log_alpha = jnp.array(tu_log_alpha)

        for net_idx, tu_ids in tus_to_remove.items():
            for tu_id in tu_ids:
                if tu_id in old_tu_id_to_idx:
                    tu_idx = old_tu_id_to_idx[tu_id]
                    modified_log_alpha = modified_log_alpha.at[net_idx, tu_idx].set(-10.0)

        params_for_commit.at(TU_LOG_ALPHA_PATH, modified_log_alpha, overwrite=True)

    committed_networks = commit_structure(stack, params_for_commit, lock_ratios=lock_ratios)

    total_removed = sum(len(tus) for tus in tus_to_remove.values())
    logger.info(
        f"[HARD-PRUNE] Committed networks, requested {total_removed} TUs, applied {applied}"
    )

    new_dmanager = DesignManager(
        targets=dmanager.targets,
        networks=committed_networks,
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
        n_tus=new_n_tus,
        n_networks=len(new_dmanager.networks),
        tu_log_alpha_init_mean=dconf.tu_log_alpha_init_mean,
        tu_log_alpha_init_std=dconf.tu_log_alpha_init_std,
        use_latent_tu_masking=dconf.use_latent_tu_masking,
        latent_tu_dim=dconf.latent_tu_dim,
        latent_tu_hidden_dim=dconf.latent_tu_hidden_dim,
        no_masking_tu_ids=new_stack.no_masking_tu_ids,
        tu_id_to_idx=new_stack.tu_id_to_idx,
    )

    expanded_old_params = jax.tree.map(
        lambda x: x[None, None, ...] if x.ndim >= 1 else x, params
    )
    new_params = _merge_surviving_params(expanded_old_params, new_params)

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
                remapped[None, None, :, :], (dconf.n_replicates, new_dmanager.n_targets, 1, 1)
            )
            new_params.at(TU_LOG_ALPHA_PATH, remapped_4d, overwrite=True)
        elif new_log_alpha.ndim == 2:
            new_params.at(TU_LOG_ALPHA_PATH, remapped, overwrite=True)

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
        f"[HARD-PRUNE] Total steps: {total_steps}, interval: {steps_per_segment}, segments: {n_segments}"
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
        segment_config = dconf.model_copy(update={
            "n_epochs": segment_epochs,
            "n_batches_per_epoch": segment_batches_per_epoch,
            "hard_pruning_enabled": False,
            "pluggable_optimizer": None,
            "seed": new_seed,
        })

        logger.info(
            f"[HARD-PRUNE] Segment {segment_idx + 1}/{n_segments}: steps {segment_start_step}-{segment_end_step}"
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
                auto_lock_topology_tus=dconf.auto_lock_topology_tus,
            )

            timer.end("prune")

            total_to_remove = sum(len(tus) for tus in tus_to_remove.values())
            if total_to_remove > 0:
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
            else:
                logger.info("[HARD-PRUNE] No TUs to remove, continuing with current structure")

    logger.info("=" * 60)
    logger.info(f"HARD-PRUNING OPTIMIZATION COMPLETE in {timer.total():.2f}s")
    logger.info("=" * 60)

    assert segment_params is not None, "No optimization segments were run"
    return segment_params, accumulated_loss_history, accumulated_step_history
