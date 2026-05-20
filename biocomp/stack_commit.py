# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Commit helpers to separate structural pruning from quantization/collapse.

This module provides the core commit functionality extracted from ComputeStack.commit().
It handles TU pruning, network cleanup, and recipe roundtrip rebuilding.

Key abstractions:
- CommitOptions: Configuration for commit behavior
- prune_network_tus(): Unified TU pruning for a single network
- rebuild_network_from_recipe(): Recipe roundtrip for clean graph structure
- commit_networks(): Core orchestration function
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING
import time as _time

import jax.numpy as jnp

from biocomp.logging_config import get_logger
from biocomp.tumasking_strategy import get_full_log_alpha
from biocomp.tracing import trace_scope

if TYPE_CHECKING:
    from .compute import ComputeStack, StackLayer
    from .network import Network
    from .parameters import ParameterTree

logger = get_logger(__name__)


class CommitStatus(Enum):
    """Status of a single network's commit/rebuild."""

    OK = "ok"
    DEGENERATE_NO_OUTPUTS = "degenerate_no_outputs"
    DEGENERATE_RECIPE_ERROR = "degenerate_recipe_error"
    DEGENERATE_EMPTY_RECIPE = "degenerate_empty_recipe"
    DEGENERATE_NO_VALID_INVERSIONS = "degenerate_no_valid_inversions"

    @property
    def is_degenerate(self) -> bool:
        return self.name.startswith("DEGENERATE")

    @property
    def is_ok(self) -> bool:
        return self == CommitStatus.OK


@dataclass(frozen=True)
class CommitResult:
    """Result of rebuilding a single network during commit."""

    status: CommitStatus
    network: Network | None  # None for degenerate commits
    diagnostics: dict[str, object] = field(default_factory=dict)


def _make_empty_network(original: Network) -> Network:
    """Create an empty placeholder network preserving name and metadata."""
    from .graphengine import GraphState
    from .network import Network

    empty = Network(compute_graph=GraphState(nodes={}, edges={}))
    empty.name = original.name
    empty.metadata = original.metadata
    return empty


@dataclass(frozen=True)
class CommitOptions:
    """Configuration for commit behavior."""

    prune_tus: bool = True
    collapse_to_part: bool = True
    preserve_ratio_states: bool = False  # If True, keep ratio min/max/init metadata
    roundtrip_rebuild: bool = True
    # Structure-only pruning should only remove explicitly selected TUs.
    cascade_disable_exclusive_neg_tus: bool = True
    cleanup_orphaned_downstream_nodes: bool = True
    preserve_input_order: bool = True
    max_rebuild_workers: int = 8

    @classmethod
    def for_structure_only(cls) -> CommitOptions:
        """Commit structural changes only (pruning), no embedding collapse."""
        return cls(
            collapse_to_part=False,
            preserve_ratio_states=True,
            cascade_disable_exclusive_neg_tus=True,
            cleanup_orphaned_downstream_nodes=True,
        )

    @classmethod
    def for_final(cls) -> CommitOptions:
        """Full commit with embedding collapse/quantization."""
        return cls(collapse_to_part=True, preserve_ratio_states=False)


@dataclass
class NetworkCommitReport:
    """Report for a single network's commit process."""

    network_idx: int
    pruned_tu_count: int = 0
    dead_ern_recs: set[tuple[str, str]] = field(default_factory=set)
    cascade_disabled_tus: set[str] = field(default_factory=set)


@dataclass
class CommitReport:
    """Report for the entire commit process."""

    per_network: list[NetworkCommitReport] = field(default_factory=list)
    commit_results: list[CommitResult] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    def add_timing(self, name: str, duration: float) -> None:
        self.timings[name] = duration

    @property
    def has_degenerate(self) -> bool:
        return any(cr.status.is_degenerate for cr in self.commit_results)

    @property
    def degenerate_indices(self) -> list[int]:
        return [i for i, cr in enumerate(self.commit_results) if cr.status.is_degenerate]


def prune_network_tus(
    net: Network,
    net_idx: int,
    log_alpha: jnp.ndarray | None,
    tu_id_to_idx: dict[str, int] | None,
    *,
    cascade_disable_exclusive_neg_tus: bool = True,
    cleanup_orphaned_downstream_nodes: bool = True,
) -> NetworkCommitReport:
    """Unified TU pruning for a single network.

    Args:
        net: Network to prune (modified in place)
        net_idx: Network index in the stack
        log_alpha: Shape (n_networks, n_tus) log_alpha values, or None if no masking
        tu_id_to_idx: TU ID to index mapping, or None if no masking

    Returns:
        NetworkCommitReport with pruning statistics
    """
    from .tumasking import get_final_mask

    assert net.compute_graph is not None, f"Network {net_idx} has no compute_graph"
    report = NetworkCommitReport(network_idx=net_idx)

    if log_alpha is None or tu_id_to_idx is None:
        net.prune_disabled_tus()
        return report

    pseudo_log_alpha = log_alpha[net_idx]

    original_output_proteins = net.get_output_proteins()
    net.prune_disabled_tus(pseudo_log_alpha, tu_id_to_idx)

    edges_to_remove = []
    for edge_id, edge in net.compute_graph.edges.items():
        tu_ids = edge.extra.get("tu_id", []) if edge.extra else []
        if not tu_ids:
            continue
        all_disabled = True
        for tu_id in tu_ids:
            if tu_id not in tu_id_to_idx:
                all_disabled = False
                break
            tu_idx = tu_id_to_idx[tu_id]
            assert 0 <= tu_idx < pseudo_log_alpha.shape[0], (
                f"tu_idx {tu_idx} out of bounds for mask shape {pseudo_log_alpha.shape}"
            )
            mask = get_final_mask(pseudo_log_alpha[tu_idx : tu_idx + 1])[0]
            if float(mask) > 0:
                all_disabled = False
                break
        if all_disabled and tu_ids:
            edges_to_remove.append(edge_id)
            report.pruned_tu_count += 1

    for edge_id in edges_to_remove:
        del net.compute_graph.edges[edge_id]

    dead_ern_recs = net._cleanup_ern_nodes(
        pseudo_log_alpha,
        tu_id_to_idx,
        cascade_disable_exclusive_neg_tus=cascade_disable_exclusive_neg_tus,
    )
    report.dead_ern_recs = dead_ern_recs

    if cascade_disable_exclusive_neg_tus:
        additional_disabled = net.metadata.pop("_additional_disabled_tus", set())
    else:
        net.metadata.pop("_additional_disabled_tus", None)
        additional_disabled = set()
    report.cascade_disabled_tus = additional_disabled

    if additional_disabled:
        for edge_id, edge in list(net.compute_graph.edges.items()):
            tu_ids = edge.extra.get("tu_id", []) if edge.extra else []
            if any(tu_id in additional_disabled for tu_id in tu_ids):
                del net.compute_graph.edges[edge_id]

        disabled_source_ids = set()
        disabled_source_keys: set[tuple[str, str]] = set()
        for _tu_id in additional_disabled:
            for node in net.compute_graph.get_nodes_by_type("source"):
                tu_names = node.extra.get("tu_names_by_slot", {})
                cotx_group = node.extra.get("cotx_group", "")
                for tu_name in tu_names.values():
                    full_tu_id = f"{tu_name}_{cotx_group}" if cotx_group else tu_name
                    if full_tu_id in additional_disabled:
                        disabled_source_ids.add(node.node_id)
                        source_id = node.extra.get("source_id")
                        if source_id is not None:
                            disabled_source_keys.add((str(source_id), str(cotx_group)))
                        break

        for edge_id, edge in list(net.compute_graph.edges.items()):
            if edge.source_id in disabled_source_ids or edge.target_id in disabled_source_ids:
                del net.compute_graph.edges[edge_id]
        for node_id in disabled_source_ids:
            if node_id in net.compute_graph.nodes:
                del net.compute_graph.nodes[node_id]

        _renormalize_aggregation_after_cascade(net, disabled_source_keys)

    if cleanup_orphaned_downstream_nodes:
        net._cleanup_orphaned_downstream_nodes()
    net._cleanup_orphaned_bias_nodes()
    net._cleanup_orphaned_input_nodes(original_output_proteins)
    net._cleanup_orphaned_transcription_nodes()

    return report


def _renormalize_aggregation_after_cascade(
    net: Network,
    disabled_source_keys: set[tuple[str, str]],
) -> None:
    """Renormalize aggregation ratios after cascade-disabled sources are removed."""
    if not disabled_source_keys:
        return
    if net.compute_graph is None:
        return

    from biocomp.ratio_schema import remove_sources_and_renormalize

    graph = net.compute_graph
    for node in graph.get_nodes_by_type("aggregation"):
        if node.extra is None:
            continue
        cotx_group = str(node.extra.get("cotx_group", ""))
        removed_source_ids = {
            source_id
            for source_id, source_group in disabled_source_keys
            if source_group == cotx_group
        }
        if not removed_source_ids:
            continue
        remove_sources_and_renormalize(node.extra, removed_source_ids)


def rebuild_network_from_recipe(
    net: Network,
    net_idx: int,
    strip_ern_recs: set[tuple[str, str]],
    options: CommitOptions,
) -> CommitResult:
    """Rebuild network via recipe roundtrip for clean graph structure.

    This ensures orphan nodes are removed and the graph is consistent.
    Returns a CommitResult with explicit status for each outcome path,
    making degenerate commits observable rather than silent.

    Args:
        net: Network to rebuild
        net_idx: Network index (for logging)
        strip_ern_recs: ERN records to strip from recipe
        options: Commit options

    Returns:
        CommitResult with status, network (None if degenerate), and diagnostics.
    """
    from .network import recipe_to_networks
    import biocomp.biorules as br

    base_diag: dict[str, object] = {"network_name": net.name, "network_idx": net_idx}

    assert net.compute_graph is not None, f"Network {net_idx} has no compute_graph"

    try:
        original_input_proteins = net.get_inverted_input_proteins()
    except (AssertionError, IndexError, KeyError):
        original_input_proteins = None
    original_input_axes = net.get_input_axes()

    output_nodes = [n for n in net.compute_graph.nodes.values() if n.node_type == "output"]
    if len(output_nodes) != 1:
        original_outputs = ()
    else:
        try:
            original_outputs = tuple(sorted(net.get_dependent_output_proteins()))
        except (AssertionError, IndexError, KeyError):
            original_outputs = ()

    if len(original_outputs) == 0:
        return CommitResult(
            status=CommitStatus.DEGENERATE_NO_OUTPUTS,
            network=None,
            diagnostics=base_diag,
        )

    try:
        recipe = net.to_recipe(strip_ern_recs=strip_ern_recs)
        recipe.strip_orphan_ern_proteins()
    except (AssertionError, IndexError, KeyError) as exc:
        return CommitResult(
            status=CommitStatus.DEGENERATE_RECIPE_ERROR,
            network=None,
            diagnostics={**base_diag, "error": str(exc)},
        )

    if not recipe.content:
        return CommitResult(
            status=CommitStatus.DEGENERATE_EMPTY_RECIPE,
            network=None,
            diagnostics=base_diag,
        )

    rebuilt = recipe_to_networks(
        recipe,
        br.ALL_RULES,
        invert=True,
        inversion_mode="all",
        skip_input_order_validation=True,
    )

    if len(rebuilt) == 0:
        logger.warning(
            f"COMMIT: Recipe '{recipe.name}' produced no valid networks "
            f"(all inversions degenerate). Returning empty network."
        )
        return CommitResult(
            status=CommitStatus.DEGENERATE_NO_VALID_INVERSIONS,
            network=None,
            diagnostics=base_diag,
        )

    if len(rebuilt) == 1:
        rebuilt_net = rebuilt[0]
    else:
        matching = [
            r
            for r in rebuilt
            if tuple(sorted(r.get_dependent_output_proteins())) == original_outputs
        ]
        assert len(matching) == 1, (
            f"COMMIT ERROR: Recipe '{recipe.name}' produced {len(rebuilt)} inversions, "
            f"but {len(matching)} match original outputs {original_outputs}. "
            f"Rebuilt: {[tuple(sorted(r.get_dependent_output_proteins())) for r in rebuilt]}"
        )
        rebuilt_net = matching[0]

    try:
        rebuilt_dep_outputs = tuple(sorted(rebuilt_net.get_dependent_output_proteins()))
    except (AssertionError, KeyError):
        rebuilt_dep_outputs = ()

    if rebuilt_dep_outputs != original_outputs:
        logger.warning(
            f"COMMIT: Recipe roundtrip changed dependent outputs from {original_outputs} "
            f"to {rebuilt_dep_outputs}. Marker TUs may have been pruned. "
            f"Using rebuilt network for consistency with inference."
        )

    original_nb_inputs = net.nb_inputs
    rebuilt_nb_inputs = rebuilt_net.nb_inputs
    if rebuilt_nb_inputs != original_nb_inputs:
        logger.warning(
            f"COMMIT: Recipe roundtrip changed nb_inputs from {original_nb_inputs} "
            f"to {rebuilt_nb_inputs}. Inverted inputs may have been lost. "
            f"Using rebuilt network (pruned network may have broken structure)."
        )

    rebuilt_net.name = net.name
    rebuilt_net.metadata = net.metadata

    if options.preserve_input_order and original_input_proteins is not None:
        try:
            rebuilt_input_proteins = rebuilt_net.get_inverted_input_proteins()
            if set(rebuilt_input_proteins) == set(original_input_proteins):
                if original_input_axes is not None:
                    rebuilt_net.apply_input_axes(original_input_axes)
                else:
                    rebuilt_net.apply_input_order(original_input_proteins)
                logger.debug(
                    f"COMMIT: Restored input ordering {original_input_proteins} "
                    f"(was {rebuilt_input_proteins})"
                )
        except (AssertionError, IndexError, KeyError) as e:
            logger.debug(f"COMMIT: Could not restore input ordering: {e}")

    return CommitResult(status=CommitStatus.OK, network=rebuilt_net)


def commit_networks(
    networks: list[Network],
    layers: list[StackLayer],
    params: ParameterTree,
    options: CommitOptions,
    tu_id_to_idx: dict[str, int] | None = None,
    node_map: dict[tuple[int, int], tuple[int, int]] | None = None,
) -> tuple[list[Network], CommitReport]:
    """Core commit implementation.

    This is the main function that ComputeStack.commit() delegates to.

    Args:
        networks: List of networks to commit
        layers: Stack layers for node-level commits
        params: Parameter tree with trained parameters
        options: Commit configuration
        tu_id_to_idx: TU ID to index mapping (for TU masking)
        node_map: Node map for stack operations

    Returns:
        Tuple of (committed_networks, report)
    """
    from .compute import ComputeStack

    with trace_scope("commit_networks", component="commit") as scope:
        scope.event(
            "start",
            "Starting network commit",
            {
                "n_networks": len(networks),
                "n_layers": len(layers),
                "prune_tus": options.prune_tus,
                "collapse_to_part": options.collapse_to_part,
                "roundtrip_rebuild": options.roundtrip_rebuild,
                "cascade_disable_exclusive_neg_tus": options.cascade_disable_exclusive_neg_tus,
                "cleanup_orphaned_downstream_nodes": options.cleanup_orphaned_downstream_nodes,
            },
        )

        report = CommitReport()
        t0 = _time.perf_counter()

        t1 = _time.perf_counter()
        network_copies = [deepcopy(net) for net in networks]
        t2 = _time.perf_counter()
        report.add_timing("deepcopy", t2 - t1)
        logger.debug(f"COMMIT TIMING: deepcopy {len(network_copies)} networks: {t2 - t1:.3f}s")

        temp_stack = ComputeStack(network_copies, layers)
        temp_stack.node_map = node_map

        t3 = _time.perf_counter()
        for layer in layers:
            layer.commit(
                params,
                stack=temp_stack,
                collapse_to_part=options.collapse_to_part,
                preserve_ratio_states=options.preserve_ratio_states,
            )
        t4 = _time.perf_counter()
        report.add_timing("layer_commits", t4 - t3)
        logger.debug(f"COMMIT TIMING: layer commits ({len(layers)} layers): {t4 - t3:.3f}s")

        t5 = _time.perf_counter()
        log_alpha = get_full_log_alpha(params)
        dead_ern_recs_by_net: dict[int, set[tuple[str, str]]] = {}

        if options.prune_tus:
            for net_idx, net in enumerate(network_copies):
                net_report = prune_network_tus(
                    net,
                    net_idx,
                    log_alpha,
                    tu_id_to_idx,
                    cascade_disable_exclusive_neg_tus=options.cascade_disable_exclusive_neg_tus,
                    cleanup_orphaned_downstream_nodes=options.cleanup_orphaned_downstream_nodes,
                )
                dead_ern_recs_by_net[net_idx] = net_report.dead_ern_recs
                report.per_network.append(net_report)

                # Log pruning decisions
                if net_report.pruned_tu_count > 0:
                    scope.decision(
                        "prune_tus",
                        outcome=net_report.pruned_tu_count,
                        reason="tu_mask_disabled",
                        inputs={
                            "network_idx": net_idx,
                            "network_name": net.name,
                            "dead_ern_recs": len(net_report.dead_ern_recs),
                            "cascade_disabled": len(net_report.cascade_disabled_tus),
                        },
                    )

        t6 = _time.perf_counter()
        report.add_timing("tu_pruning", t6 - t5)
        logger.debug(f"COMMIT TIMING: TU pruning: {t6 - t5:.3f}s")

        if options.roundtrip_rebuild:

            def _rebuild(net_idx_and_net: tuple[int, Network]) -> CommitResult:
                net_idx, net = net_idx_and_net
                strip_ern_recs = dead_ern_recs_by_net.get(net_idx, set())
                return rebuild_network_from_recipe(net, net_idx, strip_ern_recs, options)

            n_workers = min(len(network_copies), options.max_rebuild_workers)
            indexed_networks = list(enumerate(network_copies))

            if n_workers > 1:
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    commit_results = list(executor.map(_rebuild, indexed_networks))
            else:
                commit_results = [_rebuild(idx_net) for idx_net in indexed_networks]

            report.commit_results = commit_results

            for cr in commit_results:
                scope.event("commit_status", cr.status.value, cr.diagnostics)

            final_networks = [
                cr.network if cr.network is not None else _make_empty_network(network_copies[i])
                for i, cr in enumerate(commit_results)
            ]
        else:
            # No roundtrip rebuild -- all networks are OK by definition
            report.commit_results = [
                CommitResult(status=CommitStatus.OK, network=net) for net in network_copies
            ]
            final_networks = network_copies

        t7 = _time.perf_counter()
        report.add_timing("roundtrip_rebuild", t7 - t6)
        logger.debug(
            f"COMMIT TIMING: roundtrip rebuild ({len(network_copies)} nets): {t7 - t6:.3f}s"
        )

        report.add_timing("total", t7 - t0)
        logger.debug(f"COMMIT TIMING: TOTAL: {t7 - t0:.3f}s")

        scope.event(
            "complete",
            "Network commit complete",
            {
                "n_networks_out": len(final_networks),
                "total_time": t7 - t0,
                "total_pruned": sum(r.pruned_tu_count for r in report.per_network),
                "degenerate_count": len(report.degenerate_indices),
            },
        )

        return final_networks, report


def commit_structure(
    stack: ComputeStack,
    params: ParameterTree,
    lock_ratios: bool = False,  # Deprecated name, kept for compatibility
    **kwargs,
) -> tuple[list[Network], CommitReport]:
    """Commit only structural changes (pruning, graph cleanup) without collapsing embeddings."""
    assert stack.layers is not None, "Stack must be built before committing"
    options = CommitOptions.for_structure_only()
    if lock_ratios:
        from dataclasses import replace

        # lock_ratios=True means DON'T preserve ratio states
        options = replace(options, preserve_ratio_states=False)

    return commit_networks(
        stack.networks,
        stack.layers,
        params,
        options,
        tu_id_to_idx=getattr(stack, "tu_id_to_idx", None),
        node_map=stack.node_map,
    )


def commit_final(
    stack: ComputeStack,
    params: ParameterTree,
    **kwargs,
) -> tuple[list[Network], CommitReport]:
    """Full commit including collapse/quantization to discrete parts."""
    assert stack.layers is not None, "Stack must be built before committing"
    options = CommitOptions.for_final()
    return commit_networks(
        stack.networks,
        stack.layers,
        params,
        options,
        tu_id_to_idx=getattr(stack, "tu_id_to_idx", None),
        node_map=stack.node_map,
    )
