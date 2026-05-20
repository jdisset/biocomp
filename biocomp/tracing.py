# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""General decision tracing system for biocompiler (slim public facade).

This module is a re-export surface for the canonical implementation split into:
    - `tracing_config`  -- TraceConfig, TracingSettings, env loading, run context
    - `tracing_scope`   -- TraceScope, trace_scope, trace_here, decorators
    - `tracing_summary` -- summarize_*/snapshot_full_* helpers and Pydantic models
    - `tracing_compat`  -- backward-compat shims for the old designdebug API

Public API is unchanged; importers can keep using `from biocomp.tracing import X`.

Enable tracing via environment variables:
    BIOCOMP_TRACE=1                     # Enable all tracing
    BIOCOMP_TRACE_COMPONENTS=design,commit  # Enable specific components only
    BIOCOMP_TRACE_DIR=/tmp/traces       # Output directory for persisted traces
    BIOCOMP_TRACE_FULL=1                # Save full Network/Stack objects (not just summaries)

Or programmatically:
    from biocomp.tracing import configure_tracing
    configure_tracing(enabled=True, components={"design"}, output_dir=Path("/tmp/traces"))

Usage:
    from biocomp.tracing import trace_scope, trace_here, trace_decision

    with trace_scope("prune_tus", component="design") as scope:
        scope.event("start", "Beginning TU pruning", {"n_tus": 24})
        scope.snapshot("params_before", params)

Full Object Saving:
    When BIOCOMP_TRACE_FULL=1, snapshots include full serialized objects (Network,
    ComputeStack, ParameterTree) instead of just summaries. Reconstruct via:

        from biocomp.tracing import load_network_from_snapshot, load_networks_from_stack_snapshot
"""

from __future__ import annotations

from biocomp.tracing_config import (
    TRACE_RUN_SCHEMA_VERSION,
    TraceConfig,
    TraceRunContext,
    TracingSettings,
    close_trace_run,
    configure_trace_run,
    configure_tracing,
    get_trace_config,
    get_trace_run_context,
    is_tracing_active,
    should_save_full_objects,
    trace_run_event,
    trace_run_snapshot,
)
from biocomp.tracing_scope import (
    TraceEvent,
    TraceScope,
    _NullScope,
    get_current_scope,
    register_trace_point,
    trace_decision,
    trace_here,
    trace_scope,
)
from biocomp.tracing_summary import (
    GraphEdgeSummary,
    GraphNodeSummary,
    GraphSummary,
    LayerSummary,
    NetworkSummary,
    ParamsSummary,
    SourceNodeSummary,
    StackSummary,
    TUMappingSummary,
    load_network_from_snapshot,
    load_networks_from_stack_snapshot,
    serialize_graph,
    snapshot_full_network,
    snapshot_full_params,
    snapshot_full_stack,
    summarize_network,
    summarize_params,
    summarize_stack,
)
from biocomp.tracing_compat import (
    format_decision_chain,
    get_debug_summary,
    get_decisions,
    is_design_debug_enabled,
    is_plot_debug_enabled,
    list_trace_runs,
    list_traces,
    load_trace,
    load_trace_run_events,
    save_debug_state,
)

__all__ = [
    # config + run context
    "TRACE_RUN_SCHEMA_VERSION",
    "TraceConfig",
    "TraceRunContext",
    "TracingSettings",
    "close_trace_run",
    "configure_trace_run",
    "configure_tracing",
    "get_trace_config",
    "get_trace_run_context",
    "is_tracing_active",
    "should_save_full_objects",
    "trace_run_event",
    "trace_run_snapshot",
    # scope + decorators
    "TraceEvent",
    "TraceScope",
    "get_current_scope",
    "register_trace_point",
    "trace_decision",
    "trace_here",
    "trace_scope",
    # summaries + snapshots
    "GraphEdgeSummary",
    "GraphNodeSummary",
    "GraphSummary",
    "LayerSummary",
    "NetworkSummary",
    "ParamsSummary",
    "SourceNodeSummary",
    "StackSummary",
    "TUMappingSummary",
    "load_network_from_snapshot",
    "load_networks_from_stack_snapshot",
    "serialize_graph",
    "snapshot_full_network",
    "snapshot_full_params",
    "snapshot_full_stack",
    "summarize_network",
    "summarize_params",
    "summarize_stack",
    # backward-compat
    "format_decision_chain",
    "get_debug_summary",
    "get_decisions",
    "is_design_debug_enabled",
    "is_plot_debug_enabled",
    "list_trace_runs",
    "list_traces",
    "load_trace",
    "load_trace_run_events",
    "save_debug_state",
]
