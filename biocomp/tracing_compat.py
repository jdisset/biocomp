# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Backward-compatibility shims for the legacy `designdebug` API.

These wrap the canonical trace primitives in `tracing_config`/`tracing_scope` so older
call sites (`is_design_debug_enabled`, `save_debug_state`, `get_debug_summary`) keep
working unchanged.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from biocomp.logging_config import get_logger
from biocomp import tracing_config as _tc
from biocomp.tracing_config import (
    _to_numpy,
    trace_run_event,
    trace_run_snapshot,
)

logger = get_logger(__name__)


def is_design_debug_enabled() -> bool:
    """Backward compatibility: check if design tracing is enabled.

    Replaces: from biocomp.designdebug import is_design_debug_enabled
    """
    return _tc._config.is_active("design")


def is_plot_debug_enabled() -> bool:
    """Backward compatibility: check if plot tracing is enabled.

    Replaces: from biocomp.designdebug import is_plot_debug_enabled
    """
    return _tc._config.is_active("plot")


def _compute_stats(arr: np.ndarray) -> dict[str, Any]:
    """Compute stats for an array (matching old designdebug behavior)."""
    arr = np.asarray(arr)
    if arr.size == 0:
        return {"size": 0}
    try:
        return {
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr)),
            "nan_count": int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0,
        }
    except Exception:
        return {"shape": arr.shape, "dtype": str(arr.dtype), "error": "stats_failed"}


def save_debug_state(
    stage: str,
    data: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    mode: str = "design",
    force: bool = False,
) -> Path | None:
    """Backward-compat shim (replaces `designdebug.save_debug_state`): saves debug pickle and emits canonical trace events."""
    if not force and not _tc._config.is_active(mode):
        return None

    # Determine output directory
    if output_dir is not None:
        effective_output_dir = Path(output_dir) / "_debug_dumps"
    elif _tc._config.output_dir is not None:
        effective_output_dir = _tc._config.output_dir / mode
    else:
        import os as _os

        effective_output_dir = (
            Path(_os.environ.get("BIOCOMP_ROOT", "/tmp")) / "debug_dumps" / f"{mode}_fallback"
        )

    effective_output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with counter
    existing = list(effective_output_dir.glob("*.pickle"))
    nums = []
    for f in existing:
        try:
            nums.append(int(f.name.split("_")[0]))
        except (ValueError, IndexError):
            pass
    counter = (max(nums) + 1) if nums else 1

    filepath = effective_output_dir / f"{counter:04d}_{stage}.pickle"

    # Convert to numpy for pickling
    data_np = _to_numpy(data)

    # Compute shapes and stats (matching old behavior)
    shapes: dict[str, Any] = {}
    stats: dict[str, Any] = {}
    for key, val in data_np.items():
        if val is None:
            continue
        if hasattr(val, "shape"):
            shapes[key] = val.shape
            stats[key] = _compute_stats(val)
        elif isinstance(val, list | tuple) and len(val) > 0 and hasattr(val[0], "shape"):
            shapes[key] = [v.shape for v in val]
            stats[key] = [_compute_stats(v) for v in val]

    payload = {
        "stage": stage,
        "counter": counter,
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "output_dir": str(output_dir) if output_dir else str(effective_output_dir.parent),
        "shapes": shapes,
        "stats": stats,
        "data": data_np,
        "metadata": metadata or {},
    }

    with open(filepath, "wb") as f:
        pickle.dump(payload, f)

    # Emit canonical replay records from the same payload (SSOT).
    try:
        trace_run_event(
            component=mode,
            scope="save_debug_state",
            event_type="debug_state",
            message=stage,
            data={
                "legacy_path": str(filepath),
                "metadata": metadata or {},
                "shapes": shapes,
                "stats": stats,
            },
        )
        trace_run_snapshot(
            component=mode,
            scope="save_debug_state",
            name=stage,
            payload=data_np,
            data={"metadata": metadata or {}, "legacy_path": str(filepath)},
        )
    except Exception as e:
        logger.warning(f"[DEBUG-{mode.upper()}] Failed canonical trace emission for {stage}: {e}")

    logger.debug(f"[DEBUG-{mode.upper()}] Saved {stage} to {filepath}")
    return filepath


def get_debug_summary(output_dir: str | Path | None = None) -> dict[str, Any]:
    """Backward compatibility: get debug state summary.

    Replaces: from biocomp.designdebug import get_debug_summary
    """
    design_enabled = is_design_debug_enabled()
    plot_enabled = is_plot_debug_enabled()

    if not design_enabled and not plot_enabled:
        return {"design_enabled": False, "plot_enabled": False, "save_count": 0}

    result: dict[str, Any] = {"design_enabled": design_enabled, "plot_enabled": plot_enabled}

    if output_dir is not None:
        debug_dir = Path(output_dir) / "_debug_dumps"
        if debug_dir.exists():
            files = sorted(debug_dir.glob("*.pickle"))
            result.update(
                {
                    "debug_dir": str(debug_dir),
                    "save_count": len(files),
                    "files": [f.name for f in files],
                }
            )
        else:
            result["save_count"] = 0

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Replay/Analysis Utilities
# ─────────────────────────────────────────────────────────────────────────────


def load_trace(path: Path | str) -> dict[str, Any]:
    """Load a trace pickle (keys: name, component, events, snapshots, duration)."""
    try:
        import dill
    except ImportError:
        import pickle as dill  # type: ignore[import-not-found]

    with open(path, "rb") as f:
        return dill.load(f)


def list_trace_runs(trace_root: Path | str) -> list[Path]:
    """List run-scoped trace directories that contain events.jsonl."""
    root = Path(trace_root)
    runs_dir = root / "runs"
    if not runs_dir.exists():
        return []
    runs = [p for p in runs_dir.iterdir() if p.is_dir() and (p / "events.jsonl").exists()]
    return sorted(runs, key=lambda p: p.stat().st_mtime)


def load_trace_run_events(run_dir: Path | str) -> list[dict[str, Any]]:
    """Load canonical replay events from a run trace directory."""
    path = Path(run_dir) / "events.jsonl"
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def list_traces(
    trace_dir: Path | str,
    component: str | None = None,
) -> list[Path]:
    """List trace pickle files (optionally filtered by component) sorted by mtime."""
    trace_dir = Path(trace_dir)
    if not trace_dir.exists():
        return []

    if component:
        search_dir = trace_dir / component
        if not search_dir.exists():
            return []
        files = list(search_dir.glob("*.pkl"))
    else:
        files = list(trace_dir.glob("**/*.pkl"))

    return sorted(files, key=lambda p: p.stat().st_mtime)


def get_decisions(trace: dict[str, Any]) -> list[Any]:
    """Extract decision events from a trace."""
    return [e for e in trace.get("events", []) if e.event_type.startswith("decision:")]


def format_decision_chain(trace: dict[str, Any]) -> str:
    """Format decisions from a trace as a readable multi-line chain."""
    decisions = get_decisions(trace)
    if not decisions:
        return "No decisions recorded"

    lines = [f"=== {trace['name']} ({trace['component']}) ==="]
    lines.append(f"Duration: {trace.get('duration', 0):.3f}s")
    lines.append(f"Total events: {len(trace.get('events', []))}")
    lines.append("")

    for d in decisions:
        decision_type = d.event_type.replace("decision:", "")
        inputs = d.data.get("inputs", {})
        outcome = d.data.get("outcome")
        reason = d.data.get("reason", "")
        lines.append(f"  {decision_type}: {outcome} (reason: {reason})")
        if inputs:
            for k, v in inputs.items():
                lines.append(f"    {k}: {v}")

    return "\n".join(lines)
