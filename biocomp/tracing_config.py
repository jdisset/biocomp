# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Tracing configuration: `TraceConfig`, `TracingSettings`, env loading, run context.

Holds the process-global `_config` and run context (`_run_context`). All other tracing
modules read state from here so there's a single canonical instance.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, model_validator

from biocomp.logging_config import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TraceConfig:
    """Global trace configuration. Check at trace points for zero-cost disable."""

    enabled: bool = False
    components: frozenset[str] = field(default_factory=frozenset)  # empty = all enabled
    output_dir: Path | None = None
    save_full_objects: bool = False  # If True, save full Network/Stack objects (not just summaries)

    @classmethod
    def from_env(cls) -> "TraceConfig":
        """Load from BIOCOMP_TRACE_* environment variables."""
        enabled = os.getenv("BIOCOMP_TRACE", "0").lower() in ("1", "true", "yes", "on")
        components_str = os.getenv("BIOCOMP_TRACE_COMPONENTS", "")
        components = frozenset(c.strip() for c in components_str.split(",") if c.strip())
        output_dir_str = os.getenv("BIOCOMP_TRACE_DIR")
        output_dir = Path(output_dir_str) if output_dir_str else None
        save_full = os.getenv("BIOCOMP_TRACE_FULL", "0").lower() in ("1", "true", "yes", "on")
        return cls(enabled=enabled, components=components, output_dir=output_dir, save_full_objects=save_full)

    def is_active(self, component: str) -> bool:
        """Check if tracing is enabled for a specific component."""
        return self.enabled and (not self.components or component in self.components)


class TracingSettings(BaseModel):
    """Pydantic model for tracing config in YAML files."""

    enabled: bool = False
    components: list[str] | None = None
    output_dir: str | None = None
    save_full_objects: bool = False

    @model_validator(mode="after")
    def _apply(self) -> "TracingSettings":
        if self.enabled:
            configure_tracing(
                enabled=True,
                components=set(self.components) if self.components else None,
                output_dir=Path(self.output_dir) if self.output_dir else None,
                save_full_objects=self.save_full_objects,
            )
        return self


# Global instance, loaded at import
_config = TraceConfig.from_env()


def configure_tracing(
    enabled: bool = True,
    components: set[str] | None = None,
    output_dir: Path | str | None = None,
    save_full_objects: bool = False,
) -> None:
    """Programmatic tracing config (for tests, notebooks). Empty/None `components` = all."""
    global _config
    _config = TraceConfig(
        enabled=enabled,
        components=frozenset(components) if components else frozenset(),
        output_dir=Path(output_dir) if output_dir else None,
        save_full_objects=save_full_objects,
    )
    if enabled:
        logger.debug(
            f"Tracing configured: components={components or 'all'}, "
            f"output_dir={output_dir or 'None (no persistence)'}, "
            f"save_full_objects={save_full_objects}"
        )


def is_tracing_active(component: str) -> bool:
    """Check if tracing is active for a given component. Zero-cost when disabled."""
    return _config.is_active(component)


def get_trace_config() -> TraceConfig:
    """Get the current trace configuration."""
    return _config


def should_save_full_objects() -> bool:
    """Check if full object saving is enabled."""
    return _config.save_full_objects


# ─────────────────────────────────────────────────────────────────────────────
# Run context (canonical replay sink)
# ─────────────────────────────────────────────────────────────────────────────


TRACE_RUN_SCHEMA_VERSION = 1


@dataclass
class TraceRunContext:
    """Canonical run-scoped trace sink for replay."""

    run_id: str
    root_dir: Path
    run_dir: Path
    events_path: Path
    snapshots_dir: Path
    manifest_path: Path
    seq: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


_run_context: TraceRunContext | None = None


def get_trace_run_context() -> TraceRunContext | None:
    return _run_context


def _now_iso_utc() -> str:
    return datetime.now(UTC).isoformat()


def _sanitize_name(name: str) -> str:
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)
    return safe.strip("._") or "item"


def _json_safe(val: Any) -> Any:
    """Make values JSON-serializable without exploding payload size."""
    if val is None or isinstance(val, str | int | float | bool):
        return val
    if isinstance(val, list | tuple):
        return [_json_safe(v) for v in val]
    if isinstance(val, dict):
        return {str(k): _json_safe(v) for k, v in val.items()}
    if isinstance(val, np.ndarray):
        return {
            "__ndarray__": True,
            "shape": list(val.shape),
            "dtype": str(val.dtype),
            "min": float(np.nanmin(val)) if val.size else None,
            "max": float(np.nanmax(val)) if val.size else None,
            "mean": float(np.nanmean(val)) if val.size else None,
        }
    if hasattr(val, "__array__"):
        arr = np.asarray(val)
        return _json_safe(arr)
    return repr(val)


def _to_numpy(val: Any) -> Any:
    """Convert JAX arrays and similar to numpy for pickling."""
    if hasattr(val, "__array__"):
        return np.asarray(val)
    if isinstance(val, dict):
        return {k: _to_numpy(v) for k, v in val.items()}
    if isinstance(val, list | tuple):
        return type(val)(_to_numpy(v) for v in val)
    return val


def configure_trace_run(
    run_id: str | None = None,
    manifest: dict[str, Any] | None = None,
    output_dir: Path | str | None = None,
) -> str | None:
    """Initialize/replace the active run-scoped trace sink."""
    global _run_context
    if not _config.enabled and output_dir is None:
        return None

    root = Path(output_dir) if output_dir is not None else _config.output_dir
    if root is None:
        return None

    actual_run_id = run_id or f"run_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:8]}"
    runs_root = root / "runs"
    run_dir = runs_root / _sanitize_name(actual_run_id)
    snapshots_dir = run_dir / "snapshots"
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    ctx = TraceRunContext(
        run_id=actual_run_id,
        root_dir=root,
        run_dir=run_dir,
        events_path=run_dir / "events.jsonl",
        snapshots_dir=snapshots_dir,
        manifest_path=run_dir / "manifest.json",
    )
    _run_context = ctx

    manifest_payload = {
        "schema_version": TRACE_RUN_SCHEMA_VERSION,
        "run_id": actual_run_id,
        "created_at": _now_iso_utc(),
        "trace_root": str(root),
        "manifest": _json_safe(manifest or {}),
    }
    ctx.manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    trace_run_event(
        component="trace",
        scope="run",
        event_type="run_start",
        message="trace run initialized",
        data={"manifest": manifest_payload["manifest"]},
    )
    return actual_run_id


def close_trace_run(summary: dict[str, Any] | None = None) -> None:
    """Close active run context and append run_end event."""
    global _run_context
    ctx = _run_context
    if ctx is None:
        return
    trace_run_event(
        component="trace",
        scope="run",
        event_type="run_end",
        message="trace run finalized",
        data={"summary": _json_safe(summary or {})},
    )
    _run_context = None


def trace_run_snapshot(
    *,
    component: str,
    scope: str,
    name: str,
    payload: Any,
    data: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Persist a snapshot blob and emit a snapshot event."""
    ctx = _run_context
    if ctx is None:
        return None

    payload_np = _to_numpy(payload)
    blob = pickle.dumps(payload_np)
    sha = hashlib.sha256(blob).hexdigest()
    with ctx.lock:
        ctx.seq += 1
        seq = ctx.seq
    file_name = (
        f"{seq:08d}_{_sanitize_name(component)}_{_sanitize_name(scope)}_{_sanitize_name(name)}.pkl"
    )
    snapshot_path = ctx.snapshots_dir / file_name
    snapshot_path.write_bytes(blob)

    ref = {
        "kind": "pickle",
        "path": str(snapshot_path.relative_to(ctx.run_dir)),
        "sha256": sha,
        "bytes": len(blob),
        "name": name,
    }
    trace_run_event(
        component=component,
        scope=scope,
        event_type="snapshot",
        message=f"snapshot:{name}",
        data={"snapshot": ref, **(data or {})},
        seq=seq,
    )
    return ref


def trace_run_event(
    *,
    component: str,
    scope: str,
    event_type: str,
    message: str,
    data: dict[str, Any] | None = None,
    cause_id: str | None = None,
    seq: int | None = None,
    event_time: float | None = None,
) -> int | None:
    """Append one canonical JSONL event for replay."""
    ctx = _run_context
    if ctx is None:
        return None

    with ctx.lock:
        if seq is None:
            ctx.seq += 1
            seq = ctx.seq
    event = {
        "schema_version": TRACE_RUN_SCHEMA_VERSION,
        "run_id": ctx.run_id,
        "seq": seq,
        "recorded_at": _now_iso_utc(),
        "event_time": event_time if event_time is not None else time.perf_counter(),
        "component": component,
        "scope": scope,
        "event_type": event_type,
        "message": message,
        "cause_id": cause_id,
        "data": _json_safe(data or {}),
    }
    with ctx.events_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=True) + "\n")
    return seq
