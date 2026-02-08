"""General decision tracing system for biocompiler.

Context-scoped event sourcing with zero-cost disable. Uses contextvars for
async-safe, thread-local trace contexts that accumulate immutable events.

Enable tracing via environment variables:
    BIOCOMP_TRACE=1                     # Enable all tracing
    BIOCOMP_TRACE_COMPONENTS=design,commit  # Enable specific components only
    BIOCOMP_TRACE_DIR=/tmp/traces       # Output directory for persisted traces
    BIOCOMP_TRACE_FULL=1                # Save full Network/Stack objects (not just summaries)

Or programmatically:
    from biocomp.tracing import configure_tracing
    configure_tracing(enabled=True, components={"design"}, output_dir=Path("/tmp/traces"))
    configure_tracing(enabled=True, save_full_objects=True)  # Enable full object saving

Usage:
    from biocomp.tracing import trace_scope, trace_here, trace_decision

    # Context manager for scoped tracing
    with trace_scope("prune_tus", component="design") as scope:
        scope.event("start", "Beginning TU pruning", {"n_tus": 24})
        scope.snapshot("params_before", params)

        for tu_id, prob in tu_probs.items():
            if prob < 0.2:
                scope.decision("disable_tu", outcome=False, reason="below_floor",
                               inputs={"tu_id": tu_id, "prob": prob})

    # One-liner for quick trace points
    trace_here("checkpoint", component="design", data={"step": 100, "loss": 0.5})

    # Decorator for automatic function tracing
    @trace_decision("should_prune_tu", component="design")
    def should_prune(tu_id: str, prob: float) -> bool:
        return prob < 0.5

Full Object Saving:
    When BIOCOMP_TRACE_FULL=1, trace snapshots include full serialized objects
    (Network, ComputeStack, ParameterTree) instead of just summaries. This enables
    direct object reconstruction from trace files for debugging:

        from biocomp.tracing import load_network_from_snapshot, load_networks_from_stack_snapshot
        import pickle

        # Load trace file
        with open("debug_trace/network/recipe_to_networks_*.pkl", "rb") as f:
            data = pickle.load(f)

        # Reconstruct networks
        networks = [load_network_from_snapshot(n) for n in data["snapshots"]["networks_full"]]
"""

from __future__ import annotations

import inspect
import json
import os
import pickle
import hashlib
import time
import threading
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar
from uuid import uuid4

import numpy as np

from biocomp.logging_config import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


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
    def from_env(cls) -> TraceConfig:
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


from pydantic import BaseModel, model_validator


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
    """Programmatic configuration (for tests, notebooks).

    Args:
        enabled: Whether tracing is enabled
        components: Set of component names to trace. None or empty means all.
        output_dir: Directory to persist trace files. None means don't persist.
        save_full_objects: If True, save full Network/Stack objects in snapshots.
    """
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
    return datetime.now(timezone.utc).isoformat()


def _sanitize_name(name: str) -> str:
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)
    return safe.strip("._") or "item"


def _json_safe(val: Any) -> Any:
    """Make values JSON-serializable without exploding payload size."""
    if val is None or isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, (list, tuple)):
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

    actual_run_id = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:8]}"
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


def _ensure_trace_run_for_event(component: str) -> None:
    """Keep explicit control of canonical run tracing (no implicit runs)."""
    _ = component


def trace_run_snapshot(
    *,
    component: str,
    scope: str,
    name: str,
    payload: Any,
    data: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Persist a snapshot blob and emit a snapshot event."""
    _ensure_trace_run_for_event(component)
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
    _ensure_trace_run_for_event(component)
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


# ─────────────────────────────────────────────────────────────────────────────
# Event and Scope Data Structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TraceEvent:
    """Immutable event record."""

    timestamp: float
    event_type: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    cause_id: str | None = None  # links to parent decision


@dataclass
class TraceScope:
    """Named scope that groups related events."""

    name: str
    component: str
    events: list[TraceEvent] = field(default_factory=list)
    snapshots: dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.perf_counter)
    parent: TraceScope | None = None

    def event(
        self,
        event_type: str,
        message: str,
        data: dict[str, Any] | None = None,
        cause_id: str | None = None,
    ) -> str:
        """Record an event, return its ID."""
        evt = TraceEvent(
            timestamp=time.perf_counter(),
            event_type=event_type,
            message=message,
            data=data or {},
            cause_id=cause_id,
        )
        self.events.append(evt)
        return f"{self.component}:{event_type}:{evt.timestamp}"

    def snapshot(self, name: str, state: Any) -> None:
        """Save pickle-able state snapshot."""
        self.snapshots[name] = _to_numpy(state)

    def decision(
        self,
        decision_type: str,
        outcome: Any,
        reason: str,
        inputs: dict[str, Any] | None = None,
    ) -> str:
        """Record a decision with structured metadata."""
        return self.event(
            event_type=f"decision:{decision_type}",
            message=f"{decision_type} -> {outcome}",
            data={"outcome": outcome, "reason": reason, "inputs": inputs or {}},
        )


class _NullScope:
    """No-op scope when tracing disabled - true zero cost."""

    __slots__ = ()

    def event(self, *_args: Any, **_kwargs: Any) -> str:
        return ""

    def snapshot(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def decision(self, *_args: Any, **_kwargs: Any) -> str:
        return ""


# Context variable for current trace scope
_trace_context: ContextVar[TraceScope | None] = ContextVar("trace_context", default=None)


# ─────────────────────────────────────────────────────────────────────────────
# Context Manager
# ─────────────────────────────────────────────────────────────────────────────


class trace_scope:
    """Context manager for scoped tracing.

    Usage:
        with trace_scope("prune_tus", component="design") as scope:
            scope.event("start", "Beginning TU pruning", {"n_tus": 24})
            scope.snapshot("params_before", jax.device_get(params))

            for tu_id, prob in tu_probs.items():
                if prob < 0.2:
                    scope.decision("disable_tu", outcome=False, reason="below_floor",
                                   inputs={"tu_id": tu_id, "prob": prob})

    When tracing is disabled for the component, returns a _NullScope that does nothing
    (true zero-cost).
    """

    def __init__(self, name: str, component: str = "default"):
        self.name = name
        self.component = component
        self._scope: TraceScope | None = None
        self._token: Any = None

    def __enter__(self) -> TraceScope | _NullScope:
        if not _config.is_active(self.component):
            return _NullScope()

        parent = _trace_context.get()
        self._scope = TraceScope(name=self.name, component=self.component, parent=parent)
        self._token = _trace_context.set(self._scope)
        return self._scope

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        if self._scope is None:
            return

        duration = time.perf_counter() - self._scope.start_time
        self._scope.event(
            "scope_end",
            f"Completed {self.name}",
            {
                "duration": duration,
                "n_events": len(self._scope.events),
                "error": str(exc_val) if exc_val else None,
            },
        )

        if _config.output_dir:
            self._persist()

        if self._token is not None:
            _trace_context.reset(self._token)

    def _persist(self) -> None:
        """Persist scope payload to legacy pickle and canonical run stream."""
        if self._scope is None or _config.output_dir is None:
            return

        try:
            import dill
        except ImportError:
            import pickle as dill  # type: ignore[import-not-found]

        output_dir = _config.output_dir / self.component
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = int(self._scope.start_time * 1000)
        path = output_dir / f"{self._scope.name}_{ts}.pkl"

        payload = {
            "name": self._scope.name,
            "component": self._scope.component,
            "events": self._scope.events,
            "snapshots": self._scope.snapshots,
            "duration": time.perf_counter() - self._scope.start_time,
        }

        # Canonical SSOT stream for replay (run-scoped JSONL + snapshot blobs).
        try:
            for evt in self._scope.events:
                trace_run_event(
                    component=self._scope.component,
                    scope=self._scope.name,
                    event_type=evt.event_type,
                    message=evt.message,
                    data=evt.data,
                    cause_id=evt.cause_id,
                    event_time=evt.timestamp,
                )
            for snap_name, snap_payload in self._scope.snapshots.items():
                trace_run_snapshot(
                    component=self._scope.component,
                    scope=self._scope.name,
                    name=snap_name,
                    payload=snap_payload,
                )
        except Exception as e:
            logger.warning(f"Failed to persist canonical run trace for scope {self._scope.name}: {e}")

        try:
            with open(path, "wb") as f:
                dill.dump(payload, f)
            logger.debug(f"Trace persisted: {path}")
        except Exception as e:
            logger.warning(f"Failed to persist trace to {path}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────


def trace_here(
    name: str,
    component: str = "default",
    data: dict[str, Any] | None = None,
    snapshot: Any = None,
) -> None:
    """Quick trace point - records event in current scope or creates ephemeral one.

    Use for one-liner trace points when you don't need a full scope:
        trace_here("checkpoint", component="design", data={"step": 100, "loss": 0.5})
    """
    scope = _trace_context.get()
    if scope is None:
        if not _config.is_active(component):
            return
        # Create ephemeral scope for standalone trace
        with trace_scope(name, component) as ephemeral:
            if data:
                ephemeral.event("trace", name, data)
            if snapshot is not None:
                ephemeral.snapshot("data", snapshot)
    else:
        if data:
            scope.event("trace", name, data)
        if snapshot is not None:
            scope.snapshot(name, snapshot)


def get_current_scope() -> TraceScope | None:
    """Get the current trace scope, or None if not in a traced context."""
    return _trace_context.get()


# ─────────────────────────────────────────────────────────────────────────────
# Decorators
# ─────────────────────────────────────────────────────────────────────────────


def trace_decision(decision_type: str, component: str = "default") -> Callable[[F], F]:
    """Decorator to trace function as a decision point.

    Usage:
        @trace_decision("should_prune_tu", component="design")
        def should_prune(tu_id: str, prob: float) -> bool:
            return prob < 0.5

    Records the function call as a decision in the current trace scope.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            scope = _trace_context.get()
            if scope is None or not _config.is_active(component):
                return func(*args, **kwargs)

            result = func(*args, **kwargs)

            scope.decision(
                decision_type=decision_type,
                outcome=result,
                reason=f"{func.__name__}",
                inputs={"args": _safe_repr(args), "kwargs": _safe_repr(kwargs)},
            )
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def register_trace_point(
    component: str = "default",
    capture_args: bool = False,
    capture_result: bool = False,
    snapshot_args: list[str] | None = None,
) -> Callable[[F], F]:
    """Decorator to automatically trace function calls.

    Usage:
        @register_trace_point(component="design", capture_args=True, capture_result=True)
        def traced_function(a, b):
            return a + b  # Entry/exit/args/result all traced automatically

        @register_trace_point(component="commit", snapshot_args=["params", "network"])
        def commit_network(params, network, verbose=False):
            ...  # params and network are pickled as snapshots
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _config.is_active(component):
                return func(*args, **kwargs)

            with trace_scope(func.__name__, component) as scope:
                if capture_args:
                    scope.event(
                        "call",
                        f"Called {func.__name__}",
                        {"args": _safe_repr(args), "kwargs": _safe_repr(kwargs)},
                    )
                if snapshot_args:
                    # Snapshot specific arguments by name
                    sig = inspect.signature(func)
                    try:
                        bound = sig.bind(*args, **kwargs)
                        bound.apply_defaults()
                        for arg_name in snapshot_args:
                            if arg_name in bound.arguments:
                                scope.snapshot(f"arg_{arg_name}", bound.arguments[arg_name])
                    except TypeError:
                        pass  # Signature binding failed, skip snapshots

                result = func(*args, **kwargs)

                if capture_result:
                    scope.event(
                        "return",
                        f"Returned from {func.__name__}",
                        {"result": _safe_repr(result)},
                    )
                return result

        return wrapper  # type: ignore[return-value]

    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────


def _safe_repr(obj: Any, max_len: int = 200) -> str:
    """Safe string representation with length limit."""
    try:
        s = repr(obj)
        return s[:max_len] + "..." if len(s) > max_len else s
    except Exception:
        return f"<{type(obj).__name__}>"


def _to_numpy(val: Any) -> Any:
    """Convert JAX arrays and similar to numpy for pickling."""
    if hasattr(val, "__array__"):
        return np.asarray(val)
    if isinstance(val, dict):
        return {k: _to_numpy(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return type(val)(_to_numpy(v) for v in val)
    return val


# ─────────────────────────────────────────────────────────────────────────────
# Replay/Analysis Utilities
# ─────────────────────────────────────────────────────────────────────────────


def load_trace(path: Path | str) -> dict[str, Any]:
    """Load a trace from a pickle file.

    Returns:
        Dictionary with keys: name, component, events, snapshots, duration
    """
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
    """List all trace files in a directory, optionally filtered by component.

    Args:
        trace_dir: Base trace directory (BIOCOMP_TRACE_DIR)
        component: If specified, only list traces for this component

    Returns:
        List of paths to trace pickle files, sorted by modification time
    """
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


def get_decisions(trace: dict[str, Any]) -> list[TraceEvent]:
    """Extract all decision events from a trace.

    Args:
        trace: Loaded trace dictionary

    Returns:
        List of TraceEvent objects that are decisions
    """
    return [e for e in trace.get("events", []) if e.event_type.startswith("decision:")]


def format_decision_chain(trace: dict[str, Any]) -> str:
    """Format decisions from a trace as a readable chain.

    Args:
        trace: Loaded trace dictionary

    Returns:
        Multi-line string showing decision chain
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Backward Compatibility (designdebug replacement)
# ─────────────────────────────────────────────────────────────────────────────


def is_design_debug_enabled() -> bool:
    """Backward compatibility: check if design tracing is enabled.

    Replaces: from biocomp.designdebug import is_design_debug_enabled
    """
    return _config.is_active("design")


def is_plot_debug_enabled() -> bool:
    """Backward compatibility: check if plot tracing is enabled.

    Replaces: from biocomp.designdebug import is_plot_debug_enabled
    """
    return _config.is_active("plot")


def save_debug_state(
    stage: str,
    data: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    mode: str = "design",
    force: bool = False,
) -> Path | None:
    """Backward compatibility: save debug state as a trace snapshot.

    Replaces: from biocomp.designdebug import save_debug_state

    This function wraps the old interface and additionally emits to the
    canonical run-scoped trace stream.

    Args:
        stage: Name/stage of the debug state
        data: Dictionary of data to save
        metadata: Optional metadata dictionary
        output_dir: Output directory (overrides BIOCOMP_TRACE_DIR)
        mode: Component mode ("design" or "plot")
        force: If True, save even if tracing is disabled
    """
    if not force and not _config.is_active(mode):
        return None

    # Determine output directory
    if output_dir is not None:
        effective_output_dir = Path(output_dir) / "_debug_dumps"
    elif _config.output_dir is not None:
        effective_output_dir = _config.output_dir / mode
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
        elif isinstance(val, (list, tuple)) and len(val) > 0 and hasattr(val[0], "shape"):
            shapes[key] = [v.shape for v in val]
            stats[key] = [_compute_stats(v) for v in val]

    from datetime import datetime

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

    import pickle

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


# ─────────────────────────────────────────────────────────────────────────────
# Summary Models for Structured Tracing
# ─────────────────────────────────────────────────────────────────────────────


class LayerSummary(BaseModel):
    """Summary of a single stack layer."""

    layer_id: int
    layer_type: str | None = None
    n_nodes: int = 0
    namespace: str | None = None
    is_built: bool = False


class TUMappingSummary(BaseModel):
    """Summary of TU masking configuration."""

    n_tus: int
    tu_ids: list[str] = []
    inverse_tu_ids: list[str] = []
    no_masking_tu_ids: list[str] = []


class StackSummary(BaseModel):
    """Structured summary of a ComputeStack."""

    n_networks: int = 0
    network_names: list[str] = []
    n_layers: int = 0
    layers: list[LayerSummary] = []
    is_built: bool = False
    is_assembled: bool = False
    number_of_nodes: int = 0
    tu_mapping: TUMappingSummary | None = None


class SourceNodeSummary(BaseModel):
    """Summary of a source node."""

    id: int
    source_id: str | None = None


class NetworkSummary(BaseModel):
    """Structured summary of a Network."""

    name: str | None = None
    n_nodes: int = 0
    n_edges: int = 0
    node_types: dict[str, int] = {}
    source_nodes: list[SourceNodeSummary] = []
    output_nodes: list[int] = []
    tu_ids: list[str] = []
    n_tus: int = 0


class ParamsSummary(BaseModel):
    """Structured summary of a ParameterTree."""

    n_paths: int = 0
    shapes: dict[str, list[int]] = {}
    sample_paths: list[str] = []
    tags: dict[str, int] = {}


class GraphNodeSummary(BaseModel):
    """Summary of a graph node."""

    node_type: str
    extra_keys: list[str] = []
    is_inverse_of: int | None = None


class GraphEdgeSummary(BaseModel):
    """Summary of a graph edge."""

    source_id: int
    target_id: int
    from_slot: int
    to_slot: int
    content_type: str | None = None
    tu_ids: list[str] = []


class GraphSummary(BaseModel):
    """Structured serialization of a GraphState."""

    n_nodes: int = 0
    n_edges: int = 0
    nodes: dict[str, GraphNodeSummary] = {}
    edges: list[GraphEdgeSummary] = []


def summarize_params(params: Any) -> ParamsSummary:
    """Create structured summary of parameter tree for tracing."""
    if not hasattr(params, "data"):
        return ParamsSummary()

    paths = []
    shapes: dict[str, list[int]] = {}
    for path, val in params.data.iter_leaves():
        path_str = str(path)
        paths.append(path_str)
        if hasattr(val, "shape"):
            shapes[path_str] = list(val.shape)

    tags: dict[str, int] = {}
    if hasattr(params, "tagnames"):
        for tagname in params.tagnames:
            count = 0
            tag_idx = params.tagnames.index(tagname)
            for _, tag_arr in params.tags.iter_leaves():
                if tag_arr[tag_idx]:
                    count += 1
            tags[tagname] = count

    return ParamsSummary(
        n_paths=len(paths),
        shapes=shapes,
        sample_paths=paths[:10],
        tags=tags,
    )


def summarize_network(network: Any) -> NetworkSummary:
    """Create structured summary of network for tracing."""
    from collections import Counter

    name = getattr(network, "name", None)
    cg = getattr(network, "compute_graph", None)
    if cg is None:
        return NetworkSummary(name=name)

    nodes = cg.nodes
    edges = cg.edges

    source_nodes = [
        SourceNodeSummary(id=n.node_id, source_id=n.extra.get("source_id"))
        for n in nodes.values()
        if n.node_type == "source"
    ]
    output_nodes = [n.node_id for n in nodes.values() if n.node_type == "output"]

    tu_ids: set[str] = set()
    for edge in edges.values():
        if edge.extra:
            tu_ids_on_edge = edge.extra.get("tu_id", [])
            if tu_ids_on_edge:
                tu_ids.update(tu_ids_on_edge)

    return NetworkSummary(
        name=name,
        n_nodes=len(nodes),
        n_edges=len(edges),
        node_types=dict(Counter(n.node_type for n in nodes.values())),
        source_nodes=source_nodes,
        output_nodes=output_nodes,
        tu_ids=list(tu_ids),
        n_tus=len(tu_ids),
    )


def summarize_stack(stack: Any) -> StackSummary:
    """Create structured summary of ComputeStack for tracing."""
    networks = getattr(stack, "networks", None)
    network_names = (
        [getattr(n, "name", f"net_{i}") for i, n in enumerate(networks)]
        if networks
        else []
    )

    layers_attr = getattr(stack, "layers", None)
    layers = []
    if layers_attr is not None:
        for i, layer in enumerate(layers_attr):
            layers.append(
                LayerSummary(
                    layer_id=i,
                    layer_type=getattr(layer, "f_type", None),
                    n_nodes=len(layer.nodes) if layer.nodes else 0,
                    namespace=getattr(layer, "namespace", None),
                    is_built=getattr(layer, "is_built", False),
                )
            )

    tu_mapping = None
    tu_id_to_idx = getattr(stack, "tu_id_to_idx", None)
    if tu_id_to_idx is not None:
        tu_mapping = TUMappingSummary(
            n_tus=len(tu_id_to_idx),
            tu_ids=list(tu_id_to_idx.keys())[:20],
            inverse_tu_ids=list(getattr(stack, "inverse_tu_ids", set()))[:10],
            no_masking_tu_ids=list(getattr(stack, "no_masking_tu_ids", set()))[:10],
        )

    return StackSummary(
        n_networks=len(networks) if networks else 0,
        network_names=network_names,
        n_layers=len(layers),
        layers=layers,
        is_built=getattr(stack, "is_built", False),
        is_assembled=getattr(stack, "is_assembled", False),
        number_of_nodes=getattr(stack, "number_of_nodes", 0),
        tu_mapping=tu_mapping,
    )


def serialize_graph(graph: Any) -> GraphSummary:
    """Serialize graph structure for tracing (no pickle, structure only)."""
    nodes_attr = getattr(graph, "nodes", None)
    edges_attr = getattr(graph, "edges", None)

    nodes: dict[str, GraphNodeSummary] = {}
    if nodes_attr is not None:
        for nid, n in nodes_attr.items():
            nodes[str(nid)] = GraphNodeSummary(
                node_type=n.node_type,
                extra_keys=list(n.extra.keys()) if n.extra else [],
                is_inverse_of=n.is_inverse_of.node_id if n.is_inverse_of else None,
            )

    edges: list[GraphEdgeSummary] = []
    if edges_attr is not None:
        for e in edges_attr.values():
            edges.append(
                GraphEdgeSummary(
                    source_id=e.source_id,
                    target_id=e.target_id,
                    from_slot=e.from_output_slot,
                    to_slot=e.to_input_slot,
                    content_type=e.content_type,
                    tu_ids=e.extra.get("tu_id", []) if e.extra else [],
                )
            )

    return GraphSummary(
        n_nodes=len(nodes),
        n_edges=len(edges),
        nodes=nodes,
        edges=edges,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Full Object Snapshot Helpers (for save_full_objects mode)
# ─────────────────────────────────────────────────────────────────────────────


def snapshot_full_network(network: Any) -> dict[str, Any]:
    """Convert Network to pickle-safe dict with full data.

    This captures the complete network structure including compute_graph,
    enabling full reconstruction from trace files.

    Args:
        network: A Network object

    Returns:
        Dictionary with serialized network data
    """
    compute_graph = getattr(network, "compute_graph", None)
    graph_data = None
    if compute_graph is not None:
        graph_data = compute_graph.model_dump(mode="python")

    # nb_inputs/nb_outputs are properties that may fail if compute_graph is None
    try:
        nb_inputs = network.nb_inputs if compute_graph is not None else 0
    except Exception:
        nb_inputs = 0
    try:
        nb_outputs = network.nb_outputs if compute_graph is not None else 0
    except Exception:
        nb_outputs = 0

    return {
        "name": getattr(network, "name", None),
        "compute_graph": graph_data,
        "metadata": getattr(network, "metadata", {}),
        "nb_inputs": nb_inputs,
        "nb_outputs": nb_outputs,
    }


def snapshot_full_stack(stack: Any) -> dict[str, Any]:
    """Convert ComputeStack to pickle-safe dict with full data.

    This captures networks, layers, and TU mapping for reconstruction.

    Args:
        stack: A ComputeStack object

    Returns:
        Dictionary with serialized stack data
    """
    networks = getattr(stack, "networks", None)
    networks_data = [snapshot_full_network(n) for n in networks] if networks else []

    layers_attr = getattr(stack, "layers", None)
    layers_data = []
    if layers_attr is not None:
        for layer in layers_attr:
            layer_info = {
                "layer_id": getattr(layer, "layer_id", None),
                "f_type": getattr(layer, "f_type", None),
                "namespace": getattr(layer, "namespace", None),
                "is_built": getattr(layer, "is_built", False),
                "n_nodes": len(layer.nodes) if layer.nodes else 0,
                "nodes": [
                    {
                        "network_id": n.network_id,
                        "node_id": n.node_id,
                        "layer_number": n.layer_number,
                        "node_position_in_layer": n.node_position_in_layer,
                    }
                    for n in layer.nodes
                ] if layer.nodes else [],
            }
            layers_data.append(layer_info)

    return {
        "networks": networks_data,
        "layers": layers_data,
        "tu_id_to_idx": dict(getattr(stack, "tu_id_to_idx", {}) or {}),
        "n_tus": getattr(stack, "n_tus", 0),
        "inverse_tu_ids": list(getattr(stack, "inverse_tu_ids", set()) or set()),
        "no_masking_tu_ids": list(getattr(stack, "no_masking_tu_ids", set()) or set()),
        "is_built": getattr(stack, "is_built", False),
        "is_assembled": getattr(stack, "is_assembled", False),
        "number_of_nodes": getattr(stack, "number_of_nodes", 0),
    }


def snapshot_full_params(params: Any) -> dict[str, Any]:
    """Convert ParameterTree to pickle-safe dict with full values (not just shapes).

    Args:
        params: A ParameterTree object

    Returns:
        Dictionary with full parameter data (converted to numpy)
    """
    if not hasattr(params, "data"):
        return {"error": "no_data_attribute"}

    values: dict[str, Any] = {}
    for path, val in params.data.iter_leaves():
        path_str = str(path)
        values[path_str] = _to_numpy(val)

    tags: dict[str, list[str]] = {}
    if hasattr(params, "tagnames"):
        for tagname in params.tagnames:
            tagged_paths = []
            tag_idx = params.tagnames.index(tagname)
            for path, tag_arr in params.tags.iter_leaves():
                if tag_arr[tag_idx]:
                    tagged_paths.append(str(path))
            tags[tagname] = tagged_paths

    return {
        "values": values,
        "tags": tags,
        "tagnames": list(getattr(params, "tagnames", [])),
    }


def load_network_from_snapshot(data: dict[str, Any]) -> Any:
    """Reconstruct Network from full snapshot.

    Args:
        data: Dictionary from snapshot_full_network()

    Returns:
        Reconstructed Network object
    """
    from biocomp.network import Network
    from biocomp.graphengine import GraphState

    compute_graph = None
    if data.get("compute_graph") is not None:
        compute_graph = GraphState.model_validate(data["compute_graph"])

    network = Network(
        name=data.get("name"),
        compute_graph=compute_graph,
    )
    return network


def load_networks_from_stack_snapshot(data: dict[str, Any]) -> list[Any]:
    """Reconstruct networks from stack snapshot.

    Args:
        data: Dictionary from snapshot_full_stack()

    Returns:
        List of reconstructed Network objects
    """
    networks = []
    for net_data in data.get("networks", []):
        networks.append(load_network_from_snapshot(net_data))
    return networks


def should_save_full_objects() -> bool:
    """Check if full object saving is enabled."""
    return _config.save_full_objects


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
