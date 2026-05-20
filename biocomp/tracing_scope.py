# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Tracing scope: `TraceScope`, `trace_scope`, `trace_here`, `trace_decision`, decorators.

Context-scoped event sourcing primitives. The `trace_scope` context manager is the
canonical entry-point; everything else (`trace_here`, `trace_decision`,
`register_trace_point`) composes on top of it.
"""

from __future__ import annotations

import inspect
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, TypeVar
from collections.abc import Callable

from biocomp.logging_config import get_logger
from biocomp import tracing_config as _tc
from biocomp.tracing_config import (
    trace_run_event,
    trace_run_snapshot,
    _to_numpy,
)

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


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
    parent: "TraceScope" | None = None

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
        if not _tc._config.is_active(self.component):
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

        if _tc._config.output_dir:
            self._persist()

        if self._token is not None:
            _trace_context.reset(self._token)

    def _persist(self) -> None:
        """Persist scope payload to legacy pickle and canonical run stream."""
        if self._scope is None or _tc._config.output_dir is None:
            return

        try:
            import dill
        except ImportError:
            import pickle as dill  # type: ignore[import-not-found]

        output_dir = _tc._config.output_dir / self.component
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
        if not _tc._config.is_active(component):
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


def _safe_repr(obj: Any, max_len: int = 200) -> str:
    """Safe string representation with length limit."""
    try:
        s = repr(obj)
        return s[:max_len] + "..." if len(s) > max_len else s
    except Exception:
        return f"<{type(obj).__name__}>"


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
            if scope is None or not _tc._config.is_active(component):
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
            if not _tc._config.is_active(component):
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
