# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Unified design optimization runtime kernel.

Provides shared types and dispatch lifecycle management for
gradient descent and pluggable optimizer paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .logging_config import get_logger
from .step_history import StepHistorySnapshot

if TYPE_CHECKING:
    from .compute import ComputeStack
    from .logger_dispatch import LoggerDispatch
    from .parameters import ParameterTree

logger = get_logger(__name__)


@dataclass(frozen=True)
class StepArtifact:
    """Single optimization step output -- shared schema across all paths."""

    step: int
    loss: float
    step_data: dict[str, Any]
    params: ParameterTree | None = None


@dataclass(frozen=True)
class DesignRuntimeContext:
    """Immutable context for a design optimization run."""

    stack: ComputeStack
    config: object
    dispatch: LoggerDispatch
    total_steps: int


@dataclass
class DesignRuntimeResult:
    """Output of an optimization adapter's run."""

    params: ParameterTree
    loss_history: list[float]
    final_snapshot: StepHistorySnapshot
    step: int


@runtime_checkable
class StepAdapter(Protocol):
    """Adapter for optimization step implementations.

    Owns inner loop mechanics (compilation, batching, deferred sync).
    run_kernel owns dispatch lifecycle (on_start/on_end).
    """

    def run(
        self,
        ctx: DesignRuntimeContext,
        initial_params: ParameterTree,
    ) -> DesignRuntimeResult: ...


@dataclass
class GradientStepAdapter:
    """Wraps optimutils.optimize() for gradient descent optimization."""

    step_fn: Any
    optimizer_state: Any
    xbatches: Any
    ybatches: Any
    steps_per_epoch: int
    key: Any
    precompiled: bool = False
    select_best_synced_params: bool = False
    best_synced_score_fn: Any = None
    best_synced_initial_score: float | None = None
    step_offset: int = 0
    emit_step_zero: bool = True

    def run(
        self,
        ctx: DesignRuntimeContext,
        initial_params: ParameterTree,
    ) -> DesignRuntimeResult:
        from .optimutils import optimize

        params, loss_history, snapshot = optimize(
            self.step_fn,
            initial_params,
            self.optimizer_state,
            xbatches=self.xbatches,
            ybatches=self.ybatches,
            config=ctx.config,
            n_total_steps=ctx.total_steps,
            steps_per_epoch=self.steps_per_epoch,
            key=self.key,
            stack=ctx.stack,
            dispatch=ctx.dispatch,
            precompiled=self.precompiled,
            skip_lifecycle=True,
            select_best_synced_params=self.select_best_synced_params,
            best_synced_score_fn=self.best_synced_score_fn,
            best_synced_initial_score=self.best_synced_initial_score,
            step_offset=self.step_offset,
            emit_step_zero=(self.emit_step_zero and self.step_offset == 0),
        )
        return DesignRuntimeResult(
            params=params,
            loss_history=loss_history,
            final_snapshot=snapshot,
            step=self.step_offset + ctx.total_steps,
        )


def run_kernel(
    ctx: DesignRuntimeContext,
    initial_params: ParameterTree,
    adapter: StepAdapter,
) -> tuple[ParameterTree, list[float], StepHistorySnapshot]:
    """Unified design optimization kernel.

    Owns dispatch lifecycle (on_start/on_end).
    Delegates inner loop to adapter.
    """
    ctx.dispatch.on_start(ctx.config, ctx.stack)
    result = adapter.run(ctx, initial_params)
    ctx.dispatch.on_end(result.step, ctx.config, result.final_snapshot, ctx.stack)
    return result.params, result.loss_history, result.final_snapshot
