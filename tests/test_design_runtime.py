"""Tests for design_runtime: types, protocol, and kernel lifecycle."""

import pytest

from biocomp.design_runtime import (
    DesignRuntimeContext,
    DesignRuntimeResult,
    GradientStepAdapter,
    StepAdapter,
    StepArtifact,
    run_kernel,
)
from biocomp.step_history import StepHistorySnapshot


class RecordingDispatch:
    """In-memory dispatch that records lifecycle calls."""

    def __init__(self):
        self.calls: list[tuple] = []

    def on_start(self, config, stack):
        self.calls.append(("on_start",))

    def on_step(self, step, config, step_history, stack):
        self.calls.append(("on_step", step))

    def on_end(self, step, config, step_history, stack):
        self.calls.append(("on_end", step))

    def needs_params_sync(self, step):
        return False


class FakeAdapter:
    """Minimal adapter for testing run_kernel lifecycle."""

    def __init__(self, params, loss=0.5):
        self._params = params
        self._loss = loss

    def run(self, ctx, initial_params):
        return DesignRuntimeResult(
            params=self._params,
            loss_history=[self._loss],
            final_snapshot=StepHistorySnapshot(data={"loss": self._loss}),
            step=ctx.total_steps,
        )


# --- StepArtifact ---


def test_step_artifact_creation():
    a = StepArtifact(step=1, loss=0.5, step_data={"loss": 0.5})
    assert a.step == 1
    assert a.loss == 0.5
    assert a.params is None


def test_step_artifact_with_params():
    a = StepArtifact(step=1, loss=0.5, step_data={}, params="fake_params")
    assert a.params == "fake_params"


def test_step_artifact_frozen():
    a = StepArtifact(step=1, loss=0.5, step_data={})
    with pytest.raises(AttributeError):
        a.step = 2  # type: ignore[misc]


# --- DesignRuntimeContext ---


def test_context_frozen():
    ctx = DesignRuntimeContext(
        stack=None, config=None, dispatch=RecordingDispatch(), total_steps=100
    )
    assert ctx.total_steps == 100
    with pytest.raises(AttributeError):
        ctx.total_steps = 200  # type: ignore[misc]


# --- DesignRuntimeResult ---


def test_result_fields():
    snapshot = StepHistorySnapshot(data={"loss": 0.1})
    result = DesignRuntimeResult(params="p", loss_history=[0.1], final_snapshot=snapshot, step=10)
    assert result.step == 10
    assert result.loss_history == [0.1]
    assert result.final_snapshot["loss"] == 0.1


# --- Protocol conformance ---


def test_gradient_step_adapter_satisfies_protocol():
    assert issubclass(GradientStepAdapter, StepAdapter)


def test_fake_adapter_satisfies_protocol():
    assert isinstance(FakeAdapter("p"), StepAdapter)


# --- run_kernel ---


def test_run_kernel_lifecycle_order():
    dispatch = RecordingDispatch()
    ctx = DesignRuntimeContext(stack=None, config=None, dispatch=dispatch, total_steps=10)
    adapter = FakeAdapter(params="final_params", loss=0.42)

    params, loss_history, snapshot = run_kernel(ctx, "initial_params", adapter)

    assert params == "final_params"
    assert loss_history == [0.42]
    assert isinstance(snapshot, StepHistorySnapshot)
    assert len(dispatch.calls) == 2
    assert dispatch.calls[0] == ("on_start",)
    assert dispatch.calls[1] == ("on_end", 10)


def test_run_kernel_returns_adapter_snapshot():
    dispatch = RecordingDispatch()
    ctx = DesignRuntimeContext(stack=None, config=None, dispatch=dispatch, total_steps=5)
    adapter = FakeAdapter(params="p", loss=0.99)

    _, _, snapshot = run_kernel(ctx, "p", adapter)
    assert snapshot["loss"] == 0.99


def test_run_kernel_passes_total_steps_to_on_end():
    dispatch = RecordingDispatch()
    ctx = DesignRuntimeContext(stack=None, config=None, dispatch=dispatch, total_steps=42)
    adapter = FakeAdapter(params="p")

    run_kernel(ctx, "p", adapter)
    assert dispatch.calls[-1] == ("on_end", 42)
