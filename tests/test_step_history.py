import pytest
from typing import cast

from biocomp.step_history import StepHistorySnapshot, ensure_step_history_snapshot


def test_step_history_snapshot_from_raw_mapping():
    snapshot = StepHistorySnapshot.from_raw({"loss": 1.0, "tag": "ok"})
    assert snapshot["loss"] == 1.0
    assert snapshot.get("tag") == "ok"
    assert set(snapshot.keys()) == {"loss", "tag"}


def test_step_history_snapshot_from_raw_rejects_non_mapping():
    with pytest.raises(TypeError, match="expected Mapping\\[str, Any\\]"):
        StepHistorySnapshot.from_raw(cast(object, ["not", "a", "mapping"]))


def test_ensure_step_history_snapshot_accepts_snapshot():
    original = StepHistorySnapshot.from_raw({"loss": 1.0})
    normalized = ensure_step_history_snapshot(original)
    assert normalized is original


def test_ensure_step_history_snapshot_rejects_non_mapping_with_context():
    with pytest.raises(TypeError, match="test_context invariant violated"):
        ensure_step_history_snapshot(cast(object, ["bad"]), context="test_context")
