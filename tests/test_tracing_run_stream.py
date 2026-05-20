# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Tests for canonical run-scoped tracing stream."""

import json
import tempfile
from pathlib import Path

from biocomp.tracing import (
    close_trace_run,
    configure_trace_run,
    configure_tracing,
    list_trace_runs,
    load_trace_run_events,
    save_debug_state,
    trace_scope,
)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def test_trace_scope_writes_canonical_run_stream():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        configure_tracing(enabled=True, components={"test"}, output_dir=root)
        configure_trace_run(run_id="unit-run", output_dir=root, manifest={"kind": "unit"})

        with trace_scope("scope_a", component="test") as scope:
            scope.event("checkpoint", "hello", {"step": 1})
            scope.snapshot("arr", {"x": [1, 2, 3]})

        close_trace_run(summary={"ok": True})

        run_dir = root / "runs" / "unit-run"
        events_path = run_dir / "events.jsonl"
        assert events_path.exists()
        events = _read_jsonl(events_path)
        assert any(e["event_type"] == "run_start" for e in events)
        assert any(e["event_type"] == "checkpoint" for e in events)
        assert any(e["event_type"] == "snapshot" for e in events)
        assert any(e["event_type"] == "run_end" for e in events)

        snapshot_files = list((run_dir / "snapshots").glob("*.pkl"))
        assert snapshot_files, "expected at least one snapshot payload file"

        configure_tracing(enabled=False)


def test_save_debug_state_emits_into_canonical_stream():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        configure_tracing(enabled=True, components={"design"}, output_dir=root)
        configure_trace_run(run_id="debug-run", output_dir=root)

        save_debug_state(
            "unit_debug_stage",
            data={"arr": [1, 2, 3]},
            metadata={"source": "unit-test"},
            mode="design",
            force=True,
        )
        close_trace_run()

        events = load_trace_run_events(root / "runs" / "debug-run")
        debug_events = [e for e in events if e["event_type"] == "debug_state"]
        assert debug_events, "debug_state event should be present in canonical stream"
        snapshot_events = [e for e in events if e["event_type"] == "snapshot"]
        assert snapshot_events, "debug snapshot should be present in canonical stream"

        runs = list_trace_runs(root)
        assert any(p.name == "debug-run" for p in runs)

        configure_tracing(enabled=False)
