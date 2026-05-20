# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Tests for full object saving in tracing system."""

import pickle
import tempfile
from pathlib import Path


from biocomp.tracing import (
    TraceConfig,
    configure_tracing,
    load_network_from_snapshot,
    load_networks_from_stack_snapshot,
    should_save_full_objects,
    snapshot_full_network,
    snapshot_full_stack,
    trace_scope,
)


def make_simple_graph(node_types: list[str]):
    """Helper to create a simple GraphState with nodes of given types."""
    from biocomp.graphengine import GraphNode, GraphState

    nodes = {
        i: GraphNode(
            node_id=i,
            node_type=ntype,
            extra={"name": f"node_{i}"},
        )
        for i, ntype in enumerate(node_types)
    }
    return GraphState(nodes=nodes, edges={})


class TestTraceConfigFullObjects:
    """Test TraceConfig with save_full_objects flag."""

    def test_default_false(self):
        config = TraceConfig()
        assert config.save_full_objects is False

    def test_explicit_true(self):
        config = TraceConfig(save_full_objects=True)
        assert config.save_full_objects is True

    def test_from_env_default(self, monkeypatch):
        monkeypatch.delenv("BIOCOMP_TRACE_FULL", raising=False)
        config = TraceConfig.from_env()
        assert config.save_full_objects is False

    def test_from_env_enabled(self, monkeypatch):
        monkeypatch.setenv("BIOCOMP_TRACE", "1")
        monkeypatch.setenv("BIOCOMP_TRACE_FULL", "1")
        config = TraceConfig.from_env()
        assert config.save_full_objects is True

    def test_from_env_true_string(self, monkeypatch):
        monkeypatch.setenv("BIOCOMP_TRACE", "1")
        monkeypatch.setenv("BIOCOMP_TRACE_FULL", "true")
        config = TraceConfig.from_env()
        assert config.save_full_objects is True


class TestConfigureTracing:
    """Test configure_tracing with save_full_objects."""

    def test_configure_with_full_objects(self):
        configure_tracing(enabled=True, save_full_objects=True)
        assert should_save_full_objects() is True
        configure_tracing(enabled=False)  # cleanup

    def test_configure_without_full_objects(self):
        configure_tracing(enabled=True, save_full_objects=False)
        assert should_save_full_objects() is False
        configure_tracing(enabled=False)  # cleanup


class TestSnapshotNetwork:
    """Test snapshot_full_network function."""

    def test_snapshot_with_compute_graph(self):
        from biocomp.network import Network

        graph = make_simple_graph(["input", "output"])

        network = Network(name="test_network", compute_graph=graph)
        snapshot = snapshot_full_network(network)

        assert snapshot["name"] == "test_network"
        assert snapshot["compute_graph"] is not None
        assert isinstance(snapshot["compute_graph"], dict)

    def test_snapshot_without_compute_graph(self):
        from biocomp.network import Network

        network = Network(name="empty_network", compute_graph=None)
        snapshot = snapshot_full_network(network)

        assert snapshot["name"] == "empty_network"
        assert snapshot["compute_graph"] is None

    def test_snapshot_is_pickleable(self):
        from biocomp.network import Network

        graph = make_simple_graph(["input"])

        network = Network(name="test", compute_graph=graph)
        snapshot = snapshot_full_network(network)

        pickled = pickle.dumps(snapshot)
        restored = pickle.loads(pickled)
        assert restored["name"] == "test"


class TestSnapshotStack:
    """Test snapshot_full_stack function."""

    def test_snapshot_empty_stack(self):
        from biocomp.compute import ComputeStack

        stack = ComputeStack(networks=[])
        snapshot = snapshot_full_stack(stack)

        assert snapshot["networks"] == []
        assert snapshot["layers"] == []
        assert snapshot["is_built"] is False

    def test_snapshot_with_networks(self):
        from biocomp.compute import ComputeStack
        from biocomp.network import Network

        graph = make_simple_graph(["input"])
        network = Network(name="net1", compute_graph=graph)

        stack = ComputeStack(networks=[network])
        snapshot = snapshot_full_stack(stack)

        assert len(snapshot["networks"]) == 1
        assert snapshot["networks"][0]["name"] == "net1"

    def test_snapshot_with_tu_mapping(self):
        from biocomp.compute import ComputeStack
        from biocomp.graphengine import GraphState
        from biocomp.network import Network

        graph = GraphState(nodes={}, edges={})
        network = Network(name="net", compute_graph=graph)
        stack = ComputeStack(networks=[network])
        stack.tu_id_to_idx = {"TU1": 0, "TU2": 1}
        stack.n_tus = 2
        stack.inverse_tu_ids = {"TU1"}
        stack.no_masking_tu_ids = {"TU2"}

        snapshot = snapshot_full_stack(stack)

        assert snapshot["tu_id_to_idx"] == {"TU1": 0, "TU2": 1}
        assert snapshot["n_tus"] == 2
        assert "TU1" in snapshot["inverse_tu_ids"]
        assert "TU2" in snapshot["no_masking_tu_ids"]


class TestLoadNetwork:
    """Test load_network_from_snapshot function."""

    def test_roundtrip(self):
        from biocomp.network import Network

        graph = make_simple_graph(["input", "output"])

        original = Network(name="roundtrip_test", compute_graph=graph)
        snapshot = snapshot_full_network(original)

        restored = load_network_from_snapshot(snapshot)

        assert restored.name == "roundtrip_test"
        assert restored.compute_graph is not None
        assert len(restored.compute_graph.nodes) == 2

    def test_load_without_graph(self):
        snapshot = {"name": "no_graph", "compute_graph": None}
        restored = load_network_from_snapshot(snapshot)

        assert restored.name == "no_graph"
        assert restored.compute_graph is None


class TestLoadNetworksFromStack:
    """Test load_networks_from_stack_snapshot function."""

    def test_load_multiple_networks(self):
        from biocomp.compute import ComputeStack
        from biocomp.network import Network

        networks = []
        for i in range(3):
            graph = make_simple_graph(["input"])
            networks.append(Network(name=f"net{i}", compute_graph=graph))

        stack = ComputeStack(networks=networks)
        snapshot = snapshot_full_stack(stack)

        restored = load_networks_from_stack_snapshot(snapshot)

        assert len(restored) == 3
        for i, net in enumerate(restored):
            assert net.name == f"net{i}"


class TestIntegrationWithTraceScope:
    """Integration tests with trace_scope."""

    def test_full_objects_saved_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            configure_tracing(
                enabled=True,
                components={"test"},
                output_dir=Path(tmpdir),
                save_full_objects=True,
            )

            with trace_scope("test_scope", component="test") as scope:
                scope.snapshot("summary", {"key": "value"})
                if should_save_full_objects():
                    scope.snapshot("full_data", {"full": "object"})

            # Check file was created
            trace_files = list(Path(tmpdir).glob("**/*.pkl"))
            assert len(trace_files) == 1

            # Load and verify
            with open(trace_files[0], "rb") as f:
                import dill

                data = dill.load(f)

            assert "summary" in data["snapshots"]
            assert "full_data" in data["snapshots"]
            assert data["snapshots"]["full_data"]["full"] == "object"

            configure_tracing(enabled=False)  # cleanup

    def test_full_objects_not_saved_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            configure_tracing(
                enabled=True,
                components={"test"},
                output_dir=Path(tmpdir),
                save_full_objects=False,
            )

            with trace_scope("test_scope", component="test") as scope:
                scope.snapshot("summary", {"key": "value"})
                if should_save_full_objects():
                    scope.snapshot("full_data", {"full": "object"})

            trace_files = list(Path(tmpdir).glob("**/*.pkl"))
            assert len(trace_files) == 1

            with open(trace_files[0], "rb") as f:
                import dill

                data = dill.load(f)

            assert "summary" in data["snapshots"]
            assert "full_data" not in data["snapshots"]

            configure_tracing(enabled=False)  # cleanup
