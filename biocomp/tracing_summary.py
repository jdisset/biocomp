# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Structured summary and full-snapshot helpers for Network / ComputeStack / params.

Pydantic models that capture structure-only fingerprints (cheap to JSON-serialize) plus
pickle-safe full-snapshot dicts used by `save_full_objects` mode for offline replay.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from biocomp.tracing_config import _to_numpy


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
    """Pickle-safe dict for full Network reconstruction (includes compute_graph dump)."""
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
    """Pickle-safe dict for full ComputeStack reconstruction (networks, layers, TU mapping)."""
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
    """Pickle-safe dict for ParameterTree with full numpy-converted values."""
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
    """Reconstruct a Network from `snapshot_full_network()` output."""
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
    """Reconstruct Networks from `snapshot_full_stack()` output."""
    networks = []
    for net_data in data.get("networks", []):
        networks.append(load_network_from_snapshot(net_data))
    return networks
