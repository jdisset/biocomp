from typing import (
    Any,
    Optional,
    Tuple,
)
from pydantic import BaseModel
import pandas as pd

from biocomp.recipe_new import Recipe, TranscriptionUnit, CoTxList
from biocomp.logging_config import get_logger
from biocomp.library import PartsLibrary, LibraryContext
from biocomp.graphengine import GraphState, GraphNode, GraphEdge, Part, InverseSpec
from biocomp.graphrules import GraphRewritingRule


logger = get_logger(__name__)


class Network(BaseModel):
    """Pure data container for network definitions"""

    cotx: CoTxList = []
    name: Optional[str] = None

    @property
    def central_dogma_graph(self) -> pd.DataFrame:
        return build_central_dogma_graph(self, LibraryContext.get_library())


def recipe_to_network(
    recipe, rules: list[GraphRewritingRule], lib: PartsLibrary, **kwargs
) -> list[Network]: ...


def network_to_recipe(network: Network) -> Recipe: ...


class NetworkConstructionError(Exception):
    """Exception for errors during network construction"""

    pass


def graphstate_to_cdg_df(graph: GraphState) -> pd.DataFrame:
    """Converts a GraphState object to a Central Dogma Graph DataFrame."""
    node_data = {
        node.node_id: {
            "type": node.node_type,
            **node.extra,
            "predecessor": [],
            "successor": [],
        }
        for node in graph.nodes.values()
    }

    for edge in graph.edges.values():
        if edge.source_id in node_data and edge.target_id in node_data:
            node_data[edge.source_id]["successor"].append(edge.target_id)
            node_data[edge.target_id]["predecessor"].append(edge.source_id)

    for nid in node_data:
        if not node_data[nid]["predecessor"]:
            node_data[nid]["predecessor"] = None
        if not node_data[nid]["successor"]:
            node_data[nid]["successor"] = None

    return pd.DataFrame.from_dict(node_data, orient="index")


def _get_dna(tu: TranscriptionUnit, lib: PartsLibrary) -> Tuple[list[str], dict[str, list[str]]]:
    content = [s.part for s in tu.slots if s.maps_to_parameter is None and s.part is not None]
    return content, tu.params


def _get_downstream(tu: TranscriptionUnit, transform: str, lib: PartsLibrary):
    dna_content, dna_params = _get_dna(tu, lib)
    if not dna_content:
        return (), {}

    d = lib.pc.loc[dna_content]
    content = tuple(d[d[transform] == 1].index)
    params = {}
    for param_name, parts in dna_params.items():
        non_none_parts = [p for p in parts if p is not None]
        if non_none_parts:
            p = lib.pc.loc[non_none_parts]
            if p[transform].sum() > 0:
                params[param_name] = list(p[p[transform] == 1].index)
    return content, params


def _get_rna(tu: TranscriptionUnit, lib: PartsLibrary):
    return _get_downstream(tu, "transcripted", lib)


def _get_prt(tu: TranscriptionUnit, lib: PartsLibrary):
    return _get_downstream(tu, "translated", lib)


def _make_hashable(params, tu_obj):
    hashable_params = {}
    for param_name, parts in params.items():
        ref_id = tu_obj.param_ref_ids.get(param_name)
        if ref_id is not None:
            hashable_params[param_name] = (f"ref:{ref_id}",)
        else:
            hashable_params[param_name] = tuple(parts) if isinstance(parts, list) else (parts,)
    return tuple(sorted((k, v) for k, v in hashable_params.items()))


def preprocess_network_tus(network: Network, lib: PartsLibrary) -> dict[str, Any]:
    """
    Parses the network recipe and returns a dictionary with all necessary
    pre-calculated information for building either the primal or dual graph.
    """
    if not network.cotx:
        return {}

    tu_map = {}
    tu_to_cotx_map = {}
    global_tu_index = 0

    for cotx_index, cotx_group in enumerate(network.cotx):
        group_name = cotx_group.name or f"cotx_{cotx_index + 1}"
        for unit_index, unit in enumerate(cotx_group.units):
            # Create unique TUID that includes cotx group info for duplicate plasmids
            base_name = unit.name or f"TU_{global_tu_index}"
            tuid = f"{base_name}_{group_name}"
            tu_map[tuid] = unit
            tu_to_cotx_map[tuid] = group_name
            global_tu_index += 1

    tu_info = {}
    for tuid, tu in tu_map.items():
        dna_content, dna_params = _get_dna(tu, lib)
        rna_content, rna_params = _get_rna(tu, lib)
        prt_content, prt_params = _get_prt(tu, lib)
        tu_info[tuid] = {
            "tu": tu,
            "cotx_group": tu_to_cotx_map[tuid],
            "DNA_content": dna_content,
            "DNA_params": dna_params,
            "RNA_content": rna_content,
            "RNA_params": rna_params,
            "PRT_content": prt_content,
            "PRT_params": prt_params,
            "RNA_params_hashable": _make_hashable(rna_params, tu),
            "PRT_params_hashable": _make_hashable(prt_params, tu),
        }

    def has_multi_value_params(tu):
        return any(isinstance(p, list) and len(p) > 1 for p in tu.params.values())

    is_committed = any(has_multi_value_params(tu) for tu in tu_map.values()) and any(
        not has_multi_value_params(tu) for tu in tu_map.values()
    )

    return {"network": network, "tu_map": tu_map, "tu_info": tu_info, "is_committed": is_committed}


def _build_cdg_primal_from_preprocessed(
    preprocessed_data: dict[str, Any], lib: PartsLibrary, custom_outputs_parts=None
) -> GraphState:
    """Builds the primal CDG where nodes are biological entities (DNA, RNA, PRT)."""
    tu_info = preprocessed_data["tu_info"]
    is_committed = preprocessed_data["is_committed"]

    nodes, next_node_id = [], 0
    group_to_node_id, tu_to_node_id = {}, {}

    for tuid, info in tu_info.items():
        dna_group_key = ("DNA", tuid)
        rna_group_key = (
            "RNA",
            info["RNA_content"]
            if is_committed
            else (info["RNA_content"], info["RNA_params_hashable"]),
        )
        prt_group_key = (
            "PRT",
            info["PRT_content"]
            if is_committed
            else (info["PRT_content"], info["PRT_params_hashable"]),
        )

        for key in [dna_group_key, rna_group_key, prt_group_key]:
            if key not in group_to_node_id:
                group_to_node_id[key] = next_node_id
                next_node_id += 1
            tu_to_node_id[(key[0], tuid)] = group_to_node_id[key]

    node_id_to_info = {nid: {"type": key[0], "tu_ids": []} for key, nid in group_to_node_id.items()}
    for (type, tuid), nid in tu_to_node_id.items():
        node_id_to_info[nid]["tu_ids"].append(tuid)

    outputs = (custom_outputs_parts or []) + lib.parts[
        lib.parts.category == "fluo_marker"
    ].index.tolist()

    for nid, info in node_id_to_info.items():
        rep_tuid = info["tu_ids"][0]
        content = tu_info[rep_tuid][f"{info['type']}_content"]
        params = tu_info[rep_tuid][f"{info['type']}_params"]
        is_output = info["type"] == "PRT" and any(o in content for o in outputs)
        extra = {
            "tu_id": info["tu_ids"],
            "content": content,
            "params": params,
            "is_output": is_output,
            "is_input": None,
            "content_type": tuple(lib.parts.loc[p].iloc[0] for p in content) if content else (),
        }
        nodes.append(GraphNode(node_id=nid, node_type=info["type"], extra=extra))

    edges = []
    for tuid in tu_info:
        edges.append(
            GraphEdge(
                source_id=tu_to_node_id[("DNA", tuid)],
                target_id=tu_to_node_id[("RNA", tuid)],
                output_slot=0,
                input_slot=0,
                content=(),
            )
        )

    rna_nodes = [n for n in nodes if n.node_type == "RNA"]
    prt_nodes = [n for n in nodes if n.node_type == "PRT"]
    for rna_node in rna_nodes:
        rna_tu_set = set(rna_node.extra["tu_id"])
        for prt_node in prt_nodes:
            if rna_tu_set.issubset(set(prt_node.extra["tu_id"])):
                edges.append(
                    GraphEdge(
                        source_id=rna_node.node_id,
                        target_id=prt_node.node_id,
                        output_slot=0,
                        input_slot=0,
                        content=(),
                    )
                )

    unique_edges = {(e.source_id, e.target_id): e for e in edges}.values()
    nodes_dict = {n.node_id: n for n in nodes}
    edges_dict = {(e.source_id, e.target_id, e.output_slot, e.input_slot): e for e in unique_edges}
    return GraphState(nodes=nodes_dict, edges=edges_dict)


def _build_cdg_dual_from_preprocessed(
    preprocessed_data: dict[str, Any], lib: PartsLibrary, custom_outputs_parts=None
) -> GraphState:
    """Builds the dual CDG where nodes are transformations."""
    network = preprocessed_data["network"]
    tu_info = preprocessed_data["tu_info"]
    is_committed = preprocessed_data["is_committed"]
    outputs_list = (custom_outputs_parts or []) + lib.parts[
        lib.parts.category == "fluo_marker"
    ].index.tolist()

    nodes, edges = [], []
    next_node_id = 0

    # Create a mapping from (source_id, cotx_group) to normalized ratio
    source_cotx_to_ratio_map: dict[tuple[str | None, str], float] = {}
    for i, cotx in enumerate(network.cotx or []):
        group_name = cotx.name or f"cotx_{i + 1}"
        raw_ratios = cotx.ratios or [1.0] * len(cotx.units)
        ratio_sum = sum(raw_ratios)
        # Normalize ratios within each cotx group to sum to 1.0
        normalized_ratios = (
            [r / ratio_sum for r in raw_ratios]
            if ratio_sum > 0
            else [1.0 / len(cotx.units)] * len(cotx.units)
        )

        for unit, ratio in zip(cotx.units, normalized_ratios):
            source_cotx_to_ratio_map[(unit.source, group_name)] = float(ratio)

    source_nodes, tx_nodes, tl_nodes = {}, {}, {}
    output_node, dead_end_nodes = None, {}

    for (source_id, cotx_group), ratio in source_cotx_to_ratio_map.items():
        source_key = (source_id, cotx_group)
        if source_key not in source_nodes:
            source_nodes[source_key] = next_node_id
            next_node_id += 1

            source_extra = {
                "source_id": source_id,
                "cotx_group": cotx_group,
                "ratio": ratio,
            }
            nodes.append(
                GraphNode(node_id=source_nodes[source_key], node_type="source", extra=source_extra)
            )

    for tuid, info in tu_info.items():
        tu = info["tu"]

        rna_key = (
            info["RNA_content"]
            if is_committed
            else (info["RNA_content"], info["RNA_params_hashable"])
        )
        if rna_key not in tx_nodes:
            tx_nodes[rna_key] = next_node_id
            next_node_id += 1
            nodes.append(GraphNode(node_id=tx_nodes[rna_key], node_type="transcription", extra={}))

        prt_key = (
            info["PRT_content"]
            if is_committed
            else (info["PRT_content"], info["PRT_params_hashable"])
        )
        if prt_key not in tl_nodes:
            tl_nodes[prt_key] = next_node_id
            next_node_id += 1
            nodes.append(GraphNode(node_id=tl_nodes[prt_key], node_type="translation", extra={}))

    output_slot_counter = 0
    for tuid, info in tu_info.items():
        tu = info["tu"]

        cotx_group = info["cotx_group"]
        source_key = (tu.source, cotx_group)
        src_id = source_nodes[source_key]
        rna_key = (
            info["RNA_content"]
            if is_committed
            else (info["RNA_content"], info["RNA_params_hashable"])
        )
        tx_id = tx_nodes[rna_key]

        source_output_slot = getattr(tu, "position_in_source", 0) or 0

        edges.append(
            GraphEdge(
                source_id=src_id,
                target_id=tx_id,
                output_slot=source_output_slot,
                input_slot=0,
                content_type="DNA",
                content=tuple(Part(name=p, category="DNA") for p in info["DNA_content"]),
                content_embedding_names={k: tuple(v) for k, v in info["DNA_params"].items()},
                extra={"tu_id": [tuid]},
            )
        )

        prt_key = (
            info["PRT_content"]
            if is_committed
            else (info["PRT_content"], info["PRT_params_hashable"])
        )
        tl_id = tl_nodes[prt_key]
        edges.append(
            GraphEdge(
                source_id=tx_id,
                target_id=tl_id,
                output_slot=0,
                input_slot=0,
                content_type="RNA",
                content=tuple(Part(name=p, category="RNA") for p in info["RNA_content"]),
                content_embedding_names={k: tuple(v) for k, v in info["RNA_params"].items()},
            )
        )

        is_output = any(o in info["PRT_content"] for o in outputs_list)
        target_node_id = -1
        input_slot = 0

        if is_output:
            if output_node is None:
                output_node = next_node_id
                next_node_id += 1
                nodes.append(GraphNode(node_id=output_node, node_type="output", extra={}))
            target_node_id = output_node
            input_slot = output_slot_counter
            output_slot_counter += 1
        else:
            prt_content_tuple = tuple(sorted(info["PRT_content"]))
            if prt_content_tuple not in dead_end_nodes:
                dead_end_nodes[prt_content_tuple] = next_node_id
                next_node_id += 1
                nodes.append(
                    GraphNode(
                        node_id=dead_end_nodes[prt_content_tuple], node_type="deadend", extra={}
                    )
                )
            target_node_id = dead_end_nodes[prt_content_tuple]

        edges.append(
            GraphEdge(
                source_id=tl_id,
                target_id=target_node_id,
                output_slot=0,
                input_slot=input_slot,
                content_type="PRT",
                content=tuple(Part(name=p, category="PRT") for p in info["PRT_content"]),
                content_embedding_names={k: tuple(v) for k, v in info["PRT_params"].items()},
            )
        )

    unique_edges_dict = {}
    for e in edges:
        if e.content_type == "DNA":
            key = (e.source_id, e.output_slot)
        else:
            key = (e.source_id, e.target_id, e.content_type)
        if key not in unique_edges_dict:
            unique_edges_dict[key] = e

    nodes_dict = {n.node_id: n for n in nodes}
    edges_dict = {
        (e.source_id, e.target_id, e.output_slot, e.input_slot): e
        for e in unique_edges_dict.values()
    }
    return GraphState(nodes=nodes_dict, edges=edges_dict)


def build_central_dogma_graph_direct(
    network: Network, lib: PartsLibrary, custom_outputs_parts=None, dual: bool = True
) -> GraphState:
    """
    Builds a central dogma graph directly from a Network definition into a GraphState.
    Args:
        network: The Network object defining the recipe.
        lib: The parts library.
        custom_outputs_parts: Optional list of part names to be considered outputs.
        dual: If False (default), builds the primal graph where nodes are biological
              entities (DNA, RNA, PRT). If True, builds the dual graph where nodes
              are transformations (Source, Transcription, Translation).

    """
    preprocessed_data = preprocess_network_tus(network, lib)
    if not preprocessed_data:
        return GraphState(nodes={}, edges={})

    if dual:
        return _build_cdg_dual_from_preprocessed(preprocessed_data, lib, custom_outputs_parts)
    else:
        return _build_cdg_primal_from_preprocessed(preprocessed_data, lib, custom_outputs_parts)


def build_central_dogma_graph(
    network: Network, lib: PartsLibrary, custom_outputs_parts=None
) -> pd.DataFrame:
    """
    Builds the primal central dogma graph and returns it as a pandas DataFrame
    for backward compatibility with the old API.
    """
    graph_state = build_central_dogma_graph_direct(network, lib, custom_outputs_parts, dual=False)
    return graphstate_to_cdg_df(graph_state)


##────────────────────────────────────────────────────────────────────────────}}}


def graphstate_to_compute_df(graph: GraphState) -> pd.DataFrame:
    """Converts a GraphState object to a Compute Graph DataFrame."""
    node_data = {
        node.node_id: {
            "type": node.node_type,
            "is_inverse_of": node.is_inverse_of,
            **node.extra,
            "input_from": [],
            "output_to": [],
        }
        for node in graph.nodes.values()
    }

    edges_by_source = {nid: [] for nid in node_data}
    edges_by_target = {nid: [] for nid in node_data}
    for edge in graph.edges.values():
        if edge.source_id in edges_by_source:
            edges_by_source[edge.source_id].append(edge)
        if edge.target_id in edges_by_target:
            edges_by_target[edge.target_id].append(edge)

    for nid, data in node_data.items():
        outgoing = sorted(edges_by_source.get(nid, []), key=lambda e: e.output_slot)
        data["output_to"] = [(e.target_id, e.input_slot) for e in outgoing]

        incoming = sorted(edges_by_target.get(nid, []), key=lambda e: e.input_slot)
        data["input_from"] = [(e.source_id, e.output_slot) for e in incoming]

    return pd.DataFrame.from_dict(node_data, orient="index")


def compute_df_to_graphstate(compute_df: pd.DataFrame) -> GraphState:
    """Converts a Compute Graph DataFrame to a GraphState object."""
    nodes: list[GraphNode] = []
    property_columns = ["cdg_input", "cdg_output", "extra", "source_id"]

    for idx, row in compute_df.iterrows():
        extra_properties = {
            col: row[col] for col in property_columns if col in row and pd.notna(row[col])
        }
        inverse_spec = row.get("is_inverse_of")
        if isinstance(inverse_spec, dict):
            inverse_spec = InverseSpec(**inverse_spec)

        node = GraphNode(
            node_id=int(idx),
            node_type=row["type"],
            is_inverse_of=inverse_spec,
            extra=extra_properties,
        )
        nodes.append(node)

    edges: list[GraphEdge] = []
    for idx, row in compute_df.iterrows():
        outputs = row.get("output_to")
        if outputs and pd.notna(outputs).all():
            for output_slot, (target_id, input_slot) in enumerate(outputs):
                edge = GraphEdge(
                    source_id=int(idx),
                    target_id=int(target_id),
                    output_slot=int(output_slot),
                    input_slot=int(input_slot),
                    content=(),
                )
                edges.append(edge)

    nodes_dict = {n.node_id: n for n in nodes}
    edges_dict = {(e.source_id, e.target_id, e.output_slot, e.input_slot): e for e in edges}
    return GraphState(nodes=nodes_dict, edges=edges_dict)


##────────────────────────────────────────────────────────────────────────────}}}


def old_network_compg_to_graphstate(old_network) -> GraphState:
    """
    Convert an old-style network.compute_graph (DataFrame) into a GraphState.

    - Nodes: reuse old compute_graph row index as node_id and row['type'] as node_type.
      Store all other compute_graph columns under node.extra with 'cg_' prefix to preserve info.
    - Edges: built from 'output_to' with slots. Biological edges (DNA/RNA/PRT) are enriched
      with content and content_embedding_names looked up from central_dogma_graph via cdg_input/cdg_output.
    """
    cg = getattr(old_network, "compute_graph", None)
    cdg = getattr(old_network, "central_dogma_graph", None)
    if cg is None or len(cg) == 0:
        return GraphState(nodes={}, edges={})

    def to_list(v):
        if v is None:
            return []
        if isinstance(v, (list, tuple)):
            return list(v)
        return [v]

    def parts_from_cdg_row(row: pd.Series, kind: str) -> tuple[Part, ...]:
        items = to_list(row.get("content"))
        return tuple(Part(name=str(p), category=kind) for p in items)

    def embeddings_from_cdg_row(row: pd.Series) -> dict[str, tuple[str, ...]]:
        p = row.get("params") or {}
        return {k: tuple(str(x) for x in to_list(v)) for k, v in p.items()}

    nodes: list[GraphNode] = []
    for nid, row in cg.iterrows():
        extra = {}
        for col, val in row.items():
            if col == "type":
                continue
            extra[f"cg_{col}"] = val

        node_type = str(row.get("type"))
        nodes.append(GraphNode(node_id=int(nid), node_type=node_type, extra=extra))

    def desired_content_type(src_type: str, dst_type: str) -> str | None:
        if src_type == "source" and dst_type == "transcription":
            return "DNA"
        if src_type == "transcription" and dst_type in ("translation", "sequestron_ERN"):
            return "RNA"
        if src_type == "translation" and dst_type in ("output", "sequestron_ERN"):
            return "PRT"
        if src_type == "sequestron_ERN" and dst_type == "translation":
            return "RNA"  # Sequestron ERN outputs RNA to translation
        return None

    def pick_cdg_row(
        src_row: pd.Series, dst_row: pd.Series, ctype: str, out_slot: int = 0
    ) -> pd.Series | None:
        if cdg is None or len(cdg) == 0:
            return None

        # First try to use the output slot to select from source's cdg_output
        src_outputs = to_list(src_row.get("cdg_output"))
        if src_outputs and out_slot < len(src_outputs):
            try:
                cid = int(src_outputs[out_slot])
                crow = cdg.loc[cid]
                if str(crow.get("type")) == ctype:
                    return crow
            except Exception:
                pass

        # Fallback to original logic
        candidate_ids: list[int] = []
        for key, r in (("cdg_output", src_row), ("cdg_input", dst_row)):
            for x in to_list(r.get(key)):
                try:
                    candidate_ids.append(int(x))
                except Exception:
                    pass
        for cid in candidate_ids:
            try:
                crow = cdg.loc[cid]
            except Exception:
                continue
            if str(crow.get("type")) == ctype:
                return crow
        return None

    edges: list[GraphEdge] = []
    for src_id, src_row in cg.iterrows():
        outputs = src_row.get("output_to")
        if not isinstance(outputs, list):
            continue
        for out_slot, pair in enumerate(outputs):
            try:
                dst_id, in_slot = pair
            except Exception:
                dst_id, in_slot = pair[0], 0
            dst_row = cg.loc[dst_id]
            ctype = desired_content_type(str(src_row.get("type")), str(dst_row.get("type")))

            kwargs = dict(
                source_id=int(src_id),
                target_id=int(dst_id),
                output_slot=int(out_slot),
                input_slot=int(in_slot),
                content=(),
            )
            if ctype is not None:
                crow = pick_cdg_row(src_row, dst_row, ctype, out_slot)
                if crow is not None:
                    kwargs["content"] = parts_from_cdg_row(crow, ctype)
                    kwargs["content_type"] = ctype
                    kwargs["content_embedding_names"] = embeddings_from_cdg_row(crow)
            elif str(src_row.get("type")) != "numeric":
                # When ctype is None and source is NOT numeric, try to find content
                # This handles edges to dead-end nodes (like output nodes)
                # Numeric edges should never have content (they're control flow edges)
                for possible_ctype in ("DNA", "RNA", "PRT"):
                    crow = pick_cdg_row(src_row, dst_row, possible_ctype, out_slot)
                    if crow is not None:
                        kwargs["content"] = parts_from_cdg_row(crow, possible_ctype)
                        kwargs["content_type"] = possible_ctype
                        kwargs["content_embedding_names"] = embeddings_from_cdg_row(crow)
                        break
            edges.append(GraphEdge(**kwargs))

    nodes_dict = {n.node_id: n for n in nodes}
    edges_dict = {(e.source_id, e.target_id, e.output_slot, e.input_slot): e for e in edges}
    return GraphState(nodes=nodes_dict, edges=edges_dict)


##────────────────────────────────────────────────────────────────────────────}}
