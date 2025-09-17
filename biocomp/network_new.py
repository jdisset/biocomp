from typing import (
    Any,
    Optional,
    Tuple,
    cast,
)
from pydantic import BaseModel
import pandas as pd

from biocomp.recipe_new import Recipe, CoTransfection, TranscriptionUnit, CoTxList
from biocomp.logging_config import get_logger
from biocomp.library import PartsLibrary, LibraryContext
from biocomp.graphengine import GraphState, GraphNode, GraphEdge, Part, InverseSpec
from biocomp.graphrules import GraphRewritingRule


logger = get_logger(__name__)


class Network(BaseModel):
    """Pure data container for network definitions"""

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    cotx: CoTxList = []
    name: Optional[str] = None
    invert_on_build: bool = False
    compute_graph: Optional[GraphState] = None

    @property
    def central_dogma_graph(self) -> pd.DataFrame:
        return build_central_dogma_graph(self, LibraryContext.get_library())


def recipe_to_network(
    recipe, rules: list[GraphRewritingRule], lib: PartsLibrary, **kwargs
) -> list[Network]: ...


def network_to_recipe(network: Network) -> Recipe:
    ...
    # should round-trip perfectly!!


class NetworkConstructionError(Exception):
    """Exception for errors during network construction"""

    pass


## {{{              --     central dogma graph expansions     --


def build_central_dogma_graph(
    network: Network, lib: PartsLibrary, custom_outputs_parts=None
) -> pd.DataFrame:
    """
    Build a central dogma graph directly from a Network.
    """
    # Extract transcription units from cotx
    transcription_units = {}
    for group_idx, group in enumerate(network.cotx or []):
        for unit_idx, unit in enumerate(group.units):
            tu_name = unit.name or f"TU_{len(transcription_units) + 1}"
            transcription_units[tu_name] = unit

    source_to_cotx_map: dict[str | None, str] = {}
    source_to_ratio_map: dict[str | None, float] = {}
    for i, cotx in enumerate(network.cotx or []):
        group_name = cotx.name or f"cotx_{i + 1}"
        for unit, ratio in zip(cotx.units, cotx.ratios or []):
            # map each source to its cotx group and ratio
            source_to_cotx_map[unit.source] = group_name
            source_to_ratio_map[unit.source] = float(ratio)

    def make_hashable(params, tu_obj):
        """Make params hashable, considering ref_id for identical part grouping."""
        hashable_params = {}
        for param_name, parts in params.items():
            ref_id = tu_obj.param_ref_ids.get(param_name)
            if ref_id is not None:
                hashable_params[param_name] = (f"ref:{ref_id}",)
            else:
                hashable_params[param_name] = tuple(parts) if isinstance(parts, list) else (parts,)
        return tuple(sorted((k, v) for k, v in hashable_params.items()))

    def get_dna(tu: TranscriptionUnit) -> Tuple[list[str], dict[str, list[str]]]:
        content = []
        for s in tu.slots:
            if s.maps_to_parameter is None and s.part is not None:
                content.append(s.part)
        return content, tu.params

    def get_downstream(tu: TranscriptionUnit, transform: str):
        dna_content, dna_params = get_dna(tu)
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

    def get_rna(tu: TranscriptionUnit):
        return get_downstream(tu, transform="transcripted")

    def get_prt(tu: TranscriptionUnit):
        return get_downstream(tu, transform="translated")

    tu_data: list[dict] = []
    assert transcription_units is not None, "No transcription units in network"

    for tuid, t in transcription_units.items():
        dna, dna_params = get_dna(t)
        rna, rna_params = get_rna(t)
        prt, prt_params = get_prt(t)

        source_id = t.source
        cotx_group = source_to_cotx_map.get(source_id)

        tu_data.append(
            {
                "name": tuid,
                "source_id": source_id,
                "cotx_group": cotx_group,
                "DNA": dna,
                "DNA_params": dna_params,
                "DNA_params_hashable": make_hashable(dna_params, t),
                "RNA": rna,
                "RNA_params": rna_params,
                "RNA_params_hashable": make_hashable(rna_params, t),
                "PRT": prt,
                "PRT_params": prt_params,
                "PRT_params_hashable": make_hashable(prt_params, t),
            }
        )

    assert tu_data, "No transcription units in network"
    tudf = pd.DataFrame(tu_data)

    def only_one_value_per_param(params: dict[str, list[str]]) -> bool:
        return all(len(parts) <= 1 for _, parts in params.items())

    def has_non_null_ref_id(row_name: str) -> bool:
        tu_obj = transcription_units[row_name]
        return any(ref_id is not None for ref_id in tu_obj.param_ref_ids.values())

    def group_multi_param_tus(df: pd.DataFrame, node_type: str, params_col: str) -> list[list[str]]:
        grouped_tuids = []
        for _, row in df.iterrows():
            if has_non_null_ref_id(row["name"]):
                group_key = (row[node_type], row[f"{node_type}_params_hashable"])
                for i, (key, names) in enumerate(grouped_tuids):
                    if key == group_key:
                        grouped_tuids[i] = (key, names + [row["name"]])
                        break
                else:
                    grouped_tuids.append((group_key, [row["name"]]))
            else:
                grouped_tuids.append((None, [row["name"]]))
        return [names for _, names in grouped_tuids]

    def process_node_type(node_type: str, params_col: str) -> pd.DataFrame:
        def is_likely_quantized_tu(tu):
            for param_name, parts in tu.params.items():
                if (
                    param_name in ["tl_rate", "tc_rate"]
                    and isinstance(parts, list)
                    and len(parts) == 1
                ):
                    if param_name == "tl_rate" and parts[0] != "00_empty_tc":
                        return True
                    if param_name == "tc_rate" and parts[0] not in ["hEF1a"]:
                        return True
            return False

        quantized_tus = [
            tu_name for tu_name, tu in transcription_units.items() if is_likely_quantized_tu(tu)
        ]
        multi_value_tus = [
            tu_name
            for tu_name, tu in transcription_units.items()
            if any(
                isinstance(parts, list) and len(parts) > 1 and param_name in ["tl_rate", "tc_rate"]
                for param_name, parts in tu.params.items()
            )
        ]

        network_is_committed = len(quantized_tus) > 0 and len(multi_value_tus) > 0

        if network_is_committed:
            tu_ids = list(tudf.groupby(by=node_type).agg(list).name)
            return pd.DataFrame({"tu_id": tu_ids, "type": node_type})
        else:
            no_params = list(
                tudf[tudf[params_col].map(len) == 0].groupby(by=node_type).agg(list).name
            )
            try:
                one_param = (
                    tudf[tudf[params_col].map(len) > 0]
                    .groupby(by=node_type)
                    .filter(
                        lambda x: all(only_one_value_per_param(params) for params in x[params_col])
                    )
                    .groupby(by=[node_type, f"{node_type}_params_hashable"])
                    .agg(list)
                )
                one_param = [] if one_param.empty else list(one_param.name)
            except Exception as e:
                raise NetworkConstructionError(
                    f"Error grouping {node_type} with one param: {e}\ntudf:\n{tudf}"
                )
            has_many = tudf[params_col].apply(
                lambda params: any(len(v) > 1 for v in params.values())
            )
            many_param = (
                group_multi_param_tus(tudf[has_many], node_type, params_col)
                if has_many.any()
                else []
            )
            tu_ids = no_params + one_param + many_param
            return pd.DataFrame({"tu_id": tu_ids, "type": node_type})

    dna_df = pd.DataFrame({"tu_id": [[x] for x in cast(str, tudf["name"])], "type": "DNA"})
    rna_df = process_node_type("RNA", "RNA_params")
    prt_df = process_node_type("PRT", "PRT_params")
    tudf.set_index("name", inplace=True)

    cdg = pd.concat([dna_df, rna_df, prt_df], sort=False).reset_index(drop=True)
    cdg["predecessor"] = None
    cdg["successor"] = None

    dna_nodes = cdg[cdg.type == "DNA"]
    rna_nodes = cdg[cdg.type == "RNA"]
    if len(rna_nodes) == 0:
        raise NetworkConstructionError("No RNA nodes in central dogma graph")

    for i, r in dna_nodes.iterrows():
        successors = []
        for ii, rr in rna_nodes.iterrows():
            if r.tu_id[0] in rr.tu_id:
                successors.append(ii)
        cdg.at[i, "successor"] = successors

    for i_r, rna in rna_nodes.iterrows():
        successors = []
        for i_p, prt in cdg[cdg.type == "PRT"].iterrows():
            if set(rna.tu_id).issubset(set(prt.tu_id)):
                successors.append(i_p)
        cdg.at[i_r, "successor"] = successors

    cdg["predecessor"] = [list() for _ in range(len(cdg))]
    for i, r in cdg.iterrows():
        if r.successor is not None:
            for s in r.successor:
                cdg.loc[s, "predecessor"].append(i)
    cdg.loc[~cdg.predecessor.astype(bool), "predecessor"] = None

    logger.debug(f"cdg: \n{cdg}\n")

    try:
        cdg["content"] = cdg.apply(lambda x: tudf.loc[x.tu_id[0]][x.type], axis=1)
        cdg["params"] = cdg.apply(lambda x: tudf.loc[x.tu_id[0]][x.type + "_params"], axis=1)
        cdg["content_type"] = cdg.apply(
            lambda x: tuple([lib.parts.loc[p].iloc[0] for p in x.content]), axis=1
        )
    except Exception as e:
        msg = f"Error while building central dogma graph. Error: {e}"
        msg += f"\ntudf: \n{tudf}"
        msg += f"\n\ncdg: \n{cdg}"
        raise NetworkConstructionError(msg)

    cdg["source_id"] = None
    cdg["cotx_group"] = None
    cdg["cotx_ratio"] = None
    dna_mask = cdg["type"] == "DNA"
    cdg.loc[dna_mask, "source_id"] = cdg[dna_mask].apply(
        lambda row: tudf.loc[row["tu_id"][0]]["source_id"], axis=1
    )
    cdg.loc[dna_mask, "cotx_group"] = cdg[dna_mask].apply(
        lambda row: source_to_cotx_map.get(tudf.loc[row["tu_id"][0]]["source_id"]), axis=1
    )
    cdg.loc[dna_mask, "cotx_ratio"] = cdg[dna_mask].apply(
        lambda row: source_to_ratio_map.get(tudf.loc[row["tu_id"][0]]["source_id"]), axis=1
    )

    outputs = (custom_outputs_parts if custom_outputs_parts is not None else []) + lib.parts[
        lib.parts["category"] == "fluo_marker"
    ].index.tolist()

    containsOutput = lambda l, outputs: any([o in l for o in outputs])
    cdg["is_output"] = False
    cdg.loc[cdg.type == "PRT", "is_output"] = cdg.loc[cdg.type == "PRT"].tu_id.apply(
        lambda x: containsOutput(tudf.loc[x].PRT.tolist()[0], outputs)
    )
    cdg["is_input"] = None

    return cdg


def build_central_dogma_graph_from_units(
    transcription_units: dict[str, TranscriptionUnit], lib: PartsLibrary, custom_outputs_parts=None
) -> pd.DataFrame:
    """
    Backward compatibility function - builds CDG from transcription units dict.
    Creates a temporary Network and uses the main function.
    """
    # Create units from transcription units
    units = []
    for tu_name, tu in transcription_units.items():
        units.append(tu)

    # Create temporary network with single CoTransfection
    temp_network = Network(cotx=[CoTransfection(units=units)])

    # Use the main function
    return build_central_dogma_graph(temp_network, lib, custom_outputs_parts)


def _is_valid_property(value: Any) -> bool:
    """
    Checks if a property value is valid for inclusion in the 'extra' dict.
    Excludes standalone None or NaN, but allows list-like containers.
    """
    if isinstance(value, (list, tuple, pd.Series)):
        return True
    return pd.notna(value)


def cdg_df_to_graphstate(cdg_df: pd.DataFrame) -> GraphState:
    """Converts a Central Dogma Graph DataFrame to a GraphState object."""
    nodes: list[GraphNode] = []
    property_columns = [
        "tu_id",
        "content",
        "content_type",
        "params",
        "is_output",
        "is_input",
        "source_id",
        "cotx_group",
    ]

    for idx, row in cdg_df.iterrows():
        extra_properties = {
            col: row[col] for col in property_columns if col in row and _is_valid_property(row[col])
        }
        node = GraphNode(node_id=int(idx), node_type=row["type"], extra=extra_properties)
        nodes.append(node)

    edges: list[GraphEdge] = []
    for idx, row in cdg_df.iterrows():
        successors = row.get("successor")
        if isinstance(successors, list) and successors:
            for target_idx in successors:
                # At CDG stage, edges should be minimal - content is stored in nodes
                # Content will be populated by transformation rules as needed
                edge = GraphEdge(
                    source_id=int(idx),
                    target_id=int(target_idx),
                    output_slot=0,
                    input_slot=0,
                    content=(),  # Start with empty content
                    content_type=None,  # Will be set by transformation rules
                )
                edges.append(edge)

    return GraphState(nodes=nodes, edges=edges)


def cdg_df_to_dual_graphstate(cdg_df: pd.DataFrame) -> GraphState:
    """
    Converts a Central Dogma Graph DataFrame to its dual GraphState representation.

    In the dual graph:
    - Nodes represent transformations/interactions (Source, Transcription, Translation, output)
    - Edges carry biological content (DNA, RNA, PRT parts) between transformations

    This creates the base graph structure for each TU:
    [Source] --{DNA edge with content}--> [Transcription] --{RNA edge w content}--> [Translation] --{PRT edge w content}--> [output]
    """
    # Empty → empty dual graph
    if cdg_df is None or len(cdg_df) == 0:
        return GraphState(nodes=[], edges=[])

    # Helpers
    def to_list(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return []
        if isinstance(v, (list, tuple)):
            return list(v)
        return [v]

    def parts(row, kind: str) -> tuple[Part, ...]:
        return tuple(
            Part(name=str(p), category=kind) for p in map(str, to_list(row.get("content")))
        )

    def embeddings(row) -> dict[str, tuple[str, ...]]:
        p = row.get("params") or {}
        return {k: tuple(str(x) for x in to_list(v)) for k, v in p.items()}

    # Split by biological type
    dna_df = cdg_df[cdg_df["type"] == "DNA"] if "type" in cdg_df else pd.DataFrame()
    rna_df = cdg_df[cdg_df["type"] == "RNA"] if "type" in cdg_df else pd.DataFrame()
    prt_df = cdg_df[cdg_df["type"] == "PRT"] if "type" in cdg_df else pd.DataFrame()

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    next_id = 0

    def nid() -> int:
        nonlocal next_id
        i = next_id
        next_id += 1
        return i

    # Node registries (by CDG row index)
    src_by_dna: dict[int, int] = {}
    tx_by_rna: dict[int, int] = {}

    # Build transcription per RNA row: merges DNA that produce same RNA
    for rna_idx, rna_row in rna_df.iterrows():
        # Transcription node (one per RNA group)
        tx_id = nid()
        tx_by_rna[rna_idx] = tx_id
        nodes.append(GraphNode(node_id=tx_id, node_type="transcription", extra={}))

        # Ensure a source per predecessor DNA row, wire DNA → transcription
        for dna_idx in to_list(rna_row.get("predecessor")):
            if dna_idx not in src_by_dna:
                dna_row = dna_df.loc[dna_idx]
                s_id = nid()
                src_by_dna[dna_idx] = s_id
                nodes.append(
                    GraphNode(
                        node_id=s_id,
                        node_type="source",
                        extra={
                            "source_id": dna_row.get("source_id"),
                            "cotx_group": dna_row.get("cotx_group"),
                            "ratio": dna_row.get("cotx_ratio"),
                        },
                    )
                )
            # DNA edge content + metadata belong to the edge
            dna_row = dna_df.loc[dna_idx]
            edges.append(
                GraphEdge(
                    source_id=src_by_dna[dna_idx],
                    target_id=tx_id,
                    output_slot=0,
                    input_slot=0,
                    content=parts(dna_row, "DNA"),
                    content_type="DNA",
                    content_embedding_names=embeddings(dna_row),
                    extra={
                        "tu_id": [str(t) for t in to_list(dna_row.get("tu_id"))],
                    },
                )
            )

    # Single output node if any PRT is an output
    output_id = None
    if len(prt_df) and "is_output" in prt_df.columns and bool((prt_df["is_output"] == True).any()):
        output_id = nid()
        nodes.append(GraphNode(node_id=output_id, node_type="output", extra={}))

    # Create translation per PRT row; wire RNA → translation using PRT predecessors
    for prt_idx, prt_row in prt_df.iterrows():
        # Translation node representing this protein
        tl_id = nid()
        nodes.append(GraphNode(node_id=tl_id, node_type="translation", extra={}))

        # Wire all RNA predecessors (via their transcription) to this translation
        for rna_idx in to_list(prt_row.get("predecessor")):
            tx_id = tx_by_rna.get(rna_idx)
            if tx_id is None:
                continue
            rna_row = rna_df.loc[rna_idx]
            edges.append(
                GraphEdge(
                    source_id=tx_id,
                    target_id=tl_id,
                    output_slot=0,
                    input_slot=0,
                    content=parts(rna_row, "RNA"),
                    content_type="RNA",
                    content_embedding_names=embeddings(rna_row),
                )
            )

        # PRT edge to output or dead_end
        is_out = bool(prt_row.get("is_output", False))
        target = output_id if (is_out and output_id is not None) else nid()
        if target != output_id:
            nodes.append(GraphNode(node_id=target, node_type="dead_end", extra={}))
        edges.append(
            GraphEdge(
                source_id=tl_id,
                target_id=target,
                output_slot=0,
                input_slot=0,
                content=parts(prt_row, "PRT"),
                content_type="PRT",
                content_embedding_names=embeddings(prt_row),
            )
        )

    return GraphState(nodes=nodes, edges=edges)


def graphstate_to_cdg_df(graph: GraphState) -> pd.DataFrame:
    """Converts a GraphState object to a Central Dogma Graph DataFrame."""
    node_data = {
        node.node_id: {
            "type": node.node_type,
            **node.extra,
            "predecessor": [],
            "successor": [],
        }
        for node in graph.nodes
    }

    for edge in graph.edges:
        if edge.source_id in node_data and edge.target_id in node_data:
            node_data[edge.source_id]["successor"].append(edge.target_id)
            node_data[edge.target_id]["predecessor"].append(edge.source_id)

    for nid in node_data:
        if not node_data[nid]["predecessor"]:
            node_data[nid]["predecessor"] = None
        if not node_data[nid]["successor"]:
            node_data[nid]["successor"] = None

    return pd.DataFrame.from_dict(node_data, orient="index")


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                       --     compute graph     --


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
        for node in graph.nodes
    }

    edges_by_source = {nid: [] for nid in node_data}
    edges_by_target = {nid: [] for nid in node_data}
    for edge in graph.edges:
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
                    content=(),  # Compute graph edges are abstract in this format
                )
                edges.append(edge)

    return GraphState(nodes=nodes, edges=edges)


##────────────────────────────────────────────────────────────────────────────}}}
