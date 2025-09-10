from typing import (
    List,
    Dict,
    Iterable,
    Optional,
    Annotated,
    Union,
    Tuple,
    cast,
)
from pydantic import BaseModel, BeforeValidator, model_validator
from .utils import load_lib, flatten
import pandas as pd

from biocomp.logging_config import get_logger
from biocomp.library import PartsLibrary
from biocomp.graphengine import GraphState, GraphNode, GraphEdge, Part, apply_rule
from biocomp.graphrules import GraphRewritingRule

logger = get_logger(__name__)

PART_TYPE_TO_EMBEDDING_NAME = {"promoter": "tc_rate", "uORF_group": "tl_rate"}
EMBEDDING_TO_DEFAULT_PART = {"tl_rate": "00_empty_tc", "tc_rate": "hEF1a"}

## {{{                      --     Library Context     --


class LibraryContext:
    _current_lib = None

    @classmethod
    def set_library(cls, lib):
        cls._current_lib = lib

    @classmethod
    def get_library(cls):
        if cls._current_lib is None:
            return load_lib()
        return cls._current_lib

    @classmethod
    def with_library(cls, lib):
        """Context manager for temporarily setting a library"""

        class LibraryContextManager:
            def __init__(self, lib):
                self.lib = lib
                self.previous_lib = None

            def __enter__(self):
                self.previous_lib = LibraryContext._current_lib
                LibraryContext.set_library(self.lib)
                return self.lib

            def __exit__(self, exc_type, exc_val, exc_tb):
                LibraryContext.set_library(self.previous_lib)

        return LibraryContextManager(lib)


##────────────────────────────────────────────────────────────────────────────}}}


## {{{                           --     Slot     --
class Slot(BaseModel):
    """Transcription Units are made of slots which contain either a part or a list of
    possible parts that map to a quantized parameter"""

    part: Optional[Union[str, List[Optional[str]]]] = None

    # does this slot map to a parameter, like "tl_rate" or "tc_rate"?
    maps_to_parameter: Optional[str] = None

    # unique identifier for shared ("linked") parts across transcription units
    ref_id: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _wrap_plain_part(cls, value):
        # if Slot("foo") or Slot(["a","b"]), turn that into {"part": ...}
        if not isinstance(value, dict):
            if isinstance(value, str):
                value = {"part": value}
            return {"part": list(value)}
        return value

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)

        if isinstance(self.part, list):
            if not self.part or self.part == [None]:
                self.part = None
            else:
                mapped = list(set([self.__mapped_parameter(p) for p in self.part if p is not None]))
                # filter out None values (parts that don't map to any parameter)
                non_none_mapped = [m for m in mapped if m is not None]
                if len(non_none_mapped) > 1:
                    raise ValueError(
                        f"{self.part} maps to {len(non_none_mapped)} different parameters ({non_none_mapped})"
                    )
                self.maps_to_parameter = non_none_mapped[0] if non_none_mapped else None
        else:
            self.maps_to_parameter = self.__mapped_parameter(self.part)

        if self.maps_to_parameter is not None and not isinstance(self.part, list):
            self.part = [self.part]  # type: ignore

    def __mapped_parameter(self, part_name: Optional[str]) -> Optional[str]:
        """Returns the name of the parameter a part maps to, or None if it doesn't map to any"""
        lib = LibraryContext.get_library()
        if part_name is not None:
            if part_name in lib.pc.index:
                category = lib.pc.loc[part_name, "category"]
                if category in PART_TYPE_TO_EMBEDDING_NAME:
                    return PART_TYPE_TO_EMBEDDING_NAME[category]
            else:
                raise ValueError(
                    f'Unknown part: "{part_name}" (type: {type(part_name)}),library: {lib}'
                )
        return None

    def __repr__(self) -> str:
        if self.maps_to_parameter is None:
            if self.part is None:
                return "<empty slot>"
            else:
                return f"<{self.part}>"
        return f"<{self.part} -> {self.maps_to_parameter}>"


def convert_to_slot(value):
    """Convert strings or lists of strings to Slot objects"""
    if isinstance(value, Slot):
        return value
    elif isinstance(value, (str, list)):
        return Slot(part=value)
    else:
        raise ValueError(f"Cannot convert {type(value)} to Slot")


SlotType = Annotated[Union[Slot, str, List[Optional[str]]], BeforeValidator(convert_to_slot)]

##────────────────────────────────────────────────────────────────────────────}}}


## {{{                    --     Transcription Unit     --
class TranscriptionUnit(BaseModel):
    name: str = ""
    slots: List[SlotType] = []
    params: Dict = {}
    source: Optional[str] = None
    param_ref_ids: Dict[str, Optional[str]] = {}

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        self.__get_parameters()

    def __get_parameters(self):
        for s in self.slots:
            assert isinstance(s, Slot)
            if s.maps_to_parameter is not None:
                if s.maps_to_parameter in self.params:
                    raise ValueError(f"Parameter {s.maps_to_parameter} already in params")
                # replace None values with the default part for this parameter
                if isinstance(s.part, list) and s.maps_to_parameter in EMBEDDING_TO_DEFAULT_PART:
                    default = EMBEDDING_TO_DEFAULT_PART[s.maps_to_parameter]
                    self.params[s.maps_to_parameter] = [default if p is None else p for p in s.part]
                else:
                    self.params[s.maps_to_parameter] = s.part
                # track ref_id for this parameter
                self.param_ref_ids[s.maps_to_parameter] = s.ref_id

        # add default parameters
        for _, p in PART_TYPE_TO_EMBEDDING_NAME.items():
            if p not in self.params:
                try:
                    self.params[p] = [EMBEDDING_TO_DEFAULT_PART[p]]
                    self.param_ref_ids[p] = None  # default parameters have no ref_id
                except KeyError:
                    msg = f"No default part for parameter {p}"
                    msg += f" (part_type_to_parameter_name: {PART_TYPE_TO_EMBEDDING_NAME})"
                    msg += f" (parameter_to_default_part: {EMBEDDING_TO_DEFAULT_PART})"
                    raise

    def to_parts(self) -> List[Union[str, List[str]]]:
        """Convert slots back to a parts representation"""
        return [s.part if not isinstance(s.part, list) else s.part for s in self.slots]  # type: ignore

    def with_source(self, source: str) -> "TranscriptionUnit":
        """Create a copy of this TranscriptionUnit with a different source"""
        return TranscriptionUnit(name=self.name, slots=self.slots, source=source)


Unit = TranscriptionUnit  # alias for declarative API

##────────────────────────────────────────────────────────────────────────────}}}


## {{{                           --     CoTx     --
class CoTransfection(BaseModel):
    name: Optional[str] = None
    units: List[Unit]
    ratios: Optional[List[float]] = None

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        if self.ratios is None:  # equal ratios by default
            self.ratios = [1.0] * len(self.units)

    def __hash__(self):
        return hash(str(self.model_dump()))


def process_cotx_list(cotx_list: List[CoTransfection]) -> List[CoTransfection]:
    """Add names to unnamed cotx groups and sources"""

    source_counter = 0

    for i, cotx in enumerate(cotx_list):
        if cotx.name is None:
            cotx.name = f"cotx_{i + 1}"

        for unit in cotx.units:
            if unit.source is None:
                source_counter += 1
                unit.source = f"plsmd_{source_counter}"

    return cotx_list


CoTxList = Annotated[List[CoTransfection], BeforeValidator(process_cotx_list)]

##────────────────────────────────────────────────────────────────────────────}}}


class Recipe(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[dict] = None
    content: CoTxList = []


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
) -> list[Network]:
    compute_graphs = [build_central_dogma_graph(recipe)]
    for rule in rules:
        compute_graphs = flatten(
            [
                apply_rule(
                    rule,
                    graph,
                    lib=lib,
                    **kwargs,  # for example, input_markers=[...]
                )
                for graph in compute_graphs
            ]
        )
    return [
        Network(name=f"{recipe.name}_{get_output_names(graph)}", compute_graph=graph)
        for graph in compute_graphs
    ]


def network_to_recipe(network: Network) -> Recipe:
    ...
    # should round-trip perfectly!!


class NetworkConstructionError(Exception):
    """Exception for errors during network construction"""

    pass


def build_central_dogma_graph(
    network: Network, lib: PartsLibrary, custom_outputs_parts=None
) -> pd.DataFrame:
    """
    Build a central dogma graph directly from a Network.
    Handles extracting transcription units internally.
    """
    # Extract transcription units from cotx
    transcription_units = {}
    for group_idx, group in enumerate(network.cotx or []):
        for unit_idx, unit in enumerate(group.units):
            tu_name = unit.name or f"TU_{len(transcription_units) + 1}"
            transcription_units[tu_name] = unit

    def make_hashable(params, tu_obj):
        """Make params hashable, considering ref_id for identical part grouping."""
        hashable_params = {}
        for param_name, parts in params.items():
            ref_id = tu_obj.param_ref_ids.get(param_name)
            if ref_id is not None:
                # use ref_id as the hashable representation for non-null ref_ids
                hashable_params[param_name] = (f"ref:{ref_id}",)
            else:
                # use the actual parts for null ref_ids
                hashable_params[param_name] = tuple(parts) if isinstance(parts, list) else (parts,)
        return tuple(sorted((k, v) for k, v in hashable_params.items()))

    def get_dna(tu: TranscriptionUnit) -> Tuple[List[str], Dict[str, List[str]]]:
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
            # None values should have been replaced with defaults in __get_parameters
            # but filter them just in case
            non_none_parts = [p for p in parts if p is not None]
            if non_none_parts:  # only process if we have non-None parts
                p = lib.pc.loc[non_none_parts]
                if p[transform].sum() > 0:
                    params[param_name] = list(p[p[transform] == 1].index)
        return content, params

    def get_rna(tu: TranscriptionUnit):
        return get_downstream(tu, transform="transcripted")

    def get_prt(tu: TranscriptionUnit):
        return get_downstream(tu, transform="translated")

    # Build TU data
    tu: List[dict] = []
    assert transcription_units is not None, "No transcription units in network"

    for tuid, t in transcription_units.items():
        dna, dna_params = get_dna(t)
        rna, rna_params = get_rna(t)
        prt, prt_params = get_prt(t)
        tu.append(
            {
                "name": tuid,
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

    assert tu is not None, "No transcription units in network"
    tudf = pd.DataFrame(tu)

    def only_one_value_per_param(params: Dict[str, List[str]]) -> bool:
        return all(len(parts) <= 1 for _, parts in params.items())

    def has_non_null_ref_id(row_name: str) -> bool:
        tu_obj = transcription_units[row_name]
        return any(ref_id is not None for ref_id in tu_obj.param_ref_ids.values())

    def group_multi_param_tus(df: pd.DataFrame, node_type: str, params_col: str) -> List[List[str]]:
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
        # Check if this network has been through commit/quantization by looking for specific signatures:
        # 1. TUs with single quantized values that don't match expected defaults
        # 2. Mixed single-value and multi-value parameters in the same network
        def is_likely_quantized_tu(tu):
            """Check if a TU has parameters that look like they were quantized from multi-value to single-value"""
            for param_name, parts in tu.params.items():
                if (
                    param_name in ["tl_rate", "tc_rate"]
                    and isinstance(parts, list)
                    and len(parts) == 1
                ):
                    # Single-value quantized parameter, but not a default empty value
                    if param_name == "tl_rate" and parts[0] != "00_empty_tc":
                        return True
                    if param_name == "tc_rate" and parts[0] not in [
                        "hEF1a"
                    ]:  # common default promoters
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
        # Only trigger committed mode if we have clear quantized TUs AND multi-value TUs mixed together
        network_is_committed = len(quantized_tus) > 0 and len(multi_value_tus) > 0

        if network_is_committed:
            # For committed networks, group primarily by content (node_type) rather than parameters
            # This ensures TUs with collapsed parameters still get grouped with their original groups
            tu_ids = list(tudf.groupby(by=node_type).agg(list).name)
            return pd.DataFrame({"tu_id": tu_ids, "type": node_type})
        else:
            # Original behavior for non-committed networks
            # no params
            no_params = list(
                tudf[tudf[params_col].map(len) == 0].groupby(by=node_type).agg(list).name
            )
            # single param value
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
            # multi param values
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

    # transcription units are never grouped
    dna_df = pd.DataFrame({"tu_id": [[x] for x in cast(str, tudf["name"])], "type": "DNA"})
    rna_df = process_node_type("RNA", "RNA_params")
    prt_df = process_node_type("PRT", "PRT_params")
    tudf.set_index("name", inplace=True)

    # Then concatenate them:
    cdg = pd.concat([dna_df, rna_df, prt_df], sort=False).reset_index(drop=True)
    cdg["predecessor"] = None
    cdg["successor"] = None

    # connect DNA to RNA through successor list
    dna_nodes = cdg[cdg.type == "DNA"]
    rna_nodes = cdg[cdg.type == "RNA"]
    if len(rna_nodes) == 0:
        raise NetworkConstructionError("No RNA nodes in central dogma graph")

    for i, r in dna_nodes.iterrows():
        successors = []
        for ii, rr in rna_nodes.iterrows():
            assert len(r.tu_id) == 1, (
                "a DNA node should have only one value in its tu_id list (1 DNA node per Transcription Unit)"
            )
            if r.tu_id[0] in rr.tu_id:  # if we have an RNA that has the same TU as the DNA
                successors.append(ii)
        cdg.at[i, "successor"] = successors

    # connect RNA to PRT through successor list
    for i_r, rna in rna_nodes.iterrows():  # for each RNA
        successors = []
        for i_p, prt in cdg[cdg.type == "PRT"].iterrows():  # for each PRT
            if set(rna.tu_id).issubset(set(prt.tu_id)):
                successors.append(i_p)
        cdg.at[i_r, "successor"] = successors

    # now deduce the predecessor lists
    cdg["predecessor"] = [list() for _ in range(len(cdg))]
    for i, r in cdg.iterrows():
        if r.successor is not None:
            for s in r.successor:
                cdg.loc[s, "predecessor"].append(i)
    cdg.loc[~cdg.predecessor.astype(bool), "predecessor"] = None

    logger.debug(f"cdg: \n{cdg}\n")

    # We explicitly describe the part content of each node:
    try:
        cdg["content"] = cdg.apply(lambda x: tudf.loc[x.tu_id[0]][x.type], axis=1)
        cdg["content_type"] = cdg.apply(
            lambda x: tuple([lib.parts.loc[p].iloc[0] for p in x.content]), axis=1
        )
    except Exception as e:
        msg = f"Error while building central dogma graph. Error: {e}"
        msg += f"\ntudf: \n{tudf}"
        msg += f"\n\ncdg: \n{cdg}"
        raise NetworkConstructionError(msg)

    # and add the available paras with their possible parts
    cdg["params"] = cdg.apply(lambda x: tudf.loc[x.tu_id[0]][x.type + "_params"], axis=1)

    # And finally add information about the output of the whole graph:
    # by default outputs are all parts whose category is fluo_marker
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
    transcription_units: Dict[str, TranscriptionUnit], lib: PartsLibrary, custom_outputs_parts=None
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


def df_to_graphstate(df: pd.DataFrame) -> GraphState:
    """
    Convert pandas DataFrame (CDG or compute graph) to GraphState.
    Lossless conversion preserving all information.
    """
    nodes = []
    edges = []

    # Determine if this is a CDG or compute graph based on columns
    columns = df.columns.tolist()
    is_cdg = "tu_id" in columns and "predecessor" in columns and "successor" in columns
    is_compute_graph = "cdg_input" in columns and "cdg_output" in columns

    if is_cdg:
        # Convert Central Dogma Graph
        for idx, row in df.iterrows():
            # Create node
            extra = {}
            # Store all CDG-specific information in extra
            # Use explicit None checks since pd.notna() doesn't work well with lists/arrays
            if row.get("tu_id") is not None:
                extra["tu_id"] = row["tu_id"]
            if row.get("content") is not None:
                extra["content"] = row["content"]
            if row.get("content_type") is not None:
                extra["content_type"] = row["content_type"]
            if row.get("params") is not None:
                extra["params"] = row["params"]
            if row.get("is_output") is not None:
                extra["is_output"] = row["is_output"]
            if row.get("is_input") is not None:
                extra["is_input"] = row["is_input"]

            node = GraphNode(node_id=idx, node_type=row["type"], extra=extra)
            nodes.append(node)

            # Create edges from successor relationships
            if row.get("successor") is not None and row["successor"]:
                for target_idx in row["successor"]:
                    # Convert content to Part objects if available
                    content = ()
                    if row.get("content") is not None and row["content"]:
                        content = tuple(
                            Part(name=part, category="unknown") for part in row["content"]
                        )

                    edge = GraphEdge(
                        source_id=idx,
                        target_id=target_idx,
                        output_slot=0,
                        input_slot=0,
                        content=content,
                        content_type=row["type"] if row["type"] in ["DNA", "RNA", "PRT"] else None,
                    )
                    edges.append(edge)

    elif is_compute_graph:
        # Convert Compute Graph
        for idx, row in df.iterrows():
            extra = {}
            # Store all compute graph specific information in extra
            if row.get("cdg_input") is not None:
                extra["cdg_input"] = row["cdg_input"]
            if row.get("cdg_output") is not None:
                extra["cdg_output"] = row["cdg_output"]
            if row.get("input_from") is not None:
                extra["input_from"] = row["input_from"]
            if row.get("output_to") is not None:
                extra["output_to"] = row["output_to"]
            if row.get("extra") is not None:
                extra["compute_extra"] = row["extra"]
            if row.get("source_id") is not None:
                extra["source_id"] = row["source_id"]

            # Handle inverse specification
            inverse_spec = None
            if row.get("is_inverse_of") is not None:
                inverse_spec = row["is_inverse_of"]  # This should be an InverseSpec object

            node = GraphNode(
                node_id=idx, node_type=row["type"], is_inverse_of=inverse_spec, extra=extra
            )
            nodes.append(node)

            # For compute graphs, edges are typically created from input_from/output_to relationships
            # This would require additional logic based on the specific compute graph structure
    else:
        raise ValueError(f"Unknown DataFrame format. Columns: {columns}")

    return GraphState(nodes=nodes, edges=edges)


def graphstate_to_df(graph: GraphState, format_type: str = "cdg") -> pd.DataFrame:
    """
    Convert GraphState back to pandas DataFrame.
    format_type: either "cdg" (central dogma graph) or "compute" (compute graph)
    """
    if format_type == "cdg":
        # Convert back to Central Dogma Graph format
        rows = []

        for node in graph.nodes:
            row = {
                "type": node.node_type,
                "tu_id": node.extra.get("tu_id"),
                "content": node.extra.get("content"),
                "content_type": node.extra.get("content_type"),
                "params": node.extra.get("params"),
                "is_output": node.extra.get("is_output"),
                "is_input": node.extra.get("is_input"),
                "predecessor": [],
                "successor": [],
            }
            rows.append(row)

        # Reconstruct predecessor/successor relationships from edges
        for edge in graph.edges:
            source_idx = edge.source_id
            target_idx = edge.target_id

            # Add to successor list of source
            if rows[source_idx]["successor"] is None:
                rows[source_idx]["successor"] = []
            rows[source_idx]["successor"].append(target_idx)

            # Add to predecessor list of target
            if rows[target_idx]["predecessor"] is None:
                rows[target_idx]["predecessor"] = []
            rows[target_idx]["predecessor"].append(source_idx)

        # Convert empty lists to None for consistency with original format
        for row in rows:
            if not row["predecessor"]:
                row["predecessor"] = None
            if not row["successor"]:
                row["successor"] = None

        df = pd.DataFrame(rows)

    elif format_type == "compute":
        # Convert back to Compute Graph format
        rows = []

        for node in graph.nodes:
            row = {
                "type": node.node_type,
                "cdg_input": node.extra.get("cdg_input"),
                "cdg_output": node.extra.get("cdg_output"),
                "input_from": node.extra.get("input_from"),
                "output_to": node.extra.get("output_to"),
                "is_inverse_of": node.is_inverse_of,
                "extra": node.extra.get("compute_extra"),
                "source_id": node.extra.get("source_id"),
            }
            rows.append(row)

        df = pd.DataFrame(rows)

    else:
        raise ValueError(f"Unknown format_type: {format_type}. Must be 'cdg' or 'compute'")

    return df
