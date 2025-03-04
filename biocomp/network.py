from .library import PartsLibrary as PartsLibrary
import numpy as np
import pandas as pd
from . import utils as ut
import sqlite3
from typing import (
    Callable,
    List,
    Dict,
    Tuple,
    Iterable,
    Optional,
    cast,
    Sequence,
    Literal,
    Annotated,
    Union,
    Any,
)
from itertools import product
from pydantic.dataclasses import dataclass
from pydantic import BaseModel, Field, ConfigDict, BeforeValidator
from functools import cached_property
from .utils import load_lib

from biocomp.logging_config import get_logger

logger = get_logger(__name__)

part_type_to_parameter_name = {"promoter": "tc_rate", "uORF_group": "tl_rate"}
parameter_to_default_part = {"tl_rate": "00_empty_tc", "tc_rate": "hEF1a"}


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                   --     general network utils     --
# ···············································································


def target_site_formatter(x):
    return x.replace("_rec", "_{RS}")


INTERESTING_PART_TYPES = [("RNA", "ERN_recog_site_5p", target_site_formatter), ("PRT", None, None)]


@dataclass
class TUInfo:
    content: str
    content_types: tuple[str, ...]


def find_interesting_part(
    cdg: pd.DataFrame,
    tu_id: str,
    part_level: Literal["DNA", "RNA", "PRT"],
    part_type: Optional[str] = None,
    formatter: Optional[Callable[[str], str]] = None,
) -> tuple[list[str], list[tuple[str, ...]]]:
    """Modified to return both content and content_types"""
    cdg = cdg.copy()
    formatter = formatter or (lambda x: x)
    cdg = cdg[cdg.tu_id.apply(lambda x: tu_id in x)]
    level_data = cdg[cdg.type == part_level]

    interesting_parts = []
    content_types_list = []

    for _, row in level_data.iterrows():
        content_types = row["content_type"]
        contents = row["content"]
        if not isinstance(content_types, (list, tuple)) or not isinstance(contents, (list, tuple)):
            continue

        content_types = tuple(content_types)
        contents = tuple(contents)

        if part_type:
            try:
                pos = content_types.index(part_type)
                interesting_parts.append(formatter(contents[pos]))
                content_types_list.append(content_types)
            except (ValueError, IndexError):
                continue
        else:
            interesting_parts.extend(formatter(content) for content in contents)
            content_types_list.extend([content_types] * len(contents))

    return interesting_parts, content_types_list


def get_content_from_tu_id(
    cdg: pd.DataFrame, tu_id: str, interesting_part_types=INTERESTING_PART_TYPES
) -> TUInfo:
    contents = []
    all_content_types = []

    for part_level, part_type, formatter in interesting_part_types:
        parts, types = find_interesting_part(cdg, tu_id, part_level, part_type, formatter)
        contents.extend(parts)
        all_content_types.extend(types)

    return TUInfo(
        content=r"\_".join(contents),
        content_types=tuple(ct for types in all_content_types for ct in types),
    )


def sort_tus(tu_info: TUInfo) -> tuple[int, int]:
    # hierarchically sort TUs based on content types
    has_recog_site = "ERN_recog_site_5p" in tu_info.content_types
    only_fluo = all(
        ct == "fluo_marker" for ct in tu_info.content_types if ct not in ("insulator", "terminator")
    )
    return (
        -int(has_recog_site),  # having recog site is more important
        int(only_fluo),
    )


def get_ratio(
    fwd_agg: pd.Series,
    cdg: pd.DataFrame,
    sort_tus: Callable = sort_tus,
) -> tuple[tuple[TUInfo, ...], tuple[str, ...]]:
    """
    Returns two tuples:
    - The TUInfo objects for each TU in the aggregation
    - The ratios as strings (rounded to 2 decimal places)
    """

    out_tuid = fwd_agg.cdg_output
    tu_infos = [get_content_from_tu_id(cdg, tu_id) for tu_id in out_tuid]

    ratios = np.array(fwd_agg["extra"]["ratios"])
    min_ratio = np.maximum(ratios.min(), 1e-6)
    normed_ratios = np.round(ratios / min_ratio, 2)

    def is_round(x):
        return x == int(x)

    normed_ratios = [str(int(r)) if is_round(r) else str(r) for r in normed_ratios]

    # Sort both tu_infos and ratios together
    sorted_pairs = sorted(zip(tu_infos, normed_ratios), key=lambda x: sort_tus(x[0]))
    sorted_tu_infos, sorted_ratios = zip(*sorted_pairs)

    return (tuple((tu.content for tu in sorted_tu_infos)), tuple(sorted_ratios))


def get_ratios(net) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    cmp = net.compute_graph
    cdg = net.central_dogma_graph
    agg = cmp[cmp.type == "aggregation"]
    all_ratios = [get_ratio(a, cdg) for _, a in agg.iterrows()]
    return all_ratios


def get_default_input_order(net, cotx):
    """
    Get the default input order for a network:
    priority to cotx that contain an ERN
    then to cotx that contain a ERN_recog_site_5p
    then to the rest

    need to use net.get_input_proteins() which returns the name of the fluo marker
    in each cotx.
    We then use that original order to specify a new order based on the priority rules
    TODO

    """


def cotx_ratios_str(cotx):
    lines = []
    for tus, ratios in cotx:
        lines.append(":".join(tus) + " -> " + ":".join(ratios))
    return "\n".join(lines)


def generate_network_info(net):
    """Generate a dictionnary of information for a network"""
    # NOT the string version but the raw dict
    arch, seqtype = ut.get_network_family(net)
    uorf_vals, uorf_names = ut.get_all_uorf_values(net)
    cdg = net.central_dogma_graph
    genes = ut.flatten(cdg[cdg.type == "PRT"]["content"].tolist())
    markers = tuple(sorted(net.get_inverted_input_proteins()))
    all_outputs = tuple(sorted(net.get_output_proteins()))
    dependent_outputs = tuple(sorted(list(set(all_outputs) - set(markers))))
    ern_names = ut.get_all_ERNs_names(net)
    cotx = get_ratios(net)
    default_input_order = get_default_input_order(net, cotx)
    net_info = {
        "sequestron_type": seqtype,
        "architecture": arch,
        "ern_names": ern_names,
        "uorf_values": uorf_vals,
        "uorf_names": ut.flatten(uorf_names),
        "genes": genes,
        "markers": markers,
        "output_proteins": all_outputs,
        "dependent_outputs": dependent_outputs,
        "cotx": cotx,
        "cotx_str": cotx_ratios_str(cotx),
        "ern_names_str": ", ".join(ern_names),
    }
    return net_info


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                       --     base classes     --
# ···············································································
class NetworkConstructionError(Exception):
    pass


class Slot(BaseModel):
    """Transcription Units are made of slots which contain either a part or a list of
    possible parts that map to a quantized parameter"""

    lib: PartsLibrary = Field(default_factory=load_lib, repr=False)
    part: Optional[Union[str, List[str]]] = None

    # does this slot map to a parameter, like "tl_rate" or "tc_rate"?
    maps_to_parameter: Optional[str] = None

    def model_post_init(self, *args, **kwargs):
        if isinstance(self.part, list):
            if not self.part or self.part == [None]:
                self.part = None
            else:
                mapped = list(set([self.__mapped_parameter(p) for p in self.part if p is not None]))
                if len(mapped) != 1:
                    raise ValueError(f"{self.part} maps to {len(mapped)} parameters ({mapped})")
                self.maps_to_parameter = mapped[0]
        else:
            self.maps_to_parameter = self.__mapped_parameter(self.part)

        if self.maps_to_parameter is not None and not isinstance(self.part, list):
            # if the slot maps to a parameter, it must be a list (even if there's only one part)
            self.part = [self.part]  # type: ignore

    def __mapped_parameter(self, part_name: Optional[str]) -> Optional[str]:
        """Returns the name of the parameter a part maps to, or None if it doesn't map to any"""
        if part_name is not None:
            if part_name in self.lib.pc.index:
                category = self.lib.pc.loc[part_name, "category"]
                if category in part_type_to_parameter_name:
                    return part_type_to_parameter_name[category]
            else:
                raise ValueError(f"Unknown part: {part_name}")
        return None

    def __repr__(self) -> str:
        if self.maps_to_parameter is None:
            if self.part is None:
                return "<empty slot>"
            else:
                return f"<{self.part}>"
        return f"<{self.part} -> {self.maps_to_parameter}>"


# Validator for automatic Slot conversion
def convert_to_slot(value, lib=None):
    """Convert strings or lists of strings to Slot objects"""
    if isinstance(value, Slot):
        if lib is not None and value.lib is None:
            value.lib = lib
        return value
    elif isinstance(value, (str, list)):
        return Slot(part=value)
    else:
        raise ValueError(f"Cannot convert {type(value)} to Slot")


SlotType = Annotated[Slot | str | list[str], BeforeValidator(convert_to_slot)]


class TranscriptionUnit(BaseModel):
    name: str = ""
    slots: List[SlotType] = Field(default_factory=list)
    params: Dict = Field(default_factory=dict)
    source: Optional[str] = None
    lib: PartsLibrary = Field(default_factory=load_lib, repr=False)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def model_post_init(self, *args, **kwargs):
        # Ensure all slots have a library
        for slot in self.slots:
            if slot.lib is None:
                slot.lib = self.lib

        self.__get_parameters()

    def __get_parameters(self):
        for s in self.slots:
            if s.maps_to_parameter is not None:
                if s.maps_to_parameter in self.params:
                    raise ValueError(f"Parameter {s.maps_to_parameter} already in params")
                self.params[s.maps_to_parameter] = s.part

        # Add default parameters
        for _, p in part_type_to_parameter_name.items():
            if p not in self.params:
                try:
                    self.params[p] = [parameter_to_default_part[p]]
                except KeyError:
                    msg = f"No default part for parameter {p}"
                    msg += f" (part_type_to_parameter_name: {part_type_to_parameter_name})"
                    msg += f" (parameter_to_default_part: {parameter_to_default_part})"
                    raise

    def to_parts(self) -> List[Union[str, List[str]]]:
        """Convert slots back to a parts representation"""
        return [s.part if not isinstance(s.part, list) else s.part for s in self.slots]

    def with_source(self, source: str) -> "TranscriptionUnit":
        """Create a copy of this TranscriptionUnit with a different source"""
        return TranscriptionUnit(name=self.name, slots=self.slots, source=source, lib=self.lib)


Unit = TranscriptionUnit  # alias for declarative API


class TranscriptionUnitGenerator:
    def __init__(self, part_generators):
        self.name = ""
        self.part_generators = part_generators
        # a generator is a function that takes a library
        # and a list of previously generated slots in this TU
        # (and possibly other arguments) and returns a generated slot

    def generate_all(self, lib, order=None, *args, **kwargs):
        if order is None:
            order = list(range(len(self.part_generators)))

        # for each slot, generate all possible parts.
        # but a slot needs the parts from all previous slots to be generated
        # so we need to do this in order
        def _next(slots, i):
            if i == len(order):
                yield slots
            else:
                g = self.part_generators[order[i]]
                possile_parts = g(lib, slots, *args, **kwargs)
                for p in possile_parts:
                    yield from _next(slots + [Slot(part=p, lib=lib)], i + 1)

        return _next([], 0)


class GraphComputeNode:
    # a simple convenience one-off class to store the information about a node
    def __init__(self, id, type, cdg_input, cdg_output):
        self.id = id
        self.type = type
        self.cdg_input = cdg_input
        self.cdg_output = cdg_output if cdg_output is not None else -1
        self.input_from = []
        self.output_to = []
        self.extra = {}
        self.is_inverse_of = None
        self.n_outputs = None

    def removeOutput(self, other):
        other.input_from.remove(self.id)
        for i in range(len(self.output_to)):
            if self.output_to[i][0] == other.id:
                self.output_to.pop(i)
                break

    def toDict(self):
        return {
            "id": self.id,
            "type": self.type,
            "cdg_input": self.cdg_input,
            "cdg_output": self.cdg_output,
            "input_from": self.input_from,
            "output_to": self.output_to,
            "is_inverse_of": self.is_inverse_of,
            "extra": self.extra,
        }

    def __str__(self):
        return str(self.toDict())

    def __repr__(self):
        return str(self.toDict())


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


def transcription_unit_from_L1(l1id: str, lib: PartsLibrary) -> TranscriptionUnit:
    """Builds a transcription unit from an L1 id and a library
    The TU is built by concatenating all the parts from the L0s
    that are in the L1"""
    l0_cols = ["insulator", "promoter", "5'UTR", "gene", "3'UTR", "terminator"]
    L0s = lib.L1s.loc[l1id][l0_cols].tolist()
    part_cols = [f"part_{i}" for i in range(1, 7)]
    parts: List[str] = []
    for l in L0s:
        try:
            parts += [p for p in lib.L0s.loc[l][part_cols].tolist() if p]
        except Exception as e:
            msg = f"Error in L0 {l} of L1 {l1id}: {e}"
            msg += f"\npart_cols: {part_cols}"
            msg += f"\nlib.L0s[{l}]: {lib.L0s.loc[l]}"
            msg += f"\nlib.L0s: {lib.L0s}"
            raise NetworkConstructionError(msg)
    return TranscriptionUnit(slots=[Slot(part=p, lib=lib) for p in parts])


class CoTransfection(BaseModel):
    name: Optional[str] = None
    units: List[Unit]
    ratios: Optional[List[float]] = None

    def model_post_init(self, *args, **kwargs):
        if self.ratios is None:  # equal ratios by default
            self.ratios = [1.0] * len(self.units)


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


class Network(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    lib: PartsLibrary = Field(default_factory=load_lib, repr=False)
    name: Optional[str] = None
    custom_outputs: Optional[List] = Field(default=None, repr=False)
    metadata: Optional[Dict] = None

    # Raw recipe data
    raw_tu_in_sources: Optional[List[Tuple[str, str, int]]] = Field(default=None, repr=False)
    raw_aggregations: Optional[List[Tuple[int, str, float]]] = Field(default=None, repr=False)

    # Processed recipe data
    tu_inputs: Optional[pd.DataFrame] = Field(default=None, repr=False)
    tu_in_sources: Optional[pd.DataFrame] = Field(default=None, repr=False)
    aggregations: Optional[pd.DataFrame] = Field(default=None, repr=False)
    transcription_units: Optional[Dict[str, TranscriptionUnit]] = None

    # Graph data
    central_dogma_graph: Optional[pd.DataFrame] = Field(default=None, repr=False)
    compute_graph: Optional[pd.DataFrame] = Field(default=None, repr=False)

    # Field for declarative constructor (more concise for manual network creation)
    cotx: Optional[CoTxList] = Field(default=None)
    build_on_init: Optional[bool] = Field(default=True, repr=False)
    invert_on_build: bool = False

    # Private attributes
    _n_inputs: Optional[int] = None
    _n_outputs: Optional[int] = None
    _output_proteins: Optional[List[str]] = None

    def __init__(self, *args, **kwargs):
        if args and isinstance(args[0], PartsLibrary):
            # If called with positional arguments, assume it's the legacy style
            if len(args) > 4:
                raise ValueError("Too many positional arguments")

            # Map positional arguments to their respective parameter names
            params = ["lib", "recipe_name", "custom_outputs", "metadata"]
            new_kwargs = {}

            for i, arg in enumerate(args):
                new_kwargs[params[i]] = arg

            new_kwargs.update(kwargs)

            super().__init__(**new_kwargs)
        else:
            # Normal Pydantic initialization
            super().__init__(*args, **kwargs)

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)

        decl = False
        if self.cotx is not None and self.transcription_units is None:
            self._process_declarative()
            decl = True

        if self.build_on_init:
            self.build()

    def to_declarative(self) -> "Network":
        """Convert traditional network representation to declarative"""
        if not self.is_built():
            raise ValueError("Network must be built before converting to declarative")

        source_to_tus = {}
        for _, row in self.tu_in_sources.iterrows():
            source = row["source"]
            tu_name = row["TU"]
            if source not in source_to_tus:
                source_to_tus[source] = []
            source_to_tus[source].append(tu_name)

        agg_to_sources = {}
        for agg_id, row in self.aggregations.iterrows():
            sources = row["source"]
            ratios = row["ratio"]
            agg_to_sources[agg_id] = (sources, ratios)

        cotx_list = []
        for agg_id, (sources, ratios) in agg_to_sources.items():
            units = []
            for source in sources:
                for tu_name in source_to_tus.get(source, []):
                    tu = self.transcription_units[tu_name]
                    units.append(tu.with_source(source))

            cotx_list.append(CoTransfection(units=units, ratios=ratios))

        # Create new network with declarative format
        return Network(lib=self.lib, name=self.name, cotx=cotx_list, build_on_init=False)

    def _process_declarative(self):
        """
        Convert declarative cotx format to traditional format.
        Handles unified TranscriptionUnit class, multiple TUs per source,
        and maintains proper ordering of TUs within sources.
        """
        source_to_tus = {}  # source_name -> [(tu_name, position)]

        self.transcription_units = {}
        sources_data = []
        aggregations_data = {}

        for group_idx, group in enumerate(self.cotx):
            group_sources = []

            for unit_idx, unit in enumerate(group.units):
                tu_name = unit.name or f"TU_{len(self.transcription_units) + 1}"

                self.transcription_units[tu_name] = unit

                source = unit.source or f"plsmd_{len(source_to_tus) + 1}"

                # Track in source_to_tus for position calculation
                if source not in source_to_tus:
                    source_to_tus[source] = []
                position = len(source_to_tus[source])
                source_to_tus[source].append((tu_name, position))

                if source not in group_sources:
                    group_sources.append(source)

            # Store aggregation data for this group
            aggregations_data[group_idx] = {
                "source": group_sources,
                "ratio": group.ratios if group.ratios else [1.0] * len(group.units),
            }

        # Create sources dataframe with correct positions
        for source, tu_entries in source_to_tus.items():
            for tu_name, position in tu_entries:
                sources_data.append({"source": source, "TU": tu_name, "position": position})

        self.tu_in_sources = pd.DataFrame(sources_data)
        self.aggregations = pd.DataFrame(aggregations_data).T

        self.raw_tu_in_sources = [
            (row["source"], row["TU"], row["position"]) for _, row in self.tu_in_sources.iterrows()
        ]

        self.raw_aggregations = []
        for agg_id, row in self.aggregations.iterrows():
            for source, ratio in zip(row["source"], row["ratio"]):
                self.raw_aggregations.append((agg_id, source, ratio))

    @classmethod
    def legacy_init(
        cls,
        lib: PartsLibrary,
        recipe_name: str,
        custom_outputs: Optional[List] = None,
        metadata: Optional[Dict] = None,
        build: bool = False,
    ) -> "Network":
        instance = cls(
            lib=lib,
            recipe_name=recipe_name,
            custom_outputs=custom_outputs,
            metadata=metadata,
            build_on_init=build,
        )
        return instance

    def copy(self):
        from copy import deepcopy

        return deepcopy(self)

    ### {{{                    --     static constructors     --

    @classmethod
    def from_db(
        cls,
        lib: PartsLibrary,
        name: str,
        recipe_db: sqlite3.Connection,
        custom_outputs: Optional[List] = None,
        build: bool = True,
        metadata: Optional[Dict] = None,
        use_cache: Optional[str] = None,
    ):
        assert recipe_db is not None, "recipe_db cannot be None"
        recipe_db.commit()
        c = recipe_db.cursor()
        # first let's check that there is a recipe with this name
        c.execute("SELECT * FROM recipes WHERE name=?", (name,))
        assert c.fetchone() is not None, f"No recipe named {name} in database {recipe_db}."
        # Available recipes: {c.execute("SELECT name FROM recipes").fetchall()}'
        # get the transcription units
        c.execute(
            """SELECT TU, TU || '_' || aggregation as name FROM TU_in_source tis, source_in_aggregation sia, aggregations a
           WHERE tis.source = sia.source AND sia.aggregation = a.id AND a.recipe = ?
           ORDER BY name""",
            (name,),
        )
        raw_transcription_units = list(c.fetchall())  # columns: TU_id, TU_name ("TUid_aggid")
        # then get the sources
        c.execute(
            """SELECT tis.source || '_' || aggregation as source, TU || '_' || aggregation as TU, position
            FROM TU_in_source tis, source_in_aggregation sia, aggregations a
           WHERE tis.source = sia.source AND sia.aggregation = a.id AND a.recipe = ? ORDER BY source, position""",
            (name,),
        )
        raw_tu_in_sources = list(c.fetchall())  # columns: source_name, TU_name, position
        c.execute(
            """SELECT a.id, sia.source || '_' || aggregation, sia.ratio FROM aggregations a, source_in_aggregation sia
            WHERE a.id = sia.aggregation AND a.recipe = ? ORDER BY a.id""",
            (name,),
        )
        raw_aggregations = list(c.fetchall())  # columns: agg_id, source_name, ratio
        transcription_units = {
            tu[1]: transcription_unit_from_L1(tu[0], lib) for tu in raw_transcription_units
        }  # a dict of {TU_unique_name: TU}
        return cls.from_raw(
            lib,
            name,
            transcription_units,
            raw_tu_in_sources,
            raw_aggregations,
            build=build,
            use_cache=use_cache,
            custom_outputs=custom_outputs,
            metadata=metadata,
        )

    # @classmethod
    # def from_raw(
    #     cls,
    #     lib: PartsLibrary,
    #     name: str,
    #     transcription_units: dict[str, TranscriptionUnit],  # dict of {TU_name: TU}
    #     raw_tu_in_sources: List[Tuple[str, str, int]],  # list of (source_name, TU_name, position)
    #     raw_aggregations: List[Tuple[int, str, float]],  # list of (agg_id, source_name, ratio)
    #     build: bool = True,  # whether to build the network's graph
    #     use_cache: Optional[str] = None,  # path to cache
    #     **kwargs,
    # ):
    #     n = cls(lib, name, **kwargs)
    #     n.raw_tu_in_sources = raw_tu_in_sources
    #     n.raw_aggregations = raw_aggregations
    #     n.transcription_units = transcription_units
    #
    #     def actually_build():
    #         n.tu_in_sources = pd.DataFrame(
    #             n.raw_tu_in_sources, columns=["source", "TU", "position"]
    #         )
    #         n.tu_in_sources.sort_values(by="position", inplace=True)
    #         n.aggregations = (
    #             pd.DataFrame(n.raw_aggregations, columns=["id", "source", "ratio"])
    #             .groupby("id")
    #             .agg(list)
    #         )
    #         if build:
    #             n.build()
    #         return n
    #
    #     n = ut.get_cache(lambda: actually_build(), n.get_signature(), use_cache)
    #     return n

    @classmethod
    def from_raw(
        cls,
        lib: PartsLibrary,
        name: str,
        transcription_units: dict[str, TranscriptionUnit],
        raw_tu_in_sources: List[Tuple[str, str, int]],
        raw_aggregations: List[Tuple[int, str, float]],
        build: bool = True,
        use_cache: Optional[str] = None,
        **kwargs,
    ):
        n = cls(
            name=name,
            lib=lib,
            transcription_units=transcription_units,
            raw_tu_in_sources=raw_tu_in_sources,
            raw_aggregations=raw_aggregations,
            build_on_init=False,
            **kwargs,
        )

        def actually_build():
            n.tu_in_sources = pd.DataFrame(
                n.raw_tu_in_sources, columns=["source", "TU", "position"]
            )
            n.tu_in_sources.sort_values(by="position", inplace=True)
            n.aggregations = (
                pd.DataFrame(n.raw_aggregations, columns=["id", "source", "ratio"])
                .groupby("id")
                .agg(list)
            )

            if build:
                n.build()
            return n

        n = ut.get_cache(lambda: actually_build(), n.get_signature(), use_cache)
        return n

    @classmethod
    def __obsolete__from_dict(
        cls, lib, name, transcription_units, sources, aggregations, build=True
    ):
        n = cls(lib, name)

        # transcription_units = {TU_name : TU}
        n.transcription_units = transcription_units

        # sources =  {source_name: [TU1, TU2, TU3, ...], ...}
        n.tu_in_sources = pd.DataFrame(
            [
                {"source": s, "TU": t, "position": i}
                for s, tuids in sources.items()
                for i, t in enumerate(tuids)
            ]
        )
        n.tu_in_sources.sort_values(by="position", inplace=True)

        # aggregations = [[source1, source2, source3, ...], ...]
        assert n.aggregations is None, "Aggregations already set"
        n.aggregations = (
            pd.DataFrame(
                [{"id": i, "source": s, "ratio": 1} for i, a in enumerate(aggregations) for s in a]
            )
            .groupby("id")
            .agg(list)
        )

        if build:
            n.__build_central_dogma_graph()
            n.__build_compute_graph()
        return n

    ##────────────────────────────────────────────────────────────────────────────}}}

    def get_compute_types(self):
        assert isinstance(self.compute_graph, pd.DataFrame), "Compute graph not built"
        node_dict = self.compute_graph.groupby("type").apply(lambda x: x.index.to_list()).to_dict()
        return node_dict

    def build(self):
        assert self.transcription_units is not None, "No transcription units in recipe"
        assert len(self.transcription_units) > 0, f"No transcription units in recipe {self.name}"
        self.__build_central_dogma_graph(self.custom_outputs)
        self.__build_compute_graph()
        if self.invert_on_build:
            inverted = inverted_network(self)[0]
            self.compute_graph = inverted.compute_graph
            self.central_dogma_graph = inverted.central_dogma_graph
            self.transcription_units = inverted.transcription_units
            self._n_inputs = inverted._n_inputs
            self._n_outputs = inverted._n_outputs
            self._output_proteins = inverted._output_proteins

    def is_built(self) -> bool:
        return (
            self.compute_graph is not None
            and self.central_dogma_graph is not None
            and self.transcription_units is not None
        )

    ## ───────────────────────────────────── ▼ ─────────────────────────────────────
    # {{{                           --     utils     --
    # ···············································································
    def __getDna(self, tu: TranscriptionUnit) -> Tuple[List[str], Dict[str, List[str]]]:
        content = []
        for s in tu.slots:
            if s.maps_to_parameter is None:
                content.append(s.part)
        return content, tu.params

    def __getDownstream(self, tu: TranscriptionUnit, transform: str):
        dna_content, dna_params = self.__getDna(tu)
        d = self.lib.pc.loc[dna_content]
        content = tuple(d[d[transform] == 1].index)
        params = {}
        for param_name, parts in dna_params.items():
            p = self.lib.pc.loc[parts]
            if p[transform].sum() > 0:
                assert p[transform].sum() == len(p)
                params[param_name] = list(p.index)
        return content, params

    def __getRna(self, tu: TranscriptionUnit):
        return self.__getDownstream(tu, transform="transcripted")

    def __getPrt(self, tu: TranscriptionUnit):
        return self.__getDownstream(tu, transform="translated")

    def __isOutputedBy(
        self, cdg_input_node: int, compute_nodes: List[GraphComputeNode]
    ) -> List[GraphComputeNode]:
        """returns a list of all the compute nodes that have cdg_input_node as output"""
        return [n for n in compute_nodes if cdg_input_node == n.cdg_output]

    def __checkForCycles(node_map):
        def dfs(node_id, visited, rec_stack):
            visited.add(node_id)
            rec_stack.add(node_id)
            for neighbor_id in node_map[node_id].input_from:
                if neighbor_id not in visited:
                    if dfs(neighbor_id, visited, rec_stack):
                        return True
                elif neighbor_id in rec_stack:
                    return True
            rec_stack.remove(node_id)
            return False

        visited = set()
        rec_stack = set()

        for node_id in node_map.keys():
            if node_id not in visited:
                if dfs(node_id, visited, rec_stack):
                    raise NetworkConstructionError("Cycle detected in compute graph")

    def __checkUniqueNodeIDs(self, nodes):
        node_ids = {node.id for node in nodes}
        if len(node_ids) != len(nodes):
            raise ValueError("Node IDs are not unique")

    def __removeShortcuts(self, nodes, root_id):
        # removeShortcuts removes indirect links in the Compute graph,
        # turning it from a directed acyclic graph to a tree.
        node_map = {node.id: node for node in nodes}
        if root_id not in node_map:
            raise ValueError(f"Root node ID {root_id} not found")

        self.__checkUniqueNodeIDs(nodes)
        Network.__checkForCycles(node_map)
        labels = {}
        for node in nodes:
            labels[node.id] = 1
        S = set()
        S.add(root_id)
        while len(S) > 0:
            N = node_map[S.pop()]
            w = labels[N.id] + 1
            for d in N.input_from:
                if labels[d] < w:
                    labels[d] = w
                    S.add(d)
        # remove all edges which connect nodes whose labels differ by more than 1.
        for node in nodes:
            for d in node.input_from:
                if labels[node.id] + 1 < labels[d]:
                    node_map[d].removeOutput(node)

    #                                                                            }}}
    ## ─────────────────────────────────────────────────────────────────────────────

    ## ───────────────────────────────────── ▼ ─────────────────────────────────────
    # {{{                 --     build central dogma graph     --
    # ···············································································

    def __build_central_dogma_graph(self, custom_outputs_parts=None):
        tu: List[dict] = []
        assert self.transcription_units is not None, "No transcription units in network"

        def make_hashable(x):
            return tuple(sorted((k, tuple(v)) for k, v in x.items()))

        for tuid, t in self.transcription_units.items():
            dna, dna_params = self.__getDna(t)
            rna, rna_params = self.__getRna(t)
            prt, prt_params = self.__getPrt(t)
            tu.append(
                {
                    "name": tuid,
                    "DNA": dna,
                    "DNA_params": dna_params,
                    "DNA_params_hashable": make_hashable(dna_params),
                    "RNA": rna,
                    "RNA_params": rna_params,
                    "RNA_params_hashable": make_hashable(rna_params),
                    "PRT": prt,
                    "PRT_params": prt_params,
                    "PRT_params_hashable": make_hashable(prt_params),
                }
            )
        assert tu is not None, "No transcription units in network"
        tudf = pd.DataFrame(tu)

        # transcription units are never grouped
        dna_df = pd.DataFrame({"tu_id": [[x] for x in cast(str, tudf["name"])], "type": "DNA"})

        def only_one_value_per_param(params: Dict[str, List[str]]) -> bool:
            return all(len(parts) <= 1 for _, parts in params.items())

        rna_tuids_noparams = list(
            tudf[tudf["RNA_params"].map(len) == 0].groupby(by="RNA").agg(list).name  # type: ignore
        )

        try:
            rna_tuids_oneparamvalue = (
                tudf[tudf["RNA_params"].map(len) > 0]
                .groupby(by="RNA")
                .filter(lambda x: only_one_value_per_param(x["RNA_params"]))
                .groupby(by=["RNA", "RNA_params_hashable"])
                .agg(list)
            )
        except Exception as e:
            msg = f"Error while grouping RNA that have one params: {e}\n"
            msg += f"tudf: \n{tudf}"
            raise NetworkConstructionError(msg)

        rna_tuids_oneparamvalue = (
            [] if rna_tuids_oneparamvalue.empty else list(rna_tuids_oneparamvalue.name)
        )

        rna_tuids_manyparamvalues = list(tudf[tudf["RNA_params"].map(len) > 1].name)
        rna_tuids = rna_tuids_noparams + rna_tuids_oneparamvalue + rna_tuids_manyparamvalues
        rna_df = pd.DataFrame({"tu_id": rna_tuids, "type": "RNA"})

        prt_tuids_noparams = list(
            tudf[tudf["PRT_params"].map(len) == 0].groupby(by="PRT").agg(list).name
        )
        # we group PRT with same content and same parameters if they have a single parameters
        prt_tuids_oneparamvalue = (
            tudf[tudf["PRT_params"].map(len) > 0]
            .groupby(by="PRT")
            .filter(lambda x: only_one_value_per_param(x["PRT_params"]))
            .groupby(by=["PRT", "PRT_params_hashable"])
            .agg(list)
        )
        prt_tuids_oneparamvalue = (
            [] if prt_tuids_oneparamvalue.empty else list(prt_tuids_oneparamvalue.name)
        )
        prt_tuids_manyparamvalues = list(tudf[tudf["PRT_params"].map(len) > 1].name)
        prt_tuids = prt_tuids_noparams + prt_tuids_oneparamvalue + prt_tuids_manyparamvalues
        prt_df = pd.DataFrame({"tu_id": prt_tuids, "type": "PRT"})

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
                assert (
                    len(r.tu_id) == 1
                ), "a DNA node should have only one value in its tu_id list (1 DNA node per Transcription Unit)"

                if r.tu_id[0] in rr.tu_id:  # if we have an RNA that has the same TU as the DNA
                    successors.append(ii)
            cdg.loc[i, "successor"] = successors

        # connect RNA to PRT through successor list
        for i_r, rna in rna_nodes.iterrows():  # for each RNA
            successors = []
            for i_p, prt in cdg[cdg.type == "PRT"].iterrows():  # for each PRT
                if set(rna.tu_id).issubset(set(prt.tu_id)):
                    successors.append(i_p)
            cdg.loc[i_r, "successor"] = successors

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
                lambda x: tuple([self.lib.parts.loc[p].iloc[0] for p in x.content]), axis=1
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
        outputs = (
            custom_outputs_parts if custom_outputs_parts is not None else []
        ) + self.lib.parts[self.lib.parts["category"] == "fluo_marker"].index.tolist()

        containsOutput = lambda l, outputs: any([o in l for o in outputs])
        cdg["is_output"] = False
        cdg.loc[cdg.type == "PRT", "is_output"] = cdg.loc[cdg.type == "PRT"].tu_id.apply(
            lambda x: containsOutput(tudf.loc[x].PRT.tolist()[0], outputs)
        )
        cdg["is_input"] = None
        self.central_dogma_graph = cdg

    #                                                                            }}}
    ## ─────────────────────────────────────────────────────────────────────────────

    ## ───────────────────────────────────── ▼ ─────────────────────────────────────
    # {{{      --     build compute graph (from central dogma graph)     --
    # ···············································································

    # a lot of the code below is super verbose and could be simplified and optimized
    # but it's definitely not a priority right now

    def __mergeSources(self, cdf, uidGen):
        assert self.central_dogma_graph is not None, "mergeSources: Central dogma graph not built"
        # in the compute graph,
        # merge TUs that are from a same source into a single source node (aka plasmid)

        sources_tuids = self.central_dogma_graph.loc[
            cdf[cdf.type == "source"].cdg_output
        ].tu_id.apply(lambda x: x[0])

        tmpdf = pd.DataFrame(
            {"compute_id": cdf[cdf.type == "source"].index, "tuid": sources_tuids}
        ).set_index("compute_id")
        # tmpdf is a mapping between the computegraph ids of every sources and their TUids

        cdf["source_id"] = None
        # cdf['extra'] = None

        sources = {}  # plasmid name -> list of compute nodes ids

        # tu_in_sources contains the list of TUs in each source, sorted by position
        assert self.tu_in_sources is not None, "No TU in sources"

        for i, r in self.tu_in_sources.groupby("source").agg(list).iterrows():
            # but you can have sources in the db that are not in the recipe
            group = []  # group will contain the compute nodes ids of the TUs in the source
            for t in r["TU"]:
                try:
                    group.append(tmpdf[tmpdf.tuid == t].index[0])
                except IndexError:
                    msg = f"Error while merging sources. TU {t} not found in tmpdf."
                    msg += f"\n\ntmpdf: \n{tmpdf}"
                    msg += f"\nsources_tuids: \n{sources_tuids}"
                    msg += f"\ncentral dogma graph: \n{self.central_dogma_graph}"
                    msg += f"\ncdf: \n{cdf}"
                    raise NetworkConstructionError(msg)

            sources[i] = group

        for k, v in sources.items():
            nid = uidGen()
            newsource = GraphComputeNode(nid, "source", None, [cdf.loc[vv].cdg_output for vv in v])
            newsource.output_to = [cdf.loc[vv].output_to[0] for vv in v]
            # and update input_from of these nodes too
            cdf.loc[[o[0] for o in newsource.output_to], "input_from"] = [nid] * len(
                newsource.output_to
            )
            cdf = pd.concat([cdf, pd.DataFrame([newsource.toDict()]).set_index("id")]).drop(v)

            cdf.loc[nid, "source_id"] = k

        # turn every input_from that's a single int into a list
        cdf.input_from = cdf.input_from.apply(lambda x: [x] if isinstance(x, int) else x)
        return cdf

    def __addAggregations(self, cdf, uidGen):
        assert self.aggregations is not None, "No aggregations in network"
        for i, r in self.aggregations.iterrows():
            if len(r.source) > 1:
                nid = uidGen()
                newaggregation = GraphComputeNode(nid, "aggregation", None, r.source)
                # find the compute node id through the source_id column
                try:
                    newaggregation.output_to = [
                        (cdf[cdf.source_id == s].index[0], 0) for s in r.source
                    ]
                except Exception as e:
                    msg = f"Error while adding aggregation node {nid} to compute graph"
                    msg += f" (recipe {self.name}, aggregation {i}, sources {r.source})"
                    msg += f"\n\naggregations: \n{self.aggregations}\n"
                    msg += f"\n{e}"
                    msg += f"\n{cdf}"
                    raise NetworkConstructionError(msg)
                # problem: some sources have no ids!!

                # add the input_from to the cooresponding sources
                for s in r.source:
                    for source in cdf[cdf.source_id == s].index:
                        cdf.at[source, "input_from"] = [nid]

                # For aggregations, we will store a dictionnary with the name of the aggregation and the ratio of each source
                tmp = pd.DataFrame([newaggregation.toDict()]).set_index("id")
                tmp["extra"] = [{"id": i, "qtty": np.sum(r.ratio), "ratios": r.ratio}]
                cdf = pd.concat([cdf, tmp])

            else:
                # no need for an aggregation node if there is only one source
                cdf.loc[cdf.source_id == r.source[0], "extra"] = [{"qtty": np.sum(r.ratio)}]

        return cdf

    def __buildRawGraph(self, uidGen: Callable[[], int]) -> List[GraphComputeNode]:
        cdg = self.central_dogma_graph
        logger.debug(f"Building compute graph for recipe {self.name}")
        logger.debug(f"cdg: \n{cdg}\n")
        assert cdg is not None, "central dogma graph not built"
        assert isinstance(cdg, pd.DataFrame), f"cdg is not a DataFrame: {type(cdg)}"
        newnodes = []

        # we start building the compute graph by adding the output:
        output_gene_nodes = cdg[cdg.is_output]
        onode = GraphComputeNode(uidGen(), "output", [], None)
        for i, r in output_gene_nodes.iterrows():
            onode.cdg_input += [i]
        newnodes.append(onode)

        tu_in_sequestron = set()

        # then we add the sequestron nodes with an associated list of their cdg input nodes
        enabled_sequestrons = self.lib.get_enabled_sequestrons()
        for _, r in enabled_sequestrons.iterrows():
            # sequestrons have 2 inputs: negative and positive
            nlvl = cdg[cdg.type == r.negative_level]  # negative level (PRT, RNA, DNA)
            nparts = nlvl[nlvl.content.apply(lambda x: r.negative_part in x)]

            plvl = cdg[cdg.type == r.positive_level]  # positive level (PRT, RNA, DNA)
            pparts = plvl[plvl.content.apply(lambda x: r.positive_part in x)]

            olvl = cdg[cdg.type == r.output_level]  # output level (PRT, RNA, DNA)
            oparts = olvl[olvl.content.apply(lambda x: ut.isSubset(r.output_part, x))]
            if len(nparts) > 0 and len(pparts) > 0 and len(oparts.index) > 0:
                if len(pparts) != 1:
                    msg = f"Found {len(pparts)} positive parts for sequestron {r.type} (expected 1)"
                    msg += f"\n\nparts: {pparts}"
                    raise NetworkConstructionError(msg)

                if len(nparts) != 1:
                    msg = f"Found {len(nparts)} negative parts for sequestron {r.type} (expected 1)"
                    msg += f"\n\nparts: {nparts}"
                    raise NetworkConstructionError(msg)
                try:
                    cnode = GraphComputeNode(
                        uidGen(),
                        f"sequestron_{r.type}",
                        [int(nparts.index[0]), int(pparts.index[0])],
                        int(oparts.index[0]),
                    )
                    # we get a unique name for the  sequestron by concatenating the name of the negative and positive parts
                    cnode.extra = {"seq_name": f"{r.type}::{r.negative_part}#{r.positive_part}"}
                    newnodes.append(cnode)
                    # useful to track which tus are in use
                    tu_in_sequestron.update(oparts["tu_id"].iloc[0])
                    tu_in_sequestron.update(nparts["tu_id"].iloc[0])
                    tu_in_sequestron.update(pparts["tu_id"].iloc[0])
                except Exception as e:
                    msg = f"Error while building compute graph for recipe {self.name}:\n{e}"
                    msg += f"Sequestron {r.type}.\nnparts: {nparts}, pparts: {pparts}, oparts: {oparts}"
                    msg += f"\nolevel: {olvl}, oparts: {oparts}, outlevel: {r.output_level}, outpart: {r.output_part}"
                    raise NetworkConstructionError(msg)

        cg = []
        # pretty format of newnodes
        from pprint import pformat

        newnodes_str = pformat([n.toDict() for n in newnodes])

        logger.debug(f"tu_in_sequestron: {tu_in_sequestron}")
        logger.debug(f"newnodes: {newnodes_str}")

        # right now newnodes contains only the output node and the sequestron nodes
        # we're now going to go upstream from these nodes, adding the relevant translation/transcription
        # nodes until we reach the Transcription Units.

        # Before that, let's also add dead-end paths if there are any.
        # Dead-end paths start with a transcription unit and end with a gene that is not an output.
        # They're practically useless (and we can optimize them out during computation) but they're still
        # part of the graph (and merit being visualized).
        # We add all proteins that are not outputs, and whose tu is not part of any sequestron.

        # Eventually, "deadend" should receive more attention, as they will be actual interesting payloads
        # instead of simply fluorescent proteins. So we can't just wish them away.
        deadend_nodes = cdg[
            (cdg.is_output == False)
            & (cdg.type == "PRT")
            & (cdg.tu_id.apply(lambda x: all([xx not in tu_in_sequestron for xx in x])))
        ]
        for i, r in deadend_nodes.iterrows():
            cnode = GraphComputeNode(uidGen(), "deadend", [i], None)
            newnodes.append(cnode)

        # At first, each TU also has a corresponding source node that we need to add.
        # We will merge sources later (for TUs that are on a same plasmid)
        while newnodes:
            n: GraphComputeNode = newnodes.pop()
            logger.debug(f"Processing node {n.id} of type {n.type}")
            if n.type != "source":
                # for every cdg node that is an input of n
                if n.cdg_input is None:
                    msg = f"Error while building compute graph for recipe {self.name}:\n"
                    msg += f"No cdg_input for node {n.id} of type {n.type}:\n{n}"
                    msg += f"Content of its cdg_output node:\n{cdg.loc[n.cdg_output]}"
                    msg += f"CDG:\n{cdg}"
                    raise NetworkConstructionError(msg)
                for i, n_inp in enumerate(n.cdg_input):
                    # list all other nodes that also output n_inp
                    others = self.__isOutputedBy(n_inp, cg + newnodes)
                    for other in others:
                        # establish the connection between n and its parents
                        n.input_from += [other.id]
                        other.output_to += [(n.id, i)]
                    if not others:  # if n_inp is not outputed by any node we have already created
                        # then we go up the central dogma and create the matching upstream node
                        # gn is the central dogma graph node that is being transformed by this compute node
                        upstream_cdg_node = cdg.loc[n_inp]
                        nid = uidGen()
                        # we just need to know what type of transform we have.
                        # for example, if the input cdg node that our compute node expects is a protein,
                        # that means that we need to add a translation node, etc...
                        ntype = {"PRT": "translation", "RNA": "transcription", "DNA": "source"}[
                            upstream_cdg_node.type
                        ]
                        newn = GraphComputeNode(
                            nid, ntype, upstream_cdg_node.predecessor, int(n_inp)
                        )
                        newn.input_from = []
                        newn.output_to = [(n.id, i)]
                        newnodes.append(newn)
                        n.input_from += [int(nid)]
                        logger.debug(
                            f"Added node {newn.id} of type {newn.type}: {pformat(newn.toDict())}"
                        )
            cg += [n]
        return cg

    def __addNumericNodes(self, cdf, uidGen):
        # we add 1 numeric node per source or aggregation that's "at the top",
        # i.e its input_from is empty.
        logger.debug(f"comp graph before adding numeric nodes: \n{cdf}")
        topnodes = cdf[cdf.input_from.apply(len) == 0]
        logger.debug(f"Adding numeric nodes for {len(topnodes)} top nodes: {topnodes}")
        for i, r in topnodes.iterrows():
            nid = uidGen()
            newnode = GraphComputeNode(nid, "numeric", None, 1)
            newnode.output_to = [(i, 0)]
            tmp = pd.DataFrame([newnode.toDict()]).set_index("id")
            extra = {
                "role": "copy_number",
            }
            if r.extra is not None and "qtty" in r.extra:
                extra["qtty"] = r.extra["qtty"]
            tmp["extra"] = [extra]
            cdf = pd.concat([cdf, tmp])
            # don't forget to add the new node as input_from to the top node
            cdf.loc[i, "input_from"] = [[nid]]
        return cdf

    def __build_compute_graph(self):
        assert self.central_dogma_graph is not None, "central dogma graph not built yet"

        uidGen = ut.uniqueIdGenerator()

        # build the graph of interacting nodes, without any optimization,
        # basically the dual of the central dogma graph:
        cg = self.__buildRawGraph(uidGen)

        # remove shortcuts and cycles:
        self.__removeShortcuts(cg, 0)

        # convert to dataframe
        self.compute_graph = pd.DataFrame([n.toDict() for n in cg]).set_index("id").sort_index()

        # first sanity check:
        # there should be the same number of sources in the cdf compute graph as DNA nodes in the cdg
        nsources = len(self.compute_graph[self.compute_graph.type == "source"])
        ndna = len(self.central_dogma_graph[self.central_dogma_graph.type == "DNA"])
        if nsources != ndna:
            dna_in_compute_graph = self.compute_graph[
                self.compute_graph.type == "source"
            ].cdg_output.tolist()

            extradna = []
            for i, r in self.central_dogma_graph[self.central_dogma_graph.type == "DNA"].iterrows():
                if i not in dna_in_compute_graph:
                    extradna += r.tu_id
            msg = f"When building compute graph for recipe {self.name}, "
            msg += f"found {nsources} DNA sources in the graph, but {ndna} DNA nodes total."
            msg += f"\nExtra DNA nodes: {extradna}"
            raise NetworkConstructionError(msg)

        self.compute_graph = self.__mergeSources(
            self.compute_graph, uidGen
        )  # merge TUs with same source

        self.compute_graph = self.__addAggregations(
            self.compute_graph, uidGen
        )  # add aggregation nodes

        self.compute_graph = self.__addNumericNodes(
            self.compute_graph, uidGen
        )  # now add numeric nodes (constant or inuts)

        self.cleanup()
        self._sanity_check()

    #                                                                            }}}
    ## ─────────────────────────────────────────────────────────────────────────────

    def get_output_compute_node(self) -> pd.Series:
        assert isinstance(self.compute_graph, pd.DataFrame), "Network not built"
        onode = self.compute_graph[self.compute_graph["type"] == "output"]
        assert len(onode) == 1, f"Invalid number of output nodes: {len(onode)}"
        return onode.iloc[0]

    def get_output_proteins(self) -> List[str]:
        """Returns the names of the proteins that are outputs of the network"""
        if not hasattr(self, "_output_proteins") or self._output_proteins is None:
            onode = self.get_output_compute_node()
            if "cdg_input" not in onode:
                raise ValueError(f"Invalid output node: {onode}")
            assert isinstance(
                self.central_dogma_graph, pd.DataFrame
            ), "get_output_proteins: Central dogma graph not built"
            self._output_proteins = [
                self.central_dogma_graph.loc[cdg_id]["content"][0] for cdg_id in onode["cdg_input"]
            ]
        return self._output_proteins

    def get_nb_outputs(self) -> int:
        if self._n_outputs is None:
            self._n_outputs = len(self.get_output_proteins())
        return self._n_outputs

    def get_nb_inputs(self):
        if self._n_inputs is None:
            self._n_inputs = len(self.get_inverted_input_proteins())
        return self._n_inputs

    # TODO: proper cached, cleaner properties
    @property
    def nb_outputs(self) -> int:
        return self.get_nb_outputs()

    @property
    def nb_inputs(self) -> int:
        return self.get_nb_inputs()

    def get_input_from_output(self, output_arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Given an array of output values, returns the columns that are inputs of the inverted network,
        properly ordered by input number"""
        # In inverted networks, each input node has,
        # in its extra, 'input_from_output' and 'input_position'
        # (which get_inverted_input_positions uses)
        # We want to transform output_arr by reordering the columns accordingly
        if output_arr is None:
            return None
        mapping = self.get_inverted_input_positions()
        return output_arr[:, [mapping[i] for i in range(len(mapping))]]

    def get_inverted_input_proteins(self, include_biases: bool = False) -> List[str]:
        """Returns the names of the proteins that are inputs of the inverted network, ordered"""
        mapping = self.get_inverted_input_positions(include_biases)
        output_proteins = self.get_output_proteins()
        assert len(mapping) <= len(output_proteins), f"Invalid mapping: {mapping}"
        return [output_proteins[mapping[i]] for i in range(len(mapping))]

    def get_inverted_input_positions(self, include_biases: bool = False) -> Dict[int, int]:
        """Returns a mapping from input position to output position"""
        assert isinstance(self.compute_graph, pd.DataFrame), "Compute graph not built"
        mapping = {}  # input number -> output position
        mask = self.compute_graph["type"] == "input"
        if include_biases:
            mask = mask | (self.compute_graph["type"] == "bias")
        inputs = self.compute_graph[mask]
        for _, row in inputs.iterrows():
            assert "input_position" in row.extra, f"input_position not in {row.extra}"
            assert "input_from_output" in row.extra, f"input_from_output not in {row.extra}"
            assert row.extra["input_position"] not in mapping
            mapping[row.extra["input_position"]] = row.extra["input_from_output"]
        assert set(mapping.keys()) == set(range(len(mapping.keys()))), f"Invalid mapping: {mapping}"
        assert len(mapping.keys()) == len(set(mapping.values())), f"Invalid mapping: {mapping}"
        return mapping

    def get_dependent_output_proteins(self) -> List[str]:
        all_outputs = self.get_output_proteins()
        input_proteins = self.get_inverted_input_proteins(include_biases=True)
        return [p for p in all_outputs if p not in input_proteins]

    def set_input_as_bias(self, input_protein_name: Sequence[str]) -> None:
        """Sets this input protein as a bias node (instead of an input one)"""
        original_mapping = self.get_inverted_input_positions()
        output_proteins = self.get_output_proteins()
        assert (
            input_protein_name in output_proteins
        ), f"Invalid input protein name: {input_protein_name}"
        output_position = output_proteins.index(input_protein_name)
        assert output_position in original_mapping.values()
        assert isinstance(self.compute_graph, pd.DataFrame)
        inputs = self.compute_graph[self.compute_graph["type"] == "input"]
        found = False
        for i, row in inputs.iterrows():
            assert "input_position" in row.extra, f"input_position not in {row.extra}"
            assert "input_from_output" in row.extra, f"input_from_output not in {row.extra}"
            if row.extra["input_from_output"] == output_position:
                self.compute_graph.at[i, "type"] = "bias"
                found = True
                break
        assert found, f"Could not find input protein {input_protein_name} in compute graph"
        new_mapping = self.get_inverted_input_positions()
        assert len(new_mapping) == len(original_mapping) - 1, f"Invalid mapping: {new_mapping}"
        assert output_position not in new_mapping.values()
        assert len(self.get_inverted_input_proteins()) == len(new_mapping)

    def compute_node_is_upstream_of(self, node_id: int, other_node_id: int) -> bool:
        """Returns True if node_id is upstream of other_node_id"""
        assert isinstance(self.compute_graph, pd.DataFrame)
        if node_id == other_node_id:
            return True
        node = self.compute_graph.loc[node_id]
        if node.type == "output":
            return False
        for downstream_id, _ in node["output_to"]:
            if self.compute_node_is_upstream_of(downstream_id, other_node_id):
                return True
        return False

    def sort_nodes_by_upstream(self, nodes: Sequence[int]) -> List[int]:
        from functools import cmp_to_key

        def custom_cmp(a, b):
            if self.compute_node_is_upstream_of(a, b):
                return -1
            elif self.compute_node_is_upstream_of(b, a):
                return 1
            else:
                return 0

        return sorted(nodes, key=cmp_to_key(custom_cmp))

    def topological_order(self, nodes: Optional[Sequence[int]] = None) -> List[List[int]]:
        """Returns a list of lists of compute nodes from the network,
        where each node of a sublist can be computed independently of the others,
        but each sublist must be computed in order."""
        assert isinstance(self.compute_graph, pd.DataFrame)
        visited = set()
        batches = []
        nodes = set(nodes) if nodes is not None else set(self.compute_graph.index)
        while len(visited) < len(self.compute_graph):
            independent = [
                i
                for i, row in self.compute_graph.iterrows()
                if (not row["input_from"] or all([x[0] in visited for x in row["input_from"]]))
                and i not in visited
            ]
            if not independent:
                msg = f"No independent node. Remaining:{set(self.compute_graph.index) - visited}. Visited:{visited}"
                raise ValueError(msg)
            visited.update(independent)
            batches.append([i for i in independent if i in nodes])
        return [b for b in batches if len(b) > 0]

    def cleanup(self):
        if self.compute_graph is not None:
            self.compute_graph.source_id = self.compute_graph.source_id.apply(
                lambda x: str(x) if not pd.isnull(x) else None
            )
            self.compute_graph = self.compute_graph.replace({np.nan: None})
            self.compute_graph.cdg_input = self.compute_graph.cdg_input.apply(
                lambda x: [int(x)] if isinstance(x, int) else x
            )

            # reconstruct all input_froms from the output_to:

            # first make sure input_from is of the right size
            for i, r in self.compute_graph.iterrows():
                output_to_me = self.compute_graph[
                    self.compute_graph.output_to.apply(lambda x: i in [y[0] for y in x])
                ]
                try:
                    self.compute_graph.at[i, "input_from"] = [None] * len(output_to_me)
                except Exception as e:
                    msg = f"Error cleaning up compute graph: {e}\n"
                    msg += f"Trying to construct input_froms of node {i} from upstream outputs.\n"
                    msg += f"{r}"
                    msg += f"\nDetected upstream outputs:\n{output_to_me}"
                    raise NetworkConstructionError(msg)

            # then fill it
            for i, r in self.compute_graph.iterrows():
                for p, o in enumerate(r.output_to):
                    try:
                        self.compute_graph.at[o[0], "input_from"][o[1]] = (i, p)
                    except Exception as e:
                        msg = f"Error cleaning up compute graph: {e}\n"
                        msg += f"currently processing {o}.\n"
                        msg += f"\ninput node is:\n{r}"
                        msg += f"\noutput node is:\n{self.compute_graph.loc[o[0]]}"
                        raise NetworkConstructionError(msg)

            # make sure the proper quantile variable is assigned to each node
            self._assign_quantile_variable()

        self._sanity_check()

    def _sanity_check(self):
        # check that all nodes have a unique id
        if self.compute_graph is not None:
            assert len(self.compute_graph.index) == len(
                set(self.compute_graph.index)
            ), "compute graph has duplicate ids"

            # every source node should have a source_id
            for i, r in self.compute_graph[self.compute_graph.type == "source"].iterrows():
                if r.source_id is None:
                    msg = (
                        f"In compute graph for recipe {self.name}, source node {i} has no source_id"
                    )
                    msg += f"\n{self.compute_graph}"
                    raise NetworkConstructionError(msg)

        if self.central_dogma_graph is not None:
            assert len(self.central_dogma_graph.index) == len(
                set(self.central_dogma_graph.index)
            ), "central dogma graph has duplicate ids"

    def get_signature(self):
        signature = f"{self.name}:{self.metadata}\n"
        for k in sorted(self.transcription_units.keys()):
            signature += f"{k}: {self.transcription_units[k]}\n"
        signature += f"{self.raw_tu_in_sources}\n"
        signature += f"{self.raw_aggregations}"
        return signature

    def _assign_quantile_variable(self):
        """
        Assigns the correct quantile variable to each node of the compute graph.
        Proceeds by propagating from the output towards the upstream nodes.
        Inverted nodes are assigned the same quantile as their forward node.
        If a node is linked is not linked to a non-inverted one, has only one output
        but is linked downstream to paths that lead to multiple outputs,
        quantile_variable_id is set to [None].
        """
        cg = self.compute_graph

        def propagate_upstream(node, quantile_id, output_id):
            node["extra"].setdefault("quantile_variable_id", [])
            if node.is_inverse_of is not None:
                node["extra"]["quantile_variable_id"] = cg.loc[node.is_inverse_of]["extra"].get(
                    "quantile_variable_id", []
                )
            else:
                if len(node["extra"]["quantile_variable_id"]) <= output_id:
                    # append -1 until the right size
                    node["extra"]["quantile_variable_id"].extend(
                        [-1] * (output_id - len(node["extra"]["quantile_variable_id"]) + 1)
                    )
                if node["extra"]["quantile_variable_id"][output_id] == -1:
                    node["extra"]["quantile_variable_id"][output_id] = quantile_id
                else:
                    # another node already set the quantile var!
                    # It means we found a node with a single output but linked to multiple downstream
                    # paths. We could append the quantile var but the order would be random. I prefer
                    # to remove the footgun entirely and just not add a quantile variable for this node.
                    # At the time I'm writing this the only case that would happen are for the numeric nodes
                    # or the inputs. They definitely don't need the quantile var.
                    # We change the existing value to None for this special case.
                    node["extra"]["quantile_variable_id"][output_id] = None

            if node.input_from:
                for nid, oid in node.input_from:
                    propagate_upstream(cg.loc[nid], quantile_id, oid)

        # first let's remove all "quantile_variable_id" from the extra column:
        for _, node in cg.iterrows():
            node["extra"].pop("quantile_variable_id", None)

        output_node = cg[cg.type == "output"].iloc[0]
        # add the quantile variable to the output node
        output_node["extra"]["quantile_variable_id"] = list(range(len(output_node.input_from)))
        self._n_outputs = len(output_node.input_from)
        for i, (nid, oid) in enumerate(output_node.input_from):
            propagate_upstream(cg.loc[nid], i, oid)

        # treat the case where we have a "deadend" node, i.e a branch that ends
        # without being connected to the output node. The node type is litteraly "deadend"
        # we'll just assign quantile 0
        deadend_nodes = cg[cg.type == "deadend"].index
        for node_id in deadend_nodes:
            propagate_upstream(cg.loc[node_id], 0, 0)


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                       --     the inverter     --
# ···············································································
# When training the intrinsic parameters of the model (aka the simulation part),
# we actually don't really have data about the numeric values involved (which would
# logically be the inputs of the network, as they most likely represent copy numbers).
# Instead, training data is only a list of output fluorescence.
# The model needs copy numbers, or at least some number correlated to copy numbers,
# in order to compute the output.
# One solution is to make sure there's always an invertible path that links some component of
# the output to the copy numbers.
# When traning from xp data, we prepend an inverter module to the network,
# and we use (part of) the output as both input and target.

# The plan:
# -> define the inverse version of some compute nodes
# -> take a model, and find all invertible path from copy numbers to output
# -> prepend the invertible paths to the model, define inputs from output

DEFAULT_INVERSE_DICT = {
    "translation": "inv_translation",
    "transcription": "inv_transcription",
    "numeric": "inv_numeric",
    "aggregation": "inv_aggregation",
    "source": "inv_source",
}


def get_invertible_paths(network, start_node_id, inverse_dict):
    def _is_invertible(node):
        invertible = node.type in inverse_dict and len(node["input_from"]) <= 1
        return invertible

    paths = []

    # we want ALL paths from start_node_id to output nodes that consist of invertible nodes
    # we store the path as a list of (this_node_id, this_output_id) tuples except for the last node
    # (the output node), where we store (output_node_id, input_id) instead
    def _get_invertible_paths(network, node_id, path, visited, input_slot=None):
        nonlocal paths
        node = network.compute_graph.loc[node_id]

        if node.type == "output":
            # we reached an output node, we store the path and stop the search
            # output is a special case where the slot tells which output is used
            assert input_slot is not None
            paths.append(path + [(node_id, input_slot)])
            return

        if node_id not in visited and _is_invertible(node):
            visited.add(node_id)
            for output_slot, (downstream_id, downstream_input_slot) in enumerate(node["output_to"]):
                _get_invertible_paths(
                    network,
                    downstream_id,
                    path + [(node_id, output_slot)],
                    visited,
                    input_slot=downstream_input_slot,
                )

    _get_invertible_paths(network, start_node_id, [], set())

    return paths


def inverted_network(
    network: Network,
    nodes: str = "auto",
    inverse_dict=DEFAULT_INVERSE_DICT,
    mode="shortest",
    use_cache=None,
):
    logger.debug(f"Inverting network {network.name}")

    # inverse_dict: node_type -> inverse_node_type

    # First we pick the start nodes. (Numeric nodes by default, or user supplied list)
    # We will then try to find invertible paths that go from each of the start nodes to the output.
    # A path is invertible if each of its nodes has been marked as having an inverted equivalent in the inverse_dict
    # We then prepend an input node + the inverted nodes to the original network, and that's what we
    # call an inverted network.
    if nodes == "auto":  # numeric nodes as start nodes
        start_nodes = network.compute_graph[
            network.compute_graph["type"] == "numeric"
        ].index.tolist()
    elif not isinstance(nodes, Iterable):
        raise ValueError(f"Unrecognized node mode: {nodes}. Use 'auto' or a list of node ids.")
    else:  # list of nodes
        start_nodes = nodes

    def _inverted_network():
        # we compute a list of invertible paths that link each start nodes to the output
        inv_paths = {n: get_invertible_paths(network, n, inverse_dict) for n in start_nodes}

        # For each start_node, we might have more than one path.
        # In 'shortest' mode, we just pick the shortest one.
        # In the 'all' mode, we want to return every possible combination of paths per start node
        # e.g. if we have 2 start nodes, and 2 paths for each, we want to return 4 paths
        # (the cartesian product of the paths)
        if mode == "shortest":
            inversions = [{n: min(p, key=len) for n, p in inv_paths.items() if p}]
        elif mode == "all":
            inversions = [dict(zip(inv_paths.keys(), p)) for p in product(*inv_paths.values())]
        else:
            raise ValueError(f"Unrecognized mode: {mode}. Use 'shortest' or 'all'.")

        new_networks = []
        for paths in inversions:
            inputpos = 0
            new_network = network.copy()
            uidGen = ut.uniqueIdGenerator(start=new_network.compute_graph.index.max() + 1)
            for start_n, path in paths.items():
                assert len(path) > 1, "path should not be empty"
                assert start_n == path[0][0], "first node of path should be the start node"

                # first we remove the start node
                original_node = new_network.compute_graph.loc[
                    start_n
                ]  # the non inverted start node
                assert path[0][1] == 0, "first node of path should have output slot 0"
                # the output_to column is a list of tuples (node_id, slot_id). We just need the node_id
                connected_node_id = original_node["output_to"][0][0]
                new_network.compute_graph.drop(start_n, inplace=True)

                prev = connected_node_id

                for i, (node_id, output_slot) in enumerate(path[1:], 1):
                    # slot is output_id for nodes, input_id for output
                    original_node = new_network.compute_graph.loc[node_id]  # the non inverted node
                    n_type = original_node["type"]  # its type
                    nid = uidGen()  # the new node id

                    if n_type == "output":  # special case when we reach the output
                        assert i == len(path) - 1, "output node should be the last node in the path"
                        # we add an input node
                        in_n = GraphComputeNode(nid, "input", None, None)
                        in_n.output_to = [
                            (prev, 0)
                        ]  # the input node is connected to the last inverted node
                        in_n.input_from = []  # the input node has no input
                        in_n.extra = {"input_from_output": output_slot, "input_position": inputpos}
                        inputpos += 1
                        new_network.compute_graph = pd.concat(
                            [
                                new_network.compute_graph,
                                pd.DataFrame([in_n.toDict()]).set_index("id"),
                            ]
                        )
                        break

                    # General case, create a new node and prepend to prev
                    cdg_in = new_network.compute_graph.loc[prev, "cdg_output"]
                    if isinstance(cdg_in, list):
                        cdg_in = cdg_in[output_slot]
                    new_n = GraphComputeNode(
                        nid, inverse_dict[n_type], cdg_in, original_node.cdg_output
                    )
                    new_n.output_to = [(prev, 0)]

                    # inverse nodes always have only one input and one output
                    # but we need to store the original output slot id in the extra field
                    # so that we can use it when converting aggregation nodes for example
                    # (where we convert a single input / multi output node to a single input / single output node
                    # but we need to know which path, i.e slot, to use)
                    new_n.is_inverse_of = node_id
                    new_n.extra = {
                        "original_output_slot": output_slot,
                        "original_output_len": len(original_node["output_to"]),
                    }

                    # set prev input_from to new nodes
                    new_network.compute_graph.at[prev, "input_from"] = [(nid, 0)]
                    new_network.compute_graph = pd.concat(
                        [new_network.compute_graph, pd.DataFrame([new_n.toDict()]).set_index("id")]
                    )

                    prev = nid

            new_network.cleanup()
            new_networks.append(new_network)
        return new_networks

    signature = f"{nodes}::{mode}::{inverse_dict}::{network.get_signature()}"

    return ut.get_cache(lambda: _inverted_network(), signature, cache_location=use_cache)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
