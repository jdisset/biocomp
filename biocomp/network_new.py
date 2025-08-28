from typing import (
    List,
    Dict,
    Iterable,
    Optional,
    Annotated,
    Union,
)
from pydantic import BaseModel, BeforeValidator, model_validator
from .utils import load_lib, flatten

from biocomp.logging_config import get_logger
from biocomp.library import PartsLibrary
from biocomp.graphengine import GraphState
from biocomp.graphrules import GraphRewritingRule, apply_rule

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
    name: Optional[str] = None
    compute_graph: Optional[GraphState] = None

    @property
    def input_names(self): ...


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
