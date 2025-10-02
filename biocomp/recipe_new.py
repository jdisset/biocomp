from biocomp.utils import flatten
from pathlib import Path
from typing import Union, Optional, Annotated
from biocomp.library import LibraryContext, PartsLibrary, get_l0_parts, get_l1_parts, get_l1_from_l2
from pydantic import (
    BaseModel,
    BeforeValidator,
    model_validator,
    Field,
    field_serializer,
    model_serializer,
)

from biocomp.logging_config import get_logger

logger = get_logger(__name__)
PathLike = Union[str, Path]


PART_TYPE_TO_EMBEDDING_NAME = {"promoter": "tc_rate", "uORF_group": "tl_rate"}
EMBEDDING_TO_DEFAULT_PART = {"tl_rate": "00_empty_tc", "tc_rate": "hEF1a"}

## {{{                           --     Slot     --


class Slot(BaseModel):
    """Transcription Units are made of slots which contain either a part or a list of
    possible parts that map to a quantized parameter"""

    part: Optional[Union[str, list[Optional[str]]]] = None

    # does this slot map to a parameter aka embedding, like "tl_rate" or "tc_rate"?
    maps_to_parameter: Optional[str] = Field(default=None, exclude=True)

    # unique identifier for shared ("linked") parts across transcription units
    ref_id: Optional[str] = Field(
        default=None, description="Reference ID for shared parts", exclude=True
    )

    def __init__(self, value=None, **data):
        # Allow shorthand: Slot("foo") or Slot(["a","b"])
        if not data and value is not None and not isinstance(value, dict):
            if isinstance(value, str):
                data = {"part": value}
            else:
                data = {"part": list(value)}
        super().__init__(**data)

    @model_serializer(mode="plain")
    def _serialize(self):
        v = self.part
        # collapse single-item list to a string
        if isinstance(v, list) and len(v) == 1:
            return v[0]
        return v

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


SlotType = Annotated[Union[Slot, str, list[Optional[str]]], BeforeValidator(convert_to_slot)]

##────────────────────────────────────────────────────────────────────────────}}}


## {{{                    --     Transcription Unit     --
class TranscriptionUnit(BaseModel):
    name: str = ""
    slots: list[SlotType] = []
    params: dict = Field(default_factory=dict, exclude=True)  # param name -> value
    source: Optional[str] = None  # plasmid name, for example
    position_in_source: int = 0
    param_ref_ids: dict[str, Optional[str]] = Field(
        default_factory=dict, exclude=True
    )  # param name -> ref_id

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

    def to_parts(self) -> list[Union[str, list[str]]]:
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
    units: list[Unit]
    ratios: Optional[list[float]] = None

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        if self.ratios is None:  # equal ratios by default
            self.ratios = [1.0] * len(self.units)

    def __hash__(self):
        return hash(str(self.model_dump()))


def process_cotx_list(cotx_list: list[CoTransfection]) -> list[CoTransfection]:
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


CoTxList = Annotated[list[CoTransfection], BeforeValidator(process_cotx_list)]


##────────────────────────────────────────────────────────────────────────────}}}


class Recipe(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[dict] = None
    content: CoTxList = []


## {{{                      --     recipe loading     --


def expand_tu_from_lib(names, lib: PartsLibrary, **tu_args) -> list[TranscriptionUnit]:
    """
    Expand a list of name (only if size 1) into a list of TranscriptionUnits.
    Start by looking for L0 parts, then L1 parts, then L2 parts. If no parts are found,
    return a TU with the original part name(s).
    """
    if not isinstance(names, list):
        names = [names]
    if len(names) != 1 or names[0] in lib.parts.index:
        return [TranscriptionUnit(slots=names, **tu_args)]
    name = names[0]
    if parts := get_l0_parts(name, lib):
        return [TranscriptionUnit(slots=parts, **tu_args)]
    if parts := get_l1_parts(name, lib):
        tu_args["name"] = tu_args.get("name", name)
        return [TranscriptionUnit(slots=parts, **tu_args)]
    if tunames := get_l1_from_l2(name, lib):
        return flatten(
            [
                expand_tu_from_lib([name], lib, position_in_source=i, **tu_args)
                for i, name in enumerate(tunames)
                if name
            ]
        )
    return [TranscriptionUnit(slots=names, **tu_args)]


def name_transcription_unit(tu: TranscriptionUnit, lib: PartsLibrary) -> Optional[str]:
    parts = tu.to_parts()
    flat_parts = []
    for part in parts:
        if isinstance(part, list):
            flat_parts.extend(part)
        else:
            flat_parts.append(part)
    flat_parts = [p for p in flat_parts if p and p != ""]

    matching_l1s = []
    for l1_name in lib.L1s.index:
        l1_parts = get_l1_parts(l1_name, lib)
        if l1_parts and l1_parts == flat_parts:
            matching_l1s.append(l1_name)

    if not matching_l1s:
        return None
    if len(matching_l1s) == 1:
        return matching_l1s[0]

    if tu.source and tu.source in matching_l1s:
        return tu.source
    return matching_l1s[0]


def rename_all_L1_tus(recipe: Recipe, lib: PartsLibrary) -> Recipe:
    """
    Rename all TranscriptionUnits in a recipe based on L1 library matches.
    If a TU matches an L1, it gets the L1's name. Otherwise, it gets a generic name "tu_{i}".
    Returns a new Recipe with renamed TranscriptionUnits.
    """
    renamed_cotx_list = []
    tu_counter = 1

    for cotx in recipe.content:
        renamed_units = []

        for unit in cotx.units:
            l1_name = name_transcription_unit(unit, lib)

            if l1_name:
                new_name = l1_name
            else:
                new_name = f"tu_{tu_counter}"
                tu_counter += 1

            renamed_unit = TranscriptionUnit(name=new_name, slots=unit.slots, source=unit.source)
            renamed_units.append(renamed_unit)
        renamed_cotx = CoTransfection(name=cotx.name, units=renamed_units, ratios=cotx.ratios)
        renamed_cotx_list.append(renamed_cotx)

    return Recipe(
        name=recipe.name,
        description=recipe.description,
        metadata=recipe.metadata,
        content=renamed_cotx_list,
    )


def parse_description(desc):
    import re

    desc_dict = {}
    pattern = r"ng DNA\s*=\s*([\d\.]+)"
    matches = re.findall(pattern, desc)
    for match in matches:
        value = match
        desc_dict["ng_dna"] = float(value)
        desc = desc.replace(f"ng DNA = {value}", "").strip()
    return desc, desc_dict


def expand_all_tus_from_lib(recipe: Recipe, lib: PartsLibrary) -> Recipe:
    expanded_cotx_list = []
    for cotx in recipe.content:
        expanded_units = flatten(
            [expand_tu_from_lib(unit.name, lib, source=unit.source) for unit in cotx.units]
        )
        expanded_cotx_list.append(
            CoTransfection(name=cotx.name, units=expanded_units, ratios=cotx.ratios)
        )
    return Recipe(
        name=recipe.name,
        description=recipe.description,
        metadata=recipe.metadata,
        content=expanded_cotx_list,
    )


def dict_to_recipe(raw_recipe_object):
    lib = LibraryContext.get_library()
    cotxlist = []
    cnt = raw_recipe_object.get("content", [])
    for i, c in enumerate(cnt):
        ratios = [s["ratio"] for s in c["sources"]]
        units = []
        for s in c["sources"]:
            if "slots" in s:
                tu = TranscriptionUnit(
                    name=s.get("name", s["plasmid"]), slots=s["slots"], source=s["plasmid"]
                )
            else:
                assert "name" not in s, "Cannot specify 'name' without 'slots'"
                tu = TranscriptionUnit(name=s["plasmid"], source=s["plasmid"])
            units.append(tu)
        cotxlist.append(
            CoTransfection(name=c.get("name", f"cotx{i + 1}"), units=units, ratios=ratios)
        )

    if "description" in raw_recipe_object:
        desc, desc_dict = parse_description(raw_recipe_object["description"])
        if "metadata" not in raw_recipe_object:
            raw_recipe_object["metadata"] = {}
        if desc_dict:
            raw_recipe_object["metadata"].update(desc_dict)
        raw_recipe_object["description"] = desc

    metadata = raw_recipe_object.get("metadata", {})
    for k, v in raw_recipe_object.items():
        if k not in ["name", "description", "metadata", "content"]:
            print(f"Adding extra field '{k}' to recipe metadata")
            metadata[k] = v

    recipe = Recipe(
        name=raw_recipe_object.get("name", f"recipe{len(raw_recipe_object)}"),
        description=raw_recipe_object.get("description"),
        metadata=raw_recipe_object.get("metadata"),
        content=cotxlist,
    )
    return expand_all_tus_from_lib(recipe, lib)


##────────────────────────────────────────────────────────────────────────────}}}
