from biocomp.utils import flatten
from pathlib import Path
from typing import Union, Optional, Annotated
from biocomp.library import LibraryContext, PartsLibrary, get_l0_parts, get_l1_parts, get_l1_from_l2
from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    field_validator,
    model_validator,
    model_serializer,
)

from biocomp.logging_config import get_logger
from biocomp.part_embeddings import EMBEDDINGS_BY_NAME, EMBEDDINGS_BY_CATEGORY

logger = get_logger(__name__)
PathLike = Union[str, Path]

RATIO_PRECISION = 5  # precision for ratio rounding (number of decimal places)


## {{{                      --     NumRange & FluoIntensity     --


class NumRange(BaseModel):
    """Range specification for unlocked numeric parameters (ratios, biases)

    If min or max is None, it means no limit in that direction.
    """

    min: Optional[float] = None
    max: Optional[float] = None

    def contains(self, value: float) -> bool:
        """Check if value is within range"""
        if self.min is not None and value < self.min:
            return False
        if self.max is not None and value > self.max:
            return False
        return True

    def clamp(self, value: float) -> float:
        """Clamp value to range"""
        if self.min is not None:
            value = max(value, self.min)
        if self.max is not None:
            value = min(value, self.max)
        return value

    def __repr__(self) -> str:
        min_str = f"{self.min}" if self.min is not None else "-∞"
        max_str = f"{self.max}" if self.max is not None else "∞"
        return f"[{min_str}, {max_str}]"


class FluoIntensity(BaseModel):
    """Specification for fluorescence intensity bias in a CoTransfection

    Refers to a specific TU in the cotx that has a fluorescent reporter.
    Can be locked (fixed value) or unlocked (NumRange).
    """

    tu_id: int  # index of the TU in the cotx that has the fluorescent reporter
    value: Union[NumRange, float] = Field(default_factory=lambda: NumRange(min=0.0))
    protein: Optional[str] = None  # if None, assumes there's a single marker protein in the TU
    units: str = "AU"  # e.g. "EBFP2 mapped PacBlue"

    def is_locked(self) -> bool:
        """Check if bias is locked (fixed value) or unlocked (range)"""
        return isinstance(self.value, (int, float))

    def get_value(self) -> Optional[float]:
        """Get the fixed value if locked, None if unlocked"""
        if self.is_locked():
            return float(self.value)  # type: ignore
        return None

    def get_range(self) -> Optional[NumRange]:
        """Get the range if unlocked, None if locked"""
        if not self.is_locked():
            return self.value  # type: ignore
        return None


##────────────────────────────────────────────────────────────────────────────}}}


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
                if category in EMBEDDINGS_BY_CATEGORY:
                    return EMBEDDINGS_BY_CATEGORY[category].name
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
    position_in_source: Optional[int] = None
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
                if s.maps_to_parameter in EMBEDDINGS_BY_NAME:
                    default = EMBEDDINGS_BY_NAME[s.maps_to_parameter].default_part
                    if isinstance(s.part, list):
                        self.params[s.maps_to_parameter] = [
                            default if p is None else p for p in s.part
                        ]
                    else:
                        self.params[s.maps_to_parameter] = [default if s.part is None else s.part]
                else:
                    self.params[s.maps_to_parameter] = (
                        [s.part] if not isinstance(s.part, list) else s.part
                    )
                # track ref_id for this parameter
                self.param_ref_ids[s.maps_to_parameter] = s.ref_id

        # add default parameters
        for emb in EMBEDDINGS_BY_NAME.values():
            if emb.name not in self.params:
                self.params[emb.name] = [emb.default_part]
                self.param_ref_ids[emb.name] = None  # default parameters have no ref_id

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
    ratios: Optional[list[Union[NumRange, float]]] = None
    fluo_bias: Optional[FluoIntensity] = None  # if None, normal input (not a bias)

    @field_validator("ratios", mode="before")
    @classmethod
    def round_ratios(cls, v):
        if v is None:
            return v
        return [
            round(float(r), RATIO_PRECISION)
            if isinstance(r, (int, float)) and not isinstance(r, NumRange)
            else r
            for r in v
        ]

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        # Don't set default ratios here - they should be based on unique sources,
        # which requires network context. Defaults are handled in network.py
        # Validate fluo_bias tu_id if present
        if self.fluo_bias is not None:
            if self.fluo_bias.tu_id < 0 or self.fluo_bias.tu_id >= len(self.units):
                raise ValueError(
                    f"fluo_bias.tu_id {self.fluo_bias.tu_id} out of range [0, {len(self.units)})"
                )
        # Note: ratios are NOT required to sum to 1.0 at recipe level - they represent
        # relative weights that get normalized during network building/aggregation

    def has_unlocked_ratios(self) -> bool:
        """Check if any ratio is unlocked (NumRange)"""
        return self.ratios is not None and any(isinstance(r, NumRange) for r in self.ratios)

    def get_locked_ratios(self) -> Optional[list[float]]:
        """Get ratios as floats if all are locked, None otherwise"""
        if self.ratios is None:
            return None
        if self.has_unlocked_ratios():
            return None
        return [float(r) for r in self.ratios]  # type: ignore

    def get_ratio_ranges(self) -> list[Optional[NumRange]]:
        """Get ratio range for each unit (None if locked)"""
        if self.ratios is None:
            return [None] * len(self.units)
        return [r if isinstance(r, NumRange) else None for r in self.ratios]

    def has_bias(self) -> bool:
        """Check if this cotx specifies a bias (not a normal input)"""
        return self.fluo_bias is not None

    def get_tu_ratio(
        self, tu_index: int | str, wrt: Optional[int | str] = None
    ) -> Optional[Union[NumRange, float]]:
        """Get the ratio for a specific TU by index or name, optionally relative to another TU"""
        rel_index = None
        if isinstance(tu_index, str):
            tu_indices = [i for i, tu in enumerate(self.units) if tu.name == tu_index]
            if not tu_indices:
                raise ValueError(f"TU with name '{tu_index}' not found in cotx '{self.name}'")
            if len(tu_indices) > 1:
                raise ValueError(f"Multiple TUs with name '{tu_index}' found in cotx '{self.name}'")
            tu_index = tu_indices[0]
        if isinstance(wrt, str):
            wrt_indices = [i for i, tu in enumerate(self.units) if tu.name == wrt]
            if not wrt_indices:
                raise ValueError(f"TU with name '{wrt}' not found in cotx '{self.name}'")
            if len(wrt_indices) > 1:
                raise ValueError(f"Multiple TUs with name '{wrt}' found in cotx '{self.name}'")
            rel_index = wrt_indices[0]
        elif isinstance(wrt, int):
            rel_index = wrt
        else:
            assert wrt is None
        if self.ratios is None:
            return 1.0
        if tu_index < 0 or tu_index >= len(self.units):
            return self.ratios[tu_index]
        if rel_index is None:
            wrt_ratio = 1.0
        else:
            if rel_index < 0 or rel_index >= len(self.units):
                raise IndexError(f"wrt index {rel_index} out of range for cotx '{self.name}'")
            wrt_ratio = self.ratios[rel_index]
        tu_ratio = self.ratios[tu_index]
        max_wrt = wrt_ratio.max if isinstance(wrt_ratio, NumRange) else wrt_ratio
        min_wrt = wrt_ratio.min if isinstance(wrt_ratio, NumRange) else wrt_ratio
        if isinstance(tu_ratio, NumRange):
            return NumRange(
                min=(tu_ratio.min / max_wrt) if tu_ratio.min is not None else None,
                max=(tu_ratio.max / min_wrt) if tu_ratio.max is not None else None,
            )
        else:
            return tu_ratio / wrt_ratio

    def __hash__(self):
        return hash(str(self.model_dump()))


def process_cotx_list(cotx_list: list[CoTransfection | dict]) -> list[CoTransfection | dict]:
    """Add names to unnamed cotx groups and sources"""

    source_counter = 0

    for i, cotx in enumerate(cotx_list):
        # handle both dict (from YAML loading) and CoTransfection objects
        if isinstance(cotx, dict):
            if cotx.get("name") is None:
                cotx["name"] = f"cotx_{i + 1}"
            units = cotx.get("units", [])
            for unit in units:
                if isinstance(unit, dict):
                    if unit.get("source") is None:
                        source_counter += 1
                        unit["source"] = f"plsmd_{source_counter}"
                elif unit.source is None:
                    source_counter += 1
                    unit.source = f"plsmd_{source_counter}"
        else:
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
    input_order: Optional[list[str]] = None  # ordered list of input protein names

    @model_validator(mode="after")
    def _validate_input_order(self) -> "Recipe":
        if self.input_order is None:
            return self
        assert len(self.input_order) == len(set(self.input_order)), (
            f"input_order contains duplicates: {self.input_order}"
        )
        return self

    def has_input_order(self) -> bool:
        """Check if recipe has explicit input order defined."""
        return self.input_order is not None and len(self.input_order) > 0


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
