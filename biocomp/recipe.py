from biocomp.utils import flatten
from collections.abc import Sequence as ABCSequence
from pathlib import Path
from typing import Union, Optional, Annotated
from biocomp.library import LibraryContext, PartsLibrary, get_l0_parts, get_l1_parts, get_l1_from_l2
from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    model_serializer,
    model_validator,
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
    If init is provided, it sets the initial value for optimization.
    """

    min: Optional[float] = None
    max: Optional[float] = None
    init: Optional[float] = None

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


## {{{                          --     RatioSpec     --

DEFAULT_RATIO_MIN = 0.001
DEFAULT_RATIO_MAX = 10.0


class RatioSpec(BaseModel):
    """Specification for a single TU ratio in a CoTransfection.

    Allows explicit control over whether a ratio is locked (fixed) or unlocked (optimizable).

    Examples:
        # Locked ratio (fixed at 0.5)
        RatioSpec(value=0.5, locked=True)

        # Unlocked ratio with custom range
        RatioSpec(value=0.3, min=0.1, max=0.6)

        # Unlocked ratio with default range
        RatioSpec(value=0.3, locked=False)
    """

    value: float
    min: Optional[float] = None
    max: Optional[float] = None
    locked: bool = False

    @model_validator(mode="after")
    def _validate(self) -> "RatioSpec":
        if self.locked:
            if self.min is not None or self.max is not None:
                raise ValueError(
                    f"RatioSpec: locked=True is incompatible with min/max. "
                    f"Got locked=True with min={self.min}, max={self.max}"
                )
        else:
            if self.min is None:
                object.__setattr__(self, "min", DEFAULT_RATIO_MIN)
            if self.max is None:
                object.__setattr__(self, "max", DEFAULT_RATIO_MAX)
            assert self.min is not None and self.max is not None
            if self.min > self.max:
                raise ValueError(f"RatioSpec: min ({self.min}) > max ({self.max})")
            if not (self.min <= self.value <= self.max):
                raise ValueError(f"RatioSpec: value ({self.value}) not in [{self.min}, {self.max}]")
        return self

    def is_locked(self) -> bool:
        return self.locked

    def to_num_range(self) -> Optional[NumRange]:
        if self.locked:
            return None
        return NumRange(min=self.min, max=self.max)

    def __repr__(self) -> str:
        if self.locked:
            return f"RatioSpec({self.value}, locked)"
        return f"RatioSpec({self.value}, [{self.min}, {self.max}])"


def _convert_ratio_value(v) -> Union[NumRange, float, RatioSpec]:
    """Convert a ratio value from YAML/dict to proper type."""
    if isinstance(v, (NumRange, RatioSpec)):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), RATIO_PRECISION)
    if isinstance(v, dict):
        if "min" in v and "max" in v and "value" not in v:
            return NumRange(**v)
        return RatioSpec(**v)
    raise ValueError(f"Cannot convert {type(v)} to ratio: {v}")


def _convert_ratios_input(
    ratios_input: Union[list, dict, None], units: list["TranscriptionUnit"]
) -> Optional[list[Union[NumRange, float, RatioSpec]]]:
    """Convert ratios from various input formats to canonical list format.

    Supports:
    1. None -> None
    2. list[float|NumRange|RatioSpec|dict] -> list (legacy + new)
    3. dict[tu_name -> float|RatioSpec|dict] -> list (new dict syntax)
    """
    if ratios_input is None:
        return None

    if isinstance(ratios_input, list):
        return [_convert_ratio_value(r) for r in ratios_input]

    if isinstance(ratios_input, dict):
        tu_names = [u.name for u in units]
        result: list[Union[NumRange, float, RatioSpec]] = []
        for tu_name in tu_names:
            if tu_name not in ratios_input:
                raise ValueError(
                    f"Ratio dict missing TU '{tu_name}'. "
                    f"Available: {list(ratios_input.keys())}, expected: {tu_names}"
                )
            result.append(_convert_ratio_value(ratios_input[tu_name]))
        extra_keys = set(ratios_input.keys()) - set(tu_names)
        if extra_keys:
            raise ValueError(f"Ratio dict has unknown TU names: {extra_keys}. Expected: {tu_names}")
        return result

    raise ValueError(f"ratios must be list or dict, got {type(ratios_input)}")


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
    elif isinstance(value, str):
        return Slot(part=value)
    elif isinstance(value, ABCSequence):
        return Slot(part=list(value))
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
    no_masking: bool = False  # if True, TU cannot be disabled during design (always mask=1)
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

RatioType = NumRange | float | RatioSpec
RatiosInput = list[RatioType] | dict[str, float | dict]


class CoTransfection(BaseModel):
    name: Optional[str] = None
    units: list[Unit]
    ratios: Optional[RatiosInput] = None
    fluo_bias: Optional[FluoIntensity] = None

    @model_validator(mode="after")
    def _validate(self) -> "CoTransfection":
        if self.ratios is not None:
            if isinstance(self.ratios, dict):
                converted = _convert_ratios_input(self.ratios, self.units)
                object.__setattr__(self, "ratios", converted)
            elif isinstance(self.ratios, list):
                converted = [_convert_ratio_value(r) for r in self.ratios]
                object.__setattr__(self, "ratios", converted)
        if self.fluo_bias is not None:
            if self.fluo_bias.tu_id < 0 or self.fluo_bias.tu_id >= len(self.units):
                raise ValueError(
                    f"fluo_bias.tu_id {self.fluo_bias.tu_id} out of range [0, {len(self.units)})"
                )
        return self

    def _is_ratio_unlocked(self, r: RatioType) -> bool:
        if isinstance(r, NumRange):
            return True
        if isinstance(r, RatioSpec):
            return not r.is_locked()
        return False

    def _is_ratio_explicitly_locked(self, r: RatioType) -> bool:
        """Check if ratio is EXPLICITLY locked via RatioSpec(locked=True).

        This is different from "not unlocked" - a plain float value is neither
        explicitly locked nor explicitly unlocked. This distinction is important
        for design mode where random_init=True should unlock unspecified ratios
        but NOT override explicitly locked ones.
        """
        if isinstance(r, RatioSpec):
            return r.is_locked()
        return False

    def _get_ratio_value(self, r: RatioType) -> float:
        if isinstance(r, NumRange):
            if r.init is not None:
                return r.init
            return (r.min + r.max) / 2 if r.min is not None and r.max is not None else 1.0
        if isinstance(r, RatioSpec):
            return r.value
        return float(r)

    def _get_ratio_range(self, r: RatioType) -> Optional[NumRange]:
        if isinstance(r, NumRange):
            return r
        if isinstance(r, RatioSpec):
            return r.to_num_range()
        return None

    def has_unlocked_ratios(self) -> bool:
        if self.ratios is None:
            return False
        return any(self._is_ratio_unlocked(r) for r in self.ratios)

    def get_locked_ratios(self) -> Optional[list[float]]:
        if self.ratios is None:
            return None
        if self.has_unlocked_ratios():
            return None
        return [self._get_ratio_value(r) for r in self.ratios]

    def get_ratio_ranges(self) -> list[Optional[NumRange]]:
        if self.ratios is None:
            return [None] * len(self.units)
        return [self._get_ratio_range(r) for r in self.ratios]

    def get_ratio_values(self) -> list[float]:
        if self.ratios is None:
            return [1.0] * len(self.units)
        return [self._get_ratio_value(r) for r in self.ratios]

    def get_ratio_locked(self) -> list[bool]:
        """Get list of booleans indicating which ratios are explicitly locked.

        Returns True for ratios specified as RatioSpec(locked=True), False otherwise.
        This is used by design mode to distinguish between:
        - Explicitly locked ratios (should stay locked even with random_init=True)
        - Unspecified ratios (can be unlocked by random_init=True)
        """
        if self.ratios is None:
            return [False] * len(self.units)
        return [self._is_ratio_explicitly_locked(r) for r in self.ratios]

    def has_bias(self) -> bool:
        """Check if this cotx specifies a bias (not a normal input)"""
        return self.fluo_bias is not None

    def get_tu_ratio(
        self, tu_index: int | str, wrt: Optional[int | str] = None
    ) -> Optional[Union[NumRange, float]]:
        """Get the ratio for a specific TU by index or name, optionally relative to another TU."""
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
            return self._get_ratio_value(self.ratios[tu_index])

        tu_ratio = self.ratios[tu_index]
        tu_range = self._get_ratio_range(tu_ratio)
        tu_value = self._get_ratio_value(tu_ratio)

        if rel_index is None:
            wrt_value = 1.0
            wrt_range = None
        else:
            if rel_index < 0 or rel_index >= len(self.units):
                raise IndexError(f"wrt index {rel_index} out of range for cotx '{self.name}'")
            wrt_ratio = self.ratios[rel_index]
            wrt_value = self._get_ratio_value(wrt_ratio)
            wrt_range = self._get_ratio_range(wrt_ratio)

        if tu_range is not None:
            max_wrt = wrt_range.max if wrt_range else wrt_value
            min_wrt = wrt_range.min if wrt_range else wrt_value
            return NumRange(
                min=(tu_range.min / max_wrt) if tu_range.min is not None else None,
                max=(tu_range.max / min_wrt) if tu_range.max is not None else None,
            )
        return tu_value / wrt_value

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
    axis_mapping: Optional[dict[str, str]] = None  # cotx_name -> axis (x, y)

    @model_validator(mode="after")
    def _validate_input_order(self) -> "Recipe":
        if self.input_order is None:
            return self
        assert len(self.input_order) == len(set(self.input_order)), (
            f"input_order contains duplicates: {self.input_order}"
        )
        return self

    @model_validator(mode="after")
    def _validate_axis_mapping(self) -> "Recipe":
        if self.axis_mapping is None:
            return self
        valid_axes = {"x", "y"}
        for cotx, axis in self.axis_mapping.items():
            assert axis in valid_axes, f"axis_mapping[{cotx}] must be 'x' or 'y', got '{axis}'"
        return self

    def has_input_order(self) -> bool:
        """Check if recipe has explicit input order defined."""
        return self.input_order is not None and len(self.input_order) > 0

    def has_axis_mapping(self) -> bool:
        """Check if recipe has axis_mapping defined (for design scaffolds)."""
        return self.axis_mapping is not None and len(self.axis_mapping) > 0

    def get_input_order_from_axis_mapping(self) -> Optional[list[str]]:
        """Convert axis_mapping to input_order (x first, then y)."""
        if not self.has_axis_mapping():
            return None
        x_cotx = None
        y_cotx = None
        for cotx, axis in self.axis_mapping.items():
            if axis == "x":
                x_cotx = cotx
            elif axis == "y":
                y_cotx = cotx
        if x_cotx is None or y_cotx is None:
            return None
        return [x_cotx, y_cotx]


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
    sorted_flat = sorted(flat_parts)
    for l1_name in lib.L1s.index:
        l1_parts = get_l1_parts(l1_name, lib)
        if l1_parts and sorted(l1_parts) == sorted_flat:
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
