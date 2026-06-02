# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
import re

from biocomp.utils import flatten
from collections.abc import Sequence as ABCSequence
from pathlib import Path
from typing import Annotated, Literal
from biocomp.library import LibraryContext, PartsLibrary, get_l0_parts, get_l1_parts, get_l1_from_l2
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_serializer,
    model_validator,
)

from biocomp.logging_config import get_logger
from biocomp.part_embeddings import EMBEDDINGS_BY_NAME, EMBEDDINGS_BY_CATEGORY

logger = get_logger(__name__)
PathLike = str | Path

RATIO_PRECISION = 5  # precision for ratio rounding (number of decimal places)


## {{{                      --     NumRange & FluoIntensity     --


class NumRange(BaseModel):
    """Range specification for unlocked numeric parameters (ratios, biases)

    If min or max is None, it means no limit in that direction.
    If init is provided, it sets the initial value for optimization.
    """

    min: float | None = None
    max: float | None = None
    init: float | None = None

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
    value: NumRange | float = Field(default_factory=lambda: NumRange(min=0.0))
    protein: str | None = None  # if None, assumes there's a single marker protein in the TU
    units: str = "AU"  # e.g. "EBFP2 mapped PacBlue"

    def is_locked(self) -> bool:
        """Check if bias is locked (fixed value) or unlocked (range)"""
        return isinstance(self.value, int | float)

    def get_value(self) -> float | None:
        """Get the fixed value if locked, None if unlocked"""
        if self.is_locked():
            return float(self.value)  # type: ignore
        return None

    def get_range(self) -> NumRange | None:
        """Get the range if unlocked, None if locked"""
        if not self.is_locked():
            return self.value  # type: ignore
        return None


##────────────────────────────────────────────────────────────────────────────}}}


## {{{                          --     RatioSpec     --

DEFAULT_RATIO_MIN = 0.0125
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
    min: float | None = None
    max: float | None = None
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

    def to_num_range(self) -> NumRange | None:
        if self.locked:
            return None
        return NumRange(min=self.min, max=self.max)

    def __repr__(self) -> str:
        if self.locked:
            return f"RatioSpec({self.value}, locked)"
        return f"RatioSpec({self.value}, [{self.min}, {self.max}])"


def _convert_ratio_value(v) -> NumRange | float | RatioSpec:
    """Convert a ratio value from YAML/dict to proper type."""
    if isinstance(v, NumRange | RatioSpec):
        return v
    if isinstance(v, int | float):
        return round(float(v), RATIO_PRECISION)
    if isinstance(v, dict):
        if "min" in v and "max" in v and "value" not in v:
            return NumRange(**v)
        return RatioSpec(**v)
    raise ValueError(f"Cannot convert {type(v)} to ratio: {v}")


def _convert_ratios_input(
    ratios_input: list | dict | None, units: list["TranscriptionUnit"]
) -> list[NumRange | float | RatioSpec] | None:
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
        result: list[NumRange | float | RatioSpec] = []
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

    part: str | list[str | None] | None = None

    # does this slot map to a parameter aka embedding, like "tl_rate" or "tc_rate"?
    maps_to_parameter: str | None = Field(default=None, exclude=True)

    # unique identifier for shared ("linked") parts across transcription units
    ref_id: str | None = Field(
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

    def __mapped_parameter(self, part_name: str | None) -> str | None:
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


SlotType = Annotated[Slot | str | list[str | None], BeforeValidator(convert_to_slot)]

##────────────────────────────────────────────────────────────────────────────}}}


## {{{                    --     Transcription Unit     --
class TranscriptionUnit(BaseModel):
    name: str = ""
    slots: list[SlotType] = []
    params: dict = Field(default_factory=dict, exclude=True)  # param name -> value
    source: str | None = None  # plasmid name, for example
    position_in_source: int | None = None
    no_masking: bool = False  # if True, TU cannot be disabled during design (always mask=1)
    param_ref_ids: dict[str, str | None] = Field(
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

    def get_part_names(self) -> set[str]:
        """All unique part name strings across all slots."""
        parts: set[str] = set()
        for slot in self.slots:
            assert isinstance(slot, Slot)
            p = slot.part
            if isinstance(p, str):
                parts.add(p)
            elif isinstance(p, list):
                parts.update(name for name in p if isinstance(name, str))
        return parts

    def to_parts(self) -> list[str | list[str]]:
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
    name: str | None = None
    units: list[Unit]
    ratios: RatiosInput | None = None
    fluo_bias: FluoIntensity | None = None

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

    def _get_ratio_range(self, r: RatioType) -> NumRange | None:
        if isinstance(r, NumRange):
            return r
        if isinstance(r, RatioSpec):
            return r.to_num_range()
        return None

    def has_unlocked_ratios(self) -> bool:
        if self.ratios is None:
            return False
        return any(self._is_ratio_unlocked(r) for r in self.ratios)

    def get_locked_ratios(self) -> list[float] | None:
        if self.ratios is None:
            return None
        if self.has_unlocked_ratios():
            return None
        return [self._get_ratio_value(r) for r in self.ratios]

    def get_ratio_ranges(self) -> list[NumRange | None]:
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
        self, tu_index: int | str, wrt: int | str | None = None
    ) -> NumRange | float | None:
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


AxisLabel = Literal["x", "y", "z"]


class InputAxis(BaseModel):
    """One entry of a recipe's `input_axes` list: an input column identity
    (protein name or cotx group name) and an optional plot-axis label.

    Names are resolved against a Network at build time -- either form works
    and may be mixed within one list.
    """

    name: str
    axis: AxisLabel | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_scalar(cls, v):
        if isinstance(v, str):
            return {"name": v}
        if isinstance(v, InputAxis):
            return v.model_dump()
        return v


def _parse_input_axes(v):
    if v is None:
        return None
    if isinstance(v, dict):
        return [{"name": k, "axis": a} for k, a in v.items()]
    if isinstance(v, list):
        return v
    raise ValueError(f"input_axes must be list, dict, or None -- got {type(v).__name__}")


class Recipe(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    name: str | None = None
    display_name: str | None = None
    metadata: dict | None = None
    content: CoTxList = []
    input_axes: Annotated[
        list[InputAxis] | None, BeforeValidator(_parse_input_axes)
    ] = None
    cell_type: str = "HEK293FT"

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_input_fields(cls, data):
        if not isinstance(data, dict):
            return data
        legacy_order = data.pop("input_order", None)
        legacy_axis_mapping = data.pop("axis_mapping", None)
        if data.get("input_axes") is not None:
            return data
        if legacy_order is None and legacy_axis_mapping is None:
            return data
        order = list(legacy_order or [])
        # Resolve each axis_mapping label onto a column name. A key is either a
        # real name (present in input_order, or a cotx/protein name resolved
        # downstream) or a positional slot token like `x1`/`x2` (1-based) that
        # legacy matrices paired with a protein-named input_order. A positional
        # key not present in input_order maps its label onto input_order[i-1].
        pos_rx = re.compile(r"^[a-zA-Z](\d+)$")
        label_for: dict[str, str] = {}
        extra: list[dict] = []
        for name, axis in (legacy_axis_mapping or {}).items():
            m = pos_rx.match(name)
            if name not in order and m and order:
                i = int(m.group(1)) - 1
                if 0 <= i < len(order):
                    label_for[order[i]] = axis
                    continue
            if name in order:
                label_for[name] = axis
            else:
                extra.append({"name": name, "axis": axis})
        axes: list[dict] = []
        for name in order:
            ax: dict = {"name": name}
            if name in label_for:
                ax["axis"] = label_for[name]
            axes.append(ax)
        axes.extend(extra)
        data["input_axes"] = axes
        return data

    @model_validator(mode="after")
    def _validate_input_axes(self) -> "Recipe":
        if not self.input_axes:
            return self
        names = [ax.name for ax in self.input_axes]
        assert len(names) == len(set(names)), (
            f"input_axes contains duplicate names: {names}"
        )
        labels = [ax.axis for ax in self.input_axes if ax.axis is not None]
        assert len(labels) == len(set(labels)), (
            f"input_axes contains duplicate axis labels: {labels}"
        )
        return self

    def strip_orphan_ern_proteins(self) -> "Recipe":
        """Remove ERN protein TUs that have no matching rec sites anywhere in the recipe.

        After design pruning, ERN protein TUs (CasE, Csy4, PgU) may survive even when
        all their target rec sites have been pruned. These orphan proteins are biologically
        useless and should be removed. Mutates in place and returns self.
        """
        from biocomp.nodes.ern import ERN_DEFAULT_NEG_PARTS, ERN_DEFAULT_POS_PARTS

        ern_to_recs = {
            neg: set(pos_list)
            for neg, pos_list in zip(ERN_DEFAULT_NEG_PARTS, ERN_DEFAULT_POS_PARTS, strict=True)
        }
        all_rec_names = {r for recs in ern_to_recs.values() for r in recs}

        present_recs: set[str] = set()
        for cotx in self.content:
            for tu in cotx.units:
                present_recs.update(tu.get_part_names() & all_rec_names)

        orphan_proteins = {
            ern for ern, recs in ern_to_recs.items() if not (recs & present_recs)
        }
        if not orphan_proteins:
            return self

        for cotx in self.content:
            original_units = cotx.units
            original_ratios = cotx.ratios
            keep_indices = [
                i for i, tu in enumerate(original_units)
                if not (tu.get_part_names() & orphan_proteins) or tu.no_masking
            ]
            for i, tu in enumerate(original_units):
                if i not in set(keep_indices):
                    logger.info(
                        f"Stripping orphan ERN protein TU '{tu.name}' "
                        f"(orphan proteins={tu.get_part_names() & orphan_proteins}) "
                        f"from cotx '{cotx.name}'"
                    )

            if len(keep_indices) < len(original_units):
                idx_map = {old: new for new, old in enumerate(keep_indices)}
                cotx.units = [original_units[i] for i in keep_indices]
                if original_ratios is not None and len(original_ratios) == len(original_units):
                    kept_ratios = [original_ratios[i] for i in keep_indices]
                    total = sum(cotx._get_ratio_value(r) for r in kept_ratios)
                    cotx.ratios = [
                        round(cotx._get_ratio_value(r) / total, RATIO_PRECISION)
                        for r in kept_ratios
                    ] if total > 0 else kept_ratios
                if cotx.fluo_bias is not None:
                    new_idx = idx_map.get(cotx.fluo_bias.tu_id)
                    cotx.fluo_bias = (
                        cotx.fluo_bias.model_copy(update={"tu_id": new_idx})
                        if new_idx is not None else None
                    )

        self.content = [c for c in self.content if c.units]
        return self

    @classmethod
    def load_from_paper_yaml(cls, path: "str | Path") -> "Recipe":
        """Load a Recipe from a paper-recipe YAML that may carry a leading
        `_metadata:` block before `!biocomp.recipe.Recipe`.

        Handles the design-pipeline convention where recipe files are written
        as `yaml.dump({'_metadata': ...}) + dracon.dump(recipe)` -- stripping
        the metadata preamble before handing the remainder to Dracon.
        """
        import dracon as dr

        content = Path(path).read_text()
        tag = '!biocomp.recipe.Recipe'
        if f'\n{tag}' in content:
            content = content[content.index(f'\n{tag}') + 1:]
        obj = dr.loads(content, context={'Recipe': cls, 'biocomp.recipe.Recipe': cls})
        if isinstance(obj, cls):
            return obj
        if hasattr(obj, 'get'):
            recipe = obj.get('recipe', obj)
            if isinstance(recipe, cls):
                return recipe
        raise TypeError(f"loaded object is not a Recipe: {type(obj)}")

    def has_input_axes(self) -> bool:
        return bool(self.input_axes)

    @property
    def input_order(self) -> list[str] | None:
        if not self.input_axes:
            return None
        return [ax.name for ax in self.input_axes]

    @property
    def axis_mapping(self) -> dict[str, str] | None:
        if not self.input_axes:
            return None
        labels = {ax.name: ax.axis for ax in self.input_axes if ax.axis is not None}
        return labels or None

    def has_input_order(self) -> bool:
        return self.has_input_axes()

    def has_axis_mapping(self) -> bool:
        return self.axis_mapping is not None

    def _cotx_to_marker_protein(self, candidates: set[str]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for cotx in self.content:
            if not cotx.name:
                continue
            for tu in cotx.units:
                for slot in reversed(tu.slots or []):
                    slot_name = slot.part if hasattr(slot, "part") else str(slot)
                    if isinstance(slot_name, str) and slot_name in candidates:
                        mapping[cotx.name] = slot_name
                        break
                if cotx.name in mapping:
                    break
        return mapping

    def resolve_input_axes(self, network) -> list[InputAxis] | None:
        """Resolve recipe-level `input_axes` (which may reference either cotx
        names or protein names) to a list anchored on the network's input
        proteins. Returns None if the recipe declares no axes or the network
        has no inputs. Raises ValueError on an unresolvable name.
        """
        if not self.has_input_axes():
            return None
        proteins = set(network.get_inverted_input_proteins())
        if not proteins:
            return None
        cotx_to_protein = self._cotx_to_marker_protein(proteins)
        resolved: list[InputAxis] = []
        for ax in self.input_axes:
            if ax.name in proteins:
                resolved.append(InputAxis(name=ax.name, axis=ax.axis))
            elif ax.name in cotx_to_protein:
                resolved.append(InputAxis(name=cotx_to_protein[ax.name], axis=ax.axis))
            else:
                raise ValueError(
                    f"input_axes name {ax.name!r} matches neither a network "
                    f"input protein ({sorted(proteins)}) nor a cotx marker "
                    f"({sorted(cotx_to_protein)})"
                )
        return resolved

    def resolve_input_order(self, network) -> list[str] | None:
        axes = self.resolve_input_axes(network)
        return [ax.name for ax in axes] if axes is not None else None


def default_input_order_for_network(network) -> list[str]:
    """Single source of truth for the default input_order heuristic.

    Used by:
    - the recipe migration tool to fill `input_order` on recipes that
      lack it on disk
    - `recipe_to_networks()` final fallback when a Recipe somehow reaches
      runtime without `input_order` set (and no axis_mapping derivation
      applies)

    Currently: freeze whatever the post-rewriting natural input order is.
    This makes the runtime fallback a structural no-op (apply_input_order
    becomes identity) while still anchoring `recipe.input_order` on disk
    against future graph-rewriting drift after migration. Replace this body
    to switch heuristics; nothing else needs to change.
    """
    return list(network.get_inverted_input_proteins())


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


def name_transcription_unit(tu: TranscriptionUnit, lib: PartsLibrary) -> str | None:
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
        display_name=recipe.display_name,
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
        expanded_units = []
        for unit in cotx.units:
            # Preserve TUs that already carry an explicit decomposition
            # (e.g. from an inline `slots:` in the recipe JSON5). Only
            # expand when the unit has no slots or a single trivial slot
            # equal to its plasmid name -- the shape produced by
            # `TranscriptionUnit(name=plasmid, source=plasmid)` when no
            # slots were given.
            slots = list(unit.slots or [])
            is_trivial = (
                not slots
                or (len(slots) == 1 and getattr(slots[0], "part", None) == unit.name)
            )
            if is_trivial:
                expanded_units.extend(
                    expand_tu_from_lib(unit.name, lib, source=unit.source)
                )
            else:
                expanded_units.append(unit)
        expanded_cotx_list.append(
            CoTransfection(name=cotx.name, units=expanded_units, ratios=cotx.ratios)
        )
    return Recipe(
        name=recipe.name,
        display_name=recipe.display_name,
        metadata=recipe.metadata,
        content=expanded_cotx_list,
        input_axes=recipe.input_axes,
        cell_type=recipe.cell_type,
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

    if "metadata" not in raw_recipe_object:
        raw_recipe_object["metadata"] = {}
    metadata = raw_recipe_object["metadata"]

    # Legacy JSON5/YAML may carry top-level `description`; fold it into metadata.
    if "description" in raw_recipe_object and "description" not in metadata:
        metadata["description"] = raw_recipe_object["description"]

    if metadata.get("description"):
        desc, desc_dict = parse_description(metadata["description"])
        if desc_dict:
            metadata.update(desc_dict)
        metadata["description"] = desc

    FIRST_CLASS_FIELDS = {"name", "display_name", "metadata", "content",
                          "input_axes", "input_order", "axis_mapping", "cell_type"}
    for k, v in raw_recipe_object.items():
        if k not in FIRST_CLASS_FIELDS and k != "description":
            print(f"Adding extra field '{k}' to recipe metadata")
            metadata[k] = v

    recipe_kwargs = dict(
        name=raw_recipe_object.get("name", f"recipe{len(raw_recipe_object)}"),
        metadata=metadata,
        content=cotxlist,
    )
    for optional_field in ("display_name", "input_axes", "input_order", "axis_mapping", "cell_type"):
        if optional_field in raw_recipe_object:
            recipe_kwargs[optional_field] = raw_recipe_object[optional_field]
    recipe = Recipe(**recipe_kwargs)
    return expand_all_tus_from_lib(recipe, lib)


##────────────────────────────────────────────────────────────────────────────}}}
