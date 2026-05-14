"""Pure Pydantic schema for the parts database.

Each model is the source of truth for one parts-db table — used by dracon as a
typed YAML tag (!Part, !L0, !L1, ...) and as the validation surface for both
direct construction and field-by-field FK checks (performed in library.py).

No SQL: the parts database is yaml-on-disk with a pickle cache (see
biocomp.library). Field aliases (5'UTR, 3'UTR) are pure Pydantic — they survive
yaml round-trip via the alias_generator on Pydantic's ConfigDict.
"""

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator
import json5


# Legacy gsheet columns 5'UTR / 3'UTR are not valid Python identifiers,
# so we expose them via aliases and use utr5 / utr3 on the Python side.
ALIASES = {"utr5": "5'UTR", "utr3": "3'UTR", "uid": "UID"}


class PartsRecord(BaseModel):
    """Common base for every parts-db record class.

    Pydantic config enables aliasing (`5'UTR`) and lets construction accept
    either the alias or the python attribute name (populate_by_name).
    """

    model_config = ConfigDict(
        alias_generator=lambda field_name: ALIASES.get(field_name, field_name),
        populate_by_name=True,
    )


def _coerce_json_list(v):
    """Pre-validator for list[str] fields that were historically JSON-encoded strings.

    Kept for resilience: a hand-edited yaml file or a hand-written record dict
    can still pass `'["x","y"]'` and we'll accept it. New records arrive as
    real lists.
    """
    if isinstance(v, str):
        parsed = json5.loads(v)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")
        return [str(x) for x in parsed]
    return v


class Category(PartsRecord):
    name: str
    transcripted: bool
    translated: bool


class Part(PartsRecord):
    name: str
    category: str


class L0(PartsRecord):
    id: str
    notes: Optional[str] = None
    constructed: bool
    backbone: str
    part_1: Optional[str] = None
    part_2: Optional[str] = None
    part_3: Optional[str] = None
    part_4: Optional[str] = None
    part_5: Optional[str] = None
    part_6: Optional[str] = None


class L1(PartsRecord):
    id: str
    notes: Optional[str] = None
    constructed: bool
    backbone: str
    insulator: Optional[str] = None
    promoter: Optional[str] = None
    utr5: Optional[str] = None
    gene: Optional[str] = None
    utr3: Optional[str] = None
    terminator: Optional[str] = None


class L2(PartsRecord):
    id: str
    notes: Optional[str] = None
    constructed: bool
    backbone: str
    slot_1: Optional[str] = None
    slot_2: Optional[str] = None
    slot_3: Optional[str] = None
    slot_4: Optional[str] = None
    slot_5: Optional[str] = None
    slot_6: Optional[str] = None


class SequestronType(PartsRecord):
    name: str
    negative_category: str
    positive_category: str
    negative_level: str
    positive_level: str
    output_level: str
    output_side: str
    output_category: list[str] = Field(default_factory=list)
    parameter_list: list[str] = Field(default_factory=list)

    @field_validator("output_category", "parameter_list", mode="before")
    @classmethod
    def _parse_json_list(cls, v):
        return _coerce_json_list(v)


class Sequestron(PartsRecord):
    id: int
    type: str
    negative_part: str
    positive_part: str
    output_part: list[str] = Field(default_factory=list)

    @field_validator("output_part", mode="before")
    @classmethod
    def _parse_json_list(cls, v):
        return _coerce_json_list(v)
