# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from typing import Any

RATIO_SCHEMA_VERSION = 1


def _to_int_slot(slot_key: Any) -> int:
    try:
        return int(slot_key)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid ratio slot key: {slot_key!r}") from exc


def _normalize_slot_entry(slot: int, entry: dict[str, Any]) -> dict[str, Any]:
    source_id = entry.get("source_id")
    if source_id is None or source_id == "":
        raise ValueError(f"ratio_schema slot {slot}: missing source_id")
    source_id = str(source_id)
    ratio = float(entry.get("ratio", 1.0))
    ratio_range = entry.get("ratio_range")
    locked = bool(entry.get("locked", False))
    member_uid = entry.get("member_uid")
    if member_uid is None or member_uid == "":
        member_uid = source_id
    else:
        member_uid = str(member_uid)
    return {
        "member_uid": member_uid,
        "source_id": source_id,
        "ratio": ratio,
        "ratio_range": ratio_range,
        "locked": locked,
    }


def _sorted_slots(raw_slots: dict[Any, Any]) -> list[tuple[int, dict[str, Any]]]:
    slots: list[tuple[int, dict[str, Any]]] = []
    for raw_slot, raw_entry in raw_slots.items():
        slot = _to_int_slot(raw_slot)
        if not isinstance(raw_entry, dict):
            raise ValueError(f"ratio_schema slot {slot}: entry must be dict, got {type(raw_entry)}")
        slots.append((slot, _normalize_slot_entry(slot, raw_entry)))
    slots.sort(key=lambda x: x[0])
    return slots


def _assert_contiguous_slots(sorted_slots: list[tuple[int, dict[str, Any]]]) -> None:
    for expected, (slot, _) in enumerate(sorted_slots):
        if slot != expected:
            raise ValueError(
                f"ratio_schema slots must be contiguous from 0..n-1; got slot {slot} at position {expected}"
            )


def _assert_unique_sources(sorted_slots: list[tuple[int, dict[str, Any]]]) -> None:
    sources = [entry["source_id"] for _, entry in sorted_slots]
    if len(sources) != len(set(sources)):
        raise ValueError(f"ratio_schema has duplicate source_id entries: {sources}")


def _normalized_slots_dict(slots: list[tuple[int, dict[str, Any]]]) -> dict[int, dict[str, Any]]:
    return {slot: dict(entry) for slot, entry in slots}


def get_ratio_schema(extra: dict | Any, *, require: bool = True) -> dict[str, Any]:
    schema = extra.get("ratio_schema")
    if schema is None:
        if require:
            raise KeyError("Aggregation node missing ratio_schema")
        return {"version": RATIO_SCHEMA_VERSION, "slots": {}}
    if not isinstance(schema, dict):
        raise ValueError(f"ratio_schema must be dict, got {type(schema)}")
    version = schema.get("version")
    if version != RATIO_SCHEMA_VERSION:
        raise ValueError(f"Unsupported ratio_schema version {version}, expected {RATIO_SCHEMA_VERSION}")
    slots = schema.get("slots")
    if slots is None:
        raise ValueError("ratio_schema missing 'slots'")
    if not isinstance(slots, dict):
        raise ValueError(f"ratio_schema['slots'] must be dict, got {type(slots)}")
    normalized = _sorted_slots(slots)
    _assert_contiguous_slots(normalized)
    _assert_unique_sources(normalized)
    return {"version": RATIO_SCHEMA_VERSION, "slots": _normalized_slots_dict(normalized)}


def get_ordered_slots(extra: dict | Any, *, require: bool = True) -> list[tuple[int, dict[str, Any]]]:
    schema = get_ratio_schema(extra, require=require)
    return [(slot, dict(entry)) for slot, entry in schema["slots"].items()]


def get_slot_entries(extra: dict | Any, *, require: bool = True) -> list[dict[str, Any]]:
    return [entry for _, entry in get_ordered_slots(extra, require=require)]


def set_ratio_schema_slots(extra: dict | Any, slots: list[dict[str, Any]]) -> None:
    normalized = [_normalize_slot_entry(slot, entry) for slot, entry in enumerate(slots)]
    extra["ratio_schema"] = {
        "version": RATIO_SCHEMA_VERSION,
        "slots": {slot: entry for slot, entry in enumerate(normalized)},
    }


def set_ratio_schema(extra: dict | Any, schema: dict[str, Any]) -> None:
    normalized = get_ratio_schema({"ratio_schema": schema}, require=True)
    extra["ratio_schema"] = normalized


def source_slot_map(extra: dict | Any) -> dict[str, int]:
    return {entry["source_id"]: slot for slot, entry in get_ordered_slots(extra)}


def slot_count(extra: dict | Any) -> int:
    return len(get_ordered_slots(extra))


def source_ids_in_slot_order(extra: dict | Any) -> list[str]:
    return [entry["source_id"] for _, entry in get_ordered_slots(extra)]


def slot_arrays(
    extra: dict | Any,
) -> tuple[list[str], list[float], list[dict[str, Any] | None], list[bool]]:
    slots = get_slot_entries(extra)
    return (
        [entry["source_id"] for entry in slots],
        [float(entry.get("ratio", 1.0)) for entry in slots],
        [entry.get("ratio_range") for entry in slots],
        [bool(entry.get("locked", False)) for entry in slots],
    )


def update_slots_from_arrays(
    extra: dict | Any,
    *,
    ratios: list[float],
    ratio_ranges: list[dict[str, Any] | None] | None = None,
    locked: list[bool] | None = None,
    remove_zero: bool = False,
    zero_eps: float = 1e-8,
) -> None:
    slots = [dict(entry) for entry in get_slot_entries(extra)]
    if len(ratios) != len(slots):
        raise ValueError(
            f"ratio vector length {len(ratios)} does not match slot count {len(slots)}"
        )
    if ratio_ranges is not None and len(ratio_ranges) != len(slots):
        raise ValueError(
            f"ratio_ranges length {len(ratio_ranges)} does not match slot count {len(slots)}"
        )
    if locked is not None and len(locked) != len(slots):
        raise ValueError(f"locked length {len(locked)} does not match slot count {len(slots)}")

    updated_slots: list[dict[str, Any]] = []
    for idx, entry in enumerate(slots):
        ratio_val = float(ratios[idx])
        if remove_zero and abs(ratio_val) <= zero_eps:
            continue
        entry["ratio"] = ratio_val
        if ratio_ranges is not None:
            entry["ratio_range"] = ratio_ranges[idx]
        if locked is not None:
            entry["locked"] = bool(locked[idx])
        updated_slots.append(entry)

    set_ratio_schema_slots(extra, updated_slots)


def remove_sources_and_renormalize(extra: dict | Any, removed_source_ids: set[str]) -> None:
    if not removed_source_ids:
        return
    slots = [entry for _, entry in get_ordered_slots(extra)]
    kept = [dict(entry) for entry in slots if entry["source_id"] not in removed_source_ids]
    if not kept:
        set_ratio_schema_slots(extra, [])
        return
    total = sum(float(entry.get("ratio", 0.0)) for entry in kept)
    if total > 1e-9:
        for entry in kept:
            entry["ratio"] = float(entry.get("ratio", 0.0)) / total
    set_ratio_schema_slots(extra, kept)
