"""Shared parameter utilities used by tuner and design mode."""

import re
from typing import Any


def format_param_path(path: str, include_local: bool = False) -> str:
    """Convert parameter path to human-readable display name.

    Examples:
        "local/layer_6/ratios" -> "Layer 6 > Ratios"
        "local/layer_3/tl_rate" -> "Layer 3 > TL Rate"
    """
    parts = path.split("/")
    display_parts = []

    for part in parts:
        if part == "local" and not include_local:
            continue
        if part.startswith("layer_"):
            display_parts.append(f"Layer {part.replace('layer_', '')}")
        else:
            display_parts.append(part.replace("_", " ").title())

    return " > ".join(display_parts) if display_parts else path


def parse_indexed_path(path: str) -> tuple[str, list[int]]:
    """Parse path with array indices.

    Examples:
        "local/6/ratios[0][1]" -> ("local/6/ratios", [0, 1])
        "local/3/bias" -> ("local/3/bias", [])
    """
    match = re.match(r"(.+?)(\[(\d+)\])+$", path)
    if match:
        base_path = match.group(1)
        indices = [int(i) for i in re.findall(r"\[(\d+)\]", path)]
        return base_path, indices
    return path, []


def set_nested_value(obj: Any, indices: list[int], value: Any) -> None:
    """Set value at nested indices in array/list."""
    for idx in indices[:-1]:
        obj = obj[idx]
    obj[indices[-1]] = value
