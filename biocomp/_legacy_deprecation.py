# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""SSOT for legacy-plotting deprecation warnings.

Use ``warn_legacy(symbol, replacement)`` from any legacy entry point. The
warning fires once per (symbol, caller) pair to keep noisy callsites readable.
"""

import warnings

_SEEN: set[tuple[str, str]] = set()


def warn_legacy(symbol: str, replacement: str, *, stacklevel: int = 3) -> None:
    key = (symbol, replacement)
    if key in _SEEN:
        return
    _SEEN.add(key)
    warnings.warn(
        f"{symbol} is deprecated; use {replacement} instead. "
        "Pending removal once all paper-jobs YAML files migrate to jeanplot panels.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def reset_seen() -> None:
    _SEEN.clear()
