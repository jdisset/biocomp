"""Commit helpers to separate structural pruning from quantization/collapse."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .compute import ComputeStack
    from .parameters import ParameterTree


def commit_structure(
    stack: "ComputeStack",
    params: "ParameterTree",
    **kwargs,
):
    """Commit only structural changes (pruning, graph cleanup) without collapsing embeddings."""
    return stack.commit(params, collapse_to_part=False, **kwargs)


def commit_final(
    stack: "ComputeStack",
    params: "ParameterTree",
    **kwargs,
):
    """Full commit including collapse/quantization to discrete parts."""
    return stack.commit(params, collapse_to_part=True, **kwargs)
