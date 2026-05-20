# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Ratio utilities - SSOT for ratio decoding and normalization.

This module provides the single source of truth for ratio-related operations.
All ratio decoding (latent MLP vs direct) and normalization logic lives here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import jax.numpy as jnp

from .config import BIOCOMP_CONSTANTS

if TYPE_CHECKING:
    from .parameters import ParameterTree

RATIO_PRUNE_THRESHOLD = BIOCOMP_CONSTANTS["ratio"]["prune_threshold"]


def _decode_latent_ratios(z, W1, b1, W2, b2):
    """Decode latent code to ratios via MLP. No softmax - aggregation handles normalization."""
    h = jax.nn.gelu(W1 @ z + b1)
    return W2 @ h + b2


def decode_ratios(
    params: ParameterTree,
    namespace: str,
    node_idx: int,
    n_outputs: int,
) -> jnp.ndarray:
    """SSOT for ratio decoding. Works in JIT contexts.

    Uses Python `if` for mode dispatch - this is correct because
    `latent_z_path in params` is a trace-time constant (the param tree
    structure is fixed before JIT tracing).

    Args:
        params: ParameterTree containing ratio params
        namespace: Layer namespace (e.g., "local/0/aggregation_2_1")
        node_idx: Node index within the layer
        n_outputs: Number of outputs to decode

    Returns:
        Decoded and clipped ratios of shape (n_outputs,)
    """
    latent_z_path = f"{namespace}/latent_z"

    if latent_z_path in params:
        z = params[latent_z_path][node_idx]
        W1 = params[f"{namespace}/latent_W1"][node_idx]
        b1 = params[f"{namespace}/latent_b1"][node_idx]
        W2 = params[f"{namespace}/latent_W2"][node_idx]
        b2 = params[f"{namespace}/latent_b2"][node_idx]
        raw = _decode_latent_ratios(z, W1, b1, W2, b2)[:n_outputs]
    else:
        raw = params[f"{namespace}/ratios"][node_idx][:n_outputs]

    ratio_min = params[f"{namespace}/ratio_min"][node_idx][:n_outputs]
    ratio_max = params[f"{namespace}/ratio_max"][node_idx][:n_outputs]
    return jnp.clip(raw, ratio_min, ratio_max)


def decode_ratios_numpy(
    params: ParameterTree,
    namespace: str,
    node_idx: int,
    n_outputs: int,
) -> np.ndarray:
    """SSOT for ratio decoding outside JIT (numpy arrays).

    Used by design_pruning, commit, diagnostics.

    Args:
        params: ParameterTree containing ratio params
        namespace: Layer namespace (e.g., "local/0/aggregation_2_1")
        node_idx: Node index within the layer
        n_outputs: Number of outputs to decode

    Returns:
        Decoded and clipped ratios as numpy array of shape (n_outputs,)
    """
    latent_z_path = f"{namespace}/latent_z"

    if latent_z_path in params:
        z = np.asarray(params[latent_z_path][node_idx])
        W1 = np.asarray(params[f"{namespace}/latent_W1"][node_idx])
        b1 = np.asarray(params[f"{namespace}/latent_b1"][node_idx])
        W2 = np.asarray(params[f"{namespace}/latent_W2"][node_idx])
        b2 = np.asarray(params[f"{namespace}/latent_b2"][node_idx])
        raw = _decode_latent_ratios(z, W1, b1, W2, b2)[:n_outputs]
    else:
        raw = np.asarray(params[f"{namespace}/ratios"][node_idx][:n_outputs])

    ratio_min = np.asarray(params[f"{namespace}/ratio_min"][node_idx][:n_outputs])
    ratio_max = np.asarray(params[f"{namespace}/ratio_max"][node_idx][:n_outputs])
    return np.clip(raw, ratio_min, ratio_max)


def normalize_ratios_for_pruning(
    ratios,
    threshold: float = RATIO_PRUNE_THRESHOLD,
    eps: float = 1e-12,
):
    """SSOT for pruning normalization. Works with JAX or numpy arrays.

    Normalizes ratios by dividing by max absolute value, then zeros out
    values below threshold. This is used during hard pruning to identify
    which TUs have negligible contribution.

    Args:
        ratios: Array of ratios (any shape with last dim being the ratio dim)
        threshold: Values with normalized magnitude below this are zeroed
        eps: Small value to avoid division by zero

    Returns:
        Normalized ratios with small values zeroed out
    """
    xp = jnp if hasattr(ratios, 'at') else np

    A = xp.abs(ratios)
    if A.ndim == 0:
        return A

    if A.ndim == 1:
        A = A[None, :]
        m = xp.maximum(xp.max(A, axis=1, keepdims=True), eps)
        norm = A / m
        return xp.where(norm >= threshold, norm, 0.0).squeeze(0)

    if A.ndim > 2:
        orig_shape = A.shape
        A = A.reshape(-1, A.shape[-1])
        m = xp.maximum(xp.max(A, axis=1, keepdims=True), eps)
        norm = A / m
        return xp.where(norm >= threshold, norm, 0.0).reshape(orig_shape)

    m = xp.maximum(xp.max(A, axis=1, keepdims=True), eps)
    norm = A / m
    return xp.where(norm >= threshold, norm, 0.0)
