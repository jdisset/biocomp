# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Context embedding system for per-network conditioning.

Provides a "context bus" -- a flat embedding vector resolved from categorical
context variables (e.g. cell_type) that conditions every MLP via dense_mlp's
`context` parameter.

Adding a new context variable = add one entry to CONTEXT_EMBEDDINGS.
No node code changes needed.
"""

import jax
import jax.numpy as jnp
from pydantic import BaseModel
from jax.typing import ArrayLike

from biocomp.parameters import ParameterTree
from biocomp.logging_config import get_logger

NON_GRAD_TAG = (
    "non_grad"  # same constant as nodeutils.NON_GRAD_TAG, inlined to avoid circular import
)

logger = get_logger(__name__)


# ── Definition ────────────────────────────────────────────────────────────────


class ContextEmbedding(BaseModel):
    """A categorical context variable that becomes a trainable embedding."""

    name: str
    available_values: list[str]
    default_value: str
    embedding_dim: int = 2


CONTEXT_EMBEDDINGS: list[ContextEmbedding] = [
    ContextEmbedding(
        name="cell_type",
        available_values=["HEK293FT"],
        default_value="HEK293FT",
        embedding_dim=2,
    ),
]

CONTEXT_BY_NAME: dict[str, ContextEmbedding] = {e.name: e for e in CONTEXT_EMBEDDINGS}


def total_context_dim() -> int:
    return sum(e.embedding_dim for e in CONTEXT_EMBEDDINGS)


# ── Shared variational primitive ──────────────────────────────────────────────


def variational_codebook_lookup(
    mean: ArrayLike,
    logstdev: ArrayLike,
    key: ArrayLike,
    disable_variational: bool = False,
    min_logstdev: float = -10.0,
    max_logstdev: float = 5.0,
) -> tuple[jnp.ndarray, dict]:
    """Codebook entry + learned Gaussian noise.

    Shared primitive for both context embeddings and part embeddings (e.g. ERN affinity).
    """
    logstdev = jnp.clip(logstdev, min_logstdev, max_logstdev)
    if disable_variational:
        noise = jnp.zeros_like(mean)
    else:
        noise = jax.random.normal(key, jnp.asarray(mean).shape) * jnp.exp(logstdev)
    value = mean + noise
    return value, {"mean": mean, "logstdev": logstdev, "noise": noise}


# ── Parameter paths ───────────────────────────────────────────────────────────


def _codebook_means_path(name: str) -> str:
    return f"shared/context/{name}/codebook_means"


def _codebook_logstdevs_path(name: str) -> str:
    return f"shared/context/{name}/codebook_logstdevs"


def _indices_path(name: str) -> str:
    return f"global/context/{name}_indices"


# ── Value sets ────────────────────────────────────────────────────────────────


def derive_context_values(
    networks: list, override: dict[str, list[str]] | None = None
) -> dict[str, list[str]]:
    """Effective per-context-var value list for a set of networks.

    Default value first (index 0, the unknown-fallback row), then the remaining
    distinct values present in the networks, sorted for determinism. `override`
    (e.g. a loaded model's stored values) wins per context var so codebook rows stay
    aligned with how the model was trained. The set of cell types a model knows is
    thus derived from its data, not a global constant.
    """
    out: dict[str, list[str]] = {}
    for ce in CONTEXT_EMBEDDINGS:
        if override is not None and ce.name in override:
            out[ce.name] = list(override[ce.name])
            continue
        present = {net.metadata.get(ce.name, ce.default_value) for net in networks}
        out[ce.name] = [ce.default_value, *sorted(present - {ce.default_value})]
    return out


def infer_context_values_from_params(params: ParameterTree) -> dict[str, list[str]]:
    """Best-effort value list for a legacy model that has a codebook but stored no
    values: take the seed `available_values` (default first), truncated to the
    codebook row count. For the original single-cell-type models this yields the
    default alone, which is correct."""
    out: dict[str, list[str]] = {}
    for ce in CONTEXT_EMBEDDINGS:
        means_path = _codebook_means_path(ce.name)
        if means_path not in params:
            continue
        n = int(params[means_path].shape[0])
        seed = [ce.default_value, *[v for v in ce.available_values if v != ce.default_value]]
        out[ce.name] = seed[:n]
    return out


# ── Initialization ────────────────────────────────────────────────────────────


def init_context_params(
    params: ParameterTree,
    networks: list,
    key: ArrayLike,
    allow_create: bool = True,
    context_values: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """Initialize context codebooks and (always) the per-network index mapping.

    The codebook is created once, sized to the effective value list (derived from
    `networks`, or taken from `context_values` for a loaded model). The per-network
    index array is rebuilt every call so evaluation over new networks maps correctly;
    a value absent from this model's codebook falls back to the default row.
    `allow_create=False` skips a context var whose codebook is missing (back-compat
    for models trained before context embeddings existed). Returns the effective
    value list per context var (for persisting on the model).
    """
    used: dict[str, list[str]] = {}
    if not CONTEXT_EMBEDDINGS:
        return used

    values = derive_context_values(networks, context_values)

    for ce in CONTEXT_EMBEDDINGS:
        means_path = _codebook_means_path(ce.name)
        logstdevs_path = _codebook_logstdevs_path(ce.name)
        indices_path = _indices_path(ce.name)

        ce_values = values[ce.name]
        used[ce.name] = ce_values

        if means_path not in params:
            if not allow_create:
                logger.info(
                    f"Context embedding '{ce.name}' not in loaded params; skipping (back-compat)."
                )
                continue
            key, k1 = jax.random.split(key)
            params[means_path] = jax.random.normal(k1, (len(ce_values), ce.embedding_dim)) * 0.1
            params[logstdevs_path] = jnp.full((len(ce_values), ce.embedding_dim), -3.0)

        # Per-network index array, rebuilt every call (unknown value -> default row).
        value_to_idx = {v: i for i, v in enumerate(ce_values)}
        default_idx = value_to_idx[ce.default_value]
        indices = []
        for net in networks:
            val = net.metadata.get(ce.name, ce.default_value)
            if val not in value_to_idx:
                logger.debug(
                    f"Context '{ce.name}': '{val}' not in this model's codebook "
                    f"{ce_values}; using default '{ce.default_value}'."
                )
            indices.append(value_to_idx.get(val, default_idx))

        params.at(
            indices_path,
            jnp.array(indices, dtype=jnp.int32),
            tags=[NON_GRAD_TAG],
            overwrite=True,
        )

        logger.info(
            f"Context embedding '{ce.name}': {len(ce_values)} values {ce_values}, "
            f"dim={ce.embedding_dim}, {len(networks)} networks mapped"
        )
    return used


def grow_context_codebook(
    params: ParameterTree,
    source_shared: ParameterTree,
    source_values: dict[str, list[str]],
    target_values: dict[str, list[str]],
    key: ArrayLike,
) -> None:
    """Rebuild each context codebook in `params` to `target_values`, copying trained
    rows from a pretrained model by VALUE (so warm-start grows the codebook instead of
    overwriting it). Values shared with the source keep their learned mean/logstdev;
    new values get a fresh init. Robust to value-ordering differences. Mutates
    `params` and re-tags the codebook as shared."""
    for ce in CONTEXT_EMBEDDINGS:
        means_path = _codebook_means_path(ce.name)
        logstdevs_path = _codebook_logstdevs_path(ce.name)
        if means_path not in source_shared or ce.name not in source_values:
            continue
        src_idx = {v: i for i, v in enumerate(source_values[ce.name])}
        tgt = target_values[ce.name]
        key, k1 = jax.random.split(key)
        means = jax.random.normal(k1, (len(tgt), ce.embedding_dim)) * 0.1
        logstdevs = jnp.full((len(tgt), ce.embedding_dim), -3.0)
        src_means = source_shared[means_path]
        src_logstdevs = source_shared[logstdevs_path]
        for j, v in enumerate(tgt):
            if v in src_idx:
                means = means.at[j].set(src_means[src_idx[v]])
                logstdevs = logstdevs.at[j].set(src_logstdevs[src_idx[v]])
        params.at(means_path, means, tags=["shared"], overwrite=True)
        params.at(logstdevs_path, logstdevs, tags=["shared"], overwrite=True)


# ── Freezing ──────────────────────────────────────────────────────────────────


CONTEXT_PATH_PREFIX = "shared/context/"


def freeze_non_context_shared(params: ParameterTree) -> int:
    """Tag every trainable shared param OUTSIDE `shared/context/` as non_grad, so a
    warm-started run trains ONLY the cell_type embedding (the rung-1/2 'embed-only'
    mode). The existing static/dynamic split already excludes `non_grad`, so no other
    plumbing is needed. Returns the number of params frozen."""
    n = 0
    for path, _ in list(params.data.iter_leaves()):
        sp = str(path)
        if sp.startswith("shared/") and not sp.startswith(CONTEXT_PATH_PREFIX):
            params.tag(path, ["non_grad"])
            n += 1
    return n


# ── Resolution (JIT-compatible) ──────────────────────────────────────────────


def resolve_context_vector(
    params: ParameterTree,
    network_id: ArrayLike,
    key: ArrayLike,
) -> jnp.ndarray | None:
    """Resolve all context embeddings for a network into a single flat vector.

    Returns None if context is not initialized (backward compat with old models).
    """
    if not CONTEXT_EMBEDDINGS:
        return None

    # Sentinel check: if the first embedding's codebook isn't in params, context
    # was never initialized (old model). Return None -> dense_mlp no-ops.
    sentinel = _codebook_means_path(CONTEXT_EMBEDDINGS[0].name)
    if sentinel not in params:
        return None

    pieces = []
    for ce in CONTEXT_EMBEDDINGS:
        key, k_i = jax.random.split(key)
        idx = params[_indices_path(ce.name)][network_id]
        mean = params[_codebook_means_path(ce.name)][idx]
        logstdev = params[_codebook_logstdevs_path(ce.name)][idx]
        emb, _ = variational_codebook_lookup(mean, logstdev, k_i)
        pieces.append(emb)

    return jnp.concatenate(pieces)
