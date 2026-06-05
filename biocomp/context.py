# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Context embedding system for per-network conditioning.

Provides a "context bus" -- a flat embedding vector resolved from categorical
context variables (e.g. cell_type) that conditions every MLP via dense_mlp's
`context` parameter.

Adding a new context variable = add one entry to CONTEXT_EMBEDDINGS.
No node code changes needed.
"""

from collections.abc import Callable

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


# ── KL prior inputs ───────────────────────────────────────────────────────────


def context_codebook_kl_inputs(params: ParameterTree):
    """Flattened (means, logstdevs, counts, total_count) across every context codebook,
    for a KL(q || N(0,1)) prior on the cell-type space -- the SAME treatment the part
    embeddings get. Each row is weighted by the number of networks using that value, so a
    cell type keeps the same KL-vs-reconstruction balance no matter how few circuits use it
    (the few-shot row is regularized, not swamped). Returns None for a context-free model.

    Expects single-replicate params (the per-replicate loss), so the index array is 1-D.
    """
    means, logstdevs, counts = [], [], []
    total = jnp.asarray(0.0)
    for ce in CONTEXT_EMBEDDINGS:
        means_path = _codebook_means_path(ce.name)
        if means_path not in params:
            continue
        m = params[means_path]  # (rows, dim)
        ls = params[_codebook_logstdevs_path(ce.name)]
        per_row = jnp.bincount(params[_indices_path(ce.name)], length=m.shape[0]).astype(m.dtype)
        total = total + per_row.sum()
        means.append(m.reshape(-1))
        logstdevs.append(ls.reshape(-1))
        counts.append(jnp.repeat(per_row, m.shape[1]))  # one count per (row, dim) entry
    if not means:
        return None
    return jnp.concatenate(means), jnp.concatenate(logstdevs), jnp.concatenate(counts), total


# ── Value sets ────────────────────────────────────────────────────────────────


def _ordered_values(ce: ContextEmbedding, values: set[str]) -> list[str]:
    # row 0 = default (the unknown-fallback row), then the rest sorted for determinism
    return [ce.default_value, *sorted(values - {ce.default_value})]


def derive_context_values(
    networks: list, override: dict[str, list[str]] | None = None
) -> dict[str, list[str]]:
    """Per-context-var value list a set of networks needs. `override` (e.g. a loaded
    model's stored values) wins per var so codebook rows stay aligned with how the model
    was trained. So a model's known cell types come from its data, not a global constant."""
    out: dict[str, list[str]] = {}
    for ce in CONTEXT_EMBEDDINGS:
        if override is not None and ce.name in override:
            out[ce.name] = list(override[ce.name])
            continue
        out[ce.name] = _ordered_values(
            ce, {net.metadata.get(ce.name, ce.default_value) for net in networks}
        )
    return out


def union_context_values(*value_dicts: dict[str, list[str]]) -> dict[str, list[str]]:
    """Warm-start GROW semantics: per-var union of all inputs, so a fine-tune never drops
    a cell type the pretrained model knew (e.g. embed-only on line B keeps line A's row)."""
    return {
        ce.name: _ordered_values(ce, set().union(*(d.get(ce.name, []) for d in value_dicts)))
        for ce in CONTEXT_EMBEDDINGS
    }


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


def make_codebook_freeze_hook(params: ParameterTree) -> Callable | None:
    """Post-update hook for embed-only: pin every codebook row NOT referenced by the
    training data to its warm-start value, so only the adapted line's row moves (the
    codebook is a single leaf, so per-leaf tagging can't freeze it row-wise; decoupled
    weight decay / Adam momentum can't drift a pinned row). Active rows are data-driven:
    the indices the forward path uses. None when context is uninitialized."""
    if not CONTEXT_EMBEDDINGS:
        return None

    def per_rep(arr, base_ndim):  # codebook (2d) and indices (1d) are replicated on axis 0
        return arr[0] if arr.ndim > base_ndim else arr

    pinned: dict[str, tuple[jnp.ndarray, jnp.ndarray]] = {}
    for ce in CONTEXT_EMBEDDINGS:
        ipath = _indices_path(ce.name)
        if ipath not in params:
            continue
        active = jnp.unique(per_rep(params[ipath], 1))
        for path in (_codebook_means_path(ce.name), _codebook_logstdevs_path(ce.name)):
            buf = per_rep(params[path], 2)
            trainable = jnp.zeros(buf.shape[0], bool).at[active].set(True)[:, None]
            pinned[path] = (buf, trainable)
    if not pinned:
        return None

    def hook(p: ParameterTree, *a, **kw) -> ParameterTree:
        for path, (buf, trainable) in pinned.items():
            p = p.update_leaves_by_path([path], lambda v, b=buf, t=trainable: jnp.where(t, v, b))
        return p

    return hook


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


def disable_context_variational(params: ParameterTree, floor: float = -100.0) -> None:
    """In-place: pin every context codebook's logstdevs to `floor` (σ ≈ 0) so inference uses
    the codebook MEAN with no sampling -- the eval-time counterpart of the part embeddings'
    `disable_variational`. `resolve_context_vector` always samples (no flag), so callers that
    turn off the part noise at inference must call this too, or a regularized cell-type
    embedding (σ ≈ 1) injects real noise into every prediction."""
    for ce in CONTEXT_EMBEDDINGS:
        path = _codebook_logstdevs_path(ce.name)
        if path in params:
            # preserve the leaf's tags (e.g. "shared") -- a plain reassignment drops them and
            # changes the pytree metadata, which trips the jitted apply's input-structure check.
            params.at(path, jnp.ones_like(params[path]) * floor, tags=params.get_tags(path), overwrite=True)
