"""Context embedding system for per-network conditioning.

Provides a "context bus" — a flat embedding vector resolved from categorical
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


# ── Initialization ────────────────────────────────────────────────────────────


def init_context_params(
    params: ParameterTree,
    networks: list,
    key: ArrayLike,
) -> None:
    """Initialize context embedding codebooks and per-network index mapping.

    Called once during ComputeStack.init(), BEFORE the layer prepare loop.
    Idempotent: skips if paths already exist (backward compat with loaded models).
    """
    if not CONTEXT_EMBEDDINGS:
        return

    for ce in CONTEXT_EMBEDDINGS:
        means_path = _codebook_means_path(ce.name)
        logstdevs_path = _codebook_logstdevs_path(ce.name)
        indices_path = _indices_path(ce.name)

        # Skip if already initialized (e.g. loaded from a saved model)
        if means_path in params:
            logger.debug(f"Context embedding '{ce.name}' already initialized, skipping")
            continue

        key, k1 = jax.random.split(key)
        n_values = len(ce.available_values)

        # Codebook: small random init for means, conservative logstdevs
        params[means_path] = jax.random.normal(k1, (n_values, ce.embedding_dim)) * 0.1
        params[logstdevs_path] = jnp.full((n_values, ce.embedding_dim), -3.0)

        # Build value→index mapping
        value_to_idx = {v: i for i, v in enumerate(ce.available_values)}

        # Per-network index array
        indices = []
        for net in networks:
            val = net.metadata.get(ce.name, ce.default_value)
            if val not in value_to_idx:
                logger.warning(
                    f"Context '{ce.name}': unknown value '{val}' for network '{net.name}', "
                    f"using default '{ce.default_value}'"
                )
                val = ce.default_value
            indices.append(value_to_idx[val])

        params.at(indices_path, jnp.array(indices, dtype=jnp.int32), tags=[NON_GRAD_TAG])

        logger.info(
            f"Context embedding '{ce.name}': {n_values} values, dim={ce.embedding_dim}, "
            f"{len(networks)} networks mapped"
        )


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
    # was never initialized (old model). Return None → dense_mlp no-ops.
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
