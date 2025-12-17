"""Hard Concrete distribution for learnable TU masking (Louizos et al. 2018).

TU index convention: -1 = always enabled, >= 0 = index into tu_log_alpha array.
"""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from typing import Optional
import numpy as np

DEFAULT_GAMMA = -0.1
DEFAULT_ZETA = 1.1
DEFAULT_TEMPERATURE = 0.5
MIN_TEMPERATURE = 0.1
LOG_ALPHA_MIN = -3.0
LOG_ALPHA_MAX = 4.0
TU_ALWAYS_ENABLED = -1
TU_LOG_ALPHA_PATH = "design/tu_log_alpha"


def clamp_log_alpha(log_alpha: ArrayLike) -> jnp.ndarray:
    """Soft clamp log_alpha to prevent gradient death at boundaries."""
    center = (LOG_ALPHA_MAX + LOG_ALPHA_MIN) / 2
    scale = (LOG_ALPHA_MAX - LOG_ALPHA_MIN) / 2
    return center + scale * jnp.tanh((log_alpha - center) / scale)


def sample_hard_concrete(
    log_alpha: ArrayLike,
    key: ArrayLike,
    temperature: float = DEFAULT_TEMPERATURE,
    gamma: float = DEFAULT_GAMMA,
    zeta: float = DEFAULT_ZETA,
) -> jnp.ndarray:
    """Sample from Hard Concrete distribution. Returns values in [0, 1]."""
    u = jax.random.uniform(key, shape=jnp.asarray(log_alpha).shape, minval=1e-8, maxval=1 - 1e-8)
    temp_safe = jnp.maximum(temperature, MIN_TEMPERATURE)
    la = clamp_log_alpha(log_alpha)
    s = jax.nn.sigmoid((jnp.log(u) - jnp.log(1 - u) + la) / temp_safe)
    s_bar = s * (zeta - gamma) + gamma
    return jnp.clip(s_bar, 0.0, 1.0)


def sample_hard_concrete_deterministic(
    log_alpha: ArrayLike,
    temperature: float = DEFAULT_TEMPERATURE,
    gamma: float = DEFAULT_GAMMA,
    zeta: float = DEFAULT_ZETA,
) -> jnp.ndarray:
    """Deterministic Hard Concrete using median (u=0.5)."""
    temp_safe = jnp.maximum(temperature, MIN_TEMPERATURE)
    la = clamp_log_alpha(log_alpha)
    s = jax.nn.sigmoid(la / temp_safe)
    s_bar = s * (zeta - gamma) + gamma
    return jnp.clip(s_bar, 0.0, 1.0)


def get_final_mask(log_alpha: ArrayLike, threshold: float = 0.5) -> jnp.ndarray:
    """Get binary mask at commit time (1 = keep, 0 = remove).

    CRITICAL: This function determines which TUs are kept after optimization.
    The mask is binary: 1.0 (keep) or 0.0 (remove).
    """
    log_alpha = jnp.asarray(log_alpha)
    sigmoid_values = jax.nn.sigmoid(log_alpha)
    mask = (sigmoid_values >= threshold).astype(jnp.float32)
    # verify binary output
    assert jnp.all((mask == 0.0) | (mask == 1.0)), (
        f"get_final_mask BUG: mask should be binary but got values: {jnp.unique(mask)}"
    )
    return mask


def l0_penalty(
    log_alpha: ArrayLike,
    temperature: float = DEFAULT_TEMPERATURE,
    gamma: float = DEFAULT_GAMMA,
    zeta: float = DEFAULT_ZETA,
) -> jnp.ndarray:
    """Expected L0 penalty P(z > 0), differentiable w.r.t. log_alpha."""
    temp_safe = jnp.maximum(temperature, MIN_TEMPERATURE)
    la = clamp_log_alpha(log_alpha)
    return jax.nn.sigmoid(la - temp_safe * jnp.log(-gamma / zeta))


def l0_loss(
    log_alpha: ArrayLike,
    temperature: float = DEFAULT_TEMPERATURE,
    gamma: float = DEFAULT_GAMMA,
    zeta: float = DEFAULT_ZETA,
) -> jnp.ndarray:
    """Total L0 loss (sum of expected L0 penalties)."""
    return jnp.sum(l0_penalty(log_alpha, temperature, gamma, zeta))


def soft_clip(x: ArrayLike, low: float = 0.0, high: float = 1.0) -> jnp.ndarray:
    """Soft clip using tanh - maintains gradients everywhere."""
    mid = (low + high) / 2
    scale = (high - low) / 2
    normalized = (x - mid) / scale
    soft = jnp.tanh(normalized * 2) / 2 + 0.5
    return low + (high - low) * soft


def hard_concrete_from_uniform(
    u: ArrayLike,
    log_alpha: ArrayLike,
    temperature: float = DEFAULT_TEMPERATURE,
    gamma: float = DEFAULT_GAMMA,
    zeta: float = DEFAULT_ZETA,
) -> jnp.ndarray:
    """Transform uniform sample to Hard Concrete: u ~ Uniform(0,1) -> z ~ HardConcrete."""
    u = jnp.clip(u, 1e-8, 1 - 1e-8)
    temp_safe = jnp.maximum(temperature, MIN_TEMPERATURE)
    la = clamp_log_alpha(log_alpha)
    logit = (jnp.log(u) - jnp.log(1 - u) + la) / temp_safe
    s = jax.nn.sigmoid(logit)
    s_bar = s * (zeta - gamma) + gamma
    return soft_clip(s_bar, 0.0, 1.0)


def is_enabled(z: ArrayLike, threshold: float = 0.5) -> jnp.ndarray:
    return z >= threshold


def compute_input_mask(
    tu_idx: ArrayLike,
    tu_uniform_samples: ArrayLike,
    tu_log_alpha: ArrayLike,
    temperature: float = DEFAULT_TEMPERATURE,
) -> jnp.ndarray:
    """Compute mask for single input using Straight-Through Estimator."""
    def when_has_tu():
        u = tu_uniform_samples[tu_idx]
        la = tu_log_alpha[tu_idx]
        z = hard_concrete_from_uniform(u, la, temperature)
        hard_mask = jnp.where(is_enabled(z), 1.0, 0.0)
        return z + jax.lax.stop_gradient(hard_mask - z)

    return jax.lax.cond(tu_idx >= 0, when_has_tu, lambda: 1.0)


def compute_input_mask_multi(
    tu_indices: ArrayLike,
    tu_uniform_samples: ArrayLike,
    tu_log_alpha: ArrayLike,
    temperature: float = DEFAULT_TEMPERATURE,
) -> jnp.ndarray:
    """Compute mask for input with MULTIPLE TU indices. Enabled if ANY TU enabled."""
    def single_mask(tu_idx):
        return jax.lax.cond(
            tu_idx >= 0,
            lambda: compute_input_mask(tu_idx, tu_uniform_samples, tu_log_alpha, temperature),
            lambda: 0.0,
        )

    masks = jax.vmap(single_mask)(tu_indices)
    all_padding = jnp.all(tu_indices < 0)
    max_mask = jnp.max(masks)
    hard_any = jnp.where(max_mask > 0.5, 1.0, 0.0)
    ste_result = max_mask + jax.lax.stop_gradient(hard_any - max_mask)
    return jnp.where(all_padding, 1.0, ste_result)


def compute_input_masks(
    tu_indices: ArrayLike,
    tu_uniform_samples: Optional[ArrayLike],
    tu_log_alpha: Optional[ArrayLike],
    temperature: float = DEFAULT_TEMPERATURE,
) -> jnp.ndarray:
    """Compute masks for all inputs. Handles 1D (single TU) or 2D (multi-TU) indices.

    CRITICAL: This is the core TU masking function used during forward pass.
    tu_uniform_samples and tu_log_alpha must have compatible shapes.
    """
    tu_indices = jnp.asarray(tu_indices)
    n_inputs = tu_indices.shape[0]

    if tu_uniform_samples is None or tu_log_alpha is None:
        return jnp.ones(n_inputs)

    tu_uniform_samples = jnp.asarray(tu_uniform_samples)
    tu_log_alpha = jnp.asarray(tu_log_alpha)

    # shape validation - these should be 1D arrays indexed by TU
    assert tu_uniform_samples.ndim == 1, (
        f"tu_uniform_samples must be 1D (n_tus,), got {tu_uniform_samples.ndim}D. "
        f"Shape: {tu_uniform_samples.shape}. Did you forget to slice for network_id?"
    )
    assert tu_log_alpha.ndim == 1, (
        f"tu_log_alpha must be 1D (n_tus,), got {tu_log_alpha.ndim}D. "
        f"Shape: {tu_log_alpha.shape}. Did you forget to slice for network_id?"
    )
    assert tu_uniform_samples.shape == tu_log_alpha.shape, (
        f"Shape mismatch: tu_uniform_samples {tu_uniform_samples.shape} vs "
        f"tu_log_alpha {tu_log_alpha.shape}"
    )

    if tu_indices.ndim == 1:
        return jax.vmap(
            lambda idx: compute_input_mask(idx, tu_uniform_samples, tu_log_alpha, temperature)
        )(tu_indices)
    else:
        return jax.vmap(
            lambda indices: compute_input_mask_multi(indices, tu_uniform_samples, tu_log_alpha, temperature)
        )(tu_indices)


def get_default_tu_uniform_samples(n_tus: int) -> jnp.ndarray:
    return jnp.full((n_tus,), 0.5)


def get_default_tu_log_alpha(n_tus: int) -> jnp.ndarray:
    return jnp.zeros(n_tus)


def extract_tu_ids_from_network(network) -> list[str]:
    """Extract all unique TU IDs from a network's edges."""
    tu_ids = set()
    for edge in network.compute_graph.edges.values():
        if edge.extra:
            tu_ids.update(edge.extra.get("tu_id", []))
    return sorted(tu_ids)


def build_tu_id_mapping(networks: list) -> tuple[list[str], dict[str, int]]:
    """Build TU ID mapping from multiple networks."""
    all_tu_ids = set()
    for net in networks:
        all_tu_ids.update(extract_tu_ids_from_network(net))
    sorted_tu_ids = sorted(all_tu_ids)
    return sorted_tu_ids, {tu_id: i for i, tu_id in enumerate(sorted_tu_ids)}


def init_tu_log_alpha(
    n_tus: int,
    key: ArrayLike,
    init_mean: float = 2.0,
    init_std: float = 0.5,
) -> jnp.ndarray:
    """Initialize TU log_alpha parameters (positive = mostly enabled)."""
    return init_mean + init_std * jax.random.normal(key, shape=(n_tus,))


def get_tu_mask_for_node(
    tu_ids: list[str],
    tu_id_to_idx: dict[str, int],
    log_alpha_all: ArrayLike,
    key: ArrayLike,
    temperature: float = DEFAULT_TEMPERATURE,
) -> jnp.ndarray:
    """Get mask for a node: 0 if ANY TU is masked, 1 otherwise."""
    if not tu_ids:
        return jnp.array(1.0)

    mask = jnp.array(1.0)
    for tu_id in tu_ids:
        if tu_id in tu_id_to_idx:
            idx = tu_id_to_idx[tu_id]
            log_alpha = log_alpha_all[idx]
            tu_key = jax.random.fold_in(key, hash(tu_id) % (2**31))
            z = sample_hard_concrete(log_alpha, tu_key, temperature)
            tu_mask = (z >= 0.5).astype(jnp.float32)
            mask = mask * tu_mask

    return mask


def build_input_tu_indices(
    stack,
    nodelist,
    tu_id_to_idx: dict[str, int],
) -> jnp.ndarray:
    """Build TU index mapping for inputs. Shape (n_nodes, max_inputs, max_tus). -1 = padding."""
    if not nodelist:
        return jnp.array([[[]]], dtype=jnp.int32)

    max_inputs = max(len(node.get_incoming_edges(stack)) for node in nodelist)
    if max_inputs == 0:
        return jnp.full((len(nodelist), 1, 1), TU_ALWAYS_ENABLED, dtype=jnp.int32)

    max_tus = 1
    for node in nodelist:
        for edge in node.get_incoming_edges(stack):
            tu_ids = edge.extra.get("tu_id", []) if edge.extra else []
            valid_tu_ids = [tid for tid in tu_ids if tid in tu_id_to_idx]
            max_tus = max(max_tus, len(valid_tu_ids))

    tu_indices = np.full((len(nodelist), max_inputs, max_tus), TU_ALWAYS_ENABLED, dtype=np.int32)

    for i, node in enumerate(nodelist):
        edges = node.get_incoming_edges(stack)
        edges_sorted = sorted(edges, key=lambda e: e.to_input_slot)
        for j, edge in enumerate(edges_sorted):
            tu_ids = edge.extra.get("tu_id", []) if edge.extra else []
            valid_tu_ids = [tid for tid in tu_ids if tid in tu_id_to_idx]
            for k, tid in enumerate(valid_tu_ids):
                if k < max_tus:
                    tu_indices[i, j, k] = tu_id_to_idx[tid]

    return jnp.array(tu_indices, dtype=jnp.int32)


def build_output_tu_indices(
    stack,
    nodelist,
    tu_id_to_idx: dict[str, int],
    n_outputs: int,
) -> jnp.ndarray:
    """Build TU index mapping for outputs. Shape (n_nodes, n_outputs). -1 = unmapped."""
    if not nodelist:
        return jnp.array([[]], dtype=jnp.int32)

    tu_indices = np.full((len(nodelist), n_outputs), TU_ALWAYS_ENABLED, dtype=np.int32)

    for i, node in enumerate(nodelist):
        edges = node.get_outgoing_edges(stack)
        for edge in edges:
            slot = edge.from_output_slot
            if slot < n_outputs:
                tu_ids = edge.extra.get("tu_id", []) if edge.extra else []
                if len(tu_ids) == 1 and tu_ids[0] in tu_id_to_idx:
                    tu_indices[i, slot] = tu_id_to_idx[tu_ids[0]]

    return jnp.array(tu_indices, dtype=jnp.int32)


def extract_tu_ids_for_inverse_nodes(networks: list) -> set[str]:
    """Extract TU IDs that flow into inverse nodes (should never be disabled)."""
    inverse_tu_ids = set()
    for net in networks:
        graph = net.compute_graph
        for edge in graph.edges.values():
            target_node = graph.nodes.get(edge.target_id)
            if target_node and target_node.node_type.startswith("inv_"):
                tu_ids = edge.extra.get("tu_id", []) if edge.extra else []
                inverse_tu_ids.update(tu_ids)
    return inverse_tu_ids


def build_tu_id_mapping_excluding_inverse(
    networks: list,
) -> tuple[list[str], dict[str, int], set[str]]:
    """Build TU ID mapping. Returns (sorted_tu_ids, tu_id_to_idx, inverse_tu_ids)."""
    all_tu_ids = set()
    for net in networks:
        all_tu_ids.update(extract_tu_ids_from_network(net))

    inverse_tu_ids = extract_tu_ids_for_inverse_nodes(networks)
    sorted_tu_ids = sorted(all_tu_ids)
    tu_id_to_idx = {tu_id: i for i, tu_id in enumerate(sorted_tu_ids)}

    return sorted_tu_ids, tu_id_to_idx, inverse_tu_ids
