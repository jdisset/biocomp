"""TU masking for learnable enable/disable decisions.

Binary masking with Straight-Through Estimator (STE).
- Forward: binary decisions (sigmoid(log_alpha) >= 0.5)
- Backward: gradients flow through sigmoid(log_alpha)

TU index convention: -1 = always enabled, >= 0 = index into tu_log_alpha array.
"""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np

LOG_ALPHA_MIN = -3.0
LOG_ALPHA_MAX = 4.0
TU_ALWAYS_ENABLED = -1
TU_LOG_ALPHA_PATH = "design/tu_log_alpha"
TU_BINARY_MASK_PATH = "design/tu_binary_mask"
PROTECTED_TU_MASK_PATH = "design/protected_tu_mask"

LATENT_TU_Z_PATH = "design/latent_tu_z"
LATENT_TU_W1_PATH = "design/latent_tu_W1"
LATENT_TU_B1_PATH = "design/latent_tu_b1"
LATENT_TU_W2_PATH = "design/latent_tu_W2"
LATENT_TU_B2_PATH = "design/latent_tu_b2"


def clamp_log_alpha(log_alpha: ArrayLike) -> jnp.ndarray:
    """Soft clamp log_alpha to prevent gradient death at boundaries."""
    center = (LOG_ALPHA_MAX + LOG_ALPHA_MIN) / 2
    scale = (LOG_ALPHA_MAX - LOG_ALPHA_MIN) / 2
    return center + scale * jnp.tanh((log_alpha - center) / scale)


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


def binary_mask_with_ste(log_alpha: ArrayLike, threshold: float = 0.5) -> jnp.ndarray:
    """Binary mask with Straight-Through Estimator for gradient flow.

    Forward: returns binary 0/1 based on sigmoid(log_alpha) >= threshold
    Backward: gradients flow through sigmoid(log_alpha)

    This is simpler than hard concrete and provides consistent behavior
    across training, evaluation, and inference.
    """
    prob = jax.nn.sigmoid(log_alpha)
    binary = (prob >= threshold).astype(jnp.float32)
    return prob + jax.lax.stop_gradient(binary - prob)


def compute_binary_mask_single(tu_idx: ArrayLike, tu_log_alpha: ArrayLike) -> jnp.ndarray:
    """Compute binary mask for single TU index with STE.

    Args:
        tu_idx: Single TU index (-1 = always enabled, >= 0 = index into tu_log_alpha)
        tu_log_alpha: Log-alpha params, shape (n_tus,)

    Returns:
        Mask value (binary with STE gradient) or 1.0 if tu_idx < 0
    """
    safe_idx = jnp.maximum(tu_idx, 0)
    la = tu_log_alpha[safe_idx]
    mask = binary_mask_with_ste(la)
    return jnp.where(tu_idx >= 0, mask, 1.0)


MULTI_TU_SOFTMAX_TEMPERATURE = 0.1


def compute_binary_mask_multi_probabilistic_or(
    tu_indices: ArrayLike,
    tu_log_alpha: ArrayLike,
) -> jnp.ndarray:
    """Compute binary mask using Probabilistic OR (Noisy OR).

    P(any TU on) = 1 - ∏(1 - P(TU_i))

    Unlike softmax OR which uses weighted sum, probabilistic OR naturally gives
    gradient to ALL contributing TUs proportional to how much they affect the
    product. This fixes the "rich-get-richer" problem where the argmax TU
    dominates gradient flow.

    Args:
        tu_indices: TU indices array, shape (max_tus,). Padding uses -1.
        tu_log_alpha: Log-alpha params, shape (n_tus,)

    Returns:
        Single mask value: P(any on) with STE for binary forward pass.
    """
    safe_indices = jnp.maximum(tu_indices, 0)
    probs = jax.nn.sigmoid(tu_log_alpha[safe_indices])

    valid_mask = tu_indices >= 0
    masked_probs = jnp.where(valid_mask, probs, 0.0)  # invalid = prob 0

    all_padding = jnp.all(~valid_mask)

    # probabilistic OR: P(any on) = 1 - P(all off) = 1 - ∏(1 - p_i)
    prob_all_off = jnp.prod(1.0 - masked_probs)
    prob_any_on = 1.0 - prob_all_off

    # STE: forward = hard threshold, backward = soft probability
    binary_edge = (prob_any_on >= 0.5).astype(jnp.float32)
    ste_result = prob_any_on + jax.lax.stop_gradient(binary_edge - prob_any_on)

    return jnp.where(all_padding, 1.0, ste_result)


def entropy_bonus(log_alpha: ArrayLike, epsilon: float = 1e-6) -> jnp.ndarray:
    """Compute entropy bonus for TU mask probabilities.

    H = -mean[p * log(p) + (1-p) * log(1-p)]

    Higher entropy = more uncertain = exploring.
    Lower entropy = committed = exploitation.

    Use as loss term: loss += -lambda_entropy * entropy_bonus(log_alpha)
    (Negative because we WANT to maximize entropy during exploration phase)

    Args:
        log_alpha: Log-alpha parameters, any shape
        epsilon: Numerical stability clipping

    Returns:
        Scalar entropy value, normalized to [0, 1] where 1 = all at 0.5
    """
    probs = jax.nn.sigmoid(log_alpha)
    probs = jnp.clip(probs, epsilon, 1 - epsilon)
    entropy_per_tu = -(probs * jnp.log(probs) + (1 - probs) * jnp.log(1 - probs))
    return jnp.mean(entropy_per_tu) / jnp.log(2.0)  # normalize to [0, 1]


def compute_binary_mask_multi(
    tu_indices: ArrayLike,
    tu_log_alpha: ArrayLike,
    softmax_temperature: float = MULTI_TU_SOFTMAX_TEMPERATURE,
) -> jnp.ndarray:
    """Compute binary mask for multiple TU indices. Enabled if ANY TU enabled.

    All TUs receive gradients proportional to their contribution (via softmax),
    while forward pass uses hard OR (any TU enabled → edge enabled).

    Args:
        tu_indices: TU indices array, shape (max_tus,). Padding uses -1.
        tu_log_alpha: Log-alpha params, shape (n_tus,)
        softmax_temperature: Controls gradient sharpness. Lower = more peaked.

    Returns:
        Single mask value: 1.0 if any TU enabled (or all padding), 0.0 otherwise
    """
    safe_indices = jnp.maximum(tu_indices, 0)
    la_vals = tu_log_alpha[safe_indices]
    masks = binary_mask_with_ste(la_vals)

    valid_mask = tu_indices >= 0
    masked_vals = jnp.where(valid_mask, masks, 0.0)

    all_padding = jnp.all(~valid_mask)

    # Softmax over log_alphas for gradient distribution (all TUs receive gradients)
    weights = jax.nn.softmax(jnp.where(valid_mask, la_vals / softmax_temperature, -1e9))
    soft_or = jnp.sum(weights * masked_vals)

    # Hard OR for forward pass (any enabled → 1.0)
    hard_any = jnp.where(jnp.max(masked_vals) > 0.5, 1.0, 0.0)

    # STE: forward uses hard_any, backward uses soft_or
    ste_result = soft_or + jax.lax.stop_gradient(hard_any - soft_or)
    return jnp.where(all_padding, 1.0, ste_result)


def compute_binary_masks(
    tu_indices: ArrayLike,
    tu_log_alpha: ArrayLike,
    *,
    is_multi_tu: bool,
    softmax_temperature: float = MULTI_TU_SOFTMAX_TEMPERATURE,
    use_probabilistic_or: bool = False,
) -> jnp.ndarray:
    """Compute binary masks for all inputs using STE.

    This is the default TU masking function - simple, consistent, differentiable.

    Args:
        tu_indices: TU indices array.
            - For single-TU (is_multi_tu=False): shape (n_inputs,)
            - For multi-TU (is_multi_tu=True): shape (n_inputs, max_tus)
        tu_log_alpha: Log-alpha params, shape (n_tus,)
        is_multi_tu: True for input_tu_indices (OR reduction), False for output_tu_indices
        softmax_temperature: For multi-TU, controls gradient sharpness. Lower = more peaked.
        use_probabilistic_or: If True, use P(any)=1-∏(1-p) instead of softmax OR for multi-TU.
            Probabilistic OR gives better gradient flow to non-dominant TUs.

    Returns:
        Binary masks with STE gradient, shape (n_inputs,)
    """
    tu_indices = jnp.asarray(tu_indices)
    tu_log_alpha = jnp.asarray(tu_log_alpha)

    assert tu_log_alpha.ndim == 1, f"tu_log_alpha must be 1D (n_tus,), got {tu_log_alpha.ndim}D"

    if is_multi_tu:
        assert tu_indices.ndim == 2, (
            f"is_multi_tu=True but tu_indices.ndim={tu_indices.ndim}. "
            f"Expected 2D (n_inputs, max_tus), got shape {tu_indices.shape}."
        )
        if use_probabilistic_or:
            return jax.vmap(
                lambda indices: compute_binary_mask_multi_probabilistic_or(indices, tu_log_alpha)
            )(tu_indices)
        else:
            return jax.vmap(
                lambda indices: compute_binary_mask_multi(
                    indices, tu_log_alpha, softmax_temperature
                )
            )(tu_indices)
    else:
        assert tu_indices.ndim == 1, (
            f"is_multi_tu=False but tu_indices.ndim={tu_indices.ndim}. "
            f"Expected 1D (n_inputs,), got shape {tu_indices.shape}."
        )
        return jax.vmap(lambda idx: compute_binary_mask_single(idx, tu_log_alpha))(tu_indices)


L0_PENALTY_FLOOR_PROB = 0.2  # Stop L0 penalty below this probability


def l0_penalty(
    log_alpha: ArrayLike,
    floor_prob: float = L0_PENALTY_FLOOR_PROB,
    leak_coef: float = 0.0,
) -> jnp.ndarray:
    """L0 penalty with floor to prevent extinction and allow TU rebirth.

    Returns penalty proportional to P(TU enabled), but only above floor_prob.
    Below floor, penalty is 0 - the TU is "safely disabled" and gradients from
    other losses (MSE) can potentially push it back up without L0 resistance.

    Args:
        log_alpha: Log-alpha parameters for TU masking
        floor_prob: Stop penalizing when sigmoid(log_alpha) drops below this.
            Default 0.2 means: once P(enabled) < 20%, no more L0 pressure.
        leak_coef: If > 0, adds weak positive penalty below floor that creates
            gradient pressure to re-enable disabled TUs. Typical value: 0.01-0.05.
            This prevents TUs from getting permanently stuck in disabled state.

    Returns:
        Per-TU penalty in range [0, 1] (or up to 1+leak_coef with leak below floor).
    """
    la = clamp_log_alpha(log_alpha)
    prob = jax.nn.sigmoid(la)
    # Penalty only above floor, normalized to [0, 1] range
    above_floor = jnp.maximum(prob - floor_prob, 0.0) / (1.0 - floor_prob)
    if leak_coef > 0:
        # Leak: positive penalty below floor → GD pushes prob UP toward floor
        # At prob=0: penalty = leak (max pressure to re-enable)
        # At prob=floor: penalty = 0 (no extra pressure)
        below_floor = jnp.maximum(floor_prob - prob, 0.0) / floor_prob
        return above_floor + leak_coef * below_floor
    return above_floor


def commitment_penalty(log_alpha: ArrayLike, margin: float = 0.05) -> jnp.ndarray:
    """Penalize TUs near the 0.5 decision threshold to encourage commitment.

    Only affects TUs in the narrow band [0.5-margin, 0.5+margin]. TUs outside
    this band receive zero penalty, avoiding the direction problem where
    entropy-based approaches push TUs the "wrong" way.

    Args:
        log_alpha: TU log-alpha parameters
        margin: Half-width of penalty band around 0.5. Default 0.05 means
            only TUs with sigmoid(log_alpha) in [0.45, 0.55] are penalized.

    Returns:
        Per-TU penalty in [0, 1], maximum at prob=0.5, zero outside margin band.
    """
    prob = jax.nn.sigmoid(clamp_log_alpha(log_alpha))
    dist = jnp.abs(prob - 0.5)
    violation = jnp.maximum(margin - dist, 0.0)
    return (violation / margin) ** 2


def l0_loss(
    log_alpha: ArrayLike,
    floor_prob: float = L0_PENALTY_FLOOR_PROB,
    tu_threshold: float | None = None,
    excess_exponent: float = 2.0,
    leak_coef: float = 0.0,
) -> jnp.ndarray:
    """L0 loss: sum of per-TU penalties, optionally with superlinear penalty above threshold.

    DEPRECATED: Use asymmetric_l0_loss() for smoother optimization landscape.
    """
    per_tu = l0_penalty(log_alpha, floor_prob, leak_coef)
    expected_count = jnp.sum(per_tu)

    if tu_threshold is None:
        return expected_count

    sharpness = 1.0
    smooth_excess = sharpness * jax.nn.softplus((expected_count - tu_threshold) / sharpness)
    excess_penalty = (smooth_excess**excess_exponent) / (tu_threshold ** (excess_exponent - 1))

    return expected_count + excess_penalty


def asymmetric_l0_loss(
    log_alpha: ArrayLike,
    threshold: float,
    floor_prob: float = L0_PENALTY_FLOOR_PROB,
    alpha_below: float = 0.5,
    beta_above: float = 2.0,
    blend_sharpness: float = 5.0,
    leak_coef: float = 0.0,
) -> jnp.ndarray:
    """Smooth asymmetric L0 penalty: sublinear below threshold, superlinear above."""
    log_alpha = jnp.asarray(log_alpha)
    assert log_alpha.ndim == 1, f"log_alpha must be 1D (n_tus,), got {log_alpha.ndim}D"
    assert threshold > 0, f"threshold must be positive, got {threshold}"
    assert 0 <= alpha_below < 1, (
        f"alpha_below must be in [0,1) for sublinear/zero, got {alpha_below}"
    )
    assert beta_above > 1, f"beta_above must be >1 for superlinear, got {beta_above}"

    per_tu = l0_penalty(log_alpha, floor_prob, leak_coef)
    count = jnp.sum(per_tu)

    safe_threshold = jnp.maximum(threshold, 1e-6)
    z = count / safe_threshold

    w = jax.nn.sigmoid(blend_sharpness * (z - 1))

    below_term = jnp.array(0.0) if alpha_below == 0 else z**alpha_below
    above_term = z**beta_above

    return safe_threshold * ((1 - w) * below_term + w * above_term)


def decode_latent_tu_masking(
    z: ArrayLike,
    W1: ArrayLike,
    b1: ArrayLike,
    W2: ArrayLike,
    b2: ArrayLike,
) -> jnp.ndarray:
    """Decode latent code to log_alpha via MLP: z → GELU(W1@z + b1) → W2@h + b2."""
    z, W1, b1, W2, b2 = map(jnp.asarray, (z, W1, b1, W2, b2))
    assert z.ndim == 1, f"z must be 1D (latent_dim,), got {z.ndim}D"
    assert W1.ndim == 2, f"W1 must be 2D (hidden_dim, latent_dim), got {W1.ndim}D"
    assert W1.shape[1] == z.shape[0], f"W1 cols ({W1.shape[1]}) must match z dim ({z.shape[0]})"
    assert b1.shape == (W1.shape[0],), f"b1 shape {b1.shape} must be ({W1.shape[0]},)"
    assert W2.shape[1] == W1.shape[0], f"W2 cols ({W2.shape[1]}) must match hidden ({W1.shape[0]})"
    assert b2.shape == (W2.shape[0],), f"b2 shape {b2.shape} must be ({W2.shape[0]},)"

    h = jax.nn.gelu(W1 @ z + b1)
    return W2 @ h + b2


def get_log_alpha_from_params(params, network_id: int) -> jnp.ndarray:
    """Get log_alpha for a network with protected TU enforcement.

    DEPRECATED: Use tumasking_strategy.get_full_log_alpha() and index by network_id.
    This function is kept for backwards compatibility but delegates to the SSOT.
    """
    from .tumasking_strategy import get_full_log_alpha

    log_alpha_full = get_full_log_alpha(params)
    if log_alpha_full is None:
        raise ValueError("No TU masking params found (neither latent nor direct)")
    return log_alpha_full[network_id]


def extract_tu_ids_from_network(network) -> list[str]:
    """Extract all unique TU IDs from a network's edges."""
    tu_ids = set()
    for edge in network.compute_graph.edges.values():
        tu_ids.update(edge.extra.tu_id)
    return sorted(tu_ids)


def build_tu_id_mapping(networks: list) -> tuple[list[str], dict[str, int]]:
    """Build TU ID mapping from multiple networks.

    Args:
        networks: List of networks to extract TU IDs from

    Returns:
        Tuple of (sorted_tu_ids, tu_id_to_idx mapping)
    """
    all_tu_ids = set()
    for net in networks:
        all_tu_ids.update(extract_tu_ids_from_network(net))
    sorted_tu_ids = sorted(all_tu_ids)
    return sorted_tu_ids, {tu_id: i for i, tu_id in enumerate(sorted_tu_ids)}


def set_binary_tu_mask(
    params,
    tu_ids: list[str],
    tu_id_to_idx: dict[str, int],
    n_networks: int,
    enabled_tus: dict[int, set[str]] | None = None,
    disabled_tus: dict[int, set[str]] | None = None,
) -> None:
    """Set binary TU mask in params. MUST be called BEFORE first stack.apply() for JIT compatibility.

    Args:
        params: ParameterTree to modify
        tu_ids: List of all TU IDs (from build_tu_id_mapping)
        tu_id_to_idx: TU ID to index mapping (from build_tu_id_mapping)
        n_networks: Number of networks in the stack
        enabled_tus: Dict mapping network_id -> set of TU IDs to enable. Default: all enabled.
        disabled_tus: Dict mapping network_id -> set of TU IDs to disable. Default: none disabled.
            (Use either enabled_tus OR disabled_tus, not both)
    """
    assert not (enabled_tus and disabled_tus), "Use either enabled_tus OR disabled_tus, not both"
    n_tus = len(tu_ids)
    mask = jnp.ones((n_networks, n_tus))

    if enabled_tus is not None:
        mask = jnp.zeros((n_networks, n_tus))
        for net_id, tu_names in enabled_tus.items():
            for tu_name in tu_names:
                if tu_name in tu_id_to_idx:
                    mask = mask.at[net_id, tu_id_to_idx[tu_name]].set(1.0)

    if disabled_tus is not None:
        for net_id, tu_names in disabled_tus.items():
            for tu_name in tu_names:
                if tu_name in tu_id_to_idx:
                    mask = mask.at[net_id, tu_id_to_idx[tu_name]].set(0.0)

    params.at(TU_BINARY_MASK_PATH, mask, overwrite=True)


def init_tu_log_alpha(
    n_tus: int,
    key: ArrayLike,
    init_mean: float = 2.0,
    init_std: float = 0.5,
) -> jnp.ndarray:
    """Initialize TU log_alpha parameters (positive = mostly enabled)."""
    return init_mean + init_std * jax.random.normal(key, shape=(n_tus,))


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
            tu_ids = edge.extra.tu_id
            valid_tu_ids = [tid for tid in tu_ids if tid in tu_id_to_idx]
            max_tus = max(max_tus, len(valid_tu_ids))

    tu_indices = np.full((len(nodelist), max_inputs, max_tus), TU_ALWAYS_ENABLED, dtype=np.int32)

    for i, node in enumerate(nodelist):
        edges = node.get_incoming_edges(stack)
        edges_sorted = sorted(edges, key=lambda e: e.to_input_slot)
        for j, edge in enumerate(edges_sorted):
            tu_ids = edge.extra.tu_id
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
                tu_ids = edge.extra.tu_id
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
                inverse_tu_ids.update(edge.extra.tu_id)
    return inverse_tu_ids


def extract_no_masking_tu_ids(networks: list) -> set[str]:
    """Extract TU IDs with no_masking=True from edge extra."""
    no_masking_tu_ids = set()
    for net in networks:
        graph = net.compute_graph
        for edge in graph.edges.values():
            no_masking_tu_ids.update(edge.extra.no_masking_tu_ids)
    return no_masking_tu_ids


def build_tu_id_mapping_excluding_inverse(
    networks: list,
) -> tuple[list[str], dict[str, int], set[str], set[str]]:
    """Build TU ID mapping. Returns (sorted_tu_ids, tu_id_to_idx, inverse_tu_ids, no_masking_tu_ids)."""
    all_tu_ids = set()
    for net in networks:
        all_tu_ids.update(extract_tu_ids_from_network(net))

    inverse_tu_ids = extract_tu_ids_for_inverse_nodes(networks)
    no_masking_tu_ids = extract_no_masking_tu_ids(networks)
    sorted_tu_ids = sorted(all_tu_ids)
    tu_id_to_idx = {tu_id: i for i, tu_id in enumerate(sorted_tu_ids)}

    return sorted_tu_ids, tu_id_to_idx, inverse_tu_ids, no_masking_tu_ids


def _apply_binary_mask_single(tu_idx: ArrayLike, binary_mask: ArrayLike) -> jnp.ndarray:
    safe_idx = jnp.maximum(tu_idx, 0)
    mask_val = binary_mask[safe_idx]
    return jnp.where(tu_idx >= 0, mask_val, 1.0)


def _apply_binary_mask_multi(tu_indices: ArrayLike, binary_mask: ArrayLike) -> jnp.ndarray:
    safe_indices = jnp.maximum(tu_indices, 0)
    mask_vals = binary_mask[safe_indices]
    valid = tu_indices >= 0
    masked_vals = jnp.where(valid, mask_vals, 0.0)
    all_padding = jnp.all(~valid)
    return jnp.where(all_padding, 1.0, jnp.max(masked_vals))


def _apply_binary_masks(
    tu_indices: ArrayLike,
    binary_mask: ArrayLike,
    *,
    is_multi_tu: bool,
) -> jnp.ndarray:
    tu_indices = jnp.asarray(tu_indices)
    binary_mask = jnp.asarray(binary_mask)
    assert binary_mask.ndim == 1, f"binary_mask must be 1D, got {binary_mask.ndim}D"
    if is_multi_tu:
        assert tu_indices.ndim == 2, (
            f"is_multi_tu=True requires 2D tu_indices, got {tu_indices.ndim}D"
        )
        return jax.vmap(lambda idx: _apply_binary_mask_multi(idx, binary_mask))(tu_indices)
    else:
        assert tu_indices.ndim == 1, (
            f"is_multi_tu=False requires 1D tu_indices, got {tu_indices.ndim}D"
        )
        return jax.vmap(lambda idx: _apply_binary_mask_single(idx, binary_mask))(tu_indices)


def get_tu_masks(
    params,
    tu_indices: ArrayLike,
    network_id: int,
    *,
    is_multi_tu: bool,
    use_probabilistic_or: bool = False,
) -> jnp.ndarray:
    """Unified TU masking with mode-specific handling.

    Binary mask mode: Direct mask indexing (gradients flow for mask optimization).
    Log_alpha modes: Route through get_full_log_alpha() for protected TU enforcement.

    Args:
        params: ParameterTree or dict-like containing mask parameters
        tu_indices: TU indices for this node type
        network_id: Network index for slicing 2D mask arrays. REQUIRED.
        is_multi_tu: True for input_tu_indices (OR reduction), False for output_tu_indices
        use_probabilistic_or: If True, use P(any)=1-∏(1-p) for multi-TU edges instead of softmax OR.
    """
    from .tumasking_strategy import get_full_log_alpha

    tu_indices = jnp.asarray(tu_indices)
    n_inputs = tu_indices.shape[0]

    if TU_BINARY_MASK_PATH in params:
        binary_mask = jnp.asarray(params[TU_BINARY_MASK_PATH])
        assert binary_mask.ndim == 2, f"binary_mask must be 2D, got {binary_mask.ndim}D"
        return _apply_binary_masks(tu_indices, binary_mask[network_id], is_multi_tu=is_multi_tu)

    log_alpha_full = get_full_log_alpha(params)
    if log_alpha_full is None:
        return jnp.ones(n_inputs)

    tu_log_alpha = log_alpha_full[network_id]
    return compute_binary_masks(
        tu_indices, tu_log_alpha, is_multi_tu=is_multi_tu, use_probabilistic_or=use_probabilistic_or
    )
