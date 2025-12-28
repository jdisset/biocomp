"""Loss functions for circuit design optimization."""

import jax
import jax.numpy as jnp
from jax import vmap, lax
from jax.tree_util import Partial

import numpy as np
from assertpy import assert_that

from .parameters import ParameterTree
from .optimutils import as_schedule, jax_three_phase_schedule
from .tumasking import TU_LOG_ALPHA_PATH, MIN_TEMPERATURE
from .logging_config import get_logger
from .designdebug import is_design_debug_enabled, save_debug_state

logger = get_logger(__name__)


def _sanitize(x):
    return jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_coords(x):
    mu, sd = jnp.mean(x, 0, keepdims=True), jnp.std(x, 0, keepdims=True) + 1e-8
    return (x - mu) / sd


def _gauss1d(sigma, radius=5):
    x = jnp.arange(-radius, radius + 1, dtype=jnp.float32)
    k = jnp.exp(-(x**2) / (2 * sigma**2))
    return k / jnp.sum(k)


def _gauss_blur2d(x, kernel):
    def conv1d(arr, axis):
        pad = (kernel.shape[0] // 2,) * 2
        pads = [(0, 0)] * arr.ndim
        pads[axis] = pad
        arr_pad = jnp.pad(arr, pads, mode="reflect")
        if axis == -1 or axis == arr.ndim - 1:
            w = kernel[None, None, None, :]
            y = lax.conv_general_dilated(
                arr_pad[None, None], w, (1, 1), "VALID", dimension_numbers=("NCHW", "OIHW", "NCHW")
            )[0, 0]
        else:
            arr_swap = jnp.swapaxes(arr_pad, axis, -1)
            w = kernel[None, None, None, :]
            y = lax.conv_general_dilated(
                arr_swap[None, None], w, (1, 1), "VALID", dimension_numbers=("NCHW", "OIHW", "NCHW")
            )[0, 0]
            y = jnp.swapaxes(y, -1, axis)
        return y

    return conv1d(conv1d(x, -1), -2)


def sinkhorn_divergence_conv(a, b, eps, n_iters=80, uniform_mix=1e-9, min_mass=1e-6):
    """Grid-based Sinkhorn divergence using fast Gaussian convolutions."""
    assert a.shape == b.shape, f"sinkhorn a/b shape mismatch: {a.shape} vs {b.shape}"
    assert eps > 0, f"sinkhorn eps must be positive, got {eps}"

    a = jnp.maximum(_sanitize(a.astype(jnp.float32)), 1e-24)
    b = jnp.maximum(_sanitize(b.astype(jnp.float32)), 1e-24)
    a = a + min_mass / a.size  # uniform mass floor prevents grad explosion
    b = b + min_mass / b.size
    a = a / a.sum()
    b = b / b.sum()

    sigma = jnp.sqrt(eps / 2.0)
    kernel = _gauss1d(sigma)

    def blurred_with_floor(x):
        blurred = _gauss_blur2d(x, kernel)
        uniform = x.sum() / x.size
        return (1 - uniform_mix) * blurred + uniform_mix * uniform

    def sinkhorn_iters(m1, m2, n):
        u, v = jnp.ones_like(m1), jnp.ones_like(m2)

        def step(carry, _):
            u, v = carry
            u_new = m1 / blurred_with_floor(v)
            v_new = m2 / blurred_with_floor(u_new)
            return (u_new, v_new), None

        (u, v), _ = lax.scan(step, (u, v), None, length=n)
        return u, v

    def ot_cost(m1, m2, n):
        u, v = sinkhorn_iters(m1, m2, n)
        cost = eps * (
            jnp.sum(m1 * jnp.log(jnp.maximum(u, 1e-24)))
            + jnp.sum(m2 * jnp.log(jnp.maximum(v, 1e-24)))
        )
        return cost

    ot_ab = ot_cost(a, b, n_iters)
    ot_aa = ot_cost(a, a, max(20, n_iters // 2))
    ot_bb = ot_cost(b, b, max(20, n_iters // 2))
    return jnp.maximum(0.0, ot_ab - 0.5 * (ot_aa + ot_bb))


def _epsilon_from_x_median(xn, eps=1e-12):
    B = xn.shape[0]
    d2 = jnp.sum((xn[:, None, :] - xn[None, :, :]) ** 2, axis=-1)
    nn_sq = jnp.min(d2 + jnp.eye(B) * 1e9, axis=1)
    return jax.lax.stop_gradient(jnp.maximum(0.5 * jnp.median(nn_sq), eps))


def _ott_sinkhorn_div(xn, a, b, epsilon, tau=None, **solver_kw):
    from ott.geometry import pointcloud
    from ott.problems.linear import linear_problem
    from ott.solvers.linear import sinkhorn

    geom = pointcloud.PointCloud(xn, xn, epsilon=epsilon)
    solver = sinkhorn.Sinkhorn(lse_mode=True, **solver_kw)

    def ot(u, v):
        kw = {"tau_a": tau, "tau_b": tau} if tau else {}
        return solver(linear_problem.LinearProblem(geom, a=u, b=v, **kw)).reg_ot_cost

    return ot(a, b) - 0.5 * (ot(a, a) + ot(b, b))


def proj_nonneg_ste(z, leak=1e-3, cap=None):
    """Project to nonnegative with straight-through estimator."""
    z_clip = jnp.clip(z, 0.0, cap) if cap is not None else jnp.maximum(z, 0.0)
    z_leaky = jnp.where(z >= 0.0, z, leak * z)
    return z_clip + jax.lax.stop_gradient(z_leaky - z_clip)


def sinkhorn_divergence_unbalanced(x, y, yhat, epsilon=0.01, tau=0.9, cap=0.5, **kw):
    xn = _normalize_coords(_sanitize(x))
    a = proj_nonneg_ste(_sanitize(yhat), cap=cap)
    b = jnp.clip(_sanitize(y), 0.0, cap)
    eps = epsilon if epsilon else _epsilon_from_x_median(xn)
    div = _ott_sinkhorn_div(
        xn,
        a,
        b,
        eps,
        tau=tau,
        threshold=kw.get("threshold", 1e-3),
        max_iterations=kw.get("max_iterations", 300),
    )
    return jnp.maximum(_sanitize(div), 0.0)


def sinkhorn_divergence_balanced(
    x, y, yhat, epsilon=0.01, cap=None, mass_floor=1e-8, lambda_neg=1e-4, **kw
):
    xn = _normalize_coords(x)
    eps = epsilon if epsilon else 0.03 * jnp.mean(jnp.sum((xn[:, None] - xn[None]) ** 2, -1))
    a, b = proj_nonneg_ste(yhat, cap=cap), proj_nonneg_ste(y, cap=cap)
    floor = (mass_floor * jnp.maximum(jnp.sum(b), 1.0)) / a.size
    a, b = a + floor, b + floor
    a = a * (jnp.sum(b) / (jnp.sum(a) + 1e-8))
    div = _ott_sinkhorn_div(xn, a, b, eps)
    return jnp.where(jnp.isfinite(div), div, 0.0) + lambda_neg * jnp.mean(jax.nn.relu(-yhat))


def zncc_loss(x, y, yhat, eps=1e-6, **kw):
    """Zero-mean normalized cross-correlation loss."""
    assert y.shape == yhat.shape, f"zncc_loss shape mismatch: y={y.shape} vs yhat={yhat.shape}"

    y, yhat = _sanitize(y), _sanitize(yhat)
    y0, yhat0 = y - jnp.mean(y), yhat - jnp.mean(yhat)
    cov = jnp.mean(y0 * yhat0)
    var_y, var_yhat = jnp.mean(y0**2), jnp.mean(yhat0**2)
    std_product = jnp.sqrt((var_y + eps) * (var_yhat + eps))
    return 1.0 - cov / std_product


def huber_loss(x, y, yhat, delta=0.01, **kw):
    y, yhat = _sanitize(y), _sanitize(yhat)
    r = jnp.abs(yhat - y)
    return jnp.mean(jnp.where(r <= delta, 0.5 * r**2, delta * (r - 0.5 * delta)))


def huber_zncc_loss(x, y, yhat, delta=0.01, zncc_weight=0.1, **kw):
    return zncc_weight * zncc_loss(x, y, yhat) + (1 - zncc_weight) * huber_loss(
        x, y, yhat, delta=delta
    )


def wasserstein_zncc_loss(x, y, yhat, zncc_weight=0.4, **kw):
    return zncc_weight * zncc_loss(x, y, yhat) + (1 - zncc_weight) * sinkhorn_divergence_balanced(
        x, y, yhat, **kw
    )


def spectral_loss(x, y, yhat, **kw):
    return jnp.mean((jnp.abs(jnp.fft.fft2(y)) - jnp.abs(jnp.fft.fft2(yhat))) ** 2)


def mse_loss(x, y, yhat, **kw):
    assert y.shape == yhat.shape, f"mse_loss shape mismatch: y={y.shape} vs yhat={yhat.shape}"
    return jnp.mean((yhat - y) ** 2)


def simse_loss(x, y, yhat, eps=1e-8, **kw):
    """Scale-invariant MSE loss."""
    y0, yhat0 = _sanitize(y - jnp.mean(y)), _sanitize(yhat - jnp.mean(yhat))
    vy, vyhat = jnp.sum(y0**2), jnp.sum(yhat0**2)
    alpha = jnp.where(vyhat > eps, jnp.sum(y0 * yhat0) / (vyhat + eps), 0.0)
    return jnp.nan_to_num(
        jnp.sum((y0 - alpha * yhat0) ** 2) / jnp.maximum(vy, eps), nan=1.0, posinf=1.0, neginf=1.0
    )


def lncc_loss(x, y, yhat, target_neighbors=12, eps=1e-6, **kw):
    """Local normalized cross-correlation loss."""
    assert x.shape[0] == y.size, f"lncc_loss: x batch size {x.shape[0]} != y size {y.size}"
    assert y.size == yhat.size, f"lncc_loss: y size {y.size} != yhat size {yhat.size}"

    x, y, yhat = (
        _sanitize(x),
        _sanitize(jnp.asarray(y).reshape(-1)),
        _sanitize(jnp.asarray(yhat).reshape(-1)),
    )
    B = x.shape[0]
    if B <= 1:
        return jnp.array(0.0, dtype=x.dtype)

    d2 = jnp.sum((x[:, None] - x[None]) ** 2, axis=-1)
    sigma = jnp.maximum(
        0.5
        * (target_neighbors ** (1.0 / x.shape[-1]))
        * jnp.sqrt(jnp.maximum(jnp.median(jnp.min(d2 + jnp.eye(B) * 1e9, axis=1)), 0) + eps),
        eps,
    )
    K = jnp.where(jnp.isfinite(K := jnp.exp(-d2 / (2 * sigma**2 + eps))), K, 0.0)
    W = jnp.where((rs := jnp.sum(K, 1, keepdims=True)) > 0, K / (rs + eps), 0.0)

    Ey, Eyh, Ey2, Eyh2, Eyyh = W @ y, W @ yhat, W @ (y * y), W @ (yhat * yhat), W @ (y * yhat)
    ncc = (Eyyh - Ey * Eyh) / (
        jnp.sqrt(jnp.maximum(Ey2 - Ey**2, 0) * jnp.maximum(Eyh2 - Eyh**2, 0)) + eps
    )
    return 1.0 - jnp.mean(jnp.where(jnp.isfinite(ncc), ncc, 0.0))


def lncc_grid_loss(x, y, yhat, k=7, eps=1e-6, **kw):
    """Local NCC on 2D grid using box filter."""
    assert y.ndim == 2, f"lncc_grid_loss: y must be 2D grid, got {y.ndim}D"
    assert y.shape == yhat.shape, f"lncc_grid_loss shape mismatch: y={y.shape} vs yhat={yhat.shape}"

    y, yhat = _sanitize(y), _sanitize(yhat)
    r, N = k // 2, k * k

    def box2d(a):
        a = jnp.pad(a, ((r + 1, r), (r + 1, r)), mode="edge")
        s = jnp.cumsum(jnp.cumsum(a, 0), 1)
        return (
            s[: -2 * r - 1, : -2 * r - 1]
            - s[: -2 * r - 1, 2 * r + 1 :]
            - s[2 * r + 1 :, : -2 * r - 1]
            + s[2 * r + 1 :, 2 * r + 1 :]
        )

    m0, m1 = box2d(y) / N, box2d(yhat) / N
    y0c, y1c = y - m0, yhat - m1
    var_y, var_yhat = box2d(y0c**2), box2d(y1c**2)
    cov = box2d(y0c * y1c)
    std_product = jnp.sqrt((var_y + eps) * (var_yhat + eps))
    lncc = jnp.clip(cov / std_product, -1, 1)
    return 1.0 - jnp.mean(lncc)


def simse_lncc_loss(x, y, yhat, simse_weight=0.3, **kw):
    return simse_weight * simse_loss(x, y, yhat) + (1 - simse_weight) * lncc_loss(x, y, yhat)


def soft_tucount_penalty(W, max_tus=5, rel_active=1e-3, width=2e-4):
    """Penalty for having more than max_tus active TUs per co-transfection (aggregation row).

    Args:
        W: Ratio matrix with shape (n_aggregation_nodes, n_members)
        max_tus: Maximum allowed active TUs per row before penalty kicks in (default 5)
        rel_active: Threshold - ratio/max_ratio above this counts as "active" (default 1e-3)
        width: Sigmoid sharpness for soft counting (default 2e-4)

    Returns:
        Penalty value: sum of squared excesses over max_tus across all rows
    """
    A = jnp.abs(W)
    m = jnp.max(A, axis=1, keepdims=True)
    norm = jnp.where(m > 0, A / (m + 1e-12), 0.0)
    soft_count = jnp.sum(jax.nn.sigmoid((norm - rel_active) / (width + 1e-12)), axis=1)
    return jnp.sum(jnp.square(jax.nn.relu(soft_count - max_tus)))


def get_tucount_penalty_for_leaf(p, max_tus=5, rel_active=1e-3, width=2e-4):
    """Get TU count penalty for a parameter leaf (handles ArrayRef and raw arrays)."""
    kw = dict(max_tus=max_tus, rel_active=rel_active, width=width)
    if hasattr(p, "view"):
        try:
            return soft_tucount_penalty(p.view(), **kw)
        except Exception:
            return sum(soft_tucount_penalty(p.tree[path], **kw) for path in p.paths)
    return soft_tucount_penalty(p, **kw)


def ratio_spread_penalty(W, max_ratio=100.0, eps=1e-9):
    """Penalty for ratio spread exceeding max_ratio."""
    A = jnp.abs(W)
    log_max_ratio = jnp.log(max_ratio + eps)
    pos_mask = A > eps
    log_A = jnp.where(pos_mask, jnp.log(A + eps), -jnp.inf)
    log_max = jnp.max(jnp.where(pos_mask, log_A, -jnp.inf), axis=1)
    log_min = jnp.min(jnp.where(pos_mask, log_A, jnp.inf), axis=1)
    log_spread = log_max - log_min
    excess = jax.nn.relu(log_spread - log_max_ratio)
    return jnp.sum(jnp.square(excess))


def get_spread_penalty_for_leaf(p, max_ratio=100.0):
    print("enforcing max aggregation ratio of", max_ratio)
    if hasattr(p, "view"):
        try:
            return ratio_spread_penalty(p.view(), max_ratio=max_ratio)
        except Exception:
            return sum(ratio_spread_penalty(p.tree[path], max_ratio=max_ratio) for path in p.paths)
    return ratio_spread_penalty(p, max_ratio=max_ratio)


def _ratio_mask_coupling_single_target(
    params: ParameterTree,
    ratio_paths: list[str],
    tu_log_alpha_2d: jnp.ndarray,
    min_ratio_threshold: float,
    target_idx: int | jnp.ndarray,
    ratios_are_3d: bool,
) -> jnp.ndarray:
    """Compute coupling penalty for a single target's tu_log_alpha.

    Args:
        params: Parameter tree containing ratios, output_tu_indices, node_network_ids
        ratio_paths: List of paths to ratio parameters
        tu_log_alpha_2d: TU log_alpha for one target, shape (n_networks, n_tus)
        min_ratio_threshold: Coupling activates only when normalized ratio < this
        target_idx: Which target's ratios to slice (used when ratios_are_3d=True)
        ratios_are_3d: If True, expect ratios shape (n_targets, n_nodes, n_outputs) and slice by target_idx.
                       If False, expect ratios shape (n_nodes, n_outputs) and target_idx is ignored.

    Returns:
        Scalar coupling penalty for this target
    """
    assert tu_log_alpha_2d.ndim == 2, (
        f"tu_log_alpha_2d must be 2D (n_networks, n_tus), got shape {tu_log_alpha_2d.shape}. "
        f"This function processes one target at a time."
    )
    assert 0 <= min_ratio_threshold <= 1, (
        f"min_ratio_threshold must be in [0, 1], got {min_ratio_threshold}"
    )
    assert isinstance(ratios_are_3d, bool), (
        f"ratios_are_3d must be explicit bool, got {type(ratios_are_3d)}. "
        f"No silent shape detection allowed."
    )

    n_networks, n_tus = tu_log_alpha_2d.shape
    assert n_networks > 0 and n_tus > 0, f"Empty tu_log_alpha: {tu_log_alpha_2d.shape}"

    total_penalty = jnp.array(0.0)

    for ratio_path in ratio_paths:
        ratio_path_str = str(ratio_path) if not isinstance(ratio_path, str) else ratio_path
        namespace = ratio_path_str.rsplit("/ratios", 1)[0]
        tu_indices_path = f"{namespace}/output_tu_indices"
        network_ids_path = f"{namespace}/node_network_ids"

        if tu_indices_path not in params or network_ids_path not in params:
            continue

        ratios = jnp.abs(params[ratio_path])
        tu_indices = params[tu_indices_path]
        node_network_ids = params[network_ids_path]

        if ratios_are_3d:
            assert ratios.ndim == 3, (
                f"ratios_are_3d=True but ratios.ndim={ratios.ndim} at {ratio_path}. "
                f"Expected (n_targets, n_nodes, n_outputs), got {ratios.shape}."
            )
            ratios = ratios[target_idx]
            assert tu_indices.ndim == 3, (
                f"ratios_are_3d=True requires tu_indices to be 3D, got {tu_indices.ndim}D at {tu_indices_path}"
            )
            tu_indices = tu_indices[target_idx]
            assert node_network_ids.ndim == 2, (
                f"ratios_are_3d=True requires node_network_ids to be 2D, got {node_network_ids.ndim}D at {network_ids_path}"
            )
            node_network_ids = node_network_ids[target_idx]
        else:
            assert ratios.ndim == 2, (
                f"ratios_are_3d=False but ratios.ndim={ratios.ndim} at {ratio_path}. "
                f"Expected (n_nodes, n_outputs), got {ratios.shape}."
            )
            assert tu_indices.ndim == 2, (
                f"ratios_are_3d=False requires tu_indices to be 2D, got {tu_indices.ndim}D"
            )
            assert node_network_ids.ndim == 1, (
                f"ratios_are_3d=False requires node_network_ids to be 1D, got {node_network_ids.ndim}D"
            )
        assert ratios.shape == tu_indices.shape, (
            f"Shape mismatch at {ratio_path}: ratios {ratios.shape} vs tu_indices {tu_indices.shape}"
        )
        assert ratios.shape[0] == node_network_ids.shape[0], (
            f"Node count mismatch at {ratio_path}: ratios has {ratios.shape[0]} nodes, "
            f"node_network_ids has {node_network_ids.shape[0]}"
        )

        n_nodes, n_outputs = ratios.shape
        assert n_nodes > 0, "n_nodes must be > 0"
        assert n_outputs > 0, "n_outputs must be > 0"

        # normalize ratios per node using MAX normalization (not sum)
        # this ensures min_ratio_threshold has consistent meaning regardless of TU count
        # e.g., threshold=0.005 means "less than 0.5% of the largest ratio"
        ratio_max = jnp.max(ratios, axis=-1, keepdims=True)
        normalized_ratios = ratios / jnp.maximum(ratio_max, 1e-8)

        network_ids_expanded = jnp.broadcast_to(node_network_ids[:, None], (n_nodes, n_outputs))

        valid_tu_mask = (tu_indices >= 0).astype(jnp.float32)
        safe_tu_indices = jnp.maximum(tu_indices, 0)
        # clamp network_ids and tu_indices to valid range (defensive against corruption)
        safe_network_ids = jnp.clip(network_ids_expanded, 0, n_networks - 1)
        safe_tu_indices = jnp.clip(safe_tu_indices, 0, n_tus - 1)

        # index into 2D tu_log_alpha
        tu_log_alpha_per_ratio = tu_log_alpha_2d[safe_network_ids, safe_tu_indices]
        tu_enabled_prob = jax.nn.sigmoid(tu_log_alpha_per_ratio)

        # penalty only when ratio < threshold
        below_threshold = jax.nn.relu(min_ratio_threshold - normalized_ratios)
        per_element_penalty = tu_enabled_prob * below_threshold * valid_tu_mask

        # use nan_to_num for safety (NaN checks done via checkify in tests)
        per_element_penalty = jnp.nan_to_num(per_element_penalty, nan=0.0, posinf=0.0, neginf=0.0)

        total_penalty = total_penalty + jnp.sum(per_element_penalty)

    return total_penalty


def ratio_mask_coupling_penalty(
    params: ParameterTree,
    ratio_paths: list[str],
    tu_log_alpha: jnp.ndarray,
    min_ratio_threshold: float = 0.005,
    return_per_target: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    """Coupling loss: push down tu_log_alpha when ratio is below threshold.

    ONLY activates when normalized_ratio < min_ratio_threshold. When ratios are in
    acceptable range, this returns 0 (no coupling).

    This creates gradient pressure to disable TUs (via hard-concrete) when their
    corresponding ratios are too small, unifying the two disabling mechanisms.

    Uses MAX normalization (ratio / max_ratio) not sum normalization, so the
    threshold has consistent meaning regardless of TU count. E.g., threshold=0.005
    means "ratio is less than 0.5% of the largest ratio in that aggregation".

    Args:
        params: Parameter tree containing ratios, output_tu_indices, node_network_ids
        ratio_paths: List of paths to ratio parameters (e.g., ['local/layer_3/ratios'])
        tu_log_alpha: TU log_alpha array, shape (n_targets, n_networks, n_tus) or (n_networks, n_tus)
        min_ratio_threshold: Coupling activates when (ratio/max_ratio) < this (default 0.005 = 0.5%)
        return_per_target: If True, also return per-target breakdown shape (n_targets,)

    Returns:
        If return_per_target=False: Scalar coupling penalty (0 if all ratios are above threshold)
        If return_per_target=True: (scalar, per_target_penalty) where per_target has shape (n_targets,)

    Note: Runtime value checks (NaN, bounds) tested via checkify in tests.
    """
    # static assertions (evaluated at trace/call time, not runtime)
    assert isinstance(ratio_paths, list), f"ratio_paths must be a list, got {type(ratio_paths)}"
    assert isinstance(min_ratio_threshold, (int, float)), (
        f"min_ratio_threshold must be numeric, got {type(min_ratio_threshold)}"
    )
    assert 0 <= min_ratio_threshold <= 1, (
        f"min_ratio_threshold must be in [0, 1], got {min_ratio_threshold}"
    )
    assert tu_log_alpha.ndim in (2, 3), (
        f"tu_log_alpha must be 2D or 3D, got {tu_log_alpha.ndim}D with shape {tu_log_alpha.shape}"
    )

    if tu_log_alpha.ndim == 2:
        scalar = _ratio_mask_coupling_single_target(
            params,
            ratio_paths,
            tu_log_alpha,
            min_ratio_threshold,
            target_idx=0,
            ratios_are_3d=False,
        )
        if return_per_target:
            return scalar, jnp.array([scalar])
        return scalar

    assert tu_log_alpha.ndim == 3, f"tu_log_alpha must be 2D or 3D, got {tu_log_alpha.ndim}D"
    n_targets, n_networks, n_tus = tu_log_alpha.shape
    assert n_targets > 0 and n_networks > 0 and n_tus > 0, (
        f"Empty tu_log_alpha: {tu_log_alpha.shape}"
    )

    target_indices = jnp.arange(n_targets)

    def compute_for_target(target_idx, target_tu_log_alpha):
        return _ratio_mask_coupling_single_target(
            params,
            ratio_paths,
            target_tu_log_alpha,
            min_ratio_threshold,
            target_idx=target_idx,
            ratios_are_3d=True,
        )

    per_target_penalty = vmap(compute_for_target)(target_indices, tu_log_alpha)
    assert per_target_penalty.shape == (n_targets,), (
        f"per_target_penalty shape mismatch: expected ({n_targets},), got {per_target_penalty.shape}"
    )

    scalar = jnp.sum(per_target_penalty)
    if return_per_target:
        return scalar, per_target_penalty
    return scalar


def _ern_tu_tying_single_target(
    params: ParameterTree,
    ern_namespaces: list[str],
    tu_log_alpha_2d: jnp.ndarray,
    target_idx: int | jnp.ndarray,
    input_tu_indices_are_3d: bool,
) -> jnp.ndarray:
    """ERN TU tying: if pos (mRNA target) is disabled, penalize neg (ERN protein) being enabled."""
    assert tu_log_alpha_2d.ndim == 2, f"need 2D tu_log_alpha, got {tu_log_alpha_2d.shape}"
    n_networks, _ = tu_log_alpha_2d.shape
    total_penalty = jnp.array(0.0)

    for namespace in ern_namespaces:
        tu_path = f"{namespace}/input_tu_indices"
        net_path = f"{namespace}/node_network_ids"
        if tu_path not in params or net_path not in params:
            continue

        input_tu_indices = params[tu_path]
        node_network_ids = params[net_path]

        if input_tu_indices_are_3d:
            assert input_tu_indices.ndim == 4, (
                f"need 4D input_tu_indices, got {input_tu_indices.ndim}D"
            )
            assert node_network_ids.ndim == 2, (
                f"need 2D node_network_ids, got {node_network_ids.ndim}D"
            )
            input_tu_indices = input_tu_indices[target_idx]
            node_network_ids = node_network_ids[target_idx]
        else:
            assert input_tu_indices.ndim == 3, (
                f"need 3D input_tu_indices, got {input_tu_indices.ndim}D"
            )
            assert node_network_ids.ndim == 1, (
                f"need 1D node_network_ids, got {node_network_ids.ndim}D"
            )

        n_nodes, n_inputs, _ = input_tu_indices.shape
        assert n_inputs == 2, f"ERN needs 2 inputs, got {n_inputs}"

        neg_tu_indices = input_tu_indices[:, 0, :]
        pos_tu_indices = input_tu_indices[:, 1, :]

        def get_max_log_alpha(tu_indices_row, network_id):
            valid = tu_indices_row >= 0
            safe_idx = jnp.maximum(tu_indices_row, 0)
            safe_net = jnp.clip(network_id, 0, n_networks - 1)
            las = tu_log_alpha_2d[safe_net, safe_idx]
            las = jnp.where(valid, las, 10.0)  # -1 = always enabled
            return jnp.max(las)

        neg_las = vmap(get_max_log_alpha)(neg_tu_indices, node_network_ids)
        pos_las = vmap(get_max_log_alpha)(pos_tu_indices, node_network_ids)

        # penalty = P(pos disabled) * relu(neg_log_alpha - pos_log_alpha)
        pos_disabled_prob = jax.nn.sigmoid(-pos_las)
        excess = jax.nn.relu(neg_las - pos_las)
        total_penalty = total_penalty + jnp.sum(pos_disabled_prob * excess)

    return total_penalty


def ern_tu_tying_penalty(
    params: ParameterTree,
    ern_namespaces: list[str],
    tu_log_alpha: jnp.ndarray,
) -> jnp.ndarray:
    """ERN TU tying: push neg TU down when pos TU is disabled. One-way coupling."""
    if not ern_namespaces:
        return jnp.array(0.0)
    assert tu_log_alpha.ndim in (2, 3), f"need 2D or 3D tu_log_alpha, got {tu_log_alpha.ndim}D"

    if tu_log_alpha.ndim == 2:
        return _ern_tu_tying_single_target(
            params, ern_namespaces, tu_log_alpha, target_idx=0, input_tu_indices_are_3d=False
        )

    def compute_for_target(target_idx, target_la):
        return _ern_tu_tying_single_target(
            params, ern_namespaces, target_la, target_idx=target_idx, input_tu_indices_are_3d=True
        )

    return jnp.sum(vmap(compute_for_target)(jnp.arange(tu_log_alpha.shape[0]), tu_log_alpha))


def per_batch_apply(params, X, Z, keys, stack, tu_uniform=None):
    def apply_single(x, z, key):
        return stack.apply(params, x, z, key, tu_enabled_random_vars=tu_uniform)

    return vmap(apply_single)(X, Z, keys)


def per_target_apply(params, X, Z, keys, stack, tu_uniform=None):
    def apply_target(p, x, z, k, tu_u):
        return per_batch_apply(p, x, z, k, stack, tu_uniform=tu_u)

    tu_uniform_axes = 0 if tu_uniform is not None else None
    return vmap(apply_target, in_axes=(0, 1, 1, 1, tu_uniform_axes), out_axes=1)(
        params, X, Z, keys, tu_uniform
    )


@Partial(jax.jit, static_argnames=["stack"])
def per_replicate_apply(params, X, Z, keys, stack, tu_uniform=None):
    def apply_rep(p, x, z, k, tu_u):
        return per_target_apply(p, x, z, k, stack, tu_uniform=tu_u)

    tu_uniform_axes = 0 if tu_uniform is not None else None
    return vmap(apply_rep, in_axes=(0, 0, 0, 0, tu_uniform_axes))(params, X, Z, keys, tu_uniform)


@Partial(jax.jit, static_argnames=["lossfunc", "n_inputs_per_network"])
def compute_all_losses(x, y, yhatdep, lossfunc, n_inputs_per_network=2):
    assert x.shape[-1] % n_inputs_per_network == 0, (
        f"x.shape[-1]={x.shape[-1]} not divisible by n_inputs_per_network={n_inputs_per_network}. "
        "This would cause silent truncation in n_networks calculation."
    )
    n_networks = int(x.shape[-1] / n_inputs_per_network)
    batch_size, n_targets = y.shape[0], y.shape[1]

    assert_that(x).has_shape((batch_size, n_targets, n_networks * n_inputs_per_network))
    assert_that(yhatdep).has_shape((batch_size, n_targets, n_networks))
    assert_that(y).has_same_shape(yhatdep)
    assert jnp.all(jnp.isfinite(yhatdep)), "NaN/Inf in predictions will poison loss"

    xsplit = jnp.reshape(x, (batch_size, n_targets, n_networks, n_inputs_per_network))
    return vmap(vmap(lossfunc, in_axes=(1, 1, 1)), in_axes=(1, 1, 1))(xsplit, yhatdep, y)


def _sample_tu_uniform(params, key, n_samples=1):
    """Sample TU uniform random variables for Hard Concrete masking.

    Args:
        params: Parameter tree containing tu_log_alpha
        key: JAX random key
        n_samples: Number of independent samples for variance reduction

    Returns:
        If n_samples=1: shape (n_targets, n_networks, n_tus)
        If n_samples>1: shape (n_samples, n_targets, n_networks, n_tus)
    """
    if TU_LOG_ALPHA_PATH not in params:
        return None
    log_alpha = params[TU_LOG_ALPHA_PATH]
    # shape is (n_targets, n_networks, n_tus) after vmap over replicates
    assert log_alpha.ndim == 3, f"expected 3D tu_log_alpha, got {log_alpha.shape}"
    if n_samples == 1:
        return jax.random.uniform(key, log_alpha.shape, minval=1e-6, maxval=1.0 - 1e-6)
    # multiple samples for variance reduction
    shape = (n_samples,) + log_alpha.shape
    return jax.random.uniform(key, shape, minval=1e-6, maxval=1.0 - 1e-6)


def _validate_temperature_schedule(tu_temperature, total_steps: int = 10000) -> None:
    """Validate tu_temperature schedule at config time (before tracing).

    This catches config bugs early - before we enter JIT/scan where Python
    assertions don't work. Runtime safety is provided by jnp.maximum() clamping.
    """
    sched = as_schedule(tu_temperature)
    # check schedule at endpoints and midpoint
    for step in [0, total_steps // 2, total_steps]:
        temp = float(sched(step))
        assert temp >= 0, (
            f"tu_temperature schedule returns negative value {temp} at step {step}. "
            f"Temperature must be >= 0 for Hard Concrete distribution."
        )
        if temp < MIN_TEMPERATURE:
            import warnings

            warnings.warn(
                f"tu_temperature={temp} at step {step} is below MIN_TEMPERATURE={MIN_TEMPERATURE}. "
                f"Will be clamped to {MIN_TEMPERATURE} at runtime.",
                stacklevel=3,
            )


HYPEROPT_SCHEDULE_NAMESPACE = "hyperopt_schedules"


def normalize_schedule_spec(spec):
    """Convert various schedule specifications to universal three-phase params.

    Supports:
        - float/int: Constant schedule (all phases same value)
        - dict with 'start', 'end': Linear schedule over all steps
        - dict with 'phase1_value', etc.: Full three-phase schedule
        - callable: Optax schedule (NOT for hyperopt mode, use for backward compat only)

    Returns:
        dict with keys: phase1_frac, phase2_frac, phase1_value, phase2_end_value, phase3_end_value
        OR the original callable if spec is a callable (backward compat mode)

    Example:
        normalize_schedule_spec(0.5)  # constant 0.5
        normalize_schedule_spec({'start': 1.0, 'end': 0.1})  # linear decay
        normalize_schedule_spec({'phase1_frac': 0.4, ...})  # explicit three-phase
    """
    if callable(spec):
        return spec

    if isinstance(spec, (int, float)):
        return {
            'phase1_frac': 0.0,
            'phase2_frac': 0.0,
            'phase1_value': float(spec),
            'phase2_end_value': float(spec),
            'phase3_end_value': float(spec),
        }

    if isinstance(spec, dict):
        if 'start' in spec and 'end' in spec:
            return {
                'phase1_frac': 0.0,
                'phase2_frac': 1.0,
                'phase1_value': float(spec['start']),
                'phase2_end_value': float(spec['end']),
                'phase3_end_value': float(spec['end']),
            }
        if 'phase1_value' in spec:
            return {
                'phase1_frac': float(spec.get('phase1_frac', 0.4)),
                'phase2_frac': float(spec.get('phase2_frac', 0.75)),
                'phase1_value': float(spec['phase1_value']),
                'phase2_end_value': float(spec.get('phase2_end_value', spec['phase1_value'])),
                'phase3_end_value': float(spec.get('phase3_end_value', spec['phase1_value'])),
            }

    raise ValueError(f"Invalid schedule spec: {spec}. Expected float, callable, or dict with 'start'/'end' or 'phase1_value'/etc.")


def init_schedule_params(schedule_specs: dict[str, any]) -> dict[str, jnp.ndarray]:
    """Initialize schedule parameters for hyperopt mode.

    Args:
        schedule_specs: Dict mapping schedule names to specs (float, dict, or callable).
                       Callables are skipped (use standard optax mode).

    Returns:
        Dict mapping param paths to JAX arrays for the params tree.

    Example:
        init_schedule_params({
            'lambda_l0': {'phase1_value': 0.0, 'phase3_end_value': 0.01},
            'tu_temperature': {'start': 1.0, 'end': 0.02},
            'lambda_spread': 0.001,  # constant
        })
    """
    result = {}
    for name, spec in schedule_specs.items():
        normalized = normalize_schedule_spec(spec)
        if callable(normalized):
            continue
        for key, value in normalized.items():
            result[f"{HYPEROPT_SCHEDULE_NAMESPACE}/{name}_{key}"] = jnp.array(value, dtype=jnp.float32)
    return result


def _get_schedule_value(params, step, total_steps, schedule_name, schedule_or_value, schedule_ns=None):
    """Get schedule value, supporting both optax schedules and dynamic JAX-native mode.

    Args:
        params: ParameterTree with schedule params (if schedule_ns is provided)
        step: Current optimization step
        total_steps: Total steps (for JAX schedule computation)
        schedule_name: Name of the schedule (e.g., 'lambda_l0', 'tu_temperature')
        schedule_or_value: Fallback optax schedule or constant value
        schedule_ns: Namespace path for dynamic schedule params. If provided, reads
            schedule params from params[f"{schedule_ns}/{schedule_name}_*"] and uses
            jax_three_phase_schedule. If None, uses as_schedule(schedule_or_value).

    Returns:
        Scalar JAX array with the schedule value at the current step
    """
    if schedule_ns is None:
        return as_schedule(schedule_or_value)(step)

    prefix = f"{schedule_ns}/{schedule_name}"
    if f"{prefix}_phase1_value" not in params:
        return as_schedule(schedule_or_value)(step)

    return jax_three_phase_schedule(
        step,
        total_steps,
        params[f"{prefix}_phase1_frac"],
        params[f"{prefix}_phase2_frac"],
        params[f"{prefix}_phase1_value"],
        params[f"{prefix}_phase2_end_value"],
        params[f"{prefix}_phase3_end_value"],
    )


def _make_loss_func(
    stack,
    dconf,
    dmanager,
    num_z,
    ratio_paths,
    lambda_tucount,
    compute_losses_fn,
    lambda_spread=0.01,
    max_ratio=100.0,
    max_tus_per_cotx=5,
    max_prediction=1e6,
    lambda_l0=0.0,
    tu_temperature=0.5,
    tu_n_samples=4,
    lambda_coupling=0.1,
    min_ratio_threshold=0.005,
    lambda_ern_tying=0.0,
    hyperopt_schedule_ns=None,
    hyperopt_total_steps=None,
):
    """Create the loss function with optional TU sample averaging for variance reduction.

    Args:
        tu_n_samples: Number of TU mask samples to average over (default 4).
            Higher values reduce variance but increase compute cost.
        lambda_coupling: Weight for ratio-mask coupling penalty. When a ratio is below
            min_ratio_threshold, this creates gradient pressure to push down tu_log_alpha.
        min_ratio_threshold: Coupling only activates when normalized ratio < this.
            Set to 0 to disable coupling entirely.
        lambda_ern_tying: Weight for ERN TU tying penalty. When an ERN's positive input
            (mRNA target) is disabled, push the negative input (ERN protein) to also be
            disabled. Set to 0 to disable (default).
        hyperopt_schedule_ns: If provided, read schedule params from this namespace in
            the params tree and use jax_three_phase_schedule for recompilation-free hyperopt.
            Expected params: {ns}/{sched}_phase1_frac, _phase2_frac, _phase1_value, etc.
        hyperopt_total_steps: Total steps for JAX schedule computation (required if hyperopt_schedule_ns is set).
    """
    if hyperopt_schedule_ns and not hyperopt_total_steps:
        raise ValueError("hyperopt_total_steps required when hyperopt_schedule_ns is set")

    # config-time validation (Python assertions) - catches config bugs early
    if hyperopt_schedule_ns is None:
        _validate_temperature_schedule(tu_temperature, total_steps=100000)

    n_targets, n_networks = dmanager.n_targets, len(dmanager.networks)
    dep_mask = stack.get_dependent_output_mask()
    nb_dep = int(np.sum(dep_mask))
    ratio_paths = ratio_paths or []

    # per-network TU mask: only penalize TUs each network actually uses
    per_network_tu_mask = None
    if dmanager.enable_tu_masking and hasattr(stack, "get_per_network_tu_mask"):
        per_network_tu_mask = stack.get_per_network_tu_mask()
        logger.debug(f"Per-network TU mask shape: {per_network_tu_mask.shape}")

    ern_namespaces = [
        layer.namespace
        for layer in (stack.layers or [])
        if layer.f_type and layer.f_type.startswith("sequestron_ERN")
    ]

    # Debug: dump axis assignment for each target-network pair
    # This documents how X columns map to network inputs, crucial for visualization
    if is_design_debug_enabled():
        # Lazy import to avoid circular dependency
        from .design import get_design_debug_output_dir

        axis_assignments = []
        for tid, target in enumerate(dmanager.targets):
            target_name = getattr(target, "name", f"target_{tid}")
            target_input_names = getattr(target, "input_names", None)
            for net_idx, network in enumerate(dmanager.networks):
                network_name = getattr(network, "name", f"network_{net_idx}")
                try:
                    network_input_proteins = network.get_inverted_input_proteins()
                except Exception:
                    network_input_proteins = None
                axis_assignments.append(
                    {
                        "target_id": tid,
                        "target_name": target_name,
                        "target_input_names": target_input_names,  # alphabetical order = X columns
                        "network_id": net_idx,
                        "network_name": network_name,
                        "network_input_proteins": network_input_proteins,
                        # During optimization: X[:,0] -> network input slot 0 (positional)
                        # target_input_names[0] = what X[:,0] represents (e.g., 'eBFP2')
                        # network_input_proteins[0] = what network slot 0 is called (may differ)
                    }
                )
        save_debug_state(
            "axis_assignment_mapping",
            {"assignments": axis_assignments},
            {
                "n_targets": n_targets,
                "n_networks": n_networks,
                "note": "X columns are in alphabetical order of target.input_names. "
                "During optimization, X[:,i] goes to network input slot i positionally.",
            },
            output_dir=get_design_debug_output_dir(),
            mode="design",
        )

    def single_forward_pass(params, X, Z, key, tu_uniform):
        """Single forward pass with specific TU mask."""
        keys = jax.random.split(key, (X.shape[0], X.shape[1]))
        yhat, (apply_aux, full_output) = per_target_apply(
            params, X, Z, keys, stack, tu_uniform=tu_uniform
        )
        yhatdep = jnp.compress(dep_mask, yhat, axis=-1, size=nb_dep)
        yhatdep = _sanitize(yhatdep)
        yhatdep = jnp.clip(yhatdep, -max_prediction, max_prediction)
        return yhatdep, apply_aux

    total_steps = hyperopt_total_steps or 100000
    schedule_ns = hyperopt_schedule_ns

    def loss_func(dynamic, static, X, Y, Z, key, step):
        params = ParameterTree.merge(dynamic, static)
        mask_key, forward_key = jax.random.split(key)

        ratio_leaves = params.get_leaves_by_path(ratio_paths)

        # regularization penalties (independent of TU samples)
        tucount_w = _get_schedule_value(params, step, total_steps, "lambda_tucount", lambda_tucount, schedule_ns)
        tucount_penalty = tucount_w * sum(
            get_tucount_penalty_for_leaf(p, max_tus=max_tus_per_cotx) for p in ratio_leaves
        )
        tucount_penalty = _sanitize(jnp.atleast_1d(tucount_penalty))[0]
        spread_w = _get_schedule_value(params, step, total_steps, "lambda_spread", lambda_spread, schedule_ns)
        spread_penalty = spread_w * sum(
            get_spread_penalty_for_leaf(p, max_ratio=max_ratio) for p in ratio_leaves
        )
        spread_penalty = _sanitize(jnp.atleast_1d(spread_penalty))[0]

        tu_temp = _get_schedule_value(params, step, total_steps, "tu_temperature", tu_temperature, schedule_ns)
        l0_penalty = jnp.array(0.0)
        l0_penalty_per_network = None  # will be (n_targets, n_networks) if TU masking enabled
        coupling_penalty = jnp.array(0.0)
        coupling_penalty_per_target = None  # will be (n_targets,) if coupling enabled
        if TU_LOG_ALPHA_PATH in params:
            log_alpha = params[TU_LOG_ALPHA_PATH]
            # defensive: validate log_alpha shape
            assert log_alpha.ndim == 3, (
                f"log_alpha must be 3D (n_targets, n_networks, n_tus), got {log_alpha.ndim}D"
            )
            assert log_alpha.shape[0] == n_targets, (
                f"log_alpha n_targets mismatch: {log_alpha.shape[0]} vs {n_targets}"
            )
            assert log_alpha.shape[1] == n_networks, (
                f"log_alpha n_networks mismatch: {log_alpha.shape[1]} vs {n_networks}"
            )
            # NOTE: can't assert jnp.isfinite(log_alpha) here - it's a traced value
            # Use _sanitize() for NaN/Inf handling and checkify in tests for validation
            log_alpha = jnp.nan_to_num(log_alpha, nan=0.0, posinf=10.0, neginf=-10.0)

            # per-network L0: only penalize TUs each network actually uses
            # log_alpha shape: (n_targets, n_networks, n_tus)
            # per_network_tu_mask shape: (n_networks, n_tus)
            from biocomp.tumasking import l0_penalty as l0_penalty_fn

            per_tu_penalty = l0_penalty_fn(log_alpha, temperature=tu_temp)
            if per_network_tu_mask is not None:
                # defensive: validate mask shape
                assert per_network_tu_mask.shape[0] == n_networks, (
                    f"per_network_tu_mask shape mismatch: {per_network_tu_mask.shape} vs n_networks={n_networks}"
                )
                # mask zeros out unused TUs before summing
                per_tu_penalty = per_tu_penalty * per_network_tu_mask[None, :, :]
            # per-network L0 breakdown: sum over TUs, shape (n_targets, n_networks)
            l0_weight = _get_schedule_value(params, step, total_steps, "lambda_l0", lambda_l0, schedule_ns)
            l0_penalty_per_network = _sanitize(l0_weight * jnp.sum(per_tu_penalty, axis=-1))
            l0_penalty = _sanitize(jnp.atleast_1d(jnp.sum(l0_penalty_per_network)))[0]

            if min_ratio_threshold > 0 and ratio_paths:
                coupling_weight = _get_schedule_value(params, step, total_steps, "lambda_coupling", lambda_coupling, schedule_ns)
                raw_coupling, raw_coupling_per_target = ratio_mask_coupling_penalty(
                    params, ratio_paths, log_alpha, min_ratio_threshold, return_per_target=True
                )
                coupling_penalty_per_target = _sanitize(coupling_weight * raw_coupling_per_target)
                coupling_penalty = _sanitize(jnp.atleast_1d(coupling_weight * raw_coupling))[0]

            if ern_namespaces:
                tying_weight = _get_schedule_value(params, step, total_steps, "lambda_ern_tying", lambda_ern_tying, schedule_ns)
                raw_tying = ern_tu_tying_penalty(params, ern_namespaces, log_alpha)
                ern_tying_penalty_val = tying_weight * raw_tying
                ern_tying_penalty_val = _sanitize(jnp.atleast_1d(ern_tying_penalty_val))[0]
            else:
                ern_tying_penalty_val = jnp.array(0.0)
        else:
            ern_tying_penalty_val = jnp.array(0.0)

        # TU sample averaging
        extra_aux_inner = {}
        if tu_n_samples > 1 and TU_LOG_ALPHA_PATH in params:
            tu_uniforms = _sample_tu_uniform(params, mask_key, n_samples=tu_n_samples)
            forward_keys = jax.random.split(forward_key, tu_n_samples)

            def forward_with_tu(tu_u, fwd_key):
                yhatdep, _ = single_forward_pass(params, X, Z, fwd_key, tu_u)
                losses, inner_aux = compute_losses_fn(X, Y, yhatdep, step, n_targets, n_networks)
                return losses, yhatdep, inner_aux

            all_losses_stack, yhatdep_stack, inner_aux_stack = vmap(forward_with_tu)(
                tu_uniforms, forward_keys
            )
            # average losses over TU samples
            all_losses = jnp.mean(all_losses_stack, axis=0)
            # use last sample's yhatdep and aux for logging
            yhatdep = yhatdep_stack[-1]
            tu_uniform = tu_uniforms[-1]
            apply_aux = None
            # extract last sample's inner aux (sublosses etc.)
            if inner_aux_stack:
                extra_aux_inner = jax.tree.map(lambda x: x[-1], inner_aux_stack)
        else:
            tu_uniform = _sample_tu_uniform(params, mask_key, n_samples=1)
            yhatdep, apply_aux = single_forward_pass(params, X, Z, forward_key, tu_uniform)
            all_losses, extra_aux_inner = compute_losses_fn(
                X, Y, yhatdep, step, n_targets, n_networks
            )

        all_losses = _sanitize(all_losses)

        # compute TU statistics if masking is enabled
        tu_stats = {}
        if TU_LOG_ALPHA_PATH in params:
            log_alpha = params[TU_LOG_ALPHA_PATH]
            tu_probs = jax.nn.sigmoid(log_alpha)
            tu_enabled_mask = tu_probs > 0.5
            tu_stats = {
                # Aggregated statistics (backward compatibility)
                "enabled_count": jnp.sum(tu_enabled_mask),
                "total_count": jnp.array(log_alpha.size),
                "mean_prob": jnp.mean(tu_probs),
                "min_log_alpha": jnp.min(log_alpha),
                "max_log_alpha": jnp.max(log_alpha),
                "log_alpha_std": jnp.std(log_alpha),
                # Per-network breakdown: shape (n_targets, n_networks)
                # log_alpha shape is (n_targets, n_networks, n_tus)
                "enabled_count_per_network": jnp.sum(tu_enabled_mask, axis=-1),
                "mean_prob_per_network": jnp.mean(tu_probs, axis=-1),
                "min_log_alpha_per_network": jnp.min(log_alpha, axis=-1),
                "max_log_alpha_per_network": jnp.max(log_alpha, axis=-1),
                "std_log_alpha_per_network": jnp.std(log_alpha, axis=-1),
            }

        # compute ratio statistics
        ratio_stats = {}
        if ratio_leaves:
            all_ratios = []
            for p in ratio_leaves:
                if hasattr(p, "view"):
                    try:
                        all_ratios.append(jnp.abs(p.view()).ravel())
                    except Exception:
                        pass
                elif hasattr(p, "shape"):
                    all_ratios.append(jnp.abs(p).ravel())
            if all_ratios:
                ratios_flat = jnp.concatenate(all_ratios)
                ratio_stats = {
                    "min": jnp.min(ratios_flat),
                    "max": jnp.max(ratios_flat),
                    "mean": jnp.mean(ratios_flat),
                    "std": jnp.std(ratios_flat),
                    "nonzero_count": jnp.sum(ratios_flat > 1e-6),
                    "total_count": jnp.array(ratios_flat.size),
                }

        # extract sublosses from inner aux if available
        sublosses = extra_aux_inner.get("sublosses", {}) if extra_aux_inner else {}

        # compute per-network prediction statistics
        # yhatdep shape: (batch_size, n_targets, n_networks)
        pred_stats_per_network = {
            "mean": jnp.mean(yhatdep, axis=0),  # (n_targets, n_networks)
            "std": jnp.std(yhatdep, axis=0),  # (n_targets, n_networks)
            "min": jnp.min(yhatdep, axis=0),  # (n_targets, n_networks)
            "max": jnp.max(yhatdep, axis=0),  # (n_targets, n_networks)
        }

        aux = {
            "apply_aux": apply_aux,
            "all_losses": all_losses,
            "yhatdep": yhatdep,
            "X": X,  # input coordinates for diagnostic plots
            "Y": Y,  # target values for diagnostic plots
            # Scalar penalties (backward compatibility)
            "l0_penalty": l0_penalty,
            "coupling_penalty": coupling_penalty,
            "ern_tying_penalty": ern_tying_penalty_val,
            "tucount_penalty": tucount_penalty,
            "spread_penalty": spread_penalty,
            # Per-network/target penalty breakdowns
            "l0_penalty_per_network": l0_penalty_per_network,  # (n_targets, n_networks) or None
            "coupling_penalty_per_target": coupling_penalty_per_target,  # (n_targets,) or None
            # Other aux data
            "tu_uniform": tu_uniform,
            "tu_stats": tu_stats,  # includes *_per_network keys
            "ratio_stats": ratio_stats,
            "tu_temperature": tu_temp,
            "sublosses": sublosses,  # includes *_per_network keys
            "pred_stats_per_network": pred_stats_per_network,
        }

        loss = (
            all_losses.mean()
            + tucount_penalty
            + spread_penalty
            + l0_penalty
            + coupling_penalty
            + ern_tying_penalty_val
        )
        # NOTE: can't assert jnp.isfinite(loss) - it's a traced value
        # Use _sanitize for handling and checkify in tests for validation
        loss = jnp.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=1e6)
        return loss, aux

    return loss_func


def distance_loss(
    stack,
    dconf,
    dmanager,
    num_z,
    ratio_paths=None,
    epsilon=0.01,
    lambda_tucount=0.0,
    max_tus_per_cotx=5,
    lambda_spread=0.01,
    max_ratio=100.0,
    lambda_l0=0.0,
    tu_temperature=0.5,
    tu_n_samples=4,
    lambda_coupling=0.1,
    min_ratio_threshold=0.005,
    lambda_ern_tying=0.0,
    distance_func=huber_zncc_loss,
    hyperopt_schedule_ns=None,
    hyperopt_total_steps=None,
):
    def compute_losses(X, Y, yhatdep, step, n_targets, n_networks):
        yhatdep = _sanitize(yhatdep)
        all_losses = compute_all_losses(
            X, Y, yhatdep, Partial(distance_func, epsilon=as_schedule(epsilon)(step))
        )
        assert_that(all_losses).has_shape((n_targets, n_networks))
        return _sanitize(all_losses), {}

    return _make_loss_func(
        stack,
        dconf,
        dmanager,
        num_z,
        ratio_paths,
        lambda_tucount,
        compute_losses,
        lambda_spread=lambda_spread,
        max_ratio=max_ratio,
        max_tus_per_cotx=max_tus_per_cotx,
        lambda_l0=lambda_l0,
        tu_temperature=tu_temperature,
        tu_n_samples=tu_n_samples,
        lambda_coupling=lambda_coupling,
        min_ratio_threshold=min_ratio_threshold,
        lambda_ern_tying=lambda_ern_tying,
        hyperopt_schedule_ns=hyperopt_schedule_ns,
        hyperopt_total_steps=hyperopt_total_steps,
    )


def grid_distance_loss(
    stack,
    dconf,
    dmanager,
    num_z,
    ratio_paths=None,
    w_sinkhorn=1.0,
    w_lncc=0.5,
    w_mse=0.0,
    w_spectral=0.0,
    eps_sinkhorn=0.1,
    n_sinkhorn_iters=50,
    lncc_kernel=7,
    lambda_tucount=0.0,
    max_tus_per_cotx=5,
    lambda_spread=0.01,
    max_ratio=100.0,
    lambda_l0=0.0,
    tu_temperature=0.5,
    tu_n_samples=4,
    lambda_coupling=0.1,
    min_ratio_threshold=0.005,
    lambda_ern_tying=0.0,
    hyperopt_schedule_ns=None,
    hyperopt_total_steps=None,
    **kw,
):
    assert dmanager.is_lattice_mode, "grid_distance_loss requires lattice sampling"
    xres, yres = dmanager.grid_resolution
    n_networks = len(dmanager.networks)

    def compute_grid_loss_single_with_breakdown(y_img, yhat_img):
        """Compute loss with individual component breakdown for aux data."""
        y_img, yhat_img = _sanitize(y_img), _sanitize(yhat_img)

        # compute individual losses (unweighted)
        sinkhorn_l = (
            sinkhorn_divergence_conv(
                proj_nonneg_ste(yhat_img),
                proj_nonneg_ste(y_img),
                eps_sinkhorn,
                n_iters=n_sinkhorn_iters,
            )
            if w_sinkhorn > 0
            else jnp.array(0.0)
        )

        lncc_l = (
            lncc_grid_loss(None, y_img, yhat_img, k=lncc_kernel) if w_lncc > 0 else jnp.array(0.0)
        )
        mse_l = jnp.mean((y_img - yhat_img) ** 2) if w_mse > 0 else jnp.array(0.0)
        spectral_l = spectral_loss(None, y_img, yhat_img) if w_spectral > 0 else jnp.array(0.0)

        # weighted total
        total = w_sinkhorn * sinkhorn_l + w_lncc * lncc_l + w_mse * mse_l + w_spectral * spectral_l
        return total, (sinkhorn_l, lncc_l, mse_l, spectral_l)

    def compute_losses(X, Y, yhatdep, step, n_targets, n_networks_):
        yhatdep = _sanitize(yhatdep)
        # Y shape: (batch_size, n_targets, 1) - validate before squeeze
        assert Y.ndim == 3 and Y.shape[-1] == 1, (
            f"grid_distance_loss expects Y shape (batch_size, n_targets, 1), got {Y.shape}. "
            "The last dim must be 1 for proper squeeze+reshape to grid."
        )
        batch_size = Y.shape[0]
        assert batch_size == xres * yres, (
            f"batch_size={batch_size} must equal xres*yres={xres * yres} for grid reshape"
        )
        Y_images = jnp.tile(
            Y.squeeze(-1).T.reshape(n_targets, 1, yres, xres), (1, n_networks, 1, 1)
        )
        yhat_images = yhatdep.transpose(1, 2, 0).reshape(n_targets, n_networks, yres, xres)

        all_losses, (sinkhorn_losses, lncc_losses, mse_losses, spectral_losses) = vmap(
            vmap(compute_grid_loss_single_with_breakdown)
        )(Y_images, yhat_images)

        sublosses = {
            # Aggregated metrics (backward compatibility)
            "sinkhorn": _sanitize(jnp.mean(sinkhorn_losses)),
            "lncc": _sanitize(jnp.mean(lncc_losses)),
            "mse": _sanitize(jnp.mean(mse_losses)),
            "spectral": _sanitize(jnp.mean(spectral_losses)),
            "sinkhorn_weighted": _sanitize(w_sinkhorn * jnp.mean(sinkhorn_losses)),
            "lncc_weighted": _sanitize(w_lncc * jnp.mean(lncc_losses)),
            "mse_weighted": _sanitize(w_mse * jnp.mean(mse_losses)),
            "spectral_weighted": _sanitize(w_spectral * jnp.mean(spectral_losses)),
            # Per-network breakdown: shape (n_targets, n_networks)
            "sinkhorn_per_network": _sanitize(sinkhorn_losses),
            "lncc_per_network": _sanitize(lncc_losses),
            "mse_per_network": _sanitize(mse_losses),
            "spectral_per_network": _sanitize(spectral_losses),
        }
        return _sanitize(all_losses), {"yhat_images": yhat_images, "sublosses": sublosses}

    return _make_loss_func(
        stack,
        dconf,
        dmanager,
        num_z,
        ratio_paths,
        lambda_tucount,
        compute_losses,
        lambda_spread=lambda_spread,
        max_ratio=max_ratio,
        max_tus_per_cotx=max_tus_per_cotx,
        lambda_l0=lambda_l0,
        tu_temperature=tu_temperature,
        tu_n_samples=tu_n_samples,
        lambda_coupling=lambda_coupling,
        min_ratio_threshold=min_ratio_threshold,
        lambda_ern_tying=lambda_ern_tying,
        hyperopt_schedule_ns=hyperopt_schedule_ns,
        hyperopt_total_steps=hyperopt_total_steps,
    )
