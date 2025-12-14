"""Loss functions for circuit design optimization.

This module contains:
- Fast grid-based Sinkhorn divergence using Gaussian convolutions
- Various distance/similarity loss functions (ZNCC, LNCC, Huber, MSE, etc.)
- The main distance_loss and grid_distance_loss factories
- L0 penalty for TU masking (Hard Concrete distribution)
"""

import jax
import jax.numpy as jnp
from jax import vmap, lax
from jax.tree_util import Partial

import numpy as np
from assertpy import assert_that

from .parameters import ParameterTree
from .optimutils import as_schedule
from .hard_concrete import l0_loss, TU_LOG_ALPHA_PATH


# Common utilities
def _sanitize(x):
    """Replace NaN/inf with 0."""
    return jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_coords(x):
    """Standardize coordinates to zero mean, unit std."""
    mu, sd = jnp.mean(x, 0, keepdims=True), jnp.std(x, 0, keepdims=True) + 1e-8
    return (x - mu) / sd


## {{{                    --     fast OT on a grid     --


def _gauss1d(sigma, radius=5):
    x = jnp.arange(-radius, radius + 1, dtype=jnp.float32)
    k = jnp.exp(-(x**2) / (2 * sigma**2))
    return k / jnp.sum(k)


def _gauss_blur2d(x, kernel):
    """Apply separable Gaussian blur via 1D convolutions."""

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


def sinkhorn_divergence_conv(a, b, eps, n_iters=80, uniform_mix=1e-9):
    """Grid-based Sinkhorn divergence using fast Gaussian convolutions.

    Numerical stability via mixture kernel: blends Gaussian with uniform distribution
    to ensure non-zero support everywhere. This prevents gradient explosion when
    predicted and target distributions have disjoint support.

    Args:
        a, b: Input distributions (2D arrays)
        eps: Entropic regularization (controls blur bandwidth)
        n_iters: Number of Sinkhorn iterations
        uniform_mix: Weight for uniform kernel (prevents support collapse)
    """
    # Enforce strict positivity on inputs
    a = jnp.maximum(_sanitize(a.astype(jnp.float32)), 1e-24)
    b = jnp.maximum(_sanitize(b.astype(jnp.float32)), 1e-24)

    # Normalize to probability distributions
    a = a / a.sum()
    b = b / b.sum()

    sigma = jnp.sqrt(eps / 2.0)
    kernel = _gauss1d(sigma)

    def blurred_with_floor(x):
        """Apply Gaussian blur mixed with uniform distribution (teleportation floor)."""
        blurred = _gauss_blur2d(x, kernel)
        # Uniform component: global mass spread evenly
        uniform = x.sum() / x.size
        return (1 - uniform_mix) * blurred + uniform_mix * uniform

    def sinkhorn_iters(m1, m2, n):
        u, v = jnp.ones_like(m1), jnp.ones_like(m2)

        def step(carry, _):
            u, v = carry
            # Division is now safe: blurred_with_floor always has positive floor
            u_new = m1 / blurred_with_floor(v)
            v_new = m2 / blurred_with_floor(u_new)
            return (u_new, v_new), None

        (u, v), _ = lax.scan(step, (u, v), None, length=n)
        return u, v

    def ot_cost(m1, m2, n):
        u, v = sinkhorn_iters(m1, m2, n)
        # Safe log: inputs guaranteed positive by blurred_with_floor
        cost = eps * (
            jnp.sum(m1 * jnp.log(jnp.maximum(u, 1e-24)))
            + jnp.sum(m2 * jnp.log(jnp.maximum(v, 1e-24)))
        )
        return cost

    ot_ab = ot_cost(a, b, n_iters)
    ot_aa = ot_cost(a, a, max(20, n_iters // 2))
    ot_bb = ot_cost(b, b, max(20, n_iters // 2))
    return jnp.maximum(0.0, ot_ab - 0.5 * (ot_aa + ot_bb))


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                      --     loss functions     --


def proj_nonneg_ste(z, leak=1e-3, cap=None):
    """Project to nonnegative with straight-through estimator for gradients."""
    z_clip = jnp.clip(z, 0.0, cap) if cap is not None else jnp.maximum(z, 0.0)
    z_leaky = jnp.where(z >= 0.0, z, leak * z)
    return z_clip + jax.lax.stop_gradient(z_leaky - z_clip)


def _epsilon_from_x_median(xn, eps=1e-12):
    B = xn.shape[0]
    d2 = jnp.sum((xn[:, None, :] - xn[None, :, :]) ** 2, axis=-1)
    nn_sq = jnp.min(d2 + jnp.eye(B) * 1e9, axis=1)
    return jax.lax.stop_gradient(jnp.maximum(0.5 * jnp.median(nn_sq), eps))


def _ott_sinkhorn_div(xn, a, b, epsilon, tau=None, **solver_kw):
    """Core OTT Sinkhorn divergence computation."""
    from ott.geometry import pointcloud
    from ott.problems.linear import linear_problem
    from ott.solvers.linear import sinkhorn

    geom = pointcloud.PointCloud(xn, xn, epsilon=epsilon)
    solver = sinkhorn.Sinkhorn(lse_mode=True, **solver_kw)

    def ot(u, v):
        kw = {"tau_a": tau, "tau_b": tau} if tau else {}
        return solver(linear_problem.LinearProblem(geom, a=u, b=v, **kw)).reg_ot_cost

    return ot(a, b) - 0.5 * (ot(a, a) + ot(b, b))


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
    # Add floor and balance masses
    floor = (mass_floor * jnp.maximum(jnp.sum(b), 1.0)) / a.size
    a, b = a + floor, b + floor
    a = a * (jnp.sum(b) / (jnp.sum(a) + 1e-8))
    div = _ott_sinkhorn_div(xn, a, b, eps)
    return jnp.where(jnp.isfinite(div), div, 0.0) + lambda_neg * jnp.mean(jax.nn.relu(-yhat))


def zncc_loss(x, y, yhat, eps=1e-6, **kw):
    """Zero-mean normalized cross-correlation loss.

    Stability: epsilon added inside sqrt (Tikhonov regularization) to prevent
    gradient explosion when variance approaches zero. This keeps gradients
    smooth and finite even for flat outputs.
    """
    y, yhat = _sanitize(y), _sanitize(yhat)
    y0, yhat0 = y - jnp.mean(y), yhat - jnp.mean(yhat)
    cov = jnp.mean(y0 * yhat0)
    var_y = jnp.mean(y0**2)
    var_yhat = jnp.mean(yhat0**2)
    # Epsilon INSIDE sqrt: prevents gradient of sqrt from exploding at 0
    std_product = jnp.sqrt((var_y + eps) * (var_yhat + eps))
    return 1.0 - cov / std_product


def huber_loss(x, y, yhat, delta=0.01, **kw):
    """Huber loss - smooth transition between L2 and L1."""
    y, yhat = _sanitize(y), _sanitize(yhat)
    r = jnp.abs(yhat - y)
    return jnp.mean(jnp.where(r <= delta, 0.5 * r**2, delta * (r - 0.5 * delta)))


def huber_zncc_loss(x, y, yhat, delta=0.01, zncc_weight=0.1, **kw):
    """Combined Huber + ZNCC loss for robust shape matching."""
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
    """Local normalized cross-correlation loss using Gaussian kernel weighting."""
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
    """Local NCC on 2D grid using box filter (integral image).

    Stability: epsilon added inside sqrt (Tikhonov regularization) to prevent
    gradient explosion when local variance approaches zero.
    """
    y = _sanitize(y)
    yhat = _sanitize(yhat)

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
    var_y = box2d(y0c**2)
    var_yhat = box2d(y1c**2)
    cov = box2d(y0c * y1c)
    # Epsilon INSIDE sqrt: prevents gradient explosion at zero variance
    std_product = jnp.sqrt((var_y + eps) * (var_yhat + eps))
    lncc = jnp.clip(cov / std_product, -1, 1)
    return 1.0 - jnp.mean(lncc)


def simse_lncc_loss(x, y, yhat, simse_weight=0.3, **kw):
    return simse_weight * simse_loss(x, y, yhat) + (1 - simse_weight) * lncc_loss(x, y, yhat)


## {{{                      --     penalty functions     --


def soft_count_over_one_penalty(W, rel_active=1e-3, width=2e-4):
    """Penalty for having more than one active ratio per aggregation row."""
    A = jnp.abs(W)
    m = jnp.max(A, axis=1, keepdims=True)
    norm = jnp.where(m > 0, A / (m + 1e-12), 0.0)
    soft_count = jnp.sum(jax.nn.sigmoid((norm - rel_active) / (width + 1e-12)), axis=1)
    return jnp.sum(jnp.square(jax.nn.relu(soft_count - 1.0)))


def get_over1_penalty_for_leaf(p, rel_active=1e-3, width=2e-4):
    """Get over1 penalty for a ratio leaf. Handle ArrayRef with heterogeneous shapes."""
    if hasattr(p, "view"):
        try:
            return soft_count_over_one_penalty(p.view(), rel_active=rel_active, width=width)
        except Exception:
            return sum(
                soft_count_over_one_penalty(p.tree[path], rel_active=rel_active, width=width)
                for path in p.paths
            )
    return soft_count_over_one_penalty(p, rel_active=rel_active, width=width)


def ratio_spread_penalty(W, max_ratio=100.0, eps=1e-9):
    """Penalty for ratio spread exceeding max_ratio.

    Encourages ratios to stay within a reasonable range (e.g., 1:100).
    This prevents designs with extreme ratio spreads like 1:300 that are
    biologically implausible.

    Args:
        W: Ratio array shape (n_aggregations, n_ratios)
        max_ratio: Maximum allowed ratio between largest and smallest non-zero
        eps: Small constant for numerical stability
    """
    A = jnp.abs(W)
    # Per-row: find spread of active ratios in log-space
    log_max_ratio = jnp.log(max_ratio + eps)

    # For each row, compute log-spread of positive ratios
    # Use safe log with masking for zero values
    pos_mask = A > eps
    log_A = jnp.where(pos_mask, jnp.log(A + eps), -jnp.inf)

    # Per-row max and min of log ratios (only among positive values)
    log_max = jnp.max(jnp.where(pos_mask, log_A, -jnp.inf), axis=1)
    log_min = jnp.min(jnp.where(pos_mask, log_A, jnp.inf), axis=1)
    log_spread = log_max - log_min

    # Penalize spread exceeding log(max_ratio)
    excess = jax.nn.relu(log_spread - log_max_ratio)
    return jnp.sum(jnp.square(excess))


def get_spread_penalty_for_leaf(p, max_ratio=100.0):
    """Get ratio spread penalty for a ratio leaf."""
    if hasattr(p, "view"):
        try:
            return ratio_spread_penalty(p.view(), max_ratio=max_ratio)
        except Exception:
            return sum(ratio_spread_penalty(p.tree[path], max_ratio=max_ratio) for path in p.paths)
    return ratio_spread_penalty(p, max_ratio=max_ratio)


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                      --     apply helpers     --


def per_batch_apply(params, X, Z, keys, stack, tu_uniform=None):
    """Apply stack to a batch of samples.

    Args:
        tu_uniform: Uniform samples for TU masking, shape (n_tus,) or None
    """

    def apply_single(x, z, key):
        return stack.apply(params, x, z, key, tu_enabled_random_vars=tu_uniform)

    return vmap(apply_single)(X, Z, keys)


def per_target_apply(params, X, Z, keys, stack, tu_uniform=None):
    """Apply stack across targets.

    Args:
        tu_uniform: Uniform samples for TU masking, shape (n_targets, n_tus) or None
    """

    def apply_target(p, x, z, k, tu_u):
        return per_batch_apply(p, x, z, k, stack, tu_uniform=tu_u)

    # tu_uniform has shape (n_targets, n_tus), vmap over targets (axis 0)
    tu_uniform_axes = 0 if tu_uniform is not None else None
    return vmap(apply_target, in_axes=(0, 1, 1, 1, tu_uniform_axes), out_axes=1)(
        params, X, Z, keys, tu_uniform
    )


@Partial(jax.jit, static_argnames=["stack"])
def per_replicate_apply(params, X, Z, keys, stack, tu_uniform=None):
    """Apply stack across replicates.

    Args:
        tu_uniform: Uniform samples for TU masking, shape (n_replicates, n_targets, n_tus) or None
    """

    def apply_rep(p, x, z, k, tu_u):
        return per_target_apply(p, x, z, k, stack, tu_uniform=tu_u)

    tu_uniform_axes = 0 if tu_uniform is not None else None
    return vmap(apply_rep, in_axes=(0, 0, 0, 0, tu_uniform_axes))(params, X, Z, keys, tu_uniform)


@Partial(jax.jit, static_argnames=["lossfunc", "n_inputs_per_network"])
def compute_all_losses(x, y, yhatdep, lossfunc, n_inputs_per_network=2):
    """Compute per-target, per-network losses."""
    n_networks = int(x.shape[-1] / n_inputs_per_network)
    batch_size, n_targets = y.shape[0], y.shape[1]

    assert_that(x).has_shape((batch_size, n_targets, n_networks * n_inputs_per_network))
    assert_that(yhatdep).has_shape((batch_size, n_targets, n_networks))
    assert_that(y).has_same_shape(yhatdep)

    xsplit = jnp.reshape(x, (batch_size, n_targets, n_networks, n_inputs_per_network))
    return vmap(vmap(lossfunc, in_axes=(1, 1, 1)), in_axes=(1, 1, 1))(xsplit, yhatdep, y)


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                      --     loss factories     --


def _sample_tu_uniform(params, key):
    """Sample uniform values for TU masking reparameterization.

    This should be called ONCE at the start of each forward pass.
    Returns uniform samples to pass to stack.apply as tu_enabled_random_vars.
    The actual Hard Concrete transformation happens inside each node's apply().

    Returns:
        tu_uniform_samples: Shape matching log_alpha, or None if TU masking disabled
    """
    if TU_LOG_ALPHA_PATH not in params:
        return None  # No TU masking configured

    log_alpha = params[TU_LOG_ALPHA_PATH]
    shape = log_alpha.shape  # (n_targets, n_tus) when vmapped per replicate

    # Sample uniform for reparameterization trick
    return jax.random.uniform(key, shape, minval=1e-6, maxval=1.0 - 1e-6)


def _make_loss_func(
    stack,
    dconf,
    dmanager,
    num_z,
    ratio_paths,
    lambda_over1,
    compute_losses_fn,
    lambda_spread=0.01,
    max_ratio=100.0,
    max_prediction=1e6,
    lambda_l0=0.0,
    tu_temperature=0.5,
):
    """Shared loss function factory logic.

    Numerical stability is achieved at the component level (Sinkhorn, ZNCC, L0 penalty)
    rather than via global NaN masking. Inputs are sanitized and outputs clamped.

    Args:
        lambda_spread: Weight for ratio spread penalty (encourages ratios to stay
                       within max_ratio:1 range). Default 0.01.
        max_ratio: Maximum allowed ratio spread (default 100:1)
        max_prediction: Maximum allowed prediction value before clamping. Default 1e6.
        lambda_l0: Weight for L0 penalty on TU masks (encourages sparsity). Default 0.0.
        tu_temperature: Temperature for Hard Concrete sampling. Default 0.5.
    """
    n_targets, n_networks = dmanager.n_targets, len(dmanager.networks)
    dep_mask = stack.get_dependent_output_mask()
    nb_dep = int(np.sum(dep_mask))
    ratio_paths = ratio_paths or []

    def loss_func(dynamic, static, X, Y, Z, key, step):
        params = ParameterTree.merge(dynamic, static)

        # Sample uniform values for TU masking (reparameterization trick)
        # The Hard Concrete transformation happens inside each node's apply()
        mask_key, forward_key = jax.random.split(key)
        tu_uniform = _sample_tu_uniform(params, mask_key)

        keys = jax.random.split(forward_key, (X.shape[0], X.shape[1]))
        yhat, (apply_aux, full_output) = per_target_apply(
            params, X, Z, keys, stack, tu_uniform=tu_uniform
        )
        yhatdep = jnp.compress(dep_mask, yhat, axis=-1, size=nb_dep)

        # Sanitize and clamp model output to prevent explosion
        yhatdep = _sanitize(yhatdep)
        yhatdep = jnp.clip(yhatdep, -max_prediction, max_prediction)

        ratio_leaves = params.get_leaves_by_path(ratio_paths)

        # Sparsity penalty (encourages one active TU per cotransfection)
        over1_penalty = as_schedule(lambda_over1)(step) * sum(
            get_over1_penalty_for_leaf(p) for p in ratio_leaves
        )
        over1_penalty = _sanitize(jnp.atleast_1d(over1_penalty))[0]

        # Spread penalty (encourages ratios to stay within max_ratio:1)
        spread_penalty = as_schedule(lambda_spread)(step) * sum(
            get_spread_penalty_for_leaf(p, max_ratio=max_ratio) for p in ratio_leaves
        )
        spread_penalty = _sanitize(jnp.atleast_1d(spread_penalty))[0]

        # L0 penalty for TU masking (encourages sparse TU selection)
        tu_temp = (
            as_schedule(tu_temperature)(step)
            if callable(tu_temperature) or isinstance(tu_temperature, dict)
            else tu_temperature
        )
        l0_penalty = jnp.array(0.0)
        if TU_LOG_ALPHA_PATH in params and lambda_l0 > 0:
            log_alpha = params[TU_LOG_ALPHA_PATH]
            l0_penalty = as_schedule(lambda_l0)(step) * l0_loss(log_alpha, temperature=tu_temp)
            l0_penalty = _sanitize(jnp.atleast_1d(l0_penalty))[0]

        all_losses, extra_aux = compute_losses_fn(X, Y, yhatdep, step, n_targets, n_networks)
        aux = {
            "apply_aux": apply_aux,
            "all_losses": all_losses,
            "yhatdep": yhatdep,
            "l0_penalty": l0_penalty,
            "tu_uniform": tu_uniform,
            **extra_aux,
        }

        # Loss components are individually stabilized - no global NaN mask needed
        loss = all_losses.mean() + over1_penalty + spread_penalty + l0_penalty
        return loss, aux

    return loss_func


def distance_loss(
    stack,
    dconf,
    dmanager,
    num_z,
    ratio_paths=None,
    epsilon=0.01,
    lambda_over1=0.001,
    lambda_spread=0.01,
    max_ratio=100.0,
    lambda_l0=0.0,
    tu_temperature=0.5,
    distance_func=huber_zncc_loss,
):
    """Factory for point-cloud distance loss.

    Args:
        lambda_spread: Weight for ratio spread penalty (default 0.01)
        max_ratio: Maximum allowed ratio spread (default 100:1)
        lambda_l0: Weight for L0 penalty on TU masks (default 0.0, disabled)
        tu_temperature: Temperature for Hard Concrete sampling (default 0.5)
    """

    def compute_losses(X, Y, yhatdep, step, n_targets, n_networks):
        # Sanitize predictions
        yhatdep = _sanitize(yhatdep)

        all_losses = compute_all_losses(
            X, Y, yhatdep, Partial(distance_func, epsilon=as_schedule(epsilon)(step))
        )
        assert_that(all_losses).has_shape((n_targets, n_networks))

        # Sanitize final losses
        all_losses = _sanitize(all_losses)
        return all_losses, {}

    return _make_loss_func(
        stack,
        dconf,
        dmanager,
        num_z,
        ratio_paths,
        lambda_over1,
        compute_losses,
        lambda_spread=lambda_spread,
        max_ratio=max_ratio,
        lambda_l0=lambda_l0,
        tu_temperature=tu_temperature,
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
    lambda_over1=0.001,
    lambda_spread=0.01,
    max_ratio=100.0,
    lambda_l0=0.0,
    tu_temperature=0.5,
    **kw,
):
    """Factory for grid-based distance loss using fast Sinkhorn.

    Args:
        w_sinkhorn: Weight for Sinkhorn divergence (distribution matching)
        w_lncc: Weight for local NCC (shape correlation)
        w_mse: Weight for MSE (absolute error - helps with NRE)
        w_spectral: Weight for spectral loss
        lambda_spread: Weight for ratio spread penalty (default 0.01)
        max_ratio: Maximum allowed ratio spread (default 100:1)
        lambda_l0: Weight for L0 penalty on TU masks (default 0.0, disabled)
        tu_temperature: Temperature for Hard Concrete sampling (default 0.5)
    """
    assert dmanager.is_lattice_mode, "grid_distance_loss requires lattice sampling"
    xres, yres = dmanager.grid_resolution
    n_networks = len(dmanager.networks)

    def compute_grid_loss_single(y_img, yhat_img):
        y_img = _sanitize(y_img)
        yhat_img = _sanitize(yhat_img)

        loss = jnp.array(0.0)
        if w_sinkhorn > 0:
            sink = sinkhorn_divergence_conv(
                proj_nonneg_ste(yhat_img),
                proj_nonneg_ste(y_img),
                eps_sinkhorn,
                n_iters=n_sinkhorn_iters,
            )
            loss = loss + w_sinkhorn * sink
        if w_lncc > 0:
            lncc = lncc_grid_loss(None, y_img, yhat_img, k=lncc_kernel)
            loss = loss + w_lncc * lncc
        if w_mse > 0:
            loss = loss + w_mse * jnp.mean((y_img - yhat_img) ** 2)
        if w_spectral > 0:
            loss = loss + w_spectral * spectral_loss(None, y_img, yhat_img)
        return loss

    def compute_losses(X, Y, yhatdep, step, n_targets, n_networks_):
        # Sanitize predictions before loss computation
        yhatdep = _sanitize(yhatdep)

        Y_images = jnp.tile(
            Y.squeeze(-1).T.reshape(n_targets, 1, yres, xres), (1, n_networks, 1, 1)
        )
        yhat_images = yhatdep.transpose(1, 2, 0).reshape(n_targets, n_networks, yres, xres)
        all_losses = vmap(vmap(compute_grid_loss_single))(Y_images, yhat_images)

        # Sanitize final losses
        all_losses = _sanitize(all_losses)
        return all_losses, {"yhat_images": yhat_images}

    return _make_loss_func(
        stack,
        dconf,
        dmanager,
        num_z,
        ratio_paths,
        lambda_over1,
        compute_losses,
        lambda_spread=lambda_spread,
        max_ratio=max_ratio,
        lambda_l0=lambda_l0,
        tu_temperature=tu_temperature,
    )


##────────────────────────────────────────────────────────────────────────────}}}
