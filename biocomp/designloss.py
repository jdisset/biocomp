"""Loss functions for circuit design optimization."""

from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
from jax import vmap, lax
from jax.tree_util import Partial

import numpy as np
from assertpy import assert_that

from .parameters import ParameterTree
from .optimutils import as_schedule, jax_three_phase_schedule
from .tumasking import (
    L0_PENALTY_FLOOR_PROB,
    asymmetric_l0_loss,
    commitment_penalty,
    entropy_bonus,
)
from .tumasking_strategy import get_full_log_alpha
from .jaxutils import check as jax_check
from .logging_config import get_logger
from .tracing import is_design_debug_enabled, save_debug_state

logger = get_logger(__name__)


@dataclass(frozen=True)
class GridLossWeights:
    """Typed SSOT for all grid loss weights and hyperparameters.

    Every call site that needs grid loss weights should construct or receive
    an instance of this class instead of passing individual kwargs.
    """

    w_sinkhorn: float = 1.0
    w_lncc: float = 0.5
    w_mse: float = 0.0
    w_rmse: float = 0.5
    w_simse: float = 0.0
    w_spectral: float = 0.0
    w_gradient: float = 0.0
    w_contrast: float = 0.0
    w_zncc: float = 0.0
    eps_sinkhorn: float = 0.1
    n_sinkhorn_iters: int = 50
    lncc_kernel: int = 7

    _FIELDS: ClassVar[tuple[str, ...]] = (
        "w_sinkhorn", "w_lncc", "w_mse", "w_rmse", "w_simse", "w_spectral",
        "w_gradient", "w_contrast", "w_zncc", "eps_sinkhorn", "n_sinkhorn_iters", "lncc_kernel",
    )

    @classmethod
    def from_loss_kwargs(cls, kwargs: dict) -> GridLossWeights:
        """Construct from a dict, accepting both 'w_sinkhorn' and 'sinkhorn' key styles."""
        filtered: dict[str, float | int] = {}
        for field_name in cls._FIELDS:
            if field_name in kwargs:
                filtered[field_name] = kwargs[field_name]
            else:
                # try without w_ prefix (e.g. "sinkhorn" -> "w_sinkhorn")
                short = field_name[2:] if field_name.startswith("w_") else None
                if short and short in kwargs:
                    filtered[field_name] = kwargs[short]
        return cls(**filtered)

    @classmethod
    def from_design_config(cls, dconf) -> GridLossWeights:
        """Construct from a DesignConfig's loss_function.kwargs."""
        lf = getattr(dconf, "loss_function", None)
        kwargs = (getattr(lf, "kwargs", None) or {}) if lf else {}
        return cls.from_loss_kwargs(kwargs)

    def weight_names(self) -> tuple[str, ...]:
        """Return the 9 weight field names (w_* only)."""
        return tuple(f for f in self._FIELDS if f.startswith("w_"))

    def to_dict(self) -> dict[str, float | int]:
        """Return all fields as a dict."""
        return {f: getattr(self, f) for f in self._FIELDS}


@dataclass
class GridLossResult:
    """Result of grid-based loss computation.

    This is the SINGLE SOURCE OF TRUTH for grid loss computation.
    Both TunerSession and grid_distance_loss MUST use this dataclass.
    """

    total: float
    sinkhorn: float
    lncc: float
    mse: float
    rmse: float = 0.0
    simse: float = 0.0
    spectral: float = 0.0
    gradient: float = 0.0
    contrast: float = 0.0
    zncc: float = 0.0
    sinkhorn_contrib: jnp.ndarray | None = None
    lncc_contrib: jnp.ndarray | None = None

    def to_dict(self) -> dict[str, float]:
        return {
            "total": self.total,
            "sinkhorn": self.sinkhorn,
            "lncc": self.lncc,
            "mse": self.mse,
            "rmse": self.rmse,
            "simse": self.simse,
            "spectral": self.spectral,
            "gradient": self.gradient,
            "contrast": self.contrast,
            "zncc": self.zncc,
        }


def _compute_raw_grid_losses(
    Y_pred: jnp.ndarray,
    Y_target: jnp.ndarray,
    eps_sinkhorn: float | jnp.ndarray,
    n_sinkhorn_iters: int,
    lncc_kernel: int,
) -> tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
]:
    """Compute all 9 unweighted grid loss components. JIT-safe (no guards, no float() casts).

    Returns: (sinkhorn, lncc, mse, rmse, spectral, simse, zncc, contrast, gradient)
    """
    Y_pred = _sanitize(Y_pred.astype(jnp.float32))
    Y_target = _sanitize(Y_target.astype(jnp.float32))
    y_flat, yhat_flat = Y_target.ravel(), Y_pred.ravel()

    sinkhorn_l = sinkhorn_divergence_conv(
        proj_nonneg_ste(Y_pred), proj_nonneg_ste(Y_target), eps_sinkhorn, n_iters=n_sinkhorn_iters,
    )
    lncc_l = lncc_grid_loss(None, Y_target, Y_pred, k=lncc_kernel)
    mse_l = jnp.mean((Y_pred - Y_target) ** 2)
    rmse_l = jnp.sqrt(mse_l)
    spectral_l = spectral_loss(None, Y_target, Y_pred)
    simse_l = simse_loss(None, y_flat, yhat_flat)
    zncc_l = zncc_loss(None, y_flat, yhat_flat)
    target_range = jnp.max(Y_target) - jnp.min(Y_target)
    pred_range = jnp.max(Y_pred) - jnp.min(Y_pred)
    contrast_l = jax.nn.relu(target_range - pred_range)
    gradient_l = gradient_magnitude_loss(Y_target, Y_pred)

    return sinkhorn_l, lncc_l, mse_l, rmse_l, spectral_l, simse_l, zncc_l, contrast_l, gradient_l


def compute_grid_losses(
    Y_pred: jnp.ndarray,
    Y_target: jnp.ndarray,
    weights: GridLossWeights | None = None,
    return_contributions: bool = False,
    **kw,
) -> GridLossResult:
    """Compute grid-based losses - shared between tuner and design mode.

    This is the SINGLE SOURCE OF TRUTH for grid loss computation.
    Both TunerSession and grid_distance_loss MUST call this function.

    Args:
        Y_pred: Predicted grid, shape (H, W)
        Y_target: Target grid, shape (H, W)
        weights: GridLossWeights instance. If None, constructed from **kw.
        return_contributions: If True, compute per-pixel LNCC contribution
        **kw: Backward-compatible kwargs (w_sinkhorn, w_lncc, etc.) used if weights is None.

    Returns:
        GridLossResult with scalar losses and optional per-pixel contributions
    """
    assert Y_pred.shape == Y_target.shape, f"Shape mismatch: {Y_pred.shape} vs {Y_target.shape}"
    assert Y_pred.ndim == 2, f"Expected 2D grid, got {Y_pred.ndim}D"

    if weights is None:
        weights = GridLossWeights.from_loss_kwargs(kw)

    w = weights

    Y_pred = _sanitize(Y_pred.astype(jnp.float32))
    Y_target = _sanitize(Y_target.astype(jnp.float32))

    # sinkhorn is expensive — guard with w > 0
    sinkhorn_l = (
        sinkhorn_divergence_conv(
            proj_nonneg_ste(Y_pred), proj_nonneg_ste(Y_target),
            w.eps_sinkhorn, n_iters=w.n_sinkhorn_iters,
        )
        if w.w_sinkhorn > 0 else jnp.array(0.0)
    )

    lncc_l = lncc_grid_loss(None, Y_target, Y_pred, k=w.lncc_kernel) if w.w_lncc > 0 else jnp.array(0.0)
    mse_l = jnp.mean((Y_pred - Y_target) ** 2) if (w.w_mse > 0 or w.w_rmse > 0) else jnp.array(0.0)
    rmse_l = jnp.sqrt(mse_l) if w.w_rmse > 0 else jnp.array(0.0)
    simse_l = simse_loss(None, Y_target.flatten(), Y_pred.flatten()) if w.w_simse > 0 else jnp.array(0.0)
    spectral_l = spectral_loss(None, Y_target, Y_pred) if w.w_spectral > 0 else jnp.array(0.0)
    gradient_l = gradient_magnitude_loss(Y_target, Y_pred) if w.w_gradient > 0 else jnp.array(0.0)
    if w.w_contrast > 0:
        target_range = jnp.max(Y_target) - jnp.min(Y_target)
        pred_range = jnp.max(Y_pred) - jnp.min(Y_pred)
        contrast_l = jax.nn.relu(target_range - pred_range)
    else:
        contrast_l = jnp.array(0.0)
    zncc_l = zncc_loss(None, Y_target.flatten(), Y_pred.flatten()) if w.w_zncc > 0 else jnp.array(0.0)
    logger.debug(
        f"compute_grid_losses: w_mse={w.w_mse}, w_rmse={w.w_rmse}, raw_mse={float(mse_l):.6f}, raw_rmse={float(rmse_l):.6f}"
    )

    total = (
        w.w_sinkhorn * sinkhorn_l
        + w.w_lncc * lncc_l
        + w.w_mse * mse_l
        + w.w_rmse * rmse_l
        + w.w_simse * simse_l
        + w.w_spectral * spectral_l
        + w.w_gradient * gradient_l
        + w.w_contrast * contrast_l
        + w.w_zncc * zncc_l
    )

    lncc_contrib = None
    if return_contributions:
        lncc_contrib = _compute_lncc_contribution(Y_pred, Y_target, w.lncc_kernel)

    return GridLossResult(
        total=float(total),
        sinkhorn=float(sinkhorn_l),
        lncc=float(lncc_l),
        mse=float(mse_l),
        rmse=float(rmse_l),
        simse=float(simse_l),
        spectral=float(spectral_l),
        gradient=float(gradient_l),
        contrast=float(contrast_l),
        zncc=float(zncc_l),
        lncc_contrib=lncc_contrib,
    )


def _box2d_sum(a: jnp.ndarray, r: int) -> jnp.ndarray:
    """2D box filter via summed area table. r is the half-width (kernel size = 2r+1)."""
    a = jnp.pad(a, ((r + 1, r), (r + 1, r)), mode="edge")
    s = jnp.cumsum(jnp.cumsum(a, 0), 1)
    return (
        s[: -2 * r - 1, : -2 * r - 1]
        - s[: -2 * r - 1, 2 * r + 1 :]
        - s[2 * r + 1 :, : -2 * r - 1]
        + s[2 * r + 1 :, 2 * r + 1 :]
    )


def _lncc_grid_per_pixel(
    y: jnp.ndarray, yhat: jnp.ndarray, k: int = 7, eps: float = 1e-6
) -> jnp.ndarray:
    """Compute per-pixel LNCC values on a 2D grid."""
    r, N = k // 2, k * k
    m_y, m_yhat = _box2d_sum(y, r) / N, _box2d_sum(yhat, r) / N
    y_c, yhat_c = y - m_y, yhat - m_yhat
    var_y, var_yhat = _box2d_sum(y_c**2, r), _box2d_sum(yhat_c**2, r)
    cov = _box2d_sum(y_c * yhat_c, r)
    std_product = jnp.sqrt((var_y + eps) * (var_yhat + eps))
    return jnp.clip(cov / std_product, -1, 1)


def _compute_lncc_contribution(
    Y_pred: jnp.ndarray, Y_target: jnp.ndarray, k: int = 7
) -> jnp.ndarray:
    return 1.0 - _lncc_grid_per_pixel(Y_target, Y_pred, k)


def _sanitize(x):
    return jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _validate_grid_order(X: jnp.ndarray, xres: int, yres: int, jitter_tolerance: float = 0.15):
    """Validate X coordinates are in sequential grid order (for use inside JIT with checkify)."""
    X_grid = X.reshape(yres, xres, 2)

    first_row_x = X_grid[0, :, 0]
    first_row_y = X_grid[0, :, 1]
    first_col_x = X_grid[:, 0, 0]
    first_col_y = X_grid[:, 0, 1]

    x_step = jnp.array(1.0 / (xres - 1) if xres > 1 else 1.0)
    y_step = jnp.array(1.0 / (yres - 1) if yres > 1 else 1.0)
    x_tol = x_step * jitter_tolerance
    y_tol = y_step * jitter_tolerance

    row_y_spread = jnp.max(first_row_y) - jnp.min(first_row_y)
    jax_check(
        row_y_spread < y_step + y_tol,
        "Grid order violation: first row y-coords spread too large. "
        "Data may be shuffled. Set reshuffle_batches=false.",
    )

    row_x_diffs = first_row_x[1:] - first_row_x[:-1]
    row_x_monotonic = jnp.all(row_x_diffs > -x_tol)
    jax_check(
        row_x_monotonic,
        "Grid order violation: first row x-coords not monotonically increasing. "
        "Data may be shuffled. Set reshuffle_batches=false.",
    )

    col_x_spread = jnp.max(first_col_x) - jnp.min(first_col_x)
    jax_check(
        col_x_spread < x_step + x_tol,
        "Grid order violation: first column x-coords spread too large. "
        "Data may be shuffled. Set reshuffle_batches=false.",
    )

    col_y_diffs = first_col_y[1:] - first_col_y[:-1]
    col_y_monotonic = jnp.all(col_y_diffs > -y_tol)
    jax_check(
        col_y_monotonic,
        "Grid order violation: first column y-coords not monotonically increasing. "
        "Data may be shuffled. Set reshuffle_batches=false.",
    )


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
    # note: eps > 0 assertion removed to support JAX tracing with dynamic schedules
    eps = jnp.maximum(eps, 1e-6)  # clamp to avoid numerical issues

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


def proj_nonneg_ste(z, leak=1e-3, cap=None):
    """Project to nonnegative with straight-through estimator."""
    z_clip = jnp.clip(z, 0.0, cap) if cap is not None else jnp.maximum(z, 0.0)
    z_leaky = jnp.where(z >= 0.0, z, leak * z)
    return z_clip + jax.lax.stop_gradient(z_leaky - z_clip)


def zncc_loss(x, y, yhat, eps=1e-6, **kw):
    """Zero-mean normalized cross-correlation loss."""
    assert y.shape == yhat.shape, f"zncc_loss shape mismatch: y={y.shape} vs yhat={yhat.shape}"

    y, yhat = _sanitize(y), _sanitize(yhat)
    y0, yhat0 = y - jnp.mean(y), yhat - jnp.mean(yhat)
    cov = jnp.mean(y0 * yhat0)
    var_y, var_yhat = jnp.mean(y0**2), jnp.mean(yhat0**2)
    std_product = jnp.sqrt((var_y + eps) * (var_yhat + eps))
    return 1.0 - cov / std_product


def spectral_loss(x, y, yhat, **kw):
    return jnp.mean((jnp.abs(jnp.fft.fft2(y)) - jnp.abs(jnp.fft.fft2(yhat))) ** 2)


def gradient_magnitude_loss(y, yhat, eps=1e-6, **kw):
    """Compare gradient magnitudes to capture edge/shape structure.

    This loss penalizes differences in where edges/boundaries are located.
    Unlike Sinkhorn which only cares about mass distribution, this captures
    the local structure of the shape.
    """
    assert y.ndim == 2, f"gradient_magnitude_loss: y must be 2D grid, got {y.ndim}D"
    assert y.shape == yhat.shape, (
        f"gradient_magnitude_loss shape mismatch: {y.shape} vs {yhat.shape}"
    )

    y, yhat = _sanitize(y.astype(jnp.float32)), _sanitize(yhat.astype(jnp.float32))

    dy_y = jnp.diff(y, axis=0)
    dx_y = jnp.diff(y, axis=1)
    dy_yhat = jnp.diff(yhat, axis=0)
    dx_yhat = jnp.diff(yhat, axis=1)

    grad_mag_y = jnp.sqrt(dy_y[:, :-1] ** 2 + dx_y[:-1, :] ** 2 + eps)
    grad_mag_yhat = jnp.sqrt(dy_yhat[:, :-1] ** 2 + dx_yhat[:-1, :] ** 2 + eps)

    return jnp.mean((grad_mag_y - grad_mag_yhat) ** 2)


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


def lncc_grid_loss(x, y, yhat, k=7, eps=1e-6, **kw):
    assert y.ndim == 2, f"lncc_grid_loss: y must be 2D grid, got {y.ndim}D"
    assert y.shape == yhat.shape, f"lncc_grid_loss shape mismatch: y={y.shape} vs yhat={yhat.shape}"
    y, yhat = _sanitize(y), _sanitize(yhat)
    return 1.0 - jnp.mean(_lncc_grid_per_pixel(y, yhat, k, eps))


def soft_tucount_penalty(W, max_tus=5, rel_active=1e-3, width=2e-4):
    """Penalty for having more than max_tus active TUs per co-transfection (aggregation row)."""
    A = jnp.abs(W)
    if A.ndim == 0:
        return jnp.array(0.0)
    if A.ndim == 1:
        A = A[None, :]
    elif A.ndim > 2:
        A = A.reshape(-1, A.shape[-1])
    m = jnp.max(A, axis=1, keepdims=True)
    norm = jnp.where(m > 0, A / (m + 1e-12), 0.0)
    soft_count = jnp.sum(jax.nn.sigmoid((norm - rel_active) / (width + 1e-12)), axis=1)
    return jnp.sum(jnp.square(jax.nn.relu(soft_count - max_tus)))


def get_tucount_penalty_for_leaf(p, max_tus=5, rel_active=1e-3, width=2e-4):
    """Get TU count penalty for a parameter leaf (handles ArrayRef and raw arrays)."""
    kw = dict(max_tus=max_tus, rel_active=rel_active, width=width)
    if hasattr(p, "view"):
        return soft_tucount_penalty(p.view(), **kw)
    return soft_tucount_penalty(p, **kw)


def ratio_spread_penalty(W, max_ratio=100.0, eps=1e-9):
    """Penalty for ratio spread exceeding max_ratio."""
    A = jnp.abs(W)
    if A.ndim == 0:
        return jnp.array(0.0)
    if A.ndim == 1:
        A = A[None, :]
    elif A.ndim > 2:
        A = A.reshape(-1, A.shape[-1])
    log_max_ratio = jnp.log(max_ratio + eps)
    pos_mask = A > eps
    log_A = jnp.where(pos_mask, jnp.log(A + eps), -jnp.inf)
    log_max = jnp.max(jnp.where(pos_mask, log_A, -jnp.inf), axis=1)
    log_min = jnp.min(jnp.where(pos_mask, log_A, jnp.inf), axis=1)
    log_spread = log_max - log_min
    excess = jax.nn.relu(log_spread - log_max_ratio)
    return jnp.sum(jnp.square(excess))


def get_spread_penalty_for_leaf(p, max_ratio=100.0):
    if hasattr(p, "view"):
        return ratio_spread_penalty(p.view(), max_ratio=max_ratio)
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

        # MAX normalization: threshold=0.005 means "ratio < 0.5% of largest ratio in that node"
        ratio_max = jnp.max(ratios, axis=-1, keepdims=True)
        normalized_ratios = ratios / jnp.maximum(ratio_max, 1e-8)

        network_ids_expanded = jnp.broadcast_to(node_network_ids[:, None], (n_nodes, n_outputs))

        valid_tu_mask = (tu_indices >= 0).astype(jnp.float32)
        safe_tu_indices = jnp.maximum(tu_indices, 0)
        safe_network_ids = jnp.clip(network_ids_expanded, 0, n_networks - 1)
        safe_tu_indices = jnp.clip(safe_tu_indices, 0, n_tus - 1)

        tu_log_alpha_per_ratio = tu_log_alpha_2d[safe_network_ids, safe_tu_indices]
        tu_enabled_prob = jax.nn.sigmoid(tu_log_alpha_per_ratio)

        below_threshold = jax.nn.relu(min_ratio_threshold - normalized_ratios)
        per_element_penalty = tu_enabled_prob * below_threshold * valid_tu_mask

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
    yhatdep = jnp.nan_to_num(yhatdep, nan=0.0, posinf=1.0, neginf=0.0)

    xsplit = jnp.reshape(x, (batch_size, n_targets, n_networks, n_inputs_per_network))
    return vmap(vmap(lossfunc, in_axes=(1, 1, 1)), in_axes=(1, 1, 1))(xsplit, yhatdep, y)


def _compute_tu_stats(params) -> dict:
    """Compute TU masking statistics for logging.

    Includes diagnostic metrics for convergence analysis:
    - mask_entropy: binary entropy of probabilities (high=exploring, low=committed)
    - boundary_count: TUs with prob in [0.3, 0.7] (still deciding)
    - below_floor_count: TUs with prob < 0.2 (in the "graveyard")
    """
    log_alpha = get_full_log_alpha(params)
    if log_alpha is None:
        return {}

    tu_probs = jax.nn.sigmoid(log_alpha)
    tu_enabled_mask = tu_probs > 0.5
    n_tus = log_alpha.shape[-1]

    # diagnostic metrics for convergence analysis
    probs_flat = tu_probs.flatten()
    eps = 1e-6
    probs_clipped = jnp.clip(probs_flat, eps, 1 - eps)
    entropy_per_tu = -(probs_clipped * jnp.log(probs_clipped) + (1 - probs_clipped) * jnp.log(1 - probs_clipped))
    mask_entropy = jnp.mean(entropy_per_tu) / jnp.log(2.0)  # normalize to [0, 1]

    boundary_mask = (probs_flat >= 0.3) & (probs_flat <= 0.7)
    below_floor_mask = probs_flat < L0_PENALTY_FLOOR_PROB

    return {
        "log_alpha": log_alpha,
        "enabled_count": jnp.sum(tu_enabled_mask),
        "total_count": jnp.array(log_alpha.size),
        "n_tus": jnp.array(n_tus),
        "mean_prob": jnp.mean(tu_probs),
        "min_log_alpha": jnp.min(log_alpha),
        "max_log_alpha": jnp.max(log_alpha),
        "log_alpha_std": jnp.std(log_alpha),
        "enabled_count_per_network": jnp.sum(tu_enabled_mask, axis=-1),
        "mean_prob_per_network": jnp.mean(tu_probs, axis=-1),
        "min_log_alpha_per_network": jnp.min(log_alpha, axis=-1),
        "max_log_alpha_per_network": jnp.max(log_alpha, axis=-1),
        "std_log_alpha_per_network": jnp.std(log_alpha, axis=-1),
        # diagnostic metrics for TUMaskingDiagLogger
        "mask_entropy": mask_entropy,
        "boundary_count": jnp.sum(boundary_mask),
        "below_floor_count": jnp.sum(below_floor_mask),
    }


def _compute_ratio_stats(ratio_leaves: list) -> dict:
    """Compute ratio statistics for logging."""
    if not ratio_leaves:
        return {}
    all_ratios = []
    for p in ratio_leaves:
        arr = p.view() if hasattr(p, "view") else p if hasattr(p, "shape") else None
        if arr is not None:
            all_ratios.append(jnp.abs(arr).ravel())
    if not all_ratios:
        return {}
    ratios_flat = jnp.concatenate(all_ratios)
    return {
        "min": jnp.min(ratios_flat),
        "max": jnp.max(ratios_flat),
        "mean": jnp.mean(ratios_flat),
        "std": jnp.std(ratios_flat),
        "nonzero_count": jnp.sum(ratios_flat > 1e-6),
        "total_count": jnp.array(ratios_flat.size),
    }


def _dump_axis_assignments(dmanager, n_targets: int, n_networks: int) -> None:
    """Dump axis assignment debug info (only when debug enabled)."""
    if not is_design_debug_enabled():
        return
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
                    "target_input_names": target_input_names,
                    "network_id": net_idx,
                    "network_name": network_name,
                    "network_input_proteins": network_input_proteins,
                }
            )
    save_debug_state(
        "axis_assignment_mapping",
        {"assignments": axis_assignments},
        {
            "n_targets": n_targets,
            "n_networks": n_networks,
            "note": "X columns are in alphabetical order of target.input_names.",
        },
        output_dir=get_design_debug_output_dir(),
        mode="design",
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
            "phase1_frac": 0.0,
            "phase2_frac": 0.0,
            "phase1_value": float(spec),
            "phase2_end_value": float(spec),
            "phase3_end_value": float(spec),
        }

    if isinstance(spec, dict):
        if "start" in spec and "end" in spec:
            return {
                "phase1_frac": 0.0,
                "phase2_frac": 1.0,
                "phase1_value": float(spec["start"]),
                "phase2_end_value": float(spec["end"]),
                "phase3_end_value": float(spec["end"]),
            }
        if "phase1_value" in spec:
            return {
                "phase1_frac": float(spec.get("phase1_frac", 0.4)),
                "phase2_frac": float(spec.get("phase2_frac", 0.75)),
                "phase1_value": float(spec["phase1_value"]),
                "phase2_end_value": float(spec.get("phase2_end_value", spec["phase1_value"])),
                "phase3_end_value": float(spec.get("phase3_end_value", spec["phase1_value"])),
            }

    raise ValueError(
        f"Invalid schedule spec: {spec}. Expected float, callable, or dict with 'start'/'end' or 'phase1_value'/etc."
    )


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
            result[f"{HYPEROPT_SCHEDULE_NAMESPACE}/{name}_{key}"] = jnp.array(
                value, dtype=jnp.float32
            )
    return result


_SCHEDULE_FALLBACK_WARNED: set[str] = set()


def _get_schedule_value(
    params, step, total_steps, schedule_name, schedule_or_value, schedule_ns=None
):
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
        if schedule_name not in _SCHEDULE_FALLBACK_WARNED:
            _SCHEDULE_FALLBACK_WARNED.add(schedule_name)
            logger.warning(
                f"Schedule '{schedule_name}' not found in params at '{prefix}_phase1_value'. "
                f"Using fallback value. This hyperparam will NOT be optimized."
            )
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
    lambda_entropy=0.0,
    lambda_commitment=0.0,
    commitment_margin=0.05,
    l0_tu_threshold=None,
    l0_excess_exponent=2.0,
    l0_alpha_below=0.5,
    l0_beta_above=2.0,
    l0_blend_sharpness=5.0,
    l0_leak_coef=0.0,
    tu_n_samples=4,
    lambda_coupling=0.1,
    min_ratio_threshold=0.005,
    lambda_ern_tying=0.0,
    hyperopt_schedule_ns=None,
    hyperopt_total_steps=None,
    **_kw,
):
    """Create the loss function for design optimization.

    Args:
        tu_n_samples: DEPRECATED - ignored. Binary TU masking is now deterministic.
            Kept for API compatibility.
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

    n_targets, n_networks = dmanager.n_targets, len(dmanager.networks)
    dep_mask = stack.get_dependent_output_mask()
    nb_dep = int(np.sum(dep_mask))
    ratio_paths = ratio_paths or []

    # per-network TU mask: only penalize TUs each network actually uses
    per_network_tu_mask = None
    protected_tu_mask = None  # True = protected (stop_gradient), False = normal
    if dmanager.enable_tu_masking and hasattr(stack, "get_per_network_tu_mask"):
        per_network_tu_mask = stack.get_per_network_tu_mask()
        logger.debug(f"Per-network TU mask shape: {per_network_tu_mask.shape}")

        # protected TU mask: True = protected (gradients blocked), False = normal
        if hasattr(stack, "no_masking_tu_ids") and stack.no_masking_tu_ids and stack.tu_id_to_idx:
            protected_tu_mask = np.zeros(stack.n_tus, dtype=bool)
            for tu_id in stack.no_masking_tu_ids:
                if tu_id in stack.tu_id_to_idx:
                    protected_tu_mask[stack.tu_id_to_idx[tu_id]] = True
            protected_tu_mask = jnp.array(protected_tu_mask)
            n_protected = int(np.sum(protected_tu_mask))
            logger.debug(
                f"Protected TU mask: {n_protected} TUs have stop_gradient (always enabled)"
            )

    ern_namespaces = [
        layer.namespace
        for layer in (stack.layers or [])
        if layer.f_type and layer.f_type.startswith("sequestron_ERN")
    ]
    _dump_axis_assignments(dmanager, n_targets, n_networks)

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

        tucount_w = _get_schedule_value(
            params, step, total_steps, "lambda_tucount", lambda_tucount, schedule_ns
        )
        tucount_penalty = tucount_w * sum(
            get_tucount_penalty_for_leaf(p, max_tus=max_tus_per_cotx) for p in ratio_leaves
        )
        tucount_penalty = _sanitize(jnp.atleast_1d(tucount_penalty))[0]
        spread_w = _get_schedule_value(
            params, step, total_steps, "lambda_spread", lambda_spread, schedule_ns
        )
        spread_penalty = spread_w * sum(
            get_spread_penalty_for_leaf(p, max_ratio=max_ratio) for p in ratio_leaves
        )
        spread_penalty = _sanitize(jnp.atleast_1d(spread_penalty))[0]

        l0_penalty = jnp.array(0.0)
        l0_penalty_per_network = None
        coupling_penalty = jnp.array(0.0)
        coupling_penalty_per_target = None
        entropy_penalty = jnp.array(0.0)
        commitment_penalty_val = jnp.array(0.0)

        log_alpha = get_full_log_alpha(params)

        if log_alpha is not None:
            assert log_alpha.ndim == 3, (
                f"log_alpha must be 3D (n_targets, n_networks, n_tus), got {log_alpha.ndim}D"
            )
            assert log_alpha.shape[0] == n_targets, (
                f"log_alpha n_targets mismatch: {log_alpha.shape[0]} vs {n_targets}"
            )
            assert log_alpha.shape[1] == n_networks, (
                f"log_alpha n_networks mismatch: {log_alpha.shape[1]} vs {n_networks}"
            )
            log_alpha = jnp.nan_to_num(log_alpha, nan=0.0, posinf=10.0, neginf=-10.0)

            from biocomp.tumasking import l0_penalty as l0_penalty_fn

            per_tu_penalty = l0_penalty_fn(log_alpha, leak_coef=l0_leak_coef)
            if per_network_tu_mask is not None:
                assert per_network_tu_mask.shape[0] == n_networks, (
                    f"per_network_tu_mask shape mismatch: {per_network_tu_mask.shape} vs n_networks={n_networks}"
                )
                per_tu_penalty = per_tu_penalty * per_network_tu_mask[None, :, :]

            expected_count_per_network = jnp.sum(per_tu_penalty, axis=-1)

            if l0_tu_threshold is not None and l0_tu_threshold > 0:
                # vmap asymmetric_l0_loss over (targets, networks)
                def compute_asymmetric_l0(la_single_network):
                    return asymmetric_l0_loss(
                        la_single_network,
                        threshold=l0_tu_threshold,
                        alpha_below=l0_alpha_below,
                        beta_above=l0_beta_above,
                        blend_sharpness=l0_blend_sharpness,
                        leak_coef=l0_leak_coef,
                    )

                # log_alpha shape: (n_targets, n_networks, n_tus)
                # vmap over targets then networks (use masked log_alpha)
                l0_raw_per_network = vmap(vmap(compute_asymmetric_l0))(log_alpha)
            else:
                l0_raw_per_network = expected_count_per_network

            l0_weight = _get_schedule_value(
                params, step, total_steps, "lambda_l0", lambda_l0, schedule_ns
            )
            l0_penalty_per_network = _sanitize(l0_weight * l0_raw_per_network)
            l0_penalty = _sanitize(jnp.atleast_1d(jnp.sum(l0_penalty_per_network)))[0]

            if min_ratio_threshold > 0 and ratio_paths:
                coupling_weight = _get_schedule_value(
                    params, step, total_steps, "lambda_coupling", lambda_coupling, schedule_ns
                )
                # use masked log_alpha so protected TUs don't get gradient pressure
                raw_coupling, raw_coupling_per_target = ratio_mask_coupling_penalty(
                    params,
                    ratio_paths,
                    log_alpha,
                    min_ratio_threshold,
                    return_per_target=True,
                )
                coupling_penalty_per_target = _sanitize(coupling_weight * raw_coupling_per_target)
                coupling_penalty = _sanitize(jnp.atleast_1d(coupling_weight * raw_coupling))[0]

            if ern_namespaces:
                tying_weight = _get_schedule_value(
                    params, step, total_steps, "lambda_ern_tying", lambda_ern_tying, schedule_ns
                )
                # use masked log_alpha so protected TUs don't get gradient pressure
                raw_tying = ern_tu_tying_penalty(params, ern_namespaces, log_alpha)
                ern_tying_penalty_val = tying_weight * raw_tying
                ern_tying_penalty_val = _sanitize(jnp.atleast_1d(ern_tying_penalty_val))[0]
            else:
                ern_tying_penalty_val = jnp.array(0.0)

            entropy_weight = _get_schedule_value(
                params, step, total_steps, "lambda_entropy", lambda_entropy, schedule_ns
            )
            ent = entropy_bonus(log_alpha)
            entropy_penalty = _sanitize(jnp.atleast_1d(-entropy_weight * ent))[0]

            commitment_weight = _get_schedule_value(
                params, step, total_steps, "lambda_commitment", lambda_commitment, schedule_ns
            )
            commit_raw = jnp.mean(commitment_penalty(log_alpha, commitment_margin))
            commitment_penalty_val = _sanitize(jnp.atleast_1d(commitment_weight * commit_raw))[0]
        else:
            ern_tying_penalty_val = jnp.array(0.0)
            commitment_penalty_val = jnp.array(0.0)

        # single forward pass with binary TU masking (deterministic, no need for sample averaging)
        # get_tu_masks() now uses binary masking by default (not hard concrete)
        yhatdep, apply_aux = single_forward_pass(params, X, Z, forward_key, tu_uniform=None)
        all_losses, extra_aux_inner = compute_losses_fn(
            params, X, Y, yhatdep, step, n_targets, n_networks
        )

        all_losses = _sanitize(all_losses)
        tu_stats = _compute_tu_stats(params)
        ratio_stats = _compute_ratio_stats(ratio_leaves)
        sublosses = extra_aux_inner.get("sublosses", {}) if extra_aux_inner else {}

        pred_stats_per_network = {
            "mean": jnp.mean(yhatdep, axis=0),
            "std": jnp.std(yhatdep, axis=0),
            "min": jnp.min(yhatdep, axis=0),
            "max": jnp.max(yhatdep, axis=0),
        }

        aux = {
            "apply_aux": apply_aux,
            "all_losses": all_losses,
            "yhatdep": yhatdep,
            "X": X,
            "Y": Y,
            "l0_penalty": l0_penalty,
            "entropy_penalty": entropy_penalty,
            "commitment_penalty": commitment_penalty_val,
            "coupling_penalty": coupling_penalty,
            "ern_tying_penalty": ern_tying_penalty_val,
            "tucount_penalty": tucount_penalty,
            "spread_penalty": spread_penalty,
            "l0_penalty_per_network": l0_penalty_per_network,
            "coupling_penalty_per_target": coupling_penalty_per_target,
            "tu_stats": tu_stats,
            "ratio_stats": ratio_stats,
            "sublosses": sublosses,
            "pred_stats_per_network": pred_stats_per_network,
        }

        loss = (
            all_losses.mean()
            + tucount_penalty
            + spread_penalty
            + l0_penalty
            + entropy_penalty
            + commitment_penalty_val
            + coupling_penalty
            + ern_tying_penalty_val
        )
        loss = jnp.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=1e6)
        return loss, aux

    return loss_func


def grid_distance_loss(
    stack,
    dconf,
    dmanager,
    num_z,
    ratio_paths=None,
    *,
    weights: GridLossWeights | None = None,
    lambda_tucount=0.0,
    max_tus_per_cotx=5,
    lambda_spread=0.01,
    max_ratio=100.0,
    lambda_l0=0.0,
    lambda_entropy=0.0,
    lambda_commitment=0.0,
    commitment_margin=0.05,
    l0_tu_threshold=None,
    l0_excess_exponent=2.0,
    l0_alpha_below=0.5,
    l0_beta_above=2.0,
    l0_blend_sharpness=5.0,
    l0_leak_coef=0.0,
    tu_n_samples=4,
    lambda_coupling=0.1,
    min_ratio_threshold=0.005,
    lambda_ern_tying=0.0,
    hyperopt_schedule_ns=None,
    hyperopt_total_steps=None,
    **kw,
):
    assert dmanager.is_lattice_mode, "grid_distance_loss requires lattice sampling"
    assert not dconf.reshuffle_batches, (
        "CRITICAL: grid_distance_loss requires reshuffle_batches=False. "
        "The loss reshapes data to (yres, xres) assuming sequential grid order. "
        "With reshuffle_batches=True, data is permuted and the grid becomes scrambled, "
        "making sinkhorn/lncc losses meaningless. Set reshuffle_batches: false in your config."
    )
    if weights is None:
        weights = GridLossWeights.from_loss_kwargs(kw)
    glw = weights

    xres, yres = dmanager.grid_resolution
    n_networks = len(dmanager.networks)
    total_steps = hyperopt_total_steps or 100000
    schedule_ns = hyperopt_schedule_ns

    def compute_losses(params, X, Y, yhatdep, step, n_targets, n_networks_):
        yhatdep = _sanitize(yhatdep)
        assert Y.ndim == 3 and Y.shape[-1] == 1, (
            f"grid_distance_loss expects Y shape (batch_size, n_targets, 1), got {Y.shape}. "
            "The last dim must be 1 for proper squeeze+reshape to grid."
        )
        batch_size = Y.shape[0]
        assert batch_size == xres * yres, (
            f"batch_size={batch_size} must equal xres*yres={xres * yres} for grid reshape"
        )
        assert X.shape[0] == batch_size, (
            f"X shape {X.shape} batch dim incompatible with batch_size={batch_size}"
        )
        assert X.shape[-1] >= 2 and X.shape[-1] % 2 == 0, (
            f"X shape {X.shape} last dim must be >=2 and even (2 coords per network)"
        )
        X_coords = X[..., :2].reshape(-1, 2)
        _validate_grid_order(X_coords, xres, yres)

        assert yhatdep.shape == (batch_size, n_targets, n_networks_), (
            f"yhatdep shape {yhatdep.shape} != expected ({batch_size}, {n_targets}, {n_networks_})"
        )

        Y_images = jnp.tile(
            Y.squeeze(-1).T.reshape(n_targets, 1, yres, xres), (1, n_networks, 1, 1)
        )
        yhat_images = yhatdep.transpose(1, 2, 0).reshape(n_targets, n_networks, yres, xres)

        assert Y_images.shape == yhat_images.shape, (
            f"Y_images {Y_images.shape} != yhat_images {yhat_images.shape} after reshape"
        )

        w_sink = _get_schedule_value(
            params, step, total_steps, "w_sinkhorn", glw.w_sinkhorn, schedule_ns
        )
        w_lncc_val = _get_schedule_value(params, step, total_steps, "w_lncc", glw.w_lncc, schedule_ns)
        w_mse_val = _get_schedule_value(params, step, total_steps, "w_mse", glw.w_mse, schedule_ns)
        w_rmse_val = _get_schedule_value(params, step, total_steps, "w_rmse", glw.w_rmse, schedule_ns)
        w_spec = _get_schedule_value(
            params, step, total_steps, "w_spectral", glw.w_spectral, schedule_ns
        )
        w_sim = _get_schedule_value(params, step, total_steps, "w_simse", glw.w_simse, schedule_ns)
        w_zncc_val = _get_schedule_value(params, step, total_steps, "w_zncc", glw.w_zncc, schedule_ns)
        w_con = _get_schedule_value(
            params, step, total_steps, "w_contrast", glw.w_contrast, schedule_ns
        )
        w_grad = _get_schedule_value(
            params, step, total_steps, "w_gradient", glw.w_gradient, schedule_ns
        )
        eps_sink = _get_schedule_value(
            params, step, total_steps, "eps_sinkhorn", glw.eps_sinkhorn, schedule_ns
        )

        (
            sinkhorn_losses,
            lncc_losses,
            mse_losses,
            rmse_losses,
            spectral_losses,
            simse_losses,
            zncc_losses,
            contrast_losses,
            gradient_losses,
        ) = vmap(vmap(lambda y, yh: _compute_raw_grid_losses(y, yh, eps_sink, glw.n_sinkhorn_iters, glw.lncc_kernel)))(
            Y_images, yhat_images
        )

        all_losses = (
            w_sink * sinkhorn_losses
            + w_lncc_val * lncc_losses
            + w_mse_val * mse_losses
            + w_rmse_val * rmse_losses
            + w_spec * spectral_losses
            + w_sim * simse_losses
            + w_zncc_val * zncc_losses
            + w_con * contrast_losses
            + w_grad * gradient_losses
        )

        _loss_components = [
            ("sinkhorn", w_sink, sinkhorn_losses),
            ("lncc", w_lncc_val, lncc_losses),
            ("mse", w_mse_val, mse_losses),
            ("rmse", w_rmse_val, rmse_losses),
            ("spectral", w_spec, spectral_losses),
            ("simse", w_sim, simse_losses),
            ("zncc", w_zncc_val, zncc_losses),
            ("contrast", w_con, contrast_losses),
            ("gradient", w_grad, gradient_losses),
        ]
        sublosses = {}
        for name, weight, losses in _loss_components:
            sublosses[name] = _sanitize(jnp.mean(losses))
            sublosses[f"{name}_weighted"] = _sanitize(weight * jnp.mean(losses))
            sublosses[f"{name}_per_network"] = _sanitize(losses)
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
        lambda_entropy=lambda_entropy,
        lambda_commitment=lambda_commitment,
        commitment_margin=commitment_margin,
        l0_tu_threshold=l0_tu_threshold,
        l0_excess_exponent=l0_excess_exponent,
        l0_alpha_below=l0_alpha_below,
        l0_beta_above=l0_beta_above,
        l0_blend_sharpness=l0_blend_sharpness,
        l0_leak_coef=l0_leak_coef,
        tu_n_samples=tu_n_samples,
        lambda_coupling=lambda_coupling,
        min_ratio_threshold=min_ratio_threshold,
        lambda_ern_tying=lambda_ern_tying,
        hyperopt_schedule_ns=hyperopt_schedule_ns,
        hyperopt_total_steps=hyperopt_total_steps,
    )
