### {{{                          --     imports     --
import random
from functools import partial
import numpy as np
from . import datautils as du
from biocomp.utils import encode_function
from tqdm import tqdm
from biocomp.compute import ComputeStack, ComputeConfig
import biocomp.nodes as nd
from biocomp.utils import (
    EncodedPartialFunction,
    PartialFunction,
    ArbitraryModel,
    PartialFunctionResult,
)
from biocomp.network import Network
from biocomp.recipe import CoTransfection, Unit, Slot
from biocomp.train import create_counter
import biocomp.utils
from assertpy import assert_that
from . import nodes as nodes
from .parameters import ParameterTree, ParamPath
from . import utils as ut
import time
from typing import List, Tuple, Callable, Optional, NamedTuple, Union, Literal
from pydantic import Field, BaseModel, ConfigDict
from biocomp.logging_config import get_logger
import optax
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jax import vmap, jit, lax
from .jaxutils import get_looped_slice
import os
from jax.experimental import checkify

from biocomptools.modelmodel import BiocompModel
from biocomp.designutils import sample_from_svg
from biocomp.optimutils import (
    make_training_step,
    per_replicate_step,
    per_replicate_step_nonscan,
    optimize,
    as_schedule,
    OptimConfig,
    DEFAULT_OPTIMIZER,
)

from pathlib import Path
from jax.typing import ArrayLike

logger = get_logger(__name__)

##────────────────────────────────────────────────────────────────────────────}}}


## {{{                    --     fast ot on a grid:     --


# When the cost is squared Euclidean and supports lie on a uniform grid, the entropic kernel is
# K(p-q)=\exp\!\big(-\|p-q\|^2/\varepsilon\big),
# which is a Gaussian. Sinkhorn then alternates
# u \leftarrow a \oslash (K * v),\qquad
# v \leftarrow b \oslash (K * u),
# and each “matrix–vector” multiply (K\cdot) is just a Gaussian blur (separable 1D convs) or an FFT convolution. That drops the per-iteration cost from O(n^2) to ~O(n\,k) (finite support kernel) or O(n\log n) (FFT), with n=H\!\times\!W.
# --- Gaussian blur (separable, reflect padding) ---
def _gauss1d(sigma, radius=5):
    x = jnp.arange(-radius, radius + 1, dtype=jnp.float32)
    k = jnp.exp(-(x**2) / (2 * sigma**2))
    k = k / jnp.sum(k)
    return k.reshape(-1)  # Ensure 1D


def _conv1d_reflect(x, k, axis):
    pad = (k.shape[0] // 2, k.shape[0] // 2)
    pads = [(0, 0)] * x.ndim
    pads[axis] = pad
    xpad = jnp.pad(x, pads, mode="reflect")
    # lax.conv on 1D by reshaping to NCHW and using a 1D kernel in OIHW format
    if axis == -1 or axis == x.ndim - 1:
        w = k[None, None, None, :]  # OIHW: (1, 1, 1, kernel_size)
        xNCHW = xpad[None, None, :, :]
        y = lax.conv_general_dilated(
            xNCHW,
            w,
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
        return y[0, 0]
    else:
        x = jnp.swapaxes(xpad, axis, -1)
        w = k[None, None, None, :]  # OIHW: (1, 1, 1, kernel_size)
        xNCHW = x[None, None, :, :]
        y = lax.conv_general_dilated(
            xNCHW,
            w,
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )[0, 0]
        return jnp.swapaxes(y, -1, axis)


def _apply_gauss_blur2d(x, kernel):
    x = _conv1d_reflect(x, kernel, axis=-1)  # blur along W
    x = _conv1d_reflect(x, kernel, axis=-2)  # blur along H
    return x


def gauss_blur2d(x, sigma):
    k = _gauss1d(sigma)
    return _apply_gauss_blur2d(x, k)


# --- Convolutional Sinkhorn (balanced) on grid masses a,b in R^{H×W} ---
def sinkhorn_divergence_conv(a, b, eps, n_iters=80, tol=1e-6):
    """a,b: nonnegative (H,W); eps>0; returns S_epsilon(a,b) >= 0."""
    sigma = jnp.sqrt(eps / 2.0)
    a = a.astype(jnp.float32) + 1e-12
    b = b.astype(jnp.float32) + 1e-12

    # normalize to probability distributions (sum to 1)
    a_sum = jnp.maximum(a.sum(), 1e-10)
    b_sum = jnp.maximum(b.sum(), 1e-10)
    a = a / a_sum
    b = b / b_sum

    u = jnp.ones_like(a)
    v = jnp.ones_like(b)

    gauss_kernel = _gauss1d(sigma)

    def body(carry):
        u, v = carry
        Kv = _apply_gauss_blur2d(v, gauss_kernel)
        u = a / (Kv + 1e-12)
        Ku = _apply_gauss_blur2d(u, gauss_kernel)
        v = b / (Ku + 1e-12)
        return (u, v)

    def cond(carry_prev, carry_next):
        u0, v0 = carry_prev
        u1, v1 = carry_next
        du = jnp.max(jnp.abs(u1 - u0)) / (jnp.max(jnp.abs(u1)) + 1e-12)
        dv = jnp.max(jnp.abs(v1 - v0)) / (jnp.max(jnp.abs(v1)) + 1e-12)
        return jnp.maximum(du, dv) > tol

    # Run fixed iters; you can add an early-stop check if desired
    def loop(carry, _):
        return body(carry), None

    (u, v), _ = lax.scan(loop, (u, v), None, length=n_iters)

    # Dual potentials (balanced): f=eps*log u, g=eps*log v
    u_safe = jnp.maximum(u, 1e-12)
    v_safe = jnp.maximum(v, 1e-12)
    f = eps * jnp.log(u_safe)
    g = eps * jnp.log(v_safe)

    def ot_value(a_, b_):
        # Regularized OT value: eps * ( <a_, log u_> + <b_, log v_> )
        return jnp.sum(a_ * jnp.log(u_safe)) * eps + jnp.sum(b_ * jnp.log(v_safe)) * eps

    # Compute S_eps(a,b) = OT(a,b) - 0.5(OT(a,a)+OT(b,b))
    ot_ab = ot_value(a, b)

    # Self terms: re-run quickly with (a,a) and (b,b)
    def self_term(m):
        u = jnp.ones_like(m)
        v = jnp.ones_like(m)

        def loop_uv(carry, _):
            u, v = carry
            Kv = gauss_blur2d(v, sigma)
            u = m / (Kv + 1e-12)
            Ku = gauss_blur2d(u, sigma)
            v = m / (Ku + 1e-12)
            return (u, v), None

        (u, v), _ = lax.scan(loop_uv, (u, v), None, length=max(20, n_iters // 2))
        u_safe = jnp.maximum(u, 1e-12)
        v_safe = jnp.maximum(v, 1e-12)
        return eps * (jnp.sum(m * jnp.log(u_safe)) + jnp.sum(m * jnp.log(v_safe)))

    ot_aa = self_term(a)
    ot_bb = self_term(b)

    return jnp.maximum(0.0, ot_ab - 0.5 * (ot_aa + ot_bb))  # clamp tiny negatives from numerics


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                      --     loss functions     --


def proj_nonneg_ste(z, leak=1e-3, cap=None):
    # forward: hard clip to [0, cap]; backward: small slope < 0 so negatives move
    z_clip = jnp.clip(z, 0.0, cap) if cap is not None else jnp.maximum(z, 0.0)
    z_leaky = jnp.where(z >= 0.0, z, leak * z)
    return z_clip + jax.lax.stop_gradient(z_leaky - z_clip)


def _epsilon_from_x_median(xn, eps=1e-12):
    B = xn.shape[0]
    d2 = jnp.sum((xn[:, None, :] - xn[None, :, :]) ** 2, axis=-1)
    big = 1e9
    nn_sq = jnp.min(d2 + jnp.eye(B) * big, axis=1)  # exclude self
    med = jnp.median(nn_sq)
    e = 0.5 * med  # ≈ exp(-1) at NN distance
    return jax.lax.stop_gradient(jnp.maximum(e, eps))


def sinkhorn_divergence_unbalanced(
    x, y, yhat, epsilon=0.01, tau=0.9, cap=0.5, threshold=1e-3, max_iterations=300, **solver_kwargs
):
    from ott.geometry import pointcloud
    from ott.problems.linear import linear_problem
    from ott.solvers.linear import sinkhorn

    # sanitize & normalize coords
    x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = jnp.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    yh = jnp.nan_to_num(yhat, nan=0.0, posinf=0.0, neginf=0.0)

    x_mu = jnp.mean(x, 0, keepdims=True)
    x_sd = jnp.std(x, 0, keepdims=True) + 1e-8
    xn = (x - x_mu) / x_sd

    if epsilon is None:
        epsilon = _epsilon_from_x_median(xn)

    # forward masses are nonnegative; grads flow via STE
    a = _proj_nonneg_ste(yh, cap=cap)
    b = jnp.clip(y, 0.0, cap)  # target can be hard-clipped

    geom = pointcloud.PointCloud(xn, xn, epsilon=epsilon)
    solver = sinkhorn.Sinkhorn(
        lse_mode=True, threshold=threshold, max_iterations=max_iterations, **solver_kwargs
    )

    def ot(u, v):
        prob = linear_problem.LinearProblem(geom, a=u, b=v, tau_a=tau, tau_b=tau)
        return solver(prob).reg_ot_cost

    div = ot(a, b) - 0.5 * (ot(a, a) + ot(b, b))
    # guard the odd tiny negative or NaN
    return jnp.maximum(jnp.nan_to_num(div, nan=0.0, posinf=0.0, neginf=0.0), 0.0)


def sinkhorn_divergence_balanced(
    x, y, yhat, epsilon=0.01, cap=None, mass_floor=1e-8, lambda_neg=1e-4, **solver_kwargs
):
    from ott.geometry import pointcloud
    from ott.problems.linear import linear_problem
    from ott.solvers.linear import sinkhorn

    # coords normalization
    x_mu, x_sd = jnp.mean(x, 0, keepdims=True), jnp.std(x, 0, keepdims=True) + 1e-8
    xn = (x - x_mu) / x_sd
    if epsilon is None:
        diffs = xn[:, None, :] - xn[None, :, :]
        epsilon = 0.03 * jnp.mean(jnp.sum(diffs**2, -1))

    a = proj_nonneg_ste(yhat, cap=cap)
    b = proj_nonneg_ste(y, cap=cap)

    # --- tiny uniform floor to avoid zero-sum marginals ---
    n = a.size
    # scale the floor to total mass scale of b (or 1.0 if b is zero)
    ref = jnp.maximum(jnp.sum(b), 1.0)
    floor = (mass_floor * ref) / n
    a = a + floor
    b = b + floor

    # balance total mass safely
    sum_a = jnp.sum(a)
    sum_b = jnp.sum(b)
    a = a * (sum_b / (sum_a + 1e-8))

    geom = pointcloud.PointCloud(xn, xn, epsilon=epsilon)
    solver = sinkhorn.Sinkhorn(lse_mode=True, **solver_kwargs)

    def ot(u, v):
        prob = linear_problem.LinearProblem(geom, a=u, b=v)  # tau=1
        return solver(prob).reg_ot_cost

    div = ot(a, b) - 0.5 * (ot(a, a) + ot(b, b))
    neg_pen = lambda_neg * jnp.mean(jax.nn.relu(-yhat))
    return jnp.where(jnp.isfinite(div), div, 0.0) + neg_pen


def zncc_loss(x, y, yhat, eps=1e-6, **kw):
    y0 = y - jnp.mean(y)
    yhat0 = yhat - jnp.mean(yhat)
    num = jnp.mean(y0 * yhat0)

    yyhat = jnp.mean(y0**2) * jnp.mean(yhat0**2)
    yyhat = jnp.maximum(yyhat, eps)

    den = jnp.sqrt(yyhat) + eps

    return 1.0 - (num / den)


def wasserstein_zncc_loss(x, y, yhat, zncc_weight=0.4, **kw):
    wloss = sinkhorn_divergence_balanced(x, y, yhat, **kw)
    zloss = zncc_loss(x, y, yhat)
    return zncc_weight * zloss + (1 - zncc_weight) * wloss


def huber_loss(x, y, yhat, delta=0.01, **kw):
    r = yhat - y
    abs_r = jnp.abs(r)
    quad = 0.5 * (r**2)
    lin = delta * (abs_r - 0.5 * delta)
    return jnp.mean(jnp.where(abs_r <= delta, quad, lin))


def huber_zncc_loss(x, y, yhat, delta=0.01, zncc_weight=0.1, **kw):
    return zncc_weight * zncc_loss(x, y, yhat) + (1 - zncc_weight) * huber_loss(
        x, y, yhat, delta=delta
    )


def spectral_loss(x, y, yhat, **kwargs):
    Y = jnp.fft.fft2(y)
    Yh = jnp.fft.fft2(yhat)
    return jnp.mean((jnp.abs(Y) - jnp.abs(Yh)) ** 2)


def mse_loss(x: jnp.ndarray, y: jnp.ndarray, yhat: jnp.ndarray, **kwargs):
    diff = yhat - y
    return jnp.mean(diff**2)


@Partial(jax.jit, static_argnames=["lossfunc", "n_inputs_per_network"])
def compute_all_losses(x, y, yhatdep, lossfunc: Callable, n_inputs_per_network=2):
    n_networks = int(x.shape[-1] / n_inputs_per_network)
    n_inputs = n_networks * n_inputs_per_network

    batch_size = y.shape[0]
    n_targets = y.shape[1]

    # shape is (batch_size, n_targets, n_outputs)
    assert_that(x).has_shape((y.shape[0], y.shape[1], n_inputs))
    assert_that(yhatdep).has_shape((batch_size, n_targets, n_networks))
    assert_that(y).has_same_shape(yhatdep)

    # now we need to split the inputs into the different networks
    xsplit = jnp.reshape(x, (batch_size, n_targets, n_networks, n_inputs_per_network))

    per_target = vmap(lossfunc, in_axes=(1, 1, 1))
    losses = vmap(per_target, in_axes=(1, 1, 1))(xsplit, yhatdep, y)
    assert_that(losses).has_shape((n_targets, n_networks))

    return losses


def per_batch_apply(params, X, Z, keys, stack):
    return vmap(stack.apply, in_axes=(None, 0, 0, 0))(params, X, Z, keys)


def per_target_apply(params, X, Z, keys, stack):
    return vmap(Partial(per_batch_apply, stack=stack), in_axes=(0, 1, 1, 1), out_axes=1)(
        params, X, Z, keys
    )


@Partial(jit, static_argnames=["stack"])
def per_replicate_apply(params, X, Z, keys, stack):
    return vmap(Partial(per_target_apply, stack=stack))(params, X, Z, keys)


def soft_count_over_one_penalty(W, rel_active=1e-3, width=2e-4):
    """
    W: (n_aggregations, n_ratios)
    Treat entries >= rel_active * (row max) as 'active'.
    width controls how sharp the step is around rel_active.
    Penalty is zero for <=1 active per row, positive only for 2+.
    """
    A = jnp.abs(W)
    m = jnp.max(A, axis=1, keepdims=True)
    # scale-invariant normalization
    norm = jnp.where(m > 0, A / (m + 1e-12), 0.0)

    # soft indicator: ~0 below rel_active, ~1 above; sharpen with `width`
    logits = (norm - rel_active) / (width + 1e-12)
    soft_active = jax.nn.sigmoid(logits)
    soft_count = jnp.sum(soft_active, axis=1)  # ≈ number of actives in each row

    # penalize only the part above 1
    row_pen = jnp.square(jax.nn.relu(soft_count - 1.0))
    return jnp.sum(row_pen)


def lncc_grid_loss(x, y, yhat, k=7, eps=1e-6, **kw):
    def box2d(a, r):
        a = jnp.pad(a, ((r + 1, r), (r + 1, r)), mode="edge")
        s = jnp.cumsum(jnp.cumsum(a, axis=0), axis=1)
        return (
            s[: -2 * r - 1, : -2 * r - 1]
            - s[: -2 * r - 1, 2 * r + 1 :]
            - s[2 * r + 1 :, : -2 * r - 1]
            + s[2 * r + 1 :, 2 * r + 1 :]
        )

    r = k // 2
    N = k * k

    y0, y1 = y, yhat
    m0 = box2d(y0, r) / N
    m1 = box2d(y1, r) / N

    y0c = y0 - m0
    y1c = y1 - m1

    num = box2d(y0c * y1c, r)
    # eps inside sqrt to avoid infinite gradients at zero
    den = jnp.sqrt(jnp.maximum(box2d(y0c * y0c, r), 0) * jnp.maximum(box2d(y1c * y1c, r), 0) + eps)
    lncc = num / den
    lncc = jnp.clip(lncc, -1.0, 1.0)
    return 1.0 - jnp.nanmean(lncc)


def simse_loss(x, y, yhat, eps=1e-8, **kw):
    # center
    y0 = jnp.nan_to_num(y - jnp.mean(y), nan=0.0, posinf=0.0, neginf=0.0)
    yhat0 = jnp.nan_to_num(yhat - jnp.mean(yhat), nan=0.0, posinf=0.0, neginf=0.0)

    # variances (squared norms)
    vy = jnp.sum(y0**2)
    vyhat = jnp.sum(yhat0**2)

    # if prediction has ~zero variance, best α is 0 (avoid divide explosions)
    alpha = jnp.where(vyhat > eps, jnp.sum(y0 * yhat0) / (vyhat + eps), 0.0)

    # residual and normalized error
    resid = y0 - alpha * yhat0
    num = jnp.sum(resid**2)
    den = jnp.maximum(vy, eps)

    loss = num / den
    return jnp.nan_to_num(loss, nan=1.0, posinf=1.0, neginf=1.0)


def lncc_loss(
    x,
    y,
    yhat,
    target_neighbors=12,
    eps=1e-6,
    **kw,
):
    x = jnp.nan_to_num(jnp.asarray(x), nan=0.0, posinf=0.0, neginf=0.0)
    y = jnp.nan_to_num(jnp.asarray(y).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    yhat = jnp.nan_to_num(jnp.asarray(yhat).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)

    B = x.shape[0]
    if B <= 1:
        return jnp.array(0.0, dtype=x.dtype)

    diffs = x[:, None, :] - x[None, :, :]
    d2 = jnp.sum(diffs**2, axis=-1)

    # sigma from median NN distance
    big = 1e9
    nn_sq = jnp.min(d2 + jnp.eye(B) * big, axis=1)
    median = jnp.sqrt(jnp.maximum(jnp.median(nn_sq), 0.0) + eps)
    dim = x.shape[-1]
    sigma_scale = 0.5 * (target_neighbors ** (1.0 / dim))  # for 2D ~ 0.5*sqrt(k)
    sigma = jnp.maximum(sigma_scale * median, eps)

    K = jnp.exp(-d2 / (2.0 * (sigma**2) + eps))
    K = jnp.where(jnp.isfinite(K), K, 0.0)

    rowsum = jnp.sum(K, axis=1, keepdims=True)
    W = jnp.where(rowsum > 0.0, K / (rowsum + eps), 0.0)

    Ey = W @ y
    Eyhat = W @ yhat
    Ey2 = W @ (y * y)
    Eyhat2 = W @ (yhat * yhat)
    Ey_yhat = W @ (y * yhat)

    cov = Ey_yhat - Ey * Eyhat
    vy = jnp.maximum(Ey2 - Ey**2, 0.0)
    vyh = jnp.maximum(Eyhat2 - Eyhat**2, 0.0)

    den = jnp.sqrt(vy * vyh) + eps
    ncc = cov / den
    ncc = jnp.where(jnp.isfinite(ncc), ncc, 0.0)

    return 1.0 - jnp.mean(ncc)


def simse_lncc_loss(x, y, yhat, simse_weight=0.3, **kw):
    return simse_weight * simse_loss(x, y, yhat) + (1 - simse_weight) * lncc_loss(x, y, yhat)


def distance_loss(
    stack,
    dconf,
    dmanager,
    num_z,
    ratio_paths=None,
    epsilon=0.01,
    lambda_over1=0.001,
    distance_func=huber_zncc_loss,
):
    from ott.geometry import pointcloud
    from ott.solvers import linear

    n_targets = dmanager.n_targets
    n_networks = len(dmanager.networks)
    n_inputs = 2 * n_networks
    n_outputs = stack.get_nb_outputs()
    dep_mask = stack.get_dependent_output_mask()
    nb_dep = np.sum(dep_mask)

    def all_losses_func(x, target_y, yhat, epsilon):
        assert_that(x.shape).is_equal_to((dconf.batch_size, n_targets, n_inputs))
        assert_that(target_y).has_same_shape(yhat)
        assert_that(x.ndim).is_equal_to(3)
        all_losses = compute_all_losses(
            x,
            target_y,
            yhat,
            Partial(distance_func, epsilon=epsilon),
        )
        assert_that(all_losses).has_shape((n_targets, n_networks))
        return all_losses

    if ratio_paths is None:
        ratio_paths = []

    def loss_func(dynamic, static, X, Y, Z, key, step):
        params = ParameterTree.merge(dynamic, static)

        # shape of X: (batch_size, n_targets, n_infeatures)
        # yhat should be of shape (batch_size, n_targets, n_outfeatures)

        assert_that(X.shape).is_equal_to((dconf.batch_size, n_targets, n_inputs))
        assert_that(Y.shape).is_equal_to((dconf.batch_size, n_targets, n_networks))
        assert_that(Z.shape).is_equal_to((dconf.batch_size, n_targets, num_z[-1]))

        keys = jax.random.split(key, (X.shape[0], X.shape[1]))
        yhat, (apply_aux, full_output) = per_target_apply(params, X, Z, keys, stack)
        assert_that(yhat.shape).is_equal_to((*X.shape[:2], n_outputs))

        yhatdep = jnp.compress(dep_mask, yhat, axis=-1, size=nb_dep)
        assert_that(yhatdep.shape).is_equal_to(Y.shape)

        ratio_leaves = params.get_leaves_by_path(ratio_paths)
        lo1 = as_schedule(lambda_over1)(step)
        over1_penalty = lo1 * sum(
            soft_count_over_one_penalty(p.view() if hasattr(p, 'view') else p, rel_active=1e-3, width=2e-4)
            for p in ratio_leaves
        )

        all_losses = all_losses_func(X, Y, yhatdep, as_schedule(epsilon)(step))
        avgloss = all_losses.mean()
        aux = {"apply_aux": apply_aux, "all_losses": all_losses, "yhatdep": yhatdep}

        total_loss = avgloss + over1_penalty

        return total_loss, aux

    return loss_func


def grid_distance_loss(
    stack,
    dconf,
    dmanager,
    num_z,
    ratio_paths=None,
    w_sinkhorn=1.0,
    w_lncc=0.5,
    w_spectral=0.0,
    eps_sinkhorn=0.1,
    n_sinkhorn_iters=50,
    lncc_kernel=7,
    lambda_over1=0.001,
    distance_func=None,
    epsilon=None,
):
    assert dmanager.is_lattice_mode, "grid_distance_loss requires lattice sampling"
    resolution = dmanager.grid_resolution
    assert resolution is not None
    xres, yres = resolution

    n_targets = dmanager.n_targets
    n_networks = len(dmanager.networks)
    n_inputs = 2 * n_networks
    n_outputs = stack.get_nb_outputs()
    dep_mask = stack.get_dependent_output_mask()
    nb_dep = np.sum(dep_mask)

    if ratio_paths is None:
        ratio_paths = []

    def compute_grid_loss_single(y_img, yhat_img):
        loss = jnp.array(0.0)
        if w_sinkhorn > 0:
            y_pos = proj_nonneg_ste(y_img)
            yhat_pos = proj_nonneg_ste(yhat_img)
            loss = loss + w_sinkhorn * sinkhorn_divergence_conv(
                yhat_pos, y_pos, eps_sinkhorn, n_iters=n_sinkhorn_iters
            )
        if w_lncc > 0:
            loss = loss + w_lncc * lncc_grid_loss(None, y_img, yhat_img, k=lncc_kernel)
        if w_spectral > 0:
            loss = loss + w_spectral * spectral_loss(None, y_img, yhat_img)
        return loss

    def compute_all_grid_losses(Y_images, yhat_images):
        # Y_images: (n_targets, n_networks, yres, xres)
        # yhat_images: (n_targets, n_networks, yres, xres)
        per_net = vmap(compute_grid_loss_single)  # over networks
        per_target = vmap(per_net)  # over targets
        return per_target(Y_images, yhat_images)  # (n_targets, n_networks)

    def loss_func(dynamic, static, X, Y, Z, key, step):
        params = ParameterTree.merge(dynamic, static)

        batch_size = yres * xres
        assert_that(X.shape).is_equal_to((batch_size, n_targets, n_inputs))
        assert_that(Y.shape).is_equal_to((batch_size, n_targets, 1))
        assert_that(Z.shape).is_equal_to((batch_size, n_targets, num_z[-1]))

        Y_images = Y.squeeze(-1).transpose(1, 0).reshape(n_targets, yres, xres)
        Y_images = jnp.tile(Y_images[:, None, :, :], (1, n_networks, 1, 1))

        keys = jax.random.split(key, (X.shape[0], X.shape[1]))
        yhat_flat, (apply_aux, full_output) = per_target_apply(params, X, Z, keys, stack)
        assert_that(yhat_flat.shape).is_equal_to((batch_size, n_targets, n_outputs))

        yhatdep_flat = jnp.compress(dep_mask, yhat_flat, axis=-1, size=nb_dep)
        assert_that(yhatdep_flat.shape).is_equal_to((batch_size, n_targets, n_networks))

        # reshape predictions to images: (batch_size, n_targets, n_networks) -> (n_targets, n_networks, yres, xres)
        yhat_images = yhatdep_flat.transpose(1, 2, 0).reshape(n_targets, n_networks, yres, xres)

        ratio_leaves = params.get_leaves_by_path(ratio_paths)
        lo1 = as_schedule(lambda_over1)(step)
        over1_penalty = lo1 * sum(
            soft_count_over_one_penalty(p.view() if hasattr(p, 'view') else p, rel_active=1e-3, width=2e-4)
            for p in ratio_leaves
        )

        all_losses = compute_all_grid_losses(Y_images, yhat_images)
        avgloss = all_losses.mean()
        aux = {"apply_aux": apply_aux, "all_losses": all_losses, "yhat_images": yhat_images}

        total_loss = avgloss + over1_penalty

        return total_loss, aux

    return loss_func


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                     --     helper functions     --


def get_ind_params(params, target_id, ind_id):
    return jax.tree.map(lambda x: x[target_id, ind_id], params)


def plot_prediction(
    design_config,
    params,
    target_id,
    ind_id,
    net_id,
    target_x,
    target_y,
    key,
    stack,
    num_z,
    dep_output_mask,
    max_evals=30000,
):
    """Plot the prediction for a given target and individual."""
    import matplotlib.pyplot as plt

    params_ind = get_ind_params(params, target_id, ind_id)
    t_x = target_x[:, target_id]
    t_x = t_x.reshape(-1, t_x.shape[-1])[:max_evals]
    t_y = target_y[:, target_id]
    t_y = t_y.reshape(-1, t_y.shape[-1])[:max_evals]

    z = jax.random.uniform(key, (*t_x.shape[:-1], num_z))

    t_yhat = design_config.forward(params_ind, t_x, z, key, stack)
    t_yhatdep = t_yhat[..., dep_output_mask]
    assert_that(t_yhatdep.shape).is_equal_to(t_y.shape)

    loss_value = single_l2loss(t_yhatdep[:, net_id], t_y[:, net_id])

    t_x_net = t_x[:, 2 * net_id : 2 * net_id + 2]

    # 2 subplots (ground truth and prediction)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    scatter1 = ax1.scatter(
        t_x_net[:, 0], t_x_net[:, 1], c=t_y[:, net_id], s=1, cmap="viridis", vmin=0, vmax=1
    )
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Ground Truth")
    ax1.set_aspect("equal")
    plt.colorbar(scatter1, ax=ax1)
    scatter2 = ax2.scatter(
        t_x_net[:, 0], t_x_net[:, 1], c=t_yhatdep[:, net_id], s=1, cmap="viridis", vmin=0, vmax=1
    )
    ax2.set_title("Prediction")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_aspect("equal")
    plt.colorbar(scatter2, ax=ax2)
    ax2.set_title(f"Prediction (Loss: {loss_value:.4f})")
    plt.tight_layout()


##────────────────────────────────────────────────────────────────────────────}}}


## {{{                          --     design manager   --
DEFAULT_RESCALE_TARGET = {
    "x": (0.0, 0.5),
    "y": (0.0, 0.5),
    "out": (0.09, 0.42),
}


class SamplingConfig(ArbitraryModel):
    strategy: Literal["uniform", "lattice"] = "uniform"


class UniformSampling(SamplingConfig):
    strategy: Literal["uniform"] = "uniform"
    n_samples: int = 5000


class LatticeSampling(SamplingConfig):
    strategy: Literal["lattice"] = "lattice"
    resolution: tuple[int, int] = (64, 64)
    jitter_std: float = 0.0


SamplingConfigUnion = Union[UniformSampling, LatticeSampling]


class Target(BaseModel):
    path: Union[str, Path]
    name: Optional[str] = None
    rescale_to: dict = DEFAULT_RESCALE_TARGET
    xlim: tuple[float, float] = (0.0, 1.0)
    ylim: tuple[float, float] = (0.0, 1.0)
    outlim: tuple[float, float] = (0.0, 1.0)
    transform_to_log_space: bool = False
    max_is_black: bool = True


class DesignManager(BaseModel):
    """Handles loading and sampling of 2d design target data."""

    targets: list[Target]
    networks: List[Network]
    sampling: SamplingConfigUnion = Field(default_factory=UniformSampling, discriminator="strategy")

    def model_post_init(self, *a, **kw):
        super().model_post_init(*a, **kw)
        for target in self.targets:
            if target.transform_to_log_space:
                target.xlim = (0.1, 1)
                target.ylim = (0.1, 1)

    def get_samples(
        self,
        samples: int | tuple[int, ...],
        seed: Optional[int | ArrayLike] = None,
    ) -> tuple[jax.Array, jax.Array]:
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        elif isinstance(seed, ArrayLike):
            seed = int(jax.random.randint(seed, (), 0, jnp.iinfo(jnp.int32).max))

        if isinstance(self.sampling, LatticeSampling):
            return self._get_lattice_samples(samples, seed)
        return self._get_uniform_samples(samples, seed)

    def _get_uniform_samples(
        self,
        samples: int | tuple[int, ...],
        seed: int,
    ) -> tuple[list[jax.Array], list[jax.Array]]:
        if isinstance(samples, int):
            samples = (samples,)

        n_networks = samples[0]
        requested_shape = samples[1:]
        n = int(np.prod(requested_shape))

        all_xsamples, all_ysamples = [], []

        for _ in range(n_networks):
            xsamples, ysamples = [], []
            for target in self.targets:
                xsample, ysample = sample_from_svg(
                    target.path,
                    n=n,
                    seed=seed,
                    log=target.transform_to_log_space,
                    xlim=target.xlim,
                    ylim=target.ylim,
                    outlim=target.outlim,
                    rescale_to=target.rescale_to,
                    max_is_black=target.max_is_black,
                )
                xsamples.append(xsample)
                ysamples.append(ysample)

            xsamples = jnp.stack(xsamples, axis=1)
            ysamples = jnp.stack(ysamples, axis=1)

            assert_that(xsamples.shape).is_equal_to((n, len(self.targets), 2))
            assert_that(ysamples.shape).is_equal_to((n, len(self.targets), 1))

            xsamples = xsamples.reshape(*requested_shape, len(self.targets), 2)
            ysamples = ysamples.reshape(*requested_shape, len(self.targets), 1)

            all_xsamples.append(xsamples)
            all_ysamples.append(ysamples)

        return all_xsamples, all_ysamples

    def _get_lattice_samples(
        self,
        samples: int | tuple[int, ...],
        seed: int,
    ) -> tuple[list[jax.Array], list[jax.Array]]:
        assert isinstance(self.sampling, LatticeSampling)
        xres, yres = self.sampling.resolution
        jitter = self.sampling.jitter_std

        if isinstance(samples, int):
            samples = (samples,)

        n_networks = samples[0]
        requested_shape = samples[1:]
        n = int(np.prod(requested_shape))

        all_xsamples, all_ysamples = [], []

        for _ in range(n_networks):
            xsamples, ysamples = [], []
            for target in self.targets:
                X, Y_grid = sample_from_svg(
                    target.path,
                    n=n,
                    seed=seed,
                    log=target.transform_to_log_space,
                    xlim=target.xlim,
                    ylim=target.ylim,
                    outlim=target.outlim,
                    rescale_to=target.rescale_to,
                    max_is_black=target.max_is_black,
                    grid=(xres, yres),
                    grid_jitter_std=jitter,
                )
                xsamples.append(X)
                ysamples.append(Y_grid)

            xsamples = jnp.stack(xsamples, axis=1)
            ysamples = jnp.stack(ysamples, axis=1)

            n_pts = n * yres * xres
            assert_that(xsamples.shape).is_equal_to((n_pts, len(self.targets), 2))
            assert_that(ysamples.shape).is_equal_to((n, len(self.targets), yres, xres))

            ysamples_flat = ysamples.transpose(0, 2, 3, 1).reshape(n_pts, len(self.targets), 1)

            new_batch_shape = requested_shape[:-1] + (requested_shape[-1] * yres * xres,)
            xsamples = xsamples.reshape(*new_batch_shape, len(self.targets), 2)
            ysamples_flat = ysamples_flat.reshape(*new_batch_shape, len(self.targets), 1)

            all_xsamples.append(xsamples)
            all_ysamples.append(ysamples_flat)

        return all_xsamples, all_ysamples

    @property
    def is_lattice_mode(self) -> bool:
        return isinstance(self.sampling, LatticeSampling)

    @property
    def grid_resolution(self) -> Optional[tuple[int, int]]:
        if isinstance(self.sampling, LatticeSampling):
            return self.sampling.resolution
        return None

    def build_stack(self, model: BiocompModel, unlock_ratios=True):
        logger.info(f"Building stack with {len(self.networks)} design networks")
        logger.info(f"Design network names: {[n.name for n in self.networks]}")
        stack = ComputeStack(networks=self.networks)
        logger.info(f"Stack after creation has {len(stack.networks)} networks")
        if unlock_ratios:
            assert model.compute_config is not None
            assert model.compute_config.node_functions is not None

            model.compute_config.node_functions["aggregation"] = encode_function(
                partial(nd.aggregation, random_init=True)
            )

        stack.build(model.compute_config)
        logger.info(
            f"Stack built: {stack.get_nb_networks()} networks, "
            f"{stack.get_nb_inputs()} inputs, {stack.get_nb_outputs()} outputs"
        )
        logger.info(f"Stack network names after build: {[n.name for n in stack.networks]}")
        return stack

    @property
    def n_targets(self):
        return len(self.targets)


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                   --     param initialization     --


def initialize_params(stack, n_replicates, n_targets, shared_params, key):
    # could be faster if we stacked copies of the shared parameters and did the merge on the whole stack...
    # good enough for now
    def init_single(k):
        params = stack.init(k)
        _, nonshared = params.filter_by_tag(["shared"])
        return ParameterTree.merge(shared_params, nonshared)

    def init_target_params(k):
        params = vmap(init_single)(jax.random.split(k, n_targets))
        return params

    return vmap(init_target_params)(jax.random.split(key, n_replicates))


class DesignConfig(OptimConfig):
    loss_function: EncodedPartialFunction = Field(default=distance_loss)
    n_replicates: int = 4


##────────────────────────────────────────────────────────────────────────────}}}


def assert_tree_shape(tree, expected_shape, only_first_dims=True):
    """Assert that the shape of each leaf in the tree matches the expected shape."""

    N_DIMS = len(expected_shape)

    def check_shape(x):
        if isinstance(x, jax.Array):
            assert_that(x.shape[:N_DIMS] if only_first_dims else x.shape).is_equal_to(
                expected_shape
            )
        return x

    jax.tree.map(check_shape, tree)


## {{{                   --     evaluation and analysis     --


def sample_for_evaluation(
    dmanager: DesignManager,
    dconf: DesignConfig,
    final_params: ParameterTree,
    n_eval_samples: int = 5000,
    key: Optional[ArrayLike] = None,
) -> Tuple[jax.Array, jax.Array]:
    """Sample data for evaluation of trained design parameters.

    Returns:
        xraw: shape (n_networks, n_replicates, n_eval_samples, n_targets, 2)
        yraw: shape (n_networks, n_replicates, n_eval_samples, n_targets, 1)
    """
    if key is None:
        key = jax.random.key(0)

    n_networks = len(dmanager.networks)

    # sample evaluation data (use uniform sampling for evaluation, not lattice)
    original_sampling = dmanager.sampling
    dmanager.sampling = UniformSampling()
    xraw_list, yraw_list = dmanager.get_samples((n_networks, dconf.n_replicates, n_eval_samples), key)
    dmanager.sampling = original_sampling
    xraw = jnp.stack(xraw_list, axis=0)
    yraw = jnp.stack(yraw_list, axis=0)

    # assertions
    assert_that(xraw).has_shape(
        (n_networks, dconf.n_replicates, n_eval_samples, dmanager.n_targets, 2)
    )
    assert_that(yraw).has_shape(
        (n_networks, dconf.n_replicates, n_eval_samples, dmanager.n_targets, 1)
    )

    return xraw, yraw


def evaluate_design(
    dmanager: DesignManager,
    dconf: DesignConfig,
    model: BiocompModel,
    final_params: ParameterTree,
    xraw: jax.Array,
    yraw: jax.Array,
    key: Optional[ArrayLike] = None,
    max_eval_size: int = 1000,
    max_loss_size: int = 128,
) -> Tuple[jax.Array, jax.Array]:
    """Evaluate design performance on sampled data.

    Args:
        dmanager: Design manager with networks and targets
        dconf: Design configuration
        model: Trained biocomp model
        final_params: Final optimized parameters
        xraw: Input samples shape (n_networks, n_replicates, n_eval_samples, n_targets, 2)
        yraw: Target samples shape (n_networks, n_replicates, n_eval_samples, n_targets, 1)
        key: JAX random key
        max_eval_size: Maximum number of samples to process at once (for memory efficiency)

    Returns:
        yhatdep: Predictions shape (n_replicates, n_eval_samples, n_targets, n_networks)
        losses: Loss values shape (n_replicates, n_targets, n_networks)
    """
    if key is None:
        key = jax.random.key(0)

    n_networks = len(dmanager.networks)
    n_eval_samples = xraw.shape[2]

    # reshape inputs for batch processing
    X = jnp.concatenate(xraw, axis=-1)
    Y = jnp.concatenate(yraw, axis=-1)
    assert_that(X).has_shape(
        (dconf.n_replicates, n_eval_samples, dmanager.n_targets, 2 * n_networks)
    )
    assert_that(Y).has_shape((dconf.n_replicates, n_eval_samples, dmanager.n_targets, n_networks))

    # get quantile variable size
    num_z = int(final_params["global/number_of_random_variables"].ravel()[0])

    logger.info(
        f"Evaluating design with {dconf.n_replicates} replicates, "
        f"{n_eval_samples} samples, {dmanager.n_targets} targets, "
        f"{n_networks} networks, {num_z} quantile variables."
    )
    # build stack once
    stack = dmanager.build_stack(model)
    n_outputs = stack.get_nb_outputs()
    dep_mask = stack.get_dependent_output_mask()

    # determine chunk size
    chunk_size = min(max_eval_size, n_eval_samples)
    n_chunks = (n_eval_samples + chunk_size - 1) // chunk_size  # ceiling division

    # process in chunks if needed
    if n_chunks > 1:
        # we'll accumulate predictions in chunks
        yhatdep_chunks = []

        for chunk_idx in tqdm(range(n_chunks), desc="Processing chunks"):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, n_eval_samples)
            actual_chunk_size = end_idx - start_idx

            # slice the data for this chunk
            X_chunk = X[:, start_idx:end_idx]
            Y_chunk = Y[:, start_idx:end_idx]

            # generate random quantile variables for this chunk
            z_shape = (dconf.n_replicates, actual_chunk_size, dmanager.n_targets, num_z)
            Z_chunk = jax.random.uniform(jax.random.fold_in(key, chunk_idx), z_shape)

            # apply model on chunk
            chunk_keys = jax.random.split(
                jax.random.fold_in(key, chunk_idx + 1000), X_chunk.shape[:-1]
            )
            YHAT_chunk, _ = per_replicate_apply(final_params, X_chunk, Z_chunk, chunk_keys, stack)
            assert_that(YHAT_chunk).has_shape(
                (dconf.n_replicates, actual_chunk_size, dmanager.n_targets, n_outputs)
            )

            # extract dependent outputs for chunk
            yhatdep_chunk = jnp.compress(dep_mask, YHAT_chunk, axis=-1, size=sum(dep_mask))
            assert_that(yhatdep_chunk).has_shape(
                (dconf.n_replicates, actual_chunk_size, dmanager.n_targets, n_networks)
            )

            yhatdep_chunks.append(yhatdep_chunk)

        # concatenate all chunks
        yhatdep = jnp.concatenate(yhatdep_chunks, axis=1)
        assert_that(yhatdep).has_shape(
            (dconf.n_replicates, n_eval_samples, dmanager.n_targets, n_networks)
        )

    else:
        # process all at once (original implementation)
        z_shape = (dconf.n_replicates, n_eval_samples, dmanager.n_targets, num_z)
        Z = jax.random.uniform(key, z_shape)

        YHAT, _ = per_replicate_apply(
            final_params, X, Z, jax.random.split(key, X.shape[:-1]), stack
        )
        assert_that(YHAT).has_shape(
            (dconf.n_replicates, n_eval_samples, dmanager.n_targets, n_outputs)
        )

        yhatdep = np.compress(dep_mask, YHAT, axis=-1)
        assert_that(yhatdep).has_shape(
            (dconf.n_replicates, n_eval_samples, dmanager.n_targets, n_networks)
        )

    # compute losses (this should be relatively lightweight)
    loss_func = dconf.loss_function.kwargs.get("distance_func", huber_zncc_loss)

    # Handle PartialFunction objects
    if hasattr(loss_func, "get_impl"):
        loss_func = loss_func.get_impl()

    all_losses_chunks = []
    avg_over_n_losses = max(1, n_eval_samples // max_loss_size)
    for i in tqdm(list(range(avg_over_n_losses)), desc="Computing losses"):
        indices = jax.random.choice(key, n_eval_samples, shape=(max_loss_size,), replace=True)
        lX = X[:, indices]
        lY = Y[:, indices]
        lyhatdep = yhatdep[:, indices]
        losses = vmap(Partial(compute_all_losses, lossfunc=loss_func))(lX, lY, lyhatdep)
        all_losses_chunks.append(np.asarray(losses))

    losses = np.mean(np.stack(all_losses_chunks, axis=0), axis=0)
    assert_that(losses).has_shape((dconf.n_replicates, dmanager.n_targets, n_networks))
    logger.debug(f"Computed losses shape: {losses.shape}")

    return yhatdep, losses


def get_topk_replicate_network_pairs(
    losses: jax.Array,
    dmanager: DesignManager,
    dconf: DesignConfig,
    k: int = 1,
) -> List[List[Tuple[int, int, float]]]:
    """Find top-k replicate/network pairs with lowest loss for each target.

    Args:
        losses: Loss values shape (n_replicates, n_targets, n_networks)
        dmanager: Design manager with networks and targets
        dconf: Design configuration
        k: Number of top pairs to return per target

    Returns:
        List of lists, one per target, each containing k tuples of (replicate_id, network_id, loss_value)
    """
    n_replicates, n_targets, n_networks = losses.shape
    assert_that(n_replicates).is_equal_to(dconf.n_replicates)
    assert_that(n_targets).is_equal_to(dmanager.n_targets)
    assert_that(n_networks).is_equal_to(len(dmanager.networks))
    k = min(k, n_replicates * n_networks)

    best_per_target = []
    for tid in range(n_targets):
        tlosses = losses[:, tid, :]  # shape: (n_replicates, n_networks)
        flat_tlosses = tlosses.reshape((-1,))  # shape: (n_replicates * n_networks,)
        topk_flat_idx = jnp.argsort(flat_tlosses)[:k]

        # convert flat indices back to (replicate_id, network_id)
        rep_ids, net_ids = jnp.unravel_index(topk_flat_idx, (n_replicates, n_networks))
        topk_pairs = [
            (int(rep_ids[j]), int(net_ids[j]), float(flat_tlosses[topk_flat_idx[j]]))
            for j in range(k)
        ]
        best_per_target.append(topk_pairs)

    return best_per_target


def plot_design_results(
    dmanager: DesignManager,
    dconf: DesignConfig,
    xraw: jax.Array,
    yraw: jax.Array,
    topk: List[List[Tuple[int, int, float]]],
    yhatdep: Optional[jax.Array] = None,
    n_eval_samples: Optional[int] = None,
    save_dir: Optional[Path] = None,
    show_difference: bool = False,
    plot_top_k: Optional[int] = None,
) -> None:
    """Plot design results for each target showing best replicate/network combination.

    Args:
        dmanager: Design manager with networks and targets
        dconf: Design configuration
        xraw: Input samples shape (n_networks, n_replicates, n_eval_samples, n_targets, 2)
        yraw: Target samples shape (n_networks, n_replicates, n_eval_samples, n_targets, 1)
        yhatdep: Predictions shape (n_replicates, n_eval_samples, n_targets, n_networks)
        topk: Top-k results from get_topk_replicate_network_pairs
        n_eval_samples: Maximum number of samples to plot (for performance)
        save_dir: Directory to save figures (if None, just display)
        show_difference: Whether to show difference plots between prediction and target
        plot_top_k: Number of top-k designs to plot per target (default: 1, i.e., just the best)
    """
    import matplotlib.pyplot as plt
    from biocomp.plotting.plotting_core import DEFAULT_CMAP_NAME

    if n_eval_samples is None:
        n_eval_samples = xraw.shape[2]
    else:
        n_eval_samples = min(n_eval_samples, xraw.shape[2])

    # validate shapes
    n_networks = len(dmanager.networks)
    assert_that(xraw).has_shape(
        (n_networks, dconf.n_replicates, xraw.shape[2], dmanager.n_targets, 2)
    )
    assert_that(yraw).has_shape(
        (n_networks, dconf.n_replicates, yraw.shape[2], dmanager.n_targets, 1)
    )

    # determine how many top-k results to plot
    if plot_top_k is None:
        plot_top_k = 1  # default to just the best result

    for tid, target in enumerate(dmanager.targets):
        # plot multiple top-k results for this target
        n_to_plot = min(plot_top_k, len(topk[tid]))

        for rank in range(n_to_plot):
            rep_id, net_id, loss_val = topk[tid][rank]

            # get data for this specific target/network/replicate combo
            x_target = xraw[net_id, rep_id, :n_eval_samples, tid]  # shape: (n_samples, 2)
            y_target = yraw[net_id, rep_id, :n_eval_samples, tid, 0]  # squeeze last dim

            # assertions
            assert_that(x_target).has_shape((n_eval_samples, 2))
            assert_that(y_target).has_shape((n_eval_samples,))

            # create figure
            nax = 3 if show_difference else 2
            fig, axes = plt.subplots(1, nax, figsize=(nax * 5, 5), dpi=100)

            # ground truth
            sc1 = axes[0].scatter(
                x_target[:, 0], x_target[:, 1], c=y_target, cmap=DEFAULT_CMAP_NAME, s=5, alpha=0.7
            )
            axes[0].set_title("Target")
            axes[0].set_aspect("equal")
            plt.colorbar(sc1, ax=axes[0])

            if yhatdep is not None:
                # prediction
                yhat_target = yhatdep[rep_id, :n_eval_samples, tid, net_id]  # shape: (n_samples,)
                assert_that(yhat_target).has_shape((n_eval_samples,))
                assert_that(yhatdep).has_shape(
                    (dconf.n_replicates, yhatdep.shape[1], dmanager.n_targets, n_networks)
                )
                sc2 = axes[1].scatter(
                    x_target[:, 0],
                    x_target[:, 1],
                    c=yhat_target,
                    cmap=DEFAULT_CMAP_NAME,
                    s=5,
                    alpha=0.7,
                )
                axes[1].set_title(f"Prediction (rank {rank + 1}, loss={loss_val:.4f})")
                axes[1].set_aspect("equal")
                plt.colorbar(sc2, ax=axes[1])

                # difference
                if show_difference:
                    diff = yhat_target - y_target
                    assert_that(diff).has_shape((n_eval_samples,))
                    vmax = jnp.abs(diff).max()
                    sc3 = axes[2].scatter(
                        x_target[:, 0],
                        x_target[:, 1],
                        c=diff,
                        cmap="RdBu_r",
                        s=5,
                        alpha=0.7,
                        vmin=-vmax,
                        vmax=vmax,
                    )
                    axes[2].set_title(f"Difference (net: {dmanager.networks[net_id].name})")
                    axes[2].set_aspect("equal")
                    plt.colorbar(sc3, ax=axes[2])

            plt.suptitle(
                f"Target: {target.name} | Rank {rank + 1}: net {dmanager.networks[net_id].name})"
            )
            plt.tight_layout()

            if save_dir:
                # Use rank as prefix for consistency with recipe files
                save_path = (
                    Path(save_dir) / f"rank{rank + 1:02d}_{target.name}_rep{rep_id}_net{net_id}.png"
                )
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                logger.info(f"Saved figure to {save_path}")

            plt.show()


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                       --     main design function     --


def normalize_ratios_prune(current_ratios, rel_off=1e-3, eps=1e-12):
    A = jnp.abs(current_ratios)
    m = jnp.maximum(jnp.max(A, axis=1, keepdims=True), eps)
    norm = A / m
    mask = norm >= rel_off
    return jnp.where(mask, norm, 0.0)


def get_ratio_paths(params):
    ratio_paths = []
    for path, value in params.data.iter_leaves():
        if "ratio" in str(path) and "inverse" not in str(path):
            ratio_paths.append(path)
    return ratio_paths


def start(
    dmanager: DesignManager,
    dconf: DesignConfig,
    model: BiocompModel,
    loggers: Optional[List[Tuple[int, Callable]]] = None,
    async_handler=None,
):
    pkey, bkey, loop_key = jax.random.split(dconf.seed_key, 3)

    # -- initializations --
    stack = dmanager.build_stack(model)
    initial_params = initialize_params(
        stack, dconf.n_replicates, dmanager.n_targets, model.shared_params, pkey
    )
    assert_tree_shape(initial_params, (dconf.n_replicates, dmanager.n_targets))
    static, dynamic = initial_params.filter_by_tag(["non_grad", "shared"])
    initial_optimizer_state = vmap(vmap(dconf.optimizer.init))(dynamic)

    # -- get data --
    num_z = static["global/number_of_random_variables"]
    assert_that(num_z.shape[0]).is_equal_to(dconf.n_replicates)
    assert_that(jnp.all(num_z == num_z[0])).is_true()
    num_z = (dmanager.n_targets, int(num_z.ravel()[0].squeeze()))

    steps_per_epoch = max(1, dconf.n_batches_per_epoch // dconf.batches_per_step)
    total_steps = int(dconf.n_epochs * steps_per_epoch)

    logger.debug(
        f"Total steps: {total_steps}, Steps per epoch: {steps_per_epoch}, \n"
        f"Batch size: {dconf.batch_size}, Batches per step: {dconf.batches_per_step}"
    )
    assert_that(total_steps).is_greater_than(0)

    n_networks = stack.get_nb_networks()
    n_inputs = stack.get_nb_inputs()
    n_outputs = stack.get_nb_outputs()

    xbatches_list, ybatches_list = dmanager.get_samples(
        (len(dmanager.networks), steps_per_epoch, dconf.n_replicates, dconf.batches_per_step, dconf.batch_size),
        bkey,
    )

    xbatches = jnp.concatenate(xbatches_list, axis=-1)
    ybatches = ybatches_list[0]

    effective_batch_size = dconf.batch_size
    if dmanager.is_lattice_mode:
        xres, yres = dmanager.grid_resolution
        effective_batch_size *= xres * yres

    n_design_inputs = 2 * len(dmanager.networks)

    logger.info(
        f"Data generated: {len(dmanager.networks)} design networks, "
        f"n_design_inputs={n_design_inputs}, xbatches.shape={xbatches.shape}"
    )

    assert_that(xbatches).has_shape(
        (
            steps_per_epoch,
            dconf.n_replicates,
            dconf.batches_per_step,
            effective_batch_size,
            dmanager.n_targets,
            n_design_inputs,
        )
    )

    # -- step function --
    ratio_paths = get_ratio_paths(initial_params)

    def norm_ratios_hook(params, *a, **kw):
        print("Normalizing ratios...")
        return params.update_leaves_by_path(ratio_paths, normalize_ratios_prune)

    loss_func = dconf.loss_function.get_impl()(
        stack, dconf, dmanager, num_z=num_z, ratio_paths=ratio_paths
    )
    step_fn = make_training_step(
        loss_func,
        dconf.optimizer,
        dconf.keep_in_history,
        scannable=True,
        post_update_hook=norm_ratios_hook,
        updates_need_vmap=True,
        static_tags=["non_grad", "shared"],
    )

    def step(params: ParameterTree, opt_state: optax.OptState, step_key, xs, ys):
        keys = jax.random.split(step_key, dconf.n_replicates)
        assert_that(xs).has_shape(
            (
                dconf.n_replicates,
                dconf.batches_per_step,
                effective_batch_size,
                dmanager.n_targets,
                n_design_inputs,
            )
        )
        expected_y_last_dim = 1 if dmanager.is_lattice_mode else n_networks
        assert_that(ys).has_shape(
            (
                dconf.n_replicates,
                dconf.batches_per_step,
                effective_batch_size,
                dmanager.n_targets,
                expected_y_last_dim,
            )
        )
        assert_tree_shape(params, (dconf.n_replicates, dmanager.n_targets))
        assert_tree_shape(opt_state, (dconf.n_replicates, dmanager.n_targets))

        return jax.vmap(
            Partial(per_replicate_step, num_z=num_z, training_config=dconf, scannable_step=step_fn)
        )(params, opt_state, keys, xs, ys)

    return optimize(
        step,
        initial_params,
        initial_optimizer_state,
        xbatches=xbatches,
        ybatches=ybatches,
        config=dconf,
        n_total_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
        key=loop_key,
        stack=stack,
        loggers=loggers,
        async_handler=async_handler,
        verbose=True,
    )


##────────────────────────────────────────────────────────────────────────────}}}
