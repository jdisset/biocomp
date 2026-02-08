"""Design evaluation functions extracted from design.py.

Provides:
- TargetTensorLayout: enum for target tensor sharing semantics
- infer_target_layout / extract_target_slice: validate the yraw[0] assumption
- sample_for_evaluation: sample evaluation data for design quality assessment
- evaluate_design: evaluate design quality using training-consistent losses
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import vmap
from tqdm import tqdm

from biocomp.designloss import GridLossWeights, compute_grid_losses
from biocomp.logging_config import get_logger
from biocomp.tumasking import TU_LOG_ALPHA_PATH

if TYPE_CHECKING:
    from .design import DesignConfig, DesignManager
    from .parameters import ParameterTree

logger = get_logger(__name__)


class TargetTensorLayout(Enum):
    """How target tensors (yraw) are organized across networks."""

    SHARED_ACROSS_NETWORKS = "shared"
    PER_NETWORK = "per_network"


def infer_target_layout(yraw: jnp.ndarray) -> TargetTensorLayout:
    """Infer whether targets are shared across networks or per-network.

    Checks if all network slices of yraw are identical. If so, the target
    is shared and yraw[0] is the canonical slice.
    """
    if yraw.shape[0] <= 1:
        return TargetTensorLayout.SHARED_ACROSS_NETWORKS
    ref = yraw[0]
    for i in range(1, yraw.shape[0]):
        if not jnp.allclose(yraw[i], ref):
            return TargetTensorLayout.PER_NETWORK
    return TargetTensorLayout.SHARED_ACROSS_NETWORKS


def extract_target_slice(
    yraw: jnp.ndarray,
    layout: TargetTensorLayout | None = None,
) -> jnp.ndarray:
    """Extract the target tensor, validating the layout assumption.

    For SHARED_ACROSS_NETWORKS, returns yraw[0] (the shared target).
    For PER_NETWORK, raises NotImplementedError (not yet supported in eval).
    """
    if layout is None:
        layout = infer_target_layout(yraw)
    if layout == TargetTensorLayout.PER_NETWORK:
        raise NotImplementedError(
            "Per-network targets are not yet supported in evaluate_design. "
            "All network slices of yraw must be identical (shared targets)."
        )
    return yraw[0]


def sample_for_evaluation(
    dmanager: "DesignManager",
    dconf: "DesignConfig",
    final_params: "ParameterTree",
    n_eval_samples: int,
    key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample evaluation data.

    Returns (xraw, yraw) with shapes
    (n_networks, n_replicates, n_samples, n_targets, 2/1).

    In lattice mode, uses grid sampling (n_samples = xres * yres),
    ignoring n_eval_samples. In uniform mode, samples n_eval_samples
    random points.
    """
    n_networks, n_replicates, n_targets = (
        len(dmanager.networks),
        dconf.n_replicates,
        dmanager.n_targets,
    )
    seed = int(jax.random.key_data(key)[0]) % (2**31)

    if dmanager.is_lattice_mode:
        grid_res = dmanager.grid_resolution
        assert grid_res is not None
        xres, yres = grid_res
        n_samples = xres * yres
        xlist, ylist = dmanager.get_samples(
            (n_networks, n_replicates, 1), seed, share_across_networks=True
        )
    else:
        n_samples = n_eval_samples
        xlist, ylist = dmanager.get_samples(
            (n_networks, n_replicates, n_eval_samples), seed, share_across_networks=True
        )

    xraw, yraw = jnp.stack(xlist, axis=0), jnp.stack(ylist, axis=0)
    assert xraw.shape == (n_networks, n_replicates, n_samples, n_targets, 2), (
        f"xraw shape mismatch: {xraw.shape} vs expected "
        f"({n_networks}, {n_replicates}, {n_samples}, {n_targets}, 2)"
    )
    assert yraw.shape == (n_networks, n_replicates, n_samples, n_targets, 1), (
        f"yraw shape mismatch: {yraw.shape} vs expected "
        f"({n_networks}, {n_replicates}, {n_samples}, {n_targets}, 1)"
    )
    return xraw, yraw


def evaluate_design(
    dmanager: "DesignManager",
    dconf: "DesignConfig",
    model,  # BiocompModel
    final_params: "ParameterTree",
    xraw: jnp.ndarray,
    yraw: jnp.ndarray,
    key: jax.Array,
    max_eval_size: int = 64,
    max_loss_size: int = 64,
    store_predictions: bool = True,
) -> tuple[jnp.ndarray | None, jnp.ndarray]:
    """Evaluate design quality.

    Returns (predictions, losses) where losses has shape
    (n_replicates, n_targets, n_networks).

    CRITICAL: This function uses the SAME loss as training
    (grid_distance_loss weights). This ensures evaluation loss
    reflects actual design performance.
    """
    stack = dmanager.build_stack(
        model,
        unlock_ratios=False,
        auto_lock_topology_tus=dconf.auto_lock_topology_tus,
    )
    n_networks, n_replicates, n_targets, n_samples = (
        len(dmanager.networks),
        dconf.n_replicates,
        dmanager.n_targets,
        xraw.shape[2],
    )
    logger.info(f"Evaluating: {n_replicates} reps × {n_targets} targets × {n_samples} samples")

    eval_weights = GridLossWeights.from_design_config(dconf)

    logger.debug(
        f"Eval loss weights: sinkhorn={eval_weights.w_sinkhorn}, lncc={eval_weights.w_lncc}, "
        f"mse={eval_weights.w_mse}, rmse={eval_weights.w_rmse}, "
        f"simse={eval_weights.w_simse}, zncc={eval_weights.w_zncc}, gradient={eval_weights.w_gradient}"
    )

    grid_res = dmanager.grid_resolution
    assert grid_res is not None, "grid_resolution required for evaluation"
    xres, yres = grid_res
    assert n_samples == xres * yres, f"n_samples={n_samples} must equal xres*yres={xres * yres}"

    num_z_val = int(final_params["global/number_of_random_variables"][0, 0].squeeze())
    dep_mask = stack.get_dependent_output_mask()
    x_combined = xraw.transpose(1, 2, 3, 0, 4).reshape(n_replicates, n_samples, n_targets, -1)
    y_combined = extract_target_slice(yraw)

    has_tu_masking = TU_LOG_ALPHA_PATH in final_params

    if has_tu_masking:
        tu_log_alpha_full = final_params[TU_LOG_ALPHA_PATH]
        assert tu_log_alpha_full.ndim == 4, (
            f"EVALUATE BUG: tu_log_alpha should be 4D (n_reps, n_targets, n_networks, n_tus), "
            f"got {tu_log_alpha_full.ndim}D with shape {tu_log_alpha_full.shape}"
        )
        assert tu_log_alpha_full.shape[0] >= n_replicates, (
            f"EVALUATE BUG: tu_log_alpha has {tu_log_alpha_full.shape[0]} replicates "
            f"but n_replicates={n_replicates}"
        )
        assert tu_log_alpha_full.shape[1] >= n_targets, (
            f"EVALUATE BUG: tu_log_alpha has {tu_log_alpha_full.shape[1]} targets "
            f"but n_targets={n_targets}"
        )
        logger.debug(f"TU masking enabled: {tu_log_alpha_full.shape[-1]} TUs")

    all_losses, all_predictions = [], [] if store_predictions else None

    stack_apply = stack.apply
    assert stack_apply is not None, "stack.apply must be set after build"

    def apply_with_tu_mask(params, x_batch, z_batch, keys, tu_mask):
        def apply_single(x, z, k):
            return stack_apply(params, x, z, k, tu_enabled_random_vars=tu_mask)

        return vmap(apply_single)(x_batch, z_batch, keys)

    apply_batched = jax.jit(apply_with_tu_mask)
    pbar = tqdm(total=n_replicates * n_targets, desc="Evaluating", unit="rep×tgt")

    for rep_idx in range(n_replicates):
        rep_losses, rep_preds = [], [] if store_predictions else None
        for tid in range(n_targets):
            rep_params = jax.tree.map(lambda x: x[rep_idx, tid], final_params)
            x_slice, y_slice = x_combined[rep_idx, :, tid, :], y_combined[rep_idx, :, tid, :]

            tu_mask = None

            yhats = []
            for start in range(0, n_samples, max_eval_size):
                end = min(start + max_eval_size, n_samples)
                z_batch = jax.random.uniform(key, (end - start, num_z_val))
                yhat, _ = apply_batched(
                    rep_params,
                    x_slice[start:end],
                    z_batch,
                    jax.random.split(key, end - start),
                    tu_mask,
                )
                yhats.append(yhat)

            yhat_dep = jnp.compress(dep_mask, jnp.concatenate(yhats, axis=0), axis=-1)
            if store_predictions:
                assert rep_preds is not None
                rep_preds.append(yhat_dep)

            y_grid = y_slice.squeeze(-1).reshape(yres, xres)
            network_losses = []
            for net_idx in range(n_networks):
                yhat_grid = yhat_dep[:, net_idx].reshape(yres, xres)
                result = compute_grid_losses(yhat_grid, y_grid, weights=eval_weights)
                network_losses.append(result.total)
            rep_losses.append(network_losses)
            pbar.update(1)

        all_losses.append(rep_losses)
        if store_predictions:
            assert all_predictions is not None and rep_preds is not None
            all_predictions.append(jnp.stack(rep_preds, axis=0))

    pbar.close()
    losses = jnp.array(all_losses)
    logger.info(
        f"Evaluation complete. Loss: [{float(losses.min()):.4f}, {float(losses.max()):.4f}]"
    )

    if store_predictions:
        assert all_predictions is not None
        return jnp.stack(all_predictions, axis=0).transpose(0, 2, 1, 3), losses
    return None, losses
