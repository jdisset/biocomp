# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Helpers for the hard-pruning loop in design_pruning.py."""

from dataclasses import dataclass

import jax
import numpy as np

from .compute import ComputeStack
from .design import DesignConfig, DesignManager
from .parameters import ParameterTree
from .logging_config import get_logger
from biocomptools.modelmodel import BiocompModel

logger = get_logger(__name__)


def build_stack_from_dconf(
    dmanager: DesignManager,
    dconf: DesignConfig,
    model: BiocompModel,
    lock_ratios: bool = False,
) -> ComputeStack:
    """Build a ComputeStack from a DesignManager using DesignConfig parameters.

    Consolidates the repeated build_stack(..., use_latent_ratios=dconf.X, ...) pattern.
    """
    return dmanager.build_stack(
        model,
        unlock_ratios=not lock_ratios,
        use_latent_ratios=dconf.use_latent_ratios,
        latent_dim=dconf.latent_dim,
        latent_hidden_dim=dconf.latent_hidden_dim,
        auto_lock_topology_tus=dconf.auto_lock_topology_tus,
    )


@dataclass(frozen=True)
class SegmentSnapshot:
    """Snapshot of a design evaluation at a point in time (pre or post prune)."""

    xraw: jax.Array
    yraw: jax.Array
    yhat: jax.Array
    loss: jax.Array
    mean_loss: float


def evaluate_segment_snapshot(
    dmanager: DesignManager,
    dconf: DesignConfig,
    model: BiocompModel,
    params: ParameterTree,
    key: jax.Array,
    *,
    n_eval_samples: int = 256,
) -> SegmentSnapshot:
    """Run sample_for_evaluation + evaluate_design and return a frozen snapshot."""
    from .design_eval import evaluate_design, sample_for_evaluation

    xraw, yraw = sample_for_evaluation(
        dmanager, dconf, params, n_eval_samples=n_eval_samples, key=key
    )
    yhat, loss = evaluate_design(
        dmanager, dconf, model, params, xraw, yraw, key, store_predictions=True
    )
    mean_loss = float(np.asarray(loss).mean())
    return SegmentSnapshot(xraw=xraw, yraw=yraw, yhat=yhat, loss=loss, mean_loss=mean_loss)


def compare_snapshots(
    pre: SegmentSnapshot,
    post: SegmentSnapshot,
    *,
    regression_threshold: float = 0.20,
) -> dict[str, object]:
    """Compare pre/post prune snapshots and return comparison diagnostics.

    Returns a dict with keys: loss_pre, loss_post, increase_pct, is_regression.
    """
    result: dict[str, object] = {
        "loss_pre": pre.mean_loss,
        "loss_post": post.mean_loss,
        "is_regression": False,
    }
    if pre.mean_loss > 1e-8:
        increase = (post.mean_loss - pre.mean_loss) / pre.mean_loss
        result["increase_pct"] = increase * 100
        result["is_regression"] = increase > regression_threshold
    return result
