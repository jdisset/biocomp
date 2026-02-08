from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax
from assertpy import assert_that
from jax.tree_util import Partial

from .logging_config import get_logger
from .optimutils import make_training_step, per_replicate_step
from .tumasking import TU_LOG_ALPHA_PATH, LOG_ALPHA_MIN, LOG_ALPHA_MAX
from .design_session import DesignSession
from .design_runtime import GradientStepAdapter, DesignRuntimeContext, run_kernel
from .logger_dispatch import LoggerDispatch, NullDispatch

if TYPE_CHECKING:
    from .design import DesignManager, DesignConfig
    from .parameters import ParameterTree
    from biocomptools.modelmodel import BiocompModel

logger = get_logger(__name__)


def _create_norm_ratios_hook(direct_ratio_paths: list[str], source_ratio_paths: list[str]):
    """Create post-update hook for ratio normalization and log_alpha clipping."""
    from .ratio_utils import normalize_ratios_for_pruning as normalize_ratios_prune
    from .design import normalize_ratio_source_arrays

    def norm_ratios_hook(params, *a, **kw):
        if direct_ratio_paths:
            params = params.update_leaves_by_path(direct_ratio_paths, normalize_ratios_prune)
        if source_ratio_paths:
            params = normalize_ratio_source_arrays(params, source_ratio_paths, normalize_ratios_prune)
        if TU_LOG_ALPHA_PATH in params:
            params = params.update_leaves_by_path(
                [TU_LOG_ALPHA_PATH],
                lambda x: jnp.clip(x, LOG_ALPHA_MIN, LOG_ALPHA_MAX),
            )
        return params

    return norm_ratios_hook


def run_design(
    dmanager: "DesignManager",
    dconf: "DesignConfig",
    model: "BiocompModel",
    dispatch: LoggerDispatch | None = None,
    lock_ratios: bool = False,
    initial_params: "ParameterTree" | None = None,
):
    from .design import assert_tree_shape

    session = DesignSession.create(
        dmanager, dconf, model,
        lock_ratios=lock_ratios,
        initial_params=initial_params,
    )

    session.timer.start("opt_init", "[5/5] Initializing optimizer state...")
    static, dynamic = session.initial_params.filter_by_tag(["non_grad", "shared"])
    assert_tree_shape(dynamic, (dconf.n_replicates, dmanager.n_targets))
    initial_optimizer_state = jax.vmap(jax.vmap(dconf.optimizer.init))(dynamic)
    session.timer.end("opt_init")

    n_networks = session.stack.get_nb_networks()
    effective_batch_size = session.effective_batch_size
    n_design_inputs = session.n_design_inputs

    assert_that(session.xbatches).has_shape((
        session.steps_per_epoch,
        dconf.n_replicates,
        dconf.batches_per_step,
        effective_batch_size,
        dmanager.n_targets,
        n_design_inputs,
    ))

    norm_ratios_hook = _create_norm_ratios_hook(
        session.direct_ratio_paths, session.source_ratio_paths
    )

    step_fn = make_training_step(
        session.loss_fn,
        dconf.optimizer,
        dconf.keep_in_history,
        scannable=True,
        post_update_hook=norm_ratios_hook,
        updates_need_vmap=True,
        static_tags=["non_grad", "shared"],
        sanitize_grads=True,
    )

    def step(params: "ParameterTree", opt_state: optax.OptState, step_key, xs, ys):
        keys = jax.random.split(step_key, dconf.n_replicates)
        assert_that(xs).has_shape((
            dconf.n_replicates,
            dconf.batches_per_step,
            effective_batch_size,
            dmanager.n_targets,
            n_design_inputs,
        ))
        expected_y_last_dim = 1 if dmanager.is_lattice_mode else n_networks
        assert_that(ys).has_shape((
            dconf.n_replicates,
            dconf.batches_per_step,
            effective_batch_size,
            dmanager.n_targets,
            expected_y_last_dim,
        ))

        return jax.vmap(
            Partial(per_replicate_step, num_z=session.num_z, training_config=dconf, scannable_step=step_fn)
        )(params, opt_state, keys, xs, ys)

    adapter = GradientStepAdapter(
        step_fn=step,
        optimizer_state=initial_optimizer_state,
        xbatches=session.xbatches,
        ybatches=session.ybatches,
        steps_per_epoch=session.steps_per_epoch,
        key=session.loop_key,
    )
    ctx = DesignRuntimeContext(
        stack=session.stack,
        config=dconf,
        dispatch=dispatch or NullDispatch(),
        total_steps=session.total_steps,
    )

    logger.info("=" * 60)
    logger.info("STARTING OPTIMIZATION LOOP")
    logger.info("=" * 60)

    return run_kernel(ctx, session.initial_params, adapter)
