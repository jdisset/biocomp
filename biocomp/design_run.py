from __future__ import annotations

from typing import Callable, TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax
from assertpy import assert_that
from jax.tree_util import Partial, tree_leaves

from .logging_config import get_logger
from .optimutils import make_training_step, per_replicate_step, optimize
from .tumasking import TU_LOG_ALPHA_PATH, LOG_ALPHA_MIN, LOG_ALPHA_MAX

if TYPE_CHECKING:
    from .design import DesignManager, DesignConfig
    from .parameters import ParameterTree
    from biocomptools.modelmodel import BiocompModel

logger = get_logger(__name__)


def run_design(
    dmanager: "DesignManager",
    dconf: "DesignConfig",
    model: "BiocompModel",
    loggers: list[tuple[int, Callable]] | None = None,
    logger_objects: list | None = None,
    async_handler=None,
    lock_ratios: bool = False,
    initial_params: "ParameterTree" | None = None,
):
    from .design import (
        initialize_params,
        _create_loss_function,
        get_ratio_paths_and_sources,
        normalize_ratios_prune,
        normalize_ratio_source_arrays,
        assert_tree_shape,
        _PhaseTimer,
    )

    timer = _PhaseTimer()
    logger.info("=" * 60)
    logger.info("DESIGN OPTIMIZATION - INITIALIZATION PHASE")
    logger.info("=" * 60)

    pkey, bkey, loop_key = jax.random.split(dconf.seed_key, 3)

    timer.start("stack", "[1/5] Building compute stack...")
    stack = dmanager.build_stack(
        model,
        unlock_ratios=not lock_ratios,
        use_latent_ratios=dconf.use_latent_ratios,
        latent_dim=dconf.latent_dim,
        latent_hidden_dim=dconf.latent_hidden_dim,
        auto_lock_topology_tus=dconf.auto_lock_topology_tus,
    )
    timer.end("stack")

    timer.start("params", "[2/5] Initializing parameters...")
    if initial_params is None:
        n_tus = dmanager.n_tus if dmanager.enable_tu_masking else 0
        n_networks = len(dmanager.networks)
        initial_params = initialize_params(
            stack,
            dconf.n_replicates,
            dmanager.n_targets,
            model.shared_params,
            pkey,
            n_tus=n_tus,
            n_networks=n_networks,
            tu_log_alpha_init_mean=dconf.tu_log_alpha_init_mean,
            tu_log_alpha_init_std=dconf.tu_log_alpha_init_std,
            use_latent_tu_masking=dconf.use_latent_tu_masking,
            latent_tu_dim=dconf.latent_tu_dim,
            latent_tu_hidden_dim=dconf.latent_tu_hidden_dim,
            no_masking_tu_ids=stack.no_masking_tu_ids,
            tu_id_to_idx=stack.tu_id_to_idx,
        )
    timer.end("params")

    timer.start("opt_init", "[3/5] Initializing optimizer state...")
    static, dynamic = initial_params.filter_by_tag(["non_grad", "shared"])
    assert_tree_shape(dynamic, (dconf.n_replicates, dmanager.n_targets))

    jax_leaves = tree_leaves(dynamic)
    if not jax_leaves:
        raise ValueError(
            "No parameters to optimize: all parameters are either shared or marked NON_GRAD. "
            "This typically happens with zero-freedom recipes where all ratios are explicitly locked. "
            "Design optimization requires at least some unlocked parameters. "
            "Consider using a recipe with unlocked ratios (e.g., T_2_ratios_only.yaml) or "
            "check that your recipe doesn't have `locked: true` on all ratio specifications."
        )

    initial_optimizer_state = jax.vmap(jax.vmap(dconf.optimizer.init))(dynamic)
    timer.end("opt_init")

    steps_per_epoch = max(1, dconf.n_batches_per_epoch // dconf.batches_per_step)
    total_steps = int(dconf.n_epochs * steps_per_epoch)
    assert_that(total_steps).is_greater_than(0)
    logger.info(
        f"  Config: {total_steps} steps, {steps_per_epoch}/epoch, "
        f"batch={dconf.batch_size}, batches/step={dconf.batches_per_step}"
    )

    n_networks = stack.get_nb_networks()

    timer.start("samples", "[4/5] Generating training samples...")
    xbatches_list, ybatches_list = dmanager.get_samples(
        (
            len(dmanager.networks),
            steps_per_epoch,
            dconf.n_replicates,
            dconf.batches_per_step,
            dconf.batch_size,
        ),
        bkey,
        share_across_networks=True,
    )
    xbatches = jnp.concatenate(xbatches_list, axis=-1)
    ybatches = ybatches_list[0]
    timer.end("samples")

    effective_batch_size = dconf.batch_size
    if dmanager.is_lattice_mode:
        grid_res = dmanager.grid_resolution
        assert grid_res is not None
        xres, yres = grid_res
        effective_batch_size *= xres * yres

    n_design_inputs = 2 * len(dmanager.networks)
    logger.info(f"  Data: {len(dmanager.networks)} networks, xbatches.shape={xbatches.shape}")

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

    timer.start("loss_fn", "[5/5] Creating loss and step functions...")
    loss_func, num_z, direct_ratio_paths = _create_loss_function(
        stack, dmanager, dconf, initial_params
    )
    _, source_ratio_paths = get_ratio_paths_and_sources(initial_params)
    logger.debug(
        f"Ratio paths: {len(direct_ratio_paths)} direct + {len(source_ratio_paths)} ArrayRef"
    )

    def norm_ratios_hook(params, *a, **kw):
        if direct_ratio_paths:
            params = params.update_leaves_by_path(direct_ratio_paths, normalize_ratios_prune)
        if source_ratio_paths:
            params = normalize_ratio_source_arrays(
                params, source_ratio_paths, normalize_ratios_prune
            )
        if TU_LOG_ALPHA_PATH in params:
            params = params.update_leaves_by_path(
                [TU_LOG_ALPHA_PATH],
                lambda x: jnp.clip(x, LOG_ALPHA_MIN, LOG_ALPHA_MAX),
            )
        return params

    step_fn = make_training_step(
        loss_func,
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

        return jax.vmap(
            Partial(per_replicate_step, num_z=num_z, training_config=dconf, scannable_step=step_fn)
        )(params, opt_state, keys, xs, ys)

    timer.end("loss_fn")

    logger.info("-" * 60)
    logger.info(f"INITIALIZATION COMPLETE in {timer.total():.2f}s")
    timer.summary()
    logger.info("=" * 60)
    logger.info("STARTING OPTIMIZATION LOOP")
    logger.info("=" * 60)

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
        logger_objects=logger_objects,
        async_handler=async_handler,
        verbose=True,
    )
