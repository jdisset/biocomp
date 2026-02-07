from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from tqdm import tqdm

from ..logging_config import get_logger
from ..design_session import DesignSession
from ..logger_dispatch import LoggerDispatch, NullDispatch
from .codec import GenomeCodec
from .optimizers import NSGA2DesignOptimizer, NSGA2DesignState

if TYPE_CHECKING:
    from ..design import DesignManager, DesignConfig

logger = get_logger(__name__)


def run_pluggable(
    dmanager: "DesignManager",
    dconf: "DesignConfig",
    model,
    dispatch: LoggerDispatch | None = None,
    lock_ratios: bool = False,
):
    optimizer = dconf.pluggable_optimizer
    is_nsga2 = isinstance(optimizer, NSGA2DesignOptimizer)
    nsga2_n_tus = dmanager.n_tus if is_nsga2 else 0

    session = DesignSession.create(
        dmanager, dconf, model,
        lock_ratios=lock_ratios,
        n_replicates_override=1,
        sample_shape_override=(len(dmanager.networks), 1, 1, 1, dconf.batch_size),
    )
    initial_params = jax.tree.map(lambda x: x.squeeze(0), session.initial_params)

    logger.info("=" * 60)
    logger.info("DESIGN OPTIMIZATION (PLUGGABLE OPTIMIZER)")
    logger.info("=" * 60)

    session.timer.start("codec", "[5/5] Creating codec...")
    codec = GenomeCodec.from_params(
        initial_params,
        static_tags=("shared", "non_grad"),
        use_latent_ratios=dconf.use_latent_ratios,
    )
    flat_params = codec.encode(initial_params)
    logger.info(f"  Genome dimension: {codec.param_dim}")

    if codec.param_dim == 0:
        raise ValueError(
            "No parameters to optimize: genome dimension is 0. "
            "This typically happens with zero-freedom recipes where all ratios are explicitly locked."
        )
    session.timer.end("codec")

    session.timer.start("objective", "[6/6] Creating objective function...")
    x_samples = jnp.concatenate([session.xbatches], axis=-1).reshape(
        session.effective_batch_size, dmanager.n_targets, -1
    )
    y_samples = session.ybatches.reshape(session.effective_batch_size, dmanager.n_targets, -1)
    fixed_z = jnp.full((x_samples.shape[0], *session.num_z), 0.5)
    fixed_key = jax.random.key(42)
    loss_fn = session.loss_fn
    stack = session.stack

    def single_objective(flat_genome: jnp.ndarray, step: jnp.ndarray) -> float:
        params = codec.decode(flat_genome, apply_constraints=True)
        static_p, dynamic_p = params.filter_by_tag(["non_grad", "shared"])
        loss, _ = loss_fn(dynamic_p, static_p, x_samples, y_samples, fixed_z, fixed_key, step)
        return loss

    def get_yhatdep(flat_genome: jnp.ndarray, step: jnp.ndarray) -> jnp.ndarray:
        params = codec.decode(flat_genome, apply_constraints=True)
        static_p, dynamic_p = params.filter_by_tag(["non_grad", "shared"])
        _, aux = loss_fn(dynamic_p, static_p, x_samples, y_samples, fixed_z, fixed_key, step)
        return aux.get("yhatdep")

    def get_yhatdep_nsga2(flat_genome: jnp.ndarray, step: jnp.ndarray) -> jnp.ndarray:
        continuous_genes = flat_genome[nsga2_n_tus:]
        params = codec.decode(continuous_genes, apply_constraints=True)
        static_p, dynamic_p = params.filter_by_tag(["non_grad", "shared"])
        _, aux = loss_fn(dynamic_p, static_p, x_samples, y_samples, fixed_z, fixed_key, step)
        return aux.get("yhatdep")

    compiled_get_yhatdep = jax.jit(get_yhatdep_nsga2 if is_nsga2 else get_yhatdep)
    session.timer.end("objective")

    session.timer.start("opt_init", "[7/7] Initializing optimizer...")
    init_key, opt_key = jax.random.split(session.loop_key)

    if is_nsga2:
        def nsga2_objective(flat_genome: jnp.ndarray) -> float:
            continuous_genes = flat_genome[nsga2_n_tus:]
            params = codec.decode(continuous_genes, apply_constraints=True)
            static_p, dynamic_p = params.filter_by_tag(["non_grad", "shared"])
            loss, _ = loss_fn(
                dynamic_p, static_p, x_samples, y_samples, fixed_z, fixed_key, jnp.array(0)
            )
            return loss

        opt_state = optimizer.init(
            init_key, flat_params, nsga2_objective,
            n_tus=nsga2_n_tus, continuous_dim=codec.param_dim,
        )
        step_objective_fn = nsga2_objective
    else:
        compiled_pop_objective = jax.jit(jax.vmap(single_objective, in_axes=(0, None)))
        dummy_pop = jnp.zeros((optimizer._get_pop_size(codec.param_dim), codec.param_dim))
        _ = compiled_pop_objective(dummy_pop, jnp.array(0, dtype=jnp.int32)).block_until_ready()
        opt_state = optimizer.init(
            init_key, flat_params, lambda g: single_objective(g, jnp.array(0, dtype=jnp.int32))
        )
        step_objective_fn = compiled_pop_objective
    logger.info(f"  Initial loss: {float(opt_state.best_loss):.6f}")
    session.timer.end("opt_init")

    logger.info("=" * 60)
    logger.info("STARTING OPTIMIZATION LOOP")
    logger.info("=" * 60)

    dispatch = dispatch or NullDispatch()
    dispatch.on_start(None, stack)

    loss_history, step_history = [], []
    pbar = tqdm(desc="Optimizing", unit="step")

    while not optimizer.should_stop(opt_state):
        opt_key, step_key = jax.random.split(opt_key)
        current_step = jnp.array(int(opt_state.step), dtype=jnp.int32)
        opt_state, metrics = optimizer.step(opt_state, step_key, step_objective_fn, current_step)
        loss_history.append(float(opt_state.best_loss))
        step_history.append({k: float(v) if hasattr(v, "item") else v for k, v in metrics.items()})
        pbar.update(1)
        pbar.set_postfix(loss=f"{float(opt_state.best_loss):.4f}")

        step_idx = int(opt_state.step)
        yhatdep_arr = compiled_get_yhatdep(opt_state.best_params, current_step)
        step_data = {"loss": [[float(opt_state.best_loss)]], "yhatdep": yhatdep_arr, **metrics}
        if dispatch.needs_params_sync(step_idx):
            best_genome = opt_state.best_params
            if is_nsga2:
                best_genome = best_genome[nsga2_n_tus:]
            step_data["latest_params"] = codec.decode(best_genome, apply_constraints=True)
        dispatch.on_step(step_idx, None, step_data, stack)

    pbar.close()
    logger.info("=" * 60)
    logger.info(f"OPTIMIZATION COMPLETE in {session.timer.total():.2f}s")
    logger.info(f"  Final loss: {float(opt_state.best_loss):.6f}, steps: {int(opt_state.step)}")
    session.timer.summary()

    best_genome = opt_state.best_params
    if is_nsga2:
        best_genome = best_genome[nsga2_n_tus:]
    final_params = codec.decode(best_genome, apply_constraints=True)

    final_step_data = {"loss": [[float(opt_state.best_loss)]], "latest_params": final_params}

    if isinstance(opt_state, NSGA2DesignState):
        pareto_front, pareto_fitness = opt_state.pareto_front, opt_state.pareto_fitness
        if pareto_front is not None and pareto_fitness is not None:
            final_step_data["pareto_front"] = pareto_front
            final_step_data["pareto_fitness"] = pareto_fitness
            logger.info(f"  Pareto front: {pareto_fitness.shape[0]} solutions")
            logger.info(f"    Min loss: {float(jnp.min(pareto_fitness[:, 0])):.4f}")
            logger.info(f"    Min TU count: {float(jnp.min(pareto_fitness[:, 1])):.0f}")

    yhatdep_arr = compiled_get_yhatdep(
        opt_state.best_params, jnp.array(int(opt_state.step), dtype=jnp.int32)
    )
    final_step_data["yhatdep"] = yhatdep_arr
    dispatch.on_end(int(opt_state.step), None, final_step_data, stack)

    step_history.append(final_step_data)

    return jax.tree.map(lambda x: x[None, ...], final_params), loss_history, step_history
