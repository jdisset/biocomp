from __future__ import annotations

from typing import Callable, TYPE_CHECKING

import jax
import jax.numpy as jnp
from tqdm import tqdm

from ..logging_config import get_logger
from ..design import initialize_params, _create_loss_function, _PhaseTimer
from .codec import GenomeCodec
from .optimizers import NSGA2DesignOptimizer, NSGA2DesignState

if TYPE_CHECKING:
    from ..design import DesignManager, DesignConfig

logger = get_logger(__name__)


def run_pluggable(
    dmanager: DesignManager,
    dconf: DesignConfig,
    model,
    loggers: list[tuple[int, Callable]] | None = None,
    logger_objects: list | None = None,
    async_handler=None,
    lock_ratios: bool = False,
):
    timer = _PhaseTimer()
    logger.info("=" * 60)
    logger.info("DESIGN OPTIMIZATION (PLUGGABLE OPTIMIZER)")
    logger.info("=" * 60)

    pkey, bkey, loop_key = jax.random.split(dconf.seed_key, 3)
    optimizer = dconf.pluggable_optimizer

    is_nsga2 = isinstance(optimizer, NSGA2DesignOptimizer)
    n_tus = dmanager.n_tus if dmanager.enable_tu_masking else 0
    nsga2_n_tus = dmanager.n_tus if is_nsga2 else 0
    n_networks = len(dmanager.networks)

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
    initial_params = initialize_params(
        stack,
        n_replicates=1,
        n_targets=dmanager.n_targets,
        shared_params=model.shared_params,
        key=pkey,
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
    initial_params = jax.tree.map(lambda x: x.squeeze(0), initial_params)
    timer.end("params")

    timer.start("codec", "[3/5] Creating codec and loss function...")
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
            "This typically happens with zero-freedom recipes where all ratios are explicitly locked. "
            "Design optimization requires at least some unlocked parameters. "
            "Consider using a recipe with unlocked ratios (e.g., T_2_ratios_only.yaml) or "
            "check that your recipe doesn't have `locked: true` on all ratio specifications."
        )

    loss_fn, num_z, _ = _create_loss_function(stack, dmanager, dconf, initial_params)
    timer.end("codec")

    timer.start("objective", "[4/5] Creating objective function...")
    effective_batch_size = dconf.batch_size
    if dmanager.is_lattice_mode:
        grid_res = dmanager.grid_resolution
        assert grid_res is not None
        effective_batch_size *= grid_res[0] * grid_res[1]

    xbatches_list, ybatches_list = dmanager.get_samples(
        (len(dmanager.networks), 1, 1, 1, dconf.batch_size),
        bkey,
        share_across_networks=True,
    )
    x_samples = jnp.concatenate(xbatches_list, axis=-1).reshape(
        effective_batch_size, dmanager.n_targets, -1
    )
    y_samples = ybatches_list[0].reshape(effective_batch_size, dmanager.n_targets, -1)
    fixed_z = jnp.full((x_samples.shape[0], *num_z), 0.5)
    fixed_key = jax.random.key(42)

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
    timer.end("objective")

    timer.start("opt_init", "[5/5] Initializing optimizer...")
    init_key, opt_key = jax.random.split(loop_key)

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
            init_key,
            flat_params,
            nsga2_objective,
            n_tus=nsga2_n_tus,
            continuous_dim=codec.param_dim,
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
    timer.end("opt_init")

    logger.info("=" * 60)
    logger.info("STARTING OPTIMIZATION LOOP")
    logger.info("=" * 60)

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

        if loggers and any(int(opt_state.step) % p == 0 or p == -1 for p, _ in loggers):
            yhatdep_arr = compiled_get_yhatdep(opt_state.best_params, current_step)
            step_data = {"loss": [[float(opt_state.best_loss)]], "yhatdep": yhatdep_arr, **metrics}
            needs_params = logger_objects and any(
                "latest_params" in getattr(lo, "required_arrays", [])
                for lo in logger_objects
                if int(opt_state.step) % getattr(lo, "periods", 1) == 0
            )
            if needs_params:
                best_genome = opt_state.best_params
                if is_nsga2:
                    best_genome = best_genome[nsga2_n_tus:]
                step_data["latest_params"] = codec.decode(best_genome, apply_constraints=True)
            for period, callback in loggers:
                if int(opt_state.step) % period == 0 or period == -1:
                    callback(int(opt_state.step), None, step_history=step_data, stack=stack)

    pbar.close()
    logger.info("=" * 60)
    logger.info(f"OPTIMIZATION COMPLETE in {timer.total():.2f}s")
    logger.info(f"  Final loss: {float(opt_state.best_loss):.6f}, steps: {int(opt_state.step)}")
    timer.summary()

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

    if loggers:
        yhatdep_arr = compiled_get_yhatdep(
            opt_state.best_params, jnp.array(int(opt_state.step), dtype=jnp.int32)
        )
        final_step_data["yhatdep"] = yhatdep_arr
        for period, callback in loggers:
            if period == -1:
                callback(int(opt_state.step), None, step_history=final_step_data, stack=stack)

    step_history.append(final_step_data)

    return jax.tree.map(lambda x: x[None, ...], final_params), loss_history, step_history
