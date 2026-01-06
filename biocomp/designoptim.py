from enum import IntEnum
from typing import NamedTuple, Any, Callable
import math

import jax
import jax.numpy as jnp
import optax
from pydantic import BaseModel, ConfigDict

from .optimutils import (
    build_optimizer_chain,
    sanitize_gradients,
    create_gd_step_fn,
    DEFAULT_OPTIMIZER,
    DEFAULT_OPTIMIZER_SIMPLE,
)
from .logging_config import get_logger

logger = get_logger(__name__)


class OptimPhase(IntEnum):
    EC = 0
    GD = 1
    DONE = 2


class OptimizationState(NamedTuple):
    step: jnp.ndarray
    params: jnp.ndarray
    best_params: jnp.ndarray
    best_loss: jnp.ndarray
    phase: jnp.ndarray
    opt_state: Any

    def is_finite(self) -> jnp.ndarray:
        return jnp.isfinite(self.best_loss) & jnp.isfinite(self.params).all()


def _phase(p: OptimPhase) -> jnp.ndarray:
    return jnp.array(p, dtype=jnp.int32)


def _state(step, params, best_params, best_loss, phase, opt_state) -> OptimizationState:
    return OptimizationState(
        jnp.array(step, dtype=jnp.int32),
        params,
        best_params,
        jnp.array(best_loss),
        _phase(phase),
        opt_state,
    )


def _validate_init(params: jnp.ndarray, objective_fn: Callable) -> float:
    assert params.ndim == 1 and jnp.isfinite(params).all(), "params must be finite 1D"
    loss = objective_fn(params)
    assert jnp.isfinite(loss), f"initial loss non-finite: {loss}"
    return loss


def _update_best(state, new_params, loss, new_opt_state):
    is_better = loss < state.best_loss
    return state._replace(
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        best_params=jax.lax.select(is_better, new_params, state.best_params),
        best_loss=jnp.minimum(loss, state.best_loss),
    )


class GradientDescentOptimizer(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    optimizer_stack: list | None = None
    n_steps: int = 1000
    sanitize_grads: bool = True
    _optimizer: Any = None

    def model_post_init(self, __context):
        object.__setattr__(
            self,
            "_optimizer",
            build_optimizer_chain(
                self.optimizer_stack or DEFAULT_OPTIMIZER, with_lr_injection=True
            ),
        )

    def init(
        self, key: jax.random.PRNGKey, params: jnp.ndarray, objective_fn: Callable
    ) -> OptimizationState:
        loss = _validate_init(params, objective_fn)
        return _state(0, params, params, loss, OptimPhase.GD, self._optimizer.init(params))

    def step(
        self, state: OptimizationState, key: jax.random.PRNGKey, objective_fn: Callable
    ) -> tuple[OptimizationState, dict]:
        loss, grads = jax.value_and_grad(objective_fn)(state.params)
        grads = sanitize_gradients(grads) if self.sanitize_grads else grads
        updates, new_opt_state = self._optimizer.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        return _update_best(state, new_params, loss, new_opt_state), {
            "loss": loss,
            "grad_norm": optax.global_norm(grads),
            "phase": state.phase,
        }

    def should_stop(self, state: OptimizationState) -> bool:
        return int(state.step) >= self.n_steps


class EvolutionaryOptimizer(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    popsize: int | str = "auto"
    n_generations: int = 100
    sigma_init: float = 0.5
    min_sigma: float = 1e-4
    max_sigma: float = 10.0
    _es: Any = None
    _pop_size: int | None = None

    def _get_pop_size(self, dim: int) -> int:
        return (
            4 + int(3 * math.log(dim))
            if self.popsize == "auto"
            else int(self.popsize)
        )

    def init(
        self, key: jax.random.PRNGKey, params: jnp.ndarray, objective_fn: Callable
    ) -> OptimizationState:
        from evosax import CMA_ES

        loss = _validate_init(params, objective_fn)
        dim = params.shape[0]
        assert dim > 0, f"dim must be positive: {dim}"

        pop_size = self._get_pop_size(dim)
        es = CMA_ES(popsize=pop_size, num_dims=dim, sigma_init=self.sigma_init)
        object.__setattr__(self, "_es", es)
        object.__setattr__(self, "_pop_size", pop_size)

        es_state = es.initialize(jax.random.split(key)[1], init_mean=params)
        logger.info(f"CMA-ES: dim={dim}, pop={pop_size}, σ₀={self.sigma_init}")
        return _state(0, params, params, loss, OptimPhase.EC, es_state)

    def step(
        self,
        state: OptimizationState,
        key: jax.random.PRNGKey,
        objective_fn: Callable,
        step: jnp.ndarray | None = None,
    ) -> tuple[OptimizationState, dict]:
        """Execute one CMA-ES generation.

        Args:
            state: Current optimization state
            key: Random key for this step
            objective_fn: Either:
                - A pre-compiled vmapped function (pop, step) -> losses (preferred for GPU)
                - A single-sample function (genome) -> loss (legacy, will be vmapped+JIT'd)
            step: Current step number (required if objective_fn takes step as 2nd arg)
        """
        es_state = state.opt_state
        k1, _ = jax.random.split(key)

        pop, es_state = self._es.ask(k1, es_state)

        # If step is provided, objective_fn is pre-compiled vmapped: (pop, step) -> losses
        # Otherwise, it's a single-sample function that needs vmapping (legacy path)
        # NOTE: evosax CMA-ES MINIMIZES fitness, so pass loss directly (no negation!)
        if step is not None:
            raw_fit = objective_fn(pop, step)
        else:
            raw_fit = jax.jit(jax.vmap(objective_fn))(pop)

        # For invalid samples, use +inf (worst possible for minimization)
        fitness = jnp.where(jnp.isfinite(raw_fit), raw_fit, jnp.inf)
        es_state = self._es.tell(pop, fitness, es_state)

        # clamp sigma to prevent explosion (evosax doesn't do this internally)
        sigma_before = es_state.sigma
        clamped_sigma = jnp.clip(es_state.sigma, self.min_sigma, self.max_sigma)
        es_state = es_state.replace(sigma=clamped_sigma)
        sigma_clamped = sigma_before != clamped_sigma

        # evosax minimizes, so best = argmin
        best_idx = jnp.argmin(fitness)
        gen_best, gen_loss = pop[best_idx], fitness[best_idx]
        is_better = gen_loss < state.best_loss

        # compute fitness statistics for diagnostics (fitness = loss, lower is better)
        valid_fitness = jnp.where(jnp.isfinite(raw_fit), raw_fit, jnp.nan)
        fitness_std = jnp.nanstd(valid_fitness)
        fitness_min = jnp.nanmin(valid_fitness)
        fitness_max = jnp.nanmax(valid_fitness)

        # genome statistics for debugging
        mean_genome = es_state.mean
        genome_std = jnp.std(pop, axis=0).mean()
        genome_range = jnp.max(pop) - jnp.min(pop)
        best_dist_from_mean = jnp.linalg.norm(gen_best - mean_genome)

        return state._replace(
            step=state.step + 1,
            params=es_state.mean,
            opt_state=es_state,
            best_params=jax.lax.select(is_better, gen_best, state.best_params),
            best_loss=jnp.where(is_better, gen_loss, state.best_loss),
        ), {
            "gen_best_loss": gen_loss,
            "gen_mean_loss": jnp.mean(fitness),
            "sigma": clamped_sigma,
            "sigma_before_clamp": sigma_before,
            "sigma_clamped": sigma_clamped,
            "n_valid": jnp.sum(jnp.isfinite(raw_fit)),
            "fitness_std": fitness_std,
            "fitness_min": fitness_min,
            "fitness_max": fitness_max,
            "genome_std": genome_std,
            "genome_range": genome_range,
            "best_dist_from_mean": best_dist_from_mean,
            "improved": is_better,
            "phase": state.phase,
        }

    def should_stop(self, state: OptimizationState) -> bool:
        sigma = float(state.opt_state.sigma)
        return (
            sigma < self.min_sigma
            or sigma > self.max_sigma
            or int(state.step) >= self.n_generations
        )


class HybridOptimizer(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    ec: EvolutionaryOptimizer
    gd: GradientDescentOptimizer
    ec_generations: int = 50

    def init(
        self, key: jax.random.PRNGKey, params: jnp.ndarray, objective_fn: Callable
    ) -> OptimizationState:
        ec_state = self.ec.init(jax.random.split(key)[1], params, objective_fn)
        return _state(
            0,
            params,
            ec_state.best_params,
            ec_state.best_loss,
            OptimPhase.EC,
            {"ec": ec_state, "gd": None},
        )

    def step(
        self,
        state: OptimizationState,
        key: jax.random.PRNGKey,
        objective_fn: Callable,
        step: jnp.ndarray | None = None,
    ) -> tuple[OptimizationState, dict]:
        return (
            self._ec_step(state, key, objective_fn, step)
            if int(state.phase) == OptimPhase.EC
            else self._gd_step(state, key, objective_fn, step)
        )

    def _ec_step(self, state, key, objective_fn, step=None):
        new_ec, metrics = self.ec.step(state.opt_state["ec"], key, objective_fn, step)
        if int(new_ec.step) >= self.ec_generations:
            return self._handoff(state, new_ec, key, objective_fn, metrics)
        return state._replace(
            step=state.step + 1,
            params=new_ec.params,
            best_params=new_ec.best_params,
            best_loss=new_ec.best_loss,
            opt_state={"ec": new_ec, "gd": None},
        ), {**metrics, "phase": _phase(OptimPhase.EC)}

    def _handoff(self, state, ec_state, key, objective_fn, ec_metrics):
        assert jnp.isfinite(ec_state.best_params).all() and jnp.isfinite(ec_state.best_loss), (
            "EC non-finite at handoff"
        )
        gd_state = self.gd.init(jax.random.split(key)[1], ec_state.best_params, objective_fn)
        logger.info(
            f"EC→GD handoff at step {int(state.step)}, loss={float(ec_state.best_loss):.6f}"
        )
        return state._replace(
            step=state.step + 1,
            params=ec_state.best_params,
            best_params=ec_state.best_params,
            best_loss=ec_state.best_loss,
            phase=_phase(OptimPhase.GD),
            opt_state={"ec": ec_state, "gd": gd_state},
        ), {
            **ec_metrics,
            "phase": _phase(OptimPhase.GD),
            "handoff": jnp.array(True),
            "ec_final_loss": ec_state.best_loss,
        }

    def _gd_step(self, state, key, objective_fn, step=None):
        # GD uses value_and_grad on single-sample objective, step not used
        gd_state = state.opt_state["gd"]
        assert gd_state is not None, "GD state None in GD phase"
        new_gd, metrics = self.gd.step(gd_state, key, objective_fn)
        is_better = new_gd.best_loss < state.best_loss
        return state._replace(
            step=state.step + 1,
            params=new_gd.params,
            best_params=jax.lax.select(is_better, new_gd.best_params, state.best_params),
            best_loss=jnp.minimum(new_gd.best_loss, state.best_loss),
            opt_state={"ec": state.opt_state["ec"], "gd": new_gd},
        ), {**metrics, "phase": _phase(OptimPhase.GD)}

    def should_stop(self, state: OptimizationState) -> bool:
        return (
            int(state.phase) != OptimPhase.EC
            and state.opt_state["gd"] is not None
            and self.gd.should_stop(state.opt_state["gd"])
        )


def _eval_objective(codec, loss_fn, x, y, z, key, step, flat_params) -> float:
    params = codec.decode(flat_params, apply_constraints=True)
    static, dynamic = params.filter_by_tag(list(codec.static_tags))
    loss, _ = loss_fn(dynamic, static, x, y, z, jax.random.fold_in(key, step), step)
    return loss


def make_objective(
    codec,
    loss_fn: Callable,
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    base_key: jax.random.PRNGKey,
    step: int = 0,
) -> Callable:
    return lambda flat_params: _eval_objective(codec, loss_fn, x, y, z, base_key, step, flat_params)


class ObjectiveWrapper(NamedTuple):
    codec: Any
    loss_fn: Callable
    x_samples: jnp.ndarray
    y_samples: jnp.ndarray
    z_samples: jnp.ndarray
    base_key: jax.random.PRNGKey
    step: int = 0

    def __call__(self, flat_params: jnp.ndarray) -> float:
        return _eval_objective(
            self.codec,
            self.loss_fn,
            self.x_samples,
            self.y_samples,
            self.z_samples,
            self.base_key,
            self.step,
            flat_params,
        )

    def with_step(self, step: int) -> "ObjectiveWrapper":
        return self._replace(step=step)

    def with_samples(self, x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> "ObjectiveWrapper":
        return self._replace(x_samples=x, y_samples=y, z_samples=z)


def genes_to_mask(genes: jnp.ndarray, threshold: float = 0.5) -> jnp.ndarray:
    return (genes > threshold).astype(jnp.float32)


class InnerGDConfig(BaseModel):
    """Configuration for inner GD loop in multi-objective optimizers."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    optimizer_stack: list | None = None
    n_steps: int = 50
    n_replicates: int = 1
    init_perturbation: float = 0.1
    sanitize_grads: bool = True
    _optimizer: Any = None

    def model_post_init(self, __context):
        stack = self.optimizer_stack or DEFAULT_OPTIMIZER_SIMPLE
        object.__setattr__(self, "_optimizer", build_optimizer_chain(stack, with_lr_injection=False))

    @property
    def optimizer(self) -> optax.GradientTransformation:
        return self._optimizer


class NSGA2DesignState(NamedTuple):
    step: jnp.ndarray
    params: jnp.ndarray
    best_params: jnp.ndarray
    best_loss: jnp.ndarray
    phase: jnp.ndarray
    opt_state: Any
    pareto_front: jnp.ndarray | None = None
    pareto_fitness: jnp.ndarray | None = None

    def is_finite(self) -> jnp.ndarray:
        return jnp.isfinite(self.best_loss) & jnp.isfinite(self.params).all()


class NSGA2DesignOptimizer(BaseModel):
    """NSGA2 multi-objective optimizer: evolves TU masks + inner GD for continuous params."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    pop_size: int = 32
    n_generations: int = 100
    n_tus: int | None = None
    continuous_dim: int | None = None
    continuous_bounds: tuple[float, float] = (-3.0, 3.0)
    max_active_tus: int | None = None
    inner_gd: InnerGDConfig | None = None
    crossover_eta: float = 15.0
    crossover_prob: float = 0.9
    mutation_eta: float = 20.0
    _nsga2: Any = None

    def model_post_init(self, __context):
        if self.inner_gd is None:
            object.__setattr__(self, "inner_gd", InnerGDConfig())

    def init(
        self, key: jax.random.PRNGKey, params: jnp.ndarray, objective_fn: Callable,
        n_tus: int | None = None, continuous_dim: int | None = None,
    ) -> NSGA2DesignState:
        from biocomp.nsga2jax import NSGA2, NSGA2Params

        actual_n_tus = n_tus if n_tus is not None else self.n_tus
        actual_continuous_dim = continuous_dim if continuous_dim is not None else self.continuous_dim
        assert actual_n_tus is not None, "n_tus required (via init arg or class attr)"
        assert actual_continuous_dim is not None, "continuous_dim required (via init arg or class attr)"

        actual_n_tus = int(actual_n_tus)
        actual_continuous_dim = int(actual_continuous_dim)
        object.__setattr__(self, "n_tus", actual_n_tus)
        object.__setattr__(self, "continuous_dim", actual_continuous_dim)

        total_dim = actual_n_tus + actual_continuous_dim
        if params.shape[0] == actual_continuous_dim:
            tu_genes = jnp.full(actual_n_tus, 0.5)
            params = jnp.concatenate([tu_genes, params])
        assert params.shape[0] == total_dim, f"params.shape[0]={params.shape[0]} != {total_dim}"

        nsga2_params = NSGA2Params(
            crossover_eta=self.crossover_eta,
            crossover_prob=self.crossover_prob,
            mutation_eta=self.mutation_eta,
        )

        lb_cont, ub_cont = self.continuous_bounds
        lb = jnp.concatenate([jnp.zeros(actual_n_tus), jnp.full(actual_continuous_dim, lb_cont)])
        ub = jnp.concatenate([jnp.ones(actual_n_tus), jnp.full(actual_continuous_dim, ub_cont)])

        nsga2 = NSGA2(
            pop_size=self.pop_size, n_dims=total_dim, n_objectives=2, lb=lb, ub=ub, params=nsga2_params,
        )
        object.__setattr__(self, "_nsga2", nsga2)

        nsga2_state = nsga2.init(key)
        init_loss = objective_fn(params)
        logger.info(
            f"NSGA2Design: n_tus={actual_n_tus}, cont={actual_continuous_dim}, "
            f"pop={self.pop_size}, gd_steps={self.inner_gd.n_steps}, reps={self.inner_gd.n_replicates}"
        )

        return NSGA2DesignState(
            step=jnp.array(0, dtype=jnp.int32),
            params=params,
            best_params=params,
            best_loss=jnp.array(init_loss),
            phase=_phase(OptimPhase.EC),
            opt_state=nsga2_state,
            pareto_front=None,
            pareto_fitness=None,
        )

    def is_valid_genome(self, genome: jnp.ndarray) -> jnp.ndarray:
        if self.max_active_tus is None:
            return jnp.array(True)
        mask = genes_to_mask(genome[: self.n_tus])
        return jnp.sum(mask) <= self.max_active_tus

    def repair_genome(self, genome: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        if self.max_active_tus is None:
            return genome
        tu_genes = genome[: self.n_tus]
        continuous = genome[self.n_tus :]
        mask = genes_to_mask(tu_genes)
        n_active = jnp.sum(mask)
        n_to_disable = jnp.maximum(0, n_active - self.max_active_tus).astype(jnp.int32)

        def disable_one(carry, i):
            genes, k, remaining = carry
            k, subkey = jax.random.split(k)
            should_disable = i < remaining
            active_mask = genes > 0.5
            active_indices = jnp.where(active_mask, jnp.arange(len(genes)), len(genes))
            sorted_active = jnp.sort(active_indices)
            n_active_now = jnp.sum(active_mask)
            rand_idx = jax.random.randint(subkey, (), 0, jnp.maximum(n_active_now, 1))
            to_disable = sorted_active[rand_idx]
            new_genes = jnp.where(
                should_disable & (jnp.arange(len(genes)) == to_disable) & (to_disable < len(genes)),
                0.0, genes,
            )
            return (new_genes, k, remaining), None

        max_disable = self.n_tus if self.n_tus is not None else 32
        (repaired_tu_genes, _, _), _ = jax.lax.scan(
            disable_one, (tu_genes, key, n_to_disable), jnp.arange(max_disable)
        )
        return jnp.where(
            n_to_disable > 0, jnp.concatenate([repaired_tu_genes, continuous]), genome,
        )

    def _run_single_gd(
        self, tu_mask: jnp.ndarray, cont_params: jnp.ndarray, loss_fn: Callable,
    ) -> tuple[jnp.ndarray, float]:
        def masked_loss(p):
            return loss_fn(jnp.concatenate([tu_mask, p]))

        gd_step = create_gd_step_fn(self.inner_gd.optimizer, self.inner_gd.sanitize_grads)
        opt_state = self.inner_gd.optimizer.init(cont_params)

        def body(carry, _):
            p, s = carry
            p, s, _ = gd_step(p, s, masked_loss)
            return (p, s), None

        (final, _), _ = jax.lax.scan(body, (cont_params, opt_state), None, length=self.inner_gd.n_steps)
        return final, masked_loss(final)

    def _run_gd_for_individual(
        self, tu_genes: jnp.ndarray, cont_params: jnp.ndarray, loss_fn: Callable, key: jax.Array,
    ) -> tuple[jnp.ndarray, float]:
        tu_mask = genes_to_mask(tu_genes)

        if self.inner_gd.n_replicates <= 1:
            return self._run_single_gd(tu_mask, cont_params, loss_fn)

        def run_replicate(rep_key):
            perturbation = jax.random.normal(rep_key, cont_params.shape) * self.inner_gd.init_perturbation
            return self._run_single_gd(tu_mask, cont_params + perturbation, loss_fn)

        rep_keys = jax.random.split(key, self.inner_gd.n_replicates)
        all_params, all_losses = jax.vmap(run_replicate)(rep_keys)
        best_idx = jnp.argmin(all_losses)
        return all_params[best_idx], all_losses[best_idx]

    def step(
        self, state: NSGA2DesignState, key: jax.random.PRNGKey, objective_fn: Callable,
        step: jnp.ndarray | None = None,
    ) -> tuple[NSGA2DesignState, dict]:
        n_tus = self.n_tus
        ask_key, gd_key = jax.random.split(key)
        offspring = self._nsga2.ask(ask_key, state.opt_state)

        def evaluate_individual(genome: jnp.ndarray, ind_key: jax.Array) -> tuple[jnp.ndarray, jnp.ndarray]:
            tu_genes, cont_init = genome[:n_tus], genome[n_tus:]
            refined_cont, pattern_loss = self._run_gd_for_individual(tu_genes, cont_init, objective_fn, ind_key)
            tu_count = jnp.sum(genes_to_mask(tu_genes))
            return jnp.concatenate([tu_genes, refined_cont]), jnp.array([pattern_loss, tu_count])

        gd_keys = jax.random.split(gd_key, self.pop_size)

        # sequential eval to avoid memory issues with vmap over GD
        refined_list, fitness_list = [], []
        for i in range(self.pop_size):
            refined, fit = evaluate_individual(offspring[i], gd_keys[i])
            refined_list.append(refined)
            fitness_list.append(fit)
        refined_offspring = jnp.stack(refined_list)
        fitness = jnp.stack(fitness_list)
        fitness = jnp.where(jnp.isfinite(fitness), fitness, jnp.inf)

        new_nsga2_state = self._nsga2.tell(state.opt_state, refined_offspring, fitness)
        pareto_pop, pareto_fit = self._nsga2.get_pareto_front(new_nsga2_state)

        best_idx = jnp.argmin(fitness[:, 0])
        gen_best = refined_offspring[best_idx]
        gen_best_loss = fitness[best_idx, 0]
        is_better = gen_best_loss < state.best_loss

        return NSGA2DesignState(
            step=state.step + 1,
            params=gen_best,
            best_params=jax.lax.select(is_better, gen_best, state.best_params),
            best_loss=jnp.minimum(gen_best_loss, state.best_loss),
            phase=_phase(OptimPhase.EC),
            opt_state=new_nsga2_state,
            pareto_front=pareto_pop,
            pareto_fitness=pareto_fit,
        ), {
            "gen_best_loss": gen_best_loss,
            "gen_best_tu_count": fitness[best_idx, 1],
            "gen_mean_loss": jnp.nanmean(fitness[:, 0]),
            "gen_mean_tu_count": jnp.nanmean(fitness[:, 1]),
            "pareto_size": pareto_fit.shape[0] if pareto_fit is not None else 0,
            "pareto_min_loss": jnp.min(pareto_fit[:, 0]) if pareto_fit is not None else jnp.nan,
            "pareto_min_tu": jnp.min(pareto_fit[:, 1]) if pareto_fit is not None else jnp.nan,
            "phase": _phase(OptimPhase.EC),
        }

    def should_stop(self, state: NSGA2DesignState) -> bool:
        return int(state.step) >= self.n_generations

    def get_pareto_front(self, state: NSGA2DesignState) -> tuple[jnp.ndarray, jnp.ndarray]:
        return state.pareto_front, state.pareto_fitness
