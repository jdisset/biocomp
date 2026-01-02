from enum import IntEnum
from typing import NamedTuple, Any, Callable
import math

import jax
import jax.numpy as jnp
import optax
from pydantic import BaseModel, ConfigDict

from .optimutils import build_optimizer_chain, sanitize_gradients, DEFAULT_OPTIMIZER
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


class NSGA2DesignState(NamedTuple):
    """State for NSGA2 multi-objective design optimizer."""

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
    """NSGA2 multi-objective optimizer for design with binary TU masks.

    Evolves binary TU masks using NSGA2, with inner GD loop to optimize
    continuous parameters (ratios, embeddings) for each individual.

    Objectives:
        1. Pattern loss (design quality)
        2. TU count (sparsity/simplicity)

    The binary mask determines which TUs are enabled. For each mask,
    GD refines the continuous parameters to minimize pattern loss only.
    The final fitness is multi-objective: (pattern_loss, n_enabled_tus).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    pop_size: int = 32
    n_generations: int = 100
    gd_steps_per_individual: int = 50
    gd_learning_rate: float = 0.02
    mask_threshold: float = 0.5
    n_tus: int | None = None
    continuous_dim: int | None = None

    crossover_eta: float = 15.0
    crossover_prob: float = 0.9
    mutation_eta: float = 20.0

    _nsga2: Any = None
    _gd_optimizer: Any = None

    def _get_pop_size(self, dim: int) -> int:
        return self.pop_size

    def _build_gd_optimizer(self):
        return optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(self.gd_learning_rate),
        )

    def init(
        self, key: jax.random.PRNGKey, params: jnp.ndarray, objective_fn: Callable,
        n_tus: int | None = None, continuous_dim: int | None = None,
    ) -> NSGA2DesignState:
        from biocomp.nsga2jax import NSGA2, NSGA2Params

        actual_n_tus = n_tus if n_tus is not None else self.n_tus
        actual_continuous_dim = continuous_dim if continuous_dim is not None else self.continuous_dim

        if actual_n_tus is None:
            actual_n_tus = 0
            logger.warning("n_tus not set, assuming 0 TUs (no TU mask optimization)")

        if actual_continuous_dim is None:
            actual_continuous_dim = int(params.shape[0]) - actual_n_tus
            logger.info(f"Auto-detected continuous_dim: {actual_continuous_dim}")

        actual_n_tus = int(actual_n_tus)
        actual_continuous_dim = int(actual_continuous_dim)

        object.__setattr__(self, "n_tus", actual_n_tus)
        object.__setattr__(self, "continuous_dim", actual_continuous_dim)

        total_dim = actual_n_tus + actual_continuous_dim
        assert params.shape[0] == total_dim, (
            f"params dim {params.shape[0]} != n_tus ({actual_n_tus}) + continuous ({actual_continuous_dim})"
        )

        nsga2_params = NSGA2Params(
            crossover_eta=self.crossover_eta,
            crossover_prob=self.crossover_prob,
            mutation_eta=self.mutation_eta,
        )

        lb = jnp.concatenate([jnp.zeros(actual_n_tus), jnp.full(actual_continuous_dim, -3.0)])
        ub = jnp.concatenate([jnp.ones(actual_n_tus), jnp.full(actual_continuous_dim, 3.0)])

        nsga2 = NSGA2(
            pop_size=self.pop_size,
            n_dims=total_dim,
            n_objectives=2,
            lb=lb,
            ub=ub,
            params=nsga2_params,
        )
        object.__setattr__(self, "_nsga2", nsga2)
        object.__setattr__(self, "_gd_optimizer", self._build_gd_optimizer())

        nsga2_state = nsga2.init(key)

        init_loss = objective_fn(params)
        logger.info(
            f"NSGA2Design: n_tus={actual_n_tus}, continuous={actual_continuous_dim}, "
            f"pop={self.pop_size}, gd_steps={self.gd_steps_per_individual}"
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

    def _genes_to_mask(self, genes: jnp.ndarray) -> jnp.ndarray:
        return (genes > self.mask_threshold).astype(jnp.float32)

    def _run_gd_for_individual(
        self,
        tu_genes: jnp.ndarray,
        continuous_params: jnp.ndarray,
        pattern_loss_fn: Callable,
        key: jax.Array,
    ) -> tuple[jnp.ndarray, float]:
        """Run GD on continuous params with fixed TU mask."""
        tu_mask = self._genes_to_mask(tu_genes)

        def loss_with_mask(cont_params):
            full_genome = jnp.concatenate([tu_mask, cont_params])
            return pattern_loss_fn(full_genome)

        opt_state = self._gd_optimizer.init(continuous_params)

        def gd_step(carry, _):
            params, opt_state = carry
            loss, grads = jax.value_and_grad(loss_with_mask)(params)
            grads = sanitize_gradients(grads)
            updates, new_opt_state = self._gd_optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return (new_params, new_opt_state), loss

        (final_params, _), losses = jax.lax.scan(
            gd_step, (continuous_params, opt_state), None, length=self.gd_steps_per_individual
        )

        final_loss = loss_with_mask(final_params)
        return final_params, final_loss

    def step(
        self,
        state: NSGA2DesignState,
        key: jax.random.PRNGKey,
        objective_fn: Callable,
        step: jnp.ndarray | None = None,
    ) -> tuple[NSGA2DesignState, dict]:
        """Execute one NSGA2 generation with inner GD refinement."""
        n_tus = self.n_tus

        ask_key, gd_key = jax.random.split(key)
        offspring = self._nsga2.ask(ask_key, state.opt_state)

        def evaluate_individual(genome: jnp.ndarray, ind_key: jax.Array) -> tuple[jnp.ndarray, float]:
            tu_genes = genome[:n_tus]
            continuous_init = genome[n_tus:]

            refined_continuous, pattern_loss = self._run_gd_for_individual(
                tu_genes, continuous_init, objective_fn, ind_key
            )

            tu_mask = self._genes_to_mask(tu_genes)
            tu_count = jnp.sum(tu_mask)

            fitness = jnp.array([pattern_loss, tu_count])
            refined_genome = jnp.concatenate([tu_genes, refined_continuous])
            return refined_genome, fitness

        gd_keys = jax.random.split(gd_key, self.pop_size)

        refined_list = []
        fitness_list = []
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
        """Extract the current Pareto front from state."""
        return state.pareto_front, state.pareto_fitness
