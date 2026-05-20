# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""NSGA-II multi-objective optimization algorithm in JAX."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from biocomp.nsga2jax.operators import batch_polynomial_mutation, batch_sbx_crossover
from biocomp.nsga2jax.selection import binary_tournament, nsga2_select


class NSGA2Params(NamedTuple):
    """NSGA-II hyperparameters."""

    crossover_eta: float = 15.0
    crossover_prob: float = 0.9
    mutation_eta: float = 20.0
    mutation_prob: float | None = None


class NSGA2State(NamedTuple):
    """NSGA-II algorithm state."""

    population: jax.Array
    fitness: jax.Array
    generation: int


class NSGA2:
    """NSGA-II: Non-dominated Sorting Genetic Algorithm II.

    A multi-objective evolutionary algorithm that:
    1. Uses Pareto dominance for ranking
    2. Uses crowding distance for diversity preservation
    3. Uses binary tournament selection for mating
    4. Uses SBX crossover and polynomial mutation

    Reference: Deb et al. (2002) "A fast and elitist multiobjective
    genetic algorithm: NSGA-II"
    """

    def __init__(
        self,
        pop_size: int,
        n_dims: int,
        n_objectives: int,
        lb: jax.Array,
        ub: jax.Array,
        params: NSGA2Params | None = None,
    ):
        assert pop_size % 2 == 0, "pop_size must be even"
        assert lb.shape == (n_dims,)
        assert ub.shape == (n_dims,)

        self.pop_size = pop_size
        self.n_dims = n_dims
        self.n_objectives = n_objectives
        self.lb = lb
        self.ub = ub
        self.params = params or NSGA2Params()

    def init(self, key: jax.Array) -> NSGA2State:
        """Initialize random population within bounds."""
        population = (
            jax.random.uniform(key, (self.pop_size, self.n_dims)) * (self.ub - self.lb) + self.lb
        )
        fitness = jnp.full((self.pop_size, self.n_objectives), jnp.inf)
        return NSGA2State(population=population, fitness=fitness, generation=0)

    def ask(self, key: jax.Array, state: NSGA2State) -> jax.Array:
        """Generate offspring population for evaluation.

        First generation: return initial population.
        Later generations: binary tournament -> crossover -> mutation.
        """
        is_first_gen = jnp.all(state.fitness == jnp.inf)

        def first_gen(_):
            return state.population

        def later_gen(_):
            k1, k2, k3 = jax.random.split(key, 3)

            parents = binary_tournament(
                k1,
                state.population,
                state.fitness,
                self.pop_size,
            )

            offspring = batch_sbx_crossover(
                k2,
                parents,
                eta=self.params.crossover_eta,
                prob=self.params.crossover_prob,
                lb=self.lb,
                ub=self.ub,
            )

            offspring = batch_polynomial_mutation(
                k3,
                offspring,
                self.lb,
                self.ub,
                eta=self.params.mutation_eta,
                prob=self.params.mutation_prob,
            )

            return offspring

        return jax.lax.cond(is_first_gen, first_gen, later_gen, None)

    def tell(
        self,
        state: NSGA2State,
        offspring: jax.Array,
        offspring_fitness: jax.Array,
    ) -> NSGA2State:
        """Update state with evaluated offspring.

        First generation: just store fitness.
        Later generations: merge parent+offspring, select best pop_size.
        """
        assert offspring.shape == (self.pop_size, self.n_dims)
        assert offspring_fitness.shape == (self.pop_size, self.n_objectives)

        is_first_gen = jnp.all(state.fitness == jnp.inf)

        def first_gen(_):
            return NSGA2State(
                population=offspring,
                fitness=offspring_fitness,
                generation=1,
            )

        def later_gen(_):
            merged_pop = jnp.concatenate([state.population, offspring], axis=0)
            merged_fit = jnp.concatenate([state.fitness, offspring_fitness], axis=0)

            new_pop, new_fit = nsga2_select(merged_pop, merged_fit, self.pop_size)

            return NSGA2State(
                population=new_pop,
                fitness=new_fit,
                generation=state.generation + 1,
            )

        return jax.lax.cond(is_first_gen, first_gen, later_gen, None)

    def get_pareto_front(self, state: NSGA2State) -> tuple[jax.Array, jax.Array]:
        """Extract the first Pareto front from current population."""
        from biocomp.nsga2jax.pareto import pareto_front_mask

        mask = pareto_front_mask(state.fitness)
        front_pop = state.population[mask]
        front_fit = state.fitness[mask]
        return front_pop, front_fit


def run_nsga2(
    key: jax.Array,
    fitness_fn: callable,
    pop_size: int,
    n_dims: int,
    n_objectives: int,
    lb: jax.Array,
    ub: jax.Array,
    n_generations: int,
    params: NSGA2Params | None = None,
) -> NSGA2State:
    """Convenience function to run NSGA-II optimization.

    Args:
        key: JAX random key
        fitness_fn: callable(population) -> fitness, where
            population: (pop_size, n_dims)
            fitness: (pop_size, n_objectives)
        pop_size: population size (must be even)
        n_dims: number of decision variables
        n_objectives: number of objectives
        lb, ub: lower and upper bounds (n_dims,)
        n_generations: number of generations to run
        params: optional NSGA2Params

    Returns:
        Final NSGA2State
    """
    algo = NSGA2(pop_size, n_dims, n_objectives, lb, ub, params)

    key, init_key = jax.random.split(key)
    state = algo.init(init_key)

    def step(_, carry):
        state, key = carry
        key, ask_key = jax.random.split(key)

        offspring = algo.ask(ask_key, state)
        fitness = fitness_fn(offspring)
        state = algo.tell(state, offspring, fitness)

        return (state, key)

    # NOTE: using fori_loop instead of scan due to JAX tracing issues with scan
    # that cause incorrect optimization in certain conditions.
    state, _ = jax.lax.fori_loop(0, n_generations, step, (state, key))
    return state
