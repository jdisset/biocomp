"""Tests for NSGA-II algorithm."""

import jax
import jax.numpy as jnp

from biocomp.nsga2jax import NSGA2, run_nsga2
from biocomp.nsga2jax.selection import binary_tournament, nsga2_select


class TestNSGA2Select:
    def test_selects_correct_count(self):
        pop = jnp.arange(20).reshape(10, 2).astype(float)
        fitness = jnp.array(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
                [5.0, 5.0],
                [0.5, 2.0],
                [2.0, 0.5],
                [1.5, 1.5],
                [6.0, 6.0],
                [7.0, 7.0],
            ]
        )

        selected_pop, selected_fit = nsga2_select(pop, fitness, n_select=5)

        assert selected_pop.shape == (5, 2)
        assert selected_fit.shape == (5, 2)

    def test_selects_pareto_front_first(self):
        pop = jnp.eye(4)
        fitness = jnp.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ]
        )

        selected_pop, selected_fit = nsga2_select(pop, fitness, n_select=2)

        assert jnp.any(jnp.all(selected_fit == jnp.array([0.0, 1.0]), axis=1))
        assert jnp.any(jnp.all(selected_fit == jnp.array([1.0, 0.0]), axis=1))


class TestBinaryTournament:
    def test_output_shape(self):
        key = jax.random.key(0)
        pop = jax.random.uniform(key, (20, 5))
        fitness = jax.random.uniform(key, (20, 2))

        selected = binary_tournament(key, pop, fitness, n_select=10)

        assert selected.shape == (10, 5)

    def test_jit_compatible(self):
        key = jax.random.key(0)
        pop = jax.random.uniform(key, (20, 5))
        fitness = jax.random.uniform(key, (20, 2))

        jitted = jax.jit(lambda k: binary_tournament(k, pop, fitness, 10))
        selected = jitted(key)

        assert selected.shape == (10, 5)


class TestNSGA2Algorithm:
    def test_init_creates_valid_population(self):
        lb = jnp.zeros(5)
        ub = jnp.ones(5)
        algo = NSGA2(pop_size=20, n_dims=5, n_objectives=2, lb=lb, ub=ub)

        key = jax.random.key(0)
        state = algo.init(key)

        assert state.population.shape == (20, 5)
        assert state.fitness.shape == (20, 2)
        assert jnp.all(state.population >= lb)
        assert jnp.all(state.population <= ub)

    def test_ask_returns_offspring(self):
        lb = jnp.zeros(5)
        ub = jnp.ones(5)
        algo = NSGA2(pop_size=20, n_dims=5, n_objectives=2, lb=lb, ub=ub)

        key = jax.random.key(0)
        state = algo.init(key)

        key, ask_key = jax.random.split(key)
        offspring = algo.ask(ask_key, state)

        assert offspring.shape == (20, 5)

    def test_tell_updates_state(self):
        lb = jnp.zeros(5)
        ub = jnp.ones(5)
        algo = NSGA2(pop_size=20, n_dims=5, n_objectives=2, lb=lb, ub=ub)

        key = jax.random.key(0)
        state = algo.init(key)

        offspring = state.population
        fitness = jax.random.uniform(key, (20, 2))

        new_state = algo.tell(state, offspring, fitness)

        assert new_state.generation == 1
        assert not jnp.all(new_state.fitness == jnp.inf)

    def test_full_generation_cycle(self):
        lb = jnp.zeros(5)
        ub = jnp.ones(5)
        algo = NSGA2(pop_size=20, n_dims=5, n_objectives=2, lb=lb, ub=ub)

        def sphere_dtlz(x):
            f1 = jnp.sum(x**2, axis=1)
            f2 = jnp.sum((x - 1) ** 2, axis=1)
            return jnp.stack([f1, f2], axis=1)

        key = jax.random.key(42)
        state = algo.init(key)

        for _i in range(5):
            key, ask_key = jax.random.split(key)
            offspring = algo.ask(ask_key, state)
            fitness = sphere_dtlz(offspring)
            state = algo.tell(state, offspring, fitness)

        assert state.generation == 5
        assert jnp.all(jnp.isfinite(state.fitness))


class TestRunNSGA2:
    def test_zdt1_converges(self):
        def zdt1(x):
            n = x.shape[1]
            f1 = x[:, 0]
            g = 1.0 + 9.0 * jnp.sum(x[:, 1:], axis=1) / (n - 1)
            h = 1.0 - jnp.sqrt(f1 / g)
            f2 = g * h
            return jnp.stack([f1, f2], axis=1)

        n_dims = 10
        lb = jnp.zeros(n_dims)
        ub = jnp.ones(n_dims)

        key = jax.random.key(0)
        state = run_nsga2(
            key=key,
            fitness_fn=zdt1,
            pop_size=50,
            n_dims=n_dims,
            n_objectives=2,
            lb=lb,
            ub=ub,
            n_generations=50,
        )

        assert state.generation == 50
        assert jnp.all(state.fitness[:, 0] >= 0)
        assert jnp.all(state.fitness[:, 1] >= 0)

    def test_get_pareto_front(self):
        def simple_mo(x):
            return jnp.stack([jnp.sum(x**2, axis=1), jnp.sum((x - 1) ** 2, axis=1)], axis=1)

        lb = jnp.zeros(3)
        ub = jnp.ones(3)

        key = jax.random.key(0)
        algo = NSGA2(pop_size=20, n_dims=3, n_objectives=2, lb=lb, ub=ub)
        state = algo.init(key)

        for _i in range(10):
            key, ask_key = jax.random.split(key)
            offspring = algo.ask(ask_key, state)
            fitness = simple_mo(offspring)
            state = algo.tell(state, offspring, fitness)

        front_pop, front_fit = algo.get_pareto_front(state)

        assert front_pop.ndim == 2
        assert front_fit.ndim == 2
        assert front_pop.shape[0] == front_fit.shape[0]
        assert front_pop.shape[0] > 0


def test_nsga2_jit_compatible():
    lb = jnp.zeros(5)
    ub = jnp.ones(5)
    algo = NSGA2(pop_size=20, n_dims=5, n_objectives=2, lb=lb, ub=ub)

    @jax.jit
    def one_step(key, state):
        key, ask_key = jax.random.split(key)
        offspring = algo.ask(ask_key, state)
        fitness = jnp.sum(offspring**2, axis=1, keepdims=True)
        fitness = jnp.concatenate([fitness, 1 - fitness], axis=1)
        return algo.tell(state, offspring, fitness), key

    key = jax.random.key(0)
    state = algo.init(key)

    state, key = one_step(key, state)
    state, key = one_step(key, state)

    assert state.generation == 2
