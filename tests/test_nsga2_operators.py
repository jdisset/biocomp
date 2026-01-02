"""Tests for genetic operators: SBX crossover and polynomial mutation."""

import jax
import jax.numpy as jnp

from biocomp.nsga2jax.operators import (
    batch_polynomial_mutation,
    batch_sbx_crossover,
    polynomial_mutation,
    sbx_crossover,
)


class TestSBXCrossover:
    def test_output_shape(self):
        key = jax.random.key(0)
        p1 = jnp.array([1.0, 2.0, 3.0])
        p2 = jnp.array([4.0, 5.0, 6.0])

        c1, c2 = sbx_crossover(key, p1, p2)

        assert c1.shape == p1.shape
        assert c2.shape == p2.shape

    def test_respects_bounds(self):
        key = jax.random.key(42)
        p1 = jnp.array([0.0, 0.0])
        p2 = jnp.array([1.0, 1.0])
        lb = jnp.array([0.0, 0.0])
        ub = jnp.array([1.0, 1.0])

        for i in range(10):
            k = jax.random.fold_in(key, i)
            c1, c2 = sbx_crossover(k, p1, p2, lb=lb, ub=ub)
            assert jnp.all(c1 >= lb) and jnp.all(c1 <= ub)
            assert jnp.all(c2 >= lb) and jnp.all(c2 <= ub)

    def test_deterministic(self):
        key = jax.random.key(123)
        p1 = jnp.array([1.0, 2.0])
        p2 = jnp.array([3.0, 4.0])

        c1a, c2a = sbx_crossover(key, p1, p2)
        c1b, c2b = sbx_crossover(key, p1, p2)

        assert jnp.allclose(c1a, c1b)
        assert jnp.allclose(c2a, c2b)

    def test_jit_compatible(self):
        key = jax.random.key(0)
        p1 = jnp.array([1.0, 2.0])
        p2 = jnp.array([3.0, 4.0])

        jitted = jax.jit(sbx_crossover)
        c1, c2 = jitted(key, p1, p2)

        assert c1.shape == (2,)
        assert c2.shape == (2,)


class TestPolynomialMutation:
    def test_output_shape(self):
        key = jax.random.key(0)
        x = jnp.array([0.5, 0.5, 0.5])
        lb = jnp.zeros(3)
        ub = jnp.ones(3)

        mutated = polynomial_mutation(key, x, lb, ub)

        assert mutated.shape == x.shape

    def test_respects_bounds(self):
        key = jax.random.key(42)
        x = jnp.array([0.5, 0.5])
        lb = jnp.array([0.0, 0.0])
        ub = jnp.array([1.0, 1.0])

        for i in range(20):
            k = jax.random.fold_in(key, i)
            mutated = polynomial_mutation(k, x, lb, ub, prob=1.0)
            assert jnp.all(mutated >= lb) and jnp.all(mutated <= ub)

    def test_prob_zero_no_change(self):
        key = jax.random.key(0)
        x = jnp.array([0.5, 0.5])
        lb = jnp.zeros(2)
        ub = jnp.ones(2)

        mutated = polynomial_mutation(key, x, lb, ub, prob=0.0)

        assert jnp.allclose(mutated, x)

    def test_jit_compatible(self):
        key = jax.random.key(0)
        x = jnp.array([0.5, 0.5])
        lb = jnp.zeros(2)
        ub = jnp.ones(2)

        jitted = jax.jit(polynomial_mutation)
        mutated = jitted(key, x, lb, ub)

        assert mutated.shape == (2,)


class TestBatchOperators:
    def test_batch_crossover_shape(self):
        key = jax.random.key(0)
        parents = jnp.ones((10, 5))

        offspring = batch_sbx_crossover(key, parents)

        assert offspring.shape == (10, 5)

    def test_batch_mutation_shape(self):
        key = jax.random.key(0)
        pop = jnp.ones((10, 5)) * 0.5
        lb = jnp.zeros(5)
        ub = jnp.ones(5)

        mutated = batch_polynomial_mutation(key, pop, lb, ub)

        assert mutated.shape == (10, 5)

    def test_batch_crossover_respects_bounds(self):
        key = jax.random.key(42)
        parents = jax.random.uniform(key, (20, 4))
        lb = jnp.zeros(4)
        ub = jnp.ones(4)

        offspring = batch_sbx_crossover(key, parents, lb=lb, ub=ub)

        assert jnp.all(offspring >= lb)
        assert jnp.all(offspring <= ub)


def test_vmap_crossover():
    key = jax.random.key(0)
    keys = jax.random.split(key, 5)
    p1_batch = jnp.ones((5, 3))
    p2_batch = jnp.zeros((5, 3))

    vmapped = jax.vmap(sbx_crossover)
    c1_batch, c2_batch = vmapped(keys, p1_batch, p2_batch)

    assert c1_batch.shape == (5, 3)
    assert c2_batch.shape == (5, 3)
