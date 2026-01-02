"""Tests for Pareto dominance and non-dominated sorting."""

import jax
import jax.numpy as jnp
import pytest

from biocomp.nsga2jax.pareto import dominance_matrix, dominates, non_dominated_sort, pareto_front_mask


class TestDominates:
    def test_dominates_strictly_better(self):
        a = jnp.array([1.0, 1.0])
        b = jnp.array([2.0, 2.0])
        assert dominates(a, b)
        assert not dominates(b, a)

    def test_dominates_equal_not_dominating(self):
        a = jnp.array([1.0, 1.0])
        assert not dominates(a, a)

    def test_dominates_pareto_incomparable(self):
        a = jnp.array([1.0, 3.0])
        b = jnp.array([2.0, 2.0])
        assert not dominates(a, b)
        assert not dominates(b, a)

    def test_dominates_partial_better(self):
        a = jnp.array([1.0, 2.0])
        b = jnp.array([2.0, 2.0])
        assert dominates(a, b)

    def test_dominates_many_objectives(self):
        a = jnp.array([1.0, 1.0, 1.0, 1.0])
        b = jnp.array([1.0, 1.0, 1.0, 2.0])
        assert dominates(a, b)


class TestDominanceMatrix:
    def test_diagonal_false(self):
        fitness = jnp.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])
        D = dominance_matrix(fitness)
        assert not jnp.any(jnp.diag(D))

    def test_known_dominance_pattern(self):
        fitness = jnp.array(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [0.5, 3.0],
            ]
        )
        D = dominance_matrix(fitness)
        assert D[0, 1]
        assert not D[1, 0]
        assert not D[0, 2]
        assert not D[2, 0]

    def test_jit_compatible(self):
        fitness = jnp.array([[1.0, 2.0], [2.0, 1.0]])
        jitted = jax.jit(dominance_matrix)
        D = jitted(fitness)
        assert D.shape == (2, 2)


class TestNonDominatedSort:
    def test_single_front(self):
        fitness = jnp.array(
            [
                [1.0, 3.0],
                [2.0, 2.0],
                [3.0, 1.0],
            ]
        )
        ranks = non_dominated_sort(fitness)
        assert jnp.all(ranks == 0)

    def test_two_fronts(self):
        fitness = jnp.array(
            [
                [1.0, 1.0],
                [2.0, 2.0],
            ]
        )
        ranks = non_dominated_sort(fitness)
        assert ranks[0] == 0
        assert ranks[1] == 1

    def test_three_fronts(self):
        fitness = jnp.array(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ]
        )
        ranks = non_dominated_sort(fitness)
        assert jnp.array_equal(ranks, jnp.array([0, 1, 2]))

    def test_mixed_fronts(self):
        fitness = jnp.array(
            [
                [1.0, 3.0],
                [2.0, 2.0],
                [3.0, 1.0],
                [2.0, 3.0],
                [3.0, 2.0],
            ]
        )
        ranks = non_dominated_sort(fitness)
        assert ranks[0] == ranks[1] == ranks[2] == 0
        assert ranks[3] == 1
        assert ranks[4] == 1

    def test_jit_compatible(self):
        fitness = jnp.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])
        jitted = jax.jit(non_dominated_sort)
        ranks = jitted(fitness)
        assert ranks.shape == (3,)


class TestParetoFrontMask:
    def test_correct_mask(self):
        fitness = jnp.array(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [0.5, 1.5],
            ]
        )
        mask = pareto_front_mask(fitness)
        assert mask[0]
        assert not mask[1]
        assert mask[2]


@pytest.fixture
def large_population():
    key = jax.random.key(42)
    return jax.random.uniform(key, (100, 3))


def test_non_dominated_sort_large(large_population):
    ranks = non_dominated_sort(large_population)
    assert ranks.shape == (100,)
    assert jnp.all(ranks >= 0)
    assert ranks.min() == 0


def test_vmap_over_populations():
    key = jax.random.key(0)
    batch_fitness = jax.random.uniform(key, (5, 20, 2))

    vmapped_sort = jax.vmap(non_dominated_sort)
    batch_ranks = vmapped_sort(batch_fitness)

    assert batch_ranks.shape == (5, 20)
