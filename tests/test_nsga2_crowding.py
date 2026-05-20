# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Tests for crowding distance calculation."""

import jax
import jax.numpy as jnp

from biocomp.nsga2jax.crowding import crowding_distance, crowding_distance_by_front


class TestCrowdingDistance:
    def test_boundary_solutions_infinite(self):
        fitness = jnp.array(
            [
                [0.0, 1.0],
                [0.5, 0.5],
                [1.0, 0.0],
            ]
        )
        dist = crowding_distance(fitness)
        assert jnp.isinf(dist[0])
        assert jnp.isinf(dist[2])
        assert jnp.isfinite(dist[1])

    def test_interior_positive(self):
        fitness = jnp.array(
            [
                [0.0, 1.0],
                [0.25, 0.75],
                [0.5, 0.5],
                [0.75, 0.25],
                [1.0, 0.0],
            ]
        )
        dist = crowding_distance(fitness)
        assert jnp.all(dist[1:4] > 0)

    def test_masked_solutions_negative_inf(self):
        fitness = jnp.array(
            [
                [0.0, 1.0],
                [0.5, 0.5],
                [1.0, 0.0],
            ]
        )
        mask = jnp.array([True, False, True])
        dist = crowding_distance(fitness, mask)
        assert dist[1] == -jnp.inf

    def test_symmetric_spacing(self):
        fitness = jnp.array(
            [
                [0.0, 1.0],
                [0.5, 0.5],
                [1.0, 0.0],
            ]
        )
        dist = crowding_distance(fitness)
        assert jnp.allclose(dist[1], 2.0)

    def test_jit_compatible(self):
        fitness = jnp.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        jitted = jax.jit(crowding_distance)
        dist = jitted(fitness)
        assert dist.shape == (3,)


class TestCrowdingDistanceByFront:
    def test_separate_fronts(self):
        fitness = jnp.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ]
        )
        ranks = jnp.array([0, 0, 1, 2])

        dist = crowding_distance_by_front(fitness, ranks)

        assert jnp.isinf(dist[0])
        assert jnp.isinf(dist[1])

    def test_jit_compatible(self):
        fitness = jnp.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])
        ranks = jnp.array([0, 0, 1])
        jitted = jax.jit(crowding_distance_by_front)
        dist = jitted(fitness, ranks)
        assert dist.shape == (3,)


def test_vmap_crowding():
    key = jax.random.key(0)
    batch_fitness = jax.random.uniform(key, (5, 20, 2))

    vmapped = jax.vmap(crowding_distance)
    batch_dist = vmapped(batch_fitness)

    assert batch_dist.shape == (5, 20)
