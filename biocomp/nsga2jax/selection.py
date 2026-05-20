# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Selection operators for NSGA-II."""

from functools import partial

import jax
import jax.numpy as jnp

from biocomp.nsga2jax.crowding import crowding_distance_by_front
from biocomp.nsga2jax.pareto import non_dominated_sort


@partial(jax.jit, static_argnums=(2,))
def nsga2_select(
    population: jax.Array,
    fitness: jax.Array,
    n_select: int,
) -> tuple[jax.Array, jax.Array]:
    """Select top n_select individuals using NSGA-II criteria.

    Selection priority:
    1. Lower Pareto rank (front 0 > front 1 > ...)
    2. Higher crowding distance (more diversity)

    Args:
        population: (pop_size, n_dims) decision vectors
        fitness: (pop_size, n_objectives) objective values
        n_select: number of individuals to select

    Returns:
        (selected_population, selected_fitness) both with n_select rows
    """
    assert population.ndim == 2
    assert fitness.ndim == 2
    assert population.shape[0] == fitness.shape[0]

    ranks = non_dominated_sort(fitness)

    crowding = _crowding_per_front(fitness, ranks)

    order = jnp.lexsort((-crowding, ranks))
    selected_idx = order[:n_select]

    return population[selected_idx], fitness[selected_idx]


def _crowding_per_front(fitness: jax.Array, ranks: jax.Array) -> jax.Array:
    """Compute crowding distance within each front."""
    return crowding_distance_by_front(fitness, ranks)


@partial(jax.jit, static_argnums=(3,))
def binary_tournament(
    key: jax.Array,
    population: jax.Array,
    fitness: jax.Array,
    n_select: int,
) -> jax.Array:
    """Binary tournament selection based on NSGA-II criteria.

    For mating selection: compare two random individuals,
    pick the one with better rank (or crowding if tied).
    """
    assert population.ndim == 2
    assert fitness.ndim == 2
    pop_size = population.shape[0]

    ranks = non_dominated_sort(fitness)
    crowding = _crowding_per_front(fitness, ranks)

    key_i, key_j = jax.random.split(key)
    idx_i = jax.random.randint(key_i, (n_select,), 0, pop_size)
    idx_j = jax.random.randint(key_j, (n_select,), 0, pop_size)

    rank_i, rank_j = ranks[idx_i], ranks[idx_j]
    crowd_i, crowd_j = crowding[idx_i], crowding[idx_j]

    i_wins = (rank_i < rank_j) | ((rank_i == rank_j) & (crowd_i >= crowd_j))
    winners = jnp.where(i_wins, idx_i, idx_j)

    return population[winners]
