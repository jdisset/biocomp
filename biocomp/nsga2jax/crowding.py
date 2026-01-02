"""Crowding distance for diversity preservation in NSGA-II."""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def crowding_distance(fitness: jax.Array, mask: jax.Array | None = None) -> jax.Array:
    """Compute crowding distance for each solution.

    Crowding distance measures how close neighbors are in objective space.
    Higher = more isolated = more diverse = preferred.
    Boundary solutions get infinite distance.

    Args:
        fitness: (pop_size, n_objectives) objective values
        mask: optional (pop_size,) boolean, True = include in calculation

    Returns:
        (pop_size,) crowding distances. Masked-out solutions get -inf.
    """
    assert fitness.ndim == 2, f"fitness must be 2D, got {fitness.ndim}D"
    pop_size, n_obj = fitness.shape

    if mask is None:
        mask = jnp.ones(pop_size, dtype=jnp.bool_)

    n_valid = jnp.sum(mask)

    def distance_for_objective(obj_values: jax.Array) -> jax.Array:
        large_val = jnp.finfo(obj_values.dtype).max / 2
        masked_vals = jnp.where(mask, obj_values, large_val)
        sort_idx = jnp.argsort(masked_vals)
        sorted_vals = masked_vals[sort_idx]

        obj_min = sorted_vals[0]
        valid_max_idx = jnp.maximum(n_valid - 1, 0)
        obj_max = sorted_vals[valid_max_idx]
        obj_range = jnp.where(obj_max > obj_min, obj_max - obj_min, 1.0)

        prev_vals = jnp.concatenate([sorted_vals[:1], sorted_vals[:-1]])
        next_vals = jnp.concatenate([sorted_vals[1:], sorted_vals[-1:]])
        sorted_dist = (next_vals - prev_vals) / obj_range

        dist = jnp.zeros(pop_size)
        dist = dist.at[sort_idx].set(sorted_dist)

        is_boundary = jnp.arange(pop_size) == sort_idx[0]
        is_boundary = is_boundary | (jnp.arange(pop_size) == sort_idx[valid_max_idx])
        dist = jnp.where(is_boundary & mask, jnp.inf, dist)

        return jnp.where(mask, dist, 0.0)

    per_obj_distances = jax.vmap(distance_for_objective, in_axes=1, out_axes=1)(fitness)
    total_distance = jnp.sum(per_obj_distances, axis=1)

    return jnp.where(mask, total_distance, -jnp.inf)


def crowding_distance_by_front(
    fitness: jax.Array,
    ranks: jax.Array,
    max_fronts: int | None = None,
) -> jax.Array:
    """Compute crowding distance within each Pareto front separately.

    Solutions are only compared to others in the same front.

    Args:
        fitness: (pop_size, n_objectives) objective values
        ranks: (pop_size,) front rank for each solution (0 = first front)
        max_fronts: max number of fronts to process (default: pop_size)
    """
    assert fitness.ndim == 2
    assert ranks.ndim == 1
    assert fitness.shape[0] == ranks.shape[0]

    pop_size = fitness.shape[0]
    if max_fronts is None:
        max_fronts = pop_size

    def compute_for_rank(rank, carry):
        front_mask = ranks == rank
        has_members = jnp.any(front_mask)
        distances = jax.lax.cond(
            has_members,
            lambda: crowding_distance(fitness, front_mask),
            lambda: jnp.zeros(pop_size),
        )
        return carry + jnp.where(front_mask, distances, 0.0)

    total_distances = jax.lax.fori_loop(0, max_fronts, compute_for_rank, jnp.zeros(pop_size))
    return total_distances
