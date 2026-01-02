"""Pareto dominance and non-dominated sorting for multi-objective optimization."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax


@jax.jit
def dominates(a: jax.Array, b: jax.Array) -> jax.Array:
    """Check if solution `a` Pareto-dominates solution `b` (minimization).

    a dominates b iff: a <= b in all objectives AND a < b in at least one.
    """
    return jnp.all(a <= b) & jnp.any(a < b)


@jax.jit
def dominance_matrix(fitness: jax.Array) -> jax.Array:
    """Compute pairwise dominance matrix.

    Returns D where D[i,j] = True iff solution i dominates solution j.
    Shape: (pop_size, pop_size)
    """
    assert fitness.ndim == 2, f"fitness must be 2D (pop, obj), got {fitness.ndim}D"
    return jax.vmap(lambda fi: jax.vmap(lambda fj: dominates(fi, fj))(fitness))(fitness)


@jax.jit
def non_dominated_sort(fitness: jax.Array) -> jax.Array:
    """Assign Pareto front ranks to each solution (0 = first front, best).

    Uses iterative front extraction with lax.while_loop for JIT compatibility.
    """
    assert fitness.ndim == 2, f"fitness must be 2D (pop, obj), got {fitness.ndim}D"
    pop_size = fitness.shape[0]

    dom_matrix = dominance_matrix(fitness)
    dominated_by_count = jnp.sum(dom_matrix, axis=0)

    init_state = (
        jnp.zeros(pop_size, dtype=jnp.int32),  # ranks
        dominated_by_count,  # remaining domination counts
        jnp.int32(0),  # current_rank
        dominated_by_count == 0,  # current_front mask
    )

    def cond(state):
        _, _, _, front_mask = state
        return jnp.any(front_mask)

    def body(state):
        ranks, dom_counts, current_rank, front_mask = state

        ranks = jnp.where(front_mask, current_rank, ranks)

        decrement = jnp.sum(
            front_mask[:, None] * dom_matrix,
            axis=0,
        )
        dom_counts = dom_counts - decrement
        dom_counts = jnp.where(front_mask, -1, dom_counts)

        next_front = dom_counts == 0
        return ranks, dom_counts, current_rank + 1, next_front

    ranks, _, _, _ = lax.while_loop(cond, body, init_state)
    return ranks


@jax.jit
def pareto_front_mask(fitness: jax.Array) -> jax.Array:
    """Return boolean mask of solutions on the first Pareto front."""
    return non_dominated_sort(fitness) == 0
