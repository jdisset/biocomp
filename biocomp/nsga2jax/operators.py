# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Genetic operators for NSGA-II: SBX crossover and polynomial mutation."""

import jax
import jax.numpy as jnp


@jax.jit
def sbx_crossover(
    key: jax.Array,
    parent1: jax.Array,
    parent2: jax.Array,
    eta: float = 15.0,
    prob: float = 0.9,
    lb: jax.Array | None = None,
    ub: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Simulated Binary Crossover (SBX).

    Produces two offspring from two parents with distribution index eta.
    Higher eta = children closer to parents.

    Args:
        key: JAX random key
        parent1, parent2: parent solution vectors (same shape)
        eta: distribution index (typically 15-20)
        prob: crossover probability per variable
        lb, ub: optional bounds for clipping

    Returns:
        (child1, child2) tuple
    """
    assert parent1.shape == parent2.shape
    n_dims = parent1.shape[0]

    key_cross, key_u, key_swap = jax.random.split(key, 3)

    do_crossover = jax.random.uniform(key_cross, (n_dims,)) < prob
    u = jax.random.uniform(key_u, (n_dims,))

    beta = jnp.where(
        u <= 0.5,
        (2.0 * u) ** (1.0 / (eta + 1.0)),
        (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0)),
    )

    child1_raw = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
    child2_raw = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)

    child1 = jnp.where(do_crossover, child1_raw, parent1)
    child2 = jnp.where(do_crossover, child2_raw, parent2)

    swap = jax.random.uniform(key_swap, (n_dims,)) < 0.5
    child1, child2 = (
        jnp.where(swap, child2, child1),
        jnp.where(swap, child1, child2),
    )

    if lb is not None and ub is not None:
        child1 = jnp.clip(child1, lb, ub)
        child2 = jnp.clip(child2, lb, ub)

    return child1, child2


@jax.jit
def polynomial_mutation(
    key: jax.Array,
    x: jax.Array,
    lb: jax.Array,
    ub: jax.Array,
    eta: float = 20.0,
    prob: float | None = None,
) -> jax.Array:
    """Polynomial mutation.

    Args:
        key: JAX random key
        x: solution vector to mutate
        lb, ub: lower and upper bounds
        eta: distribution index (typically 20)
        prob: mutation probability per variable (default: 1/n_dims)

    Returns:
        mutated solution vector
    """
    assert x.shape == lb.shape == ub.shape
    n_dims = x.shape[0]

    if prob is None:
        prob = 1.0 / n_dims

    key_mut, key_u = jax.random.split(key)
    do_mutate = jax.random.uniform(key_mut, (n_dims,)) < prob
    u = jax.random.uniform(key_u, (n_dims,))

    delta = jnp.where(
        u < 0.5,
        (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0,
        1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0)),
    )

    x_range = ub - lb
    x_mutated = x + delta * x_range
    x_mutated = jnp.clip(x_mutated, lb, ub)

    return jnp.where(do_mutate, x_mutated, x)


@jax.jit
def batch_sbx_crossover(
    key: jax.Array,
    parents: jax.Array,
    eta: float = 15.0,
    prob: float = 0.9,
    lb: jax.Array | None = None,
    ub: jax.Array | None = None,
) -> jax.Array:
    """Apply SBX crossover to pairs of parents.

    Args:
        key: JAX random key
        parents: (pop_size, n_dims) - will be paired sequentially
        eta, prob, lb, ub: see sbx_crossover

    Returns:
        (pop_size, n_dims) offspring
    """
    assert parents.ndim == 2
    pop_size, n_dims = parents.shape
    assert pop_size % 2 == 0, "population size must be even for pairing"

    n_pairs = pop_size // 2
    p1 = parents[:n_pairs]
    p2 = parents[n_pairs:]

    keys = jax.random.split(key, n_pairs)

    def crossover_pair(k, par1, par2):
        c1, c2 = sbx_crossover(k, par1, par2, eta, prob, lb, ub)
        return jnp.stack([c1, c2])

    offspring_pairs = jax.vmap(crossover_pair)(keys, p1, p2)
    return offspring_pairs.reshape(pop_size, n_dims)


@jax.jit
def batch_polynomial_mutation(
    key: jax.Array,
    population: jax.Array,
    lb: jax.Array,
    ub: jax.Array,
    eta: float = 20.0,
    prob: float | None = None,
) -> jax.Array:
    """Apply polynomial mutation to entire population."""
    assert population.ndim == 2
    keys = jax.random.split(key, population.shape[0])
    return jax.vmap(lambda k, x: polynomial_mutation(k, x, lb, ub, eta, prob))(keys, population)
