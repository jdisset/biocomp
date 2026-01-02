"""Integration tests with known multi-objective problems."""

import time

import jax
import jax.numpy as jnp
import pytest

from biocomp.nsga2jax import NSGA2, run_nsga2
from biocomp.nsga2jax.pareto import pareto_front_mask

# =============================================================================
# Test Problems with Known Pareto Fronts
# =============================================================================


def zdt1(x: jax.Array) -> jax.Array:
    """ZDT1: convex Pareto front.

    True Pareto front: f2 = 1 - sqrt(f1), f1 in [0, 1]
    Optimal x: x[0] in [0,1], x[1:] = 0
    """
    f1 = x[:, 0]
    g = 1.0 + 9.0 * jnp.mean(x[:, 1:], axis=1)
    f2 = g * (1.0 - jnp.sqrt(f1 / g))
    return jnp.stack([f1, f2], axis=1)


def zdt2(x: jax.Array) -> jax.Array:
    """ZDT2: non-convex Pareto front.

    True Pareto front: f2 = 1 - f1^2, f1 in [0, 1]
    Optimal x: x[0] in [0,1], x[1:] = 0
    """
    f1 = x[:, 0]
    g = 1.0 + 9.0 * jnp.mean(x[:, 1:], axis=1)
    f2 = g * (1.0 - (f1 / g) ** 2)
    return jnp.stack([f1, f2], axis=1)


def binh_korn(x: jax.Array) -> jax.Array:
    """Binh-Korn problem (2D decision space, 2 objectives).

    Known Pareto front shape, bounded problem.
    """
    x1, x2 = x[:, 0], x[:, 1]
    f1 = 4 * x1**2 + 4 * x2**2
    f2 = (x1 - 5) ** 2 + (x2 - 5) ** 2
    return jnp.stack([f1, f2], axis=1)


def schaffer_n1(x: jax.Array) -> jax.Array:
    """Schaffer N1: simple 1D problem with known front.

    True Pareto front: x in [0, 2], gives f1=x^2, f2=(x-2)^2
    """
    x1 = x[:, 0]
    f1 = x1**2
    f2 = (x1 - 2) ** 2
    return jnp.stack([f1, f2], axis=1)


def sphere_sphere(x: jax.Array) -> jax.Array:
    """Two sphere functions centered at 0 and 1.

    True Pareto front: x on line from origin to ones vector.
    f1 = sum(x^2), f2 = sum((x-1)^2)
    """
    f1 = jnp.sum(x**2, axis=1)
    f2 = jnp.sum((x - 1) ** 2, axis=1)
    return jnp.stack([f1, f2], axis=1)


# =============================================================================
# Metrics for Pareto Front Quality
# =============================================================================


def generational_distance(obtained: jax.Array, true_front: jax.Array) -> float:
    """Average distance from obtained front to true front."""
    distances = jax.vmap(lambda o: jnp.min(jnp.sum((true_front - o) ** 2, axis=1)))(obtained)
    return float(jnp.sqrt(jnp.mean(distances)))


def hypervolume_2d(front: jax.Array, ref_point: jax.Array) -> float:
    """Compute 2D hypervolume indicator (area dominated by front)."""
    valid = jnp.all(front < ref_point, axis=1)
    front = front[valid]
    if front.shape[0] == 0:
        return 0.0

    sorted_idx = jnp.argsort(front[:, 0])
    front = front[sorted_idx]

    hv = 0.0
    prev_x = 0.0
    for i in range(front.shape[0]):
        width = front[i, 0] - prev_x
        height = ref_point[1] - front[i, 1]
        hv += width * height
        prev_x = front[i, 0]

    hv += (ref_point[0] - prev_x) * (ref_point[1] - front[-1, 1])
    return float(hv)


# =============================================================================
# Integration Tests
# =============================================================================


class TestZDT1:
    """Test NSGA-II on ZDT1 problem."""

    def test_convergence(self):
        n_dims, pop_size, n_gens = 30, 100, 100
        lb, ub = jnp.zeros(n_dims), jnp.ones(n_dims)

        key = jax.random.key(42)
        init_pop = jax.random.uniform(key, (pop_size, n_dims)) * (ub - lb) + lb
        init_fitness = zdt1(init_pop)

        init_front_mask = pareto_front_mask(init_fitness)
        init_front = init_fitness[init_front_mask]

        true_f1 = jnp.linspace(0, 1, 100)
        true_front = jnp.stack([true_f1, 1 - jnp.sqrt(true_f1)], axis=1)
        init_gd = generational_distance(init_front, true_front)

        state = run_nsga2(
            key=key,
            fitness_fn=zdt1,
            pop_size=pop_size,
            n_dims=n_dims,
            n_objectives=2,
            lb=lb,
            ub=ub,
            n_generations=n_gens,
        )

        final_front_mask = pareto_front_mask(state.fitness)
        final_front = state.fitness[final_front_mask]
        final_gd = generational_distance(final_front, true_front)

        assert final_gd < init_gd * 0.1, (
            f"GD improved insufficiently: {init_gd:.4f} -> {final_gd:.4f}"
        )
        assert final_gd < 0.05, f"Final GD too high: {final_gd:.4f}"

        assert jnp.all(final_front[:, 0] >= -0.01)
        assert jnp.all(final_front[:, 0] <= 1.01)

    def test_speed_jit(self):
        """Verify JIT compilation provides speedup."""
        n_dims, pop_size = 30, 100
        lb, ub = jnp.zeros(n_dims), jnp.ones(n_dims)

        algo = NSGA2(pop_size, n_dims, 2, lb, ub)
        key = jax.random.key(0)
        state = algo.init(key)

        @jax.jit
        def step(key, state):
            offspring = algo.ask(key, state)
            fitness = zdt1(offspring)
            return algo.tell(state, offspring, fitness)

        key, k = jax.random.split(key)
        _ = step(k, state)

        n_steps = 50
        start = time.perf_counter()
        for i in range(n_steps):
            key, k = jax.random.split(key)
            state = step(k, state)
        state.fitness.block_until_ready()
        elapsed = time.perf_counter() - start

        ms_per_gen = (elapsed / n_steps) * 1000
        assert ms_per_gen < 50, f"Too slow: {ms_per_gen:.1f} ms/gen (expected < 50ms)"


class TestZDT2:
    """Test NSGA-II on ZDT2 (non-convex front)."""

    def test_convergence(self):
        n_dims, pop_size, n_gens = 30, 100, 100
        lb, ub = jnp.zeros(n_dims), jnp.ones(n_dims)

        key = jax.random.key(123)

        state = run_nsga2(
            key=key,
            fitness_fn=zdt2,
            pop_size=pop_size,
            n_dims=n_dims,
            n_objectives=2,
            lb=lb,
            ub=ub,
            n_generations=n_gens,
        )

        final_front_mask = pareto_front_mask(state.fitness)
        final_front = state.fitness[final_front_mask]

        true_f1 = jnp.linspace(0, 1, 100)
        true_front = jnp.stack([true_f1, 1 - true_f1**2], axis=1)
        final_gd = generational_distance(final_front, true_front)

        assert final_gd < 0.05, f"GD too high for ZDT2: {final_gd:.4f}"


class TestSchafferN1:
    """Test on simple 1D Schaffer problem."""

    def test_finds_pareto_front(self):
        n_dims, pop_size, n_gens = 1, 50, 50
        lb, ub = jnp.array([-10.0]), jnp.array([10.0])

        key = jax.random.key(0)

        state = run_nsga2(
            key=key,
            fitness_fn=schaffer_n1,
            pop_size=pop_size,
            n_dims=n_dims,
            n_objectives=2,
            lb=lb,
            ub=ub,
            n_generations=n_gens,
        )

        front_mask = pareto_front_mask(state.fitness)
        front_x = state.population[front_mask]

        assert jnp.all(front_x >= -0.5), "Front x values should be >= 0"
        assert jnp.all(front_x <= 2.5), "Front x values should be <= 2"

        x_spread = jnp.max(front_x) - jnp.min(front_x)
        assert x_spread > 1.0, f"Poor diversity: x spread = {x_spread:.2f}"


class TestSphereSphere:
    """Test on bi-sphere problem."""

    def test_pareto_front_on_diagonal(self):
        n_dims, pop_size, n_gens = 10, 80, 80
        lb, ub = jnp.zeros(n_dims), jnp.ones(n_dims)

        key = jax.random.key(999)

        state = run_nsga2(
            key=key,
            fitness_fn=sphere_sphere,
            pop_size=pop_size,
            n_dims=n_dims,
            n_objectives=2,
            lb=lb,
            ub=ub,
            n_generations=n_gens,
        )

        front_mask = pareto_front_mask(state.fitness)
        front_x = state.population[front_mask]

        for x in front_x:
            std_dev = jnp.std(x)
            assert std_dev < 0.15, f"Solution not on diagonal: std={std_dev:.3f}"


class TestBinhKorn:
    """Test on Binh-Korn problem."""

    def test_bounded_convergence(self):
        n_dims, pop_size, n_gens = 2, 60, 60
        lb, ub = jnp.zeros(n_dims), jnp.array([5.0, 3.0])

        key = jax.random.key(42)

        init_pop = jax.random.uniform(key, (pop_size, n_dims)) * (ub - lb) + lb
        init_fitness = binh_korn(init_pop)
        init_hv = hypervolume_2d(init_fitness, jnp.array([150.0, 50.0]))

        state = run_nsga2(
            key=key,
            fitness_fn=binh_korn,
            pop_size=pop_size,
            n_dims=n_dims,
            n_objectives=2,
            lb=lb,
            ub=ub,
            n_generations=n_gens,
        )

        final_hv = hypervolume_2d(state.fitness, jnp.array([150.0, 50.0]))

        assert final_hv > init_hv, f"Hypervolume didn't improve: {init_hv:.1f} -> {final_hv:.1f}"


class TestPerformance:
    """Performance benchmarks."""

    def test_large_population(self):
        """Test with large population (500 individuals)."""
        n_dims, pop_size, n_gens = 30, 500, 20
        lb, ub = jnp.zeros(n_dims), jnp.ones(n_dims)

        key = jax.random.key(0)

        start = time.perf_counter()
        state = run_nsga2(
            key=key,
            fitness_fn=zdt1,
            pop_size=pop_size,
            n_dims=n_dims,
            n_objectives=2,
            lb=lb,
            ub=ub,
            n_generations=n_gens,
        )
        state.fitness.block_until_ready()
        elapsed = time.perf_counter() - start

        ms_per_gen = (elapsed / n_gens) * 1000
        print(f"\nLarge pop ({pop_size}): {ms_per_gen:.1f} ms/gen")

        assert ms_per_gen < 500, f"Too slow for large pop: {ms_per_gen:.1f} ms/gen"

    def test_many_objectives(self):
        """Test with 5 objectives."""

        def five_obj(x):
            return jnp.stack([jnp.sum((x - i * 0.2) ** 2, axis=1) for i in range(5)], axis=1)

        n_dims, pop_size, n_gens = 10, 100, 30
        lb, ub = jnp.zeros(n_dims), jnp.ones(n_dims)

        key = jax.random.key(0)
        state = run_nsga2(
            key=key,
            fitness_fn=five_obj,
            pop_size=pop_size,
            n_dims=n_dims,
            n_objectives=5,
            lb=lb,
            ub=ub,
            n_generations=n_gens,
        )

        assert state.fitness.shape == (pop_size, 5)
        assert jnp.all(jnp.isfinite(state.fitness))

    def test_vmap_multiple_runs(self):
        """Demonstrate vmapping over multiple independent runs."""
        n_dims, pop_size, n_gens = 10, 50, 30
        n_runs = 5
        lb, ub = jnp.zeros(n_dims), jnp.ones(n_dims)

        algo = NSGA2(pop_size, n_dims, 2, lb, ub)

        def single_run(key):
            state = algo.init(key)

            def step(_, carry):
                state, key = carry
                key, k = jax.random.split(key)
                offspring = algo.ask(k, state)
                fitness = sphere_sphere(offspring)
                new_state = algo.tell(state, offspring, fitness)
                return (new_state, key)

            final_state, _ = jax.lax.fori_loop(0, n_gens, step, (state, key))
            return final_state.fitness

        keys = jax.random.split(jax.random.key(0), n_runs)

        start = time.perf_counter()
        all_fitness = jax.vmap(single_run)(keys)
        all_fitness.block_until_ready()
        elapsed = time.perf_counter() - start

        assert all_fitness.shape == (n_runs, pop_size, 2)
        print(f"\n{n_runs} parallel runs x {n_gens} gens: {elapsed * 1000:.1f} ms total")


def test_full_example_zdt1():
    """Complete example showing NSGA-II solves ZDT1."""
    print("\n" + "=" * 60)
    print("ZDT1 Optimization Demo")
    print("=" * 60)

    n_dims, pop_size, n_gens = 30, 100, 100
    lb, ub = jnp.zeros(n_dims), jnp.ones(n_dims)
    key = jax.random.key(42)

    init_key, run_key = jax.random.split(key)
    init_pop = jax.random.uniform(init_key, (pop_size, n_dims))
    init_fitness = zdt1(init_pop)

    true_f1 = jnp.linspace(0, 1, 100)
    true_front = jnp.stack([true_f1, 1 - jnp.sqrt(true_f1)], axis=1)

    init_gd = generational_distance(init_fitness[pareto_front_mask(init_fitness)], true_front)
    print(f"Initial GD: {init_gd:.4f}")

    start = time.perf_counter()
    state = run_nsga2(
        key=run_key,
        fitness_fn=zdt1,
        pop_size=pop_size,
        n_dims=n_dims,
        n_objectives=2,
        lb=lb,
        ub=ub,
        n_generations=n_gens,
    )
    state.fitness.block_until_ready()
    elapsed = time.perf_counter() - start

    final_front = state.fitness[pareto_front_mask(state.fitness)]
    final_gd = generational_distance(final_front, true_front)

    print(f"Final GD: {final_gd:.4f}")
    print(f"Improvement: {init_gd / final_gd:.1f}x better")
    print(f"Time: {elapsed:.2f}s ({elapsed / n_gens * 1000:.1f} ms/gen)")
    print(f"Pareto front size: {final_front.shape[0]} solutions")
    print(f"f1 range: [{final_front[:, 0].min():.3f}, {final_front[:, 0].max():.3f}]")
    print(f"f2 range: [{final_front[:, 1].min():.3f}, {final_front[:, 1].max():.3f}]")

    assert final_gd < 0.05
    assert final_gd < init_gd * 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
