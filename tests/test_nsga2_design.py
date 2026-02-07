"""Tests for NSGA2 multi-objective design optimizer."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from biocomp.pluggable_opt.optimizers import (
    NSGA2DesignOptimizer,
    NSGA2DesignState,
    InnerGDConfig,
    genes_to_mask,
)
from biocomp.optimutils import create_gd_step_fn


class TestGDStepFunction:
    def test_create_gd_step_fn_reduces_loss(self):
        optimizer = optax.adam(0.01)
        gd_step = create_gd_step_fn(optimizer)
        params = jnp.ones(5)
        opt_state = optimizer.init(params)
        new_params, _, loss = gd_step(params, opt_state, lambda p: jnp.sum(p**2))
        assert jnp.all(new_params < params)
        assert loss > 0


class TestGenesToMask:
    def test_threshold_behavior(self):
        genes = jnp.array([0.3, 0.5, 0.7, 0.51, 0.49])
        mask = genes_to_mask(genes)
        assert mask.tolist() == [0.0, 0.0, 1.0, 1.0, 0.0]

    def test_custom_threshold(self):
        genes = jnp.array([0.3, 0.5, 0.7])
        mask = genes_to_mask(genes, threshold=0.4)
        assert mask.tolist() == [0.0, 1.0, 1.0]


class TestInnerGDConfig:
    def test_default_config(self):
        cfg = InnerGDConfig()
        assert cfg.n_steps == 50
        assert cfg.n_replicates == 1
        assert cfg.optimizer is not None

    def test_custom_steps(self):
        cfg = InnerGDConfig(n_steps=30, n_replicates=4)
        assert cfg.n_steps == 30
        assert cfg.n_replicates == 4


class TestNSGA2DesignOptimizer:

    @pytest.fixture
    def simple_objective(self):
        def objective(genome: jnp.ndarray) -> float:
            return jnp.sum(genome**2)
        return objective

    @pytest.fixture
    def optimizer(self):
        return NSGA2DesignOptimizer(
            pop_size=16,
            n_generations=10,
            n_tus=4,
            continuous_dim=6,
        )

    def test_init_with_provided_dims(self, optimizer, simple_objective):
        n_tus = 4
        continuous_dim = 6
        total_dim = n_tus + continuous_dim

        key = jax.random.key(42)
        params = jnp.zeros(total_dim)

        state = optimizer.init(
            key, params, simple_objective,
            n_tus=n_tus,
            continuous_dim=continuous_dim,
        )

        assert isinstance(state, NSGA2DesignState)
        assert state.step == 0
        assert state.params.shape == (total_dim,)
        assert optimizer.n_tus == n_tus
        assert optimizer.continuous_dim == continuous_dim

    def test_init_requires_dimensions(self, simple_objective):
        optimizer = NSGA2DesignOptimizer(pop_size=8, n_generations=5)
        key = jax.random.key(0)
        params = jnp.zeros(8)

        with pytest.raises(AssertionError, match="n_tus required"):
            optimizer.init(key, params, simple_objective)

    def test_step_returns_metrics(self, optimizer, simple_objective):
        n_tus = 4
        continuous_dim = 6
        total_dim = n_tus + continuous_dim

        key = jax.random.key(123)
        params = jnp.ones(total_dim) * 0.5

        state = optimizer.init(
            key, params, simple_objective,
            n_tus=n_tus,
            continuous_dim=continuous_dim,
        )

        step_key = jax.random.key(456)
        new_state, metrics = optimizer.step(state, step_key, simple_objective)

        assert new_state.step == state.step + 1
        assert "gen_best_loss" in metrics
        assert "gen_best_tu_count" in metrics
        assert "pareto_size" in metrics
        assert "phase" in metrics

    def test_multi_generation_run(self, simple_objective):
        optimizer = NSGA2DesignOptimizer(
            pop_size=8,
            n_generations=5,
            n_tus=2,
            continuous_dim=3,
            inner_gd=InnerGDConfig(n_steps=2),
        )

        n_tus = 2
        continuous_dim = 3
        total_dim = n_tus + continuous_dim

        key = jax.random.key(0)
        params = jnp.zeros(total_dim)

        state = optimizer.init(
            key, params, simple_objective,
            n_tus=n_tus,
            continuous_dim=continuous_dim,
        )

        initial_loss = float(state.best_loss)

        for _ in range(3):
            key, step_key = jax.random.split(key)
            state, _ = optimizer.step(state, step_key, simple_objective)

        assert state.step == 3
        assert state.best_loss <= initial_loss

    def test_should_stop(self, simple_objective):
        optimizer = NSGA2DesignOptimizer(
            pop_size=8,
            n_generations=10,
            n_tus=2,
            continuous_dim=4,
        )
        key = jax.random.key(0)
        params = jnp.zeros(6)

        state = optimizer.init(key, params, simple_objective, n_tus=2, continuous_dim=4)

        assert not optimizer.should_stop(state)

        state = state._replace(step=jnp.array(optimizer.n_generations))
        assert optimizer.should_stop(state)

    def test_pareto_front_extraction(self, simple_objective):
        optimizer = NSGA2DesignOptimizer(
            pop_size=16,
            n_generations=10,
            n_tus=3,
            continuous_dim=5,
            inner_gd=InnerGDConfig(n_steps=3),
        )

        n_tus = 3
        continuous_dim = 5
        total_dim = n_tus + continuous_dim

        key = jax.random.key(42)
        params = jnp.zeros(total_dim)

        state = optimizer.init(
            key, params, simple_objective,
            n_tus=n_tus,
            continuous_dim=continuous_dim,
        )

        for _i in range(5):
            key, step_key = jax.random.split(key)
            state, metrics = optimizer.step(state, step_key, simple_objective)

        pareto_front, pareto_fitness = optimizer.get_pareto_front(state)

        assert pareto_front is not None or state.pareto_front is None
        if pareto_front is not None:
            assert pareto_front.shape[0] > 0
            assert pareto_fitness.shape[1] == 2

    def test_fitness_objectives(self, simple_objective):
        optimizer = NSGA2DesignOptimizer(
            pop_size=8,
            n_generations=3,
            n_tus=4,
            continuous_dim=4,
            inner_gd=InnerGDConfig(n_steps=2),
        )

        n_tus = 4
        continuous_dim = 4
        total_dim = n_tus + continuous_dim

        key = jax.random.key(0)
        params = jnp.zeros(total_dim)

        state = optimizer.init(
            key, params, simple_objective,
            n_tus=n_tus,
            continuous_dim=continuous_dim,
        )

        key, step_key = jax.random.split(key)
        new_state, metrics = optimizer.step(state, step_key, simple_objective)

        tu_count = metrics["gen_best_tu_count"]
        assert 0 <= tu_count <= n_tus

    def test_inner_gd_config_used(self, simple_objective):
        inner_gd = InnerGDConfig(n_steps=10, n_replicates=2, init_perturbation=0.2)
        optimizer = NSGA2DesignOptimizer(
            pop_size=8,
            n_generations=3,
            n_tus=2,
            continuous_dim=4,
            inner_gd=inner_gd,
        )
        assert optimizer.inner_gd.n_steps == 10
        assert optimizer.inner_gd.n_replicates == 2

    def test_continuous_bounds_configurable(self, simple_objective):
        optimizer = NSGA2DesignOptimizer(
            pop_size=8,
            n_generations=3,
            n_tus=2,
            continuous_dim=4,
            continuous_bounds=(-5.0, 5.0),
        )
        assert optimizer.continuous_bounds == (-5.0, 5.0)


class TestNSGA2DesignIntegration:

    def test_with_design_like_objective(self):
        def design_objective(genome: jnp.ndarray) -> float:
            n_tus = 5
            tu_mask = genes_to_mask(genome[:n_tus])
            continuous = genome[n_tus:]
            pattern_loss = jnp.sum((continuous - 0.5)**2) / continuous.shape[0]
            disabled_penalty = jnp.sum(1 - tu_mask) * 0.01
            return pattern_loss + disabled_penalty

        optimizer = NSGA2DesignOptimizer(
            pop_size=16,
            n_generations=20,
            n_tus=5,
            continuous_dim=10,
            inner_gd=InnerGDConfig(n_steps=10),
        )

        n_tus = 5
        continuous_dim = 10
        total_dim = n_tus + continuous_dim

        key = jax.random.key(42)
        params = jax.random.uniform(key, (total_dim,))

        state = optimizer.init(
            key, params, design_objective,
            n_tus=n_tus,
            continuous_dim=continuous_dim,
        )

        initial_loss = float(state.best_loss)

        for _ in range(10):
            key, step_key = jax.random.split(key)
            state, metrics = optimizer.step(state, step_key, design_objective)

        assert state.best_loss < initial_loss

        pareto_fitness = state.pareto_fitness
        if pareto_fitness is not None:
            losses = pareto_fitness[:, 0]
            tu_counts = pareto_fitness[:, 1]
            assert np.any(losses < initial_loss)
            assert np.min(tu_counts) >= 0
            assert np.max(tu_counts) <= n_tus


def test_state_is_finite():
    optimizer = NSGA2DesignOptimizer(
        pop_size=8,
        n_generations=5,
        n_tus=2,
        continuous_dim=4,
        inner_gd=InnerGDConfig(n_steps=2),
    )

    def objective(genome: jnp.ndarray) -> float:
        return jnp.sum(genome**2)

    n_tus = 2
    continuous_dim = 4
    key = jax.random.key(0)
    params = jnp.zeros(n_tus + continuous_dim)

    state = optimizer.init(key, params, objective, n_tus=n_tus, continuous_dim=continuous_dim)

    assert state.is_finite()

    for _ in range(3):
        key, step_key = jax.random.split(key)
        state, _ = optimizer.step(state, step_key, objective)
        assert state.is_finite()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
