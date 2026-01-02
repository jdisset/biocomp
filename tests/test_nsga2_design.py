"""Tests for NSGA2 multi-objective design optimizer."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from unittest.mock import Mock

from biocomp.designoptim import NSGA2DesignOptimizer, NSGA2DesignState


class TestNSGA2DesignOptimizer:
    """Unit tests for NSGA2DesignOptimizer."""

    @pytest.fixture
    def simple_objective(self):
        """Simple 2-objective function for testing."""
        def objective(genome: jnp.ndarray) -> float:
            return jnp.sum(genome**2)
        return objective

    @pytest.fixture
    def optimizer(self):
        return NSGA2DesignOptimizer(
            pop_size=16,
            n_generations=10,
            gd_steps_per_individual=5,
            gd_learning_rate=0.01,
        )

    def test_init_with_provided_dims(self, optimizer, simple_objective):
        """Test initialization with explicitly provided dimensions."""
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

    def test_init_auto_detect_continuous_dim(self, simple_objective):
        """Test auto-detection of continuous dimension."""
        optimizer = NSGA2DesignOptimizer(
            pop_size=8,
            n_generations=5,
            gd_steps_per_individual=2,
        )
        n_tus = 3
        continuous_dim = 5
        total_dim = n_tus + continuous_dim

        key = jax.random.key(0)
        params = jnp.zeros(total_dim)

        state = optimizer.init(key, params, simple_objective, n_tus=n_tus)

        assert optimizer.continuous_dim == continuous_dim

    def test_genes_to_mask(self, optimizer, simple_objective):
        """Test binary mask generation from genes."""
        n_tus = 4
        key = jax.random.key(0)
        params = jnp.zeros(n_tus + 4)

        _ = optimizer.init(key, params, simple_objective, n_tus=n_tus)

        genes_low = jnp.array([0.1, 0.2, 0.3, 0.4])
        genes_high = jnp.array([0.6, 0.7, 0.8, 0.9])

        mask_low = optimizer._genes_to_mask(genes_low)
        mask_high = optimizer._genes_to_mask(genes_high)

        assert jnp.allclose(mask_low, jnp.array([0.0, 0.0, 0.0, 0.0]))
        assert jnp.allclose(mask_high, jnp.array([1.0, 1.0, 1.0, 1.0]))

    def test_step_returns_metrics(self, optimizer, simple_objective):
        """Test that step returns expected metrics."""
        n_tus = 2
        continuous_dim = 4
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
        """Test running multiple generations."""
        optimizer = NSGA2DesignOptimizer(
            pop_size=8,
            n_generations=5,
            gd_steps_per_individual=2,
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

    def test_should_stop(self, optimizer, simple_objective):
        """Test stopping condition."""
        n_tus = 2
        key = jax.random.key(0)
        params = jnp.zeros(n_tus + 4)

        state = optimizer.init(key, params, simple_objective, n_tus=n_tus)

        assert not optimizer.should_stop(state)

        state = state._replace(step=jnp.array(optimizer.n_generations))
        assert optimizer.should_stop(state)

    def test_pareto_front_extraction(self, simple_objective):
        """Test that pareto front is correctly extracted."""
        optimizer = NSGA2DesignOptimizer(
            pop_size=16,
            n_generations=10,
            gd_steps_per_individual=3,
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

        for i in range(5):
            key, step_key = jax.random.split(key)
            state, metrics = optimizer.step(state, step_key, simple_objective)

        pareto_front, pareto_fitness = optimizer.get_pareto_front(state)

        assert pareto_front is not None or state.pareto_front is None
        if pareto_front is not None:
            assert pareto_front.shape[0] > 0
            assert pareto_fitness.shape[1] == 2

    def test_fitness_objectives(self, simple_objective):
        """Test that fitness has two objectives: loss and TU count."""
        optimizer = NSGA2DesignOptimizer(
            pop_size=8,
            n_generations=3,
            gd_steps_per_individual=2,
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


class TestNSGA2DesignIntegration:
    """Integration tests for NSGA2 design optimizer."""

    def test_with_design_like_objective(self):
        """Test with an objective similar to design loss."""
        def design_objective(genome: jnp.ndarray) -> float:
            n_tus = 5
            tu_mask = (genome[:n_tus] > 0.5).astype(jnp.float32)
            continuous = genome[n_tus:]

            pattern_loss = jnp.sum((continuous - 0.5)**2) / continuous.shape[0]

            disabled_penalty = jnp.sum(1 - tu_mask) * 0.01

            return pattern_loss + disabled_penalty

        optimizer = NSGA2DesignOptimizer(
            pop_size=16,
            n_generations=20,
            gd_steps_per_individual=10,
            gd_learning_rate=0.05,
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

        assert state.best_loss < initial_loss, "Optimization should improve loss"

        pareto_fitness = state.pareto_fitness
        if pareto_fitness is not None:
            losses = pareto_fitness[:, 0]
            tu_counts = pareto_fitness[:, 1]

            assert np.any(losses < initial_loss), "Some pareto solutions should have better loss"
            assert np.min(tu_counts) >= 0, "TU count should be non-negative"
            assert np.max(tu_counts) <= n_tus, "TU count should not exceed n_tus"


def test_state_is_finite():
    """Test that state maintains finite values."""
    optimizer = NSGA2DesignOptimizer(
        pop_size=8,
        n_generations=5,
        gd_steps_per_individual=2,
    )

    def objective(genome: jnp.ndarray) -> float:
        return jnp.sum(genome**2)

    n_tus = 2
    key = jax.random.key(0)
    params = jnp.zeros(n_tus + 4)

    state = optimizer.init(key, params, objective, n_tus=n_tus)

    assert state.is_finite()

    for _ in range(3):
        key, step_key = jax.random.split(key)
        state, _ = optimizer.step(state, step_key, objective)
        assert state.is_finite(), "State should remain finite during optimization"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
