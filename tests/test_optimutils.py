#!/usr/bin/env python3
"""Tests for the optimutils module (biocomp/optimutils.py)."""

import pytest
import jax
import jax.numpy as jnp
import optax

from biocomp.optimutils import (
    OptimConfig,
    DesignOptimConfig,
    build_optimizer_chain,
    create_counter,
    as_schedule,
    compile_step,
    run_logger_callbacks,
    get_checkify_enabled,
)
from biocomp.utils import PartialFunction, PartialFunctionResult


class TestOptimConfig:
    """Test OptimConfig base class functionality."""

    def test_default_config(self):
        """Test default OptimConfig values."""
        config = OptimConfig()
        assert config.seed is not None
        assert config.batches_per_step == 4
        assert config.batch_size == 32
        assert config.n_epochs == 3
        assert config.n_replicates == 16
        assert config.keep_in_history == ["loss"]

    def test_seed_key_property(self):
        """Test seed_key returns consistent PRNGKey."""
        config = OptimConfig(seed=42)
        key1 = config.seed_key
        key2 = config.seed_key
        assert jnp.array_equal(jax.random.key_data(key1), jax.random.key_data(key2))

    def test_optimizer_property(self):
        """Test optimizer property creates valid optimizer chain."""
        config = OptimConfig(
            optimizer_stack=[
                PartialFunction(func="optax.sgd", kwargs={"learning_rate": 0.01})
            ]
        )
        optimizer = config.optimizer
        assert hasattr(optimizer, "init")
        assert hasattr(optimizer, "update")

        params = {"w": jnp.ones(3)}
        state = optimizer.init(params)
        assert state is not None

    def test_lr_injection(self):
        """Test learning rate injection for tracking."""
        config = OptimConfig(
            optimizer_stack=[
                PartialFunction(func="optax.adamw", kwargs={"learning_rate": 1e-3})
            ]
        )
        optimizer = config.create_optimizer_with_lr_injection()
        params = {"w": jnp.ones(3)}
        state = optimizer.init(params)

        lr = optax.tree_utils.tree_get(state, "learning_rate", default=None)
        assert lr is not None
        assert abs(lr - 1e-3) < 1e-6


class TestDesignOptimConfig:
    """Test DesignOptimConfig subclass."""

    def test_inherits_base(self):
        """Test inheritance from OptimConfig."""
        config = DesignOptimConfig(
            loss_function=PartialFunction(func="biocomp.design.distance_loss", kwargs={})
        )
        assert hasattr(config, "optimizer")
        assert hasattr(config, "seed_key")
        assert config.n_batches_per_epoch == 128
        assert config.reshuffle_batches is True


class TestBuildOptimizerChain:
    """Test build_optimizer_chain function."""

    def test_simple_chain(self):
        """Test building simple optimizer chain."""
        stack = [PartialFunction(func="optax.sgd", kwargs={"learning_rate": 0.01})]
        optimizer = build_optimizer_chain(stack, with_lr_injection=False)
        params = {"w": jnp.ones(3)}
        state = optimizer.init(params)
        assert state is not None

    def test_chain_with_lr_injection(self):
        """Test building chain with learning rate injection."""
        stack = [PartialFunction(func="optax.adamw", kwargs={"learning_rate": 1e-3})]
        optimizer = build_optimizer_chain(stack, with_lr_injection=True)
        params = {"w": jnp.ones(3)}
        state = optimizer.init(params)

        lr = optax.tree_utils.tree_get(state, "learning_rate", default=None)
        assert lr is not None

    def test_chain_with_schedule(self):
        """Test chain with learning rate schedule."""
        stack = [
            PartialFunction(func="optax.clip_by_global_norm", kwargs={"max_norm": 1.0}),
            PartialFunction(
                func="optax.adamw",
                kwargs={
                    "learning_rate": PartialFunctionResult(
                        func="optax.warmup_cosine_decay_schedule",
                        kwargs={
                            "init_value": 1e-7,
                            "peak_value": 1e-3,
                            "warmup_steps": 10,
                            "decay_steps": 100,
                            "end_value": 1e-5,
                        },
                    )
                },
            ),
        ]
        optimizer = build_optimizer_chain(stack, with_lr_injection=True)
        params = {"w": jnp.ones(3)}
        state = optimizer.init(params)
        assert state is not None


class TestCompileStep:
    """Test compile_step function."""

    def test_compile_simple_function(self):
        """Test compiling a simple function."""

        def step_fn(x, y):
            return x + y

        sample_args = (jnp.ones(3), jnp.ones(3))
        compiled = compile_step(step_fn, sample_args, use_checkify=False)
        result = compiled(jnp.ones(3), jnp.ones(3))
        assert jnp.allclose(result, jnp.ones(3) * 2)

    def test_compile_with_checkify(self):
        """Test compiling with checkify enabled."""

        def step_fn(x, y):
            return x + y

        sample_args = (jnp.ones(3), jnp.ones(3))
        compiled = compile_step(step_fn, sample_args, use_checkify=True)
        result = compiled(jnp.ones(3), jnp.ones(3))
        assert jnp.allclose(result, jnp.ones(3) * 2)


class TestRunLoggerCallbacks:
    """Test run_logger_callbacks function."""

    def test_period_filtering(self):
        """Test callbacks are filtered by period correctly."""
        called_periods = []

        def make_callback(period):
            def callback(step, config, step_history, stack):
                called_periods.append((period, step))

            return callback

        loggers = [
            (1, make_callback(1)),  # every step
            (5, make_callback(5)),  # every 5 steps
            (10, make_callback(10)),  # every 10 steps
        ]

        # Simulate step 10
        run_logger_callbacks(
            loggers,
            step=10,
            config=None,
            step_history={},
            stack=None,
            period_filter=lambda p, s: p > 0 and s % p == 0,
        )

        assert (1, 10) in called_periods
        assert (5, 10) in called_periods
        assert (10, 10) in called_periods

    def test_start_logger(self):
        """Test start loggers (period=0) are called correctly."""
        called = []

        def start_callback(step, config, step_history, stack):
            called.append("start")

        loggers = [(0, start_callback), (1, lambda *a, **kw: None)]

        run_logger_callbacks(
            loggers, step=0, config=None, step_history={}, stack=None, period_filter=lambda p, s: p == 0
        )

        assert called == ["start"]

    def test_end_logger(self):
        """Test end loggers (period=-1 or None) are called correctly."""
        called = []

        def end_callback(step, config, step_history, stack):
            called.append("end")

        loggers = [(-1, end_callback), (None, end_callback)]

        run_logger_callbacks(
            loggers,
            step=100,
            config=None,
            step_history={},
            stack=None,
            period_filter=lambda p, s: p is None or p == -1,
        )

        assert called == ["end", "end"]

    def test_callback_error_handling(self):
        """Test that callback errors are logged and re-raised."""

        def failing_callback(step, config, step_history, stack):
            raise ValueError("Test error")

        loggers = [(1, failing_callback)]

        with pytest.raises(ValueError, match="Test error"):
            run_logger_callbacks(
                loggers,
                step=1,
                config=None,
                step_history={},
                stack=None,
                period_filter=lambda p, s: p > 0 and s % p == 0,
            )


class TestAsSchedule:
    """Test as_schedule utility."""

    def test_scalar_to_schedule(self):
        """Test converting scalar to schedule."""
        schedule = as_schedule(0.01)
        assert abs(float(schedule(0)) - 0.01) < 1e-6
        assert abs(float(schedule(100)) - 0.01) < 1e-6

    def test_callable_passthrough(self):
        """Test callable is passed through unchanged."""

        def my_schedule(step):
            return 0.01 * (1 - step / 100)

        schedule = as_schedule(my_schedule)
        assert float(schedule(0)) == 0.01
        assert abs(float(schedule(50)) - 0.005) < 1e-6


class TestCreateCounter:
    """Test create_counter transformation."""

    def test_counter_init(self):
        """Test counter initialization."""
        counter = create_counter()
        params = {"w": jnp.ones(3)}
        state = counter.init(params)
        assert hasattr(state, "count")
        assert state.count == 0

    def test_counter_increment(self):
        """Test counter increments on update."""
        counter = create_counter()
        params = {"w": jnp.ones(3)}
        state = counter.init(params)

        grads = {"w": jnp.ones(3) * 0.1}
        updates, new_state = counter.update(grads, state)

        assert new_state.count == 1
        assert jnp.allclose(updates["w"], grads["w"])

    def test_counter_multiple_updates(self):
        """Test counter increments correctly over multiple updates."""
        counter = create_counter()
        params = {"w": jnp.ones(3)}
        state = counter.init(params)
        grads = {"w": jnp.ones(3) * 0.1}

        for i in range(5):
            _, state = counter.update(grads, state)

        assert state.count == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
