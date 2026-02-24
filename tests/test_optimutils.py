#!/usr/bin/env python3
"""Tests for the optimutils module (biocomp/optimutils.py)."""

import pytest
import jax
import jax.numpy as jnp
import optax
from typing import cast

from biocomp.optimutils import (
    OptimConfig,
    DesignOptimConfig,
    build_optimizer_chain,
    create_counter,
    set_optimizer_state_step,
    as_schedule,
    compile_step,
    optimize,
)
from biocomp.logger_dispatch import NullDispatch, LoggerDispatch
from biocomp.compute import ComputeStack
from biocomp.step_history import StepHistorySnapshot
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


class TestNullDispatch:
    """Test NullDispatch (no-op logger dispatch)."""

    def test_satisfies_protocol(self):
        dispatch = NullDispatch()
        assert isinstance(dispatch, LoggerDispatch)

    def test_on_start_is_noop(self):
        NullDispatch().on_start(None, None)

    def test_on_step_is_noop(self):
        NullDispatch().on_step(1, None, {}, None)

    def test_on_end_is_noop(self):
        NullDispatch().on_end(100, None, {}, None)

    def test_needs_params_sync_always_false(self):
        dispatch = NullDispatch()
        assert dispatch.needs_params_sync(0) is False
        assert dispatch.needs_params_sync(10) is False
        assert dispatch.needs_params_sync(100) is False


class TestOptimizeContract:
    """Return-contract tests for optimize()."""

    def test_optimize_returns_snapshot_step_history(self):
        def step(params, opt_state, step_key, xb, yb):
            del step_key, xb, yb
            return params, opt_state, {"loss": jnp.array([[1.0]])}

        cfg = DesignOptimConfig(
            loss_function=PartialFunction(func="biocomp.design.distance_loss", kwargs={}),
            n_replicates=1,
            batches_per_step=1,
            n_batches_per_epoch=1,
            n_epochs=1,
            reshuffle_batches=False,
        )

        params, loss_history, step_history = optimize(
            step=step,
            params=jnp.array([1.0]),
            opt_state=None,
            xbatches=jnp.zeros((1, 1, 1, 1)),
            ybatches=jnp.zeros((1, 1, 1, 1)),
            config=cfg,
            n_total_steps=1,
            steps_per_epoch=1,
            key=jax.random.PRNGKey(0),
            stack=cast(ComputeStack, object()),
            dispatch=NullDispatch(),
            precompiled=True,
        )

        assert params.shape == (1,)
        assert len(loss_history) == 1
        assert isinstance(step_history, StepHistorySnapshot)
        assert "loss" in step_history


class TestOptimizeBestSyncedParams:
    def _make_cfg(self):
        return DesignOptimConfig(
            loss_function=PartialFunction(func="biocomp.design.distance_loss", kwargs={}),
            n_replicates=1,
            batches_per_step=1,
            n_batches_per_epoch=5,
            n_epochs=1,
            reshuffle_batches=False,
        )

    def test_returns_final_params_by_default(self):
        def step(params, opt_state, step_key, xb, yb):
            del opt_state, step_key, xb, yb
            next_params = params + 1.0
            loss = jnp.mean((next_params - 3.0) ** 2)
            return next_params, None, {"loss": loss}

        params, _, _ = optimize(
            step=step,
            params=jnp.array([0.0]),
            opt_state=None,
            xbatches=jnp.zeros((5, 1, 1, 1)),
            ybatches=jnp.zeros((5, 1, 1, 1)),
            config=self._make_cfg(),
            n_total_steps=5,
            steps_per_epoch=5,
            key=jax.random.PRNGKey(0),
            stack=cast(ComputeStack, object()),
            dispatch=NullDispatch(),
            precompiled=True,
            defer_sync=False,
        )
        assert float(params[0]) == pytest.approx(5.0)

    def test_can_restore_best_synced_params(self):
        def step(params, opt_state, step_key, xb, yb):
            del opt_state, step_key, xb, yb
            next_params = params + 1.0
            loss = jnp.mean((next_params - 3.0) ** 2)
            return next_params, None, {"loss": loss}

        params, _, _ = optimize(
            step=step,
            params=jnp.array([0.0]),
            opt_state=None,
            xbatches=jnp.zeros((5, 1, 1, 1)),
            ybatches=jnp.zeros((5, 1, 1, 1)),
            config=self._make_cfg(),
            n_total_steps=5,
            steps_per_epoch=5,
            key=jax.random.PRNGKey(0),
            stack=cast(ComputeStack, object()),
            dispatch=NullDispatch(),
            precompiled=True,
            defer_sync=False,
            select_best_synced_params=True,
        )
        assert float(params[0]) == pytest.approx(3.0)

    def test_can_restore_best_params_from_custom_score(self):
        def step(params, opt_state, step_key, xb, yb):
            del opt_state, step_key, xb, yb
            next_params = params + 1.0
            loss = jnp.mean((next_params - 10.0) ** 2)
            return next_params, None, {"loss": loss}

        def score_fn(params, _step_history, _step):
            return float(jnp.abs(params[0] - 3.0))

        params, _, _ = optimize(
            step=step,
            params=jnp.array([0.0]),
            opt_state=None,
            xbatches=jnp.zeros((5, 1, 1, 1)),
            ybatches=jnp.zeros((5, 1, 1, 1)),
            config=self._make_cfg(),
            n_total_steps=5,
            steps_per_epoch=5,
            key=jax.random.PRNGKey(0),
            stack=cast(ComputeStack, object()),
            dispatch=NullDispatch(),
            precompiled=True,
            defer_sync=False,
            select_best_synced_params=True,
            best_synced_score_fn=score_fn,
        )
        assert float(params[0]) == pytest.approx(3.0)

    def test_best_synced_restoration_updates_step_history_latest_params(self):
        def step(params, opt_state, step_key, xb, yb):
            del opt_state, step_key, xb, yb
            next_params = params + 1.0
            loss = jnp.mean((next_params - 10.0) ** 2)
            return next_params, None, {"loss": loss, "latest_params": next_params}

        def score_fn(params, _step_history, _step):
            return float(jnp.abs(params[0] - 2.0))

        params, _, step_history = optimize(
            step=step,
            params=jnp.array([0.0]),
            opt_state=None,
            xbatches=jnp.zeros((5, 1, 1, 1)),
            ybatches=jnp.zeros((5, 1, 1, 1)),
            config=self._make_cfg(),
            n_total_steps=5,
            steps_per_epoch=5,
            key=jax.random.PRNGKey(0),
            stack=cast(ComputeStack, object()),
            dispatch=NullDispatch(),
            precompiled=True,
            defer_sync=False,
            select_best_synced_params=True,
            best_synced_score_fn=score_fn,
        )
        assert float(params[0]) == pytest.approx(2.0)
        assert "latest_params" in step_history
        assert float(step_history["latest_params"][0]) == pytest.approx(2.0)

    def test_can_restore_initial_baseline_when_all_scores_worse(self):
        def step(params, opt_state, step_key, xb, yb):
            del opt_state, step_key, xb, yb
            next_params = params + 1.0
            loss = jnp.mean((next_params - 10.0) ** 2)
            return next_params, None, {"loss": loss}

        def score_fn(params, _step_history, _step):
            del params
            return 1.0

        params, _, _ = optimize(
            step=step,
            params=jnp.array([0.0]),
            opt_state=None,
            xbatches=jnp.zeros((5, 1, 1, 1)),
            ybatches=jnp.zeros((5, 1, 1, 1)),
            config=self._make_cfg(),
            n_total_steps=5,
            steps_per_epoch=5,
            key=jax.random.PRNGKey(0),
            stack=cast(ComputeStack, object()),
            dispatch=NullDispatch(),
            precompiled=True,
            defer_sync=False,
            select_best_synced_params=True,
            best_synced_score_fn=score_fn,
            best_synced_initial_score=0.0,
        )
        assert float(params[0]) == pytest.approx(0.0)


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

        for _i in range(5):
            _, state = counter.update(grads, state)

        assert state.count == 5


class TestSetOptimizerStateStep:
    def test_sets_all_count_fields(self):
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
        optimizer = build_optimizer_chain(stack, with_lr_injection=False)
        state = optimizer.init({"w": jnp.ones((3,))})
        shifted = set_optimizer_state_step(state, 123)

        count_values = []
        for path, value in jax.tree_util.tree_flatten_with_path(shifted)[0]:
            tail = getattr(path[-1], "name", None) if path else None
            if tail == "count":
                count_values.append(int(value))

        assert count_values
        assert all(v == 123 for v in count_values)

    def test_rejects_negative_step(self):
        optimizer = build_optimizer_chain(
            [PartialFunction(func="optax.sgd", kwargs={"learning_rate": 0.01})],
            with_lr_injection=False,
        )
        state = optimizer.init({"w": jnp.ones((2,))})
        with pytest.raises(ValueError, match="step must be >= 0"):
            set_optimizer_state_step(state, -1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
