#!/usr/bin/env python3

"""
Tests for the training module (biocomp/train.py).
"""

import pytest
import jax
import jax.numpy as jnp
import optax
from biocomp.train import (
    TrainingConfig, 
    make_training_step, 
    l2_loss, 
    sorting_loss,
    create_counter,
    as_schedule
)
from biocomp.parameters import ParameterTree
from biocomp.utils import PartialFunction, PartialFunctionResult


class TestTrainingConfig:
    """Test TrainingConfig functionality."""
    
    def test_default_config(self):
        """Test default training configuration."""
        config = TrainingConfig()
        
        assert config.seed is not None
        assert config.batches_per_step == 128
        assert config.batch_size == 32
        assert config.n_epochs == 3
        assert config.n_batches == 2048
        assert config.n_replicates == 1
        assert config.keep_in_history == ["loss"]
        
    def test_optimizer_creation(self):
        """Test basic optimizer creation."""
        config = TrainingConfig(
            optimizer_stack=[
                PartialFunction(
                    func="optax.sgd",
                    kwargs={"learning_rate": 0.01}
                )
            ]
        )
        
        optimizer = config.optimizer
        assert hasattr(optimizer, 'init')
        assert hasattr(optimizer, 'update')
        
        # Test initialization with dummy params
        params = {"weights": jnp.array([1.0, 2.0])}
        state = optimizer.init(params)
        assert state is not None
        
    def test_learning_rate_injection(self):
        """Test learning rate injection for tracking."""
        config = TrainingConfig(
            optimizer_stack=[
                PartialFunction(
                    func="optax.adamw",
                    kwargs={"learning_rate": 1e-3}
                )
            ],
            keep_in_history=["loss", "learning_rate"]
        )
        
        # Test regular optimizer
        regular_opt = config.optimizer
        params = {"weights": jnp.array([1.0, 2.0])}
        regular_state = regular_opt.init(params)
        
        # Test injected optimizer
        injected_opt = config.create_optimizer_with_lr_injection()
        injected_state = injected_opt.init(params)
        
        # Check that learning rate is accessible in injected version
        lr = optax.tree_utils.tree_get(injected_state, 'learning_rate', default=None)
        assert lr is not None
        assert abs(lr - 1e-3) < 1e-6
        
    def test_complex_optimizer_stack(self):
        """Test learning rate injection with complex optimizer stack like biocomp-jobs."""
        # Recreate a simplified version of the biocomp-jobs optimizer stack
        config = TrainingConfig(
            optimizer_stack=[
                # Gradient clipping
                PartialFunction(
                    func="optax.clip_by_global_norm",
                    kwargs={"max_norm": 1.0}
                ),
                # AdamW with learning rate schedule
                PartialFunction(
                    func="optax.adamw",
                    kwargs={
                        "weight_decay": 0.001,
                        "learning_rate": PartialFunctionResult(
                            func="optax.warmup_cosine_decay_schedule",
                            kwargs={
                                "init_value": 5e-4,
                                "peak_value": 1.5e-3,
                                "warmup_steps": 100,
                                "decay_steps": 1000,
                                "end_value": 2e-5
                            }
                        )
                    }
                )
            ],
            keep_in_history=["loss", "learning_rate"]
        )
        
        # Test that injection works with complex stack
        injected_opt = config.create_optimizer_with_lr_injection()
        params = {"weights": jnp.array([1.0, 2.0])}
        state = injected_opt.init(params)
        
        # Check that learning rate is accessible via hyperparams method
        lr_found = False
        if isinstance(state, tuple):
            for state_comp in state:
                if hasattr(state_comp, 'hyperparams') and 'learning_rate' in state_comp.hyperparams:
                    lr = state_comp.hyperparams['learning_rate'] 
                    assert abs(lr - 5e-4) < 1e-6  # Should start at init_value
                    lr_found = True
                    break
        
        assert lr_found, "Learning rate should be accessible in complex optimizer stack"


class TestLossFunctions:
    """Test loss function implementations."""
    
    def test_as_schedule(self):
        """Test schedule conversion utility."""
        # Test with scalar
        schedule_scalar = as_schedule(0.01)
        assert schedule_scalar(0) == 0.01
        assert schedule_scalar(100) == 0.01
        
        # Test with callable
        def linear_schedule(step):
            return 0.01 * (1 - step / 100)
        
        schedule_callable = as_schedule(linear_schedule)
        assert schedule_callable(0) == 0.01
        assert abs(schedule_callable(50) - 0.005) < 1e-6
        
    def test_create_counter(self):
        """Test counter transformation."""
        counter = create_counter()
        
        params = {"weights": jnp.array([1.0, 2.0])}
        state = counter.init(params)
        
        assert hasattr(state, 'count')
        assert state.count == 0
        
        # Test update
        grads = {"weights": jnp.array([0.1, 0.2])}
        updates, new_state = counter.update(grads, state)
        
        # Updates should be unchanged (it's a counter, not a modifier)
        assert jnp.allclose(updates["weights"], grads["weights"])
        assert new_state.count == 1


class TestTrainingStep:
    """Test training step creation and execution."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.params = ParameterTree()
        self.params["model/weights"] = jnp.ones((3, 2))
        self.params["model/bias"] = jnp.zeros(2)
        
        # Simple loss function for testing
        def simple_loss(dynamic, static, x, y, z, key, step):
            # Handle empty static parameters
            if static.data:
                merged = ParameterTree.merge(static, dynamic)
            else:
                merged = dynamic
                
            # Simple linear model
            output = jnp.dot(x, merged["model/weights"]) + merged["model/bias"]
            loss = jnp.mean((output - y)**2)
            return loss, {"output": output}
        
        self.loss_func = simple_loss
        
        # Simple optimizer
        self.optimizer = optax.chain(
            create_counter(),
            optax.sgd(learning_rate=0.01)
        )
    
    def test_non_scannable_step(self):
        """Test non-scannable training step."""
        training_step = make_training_step(
            self.loss_func,
            self.optimizer,
            fields_to_keep_in_history=["loss"],
            scannable=False
        )
        
        # Prepare inputs
        static, dynamic = self.params.filter_by_tag(["non_grad", "local"])
        opt_state = self.optimizer.init(dynamic)
        
        x = jnp.ones((4, 3))
        y = jnp.ones((4, 2))
        z = jnp.ones((4, 1))
        key = jax.random.PRNGKey(42)
        
        # Run training step
        result = training_step(self.params, opt_state, x, y, z, key)
        
        # Check results
        assert "params" in result
        assert "loss" in result
        assert "grad" in result
        assert "opt" in result
        assert isinstance(result["loss"], jnp.ndarray)
        assert result["loss"].shape == ()  # Scalar loss
        
    def test_scannable_step(self):
        """Test scannable training step."""
        training_step = make_training_step(
            self.loss_func,
            self.optimizer,
            fields_to_keep_in_history=["loss"],
            scannable=True
        )
        
        # Prepare inputs
        static, dynamic = self.params.filter_by_tag(["non_grad", "local"])
        opt_state = self.optimizer.init(dynamic)
        
        # Prepare scan inputs
        batch_size = 4
        n_steps = 3
        
        x_batch = jnp.ones((n_steps, batch_size, 3))
        y_batch = jnp.ones((n_steps, batch_size, 2))
        z_batch = jnp.ones((n_steps, batch_size, 1))
        keys = jax.random.split(jax.random.PRNGKey(42), n_steps)
        step_indices = jnp.arange(n_steps)
        
        # Run scan
        carry = (self.params, opt_state)
        xs = (step_indices, x_batch, y_batch, z_batch, keys)
        
        final_carry, history = jax.lax.scan(training_step, carry, xs)
        
        # Check results
        final_params, final_opt_state = final_carry
        assert isinstance(final_params, ParameterTree)
        assert "loss" in history
        assert history["loss"].shape == (n_steps,)
        
    def test_learning_rate_tracking(self):
        """Test learning rate tracking in training step."""
        # Create optimizer with learning rate injection
        config = TrainingConfig(
            optimizer_stack=[
                PartialFunction(
                    func="optax.adamw",
                    kwargs={"learning_rate": 1e-3}
                )
            ],
            keep_in_history=["loss", "learning_rate"]
        )
        
        injected_optimizer = config.create_optimizer_with_lr_injection()
        
        training_step = make_training_step(
            self.loss_func,
            injected_optimizer,
            fields_to_keep_in_history=["loss", "learning_rate"],
            scannable=False
        )
        
        # Prepare inputs
        static, dynamic = self.params.filter_by_tag(["non_grad", "local"])
        opt_state = injected_optimizer.init(dynamic)
        
        x = jnp.ones((4, 3))
        y = jnp.ones((4, 2))
        z = jnp.ones((4, 1))
        key = jax.random.PRNGKey(42)
        
        # Run training step
        result = training_step(self.params, opt_state, x, y, z, key)
        
        # Check that learning rate was captured
        assert "learning_rate" in result
        lr = result["learning_rate"]
        assert abs(lr - 1e-3) < 1e-6


class TestLearningRateExtraction:
    """Test learning rate extraction methods."""
    
    def test_fixed_learning_rate(self):
        """Test extraction of fixed learning rate."""
        config = TrainingConfig(
            optimizer_stack=[
                PartialFunction(
                    func="optax.adamw",
                    kwargs={"learning_rate": 1e-3}
                )
            ]
        )
        
        optimizer = config.create_optimizer_with_lr_injection()
        params = {"weights": jnp.array([1.0, 2.0])}
        state = optimizer.init(params)
        
        # Test tree_get method
        lr = optax.tree_utils.tree_get(state, 'learning_rate', default=None)
        assert lr is not None
        assert abs(lr - 1e-3) < 1e-6
        
    def test_scheduled_learning_rate(self):
        """Test extraction of scheduled learning rate."""
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=1e-7,
            peak_value=1e-3,
            warmup_steps=5,
            decay_steps=20,
            end_value=1e-5
        )
        
        # Test direct optax injection
        wrapped_adamw = optax.inject_hyperparams(optax.adamw)
        optimizer = wrapped_adamw(learning_rate=schedule)
        
        params = {"weights": jnp.array([1.0, 2.0])}
        state = optimizer.init(params)
        
        # Check hyperparams access
        assert hasattr(state, 'hyperparams')
        assert 'learning_rate' in state.hyperparams
        
        # Initial learning rate should be close to init_value
        lr = state.hyperparams['learning_rate']
        assert abs(lr - 1e-7) < 1e-8


class TestIntegration:
    """Integration tests for training functionality."""
    
    def test_mini_training_loop(self):
        """Test a minimal training loop."""
        # Set up
        config = TrainingConfig(
            optimizer_stack=[
                PartialFunction(
                    func="optax.adamw",
                    kwargs={"learning_rate": 1e-3}
                )
            ],
            keep_in_history=["loss", "learning_rate"],
            n_replicates=1,
            batches_per_step=2,
            batch_size=4,
        )
        
        # Create model parameters
        params = ParameterTree()
        params["model/weights"] = jax.random.normal(jax.random.PRNGKey(0), (4, 3))
        params["model/bias"] = jnp.zeros(3)
        
        # Loss function
        def loss_func(dynamic, static, x, y, z, key, step):
            if static.data:
                merged = ParameterTree.merge(static, dynamic)
            else:
                merged = dynamic
                
            output = jnp.dot(x, merged["model/weights"]) + merged["model/bias"]
            loss = jnp.mean((output - y)**2)
            return loss, {"output": output}
        
        # Set up training
        static, dynamic = params.filter_by_tag(["non_grad", "local"])
        optimizer = config.create_optimizer_with_lr_injection()
        opt_state = optimizer.init(dynamic)
        
        training_step = make_training_step(
            loss_func,
            optimizer,
            fields_to_keep_in_history=config.keep_in_history,
            scannable=False
        )
        
        # Run training steps
        key = jax.random.PRNGKey(42)
        losses = []
        learning_rates = []
        
        for step in range(5):
            # Generate data
            x = jax.random.normal(key, (4, 4))
            y = jax.random.normal(key, (4, 3))
            z = jax.random.uniform(key, (4, 2))
            step_key = jax.random.fold_in(key, step)
            
            # Training step
            result = training_step(params, opt_state, x, y, z, step_key)
            
            # Update
            params = result["params"]
            opt_state = result["opt"]
            
            # Track metrics
            losses.append(float(result["loss"]))
            if "learning_rate" in result:
                learning_rates.append(float(result["learning_rate"]))
        
        # Verify training occurred
        assert len(losses) == 5
        assert len(learning_rates) == 5
        
        # Learning rates should be consistent (fixed LR)
        assert all(abs(lr - 1e-3) < 1e-6 for lr in learning_rates)
        
        # Parameters should have changed
        original_weights = jax.random.normal(jax.random.PRNGKey(0), (4, 3))
        final_weights = params["model/weights"]
        assert not jnp.allclose(original_weights, final_weights)
        
    def test_parameter_filtering(self):
        """Test parameter filtering for static vs dynamic."""
        params = ParameterTree()
        params["static_param"] = jnp.array([1.0, 2.0])
        params["dynamic_param"] = jnp.array([3.0, 4.0])
        
        # Tag some parameters as non-gradable
        params.tag(["static_param"], "non_grad")
        
        # The training code uses filter_by_tag(["non_grad", "local"]) which returns
        # (matching_params, non_matching_params)
        # So static contains params with "non_grad" OR "local" tags
        # And dynamic contains params without those tags
        static, dynamic = params.filter_by_tag(["non_grad"])
        
        # static should contain non_grad parameters
        assert "static_param" in static.data
        assert "static_param" not in dynamic.data
        
        # dynamic should contain parameters without non_grad tag
        assert "dynamic_param" in dynamic.data
        assert "dynamic_param" not in static.data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
