#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset

"""
Tests for the training module (biocomp/train.py).
"""

import pytest
import jax
import jax.numpy as jnp
import optax
from types import SimpleNamespace

from biocomp.train import TrainingConfig, sorting_loss, energy_sampling_loss, _quantization_kl_loss
from biocomp.optimutils import make_training_step, create_counter, as_schedule
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
        regular_opt.init(params)

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

    class _FakeStack:
        def __init__(self):
            self.networks = [SimpleNamespace(nb_inputs=2, nb_outputs=2)]
            self.total_nb_of_outputs = 2

        def apply(self, _params, x, z, _key):
            yhat = jnp.array(
                [
                    0.7 * x[0] + 0.3 * z[0],
                    0.2 * x[0] + 0.8 * x[1] - 0.1 * z[0],
                ]
            )
            apply_aux = {"grads_wrt_inputs": jnp.array([0.2, 0.1])}
            full_output = jnp.concatenate([x, yhat])
            return yhat, (apply_aux, full_output)

    class _ToyInverseSpec:
        def __init__(self, node_id: int, output_slot: int, output_len: int = 1):
            self.node_id = node_id
            self.output_slot = output_slot
            self.output_len = output_len

    class _ToyGraphNode:
        def __init__(self, node_type: str, is_inverse_of=None):
            self.node_type = node_type
            self.is_inverse_of = is_inverse_of

    class _ToyEdge:
        def __init__(self, content_embedding_names: dict[str, tuple[str, ...]] | None = None):
            self.content_embedding_names = content_embedding_names or {}

    class _ToyStackNode:
        def __init__(self, network_id: int, node_id: int, layer_number: int, node_position_in_layer: int, incoming_edges=None):
            self.network_id = network_id
            self.node_id = node_id
            self.layer_number = layer_number
            self.node_position_in_layer = node_position_in_layer
            self._incoming_edges = incoming_edges or []

        def get(self, stack):
            return stack._nodes[self.node_id]

        def get_forward_stacknode(self, stack):
            if self.node_id == 1:
                return stack._fwd_stacknode
            return None

        def get_incoming_edges(self, stack):
            return self._incoming_edges

    class _PairStack:
        """Minimal stack exposing one transcription/inv_transcription pair."""

        def __init__(self, inv_scale: float):
            self.networks = [SimpleNamespace(nb_inputs=1, nb_outputs=2)]
            self.total_nb_of_outputs = 2
            self.layers = [
                SimpleNamespace(f_apply=self._fwd_apply, f_input_shapes=[(1,)]),
                SimpleNamespace(f_apply=self._inv_apply, f_input_shapes=[(1,)]),
            ]
            self._inv_scale = inv_scale
            self._nodes = {
                0: TestLossFunctions._ToyGraphNode("transcription"),
                1: TestLossFunctions._ToyGraphNode(
                    "inv_transcription",
                    TestLossFunctions._ToyInverseSpec(node_id=0, output_slot=0),
                ),
            }
            tc_edge = TestLossFunctions._ToyEdge(
                content_embedding_names={"tc_rate": ("00_empty_tc",)}
            )
            self._fwd_stacknode = TestLossFunctions._ToyStackNode(
                network_id=0, node_id=0, layer_number=0, node_position_in_layer=0,
                incoming_edges=[tc_edge],
            )
            self._inv_stacknode = TestLossFunctions._ToyStackNode(
                network_id=0, node_id=1, layer_number=1, node_position_in_layer=0
            )

        def each_node(self):
            yield self._inv_stacknode

        def get_layer_namespace(self, layer_id: int) -> str:
            if layer_id == 0:
                return "local/0/fwd_transcription"
            return "local/1/inv_transcription"

        def _fwd_apply(
            self,
            value,
            *,
            random_vars,
            params,
            node_id,
            key,
            network_id=None,
            rate_override=None,
            **_,
        ):
            del random_vars, key, network_id
            rate = (
                rate_override
                if rate_override is not None
                else params["local/0/fwd_transcription/transcription_rate"][node_id]
            )
            slope = jnp.ravel(rate)[0]
            return jnp.array([value[0] * slope]), {}

        def _inv_apply(
            self,
            value,
            *,
            random_vars,
            params,
            node_id,
            key,
            network_id=None,
            rate_override=None,
            **_,
        ):
            del random_vars, key, node_id, network_id
            rate = (
                rate_override
                if rate_override is not None
                else params["local/0/fwd_transcription/transcription_rate"][0]
            )
            slope = jnp.ravel(rate)[0]
            recon = self._inv_scale * value[0] / (slope + 1e-6)
            return jnp.array([recon]), {}

        def apply(self, _params, x, z, _key):
            del z
            yhat = jnp.array([x[0], 0.5 * x[0]])
            apply_aux = {"grads_wrt_inputs": jnp.array([0.2])}
            full_output = yhat
            return yhat, (apply_aux, full_output)

    @staticmethod
    def _make_sorting_loss_params() -> ParameterTree:
        params = ParameterTree()
        params.at(
            "global/dependent_output_mask",
            jnp.array([True, False]),
            tags=["non_grad", "local"],
            overwrite=True,
        )
        params.at(
            "shared/quantization/values/tl",
            jnp.array([[0.1], [0.2]]),
            tags=["shared"],
            overwrite=True,
        )
        params.at(
            "shared/quantization/logstdevs/tl",
            jnp.array([[-3.0], [-3.0]]),
            tags=["shared"],
            overwrite=True,
        )
        params.at(
            "shared/quantization/counts/tl",
            jnp.array([[1.0], [1.0]]),
            tags=["shared"],
            overwrite=True,
        )
        return params

    @staticmethod
    def _make_inverse_pair_params() -> ParameterTree:
        params = ParameterTree()
        params.at(
            "global/dependent_output_mask",
            jnp.array([True, False]),
            tags=["non_grad", "local"],
            overwrite=True,
        )
        params.at(
            "shared/quantization/values/tl",
            jnp.array([[0.1], [0.2]]),
            tags=["shared"],
            overwrite=True,
        )
        params.at(
            "shared/quantization/logstdevs/tl",
            jnp.array([[-3.0], [-3.0]]),
            tags=["shared"],
            overwrite=True,
        )
        params.at(
            "shared/quantization/counts/tl",
            jnp.array([[1.0], [1.0]]),
            tags=["shared"],
            overwrite=True,
        )
        params.at(
            "local/0/fwd_transcription/tc_rate",
            jnp.array([[[0.7]]]),
            tags=["local"],
            overwrite=True,
        )
        return params

    def test_sorting_loss_pinball_zero_is_noop(self):
        stack = self._FakeStack()
        params = self._make_sorting_loss_params()
        x = jnp.array([[0.1, 0.2], [0.7, 0.3], [0.3, 0.9], [0.8, 0.6]])
        y = jnp.array([[0.3, 0.1], [0.4, 0.9], [0.6, 0.2], [0.2, 0.7]])
        z = jnp.array([[0.1], [0.8], [0.3], [0.5]])
        key = jax.random.PRNGKey(0)

        loss_fn = sorting_loss(
            stack,
            training_config=None,
            sorting_mse_weight=0.4,
            pinball_weight=0.0,
            kl_weight=2e-4,
            negative_grad_penalty=0.1,
        )
        loss, aux = loss_fn(params, ParameterTree(), x, y, z, key, 0)

        expected = (
            aux["sublosses"]["main_loss"] + aux["sublosses"]["kl_loss"] + aux["debug"]["ng_loss"]
        )
        assert jnp.allclose(loss, expected)
        assert "pinball_loss" in aux["sublosses"]
        assert float(aux["debug"]["pinball_weight"]) == 0.0

    def test_sorting_loss_pinball_changes_main_loss_when_enabled(self):
        stack = self._FakeStack()
        params = self._make_sorting_loss_params()
        x = jnp.array([[0.1, 0.2], [0.7, 0.3], [0.3, 0.9], [0.8, 0.6]])
        y = jnp.array([[0.9, 0.1], [0.2, 0.9], [0.6, 0.8], [0.1, 0.3]])
        z = jnp.array([[0.1], [0.8], [0.3], [0.5]])
        key = jax.random.PRNGKey(1)

        loss_no_pb, aux_no_pb = sorting_loss(
            stack,
            training_config=None,
            sorting_mse_weight=0.5,
            pinball_weight=0.0,
            kl_weight=0.0,
            negative_grad_penalty=0.0,
        )(params, ParameterTree(), x, y, z, key, 0)
        loss_full_pb, aux_full_pb = sorting_loss(
            stack,
            training_config=None,
            sorting_mse_weight=0.5,
            pinball_weight=1.0,
            kl_weight=0.0,
            negative_grad_penalty=0.0,
        )(params, ParameterTree(), x, y, z, key, 0)

        assert not jnp.allclose(aux_no_pb["sublosses"]["main_loss"], aux_full_pb["sublosses"]["main_loss"])
        assert not jnp.allclose(loss_no_pb, loss_full_pb)

    def test_sorting_loss_z_sorting_term_changes_main_loss_when_enabled(self):
        stack = self._FakeStack()
        params = self._make_sorting_loss_params()
        x = jnp.array([[0.1, 0.2], [0.7, 0.3], [0.3, 0.9], [0.8, 0.6]])
        y = jnp.array([[0.9, 0.1], [0.2, 0.9], [0.6, 0.8], [0.1, 0.3]])
        z = jnp.array([[0.8], [0.1], [0.6], [0.2]])
        key = jax.random.PRNGKey(2)

        loss_no_zsort, aux_no_zsort = sorting_loss(
            stack,
            training_config=None,
            sorting_mse_weight=0.0,
            z_sorting_mse_weight=0.0,
            pinball_weight=0.0,
            kl_weight=0.0,
            negative_grad_penalty=0.0,
        )(params, ParameterTree(), x, y, z, key, 0)
        loss_full_zsort, aux_full_zsort = sorting_loss(
            stack,
            training_config=None,
            sorting_mse_weight=0.0,
            z_sorting_mse_weight=1.0,
            pinball_weight=0.0,
            kl_weight=0.0,
            negative_grad_penalty=0.0,
        )(params, ParameterTree(), x, y, z, key, 0)

        assert not jnp.allclose(
            aux_no_zsort["sublosses"]["main_loss"], aux_full_zsort["sublosses"]["main_loss"]
        )
        assert not jnp.allclose(loss_no_zsort, loss_full_zsort)

    def test_sorting_loss_pinball_z_order_changes_main_loss(self):
        stack = self._FakeStack()
        params = self._make_sorting_loss_params()
        x = jnp.array([[0.1, 0.2], [0.7, 0.3], [0.3, 0.9], [0.8, 0.6]])
        y = jnp.array([[0.9, 0.1], [0.2, 0.9], [0.6, 0.8], [0.1, 0.3]])
        z = jnp.array([[0.8], [0.1], [0.6], [0.2]])
        key = jax.random.PRNGKey(3)

        loss_sorted_pb, aux_sorted_pb = sorting_loss(
            stack,
            training_config=None,
            sorting_mse_weight=0.4,
            pinball_weight=1.0,
            pinball_use_z_order=False,
            kl_weight=0.0,
            negative_grad_penalty=0.0,
        )(params, ParameterTree(), x, y, z, key, 0)
        loss_z_pb, aux_z_pb = sorting_loss(
            stack,
            training_config=None,
            sorting_mse_weight=0.4,
            pinball_weight=1.0,
            pinball_use_z_order=True,
            kl_weight=0.0,
            negative_grad_penalty=0.0,
        )(params, ParameterTree(), x, y, z, key, 0)

        assert not jnp.allclose(
            aux_sorted_pb["sublosses"]["main_loss"], aux_z_pb["sublosses"]["main_loss"]
        )
        assert not jnp.allclose(loss_sorted_pb, loss_z_pb)

    def test_sorting_loss_pinball_z_tau_changes_main_loss(self):
        stack = self._FakeStack()
        params = self._make_sorting_loss_params()
        x = jnp.array([[0.1, 0.2], [0.7, 0.3], [0.3, 0.9], [0.8, 0.6]])
        y = jnp.array([[0.9, 0.1], [0.2, 0.9], [0.6, 0.8], [0.1, 0.3]])
        z = jnp.array([[0.8], [0.1], [0.6], [0.2]])
        key = jax.random.PRNGKey(6)

        loss_rank_pb, aux_rank_pb = sorting_loss(
            stack,
            training_config=None,
            sorting_mse_weight=0.4,
            pinball_weight=1.0,
            pinball_use_z_tau=False,
            kl_weight=0.0,
            negative_grad_penalty=0.0,
        )(params, ParameterTree(), x, y, z, key, 0)
        loss_ztau_pb, aux_ztau_pb = sorting_loss(
            stack,
            training_config=None,
            sorting_mse_weight=0.4,
            pinball_weight=1.0,
            pinball_use_z_tau=True,
            kl_weight=0.0,
            negative_grad_penalty=0.0,
        )(params, ParameterTree(), x, y, z, key, 0)

        assert not jnp.allclose(
            aux_rank_pb["sublosses"]["main_loss"], aux_ztau_pb["sublosses"]["main_loss"]
        )
        assert not jnp.allclose(loss_rank_pb, loss_ztau_pb)

    def test_sorting_loss_cdf_calibration_changes_main_loss_when_enabled(self):
        stack = self._FakeStack()
        params = self._make_sorting_loss_params()
        x = jnp.array([[0.1, 0.2], [0.7, 0.3], [0.3, 0.9], [0.8, 0.6]])
        y = jnp.array([[0.9, 0.1], [0.2, 0.9], [0.6, 0.8], [0.1, 0.3]])
        z = jnp.array([[0.8], [0.1], [0.6], [0.2]])
        key = jax.random.PRNGKey(7)

        loss_no_cdf, aux_no_cdf = sorting_loss(
            stack,
            training_config=None,
            sorting_mse_weight=0.2,
            pinball_weight=0.2,
            cdf_calibration_weight=0.0,
            kl_weight=0.0,
            negative_grad_penalty=0.0,
        )(params, ParameterTree(), x, y, z, key, 0)
        loss_with_cdf, aux_with_cdf = sorting_loss(
            stack,
            training_config=None,
            sorting_mse_weight=0.2,
            pinball_weight=0.2,
            cdf_calibration_weight=1.0,
            cdf_calibration_temperature=0.2,
            kl_weight=0.0,
            negative_grad_penalty=0.0,
        )(params, ParameterTree(), x, y, z, key, 0)

        assert not jnp.allclose(
            aux_no_cdf["sublosses"]["main_loss"], aux_with_cdf["sublosses"]["main_loss"]
        )
        assert not jnp.allclose(loss_no_cdf, loss_with_cdf)

    def test_sorting_loss_masked_entries_do_not_create_nan(self):
        stack = self._FakeStack()
        params = self._make_sorting_loss_params()
        x = jnp.array([[0.1, 0.2], [0.7, 0.3], [0.3, 0.9], [0.8, 0.6]])
        y = jnp.array([[0.9, 0.1], [0.2, 0.9], [0.6, 0.8], [0.1, 0.3]])
        z = jnp.array([[0.8], [0.1], [0.6], [0.2]])
        key = jax.random.PRNGKey(4)

        loss, aux = sorting_loss(
            stack,
            training_config=None,
            sorting_mse_weight=0.0,
            z_sorting_mse_weight=0.0,
            pinball_weight=0.0,
            percent_batch_used=0.5,
            kl_weight=0.0,
            negative_grad_penalty=0.0,
        )(params, ParameterTree(), x, y, z, key, 0)

        assert jnp.isfinite(loss)
        assert jnp.isfinite(aux["sublosses"]["main_loss"])
        assert jnp.isfinite(aux["sublosses"]["z_sorting_mse"])

    def test_sorting_loss_inverse_consistency_penalizes_bad_inverse_pair(self):
        params = self._make_inverse_pair_params()
        x = jnp.array([[0.2], [0.4], [0.6], [0.8]])
        y = jnp.concatenate([x, 0.5 * x], axis=1)
        z = jnp.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [0.3, 0.2, 0.1],
            ]
        )
        key = jax.random.PRNGKey(123)

        stack_good = self._PairStack(inv_scale=1.0)
        stack_bad = self._PairStack(inv_scale=0.65)

        loss_good, aux_good = sorting_loss(
            stack_good,
            training_config=None,
            sorting_mse_weight=0.0,
            pinball_weight=0.0,
            cdf_calibration_weight=0.0,
            kl_weight=0.0,
            negative_grad_penalty=0.0,
            inverse_consistency_weight=1.0,
            inverse_consistency_batch_size=32,
        )(params, ParameterTree(), x, y, z, key, 0)
        loss_bad, aux_bad = sorting_loss(
            stack_bad,
            training_config=None,
            sorting_mse_weight=0.0,
            pinball_weight=0.0,
            cdf_calibration_weight=0.0,
            kl_weight=0.0,
            negative_grad_penalty=0.0,
            inverse_consistency_weight=1.0,
            inverse_consistency_batch_size=32,
        )(params, ParameterTree(), x, y, z, key, 0)

        assert "inverse_consistency_loss" in aux_good["sublosses"]
        assert float(aux_good["debug"]["inverse_consistency_n_groups"]) == 1.0
        assert float(aux_good["debug"]["inverse_consistency_n_pairs"]) == 1.0
        assert aux_bad["sublosses"]["inverse_consistency_loss"] > aux_good["sublosses"][
            "inverse_consistency_loss"
        ]
        assert loss_bad > loss_good

    def test_energy_sampling_loss_is_finite(self):
        stack = self._FakeStack()
        params = self._make_sorting_loss_params()
        x = jnp.array([[0.1, 0.2], [0.7, 0.3], [0.3, 0.9], [0.8, 0.6]])
        y = jnp.array([[0.9, 0.1], [0.2, 0.9], [0.6, 0.8], [0.1, 0.3]])
        z = jnp.array([[0.8], [0.1], [0.6], [0.2]])
        key = jax.random.PRNGKey(8)

        loss, aux = energy_sampling_loss(
            stack,
            training_config=None,
            energy_n_samples=4,
            energy_outputs_independent=True,
            energy_weight=1.0,
            kl_weight=0.0,
            negative_grad_penalty=0.0,
        )(params, ParameterTree(), x, y, z, key, 0)

        assert jnp.isfinite(loss)
        assert jnp.isfinite(aux["sublosses"]["energy_loss"])
        assert "energy_n_samples" in aux["debug"]

    def test_energy_sampling_loss_inverse_consistency_penalizes_bad_inverse_pair(self):
        params = self._make_inverse_pair_params()
        x = jnp.array([[0.2], [0.4], [0.6], [0.8]])
        y = jnp.concatenate([x, 0.5 * x], axis=1)
        z = jnp.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [0.3, 0.2, 0.1],
            ]
        )
        key = jax.random.PRNGKey(321)

        stack_good = self._PairStack(inv_scale=1.0)
        stack_bad = self._PairStack(inv_scale=0.65)

        loss_good, aux_good = energy_sampling_loss(
            stack_good,
            training_config=None,
            energy_n_samples=4,
            energy_weight=0.0,
            coverage_calibration_weight=0.0,
            tail_pinball_weight=0.0,
            kl_weight=0.0,
            negative_grad_penalty=0.0,
            inverse_consistency_weight=1.0,
            inverse_consistency_batch_size=32,
        )(params, ParameterTree(), x, y, z, key, 0)
        loss_bad, aux_bad = energy_sampling_loss(
            stack_bad,
            training_config=None,
            energy_n_samples=4,
            energy_weight=0.0,
            coverage_calibration_weight=0.0,
            tail_pinball_weight=0.0,
            kl_weight=0.0,
            negative_grad_penalty=0.0,
            inverse_consistency_weight=1.0,
            inverse_consistency_batch_size=32,
        )(params, ParameterTree(), x, y, z, key, 0)

        assert "inverse_consistency_loss" in aux_good["sublosses"]
        assert float(aux_good["debug"]["inverse_consistency_n_groups"]) == 1.0
        assert float(aux_good["debug"]["inverse_consistency_n_pairs"]) == 1.0
        assert aux_bad["sublosses"]["inverse_consistency_loss"] > aux_good["sublosses"][
            "inverse_consistency_loss"
        ]
        assert loss_bad > loss_good

    def test_energy_sampling_loss_joint_differs_from_independent(self):
        stack = self._FakeStack()
        params = self._make_sorting_loss_params()
        params.at(
            "global/dependent_output_mask",
            jnp.array([True, True]),
            tags=["non_grad", "local"],
            overwrite=True,
        )
        x = jnp.array([[0.1, 0.2], [0.7, 0.3], [0.3, 0.9], [0.8, 0.6]])
        y = jnp.array([[0.9, 0.1], [0.2, 0.9], [0.6, 0.8], [0.1, 0.3]])
        z = jnp.array([[0.8], [0.1], [0.6], [0.2]])
        key = jax.random.PRNGKey(9)

        loss_indep, aux_indep = energy_sampling_loss(
            stack,
            training_config=None,
            energy_n_samples=6,
            energy_outputs_independent=True,
            energy_weight=1.0,
            kl_weight=0.0,
            negative_grad_penalty=0.0,
        )(params, ParameterTree(), x, y, z, key, 0)
        loss_joint, aux_joint = energy_sampling_loss(
            stack,
            training_config=None,
            energy_n_samples=6,
            energy_outputs_independent=False,
            energy_weight=1.0,
            kl_weight=0.0,
            negative_grad_penalty=0.0,
        )(params, ParameterTree(), x, y, z, key, 0)

        assert not jnp.allclose(
            aux_indep["sublosses"]["energy_loss"], aux_joint["sublosses"]["energy_loss"]
        )
        assert not jnp.allclose(loss_indep, loss_joint)

    def test_energy_sampling_loss_pairwise_weight_changes_energy(self):
        stack = self._FakeStack()
        params = self._make_sorting_loss_params()
        params.at(
            "global/dependent_output_mask",
            jnp.array([True, True]),
            tags=["non_grad", "local"],
            overwrite=True,
        )
        x = jnp.array([[0.1, 0.2], [0.7, 0.3], [0.3, 0.9], [0.8, 0.6]])
        y = jnp.array([[0.9, 0.1], [0.2, 0.9], [0.6, 0.8], [0.1, 0.3]])
        z = jnp.array([[0.8], [0.1], [0.6], [0.2]])
        key = jax.random.PRNGKey(10)

        loss_low, aux_low = energy_sampling_loss(
            stack,
            training_config=None,
            energy_n_samples=6,
            energy_outputs_independent=False,
            energy_pairwise_weight=0.2,
            energy_weight=1.0,
            kl_weight=0.0,
            negative_grad_penalty=0.0,
        )(params, ParameterTree(), x, y, z, key, 0)
        loss_high, aux_high = energy_sampling_loss(
            stack,
            training_config=None,
            energy_n_samples=6,
            energy_outputs_independent=False,
            energy_pairwise_weight=0.8,
            energy_weight=1.0,
            kl_weight=0.0,
            negative_grad_penalty=0.0,
        )(params, ParameterTree(), x, y, z, key, 0)

        assert not jnp.allclose(
            aux_low["sublosses"]["energy_loss"], aux_high["sublosses"]["energy_loss"]
        )
        assert not jnp.allclose(loss_low, loss_high)

    def test_energy_sampling_loss_coverage_calibration_changes_main_loss(self):
        stack = self._FakeStack()
        params = self._make_sorting_loss_params()
        x = jnp.array([[0.1, 0.2], [0.7, 0.3], [0.3, 0.9], [0.8, 0.6]])
        y = jnp.array([[0.9, 0.1], [0.2, 0.9], [0.6, 0.8], [0.1, 0.3]])
        z = jnp.array([[0.8], [0.1], [0.6], [0.2]])
        key = jax.random.PRNGKey(11)

        loss_no_cov, aux_no_cov = energy_sampling_loss(
            stack,
            training_config=None,
            energy_n_samples=6,
            energy_outputs_independent=True,
            coverage_calibration_weight=0.0,
            energy_weight=1.0,
            kl_weight=0.0,
            negative_grad_penalty=0.0,
        )(params, ParameterTree(), x, y, z, key, 0)
        loss_cov, aux_cov = energy_sampling_loss(
            stack,
            training_config=None,
            energy_n_samples=6,
            energy_outputs_independent=True,
            coverage_calibration_weight=0.4,
            coverage_interval_low=0.1,
            coverage_interval_high=0.9,
            coverage_temperature=0.2,
            energy_weight=1.0,
            kl_weight=0.0,
            negative_grad_penalty=0.0,
        )(params, ParameterTree(), x, y, z, key, 0)

        assert not jnp.allclose(aux_no_cov["sublosses"]["main_loss"], aux_cov["sublosses"]["main_loss"])
        assert not jnp.allclose(aux_no_cov["sublosses"]["coverage_loss"], aux_cov["sublosses"]["coverage_loss"])
        assert not jnp.allclose(loss_no_cov, loss_cov)

    def test_energy_sampling_loss_tail_pinball_changes_main_loss(self):
        stack = self._FakeStack()
        params = self._make_sorting_loss_params()
        x = jnp.array([[0.1, 0.2], [0.7, 0.3], [0.3, 0.9], [0.8, 0.6]])
        y = jnp.array([[0.9, 0.1], [0.2, 0.9], [0.6, 0.8], [0.1, 0.3]])
        z = jnp.array([[0.8], [0.1], [0.6], [0.2]])
        key = jax.random.PRNGKey(12)

        loss_no_tail, aux_no_tail = energy_sampling_loss(
            stack,
            training_config=None,
            energy_n_samples=6,
            energy_outputs_independent=True,
            tail_pinball_weight=0.0,
            energy_weight=1.0,
            kl_weight=0.0,
            negative_grad_penalty=0.0,
        )(params, ParameterTree(), x, y, z, key, 0)
        loss_tail, aux_tail = energy_sampling_loss(
            stack,
            training_config=None,
            energy_n_samples=6,
            energy_outputs_independent=True,
            tail_pinball_weight=0.05,
            tail_tau_low=0.03,
            tail_tau_high=0.97,
            energy_weight=1.0,
            kl_weight=0.0,
            negative_grad_penalty=0.0,
        )(params, ParameterTree(), x, y, z, key, 0)

        assert "tail_pinball_loss" in aux_tail["sublosses"]
        assert float(aux_tail["debug"]["tail_pinball_weight"]) == pytest.approx(0.05)
        assert not jnp.allclose(aux_no_tail["sublosses"]["main_loss"], aux_tail["sublosses"]["main_loss"])
        assert not jnp.allclose(loss_no_tail, loss_tail)


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


class TestKLNormalization:
    """Test KL loss normalization for multi-dimensional embeddings."""

    @staticmethod
    def _make_kl_params(rate_dim: int, values: list[list[float]], logstds: list[list[float]], counts: list[float]) -> ParameterTree:
        params = ParameterTree()
        params.at(
            "shared/quantization/values/tc",
            jnp.array(values),
            tags=["shared"],
            overwrite=True,
        )
        params.at(
            "shared/quantization/logstdevs/tc",
            jnp.array(logstds),
            tags=["shared"],
            overwrite=True,
        )
        params.at(
            "shared/quantization/counts/tc",
            jnp.array(counts).reshape(-1, 1),
            tags=["shared"],
            overwrite=True,
        )
        return params

    def test_kl_loss_covers_all_embedding_dimensions(self):
        """Verify per-dimension KL behavior for multi-dim embeddings.

        With the normalization fix, KL is a weighted average over *embeddings*
        of the full multi-dim KL (sum over dims).  So rate_dim=2 with identical
        per-dim params should give exactly 2x the KL of rate_dim=1.

        Note: stable_sigma means the formula isn't a pure Gaussian KL, so we
        test relative behavior (zeroing a dim reduces KL) rather than KL=0.
        """
        logstd = -3.0  # matches existing test params; KL is clearly positive here

        # rate_dim=2: two embeddings with non-zero mu in both dims
        params_2d = self._make_kl_params(
            rate_dim=2,
            values=[[0.5, 0.8], [0.3, 0.6]],
            logstds=[[logstd, logstd], [logstd, logstd]],
            counts=[2.0, 3.0],
        )
        kl_2d, klw, _, _, _, _ = _quantization_kl_loss(params_2d, kl_weight=1.0, step=0)
        assert float(kl_2d) > 0, "KL should be positive for non-zero means"

        # Zero out dim 0 mu -> reduces its contribution, dim 1 still active
        params_dim1_only = self._make_kl_params(
            rate_dim=2,
            values=[[0.0, 0.8], [0.0, 0.6]],
            logstds=[[logstd, logstd], [logstd, logstd]],
            counts=[2.0, 3.0],
        )
        kl_dim1, _, _, _, _, _ = _quantization_kl_loss(params_dim1_only, kl_weight=1.0, step=0)
        assert float(kl_dim1) > 0, "KL should be positive (dim 1 still active)"
        assert float(kl_dim1) < float(kl_2d), "Zeroing dim 0 mu should decrease KL"

        # Zero out both dims mu -> KL still positive (logstd term), but smaller
        params_zero_mu = self._make_kl_params(
            rate_dim=2,
            values=[[0.0, 0.0], [0.0, 0.0]],
            logstds=[[logstd, logstd], [logstd, logstd]],
            counts=[2.0, 3.0],
        )
        kl_zero_mu, _, _, _, _, _ = _quantization_kl_loss(params_zero_mu, kl_weight=1.0, step=0)
        assert float(kl_zero_mu) < float(kl_dim1), "Zeroing both mu dims should further decrease KL"

    def test_kl_rate_dim2_is_2x_rate_dim1(self):
        """rate_dim=2 with identical per-dim params gives 2x the KL of rate_dim=1."""
        mu_val, logstd_val = 0.5, 0.2
        counts = [2.0, 3.0]

        # rate_dim=1
        params_1d = self._make_kl_params(
            rate_dim=1,
            values=[[mu_val], [mu_val]],
            logstds=[[logstd_val], [logstd_val]],
            counts=counts,
        )
        kl_1d, _, _, _, _, _ = _quantization_kl_loss(params_1d, kl_weight=1.0, step=0)

        # rate_dim=2 with same value in both dims
        params_2d = self._make_kl_params(
            rate_dim=2,
            values=[[mu_val, mu_val], [mu_val, mu_val]],
            logstds=[[logstd_val, logstd_val], [logstd_val, logstd_val]],
            counts=counts,
        )
        kl_2d, _, _, _, _, _ = _quantization_kl_loss(params_2d, kl_weight=1.0, step=0)

        assert float(kl_2d) == pytest.approx(2.0 * float(kl_1d), rel=1e-5), (
            f"rate_dim=2 KL ({float(kl_2d):.6f}) should be 2x rate_dim=1 KL ({float(kl_1d):.6f})"
        )

    def test_kl_gradient_reaches_every_dimension(self):
        """Both dimensions of a rate_dim=2 embedding receive independent gradient pressure.

        This is the definitive test: differentiate KL w.r.t. the values array and
        verify every element (embedding x dim) has a non-zero gradient.  A bug that
        routes all pressure to dim 0 would leave dim 1 gradients at zero.
        """
        values = jnp.array([[0.5, 0.8], [0.3, 0.6]])
        logstds = jnp.array([[-3.0, -3.0], [-3.0, -3.0]])
        counts = jnp.array([[2.0], [3.0]])

        def kl_of_values(vals):
            params = ParameterTree()
            params.at("shared/quantization/values/tc", vals, tags=["shared"], overwrite=True)
            params.at("shared/quantization/logstdevs/tc", logstds, tags=["shared"], overwrite=True)
            params.at("shared/quantization/counts/tc", counts, tags=["shared"], overwrite=True)
            kl, *_ = _quantization_kl_loss(params, kl_weight=1.0, step=0)
            return kl

        grad_vals = jax.grad(kl_of_values)(values)

        # Every element must have non-zero gradient
        assert grad_vals.shape == (2, 2), f"Expected (2,2) grad shape, got {grad_vals.shape}"
        for emb_idx in range(2):
            for dim_idx in range(2):
                g = float(grad_vals[emb_idx, dim_idx])
                assert abs(g) > 1e-6, (
                    f"Gradient for embedding {emb_idx}, dim {dim_idx} is ~0 ({g:.2e}); "
                    f"dim {dim_idx} is not receiving KL pressure"
                )

    def test_kl_gradient_per_dim_matches_1d_gradient(self):
        """Each dimension's gradient in rate_dim=2 matches what rate_dim=1 would give.

        Constructs rate_dim=2 with *different* values per dim, then checks that the
        gradient for each dim column equals the gradient from an equivalent rate_dim=1
        problem using that column's values.  This rules out any cross-dim coupling or
        single-dim doubling.
        """
        mu_d0 = [0.5, 0.3]
        mu_d1 = [0.8, 0.6]
        logstd_val = -3.0
        count_vals = [2.0, 3.0]

        # rate_dim=2 gradients
        values_2d = jnp.array([[mu_d0[0], mu_d1[0]], [mu_d0[1], mu_d1[1]]])
        logstds_2d = jnp.full((2, 2), logstd_val)
        counts = jnp.array(count_vals).reshape(-1, 1)

        def kl_2d(vals):
            p = ParameterTree()
            p.at("shared/quantization/values/tc", vals, tags=["shared"], overwrite=True)
            p.at("shared/quantization/logstdevs/tc", logstds_2d, tags=["shared"], overwrite=True)
            p.at("shared/quantization/counts/tc", counts, tags=["shared"], overwrite=True)
            kl, *_ = _quantization_kl_loss(p, kl_weight=1.0, step=0)
            return kl

        grad_2d = jax.grad(kl_2d)(values_2d)

        # rate_dim=1 gradient for dim 0's values
        def kl_1d_d0(vals):
            p = ParameterTree()
            p.at("shared/quantization/values/tc", vals, tags=["shared"], overwrite=True)
            p.at("shared/quantization/logstdevs/tc", jnp.full((2, 1), logstd_val), tags=["shared"], overwrite=True)
            p.at("shared/quantization/counts/tc", counts, tags=["shared"], overwrite=True)
            kl, *_ = _quantization_kl_loss(p, kl_weight=1.0, step=0)
            return kl

        grad_1d_d0 = jax.grad(kl_1d_d0)(jnp.array([[mu_d0[0]], [mu_d0[1]]]))

        # rate_dim=1 gradient for dim 1's values
        grad_1d_d1 = jax.grad(kl_1d_d0)(jnp.array([[mu_d1[0]], [mu_d1[1]]]))

        # dim 0 column of 2D grad should match the 1D-d0 grad
        assert jnp.allclose(grad_2d[:, 0], grad_1d_d0.ravel(), atol=1e-6), (
            f"Dim 0 grad mismatch: 2D={grad_2d[:, 0]}, 1D={grad_1d_d0.ravel()}"
        )
        # dim 1 column of 2D grad should match the 1D-d1 grad
        assert jnp.allclose(grad_2d[:, 1], grad_1d_d1.ravel(), atol=1e-6), (
            f"Dim 1 grad mismatch: 2D={grad_2d[:, 1]}, 1D={grad_1d_d1.ravel()}"
        )

    def test_kl_rate_dim1_unchanged(self):
        """For rate_dim=1, the fix is a no-op -- original_counts_sum == counts.sum()."""
        params = self._make_kl_params(
            rate_dim=1,
            values=[[0.3], [0.7]],
            logstds=[[-1.0], [0.5]],
            counts=[1.0, 4.0],
        )
        kl, klw, qvalues, logstds, counts, std = _quantization_kl_loss(params, kl_weight=1.0, step=0)

        from biocomp.train import stable_sigma
        mu = qvalues
        s = stable_sigma(logstds, min_std=1e-3)
        expected = 0.5 * (counts * (mu**2 + s**2 - 1 - 2 * logstds)).sum() / counts.sum()
        assert float(kl) == pytest.approx(float(expected), rel=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
