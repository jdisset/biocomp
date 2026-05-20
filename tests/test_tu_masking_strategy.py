# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Tests for tumasking_strategy.py - the SSOT for TU masking behavior.

Tests cover:
- Strategy shape contracts (init vs get_* shapes)
- Protected TU handling (always enabled, zero gradient)
- Mode-specific behavior (direct, latent MLP, binary, none)
- get_full_log_alpha SSOT function
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from biocomp.tumasking_strategy import (
    TUMaskingMode,
    NoMaskingStrategy,
    DirectLogAlphaStrategy,
    LatentMLPStrategy,
    BinaryMaskStrategy,
    build_tu_masking_strategy,
    get_full_log_alpha,
    PROTECTED_LOG_ALPHA,
)
from biocomp.tumasking import (
    TU_LOG_ALPHA_PATH,
    TU_BINARY_MASK_PATH,
    PROTECTED_TU_MASK_PATH,
    LATENT_TU_Z_PATH,
    LATENT_TU_W1_PATH,
    LATENT_TU_B1_PATH,
    LATENT_TU_W2_PATH,
    LATENT_TU_B2_PATH,
)
from biocomp.parameters import ParameterTree


class TestNoMaskingStrategy:
    def test_has_masking_false(self):
        strategy = NoMaskingStrategy(n_tus=5)
        assert strategy.has_masking is False

    def test_mode_is_none(self):
        strategy = NoMaskingStrategy(n_tus=5)
        assert strategy.mode == TUMaskingMode.NONE

    def test_get_log_alpha_returns_high_values(self):
        strategy = NoMaskingStrategy(n_tus=5)
        params = ParameterTree()
        log_alpha = strategy.get_log_alpha(params, network_id=0)
        assert log_alpha.shape == (5,)
        np.testing.assert_array_equal(log_alpha, jnp.full(5, PROTECTED_LOG_ALPHA))

    def test_get_binary_mask_returns_all_ones(self):
        strategy = NoMaskingStrategy(n_tus=5)
        params = ParameterTree()
        mask = strategy.get_binary_mask(params, network_id=0)
        assert mask.shape == (5,)
        np.testing.assert_array_equal(mask, jnp.ones(5))

    def test_param_paths_empty(self):
        strategy = NoMaskingStrategy()
        assert strategy.param_paths == ()


class TestDirectLogAlphaStrategy:
    def test_has_masking_true(self):
        strategy = DirectLogAlphaStrategy()
        assert strategy.has_masking is True

    def test_mode_is_direct(self):
        strategy = DirectLogAlphaStrategy()
        assert strategy.mode == TUMaskingMode.DIRECT

    def test_init_params_creates_correct_shapes(self):
        strategy = DirectLogAlphaStrategy(init_mean=2.0, init_std=0.5)
        params = ParameterTree()
        n_rep, n_tgt, n_net, n_tus = 2, 3, 4, 5

        strategy.init_params(
            params,
            n_replicates=n_rep,
            n_targets=n_tgt,
            n_networks=n_net,
            n_tus=n_tus,
            key=jax.random.key(0),
            protected_tu_ids=set(),
            tu_id_to_idx={},
        )

        assert TU_LOG_ALPHA_PATH in params
        assert params[TU_LOG_ALPHA_PATH].shape == (n_rep, n_tgt, n_net, n_tus)
        assert PROTECTED_TU_MASK_PATH in params
        assert params[PROTECTED_TU_MASK_PATH].shape == (n_rep, n_tgt, n_tus)

    def test_get_log_alpha_returns_correct_shape(self):
        strategy = DirectLogAlphaStrategy()
        n_net, n_tus = 2, 5

        inner_params = {
            TU_LOG_ALPHA_PATH: jax.random.normal(jax.random.key(0), (n_net, n_tus)),
            PROTECTED_TU_MASK_PATH: jnp.zeros(n_tus, dtype=bool),
        }

        log_alpha = strategy.get_log_alpha(inner_params, network_id=0)
        assert log_alpha.shape == (n_tus,)

    def test_protected_tu_always_enabled(self):
        strategy = DirectLogAlphaStrategy()
        n_net, n_tus = 2, 5
        protected_idx = 2

        inner_params = {
            TU_LOG_ALPHA_PATH: jnp.full((n_net, n_tus), -10.0),  # all would be disabled
            PROTECTED_TU_MASK_PATH: jnp.array([False, False, True, False, False]),
        }

        log_alpha = strategy.get_log_alpha(inner_params, network_id=0)
        assert float(log_alpha[protected_idx]) == PROTECTED_LOG_ALPHA

        mask = strategy.get_binary_mask(inner_params, network_id=0)
        assert float(mask[protected_idx]) == 1.0

    def test_protected_tu_no_gradient_pressure(self):
        """Protected TUs receive zero gradients via stop_gradient."""
        strategy = DirectLogAlphaStrategy()
        n_net, n_tus = 2, 5
        protected_idx = 2
        unprotected_idx = 0

        log_alpha_init = jax.random.normal(jax.random.key(0), (n_net, n_tus))
        protected_mask = jnp.array([False, False, True, False, False])

        def loss_fn(log_alpha):
            inner_params = {
                TU_LOG_ALPHA_PATH: log_alpha,
                PROTECTED_TU_MASK_PATH: protected_mask,
            }
            result = strategy.get_log_alpha(inner_params, network_id=0)
            return jnp.sum(result)

        grads = jax.grad(loss_fn)(log_alpha_init)
        tu_grads = grads[0]

        assert float(tu_grads[protected_idx]) == 0.0, "Protected TU should get no gradient"
        assert float(tu_grads[unprotected_idx]) != 0.0, "Unprotected TU should get gradient"

    def test_param_paths(self):
        strategy = DirectLogAlphaStrategy()
        assert TU_LOG_ALPHA_PATH in strategy.param_paths
        assert PROTECTED_TU_MASK_PATH in strategy.param_paths


class TestLatentMLPStrategy:
    def test_has_masking_true(self):
        strategy = LatentMLPStrategy()
        assert strategy.has_masking is True

    def test_mode_is_latent_mlp(self):
        strategy = LatentMLPStrategy()
        assert strategy.mode == TUMaskingMode.LATENT_MLP

    def test_init_params_creates_correct_shapes(self):
        latent_dim, hidden_dim = 8, 16
        strategy = LatentMLPStrategy(latent_dim=latent_dim, hidden_dim=hidden_dim)
        params = ParameterTree()
        n_rep, n_tgt, n_net, n_tus = 2, 3, 4, 5

        strategy.init_params(
            params,
            n_replicates=n_rep,
            n_targets=n_tgt,
            n_networks=n_net,
            n_tus=n_tus,
            key=jax.random.key(0),
            protected_tu_ids=set(),
            tu_id_to_idx={},
        )

        assert params[LATENT_TU_Z_PATH].shape == (n_rep, n_tgt, n_net, latent_dim)
        assert params[LATENT_TU_W1_PATH].shape == (n_rep, n_tgt, n_net, hidden_dim, latent_dim)
        assert params[LATENT_TU_B1_PATH].shape == (n_rep, n_tgt, n_net, hidden_dim)
        assert params[LATENT_TU_W2_PATH].shape == (n_rep, n_tgt, n_net, n_tus, hidden_dim)
        assert params[LATENT_TU_B2_PATH].shape == (n_rep, n_tgt, n_net, n_tus)
        assert params[PROTECTED_TU_MASK_PATH].shape == (n_rep, n_tgt, n_tus)

    def test_get_log_alpha_returns_correct_shape(self):
        latent_dim, hidden_dim = 4, 8
        strategy = LatentMLPStrategy(latent_dim=latent_dim, hidden_dim=hidden_dim)
        n_net, n_tus = 2, 5

        inner_params = {
            LATENT_TU_Z_PATH: jnp.zeros((n_net, latent_dim)),
            LATENT_TU_W1_PATH: jnp.zeros((n_net, hidden_dim, latent_dim)),
            LATENT_TU_B1_PATH: jnp.zeros((n_net, hidden_dim)),
            LATENT_TU_W2_PATH: jnp.zeros((n_net, n_tus, hidden_dim)),
            LATENT_TU_B2_PATH: jnp.ones((n_net, n_tus)) * 2.0,
            PROTECTED_TU_MASK_PATH: jnp.zeros(n_tus, dtype=bool),
        }

        log_alpha = strategy.get_log_alpha(inner_params, network_id=0)
        assert log_alpha.shape == (n_tus,)
        np.testing.assert_array_almost_equal(log_alpha, jnp.full(n_tus, 2.0), decimal=5)

    def test_protected_tu_always_enabled(self):
        latent_dim, hidden_dim = 4, 8
        strategy = LatentMLPStrategy(latent_dim=latent_dim, hidden_dim=hidden_dim)
        n_net, n_tus = 2, 5
        protected_idx = 2

        inner_params = {
            LATENT_TU_Z_PATH: jnp.zeros((n_net, latent_dim)),
            LATENT_TU_W1_PATH: jnp.zeros((n_net, hidden_dim, latent_dim)),
            LATENT_TU_B1_PATH: jnp.zeros((n_net, hidden_dim)),
            LATENT_TU_W2_PATH: jnp.zeros((n_net, n_tus, hidden_dim)),
            LATENT_TU_B2_PATH: jnp.full((n_net, n_tus), -10.0),  # all would be disabled
            PROTECTED_TU_MASK_PATH: jnp.array([False, False, True, False, False]),
        }

        log_alpha = strategy.get_log_alpha(inner_params, network_id=0)
        assert float(log_alpha[protected_idx]) == PROTECTED_LOG_ALPHA

    def test_protected_tu_no_gradient_pressure(self):
        """Protected TUs receive zero gradients in latent MLP mode."""
        latent_dim, hidden_dim = 4, 8
        strategy = LatentMLPStrategy(latent_dim=latent_dim, hidden_dim=hidden_dim)
        n_net, n_tus = 2, 5
        protected_idx = 2
        unprotected_idx = 0

        z_init = jax.random.normal(jax.random.key(0), (n_net, latent_dim))
        W1_init = jax.random.normal(jax.random.key(1), (n_net, hidden_dim, latent_dim))
        b1_init = jnp.zeros((n_net, hidden_dim))
        W2_init = jax.random.normal(jax.random.key(2), (n_net, n_tus, hidden_dim))
        b2_init = jnp.zeros((n_net, n_tus))
        protected_mask = jnp.array([False, False, True, False, False])

        def loss_fn(b2):
            inner_params = {
                LATENT_TU_Z_PATH: z_init,
                LATENT_TU_W1_PATH: W1_init,
                LATENT_TU_B1_PATH: b1_init,
                LATENT_TU_W2_PATH: W2_init,
                LATENT_TU_B2_PATH: b2,
                PROTECTED_TU_MASK_PATH: protected_mask,
            }
            log_alpha = strategy.get_log_alpha(inner_params, network_id=0)
            return jnp.sum(log_alpha)

        grads = jax.grad(loss_fn)(b2_init)
        b2_grads = grads[0]

        assert float(b2_grads[protected_idx]) == 0.0, "Protected TU b2 should get no gradient"
        assert float(b2_grads[unprotected_idx]) != 0.0, "Unprotected TU should get gradient"

    def test_param_paths(self):
        strategy = LatentMLPStrategy()
        assert LATENT_TU_Z_PATH in strategy.param_paths
        assert LATENT_TU_W1_PATH in strategy.param_paths
        assert LATENT_TU_B1_PATH in strategy.param_paths
        assert LATENT_TU_W2_PATH in strategy.param_paths
        assert LATENT_TU_B2_PATH in strategy.param_paths
        assert PROTECTED_TU_MASK_PATH in strategy.param_paths


class TestBinaryMaskStrategy:
    def test_has_masking_true(self):
        strategy = BinaryMaskStrategy()
        assert strategy.has_masking is True

    def test_mode_is_binary(self):
        strategy = BinaryMaskStrategy()
        assert strategy.mode == TUMaskingMode.BINARY

    def test_init_params_creates_all_ones(self):
        strategy = BinaryMaskStrategy()
        params = ParameterTree()
        n_rep, n_tgt, n_net, n_tus = 2, 3, 4, 5

        strategy.init_params(
            params,
            n_replicates=n_rep,
            n_targets=n_tgt,
            n_networks=n_net,
            n_tus=n_tus,
        )

        assert TU_BINARY_MASK_PATH in params
        assert params[TU_BINARY_MASK_PATH].shape == (n_rep, n_tgt, n_net, n_tus)
        np.testing.assert_array_equal(params[TU_BINARY_MASK_PATH], jnp.ones((n_rep, n_tgt, n_net, n_tus)))

    def test_get_log_alpha_converts_binary_to_pseudo(self):
        strategy = BinaryMaskStrategy()
        _n_net, _n_tus = 2, 4

        inner_params = {
            TU_BINARY_MASK_PATH: jnp.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]),
        }

        log_alpha_net0 = strategy.get_log_alpha(inner_params, network_id=0)
        assert float(log_alpha_net0[0]) == PROTECTED_LOG_ALPHA
        assert float(log_alpha_net0[1]) == -PROTECTED_LOG_ALPHA

        log_alpha_net1 = strategy.get_log_alpha(inner_params, network_id=1)
        assert float(log_alpha_net1[0]) == -PROTECTED_LOG_ALPHA
        assert float(log_alpha_net1[1]) == PROTECTED_LOG_ALPHA

    def test_get_binary_mask_returns_raw_mask(self):
        strategy = BinaryMaskStrategy()
        _n_net, _n_tus = 2, 4

        inner_params = {
            TU_BINARY_MASK_PATH: jnp.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]),
        }

        mask = strategy.get_binary_mask(inner_params, network_id=0)
        np.testing.assert_array_equal(mask, jnp.array([1.0, 0.0, 1.0, 0.0]))

    def test_set_mask_updates_params(self):
        strategy = BinaryMaskStrategy()
        params = ParameterTree()
        n_rep, n_tgt, n_net, n_tus = 1, 1, 2, 3

        strategy.init_params(params, n_replicates=n_rep, n_targets=n_tgt, n_networks=n_net, n_tus=n_tus)

        new_mask = jnp.array([[[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]])
        strategy.set_mask(params, new_mask)

        np.testing.assert_array_equal(params[TU_BINARY_MASK_PATH], new_mask)

    def test_param_paths(self):
        strategy = BinaryMaskStrategy()
        assert TU_BINARY_MASK_PATH in strategy.param_paths


class TestBuildTUMaskingStrategy:
    def test_build_none_mode(self):
        strategy = build_tu_masking_strategy(TUMaskingMode.NONE)
        assert isinstance(strategy, NoMaskingStrategy)

    def test_build_direct_mode(self):
        strategy = build_tu_masking_strategy(TUMaskingMode.DIRECT, init_mean=3.0, init_std=0.1)
        assert isinstance(strategy, DirectLogAlphaStrategy)
        assert strategy.init_mean == 3.0
        assert strategy.init_std == 0.1

    def test_build_latent_mlp_mode(self):
        strategy = build_tu_masking_strategy(
            TUMaskingMode.LATENT_MLP, latent_dim=32, hidden_dim=64
        )
        assert isinstance(strategy, LatentMLPStrategy)
        assert strategy.latent_dim == 32
        assert strategy.hidden_dim == 64

    def test_build_binary_mode(self):
        strategy = build_tu_masking_strategy(TUMaskingMode.BINARY)
        assert isinstance(strategy, BinaryMaskStrategy)


class TestGetFullLogAlpha:
    def test_returns_none_when_no_masking(self):
        params = ParameterTree()
        result = get_full_log_alpha(params)
        assert result is None

    def test_returns_direct_log_alpha(self):
        params = ParameterTree()
        log_alpha = jnp.array([[2.0, -2.0, 1.0]])
        params.at(TU_LOG_ALPHA_PATH, log_alpha)

        result = get_full_log_alpha(params)
        np.testing.assert_array_equal(result, log_alpha)

    def test_returns_pseudo_log_alpha_from_binary_mask(self):
        params = ParameterTree()
        binary_mask = jnp.array([[1.0, 0.0, 1.0]])
        params.at(TU_BINARY_MASK_PATH, binary_mask)

        result = get_full_log_alpha(params)
        assert result is not None
        assert float(result[0, 0]) == PROTECTED_LOG_ALPHA
        assert float(result[0, 1]) == -PROTECTED_LOG_ALPHA
        assert float(result[0, 2]) == PROTECTED_LOG_ALPHA

    def test_decodes_latent_mlp_3d(self):
        """Test latent MLP decode with 3D z (inside replicate vmap)."""
        params = ParameterTree()
        n_tgt, n_net = 2, 2
        latent_dim, hidden_dim, n_tus = 4, 8, 3

        z = jnp.zeros((n_tgt, n_net, latent_dim))
        W1 = jnp.zeros((n_tgt, n_net, hidden_dim, latent_dim))
        b1 = jnp.zeros((n_tgt, n_net, hidden_dim))
        W2 = jnp.zeros((n_tgt, n_net, n_tus, hidden_dim))
        b2 = jnp.ones((n_tgt, n_net, n_tus)) * 2.5

        params.at(LATENT_TU_Z_PATH, z)
        params.at(LATENT_TU_W1_PATH, W1)
        params.at(LATENT_TU_B1_PATH, b1)
        params.at(LATENT_TU_W2_PATH, W2)
        params.at(LATENT_TU_B2_PATH, b2)

        result = get_full_log_alpha(params)
        assert result is not None
        assert result.shape == (n_tgt, n_net, n_tus)
        np.testing.assert_array_almost_equal(result, b2, decimal=5)

    def test_decodes_latent_mlp_4d(self):
        """Test latent MLP decode with 4D z (full params)."""
        params = ParameterTree()
        n_rep, n_tgt, n_net = 2, 2, 2
        latent_dim, hidden_dim, n_tus = 4, 8, 3

        z = jnp.zeros((n_rep, n_tgt, n_net, latent_dim))
        W1 = jnp.zeros((n_rep, n_tgt, n_net, hidden_dim, latent_dim))
        b1 = jnp.zeros((n_rep, n_tgt, n_net, hidden_dim))
        W2 = jnp.zeros((n_rep, n_tgt, n_net, n_tus, hidden_dim))
        b2 = jnp.ones((n_rep, n_tgt, n_net, n_tus)) * 1.5

        params.at(LATENT_TU_Z_PATH, z)
        params.at(LATENT_TU_W1_PATH, W1)
        params.at(LATENT_TU_B1_PATH, b1)
        params.at(LATENT_TU_W2_PATH, W2)
        params.at(LATENT_TU_B2_PATH, b2)

        result = get_full_log_alpha(params)
        assert result is not None
        assert result.shape == (n_rep, n_tgt, n_net, n_tus)
        np.testing.assert_array_almost_equal(result, b2, decimal=5)

    def test_binary_mask_priority_over_log_alpha(self):
        """Binary mask takes priority when both are present."""
        params = ParameterTree()
        binary_mask = jnp.array([[1.0, 0.0]])
        log_alpha = jnp.array([[-10.0, 10.0]])  # opposite of binary
        params.at(TU_BINARY_MASK_PATH, binary_mask)
        params.at(TU_LOG_ALPHA_PATH, log_alpha)

        result = get_full_log_alpha(params)
        assert float(result[0, 0]) == PROTECTED_LOG_ALPHA  # from binary 1.0
        assert float(result[0, 1]) == -PROTECTED_LOG_ALPHA  # from binary 0.0

    def test_protected_tu_enforcement_direct(self):
        """get_full_log_alpha applies protected TU enforcement for direct mode."""
        params = ParameterTree()
        log_alpha = jnp.array([[-10.0, -10.0, -10.0]])  # all would be disabled
        protected_mask = jnp.array([False, True, False])  # index 1 is protected
        params.at(TU_LOG_ALPHA_PATH, log_alpha)
        params.at(PROTECTED_TU_MASK_PATH, protected_mask)

        result = get_full_log_alpha(params)
        assert float(result[0, 0]) == -10.0  # not protected
        assert float(result[0, 1]) == PROTECTED_LOG_ALPHA  # protected -> forced to 10.0
        assert float(result[0, 2]) == -10.0  # not protected

    def test_protected_tu_enforcement_latent(self):
        """get_full_log_alpha applies protected TU enforcement for latent mode."""
        params = ParameterTree()
        n_net, latent_dim, hidden_dim, n_tus = 1, 4, 8, 3

        z = jnp.zeros((n_net, latent_dim))
        W1 = jnp.zeros((n_net, hidden_dim, latent_dim))
        b1 = jnp.zeros((n_net, hidden_dim))
        W2 = jnp.zeros((n_net, n_tus, hidden_dim))
        b2 = jnp.full((n_net, n_tus), -10.0)  # all would be disabled
        protected_mask = jnp.array([False, True, False])  # index 1 is protected

        params.at(LATENT_TU_Z_PATH, z)
        params.at(LATENT_TU_W1_PATH, W1)
        params.at(LATENT_TU_B1_PATH, b1)
        params.at(LATENT_TU_W2_PATH, W2)
        params.at(LATENT_TU_B2_PATH, b2)
        params.at(PROTECTED_TU_MASK_PATH, protected_mask)

        result = get_full_log_alpha(params)
        assert float(result[0, 0]) == -10.0  # not protected
        assert float(result[0, 1]) == PROTECTED_LOG_ALPHA  # protected -> forced to 10.0
        assert float(result[0, 2]) == -10.0  # not protected

    def test_protected_tu_no_gradient_in_get_full_log_alpha(self):
        """Protected TUs receive zero gradients through get_full_log_alpha."""
        protected_idx = 1

        def loss_fn(log_alpha):
            params = {
                TU_LOG_ALPHA_PATH: log_alpha,
                PROTECTED_TU_MASK_PATH: jnp.array([False, True, False]),
            }
            result = get_full_log_alpha(params)
            return jnp.sum(result)

        log_alpha_init = jnp.array([[2.0, 2.0, 2.0]])
        grads = jax.grad(loss_fn)(log_alpha_init)

        assert float(grads[0, 0]) == 1.0  # not protected, gradient flows
        assert float(grads[0, protected_idx]) == 0.0  # protected, zero gradient
        assert float(grads[0, 2]) == 1.0  # not protected, gradient flows


class TestGetTUMasksIntegration:
    """Test get_tu_masks uses get_full_log_alpha correctly for protected TU enforcement."""

    def test_protected_tu_always_enabled_in_get_tu_masks(self):
        """Protected TUs return mask=1.0 in get_tu_masks()."""
        from biocomp.tumasking import get_tu_masks

        n_net, n_tus = 2, 5
        protected_idx = 2

        inner_params = {
            TU_LOG_ALPHA_PATH: jnp.full((n_net, n_tus), -10.0),  # all would be disabled
            PROTECTED_TU_MASK_PATH: jnp.array([False, False, True, False, False]),
        }

        tu_indices = jnp.array([0, 1, 2, 3, 4])  # single TU per input
        masks = get_tu_masks(inner_params, tu_indices, network_id=0, is_multi_tu=False)

        assert float(masks[protected_idx]) == 1.0, "Protected TU should be enabled"
        for i in [0, 1, 3, 4]:
            assert float(masks[i]) == 0.0, f"Unprotected TU {i} should be disabled"

    def test_protected_tu_latent_mode_in_get_tu_masks(self):
        """Protected TUs return mask=1.0 in get_tu_masks() with latent mode."""
        from biocomp.tumasking import get_tu_masks

        n_net, n_tus = 2, 3
        latent_dim, hidden_dim = 4, 8
        protected_idx = 1

        inner_params = {
            LATENT_TU_Z_PATH: jnp.zeros((n_net, latent_dim)),
            LATENT_TU_W1_PATH: jnp.zeros((n_net, hidden_dim, latent_dim)),
            LATENT_TU_B1_PATH: jnp.zeros((n_net, hidden_dim)),
            LATENT_TU_W2_PATH: jnp.zeros((n_net, n_tus, hidden_dim)),
            LATENT_TU_B2_PATH: jnp.full((n_net, n_tus), -10.0),  # all would be disabled
            PROTECTED_TU_MASK_PATH: jnp.array([False, True, False]),
        }

        tu_indices = jnp.array([0, 1, 2])
        masks = get_tu_masks(inner_params, tu_indices, network_id=0, is_multi_tu=False)

        assert float(masks[protected_idx]) == 1.0, "Protected TU should be enabled"
        assert float(masks[0]) == 0.0, "Unprotected TU should be disabled"
        assert float(masks[2]) == 0.0, "Unprotected TU should be disabled"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
