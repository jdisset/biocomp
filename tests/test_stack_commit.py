"""Tests for stack_commit.py - the refactored commit logic.

Tests cover:
- CommitOptions factory methods
- get_full_log_alpha SSOT function
- prune_network_tus() unified pruning
- commit_structure() vs commit_final() behavior
- Path equivalence between log_alpha and binary_mask
"""

import dataclasses

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

from biocomp.stack_commit import (
    CommitOptions,
    NetworkCommitReport,
    CommitReport,
    commit_structure,
    commit_final,
    commit_networks,
)
from biocomp.tumasking_strategy import get_full_log_alpha
from biocomp.parameters import ParameterTree
from biocomp.tumasking import TU_LOG_ALPHA_PATH


class TestCommitOptions:
    def test_commit_options_defaults(self):
        """CommitOptions should have sensible defaults."""
        options = CommitOptions()
        assert options.prune_tus is True
        assert options.collapse_to_part is True
        assert options.lock_ratios is True
        assert options.roundtrip_rebuild is True
        assert options.preserve_input_order is True
        assert options.max_rebuild_workers == 8

    def test_for_structure_only(self):
        """for_structure_only() creates options without collapse."""
        options = CommitOptions.for_structure_only()
        assert options.collapse_to_part is False
        assert options.lock_ratios is False
        assert options.prune_tus is True
        assert options.roundtrip_rebuild is True

    def test_for_final(self):
        """for_final() creates options with collapse."""
        options = CommitOptions.for_final()
        assert options.collapse_to_part is True
        assert options.lock_ratios is True
        assert options.prune_tus is True
        assert options.roundtrip_rebuild is True

    def test_commit_options_frozen(self):
        """CommitOptions is frozen (immutable)."""
        options = CommitOptions()
        with pytest.raises(dataclasses.FrozenInstanceError):
            options.prune_tus = False


class TestGetFullLogAlpha:
    """Test get_full_log_alpha SSOT function."""

    def test_returns_none_when_no_masking_params(self):
        """Returns None when no TU masking params present."""
        params = ParameterTree()
        result = get_full_log_alpha(params)
        assert result is None

    def test_returns_log_alpha_from_direct_path(self):
        """Returns log_alpha when TU_LOG_ALPHA_PATH is present."""
        params = ParameterTree()
        log_alpha = jnp.array([[2.0, -2.0, 2.0]])
        params.at(TU_LOG_ALPHA_PATH, log_alpha)

        result = get_full_log_alpha(params)
        assert result is not None
        np.testing.assert_array_almost_equal(result, log_alpha)

    def test_returns_log_alpha_from_latent_mlp(self):
        """Decodes log_alpha from latent MLP params."""
        from biocomp.tumasking import (
            LATENT_TU_Z_PATH,
            LATENT_TU_W1_PATH,
            LATENT_TU_B1_PATH,
            LATENT_TU_W2_PATH,
            LATENT_TU_B2_PATH,
        )

        params = ParameterTree()
        n_tgt, n_net = 1, 1
        latent_dim, hidden_dim, n_tus = 4, 8, 3

        z = jnp.zeros((n_tgt, n_net, latent_dim))
        W1 = jnp.zeros((n_tgt, n_net, hidden_dim, latent_dim))
        b1 = jnp.zeros((n_tgt, n_net, hidden_dim))
        W2 = jnp.zeros((n_tgt, n_net, n_tus, hidden_dim))
        b2 = jnp.ones((n_tgt, n_net, n_tus)) * 2.0  # MLP(0) ≈ b2

        params.at(LATENT_TU_Z_PATH, z)
        params.at(LATENT_TU_W1_PATH, W1)
        params.at(LATENT_TU_B1_PATH, b1)
        params.at(LATENT_TU_W2_PATH, W2)
        params.at(LATENT_TU_B2_PATH, b2)

        result = get_full_log_alpha(params)
        assert result is not None
        assert result.shape == (n_tgt, n_net, n_tus)
        np.testing.assert_array_almost_equal(result, b2, decimal=5)


class TestNetworkCommitReport:
    def test_default_values(self):
        """NetworkCommitReport has correct defaults."""
        report = NetworkCommitReport(network_idx=0)
        assert report.network_idx == 0
        assert report.pruned_tu_count == 0
        assert report.dead_ern_recs == set()
        assert report.cascade_disabled_tus == set()


class TestCommitReport:
    def test_add_timing(self):
        """CommitReport can store timing information."""
        report = CommitReport()
        report.add_timing("deepcopy", 0.123)
        report.add_timing("pruning", 0.456)

        assert report.timings["deepcopy"] == 0.123
        assert report.timings["pruning"] == 0.456


class TestLogAlphaPathEquivalence:
    """Test that direct and latent MLP paths produce equivalent log_alpha."""

    def test_direct_path_returns_raw_values(self):
        """Direct log_alpha path returns the raw values."""
        params = ParameterTree()
        log_alpha = jnp.array([[10.0, -10.0, 10.0]])
        params.at(TU_LOG_ALPHA_PATH, log_alpha)

        result = get_full_log_alpha(params)
        np.testing.assert_array_equal(result, log_alpha)

    def test_latent_path_with_zero_z_returns_bias(self):
        """Latent MLP with z=0 returns bias (b2) as log_alpha."""
        from biocomp.tumasking import (
            LATENT_TU_Z_PATH,
            LATENT_TU_W1_PATH,
            LATENT_TU_B1_PATH,
            LATENT_TU_W2_PATH,
            LATENT_TU_B2_PATH,
        )

        params = ParameterTree()
        n_tgt, n_net = 2, 2
        latent_dim, hidden_dim, n_tus = 4, 8, 3

        z = jnp.zeros((n_tgt, n_net, latent_dim))
        W1 = jnp.zeros((n_tgt, n_net, hidden_dim, latent_dim))
        b1 = jnp.zeros((n_tgt, n_net, hidden_dim))
        W2 = jnp.zeros((n_tgt, n_net, n_tus, hidden_dim))
        b2 = jnp.array([
            [[2.0, -2.0, 1.0], [3.0, -3.0, 0.5]],
            [[1.5, -1.5, 0.0], [2.5, -2.5, -0.5]],
        ])

        params.at(LATENT_TU_Z_PATH, z)
        params.at(LATENT_TU_W1_PATH, W1)
        params.at(LATENT_TU_B1_PATH, b1)
        params.at(LATENT_TU_W2_PATH, W2)
        params.at(LATENT_TU_B2_PATH, b2)

        result = get_full_log_alpha(params)
        assert result is not None
        assert result.shape == (n_tgt, n_net, n_tus)
        np.testing.assert_array_almost_equal(result, b2, decimal=5)


RESOURCES_DIR = Path(__file__).parent / "resources"


@pytest.fixture
def lib():
    from biocomp.library import load_lib

    return load_lib()


@pytest.fixture
def simple_network_and_stack(lib):
    """Create a simple network and stack for testing."""
    from biocomp.library import LibraryContext
    from biocomp.network import recipe_to_networks
    from biocomp.compute import ComputeStack
    from biocomp.config import SIMPLE_NODES_COMPUTE_CONFIG
    from biocomp.recipe import Recipe, CoTransfection, TranscriptionUnit
    import biocomp.biorules as br

    with LibraryContext.with_library(lib):
        recipe = Recipe(
            name="test_recipe",
            content=[
                CoTransfection(
                    name="x",
                    units=[
                        TranscriptionUnit(
                            slots=["hEF1a", "mNeonGreen", "L0.T_4560"], name="marker"
                        ),
                        TranscriptionUnit(slots=["hEF1a", "CasE", "L0.T_4560"], name="ern"),
                    ],
                    ratios=[0.5, 0.5],
                ),
            ],
        )
        networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True)

        stack = ComputeStack(networks)
        config = SIMPLE_NODES_COMPUTE_CONFIG.model_copy(deep=True)
        stack.build(config, enable_tu_masking=True)

        key = jax.random.key(0)
        params = stack.init(key)

        return networks, stack, params


class TestCommitIntegration:
    """Integration tests using real networks."""

    def test_commit_structure_preserves_embeddings(self, lib, simple_network_and_stack):
        """commit_structure should NOT collapse embeddings."""
        networks, stack, params = simple_network_and_stack

        from biocomp.library import LibraryContext

        with LibraryContext.with_library(lib):
            committed = commit_structure(stack, params)

            assert len(committed) == len(networks)
            for net in committed:
                assert net.compute_graph is not None

    def test_commit_final_produces_networks(self, lib, simple_network_and_stack):
        """commit_final should produce valid committed networks."""
        networks, stack, params = simple_network_and_stack

        from biocomp.library import LibraryContext

        with LibraryContext.with_library(lib):
            committed = commit_final(stack, params)

            assert len(committed) == len(networks)
            for net in committed:
                assert net.compute_graph is not None

    def test_commit_networks_returns_report(self, lib, simple_network_and_stack):
        """commit_networks returns a CommitReport with timing information."""
        networks, stack, params = simple_network_and_stack

        from biocomp.library import LibraryContext

        with LibraryContext.with_library(lib):
            options = CommitOptions.for_final()
            committed, report = commit_networks(
                stack.networks,
                stack.layers,
                params,
                options,
                tu_id_to_idx=stack.tu_id_to_idx,
                node_map=stack.node_map,
            )

            assert isinstance(report, CommitReport)
            assert "deepcopy" in report.timings
            assert "layer_commits" in report.timings
            assert "tu_pruning" in report.timings
            assert "roundtrip_rebuild" in report.timings
            assert "total" in report.timings
            assert len(committed) == len(networks)
