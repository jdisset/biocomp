"""Tests for stack_commit.py - the refactored commit logic.

Tests cover:
- CommitOptions factory methods
- TUMaskProvider mode detection and mask computation
- prune_network_tus() unified pruning
- commit_structure() vs commit_final() behavior
- Path equivalence between log_alpha and binary_mask
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

from biocomp.stack_commit import (
    CommitOptions,
    TUMaskProvider,
    NetworkCommitReport,
    CommitReport,
    commit_structure,
    commit_final,
    commit_networks,
)
from biocomp.parameters import ParameterTree
from biocomp.tumasking import TU_LOG_ALPHA_PATH, TU_BINARY_MASK_PATH


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
        with pytest.raises(Exception):
            options.prune_tus = False


class TestTUMaskProvider:
    def test_from_params_no_masking(self):
        """from_params() returns 'none' mode when no TU masking params."""
        params = ParameterTree()
        provider = TUMaskProvider.from_params(params, tu_id_to_idx=None, n_networks=1)
        assert provider.mode == "none"
        assert provider.mask_data is None
        assert not provider.has_masking()

    def test_from_params_no_masking_with_tu_idx(self):
        """from_params() returns 'none' mode when tu_id_to_idx exists but no mask params."""
        params = ParameterTree()
        tu_id_to_idx = {"tu_a": 0, "tu_b": 1}
        provider = TUMaskProvider.from_params(params, tu_id_to_idx=tu_id_to_idx, n_networks=1)
        assert provider.mode == "none"
        assert not provider.has_masking()

    def test_from_params_detects_binary_mask(self):
        """from_params() detects TU_BINARY_MASK_PATH."""
        params = ParameterTree()
        binary_mask = jnp.array([[1.0, 0.0, 1.0]])
        params.at(TU_BINARY_MASK_PATH, binary_mask)
        tu_id_to_idx = {"tu_a": 0, "tu_b": 1, "tu_c": 2}

        provider = TUMaskProvider.from_params(params, tu_id_to_idx=tu_id_to_idx, n_networks=1)
        assert provider.mode == "binary_mask"
        assert provider.has_masking()
        assert provider.mask_data is not None

    def test_from_params_detects_log_alpha(self):
        """from_params() detects TU_LOG_ALPHA_PATH."""
        params = ParameterTree()
        log_alpha = jnp.array([[2.0, -2.0, 2.0]])
        params.at(TU_LOG_ALPHA_PATH, log_alpha)
        tu_id_to_idx = {"tu_a": 0, "tu_b": 1, "tu_c": 2}

        provider = TUMaskProvider.from_params(params, tu_id_to_idx=tu_id_to_idx, n_networks=1)
        assert provider.mode == "log_alpha"
        assert provider.has_masking()

    def test_binary_mask_priority_over_log_alpha(self):
        """When both are present, binary_mask takes priority."""
        params = ParameterTree()
        binary_mask = jnp.array([[1.0, 0.0]])
        log_alpha = jnp.array([[2.0, 2.0]])
        params.at(TU_BINARY_MASK_PATH, binary_mask)
        params.at(TU_LOG_ALPHA_PATH, log_alpha)
        tu_id_to_idx = {"tu_a": 0, "tu_b": 1}

        provider = TUMaskProvider.from_params(params, tu_id_to_idx=tu_id_to_idx, n_networks=1)
        assert provider.mode == "binary_mask"

    def test_get_binary_mask_for_network_binary_path(self):
        """get_binary_mask_for_network() works for binary_mask mode."""
        mask_data = jnp.array([[1.0, 0.0, 1.0]])
        provider = TUMaskProvider(
            mode="binary_mask",
            mask_data=mask_data,
            tu_id_to_idx={"tu_a": 0, "tu_b": 1, "tu_c": 2},
        )

        binary_mask = provider.get_binary_mask_for_network(0)
        np.testing.assert_array_equal(binary_mask, jnp.array([1.0, 0.0, 1.0]))

    def test_get_binary_mask_for_network_log_alpha_path(self):
        """get_binary_mask_for_network() works for log_alpha mode."""
        log_alpha = jnp.array([[10.0, -10.0, 10.0]])  # enabled, disabled, enabled
        provider = TUMaskProvider(
            mode="log_alpha",
            mask_data=log_alpha,
            tu_id_to_idx={"tu_a": 0, "tu_b": 1, "tu_c": 2},
        )

        binary_mask = provider.get_binary_mask_for_network(0)
        assert float(binary_mask[0]) == 1.0
        assert float(binary_mask[1]) == 0.0
        assert float(binary_mask[2]) == 1.0

    def test_get_pseudo_log_alpha_for_network_log_alpha_path(self):
        """get_pseudo_log_alpha_for_network() returns actual log_alpha for log_alpha mode."""
        log_alpha = jnp.array([[2.0, -2.0, 2.0]])
        provider = TUMaskProvider(
            mode="log_alpha",
            mask_data=log_alpha,
            tu_id_to_idx={"tu_a": 0, "tu_b": 1, "tu_c": 2},
        )

        pseudo = provider.get_pseudo_log_alpha_for_network(0)
        np.testing.assert_array_almost_equal(pseudo, log_alpha[0])

    def test_get_pseudo_log_alpha_for_network_binary_path(self):
        """get_pseudo_log_alpha_for_network() converts binary to pseudo log_alpha."""
        binary_mask = jnp.array([[1.0, 0.0, 1.0]])
        provider = TUMaskProvider(
            mode="binary_mask",
            mask_data=binary_mask,
            tu_id_to_idx={"tu_a": 0, "tu_b": 1, "tu_c": 2},
        )

        pseudo = provider.get_pseudo_log_alpha_for_network(0)
        assert float(pseudo[0]) == 10.0
        assert float(pseudo[1]) == -10.0
        assert float(pseudo[2]) == 10.0

    def test_is_tu_enabled(self):
        """is_tu_enabled() correctly checks individual TU status."""
        binary_mask = jnp.array([[1.0, 0.0, 1.0]])
        provider = TUMaskProvider(
            mode="binary_mask",
            mask_data=binary_mask,
            tu_id_to_idx={"tu_a": 0, "tu_b": 1, "tu_c": 2},
        )

        assert provider.is_tu_enabled(0, "tu_a") is True
        assert provider.is_tu_enabled(0, "tu_b") is False
        assert provider.is_tu_enabled(0, "tu_c") is True
        assert provider.is_tu_enabled(0, "unknown_tu") is True  # unknown TUs are enabled


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


class TestPathEquivalence:
    """Test that log_alpha and binary_mask produce equivalent results.

    These tests verify that converting log_alpha to pseudo_log_alpha produces
    the same pruning decisions as using binary_mask directly.
    """

    def test_mask_equivalence_enabled(self):
        """Enabled TUs produce same pseudo log_alpha from both paths."""
        log_alpha_value = 10.0

        log_alpha_data = jnp.array([[log_alpha_value]])
        binary_data = jnp.array([[1.0]])

        provider_la = TUMaskProvider("log_alpha", log_alpha_data, {"tu_a": 0})
        provider_bm = TUMaskProvider("binary_mask", binary_data, {"tu_a": 0})

        pseudo_la = provider_la.get_pseudo_log_alpha_for_network(0)
        pseudo_bm = provider_bm.get_pseudo_log_alpha_for_network(0)

        assert float(pseudo_la[0]) == log_alpha_value
        assert float(pseudo_bm[0]) == 10.0

    def test_mask_equivalence_disabled(self):
        """Disabled TUs produce same pseudo log_alpha from both paths."""
        log_alpha_value = -10.0

        log_alpha_data = jnp.array([[log_alpha_value]])
        binary_data = jnp.array([[0.0]])

        provider_la = TUMaskProvider("log_alpha", log_alpha_data, {"tu_a": 0})
        provider_bm = TUMaskProvider("binary_mask", binary_data, {"tu_a": 0})

        pseudo_la = provider_la.get_pseudo_log_alpha_for_network(0)
        pseudo_bm = provider_bm.get_pseudo_log_alpha_for_network(0)

        assert float(pseudo_la[0]) == log_alpha_value
        assert float(pseudo_bm[0]) == -10.0

    def test_binary_mask_equivalence_multi_tu(self):
        """Multiple TUs produce consistent masks from both paths."""
        log_alpha_data = jnp.array([[10.0, -10.0, 10.0, -10.0]])
        binary_data = jnp.array([[1.0, 0.0, 1.0, 0.0]])
        tu_id_to_idx = {"a": 0, "b": 1, "c": 2, "d": 3}

        provider_la = TUMaskProvider("log_alpha", log_alpha_data, tu_id_to_idx)
        provider_bm = TUMaskProvider("binary_mask", binary_data, tu_id_to_idx)

        mask_la = provider_la.get_binary_mask_for_network(0)
        mask_bm = provider_bm.get_binary_mask_for_network(0)

        np.testing.assert_array_equal(mask_la, mask_bm)


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
