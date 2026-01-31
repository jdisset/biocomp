"""Tests for hard-pruning functionality in design mode."""

import pytest
import numpy as np

from biocomp.design import DesignConfig, DesignManager
from biocomp.design_pruning import _merge_surviving_params
from biocomp.tumasking import TU_LOG_ALPHA_PATH
from biocomp.parameters import ParameterTree
from biocomp.design_targets import SVGTarget, LatticeSampling


class TestHardPruningConstraints:
    """Test that hard-pruning enforces single replicate/target."""

    def test_config_accepts_single_replicate_single_target(self):
        """DesignConfig can be created with hard_pruning_enabled=True."""
        dconf = DesignConfig(
            n_replicates=1,
            n_epochs=1,
            hard_pruning_enabled=True,
        )
        assert dconf.hard_pruning_enabled is True
        assert dconf.n_replicates == 1

    def test_hard_pruning_interval_is_configurable(self):
        """hard_pruning_interval can be set."""
        dconf = DesignConfig(
            n_replicates=1,
            n_epochs=1,
            hard_pruning_enabled=True,
            hard_pruning_interval=250,
        )
        assert dconf.hard_pruning_interval == 250

    def test_hard_pruning_preserve_minimum_is_configurable(self):
        """hard_pruning_preserve_minimum_tus can be set."""
        dconf = DesignConfig(
            n_replicates=1,
            n_epochs=1,
            hard_pruning_enabled=True,
            hard_pruning_preserve_minimum_tus=2,
        )
        assert dconf.hard_pruning_preserve_minimum_tus == 2


class TestMergeSurvivingParams:
    """Test parameter carry-over between pruning cycles."""

    def test_merge_copies_matching_paths(self):
        """Params with matching paths and shapes are copied."""
        old_params = ParameterTree()
        new_params = ParameterTree()

        old_val = np.array([[1.0, 2.0]])
        new_val = np.zeros_like(old_val)

        old_params.at("local/layer_0/ratios", old_val, overwrite=None)
        new_params.at("local/layer_0/ratios", new_val, overwrite=None)

        merged = _merge_surviving_params(old_params, new_params)
        assert np.allclose(merged["local/layer_0/ratios"], old_val)

    def test_merge_skips_tu_log_alpha(self):
        """tu_log_alpha paths are skipped (handled separately)."""
        old_params = ParameterTree()
        new_params = ParameterTree()

        old_val = np.array([[5.0]])
        new_val = np.array([[2.0]])

        old_params.at(TU_LOG_ALPHA_PATH, old_val, overwrite=None)
        new_params.at(TU_LOG_ALPHA_PATH, new_val, overwrite=None)

        merged = _merge_surviving_params(old_params, new_params)
        assert np.allclose(merged[TU_LOG_ALPHA_PATH], new_val)

    def test_merge_skips_shape_mismatch(self):
        """Paths with different shapes are not copied."""
        old_params = ParameterTree()
        new_params = ParameterTree()

        old_val = np.array([[1.0, 2.0, 3.0]])
        new_val = np.array([[0.0, 0.0]])

        old_params.at("local/layer_0/ratios", old_val, overwrite=None)
        new_params.at("local/layer_0/ratios", new_val, overwrite=None)

        merged = _merge_surviving_params(old_params, new_params)
        assert np.allclose(merged["local/layer_0/ratios"], new_val)

    def test_merge_skips_latent_tu_paths(self):
        """latent_tu paths are skipped (handled separately)."""
        old_params = ParameterTree()
        new_params = ParameterTree()

        old_val = np.array([[5.0]])
        new_val = np.array([[2.0]])

        old_params.at("design/latent_tu_z", old_val, overwrite=None)
        new_params.at("design/latent_tu_z", new_val, overwrite=None)

        merged = _merge_surviving_params(old_params, new_params)
        assert np.allclose(merged["design/latent_tu_z"], new_val)

    def test_merge_handles_missing_path_in_new(self):
        """Paths in old but not new are ignored."""
        old_params = ParameterTree()
        new_params = ParameterTree()

        old_val = np.array([[1.0, 2.0]])
        new_val = np.array([[0.0, 0.0]])

        old_params.at("local/layer_0/old_only", old_val, overwrite=None)
        old_params.at("local/layer_0/both", old_val, overwrite=None)
        new_params.at("local/layer_0/both", new_val, overwrite=None)

        merged = _merge_surviving_params(old_params, new_params)
        assert "local/layer_0/old_only" not in merged
        assert np.allclose(merged["local/layer_0/both"], old_val)


class TestDesignConfigHardPruning:
    """Test DesignConfig with hard pruning settings."""

    def test_default_hard_pruning_disabled(self):
        """Hard pruning is disabled by default."""
        dconf = DesignConfig()
        assert dconf.hard_pruning_enabled is False

    def test_hard_pruning_defaults(self):
        """Check default values for hard pruning params."""
        dconf = DesignConfig(hard_pruning_enabled=True)
        assert dconf.hard_pruning_interval == 500
        assert dconf.hard_pruning_ratio_threshold == 0.01
        assert dconf.hard_pruning_preserve_minimum_tus == 1
