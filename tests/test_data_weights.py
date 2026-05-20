#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Tests for weighted data sources in training."""

import pytest
import numpy as np
from biocomp.datautils import DataManager, DataConfig


class MockNetwork:
    """Mock network for testing."""
    def __init__(self, name, nb_inputs=2, nb_outputs=1):
        self.name = name
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs
        self.metadata = {}

    def get_nb_inputs(self):
        return self.nb_inputs

    def get_nb_outputs(self):
        return self.nb_outputs

    def get_dependent_output_mask(self):
        return np.array([True] * self.nb_outputs)

    def get_output_proteins(self):
        return [f"out{i}" for i in range(self.nb_outputs)]

    def get_input_proteins(self):
        return [f"in{i}" for i in range(self.nb_inputs)]


class TestDataManagerWeights:
    """Test DataManager weight functionality."""

    @staticmethod
    def _make_valid_data(n_samples, n_features):
        """Create data within the valid range (500, 1e8)."""
        return np.random.uniform(1000, 10000, size=(n_samples, n_features))

    @staticmethod
    def _no_check_config():
        """Create data config with checks disabled."""
        cfg = DataConfig()
        cfg.perform_data_checks = False
        return cfg

    def test_default_weights(self):
        """Test that weights default to 1.0."""
        X = [self._make_valid_data(10, 2), self._make_valid_data(10, 2)]
        Y = [self._make_valid_data(10, 1), self._make_valid_data(10, 1)]
        networks = [MockNetwork("net1"), MockNetwork("net2")]

        dman = DataManager(X, Y, networks, data_cfg=self._no_check_config())
        weights = dman.get_weights()

        assert len(weights) == 2
        assert weights == [1.0, 1.0]

    def test_custom_weights(self):
        """Test custom weight assignment."""
        X = [self._make_valid_data(10, 2) for _ in range(3)]
        Y = [self._make_valid_data(10, 1) for _ in range(3)]
        networks = [MockNetwork("net1"), MockNetwork("net2"), MockNetwork("net3")]
        weights = [1.0, 2.0, 0.5]

        dman = DataManager(X, Y, networks, weights=weights, data_cfg=self._no_check_config())
        assert dman.get_weights() == [1.0, 2.0, 0.5]

    def test_weights_assertion_mismatch(self):
        """Test that mismatched weights raise assertion."""
        X = [self._make_valid_data(10, 2)]
        Y = [self._make_valid_data(10, 1)]
        networks = [MockNetwork("net1")]
        weights = [1.0, 2.0]  # wrong length

        with pytest.raises(AssertionError):
            DataManager(X, Y, networks, weights=weights, data_cfg=self._no_check_config())

    def test_make_subset_preserves_weights(self):
        """Test that make_subset preserves weight information."""
        X = [self._make_valid_data(10, 2) for _ in range(3)]
        Y = [self._make_valid_data(10, 1) for _ in range(3)]
        networks = [MockNetwork("net1"), MockNetwork("net2"), MockNetwork("net3")]
        weights = [1.0, 2.0, 3.0]

        dman = DataManager(X, Y, networks, weights=weights, data_cfg=self._no_check_config())
        subset = dman.make_subset([0, 2])

        assert subset.get_weights() == [1.0, 3.0]
        assert len(subset.get_networks()) == 2


class TestExpandWeightsToOutputs:
    """Test expand_weights_to_outputs helper function."""

    def test_expand_weights_multiple_networks(self):
        """Test per-output weight expansion with multiple networks."""
        from biocomp.train import expand_weights_to_outputs

        networks = [
            MockNetwork("net1", nb_inputs=2, nb_outputs=2),
            MockNetwork("net2", nb_inputs=2, nb_outputs=3),
        ]
        weights = [1.0, 2.0]

        per_output = expand_weights_to_outputs(weights, networks)

        # net1 has 2 outputs with weight 1.0
        # net2 has 3 outputs with weight 2.0
        expected = [1.0, 1.0, 2.0, 2.0, 2.0]
        assert per_output == expected

    def test_expand_weights_single_network(self):
        """Test single network case."""
        from biocomp.train import expand_weights_to_outputs

        networks = [MockNetwork("net1", nb_inputs=3, nb_outputs=4)]
        weights = [0.5]

        per_output = expand_weights_to_outputs(weights, networks)

        expected = [0.5, 0.5, 0.5, 0.5]
        assert per_output == expected


class TestNetworkSelectorWeights:
    """Test NetworkSelector weight field."""

    def test_network_selector_default_weight(self):
        """Test that NetworkSelector has default weight of 1.0."""
        from biocomptools.toollib.networkselector import NetworkSelector

        selector = NetworkSelector(experiment_name="test")
        assert selector.weight == 1.0

    def test_network_selector_custom_weight(self):
        """Test NetworkSelector with custom weight."""
        from biocomptools.toollib.networkselector import NetworkSelector

        selector = NetworkSelector(experiment_name="test", weight=2.5)
        assert selector.weight == 2.5


class TestNetworkSetWeights:
    """Test NetworkSet weight functionality."""

    def test_network_set_default_weight(self):
        """Test NetworkSet default weight is None."""
        from biocomptools.toollib.networkselector import NetworkSet

        ns = NetworkSet()
        assert ns.weight is None

    def test_network_set_custom_weight(self):
        """Test NetworkSet with custom weight."""
        from biocomptools.toollib.networkselector import NetworkSet

        ns = NetworkSet(weight=3.0)
        assert ns.weight == 3.0


class TestNetworkDataPairWeights:
    """Test NetworkDataPair weight property."""

    def test_network_data_pair_default_weight(self):
        """Test NetworkDataPair default weight."""
        from biocomptools.toollib.models import NetworkDataPair

        ndp = NetworkDataPair(network_name="test", datafile_path="/test/path")
        assert ndp.weight == 1.0

    def test_network_data_pair_set_weight(self):
        """Test NetworkDataPair weight setter."""
        from biocomptools.toollib.models import NetworkDataPair

        ndp = NetworkDataPair(network_name="test", datafile_path="/test/path")
        ndp.weight = 2.5
        assert ndp.weight == 2.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
