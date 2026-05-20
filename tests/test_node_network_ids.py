# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Tests for node_network_ids consistency across layer types."""

import pytest
import jax
import numpy as np
import dracon as dr
from biocomp.compute import ComputeStack, DEFAULT_COMPUTE_CONFIG
from biocomp.network import recipe_to_networks
from biocomp.recipe import Recipe
from biocomp.library import LibraryContext, load_lib
import biocomp.biorules as br
from biocomp.graphengine import is_inverse_node_type
from pathlib import Path

RESOURCES_DIR = Path(__file__).parent / "resources"
SCAFFOLD_PATH = RESOURCES_DIR / "design/architectures/two_and_one.yaml"


@pytest.fixture(scope="module")
def lib():
    return load_lib()


@pytest.fixture
def multi_network_stack(lib):
    """Create a stack with networks to test node_network_ids."""
    with LibraryContext.with_library(lib):
        # Load a recipe that produces networks
        data = dr.load(SCAFFOLD_PATH, context={"Recipe": Recipe})
        recipes = data["recipes"] if "recipes" in data else data.recipes
        networks = recipe_to_networks(recipes[0], br.ALL_RULES, invert=True)

        assert len(networks) >= 1, f"Expected >=1 networks, got {len(networks)}"

        stack = ComputeStack(networks)
        stack.build(DEFAULT_COMPUTE_CONFIG)
        return stack, networks


def test_node_network_ids_present_in_all_layers(multi_network_stack):
    """Verify that node_network_ids is stored for all layer types with per-node params."""
    stack, networks = multi_network_stack
    params = stack.init(jax.random.PRNGKey(0))

    # Get all namespaces that should have node_network_ids
    layers_with_node_params = []
    for layer in stack.layers:
        if len(layer.nodes) > 0:
            ns = stack.get_layer_namespace(layer.layer_id)
            layers_with_node_params.append((ns, layer.f_type, len(layer.nodes)))

    # Check each layer has node_network_ids
    missing = []
    for ns, f_type, n_nodes in layers_with_node_params:
        path = f"{ns}/node_network_ids"
        if path not in params:
            missing.append(f"{ns} ({f_type})")
        else:
            arr = params[path]
            assert arr.shape[-1] == n_nodes, (
                f"{path}: expected {n_nodes} nodes, got shape {arr.shape}"
            )

    # Some layers (like inv_* and shared) don't need node_network_ids
    # But aggregation, translation, transcription, ern, bias, source, output should have it
    required_types = {
        "aggregation",
        "translation",
        "transcription",
        "sequestron_ERN",
        "bias",
        "source",
        "output",
    }
    for ns, f_type, _ in layers_with_node_params:
        base_type = f_type.split("_")[0] if "_" in f_type else f_type
        # Skip inverse nodes - they share with forward
        if is_inverse_node_type(f_type):
            continue
        if base_type in required_types or any(t in f_type for t in required_types):
            path = f"{ns}/node_network_ids"
            assert path in params, f"Missing node_network_ids for {ns} ({f_type})"


def test_node_network_ids_values_valid(multi_network_stack):
    """Verify all network_ids are valid indices into stack.networks."""
    stack, networks = multi_network_stack
    params = stack.init(jax.random.PRNGKey(0))
    n_networks = len(networks)

    all_leaves = list(params.data.iter_leaves())
    for path, value in all_leaves:
        path_str = str(path)
        if "node_network_ids" not in path_str:
            continue

        arr = np.asarray(value.get_array() if hasattr(value, "get_array") else value)
        # Flatten to check all values
        flat = arr.ravel()

        assert np.all(flat >= 0), f"{path_str}: negative network_ids found: {flat[flat < 0]}"
        assert np.all(flat < n_networks), (
            f"{path_str}: network_ids >= n_networks ({n_networks}): {flat[flat >= n_networks]}"
        )


def test_node_network_ids_matches_layer_nodes(multi_network_stack):
    """Verify node_network_ids matches the actual StackNode.network_id values."""
    stack, networks = multi_network_stack
    params = stack.init(jax.random.PRNGKey(0))

    for layer in stack.layers:
        if len(layer.nodes) == 0:
            continue

        ns = stack.get_layer_namespace(layer.layer_id)
        path = f"{ns}/node_network_ids"

        if path not in params:
            continue  # Some layers don't have it (inverse nodes)

        stored_ids = np.asarray(params[path]).ravel()
        expected_ids = np.array([n.network_id for n in layer.nodes], dtype=np.int32)

        np.testing.assert_array_equal(
            stored_ids,
            expected_ids,
            err_msg=f"{path}: stored network_ids don't match layer.nodes",
        )


def test_node_network_ids_consistent_with_params_shape(multi_network_stack):
    """Verify node_network_ids length matches first dimension of per-node params."""
    stack, networks = multi_network_stack
    params = stack.init(jax.random.PRNGKey(0))

    # Find layers with node_network_ids and check other params in same namespace
    all_leaves = list(params.data.iter_leaves())
    network_id_map = {}

    for path, value in all_leaves:
        path_str = str(path)
        if "node_network_ids" in path_str:
            arr = np.asarray(value.get_array() if hasattr(value, "get_array") else value)
            namespace = path_str.rsplit("/node_network_ids", 1)[0]
            network_id_map[namespace] = arr.ravel()

    # Check params in each namespace have matching first dimension
    per_node_params = ["ratios", "tl_rate", "tc_rate", "raw_value", "affinity"]
    for path, value in all_leaves:
        path_str = str(path)

        # Find matching namespace
        namespace = None
        for ns in network_id_map:
            if path_str.startswith(ns + "/"):
                namespace = ns
                break

        if namespace is None:
            continue

        param_name = path_str.split("/")[-1]
        if param_name not in per_node_params:
            continue

        arr = np.asarray(value.get_array() if hasattr(value, "get_array") else value)
        n_network_ids = len(network_id_map[namespace])

        # First dim should match (after any replicate/target dims)
        # Typical shape: (n_nodes, ...) or (n_replicates, n_targets, n_nodes, ...)
        if arr.ndim >= 1:
            # Find the dim that matches n_network_ids
            matching_dim = None
            for d, size in enumerate(arr.shape):
                if size == n_network_ids:
                    matching_dim = d
                    break

            assert matching_dim is not None, (
                f"{path_str}: no dimension matches n_network_ids={n_network_ids}, shape={arr.shape}"
            )


def test_filtering_by_network_produces_correct_subset(multi_network_stack):
    """Verify filtering params by network produces the expected subset."""
    stack, networks = multi_network_stack
    params = stack.init(jax.random.PRNGKey(0))

    # Pick a network that has nodes
    test_network = 0

    all_leaves = list(params.data.iter_leaves())

    # Build network_id_map
    network_id_map = {}
    for path, value in all_leaves:
        path_str = str(path)
        if "node_network_ids" in path_str:
            arr = np.asarray(value.get_array() if hasattr(value, "get_array") else value).ravel()
            namespace = path_str.rsplit("/node_network_ids", 1)[0]
            network_id_map[namespace] = arr

    # For each namespace, verify filtering works
    for namespace, net_ids in network_id_map.items():
        mask = net_ids == test_network
        n_nodes_for_network = np.sum(mask)

        # Verify mask produces non-empty result for at least some networks
        if n_nodes_for_network == 0:
            # This network has no nodes in this layer - that's valid
            continue

        # Check that ratios (if present) can be filtered correctly
        ratios_path = f"{namespace}/ratios"
        if ratios_path in params:
            ratios = np.asarray(params[ratios_path])
            # ratios shape is typically (n_nodes, n_outputs)
            filtered = ratios[mask]
            assert filtered.shape[0] == n_nodes_for_network, (
                f"Filtering {ratios_path} produced wrong shape: "
                f"expected {n_nodes_for_network}, got {filtered.shape[0]}"
            )


def test_network_names_recoverable(multi_network_stack):
    """Verify we can map network_id back to network name."""
    stack, networks = multi_network_stack

    # Each network should have a unique name
    network_names = [n.name for n in networks]
    assert len(set(network_names)) == len(network_names), (
        f"Duplicate network names: {network_names}"
    )

    # Verify network_id matches position in networks list
    for i, network in enumerate(networks):
        assert i == networks.index(network), (
            f"Network {network.name} at index {i} doesn't match networks.index()"
        )
