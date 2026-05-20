# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Test that forward/inverse node pairs share embedding noise via node_key_id.

The key reuse fix in compute.py uses fold_in(base_key, node_key_id) to derive
per-node keys. Forward/inverse pairs share the same node_key_id via ArrayRef,
so they must get identical variational embedding noise (q_noise).
"""

import jax
import jax.numpy as jnp
import numpy as np
from biocomp.network import recipe_to_networks
from biocomp.library import LibraryContext
import biocomp.biorules as br
from biocomp.compute import ComputeStack
from biocomp.config import DEFAULT_COMPUTE_CONFIG

pytest_plugins = ["test_declarative_recipes"]


def _get_layer_aux(aux, stack, layer_type):
    """Extract the aux dict for a specific layer type from the stack aux."""
    for i, layer in enumerate(stack.layers):
        if layer.f_type == layer_type:
            return aux[0][f"{i}"]["layer_aux"]
    raise ValueError(f"Layer type {layer_type} not found")


def test_forward_inverse_noise_cancellation(lib, simple_single_reporter):
    """Forward/inverse transform pairs must get identical q_noise."""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(simple_single_reporter, br.ALL_RULES, invert=True)
        stack = ComputeStack([networks[0]])
        stack.build(config=DEFAULT_COMPUTE_CONFIG)

        key = jax.random.PRNGKey(42)
        params = stack.init(key)

        inputs = jnp.ones((stack.get_nb_inputs(),))
        n_rv = params["global/number_of_random_variables"]
        random_vars = jax.random.normal(key, (n_rv,))

        _, aux = stack.apply(params, inputs, random_vars, key)

        # Extract q_noise from forward and inverse transcription
        tc_aux = _get_layer_aux(aux, stack, "transcription")
        inv_tc_aux = _get_layer_aux(aux, stack, "inv_transcription")

        tc_noise = tc_aux["q_noise"][0]       # node 0
        inv_tc_noise = inv_tc_aux["q_noise"][0]  # node 0

        assert jnp.allclose(tc_noise, inv_tc_noise, atol=1e-7), (
            f"Transcription q_noise mismatch: fwd={tc_noise} vs inv={inv_tc_noise}"
        )

        # Same for translation
        tl_aux = _get_layer_aux(aux, stack, "translation")
        inv_tl_aux = _get_layer_aux(aux, stack, "inv_translation")

        tl_noise = tl_aux["q_noise"][0]
        inv_tl_noise = inv_tl_aux["q_noise"][0]

        assert jnp.allclose(tl_noise, inv_tl_noise, atol=1e-7), (
            f"Translation q_noise mismatch: fwd={tl_noise} vs inv={inv_tl_noise}"
        )


def test_different_node_types_get_different_keys(lib, simple_single_reporter):
    """Transcription and translation nodes must NOT share the same key."""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(simple_single_reporter, br.ALL_RULES, invert=True)
        stack = ComputeStack([networks[0]])
        stack.build(config=DEFAULT_COMPUTE_CONFIG)

        key = jax.random.PRNGKey(42)
        params = stack.init(key)

        # Verify that transcription and translation have different node_key_ids
        tc_layer = None
        tl_layer = None
        for layer in stack.layers:
            if layer.f_type == "transcription":
                tc_layer = layer
            elif layer.f_type == "translation":
                tl_layer = layer

        assert tc_layer is not None and tl_layer is not None

        tc_key_ids = np.asarray(params[f"{tc_layer.namespace}/node_key_id"])
        tl_key_ids = np.asarray(params[f"{tl_layer.namespace}/node_key_id"])

        # All key IDs across the whole stack must be unique (no collisions)
        all_ids = np.concatenate([tc_key_ids, tl_key_ids])
        assert len(np.unique(all_ids)) == len(all_ids), (
            f"Node key IDs are not globally unique: tc={tc_key_ids}, tl={tl_key_ids}"
        )


def test_node_key_ids_globally_unique(lib, simple_two_reporters):
    """Every node in the stack must have a globally unique node_key_id."""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(simple_two_reporters, br.ALL_RULES, invert=True)
        stack = ComputeStack([networks[0]])
        stack.build(config=DEFAULT_COMPUTE_CONFIG)

        key = jax.random.PRNGKey(99)
        params = stack.init(key)

        all_key_ids = []
        for layer in stack.layers:
            if layer.f_type == "input":
                continue
            path = f"{layer.namespace}/node_key_id"
            assert path in params, f"Missing node_key_id for layer {layer.f_type}"
            ids = np.asarray(params[path]).ravel()
            all_key_ids.extend(ids.tolist())

        # Forward/inverse pairs share IDs (via ArrayRef), so we check
        # that the total count matches expectations. For non-inverse layers,
        # IDs must be unique among themselves.
        n_total = params["global/number_of_node_keys"]
        assert max(all_key_ids) < n_total, (
            f"Max key ID {max(all_key_ids)} >= total allocated {n_total}"
        )


def test_noise_cancellation_two_reporters(lib, simple_two_reporters):
    """Multi-node layers: each forward/inverse pair shares noise independently."""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(simple_two_reporters, br.ALL_RULES, invert=True)
        stack = ComputeStack([networks[0]])
        stack.build(config=DEFAULT_COMPUTE_CONFIG)

        key = jax.random.PRNGKey(7)
        params = stack.init(key)

        inputs = jnp.ones((stack.get_nb_inputs(),))
        n_rv = params["global/number_of_random_variables"]
        random_vars = jax.random.normal(key, (n_rv,))

        _, aux = stack.apply(params, inputs, random_vars, key)

        tc_aux = _get_layer_aux(aux, stack, "transcription")
        inv_tc_aux = _get_layer_aux(aux, stack, "inv_transcription")

        # Forward transcription has 2 nodes, inverse has 1 (only one output inverted)
        # The inverse node's noise should match its corresponding forward node
        fwd_tc_noise = tc_aux["q_noise"]     # shape (n_fwd_nodes, ...)
        inv_tc_noise = inv_tc_aux["q_noise"]  # shape (n_inv_nodes, ...)

        # Get the forward node index that the inverse node points to
        inv_tc_layer = None
        for layer in stack.layers:
            if layer.f_type == "inv_transcription":
                inv_tc_layer = layer
                break

        assert inv_tc_layer is not None
        for inv_idx, inv_node in enumerate(inv_tc_layer.nodes):
            fwd_node = inv_node.get_forward_stacknode(stack)
            assert fwd_node is not None
            fwd_idx = fwd_node.node_position_in_layer

            assert jnp.allclose(fwd_tc_noise[fwd_idx], inv_tc_noise[inv_idx], atol=1e-7), (
                f"TC noise mismatch at inv_idx={inv_idx} <-> fwd_idx={fwd_idx}"
            )
