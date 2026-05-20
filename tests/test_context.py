# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Tests for the context embedding system."""

import jax
import jax.numpy as jnp
import numpy as np
from biocomp.context import (
    CONTEXT_EMBEDDINGS,
    init_context_params,
    resolve_context_vector,
    total_context_dim,
    variational_codebook_lookup,
    _codebook_means_path,
    _codebook_logstdevs_path,
    _indices_path,
)
from biocomp.parameters import ParameterTree
from biocomp.network import Network


# ── variational_codebook_lookup ───────────────────────────────────────────────


def test_variational_codebook_lookup_shape():
    mean = jnp.array([0.1, 0.2])
    logstdev = jnp.array([-3.0, -3.0])
    key = jax.random.PRNGKey(42)
    val, aux = variational_codebook_lookup(mean, logstdev, key)
    assert val.shape == (2,)
    assert aux["mean"].shape == (2,)
    assert aux["logstdev"].shape == (2,)
    assert aux["noise"].shape == (2,)


def test_variational_codebook_lookup_disable():
    mean = jnp.array([0.5, -0.5])
    logstdev = jnp.array([0.0, 0.0])
    key = jax.random.PRNGKey(0)
    val, aux = variational_codebook_lookup(mean, logstdev, key, disable_variational=True)
    np.testing.assert_array_equal(val, mean)
    np.testing.assert_array_equal(aux["noise"], jnp.zeros(2))


def test_variational_codebook_lookup_noise():
    mean = jnp.zeros(4)
    logstdev = jnp.zeros(4)  # exp(0)=1, so noise ~ N(0,1)
    key = jax.random.PRNGKey(123)
    val, aux = variational_codebook_lookup(mean, logstdev, key)
    # With noise enabled, output should differ from mean
    assert not jnp.allclose(val, mean)


def test_variational_codebook_lookup_clips_logstdev():
    mean = jnp.array([0.0])
    logstdev = jnp.array([100.0])  # should be clipped to 5.0
    key = jax.random.PRNGKey(0)
    val, aux = variational_codebook_lookup(mean, logstdev, key)
    assert float(aux["logstdev"][0]) == 5.0


# ── total_context_dim ─────────────────────────────────────────────────────────


def test_total_context_dim():
    expected = sum(ce.embedding_dim for ce in CONTEXT_EMBEDDINGS)
    assert total_context_dim() == expected
    assert total_context_dim() == 2  # default cell_type has dim=2


# ── init_context_params ───────────────────────────────────────────────────────


def _make_networks(cell_types: list[str]) -> list[Network]:
    return [Network(name=f"net_{i}", metadata={"cell_type": ct}) for i, ct in enumerate(cell_types)]


def test_init_context_params_creates_paths():
    params = ParameterTree()
    networks = _make_networks(["HEK293FT", "HEK293FT"])
    key = jax.random.PRNGKey(0)
    init_context_params(params, networks, key)

    ce = CONTEXT_EMBEDDINGS[0]
    assert _codebook_means_path(ce.name) in params
    assert _codebook_logstdevs_path(ce.name) in params
    assert _indices_path(ce.name) in params


def test_init_context_params_correct_shapes():
    params = ParameterTree()
    networks = _make_networks(["HEK293FT", "HEK293FT", "HEK293FT"])
    key = jax.random.PRNGKey(1)
    init_context_params(params, networks, key)

    ce = CONTEXT_EMBEDDINGS[0]
    means = params[_codebook_means_path(ce.name)]
    logstdevs = params[_codebook_logstdevs_path(ce.name)]
    indices = params[_indices_path(ce.name)]

    assert means.shape == (len(ce.available_values), ce.embedding_dim)
    assert logstdevs.shape == (len(ce.available_values), ce.embedding_dim)
    assert indices.shape == (3,)


def test_init_context_params_indices_mapping():
    params = ParameterTree()
    networks = _make_networks(["HEK293FT", "HEK293FT"])
    key = jax.random.PRNGKey(2)
    init_context_params(params, networks, key)

    ce = CONTEXT_EMBEDDINGS[0]
    indices = params[_indices_path(ce.name)]
    # Both networks should map to index 0 (HEK293FT is the only value)
    np.testing.assert_array_equal(indices, jnp.array([0, 0]))


def test_init_context_params_idempotent():
    params = ParameterTree()
    networks = _make_networks(["HEK293FT"])
    key = jax.random.PRNGKey(3)
    init_context_params(params, networks, key)

    ce = CONTEXT_EMBEDDINGS[0]
    original_means = params[_codebook_means_path(ce.name)].copy()

    # Call again -- should not overwrite
    init_context_params(params, networks, jax.random.PRNGKey(999))
    np.testing.assert_array_equal(params[_codebook_means_path(ce.name)], original_means)


def test_init_context_params_unknown_value_falls_back():
    params = ParameterTree()
    networks = _make_networks(["UNKNOWN_CELL_LINE"])
    key = jax.random.PRNGKey(4)
    init_context_params(params, networks, key)

    ce = CONTEXT_EMBEDDINGS[0]
    indices = params[_indices_path(ce.name)]
    # Should fall back to default (index 0)
    assert int(indices[0]) == 0


# ── resolve_context_vector ────────────────────────────────────────────────────


def test_resolve_context_vector_shape():
    params = ParameterTree()
    networks = _make_networks(["HEK293FT", "HEK293FT"])
    key = jax.random.PRNGKey(5)
    init_context_params(params, networks, key)

    vec = resolve_context_vector(params, jnp.int32(0), jax.random.PRNGKey(10))
    assert vec is not None
    assert vec.shape == (total_context_dim(),)


def test_resolve_returns_none_when_no_params():
    params = ParameterTree()
    # Don't call init_context_params
    result = resolve_context_vector(params, jnp.int32(0), jax.random.PRNGKey(0))
    assert result is None


def test_resolve_different_keys_give_different_samples():
    params = ParameterTree()
    networks = _make_networks(["HEK293FT"])
    init_context_params(params, networks, jax.random.PRNGKey(6))

    v1 = resolve_context_vector(params, jnp.int32(0), jax.random.PRNGKey(10))
    v2 = resolve_context_vector(params, jnp.int32(0), jax.random.PRNGKey(20))
    assert v1 is not None and v2 is not None
    # Different keys should give different noise samples
    assert not jnp.allclose(v1, v2)


# ── Recipe cell_type ──────────────────────────────────────────────────────────


def test_recipe_cell_type_default():
    from biocomp.recipe import Recipe

    r = Recipe()
    assert r.cell_type == "HEK293FT"


def test_recipe_cell_type_custom():
    from biocomp.recipe import Recipe

    r = Recipe(cell_type="CHO")
    assert r.cell_type == "CHO"
