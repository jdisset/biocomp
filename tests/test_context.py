# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Tests for the context embedding system."""

import jax
import jax.numpy as jnp
import numpy as np
from biocomp.context import (
    CONTEXT_EMBEDDINGS,
    context_codebook_kl_inputs,
    disable_context_variational,
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
    # fallback happens when a value is absent from THIS model's codebook -- i.e. eval/warm-start
    # against a fixed context_values that doesn't list it. (A value present in the *training*
    # networks gets its own row by design; that's not the fallback path.)
    params = ParameterTree()
    networks = _make_networks(["UNKNOWN_CELL_LINE"])
    key = jax.random.PRNGKey(4)
    ce = CONTEXT_EMBEDDINGS[0]
    init_context_params(params, networks, key, context_values={ce.name: [ce.default_value]})

    indices = params[_indices_path(ce.name)]
    assert int(indices[0]) == 0  # unknown -> default row


# ── union_context_values (warm-start grow) ────────────────────────────────────


def test_union_context_values_retains_source():
    from biocomp.context import union_context_values

    ce = CONTEXT_EMBEDDINGS[0]
    src = {ce.name: [ce.default_value, "CHO"]}  # what a backbone knows
    train = {ce.name: [ce.default_value, "ARPE19"]}  # embed trains on ARPE19 only
    out = union_context_values(src, train)[ce.name]
    assert out == [ce.default_value, "ARPE19", "CHO"]  # default first, sorted rest, CHO kept


def test_union_context_values_three_way_and_empty():
    from biocomp.context import union_context_values

    ce = CONTEXT_EMBEDDINGS[0]
    out = union_context_values(
        {ce.name: [ce.default_value, "CHO"]},
        {ce.name: ["ARPE19"]},
        {ce.name: ["CHO", "iPS"]},  # overlapping -> deduped
    )[ce.name]
    assert out == [ce.default_value, "ARPE19", "CHO", "iPS"]
    assert union_context_values()[ce.name] == [ce.default_value]  # no inputs -> default only


def test_grow_codebook_retains_source_row_by_value():
    # regression for the dropped-row bug: warm-starting on line B must KEEP line A's row.
    from biocomp.context import (
        grow_context_codebook,
        union_context_values,
        _codebook_logstdevs_path,
    )

    ce = CONTEXT_EMBEDDINGS[0]
    mpath, lpath = _codebook_means_path(ce.name), _codebook_logstdevs_path(ce.name)

    src = ParameterTree()  # a backbone that knows [HEK, CHO]
    src_means = jnp.array([[0.0, 0.0], [0.5, 0.13]])  # HEK=row0, CHO=row1
    src.at(mpath, src_means, tags=["shared"], overwrite=True)
    src.at(lpath, jnp.array([[-3.0, -3.0], [-2.0, -1.0]]), tags=["shared"], overwrite=True)
    src_values = {ce.name: [ce.default_value, "CHO"]}

    # embed-only trains on ARPE19; the grow target is the UNION, so CHO survives
    target = union_context_values(src_values, {ce.name: [ce.default_value, "ARPE19"]})
    tgt = target[ce.name]

    params = ParameterTree()  # fresh smaller codebook, regrown to target
    params.at(mpath, jnp.zeros((1, 2)), tags=["shared"], overwrite=True)
    params.at(lpath, jnp.full((1, 2), -3.0), tags=["shared"], overwrite=True)
    grow_context_codebook(params, src, src_values, target, jax.random.PRNGKey(0))

    gm = params[mpath]
    assert gm.shape == (len(tgt), 2)
    np.testing.assert_array_equal(gm[tgt.index(ce.default_value)], src_means[0])  # HEK retained
    np.testing.assert_array_equal(gm[tgt.index("CHO")], src_means[1])  # CHO retained BY VALUE
    arpe = gm[tgt.index("ARPE19")]  # new line -> fresh init, not a copy of any source row
    assert not np.allclose(arpe, src_means[0]) and not np.allclose(arpe, src_means[1])


# ── make_codebook_freeze_hook (embed-only row freezing) ───────────────────────


def _codebook(n_rows: int, dim: int, active_row: int, n_networks: int, replicates: int = 0):
    """A ParameterTree with a multi-row codebook whose indices reference only one row.
    replicates>0 adds a leading replicate axis (as a real training tree has)."""
    from biocomp.context import _codebook_logstdevs_path

    ce = CONTEXT_EMBEDDINGS[0]
    means = jnp.arange(n_rows * dim, dtype=float).reshape(n_rows, dim)
    logstdevs = jnp.full((n_rows, dim), -3.0)
    idx = jnp.full((n_networks,), active_row)
    if replicates:
        means = jnp.broadcast_to(means, (replicates, *means.shape))
        logstdevs = jnp.broadcast_to(logstdevs, (replicates, *logstdevs.shape))
        idx = jnp.broadcast_to(idx, (replicates, n_networks))
    p = ParameterTree()
    p.at(_codebook_means_path(ce.name), means, tags=["shared"], overwrite=True)
    p.at(_codebook_logstdevs_path(ce.name), logstdevs, tags=["shared"], overwrite=True)
    p.at(_indices_path(ce.name), idx, tags=["local"], overwrite=True)
    return p, ce


def test_codebook_freeze_hook_pins_inactive_rows():
    from biocomp.context import make_codebook_freeze_hook, _codebook_logstdevs_path

    p, ce = _codebook(n_rows=3, dim=2, active_row=2, n_networks=4)
    mpath, lpath = _codebook_means_path(ce.name), _codebook_logstdevs_path(ce.name)
    means0, log0 = p[mpath].copy(), p[lpath].copy()

    hook = make_codebook_freeze_hook(p)
    assert hook is not None
    p[mpath] = p[mpath] + 1.0  # simulate an optimizer step moving EVERY row
    p[lpath] = p[lpath] + 1.0
    p = hook(p)

    np.testing.assert_array_equal(p[mpath][2], means0[2] + 1.0)  # active row keeps its move
    np.testing.assert_array_equal(p[mpath][:2], means0[:2])  # inactive rows pinned
    np.testing.assert_array_equal(p[lpath][2], log0[2] + 1.0)
    np.testing.assert_array_equal(p[lpath][:2], log0[:2])


def test_codebook_freeze_hook_strips_replicate_axis():
    # buffer/mask captured from a replicated tree must apply to a single replicate's leaf
    from biocomp.context import make_codebook_freeze_hook

    p, ce = _codebook(n_rows=3, dim=2, active_row=1, n_networks=4, replicates=3)
    hook = make_codebook_freeze_hook(p)
    mpath = _codebook_means_path(ce.name)
    warm = p[mpath][0]

    single, _ = _codebook(n_rows=3, dim=2, active_row=1, n_networks=4)
    single[mpath] = single[mpath] + 5.0
    single = hook(single)

    np.testing.assert_array_equal(single[mpath][1], warm[1] + 5.0)  # active kept
    np.testing.assert_array_equal(single[mpath][0], warm[0])  # pinned
    np.testing.assert_array_equal(single[mpath][2], warm[2])


def test_codebook_freeze_hook_multiple_active_rows():
    # a partial spanning two lines: every referenced row trains, the rest freeze
    from biocomp.context import make_codebook_freeze_hook

    p, ce = _codebook(n_rows=4, dim=2, active_row=1, n_networks=4)
    mpath = _codebook_means_path(ce.name)
    p[_indices_path(ce.name)] = jnp.array([1, 1, 3, 3])  # rows 1 and 3 referenced
    means0 = p[mpath].copy()

    hook = make_codebook_freeze_hook(p)
    p[mpath] = p[mpath] + 2.0
    p = hook(p)

    for r in (1, 3):
        np.testing.assert_array_equal(p[mpath][r], means0[r] + 2.0)  # active rows move
    for r in (0, 2):
        np.testing.assert_array_equal(p[mpath][r], means0[r])  # inactive rows pinned


def test_codebook_freeze_hook_idempotent():
    from biocomp.context import make_codebook_freeze_hook

    p, ce = _codebook(n_rows=3, dim=2, active_row=2, n_networks=4)
    mpath = _codebook_means_path(ce.name)
    hook = make_codebook_freeze_hook(p)
    p[mpath] = p[mpath] + 1.0
    once = hook(p)[mpath]
    twice = hook(hook(p))[mpath]
    np.testing.assert_array_equal(once, twice)


def test_codebook_freeze_hook_none_without_context(monkeypatch):
    from biocomp import context as ctx

    monkeypatch.setattr(ctx, "CONTEXT_EMBEDDINGS", [])
    assert ctx.make_codebook_freeze_hook(ParameterTree()) is None


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


# ── context_codebook_kl_inputs ────────────────────────────────────────────────


def test_context_codebook_kl_inputs_counts_and_shapes():
    params = ParameterTree()
    nets = _make_networks(["HEK293FT", "HEK293FT", "HEK293FT", "ARPE19"])
    init_context_params(params, nets, jax.random.PRNGKey(0))

    means, logstdevs, counts, total = context_codebook_kl_inputs(params)
    ce = CONTEXT_EMBEDDINGS[0]
    rows, dim = 2, ce.embedding_dim  # HEK293FT (row 0) + ARPE19 (row 1)
    assert means.shape == (rows * dim,)
    assert logstdevs.shape == (rows * dim,)
    # 3 HEK networks -> row 0, 1 ARPE19 -> row 1; each count repeated across `dim` entries
    np.testing.assert_array_equal(counts, jnp.array([3.0] * dim + [1.0] * dim))
    assert float(total) == 4.0


def test_context_codebook_kl_inputs_none_without_context():
    assert context_codebook_kl_inputs(ParameterTree()) is None


# ── disable_context_variational ───────────────────────────────────────────────


def test_disable_context_variational_makes_resolution_deterministic():
    params = ParameterTree()
    init_context_params(params, _make_networks(["HEK293FT"]), jax.random.PRNGKey(0))
    ce = CONTEXT_EMBEDDINGS[0]
    # σ=1 so sampling would otherwise jitter the resolved vector across keys
    params[_codebook_logstdevs_path(ce.name)] = jnp.zeros_like(
        params[_codebook_logstdevs_path(ce.name)]
    )
    noisy_a = resolve_context_vector(params, 0, jax.random.PRNGKey(1))
    noisy_b = resolve_context_vector(params, 0, jax.random.PRNGKey(2))
    assert not jnp.allclose(noisy_a, noisy_b)  # sampling differs by key by default

    disable_context_variational(params)
    assert float(jnp.max(params[_codebook_logstdevs_path(ce.name)])) <= -100.0
    a = resolve_context_vector(params, 0, jax.random.PRNGKey(1))
    b = resolve_context_vector(params, 0, jax.random.PRNGKey(2))
    assert jnp.allclose(a, b, atol=1e-3)  # pinned to the mean -> key-independent
