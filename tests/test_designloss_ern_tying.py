"""Tests for ERN TU tying penalty in design loss functions."""

import pytest
import jax
import jax.numpy as jnp
from jax import grad

from biocomp.parameters import ParameterTree
from biocomp.designloss import ern_tu_tying_penalty


def make_ern_params(
    n_nodes: int = 4,
    n_networks: int = 2,
    n_tus: int = 6,
    namespace: str = "local/5/sequestron_ERN_5p",
    neg_tu_indices: jnp.ndarray | None = None,
    pos_tu_indices: jnp.ndarray | None = None,
) -> tuple[ParameterTree, list[str]]:
    """Create mock params for ERN TU tying tests."""
    params = ParameterTree()

    if neg_tu_indices is None:
        neg_tu_indices = jnp.array([[i % n_tus] for i in range(n_nodes)], dtype=jnp.int32)
    if pos_tu_indices is None:
        pos_tu_indices = jnp.array([[(i + 1) % n_tus] for i in range(n_nodes)], dtype=jnp.int32)

    # shape: (n_nodes, 2, max_tus) where 2 = [neg, pos] inputs
    input_tu_indices = jnp.stack([neg_tu_indices, pos_tu_indices], axis=1)
    network_ids = jnp.array([i % n_networks for i in range(n_nodes)], dtype=jnp.int32)

    params.at(f"{namespace}/input_tu_indices", input_tu_indices)
    params.at(f"{namespace}/node_network_ids", network_ids)

    return params, [namespace]


def test_tying_penalty_zero_when_pos_enabled():
    """When pos TU is enabled (high log_alpha), penalty should be ~0."""
    params, namespaces = make_ern_params(n_nodes=2, n_networks=1, n_tus=4)
    # high log_alpha = high enabled prob for both pos and neg
    tu_log_alpha = jnp.ones((1, 4)) * 5.0

    penalty = ern_tu_tying_penalty(params, namespaces, tu_log_alpha)

    # pos is likely enabled -> sigmoid(-5) ~ 0 -> no penalty
    assert float(penalty) < 0.01, f"High pos log_alpha should give ~0 penalty, got {penalty}"


def test_tying_penalty_zero_when_both_disabled():
    """When both pos and neg are disabled, no penalty (neg <= pos)."""
    params, namespaces = make_ern_params(n_nodes=2, n_networks=1, n_tus=4)
    # both low log_alpha
    tu_log_alpha = jnp.ones((1, 4)) * (-5.0)

    penalty = ern_tu_tying_penalty(params, namespaces, tu_log_alpha)

    assert float(penalty) == 0.0, "Both disabled -> neg_la == pos_la -> relu(0) = 0"


def test_tying_penalty_positive_when_pos_disabled_neg_enabled():
    """CORE TEST: pos disabled but neg still enabled should give positive penalty."""
    params, namespaces = make_ern_params(n_nodes=1, n_networks=1, n_tus=2)
    # neg (TU 0) is enabled, pos (TU 1) is disabled
    tu_log_alpha = jnp.array([[5.0, -5.0]])  # [neg enabled, pos disabled]

    penalty = ern_tu_tying_penalty(params, namespaces, tu_log_alpha)

    # pos_disabled_prob = sigmoid(5) ~ 0.99
    # excess = relu(5 - (-5)) = 10
    # penalty ~ 0.99 * 10 ~ 9.9
    assert float(penalty) > 5.0, f"pos disabled + neg enabled should give high penalty, got {penalty}"


def test_tying_penalty_asymmetric():
    """Penalty is one-way: high pos + low neg is fine, but low pos + high neg is penalized."""
    params, namespaces = make_ern_params(n_nodes=1, n_networks=1, n_tus=2)

    # case 1: pos enabled, neg disabled (fine)
    tu_la_fine = jnp.array([[-5.0, 5.0]])  # [neg low, pos high]
    penalty_fine = ern_tu_tying_penalty(params, namespaces, tu_la_fine)

    # case 2: pos disabled, neg enabled (penalized)
    tu_la_bad = jnp.array([[5.0, -5.0]])  # [neg high, pos low]
    penalty_bad = ern_tu_tying_penalty(params, namespaces, tu_la_bad)

    assert float(penalty_fine) < 0.1, f"pos high + neg low should be fine, got {penalty_fine}"
    assert float(penalty_bad) > 5.0, f"pos low + neg high should be penalized, got {penalty_bad}"
    assert float(penalty_bad) > 50 * float(penalty_fine), "asymmetry: bad >> fine"


def test_gradient_flows_to_tu_log_alpha():
    """Gradients should flow from tying penalty to tu_log_alpha."""
    params, namespaces = make_ern_params(n_nodes=1, n_networks=1, n_tus=2)

    def loss_fn(tu_log_alpha):
        return ern_tu_tying_penalty(params, namespaces, tu_log_alpha)

    tu_log_alpha = jnp.array([[0.0, -3.0]])  # neg=0, pos=-3 (pos somewhat disabled)
    grads = grad(loss_fn)(tu_log_alpha)

    assert jnp.any(jnp.abs(grads) > 1e-4), f"Should have non-zero gradient, got {grads}"
    assert jnp.all(jnp.isfinite(grads)), "Gradients should be finite"


def test_gradient_pushes_neg_down():
    """Gradient should push neg log_alpha DOWN when pos is disabled."""
    params, namespaces = make_ern_params(n_nodes=1, n_networks=1, n_tus=2)

    def loss_fn(tu_log_alpha):
        return ern_tu_tying_penalty(params, namespaces, tu_log_alpha)

    tu_log_alpha = jnp.array([[2.0, -3.0]])  # neg high, pos low
    grads = grad(loss_fn)(tu_log_alpha)

    # gradient for neg (idx 0) should be positive (to push down via SGD)
    assert grads[0, 0] > 0, f"Gradient for neg TU should be positive, got {grads[0, 0]}"


def test_empty_namespaces_returns_zero():
    """Empty namespace list should return zero penalty."""
    params = ParameterTree()
    tu_log_alpha = jnp.zeros((2, 6))

    penalty = ern_tu_tying_penalty(params, [], tu_log_alpha)

    assert float(penalty) == 0.0


def test_missing_params_skipped():
    """Namespaces without required params should be skipped."""
    params = ParameterTree()
    # deliberately not adding input_tu_indices or node_network_ids

    tu_log_alpha = jnp.zeros((2, 6))
    penalty = ern_tu_tying_penalty(params, ["local/5/sequestron_ERN_5p"], tu_log_alpha)

    assert float(penalty) == 0.0


def test_3d_tu_log_alpha():
    """3D tu_log_alpha (multi-target) should work with 4D input_tu_indices."""
    n_targets, n_nodes, n_networks, n_tus = 3, 2, 1, 4
    ns = "local/5/sequestron_ERN_5p"
    params = ParameterTree()

    # 4D: (n_targets, n_nodes, n_inputs, max_tus)
    input_tu_indices = jnp.broadcast_to(
        jnp.array([[[0], [1]], [[2], [3]]], dtype=jnp.int32),
        (n_targets, n_nodes, 2, 1)
    )
    # 2D: (n_targets, n_nodes)
    network_ids = jnp.broadcast_to(
        jnp.array([0, 0], dtype=jnp.int32),
        (n_targets, n_nodes)
    )
    params.at(f"{ns}/input_tu_indices", input_tu_indices)
    params.at(f"{ns}/node_network_ids", network_ids)

    tu_log_alpha_3d = jnp.ones((n_targets, n_networks, n_tus)) * 5.0

    penalty = ern_tu_tying_penalty(params, [ns], tu_log_alpha_3d)

    assert jnp.isfinite(penalty), f"Penalty should be finite, got {penalty}"


def test_invalid_tu_log_alpha_rejects_1d():
    """1D tu_log_alpha should raise."""
    params, namespaces = make_ern_params()

    with pytest.raises(AssertionError, match="2D or 3D"):
        ern_tu_tying_penalty(params, namespaces, jnp.zeros((6,)))


def test_always_enabled_tu_handled():
    """TU index -1 (always enabled) should be handled correctly."""
    params = ParameterTree()
    ns = "local/5/sequestron_ERN_5p"
    # neg has -1 (always enabled), pos has real TU
    input_tu_indices = jnp.array([[[-1], [0]]], dtype=jnp.int32)  # (1, 2, 1)
    params.at(f"{ns}/input_tu_indices", input_tu_indices)
    params.at(f"{ns}/node_network_ids", jnp.array([0], dtype=jnp.int32))

    tu_log_alpha = jnp.array([[-5.0]])  # pos disabled

    # neg is always enabled (log_alpha = 10.0), pos is disabled
    # should give penalty
    penalty = ern_tu_tying_penalty(params, [ns], tu_log_alpha)

    assert float(penalty) > 1.0, "neg always-enabled + pos disabled should penalize"


def test_jit_compilation():
    """Verify function compiles and runs under JIT."""
    params, namespaces = make_ern_params(n_nodes=2, n_networks=1, n_tus=4)

    @jax.jit
    def compute_penalty(tu_log_alpha):
        return ern_tu_tying_penalty(params, namespaces, tu_log_alpha)

    tu_log_alpha = jnp.array([[5.0, -5.0, 0.0, 0.0]])
    penalty1 = compute_penalty(tu_log_alpha)
    penalty2 = compute_penalty(tu_log_alpha)

    assert jnp.allclose(penalty1, penalty2)
    assert jnp.isfinite(penalty1)


def test_multiple_ern_layers():
    """Test with multiple ERN layer namespaces."""
    params = ParameterTree()

    for i, ns in enumerate(["local/3/sequestron_ERN_5p", "local/7/sequestron_ERN_5p"]):
        input_tu_indices = jnp.array([[[i * 2], [i * 2 + 1]]], dtype=jnp.int32)
        params.at(f"{ns}/input_tu_indices", input_tu_indices)
        params.at(f"{ns}/node_network_ids", jnp.array([0], dtype=jnp.int32))

    namespaces = ["local/3/sequestron_ERN_5p", "local/7/sequestron_ERN_5p"]
    # set up: neg enabled, pos disabled for both layers
    tu_log_alpha = jnp.array([[5.0, -5.0, 5.0, -5.0]])

    penalty = ern_tu_tying_penalty(params, namespaces, tu_log_alpha)

    # should get penalty from both layers
    single_penalty = ern_tu_tying_penalty(
        params, ["local/3/sequestron_ERN_5p"], tu_log_alpha
    )
    assert float(penalty) > 1.5 * float(single_penalty), "Multiple layers should sum penalties"
