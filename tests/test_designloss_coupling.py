"""Tests for ratio-mask coupling penalty in design loss functions.

Key success criteria:
1. Coupling penalty is ZERO when all ratios are above threshold
2. Coupling penalty is POSITIVE when ratios are below threshold
3. MAX normalization: threshold meaning is independent of TU count
4. Gradient flows through coupling penalty to tu_log_alpha
5. Runtime checks (via checkify) catch invalid inputs

Uses jax.debug.checkify for runtime assertions per codebase philosophy.
"""

import pytest
import jax
import jax.numpy as jnp
from jax.experimental import checkify
from jax import grad, vmap
import numpy as np

from biocomp.parameters import ParameterTree
from biocomp.designloss import (
    ratio_mask_coupling_penalty,
    _ratio_mask_coupling_single_target,
)


# ---------------------------------------------------------------------------
# Fixtures for mock parameter trees
# ---------------------------------------------------------------------------


def make_mock_params(
    n_nodes: int = 4,
    n_outputs: int = 3,
    n_networks: int = 2,
    n_tus: int = 6,
    ratio_values: jnp.ndarray | None = None,
    namespace: str = "local/layer_1",
    n_targets: int = 1,
) -> tuple[ParameterTree, list[str]]:
    """Create a mock parameter tree with ratios, tu_indices, and network_ids.

    Args:
        n_targets: If > 1, creates 3D arrays (n_targets, n_nodes, n_outputs). If 1, creates 2D.
    """
    params = ParameterTree()
    ns = namespace

    if ratio_values is None:
        if n_targets == 1:
            ratio_values = jnp.ones((n_nodes, n_outputs))
        else:
            ratio_values = jnp.ones((n_targets, n_nodes, n_outputs))

    assert ratio_values.ndim in (2, 3), f"ratio_values must be 2D or 3D, got {ratio_values.ndim}D"
    if n_targets == 1:
        assert ratio_values.ndim == 2, f"n_targets=1 requires 2D ratios, got {ratio_values.ndim}D"
    else:
        assert ratio_values.ndim == 3, f"n_targets={n_targets} requires 3D ratios, got {ratio_values.ndim}D"
        assert ratio_values.shape[0] == n_targets, f"ratios n_targets mismatch: {ratio_values.shape[0]} vs {n_targets}"

    params.at(f"{ns}/ratios", ratio_values)

    tu_indices_2d = jnp.array([
        [(i + j) % n_tus for j in range(n_outputs)]
        for i in range(n_nodes)
    ], dtype=jnp.int32)

    network_ids_1d = jnp.array([i % n_networks for i in range(n_nodes)], dtype=jnp.int32)

    if n_targets == 1:
        params.at(f"{ns}/output_tu_indices", tu_indices_2d)
        params.at(f"{ns}/node_network_ids", network_ids_1d)
    else:
        params.at(f"{ns}/output_tu_indices", jnp.broadcast_to(tu_indices_2d, (n_targets, n_nodes, n_outputs)))
        params.at(f"{ns}/node_network_ids", jnp.broadcast_to(network_ids_1d, (n_targets, n_nodes)))

    return params, [f"{ns}/ratios"]


# ---------------------------------------------------------------------------
# Basic Functionality Tests
# ---------------------------------------------------------------------------


def test_coupling_penalty_zero_when_all_above_threshold():
    """Coupling penalty should be EXACTLY ZERO when all ratios >= threshold."""
    params, ratio_paths = make_mock_params(
        n_nodes=4, n_outputs=3, n_networks=2, n_tus=6,
        ratio_values=jnp.ones((4, 3))  # all 1.0 -> normalized = 1.0
    )
    tu_log_alpha = jnp.zeros((2, 6))  # (n_networks, n_tus), sigmoid(0)=0.5

    penalty = ratio_mask_coupling_penalty(
        params, ratio_paths, tu_log_alpha, min_ratio_threshold=0.005
    )

    assert float(penalty) == 0.0, (
        f"Penalty should be 0.0 when all ratios above threshold, got {penalty}"
    )


def test_coupling_penalty_zero_for_uniform_ratios():
    """With uniform ratios, MAX normalization gives all 1.0 -> zero penalty."""
    params, ratio_paths = make_mock_params(
        n_nodes=4, n_outputs=5, n_networks=2, n_tus=10,
        ratio_values=jnp.ones((4, 5)) * 0.5  # all 0.5 -> normalized = 1.0 (MAX norm)
    )
    tu_log_alpha = jnp.ones((2, 10)) * 3.0  # high log_alpha = high enabled prob

    penalty = ratio_mask_coupling_penalty(
        params, ratio_paths, tu_log_alpha, min_ratio_threshold=0.1
    )

    assert float(penalty) == 0.0, (
        f"With uniform ratios, MAX norm gives all 1.0, so penalty should be 0. Got {penalty}"
    )


def test_coupling_penalty_positive_when_below_threshold():
    """Coupling penalty should be > 0 when some ratios are below threshold."""
    # set up ratios where one output has ratio << others
    ratio_values = jnp.array([
        [1.0, 1.0, 0.001],  # 3rd output: 0.001/1.0 = 0.001 < 0.005
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ])
    params, ratio_paths = make_mock_params(
        n_nodes=4, n_outputs=3, n_networks=2, n_tus=6,
        ratio_values=ratio_values
    )
    # high log_alpha = high enabled probability -> high penalty for below-threshold TU
    tu_log_alpha = jnp.ones((2, 6)) * 5.0

    penalty = ratio_mask_coupling_penalty(
        params, ratio_paths, tu_log_alpha, min_ratio_threshold=0.005
    )

    assert float(penalty) > 0, (
        f"Penalty should be > 0 when ratio 0.001 < threshold 0.005, got {penalty}"
    )


def test_coupling_penalty_increases_with_tu_enabled_prob():
    """Higher tu_log_alpha (enabled prob) should give higher penalty for below-threshold ratios."""
    # one ratio is tiny (below threshold)
    ratio_values = jnp.array([
        [1.0, 0.0001, 1.0],  # 0.0001/1.0 = 0.0001 < 0.005
        [1.0, 1.0, 1.0],
    ])
    params, ratio_paths = make_mock_params(
        n_nodes=2, n_outputs=3, n_networks=1, n_tus=4,
        ratio_values=ratio_values
    )

    # low log_alpha -> low enabled prob -> low penalty
    tu_log_alpha_low = jnp.ones((1, 4)) * (-5.0)  # sigmoid(-5) ≈ 0.007
    penalty_low = ratio_mask_coupling_penalty(
        params, ratio_paths, tu_log_alpha_low, min_ratio_threshold=0.005
    )

    # high log_alpha -> high enabled prob -> high penalty
    tu_log_alpha_high = jnp.ones((1, 4)) * 5.0  # sigmoid(5) ≈ 0.993
    penalty_high = ratio_mask_coupling_penalty(
        params, ratio_paths, tu_log_alpha_high, min_ratio_threshold=0.005
    )

    assert float(penalty_high) > float(penalty_low), (
        f"Higher tu_enabled_prob should give higher penalty. "
        f"low_alpha penalty: {penalty_low}, high_alpha penalty: {penalty_high}"
    )


# ---------------------------------------------------------------------------
# MAX Normalization Tests
# ---------------------------------------------------------------------------


def test_max_normalization_threshold_independent_of_tu_count():
    """Threshold meaning should be consistent regardless of how many TUs in aggregation.

    With MAX normalization:
    - threshold=0.005 means "ratio < 0.5% of largest ratio"
    - This should work the same for 2 TUs or 20 TUs
    """
    # case 1: 2 outputs, one tiny
    ratios_2 = jnp.array([[1.0, 0.004]])  # 0.004/1.0 = 0.004 < 0.005
    params_2, paths_2 = make_mock_params(
        n_nodes=1, n_outputs=2, n_networks=1, n_tus=2,
        ratio_values=ratios_2
    )
    tu_log_alpha_2 = jnp.ones((1, 2)) * 5.0
    penalty_2 = ratio_mask_coupling_penalty(
        params_2, paths_2, tu_log_alpha_2, min_ratio_threshold=0.005
    )

    # case 2: 10 outputs, one tiny
    ratios_10 = jnp.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.004]])
    params_10, paths_10 = make_mock_params(
        n_nodes=1, n_outputs=10, n_networks=1, n_tus=10,
        ratio_values=ratios_10
    )
    tu_log_alpha_10 = jnp.ones((1, 10)) * 5.0
    penalty_10 = ratio_mask_coupling_penalty(
        params_10, paths_10, tu_log_alpha_10, min_ratio_threshold=0.005
    )

    # both should be positive (below threshold)
    assert float(penalty_2) > 0, f"2-output case penalty should be > 0, got {penalty_2}"
    assert float(penalty_10) > 0, f"10-output case penalty should be > 0, got {penalty_10}"

    # with SUM normalization, 0.004/sum would differ wildly
    # with MAX normalization, both give 0.004/1.0 = 0.004 < 0.005
    # penalties should be similar (both trigger coupling for one below-threshold ratio)
    # note: not exactly equal due to different TU index mapping, but order of magnitude similar
    ratio = float(penalty_10) / float(penalty_2) if penalty_2 > 1e-10 else float('inf')
    assert 0.1 < ratio < 10, (
        f"MAX normalization should give similar penalties regardless of TU count. "
        f"2-output: {penalty_2}, 10-output: {penalty_10}, ratio: {ratio}"
    )


def test_max_normalization_exact_values():
    """Verify MAX normalization produces expected normalized values."""
    # ratios [1.0, 0.5, 0.1] -> max=1.0 -> normalized = [1.0, 0.5, 0.1]
    ratio_values = jnp.array([[1.0, 0.5, 0.1]])
    params, paths = make_mock_params(
        n_nodes=1, n_outputs=3, n_networks=1, n_tus=3,
        ratio_values=ratio_values
    )

    # with threshold=0.15, only 0.1 < 0.15 is below threshold
    tu_log_alpha = jnp.ones((1, 3)) * 5.0  # high enabled prob
    penalty_015 = ratio_mask_coupling_penalty(
        params, paths, tu_log_alpha, min_ratio_threshold=0.15
    )

    # with threshold=0.05, no ratio is below threshold (0.1 > 0.05)
    penalty_005 = ratio_mask_coupling_penalty(
        params, paths, tu_log_alpha, min_ratio_threshold=0.05
    )

    assert float(penalty_015) > 0, "0.1 < 0.15 should trigger penalty"
    assert float(penalty_005) == 0.0, "0.1 > 0.05 should not trigger penalty"


# ---------------------------------------------------------------------------
# Gradient Flow Tests
# ---------------------------------------------------------------------------


def test_gradient_flows_to_tu_log_alpha():
    """Gradients should flow from coupling penalty to tu_log_alpha."""
    ratio_values = jnp.array([[1.0, 0.001]])  # 0.001 < 0.005 threshold
    params, paths = make_mock_params(
        n_nodes=1, n_outputs=2, n_networks=1, n_tus=2,
        ratio_values=ratio_values
    )

    def loss_fn(tu_log_alpha):
        return ratio_mask_coupling_penalty(
            params, paths, tu_log_alpha, min_ratio_threshold=0.005
        )

    tu_log_alpha_init = jnp.zeros((1, 2))
    grads = grad(loss_fn)(tu_log_alpha_init)

    # gradients should be non-zero for the TU associated with the tiny ratio
    assert jnp.any(jnp.abs(grads) > 1e-8), (
        f"Gradients should be non-zero to push down tu_log_alpha, got {grads}"
    )
    assert jnp.all(jnp.isfinite(grads)), f"Gradients contain NaN/Inf: {grads}"


def test_gradient_pushes_down_tu_log_alpha():
    """Gradient direction should push tu_log_alpha DOWN (reduce enabled prob).

    When a ratio is below threshold, the coupling loss is:
    penalty = sigmoid(tu_log_alpha) * (threshold - normalized_ratio)

    d(penalty)/d(tu_log_alpha) = sigmoid'(tu_log_alpha) * (threshold - normalized_ratio)
                                = sigmoid(1-sigmoid) * positive_value > 0

    So gradient is POSITIVE, meaning we should SUBTRACT gradient to minimize loss,
    which pushes tu_log_alpha DOWN.
    """
    ratio_values = jnp.array([[1.0, 0.001]])
    params, paths = make_mock_params(
        n_nodes=1, n_outputs=2, n_networks=1, n_tus=2,
        ratio_values=ratio_values
    )

    def loss_fn(tu_log_alpha):
        return ratio_mask_coupling_penalty(
            params, paths, tu_log_alpha, min_ratio_threshold=0.005
        )

    tu_log_alpha_init = jnp.zeros((1, 2))
    grads = grad(loss_fn)(tu_log_alpha_init)

    # gradient should be positive for TU 1 (associated with tiny ratio at output slot 1)
    # TU index for node 0, output slot 1 is (0+1) % 2 = 1
    tu_idx_for_tiny_ratio = 1
    assert grads[0, tu_idx_for_tiny_ratio] > 0, (
        f"Gradient for below-threshold TU should be positive (so subtracting pushes down). "
        f"Got gradient: {grads[0, tu_idx_for_tiny_ratio]}"
    )


def test_no_gradient_for_above_threshold():
    """No gradient should flow for TUs whose ratios are above threshold."""
    ratio_values = jnp.ones((2, 3))  # all 1.0 -> all normalized to 1.0 -> all above threshold
    params, paths = make_mock_params(
        n_nodes=2, n_outputs=3, n_networks=1, n_tus=4,
        ratio_values=ratio_values
    )

    def loss_fn(tu_log_alpha):
        return ratio_mask_coupling_penalty(
            params, paths, tu_log_alpha, min_ratio_threshold=0.005
        )

    tu_log_alpha_init = jnp.ones((1, 4)) * 3.0
    grads = grad(loss_fn)(tu_log_alpha_init)

    assert jnp.allclose(grads, 0.0, atol=1e-8), (
        f"All ratios above threshold -> all gradients should be zero. Got {grads}"
    )


# ---------------------------------------------------------------------------
# 3D tu_log_alpha Tests (Multi-Target)
# ---------------------------------------------------------------------------


def test_3d_tu_log_alpha_shape_accepted():
    """3D tu_log_alpha requires 3D ratios - both must be consistent."""
    n_targets, n_networks, n_tus = 3, 2, 6
    n_nodes, n_outputs = 2, 3

    params, paths = make_mock_params(
        n_nodes=n_nodes, n_outputs=n_outputs, n_networks=n_networks, n_tus=n_tus,
        n_targets=n_targets,
    )

    tu_log_alpha_3d = jnp.zeros((n_targets, n_networks, n_tus))
    penalty = ratio_mask_coupling_penalty(params, paths, tu_log_alpha_3d, min_ratio_threshold=0.005)

    assert jnp.isfinite(penalty), f"Penalty should be finite, got {penalty}"
    assert float(penalty) == 0.0, "Uniform ratios should give zero penalty"


def test_3d_aggregates_over_targets():
    """Penalty should aggregate (sum) over all targets."""
    n_targets, n_networks, n_tus = 3, 1, 4
    n_nodes, n_outputs = 2, 3

    ratio_2d = jnp.array([[1.0, 0.001, 1.0], [1.0, 1.0, 1.0]])
    ratio_3d = jnp.broadcast_to(ratio_2d, (n_targets, n_nodes, n_outputs))

    params_2d, paths = make_mock_params(
        n_nodes=n_nodes, n_outputs=n_outputs, n_networks=n_networks, n_tus=n_tus,
        ratio_values=ratio_2d, n_targets=1,
    )
    params_3d, _ = make_mock_params(
        n_nodes=n_nodes, n_outputs=n_outputs, n_networks=n_networks, n_tus=n_tus,
        ratio_values=ratio_3d, n_targets=n_targets,
    )

    tu_log_alpha_2d = jnp.ones((n_networks, n_tus)) * 5.0
    penalty_2d = ratio_mask_coupling_penalty(params_2d, paths, tu_log_alpha_2d, min_ratio_threshold=0.005)

    tu_log_alpha_3d = jnp.ones((n_targets, n_networks, n_tus)) * 5.0
    penalty_3d = ratio_mask_coupling_penalty(params_3d, paths, tu_log_alpha_3d, min_ratio_threshold=0.005)

    ratio = float(penalty_3d) / float(penalty_2d) if penalty_2d > 1e-10 else float('inf')
    assert 2.5 < ratio < 3.5, f"3-target penalty should be ~3x single-target: {penalty_2d} vs {penalty_3d}"


def test_3d_ratios_uses_correct_target_slice():
    """CRITICAL: Each target must use its OWN ratios, not target 0's ratios.

    Bug caught: ratios[0] was used for all targets instead of ratios[target_idx].
    """
    n_targets, n_networks, n_tus = 3, 1, 4
    n_nodes, n_outputs = 2, 3
    ns = "local/layer_1"

    ratios_3d = jnp.array([
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        [[1.0, 0.001, 1.0], [1.0, 1.0, 1.0]],
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    ])

    tu_indices_2d = jnp.array([[(i + j) % n_tus for j in range(n_outputs)] for i in range(n_nodes)], dtype=jnp.int32)
    network_ids_1d = jnp.array([0] * n_nodes, dtype=jnp.int32)

    params = ParameterTree()
    params.at(f"{ns}/ratios", ratios_3d)
    params.at(f"{ns}/output_tu_indices", jnp.broadcast_to(tu_indices_2d, (n_targets, n_nodes, n_outputs)))
    params.at(f"{ns}/node_network_ids", jnp.broadcast_to(network_ids_1d, (n_targets, n_nodes)))

    tu_log_alpha_3d = jnp.ones((n_targets, n_networks, n_tus)) * 5.0
    penalty = ratio_mask_coupling_penalty(params, [f"{ns}/ratios"], tu_log_alpha_3d, min_ratio_threshold=0.005)

    assert float(penalty) > 0.001, f"Expected positive penalty from target 1, got {penalty}. Bug: using ratios[0] for all?"

    params_t1 = ParameterTree()
    params_t1.at(f"{ns}/ratios", ratios_3d[1])
    params_t1.at(f"{ns}/output_tu_indices", tu_indices_2d)
    params_t1.at(f"{ns}/node_network_ids", network_ids_1d)

    penalty_t1 = ratio_mask_coupling_penalty(params_t1, [f"{ns}/ratios"], tu_log_alpha_3d[1], min_ratio_threshold=0.005)

    assert float(penalty_t1) > 0.001, f"Target 1 should have positive penalty"
    assert abs(float(penalty) - float(penalty_t1)) < 0.001, f"Total {penalty} should ≈ target 1's {penalty_t1}"


# ---------------------------------------------------------------------------
# Edge Cases and Boundary Tests
# ---------------------------------------------------------------------------


def test_empty_ratio_paths():
    """Empty ratio_paths should return zero penalty."""
    params = ParameterTree()
    tu_log_alpha = jnp.zeros((2, 6))

    penalty = ratio_mask_coupling_penalty(
        params, [], tu_log_alpha, min_ratio_threshold=0.005
    )

    assert float(penalty) == 0.0, f"Empty ratio_paths should give zero penalty"


def test_missing_tu_indices_skipped():
    """Paths without tu_indices should be skipped gracefully."""
    params = ParameterTree()
    params.at("local/layer_1/ratios", jnp.ones((2, 3)))
    # deliberately NOT adding output_tu_indices or node_network_ids

    tu_log_alpha = jnp.zeros((2, 6))

    penalty = ratio_mask_coupling_penalty(
        params, ["local/layer_1/ratios"], tu_log_alpha, min_ratio_threshold=0.005
    )

    # should not crash, just return 0
    assert float(penalty) == 0.0, "Missing tu_indices should be skipped"


def test_all_zero_ratios():
    """All zero ratios should not cause division by zero."""
    ratio_values = jnp.zeros((2, 3))  # all zero
    params, paths = make_mock_params(
        n_nodes=2, n_outputs=3, n_networks=1, n_tus=4,
        ratio_values=ratio_values
    )
    tu_log_alpha = jnp.zeros((1, 4))

    penalty = ratio_mask_coupling_penalty(
        params, paths, tu_log_alpha, min_ratio_threshold=0.005
    )

    # max(zeros) = 0, but we have jnp.maximum(ratio_max, 1e-8) protection
    assert jnp.isfinite(penalty), f"Zero ratios should not cause NaN, got {penalty}"


def test_threshold_boundary_exact():
    """Test exact threshold boundary behavior."""
    # ratio exactly AT threshold should NOT trigger penalty (>=, not >)
    ratio_values = jnp.array([[1.0, 0.005]])  # 0.005/1.0 = 0.005 == threshold
    params, paths = make_mock_params(
        n_nodes=1, n_outputs=2, n_networks=1, n_tus=2,
        ratio_values=ratio_values
    )
    tu_log_alpha = jnp.ones((1, 2)) * 5.0

    penalty = ratio_mask_coupling_penalty(
        params, paths, tu_log_alpha, min_ratio_threshold=0.005
    )

    # relu(0.005 - 0.005) = relu(0) = 0 -> no penalty
    assert float(penalty) == 0.0, (
        f"Ratio exactly at threshold should give zero penalty, got {penalty}"
    )


# ---------------------------------------------------------------------------
# Static Assertion Tests
# ---------------------------------------------------------------------------


def test_rejects_invalid_threshold_negative():
    """Negative threshold should raise AssertionError."""
    params, paths = make_mock_params()
    tu_log_alpha = jnp.zeros((2, 6))

    with pytest.raises(AssertionError, match="must be in.*0.*1"):
        ratio_mask_coupling_penalty(params, paths, tu_log_alpha, min_ratio_threshold=-0.1)


def test_rejects_invalid_threshold_above_one():
    """Threshold > 1 should raise AssertionError."""
    params, paths = make_mock_params()
    tu_log_alpha = jnp.zeros((2, 6))

    with pytest.raises(AssertionError, match="must be in.*0.*1"):
        ratio_mask_coupling_penalty(params, paths, tu_log_alpha, min_ratio_threshold=1.5)


def test_rejects_invalid_tu_log_alpha_1d():
    """1D tu_log_alpha should raise AssertionError."""
    params, paths = make_mock_params()
    tu_log_alpha_1d = jnp.zeros((10,))

    with pytest.raises(AssertionError, match="must be 2D or 3D"):
        ratio_mask_coupling_penalty(params, paths, tu_log_alpha_1d)


def test_rejects_invalid_tu_log_alpha_4d():
    """4D tu_log_alpha should raise AssertionError."""
    params, paths = make_mock_params()
    tu_log_alpha_4d = jnp.zeros((2, 3, 4, 5))

    with pytest.raises(AssertionError, match="must be 2D or 3D"):
        ratio_mask_coupling_penalty(params, paths, tu_log_alpha_4d)


# ---------------------------------------------------------------------------
# JAX Checkify Tests for Runtime Validation
# ---------------------------------------------------------------------------


def test_checkify_catches_nan_in_tu_log_alpha():
    """Checkify should catch NaN values in tu_log_alpha at runtime.

    This tests runtime value validation using JAX's checkify system,
    which is critical for catching silent failures in JIT-compiled code.
    """
    ratio_values = jnp.array([[1.0, 0.001]])
    params, paths = make_mock_params(
        n_nodes=1, n_outputs=2, n_networks=1, n_tus=2,
        ratio_values=ratio_values
    )

    def checked_coupling(tu_log_alpha):
        # runtime NaN check (this is what checkify enables in JIT)
        has_nan = jnp.any(jnp.isnan(tu_log_alpha))
        checkify.check(~has_nan, "tu_log_alpha contains NaN")
        return ratio_mask_coupling_penalty(params, paths, tu_log_alpha)

    checked_fn = checkify.checkify(checked_coupling, errors=checkify.user_checks)
    jitted_fn = jax.jit(checked_fn)

    # valid input should work
    tu_log_alpha_valid = jnp.zeros((1, 2))
    err, result = jitted_fn(tu_log_alpha_valid)
    err.throw()  # no error

    # NaN input should trigger checkify error
    tu_log_alpha_nan = jnp.array([[0.0, jnp.nan]])
    err, _ = jitted_fn(tu_log_alpha_nan)
    with pytest.raises(checkify.JaxRuntimeError, match="NaN"):
        err.throw()


def test_checkify_catches_inf_in_ratios():
    """Checkify should catch Inf values in ratios at runtime."""
    params = ParameterTree()
    # inject Inf into ratios
    ratio_values = jnp.array([[1.0, jnp.inf]])
    params.at("local/layer_1/ratios", ratio_values)
    params.at("local/layer_1/output_tu_indices", jnp.array([[0, 1]], dtype=jnp.int32))
    params.at("local/layer_1/node_network_ids", jnp.array([0], dtype=jnp.int32))
    paths = ["local/layer_1/ratios"]

    def checked_coupling(tu_log_alpha):
        ratios = params["local/layer_1/ratios"]
        has_inf = jnp.any(jnp.isinf(ratios))
        checkify.check(~has_inf, "ratios contains Inf")
        return ratio_mask_coupling_penalty(params, paths, tu_log_alpha)

    checked_fn = checkify.checkify(checked_coupling, errors=checkify.user_checks)
    jitted_fn = jax.jit(checked_fn)

    tu_log_alpha = jnp.zeros((1, 2))
    err, _ = jitted_fn(tu_log_alpha)
    with pytest.raises(checkify.JaxRuntimeError, match="Inf"):
        err.throw()


def test_checkify_full_coupling_loop():
    """End-to-end test: checkify catches issues in optimization-like loop."""
    ratio_values = jnp.array([[1.0, 0.001]])
    params, paths = make_mock_params(
        n_nodes=1, n_outputs=2, n_networks=1, n_tus=2,
        ratio_values=ratio_values
    )

    def optimization_step(tu_log_alpha, step_idx):
        """Simulate one optimization step that could produce NaN."""
        penalty = ratio_mask_coupling_penalty(params, paths, tu_log_alpha)

        # simulate gradient step
        grads = jax.grad(
            lambda la: ratio_mask_coupling_penalty(params, paths, la)
        )(tu_log_alpha)

        # simulate potential NaN-producing update
        lr = 0.1
        new_tu_log_alpha = tu_log_alpha - lr * grads

        # runtime check for NaN in result
        checkify.check(
            jnp.all(jnp.isfinite(new_tu_log_alpha)),
            "tu_log_alpha became NaN/Inf after update"
        )

        return new_tu_log_alpha, penalty

    checked_step = checkify.checkify(optimization_step, errors=checkify.user_checks)
    jitted_step = jax.jit(checked_step)

    # run a few steps
    tu_log_alpha = jnp.zeros((1, 2))
    for i in range(5):
        err, (tu_log_alpha, penalty) = jitted_step(tu_log_alpha, i)
        err.throw()  # should not raise for valid inputs


def test_penalty_finite_for_extreme_log_alpha():
    """Penalty should remain finite even for extreme tu_log_alpha values."""
    ratio_values = jnp.array([[1.0, 0.001]])
    params, paths = make_mock_params(
        n_nodes=1, n_outputs=2, n_networks=1, n_tus=2,
        ratio_values=ratio_values
    )

    # test extreme positive
    tu_log_alpha_pos = jnp.ones((1, 2)) * 100.0
    penalty_pos = ratio_mask_coupling_penalty(params, paths, tu_log_alpha_pos)
    assert jnp.isfinite(penalty_pos), f"Extreme positive log_alpha gave {penalty_pos}"

    # test extreme negative
    tu_log_alpha_neg = jnp.ones((1, 2)) * (-100.0)
    penalty_neg = ratio_mask_coupling_penalty(params, paths, tu_log_alpha_neg)
    assert jnp.isfinite(penalty_neg), f"Extreme negative log_alpha gave {penalty_neg}"


# ---------------------------------------------------------------------------
# Integration Test with Real-ish Setup
# ---------------------------------------------------------------------------


def test_multiple_ratio_paths():
    """Test with multiple aggregation layers (multiple ratio_paths)."""
    params = ParameterTree()

    # layer 1
    params.at("local/layer_1/ratios", jnp.ones((2, 3)))
    params.at("local/layer_1/output_tu_indices", jnp.array([[0, 1, 2], [1, 2, 3]], dtype=jnp.int32))
    params.at("local/layer_1/node_network_ids", jnp.array([0, 1], dtype=jnp.int32))

    # layer 2: has one tiny ratio
    params.at("local/layer_2/ratios", jnp.array([[1.0, 0.001], [1.0, 1.0]]))
    params.at("local/layer_2/output_tu_indices", jnp.array([[4, 5], [5, 6]], dtype=jnp.int32))
    params.at("local/layer_2/node_network_ids", jnp.array([0, 1], dtype=jnp.int32))

    paths = ["local/layer_1/ratios", "local/layer_2/ratios"]
    tu_log_alpha = jnp.ones((2, 8)) * 5.0

    penalty = ratio_mask_coupling_penalty(params, paths, tu_log_alpha, min_ratio_threshold=0.005)

    # penalty should be positive (from layer_2's tiny ratio)
    assert float(penalty) > 0, f"Should have positive penalty from layer_2's tiny ratio"


def test_jit_compilation():
    """Verify the function compiles and runs under JIT."""
    params, paths = make_mock_params(
        n_nodes=4, n_outputs=3, n_networks=2, n_tus=6,
        ratio_values=jnp.array([
            [1.0, 1.0, 0.001],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ])
    )

    @jax.jit
    def compute_penalty(tu_log_alpha):
        return ratio_mask_coupling_penalty(params, paths, tu_log_alpha)

    tu_log_alpha = jnp.zeros((2, 6))

    # first call triggers compilation
    penalty1 = compute_penalty(tu_log_alpha)

    # second call uses cached compilation
    penalty2 = compute_penalty(tu_log_alpha)

    assert jnp.allclose(penalty1, penalty2), "JIT should produce consistent results"
    assert jnp.isfinite(penalty1), "Penalty should be finite"


def test_vmap_over_targets():
    """Verify vmapping over multiple targets works correctly."""
    params, paths = make_mock_params(
        n_nodes=2, n_outputs=3, n_networks=1, n_tus=4,
        ratio_values=jnp.array([[1.0, 0.001, 1.0], [1.0, 1.0, 1.0]])
    )

    # batch of tu_log_alpha for different targets
    tu_log_alpha_batch = jnp.ones((5, 1, 4)) * jnp.linspace(-2, 5, 5)[:, None, None]

    def single_penalty(tu_log_alpha_2d):
        return ratio_mask_coupling_penalty(
            params, paths, tu_log_alpha_2d, min_ratio_threshold=0.005
        )

    penalties = vmap(single_penalty)(tu_log_alpha_batch)

    assert penalties.shape == (5,), f"Expected (5,) penalties, got {penalties.shape}"
    assert jnp.all(jnp.isfinite(penalties)), "All penalties should be finite"
    # penalties should increase with higher log_alpha (higher enabled prob)
    assert penalties[-1] > penalties[0], (
        f"Higher log_alpha should give higher penalty. "
        f"low: {penalties[0]}, high: {penalties[-1]}"
    )
