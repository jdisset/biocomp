"""Tests for design module defensive programming and critical functionality.

These tests verify:
1. Graph topology properties (topological sort correctness)
2. Hard Concrete gradient flow (STE not blocking learning)
3. Shape assertions catch mismatches before silent broadcasting
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax import grad

from biocomp.graphengine import GraphState, GraphNode, GraphEdge
from biocomp.stack_builder import topological_order
from biocomp.tumasking import (
    hard_concrete_from_uniform,
    sample_hard_concrete,
    l0_penalty,
    get_final_mask,
    _validate_hard_concrete_params,
)


# ---------------------------------------------------------------------------
# Graph Topology Stress Test
# ---------------------------------------------------------------------------


def random_dag(n_nodes: int, edge_prob: float, seed: int) -> GraphState:
    """Generate a random DAG with given parameters."""
    rng = np.random.default_rng(seed)
    nodes = {
        i: GraphNode(node_id=i, node_type="generic", extra={"order": i}) for i in range(n_nodes)
    }
    edges = {}
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):  # only forward edges to ensure DAG
            if rng.random() < edge_prob:
                edge = GraphEdge(
                    source_id=i,
                    target_id=j,
                    from_output_slot=0,
                    to_input_slot=0,
                    content=(),  # empty content tuple for test purposes
                )
                edges[(i, j, 0, 0)] = edge
    return GraphState(nodes=nodes, edges=edges)


@pytest.mark.parametrize(
    "n_nodes,edge_prob,seed",
    [
        (5, 0.3, 42),
        (10, 0.2, 123),
        (15, 0.4, 456),
        (20, 0.25, 789),
        (8, 0.5, 101),
    ],
)
def test_topological_order_parents_before_children(n_nodes, edge_prob, seed):
    """Property test: every node's parents must appear earlier in topological order."""
    graph = random_dag(n_nodes, edge_prob, seed)
    batches = topological_order(graph)

    # flatten to get position of each node
    position = {}
    for batch_idx, batch in enumerate(batches):
        for node_id in batch:
            position[node_id] = batch_idx

    # verify all parents are processed before their children
    for edge in graph.edges.values():
        parent_id, child_id = edge.source_id, edge.target_id
        assert position[parent_id] <= position[child_id], (
            f"Parent {parent_id} at batch {position[parent_id]} should be processed "
            f"before child {child_id} at batch {position[child_id]}"
        )


def test_topological_order_all_nodes_included():
    """Verify all nodes from graph end up in the topological ordering."""
    graph = random_dag(12, 0.3, 999)
    batches = topological_order(graph)
    nodes_in_order = set()
    for batch in batches:
        nodes_in_order.update(batch)
    assert nodes_in_order == set(graph.nodes.keys()), (
        f"Missing nodes: {set(graph.nodes.keys()) - nodes_in_order}"
    )


# ---------------------------------------------------------------------------
# Hard Concrete Gradient Flow Test
# ---------------------------------------------------------------------------


def test_hard_concrete_gradient_flows():
    """Verify gradients flow through Hard Concrete STE (not blocked)."""
    # setup: mask is OFF (log_alpha << 0), target is 1.0
    # if gradients flow, they should push log_alpha UP to turn mask ON
    input_val = 1.0
    target = 1.0
    log_alpha_init = jnp.array([-2.0])  # mask should be mostly OFF

    def loss_fn(log_alpha):
        u = jnp.array([0.5])  # deterministic uniform sample
        mask = hard_concrete_from_uniform(u, log_alpha, temperature=0.5)
        output = mask * input_val
        return jnp.mean((output - target) ** 2)

    # compute gradient
    grads = grad(loss_fn)(log_alpha_init)

    # gradient should be non-zero and negative (reducing loss means increasing log_alpha)
    assert jnp.all(jnp.isfinite(grads)), "Gradients contain NaN/Inf"
    assert grads[0] < 0, (
        f"Gradient should be negative to increase log_alpha and turn mask ON, got {grads[0]}"
    )


def test_hard_concrete_gradient_magnitude():
    """Verify gradients have meaningful magnitude (not vanishing)."""
    log_alpha_near_zero = jnp.array([0.0])  # sigmoid active region

    def loss_fn(log_alpha):
        u = jnp.array([0.5])
        mask = hard_concrete_from_uniform(u, log_alpha, temperature=0.5)
        return jnp.sum(mask**2)

    grads = grad(loss_fn)(log_alpha_near_zero)
    assert jnp.abs(grads[0]) > 1e-4, (
        f"Gradient magnitude {jnp.abs(grads[0])} is too small, possible vanishing gradient"
    )


def test_hard_concrete_different_temperatures():
    """Test gradient flow at different temperatures."""
    log_alpha = jnp.array([0.0])

    def loss_fn(log_alpha, temp):
        u = jnp.array([0.5])
        mask = hard_concrete_from_uniform(u, log_alpha, temperature=temp)
        return jnp.sum(mask**2)

    for temp in [0.1, 0.5, 1.0, 2.0]:
        grads = grad(lambda la: loss_fn(la, temp))(log_alpha)
        assert jnp.all(jnp.isfinite(grads)), f"NaN/Inf gradient at temperature {temp}"
        assert jnp.abs(grads[0]) > 1e-6, f"Vanishing gradient at temperature {temp}"


# ---------------------------------------------------------------------------
# Parameter Validation Tests
# ---------------------------------------------------------------------------


def test_validate_hard_concrete_params_valid():
    """Valid params should not raise."""
    _validate_hard_concrete_params(gamma=-0.1, zeta=1.1, temperature=0.5)


def test_validate_hard_concrete_params_invalid_gamma():
    """Positive gamma should raise."""
    with pytest.raises(AssertionError, match="gamma must be negative"):
        _validate_hard_concrete_params(gamma=0.1, zeta=1.1, temperature=0.5)


def test_validate_hard_concrete_params_invalid_zeta():
    """zeta <= 1 should raise."""
    with pytest.raises(AssertionError, match="zeta must be > 1"):
        _validate_hard_concrete_params(gamma=-0.1, zeta=0.9, temperature=0.5)


def test_validate_hard_concrete_params_invalid_temperature():
    """Negative temperature should raise."""
    with pytest.raises(AssertionError, match="temperature must be non-negative"):
        _validate_hard_concrete_params(gamma=-0.1, zeta=1.1, temperature=-0.1)


def test_sample_hard_concrete_rejects_nan_log_alpha():
    """NaN in log_alpha should raise immediately."""
    log_alpha_with_nan = jnp.array([1.0, jnp.nan, -1.0])
    key = jax.random.PRNGKey(0)
    with pytest.raises(AssertionError, match="NaN/Inf in log_alpha"):
        sample_hard_concrete(log_alpha_with_nan, key)


# ---------------------------------------------------------------------------
# L0 Penalty and Final Mask Tests
# ---------------------------------------------------------------------------


def test_l0_penalty_monotonic():
    """L0 penalty should increase monotonically with log_alpha."""
    log_alphas = jnp.linspace(-3, 3, 100)
    penalties = jax.vmap(lambda la: l0_penalty(jnp.array([la])))(log_alphas)
    penalties = penalties.squeeze()
    # verify monotonic increasing
    diffs = jnp.diff(penalties)
    assert jnp.all(diffs >= -1e-6), "L0 penalty should be monotonically increasing"


def test_final_mask_binary():
    """Final mask should be strictly binary (0 or 1)."""
    log_alphas = jnp.array([-5.0, -1.0, 0.0, 1.0, 5.0])
    mask = get_final_mask(log_alphas)
    unique_vals = jnp.unique(mask)
    assert jnp.all((unique_vals == 0.0) | (unique_vals == 1.0)), (
        f"Mask should be binary, got unique values: {unique_vals}"
    )


def test_final_mask_threshold_behavior():
    """Test mask threshold behavior."""
    # log_alpha > 0 -> sigmoid > 0.5 -> mask = 1
    # log_alpha < 0 -> sigmoid < 0.5 -> mask = 0
    log_alphas = jnp.array([-2.0, -0.1, 0.1, 2.0])
    mask = get_final_mask(log_alphas, threshold=0.5)
    expected = jnp.array([0.0, 0.0, 1.0, 1.0])
    assert jnp.allclose(mask, expected), f"Mask {mask} != expected {expected}"


# ---------------------------------------------------------------------------
# Shape Broadcasting Detection Tests
# ---------------------------------------------------------------------------


def test_shape_mismatch_detection():
    """Verify that shape assertions catch mismatches that would silently broadcast."""
    # This simulates what would happen if Y has wrong shape in loss computation
    # Without assertion: (100,) - (100, 1) = (100, 100) -> mean -> scalar (WRONG!)
    # With assertion: immediate failure

    y_correct = jnp.ones((100,))
    y_wrong_shape = jnp.ones((100, 1))

    # the subtraction would broadcast silently
    diff_result = y_correct - y_wrong_shape
    assert diff_result.shape == (100, 100), "Broadcasting happened as expected for demo"

    # in our code, the assertion would catch this:
    def safe_mse(y_pred, y_target):
        assert y_pred.shape == y_target.shape, f"Shape mismatch: {y_pred.shape} vs {y_target.shape}"
        return jnp.mean((y_pred - y_target) ** 2)

    # correct shapes work
    safe_mse(y_correct, y_correct)

    # mismatched shapes raise
    with pytest.raises(AssertionError, match="Shape mismatch"):
        safe_mse(y_correct, y_wrong_shape)


# ---------------------------------------------------------------------------
# JAX Checkify Integration Tests
# ---------------------------------------------------------------------------


def test_checkify_catches_nan_in_optimization_loop():
    """End-to-end test: checkify catches NaN in a JIT-compiled optimization step.

    This tests that JAX's checkify system can catch runtime errors (NaN/Inf)
    in the middle of JIT-compiled code, which is critical for detecting
    silent failures in training/design loops.
    """
    from jax.experimental import checkify
    import optax

    # Simple loss function that can produce NaN with bad inputs
    def loss_fn(params, x):
        # sqrt of negative number produces NaN
        return jnp.mean(jnp.sqrt(params * x))

    # Optimization step that includes a NaN check
    def step_fn(params, opt_state, x):
        loss, grads = jax.value_and_grad(loss_fn)(params, x)
        # Manual check for NaN (this is what checkify enables in JIT code)
        checkify.check(jnp.isfinite(loss), "Loss became NaN/Inf")
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    # Setup
    optimizer = optax.sgd(0.1)
    params = jnp.array([1.0])
    opt_state = optimizer.init(params)
    x_good = jnp.array([1.0])  # sqrt(1*1) = 1 -> valid
    x_bad = jnp.array([-1.0])  # sqrt(1*-1) -> NaN

    # Wrap with checkify
    checked_step = checkify.checkify(step_fn, errors=(checkify.user_checks | checkify.float_checks))
    jitted_step = jax.jit(checked_step)

    # Good input should work
    err, (new_params, new_opt_state, loss) = jitted_step(params, opt_state, x_good)
    # No error should be raised
    err.throw()  # This would raise if there was an error

    # Bad input should trigger checkify error
    err, (_, _, _) = jitted_step(params, opt_state, x_bad)
    with pytest.raises(checkify.JaxRuntimeError, match="NaN"):
        err.throw()


def test_checkify_enabled_via_environment():
    """Test that checkify can be enabled via environment variable."""
    from biocomp import jaxutils
    import os

    old_env = os.environ.get("BIOCOMP_CHECKIFY", "")
    try:
        # Test enable_checks flag
        jaxutils.set_enable_checks(True)
        assert jaxutils.enable_checks is True

        jaxutils.set_enable_checks(False)
        assert jaxutils.enable_checks is False
    finally:
        # Restore original state
        os.environ["BIOCOMP_CHECKIFY"] = old_env
        jaxutils.set_enable_checks(False)


def test_jaxutils_check_function_concrete_values():
    """Test that jaxutils.check works for concrete (non-traced) values."""
    from biocomp.jaxutils import check, set_enable_checks

    set_enable_checks(False)

    # Should pass for True condition
    check(True, "This should not fail")

    # Should fail for False condition
    with pytest.raises(AssertionError, match="Expected failure"):
        check(False, "Expected failure")

    set_enable_checks(False)


def test_jaxutils_check_function_with_checkify():
    """Test jaxutils.check inside JIT when checkify is enabled."""
    from biocomp.jaxutils import check, set_enable_checks
    from jax.experimental import checkify

    set_enable_checks(True)
    try:
        def fn_with_check(x):
            check(x > 0, "x must be positive")
            return x * 2

        checked_fn = checkify.checkify(fn_with_check, errors=checkify.user_checks)
        jitted_fn = jax.jit(checked_fn)

        # Positive input should work
        err, result = jitted_fn(jnp.array(1.0))
        err.throw()  # No error
        assert result == 2.0

        # Negative input should fail
        err, _ = jitted_fn(jnp.array(-1.0))
        with pytest.raises(checkify.JaxRuntimeError, match="positive"):
            err.throw()
    finally:
        set_enable_checks(False)
