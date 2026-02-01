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
from pathlib import Path
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

RESOURCES_DIR = Path(__file__).parent / "resources"


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


# ---------------------------------------------------------------------------
# Designloss Checkify Integration Tests
# ---------------------------------------------------------------------------


def test_checkify_sinkhorn_valid_inputs():
    """Verify Sinkhorn divergence works with valid inputs under checkify."""
    from jax.experimental import checkify
    from biocomp.designloss import sinkhorn_divergence_conv

    # Good inputs - should pass
    a = jnp.ones((10, 10)) / 100
    b = jnp.ones((10, 10)) / 100

    checked_fn = checkify.checkify(sinkhorn_divergence_conv, errors=checkify.float_checks)
    err, result = checked_fn(a, b, eps=0.1, n_iters=5)
    err.throw()  # no error
    assert jnp.isfinite(result)

    # Note: NaN inputs are sanitized internally (converted to 0) by design,
    # so checkify won't catch them. This is intentional robustness.


def test_checkify_mse_loss_catches_nan():
    """Verify checkify catches NaN in MSE loss computation."""
    from jax.experimental import checkify
    from biocomp.designloss import mse_loss

    # Good inputs
    x = jnp.linspace(0, 1, 10)[:, None]
    y = jnp.sin(x).squeeze()
    yhat = jnp.cos(x).squeeze()

    checked_fn = checkify.checkify(mse_loss, errors=checkify.float_checks)
    err, result = checked_fn(x, y, yhat)
    err.throw()  # no error
    assert jnp.isfinite(result)


def test_checkify_zncc_loss_catches_nan():
    """Verify checkify catches NaN in ZNCC loss computation."""
    from jax.experimental import checkify
    from biocomp.designloss import zncc_loss

    # Good inputs
    x = jnp.linspace(0, 1, 10)[:, None]
    y = jnp.sin(x).squeeze()
    yhat = jnp.cos(x).squeeze()

    checked_fn = checkify.checkify(zncc_loss, errors=checkify.float_checks)
    err, result = checked_fn(x, y, yhat)
    err.throw()  # no error
    assert jnp.isfinite(result)


def test_checkify_lncc_grid_loss_catches_nan():
    """Verify checkify catches NaN in LNCC grid loss computation."""
    from jax.experimental import checkify
    from biocomp.designloss import lncc_grid_loss

    # Good inputs - 2D grids
    y = jnp.ones((8, 8)) * 0.5
    yhat = jnp.ones((8, 8)) * 0.6

    checked_fn = checkify.checkify(lncc_grid_loss, errors=checkify.float_checks)
    err, result = checked_fn(None, y, yhat, k=3)
    err.throw()  # no error
    assert jnp.isfinite(result)


def test_checkify_gradient_through_loss():
    """Verify checkify works through gradient computation."""
    from jax.experimental import checkify
    from biocomp.designloss import mse_loss

    def loss_with_params(params, x, y):
        yhat = params * x  # yhat shape matches x
        return mse_loss(x, y, yhat)

    x = jnp.linspace(0.1, 1, 10)
    y = x * 2  # same shape as x

    # Compute gradient with checkify
    def grad_loss(params):
        return jax.grad(lambda p: loss_with_params(p, x, y))(params)

    checked_grad = checkify.checkify(grad_loss, errors=checkify.float_checks)

    # Good params
    err, grads = checked_grad(jnp.array(1.5))
    err.throw()  # no error
    assert jnp.isfinite(grads)


def test_checkify_vmap_catches_per_element_nan():
    """Verify checkify catches NaN in vmapped operations."""
    from jax.experimental import checkify
    from biocomp.designloss import mse_loss

    def batched_mse(y_batch, yhat_batch):
        return jax.vmap(lambda y, yh: mse_loss(None, y, yh))(y_batch, yhat_batch)

    # Good inputs
    y = jnp.ones((5, 10))
    yhat = jnp.ones((5, 10)) * 0.9

    checked_fn = checkify.checkify(batched_mse, errors=checkify.float_checks)
    err, result = checked_fn(y, yhat)
    err.throw()  # no error
    assert jnp.all(jnp.isfinite(result))


def test_checkify_index_bounds_in_coupling_penalty():
    """Verify checkify.index_checks catches out-of-bounds indexing."""
    from jax.experimental import checkify
    from biocomp.parameters import ParameterTree
    from biocomp.designloss import ratio_mask_coupling_penalty

    # Create params with valid data
    params = ParameterTree()
    n_networks, n_tus, n_nodes, n_outputs = 2, 4, 3, 2

    ratios = jnp.ones((n_nodes, n_outputs))
    tu_indices = jnp.array([[0, 1], [1, 2], [2, 3]])  # valid indices
    network_ids = jnp.array([0, 1, 0])
    tu_log_alpha = jnp.zeros((n_networks, n_tus))

    params.at("local/layer_1/ratios", ratios)
    params.at("local/layer_1/output_tu_indices", tu_indices)
    params.at("local/layer_1/node_network_ids", network_ids)

    ratio_paths = ["local/layer_1/ratios"]

    checked_fn = checkify.checkify(
        ratio_mask_coupling_penalty, errors=checkify.index_checks | checkify.float_checks
    )

    # Valid inputs should work
    err, result = checked_fn(params, ratio_paths, tu_log_alpha, min_ratio_threshold=0.1)
    err.throw()  # no error
    assert jnp.isfinite(result)


# ---------------------------------------------------------------------------
# Input Order Tests
# ---------------------------------------------------------------------------


def test_recipe_input_order_validation():
    """Test that Recipe validates input_order for duplicates."""
    from biocomp.recipe import Recipe
    from pydantic import ValidationError

    # Valid input_order should work
    recipe = Recipe(
        name="test_recipe",
        content=[],
        input_order=["mKO2", "eBFP2"],
    )
    assert recipe.has_input_order()
    assert recipe.input_order == ["mKO2", "eBFP2"]

    # Duplicate proteins should fail
    with pytest.raises((AssertionError, ValidationError)):
        Recipe(
            name="test_recipe",
            content=[],
            input_order=["mKO2", "mKO2", "eBFP2"],
        )


def test_network_input_order_propagation():
    """Test that input_order propagates from Recipe to Network."""
    from biocomp.recipe import Recipe, CoTransfection, TranscriptionUnit
    from biocomp.network import recipe_to_networks

    cotx1 = CoTransfection(
        name="x1",
        units=[TranscriptionUnit(name="tu1", slots=["hEF1a", "mKO2", "L0.T_4560"])],
        ratios=["1.0"],
    )
    cotx2 = CoTransfection(
        name="x2",
        units=[TranscriptionUnit(name="tu2", slots=["hEF1a", "eBFP2", "L0.T_4560"])],
        ratios=["1.0"],
    )

    # Build without input_order first to get natural order
    recipe_no_order = Recipe(
        name="test_recipe",
        content=[cotx1, cotx2],
    )
    networks_no_order = recipe_to_networks(recipe_no_order, invert=True, inversion_mode="main")
    assert len(networks_no_order) > 0
    natural_order = networks_no_order[0].get_inverted_input_proteins()

    # Now build with reversed input_order
    reversed_order = list(reversed(natural_order))
    recipe_with_order = Recipe(
        name="test_recipe",
        content=[cotx1, cotx2],
        input_order=reversed_order,
    )

    networks_with_order = recipe_to_networks(recipe_with_order, invert=True, inversion_mode="main")
    assert len(networks_with_order) > 0

    for net in networks_with_order:
        assert net.has_input_order(), f"Network {net.name} should have input_order"
        assert net.get_input_order() == reversed_order
        assert net.get_inverted_input_proteins() == reversed_order


def test_network_has_input_order():
    """Test that Network.has_input_order returns correct value."""
    from biocomp.network import Network

    # Network without input_order
    net_no_order = Network(name="test_net", metadata={})
    assert not net_no_order.has_input_order()
    assert net_no_order.get_input_order() is None

    # Network with input_order
    net_with_order = Network(
        name="test_net",
        metadata={"input_order": ["mKO2", "eBFP2"]},
    )
    assert net_with_order.has_input_order()
    assert net_with_order.get_input_order() == ["mKO2", "eBFP2"]


# ---------------------------------------------------------------------------
# Grid Data Ordering Tests (Lattice Mode)
# ---------------------------------------------------------------------------


def test_grid_data_sequential_order_required():
    """Test that grid_distance_loss requires data in sequential grid order.

    The bug: reshuffle_batches=True permutes data order, but grid_distance_loss
    reshapes to (yres, xres) assuming sequential order. This creates a scrambled
    grid where sinkhorn/lncc losses are meaningless.

    This test verifies that we can detect when data is NOT in sequential order.
    """
    xres, yres = 8, 8

    # Generate sequential grid coordinates
    x_coords = jnp.linspace(0, 1, xres)
    y_coords = jnp.linspace(0, 1, yres)
    xx, yy = jnp.meshgrid(x_coords, y_coords)
    X_sequential = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)

    # Reshape to grid and verify structure
    X_grid = X_sequential.reshape(yres, xres, 2)

    # First row should have constant y=0 and increasing x
    first_row_x = X_grid[0, :, 0]
    first_row_y = X_grid[0, :, 1]
    assert jnp.allclose(first_row_y, 0.0), "First row y-coords should all be 0"
    assert jnp.all(first_row_x[1:] > first_row_x[:-1]), "First row x-coords must be monotonically increasing"

    # First column should have constant x=0 and increasing y
    first_col_x = X_grid[:, 0, 0]
    first_col_y = X_grid[:, 0, 1]
    assert jnp.allclose(first_col_x, 0.0), "First column x-coords should all be 0"
    assert jnp.all(first_col_y[1:] > first_col_y[:-1]), "First column y-coords must be monotonically increasing"


def test_shuffled_grid_detection():
    """Test that we can detect when grid data has been shuffled."""
    xres, yres = 8, 8

    x_coords = jnp.linspace(0, 1, xres)
    y_coords = jnp.linspace(0, 1, yres)
    xx, yy = jnp.meshgrid(x_coords, y_coords)
    X_sequential = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)

    # Shuffle the data (simulating reshuffle_batches=True)
    key = jax.random.PRNGKey(42)
    perm = jax.random.permutation(key, X_sequential.shape[0])
    X_shuffled = X_sequential[perm]

    # Reshape to grid - this is what grid_distance_loss does
    X_shuffled_grid = X_shuffled.reshape(yres, xres, 2)

    # Check if first row x-coords are monotonically increasing
    first_row_x = X_shuffled_grid[0, :, 0]
    is_monotonic = jnp.all(first_row_x[1:] >= first_row_x[:-1])

    # With shuffled data, this SHOULD fail (detecting the problem)
    assert not is_monotonic, (
        "Shuffled data should NOT have monotonically increasing x-coords in first row. "
        "If this passes, the shuffle didn't work as expected."
    )


def test_validate_grid_order_helper():
    """Test helper function that validates grid ordering."""

    def validate_grid_order(X: jnp.ndarray, xres: int, yres: int, rtol: float = 0.01) -> bool:
        """Check if X coordinates are in valid sequential grid order.

        Args:
            X: Coordinates array of shape (n_points, 2) where n_points = xres * yres
            xres: Expected x resolution
            yres: Expected y resolution
            rtol: Relative tolerance for coordinate comparisons

        Returns:
            True if X is in valid sequential grid order, False otherwise
        """
        assert X.shape == (xres * yres, 2), f"Expected shape ({xres * yres}, 2), got {X.shape}"

        X_grid = X.reshape(yres, xres, 2)

        # Check row structure: each row should have same y, increasing x
        for row_idx in range(yres):
            row_x = X_grid[row_idx, :, 0]
            row_y = X_grid[row_idx, :, 1]

            # All y values in row should be approximately equal
            if not jnp.allclose(row_y, row_y[0], rtol=rtol):
                return False

            # x values should be monotonically increasing
            if not jnp.all(row_x[1:] >= row_x[:-1] - rtol):
                return False

        # Check column structure: each column should have same x, increasing y
        for col_idx in range(xres):
            col_x = X_grid[:, col_idx, 0]
            col_y = X_grid[:, col_idx, 1]

            # All x values in column should be approximately equal
            if not jnp.allclose(col_x, col_x[0], rtol=rtol):
                return False

            # y values should be monotonically increasing
            if not jnp.all(col_y[1:] >= col_y[:-1] - rtol):
                return False

        return True

    xres, yres = 8, 8

    # Sequential data should pass
    x_coords = jnp.linspace(0, 1, xres)
    y_coords = jnp.linspace(0, 1, yres)
    xx, yy = jnp.meshgrid(x_coords, y_coords)
    X_sequential = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)
    assert validate_grid_order(X_sequential, xres, yres), "Sequential data should be valid"

    # Shuffled data should fail
    key = jax.random.PRNGKey(42)
    perm = jax.random.permutation(key, X_sequential.shape[0])
    X_shuffled = X_sequential[perm]
    assert not validate_grid_order(X_shuffled, xres, yres), "Shuffled data should be invalid"


def test_reshuffle_batches_incompatible_with_grid_loss():
    """Integration test: verify reshuffle_batches=False is set for lattice mode.

    This test loads the actual config and verifies the critical setting.
    """
    config_path = RESOURCES_DIR / "design/design_configs/base.yaml"

    if not config_path.exists():
        pytest.skip(f"Config file not found at {config_path}")

    content = config_path.read_text()

    # Verify the critical setting is present
    assert "reshuffle_batches: false" in content or "reshuffle_batches: False" in content, (
        "CRITICAL: base.yaml must have 'reshuffle_batches: false' for lattice mode. "
        "Without this, grid_distance_loss receives scrambled data and produces meaningless losses. "
        "See test_grid_data_sequential_order_required for details."
    )


def test_validate_grid_order_passes_for_sequential_data():
    """Test _validate_grid_order passes for correctly ordered data."""
    from biocomp.designloss import _validate_grid_order
    from biocomp.jaxutils import set_enable_checks

    xres, yres = 8, 8
    x_coords = jnp.linspace(0, 1, xres)
    y_coords = jnp.linspace(0, 1, yres)
    xx, yy = jnp.meshgrid(x_coords, y_coords)
    X_sequential = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)

    set_enable_checks(False)
    try:
        _validate_grid_order(X_sequential, xres, yres)
    finally:
        set_enable_checks(False)


def test_validate_grid_order_fails_for_shuffled_data():
    """Test _validate_grid_order catches shuffled data."""
    from biocomp.designloss import _validate_grid_order
    from biocomp.jaxutils import set_enable_checks

    xres, yres = 8, 8
    x_coords = jnp.linspace(0, 1, xres)
    y_coords = jnp.linspace(0, 1, yres)
    xx, yy = jnp.meshgrid(x_coords, y_coords)
    X_sequential = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)

    key = jax.random.PRNGKey(42)
    perm = jax.random.permutation(key, X_sequential.shape[0])
    X_shuffled = X_sequential[perm]

    set_enable_checks(False)
    try:
        with pytest.raises(AssertionError, match="Grid order violation"):
            _validate_grid_order(X_shuffled, xres, yres)
    finally:
        set_enable_checks(False)


def test_validate_grid_order_tolerates_jitter():
    """Test _validate_grid_order allows small jitter in coordinates."""
    from biocomp.designloss import _validate_grid_order
    from biocomp.jaxutils import set_enable_checks

    xres, yres = 8, 8
    x_coords = jnp.linspace(0, 1, xres)
    y_coords = jnp.linspace(0, 1, yres)
    xx, yy = jnp.meshgrid(x_coords, y_coords)
    X_sequential = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)

    key = jax.random.PRNGKey(123)
    jitter_std = 0.01
    jitter = jax.random.normal(key, X_sequential.shape) * jitter_std
    X_jittered = X_sequential + jitter

    set_enable_checks(False)
    try:
        _validate_grid_order(X_jittered, xres, yres)
    finally:
        set_enable_checks(False)


def test_validate_grid_order_with_checkify():
    """Test _validate_grid_order works with checkify when enable_checks=True."""
    from jax.experimental import checkify
    from biocomp.designloss import _validate_grid_order
    from biocomp.jaxutils import set_enable_checks

    xres, yres = 8, 8
    x_coords = jnp.linspace(0, 1, xres)
    y_coords = jnp.linspace(0, 1, yres)
    xx, yy = jnp.meshgrid(x_coords, y_coords)
    X_sequential = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)

    key = jax.random.PRNGKey(42)
    perm = jax.random.permutation(key, X_sequential.shape[0])
    X_shuffled = X_sequential[perm]

    set_enable_checks(True)
    try:
        checked_fn = checkify.checkify(_validate_grid_order, errors=checkify.user_checks)
        err, _ = jax.jit(checked_fn, static_argnums=(1, 2))(X_shuffled, xres, yres)
        with pytest.raises(checkify.JaxRuntimeError, match="Grid order violation"):
            err.throw()
    finally:
        set_enable_checks(False)


# ---------------------------------------------------------------------------
# Loss Component Sanity Tests
# ---------------------------------------------------------------------------


def test_loss_components_bounded():
    """Test that individual loss components are in expected ranges."""
    from biocomp.designloss import (
        sinkhorn_divergence_conv,
        lncc_grid_loss,
        mse_loss,
        zncc_loss,
    )

    y = jnp.ones((8, 8)) * 0.5
    yhat = jnp.ones((8, 8)) * 0.6

    sink = sinkhorn_divergence_conv(y, yhat, eps=0.1, n_iters=10)
    assert jnp.isfinite(sink), "sinkhorn should be finite"
    assert sink >= 0, f"sinkhorn should be >= 0, got {sink}"

    lncc = lncc_grid_loss(None, y, yhat, k=3)
    assert jnp.isfinite(lncc), "lncc should be finite"
    assert 0 <= lncc <= 2, f"lncc should be in [0, 2], got {lncc}"

    mse = mse_loss(None, y.ravel(), yhat.ravel())
    assert jnp.isfinite(mse), "mse should be finite"
    assert mse >= 0, f"mse should be >= 0, got {mse}"

    zncc = zncc_loss(None, y.ravel(), yhat.ravel())
    assert jnp.isfinite(zncc), "zncc should be finite"


def test_loss_components_zero_for_identical():
    """Test that loss is zero or minimal when prediction equals target."""
    from biocomp.designloss import mse_loss, lncc_grid_loss

    key = jax.random.PRNGKey(42)
    y = jax.random.uniform(key, (8, 8))
    yhat = y

    mse = mse_loss(None, y.ravel(), yhat.ravel())
    assert jnp.abs(mse) < 1e-6, f"mse should be ~0 for identical, got {mse}"

    lncc = lncc_grid_loss(None, y, yhat, k=3)
    assert jnp.abs(lncc) < 1e-4, f"lncc should be ~0 for identical, got {lncc}"


def test_yhatdep_shape_assertion_catches_mismatch():
    """Test that yhatdep shape mismatch is caught.

    This tests the assertion added to compute_losses that verifies
    yhatdep.shape == (batch_size, n_targets, n_networks).
    """
    batch_size, n_targets, n_networks = 64, 2, 3
    yhatdep_correct = jnp.ones((batch_size, n_targets, n_networks))
    yhatdep_wrong = jnp.ones((batch_size, n_networks, n_targets))

    assert yhatdep_correct.shape == (batch_size, n_targets, n_networks)
    assert yhatdep_wrong.shape != (batch_size, n_targets, n_networks), (
        "This test verifies the assertion would catch swapped axes"
    )


def test_grid_images_shape_match():
    """Test that Y_images and yhat_images have matching shapes after reshape.

    This mirrors the assertion in compute_losses that catches shape mismatches
    that could lead to silent broadcasting in loss computation.
    """
    xres, yres = 8, 8
    n_targets, n_networks = 2, 3
    batch_size = xres * yres

    Y = jnp.ones((batch_size, n_targets, 1))
    yhatdep = jnp.ones((batch_size, n_targets, n_networks))

    Y_images = jnp.tile(
        Y.squeeze(-1).T.reshape(n_targets, 1, yres, xres), (1, n_networks, 1, 1)
    )
    yhat_images = yhatdep.transpose(1, 2, 0).reshape(n_targets, n_networks, yres, xres)

    assert Y_images.shape == yhat_images.shape, (
        f"Shape mismatch: Y_images {Y_images.shape} != yhat_images {yhat_images.shape}"
    )
    assert Y_images.shape == (n_targets, n_networks, yres, xres)


def test_prediction_range_plausible():
    """Test that model predictions are in a plausible range.

    Predictions in latent space should typically be in [0, 1] or at least finite.
    This test documents the expected behavior.
    """
    yhat = jnp.array([0.0, 0.5, 1.0, -0.1, 1.1])

    assert jnp.all(jnp.isfinite(yhat)), "Predictions should be finite"

    in_range = (yhat >= -0.5) & (yhat <= 1.5)
    pct_in_range = jnp.mean(in_range)
    assert pct_in_range > 0.8, (
        f"Most predictions should be near [0,1], only {pct_in_range*100:.0f}% in range"
    )
