"""Tests for TU masking utilities (Hard Concrete distribution)."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from biocomp.tumasking import (
    sample_hard_concrete,
    sample_hard_concrete_deterministic,
    get_final_mask,
    l0_penalty,
    l0_loss,
    get_tu_mask_for_node,
    init_tu_log_alpha,
    L0_PENALTY_FLOOR_PROB,
    asymmetric_l0_loss,
    decode_latent_tu_masking,
)


def test_sample_hard_concrete_bounds():
    """Samples should be in [0, 1]."""
    key = jax.random.key(42)
    log_alphas = jnp.array([-5.0, -1.0, 0.0, 1.0, 5.0])
    n_samples = 1000

    for log_alpha in log_alphas:
        keys = jax.random.split(key, n_samples)
        samples = jax.vmap(lambda k: sample_hard_concrete(log_alpha, k))(keys)
        assert jnp.all(samples >= 0.0), f"samples should be >= 0, got min={samples.min()}"
        assert jnp.all(samples <= 1.0), f"samples should be <= 1, got max={samples.max()}"


def test_sample_hard_concrete_sparsity():
    """Lower log_alpha should give more zeros."""
    key = jax.random.key(42)
    n_samples = 5000

    p_zeros = []
    log_alphas = [-5.0, -2.0, 0.0, 2.0, 5.0]

    for log_alpha in log_alphas:
        keys = jax.random.split(key, n_samples)
        samples = jax.vmap(lambda k: sample_hard_concrete(log_alpha, k))(keys)
        p_zero = (samples == 0.0).mean()
        p_zeros.append(float(p_zero))

    # P(z=0) should decrease as log_alpha increases
    for i in range(len(p_zeros) - 1):
        assert p_zeros[i] > p_zeros[i + 1], (
            f"P(z=0) should decrease with log_alpha: {list(zip(log_alphas, p_zeros, strict=False))}"
        )


def test_sample_hard_concrete_exact_zeros_and_ones():
    """Hard Concrete should produce exactly 0 and exactly 1."""
    key = jax.random.key(42)
    n_samples = 5000

    # With extreme log_alpha, should get exact 0s or 1s
    keys = jax.random.split(key, n_samples)
    samples_low = jax.vmap(lambda k: sample_hard_concrete(-5.0, k))(keys)
    samples_high = jax.vmap(lambda k: sample_hard_concrete(5.0, k))(keys)

    # Check we have exact zeros
    n_exact_zeros = int((samples_low == 0.0).sum())
    assert n_exact_zeros > n_samples * 0.8, (
        f"Expected mostly zeros, got {n_exact_zeros}/{n_samples}"
    )

    # Check we have exact ones
    n_exact_ones = int((samples_high == 1.0).sum())
    assert n_exact_ones > n_samples * 0.8, f"Expected mostly ones, got {n_exact_ones}/{n_samples}"


def test_sample_hard_concrete_deterministic():
    """Deterministic version should be consistent."""
    log_alphas = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])

    z1 = sample_hard_concrete_deterministic(log_alphas)
    z2 = sample_hard_concrete_deterministic(log_alphas)

    np.testing.assert_array_equal(z1, z2)
    assert jnp.all(z1 >= 0.0) and jnp.all(z1 <= 1.0)


def test_get_final_mask():
    """Final mask should be binary based on sigmoid(log_alpha)."""
    log_alphas = jnp.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    masks = get_final_mask(log_alphas)

    # sigmoid(-3) ≈ 0.047, sigmoid(-0.5) ≈ 0.38, sigmoid(0) = 0.5, etc.
    expected = jnp.array([0.0, 0.0, 1.0, 1.0, 1.0])  # threshold at 0.5
    np.testing.assert_array_equal(masks, expected)


def test_l0_penalty_range():
    """L0 penalty should be in [0, 1) with floor behavior."""
    # Below floor (sigmoid < 0.2): penalty = 0
    log_alphas_low = jnp.array([-10.0, -3.0])  # sigmoid ≈ 0, 0.05
    penalties_low = l0_penalty(log_alphas_low)
    assert jnp.all(penalties_low == 0.0), "L0 penalty should be 0 below floor"

    # Above floor: penalty in (0, 1)
    log_alphas_high = jnp.array([0.0, 3.0, 10.0])  # sigmoid ≈ 0.5, 0.95, 1.0
    penalties_high = l0_penalty(log_alphas_high)
    assert jnp.all(penalties_high > 0.0), "L0 penalty should be > 0 above floor"
    assert jnp.all(penalties_high < 1.0), "L0 penalty should be < 1"


def test_l0_penalty_monotonic():
    """L0 penalty should be monotonically non-decreasing with log_alpha."""
    log_alphas = jnp.linspace(-5.0, 5.0, 20)
    penalties = l0_penalty(log_alphas)

    # Should be monotonically non-decreasing (flat at 0 below floor, then increasing)
    diffs = penalties[1:] - penalties[:-1]
    assert jnp.all(diffs >= 0), "L0 penalty should be non-decreasing in log_alpha"

    # Strictly increasing above floor (sigmoid > 0.2, log_alpha > -1.39)
    above_floor_idx = log_alphas > -1.0  # safely above floor
    above_floor_alphas = log_alphas[above_floor_idx]
    above_floor_penalties = l0_penalty(above_floor_alphas)
    above_floor_diffs = above_floor_penalties[1:] - above_floor_penalties[:-1]
    assert jnp.all(above_floor_diffs > 0), "L0 penalty should be strictly increasing above floor"


def test_l0_penalty_floor_behavior():
    """L0 penalty floor allows TU 'rebirth' by removing L0 pressure below threshold."""
    # Note: clamp_log_alpha applies soft tanh clamping, so we need values well below floor
    # to ensure the clamped value is still below floor_prob threshold

    # Well below floor (log_alpha=-3 -> clamped sigmoid << 0.2): penalty = 0
    penalty_well_below = l0_penalty(jnp.array(-3.0))
    assert penalty_well_below == 0.0, (
        f"Well below floor, penalty should be 0, got {penalty_well_below}"
    )

    # Slightly below floor (log_alpha=-2): sigmoid after clamping ~0.15 < 0.2
    penalty_below = l0_penalty(jnp.array(-2.0))
    assert penalty_below == 0.0, f"Below floor, penalty should be 0, got {penalty_below}"

    # Above floor (log_alpha=0): sigmoid = 0.5 > 0.2
    penalty_above = l0_penalty(jnp.array(0.0))
    expected_above = (0.5 - L0_PENALTY_FLOOR_PROB) / (1.0 - L0_PENALTY_FLOOR_PROB)
    np.testing.assert_allclose(penalty_above, expected_above, rtol=0.01)

    # At high log_alpha: penalty approaches 1 (normalized)
    # Note: clamp_log_alpha limits range, so sigmoid doesn't reach exactly 1
    penalty_max = l0_penalty(jnp.array(10.0))
    assert penalty_max > 0.95, f"High log_alpha should give penalty near 1, got {penalty_max}"

    # Gradient well below floor is 0 (flat region)
    grad_below_floor = jax.grad(lambda la: l0_penalty(la))(jnp.array(-3.0))
    assert grad_below_floor == 0.0, "Gradient should be 0 well below floor"

    # Gradient above floor is positive
    grad_above_floor = jax.grad(lambda la: l0_penalty(la))(jnp.array(0.0))
    assert grad_above_floor > 0.0, "Gradient should be positive above floor"


def test_l0_penalty_gradient():
    """L0 penalty should be differentiable."""
    log_alpha = jnp.array(0.0)

    grad_fn = jax.grad(lambda la: l0_penalty(la))
    grad = grad_fn(log_alpha)

    assert not jnp.isnan(grad), "Gradient should not be NaN"
    assert grad > 0, "Gradient of L0 penalty should be positive"


def test_l0_loss():
    """L0 loss should sum penalties."""
    log_alphas = jnp.array([0.0, 0.0, 0.0])
    loss = l0_loss(log_alphas)

    expected = 3 * l0_penalty(jnp.array(0.0))
    np.testing.assert_allclose(loss, expected, rtol=1e-5)


def test_l0_loss_threshold_nonlinear():
    """L0 loss with threshold should be gentle below and harsh above."""
    threshold = 12.0

    # Create log_alphas that give ~6 expected TUs (below threshold)
    # sigmoid(2.0) ≈ 0.88, after floor: (0.88 - 0.2) / 0.8 ≈ 0.85 per TU
    # 7 TUs * 0.85 ≈ 6 expected count
    log_alphas_low = jnp.full((7,), 2.0)
    loss_low_linear = l0_loss(log_alphas_low, tu_threshold=None)
    loss_low_nonlin = l0_loss(log_alphas_low, tu_threshold=threshold)

    # Below threshold: nonlinear should be similar to linear (just small excess from softplus)
    assert loss_low_nonlin < loss_low_linear * 1.5, (
        f"Below threshold, nonlinear loss ({loss_low_nonlin}) should not be much larger than linear ({loss_low_linear})"
    )

    # Create log_alphas that give ~30 expected TUs (well above threshold)
    log_alphas_high = jnp.full((35,), 2.0)
    loss_high_linear = l0_loss(log_alphas_high, tu_threshold=None)
    loss_high_nonlin = l0_loss(log_alphas_high, tu_threshold=threshold)

    # Above threshold: nonlinear should be larger than linear
    assert loss_high_nonlin > loss_high_linear, (
        f"Above threshold, nonlinear loss ({loss_high_nonlin}) should be larger than linear ({loss_high_linear})"
    )

    # Verify the penalty ratio grows above threshold vs below
    ratio_low = loss_low_nonlin / loss_low_linear
    ratio_high = loss_high_nonlin / loss_high_linear
    assert ratio_high > ratio_low, (
        f"Penalty ratio should increase above threshold: {ratio_high:.3f} vs {ratio_low:.3f}"
    )


def test_get_tu_mask_single_tu():
    """Single TU masking."""
    tu_id_to_idx = {"tu_a": 0, "tu_b": 1}
    log_alpha_all = jnp.array([5.0, -5.0])  # tu_a on, tu_b off
    key = jax.random.key(42)

    # tu_a should be mostly on
    n_on = 0
    for i in range(100):
        mask_a = get_tu_mask_for_node(
            ["tu_a"], tu_id_to_idx, log_alpha_all, jax.random.fold_in(key, i)
        )
        n_on += int(mask_a > 0.5)
    assert n_on > 80, f"tu_a should be mostly on, got {n_on}/100"

    # tu_b should be mostly off
    n_on = 0
    for i in range(100):
        mask_b = get_tu_mask_for_node(
            ["tu_b"], tu_id_to_idx, log_alpha_all, jax.random.fold_in(key, i)
        )
        n_on += int(mask_b > 0.5)
    assert n_on < 20, f"tu_b should be mostly off, got {n_on}/100"


def test_get_tu_mask_multiple_tus():
    """Multiple TU masking (AND logic)."""
    tu_id_to_idx = {"tu_a": 0, "tu_b": 1}
    # Both on
    log_alpha_all = jnp.array([5.0, 5.0])
    key = jax.random.key(42)

    n_on = 0
    for i in range(100):
        mask = get_tu_mask_for_node(
            ["tu_a", "tu_b"], tu_id_to_idx, log_alpha_all, jax.random.fold_in(key, i)
        )
        n_on += int(mask > 0.5)
    assert n_on > 60, f"both TUs on should give mostly 1, got {n_on}/100"

    # One on, one off → should be mostly off
    log_alpha_all = jnp.array([5.0, -5.0])
    n_on = 0
    for i in range(100):
        mask = get_tu_mask_for_node(
            ["tu_a", "tu_b"], tu_id_to_idx, log_alpha_all, jax.random.fold_in(key, i)
        )
        n_on += int(mask > 0.5)
    assert n_on < 30, f"one TU off should give mostly 0, got {n_on}/100"


def test_get_tu_mask_empty():
    """Empty TU list should give mask=1."""
    mask = get_tu_mask_for_node([], {}, jnp.array([]), jax.random.key(42))
    assert mask == 1.0


def test_init_tu_log_alpha():
    """Initialize log_alpha parameters."""
    key = jax.random.key(42)
    log_alphas = init_tu_log_alpha(10, key, init_mean=2.0, init_std=0.5)

    assert log_alphas.shape == (10,)
    assert jnp.mean(log_alphas) > 1.0, "Mean should be positive"


def test_temperature_effect():
    """Lower temperature should make distribution harder (more 0s and 1s)."""
    key = jax.random.key(42)
    log_alpha = 0.0
    n_samples = 2000

    def count_extremes(temp):
        keys = jax.random.split(key, n_samples)
        samples = jax.vmap(lambda k: sample_hard_concrete(log_alpha, k, temperature=temp))(keys)
        n_zeros = int((samples == 0.0).sum())
        n_ones = int((samples == 1.0).sum())
        return n_zeros + n_ones

    extremes_warm = count_extremes(1.0)
    extremes_cold = count_extremes(0.1)

    assert extremes_cold > extremes_warm, (
        f"Lower temp should give more extremes: warm={extremes_warm}, cold={extremes_cold}"
    )


def test_coordinated_sampling():
    """Same tu_id should give same sample with same key."""
    tu_id_to_idx = {"tu_a": 0}
    log_alpha_all = jnp.array([0.0])
    key = jax.random.key(42)

    # Same TU ID + same key → same result
    mask1 = get_tu_mask_for_node(["tu_a"], tu_id_to_idx, log_alpha_all, key)
    mask2 = get_tu_mask_for_node(["tu_a"], tu_id_to_idx, log_alpha_all, key)
    np.testing.assert_equal(mask1, mask2)


# --- Tests for compute_input_masks with explicit is_multi_tu flag ---

from biocomp.tumasking import compute_input_masks  # noqa: E402


def test_compute_input_masks_single_tu_requires_1d():
    """is_multi_tu=False requires 1D tu_indices."""
    tu_indices = jnp.array([0, 1, 2])  # 1D - single TU per input
    tu_uniform = jnp.full((4,), 0.5)
    tu_log_alpha = jnp.zeros(4)

    masks = compute_input_masks(tu_indices, tu_uniform, tu_log_alpha, is_multi_tu=False)
    assert masks.shape == (3,), f"Expected (3,), got {masks.shape}"


def test_compute_input_masks_multi_tu_requires_2d():
    """is_multi_tu=True requires 2D tu_indices."""
    tu_indices = jnp.array([[0, 1], [1, 2], [2, 3]])  # 2D - multi TU per input
    tu_uniform = jnp.full((4,), 0.5)
    tu_log_alpha = jnp.zeros(4)

    masks = compute_input_masks(tu_indices, tu_uniform, tu_log_alpha, is_multi_tu=True)
    assert masks.shape == (3,), f"Expected (3,), got {masks.shape}"


def test_compute_input_masks_single_tu_rejects_2d():
    """is_multi_tu=False must reject 2D tu_indices."""
    tu_indices = jnp.array([[0, 1], [1, 2]])  # 2D - wrong for single TU
    tu_uniform = jnp.full((4,), 0.5)
    tu_log_alpha = jnp.zeros(4)

    with pytest.raises(AssertionError, match=r"is_multi_tu=False but tu_indices.ndim=2"):
        compute_input_masks(tu_indices, tu_uniform, tu_log_alpha, is_multi_tu=False)


def test_compute_input_masks_multi_tu_rejects_1d():
    """is_multi_tu=True must reject 1D tu_indices."""
    tu_indices = jnp.array([0, 1, 2])  # 1D - wrong for multi TU
    tu_uniform = jnp.full((4,), 0.5)
    tu_log_alpha = jnp.zeros(4)

    with pytest.raises(AssertionError, match=r"is_multi_tu=True but tu_indices.ndim=1"):
        compute_input_masks(tu_indices, tu_uniform, tu_log_alpha, is_multi_tu=True)


def test_compute_input_masks_none_inputs_returns_ones():
    """When tu_uniform or tu_log_alpha is None, returns all ones."""
    tu_indices_1d = jnp.array([0, 1, 2])
    tu_indices_2d = jnp.array([[0, 1], [1, 2], [2, 3]])

    # None tu_uniform
    masks = compute_input_masks(tu_indices_1d, None, jnp.zeros(4), is_multi_tu=False)
    np.testing.assert_array_equal(masks, jnp.ones(3))

    # None tu_log_alpha
    masks = compute_input_masks(tu_indices_2d, jnp.full((4,), 0.5), None, is_multi_tu=True)
    np.testing.assert_array_equal(masks, jnp.ones(3))


# --- Tests for asymmetric_l0_loss ---


def test_asymmetric_l0_loss_zero_at_zero():
    """f(0) should be 0."""
    log_alpha = jnp.full((10,), -10.0)  # all TUs fully disabled
    loss = asymmetric_l0_loss(log_alpha, threshold=5.0)
    assert float(loss) < 1e-6, f"loss at zero TUs should be ~0, got {loss}"


def test_asymmetric_l0_loss_at_threshold():
    """f(threshold) ≈ threshold (anchor point)."""
    threshold = 5.0
    log_alpha = jnp.full((10,), 1.0)  # sigmoid(1) ≈ 0.73, after floor ~0.66
    count = jnp.sum(l0_penalty(log_alpha))
    assert count > 0, "sanity check: expected positive count"

    # create log_alpha that gives count ≈ threshold
    # need to tune: at floor=0.2, per_tu = (sigmoid(la) - 0.2) / 0.8
    # for count=5 with 10 TUs, need per_tu=0.5, so sigmoid(la)=0.6 -> la≈0.4
    la_tuned = jnp.full((10,), 0.4)
    count_tuned = float(jnp.sum(l0_penalty(la_tuned)))
    loss_tuned = float(asymmetric_l0_loss(la_tuned, threshold=threshold))

    # at count=threshold, loss ≈ threshold (within blend transition zone)
    assert abs(loss_tuned - count_tuned) < threshold * 0.5, (
        f"at count≈{count_tuned:.2f}, loss={loss_tuned:.2f} should be close to count"
    )


def test_asymmetric_l0_loss_sublinear_below():
    """Below threshold, marginal penalty should be less than 1 (sublinear growth)."""
    threshold = 12.0

    # test two points below threshold and verify marginal rate < 1
    la_low = jnp.full((4,), 2.0)
    la_high = jnp.full((8,), 2.0)
    loss_low = float(asymmetric_l0_loss(la_low, threshold=threshold))
    loss_high = float(asymmetric_l0_loss(la_high, threshold=threshold))
    count_low = float(jnp.sum(l0_penalty(la_low)))
    count_high = float(jnp.sum(l0_penalty(la_high)))

    # marginal rate = d(loss)/d(count)
    marginal_rate = (loss_high - loss_low) / (count_high - count_low + 1e-6)
    assert marginal_rate < 1.0, (
        f"below threshold: marginal rate={marginal_rate:.4f} should be < 1 (sublinear)"
    )


def test_asymmetric_l0_loss_superlinear_above():
    """Above threshold, marginal penalty should be greater than 1 (superlinear growth)."""
    threshold = 5.0

    # test two points above threshold and verify marginal rate > 1
    la_low = jnp.full((12,), 2.0)  # ~10 expected TUs
    la_high = jnp.full((20,), 2.0)  # ~17 expected TUs
    loss_low = float(asymmetric_l0_loss(la_low, threshold=threshold))
    loss_high = float(asymmetric_l0_loss(la_high, threshold=threshold))
    count_low = float(jnp.sum(l0_penalty(la_low)))
    count_high = float(jnp.sum(l0_penalty(la_high)))

    # marginal rate = d(loss)/d(count)
    marginal_rate = (loss_high - loss_low) / (count_high - count_low + 1e-6)
    assert marginal_rate > 1.0, (
        f"above threshold: marginal rate={marginal_rate:.4f} should be > 1 (superlinear)"
    )


def test_asymmetric_l0_loss_differentiable():
    """Gradient should exist and be finite."""
    log_alpha = jnp.full((10,), 0.5)
    grad_fn = jax.grad(lambda la: asymmetric_l0_loss(la, threshold=5.0).sum())
    grad = grad_fn(log_alpha)
    assert jnp.all(jnp.isfinite(grad)), f"gradient should be finite, got {grad}"
    assert jnp.any(grad != 0), "gradient should be nonzero"


def test_asymmetric_l0_loss_smooth_transition():
    """Loss should be smooth (no jumps) around threshold."""
    threshold = 10.0

    # compute losses at a few points around threshold
    losses = []
    for c in [8, 9, 10, 11, 12]:
        # tune log_alpha to get target count
        # per_tu ≈ (sigmoid(la) - 0.2) / 0.8, for c TUs with 20 elements: per_tu = c/20
        target_per_tu = c / 20.0
        target_sigmoid = target_per_tu * 0.8 + 0.2
        la_val = jnp.log(target_sigmoid / (1 - target_sigmoid))
        la = jnp.full((20,), float(la_val))
        losses.append(float(asymmetric_l0_loss(la, threshold=threshold)))

    # check monotonicity
    for i in range(len(losses) - 1):
        assert losses[i] <= losses[i + 1] + 0.1, f"loss should be non-decreasing: {losses}"


# --- Tests for decode_latent_tu_masking ---


def test_decode_latent_tu_masking_shape():
    """Output shape should be (n_tus,)."""
    latent_dim, hidden_dim, n_tus = 8, 16, 10
    z = jnp.zeros(latent_dim)
    W1 = jnp.zeros((hidden_dim, latent_dim))
    b1 = jnp.zeros(hidden_dim)
    W2 = jnp.zeros((n_tus, hidden_dim))
    b2 = jnp.ones(n_tus) * 2.0

    log_alpha = decode_latent_tu_masking(z, W1, b1, W2, b2)
    assert log_alpha.shape == (n_tus,), f"expected ({n_tus},), got {log_alpha.shape}"


def test_decode_latent_tu_masking_at_zero():
    """decode(0) ≈ b2 (since z=0 -> h=gelu(0)=0 -> W2@h=0 -> output=b2)."""
    latent_dim, hidden_dim, n_tus = 8, 16, 10
    z = jnp.zeros(latent_dim)
    W1 = jax.random.normal(jax.random.key(0), (hidden_dim, latent_dim)) * 0.1
    b1 = jnp.zeros(hidden_dim)
    W2 = jax.random.normal(jax.random.key(1), (n_tus, hidden_dim)) * 0.1
    b2 = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, -3.0, 0.0, 0.5])

    log_alpha = decode_latent_tu_masking(z, W1, b1, W2, b2)
    np.testing.assert_allclose(log_alpha, b2, atol=0.1)


def test_decode_latent_tu_masking_gradient_flow():
    """Gradients should flow through the MLP."""
    latent_dim, hidden_dim, n_tus = 8, 16, 10
    z = jax.random.normal(jax.random.key(0), (latent_dim,)) * 0.1
    W1 = jax.random.normal(jax.random.key(1), (hidden_dim, latent_dim)) * 0.1
    b1 = jnp.zeros(hidden_dim)
    W2 = jax.random.normal(jax.random.key(2), (n_tus, hidden_dim)) * 0.1
    b2 = jnp.zeros(n_tus)

    def loss_fn(z, W1, b1, W2, b2):
        la = decode_latent_tu_masking(z, W1, b1, W2, b2)
        return jnp.sum(la**2)

    grads = jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4))(z, W1, b1, W2, b2)

    for _i, (name, g) in enumerate(zip(["z", "W1", "b1", "W2", "b2"], grads, strict=False)):
        assert jnp.all(jnp.isfinite(g)), f"grad_{name} should be finite"
        # z, W1, W2 should have nonzero gradients; b1 may be zero if gelu(0)=0
        if name in ["z", "W1", "W2", "b2"]:
            assert jnp.any(g != 0), f"grad_{name} should be nonzero"


def test_decode_latent_tu_masking_vmap():
    """Should work with vmap over batch dimension."""
    batch, latent_dim, hidden_dim, n_tus = 4, 8, 16, 10
    z = jax.random.normal(jax.random.key(0), (batch, latent_dim)) * 0.1
    W1 = jax.random.normal(jax.random.key(1), (batch, hidden_dim, latent_dim)) * 0.1
    b1 = jnp.zeros((batch, hidden_dim))
    W2 = jax.random.normal(jax.random.key(2), (batch, n_tus, hidden_dim)) * 0.1
    b2 = jnp.zeros((batch, n_tus))

    log_alpha = jax.vmap(decode_latent_tu_masking)(z, W1, b1, W2, b2)
    assert log_alpha.shape == (batch, n_tus), f"expected ({batch}, {n_tus}), got {log_alpha.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
