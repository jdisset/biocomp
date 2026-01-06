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
    extract_tu_ids_from_network,
    build_tu_id_mapping,
    init_tu_log_alpha,
    DEFAULT_GAMMA,
    DEFAULT_ZETA,
    L0_PENALTY_FLOOR_PROB,
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
            f"P(z=0) should decrease with log_alpha: {list(zip(log_alphas, p_zeros))}"
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
    assert n_exact_zeros > n_samples * 0.8, f"Expected mostly zeros, got {n_exact_zeros}/{n_samples}"

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
    assert penalty_well_below == 0.0, f"Well below floor, penalty should be 0, got {penalty_well_below}"

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
        mask_a = get_tu_mask_for_node(["tu_a"], tu_id_to_idx, log_alpha_all, jax.random.fold_in(key, i))
        n_on += int(mask_a > 0.5)
    assert n_on > 80, f"tu_a should be mostly on, got {n_on}/100"

    # tu_b should be mostly off
    n_on = 0
    for i in range(100):
        mask_b = get_tu_mask_for_node(["tu_b"], tu_id_to_idx, log_alpha_all, jax.random.fold_in(key, i))
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
        mask = get_tu_mask_for_node(["tu_a", "tu_b"], tu_id_to_idx, log_alpha_all, jax.random.fold_in(key, i))
        n_on += int(mask > 0.5)
    assert n_on > 60, f"both TUs on should give mostly 1, got {n_on}/100"

    # One on, one off → should be mostly off
    log_alpha_all = jnp.array([5.0, -5.0])
    n_on = 0
    for i in range(100):
        mask = get_tu_mask_for_node(["tu_a", "tu_b"], tu_id_to_idx, log_alpha_all, jax.random.fold_in(key, i))
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

from biocomp.tumasking import compute_input_masks


def test_compute_input_masks_single_tu_requires_1d():
    """is_multi_tu=False requires 1D tu_indices."""
    tu_indices = jnp.array([0, 1, 2])  # 1D - single TU per input
    tu_uniform = jnp.full((4,), 0.5)
    tu_log_alpha = jnp.zeros(4)

    masks = compute_input_masks(
        tu_indices, tu_uniform, tu_log_alpha, is_multi_tu=False
    )
    assert masks.shape == (3,), f"Expected (3,), got {masks.shape}"


def test_compute_input_masks_multi_tu_requires_2d():
    """is_multi_tu=True requires 2D tu_indices."""
    tu_indices = jnp.array([[0, 1], [1, 2], [2, 3]])  # 2D - multi TU per input
    tu_uniform = jnp.full((4,), 0.5)
    tu_log_alpha = jnp.zeros(4)

    masks = compute_input_masks(
        tu_indices, tu_uniform, tu_log_alpha, is_multi_tu=True
    )
    assert masks.shape == (3,), f"Expected (3,), got {masks.shape}"


def test_compute_input_masks_single_tu_rejects_2d():
    """is_multi_tu=False must reject 2D tu_indices."""
    tu_indices = jnp.array([[0, 1], [1, 2]])  # 2D - wrong for single TU
    tu_uniform = jnp.full((4,), 0.5)
    tu_log_alpha = jnp.zeros(4)

    with pytest.raises(AssertionError, match=r"is_multi_tu=False but tu_indices.ndim=2"):
        compute_input_masks(
            tu_indices, tu_uniform, tu_log_alpha, is_multi_tu=False
        )


def test_compute_input_masks_multi_tu_rejects_1d():
    """is_multi_tu=True must reject 1D tu_indices."""
    tu_indices = jnp.array([0, 1, 2])  # 1D - wrong for multi TU
    tu_uniform = jnp.full((4,), 0.5)
    tu_log_alpha = jnp.zeros(4)

    with pytest.raises(AssertionError, match=r"is_multi_tu=True but tu_indices.ndim=1"):
        compute_input_masks(
            tu_indices, tu_uniform, tu_log_alpha, is_multi_tu=True
        )


def test_compute_input_masks_none_inputs_returns_ones():
    """When tu_uniform or tu_log_alpha is None, returns all ones."""
    tu_indices_1d = jnp.array([0, 1, 2])
    tu_indices_2d = jnp.array([[0, 1], [1, 2], [2, 3]])

    # None tu_uniform
    masks = compute_input_masks(
        tu_indices_1d, None, jnp.zeros(4), is_multi_tu=False
    )
    np.testing.assert_array_equal(masks, jnp.ones(3))

    # None tu_log_alpha
    masks = compute_input_masks(
        tu_indices_2d, jnp.full((4,), 0.5), None, is_multi_tu=True
    )
    np.testing.assert_array_equal(masks, jnp.ones(3))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
