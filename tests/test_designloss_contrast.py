import numpy as np
import jax
import jax.numpy as jnp

from biocomp.designloss import compute_grid_losses, _compute_contrast_loss


def _contrast_v2_kwargs():
    return {
        "gap_weight": 1.0,
        "hi_weight": 1.0,
        "lo_weight": 1.0,
        "barrier_weight": 1.0,
        "min_frac": 1.0,
    }


def test_compute_grid_losses_includes_contrast_weight_in_total():
    y_target = jnp.linspace(0.05, 0.45, 32 * 32, dtype=jnp.float32).reshape(32, 32)
    y_pred = jnp.linspace(0.04, 0.06, 32 * 32, dtype=jnp.float32).reshape(32, 32)

    base = compute_grid_losses(
        y_pred,
        y_target,
        w_sinkhorn=0.0,
        w_lncc=0.0,
        w_mse=0.0,
        w_rmse=0.0,
        w_simse=0.0,
        w_spectral=0.0,
        w_gradient=0.0,
        w_zncc=0.0,
        w_contrast=0.0,
    )
    unit = compute_grid_losses(
        y_pred,
        y_target,
        w_sinkhorn=0.0,
        w_lncc=0.0,
        w_mse=0.0,
        w_rmse=0.0,
        w_simse=0.0,
        w_spectral=0.0,
        w_gradient=0.0,
        w_zncc=0.0,
        w_contrast=1.0,
    )
    boosted = compute_grid_losses(
        y_pred,
        y_target,
        w_sinkhorn=0.0,
        w_lncc=0.0,
        w_mse=0.0,
        w_rmse=0.0,
        w_simse=0.0,
        w_spectral=0.0,
        w_gradient=0.0,
        w_zncc=0.0,
        w_contrast=5.0,
    )

    expected_delta = 5.0 * unit.contrast
    np.testing.assert_allclose(boosted.total - base.total, expected_delta, rtol=1e-6, atol=1e-6)


def test_target_gap_contrast_gradient_is_dense_for_nonflat_prediction():
    y_target = jnp.linspace(0.05, 0.45, 32 * 32, dtype=jnp.float32).reshape(32, 32)
    y_pred = jnp.linspace(0.04, 0.06, 32 * 32, dtype=jnp.float32).reshape(32, 32)

    def contrast_only(yh: jnp.ndarray):
        return _compute_contrast_loss(
            yh,
            y_target,
            mode="target_gap",
            focus=8.0,
            **_contrast_v2_kwargs(),
        )

    grads = jax.grad(contrast_only)(y_pred)
    nonzero = int(np.count_nonzero(np.abs(np.asarray(grads)) > 1e-12))
    assert nonzero > (y_pred.size // 2), f"Expected dense gradient support, got {nonzero} nonzero entries"


def test_target_gap_contrast_gradient_nonzero_for_flat_prediction():
    y_target = jnp.linspace(0.05, 0.45, 32 * 32, dtype=jnp.float32).reshape(32, 32)
    y_pred = jnp.full((32, 32), 0.05, dtype=jnp.float32)

    def contrast_only(yh: jnp.ndarray):
        return _compute_contrast_loss(
            yh,
            y_target,
            mode="target_gap",
            focus=8.0,
            **_contrast_v2_kwargs(),
        )

    grads = jax.grad(contrast_only)(y_pred)
    assert np.count_nonzero(np.asarray(grads)) > 0


def test_legacy_range_relu_contrast_gradient_zero_for_flat_prediction():
    y_target = jnp.linspace(0.05, 0.45, 32 * 32, dtype=jnp.float32).reshape(32, 32)
    y_pred = jnp.full((32, 32), 0.05, dtype=jnp.float32)

    def contrast_only(yh: jnp.ndarray):
        return _compute_contrast_loss(
            yh,
            y_target,
            mode="range_relu",
            focus=8.0,
            **_contrast_v2_kwargs(),
        )

    grads = jax.grad(contrast_only)(y_pred)
    assert np.count_nonzero(np.asarray(grads)) == 0


def test_compute_grid_losses_accepts_contrast_mode_and_focus_kwargs():
    y_target = jnp.linspace(0.05, 0.45, 32 * 32, dtype=jnp.float32).reshape(32, 32)
    y_pred = jnp.full((32, 32), 0.05, dtype=jnp.float32)

    out = compute_grid_losses(
        y_pred,
        y_target,
        w_sinkhorn=0.0,
        w_lncc=0.0,
        w_mse=0.0,
        w_rmse=0.0,
        w_simse=0.0,
        w_spectral=0.0,
        w_gradient=0.0,
        w_zncc=0.0,
        w_contrast=1.0,
        contrast_mode="target_gap",
        contrast_focus=12.0,
    )
    assert out.contrast > 0.0


def test_target_gap_v2_penalizes_low_dynamic_range_prediction():
    y_target = jnp.linspace(0.05, 0.45, 32 * 32, dtype=jnp.float32).reshape(32, 32)
    y_pred = jnp.linspace(0.04, 0.08, 32 * 32, dtype=jnp.float32).reshape(32, 32)

    c = _compute_contrast_loss(
        y_pred,
        y_target,
        mode="target_gap",
        focus=8.0,
        **_contrast_v2_kwargs(),
    )
    assert float(c) > 0.0
