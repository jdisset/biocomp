import numpy as np

from biocomp.designutils import compute_grid_metrics, side_by_side_txt_plot


def _contrast_only_weights(w_contrast: float = 50.0) -> dict:
    return {
        "w_sinkhorn": 0.0,
        "w_lncc": 0.0,
        "w_mse": 0.0,
        "w_rmse": 0.0,
        "w_simse": 0.0,
        "w_spectral": 0.0,
        "w_gradient": 0.0,
        "w_zncc": 0.0,
        "w_contrast": float(w_contrast),
        "contrast_mode": "target_gap",
        "contrast_focus": 8.0,
    }


def test_compute_grid_metrics_total_includes_contrast_when_nonzero():
    target = np.linspace(0.05, 0.45, 32 * 32, dtype=np.float32).reshape(32, 32)
    pred = np.linspace(0.05, 0.06, 32 * 32, dtype=np.float32).reshape(32, 32)
    weights = _contrast_only_weights(50.0)

    metrics = compute_grid_metrics(target, pred, loss_weights=weights)

    assert metrics["contrast"] > 0.0
    assert metrics["contrast_weighted"] > 0.0
    assert "contrast_target_gap" in metrics
    assert "contrast_pred_gap" in metrics
    assert "pred_q95_q05_gap" in metrics
    assert metrics["pred_q95_q05_gap"] > 0.0
    np.testing.assert_allclose(
        metrics["weighted_total"],
        metrics["contrast_weighted"],
        rtol=1e-6,
        atol=1e-6,
    )


def test_side_by_side_txt_plot_shows_zero_contrast_row_when_weight_active():
    target = np.linspace(0.05, 0.45, 32 * 32, dtype=np.float32).reshape(32, 32)
    pred = target.copy()  # contrast deficit is exactly zero
    weights = _contrast_only_weights(50.0)

    txt, metrics = side_by_side_txt_plot(
        target,
        pred,
        height=8,
        width=16,
        loss_weights=weights,
        compute_metrics=True,
    )

    assert abs(metrics["contrast"]) < 1e-12
    assert abs(metrics["contrast_weighted"]) < 1e-12
    assert "contrast" in txt
