#!/usr/bin/env python
"""Test that reproduces the logged vs committed correlation discrepancy.

Bug: After longer optimization runs with use_latent_ratios=True:
- During optimization: Corr ~0.92 with target (T shape)
- After commit: Corr ~0.28 with target (completely different shape)

The discrepancy grows with more optimization steps.
"""
import pytest
import numpy as np
from pathlib import Path
import os

import dracon as dr

from biocomp.design import DesignManager, DesignConfig, start as design_start, Target
from biocomp.design_targets import LatticeSampling
from biocomp.network import recipe_to_networks
from biocomp.recipe import Recipe
import biocomp.biorules as br
import biocomp.compute as cmp
from biocomp.jaxutils import tree_get


def load_recipe(path: str) -> Recipe:
    data = dr.load(path)
    if hasattr(data, "keys") and "recipe" in data:
        return data["recipe"]
    if isinstance(data, Recipe):
        return data
    raise ValueError(f"Could not find Recipe in {path}")


def load_model():
    model_path = os.environ.get("BIOCOMP_DESIGNER_MODEL")
    if not model_path or not Path(model_path).exists():
        pytest.skip("BIOCOMP_DESIGNER_MODEL not set or doesn't exist")

    from biocomptools.modelmodel import BiocompModel
    return BiocompModel.load(model_path)


def load_target(target_name: str = "MIT_T") -> Target:
    biocomp_root = os.environ.get("BIOCOMP_ROOT", "")
    target_path = Path(biocomp_root) / "Designs" / f"{target_name}.svg"

    if not target_path.exists():
        pytest.skip(f"Target file not found: {target_path}")

    return Target(path=str(target_path), name=target_name)


@pytest.fixture
def design_setup_long():
    """Setup for longer optimization that triggers the bug."""
    model = load_model()

    recipe_path = "biocomp-jobs/design/architectures/T_2_fully_unlocked.yaml"
    recipe = load_recipe(recipe_path)

    networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True, inversion_mode="all")
    assert len(networks) > 0, "No networks built from recipe"

    target = load_target("MIT_T")

    return {
        "model": model,
        "networks": networks,
        "target": target,
        "recipe": recipe,
    }


def test_logged_vs_committed_long_optimization(design_setup_long):
    """Test that reproduces the bug with longer optimization.

    The bug: with use_latent_ratios=True and longer optimization,
    the logged prediction shows good correlation with target (~0.92),
    but the committed network prediction shows completely different
    correlation (~0.28).

    This test MUST FAIL to demonstrate the bug exists.
    """
    model = design_setup_long["model"]
    networks = design_setup_long["networks"]
    target = design_setup_long["target"]

    # Use 40x40 resolution like the failing production run
    res = (40, 40)
    dmanager = DesignManager(
        targets=[target],
        networks=networks,
        sampling=LatticeSampling(resolution=res, jitter_std=0.01),
        enable_tu_masking=True,  # Like production
    )

    # Key parameters from user's failing run with exact loss function weights
    from biocomp.utils import PartialFunction
    from biocomp.design import grid_distance_loss

    dconf = DesignConfig(
        n_epochs=10,
        n_batches_per_epoch=64,
        n_replicates=1,
        reshuffle_batches=False,
        batch_size=1,
        batches_per_step=1,
        use_latent_ratios=True,  # KEY: This causes the bug
        tu_log_alpha_init_mean=2.0,
        tu_log_alpha_init_std=0.1,
        seed=1920764948,
        # Exact loss function weights from user's failing run
        loss_function=PartialFunction(
            func=grid_distance_loss,
            kwargs={
                "w_sinkhorn": 0.1,
                "w_lncc": 0.0,
                "w_mse": 0.0,
                "w_rmse": 0.05,
                "w_spectral": 0.0,
                "w_simse": 0.1,
                "w_zncc": 0.5,  # High weight on correlation
                "w_contrast": 0.05,
                "w_gradient": 0.0,
                "eps_sinkhorn": 0.1,
                "n_sinkhorn_iters": 50,
                "lambda_tucount": 0.0,
                "max_tus_per_cotx": 5,
                "lambda_spread": 0.0,
                "max_ratio": 120.0,
                "tu_n_samples": 1,
                "lambda_ern_tying": 0.0,
                "lambda_coupling": 0.0,
                "min_ratio_threshold": 0.005,
            },
        ),
    )

    print("\n" + "=" * 70)
    print("Running LONG optimization to reproduce logged vs committed bug")
    print(f"  Epochs: {dconf.n_epochs}")
    print(f"  Batches per epoch: {dconf.n_batches_per_epoch}")
    print(f"  Total steps: {int(dconf.n_epochs * dconf.n_batches_per_epoch)}")
    print(f"  use_latent_ratios: {dconf.use_latent_ratios}")
    print("=" * 70)

    final_params, loss_history, step_history = design_start(
        dmanager=dmanager,
        dconf=dconf,
        model=model,
        loggers=None,
    )

    # Get logged prediction from final step
    logged_yhatdep = step_history.get("yhatdep")
    assert logged_yhatdep is not None, "No yhatdep in final step history"
    logged_yhatdep = np.asarray(logged_yhatdep)

    xres, yres = dmanager.grid_resolution
    X_lat, Y_target = target.get_lattice(resolution=(xres, yres), seed=0)

    logged_pred = logged_yhatdep[0, 0, :, 0, 0].reshape(yres, xres)
    Y_target_grid = np.asarray(Y_target).reshape(yres, xres)

    logged_corr = float(np.corrcoef(Y_target_grid.flatten(), logged_pred.flatten())[0, 1])

    # Commit using NEW stack (like production does)
    stack = cmp.ComputeStack(networks=dmanager.networks)
    stack.build(model.compute_config, enable_tu_masking=True)

    bparams = tree_get(final_params, (0, 0))
    committed_networks = stack.commit(bparams)

    assert len(committed_networks) > 0, "No networks returned from commit"
    committed_network = committed_networks[0]

    # Get prediction from committed network using NetworkModel (like production)
    from biocomptools.modelmodel import NetworkModel
    from biocomptools.toollib.networkprediction import NetworkPrediction

    nm = NetworkModel(model=model, network=committed_network)
    pred = NetworkPrediction(
        predict_at=[X_lat],
        network_model=nm,
        already_latent=True,
    )
    data = pred.get_data(rescale_latent=False)[0]
    committed_pred = np.asarray(data.y).reshape(yres, xres)

    committed_corr = float(np.corrcoef(Y_target_grid.flatten(), committed_pred.flatten())[0, 1])

    print("\n" + "=" * 70)
    print("CORRELATION WITH TARGET COMPARISON (LONG OPTIMIZATION)")
    print("=" * 70)
    print(f"  Logged correlation with target:    {logged_corr:.4f}")
    print(f"  Committed correlation with target: {committed_corr:.4f}")
    print(f"  Delta:                             {abs(logged_corr - committed_corr):.4f}")
    print("-" * 70)
    print(f"  Logged prediction range:    [{logged_pred.min():.4f}, {logged_pred.max():.4f}]")
    print(f"  Committed prediction range: [{committed_pred.min():.4f}, {committed_pred.max():.4f}]")
    print("=" * 70)

    # The key assertion: correlation with target must be approximately the same
    tolerance = 0.10  # 10% tolerance
    delta = abs(logged_corr - committed_corr)

    assert delta < tolerance, (
        f"BUG REPRODUCED: Correlation with target differs between logged and committed!\n"
        f"  Logged correlation:    {logged_corr:.4f}\n"
        f"  Committed correlation: {committed_corr:.4f}\n"
        f"  Delta: {delta:.4f} (tolerance: {tolerance})\n"
        f"  Logged pred range:    [{logged_pred.min():.4f}, {logged_pred.max():.4f}]\n"
        f"  Committed pred range: [{committed_pred.min():.4f}, {committed_pred.max():.4f}]\n"
        f"This means the committed network produces COMPLETELY DIFFERENT predictions than during optimization."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
