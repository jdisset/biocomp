#!/usr/bin/env python
"""Test that logged predictions during design optimization match committed network predictions.

This test reproduces the bug where:
- During optimization: logged predictions show good correlation with target (e.g., 0.91)
- After commit: committed network predictions show DIFFERENT correlation with target (e.g., 0.49)

The key metric is CORRELATION WITH TARGET - this must match between logged and committed.
"""
import pytest
import numpy as np
from pathlib import Path

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
    import os
    model_path = os.environ.get("BIOCOMP_DESIGNER_MODEL")
    if not model_path or not Path(model_path).exists():
        pytest.skip("BIOCOMP_DESIGNER_MODEL not set or doesn't exist")

    from biocomptools.modelmodel import BiocompModel
    return BiocompModel.load(model_path)


def load_target(target_name: str = "MIT_T") -> Target:
    from biocomp.design import Target
    import os

    biocomp_root = os.environ.get("BIOCOMP_ROOT", "")
    target_path = Path(biocomp_root) / "Designs" / f"{target_name}.svg"

    if not target_path.exists():
        pytest.skip(f"Target file not found: {target_path}")

    return Target(path=str(target_path), name=target_name)


@pytest.fixture
def design_setup():
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


def test_logged_vs_committed_correlation_with_target(design_setup):
    """Test that correlation-with-target matches between logged and committed predictions.

    The bug: optimization shows Corr=0.91, but after commit the same network shows Corr=0.49.
    This test MUST FAIL if the bug exists.
    """
    model = design_setup["model"]
    networks = design_setup["networks"]
    target = design_setup["target"]

    res = (32, 32)
    dmanager = DesignManager(
        targets=[target],
        networks=networks,
        sampling=LatticeSampling(resolution=res, jitter_std=0.01),
        enable_tu_masking=False,
    )

    dconf = DesignConfig(
        n_epochs=5,
        n_batches_per_epoch=32,
        n_replicates=1,
        reshuffle_batches=False,
        batch_size=1,
        batches_per_step=1,
        use_latent_ratios=True,
    )

    final_params, loss_history, step_history = design_start(
        dmanager=dmanager,
        dconf=dconf,
        model=model,
        loggers=None,
    )

    logged_yhatdep = step_history.get("yhatdep")
    assert logged_yhatdep is not None, "No yhatdep in final step history"
    logged_yhatdep = np.asarray(logged_yhatdep)

    xres, yres = dmanager.grid_resolution
    X_lat, Y_target = target.get_lattice(resolution=(xres, yres), seed=0)

    logged_pred = logged_yhatdep[0, 0, :, 0, 0].reshape(yres, xres)
    Y_target_grid = np.asarray(Y_target).reshape(yres, xres)

    logged_corr = float(np.corrcoef(Y_target_grid.flatten(), logged_pred.flatten())[0, 1])

    # Commit and get prediction from committed network
    stack = cmp.ComputeStack(networks=dmanager.networks)
    stack.build(model.compute_config, enable_tu_masking=dmanager.enable_tu_masking)

    bparams = tree_get(final_params, (0, 0))
    committed_networks = stack.commit(bparams)

    assert len(committed_networks) > 0, "No networks returned from commit"
    committed_network = committed_networks[0]

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
    print("CORRELATION WITH TARGET COMPARISON")
    print("=" * 70)
    print(f"  Logged correlation with target:    {logged_corr:.4f}")
    print(f"  Committed correlation with target: {committed_corr:.4f}")
    print(f"  Delta:                             {abs(logged_corr - committed_corr):.4f}")
    print("-" * 70)
    print(f"  Logged prediction range:    [{logged_pred.min():.4f}, {logged_pred.max():.4f}]")
    print(f"  Committed prediction range: [{committed_pred.min():.4f}, {committed_pred.max():.4f}]")

    # The key assertion: correlation with target must be approximately the same
    tolerance = 0.05  # 5% tolerance for correlation difference
    delta = abs(logged_corr - committed_corr)

    assert delta < tolerance, (
        f"BUG DETECTED: Correlation with target differs between logged and committed!\n"
        f"  Logged correlation:    {logged_corr:.4f}\n"
        f"  Committed correlation: {committed_corr:.4f}\n"
        f"  Delta: {delta:.4f} (tolerance: {tolerance})\n"
        f"This means the committed network produces different predictions than during optimization."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
