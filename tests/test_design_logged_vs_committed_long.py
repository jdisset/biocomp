#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Regression test: logged vs committed predictions must match with use_latent_ratios=True."""

import pytest
import numpy as np
from pathlib import Path
import os

import dracon as dr

from biocomp.design import (
    DesignManager,
    DesignConfig,
    start as design_start,
)
from biocomp.design_targets import SVGTarget
from biocomp.designloss import grid_distance_loss
from biocomp.design_targets import LatticeSampling
from biocomp.network import recipe_to_networks
from biocomp.recipe import Recipe
import biocomp.biorules as br
import biocomp.compute as cmp
from biocomp.jaxutils import tree_get
from biocomp.utils import PartialFunction

RESOURCES_DIR = Path(__file__).parent / "resources"


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


def load_target(target_name: str = "MIT_T") -> SVGTarget:
    target_path = RESOURCES_DIR / "designs" / f"{target_name}.svg"
    if not target_path.exists():
        pytest.skip(f"Target file not found: {target_path}")
    return SVGTarget(path=str(target_path), name=target_name)


@pytest.fixture
def design_setup_long():
    model = load_model()
    recipe_path = RESOURCES_DIR / "design/architectures/T_2_fully_unlocked.yaml"
    recipe = load_recipe(str(recipe_path))
    networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True, inversion_mode="all")
    assert len(networks) > 0, "no networks built from recipe"
    return {"model": model, "networks": networks, "target": load_target("MIT_T"), "recipe": recipe}


def test_logged_vs_committed_long_optimization(design_setup_long):
    model, networks, target = (
        design_setup_long["model"],
        design_setup_long["networks"],
        design_setup_long["target"],
    )

    res = (40, 40)
    dmanager = DesignManager(
        targets=[target],
        networks=networks,
        sampling=LatticeSampling(resolution=res, jitter_std=0.01),
        enable_tu_masking=True,
    )

    from biocomp.tumasking_strategy import TUMaskingMode
    from biocomp.design import TUMaskingParams

    dconf = DesignConfig(
        n_epochs=10,
        n_batches_per_epoch=64,
        n_replicates=1,
        reshuffle_batches=False,
        batch_size=1,
        batches_per_step=1,
        use_latent_ratios=True,
        tu_masking=TUMaskingParams(mode=TUMaskingMode.DIRECT, init_mean=2.0, init_std=0.1),
        seed=1920764948,
        loss_function=PartialFunction(
            func=grid_distance_loss,
            kwargs={
                "w_sinkhorn": 0.1,
                "w_lncc": 0.0,
                "w_mse": 0.0,
                "w_rmse": 0.05,
                "w_spectral": 0.0,
                "w_simse": 0.1,
                "w_zncc": 0.5,
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

    final_params, _, step_history = design_start(
        dmanager=dmanager, dconf=dconf, model=model
    )

    logged_yhatdep = step_history.get("yhatdep")
    assert logged_yhatdep is not None, "no yhatdep in step_history"
    logged_yhatdep = np.asarray(logged_yhatdep)

    xres, yres = dmanager.grid_resolution
    X_lat, Y_target = target.get_lattice(resolution=(xres, yres), seed=0)
    logged_pred = logged_yhatdep[0, 0, :, 0, 0].reshape(yres, xres)
    Y_target_grid = np.asarray(Y_target).reshape(yres, xres)
    logged_corr = float(np.corrcoef(Y_target_grid.flatten(), logged_pred.flatten())[0, 1])

    stack = cmp.ComputeStack(networks=dmanager.networks)
    stack.build(model.compute_config, enable_tu_masking=True)
    bparams = tree_get(final_params, (0, 0))
    committed_networks = stack.commit(bparams)
    assert len(committed_networks) > 0, "no networks from commit"
    committed_network = committed_networks[0]

    from biocomptools.modelmodel import NetworkModel
    from biocomptools.toollib.networkprediction import NetworkPrediction

    nm = NetworkModel(model=model, network=committed_network)
    pred = NetworkPrediction(predict_at=[X_lat], network_model=nm, already_latent=True)
    committed_pred = np.asarray(pred.get_data(rescale_latent=False)[0].y).reshape(yres, xres)
    committed_corr = float(np.corrcoef(Y_target_grid.flatten(), committed_pred.flatten())[0, 1])

    delta = abs(logged_corr - committed_corr)
    assert delta < 0.10, (
        f"logged vs committed correlation mismatch: logged={logged_corr:.4f}, committed={committed_corr:.4f}, "
        f"delta={delta:.4f}, logged_range=[{logged_pred.min():.4f}, {logged_pred.max():.4f}], "
        f"committed_range=[{committed_pred.min():.4f}, {committed_pred.max():.4f}]"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
