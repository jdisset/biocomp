#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Regression test: logged vs committed predictions must match with use_latent_ratios=True."""

import pytest
import numpy as np
from pathlib import Path
import os

import dracon as dr

from biocomp.design import DesignManager, DesignConfig, start as design_start
from biocomp.design_targets import SVGTarget
from biocomp.design_targets import LatticeSampling
from biocomp.network import recipe_to_networks
from biocomp.recipe import Recipe
import biocomp.biorules as br
import biocomp.compute as cmp
from biocomp.jaxutils import tree_get

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
def design_setup():
    model = load_model()
    recipe_path = RESOURCES_DIR / "design/architectures/T_2_fully_unlocked.yaml"
    recipe = load_recipe(str(recipe_path))
    networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True, inversion_mode="all")
    assert len(networks) > 0, "no networks built from recipe"
    return {"model": model, "networks": networks, "target": load_target("MIT_T"), "recipe": recipe}


def test_logged_vs_committed_correlation_with_target(design_setup):
    model, networks, target = (
        design_setup["model"],
        design_setup["networks"],
        design_setup["target"],
    )

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
    stack.build(model.compute_config, enable_tu_masking=dmanager.enable_tu_masking)
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
    assert delta < 0.05, (
        f"logged vs committed correlation mismatch: logged={logged_corr:.4f}, committed={committed_corr:.4f}, "
        f"delta={delta:.4f}, logged_range=[{logged_pred.min():.4f}, {logged_pred.max():.4f}], "
        f"committed_range=[{committed_pred.min():.4f}, {committed_pred.max():.4f}]"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
