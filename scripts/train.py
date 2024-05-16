from biocomp import utils as ut
import biocomp as bc
import jax
import biocomp.compute as cmp
from biocomp.train import DEFAULT_TRAINING_CONFIG as training_config
from pathlib import Path
import json
from functools import partial
import numpy as np
import argparse
from biocomp import scriptutils as su
import biocomp.datautils as du
from biocomp import nodes

config = cmp.ComputeConfig()
config.set_impl('transcription', nodes.transcription)
config.set_impl('translation', nodes.translation)
config.set_impl('inv_transcription', nodes.inv_transcription)
config.set_impl('inv_translation', nodes.inv_translation)
config.set_impl('sequestron_ERN', nodes.ERN5p)
config.set_impl('source', nodes.source_new)
config.set_impl('inv_source', nodes.inv_source_new)
config.set_impl('bias', nodes.bias)
config.set_impl('numeric', nodes.bias)
config.set_impl('aggregation', nodes.aggregation)
config.set_impl('inv_aggregation', nodes.inv_aggregation)
config.set_impl('output', nodes.grouped_output)
config.set_impl('deadend', nodes.single_passthrough)


XP = {
    "bt": "2023-04-03_Constraints_Pgu_Bleedthrough",
    # "cascades": "2023-04-18_Constraints_PguCascades",
    # "csy4matrix": "2023-03-26_MatrixCsy4",
    # "casematrix": "2023-02-16_Matrix",
    # "uorfs": "2022-11-10_uORFs_and_company",
}

color_aliases = {"1XIRFP720": "IRFP720"}
lib = ut.load_lib()

with ut.timer(f"Loading data and building networks for {XP.keys()}"):
    loadedxp = {
        xpname: ut.load_xp(
            xppath,
            lib,
            data_path="./data/calibrated_data_v2",
            color_aliases=color_aliases,
        )
        for xpname, xppath in XP.items()
    }
    training = du.DataManager.from_xps(loadedxp.values(), training_config, inverse="all")

stack = training.build_compute_stack(config)
key = jax.random.PRNGKey(training_config["rng_key"])
params = stack.init(key)
params, loss_history, epoch_history = bc.train.start(
    training,
    training_config,
    config,
    [(1, bc.trainutils.console_log)],
)
