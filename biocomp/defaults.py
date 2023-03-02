from . import nodes as nodes
from . import nn_nodes as nn
from functools import partial

T_SIZE = 32
T_DEPTH = 3
I_SIZE = 32
I_DEPTH = 2
I_OUT = 8
ERN_SIZE = 64
ERN_DEPTH = 3
MEFL_SIZE = 32
MEFL_DEPTH = 3

DEFAULT_NN_NODES = dict(
    nodes.DEFAULT_COMPUTE_NODES_DICT,
    **{
        'output': partial(nn.output, wsize=MEFL_SIZE, depth=MEFL_DEPTH),
        'transcription': partial(
            nn.transcription,
            outer_wsize=T_SIZE,
            outer_depth=T_DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
            inner_out=I_OUT,
        ),
        'translation': partial(
            nn.translation,
            outer_wsize=T_SIZE,
            outer_depth=T_DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
            inner_out=I_OUT,
        ),
        'inv_transcription': partial(
            nn.inv_transcription,
            outer_wsize=T_SIZE,
            outer_depth=T_DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
            inner_out=I_OUT,
        ),
        'inv_translation': partial(
            nn.inv_translation,
            outer_wsize=T_SIZE,
            outer_depth=T_DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
            inner_out=I_OUT,
        ),
        'sequestron_ERN': partial(nn.ERN5p, wsize=ERN_SIZE, depth=ERN_DEPTH),
        'sequestron_ERN3p': partial(nn.ERN3p, wsize=ERN_SIZE, depth=ERN_DEPTH),
    },
)

DEFAULT_DATA_CONFIG = {
    "batch_size": 16,
    "n_batches": 2048,
    "kde_bw_method": 0.05,
    "log_factor": 5e3,
    "max_value": 5e7,
    "density_quantile_threshold": 0.07,
}

DEFAULT_TRAINING_CONFIG = {
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "adam_w_decay": 1e-7,
    "rng_key": 42,
    "epochs": 300,
    "n_replicates": 1,
    "n_epochs_per_batch_rotation": 16,
    "negative_grad_penalty": 0.1,
    "huber_quantile_loss_delta": 0.1,
    "static_params": [['node']],
    "node_impl": DEFAULT_NN_NODES,
}

DEFAULT_CONFIG = {
    **DEFAULT_DATA_CONFIG,
    **DEFAULT_TRAINING_CONFIG,
}
