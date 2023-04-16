from . import nodes as nodes
from . import compute as cmp
from . import nodes_old as nodes_old
from functools import partial


DEFAULT_TRAINING_CONFIG = {
    # -------- training config --------
    "rng_key": 42,
    "negative_grad_penalty": 0.1,
    "huber_quantile_loss_delta": 0.1,
    "static_params": ['/__static__','/node'],
    "cache_dir": "./.training_cache",
    'optimizer': 'adam',
    'epochs': 150,
    'schedule': 'cosine',
    'learning_rate': 0.002,
    'end_learning_rate': 5e-6,
    'warmup_epochs': 25,
    'steps_per_epoch': 128,
    'decay_epochs': 125,
    'adam_w_decay': 0.001,
    # -------- data config --------
    "batch_size": 16,
    "n_batches": 2048,
    "data_scaling_log_factor": 2e4,
    "data_scaling_max_value": 5e7,
    "data_sampling_kde_bw_method": 0.1,
    "data_sampling_density_quantile_threshold": 0.025, # threshold = min of both
    "data_sampling_coords_for_density_threshold": 0.3, # threshold = min of both
}

DEFAULT_COMPUTE_CONFIG = cmp.ComputeConfigManager()
DEFAULT_COMPUTE_CONFIG.set('transcription', nodes.transcription)
DEFAULT_COMPUTE_CONFIG.set('translation', nodes.translation)
DEFAULT_COMPUTE_CONFIG.set('inv_transcription', nodes.inv_transcription)
DEFAULT_COMPUTE_CONFIG.set('inv_translation', nodes.inv_translation)
DEFAULT_COMPUTE_CONFIG.set('sequestron_ERN', nodes.ERN5p)
DEFAULT_COMPUTE_CONFIG.set('sequestron_ERN3p', nodes.ERN3p)
DEFAULT_COMPUTE_CONFIG.set('source', nodes.source)
DEFAULT_COMPUTE_CONFIG.set('inv_source', nodes.inv_source)
DEFAULT_COMPUTE_CONFIG.set('numeric', nodes.numeric)
DEFAULT_COMPUTE_CONFIG.set('inv_numeric', nodes.inv_numeric)
DEFAULT_COMPUTE_CONFIG.set('aggregation', nodes.aggregation)
DEFAULT_COMPUTE_CONFIG.set('inv_aggregation', nodes.inv_aggregation)
DEFAULT_COMPUTE_CONFIG.set('output', nodes.grouped_output)
DEFAULT_COMPUTE_CONFIG.set('deadend', nodes.single_passthrough)
