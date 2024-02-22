### {{{                          --     imports     --
import jax
from typing import Tuple
from datetime import datetime
import jax.numpy as jnp
from jax import jit, vmap, grad, value_and_grad
from pathlib import Path
from jax.tree_util import Partial

# original partial:
from functools import partial
import json
import pandas as pd
import optax
import matplotlib.pyplot as plt
import numpy as np
import joblib
from joblib import Parallel, delayed
from . import datautils as du
from . import plotutils as pu
from . import trainutils as tu
from . import utils as ut
from . import nodes as nodes
from . import compute as cmp
from .utils import check, checkwrap
from .parameters import ParameterTree

import wandb as wb
import os
import time
from tqdm import tqdm

from typing import List, Tuple, Dict, Any, Callable, Collection, Optional, Union

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                      --     loss functions     --


def check_XYZ(X, Y, Z, stack):
    nb_inputs = sum([n.get_nb_inputs() for n in stack.networks])
    nb_outputs = sum([n.get_nb_outputs() for n in stack.networks])
    assert X.ndim == Y.ndim == Z.ndim == 2, "X, Y, and Z must have 2 dimensions"
    assert X.shape[0] == Y.shape[0] == Z.shape[0], "X, Y, and Z must have the same number of rows"
    assert (
        X.shape[1] == nb_inputs
    ), "X must have as many columns as the total number of inputs in the stack"
    assert (
        Y.shape[1] == Z.shape[1] == nb_outputs
    ), "Y and Z must have as many columns as the total number of outputs in the stack"


def quantile_loss_with_grads(stack, huber_quantile_loss_delta, negative_grad_penalty):

    batch_apply = jax.vmap(stack.apply, in_axes=(None, 0, 0, 0))

    def loss_func(dynamic, static, X, Y, Z, key):
        check_XYZ(X, Y, Z, stack)
        params = ParameterTree.merge(dynamic, static)
        keys = jax.random.split(key, X.shape[0])
        yhat, grads = batch_apply(params, X, Z, keys)
        assert yhat.shape == Y.shape, "yhat and Y must have the same shape"
        error = yhat - Y
        quantile_loss = jnp.mean(tu.huber_quantile_loss(error, Z, delta=huber_quantile_loss_delta))

        # grads is the concatenated and flattened jacobian of
        # translate, transcript, and output nodes wrt their inputs
        # they should be monotonically increasing so we add a loss term

        negative_grads = jnp.mean(jnp.clip(-grads, 0, None))
        return quantile_loss + negative_grad_penalty * negative_grads

    return loss_func


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                       --     main training function     --


def start(
    dman: du.DataManager,
    training_config: Dict[str, Any],
    compute_config: cmp.ComputeConfigManager,
    loggers: Optional[List[Tuple[int, Callable]]] = None,
    seed: Optional[int] = None,
) -> Tuple[ParameterTree, List[jnp.ndarray]]:

    # Note on loggers:
    # loggers is a list of tuples (period:int, logger: Callable)
    # period is the number of epochs between two calls to the logger
    # if period is -1 or None, the logger will be called at the end of the training
    # if period is 0 or 1, the logger will be called at every epoch
    # all loggers are called at the beginning of the training run
    # with epoch=0 and epoch_history=None

    ut.logger.debug(f"Training config: {training_config}")
    ut.logger.debug(f"Compute config: {compute_config.config}")

    # --- get constants from training config (making sure they are there)
    N_REPLICATES = training_config.get('n_replicates', 1)
    N_BATCHES = training_config['n_batches']
    N_EPOCHS = training_config['epochs']
    BATCH_SIZE = training_config['batch_size']
    KEEP_IN_HISTORY = training_config.get('keep_in_history', ['loss'])
    STEPS_PER_EPOCH = max(1, int(training_config['steps_per_epoch']))
    RNG_KEY = seed or training_config['rng_key']
    LOSS_FUNCTION = training_config['loss_function']

    # --- init & batches generation

    key = jax.random.PRNGKey(RNG_KEY)

    stack, params = tu.init_stack(compute_config, dman, N_REPLICATES, key)
    assert stack.apply is not None
    assert params is not None

    xbatches, ybatches = tu.generate_batches(dman, N_REPLICATES, N_BATCHES, BATCH_SIZE, key)

    static, dynamic = params.filter_by_tag(['non_grad', 'local'])

    optimizer = tu.get_optimizer(training_config)
    opt_state = vmap(optimizer.init)(dynamic)

    ut.logger.info(
        f"""Done initializing optimizer,
        n_replicates: {N_REPLICATES}
        batches: {xbatches.shape[1]}
        steps per epoch: {STEPS_PER_EPOCH}
        random seed: {RNG_KEY}"""
    )

    # --- loss & update functions

    loss_func = ut.deserialize_function(LOSS_FUNCTION)
    assert callable(loss_func)
    scannable_step = tu.make_scannable_step(
        loss_func, optimizer, fields_to_keep_in_history=KEEP_IN_HISTORY, scannable=True
    )

    def per_replicate_epoch_step(start_params, start_opt_state, key, xbatches, ybatches):
        assert xbatches.shape[:-1] == (STEPS_PER_EPOCH, BATCH_SIZE)
        assert ybatches.shape[:-1] == (STEPS_PER_EPOCH, BATCH_SIZE)
        pscan = ut.progress_scan(STEPS_PER_EPOCH, message='Training model')
        zbatches = jax.random.uniform(key, ybatches.shape)
        assert zbatches.shape == ybatches.shape
        batch_keys = jax.random.split(key, STEPS_PER_EPOCH)
        sstep = pscan(scannable_step)
        (final_params, final_opt_state), epoch_history = jax.lax.scan(
            sstep,
            (start_params, start_opt_state),
            (jnp.arange(STEPS_PER_EPOCH), xbatches, ybatches, zbatches, batch_keys),
        )
        return final_params, final_opt_state, epoch_history

    def epoch_step(params: ParameterTree, opt_state: optax.OptState, epoch_key, xs, ys):
        keys = jax.random.split(epoch_key, N_REPLICATES)
        assert xs.shape[:-1] == ys.shape[:-1] == (N_REPLICATES, STEPS_PER_EPOCH, BATCH_SIZE)
        return jax.vmap(per_replicate_epoch_step)(params, opt_state, keys, xs, ys)

    with ut.timer('Lowering and compiling the epoch_step function'):
        xb = ut.get_looped_slice(xbatches, 0, STEPS_PER_EPOCH, axis=1)
        yb = ut.get_looped_slice(ybatches, 0, STEPS_PER_EPOCH, axis=1)
        lowered = jax.jit(epoch_step).lower(params, opt_state, key, xb, yb)
        compiled_epoch_step = lowered.compile()

    # --- main training loop

    if loggers is None:
        loggers = [(1, tu.console_log)]

    assert isinstance(loggers, list)

    # call all loggers at the beginning of the training
    for _, l in loggers:
        l(epoch=0, training_config=training_config)

    ut.logger.info(f'Begin training for {N_EPOCHS} epochs')

    epoch_history = {}

    loss_history = []

    for i, epoch_key in enumerate(jax.random.split(key, N_EPOCHS), 1):
        t0 = time.time()
        xb = ut.get_looped_slice(xbatches, i * STEPS_PER_EPOCH, (i + 1) * STEPS_PER_EPOCH, axis=1)
        yb = ut.get_looped_slice(ybatches, i * STEPS_PER_EPOCH, (i + 1) * STEPS_PER_EPOCH, axis=1)
        params, opt_state, epoch_history = compiled_epoch_step(params, opt_state, epoch_key, xb, yb)
        epoch_history['epoch_time'] = time.time() - t0
        epoch_history['latest_params'] = params
        if 'loss' in epoch_history:
            loss_history.append(epoch_history['loss'])

        for t, l in loggers:
            if t is not None:
                if (t == 0 or (i % t == 0 and t > 0)) or i == N_EPOCHS:
                    l(
                        epoch=i,
                        training_config=training_config,
                        epoch_history=epoch_history,
                        nbatches=STEPS_PER_EPOCH,
                    )

    for t, l in loggers:
        if t is None or t == -1:
            l(
                epoch=N_EPOCHS,
                training_config=training_config,
                epoch_history=epoch_history,
            )

    ut.logger.info(f'End of training for {N_EPOCHS} epochs')

    return params, loss_history


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                  --     training program helper     --

DEFAULT_LOSS = partial(
    quantile_loss_with_grads, huber_quantile_loss_delta=0.1, negative_grad_penalty=0.01
)
DEFAULT_LOSS_SERIALIZED = ut.serialize_function(DEFAULT_LOSS)

DEFAULT_TRAINING_CONFIG = {
    # -------- training config --------
    # training loop
    "rng_key": 42,
    "negative_grad_penalty": 0.1,
    "huber_quantile_loss_delta": 0.1,
    'optimizer': 'adam',
    'epochs': 150,
    'schedule': 'cosine',
    'learning_rate': 1e-3,
    'end_learning_rate': 1e-5,
    'warmup_epochs': 15,
    'steps_per_epoch': 128,
    'decay_epochs': 130,
    'adam_w_decay': 0.001,
    'max_gradient_norm': 1.0,
    # batches
    "batch_size": 32,
    "n_batches": 2048,
    "loss_function": DEFAULT_LOSS_SERIALIZED,
}

import argparse
import json
from pathlib import Path


class UpdateConfigAction(argparse.Action):
    def __init__(self, config_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_name = config_name

    def __call__(self, parser, namespace, values, option_string=None):
        updates = getattr(namespace, f"{self.config_name}_updates", None)
        if updates is None:
            updates = []
        updates.append(values)
        setattr(namespace, f"{self.config_name}_updates", updates)


class TrainingProgram:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self._add_base_arguments()

    def _add_base_arguments(self):
        self.parser.add_argument(
            '--wandb_project', type=str, default=None, help='name of wandb project'
        )
        self.parser.add_argument(
            '--compute_config_file', type=str, default=None, help='path to compute config'
        )
        self.parser.add_argument(
            '--training_config_file', type=str, default=None, help='path to training config'
        )
        self.parser.add_argument(
            '--data_config_file', type=str, default=None, help='path to data config'
        )
        self.parser.add_argument(
            '--local_save_dir', type=str, default='./results', help='path to save results'
        )
        self.parser.add_argument(
            '--seed', type=int, default=None, help='random seed (default: random)'
        )
        self.parser.add_argument(
            '--enable_checks',
            action='store_true',
            help='enable checks (default: False)',
        )
        self.parser.add_argument(
            '--loglevel', type=str, default='info', help='log level (default: debug)'
        )
        # self.parser.add_argument(
        # '--device', type=str, default='cpu', help='jax device (default: cpu)'
        # )
        self.parser.add_argument(
            '--data_path',
            type=str,
            default='./data/calibrated_data',
            help='path to xp data directory',
        )
        self.parser.add_argument(
            '--wandb_plot_period',
            type=int,
            default=-1,  # only at the end
            help='wandb plot period, None = no plots, -1 = only at the end',
        )

        self.parser.add_argument(
            '--wandb_eval_period',
            type=int,
            default=-1,
            help='wandb eval plot period, None = no plots, -1 = only at the end',
        )
        self.parser.add_argument(
            '--wandb_save_period',
            type=int,
            default=-1,
            help='wandb params save period, None = no save, -1 = only at the end',
        )

        self.parser.add_argument(
            '--config',
            type=str,
            action=partial(UpdateConfigAction, 'config'),
            help='update training_config with format: <parameter>=<value>',
        )

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def parse_args(self, default_args=None):

        import sys

        is_notebook = 'ipykernel' in sys.modules

        ut.logger.info(f'is_notebook: {is_notebook}')

        extra_args = default_args if default_args is not None else []

        # combine parsed args and extra_args. parsed args have priority over extra_args.
        # if we're in a notebook, only use extra_args. Otherwise we can combine them.
        if is_notebook:
            self.args = self.parser.parse_args(extra_args)
        else:
            self.args = self.parser.parse_args(extra_args + sys.argv[1:])
            ut.logger.info(f'args: {self.args}')

        # load the 3 config files (training, compute, data)
        self.training_config = DEFAULT_TRAINING_CONFIG
        if self.args.training_config_file is not None:
            if not Path(self.args.training_config_file).is_file():
                raise ValueError(f'{self.args.training_config_file} is not a file')
            self.training_config = json.load(open(self.args.training_config_file))

        self.compute_config = cmp.DEFAULT_COMPUTE_CONFIG
        if self.args.compute_config_file is not None:
            if not Path(self.args.compute_config_file).is_file():
                raise ValueError(f'{self.args.compute_config_file} is not a file')
            self.compute_config = cmp.ComputeConfigManager.from_file(self.args.compute_config_file)

        self.data_config = du.DEFAULT_DATA_CONFIG
        if self.args.data_config_file is not None:
            if not Path(self.args.data_config_file).is_file():
                raise ValueError(f'{self.args.data_config_file} is not a file')
            self.data_config = json.load(open(self.args.data_config_file))

        if self.args.enable_checks:
            ut.set_enable_checks(True)

        self.local_save_dir = Path(self.args.local_save_dir)
        ut.logger.info(f"Saving results to {self.local_save_dir}")

        # loglevel
        ut.set_loglevel(self.args.loglevel)

        if self.args.seed is not None:
            self.seed = self.args.seed
        else:
            self.seed = np.random.randint(0, 2**32)

    def update_config_from_args(self):
        # Apply updates to the training_config dict
        print(f'config_updates: {self.args.config_updates}')
        updates = getattr(self.args, f"config_updates", [])
        for update in updates:
            ut.logger.info(f"Updating training_config with {update}")
            parameter, value = update.split('=')
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass  # Keep value as a string if it's not JSON-parseable
            self.training_config[parameter] = value

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        elif hasattr(self, 'args') and hasattr(self.args, attr):
            return getattr(self.args, attr)
        else:
            raise AttributeError(f"{self.__class__.__name__} object has no attribute '{attr}'")

    def __DEPRECATED_start_training(
        self,
        training: du.DataManager,
        validation: Optional[du.DataManager] = None,
        extra_loggers: List[Tuple[int, Callable]] = [],
    ):

        # we update the training config with the command line arguments
        # after parsing the command line arguments, because some other program
        # might have added some arguments to the training config

        self.update_config_from_args()

        prog_config = self.args.__dict__.copy()

        self.training_config['program_config'] = prog_config

        if self.wandb_project is not None:
            loggers = tu.setup_wandb_logging(
                self.wandb_project,
                training,
                self.training_config,
                self.compute_config,
                self.data_config,
                plot_period=self.wandb_plot_period,
                params_save_period=self.wandb_save_period,
            )

        # if validation is not None:
        # with ut.timer('Validation stack initialization'):
        # key = jax.random.PRNGKey(self.seed)
        # vstack = validation.build_compute_stack(self.compute_config)
        # base_params = vstack.init(key)
        # loggers.append(
        # (
        # self.wandb_eval_period,
        # partial(
        # wandb_plot_pred,
        # dman=validation,
        # base_params=base_params,
        # log_key='Validation',
        # ),
        # )
        # )

        else:
            loggers = [
                (1, console_log),
                (
                    -1,
                    partial(
                        local_save,
                        compute_config=self.compute_config,
                        trainer_config=self.training_config,
                        save_dir=self.local_save_dir,
                    ),
                ),
            ]

        loggers += extra_loggers

        return start(training, self.training_config, self.compute_config, loggers, seed=self.seed)


##────────────────────────────────────────────────────────────────────────────}}}
