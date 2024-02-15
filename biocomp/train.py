## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     import and init     --
# ···············································································
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
from . import utils as ut
from . import nodes as nodes
from . import compute as cmp
from .utils import check, checkwrap
from .parameters import ParameterTree

import wandb as wb
import os
import time
from tqdm import tqdm

from typing import List, Tuple, Dict, Any, Callable, Collection, Optional

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     loss functions     --
# ···············································································


def mse_loss(y, y_hat, n_outputs=None):
    if n_outputs is None:
        n_outputs = y.shape[1]
    assert y_hat.ndim == 2 and y.ndim == 2
    return jnp.mean((y[:, :n_outputs] - y_hat[:, :n_outputs]) ** 2)


def huber_quantile_loss(e, q, delta=0.1):
    return jnp.where(
        jnp.abs(e) <= delta, 0.5 * e**2, delta * (jnp.abs(e) - 0.5 * delta)
    ) * jnp.where(e < 0, q, (1.0 - q))


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     loggers     --
# ···············································································


@Partial(jit, static_argnums=(1,))
def compstats(v, smooth_win=1):
    medians = vmap(jnp.median)(v)
    mins = vmap(jnp.min)(v)
    maxs = vmap(jnp.max)(v)
    p20s = vmap(lambda x: jnp.percentile(x, 20))(v)
    p80s = vmap(lambda x: jnp.percentile(x, 80))(v)
    if smooth_win > 1:
        medians = jnp.convolve(medians, jnp.ones(smooth_win) / smooth_win, mode='same')
        p80s = jnp.convolve(p80s, jnp.ones(smooth_win) / smooth_win, mode='same')
        p20s = jnp.convolve(p20s, jnp.ones(smooth_win) / smooth_win, mode='same')
        maxs = jnp.convolve(maxs, jnp.ones(smooth_win) / smooth_win, mode='same')
        mins = jnp.convolve(mins, jnp.ones(smooth_win) / smooth_win, mode='same')
    return medians, p20s, p80s, mins, maxs


def get_epoch_stats(epoch_data, smooth_win=1):
    stats = {'grad': {}, 'params': {}}
    if 'grad' in epoch_data:
        for k, v in epoch_data['grad']['shared'].items():
            stats['grad'][k] = compstats(v)
    if 'params' in epoch_data:
        for k, v in epoch_data['params']['shared'].items():
            stats['params'][k] = compstats(v)
    return stats


def local_save(
    epoch, compute_config, training_config, epoch_history=None, save_dir=None, full_save=False, **_
):
    assert save_dir is not None
    if epoch_history is None:
        return

    if 'latest_params' not in epoch_history:
        ut.logger.warning("No params for plotting evaluations")
        return

    t0 = time.time()

    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True)

    compute_conf_path = Path(save_dir) / 'compute_config.json'
    if not compute_conf_path.exists():
        compute_config.export(compute_conf_path)

    training_conf_path = Path(save_dir) / 'training_config.json'
    if not training_conf_path.exists():
        with open(training_conf_path, 'w') as f:
            json.dump(training_config, f)

    if full_save:
        full_save_until_epoch = full_save if isinstance(full_save, int) else 2
        if epoch <= full_save_until_epoch:
            du.save(epoch_history, f'{save_dir}/epoch_{epoch}_full.pkl')

    params = epoch_history['latest_params']
    loss = np.array(epoch_history['loss'])

    # first we rename the old params
    for f in Path(save_dir).glob('latest_params.pkl'):
        f.rename(f'{save_dir}/old_params.pkl')

    # then we save the new ones
    du.save(params, f'{save_dir}/latest_params.pkl')

    # then we delete the old one
    for f in Path(save_dir).glob('old_params.pkl'):
        f.unlink()

    ut.logger.info(f"Saving epoch to disk took {time.time() - t0:.2f}s")


def wandb_plot_pred(dman, epoch_history=None, base_params=None, log_key=None, **_):
    if epoch_history is None:
        return

    import matplotlib

    matplotlib.pyplot.switch_backend('Agg')
    import traceback
    from tqdm import tqdm

    networks = dman.get_networks()
    stack = dman.get_compute_stack()

    if 'latest_params' not in epoch_history:
        ut.logger.warning("No params for plotting evaluations")
        return

    params = epoch_history['latest_params']

    if base_params is not None:
        local, _ = base_params.filter_by_tag(['local'])
        _, shared = params.filter_by_tag(['local'])
        params = ParameterTree.merge(local, shared)

    with ut.timer('wandb_plot_pred'):
        N_SAMPLES_PER_CHUNK = 2000
        N_CHUNKS = 5

        N_SAMPLES_TOTAL = N_SAMPLES_PER_CHUNK * N_CHUNKS

        key = jax.random.PRNGKey(0)
        X, Y = dman.get_uniform_samples(key, N_SAMPLES_TOTAL)
        assert len(X) == len(Y)
        assert len(X) == len(networks)

        X = [np.expand_dims(arr, axis=1) if arr.ndim == 1 else arr for arr in X]
        Y = [np.expand_dims(arr, axis=1) if arr.ndim == 1 else arr for arr in Y]

        ALLX = np.concatenate(X, axis=1)

        assert ALLX.shape == (
            N_SAMPLES_TOTAL,
            stack.total_nb_of_inputs,
        ), f"{ALLX.shape} != {(N_SAMPLES_TOTAL, stack.total_nb_of_inputs)}"

        @jit
        def compute(params, XX, Q, keys):
            res, _ = stack.apply(params, XX, Q, keys)
            return res

        ALLX_CHUNKS = np.split(ALLX, N_CHUNKS, axis=0)

        YHAT = []

        for chunk_id, XX in enumerate(tqdm(ALLX_CHUNKS, desc='wandb_plot_pred chunks')):
            Q = jax.random.uniform(key, (N_SAMPLES_PER_CHUNK, stack.total_nb_of_outputs))
            keys = jax.random.split(key, N_SAMPLES_PER_CHUNK)
            key = keys[-1]
            yhat_chunk = vmap(compute, in_axes=(None, 0, 0, 0))(params, XX, Q, keys)
            YHAT.append(np.array(yhat_chunk))

        YHAT = np.concatenate(YHAT, axis=0)

        def plot_prediction(index):
            try:
                out_id = stack.get_network_global_output_id(index)
                n_out = networks[index].get_nb_outputs()
                x, y = X[index], Y[index]
                yhat = YHAT[: x.shape[0], out_id : out_id + n_out]
                assert yhat.shape == y.shape, f"{yhat.shape} != {y.shape}"
                error = np.abs(y - yhat).mean()
                fig = du.report(params, dman, index, use_x_y_yhat=(x, y, yhat), res=64)
                img = wb.Image(fig, caption=f'{networks[index].name}, error={error:.4f}')
                plt.close()
                plt.close(fig)
                plt.close('all')
                return img, error

            except Exception as e:
                ut.logger.warning(f"Failed to plot prediction {index}: {e}")
                traceback.print_exc()
                return None, None

        pred = [plot_prediction(i) for i in tqdm(list(range(len(networks))))]
        pred = [p for p in pred if p[0] is not None]
        predimg, prederr = zip(*pred)

        if log_key is None:
            log_key = 'Evaluations'

        wb.log({f'{log_key}': predimg})
        wb.log({f'{log_key}_err': prederr})


def wandb_log_epoch(epoch_history=None, **_):
    if epoch_history is not None:
        # measure time now:
        losses = np.array(epoch_history['loss'])
        for loss in losses:
            wb.log({'loss': loss})
        wb.log({'epoch_time': epoch_history['epoch_time']})


def console_log(epoch, training_config, epoch_history=None, **_):
    if epoch_history is not None and len(epoch_history['loss']) > 0:
        losses = np.array(epoch_history['loss'])
        # make it 2d if it's 1d
        if losses.ndim == 1:
            losses = losses[:, None]

        avg_losses = np.mean(losses, axis=1)
        best_id = np.argmin(avg_losses)
        best_std = np.std(losses[best_id])
        avg_std = np.std(avg_losses)
        avg_avg = np.mean(avg_losses)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        fmt = lambda x: f'{x:.1e}' if x < 1e-3 or x > 1e3 else f'{x:.3f}'

        ut.logger.info(
            f"""[{epoch}/{training_config["epochs"]} in {epoch_history["epoch_time"]:.2f}s]
             best loss: {fmt(avg_losses[best_id])} ± {fmt(best_std)} (replicate n° {best_id+1}/{len(losses)})
             replicates avg: {fmt(avg_avg)} ± {fmt(avg_std)} """
        )


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

### {{{                       --     main function     --


def get_optimizer(cfg):

    learning_rate = cfg['learning_rate']

    if 'schedule' in cfg:
        if cfg['schedule'] == 'cosine':
            steps_per_epoch = cfg['steps_per_epoch']
            max_learning_rate = cfg['learning_rate']
            warmup_steps = cfg['warmup_epochs'] * steps_per_epoch
            decay_steps = cfg['decay_epochs'] * steps_per_epoch
            end_learning_rate = cfg['end_learning_rate']
            learning_rate = optax.warmup_cosine_decay_schedule(
                init_value=1e-7,
                peak_value=max_learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
                end_value=end_learning_rate,
            )
        elif cfg['schedule'] == 'constant':
            learning_rate = cfg['learning_rate']
        else:
            raise ValueError(f"Unknown learning rate schedule {cfg['schedule']}")

    optimizers = {
        'sgd': optax.sgd(learning_rate=learning_rate),
        'adamw': optax.adamw(learning_rate=learning_rate, weight_decay=cfg['adam_w_decay']),
        'adam': optax.adam(learning_rate=learning_rate),
        'amsgrad': optax.amsgrad(learning_rate=learning_rate),
    }
    assert (
        cfg['optimizer'] in optimizers.keys()
    ), f"Optimizer {cfg['optimizer']} not available. Available optimizers are {optimizers.keys()}"
    optimizer = optimizers[cfg['optimizer']]

    gradient_clip = optax.clip_by_global_norm(cfg['max_gradient_norm'])
    optimizer = optax.chain(gradient_clip, optimizer)

    return optimizer


def setup_wandb_logging(
    project,
    dman,
    training_config,
    compute_config,
    data_config,
    plot_period=-1,  # only at the end
    params_save_period=-1,  # only at the end
    entity='jdisset',
    **kw,
):
    import wandb as wb

    full_config = {**training_config, **compute_config.config, **data_config}
    wb.init(config=full_config, project=project, entity=entity, **kw)

    save_dir = Path(wb.run.dir)
    loggers = [
        (1, console_log),
        (
            params_save_period,
            partial(
                local_save,
                compute_config=compute_config,
                training_config=training_config,
                save_dir=save_dir,
            ),
        ),
        (1, wandb_log_epoch),
        (plot_period, partial(wandb_plot_pred, dman=dman)),
    ]
    return loggers


def get_memory(config):
    cache_folder = config.get("cache_folder", None)
    if cache_folder:
        cache_folder = Path(cache_folder).expanduser()
        cache_folder.mkdir(parents=True, exist_ok=True)
        return joblib.Memory(cache_folder, verbose=0)
    else:
        return joblib.Memory(None, verbose=0)


def start(
    dman: du.DataManager,
    training_config,
    compute_config: cmp.ComputeConfigManager,
    loggers=None,
    seed=None,
):
    # Note on loggers:
    # loggers is a list of tuples (period:int, logger: Callable)
    # period is the number of epochs between two calls to the logger
    # if period is -1 or None, the logger will be called at the end of the training
    # if period is 0 or 1, the logger will be called at every epoch
    # all loggers are called at the beginning of the training run
    # with epoch=0 and epoch_history=None

    ut.logger.debug(f"About to start training")
    ut.logger.debug(f"Training config: {training_config}")
    ut.logger.debug(f"Compute config: {compute_config.config}")

    # --- get constants from training config (making sure they are there)
    NEGATIVE_GRAD_PENALTY = training_config['negative_grad_penalty']
    N_REPLICATES = training_config.get('n_replicates', 1)
    N_BATCHES = training_config['n_batches']
    N_EPOCHS = training_config['epochs']
    BATCH_SIZE = training_config['batch_size']
    KEEP_IN_HISTORY = training_config.get('keep_in_history', ['loss'])
    STEPS_PER_EPOCH = max(1, int(training_config['steps_per_epoch']))
    RNG_KEY = seed or training_config['rng_key']
    HUBER_QUANTILE_LOSS_DELTA = float(training_config['huber_quantile_loss_delta'])

    # --- init & batches generation

    def init_stack(key) -> Tuple[cmp.ComputeStack, ParameterTree]:
        stack = dman.build_compute_stack(compute_config)
        assert stack.init is not None
        with ut.timer('Stack initialization'):
            params = vmap(stack.init)(jax.random.split(key, N_REPLICATES))
        return stack, params

    def generate_batches(key):
        total_n_batches = N_BATCHES * N_REPLICATES

        with ut.timer('Generating batches'):
            xbatches, ybatches = dman.get_batches(total_n_batches, BATCH_SIZE, key)
        # current shape is (R*B,N,F), final shape should be (R,B,N,F)
        # R: replicates, B: batches, N: data, F: features
        xbatches = xbatches.reshape(N_REPLICATES, N_BATCHES, *xbatches.shape[1:])
        ybatches = ybatches.reshape(N_REPLICATES, N_BATCHES, *ybatches.shape[1:])

        assert xbatches.shape[:-1] == (
            N_REPLICATES,
            N_BATCHES,
            BATCH_SIZE,
        )
        assert ybatches.shape[:-1] == (
            N_REPLICATES,
            N_BATCHES,
            BATCH_SIZE,
        )

        return xbatches, ybatches

    ut.logger.info(f"Using random seed {RNG_KEY}")
    key = jax.random.PRNGKey(RNG_KEY)

    stack, params = init_stack(key)
    assert params is not None

    xbatches, ybatches = generate_batches(key)

    optimizer = get_optimizer(training_config)
    static, dynamic = params.filter_by_tag(['non_grad', 'local'])
    ut.logger.info(f"Split params between dynamic and static. Now intializing optimizer.")
    opt_state = vmap(optimizer.init)(dynamic)

    ut.logger.info(
        f"""Done initializing optimizer,
    n_replicates: {N_REPLICATES},
    batches: {xbatches.shape[1]},
    steps per epoch: {STEPS_PER_EPOCH}"""
    )

    # --- loss & update functions

    assert stack.apply is not None
    batch_apply = jax.vmap(stack.apply, in_axes=(None, 0, 0, 0))

    def check_XYZ(X, Y, Z, stack):
        nb_inputs = sum([n.get_nb_inputs() for n in stack.networks])
        nb_outputs = sum([n.get_nb_outputs() for n in stack.networks])
        assert X.ndim == Y.ndim == Z.ndim == 2, "X, Y, and Z must have 2 dimensions"
        assert (
            X.shape[0] == Y.shape[0] == Z.shape[0]
        ), "X, Y, and Z must have the same number of rows"
        assert (
            X.shape[1] == nb_inputs
        ), "X must have as many columns as the total number of inputs in the stack"
        assert (
            Y.shape[1] == Z.shape[1] == nb_outputs
        ), "Y and Z must have as many columns as the total number of outputs in the stack"

    def loss_func(dynamic, static, X, Y, Z, key):

        check_XYZ(X, Y, Z, stack)

        params = ParameterTree.merge(dynamic, static)
        keys = jax.random.split(key, X.shape[0])

        yhat, grads = batch_apply(params, X, Z, keys)
        assert yhat.shape == Y.shape, "yhat and Y must have the same shape"

        error = yhat - Y
        quantile_loss = jnp.mean(huber_quantile_loss(error, Z, delta=HUBER_QUANTILE_LOSS_DELTA))

        # grads is the concatenated and flattened jacobian of
        # translate, transcript, and output nodes wrt their inputs
        # they should be monotonically increasing so we add a loss term

        negative_grads = jnp.mean(jnp.clip(-grads, 0, None))

        return quantile_loss + NEGATIVE_GRAD_PENALTY * negative_grads

    def training_step(params, opt_state, x, y, z, key):
        static, dynamic = params.filter_by_tag(['non_grad', 'local'])
        loss, grads = value_and_grad(loss_func, has_aux=False)(dynamic, static, x, y, z, key)
        updates, opt_state = optimizer.update(grads, opt_state, dynamic)
        dynamic = optax.apply_updates(dynamic, updates)
        params = ParameterTree.merge(static, dynamic)
        res = {
            'params': params,
            'loss': loss,
            'grad': grads,
            'opt': opt_state,
        }
        return res

    def scannable_step(carry, i_x_y_z_k):
        params, opt_state = carry
        i, x, y, z, k = i_x_y_z_k
        updt = training_step(params, opt_state, x, y, z, k)
        params, opt_state = updt['params'], updt['opt']
        history = {k: updt[k] for k in KEEP_IN_HISTORY}
        return (params, opt_state), history

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
        print(keys.shape)
        assert xs.shape[:-1] == ys.shape[:-1] == (N_REPLICATES, STEPS_PER_EPOCH, BATCH_SIZE)
        return jax.vmap(per_replicate_epoch_step)(params, opt_state, keys, xs, ys)

    print(xbatches.shape, ybatches.shape)
    with ut.timer('Lowering the epoch_step function before compilation'):
        xb = ut.get_looped_slice(xbatches, 0, STEPS_PER_EPOCH, axis=1)
        yb = ut.get_looped_slice(ybatches, 0, STEPS_PER_EPOCH, axis=1)
        lowered = jax.jit(epoch_step).lower(params, opt_state, key, xb, yb)

    with ut.timer('Compiling the epoch_step function'):
        compiled_epoch_step = lowered.compile()

    # --- main training loop
    loggers = [(1, console_log)]

    if loggers is None:
        loggers = [(1, console_log)]

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

    def start_training(
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
            loggers = setup_wandb_logging(
                self.wandb_project,
                training,
                self.training_config,
                self.compute_config,
                self.data_config,
                plot_period=self.wandb_plot_period,
                params_save_period=self.wandb_save_period,
            )

            if validation is not None:
                with ut.timer('Validation stack initialization'):
                    key = jax.random.PRNGKey(self.seed)
                    vstack = validation.build_compute_stack(self.compute_config)
                    base_params = vstack.init(key)
                    loggers.append(
                        (
                            self.wandb_eval_period,
                            partial(
                                wandb_plot_pred,
                                dman=validation,
                                base_params=base_params,
                                log_key='Validation',
                            ),
                        )
                    )
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
