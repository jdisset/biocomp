## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     import and init     --
# ···············································································
from jax.tree_util import Partial as partial
import jax
from typing import Tuple
from datetime import datetime
import jax.numpy as jnp
from jax import jit, vmap, grad, value_and_grad
from pathlib import Path
from jax.tree_util import Partial as partial
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
from . import nodes_old as nodes_old
from . import compute as cmp
from .utils import check, checkwrap

import wandb as wb
import os
import time
from tqdm import tqdm

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


@partial(jit, static_argnums=(1,))
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


def local_save(epoch, epoch_history=None, save_dir=None, full_save=False, **_):
    assert save_dir is not None
    if epoch_history is None:
        return

    if 'latest_params' not in epoch_history:
        ut.logger.warning("No params for plotting evaluations")
        return

    t0 = time.time()

    if full_save:
        full_save_until_epoch = full_save if isinstance(full_save, int) else 2
        if epoch <= full_save_until_epoch:
            du.save(epoch_history, f'{save_dir}/epoch_{epoch}_full.pkl')

    # params = ut.tree_get(epoch_history['params'], -1)
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
        params = stack.use_shared_params(base_params, params)

    with ut.timer('wandb_plot_pred'):
        N_SAMPLES = 20000
        key = jax.random.PRNGKey(0)
        X, Y = dman.get_uniform_samples(key, N_SAMPLES)
        assert len(X) == len(Y)
        assert len(X) == len(networks)

        ALLX = jnp.concatenate(X, axis=1)
        assert ALLX.shape == (
            N_SAMPLES,
            stack.total_nb_of_inputs,
        ), f"{ALLX.shape} != {(N_SAMPLES, stack.total_nb_of_inputs)}"

        Q = jax.random.uniform(key, (N_SAMPLES, stack.total_nb_of_outputs))
        keys = jax.random.split(key, N_SAMPLES)

        def compute(params, XX, Q, keys):
            res, _ = stack.apply(params, XX, Q, keys)
            return res

        YHAT = jit(vmap(compute, in_axes=(None, 0, 0, 0)))(params, ALLX, Q, keys)

        def plot_prediction(index):
            try:
                out_id = stack.get_network_global_output_id(index)
                n_out = networks[index].get_nb_outputs()
                x, y = X[index], Y[index]
                yhat = YHAT[: x.shape[0], out_id : out_id + n_out]
                assert yhat.shape == y.shape, f"{yhat.shape} != {y.shape}"
                error = jnp.abs(y - yhat).mean()
                fig, ax = du.report(params, dman, index, use_x_y_yhat=(x, y, yhat), res=64)
                img = wb.Image(fig, caption=f'{networks[index].name}, error={error:.4f}')
                plt.close(fig)
                return img, error

            except Exception as e:
                ut.logger.warning(f"Failed to plot prediction {index}: {e}")
                traceback.print_exc()
                return None

        pred = [plot_prediction(i) for i in tqdm(list(range(len(networks))))]
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
        loss = np.array(epoch_history['loss'])
        avg = np.mean(loss)
        std = np.std(loss)
        lmin, lmax = jnp.min(loss), jnp.max(loss)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        fmt = lambda x: f'{x:.1e}' if x < 1e-3 or x > 1e3 else f'{x:.3f}'
        ut.logger.info(
            f"""[{epoch}/{training_config["epochs"]}] \
        loss: {fmt(avg)} ± {fmt(std)} [min {fmt(lmin)}, max {fmt(lmax)}] in \
        {epoch_history["epoch_time"]:.2f}s"""
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
    plot_period=-1,  # only at the end
    params_save_period=-1,  # only at the end
    entity='jdisset',
    **kw,
):
    import wandb as wb

    full_config = {**training_config, **compute_config.get_config()}

    wb.init(config=full_config, project=project, entity=entity, **kw)
    save_dir = Path(wb.run.dir)
    loggers = [
        (1, console_log),
        (params_save_period, partial(local_save, save_dir=save_dir)),
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


def start(dman: du.DataManager, training_config, compute_config, loggers=None, seed=None):

    ut.logger.debug(f"About to start training")
    ut.logger.debug(f"Training config: {training_config}")
    ut.logger.debug(f"Compute config: {compute_config.get_config()}")

    if seed is not None:
        training_config['rng_key'] = seed
    ut.logger.info(f"Going to train with random seed {training_config['rng_key']}")
    key = jax.random.PRNGKey(training_config['rng_key'])

    # --- cached init & batches generation
    memory = get_memory(training_config)

    @memory.cache
    def init_stack(dman, key):
        stack = dman.build_compute_stack(compute_config)
        with ut.timer('Stack initialization'):
            params = stack.init(key)
        return stack, params

    @memory.cache
    def generate_batches(dman, key):
        with ut.timer('Generating batches'):
            xbatches, ybatches = dman.get_batches(key)  # (B,M,N,F) shape
        return xbatches, ybatches

    stack, params = init_stack(dman, key)
    xbatches, ybatches = generate_batches(dman, key)
    optimizer = get_optimizer(training_config)
    dynamic, _ = ut.split_params(params, training_config['static_params'])
    opt_state = optimizer.init(dynamic)
    total_batches = training_config['n_batches']
    assert total_batches == xbatches.shape[0] == ybatches.shape[0]
    steps_per_epoch = max(1, int(training_config['steps_per_epoch']))

    # --- loss & update functions

    vmapped_compute = jax.vmap(stack.apply, in_axes=(None, 0, 0, 0))

    @jit
    def loss_func(dynamic, static, X, Y, Z, key):
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

        params = ut.assemble_params(dynamic, static)
        keys = jax.random.split(key, X.shape[0])

        yhat, grads = vmapped_compute(params, X, Z, keys)
        assert yhat.shape == Y.shape, "yhat and Y must have the same shape"

        error = yhat - Y
        quantile_loss = jnp.mean(
            huber_quantile_loss(error, Z, delta=training_config['huber_quantile_loss_delta'])
        )

        # grads is the concatenated and flattened jacobian of
        # translate, transcript, and output nodes wrt their inputs
        # they should be monotonically increasing so we add a loss term
        negative_grads = jnp.mean(jnp.where(grads < 0, -grads, 0))
        return quantile_loss + training_config['negative_grad_penalty'] * negative_grads


    def training_step(params, opt_state, x, y, z, key):
        dynamic, static = ut.split_params(params, training_config['static_params'])
        loss, grads = value_and_grad(loss_func, has_aux=False)(dynamic, static, x, y, z, key)
        updates, opt_state = optimizer.update(grads, opt_state, dynamic)

        # if there's any nan in updates

        msg = ''

        for k,v in grads.items():
            if jnp.any(jnp.isnan(v)):
                msg += f'\nGRAD:{k} has NaNs'

        for k,v in updates.items():
            if jnp.any(jnp.isnan(v)):
                msg += f'\nUPDT:{k} has NaNs'

        if msg != '':
            msg += f'\nloss:{loss}'
            raise ValueError(msg)


        dynamic = optax.apply_updates(dynamic, updates)
        params = ut.assemble_params(dynamic, static)

        res = {
            'params': params,
            'loss': loss,
            'grad': grads,
            'opt': opt_state,
        }
        return res

    keep_in_history = training_config.get('keep_in_history', ['loss'])

    def scannable_step(carry, i_x_y_z_k):
        params, opt_state = carry
        i, x, y, z, k = i_x_y_z_k
        updt = training_step(params, opt_state, x, y, z, k)
        params, opt_state = updt['params'], updt['opt']
        history = {k: updt[k] for k in keep_in_history}
        return (params, opt_state), history

    @jit
    def epoch_step(start_params, start_opt_state, epoch_key, xbs, ybs):
        pscan = ut.progress_scan(steps_per_epoch, message='Training model')
        zbatches = jax.random.uniform(epoch_key, ybs.shape)
        batch_keys = jax.random.split(epoch_key, steps_per_epoch)
        sstep = pscan(scannable_step)
        (final_params, final_opt_state), epoch_history = jax.lax.scan(
            sstep,
            (start_params, start_opt_state),
            (jnp.arange(steps_per_epoch), xbs, ybs, zbatches, batch_keys),
        )
        return final_params, final_opt_state, epoch_history

    def epoch_step_no_scan(start_params, start_opt_state, epoch_key, xbs, ybs):
        zbatches = jax.random.uniform(epoch_key, ybs.shape)
        batch_keys = jax.random.split(epoch_key, steps_per_epoch)
        all_history = []
        tstep = training_step
        for i, (x, y, z, k) in tqdm(
            enumerate(zip(xbs, ybs, zbatches, batch_keys)), total=steps_per_epoch
        ):
            updt = tstep(start_params, start_opt_state, x, y, z, k)
            start_params, start_opt_state = updt['params'], updt['opt']
            history = {k: updt[k] for k in keep_in_history}
            all_history.append(history)
        epoch_history = {k: jnp.stack([h[k] for h in all_history]) for k in keep_in_history}
        return start_params, start_opt_state, epoch_history

    epoch_step = epoch_step if not ut.enable_checks else epoch_step_no_scan

    # --- main training loop

    if loggers is None:
        loggers = [(1, console_log)]

    for _, l in loggers:
        l(epoch=0, training_config=training_config)

    ut.logger.info(f'Begin training for {training_config["epochs"]} epochs')

    for i, epoch_key in enumerate(jax.random.split(key, training_config['epochs']), 1):
        t0 = time.time()
        xb = ut.get_looped_slice(xbatches, i * steps_per_epoch, (i + 1) * steps_per_epoch)
        yb = ut.get_looped_slice(ybatches, i * steps_per_epoch, (i + 1) * steps_per_epoch)
        params, opt_state, epoch_history = epoch_step(params, opt_state, epoch_key, xb, yb)
        epoch_history['epoch_time'] = time.time() - t0
        epoch_history['latest_params'] = params

        for t, l in loggers:
            if t is not None:
                if (t == 0 or (i % t == 0 and t > 0)) or i == training_config['epochs']:
                    l(
                        epoch=i,
                        training_config=training_config,
                        epoch_history=epoch_history,
                        nbatches=steps_per_epoch,
                    )

    return params, epoch_history


##────────────────────────────────────────────────────────────────────────────}}}


### {{{                  --     training program helper     --

DEFAULT_TRAINING_CONFIG = {
    # -------- training config --------
    "rng_key": 42,
    "negative_grad_penalty": 0.1,
    "huber_quantile_loss_delta": 0.1,
    "static_params": ['/__static__', '/node'],
    "cache_dir": "./.training_cache",
    'optimizer': 'adam',
    'epochs': 128,
    'schedule': 'cosine',
    'learning_rate': 1e-3,
    'end_learning_rate': 1e-5,
    'warmup_epochs': 10,
    'steps_per_epoch': 128,
    'decay_epochs': 110,
    'adam_w_decay': 0.001,
    'max_gradient_norm': 1.0,
    # -------- data config --------
    "batch_size": 32,
    "n_batches": 2048,
    "data_scaling_log_factor": 5e4,
    "data_scaling_max_value": 5e7,
    "data_sampling_kde_bw_method": 0.02,
    "data_sampling_density_quantile_threshold": 0.025,  # threshold = min of both
    "data_sampling_coords_for_density_threshold": 0.15,  # threshold = min of both
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
            '--loglevel', type=str, default='info', help='log level (default: info)'
        )
        self.parser.add_argument(
            '--device', type=str, default='cpu', help='jax device (default: cpu)'
        )
        self.parser.add_argument(
            '--data_path',
            type=str,
            default='./data/calibrated_data',
            help='path to xp data directory',
        )
        self.parser.add_argument(
            '--wandb_plot_period',
            type=int,
            default=-1,
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

        if self.args.compute_config_file is not None:
            if not Path(self.args.compute_config_file).is_file():
                raise ValueError(f'{self.args.compute_config_file} is not a file')
            self.compute_config = cmp.ComputeConfigManager.from_file(self.args.compute_config_file)
        else:
            self.compute_config = cmp.DEFAULT_COMPUTE_CONFIG

        if self.args.training_config_file is not None:
            if not Path(self.args.training_config_file).is_file():
                raise ValueError(f'{self.args.training_config_file} is not a file')
            self.training_config = json.load(open(self.args.training_config_file))
        else:
            self.training_config = DEFAULT_TRAINING_CONFIG

        if self.args.enable_checks:
            ut.set_enable_checks(True)

        self.local_save_dir = Path(self.args.local_save_dir)
        ut.logger.info(f"Saving results to {self.local_save_dir}")

        # loglevel
        ut.set_loglevel(self.args.loglevel)

        # device
        self.device = jax.devices(self.args.device)[0]

        if self.args.seed is not None:
            self.seed = self.args.seed
        else:
            self.seed = np.random.randint(0, 2**32)

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

    def start_training(self, training: du.DataManager, validation: du.DataManager = None):

        prog_config = self.args.__dict__.copy()

        self.training_config['program_config'] = prog_config

        if self.wandb_project is not None:
            loggers = setup_wandb_logging(
                self.wandb_project,
                training,
                self.training_config,
                self.compute_config,
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
                (-1, partial(local_save, save_dir=self.local_save_dir)),
            ]

        start(training, self.training_config, self.compute_config, loggers, seed=self.seed)


##────────────────────────────────────────────────────────────────────────────}}}
