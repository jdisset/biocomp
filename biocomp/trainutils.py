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

### {{{                       --     logging tools     --


def initialize_wandb(project, entity, full_config, **kw):
    import wandb as wb

    wb.init(config=full_config, project=project, entity=entity, **kw)
    return wb


def setup_wandb_logging(
    project,
    training_config,
    compute_config,
    data_config,
    params_save_period=-1,  # only at the end
    entity='jdisset',
    **kw,
):

    full_config = {**training_config, **compute_config.config, **data_config}
    wb = initialize_wandb(project, entity, full_config, **kw)
    save_dir = Path(wb.run.dir)
    loggers = [
        (1, console_log),
        (
            params_save_period,
            partial(
                local_save,
                compute_config=compute_config,
                training_config=training_config,
                data_config=data_config,
                save_dir=save_dir,
            ),
        ),
        (1, wandb_log_epoch),
    ]
    return loggers


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
    epoch,
    compute_config,
    training_config,
    data_config,
    epoch_history=None,
    save_dir=None,
    full_save=False,
    **_,
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

    if compute_config is not None:
        compute_conf_path = Path(save_dir) / 'compute_config.json'
        if not compute_conf_path.exists():
            compute_config.export(compute_conf_path)


    if training_config is not None:
        training_conf_path = Path(save_dir) / 'training_config.json'
        if not training_conf_path.exists():
            with open(training_conf_path, 'w') as f:
                json.dump(training_config, f)

    if data_config is not None:
        data_conf_path = Path(save_dir) / 'data_config.json'
        if not data_conf_path.exists():
            with open(data_conf_path, 'w') as f:
                json.dump(data_config, f)


    if full_save:
        full_save_until_epoch = full_save if isinstance(full_save, int) else 2
        if epoch <= full_save_until_epoch:
            ut.save(epoch_history, f'{save_dir}/epoch_{epoch}_full.pkl')

    params = epoch_history['latest_params']

    # first we rename the old params
    for f in Path(save_dir).glob('latest_params.pkl'):
        f.rename(f'{save_dir}/old_params.pkl')

    # then we save the new ones
    ut.save(params, f'{save_dir}/latest_params.pkl')

    # then we delete the old one
    for f in Path(save_dir).glob('old_params.pkl'):
        f.unlink()

    ut.logger.info(f"Saving epoch to disk took {time.time() - t0:.2f}s")


def wandb_plot_pred(
    dman: du.DataManager, params: ParameterTree, local_params=None, log_key=None, **_
):

    import matplotlib

    # matplotlib.pyplot.switch_backend('Agg')
    import traceback
    from tqdm import tqdm

    networks = dman.get_networks()
    stack = dman.get_compute_stack()

    if local_params is not None:
        local, _ = local_params.filter_by_tag(['local'])
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
                fig = pu.report(params, dman, index, use_x_y_yhat=(x, y, yhat), res=64)
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
        losses = np.array(epoch_history['loss'])
        # shape of losses = (n_replicates, n_batches)
        if losses.ndim == 1:
            for loss in losses:
                wb.log({'loss': loss})
        else:
            epoch = epoch_history.get('epoch', 0)
            mean_loss = np.mean(losses, axis=0)
            std_loss = np.std(losses, axis=0)
            min_loss = np.min(losses, axis=0)
            max_loss = np.max(losses, axis=0)

            for i in range(mean_loss.shape[0]):
                wb.log(
                    {
                        'loss/avg': mean_loss[i],
                        'loss/std': std_loss[i],
                        'loss/min': min_loss[i],
                        'loss/max': max_loss[i],
                    }
                )

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
            f"""[{epoch}/{training_config["n_epochs"]} in {epoch_history["epoch_time"]:.2f}s]
             best loss: {fmt(avg_losses[best_id])} ± {fmt(best_std)} (replicate n° {best_id+1}/{len(losses)})
             replicates avg: {fmt(avg_avg)} ± {fmt(avg_std)} """
        )


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                       --     get_optimizer     --


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


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                    --     base loss functions     --


def mse_loss(y, y_hat, n_outputs=None):
    if n_outputs is None:
        n_outputs = y.shape[1]
    assert y_hat.ndim == 2 and y.ndim == 2
    return jnp.mean((y[:, :n_outputs] - y_hat[:, :n_outputs]) ** 2)


def huber_quantile_loss(e, q, delta=0.1):
    return jnp.where(
        jnp.abs(e) <= delta, 0.5 * e**2, delta * (jnp.abs(e) - 0.5 * delta)
    ) * jnp.where(e < 0, q, (1.0 - q))


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                     --     helper functions     --

ndArray = Union[jnp.ndarray, np.ndarray]


def init_stack(
    compute_config: cmp.ComputeConfigManager,
    datamanager: du.DataManager,
    n_replicates: int,
    key: jnp.ndarray,
) -> Tuple[cmp.ComputeStack, ParameterTree]:

    stack = datamanager.build_compute_stack(compute_config)
    assert stack.init is not None
    with ut.timer('Stack initialization'):
        params = vmap(stack.init)(jax.random.split(key, n_replicates))
    return stack, params


def generate_batches(
    datamanager: du.DataManager,
    n_replicates: int,
    n_batches: int,
    batch_size: int,
    key: ndArray,
) -> Tuple[ndArray, ndArray]:

    total_n_batches = n_replicates * n_batches

    with ut.timer('Generating batches'):
        xbatches, ybatches = datamanager.get_batches(total_n_batches, batch_size, key)
    # current shape is (R*B,N,F), final shape should be (R,B,N,F)
    # R: replicates, B: batches, N: data, F: features
    xbatches = xbatches.reshape(n_replicates, n_batches, *xbatches.shape[1:])
    ybatches = ybatches.reshape(n_replicates, n_batches, *ybatches.shape[1:])

    assert isinstance(xbatches, ndArray)
    assert isinstance(ybatches, ndArray)

    assert xbatches.shape[:-1] == (
        n_replicates,
        n_batches,
        batch_size,
    )
    assert ybatches.shape[:-1] == (
        n_replicates,
        n_batches,
        batch_size,
    )

    return xbatches, ybatches


def make_training_step(loss_func, optimizer, fields_to_keep_in_history=('loss',), scannable=True):

    def base_training_step(params, opt_state, x, y, z, key):
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

    training_step = base_training_step

    if scannable:

        def scannable_training_step(carry, i_x_y_z_k):
            params, opt_state = carry
            i, x, y, z, k = i_x_y_z_k
            updt = base_training_step(params, opt_state, x, y, z, k)
            params, opt_state = updt['params'], updt['opt']
            history = {k: updt[k] for k in fields_to_keep_in_history}
            return (params, opt_state), history

        training_step = scannable_training_step

    return training_step


##────────────────────────────────────────────────────────────────────────────}}}
