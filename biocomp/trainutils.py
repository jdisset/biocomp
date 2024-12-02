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


@Partial(jit, static_argnums=(1,))
def compstats(v, smooth_win=1):
    medians = vmap(jnp.median)(v)
    mins = vmap(jnp.min)(v)
    maxs = vmap(jnp.max)(v)
    p20s = vmap(lambda x: jnp.percentile(x, 20))(v)
    p80s = vmap(lambda x: jnp.percentile(x, 80))(v)
    if smooth_win > 1:
        medians = jnp.convolve(medians, jnp.ones(smooth_win) / smooth_win, mode="same")
        p80s = jnp.convolve(p80s, jnp.ones(smooth_win) / smooth_win, mode="same")
        p20s = jnp.convolve(p20s, jnp.ones(smooth_win) / smooth_win, mode="same")
        maxs = jnp.convolve(maxs, jnp.ones(smooth_win) / smooth_win, mode="same")
        mins = jnp.convolve(mins, jnp.ones(smooth_win) / smooth_win, mode="same")
    return medians, p20s, p80s, mins, maxs


def get_epoch_stats(epoch_data, smooth_win=1):
    stats = {"grad": {}, "params": {}}
    if "grad" in epoch_data:
        for k, v in epoch_data["grad"]["shared"].items():
            stats["grad"][k] = compstats(v)
    if "params" in epoch_data:
        for k, v in epoch_data["params"]["shared"].items():
            stats["params"][k] = compstats(v)
    return stats


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
        local, _ = local_params.filter_by_tag(["local"])
        _, shared = params.filter_by_tag(["local"])
        params = ParameterTree.merge(local, shared)

    with ut.timer("wandb_plot_pred"):
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

        for chunk_id, XX in enumerate(tqdm(ALLX_CHUNKS, desc="wandb_plot_pred chunks")):
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
                img = wb.Image(fig, caption=f"{networks[index].name}, error={error:.4f}")
                plt.close()
                plt.close(fig)
                plt.close("all")
                return img, error

            except Exception as e:
                ut.logger.warning(f"Failed to plot prediction {index}: {e}")
                traceback.print_exc()
                return None, None

        pred = [plot_prediction(i) for i in tqdm(list(range(len(networks))))]
        pred = [p for p in pred if p[0] is not None]
        predimg, prederr = zip(*pred)

        if log_key is None:
            log_key = "Evaluations"

        wb.log({f"{log_key}": predimg})
        wb.log({f"{log_key}_err": prederr})


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
    compute_config: cmp.ComputeConfig,
    datamanager: du.DataManager,
    n_replicates: int,
    key: jnp.ndarray,
) -> Tuple[cmp.ComputeStack, ParameterTree]:
    stack = datamanager.build_compute_stack(compute_config)
    assert stack.init is not None
    with ut.timer("Stack initialization"):
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

    with ut.timer("Generating batches"):
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


def get_step_count(opt_state):
    # Navigate through the optimizer state chain until we find the adam state
    for state in opt_state:
        if hasattr(
            state, "count"
        ):  # Adam state has a count attribute, as well as most other optimizers
            return state.count
    return 0


def make_training_step(loss_func, optimizer, fields_to_keep_in_history=("loss",), scannable=True):
    def base_training_step(params, opt_state, x, y, z, key):
        static, dynamic = params.filter_by_tag(["non_grad", "local"])

        (loss, aux), grads = value_and_grad(loss_func, has_aux=True)(
            dynamic, static, x, y, z, key, opt_state[0].count
        )

        updates, opt_state = optimizer.update(grads, opt_state, dynamic)
        dynamic = optax.apply_updates(dynamic, updates)
        params = ParameterTree.merge(static, dynamic)
        res = {
            "params": params,
            "loss": loss,
            "grad": grads,
            "opt": opt_state,
            "x": x,
            "y": y,
            "z": z,
            "key": key,
            **aux,
        }
        return res

    training_step = base_training_step

    if scannable:

        def scannable_training_step(carry, i_x_y_z_k):
            params, opt_state = carry
            i, x, y, z, k = i_x_y_z_k
            updt = base_training_step(params, opt_state, x, y, z, k)
            params, opt_state = updt["params"], updt["opt"]
            history = {k: updt[k] for k in fields_to_keep_in_history}
            return (params, opt_state), history

        training_step = scannable_training_step

    return training_step


##────────────────────────────────────────────────────────────────────────────}}}
