#### {{{                          --     imports     --
from biocomp import utils as ut
from contextlib import contextmanager
import scriptutils as su
from pathlib import Path
import jax.numpy as jnp
import numpy as np
import jax
from jax import jit, vmap, value_and_grad
from jax.tree_util import Partial as partial
from tqdm import tqdm
import biocomp.defaults as bdf
import pandas as pd
from rich import print as pprint
import biocomp.datautils as du


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                      --     loading csy4 matrix xp     --
with ut.timer('from zero to ready-to-compile'):
    config = bdf.DEFAULT_CONFIG
    lib = su.load_lib()
    matrix_xp = su.load_xp('2023-02-16_Matrix', lib, data_path='./data/calibrated_data')
    dman_full = du.DataManager.from_xps([matrix_xp], config, inverse='all')

dman = dman_full.make_subset(list(range(10)))
##────────────────────────────────────────────────────────────────────────────}}}

### {{{                      --     quantify uorfs     --

def get_uorf_value(param):
    if 'tl_rate' in param:
        u = param['tl_rate'][0].split('_')[0]
        try:
            v = int(u[:-1]) * 10
        except ValueError:
            v = 0
        if u[-1] == 'w':
            v = v - 5
        return v
    else:
        return 0


def get_uorf_values(network):
    cdg = network.central_dogma_graph
    ERN_inputs = network.compute_graph[network.compute_graph['type'] == 'sequestron_ERN'][
        'cdg_input'
    ].values[0]
    cdgin = cdg.loc[ERN_inputs]
    ern_side = cdg.loc[cdgin.iloc[0].predecessor[0]]
    recog_side = cdgin.iloc[1]
    values = (get_uorf_value(ern_side.params), get_uorf_value(recog_side.params))
    return values


uorf_dict = {}
for i, n in enumerate(dman.get_networks()):
    has_ERN_node = n.compute_graph['type'] == 'sequestron_ERN'
    if has_ERN_node.any():
        uorf_dict[get_uorf_values(n)] = i

uorf_dict

##────────────────────────────────────────────────────────────────────────────}}}

import biocomp.train as tr
import optax
import time

key = jax.random.PRNGKey(config['rng_key'])
stack = dman.get_compute_stack()

config['static_params'] = ['/__static__','/node']



dman.data_cfg['batch_size'] = 8
dman.data_cfg['n_batches'] = 32

# --- init

with ut.timer('Stack initialization'):
    params = stack.init(jax.random.PRNGKey(0))

optimizer = tr.get_optimizer(config)
dynamic, _ = ut.split_params(params, config['static_params'])
opt_state = optimizer.init(dynamic)

# batches
with ut.timer('Getting batches'):
    xbatches, ybatches = dman.get_batches(key)  # (B,M,N,F) shape

total_batches = config['n_batches']
assert total_batches == xbatches.shape[0] == ybatches.shape[0]
nbatches_per_epoch = total_batches // config['n_epochs_per_batch_rotation']

# --- loss and updates
##

vmapped_compute = jax.vmap(stack.apply, in_axes=(None, 0, 0, 0))

def loss_func(dynamic, static, X, Y, Z, key):
    assert X.ndim == Y.ndim == Z.ndim == 2
    assert X.shape[0] == Y.shape[0] == Z.shape[0]
    assert X.shape[1] == sum([n.get_nb_inputs() for n in stack.networks])
    assert Y.shape[1] == Z.shape[1] == sum([n.get_nb_outputs() for n in stack.networks])

    params = ut.assemble_params(dynamic, static)
    keys = jax.random.split(key, X.shape[0])

    yhat = vmapped_compute(params, X, Z, keys)
    assert yhat.shape == Y.shape

    error = yhat - Y
    quantile_loss = jnp.mean(
        tr.huber_quantile_loss(error, Z, delta=config['huber_quantile_loss_delta'])
    )

    return quantile_loss

def training_step(params, opt_state, x, y, z, key):
    dynamic, static = ut.split_params(params, config['static_params'])
    loss, grads = value_and_grad(loss_func)(dynamic, static, x, y, z, key)
    updates, opt_state = optimizer.update(grads, opt_state, dynamic)

    dynamic = optax.apply_updates(dynamic, updates)
    params = ut.assemble_params(dynamic, static)

    res = {
        'params': params,
        'loss': loss,
        'grad': grads,
        'opt': opt_state,
    }
    return res

@ut.progress_scan(nbatches_per_epoch, message='Training model')
def scannable_step(carry, i_x_y_z_k):
    params, opt_state = carry
    i, x, y, z, k = i_x_y_z_k
    updt = training_step(params, opt_state, x, y, z, k)
    params, opt_state = updt['params'], updt['opt']
    return (params, opt_state), updt

@jit
def epoch_step(start_params, start_opt_state, epoch_key, xbs, ybs):
    zbatches = jax.random.uniform(epoch_key, ybs.shape)
    batch_keys = jax.random.split(epoch_key, nbatches_per_epoch)
    (final_params, final_opt_state), epoch_history = jax.lax.scan(
        scannable_step,
        (start_params, start_opt_state),
        (jnp.arange(nbatches_per_epoch), xbs, ybs, zbatches, batch_keys),
    )
    return final_params, final_opt_state, epoch_history


config['epochs'] = 5

for i, epoch_key in enumerate(jax.random.split(key, config['epochs']), 1):
    t0 = time.time()
    batch_rotation = i % config['n_epochs_per_batch_rotation']
    start_idx = batch_rotation * nbatches_per_epoch
    end_idx = start_idx + nbatches_per_epoch
    params, opt_state, epoch_history = epoch_step(
        params, opt_state, epoch_key, xbatches[start_idx:end_idx], ybatches[start_idx:end_idx]
    )
    print(f'Epoch {i} took {time.time() - t0:.2f} seconds')




##






if loggers is None:
    loggers = [
        (1, console_log),
    ]

print('Initial logger calls')
for _, l in loggers:
    l(0, config)

print(f'Begin training for {config["epochs"]} epochs')

for i, epoch_key in enumerate(jax.random.split(key, config['epochs']), 1):
    t0 = time.time()
    batch_rotation = i % config['n_epochs_per_batch_rotation']
    start_idx = batch_rotation * nbatches_per_epoch
    end_idx = start_idx + nbatches_per_epoch
    params, opt_state, epoch_history = epoch_step(
        params, opt_state, epoch_key, xbatches[start_idx:end_idx], ybatches[start_idx:end_idx]
    )
    epoch_history['epoch_time'] = time.time() - t0
    for t, l in loggers:
        if i % t == 0 or i == cfg['epochs']:
            l(i, config, epoch_history=epoch_history, nbatches=nbatches_per_epoch)
