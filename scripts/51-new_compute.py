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
# dman = dman_full
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

### {{{                   --     manual training loop     --
import biocomp.train as tr
import optax
import time

key = jax.random.PRNGKey(config['rng_key'])
stack = dman.get_compute_stack()

config['static_params'] = ['/__static__','/node']

dman.data_cfg['batch_size'] = 16
dman.data_cfg['n_batches'] = 2048

## --- init

with ut.timer('Stack initialization'):
    params = stack.init(jax.random.PRNGKey(0))
    optimizer = tr.get_optimizer(config)
    dynamic, _ = ut.split_params(params, config['static_params'])
    opt_state = optimizer.init(dynamic)

## --- batches
with ut.timer('Getting batches'):
    xbatches, ybatches = dman.get_batches(key)  # (B,M,N,F) shape

total_batches = config['n_batches']
assert total_batches == xbatches.shape[0] == ybatches.shape[0]
nbatches_per_epoch = total_batches // config['n_epochs_per_batch_rotation']

## --- loss and updates

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

xbatches.shape

config['epochs'] = 100

nbatches_per_epoch

for i, epoch_key in enumerate(jax.random.split(key, config['epochs']), 1):
    t0 = time.time()
    batch_rotation = i % config['n_epochs_per_batch_rotation']
    start_idx = batch_rotation * nbatches_per_epoch
    end_idx = start_idx + nbatches_per_epoch
    params, opt_state, epoch_history = epoch_step(
        params, opt_state, epoch_key, xbatches[start_idx:end_idx], ybatches[start_idx:end_idx]
    )
    print(f'Epoch {i} took {time.time() - t0:.2f} seconds')
    print(f'Epoch {i} loss: {epoch_history["loss"].mean()}')



##────────────────────────────────────────────────────────────────────────────}}}

#TODO:
# tl-rates depend on anything on the 5'. 
# Oh but then you'll have a special case for every combination of uOrf and recog site
# one way to solve that would be to do some kind of arithmetic on the rates or a multi-channel param (one for recognition, one for uorf)


### {{{                    --     testing subnetworks     --

from jax.experimental import checkify
errors = checkify.user_checks | checkify.index_checks | checkify.float_checks | checkify.nan_checks

params = jax.tree_util.tree_map(lambda x: jnp.array(x), params)

dman.build_compute_stack()
c = dman.get_compute_stack()
full_input = jax.random.uniform(jax.random.PRNGKey(0), (c.total_nb_of_inputs,))
full_quantile = jax.random.uniform(jax.random.PRNGKey(0), (c.total_nb_of_outputs,))
checked_apply = checkify.checkify(c.apply, errors=errors)
err, full_res = checked_apply(params, full_input, full_quantile, jax.random.PRNGKey(0))
err.throw()
print('ok')


def get_stack(dman, net_id, params):
    stack, pf = dman.get_individual_compute_stack(net_id)
    p = pf(params)
    return stack, p

input_start = 0
for i in range(len(dman.get_networks())):
    stack, p = get_stack(dman, i, params)
    input_end = input_start + stack.total_nb_of_inputs
    output_start = c.get_network_global_output_id(i)
    output_end = output_start + stack.total_nb_of_outputs
    inp = full_input[input_start:input_end]
    quantile = full_quantile[output_start:output_end]
    all_node_ids = [n.node_id for n in stack.each_node()]
    checked_apply_local = checkify.checkify(stack.apply, errors=errors)
    err, res = checked_apply_local(p, inp, quantile, jax.random.PRNGKey(0))
    err.throw()
    print(f'res = {res}')
    desired_output = full_res[output_start:output_end]
    assert np.allclose(res, desired_output)
    input_start = input_end

print('done checking')
##────────────────────────────────────────────────────────────────────────────}}}

fig, ax = du.report(params, dman, 0)
