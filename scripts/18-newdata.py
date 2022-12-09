## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     imports     --
# ···············································································
import biocomp as bc
import pandas as pd
import biocomp.compute as bcc
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import time
import biocomp.utils as bu
import scriptutils as ut
import jax
from jax import jit, vmap, grad, value_and_grad
import jax.numpy as jnp
import random
import biocomp.datautils as du
import optax
from tqdm import tqdm
import biocomp.nodes as bn
import biocomp.compute as bcc
import json5

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     setup     --
# ···············································································
lib = ut.load_lib()
xp = ut.load_xp('E20221124A_ERNbandpassV2', lib)


def inv_fused_nn(get_param, get_quantized, wsize=64, depth=2, **_):
    def apply(value, rng_key):
        k0, k1, k2 = jax.random.split(rng_key, 3)
        return value

    return apply


node_impl = dict(
    bc.nodes.DEFAULT_COMPUTE_NODES_DICT,
    **{
        'inv_fused': partial(inv_fused_nn, wsize=64, depth=2),
        'output': partial(bn.output_nn, wsize=64, depth=2),
        'transcription': partial(bn.transcription_nn, wsize=64, depth=2),
        'translation': partial(bn.translation_nn, wsize=64, depth=2),
    },
)

random.seed()
cfg = {
    "node_remap": {},
    "optimizer": "sgd",
    "learning_rate": 0.001,
    "adam_w_decay": 0.0001,
    "rng_key": np.random.randint(0, 2**32),
    "epochs": 10,
    "compile_training": True,
    "batch_size": 512,
    "norm_factor": 1e6,
    "balance_bin_resolution": 0.5,
    "balance_threshold_quantile": 0.4,
    "balance_threshold_min": 40,
    "node_impl": node_impl,
}

rng = jax.random.PRNGKey(cfg['rng_key'])
models = xp.get_models(node_impl=cfg['node_impl'])

for m in models.values():
    bc.network.fuse_consecutive(
        m.network.compute_graph, ("inv_translation", "inv_transcription"), "inv_fused"
    )
    m.build(node_impl=cfg['node_impl'])


m = models.values().__iter__().__next__()
n = m.network
ut.plot_networks([n])

X, Y = bc.train.preprocess_data(models, xp.get_Y(models), cfg)
batch_size = cfg['batch_size'] // len(models)
x_batches, y_batches = du.make_batches_uniform_sampling(
    Y.values(), batch_size, rng, models.values()
)

loggers = {
    1: bc.train.console_log,
    1: partial(bc.train.wandb_log_epoch, project='bp_train_00'),
    10: partial(bc.train.wandb_log_plot, project='bp_train_00', models=models, X=X, Y=Y),
}

# ut.plot_networks(xp.networks.values())

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     actual training     --
# ···············································································

train_history = bc.train.train_models(models.values(), x_batches, y_batches, cfg, loggers)

train_history
print('done')

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                     --     generate networks     --
# ···············································································

rng = jax.random.PRNGKey(1)


def transform_nn(
    get_param,
    get_quantized,
    transform_name,
    outer_wsize=64,
    outer_depth=2,
    outer_activation=jax.nn.relu,
    inner_wsize=32,
    inner_depth=2,
    inner_out=4,
    inner_activation=jax.nn.relu,
    rate_dim=1,
    tr_namespace='',
    **_,
):
    def inner(value, rate_embeding, key):
        # TODO idea: to give more flexibility, we could add the index of the
        # value as this might allow clever padding of the sum
        # we'd then need to make sure that the index is unique for each
        # while, probably, being random (to avoid any "preferred" order)
        """For a single source, computes a latent output from the concatenation of
        the rate embedding and the source value.
        All of these outputs will then be summed up and passed through a final layer.
        """
        if value.ndim == 0:
            value = value.reshape((1,))
        if rate_embeding.ndim == 0:
            rate_embeding = rate_embeding.reshape((1,))
        inputs = jnp.concatenate([value, rate_embeding], axis=-1)

        return bn.nn_dense_multilevel(
            inputs,
            inner_wsize,
            inner_out,
            depth=inner_depth,
            get_param=get_param,
            key=key,
            name=f'{tr_namespace}{transform_name}_inner',
            activation=inner_activation,
        )

    def apply(*values, rng_key):

        k0, k1, k2 = jax.random.split(rng_key, 3)
        val = jnp.array(values)

        rate_name = f'{transform_name}_rate'
        rate_shape = (val.shape[0], rate_dim)
        rates = get_quantized(
            rate_name,
            get_param(rate_name, init=bn.continuous_initializer(k0, rate_shape)),
            mode='input_edges',
        )

        assert val.shape[0] == rates.shape[0]

        # first we apply a simple inner layer to all inputs and sum them:
        inner_out = jnp.sum(jax.vmap(inner, in_axes=(0, 0, None))(val, rates, k1), axis=0)

        # then we apply a final outer layer to the summed output:
        return bn.nn_dense_multilevel(
            inner_out,
            outer_wsize,
            1,
            depth=outer_depth,
            get_param=get_param,
            key=k2,
            name=f'{tr_namespace}{transform_name}_outer',
            activation=outer_activation,
        )

    return apply


transcription = partial(transform_nn, transform_name='tc')
translation = partial(transform_nn, transform_name='tl')
inv_transcription = partial(transform_nn, transform_name='tc', tr_namespace='inv_')
inv_translation = partial(transform_nn, transform_name='tl', tr_namespace='inv_')


random.seed()
node_impl = dict(
    bc.nodes.DEFAULT_COMPUTE_NODES_DICT,
    **{
        # 'output': partial(bn.output_nn, wsize=64, depth=2),
        # 'transcription': transcription,
        # 'translation': translation,
        # 'inv_transcription': inv_transcription,
        # 'inv_translation': inv_translation,
    },
)
cfg = {
    "node_remap": {},
    "optimizer": "sgd",
    "learning_rate": 0.001,
    "adam_w_decay": 0.0001,
    "rng_key": np.random.randint(0, 2**32),
    "epochs": 10,
    "compile_training": True,
    "batch_size": 512,
    "norm_factor": 1e6,
    "balance_bin_resolution": 0.5,
    "balance_threshold_quantile": 0.4,
    "balance_threshold_min": 40,
    "node_impl": node_impl,
}


def P(name):
    return bc.Slot(lib, name)
tus = {'A': bc.TranscriptionUnit([P('hEF1a'), P('NeonGreen')])}
aggregations = [['A']]
sources = {tu_name: [tu_name] for tu_name, tu in tus.items()}
n = bc.Network.from_dict(lib, '', tus, sources, aggregations)
inv = bc.inverted_network(n)
n.set_inputs([5])
m = bc.ComputeGraphModel(n)
m.build(node_impl=node_impl)
minv = bc.ComputeGraphModel(inv)
minv.build(node_impl=node_impl)


params, _ = m.init(rng)
def pytree_keys(pytree):
    return jax.tree_map(lambda x: x.shape, pytree)
pytree_keys(params)

inv_params, _ = minv.init(rng)
pytree_keys(inv_params)

m.collect_all_results(params, np.array([1.0]), rng)
minv.collect_all_results(inv_params, np.array([1.0]), rng)

ut.plot_networks([n, inv])
# ut.plot_node('translation', params, m)

# let's generate some data that's just a scaled log of the input. Y should always be positive
x0 = np.geomspace(1, 100, 10000)
x0 = x0[np.random.permutation(len(x0))]
x0 = x0.reshape((len(x0), 1))
y0 = np.log(x0)

models = [m, minv]
X = jnp.array([x0,y0]) # shape is (2, 5000, 1) aka (n_models, n_samples, n_features)
Y = jnp.array([y0,y0])

models = [m]
X = jnp.array([x0]) # shape is (2, 5000, 1) aka (n_models, n_samples, n_features)
Y = jnp.array([y0])

n_models = len(models)

def batches(X, Y, batch_size=64):
    # return dimensions should be (n_batches, n_models, batch_size, n_features)
    n_samples = X.shape[1]
    n_batches = n_samples // batch_size
    x_batches = X[:, :n_batches*batch_size].reshape((n_models, n_batches, batch_size, 1))
    y_batches = Y[:, :n_batches*batch_size].reshape((n_models, n_batches, batch_size, 1))
    return x_batches.transpose((1, 0, 2, 3)), y_batches.transpose((1, 0, 2, 3))

x_batches, y_batches = batches(X, Y, batch_size=64)


loggers = {
    1: bc.train.console_log,
    # 1: partial(bc.train.wandb_log_epoch, project='bp_train_00'),
}

train_history = bc.train.train_models(models, x_batches, y_batches, cfg, loggers)
train_history

best_params = train_history['params'][-1]

ut.plot_node('transcription', best_params, m)

# plot predictions

def plot_predictions(model, params, X, Y):
    Y_pred = vmap(model, in_axes=(None, 0, None))(params, X, rng)
    plt.figure(figsize=(10, 5))
    plt.scatter(X, Y, s=1, alpha=0.5, label='data')
    plt.scatter(X, Y_pred, s=1, alpha=0.5, label='predictions')
    plt.legend()

plot_predictions(m, best_params, X[0], Y[0])





#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

