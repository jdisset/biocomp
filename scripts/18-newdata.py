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
# {{{                   --     nn transform function     --
# ···············································································


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


def batches(X, Y, batch_size=64):
    # return dimensions should be (n_batches, n_models, batch_size, n_features)
    n_samples = X.shape[1]
    n_batches = n_samples // batch_size
    x_batches = X[:, : n_batches * batch_size].reshape((n_models, n_batches, batch_size, 1))
    y_batches = Y[:, : n_batches * batch_size].reshape((n_models, n_batches, batch_size, 1))
    return x_batches.transpose((1, 0, 2, 3)), y_batches.transpose((1, 0, 2, 3))


def plot_predictions(model, params, X, Y):
    Y_pred = vmap(model, in_axes=(None, 0, None))(params, X, rng)
    plt.figure(figsize=(10, 5))
    plt.scatter(X, Y, s=1, alpha=0.5, label='data')
    plt.scatter(X, Y_pred, s=1, alpha=0.5, label='predictions')
    plt.legend()


def models_data_fig(models, Y):
    for sample, model in models.items():
        out_proteins = model.get_output_proteins()
        in_proteins = model.get_inverted_input_proteins()
        z_prot = set(out_proteins) - set(in_proteins)
        print(f'{sample}: {in_proteins} -> {out_proteins} [diff:{z_prot}]')
        if len(z_prot) == 1 and len(in_proteins) == 2:
            fig, ax = du.model_heatmap(model, Y[sample])
            plt.show()
        else:
            fig, ax = du.model_parallel_coords(model, Y[sample])
            plt.show()
        yield sample, model, fig, ax


def plot_models_data(models, Y):
    for sample, model, fig, ax in models_data_fig(models, Y):
        # fig.savefig(f'./data/{sample}.png')
        plt.close(fig)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     config     --
# ···············································································

random.seed()
WSIZE = 64
DEPTH = 2
I_SIZE = 64
I_DEPTH = 1
node_impl = dict(
    bc.nodes.DEFAULT_COMPUTE_NODES_DICT,
    **{
        'output': partial(bn.output_nn, wsize=WSIZE, depth=DEPTH),
        'transcription': partial(
            transcription,
            outer_wsize=WSIZE,
            outer_depth=DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
        ),
        'translation': partial(
            translation,
            outer_wsize=WSIZE,
            outer_depth=DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
        ),
        'inv_transcription': partial(inv_transcription, outer_wsize=WSIZE, outer_depth=DEPTH),
        'inv_translation': partial(inv_translation, outer_wsize=WSIZE, outer_depth=DEPTH),
    },
)
cfg = {
    "optimizer": "adam",
    "learning_rate": 0.001,
    "adam_w_decay": 0.0001,
    "rng_key": np.random.randint(0, 2**32),
    "epochs":  500,
    "compile_training": True,
    "batch_size": 300,
    "norm_factor": 1e6,
    "balance_bin_resolution": 0.5,
    "balance_threshold_quantile": 0.4,
    "balance_threshold_min": 40,
    "node_impl": node_impl,
}


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     setup     --
# ···············································································
lib = ut.load_lib()
xp = ut.load_xp('E20221124A_ERNbandpassV2', lib)

rng = jax.random.PRNGKey(cfg['rng_key'])
models = xp.get_models(node_impl=cfg['node_impl'])
# NMODELS = 4
# models = {k: v for k, v in list(models.items())[:NMODELS]}

X, Y = bc.train.preprocess_data(models, xp.get_Y(models), cfg)
batch_size = cfg['batch_size'] // len(models)
x_batches, y_batches = du.make_batches_uniform_sampling(
    Y.values(), batch_size, rng, models.values()
)

# reduce x_batches to only 100
# x_batches = x_batches[:10]

# ut.plot_networks([m.network for m in models.values()])
# plot_models_data(models, Y)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     training     --
# ···············································································
import wandb as wb

project = 'bp_train_00'

wb.init(config=cfg, project=project, entity="jdisset", reinit=True)

def wandb_plot_pred(history, epoch, cfg, models, X, Y, project=None, **_):
    if epoch == 0 and project is not None:
        gtruth = []
        for sample, model, fig, ax in models_data_fig(models, Y):
            gtruth.append(wb.Image(fig, caption=f'{model.network.name} ground truth'))
            plt.close(fig)
        wb.log({'ground truth': gtruth}, step=epoch)

    params = history['params'][-1]
    jitted_models = {
        s: jit(jax.vmap(partial(m, rng_key=jax.random.PRNGKey(0)), in_axes=(None, 0)))
        for s, m in models.items()
    }

    Y_pred = {s: jitted_models[s](params, X[s]) for s in models}

    pred = []
    for sample, model, fig, ax in models_data_fig(models, Y_pred):
        pred.append(wb.Image(fig, caption=f'{model.network.name} predicted'))
        plt.close(fig)

    wb.log({'prediction': pred}, step=epoch)

def wandb_log_epoch(history, epoch, cfg, **_):
    loss = float(history['loss'][-1])
    params = history['params'][-1]
    wb.log({'loss': loss}, step=epoch)
    wb.log({'shared_params': params['shared']}, step=epoch)
    wb.log({'params': params}, step=epoch)

loggers = {
    1: bc.train.console_log,
    1: wandb_log_epoch,
    1: partial(wandb_plot_pred, models=models, X=X, Y=Y),
}

train_history = bc.train.train_models(models.values(), x_batches, y_batches, cfg, loggers)

wb.log({'train_history': train_history})

print('done')




#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                        --     toy problem     --
# ···············································································
# def P(name):
# return bc.Slot(lib, name)


# tus = {'A': bc.TranscriptionUnit([P('hEF1a'), P('NeonGreen')])}
# aggregations = [['A']]
# sources = {tu_name: [tu_name] for tu_name, tu in tus.items()}
# n = bc.Network.from_dict(lib, '', tus, sources, aggregations)
# inv = bc.inverted_network(n)
# n.set_inputs([5])

# m = bc.ComputeGraphModel(n)
# m.build(node_impl=node_impl)
# minv = bc.ComputeGraphModel(inv)
# minv.build(node_impl=node_impl)

# # let's generate some data that's just a scaled log of the input. Y should always be positive
# x0 = np.geomspace(1, 100, 5000)
# x0 = x0[np.random.permutation(len(x0))]
# x0 = x0.reshape((len(x0), 1))
# y0 = np.log(x0)

# models = [m, minv]
# X = jnp.array([x0, y0])  # shape is (2, 5000, 1) aka (n_models, n_samples, n_features)
# Y = jnp.array([y0, y0])

# # models = [m]
# # X = jnp.array([x0]) # shape is (2, 5000, 1) aka (n_models, n_samples, n_features)
# # Y = jnp.array([y0])

# treeshape = lambda t: jax.tree_map(lambda x: x.shape, t)

# n_models = len(models)
# x_batches, y_batches = batches(X, Y, batch_size=64)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     manual training     --
# ···············································································

# key = jax.random.PRNGKey(np.random.randint(0, 2**32))
# cfg = {**bc.train.DEFAULT_CFG, **cfg}
# optimizer = optax.adam(learning_rate=cfg['learning_rate'])
# # key = jax.random.PRNGKey(cfg['rng_key'])
# ikeys = jax.random.split(key, len(models))
# params, constraints = {}, {}
# print('Initializing parameters')
# for m, k in zip(models, ikeys):
# params, constraints = m.init(k, pre_params=params, pre_constraints=constraints)
# dynamic, _ = bu.split_params(params, cfg['static_params'])
# opt_state = optimizer.init(dynamic)
# history = {
# 'params': [params],
# 'opt': [opt_state],
# 'grad': [None],
# 'loss': [float('inf')],
# }


# def mse_loss(y, y_hat, n_outputs=None):
# if n_outputs is None:
# n_outputs = y.shape[1]
# assert y_hat.ndim == 2 and y.ndim == 2
# return jnp.mean((y[:, :n_outputs] - y_hat[:, :n_outputs]) ** 2)


# def loss_func(dynamic, static, X, Y, rng_key):
# nmodels = len(models)
# assert len(X) == nmodels, f"Expected {nmodels} models, got {X.shape}"
# assert len(Y) == nmodels
# params = bu.assemble_params(dynamic, static)
# K = jax.random.split(rng_key, nmodels)
# res = jnp.array(
# [
# mse_loss(vmap(partial(m, params, rng_key=k))(x[:, : m.n_inputs]), y, m.n_outputs)
# for m, x, y, k in zip(models, X, Y, K)
# ]
# ).mean()
# return res


# def training_step(params, opt_state, key, x, y):
# dynamic, static = bu.split_params(params, [['node']])
# loss, grads = jax.value_and_grad(loss_func)(dynamic, static, x, y, key)
# updates, opt_state = optimizer.update(grads, opt_state, dynamic)
# dynamic = optax.apply_updates(dynamic, updates)
# dynamic = bu.apply_constraints(dynamic, constraints)
# params = bu.assemble_params(dynamic, static)
# res = {
# 'params': params,
# 'loss': loss,
# 'grad': grads,
# 'opt': opt_state,
# }
# return res


# step = jit(training_step)


# print('Beginning training')
# for i, k in enumerate(jax.random.split(key, 20), 1):
# for x, y in tqdm(zip(x_batches, y_batches), total=len(x_batches), desc=f'Epoch {i}'):
# updt = step(params, opt_state, k, x, y)
# params, opt_state = updt['params'], updt['opt']
# print(updt['loss'])


# plot_predictions(m, params, X[0], Y[0])

# best_params = train_history['params'][-1]
# treeshape(best_params)
# ut.plot_node('transcription', best_params, m, vlim=[0, 100])
# ut.plot_node('translation', best_params, m, vlim=[0, 100])
# plot_predictions(m, best_params, X[0], Y[0])
# plot_predictions(minv, best_params, X[1], Y[1])
# out, allres = minv.collect_all_results(best_params, jnp.array([1.0]), rng)
# ut.plot_networks([minv.network], outputs=[allres], H=2000, figsize=(8, 15))
# out, allres = m.collect_all_results(best_params, jnp.array([1.0]), rng)
# ut.plot_networks([m.network], outputs=[allres], W=500)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
