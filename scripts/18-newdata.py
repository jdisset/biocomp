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


import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10.0, 10.0]
plt.rcParams['figure.dpi'] = 300

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


def P(name):
    return bc.Slot(lib, name)


def batch(X, Y, batch_size):
    # We will yield batches of shape (n_models, batch_size, n_features)
    n_models, n_samples, n_features = X.shape
    n_batches = n_samples // batch_size
    for i in range(n_batches):
        x_batch = X[:, i * batch_size : (i + 1) * batch_size, :]
        y_batch = Y[:, i * batch_size : (i + 1) * batch_size, :]
        yield x_batch, y_batch

def get_batches(X, Y, batch_size):
    # return all the batches at once
    x_batches = []
    y_batches = []
    for x_batch, y_batch in batch(X, Y, batch_size):
        x_batches.append(x_batch)
        y_batches.append(y_batch)
    return jnp.array(x_batches), jnp.array(y_batches)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     config     --
# ···············································································

random.seed()
WSIZE = 128
DEPTH = 2
I_SIZE = 128
I_DEPTH = 2
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
    # "rng_key": np.random.randint(0, 2**32),
    "rng_key": 130625,
    "epochs": 500,
    "compile_training": True,
    "batch_size": 300,
    "norm_factor": 1e6,
    "balance_bin_resolution": 0.5,
    "balance_threshold_quantile": 0.4,
    "balance_threshold_min": 40,
    "node_impl": node_impl,
}

lib = ut.load_lib()
rng = jax.random.PRNGKey(cfg['rng_key'])

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     setup     --
# ···············································································
xp = ut.load_xp('E20221124A_ERNbandpassV2', lib)

rng = jax.random.PRNGKey(cfg['rng_key'])
models = xp.get_models(node_impl=cfg['node_impl'])
# NMODELS = 4
# models = {k: v for k, v in list(models.items())[:NMODELS]}

X, Y = bc.train.preprocess_data(models, xp.get_Y(models), cfg)
X
batch_size = cfg['batch_size'] // len(models)
x_batches, y_batches = du.make_batches_uniform_sampling(
    Y.values(), batch_size, rng, models.values()
)

# reduce x_batches to only 100
# x_batches = x_batches[:10]

# ut.plot_networks([m.network for m in models.values()], H=800, W=500, figsize=(5, 8))
# ut.plot_networks([m.network for m in models.values()])  # , H=800, W=500, figsize=(5, 8))
# ut.plot_networks([m.network for m in models.values()], [f'{save_path}/{name}.png' for name in models.keys()])
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
# {{{                        --     toy problem 2    --
# ···············································································

tus = {
    'A': bc.TranscriptionUnit([P('hEF1a'), P('NeonGreen')]),
    'B': bc.TranscriptionUnit([P('hEF1a'), P('eBFP')]),
}
aggregations = [['A', 'B']]
sources = {tu_name: [tu_name] for tu_name, tu in tus.items()}
n = bc.Network.from_dict(lib, '', tus, sources, aggregations)
ratio = [2.2, 1.0]
n.compute_graph.loc[9]['extra']['ratios'] = ratio
n.compute_graph
inv = bc.inverted_network(n)
inv.compute_graph.extra.tolist()
inv.compute_graph.at[15, 'extra']['input_from_output'] = 0
inv.compute_graph.at[11, 'extra']['original_output_slot'] = 0
# there are 2 possibilities for the inversion:
# since we have 2 fluo prots coming straight out of the aggregation node
# we should have 2 possible inverse path, one per protein.
# They should be equivalent (however some differences might appear in practice, and
# for final training it would be better to try all combinations, or, at least,
# get the highest ratio (probably)
# Here we can manually force the generation of the 2 like so:
inv2 = bc.inverted_network(n)
inv2.compute_graph.at[15, 'extra']['input_from_output'] = 1
inv2.compute_graph.at[11, 'extra']['original_output_slot'] = 1

n.set_inputs([10])
m = bc.ComputeGraphModel(n)
m.build(node_impl=node_impl)
minv = bc.ComputeGraphModel(inv)
minv.build(node_impl=node_impl)
minv2 = bc.ComputeGraphModel(inv2)
minv2.build(node_impl=node_impl)

# let's generate some data that's just a scaled log of the input. Y should always be positive
x0 = np.geomspace(1, 100, 5000)
x0 = x0[np.random.permutation(len(x0))]
y0 = jnp.array([np.log(x0 * ratio[0]), np.log(x0 * ratio[1])]).T
x0 = x0.reshape((len(x0), 1))

save_path = '/Users/jeandisset/Documents/nets_for_pres'
from pathlib import Path

# make path :
Path(save_path).mkdir(parents=True, exist_ok=True)


models = [m, minv, minv2]
ut.plot_networks([n, inv, inv2])
# ut.plot_networks([n, inv, inv2], [f'{save_path}/{i}.png' for i in range(3)])

X = jnp.array([x0, y0[:, 0, None], y0[:, 1, None]])  # shape is (n_models, n_samples, n_features)
Y = jnp.array([y0, y0, y0])

x_batches, y_batches = get_batches(X, Y, batch_size=64)

n_models = len(models)

print(f'X.shape: {X.shape}')
print(f'Y.shape: {Y.shape}')

loggers = {
    1: bc.train.console_log,
}

cfg['epochs'] = 10
train_history = bc.train.train_models(models, x_batches, y_batches, cfg, loggers)

best_params = train_history['params'][-1]
best_params
ut.plot_node('transcription', best_params, m, vlim=[0, 100])
ut.plot_node('translation', best_params, m, vlim=[0, 100])
X[0].shape
Y[0].shape


def plot_predictions(model, params, X, Y):
    Y_pred = vmap(model, in_axes=(None, 0, None))(params, X, rng)
    plt.figure(figsize=(10, 5))
    plt.scatter(X, Y, s=1, alpha=0.5, label='data')
    plt.scatter(X, Y_pred, s=1, alpha=0.5, label='predictions')
    plt.legend()


def plot_predictions_2d(model, params, X, Y):
    # X is 1D, Y is 2D
    Y_pred = vmap(model, in_axes=(None, 0, None))(params, X, rng)
    plt.figure(figsize=(10, 5))
    datacolor = 'tab:blue'
    plt.scatter(X, Y[:, 0], s=20, alpha=0.25, label='data_1', color=datacolor, marker='o')
    plt.scatter(X, Y[:, 1], s=20, alpha=0.25, label='data_2', color=datacolor, marker='x')
    predcolor = 'tab:orange'
    plt.scatter(
        X, Y_pred[:, 0], s=20, alpha=0.25, label='predictions_1', color=predcolor, marker='o'
    )
    plt.scatter(
        X, Y_pred[:, 1], s=20, alpha=0.25, label='predictions_2', color=predcolor, marker='x'
    )
    plt.legend()


plot_predictions_2d(m, best_params, X[0], Y[0])
plot_predictions_2d(minv, best_params, X[1], Y[1])
plot_predictions_2d(minv2, best_params, X[2], Y[2])


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

# ----------------- more complex synthetic data ----------------------

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{            --     some base functions for generation     --
# ···············································································
def hill_eq(x, k, n):
    return x**n / (k**n + x**n)


@jit
@vmap
def dna_to_rna(x):
    return hill_eq(x, 2.5, 2) * 10


@jit
@vmap
def rna_to_protein(x):
    return hill_eq(x, 5, 1.5) * 10


@jit
@vmap
def protein_to_fluo(x):
    return hill_eq(x, 1.75, 2) * 10


@jit
@vmap
def simple_ern(neg, pos):
    return jax.nn.relu(pos - neg)


# plot rna and protein production
x = np.linspace(0, 10, 1000)
plt.plot(x, dna_to_rna(x))
plt.plot(x, rna_to_protein(x))
plt.legend(['dna -> rna', 'rna -> protein'])


def plot3d(f, ax):
    x = np.linspace(0, 10, 1000)
    y = np.linspace(0, 10, 1000)
    # set projection to 3d
    ax = plt.axes(projection='3d')
    X, Y = np.meshgrid(x, y)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    Z = f(xy[:, 0], xy[:, 1]).reshape(X.shape)
    surf = ax.plot_surface(X, Y, Z, cmap='YlGnBu_r', linewidth=0, antialiased=False)
    ax.set_xlabel('ERN in')
    ax.set_ylabel('RNA in')
    ax.set_zlabel('RNA out')


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                --     synthetic network --
# ···············································································

tus = {
    'Marker_ERN': bc.TranscriptionUnit([P('hEF1a'), P('mKate')]),
    'Marker_rec': bc.TranscriptionUnit([P('hEF1a'), P('eBFP')]),
    'rec': bc.TranscriptionUnit([P('hEF1a'), P('CasE_rec'), P('NeonGreen')]),
    'ERN': bc.TranscriptionUnit([P('hEF1a'), P('CasE')]),
}
aggregations = [['Marker_ERN', 'ERN'], ['Marker_rec', 'rec']]
sources = {tu_name: [tu_name] for tu_name, tu in tus.items()}
n = bc.Network.from_dict(lib, '', tus, sources, aggregations)
inv = bc.inverted_network(n)
n.set_numeric_as_input()

# ut.plot_networks([n, inv])

# random samples in [0,10]
nsamples = 2000
dna_0 = jax.random.uniform(rng, (nsamples, 2), minval=0, maxval=10)
dna_1 = jax.random.uniform(rng, (nsamples, 2), minval=0.5, maxval=4)
dna_2 = jax.random.uniform(rng, (nsamples, 2), minval=0.5, maxval=3)
dna = jnp.concatenate([dna_0, dna_1, dna_2])
# actually we want a logarithmic distribution (the closer to 0, the more likely)
# dna = np.random.lognormal(0, 1, size=(nsamples, 2))

# space = jnp.linspace(0,10,120)
# dna = jnp.array(jnp.meshgrid(space, space)).T.reshape(-1, 2)
# dna = dna[jax.random.permutation(rng, dna.shape[0]), :]

rna = dna_to_rna(dna)
prt = rna_to_protein(rna)
out = {
    "NeonGreen": rna_to_protein(simple_ern(prt[:, 0], rna[:, 1])),
    "mKate": prt[:, 0],
    "eBFP": prt[:, 1],
}


# plot the data
# we want to plot:
# - rna from dna
# - protein from rna
# - output from input

fig, axs = plt.subplots(4, 1, figsize=(5, 20))
axs[0].set_xlabel('dna')
axs[0].set_ylabel('rna')
axs[0].scatter(dna[:, 0], rna[:, 0], s=1, alpha=0.5, label='rna_1')
axs[0].scatter(dna[:, 1], rna[:, 1], s=1, alpha=0.5, label='rna_2')
axs[0].set_xlim(0, 12)
axs[0].set_ylim(0, 12)

axs[1].set_xlabel('dna')
axs[1].set_ylabel('protein')
axs[1].scatter(dna[:, 0], out['mKate'], s=1, alpha=0.5, label='mKate')
axs[1].scatter(dna[:, 1], out['eBFP'], s=1, alpha=0.5, label='eBFP')
axs[1].set_xlim(0, 12)
axs[1].set_ylim(0, 12)

axs[2].scatter(
    out['mKate'], rna[:, 1], c=out['NeonGreen'], s=1, alpha=1, label='NeonGreen', cmap='YlGnBu_r'
)
axs[2].set_xlabel('ern')
axs[2].set_ylabel('rna')
axs[2].set_title('remaining rna after ern sequestration')
axs[2].set_xlim(0, 10)
axs[2].set_ylim(0, 10)

axs[3].scatter(
    out['mKate'], out['eBFP'], c=out['NeonGreen'], s=1, alpha=1, label='NeonGreen', cmap='YlGnBu_r'
)
axs[3].set_xlabel('mKate')
axs[3].set_ylabel('eBFP')
axs[3].set_title('output from input (fluo)')
axs[3].set_xlim(0, 10)
axs[3].set_ylim(0, 10)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                    --     now we try to train     --
# ···············································································

m = bc.ComputeGraphModel(n)
m.build(node_impl=node_impl)
minv = bc.ComputeGraphModel(inv)
minv.build(node_impl=node_impl)
models = [m, minv]
# we have to be careful on the order of outputs
# shape of X and Y is (n_models, n_samples, n_features)
assert m.get_output_proteins() == minv.get_output_proteins()
y = jnp.array([out[p] for p in minv.get_output_proteins()]).T
assert jnp.all(jnp.array([out['mKate'], out['eBFP']]).T == minv.get_input_from_output(y))
X = jnp.array([dna, minv.get_input_from_output(y)])
Y = jnp.array([y, y])

selected = (0,2)
models = models[selected[0]:selected[1]]
X = X[selected[0]:selected[1]]
Y = Y[selected[0]:selected[1]]

n_models = len(models)
x_batches, y_batches = get_batches(X, Y, batch_size=64)

loggers = {
    1: bc.train.console_log,
}

cfg['epochs'] = 50
train_history = bc.train.train_models(models, x_batches, y_batches, cfg, loggers)


fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(train_history['loss'])
# log scale for y
ax.set_yscale('log')
ax.set_xlabel('epoch')
ax.set_ylabel('loss')

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                           --     plots     --
#···············································································

best_params = train_history['params'][-1]

fig, ax = ut.plot_node('translation', best_params, m, xlim=[-10, 10], ylim=[-5, 25])
x = jnp.linspace(0, 10, 100)
y = dna_to_rna(x)
ax.plot(x, y, color='k', label='dna_to_rna')
ax.legend()

fig, ax = ut.plot_node('transcription', best_params, m, xlim=[-10, 10], ylim=[-5, 25])
x = jnp.linspace(-10, 10, 100)
y = rna_to_protein(x)
ax.plot(x, y, color='k', label='rna_to_prt')
ax.legend()

fig, ax = ut.plot_node('output', best_params, m, n_inputs=1, xlim=[-10,10], ylim=[-5, 25])
x = jnp.linspace(-10, 10, 100)
y = x
ax.plot(x, y, color='k', label='data')
ax.legend()

ut.plot_node('sequestron_ERN', best_params, m, n_inputs=2, mode='3d')

def plot_predictions(model, params, X, Y):
    Y_pred = vmap(model, in_axes=(None, 0, None))(params, X, rng)
    plt.figure(figsize=(10, 5))
    plt.scatter(X, Y, s=1, alpha=0.5, label='data')
    plt.scatter(X, Y_pred, s=1, alpha=0.5, label='predictions')
    plt.legend()

def plot_predictions_2d(model, params, X, Y, title=None):
    # X is 1D, Y is 2D
    Y_pred = vmap(model, in_axes=(None, 0, None))(params, X, rng)
    plt.figure(figsize=(5, 5))
    datacolor = 'tab:blue'
    plt.scatter(X[:, 0], Y[:, 2], s=20, alpha=0.25, label='data', color=datacolor, marker='o')
    # plt.scatter(X[:, 0], Y[:, 1], s=20, alpha=0.25, label='data_2', color=datacolor, marker='x')
    predcolor = 'tab:orange'
    plt.scatter(
        X[:, 1], Y_pred[:, 1], s=20, alpha=0.25, label='predictions', color=predcolor, marker='o'
    )
    # plt.scatter(X[:, 0], Y_pred[:, 1], s=20, alpha=0.25, label='predictions_2', color=predcolor, marker='x')
    plt.legend()
    if title is not None:
        plt.title(title)

def plot_all_heatmaps(X, Y, title=None, orient='v', outputnames=None):
    n_axes = Y.shape[1]
    assert(X.shape[1] == 2)
    if orient=='v':
        fig, axes = plt.subplots(n_axes, 1, figsize=(5, 5 * n_axes))
    else:
        fig, axes = plt.subplots(1, n_axes, figsize=(5 * n_axes, 5))
    if outputnames is None:
        outputnames = [f'output {i}' for i in range(n_axes)]
    for i in range(n_axes):
        axes[i].scatter(X[:, 0], X[:, 1], c=Y[:, i], s=1, alpha=1, cmap='YlGnBu_r')
        axes[i].set_xlabel('input 1')
        axes[i].set_ylabel('input 2')
        axes[i].set_title(outputnames[i])
        axes[i].set_xlim(0, 10)
        axes[i].set_ylim(0, 10)
    # add colorbar:
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05/n_axes, 0.7])
    fig.colorbar(axes[0].collections[0], cax=cbar_ax)
    if title is not None:
        fig.suptitle(title)

# plot_all_heatmaps(X[0], Y[0], 'data', orient='h', outputnames=m.get_output_proteins())
Y_pred = vmap(m, in_axes=(None, 0, None))(best_params, X[0], rng)
# plot_all_heatmaps(X[0], Y_pred, 'predictions', orient='h', outputnames=m.get_output_proteins())
error = jnp.abs(Y_pred - Y[0])
plot_all_heatmaps(X[0], error, 'error', orient='h', outputnames=m.get_output_proteins())

du.model_heatmap(models[1], Y[1])


plot_predictions_2d(m, best_params, X[0], Y[0])
plot_predictions_2d(minv, best_params, X[1], Y[1], 'inverse predictions (should be y=x)')

r, d = vmap(m.collect_all_results, in_axes=(None, 0, None))(best_params, X[0], rng)
ngout = r[:, 0]

# scatter case vs rna (X[0][:,0] vs X[0][:,1]), color by 'ngout'
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sc = ax.scatter(X[0][:, 0], X[0][:, 1], c=ngout, s=1, alpha=0.5)
ax.set_xlabel('CasE dna qtty')
ax.set_ylabel('Rec dna qtty')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title('predicted neongreen for fwd net')
fig.colorbar(sc, ax=ax)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sc = ax.scatter(X[0][:, 0], X[0][:, 1], c=Y[0][:, 0], s=1, alpha=0.5)
ax.set_xlabel('CasE dna qtty')
ax.set_ylabel('Rec dna qtty')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title('true neongreen for fwd net')
fig.colorbar(sc, ax=ax)

_, allres = m.collect_all_results(best_params, jnp.array([10.0, 5.0]), rng)

ut.plot_networks([m.network], outputs=[allres], H=2000)

dna_to_rna(jnp.array([5.0]))
rna_to_protein(dna_to_rna(jnp.array([5.0])))


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                    --    archive [manual training]     --
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
