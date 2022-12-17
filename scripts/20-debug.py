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
# {{{                          --     config     --
# ···············································································

WSIZE = 32
DEPTH = 2
I_SIZE = 32
I_DEPTH = 2
I_OUT = 8
ERN_SIZE = 64
ERN_DEPTH = 2
node_impl = dict(
    bc.nodes.DEFAULT_COMPUTE_NODES_DICT,
    **{
        'output': partial(bc.nn.output, wsize=128, depth=3),
        'transcription': partial(
            bc.nn.transcription,
            outer_wsize=WSIZE,
            outer_depth=DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
            inner_out=I_OUT,
        ),
        'translation': partial(
            bc.nn.translation,
            outer_wsize=WSIZE,
            outer_depth=DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
            inner_out=I_OUT,
        ),
        'inv_transcription': partial(bc.nn.inv_transcription, outer_wsize=WSIZE, outer_depth=DEPTH),
        'inv_translation': partial(bc.nn.inv_translation, outer_wsize=WSIZE, outer_depth=DEPTH),
        'sequestron_ERN': partial(bc.nn.sequestron_ERN, wsize=ERN_SIZE, depth=ERN_DEPTH),
    },
)
cfg = {
    "optimizer": "adam",
    "learning_rate": 0.0001,
    "adam_w_decay": 0.0001,
    "rng_key": np.random.randint(0, 2**32),
    # "rng_key": 11325,
    "epochs": 10,
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
print(rng)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                           --     utils     --
#···············································································

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
# {{{            --     some base functions for generation     --
# ···············································································
def hill_eq(x, k, n):
    return x**n / (k**n + x**n)

@jit
@vmap
def dna_to_rna(x):
    return hill_eq(x, 3.5, 1.2) * 12

@jit
@vmap
def rna_to_protein(x):
    return hill_eq(x, 5, 1.5) * 12

@jit
@vmap
def protein_to_fluo(x):
    return hill_eq(x, 1.75, 2) * 12

@jit
@vmap
def simple_ern(neg, pos):
    return jax.nn.relu(pos - neg)

# plot rna and protein production
x = np.linspace(0, 10, 1000)
plt.plot(x, dna_to_rna(x))
plt.plot(x, rna_to_protein(x))
plt.legend(['dna -> rna', 'rna -> protein'])

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                --     synthetic network --
# ···············································································

def P(name):
    return bc.Slot(lib, name)
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
nsamples = 10000
dna = jax.random.uniform(rng, (nsamples, 2), minval=0, maxval=8)
rna = dna_to_rna(dna)
prt = rna_to_protein(rna)
out = {
    "NeonGreen": rna_to_protein(simple_ern(prt[:, 0], rna[:, 1])),
    "mKate": prt[:, 0],
    "eBFP": prt[:, 1],
}


def plot_samples(dna, rna, out):
    fig, axs = plt.subplots(4, 1, figsize=(5, 20))
    axs[0].set_xlabel('dna')
    axs[0].set_ylabel('rna')
    axs[0].scatter(dna[:, 0], rna[:, 0], s=1, alpha=0.5, label='rna_1')
    axs[0].scatter(dna[:, 1], rna[:, 1], s=1, alpha=0.5, label='rna_2')

    axs[1].set_xlabel('dna')
    axs[1].set_ylabel('protein')
    axs[1].scatter(dna[:, 0], out['mKate'], s=1, alpha=0.5, label='mKate')
    axs[1].scatter(dna[:, 1], out['eBFP'], s=1, alpha=0.5, label='eBFP')

    axs[2].scatter(
        out['mKate'], rna[:, 1], c=out['NeonGreen'], s=1, alpha=1, label='NeonGreen', cmap='YlGnBu_r'
    )
    axs[2].set_xlabel('ern')
    axs[2].set_ylabel('rna')
    axs[2].set_title('remaining rna after ern sequestration')

    axs[3].scatter(
        out['mKate'], out['eBFP'], c=out['NeonGreen'], s=1, alpha=1, label='NeonGreen', cmap='YlGnBu_r'
    )
    axs[3].set_xlabel('mKate')
    axs[3].set_ylabel('eBFP')
    axs[3].set_title('output from input (fluo)')

plot_samples(dna, rna, out)

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

cfg['epochs'] = 120
train_history = bc.train.train_models(models, x_batches, y_batches, cfg, loggers)


fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(train_history['loss'])
# log scale for y
ax.set_yscale('log')
ax.set_xlabel('epoch')
ax.set_ylabel('loss')


import json


def pytree_to_np(t):
    def conv(x):
        if isinstance(x, jnp.ndarray):
            return np.array(x)
        else:
            return x
    return jax.tree_map(conv, t)


du.save(train_history, 'train_history.pkl')
th = du.load('train_history.pkl')

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

extra = m.network.compute_graph[m.network.compute_graph.type == 'sequestron_ERN'].extra.to_list()
[ut.plot_node('sequestron_ERN', best_params, m, xlim=(-10, 100), n_inputs=2, mode='3d', extra_args=ex) for ex in extra]
# ut.plot_node('sequestron_ERN', best_params, m, n_inputs=2, mode='3d')

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



r, d = vmap(m.collect_all_results, in_axes=(None, 0, None))(best_params, X[0], rng)
ngout = r[:, 0]
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
plot_predictions_2d(minv, best_params, X[1], Y[1], 'inverse predictions (should be y=x)')



#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                           --     ideas     --
#···············································································
# TODO
# Learn stuff progressively using decaying weights in the loss function:
# learn a reasonable tc and tl (some kind of hill shaped function)
# learn a relu-like ERN 
# learn the forward synthetic network
# learn the inverse synthetic network
# maybe aven assign weights to the different networks (could be an easy way to add them progressively)
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
