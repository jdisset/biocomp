## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     imports     --
# ···············································································
import biocomp as bc
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import biocomp.utils as bu
import scriptutils as ut
import jax
from jax import jit, vmap, grad, value_and_grad
import jax.numpy as jnp
import biocomp.datautils as du
import optax
from tqdm import tqdm
import biocomp.nodes as bn
import biocomp.compute as bcc


import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10.0, 10.0]
plt.rcParams['figure.dpi'] = 200

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     config     --
# ···············································································

T_SIZE = 64
T_DEPTH = 3
I_SIZE = 64
I_DEPTH = 2
I_OUT = 8
ERN_SIZE = 128
ERN_DEPTH = 3
MEFL_SIZE = 64
MEFL_DEPTH = 3
node_impl = dict(
    bc.nodes.DEFAULT_COMPUTE_NODES_DICT,
    **{
        'output': partial(bc.nn.output, wsize=MEFL_SIZE, depth=MEFL_DEPTH),
        'transcription': partial(
            bc.nn.transcription,
            outer_wsize=T_SIZE,
            outer_depth=T_DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
            inner_out=I_OUT,
        ),
        'translation': partial(
            bc.nn.translation,
            outer_wsize=T_SIZE,
            outer_depth=T_DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
            inner_out=I_OUT,
        ),
        'inv_transcription': partial(
            bc.nn.inv_transcription,
            outer_wsize=T_SIZE,
            outer_depth=T_DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
            inner_out=I_OUT,
        ),
        'inv_translation': partial(
            bc.nn.inv_translation,
            outer_wsize=T_SIZE,
            outer_depth=T_DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
            inner_out=I_OUT,
        ),
        'sequestron_ERN': partial(bc.nn.ERN5p, wsize=ERN_SIZE, depth=ERN_DEPTH),
        'sequestron_ERN3p': partial(bc.nn.ERN3p, wsize=ERN_SIZE, depth=ERN_DEPTH),
    },
)
cfg = {
    "optimizer": "adam",
    "learning_rate": 0.0001,
    "adam_w_decay": 0.0001,
    "rng_key": np.random.randint(0, 2**32),
    # "rng_key": 11325,
    "epochs": 200,
    "compile_training": True,
    "batch_size": 8,
    "norm_factor": 1e7,
    "balance_bin_resolution": 0.5,
    "balance_threshold_quantile": 0.4,
    "balance_threshold_min": 40,
    "node_impl": node_impl,
    # "nmodels": 1,
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
all_models = xp.get_models(node_impl=cfg['node_impl'])


def pick_n_random_models(n, models):
    # select n models from models (which is a dict of modelname -> model)
    keys = list(models.keys())
    np.random.shuffle(keys)
    return {k: models[k] for k in keys[:n]}


models = pick_n_random_models(cfg['nmodels'], models)

print(f'Picked models: {list(models.keys())}')


X, Y = bc.train.preprocess_data(models, xp.get_Y(models), cfg)
batch_size = cfg['batch_size']
x_batches, y_batches = du.make_batches_uniform_sampling(
    Y.values(), batch_size, rng, models.values()
)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --      load run saves     --
# ···············································································
from pathlib import Path

run_dir = '../__out/fromrachael/atomic-planet-9'
# all epoch files are saved as run_dir/epoch_XX.pkl
epoch_filenames = sorted(Path(run_dir).glob('epoch_*.pkl'))
epoch_nums = [int(f.stem.split('_')[1]) for f in epoch_filenames]
epoch_data = {
    i: du.load(epoch_filename) for i, epoch_filename in tqdm(zip(epoch_nums, epoch_filenames))
}

# epoch_data[1].keys() is dict_keys(['grad', 'loss', 'magnitude', 'opt', 'params'])
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     loss plot     --
# ···············································································
d = epoch_data[1]
plt.plot(d['loss'])

loss_stats = {
    i: {
        'median': jnp.median(d['loss'], axis=0),
        '90%': jnp.percentile(d['loss'], jnp.array([5, 95]), axis=0),
        'min': jnp.min(d['loss'], axis=0),
        'max': jnp.max(d['loss'], axis=0),
    }
    for i, d in epoch_data.items()
}


def plot_loss_stats(loss_stats):
    fig, ax = plt.subplots()
    # light blue
    color = '#a6cee3'
    medians = [d['median'] for d in loss_stats.values()]
    mins = [d['min'] for d in loss_stats.values()]
    maxs = [d['max'] for d in loss_stats.values()]
    p5s = [d['90%'][0] for d in loss_stats.values()]
    p95s = [d['90%'][1] for d in loss_stats.values()]
    ax.plot(medians, color=color)
    ax.fill_between(range(len(medians)), mins, maxs, alpha=0.3, color=color)
    ax.fill_between(range(len(medians)), p5s, p95s, alpha=0.3, color=color)
    return fig, ax


fig, ax = plot_loss_stats(loss_stats)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     grad plot     --
# ···············································································

d = epoch_data[1]
g = d['grad']['shared']
p = d['params']['shared']


k = list(g.keys())[12]
v = g[k]


def get_epoch_stats(epoch_data, smooth_win=1):
    def comp(v):
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
    g = d['grad']['shared']
    p = d['params']['shared']
    stats = {'grad':{}, 'params':{}}
    for k, v in g.items():
        stats['grad'][k] = comp(v)
    for k, v in p.items():
        stats['params'][k] = comp(v)
    return stats

stats = get_epoch_stats(v)

def plot_epoch(v, title, smooth_win=1):
    fig, ax = plt.subplots()
    # light blue
    colors = ['#d5573e', '#df743c', '#e6913f', '#e9ae48']
    alpha = 0.5
    ax.fill_between(range(len(medians)), mins, maxs, alpha=alpha, color=colors[3], label='min/max')
    ax.fill_between(range(len(medians)), p5s, p95s, alpha=alpha, color=colors[2], label='95%')
    ax.fill_between(range(len(medians)), p20s, p80s, alpha=alpha, color=colors[1], label='80%')
    ax.plot(medians, color=colors[0])
    # use sym log scale
    max = 100
    ax.set_ylim(-max, max)
    ax.set_yscale('symlog', linthresh=1e-3)
    # add title and legends and labels and everything. make it pretty
    ax.set_title(title)
    ax.set_xlabel('step')
    ax.set_ylabel('magnitude')
    ax.legend()
    return fig, ax


out_dir = Path(run_dir) / 'plots'
out_dir.mkdir(exist_ok=True)

for i, (k, v) in enumerate(g.items()):
    print(i, k)
    fig, ax = plot_epoch(v, f'gradient for {k}', smooth_win=10)
    fig.savefig(out_dir / f'grad_{i}_{k}.png')


for i, (k, v) in enumerate(p.items()):
    print(i, k)
    fig, ax = plot_epoch(v, f'params for {k}', smooth_win=10)
    fig.savefig(out_dir / f'params_{i}_{k}.png')

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{               --     training with only one sample     --
#···············································································

# models = pick_n_random_models(1, all_models)
# print(f'Picked models: {list(models.keys())}')

selected = ['104+102R+102I']
models={k: v for k, v in all_models.items() if k in selected}

ut.plot_networks([m.network for m in models.values()], H=2000)

X, Y = bc.train.preprocess_data(models, xp.get_Y(models), cfg)
# batch_size = cfg['batch_size']
batch_size = 12
x_batches, y_batches = du.make_batches_uniform_sampling(
    Y.values(), batch_size, rng, models.values()
)

##

x = x_batches
y = y_batches
m = list(models.values())[0]
du.model_parallel_coords(m, y.squeeze(), maxval=200, minval=1e-4, n_samples=200)

m.get_output_proteins()
m.get_inverted_input_proteins()
x[0]
y[0]

##

ydata = y_batches.reshape(-1, y_batches.shape[-1])
du.model_heatmap(m, ydata, outer_resolution=1)


##



def console_log(epoch, cfg, epoch_history=None, **_):
    if epoch_history is not None:
        loss = np.array(epoch_history['loss'])
        avg = np.mean(loss)
        std = np.std(loss)
        lmin, lmax = jnp.min(loss), jnp.max(loss)
        print(
            f'[{epoch}/{cfg["epochs"]}] loss: {avg:.3f} ± {std:.3f} [min {lmin:.3f}, max {lmax:.3f}]'
        )

loggers = [
        (1, console_log),
    ]

cfg['epochs'] = 1
# train_history = bc.train.train_models(models.values(), x, y, cfg, loggers)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

params = bu.get_pytree(train_history['params'], len(x_batches)-1)

p, c = m.init(jax.random.PRNGKey(0))
params['node'] = p['node']
m.network.get_compute_types()
ut.plot_node('translation', params, m)
ut.plot_node('transcription', params, m)

extra = m.network.compute_graph[m.network.compute_graph.type == 'sequestron_ERN'].extra.to_list()
[ut.plot_node('sequestron_ERN', params, m, xlim=(-100, 10), n_inputs=2, mode='3d', extra_args=ex) for ex in extra]


##
# let's plot y points in a 2d space using a PCA:
import sklearn.decomposition
# we only keep the last dimension (of size nfeatures) of y and flatten the rest
ydata = y_batches.reshape(-1, y_batches.shape[-1])
xdata = x_batches.reshape(-1, x_batches.shape[-1])
pca = sklearn.decomposition.PCA(n_components=2)
pca.fit(ydata)
y_pca = pca.transform(ydata)
fig, ax = plt.subplots()
ax.scatter(y_pca[:, 0], y_pca[:, 1], s=1, alpha=0.1)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('PCA of data points')

##
# now let's plot the same points in the same space but using the model's output
ypred = vmap(m, in_axes=(None, 0, None))(params, xdata, jax.random.PRNGKey(0))
ypred_pca = pca.transform(ypred)
fig, ax = plt.subplots()
ax.scatter(y_pca[:, 0], y_pca[:, 1], s=1, alpha=0.1, label='data')
ax.scatter(ypred_pca[:, 0], ypred_pca[:, 1], s=1, alpha=0.1, label='model')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('PCA of data points')
ax.legend()
##

l2error = np.linalg.norm(ydata - ypred, axis=1)
fig, ax = plt.subplots()

bins = np.logspace(0, 3, 100)
ax.hist(l2error, bins=bins, log=True, color='C0', alpha=1, label='L2 error')
ax.set_xscale('log')
ax.set_xlabel('L2 error')
ax.set_ylabel('count')
ax.set_title('L2 error distribution')



##
ydata = y_batches.reshape(-1, y_batches.shape[-1])
xdata = x_batches.reshape(-1, x_batches.shape[-1])
ypred = vmap(m, in_axes=(None, 0, None))(params, xdata, jax.random.PRNGKey(0))
du.model_heatmap(m, ypred, outer_resolution=1)


