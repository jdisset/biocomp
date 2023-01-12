### {{{                          --     imports     --
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
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10.0, 10.0]
plt.rcParams['figure.dpi'] = 200
##────────────────────────────────────────────────────────────────────────────}}}

### {{{                          --     config     --
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
    "rng_key": np.random.randint(0, 2**32),
    # "rng_key": 11325,
    "epochs": 1000,
    "compile_training": True,
    "batch_size": 8,
    "norm_factor": 1e6,
    "balance_bin_resolution": 0.5,
    "balance_threshold_quantile": 0.4,
    "balance_threshold_min": 40,
    "node_impl": node_impl,
    "nmodels": 28,
}
##────────────────────────────────────────────────────────────────────────────}}}

### {{{                         --     load data     --
lib = ut.load_lib()
# xp = ut.load_xp('E20221124A_ERNbandpassV2', lib)
xp = ut.load_xp('20220501-GW-l1vsl2', lib)
rng = jax.random.PRNGKey(cfg['rng_key'])
models = xp.get_models(node_impl=cfg['node_impl'])
X_raw, Y_raw = xp.get_XY(models)
##────────────────────────────────────────────────────────────────────────────}}}

### {{{                      --     display models     --

ut.plot_networks([m.network for m in models.values()])

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                    --     plotting functions     --
def mkfig(rows, cols, size=(6, 6)):
    fig, ax = plt.subplots(rows, cols, figsize=(cols * size[0], rows * size[1]))
    return fig, ax


def plot_over_1d(f, vmin=-2, vmax=2, resolution=200, title=None, ax=None):
    x = jnp.linspace(vmin, vmax, resolution)
    y = f(x)
    if ax is None:
        _, ax = mkfig(1, 1)
    ax.plot(x, y)
    if title is not None:
        ax.set_title(title)
    return ax


vmin = -2
vmax = 2
resolution = 100


def plot_over_2d(
    f,
    vmin=-2,
    vmax=2,
    resolution=200,
    title=None,
    ax=None,
    add_colorbar=True,
    value_range=None,
):
    x = jnp.linspace(vmin, vmax, resolution)
    xy = jnp.vstack(map(jnp.ravel, jnp.meshgrid(x, x))).T
    z = f(xy.T).reshape((resolution, resolution))
    if ax is None:
        _, ax = mkfig(1, 1)
    if value_range is None:
        value_range = (z.min(), z.max())
    im = ax.pcolormesh(x, x, z, vmin=value_range[0], vmax=value_range[1], cmap='inferno')
    if add_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        cax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.0e}".format(x)))
    ax.set_aspect('equal')
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    if title is not None:
        ax.set_title(title)
    return ax


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                 --     select a model to work on     --
# modelname = '(1:5; 100+73)+100i'
modelname = 'L2all_pGW42+10'
model = models[modelname]
X_r, Y_r = X_raw[modelname], Y_raw[modelname]

ut.plot_networks([model.network], ['/Users/jeandisset/Desktop/model_l2.pdf'])

##────────────────────────────────────────────────────────────────────────────}}}

# ─────────────────────────────────────────────────────────────────────────────
#                                  PREPROCESSING
# ───────────────────────────────────── ▼ ─────────────────────────────────────
### {{{            --     rescale and move to log space     --


def rescale(X, norm_factor):
    return np.log10(1.0 + (X / norm_factor))


cfg['norm_factor'] = 1e3

X = rescale(X_r, cfg['norm_factor'])
Y = rescale(Y_r, cfg['norm_factor'])

##────────────────────────────────────────────────────────────────────────────}}}

### {{{            --     estimate density of X using kde     --

x_prots = model.get_inverted_input_proteins()
y_prots = model.get_output_proteins()

from jax.scipy.stats import gaussian_kde

kde = gaussian_kde(X.T)

plot_over_2d(
    kde.evaluate,
    vmin=0,
    vmax=3,
    resolution=200,
    title=f'kde of X ({len(X)} points)',
    value_range=(0, 2.5),
)

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                         --     rebalance     --
densities = kde(X.T) + 1e-19
threshold = np.quantile(densities, 0.07)
diceroll = jax.random.uniform(rng, shape=(len(densities),))
selected = np.where((densities < threshold) | (diceroll < (threshold / (densities * 1.2))))[0]

Xrebalanced = X[selected]
Yrebalanced = Y[selected]
kde_balanced = gaussian_kde(Xrebalanced.T)
plot_over_2d(
    kde_balanced,
    vmin=0,
    vmax=3,
    resolution=200,
    title=f'kde of X after rebalancing ({len(Xrebalanced)} points)',
    value_range=(0, 0.25),
)

##────────────────────────────────────────────────────────────────────────────}}}

# ─────────────────────────────────────────────────────────────────────────────
#                                   MEFL-MEFL
# ───────────────────────────────────── ▼ ─────────────────────────────────────

### {{{            --    load the 1:1 coTx of fluo proteins     ##
from pathlib import Path
import pandas as pd
import itertools

fluoxp_folder = Path(ut.DEFAULT_XP_PATH / '2022-12-5_11,3a,30_ngDNA_ng')
color_channels = {
    "eBFP": "Pacific_Blue_A",
    "tagBFP": "Pacific_Blue_A",
    "eYFP": "FITC_A",
    "L0.G_mNeonGreen": "FITC_A",
    "mKate": "PE_Texas_Red_A",
    "iRFP720": "APC_Alexa_700_A",
}
dna_qtties = np.arange(1, 11) * 100
fluoxp_data = [
    fluoxp_folder / 'data' / f'{n}ngBYR.2022-12-5_11,3a,30_ngDNA_ng.csv' for n in dna_qtties
]

fluoxp_data = {n: pd.read_csv(f) for n, f in zip(dna_qtties, fluoxp_data)}

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                       --     scatter plot     --
dnaqtty = 500
fluodf = fluoxp_data[dnaqtty]
available_channels = fluodf.columns
# scatter plot of the data for every pair of channels
pairs = list(itertools.combinations(available_channels, 2))
npairs = len(pairs)
fig, axes = mkfig(1, npairs)
for i, (ax, (x, y)) in enumerate(zip(axes, pairs)):
    ax.scatter(fluodf[x], fluodf[y], s=1)
    ax.set_xlim(1e3, 1e10)
    ax.set_ylim(1e3, 1e10)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_aspect('equal')
fig.suptitle(f'{dnaqtty}ng DNA')
fig.tight_layout()


pairs = pairs + [(b, a) for a, b in pairs]
mefl_mefl = np.vstack([fluodf[list(p)].values for p in pairs]) / cfg['norm_factor']
# scatter:
fig, ax = mkfig(1, 1)
ax.scatter(mefl_mefl[:, 0], mefl_mefl[:, 1], s=0.1, alpha=0.05, edgecolors='none')
ax.set_xlim(1e3, 1e10)
ax.set_ylim(1e3, 1e10)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_aspect('equal')
fig.suptitle(f'all pairs together')
fig.tight_layout()

##────────────────────────────────────────────────────────────────────────────}}}

### {{{     --     model definition to fit normal distributions     --

H_SIZE = 32
N_LAYERS = 4


def get_param(name, init, params, shared=False):
    if name not in params:
        print(f'initializing {name}')
        params[name] = init()
    return params[name]


def dense_layer(input_values, output_size, get_param, key, name):
    input_size = 1 if input_values.shape == () else input_values.shape[0]
    w = get_param(f'{name}_w', init=bu.he_initializer(key, (input_size, output_size)), shared=True)
    b = get_param(f'{name}_b', init=lambda: jnp.zeros((output_size,)), shared=True)
    try:
        return jnp.dot(input_values, w) + b
    except ValueError as e:
        msg = f'input_values.shape: {input_values.shape}\n'
        msg += f'w.shape: {w.shape}\n'
        msg += f'b.shape: {b.shape}\n'
        raise ValueError(msg) from e


activation = jax.nn.leaky_relu


def fmodel(params, x, key):
    k0, k1, k2 = jax.random.split(key, 3)
    get_p = partial(get_param, params=params)
    x = dense_layer(x, H_SIZE, get_p, k0, 'm_dense0')
    x = activation(x)
    for i in range(N_LAYERS - 1):
        x = dense_layer(x, H_SIZE, get_p, k1, f'm_dense{i+1}')
        x = activation(x)
    return dense_layer(x, 2, get_p, k2, 'm_dense_out').squeeze()


vm = vmap(fmodel, in_axes=(None, 0, None))

##                                                                            }}}

### {{{                            --     fit     --

from jax.scipy.stats import norm

rng_key = jax.random.PRNGKey(42)
choice = jax.random.choice(rng_key, mefl_mefl.shape[0], shape=(20000,))
XY = mefl_mefl[choice, :]
XY = jnp.log10(1 + XY)


def loss_fn(params, x, y, key):
    out = vm(params, x, key)
    mu, sigma = out[:, 0], out[:, 1]
    return -jnp.mean(norm.logpdf(y, loc=mu, scale=sigma))


params = {}
key = rng_key
model(params, jnp.zeros(1), rng_key)
opt = optax.adam(learning_rate=1e-5)
opt_state = opt.init(params)


@jit
def update(params, opt_state, x, y, key):
    loss, grads = value_and_grad(loss_fn)(params, x, y, key)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


n_epochs = 50
batch_size = 100


losses = []
for epoch in range(n_epochs):
    for i in range(0, XY.shape[0], batch_size):
        x = XY[i : i + batch_size, 0]
        y = XY[i : i + batch_size, 1]
        params, opt_state, loss = update(params, opt_state, x, y, key)
        losses.append(loss)
    print(f'epoch {epoch}, loss {loss}')

# plot loss
fig, ax = mkfig(1, 1)
ax.plot(losses)
ax.set_xlabel('batch')
ax.set_ylabel('loss')
# ax.set_yscale('log')
##────────────────────────────────────────────────────────────────────────────}}}

### {{{                     --     plot mu and sigma     --

x = jnp.linspace(0, 7, 100)
out = vm(params, x, rng_key)
fig, axes = mkfig(1, 1)
axes.plot(x, out[:, 0], label='mu')
axes.plot(x, out[:, 1], label='sigma')
axes.legend()

##────────────────────────────────────────────────────────────────────────────}}}

### {{{             --     scatter plot of XY + mu and sigma     --
x = jnp.linspace(0, 3, 100)
out = vm(params, x, rng_key)
mu, sigma = out[:, 0], out[:, 1]
# sigma_original = sigma * (10**mu) * np.sqrt(10**(2 * sigma**2) - 1)
fig, ax = mkfig(1, 1)

ax.plot(x, mu, label='mu', color='red', alpha=0.5)
ax.fill_between(x, mu - sigma, mu + sigma, color='red', alpha=0.1, label='sigma')
ax.scatter(mefl_mefl[:, 0], mefl_mefl[:, 1], s=2, alpha=0.05, edgecolors='none')
# limits:
ax.legend()

##────────────────────────────────────────────────────────────────────────────}}}

# ─────────────────────────────────────────────────────────────────────────────
#                             LEARNING THE ERN XP
# ───────────────────────────────────── ▼ ─────────────────────────────────────

# We probably don't need the neighborhood + weighted quantile thingy. I think just using
# the quantile loss (and making the quantile a parameter of the network that we can then
# sample uniformly) should be enough and should be more flexible than the neighborhood approach

# Let's start by building a simple NN that takes X and Z and outputs Y
### {{{                           --     model     --
H_SIZE = 512
N_LAYERS = 6
activation = jax.nn.leaky_relu


def ERN_model(params, x, z, key):
    k0, k1, k2 = jax.random.split(key, 3)
    get_p = partial(get_param, params=params)
    x = jnp.concatenate([x.flatten(), z])
    x = dense_layer(x, H_SIZE, get_p, k0, 'm_dense0')
    x = activation(x)
    for i in range(N_LAYERS - 1):
        x = dense_layer(x, H_SIZE, get_p, k1, f'm_dense{i+1}')
        x = activation(x)
        x = jnp.concatenate([x.flatten(), z])
    return dense_layer(x, 3, get_p, k2, 'm_dense_out').squeeze()


vern = vmap(ERN_model, in_axes=(None, 0, 0, None))  # JULES!!!!

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                            --     fit     --

rng_key = jax.random.PRNGKey(42)
zsize = Yrebalanced.shape[1]


# y_true = YY[:10]
# x = XX[:10]


def quantile_loss(e, q):
    return jnp.maximum(q * e, (q - 1.0) * e)


def loss_fn(params, x, y_true, key):
    z = jax.random.uniform(key, shape=(y_true.shape[0], zsize))
    y_pred = vern(params, x, z, key)
    e = y_true - y_pred
    vq = vmap(quantile_loss, in_axes=(1, 1))
    err = vq(e, z).T
    return jnp.mean(err)


@jit
def update(params, opt_state, x, y, key):
    loss, grads = value_and_grad(loss_fn)(params, x, y, key)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


params = {}
key = rng_key
ERN_model(params, jnp.zeros(Xrebalanced.shape[1]), jnp.zeros(Yrebalanced.shape[1]), rng_key)
opt = optax.adam(learning_rate=1e-6)
opt_state = opt.init(params)

n_epochs = 6000
batch_size = 300

losses = []
key = rng_key
for epoch in range(n_epochs):
    key, subkey = jax.random.split(key)
    reorder = jax.random.permutation(key, Xrebalanced.shape[0])
    XX = Xrebalanced[reorder]
    YY = Yrebalanced[reorder]
    batch_keys = jax.random.split(subkey, XX.shape[0] // batch_size)
    for i in range(0, XX.shape[0], batch_size):
        x = XX[i : i + batch_size]
        y = YY[i : i + batch_size]
        params, opt_state, loss = update(params, opt_state, x, y, batch_keys[i])
        losses.append(loss)
    print(f'epoch {epoch}, loss {loss}')

# plot loss
fig, ax = mkfig(1, 1)
ax.plot(losses)
ax.set_xlabel('batch')
ax.set_ylabel('loss')
ax.set_yscale('log')

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                         --     evaluate     --

# evaluate on a grid
res = 150
x = jnp.linspace(0, 3, res)
xygrid = jnp.array(np.meshgrid(x, x)).T.reshape(-1, 2)
n_z_per_x = 100
xx = np.tile(xygrid, (n_z_per_x, 1, 1))
z = jax.random.uniform(rng_key, shape=(xx.shape[0], xx.shape[1], zsize))
vvern = jit(vmap(vern, in_axes=(None, 0, 0, None)))
out = vvern(params, xx, z, rng_key)

##
# plot all the out poins, per xx[0, :, 0]
fig, ax = mkfig(1, 1)
for i in range(n_z_per_x):
    ax.scatter(xx[0, :, 1], out[i, :, 0], s=10, alpha=0.01, edgecolors='none', color='red')
ax.set_xlabel('input (eBFP)')
ax.set_ylabel('output (eBFP)')

##
out_mean = out.mean(axis=0)
# plot the mean value
fig, ax = mkfig(1, 1)
# use pcolormesh to get a nice color map
ax.set_aspect('equal')
im = ax.pcolormesh(
    xx[0, :, 0].reshape(res, res),
    xx[0, :, 1].reshape(res, res),
    out_mean[:, 1].reshape(res, res),
    cmap='YlGnBu',
)
fig.colorbar(im, ax=ax)
ax.set_xlabel('mKate')
ax.set_ylabel('eBFP')
fig.suptitle(f'mean YFP output ({n_z_per_x} samples per input)')
fig.tight_layout()


##────────────────────────────────────────────────────────────────────────────}}}


### {{{             --     plot real data using neighborhood     --
from scipy.spatial import cKDTree

tree = cKDTree(X)


x = xygrid
y = Y[:, output_id['eYFP']]


def get_knn_mean(x, y, knn=100, min_points=20):
    distances, indices = tree.query(x, k=knn, distance_upper_bound=0.2)
    mask = distances == np.inf
    nb_points = (~mask).sum(axis=1)
    nb_points
    # plt.plot(np.sort(nb_points))
    # weights = 1 / distances
    gausspdf = (
        lambda x, mu, sigma: 1
        / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    )
    weights = gausspdf(distances, 0, 0.1)
    indices[mask] = 0
    weights[mask] = 0
    weights[nb_points < min_points, :] = np.nan
    avg = np.average(y[indices], axis=1, weights=weights)
    return avg


output_names = model.get_output_proteins()
input_names = model.get_inverted_input_proteins()
output_id = {name: i for i, name in enumerate(output_names)}
input_id = {name: i for i, name in enumerate(input_names)}

res = 150
x = jnp.linspace(0, 3, res)
xygrid = jnp.array(np.meshgrid(x, x)).T.reshape(-1, 2)
fig, ax = mkfig(1, 1)
ax.set_aspect('equal')

knn=100

# cmap with grey when nan
cmap = plt.get_cmap('YlGnBu')
cmap.set_bad(color='#EEEEEE')
avg = get_knn_mean(xygrid, Y[:, output_id['eYFP']], knn)
im = ax.pcolormesh(
    xygrid[:, input_id['mKate']].reshape(res, res),
    xygrid[:, input_id['eBFP']].reshape(res, res),
    avg.reshape(res, res),
    cmap=cmap,
)
# add contour
ax.contour(
    xygrid[:, input_id['mKate']].reshape(res, res),
    xygrid[:, input_id['eBFP']].reshape(res, res),
    avg.reshape(res, res),
    levels=4,
    colors='black',
    linewidths=0.25,
)
ax.set_xlabel('mKate')
ax.set_ylabel('eBFP')
loglabels = 10**x - 1
tickfreq = res // 5
ax.set_xticks(x[::tickfreq])
ax.set_xticklabels([f'{l:.0e}' for l in loglabels[::tickfreq]])
ax.set_yticks(x[::tickfreq])
ax.set_yticklabels([f'{l:.0e}' for l in loglabels[::tickfreq]])
fig.colorbar(im, ax=ax, shrink=0.5)
# remove border
for spine in ax.spines.values():
    spine.set_visible(False)


fig.tight_layout()
fig.suptitle(f'Original data\nmean YFP output\n(20<k<{knn} n neighbors average)')


##────────────────────────────────────────────────────────────────────────────}}}
