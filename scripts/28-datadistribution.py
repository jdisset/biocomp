## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     imports     -
# ···············································································
import biocomp as bc
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import biocomp.utils as bu
import pandas as pd
from matplotlib.colors import LogNorm
import scriptutils as ut
import jax
from jax import jit, vmap, grad, value_and_grad
import jax.numpy as jnp
import biocomp.datautils as du
from scipy.spatial import KDTree
import optax
from tqdm import tqdm
import biocomp.nodes as bn
from jax.scipy.stats import gaussian_kde
import biocomp.compute as bcc


import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10.0, 10.0]
plt.rcParams['figure.dpi'] = 200


def mkfig(rows, cols, size=(6, 6)):
    fig, ax = plt.subplots(rows, cols, figsize=(cols * size[0], rows * size[1]))
    return fig, ax


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

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                           --     load     --
# ···············································································
lib = ut.load_lib()
xp = ut.load_xp('E20221124A_ERNbandpassV2', lib)
models = xp.get_models(node_impl=cfg['node_impl'])
X, Y = xp.get_XY(models)
mname = "103+103i+101R"
model = models[mname]
x, y = X[mname], Y[mname]
x = x / 1e6
y = y / 1e6

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                       --     plot networks     --
# ···············································································

# ut.plot_networks([m.network for m in models.values()])

mname = "103+103i+101R"
model = models[mname]
# ut.plot_networks([model.network], ['/Users/jeandisset/Desktop/model.pdf'])

x, y = X[mname], Y[mname]
x = x / 1e6
y = y / 1e6

fig, ax = du.model_heatmap(model, y)

##

out_proteins = model.get_output_proteins()
in_proteins = model.get_inverted_input_proteins()
z_prot = list(set(out_proteins) - set(in_proteins))[0]
z_prot

out_proteins
stats, bins = du.binstats(y, out_proteins, bin_proteins=out_proteins, resolution=0.5)
key = jax.random.PRNGKey(0)
y_bal = du.balance_per_bin(y, stats, key, threshold_min=10, threshold_quantile=0.3)
y_bal.shape
fig, ax = du.model_heatmap(model, y_bal)
y_bal.shape
x_bal = model.get_input_from_output(y_bal)

y_bal.shape


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

xdata, ydata = X[mname], Y[mname]
xdata = xdata / 1e6
ydata = ydata / 1e6

out_proteins = model.get_output_proteins()
in_proteins = model.get_inverted_input_proteins()

in_proteins


tree = KDTree(xdata)


def get_neighbors(coords, radius, tree=tree):
    indices = jnp.array(tree.query_ball_point(jnp.array(coords), radius))
    return indices


tagBFPbounds = (10, 50)
x_inbounds_unbal = xdata[xdata[:, 0] > tagBFPbounds[0]]
x_inbounds_unbal = x_inbounds_unbal[x_inbounds_unbal[:, 0] < tagBFPbounds[1]][:, 1:]
kde_unbal = gaussian_kde(x_inbounds_unbal.T)

x_inbounds_bal = x_bal[x_bal[:, 0] > tagBFPbounds[0]]
x_inbounds_bal = x_inbounds_bal[x_inbounds_bal[:, 0] < tagBFPbounds[1]][:, 1:]
kde_bal = gaussian_kde(x_inbounds_bal.T, bw_method=0.01)

# grid point generation
vmin, vmax = 1e-4, ydata.max()
log_resolution = 0.025
log_bins = np.array(du.mk_log_grid(vmin, vmax, log_resolution)).squeeze()
log_midpoints = (log_bins[1:] + log_bins[:-1]) / 2
log_nbpoints = len(log_midpoints)
log_xy = jnp.vstack(map(jnp.ravel, jnp.meshgrid(log_midpoints, log_midpoints))).T
lin_nbpoints = 100
lin_midpoints = jnp.linspace(vmin, vmax, lin_nbpoints)
lin_xy = jnp.vstack(map(jnp.ravel, jnp.meshgrid(lin_midpoints, lin_midpoints))).T


## rebalance using kde

##

kde_eval_unbal = kde_unbal.evaluate(log_xy.T).reshape(log_nbpoints, log_nbpoints)
# kde_eval_unbal = kde_eval_unbal / kde_eval_unbal.max()

def logplot(z, title='', zmin=None, zmax=None):
    fig, ax = plt.subplots()
    if zmin is None:
        zmin = 1e-5
    if zmax is None:
        zmax = z.max()
    cmap = plt.cm.get_cmap('YlGnBu')
    cmap.set_bad(color='#CCCCCC')
    # set 0 to nan
    z = jnp.where(z == 0, jnp.nan, z)
    im = ax.pcolormesh(
        log_midpoints,
        log_midpoints,
        z,
        cmap=cmap,
        norm=LogNorm(vmin=zmin, vmax=zmax),
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('density')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_xlabel(f'{in_proteins[1]}')
    ax.set_ylabel(f'{in_proteins[2]}')
    ax.set_title(title)
    return fig, ax

logplot(
    kde_eval_unbal,
    f'kde for UNBALANCED {in_proteins[1]} and {in_proteins[2]} at tagBFP bounds {tagBFPbounds}',
)

kde_eval_bal = kde_bal.evaluate(log_xy.T).reshape(log_nbpoints, log_nbpoints)
# kde_eval_bal = kde_eval_bal / kde_eval_bal.max()
logplot(
    kde_eval_bal,
    f'kde for BALANCED {in_proteins[1]} and {in_proteins[2]} at tagBFP bounds {tagBFPbounds}',
)


##
radii = (0.2 / jnp.maximum(kde_eval_unbal, 1e-5)) ** 0.5
tree_in_bounds = KDTree(x_inbounds_unbal)
neighborhoods = [
    get_neighbors(coords, r, tree=tree_in_bounds)
    for coords, r in tqdm(list(zip(log_xy, radii.ravel())))
]
##
neigh_means = jnp.array([y[neigh][:, 0].mean() for neigh in tqdm(neighborhoods)])
neigh_means = neigh_means.reshape(len(log_midpoints), len(log_midpoints))
fig, ax = mkfig(1, 1)
# if nan, just display grey
cmap = plt.cm.get_cmap('YlGnBu')
cmap.set_bad(color='#CCCCCC')
vmax = jnp.nanmax(neigh_means)
im = ax.pcolormesh(
    log_midpoints,
    log_midpoints,
    neigh_means,
    cmap='YlGnBu',
    norm=LogNorm(vmin=1e-1, vmax=vmax),
)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('mean yfp')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(vmin, vmax)
ax.set_ylim(vmin, vmax)
ax.set_xlabel(f'{in_proteins[1]}')
ax.set_ylabel(f'{in_proteins[2]}')
ax.set_title(f'mean yfp for {in_proteins[1]} and {in_proteins[2]} at tagBFP bounds {tagBFPbounds}')



## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                            --     old     --
#···············································································

##
# B, R, K

coords = (3, 3, 4)

# coords = (7,7,4)

bounds = [(b[c], b[1 + c]) for b, c in zip(bins.values(), coords)]
bounds
ids = stats.loc[coords, ('indices', '')]
yfp = y[ids][:, 0]
yfp_sorted = jnp.sort(yfp)
yfp.mean()

yfp.shape

out_proteins
yfp.shape

plt.plot(yfp_sorted)
plt.show()

##
kde = gaussian_kde(yfp, bw_method=0.1)

log_resolution = 200
dmin, dmax = 0.1, 40

# histogram the data
# log bins
lbins = np.logspace(np.log10(dmin), np.log10(dmax), log_resolution)
counts, bin_edges = np.histogram(yfp, bins=lbins)
bin_edges
counts
# plot:
fig, ax = plt.subplots()
x = jnp.linspace(0, yfp.max(), log_resolution)
widths = jnp.log(bin_edges[1:]) - jnp.log(bin_edges[:-1])
ax.bar(bin_edges[:-1], counts, 0.2 * widths, color='b')
ax.set_xscale('log')
# kde_eval = kde.evaluate(x)

# # ax.plot(x, kde_eval)
# # log x axis
# ax.set_xscale('log')
# ax.set_xlim(dmin, dmax)
# ax.set_title(f'kde density for yfp at BRK bounds {[(int(b[0]), int(b[1])) for b in  bounds]}')
# plt.show()


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


