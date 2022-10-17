## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                 --     import and init     --
# ···············································································
from jax.tree_util import Partial as partial
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, value_and_grad
import biocomp as bc
import biocomp.utils as bu
import scriptutils as ut
import datautils as du
from pathlib import Path
import json5
import json
import sqlite3
from tqdm import tqdm
import pandas as pd
import numpy as np

lib = ut.getLibFromGoogleSheet()

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

base_path = Path("/Users/jeandisset/Dropbox (MIT)/Biocomp/")
xp_path = base_path / "Experiments"
recipe_path = base_path / "Recipes"
experiments = [x.name for x in xp_path.iterdir() if x.is_dir()]

xp = bc.XP(experiments[0], xp_path, recipe_path, lib)
models = xp.get_models(inverse=True)
X, Y = xp.get_XY(models)


cfg = {
    "learning_rate": 0.001,
    "adam_w_decay": 0.01,
    "clipping": 0.001,
    "n_replicates": 10,
    "initial_param_scaling": 0.01,
    "normalize_data": False,
    "epochs": 100,
    "log_rate": 50,
    "rng_key": 1,
}

sample = 'CoTX-All'
model = models[sample]
x, y = X[sample], Y[sample]

# params_hist, loss_hist = bc.train_single_model(model, x, y, cfg)


data = np.array(y)


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                       --     heatmap draft     --
# ···············································································

bin_columns = [0, 2]
stat_column = 1
nbins = 20
log = True

vmin, vmax = data[:, bin_columns].min(axis=0), data[:, bin_columns].max(axis=0)
if log:
    bins = np.geomspace(vmin, vmax, nbins)
else:
    bins = np.linspace(vmin, vmax, nbins)
coords = np.array([np.digitize(data[:, i], bins[:, b]) for b, i in enumerate(bin_columns)]).T
df = pd.DataFrame(data)
df['coords'] = [tuple(x) for x in coords]
df = df.groupby('coords').agg({stat_column: ['mean', 'std', 'count']})
df.columns = ['mean', 'std', 'count']
bins

# plot all losses. They are stored in a list where each element
# is a point in history for each replicate


def heatmap(
    fulldata,
    xyc_axis,
    axis_names,
    title='',
    nbins=10,
    truncate_at_quantile=0.99,
    figsize=(7, 15),
    logscale=True,
    show_counts=True,
):

    count_threshold = 1

    xax, yax, cax = xyc_axis

    upperquantiles = np.quantile(fulldata, truncate_at_quantile, axis=0)
    data = fulldata[np.less(fulldata, upperquantiles).all(axis=1)]
    lowerquantiles = np.quantile(fulldata, 1 - truncate_at_quantile, axis=0)
    data = data[np.greater(data, lowerquantiles).all(axis=1)]

    # 2 plots side by side
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if logscale:
        bins_x = np.geomspace(1, np.max(data[:, xax]), nbins)
        bins_y = np.geomspace(1, np.max(data[:, yax]), nbins)
    else:
        bins_x = np.linspace(0, np.max(data[:, xax]), nbins)
        bins_y = np.linspace(0, np.max(data[:, yax]), nbins)

    coords = np.digitize(data[:, 0], bins_x, right=True), np.digitize(
        data[:, 2], bins_y, right=True
    )
    coords = np.array(coords) - 1

    means = np.zeros((len(bins_x), len(bins_y)))
    counts = np.zeros((len(bins_x), len(bins_y)))
    for i in range(len(bins_x)):
        for j in range(len(bins_y)):
            idx = (coords[0] == i) & (coords[1] == j)
            means[i, j] = np.mean(data[idx, 1])
            counts[i, j] = np.sum(idx)
            if counts[i, j] <= count_threshold:
                means[i, j] = np.nan

    # plot means
    ax = axes[0]
    # ax.imshow(means, origin='lower', aspect='auto', cmap='viridis')
    # we want a grid delineating the bins
    ax.imshow(
        means,
        origin='lower',
        aspect='auto',
        cmap='viridis',
    )
    ax.set_xlabel(axis_names[xax])
    ax.set_ylabel(axis_names[yax])
    ax.set_title('Mean ' + axis_names[cax])
    # label bins with their center value
    ax.set_xticks(np.arange(len(bins_x))[::4])
    ax.set_xticklabels([f'{x:.1e}' for x in bins_x[::4]])
    ax.set_yticks(np.arange(len(bins_y))[::2])
    ax.set_yticklabels([f'{x:.1e}' for x in bins_y[::2]])

    # write the count at the center of each bin
    if show_counts:
        for i in range(len(bins_x)):
            for j in range(len(bins_y)):
                ax.text(i, j, f'{counts[i, j]:.0f}', ha='center', va='center', color='black')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(ax.images[0], cax=cax)

    # plot counts
    ax = axes[1]
    ax.imshow(counts, origin='lower', aspect='auto', cmap='viridis')
    ax.set_xlabel(axis_names[xax])
    ax.set_ylabel(axis_names[yax])
    ax.set_title('Count')
    ax.set_xticks([])
    ax.set_yticks([])

    fig.suptitle(f'{title}\n{"linear" if not logscale else "log"} scale')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(ax.images[0], cax=cax)

    plt.show()

    return data, coords, counts


out_proteins = model.get_output_proteins()

# coords, counts = heatmap(data, [0, 2, 1], out_proteins, figsize=(15, 7), logscale=True, nbins=10, title='CoTX-All', truncate_at_quantile=0.999, show_counts=False)
dd, coords, counts = heatmap(
    data,
    [0, 2, 1],
    out_proteins,
    figsize=(15, 7),
    logscale=True,
    nbins=20,
    title='CoTX-All',
    truncate_at_quantile=0.999,
    show_counts=False,
)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{              --     various resampling experiments     --
# ···············································································
counts.ravel()
threshold = np.quantile(counts.ravel(), 0.6)
# now group per coords
unique_coords = np.unique(coords, axis=1)
balanced_data = []
unique_coords.T
for coord in unique_coords.T:
    idx = (coords[0] == coord[0]) & (coords[1] == coord[1])
    # we append threshold random samples
    if np.sum(idx) <= threshold:
        balanced_data.append(dd[idx, :])
    else:
        balanced_data.append(dd[idx][np.random.choice(np.sum(idx), int(threshold), replace=False)])

balanced_data = np.concatenate(balanced_data, axis=0)
_ = heatmap(
    balanced_data,
    [0, 2, 1],
    out_proteins,
    figsize=(15, 7),
    logscale=True,
    nbins=20,
    title='CoTX-All',
    truncate_at_quantile=1.0,
    show_counts=False,
)
_ = heatmap(
    balanced_data,
    [0, 2, 1],
    out_proteins,
    figsize=(15, 7),
    logscale=False,
    nbins=20,
    title='CoTX-All',
    truncate_at_quantile=1.0,
    show_counts=False,
)

data.shape
balanced_data.shape
from sklearn.neighbors import KernelDensity

xydata = data[:, [0, 2]]
kde = KernelDensity(kernel='gaussian').fit(xydata)

xax, yax, cax = [0, 2, 1]
nbins = 100
logscale = True
if logscale:
    bins_x = np.logspace(np.min(np.log10(data[:, xax])), np.max(np.log10(data[:, xax])), nbins)
    bins_y = np.logspace(np.min(np.log10(data[:, yax])), np.max(np.log10(data[:, yax])), nbins)
else:
    bins_x = np.linspace(np.min(data[:, xax]), np.max(data[:, xax]), nbins)
    bins_y = np.linspace(np.min(data[:, yax]), np.max(data[:, yax]), nbins)
mesh = np.meshgrid(bins_x, bins_y)
xy = np.vstack([mesh[0].ravel(), mesh[1].ravel()]).T
z = kde.score_samples(xy)
# z = np.log(z)
z = z.reshape(mesh[0].shape)
plt.imshow(z, origin='lower', aspect='auto', cmap='viridis')
plt.show()


# now let's try to sample from the kde
samples = kde.sample(100000)
plt.scatter(samples[:, 0], samples[:, 1], s=1)
plt.show()

# not let's reduce the data so that we flatten the distribution
# basically the probability of a point staying is inversely proportional to
# the density at this location
kde = KernelDensity(kernel='gaussian').fit(xydata)
Z = kde.score_samples(xydata)
dd = np.random.rand(len(xydata)) < 1 / (np.exp(Z) / np.max(np.exp(Z)))
newdata = xydata[dd]
plt.scatter(newdata[:, 0], newdata[:, 1], s=1)
plt.show()

kde = KernelDensity(kernel='gaussian', bandwidth=100).fit(newdata)
mesh = np.meshgrid(np.linspace(0, 10000, 100), np.linspace(0, 10000, 100))
xy = np.vstack([mesh[0].ravel(), mesh[1].ravel()]).T
z = np.exp(kde.score_samples(xy))
z = np.log(z)
z = z.reshape(mesh[0].shape)
plt.imshow(z, origin='lower', aspect='auto', cmap='viridis')
plt.show()


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{            --     quick test of the inverse functions     --
# ···············································································

params = {}
get_p = partial(bc.compute.get_param, params)


def no_quantize(paramname, values, **_):
    return values


x = np.array([0.1, 0.2, 1.0, 1000.0])

print('translation')
rng_key = jax.random.PRNGKey(0)
tl = bc.compute.translation(partial(get_p, nodeid=1), no_quantize)
inv_tl = bc.compute.inv_translation(partial(get_p, nodeid=1), no_quantize)
y = np.array([tl(xx, rng_key=rng_key) for xx in x]).squeeze()
y_inv = np.array([inv_tl(yy, rng_key=rng_key) for yy in y]).squeeze()
print(np.allclose(x, y_inv))


print('transcription')
rng_key = jax.random.PRNGKey(0)
tl = bc.compute.transcription(partial(get_p, nodeid=2), no_quantize)
inv_tl = bc.compute.inv_transcription(partial(get_p, nodeid=2), no_quantize)
y = np.array([tl(xx, rng_key=rng_key) for xx in x]).squeeze()
y_inv = np.array([inv_tl(yy, rng_key=rng_key) for yy in y]).squeeze()
print(np.allclose(x, y_inv))

print('aggregation')
rng_key = jax.random.PRNGKey(0)
tl = bc.compute.aggregation(partial(get_p, nodeid=3), no_quantize, n_outputs=2)
y = np.array([tl(xx, rng_key=rng_key) for xx in x]).squeeze()

inv_tl_0 = bc.compute.inv_aggregation(
    partial(get_p, nodeid=3), no_quantize, original_output_len=2, original_output_slot=0
)
y_inv_0 = np.array([inv_tl_0(yy[0], rng_key=rng_key) for yy in y]).squeeze()
print(np.allclose(x, y_inv_0))
inv_tl_1 = bc.compute.inv_aggregation(
    partial(get_p, nodeid=3), no_quantize, original_output_len=2, original_output_slot=1
)
y_inv_1 = np.array([inv_tl_1(yy[1], rng_key=rng_key) for yy in y]).squeeze()
print(np.allclose(x, y_inv_1))


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                        --      rebalance 1     --
# ···············································································

# rebalance 1 data
sample = 'CoTX-All'
model = models[sample]
x, y = X[sample], Y[sample]

out_proteins = model.get_output_proteins()
axis_names = [out_proteins[0], out_proteins[2], out_proteins[1]]

stats, bins = du.binstats(data, [0, 2], 1, nbins=32)
du.heatmap(stats, bins, cmap='YlGnBu', figscale=0.7, axis_names=axis_names)

balanced_data = du.balance_per_bin(data, stats, threshold_quantile=0.4, threshold_min=20)

bdf, bbins = du.binstats(balanced_data, [0, 2], 1, nbins=32)
du.heatmap(bdf, bbins, cmap='YlGnBu', figscale=0.7, axis_names=axis_names)

# quick check that we didn't change the distribution too much
chg = np.mean(np.abs(bdf['mean'] - stats['mean'])) / np.std(stats['mean'])
print(f'change in mean: {chg*100.0:.1f}%')

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                       --     rebalance all     --
# ···············································································

# now all the data, individually. We build a new X and Y, X_balanced and Y_balanced
X_balanced = {}
Y_balanced = {}
nbins = 20
for sample, model in models.items():
    print('-' * 80)
    data = np.array(Y[sample])
    out_proteins = model.get_output_proteins()
    za = out_proteins.index('eYFP')
    xa, ya = out_proteins.index('eBFP'), out_proteins.index('mKate')
    stats, bins = du.binstats(data, [xa, ya], za, nbins=nbins)
    Y_bal = du.balance_per_bin(data, stats, threshold_quantile=0.4, threshold_min=20)
    X_bal = model.get_input_from_output(Y_bal)
    X_balanced[sample] = X_bal
    Y_balanced[sample] = Y_bal
    # plot heatmap
    # before:
    print(f'before: {sample}')
    du.heatmap(
        stats,
        bins,
        figscale=0.7,
        axis_names=[out_proteins[xa], out_proteins[ya], out_proteins[za]],
        title=f'{sample} unbalanced',
        subtitle=f'{len(data)} points',
        # filename=f'../__out/unbalanced_{sample}.png',
    )
    # after:
    print(f'after: {sample}')
    bdf, bbins = du.binstats(Y_bal, [xa, ya], za, nbins=nbins)
    chg = np.mean(np.abs(bdf['mean'] - stats['mean'])) / np.std(stats['mean'])

    du.heatmap(
        bdf,
        bbins,
        figscale=0.7,
        axis_names=[out_proteins[xa], out_proteins[ya], out_proteins[za]],
        title=f'{sample} balanced',
        subtitle=f'{len(Y_bal)} points ({len(Y_bal)/len(data)*100:.1f}% of original) | changed bin means by {chg*100.0:.1f}% of std',
        # filename=f'../__out/balanced_{sample}.png',
    )

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


cfg = {
    "learning_rate": 0.001,
    "adam_w_decay": 0.01,
    "clipping": 0.001,
    "n_replicates": 50,
    "initial_param_scaling": 0.01,
    "normalize_data": False,
    "epochs": 500,
    "log_rate": 50,
    "rng_key": 1,
}

sample = 'CoTX-All'
model = models[sample]
# x, y = X_balanced[sample], Y_balanced[sample]
x, y = X[sample], Y[sample]

params_hist, loss_hist = bc.train_single_model(model, x, y, cfg)

##


from time import time

best_run = np.argmin(loss_hist[-1, :])
best_loss = loss_hist[:, best_run]
best_params = bu.get_params(params_hist, best_run)


## plot loss
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
# scale log
plt.semilogy(best_loss, alpha=0.2, color='k')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('loss history')
plt.show()

out_proteins = model.get_output_proteins()
za = out_proteins.index('eYFP')
xa, ya = out_proteins.index('eBFP'), out_proteins.index('mKate')
df, bins = du.binstats(y, [xa, ya], za, nbins=nbins)
du.heatmap(
    df,
    bins,
    figscale=1.0,
    axis_names=[out_proteins[xa], out_proteins[ya], out_proteins[za]],
    title=f'{sample} data',
)


bins.min()
nsamples = 50000
best_param = best_params[-1]

rng = jax.random.PRNGKey(10)
ypred = vmap(partial(model, best_param, rng_key=rng))(x).squeeze()

d, b = du.binstats(ypred, [xa, ya], za, nbins=nbins)


du.heatmap(
    d,
    b,
    figscale=1.0,
    axis_names=[out_proteins[xa], out_proteins[ya], out_proteins[za]],
    title=f'{sample} predicted',
)


# X AND Y are MODIFIED, which shouldn't be possible since the inverse path should be "perfect"
# is there some kind of reordering of the output happening?
# also why do I have to squeeze the output?


# TODO: same scale for all plots



## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                   --     testing simple models     --
#···············································································
dbpath = ":memory:"
dbconn = sqlite3.connect(dbpath)
bc.import_recipes_to_sql([f"../testrecipe.json5"], dbconn, lib)
network = bc.Network(lib, "singlefluo", dbconn)
inv_network = bc.inverted_network(network)

model = bc.ComputeGraphModel(network)
model.build()
inv_model = bc.ComputeGraphModel(inv_network)
inv_model.build()

rng_key = jax.random.PRNGKey(1)
params = model.init(rng_key)
rng_key = jax.random.PRNGKey(2)
inv_params = inv_model.init(rng_key)

import json
jparams = jax.tree_map(lambda x: x.tolist(), best_params[-1])
json.dumps(jparams)


inv_model(params, [20.0], rng_key=rng_key)
ut.print_xla(jit(partial(inv_model, params = inv_params)), inputs = [20.0], rng_key=rng_key)
inv_params



#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


# TODO get better dynamic range for the heatmap. Fixed.


