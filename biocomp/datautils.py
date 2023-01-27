# {{{                          --     imports     --
# ···············································································
import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial
from jax import jit, vmap
import numpy as np
import pandas as pd
import biocomp as bc
import scriptutils as ut
from pathlib import Path
import json5
import json
from . import defaults as dft
from tqdm import tqdm
import matplotlib.pyplot as plt
from jax.scipy.stats import gaussian_kde
import itertools

#                                                                            }}}

# ─────────────────────────────────────────────────────────────────────────────
#                            GENERAL PURPOSE TOOLS
# ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     data load/save     --
# ···············································································
import pickle


def save(data, path, overwrite=False, rename_if_exists=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        elif rename_if_exists:
            path = path.with_name(path.stem + '_' + path.suffix)
        else:
            raise RuntimeError(f'File {path} already exists.')
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def load(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Not a file: {path}')
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


#                                                                            }}}
### {{{                    --     plot styling tools     --
from mpl_toolkits.axes_grid1 import make_axes_locatable


def mkfig(rows, cols, size=(7, 7)):
    fig, ax = plt.subplots(rows, cols, figsize=(cols * size[0], rows * size[1]))
    return fig, ax


def remove_spines(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)


def remove_topright_spines(ax):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)


def style_violin(parts):
    for pc in parts['bodies']:
        pc.set_facecolor('k')
        pc.set_edgecolor('k')
        pc.set_linewidth(1)
    parts['cbars'].set_linewidth(0)
    parts['cmaxes'].set_color('black')
    parts['cmaxes'].set_linewidth(0.5)
    parts['cmins'].set_color('black')
    parts['cmins'].set_linewidth(0.5)


##────────────────────────────────────────────────────────────────────────────}}}

# ─────────────────────────────────────────────────────────────────────────────
#                         DATA MANAGEMENT AND BATCHING
# ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                       --     data manager     --
# ···············································································

def data_checks(X, Y, models):
    assert len(X) == len(Y)
    assert len(models) == len(X)
    namespaces = [m.node_namespace for m in models]
    assert len(set(namespaces)) == len(namespaces), 'Duplicate namespaces in models.'

    for x, y, m in zip(X, Y, models):
        assert x.shape[0] == y.shape[0], f"shape mismatch"
        assert x.shape[1] == m.n_inputs, f"input shape mismatch"
        assert y.shape[1] == m.n_outputs, f"output shape mismatch"
        outp = m.get_output_proteins()  # name of output proteins
        inp = m.get_inverted_input_proteins()  # name of input proteins
        in_pos = m.get_inverted_input_positions()
        assert len(inp) == len(in_pos)
        assert len(inp) == len(set(inp))
        assert all(iname in outp for iname in inp)
        for ipos, outpos in in_pos.items():
            assert inp[ipos] == outp[outpos]
            assert jnp.all(x[:, ipos] == y[:, outpos])


@jit
def optimal_density_subsample(X, kde, rng, quantile_threshold=0.1):
    EPSILON = 1e-12
    HIGH_DENSITIES_PENALTY = 1.05
    densities = kde.evaluate(X.T) + EPSILON
    threshold = jnp.quantile(densities, quantile_threshold)
    diceroll = jax.random.uniform(rng, shape=(len(densities),))
    selected = (densities < threshold) | (
        diceroll < (threshold / (densities * HIGH_DENSITIES_PENALTY))
    )
    return selected


# @partial(jit, static_argnames=('batch_size', 'n_batches', 'quantile_threshold'))
def sample_batches_direct(X, Y, batch_size, n_batches, kde, rng, quantile_threshold=0.1):
    assert X.shape[0] == Y.shape[0]
    EPSILON = 1e-12
    HIGH_DENSITIES_PENALTY = 1.0

    # select batch_size * n_batches random points, weight by inverse of density
    densities = kde.evaluate(X.T) + EPSILON
    threshold = np.quantile(densities, quantile_threshold)
    selection_proba = np.minimum(1.0, (threshold / (densities * HIGH_DENSITIES_PENALTY)))
    selection_proba /= np.sum(selection_proba)
    # indices = jax.random.choice(rng, X.shape[0], shape=(batch_size * n_batches,), p=selection_proba)
    np.random.seed(rng[0])
    indices = np.random.choice(X.shape[0], size=(batch_size * n_batches,), p=selection_proba)

    Xsub = X[indices]
    Ysub = Y[indices]

    # Xbatches = jnp.split(Xsub, n_batches)
    # Ybatches = jnp.split(Ysub, n_batches)
    # return jnp.array(Xbatches), jnp.array(Ybatches)

    # or with reshape:
    Xbatches = Xsub.reshape((n_batches, batch_size, Xsub.shape[1]))
    Ybatches = Ysub.reshape((n_batches, batch_size, Ysub.shape[1]))
    return Xbatches, Ybatches

# @partial(jit, static_argnames=('batch_size', 'n_batches', 'density_quantile_threshold'))
def _get_batches(X, Y, kdes, rng_key, batch_size, n_batches, density_quantile_threshold):
    all_batches = [
        sample_batches_direct(
            x,
            y,
            batch_size,
            n_batches,
            kde,
            rng,
            quantile_threshold=density_quantile_threshold,
        )
        for x, y, kde, rng in tqdm(
            list(zip(X, Y, kdes, jax.random.split(rng_key, len(X)))),
            desc='generating batches',
        )
    ]
    xbatches, ybatches = zip(*all_batches)
    # concat along the feature axis (last dimension)
    xbatches, ybatches = np.concatenate(tuple(xbatches), axis=2), np.concatenate(
        tuple(ybatches), axis=2
    )
    assert xbatches.shape == (n_batches, batch_size, sum([x.shape[1] for x in X]))
    assert ybatches.shape == (n_batches, batch_size, sum([y.shape[1] for y in Y]))
    # (N_BATCHES, BATCH_SIZE, N_MODELS * FEATURES)
    return xbatches, ybatches



class DataManager:
    def __init__(self, X: list, Y: list, models: list, cfg: dict = dft.DEFAULT_CONFIG):
        self.cfg = cfg
        self._raw_X = [np.array(x) for x in X]
        self._raw_Y = [np.array(y) for y in Y]
        self._models = models
        self._jitted_models = [jit(m) for m in models]
        self._X = self.rescale(self._raw_X)
        self._Y = self.rescale(self._raw_Y)
        self._kdes = [gaussian_kde(x.T, bw_method=cfg['kde_bw_method']) for x in self._X]
        self.indices = None
        data_checks(X, Y, models)

    def set_subset(self, indices):
        self.indices = indices

    def rescale(self, X):
        factor = self.cfg['log_factor']
        maxv = self.cfg['max_value']
        return [np.log10(1 + (x / factor)) / np.log10(maxv / factor) for x in X]

    def unscale(self, X):
        factor = self.cfg['log_factor']
        maxv = self.cfg['max_value']
        return [factor * (np.power(maxv / factor, x) - 1) for x in X]

    def get_batches(self, rng_key):
        xbatches, ybatches = _get_batches(
            self.get_X(),
            self.get_Y(),
            self.get_kdes(),
            rng_key,
            self.cfg['batch_size'],
            self.cfg['n_batches'],
            self.cfg['density_quantile_threshold'],
        )
        assert xbatches.shape[2] == sum([m.n_inputs for m in self.get_models()])
        assert ybatches.shape[2] == sum([m.n_outputs for m in self.get_models()])
        return xbatches, ybatches

    def __get(self, fromlist):
        if self.indices is not None:
            return [fromlist[i] for i in self.indices]
        return fromlist

    def get_models(self):
        return self.__get(self._models)

    def get_kdes(self):
        return self.__get(self._kdes)

    def get_X(self):
        return self.__get(self._X)

    def get_Y(self):
        return self.__get(self._Y)

    def get_raw_X(self):
        return self.__get(self._raw_X)

    def get_raw_Y(self):
        return self.__get(self._raw_Y)

    def get_jitted_models(self):
        return self.__get(self._jitted_models)

    @classmethod
    def from_xps(cls, xplist, config=dft.DEFAULT_CONFIG, **kw):
        models, samples = zip(
            *[xp.build_models(node_impl=config['node_impl'], **kw) for xp in xplist]
        )
        X, Y = zip(*[xp.get_XY(m, s) for xp, m, s in zip(xplist, models, samples)])
        X, Y, models = (
            list(itertools.chain(*X)),
            list(itertools.chain(*Y)),
            list(itertools.chain(*models)),
        )
        return cls(X, Y, models, config)


#                                                                            }}}

# {{{                         --     batches     --
# ···············································································


def split_array_uniform(arr, n_batches, rng_key):
    n = len(arr)
    batch_size = n // n_batches
    a = jax.random.permutation(rng_key, arr)
    return [a[i * batch_size : (i + 1) * batch_size] for i in range(n_batches)]


def split_array_to_len(arr, l, rng_key):
    a = jax.random.permutation(rng_key, arr)
    return [a[i * l : (i + 1) * l] for i in range(len(arr) // l)]


def make_batches_dict(Y, n_batches, rng_key, models):
    y_ = {s: split_array_uniform(Y[s], n_batches, rng_key) for s in Y.keys()}
    y_batches = [{s: y_[s][i] for s in y_.keys()} for i in range(n_batches)]
    # get x_batches from y_batches
    x_batches = []
    for y_batch in y_batches:
        x_batch = {}
        for s in y_batch.keys():
            x_batch[s] = models[s].get_input_from_output(y_batch[s])
        x_batches.append(x_batch)
    return x_batches, y_batches


def make_batches_uniform_sampling(
    X: list[np.ndarray],
    Y: list[np.ndarray],
    batch_size: int,
    rng_key,
    total_size=None,
):
    """Split data into batches of equal size, for a list of arrays (one per sample).
    Each array might not have the same size originally, but the batches will
    have the same size and there will be the same amount of batches for Each
    sample in the end. To do so, we randomly sample from the arrays to get the
    same size for each batch.

    batch_size: int, the size of each batch (per sample)
    total_size: batch_size * n_batches. If None, it uses the largest array size.

    returns: x_batches, y_batches. Dimensions are (n_batches, n_models, batch_size, n_features)

    """

    if total_size is None:  # use the largest array as target total size
        total_size = max(len(y) for y in Y)

    n_batches = total_size // batch_size

    ylist = [jax.random.choice(rng_key, jnp.array(y), (total_size,)) for y in tqdm(Y)]
    xlist = [m.get_input_from_output(ylist[i]) for i, m in enumerate(models)]

    n_outputs = max(y.shape[1] for y in ylist)
    n_inputs = max(x.shape[1] for x in xlist)

    # add 0 pad
    y_p = jnp.array([np.pad(y, ((0, 0), (0, n_outputs - y.shape[1]))) for y in ylist])
    x_p = jnp.array([np.pad(x, ((0, 0), (0, n_inputs - x.shape[1]))) for x in xlist])

    y_batches = jnp.array(np.split(y_p[:, : n_batches * batch_size], n_batches, axis=1))
    x_batches = jnp.array(np.split(x_p[:, : n_batches * batch_size], n_batches, axis=1))

    assert y_batches.shape == (n_batches, len(Y), batch_size, n_outputs)
    assert x_batches.shape == (n_batches, len(Y), batch_size, n_inputs)

    return x_batches, y_batches


def batch(X, Y, batch_size, n_batches=None):
    """Yields batches of data from X and Y."""
    n = X.shape[0]
    if n_batches is None:
        n_batches = n // batch_size
    # using sampling with replacement
    for i in range(n_batches):
        idx = np.random.choice(n, size=batch_size, replace=True)
        yield X[idx], Y[idx]


#                                                                            }}}

# ─────────────────────────────────────────────────────────────────────────────
#                            NEIGHBORHOOD BASED TOOLS
# ───────────────────────────────────── ▼ ─────────────────────────────────────
### {{{              --     knn and spatial partitionning    --
from scipy.spatial import cKDTree


@jax.jit
def weighted_quantile(data, weights, qu):
    ix = jnp.argsort(data)
    data = data[ix]
    weights = weights[ix]
    cdf = (jnp.cumsum(weights) - 0.5 * weights) / jnp.sum(weights)
    return jnp.interp(qu, cdf, data)


def get_knn(x, tree, knn=500, min_points=20, radius=0.05):
    distances, indices = tree.query(x, k=knn, distance_upper_bound=radius)
    mask = distances == np.inf
    nb_points = (~mask).sum(axis=1)
    gausspdf = (
        lambda x, mu, sigma: 1
        / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    )
    weights = gausspdf(distances, 0, radius / 2)
    indices[mask] = 0
    weights[mask] = 0
    weights[nb_points < min_points, :] = np.nan
    return indices, weights


def get_knn_mean(x, y, tree, **kw):
    indices, weights = get_knn(x, tree, **kw)
    avg = np.average(y[indices], axis=1, weights=weights)
    return avg


def get_knn_quantile(x, y, tree, qu, **kw):
    indices, weights = get_knn(x, tree, **kw)
    q = jax.vmap(weighted_quantile, in_axes=(0, 0, None))(y[indices], weights, qu)
    return q


def get_knn_smooth(xquery, logY, tree, knn=100, min_points=10, method='mean', **kw):
    if method == 'mean':
        Z = get_knn_mean(xquery, logY, knn=knn, min_points=min_points, tree=tree)
    elif method == 'quantile':
        assert 'qu' in kw
        Z = get_knn_quantile(xquery, logY, knn=knn, min_points=min_points, tree=tree, **kw)
    else:
        raise ValueError(f'Unknown method {method}')
    return Z


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                      --     heatmap methods     --


def heatmap_old(Z, x, ax):
    res = Z.shape[0]
    YY, XX = np.meshgrid(x, x)
    cmap = plt.get_cmap('YlGnBu')
    cmap.set_bad(color='#EEEEEE')
    ax.set_aspect('equal')
    im = ax.pcolormesh(
        XX,
        YY,
        Z,
        cmap=cmap,
    )
    # add contour
    ax.contour(
        XX,
        YY,
        Z,
        levels=4,
        linewidths=0.25,
    )
    loglabels = 10**x - 1
    tickfreq = res // 5
    ax.set_xticks(x[::tickfreq])
    ax.set_xticklabels([f'{l:.0e}' for l in loglabels[::tickfreq]])
    ax.set_yticks(x[::tickfreq])
    ax.set_yticklabels([f'{l:.0e}' for l in loglabels[::tickfreq]])

    # add colorbar to ax
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def smooth_heatmap(logX, logY, Z=None, x=None, ax=None, **kw):
    if ax is None:
        fig, ax = mkfig(1, 1)
    Z, x = get_knn_smooth(logX, logY, **kw)
    return heatmap_old(Z, x, ax)


def heatmap(
    ax,
    Z,
    vmin=0.1,
    vmax=1,
    ticks=[],
    ticklabels=[],
    transform=None,
    text='',
    connector=False,
    connector_orientation='bottom',
    contours=True,
    colorbar=False,
):
    cmap = plt.get_cmap('YlGnBu')
    cmap.set_bad(color='#EEEEEE')
    trans_data = ax.transData
    if transform is not None:
        trans_data = trans_data + transform
    im = ax.imshow(
        Z.T,
        origin='lower',
        aspect=1,
        cmap=cmap,
        vmin=0.1,
        vmax=vmax,
        transform=trans_data,
        interpolation='none',
    )
    # add contour
    if contours:
        ax.contour(
            Z.T,
            levels=4,
            linewidths=0.3,
            transform=trans_data,
        )

    x1, x2, y1, y2 = im.get_extent()
    w = x2 - x1
    h = y1 - y2

    ax.plot(
        [x1, x2, x2, x1, x1],
        [y1, y1, y2, y2, y1],
        "-",
        color='#AAAAAA',
        transform=trans_data,
        linewidth=0.2,
    )

    # ticks:
    if len(ticks) > 0:
        # rescale ticks to image coordinates (they are btwn 0 and 1 to start)
        sc_ticks = ticks * Z.shape[0]
        ax.set_xticks(sc_ticks)
        ax.set_xticklabels(ticklabels)
        ax.set_yticks(sc_ticks)
        ax.set_yticklabels(ticklabels)

    # colorbar
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=6)
        # no cbar spines
        for spine in cbar.ax.spines.values():
            spine.set_visible(False)

        # use same ticks if present
        if len(ticks) > 0:
            valid = ticks >= vmin
            diff = len(ticks)
            ticks = ticks[valid]
            diff -= len(ticks)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(ticklabels[diff:])

    if connector:
        if connector_orientation == 'bottom':
            txth = y1 + 0.4 * h
            ycoords = [y1 + 0.05 * h, y1 + 0.3 * h]
        else:
            txth = y2 - 0.4 * h
            ycoords = [y2 - 0.05 * h, y2 - 0.3 * h]
        ax.plot(
            [w * 0.5, w * 0.5],
            ycoords,
            ":",
            color='grey',
            transform=trans_data,
            linewidth=1,
        )
        ax.text(w * 0.5, txth, text, horizontalalignment='center', transform=trans_data)

    return im


def smooth_1d(x, y, model, rescaler, ax, res=500, xmin=0, xmax=1):
    tree = cKDTree(x)

    input_name = model.get_inverted_input_proteins()
    output_names = model.get_output_proteins()
    assert len(output_names) == 2
    assert len(input_name) == 1
    output = list(set(output_names) - set(input_name))
    output_pos = output_names.index(output[0])
    y = y[:, output_pos]

    unscaled_ticks = np.logspace(0, 12, 13)
    ticks = np.array(rescaler(unscaled_ticks))
    ticks = ticks[ticks < xmax]
    tlabels = [
        scformat.format("{:m}", x) if i > 1 else ''
        for i, x in enumerate(unscaled_ticks[: len(ticks)])
    ]

    xx = jnp.linspace(xmin, xmax, res).reshape(-1, 1)
    z = get_knn_mean(xx, y, tree)
    zq1 = get_knn_quantile(xx, y, qu=0.1, tree=tree)
    zq9 = get_knn_quantile(xx, y, qu=0.9, tree=tree)
    ax.plot(xx, z, c='k')
    ax.fill_between(xx[:, 0], zq1, zq9, alpha=0.25, color='k')
    ax.set_title(f'{model.network.name}\nSmoothed mean and [0.1 - 0.9] quantile')
    ax.set_xlabel(input_name[0])
    ax.set_ylabel(output[0])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tlabels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tlabels)


def smooth_2d(x, y, model, rescaler, ax, res=200, xmin=0, xmax=1, xslice=None, **kw):
    input_name = model.get_inverted_input_proteins()
    output_names = model.get_output_proteins()
    
    reordered_input = sorted(input_name)[::-1]
    if reordered_input != input_name:
        x = x[:, [input_name.index(i) for i in reordered_input]]
    
    assert len(output_names) == 3
    assert len(input_name) == 2
    output = list(set(output_names) - set(input_name))
    output_pos = output_names.index(output[0])
    unscaled_ticks = np.logspace(0, 12, 13)
    ticks = np.array(rescaler(unscaled_ticks))
    ticks = ticks[ticks < xmax]
    tlabels = [
        scformat.format("{:m}", x) if i > 1 else ''
        for i, x in enumerate(unscaled_ticks[: len(ticks)])
    ]

    y = y[:, output_pos]

    tree = cKDTree(x)
    xx = jnp.linspace(xmin, xmax, res)
    xygrid = jnp.array(np.meshgrid(xx, xx)).T.reshape(-1, 2)

    if x.shape[1] > 2:
        assert xslice.shape == (x.shape[1] - 2,)
        xquery = jnp.concatenate([xygrid, jnp.tile(xslice, (xygrid.shape[0], 1))], axis=1)
    else:
        xquery = xygrid

    print(kw)
    z = get_knn_smooth(xquery, y, tree=tree, **kw).reshape(res, res)

    heatmap(ax, z, ticks=ticks, ticklabels=tlabels, colorbar=True)
    ax.set_title(f'{model.network.name}\n{output[0]} smoothed mean')
    ax.set_xlabel(reordered_input[0])
    ax.set_ylabel(reordered_input[1])

    # remove plot border (the frame, I think?)
    for spine in ax.spines.values():
        spine.set_visible(False)


def smooth_3d(x, y, model, rescaler, ax, slices=np.linspace(0, 1, 5), **kw):
    # we'll divide the third axis into slices
    inputs = model.get_inverted_input_proteins()
    # divide ax using make_axes_locatable
    divider = make_axes_locatable(ax)
    axes = [ax]
    width = 1 / (len(slices) - 1)
    for i in range(len(slices) - 1):
        axes.append(divider.append_axes('top', size=width, pad=0.01))
    # plot each slice
    for i, s in enumerate(slices):
        smooth_2d(x, y, model, rescaler, axes[i], xslice=np.array([s]), **kw)
        axes[i].set_title(f'{inputs[2]} ≈ {s:.2f}')


def setup_fig(title):
    fig, ax = plt.subplots(1, 1)
    fig.patch.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.suptitle(title)
    return fig, ax


import matplotlib.transforms as mtransforms


def timelapse_persp(Q, title, labels=None, outputfile=None, show=True, **kw):
    overlap = 0.1
    vmax = np.nanmax(Q)
    fig, ax = setup_fig(title)
    w, _ = Q[0].shape
    fig.set_size_inches(len(Q) * 3, 6)
    for i, s in enumerate(Q):
        transform = mtransforms.Affine2D().scale(1, 1.5)
        transform = transform.skew_deg(0, 20).translate(w * (1 - overlap) * i, 0)
        im = do_plot(ax, s, vmax, transform, labels[i], **kw)
    ax.set_xlim(-0.5 * w, (len(Q) - overlap) * w)
    cbar_ax = fig.add_axes([0.85, 0.28, 0.015, 0.5])
    cb = fig.colorbar(im, cax=cbar_ax, drawedges=False)
    cb.outline.set_linewidth(0)
    cb.ax.locator_params(nbins=5)
    if outputfile is not None:
        plt.savefig(outputfile, dpi=150)
    if show:
        plt.show()
    plt.close()


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                --     summary model plot functions     --
def model_plot(model: bc.ComputeGraphModel, X, Y, rescaler, ax, kde=None, **kw):
    x, y = X, Y
    if kde is not None:
        rng = jax.random.PRNGKey(0)
        subsample = optimal_density_subsample(x, kde, rng, quantile_threshold=0.1)
        x, y = x[subsample], y[subsample]

    ninputs = model.n_inputs
    if ninputs == 1:
        smooth_1d(x, y, model, rescaler, ax, **kw)
    elif ninputs == 2:
        smooth_2d(x, y, model, rescaler, ax, **kw)
    elif ninputs == 3:
        smooth_3d(x, y, model, rescaler, ax, **kw)


def eval_model_plot(
    model: bc.ComputeGraphModel,
    params,
    rescaler,
    ax,
    npoints=50000,
    key=jax.random.PRNGKey(0),
    jitted=None,
    **kw,
):

    k_i, k_q = jax.random.split(key)
    inputs = jax.random.uniform(k_i, (npoints, model.n_inputs))
    quantiles = jax.random.uniform(k_q, (npoints, model.n_outputs))
    keys = jax.random.split(key, npoints)
    jm = jitted or model
    results = vmap(jm, in_axes=(None, 0, 0, 0))(params, inputs, quantiles, keys)
    model_plot(model, inputs, results, rescaler, ax, **kw)


def report(params, dman, id, suptitle=''):
    fig, ax = mkfig(1, 2, size=(4, 4))
    model = dman.get_models()[id]
    mX = dman.get_X()[id]
    mY = dman.get_Y()[id]
    model_plot(model, mX, mY, dman.rescale, ax[0], kde=dman.get_kdes()[id])
    eval_model_plot(model, params, dman.rescale, ax[1], jitted=dman.get_jitted_models()[id])
    ax[0].set_title(f'Original data (mean)')
    ax[1].set_title(f'Predicted (mean)')
    fig.suptitle(f'{suptitle} {model.node_namespace}')
    return fig, ax


##────────────────────────────────────────────────────────────────────────────}}}






### {{{                         --     archives     --
# ─────────────────────────────────────────────────────────────────────────────
#                              BINNING BASED TOOLS
# ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     binstats     --
# ···············································································


def binstats_nbins(data, bin_columns, stat_column, nbins=20, log=True, stats=["mean", "count"]):
    vmin, vmax = data[:, bin_columns].min(axis=0), data[:, bin_columns].max(axis=0)
    vmin = np.ones_like(vmin)
    if log:
        bins = np.geomspace(vmin, vmax, nbins)
    else:
        bins = np.linspace(vmin, vmax, nbins)
    coords = np.array(
        [np.digitize(data[:, i], bins[:, b], right=True) for b, i in enumerate(bin_columns)]
    ).T
    df = pd.DataFrame(data)
    df['coords'] = [tuple(x) for x in coords]
    df = df.reset_index().groupby('coords').agg({stat_column: stats, 'index': lambda x: list(x)})
    df.columns = stats + ['indices']
    return df, bins


DUTILS_BINNING_POWER_RANGE = 30
DUTILS_BINNING_VMAX_EPSILON = 10.0 ** (-DUTILS_BINNING_POWER_RANGE)


def mk_log_grid(vmin, vmax, resolution):
    """
    Generate logarithmically spaced bins with a specified resolution, between vmin and vmax
    Returns A list of arrays of bin edges, one per dimension.
    """
    vmin, vmax = np.asarray(vmin), np.asarray(vmax)
    assert vmin.shape == vmax.shape
    if vmin.ndim == 0:
        vmin, vmax = vmin[None], vmax[None]
    powers = 10.0 ** np.arange(-DUTILS_BINNING_POWER_RANGE, DUTILS_BINNING_POWER_RANGE, resolution)
    first_bin = np.array(
        [powers[np.clip(np.searchsorted(powers, v) - 1, 0, len(powers) - 1)] for v in vmin]
    )
    last_bin = np.array(
        [powers[np.clip(np.searchsorted(powers, v), 0, len(powers) - 1)] for v in vmax]
    )
    nbins = np.ceil(np.log10(last_bin / first_bin) / resolution).astype(int)
    nbins = np.maximum(nbins, 1)
    bin_edges = [
        np.geomspace(first_bin[i], last_bin[i], nbins[i] + 1) for i in range(len(first_bin))
    ]
    return bin_edges


def binstats(
    data,
    output_protein_names,
    bin_proteins=None,
    resolution=0.5,
    bin_min=1e-12,
    bin_max=None,
    force_minmax=False,
):
    """
    Calculate statistics (mean and count) for bins in multi-dimensional space.

    Parameters
    ----------
    data : array-like
        The data to be binned.
    output_protein_names : list of str
        The names of the features in the input data.
    bin_proteins : list of str, optional
        The names of the features to use for binning.
    resolution : float, optional
        The resolution of the bins, expressed as the number of decades per bin.
    bin_min : float, optional
        The minimum value for the bins.
    bin_max : float, optional
        The maximum value for the bins.
    force_minmax : bool, optional
        If `True`, the bins will always start at `bin_min` and end at `bin_max`.

    Returns
    -------
    df2 : DataFrame
        A DataFrame containing the binned data, with columns for the mean and count of each bin and rows indexed by the bin coordinates.
    bins : dict
        A dictionary containing the bin edges for each feature used for binning.
    """
    if bin_proteins is None:
        bin_proteins = output_protein_names
    bin_axisid = [output_protein_names.index(p) for p in bin_proteins]
    vmin, vmax = (
        data[:, bin_axisid].min(axis=0),
        data[:, bin_axisid].max(axis=0) + DUTILS_BINNING_VMAX_EPSILON,
    )
    if bin_min is not None:
        if force_minmax:
            vmin = np.ones_like(vmin) * bin_min
        else:
            vmin = np.maximum(vmin, bin_min)
    if bin_max is not None:
        if force_minmax:
            vmax = np.ones_like(vmax) * bin_max
        else:
            vmax = np.minimum(vmax, bin_max)

    bin_edges = mk_log_grid(vmin, vmax, resolution)
    coords = np.array([np.digitize(data[:, i], be) for i, be in zip(bin_axisid, bin_edges)]).T - 1
    df = pd.DataFrame(data, columns=output_protein_names)
    df['coord'] = [tuple(c) for c in coords]
    df2 = df.groupby('coord').agg(['mean']).reset_index()
    df2['indices'] = df.reset_index().groupby('coord').agg({'index': lambda x: list(x)}).values
    df2['indices'] = df2['indices'].apply(lambda x: np.array(x, dtype=int))
    df2['count'] = df2['indices'].apply(len)
    for i, p in enumerate(bin_proteins):
        df2[('coords', p)] = df2['coord'].apply(lambda x: x[i])
    df2 = df2.drop('coord', axis=1, level=0)
    df2 = df2.set_index([('coords', p) for p in bin_proteins])

    bins = {p: be for p, be in zip(bin_proteins, bin_edges)}
    return df2, bins


#                                                                            }}}
# {{{               --     balancing individual dataset     --
# ···············································································


def balance_per_bin(
    data, statdf, rng_key, threshold_quantile=0.4, threshold_min=20, replacement=True
) -> np.ndarray:
    assert 'count' in statdf.columns
    assert 'indices' in statdf.columns
    threshold = max(
        jnp.quantile(statdf[statdf['count'] > 0]['count'].values, threshold_quantile), threshold_min
    )
    balanced_indices = []
    for i, row in tqdm(statdf.iterrows()):
        # each row describes a bin with:
        # - count: number of datapoints in the bin
        # - indices: indices of datapoints in the bin
        n = min(int(threshold), int(row['count']))
        # if replacement:
        # n = int(threshold)
        choice = jax.random.choice(
            rng_key,
            row['indices'][0],
            shape=(n,),
            replace=replacement,
        )
        balanced_indices.append(choice)
    balanced_data = data[jnp.concatenate(balanced_indices), :]
    return balanced_data


def balance_each_dataset(
    models: dict[str, bc.ComputeGraphModel],
    Y: dict[str, np.ndarray],
    rng_key,
    bin_resolution=0.5,
    threshold_quantile=0.4,
    threshold_min=20,
    replacement=True,
):
    """balances each dataset individually (not across datasets but within each dataset,
    so that each dataset aims for a similar number of samples per bin)"""
    X_balanced, Y_balanced = {}, {}
    for sample, model in tqdm(models.items(), desc='balancing datasets'):
        data = Y[sample]
        out_proteins = model.get_output_proteins()
        in_proteins = model.get_inverted_input_proteins()
        stats, _ = binstats(data, out_proteins, in_proteins, resolution=bin_resolution)
        rng_key, subkey = jax.random.split(rng_key)
        Y_balanced[sample] = balance_per_bin(
            data,
            stats,
            subkey,
            threshold_quantile=threshold_quantile,
            threshold_min=threshold_min,
            replacement=replacement,
        )
        X_balanced[sample] = model.get_input_from_output(Y_balanced[sample])
    return X_balanced, Y_balanced


#                                                                            }}}
# {{{                       --     plot heatmap     --
# ···············································································
import string


class ShortScientificFormatter(string.Formatter):
    def format_field(self, value, format_spec):
        if format_spec == 'm':
            if value < 1000:
                if value == int(value):
                    return super().format_field(int(value), '')
                else:
                    return super().format_field(value, '.1f')
            else:
                if value == int(value):
                    return super().format_field(value, '.0e').replace('e+0', 'e').replace('e+', 'e')
                else:
                    return super().format_field(value, '.1e').replace('e+0', 'e').replace('e+', 'e')
        else:
            return super().format_field(value, format_spec)


scformat = ShortScientificFormatter()
fmt = scformat


def model_parallel_coords(
    model, y, n_samples=500, cmap='Spectral_r', title=None, maxval=1e3, minval=1e-5
):
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    out_proteins = model.get_output_proteins()
    in_proteins = model.get_inverted_input_proteins()
    stats, bins = binstats(y, out_proteins, resolution=0.25)
    prot_diff = set(out_proteins) - set(in_proteins)

    # the plot will have one coordinate (vertical line) per out_protein.
    # each line is taken from stats (the mean of the bin)
    # stats is a dataframe with columns [('protein_name', 'mean'), ...]
    mean_values = jnp.array([stats[p]['mean'].values for p in out_proteins]).T
    choice = jax.random.choice(
        jax.random.PRNGKey(0), mean_values.shape[0], shape=(n_samples,), replace=True
    )
    mean_values = mean_values[choice]

    # 1 subplot per prot_diff
    fig, axes = plt.subplots(len(prot_diff), 1, figsize=(10, 9 * len(prot_diff)))

    if len(prot_diff) == 1:
        axes = [axes]

    for z_prot, ax in zip(prot_diff, axes):
        z_values = stats[z_prot]['mean'].values[choice]

        # x axis should be each protein name (we display a vertical line for each protein)
        ax.set_xticks(range(len(out_proteins)))
        ax.set_xticklabels(out_proteins, rotation=40)
        ax.vlines(range(len(out_proteins)), 0, maxval * 2, alpha=0.2, color='black')

        # y axis should be the mean value of the bin
        ax.set_yscale('log')
        cmap = plt.get_cmap(cmap)
        norm = colors.LogNorm(vmin=minval, vmax=maxval)
        clrs = [cmap(norm(z)) for z in z_values]

        for i, (x, c) in enumerate(zip(mean_values, clrs)):
            ax.plot(x, color=c, alpha=0.25, linewidth=2)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(sm, cax=cax, orientation='vertical', label=z_prot)

        ax.set_ylim(minval, maxval * 2)
        ax.set_title(f'{model.network.name if title is None else title}')

    return fig, axes


def model_heatmap(
    model,
    ydata,
    inner_resolution=0.5,
    outer_resolution=1,
    base_size=10,
    mincount=300,
    z_prot=None,
    lims=(1e-4, 1e2),
    title=None,
):
    """can be used to plot heatmaps for a model with up to 4 inputs"""
    out_proteins = model.get_output_proteins()
    in_proteins = model.get_inverted_input_proteins()
    if z_prot is None:
        z_prot = set(out_proteins) - set(in_proteins)
        z_prot = list(z_prot)[0]

    # we have more than 2 in_proteins: we will plot an array of heatmaps
    # for 3 in_proteins, we will have only one axis of groups
    # for 4 in_proteins, we will have 2 axes of groups

    import itertools

    n_in_proteins = len(in_proteins)
    axes_proteins = in_proteins[: n_in_proteins - 2]
    ax_stats = [
        binstats(ydata, out_proteins, bin_proteins=[axp], resolution=outer_resolution)
        for axp in axes_proteins
    ]
    group_indices = [s[s['count'] > mincount]['indices'] for s, _ in ax_stats]
    group_mean = [
        s[s['count'] > mincount][(axp, 'mean')] for axp, (s, _) in zip(axes_proteins, ax_stats)
    ]

    axes_len = [len(g) for g in group_mean]
    n_axes = len(axes_len)

    if n_axes == 1:
        fig, axes = plt.subplots(1, axes_len[0], figsize=(base_size * axes_len[0], base_size))
        axes = axes.flatten()
    elif n_axes == 2:
        fig, axes = plt.subplots(
            axes_len[0], axes_len[1], figsize=(base_size * axes_len[1], base_size * axes_len[0])
        )
        axes = axes.flatten()
    elif n_axes == 0:
        fig, axes = plt.subplots(1, 1, figsize=(base_size, base_size))
        axes = [axes]
    else:
        raise ValueError(f'Cannot plot {n_axes} axes')

    bin_prots = set(in_proteins) - set(axes_proteins)

    ax_prot_ranges = dict()
    for i, (ax, indices, mean) in enumerate(
        zip(axes, itertools.product(*group_indices), itertools.product(*group_mean))
    ):
        data = ydata[indices]
        ax_prot_ranges[i] = [
            (data[:, out_proteins.index(p)].min(), data[:, out_proteins.index(p)].max())
            for p in axes_proteins
        ]
        ax_stats, ax_bins = binstats(
            data,
            out_proteins,
            bin_proteins=bin_prots,
            resolution=inner_resolution,
            bin_min=lims[0],
            bin_max=lims[1],
            force_minmax=True,
        )
        old_heatmap(
            ax_stats,
            ax_bins,
            axes=[ax],
            stat_columns=['mean'],
            z_protein=z_prot,
            lims={'mean': lims},
            show=False,
            count_threshold=2,
        )
        # remove axis labels except for the last row and first column
        if n_axes == 1 and i < axes_len[0] - 1:
            ax.set_xlabel('')
        elif n_axes == 2:
            if i % axes_len[1] != 0:
                ax.set_ylabel('')
            if i // axes_len[1] != axes_len[0] - 1:
                ax.set_xlabel('')

        axtitle = ''
        if n_axes > 0:
            axtitle = f'{axes_proteins[0]}={mean[0]:.2f}'
        if n_axes == 2:
            axtitle += f', {axes_proteins[1]}={mean[1]:.2f}'
        ax.set_title(axtitle)

    if title is None:
        title = f'Heatmap for {z_prot}'

    fig.suptitle(title)

    return fig, axes


def model_heatmap_old(model, y, resolution=0.5):
    out_proteins = model.get_output_proteins()
    in_proteins = model.get_inverted_input_proteins()
    stats, bins = binstats(y, out_proteins, in_proteins, resolution=resolution)
    z_prot = set(out_proteins) - set(in_proteins)
    fig, ax = old_heatmap(
        stats,
        bins,
        figscale=0.6,
        stat_columns=['mean'],
        z_protein=z_prot.pop(),
        lims={'mean': (1e-3, 1e3)},
        title=f'{model.network.name} data',
        subtitle=f'{len(y)} data points',
        show=False,
    )
    return fig, ax


def old_heatmap(
    statdf,
    bins,
    fig=None,
    axes=None,
    stat_columns=['mean'],
    z_protein=None,
    lims={},
    figscale=1.0,
    count_threshold=2,
    cmap='YlGnBu',
    title=None,
    subtitle=None,
    filename=None,
    show=True,
    **kwargs,
):

    from matplotlib.colors import LogNorm

    fontsize = 12 * figscale

    nstats = len(stat_columns)
    figsize = (figscale * 10 * nstats + (nstats - 1) * 4 * figscale, figscale * 10)
    assert len(axes) == nstats

    if axes is None:
        assert fig is None
        fig, axes = plt.subplots(1, nstats, figsize=figsize)
        if nstats == 1:
            axes = [axes]
    else:
        assert len(axes) == nstats

    df = statdf[statdf['count'] >= count_threshold]

    xy_axis = [n[1] for n in df.index.names]

    if z_protein is None:
        z_axis = [c[0] for c in df.columns if c[1] and c[0] not in xy_axis]
        assert len(z_axis) == 1
        z_axis = z_axis[0]
    else:
        z_axis = z_protein

    for stat, ax in zip(stat_columns, axes):
        if stat == 'indices':
            continue
        # get order in which xy_axis appear in columns
        xbin, ybin = bins[xy_axis[0]], bins[xy_axis[1]]
        Z = np.full((len(xbin), len(ybin)), np.nan)
        # coords tuple is the index of the df
        statcol = 'count' if stat == 'count' else (z_axis, stat)
        for coords, value in df[statcol].items():
            Z[coords] = max(value, 1e-20)

        # nans should be grey
        cmap = plt.get_cmap(cmap)
        cmap.set_bad(color='#EEEEEE')
        vmin = 0 if stat == 'count' else None

        norm = None
        if stat != 'count':
            if stat in lims:
                vmin, vmax = lims[stat]
            else:
                vmin, vmax = np.nanmin(Z), np.nanmax(Z)
            norm = LogNorm(vmin=vmin, vmax=vmax)

        im = ax.imshow(Z.T, origin='lower', norm=norm, cmap=cmap)

        if stat == 'count':
            ax.set_title('count', fontsize=fontsize)
        else:
            ax.set_title(f'{z_axis} {stat}', fontsize=fontsize)

        ax.set_xlabel(xy_axis[0])
        ax.set_ylabel(xy_axis[1])

        # add white grid lines to separate bins
        ax.set_xticks(np.arange(len(xbin) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(ybin) + 1) - 0.5, minor=True)
        grid_thickness = 50.0 * figscale / max(len(bins), 15)
        if grid_thickness > 0.75:
            grid_thickness = max(1.00, grid_thickness)
            ax.grid(which='minor', axis='both', linestyle='-', color='w', linewidth=grid_thickness)
        ax.tick_params(which='minor', bottom=False, left=False)

        tick_freq = max(1, int(len(xbin) / 7))
        ax.set_xticks(np.arange(len(xbin))[::tick_freq], minor=False)
        ax.set_yticks(np.arange(len(ybin))[::tick_freq], minor=False)
        # xl = [f"{x:.0f}" if x < 100 else fmt.format("{:m}", x) for x in xbin[::tick_freq]]
        # yl = [f"{x:.0f}" if x < 100 else fmt.format("{:m}", x) for x in ybin[::tick_freq]]
        xl = [fmt.format("{:m}", x) for x in xbin[::tick_freq]]
        yl = [fmt.format("{:m}", x) for x in ybin[::tick_freq]]
        ax.set_xticklabels(xl, minor=False)
        ax.set_yticklabels(yl, minor=False)

        # remove border
        for spine in ax.spines.values():
            spine.set_visible(False)

        if fig is not None:
            width = ax.get_position().width
            height = ax.get_position().height
            cax = fig.add_axes(
                [
                    ax.get_position().x1 + 0.02 * figscale,
                    ax.get_position().y0 + height * 0.25,
                    0.017 * figscale,
                    height / 2,
                ]
            )
            from matplotlib.ticker import LogFormatterSciNotation

            cb_format = None
            if norm is not None:
                cb_format = LogFormatterSciNotation()
            fig.colorbar(im, cax=cax, format=cb_format)
            cax.tick_params(labelsize=fontsize)
            cax.yaxis.get_offset_text().set_fontsize(fontsize)
            font = 'Roboto Mono Light for Powerline'
            if font in plt.rcParams['font.family']:
                cax.yaxis.get_offset_text().set_fontname(font)
                for item in (
                    [ax.xaxis.label, ax.yaxis.label]
                    + ax.get_xticklabels()
                    + ax.get_yticklabels()
                    + cax.get_xticklabels()
                    + cax.get_yticklabels()
                ):
                    item.set_fontname(font)

        # set all font sizes
        ax.tick_params(labelsize=fontsize)
        ax.title.set_fontsize(fontsize * 1.4)
        ax.xaxis.label.set_fontsize(fontsize)
        ax.yaxis.label.set_fontsize(fontsize)

        # if 'Arial'  in plt.rcParams['font.family']:
        # ax.title.set_fontname('Arial')
        ax.title.set_fontweight('light')
        ax.title.set_fontstretch('expanded')

    title = title if title is not None else f'{z_axis} stats'
    # suptitle should be higher than default

    if fig is not None:
        fig.suptitle(
            title,
            fontsize=fontsize * 1.8,
            fontweight='light',
            fontstretch='expanded',
            # fontname='Arial',
            y=0.99,
        )

    if subtitle and fig is not None:
        # increase figure height
        # fig.set_figheight(fig.get_figheight() * 1.1)
        # write smaller, below the title, italic, centered
        fig.text(
            0.5,
            0.95,
            subtitle,
            fontsize=fontsize * 1.2,
            fontweight='light',
            fontstretch='expanded',
            # fontname='Arial',
            horizontalalignment='center',
            verticalalignment='top',
            style='italic',
        )

    if filename:
        from matplotlib.transforms import Bbox

        plt.savefig(
            filename, dpi=300, bbox_inches=Bbox([[0.5 * figscale, 0], [figsize[0], figsize[1]]])
        )

    if show:
        plt.show()

    return fig, axes


# example usage
# out_proteins = model.get_output_proteins()
# in_proteins = model.get_inverted_input_proteins()
# stats, bins = binstats(y, out_proteins, in_proteins, resolution=0.5)
# heatmap(
# stats,
# bins,
# figscale=0.6,
# stat_columns=['mean','count'],
# z_protein='eYFP',
# lims={'mean': (1e3, 1e8)},
# title=f'{model.network.name}',
# subtitle=f'{len(y)} data points',
# )

#                                                                            }}}

# {{{                       --     data manager     --
# ···············································································


class DataManager_bin:
    def __init__(self, X: dict, Y: dict, models: dict, enable_checks=True):
        self.X = X
        self.Y = Y
        self.models = models
        self.enable_checks = enable_checks

        basic_data_checks(X, Y, models)

        if self.enable_checks:
            advanced_data_checks(X, Y, models)
            self.N_TEST = 100

    def normalize(self, factor):
        # for now, let's just scale the data
        # we can add more later
        self.X = jax.tree_map(lambda x: x / factor, self.X)
        self.Y = jax.tree_map(lambda x: x / factor, self.Y)
        basic_data_checks(self.X, self.Y, self.models)
        if self.enable_checks:
            advanced_data_checks(self.X, self.Y, self.models)
        return self.X, self.Y

    def consolidated(raw_X, raw_Y, rng_key):
        keys = list(raw_X.keys())
        # all sets don't necessary have the same shape, so we want to:
        # - pad with zeros to the right on the feature axis
        # - fill with random samples (from the same set of course) on the sample axis
        # first, let's get the max shape
        max_shape_x = (0, 0)
        max_shape_y = (0, 0)
        for k in keys:
            max_shape_x = max(max_shape_x, raw_X[k].shape)
            max_shape_y = max(max_shape_y, raw_Y[k].shape)
        assert max_shape_x[0] == max_shape_y[0]
        # now, let's pad and fill
        X = {}
        Y = {}
        for k in keys:
            X[k] = np.zeros(max_shape_x)
            Y[k] = np.zeros(max_shape_y)
            assert raw_X[k].shape[0] == raw_Y[k].shape[0]
            rng_key, subkey = jax.random.split(rng_key)
            # indices contains all the indices of the samples in the set, in a loop
            # until we have enough samples to fill the set
            shuff_range = jax.random.permutation(rng_key, raw_X[k].shape[0])
            indices = np.tile(shuff_range, max_shape_x[0] // raw_X[k].shape[0] + 1)[
                : max_shape_x[0]
            ]
            indices = jax.random.permutation(subkey, indices)
            X[k][: raw_X[k].shape[0], : raw_X[k].shape[1]] = raw_X[k][indices]
            Y[k][: raw_Y[k].shape[0], : raw_Y[k].shape[1]] = raw_Y[k][indices]
        return X, Y

    def preprocess(self, rng_key, cfg):
        XX, YY = balance_each_dataset(
            self.models,
            self.Y,
            rng_key,
            bin_resolution=cfg['balance_bin_resolution'],
            threshold_quantile=cfg['balance_threshold_quantile'],
            threshold_min=cfg['balance_threshold_min'],
        )

        basic_data_checks(XX, YY, self.models)
        if self.enable_checks:
            advanced_data_checks(XX, YY, self.models)
            # let's also pick n random samples for each YY and check
            # - that they can be found in the original Y
            # - that the matching X is the same
            for k, v in YY.items():
                indices = jax.random.choice(rng_key, v.shape[0], (self.N_TEST,))
                assert np.all(jax.vmap(jnp.all)(v[indices] == self.Y[k]))
                assert np.all(jax.vmap(jnp.all)(XX[k][indices] == self.X[k]))

        self.X = XX
        self.Y = YY
        self.normalize(cfg['normalize_factor'])

    def postscale(self, data, factor):
        return jax.tree_map(lambda x: x * factor, data)


#                                                                            }}}

def sample_batches(
    X, Y, batch_size, n_batches, kde, rng, quantile_threshold=0.1, x_pad_to=None, y_pad_to=None
):
    assert X.shape[0] == Y.shape[0]
    total_size = batch_size * n_batches

    selection = density_subsample(X, kde, rng, quantile_threshold)
    Xsub, Ysub = X[selection], Y[selection]

    while Xsub.shape[0] < total_size:
        print('resampling')
        rng, _ = jax.random.split(rng)
        selection = density_subsample(X, kde, rng, quantile_threshold)
        Xsub, Ysub = jnp.concatenate((Xsub, X[selection])), jnp.concatenate((Ysub, Y[selection]))

    assert Xsub.shape[0] == Ysub.shape[0]
    indices = jax.random.choice(rng, Xsub.shape[0], shape=(total_size,), replace=False)

    # pad with nans to the right if necessary
    if x_pad_to is not None:
        Xsub = jnp.pad(Xsub, ((0, 0), (0, x_pad_to - Xsub.shape[1])), constant_values=jnp.nan)
    if y_pad_to is not None:
        Ysub = jnp.pad(Ysub, ((0, 0), (0, y_pad_to - Ysub.shape[1])), constant_values=jnp.nan)

    Xbatches = jnp.split(Xsub[indices], n_batches)
    Ybatches = jnp.split(Ysub[indices], n_batches)

    return Xbatches, Ybatches


# @partial(jit, static_argnames=('batch_size', 'n_batches', 'x_pad_to', 'y_pad_to'))
def sample_batches_jit(
    X, Y, batch_size, n_batches, kde, rng, quantile_threshold=0.1, x_pad_to=None, y_pad_to=None
):
    assert X.shape[0] == Y.shape[0]
    total_size = batch_size * n_batches
    rngk, rng = jax.random.split(rng)

    selection = density_subsample(X, kde, rngk, quantile_threshold).astype(jnp.int32)

    selection, rng = jax.lax.while_loop(
        lambda x: jnp.sum(x[0]) < total_size,
        lambda x: tuple(
            [
                x[0] + density_subsample(X, kde, rngk, quantile_threshold).astype(jnp.int32),
                jax.random.split(x[1])[0],
            ]
        ),
        (selection, rngk),
    )

    # while jnp.sum(selection) < total_size:
    # rng, _ = jax.random.split(rng)
    # selection = selection + density_subsample(X, kde, rng, quantile_threshold).astype(jnp.int32)

    selection_probability = selection / jnp.sum(selection)
    indices = jax.random.choice(rng, X.shape[0], shape=(total_size,), p=selection_probability)

    Xsub = X[indices]
    Ysub = Y[indices]

    # pad with nans to the right if necessary
    if x_pad_to is not None:
        Xsub = jnp.pad(Xsub, ((0, 0), (0, x_pad_to - Xsub.shape[1])), constant_values=jnp.nan)
    if y_pad_to is not None:
        Ysub = jnp.pad(Ysub, ((0, 0), (0, y_pad_to - Ysub.shape[1])), constant_values=jnp.nan)

    Xbatches = jnp.split(Xsub, n_batches)
    Ybatches = jnp.split(Ysub, n_batches)

    return Xbatches, Ybatches


def jitable_sample_batches_attempt(
    X, Y, batch_size, n_batches, kde, rng, quantile_threshold=0.07, x_pad_to=None, y_pad_to=None
):
    assert X.shape[0] == Y.shape[0]
    total_size = batch_size * n_batches
    rngk, _ = jax.random.split(rng)

    @jit
    def select_indices():
        selection = density_subsample(X, kde, rngk, quantile_threshold)[None, :]
        selection = jax.lax.while_loop(
            lambda x: jnp.sum(x) < total_size,
            lambda x: jnp.concatenate(
                (x, density_subsample(X, kde, rngk, quantile_threshold)[None, :]), axis=0
            ),
            selection,
        )
        n_selection = jnp.sum(selection)
        row_id = jnp.arange(X.shape[0])[None, :]
        row_id = jnp.tile(row_id, (selection.shape[0], 1))
        preselected = jnp.where(selection, row_id, -1).reshape(-1)
        preselected = jnp.sort(preselected)[n_selection:]
        return preselected

    preselected = select_indices()
    indices = jax.random.choice(rngk, preselected, shape=(total_size,), replace=False)
    Xsub, Ysub = X[indices], Y[indices]
    # pad with nans to the right if necessary
    if x_pad_to is not None:
        Xsub = jnp.pad(Xsub, ((0, 0), (0, x_pad_to - Xsub.shape[1])), constant_values=jnp.nan)
    if y_pad_to is not None:
        Ysub = jnp.pad(Ysub, ((0, 0), (0, y_pad_to - Ysub.shape[1])), constant_values=jnp.nan)

    assert Xsub.shape[0] == Ysub.shape[0]
    Xbatches = jnp.split(Xsub, n_batches)
    Ybatches = jnp.split(Ysub, n_batches)

    return Xbatches, Ybatches


##────────────────────────────────────────────────────────────────────────────}}}
