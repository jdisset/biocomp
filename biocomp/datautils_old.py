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

##────────────────────────────────────────────────────────────────────────────}}}

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


def remove_axis_and_spines(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def set_size(w, h, axes, fig):
    """w, h: width, height in inches"""
    l = min([ax.figure.subplotpars.left for ax in axes])
    r = max([ax.figure.subplotpars.right for ax in axes])
    t = max([ax.figure.subplotpars.top for ax in axes])
    b = min([ax.figure.subplotpars.bottom for ax in axes])
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    fig.set_size_inches(figw, figh)


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
    HIGH_DENSITIES_PENALTY = 1.00
    densities = kde.evaluate(X.T) + EPSILON
    threshold = jnp.quantile(densities, quantile_threshold)
    diceroll = jax.random.uniform(rng, shape=(len(densities),))
    selected = (densities < threshold) | (
        diceroll < (threshold / (densities * HIGH_DENSITIES_PENALTY))
    )
    return selected


# @partial(jit, static_argnames=('batch_size', 'n_batches', 'quantile_threshold'))
def sample_batches_direct(
    X, Y, batch_size, n_batches, kde, rng, quantile_threshold=0.05, density_coords=0.3
):
    assert X.shape[0] == Y.shape[0]
    EPSILON = 1e-16
    HIGH_DENSITIES_PENALTY = 1.0

    # select batch_size * n_batches random points, weight by inverse of density
    densities = kde.evaluate(X.T) + EPSILON
    threshold = jnp.quantile(densities, quantile_threshold)
    midX = jnp.ones((X.shape[1],)) * density_coords
    density_at_midX = kde.evaluate(midX.T)
    threshold = min(threshold, density_at_midX)
    selection_proba = jnp.minimum(1.0, (threshold / (densities * HIGH_DENSITIES_PENALTY)))
    indices = jax.random.choice(rng, X.shape[0], shape=(batch_size * n_batches,), p=selection_proba)
    # or with numpy:
    # selection_proba /= np.sum(selection_proba)
    # indices = np.random.choice(X.shape[0], size=(batch_size * n_batches,), p=selection_proba)

    Xsub = jnp.take(X, indices, axis=0)
    Ysub = jnp.take(Y, indices, axis=0)

    Xbatches = Xsub.reshape((n_batches, batch_size, Xsub.shape[1]))
    Ybatches = Ysub.reshape((n_batches, batch_size, Ysub.shape[1]))
    return Xbatches, Ybatches


# @partial(jit, static_argnames=('batch_size', 'n_batches', 'density_quantile_threshold'))
def _get_batches(
    X, Y, kdes, rng_key, batch_size, n_batches, density_quantile_threshold, density_coords
):
    all_batches = [
        sample_batches_direct(
            x,
            y,
            batch_size,
            n_batches,
            kde,
            rng,
            quantile_threshold=density_quantile_threshold,
            density_coords=density_coords,
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
        MAX_VAL = 1.5
        assert max([x.max() for x in self._X]) < MAX_VAL
        assert max([y.max() for y in self._Y]) < MAX_VAL
        self.indices = None
        self.gen_kdes()
        data_checks(X, Y, models)

    def set_subset(self, indices):
        self.indices = indices

    def gen_kdes(self, bw=None, max_n=10000):
        if bw is None:
            bw = self.cfg['kde_bw_method']
        key = jax.random.PRNGKey(0)
        self._kdes = [
            gaussian_kde(
                x[jax.random.choice(key, x.shape[0], shape=(max_n,))].T,
                bw_method=bw,
            )
            for x in self._X
        ]

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
            self.cfg['coords_for_density_threshold'],
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

    # def make_subset(self, ids):
    # sub_x = [self._raw_X[i] for i in ids]
    # sub_y = [self._raw_Y[i] for i in ids]
    # sub_models = [self._models[i] for i in ids]
    # return DataManager(sub_x, sub_y, sub_models, self.cfg)


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


def get_knn(x, tree, knn=500, min_points=20, radius=0.1, **_):
    distances, indices = tree.query(x, k=knn, distance_upper_bound=radius)
    mask = distances == np.inf
    nb_points = (~mask).sum(axis=1)
    gausspdf = (
        lambda x, mu, sigma: 1
        / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    )
    weights = gausspdf(distances, 0, radius / 3)
    indices[mask] = 0
    weights[mask] = 0
    weights[nb_points < min_points, :] = np.nan
    return indices, weights


def get_knn_mean(x, y, tree, **kw):
    indices, weights = get_knn(x, tree, **kw)
    avg = np.average(y[indices], axis=1, weights=weights)
    density = np.nansum(weights, axis=1)
    return avg, density


def get_knn_quantile(x, y, tree, qu, **kw):
    indices, weights = get_knn(x, tree, **kw)
    q = jax.vmap(weighted_quantile, in_axes=(0, 0, None))(y[indices], weights, qu)
    density = np.nansum(weights, axis=1)
    return q, density


def get_knn_smooth(xquery, logY, tree, knn=500, min_points=20, method='mean', **kw):
    if method == 'mean':
        Z, p = get_knn_mean(xquery, logY, knn=knn, min_points=min_points, tree=tree, **kw)
    elif method == 'quantile':
        assert 'qu' in kw
        Z, p = get_knn_quantile(xquery, logY, knn=knn, min_points=min_points, tree=tree, **kw)
    else:
        raise ValueError(f'Unknown method {method}')
    return Z, p


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                      --     heatmap methods     --


def heatmap(
    ax,
    Z,
    vmin=0,
    vmax=1,
    ticks=[],
    ticklabels=[],
    transform=None,
    text='',
    connector=False,
    connector_orientation='bottom',
    contours=3,
    colorbar=True,
    opacities=None,
    **_,
):
    cmap = plt.get_cmap('YlGnBu')
    cmap.set_bad(color='#EEEEEE')
    trans_data = ax.transData
    if transform is not None:
        trans_data = trans_data + transform
    if opacities is None:
        opacities = np.ones_like(Z)
    im = ax.imshow(
        Z.T,
        origin='lower',
        aspect=1,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=trans_data,
        interpolation='none',
        alpha=opacities.T,
    )
    # add contour
    if contours is not None:
        ax.contour(
            Z.T,
            levels=contours,
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


def smooth(x, y, model, rescale, ax, **kw):
    ninputs = model.n_inputs
    if ninputs == 1:
        smooth_1d(x, y, model, rescale, ax, **kw)
    elif ninputs == 2:
        smooth_2d(x, y, model, rescale, ax, **kw)
    elif ninputs == 3:
        smooth_3d(x, y, model, rescale, ax, **kw)
    else:
        raise NotImplementedError(f'Cannot plot {ninputs} inputs')


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


def model_ticks_and_labels(model, rescaler, xmax=1, desired_order=None):
    input_names = model.get_inverted_input_proteins()
    output_names = model.get_output_proteins()

    if desired_order is not None:
        reordered_input_names = [input_names[i] for i in desired_order]
        input_order=desired_order
    else:
        reordered_input_names = sorted(input_names)
        input_order = [input_names.index(i) for i in reordered_input_names]

    assert len(output_names) == (len(input_names) + 1)
    output_name = list(set(output_names) - set(input_names))[0]
    output_pos = output_names.index(output_name)
    unscaled_ticks = np.logspace(0, 12, 13)
    ticks = np.array(rescaler(unscaled_ticks))
    ticks = ticks[ticks < xmax]
    tlabels = [
        scformat.format("{:m}", x) if i > 1 else ''
        for i, x in enumerate(unscaled_ticks[: len(ticks)])
    ]
    return input_order, reordered_input_names, output_pos, output_name, ticks, tlabels


def smooth_2d(
    x,
    y,
    model,
    rescaler,
    ax,
    res=200,
    xmin=0,
    xmax=1,
    xslice=None,
    input_order=None,
    title=True,
    density_plot=False,
    density_as_alpha=False,
    density_threshold = 10,
    **kw,
):

    input_order, input_names, output_pos, output_name, ticks, tlabels = model_ticks_and_labels(
        model, rescaler, xmax=xmax, desired_order=input_order
    )

    y = y[:, output_pos]
    x = x[:, input_order]

    tree = cKDTree(x)
    xx = jnp.linspace(xmin, xmax, res)
    xygrid = jnp.array(np.meshgrid(xx, xx)).T.reshape(-1, 2)
    if x.shape[1] > 2:
        assert xslice.shape == (x.shape[1] - 2,)
        xquery = jnp.concatenate([xygrid, jnp.tile(xslice, (xygrid.shape[0], 1))], axis=1)
    else:
        xquery = xygrid
    z, p = get_knn_smooth(xquery, y, tree=tree, **kw)
    z = z.reshape(res, res)
    p = p.reshape(res, res)
    opacities = np.ones_like(z) if not density_as_alpha else np.minimum(p / density_threshold, 1.0)
    p = np.where(np.isnan(z), 1, p)
    if density_plot:
        z = p

    heatmap(ax, z, ticks=ticks, ticklabels=tlabels, opacities=opacities, **kw)
    if x.shape[1] > 2:
        ax.text(0.35, 0.9, f'{input_names[2]} ≈ {xslice[0]:.2f}', fontsize=8, transform=ax.transAxes)
    ax.set_xlabel(input_names[0])
    ax.set_ylabel(input_names[1])
    remove_spines(ax)

    ttle = None
    if title is True:
        ttle = f'{model.network.name}\n{output_name} smoothed mean'
    elif title is not None:
        ttle = title
    if ttle is not None:
        ax.set_title(ttle)


def smooth_3d(x, y, model, rescaler, ax=None, slices=np.linspace(0, 0.65, 4), axes=None, **kw):
    # we'll divide the third axis into slices
    # divide ax using make_axes_locatable
    if axes is None:
        assert ax is not None
        divider = make_axes_locatable(ax)
        axes = [ax]
        width = 1 / (len(slices) - 1)
        for i in range(len(slices) - 1):
            axes.append(divider.append_axes('top', size=width, pad=0.01))
        each_w, each_h = axes[0].get_position().size
        each_w /= len(slices)
        each_h /= len(slices)
        pos = ax.get_position()

        # resize all axes  so that they are square and fit in the original ax
        for i, a in enumerate(axes):
            a.set_position([pos.x0 + i * each_w + i * 0.05, pos.y0, each_w, each_h])
            # plot each slice
    for i, s in enumerate(slices):
        smooth_2d(x, y, model, rescaler, axes[i], xslice=np.array([slices[i]]), **kw)

    # resize all axes  so that they are square and fit in the original ax
    for i, a in enumerate(axes):
        if i > 0:
            a.set_yticks([])
            a.set_ylabel('')
        if i < len(axes) - 1:
            a.get_images()[0].colorbar.remove()
        a.set_xticks([])
        remove_spines(a)
        a.set_title('')

    # write title on top
    axes[0].set_title(
        f'{model.network.name}\n{model.get_output_proteins()[0]} smoothed mean', fontsize=8
    )


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
def model_plot(
    dman: DataManager, model_id: int, ax, kde=None, density_quantile_threshold=0.05, **kw
):
    model = dman.get_models()[model_id]
    x, y = dman.get_X()[model_id], dman.get_Y()[model_id]

    if kde is not False:
        if kde is None:
            kde = dman.get_kdes()[model_id]
        rng = jax.random.PRNGKey(0)
        subsample = optimal_density_subsample(
            x, kde, rng, quantile_threshold=density_quantile_threshold
        )
        x, y = x[subsample], y[subsample]

    smooth(x, y, model, dman.rescale, ax, **kw)


def eval_model_plot(
    params,
    dman,
    id,
    ax,
    npoints_eval=10000,
    quantile_range=[0.2, 0.8],
    key=jax.random.PRNGKey(0),
    xrange_eval=None,
    **kw,
):

    k_i, k_q = jax.random.split(key)
    if xrange_eval is None:
        xrange_eval = jnp.array([[0, 0], [1, 1]])

    model = dman.get_models()[id]
    jm = dman.get_jitted_models()[id]

    x = jax.random.uniform(
        k_i, (npoints_eval, model.n_inputs), minval=xrange_eval[0], maxval=xrange_eval[1]
    )
    quantiles = jax.random.uniform(
        k_q, (npoints_eval, model.n_outputs), minval=quantile_range[0], maxval=quantile_range[1]
    )
    keys = jax.random.split(key, npoints_eval)
    y = vmap(jm, in_axes=(None, 0, 0, 0))(params, x, quantiles, keys)

    xmin, xmax = jnp.min(x, axis=0)[0], jnp.max(x, axis=0)[0]

    smooth(x, y, model, dman.rescale, ax, xmin=xmin, xmax=xmax, **kw)


def eval_model_grid(
    params,
    dman,
    id,
    ax,
    quantile=0.5,
    key=jax.random.PRNGKey(0),
    xrange_eval=None,
    res=100,
    **kw,
):
    k_i, k_q = jax.random.split(key)
    if xrange_eval is None:
        xrange_eval = jnp.array([0, 1])
    model = dman.get_models()[id]
    jm = dman.get_jitted_models()[id]
    xx = jnp.linspace(xrange_eval[0], xrange_eval[1], res)
    xygrid = jnp.array(np.meshgrid(xx, xx)).T.reshape(-1, 2)
    input_order, input_names, output_pos, output_name, ticks, tlabels = model_ticks_and_labels(
        model, dman.rescale
    )
    keys = jax.random.split(key, xygrid.shape[0])
    xygrid = xygrid[:, input_order]
    y = vmap(jm, in_axes=(None, 0, None, 0))(
        params, xygrid, jnp.ones((model.n_outputs,)) * quantile, keys
    )
    z = y[:, output_pos].reshape(res, res)
    heatmap(ax, z, ticks=ticks, ticklabels=tlabels, **kw)
    ax.set_xlabel(input_names[0])
    ax.set_ylabel(input_names[1])
    remove_spines(ax)


def model_at_x(params, dman, id, key=jax.random.PRNGKey(0), quantile=None, **_):
    jm = dman.get_jitted_models()[id]
    x, y = dman.get_X()[id], dman.get_Y()[id]
    keys = jax.random.split(key, x.shape[0])

    if quantile is not None:
        Q = jnp.ones(y.shape) * quantile
    else:
        Q = jax.random.uniform(key, y.shape)

    yhat = vmap(jm, in_axes=(None, 0, 0, 0))(params, x, Q, keys)
    return x, y, yhat


def plot_model_at_x(params, dman, id, ax, **kw):
    x, y, yhat = model_at_x(params, dman, id, **kw)
    model = dman.get_models()[id]
    smooth(x, yhat, model, dman.rescale, ax, **kw)


def plot_model_diff(params, dman, id, ax, **kw):
    x, y, yhat = model_at_x(params, dman, id, **kw)
    model = dman.get_models()[id]
    err = jnp.abs(y - yhat)
    smooth(x, err, model, dman.rescale, ax, **kw)


def report(params, dman, id, suptitle='', **kw):
    fig, ax = mkfig(1, 2, size=(4, 4))
    model_plot(dman, id, ax[0], **kw)
    plot_model_at_x(params, dman, id, ax[1], **kw)
    # eval_model_plot(params, dman, id, ax[1], **kw)
    ax[0].set_title(f'Original data (mean)')
    ax[1].set_title(f'Predicted (mean)')
    model = dman.get_models()[id]
    fig.suptitle(f'{suptitle} {model.node_namespace}')
    fig.tight_layout()
    return fig, ax


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                  --     Fluo distribution plots     --


def get_bio_color(name, default='k'):
    import difflib

    colors = {'ebfp': '#529edb', 'eyfp': '#fbda73', 'mkate': '#f75a5a', 'neongreen': '#33f397'}
    colors['fitc'] = colors['neongreen']
    colors['pe_texas_red'] = colors['mkate']
    colors['pacific_blue'] = colors['ebfp']
    closest = difflib.get_close_matches(name.lower(), colors.keys(), n=1)
    if len(closest) == 0:
        color = default
    else:
        color = colors[closest[0]]
    return color


@jit
def loglog(x):
    return jnp.where(x > 1, jnp.log10(x), jnp.where(x < -1, -jnp.log10(-x), 0))


@jit
def inv_loglog(x):
    return jnp.where(x > 0, 10**x, jnp.where(x < 0, -(10**-x), 0))


def fluo_scatter(rawx, pnames, title=None, types=None, fname=None, logscale=True):
    fig, axes = plt.subplots(1, len(pnames), figsize=(1.25 * len(pnames), 10), sharey=True)
    if len(pnames) == 1:
        axes = [axes]
    if types is None:
        types = [''] * len(pnames)
    for xid, ax in enumerate(axes):
        color = get_bio_color(pnames[xid])
        xcoords = jax.random.normal(jax.random.PRNGKey(0), (rawx.shape[0],)) * 0.1
        ax.scatter(xcoords, rawx[:, xid], color=color, alpha=0.03, s=5, zorder=10, lw=0)
        if logscale:
            ax.set_yscale('symlog')
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(min(rawx.min(), 0), rawx.max())
        ax.set_xlabel(f'{pnames[xid]} {types[xid]}', rotation=0, labelpad=20, fontsize=10)
        remove_spines(ax)
        ax.set_xticks([])

    if title is not None:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    if fname is not None:
        fig.savefig(fname)


def density_plot_1d(
    x,
    sample_at,
    ax,
    color='k',
    label=None,
    ticks=None,
    ticks_labels=None,
    bw_method=None,
    x2=None,
    show_quantiles=[0.005, 0.995],
    **kw,
):
    if bw_method is None:
        bw_method = 0.01
    left_kde = gaussian_kde(x.T, bw_method=bw_method)
    left_densities = left_kde(sample_at.T)
    if x2 is not None:
        right_kde = gaussian_kde(x2.T, bw_method=bw_method)
        right_densities = right_kde(sample_at.T)
    else:
        x2 = x
        right_kde = left_kde
        right_densities = left_densities

    left_densities = (left_densities / left_densities.max()) * 0.4
    right_densities = (right_densities / right_densities.max()) * 0.4

    ax.plot(-left_densities, sample_at, color='k', alpha=1, lw=0.5)
    ax.plot(right_densities, sample_at, color='k', alpha=1, lw=0.5)

    if show_quantiles is not None:
        maxleft = sample_at[left_densities.argmax()]
        q1 = jnp.quantile(x, show_quantiles[0])
        q9 = jnp.quantile(x, show_quantiles[-1])
        ax.plot([-0.5, 0], [q1, q1], color=color, lw=1)
        ax.plot([-0.5, 0], [q9, q9], color=color, lw=1)
        # ax.plot([-0.5, 0], [maxleft, maxleft], color='k', lw=1)
        ax.fill_betweenx([q1, q9], -0.5, 0, color=color, alpha=0.1, lw=0)
        maxright = sample_at[right_densities.argmax()]
        q1 = jnp.quantile(x2, show_quantiles[0])
        q9 = jnp.quantile(x2, show_quantiles[-1])
        ax.plot([0, 0.5], [q1, q1], color=color, lw=1)
        ax.plot([0, 0.5], [q9, q9], color=color, lw=1)
        # ax.plot([0, 0.5], [maxright, maxright], color='k', lw=1)
        ax.fill_betweenx([q1, q9], 0, 0.5, color=color, alpha=0.1, lw=0)

    ax.fill_betweenx(sample_at, -left_densities, 0, color=color, alpha=1, lw=0)
    ax.fill_betweenx(sample_at, 0, right_densities, color=color, alpha=1, lw=0)
    ax.axvline(0, color='k', alpha=0.5, lw=0.5, dashes=(10, 10), dash_capstyle='round')
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 0.5)
    remove_axis_and_spines(ax)
    if label is not None:
        ax.set_xlabel(label, rotation=0, labelpad=20, fontsize=10)
    if ticks is not None:
        for t in ticks:
            ax.axhline(
                t,
                xmin=-0.2,
                xmax=1,
                c='#777777',
                linewidth=0.2,
                zorder=0,
                clip_on=False,
                alpha=1,
                dashes=(10, 20),
                dash_capstyle='round',
            )

        if ticks_labels is not None:
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticks_labels)
            ax.tick_params(axis='y', which='both', length=0, pad=30)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(8)
                tick.label.set_color('grey')


def fluo_densities(
    rawx, pnames, xmin=None, xmax=None, res=2000, title=None, types=None, logscale=True, **kw
):
    fig, axes = plt.subplots(1, len(pnames), figsize=(1.5 * len(pnames), 10))

    if logscale:
        X = loglog(rawx)
    else:
        X = rawx
    xmin = xmin if xmin is not None else np.floor(X.min())
    xmax = xmax if xmax is not None else np.ceil(X.max())

    ticks = np.arange(xmin, xmax + 1, 1)
    sample_at = jnp.linspace(xmin, xmax, res)
    if logscale:
        ylabels = [scformat.format("{:m}", x) for x in inv_loglog(ticks)]
    else:
        ylabels = [scformat.format("{:m}", x) for x in ticks]

    if types is None:
        types = [''] * len(pnames)
    for xid, ax in enumerate(axes):
        color = get_bio_color(pnames[xid], default='#AAAAAA')
        tlabels = ylabels if xid == 0 else None
        density_plot_1d(
            X[:, xid],
            sample_at,
            ax,
            color=color,
            label=f'{pnames[xid]} {types[xid]}',
            ticks=ticks,
            ticks_labels=tlabels,
            **kw,
        )
        ax.set_ylim(xmin, xmax)
    if title is not None:
        fig.suptitle(
            title,
            fontsize=10,
            y=0.85,
            x=0.45,
        )
    fig.tight_layout()


def model_fluo_distributions(dman, model_id, method='scatter', **kwargs):
    model = dman.get_models()[model_id]
    rawx = dman.get_raw_X()[model_id]
    rawy = dman.get_raw_Y()[model_id]
    input_names = model.get_inverted_input_proteins()
    reordered_input = sorted(input_names)
    output_names = model.get_output_proteins()
    output = list(set(output_names) - set(input_names))
    output_pos = output_names.index(output[0])
    if reordered_input != input_names:
        rawx = rawx[:, [input_names.index(i) for i in reordered_input]]
    rawx = jnp.hstack([rawx, rawy[:, output_pos][:, None]])
    pnames = reordered_input + output
    types = ['[in]'] * len(reordered_input) + ['[out]']
    if method == 'scatter':
        fluo_scatter(rawx, pnames, types=types, **kwargs)
    elif method == 'kde':
        fluo_densities(rawx, pnames, types=types, **kwargs)


##────────────────────────────────────────────────────────────────────────────}}}
