# {{{                          --     imports     --
# ···············································································
import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial
from jax import jit, vmap
import numpy as np
import pandas as pd
import biocomp as bc
from . import utils as ut
from pathlib import Path
import json
from . import defaults as dft
from . import nodes as nd
from . import compute as cmp
from .compute import ComputeStack
from tqdm import tqdm
import matplotlib.pyplot as plt
from jax.scipy.stats import gaussian_kde
import itertools
import hashlib
import pickle
from matplotlib.ticker import FixedLocator, FuncFormatter
import matplotlib.ticker as ticker

##────────────────────────────────────────────────────────────────────────────}}}


# ─────────────────────────────────────────────────────────────────────────────
#                            GENERAL PURPOSE TOOLS
# ───────────────────────────────────── ▼ ─────────────────────────────────────
### {{{              --     model retrieval and loss plots     --
def get_best_run_id(losses, smooth_window=20, return_smooth_losses=False):
    from scipy.ndimage import gaussian_filter1d

    smoothed_losses = [gaussian_filter1d(loss, smooth_window) for loss in losses]
    best_loss = np.argmin([loss[-1] for loss in smoothed_losses])
    if return_smooth_losses:
        return best_loss, smoothed_losses
    return best_loss


def losses_plot(losses, ax=None, smooth_window=200, runs=None):
    best_loss_id, smoothed_losses = get_best_run_id(
        losses, smooth_window=smooth_window, return_smooth_losses=True
    )
    if ax is None:
        fig, ax = mkfig(1, 1, (7, 5))
    for index, loss in enumerate(smoothed_losses):
        color = '#00A1D9' if index == best_loss_id else 'k'
        alpha = 1 if index == best_loss_id else 0.25
        size = 1.5 if index == best_loss_id else 0.5
        ax.plot(
            loss,
            color=color,
            alpha=alpha,
            linewidth=size,
            label='Best run' if index == best_loss_id else None,
        )
        ax.set_yscale('log')
    ax.set_xlabel('Batches seen')
    ax.set_ylabel('Loss')
    ax.set_ylim(np.min([np.min(s) for s in smoothed_losses]) * 0.7)
    n_losses = sum(len(l) > 1 for l in smoothed_losses)
    ax.set_title(f'Smoothed losses for {n_losses} runs')
    ax.legend()
    # add name of best run (centered)
    if runs is not None:
        best_run = runs[best_loss_id]
        ax.text(
            0.5,
            0.01,
            f'Best run: "{best_run.name}" with {smoothed_losses[best_loss_id][-1]:.1e}',
            transform=ax.transAxes,
            horizontalalignment='center',
            verticalalignment='bottom',
        )
    return best_loss_id


def retrieve_wandb_results(project_name, entity='jdisset', with_losses=True, **kw):
    import wandb
    import pickle
    from concurrent.futures import ThreadPoolExecutor

    wandb.login()
    api = wandb.Api()
    project_path = f"{entity}/{project_name}" if entity else project_name
    runs = api.runs(project_path, **kw)

    if with_losses:

        def get_loss_history(run):
            if 'loss' in run.summary and run.summary['loss'] is not None:
                history = run.scan_history(keys=['loss'], page_size=25000)
                losses = [row["loss"] for row in history]
                return np.array(losses)
            else:
                return np.array([np.inf])

        with ThreadPoolExecutor() as executor:
            full_losses = list(tqdm(executor.map(get_loss_history, runs), total=len(runs)))

        return runs, full_losses

    return runs


def get_wandb_trained_params(run, save_to=None):
    if save_to is None:
        save_to = Path(f'/tmp/biocomp_runs/{run.name}')
    save_to.mkdir(parents=True, exist_ok=True)
    param_file = run.file('latest_params.pkl').download(replace=True, root=save_to)
    trained_params = ut.load(param_file.name)
    shared_trained_params, local = trained_params.filter_by_tag('shared')
    compute_config_file = run.file('compute_config.json').download(replace=True, root=save_to)
    training_config_file = run.file('training_config.json').download(replace=True, root=save_to)
    compute_config = cmp.ComputeConfigManager.from_file(compute_config_file.name)
    with open(training_config_file.name, 'r') as f:
        training_config = json.load(f)
    shared_trained_params.set_read_only(True)
    return shared_trained_params, compute_config, training_config, local


def get_wandb_archive(run, save_path=None, filename=None):
    (
        shared_trained_params,
        compute_config,
        training_config,
        local,
    ) = get_wandb_trained_params(run, save_to=None)

    archive = {
        'shared_parameters': shared_trained_params,
        'local_parameters': local,
        'compute_config': compute_config,
        'training_config': training_config,
        'metadata': run.metadata,
    }

    archive_path = None
    if save_path is not None:
        if filename is None:
            date_started = run.metadata['startedAt'].split('T')[0]
            filename = f'{date_started}_{run.name}.pkl'
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        archive_path = save_path / filename
        with open(archive_path, 'wb') as f:
            pickle.dump(archive, f)
            ut.logger.info(f'Saved training archive to {archive_path}')

    return archive, archive_path


##────────────────────────────────────────────────────────────────────────────}}}

# ─────────────────────────────────────────────────────────────────────────────
#                         DATA MANAGEMENT AND BATCHING
# ───────────────────────────────────── ▼ ─────────────────────────────────────
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


# @partial(jit, static_argnames=('batch_size', 'n_batches', 'quantile_threshold', 'density_coords'))
def sample_batches_direct(
    X, Y, batch_size, n_batches, kde, densities, rng, quantile_threshold=0.05, density_coords=0.3
):
    assert X.shape[0] == Y.shape[0]
    EPSILON = 1e-16
    HIGH_DENSITIES_PENALTY = 1.0

    # select batch_size * n_batches random points, weight by inverse of density
    threshold = np.quantile(densities + EPSILON, quantile_threshold)
    midX = np.ones((X.shape[1],)) * density_coords
    density_at_midX = kde.evaluate(midX.T)
    if density_at_midX > 0:
        threshold = np.minimum(threshold, density_at_midX)

    # with jax:
    # selection_proba = jnp.minimum(1.0, (threshold / (densities * HIGH_DENSITIES_PENALTY + EPSILON)))
    # indices = jax.random.choice(rng, X.shape[0], shape=(batch_size * n_batches,), p=selection_proba)
    # Xsub = np.take(X, indices, axis=0)
    # Ysub = np.take(Y, indices, axis=0)

    # or with numpy:
    seed = jax.random.randint(rng, (1,), minval=0, maxval=2**28)[0]
    rng = np.random.RandomState(seed)
    selection_proba = np.minimum(1.0, (threshold / (densities * HIGH_DENSITIES_PENALTY + EPSILON)))
    selection_proba /= np.sum(selection_proba)
    try:
        indices = rng.choice(
            X.shape[0], size=(batch_size * n_batches,), p=selection_proba, replace=True
        )
    except ValueError:
        n_nans = np.sum(np.isnan(selection_proba))
        ut.logger.warning(
            f'Sampling failed, {n_nans} / {len(selection_proba)} NaNs in selection_proba.'
        )
        selection_proba[np.isnan(selection_proba)] = 0.0
        selection_proba /= np.sum(selection_proba)
        indices = rng.choice(
            X.shape[0], size=(batch_size * n_batches,), p=selection_proba, replace=True
        )

    Xsub = X[indices]
    Ysub = Y[indices]

    Xbatches = Xsub.reshape((n_batches, batch_size, Xsub.shape[1]))
    Ybatches = Ysub.reshape((n_batches, batch_size, Ysub.shape[1]))
    return Xbatches, Ybatches


# @partial(jit, static_argnames=('batch_size', 'n_batches', 'density_quantile_threshold', 'density_coords'))
def _get_batches(
    X,
    Y,
    kdes,
    densities,
    rng_key,
    batch_size,
    n_batches,
    density_quantile_threshold,
    density_coords,
):
    all_batches = [
        sample_batches_direct(
            x,
            y,
            batch_size,
            n_batches,
            kde,
            d,
            rng,
            quantile_threshold=density_quantile_threshold,
            density_coords=density_coords,
        )
        for x, y, kde, d, rng in tqdm(
            list(zip(X, Y, kdes, densities, jax.random.split(rng_key, len(X)))),
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


DEFAULT_DENSITIES_CACHE_DIR = '../__cache/biocomp_densities_cache'


def tr(x, offset=3e3, maxv=5e7, factor=50, threshold=300, compression=0.4):
    loff = ut.log_poly_log(offset / factor, threshold=threshold, compression=compression)
    lmv = ut.log_poly_log(maxv / factor, threshold=threshold, compression=compression)
    xp = ut.log_poly_log(1 + x / factor, threshold=threshold, compression=compression) - loff
    y = xp / (lmv - loff)
    return y


def inv_tr(y, offset=3e3, maxv=5e7, factor=50, threshold=300, compression=0.4):
    loff = ut.log_poly_log(offset / factor, threshold=threshold, compression=compression)
    lmv = ut.log_poly_log(maxv / factor, threshold=threshold, compression=compression)
    yp = y * (lmv - loff) + loff
    ypinv = ut.inverse_log_poly_log(yp, threshold=threshold, compression=compression)
    x = factor * (ypinv - 1)
    return x


class DataManager:
    """The DataManager handles XP data and their matching compute stacks"""

    def __init__(
        self,
        X: list,
        Y: list,
        networks: list,
        data_cfg: dict,
    ):
        self.data_cfg = data_cfg
        self._raw_X = [np.array(x) for x in X]
        self._raw_Y = [np.array(y) for y in Y]

        # remove invalid values:
        for i in range(len(self._raw_X)):
            invalid_at = np.isnan(self._raw_X[i]).any(axis=1)
            invalid_at = invalid_at | (np.isnan(self._raw_Y[i]).any(axis=1))
            invalid_at = invalid_at | (self._raw_X[i] < data_cfg['data_min_value']).any(axis=1)
            invalid_at = invalid_at | (self._raw_Y[i] > data_cfg['data_max_value']).any(axis=1)
            percentnan = 100.0 * invalid_at.sum() / len(invalid_at)
            if percentnan > 0.0:
                ut.logger.debug(
                    f'Removing {invalid_at.sum()} invalid points for net {i} ({percentnan:.2f} %)'
                )
                self._raw_X[i] = self._raw_X[i][~invalid_at]
                self._raw_Y[i] = self._raw_Y[i][~invalid_at]

        self._networks = networks
        self._X = self.rescale(self._raw_X)
        self._Y = self.rescale(self._raw_Y)

        # MAX_VAL = 1.25
        # assert max([x.max() for x in self._X]) < MAX_VAL, max([x.max() for x in self._X])
        # assert max([y.max() for y in self._Y]) < MAX_VAL, max([y.max() for y in self._Y])
        self.gen_kdes()
        self.compute_stack = None
        self._densities = None
        self.individual_compute_stacks = {}
        # data_checks(X, Y, models)
        ut.logger.info(f'Initialized a DataManager with {len(self._networks)} networks')

    def make_subset(self, network_ids):
        sub_x = [self._raw_X[i] for i in network_ids]
        sub_y = [self._raw_Y[i] for i in network_ids]
        sub_networks = [self._networks[i] for i in network_ids]
        return DataManager(sub_x, sub_y, sub_networks, self.data_cfg)

    def build_compute_stack(self, compute_cfg, **kwargs):
        """Build/Get the composite compute stack of all networks"""
        self.compute_stack = ComputeStack(self._networks)
        self.compute_stack.build(compute_cfg, **kwargs)
        return self.compute_stack

    def get_compute_stack(self):
        if self.compute_stack is None:
            raise ValueError('Compute stack not built yet.')
        return self.compute_stack

    def get_individual_compute_stack(self, network_id):
        """Build/Get a compute stack for a single network"""
        if network_id not in self.individual_compute_stacks:
            self.individual_compute_stacks[network_id] = self.compute_stack.make_subset(
                [network_id]
            )
        # actually returns a tuple of (stack, get_param_subset)
        return self.individual_compute_stacks[network_id]

    def gen_kdes(self, bw=None, max_n=None):
        """Generate KDEs to get the data densities of each sample"""

        ut.logger.debug('Generating KDEs for data density estimation')

        if bw is None:
            bw = self.data_cfg['data_sampling_kde_bw_method']
        if max_n is None:
            max_n = int(self.data_cfg['data_sampling_max_density_samples'])

        # just grap max_n for each self._X using numpy
        self._kde_bw = bw

        npoints = [min(x.shape[0], max_n) for x in self._X]
        ut.logger.debug(f'Using {npoints} points for KDE estimation')
        xindices = [
            np.random.choice(x.shape[0], size=n, replace=False) for x, n in zip(self._X, npoints)
        ]
        self._kdes = [
            gaussian_kde(
                x[xi].T,
                bw_method=bw,
            )
            for x, xi in zip(self._X, xindices)
        ]
        ut.logger.debug('Done generating KDEs')

    def compute_densities(self, max_chunk=50000, cache_dir=None):
        """Compute the densities at each data point in the dataset, for each sample"""
        import hashlib
        from pathlib import Path
        import base64

        ut.logger.debug('Computing densities')

        if cache_dir is None:
            if 'densities_cache_location' in self.data_cfg:
                self.cache_dir = self.data_cfg['densities_cache_location']
            else:
                self.cache_dir = DEFAULT_DENSITIES_CACHE_DIR

        ut.logger.debug(f'Using cache dir {self.cache_dir}')

        def get_signature(kde, x):
            n = x.shape[0]
            stepsize = max(n // 100, 1)
            xsig = f'{x.shape}_{x[::stepsize]}'
            ksig = f'{kde.factor:.20f}_{kde.n}_{kde.d}'
            return f'{xsig}_{ksig}'

        def compute_d(kde, x):
            # cut in chunks to avoid memory issues
            n = x.shape[0]
            allarr = []
            i = 0
            while i < n:
                allarr.append(kde.evaluate(x[i : min(i + max_chunk, n)].T))
                i += max_chunk
            res = np.concatenate(allarr)
            assert res.shape == (n,)

            return res

        self._densities = [
            ut.get_cache(lambda: compute_d(kde, x), get_signature(kde, x), self.cache_dir)
            for kde, x in tqdm(list(zip(self._kdes, self._X)), desc='computing densities')
        ]

        ut.logger.debug(f'Done computing {len(self._densities)} densities')

    def rescale(self, X):
        return [
            tr(
                x,
                offset=self.data_cfg['data_log_offset'],
                maxv=self.data_cfg['data_max_value'],
                factor=self.data_cfg['data_log_factor'],
                threshold=self.data_cfg['data_log_poly_threshold'],
                compression=self.data_cfg['data_log_poly_compression'],
            )
            for x in X
        ]

    def unscale(self, X):
        return [
            inv_tr(
                x,
                offset=self.data_cfg['data_log_offset'],
                maxv=self.data_cfg['data_max_value'],
                factor=self.data_cfg['data_log_factor'],
                threshold=self.data_cfg['data_log_poly_threshold'],
                compression=self.data_cfg['data_log_poly_compression'],
            )
            for x in X
        ]

    def get_batches(self, rng_key):
        if self._densities is None:
            self.compute_densities()
        xbatches, ybatches = _get_batches(
            self.get_X(),
            self.get_Y(),
            self.get_kdes(),
            self._densities,
            rng_key,
            self.data_cfg['batch_size'],
            self.data_cfg['n_batches'],
            self.data_cfg['data_sampling_density_quantile_threshold'],
            self.data_cfg['data_sampling_coords_for_density_threshold'],
        )
        assert xbatches.shape[2] == sum([n.get_nb_inputs() for n in self._networks])
        assert ybatches.shape[2] == sum([n.get_nb_outputs() for n in self._networks])
        return xbatches, ybatches

    def get_uniform_samples(self, rng_key, n_samples=10000):
        if self._densities is None:
            self.compute_densities()
        all_b = [
            sample_batches_direct(
                x,
                y,
                n_samples,
                1,
                kde,
                d,
                rng,
                quantile_threshold=self.data_cfg['data_sampling_density_quantile_threshold'],
                density_coords=self.data_cfg['data_sampling_coords_for_density_threshold'],
            )
            for x, y, kde, d, rng in zip(
                self.get_X(),
                self.get_Y(),
                self.get_kdes(),
                self._densities,
                jax.random.split(rng_key, len(self._networks)),
            )
        ]
        X, Y = zip(*all_b)
        X = [x.squeeze() for x in X]
        Y = [y.squeeze() for y in Y]
        return X, Y

    def get_networks(self):
        return self._networks

    def get_network(self, i):
        return self._networks[i]

    def get_kdes(self):
        return self._kdes

    def get_X(self):
        return self._X

    def get_Y(self):
        return self._Y

    def get_raw_X(self):
        return self._raw_X

    def get_raw_Y(self):
        return self._raw_Y

    @classmethod
    def from_xps(cls, xplist, config=cmp.DEFAULT_COMPUTE_CONFIG, **kw):
        network_cache_location = None
        if 'network_cache_location' in config:
            network_cache_location = Path(config['network_cache_location'])

        # build all networks and get all sample names, for each xp
        # networks, samples = zip(*[xp.build_networks(**kw) for xp in xplist])
        net_sample_pairs = []
        for xp in xplist:
            net_sample_pairs.append(
                ut.get_cache(
                    lambda: xp.build_networks(**kw), f'{str(xp)}_net', network_cache_location
                )
            )

        networks, samples = zip(*net_sample_pairs)

        # get all X (independent vars) and Y (dependent vars) for each xp
        # X, Y = zip(*[xp.get_XY(n, s) for xp, n, s in zip(xplist, networks, samples)])
        XY_pairs = []
        for xp, n, s in zip(xplist, networks, samples):
            XY_pairs.append(
                ut.get_cache(lambda: xp.get_XY(n, s, **kw), f'{str(xp)}_XY', network_cache_location)
            )

        X, Y = zip(*XY_pairs)
        # get everything as a long concatenated list
        X, Y, networks = (
            list(itertools.chain(*X)),
            list(itertools.chain(*Y)),
            list(itertools.chain(*networks)),
        )
        return cls(X, Y, networks, config)


#                                                                            }}}


# ─────────────────────────────────────────────────────────────────────────────
#                            NEIGHBORHOOD BASED TOOLS
# ───────────────────────────────────── ▼ ─────────────────────────────────────
### {{{                    --     plot styling tools     --
from mpl_toolkits.axes_grid1 import make_axes_locatable


def mkfig(rows, cols, size=(7, 7), **kw):
    fig, ax = plt.subplots(rows, cols, figsize=(cols * size[0], rows * size[1]), **kw)
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
### {{{              --     knn and spatial partitionning    --
from scipy.spatial import cKDTree


@jax.jit
def weighted_quantile(data, weights, qu):
    ix = jnp.argsort(data)
    data = data[ix]
    weights = weights[ix]
    cdf = (jnp.cumsum(weights) - 0.5 * weights) / jnp.sum(weights)
    return jnp.interp(qu, cdf, data)


# TODO: jax version using grid space partitionning
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
    secondticks=[],
    transform=None,
    text='',
    connector=False,
    connector_orientation='bottom',
    contours=3,
    colorbar=True,
    opacities=None,
    get_cbar_ticks=None,
    **_,
):
    cmap = plt.get_cmap('YlGnBu')
    cmap.set_bad(color='#EEEEEE')
    trans_data = ax.transData
    if transform is not None:
        trans_data = trans_data + transform
    if opacities is None:
        opacities = np.ones_like(Z)

    if vmin is None:
        vmin = np.nanmin(Z)
    if vmax is None:
        vmax = np.nanmax(Z)

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
        # secondticks:
        if len(secondticks) > 0:
            print(f'secondticks = {secondticks}')
            sc_secondticks = secondticks * Z.shape[0]
            ax.set_xticks(sc_secondticks, minor=True)
            ax.set_yticks(sc_secondticks, minor=True)

    # colorbar
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=6)
        # no cbar spines
        for spine in cbar.ax.spines.values():
            spine.set_visible(False)

        if get_cbar_ticks is not None:
            # get ticks every 0.1 decade
            unscaled_ticks = np.geomspace(inv_tr(vmin), inv_tr(vmax), 5, endpoint=True)
            ticks = np.array(tr(unscaled_ticks))
            print(f'unscaled_ticks : {unscaled_ticks}')
            print(f'vmin = {vmin}, vmax = {vmax}, ticks = {ticks}')
            ticks = ticks[ticks < vmax]
            ticks = ticks[ticks > vmin]
            ticklabels = [scformat.format("{:m}", inv_tr(x)) for x in ticks]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(ticklabels)

        # # use same ticks if present
        # if len(ticks) > 0:
        # valid = ticks >= vmin
        # diff = len(ticks)
        # ticks = ticks[valid]
        # diff -= len(ticks)
        # cbar.set_ticks(ticks)
        # cbar.set_ticklabels(ticklabels[diff:])

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


def scatter(x, y, network, *args, **kw):
    ninputs = network.get_nb_inputs()
    if ninputs == 2:
        return scatter_1d(x, y, network, *args, **kw)
    if ninputs == 2:
        return scatter_2d(x, y, network, *args, **kw)
    if ninputs == 3:
        return scatter_3d_interactive(x, y, network, *args, **kw)
    else:
        raise NotImplementedError(f'Cannot scater plot {ninputs} inputs')


from labellines import labelLine, labelLines


def smooth_line_slices(
    x, y, network, rescale, slices, input_order, axes=None, ax=None, xmin=0, xmax=1, **kwargs
):
    # slices is a list of list of slice values. (max 2 dimensions)
    assert len(slices) <= 2, 'Can only slice maximum 2 dimensions'
    outerslices = slices[1] if len(slices) > 1 else []
    innerslices = slices[0] if len(slices) > 0 else []

    (
        input_order,
        input_names,
        output_pos,
        output_name,
        ticks,
        tlabels,
        secondticks,
    ) = network_ticks_and_labels(network, rescale, xmax=xmax, desired_order=input_order)

    if axes is None:
        assert ax is not None
        nplots = len(outerslices)
        axes = [ax] + [ax.figure.add_subplot(nplots, 1, i + 1) for i in range(nplots - 1)]

    print(f'input names = {input_names}')
    assert len(axes) == len(outerslices), 'Number of axes must match number of outer slices'
    print(f'len(axes) = {len(axes)}')
    print(f'outerslices: {input_names[input_order[2]]} at {outerslices}')
    print(f'innerslices: {input_names[input_order[1]]} at {innerslices}')

    for i, outsl in enumerate(outerslices):
        iax = axes[i]
        for insl in innerslices:
            smooth_line_plot(
                x,
                y,
                network,
                rescale,
                ax=iax,
                slice_at=[insl, outsl],
                input_order=input_order,
                xmax=x.max(),
                label=f'{input_names[input_order[1]]} ≈ {insl}%',
                **kwargs,
            )
        iax.text(
            0.5,
            0.75,
            f'{input_names[input_order[2]]}={int(outsl*100)}%',
            transform=iax.transAxes,
            ha='center',
            va='center',
        )
        # labelLines(iax.get_lines(), zorder=2.5)


def smooth(x, y, network, rescale, ax, **kw):
    ninputs = network.get_nb_inputs()
    if ninputs == 1:
        scatter_1d(x, y, network, rescale, ax, **kw)
    elif ninputs == 2:
        smooth_2d(x, y, network, rescale, ax, **kw)
    elif ninputs == 3:
        smooth_3d(x, y, network, rescale, ax, **kw)
    else:
        raise NotImplementedError(f'Cannot plot {ninputs} inputs')


def smooth_1d(x, y, network, rescaler, ax, res=500, xmin=0, xmax=1, input_order=None):
    tree = cKDTree(x)

    (
        input_order,
        input_names,
        output_pos,
        output_name,
        ticks,
        tlabels,
        secondticks,
    ) = network_ticks_and_labels(network, rescaler, xmax=xmax, desired_order=input_order)

    assert len(input_names) == 1

    y = y[:, output_pos]
    x = x[:, input_order]

    unscaled_ticks = np.logspace(0, 12, 13)
    ticks = np.array(rescaler(unscaled_ticks))
    ticks = ticks[ticks < xmax]
    tlabels = [
        scformat.format("{:m}", x) if i > 1 else ''
        for i, x in enumerate(unscaled_ticks[: len(ticks)])
    ]

    xx = jnp.linspace(xmin, xmax, res).reshape(-1, 1)
    z = get_knn_mean(xx, y, tree)
    if len(z) == 0:
        return
    try:
        ax.plot(xx, z, color='k')
    except ValueError as e:
        ut.logger.warning(f'Could not plot: {e}.\nxx: {xx}\nz: {z}')
        pass
    try:
        zq1 = get_knn_quantile(xx, y, qu=0.1, tree=tree)
        zq9 = get_knn_quantile(xx, y, qu=0.9, tree=tree)
        ax.fill_between(xx[:, 0], zq1, zq9, alpha=0.25, color='k')
    except ValueError as e:
        ut.logger.warning(f'Could not fill between: {e}.\nzq1: {zq1}\nzq9: {zq9}')
        pass

    ax.set_title(f'{network.name}\nSmoothed mean and [0.1 - 0.9] quantile')
    ax.set_xlabel(input_names[0])
    ax.set_ylabel(output_name)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tlabels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tlabels)


def network_ticks_and_labels(network, rescaler, xmin=0, xmax=1, desired_order=None):
    input_names = network.get_inverted_input_proteins()
    output_names = network.get_output_proteins()

    if desired_order is not None:
        assert len(desired_order) == len(input_names), f'Wrong number of inputs: {desired_order}'
        reordered_input_names = [input_names[i] for i in desired_order]
        input_order = desired_order
    else:
        reordered_input_names = sorted(input_names)
        input_order = [input_names.index(i) for i in reordered_input_names]

    assert len(output_names) == (len(input_names) + 1)
    output_name = list(set(output_names) - set(input_names))[0]
    output_pos = output_names.index(output_name)

    unscaled_ticks = np.logspace(0, 12, 13)
    ticks = np.array(rescaler(unscaled_ticks))
    valid_ticks = (ticks <= xmax) & (ticks >= xmin)
    # valid_ticks = np.ones_like(ticks, dtype=bool)
    ticks = ticks[valid_ticks]
    tlabels = [scformat.format("{:m}", x) for x in unscaled_ticks[valid_ticks]]
    # secondary ticks every 0.1 decade
    unscaled_secondary_ticks = np.geomspace(inv_tr(xmin), inv_tr(xmax), 5, endpoint=True)
    secondary_ticks = np.array(tr(unscaled_secondary_ticks))
    secondary_valid_ticks = (secondary_ticks <= xmax) & (secondary_ticks >= xmin)
    secondary_ticks = secondary_ticks[secondary_valid_ticks]

    return (
        input_order,
        reordered_input_names,
        output_pos,
        output_name,
        ticks,
        tlabels,
        secondary_ticks,
    )


import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.offline as pyo

import plotly.graph_objects as go

import plotly.graph_objs as go
import plotly.subplots as sp
from IPython.display import display

import plotly.graph_objects as go
import plotly.offline as pyo
import numpy as np


def scatter_3d_interactive(
    x,
    y,
    network,
    rescaler,
    xmin=0,
    xmax=1,
    title=None,
    input_order=None,
    key=jax.random.PRNGKey(0),
    size=10,
    colorbar=True,
    lw=0.01,
    filename=None,
    **kw,
):
    (
        input_order,
        input_names,
        output_pos,
        output_name,
        ticks,
        ticklabels,
        secondticks,
    ) = network_ticks_and_labels(network, rescaler, xmax=xmax, desired_order=input_order)

    random_order = jax.random.permutation(key, len(x))
    y = y[random_order, output_pos]
    x = x[random_order][:, input_order]

    fig = go.Figure()

    scatter = go.Scatter3d(
        x=x[:, 0],
        y=x[:, 1],
        z=x[:, 2],
        mode='markers',
        marker=dict(size=size, color=y, colorscale='YlGnBu', line=dict(color='black', width=lw)),
    )

    fig.add_trace(scatter)

    fig.update_layout(
        scene=dict(
            xaxis_title=input_names[0],
            yaxis_title=input_names[1],
            zaxis_title=input_names[2],
            xaxis=dict(showspikes=False, showbackground=False, tickvals=ticks, ticktext=ticklabels),
            yaxis=dict(showspikes=False, showbackground=False, tickvals=ticks, ticktext=ticklabels),
            zaxis=dict(showspikes=False, showbackground=False, tickvals=ticks, ticktext=ticklabels),
        ),
        width=1000,
        height=800,
    )

    if colorbar:
        cbar_trace = go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode='markers',
            marker=dict(
                size=0,
                cmin=y.min(),
                cmax=y.max(),
                colorscale='YlGnBu',
                showscale=True,
                colorbar=dict(title=output_name, tickvals=ticks, ticktext=ticklabels),
            ),
        )

        fig.add_trace(cbar_trace)

    ttle = None
    if title is True:
        ttle = f'{network.name}\n{output_name} smoothed mean'
    elif title is not None:
        ttle = title
    if ttle is not None:
        fig.update_layout(title=ttle)

    if filename is None:
        return pyo.plot(fig, auto_open=True)
    else:
        return pyo.plot(fig, filename=filename, auto_open=False)


def scatter_3d(
    x,
    y,
    network,
    rescaler,
    fig,
    n_views,
    xmin=0,
    xmax=1,
    title=None,
    input_order=None,
    key=jax.random.PRNGKey(0),
    size=10,
    colorbar=True,
    lw=0.1,
    **kw,
):
    (
        input_order,
        input_names,
        output_pos,
        output_name,
        ticks,
        ticklabels,
        secondticks,
    ) = network_ticks_and_labels(network, rescaler, xmax=xmax, desired_order=input_order)

    cmap = plt.get_cmap('YlGnBu')
    random_order = jax.random.permutation(key, len(x))
    y = y[random_order, output_pos]
    x = x[random_order][:, input_order]

    azim_values = np.linspace(0, 270, n_views)

    for i, azim in enumerate(azim_values):
        ax = fig.add_subplot(1, n_views, i + 1, projection='3d')
        sc = ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y, cmap=cmap, s=size, lw=lw, edgecolor='k')
        ax.set_xlabel(input_names[0])
        ax.set_ylabel(input_names[1])
        ax.set_zlabel(input_names[2])

        if len(ticks) > 0:
            sc_ticks = ticks
            ax.set_xticks(sc_ticks)
            ax.set_xticklabels(ticklabels)
            ax.set_yticks(sc_ticks)
            ax.set_yticklabels(ticklabels)
            ax.set_zticks(sc_ticks)
            ax.set_zticklabels(ticklabels)

        # if colorbar and i == n_views - 1:  # Only show colorbar on the last plot
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("top", size="4%", pad=0.05)
        # cbar = plt.colorbar(sc, cax=cax)
        # cbar.ax.tick_params(labelsize=6)
        # for spine in cbar.ax.spines.values():
        # spine.set_visible(False)

        # if len(ticks) > 0:
        # valid = ticks >= xmin
        # diff = len(ticks)
        # ticks = ticks[valid]
        # diff -= len(ticks)
        # cbar.set_ticks(ticks)
        # cbar.set_ticklabels(ticklabels[diff:])

        ttle = None

        if title is True:
            ttle = f'{network.name}\n{output_name} smoothed mean'
        elif title is not None:
            ttle = title
        if ttle is not None:
            ax.set_title(ttle)

        # Rotate the axes
        ax.view_init(elev=10, azim=azim)


def scatter_2d(
    x,
    y,
    network,
    rescaler,
    ax,
    xmin=0,
    xmax=1,
    title=None,
    input_order=None,
    key=jax.random.PRNGKey(0),
    size=10,
    colorbar=True,
    lw=0.1,
    **kw,
):
    (
        input_order,
        input_names,
        output_pos,
        output_name,
        ticks,
        ticklabels,
        secondticks,
    ) = network_ticks_and_labels(network, rescaler, xmax=xmax, desired_order=input_order)

    cmap = plt.get_cmap('YlGnBu')
    random_order = jax.random.permutation(key, len(x))
    y = y[random_order, output_pos]
    x = x[random_order][:, input_order]

    sc = ax.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap, s=size, lw=lw, edgecolor='k')

    ax.set_xlabel(input_names[0])
    ax.set_ylabel(input_names[1])

    # remove right and top spine
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # ticks:
    if len(ticks) > 0:
        # rescale ticks to image coordinates (they are btwn 0 and 1 to start)
        sc_ticks = ticks
        ax.set_xticks(sc_ticks)
        ax.set_xticklabels(ticklabels)
        ax.set_yticks(sc_ticks)
        ax.set_yticklabels(ticklabels)

    # colorbar
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.05)
        cbar = plt.colorbar(sc, cax=cax)
        cbar.ax.tick_params(labelsize=6)
        # no cbar spines
        for spine in cbar.ax.spines.values():
            spine.set_visible(False)

        # use same ticks if present
        if len(ticks) > 0:
            valid = ticks >= xmin
            diff = len(ticks)
            ticks = ticks[valid]
            diff -= len(ticks)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(ticklabels[diff:])

    ttle = None

    if title is True:
        ttle = f'{network.name}\n{output_name} smoothed mean'
    elif title is not None:
        ttle = title
    if ttle is not None:
        ax.set_title(ttle)


def scatter_1d(
    x,
    y,
    network,
    rescaler,
    ax,
    xmin=0,
    xmax=1,
    title=None,
    input_order=None,
    max_n=20000,
    s=10,
    alpha=0.1,
    lw=0,
    key=jax.random.PRNGKey(0),
    use_y_as_x=True,
    **kw,
):
    (
        input_order,
        input_names,
        output_pos,
        output_name,
        ticks,
        ticklabels,
        secondticks,
    ) = network_ticks_and_labels(network, rescaler, xmax=xmax, desired_order=input_order)

    assert input_order == [0]

    random_order = jax.random.permutation(key, min(max_n, len(x)))
    if use_y_as_x:
        other_pos = 1 - output_pos
        x = y[random_order, other_pos].squeeze()
    else:
        x = x[random_order].squeeze()

    y = y[random_order, output_pos]

    sc = ax.scatter(x, y, s=s, lw=lw, edgecolor='k', alpha=alpha, color='k')

    ax.set_xlabel(input_names[0])
    ax.set_ylabel(output_name)

    # remove right and top spine
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # ticks:
    if len(ticks) > 0:
        sc_ticks = ticks
        ax.set_xticks(sc_ticks)
        ax.set_xticklabels(ticklabels)
        ax.set_yticks(sc_ticks)
        ax.set_yticklabels(ticklabels)

    ttle = None

    if title is True:
        ttle = f'{network.name}'
    elif title is not None:
        ttle = title
    if ttle is not None:
        ax.set_title(ttle)


def smooth_2d(
    x,
    y,
    network,
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
    density_threshold=10,
    use_y_as_x=False,
    **kw,
):
    (
        input_order,
        input_names,
        output_pos,
        output_name,
        ticks,
        tlabels,
        secondticks,
    ) = network_ticks_and_labels(network, rescaler, xmax=xmax, desired_order=input_order)

    if use_y_as_x:
        output_names = network.get_output_proteins()
        # find input_names in output_names
        xind = [output_names.index(i) for i in input_names]
        x = y[:, xind]
    else:
        x = x[:, input_order]

    y = y[:, output_pos]

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
    heatmap(
        ax, z, ticks=ticks, ticklabels=tlabels, secondticks=secondticks, opacities=opacities, **kw
    )
    if x.shape[1] > 2:
        ax.text(
            0.35, 0.9, f'{input_names[2]} ≈ {xslice[0]:.2f}', fontsize=8, transform=ax.transAxes
        )
    ax.set_xlabel(input_names[0])
    ax.set_ylabel(input_names[1])
    remove_spines(ax)
    ttle = None
    if title is True:
        ttle = f'{network.name}\n{output_name} smoothed mean'
    elif title is not None:
        ttle = title
    if ttle is not None:
        ax.set_title(ttle)


def smooth_3d(x, y, network, rescaler, ax=None, slices=np.linspace(0, 0.65, 4), axes=None, **kw):
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

        def get_cbar_ticks(vmin, vmax):
            (
                in_order,
                in_names,
                out_pos,
                out_name,
                vticks,
                vtlabels,
                secondticks,
            ) = network_ticks_and_labels(network, rescaler, xmin=vmin, xmax=vmax)
            return vticks, vtlabels

        smooth_2d(
            x,
            y,
            network,
            rescaler,
            axes[i],
            xslice=np.array([slices[i]]),
            get_cbar_ticks=get_cbar_ticks,
            **kw,
        )

    # resize all axes  so that they are square and fit in the original ax
    for i, a in enumerate(axes):
        if i > 0:
            a.set_yticks([])
            a.set_ylabel('')
        # if i < len(axes) - 1:
        # a.get_images()[0].colorbar.remove()
        a.set_xticks([])
        remove_spines(a)
        a.set_title('')

    # write title on top
    axes[0].set_title(
        f'{network.name}\n{network.get_output_proteins()[0]} smoothed mean', fontsize=8
    )


def smooth_line_plot(
    x,
    y,
    network,
    rescaler,
    ax,
    res=200,
    xmin=0,
    xmax=1,
    slice_at=None,
    input_order=None,
    title=True,
    label=None,
    color=None,
    lw=1,
    **kw,
):
    (
        input_order,
        input_names,
        output_pos,
        output_name,
        ticks,
        tlabels,
        secondticks,
    ) = network_ticks_and_labels(network, rescaler, xmax=xmax, desired_order=input_order)
    y = y[:, output_pos]
    x = x[:, input_order]
    tree = cKDTree(x)

    xquery = np.linspace(xmin, xmax, res).reshape(-1, 1)
    if slice_at is None:
        slice_at = np.array([])
    slice_at = np.array(slice_at)

    if x.shape[1] > 1:
        assert slice_at.shape == (x.shape[1] - 1,)
        xquery = np.concatenate([xquery, np.tile(slice_at, (xquery.shape[0], 1))], axis=1)

    z, _ = get_knn_smooth(xquery, y, tree=tree, **kw)

    if label is None:
        label = ''
        for i, s in enumerate(slice_at):
            label += f'{input_names[i+1]} ≈ {s:.2f}\n'

    ax.plot(xquery[:, 0], z, label=label, color=color, lw=lw)
    ax.set_xlabel(input_names[0])
    ax.set_ylabel(output_name)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ticks:
    if len(ticks) > 0:
        # rescale ticks to image coordinates (they are btwn 0 and 1 to start)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tlabels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(tlabels)

    ttle = None
    if title is True:
        ttle = f'{network.name}\n{output_name} smoothed mean'
    elif title is not None:
        ttle = title
    if ttle is not None:
        ax.set_title(ttle)


def setup_clean_fig(title):
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
    fig, ax = setup_clean_fig(title)
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


def network_plot(
    dman: DataManager,
    network_id: int,
    *args,
    kde=None,
    density_quantile_threshold=0.05,
    use_xy=None,
    method='smooth',
    **kw,
):
    network = dman.get_networks()[network_id]
    if use_xy is None:
        x, y = dman.get_X()[network_id], dman.get_Y()[network_id]
    else:
        x, y = use_xy

    if kde is not False:
        if kde is None:
            kde = dman.get_kdes()[network_id]
        rng = jax.random.PRNGKey(0)
        subsample = optimal_density_subsample(
            x, kde, rng, quantile_threshold=density_quantile_threshold
        )
        x, y = x[subsample], y[subsample]

    if method == 'smooth':
        return smooth(x, y, network, dman.rescale, *args, **kw)
    elif method == 'scatter':
        return scatter(x, y, network, dman.rescale, *args, **kw)
    elif method == 'smooth_lines':
        return smooth_line_slices(x, y, network, dman.rescale, *args, **kw)


def eval_network_plot(
    params,
    dman,
    id,
    ax,
    npoints_eval=20000,
    quantile_range=[0.2, 0.8],
    key=jax.random.PRNGKey(0),
    xrange_eval=None,
    **kw,
):
    k_i, k_q = jax.random.split(key)
    if xrange_eval is None:
        xrange_eval = jnp.array([[0, 0], [1, 1]])

    network = dman.get_networks()[id]
    jm = jit(dman.get_individual_compute_stack(id).apply)

    x = jax.random.uniform(
        k_i, (npoints_eval, network.get_nb_inputs()), minval=xrange_eval[0], maxval=xrange_eval[1]
    )
    quantiles = jax.random.uniform(
        k_q, (npoints_eval, network.n_outputs), minval=quantile_range[0], maxval=quantile_range[1]
    )
    keys = jax.random.split(key, npoints_eval)
    y = vmap(jm, in_axes=(None, 0, 0, 0))(params, x, quantiles, keys)

    xmin, xmax = jnp.min(x, axis=0)[0], jnp.max(x, axis=0)[0]

    smooth(x, y, network, dman.rescale, ax, xmin=xmin, xmax=xmax, **kw)


def get_stack(dman, net_id, params):
    stack, pf = dman.get_individual_compute_stack(net_id)
    p = pf(params)
    return stack, p


def eval_network_on_grid(
    params,
    network,
    stack,
    ax,
    rescale=tr,
    key=jax.random.PRNGKey(0),
    xrange_eval=(0, 1),
    n_repeats=10,
    quantile_range=(0.2, 0.8),
    res=100,
    input_order=None,
    **kw,
):
    jm = jit(stack.apply)

    (
        input_order,
        input_names,
        output_pos,
        output_name,
        ticks,
        tlabels,
        secondticks,
    ) = network_ticks_and_labels(network, rescale, xmax=xrange_eval[1], desired_order=input_order)

    k_i, k_q = jax.random.split(key)
    if xrange_eval is None:
        xrange_eval = jnp.array([0, 1])

    xx = jnp.linspace(xrange_eval[0], xrange_eval[1], res)
    x = jnp.array(np.meshgrid(xx, xx)).T.reshape(-1, 2)

    def compute(k):
        quantiles = jax.random.uniform(
            k,
            (len(x), network.get_nb_outputs()),
            minval=quantile_range[0],
            maxval=quantile_range[1],
        )
        keys = jax.random.split(k, len(x))
        y, _ = vmap(jm, in_axes=(None, 0, 0, 0))(params, x, quantiles, keys)
        return y

    keys = jax.random.split(key, n_repeats)
    all_y = vmap(compute)(keys)
    y_mean = jnp.mean(all_y, axis=0)

    z = y_mean[:, output_pos]
    z = z.reshape(res, res)

    heatmap(ax, z, ticks=ticks, ticklabels=tlabels, **kw)
    ax.set_xlabel(input_names[0])
    ax.set_ylabel(input_names[1])
    remove_spines(ax)


def eval_model_grid(
    params,
    dman,
    id,
    ax,
    **kw,
):
    network = dman.get_networks()[id]
    stack, p = get_stack(dman, id, params)
    return eval_network_on_grid(params, network, stack, ax, rescale=dman.rescale, **kw)


def model_at_x(params, dman: DataManager, id, key=jax.random.PRNGKey(0), quantile=None, **_):
    stack, p = get_stack(dman, id, params)

    x, y = dman.get_X()[id], dman.get_Y()[id]
    keys = jax.random.split(key, x.shape[0])

    if quantile is not None:
        Q = jnp.ones(y.shape) * quantile
    else:
        Q = jax.random.uniform(key, y.shape)

    yhat, _ = jit(vmap(stack.apply, in_axes=(None, 0, 0, 0)))(p, x, Q, keys)

    return x, y, yhat


def plot_model_at_x(params, dman, id, ax, **kw):
    x, _, yhat = model_at_x(params, dman, id, **kw)
    net = dman.get_networks()[id]
    smooth(x, yhat, net, dman.rescale, ax, **kw)


def plot_model_diff(params, dman, id, ax, **kw):
    x, y, yhat = model_at_x(params, dman, id, **kw)
    net = dman.get_networks()[id]
    err = jnp.abs(y - yhat)
    smooth(x, err, net, dman.rescale, ax, **kw)


def report(params, dman, id, suptitle='', use_x_y_yhat=None, **kw):
    if use_x_y_yhat is not None:
        x, y, yhat = use_x_y_yhat
        assert len(x) == len(y), 'x and y must have the same length'
        assert y.shape == yhat.shape, 'y and yhat must have the same shape'
        ndim = x.shape[1]
        if ndim <= 2:
            fig, ax = mkfig(1, 2, size=(4, 4))
            network_plot(dman, id, ax[0], use_xy=(x, y), kde=False, **kw)
            network_plot(dman, id, ax[1], use_xy=(x, yhat), kde=False, **kw)
            ax[0].set_title(f'Original data (mean)')
            ax[1].set_title(f'Predicted (mean)')
        elif ndim == 3:
            fig, axes = mkfig(2, 4, size=(4, 4))
            contours = np.linspace(0, 0.8, 5)
            top_row_axes = axes[0, :]
            bottom_row_axes = axes[1, :]
            slices = (np.linspace(0.1, 0.8, 4),)
            network_plot(
                dman,
                id,
                ax=None,
                axes=top_row_axes,
                contours=contours,
                slices=np.linspace(0.1, 0.8, 4),
                use_xy=(x, y),
                **kw,
            )
            network_plot(
                dman,
                id,
                ax=None,
                axes=bottom_row_axes,
                contours=contours,
                slices=np.linspace(0.1, 0.8, 4),
                use_xy=(x, yhat),
                **kw,
            )
            for ax in axes.flatten():
                ax.set_title('')
            axes[0, 0].set_title(f'Original data (mean)')
            axes[1, 0].set_title(f'Predicted (mean)')
        else:
            raise ValueError(f'ndim={ndim} not supported')
    else:
        fig, ax = mkfig(1, 2, size=(4, 4))
        network_plot(dman, id, ax[0], **kw)
        plot_model_at_x(params, dman, id, ax[1], **kw)
        ax[0].set_title(f'Original data (mean)')
        ax[1].set_title(f'Predicted (mean)')

    network = dman.get_networks()[id]
    fig.suptitle(f'{suptitle} {network.name}')
    fig.tight_layout()
    return fig


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                  --     Fluo distribution plots     --


def powers_of_ten(xmin, xmax, skip_10=False):
    bounds = np.array([xmin, xmax])
    logbounds = np.sign(bounds) * np.floor(
        np.maximum(np.log10(np.maximum(np.abs(bounds), 0.1)), 0)
    ).astype(int)
    powers = np.arange(logbounds[0], logbounds[1] + 1)
    if skip_10:
        powers = np.delete(powers, np.where(np.abs(powers) == 1))
    return np.power(10, np.abs(powers)) * np.sign(powers)


def format_powers(x, _):
    abs_x = abs(x)
    if abs_x < 1000:
        if x == int(x):
            return f'{x:.0f}'  # No decimal point
        else:
            return f'{x:.1f}'  # Up to 1 decimal point
    else:
        E = int(np.log10(abs_x))
        if x == int(x):
            return r'${0:.0f}e{1}$'.format(x // 10**E, E)
        else:
            return r'${0:.1f}e{1}$'.format(x / 10**E, E)


class PowerFormatter(ticker.Formatter):
    def __init__(self, values, skip10=False):
        self.values = values
        self.skip10 = skip10

    def __call__(self, x, pos):
        v = self.values[pos]
        if self.skip10 and abs(v) == 10:
            return ''
        return format_powers(v, None)


def setup_symlog(
    ax, xlims=None, ylims=None, linthresh=200, linscale=0.4, skip10=True, margins=0.05
):
    # ticks at all powers of 10:
    # tr = partial(symlog, linthresh=linthresh, linscale=linscale)
    # invtr = partial(inverse_symlog, linthresh=linthresh, linscale=linscale)
    tr = partial(ut.log_poly_log, threshold=linthresh, compression=linscale)
    invtr = partial(ut.inverse_log_poly_log, threshold=linthresh, compression=linscale)
    xlims_tr, ylims_tr = None, None
    if xlims is not None:
        xlims_tr = tr(np.asarray(xlims))
        xp10 = powers_of_ten(*xlims)
        xlims_margin = xlims_tr + np.array([-1, 1]) * margins * np.diff(xlims_tr)
        ax.set_xlim(xlims_margin)
        ax.set_xticks(tr(xp10))
        ax.xaxis.set_major_formatter(PowerFormatter(xp10, skip10=skip10))
    if ylims is not None:
        ylims_tr = tr(np.asarray(ylims))
        yp10 = powers_of_ten(*ylims)
        ylims_margin = ylims_tr + np.array([-1, 1]) * margins * np.diff(ylims_tr)
        ax.set_ylim(ylims_margin)
        ax.set_yticks(tr(yp10))
        ax.yaxis.set_major_formatter(PowerFormatter(yp10, skip10=skip10))
    return tr, invtr, xlims_tr, ylims_tr


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


def fluo_scatter(
    rawx,
    pnames,
    xmin=0,
    xmax=None,
    title=None,
    types=None,
    fname=None,
    logscale=True,
    alpha=0.1,
    maxn=50000,
    s=2,
    **_,
):
    fig, axes = plt.subplots(1, len(pnames), figsize=(1.25 * len(pnames), 10), sharey=True)

    if len(pnames) == 1:
        axes = [axes]

    if types is None:
        types = [''] * len(pnames)

    if xmin is None:
        xmin = rawx.min()
    if xmax is None:
        xmax = rawx.max()

    X = rawx.copy()
    if len(X) > maxn:
        X = X[np.random.choice(len(X), maxn, replace=False)]

    tr = lambda x: x
    itr = tr
    for xid, ax in enumerate(axes):
        color = get_bio_color(pnames[xid])
        xcoords = np.random.normal(0, 0.1, (X.shape[0],))
        if logscale:
            tr, itr, _, ytr = setup_symlog(ax, None, ylims=[xmin, xmax])
        else:
            ax.set_ylim(xmin, xmax)
        ax.scatter(xcoords, tr(X[:, xid]), color=color, alpha=alpha, s=s, zorder=10, lw=0)

        remove_spines(ax)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xlabel(f'{pnames[xid]} {types[xid]}', rotation=0, labelpad=20, fontsize=10)
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
    rawx, pnames, xmin=None, xmax=None, res=1000, title=None, types=None, logscale=False, **kw
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
            y=0.95,
            x=0.45,
        )
    fig.tight_layout()


def get_best_run_id(losses, smooth_window=20, return_smooth_losses=False):
    from scipy.ndimage import gaussian_filter1d

    smoothed_losses = [gaussian_filter1d(loss, smooth_window) for loss in losses]
    best_loss = np.argmin([loss[-1] for loss in smoothed_losses])
    if return_smooth_losses:
        return best_loss, smoothed_losses
    return best_loss


def losses_plot(losses, ax=None, smooth_window=200, runs=None):
    best_loss_id, smoothed_losses = get_best_run_id(
        losses, smooth_window=smooth_window, return_smooth_losses=True
    )
    if ax is None:
        fig, ax = mkfig(1, 1, (7, 5))
    for index, loss in enumerate(smoothed_losses):
        color = '#00A1D9' if index == best_loss_id else 'k'
        alpha = 1 if index == best_loss_id else 0.25
        size = 1.5 if index == best_loss_id else 0.5
        ax.plot(
            loss,
            color=color,
            alpha=alpha,
            linewidth=size,
            label='Best run' if index == best_loss_id else None,
        )
        ax.set_yscale('log')
    ax.set_xlabel('Batches seen')
    ax.set_ylabel('Loss')
    ax.set_ylim(np.min([np.min(s) for s in smoothed_losses]) * 0.7)
    n_losses = sum(len(l) > 1 for l in smoothed_losses)
    ax.set_title(f'Smoothed losses for {n_losses} runs')
    ax.legend()
    # add name of best run (centered)
    if runs is not None:
        best_run = runs[best_loss_id]
        ax.text(
            0.5,
            0.01,
            f'Best run: "{best_run.name}" with {smoothed_losses[best_loss_id][-1]:.1e}',
            transform=ax.transAxes,
            horizontalalignment='center',
            verticalalignment='bottom',
        )
    return best_loss_id


def retrieve_wandb_results(project_name, entity='jdisset', with_losses=True, **kw):
    import wandb
    import pickle
    from concurrent.futures import ThreadPoolExecutor

    wandb.login()
    api = wandb.Api()
    project_path = f"{entity}/{project_name}" if entity else project_name
    runs = api.runs(project_path, **kw)

    if with_losses:

        def get_loss_history(run):
            if 'loss' in run.summary and run.summary['loss'] is not None:
                history = run.scan_history(keys=['loss'], page_size=25000)
                losses = [row["loss"] for row in history]
                return np.array(losses)
            else:
                return np.array([np.inf])

        with ThreadPoolExecutor() as executor:
            full_losses = list(tqdm(executor.map(get_loss_history, runs), total=len(runs)))

        return runs, full_losses

    return runs


def get_wandb_trained_params(run, save_to=None):
    if save_to is None:
        save_to = Path(f'/tmp/biocomp_runs/{run.name}')
    save_to.mkdir(parents=True, exist_ok=True)
    param_file = run.file('latest_params.pkl').download(replace=True, root=save_to)
    trained_params = ut.load(param_file.name)
    shared_trained_params, local = trained_params.filter_by_tag('shared')
    compute_config_file = run.file('compute_config.json').download(replace=True, root=save_to)
    training_config_file = run.file('training_config.json').download(replace=True, root=save_to)
    compute_config = cmp.ComputeConfigManager.from_file(compute_config_file.name)
    with open(training_config_file.name, 'r') as f:
        training_config = json.load(f)
    shared_trained_params.set_read_only(True)
    return shared_trained_params, compute_config, training_config, local


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

