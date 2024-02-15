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

# from jax.scipy.stats import gaussian_kde
from scipy.stats import gaussian_kde
import itertools
import hashlib
import pickle
from matplotlib.ticker import FixedLocator, FuncFormatter
import matplotlib.ticker as ticker

from typing import Optional, Union, List, Tuple, Callable, Collection, Any

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


def network_data_check(x, y, network):
    n_inputs = network.get_nb_inputs()
    n_outputs = network.get_nb_outputs()
    assert x.shape[0] == y.shape[0], f"shape mismatch: {x.shape[0]} != {y.shape[0]}"
    assert x.shape[1] == n_inputs, f"input shape mismatch: {x.shape[1]} != {n_inputs}"
    assert y.shape[1] == n_outputs, f"output shape mismatch: {y.shape[1]} != {n_outputs}"
    outp = network.get_output_proteins()  # name of output proteins
    inp = network.get_inverted_input_proteins()  # name of input proteins
    in_pos = network.get_inverted_input_positions()
    assert len(inp) == n_inputs
    assert len(outp) == n_outputs
    assert len(in_pos) == n_inputs
    assert len(inp) == len(set(inp))
    assert all(iname in outp for iname in inp)
    for ipos, outpos in in_pos.items():
        assert inp[ipos] == outp[outpos]
        x_nonan_mask = ~np.isnan(x[:, ipos])
        y_nonan_mask = ~np.isnan(y[:, outpos])
        assert np.all(x_nonan_mask == y_nonan_mask)
        assert np.all(x[x_nonan_mask, ipos] == y[y_nonan_mask, outpos])


# @jit
# def optimal_density_subsample(X, kde, rng, quantile_threshold=0.1):
# EPSILON = 1e-12
# HIGH_DENSITIES_PENALTY = 1.00
# densities = kde.evaluate(X.T) + EPSILON
# threshold = jnp.quantile(densities, quantile_threshold)
# diceroll = jax.random.uniform(rng, shape=(len(densities),))
# selected = (densities < threshold) | (
# diceroll < (threshold / (densities * HIGH_DENSITIES_PENALTY))
# )
# return selected


# non-jax version
def optimal_density_subsample(X, kde, rng, quantile_threshold=0.1):
    EPSILON = 1e-12
    HIGH_DENSITIES_PENALTY = 1.00
    densities = kde.evaluate(X.T) + EPSILON
    threshold = np.quantile(densities, quantile_threshold)
    dice = np.random.RandomState(rng)
    diceroll = dice.uniform(0, 1, len(densities))
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


DEFAULT_DATA_CACHE_DIR = '../__cache/biocomp_densities_cache'
DEFAULT_DATA_CONFIG = {
    'data_min_value': 500,
    'data_max_value': 100000000.0,
    'data_log_offset': 3000.0,
    'data_log_factor': 100,
    'data_log_poly_threshold': 300,
    'data_log_poly_compression': 0.4,
    'data_sampling_kde_bw_method': 0.02,
    'data_sampling_max_density_samples': 4000,
    'data_sampling_density_quantile_threshold': 0.025,
    'data_sampling_coords_for_density_threshold': 0.15,
}


class DataManager:
    """The DataManager handles XP data and their matching compute stacks"""

    def __init__(
        self,
        X: list,
        Y: list,
        networks: list,
        data_cfg: dict = DEFAULT_DATA_CONFIG,
        data_chekcs=True,
        cache_location: Optional[Union[Path, str]] = DEFAULT_DATA_CACHE_DIR,
    ):
        self.data_cfg = data_cfg
        self.cache_dir = cache_location
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
        if data_chekcs:
            ut.logger.debug('Running data checks')
            for x, y, n in zip(self._X, self._Y, self._networks):
                network_data_check(x, y, n)
        ut.logger.info(f'Initialized a DataManager with {len(self._networks)} networks')

    def make_subset(self, network_ids):
        sub_x = [self._raw_X[i] for i in network_ids]
        sub_y = [self._raw_Y[i] for i in network_ids]
        sub_networks = [self._networks[i] for i in network_ids]
        return DataManager(sub_x, sub_y, sub_networks, self.data_cfg)

    def build_compute_stack(self, compute_cfg, **kwargs) -> ComputeStack:
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

    def compute_densities(self, max_chunk=50000):
        """Compute the densities at each data point in the dataset, for each sample"""

        ut.logger.debug('Computing densities')
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

    def get_batches(self, n_batches, batch_size, rng_key):
        if self._densities is None:
            self.compute_densities()
        xbatches, ybatches = _get_batches(
            self.get_X(),
            self.get_Y(),
            self.get_kdes(),
            self._densities,
            rng_key,
            batch_size,
            n_batches,
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
    def from_xps(
        cls, xplist, config=cmp.DEFAULT_COMPUTE_CONFIG, cache_location=DEFAULT_DATA_CACHE_DIR
    ):

        # build all networks and get all sample names, for each xp
        # networks, samples = zip(*[xp.build_networks(**kw) for xp in xplist])
        net_sample_pairs = []
        for xp in xplist:
            net_sample_pairs.append(
                ut.get_cache(lambda: xp.build_networks(**kw), f'{str(xp)}_net', cache_location)
            )

        networks, samples = zip(*net_sample_pairs)

        # get all X (independent vars) and Y (dependent vars) for each xp
        # X, Y = zip(*[xp.get_XY(n, s) for xp, n, s in zip(xplist, networks, samples)])
        XY_pairs = []
        for xp, n, s in zip(xplist, networks, samples):
            XY_pairs.append(
                ut.get_cache(lambda: xp.get_XY(n, s, **kw), f'{str(xp)}_XY', cache_location)
            )

        X, Y = zip(*XY_pairs)
        # get everything as a long concatenated list
        X, Y, networks = (
            list(itertools.chain(*X)),
            list(itertools.chain(*Y)),
            list(itertools.chain(*networks)),
        )
        return cls(X, Y, networks, config, cache_location=cache_location)


#                                                                            }}}
