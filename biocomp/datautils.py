# {{{                          --     imports     --
# ···············································································
import jax
import jax.numpy as jnp
from dataclasses import field
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
from dataclasses import dataclass

# from jax.scipy.stats import gaussian_kde
from scipy.stats import gaussian_kde
import itertools
import hashlib
import pickle
from matplotlib.ticker import FixedLocator, FuncFormatter
import matplotlib.ticker as ticker
from .network import Network

from typing import Optional, Union, List, Tuple, Callable, Collection, Any

ndArray = Union[np.ndarray, jnp.ndarray]
##────────────────────────────────────────────────────────────────────────────}}}

## {{{                       --     data rescaler     --
from pydantic import BaseModel, ValidationError, Field, field_validator

NumLike = Union[float, int, np.ndarray, jnp.ndarray]
NdArray = Union[np.ndarray, jnp.ndarray]

class DataRescaler(BaseModel):
    def fwd(self, x: NumLike) -> NdArray:
        raise NotImplementedError()
    def inv(self, y: NumLike) -> NdArray:
        raise NotImplementedError()


class ValueRange(BaseModel):
    min: float = 0
    max: float = 1


class CompressedSymLogRescaler(DataRescaler):
    """
    Rescale values from input_range to [0, 1], with tolerance for outside values.

    Uses a symmetric log transform to accept negative values (although they are not recommended).
    We use a log-poly-log transform where the part [-threshold, threshold] is a cubic polynomial,
    and the rest is log10.

    Also uses a low_end_compression factor to "squish" low values, which is useful for
    fluorescence data where low values are often noisy even though they just mean "no fluorescence".
    """

    # NOTE: with this, low values are squished symmetrically around input_range.min,
    #       meaning if they get much lower than input_range.min, they will again grow below 0 as fast as
    #       they grow in the positive direction. Should be fine for our datasets.
    #       Ideally, we should probably switch to making an asymmetric transform,
    #       where we still accept negative values, but they get exponentially squished towards 0.
    #       One option would be to use a sigmoid but I don't like the idea of abandoning log transforms
    #       because it allows for direct plotting and interpretation of this type of biological data.
    #       A sigmoid would also compress both ends of the range, which doesn't make sense for
    #       what we want. Keeping in mind that the neural net is not the only "consumer" of this
    #       rescale data - (we use it to resample and plot). Could use multiple rescalers, but I think
    #       there's a way to keep it simple and still have a good rescaler for all purposes.
    #

    input_range: ValueRange = Field(default_factory=lambda: ValueRange(min=500, max=1e8))
    low_end_compression: float = 100  # compression coefficient for low values
    poly_region_threshold: float = 300  # where we switch from log to poly
    poly_region_coef: float = 0.4  # how much we compress the poly part

    def __post_init__(self):
        self.__symlog = partial(
            ut.log_poly_log, threshold=self.poly_region_threshold, compression=self.poly_region_coef
        )
        self.__invsymlog = partial(
            ut.inverse_log_poly_log,
            threshold=self.poly_region_threshold,
            compression=self.poly_region_coef,
        )
        self.__log_start = self.__symlog(self.input_range.min / self.low_end_compression)
        self.__log_end = self.__symlog(self.input_range.max / self.low_end_compression)

    def fwd(self, x):
        xp = self.__symlog(1 + x / self.low_end_compression) - self.__log_start
        y = xp / (self.__log_end - self.__log_start)
        return y

    def inv(self, y):
        yp = y * (self.__log_end - self.__log_start) + self.__log_start
        ypinv = self.__invsymlog(yp)
        x = self.low_end_compression * (ypinv - 1)
        return x


# TODO, low priority: I'm thinking something like an exp (for negative values) to log transition:
# class CompressedExpLogRescaler:
# """
# Rescale values from input_range to [0, 1], with tolerance for outside values.
# Lower values will always be > 0, but higher values continue to increase logarithmically.
# """

# LOG10 = np.log(10)
# PLOG10 = scipy.special.lambertw(1/LOG10).real
# T10 = np.exp(PLOG10) - (np.log(PLOG10 / LOG10) / LOG10)
# XTHRESH = PLOG10 / LOG10
# YTHRESH = 10**(XTHRESH)

#     def explog10(self, x):
# return np.where(x > XTHRESH, np.log10(x)+T10, 10**x)

# def inv_explog10(self, y):
# return np.where(y < YTHRESH, 10**(y-T10), np.log10(y))


##────────────────────────────────────────────────────────────────────────────}}}
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

## {{{                           --     utils     --

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

##────────────────────────────────────────────────────────────────────────────}}}

## {{{                 --     batching & resampling     --

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

def sample_batches_direct(
    X: ndArray,
    Y: ndArray,
    batch_size: int,
    n_batches: int,
    kde: gaussian_kde,
    densities: ndArray,  # densities at each point in X
    rng,
    density_threshold_quantile=0.05,  # Compute the density threshold using the quantile of the density distribution
    density_threshold_coords=0.3,  # Compute the density threshold using the value of the density at this coordinate
):

    """
    Sample batches from X and Y, with a probability of including a point
    inversely proportional to the density at that point.
    This is done to avoid oversampling of high-density regions (usually untransfected cells),
    which can lead to overfitting.

    We use a threshold on the density distribution to decide which points to always include, and
    which to randomly sample:below this threshold density, points are always selected,
    above this density, points are selected with a probability inversely proportional
    to their density.

    2 ways of setting the threshold:
    - using a quantile of the density distribution (quantile_threshold)
        i.e. if quantile_threshold=0.05, any point that is in a neighborhood that's
        more dense than 95% of the data sees its probability of being selected reduced.
    - using the value of the density at a specific coordinate (density_coords)
    when both are set, the minimum of the two is used as the threshold.

    """

    assert X.shape[0] == Y.shape[0]
    assert densities.shape == (X.shape[0],)

    EPSILON = 1e-16
    HIGH_DENSITIES_PENALTY = 1.0

    threshold = np.inf
    if density_threshold_quantile is not None:
        threshold = np.quantile(densities + EPSILON, density_threshold_quantile)
    if density_threshold_coords is not None:
        midX = np.ones((X.shape[1],)) * density_threshold_coords
        density_at_midX = kde.evaluate(midX.T)
        if density_at_midX > 0:
            threshold = np.minimum(threshold, density_at_midX)

    # select batch_size * n_batches random points, weight by inverse of density
    # with numpy:
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

    # reshape to (n_batches, batch_size, n_features)
    Xbatches = Xsub.reshape((n_batches, batch_size, Xsub.shape[1]))
    Ybatches = Ysub.reshape((n_batches, batch_size, Ysub.shape[1]))
    return Xbatches, Ybatches


##────────────────────────────────────────────────────────────────────────────}}}

# {{{                       --     data manager     --
# ···············································································

# in general, could use a custom instancer that's similar to 
# hydra's: can use the same syntax: a _target_ field will indicate 
# it needs to be instantiated as an obj of the specified type,
# but a class can also pass a type that will be the default _target_


class ResamplingConfig(BaseModel):
    kde_bw_method: float = 0.02
    kde_samples: int = 4000
    density_chunksize: int = 50000
    density_threshold_quantile: float = 0.025
    density_threshold_coords: float = 0.15


class DataConfig(BaseModel):
    valid_raw_value_range: ValueRange = Field(default_factory=lambda: ValueRange(min=500, max=1e8))
    acceptable_out_of_range_fraction_in_raw_data: float = 0.05
    perform_data_checks: bool = True

    resampling: ResamplingConfig = Field(default_factory=ResamplingConfig)

    rescaler: DataRescaler = Field(default_factory=CompressedSymLogRescaler)


DEFAULT_DATA_CONFIG = DataConfig()
DEFAULT_DATA_CACHE_DIR = '../__cache/biocomp_densities_cache'

class DataManager:

    """
    The DataManager handles:
    - storage of, and access to stack data and the associated networks
    - rescaling of the data to a [0, 1] log space.
    - batching and density-based resampling (to avoid over-representation of high-density regions)
    - building and wrapping the matching compute stack
    """

    def __init__(
        self,
        X: list[ndArray],
        Y: list[ndArray],
        networks: list[Network],
        data_cfg: DataConfig = DEFAULT_DATA_CONFIG,
        cache_location: Optional[Union[Path, str]] = DEFAULT_DATA_CACHE_DIR,
    ):

        assert len(X) == len(Y) == len(networks)

        self.data_cfg = data_cfg
        self.cache_dir = cache_location
        self._raw_X = [np.array(x) for x in X]
        self._raw_Y = [np.array(y) for y in Y]

        # remove invalid values (NaNs, out of range)
        for i in range(len(self._raw_X)):
            invalid_at = np.isnan(self._raw_X[i]).any(axis=1)
            invalid_at = invalid_at | (np.isnan(self._raw_Y[i]).any(axis=1))
            invalid_at = invalid_at | (self._raw_X[i] < data_cfg.valid_raw_value_range.min).any(
                axis=1
            )
            invalid_at = invalid_at | (self._raw_Y[i] > data_cfg.valid_raw_value_range.max).any(
                axis=1
            )
            invalid_fraction = invalid_at.sum() / len(invalid_at)
            if invalid_fraction > data_cfg.acceptable_out_of_range_fraction_in_raw_data:
                raise ValueError(
                    f'Too many invalid values in {networks[i].name} raw data ({100*invalid_fraction:.2f}%)'
                )
            if invalid_fraction > 0.0:
                ut.logger.debug(
                    f'Removing {invalid_at.sum()} invalid points for net {i} ({100*invalid_fraction:.2f}%)'
                )
                self._raw_X[i] = self._raw_X[i][~invalid_at]
                self._raw_Y[i] = self._raw_Y[i][~invalid_at]

        self._networks = networks
        self._X = self.rescale(self._raw_X)
        self._Y = self.rescale(self._raw_Y)

        self.generate_kdes()
        self.compute_stack = None
        self._densities = None
        self.individual_compute_stacks = {}
        if self.data_cfg.perform_data_checks:
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

    def generate_kdes(self):
        """Generate KDEs to get the data densities of each sample"""
        ut.logger.debug('Generating KDEs for data density estimation')
        npoints = [min(x.shape[0], int(self.data_cfg.resampling.kde_samples)) for x in self._X]
        ut.logger.debug(f'Using {npoints} points for KDE estimation')
        xindices = [
            np.random.choice(x.shape[0], size=n, replace=False) for x, n in zip(self._X, npoints)
        ]
        self._kdes = [
            gaussian_kde(
                x[xi].T,
                bw_method=self.data_cfg.resampling.kde_bw_method,
            )
            for x, xi in zip(self._X, xindices)
        ]
        ut.logger.debug('Done generating KDEs')

    def compute_densities(self):
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
                allarr.append(
                    kde.evaluate(x[i : min(i + self.data_cfg.resampling.density_chunksize, n)].T)
                )
                i += self.data_cfg.resampling.density_chunksize
            res = np.concatenate(allarr)
            assert res.shape == (n,)
            return res

        self._densities = [
            ut.get_cache(lambda: compute_d(kde, x), get_signature(kde, x), self.cache_dir)
            for kde, x in tqdm(list(zip(self._kdes, self._X)), desc='computing densities')
        ]

        ut.logger.debug(f'Done computing {len(self._densities)} densities')

    def rescale(self, X):
        return [self.data_cfg.rescaler.fwd(x) for x in X]

    def unscale(self, X):
        return [self.data_cfg.rescaler.inv(x) for x in X]

    def get_batches(self, n_batches, batch_size, rng_key, concat_along_feature_axis=True):
        """
        Generate batches of data from the dataset, using the KDEs to resample the data
        and avoid over-representation of high-density regions.
        """
        if self._densities is None:
            self.compute_densities()
            assert self._densities is not None
        all_batches = [
            sample_batches_direct(
                x,
                y,
                batch_size,
                n_batches,
                kde,
                d,
                rng,
                density_threshold_quantile=self.data_cfg.resampling.density_threshold_quantile,
                density_threshold_coords=self.data_cfg.resampling.density_threshold_coords,
            )
            for x, y, kde, d, rng in tqdm(
                list(
                    zip(
                        self.get_X(),
                        self.get_Y(),
                        self.get_kdes(),
                        self._densities,
                        jax.random.split(rng_key, len(self._networks)),
                    )
                ),
                desc='generating batches',
            )
        ]
        xbatches, ybatches = zip(*all_batches)
        if concat_along_feature_axis:
            # concat along the feature axis (last dimension)
            # resulting shape is (N_BATCHES, BATCH_SIZE, N_MODELS * FEATURES)
            xbatches, ybatches = np.concatenate(tuple(xbatches), axis=2), np.concatenate(
                tuple(ybatches), axis=2
            )
            assert xbatches.shape == (n_batches, batch_size, sum([x.shape[1] for x in self._X]))
            assert ybatches.shape == (n_batches, batch_size, sum([y.shape[1] for y in self._Y]))
            assert xbatches.shape[2] == sum([n.get_nb_inputs() for n in self._networks])
            assert ybatches.shape[2] == sum([n.get_nb_outputs() for n in self._networks])

        return xbatches, ybatches

    def get_uniform_samples(self, rng_key, n_samples: int):
        xb, yb = self.get_batches(1, n_samples, rng_key, concat_along_feature_axis=False)
        return [x.squeeze() for x in xb], [y.squeeze() for y in yb]

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


#                                                                            }}}

