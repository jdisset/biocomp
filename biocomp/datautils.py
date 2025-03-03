# {{{                          --     imports     --
# ···············································································
import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial
import numpy as np
from . import utils as ut
from .utils import ArbitraryModel
from pathlib import Path
from .compute import ComputeStack
from tqdm import tqdm
from scipy.stats import gaussian_kde
from multiprocessing import Value, Pool
from ctypes import c_int
import itertools
from .network import Network
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Union, List, Tuple, Callable, Collection, Any

from biocomp.logging_config import get_logger

logger = get_logger(__name__)

ndArray = Union[np.ndarray, jnp.ndarray]
##────────────────────────────────────────────────────────────────────────────}}}

## {{{                       --     data rescaler     --

NumLike = Union[float, int, np.ndarray, jnp.ndarray]
NdArray = Union[np.ndarray, jnp.ndarray]


class DataRescaler(ArbitraryModel):
    def fwd(self, x):
        raise NotImplementedError

    def inv(self, y):
        raise NotImplementedError


class ValueRange(BaseModel):
    min: float = 0
    max: float = 1


class IdentityRescaler(DataRescaler):
    def fwd(self, x):
        return x

    def inv(self, y):
        return y


class LogPlusOneRescaler(DataRescaler):
    def fwd(self, x):
        return np.log10(x + 1)

    def inv(self, y):
        return 10**y - 1


class LogPolyLogRescaler(DataRescaler):
    poly_region_threshold: float = 300  # where we switch from log to poly
    poly_region_coef: float = 0.4  # how much we compress the poly part

    def model_post_init(self, *_):
        self.__symlog = partial(
            ut.log_poly_log, threshold=self.poly_region_threshold, compression=self.poly_region_coef
        )
        self.__invsymlog = partial(
            ut.inverse_log_poly_log,
            threshold=self.poly_region_threshold,
            compression=self.poly_region_coef,
        )

    def fwd(self, x):
        return self.__symlog(x)

    def inv(self, y):
        return self.__invsymlog(y)


class LogisticLogRescaler(DataRescaler):
    """
    Rescale values from input_range to [0, 1], switching from a logistic to a logarithmic function at T.
    """

    max_val: float = 1e8  # point at which f = 1
    thresh: float = 100  # transition point
    k: float = 0.1  # steepness of the logistic function
    lshift: float = 1  # shift in the logarithmic part

    def model_post_init(self, *_):
        self._A, self._B = self.calculate_log_constants()

    def calculate_log_constants(self):
        A = 1 / np.log(self.max_val - self.thresh + self.lshift)
        B = 1 / (1 + np.exp(-self.k * (self.thresh - self.thresh))) - A * np.log(self.lshift)
        return A, B

    def logistic(self, x, T, k):
        return 1 / (1 + np.exp(-k * (x - T)))

    def fwd(self, x: np.ndarray) -> np.ndarray:
        result = np.where(
            x < self.thresh,
            self.logistic(x, self.thresh, self.k),
            self._A * np.log(x - self.thresh + self.lshift) + self._B,
        )
        return result

    def inv(self, y: np.ndarray) -> np.ndarray:
        # Inverse function
        inv_log = (y - self._B) / self._A
        result = np.where(
            y < self.logistic(self.thresh, self.thresh, self.k),
            self.thresh - (1 / self.k) * np.log((1 / y) - 1),
            np.exp(inv_log) + self.thresh - self.lshift,
        )
        return result


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

    def model_post_init(self, *_):
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
# def explog10(self, x):
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
    kde,
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
        logger.warning(
            f"Sampling failed, {n_nans} / {len(selection_proba)} NaNs in selection_proba."
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


def fast_batch_sampling(
    args: tuple,
) -> Tuple[ndArray, ndArray]:
    """
    Optimized version of sample_batches_direct that precomputes probabilities
    and uses vectorized operations.
    """
    (
        X,
        Y,
        batch_size,
        n_batches,
        densities,
        kde_points,
        kde_bw,
        rng_key,
        density_threshold_quantile,
        density_threshold_coords,
    ) = args

    # Compute selection probabilities
    EPSILON = 1e-16
    HIGH_DENSITIES_PENALTY = 1.0

    threshold = np.inf
    if density_threshold_quantile is not None:
        threshold = np.quantile(densities + EPSILON, density_threshold_quantile)
    if density_threshold_coords is not None:
        kde = gaussian_kde(kde_points, bw_method=kde_bw)
        midX = np.ones((X.shape[1],)) * density_threshold_coords
        density_at_midX = kde.evaluate(midX.T)
        if density_at_midX > 0:
            threshold = np.minimum(threshold, density_at_midX)

    selection_proba = np.minimum(1.0, (threshold / (densities * HIGH_DENSITIES_PENALTY + EPSILON)))
    selection_proba[np.isnan(selection_proba)] = 0.0
    selection_proba /= np.sum(selection_proba)

    # Generate all indices at once
    total_indices = batch_size * n_batches
    seed = jax.random.randint(rng_key, (1,), minval=0, maxval=2**28)[0]
    rng = np.random.RandomState(seed)

    indices = rng.choice(len(selection_proba), size=total_indices, p=selection_proba, replace=True)

    # Vectorized selection and reshaping
    Xbatches = X[indices].reshape(n_batches, batch_size, X.shape[1])
    Ybatches = Y[indices].reshape(n_batches, batch_size, Y.shape[1])

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
    acceptable_out_of_range_fraction_in_raw_data: float = 0.04
    perform_data_checks: bool = True
    resampling: ResamplingConfig = Field(default_factory=ResamplingConfig)
    rescaler: DataRescaler = Field(default_factory=CompressedSymLogRescaler)


DEFAULT_DATA_CONFIG = DataConfig(rescaler=CompressedSymLogRescaler())
DEFAULT_DATA_CACHE_DIR = "../__cache/biocomp_densities_cache"


def worker_init(counter):
    """Make counter available to the worker processes"""
    global progress_counter
    progress_counter = counter


def compute_single_density(args):
    """
    Helper function to compute density for a single file/sample.
    Recreates the KDE in the worker process.
    """
    kde_points, kde_bw, x, chunksize, cache_dir, signature = args

    def compute_d(kde_points, kde_bw, x, chunksize):
        # Recreate the KDE in this process
        kde = gaussian_kde(kde_points, bw_method=kde_bw)

        # cut in chunks to avoid memory issues
        n = x.shape[0]
        allarr = []
        i = 0
        while i < n:
            allarr.append(kde.evaluate(x[i : min(i + chunksize, n)].T))
            i += chunksize
        res = np.concatenate(allarr)
        assert res.shape == (n,)
        return res

    return ut.get_cache(lambda: compute_d(kde_points, kde_bw, x, chunksize), signature, cache_dir)


def compute_selection_probabilities(
    X: ndArray,
    densities: ndArray,
    density_threshold_quantile: float,
    density_threshold_coords: float,
    kde_points: ndArray,
    kde_bw: float,
) -> ndArray:
    """
    Precompute selection probabilities for batch sampling.
    Vectorized operations for better performance.
    """
    EPSILON = 1e-16
    HIGH_DENSITIES_PENALTY = 1.0

    threshold = np.inf
    if density_threshold_quantile is not None:
        threshold = np.quantile(densities + EPSILON, density_threshold_quantile)
    if density_threshold_coords is not None:
        # Create KDE only once if needed
        kde = gaussian_kde(kde_points, bw_method=kde_bw)
        midX = np.ones((X.shape[1],)) * density_threshold_coords
        density_at_midX = kde.evaluate(midX.T)
        if density_at_midX > 0:
            threshold = np.minimum(threshold, density_at_midX)

    # Vectorized probability computation
    selection_proba = np.minimum(1.0, (threshold / (densities * HIGH_DENSITIES_PENALTY + EPSILON)))
    selection_proba[np.isnan(selection_proba)] = 0.0
    selection_proba /= np.sum(selection_proba)

    return selection_proba


def generate_batch_indices(
    n_samples: int, selection_proba: ndArray, rng: np.random.RandomState
) -> ndArray:
    """
    Generate batch indices using vectorized operations.
    """
    return rng.choice(len(selection_proba), size=n_samples, p=selection_proba, replace=True)


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
            invalid_at = invalid_at | (self._raw_X[i] > data_cfg.valid_raw_value_range.max).any(
                axis=1
            )
            invalid_at = invalid_at | (self._raw_Y[i] < data_cfg.valid_raw_value_range.min).any(
                axis=1
            )
            invalid_at = invalid_at | (self._raw_Y[i] > data_cfg.valid_raw_value_range.max).any(
                axis=1
            )
            invalid_fraction = invalid_at.sum() / len(invalid_at)
            if invalid_fraction > data_cfg.acceptable_out_of_range_fraction_in_raw_data:
                raise ValueError(
                    f"Too many invalid values in {networks[i].name} raw data ({100*invalid_fraction:.2f}%)"
                )
            if invalid_fraction > 0.0:
                logger.debug(
                    f"Removing {invalid_at.sum()} invalid points for net {i} ({100*invalid_fraction:.2f}%)"
                )
                self._raw_X[i] = self._raw_X[i][~invalid_at]
                self._raw_Y[i] = self._raw_Y[i][~invalid_at]

        self._networks = networks
        self._X = self.rescale(self._raw_X)
        self._Y = self.rescale(self._raw_Y)

        # Generate KDE sample points upfront (replaces generate_kdes)
        self._kde_points = []
        self._kde_bws = []
        for x in self._X:
            npoints = min(x.shape[0], int(self.data_cfg.resampling.kde_samples))
            xindices = np.random.choice(x.shape[0], size=npoints, replace=False)
            self._kde_points.append(x[xindices].T)
            # We can compute the bandwidth factor directly
            kde = gaussian_kde(x[xindices].T, bw_method=self.data_cfg.resampling.kde_bw_method)
            self._kde_bws.append(kde.factor)

        self.compute_stack = None
        self._densities = None
        self.individual_compute_stacks = {}
        if self.data_cfg.perform_data_checks:
            logger.debug("Running data checks")
            for x, y, n in zip(self._X, self._Y, self._networks):
                network_data_check(x, y, n)
        logger.info(f"Initialized a DataManager with {len(self._networks)} networks")

    def compute_densities(self, n_workers: int = 4):
        """
        Compute the densities at each data point in the dataset, for each sample,
        using parallel processing at the file level.
        """
        logger.debug(f"Computing densities in parallel with {n_workers} workers")
        logger.debug(f"Using cache dir {self.cache_dir}")

        def get_signature(kde_points, kde_bw, x):
            n = x.shape[0]
            stepsize = max(n // 100, 1)
            xsig = f"{x.shape}_{x[::stepsize]}"
            ksig = f"{kde_bw:.20f}_{kde_points.shape[0]}_{kde_points.shape[1]}"
            return f"{xsig}_{ksig}"

        # Prepare arguments for each file
        compute_args = [
            (
                kde_points,
                kde_bw,
                x,
                self.data_cfg.resampling.density_chunksize,
                self.cache_dir,
                get_signature(kde_points, kde_bw, x),
            )
            for kde_points, kde_bw, x in zip(self._kde_points, self._kde_bws, self._X)
        ]

        # Compute densities in parallel
        with Pool(n_workers) as pool:
            self._densities = list(
                tqdm(
                    pool.imap(compute_single_density, compute_args),
                    total=len(self._X),
                    desc="Computing densities",
                )
            )

        logger.debug(f"Done computing {len(self._densities)} densities")

    def get_batches(self, n_batches, batch_size, rng_key, concat_along_feature_axis=True):
        """
        Generate batches of data from the dataset using optimized sampling.
        """
        if self._densities is None:
            self.compute_densities()
            assert self._densities is not None

        # Create progress bar
        pbar = tqdm(total=len(self._networks), desc="Generating batches")

        # Prepare args for parallel processing
        sample_args = [
            (
                x,
                y,
                batch_size,
                n_batches,
                d,
                kp,
                kb,
                rng,
                self.data_cfg.resampling.density_threshold_quantile,
                self.data_cfg.resampling.density_threshold_coords,
            )
            for x, y, d, kp, kb, rng in zip(
                self._X,
                self._Y,
                self._densities,
                self._kde_points,
                self._kde_bws,
                jax.random.split(rng_key, len(self._networks)),
            )
        ]

        with Pool(min(len(self._networks), 8)) as pool:
            all_batches = list(pool.imap(fast_batch_sampling, sample_args))
            pbar.update(len(self._networks))

        pbar.close()

        xbatches, ybatches = zip(*all_batches)

        if concat_along_feature_axis:
            xbatches = np.concatenate(xbatches, axis=2)
            ybatches = np.concatenate(ybatches, axis=2)

            expected_x_features = sum(x.shape[1] for x in self._X)
            expected_y_features = sum(y.shape[1] for y in self._Y)
            assert xbatches.shape == (n_batches, batch_size, expected_x_features)
            assert ybatches.shape == (n_batches, batch_size, expected_y_features)
            assert xbatches.shape[2] == sum(n.get_nb_inputs() for n in self._networks)
            assert ybatches.shape[2] == sum(n.get_nb_outputs() for n in self._networks)

        return xbatches, ybatches

    def _old_get_batches(self, n_batches, batch_size, rng_key, concat_along_feature_axis=True):
        """
        Generate batches of data from the dataset, using the KDEs to resample the data
        and avoid over-representation of high-density regions.
        """
        if self._densities is None:
            self.compute_densities()
            assert self._densities is not None

        # Create temporary KDEs for sampling (we need them for density_at_midX calculation)
        kdes = [
            gaussian_kde(points, bw_method=bw)
            for points, bw in zip(self._kde_points, self._kde_bws)
        ]

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
                        kdes,
                        self._densities,
                        jax.random.split(rng_key, len(self._networks)),
                    )
                ),
                desc="generating batches",
            )
        ]
        xbatches, ybatches = zip(*all_batches)
        if concat_along_feature_axis:
            xbatches, ybatches = (
                np.concatenate(tuple(xbatches), axis=2),
                np.concatenate(tuple(ybatches), axis=2),
            )
            assert xbatches.shape == (n_batches, batch_size, sum([x.shape[1] for x in self._X]))
            assert ybatches.shape == (n_batches, batch_size, sum([y.shape[1] for y in self._Y]))
            assert xbatches.shape[2] == sum([n.get_nb_inputs() for n in self._networks])
            assert ybatches.shape[2] == sum([n.get_nb_outputs() for n in self._networks])

        return xbatches, ybatches

    @classmethod
    def from_xps(cls, xplist, config, **kw):
        network_cache_location = None
        if "network_cache_location" in config:
            network_cache_location = Path(config["network_cache_location"])

        net_sample_pairs = []
        for xp in xplist:
            net_sample_pairs.append(
                ut.get_cache(
                    lambda: xp.build_networks(**kw), f"{str(xp)}_net", network_cache_location
                )
            )

        networks, samples = zip(*net_sample_pairs)

        XY_pairs = []
        for xp, n, s in zip(xplist, networks, samples):
            XY_pairs.append(
                ut.get_cache(lambda: xp.get_XY(n, s), f"{str(xp)}_XY", network_cache_location)
            )

        X, Y = zip(*XY_pairs)
        # get everything as a long concatenated list
        X, Y, networks = (
            list(itertools.chain(*X)),
            list(itertools.chain(*Y)),
            list(itertools.chain(*networks)),
        )
        return cls(X, Y, networks)

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
            raise ValueError("Compute stack not built yet.")
        return self.compute_stack

    def rescale(self, X):
        return [self.data_cfg.rescaler.fwd(x) for x in X]

    def unscale(self, X):
        return [self.data_cfg.rescaler.inv(x) for x in X]

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
