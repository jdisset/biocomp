### {{{                          --     imports     --
import random
from functools import partial
from pathlib import Path
from typing import List, Tuple, Callable, Optional, Union, Literal

import numpy as np
import optax
import jax
import jax.numpy as jnp
from jax import vmap
from jax.tree_util import Partial
from jax.typing import ArrayLike

from assertpy import assert_that
from pydantic import Field, BaseModel, ConfigDict
from tqdm import tqdm

from biocomp.utils import encode_function, EncodedPartialFunction, ArbitraryModel
from biocomp.compute import ComputeStack
import biocomp.nodes as nd
from biocomp.network import Network
from .parameters import ParameterTree
from biocomp.logging_config import get_logger
from biocomptools.modelmodel import BiocompModel
from biocomp.designutils import sample_from_svg, data_to_lattice_2d
from biocomp.optimutils import make_training_step, per_replicate_step, optimize, DesignOptimConfig
from biocomp.designloss import distance_loss, grid_distance_loss  # noqa: F401 - re-exported for config compatibility

logger = get_logger(__name__)

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                     --     helper functions     --


def get_ind_params(params, target_id, ind_id):
    return jax.tree.map(lambda x: x[target_id, ind_id], params)


def plot_prediction(
    design_config,
    params,
    target_id,
    ind_id,
    net_id,
    target_x,
    target_y,
    key,
    stack,
    num_z,
    dep_output_mask,
    max_evals=30000,
):
    """Plot the prediction for a given target and individual."""
    import matplotlib.pyplot as plt

    params_ind = get_ind_params(params, target_id, ind_id)
    t_x = target_x[:, target_id]
    t_x = t_x.reshape(-1, t_x.shape[-1])[:max_evals]
    t_y = target_y[:, target_id]
    t_y = t_y.reshape(-1, t_y.shape[-1])[:max_evals]

    z = jax.random.uniform(key, (*t_x.shape[:-1], num_z))

    t_yhat = design_config.forward(params_ind, t_x, z, key, stack)
    t_yhatdep = t_yhat[..., dep_output_mask]
    assert_that(t_yhatdep.shape).is_equal_to(t_y.shape)

    loss_value = float(jnp.mean((t_yhatdep[:, net_id] - t_y[:, net_id]) ** 2))

    t_x_net = t_x[:, 2 * net_id : 2 * net_id + 2]

    # 2 subplots (ground truth and prediction)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    scatter1 = ax1.scatter(
        t_x_net[:, 0], t_x_net[:, 1], c=t_y[:, net_id], s=1, cmap="viridis", vmin=0, vmax=1
    )
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Ground Truth")
    ax1.set_aspect("equal")
    plt.colorbar(scatter1, ax=ax1)
    scatter2 = ax2.scatter(
        t_x_net[:, 0], t_x_net[:, 1], c=t_yhatdep[:, net_id], s=1, cmap="viridis", vmin=0, vmax=1
    )
    ax2.set_title("Prediction")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_aspect("equal")
    plt.colorbar(scatter2, ax=ax2)
    ax2.set_title(f"Prediction (Loss: {loss_value:.4f})")
    plt.tight_layout()


##────────────────────────────────────────────────────────────────────────────}}}


## {{{                          --     design manager   --
DEFAULT_RESCALE_TARGET = {
    "x": (0.0, 0.5),
    "y": (0.0, 0.5),
    "out": (0.09, 0.42),
}


class SamplingConfig(ArbitraryModel):
    strategy: Literal["uniform", "lattice"] = "uniform"


class UniformSampling(SamplingConfig):
    strategy: Literal["uniform"] = "uniform"
    n_samples: int = 5000


class LatticeSampling(SamplingConfig):
    strategy: Literal["lattice"] = "lattice"
    resolution: tuple[int, int] = (64, 64)
    jitter_std: float = 0.0


SamplingConfigUnion = Union[UniformSampling, LatticeSampling]


class Target(BaseModel):
    path: Union[str, Path]
    name: Optional[str] = None
    rescale_to: dict = DEFAULT_RESCALE_TARGET
    xlim: tuple[float, float] = (0.0, 1.0)
    ylim: tuple[float, float] = (0.0, 1.0)
    outlim: tuple[float, float] = (0.0, 1.0)
    transform_to_log_space: bool = False
    max_is_black: bool = True


class DataTarget(BaseModel):
    """Design target derived from experimental data.
    Instead of loading a target function from an SVG file, this class uses
    experimental data (X, Y arrays) and interpolates it to a lattice for design.
    This is useful for "reconstruction" design tasks where we want to find circuit
    parameters that reproduce known experimental behavior.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    X: np.ndarray  # input data (n_samples, n_dims)
    Y: np.ndarray  # output data (n_samples,) or (n_samples, 1)
    name: Optional[str] = None
    xlim: Optional[tuple[float, float]] = None  # if None, use data range
    ylim: Optional[tuple[float, float]] = None  # if None, use data range
    outlim: Optional[tuple[float, float]] = None  # if None, use data range
    z_slice: Optional[float] = None  # for 3D data, which z-slice to use
    z_tolerance: float = 0.05

    # Original network that produced this data (for baseline comparison)
    original_network: Optional[Network] = None

    # cached lattice data
    _lattice_X: Optional[np.ndarray] = None
    _lattice_Y: Optional[np.ndarray] = None

    @classmethod
    def from_plot_data(cls, plot_data, rescaler=None, **kwargs):
        """Create a DataTarget from a PlotData object.

        Args:
            plot_data: PlotData object containing x, y data and metadata
            rescaler: Optional rescaler to convert data to latent space
            **kwargs: Additional arguments passed to DataTarget constructor
        """
        X = np.asarray(plot_data.x)
        Y = np.asarray(plot_data.y)

        # optionally convert to latent space
        if rescaler is not None:
            X = rescaler.fwd(X)
            Y = rescaler.fwd(Y)

        name = kwargs.pop("name", plot_data.metadata.get("network_name", "data_target"))

        return cls(X=X, Y=Y, name=name, **kwargs)

    def get_lattice(
        self, resolution: tuple[int, int] = (48, 48), force_recompute: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the interpolated lattice representation of the data.

        Returns:
            X_lattice: Grid coordinates (n_points, 2)
            Y_lattice: Interpolated values (yres, xres) with NaNs filled
        """
        if self._lattice_X is not None and self._lattice_Y is not None and not force_recompute:
            return self._lattice_X, self._lattice_Y

        X_samples, Y_samples = data_to_lattice_2d(
            self.X,
            self.Y,
            xlims=self.xlim,
            ylims=self.ylim,
            resolution=resolution,
        )

        # Fill NaNs with the mean of valid values to prevent NaN propagation in loss
        nan_mask = np.isnan(Y_samples)
        if nan_mask.any():
            valid_mean = np.nanmean(Y_samples)
            Y_samples = np.where(nan_mask, valid_mean, Y_samples)
            logger.debug(
                f"Filled {nan_mask.sum()} NaN values in lattice with mean={valid_mean:.4f}"
            )

        self._lattice_X = X_samples
        self._lattice_Y = Y_samples

        return self._lattice_X, self._lattice_Y


TargetUnion = Union[Target, DataTarget]


class DesignManager(BaseModel):
    """Handles loading and sampling of 2d design target data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    targets: list[TargetUnion]
    networks: List[Network]
    sampling: SamplingConfigUnion = Field(default_factory=UniformSampling, discriminator="strategy")

    def model_post_init(self, *a, **kw):
        super().model_post_init(*a, **kw)
        for target in self.targets:
            if isinstance(target, Target) and target.transform_to_log_space:
                target.xlim = (0.1, 1)
                target.ylim = (0.1, 1)

    @property
    def has_data_targets(self) -> bool:
        return any(isinstance(t, DataTarget) for t in self.targets)

    def _sample_from_target(
        self,
        target: TargetUnion,
        n: int,
        seed: int,
        grid: Optional[tuple[int, int]] = None,
        jitter: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample from either Target or DataTarget."""
        if isinstance(target, DataTarget):
            if grid is not None:
                # Lattice mode: use cached KNN-interpolated grid
                X_lattice, Y_lattice = target.get_lattice(resolution=grid)
                X_tiled = np.tile(X_lattice, (n, 1))
                Y_tiled = np.tile(Y_lattice[None, ...], (n, 1, 1))  # (n, yres, xres)
                return X_tiled, Y_tiled
            else:
                # Uniform mode: randomly sample from stored data
                rng = np.random.default_rng(seed)
                n_samples = len(target.X)
                indices = rng.choice(n_samples, size=n, replace=True)
                X_sampled = target.X[indices]  # (n, 2)
                Y_sampled = target.Y[indices]  # (n, 1) or (n,)
                if Y_sampled.ndim == 1:
                    Y_sampled = Y_sampled[:, None]
                return X_sampled, Y_sampled
        else:
            # SVG-based Target
            return sample_from_svg(
                target.path,
                n=n,
                seed=seed,
                log=target.transform_to_log_space,
                xlim=target.xlim,
                ylim=target.ylim,
                outlim=target.outlim,
                rescale_to=target.rescale_to,
                max_is_black=target.max_is_black,
                grid=grid,
                grid_jitter_std=jitter,
            )

    def get_samples(
        self,
        samples: int | tuple[int, ...],
        seed: Optional[int | ArrayLike] = None,
    ) -> tuple[jax.Array, jax.Array]:
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        elif not isinstance(seed, int):
            # Convert JAX array to int seed
            key_data = jax.random.key_data(seed)
            seed = int(key_data[0]) % (2**31)

        if isinstance(self.sampling, LatticeSampling):
            return self._get_lattice_samples(samples, seed)
        return self._get_uniform_samples(samples, seed)

    def _get_uniform_samples(
        self,
        samples: int | tuple[int, ...],
        seed: int,
    ) -> tuple[list[jax.Array], list[jax.Array]]:
        if isinstance(samples, int):
            samples = (samples,)

        n_networks = samples[0]
        requested_shape = samples[1:]
        n = int(np.prod(requested_shape))

        all_xsamples, all_ysamples = [], []

        for _ in range(n_networks):
            xsamples, ysamples = [], []
            for target in self.targets:
                xsample, ysample = self._sample_from_target(target, n=n, seed=seed, grid=None)
                xsamples.append(xsample)
                ysamples.append(ysample)

            xsamples = jnp.stack(xsamples, axis=1)
            ysamples = jnp.stack(ysamples, axis=1)

            assert_that(xsamples.shape).is_equal_to((n, len(self.targets), 2))
            assert_that(ysamples.shape).is_equal_to((n, len(self.targets), 1))

            xsamples = xsamples.reshape(*requested_shape, len(self.targets), 2)
            ysamples = ysamples.reshape(*requested_shape, len(self.targets), 1)

            all_xsamples.append(xsamples)
            all_ysamples.append(ysamples)

        return all_xsamples, all_ysamples

    def _get_lattice_samples(
        self,
        samples: int | tuple[int, ...],
        seed: int,
    ) -> tuple[list[jax.Array], list[jax.Array]]:
        assert isinstance(self.sampling, LatticeSampling)
        xres, yres = self.sampling.resolution
        jitter = self.sampling.jitter_std

        if isinstance(samples, int):
            samples = (samples,)

        n_networks = samples[0]
        requested_shape = samples[1:]
        n = int(np.prod(requested_shape))

        all_xsamples, all_ysamples = [], []

        for _ in range(n_networks):
            xsamples, ysamples = [], []
            for target in self.targets:
                X, Y_grid = self._sample_from_target(
                    target, n=n, seed=seed, grid=(xres, yres), jitter=jitter
                )
                xsamples.append(X)
                ysamples.append(Y_grid)

            xsamples = jnp.stack(xsamples, axis=1)
            ysamples = jnp.stack(ysamples, axis=1)

            n_pts = n * yres * xres
            assert_that(xsamples.shape).is_equal_to((n_pts, len(self.targets), 2))
            assert_that(ysamples.shape).is_equal_to((n, len(self.targets), yres, xres))

            ysamples_flat = ysamples.transpose(0, 2, 3, 1).reshape(n_pts, len(self.targets), 1)

            new_batch_shape = requested_shape[:-1] + (requested_shape[-1] * yres * xres,)
            xsamples = xsamples.reshape(*new_batch_shape, len(self.targets), 2)
            ysamples_flat = ysamples_flat.reshape(*new_batch_shape, len(self.targets), 1)

            all_xsamples.append(xsamples)
            all_ysamples.append(ysamples_flat)

        return all_xsamples, all_ysamples

    @property
    def is_lattice_mode(self) -> bool:
        return isinstance(self.sampling, LatticeSampling)

    @property
    def grid_resolution(self) -> Optional[tuple[int, int]]:
        if isinstance(self.sampling, LatticeSampling):
            return self.sampling.resolution
        return None

    def build_stack(self, model: BiocompModel, unlock_ratios=True):
        logger.info(f"Building stack with {len(self.networks)} design networks")
        logger.info(f"Design network names: {[n.name for n in self.networks]}")
        stack = ComputeStack(networks=self.networks)
        logger.info(f"Stack after creation has {len(stack.networks)} networks")
        if unlock_ratios:
            assert model.compute_config is not None
            assert model.compute_config.node_functions is not None

            model.compute_config.node_functions["aggregation"] = encode_function(
                partial(nd.aggregation, random_init=True)
            )

        stack.build(model.compute_config)
        logger.info(
            f"Stack built: {stack.get_nb_networks()} networks, "
            f"{stack.get_nb_inputs()} inputs, {stack.get_nb_outputs()} outputs"
        )
        logger.info(f"Stack network names after build: {[n.name for n in stack.networks]}")
        return stack

    @property
    def n_targets(self):
        return len(self.targets)


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                   --     param initialization     --


def initialize_params(stack, n_replicates, n_targets, shared_params, key):
    # could be faster if we stacked copies of the shared parameters and did the merge on the whole stack...
    # good enough for now
    def init_single(k):
        params = stack.init(k)
        _, nonshared = params.filter_by_tag(["shared"])
        return ParameterTree.merge(shared_params, nonshared)

    def init_target_params(k):
        params = vmap(init_single)(jax.random.split(k, n_targets))
        return params

    return vmap(init_target_params)(jax.random.split(key, n_replicates))


class DesignConfig(DesignOptimConfig):
    loss_function: EncodedPartialFunction = Field(default=distance_loss)
    n_replicates: int = 4
    keep_in_history: List[str] = ["loss", "all_losses"]


##────────────────────────────────────────────────────────────────────────────}}}


def assert_tree_shape(tree, expected_shape, only_first_dims=True):
    """Assert that the shape of each leaf in the tree matches the expected shape."""

    N_DIMS = len(expected_shape)

    def check_shape(x):
        if isinstance(x, jax.Array):
            assert_that(x.shape[:N_DIMS] if only_first_dims else x.shape).is_equal_to(
                expected_shape
            )
        return x

    jax.tree.map(check_shape, tree)


## {{{                   --     evaluation and analysis     --


def get_topk_replicate_network_pairs(
    losses: jax.Array,
    dmanager: DesignManager,
    dconf: DesignConfig,
    k: int = 1,
) -> List[List[Tuple[int, int, float]]]:
    """Find top-k replicate/network pairs with lowest loss for each target.

    Args:
        losses: Loss values shape (n_replicates, n_targets, n_networks)
        dmanager: Design manager with networks and targets
        dconf: Design configuration
        k: Number of top pairs to return per target

    Returns:
        List of lists, one per target, each containing k tuples of (replicate_id, network_id, loss_value)
    """
    n_replicates, n_targets, n_networks = losses.shape
    assert_that(n_replicates).is_equal_to(dconf.n_replicates)
    assert_that(n_targets).is_equal_to(dmanager.n_targets)
    assert_that(n_networks).is_equal_to(len(dmanager.networks))
    k = min(k, n_replicates * n_networks)

    best_per_target = []
    for tid in range(n_targets):
        tlosses = losses[:, tid, :]  # shape: (n_replicates, n_networks)
        flat_tlosses = tlosses.reshape((-1,))  # shape: (n_replicates * n_networks,)
        topk_flat_idx = jnp.argsort(flat_tlosses)[:k]

        # convert flat indices back to (replicate_id, network_id)
        rep_ids, net_ids = jnp.unravel_index(topk_flat_idx, (n_replicates, n_networks))
        topk_pairs = [
            (int(rep_ids[j]), int(net_ids[j]), float(flat_tlosses[topk_flat_idx[j]]))
            for j in range(k)
        ]
        best_per_target.append(topk_pairs)

    return best_per_target


def plot_design_results(
    dmanager: DesignManager,
    dconf: DesignConfig,
    xraw: jax.Array,
    yraw: jax.Array,
    topk: List[List[Tuple[int, int, float]]],
    yhatdep: Optional[jax.Array] = None,
    n_eval_samples: Optional[int] = None,
    save_dir: Optional[Path] = None,
    show_difference: bool = False,
    plot_top_k: Optional[int] = None,
) -> None:
    """Plot design results for each target showing best replicate/network combination.

    Args:
        dmanager: Design manager with networks and targets
        dconf: Design configuration
        xraw: Input samples shape (n_networks, n_replicates, n_eval_samples, n_targets, 2)
        yraw: Target samples shape (n_networks, n_replicates, n_eval_samples, n_targets, 1)
        yhatdep: Predictions shape (n_replicates, n_eval_samples, n_targets, n_networks)
        topk: Top-k results from get_topk_replicate_network_pairs
        n_eval_samples: Maximum number of samples to plot (for performance)
        save_dir: Directory to save figures (if None, just display)
        show_difference: Whether to show difference plots between prediction and target
        plot_top_k: Number of top-k designs to plot per target (default: 1, i.e., just the best)
    """
    import matplotlib.pyplot as plt
    from biocomp.plotting.plotting_core import DEFAULT_CMAP_NAME

    if n_eval_samples is None:
        n_eval_samples = xraw.shape[2]
    else:
        n_eval_samples = min(n_eval_samples, xraw.shape[2])

    # validate shapes
    n_networks = len(dmanager.networks)
    assert_that(xraw).has_shape(
        (n_networks, dconf.n_replicates, xraw.shape[2], dmanager.n_targets, 2)
    )
    assert_that(yraw).has_shape(
        (n_networks, dconf.n_replicates, yraw.shape[2], dmanager.n_targets, 1)
    )

    # determine how many top-k results to plot
    if plot_top_k is None:
        plot_top_k = 1  # default to just the best result

    for tid, target in enumerate(dmanager.targets):
        # plot multiple top-k results for this target
        n_to_plot = min(plot_top_k, len(topk[tid]))

        for rank in range(n_to_plot):
            rep_id, net_id, loss_val = topk[tid][rank]

            # get data for this specific target/network/replicate combo
            x_target = xraw[net_id, rep_id, :n_eval_samples, tid]  # shape: (n_samples, 2)
            y_target = yraw[net_id, rep_id, :n_eval_samples, tid, 0]  # squeeze last dim

            # assertions
            assert_that(x_target).has_shape((n_eval_samples, 2))
            assert_that(y_target).has_shape((n_eval_samples,))

            # create figure
            nax = 3 if show_difference else 2
            fig, axes = plt.subplots(1, nax, figsize=(nax * 5, 5), dpi=100)

            # ground truth
            sc1 = axes[0].scatter(
                x_target[:, 0], x_target[:, 1], c=y_target, cmap=DEFAULT_CMAP_NAME, s=5, alpha=0.7
            )
            axes[0].set_title("Target")
            axes[0].set_aspect("equal")
            plt.colorbar(sc1, ax=axes[0])

            if yhatdep is not None:
                # prediction
                yhat_target = yhatdep[rep_id, :n_eval_samples, tid, net_id]  # shape: (n_samples,)
                assert_that(yhat_target).has_shape((n_eval_samples,))
                assert_that(yhatdep).has_shape(
                    (dconf.n_replicates, yhatdep.shape[1], dmanager.n_targets, n_networks)
                )
                sc2 = axes[1].scatter(
                    x_target[:, 0],
                    x_target[:, 1],
                    c=yhat_target,
                    cmap=DEFAULT_CMAP_NAME,
                    s=5,
                    alpha=0.7,
                )
                axes[1].set_title(f"Prediction (rank {rank + 1}, loss={loss_val:.4f})")
                axes[1].set_aspect("equal")
                plt.colorbar(sc2, ax=axes[1])

                # difference
                if show_difference:
                    diff = yhat_target - y_target
                    assert_that(diff).has_shape((n_eval_samples,))
                    vmax = jnp.abs(diff).max()
                    sc3 = axes[2].scatter(
                        x_target[:, 0],
                        x_target[:, 1],
                        c=diff,
                        cmap="RdBu_r",
                        s=5,
                        alpha=0.7,
                        vmin=-vmax,
                        vmax=vmax,
                    )
                    axes[2].set_title(f"Difference (net: {dmanager.networks[net_id].name})")
                    axes[2].set_aspect("equal")
                    plt.colorbar(sc3, ax=axes[2])

            plt.suptitle(
                f"Target: {target.name} | Rank {rank + 1}: net {dmanager.networks[net_id].name})"
            )
            plt.tight_layout()

            if save_dir:
                # Use rank as prefix for consistency with recipe files
                save_path = (
                    Path(save_dir) / f"rank{rank + 1:02d}_{target.name}_rep{rep_id}_net{net_id}.png"
                )
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                logger.info(f"Saved figure to {save_path}")

            plt.show()


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                       --     main design function     --


def normalize_ratios_prune(current_ratios, rel_off=1e-3, eps=1e-12):
    A = jnp.abs(current_ratios)
    m = jnp.maximum(jnp.max(A, axis=1, keepdims=True), eps)
    norm = A / m
    mask = norm >= rel_off
    return jnp.where(mask, norm, 0.0)


def get_ratio_paths(params):
    ratio_paths = []
    for path, value in params.data.iter_leaves():
        if "ratio" in str(path) and "inverse" not in str(path):
            ratio_paths.append(path)
    return ratio_paths


def start(
    dmanager: DesignManager,
    dconf: DesignConfig,
    model: BiocompModel,
    loggers: Optional[List[Tuple[int, Callable]]] = None,
    async_handler=None,
):
    import time

    t0 = time.time()
    pkey, bkey, loop_key = jax.random.split(dconf.seed_key, 3)

    logger.info("Building compute stack...")
    stack = dmanager.build_stack(model)
    logger.info(f"Stack built in {time.time() - t0:.2f}s")

    t1 = time.time()
    logger.info("Initializing parameters...")
    initial_params = initialize_params(
        stack, dconf.n_replicates, dmanager.n_targets, model.shared_params, pkey
    )
    assert_tree_shape(initial_params, (dconf.n_replicates, dmanager.n_targets))
    static, dynamic = initial_params.filter_by_tag(["non_grad", "shared"])
    initial_optimizer_state = vmap(vmap(dconf.optimizer.init))(dynamic)
    logger.info(f"Parameters initialized in {time.time() - t1:.2f}s")

    # -- get data --
    num_z = static["global/number_of_random_variables"]
    assert_that(num_z.shape[0]).is_equal_to(dconf.n_replicates)
    assert_that(jnp.all(num_z == num_z[0])).is_true()
    num_z = (dmanager.n_targets, int(num_z.ravel()[0].squeeze()))

    steps_per_epoch = max(1, dconf.n_batches_per_epoch // dconf.batches_per_step)
    total_steps = int(dconf.n_epochs * steps_per_epoch)

    logger.debug(
        f"Total steps: {total_steps}, Steps per epoch: {steps_per_epoch}, \n"
        f"Batch size: {dconf.batch_size}, Batches per step: {dconf.batches_per_step}"
    )
    assert_that(total_steps).is_greater_than(0)

    n_networks = stack.get_nb_networks()

    t2 = time.time()
    logger.info("Generating training samples...")
    xbatches_list, ybatches_list = dmanager.get_samples(
        (
            len(dmanager.networks),
            steps_per_epoch,
            dconf.n_replicates,
            dconf.batches_per_step,
            dconf.batch_size,
        ),
        bkey,
    )

    xbatches = jnp.concatenate(xbatches_list, axis=-1)
    ybatches = ybatches_list[0]
    logger.info(f"Samples generated in {time.time() - t2:.2f}s")

    effective_batch_size = dconf.batch_size
    if dmanager.is_lattice_mode:
        xres, yres = dmanager.grid_resolution
        effective_batch_size *= xres * yres

    n_design_inputs = 2 * len(dmanager.networks)

    logger.info(
        f"Data generated: {len(dmanager.networks)} design networks, "
        f"n_design_inputs={n_design_inputs}, xbatches.shape={xbatches.shape}"
    )

    assert_that(xbatches).has_shape(
        (
            steps_per_epoch,
            dconf.n_replicates,
            dconf.batches_per_step,
            effective_batch_size,
            dmanager.n_targets,
            n_design_inputs,
        )
    )

    # -- step function --
    ratio_paths = get_ratio_paths(initial_params)

    def norm_ratios_hook(params, *a, **kw):
        logger.debug("Normalizing ratios...")
        return params.update_leaves_by_path(ratio_paths, normalize_ratios_prune)

    loss_func = dconf.loss_function.get_impl()(
        stack, dconf, dmanager, num_z=num_z, ratio_paths=ratio_paths
    )
    step_fn = make_training_step(
        loss_func,
        dconf.optimizer,
        dconf.keep_in_history,
        scannable=True,
        post_update_hook=norm_ratios_hook,
        updates_need_vmap=True,
        static_tags=["non_grad", "shared"],
    )

    def step(params: ParameterTree, opt_state: optax.OptState, step_key, xs, ys):
        keys = jax.random.split(step_key, dconf.n_replicates)
        assert_that(xs).has_shape(
            (
                dconf.n_replicates,
                dconf.batches_per_step,
                effective_batch_size,
                dmanager.n_targets,
                n_design_inputs,
            )
        )
        expected_y_last_dim = 1 if dmanager.is_lattice_mode else n_networks
        assert_that(ys).has_shape(
            (
                dconf.n_replicates,
                dconf.batches_per_step,
                effective_batch_size,
                dmanager.n_targets,
                expected_y_last_dim,
            )
        )
        assert_tree_shape(params, (dconf.n_replicates, dmanager.n_targets))
        assert_tree_shape(opt_state, (dconf.n_replicates, dmanager.n_targets))

        return jax.vmap(
            Partial(per_replicate_step, num_z=num_z, training_config=dconf, scannable_step=step_fn)
        )(params, opt_state, keys, xs, ys)

    return optimize(
        step,
        initial_params,
        initial_optimizer_state,
        xbatches=xbatches,
        ybatches=ybatches,
        config=dconf,
        n_total_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
        key=loop_key,
        stack=stack,
        loggers=loggers,
        async_handler=async_handler,
        verbose=True,
    )


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                   --     evaluation functions     --


def sample_for_evaluation(
    dmanager: DesignManager,
    dconf: DesignConfig,
    final_params: ParameterTree,
    n_eval_samples: int,
    key: jax.Array,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample evaluation data for design assessment.

    Args:
        dmanager: Design manager with targets and networks
        dconf: Design configuration
        final_params: Optimized parameters
        n_eval_samples: Number of evaluation samples
        key: Random key for sampling

    Returns:
        xraw: Input samples (n_networks, n_replicates, n_eval_samples, n_targets, 2)
        yraw: Target values (n_networks, n_replicates, n_eval_samples, n_targets, 1)
    """
    n_networks = len(dmanager.networks)
    n_replicates = dconf.n_replicates
    n_targets = dmanager.n_targets

    # Convert key to seed using a safe method
    key_data = jax.random.key_data(key)
    seed = int(key_data[0]) % (2**31)

    # Force uniform sampling for evaluation (not lattice)
    xlist, ylist = dmanager._get_uniform_samples((n_networks, n_replicates, n_eval_samples), seed)

    # stack network dimension
    xraw = jnp.stack(xlist, axis=0)
    yraw = jnp.stack(ylist, axis=0)

    expected_x_shape = (n_networks, n_replicates, n_eval_samples, n_targets, 2)
    expected_y_shape = (n_networks, n_replicates, n_eval_samples, n_targets, 1)

    assert xraw.shape == expected_x_shape, f"xraw shape {xraw.shape} != expected {expected_x_shape}"
    assert yraw.shape == expected_y_shape, f"yraw shape {yraw.shape} != expected {expected_y_shape}"

    return xraw, yraw


def evaluate_design(
    dmanager: DesignManager,
    dconf: DesignConfig,
    model,  # BiocompModel
    final_params: ParameterTree,
    xraw: jnp.ndarray,
    yraw: jnp.ndarray,
    key: jax.Array,
    max_eval_size: int = 64,
    max_loss_size: int = 64,
    store_predictions: bool = True,
) -> Tuple[Optional[jnp.ndarray], jnp.ndarray]:
    """Evaluate design quality by running predictions and computing losses.

    Args:
        dmanager: Design manager with targets and networks
        dconf: Design configuration
        model: Trained biocomp model
        final_params: Optimized parameters
        xraw: Input samples (n_networks, n_replicates, n_samples, n_targets, 2)
        yraw: Target values (n_networks, n_replicates, n_samples, n_targets, 1)
        key: Random key
        max_eval_size: Max batch size for forward pass
        max_loss_size: Max batch size for loss computation
        store_predictions: Whether to store and return full predictions (memory intensive)

    Returns:
        yhatdep: Predictions (n_replicates, n_samples, n_targets, n_networks) or None
        losses: Per-replicate/target/network losses (n_replicates, n_targets, n_networks)
    """
    stack = dmanager.build_stack(model, unlock_ratios=False)

    n_networks = len(dmanager.networks)
    n_replicates = dconf.n_replicates
    n_targets = dmanager.n_targets
    n_samples = xraw.shape[2]

    logger.info(
        f"Starting evaluation: {n_replicates} replicates × {n_targets} targets × {n_samples} samples"
    )

    num_z = final_params["global/number_of_random_variables"]
    assert num_z.shape[0] == n_replicates
    num_z_val = int(num_z[0, 0].squeeze())

    dep_mask = stack.get_dependent_output_mask()

    # reshape for evaluation: we need to combine network inputs
    # xraw shape: (n_networks, n_replicates, n_samples, n_targets, 2)
    # we need: (n_replicates, n_samples, n_targets, n_networks * 2)
    x_combined = xraw.transpose(1, 2, 3, 0, 4).reshape(n_replicates, n_samples, n_targets, -1)

    # yraw: just take first network's target (they're all same)
    y_combined = yraw[0]  # (n_replicates, n_samples, n_targets, 1)

    all_losses = []
    all_predictions = [] if store_predictions else None

    # Pre-compile the apply function for speed
    apply_batched = jax.vmap(stack.apply, in_axes=(None, 0, 0, 0))

    total_iterations = n_replicates * n_targets
    pbar = tqdm(total=total_iterations, desc="Evaluating designs", unit="rep×tgt")

    for rep_idx in range(n_replicates):
        rep_losses_per_target = []
        rep_predictions_per_target = [] if store_predictions else None

        for tid in range(n_targets):
            # get params for this replicate and target
            rep_params = jax.tree.map(lambda x: x[rep_idx, tid], final_params)

            x_slice = x_combined[rep_idx, :, tid, :]  # (n_samples, n_inputs)
            y_slice = y_combined[rep_idx, :, tid, :]  # (n_samples, 1)

            # forward pass in chunks
            yhats = []
            for start in range(0, n_samples, max_eval_size):
                end = min(start + max_eval_size, n_samples)
                x_batch = x_slice[start:end]
                z_batch = jax.random.uniform(key, (end - start, num_z_val))
                keys_batch = jax.random.split(key, end - start)

                yhat, _ = apply_batched(rep_params, x_batch, z_batch, keys_batch)
                yhats.append(yhat)

            yhat_full = jnp.concatenate(yhats, axis=0)
            yhat_dep = jnp.compress(dep_mask, yhat_full, axis=-1)  # (n_samples, n_networks)

            if store_predictions:
                rep_predictions_per_target.append(yhat_dep)

            # compute per-network loss (vectorized)
            y_expanded = jnp.tile(y_slice, (1, n_networks))  # (n_samples, n_networks)
            net_losses = jnp.mean((yhat_dep - y_expanded) ** 2, axis=0)  # (n_networks,)
            rep_losses_per_target.append(net_losses.tolist())

            pbar.update(1)

        all_losses.append(rep_losses_per_target)
        if store_predictions:
            # Stack predictions for this replicate: (n_targets, n_samples, n_networks)
            all_predictions.append(jnp.stack(rep_predictions_per_target, axis=0))

    pbar.close()

    losses = jnp.array(all_losses)  # (n_replicates, n_targets, n_networks)
    logger.info(
        f"Evaluation complete. Loss range: [{float(losses.min()):.4f}, {float(losses.max()):.4f}]"
    )

    if store_predictions:
        # Stack all replicate predictions: (n_replicates, n_targets, n_samples, n_networks)
        yhatdep = jnp.stack(all_predictions, axis=0)
        # Transpose to expected shape: (n_replicates, n_samples, n_targets, n_networks)
        yhatdep = yhatdep.transpose(0, 2, 1, 3)
        return yhatdep, losses

    return None, losses


def compute_baseline_loss(
    dmanager: DesignManager,
    model,  # BiocompModel
    n_samples: int = 1000,
    seed: int = 42,
    max_batch_size: int = 200,
) -> dict:
    """Compute baseline loss for DataTargets that have original_network.

    This runs the model's prediction on the original network (ground truth recipe)
    and compares it against the actual experimental data. This gives us two baselines:
    1. Data loss: How well does the experimental data match itself (always 0 for MSE)
    2. Model loss: How well does the model predict the original network's behavior

    Args:
        dmanager: Design manager with DataTargets
        model: Trained biocomp model
        n_samples: Number of samples to use for evaluation
        seed: Random seed
        max_batch_size: Max batch size for forward pass

    Returns:
        Dict with baseline info per target:
        {
            'target_name': {
                'has_original_network': bool,
                'model_prediction_loss': float,  # Model prediction vs actual data
                'original_network_name': str,
            }
        }
    """
    results = {}
    rng = np.random.default_rng(seed)

    for tid, target in enumerate(dmanager.targets):
        target_name = target.name or f"target_{tid}"

        if not isinstance(target, DataTarget):
            results[target_name] = {"has_original_network": False}
            continue

        if target.original_network is None:
            results[target_name] = {"has_original_network": False}
            continue

        # Build a stack with just the original network
        original_network = target.original_network
        stack = ComputeStack(networks=[original_network])
        stack.build(model.compute_config)

        # Get params for this network from the model
        params = stack.init(jax.random.key(seed))
        shared_params = model.shared_params
        _, nonshared = params.filter_by_tag(["shared"])
        params = ParameterTree.merge(shared_params, nonshared)

        # Sample from the target data
        n_data = len(target.X)
        indices = rng.choice(n_data, size=min(n_samples, n_data), replace=False)
        X_sample = target.X[indices]  # (n_samples, 2)
        Y_sample = target.Y[indices]  # (n_samples,) or (n_samples, 1)
        if Y_sample.ndim == 1:
            Y_sample = Y_sample[:, None]

        # Get random variables config
        num_z_val = params["global/number_of_random_variables"]
        num_z = int(num_z_val.squeeze() if hasattr(num_z_val, "squeeze") else num_z_val)
        dep_mask = stack.get_dependent_output_mask()

        # Run forward pass in batches
        apply_batched = jax.vmap(stack.apply, in_axes=(None, 0, 0, 0))
        yhats = []
        key = jax.random.key(seed)

        for start in range(0, len(X_sample), max_batch_size):
            end = min(start + max_batch_size, len(X_sample))
            x_batch = jnp.array(X_sample[start:end])
            z_batch = jax.random.uniform(key, (end - start, num_z))
            keys_batch = jax.random.split(key, end - start)

            yhat, _ = apply_batched(params, x_batch, z_batch, keys_batch)
            yhats.append(yhat)

        yhat_full = jnp.concatenate(yhats, axis=0)
        yhat_dep = jnp.compress(dep_mask, yhat_full, axis=-1)  # (n_samples, 1)

        # Compute MSE loss
        model_loss = float(jnp.mean((yhat_dep - Y_sample) ** 2))

        results[target_name] = {
            "has_original_network": True,
            "model_prediction_loss": model_loss,
            "original_network_name": original_network.name,
            "n_samples": len(X_sample),
        }

        logger.info(
            f"Baseline for '{target_name}': model_loss={model_loss:.6f} (original network: {original_network.name})"
        )

    return results


##────────────────────────────────────────────────────────────────────────────}}}
