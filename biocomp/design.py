from __future__ import annotations
import random
from functools import partial
from pathlib import Path
from typing import Literal, Callable, Any, TYPE_CHECKING
import warnings

import numpy as np
import optax
import jax
import jax.numpy as jnp
from jax import vmap
from jax.tree_util import Partial
from jax.typing import ArrayLike

from assertpy import assert_that
from pydantic import BaseModel, ConfigDict, Field, model_validator
from tqdm import tqdm

from biocomp.utils import encode_function, EncodedPartialFunction
from biocomp.compute import ComputeStack
import biocomp.nodes as nd
from biocomp.network import Network
from .parameters import ParameterTree
from biocomp.logging_config import get_logger
from biocomp.optimutils import make_training_step, per_replicate_step, optimize, DesignOptimConfig

if TYPE_CHECKING:
    from biocomptools.modelmodel import BiocompModel
from biocomp.designloss import distance_loss, grid_distance_loss  # noqa: F401 - re-exported
from biocomp.tumasking import build_tu_id_mapping, TU_LOG_ALPHA_PATH, LOG_ALPHA_MIN, LOG_ALPHA_MAX
from biocomp.designdebug import save_debug_state, is_design_debug_enabled

# re-export target classes and sampling configs for backward compatibility
from biocomp.design_targets import (  # noqa: F401
    SamplingConfig,
    UniformSampling,
    LatticeSampling,
    SamplingConfigUnion,
    TargetBase,
    SVGTarget,
    Target,
    DataTarget,
    TargetUnion,
    DEFAULT_RESCALE_TARGET,
)

logger = get_logger(__name__)

_debug_output_dir: str | None = None


def set_design_debug_output_dir(output_dir: str | None):
    global _debug_output_dir
    _debug_output_dir = output_dir


def get_design_debug_output_dir() -> str | None:
    return _debug_output_dir


class DesignManager(BaseModel):
    """Handles loading and sampling of 2d design target data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    targets: list[TargetUnion]
    networks: list[Network]
    sampling: SamplingConfigUnion = Field(default_factory=UniformSampling, discriminator="strategy")

    enable_tu_masking: bool = False
    _tu_ids: list[str] | None = None
    _tu_id_to_idx: dict[str, int] | None = None

    @model_validator(mode="after")
    def _validate_input_order_consistency(self) -> "DesignManager":
        """Warn if scaffold networks lack input_order with DataTargets.

        For design with DataTargets, scaffold networks should have explicit input_order
        to ensure correct alignment between target data X columns and network inputs.
        Without input_order, protein-name matching is attempted which may fail or give
        incorrect results when scaffold proteins differ from target proteins.
        """
        if not self.has_data_targets:
            return self

        networks_without_order = [n.name for n in self.networks if not n.has_input_order()]

        if networks_without_order:
            target_with_different_proteins = []
            for t in self.targets:
                if not isinstance(t, DataTarget):
                    continue
                t_names = getattr(t, "input_names", None)
                if t_names is None and t.original_network is not None:
                    t_names = sorted(t.original_network.get_inverted_input_proteins())
                if t_names:
                    for net in self.networks:
                        if not net.has_input_order():
                            net_proteins = net.get_inverted_input_proteins()
                            if set(t_names) != set(net_proteins):
                                target_with_different_proteins.append(
                                    f"{t.name}: target={t_names}, network={net_proteins}"
                                )

            if target_with_different_proteins:
                warnings.warn(
                    f"Design networks without input_order have different proteins than DataTargets. "
                    f"This may cause axis alignment issues. Consider adding input_order to scaffold recipes. "
                    f"Mismatches: {target_with_different_proteins[:3]}",
                    stacklevel=3,
                )
        return self

    @property
    def has_data_targets(self) -> bool:
        return any(isinstance(t, DataTarget) for t in self.targets)

    def _ensure_tu_mapping(self):
        """Build TU ID mapping if not already built."""
        if self._tu_id_to_idx is None:
            self._tu_ids, self._tu_id_to_idx = build_tu_id_mapping(self.networks)
            logger.info(f"Built TU mapping: {len(self._tu_ids)} TUs")

    @property
    def n_tus(self) -> int:
        """Number of unique TUs across all networks."""
        self._ensure_tu_mapping()
        return len(self._tu_ids)

    @property
    def tu_id_to_idx(self) -> dict[str, int]:
        """Mapping from TU ID string to index."""
        self._ensure_tu_mapping()
        return self._tu_id_to_idx

    def _sample_from_target(
        self,
        target: TargetUnion,
        n: int,
        seed: int,
        grid: tuple[int, int] | None = None,
        jitter: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample from target using its interface methods."""
        if grid is not None:
            X_lattice, Y_lattice = target.get_lattice(resolution=grid, seed=seed)
            X_tiled = np.tile(X_lattice, (n, 1))
            Y_tiled = np.tile(Y_lattice[None, ...], (n, 1, 1))
            return X_tiled, Y_tiled
        else:
            return target.sample_uniform(n=n, seed=seed)

    def get_samples(
        self,
        samples: int | tuple[int, ...],
        seed: int | ArrayLike | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        elif not isinstance(seed, int):
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
        noise_std = self.sampling.noise_std

        if isinstance(samples, int):
            samples = (samples,)

        n_networks = samples[0]
        requested_shape = samples[1:]
        n = int(np.prod(requested_shape))

        all_xsamples, all_ysamples = [], []

        for net_idx in range(n_networks):
            xsamples, ysamples = [], []
            for t_idx, target in enumerate(self.targets):
                X, Y_grid = self._sample_from_target(
                    target, n=n, seed=seed, grid=(xres, yres), jitter=jitter
                )
                if noise_std > 0:
                    rng = np.random.default_rng(seed + t_idx * 7919 + net_idx * 6971)
                    Y_grid = Y_grid + rng.normal(0, noise_std, Y_grid.shape)
                    if hasattr(target, "latent_out"):
                        Y_grid = np.clip(Y_grid, target.latent_out[0], target.latent_out[1])
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

        # Debug: comprehensive snapshot of lattice sampling
        if is_design_debug_enabled() and all_xsamples:
            save_debug_state(
                "DesignManager_lattice_samples",
                {
                    "xsamples": all_xsamples[0],
                    "ysamples": all_ysamples[0],
                },
                {
                    "resolution": (xres, yres),
                    "jitter": jitter,
                    "n_networks": n_networks,
                    "n_targets": len(self.targets),
                    "target_names": [
                        getattr(t, "name", f"target_{i}") for i, t in enumerate(self.targets)
                    ],
                    "target_input_names": [getattr(t, "input_names", None) for t in self.targets],
                    "xsamples_shape": all_xsamples[0].shape,
                    "ysamples_shape": all_ysamples[0].shape,
                },
                output_dir=get_design_debug_output_dir(),
                mode="design",
            )

        return all_xsamples, all_ysamples

    @property
    def is_lattice_mode(self) -> bool:
        return isinstance(self.sampling, LatticeSampling)

    @property
    def grid_resolution(self) -> tuple[int, int] | None:
        if isinstance(self.sampling, LatticeSampling):
            return self.sampling.resolution
        return None

    def build_stack(self, model: BiocompModel, unlock_ratios=True):
        logger.info(f"Building stack with {len(self.networks)} design networks")
        logger.info(f"Design network names: {[n.name for n in self.networks]}")
        stack = ComputeStack(networks=self.networks)
        logger.info(f"Stack after creation has {len(stack.networks)} networks")

        # Deep copy compute_config to avoid mutating the original model
        # (mutation would change model.signature since it's computed from compute_config)
        compute_config = model.compute_config.model_copy(deep=True)

        if unlock_ratios:
            assert compute_config is not None
            assert compute_config.node_functions is not None

            compute_config.node_functions["aggregation"] = encode_function(
                partial(nd.aggregation, random_init=True)
            )

        stack.build(compute_config, enable_tu_masking=self.enable_tu_masking)

        logger.info(
            f"Stack built: {stack.get_nb_networks()} networks, "
            f"{stack.get_nb_inputs()} inputs, {stack.get_nb_outputs()} outputs"
        )
        logger.info(f"Stack network names after build: {[n.name for n in stack.networks]}")
        return stack

    @property
    def n_targets(self):
        return len(self.targets)


def initialize_params(
    stack,
    n_replicates,
    n_targets,
    shared_params,
    key,
    n_tus: int = 0,
    n_networks: int = 1,
    tu_log_alpha_init_mean: float = 2.0,
    tu_log_alpha_init_std: float = 0.5,
):
    """Initialize parameters for design optimization.

    When TU masking is enabled (n_tus > 0), tu_log_alpha has shape
    (n_replicates, n_targets, n_networks, n_tus) to allow independent
    TU masking per network/scaffold.
    """
    tu_key, init_key = jax.random.split(key)

    def init_single(k):
        params = stack.init(k)
        _, nonshared = params.filter_by_tag(["shared"])
        return ParameterTree.merge(shared_params, nonshared)

    def init_target_params(k):
        params = vmap(init_single)(jax.random.split(k, n_targets))
        return params

    params = vmap(init_target_params)(jax.random.split(init_key, n_replicates))

    if n_tus > 0:
        assert n_networks > 0, f"n_tus={n_tus} but n_networks={n_networks}"
        expected_shape = (n_replicates, n_targets, n_networks, n_tus)
        tu_log_alpha = tu_log_alpha_init_mean + tu_log_alpha_init_std * jax.random.normal(
            tu_key, shape=expected_shape
        )
        assert tu_log_alpha.shape == expected_shape, (
            f"tu_log_alpha shape {tu_log_alpha.shape} != {expected_shape}"
        )
        params.at(TU_LOG_ALPHA_PATH, tu_log_alpha, overwrite=None)
        logger.info(
            f"Initialized TU log_alpha: {n_tus} TUs × {n_networks} networks (mean={tu_log_alpha_init_mean}, std={tu_log_alpha_init_std})"
        )

    return params


class DesignConfig(DesignOptimConfig):
    loss_function: EncodedPartialFunction = Field(default=distance_loss)
    n_replicates: int = 4
    keep_in_history: list[str] | Literal["all"] = "all"
    tu_log_alpha_init_mean: float = 2.0
    tu_log_alpha_init_std: float = 0.5

    pluggable_optimizer: Any = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def uses_pluggable_optimizer(self) -> bool:
        """Check if using new-style pluggable optimizer."""
        return self.pluggable_optimizer is not None

    @property
    def optimizer(self) -> Any:
        """Return optax optimizer for backward compat, or pluggable if set."""
        if self.pluggable_optimizer is not None:
            return self.pluggable_optimizer
        return super().optimizer

    def get_pluggable_optimizer(self) -> Any:
        """Get pluggable optimizer, creating from optimizer_stack if not set."""
        if self.pluggable_optimizer is not None:
            return self.pluggable_optimizer
        from biocomp.designoptim import GradientDescentOptimizer

        total_steps = int(self.n_epochs * max(1, self.n_batches_per_epoch // self.batches_per_step))
        return GradientDescentOptimizer(
            optimizer_stack=self.optimizer_stack,
            n_steps=total_steps,
            sanitize_grads=True,
        )


def assert_tree_shape(tree, expected_shape, only_first_dims=True):
    N_DIMS = len(expected_shape)

    def check_shape(x):
        if isinstance(x, jax.Array):
            assert_that(x.shape[:N_DIMS] if only_first_dims else x.shape).is_equal_to(
                expected_shape
            )
        return x

    jax.tree.map(check_shape, tree)


def get_topk_replicate_network_pairs(
    losses: jax.Array,
    dmanager: DesignManager,
    dconf: DesignConfig,
    k: int = 1,
) -> list[list[tuple[int, int, float]]]:
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
    topk: list[list[tuple[int, int, float]]],
    yhatdep: jax.Array | None = None,
    n_eval_samples: int | None = None,
    save_dir: Path | None = None,
    show_difference: bool = False,
    plot_top_k: int | None = None,
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

    n_networks = len(dmanager.networks)
    assert_that(xraw).has_shape(
        (n_networks, dconf.n_replicates, xraw.shape[2], dmanager.n_targets, 2)
    )
    assert_that(yraw).has_shape(
        (n_networks, dconf.n_replicates, yraw.shape[2], dmanager.n_targets, 1)
    )

    if plot_top_k is None:
        plot_top_k = 1

    for tid, target in enumerate(dmanager.targets):
        n_to_plot = min(plot_top_k, len(topk[tid]))

        for rank in range(n_to_plot):
            rep_id, net_id, loss_val = topk[tid][rank]

            x_target = xraw[net_id, rep_id, :n_eval_samples, tid]
            y_target = yraw[net_id, rep_id, :n_eval_samples, tid, 0]

            assert_that(x_target).has_shape((n_eval_samples, 2))
            assert_that(y_target).has_shape((n_eval_samples,))

            nax = 3 if show_difference else 2
            fig, axes = plt.subplots(1, nax, figsize=(nax * 5, 5), dpi=100)

            sc1 = axes[0].scatter(
                x_target[:, 0], x_target[:, 1], c=y_target, cmap=DEFAULT_CMAP_NAME, s=5, alpha=0.7
            )
            axes[0].set_title("Target")
            axes[0].set_aspect("equal")
            plt.colorbar(sc1, ax=axes[0])

            if yhatdep is not None:
                yhat_target = yhatdep[rep_id, :n_eval_samples, tid, net_id]
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


RATIO_PRUNE_THRESHOLD = 1.0 / 120.0


def normalize_ratios_prune(current_ratios, threshold=RATIO_PRUNE_THRESHOLD, eps=1e-12):
    A = jnp.abs(current_ratios)
    if A.ndim == 0:
        return A
    if A.ndim == 1:
        A = A[None, :]
        m = jnp.maximum(jnp.max(A, axis=1, keepdims=True), eps)
        norm = A / m
        return jnp.where(norm >= threshold, norm, 0.0).squeeze(0)
    if A.ndim > 2:
        orig_shape = A.shape
        A = A.reshape(-1, A.shape[-1])
        m = jnp.maximum(jnp.max(A, axis=1, keepdims=True), eps)
        norm = A / m
        return jnp.where(norm >= threshold, norm, 0.0).reshape(orig_shape)
    m = jnp.maximum(jnp.max(A, axis=1, keepdims=True), eps)
    norm = A / m
    return jnp.where(norm >= threshold, norm, 0.0)


def get_ratio_paths_and_sources(params):
    from biocomp.parameters import isArrayRef

    direct_paths, aref_sources, aref_count = [], set(), 0
    for path, value in params.data.iter_leaves():
        path_str = str(path)
        if "ratio" in path_str and "inverse" not in path_str:
            if isArrayRef(value):
                aref_count += 1
                aref_sources.update(str(sp) for sp in value.paths)
            else:
                direct_paths.append(path)
    if aref_count:
        logger.info(f"Found {aref_count} ArrayRef ratio paths -> {len(aref_sources)} source arrays")
    return direct_paths, list(aref_sources)


def get_ratio_paths(params):
    return get_ratio_paths_and_sources(params)[0]


def normalize_ratio_source_arrays(params, source_paths, normalize_func):
    from biocomp.parameters import flatten_PTree, unflatten_PTree, ParamPath, ParameterTree

    source_set = set(source_paths)
    flat_leaves, (keys, read_only) = flatten_PTree(params.data)
    new_leaves = list(flat_leaves)
    for i, key in enumerate(keys):
        if isinstance(key, ParamPath) and str(key) in source_set and flat_leaves[i] is not None:
            new_leaves[i] = normalize_func(flat_leaves[i])
    new_data = unflatten_PTree((keys, read_only), tuple(new_leaves))
    return ParameterTree(
        data=new_data, tags=params.tags, tagnames=params.tagnames, read_only=params.read_only
    )


def _start_pluggable(
    dmanager: DesignManager,
    dconf: DesignConfig,
    model: BiocompModel,
    loggers: list[tuple[int, Callable]] | None = None,
    async_handler=None,
):
    import time
    from .designcodec import GenomeCodec

    timings = {}
    t_total = time.perf_counter()

    logger.info("=" * 60)
    logger.info("DESIGN OPTIMIZATION (PLUGGABLE OPTIMIZER)")
    logger.info("=" * 60)

    pkey, bkey, loop_key = jax.random.split(dconf.seed_key, 3)
    optimizer = dconf.pluggable_optimizer

    # Phase 1: Build compute stack
    t0 = time.perf_counter()
    logger.info("[1/6] Building compute stack...")
    stack = dmanager.build_stack(model)
    timings["stack_build"] = time.perf_counter() - t0
    logger.info(f"  -> Stack built in {timings['stack_build']:.2f}s")

    # Phase 2: Initialize parameters (single replicate for pluggable optimizer)
    t1 = time.perf_counter()
    logger.info("[2/6] Initializing parameters...")
    n_tus = dmanager.n_tus if dmanager.enable_tu_masking else 0
    n_networks = len(dmanager.networks)

    # Use n_replicates=1 for pluggable optimizer - squeeze later
    initial_params = initialize_params(
        stack,
        n_replicates=1,
        n_targets=dmanager.n_targets,
        shared_params=model.shared_params,
        key=pkey,
        n_tus=n_tus,
        n_networks=n_networks,
        tu_log_alpha_init_mean=dconf.tu_log_alpha_init_mean,
        tu_log_alpha_init_std=dconf.tu_log_alpha_init_std,
    )
    # Squeeze replicate dim: (1, n_targets, ...) -> (n_targets, ...)
    initial_params = jax.tree.map(lambda x: x.squeeze(0), initial_params)
    timings["param_init"] = time.perf_counter() - t1
    logger.info(f"  -> Parameters initialized in {timings['param_init']:.2f}s")

    # Phase 3: Create codec for param encoding
    t2 = time.perf_counter()
    logger.info("[3/6] Creating parameter codec...")
    static, dynamic = initial_params.filter_by_tag(["non_grad", "shared"])
    codec = GenomeCodec.from_params(initial_params, static_tags=("shared", "non_grad"))
    flat_params = codec.encode(initial_params)
    logger.info(f"  Genome dimension: {codec.param_dim}")
    timings["codec"] = time.perf_counter() - t2
    logger.info(f"  -> Codec created in {timings['codec']:.2f}s")

    # Phase 4: Create loss function and objective
    t3 = time.perf_counter()
    logger.info("[4/6] Creating objective function...")
    num_z = static["global/number_of_random_variables"]
    num_z = (dmanager.n_targets, int(num_z.ravel()[0].squeeze()))
    direct_ratio_paths, source_ratio_paths = get_ratio_paths_and_sources(initial_params)

    loss_fn = dconf.loss_function.get_impl()(
        stack, dconf, dmanager, num_z=num_z, ratio_paths=direct_ratio_paths
    )

    effective_batch_size = dconf.batch_size
    if dmanager.is_lattice_mode:
        xres, yres = dmanager.grid_resolution
        effective_batch_size *= xres * yres

    xbatches_list, ybatches_list = dmanager.get_samples(
        (len(dmanager.networks), 1, 1, 1, dconf.batch_size), bkey
    )
    x_samples = jnp.concatenate(xbatches_list, axis=-1)
    x_samples = x_samples.reshape(effective_batch_size, dmanager.n_targets, -1)
    y_samples = ybatches_list[0].reshape(effective_batch_size, dmanager.n_targets, -1)

    # CMA-ES requires deterministic objective: fix z to mean (0.5) and use fixed RNG key
    fixed_z = jnp.full((x_samples.shape[0], *num_z), 0.5)
    fixed_key = jax.random.key(42)

    def single_objective(flat_genome: jnp.ndarray, step: jnp.ndarray) -> float:
        params = codec.decode(flat_genome, apply_constraints=True)
        static_p, dynamic_p = params.filter_by_tag(["non_grad", "shared"])
        loss, _ = loss_fn(dynamic_p, static_p, x_samples, y_samples, fixed_z, fixed_key, step)
        return loss

    vmapped_objective = jax.vmap(single_objective, in_axes=(0, None))

    logger.info("  Compiling vmapped objective (AOT)...")
    t_compile = time.perf_counter()
    compiled_pop_objective = jax.jit(vmapped_objective)
    dummy_pop = jnp.zeros((optimizer._get_pop_size(codec.param_dim), codec.param_dim))
    _ = compiled_pop_objective(dummy_pop, jnp.array(0, dtype=jnp.int32)).block_until_ready()
    compile_time = time.perf_counter() - t_compile
    logger.info(f"  -> Compiled in {compile_time:.2f}s")

    def objective_fn_init(flat_genome: jnp.ndarray) -> float:
        return single_objective(flat_genome, jnp.array(0, dtype=jnp.int32))

    def get_yhatdep(flat_genome: jnp.ndarray, step: jnp.ndarray) -> jnp.ndarray:
        params = codec.decode(flat_genome, apply_constraints=True)
        static_p, dynamic_p = params.filter_by_tag(["non_grad", "shared"])
        _, aux = loss_fn(dynamic_p, static_p, x_samples, y_samples, fixed_z, fixed_key, step)
        return aux.get("yhatdep")

    compiled_get_yhatdep = jax.jit(get_yhatdep)

    timings["objective"] = time.perf_counter() - t3
    logger.info(f"  -> Objective created in {timings['objective']:.2f}s")

    # Phase 5: Initialize optimizer
    t4 = time.perf_counter()
    logger.info("[5/6] Initializing optimizer...")
    init_key, opt_key = jax.random.split(loop_key)
    opt_state = optimizer.init(init_key, flat_params, objective_fn_init)
    logger.info(f"  Initial loss: {float(opt_state.best_loss):.6f}")
    timings["opt_init"] = time.perf_counter() - t4
    logger.info(f"  -> Optimizer initialized in {timings['opt_init']:.2f}s")

    # Phase 6: Optimization loop
    logger.info("=" * 60)
    logger.info("STARTING OPTIMIZATION LOOP")
    logger.info("=" * 60)

    loss_history = []
    step_history = []
    pbar = tqdm(desc="Optimizing", unit="step")

    while not optimizer.should_stop(opt_state):
        opt_key, step_key = jax.random.split(opt_key)
        current_step = jnp.array(int(opt_state.step), dtype=jnp.int32)
        # Pass the pre-compiled population objective and current step
        opt_state, metrics = optimizer.step(
            opt_state, step_key, compiled_pop_objective, current_step
        )

        loss_history.append(float(opt_state.best_loss))
        step_history.append({k: float(v) if hasattr(v, "item") else v for k, v in metrics.items()})

        pbar.update(1)
        pbar.set_postfix(loss=f"{float(opt_state.best_loss):.4f}")

        # Call loggers - pass current step's metrics as step_history dict
        if loggers:
            should_log = any(
                int(opt_state.step) % p == 0 or p == -1 for p, _ in loggers
            )
            yhatdep_arr = None
            if should_log:
                yhatdep_arr = compiled_get_yhatdep(opt_state.best_params, current_step)
            current_step_data = {
                "loss": [[float(opt_state.best_loss)]],
                "yhatdep": yhatdep_arr,
                **metrics,
            }
            for period, callback in loggers:
                if int(opt_state.step) % period == 0 or period == -1:
                    callback(
                        int(opt_state.step),
                        None,
                        step_history=current_step_data,
                        stack=stack,
                    )

    pbar.close()

    timings["total"] = time.perf_counter() - t_total
    logger.info("=" * 60)
    logger.info(f"OPTIMIZATION COMPLETE in {timings['total']:.2f}s")
    logger.info(f"  Final loss: {float(opt_state.best_loss):.6f}")
    logger.info(f"  Total steps: {int(opt_state.step)}")
    logger.info("=" * 60)

    # Decode final params and add replicate dimension for compatibility
    final_params = codec.decode(opt_state.best_params, apply_constraints=True)
    final_params = jax.tree.map(lambda x: x[None, ...], final_params)

    return final_params, loss_history, step_history


def start(
    dmanager: DesignManager,
    dconf: DesignConfig,
    model: BiocompModel,
    loggers: list[tuple[int, Callable]] | None = None,
    async_handler=None,
):
    # Dispatch to pluggable optimizer loop if configured
    if dconf.uses_pluggable_optimizer:
        logger.info("Using pluggable optimizer: %s", type(dconf.pluggable_optimizer).__name__)
        return _start_pluggable(dmanager, dconf, model, loggers, async_handler)

    import time

    timings = {}
    t_total = time.perf_counter()

    logger.info("=" * 60)
    logger.info("DESIGN OPTIMIZATION - INITIALIZATION PHASE")
    logger.info("=" * 60)

    pkey, bkey, loop_key = jax.random.split(dconf.seed_key, 3)

    # Phase 1: Build compute stack
    t0 = time.perf_counter()
    logger.info("[1/5] Building compute stack...")
    stack = dmanager.build_stack(model)
    timings["stack_build"] = time.perf_counter() - t0
    logger.info(f"  -> Stack built in {timings['stack_build']:.2f}s")

    # Phase 2: Initialize parameters
    t1 = time.perf_counter()
    logger.info("[2/5] Initializing parameters...")
    n_tus = dmanager.n_tus if dmanager.enable_tu_masking else 0
    n_networks = len(dmanager.networks)
    initial_params = initialize_params(
        stack,
        dconf.n_replicates,
        dmanager.n_targets,
        model.shared_params,
        pkey,
        n_tus=n_tus,
        n_networks=n_networks,
        tu_log_alpha_init_mean=dconf.tu_log_alpha_init_mean,
        tu_log_alpha_init_std=dconf.tu_log_alpha_init_std,
    )
    assert_tree_shape(initial_params, (dconf.n_replicates, dmanager.n_targets))
    timings["param_init"] = time.perf_counter() - t1
    logger.info(f"  -> Parameters initialized in {timings['param_init']:.2f}s")

    # Phase 3: Initialize optimizer state
    t2 = time.perf_counter()
    logger.info("[3/5] Initializing optimizer state...")
    static, dynamic = initial_params.filter_by_tag(["non_grad", "shared"])
    initial_optimizer_state = vmap(vmap(dconf.optimizer.init))(dynamic)
    timings["opt_init"] = time.perf_counter() - t2
    logger.info(f"  -> Optimizer state initialized in {timings['opt_init']:.2f}s")

    # -- get data --
    num_z = static["global/number_of_random_variables"]
    assert_that(num_z.shape[0]).is_equal_to(dconf.n_replicates)
    assert_that(jnp.all(num_z == num_z[0])).is_true()
    num_z = (dmanager.n_targets, int(num_z.ravel()[0].squeeze()))

    steps_per_epoch = max(1, dconf.n_batches_per_epoch // dconf.batches_per_step)
    total_steps = int(dconf.n_epochs * steps_per_epoch)

    logger.info(
        f"  Config: {total_steps} total steps, {steps_per_epoch} steps/epoch, "
        f"batch_size={dconf.batch_size}, batches_per_step={dconf.batches_per_step}"
    )
    assert_that(total_steps).is_greater_than(0)

    n_networks = stack.get_nb_networks()

    # Phase 4: Generate samples
    t3 = time.perf_counter()
    logger.info("[4/5] Generating training samples...")
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
    timings["sample_gen"] = time.perf_counter() - t3
    logger.info(f"  -> Samples generated in {timings['sample_gen']:.2f}s")

    effective_batch_size = dconf.batch_size
    if dmanager.is_lattice_mode:
        xres, yres = dmanager.grid_resolution
        effective_batch_size *= xres * yres

    n_design_inputs = 2 * len(dmanager.networks)

    logger.info(
        f"  Data: {len(dmanager.networks)} design networks, "
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

    # Phase 5: Create loss and step functions
    t4 = time.perf_counter()
    logger.info("[5/5] Creating loss and step functions...")
    direct_ratio_paths, source_ratio_paths = get_ratio_paths_and_sources(initial_params)
    ratio_paths = direct_ratio_paths
    logger.debug(
        f"Ratio normalization: {len(direct_ratio_paths)} direct + {len(source_ratio_paths)} ArrayRef source paths"
    )

    def norm_ratios_hook(params, *a, **kw):
        # First, normalize direct ratio paths (non-ArrayRef)
        if direct_ratio_paths:
            params = params.update_leaves_by_path(direct_ratio_paths, normalize_ratios_prune)
        # Then, normalize source arrays that back ArrayRef ratios
        if source_ratio_paths:
            params = normalize_ratio_source_arrays(
                params, source_ratio_paths, normalize_ratios_prune
            )
        # clamp tu_log_alpha (hard_concrete has soft clamp, this is a safety bound)
        if TU_LOG_ALPHA_PATH in params:
            params = params.update_leaves_by_path(
                [TU_LOG_ALPHA_PATH], lambda x: jnp.clip(x, LOG_ALPHA_MIN, LOG_ALPHA_MAX)
            )
        return params

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
        sanitize_grads=True,
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

    timings["loss_step_fn"] = time.perf_counter() - t4
    logger.info(f"  -> Loss/step functions created in {timings['loss_step_fn']:.2f}s")

    # Summary of initialization
    timings["total_init"] = time.perf_counter() - t_total
    logger.info("-" * 60)
    logger.info(f"INITIALIZATION COMPLETE in {timings['total_init']:.2f}s")
    logger.info(
        f"  Stack build:     {timings['stack_build']:.2f}s ({timings['stack_build'] / timings['total_init'] * 100:.1f}%)"
    )
    logger.info(
        f"  Param init:      {timings['param_init']:.2f}s ({timings['param_init'] / timings['total_init'] * 100:.1f}%)"
    )
    logger.info(
        f"  Optimizer init:  {timings['opt_init']:.2f}s ({timings['opt_init'] / timings['total_init'] * 100:.1f}%)"
    )
    logger.info(
        f"  Sample gen:      {timings['sample_gen']:.2f}s ({timings['sample_gen'] / timings['total_init'] * 100:.1f}%)"
    )
    logger.info(
        f"  Loss/step fn:    {timings['loss_step_fn']:.2f}s ({timings['loss_step_fn'] / timings['total_init'] * 100:.1f}%)"
    )
    logger.info("=" * 60)
    logger.info("STARTING OPTIMIZATION LOOP")
    logger.info("=" * 60)

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


def sample_for_evaluation(
    dmanager: DesignManager,
    dconf: DesignConfig,
    final_params: ParameterTree,
    n_eval_samples: int,
    key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample evaluation data. Returns (xraw, yraw) with shapes (n_networks, n_replicates, n_samples, n_targets, 2/1)."""
    n_networks, n_replicates, n_targets = (
        len(dmanager.networks),
        dconf.n_replicates,
        dmanager.n_targets,
    )
    seed = int(jax.random.key_data(key)[0]) % (2**31)
    xlist, ylist = dmanager._get_uniform_samples((n_networks, n_replicates, n_eval_samples), seed)
    xraw, yraw = jnp.stack(xlist, axis=0), jnp.stack(ylist, axis=0)
    assert xraw.shape == (n_networks, n_replicates, n_eval_samples, n_targets, 2)
    assert yraw.shape == (n_networks, n_replicates, n_eval_samples, n_targets, 1)
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
) -> tuple[jnp.ndarray | None, jnp.ndarray]:
    """Evaluate design quality. Returns (predictions, losses) where losses has shape (n_replicates, n_targets, n_networks).

    CRITICAL: This function MUST apply TU masks to be consistent with training.
    Without TU masks, evaluation loss will NOT reflect actual design performance.
    """
    stack = dmanager.build_stack(model, unlock_ratios=False)
    n_networks, n_replicates, n_targets, n_samples = (
        len(dmanager.networks),
        dconf.n_replicates,
        dmanager.n_targets,
        xraw.shape[2],
    )
    logger.info(f"Evaluating: {n_replicates} reps × {n_targets} targets × {n_samples} samples")

    num_z_val = int(final_params["global/number_of_random_variables"][0, 0].squeeze())
    dep_mask = stack.get_dependent_output_mask()
    x_combined = xraw.transpose(1, 2, 3, 0, 4).reshape(n_replicates, n_samples, n_targets, -1)
    y_combined = yraw[0]

    has_tu_masking = TU_LOG_ALPHA_PATH in final_params

    if has_tu_masking:
        tu_log_alpha_full = final_params[TU_LOG_ALPHA_PATH]
        assert tu_log_alpha_full.ndim == 4, (
            f"EVALUATE BUG: tu_log_alpha should be 4D (n_reps, n_targets, n_networks, n_tus), "
            f"got {tu_log_alpha_full.ndim}D with shape {tu_log_alpha_full.shape}"
        )
        assert tu_log_alpha_full.shape[0] >= n_replicates, (
            f"EVALUATE BUG: tu_log_alpha has {tu_log_alpha_full.shape[0]} replicates "
            f"but n_replicates={n_replicates}"
        )
        assert tu_log_alpha_full.shape[1] >= n_targets, (
            f"EVALUATE BUG: tu_log_alpha has {tu_log_alpha_full.shape[1]} targets "
            f"but n_targets={n_targets}"
        )
        logger.debug(f"TU masking enabled: {tu_log_alpha_full.shape[-1]} TUs")

    all_losses, all_predictions = [], [] if store_predictions else None

    def apply_with_tu_mask(params, x_batch, z_batch, keys, tu_mask):
        def apply_single(x, z, k):
            return stack.apply(params, x, z, k, tu_enabled_random_vars=tu_mask)

        return vmap(apply_single)(x_batch, z_batch, keys)

    apply_batched = jax.jit(apply_with_tu_mask)
    pbar = tqdm(total=n_replicates * n_targets, desc="Evaluating", unit="rep×tgt")

    for rep_idx in range(n_replicates):
        rep_losses, rep_preds = [], [] if store_predictions else None
        for tid in range(n_targets):
            rep_params = jax.tree.map(lambda x: x[rep_idx, tid], final_params)
            x_slice, y_slice = x_combined[rep_idx, :, tid, :], y_combined[rep_idx, :, tid, :]

            # compute deterministic TU mask for evaluation
            # CRITICAL: This ensures evaluation uses same TU masking as training
            if has_tu_masking:
                tu_log_alpha = rep_params[TU_LOG_ALPHA_PATH]
                assert tu_log_alpha.ndim == 2, (
                    f"EVALUATE BUG: sliced tu_log_alpha should be 2D (n_networks, n_tus), "
                    f"got {tu_log_alpha.ndim}D with shape {tu_log_alpha.shape}"
                )
                assert tu_log_alpha.shape[0] == n_networks, (
                    f"EVALUATE BUG: tu_log_alpha has {tu_log_alpha.shape[0]} networks "
                    f"but stack has {n_networks}"
                )
                # use sigmoid(log_alpha) as the "uniform sample" which gives deterministic output
                # sigmoid(log_alpha) >= 0.5 means TU is enabled (same as get_final_mask threshold)
                tu_mask = jax.nn.sigmoid(tu_log_alpha)
                assert tu_mask.shape == tu_log_alpha.shape, "tu_mask shape mismatch"
            else:
                tu_mask = None

            yhats = []
            for start in range(0, n_samples, max_eval_size):
                end = min(start + max_eval_size, n_samples)
                z_batch = jax.random.uniform(key, (end - start, num_z_val))
                yhat, _ = apply_batched(
                    rep_params,
                    x_slice[start:end],
                    z_batch,
                    jax.random.split(key, end - start),
                    tu_mask,
                )
                yhats.append(yhat)

            yhat_dep = jnp.compress(dep_mask, jnp.concatenate(yhats, axis=0), axis=-1)
            if store_predictions:
                rep_preds.append(yhat_dep)
            rep_losses.append(
                jnp.mean((yhat_dep - jnp.tile(y_slice, (1, n_networks))) ** 2, axis=0).tolist()
            )
            pbar.update(1)

        all_losses.append(rep_losses)
        if store_predictions:
            all_predictions.append(jnp.stack(rep_preds, axis=0))

    pbar.close()
    losses = jnp.array(all_losses)
    logger.info(
        f"Evaluation complete. Loss: [{float(losses.min()):.4f}, {float(losses.max()):.4f}]"
    )

    if store_predictions:
        return jnp.stack(all_predictions, axis=0).transpose(0, 2, 1, 3), losses
    return None, losses


def compute_baseline_loss(
    dmanager: DesignManager,
    model,  # BiocompModel
    n_samples: int = 1000,
    seed: int = 42,
    max_batch_size: int = 200,
) -> dict:
    """Compute baseline loss for DataTargets with original_network."""
    try:
        from biocomptools.modelmodel import NetworkModel
    except ImportError:
        logger.warning("biocomptools not available, cannot compute baseline")
        return {}

    results, rng = {}, np.random.default_rng(seed)

    for tid, target in enumerate(dmanager.targets):
        target_name = target.name or f"target_{tid}"
        if not isinstance(target, DataTarget) or target.original_network is None:
            results[target_name] = {"has_original_network": False}
            continue

        original_network = target.original_network
        indices = rng.choice(len(target.X), size=min(n_samples, len(target.X)), replace=False)
        X_sample, Y_sample = np.asarray(target.X[indices]), np.asarray(target.Y[indices])
        if Y_sample.ndim == 1:
            Y_sample = Y_sample[:, None]

        nm = NetworkModel(
            model=model, network=original_network, max_points_per_batch=max_batch_size
        )
        yhat, _ = nm.predict(X_sample)
        yhat_mean = np.mean(yhat, axis=-1, keepdims=True) if yhat.shape[-1] > 1 else yhat

        model_loss = float(np.mean((yhat_mean - Y_sample) ** 2))

        results[target_name] = {
            "has_original_network": True,
            "model_prediction_loss": model_loss,
            "original_network_name": original_network.name,
            "n_samples": len(X_sample),
        }
        logger.info(f"Baseline '{target_name}': loss={model_loss:.6f} ({original_network.name})")

    return results
