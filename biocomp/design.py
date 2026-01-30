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
from biocomp.config import BIOCOMP_CONSTANTS
import biocomp.nodes as nd
from biocomp.network import Network
from .parameters import ParameterTree
from biocomp.logging_config import get_logger
from biocomp.optimutils import make_training_step, per_replicate_step, optimize, DesignOptimConfig
from biocomp.tumasking import PROTECTED_TU_MASK_PATH

if TYPE_CHECKING:
    from biocomptools.modelmodel import BiocompModel
from biocomp.designloss import (  # noqa: F401 - re-exported
    grid_distance_loss,
    simse_loss,
    zncc_loss,
    gradient_magnitude_loss,
    lncc_grid_loss,
    sinkhorn_divergence_conv,
    spectral_loss,
    proj_nonneg_ste,
    _sanitize,
)
from biocomp.tumasking import (
    build_tu_id_mapping,
    TU_LOG_ALPHA_PATH,
    LOG_ALPHA_MIN,
    LOG_ALPHA_MAX,
    LATENT_TU_Z_PATH,
    LATENT_TU_W1_PATH,
    LATENT_TU_B1_PATH,
    LATENT_TU_W2_PATH,
    LATENT_TU_B2_PATH,
)
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


class _PhaseTimer:
    """Minimal helper for timing optimization phases."""

    def __init__(self):
        import time

        self._time = time
        self._timings: dict[str, float] = {}
        self._t0 = time.perf_counter()
        self._phase_start: float = time.perf_counter()

    def start(self, name: str, msg: str):
        logger.info(msg)
        self._phase_start = self._time.perf_counter()

    def end(self, name: str):
        self._timings[name] = self._time.perf_counter() - self._phase_start
        logger.info(f"  -> {self._timings[name]:.2f}s")

    def total(self) -> float:
        return self._time.perf_counter() - self._t0

    def summary(self):
        total = self.total()
        for name, t in self._timings.items():
            logger.info(f"  {name:15s} {t:.2f}s ({t / total * 100:.1f}%)")


class DesignManager(BaseModel):
    """Handles loading and sampling of 2d design target data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    targets: list[TargetUnion]
    networks: list[Network]
    sampling: SamplingConfigUnion = Field(default_factory=LatticeSampling, discriminator="strategy")

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
        assert self._tu_ids is not None
        return len(self._tu_ids)

    @property
    def tu_id_to_idx(self) -> dict[str, int]:
        """Mapping from TU ID string to index."""
        self._ensure_tu_mapping()
        assert self._tu_id_to_idx is not None
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
    ) -> tuple[list[jax.Array], list[jax.Array]]:
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
        assert len(requested_shape) >= 1, (
            "samples must have at least 2 elements: (n_networks, batch_size, ...)"
        )
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
                    latent_out = getattr(target, "latent_out", None)
                    if latent_out is not None:
                        Y_grid = np.clip(Y_grid, latent_out[0], latent_out[1])
                xsamples.append(X)
                ysamples.append(Y_grid)

            xsamples_stacked = jnp.stack(xsamples, axis=1)
            ysamples_stacked = jnp.stack(ysamples, axis=1)

            n_pts = n * yres * xres
            assert_that(xsamples_stacked.shape).is_equal_to((n_pts, len(self.targets), 2))
            assert_that(ysamples_stacked.shape).is_equal_to((n, len(self.targets), yres, xres))

            ysamples_flat = ysamples_stacked.transpose(0, 2, 3, 1).reshape(
                n_pts, len(self.targets), 1
            )

            new_batch_shape = requested_shape[:-1] + (requested_shape[-1] * yres * xres,)
            xsamples_reshaped = xsamples_stacked.reshape(*new_batch_shape, len(self.targets), 2)
            ysamples_reshaped = ysamples_flat.reshape(*new_batch_shape, len(self.targets), 1)

            all_xsamples.append(xsamples_reshaped)
            all_ysamples.append(ysamples_reshaped)

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

    def build_stack(
        self,
        model: BiocompModel,
        unlock_ratios: bool = True,
        use_latent_ratios: bool = False,
        latent_dim: int = 8,
        latent_hidden_dim: int = 16,
        auto_lock_topology_tus: bool = True,
    ) -> ComputeStack:
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
                partial(
                    nd.aggregation,
                    random_init=True,
                    use_latent_ratios=use_latent_ratios,
                    latent_dim=latent_dim,
                    latent_hidden_dim=latent_hidden_dim,
                )
            )

        stack.build(
            compute_config,
            enable_tu_masking=self.enable_tu_masking,
            auto_lock_topology_tus=auto_lock_topology_tus,
        )

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
    use_latent_tu_masking: bool = False,
    latent_tu_dim: int = 16,
    latent_tu_hidden_dim: int = 32,
    no_masking_tu_ids: set[str] | None = None,
    tu_id_to_idx: dict[str, int] | None = None,
):
    """Initialize design params. If use_latent_tu_masking, creates MLP params instead of direct log_alpha."""
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

        if use_latent_tu_masking:
            k_z, k_w1, k_w2, tu_key = jax.random.split(tu_key, 4)

            init_log_alpha = tu_log_alpha_init_mean + tu_log_alpha_init_std * jax.random.normal(
                tu_key, shape=(n_replicates, n_targets, n_networks, n_tus)
            )
            latent_z = (
                jax.random.normal(k_z, shape=(n_replicates, n_targets, n_networks, latent_tu_dim))
                * 0.1
            )
            W1 = jax.random.normal(
                k_w1,
                shape=(n_replicates, n_targets, n_networks, latent_tu_hidden_dim, latent_tu_dim),
            ) * jnp.sqrt(2.0 / latent_tu_dim)
            b1 = jnp.zeros((n_replicates, n_targets, n_networks, latent_tu_hidden_dim))
            W2 = (
                jax.random.normal(
                    k_w2, shape=(n_replicates, n_targets, n_networks, n_tus, latent_tu_hidden_dim)
                )
                * jnp.sqrt(2.0 / latent_tu_hidden_dim)
                * 0.1
            )
            b2 = init_log_alpha  # MLP(0) ≈ init_log_alpha

            params.at(LATENT_TU_Z_PATH, latent_z, overwrite=None)
            params.at(LATENT_TU_W1_PATH, W1, overwrite=None)
            params.at(LATENT_TU_B1_PATH, b1, overwrite=None)
            params.at(LATENT_TU_W2_PATH, W2, overwrite=None)
            params.at(LATENT_TU_B2_PATH, b2, overwrite=None)
            logger.info(
                f"Initialized latent TU masking: {n_tus} TUs × {n_networks} networks, "
                f"latent_dim={latent_tu_dim}, hidden_dim={latent_tu_hidden_dim}"
            )
        else:
            expected_shape = (n_replicates, n_targets, n_networks, n_tus)
            tu_log_alpha = tu_log_alpha_init_mean + tu_log_alpha_init_std * jax.random.normal(
                tu_key, shape=expected_shape
            )
            assert tu_log_alpha.shape == expected_shape, (
                f"tu_log_alpha shape {tu_log_alpha.shape} != {expected_shape}"
            )
            params.at(TU_LOG_ALPHA_PATH, tu_log_alpha, overwrite=None)
            logger.info(
                f"Initialized TU log_alpha: {n_tus} TUs × {n_networks} networks "
                f"(mean={tu_log_alpha_init_mean}, std={tu_log_alpha_init_std})"
            )

        protected_tu_mask_1d = jnp.zeros(n_tus, dtype=bool)
        if no_masking_tu_ids and tu_id_to_idx:
            protected_indices = [
                tu_id_to_idx[tu_id] for tu_id in no_masking_tu_ids if tu_id in tu_id_to_idx
            ]
            if protected_indices:
                protected_tu_mask_1d = protected_tu_mask_1d.at[jnp.array(protected_indices)].set(
                    True
                )
                idx_arr = jnp.array(protected_indices)
                if use_latent_tu_masking:
                    b2_val = params[LATENT_TU_B2_PATH]
                    b2_val = b2_val.at[..., idx_arr].set(10.0)
                    params.at(LATENT_TU_B2_PATH, b2_val, overwrite=True)
                else:
                    tu_log_alpha_val = params[TU_LOG_ALPHA_PATH]
                    tu_log_alpha_val = tu_log_alpha_val.at[..., idx_arr].set(10.0)
                    params.at(TU_LOG_ALPHA_PATH, tu_log_alpha_val, overwrite=True)
                logger.info(f"Protected {len(protected_indices)} TUs from masking (log_alpha=10.0)")
        protected_tu_mask = jnp.tile(
            protected_tu_mask_1d[None, None, :], (n_replicates, n_targets, 1)
        )
        params.at(PROTECTED_TU_MASK_PATH, protected_tu_mask, overwrite=None, tags=["non_grad"])

    return params


class DesignConfig(DesignOptimConfig):
    loss_function: EncodedPartialFunction = Field(default=grid_distance_loss)  # pyright: ignore[reportAssignmentType]
    n_replicates: int = 4
    keep_in_history: list[str] | Literal["all"] = "all"
    tu_log_alpha_init_mean: float = 2.0
    tu_log_alpha_init_std: float = 0.5

    use_latent_ratios: bool = False
    latent_dim: int = 8
    latent_hidden_dim: int = 16

    enable_tu_masking: bool = False
    use_latent_tu_masking: bool = False
    latent_tu_dim: int = 16
    latent_tu_hidden_dim: int = 32
    auto_lock_topology_tus: bool = True  # auto-detect TUs whose masking would change topology

    # TU masking convergence improvements (see tu_masking_introduction.md)
    use_probabilistic_or: bool = (
        False  # probabilistic OR for multi-TU edges (requires code integration)
    )
    use_two_timescale: bool = False  # slower LR for log_alpha via optax.multi_transform
    tu_mask_lr_scale: float = 0.1  # LR multiplier for log_alpha when use_two_timescale=True

    # Hard-pruning configuration - periodic removal of low-ratio TUs with stack rebuild
    hard_pruning_enabled: bool = False
    hard_pruning_interval: int = 500  # prune every N steps
    hard_pruning_ratio_threshold: float = 0.01  # 1:100 threshold for normalized ratios
    hard_pruning_preserve_minimum_tus: int = 1  # never prune below this per network

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
            use_two_timescale=self.use_two_timescale,
            tu_mask_lr_scale=self.tu_mask_lr_scale,
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


RATIO_PRUNE_THRESHOLD = BIOCOMP_CONSTANTS["ratio"]["prune_threshold"]


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
        if (
            "ratio" in path_str
            and "inverse" not in path_str
            and "ratio_min" not in path_str
            and "ratio_max" not in path_str
        ):
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


def _create_loss_function(
    stack: ComputeStack,
    dmanager: DesignManager,
    dconf: DesignConfig,
    params: ParameterTree,
) -> tuple[Callable, tuple[int, int], list[str]]:
    """Create loss function for design optimization.

    Returns: (loss_fn, num_z, ratio_paths)
    """
    static, _ = params.filter_by_tag(["non_grad", "shared"])
    num_z_arr = static["global/number_of_random_variables"]
    num_z = (dmanager.n_targets, int(num_z_arr.ravel()[0]))
    direct_ratio_paths, _ = get_ratio_paths_and_sources(params)
    loss_fn = dconf.loss_function.get_impl()(
        stack, dconf, dmanager, num_z=num_z, ratio_paths=direct_ratio_paths
    )
    return loss_fn, num_z, direct_ratio_paths


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


def identify_tus_to_prune(
    params: ParameterTree,
    stack: ComputeStack,
    dmanager: DesignManager,
    ratio_threshold: float,
    use_soft_pruning: bool,
    preserve_minimum: int,
) -> dict[int, set[str]]:
    """Identify TUs to remove for each network based on normalized ratios.

    Returns dict mapping network_id -> set of TU IDs to remove.

    Logic:
    1. Use introspection to get per-TU normalized ratios from aggregation nodes
    2. For each network, collect TUs where normalized ratio < threshold
    3. If soft-pruning enabled (tu_log_alpha present), include TUs where sigmoid(log_alpha) < 0.5
    4. Exclude TUs in stack.no_masking_tu_ids (topology-changing TUs)
    5. Ensure at least preserve_minimum TUs remain
    """
    from biocomp.paramintrospect import introspect_stack, aggregate_by_tu, ParamKind

    tus_to_remove: dict[int, set[str]] = {}
    no_masking_tu_ids = stack.no_masking_tu_ids or set()
    tu_id_to_idx = stack.tu_id_to_idx or {}

    has_tu_log_alpha = TU_LOG_ALPHA_PATH in params

    for net_idx in range(len(stack.networks)):
        infos = introspect_stack(stack, params, net_idx)
        tu_data = aggregate_by_tu(infos)

        candidates: set[str] = set()
        all_tu_ids: set[str] = set()

        for tu_id, entries in tu_data.items():
            all_tu_ids.add(tu_id)
            if tu_id in no_masking_tu_ids:
                continue

            for node_type, tg in entries:
                for pv in tg.params:
                    if pv.kind == ParamKind.RATIO:
                        ratio_val = float(
                            pv.value if isinstance(pv.value, (int, float)) else pv.value[0]
                        )
                        if ratio_val < ratio_threshold:
                            candidates.add(tu_id)
                            break
                if tu_id in candidates:
                    break

        if use_soft_pruning and has_tu_log_alpha:
            tu_log_alpha = params[TU_LOG_ALPHA_PATH]
            assert tu_log_alpha.ndim >= 2, (
                f"tu_log_alpha must be at least 2D, got {tu_log_alpha.ndim}D"
            )
            network_log_alpha = tu_log_alpha[net_idx]
            if network_log_alpha.ndim > 1:
                network_log_alpha = network_log_alpha.reshape(-1)

            for tu_id in all_tu_ids:
                if tu_id in no_masking_tu_ids:
                    continue
                if tu_id in tu_id_to_idx:
                    tu_idx = tu_id_to_idx[tu_id]
                    if tu_idx < len(network_log_alpha):
                        prob = float(jax.nn.sigmoid(network_log_alpha[tu_idx]))
                        if prob < 0.5:
                            candidates.add(tu_id)

        remaining = len(all_tu_ids) - len(candidates)
        if remaining < preserve_minimum:
            sorted_candidates = sorted(candidates)
            n_to_keep = preserve_minimum - remaining
            candidates = set(sorted_candidates[n_to_keep:])

        tus_to_remove[net_idx] = candidates

    return tus_to_remove


def _remap_tu_log_alpha(
    old_log_alpha: jnp.ndarray,
    old_tu_id_to_idx: dict[str, int],
    new_tu_id_to_idx: dict[str, int],
    init_value: float = 2.0,
) -> jnp.ndarray:
    """Remap tu_log_alpha from old to new TU indexing.

    Surviving TUs retain their log_alpha values; new array is sized for new TU count.
    """
    n_networks = old_log_alpha.shape[0]
    n_new_tus = len(new_tu_id_to_idx)
    new_log_alpha = jnp.full((n_networks, n_new_tus), init_value)

    old_idx_to_id = {v: k for k, v in old_tu_id_to_idx.items()}

    for old_idx in range(old_log_alpha.shape[-1]):
        tu_id = old_idx_to_id.get(old_idx)
        if tu_id and tu_id in new_tu_id_to_idx:
            new_idx = new_tu_id_to_idx[tu_id]
            new_log_alpha = new_log_alpha.at[:, new_idx].set(old_log_alpha[:, old_idx])

    return new_log_alpha


def hard_prune_and_rebuild(
    dmanager: DesignManager,
    dconf: DesignConfig,
    model: "BiocompModel",
    stack: ComputeStack,
    params: ParameterTree,
    tus_to_remove: dict[int, set[str]],
    key: jax.Array,
) -> tuple[DesignManager, ComputeStack, ParameterTree]:
    """Execute hard pruning: mark TUs disabled, commit, rebuild.

    Steps:
    1. For TUs to remove: set their tu_log_alpha to -10.0 (marks as disabled)
    2. Call stack.commit(params) to get committed networks
    3. Create new DesignManager with committed networks
    4. Rebuild ComputeStack via dmanager.build_stack()
    5. Initialize new params, transferring what's possible
    6. Return (new_dmanager, new_stack, new_params)
    """
    from copy import deepcopy

    old_tu_id_to_idx = stack.tu_id_to_idx or {}
    old_n_tus = len(old_tu_id_to_idx)
    n_networks = len(stack.networks)

    params_for_commit = deepcopy(params)

    if TU_LOG_ALPHA_PATH in params_for_commit and old_tu_id_to_idx:
        tu_log_alpha = params_for_commit[TU_LOG_ALPHA_PATH]
        modified_log_alpha = jnp.array(tu_log_alpha)

        for net_idx, tu_ids in tus_to_remove.items():
            for tu_id in tu_ids:
                if tu_id in old_tu_id_to_idx:
                    tu_idx = old_tu_id_to_idx[tu_id]
                    modified_log_alpha = modified_log_alpha.at[net_idx, tu_idx].set(-10.0)

        params_for_commit.at(TU_LOG_ALPHA_PATH, modified_log_alpha, overwrite=True)

    committed_networks = stack.commit(params_for_commit)

    total_removed = sum(len(tus) for tus in tus_to_remove.values())
    logger.info(f"[HARD-PRUNE] Committed networks, removed {total_removed} TUs total")

    new_dmanager = DesignManager(
        targets=dmanager.targets,
        networks=committed_networks,
        sampling=dmanager.sampling,
        enable_tu_masking=dmanager.enable_tu_masking,
    )

    pkey, init_key = jax.random.split(key)
    new_stack = new_dmanager.build_stack(
        model,
        unlock_ratios=True,
        use_latent_ratios=dconf.use_latent_ratios,
        latent_dim=dconf.latent_dim,
        latent_hidden_dim=dconf.latent_hidden_dim,
        auto_lock_topology_tus=dconf.auto_lock_topology_tus,
    )

    new_n_tus = new_dmanager.n_tus if new_dmanager.enable_tu_masking else 0
    new_tu_id_to_idx = new_dmanager.tu_id_to_idx if new_dmanager.enable_tu_masking else {}

    logger.info(f"[HARD-PRUNE] Rebuilt stack: {old_n_tus} -> {new_n_tus} TUs")

    new_params = initialize_params(
        new_stack,
        dconf.n_replicates,
        new_dmanager.n_targets,
        model.shared_params,
        init_key,
        n_tus=new_n_tus,
        n_networks=len(new_dmanager.networks),
        tu_log_alpha_init_mean=dconf.tu_log_alpha_init_mean,
        tu_log_alpha_init_std=dconf.tu_log_alpha_init_std,
        use_latent_tu_masking=dconf.use_latent_tu_masking,
        latent_tu_dim=dconf.latent_tu_dim,
        latent_tu_hidden_dim=dconf.latent_tu_hidden_dim,
        no_masking_tu_ids=new_stack.no_masking_tu_ids,
        tu_id_to_idx=new_stack.tu_id_to_idx,
    )

    if TU_LOG_ALPHA_PATH in params and TU_LOG_ALPHA_PATH in new_params and new_n_tus > 0:
        old_log_alpha = params[TU_LOG_ALPHA_PATH]
        if old_log_alpha.ndim == 4:
            old_2d = old_log_alpha[0, 0]
        elif old_log_alpha.ndim == 2:
            old_2d = old_log_alpha
        else:
            old_2d = old_log_alpha.reshape(n_networks, -1)

        remapped = _remap_tu_log_alpha(old_2d, old_tu_id_to_idx, new_tu_id_to_idx)

        new_log_alpha = new_params[TU_LOG_ALPHA_PATH]
        if new_log_alpha.ndim == 4:
            remapped_4d = jnp.tile(
                remapped[None, None, :, :], (dconf.n_replicates, new_dmanager.n_targets, 1, 1)
            )
            new_params.at(TU_LOG_ALPHA_PATH, remapped_4d, overwrite=True)
        elif new_log_alpha.ndim == 2:
            new_params.at(TU_LOG_ALPHA_PATH, remapped, overwrite=True)

    return new_dmanager, new_stack, new_params


def _start_with_hard_pruning(
    dmanager: DesignManager,
    dconf: DesignConfig,
    model: "BiocompModel",
    loggers: list[tuple[int, Callable]] | None = None,
    logger_objects: list | None = None,
    async_handler=None,
    lock_ratios: bool = False,
):
    """Design optimization with periodic hard-pruning.

    Runs optimization in segments, hard-pruning between segments.
    """
    timer = _PhaseTimer()
    logger.info("=" * 60)
    logger.info("DESIGN OPTIMIZATION WITH HARD-PRUNING")
    logger.info("=" * 60)

    pkey, bkey, loop_key = jax.random.split(dconf.seed_key, 3)

    steps_per_epoch = max(1, dconf.n_batches_per_epoch // dconf.batches_per_step)
    total_steps = int(dconf.n_epochs * steps_per_epoch)
    steps_per_segment = dconf.hard_pruning_interval
    n_segments = (total_steps + steps_per_segment - 1) // steps_per_segment

    logger.info(
        f"[HARD-PRUNE] Total steps: {total_steps}, interval: {steps_per_segment}, segments: {n_segments}"
    )

    current_dmanager = dmanager
    accumulated_loss_history: list = []
    accumulated_step_history: list = []
    global_step_offset = 0
    segment_params: ParameterTree | None = None

    for segment_idx in range(n_segments):
        segment_start_step = segment_idx * steps_per_segment
        segment_end_step = min((segment_idx + 1) * steps_per_segment, total_steps)
        segment_steps = segment_end_step - segment_start_step

        if segment_steps <= 0:
            break

        segment_epochs = max(1, (segment_steps + steps_per_epoch - 1) // steps_per_epoch)

        segment_config = DesignConfig(
            n_replicates=dconf.n_replicates,
            n_epochs=segment_epochs,
            batch_size=dconf.batch_size,
            n_batches_per_epoch=dconf.n_batches_per_epoch,
            batches_per_step=dconf.batches_per_step,
            reshuffle_batches=dconf.reshuffle_batches,
            tu_log_alpha_init_mean=dconf.tu_log_alpha_init_mean,
            tu_log_alpha_init_std=dconf.tu_log_alpha_init_std,
            use_latent_ratios=dconf.use_latent_ratios,
            latent_dim=dconf.latent_dim,
            latent_hidden_dim=dconf.latent_hidden_dim,
            enable_tu_masking=dconf.enable_tu_masking,
            use_latent_tu_masking=dconf.use_latent_tu_masking,
            latent_tu_dim=dconf.latent_tu_dim,
            latent_tu_hidden_dim=dconf.latent_tu_hidden_dim,
            auto_lock_topology_tus=dconf.auto_lock_topology_tus,
            use_probabilistic_or=dconf.use_probabilistic_or,
            use_two_timescale=dconf.use_two_timescale,
            tu_mask_lr_scale=dconf.tu_mask_lr_scale,
            hard_pruning_enabled=False,
            pluggable_optimizer=dconf.pluggable_optimizer,
            loss_function=dconf.loss_function,
            optimizer_stack=dconf.optimizer_stack,
            keep_in_history=dconf.keep_in_history,
            seed=int(jax.random.key_data(jax.random.fold_in(loop_key, segment_idx))[0]) % (2**31),
        )

        logger.info(
            f"[HARD-PRUNE] Segment {segment_idx + 1}/{n_segments}: steps {segment_start_step}-{segment_end_step}"
        )

        segment_params, segment_loss_history, segment_step_history = start(
            current_dmanager,
            segment_config,
            model,
            loggers=loggers,
            logger_objects=logger_objects,
            async_handler=async_handler,
            lock_ratios=lock_ratios,
        )

        accumulated_loss_history.extend(segment_loss_history)
        accumulated_step_history.extend(segment_step_history)
        global_step_offset += segment_steps

        if segment_idx < n_segments - 1:
            timer.start("prune", "[HARD-PRUNE] Identifying TUs to prune...")

            temp_stack = current_dmanager.build_stack(
                model,
                unlock_ratios=not lock_ratios,
                use_latent_ratios=dconf.use_latent_ratios,
                latent_dim=dconf.latent_dim,
                latent_hidden_dim=dconf.latent_hidden_dim,
                auto_lock_topology_tus=dconf.auto_lock_topology_tus,
            )

            single_rep_params = jax.tree.map(lambda x: x[0] if x.ndim > 2 else x, segment_params)

            tus_to_remove = identify_tus_to_prune(
                single_rep_params,
                temp_stack,
                current_dmanager,
                ratio_threshold=dconf.hard_pruning_ratio_threshold,
                use_soft_pruning=dconf.enable_tu_masking,
                preserve_minimum=dconf.hard_pruning_preserve_minimum_tus,
            )

            timer.end("prune")

            total_to_remove = sum(len(tus) for tus in tus_to_remove.values())
            if total_to_remove > 0:
                prune_key = jax.random.fold_in(loop_key, segment_idx + 1000)
                current_dmanager, _, _ = hard_prune_and_rebuild(
                    current_dmanager,
                    dconf,
                    model,
                    temp_stack,
                    single_rep_params,
                    tus_to_remove,
                    prune_key,
                )
                logger.info(f"[HARD-PRUNE] Removed {total_to_remove} TUs, rebuilt stack")
            else:
                logger.info("[HARD-PRUNE] No TUs to remove, continuing with current structure")

    logger.info("=" * 60)
    logger.info(f"HARD-PRUNING OPTIMIZATION COMPLETE in {timer.total():.2f}s")
    logger.info("=" * 60)

    assert segment_params is not None, "No optimization segments were run"
    return segment_params, accumulated_loss_history, accumulated_step_history


def _start_pluggable(
    dmanager: DesignManager,
    dconf: DesignConfig,
    model: BiocompModel,
    loggers: list[tuple[int, Callable]] | None = None,
    logger_objects: list | None = None,
    async_handler=None,
    lock_ratios: bool = False,
):
    from .designcodec import GenomeCodec

    timer = _PhaseTimer()
    logger.info("=" * 60)
    logger.info("DESIGN OPTIMIZATION (PLUGGABLE OPTIMIZER)")
    logger.info("=" * 60)

    pkey, bkey, loop_key = jax.random.split(dconf.seed_key, 3)
    optimizer = dconf.pluggable_optimizer
    from .designoptim import NSGA2DesignOptimizer

    is_nsga2 = isinstance(optimizer, NSGA2DesignOptimizer)
    n_tus = dmanager.n_tus if dmanager.enable_tu_masking else 0
    nsga2_n_tus = dmanager.n_tus if is_nsga2 else 0
    n_networks = len(dmanager.networks)

    timer.start("stack", "[1/5] Building compute stack...")
    stack = dmanager.build_stack(
        model,
        unlock_ratios=not lock_ratios,
        use_latent_ratios=dconf.use_latent_ratios,
        latent_dim=dconf.latent_dim,
        latent_hidden_dim=dconf.latent_hidden_dim,
        auto_lock_topology_tus=dconf.auto_lock_topology_tus,
    )
    timer.end("stack")

    timer.start("params", "[2/5] Initializing parameters...")
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
        use_latent_tu_masking=dconf.use_latent_tu_masking,
        latent_tu_dim=dconf.latent_tu_dim,
        latent_tu_hidden_dim=dconf.latent_tu_hidden_dim,
        no_masking_tu_ids=stack.no_masking_tu_ids,
        tu_id_to_idx=stack.tu_id_to_idx,
    )
    initial_params = jax.tree.map(lambda x: x.squeeze(0), initial_params)
    timer.end("params")

    timer.start("codec", "[3/5] Creating codec and loss function...")
    codec = GenomeCodec.from_params(
        initial_params,
        static_tags=("shared", "non_grad"),
        use_latent_ratios=dconf.use_latent_ratios,
    )
    flat_params = codec.encode(initial_params)
    logger.info(f"  Genome dimension: {codec.param_dim}")

    if codec.param_dim == 0:
        raise ValueError(
            "No parameters to optimize: genome dimension is 0. "
            "This typically happens with zero-freedom recipes where all ratios are explicitly locked. "
            "Design optimization requires at least some unlocked parameters. "
            "Consider using a recipe with unlocked ratios (e.g., T_2_ratios_only.yaml) or "
            "check that your recipe doesn't have `locked: true` on all ratio specifications."
        )

    loss_fn, num_z, _ = _create_loss_function(stack, dmanager, dconf, initial_params)
    timer.end("codec")

    timer.start("objective", "[4/5] Creating objective function...")
    effective_batch_size = dconf.batch_size
    if dmanager.is_lattice_mode:
        grid_res = dmanager.grid_resolution
        assert grid_res is not None
        effective_batch_size *= grid_res[0] * grid_res[1]

    xbatches_list, ybatches_list = dmanager.get_samples(
        (len(dmanager.networks), 1, 1, 1, dconf.batch_size), bkey
    )
    x_samples = jnp.concatenate(xbatches_list, axis=-1).reshape(
        effective_batch_size, dmanager.n_targets, -1
    )
    y_samples = ybatches_list[0].reshape(effective_batch_size, dmanager.n_targets, -1)
    fixed_z = jnp.full((x_samples.shape[0], *num_z), 0.5)
    fixed_key = jax.random.key(42)

    def single_objective(flat_genome: jnp.ndarray, step: jnp.ndarray) -> float:
        params = codec.decode(flat_genome, apply_constraints=True)
        static_p, dynamic_p = params.filter_by_tag(["non_grad", "shared"])
        loss, _ = loss_fn(dynamic_p, static_p, x_samples, y_samples, fixed_z, fixed_key, step)
        return loss

    def get_yhatdep(flat_genome: jnp.ndarray, step: jnp.ndarray) -> jnp.ndarray:
        params = codec.decode(flat_genome, apply_constraints=True)
        static_p, dynamic_p = params.filter_by_tag(["non_grad", "shared"])
        _, aux = loss_fn(dynamic_p, static_p, x_samples, y_samples, fixed_z, fixed_key, step)
        return aux.get("yhatdep")

    def get_yhatdep_nsga2(flat_genome: jnp.ndarray, step: jnp.ndarray) -> jnp.ndarray:
        continuous_genes = flat_genome[nsga2_n_tus:]
        params = codec.decode(continuous_genes, apply_constraints=True)
        static_p, dynamic_p = params.filter_by_tag(["non_grad", "shared"])
        _, aux = loss_fn(dynamic_p, static_p, x_samples, y_samples, fixed_z, fixed_key, step)
        return aux.get("yhatdep")

    compiled_get_yhatdep = jax.jit(get_yhatdep_nsga2 if is_nsga2 else get_yhatdep)
    timer.end("objective")

    timer.start("opt_init", "[5/5] Initializing optimizer...")
    init_key, opt_key = jax.random.split(loop_key)

    if is_nsga2:

        def nsga2_objective(flat_genome: jnp.ndarray) -> float:
            continuous_genes = flat_genome[nsga2_n_tus:]
            params = codec.decode(continuous_genes, apply_constraints=True)
            static_p, dynamic_p = params.filter_by_tag(["non_grad", "shared"])
            loss, _ = loss_fn(
                dynamic_p, static_p, x_samples, y_samples, fixed_z, fixed_key, jnp.array(0)
            )
            return loss

        opt_state = optimizer.init(
            init_key,
            flat_params,
            nsga2_objective,
            n_tus=nsga2_n_tus,
            continuous_dim=codec.param_dim,
        )
        step_objective_fn = nsga2_objective
    else:
        compiled_pop_objective = jax.jit(jax.vmap(single_objective, in_axes=(0, None)))
        dummy_pop = jnp.zeros((optimizer._get_pop_size(codec.param_dim), codec.param_dim))
        _ = compiled_pop_objective(dummy_pop, jnp.array(0, dtype=jnp.int32)).block_until_ready()
        opt_state = optimizer.init(
            init_key, flat_params, lambda g: single_objective(g, jnp.array(0, dtype=jnp.int32))
        )
        step_objective_fn = compiled_pop_objective
    logger.info(f"  Initial loss: {float(opt_state.best_loss):.6f}")
    timer.end("opt_init")

    logger.info("=" * 60)
    logger.info("STARTING OPTIMIZATION LOOP")
    logger.info("=" * 60)

    loss_history, step_history = [], []
    pbar = tqdm(desc="Optimizing", unit="step")

    while not optimizer.should_stop(opt_state):
        opt_key, step_key = jax.random.split(opt_key)
        current_step = jnp.array(int(opt_state.step), dtype=jnp.int32)
        opt_state, metrics = optimizer.step(opt_state, step_key, step_objective_fn, current_step)
        loss_history.append(float(opt_state.best_loss))
        step_history.append({k: float(v) if hasattr(v, "item") else v for k, v in metrics.items()})
        pbar.update(1)
        pbar.set_postfix(loss=f"{float(opt_state.best_loss):.4f}")

        if loggers and any(int(opt_state.step) % p == 0 or p == -1 for p, _ in loggers):
            yhatdep_arr = compiled_get_yhatdep(opt_state.best_params, current_step)
            step_data = {"loss": [[float(opt_state.best_loss)]], "yhatdep": yhatdep_arr, **metrics}
            needs_params = logger_objects and any(
                "latest_params" in getattr(lo, "required_arrays", [])
                for lo in logger_objects
                if int(opt_state.step) % getattr(lo, "periods", 1) == 0
            )
            if needs_params:
                best_genome = opt_state.best_params
                if is_nsga2:
                    best_genome = best_genome[nsga2_n_tus:]
                step_data["latest_params"] = codec.decode(best_genome, apply_constraints=True)
            for period, callback in loggers:
                if int(opt_state.step) % period == 0 or period == -1:
                    callback(int(opt_state.step), None, step_history=step_data, stack=stack)

    pbar.close()
    logger.info("=" * 60)
    logger.info(f"OPTIMIZATION COMPLETE in {timer.total():.2f}s")
    logger.info(f"  Final loss: {float(opt_state.best_loss):.6f}, steps: {int(opt_state.step)}")
    timer.summary()

    best_genome = opt_state.best_params
    if is_nsga2:
        best_genome = best_genome[nsga2_n_tus:]
    final_params = codec.decode(best_genome, apply_constraints=True)

    final_step_data = {"loss": [[float(opt_state.best_loss)]], "latest_params": final_params}

    from .designoptim import NSGA2DesignState

    if isinstance(opt_state, NSGA2DesignState):
        pareto_front, pareto_fitness = opt_state.pareto_front, opt_state.pareto_fitness
        if pareto_front is not None and pareto_fitness is not None:
            final_step_data["pareto_front"] = pareto_front
            final_step_data["pareto_fitness"] = pareto_fitness
            logger.info(f"  Pareto front: {pareto_fitness.shape[0]} solutions")
            logger.info(f"    Min loss: {float(jnp.min(pareto_fitness[:, 0])):.4f}")
            logger.info(f"    Min TU count: {float(jnp.min(pareto_fitness[:, 1])):.0f}")

    if loggers:
        yhatdep_arr = compiled_get_yhatdep(
            opt_state.best_params, jnp.array(int(opt_state.step), dtype=jnp.int32)
        )
        final_step_data["yhatdep"] = yhatdep_arr
        for period, callback in loggers:
            if period == -1:
                callback(int(opt_state.step), None, step_history=final_step_data, stack=stack)

    step_history.append(final_step_data)

    return jax.tree.map(lambda x: x[None, ...], final_params), loss_history, step_history


def start(
    dmanager: DesignManager,
    dconf: DesignConfig,
    model: BiocompModel,
    loggers: list[tuple[int, Callable]] | None = None,
    logger_objects: list | None = None,
    async_handler=None,
    lock_ratios: bool = False,
):
    if dconf.hard_pruning_enabled:
        logger.info("Using hard-pruning mode with interval=%d", dconf.hard_pruning_interval)
        return _start_with_hard_pruning(
            dmanager, dconf, model, loggers, logger_objects, async_handler, lock_ratios
        )

    if dconf.uses_pluggable_optimizer:
        logger.info("Using pluggable optimizer: %s", type(dconf.pluggable_optimizer).__name__)
        return _start_pluggable(
            dmanager, dconf, model, loggers, logger_objects, async_handler, lock_ratios
        )

    timer = _PhaseTimer()
    logger.info("=" * 60)
    logger.info("DESIGN OPTIMIZATION - INITIALIZATION PHASE")
    logger.info("=" * 60)

    pkey, bkey, loop_key = jax.random.split(dconf.seed_key, 3)

    timer.start("stack", "[1/5] Building compute stack...")
    stack = dmanager.build_stack(
        model,
        unlock_ratios=not lock_ratios,
        use_latent_ratios=dconf.use_latent_ratios,
        latent_dim=dconf.latent_dim,
        latent_hidden_dim=dconf.latent_hidden_dim,
        auto_lock_topology_tus=dconf.auto_lock_topology_tus,
    )
    timer.end("stack")

    timer.start("params", "[2/5] Initializing parameters...")
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
        use_latent_tu_masking=dconf.use_latent_tu_masking,
        latent_tu_dim=dconf.latent_tu_dim,
        latent_tu_hidden_dim=dconf.latent_tu_hidden_dim,
        no_masking_tu_ids=stack.no_masking_tu_ids,
        tu_id_to_idx=stack.tu_id_to_idx,
    )
    timer.end("params")

    timer.start("opt_init", "[3/5] Initializing optimizer state...")
    static, dynamic = initial_params.filter_by_tag(["non_grad", "shared"])
    assert_tree_shape(dynamic, (dconf.n_replicates, dmanager.n_targets))

    # Check if there are any actual JAX arrays to optimize (not just ArrayRefs)
    jax_leaves = jax.tree.leaves(dynamic)
    if not jax_leaves:
        raise ValueError(
            "No parameters to optimize: all parameters are either shared or marked NON_GRAD. "
            "This typically happens with zero-freedom recipes where all ratios are explicitly locked. "
            "Design optimization requires at least some unlocked parameters. "
            "Consider using a recipe with unlocked ratios (e.g., T_2_ratios_only.yaml) or "
            "check that your recipe doesn't have `locked: true` on all ratio specifications."
        )

    initial_optimizer_state = vmap(vmap(dconf.optimizer.init))(dynamic)
    timer.end("opt_init")

    steps_per_epoch = max(1, dconf.n_batches_per_epoch // dconf.batches_per_step)
    total_steps = int(dconf.n_epochs * steps_per_epoch)
    assert_that(total_steps).is_greater_than(0)
    logger.info(
        f"  Config: {total_steps} steps, {steps_per_epoch}/epoch, "
        f"batch={dconf.batch_size}, batches/step={dconf.batches_per_step}"
    )

    n_networks = stack.get_nb_networks()

    timer.start("samples", "[4/5] Generating training samples...")
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
    timer.end("samples")

    effective_batch_size = dconf.batch_size
    if dmanager.is_lattice_mode:
        grid_res = dmanager.grid_resolution
        assert grid_res is not None
        xres, yres = grid_res
        effective_batch_size *= xres * yres

    n_design_inputs = 2 * len(dmanager.networks)
    logger.info(f"  Data: {len(dmanager.networks)} networks, xbatches.shape={xbatches.shape}")

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

    timer.start("loss_fn", "[5/5] Creating loss and step functions...")
    loss_func, num_z, direct_ratio_paths = _create_loss_function(
        stack, dmanager, dconf, initial_params
    )
    _, source_ratio_paths = get_ratio_paths_and_sources(initial_params)
    logger.debug(
        f"Ratio paths: {len(direct_ratio_paths)} direct + {len(source_ratio_paths)} ArrayRef"
    )

    def norm_ratios_hook(params, *a, **kw):
        if direct_ratio_paths:
            params = params.update_leaves_by_path(direct_ratio_paths, normalize_ratios_prune)
        if source_ratio_paths:
            params = normalize_ratio_source_arrays(
                params, source_ratio_paths, normalize_ratios_prune
            )
        if TU_LOG_ALPHA_PATH in params:
            params = params.update_leaves_by_path(
                [TU_LOG_ALPHA_PATH],  # pyright: ignore[reportArgumentType]
                lambda x: jnp.clip(x, LOG_ALPHA_MIN, LOG_ALPHA_MAX),
            )
        return params

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

        return jax.vmap(
            Partial(per_replicate_step, num_z=num_z, training_config=dconf, scannable_step=step_fn)
        )(params, opt_state, keys, xs, ys)

    timer.end("loss_fn")

    logger.info("-" * 60)
    logger.info(f"INITIALIZATION COMPLETE in {timer.total():.2f}s")
    timer.summary()
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
        logger_objects=logger_objects,
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
    """Sample evaluation data. Returns (xraw, yraw) with shapes (n_networks, n_replicates, n_samples, n_targets, 2/1).

    In lattice mode, uses grid sampling (n_samples = xres * yres), ignoring n_eval_samples.
    In uniform mode, samples n_eval_samples random points.
    """
    n_networks, n_replicates, n_targets = (
        len(dmanager.networks),
        dconf.n_replicates,
        dmanager.n_targets,
    )
    seed = int(jax.random.key_data(key)[0]) % (2**31)

    if dmanager.is_lattice_mode:
        grid_res = dmanager.grid_resolution
        assert grid_res is not None
        xres, yres = grid_res
        n_samples = xres * yres
        xlist, ylist = dmanager.get_samples((n_networks, n_replicates, 1), seed)
    else:
        n_samples = n_eval_samples
        xlist, ylist = dmanager.get_samples((n_networks, n_replicates, n_eval_samples), seed)

    xraw, yraw = jnp.stack(xlist, axis=0), jnp.stack(ylist, axis=0)
    assert xraw.shape == (n_networks, n_replicates, n_samples, n_targets, 2), (
        f"xraw shape mismatch: {xraw.shape} vs expected ({n_networks}, {n_replicates}, {n_samples}, {n_targets}, 2)"
    )
    assert yraw.shape == (n_networks, n_replicates, n_samples, n_targets, 1), (
        f"yraw shape mismatch: {yraw.shape} vs expected ({n_networks}, {n_replicates}, {n_samples}, {n_targets}, 1)"
    )
    return xraw, yraw


def _compute_grid_loss_for_eval(
    y_img: jnp.ndarray,
    yhat_img: jnp.ndarray,
    w_sinkhorn: float,
    w_lncc: float,
    w_mse: float,
    w_rmse: float,
    w_simse: float,
    w_zncc: float,
    w_gradient: float,
    w_spectral: float,
    w_contrast: float,
    eps_sinkhorn: float,
    n_sinkhorn_iters: int,
    lncc_kernel: int,
) -> jax.Array | float:
    """Compute grid loss for evaluation - matches grid_distance_loss computation."""
    y_img, yhat_img = _sanitize(y_img), _sanitize(yhat_img)
    y_flat, yhat_flat = y_img.ravel(), yhat_img.ravel()

    sinkhorn_l = (
        sinkhorn_divergence_conv(
            proj_nonneg_ste(yhat_img),
            proj_nonneg_ste(y_img),
            eps_sinkhorn,
            n_iters=n_sinkhorn_iters,
        )
        if w_sinkhorn > 0
        else 0.0
    )
    lncc_l = lncc_grid_loss(None, y_img, yhat_img, k=lncc_kernel) if w_lncc > 0 else 0.0
    mse_l = jnp.mean((y_img - yhat_img) ** 2) if (w_mse > 0 or w_rmse > 0) else 0.0
    rmse_l = jnp.sqrt(mse_l) if w_rmse > 0 else 0.0
    simse_l = simse_loss(None, y_flat, yhat_flat) if w_simse > 0 else 0.0
    zncc_l = zncc_loss(None, y_flat, yhat_flat) if w_zncc > 0 else 0.0
    spectral_l = spectral_loss(None, y_img, yhat_img) if w_spectral > 0 else 0.0
    gradient_l = gradient_magnitude_loss(y_img, yhat_img) if w_gradient > 0 else 0.0
    contrast_l = (
        jax.nn.relu((jnp.max(y_img) - jnp.min(y_img)) - (jnp.max(yhat_img) - jnp.min(yhat_img)))
        if w_contrast > 0
        else 0.0
    )

    return (
        w_sinkhorn * sinkhorn_l
        + w_lncc * lncc_l
        + w_mse * mse_l
        + w_rmse * rmse_l
        + w_simse * simse_l
        + w_zncc * zncc_l
        + w_spectral * spectral_l
        + w_gradient * gradient_l
        + w_contrast * contrast_l
    )


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

    CRITICAL: This function uses the SAME loss as training (grid_distance_loss weights).
    This ensures evaluation loss reflects actual design performance.
    """
    stack = dmanager.build_stack(model, unlock_ratios=False)
    n_networks, n_replicates, n_targets, n_samples = (
        len(dmanager.networks),
        dconf.n_replicates,
        dmanager.n_targets,
        xraw.shape[2],
    )
    logger.info(f"Evaluating: {n_replicates} reps × {n_targets} targets × {n_samples} samples")

    # extract loss weights from dconf.loss_function (same as training)
    loss_kwargs = getattr(dconf.loss_function, "kwargs", {}) or {}
    w_sinkhorn = float(loss_kwargs.get("w_sinkhorn", 1.0))
    w_lncc = float(loss_kwargs.get("w_lncc", 0.5))
    w_mse = float(loss_kwargs.get("w_mse", 0.0))
    w_rmse = float(loss_kwargs.get("w_rmse", 0.5))
    w_simse = float(loss_kwargs.get("w_simse", 0.0))
    w_zncc = float(loss_kwargs.get("w_zncc", 0.0))
    w_gradient = float(loss_kwargs.get("w_gradient", 0.0))
    w_spectral = float(loss_kwargs.get("w_spectral", 0.0))
    w_contrast = float(loss_kwargs.get("w_contrast", 0.0))
    eps_sinkhorn = float(loss_kwargs.get("eps_sinkhorn", 0.1))
    n_sinkhorn_iters = int(loss_kwargs.get("n_sinkhorn_iters", 50))
    lncc_kernel = int(loss_kwargs.get("lncc_kernel", 7))

    logger.debug(
        f"Eval loss weights: sinkhorn={w_sinkhorn}, lncc={w_lncc}, mse={w_mse}, rmse={w_rmse}, "
        f"simse={w_simse}, zncc={w_zncc}, gradient={w_gradient}"
    )

    # grid resolution for reshaping predictions
    grid_res = dmanager.grid_resolution
    assert grid_res is not None, "grid_resolution required for evaluation"
    xres, yres = grid_res
    assert n_samples == xres * yres, f"n_samples={n_samples} must equal xres*yres={xres * yres}"

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

    stack_apply = stack.apply
    assert stack_apply is not None, "stack.apply must be set after build"

    def apply_with_tu_mask(params, x_batch, z_batch, keys, tu_mask):
        def apply_single(x, z, k):
            return stack_apply(params, x, z, k, tu_enabled_random_vars=tu_mask)

        return vmap(apply_single)(x_batch, z_batch, keys)

    apply_batched = jax.jit(apply_with_tu_mask)
    pbar = tqdm(total=n_replicates * n_targets, desc="Evaluating", unit="rep×tgt")

    for rep_idx in range(n_replicates):
        rep_losses, rep_preds = [], [] if store_predictions else None
        for tid in range(n_targets):
            rep_params = jax.tree.map(lambda x: x[rep_idx, tid], final_params)
            x_slice, y_slice = x_combined[rep_idx, :, tid, :], y_combined[rep_idx, :, tid, :]

            # binary masking is now the default in get_tu_masks()
            # no need to compute tu_mask here - get_tu_masks will use log_alpha from params
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
                assert rep_preds is not None
                rep_preds.append(yhat_dep)

            # compute grid-based loss matching training (not simple MSE!)
            # reshape to grid format: (n_samples, n_networks) -> (n_networks, yres, xres)
            y_grid = y_slice.squeeze(-1).reshape(yres, xres)
            network_losses = []
            for net_idx in range(n_networks):
                yhat_grid = yhat_dep[:, net_idx].reshape(yres, xres)
                loss_val = _compute_grid_loss_for_eval(
                    y_grid,
                    yhat_grid,
                    w_sinkhorn,
                    w_lncc,
                    w_mse,
                    w_rmse,
                    w_simse,
                    w_zncc,
                    w_gradient,
                    w_spectral,
                    w_contrast,
                    eps_sinkhorn,
                    n_sinkhorn_iters,
                    lncc_kernel,
                )
                network_losses.append(float(loss_val))
            rep_losses.append(network_losses)
            pbar.update(1)

        all_losses.append(rep_losses)
        if store_predictions:
            assert all_predictions is not None and rep_preds is not None
            all_predictions.append(jnp.stack(rep_preds, axis=0))

    pbar.close()
    losses = jnp.array(all_losses)
    logger.info(
        f"Evaluation complete. Loss: [{float(losses.min()):.4f}, {float(losses.max()):.4f}]"
    )

    if store_predictions:
        assert all_predictions is not None
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
