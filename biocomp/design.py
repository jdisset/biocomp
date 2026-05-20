# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from __future__ import annotations
import random
from functools import partial
from typing import Literal, Any, TYPE_CHECKING
from collections.abc import Callable
import warnings

import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap
from jax.typing import ArrayLike

from assertpy import assert_that
from pydantic import BaseModel, ConfigDict, Field, model_validator

from biocomp.utils import encode_function, EncodedPartialFunction
from biocomp.compute import ComputeStack
import biocomp.nodes as nd
from biocomp.network import Network
from .parameters import ParameterTree
from .step_history import StepHistorySnapshot
from biocomp.logging_config import get_logger
from biocomp.optimutils import DesignOptimConfig
from biocomp.logger_dispatch import LoggerDispatch

if TYPE_CHECKING:
    from biocomptools.modelmodel import BiocompModel
from biocomp.design_eval import (  # noqa: F401  # re-exported for backward compat
    evaluate_design,
    sample_for_evaluation,
)
from biocomp.designloss import (
    grid_distance_loss,
)
from biocomp.tumasking import (
    build_tu_id_mapping,
)
from biocomp.tumasking_strategy import (
    TUMaskingMode,
    TUMaskingStrategy,
    build_strategy_from_config,
)
from biocomp.tracing import (
    save_debug_state,
    is_design_debug_enabled,
    TracingSettings,
    should_save_full_objects,
    trace_scope,
    snapshot_full_params,
)

from biocomp.design_targets import (
    LatticeSampling,
    SamplingConfigUnion,
    DataTarget,
    TargetUnion,
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
    sampling: SamplingConfigUnion = Field(default_factory=LatticeSampling, discriminator="strategy")
    enable_tu_masking: bool = False

    _tu_ids: list[str] | None = None
    _tu_id_to_idx: dict[str, int] | None = None

    def _ensure_tu_mapping(self):
        if self._tu_id_to_idx is None:
            self._tu_ids, self._tu_id_to_idx = build_tu_id_mapping(self.networks)

    @property
    def n_tus(self) -> int:
        self._ensure_tu_mapping()
        assert self._tu_ids is not None
        return len(self._tu_ids)

    @property
    def tu_id_to_idx(self) -> dict[str, int]:
        self._ensure_tu_mapping()
        assert self._tu_id_to_idx is not None
        return self._tu_id_to_idx

    @model_validator(mode="after")
    def _validate_input_order_consistency(self) -> DesignManager:
        """Warn if scaffold networks lack input_order with DataTargets."""
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
            X_lattice, Y_lattice = target.get_lattice(resolution=grid, seed=seed, jitter=jitter)
            X_tiled = np.tile(X_lattice, (n, 1))
            Y_tiled = np.tile(Y_lattice[None, ...], (n, 1, 1))
            return X_tiled, Y_tiled
        else:
            return target.sample_uniform(n=n, seed=seed)

    def get_samples(
        self,
        samples: int | tuple[int, ...],
        seed: int | ArrayLike | None = None,
        share_across_networks: bool = False,
    ) -> tuple[list[jax.Array], list[jax.Array]]:
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        elif not isinstance(seed, int):
            key_data = jax.random.key_data(seed)
            seed = int(key_data[0]) % (2**31)

        if isinstance(samples, int):
            samples = (samples,)

        n_networks = samples[0]
        if share_across_networks and n_networks > 1:
            base_samples = (1, *samples[1:])
            if isinstance(self.sampling, LatticeSampling):
                xs_list, ys_list = self._get_lattice_samples(base_samples, seed)
            else:
                xs_list, ys_list = self._get_uniform_samples(base_samples, seed)
            xs = [xs_list[0]] * n_networks
            ys = [ys_list[0]] * n_networks
            return xs, ys

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
        enable_tu_masking: bool | None = None,
    ) -> ComputeStack:
        logger.info(f"Building stack with {len(self.networks)} design networks")
        logger.info(f"Design network names: {[n.name for n in self.networks]}")
        stack = ComputeStack(networks=self.networks)
        logger.info(f"Stack after creation has {len(stack.networks)} networks")

        compute_config = model.compute_config.model_copy(deep=True)
        compute_config.backfill_from_defaults()
        compute_config.detect_output_compat(model.shared_params)

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

        effective_tu_masking = (
            enable_tu_masking if enable_tu_masking is not None else self.enable_tu_masking
        )
        stack.build(
            compute_config,
            enable_tu_masking=effective_tu_masking,
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
    n_replicates: int,
    n_targets: int,
    shared_params,
    key,
    strategy: TUMaskingStrategy | None = None,
    n_tus: int = 0,
    n_networks: int = 1,
    no_masking_tu_ids: set[str] | None = None,
    tu_id_to_idx: dict[str, int] | None = None,
):
    """Initialize design params using TU masking strategy."""
    tu_key, init_key = jax.random.split(key)

    def init_single(k):
        params = stack.init(k)
        _, nonshared = params.filter_by_tag(["shared"])
        return ParameterTree.merge(shared_params, nonshared)

    def init_target_params(k):
        params = vmap(init_single)(jax.random.split(k, n_targets))
        return params

    params = vmap(init_target_params)(jax.random.split(init_key, n_replicates))

    # Snapshot params after vmap where values are concrete
    if should_save_full_objects():
        with trace_scope("design_params_init", component="design") as scope:
            scope.snapshot("params_full", snapshot_full_params(params))

    if strategy is not None and strategy.has_masking and n_tus > 0:
        assert n_networks > 0, f"n_tus={n_tus} but n_networks={n_networks}"
        strategy.init_params(
            params,
            n_replicates=n_replicates,
            n_targets=n_targets,
            n_networks=n_networks,
            n_tus=n_tus,
            key=tu_key,
            protected_tu_ids=no_masking_tu_ids or set(),
            tu_id_to_idx=tu_id_to_idx or {},
        )
        logger.info(
            f"Initialized TU masking ({strategy.mode.value}): {n_tus} TUs x {n_networks} networks"
        )

    return params


class TUMaskingParams(BaseModel):
    """Configuration for TU masking strategy."""

    mode: TUMaskingMode = TUMaskingMode.NONE
    latent_dim: int = 16
    hidden_dim: int = 32
    init_mean: float = 2.0
    init_std: float = 0.5

    model_config = ConfigDict(use_enum_values=False)


class DesignConfig(DesignOptimConfig):
    loss_function: EncodedPartialFunction = Field(default=grid_distance_loss)  # pyright: ignore[reportAssignmentType]
    n_replicates: int = 4
    keep_in_history: list[str] | Literal["all"] = "all"

    use_latent_ratios: bool = False
    latent_dim: int = 8
    latent_hidden_dim: int = 16

    tu_masking: TUMaskingParams = Field(default_factory=TUMaskingParams)
    auto_lock_topology_tus: bool = True

    use_probabilistic_or: bool = False
    use_two_timescale: bool = False
    tu_mask_lr_scale: float = 0.1

    hard_pruning_enabled: bool = False
    hard_pruning_interval: int = 500
    hard_pruning_ratio_threshold: float = 0.01
    hard_pruning_preserve_minimum_tus: int = 1
    hard_pruning_prune_margin: float = 0.1
    hard_pruning_disable_tu_masking_final_segment: bool = False
    hard_pruning_commit_aware_final_guard: bool = True
    hard_pruning_commit_aware_selection_interval: int = 64
    hard_pruning_top_percent: float | None = None
    hard_pruning_min_networks: int | None = None

    pluggable_optimizer: Any = None

    tracing: TracingSettings = Field(default_factory=TracingSettings)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_hard_pruning_network_selection(self) -> DesignConfig:
        if self.hard_pruning_top_percent is not None and not (
            0.0 < self.hard_pruning_top_percent <= 100.0
        ):
            raise ValueError(
                f"hard_pruning_top_percent must be in (0, 100], got {self.hard_pruning_top_percent}"
            )
        if self.hard_pruning_min_networks is not None and self.hard_pruning_min_networks < 1:
            raise ValueError(
                f"hard_pruning_min_networks must be >= 1, got {self.hard_pruning_min_networks}"
            )
        if self.hard_pruning_commit_aware_selection_interval < 1:
            raise ValueError(
                "hard_pruning_commit_aware_selection_interval must be >= 1, got "
                f"{self.hard_pruning_commit_aware_selection_interval}"
            )
        return self

    @property
    def enable_tu_masking(self) -> bool:
        """True if TU masking is enabled (any mode except NONE)."""
        return self.tu_masking.mode != TUMaskingMode.NONE

    @property
    def uses_pluggable_optimizer(self) -> bool:
        return self.pluggable_optimizer is not None

    @property
    def optimizer(self) -> Any:
        if self.pluggable_optimizer is not None:
            return self.pluggable_optimizer
        return super().optimizer

    def get_pluggable_optimizer(self) -> Any:
        if self.pluggable_optimizer is not None:
            return self.pluggable_optimizer
        from biocomp.pluggable_opt.optimizers import GradientDescentOptimizer

        total_steps = int(self.n_epochs * max(1, self.n_batches_per_epoch // self.batches_per_step))
        return GradientDescentOptimizer(
            optimizer_stack=self.optimizer_stack,
            n_steps=total_steps,
            sanitize_grads=True,
            use_two_timescale=self.use_two_timescale,
            tu_mask_lr_scale=self.tu_mask_lr_scale,
        )

    def build_tu_masking_strategy(self) -> TUMaskingStrategy:
        return build_strategy_from_config(self)


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


from .design_pruning import run_with_hard_pruning as _start_with_hard_pruning  # noqa: E402


def start(
    dmanager: DesignManager,
    dconf: DesignConfig,
    model: BiocompModel,
    dispatch: LoggerDispatch | None = None,
    lock_ratios: bool = False,
    initial_params: ParameterTree | None = None,
    initial_step: int = 0,
    select_best_synced_params: bool = False,
    best_synced_score_fn=None,
    best_synced_initial_score: float | None = None,
):
    def _with_snapshot(result):
        params, losses, step_history = result
        return params, losses, StepHistorySnapshot.from_raw(step_history)

    if dconf.hard_pruning_enabled:
        logger.info("Using hard-pruning mode with interval=%d", dconf.hard_pruning_interval)
        params, loss, steps, _ = _start_with_hard_pruning(
            dmanager,
            dconf,
            model,
            dispatch=dispatch,
            lock_ratios=lock_ratios,
        )
        return params, loss, StepHistorySnapshot.from_raw(steps)

    if dconf.uses_pluggable_optimizer:
        logger.info("Using pluggable optimizer: %s", type(dconf.pluggable_optimizer).__name__)
        from .pluggable_opt.run_pluggable import run_pluggable

        return _with_snapshot(
            run_pluggable(
                dmanager,
                dconf,
                model,
                dispatch=dispatch,
                lock_ratios=lock_ratios,
            )
        )
    from .design_run import run_design

    return _with_snapshot(
        run_design(
            dmanager,
            dconf,
            model,
            dispatch=dispatch,
            lock_ratios=lock_ratios,
            initial_params=initial_params,
            initial_step=initial_step,
            select_best_synced_params=select_best_synced_params,
            best_synced_score_fn=best_synced_score_fn,
            best_synced_initial_score=best_synced_initial_score,
        )
    )


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
