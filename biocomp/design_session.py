"""DesignSession: shared initialization for design optimization runners.

This module provides a single source of truth for design session setup,
eliminating duplicated initialization code across run_design, run_pluggable,
and run_with_hard_pruning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable
import time

import jax
import jax.numpy as jnp
from jax.tree_util import tree_leaves

if TYPE_CHECKING:
    from .compute import ComputeStack
    from .parameters import ParameterTree
    from .design import DesignManager, DesignConfig
    from .tumasking_strategy import TUMaskingStrategy
    from biocomptools.modelmodel import BiocompModel

from .logging_config import get_logger

logger = get_logger(__name__)


class PhaseTimer:
    """Minimal helper for timing optimization phases."""

    def __init__(self):
        self._timings: dict[str, float] = {}
        self._t0 = time.perf_counter()
        self._phase_start: float = time.perf_counter()

    def start(self, name: str, msg: str):
        logger.info(msg)
        self._phase_start = time.perf_counter()

    def end(self, name: str):
        self._timings[name] = time.perf_counter() - self._phase_start
        logger.info(f"  -> {self._timings[name]:.2f}s")

    def total(self) -> float:
        return time.perf_counter() - self._t0

    def summary(self):
        total = self.total()
        for name, t in self._timings.items():
            logger.info(f"  {name:15s} {t:.2f}s ({t / total * 100:.1f}%)")


@dataclass
class DesignSession:
    """Encapsulates shared design session state and initialization.

    This is the single source of truth for design session setup. All design
    runners (run_design, run_pluggable, run_with_hard_pruning) should use
    DesignSession.create() instead of duplicating initialization code.
    """

    dmanager: "DesignManager"
    dconf: "DesignConfig"
    model: "BiocompModel"
    stack: "ComputeStack"
    strategy: "TUMaskingStrategy"
    initial_params: "ParameterTree"
    xbatches: jnp.ndarray
    ybatches: jnp.ndarray
    loss_fn: Callable
    num_z: tuple[int, int]
    direct_ratio_paths: list[str]
    source_ratio_paths: list[str]
    timer: PhaseTimer
    keys: tuple[jax.Array, jax.Array, jax.Array]
    effective_batch_size: int
    steps_per_epoch: int
    total_steps: int
    n_design_inputs: int = field(init=False)

    def __post_init__(self):
        self.n_design_inputs = 2 * len(self.dmanager.networks)

    @classmethod
    def create(
        cls,
        dmanager: "DesignManager",
        dconf: "DesignConfig",
        model: "BiocompModel",
        lock_ratios: bool = False,
        initial_params: "ParameterTree | None" = None,
        n_replicates_override: int | None = None,
        sample_shape_override: tuple[int, ...] | None = None,
    ) -> "DesignSession":
        """Create a design session with all shared initialization."""
        from .design import initialize_params, _create_loss_function, get_ratio_paths_and_sources

        timer = PhaseTimer()
        logger.info("=" * 60)
        logger.info("DESIGN SESSION INITIALIZATION")
        logger.info("=" * 60)

        pkey, bkey, loop_key = jax.random.split(dconf.seed_key, 3)
        n_replicates = n_replicates_override if n_replicates_override is not None else dconf.n_replicates

        strategy = dconf.build_tu_masking_strategy()
        logger.info(f"TU masking strategy: {strategy.mode.value}")

        timer.start("stack", "[1/5] Building compute stack...")
        stack = dmanager.build_stack(
            model,
            unlock_ratios=not lock_ratios,
            use_latent_ratios=dconf.use_latent_ratios,
            latent_dim=dconf.latent_dim,
            latent_hidden_dim=dconf.latent_hidden_dim,
            auto_lock_topology_tus=dconf.auto_lock_topology_tus,
            enable_tu_masking=strategy.has_masking,
        )
        timer.end("stack")

        timer.start("params", "[2/5] Initializing parameters...")
        if initial_params is None:
            n_tus = stack.n_tus if strategy.has_masking else 0
            n_networks = len(dmanager.networks)
            initial_params = initialize_params(
                stack,
                n_replicates,
                dmanager.n_targets,
                model.shared_params,
                pkey,
                strategy=strategy,
                n_tus=n_tus,
                n_networks=n_networks,
                no_masking_tu_ids=stack.no_masking_tu_ids,
                tu_id_to_idx=stack.tu_id_to_idx,
            )
        timer.end("params")

        jax_leaves = tree_leaves(initial_params.filter_by_tag(["non_grad", "shared"])[1])
        if not jax_leaves:
            raise ValueError(
                "No parameters to optimize: all parameters are either shared or marked NON_GRAD. "
                "This typically happens with zero-freedom recipes where all ratios are explicitly locked."
            )

        steps_per_epoch = max(1, dconf.n_batches_per_epoch // dconf.batches_per_step)
        total_steps = int(dconf.n_epochs * steps_per_epoch)
        assert total_steps > 0, f"total_steps must be > 0, got {total_steps}"
        logger.info(
            f"  Config: {total_steps} steps, {steps_per_epoch}/epoch, "
            f"batch={dconf.batch_size}, batches/step={dconf.batches_per_step}"
        )

        timer.start("samples", "[3/5] Generating training samples...")
        if sample_shape_override is not None:
            sample_shape = sample_shape_override
        else:
            sample_shape = (
                len(dmanager.networks),
                steps_per_epoch,
                n_replicates,
                dconf.batches_per_step,
                dconf.batch_size,
            )
        xbatches_list, ybatches_list = dmanager.get_samples(
            sample_shape,
            bkey,
            share_across_networks=True,
        )
        xbatches = jnp.concatenate(xbatches_list, axis=-1)
        ybatches = ybatches_list[0]
        timer.end("samples")

        effective_batch_size = dconf.batch_size
        if dmanager.is_lattice_mode:
            grid_res = dmanager.grid_resolution
            assert grid_res is not None
            effective_batch_size *= grid_res[0] * grid_res[1]

        logger.info(f"  Data: {len(dmanager.networks)} networks, xbatches.shape={xbatches.shape}")

        timer.start("loss_fn", "[4/5] Creating loss function...")
        loss_fn, num_z, direct_ratio_paths = _create_loss_function(
            stack, dmanager, dconf, initial_params
        )
        _, source_ratio_paths = get_ratio_paths_and_sources(initial_params)
        logger.debug(
            f"Ratio paths: {len(direct_ratio_paths)} direct + {len(source_ratio_paths)} ArrayRef"
        )
        timer.end("loss_fn")

        logger.info("-" * 60)
        logger.info(f"SESSION INIT COMPLETE in {timer.total():.2f}s")
        timer.summary()

        return cls(
            dmanager=dmanager,
            dconf=dconf,
            model=model,
            stack=stack,
            strategy=strategy,
            initial_params=initial_params,
            xbatches=xbatches,
            ybatches=ybatches,
            loss_fn=loss_fn,
            num_z=num_z,
            direct_ratio_paths=direct_ratio_paths,
            source_ratio_paths=source_ratio_paths,
            timer=timer,
            keys=(pkey, bkey, loop_key),
            effective_batch_size=effective_batch_size,
            steps_per_epoch=steps_per_epoch,
            total_steps=total_steps,
        )

    @property
    def pkey(self) -> jax.Array:
        return self.keys[0]

    @property
    def bkey(self) -> jax.Array:
        return self.keys[1]

    @property
    def loop_key(self) -> jax.Array:
        return self.keys[2]

    @property
    def n_networks(self) -> int:
        return len(self.dmanager.networks)
