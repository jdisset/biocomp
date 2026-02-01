"""TU masking strategies - SSOT for all TU masking behavior.

This module provides the single source of truth for TU masking. All TU masking
mode detection/branching happens here and ONLY here.

Strategies:
- NoMaskingStrategy: All TUs always enabled, no params
- DirectLogAlphaStrategy: Direct log_alpha params per TU (standard)
- LatentMLPStrategy: Latent z -> MLP -> log_alpha (more expressive)
- BinaryMaskStrategy: Fixed binary mask, used by hard pruning

Shape Contract:
    Strategy methods operate on "inner params" - params already sliced to one
    (replicate, target) by the vmapped loss function:
    - Full shape at init: (n_rep, n_tgt, n_net, n_tus)
    - Inner shape at get_*: (n_net, n_tus)
    - get_log_alpha(params, network_id) indexes params[PATH][network_id] -> (n_tus,)
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Protocol

import jax
import jax.numpy as jnp

from biocomp.tumasking import (
    LATENT_TU_B1_PATH,
    LATENT_TU_B2_PATH,
    LATENT_TU_W1_PATH,
    LATENT_TU_W2_PATH,
    LATENT_TU_Z_PATH,
    PROTECTED_TU_MASK_PATH,
    TU_BINARY_MASK_PATH,
    TU_LOG_ALPHA_PATH,
    decode_latent_tu_masking,
)

if TYPE_CHECKING:
    from biocomp.parameters import ParameterTree

PROTECTED_LOG_ALPHA = 10.0  # sigmoid(10) ~ 0.99995, guarantees always-enabled


class TUMaskingMode(Enum):
    NONE = "none"
    DIRECT = "direct"
    LATENT_MLP = "latent_mlp"
    BINARY = "binary"


class TUMaskingStrategy(Protocol):
    """Interface for TU masking strategies. SSOT for all TU masking behavior."""

    @property
    def has_masking(self) -> bool:
        """True if this strategy applies TU masking. Used to skip TU penalties."""
        ...

    @property
    def mode(self) -> TUMaskingMode:
        """The masking mode for this strategy."""
        ...

    def init_params(
        self,
        params: "ParameterTree",
        *,
        n_replicates: int,
        n_targets: int,
        n_networks: int,
        n_tus: int,
        key: jax.Array,
        protected_tu_ids: set[str],
        tu_id_to_idx: dict[str, int],
    ) -> None:
        """Initialize TU masking params. Writes to full (n_rep, n_tgt, ...) shapes."""
        ...

    def get_log_alpha(self, params: "ParameterTree", network_id: int) -> jax.Array:
        """Get log_alpha for one network from INNER params (already sliced).

        Args:
            params: Inner params with TU leaves shaped (n_networks, n_tus) or similar.
            network_id: Index into first dimension.

        Returns:
            Shape (n_tus,). Protected TUs get PROTECTED_LOG_ALPHA with stop_gradient.
        """
        ...

    def get_binary_mask(self, params: "ParameterTree", network_id: int) -> jax.Array:
        """Get binary mask for one network. Returns (n_tus,) with 0.0 or 1.0."""
        ...

    @property
    def param_paths(self) -> tuple[str, ...]:
        """Paths this strategy writes to."""
        ...


class NoMaskingStrategy:
    """All TUs always enabled. No params, no penalties."""

    def __init__(self, n_tus: int = 0):
        self._n_tus = n_tus

    @property
    def has_masking(self) -> bool:
        return False

    @property
    def mode(self) -> TUMaskingMode:
        return TUMaskingMode.NONE

    def init_params(self, params: "ParameterTree", *, n_tus: int, **_) -> None:
        self._n_tus = n_tus

    def get_log_alpha(self, params: "ParameterTree", network_id: int) -> jax.Array:
        return jnp.full((self._n_tus,), PROTECTED_LOG_ALPHA)

    def get_binary_mask(self, params: "ParameterTree", network_id: int) -> jax.Array:
        return jnp.ones(self._n_tus, dtype=jnp.float32)

    @property
    def param_paths(self) -> tuple[str, ...]:
        return ()


class DirectLogAlphaStrategy:
    """Direct log_alpha params per TU."""

    def __init__(self, init_mean: float = 2.0, init_std: float = 0.5):
        self.init_mean = init_mean
        self.init_std = init_std

    @property
    def has_masking(self) -> bool:
        return True

    @property
    def mode(self) -> TUMaskingMode:
        return TUMaskingMode.DIRECT

    def init_params(
        self,
        params: "ParameterTree",
        *,
        n_replicates: int,
        n_targets: int,
        n_networks: int,
        n_tus: int,
        key: jax.Array,
        protected_tu_ids: set[str],
        tu_id_to_idx: dict[str, int],
    ) -> None:
        log_alpha = self.init_mean + self.init_std * jax.random.normal(
            key, shape=(n_replicates, n_targets, n_networks, n_tus)
        )
        params.at(TU_LOG_ALPHA_PATH, log_alpha, overwrite=None)

        protected_mask_1d = jnp.zeros(n_tus, dtype=bool)
        for tu_id in protected_tu_ids:
            if tu_id in tu_id_to_idx:
                idx = tu_id_to_idx[tu_id]
                protected_mask_1d = protected_mask_1d.at[idx].set(True)
        protected_mask = jnp.tile(protected_mask_1d[None, None, :], (n_replicates, n_targets, 1))
        params.at(PROTECTED_TU_MASK_PATH, protected_mask, overwrite=None, tags=["non_grad"])

    def get_log_alpha(self, params: "ParameterTree", network_id: int) -> jax.Array:
        """Get log_alpha from INNER params. Protected TU enforcement here ONLY.

        For protected TUs:
        - Returns PROTECTED_LOG_ALPHA (constant, not from params)
        - Wrapped in stop_gradient -> zero gradient to TU_LOG_ALPHA_PATH[..., protected_idx]
        """
        raw_log_alpha = params[TU_LOG_ALPHA_PATH][network_id]
        protected_mask = params[PROTECTED_TU_MASK_PATH]

        return jnp.where(
            protected_mask,
            jax.lax.stop_gradient(jnp.full_like(raw_log_alpha, PROTECTED_LOG_ALPHA)),
            raw_log_alpha,
        )

    def get_binary_mask(self, params: "ParameterTree", network_id: int) -> jax.Array:
        log_alpha = self.get_log_alpha(params, network_id)
        return (jax.nn.sigmoid(log_alpha) >= 0.5).astype(jnp.float32)

    @property
    def param_paths(self) -> tuple[str, ...]:
        return (TU_LOG_ALPHA_PATH, PROTECTED_TU_MASK_PATH)


class LatentMLPStrategy:
    """Latent MLP decodes z -> log_alpha via 2-layer MLP with GELU."""

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 32,
        init_mean: float = 2.0,
        init_std: float = 0.5,
    ):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.init_mean = init_mean
        self.init_std = init_std

    @property
    def has_masking(self) -> bool:
        return True

    @property
    def mode(self) -> TUMaskingMode:
        return TUMaskingMode.LATENT_MLP

    def init_params(
        self,
        params: "ParameterTree",
        *,
        n_replicates: int,
        n_targets: int,
        n_networks: int,
        n_tus: int,
        key: jax.Array,
        protected_tu_ids: set[str],
        tu_id_to_idx: dict[str, int],
    ) -> None:
        """Initialize latent MLP params.

        Architecture: log_alpha = W2 @ gelu(W1 @ z + b1) + b2

        Init strategy for approximate N(init_mean, init_std) output:
        - z ~ N(0, 0.1^2): small so MLP(z) ~ MLP(0) = b2 initially
        - W1 ~ N(0, sqrt(2/latent_dim)): He init for GELU
        - b1 = 0
        - W2 ~ N(0, 0.1 * sqrt(2/hidden_dim)): small to reduce output variance
        - b2 = init_mean: so MLP(0) ~ init_mean
        """
        k_z, k_w1, k_w2 = jax.random.split(key, 3)

        R, T, N = n_replicates, n_targets, n_networks
        L, H = self.latent_dim, self.hidden_dim

        z = jax.random.normal(k_z, (R, T, N, L)) * 0.1
        W1 = jax.random.normal(k_w1, (R, T, N, H, L)) * jnp.sqrt(2.0 / L)
        b1 = jnp.zeros((R, T, N, H))
        W2 = jax.random.normal(k_w2, (R, T, N, n_tus, H)) * jnp.sqrt(2.0 / H) * 0.1
        b2 = jnp.full((R, T, N, n_tus), self.init_mean)

        params.at(LATENT_TU_Z_PATH, z, overwrite=None)
        params.at(LATENT_TU_W1_PATH, W1, overwrite=None)
        params.at(LATENT_TU_B1_PATH, b1, overwrite=None)
        params.at(LATENT_TU_W2_PATH, W2, overwrite=None)
        params.at(LATENT_TU_B2_PATH, b2, overwrite=None)

        protected_mask_1d = jnp.zeros(n_tus, dtype=bool)
        for tu_id in protected_tu_ids:
            if tu_id in tu_id_to_idx:
                idx = tu_id_to_idx[tu_id]
                protected_mask_1d = protected_mask_1d.at[idx].set(True)
        protected_mask = jnp.tile(protected_mask_1d[None, None, :], (R, T, 1))
        params.at(PROTECTED_TU_MASK_PATH, protected_mask, overwrite=None, tags=["non_grad"])

    def get_log_alpha(self, params: "ParameterTree", network_id: int) -> jax.Array:
        """Decode latent MLP -> log_alpha, enforce protected TUs.

        Inner param shapes after slicing:
        - z: (n_networks, latent_dim)
        - W1: (n_networks, hidden_dim, latent_dim)
        - b1: (n_networks, hidden_dim)
        - W2: (n_networks, n_tus, hidden_dim)
        - b2: (n_networks, n_tus)
        """
        z = params[LATENT_TU_Z_PATH][network_id]
        W1 = params[LATENT_TU_W1_PATH][network_id]
        b1 = params[LATENT_TU_B1_PATH][network_id]
        W2 = params[LATENT_TU_W2_PATH][network_id]
        b2 = params[LATENT_TU_B2_PATH][network_id]

        raw_log_alpha = decode_latent_tu_masking(z, W1, b1, W2, b2)
        protected_mask = params[PROTECTED_TU_MASK_PATH]

        return jnp.where(
            protected_mask,
            jax.lax.stop_gradient(jnp.full_like(raw_log_alpha, PROTECTED_LOG_ALPHA)),
            raw_log_alpha,
        )

    def get_binary_mask(self, params: "ParameterTree", network_id: int) -> jax.Array:
        log_alpha = self.get_log_alpha(params, network_id)
        return (jax.nn.sigmoid(log_alpha) >= 0.5).astype(jnp.float32)

    @property
    def param_paths(self) -> tuple[str, ...]:
        return (
            LATENT_TU_Z_PATH,
            LATENT_TU_W1_PATH,
            LATENT_TU_B1_PATH,
            LATENT_TU_W2_PATH,
            LATENT_TU_B2_PATH,
            PROTECTED_TU_MASK_PATH,
        )


class BinaryMaskStrategy:
    """Fixed binary mask, used by hard pruning. Mask values are not optimized.

    Shape handling:
    - init_params() creates FULL shape: (n_rep, n_tgt, n_net, n_tus)
    - get_*() methods receive INNER params (already sliced by vmap): (n_net, n_tus)
    - set_mask() writes FULL shape (must be called OUTSIDE the vmapped context)
    """

    def __init__(self):
        self._n_tus = 0

    @property
    def has_masking(self) -> bool:
        return True

    @property
    def mode(self) -> TUMaskingMode:
        return TUMaskingMode.BINARY

    def init_params(
        self,
        params: "ParameterTree",
        *,
        n_replicates: int,
        n_targets: int,
        n_networks: int,
        n_tus: int,
        **_,
    ) -> None:
        self._n_tus = n_tus
        binary_mask = jnp.ones((n_replicates, n_targets, n_networks, n_tus))
        params.at(TU_BINARY_MASK_PATH, binary_mask, overwrite=None, tags=["non_grad"])

    def get_log_alpha(self, params: "ParameterTree", network_id: int) -> jax.Array:
        binary_mask = params[TU_BINARY_MASK_PATH][network_id]
        return jnp.where(binary_mask > 0.5, PROTECTED_LOG_ALPHA, -PROTECTED_LOG_ALPHA)

    def get_binary_mask(self, params: "ParameterTree", network_id: int) -> jax.Array:
        return params[TU_BINARY_MASK_PATH][network_id]

    def set_mask(self, params: "ParameterTree", binary_mask: jax.Array) -> None:
        """Update binary mask. Called OUTSIDE vmap, writes FULL shape."""
        params.at(TU_BINARY_MASK_PATH, binary_mask, overwrite=True)

    @property
    def param_paths(self) -> tuple[str, ...]:
        return (TU_BINARY_MASK_PATH,)


def build_tu_masking_strategy(
    mode: TUMaskingMode,
    *,
    init_mean: float = 2.0,
    init_std: float = 0.5,
    latent_dim: int = 16,
    hidden_dim: int = 32,
) -> TUMaskingStrategy:
    """Build strategy from mode. Single factory, single source of truth."""
    if mode == TUMaskingMode.NONE:
        return NoMaskingStrategy()
    elif mode == TUMaskingMode.DIRECT:
        return DirectLogAlphaStrategy(init_mean=init_mean, init_std=init_std)
    elif mode == TUMaskingMode.LATENT_MLP:
        return LatentMLPStrategy(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            init_mean=init_mean,
            init_std=init_std,
        )
    elif mode == TUMaskingMode.BINARY:
        return BinaryMaskStrategy()
    else:
        raise ValueError(f"Unknown TU masking mode: {mode}")


def build_strategy_from_config(dconf) -> TUMaskingStrategy:
    """Build strategy from DesignConfig. Convenience wrapper."""
    tu_cfg = dconf.tu_masking
    return build_tu_masking_strategy(
        mode=tu_cfg.mode,
        init_mean=tu_cfg.init_mean,
        init_std=tu_cfg.init_std,
        latent_dim=tu_cfg.latent_dim,
        hidden_dim=tu_cfg.hidden_dim,
    )


def _apply_protected_mask(
    raw_log_alpha: jax.Array,
    protected_mask: jax.Array | None,
) -> jax.Array:
    """Apply protected TU enforcement to log_alpha values.

    Protected TUs are forced to PROTECTED_LOG_ALPHA with stop_gradient to ensure:
    1. Forward pass always treats them as enabled (sigmoid(10) ~ 1)
    2. No gradient flows to their underlying params
    """
    if protected_mask is None:
        return raw_log_alpha
    return jnp.where(
        protected_mask,
        jax.lax.stop_gradient(jnp.full_like(raw_log_alpha, PROTECTED_LOG_ALPHA)),
        raw_log_alpha,
    )


def get_full_log_alpha(params: "ParameterTree") -> jax.Array | None:
    """Extract log_alpha from FULL params with protected TU enforcement.

    SSOT for mode detection AND protected TU handling.
    Used by loss functions, diagnostics, and commit that need log_alpha.
    Returns None if no TU masking params present.

    Priority: binary_mask > latent MLP > direct log_alpha
    For binary_mask, converts 1.0 -> 10.0, 0.0 -> -10.0 (pseudo log_alpha).

    Protected TUs are enforced to PROTECTED_LOG_ALPHA with stop_gradient,
    ensuring they are always enabled and receive no gradient pressure.

    Shape: (n_networks, n_tus) or (n_targets, n_networks, n_tus) or
    (n_rep, n_tgt, n_net, n_tus) depending on context.
    """
    if TU_BINARY_MASK_PATH in params:
        binary_mask = jnp.asarray(params[TU_BINARY_MASK_PATH])
        return jnp.where(binary_mask > 0.5, PROTECTED_LOG_ALPHA, -PROTECTED_LOG_ALPHA)

    protected_mask = None
    if PROTECTED_TU_MASK_PATH in params:
        protected_mask = jnp.asarray(params[PROTECTED_TU_MASK_PATH])

    if LATENT_TU_Z_PATH in params:
        z = params[LATENT_TU_Z_PATH]
        W1 = params[LATENT_TU_W1_PATH]
        b1 = params[LATENT_TU_B1_PATH]
        W2 = params[LATENT_TU_W2_PATH]
        b2 = params[LATENT_TU_B2_PATH]
        if z.ndim == 2:  # (n_net, latent_dim) - commit context (already sliced)
            raw = jax.vmap(decode_latent_tu_masking)(z, W1, b1, W2, b2)
        elif z.ndim == 3:  # (n_tgt, n_net, latent_dim) - inside replicate vmap
            raw = jax.vmap(jax.vmap(decode_latent_tu_masking))(z, W1, b1, W2, b2)
        elif z.ndim == 4:  # (n_rep, n_tgt, n_net, latent_dim) - full params
            raw = jax.vmap(jax.vmap(jax.vmap(decode_latent_tu_masking)))(z, W1, b1, W2, b2)
        else:
            raise ValueError(f"Unexpected z.ndim={z.ndim}")
        return _apply_protected_mask(raw, protected_mask)

    if TU_LOG_ALPHA_PATH in params:
        raw = jnp.asarray(params[TU_LOG_ALPHA_PATH])
        return _apply_protected_mask(raw, protected_mask)

    return None
