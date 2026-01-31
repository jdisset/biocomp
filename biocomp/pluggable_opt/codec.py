from typing import Callable

import jax.numpy as jnp
from jax import flatten_util
from jax.experimental import checkify
from pydantic import BaseModel, ConfigDict

from ..parameters import ParameterTree
from ..tumasking import TU_LOG_ALPHA_PATH, LOG_ALPHA_MIN, LOG_ALPHA_MAX
from ..design import normalize_ratios_prune, get_ratio_paths_and_sources, RATIO_PRUNE_THRESHOLD
from ..logging_config import get_logger

logger = get_logger(__name__)


class GenomeCodec(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    unravel_fn: Callable[[jnp.ndarray], ParameterTree]
    param_dim: int
    static_params: ParameterTree
    static_tags: tuple[str, ...]
    ratio_prune_threshold: float = RATIO_PRUNE_THRESHOLD
    tu_log_alpha_min: float = LOG_ALPHA_MIN
    tu_log_alpha_max: float = LOG_ALPHA_MAX
    direct_ratio_paths: tuple[str, ...] = ()
    source_ratio_paths: tuple[str, ...] = ()
    use_latent_ratios: bool = False

    @staticmethod
    def from_params(
        params: ParameterTree, static_tags: tuple[str, ...] = ("shared", "non_grad"), **cfg
    ) -> "GenomeCodec":
        static, dynamic = params.filter_by_tag(list(static_tags))
        flat, unravel = flatten_util.ravel_pytree(dynamic)
        direct, source = get_ratio_paths_and_sources(params)
        logger.info(
            f"GenomeCodec: {flat.shape[0]} params, {len(direct)} direct + {len(source)} source ratio paths"
        )
        return GenomeCodec(
            unravel_fn=unravel,
            param_dim=int(flat.shape[0]),
            static_params=static,
            static_tags=static_tags,
            direct_ratio_paths=tuple(str(p) for p in direct),
            source_ratio_paths=tuple(source),
            **cfg,
        )

    def encode(self, params: ParameterTree) -> jnp.ndarray:
        _, dynamic = params.filter_by_tag(list(self.static_tags))
        flat, _ = flatten_util.ravel_pytree(dynamic)
        assert flat.shape[0] == self.param_dim, f"dim {flat.shape[0]} != {self.param_dim}"
        return flat

    def decode(self, genome: jnp.ndarray, apply_constraints: bool = True) -> ParameterTree:
        assert genome.shape[-1] == self.param_dim, f"dim {genome.shape[-1]} != {self.param_dim}"
        dynamic = self.unravel_fn(genome)
        params = (
            ParameterTree.merge(self.static_params, dynamic) if self.static_params.data else dynamic
        )
        return self._apply_constraints(params) if apply_constraints else params

    def _apply_constraints(self, params: ParameterTree) -> ParameterTree:
        if not self.use_latent_ratios:
            def normalize(x):
                return normalize_ratios_prune(x, threshold=self.ratio_prune_threshold)

            if self.direct_ratio_paths:
                params = params.update_leaves_by_path(list(self.direct_ratio_paths), normalize)
            if self.source_ratio_paths:
                from ..design import normalize_ratio_source_arrays

                params = normalize_ratio_source_arrays(params, list(self.source_ratio_paths), normalize)
        if TU_LOG_ALPHA_PATH in params:
            params = params.update_leaves_by_path(
                [TU_LOG_ALPHA_PATH],
                lambda x: jnp.clip(x, self.tu_log_alpha_min, self.tu_log_alpha_max),
            )
        return params

    def validate_genome(self, genome: jnp.ndarray) -> None:
        checkify.check(genome.shape[-1] == self.param_dim, "genome dim mismatch")
        checkify.check(jnp.isfinite(genome).all(), "genome non-finite")

    def bounds(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.full(self.param_dim, -jnp.inf), jnp.full(self.param_dim, jnp.inf)
