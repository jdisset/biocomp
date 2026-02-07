import os as _os

# enable JAX persistent compilation cache for faster repeated runs
_jax_cache_dir = _os.environ.get(
    "JAX_COMPILATION_CACHE_DIR", _os.path.expanduser("~/.cache/jax_biocomp")
)
if _jax_cache_dir:
    _os.makedirs(_jax_cache_dir, exist_ok=True)
    _os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _jax_cache_dir)

import jax as _jax  # noqa: E402

_jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
_jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

from .library import PartsLibrary as PartsLibrary  # noqa: E402

from .network import (  # noqa: E402
    Network as Network,
    recipe_to_networks as recipe_to_networks,
)

from .recipe import (  # noqa: E402
    Recipe as Recipe,
    CoTransfection as CoTransfection,
    TranscriptionUnit as TranscriptionUnit,
    Slot as Slot,
)

from .designloss import GridLossWeights as GridLossWeights  # noqa: E402

from .pluggable_opt.codec import GenomeCodec as GenomeCodec  # noqa: E402

from .pluggable_opt.optimizers import (  # noqa: E402
    OptimPhase as OptimPhase,
    OptimizationState as OptimizationState,
    GradientDescentOptimizer as GradientDescentOptimizer,
    EvolutionaryOptimizer as EvolutionaryOptimizer,
    HybridOptimizer as HybridOptimizer,
    ObjectiveWrapper as ObjectiveWrapper,
)
