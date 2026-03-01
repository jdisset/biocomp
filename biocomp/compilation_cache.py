"""Compilation caching for training and inference.

Two-level cache:
1. In-process dict cache (by signature) — avoids recompilation
   when loggers create new NetworkModel instances within the same run.
2. Disk cache via serialize_executable — avoids recompilation across
   process restarts (best-effort; some executables can't be deserialized
   due to XLA backend limitations like unsupported iota instruction).
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
import json
import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Callable, TypeVar

from biocomp.logging_config import get_logger
from biocomp import utils as ut

if TYPE_CHECKING:
    from biocomp.compute import ComputeConfig, ComputeStack
    from biocomp.optimutils import TrainingConfig

logger = get_logger(__name__)

T = TypeVar("T")

COMPILATION_CACHE_DIR = Path(
    os.environ.get("BIOCOMP_COMPILATION_CACHE_DIR", os.path.expanduser("~/.cache/biocomp_compiled"))
)

_PROCESS_CACHE: dict[str, object] = {}


### {{{                     --     fingerprinting     --


def _collect_module_source_files(mod_name: str) -> list[Path]:
    """Get all source files for a module. If it's a package, include all .py files."""
    mod = importlib.import_module(mod_name)
    source_file = Path(inspect.getfile(mod))

    if source_file.name == "__init__.py":
        return sorted(source_file.parent.glob("*.py"))
    return [source_file]


def _hash_module_sources(module_names: frozenset[str]) -> str:
    """SHA-256 of source files for the given modules."""
    h = hashlib.sha256()
    for mod_name in sorted(module_names):
        try:
            for source_file in _collect_module_source_files(mod_name):
                h.update(source_file.read_bytes())
        except (TypeError, OSError) as e:
            logger.debug(f"Could not hash source for {mod_name}: {e}")
            h.update(mod_name.encode())
    return h.hexdigest()


def source_fingerprint(config: ComputeConfig) -> str:
    """SHA-256 of source files of all node implementation modules + compute/parameters."""
    modules_to_hash = {"biocomp.compute", "biocomp.parameters"}

    if config.node_functions:
        for pf in config.node_functions.values():
            func_str = pf.func if isinstance(pf.func, str) else ut.get_fname(pf.func)
            module_name = func_str.rsplit(".", 1)[0] if "." in func_str else None
            if module_name:
                modules_to_hash.add(module_name)

    return _hash_module_sources(frozenset(modules_to_hash))


@functools.lru_cache(maxsize=1)
def device_fingerprint() -> str:
    """Hash of device info + JAX/jaxlib versions. Cached (devices don't change in-process)."""
    import jax
    import jaxlib

    h = hashlib.sha256()
    devices = jax.devices()
    h.update(devices[0].device_kind.encode())
    h.update(devices[0].platform.encode())
    h.update(str(len(devices)).encode())
    h.update(jax.__version__.encode())
    h.update(jaxlib.version.__version__.encode())
    return h.hexdigest()


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                --     compilation signature     --


class CompilationSignature:
    """Deterministic signature capturing everything that affects compiled behavior."""

    @staticmethod
    def for_stack(stack: ComputeStack) -> str:
        """Signature for a built compute stack."""
        h = hashlib.sha256()

        for net in stack.networks:
            h.update(
                json.dumps(
                    net.to_recipe().model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            )

        if stack.config:
            h.update(
                json.dumps(
                    stack.config.model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            )

        h.update(str(stack.n_tus).encode())
        if stack.tu_id_to_idx:
            h.update(json.dumps(sorted(stack.tu_id_to_idx.items())).encode())

        if stack.config:
            h.update(source_fingerprint(stack.config).encode())

        h.update(device_fingerprint().encode())

        return h.hexdigest()

    @staticmethod
    def for_training_step(
        stack_sig: str,
        training_config_dump: str,
        loss_source_hash: str,
        arg_shapes: dict[str, object],
    ) -> str:
        """Signature for a full training step compilation."""
        h = hashlib.sha256()
        h.update(stack_sig.encode())
        h.update(training_config_dump.encode())
        h.update(loss_source_hash.encode())
        h.update(json.dumps(arg_shapes, sort_keys=True).encode())
        return h.hexdigest()


def training_config_compilation_dump(training_config: TrainingConfig) -> str:
    """JSON dump of training config fields that affect compilation.

    Excludes seed, n_epochs, n_batches, keep_in_history, streaming_batches,
    clear_source_data — these don't affect the compiled function's structure.
    """
    d = training_config.model_dump(mode="json")
    for key in (
        "seed",
        "n_epochs",
        "n_batches",
        "keep_in_history",
        "streaming_batches",
        "clear_source_data",
    ):
        d.pop(key, None)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                   --     compiled serializer     --


def _compiled_save(compiled, path: Path):
    """Save a jax.stages.Compiled object to disk (best-effort)."""
    from jax.experimental.serialize_executable import serialize

    serialized_bytes, in_tree, out_tree = serialize(compiled)
    with open(path, "wb") as f:
        pickle.dump((serialized_bytes, in_tree, out_tree), f)


def _compiled_load(path: Path):
    """Load a jax.stages.Compiled object from disk.

    If deserialization fails (e.g. unsupported instruction opcode),
    deletes the corrupt cache file and raises.
    """
    from jax.experimental.serialize_executable import deserialize_and_load

    with open(path, "rb") as f:
        serialized_bytes, in_tree, out_tree = pickle.load(f)
    try:
        return deserialize_and_load(serialized_bytes, in_tree, out_tree)
    except Exception:
        logger.warning(f"Deserialization failed for {path.name}, removing corrupt cache file")
        path.unlink(missing_ok=True)
        raise


COMPILED_SERIALIZER: dict[str, Callable[..., object]] = {
    "save": _compiled_save,
    "load": _compiled_load,
}


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                    --     cached compilation     --


def cached_compile(
    compile_fn: Callable[[], T],
    signature: str,
    cache_dir: Path | str | None = None,
) -> T:
    """Cache a compiled JAX executable with two-level caching.

    Level 1: In-process dict cache (always hits within same run).
    Level 2: Disk cache via serialize_executable (best-effort, may fail
             for some executables due to XLA backend limitations).

    Args:
        compile_fn: zero-arg callable that returns a jax.stages.Compiled
        signature: deterministic cache key
        cache_dir: override cache directory (defaults to COMPILATION_CACHE_DIR)
    """
    if signature in _PROCESS_CACHE:
        logger.debug(f"In-process cache hit for {signature[:16]}")
        return _PROCESS_CACHE[signature]  # type: ignore[return-value]

    if cache_dir is None:
        cache_dir = COMPILATION_CACHE_DIR
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)

    try:
        result = ut.get_cache(
            gen_f=compile_fn,
            signature=signature,
            cache_location=cache_dir,
            serializer=COMPILED_SERIALIZER,
        )
    except Exception as e:
        logger.info(f"Disk cache failed ({type(e).__name__}: {e}), compiling fresh")
        result = compile_fn()

    _PROCESS_CACHE[signature] = result
    return result  # type: ignore[return-value]


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                   --     helper utilities     --


def loss_function_source_hash(training_config: TrainingConfig) -> str:
    """Hash the source of the loss function module used in training."""
    loss_func_str = training_config.loss_function.func
    if isinstance(loss_func_str, str) and "." in loss_func_str:
        module_name = loss_func_str.rsplit(".", 1)[0]
    else:
        module_name = "biocomp.train"
    return _hash_module_sources(frozenset({module_name}))


def extract_arg_shapes(*args: object) -> dict[str, object]:
    """Extract shapes from sample arguments for signature computation."""
    from biocomp.parameters import ParameterTree

    def _shape_of(x: object) -> object:
        s = getattr(x, "shape", None)
        return list(s) if s is not None else str(type(x))

    shapes: dict[str, object] = {}
    for i, arg in enumerate(args):
        if isinstance(arg, ParameterTree):
            leaf_shapes: dict[str, object] = {}
            for path, leaf in arg.data.iter_leaves():  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]
                leaf_shapes[str(path)] = _shape_of(leaf)
            shapes[f"arg_{i}_ptree"] = leaf_shapes
        else:
            shapes[f"arg_{i}"] = _shape_of(arg)
    return shapes


##────────────────────────────────────────────────────────────────────────────}}}
