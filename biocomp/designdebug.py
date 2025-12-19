"""Debug utilities for comprehensive state capture during design and plot operations.

This module provides standardized pickle saving for debugging axis inversion
and other design/plot-related issues. Debug saves are controlled by:
- biocomptools config: debug.design.enabled / debug.plot.enabled
- Environment variables: BIOCOMP_DEBUG_DESIGN / BIOCOMP_DEBUG_PLOT

Usage:
    # Enable via config (in biocomptools/configs/default.yaml):
    debug:
      design:
        enabled: true
      plot:
        enabled: true

    # Or via environment variables:
    export BIOCOMP_DEBUG_DESIGN=1
    export BIOCOMP_DEBUG_PLOT=1

Debug pickles are saved to: {output_dir}/_debug_dumps/
- For design: output_dir is the design run directory
- For plots: output_dir is the figure output directory

Each pickle contains:
- stage: descriptive name of where the save occurred
- timestamp: ISO format timestamp
- shapes: dict of array shapes
- stats: dict of array statistics (min, max, mean)
- data: dict of actual data arrays
- metadata: additional context
"""

import os
import pickle
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from biocomp.logging_config import get_logger

logger = get_logger(__name__)

# Thread-safe counter for save ordering
_save_lock = Lock()
_save_counters: dict[str, int] = {}  # per-directory counters


def _get_debug_config() -> dict:
    """Get debug config from biocomptools config, with env var fallback."""
    try:
        from biocomptools.toollib.config import config
        return config.get('debug', {})
    except Exception:
        return {}


def is_design_debug_enabled() -> bool:
    """Check if design debug mode is enabled."""
    # Check env var first (takes precedence)
    env_val = os.environ.get("BIOCOMP_DEBUG_DESIGN", "")
    if env_val:
        return env_val.lower() in ("1", "true", "yes", "on")
    # Check config
    cfg = _get_debug_config()
    return bool(cfg.get('design', {}).get('enabled', False))


def is_plot_debug_enabled() -> bool:
    """Check if plot debug mode is enabled."""
    # Check env var first (takes precedence)
    env_val = os.environ.get("BIOCOMP_DEBUG_PLOT", "")
    if env_val:
        return env_val.lower() in ("1", "true", "yes", "on")
    # Check config
    cfg = _get_debug_config()
    return bool(cfg.get('plot', {}).get('enabled', False))


def _get_debug_dir(output_dir: str | Path) -> Path:
    """Get the debug dump directory for a given output directory."""
    debug_dir = Path(output_dir) / "_debug_dumps"
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


def _get_next_counter(debug_dir: Path) -> int:
    """Get next counter for a debug directory (thread-safe)."""
    with _save_lock:
        key = str(debug_dir)
        if key not in _save_counters:
            # Initialize from existing files
            existing = list(debug_dir.glob("*.pickle"))
            if existing:
                nums = []
                for f in existing:
                    try:
                        nums.append(int(f.name.split("_")[0]))
                    except (ValueError, IndexError):
                        pass
                _save_counters[key] = max(nums) + 1 if nums else 1
            else:
                _save_counters[key] = 1
        counter = _save_counters[key]
        _save_counters[key] += 1
        return counter


def _to_numpy(val: Any) -> Any:
    """Convert JAX arrays to numpy for pickling."""
    if hasattr(val, "__array__"):
        return np.asarray(val)
    elif isinstance(val, dict):
        return {k: _to_numpy(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return type(val)(_to_numpy(v) for v in val)
    return val


def _compute_stats(arr: np.ndarray) -> dict:
    """Compute basic statistics for an array."""
    arr = np.asarray(arr)
    if arr.size == 0:
        return {"size": 0}
    try:
        return {
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr)),
            "nan_count": int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0,
        }
    except Exception:
        return {"shape": arr.shape, "dtype": str(arr.dtype), "error": "stats_failed"}


def save_debug_state(
    stage: str,
    data: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    mode: str = "design",
    force: bool = False,
) -> Path | None:
    """Save debug state to a pickle file.

    Args:
        stage: Descriptive name for this checkpoint (e.g., "get_reordered_X_entry")
        data: Dictionary of data to save (arrays, scalars, etc.)
        metadata: Additional context information
        output_dir: Directory where debug dumps should be saved.
                   If None, uses a temp fallback location.
        mode: "design" or "plot" - controls which enable flag is checked
        force: Save even if debug mode is disabled

    Returns:
        Path to saved file, or None if debug mode disabled
    """
    # Check if enabled
    if not force:
        if mode == "design" and not is_design_debug_enabled():
            return None
        elif mode == "plot" and not is_plot_debug_enabled():
            return None

    # Determine output directory
    if output_dir is None:
        # Fallback to temp location
        biocomp_root = os.environ.get("BIOCOMP_ROOT", "/tmp")
        output_dir = Path(biocomp_root) / "debug_dumps" / f"{mode}_fallback"

    debug_dir = _get_debug_dir(output_dir)
    counter = _get_next_counter(debug_dir)
    timestamp = datetime.now().isoformat()
    filename = f"{counter:04d}_{stage}.pickle"
    filepath = debug_dir / filename

    # Convert all data to numpy for pickling
    data_np = _to_numpy(data)

    # Compute shapes and stats
    shapes = {}
    stats = {}
    for key, val in data_np.items():
        if val is None:
            continue
        if hasattr(val, "shape"):
            shapes[key] = val.shape
            stats[key] = _compute_stats(val)
        elif isinstance(val, (list, tuple)) and len(val) > 0 and hasattr(val[0], "shape"):
            shapes[key] = [v.shape for v in val]
            stats[key] = [_compute_stats(v) for v in val]

    payload = {
        "stage": stage,
        "counter": counter,
        "timestamp": timestamp,
        "mode": mode,
        "output_dir": str(output_dir),
        "shapes": shapes,
        "stats": stats,
        "data": data_np,
        "metadata": metadata or {},
    }

    with open(filepath, "wb") as f:
        pickle.dump(payload, f)

    logger.debug(f"[DEBUG-{mode.upper()}] Saved {stage} to {filepath}")
    return filepath


# ============================================================================
# Design-specific debug helpers
# ============================================================================

def save_design_target_state(
    stage: str,
    target: Any,
    network: Any | None = None,
    X_original: np.ndarray | None = None,
    X_reordered: np.ndarray | None = None,
    reorder_indices: list | None = None,
    output_dir: str | Path | None = None,
):
    """Save DataTarget reordering state for debugging axis inversion."""
    if not is_design_debug_enabled():
        return

    data = {}
    metadata = {}

    if target is not None:
        metadata["target_name"] = getattr(target, "name", "unknown")
        metadata["target_type"] = type(target).__name__
        if hasattr(target, "input_names"):
            metadata["target_input_names"] = target.input_names
        if hasattr(target, "X"):
            data["target_X"] = target.X
        if hasattr(target, "Y"):
            data["target_Y"] = target.Y
        if hasattr(target, "lattice_x_extent"):
            metadata["lattice_x_extent"] = target.lattice_x_extent
            metadata["lattice_y_extent"] = target.lattice_y_extent

    if network is not None:
        metadata["network_name"] = getattr(network, "name", "unknown")
        if hasattr(network, "get_inverted_input_proteins"):
            metadata["network_input_order"] = network.get_inverted_input_proteins()

    if X_original is not None:
        data["X_original"] = X_original
    if X_reordered is not None:
        data["X_reordered"] = X_reordered
    if reorder_indices is not None:
        metadata["reorder_indices"] = reorder_indices

    save_debug_state(stage, data, metadata, output_dir=output_dir, mode="design")


def save_lattice_samples_state(
    stage: str,
    xsamples: Any,
    ysamples: Any,
    resolution: tuple | None = None,
    jitter: float | None = None,
    targets: list | None = None,
    output_dir: str | Path | None = None,
):
    """Save lattice sampling state for debugging."""
    if not is_design_debug_enabled():
        return

    data = {"xsamples": xsamples, "ysamples": ysamples}
    metadata = {}

    if resolution is not None:
        metadata["resolution"] = resolution
    if jitter is not None:
        metadata["jitter"] = jitter
    if targets is not None:
        metadata["target_names"] = [getattr(t, "name", f"target_{i}") for i, t in enumerate(targets)]
        metadata["target_input_names"] = [
            getattr(t, "input_names", None) for t in targets
        ]

    save_debug_state(stage, data, metadata, output_dir=output_dir, mode="design")


def save_design_result_state(
    stage: str,
    result: Any,
    gt_X: np.ndarray | None = None,
    gt_Y: np.ndarray | None = None,
    pred_X: np.ndarray | None = None,
    pred_Y: np.ndarray | None = None,
    lattice_X: np.ndarray | None = None,
    lattice_Y: np.ndarray | None = None,
    output_dir: str | Path | None = None,
):
    """Save DesignResult state for debugging plot axis issues."""
    if not is_design_debug_enabled():
        return

    data = {}
    metadata = {}

    if result is not None:
        metadata["target_name"] = getattr(result, "target_name", "unknown")
        metadata["rank"] = getattr(result, "rank", -1)
        metadata["replicate"] = getattr(result, "replicate", -1)
        metadata["loss"] = getattr(result, "loss", float("nan"))

        target = getattr(result, "target", None)
        if target is not None:
            metadata["target_input_names"] = getattr(target, "input_names", None)
            if hasattr(target, "original_network") and target.original_network is not None:
                metadata["original_network_inputs"] = target.original_network.get_inverted_input_proteins()

        network = getattr(result, "network", None)
        if network is not None:
            metadata["designed_network_name"] = getattr(network, "name", "unknown")
            if hasattr(network, "get_inverted_input_proteins"):
                metadata["designed_network_inputs"] = network.get_inverted_input_proteins()

    if gt_X is not None:
        data["gt_X"] = gt_X
    if gt_Y is not None:
        data["gt_Y"] = gt_Y
    if pred_X is not None:
        data["pred_X"] = pred_X
    if pred_Y is not None:
        data["pred_Y"] = pred_Y
    if lattice_X is not None:
        data["lattice_X"] = lattice_X
    if lattice_Y is not None:
        data["lattice_Y"] = lattice_Y

    save_debug_state(stage, data, metadata, output_dir=output_dir, mode="design")


def save_prediction_state(
    stage: str,
    X_input: np.ndarray | None = None,
    Y_output: np.ndarray | None = None,
    skip_input_reorder: bool | None = None,
    already_latent: bool | None = None,
    network_name: str | None = None,
    input_order: list | None = None,
    output_dir: str | Path | None = None,
    mode: str = "design",
):
    """Save NetworkPrediction state for debugging."""
    enabled = is_design_debug_enabled() if mode == "design" else is_plot_debug_enabled()
    if not enabled:
        return

    data = {}
    metadata = {}

    if X_input is not None:
        data["X_input"] = X_input
    if Y_output is not None:
        data["Y_output"] = Y_output

    if skip_input_reorder is not None:
        metadata["skip_input_reorder"] = skip_input_reorder
    if already_latent is not None:
        metadata["already_latent"] = already_latent
    if network_name is not None:
        metadata["network_name"] = network_name
    if input_order is not None:
        metadata["input_order"] = input_order

    save_debug_state(stage, data, metadata, output_dir=output_dir, mode=mode)


# ============================================================================
# Plot-specific debug helpers
# ============================================================================

def save_plot_data_state(
    stage: str,
    plot_data: Any = None,
    X: np.ndarray | None = None,
    Y: np.ndarray | None = None,
    input_names: list | None = None,
    output_name: str | None = None,
    network_name: str | None = None,
    output_dir: str | Path | None = None,
):
    """Save PlotData state for debugging plot generation."""
    if not is_plot_debug_enabled():
        return

    data = {}
    metadata = {}

    if plot_data is not None:
        if hasattr(plot_data, 'xval'):
            data["X"] = plot_data.xval
        if hasattr(plot_data, 'yval'):
            data["Y"] = plot_data.yval
        if hasattr(plot_data, 'input_names'):
            metadata["input_names"] = plot_data.input_names
        if hasattr(plot_data, 'output_name'):
            metadata["output_name"] = plot_data.output_name
        if hasattr(plot_data, 'metadata'):
            metadata["plot_metadata"] = plot_data.metadata

    if X is not None:
        data["X"] = X
    if Y is not None:
        data["Y"] = Y
    if input_names is not None:
        metadata["input_names"] = input_names
    if output_name is not None:
        metadata["output_name"] = output_name
    if network_name is not None:
        metadata["network_name"] = network_name

    save_debug_state(stage, data, metadata, output_dir=output_dir, mode="plot")


def save_figure_state(
    stage: str,
    figure_path: str | Path | None = None,
    data_sources: list | None = None,
    plot_tasks: list | None = None,
    output_dir: str | Path | None = None,
):
    """Save Figure state before rendering for debugging."""
    if not is_plot_debug_enabled():
        return

    data = {}
    metadata = {}

    if figure_path is not None:
        metadata["figure_path"] = str(figure_path)

    if data_sources is not None:
        for i, ds in enumerate(data_sources):
            if hasattr(ds, 'x'):
                data[f"data_source_{i}_X"] = ds.x
            if hasattr(ds, 'y'):
                data[f"data_source_{i}_Y"] = ds.y
            if hasattr(ds, 'metadata'):
                metadata[f"data_source_{i}_metadata"] = ds.metadata

    if plot_tasks is not None:
        metadata["n_plot_tasks"] = len(plot_tasks)

    save_debug_state(stage, data, metadata, output_dir=output_dir, mode="plot")


def get_debug_summary(output_dir: str | Path | None = None) -> dict:
    """Get summary of all debug saves in a directory."""
    design_enabled = is_design_debug_enabled()
    plot_enabled = is_plot_debug_enabled()

    if not design_enabled and not plot_enabled:
        return {"design_enabled": False, "plot_enabled": False, "save_count": 0}

    result = {
        "design_enabled": design_enabled,
        "plot_enabled": plot_enabled,
    }

    if output_dir is not None:
        debug_dir = Path(output_dir) / "_debug_dumps"
        if debug_dir.exists():
            files = sorted(debug_dir.glob("*.pickle"))
            result["debug_dir"] = str(debug_dir)
            result["save_count"] = len(files)
            result["files"] = [f.name for f in files]
        else:
            result["save_count"] = 0

    return result
