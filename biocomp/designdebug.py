"""Debug state capture for design/plot operations.

Enable via env vars: BIOCOMP_DEBUG_DESIGN=1 or BIOCOMP_DEBUG_PLOT=1
Or via config: debug.design.enabled / debug.plot.enabled

Saves pickles to {output_dir}/_debug_dumps/ with stage, timestamp, shapes, stats, data, metadata.
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

_save_lock = Lock()
_save_counters: dict[str, int] = {}


def _get_debug_config() -> dict:
    try:
        from biocomptools.toollib.config import config
        return config.get('debug', {})
    except Exception:
        return {}


def is_design_debug_enabled() -> bool:
    env_val = os.environ.get("BIOCOMP_DEBUG_DESIGN", "")
    if env_val:
        return env_val.lower() in ("1", "true", "yes", "on")
    return bool(_get_debug_config().get('design', {}).get('enabled', False))


def is_plot_debug_enabled() -> bool:
    env_val = os.environ.get("BIOCOMP_DEBUG_PLOT", "")
    if env_val:
        return env_val.lower() in ("1", "true", "yes", "on")
    return bool(_get_debug_config().get('plot', {}).get('enabled', False))


def _get_debug_dir(output_dir: str | Path) -> Path:
    debug_dir = Path(output_dir) / "_debug_dumps"
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


def _get_next_counter(debug_dir: Path) -> int:
    with _save_lock:
        key = str(debug_dir)
        if key not in _save_counters:
            existing = list(debug_dir.glob("*.pickle"))
            nums = []
            for f in existing:
                try:
                    nums.append(int(f.name.split("_")[0]))
                except (ValueError, IndexError):
                    pass
            _save_counters[key] = (max(nums) + 1) if nums else 1
        counter = _save_counters[key]
        _save_counters[key] += 1
        return counter


def _to_numpy(val: Any) -> Any:
    if hasattr(val, "__array__"):
        return np.asarray(val)
    if isinstance(val, dict):
        return {k: _to_numpy(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return type(val)(_to_numpy(v) for v in val)
    return val


def _compute_stats(arr: np.ndarray) -> dict:
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
    if not force:
        enabled = is_design_debug_enabled() if mode == "design" else is_plot_debug_enabled()
        if not enabled:
            return None

    if output_dir is None:
        output_dir = Path(os.environ.get("BIOCOMP_ROOT", "/tmp")) / "debug_dumps" / f"{mode}_fallback"

    debug_dir = _get_debug_dir(output_dir)
    counter = _get_next_counter(debug_dir)
    filepath = debug_dir / f"{counter:04d}_{stage}.pickle"

    data_np = _to_numpy(data)
    shapes, stats = {}, {}
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
        "stage": stage, "counter": counter, "timestamp": datetime.now().isoformat(),
        "mode": mode, "output_dir": str(output_dir), "shapes": shapes,
        "stats": stats, "data": data_np, "metadata": metadata or {},
    }
    with open(filepath, "wb") as f:
        pickle.dump(payload, f)

    logger.debug(f"[DEBUG-{mode.upper()}] Saved {stage} to {filepath}")
    return filepath


def get_debug_summary(output_dir: str | Path | None = None) -> dict:
    design_enabled, plot_enabled = is_design_debug_enabled(), is_plot_debug_enabled()
    if not design_enabled and not plot_enabled:
        return {"design_enabled": False, "plot_enabled": False, "save_count": 0}

    result = {"design_enabled": design_enabled, "plot_enabled": plot_enabled}
    if output_dir is not None:
        debug_dir = Path(output_dir) / "_debug_dumps"
        if debug_dir.exists():
            files = sorted(debug_dir.glob("*.pickle"))
            result.update({"debug_dir": str(debug_dir), "save_count": len(files), "files": [f.name for f in files]})
        else:
            result["save_count"] = 0
    return result
