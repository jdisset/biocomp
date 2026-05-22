# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Shim layer over jeanplot's 2D smooth-heatmap kernels.

Re-exports ``smooth_2d``, ``smooth_grad_magnitude_2d``, ``gradient_field_2d``,
``knn_grid``, ``knn_gradient_grid``, ``KnnGradientField`` from
``jeanplot.plots.smooth_2d`` / ``jeanplot.plots.smooth_kernel`` and the
``GridData``/serialization helpers from ``jeanplot.data.grid``. Renders use
biocomp's default colormap config (registered at import time by
``biocomp.plotting.plotting_core``).
"""

import matplotlib as mpl  # noqa: F401  (imported for legacy callers)
import matplotlib.pyplot as plt  # noqa: F401  (imported for legacy callers)
import numpy as np  # noqa: F401  (imported for legacy callers)

from jeanplot.data.grid import (  # noqa: F401
    GridData,
    extract_grid_data,
    grid_data_from_b64,
    grid_data_to_b64,
)
from jeanplot.plots.colorbar import colorbar  # noqa: F401
from jeanplot.plots.smooth_2d import (  # noqa: F401
    KnnGradientField,
    gradient_field_2d,
    knn_gradient_grid,
    smooth_2d,
    smooth_grad_magnitude_2d,
)
from jeanplot.plots.smooth_kernel import (  # noqa: F401
    _KNN_GRID_CACHE,
    _KNN_GRID_CACHE_MAX,
    _finite_xy,
    _knn_grid_cache_key,
    _render_smooth_heatmap,
    _resolve_lims,
    _resolve_vlims,
    knn_grid,
)

# Force biocomp default colormap registration at import time. Without this,
# jeanplot's heatmap() may run before biocomp.plotting.plotting_core has
# registered the custom cmaps.
from . import plotting_core as _pc  # noqa: F401


def print_rc_params():
    for key, value in mpl.rcParams.items():
        print(f"{key}: {value}")


__all__ = [
    "GridData",
    "KnnGradientField",
    "_KNN_GRID_CACHE",
    "_KNN_GRID_CACHE_MAX",
    "_finite_xy",
    "_knn_grid_cache_key",
    "_render_smooth_heatmap",
    "_resolve_lims",
    "_resolve_vlims",
    "colorbar",
    "extract_grid_data",
    "gradient_field_2d",
    "grid_data_from_b64",
    "grid_data_to_b64",
    "knn_gradient_grid",
    "knn_grid",
    "print_rc_params",
    "smooth_2d",
    "smooth_grad_magnitude_2d",
]
