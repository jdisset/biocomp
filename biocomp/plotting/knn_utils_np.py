# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Shim layer over jeanplot's numpy KNN kernels.

For back-compat, ``BIOCOMP_KNN_*`` env vars are mirrored onto the jeanplot-side
``JEANPLOT_KNN_*`` names before the canonical module is loaded.
"""

import os

# Propagate biocomp's env names to jeanplot's before its KNN module imports.
for _bc, _jp in [
    ("BIOCOMP_KNN_BACKEND", "JEANPLOT_KNN_BACKEND"),
    ("BIOCOMP_KNN_WORKERS", "JEANPLOT_KNN_WORKERS"),
    ("BIOCOMP_KNN_MEAN_CHUNK_SIZE", "JEANPLOT_KNN_MEAN_CHUNK_SIZE"),
    ("BIOCOMP_KNN_ANN_M", "JEANPLOT_KNN_ANN_M"),
    ("BIOCOMP_KNN_ANN_EF_CONSTRUCTION", "JEANPLOT_KNN_ANN_EF_CONSTRUCTION"),
    ("BIOCOMP_KNN_ANN_EF_SEARCH", "JEANPLOT_KNN_ANN_EF_SEARCH"),
]:
    if _bc in os.environ and _jp not in os.environ:
        os.environ[_jp] = os.environ[_bc]

from jeanplot.knn.density import (  # noqa: E402, F401
    knn_density,
    knn_density_chunked,
)
from jeanplot.knn.gaussian import (  # noqa: E402, F401
    _knn_mean_from_indices_weights,
    get_gaussian_weighted_knn,
    get_knn_mean_and_variance,
    get_knn_mean_only,
)
from jeanplot.knn.tree import (  # noqa: E402, F401
    KNN_ANN_EF_CONSTRUCTION,
    KNN_ANN_EF_SEARCH,
    KNN_ANN_M,
    KNN_BACKEND,
    KNN_MEAN_CHUNK_SIZE,
    KNN_WORKERS,
    _env_int,
    _query,
    _resolve_backend,
    _resolve_threads,
    make_tree,
)

__all__ = [
    "KNN_ANN_EF_CONSTRUCTION",
    "KNN_ANN_EF_SEARCH",
    "KNN_ANN_M",
    "KNN_BACKEND",
    "KNN_MEAN_CHUNK_SIZE",
    "KNN_WORKERS",
    "_env_int",
    "_knn_mean_from_indices_weights",
    "_query",
    "_resolve_backend",
    "_resolve_threads",
    "get_gaussian_weighted_knn",
    "get_knn_mean_and_variance",
    "get_knn_mean_only",
    "knn_density",
    "knn_density_chunked",
    "make_tree",
]
