# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Shim layer over jeanplot's JAX-backed KNN kernels."""

from jeanplot.knn.jax_kernel import (  # noqa: F401
    get_gaussian_weighted_knn,
    get_knn_mean_and_variance,
    get_knn_quantile,
    query_kdtree,
    weighted_quantile,
)

__all__ = [
    "get_gaussian_weighted_knn",
    "get_knn_mean_and_variance",
    "get_knn_quantile",
    "query_kdtree",
    "weighted_quantile",
]
