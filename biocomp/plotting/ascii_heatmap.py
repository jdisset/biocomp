# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Shim layer over jeanplot's ASCII heatmap kernel."""

from jeanplot.plots.ascii_heatmap import (  # noqa: F401
    CMAP_B,
    CMAP_B_EXT,
    CMAP_S,
    GRAY,
    _color_seq,
    _resample,
    _resample_mean,
    _resample_nearest,
    format_axis_labels,
    format_title,
    heatmap,
    heatmap_bigram,
    heatmap_with_labels,
    imshow,
)

__all__ = [
    "CMAP_B",
    "CMAP_B_EXT",
    "CMAP_S",
    "GRAY",
    "_color_seq",
    "_resample",
    "_resample_mean",
    "_resample_nearest",
    "format_axis_labels",
    "format_title",
    "heatmap",
    "heatmap_bigram",
    "heatmap_with_labels",
    "imshow",
]
