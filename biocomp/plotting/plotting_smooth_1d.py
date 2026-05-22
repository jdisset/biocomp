# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Shim layer over jeanplot's 1D smooth-curve kernel."""

from jeanplot.plots.smooth_1d import (  # noqa: F401
    DEFAULT_MARKER_ROTATION,
    _annotate_theta,
    _draw_tail_fits,
    _linear_fit_overlay,
    make_n_props,
    smooth_1d,
)

# Force biocomp default colormap registration at import time.
from . import plotting_core as _pc  # noqa: F401

__all__ = [
    "DEFAULT_MARKER_ROTATION",
    "_annotate_theta",
    "_draw_tail_fits",
    "_linear_fit_overlay",
    "make_n_props",
    "smooth_1d",
]
