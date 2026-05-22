# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Shim layer over jeanplot's ASCII smooth-plot kernels."""

from jeanplot.plots.txt import (  # noqa: F401
    TextPlotResult,
    _knn_grid,
    _make_xy_grid,
    smooth_1d_txt,
    smooth_2d_txt,
    smooth_3d_txt,
)

# Biology-domain dispatch map: maps biocomp drawing-function paths to their
# ASCII counterparts. Lives in biocomp because the source paths are
# biocomp-specific.
TXT_PLOT_FUNCTION_MAP = {
    "biocomp.plotting.plotting_smooth.smooth_1d": smooth_1d_txt,
    "biocomp.plotting.plotting_smooth.smooth_2d": smooth_2d_txt,
    "biocomp.plotting.plotting_3d.smooth_3d": smooth_3d_txt,
    "biocomp.plotutils.smooth": None,
}


def get_txt_plot_function(original_func_name: str):
    if original_func_name in TXT_PLOT_FUNCTION_MAP:
        return TXT_PLOT_FUNCTION_MAP[original_func_name]
    for key, func in TXT_PLOT_FUNCTION_MAP.items():
        if original_func_name.endswith(key.split(".")[-1]):
            return func
    return None


__all__ = [
    "TXT_PLOT_FUNCTION_MAP",
    "TextPlotResult",
    "_knn_grid",
    "_make_xy_grid",
    "get_txt_plot_function",
    "smooth_1d_txt",
    "smooth_2d_txt",
    "smooth_3d_txt",
]
