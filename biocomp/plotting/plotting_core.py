# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Biocomp shim over jeanplot's plotting kernels.

Generic plotting kernels (axis setup, heatmap, KNN, otsu, formatters) now live
in :mod:`jeanplot.plots` / :mod:`jeanplot.knn`. This module re-exports them under
their historical biocomp names and adds the biology-specific bits (cmap config,
protein-name reordering, network tick helpers).
"""

import difflib
import os
from os import getenv

import dracon as dr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

from biocomp import utils as ut
from biocomp.logging_config import get_logger
from jeanplot.data.rescaler import DataRescaler  # noqa: F401
from jeanplot.knn.density import (
    _ball_volume,  # noqa: F401
    per_point_knn_density,
    uniform_resampling,
)
from jeanplot.plots.heatmap import (
    _otsu_threshold,  # noqa: F401
    _resolve_symbolic_level,  # noqa: F401
    _smooth_otsu_threshold,  # noqa: F401
    heatmap,
)
from jeanplot.plots.smooth_kernel import (
    array_content_key,
    build_tree as _jp_build_tree,
    knn_stats,
    weighted_kde_1d,
)
from jeanplot.plots.ticks import (
    PowerFormatter,
    _install_overlap_skip,  # noqa: F401
    format_powers,
    get_transformed_ticks_and_labels,
    powers_of_ten,
    setup_symlog_axis as _jp_setup_symlog_axis,
    setup_symlog_xaxis,
    setup_symlog_yaxis,
    setup_transformed_axis,
    setup_transformed_axis_generic,
    setup_xaxis,
    setup_yaxis,
)


def build_tree(x, use_jax=False):
    """biocomp-compatible wrapper around ``jeanplot.plots.smooth_kernel.build_tree``.

    Legacy biocomp callers pass ``use_jax=False`` explicitly; jeanplot's tree
    builder always uses the numpy/usearch path. The kwarg is accepted and
    ignored.
    """
    if use_jax:
        import jaxkd as jk
        import jax

        return jax.jit(jk.build_tree)(x)
    return _jp_build_tree(x)


logger = get_logger(__name__)

configurable = ut.configurable_decorator("biocomp.plotting")

os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"

# Biology-specific colormap config ─────────────────────────────────────────────
BIOCOMP_COLORS = dr.resolve_all_lazy(dr.load("pkg:biocomp:config/biocomp_colors.yaml"))
cmap_definitions = BIOCOMP_COLORS["color_maps"] or {}

CUSTOM_CMAPS = {
    k: mcolors.LinearSegmentedColormap.from_list(k, v, N=256) for k, v in cmap_definitions.items()
}

for k, v in CUSTOM_CMAPS.items():
    if k in plt.colormaps():
        plt.colormaps.unregister(k)
    plt.colormaps.register(v, name=k)

DEFAULT_CMAP_NAME = BIOCOMP_COLORS["default_color_map"] or "viridis"

# Ensure heatmap() picks up biocomp's default cmap by patching at import time.
# jeanplot.plots.heatmap.heatmap defaults to "viridis"; biocomp callers expect
# the configured default. We wrap so the default still falls back correctly.
_jp_heatmap = heatmap


@configurable
def heatmap(  # type: ignore[no-redef]  # noqa: F811
    ax,
    xy_grid,
    output_values,
    *args,
    cmap=DEFAULT_CMAP_NAME,
    **kwargs,
):
    return _jp_heatmap(ax, xy_grid, output_values, *args, cmap=cmap, **kwargs)


USE_KNN_JAX = getenv("BC_KNN_USE_JAX", default=False)


# Biology-specific helpers ─────────────────────────────────────────────────────
def get_bio_color(name, default="k"):
    colors = {"ebfp": "#529edb", "eyfp": "#fbda73", "mkate": "#f75a5a", "neongreen": "#33f397"}
    colors["fitc"] = colors["neongreen"]
    colors["pe_texas_red"] = colors["mkate"]
    colors["pacific_blue"] = colors["ebfp"]
    closest = difflib.get_close_matches(name.lower(), colors.keys(), n=1)
    if not closest:
        return default
    return colors[closest[0]]


def get_reordered_protein_names(
    network, input_order=None, protein_aliases=None, only_dependent_outputs=True, **_
):
    """Resolve a column convention for X (and dependent-output positions for Y).

    See historical biocomp docstring for full semantics. Returns
    ``(in_order, output_pos, reordered_input_names, output_name)``.
    """
    input_names = network.get_inverted_input_proteins()
    output_names = network.get_output_proteins(only_dependent_outputs=only_dependent_outputs)

    lower_input_names = [n.lower() for n in input_names]
    lower_protein_aliases = (
        {k.lower(): v for k, v in protein_aliases.items()} if protein_aliases else {}
    )

    if input_order is None:
        in_order = list(range(len(input_names)))
        reordered_input_names = list(input_names)
    elif input_order == "inv":
        in_order = list(range(len(input_names) - 1, -1, -1))
        reordered_input_names = [input_names[i] for i in in_order]
    else:
        old_order = list(input_order)
        resolved: list = []
        if any(isinstance(i, str) for i in old_order):
            for iname in old_order:
                if isinstance(iname, str):
                    if iname == "*":
                        resolved.append("*")
                    else:
                        iname_low = iname.lower()
                        if iname_low in lower_input_names:
                            resolved.append(lower_input_names.index(iname_low))
                        elif iname_low in lower_protein_aliases:
                            resolved.append(
                                lower_input_names.index(lower_protein_aliases[iname_low])
                            )
                        else:
                            raise ValueError(f"Invalid protein name: {iname}")
                else:
                    assert isinstance(iname, int | np.integer), f"Invalid protein index: {iname}"
                    assert iname in range(len(input_names)), f"Invalid protein index: {iname}"
                    resolved.append(int(iname))
        else:
            resolved = [int(i) for i in old_order]

        assert len(resolved) == len(input_names), (
            f"Wrong number of inputs: {resolved=}, {input_names=}"
        )

        if "*" in resolved:
            missing = set(range(len(input_names))) - set(resolved)
            resolved = [i if i != "*" else missing.pop() for i in resolved]

        in_order = resolved
        reordered_input_names = [input_names[i] for i in in_order]

    output_name = list(output_names)
    if len(output_name) > 1:
        logger.debug(f"multiple output proteins found: {output_name}")
    all_outputs = network.get_output_proteins(only_dependent_outputs=False)
    output_pos = [all_outputs.index(n) for n in output_name]

    if protein_aliases is not None:
        reordered_input_names = [protein_aliases.get(n, n) for n in reordered_input_names]
        output_name = [protein_aliases.get(n, n) for n in output_name]

    if len(output_pos) == 1:
        return in_order, output_pos[0], reordered_input_names, output_name[0]
    return in_order, output_pos, reordered_input_names, output_name


def network_ticks_and_labels(network, rescaler, xmin=0, xmax=1, **kw):
    from biocomp.plotutils import ShortScientificFormatter

    scformat = ShortScientificFormatter()
    unscaled_ticks = np.logspace(0, 12, 13)
    ticks = np.array(rescaler.fwd(unscaled_ticks))
    valid_ticks = (ticks <= xmax) & (ticks >= xmin)
    ticks = ticks[valid_ticks]
    tlabels = [scformat.format("{:m}", x) for x in unscaled_ticks[valid_ticks]]
    secondary_ticks = []
    rpnames = get_reordered_protein_names(network, **kw)
    return *rpnames, ticks, tlabels, secondary_ticks


def setup_symlog_axis(
    ax, xaxis_lims=None, yaxis_lims=None, linthresh=200, linscale=0.4, margins=0.05, **kw
):
    from functools import partial

    tr = partial(ut.log_poly_log, threshold=linthresh, compression=linscale)
    invtr = partial(ut.inverse_log_poly_log, threshold=linthresh, compression=linscale)
    return _jp_setup_symlog_axis(
        ax,
        xaxis_lims=xaxis_lims,
        yaxis_lims=yaxis_lims,
        transform=tr,
        inv_transform=invtr,
        margins=margins,
        **kw,
    )


# Wrap for @configurable
setup_xaxis = configurable(setup_xaxis)
setup_yaxis = configurable(setup_yaxis)
setup_transformed_axis = configurable(setup_transformed_axis)
knn_stats = configurable(knn_stats)


__all__ = [
    "BIOCOMP_COLORS",
    "CUSTOM_CMAPS",
    "DEFAULT_CMAP_NAME",
    "DataRescaler",
    "PowerFormatter",
    "USE_KNN_JAX",
    "_ball_volume",
    "_install_overlap_skip",
    "_otsu_threshold",
    "_resolve_symbolic_level",
    "_smooth_otsu_threshold",
    "array_content_key",
    "build_tree",
    "configurable",
    "format_powers",
    "get_bio_color",
    "get_reordered_protein_names",
    "get_transformed_ticks_and_labels",
    "heatmap",
    "knn_stats",
    "network_ticks_and_labels",
    "per_point_knn_density",
    "powers_of_ten",
    "setup_symlog_axis",
    "setup_symlog_xaxis",
    "setup_symlog_yaxis",
    "setup_transformed_axis",
    "setup_transformed_axis_generic",
    "setup_xaxis",
    "setup_yaxis",
    "uniform_resampling",
    "weighted_kde_1d",
]
