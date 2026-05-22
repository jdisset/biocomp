# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Biology-specific plotting helpers.

Generic kernels (heatmap, knn, ticks, KDE, density, ...) live in
:mod:`jeanplot.plots` / :mod:`jeanplot.knn`. This module hosts only what
depends on biocomp domain types: cmap registration, protein-name
reordering, network-aware tick helpers, and the biocomp log-poly-log
symlog axis.
"""

import difflib
import os
from functools import partial

import dracon as dr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

from biocomp import utils as ut
from biocomp.logging_config import get_logger
from jeanplot.plots.heatmap import heatmap as _jp_heatmap
from jeanplot.plots.ticks import setup_symlog_axis as _jp_setup_symlog_axis

logger = get_logger(__name__)
configurable = ut.configurable_decorator("biocomp.plotting")

os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"

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


@configurable
def heatmap(ax, xy_grid, output_values, *args, cmap=DEFAULT_CMAP_NAME, **kwargs):
    return _jp_heatmap(ax, xy_grid, output_values, *args, cmap=cmap, **kwargs)


_BIO_COLORS = {
    "ebfp": "#529edb",
    "eyfp": "#fbda73",
    "mkate": "#f75a5a",
    "neongreen": "#33f397",
    "fitc": "#33f397",
    "pe_texas_red": "#f75a5a",
    "pacific_blue": "#529edb",
}


def get_bio_color(name, default="k"):
    closest = difflib.get_close_matches(name.lower(), _BIO_COLORS.keys(), n=1)
    return _BIO_COLORS[closest[0]] if closest else default


def get_reordered_protein_names(
    network, input_order=None, protein_aliases=None, only_dependent_outputs=True, **_
):
    input_names = network.get_inverted_input_proteins()
    output_names = network.get_output_proteins(only_dependent_outputs=only_dependent_outputs)

    lower_input_names = [n.lower() for n in input_names]
    lower_protein_aliases = (
        {k.lower(): v for k, v in protein_aliases.items()} if protein_aliases else {}
    )

    if input_order is None:
        in_order = list(range(len(input_names)))
    elif input_order == "inv":
        in_order = list(range(len(input_names) - 1, -1, -1))
    else:
        resolved: list = []
        if any(isinstance(i, str) for i in input_order):
            for iname in input_order:
                if isinstance(iname, str):
                    if iname == "*":
                        resolved.append("*")
                        continue
                    low = iname.lower()
                    if low in lower_input_names:
                        resolved.append(lower_input_names.index(low))
                    elif low in lower_protein_aliases:
                        resolved.append(lower_input_names.index(lower_protein_aliases[low]))
                    else:
                        raise ValueError(f"Invalid protein name: {iname}")
                else:
                    assert isinstance(iname, int | np.integer), f"Invalid protein index: {iname}"
                    assert iname in range(len(input_names)), f"Invalid protein index: {iname}"
                    resolved.append(int(iname))
        else:
            resolved = [int(i) for i in input_order]

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
    unscaled = np.logspace(0, 12, 13)
    ticks = np.array(rescaler.fwd(unscaled))
    valid = (ticks <= xmax) & (ticks >= xmin)
    ticks = ticks[valid]
    tlabels = [scformat.format("{:m}", x) for x in unscaled[valid]]
    return *get_reordered_protein_names(network, **kw), ticks, tlabels, []


def setup_symlog_axis(
    ax, xaxis_lims=None, yaxis_lims=None, linthresh=200, linscale=0.4, margins=0.05, **kw
):
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
