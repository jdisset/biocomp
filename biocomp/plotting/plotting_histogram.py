# {{{                          --     imports     --
# ···············································································
import jax
import jax.numpy as jnp
from matplotlib import scale as mscale
from functools import partial
from scipy.spatial import cKDTree
from jax import jit, vmap
import numpy as np
from biocomp import utils as ut
from biocomp import datautils as du
from biocomp import compute as cmp
from biocomp.datautils import DataManager
import matplotlib.pyplot as plt
from jax.scipy.stats import gaussian_kde
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import difflib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import string
from labellines import labelLine, labelLines
from jax.typing import ArrayLike
from typing import Tuple
import os
from typing import Union, Sequence, List, Tuple, Dict, Any, Optional, Callable
from matplotlib.ticker import ScalarFormatter, NullFormatter, MaxNLocator
from matplotlib import colors as mcolors
from pkg_resources import resource_filename
from . import plotting_core as pc
from .plotting_core import (
    DEFAULT_CMAP_NAME,
    setup_transformed_axis,
    get_reordered_protein_names,
    network_ticks_and_labels,
    make_xy_grid,
    knn_avg,
    get_knn_quantile,
    format_powers,
    apply_style,
    heatmap,
)

NdArray = Union[np.ndarray, jnp.ndarray]
configurable = pc.configurable
##────────────────────────────────────────────────────────────────────────────}}}
# ---- density histograms

### {{{                       --     density histogram     --


@configurable
def histogram(
    X: NdArray,
    Y: NdArray,
    input_names: Sequence[str],
    output_name: str,
    rescaler: Callable,
    ax,
    nbins=(256, 256),
    xlims=(0, 1),
    ylims=(0, 1),
    vlims=(0.001, None),
    cmap=DEFAULT_CMAP_NAME,
    noise_smooth=0,
    use_log_density=True,
    draw_colorbar=False,
):
    assert X.shape[1] == 1

    if isinstance(nbins, int):
        nbins = [nbins, nbins]

    assert X.shape[1] == 1
    assert Y.shape[1] == 1

    xres = np.abs(np.subtract(*xlims)) / nbins[0]
    yres = np.abs(np.subtract(*ylims)) / nbins[1]

    X = X + np.random.normal(size=X.shape) * noise_smooth * xres
    Y = Y + np.random.normal(size=Y.shape) * noise_smooth * yres

    h, xedges, yedges = np.histogram2d(
        X,
        Y,
        bins=nbins,
        density=False,
        range=[xlims, ylims],
    )

    if use_log_density:
        h = np.log(h + 1)

    setup_transformed_axis(
        ax,
        xaxis_lims=xlims,
        yaxis_lims=ylims,
        rescaler=rescaler,
        margins=0.0,
    )

    im = ax.imshow(
        h.T,  # matplotlib wants it transposed
        extent=[*xlims, *ylims],
        origin='lower',
        aspect='auto',
        cmap=cmap,
        vmin=vlims[0],
        vmax=vlims[1],
    )

    ax.set_xlabel(input_names[0])
    ax.set_ylabel(output_name)

    # show grid, including minor grid
    ax.grid(color='k', alpha=0.25, linestyle='-', linewidth=0.2, which='major')
    ax.grid(color='k', alpha=0.1, linestyle='-', linewidth=0.1, which='minor')

    if draw_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        apply_style(cbar.ax)
        clabel = 'log(density)' if use_log_density else 'density'
        cbar.set_label(clabel, fontsize=8)
        for spine in cbar.ax.spines.values():
            spine.set_linewidth(0.2)


##────────────────────────────────────────────────────────────────────────────}}}
