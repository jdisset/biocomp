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
    default_style,
    heatmap,
)

NdArray = Union[np.ndarray, jnp.ndarray]
configurable = pc.configurable
##────────────────────────────────────────────────────────────────────────────}}}

# ---- scatter plots
### {{{         --     main scatter method (route to 1D, 2D, 3D)     --
def scatter(x, y, network, *args, **kw):
    ninputs = network.get_nb_inputs()
    if ninputs == 1:
        return scatter_1d(x, y, network, *args, **kw)
    if ninputs == 2:
        return scatter_2d(x, y, network, *args, **kw)
    if ninputs == 3:
        return scatter_3d_interactive(x, y, network, *args, **kw)
    else:
        raise NotImplementedError(f'Cannot scater plot {ninputs} inputs')


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                            --     1D     --
def scatter_1d(
    x,
    y,
    network,
    rescaler,
    ax,
    xmin=0,
    xmax=1,
    title=None,
    max_n=20000,
    s=10,
    alpha=0.1,
    lw=0,
    key=jax.random.PRNGKey(0),
    use_y_as_x=False,
    **kw,
):
    protein_order, protein_names = get_reordered_protein_names(network, **kw)
    input_order, output_pos = protein_order[:-1], protein_order[-1]
    input_names, output_name = protein_names[:-1], protein_names[-1]
    random_order = np.random.permutation(min(max_n, len(x)))

    if use_y_as_x:
        other_pos = 1 - output_pos
        x = y[random_order, other_pos].squeeze()
    else:
        x = x[random_order].squeeze()

    y = y[random_order, output_pos]

    sc = ax.scatter(x, y, s=s, lw=lw, edgecolor='k', alpha=alpha, color='k')

    ax.set_xlabel(input_names[0])
    ax.set_ylabel(output_name)

    xlims = np.array([xmin, xmax])
    setup_transformed_axis(
        ax,
        xaxis_lims=xlims,
        yaxis_lims=xlims,
        rescaler=rescaler,
        margins=0.0,
        **kw,
    )

    ttle = None

    if title is True:
        ttle = f'{network.name}'
    elif title is not None:
        ttle = title
    if ttle is not None:
        ax.set_title(ttle)


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                            --     2D     --
def scatter_2d(
    x,
    y,
    network,
    rescaler,
    ax,
    xmin=0,
    xmax=1,
    title=None,
    key=jax.random.PRNGKey(0),
    size=10,
    colorbar=True,
    lw=0.1,
    cmap=DEFAULT_CMAP_NAME,
    xlims=None,
    ylims=None,
    **kw,
):
    protein_order, protein_names = get_reordered_protein_names(network, **kw)
    input_order, output_pos = protein_order[:-1], protein_order[-1]
    input_names, output_name = protein_names[:-1], protein_names[-1]

    random_order = jax.random.permutation(key, len(x))
    y = y[random_order, output_pos]
    x = x[random_order][:, input_order]

    setup_transformed_axis(
        ax,
        xaxis_lims=xlims,
        yaxis_lims=xlims,
        rescaler=rescaler,
        margins=0.0,
        **kw,
    )

    sc = ax.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap, s=size, lw=lw, edgecolor='k')

    ax.set_xlabel(input_names[0])
    ax.set_ylabel(input_names[1])

    ttle = None

    if title is True:
        ttle = f'{network.name}\n{output_name} smoothed mean'
    elif title is not None:
        ttle = title
    if ttle is not None:
        ax.set_title(ttle)


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                            --     3D     --
def scatter_3d_interactive(
    x,
    y,
    network,
    rescaler,
    xlims=(0, 1),
    title=None,
    key=jax.random.PRNGKey(0),
    size=10,
    colorbar=True,
    lw=0.01,
    filename=None,
    **kw,
):
    xmin, xmax = xlims

    protein_order, protein_names = get_reordered_protein_names(network, **kw)
    input_order, output_pos = protein_order[:-1], protein_order[-1]
    input_names, output_name = protein_names[:-1], protein_names[-1]

    random_order = jax.random.permutation(key, len(x))
    y = y[random_order, output_pos]
    x = x[random_order][:, input_order]

    fig = go.Figure()

    scatter = go.Scatter3d(
        x=x[:, 0],
        y=x[:, 1],
        z=x[:, 2],
        mode='markers',
        marker=dict(
            size=size, color=y, colorscale='YlGnBu', opacity=1, line=dict(color='black', width=lw)
        ),
    )

    fig.add_trace(scatter)

    fig.update_layout(
        scene=dict(
            xaxis_title=input_names[0],
            yaxis_title=input_names[1],
            zaxis_title=input_names[2],
            xaxis=dict(
                showspikes=False, showbackground=False
            ),  # tickvals=ticks, ticktext=ticklabels),
            yaxis=dict(
                showspikes=False, showbackground=False
            ),  # tickvals=ticks, ticktext=ticklabels),
            zaxis=dict(
                showspikes=False, showbackground=False
            ),  # tickvals=ticks, ticktext=ticklabels),
        ),
        width=1000,
        height=800,
    )

    if colorbar:
        cbar_trace = go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode='markers',
            marker=dict(
                size=0,
                cmin=y.min(),
                cmax=y.max(),
                colorscale='YlGnBu',
                showscale=True,
                colorbar=dict(title=output_name),  # tickvals=ticks, ticktext=ticklabels),
            ),
        )

        fig.add_trace(cbar_trace)

    ttle = None
    if title is True:
        ttle = f'{network.name}\n{output_name} smoothed mean'
    elif title is not None:
        ttle = title
    if ttle is not None:
        fig.update_layout(title=ttle)

    if filename is None:
        return pyo.plot(fig, auto_open=True)
    else:
        return pyo.plot(fig, filename=filename, auto_open=False)


def scatter_3d(
    x,
    y,
    network,
    rescaler,
    fig,
    n_views,
    xmin=0,
    xmax=1,
    title=None,
    key=jax.random.PRNGKey(0),
    size=10,
    colorbar=True,
    lw=0.1,
    **kw,
):
    (
        input_order,
        input_names,
        output_pos,
        output_name,
        ticks,
        ticklabels,
        secondticks,
    ) = network_ticks_and_labels(network, rescaler, xmax=xmax, **kw)

    cmap = plt.get_cmap('YlGnBu')
    random_order = jax.random.permutation(key, len(x))
    y = y[random_order, output_pos]
    x = x[random_order][:, input_order]

    azim_values = np.linspace(0, 270, n_views)

    for i, azim in enumerate(azim_values):
        ax = fig.add_subplot(1, n_views, i + 1, projection='3d')
        sc = ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y, cmap=cmap, s=size, lw=lw, edgecolor='k')
        ax.set_xlabel(input_names[0])
        ax.set_ylabel(input_names[1])
        ax.set_zlabel(input_names[2])

        if len(ticks) > 0:
            sc_ticks = ticks
            ax.set_xticks(sc_ticks)
            ax.set_xticklabels(ticklabels)
            ax.set_yticks(sc_ticks)
            ax.set_yticklabels(ticklabels)
            ax.set_zticks(sc_ticks)
            ax.set_zticklabels(ticklabels)


        ttle = None

        if title is True:
            ttle = f'{network.name}\n{output_name} smoothed mean'
        elif title is not None:
            ttle = title
        if ttle is not None:
            ax.set_title(ttle)

        # Rotate the axes
        ax.view_init(elev=10, azim=azim)


##────────────────────────────────────────────────────────────────────────────}}}


