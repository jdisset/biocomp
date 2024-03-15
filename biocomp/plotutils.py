# {{{                          --     imports     --
# ···············································································
import jax
import jax.numpy as jnp
from matplotlib import scale as mscale
from jax.tree_util import Partial as partial
from scipy.spatial import cKDTree
from jax import jit, vmap
import numpy as np
from . import utils as ut
from . import datautils as du
from . import compute as cmp
from . import parameters as pm
from .datautils import DataManager
import matplotlib.pyplot as plt
from jax.scipy.stats import gaussian_kde
import matplotlib.ticker as ticker
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.offline as pyo
import plotly.graph_objects as go
import plotly.graph_objs as go
import plotly.graph_objects as go
import plotly.offline as pyo
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


# import plotting_core, plotting_smooth and plotting_histogram, all placed under the plotting/ directory:
from .plotting.plotting_core import (
    configurable,
    mkfig,
    extract_plot_data_from_network,
)

from .plotting.plotting_histogram import (
    histogram,
)

from .plotting.plotting_scatter import (
    scatter,
)


ndArray = Union[np.ndarray, jnp.ndarray]

from .plotting import plotting_core as pc


logger = ut.setup_logger('biocomp.plotutils')
configurable = ut.configurable_decorator('biocomp.plotutils')

##────────────────────────────────────────────────────────────────────────────}}}


# ---- network/recipe plots
### {{{                --     new network plot functions     --


@configurable
def plot_network(
    network,
    x,
    y,
    rescaler,
    ax=None,
    kde=None,
    density_quantile_threshold=0.05,
    method='smooth',
    use_y_as_x=False,
    input_order=None,
    protein_aliases=None,
    smooth_params={},
    scatter_params={},
    histogram_params={},
    **kw
):
    """plot a network on a given axis (or list of axes)"""

    if kde is not False and kde is not None:
        rng = jax.random.PRNGKey(0)
        subsample = du.optimal_density_subsample(
            x, kde, rng, quantile_threshold=density_quantile_threshold
        )
        x, y = x[subsample], y[subsample]

    x, y, input_names, output_name = extract_plot_data_from_network(
        network, x, y, input_order, protein_aliases, use_y_as_x
    )

    if method == 'smooth':
        return smooth(x, y, input_names, output_name, rescaler, ax=ax, **smooth_params, **kw)
    elif method == 'scatter':
        return scatter(x, y, input_names, output_name, rescaler, ax=ax, **scatter_params, **kw)
    elif method == 'histogram':
        return histogram(x, y, input_names, output_name, rescaler, ax=ax, **histogram_params, **kw)
    else:
        raise NotImplementedError(f'Unknown plotting method {method}')


# network_figure_*d exist to allow for a different configuration path
# per number of dimensions (1D, 2D, 3D)


@configurable
def network_figure_1d(network, x, y, rescaler, mkfig_params={}, plot_network_params={}):
    assert network.get_nb_inputs() == 1
    fig, ax = mkfig(1, 1, **mkfig_params)
    plot_network(network, x, y, rescaler, ax=ax, **plot_network_params)
    return fig


@configurable
def network_figure_2d(network, x, y, rescaler, mkfig_params={}, plot_network_params={}):
    assert network.get_nb_inputs() == 2
    fig, ax = mkfig(1, 1, **mkfig_params)
    plot_network(network, x, y, rescaler, ax=ax, **plot_network_params)
    return fig


@configurable
def network_figure_3d(network, x, y, rescaler, zslices=(0,), mkfig_params={}, **plot_network_params):
    assert network.get_nb_inputs() == 3
    nslices = len(zslices)
    fig, axes = mkfig(1, nslices, **mkfig_params)
    if nslices == 1:
        axes = np.array([axes])
    plot_network(network, x, y, rescaler, ax=axes, zslices = zslices, **plot_network_params)
    return fig


# network_figure is the main entry point for plotting networks


@configurable
def network_figure(
    network,
    x,
    y,
    rescaler,
    network_figure_1d_params={},
    network_figure_2d_params={},
    network_figure_3d_params={},
):
    n_inputs = network.get_nb_inputs()
    if n_inputs == 1:
        return network_figure_1d(network, x, y, rescaler, **network_figure_1d_params)
    elif n_inputs == 2:
        return network_figure_2d(network, x, y, rescaler, **network_figure_2d_params)
    elif n_inputs == 3:
        return network_figure_3d(network, x, y, rescaler, **network_figure_3d_params)
    else:
        raise ValueError(f'Network with {n_inputs} inputs is not supported')


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                          --     main smooth dispatcher (route to 1D, 2D, 3D)    --
from .plotting.plotting_3d import smooth_3d
from .plotting.plotting_smooth import smooth_2d

@configurable
def smooth(X, Y, input_names, output_name, rescaler, smooth_2d_params={}, smooth_3d_params={}, **kw):
    ninputs = X.shape[1]
    if ninputs == 2:
        smooth_2d(X, Y, input_names, output_name, rescaler, **smooth_2d_params, **kw)
    elif ninputs == 3:
        smooth_3d(X, Y, input_names, output_name, rescaler, **smooth_3d_params, **kw)
    else:
        raise NotImplementedError(f'Cannot plot {ninputs} inputs in smooth mode')


##────────────────────────────────────────────────────────────────────────────}}}
