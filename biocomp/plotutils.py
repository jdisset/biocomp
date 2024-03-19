# {{{                          --     imports     --
# ···············································································
from dataclasses import dataclass
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
    PlotData,
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


@dataclass
class FigureConfig:
    title: Optional[str] = None
    size: Optional[Tuple[int, int]] = None
    dpi: Optional[int] = 300


# ---- network/recipe plots
### {{{                --     new network plot functions     --

NdArray = Union[np.ndarray, jnp.ndarray]


def auto_plot(
    plot_data: PlotData,
    method: str = 'smooth',
    smooth_params: Dict[str, Any] = {},
    scatter_params: Dict[str, Any] = {},
    histogram_params: Dict[str, Any] = {},
    **kw,
) -> None:

    if method == 'smooth':
        return smooth(**smooth_params, **kw)
    elif method == 'scatter':
        return scatter(**scatter_params, **kw)
    elif method == 'histogram':
        return histogram(**histogram_params, **kw)
    else:
        raise NotImplementedError(f'Unknown plotting method {method}')


# network_figure_*d exist to allow for a different configuration path
# per number of dimensions (1D, 2D, 3D)


@configurable
def network_figure_1d(
    plot_data: PlotData,
    mkfig_params={},
    auto_plot_params={},
    **kw,
):
    fig, ax = mkfig(1, 1, **mkfig_params)
    auto_plot(
        plot_data,
        ax=ax,
        **auto_plot_params,
        **kw,
    )
    return fig


@configurable
def network_figure_2d(
    plot_data: PlotData,
    mkfig_params={},
    auto_plot_params={},
    **kw,
):
    fig, ax = mkfig(1, 1, **mkfig_params)
    auto_plot(
        plot_data,
        ax=ax,
        **auto_plot_params,
        **kw,
    )
    return fig


@configurable
def network_figure_3d(
    plot_data: PlotData,
    zslices=(0,),
    mkfig_params={},
    auto_plot_params={},
    **kw,
):
    nslices = len(zslices)
    fig, axes = mkfig(1, nslices, **mkfig_params)
    if nslices == 1:
        axes = np.array([axes])

    auto_plot(
        plot_data,
        ax=axes,
        **auto_plot_params,
        **kw,
    )

    return fig


# network_figure is the main entry point for plotting networks


@configurable
def network_figure(
    network,
    x,
    y,
    rescaler,
    input_order=None,
    protein_aliases: Dict[str, str] = {},
    use_y_as_x: bool = False,
    network_figure_1d_params={},
    network_figure_2d_params={},
    network_figure_3d_params={},
):
    n_inputs = network.get_nb_inputs()
    if input_order is None:
        input_order = list(range(n_inputs))

    if n_inputs == 1:
        f = network_figure_1d
        params = network_figure_1d_params
    elif n_inputs == 2:
        f = network_figure_2d
        params = network_figure_2d_params
    elif n_inputs == 3:
        f = network_figure_3d
        params = network_figure_3d_params
    else:
        raise ValueError(f'Network with {n_inputs} inputs is not supported')

    plot_data = extract_plot_data_from_network(
        network=network,
        x=x,
        y=y,
        rescaler=rescaler,
        input_order=input_order,
        protein_aliases=protein_aliases,
        use_y_as_x=use_y_as_x,
    )

    return f(plot_data, **params)


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                          --     main smooth dispatcher (route to 1D, 2D, 3D)    --
from .plotting.plotting_3d import smooth_3d
from .plotting.plotting_smooth import smooth_2d


@configurable
def smooth(
    x, y, input_names, output_name, rescaler, smooth_2d_params={}, smooth_3d_params={}, **kw
):
    ninputs = x.shape[1]
    if ninputs == 2:
        smooth_2d(x, y, input_names, output_name, rescaler, **smooth_2d_params, **kw)
    elif ninputs == 3:
        smooth_3d(x, y, input_names, output_name, rescaler, **smooth_3d_params, **kw)
    else:
        raise NotImplementedError(f'Cannot plot {ninputs} inputs in smooth mode')


##────────────────────────────────────────────────────────────────────────────}}}
