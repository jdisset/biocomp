# {{{                          --     imports     --
# ···············································································
from dataclasses import dataclass, asdict, field
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
import matplotlib as mpl

from matplotlib.axes import Axes
from matplotlib.figure import Figure

# import plotting_core, plotting_smooth and plotting_histogram, all placed under the plotting/ directory:
from .plotting.plotting_core import (
    configurable,
    mkfig,
    extract_plot_data_from_network,
    PlotData,
)

# from .plotting.plotting_histogram import (
# histogram,
# )
# from .plotting.plotting_scatter import (
# scatter,
# )


NdArray = Union[np.ndarray, jnp.ndarray]

from .plotting import plotting_core as pc


logger = ut.setup_logger('biocomp.plotutils')
configurable = ut.configurable_decorator('biocomp.plotutils')

##────────────────────────────────────────────────────────────────────────────}}}


@dataclass
class FigureConfig:
    axes_size: Tuple[int, int] = (4, 4)  # individual axes size
    dpi: float = 300
    # TODO: add support for automatic layout (finding things like zslices for example)
    layout: Optional[Tuple[int, int]] = (1, 1)  # row, column
    title: Optional[str] = None  # can use variables from metadata


@dataclass
class DataDimensions:
    input_dim: int
    output_dim: int


# ---- network/recipe plots
### {{{                --     new network plot functions     --


def get_data_dimensions(plot_data: PlotData) -> DataDimensions:
    assert (
        len(plot_data.input_names) == plot_data.x.shape[1]
    ), f'{plot_data.input_names=}, {plot_data.x.shape=}'
    assert plot_data.y.shape == (
        plot_data.x.shape[0],
        1,
    ), f'{plot_data.y.shape=}, {plot_data.x.shape=}'
    return DataDimensions(len(plot_data.input_names), 1)


DEFAULT_PLOT_METHOD_PREFERENCE: Dict[int, str] = {
    1: 'smooth',
    2: 'smooth',
    3: 'smooth',
}


@configurable
def auto_plot(
    plot_data: PlotData,
    figure_config: Optional[FigureConfig] = None,
    ax: Optional[Union[Axes, Sequence[Axes]]] = None,
    use_plot_method: Optional[str] = 'auto',  # could be 'auto' or None, or a specific method
    plot_method_per_input_dim: Optional[Dict[int, str]] = DEFAULT_PLOT_METHOD_PREFERENCE,
    rc_context: Dict[str, Any] = {}, #pc.DEFAULT_RC_PARAMS,
    smooth_params: Dict[str, Any] = {},
    scatter_params: Dict[str, Any] = {},
    histogram_params: Dict[str, Any] = {},
    **kw,
) -> None:

    dim = get_data_dimensions(plot_data)
    if use_plot_method is None or use_plot_method == 'auto':
        assert plot_method_per_input_dim is not None
        use_plot_method = plot_method_per_input_dim.get(dim.input_dim, 'smooth')

    VALID_METHODS = ['smooth', 'scatter', 'histogram']
    if use_plot_method not in VALID_METHODS:
        raise ValueError(f'Unknown plotting method {use_plot_method}. Available: {VALID_METHODS}')

    if figure_config is None:
        figure_config = FigureConfig()

    with mpl.rc_context(rc_context):

        # first we check that we have axes to plot on
        # if not, we need to make a new figure
        assert dim.output_dim == 1, 'Only single output plots are supported'

        if ax is None:
            assert (
                figure_config.layout is not None
            ), 'Layout must be specified if axes are not provided'
            cols, rows = figure_config.layout
            print(f'Creating new figure with {cols}x{rows} axes')
            fig, ax = plt.subplots(
                *figure_config.layout,
                figsize=(cols * figure_config.axes_size[0], rows * figure_config.axes_size[1]),
                dpi=figure_config.dpi,
            )
        else:
            if not isinstance(ax, (list, tuple)):
                assert isinstance(ax, Axes), f'ax type is {type(ax)}'
                ax = [ax]
            fig = ax[0].get_figure()

        assert isinstance(fig, Figure)
        if figure_config.title is not None:
            fig.suptitle(figure_config.title)

        if use_plot_method == 'smooth':
            return smooth(plot_data, ax, **smooth_params, **kw)

        else:
            raise NotImplementedError(f'Unimplemented plotting method {method}')


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                          --     main smooth dispatcher (route to 1D, 2D, 3D)    --
from .plotting.plotting_3d import smooth_3d
from .plotting.plotting_smooth import smooth_2d


@configurable
def smooth(
    plot_data: PlotData, ax, smooth_1d_params={}, smooth_2d_params={}, smooth_3d_params={}, **kw
):
    dim = get_data_dimensions(plot_data)
    match (dim.input_dim, dim.output_dim):
        case (2, 1):
            return smooth_2d(
                X=plot_data.x,
                Y=plot_data.y,
                input_names=plot_data.input_names,
                output_name=plot_data.output_name,
                rescaler=plot_data.rescaler,
                ax=ax,
                **smooth_2d_params,
                **kw,
            )
        case (3, 1):
            return smooth_3d(
                X=plot_data.x,
                Y=plot_data.y,
                input_names=plot_data.input_names,
                output_name=plot_data.output_name,
                rescaler=plot_data.rescaler,
                ax=ax,
                **smooth_3d_params,
                **kw,
            )
        case _:
            raise ValueError(
                f'Plotting {dim.input_dim} inputs and {dim.output_dim} outputs is not supported'
            )


##────────────────────────────────────────────────────────────────────────────}}}
