# {{{                          --     imports     --
# ···············································································
from dataclasses import dataclass, field
import jax.numpy as jnp
from matplotlib import scale as mscale
from jax.tree_util import Partial
from functools import partial
import numpy as np
from . import utils as ut
from . import datautils as du
from .datautils import DataRescaler
import matplotlib.pyplot as plt
from biocomp.network import Network
import matplotlib.ticker as ticker
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.offline as pyo
import plotly.graph_objects as go
import plotly.graph_objs as go
import plotly.graph_objects as go
import plotly.offline as pyo
import numpy as np
import string
import os
from typing import (
    Union,
    Self,
    Sequence,
    List,
    Tuple,
    Dict,
    Any,
    Optional,
    Callable,
    TypeVar,
    Tuple,
    TypeAlias,
)
from matplotlib.ticker import ScalarFormatter, NullFormatter, MaxNLocator
import matplotlib as mpl

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import BaseModel, ValidationError, Field, field_validator, model_validator

from .plotting import plotting_core as pc

logger = ut.setup_logger('biocomp.plotting')
configurable = ut.configurable_decorator('biocomp.plotting')
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'

##────────────────────────────────────────────────────────────────────────────}}}


# ---- network/recipe plots

## {{{                      --     plot data class     --

T = TypeVar('T')
Pair = Tuple[T, T]
NdArray: TypeAlias = Union[np.ndarray, jnp.ndarray]
NumLike: TypeAlias = Union[np.ndarray, jnp.ndarray, float, int]


class DataDimensions(BaseModel):
    input: int
    output: int


class PlotData(BaseModel):

    x: NdArray
    y: NdArray

    input_names: Optional[List[str]]
    output_name: Optional[str]

    rescaler: Optional[DataRescaler] = None

    @property
    def dimensions(self) -> DataDimensions:
        return DataDimensions(input=self.x.shape[1], output=1)


    @model_validator(mode='after')
    def check_shapes(self) -> Self:

        if self.x.ndim == 1:
            self.x = self.x.reshape(-1, 1)

        if self.y.ndim == 1:
            self.y = self.y.reshape(-1, 1)

        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError('X and Y must have the same number of samples')

        if self.y.shape[1] != 1:
            raise ValueError('Y must be a 1D array')

        return self

    class Config:
        arbitrary_types_allowed = True


@dataclass(kw_only=True)
class FigureLayout:
    pass


@dataclass(kw_only=True)
class SimpleFigureLayout(FigureLayout):
    cols: int = 1
    rows: int = 1
    axes_size: Pair[int] = (5, 5)
    dpi: int = 300

    def make_figure(self) -> Tuple[Figure, Axes]:
        fig, ax = plt.subplots(
            self.rows,
            self.cols,
            figsize=(self.cols * self.axes_size[0], self.rows * self.axes_size[1]),
            dpi=self.dpi,
        )
        return fig, ax


@dataclass(kw_only=True)
class FigureSpec:
    title: Optional[str] = None
    output_dir: str = './'
    output_file: str = 'unnamed.png'
    extra_info: Optional[Dict[str, Any]] = None
    layout: FigureLayout = field(default_factory=SimpleFigureLayout)


##────────────────────────────────────────────────────────────────────────────}}}


## {{{                   --     DataRescaler wrapper     --
def get_rescaler(rescaler, **kw):
    if isinstance(rescaler, DataRescaler):
        return rescaler
    if isinstance(rescaler, (tuple, list)):
        assert len(rescaler) == 2, 'Rescaler must be a tuple of (fwd, inv) functions'
        assert callable(rescaler[0]) and callable(
            rescaler[1]
        ), 'Rescaler must be a tuple of (fwd, inv) functions'
        return DataRescaler(partial(rescaler[0], **kw), partial(rescaler[1], **kw))


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                       --     network utils     --


def extract_plot_data_from_network(
    network: Network,
    X: NdArray,
    Y: NdArray,
    rescaler: DataRescaler,
    input_order: Optional[Sequence[int]] = None,
    protein_aliases: Optional[Dict[str, str]] = None,
    use_y_as_x: bool = False,
) -> PlotData:

    if input_order is None:
        input_order = np.arange(network.get_nb_inputs())
    if protein_aliases is None:
        protein_aliases = {}

    protein_order, protein_names = get_reordered_protein_names(
        network, input_order, protein_aliases
    )

    input_order, output_pos = protein_order[:-1], protein_order[-1]
    input_names, output_name = protein_names[:-1], protein_names[-1]

    if use_y_as_x:
        output_names = network.get_output_proteins()
        xind = [output_names.index(i) for i in input_names]
        x = Y[:, xind]
    else:
        x = X[:, input_order]

    y = Y[:, output_pos]
    y = Y.reshape(-1, 1)

    assert x.shape[1] == len(input_order)
    assert y.shape[0] == X.shape[0]

    return PlotData(
        x=x,
        y=y,
        input_names=input_names,
        output_name=output_name,
        rescaler=rescaler,
    )


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                        --     misc utils     --
def make_xy_grid(xmin, xmax, ymin=None, ymax=None, xres=100, yres=None):
    ymin = ymin if ymin is not None else xmin
    ymax = ymax if ymax is not None else xmax
    yres = yres if yres is not None else xres
    xx = np.linspace(xmin, xmax, xres)
    yy = np.linspace(ymin, ymax, yres)
    X, Y = np.meshgrid(xx, yy)
    # we want to return as a big array of shape (res**2, 2)
    return np.vstack([X.ravel(), Y.ravel()]).T


def get_web_font(url, font_name):
    import tempfile
    from pathlib import Path
    import urllib
    from rich import print
    from matplotlib import font_manager

    # Create a temporary directory for the font file
    path = Path(tempfile.mkdtemp())

    # URL and downloaded path of the font
    url_font = url
    path_font = path / f"{font_name}.ttf"

    # Download the font to our temporary directory
    urllib.request.urlretrieve(url_font, path_font)
    # Create a Matplotlib Font object from our `.ttf` file
    font = font_manager.FontEntry(fname=str(path_font), name=font_name)

    # Register this object with Matplotlib's ttf list
    font_manager.fontManager.ttflist.append(font)
    return font


def to_display_units(x, ax):
    """Convert x from data units to display units"""
    ppd = 72.0 / ax.figure.dpi
    trans = ax.transData.transform
    return ((trans((1, x)) - trans((0, 0))) * ppd)[1]


def to_data_units(y_display, ax):
    """Convert y from display units to data units"""
    ppd = 72.0 / ax.figure.dpi
    trans_inv = ax.transData.inverted().transform
    origin = trans_inv((0, 0))
    point_in_data_units = trans_inv((0, y_display / ppd))
    return point_in_data_units[1] - origin[1]


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                    --     misc plot styling tools     --

DEFAULT_GREY = '#777777'

DEFAULT_RC_PARAMS = {
    'figure.facecolor': 'white',
    'font.family': 'sans-serif',
    'font.style': 'normal',
    'font.variant': 'normal',
    'font.weight': 'normal',
    'font.stretch': 'normal',
    'font.size': 10,
    'font.sans-serif': 'Roboto, DejaVu Sans, Bitstream Vera Sans, Computer Modern Sans Serif, Lucida Grande, Verdana, Geneva, Lucid, Arial, Helvetica, Avant Garde, sans-serif',
    'font.monospace': 'Roboto Mono, DejaVu Sans Mono, Bitstream Vera Sans Mono, Computer Modern Typewriter, Andale Mono, Nimbus Mono L, Courier New, Courier, Fixed, Terminal, monospace',
    'text.usetex': 'True',
    'text.latex.preamble': '\\usepackage{cmbright}',
    'mathtext.fontset': 'custom',
    'mathtext.bf': 'sans:bold',
    'mathtext.bfit': 'sans:italic:bold',
    'mathtext.cal': 'cursive',
    'mathtext.it': 'sans:italic',
    'mathtext.rm': 'sans',
    'mathtext.sf': 'sans',
    'mathtext.tt': 'monospace',
    'mathtext.fallback': 'stixsans',
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.labelsize': 10,
    'axes.labelweight': 'normal',
    'axes.labelcolor': DEFAULT_GREY,
    'axes.titlesize': 12,
    'axes.titleweight': 'normal',
    'axes.titlecolor': DEFAULT_GREY,
    'xtick.bottom': True,
    'xtick.labelbottom': True,
    'xtick.top': False,
    'xtick.labeltop': False,
    'xtick.major.size': 5,
    'xtick.major.width': 0.4,
    'xtick.minor.size': 2,
    'xtick.minor.width': 0.2,
    'ytick.left': True,
    'ytick.labelleft': True,
    'ytick.right': False,
    'ytick.labelright': False,
    'ytick.major.size': 5,
    'ytick.major.width': 0.4,
    'ytick.minor.size': 2,
    'ytick.minor.width': 0.2,
}


@dataclass
class SpineProps:
    visible: bool = True
    linewidth: float = 0.5
    color: str = DEFAULT_GREY


DEFAULT_SPINE_PROPS: Dict[str, SpineProps] = {
    'top': SpineProps(visible=False),
    'right': SpineProps(visible=False),
    'bottom': SpineProps(visible=True),
    'left': SpineProps(visible=True),
}

DEFAULT_TICK_PARAMS: List[Dict[str, Any]] = [
    {'axis': 'both', 'which': 'both', 'labelsize': 8, 'direction': 'out'},
    {'axis': 'both', 'which': 'major', 'length': 5, 'width': 0.4},
    {'axis': 'both', 'which': 'minor', 'length': 2, 'width': 0.2},
]


@dataclass
class FontProps:
    family: Optional[str] = 'Arial'
    size: Optional[int] = 10
    weight: Optional[str] = 'normal'
    style: Optional[str] = 'normal'
    color: Optional[str] = 'black'


@dataclass
class PlotStyle:
    facecolor: Optional[str] = 'white'
    spine_props: Optional[Dict[str, SpineProps]] = field(
        default_factory=lambda: DEFAULT_SPINE_PROPS
    )
    tick_params: Optional[List[Dict[str, Any]]] = field(default_factory=lambda: DEFAULT_TICK_PARAMS)
    # labelsize: Optional[int] = 10
    label_font: Optional[FontProps] = field(default_factory=FontProps)
    title_font: Optional[FontProps] = field(default_factory=FontProps)


DEFAULT_STYLE = PlotStyle()


def apply_style(ax, style: PlotStyle = DEFAULT_STYLE):
    fig = ax.get_figure()

    # if style.facecolor is not None:
    # fig.patch.set_facecolor(style.facecolor)
    # if style.spine_props is not None:
    # for spine, props in style.spine_props.items():
    # ax.spines[spine].set_visible(props.visible)
    # ax.spines[spine].set_linewidth(props.linewidth)
    # ax.spines[spine].set_color(props.color)
    # if style.tick_params is not None:
    # for tp in style.tick_params:
    # ax.tick_params(**tp)

    # if style.label_font is not None:
    # ax.xaxis.label.set_fontproperties(style.label_font)
    # ax.yaxis.label.set_fontproperties(style.label_font)

    # if style.title_font is not None:
    # ax.title.set_fontproperties(style.title_font)

    # ax.get_xaxis().tick_bottom()
    # ax.get_yaxis().tick_left()


@configurable
def mkfig(
    rows: int = 1,
    cols: int = 1,
    size: Tuple[float, float] = (4, 4),
    dpi: float = 300,
    title: Optional[str] = None,
    style: PlotStyle = DEFAULT_STYLE,
):
    fig, ax = plt.subplots(rows, cols, figsize=(cols * size[0], rows * size[1]), dpi=dpi, **kw)

    if rows == 1 and cols == 1:
        apply_style(ax)
    else:
        for a in ax.flatten():
            apply_style(a)

    if title is not None:
        fig.suptitle(str(title))

    return fig, ax


def remove_spines(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)


def remove_axis_and_spines(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    remove_spines(ax)


class ShortScientificFormatter(string.Formatter):
    def format_field(self, value, format_spec):
        if format_spec == 'm':
            if value < 1000:
                if value == int(value):
                    return super().format_field(int(value), '')
                else:
                    return super().format_field(value, '.1f')
            else:
                if value == int(value):
                    return super().format_field(value, '.0e').replace('e+0', 'e').replace('e+', 'e')
                else:
                    return super().format_field(value, '.1e').replace('e+0', 'e').replace('e+', 'e')
        else:
            return super().format_field(value, format_spec)


scformat = ShortScientificFormatter()


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                --     new network plot functions     --


DEFAULT_PLOT_METHOD_PREFERENCE: Dict[int, str] = {
    1: 'smooth',
    2: 'smooth',
    3: 'smooth',
}


@configurable
def auto_plot(
    plot_data: PlotData,
    figure_spec: Optional[FigureSpec] = None,
    ax: Optional[Union[Axes, Sequence[Axes]]] = None,
    use_plot_method: Optional[str] = 'auto',  # could be 'auto' or None, or a specific method
    plot_method_per_input_dim: Optional[Dict[int, str]] = DEFAULT_PLOT_METHOD_PREFERENCE,
    rc_context: Dict[str, Any] = {},  # pc.DEFAULT_RC_PARAMS,
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

    if figure_spec is None:
        figure_spec = FigureSpec()

    with mpl.rc_context(rc_context):

        # first we check that we have axes to plot on
        # if not, we need to make a new figure
        assert dim.output_dim == 1, 'Only single output plots are supported'

        if ax is None:
            assert (
                figure_spec.layout is not None
            ), 'Layout must be specified if axes are not provided'
            cols, rows = figure_spec.layout
            print(f'Creating new figure with {cols}x{rows} axes')
            fig, ax = plt.subplots(
                *figure_spec.layout,
                figsize=(cols * figure_spec.axes_size[0], rows * figure_spec.axes_size[1]),
                dpi=figure_spec.dpi,
            )
        else:
            if not isinstance(ax, (list, tuple)):
                assert isinstance(ax, Axes), f'ax type is {type(ax)}'
                ax = [ax]
            fig = ax[0].get_figure()

        assert isinstance(fig, Figure)
        if figure_spec.title is not None:
            fig.suptitle(figure_spec.title)

        if use_plot_method == 'smooth':
            return smooth(plot_data, ax, **smooth_params, **kw)

        else:
            raise NotImplementedError(f'Unimplemented plotting method {method}')


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                          --     main smooth dispatcher (route to 1D, 2D, 3D)    --

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
