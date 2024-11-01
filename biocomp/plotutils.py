# {{{                          --     imports     --
# ···············································································
from dataclasses import dataclass, field
import jax.numpy as jnp
from functools import partial
import numpy as np
from biocomp import utils as ut
from biocomp.datautils import DataRescaler
import matplotlib.pyplot as plt
from biocomp.network import Network
from biocomp.utils import ArbitraryModel, build_if_has_target
import string
import os
from typing import (
    Union,
    Self,
    Annotated,
    Sequence,
    List,
    Tuple,
    Dict,
    Any,
    Optional,
    Callable,
    TypeVar,
    TypeAlias,
)
import matplotlib as mpl

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import (
    BaseModel,
    Field,
    BeforeValidator,
)

from pathlib import Path
from biocomp.plotting import plotting_core as pc
from biocomp.logging_config import get_logger

logger = get_logger(__name__)

configurable = ut.configurable_decorator("biocomp.plotting")
os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"

##────────────────────────────────────────────────────────────────────────────}}}


# ---- network/recipe plots

## {{{                      --     plot data class     --

T = TypeVar("T")
Pair: TypeAlias = Tuple[T, T]
ListOrSingle: TypeAlias = Union[List[T], T]
NdArray: TypeAlias = Union[np.ndarray, jnp.ndarray]
NumLike: TypeAlias = Union[np.ndarray, jnp.ndarray, float, int]


class DataDimensions(BaseModel):
    input: int
    output: int


class PlotData(ArbitraryModel):
    xval: Optional[NdArray]
    yval: Optional[NdArray]

    input_names: List[str] = []
    output_name: str = "output"

    metadata: Dict[str, Any] = {}

    @property
    def x(self) -> NdArray:
        assert self.xval is not None
        self.check_shapes()
        return self.xval

    @property
    def y(self) -> NdArray:
        assert self.yval is not None
        self.check_shapes()
        return self.yval

    @property
    def dimensions(self) -> DataDimensions:
        return DataDimensions(input=len(self.input_names), output=1)

    def check_shapes(self) -> Self:
        assert self.xval is not None
        assert self.yval is not None

        if self.xval.ndim == 1:
            self.xval = self.xval.reshape(-1, 1)

        if self.yval.ndim == 1:
            self.yval = self.yval.reshape(-1, 1)

        if self.xval.shape[0] != self.yval.shape[0]:
            raise ValueError("X and Y must have the same number of samples")

        if self.yval.shape[1] != 1:
            raise ValueError("Y must be a 1D array")

        if self.xval.shape[1] != len(self.input_names):
            raise ValueError(
                f"X shape {self.xval.shape} does not match input names {self.input_names}"
            )

        return self

    def __deepcopy__(self, memo):
        return self


class LazyPlotData(PlotData):
    get_xy: Callable[[PlotData], Tuple[NdArray, NdArray]]

    xval: Optional[NdArray] = None
    yval: Optional[NdArray] = None

    @property
    def x(self) -> NdArray:
        self.set_xy()
        assert self.xval is not None
        return self.xval

    @property
    def y(self) -> NdArray:
        self.set_xy()
        assert self.yval is not None
        return self.yval

    def set_xy(self):
        if self.xval is None:
            self.xval, self.yval = self.get_xy.__call__(self)

    @property
    def dimensions(self) -> DataDimensions:
        return DataDimensions(input=self.x.shape[1], output=1)

    def __deepcopy__(self, memo):
        return self


def ax_to_list(ax) -> Sequence:
    if isinstance(ax, np.ndarray):
        return ax.tolist()
    return ut.as_list(ax)


SequenceND: TypeAlias = Sequence[T] | Sequence[Sequence[T]] | Sequence[Sequence[Sequence[T]]]


class FigAx(ArbitraryModel):
    figure: Figure
    ax: Annotated[SequenceND[Axes], BeforeValidator(ax_to_list)]

    @property
    def flat_ax(self) -> List[Axes]:
        return ut.flatten(self.ax)

    @property
    def n_axes(self) -> int:
        return len(self.flat_ax)


class FigureLayout(ArbitraryModel):
    def make_figure(self) -> FigAx:
        raise NotImplementedError()

    def finalize(self, figax: FigAx) -> None:
        pass


def get_figsize_default():
    fs = mpl.rcParams["figure.figsize"]
    print(f"Default figsize: {fs}")
    return fs


class SimpleLayout(FigureLayout):
    rows: int = 1
    cols: int = 1
    axes_size: Optional[Pair[float]] = None 
    kwargs: Dict[str, Any] = {}

    def make_figure(self, **kw) -> FigAx:
        if self.axes_size is None:
            self.axes_size = get_figsize_default()

        fig, ax = plt.subplots(
            self.rows,
            self.cols,
            figsize=(self.cols * self.axes_size[0], self.rows * self.axes_size[1]),
            **self.kwargs,
            **kw,
        )
        return FigAx(figure=fig, ax=ax)

    def finalize(self, figax: FigAx) -> None:
        figax.figure.tight_layout()
        pass


ValidatedFigureLayout = Annotated[
    FigureLayout,
    BeforeValidator(
        partial(
            build_if_has_target,
            available_module_names=["biocomp.plotutils", "__main__"],
        )
    ),
]


class FigureSpec(ArbitraryModel):
    title: Optional[str] = None
    output_dir: str = "./"
    output_file: Optional[str] = "unnamed.png"
    extra_args: Dict[str, Any] = {}
    layout: ValidatedFigureLayout = Field(default_factory=SimpleLayout)

    def make_figure(self) -> FigAx:
        return self.layout.make_figure(**self.extra_args)

    def save_figure(self, figax: FigAx) -> None:
        assert self.output_file is not None
        self._output_path = Path(self.output_dir) / self.output_file
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        figax.figure.savefig(self._output_path, bbox_inches="tight")

    def finalize(self, figax: FigAx) -> None:
        if self.title is not None:
            figax.figure.suptitle(self.title)
        self.layout.finalize(figax)
        if self.output_file is not None:
            self.save_figure(figax)

        plt.close(figax.figure)


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                       --     network utils     --


def get_reordered_protein_names(
    network: Network,
    input_order: Optional[Sequence[int] | Sequence[str]] = None,
    protein_aliases: Optional[Dict[str, str]] = None,
):
    protein_aliases = protein_aliases or {}
    protein_order, protein_names = pc.get_reordered_protein_names(
        network,
        input_order,
        protein_aliases,
    )
    input_order, output_pos = protein_order[:-1], protein_order[-1]
    input_names, output_name = protein_names[:-1], protein_names[-1]

    return input_order, output_pos, input_names, output_name


def extract_plot_data_from_network(
    network: Network,
    X: NdArray,
    Y: NdArray,
    input_order: Optional[Sequence[int] | Sequence[str]] = None,
    protein_aliases: Optional[Dict[str, str]] = None,
    **kw,
) -> PlotData:
    input_order, output_pos, input_names, output_name = get_reordered_protein_names(
        network, input_order, protein_aliases
    )

    assert X.shape[0] == Y.shape[0], f"X shape: {X.shape}, Y shape: {Y.shape}"
    x = X[:, input_order]
    y = Y[:, output_pos].reshape(-1, 1)
    assert x.shape[1] == len(input_order), f"X shape: {x.shape}, input_order: {input_order}"
    assert y.shape[0] == x.shape[0], f"y shape: {y.shape}, x shape: {x.shape}"

    return PlotData(
        xval=x,
        yval=y,
        input_names=input_names,
        output_name=output_name,
        **kw,
    )


def extract_lazy_plot_data_from_network(
    network: Network,
    get_XY: Callable[[PlotData], Tuple[NdArray, NdArray]],
    input_order: Optional[Sequence[int] | Sequence[str]] = None,
    protein_aliases: Optional[Dict[str, str]] = None,
    **kw,
) -> LazyPlotData:
    input_order, output_pos, input_names, output_name = get_reordered_protein_names(
        network, input_order, protein_aliases
    )

    def get_xy(pdata):
        X, Y = get_XY(pdata)
        x = X[:, input_order]
        y = Y[:, output_pos].reshape(-1, 1)
        return x, y

    pdata = LazyPlotData(
        get_xy=get_xy,
        input_names=input_names,
        output_name=output_name,
        **kw,
    )
    print(f"Lazy plot data: {pdata.input_names} -> {pdata.output_name}")
    return pdata


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

DEFAULT_GREY = "#777777"


@dataclass
class SpineProps:
    visible: bool = True
    linewidth: float = 0.5
    color: str = DEFAULT_GREY


DEFAULT_SPINE_PROPS: Dict[str, SpineProps] = {
    "top": SpineProps(visible=False),
    "right": SpineProps(visible=False),
    "bottom": SpineProps(visible=True),
    "left": SpineProps(visible=True),
}

DEFAULT_TICK_PARAMS: List[Dict[str, Any]] = [
    {"axis": "both", "which": "both", "labelsize": 8, "direction": "out"},
    {"axis": "both", "which": "major", "length": 5, "width": 0.4},
    {"axis": "both", "which": "minor", "length": 2, "width": 0.2},
]


@dataclass
class FontProps:
    family: Optional[str] = "Arial"
    size: Optional[int] = 10
    weight: Optional[str] = "normal"
    style: Optional[str] = "normal"
    color: Optional[str] = "black"


@dataclass
class PlotStyle:
    facecolor: Optional[str] = "white"
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
    def format_field(self, value, format_spec, precision=1):
        if format_spec == "m":
            if value < 1000:
                if value == int(value):
                    return super().format_field(int(value), "")
                else:
                    # use required precision:
                    return super().format_field(value, f".{precision}f")
            else:
                if value == int(value):
                    return super().format_field(value, ".0e").replace("e+0", "e").replace("e+", "e")
                else:
                    return super().format_field(value, ".1e").replace("e+0", "e").replace("e+", "e")
        else:
            return super().format_field(value, format_spec)


scformat = ShortScientificFormatter()


##────────────────────────────────────────────────────────────────────────────}}}


## {{{                --     new network plot functions     --


DEFAULT_PLOT_METHOD_PREFERENCE: Dict[int, str] = {
    1: "smooth",
    2: "smooth",
    3: "smooth",
}


@configurable
def auto_plot(
    plot_data: PlotData,
    figure_spec: Optional[FigureSpec] = None,
    ax: Optional[Union[Axes, Sequence[Axes]]] = None,
    use_plot_method: Optional[str] = "auto",  # could be 'auto' or None, or a specific method
    plot_method_per_input_dim: Optional[Dict[int, str]] = DEFAULT_PLOT_METHOD_PREFERENCE,
    rc_context: Dict[str, Any] = {},  # pc.DEFAULT_RC_PARAMS,
    smooth_params: Dict[str, Any] = {},
    scatter_params: Dict[str, Any] = {},
    histogram_params: Dict[str, Any] = {},
    **kw,
) -> None:
    dim = plot_data.dimensions
    if use_plot_method is None or use_plot_method == "auto":
        assert plot_method_per_input_dim is not None
        use_plot_method = plot_method_per_input_dim.get(dim.input, "smooth")

    VALID_METHODS = ["smooth", "scatter", "histogram"]
    if use_plot_method not in VALID_METHODS:
        raise ValueError(f"Unknown plotting method {use_plot_method}. Available: {VALID_METHODS}")

    if figure_spec is None:
        figure_spec = FigureSpec()

    with mpl.rc_context(rc_context):
        # first we check that we have axes to plot on
        # if not, we need to make a new figure
        assert dim.output == 1, "Only single output plots are supported"

        if ax is None:
            assert (
                figure_spec.layout is not None
            ), "Layout must be specified if axes are not provided"
            cols, rows = figure_spec.layout
            fig, ax = plt.subplots(
                *figure_spec.layout,
                figsize=(cols * figure_spec.axes_size[0], rows * figure_spec.axes_size[1]),
                dpi=figure_spec.dpi,
            )
        else:
            if not isinstance(ax, (list, tuple)):
                assert isinstance(ax, Axes), f"ax type is {type(ax)}"
                ax = [ax]
            fig = ax[0].get_figure()

        assert isinstance(fig, Figure)
        if figure_spec.title is not None:
            fig.suptitle(figure_spec.title)

        if use_plot_method == "smooth":
            return smooth(plot_data, ax, **smooth_params, **kw)

        else:
            raise NotImplementedError(f"Unimplemented plotting method {method}")


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                          --     main smooth dispatcher (route to 1D, 2D, 3D)    --

from .plotting.plotting_3d import smooth_3d
from .plotting.plotting_smooth import smooth_2d, smooth_1d


def combine_dicts(*kwarg_lists):
    res = {}
    for kw in kwarg_lists:
        res.update(kw)
    return res


@configurable
def smooth(
    plot_data: PlotData,
    ax,
    rescaler: DataRescaler,
    force_dim: Optional[int] = None,
    smooth_1d_params={},
    smooth_2d_params={},
    smooth_3d_params={},
    **kw,
):
    dim = plot_data.dimensions
    x = rescaler.fwd(plot_data.x)
    y = rescaler.fwd(plot_data.y)

    if force_dim is None:
        match (dim.input, dim.output):
            case (1, 1):
                force_dim = 1
            case (2, 1):
                force_dim = 2
            case (3, 1):
                force_dim = 3
            case _:
                raise ValueError(
                    f"Plotting {dim.input} inputs and {dim.output} outputs is not supported"
                )

    if force_dim == 1:
        return smooth_1d(
            X=x,
            Y=y,
            input_names=plot_data.input_names,
            output_name=plot_data.output_name,
            rescaler=rescaler,
            ax=ax,
            **combine_dicts(
                smooth_1d_params,
                kw,
            ),
        )

    if force_dim == 2:
        return smooth_2d(
            X=x,
            Y=y,
            input_names=plot_data.input_names,
            output_name=plot_data.output_name,
            rescaler=rescaler,
            ax=ax,
            **combine_dicts(
                smooth_2d_params,
                kw,
            ),
        )

    if force_dim == 3:
        return smooth_3d(
            X=x,
            Y=y,
            input_names=plot_data.input_names,
            output_name=plot_data.output_name,
            rescaler=rescaler,
            ax=ax,
            **combine_dicts(
                smooth_3d_params,
                kw,
            ),
        )
    else:
        raise ValueError(f"Unknown force_dim value {force_dim}")


##────────────────────────────────────────────────────────────────────────────}}}


DEFAULT_VIOLIN_PARAMS = {
    "showmeans": False,
    "showmedians": True,
    "showextrema": False,
    "bw_method": 0.1,
    "points": 2000,
    "vert": True,
}


@configurable
def violin_style(
    parts,
    facecolor="#bbb",
    edgecolor="#777",
    linewidth=0.5,
    cmean_color="#000",
    cmedian_color="#222",
    alpha=0.5,
):
    for pc in parts["bodies"]:
        pc.set_facecolor(facecolor)
        pc.set_edgecolor(edgecolor)
        pc.set_linewidth(linewidth)
        pc.set_alpha(alpha)

    if "cmeans" in parts:
        for pc in ut.as_list(parts["cmeans"]):
            pc.set_color(cmean_color)
            pc.set_linewidth(linewidth)

    if "cmedians" in parts:
        for pc in ut.as_list(parts["cmedians"]):
            pc.set_color(cmedian_color)
            pc.set_linewidth(linewidth)


@configurable
def normalized_violin(
    plot_data: PlotData,
    ax,
    rescaler,
    title: Optional[str] = None,
    xlims=(0, 1),
    ylims=(0, 1),
    vlims=(0, 1.5),
    xbins=20,
    draw_xlabel=True,
    draw_ylabel=True,
    cmap=pc.DEFAULT_CMAP_NAME,
    violin_params={},
    violin_style_params={},
    mean_marker="o",
    mean_color="black",
    mean_size=7,
    mean_linewidth=0.3,
    mean_linealpha=0.25,
    ratio_uses_rescaled_values=True,
    whisker_pos=(0.1, 0.9),
    whisker_color="#333333",
    whisker_linewidth=0.5,
    write_y_bounds=True,
    use_log_density=True,
):
    violin_params = {**DEFAULT_VIOLIN_PARAMS, **violin_params}

    dim = plot_data.dimensions

    x = rescaler.fwd(plot_data.x)
    y = rescaler.fwd(plot_data.y)
    assert dim.output == 1, "Only single output plots are supported"
    assert dim.input == 2, "Only 2D input plots are supported"

    # keep only inbounds data
    xlims = (
        xlims[0] if xlims[0] is not None else x[:, 0].min(),
        xlims[1] if xlims[1] is not None else x[:, 0].max(),
    )
    ylims = (
        ylims[0] if ylims[0] is not None else x[:, 1].min(),
        ylims[1] if ylims[1] is not None else x[:, 1].max(),
    )
    mask = (
        (x[:, 0] >= xlims[0])
        & (x[:, 0] <= xlims[1])
        & (x[:, 1] >= ylims[0])
        & (x[:, 1] <= ylims[1])
    )
    x = x[mask]
    y = y[mask]

    # now for each bin in x1, we want to plot a violin plot of y/x2
    x1 = x[:, 0]
    x2 = x[:, 1]

    if ratio_uses_rescaled_values:
        normed_y = y / x2[:, None]
    else:
        normed_y = rescaler.inv(y) / rescaler.inv(x2[:, None])

    x1_bins = np.linspace(*xlims, xbins)
    bin_inds = np.digitize(x1, x1_bins)
    x1_centers = 0.5 * (x1_bins[:-1] + x1_bins[1:])

    width = (x1_bins[1] - x1_bins[0]) * 0.8

    cmap = plt.get_cmap(cmap)
    quantiles = np.nanquantile(normed_y, whisker_pos, axis=0)
    binned_normed_y = [normed_y[bin_inds == i] for i in range(1, len(x1_bins))]
    mean_ys = np.array([np.nanmean(ny) for ny in binned_normed_y])

    for i, x1_center in enumerate(x1_centers):
        ny = binned_normed_y[i]
        if ny.size == 0:
            continue

        parts = ax.violinplot(ny, positions=[x1_center], widths=width, **violin_params)

        # # now we actually want to use the log density so we will compute the kde separately
        # kde = gaussian_kde(ny)
        # x = np.linspace(vlims[0], vlims[1], 1000)
        # y = kde(x)
        # # we can use mpl violin now
        # parts = ax.violin(x=x1_center, y=y, **violin_params)

        # meany = np.nanmean(ny)
        meany = mean_ys[i]
        facecolor = mpl.colors.rgb2hex(cmap(meany))
        violin_style(parts, **{"facecolor": facecolor, **violin_style_params})
        # add whiskers
        ax.plot([x1_center, x1_center], quantiles, color=whisker_color, linewidth=whisker_linewidth)
        # add mean markers
    ax.scatter(
        x1_centers,
        mean_ys,
        marker=mean_marker,
        color=mean_color,
        s=mean_size,
        linewidth=mean_linewidth,
        zorder=10,
    )
    if mean_linealpha > 0 and mean_linewidth > 0:
        ax.plot(
            x1_centers, mean_ys, color=mean_color, linewidth=mean_linewidth, alpha=mean_linealpha
        )

    pc.setup_transformed_xaxis(
        ax,
        xaxis_lims=xlims,
        rescaler=rescaler,
        margins=0.0,
    )

    ax.set_ylim(vlims)

    if write_y_bounds:
        tr_min, tr_max = rescaler.inv(np.array(ylims).reshape(-1, 1))
        tr_min = scformat.format_field(tr_min[0], "m", 0)
        tr_max = scformat.format_field(tr_max[0], "m", 0)
        latext = f"{plot_data.input_names[1]} $\\in [{tr_min}, {tr_max}]$"
        ax.text(
            0.7,
            0.9,
            latext,
            fontsize=7,
            transform=ax.transAxes,
            fontdict={"family": "monospace"},
            color=DEFAULT_GREY,
            ha="left",
            va="top",
        )

    if title is not None:
        ax.set_title(title)

    if draw_xlabel:
        ax.set_xlabel(plot_data.input_names[0])
    if draw_ylabel:
        ax.set_ylabel(f"{plot_data.output_name} / {plot_data.input_names[1]}")
