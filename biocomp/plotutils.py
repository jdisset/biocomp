# {{{                          --     imports     --
# ···············································································
import jax.numpy as jnp
import numpy as np
from biocomp import utils as ut
from biocomp.datautils import DataRescaler
import matplotlib.pyplot as plt
from biocomp.network import Network
from biocomp.utils import ArbitraryModel
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
    input: int = 0
    output: int = 0


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
        if not isinstance(self.input_names, list):
            logger.warning(f"Input names are not a list: {self.input_names}")
            return DataDimensions()
        if len(self.input_names) > 0:
            return DataDimensions(input=len(self.input_names), output=1)
        return DataDimensions(input=0, output=1)

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
        if not isinstance(self.input_names, list):
            logger.warning(f"Input names are not a list: {self.input_names}")
            return DataDimensions()
        if len(self.input_names) > 0:
            return DataDimensions(input=len(self.input_names), output=1)
        return DataDimensions(input=0, output=1)

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
    return fs


class SimpleLayout(FigureLayout):
    rows: int = 1
    cols: int = 1
    axes_size: Optional[Pair[float]] = None
    kwargs: Dict[str, Any] = {}
    wspace: Optional[float] = None
    hspace: Optional[float] = None

    def make_figure(self, **kw) -> FigAx:
        if self.axes_size is None:
            self.axes_size = get_figsize_default()

        # Create figure and axes
        fig, ax = plt.subplots(
            self.rows,
            self.cols,
            figsize=(self.cols * self.axes_size[0], self.rows * self.axes_size[1]),
            **self.kwargs,
            **kw,
        )
        return FigAx(figure=fig, ax=ax)

    def finalize(self, figax: FigAx) -> None:
        if self.wspace is None and self.hspace is None:
            figax.figure.tight_layout()
        else:
            figax.figure.tight_layout(w_pad=self.wspace, h_pad=self.hspace)


class GridLayout(FigureLayout):
    rows: int = 1
    cols: int = 1
    axes_size: Optional[Pair[float]] = None
    kwargs: Dict[str, Any] = {}
    wspace: Optional[float] = None
    hspace: Optional[float] = None
    col_widths: Optional[List[float]] = None
    row_heights: Optional[List[float]] = None

    def __init__(self, **data):
        super().__init__(**data)
        self._validate_dimensions()

    def _validate_dimensions(self) -> None:
        if self.col_widths is not None:
            if len(self.col_widths) != self.cols:
                raise ValueError(f"col_widths must have length {self.cols}")
            if abs(sum(self.col_widths) - 1.0) > 1e-6:
                raise ValueError("col_widths must sum to 1")

        if self.row_heights is not None:
            if len(self.row_heights) != self.rows:
                raise ValueError(f"row_heights must have length {self.rows}")
            if abs(sum(self.row_heights) - 1.0) > 1e-6:
                raise ValueError("row_heights must sum to 1")

    def make_figure(self, **kw) -> FigAx:
        if self.axes_size is None:
            default_size = get_figsize_default()
            self.axes_size = default_size

        wspace = 0.2 if self.wspace is None else self.wspace
        hspace = 0.2 if self.hspace is None else self.hspace

        spacing_width = wspace * (self.cols - 1) * self.axes_size[0]
        spacing_height = hspace * (self.rows - 1) * self.axes_size[1]

        margin = 0.05
        margin_width = 2 * margin * self.axes_size[0]
        margin_height = 2 * margin * self.axes_size[1]

        # total figure size including spacing and margins
        fig_width = (self.axes_size[0] * self.cols) + spacing_width + margin_width
        fig_height = (self.axes_size[1] * self.rows) + spacing_height + margin_height

        fig = plt.figure(figsize=(fig_width, fig_height))

        gs = fig.add_gridspec(
            self.rows,
            self.cols,
            width_ratios=self.col_widths,
            height_ratios=self.row_heights,
            wspace=wspace,
            hspace=hspace,
            top=1 - margin,
            bottom=margin,
            left=margin,
            right=1 - margin,
            **self.kwargs,
            **kw,
        )

        # Create axes as a nested list structure
        axes = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(fig.add_subplot(gs[i, j]))
            axes.append(row)

        return FigAx(figure=fig, ax=axes)

    def finalize(self, figax: FigAx) -> None:
        """
        Finalize the figure layout.
        """
        if self.wspace is None and self.hspace is None:
            figax.figure.tight_layout()


class FigureSpec(ArbitraryModel):
    title: Optional[str] = None
    title_kwargs: Dict[str, Any] = {}
    output_dir: str = "./"
    output_file: Optional[str] = "unnamed.png"
    extra_args: Dict[str, Any] = {}
    layout: FigureLayout = Field(default_factory=SimpleLayout)

    @property
    def output_path(self) -> Path:
        assert self.output_file is not None
        return Path(self.output_dir) / self.output_file

    def make_figure(self) -> FigAx:
        return self.layout.make_figure(**self.extra_args)

    def save_figure(self, figax: FigAx) -> None:
        assert self.output_file is not None
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        figax.figure.savefig(self.output_path, bbox_inches="tight")

    def finalize(self, figax: FigAx) -> None:
        if self.title is not None:
            figax.figure.suptitle(self.title, **self.title_kwargs)
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


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                    --     misc plot styling tools     --

DEFAULT_GREY = "#777777"


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
## {{{                          --     main smooth dispatcher (route to 1D, 2D, 3D)    --

from .plotting.plotting_3d import smooth_3d
from .plotting.plotting_smooth import smooth_2d, smooth_1d
from .plotting.plotting_scatter import grid_histogram


def combine_dicts(*kwarg_lists):
    res = {}
    for kw in kwarg_lists:
        res.update(kw)
    return res


@configurable
def histogram(
    plot_data: PlotData,
    ax,
    rescaler: DataRescaler,
    grid_histogram_params={},
    **kw,
):
    dim = plot_data.dimensions
    x = rescaler.fwd(plot_data.x)
    y = rescaler.fwd(plot_data.y)

    if (dim.input, dim.output) != (1, 1):
        raise ValueError(
            f"Histogram plotting currently only supports 1 input and 1 output, "
            f"got {dim.input} inputs and {dim.output} outputs"
        )

    return grid_histogram(
        X=x,
        Y=y,
        input_names=plot_data.input_names,
        output_name=plot_data.output_name,
        rescaler=rescaler,
        ax=ax,
        **combine_dicts(
            grid_histogram_params,
            kw,
        ),
    )


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
