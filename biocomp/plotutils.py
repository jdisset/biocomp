# {{{                          --     imports     --
# ···············································································
import numpy as np
from biocomp import utils as ut
from biocomp.datautils import DataRescaler, IdentityRescaler
import matplotlib.pyplot as plt
from biocomp.network import Network
from biocomp.utils import ArbitraryModel
from dracon.utils import dict_like, list_like
import string
import os
from typing import (
    Union,
    Self,
    Literal,
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


configurable = ut.configurable_decorator("biocomp.plotting")
os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"

##────────────────────────────────────────────────────────────────────────────}}}

logger = get_logger(__name__)


class PlotFunctionResult:
    """Return type for plot functions that want to attach computed metadata.

    Tuple-unpackable for backward compat: `a, b = plot_func(...)` still works.
    """

    __slots__ = ("rendering", "metadata")

    def __init__(self, rendering: Any, metadata: dict[str, Any] | None = None):
        self.rendering = rendering
        self.metadata = metadata or {}

    def __iter__(self):
        return iter(self.rendering)

    def __len__(self):
        return len(self.rendering)

    def __getitem__(self, idx):
        return self.rendering[idx]


# ---- network/recipe plots

## {{{                      --     plot data class     --

T = TypeVar("T")
Pair: TypeAlias = Tuple[T, T]
ListOrSingle: TypeAlias = Union[List[T], T]
NdArray: TypeAlias = np.ndarray
NumLike: TypeAlias = Union[np.ndarray, float, int]


class DataDimensions(BaseModel):
    input: int = 0
    output: int = 0


def asarray(x):
    return np.asarray(x, dtype=np.float32) if x is not None else None


class PlotData(ArbitraryModel):
    xval: Annotated[Optional[NdArray], BeforeValidator(asarray)]
    yval: Annotated[Optional[NdArray], BeforeValidator(asarray)]

    input_names: List[str] = []
    output_name: str | List[str] = "output"

    # Canonical protein-name identity of each X column, in the network's
    # `get_inverted_input_proteins()` namespace (no display aliases applied).
    # `None` means "X is not anchored to a specific network's wiring" — the
    # boundary assertions at NetworkPrediction will skip identity checking
    # in that case (used for design-space PlotData with placeholder X1/X2
    # labels). Producers of network-aligned PlotData (extract_*_from_network)
    # MUST set this so X-column scrambling can be detected at handoff.
    column_proteins: Optional[List[str]] = None

    metadata: Dict[str, Any] = {}

    force_single_output: bool = True

    disable_check_shapes: bool = False

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
        self.check_shapes()
        if not isinstance(self.input_names, list):
            logger.warning(f"Input names are not a list: {self.input_names}")
            return DataDimensions()
        if len(self.input_names) > 0:
            return DataDimensions(input=len(self.input_names), output=1)
        return DataDimensions(input=0, output=1)

    def check_shapes(self) -> Self:
        if self.disable_check_shapes:
            return self
        assert self.xval is not None
        assert self.yval is not None

        if self.xval.ndim == 1:
            self.xval = self.xval.reshape(-1, 1)

        if self.yval.ndim == 1:
            self.yval = self.yval.reshape(-1, 1)

        if self.xval.shape[0] != self.yval.shape[0]:
            raise ValueError(
                f"X and Y must have the same number of samples. Shapes are {self.xval.shape} and {self.yval.shape}"
            )

        if self.yval.shape[1] > 1:
            assert len(self.output_name) == self.yval.shape[1], (
                f"Output name {self.output_name} does not match the number of outputs {self.yval.shape[1]}"
            )
            if self.force_single_output:
                # we just put the extra outputs as inputs
                print(f"Y has {self.yval.shape[1]} outputs!!!")
                logger.warning(
                    f"Y has {self.yval.shape[1]} outputs, but only 1 output is expected. "
                    f"Using the first output as the main output."
                )
                newxval = np.concatenate([self.xval, self.yval[:, 1:]], axis=1)
                self.xval = newxval
                self.yval = self.yval[:, :1]
                self.input_names.extend(self.output_name[1:])
                self.output_name = self.output_name[0]
                print(f"New xval shape: {self.xval.shape}, new yval shape: {self.yval.shape}")

        return self

    def __deepcopy__(self, memo):
        return self


class LazyPlotData(PlotData):
    get_xy: Callable[[PlotData], Tuple[NdArray, NdArray]]

    xval: Annotated[Optional[NdArray], BeforeValidator(asarray)] = None
    yval: Annotated[Optional[NdArray], BeforeValidator(asarray)] = None

    @property
    def x(self) -> NdArray:
        self.set_xy()
        assert self.xval is not None
        return asarray(self.xval)  # type: ignore

    @property
    def y(self) -> NdArray:
        self.set_xy()
        assert self.yval is not None
        return asarray(self.yval)  # type: ignore

    def set_xy(self):
        if self.xval is None:
            self.xval, self.yval = self.get_xy.__call__(self)
        self.check_shapes()

    @property
    def dimensions(self) -> DataDimensions:
        self.set_xy()
        if not isinstance(self.input_names, list):
            logger.warning(f"Input names are not a list: {self.input_names}")
            return DataDimensions()
        if len(self.input_names) > 0:
            return DataDimensions(input=len(self.input_names), output=1)
        return DataDimensions(input=0, output=1)

    def __deepcopy__(self, memo):
        return self

    def __repr__(self):
        if self.xval is None:
            return f"LazyPlotData[not loaded, get_xy={self.get_xy}]"
        else:
            return f"LazyPlotData[loaded with {len(self.xval)} samples]"

    def __str__(self):
        return self.__repr__()


def ax_to_list(ax):
    if ax is None:
        return None
    if isinstance(ax, list):
        return ax
    if isinstance(ax, np.ndarray):
        return ax.tolist()
    return ut.as_list(ax)


SequenceND: TypeAlias = Sequence[T] | Sequence[Sequence[T]] | Sequence[Sequence[Sequence[T]]]


class FigAx(ArbitraryModel):
    figure: Figure
    ax: Annotated[Optional[SequenceND[Axes]], BeforeValidator(ax_to_list)] = None
    subfigs: Any = None

    @property
    def flat_ax(self) -> List[Axes]:
        return ut.flatten(self.ax)

    @property
    def n_axes(self) -> int:
        return len(self.flat_ax)


class FigureLayout(ArbitraryModel):
    def make_figure(self):
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

    def make_figure(self, **kw):
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


FIGURE_METADATA_KEY = "FigureMetadata"


def sanitize_for_json(obj, max_depth: int = 50, _depth: int = 0):
    """Recursively convert objects to JSON-serializable form, including tuple keys."""
    if _depth > max_depth:
        return str(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, tuple):
        return [sanitize_for_json(v, max_depth, _depth + 1) for v in obj]
    if dict_like(obj):
        return {
            (
                str(k) if not isinstance(k, (str, int, float, bool, type(None))) else k
            ): sanitize_for_json(v, max_depth, _depth + 1)
            for k, v in obj.items()
        }
    if list_like(obj):
        return [sanitize_for_json(v, max_depth, _depth + 1) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if hasattr(obj, "__dict__"):
        return {
            k: sanitize_for_json(v, max_depth, _depth + 1)
            for k, v in obj.__dict__.items()
            if not k.startswith("_")
        }
    return str(obj)


class MergeSpec(ArbitraryModel):
    """Specification for merging multiple subfigures into one output.

    Modes:
        - "grid": Arrange figures in a grid layout (default)
        - "pages": Each figure becomes a separate page in a PDF
    """

    mode: Literal["grid", "pages"] = "grid"
    rows: int = 1
    cols: int = 1
    row_heights: Optional[List[float]] = None
    col_widths: Optional[List[float]] = None
    output_dir: str = "./"
    output_file: str = "merged.png"
    title: Optional[str] = None
    hspace: int = 10
    vspace: int = 10
    bg_color: str = "white"
    delete_intermediates: bool = True

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir) / self.output_file


class FigureSpec(ArbitraryModel):
    title: Optional[str] = None
    title_kwargs: Dict[str, Any] = {}
    output_dir: str = "./"
    output_file: Optional[str] = "unnamed.png"
    extra_args: Dict[str, Any] = {}
    layout: FigureLayout = Field(default_factory=SimpleLayout)
    dpi: int = 300
    metadata: Dict[str, Any] = {}

    @property
    def output_path(self) -> Path:
        assert self.output_file is not None
        return Path(self.output_dir) / self.output_file

    def make_figure(self) -> FigAx:
        fax = self.layout.make_figure(**self.extra_args)
        return fax

    def save_figure(self, figax: FigAx) -> None:
        import io
        import json
        import shutil
        import tempfile
        import time
        from datetime import datetime

        from PIL import Image
        from PIL.PngImagePlugin import PngInfo

        assert self.output_file is not None
        output_path = self.output_path
        parent_dir = output_path.parent

        # For cloud-synced dirs (Dropbox, etc.), save to temp then move atomically
        is_cloud_sync = "dropbox" in str(parent_dir).lower()
        if is_cloud_sync:
            temp_dir = Path(tempfile.mkdtemp(prefix="biocomp_plot_"))
            temp_path = temp_dir / output_path.name
        else:
            temp_dir = None
            temp_path = output_path

        # Ensure directory exists (only needed if not using temp)
        if not is_cloud_sync:
            for attempt in range(10):
                parent_dir.mkdir(parents=True, exist_ok=True)
                try:
                    list(parent_dir.iterdir())
                    break
                except OSError:
                    time.sleep(0.2 * (attempt + 1))
            else:
                raise FileNotFoundError(f"Could not create directory: {parent_dir}")

        # sanitize metadata to handle tuple keys, numpy arrays, etc.
        sanitized_metadata = sanitize_for_json(self.metadata)
        metadata_json = json.dumps({FIGURE_METADATA_KEY: sanitized_metadata}, indent=2)

        timestamp = datetime.now().isoformat()

        full_metadata = {
            "Creator": "biocomp",
            "Author": "biocomp",
            "Subject": metadata_json,
            "Title": self.title or "Biocomp Figure",
            "CreationDate": timestamp,
        }

        if str(output_path).lower().endswith(".png"):
            buf = io.BytesIO()
            figax.figure.savefig(buf, format="png", bbox_inches="tight", dpi=self.dpi)
            buf.seek(0)
            with Image.open(buf) as img:
                metadata = PngInfo()
                for key, value in full_metadata.items():
                    metadata.add_text(key, value)
                img.save(temp_path, pnginfo=metadata)
        elif str(output_path).lower().endswith(".pdf"):
            full_metadata["CreationDate"] = datetime.now()  # type: ignore
            figax.figure.savefig(temp_path, metadata=full_metadata, bbox_inches="tight")
        elif str(output_path).lower().endswith(".svg"):
            figax.figure.savefig(temp_path, format="svg", bbox_inches="tight")
            self._postprocess_svg(temp_path, full_metadata)
        else:
            logger.warning(
                f"Saving figure to {output_path} in {output_path.suffix} format. "
                f"Only PNG, PDF, and SVG formats have full metadata support."
            )
            figax.figure.savefig(temp_path, metadata=full_metadata, bbox_inches="tight")

        # Move from temp to final destination for cloud-synced directories
        if is_cloud_sync:
            for attempt in range(10):
                parent_dir.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.move(str(temp_path), str(output_path))
                    break
                except OSError:
                    time.sleep(0.5 * (attempt + 1))
            else:
                raise FileNotFoundError(f"Could not save to cloud-synced directory: {output_path}")
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _postprocess_svg(self, svg_path: Path, full_metadata: Dict[str, Any]) -> None:
        """Post-process SVG file to add custom metadata and attributes"""
        import re

        # Read the original SVG content
        with open(svg_path, "r", encoding="utf-8") as f:
            svg_content = f.read()

        # Find and update biocomp-tagged elements
        def process_biocomp_element(match):
            full_tag = match.group(0)
            gid = match.group(1)

            if gid.startswith("biocomp_"):
                try:
                    parts = gid.split("_", 2)
                    if len(parts) >= 3:
                        elem_type = parts[1]
                        elem_data = parts[2]

                        # Build data attributes
                        attrs = f' data-biocomp-type="{elem_type}" data-biocomp-data="{elem_data}"'

                        # For 3d slices, parse the z-value
                        if elem_type == "3dslice":
                            z_parts = elem_data.split("z")
                            if len(z_parts) >= 2:
                                attrs += f' data-z-value="{z_parts[1]}"'
                                attrs += f' data-slice-index="{z_parts[0].rstrip("_")}"'
                                # Only add class to group elements (not images)
                                if not elem_data.endswith("_image"):
                                    attrs += ' class="biocomp-3d-slice"'

                        # Insert attributes before the closing > or />
                        if full_tag.endswith("/>"):
                            return full_tag[:-2] + attrs + "/>"
                        else:
                            return full_tag[:-1] + attrs + ">"
                except Exception as e:
                    logger.debug(f"Could not process biocomp element '{gid}': {e}")

            return full_tag

        # Apply processing to all elements with biocomp IDs (including self-closing tags)
        svg_content = re.sub(
            r'<[^>]+id="(biocomp_[^"]*)"[^>]*/?>', process_biocomp_element, svg_content
        )

        # Add biocomp metadata to existing metadata section or create new one
        biocomp_metadata_xml = "  <!-- Biocomp Metadata -->\n"
        for key, value in full_metadata.items():
            # Escape XML special characters
            escaped_value = (
                str(value)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
            )
            biocomp_metadata_xml += f'  <{key} type="biocomp-metadata">{escaped_value}</{key}>\n'

        if "<metadata>" in svg_content:
            # Insert before closing metadata tag
            svg_content = svg_content.replace("</metadata>", biocomp_metadata_xml + "</metadata>")
        else:
            # Create new metadata section after opening svg tag
            metadata_xml = f"<metadata>\n{biocomp_metadata_xml}</metadata>\n"
            svg_content = re.sub(r"(<svg[^>]*>)", r"\1\n" + metadata_xml, svg_content)

        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_content)

    def finalize(self, figax: FigAx) -> None:
        if self.title is not None:
            figax.figure.suptitle(self.title, **self.title_kwargs)
        self.layout.finalize(figax)
        if self.output_file is not None:
            self.save_figure(figax)

        plt.close(figax.figure)


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                       --     network utils     --


# SSOT re-export: the canonical definition lives in biocomp.plotting.plotting_core.
# Returns (in_order, output_pos, reordered_input_names, output_name).
get_reordered_protein_names = pc.get_reordered_protein_names


def extract_plot_data_from_network(
    network: Network,
    X: NdArray,
    Y: NdArray,
    input_order: Optional[Sequence[int] | Sequence[str]] = None,
    protein_aliases: Optional[Dict[str, str]] = None,
    only_dependent_outputs: bool = True,
    **kw,
) -> PlotData:
    input_order, output_pos, input_names, output_name = get_reordered_protein_names(
        network, input_order, protein_aliases, only_dependent_outputs
    )

    assert X.shape[0] == Y.shape[0], f"X shape: {X.shape}, Y shape: {Y.shape}"
    x = X[:, input_order]
    y = Y[:, output_pos].reshape(-1, 1)
    assert x.shape[1] == len(input_order), f"X shape: {x.shape}, input_order: {input_order}"
    assert y.shape[0] == x.shape[0], f"y shape: {y.shape}, x shape: {x.shape}"

    raw_input_names = network.get_inverted_input_proteins()
    column_proteins = [raw_input_names[i] for i in input_order]

    d = PlotData(
        xval=x,
        yval=y,
        input_names=input_names,
        output_name=output_name,
        column_proteins=column_proteins,
        **kw,
    )

    d.metadata["input_order"] = input_order
    d.metadata["output_pos"] = output_pos
    d.metadata["input_names"] = input_names
    d.metadata["output_name"] = output_name
    return d


def extract_lazy_plot_data_from_network(
    network: Network,
    get_XY: Callable[[PlotData], Tuple[NdArray, NdArray]],
    input_order: Optional[Sequence[int] | Sequence[str] | Literal["inv"]] = None,
    protein_aliases: Optional[Dict[str, str]] = None,
    only_dependent_outputs: bool = True,
    **kw,
) -> LazyPlotData:
    input_order, output_pos, input_names, output_name = get_reordered_protein_names(
        network, input_order, protein_aliases, only_dependent_outputs
    )

    logger.debug(
        f"extract_lazy_plot_data: {network.name} input_order={input_order} inputs={input_names} output={output_name}"
    )

    def get_xy(pdata: PlotData) -> Tuple[NdArray, NdArray]:
        logger.debug("get_xy({pdata}) called")
        assert isinstance(pdata, PlotData), f"pdata must be a PlotData, got {type(pdata)}"
        X, Y = get_XY(pdata)
        x = X[:, input_order]
        y = Y[:, output_pos]
        # make sure y is a column vector if 1d
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        logger.debug(f"get_xy: x{x.shape} y{y.shape}")
        return x, y

    raw_input_names = network.get_inverted_input_proteins()
    column_proteins = [raw_input_names[i] for i in input_order]

    d = LazyPlotData(
        get_xy=get_xy,
        input_names=input_names,
        output_name=output_name,
        column_proteins=column_proteins,
        **kw,
    )

    d.metadata["input_order"] = input_order
    d.metadata["output_pos"] = output_pos
    d.metadata["input_names"] = input_names
    d.metadata["output_name"] = output_name

    logger.debug(f"Extracted lazy plot data from network {network}: {d}")

    return d


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

from .plotting.plotting_3d import smooth_3d  # noqa: E402
from .plotting.plotting_smooth import smooth_2d, smooth_1d  # noqa: E402
from .plotting.plotting_scatter import grid_histogram  # noqa: E402


def combine_dicts(*kwarg_lists):
    res = {}
    for kw in kwarg_lists:
        res.update(kw)
    return res


@configurable
def histogram(
    plot_data: PlotData,
    ax,
    rescaler: Optional[DataRescaler] = None,
    grid_histogram_params=None,
    **kw,
):
    if grid_histogram_params is None:
        grid_histogram_params = {}
    if rescaler is None:
        rescaler = IdentityRescaler()

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
    smooth_1d_params=None,
    smooth_2d_params=None,
    smooth_3d_params=None,
    **kw,
):
    if smooth_3d_params is None:
        smooth_3d_params = {}
    if smooth_2d_params is None:
        smooth_2d_params = {}
    if smooth_1d_params is None:
        smooth_1d_params = {}
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


TXT_KNN_DEFAULTS = {"knn_stats_params": {"radius": 0.2, "k": 100, "min_points": 1}}


@configurable
def smooth_txt(
    plot_data: PlotData,
    ax=None,
    rescaler: DataRescaler = None,
    force_dim: Optional[int] = None,
    smooth_1d_params=None,
    smooth_2d_params=None,
    smooth_3d_params=None,
    **kw,
):
    from biocomp.plotting.plotting_txt import smooth_1d_txt, smooth_2d_txt, smooth_3d_txt

    if smooth_3d_params is None:
        smooth_3d_params = {}
    if smooth_2d_params is None:
        smooth_2d_params = {}
    if smooth_1d_params is None:
        smooth_1d_params = {}
    if rescaler is None:
        rescaler = IdentityRescaler()

    if "knn_grid_params" not in smooth_2d_params:
        smooth_2d_params = {**smooth_2d_params, "knn_grid_params": TXT_KNN_DEFAULTS}
    if "knn_grid_params" not in smooth_3d_params:
        smooth_3d_params = {**smooth_3d_params, "knn_grid_params": TXT_KNN_DEFAULTS}

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
        return smooth_1d_txt(
            X=x,
            Y=y,
            input_names=plot_data.input_names,
            output_name=plot_data.output_name,
            rescaler=rescaler,
            ax=ax,
            **combine_dicts(smooth_1d_params, kw),
        )

    if force_dim == 2:
        return smooth_2d_txt(
            X=x,
            Y=y,
            input_names=plot_data.input_names,
            output_name=plot_data.output_name,
            rescaler=rescaler,
            ax=ax,
            **combine_dicts(smooth_2d_params, kw),
        )

    if force_dim == 3:
        return smooth_3d_txt(
            X=x,
            Y=y,
            input_names=plot_data.input_names,
            output_name=plot_data.output_name,
            rescaler=rescaler,
            ax=ax,
            **combine_dicts(smooth_3d_params, kw),
        )

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
    for body in parts["bodies"]:
        body.set_facecolor(facecolor)
        body.set_edgecolor(edgecolor)
        body.set_linewidth(linewidth)
        body.set_alpha(alpha)

    if "cmeans" in parts:
        for part in ut.as_list(parts["cmeans"]):
            part.set_color(cmean_color)
            part.set_linewidth(linewidth)

    if "cmedians" in parts:
        for part in ut.as_list(parts["cmedians"]):
            part.set_color(cmedian_color)
            part.set_linewidth(linewidth)


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
    violin_params=None,
    violin_style_params=None,
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
    if violin_style_params is None:
        violin_style_params = {}
    if violin_params is None:
        violin_params = {}
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
