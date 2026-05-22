# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
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
    Self,
    Literal,
    Annotated,
    Any,
    TypeVar,
    TypeAlias,
)
from collections.abc import Sequence, Callable
import matplotlib as mpl

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import (
    BaseModel,
    Field,
    BeforeValidator,
    PrivateAttr,
)

from pathlib import Path
from biocomp.plotting import plotting_core as pc
from biocomp.logging_config import get_logger
from biocomp._legacy_deprecation import warn_legacy


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
Pair: TypeAlias = tuple[T, T]
ListOrSingle: TypeAlias = list[T] | T
NdArray: TypeAlias = np.ndarray
NumLike: TypeAlias = np.ndarray | float | int


class DataDimensions(BaseModel):
    input: int = 0
    output: int = 0


def asarray(x):
    return np.asarray(x, dtype=np.float32) if x is not None else None


class PlotData(ArbitraryModel):
    xval: Annotated[NdArray | None, BeforeValidator(asarray)]
    yval: Annotated[NdArray | None, BeforeValidator(asarray)]

    input_names: list[str] = []
    output_name: str | list[str] = "output"

    # Canonical protein-name identity of each X column, in the network's
    # `get_inverted_input_proteins()` namespace (no display aliases applied).
    # `None` means "X is not anchored to a specific network's wiring" -- the
    # boundary assertions at NetworkPrediction will skip identity checking
    # in that case (used for design-space PlotData with placeholder X1/X2
    # labels). Producers of network-aligned PlotData (extract_*_from_network)
    # MUST set this so X-column scrambling can be detected at handoff.
    column_proteins: list[str] | None = None

    metadata: dict[str, Any] = {}

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
    get_xy: Callable[[PlotData], tuple[NdArray, NdArray]]

    xval: Annotated[NdArray | None, BeforeValidator(asarray)] = None
    yval: Annotated[NdArray | None, BeforeValidator(asarray)] = None

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
    ax: Annotated[SequenceND[Axes] | None, BeforeValidator(ax_to_list)] = None
    subfigs: Any = None
    _subax_cache: dict[int, dict[str, Any]] = PrivateAttr(default_factory=dict)

    @property
    def flat_ax(self) -> list[Axes]:
        return ut.flatten(self.ax)

    @property
    def n_axes(self) -> int:
        return len(self.flat_ax)

    def subdivide(self, axnum: int, spec: dict[str, Any]) -> dict[str, Any]:
        """Subdivide ``flat_ax[axnum]`` into named sub-axes per ``spec``.

        Idempotent + cached by ``axnum``: multiple plot tasks targeting the
        same parent cell can each call ``subdivide(axnum, spec)`` and share
        the resulting layout. The first call removes the parent ax and
        creates the sub-axes; subsequent calls return the cached dict.

        ``spec`` shape::

            {
              "regions": {
                "<name>": {
                  "x": float, "y": float,        # bottom-left corner, frac of parent bbox
                  "w": float, "h": float,        # size, frac of parent bbox
                  "grid": [R, C]?,               # optional sub-grid
                  "vgap_frac": float?,           # row gap, frac of region height
                  "hgap_frac": float?,           # col gap, frac of region width
                },
                ...
              }
            }

        Without ``grid``, the entry maps to a single ``Axes``. With
        ``grid: [R, C]``, the entry maps to a nested ``list[R][C]`` of
        ``Axes`` with row 0 on top (matplotlib data-coord convention).
        """
        cache = self._subax_cache
        if axnum in cache:
            return cache[axnum]

        parent = self.flat_ax[axnum]
        bbox = parent.get_position()
        parent.remove()
        fig = self.figure

        out: dict[str, Any] = {}
        for name, region in spec["regions"].items():
            x = bbox.x0 + region["x"] * bbox.width
            y = bbox.y0 + region["y"] * bbox.height
            w = region["w"] * bbox.width
            h = region["h"] * bbox.height

            grid = region.get("grid")
            if grid is None:
                out[name] = fig.add_axes((x, y, w, h))
                continue

            R, C = int(grid[0]), int(grid[1])
            assert R > 0 and C > 0, f"grid must have positive dims, got {grid}"
            vgap = float(region.get("vgap_frac", 0.0)) * h
            hgap = float(region.get("hgap_frac", 0.0)) * w
            cell_h = (h - (R - 1) * vgap) / R
            cell_w = (w - (C - 1) * hgap) / C
            assert cell_h > 0 and cell_w > 0, (
                f"subdivide: gaps too large for region {name!r} ({R}x{C}, h={h}, w={w})"
            )

            rows = []
            for r in range(R):
                cells = []
                for c in range(C):
                    cx = x + c * (cell_w + hgap)
                    cy = y + (R - 1 - r) * (cell_h + vgap)
                    cells.append(fig.add_axes((cx, cy, cell_w, cell_h)))
                rows.append(cells)
            out[name] = rows

        cache[axnum] = out
        return out


def compute_shared_vlims(
    y,
    quantiles: Sequence[float] = (0.01, 0.99),
    rescaler: DataRescaler | None = None,
) -> tuple:
    """Quantile-based shared color scale for a Y array (in latent space).

    Used to share vlims across the cube view's internal slices and across
    grid-mode side slices when consistency is preferred over per-cell
    contrast. Returns ``(None, None)`` for empty / all-NaN inputs.
    """
    arr = np.asarray(y)
    if rescaler is not None:
        arr = np.asarray(rescaler.fwd(arr))
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (None, None)
    return (
        float(np.quantile(finite, quantiles[0])),
        float(np.quantile(finite, quantiles[1])),
    )


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
    axes_size: Pair[float] | None = None
    kwargs: dict[str, Any] = {}
    wspace: float | None = None
    hspace: float | None = None

    def model_post_init(self, *a, **kw):
        super().model_post_init(*a, **kw)
        warn_legacy("biocomp.plotutils.SimpleLayout", "jeanplot.Figure + Container layout")

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
    axes_size: Pair[float] | None = None
    kwargs: dict[str, Any] = {}
    wspace: float | None = None
    hspace: float | None = None
    col_widths: list[float] | None = None
    row_heights: list[float] | None = None

    def __init__(self, **data):
        super().__init__(**data)
        self._validate_dimensions()
        warn_legacy("biocomp.plotutils.GridLayout", "jeanplot.Figure + Container layout")

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


class MultiRowGridLayout(FigureLayout):
    """A figure laid out as a vertical stack of rows where each row has its
    own column count. Each row gets an independent ``subgridspec`` so rows
    don't have to share a column structure.

    ``rows`` is a list of per-row column-width lists (relative within the row).
    ``row_heights`` is a list of relative row heights. Both are normalised
    internally -- the absolute scale is set by ``figure_size``.

    ``flat_ax`` ordering is row-major: row 0 cells first (left-to-right),
    then row 1, etc. That matches the ``axnum`` convention the row template
    uses to dispatch atomic tasks.
    """

    rows: list[list[float]]
    row_heights: list[float]
    figure_size: Pair[float] = (12.0, 8.0)
    wspace: float = 0.2
    hspace: float = 0.2
    margin: float = 0.05
    # `gap_mask[i][j] == True` marks cell (i,j) as a layout-only spacer:
    # the axes is created so `flat_ax` indexing stays aligned with the
    # original column count, but its frame/spines/ticks are hidden so the
    # column reads as pure whitespace. Same shape as `rows`. Default
    # `None` = no gaps (preserves original behaviour).
    gap_mask: list[list[bool]] | None = None
    kwargs: dict[str, Any] = {}

    def __init__(self, **data):
        super().__init__(**data)
        warn_legacy(
            "biocomp.plotutils.MultiRowGridLayout",
            "jeanplot.Figure + Container with row/col layouts",
        )
        assert len(self.rows) == len(self.row_heights), (
            f"rows ({len(self.rows)}) / row_heights ({len(self.row_heights)}) length mismatch"
        )
        for i, r in enumerate(self.rows):
            assert len(r) > 0, f"row {i} has no columns"
            assert all(w > 0 for w in r), f"row {i} has non-positive widths: {r}"
        assert all(h > 0 for h in self.row_heights), (
            f"row_heights must be positive: {self.row_heights}"
        )
        if self.gap_mask is not None:
            assert len(self.gap_mask) == len(self.rows), (
                f"gap_mask rows ({len(self.gap_mask)}) != rows ({len(self.rows)})"
            )
            for i, (gm, r) in enumerate(zip(self.gap_mask, self.rows)):
                assert len(gm) == len(r), (
                    f"gap_mask row {i} length ({len(gm)}) != row width count ({len(r)})"
                )

    def make_figure(self, **kw) -> FigAx:
        fig = plt.figure(figsize=tuple(self.figure_size))
        outer = fig.add_gridspec(
            len(self.rows),
            1,
            height_ratios=self.row_heights,
            hspace=self.hspace,
            top=1 - self.margin,
            bottom=self.margin,
            left=self.margin,
            right=1 - self.margin,
            **self.kwargs,
            **kw,
        )
        axes: list[list[Axes]] = []
        for i, row_widths in enumerate(self.rows):
            inner = outer[i].subgridspec(
                1, len(row_widths), width_ratios=row_widths, wspace=self.wspace
            )
            row_axes = [fig.add_subplot(inner[0, j]) for j in range(len(row_widths))]
            if self.gap_mask is not None:
                for j, is_gap in enumerate(self.gap_mask[i]):
                    if is_gap:
                        row_axes[j].set_axis_off()
            axes.append(row_axes)
        return FigAx(figure=fig, ax=axes)

    def finalize(self, figax: FigAx) -> None:
        # Explicit gridspec; do not let tight_layout override our spacing.
        return None


FIGURE_METADATA_KEY = "FigureMetadata"


def sanitize_for_json(obj, max_depth: int = 50, _depth: int = 0):
    """Recursively convert objects to JSON-serializable form, including tuple keys."""
    if _depth > max_depth:
        return str(obj)
    if obj is None or isinstance(obj, bool | int | float | str):
        return obj
    if isinstance(obj, tuple):
        return [sanitize_for_json(v, max_depth, _depth + 1) for v in obj]
    if dict_like(obj):
        return {
            (
                str(k) if not isinstance(k, str | int | float | bool | type(None)) else k
            ): sanitize_for_json(v, max_depth, _depth + 1)
            for k, v in obj.items()
        }
    if list_like(obj):
        return [sanitize_for_json(v, max_depth, _depth + 1) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer | np.floating):
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
    row_heights: list[float] | None = None
    col_widths: list[float] | None = None
    output_dir: str = "./"
    output_file: str = "merged.png"
    title: str | None = None
    hspace: int = 10
    vspace: int = 10
    bg_color: str = "white"
    delete_intermediates: bool = True

    def model_post_init(self, *a, **kw):
        super().model_post_init(*a, **kw)
        warn_legacy("biocomp.plotutils.MergeSpec", "jeanplot.Figure merge_subfigures")

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir) / self.output_file


class FigureSpec(ArbitraryModel):
    title: str | None = None
    title_kwargs: dict[str, Any] = {}
    # Optional second line rendered separately via `fig.text()` so it can
    # carry different kwargs (e.g. `fontweight: normal` while the bold
    # `figure.titleweight` rcParam keeps the main title bold). Positioned
    # via its own `subtitle_kwargs` (x/y in figure-coord fractions);
    # defaults sit just below the title.
    subtitle: str | None = None
    subtitle_kwargs: dict[str, Any] = {}
    output_dir: str = "./"
    output_file: str | None = "unnamed.png"
    # Additional output paths to write the same figure to (each may use a
    # different format/extension). Useful when one render should be saved as
    # both PDF and SVG, etc. Paths are resolved exactly like `output_path`.
    extra_output_paths: list[str] = []
    extra_args: dict[str, Any] = {}
    layout: FigureLayout = Field(default_factory=SimpleLayout)
    dpi: int = 300
    metadata: dict[str, Any] = {}

    def model_post_init(self, *a, **kw):
        super().model_post_init(*a, **kw)
        warn_legacy("biocomp.plotutils.FigureSpec", "jeanplot.panels.figure.Figure")

    @property
    def output_path(self) -> Path:
        assert self.output_file is not None
        return Path(self.output_dir) / self.output_file

    def make_figure(self) -> FigAx:
        fax = self.layout.make_figure(**self.extra_args)
        return fax

    def save_figure(self, figax: FigAx) -> None:
        import json
        from datetime import datetime

        assert self.output_file is not None

        sanitized_metadata = sanitize_for_json(self.metadata)
        metadata_json = json.dumps(
            {FIGURE_METADATA_KEY: sanitized_metadata},
            separators=(",", ":"),
        )
        full_metadata = {
            "Creator": "biocomp",
            "Author": "biocomp",
            "Subject": metadata_json,
            "Title": self.title or "Biocomp Figure",
            "CreationDate": datetime.now().isoformat(),
        }

        figax.figure.canvas.draw()

        paths = [self.output_path] + [Path(p) for p in self.extra_output_paths]
        tight_bbox = None
        if len(paths) > 1:
            try:
                tight_bbox = self._compute_tight_bbox(figax.figure, pad=0.1)
            except Exception as e:
                logger.debug(f"tight bbox precompute failed: {e}; falling back to per-save")
        for path in paths:
            self._save_to_path(figax, path, full_metadata, bbox_inches=tight_bbox)

    @staticmethod
    def _compute_tight_bbox(figure, pad: float = 0.1):
        bbox = figure.get_tightbbox(figure._get_renderer())
        return bbox.padded(pad, pad) if pad else bbox

    def _save_to_path(
        self, figax: FigAx, output_path: Path, full_metadata: dict[str, Any],
        bbox_inches=None,
    ) -> None:
        import shutil
        import tempfile
        import time
        from datetime import datetime

        if bbox_inches is None:
            bbox_inches = "tight"

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

        if str(output_path).lower().endswith(".png"):
            figax.figure.savefig(
                temp_path,
                format="png",
                bbox_inches=bbox_inches,
                dpi=self.dpi,
                metadata={k: str(v) for k, v in full_metadata.items()},
            )
        elif str(output_path).lower().endswith(".pdf"):
            pdf_metadata = {**full_metadata, "CreationDate": datetime.now()}
            with mpl.rc_context({"pdf.compression": 1}):
                figax.figure.savefig(temp_path, metadata=pdf_metadata, bbox_inches=bbox_inches)
        elif str(output_path).lower().endswith(".svg"):
            figax.figure.savefig(temp_path, format="svg", bbox_inches=bbox_inches)
            self._postprocess_svg(temp_path, full_metadata)
        else:
            logger.warning(
                f"Saving figure to {output_path} in {output_path.suffix} format. "
                f"Only PNG, PDF, and SVG formats have full metadata support."
            )
            figax.figure.savefig(temp_path, metadata=full_metadata, bbox_inches=bbox_inches)

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

    def _postprocess_svg(self, svg_path: Path, full_metadata: dict[str, Any]) -> None:
        """Post-process SVG file to add custom metadata and attributes"""
        import re

        # Read the original SVG content
        with open(svg_path, encoding="utf-8") as f:
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
        if self.subtitle is not None:
            # Subtitle inherits horizontal placement from title (so they're
            # vertically stacked at the same x) but pins `va: top` so the
            # subtitle hangs *below* the title's anchor. The y offset
            # approximates the title's vertical footprint in figure-frac
            # coords using its fontsize and the figure height.
            tk = self.title_kwargs
            title_y = float(tk.get("y", 0.98))
            title_fs = float(tk.get("fontsize", 12))
            fig_h_in = float(figax.figure.get_size_inches()[1])
            # 1 pt = 1/72 in. Approx line height = 1.4 * fontsize.
            title_height_frac = (1.4 * title_fs) / 72.0 / max(fig_h_in, 1e-3)
            # If title's `va` puts the anchor at its bottom, the title
            # text rises above `title_y` and the subtitle must drop
            # *below* `title_y` (no extra offset). For other anchors,
            # offset down by the title's footprint.
            if tk.get("va", "top") == "bottom":
                default_sub_y = title_y - 0.005
            else:
                default_sub_y = title_y - title_height_frac
            sk = {
                "x": tk.get("x", 0.5),
                "y": default_sub_y,
                "ha": tk.get("ha", "center"),
                "va": "top",
                "fontweight": "normal",
                "fontsize": max(int(title_fs) - 2, 7),
                **self.subtitle_kwargs,
            }
            figax.figure.text(sk.pop("x"), sk.pop("y"), self.subtitle, **sk)
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
    input_order: Sequence[int] | Sequence[str] | None = None,
    protein_aliases: dict[str, str] | None = None,
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
    get_XY: Callable[[PlotData], tuple[NdArray, NdArray]],
    input_order: Sequence[int] | Sequence[str] | Literal["inv"] | None = None,
    protein_aliases: dict[str, str] | None = None,
    only_dependent_outputs: bool = True,
    **kw,
) -> LazyPlotData:
    input_order, output_pos, input_names, output_name = get_reordered_protein_names(
        network, input_order, protein_aliases, only_dependent_outputs
    )

    logger.debug(
        f"extract_lazy_plot_data: {network.name} input_order={input_order} inputs={input_names} output={output_name}"
    )

    def get_xy(pdata: PlotData) -> tuple[NdArray, NdArray]:
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


def diagonal_xy(X, angle_deg=45.0):
    X = np.asarray(X)
    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)
    return np.column_stack([c * X[:, 1] - s * X[:, 0], s * X[:, 1] + c * X[:, 0]])


def diagonal_xy_raw(X_lat, rescaler, angle_deg=45.0):
    X_raw = np.asarray(rescaler.inv(X_lat))
    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)
    s_raw = c * X_raw[:, 1] - s * X_raw[:, 0]
    t_raw = s * X_raw[:, 1] + c * X_raw[:, 0]
    return np.column_stack([rescaler.fwd(s_raw), rescaler.fwd(t_raw)])


def diagonal_slice_path_latent(t_raw, s_raw_arr, rescaler, angle_deg=45.0):
    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)
    s_arr = np.asarray(s_raw_arr)
    x1_raw = c * t_raw - s * s_arr
    x2_raw = c * s_arr + s * t_raw
    return np.asarray(rescaler.fwd(x1_raw)), np.asarray(rescaler.fwd(x2_raw))


def plot_diagonal_paths(ax, t_raw_values, s_raw_range, rescaler,
                        colors=None, n=400, angle_deg=45.0, line_props=None):
    s_arr = np.linspace(s_raw_range[0], s_raw_range[1], n)
    line_props = dict(line_props or {})
    for i, t_raw in enumerate(t_raw_values):
        x1_lat, x2_lat = diagonal_slice_path_latent(t_raw, s_arr, rescaler, angle_deg)
        kw = dict(line_props)
        if colors is not None:
            kw["color"] = colors[i]
        ax.plot(x1_lat, x2_lat, **kw)


_SLICE_AXES = ("x", "y", "s", "t")


def slice_panel_args(slice_axis, X_lat, rescaler, slice_values_raw, input_names=None):
    """Build (X, slices_latent, input_names) for `smooth_1d` panel given a slice mode.

    slice_axis: one of "x" | "y" | "s" | "t" -- the axis along which the 1D curve
                varies. The orthogonal axis is what's held fixed (sliced).
    X_lat:      (N, 2) latent-space inputs in native order [col 0, col 1].
    slice_values_raw: list of raw-fluo values to slice at, in the units of the
                      sliced (held-fixed) axis.
    input_names: optional [name_col0, name_col1] in native order (typically
                 [activator, inhibitor]). Used to build display labels.
    """
    if slice_axis not in _SLICE_AXES:
        raise ValueError(f"slice_axis must be one of {_SLICE_AXES}, got {slice_axis!r}")
    X_lat = np.asarray(X_lat)
    slices_latent = [[float(v)] for v in rescaler.fwd(np.asarray(slice_values_raw))]
    n0, n1 = (input_names or ["x", "y"])

    if slice_axis == "x":
        X = X_lat
        names = [n0, n1]
    elif slice_axis == "y":
        X = X_lat[:, [1, 0]]
        names = [n1, n0]
    elif slice_axis == "s":
        X = diagonal_xy_raw(X_lat, rescaler)
        names = [f"({n0} − {n1}) / √2", f"({n0} + {n1}) / √2"]
    else:  # "t"
        X = diagonal_xy_raw(X_lat, rescaler)[:, [1, 0]]
        names = [f"({n0} + {n1}) / √2", f"({n0} − {n1}) / √2"]

    return {"X": X, "slices_latent": slices_latent, "input_names": names}


def plot_slice_overlay(ax, slice_axis, slice_values_raw, rescaler,
                       var_range_raw=None, colors=None, n=400, line_props=None):
    """Overlay 1D-slice paths on a 2D heatmap (latent display coords)."""
    if slice_axis not in _SLICE_AXES:
        raise ValueError(f"slice_axis must be one of {_SLICE_AXES}, got {slice_axis!r}")
    line_props = dict(line_props or {})

    if slice_axis in ("x", "y"):
        slice_lat = rescaler.fwd(np.asarray(slice_values_raw))
        draw = ax.axhline if slice_axis == "x" else ax.axvline
        for i, v in enumerate(slice_lat):
            kw = dict(line_props)
            if colors is not None:
                kw["color"] = colors[i]
            draw(float(v), **kw)
        return

    if var_range_raw is None:
        raise ValueError(f"var_range_raw required for slice_axis={slice_axis!r}")
    cos45 = sin45 = np.cos(np.deg2rad(45.0))
    var_arr = np.linspace(var_range_raw[0], var_range_raw[1], n)
    for i, fixed_raw in enumerate(slice_values_raw):
        if slice_axis == "s":
            x1_raw = cos45 * fixed_raw - sin45 * var_arr
            x2_raw = cos45 * var_arr + sin45 * fixed_raw
        else:  # "t"
            x1_raw = cos45 * var_arr - sin45 * fixed_raw
            x2_raw = cos45 * fixed_raw + sin45 * var_arr
        x1_lat = rescaler.fwd(x1_raw)
        x2_lat = rescaler.fwd(x2_raw)
        kw = dict(line_props)
        if colors is not None:
            kw["color"] = colors[i]
        ax.plot(x1_lat, x2_lat, **kw)


def plot_slice_chords(ax, X, Y, slices, xlims, rescaler=None, colors=None,
                      knn_stats_params=None, res=100, n_curve=200,
                      chord_props=None, **_kw):
    """Draw a `linear-in-raw-fluo` reference curve per slice.

    The reference is the straight line in RAW fluo space connecting each
    smoothed slice's leftmost / rightmost finite endpoints, mapped back
    to the latent display via `rescaler.fwd`. Curved on the heatmap by
    construction — that curvature is the log warp, not the biology.
    Deviation between the actual smoothed slice and this reference is
    nonlinearity in the molecular mechanism's native units.

    Falls back to a straight latent-space chord when `rescaler is None`.
    """
    from biocomp.plotting.plotting_core import knn_stats, build_tree

    X = np.asarray(X)
    Y = np.asarray(Y)
    slices = np.asarray(slices)
    knn_stats_params = dict(knn_stats_params or {})
    knn_radius = float(knn_stats_params.get("radius", 0.075))

    xmin = float(X[:, 0].min() if xlims[0] is None else xlims[0])
    xmax = float(X[:, 0].max() if xlims[1] is None else xlims[1])
    xquery_min = max(xmin, float(X[:, 0].min()) + knn_radius * 0.5)
    xquery_max = min(xmax, float(X[:, 0].max()) - knn_radius)
    xq = np.linspace(xquery_min, xquery_max, res)

    tree = build_tree(X)
    nslices = slices.shape[0]
    n_input = X.shape[1]
    chord_props = dict(chord_props or {})

    for i in range(nslices):
        query = xq.reshape(-1, 1)
        if n_input > 1:
            query = np.hstack([query, np.tile(slices[i], (query.shape[0], 1))])
        knn_mean = np.asarray(knn_stats(query, Y, tree=tree, stats=["mean"],
                                        **knn_stats_params)).reshape(-1)
        finite = np.isfinite(knn_mean)
        if not finite.any():
            continue
        idx = np.where(finite)[0]
        x_lo_lat, x_hi_lat = float(xq[idx[0]]), float(xq[idx[-1]])
        y_lo_lat, y_hi_lat = float(knn_mean[idx[0]]), float(knn_mean[idx[-1]])

        kw = dict(chord_props)
        if colors is not None:
            kw["color"] = colors[i]

        if rescaler is None:
            ax.plot([x_lo_lat, x_hi_lat], [y_lo_lat, y_hi_lat], **kw)
            continue

        x_lo_raw, x_hi_raw = float(rescaler.inv(x_lo_lat)), float(rescaler.inv(x_hi_lat))
        y_lo_raw, y_hi_raw = float(rescaler.inv(y_lo_lat)), float(rescaler.inv(y_hi_lat))
        x_raw = np.linspace(x_lo_raw, x_hi_raw, n_curve)
        t = (x_raw - x_lo_raw) / (x_hi_raw - x_lo_raw)
        y_raw = y_lo_raw + t * (y_hi_raw - y_lo_raw)
        ax.plot(rescaler.fwd(x_raw), rescaler.fwd(y_raw), **kw)


def plot_addition_vs_removal_overlay(
    ax,
    X_lat,
    Y_lat,
    slice_values_raw,
    anchor_raw_values,
    rescaler,
    colors=None,
    knn_stats_params=None,
    max_centroid_offset_frac=0.0,
    line_props=None,
    res=200,
    **_kw,
):
    """Compare 'add inhibitor' vs 'remove activator' on the x-mode slice plot.

    Slice plot's x-axis is the inhibitor (X_lat col 0); each slice is at a
    fixed activator level (X_lat col 1). For each (anchor x1, slice x2)
    pair, draws a comparison curve emerging from the anchor point
    (x1 = anchor, x2 = slice_value) and extending rightward as x2 sweeps
    from slice_value down to 0. Shifts are in LATENT units so the comparison
    is a clean translation of y_lat(x2_lat) at fixed x1 — preserving the
    floor / transition / saturation shape of the surface. One latent step
    = one fold change of either knob (matches the original slice's
    parameterization, which is also linear in latent).

    Smoothing matches `smooth_1d` (same tree, kernel, and centroid-offset
    boundary filter). `colors`, if provided, has length
    len(anchor_raw_values): one color per anchor, shared across slices.
    """
    from biocomp.plotting.plotting_core import knn_stats, build_tree

    X_lat = np.asarray(X_lat)
    Y_lat = np.asarray(Y_lat)
    anchor_raw_values = list(anchor_raw_values)
    if not anchor_raw_values:
        return

    knn_stats_params = dict(knn_stats_params or {})
    knn_stats_params.pop("avg_method", None)
    knn_radius = float(knn_stats_params.get("radius", 0.075))
    knn_stats_params["radius"] = knn_radius
    sigma_in_radius = float(knn_stats_params.get("sigma_in_radius", 3.0))
    offset_cutoff = (
        max_centroid_offset_frac * (knn_radius / sigma_in_radius)
        if max_centroid_offset_frac > 0.0
        else None
    )

    line_props = dict(line_props or {})
    tree = build_tree(X_lat)

    for a, anchor_raw in enumerate(anchor_raw_values):
        anchor_lat = float(rescaler.fwd(float(anchor_raw)))
        kw_base = dict(line_props)
        if colors is not None:
            kw_base["color"] = colors[a]
        for slice_raw in slice_values_raw:
            slice_lat = float(rescaler.fwd(float(slice_raw)))
            delta_lat = np.linspace(0.0, slice_lat, res)
            x2_lat = slice_lat - delta_lat
            plot_x_lat = anchor_lat + delta_lat
            query = np.column_stack([np.full(res, anchor_lat), x2_lat])

            requested = ["mean", "variance"]
            if offset_cutoff is not None:
                requested.append("centroid_offset")
            knn_result = knn_stats(
                query, Y_lat, tree=tree, stats=requested, **knn_stats_params,
            )
            if offset_cutoff is not None:
                knn_mean, _knn_var, knn_offset = knn_result
                boundary = np.asarray(knn_offset) > offset_cutoff
                y_lat = np.where(boundary, np.nan, np.asarray(knn_mean).reshape(-1))
            else:
                knn_mean, _knn_var = knn_result
                y_lat = np.asarray(knn_mean).reshape(-1)

            ax.plot(plot_x_lat, y_lat, **kw_base)


class IdentityRescaler:
    def fwd(self, x):
        return np.asarray(x)

    def inv(self, x):
        return np.asarray(x)


IDENTITY_RESCALER = IdentityRescaler()


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
from .plotting.plotting_smooth import (  # noqa: E402
    smooth_2d, smooth_1d, smooth_grad_magnitude_2d, gradient_field_2d,
)
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
    rescaler: DataRescaler | None = None,
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
    force_dim: int | None = None,
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
    force_dim: int | None = None,
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
    title: str | None = None,
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
