import numpy as np
from numpy.typing import NDArray
import re
from matplotlib.colors import to_rgb
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from svgelements import SVG, Shape, Group

import jax
import jax.numpy as jnp

from biocomp.logging_config import get_logger
from biocomp.plotting.ascii_heatmap import CMAP_S

if TYPE_CHECKING:
    from biocomp.compute import ComputeStack
    from biocomp.design import DesignManager
    from biocomp.design_targets import Target
    from biocomp.designloss import GridLossResult
    from biocomp.network import Network
    from biocomp.parameters import ParameterTree
    from biocomptools.modelmodel import BiocompModel
    from biocomptools.toollib.networkprediction import PlotData

NdArray = np.ndarray
logger = get_logger(__name__)


def build_design_stack(
    dmanager: "DesignManager",
    model: "BiocompModel",
    *,
    unlock_ratios: bool = True,
    use_latent_ratios: bool = False,
    latent_dim: int = 0,
    latent_hidden_dim: int = 0,
    auto_lock_topology_tus: bool = True,
) -> "ComputeStack":
    """Single source of truth for design stack construction.

    All design-related stack building should use this helper to ensure
    consistent TU masking behavior across training, logging, and commit.
    """
    return dmanager.build_stack(
        model,
        unlock_ratios=unlock_ratios,
        use_latent_ratios=use_latent_ratios,
        latent_dim=latent_dim,
        latent_hidden_dim=latent_hidden_dim,
        auto_lock_topology_tus=auto_lock_topology_tus,
    )


def predict_design_grid(
    model: "BiocompModel",
    networks: list["Network"],
    target: "Target",
    resolution: tuple[int, int],
    seed: int = 0,
) -> tuple[list["PlotData"], np.ndarray]:
    """Single source of truth for design grid prediction.

    Ensures consistent prediction flags:
    - already_latent=True (design targets are in latent space)
    - z_value=0.0 (deterministic for reproducibility)
    - disable_variational=True
    - skip_input_reorder=True (design uses positional inputs)

    Returns:
        (data_list, Y_target) - prediction data and target grid
    """
    from biocomptools.modelmodel import NetworkModel
    from biocomptools.toollib.networkprediction import NetworkPrediction

    X_lat, Y_target = target.get_lattice(resolution=resolution, seed=seed)
    nm = NetworkModel(model=model, network=networks)

    pred = NetworkPrediction(
        predict_at=[X_lat] * len(networks),
        network_model=nm,
        already_latent=True,
        z_value=0.0,
        disable_variational=True,
        skip_input_reorder=True,
        seed=seed,
    )
    return pred.get_data(rescale_latent=False), Y_target


def _parse_svg_path(d):
    from svgpath2mpl import parse_path

    mpath = parse_path(d)
    return [tuple(pt) for pt in mpath.vertices]


def _greyscale(fill, max_is_black):
    try:
        r, g, b = to_rgb(fill)
    except ValueError:
        m = re.match(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", fill)
        if m:
            r, g, b = [int(x) / 255 for x in m.groups()]
        else:
            return 0.0 if max_is_black else 1.0
    grey = (r + g + b) / 3.0
    return 1.0 - grey if max_is_black else grey


def _extract_shapes_from_svg(svg_path, max_is_black):
    """Extract shapes from SVG using svgelements (handles all transforms automatically)."""
    from svgpath2mpl import parse_path

    svg = SVG.parse(str(svg_path))

    vx = svg.viewbox.x if svg.viewbox else 0
    vy = svg.viewbox.y if svg.viewbox else 0
    # Use svgelements' computed width/height which includes unit conversion (pt→px).
    # This ensures path coordinates and sample point coordinates are in the same space.
    # Without this, SVGs with pt units (like matplotlib exports) have mismatched coordinates.
    vw = svg.width if svg.width else (svg.viewbox.width if svg.viewbox else 100)
    vh = svg.height if svg.height else (svg.viewbox.height if svg.viewbox else 100)

    masked_elements = set()
    for el in svg.elements():
        if isinstance(el, Group) and el.values.get("mask"):
            for child in el:
                if isinstance(child, Shape):
                    masked_elements.add(id(child))

    paths, greys = [], []
    for el in svg.elements():
        if not isinstance(el, Shape):
            continue
        if masked_elements and id(el) not in masked_elements:
            continue

        fill = el.fill
        if fill is None or str(fill).lower() == "none":
            continue

        path_d = el.d()
        if not path_d:
            continue

        mpath = parse_path(path_d)
        grey = _greyscale(str(fill), max_is_black)
        paths.append(mpath)
        greys.append(grey)

    return paths, np.asarray(greys), (vx, vy, vw, vh)


def _generate_svg_sample_points(
    n, viewbox, latent, vx, vy, vw, vh, rng, log, grid=None, grid_jitter_std=None
):
    """Generate sample points in SVG coordinate space and return latent coordinates.

    Args:
        viewbox: (viewbox_x, viewbox_y) - fraction of SVG to sample (0-1 normalized)
        latent: (latent_x, latent_y) - output latent coordinate ranges
    """
    viewbox_x, viewbox_y = viewbox
    latent_x, latent_y = latent

    if grid:
        xres, yres = grid
        if log:
            eps = 1e-6
            x_vals = (
                np.logspace(np.log10(eps + viewbox_x[0] * vw), np.log10(viewbox_x[1] * vw), xres)
                + vx
            )
            y_vals = (
                vh
                - np.logspace(np.log10(eps + viewbox_y[0] * vh), np.log10(viewbox_y[1] * vh), yres)
                + vy
            )
        else:
            # Sample at cell centers (half-step inset) to avoid boundary detection issues
            # with contains_point() which can miss points exactly on rect edges
            x_step = (viewbox_x[1] - viewbox_x[0]) * vw / xres
            y_step = (viewbox_y[1] - viewbox_y[0]) * vh / yres
            x_start = viewbox_x[0] * vw + vx + x_step / 2
            x_end = viewbox_x[1] * vw + vx - x_step / 2
            y_start = (1 - viewbox_y[0]) * vh + vy - y_step / 2
            y_end = (1 - viewbox_y[1]) * vh + vy + y_step / 2
            x_vals = np.linspace(x_start, x_end, xres)
            y_vals = np.linspace(y_start, y_end, yres)

        sx_grid, sy_grid = np.meshgrid(x_vals, y_vals)
        sx_list, sy_list = [], []

        for _ in range(n):
            sx_sample, sy_sample = sx_grid.copy(), sy_grid.copy()
            if grid_jitter_std and grid_jitter_std > 0:
                x_spacing = (x_vals[-1] - x_vals[0]) / (xres - 1) if xres > 1 else 0
                y_spacing = (y_vals[-1] - y_vals[0]) / (yres - 1) if yres > 1 else 0
                x_spacing = abs(x_spacing)
                y_spacing = abs(y_spacing)
                sx_sample += rng.normal(0, grid_jitter_std * x_spacing, sx_sample.shape)
                sy_sample += rng.normal(0, grid_jitter_std * y_spacing, sy_sample.shape)
            sx_list.append(sx_sample.flatten())
            sy_list.append(sy_sample.flatten())

        sx, sy = np.concatenate(sx_list), np.concatenate(sy_list)
    else:
        if log:
            eps = 1e-6
            sx = (
                10 ** rng.uniform(np.log10(eps + viewbox_x[0] * vw), np.log10(viewbox_x[1] * vw), n)
                + vx
            )
            sy = (
                vh
                - 10
                ** rng.uniform(np.log10(eps + viewbox_y[0] * vh), np.log10(viewbox_y[1] * vh), n)
                + vy
            )
        else:
            sx = rng.uniform(viewbox_x[0] * vw + vx, viewbox_x[1] * vw + vx, n)
            sy = rng.uniform((1 - viewbox_y[1]) * vh + vy, (1 - viewbox_y[0]) * vh + vy, n)

    # Convert SVG coordinates to latent space
    # KEY: Normalize against the VIEWBOX range, so cropped region maps to FULL latent range
    if log:
        eps = 1e-6
        x_log = np.log10(sx - vx)
        y_log = np.log10(vh - (sy - vy))
        x_log_min = np.log10(eps + viewbox_x[0] * vw)
        x_log_max = np.log10(viewbox_x[1] * vw)
        y_log_min = np.log10(eps + viewbox_y[0] * vh)
        y_log_max = np.log10(viewbox_y[1] * vh)
        x_norm = (x_log - x_log_min) / (x_log_max - x_log_min + eps)
        y_norm = (y_log - y_log_min) / (y_log_max - y_log_min + eps)
    else:
        # Normalize against viewbox, not full SVG
        svg_x_min = viewbox_x[0] * vw + vx
        svg_x_max = viewbox_x[1] * vw + vx
        svg_y_min = (1 - viewbox_y[1]) * vh + vy
        svg_y_max = (1 - viewbox_y[0]) * vh + vy
        x_norm = (sx - svg_x_min) / (svg_x_max - svg_x_min) if svg_x_max != svg_x_min else 0.5
        y_norm = (
            1.0 - ((sy - svg_y_min) / (svg_y_max - svg_y_min)) if svg_y_max != svg_y_min else 0.5
        )

    x_latent = x_norm * (latent_x[1] - latent_x[0]) + latent_x[0]
    y_latent = y_norm * (latent_y[1] - latent_y[0]) + latent_y[0]
    X = np.column_stack((x_latent, y_latent))

    return X, sx, sy


def _assign_greyscale_values(sx, sy, paths, greys, max_is_black, outlim, grid_shape=None):
    default_background = 0.0 if max_is_black else 1.0
    Y = np.full(len(sx), default_background)
    pts = np.column_stack((sx, sy))
    if len(paths) != len(greys):
        return Y if grid_shape is None else Y.reshape(grid_shape)
    for p, g in zip(paths, greys, strict=False):
        Y[p.contains_points(pts)] = g
    # Map grayscale [0,1] to latent output range
    Y = Y * (outlim[1] - outlim[0]) + outlim[0]
    if grid_shape:
        n, yres, xres = grid_shape
        Y = Y.reshape(n, yres, xres)
    return Y


def sample_from_svg(
    svg_path,
    n=None,
    *,
    seed=None,
    log=False,
    max_is_black=True,
    grid=None,
    grid_jitter_std=None,
    # New API parameters
    viewbox_x: tuple[float, float] = None,
    viewbox_y: tuple[float, float] = None,
    latent_x: tuple[float, float] = (0.0, 0.6),
    latent_y: tuple[float, float] = (0.0, 0.6),
    latent_out: tuple[float, float] = (0.0, 0.6),
    # Legacy parameters (deprecated)
    rescale_to=None,
    xlim=None,
    ylim=None,
    outlim=None,
    lattice_x_extent=None,
    lattice_y_extent=None,
    img_latent_xlim=None,
    img_latent_ylim=None,
    img_latent_outlim=None,
):
    """Sample points from an SVG file and return latent coordinates + values.

    Args:
        viewbox_x, viewbox_y: Fraction of SVG to sample (0-1 normalized). Default (0,1) = full image.
        latent_x, latent_y: Output coordinate range. The viewbox region maps to this full range.
        latent_out: Output value range. Grayscale 0-1 maps to this range.

    The key semantic: viewbox crops the SVG, latent defines the output coordinate system.
    A cropped viewbox still maps to the FULL latent range.
    """
    import warnings

    svg_path = Path(svg_path).expanduser().resolve()
    seed = seed or np.random.randint(0, 2**32 - 1)

    # Handle legacy parameters
    legacy_used = any(
        p is not None
        for p in [
            xlim,
            ylim,
            outlim,
            rescale_to,
            lattice_x_extent,
            lattice_y_extent,
            img_latent_xlim,
            img_latent_ylim,
            img_latent_outlim,
        ]
    )
    if legacy_used:
        warnings.warn(
            "sample_from_svg: legacy parameters are deprecated. "
            "Use viewbox_x/y and latent_x/y/out instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        viewbox_x = viewbox_x or lattice_x_extent or xlim
        viewbox_y = viewbox_y or lattice_y_extent or ylim
        latent_x = img_latent_xlim or latent_x
        latent_y = img_latent_ylim or latent_y
        latent_out = img_latent_outlim or outlim or latent_out
        if rescale_to:
            viewbox_x = viewbox_x or tuple(rescale_to.get("x", (0, 1)))
            viewbox_y = viewbox_y or tuple(rescale_to.get("y", (0, 1)))
            latent_out = tuple(rescale_to.get("out", latent_out))

    # Apply defaults for viewbox
    default_viewbox = (0.1, 1.0) if log else (0.0, 1.0)
    viewbox_x = viewbox_x or default_viewbox
    viewbox_y = viewbox_y or default_viewbox

    if grid:
        n = n or 1
        xres, yres = grid
        grid_shape = (n, yres, xres)
    elif n is None:
        raise ValueError("n must be specified when not using grid sampling")
    else:
        grid_shape = None

    rng = np.random.default_rng(seed)
    paths, greys, (vx, vy, vw, vh) = _extract_shapes_from_svg(svg_path, max_is_black)

    X, sx, sy = _generate_svg_sample_points(
        n,
        (viewbox_x, viewbox_y),
        (latent_x, latent_y),
        vx,
        vy,
        vw,
        vh,
        rng,
        log,
        grid,
        grid_jitter_std,
    )
    Y = _assign_greyscale_values(sx, sy, paths, greys, max_is_black, latent_out, grid_shape)

    return (X, Y) if grid else (X, Y[:, None])


def sample_from_data(
    X: NdArray,  # input data, already latent-space rescaled (n_samples, n_dims)
    Y: NdArray,  # corresponding outputs (n_samples,) or (n_samples, 1)
    n: int = 1,  # how many grids to generate. if grid_jitter_std is 0, they will be duplicates
    zslice: Optional[NdArray] = None,  # for >2D data, slice value(s) for dims beyond 2
    xlims: Optional[tuple[float, float]] = None,
    ylims: Optional[tuple[float, float]] = None,
    vlims: Optional[
        tuple[float, float]
    ] = None,  # output value clipping (not used in sampling, but returned for reference)
    sampling_grid: tuple[int, int] = (48, 48),
    grid_jitter_std: float = 0.0,
    k: int = 128,
    min_points: int = 20,
) -> tuple[NDArray, NDArray]:
    """Sample from experimental data by KNN interpolation onto a regular grid.

    Similar to knn_grid in plotting_smooth.py, but designed for design mode.

    Args:
        X: Input coordinates (n_samples, n_dims), already in latent space
        Y: Output values (n_samples,) or (n_samples, 1)
        n: Number of grids to generate (useful for jittered sampling)
        zslice: For 3D+ data, the z-coordinate to slice at
        xlims: X-axis limits (min, max). If None, derived from data.
        ylims: Y-axis limits (min, max). If None, derived from data.
        vlims: Output value limits (not used for clipping, just metadata)
        sampling_grid: (xres, yres) resolution of the output grid
        grid_jitter_std: Std of jitter to add to grid points (as fraction of grid spacing)
        k: Number of nearest neighbors for KNN interpolation
        min_points: Minimum number of valid points in neighborhood

    Returns:
        X_grid: Grid coordinates (n * xres * yres, 2)
        Y_grid: Interpolated output values (n, yres, xres)
    """
    from biocomp.plotting.plotting_core import knn_stats, build_tree
    from biocomp.plotutils import make_xy_grid

    X = np.asarray(X)
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y[:, None]  # ensure Y is (n, 1)

    # filter out nan/inf values
    mask = np.all(np.isfinite(X), axis=1)
    mask = mask & np.all(np.isfinite(Y), axis=1)
    X_clean = X[mask]
    Y_clean = Y[mask]

    if len(X_clean) == 0:
        raise ValueError("No finite data points available for sampling")

    n_dims = X_clean.shape[1]

    # derive limits from data if not provided
    if xlims is None:
        xlims = (float(X_clean[:, 0].min()), float(X_clean[:, 0].max()))
    if ylims is None:
        ylims = (float(X_clean[:, 1].min()), float(X_clean[:, 1].max()))

    xres, yres = sampling_grid
    xmin, xmax = xlims
    ymin, ymax = ylims

    # build base grid
    xy_base = make_xy_grid(xmin, xmax, ymin=ymin, ymax=ymax, xres=xres, yres=yres)

    # handle higher dimensions (>2D) by appending z-slice
    if n_dims > 2:
        if zslice is None:
            raise ValueError(f"Data has {n_dims} dimensions but no zslice provided")
        zslice = np.atleast_1d(zslice)
        if zslice.shape[0] != n_dims - 2:
            raise ValueError(f"zslice must have {n_dims - 2} elements, got {zslice.shape[0]}")
        # we'll append zslice to each query point
        n_extra_dims = n_dims - 2
    else:
        n_extra_dims = 0
        zslice = None

    # build tree on the full coordinate space
    tree = build_tree(X_clean)

    all_xgrids = []
    all_ygrids = []

    rng = np.random.default_rng()

    for _i in range(n):
        xy_query = xy_base.copy()

        # add jitter
        if grid_jitter_std > 0:
            x_spacing = (xmax - xmin) / (xres - 1) if xres > 1 else 0
            y_spacing = (ymax - ymin) / (yres - 1) if yres > 1 else 0
            jitter_x = rng.normal(0, grid_jitter_std * x_spacing, xy_query.shape[0])
            jitter_y = rng.normal(0, grid_jitter_std * y_spacing, xy_query.shape[0])
            xy_query[:, 0] += jitter_x
            xy_query[:, 1] += jitter_y

        # append z-slice coordinates if needed
        if n_extra_dims > 0:
            z_tile = np.tile(zslice, (xy_query.shape[0], 1))
            xquery_full = np.hstack([xy_query, z_tile])
        else:
            xquery_full = xy_query

        # get KNN interpolated values
        output_values = knn_stats(
            xquery_full, Y_clean, tree=tree, stats="mean", k=k, min_points=min_points
        )
        output_values = np.asarray(output_values).squeeze()

        all_xgrids.append(xy_query)
        # reshape output to (yres, xres) for grid format
        all_ygrids.append(output_values.reshape(yres, xres))

    # stack results
    X_grid = np.concatenate(all_xgrids, axis=0)
    Y_grid = np.stack(all_ygrids, axis=0)  # (n, yres, xres)

    return X_grid, Y_grid


def data_to_lattice_2d(
    X: NdArray,
    Y: NdArray,
    xlims: Optional[tuple[float, float]] = None,
    ylims: Optional[tuple[float, float]] = None,
    resolution: tuple[int, int] = (48, 48),
    k: int = 128,
    min_points: int = 20,
) -> tuple[NDArray, NDArray]:
    """Convert scattered 2D data to a regular lattice using KNN interpolation.

    Convenience wrapper around sample_from_data for single-grid sampling.

    Returns:
        X_lattice: Grid coordinates (xres * yres, 2)
        Y_lattice: Interpolated values (yres, xres)
    """
    X_grid, Y_grid = sample_from_data(
        X, Y, n=1, xlims=xlims, ylims=ylims, sampling_grid=resolution, k=k, min_points=min_points
    )
    return X_grid, Y_grid[0]


@dataclass
class RecipeEvaluationResult:
    """Result of evaluating a recipe against a target."""

    prediction: np.ndarray
    target: np.ndarray
    X: np.ndarray
    grid_resolution: tuple[int, int]
    sublosses: "GridLossResult"
    txt_plot: str
    params: "ParameterTree"
    recipe: Any
    network: Any
    stack: "ComputeStack"

    def prediction_grid(self) -> np.ndarray:
        return self.prediction.reshape(self.grid_resolution[1], self.grid_resolution[0])

    def target_grid(self) -> np.ndarray:
        return self.target.reshape(self.grid_resolution[1], self.grid_resolution[0])


def load_recipe(recipe_path):
    """Load a recipe from a YAML file."""
    import dracon as dr

    config = dr.load(str(recipe_path))
    return config["recipe"]


def load_target(target_path):
    """Load a target from a YAML file."""
    import os
    import dracon as dr
    from biocomp.design_targets import (
        SVGTarget,
        Target,
        DataTarget,
        LatticeSampling,
        UniformSampling,
    )

    config = dr.load(
        str(target_path),
        context={
            "SVGTarget": SVGTarget,
            "Target": Target,
            "DataTarget": DataTarget,
            "LatticeSampling": LatticeSampling,
            "UniformSampling": UniformSampling,
            "BIOCOMP_ROOT": os.environ.get("BIOCOMP_ROOT", ""),
        },
    )
    if isinstance(config, list):
        return config[0]
    elif isinstance(config, dict) and "targets" in config:
        return config["targets"][0]
    return config


def build_network_from_recipe(recipe, invert: bool = True):
    """Build a network from a recipe."""
    from biocomp.network import recipe_to_networks
    import biocomp.biorules as br

    networks = recipe_to_networks(recipe, br.ALL_RULES, invert=invert, inversion_mode="main")
    assert len(networks) == 1, f"Expected 1 network, got {len(networks)}"
    return networks[0]


def init_params_for_network(
    model,
    network,
    stack=None,
    key=None,
):
    """Initialize parameters for a network using the model's compute config.

    Args:
        model: BiocompModel to use
        network: Network to initialize
        stack: Optional pre-built ComputeStack
        key: Optional random key

    Returns:
        (params, stack) tuple
    """
    from biocomp.compute import ComputeStack
    from biocomp.parameters import ParameterTree

    if stack is None:
        stack = ComputeStack(networks=[network])
        stack.build(model.compute_config, enable_tu_masking=False)

    if key is None:
        key = jax.random.PRNGKey(42)

    init_params = stack.init(key)
    _, nonshared = init_params.filter_by_tag(["shared"])
    params = ParameterTree.merge(model.shared_params, nonshared)

    return params, stack


def generate_lattice_prediction(
    params,
    stack,
    X_latent: np.ndarray,
    key=None,
) -> np.ndarray:
    """Generate predictions on a lattice grid.

    Args:
        params: ParameterTree with model parameters
        stack: Built ComputeStack
        X_latent: Lattice grid coordinates (n_points, n_dims)
        key: Optional random key

    Returns:
        Predictions array (n_points,)
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    batch_size = X_latent.shape[0]

    # Get num_z from params if available, else use 0 (deterministic)
    num_z_path = "global/number_of_random_variables"
    if num_z_path in params:
        val = params[num_z_path]
        num_z = int(val.ravel()[0]) if hasattr(val, "ravel") else int(val)
    else:
        num_z = 0
    Z = jnp.zeros((batch_size, num_z))
    keys = jax.random.split(key, batch_size)

    def apply_batch(params, x_batch, z_batch, keys):
        def apply_single(x, z, k):
            return stack.apply(params, x, z, k)[0]

        return jax.vmap(apply_single)(x_batch, z_batch, keys)

    # JIT the batched apply for proper JAX tracing
    apply_jit = jax.jit(apply_batch)
    yhat = apply_jit(params, X_latent, Z, keys)

    dep_mask = stack.get_dependent_output_mask()
    yhat_dep = np.asarray(jnp.compress(dep_mask, yhat, axis=-1))

    if yhat_dep.shape[-1] > 1:
        yhat_dep = yhat_dep.mean(axis=-1, keepdims=True)

    return yhat_dep.squeeze(-1)


def evaluate_recipe_with_sublosses(
    recipe_path,
    target_path,
    model,
    resolution: tuple[int, int] = (48, 48),
    key=None,
    params=None,
    stack=None,
) -> RecipeEvaluationResult:
    """Evaluate a recipe against a target and return comprehensive results.

    Args:
        recipe_path: Path to the recipe YAML file
        target_path: Path to the target YAML file
        model: BiocompModel to use for prediction
        resolution: Grid resolution for evaluation
        key: Optional random key
        params: Optional pre-initialized parameters (for post-optimization evaluation)
        stack: Optional pre-built ComputeStack

    Returns:
        RecipeEvaluationResult with prediction, target, sublosses, and txt-plot
    """
    from biocomp.designloss import compute_grid_losses
    from biocomp.plotting.plotting_txt import smooth_2d_txt

    if key is None:
        key = jax.random.PRNGKey(42)

    recipe = load_recipe(recipe_path)
    target = load_target(target_path)

    network = build_network_from_recipe(recipe)

    if params is None or stack is None:
        params, stack = init_params_for_network(model, network, stack, key)

    X_latent, Y_target_grid = target.get_lattice(resolution, seed=0)
    Y_target = Y_target_grid.flatten()

    prediction = generate_lattice_prediction(params, stack, X_latent, key)

    Y_pred_grid = prediction.reshape(resolution[1], resolution[0])
    Y_target_2d = Y_target.reshape(resolution[1], resolution[0])

    sublosses = compute_grid_losses(
        jnp.array(Y_pred_grid),
        jnp.array(Y_target_2d),
        w_sinkhorn=1.0,
        w_lncc=0.5,
        w_mse=0.0,
        w_rmse=0.5,
        return_contributions=False,
    )

    txt_result = smooth_2d_txt(
        X_latent,
        prediction.reshape(-1, 1),
        input_names=["x1", "x2"],
        output_name="y",
        title=f"Prediction: {recipe.name}",
        xres=48,
        yres=24,
    )

    return RecipeEvaluationResult(
        prediction=prediction,
        target=Y_target,
        X=X_latent,
        grid_resolution=resolution,
        sublosses=sublosses,
        txt_plot=str(txt_result),
        params=params,
        recipe=recipe,
        network=network,
        stack=stack,
    )


def get_committed_values(params, stack) -> dict:
    """Extract committed values (ratios, bias, parts) from params and stack.

    Returns dict with:
        - ratios: list of (node_name, ratio_values) tuples
        - bias: list of (node_name, path_suffix, bias_value) tuples
    """
    result = {"ratios": [], "bias": []}

    for layer_idx, layer in enumerate(stack.layers):
        ns = stack.get_layer_namespace(layer_idx)

        if "aggregation" in ns and "inv" not in ns:
            ratio_path = f"{ns}/ratios"
            if ratio_path in params:
                ratios = np.asarray(params[ratio_path])
                for node_idx in range(ratios.shape[0]):
                    node = layer.nodes[node_idx]
                    node_data = node.get(stack)
                    node_name = getattr(node_data, "name", f"node_{node_idx}")
                    result["ratios"].append((node_name, ratios[node_idx].tolist()))

        if "bias" in ns or "hard_bias" in ns:
            for path_suffix in ["raw_value", "scale", "value"]:
                bias_path = f"{ns}/{path_suffix}"
                if bias_path in params:
                    val = params[bias_path]
                    if hasattr(val, "shape") and val.size > 1:
                        bias_val = np.asarray(val).tolist()
                    else:
                        bias_val = float(np.asarray(val).ravel()[0])
                    result["bias"].append((ns, path_suffix, bias_val))

    return result


def compare_evaluation_results(
    before: RecipeEvaluationResult,
    after: RecipeEvaluationResult,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Compare two evaluation results and return validation checks.

    Returns dict with boolean results for each check:
        - predictions_match: Are predictions within tolerance?
        - sublosses_match: Do all sublosses match?
        - txt_plots_match: Are txt-plots string-identical?
    """
    pred_close = np.allclose(before.prediction, after.prediction, rtol=rtol, atol=atol)

    before_sl = before.sublosses.to_dict()
    after_sl = after.sublosses.to_dict()
    sl_match = all(abs(before_sl[k] - after_sl[k]) < atol for k in before_sl)

    txt_match = before.txt_plot == after.txt_plot

    return {
        "predictions_match": pred_close,
        "sublosses_match": sl_match,
        "txt_plots_match": txt_match,
        "max_prediction_diff": float(np.max(np.abs(before.prediction - after.prediction))),
        "subloss_diffs": {k: abs(before_sl[k] - after_sl[k]) for k in before_sl},
    }


LOSS_ORDER = [
    "sinkhorn",
    "lncc",
    "rmse",
    "mse",
    "simse",
    "spectral",
    "gradient",
    "contrast",
    "zncc",
]
DEFAULT_LOSS_WEIGHTS = {
    "sinkhorn": 1.0,
    "lncc": 0.5,
    "rmse": 0.5,
    "mse": 0.0,
    "simse": 0.0,
    "spectral": 0.0,
    "gradient": 0.0,
    "contrast": 0.0,
    "zncc": 0.0,
}


def compute_grid_metrics(
    target_grid: np.ndarray,
    prediction_grid: np.ndarray,
    loss_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute comprehensive grid-based metrics for target vs prediction comparison.

    Args:
        target_grid: 2D array (H, W) of target pattern
        prediction_grid: 2D array (H, W) of predicted pattern
        loss_weights: Optional dict of loss weights {loss_name: weight}.
            If None, uses DEFAULT_LOSS_WEIGHTS.

    Returns:
        Dict containing:
            - Individual loss values (sinkhorn, lncc, rmse, mse, etc.)
            - Weighted versions of each loss ({name}_weighted)
            - weighted_total: sum of weighted losses
            - correlation: Pearson correlation coefficient
            - pred_range: (min, max) of prediction
            - target_range: (min, max) of target
            - weights: the loss weights used
    """
    from biocomp.designloss import compute_grid_losses

    assert target_grid.ndim == 2, f"target_grid must be 2D, got {target_grid.ndim}D"
    assert prediction_grid.ndim == 2, f"prediction_grid must be 2D, got {prediction_grid.ndim}D"
    assert target_grid.shape == prediction_grid.shape, (
        f"Shape mismatch: target={target_grid.shape} vs pred={prediction_grid.shape}"
    )

    # Normalize weight keys: accept both "zncc" and "w_zncc" formats
    raw_weights = loss_weights or {}
    normalized = {}
    for k, v in raw_weights.items():
        key = k[2:] if k.startswith("w_") else k
        normalized[key] = v
    weights = {**DEFAULT_LOSS_WEIGHTS, **normalized}

    result = compute_grid_losses(
        jnp.array(prediction_grid),
        jnp.array(target_grid),
        w_sinkhorn=weights["sinkhorn"],
        w_lncc=weights["lncc"],
        w_mse=weights["mse"],
        w_rmse=weights["rmse"],
        w_simse=weights["simse"],
        w_spectral=weights["spectral"],
        w_gradient=weights["gradient"],
        w_contrast=weights["contrast"],
        w_zncc=weights["zncc"],
    )

    loss_dict_raw = result.to_dict()

    metrics: dict[str, Any] = {}
    weighted_total = 0.0

    for name in LOSS_ORDER:
        if name in loss_dict_raw and name != "total":
            raw_val = float(loss_dict_raw[name])
            weight = weights.get(name, 0.0)
            weighted_val = raw_val * weight
            metrics[name] = raw_val
            metrics[f"{name}_weighted"] = weighted_val
            weighted_total += weighted_val

    metrics["weighted_total"] = weighted_total
    metrics["weights"] = weights

    correlation = float(np.corrcoef(target_grid.ravel(), prediction_grid.ravel())[0, 1])
    metrics["correlation"] = correlation

    metrics["pred_range"] = (float(prediction_grid.min()), float(prediction_grid.max()))
    metrics["target_range"] = (float(target_grid.min()), float(target_grid.max()))

    return metrics


def side_by_side_txt_plot(
    target_grid: np.ndarray,
    prediction_grid: np.ndarray,
    height: int = 10,
    width: int = 24,
    loss_weights: dict[str, float] | None = None,
    title_target: str = "TARGET",
    title_prediction: str = "PREDICTION",
    training_losses: dict[str, float] | None = None,
    training_penalties: dict[str, float] | None = None,
    training_grid_total: float | None = None,
    shared_colorbar: bool = False,
    xlim: tuple[float, float] = (0.0, 1.0),
    ylim: tuple[float, float] = (0.0, 1.0),
    show_axes: bool = True,
    compute_metrics: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Create side-by-side ASCII heatmaps of target vs prediction with loss metrics table.

    Args:
        target_grid: 2D array (H, W) of target pattern
        prediction_grid: 2D array (H, W) of predicted pattern
        height: Height of each heatmap in characters
        width: Width of each heatmap in characters
        loss_weights: Optional dict of loss weights {loss_name: weight}.
            If None, uses defaults: sinkhorn=1.0, lncc=0.5, rmse=0.5
        title_target: Title for target heatmap
        title_prediction: Title for prediction heatmap
        training_losses: Optional dict of training losses {loss_name: weighted_value}.
            Keys should match LOSS_ORDER (sinkhorn, lncc, rmse, etc.).
            If provided, shows Training vs Eval comparison table with deltas.
        training_penalties: Optional dict of training penalties {penalty_name: value}.
            If provided, shows penalties in the table (training-only, no eval equivalent).
        training_grid_total: Optional training grid total for comparison.
            When provided, shows delta between training and eval totals.
        shared_colorbar: If True, use shared vmin/vmax across both plots with single
            colorbar. If False (default), each plot has independent color range.
        xlim: X-axis range (min, max) for tick labels. Default (0.0, 1.0).
        ylim: Y-axis range (min, max) for tick labels. Default (0.0, 1.0).
        show_axes: If True (default), show min/max tick labels on x and y axes.
        compute_metrics: If True (default), compute and return comprehensive metrics
            via compute_grid_metrics(). If False, return empty metrics dict (faster).

    Returns:
        (txt_output, metrics_dict) where:
            - txt_output: Formatted string with side-by-side heatmaps (and loss table if compute_metrics=True)
            - metrics_dict: Dict from compute_grid_metrics() with losses, correlation, ranges.
              Also includes training_penalties, training_grid_total, delta_grid if provided.
              Empty dict if compute_metrics=False.
    """
    from biocomp.plotting.ascii_heatmap import heatmap

    assert target_grid.ndim == 2, f"target_grid must be 2D, got {target_grid.ndim}D"
    assert prediction_grid.ndim == 2, f"prediction_grid must be 2D, got {prediction_grid.ndim}D"
    assert target_grid.shape == prediction_grid.shape, (
        f"Shape mismatch: target={target_grid.shape} vs pred={prediction_grid.shape}"
    )

    if compute_metrics:
        metrics = compute_grid_metrics(target_grid, prediction_grid, loss_weights)
    else:
        metrics = {}

    if shared_colorbar:
        vmin_target = vmin_pred = min(float(target_grid.min()), float(prediction_grid.min()))
        vmax_target = vmax_pred = max(float(target_grid.max()), float(prediction_grid.max()))
    else:
        vmin_target, vmax_target = float(target_grid.min()), float(target_grid.max())
        vmin_pred, vmax_pred = float(prediction_grid.min()), float(prediction_grid.max())

    # Flip vertically for proper orientation (origin at bottom-left like scientific plots)
    target_flipped = np.flipud(target_grid)
    pred_flipped = np.flipud(prediction_grid)

    target_hm = heatmap(
        target_flipped,
        vmin=vmin_target,
        vmax=vmax_target,
        xres=width,
        yres=height,
        show_colorbar=False,
        resample="mean",
    )
    pred_hm = heatmap(
        pred_flipped,
        vmin=vmin_pred,
        vmax=vmax_pred,
        xres=width,
        yres=height,
        show_colorbar=False,
        resample="mean",
    )

    target_lines = target_hm.split("\n")
    pred_lines = pred_hm.split("\n")

    max_lines = max(len(target_lines), len(pred_lines))
    target_lines += [""] * (max_lines - len(target_lines))
    pred_lines += [""] * (max_lines - len(pred_lines))

    gap = "    "
    lines = []

    y_label_width = 5 if show_axes else 0
    y_min_str = f"{ylim[0]:.1f}" if show_axes else ""
    y_max_str = f"{ylim[1]:.1f}" if show_axes else ""

    if show_axes:
        title_pad = " " * y_label_width
        lines.append(
            f"{title_pad}{title_target.center(width)}{gap}{title_pad}{title_prediction.center(width)}"
        )
    else:
        lines.append(f"{title_target.center(width)}{gap}{title_prediction.center(width)}")

    for i, (t_line, p_line) in enumerate(zip(target_lines, pred_lines, strict=False)):
        t_padded = t_line.ljust(width) if len(t_line) < width else t_line[:width]
        p_padded = p_line.ljust(width) if len(p_line) < width else p_line[:width]

        if show_axes:
            if i == 0:
                y_label = f"{y_max_str:>{y_label_width - 1}}┤"
            elif i == max_lines - 1:
                y_label = f"{y_min_str:>{y_label_width - 1}}┤"
            else:
                y_label = " " * (y_label_width - 1) + "│"
            lines.append(f"{y_label}{t_padded}{gap}{y_label}{p_padded}")
        else:
            lines.append(f"{t_padded}{gap}{p_padded}")

    if show_axes:
        x_min_str = f"{xlim[0]:.1f}"
        x_max_str = f"{xlim[1]:.1f}"
        x_axis_line = f"{x_min_str}{' ' * (width - len(x_min_str) - len(x_max_str))}{x_max_str}"
        y_pad = " " * y_label_width
        lines.append(f"{y_pad}{x_axis_line}{gap}{y_pad}{x_axis_line}")

    cmap_chars = CMAP_S[5]
    if show_axes:
        cb_pad = " " * y_label_width
    else:
        cb_pad = ""

    if shared_colorbar:
        cb_str = f"{vmin_target:.2f} {cmap_chars} {vmax_target:.2f}"
        total_width = (width + y_label_width) * 2 + len(gap) if show_axes else width * 2 + len(gap)
        lines.append(cb_str.center(total_width))
    else:
        target_cb = f"{vmin_target:.2f} {cmap_chars} {vmax_target:.2f}"
        pred_cb = f"{vmin_pred:.2f} {cmap_chars} {vmax_pred:.2f}"
        lines.append(f"{cb_pad}{target_cb.center(width)}{gap}{cb_pad}{pred_cb.center(width)}")

    delta_grid = 0.0
    if compute_metrics:
        weighted_total = float(metrics["weighted_total"])
        training_grid_total = (
            float(training_grid_total) if training_grid_total is not None else None
        )
        show_training_comparison = training_losses is not None or training_grid_total is not None

        if show_training_comparison:
            delta_grid = (
                abs(training_grid_total - weighted_total)
                if training_grid_total is not None
                else 0.0
            )
            lines.append("")
            lines.append("┌─────────────────┬───────────┬───────────┬─────────┐")
            lines.append("│ Component       │  Training │      Eval │   Delta │")
            lines.append("├─────────────────┼───────────┼───────────┼─────────┤")

            train_losses = training_losses or {}
            for name in LOSS_ORDER:
                weighted_key = f"{name}_weighted"
                if weighted_key in metrics:
                    eval_w = float(metrics[weighted_key])
                    train_w_raw = train_losses.get(name, train_losses.get(weighted_key, None))
                    train_w = float(train_w_raw) if train_w_raw is not None else None
                    if eval_w > 0 or (train_w is not None and train_w > 0):
                        if train_w is not None:
                            delta = abs(train_w - eval_w)
                            lines.append(
                                f"│ {name:15} │ {train_w:9.4f} │ {eval_w:9.4f} │ {delta:7.4f} │"
                            )
                        else:
                            lines.append(f"│ {name:15} │       n/a │ {eval_w:9.4f} │         │")

            lines.append("├─────────────────┼───────────┼───────────┼─────────┤")
            if training_grid_total is not None:
                lines.append(
                    f"│ {'GRID TOTAL':15} │ {training_grid_total:9.4f} │ {weighted_total:9.4f} │ {delta_grid:7.4f} │"
                )
            else:
                lines.append(f"│ {'GRID TOTAL':15} │       n/a │ {weighted_total:9.4f} │         │")

            if training_penalties:
                lines.append("├─────────────────┼───────────┼───────────┼─────────┤")
                penalty_order = [
                    "l0_penalty",
                    "spread_penalty",
                    "coupling_penalty",
                    "tucount_penalty",
                    "ern_tying_penalty",
                ]
                penalty_sum = 0.0
                for pen_name in penalty_order:
                    if pen_name in training_penalties:
                        pen_val = float(training_penalties[pen_name])
                        penalty_sum += pen_val
                        lines.append(f"│ {pen_name:15} │ {pen_val:9.4f} │       n/a │         │")

                if training_grid_total is not None:
                    total_w_pen = training_grid_total + penalty_sum
                    lines.append("├─────────────────┼───────────┼───────────┼─────────┤")
                    lines.append(
                        f"│ {'TOTAL + PENALTY':15} │ {total_w_pen:9.4f} │       n/a │         │"
                    )

            lines.append("└─────────────────┴───────────┴───────────┴─────────┘")
        else:
            lines.append("")
            lines.append("┌────────────┬──────────┬──────────┐")
            lines.append("│ Loss       │ Unweight │ Weighted │")
            lines.append("├────────────┼──────────┼──────────┤")

            for name in LOSS_ORDER:
                if name in metrics:
                    raw_val = float(metrics[name])
                    weighted_val = float(metrics.get(f"{name}_weighted", 0.0))
                    if raw_val > 1e-8 or weighted_val > 1e-8:
                        lines.append(f"│ {name:10} │ {raw_val:8.4f} │ {weighted_val:8.4f} │")

            lines.append("├────────────┼──────────┼──────────┤")
            lines.append(f"│ {'TOTAL':10} │ {'':8} │ {weighted_total:8.4f} │")

            if training_penalties:
                lines.append("├────────────┼──────────┼──────────┤")
                penalty_order = [
                    "l0_penalty",
                    "spread_penalty",
                    "coupling_penalty",
                    "tucount_penalty",
                    "ern_tying_penalty",
                ]
                penalty_sum = 0.0
                for pen_name in penalty_order:
                    if pen_name in training_penalties:
                        pen_val = float(training_penalties[pen_name])
                        if pen_val > 1e-8:
                            penalty_sum += pen_val
                            lines.append(f"│ {pen_name:10} │ {'':8} │ {pen_val:8.4f} │")

                if penalty_sum > 1e-8:
                    total_w_pen = weighted_total + penalty_sum
                    lines.append("├────────────┼──────────┼──────────┤")
                    lines.append(f"│ {'TOTAL+PEN':10} │ {'':8} │ {total_w_pen:8.4f} │")

            lines.append("└────────────┴──────────┴──────────┘")

    metrics_out = {
        **metrics,
        "training_losses": training_losses,
        "training_grid_total": training_grid_total,
        "training_penalties": training_penalties,
        "delta_grid": delta_grid,
    }

    return "\n".join(lines), metrics_out
