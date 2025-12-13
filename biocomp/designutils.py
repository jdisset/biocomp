import numpy as np
from numpy.typing import NDArray
from matplotlib.path import Path as MPath
from xml.etree import ElementTree as ET
import re
from matplotlib.colors import to_rgb
from pathlib import Path
from typing import Optional
from biocomp.logging_config import get_logger

NdArray = np.ndarray
logger = get_logger(__name__)


def _parse_transform(transform_str):
    if not transform_str:
        return np.eye(3)

    rotate_match = re.match(r"rotate\(([-\d.]+)(?:\s+([-\d.]+)\s+([-\d.]+))?\)", transform_str)
    if not rotate_match:
        return np.eye(3)

    angle = float(rotate_match.group(1)) * np.pi / 180
    cx = float(rotate_match.group(2)) if rotate_match.group(2) else 0
    cy = float(rotate_match.group(3)) if rotate_match.group(3) else 0

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [cos_a, -sin_a, cx - cx * cos_a + cy * sin_a],
            [sin_a, cos_a, cy - cx * sin_a - cy * cos_a],
            [0, 0, 1],
        ]
    )


def _apply_transform(pts, transform_matrix):
    if np.array_equal(transform_matrix, np.eye(3)):
        return pts
    pts_h = np.column_stack([pts, np.ones(len(pts))])
    return (pts_h @ transform_matrix.T)[:, :2]


def _inside_edge(pt, edge_pt1, edge_pt2):
    return (edge_pt2[0] - edge_pt1[0]) * (pt[1] - edge_pt1[1]) >= (edge_pt2[1] - edge_pt1[1]) * (
        pt[0] - edge_pt1[0]
    )


def _line_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def _clip_polygon_to_rect(polygon_pts, rect_bounds):
    # sutherland-hodgman algorithm
    x_min, y_min, x_max, y_max = rect_bounds
    rect_edges = [
        [(x_min, y_min), (x_max, y_min)],
        [(x_max, y_min), (x_max, y_max)],
        [(x_max, y_max), (x_min, y_max)],
        [(x_min, y_max), (x_min, y_min)],
    ]

    output_pts = list(polygon_pts)

    for edge in rect_edges:
        if not output_pts:
            break

        input_pts = output_pts
        output_pts = []

        if not input_pts:
            continue

        prev_pt = input_pts[-1]

        for curr_pt in input_pts:
            curr_inside = _inside_edge(curr_pt, edge[0], edge[1])
            prev_inside = _inside_edge(prev_pt, edge[0], edge[1])

            if curr_inside:
                if not prev_inside:
                    intersection = _line_intersection(prev_pt, curr_pt, edge[0], edge[1])
                    if intersection:
                        output_pts.append(intersection)
                output_pts.append(curr_pt)
            elif prev_inside:
                intersection = _line_intersection(prev_pt, curr_pt, edge[0], edge[1])
                if intersection:
                    output_pts.append(intersection)

            prev_pt = curr_pt

    return np.array(output_pts) if output_pts else np.array([])


def _parse_svg_path(d):
    # parse basic svg path commands
    tok, out, i, x, y = re.findall(r"[MLHVZmlhvz]|[-+]?\d*\.?\d+", d), [], 0, 0, 0
    while i < len(tok):
        c = tok[i]
        if c in "ML":
            x, y = map(float, tok[i + 1 : i + 3])
            out.append((x, y))
            i += 3
        elif c == "H":
            x = float(tok[i + 1])
            out.append((x, y))
            i += 2
        else:
            i += 1
    return out


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


def _get_masked_elements(root):
    masked_elements = []
    for group in root.iter():
        if group.tag.endswith("g") and group.get("mask"):
            for el in group.iter():
                if el != group:
                    masked_elements.append(el)
    return masked_elements if masked_elements else root.iter()


def _create_circle_path(cx, cy, r, n_points=32):
    angles = np.linspace(0, 2 * np.pi, n_points + 1)
    return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]


def _process_rect(el, vw, vh, vx, vy, max_is_black):
    x = float(el.get("x", 0))
    y = float(el.get("y", 0))
    w = float(el.get("width", vw))
    h = float(el.get("height", vh))
    pts = np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])

    transform = el.get("transform", "")
    if transform:
        pts = _apply_transform(pts, _parse_transform(transform))
        pts = _clip_polygon_to_rect(pts, (vx, vy, vx + vw, vy + vh))
        if len(pts) < 3:
            return None, None
        if len(pts) > 0 and not np.array_equal(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])
    else:
        pts = np.vstack([pts, pts[0]])

    return MPath(pts), _greyscale(el.get("fill", "none"), max_is_black)


def _extract_shapes_from_svg(svg_path, max_is_black):
    root = ET.parse(svg_path).getroot()
    vx, vy, vw, vh = map(float, root.get("viewBox", "0 0 100 100").split())

    paths, greys = [], []
    elements_to_process = _get_masked_elements(root)

    for el in elements_to_process:
        fill = el.get("fill", "none")
        if fill in ("none", "white"):
            continue

        if el.tag.endswith("rect"):
            path, grey = _process_rect(el, vw, vh, vx, vy, max_is_black)
            if path:
                paths.append(path)
                greys.append(grey)
        elif el.tag.endswith("path"):
            pts = _parse_svg_path(el.get("d", ""))
            if len(pts) >= 3:
                paths.append(MPath(pts + [pts[0]]))
                greys.append(_greyscale(fill, max_is_black))
        elif el.tag.endswith("circle"):
            cx = float(el.get("cx", 0))
            cy = float(el.get("cy", 0))
            r = float(el.get("r", 0))
            pts = _create_circle_path(cx, cy, r)
            paths.append(MPath(pts))
            greys.append(_greyscale(fill, max_is_black))

    return paths, np.asarray(greys), (vx, vy, vw, vh)


def _generate_svg_sample_points(
    n, lattice_extent, img_latent_lim, vx, vy, vw, vh, rng, log, grid=None, grid_jitter_std=None
):
    """Generate sample points in SVG coordinate space and return latent coordinates.

    Args:
        lattice_extent: (x_extent, y_extent) - final latent space bounds
        img_latent_lim: (x_lim, y_lim) - SVG-to-latent mapping parameters
    """
    x_extent, y_extent = lattice_extent
    x_lim, y_lim = img_latent_lim

    if grid:
        xres, yres = grid
        if log:
            eps = 1e-6
            x_vals = (
                np.logspace(np.log10(eps + x_extent[0] * vw), np.log10(x_extent[1] * vw), xres) + vx
            )
            y_vals = (
                vh
                - np.logspace(np.log10(eps + y_extent[0] * vh), np.log10(y_extent[1] * vh), yres)
                + vy
            )
        else:
            x_vals = np.linspace(x_extent[0] * vw + vx, x_extent[1] * vw + vx, xres)
            y_vals = np.linspace((1 - y_extent[1]) * vh + vy, (1 - y_extent[0]) * vh + vy, yres)

        sx_grid, sy_grid = np.meshgrid(x_vals, y_vals)
        sx_list, sy_list = [], []

        for _ in range(n):
            sx_sample, sy_sample = sx_grid.copy(), sy_grid.copy()
            if grid_jitter_std and grid_jitter_std > 0:
                x_spacing = (x_vals[-1] - x_vals[0]) / (xres - 1) if xres > 1 else 0
                y_spacing = (y_vals[-1] - y_vals[0]) / (yres - 1) if yres > 1 else 0
                sx_sample += rng.normal(0, grid_jitter_std * x_spacing, sx_sample.shape)
                sy_sample += rng.normal(0, grid_jitter_std * y_spacing, sy_sample.shape)
            sx_list.append(sx_sample.flatten())
            sy_list.append(sy_sample.flatten())

        sx, sy = np.concatenate(sx_list), np.concatenate(sy_list)
    else:
        if log:
            eps = 1e-6
            sx = (
                10 ** rng.uniform(np.log10(eps + x_extent[0] * vw), np.log10(x_extent[1] * vw), n)
                + vx
            )
            sy = (
                vh
                - 10 ** rng.uniform(np.log10(eps + y_extent[0] * vh), np.log10(y_extent[1] * vh), n)
                + vy
            )
        else:
            sx = rng.uniform(x_extent[0] * vw + vx, x_extent[1] * vw + vx, n)
            sy = rng.uniform((1 - y_extent[1]) * vh + vy, (1 - y_extent[0]) * vh + vy, n)

    # Convert SVG coordinates to latent space
    if log:
        X = np.column_stack((np.log10(sx - vx), np.log10(vh - (sy - vy))))
    else:
        # Map SVG coords to [0,1] normalized, then scale by img_latent_lim
        x_norm = (sx - vx) / vw
        y_norm = (vh - (sy - vy)) / vh
        x_latent = x_norm * (x_lim[1] - x_lim[0]) + x_lim[0]
        y_latent = y_norm * (y_lim[1] - y_lim[0]) + y_lim[0]
        X = np.column_stack((x_latent, y_latent))

    return X, sx, sy


def _assign_greyscale_values(sx, sy, paths, greys, max_is_black, outlim, grid_shape=None):
    default_background = 0.0 if max_is_black else 1.0
    Y = np.full(len(sx), default_background)
    pts = np.column_stack((sx, sy))
    for p, g in zip(paths, greys):
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
    lattice_x_extent: tuple[float, float] = None,
    lattice_y_extent: tuple[float, float] = None,
    img_latent_xlim: tuple[float, float] = (0.0, 1.0),
    img_latent_ylim: tuple[float, float] = (0.0, 1.0),
    img_latent_outlim: tuple[float, float] = (0.0, 1.0),
    # Legacy parameters (deprecated)
    rescale_to=None,
    xlim=None,
    ylim=None,
    outlim=None,
):
    """Sample points from an SVG file and return latent coordinates + values.

    New API:
        lattice_x_extent, lattice_y_extent: Final latent space bounds for sampling
        img_latent_xlim, img_latent_ylim: How SVG viewBox maps to latent coordinates
        img_latent_outlim: How SVG grayscale maps to latent output values

    Legacy API (deprecated):
        xlim, ylim, outlim, rescale_to: Old parameter names, will emit warnings
    """
    import warnings

    svg_path = Path(svg_path).expanduser().resolve()
    seed = seed or np.random.randint(0, 2**32 - 1)

    # Handle legacy parameters
    if xlim is not None or ylim is not None or outlim is not None or rescale_to is not None:
        warnings.warn(
            "sample_from_svg: xlim/ylim/outlim/rescale_to are deprecated. "
            "Use lattice_*_extent and img_latent_*lim instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Legacy mode: xlim/ylim were used for both extent and mapping, rescale_to for final coords
        if rescale_to:
            lattice_x_extent = lattice_x_extent or tuple(rescale_to.get("x", (0, 1)))
            lattice_y_extent = lattice_y_extent or tuple(rescale_to.get("y", (0, 1)))
            img_latent_outlim = outlim or tuple(rescale_to.get("out", (0, 1)))
        else:
            # Without rescale_to, xlim/ylim were both extent and mapping
            lattice_x_extent = lattice_x_extent or xlim
            lattice_y_extent = lattice_y_extent or ylim
            img_latent_outlim = outlim or (0, 1)
        img_latent_xlim = xlim or (0, 1)
        img_latent_ylim = ylim or (0, 1)

    # Apply defaults
    default_extent = (0.1, 1.0) if log else (0.0, 1.0)
    lattice_x_extent = lattice_x_extent or default_extent
    lattice_y_extent = lattice_y_extent or default_extent

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
        (lattice_x_extent, lattice_y_extent),
        (img_latent_xlim, img_latent_ylim),
        vx,
        vy,
        vw,
        vh,
        rng,
        log,
        grid,
        grid_jitter_std,
    )
    Y = _assign_greyscale_values(sx, sy, paths, greys, max_is_black, img_latent_outlim, grid_shape)

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

    for i in range(n):
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
    return X_grid, Y_grid[0]  # return single grid (squeeze n dimension)
