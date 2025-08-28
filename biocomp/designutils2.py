import numpy as np
from matplotlib.path import Path as MPath
from xml.etree import ElementTree as ET
import re
from matplotlib.colors import to_rgb
from pathlib import Path


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


def _generate_sample_points(
    n, xlim, ylim, vx, vy, vw, vh, rng, log, grid=None, grid_jitter_std=None
):
    if grid:
        # fixed grid sampling with optional jitter
        xres, yres = grid
        if log:
            eps = 1e-6
            x_vals = np.logspace(np.log10(eps + xlim[0] * vw), np.log10(xlim[1] * vw), xres) + vx
            y_vals = (
                vh - np.logspace(np.log10(eps + ylim[0] * vh), np.log10(ylim[1] * vh), yres) + vy
            )
        else:
            x_vals = np.linspace(xlim[0] * vw + vx, xlim[1] * vw + vx, xres)
            y_vals = np.linspace((1 - ylim[1]) * vh + vy, (1 - ylim[0]) * vh + vy, yres)

        # create base grid
        sx_grid, sy_grid = np.meshgrid(x_vals, y_vals)

        # replicate grid n times and add jitter if requested
        sx_list, sy_list = [], []
        for _ in range(n):
            sx_sample = sx_grid.copy()
            sy_sample = sy_grid.copy()

            if grid_jitter_std is not None and grid_jitter_std > 0:
                # add gaussian noise
                x_spacing = (x_vals[-1] - x_vals[0]) / (xres - 1) if xres > 1 else 0
                y_spacing = (y_vals[-1] - y_vals[0]) / (yres - 1) if yres > 1 else 0
                sx_sample += rng.normal(0, grid_jitter_std * x_spacing, sx_sample.shape)
                sy_sample += rng.normal(0, grid_jitter_std * y_spacing, sy_sample.shape)

            sx_list.append(sx_sample.flatten())
            sy_list.append(sy_sample.flatten())

        sx = np.concatenate(sx_list)
        sy = np.concatenate(sy_list)

        if log:
            X = np.column_stack((np.log10(sx - vx), np.log10(vh - (sy - vy))))
        else:
            X = np.column_stack(
                (
                    (sx - vx) / vw * (xlim[1] - xlim[0]) + xlim[0],
                    (vh - (sy - vy)) / vh * (ylim[1] - ylim[0]) + ylim[0],
                )
            )
    else:
        # random uniform sampling
        if log:
            eps = 1e-6
            sx = 10 ** rng.uniform(np.log10(eps + xlim[0] * vw), np.log10(xlim[1] * vw), n) + vx
            sy = (
                vh - 10 ** rng.uniform(np.log10(eps + ylim[0] * vh), np.log10(ylim[1] * vh), n) + vy
            )
            X = np.column_stack((np.log10(sx - vx), np.log10(vh - (sy - vy))))
        else:
            sx = rng.uniform(xlim[0] * vw + vx, xlim[1] * vw + vx, n)
            sy = rng.uniform((1 - ylim[1]) * vh + vy, (1 - ylim[0]) * vh + vy, n)
            X = np.column_stack(
                (
                    (sx - vx) / vw * (xlim[1] - xlim[0]) + xlim[0],
                    (vh - (sy - vy)) / vh * (ylim[1] - ylim[0]) + ylim[0],
                )
            )
    return X, sx, sy


def _assign_greyscale_values(sx, sy, paths, greys, max_is_black, outlim, grid_shape=None):
    default_background = 0.0 if max_is_black else 1.0
    Y = np.full(len(sx), default_background)
    pts = np.column_stack((sx, sy))
    for p, g in zip(paths, greys):
        Y[p.contains_points(pts)] = g
    Y = Y * (outlim[1] - outlim[0]) + outlim[0]

    # reshape if grid sampling
    if grid_shape:
        n, xres, yres = grid_shape
        # Y shape is (n * yres * xres,) -> reshape to (n, yres, xres)
        Y = Y.reshape(n, yres, xres)

    return Y


def _rescale_outputs(X, Y, xlim, ylim, outlim, rescale_to):
    xrescale = rescale_to.get("x", (0, 1))
    yrescale = rescale_to.get("y", (0, 1))
    outrescale = rescale_to.get("out", (0, 1))

    X[:, 0] = (X[:, 0] - xlim[0]) / (xlim[1] - xlim[0]) * (xrescale[1] - xrescale[0]) + xrescale[0]
    X[:, 1] = (X[:, 1] - ylim[0]) / (ylim[1] - ylim[0]) * (yrescale[1] - yrescale[0]) + yrescale[0]

    # handle both flat and grid-shaped Y
    Y_flat = Y.flatten() if Y.ndim > 1 else Y
    Y_flat = (Y_flat - outlim[0]) / (outlim[1] - outlim[0]) * (
        outrescale[1] - outrescale[0]
    ) + outrescale[0]

    if Y.ndim > 1:
        Y = Y_flat.reshape(Y.shape)
    else:
        Y = Y_flat

    return X, Y


def sample_from_svg(
    svg_path,
    n=None,
    rescale_to=None,
    xlim=None,
    ylim=None,
    outlim=(0, 1),
    *,
    seed=None,
    log=False,
    max_is_black=True,
    grid=None,
    grid_jitter_std=None,
):
    # sample from svg with greyscale intensity
    svg_path = Path(svg_path).expanduser().resolve()

    xlim = xlim or ((0.1, 1) if log else (0, 1))
    ylim = ylim or ((0.1, 1) if log else (0, 1))
    rescale_to = rescale_to or {}
    seed = seed or np.random.randint(0, 2**32 - 1)

    # validate n and grid
    if grid:
        n = n or 1  # default to 1 full lattice sample
        xres, yres = grid
        grid_shape = (n, xres, yres)
    elif n is None:
        raise ValueError("n must be specified when not using grid sampling")
    else:
        grid_shape = None

    rng = np.random.default_rng(seed)

    paths, greys, (vx, vy, vw, vh) = _extract_shapes_from_svg(svg_path, max_is_black)
    X, sx, sy = _generate_sample_points(
        n, xlim, ylim, vx, vy, vw, vh, rng, log, grid, grid_jitter_std
    )
    Y = _assign_greyscale_values(sx, sy, paths, greys, max_is_black, outlim, grid_shape)
    X, Y = _rescale_outputs(X, Y, xlim, ylim, outlim, rescale_to)

    # adjust output shape for grid sampling
    if grid:
        # X is (n * yres * xres, 2), keep as is
        # Y is already (n, yres, xres) from _assign_greyscale_values
        return X, Y
    else:
        # uniform sampling: Y needs extra dimension
        return X, Y[:, None]
