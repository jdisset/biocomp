import numpy as np
from matplotlib.path import Path as MPath
from xml.etree import ElementTree as ET
import re
from matplotlib.colors import to_rgb


def sample_from_svg(
    svg_path,
    n,
    rescale_to=None,
    xlim=None,
    ylim=None,
    outlim=(0, 1),
    *,
    seed=None,
    log=False,
    max_is_black=True,
):
    """
    Sample (X, Y) pairs from coloured regions of an SVG,
    with Y as the greyscale intensity (0=black, 1=white).
    Now supports background masks (rectangular) and transformations.
    """
    from pathlib import Path

    svg_path = Path(svg_path).expanduser().resolve()
    
    def _parse_transform(transform_str):
        """Parse SVG transform attribute and return transformation matrix"""
        if not transform_str:
            return np.eye(3)
        
        # handle rotate(angle cx cy) - most common in our SVGs
        rotate_match = re.match(r'rotate\(([-\d.]+)(?:\s+([-\d.]+)\s+([-\d.]+))?\)', transform_str)
        if rotate_match:
            angle = float(rotate_match.group(1)) * np.pi / 180
            cx = float(rotate_match.group(2)) if rotate_match.group(2) else 0
            cy = float(rotate_match.group(3)) if rotate_match.group(3) else 0
            
            # create rotation matrix around (cx, cy)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            # translate to origin, rotate, translate back
            mat = np.array([
                [cos_a, -sin_a, cx - cx * cos_a + cy * sin_a],
                [sin_a, cos_a, cy - cx * sin_a - cy * cos_a],
                [0, 0, 1]
            ])
            return mat
        
        # could add support for other transforms (translate, scale, etc.) if needed
        return np.eye(3)
    
    def _apply_transform(pts, transform_matrix):
        """Apply transformation matrix to points"""
        if np.array_equal(transform_matrix, np.eye(3)):
            return pts
        
        # convert to homogeneous coordinates
        pts_h = np.column_stack([pts, np.ones(len(pts))])
        # apply transformation
        pts_transformed = pts_h @ transform_matrix.T
        # convert back to 2D
        return pts_transformed[:, :2]
    
    def _clip_polygon_to_rect(polygon_pts, rect_bounds):
        """Clip polygon to rectangle using Sutherland-Hodgman algorithm"""
        def inside_edge(pt, edge_pt1, edge_pt2):
            # check if point is on the left side of the edge (inside)
            return (edge_pt2[0] - edge_pt1[0]) * (pt[1] - edge_pt1[1]) >= \
                   (edge_pt2[1] - edge_pt1[1]) * (pt[0] - edge_pt1[0])
        
        def line_intersection(p1, p2, p3, p4):
            # find intersection of line p1-p2 with line p3-p4
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            x4, y4 = p4
            
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-10:
                return None
            
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
        
        x_min, y_min, x_max, y_max = rect_bounds
        rect_edges = [
            [(x_min, y_min), (x_max, y_min)],  # bottom
            [(x_max, y_min), (x_max, y_max)],  # right
            [(x_max, y_max), (x_min, y_max)],  # top
            [(x_min, y_max), (x_min, y_min)]   # left
        ]
        
        output_pts = list(polygon_pts)
        
        for edge in rect_edges:
            if len(output_pts) == 0:
                break
                
            input_pts = output_pts
            output_pts = []
            
            if len(input_pts) == 0:
                continue
                
            prev_pt = input_pts[-1]
            
            for curr_pt in input_pts:
                curr_inside = inside_edge(curr_pt, edge[0], edge[1])
                prev_inside = inside_edge(prev_pt, edge[0], edge[1])
                
                if curr_inside:
                    if not prev_inside:
                        # entering the edge
                        intersection = line_intersection(prev_pt, curr_pt, edge[0], edge[1])
                        if intersection:
                            output_pts.append(intersection)
                    output_pts.append(curr_pt)
                elif prev_inside:
                    # leaving the edge
                    intersection = line_intersection(prev_pt, curr_pt, edge[0], edge[1])
                    if intersection:
                        output_pts.append(intersection)
                
                prev_pt = curr_pt
        
        return np.array(output_pts) if output_pts else np.array([])

    def _coords(d):
        # Parse (x, y) from basic SVG path syntax (M L H)
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

    def greyscale(fill):
        try:
            r, g, b = to_rgb(fill)
        except ValueError:
            m = re.match(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", fill)
            if m:
                r, g, b = [int(x) / 255 for x in m.groups()]
            else:
                return 1.0 if not max_is_black else 0.0
        grey = (r + g + b) / 3.0
        return 1.0 - grey if max_is_black else grey

    if xlim is None:
        xlim = (0, 1) if not log else (0.1, 1)
    if ylim is None:
        ylim = (0, 1) if not log else (0.1, 1)
    seed = seed or np.random.randint(0, 2**32 - 1)

    if rescale_to is None:
        rescale_to = {}

    root = ET.parse(svg_path).getroot()
    vx, vy, vw, vh = map(float, root.get("viewBox", "0 0 100 100").split())
    rng = np.random.default_rng(seed)

    # masks are handled by filtering elements in masked groups, not by rect override

    # parse filled polygons and rects - only process elements inside masked groups
    paths, greys = [], []
    
    # find masked group elements
    masked_elements = []
    for group in root.iter():
        if group.tag.endswith("g") and group.get("mask"):
            # this group has a mask, process its children
            for el in group.iter():
                if el != group:  # don't include the group itself
                    masked_elements.append(el)
    
    # if no masked groups, process all elements (backwards compatibility)
    elements_to_process = masked_elements if masked_elements else root.iter()
    
    for el in elements_to_process:
        fill = el.get("fill", "none")
        if fill == "none" or fill == "white":  # skip white fill as it's background
            continue
        if el.tag.endswith("rect"):
            x = float(el.get("x", 0))
            y = float(el.get("y", 0))
            w = float(el.get("width", vw))
            h = float(el.get("height", vh))
            pts = np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
            
            # apply transformation if present
            transform = el.get("transform", "")
            if transform:
                transform_matrix = _parse_transform(transform)
                pts = _apply_transform(pts, transform_matrix)
                
                # clip to viewbox if transformed
                pts = _clip_polygon_to_rect(pts, (vx, vy, vx + vw, vy + vh))
                
                if len(pts) >= 3:  # need at least 3 points for a valid polygon
                    # close the polygon
                    if len(pts) > 0 and not np.array_equal(pts[0], pts[-1]):
                        pts = np.vstack([pts, pts[0]])
                    paths.append(MPath(pts))
                    greys.append(greyscale(fill))
            else:
                # no transformation, add as-is
                pts = np.vstack([pts, pts[0]])  # close the polygon
                paths.append(MPath(pts))
                greys.append(greyscale(fill))
        elif el.tag.endswith("path"):
            pts = _coords(el.get("d", ""))
            if len(pts) >= 3:
                paths.append(MPath(pts + [pts[0]]))
                greys.append(greyscale(fill))
        elif el.tag.endswith("circle"):
            # parse circle parameters
            cx = float(el.get("cx", 0))
            cy = float(el.get("cy", 0))
            r = float(el.get("r", 0))
            # create circle path using parametric equations
            n_points = 32  # number of points to approximate the circle
            angles = np.linspace(0, 2 * np.pi, n_points + 1)
            pts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]
            paths.append(MPath(pts))
            greys.append(greyscale(fill))

    greys = np.asarray(greys)

    # sample
    if log:
        eps = 1e-6
        sx = 10 ** rng.uniform(np.log10(eps + xlim[0] * vw), np.log10(xlim[1] * vw), n) + vx
        sy = vh - 10 ** rng.uniform(np.log10(eps + ylim[0] * vh), np.log10(ylim[1] * vh), n) + vy
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

    # --- Assign greyscale values based on polygon ---
    # Initialize to white (0.0 when max_is_black=True, 1.0 when max_is_black=False)
    default_background = 0.0 if max_is_black else 1.0
    Y = np.full(n, default_background)
    pts = np.column_stack((sx, sy))
    for p, g in zip(paths, greys):
        Y[p.contains_points(pts)] = g

    Y = Y * (outlim[1] - outlim[0]) + outlim[0]

    # rescale
    xrescale = rescale_to.get("x", (0, 1))
    yrescale = rescale_to.get("y", (0, 1))
    outrescale = rescale_to.get("out", (0, 1))
    X[:, 0] = (X[:, 0] - xlim[0]) / (xlim[1] - xlim[0]) * (xrescale[1] - xrescale[0]) + xrescale[0]
    X[:, 1] = (X[:, 1] - ylim[0]) / (ylim[1] - ylim[0]) * (yrescale[1] - yrescale[0]) + yrescale[0]
    Y = (Y - outlim[0]) / (outlim[1] - outlim[0]) * (outrescale[1] - outrescale[0]) + outrescale[0]

    return X, Y[:, None]
