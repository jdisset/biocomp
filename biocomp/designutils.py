import numpy as np
from matplotlib.path import Path as MPath
from xml.etree import ElementTree as ET
import re
from matplotlib.colors import to_rgb
from matplotlib.path import Path as MPath


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
    Now supports background masks (rectangular).
    """
    from pathlib import Path

    svg_path = Path(svg_path).expanduser().resolve()

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

    # masks and viewbox limits
    mask_rect = None
    for el in root.iter():
        if el.tag.endswith("mask"):
            for child in el:
                if child.tag.endswith("rect"):
                    x = float(child.get("x", 0))
                    y = float(child.get("y", 0))
                    w = float(child.get("width", vw))
                    h = float(child.get("height", vh))
                    mask_rect = (x, y, x + w, y + h)

    # parse filled polygons and rects
    paths, greys = [], []
    for el in root.iter():
        fill = el.get("fill", "none")
        if fill == "none":
            continue
        if el.tag.endswith("rect"):
            x = float(el.get("x", 0))
            y = float(el.get("y", 0))
            w = float(el.get("width", vw))
            h = float(el.get("height", vh))
            pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
            paths.append(MPath(pts))
            greys.append(greyscale(fill))
        elif el.tag.endswith("path"):
            pts = _coords(el.get("d", ""))
            if len(pts) >= 3:
                paths.append(MPath(pts + [pts[0]]))
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
    Y = np.full(n, 1.0)
    pts = np.column_stack((sx, sy))
    for p, g in zip(paths, greys):
        Y[p.contains_points(pts)] = g

    # --- Apply mask rect if present: points outside mask are background ---
    if mask_rect:
        x0, y0, x1, y1 = mask_rect
        inside_mask = (sx >= x0) & (sx <= x1) & (sy >= y0) & (sy <= y1)
        Y[~inside_mask] = 1.0  # or whatever background you want

    Y = Y * (outlim[1] - outlim[0]) + outlim[0]

    # rescale
    xrescale = rescale_to.get("x", (0, 1))
    yrescale = rescale_to.get("y", (0, 1))
    outrescale = rescale_to.get("out", (0, 1))
    X[:, 0] = (X[:, 0] - xlim[0]) / (xlim[1] - xlim[0]) * (xrescale[1] - xrescale[0]) + xrescale[0]
    X[:, 1] = (X[:, 1] - ylim[0]) / (ylim[1] - ylim[0]) * (yrescale[1] - yrescale[0]) + yrescale[0]
    Y = (Y - outlim[0]) / (outlim[1] - outlim[0]) * (outrescale[1] - outrescale[0]) + outrescale[0]

    return X, Y[:, None]
