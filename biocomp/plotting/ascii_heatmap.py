# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
import numpy as np
from numpy.typing import NDArray
from typing import Literal

GRAY = np.array([16] + list(range(232, 256)) + [231])
CMAP_S = {
    5: " ░▒▓█",
    6: " ░▚▒▓█",
    8: " ·=○@░▒▓█",
    10: "0123456789",
    12: " ·=≡○♣@░▚▒▓█",
    16: " ·-:=+≡○◎♣@░▚▒▓█",
}
CMAP_B = {
    5: ["  ", "░░", "▒▒", "▓▓", "██"],
    8: ["  ", "░░", "░▒", "▒▒", "▒▓", "▓▓", "▓█", "██"],
    10: ["00", "11", "22", "33", "44", "55", "66", "77", "88", "99"],
    12: ["  ", "-=", "##", "#@", "░░", "░▒", "▒▒", "▒▓", "▓▓", "▓█", "▓█", "██"],
    16: [
        "  ",
        "-=",
        "**",
        "*#",
        "#@",
        "░░",
        "@▚",
        "░▒",
        "▒▒",
        "▒▓",
        "▒▓",
        "▓▓",
        "▒▓",
        "▓▓",
        "▓█",
        "██",
    ],
}
CMAP_B_EXT = {
    10: ["  ", "++", "□#", "◐■", "░▚", "■▒", "▒▒", "▒▓", "▓▓", "██"],
    16: [
        "  ",
        "-=",
        "=○",
        "*◑",
        "◑◑",
        "◐■",
        "◑▚",
        "■▚",
        "▚▚",
        "▚▒",
        "▒▒",
        "▚█",
        "▒▓",
        "▓▓",
        "▓█",
        "██",
    ],
}


def _color_seq(vals: NDArray[np.floating], bigram: bool = False) -> str:
    ch = "██" if bigram else "█"
    colors = GRAY[np.clip((np.asarray(vals) * 25).astype(int), 0, 25)]
    changes = np.concatenate([[True], colors[1:] != colors[:-1]])
    parts: list[str] = []
    for c, changed in zip(colors, changes, strict=False):
        if changed:
            parts.append(f"\033[38;5;{c}m")
        parts.append(ch)
    return "".join(parts) + "\033[0m"


def _resample_nearest(data: NDArray[np.floating], h: int, w: int) -> NDArray[np.floating]:
    oh, ow = data.shape
    yi = np.linspace(0, oh - 1, h).astype(int)
    xi = np.linspace(0, ow - 1, w).astype(int)
    return data[np.ix_(yi, xi)]


def _resample_mean(data: NDArray[np.floating], h: int, w: int) -> NDArray[np.floating]:
    oh, ow = data.shape
    xb = np.linspace(0, ow, w + 1).astype(int)
    xb[-1] = ow
    yb = np.linspace(0, oh, h + 1).astype(int)
    yb[-1] = oh
    xw, yw = np.maximum(np.diff(xb), 1), np.maximum(np.diff(yb), 1)
    col_means = np.add.reduceat(data, xb[:-1], axis=1) / xw
    return np.add.reduceat(col_means, yb[:-1], axis=0) / yw[:, None]


def _resample(
    data: NDArray[np.floating], h: int, w: int, method: str = "nearest"
) -> NDArray[np.floating]:
    if h == data.shape[0] and w == data.shape[1]:
        return data
    return _resample_nearest(data, h, w) if method == "nearest" else _resample_mean(data, h, w)


def heatmap(
    data: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    xres: int | None = None,
    yres: int | None = None,
    cmap: str | list[str] | None = None,
    mode: Literal["single", "bigram"] = "single",
    levels: int = 5,
    show_colorbar: bool = True,
    border: bool = False,
    color: bool = False,
    resample: Literal["nearest", "mean"] = "mean",
) -> str:
    data = np.asarray(data, dtype=float)
    assert data.ndim == 2, f"Expected 2D array, got {data.ndim}D"
    vmin = vmin if vmin is not None else float(np.nanmin(data))
    vmax = vmax if vmax is not None else float(np.nanmax(data))
    xres, yres = xres or 64, yres or 32
    if xres != data.shape[1] or yres != data.shape[0]:
        data = _resample(data, yres, xres, resample)
    cmap = cmap or (
        CMAP_B.get(levels, CMAP_B[16]) if mode == "bigram" else CMAP_S.get(levels, CMAP_S[16])
    )
    cmap = list(cmap) if isinstance(cmap, str) else cmap
    cmap_arr = np.array(cmap)
    n = len(cmap)
    norm = np.clip((data - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(data), 0, 1)
    norm = np.nan_to_num(norm, nan=0.0)
    idx = np.clip((norm * (n - 1)).astype(int), 0, n - 1)
    bigram, w = mode == "bigram", xres * 2 if mode == "bigram" else xres
    lines = []
    if border:
        lines.append("┌" + "─" * w + "┐")
    if color:
        for i in range(idx.shape[0]):
            ln = _color_seq(norm[i], bigram)
            lines.append(("│" + ln + "│") if border else ln)
    else:
        for row in cmap_arr[idx]:
            ln = "".join(row)
            lines.append(("│" + ln + "│") if border else ln)
    if border:
        lines.append("└" + "─" * w + "┘")
    if show_colorbar:
        bar = _color_seq(np.linspace(0, 1, 26), bigram) if color else "".join(cmap)
        lines.extend(["", f"{vmin:.2g} {bar} {vmax:.2g}"])
    return "\n".join(lines)


def heatmap_bigram(
    data: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    xres: int | None = None,
    yres: int | None = None,
    levels: int = 16,
    extended: bool = False,
    **kw,
) -> str:
    cmap = (CMAP_B_EXT if extended else CMAP_B).get(
        levels, (CMAP_B_EXT if extended else CMAP_B)[16]
    )
    return heatmap(data, vmin, vmax, xres, yres, cmap=cmap, mode="bigram", **kw)


def imshow(data: np.ndarray, **kw) -> None:
    print(heatmap(data, **kw))


def format_title(title: str, width: int) -> str:
    if len(title) > width:
        title = title[: width - 3] + "..."
    return title.center(width)


def format_axis_labels(
    xlabel: str | None = None,
    ylabel: str | None = None,
    width: int = 64,
) -> tuple[str, str]:
    x_line = xlabel.center(width) if xlabel else ""
    y_line = ylabel if ylabel else ""
    return x_line, y_line


def heatmap_with_labels(
    data: np.ndarray,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    xres: int = 64,
    yres: int = 32,
    show_colorbar: bool = True,
    **kw,
) -> str:
    lines = []
    if title:
        lines.append(format_title(title, xres))
        lines.append("")
    heat = heatmap(
        data, vmin=vmin, vmax=vmax, xres=xres, yres=yres, show_colorbar=show_colorbar, **kw
    )
    heat_lines = heat.split("\n")
    if ylabel:
        mid = len([l for l in heat_lines if l and not l.startswith(" ")]) // 2
        for i, line in enumerate(heat_lines):
            if i == mid:
                lines.append(f"{ylabel} {line}")
            else:
                lines.append(" " * (len(ylabel) + 1) + line)
    else:
        lines.extend(heat_lines)
    if xlabel:
        lines.append(xlabel.center(xres + (len(ylabel) + 1 if ylabel else 0)))
    return "\n".join(lines)
