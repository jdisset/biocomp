# {{{                          --     imports     --
# ···············································································
from typing import Optional, Sequence
from numpy.typing import NDArray as NdArray
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
from . import plotting_core as pc
from .plotting_core import (
    setup_transformed_axis,
    get_reordered_protein_names,
    network_ticks_and_labels,
)

configurable = pc.configurable
##────────────────────────────────────────────────────────────────────────────}}}


# ---- scatter plots
### {{{                            --     3D     --
def scatter_3d_interactive(
    x,
    y,
    network,
    rescaler,
    *,
    xlims=(0, 1),
    title=None,
    size=10,
    colorbar=True,
    lw=0.01,
    filename=None,
    **kw,
):
    import plotly.graph_objects as go
    import plotly.offline as pyo

    xmin, xmax = xlims  # noqa: F841

    input_order, output_pos, input_names, output_name = get_reordered_protein_names(
        network, **kw
    )

    random_order = np.random.permutation(len(x))
    y = y[random_order, output_pos]
    x = x[random_order][:, input_order]

    fig = go.Figure()

    scatter = go.Scatter3d(
        x=x[:, 0],
        y=x[:, 1],
        z=x[:, 2],
        mode="markers",
        marker=dict(
            size=size, color=y, colorscale="YlGnBu", opacity=1, line=dict(color="black", width=lw)
        ),
    )

    fig.add_trace(scatter)

    fig.update_layout(
        scene=dict(
            xaxis_title=input_names[0],
            yaxis_title=input_names[1],
            zaxis_title=input_names[2],
            xaxis=dict(
                showspikes=False, showbackground=False
            ),  # tickvals=ticks, ticktext=ticklabels),
            yaxis=dict(
                showspikes=False, showbackground=False
            ),  # tickvals=ticks, ticktext=ticklabels),
            zaxis=dict(
                showspikes=False, showbackground=False
            ),  # tickvals=ticks, ticktext=ticklabels),
        ),
        width=1000,
        height=800,
    )

    if colorbar:
        cbar_trace = go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode="markers",
            marker=dict(
                size=0,
                cmin=y.min(),
                cmax=y.max(),
                colorscale="YlGnBu",
                showscale=True,
                colorbar=dict(title=output_name),  # tickvals=ticks, ticktext=ticklabels),
            ),
        )

        fig.add_trace(cbar_trace)

    ttle = None
    if title is True:
        ttle = f"{network.name}\n{output_name} smoothed mean"
    elif title is not None:
        ttle = title
    if ttle is not None:
        fig.update_layout(title=ttle)

    if filename is None:
        return pyo.plot(fig, auto_open=True)
    else:
        return pyo.plot(fig, filename=filename, auto_open=False)


def scatter_3d(
    x,
    y,
    network,
    rescaler,
    fig,
    n_views,
    xmin=0,
    xmax=1,
    title=None,
    size=10,
    lw=0.1,
    **kw,
):
    (
        input_order,
        output_pos,
        input_names,
        output_name,
        ticks,
        ticklabels,
        secondticks,
    ) = network_ticks_and_labels(network, rescaler, xmax=xmax, **kw)

    cmap = plt.get_cmap("YlGnBu")
    random_order = np.random.permutation(len(x))
    y = y[random_order, output_pos]
    x = x[random_order][:, input_order]

    azim_values = np.linspace(0, 270, n_views)

    for i, azim in enumerate(azim_values):
        ax = fig.add_subplot(1, n_views, i + 1, projection="3d")
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y, cmap=cmap, s=size, lw=lw, edgecolor="k")
        ax.set_xlabel(input_names[0])
        ax.set_ylabel(input_names[1])
        ax.set_zlabel(input_names[2])

        if len(ticks) > 0:
            sc_ticks = ticks
            ax.set_xticks(sc_ticks)
            ax.set_xticklabels(ticklabels)
            ax.set_yticks(sc_ticks)
            ax.set_yticklabels(ticklabels)
            ax.set_zticks(sc_ticks)
            ax.set_zticklabels(ticklabels)

        ttle = None

        if title is True:
            ttle = f"{network.name}\n{output_name} smoothed mean"
        elif title is not None:
            ttle = title
        if ttle is not None:
            ax.set_title(ttle)

        # Rotate the axes
        ax.view_init(elev=10, azim=azim)


##────────────────────────────────────────────────────────────────────────────}}}


def make_density_cmap(name=None, alpha_start=1.0, alpha_end=1.0, base_cmap="Spectral_r"):
    """Create a custom colormap for density visualization."""
    ncolors = 256
    color_array = plt.get_cmap(base_cmap)(range(ncolors))
    color_array[:, -1] = np.linspace(alpha_start, alpha_end, ncolors)
    map = LinearSegmentedColormap.from_list(name=name, colors=color_array)
    map.set_under("w", alpha=0)
    return map


DEFAULT_DENSITY_CMAP = make_density_cmap("density", alpha_start=1.0, alpha_end=1.0)


@configurable
def grid_histogram(
    X: NdArray,
    Y: NdArray,
    input_names: Sequence[str],
    output_name: str,
    rescaler,
    ax,
    title: Optional[str] = None,
    xtitle: Optional[str] = None,
    ytitle: Optional[str] = None,
    xlims=(None, None),
    ylims=(None, None),
    vlims=(0, None),
    draw_xlabel=True,
    draw_ylabel=True,
    res=300,  # bins per unit
    draw_colorbar=True,
    use_log_density=True,
    cmap=None,
    margins=0.01,
    noise_smooth=0.25,
    colorbar_params: dict = None,
):
    if colorbar_params is None:
        colorbar_params = {}
    assert X.shape[1] == 1
    assert Y.shape[1] == 1

    mask = ~(np.isnan(X) | np.isnan(Y))
    X = X[mask]
    Y = Y[mask]

    xmin, xmax = np.min(X), np.max(X)
    ymin, ymax = np.min(Y), np.max(Y)

    xmin = xmin if xlims[0] is None else xlims[0]
    xmax = xmax if xlims[1] is None else xlims[1]
    ymin = ymin if ylims[0] is None else ylims[0]
    ymax = ymax if ylims[1] is None else ylims[1]
    xmargins = margins * (xmax - xmin)
    ymargins = margins * (ymax - ymin)
    xmin -= xmargins
    xmax += xmargins
    ymin -= ymargins
    ymax += ymargins

    nbins_x = int(res * (xmax - xmin))
    nbins_y = int(res * (ymax - ymin))

    if noise_smooth > 0:
        xres = (xmax - xmin) / nbins_x
        yres = (ymax - ymin) / nbins_y
        X = X + np.random.normal(size=X.shape) * noise_smooth * xres
        Y = Y + np.random.normal(size=Y.shape) * noise_smooth * yres

    h, xedges, yedges = np.histogram2d(
        X,
        Y,
        bins=[nbins_x, nbins_y],
        range=[[xmin, xmax], [ymin, ymax]],
        density=False,
    )
    h = np.ma.masked_where(h == 0, h)

    from biocomp.datautils import IdentityRescaler, LogPlusOneRescaler

    density_rescaler = IdentityRescaler() if not use_log_density else LogPlusOneRescaler()

    h = density_rescaler.fwd(h)

    if cmap is None:
        cmap = DEFAULT_DENSITY_CMAP

    setup_transformed_axis(
        ax,
        xaxis_lims=[xmin, xmax],
        yaxis_lims=[ymin, ymax],
        rescaler=rescaler,
        margins=0.0,
    )

    im = ax.imshow(
        h.T,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vlims[0],
        vmax=vlims[1],
        interpolation="nearest",
    )

    ax.set_clip_path(ax.patch)

    if draw_xlabel:
        ax.set_xlabel(xtitle if xtitle is not None else input_names[0])
    if draw_ylabel:
        ax.set_ylabel(ytitle if ytitle is not None else output_name)
    if title is not None:
        ax.set_title(title)

    # show grid, including minor grid
    ax.grid(color="k", alpha=0.25, linestyle="-", linewidth=0.2, which="major")
    ax.grid(color="k", alpha=0.1, linestyle="-", linewidth=0.1, which="minor")

    if draw_colorbar:
        from biocomp.plotting.plotting_smooth import colorbar

        cbar = colorbar(
            ax,
            im,
            density_rescaler,
            vlims,
            **{**colorbar_params, "label": "Density"},
        )

    return im, cbar
