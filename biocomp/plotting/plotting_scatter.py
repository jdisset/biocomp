# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Network-aware 3D scatter (matplotlib + plotly). Generic density helpers live in jeanplot.plots.scatter."""

import numpy as np
import matplotlib.pyplot as plt

from .plotting_core import get_reordered_protein_names, network_ticks_and_labels


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

    input_order, output_pos, input_names, output_name = get_reordered_protein_names(network, **kw)

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


