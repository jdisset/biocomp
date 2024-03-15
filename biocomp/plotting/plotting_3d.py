## {{{                          --     imports     --
import numpy as np
from .plotting_core import (
    mkfig,
    get_reordered_protein_names,
    format_powers,
    default_style,
    setup_transformed_axis,
    get_transformed_ticks_and_labels,
    extract_plot_data_from_network,
)
from .plotting_smooth import (
    smooth_2d,
)

from . import plotting_core as pc
from typing import Union, Sequence, List, Tuple, Dict, Any, Optional, Callable
from matplotlib import pyplot as plt
from functools import partial
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import numpy as np
from biocomp import plotutils as pu
from biocomp import utils as ut

import jax.numpy as jnp

NdArray = Union[np.ndarray, jnp.ndarray]
configurable = pc.configurable


# to get the plt.Axes type:
from matplotlib.axes import Axes

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                         --     3d misc --

# TODO: move to config system

CUBE_SPINE_PROPS = dict(lw=0.5, color='#888888', ls='-')
CUBE_SPINE_PROPS_HIDDEN = ut.updated_dict(CUBE_SPINE_PROPS, dict(ls=':', alpha=0.5))


def plot_face(ax, visible_spines=('bottom', 'left'), hidden_spines=('top', 'right')):
    if hidden_spines is None:
        hidden_spines = []
    if visible_spines is None:
        visible_spines = []
    # not ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks([], minor=True)
    ax.set_yticks([], minor=True)
    # remove facecolor
    ax.patch.set_facecolor('none')
    # linewidth and color of spines using the CUBE_SPINE_PROPS
    for spine in ax.spines.values():
        spine.set_visible(False)
    for spine in visible_spines:
        spine = ax.spines[spine]
        spine.set_visible(True)
        spine.set(**CUBE_SPINE_PROPS)
    for spine in hidden_spines:
        spine = ax.spines[spine]
        spine.set_visible(True)
        spine.set(**CUBE_SPINE_PROPS_HIDDEN)


def to_display_units(x, ax):
    """Convert x from data units to display units"""
    ppd = 72.0 / ax.figure.dpi
    trans = ax.transData.transform
    return ((trans((1, x)) - trans((0, 0))) * ppd)[1]


def to_data_units(y_display, ax):
    """Convert y from display units to data units"""
    ppd = 72.0 / ax.figure.dpi
    trans_inv = ax.transData.inverted().transform
    origin = trans_inv((0, 0))
    point_in_data_units = trans_inv((0, y_display / ppd))
    return point_in_data_units[1] - origin[1]


def cabinet_project(x, y, z, alpha=np.pi / 4, d=0.5):
    return np.array([x + d * z * np.cos(alpha), y + d * z * np.sin(alpha)])


PROJ_ALPHA = np.pi / 4
PROJ_D = 0.5

DEFAULT_SLICE_TICKS_PROPS = [
    {
        'length': 54,
        'direction': (1, 0, 0),
        'props': dict(color='k', lw=0.2, dashes=[5, 5], alpha=0.5),
    }
]


DEFAULT_MAJOR_TICKS_PROPS = [
    {'length': 8, 'direction': (1, 0, 0), 'props': dict(color='k', lw=0.4)},
]
DEFAULT_MINOR_TICKS_PROPS = [
    {'length': 2, 'direction': (1, 0, 0), 'props': dict(color='k', lw=0.2)},
]

max_int = np.iinfo(np.int32).max
CUBE_EDGE_PROPS_VISIBLE = {
    'props': {**CUBE_SPINE_PROPS, 'zorder': +max_int - 10},
    'offset': (0, 0),
}

CUBE_EDGE_PROPS_HIDDEN = {
    'props': {**CUBE_SPINE_PROPS_HIDDEN, 'zorder': -max_int + 10},
    'offset': (0, 0),
}

DEFAULT_LABEL_PROPS = dict(
    ha='left',
    va='center',
    fontsize=8,
    bbox=dict(facecolor='white', alpha=1, edgecolor='none', pad=-0.25),
)
DEFAULT_SLICE_LABEL_PROPS = dict(ha='left', va='center', fontsize=7)
DEFAULT_TITLE_PROPS = dict(ha='center', va='center', fontsize=8, rotation=PROJ_ALPHA * 180 / np.pi)

DEFAULT_CUBE_EDGE_PROPS = {
    'br': {
        **CUBE_EDGE_PROPS_VISIBLE,
        'offset': (0.0, 0),  # percentage of axes units
        'ticks': {
            'major': DEFAULT_MAJOR_TICKS_PROPS,
            'minor': DEFAULT_MINOR_TICKS_PROPS,
            'slice': DEFAULT_SLICE_TICKS_PROPS,
        },
        'labels': {
            'major': {'offset': (10, 0), 'props': DEFAULT_LABEL_PROPS},
            'slice': {'offset': (55, 0), 'props': DEFAULT_SLICE_LABEL_PROPS},
        },
        'title': {'offset': (40, 0), 'props': DEFAULT_TITLE_PROPS},
    },
    'bl': CUBE_EDGE_PROPS_HIDDEN,
    'tl': CUBE_EDGE_PROPS_VISIBLE,
    'tr': CUBE_EDGE_PROPS_VISIBLE,
}

plot_front = partial(
    plot_face, visible_spines=['bottom', 'left', 'top', 'right'], hidden_spines=None
)
plot_back = partial(plot_face, visible_spines=['top', 'right'], hidden_spines=['bottom', 'left'])

from typing import List, Tuple, Dict, Any, Optional, Sequence, Union

project = partial(cabinet_project, alpha=PROJ_ALPHA, d=PROJ_D)


def draw_tick(ax, position, direction, length, props):
    # position and direction are in 3d world coordinates
    # length is in display units
    position, direction = np.asarray(position), np.asarray(direction)
    length = to_data_units(length, ax)
    tproj_start = project(*position)
    tproj_end = project(*(position + direction * length))
    ax.plot([tproj_start[0], tproj_end[0]], [tproj_start[1], tproj_end[1]], **props)


def draw_text(ax, position, label, offset, props):
    position = np.asarray(position)
    offset = np.asarray(offset)
    offset = np.array([to_data_units(offset[0], ax), to_data_units(offset[1], ax)])
    tproj_position = project(*position)
    t = ax.text(tproj_position[0] + offset[0], tproj_position[1] + offset[1], label, **props)
    if 'bbox' in props:
        t.set_bbox(props['bbox'])


def draw_z_axis(ax, xpos, ypos, zlim, axis_offset, props=CUBE_SPINE_PROPS, **_):
    xo, yo = axis_offset
    zcoords_world = np.array([[xpos + xo, xpos + xo], [ypos + yo, ypos + yo], zlim])
    zcoords_proj = np.array((project(*zcoords_world[:, 0]), project(*zcoords_world[:, 1])))
    ax.plot(zcoords_proj[:, 0], zcoords_proj[:, 1], **props)


def draw_z_ticks_along_axis(
    ax,
    xpos,
    ypos,
    axis_offset=None,
    ticks: Optional[Sequence[float]] = None,
    tick_props: Optional[Union[List, Dict[str, Any]]] = None,
    **_,
):
    xpos, ypos = np.array([xpos, ypos]) + axis_offset
    if ticks is not None and tick_props is not None:
        for tick in ticks:
            if isinstance(tick_props, dict):
                tick_props = [tick_props]
            for tick_prop in tick_props:
                draw_tick(ax, (xpos, ypos, tick), **tick_prop)


def draw_z_labels(ax, xpos, ypos, axis_offset, labels, **props):
    xpos, ypos = np.array([xpos, ypos]) + axis_offset
    assert isinstance(labels, list)
    for z, label, ltype in labels:
        assert ltype in props
        draw_text(ax, position=np.array([xpos, ypos, z]), label=label, **props[ltype])


def draw_z_title(
    ax,
    xpos,
    ypos,
    zlim,
    axis_offset=None,
    z_title=None,
    **title_props,
):
    xpos, ypos = np.array([xpos, ypos]) + axis_offset
    if title_props is not None and z_title is not None:
        tpos_world = np.array([xpos, ypos, np.mean(zlim)])  # center of the axis
        draw_text(ax, position=tpos_world, label=z_title, **title_props)


def main_ax_lims(ax, xlim, ylim, set_lims=True):
    all_xlims = [ax.get_xlim(), xlim]
    all_ylims = [ax.get_ylim(), ylim]
    ax_lims = np.array(
        [[np.min(all_xlims), np.max(all_xlims)], [np.min(all_ylims), np.max(all_ylims)]]
    )
    # add 5% padding
    ax_lims += np.array([[-1, 1], [-1, 1]]) * 0.05 * np.abs(ax_lims[:, 1] - ax_lims[:, 0])
    ax_size = np.abs(ax_lims[:, 1] - ax_lims[:, 0])
    if set_lims:
        ax.axis('off')
        ax.set_aspect('equal')
        ax.set_xlim(ax_lims[0])
        ax.set_ylim(ax_lims[1])
    return ax_lims, ax_size


def get_edge_pos(edge, xlim, ylim):
    return (xlim[0] if edge[1] == 'l' else xlim[1], ylim[0] if edge[0] == 'b' else ylim[1])


def to_ax_coords(x, y, ax_lims, ax_size):
    return (x - ax_lims[0, 0]) / ax_size[0], (y - ax_lims[1, 0]) / ax_size[1]


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                    --     actual 3d function     --

# uses the previous code to plot the 3d slices


def cabinet_project(x, y, z, alpha=np.pi / 4, d=0.5):
    return np.array([x + d * z * np.cos(alpha), y + d * z * np.sin(alpha)])


def get_axis_offsets(cube_edge_props, xlim, ylim):
    axis_offsets = {}
    for edge, props in cube_edge_props.items():
        if 'offset' in props:
            axis_offset = props.pop('offset', None)
            axis_offset = np.array(0, 0) if axis_offset is None else np.array(axis_offset)
            axis_offset = axis_offset * np.array((xlim[1] - xlim[0], ylim[1] - ylim[0]))
            axis_offsets[edge] = axis_offset
        else:
            axis_offsets[edge] = np.array([0, 0])
    return axis_offsets


def plot_3d_slices(
    ax,
    slice_functions: List,
    slice_zpositions: List,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    zlim: Tuple[float, float],
    zticks: Dict[str, List[float]],
    zlabels: Optional[List[Tuple[float, str, str]]] = None,
    cube_edge_props: Dict[str, Any] = DEFAULT_CUBE_EDGE_PROPS,
    z_title: Optional[str] = None,
    **kwargs,
):

    if zlabels is None:
        if zticks is not None and 'major' in zticks:
            zlabels = [(z, str(z), 'major') for z in zticks['major']]

    axis_offsets = get_axis_offsets(cube_edge_props, xlim, ylim)

    for edge, props in cube_edge_props.items():
        axis_offset = axis_offsets.get(edge, None)
        draw_z_axis(ax, *get_edge_pos(edge, xlim, ylim), zlim, axis_offset=axis_offset, **props)

    ax_lims, ax_size = main_ax_lims(ax, xlim, ylim)

    for edge, props in cube_edge_props.items():
        axis_offset = axis_offsets.get(edge, None)
        if 'ticks' in props:
            for ztype, ticks in zticks.items():
                if ztype in props['ticks']:
                    draw_z_ticks_along_axis(
                        ax,
                        *get_edge_pos(edge, xlim, ylim),
                        axis_offset=axis_offset,
                        ticks=ticks,
                        tick_props=props['ticks'][ztype],
                    )
        if 'title' in props:
            draw_z_title(
                ax,
                *get_edge_pos(edge, xlim, ylim),
                zlim,
                z_title=z_title,
                axis_offset=axis_offset,
                **props['title'],
            )
        if props.get('labels') is not None and zlabels is not None:
            assert isinstance(zlabels, list)
            draw_z_labels(
                ax,
                *get_edge_pos(edge, xlim, ylim),
                labels=zlabels,
                axis_offset=axis_offset,
                **props['labels'],
            )

    # plot the slices
    for i, (f, z) in enumerate(zip(slice_functions, slice_zpositions)):
        axin = ax.inset_axes([0, 0, 1, 1], zorder=-z)
        pc.default_style(axin)
        f(axin)
        inset_coords_world = np.array([axin.get_xlim(), axin.get_ylim()])
        inset_size_world = np.abs(inset_coords_world[:, 1] - inset_coords_world[:, 0])
        inset_size_ax = inset_size_world / ax_size
        # project the z value
        inset_coords_proj = project(inset_coords_world[0, 0], inset_coords_world[1, 0], z)
        inset_coords_ax = to_ax_coords(*inset_coords_proj, ax_lims, ax_size)
        # bit of a hack but it's to avoid inset_axes ignoring set_position after creation (mpl v3.8)
        ip = InsetPosition(
            ax, [inset_coords_ax[0], inset_coords_ax[1], inset_size_ax[0], inset_size_ax[1]]
        )
        axin.set_axes_locator(ip)


##────────────────────────────────────────────────────────────────────────────}}}


def cube_face(
    ax,
    xlims,
    ylims,
    facecolor='none',
    visible_spines=('bottom', 'left'),
    hidden_spines=('top', 'right'),
):
    if hidden_spines is None:
        hidden_spines = []
    if visible_spines is None:
        visible_spines = []
    pc.default_style(ax)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    # not ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks([], minor=True)
    ax.set_yticks([], minor=True)
    ax.patch.set_facecolor(facecolor)
    # linewidth and color of spines using the CUBE_SPINE_PROPS
    for spine in ax.spines.values():
        spine.set_visible(False)
    for spine in visible_spines:
        spine = ax.spines[spine]
        spine.set_visible(True)
        spine.set(**CUBE_SPINE_PROPS)
    for spine in hidden_spines:
        spine = ax.spines[spine]
        spine.set_visible(True)
        spine.set(**CUBE_SPINE_PROPS_HIDDEN)


# def get_protein_name(name, protein_aliases=None):
# name = name.upper()
# if protein_aliases is not None:
# return protein_aliases.get(name, name)


def front_face_bl(
    ax,
    xlims,
    ylims,
    input_names: Sequence[str],
    rescaler: pc.DataRescaler,
    labelpad=True,
    ticks=False,
):
    # in order to get the correct zorder,
    # we plot the face in 2 steps, first the ones that can be covered,
    # then the ones that cover
    max_int = np.iinfo(np.int32).max
    cube_face(
        ax,
        xlims,
        ylims,
        facecolor='none',
        visible_spines=['bottom', 'left'],
        hidden_spines=[],
    )
    ax.set_xlabel(input_names[0])
    ax.set_ylabel(input_names[1])
    ax.set_zorder(-max_int)

    if ticks:
        setup_transformed_axis(
            ax,
            xaxis_lims=xlims,
            yaxis_lims=ylims,
            rescaler=rescaler,
            margins=0.0,
        )
    # add margin to labels
    if labelpad and not ticks:
        ax.xaxis.labelpad = 18
        ax.yaxis.labelpad = 22


def front_face_tr(
    ax,
    xlims,
    ylims,
):
    max_int = np.iinfo(np.int32).max
    cube_face(
        ax,
        xlims,
        ylims,
        facecolor='none',
        visible_spines=['top', 'right'],
        hidden_spines=[],
    )
    ax.set_zorder(max_int - 10)


def back_face(ax, xlims, ylims):
    cube_face(
        ax,
        xlims,
        ylims,
        facecolor='none',
        visible_spines=['top', 'right'],
        hidden_spines=['bottom', 'left'],
    )


@configurable
def smooth_3d(
    X: NdArray,
    Y: NdArray,
    input_names: Sequence[str],
    output_name: str,
    rescaler: pc.DataRescaler,
    ax: Sequence[Axes],
    zslices: NdArray,
    xlims=(0, 1),
    ylims=(None, None),
    zlims=(None, None),
    vlims=(None, None),
    draw_colorbar: Optional[bool] = None,  # None means only last slice
    smooth_2d_params: Dict = {},
    **kw,
):

    assert isinstance(ax, (list, tuple, np.ndarray))

    if len(ax) != len(zslices):
        raise ValueError(
            f'axes and slices must have the same length, got {len(ax)} and {len(zslices)}'
        )

    ylims = xlims if ylims == (None, None) else ylims
    zlims = xlims if zlims == (None, None) else zlims

    def format_value(x: float):
        return f'{format_powers(rescaler.inv(x), n_decimals=0)}'

    def plot_slice(sl_ax: Axes, i: int):
        colorbar = (i == len(ax) - 1) if draw_colorbar is None else draw_colorbar
        im, contour = smooth_2d(
            X,
            Y,
            input_names,
            output_name,
            rescaler=rescaler,
            ax=sl_ax,
            zslice=np.array([zslices[i]]),
            xlims=xlims,
            ylims=ylims,
            vlims=vlims,
            **smooth_2d_params,
        )
        if contour is not None:
            sl_ax.clabel(contour, inline_spacing=20, fontsize=5, fmt=format_value)
        # remove x and y labels
        sl_ax.set_xlabel('')
        sl_ax.set_ylabel('')
        sl_ax.set_facecolor('none')
        # zorder of ticklabels:
        sl_ax.yaxis.label.set_zorder(2)
        sl_ax.xaxis.label.set_zorder(2)

        for label in sl_ax.get_xticklabels() + sl_ax.get_yticklabels():
            label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=1, pad=0.75, zorder=1.5))

        if colorbar:
            # put it on the right, with a lot of padding
            # create an inset axis
            cax = sl_ax.inset_axes((1.1, 0.4, 0.04, 0.52))

            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=6)
            pc.default_style(cbar.ax)
            cbar.ax.tick_params(axis='both', which='both', direction='out', pad=2, labelsize=8)
            # ticks and labels to the right
            cbar.ax.yaxis.tick_right()
            cbar.ax.yaxis.set_label_position('left')

            # add title to the right, vertical, along the colorbar
            # using the y axis for labels

            cbar.ax.set_ylabel(output_name, rotation=90, fontsize=8, labelpad=5)

            for spine in cbar.ax.spines.values():
                spine.set_linewidth(0.2)

            vmin, vmax = im.get_clim()
            setup_transformed_axis(
                cbar.ax,
                yaxis_lims=[vmin, vmax],
                rescaler=rescaler,
                margins=0.0,
                # **kw,
            )

    zticks, zlabels = get_transformed_ticks_and_labels(
        zlims + np.array((0.1, 0)), rescaler=rescaler, skip_ticklabel_range=(-10, 2000)
    )

    major_zlabels: List[Tuple[float, str, str]] = [(float(z), s, 'major') for z, s in zlabels]

    # remove first major ticks
    zticks['major'] = np.asarray(zticks['major'])
    zticks['major'] = zticks['major'][zticks['major'] > 0.0]
    # same for minor ticks
    zticks['minor'] = np.asarray(zticks['minor'])
    zticks['minor'] = zticks['minor'][zticks['minor'] > 0.0]

    import copy

    for i, s in enumerate(zslices):
        # now add a special tick for the slices
        s_zticks = copy.deepcopy(zticks)
        s_zlabels = copy.deepcopy(major_zlabels)
        s_zticks['slice'] = [s]
        slice_label = f'$ \\approx $ {format_powers(rescaler.inv(s), n_decimals=0)}'
        s_zlabels.append((s, slice_label, 'slice'))
        plot_3d_slices(
            ax[i],
            [
                partial(
                    front_face_bl,
                    labelpad=i == 0,
                    xlims=xlims,
                    ylims=ylims,
                    input_names=input_names,
                    rescaler=rescaler,
                ),
                partial(front_face_tr, xlims=xlims, ylims=ylims),
                partial(plot_slice, i=i),
                partial(back_face, xlims=xlims, ylims=ylims),
            ],
            [zlims[0], zlims[0], s, zlims[1]],
            xlim=xlims,
            ylim=ylims,
            zlim=zlims,
            zticks=s_zticks,
            zlabels=s_zlabels,
            cube_edge_props=DEFAULT_CUBE_EDGE_PROPS,
            z_title=input_names[2],
        )


def smooth_3d_old(
    X,
    Y,
    network,
    rescaler,
    slices=np.linspace(0, 0.65, 4),
    axes=None,
    top_ax=None,
    input_order=None,
    protein_aliases=None,
    **kw,
):

    max_int = np.iinfo(np.int32).max
    assert axes is not None
    if len(axes) != len(slices):
        raise ValueError(
            f'axes and slices must have the same length, got {len(axes)} and {len(slices)}'
        )

    porder, pnames = get_reordered_protein_names(network, **kw)
    kw.pop('ax', None)
    xlims, ylims = kw['xlims'], kw['ylims']
    zlims = kw.get('zlims', kw['xlims'])

    x, y, input_names, output_name = extract_plot_data_from_network(
        network, X, Y, input_order, protein_aliases
    )

    def format_value(x):
        return f'{format_powers(rescaler.inv(x), n_decimals=0)}'

    def plot_slice(ax, i):
        colorbar = i == len(axes) - 1
        im, contour, vmin, vmax = smooth_2d(
            x,
            y,
            network,
            rescaler,
            ax=ax,
            zslice=np.array([slices[i]]),
            draw_colorbar=False,
            rescaler=rescaler,
            # **kw,
        )
        if contour is not None:
            ax.clabel(contour, inline_spacing=20, fontsize=5, fmt=format_value)
        # remove x and y labels
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_facecolor('none')
        # zorder of ticklabels:
        ax.yaxis.label.set_zorder(2)
        ax.xaxis.label.set_zorder(2)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=1, pad=0.75, zorder=1.5))

        if colorbar:
            # put it on the right, with a lot of padding
            # create an inset axis
            cax = axes[i].inset_axes([1.1, 0.4, 0.04, 0.52])

            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=6)
            pc.default_style(cbar.ax)
            cbar.ax.tick_params(axis='both', which='both', direction='out', pad=2, labelsize=8)
            # ticks and labels to the right
            cbar.ax.yaxis.tick_right()
            cbar.ax.yaxis.set_label_position('left')

            # add title to the right, vertical, along the colorbar
            # using the y axis for labels

            cbar.ax.set_ylabel(output_name, rotation=90, fontsize=8, labelpad=5)

            for spine in cbar.ax.spines.values():
                spine.set_linewidth(0.2)

            setup_transformed_axis(
                cbar.ax,
                yaxis_lims=[vmin, vmax],
                rescaler=rescaler,
                margins=0.0,
                **kw,
            )

    def front_face_bl(ax, labelpad=True, ticks=False):
        # in order to get the correct zorder,
        # we plot the face in 2 steps, first the ones that can be covered,
        # then the ones that cover
        cube_face(
            ax,
            xlims,
            ylims,
            facecolor='none',
            visible_spines=['bottom', 'left'],
            hidden_spines=[],
        )
        ax.set_xlabel(input_names[0])
        ax.set_ylabel(input_names[1])
        ax.set_zorder(-max_int)

        if ticks:
            setup_transformed_axis(
                ax,
                xaxis_lims=xlims,
                yaxis_lims=ylims,
                rescaler=rescaler,
                margins=0.0,
                **kw,
            )
        # add margin to labels
        if labelpad and not ticks:
            ax.xaxis.labelpad = 18
            ax.yaxis.labelpad = 22

    def front_face_tr(ax):
        cube_face(
            ax,
            xlims,
            ylims,
            facecolor='none',
            visible_spines=['top', 'right'],
            hidden_spines=[],
        )
        ax.set_zorder(max_int - 10)

    def back_face(ax):
        cube_face(
            ax,
            xlims,
            ylims,
            facecolor='none',
            visible_spines=['top', 'right'],
            hidden_spines=['bottom', 'left'],
        )

    zticks, zlabels = get_transformed_ticks_and_labels(
        zlims + np.array((0.1, 0)), rescaler=rescaler, skip_ticklabel_range=(-10, 2000)
    )
    zlabels = [(z, s, 'major') for z, s in zlabels]
    # remove first major ticks
    zticks['major'] = np.asarray(zticks['major'])
    zticks['major'] = zticks['major'][zticks['major'] > 0.0]
    # same for minor ticks
    zticks['minor'] = np.asarray(zticks['minor'])
    zticks['minor'] = zticks['minor'][zticks['minor'] > 0.0]

    import copy

    for i, s in enumerate(slices):
        # now add a special tick for the slices
        s_zticks = copy.deepcopy(zticks)
        s_zlabels = copy.deepcopy(zlabels)
        s_zticks['slice'] = [s]
        slice_label = f'$ \\approx $ {format_powers(rescaler.inv(s), n_decimals=0)}'
        s_zlabels.append((s, slice_label, 'slice'))
        plot_3d_slices(
            axes[i],
            [
                partial(front_face_bl, labelpad=i == 0),
                front_face_tr,
                partial(plot_slice, i=i),
                back_face,
            ],
            [zlims[0], zlims[0], s, zlims[1]],
            xlim=xlims,
            ylim=ylims,
            zlim=zlims,
            zticks=s_zticks,
            zlabels=s_zlabels,
            cube_edge_props=DEFAULT_CUBE_EDGE_PROPS,
            z_title=input_names[2],
        )
