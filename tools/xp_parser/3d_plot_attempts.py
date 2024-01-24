### {{{                        --     cube smooth attempts    --

dmanid = net_id_to_dman_id[net_id]
network = dman.get_networks()[dmanid]
x, y = dman.get_X()[dmanid], dman.get_Y()[dmanid]
rescaler = pu.DataRescaler.from_data_manager(dman)
porder, pnames = pu.get_reordered_protein_names(network)
plot_config = BASE_DEFAULT_CONFIG
plot_config = ut.updated_dict(plot_config, DEFAULT_3D_CONFIG)
nslices = len(plot_config['slices'])
# fig, axes = pu.mkfig(1, nslices, size=plot_config['size'])
slices = (0.1, 0.3, 0.5)
slice_images = []

import matplotlib.transforms as mtransforms
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
from tempfile import mkdtemp
from os import path


cmap = pu.DEFAULT_CMAP
bad_color = '#EEEEEE00'
res = 100
vlims = (-0.027, 0.85)
contours = 3
kw = {}


def style_3d(ax):
    fig = ax.get_figure()
    fig.patch.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # long thin ticks
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.25)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.get_xaxis().tick_bottom()
    # font
    ax.tick_params(axis='both', which='both', labelsize=8)
    ax.tick_params(axis='both', which='major', length=5, width=0.4)
    ax.tick_params(axis='both', which='minor', length=2, width=0.12)
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)
    # tick outside
    ax.tick_params(axis='both', which='both', direction='out')
    # spine color
    ax.spines['bottom'].set_color('#777777')
    ax.spines['left'].set_color('#777777')


for which_slice in range(3):
    # ax = axes[0]
    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    style_3d(ax)

    for i, s in enumerate(slices):
        xslice = np.asarray([s])

        protein_order, protein_names = pu.get_reordered_protein_names(network, **kw)
        input_order, output_pos = protein_order[:-1], protein_order[-1]
        input_names, output_name = protein_names[:-1], protein_names[-1]
        xy_grid, output_values, opacities = pu.prepare_smooth_2d(
            x, y, network, input_names, input_order, output_pos, res, xlims, xslice, **kw
        )

        cmap = plt.get_cmap(cmap)
        cmap.set_bad(color=bad_color)

        xres = len(np.unique(xy_grid[:, 0]))
        yres = len(np.unique(xy_grid[:, 1]))

        xlims = np.array([xy_grid[:, 0].min(), xy_grid[:, 0].max()])
        ylims = np.array([xy_grid[:, 1].min(), xy_grid[:, 1].max()])

        vmin, vmax = vlims
        vmin = vmin if vmin is not None else np.nanmin(output_values)
        vmax = vmax if vmax is not None else np.nanmax(output_values)

        Z = output_values.reshape((xres, yres))
        opacities = np.ones_like(Z) if opacities is None else opacities.reshape((xres, yres))

        X, Y = np.meshgrid(np.linspace(*xlims, num=xres), np.linspace(*ylims, num=yres))

        Z_coord = np.ones_like(X) * s

        # Create an RGBA color array
        colors = cmap(Z / vmax)
        alpha_multiplier = 1 if (i == which_slice) else 0.0
        colors[..., -1] *= alpha_multiplier * opacities

        # Plot the surface with the RGBA colors
        ax.plot_surface(X, Y, Z_coord, facecolors=colors, rstride=1, cstride=1)

        # Add contour lines if needed
        # if contours is not None:
        # ax.contour(X, Y, Z, zdir='y', offset=s, levels=contours, linestyles="solid", linewidths=0.25, alpha=alpha_multiplier*0.5)

        ax.invert_zaxis()
        ax.view_init(elev=30, azim=25, vertical_axis='y')
        pu.setup_transformed_axis(ax, xlims, ylims, rescaler)
        # no grid:
        ax.grid(False)


# def setup_transformed_axis(
# ax, xaxis_lims=None, yaxis_lims=None, rescaler=None, margins=0.05, transform=None, **kw
# ):
# if xaxis_lims is not None:
# xaxis_lims = setup_transformed_xaxis(ax, xaxis_lims, rescaler, margins=margins, **kw)
# if yaxis_lims is not None:
# yaxis_lims = setup_transformed_yaxis(ax, yaxis_lims, rescaler, margins=margins, **kw)
# return xaxis_lims, yaxis_lims


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                    --     manual smooth cube     --

plot_config = BASE_DEFAULT_CONFIG
dmanid = net_id_to_dman_id[net_id]
network = dman.get_networks()[dmanid]
x, y = dman.get_X()[dmanid], dman.get_Y()[dmanid]
rescaler = pu.DataRescaler.from_data_manager(dman)
porder, pnames = pu.get_reordered_protein_names(network)
plot_config = BASE_DEFAULT_CONFIG
plot_config = ut.updated_dict(plot_config, DEFAULT_3D_CONFIG)
plot_config['slices'] = (0.3,)
nslices = len(plot_config['slices'])
fig, axes = pu.mkfig(1, nslices, size=plot_config['size'])
if not isinstance(axes, list):
    axes = [axes]
slice_images = []
pu.network_plot(dman, dmanid, ax=ax, axes=axes, **plot_config)

##────────────────────────────────────────────────────────────────────────────}}}
