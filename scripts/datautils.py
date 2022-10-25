## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     imports     --
# ···············································································
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import biocomp as bc
import scriptutils as ut
from pathlib import Path
import json5
import json
import sqlite3
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     binstats     --
# ···············································································


def binstats_nbins(data, bin_columns, stat_column, nbins=20, log=True, stats=["mean", "count"]):
    vmin, vmax = data[:, bin_columns].min(axis=0), data[:, bin_columns].max(axis=0)
    vmin = np.ones_like(vmin)
    if log:
        bins = np.geomspace(vmin, vmax, nbins)
    else:
        bins = np.linspace(vmin, vmax, nbins)
    coords = np.array(
        [np.digitize(data[:, i], bins[:, b], right=True) for b, i in enumerate(bin_columns)]
    ).T
    df = pd.DataFrame(data)
    df['coords'] = [tuple(x) for x in coords]
    df = df.reset_index().groupby('coords').agg({stat_column: stats, 'index': lambda x: list(x)})
    df.columns = stats + ['indices']
    return df, bins


def binstats(data, protein_names, bin_axis=None, resolution=0.5, bin_min=None, bin_max=None):
    """Calculate statistics (mean, count), for each bin in len(bin_axis) dimensions."""
    if bin_axis is None:
        bin_axis = protein_names
    bin_axisid = [protein_names.index(p) for p in bin_axis]
    POWER_RANGE = 20
    VMAX_EPSILON = 10.0**(-POWER_RANGE)
    vmin, vmax = data[:, bin_axisid].min(axis=0), data[:, bin_axisid].max(axis=0) + VMAX_EPSILON
    if bin_min is not None:
        vmin = max(vmin, bin_min)
    if bin_max is not None:
        vmax = min(vmax, bin_max)
    # we want to bin the data using logaritmic bins with a fixed resolution, not a specific number of bins
    powers = 10.0**np.arange(-POWER_RANGE, POWER_RANGE, resolution)
    first_bin = np.array([powers[powers < v].max() for v in vmin])
    last_bin = np.array([powers[powers > v].min() for v in vmax])
    nbins = np.ceil(np.log10(last_bin / first_bin) / resolution).astype(int)
    bin_edges = [np.geomspace(first_bin[i], last_bin[i], nbins[i] + 1) for i in range(len(first_bin))]
    coords = np.array([np.digitize(data[:, i], be) for i, be in zip(bin_axisid, bin_edges)]).T - 1
    df = pd.DataFrame(data, columns=protein_names)
    df['coord'] = [tuple(c) for c in coords]
    df2 = df.groupby('coord').agg(['mean']).reset_index()
    df2['count'] = df.groupby('coord').size().values
    df2['indices'] = df.reset_index().groupby('coord').agg({'index': lambda x: list(x)}).values

    for i, p in enumerate(bin_axis):
        df2[('coords',p)] = df2['coord'].apply(lambda x: x[i])
    df2 = df2.drop('coord', axis=1, level=0)
    df2 = df2.set_index([('coords', p) for p in bin_axis])

    bins = {p: be for p, be in zip(bin_axis, bin_edges)}
    return df2, bins


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


def balance_per_bin(data, statdf, threshold_quantile=0.4, threshold_min=20) -> np.ndarray:
    assert 'count' in statdf.columns
    assert 'indices' in statdf.columns
    threshold = max(
        np.quantile(statdf[statdf['count'] > 0]['count'], threshold_quantile), threshold_min
    )
    balanced_indices = []
    for i, row in statdf.iterrows():
        balanced_indices.append(
            np.random.choice(row['indices'], min(int(threshold), int(row['count'])), replace=False)
        )
    balanced_data = data[np.concatenate(balanced_indices), :]
    return balanced_data


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                       --     plot heatmap     --
# ···············································································
import string


class MyFormatter(string.Formatter):
    def format_field(self, value, format_spec):
        if format_spec == 'm':
            return super().format_field(value, '.1e').replace('e+0', 'e').replace('e+', 'e')
        else:
            return super().format_field(value, format_spec)


fmt = MyFormatter()


def heatmap(statdf, bins, stat_columns=['mean'], z_protein = None, lims = {}, figscale=1.0, count_threshold=2, cmap='YlGnBu', title=None, subtitle=None, filename=None, **kwargs):

    from matplotlib.colors import LogNorm

    fontsize = 12 * figscale

    nstats = len(stat_columns)
    figsize = (figscale * 10 * nstats + (nstats - 1) * 4 * figscale, figscale * 10)
    fig, axes = plt.subplots(1, nstats, figsize=figsize)
    if nstats == 1:
        axes = [axes]

    df = statdf[statdf['count'] >= count_threshold]


    xy_axis = [n[1] for n in df.index.names]

    if z_protein is None:
        z_axis = [c[0] for c in df.columns if c[1] and c[0] not in xy_axis]
        assert(len(z_axis) == 1)
        z_axis = z_axis[0]
    else:
        z_axis = z_protein


    for stat, ax in zip(stat_columns, axes):
        if stat == 'indices':
            continue
        # get order in which xy_axis appear in columns
        xbin, ybin = bins[xy_axis[0]], bins[xy_axis[1]]
        Z = np.full((len(xbin), len(ybin)), np.nan)
        # coords tuple is the index of the df
        statcol = 'count' if stat == 'count' else (z_axis, stat)
        for coords, value in df[statcol].items():
            Z[coords] = value

        # nans should be grey
        cmap = plt.get_cmap(cmap)
        cmap.set_bad(color='#EEEEEE')
        vmin = 0 if stat == 'count' else None

        norm = None
        if stat != 'count':
            if stat in lims:
                vmin, vmax = lims[stat]
            else:
                vmin, vmax = np.nanmin(Z), np.nanmax(Z)
            norm = LogNorm(vmin=vmin, vmax=vmax)

        im = ax.imshow(Z.T, origin='lower', norm=norm, cmap=cmap)

        ax.set_title(stat)

        ax.set_xlabel(xy_axis[0])
        ax.set_ylabel(xy_axis[1])

        # add white grid lines to separate bins
        ax.set_xticks(np.arange(len(xbin) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(ybin) + 1) - 0.5, minor=True)
        grid_thickness = 60.0 * figscale / max(len(bins), 15)
        if grid_thickness > 0.75:
            grid_thickness = max(1.25, grid_thickness)
            ax.grid(which='minor', axis='both', linestyle='-', color='w', linewidth=grid_thickness)
        ax.tick_params(which='minor', bottom=False, left=False)

        tick_freq = max(1, int(len(xbin) / 10))
        ax.set_xticks(np.arange(len(xbin))[::tick_freq], minor=False)
        ax.set_yticks(np.arange(len(ybin))[::tick_freq], minor=False)
        xl = [f"{x:.0f}" if x < 100 else fmt.format("{:m}", x) for x in xbin[::tick_freq]]
        yl = [f"{x:.0f}" if x < 100 else fmt.format("{:m}", x) for x in ybin[::tick_freq]]
        ax.set_xticklabels(xl, minor=False)
        ax.set_yticklabels(yl, minor=False)

        # remove border
        for spine in ax.spines.values():
            spine.set_visible(False)

        width = ax.get_position().width
        height = ax.get_position().height
        cax = fig.add_axes(
            [
                ax.get_position().x1 + 0.02 * figscale,
                ax.get_position().y0 + height * 0.25,
                0.017 * figscale,
                height / 2,
            ]
        )
        from matplotlib.ticker import LogFormatterSciNotation
        cb_format = None
        if norm is not None:
            cb_format = LogFormatterSciNotation()
        fig.colorbar(im, cax=cax, format=cb_format)

        # set all font sizes
        ax.tick_params(labelsize=fontsize)
        ax.title.set_fontsize(fontsize * 1.4)
        ax.xaxis.label.set_fontsize(fontsize)
        ax.yaxis.label.set_fontsize(fontsize)
        cax.tick_params(labelsize=fontsize)
        cax.yaxis.get_offset_text().set_fontsize(fontsize)

        # change font to custom (roboto)
        font = 'Roboto Mono Light for Powerline'
        for item in (
            [ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
            + cax.get_xticklabels()
            + cax.get_yticklabels()
        ):
            item.set_fontname(font)

        cax.yaxis.get_offset_text().set_fontname(font)
        ax.title.set_fontname('Arial')
        # set font weight to not bold
        ax.title.set_fontweight('light')
        ax.title.set_fontstretch('expanded')

    title = title if title is not None else f'{z_axis} stats'
    # suptitle should be higher than default
    fig.suptitle(title, fontsize=fontsize * 1.8, fontweight='light', fontstretch='expanded', fontname='Arial', y=0.99)

    if subtitle:
        # increase figure height
        # fig.set_figheight(fig.get_figheight() * 1.1)
        # write smaller, below the title, italic, centered
        fig.text(
            0.5,
            0.95,
            subtitle,
            fontsize=fontsize * 1.2,
            fontweight='light',
            fontstretch='expanded',
            fontname='Arial',
            horizontalalignment='center',
            verticalalignment='top',
            style='italic',
        )

    if filename:
        from matplotlib.transforms import Bbox
        plt.savefig(filename, dpi=300, bbox_inches=Bbox([[0.5*figscale, 0], [figsize[0], figsize[1]]]))

    plt.show()


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     archive     --
#···············································································

def heatmap_back(df, bins, axis_names=None, figscale=1.0, count_threshold=1, cmap='YlGnBu', title=None, subtitle=None, filename=None, **kwargs):

    fontsize = 12 * figscale

    nstats = len(df.columns) - (1 if 'indices' in df.columns else 0)
    figsize = (figscale * 10 * nstats + (nstats - 1) * 4 * figscale, figscale * 10)
    fig, axes = plt.subplots(1, nstats, figsize=figsize)

    df = df[df['count'] >= count_threshold]

    for stat, ax in zip(df.columns, axes):
        if stat == 'indices':
            continue
        Z = np.full((np.shape(bins)[0],) * 2, np.nan)
        # coords tuple is the index of the df
        for coords, value in df[stat].items():
            Z[coords] = value

        # nans should be grey
        cmap = plt.get_cmap(cmap)
        cmap.set_bad(color='#EEEEEE')
        vmin = 0 if stat == 'count' else None
        im = ax.imshow(Z, origin='lower', vmin=vmin, cmap=cmap)
        ax.set_title(stat)
        if axis_names:
            ax.set_xlabel(axis_names[0])
            ax.set_ylabel(axis_names[1])

        # add white grid lines to separate bins
        ax.set_xticks(np.arange(len(bins[:, 0]) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(bins[:, 1]) + 1) - 0.5, minor=True)
        grid_thickness = 60.0 * figscale / max(len(bins), 15)
        if grid_thickness > 0.75:
            grid_thickness = max(1.25, grid_thickness)
            ax.grid(which='minor', axis='both', linestyle='-', color='w', linewidth=grid_thickness)
        ax.tick_params(which='minor', bottom=False, left=False)

        tick_freq = max(1, int(len(bins[:, 0]) / 10))
        ax.set_xticks(np.arange(len(bins[:, 0]))[::tick_freq], minor=False)
        ax.set_yticks(np.arange(len(bins[:, 1]))[::tick_freq], minor=False)
        xl = [f"{x:.0f}" if x < 100 else fmt.format("{:m}", x) for x in bins[:, 0][::tick_freq]]
        yl = [f"{x:.0f}" if x < 100 else fmt.format("{:m}", x) for x in bins[:, 1][::tick_freq]]
        ax.set_xticklabels(xl, minor=False)
        ax.set_yticklabels(yl, minor=False)

        # remove border
        for spine in ax.spines.values():
            spine.set_visible(False)

        width = ax.get_position().width
        height = ax.get_position().height
        cax = fig.add_axes(
            [
                ax.get_position().x1 + 0.02 * figscale,
                ax.get_position().y0 + height * 0.25,
                0.017 * figscale,
                height / 2,
            ]
        )
        fig.colorbar(im, cax=cax)

        # set all font sizes
        ax.tick_params(labelsize=fontsize)
        ax.title.set_fontsize(fontsize * 1.4)
        ax.xaxis.label.set_fontsize(fontsize)
        ax.yaxis.label.set_fontsize(fontsize)
        cax.tick_params(labelsize=fontsize)
        cax.yaxis.get_offset_text().set_fontsize(fontsize)

        # change font to custom (roboto)
        font = 'Roboto Mono Light for Powerline'
        for item in (
            [ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
            + cax.get_xticklabels()
            + cax.get_yticklabels()
        ):
            item.set_fontname(font)

        cax.yaxis.get_offset_text().set_fontname(font)
        ax.title.set_fontname('Arial')
        # set font weight to not bold
        ax.title.set_fontweight('light')
        ax.title.set_fontstretch('expanded')

    title = title if title is not None else f'{axis_names[2]} stats'
    # suptitle should be higher than default
    fig.suptitle(title, fontsize=fontsize * 1.8, fontweight='light', fontstretch='expanded', fontname='Arial', y=0.99)

    if subtitle:
        # increase figure height
        # fig.set_figheight(fig.get_figheight() * 1.1)
        # write smaller, below the title, italic, centered
        fig.text(
            0.5,
            0.95,
            subtitle,
            fontsize=fontsize * 1.2,
            fontweight='light',
            fontstretch='expanded',
            fontname='Arial',
            horizontalalignment='center',
            verticalalignment='top',
            style='italic',
        )

    if filename:
        from matplotlib.transforms import Bbox
        plt.savefig(filename, dpi=300, bbox_inches=Bbox([[0.5*figscale, 0], [figsize[0], figsize[1]]]))

    plt.show()



#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


