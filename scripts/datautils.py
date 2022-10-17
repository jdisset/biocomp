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


def binstats(data, bin_columns, stat_column, nbins=20, log=True, stats=["mean", "count"]):
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


def heatmap(df, bins, axis_names=None, figscale=1.0, count_threshold=1, cmap='YlGnBu', title=None, subtitle=None, filename=None, **kwargs):

    fontsize = 12 * figscale

    nstats = len(df.columns) - (1 if 'indices' in df.columns else 0)
    figsize = (figscale * 10 * nstats + (nstats - 1) * 4 * figscale, figscale * 10)
    fig, axes = plt.subplots(1, nstats, figsize=figsize)

    df = df[df['count'] > count_threshold]

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
