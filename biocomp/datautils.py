## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     imports     --
# ···············································································
import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial
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


def binstats(data, protein_names, bin_axis=None, resolution=0.5, bin_min=1e-12, bin_max=None):
    """Calculate statistics (mean, count), for each bin in len(bin_axis) dimensions."""
    if bin_axis is None:
        bin_axis = protein_names
    bin_axisid = [protein_names.index(p) for p in bin_axis]
    POWER_RANGE = 30
    VMAX_EPSILON = 10.0 ** (-POWER_RANGE)
    vmin, vmax = data[:, bin_axisid].min(axis=0), data[:, bin_axisid].max(axis=0) + VMAX_EPSILON
    if bin_min is not None:
        vmin = np.maximum(vmin, bin_min)
    if bin_max is not None:
        vmax = np.minimum(vmax, bin_max)
    # we want to bin the data using logaritmic bins with a fixed resolution, not a specific number of bins
    powers = 10.0 ** np.arange(-POWER_RANGE, POWER_RANGE, resolution)

    first_bin = []
    for v in vmin:
        first_bin.append(
            powers[np.clip(np.searchsorted(powers, v, side='right'), 0, len(powers) - 1)]
        )
    last_bin = []
    for v in vmax:
        last_bin.append(
            powers[np.clip(np.searchsorted(powers, v, side='right'), 0, len(powers) - 1)]
        )
    first_bin = np.array(first_bin)
    last_bin = np.array(last_bin)

    nbins = np.ceil(np.log10(last_bin / first_bin) / resolution).astype(int)
    #TODO: check why I have to do this???:
    nbins = np.maximum(nbins, 1)
    bin_edges = [
        np.geomspace(first_bin[i], last_bin[i], nbins[i] + 1) for i in range(len(first_bin))
    ]
    coords = np.array([np.digitize(data[:, i], be) for i, be in zip(bin_axisid, bin_edges)]).T - 1
    df = pd.DataFrame(data, columns=protein_names)
    df['coord'] = [tuple(c) for c in coords]
    df2 = df.groupby('coord').agg(['mean']).reset_index()
    df2['indices'] = df.reset_index().groupby('coord').agg({'index': lambda x: list(x)}).values
    df2['indices'] = df2['indices'].apply(lambda x: np.array(x, dtype=int))
    df2['count'] = df2['indices'].apply(len)

    for i, p in enumerate(bin_axis):
        df2[('coords', p)] = df2['coord'].apply(lambda x: x[i])
    df2 = df2.drop('coord', axis=1, level=0)
    df2 = df2.set_index([('coords', p) for p in bin_axis])

    bins = {p: be for p, be in zip(bin_axis, bin_edges)}
    return df2, bins


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{               --     balancing individual dataset     --
# ···············································································


def balance_per_bin(data, statdf, threshold_quantile=0.4, threshold_min=20) -> np.ndarray:
    assert 'count' in statdf.columns
    assert 'indices' in statdf.columns
    threshold = max(
        np.quantile(statdf[statdf['count'] > 0]['count'], threshold_quantile), threshold_min
    )
    balanced_indices = []
    for i, row in statdf.iterrows():
        balanced_indices.append(
            np.random.choice(
                row['indices'][0], min(int(threshold), int(row['count'])), replace=False
            )
        )
    balanced_data = data[np.concatenate(balanced_indices), :]
    return balanced_data


def balance_each_dataset(
    models: dict[str, bc.ComputeGraphModel],
    Y: dict[str, np.ndarray],
    bin_resolution=0.5,
    threshold_quantile=0.4,
    threshold_min=20,
):
    """balances each dataset individually (not across datasets but within each dataset,
    so that each dataset aims for a similar number of samples per bin)"""
    X_balanced, Y_balanced = {}, {}
    for sample, model in tqdm(models.items(), desc='balancing datasets'):
        data = Y[sample]
        out_proteins = model.get_output_proteins()
        in_proteins = model.get_inverted_input_proteins()
        stats, _ = binstats(data, out_proteins, in_proteins, resolution=bin_resolution)
        Y_balanced[sample] = balance_per_bin(
            data, stats, threshold_quantile=threshold_quantile, threshold_min=threshold_min
        )
        X_balanced[sample] = model.get_input_from_output(Y_balanced[sample])
    return X_balanced, Y_balanced


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


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


def heatmap(
    statdf,
    bins,
    stat_columns=['mean'],
    z_protein=None,
    lims={},
    figscale=1.0,
    count_threshold=2,
    cmap='YlGnBu',
    title=None,
    subtitle=None,
    filename=None,
    show=True,
    **kwargs,
):

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
        assert len(z_axis) == 1
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
            Z[coords] = max(value, 1e-20)

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

        if stat == 'count':
            ax.set_title('count', fontsize=fontsize)
        else:
            ax.set_title(f'{z_axis} {stat}', fontsize=fontsize)

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

        tick_freq = max(1, int(len(xbin) / 7))
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
    fig.suptitle(
        title,
        fontsize=fontsize * 1.8,
        fontweight='light',
        fontstretch='expanded',
        fontname='Arial',
        y=0.99,
    )

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

        plt.savefig(
            filename, dpi=300, bbox_inches=Bbox([[0.5 * figscale, 0], [figsize[0], figsize[1]]])
        )

    if show:
        plt.show()

    return fig, axes


# example usage
# out_proteins = model.get_output_proteins()
# in_proteins = model.get_inverted_input_proteins()
# stats, bins = binstats(y, out_proteins, in_proteins, resolution=0.5)
# heatmap(
# stats,
# bins,
# figscale=0.6,
# stat_columns=['mean','count'],
# z_protein='eYFP',
# lims={'mean': (1e3, 1e8)},
# title=f'{model.network.name}',
# subtitle=f'{len(y)} data points',
# )

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     data load/save     --
# ···············································································
import pickle


def save(data, path, overwrite=False, suffix='.pickle'):

    path = Path(path)
    if path.suffix != suffix:
        path = path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise RuntimeError(f'File {path} already exists.')
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def load(path, suffix='.pickle'):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Not a file: {path}')
    if path.suffix != suffix:
        raise ValueError(f'Not a {suffix} file: {path}')
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


def split_array_uniform(arr, n_batches, rng_key):
    n = len(arr)
    batch_size = n // n_batches
    a = jax.random.permutation(rng_key, arr)
    return [a[i * batch_size : (i + 1) * batch_size] for i in range(n_batches)]


def split_array_to_len(arr, l, rng_key):
    a = jax.random.permutation(rng_key, arr)
    return [a[i * l : (i + 1) * l] for i in range(len(arr) // l)]


def make_batches_dict(Y, n_batches, rng_key, models):
    y_ = {s: split_array_uniform(Y[s], n_batches, rng_key) for s in Y.keys()}
    y_batches = [{s: y_[s][i] for s in y_.keys()} for i in range(n_batches)]
    # get x_batches from y_batches
    x_batches = []
    for y_batch in y_batches:
        x_batch = {}
        for s in y_batch.keys():
            x_batch[s] = models[s].get_input_from_output(y_batch[s])
        x_batches.append(x_batch)
    return x_batches, y_batches



def make_batches_uniform_sampling(Y:list[np.ndarray], batch_size:int, rng_key, models:list[bc.ComputeGraphModel], total_size=None):
    """Split data into batches of equal size, for a list of arrays (one per sample).
    Each array might not have the same size originally, but the batches will
    have the same size and there will be the same amount of batches for Each
    sample in the end. To do so, we randomly sample from the arrays to get the
    same size for each batch.

    batch_size: int, the size of each batch (per sample)
    total_size: batch_size * n_batches. If None, it uses the largest array size.

    returns: x_batches, y_batches. Dimensions are (n_batches, n_models, batch_size, n_features)

    """

    if total_size is None:  # use the largest array as target total size
        total_size = max(len(y) for y in Y)

    n_batches = total_size // batch_size

    ylist = [jax.random.choice(rng_key, jnp.array(y), (total_size,)) for y in tqdm(Y)]
    xlist = [m.get_input_from_output(ylist[i]) for i, m in enumerate(models)]

    n_outputs = max(y.shape[1] for y in ylist)
    n_inputs = max(x.shape[1] for x in xlist)

    # add 0 pad
    y_p = jnp.array([np.pad(y, ((0, 0), (0, n_outputs - y.shape[1]))) for y in ylist])
    x_p = jnp.array([np.pad(x, ((0, 0), (0, n_inputs - x.shape[1]))) for x in xlist])

    y_batches = jnp.array(np.split(y_p[:, :n_batches * batch_size], n_batches, axis=1))
    x_batches = jnp.array(np.split(x_p[:, :n_batches * batch_size], n_batches, axis=1))

    assert y_batches.shape == (n_batches, len(Y), batch_size, n_outputs)
    assert x_batches.shape == (n_batches, len(Y), batch_size, n_inputs)

    return x_batches, y_batches
