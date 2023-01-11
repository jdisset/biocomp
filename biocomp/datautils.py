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
from tqdm import tqdm
import matplotlib.pyplot as plt

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                       --     data manager     --
# ···············································································


def check_model(m, x, y):
    # simple sanity checks for the model
    outp = m.get_output_proteins()  # name of output proteins
    inp = m.get_inverted_input_proteins()  # name of input proteins
    in_pos = m.get_inverted_input_positions()
    assert len(inp) == len(in_pos)
    assert len(inp) == len(set(inp))
    for iname in inp:
        assert iname in outp
    for ipos, outpos in in_pos.items():
        assert inp[ipos] == outp[outpos]
        assert np.all(x[:, ipos] == y[:, outpos])
    mdef = bc.ComputeGraphModel(m.network)
    mdef.build(bc.nodes.DEFAULT_COMPUTE_NODES_DICT)
    zerorng = jax.random.PRNGKey(0)
    p, _ = mdef.init(zerorng)
    vmapped = jax.jit(jax.vmap(mdef, in_axes=(None, 0, None)))
    ydef = vmapped(p, x, zerorng)
    for ipos, outpos in in_pos.items():
        assert np.allclose(x[:, ipos], ydef[:, outpos])


def basic_data_checks(X, Y, models):
    # basic checks
    assert set(X.keys()) == set(Y.keys()), 'X and Y must have the same keys'
    assert set(X.keys()) == set(models.keys()), 'data and models must have the same keys'

    for k, v in X.items():
        assert v.shape[0] == Y[k].shape[0], f"shape mismatch for key {k}"
        assert v.shape[1] == models[k].n_inputs, f"input shape mismatch for key {k}"
        assert Y[k].shape[1] == models[k].n_outputs, f"output shape mismatch for key {k}"


def advanced_data_checks(X, Y, models):
    for k, m in tqdm(models.items()):
        check_model(m, X[k], Y[k])


def shuffled_data(X, Y, models, rng_key):
    """shuffles each sample's data"""
    basic_data_checks(X, Y, models)
    new_order = {}
    keys = jax.random.split(rng_key, len(X))
    for sample, key in zip(X.keys(), keys):
        new_order[sample] = jax.random.permutation(key, X[sample].shape[0])
    Xout, Yout = {}, {}
    for sample in X.keys():
        Xout[sample] = X[sample][new_order[sample]]
        Yout[sample] = Y[sample][new_order[sample]]

    return Xout, Yout


def test_train_split(X, Y, models, rng_key, test_size=0.8):
    """get a random split train/test. Returns (Xtrain, Xtest, Ytrain, Ytest)"""
    basic_data_checks(X, Y, models)
    Xshuf, Yshuf = shuffled_data(X, Y, models, rng_key)
    # same is doable using jax.tree_map:
    X_train, X_test = jax.tree_map(lambda x: x[: int(x.shape[0] * test_size)], Xshuf), jax.tree_map(
        lambda x: x[int(x.shape[0] * test_size) :], Xshuf
    )
    Y_train, Y_test = jax.tree_map(lambda x: x[: int(x.shape[0] * test_size)], Yshuf), jax.tree_map(
        lambda x: x[int(x.shape[0] * test_size) :], Yshuf
    )
    return (X_train, Y_train), (X_test, Y_test)


class DataManager:
    def __init__(self, X: dict, Y: dict, models: dict, enable_checks=True):
        self.X = X
        self.Y = Y
        self.models = models
        self.enable_checks = enable_checks

        basic_data_checks(X, Y, models)

        if self.enable_checks:
            advanced_data_checks(X, Y, models)
            self.N_TEST = 100

    def normalize(self, factor):
        # for now, let's just scale the data
        # we can add more later
        self.X = jax.tree_map(lambda x: x / factor, self.X)
        self.Y = jax.tree_map(lambda x: x / factor, self.Y)
        basic_data_checks(self.X, self.Y, self.models)
        if self.enable_checks:
            advanced_data_checks(self.X, self.Y, self.models)
        return self.X, self.Y

    def consolidated(raw_X, raw_Y, rng_key):
        keys = list(raw_X.keys())
        # all sets don't necessary have the same shape, so we want to:
        # - pad with zeros to the right on the feature axis
        # - fill with random samples (from the same set of course) on the sample axis
        # first, let's get the max shape
        max_shape_x = (0, 0)
        max_shape_y = (0, 0)
        for k in keys:
            max_shape_x = max(max_shape_x, raw_X[k].shape)
            max_shape_y = max(max_shape_y, raw_Y[k].shape)
        assert max_shape_x[0] == max_shape_y[0]
        # now, let's pad and fill
        X = {}
        Y = {}
        for k in keys:
            X[k] = np.zeros(max_shape_x)
            Y[k] = np.zeros(max_shape_y)
            assert raw_X[k].shape[0] == raw_Y[k].shape[0]
            rng_key, subkey = jax.random.split(rng_key)
            # indices contains all the indices of the samples in the set, in a loop
            # until we have enough samples to fill the set
            shuff_range = jax.random.permutation(rng_key, raw_X[k].shape[0])
            indices = np.tile(shuff_range, max_shape_x[0] // raw_X[k].shape[0] + 1)[
                : max_shape_x[0]
            ]
            indices = jax.random.permutation(subkey, indices)
            X[k][: raw_X[k].shape[0], : raw_X[k].shape[1]] = raw_X[k][indices]
            Y[k][: raw_Y[k].shape[0], : raw_Y[k].shape[1]] = raw_Y[k][indices]
        return X, Y

    def preprocess(self, rng_key, cfg):
        XX, YY = balance_each_dataset(
            self.models,
            self.Y,
            rng_key,
            bin_resolution=cfg['balance_bin_resolution'],
            threshold_quantile=cfg['balance_threshold_quantile'],
            threshold_min=cfg['balance_threshold_min'],
        )

        basic_data_checks(XX, YY, self.models)
        if self.enable_checks:
            advanced_data_checks(XX, YY, self.models)
            # let's also pick n random samples for each YY and check
            # - that they can be found in the original Y
            # - that the matching X is the same
            for k, v in YY.items():
                indices = jax.random.choice(rng_key, v.shape[0], (self.N_TEST,))
                assert np.all(jax.vmap(jnp.all)(v[indices] == self.Y[k]))
                assert np.all(jax.vmap(jnp.all)(XX[k][indices] == self.X[k]))

        self.X = XX
        self.Y = YY
        self.normalize(cfg['normalize_factor'])

    def postscale(self, data, factor):
        return jax.tree_map(lambda x: x * factor, data)


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


DUTILS_BINNING_POWER_RANGE = 30
DUTILS_BINNING_VMAX_EPSILON = 10.0 ** (-DUTILS_BINNING_POWER_RANGE)


def mk_log_grid(vmin, vmax, resolution):
    """
    Generate logarithmically spaced bins with a specified resolution, between vmin and vmax
    Returns A list of arrays of bin edges, one per dimension.
    """
    vmin, vmax = np.asarray(vmin), np.asarray(vmax)
    assert vmin.shape == vmax.shape
    if vmin.ndim == 0:
        vmin, vmax = vmin[None], vmax[None]
    powers = 10.0 ** np.arange(-DUTILS_BINNING_POWER_RANGE, DUTILS_BINNING_POWER_RANGE, resolution)
    first_bin = np.array(
        [powers[np.clip(np.searchsorted(powers, v) - 1, 0, len(powers) - 1)] for v in vmin]
    )
    last_bin = np.array(
        [powers[np.clip(np.searchsorted(powers, v), 0, len(powers) - 1)] for v in vmax]
    )
    nbins = np.ceil(np.log10(last_bin / first_bin) / resolution).astype(int)
    nbins = np.maximum(nbins, 1)
    bin_edges = [
        np.geomspace(first_bin[i], last_bin[i], nbins[i] + 1) for i in range(len(first_bin))
    ]
    return bin_edges


def binstats(
    data,
    output_protein_names,
    bin_proteins=None,
    resolution=0.5,
    bin_min=1e-12,
    bin_max=None,
    force_minmax=False,
):
    """
    Calculate statistics (mean and count) for bins in multi-dimensional space.

    Parameters
    ----------
    data : array-like
        The data to be binned.
    output_protein_names : list of str
        The names of the features in the input data.
    bin_proteins : list of str, optional
        The names of the features to use for binning.
    resolution : float, optional
        The resolution of the bins, expressed as the number of decades per bin.
    bin_min : float, optional
        The minimum value for the bins.
    bin_max : float, optional
        The maximum value for the bins.
    force_minmax : bool, optional
        If `True`, the bins will always start at `bin_min` and end at `bin_max`.

    Returns
    -------
    df2 : DataFrame
        A DataFrame containing the binned data, with columns for the mean and count of each bin and rows indexed by the bin coordinates.
    bins : dict
        A dictionary containing the bin edges for each feature used for binning.
    """
    if bin_proteins is None:
        bin_proteins = output_protein_names
    bin_axisid = [output_protein_names.index(p) for p in bin_proteins]
    vmin, vmax = (
        data[:, bin_axisid].min(axis=0),
        data[:, bin_axisid].max(axis=0) + DUTILS_BINNING_VMAX_EPSILON,
    )
    if bin_min is not None:
        if force_minmax:
            vmin = np.ones_like(vmin) * bin_min
        else:
            vmin = np.maximum(vmin, bin_min)
    if bin_max is not None:
        if force_minmax:
            vmax = np.ones_like(vmax) * bin_max
        else:
            vmax = np.minimum(vmax, bin_max)

    bin_edges = mk_log_grid(vmin, vmax, resolution)
    coords = np.array([np.digitize(data[:, i], be) for i, be in zip(bin_axisid, bin_edges)]).T - 1
    df = pd.DataFrame(data, columns=output_protein_names)
    df['coord'] = [tuple(c) for c in coords]
    df2 = df.groupby('coord').agg(['mean']).reset_index()
    df2['indices'] = df.reset_index().groupby('coord').agg({'index': lambda x: list(x)}).values
    df2['indices'] = df2['indices'].apply(lambda x: np.array(x, dtype=int))
    df2['count'] = df2['indices'].apply(len)
    for i, p in enumerate(bin_proteins):
        df2[('coords', p)] = df2['coord'].apply(lambda x: x[i])
    df2 = df2.drop('coord', axis=1, level=0)
    df2 = df2.set_index([('coords', p) for p in bin_proteins])

    bins = {p: be for p, be in zip(bin_proteins, bin_edges)}
    return df2, bins


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{               --     balancing individual dataset     --
# ···············································································


def balance_per_bin(
    data, statdf, rng_key, threshold_quantile=0.4, threshold_min=20, replacement=True
) -> np.ndarray:
    assert 'count' in statdf.columns
    assert 'indices' in statdf.columns
    threshold = max(
        jnp.quantile(statdf[statdf['count'] > 0]['count'].values, threshold_quantile), threshold_min
    )
    balanced_indices = []
    for i, row in tqdm(statdf.iterrows()):
        # each row describes a bin with:
        # - count: number of datapoints in the bin
        # - indices: indices of datapoints in the bin
        n = min(int(threshold), int(row['count']))
        # if replacement:
            # n = int(threshold)
        choice = jax.random.choice(
                rng_key,
                row['indices'][0],
                shape=(n,),
                replace=replacement,
            )
        balanced_indices.append(choice)
    balanced_data = data[jnp.concatenate(balanced_indices), :]
    return balanced_data


def balance_each_dataset(
    models: dict[str, bc.ComputeGraphModel],
    Y: dict[str, np.ndarray],
    rng_key,
    bin_resolution=0.5,
    threshold_quantile=0.4,
    threshold_min=20,
    replacement=True,
):
    """balances each dataset individually (not across datasets but within each dataset,
    so that each dataset aims for a similar number of samples per bin)"""
    X_balanced, Y_balanced = {}, {}
    for sample, model in tqdm(models.items(), desc='balancing datasets'):
        data = Y[sample]
        out_proteins = model.get_output_proteins()
        in_proteins = model.get_inverted_input_proteins()
        stats, _ = binstats(data, out_proteins, in_proteins, resolution=bin_resolution)
        rng_key, subkey = jax.random.split(rng_key)
        Y_balanced[sample] = balance_per_bin(
            data,
            stats,
            subkey,
            threshold_quantile=threshold_quantile,
            threshold_min=threshold_min,
            replacement=replacement,
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


def model_parallel_coords(
    model, y, n_samples=500, cmap='Spectral_r', title=None, maxval=1e3, minval=1e-5
):
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    out_proteins = model.get_output_proteins()
    in_proteins = model.get_inverted_input_proteins()
    stats, bins = binstats(y, out_proteins, resolution=0.25)
    prot_diff = set(out_proteins) - set(in_proteins)

    # the plot will have one coordinate (vertical line) per out_protein.
    # each line is taken from stats (the mean of the bin)
    # stats is a dataframe with columns [('protein_name', 'mean'), ...]
    mean_values = jnp.array([stats[p]['mean'].values for p in out_proteins]).T
    choice = jax.random.choice(
        jax.random.PRNGKey(0), mean_values.shape[0], shape=(n_samples,), replace=True
    )
    mean_values = mean_values[choice]

    # 1 subplot per prot_diff
    fig, axes = plt.subplots(len(prot_diff), 1, figsize=(10, 9 * len(prot_diff)))

    if len(prot_diff) == 1:
        axes = [axes]

    for z_prot, ax in zip(prot_diff, axes):
        z_values = stats[z_prot]['mean'].values[choice]

        # x axis should be each protein name (we display a vertical line for each protein)
        ax.set_xticks(range(len(out_proteins)))
        ax.set_xticklabels(out_proteins, rotation=40)
        ax.vlines(range(len(out_proteins)), 0, maxval * 2, alpha=0.2, color='black')

        # y axis should be the mean value of the bin
        ax.set_yscale('log')
        cmap = plt.get_cmap(cmap)
        norm = colors.LogNorm(vmin=minval, vmax=maxval)
        clrs = [cmap(norm(z)) for z in z_values]

        for i, (x, c) in enumerate(zip(mean_values, clrs)):
            ax.plot(x, color=c, alpha=0.25, linewidth=2)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(sm, cax=cax, orientation='vertical', label=z_prot)

        ax.set_ylim(minval, maxval * 2)
        ax.set_title(f'{model.network.name if title is None else title}')

    return fig, axes


def model_heatmap(
    model,
    ydata,
    inner_resolution=0.5,
    outer_resolution=1,
    base_size=10,
    mincount=300,
    z_prot=None,
    lims=(1e-4, 1e2),
    title=None,
):
    """can be used to plot heatmaps for a model with up to 4 inputs"""
    out_proteins = model.get_output_proteins()
    in_proteins = model.get_inverted_input_proteins()
    if z_prot is None:
        z_prot = set(out_proteins) - set(in_proteins)
        z_prot = list(z_prot)[0]

    # we have more than 2 in_proteins: we will plot an array of heatmaps
    # for 3 in_proteins, we will have only one axis of groups
    # for 4 in_proteins, we will have 2 axes of groups

    import itertools

    n_in_proteins = len(in_proteins)
    axes_proteins = in_proteins[: n_in_proteins - 2]
    ax_stats = [
        binstats(ydata, out_proteins, bin_proteins=[axp], resolution=outer_resolution)
        for axp in axes_proteins
    ]
    group_indices = [s[s['count'] > mincount]['indices'] for s, _ in ax_stats]
    group_mean = [
        s[s['count'] > mincount][(axp, 'mean')] for axp, (s, _) in zip(axes_proteins, ax_stats)
    ]

    axes_len = [len(g) for g in group_mean]
    n_axes = len(axes_len)

    if n_axes == 1:
        fig, axes = plt.subplots(1, axes_len[0], figsize=(base_size * axes_len[0], base_size))
        axes = axes.flatten()
    elif n_axes == 2:
        fig, axes = plt.subplots(
            axes_len[0], axes_len[1], figsize=(base_size * axes_len[1], base_size * axes_len[0])
        )
        axes = axes.flatten()
    elif n_axes == 0:
        fig, axes = plt.subplots(1, 1, figsize=(base_size, base_size))
        axes = [axes]
    else:
        raise ValueError(f'Cannot plot {n_axes} axes')

    bin_prots = set(in_proteins) - set(axes_proteins)

    ax_prot_ranges = dict()
    for i, (ax, indices, mean) in enumerate(
        zip(axes, itertools.product(*group_indices), itertools.product(*group_mean))
    ):
        data = ydata[indices]
        ax_prot_ranges[i] = [
            (data[:, out_proteins.index(p)].min(), data[:, out_proteins.index(p)].max())
            for p in axes_proteins
        ]
        ax_stats, ax_bins = binstats(
            data,
            out_proteins,
            bin_proteins=bin_prots,
            resolution=inner_resolution,
            bin_min=lims[0],
            bin_max=lims[1],
            force_minmax=True,
        )
        heatmap(
            ax_stats,
            ax_bins,
            axes=[ax],
            stat_columns=['mean'],
            z_protein=z_prot,
            lims={'mean': lims},
            show=False,
            count_threshold=4,
        )
        # remove axis labels except for the last row and first column
        if n_axes == 1 and i < axes_len[0] - 1:
            ax.set_xlabel('')
        elif n_axes == 2:
            if i % axes_len[1] != 0:
                ax.set_ylabel('')
            if i // axes_len[1] != axes_len[0] - 1:
                ax.set_xlabel('')

        axtitle = ''
        if n_axes > 0:
            axtitle = f'{axes_proteins[0]}={mean[0]:.2f}'
        if n_axes == 2:
            axtitle += f', {axes_proteins[1]}={mean[1]:.2f}'
        ax.set_title(axtitle)

    if title is None:
        title = f'Heatmap for {z_prot}'

    fig.suptitle(title)

    return fig, axes


def model_heatmap_old(model, y, resolution=0.5):
    out_proteins = model.get_output_proteins()
    in_proteins = model.get_inverted_input_proteins()
    stats, bins = binstats(y, out_proteins, in_proteins, resolution=resolution)
    z_prot = set(out_proteins) - set(in_proteins)
    fig, ax = heatmap(
        stats,
        bins,
        figscale=0.6,
        stat_columns=['mean'],
        z_protein=z_prot.pop(),
        lims={'mean': (1e-3, 1e3)},
        title=f'{model.network.name} data',
        subtitle=f'{len(y)} data points',
        show=False,
    )
    return fig, ax


def heatmap(
    statdf,
    bins,
    fig=None,
    axes=None,
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
    assert len(axes) == nstats

    if axes is None:
        assert fig is None
        fig, axes = plt.subplots(1, nstats, figsize=figsize)
        if nstats == 1:
            axes = [axes]
    else:
        assert len(axes) == nstats

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
        grid_thickness = 50.0 * figscale / max(len(bins), 15)
        if grid_thickness > 0.75:
            grid_thickness = max(1.00, grid_thickness)
            ax.grid(which='minor', axis='both', linestyle='-', color='w', linewidth=grid_thickness)
        ax.tick_params(which='minor', bottom=False, left=False)

        tick_freq = max(1, int(len(xbin) / 7))
        ax.set_xticks(np.arange(len(xbin))[::tick_freq], minor=False)
        ax.set_yticks(np.arange(len(ybin))[::tick_freq], minor=False)
        # xl = [f"{x:.0f}" if x < 100 else fmt.format("{:m}", x) for x in xbin[::tick_freq]]
        # yl = [f"{x:.0f}" if x < 100 else fmt.format("{:m}", x) for x in ybin[::tick_freq]]
        xl = [fmt.format("{:m}", x) for x in xbin[::tick_freq]]
        yl = [fmt.format("{:m}", x) for x in ybin[::tick_freq]]
        ax.set_xticklabels(xl, minor=False)
        ax.set_yticklabels(yl, minor=False)

        # remove border
        for spine in ax.spines.values():
            spine.set_visible(False)

        if fig is not None:
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
            cax.tick_params(labelsize=fontsize)
            cax.yaxis.get_offset_text().set_fontsize(fontsize)
            font = 'Roboto Mono Light for Powerline'
            if font in plt.rcParams['font.family']:
                cax.yaxis.get_offset_text().set_fontname(font)
                for item in (
                    [ax.xaxis.label, ax.yaxis.label]
                    + ax.get_xticklabels()
                    + ax.get_yticklabels()
                    + cax.get_xticklabels()
                    + cax.get_yticklabels()
                ):
                    item.set_fontname(font)

        # set all font sizes
        ax.tick_params(labelsize=fontsize)
        ax.title.set_fontsize(fontsize * 1.4)
        ax.xaxis.label.set_fontsize(fontsize)
        ax.yaxis.label.set_fontsize(fontsize)

        # if 'Arial'  in plt.rcParams['font.family']:
        # ax.title.set_fontname('Arial')
        ax.title.set_fontweight('light')
        ax.title.set_fontstretch('expanded')

    title = title if title is not None else f'{z_axis} stats'
    # suptitle should be higher than default

    if fig is not None:
        fig.suptitle(
            title,
            fontsize=fontsize * 1.8,
            fontweight='light',
            fontstretch='expanded',
            # fontname='Arial',
            y=0.99,
        )

    if subtitle and fig is not None:
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
            # fontname='Arial',
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


def save(data, path, overwrite=False, rename_if_exists=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        elif rename_if_exists:
            path = path.with_name(path.stem + '_' + path.suffix)
        else:
            raise RuntimeError(f'File {path} already exists.')
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def load(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Not a file: {path}')
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     batches     --
# ···············································································


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


def make_batches_uniform_sampling(
    X: list[np.ndarray],
    Y: list[np.ndarray],
    batch_size: int,
    rng_key,
    total_size=None,
):
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

    y_batches = jnp.array(np.split(y_p[:, : n_batches * batch_size], n_batches, axis=1))
    x_batches = jnp.array(np.split(x_p[:, : n_batches * batch_size], n_batches, axis=1))

    assert y_batches.shape == (n_batches, len(Y), batch_size, n_outputs)
    assert x_batches.shape == (n_batches, len(Y), batch_size, n_inputs)

    return x_batches, y_batches


def batch(X, Y, batch_size, n_batches=None):
    """Yields batches of data from X and Y."""
    n = X.shape[0]
    if n_batches is None:
        n_batches = n // batch_size
    # using sampling with replacement
    for i in range(n_batches):
        idx = np.random.choice(n, size=batch_size, replace=True)
        yield X[idx], Y[idx]


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
