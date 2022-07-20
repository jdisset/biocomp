# from logging.config import valid_ident
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
from scipy import stats

'''
A bunch of helper functions to manipulate flow data
'''


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(
        im, ax=ax, fraction=0.038, pad=0.04, **cbar_kw
    )  # was fraction=0.046
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30,
    #          ha="right", rotation_mode="anchor")
    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def subset(df, *args):
    '''
    returns subset of dataframe based on identity operation
    e.g. subset(df, ('Cells', 'Cells_2'))

    returns: Pandas DF (subset of input)

    PARAMS
    ------
    `df` : pandas dataframe from cytoflow.
    `*args` : comparison (arbitrary number). Type: Tuple.
    '''

    for arg in args:
        df = df[df[arg[0]] == arg[1]]

    return df


def cut_list(x, y, x_min=None, x_max=None, y_min=None, y_max=None):
    # MIN MAX LIST UPDATES
    if x_min:
        x = [n for n in x if n >= x_min]
    if x_max:
        x = [n for n in x if n <= x_max]
    if y_min:
        y = [n for n in y if n >= y_min]
    if y_max:
        y = [n for n in y if n <= y_max]

    return x, y


# create array from tube
def create_arr(tube, x, x_name, y, y_name, z_name, stat='logmean', x_norm_names=[]):
    '''
    creates a 2D numpy array along axes `x` and `y` with values `z`.
    Requires a single `tube` to be passed.
    Note: x and y are actually switched, but I am too lazy to change that rn.

    PARAMS
    ------
    `tube` : (pandas.df). Filtered df of one single tube.
    `x` : (list). sorted list of inputs (e.g. mKateBins/copy number)
    `x_name` : (str). Name of bin that should be on x axis of array. REQ: needs the be same as column name in tube df.
    `y` : (list) : see x
    `y_name` : (str). See x_name.
    `z_name` : (str). Name color axis value. Will compute the `stat` based on x and y bin.
    `kwargs` : (dict). kwargs for stats that require more than one array.
    `stat` : (str), default='logmean'. Statistic to create array. Options:
        'logmean' : log of mean
        'mean' : mean
        'geomean' : geometric mean
        'geomean95' : geometric mean of 95% CI
        'loggeomean' : log of geometric mean
        'std' : standard deviation from mean
        'logstd' : log of standard deviation from mean
        'geostd' : standard deviation from geometric mean
        'loggeostd' : log of standard deviation from geometric mean (NOT DONE)
        'variance' : variance
        'CV' : Coefficient of Variance
        'normalisedMean' : REQUIRES: {x_norm_names=(list)}. fit linear function, take predicted value for the bin value
        'normalisedStd' : REQUIRES: {x_norm_names=(list)}. normalises all df_col values by x_vals, then returns std.
    '''

    # DEFINE STATISTICS
    def _logmean(df_col, **kwargs):
        return np.log(df_col.mean())

    def _logstd(df_col, **kwargs):
        return np.log(df_col.std())

    def _std(df_col, **kwargs):
        return df_col.std()

    def _mean(df_col, **kwargs):
        return df_col.mean()

    def _variance(df_col, **kwargs):
        return df_col.var()

    def _logvariance(df_col, **kwargs):
        return np.log(df_col.var())

    def _logmedian(df_col, **kwargs):
        return np.log(df_col.median())

    def _logquantileDist(df_col, **kwargs):
        try:
            q75, q25 = np.percentile(df_col, [75, 25])
            iqr = np.abs(q75 - q25)
        except IndexError:
            iqr = np.nan
        return np.log(iqr)

    def _median(df_col, **kwargs):
        return df_col.median()

    def _quantileDist(df_col, **kwargs):
        try:
            q75, q25 = np.percentile(df_col, [75, 25])
            iqr = np.abs(q75 - q25)
        except IndexError:
            iqr = np.nan
        return iqr

    def _coefficientVariation(df_col, **kwargs):
        mean = df_col.mean()
        std = df_col.std()
        cv = std/mean
        return cv

    def _geomean(df_col, **kwargs):
        return stats.gmean(df_col)

    def _loggeomean(df_col, **kwargs):
        return np.log(stats.gmean(df_col))

    def _geostd(df_col, **kwargs):
        return stats.gstd(df_col)

    def _geoCV(df_col, **kwargs):
        gstd = stats.gstd(df_col)
        gmean = stats.gmean(df_col)
        gcv = gstd/gmean
        return gcv

    def _count(df_col, **kwargs):
        return df_col.count()

    def _normalisedMean(df_col, x_vals=None, check_val=None, **kwargs):
        if x_vals == [] or check_val is None:
            raise ValueError(
                'Requires x_vals and check_val to compute normalised mean')

        '''
        df_col = list(df_col)

        comp_vals = df_col/x_vals
        mean_val = stats.gmean(comp_vals)
        std_val = stats.gstd(comp_vals)
        
        for i, v in enumerate(comp_vals):
            if v < mean_val-5*std_val or v > mean_val+5*std_val:
                x_vals[i] = mean_val
                df_col[i] = mean_val
        
        new_x_vals = []
        new_df_col = []
        for i, (xval, dfcolval) in enumerate(zip(x_vals, df_col)):
            if xval or dfcolval is not np.nan:
                new_x_vals.append(xval)
                new_df_col.append(dfcolval)
        
        x_vals = new_x_vals
        df_col = new_df_col
        '''

        coef = np.polyfit(x_vals, df_col, 1)
        poly1d_fn = np.poly1d(coef)

        return poly1d_fn(check_val)

    def _normalisedStd(df_col, x_vals=None, **kwargs):
        if x_vals is None:
            raise ValueError('Requires x_vals to compute normalised std')

        comp_vals = df_col/x_vals
        res = stats.gstd(comp_vals)

        return res

    stat_dict = {
        'logmean': _logmean,
        'logstd': _logstd,
        'mean': _mean,
        'std': _std,
        'variance': _variance,
        'logvariance': _logvariance,
        'logmedian': _logmedian,
        'logquantileDist': _logquantileDist,
        'median': _median,
        'quantileDist': _quantileDist,
        'CV': _coefficientVariation,
        'geomean': _geomean,
        'loggeomean': _loggeomean,
        'geostd': _geostd,
        'geoCV': _geoCV,
        'count': _count,
        'normalisedMean': _normalisedMean,
        'normalisedStd': _normalisedStd,
    }

    statistic = stat_dict[stat]

    # x = list(np.sort(list(tube[x_name].unique())))[::-1]
    # y = list(np.sort(list(tube[y_name].unique())))
    arr = np.zeros((len(x), len(y)))
    for i, xval in enumerate(x):
        for j, yval in enumerate(y):
            # print(i, xval, j, yval)
            filtered_df = subset(tube, (x_name, xval), (y_name, yval))

            x_norm = []
            for x_norm_name in x_norm_names:
                x_norm.append(filtered_df[x_norm_name])

            # np.sqrt(np.sum(np.square(x_norm), axis=0))
            x_norm = np.linalg.norm(x_norm, axis=0)
            norm_check_val = np.sqrt(xval**2 + yval**2)
            stat_val = statistic(
                filtered_df[z_name], x_vals=x_norm, check_val=norm_check_val)
            if stat_val > 0:
                arr[i, j] = stat_val
            elif stat_val <= 0:
                arr[i, j] = 0
            else:
                arr[i, j] = np.nan

    return arr

# create a bunch of heatmaps


def create_heatmap(cell_singlets, save_directory, x_name, y_name, z_name, x_range=None, y_range=None, z_lim=None, tube_names=None):
    '''
    quick function to plot heatmaps for all tubes into a folder

    PARAMS
    ------
    `cell_singlets` (pandas df) : df of gated and already filtered single cell events.   
    `save_directory` (str) : folder where to save all generated heatmaps
    '''

    # cytoflow specific constant names
    # could move this into args if need to modify.
    TUBE_NAME = 'TUBE_NAME'
    if tube_names is None:
        tube_names = list(cell_singlets[TUBE_NAME].unique())

    for name in tube_names:
        tube = subset(cell_singlets, (TUBE_NAME, name))

        if x_range is None:
            x = list(np.sort(list(tube[x_name].unique())))[::-1]
        else:
            x = x_range

        if y_range is None:
            y = list(np.sort(list(tube[y_name].unique())))
        else:
            y = y_range

        arr = create_arr(tube, x, x_name, y, y_name, z_name)

        fig, ax = plt.subplots()

        im, cbar = heatmap(arr, x, y, ax=ax,
                           cmap="viridis", cbarlabel=f"log({z_name})")  # cividis
        # texts = annotate_heatmap(im, valfmt="{x:.1f} t")
        if z_lim is not None:
            im.set_clim(z_lim)

        ax.set_title(name)
        ax.set_xlabel(y_name)
        ax.set_label(x_name)
        # ax.xaxis.set_label_position('top')
        # ax.yaxis.set_label_position('left')
        plt.grid(False)

        fig.tight_layout()

        save_path = os.path.join(save_directory, f'{name}.png')
        plt.savefig(save_path)


if __name__ == '__main__':
    '''
    Ignore this, just some quick test if I forget how to do sth
    '''
    x = [1, 2, 3, 4, 5]
    x_min = None
    if x_min:
        x = [n for n in x if n >= x_min]
    print(x)
