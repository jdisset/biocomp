import pandas as pd
from time import time
from pathlib import Path
import urllib.parse
from pyppeteer import launch
import sys
import gspread
from functools import partial
import asyncio
from tqdm import tqdm
import json
import numpy as np
from types import SimpleNamespace
import biocomp as bc
import pickle
from PIL import Image
import biocomp.datautils as du
import rich
from rich.console import Console
from rich.progress import track
from typing import List
import cProfile


class profiler:
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self.profiler = cProfile.Profile()
        self.profiler.enable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.disable()
        self.profiler.dump_stats(self.filename)


class ddict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return ddict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def is_interactive():
    import matplotlib as mpl

    return mpl.is_interactive()


DEFAULT_DATA_PATH = Path("~/Dropbox (MIT)/Biocomp/").expanduser()
DEFAULT_XP_PATH = DEFAULT_DATA_PATH / "Experiments"
DEFAULT_RECIPE_PATH = DEFAULT_DATA_PATH / "Recipes"
DEFAULT_LIB_PATH = Path("~/Code/Weiss/biocomp/__cache/lib.pickle").expanduser()

# we check if there is a file named ~/.biocomp.json
# if so, we load it and use the paths defined there
# otherwise, we use the default paths defined above
GLOBAL_CONFIG_PATH = Path.home() / '.biocomp.json'
if GLOBAL_CONFIG_PATH.exists():
    with open(GLOBAL_CONFIG_PATH) as f:
        config = json.load(f)
        DEFAULT_XP_PATH = Path(config.get('xp_path', DEFAULT_XP_PATH))
        DEFAULT_RECIPE_PATH = Path(config.get('recipe_path', DEFAULT_RECIPE_PATH))
        DEFAULT_LIB_PATH = Path(config.get('lib_path', DEFAULT_LIB_PATH))

# we also check the environment variables to see if they define the paths
# if so, we use them in priority

import os

if 'BIOCOMP_XP_PATH' in os.environ:
    DEFAULT_XP_PATH = Path(os.environ['BIOCOMP_XP_PATH'])
if 'BIOCOMP_RECIPE_PATH' in os.environ:
    DEFAULT_RECIPE_PATH = Path(os.environ['BIOCOMP_RECIPE_PATH'])
if 'BIOCOMP_LIB_PATH' in os.environ:
    DEFAULT_LIB_PATH = Path(os.environ['BIOCOMP_LIB_PATH'])

# convenience loading functions with default paths
def load_xp(xpname, lib, xp_path=DEFAULT_XP_PATH, recipe_path=DEFAULT_RECIPE_PATH, **kwargs):
    xp = bc.XP(xpname, xp_path, recipe_path, lib, **kwargs)
    return xp


def list_xp(xp_path=DEFAULT_XP_PATH):
    return [x.name for x in xp_path.iterdir() if x.is_dir()]


def load_lib(lib_path=DEFAULT_LIB_PATH):
    return du.load(lib_path)


from matplotlib.colors import LinearSegmentedColormap

TEALS_CMAP = LinearSegmentedColormap.from_list(
    "teals",
    [
        [0.957, 0.913, 0.804],
        [0.613, 0.745, 0.734],
        [0.465, 0.672, 0.635],
        [0.272, 0.507, 0.535],
        [0.01, 0.1, 0.15],
    ],
)

DEFAULT_CMAP = TEALS_CMAP

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{              --     Streamlit utils and components     --
# ···············································································
import streamlit as st
from st_aggrid import AgGrid


def getStState(varname, func, *args, **kwargs):
    if varname not in st.session_state:
        st.session_state[varname] = None
    if varname not in st.session_state:  # probably streamlit is not runing?
        print('Warning: streamlit session_state is disabled')
        return func(*args, **kwargs)
    if st.session_state[varname] is None:
        st.session_state[varname] = func(*args, **kwargs)
    return st.session_state[varname]


def md(t):
    return st.markdown(t)


def h1(t):
    return md(f'# {t}')


def h2(t):
    return md(f'## {t}')


def h3(t):
    return md(f'### {t}')


def h4(t):
    return md(f'#### {t}')


def b():
    return md('---')


def ag(df):
    rowH = 29
    AgGrid(
        df.reset_index(), fit_columns_on_grid_load=True, theme='light', height=(len(df) + 1) * rowH
    )


# -- custom streamlit components
import streamlit.components.v1 as components

if not is_interactive():
    _component_func = components.declare_component("ned_component", url="http://localhost:1234")
else:
    _component_func = lambda: None




def dnaOutput(nodes, key=None, func=_component_func, **kwargs):
    tnodes = [bc.ut.updated_dict(n, {'data': {'id': n['id']}}) for n in nodes if n['type'] == 'DNA']
    return func(nodes=tnodes, output_type='DNA', key=key, **kwargs)


def drawCentralDogmaGraph(gdf, key=None, func=_component_func):
    nodes = [{'id': f'{i}', 'type': n.type, 'data': n.to_dict()} for i, n in gdf.iterrows()]
    edges = [
        {'id': f'{i}', 'source': f'{i}', 'target': f'{n.successor}'}
        for i, n in gdf.iterrows()
        if n.successor
    ]
    tnodes = [bc.ut.updated_dict(n, {'data': {'id': n['id']}}) for n in nodes]
    return func(nodes=tnodes, edges=edges, output_type='GRN', key=key)  # {{{}}}


def network_to_graph(network):
    """ Turns a network object into a fully serializable dictionary that can be
    safely interpreted by the frontend UI """

    compg = network.compute_graph
    cdg = network.central_dogma_graph

    nodes = [
        {'id': str(i), 'type': n.type, 'data': bc.ut.updated_dict(n.to_dict(), {'id': i})}
        for i, n in compg.iterrows()
    ]

    node_id_to_index = {int(n['id']): i for i, n in enumerate(nodes)}

    uidGen = bc.ut.uniqueIdGenerator()
    edges = []
    has_output_values = "output_values" in compg.columns
    for i, n in compg.iterrows():
        if n.output_to:
            for n_out, (o, h) in enumerate(n.output_to):
                cdgin = compg.loc[o].cdg_input
                srccdg = None # if available, we join the "central dogma" information to the edge
                if cdgin is not None and not isinstance(cdgin, str) and isinstance(cdgin, list):
                    assert all(
                        [isinstance(x, int) for x in cdgin]
                    ), "cdg_input must be a list of integers"
                    srccdg = cdg.loc[cdgin[h]].to_dict()
                edge = {
                    'id': f'edge_{uidGen()}',
                    'source': str(i),
                    'sourceHandle': str(n_out + 1),
                    'target': str(o),
                    'targetHandle': str(h),
                    'data': {
                        'source_node_list_id': node_id_to_index[i],
                        'target_node_list_id': node_id_to_index[o],
                        'srccdg': srccdg,
                        'tgthandle': str(h),
                        'outputValue': n.output_values[n_out] if has_output_values else '',
                    },
                }
                edges.append(edge)

    return make_json_compatible(nodes), make_json_compatible(edges)


def drawComputeGraph(network, func=_component_func, **kwargs):
    nodes, edges = network_to_graph(network)
    return func(nodes=nodes, edges=edges, output_type='COMPUTE', **kwargs)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                   --     google sheet helpers     --
# ···············································································

GOOGLE_APP_CREDENTIALS = '/Users/jeandisset/.google/biocomp/key.json'
SHEET_KEY = '1K_2bt90E-Wk-A9PYGXGbKDJy-olojKtksy1jxCQAzME'
# This function grabs the content of a google sheet and returns a pandas dataframe:
def getGoogleSheet(key, sheet_name, credentials=GOOGLE_APP_CREDENTIALS):
    gspread_client = gspread.service_account(filename=credentials)
    workbook = gspread_client.open_by_key(key)
    sheet = workbook.worksheet(sheet_name)
    data = sheet.get_all_values()
    headers = data.pop(0)
    df = pd.DataFrame(data, columns=headers)
    df = df.set_index(df.columns[0])
    return df


def getAllGoogleSheets(key=SHEET_KEY, credentials=GOOGLE_APP_CREDENTIALS):
    gspread_client = gspread.service_account(filename=credentials)
    workbook = gspread_client.open_by_key(key)
    sheets = workbook.worksheets()
    sheets_dict = {}
    for sheet in track(sheets, description='Loading library sheets'):
        df = pd.DataFrame(sheet.get_all_records())
        df.set_index(df.columns[0], inplace=True)
        sheets_dict[sheet.title] = df
    lib = SimpleNamespace(**sheets_dict)
    return lib


def listGoogleSpreadsheets(credentials=GOOGLE_APP_CREDENTIALS):
    gspread_client = gspread.service_account(filename=credentials)
    spreadsheets = gspread_client.openall()
    if spreadsheets:
        print("Available spreadsheet workbooks:")
        for spreadsheet in spreadsheets:
            print("Title:", spreadsheet.title, "URL:", spreadsheet.url)
    else:
        print("No spreadsheets available")
        print("Please share the spreadsheet with Service Account email")


def getLibFromGoogleSheet(key=SHEET_KEY, credentials=GOOGLE_APP_CREDENTIALS):
    l = getAllGoogleSheets(key, credentials)
    lib = bc.PartsLibrary(
        l.parts, l.L0s, l.L1s, l.L2s, l.categories, l.sequestrons, l.sequestron_types
    )
    return lib


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                   --     react screen captures     --
# ···············································································

DEFAULT_COMPONENT_PATH = Path('../biocomp-ui/frontend/dist/static_index.html')

browser = None


async def get_global_browser():
    global browser
    if browser is None:
        print('launching browser')
        browser = await launch()
    else:
        print('browser already exists')
    return browser


def make_batches(L, n):
    perbatch = len(L) / n
    return [L[int(perbatch * i) : int(perbatch * (i + 1))] for i in range(n)]


def screenCaptures(
    f,
    *args,
    out_dir_path='./',
    filenames=None,
    module_path=DEFAULT_COMPONENT_PATH,
    width=1500,
    height=1500,
    n_batches=1,
):

    params = []

    def param_extractor(**kwargs):
        nonlocal params
        r = {**kwargs}
        params.append(r)

    for a in zip(*args):
        f(*a, func=param_extractor)

    pj = [urllib.parse.quote_plus(json.dumps(p)) for p in params]
    urls = ['file://' + str(module_path.resolve()) + '?args=' + p for p in pj]
    # print(f'urls: {urls}')

    if filenames is not None:
        assert len(filenames) == len(params)
        outfiles = filenames
    else:
        outpath = Path(out_dir_path)
        outpath.mkdir(parents=True, exist_ok=True)
        outfiles = [str(outpath / f'{i}.pdf') for i in range(len(params))]

    url_batches = make_batches(urls, n_batches)
    file_batches = make_batches(outfiles, n_batches)

    async def main(urls, outfiles):
        # browser = await launch()
        browser = await get_global_browser()

        async def take(url, outfile):
            page = await browser.newPage()
            await page.goto(url, {'waitUntil': 'networkidle0'})
            await asyncio.sleep(0.1)
            # await page.setViewport({'width': width, 'height': height + 50})

            # test if outfile ends with .pdf or .png
            if outfile.endswith('.png'):
                await page.screenshot(
                    {
                        'path': outfile,
                        'omitBackground': True,
                        'type': 'png',
                        'clip': {'x': 0, 'y': 0, 'width': width, 'height': height},
                    }
                )
            elif outfile.endswith('.pdf'):
                await page.pdf(
                    {
                        'path': outfile,
                        'width': width,
                        'height': height,
                        'pageRanges': '1',
                        'printBackground': False,
                    }
                )
            else:
                raise ValueError(f'screenshot filename must end with .pdf or .png')
            print('saved', outfile)
            await page.close()

        await asyncio.gather(*[take(*args) for args in zip(urls, outfiles)])

        # await browser.close()

    start = time()
    loop = asyncio.get_event_loop()
    for args in zip(url_batches, file_batches):
        loop.run_until_complete(main(*args))
    try:
        loop.close()
    except:
        pass
    end = time()
    print(f'Saved all screenshots in {end-start}s')


def plot_networks(
    nets: List[bc.Network],
    filenames=None,
    axes=None,
    H=2000,
    W=800,
    outputs=None,
    figsize=(20, 20),
    show=False,
    show_title=True,
):
    import nest_asyncio

    nest_asyncio.apply()

    if filenames is None:
        show = True
        import tempfile
        filenames = [tempfile.mktemp(suffix='.png') for _ in nets]

    if outputs is not None:
        assert len(outputs) == len(nets)
        # make a copy of all nets:
        nets = [net.copy() for net in nets]
        for net, output in zip(nets, outputs):
            # output is a dict (node_id -> output)
            net.compute_graph['output_values'] = None
            for node_id, o in output.items():
                outp = o if o.ndim > 0 else [o]
                net.compute_graph['output_values'][node_id] = np.array(outp)

    def drawComputeGraph(network, func=_component_func, **kwargs):
        nodes, edges = network_to_graph(network)
        return func(nodes=nodes, edges=edges, output_type='COMPUTE', height=H, width=W, **kwargs)

    screenCaptures(
        drawComputeGraph,
        nets,
        filenames=filenames,
        height=H,
        width=W,
    )

    if show:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        if axes is None:
            axes = [plt.subplots(figsize=figsize)[1] for _ in nets]
        for f, n, (ax) in zip(filenames, nets, axes):
            img = mpimg.imread(f)
            # we want no border, nothing other than the image
            ax.set_axis_off()
            ax.imshow(img)
            if show_title:
                ax.text(
                    0.90,
                    0.95,
                    n.name,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes,
                )
            ax.patch.set_facecolor('white')


def plot_cdg(nets: List[bc.Network], filenames):
    import nest_asyncio

    nest_asyncio.apply()
    H = 1000
    W = 1000

    def draw_cdg(net, *a, **kw):
        drawCentralDogmaGraph(net.central_dogma_graph, *a, **kw)

    screenCaptures(
        draw_cdg,
        nets,
        filenames=filenames,
        height=H,
        width=W,
    )


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     plotting     --
# ···············································································

import matplotlib.pyplot as plt


def plotBestLoss(best, others, title='', outfile=None, vmax=None):
    fig, a = plt.subplots(1, 1, figsize=(6, 5))
    for l in others:
        a.plot(l, color="#aaaaaa", linewidth=1)
    a.plot(best, color="red", linewidth=2.5)
    fig.suptitle(title)
    if vmax is not None:
        a.set_ylim([0, vmax])
    if outfile is not None:
        fig.savefig(outfile, dpi=100)
        plt.close()
    else:
        plt.show()
    return fig


def grid_map(F, xrange, yrange, meshres):
    import jax

    XX, YY = np.meshgrid(
        np.linspace(xrange[0], xrange[1], meshres[0]),
        np.linspace(yrange[0], yrange[1], meshres[1]),
        indexing='xy',
    )
    coords = np.column_stack((XX.ravel(), YY.ravel()))
    ZZ = jax.vmap(F)(coords).reshape(XX.shape)
    return XX, YY, ZZ


def plotFuncOutput(F, ax, xrange=(0, 1), yrange=(0, 1), meshres=(500, 500), cmap=None):
    if cmap is None:
        cmap = "Reds"
    XX, YY, ZZ = grid_map(F, xrange, yrange, meshres)
    pc = ax.pcolormesh(XX, YY, ZZ, cmap=cmap, shading='auto', vmin=0)
    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)
    return pc, XX, YY, ZZ


def plotModelOutput(
    model,
    params,
    figsize=(12, 10),
    meshres=(500, 500),
    xrange=(0, 1),
    yrange=(0, 1),
    outfile=None,
    title='',
):
    from jax import tree_util as pytree

    flist = [[0.0, 0.493, 0.579], [0.896, 0.866, 0.806], [0.844, 0.1, 0.111]]

    cmap = DEFAULT_CMAP
    plt.rcParams["axes.grid"] = False

    fig, a = plt.subplots(1, 1, figsize=figsize)
    pc, XX, YY, ZZ = plotFuncOutput(pytree.Partial(model, params), a, xrange, yrange, meshres, cmap)
    a.contour(XX, YY, ZZ, 1, colors='black', linewidths=1, alpha=0.7)
    a.set_xlabel('predicted')
    a.xaxis.set_ticks([])
    a.yaxis.set_ticks([])
    a.set_aspect('equal')
    cax = a.inset_axes([1.04, 0.2, 0.05, 0.6], transform=a.transAxes)
    fig.colorbar(pc, ax=a, cax=cax)

    fig.suptitle(title)

    if outfile is not None:
        fig.savefig(outfile, dpi=120)
    else:
        plt.show()
    plt.close()
    return fig


def trainingMovie(
    model, compg_history, params, best_loss, losses, outdir='../__out/movie_00', step=1
):
    outpath = Path(outdir)
    losspath = outpath / 'loss/'
    predpath = outpath / 'pred/'
    graphpath = outpath / 'graph/'
    losspath.mkdir(parents=True, exist_ok=True)
    predpath.mkdir(parents=True, exist_ok=True)
    graphpath.mkdir(parents=True, exist_ok=True)

    # assert len(compg_history) == len(best_loss)

    n_batches = int(len(compg_history) / step / 50) + 1
    screenCaptures(
        partial(drawComputeGraph, height=2000),
        compg_history[::step],
        out_dir_path=graphpath,
        height=2000,
        width=1500,
        n_batches=n_batches,
    )

    c = 0
    for i in tqdm(list(range(0, len(params), step)), 'Saving loss and predictions'):
        l = best_loss[i]
        plotBestLoss(best_loss[:i], losses, f'Best loss ={l:.4f}', str(losspath / f'{c}.png'))
        plotModelOutput(model, params[i], outfile=str(predpath / f'{c}.png'))
        c += 1


def plotGrads(gradlist):
    import jax
    import jax.numpy as jnp

    vmax = 10
    agg = 100
    nbins = 82
    flattened_grads = np.array(
        [np.concatenate([f.flatten() for g in grad for f in g]) for grad in gradlist]
    )
    s = np.add.reduceat(flattened_grads, range(0, len(flattened_grads), agg))
    bins = np.concatenate(
        [
            np.zeros(1),
            np.logspace(np.log10(1e-4), np.log10(vmax + 1), nbins // 2 - 1, endpoint=True),
            np.array([np.inf]),
        ]
    )
    bins = np.concatenate([-bins[1:][::-1], bins])

    def hist(bins, a):
        h = jnp.histogram(a, bins=bins)[0]
        return h / jnp.max(h)

    gradhist = jax.vmap(jax.jit(partial(hist, bins)))(s)

    labels_interval = 10
    binlabels = [f'{l:.1E}' for l in bins]
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.pcolormesh(gradhist.T, cmap='GnBu')
    ax.set_xlabel('epochs')
    ax.set_yticks(np.arange(len(bins))[1:-1:labels_interval], binlabels[1:-1:labels_interval])
    plt.show()


# ut.plotBestLoss(best_loss, losses, 'Best loss history')
# ut.plotModelOutput(model, best_params)

##
# import nest_asyncio
# nest_asyncio.apply()
# ut.screenCaptures(partial(ut.drawComputeGraph, height=2000), compg_history[::10], out_dir_path='../__out/test', height=2000, width=1500)


def plot_node(
    ntype,
    params,
    model,
    xlim=[0.0, 1.0],
    ylim=[0.0, 1.0],
    n_samples=200,
    figsize=(12, 7),
    n_inputs=1,
    cmap=DEFAULT_CMAP,
    mode='heatmap',
    extra_args=None,
):
    import jax
    import jax.numpy as jnp

    quantized_per_type = model.get_quantized_parameters_per_node_type(params)
    # quantized per type is a dict of node_name -> dict
    # for each node_name that starts with inv_, we need to copy the dict of the node without the inv_ prefix
    for k in list(quantized_per_type.keys()):
        if k.startswith('inv_'):
            if k[4:] in quantized_per_type:
                quantized_per_type[k].update(quantized_per_type[k[4:]])

    def get_q(__, v, **_):
        return v

    def get_p(param_name, shared=False, index={}, **_):
        if shared:
            return params['shared'][param_name]
        else:
            # if ntype starts with inv_ then we need to remove the inv_ prefix
            val: dict(str, jnp.array) = quantized_per_type[ntype][param_name]
            val = [val[k] for k in sorted(val.keys())]
            return val[index[param_name]]

    counter_max = {k: len(v) for k, v in quantized_per_type[ntype].items()}
    counter_val = {k: 0 for k in counter_max.keys()}
    counter_order = list(counter_max.keys())
    n_combinations = np.prod(list(counter_max.values()))

    if n_inputs == 1:
        X = np.linspace(xlim[0], xlim[1], n_samples).reshape(-1, 1)
    elif n_inputs == 2:
        x = np.linspace(xlim[0], xlim[1], n_samples)
        X = np.array(np.meshgrid(x, x)).T.reshape(-1, 2)

    else:
        raise NotImplementedError()
    print('X shape:', X.shape)

    if n_inputs == 1:
        # simple plots, on the same figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_xlabel('input')
        ax.set_ylabel('output')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # set correct ratio (depends on xlim and ylim)
        ax.set_aspect('equal')

    elif n_inputs == 2:
        # we will have a grid of n_combinations heatmaps
        n_rows = int(np.ceil(np.sqrt(n_combinations)))
        n_cols = int(np.ceil(n_combinations / n_rows))
        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
        ax = np.array(ax).flatten()
        img = []

    fig.suptitle(f'{ntype} node')
    if extra_args is not None:
        fig.text(0.5, 0.92, extra_args, ha='center', fontsize=10)
    ax_idx = 0

    while True:

        def vf(x, z):
            if extra_args is not None:
                f = model.node_impl[ntype](partial(get_p, index=counter_val), get_q, **extra_args)
            else:
                f = model.node_impl[ntype](partial(get_p, index=counter_val), get_q)
            return f(*x, quantile=z, rng_key=jax.random.PRNGKey(0))

        Y = jax.vmap(vf, in_axes=(0, None))(X, jnp.asarray([0.5]))

        values = [
            sorted(quantized_per_type[ntype][pname].keys())[counter_val[pname]]
            for pname in counter_order
        ]
        label = f'''{", ".join([f"{pname}={v.split('::')[0]}" for pname, v in zip(counter_order, values)])}'''

        if n_inputs == 1:
            ax.plot(X, Y, label=label)
        else:
            if mode == 'heatmap':
                Y = Y.reshape(n_samples, n_samples)
                img.append(
                    ax[ax_idx].pcolormesh(
                        X[:, 0].reshape(n_samples, n_samples),
                        X[:, 1].reshape(n_samples, n_samples),
                        Y,
                        cmap=cmap,
                    )
                )
                # label the axes
                ax[ax_idx].set_xlabel('input 1')
                ax[ax_idx].set_ylabel('input 2')
                ax[ax_idx].set_title(label)
                ax[ax_idx].set_aspect('equal')
            elif mode == '3d':
                ax[ax_idx].remove()
                ax[ax_idx] = fig.add_subplot(n_rows, n_cols, ax_idx + 1, projection='3d')
                # then plot
                Y = Y.reshape(n_samples, n_samples)
                img.append(
                    ax[ax_idx].plot_surface(
                        X[:, 0].reshape(n_samples, n_samples),
                        X[:, 1].reshape(n_samples, n_samples),
                        Y,
                        cmap=cmap,
                        edgecolor='k',
                        linewidth=0.1,
                    )
                )
                # label the axes
                ax[ax_idx].set_xlabel('input 1')
                ax[ax_idx].set_ylabel('input 2')
                ax[ax_idx].set_zlabel('output')
                ax[ax_idx].set_title(label)
            else:
                raise NotImplementedError()

            ax_idx += 1

        # increment counter until we reach the max for each parameter
        # starting from first parameter
        for i in range(len(counter_order)):
            counter_val[counter_order[i]] += 1
            if counter_val[counter_order[i]] < counter_max[counter_order[i]]:
                break
            else:
                counter_val[counter_order[i]] = 0
        if all([v == 0 for v in counter_val.values()]):
            break

    if n_inputs == 1:
        ax.legend()

    if n_inputs == 2:
        for i in range(ax_idx, len(ax)):
            ax[i].axis('off')
        # also use the same colorbar for all heatmaps, with the same range
        if mode == 'heatmap':
            vmin = np.min([i.get_array().min() for i in img])
            vmax = np.max([i.get_array().max() for i in img])
            for i in img:
                i.set_clim(vmin, vmax)
            fig.colorbar(img[0], ax=ax, shrink=0.6)
        else:  # 3d
            # we want to have the same zlim for all plots
            zlim = np.array([ax[i].get_zlim() for i in range(ax_idx)])
            zlim = np.array([zlim[:, 0].min(), zlim[:, 1].max()])
            for i in range(ax_idx):
                ax[i].set_zlim(zlim)
            # also add a colorbar (somewhere that is not on top of the plots)
            fig.colorbar(img[0], ax=ax, shrink=0.6)

    return fig, ax


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                        --     jax prints     --
# ···············································································


def get_jaxpr(fun, *args, **kwargs):
    import jax

    return jax.make_jaxpr(fun)(*args, **kwargs)


def print_jaxpr(fun, *args, **kwargs):
    get_jaxpr(fun, *args, **kwargs).pretty_print()


def get_xla(fun, *args, static_argnums=(), **kwargs):
    import jax
    import jaxlib.xla_extension as xla_ext

    console = Console(highlighter=rich.highlighter.ReprHighlighter())
    c = jax.xla_computation(fun, static_argnums=static_argnums)(*args, **kwargs)
    backend = jax.lib.xla_bridge.get_backend()
    e = backend.compile(c)
    option = xla_ext.HloPrintOptions.short_parsable()
    out = e.hlo_modules()[0].to_string(option)
    return out


def print_xla(fun, *args, static_argnums=(), **kwargs):
    print(get_xla(fun, *args, **kwargs))


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     load save     --
# ···············································································
def readimg(p, threshold=None, size=None):
    import jax.numpy as jnp

    im = Image.open(p)
    if size is not None:
        im = im.resize(size)
    im = jnp.array(im) / 255
    if threshold:
        im = (im > threshold) * 1.0
    return im


def np_converter(obj):
    import jax.numpy as jnp

    if isinstance(obj, (np.integer, jnp.integer)):
        return int(obj)
    elif isinstance(obj, (np.floating, jnp.floating)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, jnp.ndarray)):
        return obj.tolist()
    elif isinstance(obj, np.bool_) or isinstance(obj, jnp.bool_):
        return bool(obj)
    elif np.isnan(obj) or jnp.isnan(obj):
        return None


# parse_float=lambda x: round(float(x), 3)
def make_json_compatible(o, converter=np_converter, float_precision=None):
    if float_precision is not None:
        return json.loads(
            json.dumps(o, default=converter), parse_float=lambda x: round(float(x), float_precision)
        )
    else:
        return json.loads(json.dumps(o, default=converter))


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                   --      hooked_scan for jax     --
# ···············································································


def hooked_scan(num_samples, on_update, call_rate=1):
    import jax
    from jax import lax
    from jax.experimental import host_callback

    def update(args, transform):
        result, iternum = args
        carry, acc = result
        on_update(acc, iternum)

    def _update_(result, iter_num):
        return lax.cond(
            (iter_num % call_rate == 0) | (iter_num == num_samples - 1),
            lambda _: host_callback.id_tap(update, (result, iter_num), result=result),
            lambda _: result,
            operand=None,
        )

    def _hooked_scan(func):
        @jax.jit
        def wrapper(carry, x):
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x
            result = func(carry, x)
            return _update_(result, iter_num)

        return wrapper

    return _hooked_scan


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
