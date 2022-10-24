import pandas as pd
from time import time
from jax import vmap, jit
from pathlib import Path
from pyppeteer import launch
from jax import tree_util as pytree
import jax.numpy as jnp
import urllib.parse
import sys
import streamlit as st
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
import copy
from rich import print as rprint
import rich
from rich.console import Console
import jaxlib.xla_extension as xla_ext
from rich.progress import track

from typing import List


class ddict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def is_interactive():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return True  # Terminal running IPython
        if not hasattr(sys, 'ps1'):
            return True
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def np_converter(obj):
    # print type:
    print(type(obj))
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif np.isnan(obj):
        return None


def make_json_compatible(o):
    return json.loads(json.dumps(o, default=np_converter))


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{              --     Streamlit utils and components     --
# ···············································································
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




def computeGraph(nodes, edges, key=None, func=_component_func, **kwargs):
    def filterType(n):
        if n['type'] == 'input':
            n['type'] = 'in'
        if n['type'] == 'output':
            n['type'] = 'out'
        return n

    tnodes = [filterType(n) for n in nodes]
    return func(nodes=tnodes, edges=edges, output_type='COMPUTE', key=key, **kwargs)


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

def drawComputeGraph(df, func=None, cdg=None, **kwargs):
    uidGen = bc.ut.uniqueIdGenerator()
    nodes = [
        {'id': str(i), 'type': n.type, 'data': bc.ut.updated_dict(n.to_dict(), {'id': i})}
        for i, n in df.iterrows()
    ]
    edges = []
    for i, n in df.iterrows():
        if n.output_to:
            for n_out, (o, h) in enumerate(n.output_to):
                srccdg = None
                if cdg is not None:
                    cdgin = df.loc[o].cdg_input
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
                        'srcdata': df.loc[i].to_dict(),
                        'tgtdata': df.loc[o].to_dict(),
                        'srccdg': srccdg,
                        'tgthandle': str(h),
                    },
                }
                edges.append(edge)

    if func is None:
        return computeGraph(make_json_compatible(nodes), make_json_compatible(edges), **kwargs)
    else:
        return computeGraph(
            make_json_compatible(nodes), make_json_compatible(edges), func=func, **kwargs
        )


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

browser=None
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
            # await page.screenshot({'path': outfile+'.png', 'omitBackground': True, 'type': 'png', 'clip':{'x': 0, 'y': 0, 'width': width, 'height': height}})
            await page.pdf(
                {
                    'path': outfile,
                    'width': width,
                    'height': height,
                    'pageRanges': '1',
                    'printBackground': False,
                }
            )
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



def plot_networks(nets: List[bc.Network], filenames):
    import nest_asyncio
    nest_asyncio.apply()

    H = 1000
    W = 1000

    def draw_network(net, *a, **kw):
        drawComputeGraph(net.compute_graph, *a, height=H, width=W, cdg=net.central_dogma_graph, **kw)

    screenCaptures(
        draw_network,
        nets,
        filenames=filenames,
        height=H,
        width=W,
    )

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
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from pathos.pools import ProcessPool

import jax
from multiprocess import Process


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
    flist = [[0.0, 0.493, 0.579], [0.896, 0.866, 0.806], [0.844, 0.1, 0.111]]
    teals = [
        [0.957, 0.913, 0.804],
        [0.613, 0.745, 0.734],
        [0.465, 0.672, 0.635],
        [0.272, 0.507, 0.535],
        [0.01, 0.1, 0.15],
    ]
    cmap = LinearSegmentedColormap.from_list("", teals)
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

    gradhist = vmap(jit(partial(hist, bins)))(s)

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


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


def print_jaxpr(fun, *args, **kwargs):
    print(jax.make_jaxpr(fun)(*args, **kwargs))


def print_xla(fun, *args, **kwargs):
    console = Console(highlighter=rich.highlighter.ReprHighlighter())
    c = jax.xla_computation(fun)(*args, **kwargs)
    backend = jax.lib.xla_bridge.get_backend()
    e = backend.compile(c)
    option = xla_ext.HloPrintOptions.short_parsable()
    out = e.hlo_modules()[0].to_string(option)
    print(out)


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


def readimg(p, threshold=None, size=None):
    im = Image.open(p)
    if size is not None:
        im = im.resize(size)
    im = jnp.array(im) / 255
    if threshold:
        im = (im > threshold) * 1.0
    return im


from jax import lax
from jax.experimental import host_callback


def hooked_scan(num_samples, on_update, call_rate=1):
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
