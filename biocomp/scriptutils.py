import pandas as pd
from time import time
from pathlib import Path
import urllib.parse
from pyppeteer import launch
import sys
import os
from functools import partial
import asyncio
from PIL import Image
from tqdm import tqdm
import json
import numpy as np
import biocomp as bc
import pickle
import biocomp.datautils as du
import biocomp.utils as ut
import rich
from rich.console import Console
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

from matplotlib.colors import LinearSegmentedColormap


### {{{              --     streamlit utils and components     --
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


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                    --     graph serialization     --
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

    return ut.make_json_compatible(nodes), ut.make_json_compatible(edges)


def drawComputeGraph(network, func=_component_func, **kwargs):
    nodes, edges = network_to_graph(network)
    return func(nodes=nodes, edges=edges, output_type='COMPUTE', **kwargs)

##────────────────────────────────────────────────────────────────────────────}}}


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


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
