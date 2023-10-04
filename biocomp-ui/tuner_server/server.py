### {{{                          --     imports     --
import sys
import urllib

from dataclasses import dataclass
from typing import List
import pprint
import io

sys.path.append('../../scripts/')
from flask_cors import CORS
from flask import Flask, request
from pathlib import Path
import pandas as pd
import json
from flask_cors import CORS, cross_origin
import biocomp as bc
from tqdm import tqdm
import json5
import scriptutils as su

from biocomp import utils as ut
import biocomp.datautils as du
import time
import biocomp.train as train
import biocomp.compute as cmp
import biocomp.parameters as pm
import biocomp as bc
from biocomp.parameters import ParameterTree
from jax.tree_util import Partial as partial
import jax.tree_util as jtu
from pathlib import Path
import jax.numpy as jnp
from copy import deepcopy
import optax
from tqdm import tqdm
import numpy as np
import jax
from jax import jit, grad, vmap, random, value_and_grad
from jax import numpy as jnp
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

from tempfile import NamedTemporaryFile
import base64
from flask import current_app

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                           --     utils     --


def tag_tunable_params(stack, params):
    local_params, _ = params.filter_by_tag('local')
    tunable_param_names = ['tl_rate', 'ratios', 'value']
    nlayers = len(stack.layers)
    for i in range(nlayers):
        local_param_prefix = f'local/{i}'
        if local_param_prefix in params.data:
            sub_params = params[local_param_prefix]
            for l, v in sub_params.iter_leaves():
                if any([p in l for p in tunable_param_names]):
                    fullpath = f'{local_param_prefix}/{l}'
                    isref = pm.isArrayRef(
                        local_params.data.get_at(fullpath, get_leaf_value=False).value
                    )
                    if not isref:
                        params.tag(fullpath, 'tunable')


# Now let's make a cnode_id: list of (path, loc, value) pairs
def make_param_map(stack, tunable_params):
    from collections import defaultdict

    param_map = defaultdict(list)
    assert len(stack.networks) == 1
    for l, v in tunable_params.data.iter_leaves():
        layer_id = int(l.path[1])
        layer = stack.layers[layer_id]
        pname = l.path[-1]
        for i, n in enumerate(layer.nodes):
            cid = n.compute_node_id
            param_map[cid].append((str(l), i, pname, v[i].tolist()))
    return dict(param_map)


def apply_param_map(param_map, params):
    new_params = deepcopy(params)
    for cid, path_loc_val in param_map.items():
        for path, loc, name, val in path_loc_val:
            old_val = np.array(new_params[path])
            old_val[loc] = val
            new_params[path] = old_val
            new_params.tags[path] = params.tags[path]
    return new_params


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                      --     load parameters     --

biocompdir = Path('~/.biocompiler').expanduser()

# parameters
training_archive = du.load(biocompdir / 'training_archives/20230923_fulltrain_v0.pkl')
shared_parameters = training_archive['parameters']
compute_config = training_archive['compute_config']
training_config = training_archive['training_config']
compute_config.set_impl('bias', bc.nodes.bias)


def init_stack(stack, rng):
    local_params, _ = stack.init(rng).filter_by_tag('local')
    local_params.data.check()
    full_params = ParameterTree.merge(local_params, shared_parameters)
    return full_params


def get_output_indices(stack):
    out_indices = []
    for n_id, n in enumerate(stack.networks):
        output_protein_names = n.get_dependent_output_proteins()
        print(f'output_protein_names: {output_protein_names}')
        assert len(output_protein_names) == 1
        output_id = n.get_output_proteins().index(output_protein_names[0])
        out_indices.append(stack.get_network_global_output_id(n_id, output_id))
    return jnp.array(out_indices)


class NetworkEntry:
    def __init__(self, name, network, use_base_tunable=None, rng=None):
        self.name = name
        self.network = network

        self.output_indices = get_output_indices(cmp.ComputeStack([self.network]))

        self.stack = cmp.ComputeStack([network])
        self.stack.build(compute_config)

        rng = random.PRNGKey(0) if rng is None else rng
        full_params = init_stack(self.stack, rng)
        tag_tunable_params(self.stack, full_params)

        self.base_tunable_params, self.non_tunable_params = full_params.filter_by_tag('tunable')

        if use_base_tunable is not None:
            self.set_base_tunable_params(use_base_tunable)
        else:
            for l, v in self.base_tunable_params.data.iter_leaves():
                if 'ratios' in l.path:
                    self.base_tunable_params[l] = np.zeros_like(v)
        key = jax.random.PRNGKey(0)
        for l, v in self.base_tunable_params.data.iter_leaves():
            if 'tl_rate' in l.path:
                self.base_tunable_params[l] = jax.random.uniform(key, shape=v.shape, minval=-0.1, maxval=0.4)


    def set_base_tunable_params(self, pmap):
        self.base_tunable_params = apply_param_map(pmap, self.base_tunable_params)
        key = jax.random.PRNGKey(0)
        for l, v in self.base_tunable_params.data.iter_leaves():
            if 'tl_rate' in l.path:
                self.base_tunable_params[l] = jax.random.uniform(key, shape=v.shape, minval=-0.1, maxval=0.4)

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_at(self, params, X, Z, key):
        keys = jax.random.split(key, X.shape[0])
        vmapped_compute = jax.vmap(self.stack.apply, in_axes=(None, 0, 0, 0))
        full_yhat, _ = vmapped_compute(params, X, Z, keys)
        yhat = full_yhat[:, self.output_indices]
        if yhat.ndim == 1:
            yhat = yhat.reshape(-1, 1)
        return yhat

    def get_graph(self):
        nodes, edges = su.network_to_graph(self.network)
        return {'nodes': nodes, 'edges': edges, 'output_type': 'COMPUTE'}

    def get_param_map(self):
        return make_param_map(self.stack, self.base_tunable_params)


network_registry: dict[str, NetworkEntry] = {}


def tr(x):
    return du.tr(
        np.asarray(x),
        offset=training_config['data_log_offset'],
        maxv=training_config['data_max_value'],
        factor=training_config['data_log_factor'],
        threshold=training_config['data_log_poly_threshold'],
        compression=training_config['data_log_poly_compression'],
    )

def invtr(x):
    return du.inv_tr(
        np.asarray(x),
        offset=training_config['data_log_offset'],
        maxv=training_config['data_max_value'],
        factor=training_config['data_log_factor'],
        threshold=training_config['data_log_poly_threshold'],
        compression=training_config['data_log_poly_compression'],
    )

invtr(0.5)/invtr(0.3)
x = np.linspace(0.3, 0.5, 10)
y = invtr(x)/invtr(0.3)
# plot x vs y
plt.plot(x, y)

def make_symlog_ax(
    ax, xlims=None, ylims=None, skip10=True, margins=0.05
):
    xlims_tr, ylims_tr = None, None
    if xlims is not None:
        xlims_tr = tr(np.asarray(xlims))
        xp10 = du.powers_of_ten(*xlims)
        xlims_margin = xlims_tr + np.array([-1, 1]) * margins * np.diff(xlims_tr)
        ax.set_xlim(xlims_margin)
        ax.set_xticks(tr(xp10))
        ax.xaxis.set_major_formatter(du.PowerFormatter(xp10, skip10=skip10))
    if ylims is not None:
        ylims_tr = tr(np.asarray(ylims))
        yp10 = du.powers_of_ten(*ylims)
        ylims_margin = ylims_tr + np.array([-1, 1]) * margins * np.diff(ylims_tr)
        ax.set_ylim(ylims_margin)
        ax.set_yticks(tr(yp10))
        ax.yaxis.set_major_formatter(du.PowerFormatter(yp10, skip10=skip10))
    return xlims_tr, ylims_tr


def plot_eval(network_entry, params, res=100, xlims=(0, 1), vlims=(0, 1)):

    # X = np.meshgrid(np.linspace(*xlims, res), np.linspace(*xlims, res))
    # X = np.stack(X, axis=-1).reshape(-1, 2)

    key = jax.random.PRNGKey(0)
    xx = np.linspace(*xlims, res)
    X = np.array(np.meshgrid(xx, xx)).T.reshape(-1, 2)
    Z = jnp.ones((res * res, network_entry.stack.total_nb_of_outputs)) * 0.5
    Y = network_entry.evaluate_at(params, X, Z, key)

    fig, ax = du.mkfig(1, 1)
    im = ax.imshow(
        Y.reshape(res, res),
        extent=[*xlims, *xlims],
        cmap='YlGnBu',
        origin='lower',
        # vmin=vlims[0],
        # vmax=vlims[1],
    )


    # xlims_tr, ylims_tr = make_symlog_ax(ax, xlims=invtr(xlims), ylims=invtr(xlims))

    # colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cb = fig.colorbar(im, cax=cax, orientation='vertical')


    vmin = np.nanmin(Y)
    vmax = np.nanmax(Y)
    unscaled_ticks = np.geomspace(du.inv_tr(vmin), du.inv_tr(vmax), 5, endpoint=True)
    ticks = np.array(tr(unscaled_ticks))
    print(f'unscaled_ticks : {unscaled_ticks}')
    print(f'vmin = {vmin}, vmax = {vmax}, ticks = {ticks}')
    ticks = ticks[ticks < vmax]
    ticks = ticks[ticks > vmin]
    ticklabels = [ du.scformat.format("{:m}", du.inv_tr(x)) for x in ticks ]
    cb.set_ticks(ticks)
    cb.set_ticklabels(ticklabels)



    # cb.set_label('fold change')
    # cb.set_ticks(np.linspace(*vlims, 5))
    # cb.set_ticklabels(np.round(invtr(np.linspace(*vlims, 5))/invtr(vlims[0]), 2))

    ax.set_xlabel('$X_1$')
    ax.set_ylabel('$X_2$')
    # set label font size
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    # and for cbar
    cb.ax.tick_params(labelsize=20)


    # hide x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax


def get_network(
    name: str, use_base_tunable=None, cachedir=biocompdir / 'cache/networks', rng=None
) -> NetworkEntry:
    global network_registry
    fullpath = str(cachedir / f'{name}.pkl')

    if fullpath not in network_registry:
        current_app.logger.info(f'Network {name} not found in registry, loading from {cachedir}')
        if not Path(fullpath).exists():
            current_app.logger.error(f'Network {name} not found in {cachedir}')
            return None
        network = du.load(cachedir / f'{name}.pkl')
        network_entry = NetworkEntry(name, network, rng)
        network_registry[fullpath] = network_entry

    if use_base_tunable is not None:
        print(f'Using base tunable params: {use_base_tunable}')
        network_registry[fullpath].set_base_tunable_params(use_base_tunable)

    return network_registry[fullpath]


def get_network_names(cachedir=biocompdir / 'cache/networks'):
    return [p.stem for p in cachedir.glob('*.pkl')]


get_network_names()

##────────────────────────────────────────────────────────────────────────────}}}


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/')
def index():
    return 'Hello from the tuner server!'


@app.route('/network/<network_name>')
def _network(network_name):
    current_app.logger.info(f'Getting network {network_name}')
    use_base_tunable = request.args.get('init_params')
    if use_base_tunable is not None:
        # it's a json string in base64
        use_base_tunable = json.loads(base64.b64decode(use_base_tunable))
    current_app.logger.info(f'Using base tunable params: {use_base_tunable}')
    network = get_network(network_name, use_base_tunable=use_base_tunable)
    return json.dumps(network.get_graph())


@app.route('/params/<network_name>')
def _params(network_name):
    current_app.logger.info(f'Getting params for {network_name}')
    network = get_network(network_name)
    return json.dumps(network.get_param_map())


@app.route('/simulate', methods=['POST'])
def _simulate():
    data = request.json
    pmap = data['params']
    if pmap is None:
        return json.dumps({'image': None})

    network = get_network(data['network_name'])
    new_tunable = apply_param_map(pmap, network.base_tunable_params)
    params = ParameterTree.merge(new_tunable, network.non_tunable_params)
    print(new_tunable.data)

    fig, _ = plot_eval(network, params, res=100, xlims=(0.2, 0.75), vlims=(0.3, 0.6))
    imIObytes = io.BytesIO()
    fig.savefig(imIObytes, format='png')
    imIObytes.seek(0)
    data = 'data:image/png;base64,' + base64.b64encode(imIObytes.read()).decode('utf-8')
    plt.close(fig)
    return json.dumps({'image': data})


if __name__ == '__main__':
    # app.run(host="0.0.0.0", port="4321")
    app.run(host="0.0.0.0", port="4321", debug=True, use_reloader=True)
