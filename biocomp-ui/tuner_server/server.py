### {{{                          --     imports     --
import sys
import urllib

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


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                      --     load parameters     --

biocompdir = Path('~/.biocompiler').expanduser()

cachedir = biocompdir / 'cache/networks'
NETWORK = du.load(cachedir / 'full_bandpass.pkl')


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


# generate the compute stack
stack = cmp.ComputeStack([NETWORK])
stack.build(compute_config)


output_indices = get_output_indices(stack)

rng = jax.random.PRNGKey(0)
full_params = init_stack(stack, rng)


nodes, edges = su.network_to_graph(NETWORK)
network_json = json.dumps({'nodes': nodes, 'edges': edges, 'output_type': 'COMPUTE'})


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


tag_tunable_params(stack, full_params)
tunable_params, non_tunable = full_params.filter_by_tag('tunable')


# Now let's make a cnode_id: list of (path, loc, value) pairs
def make_param_map(stack, tunable_params):
    from collections import defaultdict
    param_map = defaultdict(list)
    assert len(stack.networks) == 1
    for l, v in tunable_params.data.iter_leaves():
        layer_id = int(l.path[1])
        layer = stack.layers[layer_id]
        pname = l.path[-1]
        for i,n in enumerate(layer.nodes):
            cid = n.compute_node_id
            param_map[cid].append((str(l), i, pname, v[i].tolist()))
    return dict(param_map)

param_map = make_param_map(stack, tunable_params)
strmap = json.dumps(param_map)
# param_map = json.loads(strmap)


def apply_param_map(param_map, params):
    new_params = deepcopy(params)
    for cid, path_loc_val in param_map.items():
        for path, loc, name, val in path_loc_val:
            old_val = np.array(params[path])
            old_val[loc] = val
            new_params[path] = old_val
            new_params.tags[path] = params.tags[path]
    return new_params


new_params = apply_param_map(param_map, tunable_params)

vmapped_compute = jax.vmap(stack.apply, in_axes=(None, 0, 0, 0))

@jit
def evaluate_at(params, X, Z, key):
    keys = jax.random.split(key, X.shape[0])
    full_yhat, _ = vmapped_compute(params, X, Z, keys)
    yhat = full_yhat[:, output_indices]
    if yhat.ndim == 1:
        yhat = yhat.reshape(-1, 1)
    return yhat

def plot_eval(params, res=100, xlims=(0, 0.7)):
    key = jax.random.PRNGKey(0)
    X = np.meshgrid(np.linspace(*xlims, res), np.linspace(*xlims, res))
    X = np.stack(X, axis=-1).reshape(-1, 2)
    Z = jnp.ones((res * res, stack.total_nb_of_outputs)) * 0.5
    Y = evaluate_at(params, X, Z, key)
    fig, ax = du.mkfig(1, 1)
    im = ax.imshow(
        Y.reshape(res, res),
        extent=[*xlims, *xlims],
        cmap='YlGnBu',
        origin='lower',
        vmin=0,
        vmax=0.6,
    )
    ax.contour(
        Y.reshape(res, res),
        [0.2, 0.4, 0.5],
        extent=[*xlims, *xlims],
        colors='k',
        origin='lower',
        alpha=0.25,
    )
    # colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xlabel('$X_1$')
    ax.set_ylabel('$X_2$')
    return fig, ax

##────────────────────────────────────────────────────────────────────────────}}}


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/')
def index():
    return 'Hello from the tuner server!'


@app.route('/params')
def _params():
    return json.dumps(param_map)

@app.route('/network')
def _network():
    return network_json



@app.route('/simulate', methods=['POST'])
def _simulate():
    data = request.json
    params = data['params']
    new_tunable = apply_param_map(params, tunable_params)
    full_params = ParameterTree.merge(new_tunable, non_tunable)
    full_params.data.check()

    # run simulation
    fig, ax = plot_eval(full_params, res=50, xlims=(0, 0.7))
    with NamedTemporaryFile(suffix='.png') as f:
        fig.savefig(f.name)
        f.seek(0)
        data = f.read()

    plt.close(fig)
    plt.close('all')

    data = 'data:image/png;base64,'+base64.b64encode(data).decode('utf-8')
    return json.dumps({'image': data})



if __name__ == '__main__':
    # app.run(host="0.0.0.0", port="4321")
    app.run(host="0.0.0.0", port="4321", debug=True, use_reloader=True)
