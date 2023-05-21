### {{{                          --     imports     --
import matplotlib

import biocomp as bc
from biocomp import datautils as du
from functools import partial
from datetime import datetime
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import jax
from jax import jit, vmap, value_and_grad
import jax.numpy as jnp
import pickle

from biocomp import utils as ut
import scriptutils as su
import biocomp.datautils as du
from biocomp import train
from biocomp import compute as cmp

# matplotlib.use('agg')
matplotlib.rcParams['figure.dpi'] = 200

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                --     retrieving runs and losses     --

project_name = 'cascades_v1'
runs, losses = du.retrieve_wandb_results(project_name)

##
##────────────────────────────────────────────────────────────────────────────}}}

### {{{                         --     loss plot     --

fig, ax = du.mkfig(1, 1, (7, 5))
with ut.timer('Loss plot'):
    du.losses_plot(losses, ax, runs=runs)
fig.savefig('/Users/jeandisset/Desktop/bestloss.pdf')
best_run = runs[du.get_best_run_id(losses)]
print('Best run:', best_run.name)

##────────────────────────────────────────────────────────────────────────────}}}##

### {{{                      --     loading xp     --

training_config = train.DEFAULT_TRAINING_CONFIG
compute_config = cmp.DEFAULT_COMPUTE_CONFIG

XP = {
    'bt': '2023-04-03_Constraints_Pgu_Bleedthrough',
    'cascades': '2023-04-18_Constraints_PguCascades',
    'csy4matrix': '2023-03-26_MatrixCsy4',
    'casematrix': '2023-02-16_Matrix',
}
xpnames = ['bt', 'cascades', 'csy4matrix', 'casematrix']

with ut.timer(f'Loading data and building networks for {xpnames}'):
    lib = su.load_lib()
    loadedxp = {xpname: su.load_xp(XP[xpname], lib) for xpname in xpnames}
    dman_full = du.DataManager.from_xps(loadedxp.values(), training_config, inverse='all')

all_networks = dman_full.get_networks()
net_xp = [n.metadata['from_xp'] for n in all_networks]
net_name = [n.name for n in all_networks]

##────────────────────────────────────────────────────────────────────────────}}}

### {{{               --     training and validation sets     --

# list net names that have cascade in the name:
inert_nets = {n: i for i, n in enumerate(net_name) if 'inert' in n.lower()}
cascade_nets = {
    n: i for i, n in enumerate(net_name) if 'cascade' in n.lower() and 'inert' not in n.lower()
}

# training set is all networks except the ones in inert or cascade
training_set = [
    i
    for i, _ in enumerate(net_name)
    if i not in inert_nets.values() and i not in cascade_nets.values()
]

validation_set = [
    i for i, _ in enumerate(net_name) if i not in inert_nets.values() and i in cascade_nets.values()
]

n_outputs = [n.get_nb_outputs() for n in all_networks]


##────────────────────────────────────────────────────────────────────────────}}}

import joblib

with ut.timer('Building compute stack'):
    key = jax.random.PRNGKey(0)
    full_stack = dman_full.build_compute_stack(compute_config)
    substack, _ = full_stack.make_subset([50])

try:
    best_params = joblib.load(f'../__cache/best_params.pkl')
    print('Loaded best params from cache')
except FileNotFoundError:
    with ut.timer('Stack initialization'):
        base_params = full_stack.init(key)

    tmp_dir = Path(f'./{project_name}')
    param_file = best_run.file('latest_params.pkl').download(replace=True, root=tmp_dir)
    with open(param_file.name, 'rb') as f:
        trained_params = pickle.load(f)

    best_params = full_stack.use_shared_params(base_params, trained_params)
    joblib.dump(best_params, f'../__cache/best_params.pkl')

### {{{                    --     training data plots     --

savepath = Path(f'~/Desktop/predictions/lvl2_cascades_v1').expanduser()
savepath.mkdir(parents=True, exist_ok=True)

# bash command to remove all files with "==" in the name:
# find . -name "*==*" -type f -delete
import matplotlib
import matplotlib as plt

# matplotlib.pyplot.switch_backend('Agg')
dman_full._densities = None

networks = dman_full.get_networks()
stack = full_stack
params = best_params

net_ids = list(range(len(networks)))[50:54]


def smooth_line_plots_slices(
    x, y, net, rescale, mkslices, ngslices, axes, input_order, input_names, **kwargs
):
    for i, ng in enumerate(ngslices):
        ax = axes[i]
        for mk in mkslices:
            du.smooth_line_plots(
                x,
                y,
                net,
                rescale,
                ax=ax,
                slice=[mk, ng],
                input_order=input_order,
                radius=0.3,
                label=f'{input_names[input_order[1]]} ≈ {int(mk*100)}%',
                lw=2,
            )
        # symbol for approx equal is:
        # add text for ng levl:
        ax.text(
            0.5,
            0.75,
            f'{input_names[input_order[2]]}={int(ng*100)}%',
            transform=ax.transAxes,
            ha='center',
            va='center',
        )


with ut.timer('pred plot'):
    N_SAMPLES_PER_CHUNK = 5000
    N_CHUNKS = 3

    N_SAMPLES_TOTAL = N_SAMPLES_PER_CHUNK * N_CHUNKS

    key = jax.random.PRNGKey(0)
    X, Y = dman_full.get_uniform_samples(key, N_SAMPLES_TOTAL)
    assert len(X) == len(Y)
    assert len(X) == len(networks)

    X = [np.expand_dims(arr, axis=1) if arr.ndim == 1 else arr for arr in X]
    Y = [np.expand_dims(arr, axis=1) if arr.ndim == 1 else arr for arr in Y]

    ALLX = np.concatenate(X, axis=1)

    assert ALLX.shape == (
        N_SAMPLES_TOTAL,
        stack.total_nb_of_inputs,
    ), f"{ALLX.shape} != {(N_SAMPLES_TOTAL, stack.total_nb_of_inputs)}"

    @jit
    def compute(params, XX, Q, keys):
        res, _ = stack.apply(params, XX, Q, keys)
        return res

    ALLX_CHUNKS = np.split(ALLX, N_CHUNKS, axis=0)

    YHAT = []

    for chunk_id, XX in enumerate(tqdm(ALLX_CHUNKS, desc='plot_pred chunks')):
        Q = jax.random.uniform(key, (N_SAMPLES_PER_CHUNK, stack.total_nb_of_outputs))
        keys = jax.random.split(key, N_SAMPLES_PER_CHUNK)
        key = keys[-1]
        yhat_chunk = vmap(compute, in_axes=(None, 0, 0, 0))(params, XX, Q, keys)
        YHAT.append(np.array(yhat_chunk))

    YHAT = np.concatenate(YHAT, axis=0)

    def plot_prediction(index):
        out_id = stack.get_network_global_output_id(index)
        n_out = networks[index].get_nb_outputs()
        x, y = X[index], Y[index]
        yhat = YHAT[: x.shape[0], out_id : out_id + n_out]
        assert yhat.shape == y.shape, f"{yhat.shape} != {y.shape}"
        error = np.abs(y - yhat).mean()
        input_order = [0, 1, 2]

        # fig = du.report(params, dman_full, index, use_x_y_yhat=(x, y, yhat), res=128, input_order=input_order)
        # fig.suptitle(f'{fig._suptitle.get_text()}\nerror: {error:.3f}\n{seen}')

        ngslices = [0.25, 0.5, 0.75]
        mkslices = [0.05, 0.3, 0.5]

        fig, allaxes = du.mkfig(2, 3)
        input_order = ([0, 1, 2],)
        net = networks[index]
        input_names = net.get_inverted_input_proteins()

        smooth_line_plots_slices(
            x,
            y,
            net=net,
            rescale=dman_full.rescale,
            mkslices=mkslices,
            ngslices=ngslices,
            axes=allaxes[0],
            input_order=input_order,
            input_names=input_names,
        )

        smooth_line_plots_slices(
            x,
            yhat,
            net=net,
            rescale=dman_full.rescale,
            mkslices=mkslices,
            ngslices=ngslices,
            axes=allaxes[1],
            input_order=input_order,
            input_names=input_names,
        )

        seen = index in training_set
        seen = '* not used for training *' if not seen else '(in training set)'
        # add error to title
        fig.tight_layout()
        return fig

    for index in net_ids:
        try:
            fig = plot_prediction(index)
            name = net_name[index]
            fig.savefig(savepath / f'{index}_{name}_wlines.pdf', dpi=200)
            # plt.close(fig)
            # plt.close('all')
        except Exception as e:
            # add traceback
            import traceback

            print(f'Error while plotting {index}: {e}')
            traceback.print_exc()
    # plt.close('all')


##────────────────────────────────────────────────────────────────────────────}}}

nid = 216
# X = dman_full.get_X()[nid]
# Y = dman_full.get_Y()[nid]
# net = networks[nid]

# single_net_trace:

dman = dman_full.make_subset([nid])
stack = dman.build_compute_stack(compute_config)

N_SAMPLES_PER_CHUNK = 200
N_CHUNKS = 1
N_SAMPLES_TOTAL = N_SAMPLES_PER_CHUNK * N_CHUNKS
key = jax.random.PRNGKey(0)
X, Y = dman.get_uniform_samples(key, N_SAMPLES_TOTAL)
assert len(X) == len(Y)
X = [np.expand_dims(arr, axis=1) if arr.ndim == 1 else arr for arr in X]
Y = [np.expand_dims(arr, axis=1) if arr.ndim == 1 else arr for arr in Y]
ALLX = np.concatenate(X, axis=1)
assert ALLX.shape == (
    N_SAMPLES_TOTAL,
    stack.total_nb_of_inputs,
), f"{ALLX.shape} != {(N_SAMPLES_TOTAL, stack.total_nb_of_inputs)}"


@jit
def compute(params, XX, Q, keys):
    allres = stack.apply_with_trace(params, XX, Q, keys)
    return allres


ALLX_CHUNKS = np.split(ALLX, N_CHUNKS, axis=0)

YHAT = []

for chunk_id, XX in enumerate(tqdm(ALLX_CHUNKS, desc='plot_pred chunks')):
    Q = jax.random.uniform(key, (N_SAMPLES_PER_CHUNK, stack.total_nb_of_outputs))
    keys = jax.random.split(key, N_SAMPLES_PER_CHUNK)
    key = keys[-1]
    yhat_chunk = vmap(compute, in_axes=(None, 0, 0, 0))(params, XX, Q, keys)
    YHAT.append(np.array(yhat_chunk))

YHAT = np.concatenate(YHAT, axis=0)

out_id = stack.get_network_global_output_id(0)
n_out = networks[0].get_nb_outputs()
x, y = X[0], Y[0]
# yhat = YHAT[: x.shape[0], out_id : out_id + n_out]

# stack.node_map is {(net_id, compute_node_id): (layer_id, node_position)}

# net.compute_graph # pandas dataframe
# for each row  in compute graph:
# net_vnodes = {nid: stack.layers[lid].nodes[npos]

# net_vnodes = {}
# for nid, row in net.compute_graph.iterrows():
# lid, npos = stack.node_map[(0, nid)]
# net_vnodes[nid] = stack.layers[lid].nodes[npos]

# using stack.get_node_output_start_index(node: VirtualNode, output_slot: int) -> int:

# output_pos = {} # {nid: (start, end)}
# for nid, vnode in net_vnodes.items():
# cnode = net.compute_graph.loc[nid]
# ntype = cnode['type']
# n_inputs = len(cnode['input_from'])
# n_outputs = len(cnode['output_to'])

# data to generate:
# - a list of list of vnodes. One list per layer, storing the type and the number of outputs

##

net = dman.get_networks()[0]

vnode_data = []
for lid, layer in enumerate(stack.layers):
    obj = {
        'name': layer.f_type,
        'input_shapes': layer.f_input_shapes,
        'output_shapes': layer.f_out_shapes,
        'layer_id': lid,
    }
    layer_data = []
    for p, n in enumerate(layer.nodes):
        nid = n.node_id
        # using stack.get_node_output_start_index(node: VirtualNode, output_slot: int) -> int:
        output_start_indices = [
            stack.get_node_output_start_index(n, slot) for slot, _ in enumerate(layer.f_out_shapes)
        ]
        output_length = [np.prod(shape) for shape in layer.f_out_shapes]
        cnode = n.get_compute_node()
        out_to = [stack.node_map[(0, oid)] for oid,_ in cnode['output_to']]
        nobj = {
            'node_id': nid,
            'column': p,
            'output_start': output_start_indices,
            'output_length': output_length,
            'output_to': out_to,
            **obj,
        }
        layer_data.append(nobj)
    vnode_data.append(layer_data)

@partial(vmap, in_axes=(0, None), out_axes=0)
@partial(jit, static_argnums=(1,))
def trace_points(output_row, vnode_data):
    trace = []
    for layer in vnode_data:
        ltrace = []
        for v in layer:
            outputs = [
                output_row[start : start + length].reshape(shape)
                for start, length, shape in zip(
                    v["output_start"], v["output_length"], v["output_shapes"]
                )
            ]
            ltrace.append(outputs)
        trace.append(ltrace)

    return trace

frozen_vnode_data = ut.freeze(vnode_data)
traces = trace_points(YHAT, frozen_vnode_data)

import json
jt = json.dumps(su.make_json_compatible(traces, float_precision=4), indent=2)
jd = json.dumps(su.make_json_compatible(frozen_vnode_data), indent=2)
# write to disk
with open('layout.json', 'w') as f:
    f.write(jd)
with open('data.json', 'w') as f:
    f.write(jt)

