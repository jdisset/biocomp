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

project_name = 'cascades_v2'
runs, losses = du.retrieve_wandb_results(project_name)

##
##────────────────────────────────────────────────────────────────────────────}}}

### {{{                         --     loss plot     --

fig, ax = du.mkfig(1, 1, (7, 5))
with ut.timer('Loss plot'):
    du.losses_plot(losses, ax, runs=runs)
fig.savefig('/Users/jeandisset/Desktop/bestloss_v2.pdf')
run = runs[du.get_best_run_id(losses)]
print('Best run:', run.name)

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
    loadedxp = {
        xpname: su.load_xp(XP[xpname], lib, data_path='./data/calibrated_data_v2')
        for xpname in xpnames
    }
    dman_full = du.DataManager.from_xps(loadedxp.values(), training_config, inverse='all')

all_networks = dman_full.get_networks()
net_xp = [n.metadata['from_xp'] for n in all_networks]
net_name = [n.name for n in all_networks]
print('done')

##
dirname = Path('~/Desktop/predictions/lvl2_cascades_newcalibry_v3/networks').expanduser()
dirname.mkdir(parents=True, exist_ok=True)

fnames = [f'{dirname}/{i}_{n}.pdf' for i, n in enumerate(net_name)]

su.plot_networks(all_networks, filenames=fnames)

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
print('done')


##────────────────────────────────────────────────────────────────────────────}}}

import joblib

with ut.timer('Building compute stack'):
    key = jax.random.PRNGKey(0)
    full_stack = dman_full.build_compute_stack(compute_config)
    substack, _ = full_stack.make_subset([50])

##

try:
    best_params = joblib.load(f'../__cache/best_params.pkl')
    print('Loaded best params from cache')
except FileNotFoundError:

    # with ut.timer('Stack initialization'):
        # base_params = full_stack.init(key)

    tmp_dir = Path(f'./{project_name}')
    param_file = run.file('latest_params.pkl').download(replace=True, root=tmp_dir)

    with open(param_file.name, 'rb') as f:
        trained_params = pickle.load(f)

    best_params = full_stack.use_shared_params(base_params, trained_params)
    joblib.dump(best_params, f'../__cache/best_params.pkl')
print('done')


### {{{                    --     training data plots     --

savepath = Path(f'~/Desktop/predictions/lvl2_cascades_newcalibry_v3').expanduser()
savepath.mkdir(parents=True, exist_ok=True)

# bash command to remove all files with "==" in the name:
# find . -name "*==*" -type f -delete
import matplotlib
import matplotlib as plt

# matplotlib.pyplot.switch_backend('Agg')
# dman_full._densities = None

networks = dman_full.get_networks()
stack = full_stack

# net_ids = list(range(len(networks)))[214:220]
net_ids = list(range(len(networks)))

dman = dman_full.make_subset(net_ids)
stack = dman.build_compute_stack(compute_config)

params = stack.init(key)
params = full_stack.use_shared_params(params, best_params)
networks = dman.get_networks()

print('done')
##


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

        net.get_output_proteins()


with ut.timer('pred plot'):
    N_SAMPLES_PER_CHUNK = 5000
    N_CHUNKS = 3

    N_SAMPLES_TOTAL = N_SAMPLES_PER_CHUNK * N_CHUNKS

    key = jax.random.PRNGKey(0)
    X, Y = dman.get_uniform_samples(key, N_SAMPLES_TOTAL)
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

        ninputs = networks[index].get_nb_inputs()
        input_order = [0, 1, 2, 3][:ninputs]

        fig = du.report(
            params,
            dman,
            index,
            use_x_y_yhat=(x, y, yhat),
            res=128,
            input_order=input_order,
            use_y_as_x=True,
        )
        fig.suptitle(f'{fig._suptitle.get_text()} using y as x\nerror: {error:.3f}')

        # ngslices = [0.25, 0.5, 0.75]
        # mkslices = [0.05, 0.3, 0.5]

        # fig, allaxes = du.mkfig(2, 3)
        # input_order = ([0, 1, 2],)
        # net = networks[index]
        # input_names = net.get_inverted_input_proteins()

        # smooth_line_plots_slices(
        # x,
        # y,
        # net=net,
        # rescale=dman.rescale,
        # mkslices=mkslices,
        # ngslices=ngslices,
        # axes=allaxes[0],
        # input_order=input_order,
        # input_names=input_names,
        # )

        # smooth_line_plots_slices(
        # x,
        # yhat,
        # net=net,
        # rescale=dman.rescale,
        # mkslices=mkslices,
        # ngslices=ngslices,
        # axes=allaxes[1],
        # input_order=input_order,
        # input_names=input_names,
        # )

        seen = index in training_set
        seen = '* not used for training *' if not seen else '(in training set)'
        # add error to title
        fig.tight_layout()
        return fig

    # # for index in [103, 104, 105, 106]:
    # for index in range(50, 60):
    for index in tqdm(range(len(networks)), desc='plot_pred'):
        try:
            fig = plot_prediction(index)
            name = net_name[index]
            fig.savefig(savepath / f'{index}_{name}_y_as_x.pdf', dpi=200)
            print(f'saved {index} {name}')
            plt.close('all')
        except Exception as e:
            # add traceback
            import traceback

            print(f'Error while plotting {index}: {e}')
            traceback.print_exc()
    # plt.close('all')
    print(savepath)

print('done')


##────────────────────────────────────────────────────────────────────────────}}}


# single_net_trace:

dman = dman_full.make_subset([nid])
stack = dman.build_compute_stack(compute_config)

params = stack.init(key)
params = full_stack.use_shared_params(params, best_params)

N_SAMPLES_PER_CHUNK = 5000
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


def get_node_info(network, node, outslot):
    info = ''
    nid = node.name

    if node.type == 'output':
        cdgin = network.central_dogma_graph.iloc[node.cdg_input[outslot]]
        content = cdgin['content']
        info = f'{content[0]}'

    elif node.type == 'input':
        input_from_output = node.extra['input_from_output']
        if input_from_output is not None:
            output_names = network.get_output_proteins()
            info = f'{output_names[input_from_output]}'

    elif node.type.startswith('sequestron_ERN'):
        cdgin = network.central_dogma_graph.iloc[node.cdg_input[0]]
        content = cdgin['content']
        info = f'{content[0]}'

    elif node.type.startswith('aggregation'):
        # extra has 'ratio'
        info = f'ratio: {100.0*node.extra["ratios"][outslot]:.1f} %'

    elif node.type == 'transcription' or node.type == 'translation':
        if isinstance(node.cdg_output, int):
            cdg = network.central_dogma_graph.iloc[node.cdg_output]
            if cdg['content'] is not None:
                content = ', '.join([c for c in cdg['content']])
                info = f'{content}'

    if node.source_id is not None:
        n = node.source_id.split('_')[:-1]
        n = '_'.join(n)
        info += f'{n}'

    return info


net = dman.get_networks()[0]

net.compute_graph
net.central_dogma_graph

vnode_data = []
nid_to_uid = {}
uid = 0
for lid, layer in enumerate(stack.layers):
    obj = {
        'type': layer.f_type,
        'input_shapes': layer.f_input_shapes,
        'output_shapes': layer.f_out_shapes,
        'layer_id': lid,
    }
    layer_data = []
    actual_column = 0
    for col, n in enumerate(layer.nodes):
        # using stack.get_node_output_start_index(node: VirtualNode, output_slot: int) -> int:
        output_start_indices = [
            stack.get_node_output_start_index(n, slot) for slot, _ in enumerate(layer.f_out_shapes)
        ]
        output_length = [np.prod(shape) for shape in layer.f_out_shapes]
        cnode = n.get_compute_node()
        nid = cnode.name
        # let's create a node for each output
        # we need to keep track of the column of each node for later
        out_to = cnode['output_to'] if len(cnode['output_to']) > 0 else [(None, None)]
        if cnode.type == 'output':
            out_to = [(None, None)] * len(cnode['input_from'])
        for slotid, (target_nid, target_slot) in enumerate(out_to):
            nobj = {
                'node_id': nid,
                'original_column': col,
                'column': actual_column,
                'slot': slotid,
                'target_nid': target_nid,
                'target_slot': target_slot,
                'output_start': output_start_indices,
                'output_length': output_length,
                'n_inputs': len(layer.f_input_shapes),
                'n_outputs': len(layer.f_out_shapes),
                'info': get_node_info(net, cnode, slotid),
                'uid': uid,
                **obj,
            }
            nid_to_uid.setdefault(nid, []).append(uid)
            vnode_data.append(nobj)
            actual_column += 1
            uid += 1

for n in vnode_data:
    if n['target_nid'] is not None:
        this_cnode = net.compute_graph.loc[n['node_id']]
        n['output_to'] = nid_to_uid[n['target_nid']]
        target_cnode = net.compute_graph.loc[n['target_nid']]
        if target_cnode['type'] == 'output':
            n['output_to'] = [nid_to_uid[n['target_nid']][n['target_slot']]]
    else:
        n['output_to'] = []


len(vnode_data)


@partial(vmap, in_axes=(0, None), out_axes=0)
@partial(jit, static_argnums=(1,))
def trace_points(output_row, vnode_data):
    trace = []
    current_nid = -1
    for vnode in vnode_data:
        if vnode['node_id'] != current_nid:
            outputs = [
                output_row[start : start + length].reshape(shape)
                for start, length, shape in zip(
                    vnode["output_start"], vnode["output_length"], vnode["output_shapes"]
                )
            ]
            trace.extend(outputs)
            current_nid = vnode['node_id']
    return trace


frozen_vnode_data = ut.freeze(vnode_data)
traces = trace_points(YHAT, frozen_vnode_data)

frozen_vnode_data

[n['uid'] for n in vnode_data]

assert len(vnode_data) == len(traces)

# remove output_start, outout_length, output_shapes
for v in vnode_data:
    del v["output_start"]
    del v["output_length"]
    del v["output_shapes"]

import msgpack

mt = msgpack.packb(su.make_json_compatible(traces))

layoutinfo = {'network_name': net.name, 'layout': vnode_data}


md = msgpack.packb(su.make_json_compatible(layoutinfo))

with open('../biocomp-ui/frontend/tracer/layoutData.bin', 'wb') as f:
    f.write(md)

with open('../biocomp-ui/frontend/tracer/pointData.bin', 'wb') as f:
    f.write(mt)

print('done')
##

# from jinja2 import Environment, FileSystemLoader

# template_folder_path = Path('../biocomp-ui/frontend/tracer/templates')
# env = Environment(loader=FileSystemLoader(template_folder_path))
# template = env.get_template('data_template.js')

# rendered = template.render(pointData=jt, layoutData=jd)

# with open('../biocomp-ui/frontend/tracer/data.js', 'w') as f:
# f.write(rendered)

### {{{               --     experimenting with rescaling     --
netid = 0
n = dman_full.get_network(netid)
pnames = n.get_inverted_input_proteins()
pnames
rx = dman_full._raw_X[netid]
x = dman_full._X[netid]
above = (rx > 800).all(axis=1)
rx = rx[above]
x = x[above]

##
factor = dman_full.data_cfg['data_scaling_log_factor']
maxv = dman_full.data_cfg['data_scaling_max_value']
offset = 3e3
factor = 100
maxv = 5e7
current_tr = lambda x: (np.log10(1 + np.clip(x / factor, 0, None))) - np.log10(offset / factor)
# / np.log10(maxv / factor)

DEFAULT_LOG_RESCALE = 0.1
DEFAULT_LOG_OFFSET = 5000


def logoffset(x, scale=DEFAULT_LOG_RESCALE, offset=DEFAULT_LOG_OFFSET):
    return jnp.log(jnp.clip(x + offset, 1, None)) - jnp.log(offset)


def tr(x, offset=3e3, maxv=5e7, factor=50, threshold=300, compression=0.4):
    loff = ut.log_poly_log(offset / factor, threshold=threshold, compression=compression)
    lmv = ut.log_poly_log(maxv / factor, threshold=threshold, compression=compression)
    xp = ut.log_poly_log(1 + x / factor, threshold=threshold, compression=compression) - loff
    y = xp / (lmv - loff)
    return y

def inv_tr(y, offset=3e3, maxv=5e7, factor=50, threshold=300, compression=0.4):
    loff = ut.log_poly_log(offset / factor, threshold=threshold, compression=compression)
    lmv = ut.log_poly_log(maxv / factor, threshold=threshold, compression=compression)
    yp = y * (lmv - loff) + loff
    ypinv = ut.inverse_log_poly_log(yp , threshold=threshold, compression=compression)
    x = factor * (ypinv  - 1)
    return x


# du.fluo_scatter(new_tr(rx), pnames, xmin=0, xmax=1.5, logscale=False)
du.fluo_scatter(tr(rx), pnames, logscale=False, xmin=-0.3, xmax=1.5)
# du.fluo_scatter(rx, pnames, logscale=True)
# du.fluo_scatter(x2, pnames, logscale=True)
# du.fluo_scatter(logoffset(rx), pnames, logscale=False)
##


du.fluo_scatter(rx, pnames, xmin=0, xmax=1e7, logscale=True)
du.fluo_scatter(x, pnames, logscale=False)


# count nans in x:
np.isnan(x).sum(axis=0)


##────────────────────────────────────────────────────────────────────────────}}}
