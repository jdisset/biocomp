### {{{                          --     imports     --
from biocomp import utils as ut
import scriptutils as su
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
from matplotlib import pyplot as plt

##────────────────────────────────────────────────────────────────────────────}}}
### {{{                      --     load parameters     --
training_archive = du.load('../__results/training_archives/20230923_fulltrain_v0.pkl')
shared_parameters = training_archive['parameters']
compute_config = training_archive['compute_config']
training_config = training_archive['training_config']
compute_config.set_impl('bias', bc.nodes.bias)
##────────────────────────────────────────────────────────────────────────────}}}
### {{{                     --     generate networks     --

lib = su.load_lib()


def any_uorf(lib, *_, **__):
    all_uORFs = lib.pc[lib.pc.category == 'uORF_group'].index.tolist()
    return [all_uORFs]


def P(name):
    return bc.Slot(lib, name)


def TU(*parts):
    partlist = [P('hEF1a')] + list(parts)
    return bc.TranscriptionUnit(partlist)


uorfs = P(any_uorf(lib)[0][:8])
# 'Csy4+uOrfs': bc.TranscriptionUnit([promoter, P('Csy4'), P(any_uorf(lib)[0])]),

ERNs = ['CasE', 'PgU', 'Csy4']
ern = [P(ern) for ern in ERNs]
rec = [P(ern + '_rec') for ern in ERNs]
biascolor = 'NeonGreen'
x0color_part = P('mKate')
x1color_part = P('eBFP')
outcolor_part = P('iRFP720')
biascolor_part = P(biascolor)
tus_bp = {
    # node A
    'A_pos_0': TU(rec[0], uorfs, ern[2]),
    'A_pos_1': TU(rec[0], uorfs, ern[2]),
    'A_pos_2': TU(rec[0], uorfs, ern[2]),
    'A_neg_0': TU(ern[0]),
    'A_neg_1': TU(ern[0]),
    'A_neg_2': TU(ern[0]),
    # node B
    'B_pos_0': TU(rec[1], uorfs, ern[2]),
    'B_pos_1': TU(rec[1], uorfs, ern[2]),
    'B_pos_2': TU(rec[1], uorfs, ern[2]),
    'B_neg_0': TU(ern[1]),
    'B_neg_1': TU(ern[1]),
    'B_neg_2': TU(ern[1]),
    # output node
    'C_pos_0': TU(rec[2], outcolor_part),
    'C_pos_1': TU(rec[2], outcolor_part),
    'C_pos_2': TU(rec[2], outcolor_part),
    'C_neg_0': TU(ern[2]),
    'C_neg_1': TU(ern[2]),
    'C_neg_2': TU(ern[2]),
    # colors
    'x0color': TU(x0color_part),
    'x1color': TU(x1color_part),
    'biascolor': TU(biascolor_part),
}


# everything everywhere all at once:
aggregations_bp = [
    ['A_pos_0', 'A_neg_0', 'B_pos_0', 'B_neg_0', 'C_pos_0', 'C_neg_0', 'x0color'],  # x0
    ['A_pos_1', 'A_neg_1', 'B_pos_1', 'B_neg_1', 'C_pos_1', 'C_neg_1', 'x1color'],  # x1
    ['A_pos_2', 'A_neg_2', 'B_pos_2', 'B_neg_2', 'C_pos_2', 'C_neg_2', 'biascolor'],  # bias
]


sources_bp = {
    tu_name: [tu_name] for tu_name, tu in tus_bp.items() if tu_name in ut.flatten(aggregations_bp)
}
used_tus_bp = {
    tu_name: tu for tu_name, tu in tus_bp.items() if tu_name in ut.flatten(aggregations_bp)
}

n_bp = bc.Network.from_dict(lib, 'bp_attempt', used_tus_bp, sources_bp, aggregations_bp)
bp_net = bc.inverted_network(n_bp)[0]

bp_net.set_input_as_bias(biascolor)


networks = [bp_net]


NETWORK = networks[0]

# save stack pickle in ../__cache/
cachedir = Path('~/.biocompiler/cache/networks/').expanduser()
cachedir.mkdir(exist_ok=True, parents=True)
du.save(NETWORK, cachedir / 'full_bandpass_csy4.pkl')
# reload
NETWORK = du.load(cachedir / 'full_bandpass_csy4.pkl')


def save_net(net, name):
    cachedir = Path('~/.biocompiler/cache/networks/').expanduser()
    cachedir.mkdir(exist_ok=True, parents=True)
    du.save(net, cachedir / f'{name}.pkl')


##────────────────────────────────────────────────────────────────────────────}}}

su.plot_networks(networks, W=3500, H=3000, show=True, figsize=(35, 30))

NETWORK.central_dogma_graph
NETWORK.central_dogma_graph

nodes, edges = su.network_to_graph(networks[0])
type(nodes[0])

import json
import urllib.parse

pj = urllib.parse.quote_plus(json.dumps({'nodes': nodes, 'edges': edges}))


NETWORK.central_dogma_graph
NETWORK.compute_graph
##

rng = jax.random.PRNGKey(0)


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

full_params = init_stack(stack, rng)

full_params.filter_by_tag('local')[0].data

stack.layers[0].nodes

##

xppath = '2023-10-03_BPv2_C31BP'
x = su.load_xp(
    xppath, su.load_lib(), data_path=None, recipe_path=su.DEFAULT_XP_PATH / xppath / 'recipes'
)
nets, netnames = x.build_networks()
netnames

##
nets[0].get_dependent_output_proteins()
nets[0].get_inverted_input_proteins()

for i, network in enumerate(nets[2:3]):
    print(f'net {i}: {netnames[i]}')

    stack = cmp.ComputeStack([network])
    stack.build(compute_config)
    output_indices = get_output_indices(stack)
    full_params = init_stack(stack, rng)

    key = jax.random.PRNGKey(0)

    vmapped_compute = jax.vmap(stack.apply, in_axes=(None, 0, 0, 0))

    def evaluate_at(params, X, Z, key):
        keys = jax.random.split(key, X.shape[0])
        print(f'X.shape: {X.shape}')
        full_yhat, _ = vmapped_compute(params, X, Z, keys)
        print(f'full_yhat: {full_yhat}')
        yhat = full_yhat[:, output_indices]
        if yhat.ndim == 1:
            yhat = yhat.reshape(-1, 1)
        return yhat

    def plot_eval(params, res=100, xlims=(0, 0.7)):
        X = np.meshgrid(np.linspace(*xlims, res), np.linspace(*xlims, res))
        X = np.stack(X, axis=-1).reshape(-1, 2)
        Z = jnp.ones((res * res, stack.total_nb_of_outputs)) * 0.5
        Y = jax.jit(evaluate_at)(params, X, Z, key)
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

    plot_eval(full_params, res=100, xlims=(0, 0.7))

##

key = jax.random.PRNGKey(0)
xppath = 'fake_tests'
x = su.load_xp(
    xppath, su.load_lib(), data_path=None, recipe_path=su.DEFAULT_XP_PATH / xppath / 'recipes'
)
nets, netnames = x.build_networks()
stacks = [cmp.ComputeStack([net]) for net in nets]
for stack in stacks:
    stack.build(compute_config)

# su.plot_networks([nets[11]])
jax.clear_caches()


for netname, network, stack in list(zip(netnames, nets, stacks))[:]:
    output_indices = get_output_indices(stack)
    params = init_stack(stack, rng)
    xlims = (0, 0.8)

    n_points = 20000
    n_inputs = stack.total_nb_of_inputs

    network.get_inverted_input_proteins()
    apply = jax.jit(jax.vmap(stack.apply, in_axes=(None, 0, 0, 0)))
    X = jax.random.uniform(rng, (n_points, n_inputs), minval=xlims[0], maxval=xlims[1])
    Z = jax.random.uniform(rng, (X.shape[0], stack.total_nb_of_outputs), minval=0.25, maxval=0.75)
    keys = jax.random.split(key, X.shape[0])
    Y, _ = apply(params, X, Z, keys)

    fig, ax = du.mkfig(1, 1, (15, 15))
    # du.smooth(X, Y, network, rescale=du.tr, ax=ax, res=200, input_order=[0,1,2], slices=[0.2, 0.4, 0.65], vmin=0.2, vmax=0.6)
    du.smooth(X, Y, network, rescale=du.tr, ax=ax, res=200, input_order=[0,1,2], slices=[0.2, 0.4, 0.65], vmin=None, vmax=None)
    # du smooth creates 2 more axes
    ax1 = fig.axes[1]
    ax1.set_title(f'Predictions for xp {netname}', y=1.05)

    plt.show()
    savedir = Path('~/Desktop/bppredictions/10_03/relative_scalev2').expanduser()
    savedir.mkdir(exist_ok=True, parents=True)
    fig.savefig(savedir / f'{netname}.png', dpi=300, bbox_inches='tight')




##
XP = {'BPattempt': '2023-10-03_BPv2_C31BP'}
xpname = 'BPattempt'
with ut.timer(f'Loading data and building networks for {XP[xpname]}'):
    lib = su.load_lib()
    bp_xp = su.load_xp(XP[xpname], lib, data_path='./data/calibrated_data_v3', recipe_path=su.DEFAULT_DATA_PATH / 'Experiments' / XP[xpname] / 'recipes')
    dman_full = du.DataManager.from_xps([bp_xp], training_config, inverse='all')

##

# su.plot_networks([dman_full.get_networks()[0]], W=2000, H=4000)
n = dman_full.get_networks()[0]
n.get_inverted_input_proteins()


from matplotlib import pyplot as plt

savedir = Path('~/Desktop/bppredictions/10_03/relative_scalev2/realdata/').expanduser()
savedir.mkdir(exist_ok=True, parents=True)
for i in range(len(dman_full.get_networks()))[:]:
    fig, ax = du.mkfig(1,1, (14,14), dpi=200)
    du.network_plot(dman_full, i, ax=ax, input_order=[0,1,2])
    fig.savefig(savedir / f'network_{i}.pdf')
    plt.show()
    plt.close(fig)
    print(f'Saved network {i}')





# su.plot_networks([network])

##
def save_net(net, name):
    cachedir = Path('~/.biocompiler/cache/networks/').expanduser()
    cachedir.mkdir(exist_ok=True, parents=True)
    du.save(net, cachedir / f'{name}.pkl')


##


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


param_map = make_param_map(stack, tunable_params)

param_map

import json

strmap = json.dumps(param_map)
param_map = json.loads(strmap)


def apply_param_map(param_map, params):
    new_params = deepcopy(tunable_params)
    for cid, path_loc_val in param_map.items():
        for path, loc, name, val in path_loc_val:
            old_val = np.array(params[path])
            old_val[loc] = val
            new_params[path] = old_val
            new_params.tags[path] = params.tags[path]
    return new_params


new_params = apply_param_map(param_map, tunable_params)
##



x = np.linspace(0, 1, 100)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(x, x, label='x')
ax.plot(x, 2*x, label='x')
ax.plot(x, 3*x, label='x')
ax.plot(x, 4*x, label='x')

# log scale
ax.set_yscale('log')
ax.set_xscale('log')



