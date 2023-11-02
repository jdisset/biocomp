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

from biocomp import utils as ut
import scriptutils as su
import biocomp.datautils as du
from biocomp import train
from biocomp import compute as cmp

# matplotlib.use('agg')

##────────────────────────────────────────────────────────────────────────────}}}
### {{{                      --     load parameters     --
training_archive = du.load('../__results/training_archives/20230923_fulltrain_v0.pkl')
shared_parameters = training_archive['parameters']
compute_config = training_archive['compute_config']
training_config = training_archive['training_config']
compute_config.set_impl('bias', bc.nodes.bias)
##────────────────────────────────────────────────────────────────────────────}}}

XP = {'BPattempt': '2023-10-16_ConstraintsV2_1_i720'}
xpname = 'BPattempt'
with ut.timer(f'Loading data and building networks for {XP[xpname]}'):
    lib = su.load_lib()
    bp_xp = su.load_xp(
        XP[xpname],
        lib,
        data_path='./data/calibrated_data_v3',
        recipe_path=su.DEFAULT_DATA_PATH / 'Experiments' / XP[xpname] / 'recipes',
    )
    dman_full = du.DataManager.from_xps([bp_xp], training_config, inverse='all')

##

names = [n.name for n in dman_full.get_networks()]
names



##

# su.plot_networks([dman_full.get_networks()[0]], W=2000, H=4000)
from matplotlib import pyplot as plt

savedir = Path('~/Desktop/BP_ATTEMPT_10_03').expanduser()

savedir_data = savedir / 'data'
savedir_data.mkdir(exist_ok=True, parents=True)

args = {
    'input_order': [0, 1, 2],
    'slices': [0.2, 0.4, 0.65],
    'vmin': None,
    'vmax': None,
    'radius': 0.07,
    'min_points': 20,
}

for i in range(len(dman_full.get_networks()))[:]:
    fig, ax = du.mkfig(1, 1, (15, 15))
    du.network_plot(dman_full, i, ax=ax, **args)
    netname = dman_full.get_networks()[i].name
    ax1 = fig.axes[1]
    ax1.set_title(f'Real data for xp {netname}', y=1.05)
    # fig.savefig(savedir_data / f'network_{i}.png')
    fig.savefig(savedir_data / f'{netname}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f'Saved network {i}')

##

from biocomp.parameters import ParameterTree

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

key = jax.random.PRNGKey(0)
stacks = [cmp.ComputeStack([net]) for net in dman_full.get_networks()]
for stack in stacks:
    stack.build(compute_config)

##


for network, stack in list(zip(dman_full.get_networks(), stacks))[:]:
    netname = network.name
    output_indices = get_output_indices(stack)
    params = init_stack(stack, rng)
    xlims = (0, 0.8)

    n_points = 60000
    n_inputs = stack.total_nb_of_inputs

    network.get_inverted_input_proteins()
    apply = jax.jit(jax.vmap(stack.apply, in_axes=(None, 0, 0, 0)))
    X = jax.random.uniform(rng, (n_points, n_inputs), minval=xlims[0], maxval=xlims[1])
    Z = jax.random.uniform(rng, (X.shape[0], stack.total_nb_of_outputs), minval=0.3, maxval=0.7)
    keys = jax.random.split(key, X.shape[0])
    Y, _ = apply(params, X, Z, keys)

    fig, ax = du.mkfig(1, 1, (15, 15))
    du.smooth(X, Y, network, rescale=du.tr, ax=ax, res=200, **args)
    ax1 = fig.axes[1]
    ax1.set_title(f'Predictions for xp {netname}', y=1.05)

    plt.show()
    savedir_pred = savedir / 'predictions'
    savedir_pred.mkdir(exist_ok=True, parents=True)
    fig.savefig(savedir_pred / f'{netname}.png', dpi=300, bbox_inches='tight')




