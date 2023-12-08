### {{{                          --     imports     --
from biocomp import utils as ut
import scriptutils as su
import biocomp.datautils as du
import biocomp.plotutils as pu
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
xpname = '2023-10-31_ConstraintsV2_4'
lib = su.load_lib()

xp = su.load_xp(
    xpname,
    lib,
    data_path='./data/calibrated_data_v3',
    recipe_path='./recipes',
)
dman_full = du.DataManager.from_xps([xp], training_config, inverse='all')

##

from matplotlib import pyplot as plt

savedir = Path('~/Desktop/ebfp2weird/').expanduser()
savedir.mkdir(exist_ok=True, parents=True)

### {{{                           --     cmaps     --
import matplotlib.colors as mcolors

cmap = 'Blues'
cmap = 'GnBu'
cmap = 'BuPu'

blues = [
    '#F9F7F5',
    '#EEECEA',
    '#B0CCD6',
    '#6CAFC3',
    '#2974A4',
    '#3B4B90',
    '#3D1277',
    '#22044B',
]

greens = [
    '#F9F7F5',
    '#E2EADA',
    '#CBE4BB',
    '#9DDDAA',
    '#4CCDAB',
    '#30A78F',
    '#1F7D73',
    '#0C5558',
]

reds = [
    '#F5F5F5',
    '#F1E6E5',
    '#F3CFBC',
    '#EF957D',
    '#D3494B',
    '#B00031',
    '#840137',
    '#560140',
]


cmaps = {
    'blues': mcolors.LinearSegmentedColormap.from_list('cm', blues, N=256),
    'greens': mcolors.LinearSegmentedColormap.from_list('cm', greens, N=256),
    'reds': mcolors.LinearSegmentedColormap.from_list('cm', reds, N=256),
}


##────────────────────────────────────────────────────────────────────────────}}}
##

dman_full.get_raw_X()[0].shape
dman_full.get_networks()[0].get_output_proteins()

for i in range(len(dman_full.get_networks()))[19:]:

    rawx = dman_full.get_raw_Y()[i]
    pnames = dman_full.get_networks()[i].get_output_proteins()
    # fig, _ = pu.fluo_scatter(rawx, pnames)

    net = dman_full.get_networks()[i]
    nname = dman_full.get_networks()[i].name

    fig.suptitle(f'Network {nname}')
    fig.set_dpi(300)

    nprots = len(pnames)
    input_order = [0, 1, 2]

    # fig, axes = pu.mkfig(3, 1, (3, 3), dpi=200)
    if nprots <= 3:
        fig, ax = pu.mkfig(1, 1, (3, 3), dpi=300)
        axes = None
    else:
        fig, axes = pu.mkfig(3, 1, (3, 3), dpi=200)
        ax = None

    if nprots == 2:
        continue
    pu.network_plot(
        dman_full,
        i,
        ax=ax,
        xmax=0.75,
        radius=0.15,
        vmax=0.6,
        cmap=cmaps['blues'],
        axes=axes,
        input_order=input_order[:nprots - 1],
        slices=[0.1, 0.3, 0.5]
    )
    # set font size of ax title
    ax.title.set_fontsize(8)
    # remove fig title
    fig.suptitle('')

    # pu.network_plot(dman_full, i, ax=ax, input_order=[0,1,2], method='smooth')

    fig.tight_layout()

    fig.savefig(savedir / f'network_plot_{i}_{nname}.png')
    plt.show()
    plt.close(fig)
    print(f'Saved network {i}')

##
nnames = [net.name for net in dman_full.get_networks()]
bpi = nnames.index('BPv2_T17')

orig_x = dman_full.get_X()[bpi]

bpnet = dman_full.get_networks()[bpi]

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
# stacks = [cmp.ComputeStack([net]) for net in dman_full.get_networks()]
# for stack in stacks:
    # stack.build(compute_config)

stack = cmp.ComputeStack([bpnet])
stack.build(compute_config)

##
# for network, stack in list(zip(dman_full.get_networks(), stacks))[:]:

# netname = network.name
# output_indices = get_output_indices(stack)
# remove latex fonts
plt.rcParams['text.usetex'] = False
params = init_stack(stack, rng)
xlims = (0, 0.8)

n_points = 60000
n_inputs = stack.total_nb_of_inputs

bpnet.get_inverted_input_proteins()
apply = jax.jit(jax.vmap(stack.apply, in_axes=(None, 0, 0, 0)))
# X = jax.random.uniform(rng, (n_points, n_inputs), minval=xlims[0], maxval=xlims[1])
X = orig_x
Z = jax.random.uniform(rng, (X.shape[0], stack.total_nb_of_outputs), minval=0.1, maxval=0.9)
keys = jax.random.split(key, X.shape[0])
Y, _ = apply(params, X, Z, keys)

##
fig, ax = du.mkfig(1, 1, (15, 15))
du.smooth(X, Y, bpnet, rescale=du.tr, ax=ax, res=150, min_radius=0.01, vmax=0.5, slices=[0.1, 0.3, 0.5])
ax1 = fig.axes[1]
# ax1.set_title(f'Predictions for xp {netname}', y=1.05)

plt.show()

##

netid = -1
nets = dman_full.get_networks()
net = nets[netid]
rawy = dman_full.get_raw_Y()[netid]
y = dman_full.get_Y()[netid]
pnames = net.get_output_proteins()
# scatter plot
# avg ratio of raw_y

fig, ax = pu.mkfig(1, 1, (5, 5), dpi=200)

ax.scatter(y[:, 0], y[:, 1], s=1, alpha=0.05)





