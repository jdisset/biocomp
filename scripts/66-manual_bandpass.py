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

ERNs = ['CasE', 'Csy4', 'PgU']
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
    # colors
    'x0color': TU(x0color_part),
    'x1color': TU(x1color_part),
    'biascolor': TU(biascolor_part),
    # output node
    'C_pos': TU(rec[2], outcolor_part),
    'C_neg': TU(ern[2]),
}

# # simple:
# aggregations_bp = [
# ['A_pos_0', 'B_neg_0', 'x0color'],  # x0
# ['A_neg_0', 'B_pos_0', 'x1color'],  # x1
# ['C_pos', 'C_neg', 'A_neg_1', 'B_neg_1', 'biascolor'],  # biases
# ]


# everything everywhere all at once:
aggregations_bp = [
    ['A_pos_0', 'A_neg_0', 'B_pos_0', 'B_neg_0', 'x0color'],  # x0
    ['A_pos_1', 'A_neg_1', 'B_pos_1', 'B_neg_1', 'x1color'],  # x1
    ['A_pos_2', 'A_neg_2', 'B_pos_2', 'B_neg_2', 'C_pos', 'C_neg', 'biascolor'],  # biases
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

# dirname = Path('~/Desktop/bandpass_attempt/v0/networks/').expanduser()
# dirname.mkdir(parents=True, exist_ok=True)
# su.plot_networks(networks, filenames=[f'{dirname}/network_{i}.pdf' for i in range(len(networks))])
# su.plot_networks(networks, W=4500, H=4000, show=True, figsize=(22, 20))

NETWORK = networks[0]


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                      --     load parameters     --
training_archive = du.load('../__results/training_archives/20230923_fulltrain_v0.pkl')
shared_parameters = training_archive['parameters']
compute_config = training_archive['compute_config']
training_config = training_archive['training_config']
compute_config.set_impl('bias', bc.nodes.bias)

def get_output_indices(stack, output_protein_name):
    out_indices = []
    for n_id, n in enumerate(stack.networks):
        output_id = n.get_output_proteins().index(output_protein_name)
        out_indices.append(stack.get_network_global_output_id(n_id, output_id))
    return jnp.array(out_indices)

def init_stack(rng, stack, shared_parameters):
    local_params, _ = stack.init(rng).filter_by_tag('local')
    local_params.data.check()
    full_params = ParameterTree.merge(local_params, shared_parameters)
    return full_params

bias_protein_names = ['NeonGreen']
output_protein_name = 'iRFP720'
key = random.PRNGKey(0)

# generate the compute stack
stack = cmp.ComputeStack([NETWORK])
stack.build(compute_config)
output_indices = get_output_indices(stack, output_protein_name)
full_params = init_stack(key, stack, shared_parameters)
full_params.data.check()
static_params, dynamic_params = full_params.filter_by_tag(['shared', 'non_grad'], mode='any')

##────────────────────────────────────────────────────────────────────────────}}}

vmapped_compute = jax.vmap(stack.apply, in_axes=(None, 0, 0, 0))
@jit
def evaluate_at(params, X, Z, key):
    keys = jax.random.split(key, X.shape[0])
    full_yhat, _ = vmapped_compute(params, X, Z, keys)
    yhat = full_yhat[:, output_indices]
    if yhat.ndim == 1:
        yhat = yhat.reshape(-1, 1)
    return yhat

def plot_eval(params, res=100, xlims=(0, 1)):
    X = np.meshgrid(np.linspace(*xlims, res), np.linspace(*xlims, res))
    X = np.stack(X, axis=-1).reshape(-1, 2)
    Z = jnp.ones((res * res, stack.total_nb_of_outputs)) * 0.5
    Y = jax.jit(evaluate_at)(params, X, Z, key)
    fig, ax = du.mkfig(1, 1)
    im = ax.imshow(Y.reshape(res, res), extent=[*xlims, *xlims], cmap='YlGnBu', origin='lower')
    ax.contour(
        Y.reshape(res, res), 2, extent=[*xlims, *xlims], colors='k', origin='lower', alpha=0.25
    )
    # colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xlabel('$X_1$')
    ax.set_ylabel('$X_2$')
    return fig, ax

# plot_eval(full_params)

##
import streamlit as st


def make_slider_group(name, param):
    sliders = np.array([])
    if isinstance(param, (np.ndarray, jnp.ndarray)) and 'tc' not in name:
        for i, v in enumerate(param.ravel()):
            sliders = np.append(
                sliders,
                st.slider(
                    f'{name}_{i}',
                    min_value=0.0,
                    max_value=1.0,
                    value=float(v),
                    step=0.01,
                )

            )
        sliders = sliders.reshape(param.shape)
        return sliders
    return param


sliderparams = pm.ParameterTree()
with st.sidebar:
    for k, v in dynamic_params.data.iter_leaves():
        sliderparams[k] = make_slider_group(k, v)
        sliderparams.tags[k] = dynamic_params.tags[k]

# st.text(sliderparams.data)

params = ParameterTree.merge(sliderparams, static_params)
f, a = plot_eval(params, res=40)
st.pyplot(f)
