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

savepath = Path(f'~/Desktop/predictions/lvl2_cascades_v1/nets').expanduser()
savepath.mkdir(parents=True, exist_ok=True)
plotid = range(len(all_networks))
su.plot_networks([all_networks[i] for i in plotid], [(savepath/f'net_{i}.pdf').as_posix() for i in plotid])

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


key = jax.random.PRNGKey(0)
full_stack = dman_full.build_compute_stack(compute_config)

with ut.timer('Stack initialization'):
    base_params = full_stack.init(key)

# 373s -> 359s ->
##

tmp_dir = Path(f'./{project_name}')
param_file = best_run.file('latest_params.pkl').download(replace=True, root=tmp_dir)
with open(param_file.name, 'rb') as f:
    trained_params = pickle.load(f)

best_params = full_stack.use_shared_params(base_params, trained_params)


##

# put back normal ipython matplotlib backend
import matplotlib as plt
# matplotlib.pyplot.switch_backend('Agg')

fig, ax = du.mkfig(1, 1)
du.network_plot(dman_full, 72, ax=ax)

fig.savefig('/Users/jeandisset/Desktop/test.png')
print('Network name:', dman_full.get_networks()[72].name)

### {{{                    --     training data plots     --

savepath = Path(f'~/Desktop/predictions/lvl2_cascades_v1').expanduser()
savepath.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.pyplot.switch_backend('Agg')

networks = dman_full.get_networks()
stack = full_stack
params = best_params

net_ids = list(range(len(networks)))[72:92]

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
        fig = du.report(params, dman_full, index, use_x_y_yhat=(x, y, yhat), res=128)
        seen = index in training_set
        seen = '* not used for training *' if not seen else '(in training set)'
        # add error to title
        fig.suptitle(f'{fig._suptitle.get_text()}\nerror: {error:.3f}\n{seen}')
        fig.tight_layout()
        return fig

    for index in net_ids:
        try:
            fig = plot_prediction(index)
            name = net_name[index]
            fig.savefig(savepath / f'{index}_{name}.pdf', dpi=200)
            plt.close(fig)
            plt.close('all')
        except Exception as e:
            # add traceback
            import traceback
            print(f'Error while plotting {index}: {e}')
            traceback.print_exc()
    plt.close('all')


##────────────────────────────────────────────────────────────────────────────}}}

# plot per layer
##

networks = dman_full.get_networks()
stack = full_stack
params = best_params.copy()

# fig, axes = du.mkfig(1, 4)
# du.network_plot(dman_full, 50, ax=None, axes=axes, input_order=[0, 1, 2], xbin=[0.0, 0.9])

X = dman_full.get_X()[nid]
Y = dman_full.get_Y()[nid]
net = networks[50]
fig, ax = du.mkfig(1, 1)

du.smooth_line_plots(
    X,
    Y,
    net,
    dman_full.rescale,
    ax=ax,
    slice=[0.45,0.5],
    radius=0.2,
    input_order=[0, 1, 2],
)



##
nid = 50
X = dman_full.get_X()[nid]
Y = dman_full.get_Y()[nid]
net = networks[50]
net.get_inverted_input_proteins()
net.get_output_proteins()

# neongreeen is 0
# irfpout is 1
# mkate is 2
# tagbfp is 3

# bins:

ngbin = [0.5, 0.6]
mkatebin = [0.3, 0.4]
binned_Y = Y[(Y[:, 0] > ngbin[0]) & (Y[:, 0] < ngbin[1]) & (Y[:, 2] > mkatebin[0]) & (Y[:, 2] < mkatebin[1])]

# scatter plot of irfpout( yaxis) vs tagbfp (xaxis)
fig, ax = du.mkfig(1,1)
ax.scatter(binned_Y[:, 3], binned_Y[:, 1], s=1, c='k')
# add regression from scipy
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(binned_Y[:, 3], binned_Y[:, 1])
ax.plot(binned_Y[:, 3], intercept + slope*binned_Y[:, 3], 'r', label='fitted line')
ax.legend()
ax.set_xlabel('tagbfp')
ax.set_ylabel('irfpout')




# TODO
# plot cascades as dosage response curves of the first ERN for several bins of ERN_2 and output_DNA
