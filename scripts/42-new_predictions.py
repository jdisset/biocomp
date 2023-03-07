### {{{                          --     imports     --
import biocomp as bc
from biocomp import datautils as du
from jax.scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from biocomp.calibration import Calibration
import scriptutils as ut
from pathlib import Path
import json5
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import gaussian_kde
import jax
import optax
from jax import jit, vmap, value_and_grad
from jax.tree_util import Partial as partial
from tqdm import tqdm
import biocomp.defaults as bdf
import pandas as pd
import copy

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                        --     node config     --
T_SIZE = 64
T_DEPTH = 4
I_SIZE = 64
I_DEPTH = 3
I_OUT = 8
ERN_SIZE = 128
ERN_DEPTH = 4
MEFL_SIZE = 64
MEFL_DEPTH = 4

node_impl = dict(
    bc.nodes.DEFAULT_COMPUTE_NODES_DICT,
    **{
        'output': partial(bc.nn.output, wsize=MEFL_SIZE, depth=MEFL_DEPTH),
        'transcription': partial(
            bc.nn.transcription,
            outer_wsize=T_SIZE,
            outer_depth=T_DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
            inner_out=I_OUT,
        ),
        'translation': partial(
            bc.nn.translation,
            outer_wsize=T_SIZE,
            outer_depth=T_DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
            inner_out=I_OUT,
        ),
        'inv_transcription': partial(
            bc.nn.inv_transcription,
            outer_wsize=T_SIZE,
            outer_depth=T_DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
            inner_out=I_OUT,
        ),
        'inv_translation': partial(
            bc.nn.inv_translation,
            outer_wsize=T_SIZE,
            outer_depth=T_DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
            inner_out=I_OUT,
        ),
        'sequestron_ERN': partial(bc.nn.ERN5p, wsize=ERN_SIZE, depth=ERN_DEPTH),
        'sequestron_ERN3p': partial(bc.nn.ERN3p, wsize=ERN_SIZE, depth=ERN_DEPTH),
    },
)

config = {
    **bdf.DEFAULT_CONFIG,
    **{
        'node_impl': node_impl,
        'rng_key': np.random.randint(0, 2**32),
        "batch_size": 16,
        "n_batches": 4,
        "epochs": 12,
    },
}

##────────────────────────────────────────────────────────────────────────────}}}

lib = ut.load_lib()
matrix_xp = ut.load_xp('2023-02-16_Matrix', lib, data_path='./data/calibrated_data')
dman_full = du.DataManager.from_xps([matrix_xp], config, inverse='all')
names = [m.node_namespace for m in dman_full.get_models()]
training_set = [0] + [i for i, n in enumerate(names) if 'inert' in n.lower()]

### {{{                      --     quantify uorfs     --
# isolate the CasE uorf number. it follows the pattern ... CasE{N[x or w]} ...
import re

case_uorf_numbers = []
caseR_uorf_numbers = []
for n in names:
    s = re.search(r'CasE\d+[xw]', n)
    s_r = re.search(r'CasER\d+[xw]', n)
    if s:
        # isolate the number, multiply by 10, subtract 5 if w
        num = int(s.group(0)[4:-1]) * 10
        if s.group(0)[-1] == 'w':
            num = num - 5
        case_uorf_numbers.append(num)
        if s_r:
            rnum = int(s_r.group(0)[5:-1]) * 10
            if s_r.group(0)[-1] == 'w':
                rnum = rnum - 5
        else:
            rnum = 0
        caseR_uorf_numbers.append(rnum)
    else:
        case_uorf_numbers.append(None)
        caseR_uorf_numbers.append(None)
case_uorf_numbers = np.array(case_uorf_numbers)
caseR_uorf_numbers = np.array(caseR_uorf_numbers)
# remove last 2 values
case_uorf_numbers = case_uorf_numbers[:-2]
caseR_uorf_numbers = caseR_uorf_numbers[:-2]
unique_vals = np.sort(np.unique(case_uorf_numbers[case_uorf_numbers != None]))

N = len(unique_vals)


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                        --     plot matrix     --
fig, axes = du.mkfig(N, N, (2, 2))
for m, (E, R) in tqdm(enumerate(list(zip(case_uorf_numbers, caseR_uorf_numbers))[:])):
    print(E, R, m)
    if E is None or R is None:
        continue
    E = int(E)
    R = int(R)
    i = np.where(unique_vals == E)[0][0]
    j = np.where(unique_vals == R)[0][0]
    ax = axes[j, i]
    title = ''
    contours = np.linspace(0, 1, 7)
    du.model_plot(
        dman_full,
        m,
        ax=ax,
        radius=0.125,
        knn=300,
        min_points=20,
        colorbar=False,
        title=title,
        contours=contours,
        res=100,
    )
    # remove left ticks except if j == 0
    if j != N - 1:
        ax.set_xticks([])
        ax.set_xlabel('')
    if i != 0:
        ax.set_yticks([])
        ax.set_ylabel('')

    # add E and R labels for the whole grid
    if j == 0:  # first row
        ax.text(0.5, 1.1, f'E: {E/10:.1f}x', transform=ax.transAxes, ha='center', va='bottom')
    if i == N - 1:  # last column, write to the right
        ax.text(
            1.1, 0.5, f'R: {R/10:.1f}x', transform=ax.transAxes, ha='left', va='center', rotation=90
        )

fig.tight_layout()
savepath = Path('~/Desktop/matrixdata').expanduser()
savepath.mkdir(parents=True, exist_ok=True)
fig.savefig(savepath / 'matrix_data_smooth_default_v3.pdf')
print('done')
##────────────────────────────────────────────────────────────────────────────}}})

### {{{                        --     load params     --
# get wandb run
import wandb as wb
import pickle

project_name = 'matrix_train_v1'
# run_code='fxhnbjvb' # 1 corner
run_code = 'x32fpgne'  # 4 corners
# run_code='rd1y0rgp' # 3 corners
run = wb.Api().run(f'{project_name}/{run_code}')

# load latest params (latest_params.pkl)
param_file = run.file('latest_params.pkl').download(replace=True)
with open(param_file.name, 'rb') as f:
    trained_params = pickle.load(f)

# initialize node params for each model
models = dman_full.get_models()
key = jax.random.PRNGKey(0)
params = {}
for m, k in tqdm(zip(models, jax.random.split(key, len(models)))):
    params, _ = m.init(k, pre_params=params)

# use trained params to initialize the shared weights
params['shared'] = trained_params['shared']


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                        --     plot prediction matrix     --

du.plot_model_at_x(params, dman_full, 0, ax)

x, y, yhat = du.model_at_x(params, dman_full, 0)
y
np.abs(y - yhat).mean(axis=0)
m = dman_full.get_models()[0]
m.get_output_proteins()

fig, ax = du.mkfig(1, 1)
du.plot_model_diff(params, dman_full, 0, ax)

du.report(params, dman_full, 0)

##

fig, axes = du.mkfig(N, N, (2, 2))
for m, (E, R) in tqdm(enumerate(list(zip(case_uorf_numbers, caseR_uorf_numbers))[:])):
    print(E, R, m)
    if E is None or R is None:
        continue
    E = int(E)
    R = int(R)
    i = np.where(unique_vals == E)[0][0]
    j = np.where(unique_vals == R)[0][0]
    ax = axes[j, i]
    title = ''
    contours = np.linspace(0, 1, 7)
    du.plot_model_at_x(
        params,
        dman_full,
        m,
        ax,
        colorbar=False,
        title=title,
        contours=contours,
        radius=0.125,
        knn=300,
        min_points=20,
        res=100,
    )
    if j != N - 1:
        ax.set_xticks([])
        ax.set_xlabel('')
    if i != 0:
        ax.set_yticks([])
        ax.set_ylabel('')

    # add E and R labels for the whole grid
    if j == 0:  # first row
        ax.text(0.5, 1.1, f'E: {E/10:.1f}x', transform=ax.transAxes, ha='center', va='bottom')
    if i == N - 1:  # last column, write to the right
        ax.text(
            1.1, 0.5, f'R: {R/10:.1f}x', transform=ax.transAxes, ha='left', va='center', rotation=90
        )

fig.tight_layout()
fig.savefig(savepath / 'matrix_pred_smooth_4corners_v2.pdf')
print('done')
##────────────────────────────────────────────────────────────────────────────}}})

models = dman_full.get_models()

i = 50
models[i].node_namespace
models[i].network.central_dogma_graph

k = list(params['shared'].keys())
tl_rate_keys = [key for key in k if '::tl_rate' in key]
tl_rate_keys
[params['shared'][key] for key in tl_rate_keys]

import copy

params_copy = copy.deepcopy(params)
params_copy['shared']['3x_uORF::tl_rate'] = jnp.array([0.0])


dman_full.set_subset(training_set)

fig, ax = du.mkfig(1, 1, (4, 4))
du.report(params_copy, dman_full, 50)

dman_full.set_subset(np.arange(len(names)))
ut.plot_networks([m.network], figsize=(30, 30), H=3000, W=1500)

m = dman_full.get_models()[50]
ut.plot_node('translation', params, m, xlim=(0, 1.2), ylim=(0, 1.2))
ut.plot_node('transcription', params, m, xlim=(0, 1.2), ylim=(0, 1.2))
# ut.plot_node('inv_transcription', params, m, xlim=(0, 1.2), ylim=(0, 1.2))
# ut.plot_node('inv_translation', params, m, xlim=(0, 1.2), ylim=(0, 1.2))
# ut.plot_node('inv_transcription', params, m, xlim=(0, 1.2), ylim=(0, 1.2))

extra = m.network.compute_graph[m.network.compute_graph.type == 'sequestron_ERN'].extra.to_list()
ut.plot_node('sequestron_ERN', params, m, xlim=(0, 1.2), n_inputs=2, extra_args=extra[0], mode='3d')
ut.plot_node('output', params, m, xlim=(0, 1.2), ylim=(0, 1.2))

m = dman_full.get_models()[54]
ut.plot_networks([m.network], figsize=(30, 30), H=3000, W=1500)

# for model 50
# translation node 8
# from cdg input 5, that has tl_rate = 3x_uORF
m.network.compute_graph
m.network.central_dogma_graph

testp, _ = m.init(key)
[k for k in list(testp['shared'].keys()) if 'tc_rate' in k]

##

# names
params_copy = copy.deepcopy(params)
# m 53 is 3x uorf
# params_copy['shared']['3x_uORF::tl_rate'] = jnp.array([1.0653249])
# params_copy['shared']['empty::tc_rate'] = 100000
m.collect_all_results(params_copy, jnp.zeros((2, 1)), jnp.ones((2,)) / 2, key)

y, g, r = m.collect_all_results(
    params_copy,
    jnp.zeros((2, 1)),
    jnp.ones((2,)) / 2,
    key,
    with_grad=['translation', 'transcription', 'output'],
)

from biocomp import utils as bu

bu.flat_concat(*g)

m = dman_full.get_models()[54]
testp, _ = m.init(key)
m.apply_and_negative_grad(
    params_copy,
    jnp.zeros((2, 1)),
    jnp.ones((2,)) / 2,
    key,
    override_w_uniform=['translation', 'transcription', 'output'],
)

du.report(params_copy, dman_full, 54)

##
# hypothesis: ERN outputs in a different domain than transcription, and
# the translation node has learnt 2 different functions for the 2 domains
# (and the ERN one doesn't know how to deal with uORFs)
model.network.compute_graph
ut.plot_networks([model.network], figsize=(30, 30), H=3000, W=1500)

p = copy.deepcopy(params)
npoints_eval = 20000
quantile_range = [0.25, 0.75]
key = jax.random.PRNGKey(0)
xrange_eval = None
model = dman_full.get_models()[50]
k_i, k_q = jax.random.split(key)
if xrange_eval is None:
    xrange_eval = jnp.array([[0, 0], [1, 1]])
x = jax.random.uniform(
    k_i, (npoints_eval, model.n_inputs), minval=xrange_eval[0], maxval=xrange_eval[1]
)
quantiles = jax.random.uniform(
    k_q, (npoints_eval, model.n_outputs), minval=quantile_range[0], maxval=quantile_range[1]
)
keys = jax.random.split(key, npoints_eval)
y, allres = jit(vmap(model.collect_all_results, in_axes=(None, 0, 0, 0)))(
    params, x, quantiles, keys
)

allres[1].shape
allres[7].shape
allres[8].shape
res = jnp.vstack(
    [
        allres[18],
        allres[12],
        allres[1],
        allres[7],
        allres[8],
        allres[0][:, 0],
        allres[0][:, 1],
        allres[0][:, 2],
    ]
).T
du.fluo_scatter(
    res,
    [
        'inverse out',
        'tx out',
        'ern out',
        'post-tx TL',
        'post-ERN TL',
        'eBFP out',
        'eYFP out',
        'mKate out',
    ],
    logscale=False,
)

res = jnp.vstack([allres[18], allres[20]]).T
du.fluo_scatter(res, ['inverse-out'], logscale=False)
model.get_output_proteins()


jnp.asarray(6).shape
jax.random.uniform(key, shape=())
