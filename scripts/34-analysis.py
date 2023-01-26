### {{{                          --     imports     --
import datetime
import biocomp as bc
import matplotlib.pyplot as plt
import numpy as np
import time
from functools import partial
import biocomp.utils as bu
import scriptutils as ut
import jax
from jax import jit, vmap, grad, value_and_grad
import jax.numpy as jnp
import biocomp.datautils as du
import optax
from pathlib import Path
from tqdm import tqdm
import biocomp.nodes as bn
import biocomp.compute as bcc
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [7.0, 7.0]
plt.rcParams['figure.dpi'] = 200

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
##────────────────────────────────────────────────────────────────────────────}}}

lib = ut.load_lib()
uorf_xp = ut.load_xp('2022-11-10_uORFs_and_company', lib)
ern_xp = ut.load_xp('20220501-GW-l1vsl2', lib)


config = {
    **bc.train.DEFAULT_CFG,
    **{
        'node_impl': node_impl,
        'epochs': 30,
        # "rng_key": np.random.randint(0, 2 ** 32),
        "rng_key": 1,
    },
}

dman = du.DataManager.from_xps([uorf_xp, ern_xp], config, inverse='all')
dman.set_subset([0,47])
key = jax.random.PRNGKey(config['rng_key'])
jmodels = dman.get_jitted_models()
models = dman.get_models()


# get wandb run
import wandb as wb
import pickle
project_name='quantile_v1'
run_code ='usby7330'
run = wb.Api().run(f'{project_name}/{run_code}')

# load latest params (latest_params.pkl)
param_file = run.file('latest_params.pkl').download(replace=True)
with open(param_file.name, 'rb') as f:
    params = pickle.load(f)

params['shared']

# mparams = {}
# m = models[47]
# m.node_namespace = None
# m.init(key, mparams)
# full_params = params
# full_params['node'] = mparams['node']

##
ut.plot_networks([m.network])

ut.plot_node('inv_translation', full_params, m, xlim=(0, 1), ylim=(-0.1, 2))
ut.plot_node('inv_transcription', full_params, m, xlim=(-0.1, 2), ylim=(0, 2))
ut.plot_node('transcription', full_params, m, xlim=(0, 2), ylim=(-0.05, 0.6))
ut.plot_node('translation', full_params, m, xlim=(-0.05, .6), ylim=(-0.01, 0.8))

extra = m.network.compute_graph[m.network.compute_graph.type == 'sequestron_ERN'].extra.to_list()
ut.plot_node('sequestron_ERN', full_params, m, xlim=(-0.01, 0.8), n_inputs=2,extra_args=extra[0])
ut.plot_node('output', full_params, m, xlim=(0, 3), ylim=(-0.3, 1.3))

m.apply_and_grad(full_params, np.array([0.5,0.5]), np.array([0.5,0.5,0.5]), key)

##
nmodels = len(models)
x_start = np.cumsum([m.n_inputs for m in models])[:-1]
y_start = np.cumsum([m.n_outputs for m in models])[:-1]

def flat_concat(*arrays):
    return jnp.concatenate([a.ravel() for a in arrays])

def apply_models(params, x, z, key):
    keys = jax.random.split(key, nmodels)
    xs = jnp.split(x, x_start)
    zs = jnp.split(z, y_start)
    res = [m.apply_and_grad(params, xx, zz, k) for m, xx, zz, k in zip(models, xs, zs, keys)]
    yhat, grads = zip(*res)
    return jnp.concatenate(yhat, axis=0), jnp.min(flat_concat(*grads))

apply_models(params, np.ones(20), np.ones(30), key)

keys = jax.random.split(key, 2)
yhat = vmap(apply_models, in_axes=(None, 0, 0, 0))(params, np.ones((2,20)), np.ones((2,30)), keys)

yhat


# TODO:
# force > 0
# force symmetry of inv/fwd nodes


### {{{                     --     uorfs on ern side     --


def any_uorf(lib, *_, **__):
    all_uORFs = lib.pc[lib.pc.category == 'uORF_group'].index.tolist()
    return [all_uORFs]


def P(name):
    return bc.Slot(lib, name)


uorfs = any_uorf(lib)[0]
any_uorf(lib)
uorfs = uorfs[:8]

import itertools
rec_uorfs = ['empty_tl'] + uorfs
ern_uorfs = ['empty_tl'] + uorfs
all_uorfs = itertools.product(rec_uorfs, ern_uorfs)
all_uorfs = list(all_uorfs)

NROWS = len(rec_uorfs)

invns = []
for rec_uorf, ern_uorf in tqdm(all_uorfs):
    tus = {
        'CasE': bc.TranscriptionUnit([P('hEF1a'),P(ern_uorf), P('CasE')]),
        'CasE_marker': bc.TranscriptionUnit([P('hEF1a'), P('mKate')]),
        'rec+eYFP': bc.TranscriptionUnit([P('hEF1a'), P(rec_uorf), P('CasE_rec'), P('eYFP')]),
        'rec_marker': bc.TranscriptionUnit([P('hEF1a'), P('eBFP')]),
    }
    aggregations = [['CasE','CasE_marker'], ['rec+eYFP', 'rec_marker']]
    sources = {tu_name: [tu_name] for tu_name, tu in tus.items()}
    n = bc.Network.from_dict(lib, 'v1', tus, sources, aggregations)
    invn = bc.inverted_network(n)[0]
    invns.append(invn)

# ut.plot_networks(invns)

##

invmodels = []
for invn in invns:
    model = bc.ComputeGraphModel(invn)
    model.build(node_impl = node_impl)
    mparams = {}
    model.init(key, mparams)
    full_params = params
    full_params['node'] = mparams['node']
    # model(full_params, np.array([0.5,0.5]), np.array([0.5,0.5,0.5]), rng_key=key)
    invmodels.append(model)

##

fig, axes = du.mkfig(NROWS, NROWS)
for i,j in tqdm(list(itertools.product(range(NROWS), range(NROWS)))):
    ax = axes[i,j]
    model = invmodels[i*NROWS+j]
    du.eval_model_plot(model, full_params, dman.rescale, ax)
    ax.set_title(f'REC:{rec_uorfs[i]}-ERN:{ern_uorfs[j]}')
print('done')


##────────────────────────────────────────────────────────────────────────────}}}
