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

import biocomp.defaults as bdf

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

### {{{                        --     load params     --
# get wandb run
import wandb as wb
import pickle
project_name='quantile_v2'
run_code ='v1ruml8t'
run_code='q0q94g0w'
run = wb.Api().run(f'{project_name}/{run_code}')

# load latest params (latest_params.pkl)
param_file = run.file('latest_params.pkl').download(replace=True)
with open(param_file.name, 'rb') as f:
    params = pickle.load(f)

##────────────────────────────────────────────────────────────────────────────}}}lib = ut.load_lib()

### {{{                         --     load xps     --
lib = ut.load_lib()

config = {
    **bdf.DEFAULT_CONFIG,
    **{
        'node_impl': node_impl,
        'epochs': 30,
        # "rng_key": np.random.randint(0, 2 ** 32),
        "rng_key": 1,
    },
}
# config2 = {**config, **{'log_factor':1e3, 'max_value':1e9}}

key = jax.random.PRNGKey(config['rng_key'])

uorf_xp = ut.load_xp('2022-11-10_uORFs_and_company', lib)
ern_xp = ut.load_xp('20220501-GW-l1vsl2', lib)
real_data = '2023-01-22_CasE_ALLuORFs'
real_xp = ut.load_xp(real_data, lib)
real_dman = du.DataManager.from_xps([real_xp], config)
##
mass_xp = ut.load_xp('E20221012A_massCtrls', lib)
mass_dman = du.DataManager.from_xps([mass_xp], config)
# ut.plot_networks([m.network for m in mass_dman.get_models()])
mass_mnames = [m.node_namespace for m in mass_dman.get_models()]
mass_mnames

plot_dist2d(mass_dman, 0)

##────────────────────────────────────────────────────────────────────────────}}}
from jax.scipy.stats import gaussian_kde

def plot_dist2d(dman, mid):
    fig, ax = du.mkfig(1,1, (10,10))
    mnames = [m.node_namespace for m in dman.get_models()]
    model = dman.get_models()[mid]
    rawx = dman.get_raw_X()[mid]
    input_name = model.get_inverted_input_proteins()
    reordered_input = sorted(input_name)[::-1]
    if reordered_input != input_name:
        rawx = rawx[:, [input_name.index(i) for i in reordered_input]]
    XX  = np.array([rawx[:,0], rawx[:,1]]).T
    ax.scatter(XX[:,0], XX[:,1], s=1,  alpha=1, color='k', linewidth=0, marker=',')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(reordered_input[0])
    ax.set_ylabel(reordered_input[1])
    ax.set_xlim(1, 1e9)
    ax.set_ylim(1, 1e9)
    kde = gaussian_kde(XX.T, bw_method=1)
    densities = kde(XX.T)
    max_density_coords = XX[np.argmax(densities)]
    # mark with a red cross
    # ax.scatter(max_density_coords[0], max_density_coords[1], marker='x', color='r', s=100, linewidth=2)
    ax.set_title(f'{mnames[mid]}\n raw x data distribution')

def plot_dist1d(dman, mid):
    fig, ax = du.mkfig(1,1, (10,10))
    mnames = [m.node_namespace for m in dman.get_models()]
    model = dman.get_models()[mid]
    rawx = dman.get_raw_X()[mid]
    rawy = dman.get_raw_Y()[mid]
    input_name = model.get_inverted_input_proteins()
    output_names = model.get_output_proteins()
    output = list(set(output_names) - set(input_name))
    output_pos = output_names.index(output[0])
    rawy = rawy[:, output_pos]
    XX  = np.array([rawx[:,0], rawy]).T
    ax.scatter(rawx, rawy, s=1,  alpha=1, color='k', linewidth=0, marker=',')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(input_name[0])
    ax.set_ylabel(output[0])
    ax.set_xlim(1, 1e9)
    ax.set_ylim(1, 1e9)
    kde = gaussian_kde(XX.T, bw_method=1)
    densities = kde(XX.T)
    max_density_coords = XX[np.argmax(densities)]
    # mark with a red cross
    ax.scatter(max_density_coords[0], max_density_coords[1], marker='x', color='r', s=100, linewidth=2)
    ax.set_title(f'{mnames[mid]}\n raw x data distribution')


dman = du.DataManager.from_xps([uorf_xp, ern_xp], config, inverse='all')
mnames = [m.node_namespace for m in dman.get_models()]

plot_dist2d(dman, -1)
plot_dist1d(dman, 0)

plot_dist2d(dman, -2)
plot_dist2d(dman, -3)


print('done')

##

real_data = '2023-01-22_CasE_ALLuORFs'
real_xp = ut.load_xp(real_data, lib)
# config2 = {**config, **{'log_factor':1e3, 'max_value':1e9}}
real_dman = du.DataManager.from_xps([real_xp], config)
real_mnames = [m.network.name for m in real_dman.get_models()]


rx = real_dman.get_raw_X().copy()
ry = real_dman.get_raw_Y().copy()

## 
a = np.array([0.85, 1.2])
b = - np.array([1.2, 4.2])
rrx = [10**jnp.clip(a * jnp.log10(r)+ b, 0) for r in rx]
# rry = [10**jnp.clip(a * jnp.log10(r)+ b, 0) for r in ry]

real_dman._raw_X = rrx
# real_dman._raw_Y[0] = 10**rry
# real_dman._raw_X = real_dman.unscale(real_dman._X)
# real_dman._raw_Y = real_dman.rescale(real_dman._Y)
# self._X = self.rescale(self._raw_X)
plot_dist2d(real_dman, 1)
##

real_dman._X = real_dman.rescale(real_dman._raw_X)

real_dman._kdes = [gaussian_kde(x.T, bw_method=0.05) for x in real_dman._X]
# real_dman._Y = real_dman.rescale(real_dman._raw_Y)

real_mnames
fig, axes = du.mkfig(1,9)

maxy = []
for i in range(9):
    mid = i + 1
    model = real_dman.get_models()[mid]
    input_name = model.get_inverted_input_proteins()
    output_names = model.get_output_proteins()
    output = list(set(output_names) - set(input_name))
    output_pos = output_names.index(output[0])
    mY = real_dman.get_Y()[mid]
    maxy.append(mY[:, output_pos].max())
vmax = max(maxy)



for i in range(9):
    mid = i + 1
    ax = axes[i]
    model = real_dman.get_models()[mid]
    mX = real_dman.get_X()[mid]
    mY = real_dman.get_Y()[mid]
    du.model_plot(model, mX, mY, real_dman.rescale, ax, kde=real_dman.get_kdes()[mid], vmax=vmax)




### {{{                        --     plot nodes     --

ut.plot_networks([m.network])

# ut.plot_node('inv_translation', full_params, m, xlim=(0, 1), ylim=(-0.1, 2))
# ut.plot_node('inv_transcription', full_params, m, xlim=(-0.1, 2), ylim=(0, 2))
# ut.plot_node('transcription', full_params, m, xlim=(0, 2), ylim=(-0.05, 0.6))
# ut.plot_node('translation', full_params, m, xlim=(-0.05, .6), ylim=(-0.01, 0.8))
# ut.plot_node('output', full_params, m, xlim=(0, 3), ylim=(-0.3, 1.3))


ut.plot_node('inv_translation', full_params, m, xlim=(0, 1), ylim=(-1, 2))
ut.plot_node('inv_transcription', full_params, m, xlim=(0, 1), ylim=(-1, 2))
ut.plot_node('transcription', full_params, m, xlim=(0, 2), ylim=(0, 2))
ut.plot_node('translation', full_params, m, xlim=(0, 2), ylim=(0, 2))
# ut.plot_node('translation', full_params, m, xlim=(-0.05, .6), ylim=(-0.01, 0.8))
ut.plot_node('output', full_params, m, xlim=(0, 2), ylim=(0, 2))

# extra = m.network.compute_graph[m.network.compute_graph.type == 'sequestron_ERN'].extra.to_list()
# ut.plot_node('sequestron_ERN', full_params, m, xlim=(-0.01, 0.8), n_inputs=2,extra_args=extra[0])


##────────────────────────────────────────────────────────────────────────────}}}m.apply_and_grad(full_params, np.array([0.5,0.5]), np.array([0.5,0.5,0.5]), key)

### {{{                     --     uorfs on ern side     --

train_dman = du.DataManager.from_xps([ern_xp, uorf_xp], config, inverse='all')

models = train_dman.get_models()
mparams = {}
m = models[47]
m.node_namespace = None
m.init(key, mparams)
full_params = params
full_params['node'] = mparams['node']


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

# fig, axes = du.mkfig(NROWS, NROWS)
# for i,j in tqdm(list(itertools.product(range(NROWS), range(NROWS)))):
    # ax = axes[i,j]
    # model = invmodels[i*NROWS+j]
    # du.eval_model_plot(model, full_params, dman.rescale, ax)
    # ax.set_title(f'REC:{rec_uorfs[i]}-ERN:{ern_uorfs[j]}')
# print('done')

##

# maxy = []
# for i in tqdm(list(range(NROWS))[:]):
    # model = invmodels[i]
    # input_name = model.get_inverted_input_proteins()
    # output_names = model.get_output_proteins()
    # output = list(set(output_names) - set(input_name))
    # output_pos = output_names.index(output[0])
    # mY = real_dman.get_Y()[mid]
    # maxy.append(mY[:, output_pos].max())
# vmax = max(maxy)

fig, axes = du.mkfig(1, NROWS)
for i in tqdm(list(range(NROWS))[:]):
    ax = axes[i]
    model = invmodels[i]
    du.eval_model_plot(model, full_params, dman.rescale, ax, npoints=100000, vmax=vmax)
    ax.set_title(f'ERN:{ern_uorfs[i]}')
print('done')

##────────────────────────────────────────────────────────────────────────────}}}
