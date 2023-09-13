### {{{                          --     imports     --
import matplotlib

import biocomp as bc
from biocomp import datautils as du
from jax.tree_util import Partial as partial
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
from evosax import CMA_ES
from evosax.utils import ESLog, FitnessShaper
import os
import joblib
import datetime

from matplotlib import pyplot as plt

dirname = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# matplotlib.use('agg')
matplotlib.rcParams['figure.dpi'] = 200

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                     --     generate networks     --

lib = su.load_lib()


def sequestron_ERN3p(get_param, get_quantized, **_):
    def apply(rna, ern, **_):
        # return rna * (1.0 - jnp.exp(-ern))
        return jnp.relu(ern - rna)

    return apply


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
colors = [P('mKate'), P('eBFP'), P('NeonGreen'), P('iRFP720')]

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
    'x0color': TU(colors[0]),
    'x1color': TU(colors[1]),
    'biascolor': TU(colors[2]),
    # output node
    'C_pos': TU(rec[2], colors[3]),
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


tus_single = {
    'A_pos': TU(rec[0], colors[3]),
    'A_neg': TU(ern[0]),
    'x0color': TU(colors[0]),
    'x1color': TU(colors[1]),
}
aggregations_single = [
    ['A_pos', 'x0color'],  # x0
    ['A_neg', 'x1color'],  # x1
]
sources_single = {tu_name: [tu_name] for tu_name, tu in tus_single.items()}
n_single = bc.Network.from_dict(lib, 'single_ERN', tus_single, sources_single, aggregations_single)
single_net = bc.inverted_network(n_single)[0]

# networks = [single_net]
networks = [bp_net]

# dirname = Path('~/Desktop/bandpass_attempt/v0/networks/').expanduser()
# dirname.mkdir(parents=True, exist_ok=True)
# su.plot_networks(networks, filenames=[f'{dirname}/network_{i}.pdf' for i in range(len(networks))])
su.plot_networks(networks, W=4500, H=4000, show=True, figsize=(22, 20))

NETWORK = networks[0]


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                       --     evo settings     --
evo_config = {
    'generations': 100,
    'popsize': 100,
    'elite_ratio': 0.2,
    'init_min': 0.1,
    'init_max': 1.0,
    'clip_min': 0.0,
    'clip_max': 1.0,
    'rng_key': 0,
    'static_params': [ut.STATIC_PATH, ut.SHARED_PATH],
}
rng = jax.random.PRNGKey(evo_config['rng_key'])
k, _ = jax.random.split(rng)
compute_config = cmp.DEFAULT_COMPUTE_CONFIG
training_params = ut.tree_to_np(joblib.load(f'../__cache/best_params.pkl'))

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                       --     param compression    --


def tkeys(d):
    if isinstance(d, dict):
        ks = {}
        for k, v in d.items():
            ks[k] = tkeys(v)
        return ks
    else:
        return d.shape

def add_param_labels(par, cstack):
    allnodes = cstack.get_all_nodes()
    shortnames = [f'{n.compute_node_id}' for n in allnodes]
    def create_names(t, base_name=None, skip=('node',)):
        if isinstance(t, dict):
            ks = {}
            for k, v in t.items():
                bn = f'{base_name}/' if base_name else ''
                n = f'{bn}{k}' if k not in skip else bn
                ks[k] = create_names(v, base_name=n, skip=skip)
            return ks
        else:
            if isinstance(t, (np.ndarray, jnp.ndarray)):
                return [f'{base_name}/{shortnames[i]}' for i in range(len(t))]
    return create_names(par)

def compress_params(dparams, mask, cstack):
    pflat, pdef = jax.tree_util.tree_flatten(dparams)
    mflat, mdef = jax.tree_util.tree_flatten(mask)
    labels = add_param_labels(dparams, cstack)
    shapes = tuple([p.shape[1:] for p in pflat])
    lflat, ldef = jax.tree_util.tree_flatten(labels)
    # full_Mflat = np.concatenate(
        # [np.repeat(m, np.prod(np.asarray(s))) for m, s in zip(mflat, shapes)]
    # )
    full_Mflat, full_Lflat = [], []
    i=0
    for m, s in zip(mflat, shapes):
        nprod = np.prod(np.asarray(s))
        full_Mflat.extend(np.repeat(m, nprod))
        n = m.shape[0]
        full_Lflat.extend(np.repeat(lflat[i:i+n], nprod ))
        i += n
    full_Mflat = np.array(full_Mflat)
    full_Lflat = np.array(full_Lflat)

    assert full_Mflat.shape == full_Lflat.shape, 'mask and labels do not match'

    full_Pflat = np.concatenate([p.ravel() for p in pflat])
    compressed = full_Pflat[full_Mflat]

    split_at_indices = np.cumsum([np.size(p) for p in pflat])
    uncompress_func = partial(
        uncompress_params, shapes=shapes, indices=split_at_indices, mflat=full_Mflat, pdef=pdef
    )

    restored = uncompress_func(compressed)
    r = jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda x, y: np.all(x == y), restored, dparams)
    )
    assert r, 'uncompress function does not work'

    clabels = full_Lflat[full_Mflat]
    return compressed, uncompress_func, clabels


def uncompress_params(compressed_params, shapes, indices, mflat, pdef):
    idx = jnp.where(mflat, jnp.cumsum(mflat) - 1, -1)
    full_Pflat_restored = jnp.where(idx == -1, 0, compressed_params[idx])
    Pflat_restored = jnp.split(full_Pflat_restored, indices)[:-1]
    P_reshaped = [p.reshape((-1, *s)) for p, s in zip(Pflat_restored, shapes)]
    P_restored = jax.tree_util.tree_unflatten(pdef, P_reshaped)
    return P_restored


dynamic, static_params = ut.split_params(new_params, evo_config['static_params'])
is_init = new_params['__static__']['is_init']
dyn_mask, _ = ut.split_params(is_init, evo_config['static_params'])
compressed_params, uncompress, labels = compress_params(dynamic, dyn_mask, compute_stack)
restored_params = jit(uncompress)(compressed_params)


##────────────────────────────────────────────────────────────────────────────}}}

### {{{             --     generate random bandpass designs     --
# in logspace.

training_config = train.DEFAULT_TRAINING_CONFIG
logtr_conf = {
    'offset': training_config['data_log_offset'],
    'maxv': training_config['data_max_value'],
    'factor': training_config['data_log_factor'],
    'threshold': training_config['data_log_poly_threshold'],
    'compression': training_config['data_log_poly_compression'],
}

tr = partial(du.tr, **logtr_conf)
inv_tr = partial(du.inv_tr, **logtr_conf)

vlims = np.array([logtr_conf['offset'], logtr_conf['maxv']])

vlims_log = tr(vlims) * 0.9

vrange = vlims_log[1] - vlims_log[0]
on_value = 0.4
off_value = 0.0
pmargin = 0.1

# define region limits by a point and a normal vector


# normals = jax.random.normal(rng, (NBORDERS, 2))
# normals = normals / jnp.linalg.norm(normals, axis=1, keepdims=True)


def gen_unit(teta_min, teta_max, key):
    teta = jax.random.uniform(key, (1,), minval=teta_min, maxval=teta_max)[0]
    return jnp.array([jnp.cos(teta), jnp.sin(teta)])


def is_inside(x, p, n):
    return jnp.dot(x - p, n) < 0


def inside_all(x, pvec, nvec):
    return jnp.all(vmap(is_inside, in_axes=(None, 0, 0))(x, pvec, nvec), axis=0)


@jit
def gen_bandpass(key):
    k0, k1, k2, k3 = jax.random.split(key, 4)

    left_normals = gen_unit(np.pi / 2, np.pi, k0)
    right_normals = -left_normals + jax.random.normal(k1, (1, 2)) * 0.2

    left_points = jax.random.uniform(
        k2, (1, 2), minval=vlims_log[0] + vrange * pmargin, maxval=vlims_log[1] - vrange * pmargin
    )
    d = jax.random.uniform(k3, (1,), minval=0.1, maxval=0.85)[0]
    right_points = left_points - d * (vlims_log[1] - vlims_log[0]) * left_normals

    points = jnp.concatenate([left_points, right_points], axis=0)
    normals = jnp.vstack([left_normals, right_normals])
    return points, normals


@partial(jit, static_argnums=(2,))
def gen_bandpass_xz(vlims, key, nsamples=10000):
    k0, k1 = jax.random.split(key, 2)
    points, normals = gen_bandpass(k0)
    x = jax.random.uniform(k1, (nsamples, 2), minval=vlims[0], maxval=vlims[1])
    z = vmap(inside_all, in_axes=(0, None, None))(x, points, normals)
    z = jnp.where(z, on_value, off_value)
    return x, z, (points, normals)


NBP = 10
bandpasses = [gen_bandpass_xz(vlims_log, k, nsamples=2000) for k in jax.random.split(rng, NBP)]

for x, z, _ in bandpasses:
    fig, ax = plt.subplots()
    ax.scatter(x[:, 0], x[:, 1], c=z, s=2, cmap='inferno', alpha=0.75, vmin=0, vmax=0.6)
    ax.set_xlim(vlims_log)
    ax.set_ylim(vlims_log)

BP = bandpasses[1]

##────────────────────────────────────────────────────────────────────────────}}}

### {{{       --     fitness & add the intensity of bias as extra parameter     --
def generate_fitness(
    x,
    y,
    network,
    evo_config,
    compute_config,
    training_params,
    key,
    out_protein='iRFP720',
    bias_protein='NeonGreen',
    outside_penalty=0.05,
):

    k0, k1, k2 = jax.random.split(key, 3)

    # generate the compute stack
    compute_stack = cmp.ComputeStack([network])
    compute_stack.build(compute_config)
    cstack_params = compute_stack.init(rng)
    new_params = compute_stack.use_shared_params(cstack_params, training_params)
    dynamic, static_params = ut.split_params(new_params, evo_config['static_params'])
    is_init = static_params['__static__']['is_init']
    dyn_mask, _ = ut.split_params(is_init, evo_config['static_params'])
    compressed_params, uncompress, labels = compress_params(dynamic, dyn_mask, compute_stack)

    output_id = network.get_output_proteins().index(out_protein)
    if bias_protein is None:
        bias_indices = []
    else:
        bias_indices = np.asarray([network.get_inverted_input_proteins().index(bias_protein)])

    nbias = len(bias_indices)

    vcompute = jax.vmap(compute_stack.apply, in_axes=(None, 0, 0, 0))

    flat_params_size = len(compressed_params) + nbias

    labels = list(labels) + [f'bias_{i}' for i in range(nbias)]
    # Q = jax.random.uniform(k0, (x.shape[0], 4))
    Q_std = 0.1
    Q = jnp.clip(jax.random.normal(rng, (x.shape[0], 4)) * Q_std + 0.5, 0.1, 0.9)

    def make_full_x(x, extra_x):
        if nbias > 0:
            full_x = jnp.insert(x, bias_indices, extra_x, axis=1)
        else:
            full_x = x
        return full_x

    def reconstruct_params_and_biases(flat_params):
        # separate NN params from optimized inputs (aka biases)
        if nbias == 0:
            compressed_network_params = flat_params
            extra_x = []
        else:
            compressed_network_params = flat_params[:-nbias]
            extra_x = flat_params[-nbias:]
        dyn_network_params = uncompress(compressed_network_params)
        params = ut.assemble_params(dyn_network_params, static_params)
        params = ut.tree_to_jax(params)
        return params, partial(make_full_x, extra_x=extra_x)

    MIN_PARAM = 0.0
    MAX_PARAM = 1.0

    def compute(flat_params, x, Q, k):
        clipped_params = jnp.clip(flat_params, MIN_PARAM, MAX_PARAM)
        params, make_full_x = reconstruct_params_and_biases(clipped_params)
        full_x = make_full_x(x)
        keys = jax.random.split(k, x.shape[0])
        yhat, _ = vcompute(params, full_x, Q, keys)
        yhat = yhat[:, output_id]
        return yhat

    def fitness_fn(flat_params):

        yhat = compute(flat_params, x, Q, k1)
        score = jnp.mean((yhat - y) ** 2)
        # anything outside of 0, 1 is penalized
        under = flat_params < MIN_PARAM
        over = flat_params > MAX_PARAM
        outside = jnp.where(
            under, MIN_PARAM - flat_params, jnp.where(over, flat_params - MAX_PARAM, 0)
        )
        outside = jnp.sum(outside) / jnp.maximum(jnp.sum(under + over), 1)

        return score + outside * outside_penalty

    return fitness_fn, flat_params_size, compute, reconstruct_params_and_biases, labels


x, y, _ = BP

bias_protein = None
if 'NeonGreen' in NETWORK.get_inverted_input_proteins():
    bias_protein = 'NeonGreen'

(fitness_fn, flat_params_size, compute, reconstruct_params_and_biases, labels) = generate_fitness(
    x, y, NETWORK, evo_config, compute_config, training_params, rng, bias_protein=bias_protein
)

# fpar = jax.random.uniform(k, flat_params_shape, minval=cfg['init_min'], maxval=cfg['init_max'])

# generate the compute stack
compute_stack = cmp.ComputeStack([NETWORK])
compute_stack.build(compute_config)
cstack_params = compute_stack.init(rng)
new_params = compute_stack.use_shared_params(cstack_params, training_params)
dynamic, static_params = ut.split_params(new_params, evo_config['static_params'])
is_init = static_params['__static__']['is_init']
dyn_mask, _ = ut.split_params(is_init, evo_config['static_params'])

static_params['shared']['qvals']

tlrates = training_params['shared']['qvals']['tl_rate'][0].ravel()
plt.plot(tlrates)

tkeys(training_params['__static__'])
tkeys(training_params['__static__']['qmasks'])
new_params['__static__']['qmasks']['tl_rate']
new_params['__static__']['qmasks']['tl_rate']
tkeys(training_params)


##────────────────────────────────────────────────────────────────────────────}}}

strategy = CMA_ES(
    popsize=evo_config['popsize'], num_dims=flat_params_size, elite_ratio=evo_config['elite_ratio']
)
es_params = strategy.default_params
es_params.replace(
    init_min=evo_config['init_min'],
    init_max=evo_config['init_max'],
    clip_min=evo_config['clip_min'],
    clip_max=evo_config['clip_max'],
)
state = strategy.initialize(rng, es_params)

history = {
    'fitnesses': [],
    'individuals': [],
}

vm_fitness = jit(vmap(fitness_fn))

for g in tqdm(list(range(evo_config['generations'])), desc='generations'):
    rng, rng_gen, rng_eval = jax.random.split(rng, 3)
    samples, state = strategy.ask(rng_gen, state, es_params)
    fitnesses = vm_fitness(samples)
    state = strategy.tell(samples, fitnesses, state, es_params)
    history['fitnesses'].append(fitnesses)
    f_argmin = jnp.argmin(fitnesses)
    history['individuals'].append(samples)
    print("Generation: ", g, "Performance: ", state.best_fitness)

best_fitness = state.best_fitness
best_params = state.best_member

best_fitness
savedir = Path('./results')
savedir.mkdir(exist_ok=True)

# save the best network
full_best_params, _ = reconstruct_params_and_biases(best_params)
full_best_params = ut.tree_to_np(full_best_params)
# save
fname = f'{NETWORK.name}_cmaes_{evo_config["popsize"]}_{evo_config["generations"]}_best_params_f={best_fitness:.4f}.pkl'
with open(savedir / fname, 'wb') as f:
    pickle.dump(best_params, f)

##

# plot fitness history
allfitnesses = np.asarray(history['fitnesses'])
bestfitnesses = np.nanmin(allfitnesses, axis=1)
medianfitnesses = np.nanmedian(allfitnesses, axis=1)
fig, ax = plt.subplots()
ax.plot(bestfitnesses, label='best', color='red')
ax.plot(medianfitnesses, label='median', ls='--', color='black')
ax.legend()
ax.set_xlabel('generation')
ax.set_ylabel('fitness')
ax.set_yscale('log')
ax.set_title('fitness history')
fname = f'{NETWORK.name}_cmaes_{evo_config["popsize"]}_{evo_config["generations"]}_run_plot.png'
fig.savefig(savedir / fname, dpi=300)

best_params
##
allindividuals = np.asarray(history['individuals'])
allindividuals.shape
bestfitnesses_idx = np.argmin(allfitnesses, axis=0)
bestfitnesses_idx.shape
bestindividuals = np.take_along_axis(
    allindividuals, bestfitnesses_idx[:, None, None], axis=1
).squeeze().T
variance_per_param = np.log(1+np.var(allindividuals, axis=1)).T
variance_per_param.shape
# plot as heatmap
fig, ax = plt.subplots(figsize=(10, 10))
# ax.imshow(bestindividuals, aspect='auto', cmap='viridis')
# as plots:
for i in range(bestindividuals.shape[0]):
    ax.plot(bestindividuals[i], label=labels[i])
# ax.legend()

# write labels as xticks
len(labels)
# ax.set_yticks(np.arange(len(labels)))
# ax.set_yticklabels(labels)
ax.set_ylabel('parameter')
ax.set_xlabel('generation')
ax.set_title('best individuals')

##

# reconstruct the best network
x, y, _ = BP
# Q = jax.random.uniform(rng, (x.shape[0], 4))
# Q = jnp.ones((x.shape[0], 4))*0.5
Q_std = 0.1
Q = jnp.clip(jax.random.normal(rng, (x.shape[0], 4)) * Q_std + 0.5, 0.1, 0.9)

best_params
par = best_params
# par = jnp.ones_like(best_params) * 0.5
yhat = jit(compute)(par, x, Q, rng)
err = (yhat - y) ** 2

np.nanmin(allfitnesses)

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
vmax = np.max([y.max(), yhat.max()])
s0 = axes[0].scatter(x[:, 0], x[:, 1], c=y, s=10, vmin=0, vmax=vmax)
s1 = axes[1].scatter(x[:, 0], x[:, 1], c=yhat, s=10, vmin=0, vmax=vmax)
fig.colorbar(s0, ax=axes[0])
axes[0].set_title('Target')
axes[1].set_title('Predicted')
fname = f'{NETWORK.name}_cmaes_{evo_config["popsize"]}_{evo_config["generations"]}_fitness{best_fitness:.3f}.png'
fig.savefig(savedir / fname, dpi=300)


yhat

### {{{                          --     archive     --
# mask = new_params['__static__']['is_init']

# mr, unravel_m = jax.flatten_util.ravel_pytree(mask['node'])
# dr, unravel_d = jax.flatten_util.ravel_pytree(dynamic['node'])

# mf, md = jax.tree_util.tree_flatten(mask['node'])
# df, dd = jax.tree_util.tree_flatten(dynamic['node'])

# keys(mask)


# def compress_flatten_params(dynamic, mask):
# Pflat, Pdef = jax.tree_util.tree_flatten(dynamic)
# Mflat, Mdef = jax.tree_util.tree_flatten(mask)
# assert Pdef == Mdef
# res = jnp.concatenate([p[m].ravel() for p, m in zip(Pflat, Mflat)])
# shapes = [p.shape[1:] for p in Pflat]
# descriptor = (shapes, Mflat, Mdef)
# return res, descriptor


# def uncompress_flatten_params(compressed_params, descriptor):
# shapes, mask_flattened, mask_descript = descriptor
# params = []
# idx = 0
# for s, m in zip(shapes, mask_flattened):
# mask_size = m.shape[0]
# flat_size = jnp.prod(jnp.array(s))
# npositive_mask = jnp.sum(m)
# next_idx = idx + npositive_mask*flat_size
# resized = compressed_params[idx:next_idx].reshape((-1, *s))
# ids = jnp.where(m)[0]
# full = jnp.zeros((mask_size, *s))
# full = full.at[ids].set(resized)
# params.append(full)
# idx = next_idx
# return jax.tree_util.tree_unflatten(mask_descript, params)

# compressed_params, descriptor = compress_flatten_params(dynamic['node'], mask['node'])
# restored_params = uncompress_flatten_params(compressed_params, descriptor)
# assert np.all(jax.flatten_util.ravel_pytree(dynamic['node'])[0] == jax.flatten_util.ravel_pytree(restored_params)[0])


##────────────────────────────────────────────────────────────────────────────}}}
