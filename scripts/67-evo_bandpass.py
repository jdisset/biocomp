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
NETWORK = bc.inverted_network(n_bp)[0]
NETWORK.set_input_as_bias(biascolor)
# dirname = Path('~/Desktop/bandpass_attempt/v0/networks/').expanduser()
# dirname.mkdir(parents=True, exist_ok=True)
# su.plot_networks(networks, filenames=[f'{dirname}/network_{i}.pdf' for i in range(len(networks))])
su.plot_networks([NETWORK], W=4500, H=4000, show=True, figsize=(22, 20))

##────────────────────────────────────────────────────────────────────────────}}}
### {{{               --     load parameters & initialize stack    --

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
full_params


def tag_single_quantized_params(params):
    for k, _ in params.data.iter_leaves():
        if k[-1] == 'affinity' or k[-1] == 'tc_rate':
            params.tag(k, 'single_val')
        if k[-1] == 'tl_rate':
            params.tag(k, 'quantized')
            qm = params[pm.ParamPath(k[:-1] + ['tl_rate_quantization_mask'])]
            if np.all(np.sum(qm, axis=0) <= 1):
                params.tag(k, 'single_val')
    return params


full_params = tag_single_quantized_params(full_params)
static_params, dynamic_params = full_params.filter_by_tag(
    ['shared', 'non_grad', 'single_val'], mode='any'
)

flat_dynamic, unravel_params = jax.flatten_util.ravel_pytree(dynamic_params)
assert unravel_params(flat_dynamic) == dynamic_params


##────────────────────────────────────────────────────────────────────────────}}}
### {{{             --     generate random bandpass designs     --
# in logspace.

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

vlims_log = tr(vlims) * 0.90

vrange = vlims_log[1] - vlims_log[0]
on_value = 1.0
off_value = 0.0
pmargin = 0.1


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

    left_normals = gen_unit(-np.pi, np.pi, k0)

    right_normals = -left_normals + jax.random.normal(k1, (1, 2)) * 0.2

    left_points = jax.random.uniform(
        k2, (1, 2), minval=vlims_log[0] + vrange * pmargin, maxval=vlims_log[1] - vrange * pmargin
    )
    d = jax.random.uniform(k3, (1,), minval=0.5, maxval=0.85)[0]
    right_points = left_points - d * (vlims_log[1] - vlims_log[0]) * left_normals

    points = jnp.concatenate([left_points, right_points], axis=0)
    normals = jnp.vstack([left_normals, right_normals])
    return points, normals


@partial(jit, static_argnums=(2,))
def gen_bandpass_xz(vlims, key, nsamples=10000):
    k0, k1 = jax.random.split(key, 2)
    points, normals = gen_bandpass(k0)
    x = jax.random.uniform(k1, (nsamples, 2), minval=vlims[0], maxval=vlims[1])
    y = vmap(inside_all, in_axes=(0, None, None))(x, points, normals)
    y = jnp.where(y, on_value, off_value)
    y = y.reshape(-1, 1)
    return x, y


rng = jax.random.PRNGKey(1)
NBP = 15
bandpasses = [gen_bandpass_xz(vlims_log, k, nsamples=2000) for k in jax.random.split(rng, NBP)]

for i, (x, z) in enumerate(bandpasses):
    print(f'bandpass {i}')
    fig, ax = plt.subplots()
    ax.scatter(x[:, 0], x[:, 1], c=z, s=2, cmap='YlGnBu', alpha=0.75, vmin=0, vmax=0.6)
    ax.set_xlim(vlims_log)
    ax.set_ylim(vlims_log)
    plt.show()

##────────────────────────────────────────────────────────────────────────────}}}

evo_config = {
    'generations': 30,
    'popsize': 50,
    'elite_ratio': 0.4,
    'init_min': -0.5,
    'init_max': 1.5,
    'clip_min': -0.5,
    'clip_max': 1.5,
    'rng_key': 22,
}

rng = jax.random.PRNGKey(evo_config['rng_key'])
k, _ = jax.random.split(rng)
from evosax import CMA_ES
from evosax.utils import ESLog, FitnessShaper

strategy = CMA_ES(
    popsize=evo_config['popsize'],
    num_dims=flat_dynamic.shape[0],
    elite_ratio=evo_config['elite_ratio'],
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

# jax.clear_caches()

vmapped_compute = jax.vmap(stack.apply, in_axes=(None, 0, 0, 0))


@jit
def evaluate_at(params, X, Z, key):
    keys = jax.random.split(key, X.shape[0])
    full_yhat, _ = vmapped_compute(params, X, Z, keys)
    yhat = full_yhat[:, output_indices]
    if yhat.ndim == 1:
        yhat = yhat.reshape(-1, 1)
    return yhat


def plot_eval(params, res=100, xlims=(0, 1), vmin=0, vmax=1):
    X = np.meshgrid(np.linspace(*xlims, res), np.linspace(*xlims, res))
    X = np.stack(X, axis=-1).reshape(-1, 2)
    Z = jnp.ones((res * res, stack.total_nb_of_outputs)) * 0.5
    Y = evaluate_at(params, X, Z, key)
    fig, ax = du.mkfig(1, 1)
    im = ax.imshow(
        Y.reshape(res, res),
        extent=[*xlims, *xlims],
        cmap='YlGnBu',
        origin='lower',
        vmin=vmin,
        vmax=vmax,
    )
    ax.contour(
        Y.reshape(res, res),
        2,
        extent=[*xlims, *xlims],
        colors='k',
        origin='lower',
        alpha=0.25,
        vmin=vmin,
        vmax=vmax,
    )
    # colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xlabel('$X_1$')
    ax.set_ylabel('$X_2$')
    return fig, ax


x, y = bandpasses[0]


accuracy_weight = 1.0
dynrange_weight = 1.0
def fitness_func(flat_dynamic, key):
    full_params = ParameterTree.merge(static_params, unravel_params(flat_dynamic))
    z = jax.random.uniform(key, (x.shape[0], stack.total_nb_of_outputs))
    yhat = evaluate_at(full_params, x, z, key)
    assert yhat.shape == y.shape, f"yhat shape: {yhat.shape}, y shape: {y.shape}"


    y_on = (y > 0.5).squeeze()
    y_off = (y < 0.5).squeeze()
    n_on = jnp.sum(y_on)
    n_off = jnp.sum(y_off)
    yhat_on_avg = jnp.sum(jnp.where(y_on, yhat, 0)) / jnp.maximum(n_on, 1)
    yhat_off_avg = jnp.sum(jnp.where(y_off, yhat, 0)) / jnp.maximum(n_off, 1)
    return yhat_off_avg - yhat_on_avg


@jit
def update(key, state, es_params):
    samples, state = strategy.ask(key, state, es_params)
    fitnesses = vmap(fitness_func)(samples, jax.random.split(key, samples.shape[0]))
    state = strategy.tell(samples, fitnesses, state, es_params)
    return state, fitnesses, samples


for g in tqdm(list(range(evo_config['generations'])), desc='generations'):
    rng, key = jax.random.split(rng)
    state, fitnesses, samples = update(rng, state, es_params)
    f_argmin = jnp.argmin(fitnesses)
    history['fitnesses'].append(fitnesses)
    history['individuals'].append(samples)
    history['best_fitness'] = fitnesses[f_argmin]
    history['best_individual'] = samples[f_argmin]
    print("Generation: ", g, "Performance: ", state.best_fitness)

best_fitness = state.best_fitness
best_params = state.best_member

##
from datetime import datetime

savedir = Path('./results') / 'cmaes' / datetime.now().strftime('%Y%m%d_%H%M%S')
savedir.mkdir(parents=True, exist_ok=True)

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
# ax.set_yscale('log')
ax.set_title('fitness history')
fname = f'{NETWORK.name}_cmaes_{evo_config["popsize"]}_{evo_config["generations"]}_run_plot.png'
fig.savefig(savedir / fname, dpi=300)

best_fullparams = ParameterTree.merge(static_params, unravel_params(best_params))
plot_eval(best_fullparams, res=100, xlims=(0, 1))

##

logtr_conf = {
    'offset': training_config['data_log_offset'],
    'maxv': training_config['data_max_value'],
    'factor': training_config['data_log_factor'],
    'threshold': training_config['data_log_poly_threshold'],
    'compression': training_config['data_log_poly_compression'],
}
tr = partial(du.tr, **logtr_conf)
inv_tr = partial(du.inv_tr, **logtr_conf)



@jit
def bp_eval(x, points, normals):
    y = vmap(inside_all, in_axes=(0, None, None))(x, points, normals)
    y = y.reshape(-1, 1)
    return y


res = 150
xlims = (0, 1)
vmin = 0
vmax = 1
X = np.meshgrid(np.linspace(*xlims, res), np.linspace(*xlims, res))
X = np.stack(X, axis=-1).reshape(-1, 2)
Z = jnp.ones((res * res, stack.total_nb_of_outputs)) * 0.5

key = jax.random.PRNGKey(13)
Y = bp_eval(X, *gen_bandpass(key))

fig, ax = du.mkfig(1, 1)
im = ax.imshow(
    Y.reshape(res, res),
    extent=[*xlims, *xlims],
    cmap='YlGnBu',
    origin='lower',
    vmin=vmin,
    vmax=vmax,
)
ax.contour(
    Y.reshape(res, res),
    2,
    extent=[*xlims, *xlims],
    colors='k',
    origin='lower',
    alpha=0.25,
    vmin=vmin,
    vmax=vmax,
)

# colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.1)
fig.colorbar(im, cax=cax, orientation='vertical')
ax.set_xlabel('$X_1$')
ax.set_ylabel('$X_2$')












