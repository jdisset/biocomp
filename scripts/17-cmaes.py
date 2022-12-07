## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     import     --
# ···············································································
import biocomp as bc
import biocomp.compute as bcc
import scriptutils as ut
import jax.numpy as jnp
from tqdm import tqdm
import numpy as np
import jax
import optax
import random
import biocomp.datautils as du
import matplotlib.pyplot as plt
import itertools
from functools import partial
import biocomp.nodes as bn
import scriptutils as ut
import pickle
import jax.numpy as jnp
from evosax import CMA_ES

import more_itertools as mit


from tqdm import tqdm
from evosax.utils import ESLog, FitnessShaper

random.seed()
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                     --     generate networks     --
# ···············································································

lib = ut.load_lib()


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


# 'Csy4+uOrfs': bc.TranscriptionUnit([P('hEF1a'), P('Csy4'), P(any_uorf(lib)[0])]),

uorfs = any_uorf(lib)[0]

tus = {
    'Csy4#1': bc.TranscriptionUnit([P('hEF1a'), P('Csy4')]),
    'Csy4#2': bc.TranscriptionUnit([P('hEF1a'), P('Csy4')]),
    'PgU#1': bc.TranscriptionUnit([P('hEF1a'), P('CasE_rec'), P('PgU')]),
    'PgU#2': bc.TranscriptionUnit([P('hEF1a'), P('CasE_rec'), P('PgU')]),
    'B_bias': bc.TranscriptionUnit([P('hEF1a'), P('Csy4_rec'), P('PgU')]),
    'A_bias': bc.TranscriptionUnit([P('hEF1a'), P('CasE')]),
    'out': bc.TranscriptionUnit([P('hEF1a'), P('PgU_rec'), P('NeonGreen')]),
}
aggregations = [['Csy4#1', 'PgU#1'], ['Csy4#2', 'PgU#2'], ['B_bias'], ['A_bias'], ['out']]

tus = {
    'Csy4rec#1': bc.TranscriptionUnit([P('hEF1a'), P('Csy4_rec'), P('CasE')]),
    'Csy4#2': bc.TranscriptionUnit([P('hEF1a'), P('Csy4')]),
    'PgUrec#2': bc.TranscriptionUnit([P('hEF1a'), P('PgU_rec'), P('CasE')]),
    'PgU#1': bc.TranscriptionUnit([P('hEF1a'), P('PgU')]),
    'PgUrec_bias': bc.TranscriptionUnit([P('hEF1a'), P('PgU_rec'), P('CasE')]),
    'B_bias': bc.TranscriptionUnit([P('hEF1a'), P('PgU')]),
    'A_bias': bc.TranscriptionUnit([P('hEF1a'), P('Csy4')]),
    'out': bc.TranscriptionUnit([P('hEF1a'), P('CasE_rec'), P('NeonGreen')]),
}
# aggregations = [['Csy4rec#1', 'PgU#1'], ['Csy4#2', 'PgUrec#2'], ['out'], ['A_bias'], ['B_bias']]
aggregations = [['Csy4rec#1', 'PgU#1'], ['Csy4#2', 'PgUrec#2'], ['out', 'A_bias', 'B_bias', 'PgUrec_bias']]
# aggregations = [['Csy4rec#1', 'PgU#1'], ['Csy4#2', 'PgUrec#2'], ['out']]

sources = {tu_name: [tu_name] for tu_name, tu in tus.items()}

networks = []

n = bc.Network.from_dict(lib, 'v1', tus, sources, aggregations)
n2 = bc.Network.from_dict(lib, 'v2', tus, sources, aggregations)
inputs = (
    n.compute_graph[n.compute_graph.type == 'aggregation']
    .input_from.apply(lambda x: x[0][0])
    .to_list()
)
n.set_inputs([inputs[0],inputs[1]])
n2.set_inputs([inputs[1],inputs[0]])
# n2.set_inputs(list(reversed(inputs)))
networks = [n, n2]

print(f'Generated {len(networks)} networks')

import os

import datetime

dirname = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# os.makedirs(f'../__out/{dirname}', exist_ok=True)

# ut.plot_networks(networks, [f'../__out/{dirname}/{n.name}.pdf' for n in networks])
ut.plot_networks(networks)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                     --     load target data     --
# ···············································································

img = plt.imread('../data/bandpass_inc.png')
target = img[:, :, 0]
target = target[:, ::-1]

N_SAMPLES = 5000
x = np.random.randint(0, target.shape[0], size=(N_SAMPLES, 2))
y = target[x[:, 0], x[:, 1]]
Y = y[:, None]
# normalize x
X = x / max(target.shape)

# jk, let's try a target of just ones if x[0] < 0.5 and zeros otherwise
# y = (x[:, 0] < 0.5).astype(np.float32)
# add 1 dimension to y:

# plot target
plt.figure(figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], c=Y, s=10)
plt.show()

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                           --     model     --
#···············································································
net = networks[1]
model = bc.ComputeGraphModel(net)
model.build()
params, constraints = model.init(jax.random.PRNGKey(np.random.randint(0, 1000000)))
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                           --     utils     --
#···············································································
def flatten_params(params):
    leaves, treedef = jax.tree_util.tree_flatten(params)
    flat_leaves = [l.flatten() for l in leaves]
    shapes = [l.shape for l in leaves]
    flat_params = np.concatenate(flat_leaves)
    return flat_params, (shapes, treedef)

def unflatten_params(flat_params, pdef):
    shapes, treedef = pdef
    splits = np.cumsum([np.prod(s) for s in shapes], dtype=np.int32)
    leaves = []
    start = 0
    for sp, sh in zip(splits, shapes):
        leaves.append(flat_params[start:sp].reshape(sh))
        start = sp
    params = jax.tree_util.tree_unflatten(treedef, leaves)
    return params

flat_params, pdef = flatten_params(params)

def loss_fn(params, x, y):
    vm = jax.vmap(model, in_axes=(None, 0, None))
    yhat = vm(params, x, jax.random.PRNGKey(0))
    return jnp.mean((yhat - y) ** 2)

def fitness_fn(flat_params, x, y, reg=0.001, neg_reg=100.0):
    p = unflatten_params(flat_params, pdef)
    l = loss_fn(p, x, y)
    # we want to add a penalty that encourages 
    # parameters in flat_params to be close to one
    penalty = reg * jnp.mean((flat_params - 1) ** 2)
    # let's also penalize negative values
    penalty += neg_reg * jnp.mean(jnp.where(flat_params < 0, flat_params, 0)**2)
    return l + penalty

def plot_fitnesses(fitnesses):
    # we will plot:
    # - best fitness
    # - median
    fitarr = np.array(fitnesses)
    print(f'Best fitness: {fitnesses[-1].min()}')
    print(f'fitnes shape: {fitarr.shape}')
    best = np.min(fitarr, axis=1)
    median = np.median(fitarr, axis=1)
    smoothwin = 5
    best = np.convolve(best, np.ones(smoothwin)/smoothwin, mode='valid')
    median = np.convolve(median, np.ones(smoothwin)/smoothwin, mode='valid')
    plt.figure(figsize=(15, 10))
    plt.plot(best, label='best')
    plt.plot(median, label='median')
    # use a log scale for the y axis
    plt.yscale('log')
    plt.ylim(0, 10)
    plt.legend()
    plt.show()

def plot_predictions(params, model, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    vm = jax.vmap(model, in_axes=(None, 0, None))
    xg = np.linspace(0, 2, 256)
    X, Y = np.meshgrid(xg, xg)
    X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)
    xy = np.concatenate([X, Y], axis=1)
    yhat = vm(params, xy, jax.random.PRNGKey(0))
    sc = ax.scatter(X, Y, c=yhat, marker='x', cmap='viridis', vmin=0, vmax=1.1)
    if ax is None:
        fig.colorbar(sc, ax=ax)
    plt.show()
    return sc

vm_fitness = jax.jit(jax.vmap(fitness_fn, in_axes=(0, None, None)))
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                           --     CMAES     --
#···············································································
rng = jax.random.PRNGKey(3)
strategy = CMA_ES(popsize=500, num_dims=flat_params.shape[0])
es_params = strategy.default_params.replace(init_min=0, init_max=1)
state = strategy.initialize(rng, es_params)

num_generations = 400
fitnesses = []

for t in tqdm(list(range(num_generations))):
    rng, rng_gen, rng_eval = jax.random.split(rng, 3)
    ps, state = strategy.ask(rng_gen, state, es_params)
    fitness = vm_fitness(ps, X, Y)
    state = strategy.tell(ps, fitness, state, es_params)
    fitnesses.append(fitness)


plot_fitnesses(fitnesses)
best_params = unflatten_params(state.best_member, pdef)
plot_predictions(best_params, model)
best_params

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


import biocomp.evo as ev
import biocomp.utils as bu

best_params, history = ev.optimize_model(model, best_params, X, Y)

plot_fitnesses(history['fitnesses'])
plot_predictions(best_params, model)

