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

import more_itertools as mit

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

sources = {tu_name: [tu_name] for tu_name, tu in tus.items()}

networks = []
aggregations = [['Csy4#1', 'PgU#1'], ['Csy4#2', 'PgU#2'], ['B_bias'], ['A_bias'], ['out']]
n = bc.Network.from_dict(lib, 'v1', tus, sources, aggregations)
n2 = n.copy()
n2.name = 'v2'
inputs = (
    n.compute_graph[n.compute_graph.type == 'aggregation']
    .input_from.apply(lambda x: x[0][0])
    .to_list()
)
n.set_inputs(inputs)
networks.append(n)
n2.set_inputs(reversed(inputs))
networks.append(n2)


# aggregations = [[source_name] for source_name in sources.keys()]
# # get all possible ordered pairs of tus
# pairs = list(itertools.permutations(tus.keys(), 2))
# networks = []
# for p in pairs:
# t = tus.copy()
# t[f'{p[0]}_bias'] = t[p[0]]
# t[f'{p[1]}_bias'] = t[p[1]]
# sources = {tu_name: [tu_name] for tu_name, tu in t.items()}
# aggregations = [[source_name] for source_name in sources.keys()]
# name = f'{p[0]}_x_{p[1]}'
# n = bc.Network.from_dict(lib, name, t, sources, aggregations)
# # select the input_from of the source that is to become an input (it should be a numeric node)
# inp0 = n.compute_graph[n.compute_graph.source_id == p[0]].input_from.values[0][0][0]
# inp1 = n.compute_graph[n.compute_graph.source_id == p[1]].input_from.values[0][0][0]
# n.set_inputs([inp0, inp1])
# networks.append(n)


print(f'Generated {len(networks)} networks')

import os

import datetime

dirname = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

os.makedirs(f'../__out/{dirname}', exist_ok=True)

ut.plot_networks(networks, [f'../__out/{dirname}/{n.name}.pdf' for n in networks])
ut.plot_networks(networks)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                     --     load target data     --
# ···············································································

img = plt.imread('../data/bandpass_dec.png')
target = img[:, :, 0]
target = target[:, ::-1]

N_SAMPLES = 2500
x = np.random.randint(0, target.shape[0], size=(N_SAMPLES, 2))
y = target[x[:, 0], x[:, 1]]
y = y[:, None]
# normalize x
x = x / max(target.shape)

# jk, let's try a target of just ones if x[0] < 0.5 and zeros otherwise
# y = (x[:, 0] < 0.5).astype(np.float32)
# add 1 dimension to y:

# plot target
plt.figure(figsize=(10, 10))
plt.scatter(x[:, 0], x[:, 1], c=y, s=10)
plt.show()

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                           --     train     --
# ···············································································


def tree_shape(t):
    return jax.tree_map(lambda x: x.shape, t)


def history_summary(model, history):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # plot loss:

    ax[0].plot(history['loss'], label='train')
    ax[0].set_ylim(0, 2.0)
    ax[0].set_title('Loss')

    best_params, (best_epoch, best_replicate) = bc.train.get_best_params(history)
    print(
        f'Best epoch: {best_epoch}, replicate: {best_replicate}, best loss: {history["loss"][best_epoch][best_replicate]:.3f}'
    )
    bestloss = history['loss'][best_epoch][best_replicate]
    rng_key = jax.random.PRNGKey(0)
    yhat = jax.vmap(model, in_axes=(None, 0, None))(best_params, x, rng_key)

    ax[1].scatter(x[:, 0], x[:, 1], c=y, marker='x', cmap='Blues', vmin=0, vmax=2.0)
    ax[1].set_title('Target')

    ax[2].scatter(x[:, 0], x[:, 1], c=yhat, marker='x', cmap='Blues', vmin=0, vmax=2.0)
    ax[2].set_title('Prediction')

    # plt.show()
    os.makedirs(f'../__out/{dirname}/res', exist_ok=True)
    plt.savefig(f'../__out/{dirname}/res/{model.network.name}_{bestloss:.3f}.png')

    with open(f'../__out/{dirname}/res/{model.network.name}_{bestloss:.3f}.pkl', 'wb') as f:
        pickle.dump(best_params, f)


config = {
    'epochs': 200,
    'n_replicates': 100,
    'learning_rate': 0.001,
    'batch_size': 256,
    'static_params': [],
    'rng_key': np.random.randint(0, 1000000),
}
loggers = {10: bc.train.log_w_replicates}

node_impls = dict(
    bc.nodes.DEFAULT_COMPUTE_NODES_DICT,
    **{
        'sequestron_ERN3p': sequestron_ERN3p,
    },
)

histories = []
for i, net in enumerate(networks[:1]):
    model = bc.ComputeGraphModel(net)
    model.build(node_impl=node_impls)
    # params, constraints = model.init(jax.random.PRNGKey(0))
    print(f'Fitting {net.name} ({i+1}/{len(networks)})')
    # ut.plot_networks([net])
    history = bc.train.train_model(model, x, y, config, loggers)
    final_loss = np.array(history['loss'][-1])
    best_final_loss = final_loss.min()
    if best_final_loss < 0.2:
        print(f'YAAAAAAAAAAAAAAAAYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYAAAAAAAAAYAYAY')
        print(f'Best final loss: {best_final_loss:.3f}')
        print(f'\n\n\n')
    histories.append(history)
    history_summary(model, history)





#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

best_params, (best_epoch, best_replicate) = bc.train.get_best_params(history)
best_params

ut.plot_node('translation', best_params, model)
ut.plot_node('transcription', best_params, model)
ut.plot_node('sequestron_ERN', best_params, model, n_inputs=2)

history['grad'][1]


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     manual training     --
# ···············································································

params, constraints = model.init(jax.random.PRNGKey(np.random.randint(0, 1000000)))

# params['shared'] = {
    # 'hEF1a::tc_rate': 2.0,
    # 'tc_deg': 1.0,
    # 'empty_tc::tl_rate': 2.0,
    # 'tl_deg': 1.0,
# }

params['shared'] = {
    'hEF1a::tc_rate': 2.0,
    'tc_deg': 1.0,
    'empty_tc::tl_rate': 2.0,
    'tl_deg': 1.0,
}

params['node'][29]['value'] = np.array([0.6]) # b_a
params['node'][30]['value'] = np.array([1.5]) # b_b
params['node'][31]['value'] = np.array([0.25]) # b_c


optimizer = optax.sgd(0.001)
optimizer_state = optimizer.init(params)


def loss_fn(params, x, y):
    vm = jax.vmap(model, in_axes=(None, 0, None))
    yhat = vm(params, x, jax.random.PRNGKey(0))
    return jnp.mean((yhat - y) ** 2)

def training_step(params, opt_state, x, y):
    loss, grad = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


print('Training...')
step = jax.jit(training_step)
for i in range(500):
    params, optimizer_state, loss = step(params, optimizer_state, x, y)

print(f'Epoch {i} loss: {loss:.3f}')

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


params, constraints = model.init(jax.random.PRNGKey(np.random.randint(0, 1000000)))

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                       --     manual tuning     --
# ···············································································

n = model.network.copy()

params['shared'] = {
    'hEF1a::tc_rate': 2.0,
    'tc_deg': 1.0,
    'empty_tc::tl_rate': 2.0,
    'tl_deg': 1.0,
}

params['node'][29]['value'] = np.array([0.6]) # b_a
params['node'][30]['value'] = np.array([1.5]) # b_b
params['node'][31]['value'] = np.array([0.25]) # b_c

# _, outputs = model.collect_all_results(params, np.array([2.0, 0.0]), jax.random.PRNGKey(0))
# # output is a dict (node_id -> output)
# n.compute_graph['output_values'] = None
# for node_id, output in outputs.items():
    # outp = output if output.ndim > 0 else [output]
    # n.compute_graph['output_values'][node_id] = np.array(outp)
# ut.plot_networks([n])


vm = jax.vmap(model, in_axes=(None, 0, None))
xg = np.linspace(0, 1, 100)
X, Y = np.meshgrid(xg, xg)
X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)
xy = np.concatenate([X, Y], axis=1)

def plot_predictions(params, xy, ax):
    yhat = vm(params, xy, jax.random.PRNGKey(0))
    sc = ax.scatter(X, Y, c=yhat, marker='x', cmap='viridis', vmin=0, vmax=1.1)
    return sc


fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sc = plot_predictions(params, xy, ax)
fig.colorbar(sc, ax=ax)
plt.show()

yhat = vm(params, x, jax.random.PRNGKey(0))
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(x[:, 0], x[:, 1], c=y, marker='x', cmap='viridis', vmin=0, vmax=1.0)
ax[1].scatter(x[:, 0], x[:, 1], c=yhat, marker='x', cmap='viridis', vmin=0, vmax=1.0)
ax[0].set_title('Target')
ax[1].set_title('Prediction')

plt.show()


loss_fn(params, x, y)


np.array([0.0, 1.0]) - np.array([[0.0], [1.0]])



# # now let's plot predictions for variations of each shared parameter
# param_to_sweep = ['hEF1a::tc_rate', 'empty_tc::tl_rate']
# param_values = np.linspace(0, 3.0, 10)
# # plot a matrix of predictions
# fig, ax = plt.subplots(len(param_values), len(param_values), figsize=(10, 10))
# for i, p1 in enumerate(param_values):
    # for j, p2 in enumerate(param_values):
        # params['shared'][param_to_sweep[0]] = p1
        # params['shared'][param_to_sweep[1]] = p2
        # fig, ax, cbar = plot_predictions(params, xy)
        # ax.set_title(f'{param_to_sweep[0]}: {p1:.2f}, {param_to_sweep[1]}: {p2:.2f}')
        # # write the parameter values on the plot
        # ax.text(0.05, 0.95, f'{param_to_sweep[0]}: {p1:.2f}', transform=ax.transAxes, va='top')
        # ax.text(0.05, 0.05, f'{param_to_sweep[1]}: {p2:.2f}', transform=ax.transAxes, va='bottom')
# plt.show()



#                                                                            }}}

## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     archive     --
# ···············································································
# tus = {
# 'Csy4_rec+CasE': bc.TranscriptionUnit([P('hEF1a'), P('Csy4_rec'), P('CasE'), P(uorfs)]),
# 'Csy4': bc.TranscriptionUnit([P('hEF1a'), P('Csy4'), P(uorfs)]),
# 'CasE_rec+PgU': bc.TranscriptionUnit([P('hEF1a'), P('CasE_rec'), P('PgU'), P(uorfs)]),
# 'PgU+NeonGreen': bc.TranscriptionUnit([P('hEF1a'), P('PgU_rec3p'), P('NeonGreen'), P(uorfs)]),
# }

# tus = {
# 'Csy4_case5#1': bc.TranscriptionUnit([P('hEF1a'), P('CasE_rec'), P('Csy4'), P(uorfs)]),
# 'Csy4_case5#2': bc.TranscriptionUnit([P('hEF1a'), P('CasE_rec'), P('Csy4'), P(uorfs)]),
# 'Csy4_pgu3#1': bc.TranscriptionUnit([P('hEF1a'), P('Csy4'), P('PgU_rec'), P(uorfs)]),
# 'Csy4_pgu3#2': bc.TranscriptionUnit([P('hEF1a'), P('Csy4'), P('PgU_rec'), P(uorfs)]),
# 'CasE#1': bc.TranscriptionUnit([P('hEF1a'), P('CasE'), P(uorfs)]),
# 'CasE#2': bc.TranscriptionUnit([P('hEF1a'), P('CasE'), P(uorfs)]),
# 'PgU#1': bc.TranscriptionUnit([P('hEF1a'), P('PgU'), P(uorfs)]),
# 'PgU#2': bc.TranscriptionUnit([P('hEF1a'), P('PgU'), P(uorfs)]),
# 'out': bc.TranscriptionUnit([P('hEF1a'),P('Csy4_rec'), P('NeonGreen'), P(uorfs)]),
# }


# biases = {
# }

# # and also generate all groupings of 3 groups of tus
# # we want 3 groups: in0, in1, rest
# # rest can contain 0 tus or more, in0 and in1 can contain 1 or more. No duplicates

# groups2 = [A for a, b in mit.set_partitions(tus.keys(), 2) for A in ([a, b, []], [b, a, []])]
# groups3 = [
# A
# for a, b, c in mit.set_partitions(tus.keys(), 3)
# for A in ([a, b, c], [a, c, b], [b, a, c], [b, c, a], [c, a, b], [c, b, a])
# ]
# groups = groups2 + groups3

# # remove groups where any of the first 2 subgroup is != 2
# groups = [g for g in groups if len(g[0]) == 2 and len(g[1]) == 2]
# # remove groups where both first groups are 1
# # groups = [g for g in groups if not (len(g[0]) == 1 and len(g[1]) == 1)]

# def same_source(tu_pair):
# if len(tu_pair) == 2:
# return tu_pair[0].split('#')[0] == tu_pair[1].split('#')[0]
# else:
# return False

# # remove groups where any of the first 2 subgroup contains a pair of tus with the same source
# groups = [g for g in groups if not (same_source(g[0]) or same_source(g[1]))]

# len(groups)

# def priority_to_1(p1, p2):
# # if in pair 1 or pair 2 there is a tu#2, there should be the same tu#1 in the other pair
# tu1inp1 = [tu.split('#')[0] for tu in p1 if '#1' in tu]
# tu2inp1 = [tu.split('#')[0] for tu in p1 if '#2' in tu]

# tu1inp2 = [tu.split('#')[0] for tu in p2 if '#1' in tu]
# tu2inp2 = [tu.split('#')[0] for tu in p2 if '#2' in tu]

# # every tu2 should have a tu1
# return set(tu2inp1).issubset(set(tu1inp1)) and set(tu2inp2).issubset(set(tu1inp2))

# groups = [g for g in groups if priority_to_1(g[0], g[1])]


# # remove groups where
# len(groups)

# networks = []


# for inp_0, inp_1, rest in tqdm(groups):
# t = tus.copy()
# # for i in inp_0:
# # t[f'{i}_bias'] = t[i]
# # for i in inp_1:
# # t[f'{i}_bias'] = t[i]

# # add biases
# t.update(biases)

# sources = {tu_name: [tu_name] for tu_name, tu in t.items()}

# aggregations = [inp_0, inp_1, *[[s] for s in rest]]
# aggregations
# name = f'{"|".join(inp_0)}_{"|".join(inp_1)}_{"|".join(rest)}'

# n = bc.Network.from_dict(lib, name, t, sources, aggregations)
# # all numeric connected to aggregations are inputs
# input_ids = []

# def get_source_node(graph, source_id):
# return graph[graph.source_id == source_id]

# for inp in (inp_0, inp_1):
# if len(inp) == 1:
# # we are looking for a source:
# input_ids.append(get_source_node(n.compute_graph, inp[0]).input_from.values[0][0][0])
# else:
# # we are looking for an aggregation
# ag = n.compute_graph.loc[
# get_source_node(n.compute_graph, inp[0]).input_from.values[0][0][0]
# ]
# # check that we have only one result, and it is an aggregation
# assert ag.type == 'aggregation'

# input_ids.append(ag.input_from[0][0])

# n.set_inputs(input_ids)
# networks.append(n)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
