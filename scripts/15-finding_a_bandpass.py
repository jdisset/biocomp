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
import jax.numpy as jnp

import more_itertools as mit

random.seed()

# TODO
# [ ] generate a graph with a ternary ERN
# [x] OR, FOR NOW, just use 2 rec_5' chained together. We'll add the 3' later
# [ ] generate a simple graph to get initial params of default nodes, and put values that seem reasonable
# [ ] generate multiple networks that could be interesting (all input permutations)
# [ ] generate bandpass data
# [ ] optimize copy numbers to find bandpass


lib = ut.load_lib()


def sequestron_ERN3p(get_param, get_quantized, **_):
    def apply(rna, ern, **_):
        return rna * (1.0 - jnp.exp(-ern))

    return apply


def any_uorf(lib, *_, **__):
    all_uORFs = lib.pc[lib.pc.category == 'uORF_group'].index.tolist()
    return [all_uORFs]


def P(name):
    return bc.Slot(lib, name)


# 'Csy4+uOrfs': bc.TranscriptionUnit([P('hEF1a'), P('Csy4'), P(any_uorf(lib)[0])]),

uorfs = any_uorf(lib)[0]
tus = {
    'Csy4_rec+CasE': bc.TranscriptionUnit([P('hEF1a'), P('Csy4_rec'), P('CasE'), P(uorfs)]),
    'Csy4': bc.TranscriptionUnit([P('hEF1a'), P('Csy4'), P(uorfs)]),
    'CasE_rec+PgU': bc.TranscriptionUnit([P('hEF1a'), P('CasE_rec'), P('PgU'), P(uorfs)]),
    'PgU+NeonGreen': bc.TranscriptionUnit([P('hEF1a'), P('PgU_rec3p'), P('NeonGreen'), P(uorfs)]),
}
# sources = {tu_name: [tu_name] for tu_name, tu in tus.items()}
# aggregations = [[source_name] for source_name in sources.keys()]
# n = bc.Network.from_dict(lib, 'name', tus, sources, aggregations)
# ut.plot_networks([n])

biases = {
    'CasE_bias': bc.TranscriptionUnit([P('hEF1a'), P('CasE'), P(uorfs)]),
    'PgU_bias': bc.TranscriptionUnit([P('hEF1a'), P('PgU'), P(uorfs)]),
    'NeonGreen_bias': bc.TranscriptionUnit([P('hEF1a'), P('NeonGreen'), P(uorfs)]),
}

# and also generate all groupings of 3 groups of tus
# we want 3 groups: in0, in1, rest
# rest can contain 0 tus or more, in0 and in1 can contain 1 or more. No duplicates

groups2 = [A for a, b in mit.set_partitions(tus.keys(), 2) for A in ([a, b, []], [b, a, []])]
groups3 = [
    A
    for a, b, c in mit.set_partitions(tus.keys(), 3)
    for A in ([a, b, c], [a, c, b], [b, a, c], [b, c, a], [c, a, b], [c, b, a])
]
groups = groups2 + groups3

networks = []


inp_0, inp_1, rest = groups[30]
for inp_0, inp_1, rest in tqdm(groups):
    t = tus.copy()
    for i in inp_0:
        t[f'{i}_bias'] = t[i]
    for i in inp_1:
        t[f'{i}_bias'] = t[i]

    # add biases
    t.update(biases)

    sources = {tu_name: [tu_name] for tu_name, tu in t.items()}

    aggregations = [inp_0, inp_1, *[[s] for s in rest]]
    aggregations
    name = f'{"|".join(inp_0)}_{"|".join(inp_1)}_{"|".join(rest)}'

    n = bc.Network.from_dict(lib, name, t, sources, aggregations)
    # all numeric connected to aggregations are inputs
    input_ids = []

    def get_source_node(graph, source_id):
        return graph[graph.source_id == source_id]

    for inp in (inp_0, inp_1):
        if len(inp) == 1:
            # we are looking for a source:
            input_ids.append(get_source_node(n.compute_graph, inp[0]).input_from.values[0][0][0])
        else:
            # we are looking for an aggregation
            ag = n.compute_graph.loc[
                get_source_node(n.compute_graph, inp[0]).input_from.values[0][0][0]
            ]
            # check that we have only one result, and it is an aggregation
            assert ag.type == 'aggregation'

            input_ids.append(ag.input_from[0][0])

    n.set_inputs(input_ids)
    networks.append(n)


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

# ut.plot_networks(networks, [f'../__out/{n.name}.pdf' for n in networks])

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                     --     load target data     --
# ···············································································

img = plt.imread('../data/bandpass_dec.png')
target = img[:, :, 0]

N_SAMPLES = 5000
x = np.random.randint(0, target.shape[0], size=(N_SAMPLES, 2))
y = target[x[:, 0], x[:, 1]]

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

import pickle


def history_summary(model, history):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # plot loss:
    ax[0].plot(history['loss'], label='train')
    ax[0].set_title('Loss')
    ax[0].legend()

    best_params, (best_epoch, best_replicate) = bc.train.get_best_params(history)
    print(
        f'Best epoch: {best_epoch}, replicate: {best_replicate}, best loss: {history["loss"][best_epoch][best_replicate]:.3f}'
    )
    bestloss = history['loss'][best_epoch][best_replicate]
    rng_key = jax.random.PRNGKey(0)
    yhat = jax.vmap(model, in_axes=(None, 0, None))(best_params, x, rng_key)
    # then plot predicted on the fig
    ax[1].scatter(x[:, 0], x[:, 1], c=yhat, marker='x', cmap='Blues')
    ax[1].set_title('Predicted')
    # save fig
    plt.savefig(f'../__out/{model.network.name}_{bestloss:.3f}.pdf')
    plt.show()

    # save params
    with open(f'../__out/{model.network.name}_{bestloss:.3f}.pkl', 'wb') as f:
        pickle.dump(best_params, f)

    # plot target scatter from x, y with crosses instead of circles
    plt.scatter(x[:, 0], x[:, 1], c=y, marker='x', cmap='Blues')
    plt.show()


config = {
    'epochs': 500,
    'n_replicates': 10,
    'learning_rate': 0.001,
    'batch_size': 128,
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
    histories.append(history)
    history_summary(model, history)
