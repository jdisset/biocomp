## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     imports     -
# ···············································································

import biocomp.datautils as du
import matplotlib.pyplot as plt
from functools import partial
import optax
import biocomp as bc
import biocomp.nodes as bn
import biocomp.compute as bcc
import numpy as np
import scriptutils as ut
import jax
import random
import jax.numpy as jnp

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

def plot_node_results(model, X, all_results):
    node_types = model.network.compute_graph.groupby('type').groups
    for node_type, node_ids in node_types.items():
        for node_id in node_ids:
            plt.plot(X, all_results[node_id], 'o', label=node_id, alpha=0.5, markersize=2)
        plt.legend()
        plt.title(node_type)
        plt.show()
lib = ut.load_lib()


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                --     learning identity single tu     --
#···············································································

recipe = {
    'name': "justbfp",
    'content': [
        {'sources': [{'plasmid': "pAK0022"}]},
    ],
}
n = bc.recipe.network_from_recipe(recipe, lib)

inv_n = bc.network.inverted_network(n)
inv_n.compute_graph.loc[5].extra
bc.network.fuse_consecutive(
    inv_n.compute_graph, ("inv_translation", "inv_transcription"), "inv_fused"
)


def inv_fused_nn(get_param, get_quantized, wsize=64, depth=2, **_):
    def apply(value, rng_key):
        k0, k1, k2 = jax.random.split(rng_key, 3)
        return value
    return apply


rng_key = jax.random.PRNGKey(0)

node_impl = dict(
    bc.nodes.DEFAULT_COMPUTE_NODES_DICT,
    **{
        'inv_fused': partial(inv_fused_nn, wsize=64, depth=2),
        'output': partial(bn.output_nn, wsize=64, depth=2),
        'transcription': partial(bn.transcription_nn, wsize=64, depth=2),
        'translation': partial(bn.translation_nn, wsize=64, depth=2),
    },
)



model = bc.ComputeGraphModel(inv_n)
model.build(node_impl=config['node_impl'])

params, constraints = model.init(rng_key)

Y = np.random.rand(1000, 1) * 5.0
X = np.exp(Y)
models = [model]

xbatches, ybatches = du.make_batches_uniform_sampling([Y], 1000, rng_key, models)

loggers = {50: bc.train.console_log}

config = {
    'epochs': 500,
    'node_impl': node_impl,
}

ph, lh, gh, oh = bc.train.train_models(models, xbatches, ybatches, config, loggers=loggers)


yhat = jax.vmap(partial(model, ph[-1], rng_key=rng_key))(X)
plt.plot(X, yhat, 'o')
plt.show()

# now we want to plot the function computed by each node
_, all_results = jax.vmap(partial(model.collect_all_results, p, rng_key=rng_key))(X)
# all_results is a dictionary with keys the node ids and values the results of the
# node for each input



#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

xp = ut.load_xp('20221012A_massCtrls', lib)
models_d = xp.get_models(node_impl=config['node_impl'])
_, Yd = xp.get_XY(models_d)
Xd, Yd = bc.train.prep_data(models_d, Yd)

NMODELS = 1
models = list(models_d.values())[:NMODELS]
Y = list(Yd.values())[:NMODELS]
X = list(Xd.values())[:NMODELS]

for m in models:
    bc.network.fuse_consecutive(
        m.network.compute_graph, ("inv_translation", "inv_transcription"), "inv_fused"
    )
    m.build(node_impl=config['node_impl'])

rng_key = jax.random.PRNGKey(np.random.randint(0, 2**32))
xbatches, ybatches = du.make_batches_uniform_sampling(Y, 100, rng_key, models)

node_impl = dict(
    bc.nodes.DEFAULT_COMPUTE_NODES_DICT,
    **{
        'inv_fused': partial(inv_fused_nn, wsize=64, depth=2),
        'output': partial(bn.output_nn, wsize=64, depth=2),
        'transcription': partial(bn.transcription_nn, wsize=64, depth=2),
        'translation': partial(bn.translation_nn, wsize=64, depth=2),
    },
)
cfg = {
    'epochs': 100,
}
ph, lh, gh, oh = bc.train.train_models(models, xbatches, ybatches, cfg, loggers=loggers)

gh[1]
ph[0]
oh[1]

p = ph[-1]

model = models[0]
_, all_results = jax.vmap(partial(model.collect_all_results, p, rng_key=rng_key))(X[0])

plot_node_results(model, X[0], all_results)


# model.network.compute_graph
# ut.plot_networks([model.network], ['/Users/jeandisset/Desktop/case_out.pdf'] )
