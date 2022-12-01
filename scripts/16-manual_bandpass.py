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

from jax import vmap, jit, grad, value_and_grad
from jax.tree_util import tree_map

random.seed()

lib = ut.load_lib()

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                  --     manual network creation     --
# ···············································································
def P(name):
    return bc.Slot(lib, name)


def any_uorf(lib, *_, **__):
    all_uORFs = lib.pc[lib.pc.category == 'uORF_group'].index.tolist()
    return [all_uORFs]


uorfs = any_uorf(lib)[0]

tus = {
    'Csy4#1': bc.TranscriptionUnit([P('hEF1a'), P('Csy4'),P(uorfs)]),
    'Csy4#2': bc.TranscriptionUnit([P('hEF1a'), P('Csy4'), P(uorfs)]),
    'PgU#1': bc.TranscriptionUnit([P('hEF1a'), P('CasE_rec'), P('PgU'),P(uorfs)]),
    'PgU#2': bc.TranscriptionUnit([P('hEF1a'), P('CasE_rec'), P('PgU'),P(uorfs)]),
    'B_bias': bc.TranscriptionUnit([P('hEF1a'), P('Csy4_rec'), P('PgU'),P(uorfs)]),
    'A_bias': bc.TranscriptionUnit([P('hEF1a'), P('CasE'), P(uorfs)]),
    'out': bc.TranscriptionUnit([P('hEF1a'), P('PgU_rec'), P('NeonGreen'),P(uorfs)]),
}

sources = {tu_name: [tu_name] for tu_name, tu in tus.items()}
aggregations = [['Csy4#1', 'PgU#1'], ['Csy4#2', 'PgU#2'], ['B_bias'], ['A_bias'], ['out']]
n = bc.Network.from_dict(lib, 'v1', tus, sources, aggregations)
inputs = (
    n.compute_graph[n.compute_graph.type == 'aggregation']
    .input_from.apply(lambda x: x[0][0])
    .to_list()
)
n.set_inputs(inputs)

ut.plot_networks([n], ['/Users/jeandisset/Desktop/16-manual_bandpass_uorfs.pdf'])

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                     --     load target data     --
# ···············································································

img = plt.imread('../data/NIHbandpass.png')
target = img[:, :, 0]
target = target[:, ::-1]

N_SAMPLES = 5000
x = np.random.randint(0, target.shape[0], size=(N_SAMPLES, 2))
y = target[x[:, 0], x[:, 1]]
# normalize x
x = x / max(target.shape)

# plot target
plt.figure(figsize=(10, 10))
plt.scatter(x[:, 0], x[:, 1], c=y, s=10)
plt.show()

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


model: bc.ComputeGraphModel = bc.ComputeGraphModel(n)
model.build(
    {
        **bc.nodes.DEFAULT_COMPUTE_NODES_DICT,
        **{
            'translation': partial(bc.nodes.translation_nn, depth=1),
            'sequestron_ERN': partial(bc.nodes.ERN_nn_multi,depth=3),
        },
    }
)

rng = jax.random.PRNGKey(np.random.randint(0, 2**32))
params, constraints = model.init(rng)
# ut.plot_node('translation', params, model)

# find all the extra values for sequestron_ERN in model.network.compute_graph
extra = model.network.compute_graph[model.network.compute_graph.type == 'sequestron_ERN'].extra.to_list()

# [ut.plot_node('sequestron_ERN', params, model, vlim=(-10, 100), n_inputs=2, mode='3d', extra_args=ex) for ex in extra]
ut.plot_node('translation', params, model)

# FREE THE RATIOS

