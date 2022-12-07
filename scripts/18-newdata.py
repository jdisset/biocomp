## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     imports     --
# ···············································································
import biocomp as bc
import pandas as pd
import biocomp.compute as bcc
import numpy as np
from functools import partial
import time
import biocomp.utils as bu
import scriptutils as ut
import jax
from jax import jit, vmap, grad, value_and_grad
import jax.numpy as jnp
import random
import biocomp.datautils as du
import optax
from tqdm import tqdm
import json5

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

lib = ut.load_lib()
xp = ut.load_xp('E20221124A_ERNbandpassV2', lib)
# xp = ut.load_xp('20221012A_massCtrls', lib)

xp.networks
nets = list(xp.networks.values())

n = xp.networks['BY+RY+iY']
models = xp.get_models(node_impl=bc.nodes.DEFAULT_COMPUTE_NODES_DICT)
#print(len(xp.networks))
ut.plot_networks([n])
ut.plot_networks(nets)
n.central_dogma_graph
n.compute_graph
n.tu_in_sources


##
uidGen = bu.uniqueIdGenerator()
cg = n._Network__buildRawGraph(uidGen)
n._Network__removeShortcuts(cg, 0)
n.compute_graph = pd.DataFrame([n.toDict() for n in cg]).set_index('id').sort_index()

cdf = n.compute_graph

sources_tuids = n.central_dogma_graph.loc[
    cdf[cdf.type == 'source'].cdg_output
].tu_id.apply(lambda x: x[0])

sources_tuids

tmpdf = pd.DataFrame(
    {'compute_id': cdf[cdf.type == 'source'].index, 'tuid': sources_tuids}
).set_index('compute_id')

tmpdf

n.transcription_units

n.aggregations
n.tu_in_sources


c = n.db.cursor()
# get the transcription units
c.execute(
    """SELECT TU FROM TU_in_source tis, source_in_aggregation sia, aggregations a
   WHERE tis.source = sia.source AND sia.aggregation = a.id AND a.recipe = ?""",
    (n.name,),
)

tus = {
    tu[0]: bc.network.transcription_unit_from_L1(tu[0], lib) for tu in c.fetchall()
}

tus


# problem_recipe = [ut.DEFAULT_RECIPE_PATH / '108+100+102i.recipe.json5']
# recipes = [json5.load(open(r)) for r in tqdm(problem_recipe)]
# base_conn = sqlite3.connect(':memory:')
# bc.recipe.recipes_to_sql(recipes, base_conn, lib)
# n = bc.Network(lib, '108+100+102i', base_conn)
# print('done loading')


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                           --     init     --
# ···············································································

random.seed()
cfg = {
    "node_remap": {},
    "optimizer": "sgd",
    "learning_rate": 0.001,
    "adam_w_decay": 0.0001,
    "loss_function": "mse",
    "rng_key": 42,
    "epochs": 10000,
    "n_replicates": 1,
    "compile_training": True,
    "n_batches": 32,
    "norm_factor": 1e6,
    "balance_bin_resolution": 0.5,
    "balance_threshold_quantile": 0.4,
    "balance_threshold_min": 40,
    "log_rate": 1,
    "plot_rate": 100,
    "node_remap": {
        "sequestron_ERN": "ERN_with_affinity",
        "transcription": "transcription_nn",
        "inv_transcription": "inverse_transcription_nn",
        "translation": "translation_nn",
        "inv_translation": "inverse_translation_nn",
    },
    "save_rate": 100,
}
optimizer = optax.sgd(learning_rate=cfg['learning_rate'])

key = jax.random.PRNGKey(cfg['rng_key'])
ikeys = jax.random.split(key, len(models))

params = {}
constraints = {}

for s, m, k in zip(models.keys(), models.values(), ikeys):
    params, constraints = m.init(k, pre_params=params, pre_constraints=constraints)

opt_state = optimizer.init(params)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

# TODO:
# why do I merge L1s into L2s? ex(73+73i+100R)



## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     create network     --
#···············································································
def P(name):
    return bc.Slot(lib, name)

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
sources = {tu_name: [tu_name] for tu_name, tu in tus.items()}
n = bc.Network.from_dict(lib, 'v1', tus, sources, aggregations)
n.compute_graph
n.central_dogma_graph

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


