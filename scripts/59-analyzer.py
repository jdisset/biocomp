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
from biocomp.nodes import translation
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

node_type = 'translation'
input_shapes = [(1,)]
n_outputs = 1
rng_key = jax.random.PRNGKey(0)
compute_config = cmp.DEFAULT_COMPUTE_CONFIG

impl = compute_config.get_impl(node_type)(
    input_shapes=input_shapes, n_outputs=n_outputs, stack=None
)
f_prepare, f_apply, f_out_shapes = impl

node = cmp.VirtualNode()
node.type_signature = node_type
node.node_id = 0

params = {}
f_prepare(params, [node], rng_key)

NEVAL = 10
x = np.linspace(0, 1, NEVAL)
z = np.ones((NEVAL,1)) * 0.5

@vmap
def apply (x, z):
    return f_apply(x, quantiles=z, params=params, node_id=node.node_id, key=rng_key)

y = apply(x,z)

compute_config.export('/tmp/comp_config.json')

loaded = cmp.ComputeConfigManager.from_file('/tmp/comp_config.json')

loaded_impl = loaded.get_impl(node_type)(
    input_shapes=input_shapes, n_outputs=n_outputs, stack=None
)

loaded.get_impl(node_type)

ut.get_git_commit_hash()
ut.get_biocomp_version()

##

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




##────────────────────────────────────────────────────────────────────────────}}}

### {{{       --     fitness & add the intensity of bias as extra parameter     --
# generate the compute stack
compute_stack = cmp.ComputeStack([NETWORK])


with ut.timer('stack build'):
    compute_stack.build(compute_config)
with ut.timer('param init'):
    cstack_params = compute_stack.init(rng)

type(ut.params_to_numpy(cstack_params)['shared']['tc_inner_0_w'])
type(ut.params_to_jax(cstack_params)['shared']['tc_inner_0_w'])


##────────────────────────────────────────────────────────────────────────────}}}



