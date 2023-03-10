### {{{                          --     imports     --
import biocomp as bc
from biocomp import datautils as du
from jax.scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from biocomp.calibration import Calibration
import scriptutils as ut
from pathlib import Path
import json5
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import gaussian_kde
import jax
import optax
from jax import jit, vmap, value_and_grad
from jax.tree_util import Partial as partial
from tqdm import tqdm
import biocomp.defaults as bdf
import pandas as pd
import copy

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

config = {
    **bdf.DEFAULT_CONFIG,
    **{
        'node_impl': node_impl,
        'rng_key': np.random.randint(0, 2**32),
        "batch_size": 16,
        "n_batches": 4,
        "epochs": 12,
    },
}

##────────────────────────────────────────────────────────────────────────────}}}

lib = ut.load_lib()
lib.disable_all_sequestrons()
lib.set_enabled_sequestrons(['ERN'])
cascade_xp = ut.load_xp('2023-03-03_CascadesV1', lib, data_path='./data/calibrated_data')
dman = du.DataManager.from_xps([cascade_xp], config, inverse='all')
names = [m.node_namespace for m in dman.get_models()]

names


##
ut.plot_networks([m.network for m in dman.get_models()])

##
fig, axes = du.mkfig(3,1)
axes = axes.flatten()
du.model_plot(dman, 0, axes[0])
du.model_plot(dman, 1, axes[1])
du.model_plot(dman, 2, axes[2])

X = dman.get_raw_X()
xcase = X[0]
xcase.shape
xcase.max()

m = dman.get_models()[0]
pnames = m.get_inverted_input_proteins()
du.fluo_densities(X[0], pnames)

ngselected = xcase[:,0] > 10000
xcase[ngselected].shape

##
fig, axes = du.mkfig(1,1)
du.model_plot(dman, 7, axes)

##

m7 = dman.get_models()[7]

m7.get_output_proteins()
m7.get_inverted_input_proteins()

##
names
start_id = 3
end_id = 9
# nm = len(dman.get_models())
fig, axes = du.mkfig(end_id - start_id,4, (4,4))
for i in range(start_id, end_id):
    print(f'plotting {i}')
    # get 4 axes of this row:
    axes_row = axes[i - start_id]
    du.model_plot(dman, i, ax=None,  axes = axes[i - start_id], min_points=5, radius=0.2)

fig.tight_layout()
axes_row = axes[i - start_id]
# save to desktop
fname = Path('~/Desktop/2021-03-03_cascade_plots.png').expanduser()
fig.savefig(fname, dpi=300)

##

r = dman.get_models()[20]

r.network.compute_graph
r.network.central_dogma_graph

# L1.ST2-3_mNG_6
# L1_PhiC31NLS_V2_6
lib.seqs

# When building compute graph for recipe RecombCascadeCsy4HL, found 6 DNA sources in the graph, but 7 DNA nodes total.
# Extra DNA nodes: ['L1_attR_invNG_attL_10']
