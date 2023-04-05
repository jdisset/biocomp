### {{{                          --     imports     --
import biocomp as bc
from biocomp import datautils as du
from jax.scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
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
matrix_xp = ut.load_xp('2023-03-26_MatrixCsy4', lib, data_path='./data/calibrated_data')
dman = du.DataManager.from_xps([matrix_xp], config, inverse='all')
names = [m.node_namespace for m in dman.get_models()]

ut.plot_networks([m.network for m in dman.get_models()[:2]])

### {{{                      --     quantify uorfs     --
models = dman.get_models()


def get_uorf_value(param):
    if 'tl_rate' in param:
        u = param['tl_rate'][0].split('_')[0]
        try:
            v = int(u[:-1]) * 10
        except ValueError:
            v = 0
        if u[-1] == 'w':
            v = v - 5
        return v
    else:
        return 0


def get_uorf_values(network):
    cdg = network.central_dogma_graph
    ERN_inputs = network.compute_graph[network.compute_graph['type'] == 'sequestron_ERN'][
        'cdg_input'
    ].values[0]
    cdgin = cdg.loc[ERN_inputs]
    ern_side = cdg.loc[cdgin.iloc[0].predecessor[0]]
    recog_side = cdgin.iloc[1]
    values = (get_uorf_value(ern_side.params), get_uorf_value(recog_side.params))
    return values


uorf_dict = {}
for i, m in enumerate(models):
    has_ERN_node = m.network.compute_graph['type'] == 'sequestron_ERN'
    if has_ERN_node.any():
        uorf_dict[get_uorf_values(m.network)] = i


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                        --     plot matrix     --

n_unique_ern_side = len(np.unique([v[0] for v in uorf_dict.keys()]))
n_unique_recog_side = len(np.unique([v[1] for v in uorf_dict.keys()]))

ER_to_rowcol = {
    (E, R): (r, c)
    for c, E in enumerate(np.unique([v[0] for v in uorf_dict.keys()]))
    for r, R in enumerate(np.unique([v[1] for v in uorf_dict.keys()]))
}
ER_to_rowcol

fig, axes = du.mkfig(n_unique_ern_side, n_unique_recog_side, (2, 2))
for (E, R), m in tqdm(list(uorf_dict.items())[:]):
    print(E, R, m)
    r, c = ER_to_rowcol[(E, R)]
    ax = axes[r, c]
    title = ''
    contours = np.linspace(0, 1, 7)
    du.model_plot(
        dman,
        m,
        ax=ax,
        radius=0.15,
        knn=4000,
        min_points=20,
        colorbar=False,
        title=title,
        contours=contours,
        res=100,
    )
    # remove left ticks except if j == 0
    if r !=  n_unique_recog_side - 1:
        ax.set_xticks([])
        ax.set_xlabel('')
    if c != 0:
        ax.set_yticks([])
        ax.set_ylabel('')

    # add E and R labels for the whole grid
    if r == 0:  # first row
        ax.text(0.5, 1.1, f'E: {E/10:.1f}x', transform=ax.transAxes, ha='center', va='bottom')

    # last column, write to the right
    if c ==  n_unique_ern_side - 1:
        ax.text(
            1.1, 0.5, f'R: {R/10:.1f}x', transform=ax.transAxes, ha='left', va='center', rotation=90
        )

fig.tight_layout()
savepath = Path('~/Desktop/matrixdata_csy4').expanduser()
savepath.mkdir(parents=True, exist_ok=True)
fig.savefig(savepath / 'csy4_matrix_data.pdf')
print('done')

##────────────────────────────────────────────────────────────────────────────}}})
