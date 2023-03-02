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
matrix_xp = ut.load_xp('2023-02-16_Matrix', lib, data_path='./data/calibrated_data')
dman_full = du.DataManager.from_xps([matrix_xp], config, inverse='all')
names = [m.node_namespace for m in dman_full.get_models()]


### {{{                        --     plot matrix     --

names = [m.node_namespace for m in dman_full.get_models()]
# isolate the CasE uorf number. it follows the pattern ... CasE{N[x or w]} ...
import re

case_uorf_numbers = []
caseR_uorf_numbers = []
for n in names:
    s = re.search(r'CasE\d+[xw]', n)
    s_r = re.search(r'CasER\d+[xw]', n)
    if s:
        # isolate the number, multiply by 10, subtract 5 if w
        num = int(s.group(0)[4:-1]) * 10
        if s.group(0)[-1] == 'w':
            num = num - 5
        case_uorf_numbers.append(num)
        if s_r:
            rnum = int(s_r.group(0)[5:-1]) * 10
            if s_r.group(0)[-1] == 'w':
                rnum = rnum - 5
        else:
            rnum = 0
        caseR_uorf_numbers.append(rnum)
    else:
        case_uorf_numbers.append(None)
        caseR_uorf_numbers.append(None)
case_uorf_numbers = np.array(case_uorf_numbers)
caseR_uorf_numbers = np.array(caseR_uorf_numbers)
# remove last 2 values
case_uorf_numbers = case_uorf_numbers[:-2]
caseR_uorf_numbers = caseR_uorf_numbers[:-2]
unique_vals = np.sort(np.unique(case_uorf_numbers[case_uorf_numbers != None]))

N = len(unique_vals)

fig, axes = du.mkfig(N, N, (2, 2))
for m, (E, R) in tqdm(enumerate(list(zip(case_uorf_numbers, caseR_uorf_numbers))[:])):
    print(E, R, m)
    if E is None or R is None:
        continue
    E = int(E)
    R = int(R)
    i = np.where(unique_vals == E)[0][0]
    j = np.where(unique_vals == R)[0][0]
    ax = axes[j, i]
    title = ''
    contours = np.linspace(0, 1, 7)
    du.model_plot(
        dman_full,
        m,
        ax=ax,
        radius=0.2,
        knn=500,
        min_points=5,
        colorbar=False,
        res=100,
        title=title,
        contours=contours,
    )
    # remove left ticks except if j == 0
    if j != N - 1:
        ax.set_xticks([])
        ax.set_xlabel('')
    if i != 0:
        ax.set_yticks([])
        ax.set_ylabel('')

    # add E and R labels for the whole grid
    if j == 0:  # first row
        ax.text(0.5, 1.1, f'E: {E/10:.1f}x', transform=ax.transAxes, ha='center', va='bottom')
    if i == N - 1:  # last column, write to the right
        ax.text(
            1.1, 0.5, f'R: {R/10:.1f}x', transform=ax.transAxes, ha='left', va='center', rotation=90
        )


fig.tight_layout()
fig.savefig(Path('~/Desktop/matrix_data_smooth.pdf').expanduser())
print('done')
##────────────────────────────────────────────────────────────────────────────}}})

training_set = [0] + [i for i, n in enumerate(names) if 'inert' in n.lower()]

for i in training_set:
    print(i, names[i])
    dman_full.set_subset([i])
    # loggers = bc.train.setup_wandb_logging('matrix_train_v0', dman_full, config)
    bc.train.start(dman_full, config)
    print('done')

##
dman_full.set_subset(training_set)
bc.train.start(dman_full, config)
print('done for all')



##

m = dman_full.get_models()[9]
# ut.plot_networks([m.network])
m.network.compute_graph
# --- init
# params
params, constraints = {}, {}
key = jax.random.PRNGKey(0)
params, constraints = m.init(key, pre_params=params, pre_constraints=constraints)
##
dman_full.set_subset(np.arange(0, len(names)))
du.model_fluo_distributions(dman_full, 3)

dman_full.get_batches(key)

du.fluo_scatter(
