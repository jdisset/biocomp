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
        "n_batches": 2048,
        "epochs": 100,
    },
}

##────────────────────────────────────────────────────────────────────────────}}}

lib = ut.load_lib()
matrix_xp = ut.load_xp('2023-02-16_Matrix', lib, data_path='./data/calibrated_data')
dman_full = du.DataManager.from_xps([matrix_xp], config, inverse='all')
names = [m.node_namespace for m in dman_full.get_models()]
names

### {{{                      --     quantify uorfs     --
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

uorf_dict = {
    (e, r): i
    for i, (e, r) in enumerate(zip(case_uorf_numbers, caseR_uorf_numbers))
    if e != None and r != None
}

single_uorfs = [i for i, n in enumerate(names) if 'inert' in n.lower()]
##────────────────────────────────────────────────────────────────────────────}}}

unique_vals
uorf_max = 80
corners_tl_tr_bl_br = [
    uorf_dict[(e, r)] for e, r in [(0, 0), (0, uorf_max), (uorf_max, 0), (uorf_max, uorf_max)]
]
top_row = [uorf_dict[(e, 0)] for e in unique_vals]
bottom_row = [uorf_dict[(e, uorf_max)] for e in unique_vals]
left_col = [uorf_dict[(0, r)] for r in unique_vals]
right_col = [uorf_dict[(uorf_max, r)] for r in unique_vals]

# get training set from cli argument. Could be: "tl", "3corner", "4corner", "extreme_row_col", "1st_row_col"
import sys
mode = sys.argv[1]
training_set = single_uorfs
if mode == 'tl':
    training_set += corners_tl_tr_bl_br[:1]
elif mode == '3corner':
    training_set += corners_tl_tr_bl_br[:3]
elif mode == '4corner':
    training_set += corners_tl_tr_bl_br
elif mode == 'extreme_row_col':
    training_set += top_row + bottom_row + left_col + right_col
elif mode == '1st_row_col':
    training_set += top_row + left_col


print(f'mode: {mode}, training set: {training_set}')

dman_full.set_subset(training_set)
loggers = bc.train.setup_wandb_logging('matrix_train_v1', dman_full, config)
bc.train.start(dman_full, config, loggers)
print('done')
