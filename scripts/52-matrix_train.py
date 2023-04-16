#### {{{                          --     imports     --
from biocomp import utils as ut
from contextlib import contextmanager
import scriptutils as su
from pathlib import Path
import jax.numpy as jnp
import numpy as np
import jax
from jax import jit, vmap, value_and_grad
from jax.tree_util import Partial as partial
from tqdm import tqdm
import biocomp.defaults as bdf
import pandas as pd
from rich import print as pprint
import biocomp.datautils as du
import biocomp.utils as bu
import json
import biocomp.train as train
from biocomp.compute import ComputeConfigManager

##────────────────────────────────────────────────────────────────────────────}}}

# xpname = '2023-02-16_Matrix'
# xpname = '2023-03-26_MatrixCsy4'


### {{{                        --     parse args     --
prog = train.TrainingProgram()
prog.add_argument('--xpname', type=str, default='2023-02-16_Matrix', help='name of experiment to load')
prog.add_argument('--training_set', type=str, default='1_corner', help='name of training set to use')
prog.parse_args()
##────────────────────────────────────────────────────────────────────────────}}}

### {{{                      --     loading matrix xp     --
with ut.timer('Loading data and building networks'):
    lib = su.load_lib()
    matrix_xp = su.load_xp(prog.xpname, lib, data_path='./data/calibrated_data')
    dman_full = du.DataManager.from_xps([matrix_xp], prog.training_config, inverse='all')
##────────────────────────────────────────────────────────────────────────────}}}

### {{{                      --     quantify uorfs     --

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


def get_max_uorf(network):
    cdg = network.central_dogma_graph
    params = cdg.params.values
    uorfs = [get_uorf_value(p) for p in params]
    return max(uorfs)


uorf_dict = {}
for i, n in enumerate(dman_full.get_networks()):
    has_ERN_node = n.compute_graph['type'] == 'sequestron_ERN'
    if has_ERN_node.any():
        uorf_dict[get_uorf_values(n)] = i
    # else:
    # uorf_dict[(get_max_uorf(n),)] = i

uorf_dict
single_uorfs = [i for i in range(len(dman_full.get_networks())) if i not in uorf_dict.values()]

# single_names = [n.name for i, n in enumerate(dman_full.get_networks()) if i in single_uorfs]



##────────────────────────────────────────────────────────────────────────────}}}

MAX_UORF = 80
TRAINING_SETS = {
    '1_corner': [(0, 0)],
    '2_corners_recog': [(0, 0), (0, MAX_UORF)],
    '2_corners_ern': [(0, 0), (MAX_UORF, 0)],
    '2_corners_diag': [(0, 0), (MAX_UORF, MAX_UORF)],
    '3_corners': [(0, 0), (0, MAX_UORF), (MAX_UORF, 0)],
    '4_corners': [(0, 0), (0, MAX_UORF), (MAX_UORF, 0), (MAX_UORF, MAX_UORF)],
    'all': list(uorf_dict.keys()),
}
TRAINING_SETS.keys()

subset = single_uorfs + [uorf_dict[i] for i in TRAINING_SETS[prog.training_set]]

dman = dman_full.make_subset(subset)

if prog.wandb_project is not None:
    loggers = train.setup_wandb_logging(prog.wandb_project, dman, prog.training_config, prog.compute_config)
else:
    loggers = [
        (1, train.console_log),
        (100, partial(train.local_save, save_dir=prog.local_save_dir)),
    ]

train.start(dman, prog.training_config, prog.compute_config, loggers, seed=prog.seed)

