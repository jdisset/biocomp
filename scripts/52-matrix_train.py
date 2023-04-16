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

MAX_UORF = 80
TRAINING_SETS = {
    '1_corner': [(0, 0)],
    '2_corners_recog': [(0, 0), (0, MAX_UORF)],
    '2_corners_ern': [(0, 0), (MAX_UORF, 0)],
    '2_corners_diag': [(0, 0), (MAX_UORF, MAX_UORF)],
    '3_corners': [(0, 0), (0, MAX_UORF), (MAX_UORF, 0)],
    '4_corners': [(0, 0), (0, MAX_UORF), (MAX_UORF, 0), (MAX_UORF, MAX_UORF)],
}

training_config = bdf.DEFAULT_TRAINING_CONFIG
compute_config = bdf.DEFAULT_COMPUTE_CONFIG

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--xpname', type=str, default='2023-02-16_Matrix', help='name of experiment to load')
parser.add_argument('--wandb_project', type=str, default=None, help='name of wandb project')
parser.add_argument('--training_set', type=str, default='1_corner', help='name of training set to use')
parser.add_argument('--compute_config', type=str, default=None, help='path to compute config')
parser.add_argument('--train_config', type=str, default=None, help='path to training config')
parser.add_argument('--local_save_dir', type=str, default='./results', help='path to save results')
args = parser.parse_args()

xpname = args.xpname
wandb_project = args.wandb_project
training_set = args.training_set
local_save_dir = Path(args.local_save_dir)

if args.compute_config is not None:
    if not Path(args.compute_config).is_file():
        raise ValueError(f'{args.compute_config} is not a file')
    compute_config = ComputeConfigManager.from_file(args.compute_config)

if args.train_config is not None:
    if not Path(args.train_config).is_file():
        raise ValueError(f'{args.train_config} is not a file')
    training_config = json.load(open(args.train_config))




### {{{                      --     loading matrix xp     --
with ut.timer('Loading data and building networks'):
    lib = su.load_lib()
    matrix_xp = su.load_xp(xpname, lib, data_path='./data/calibrated_data')
    dman_full = du.DataManager.from_xps([matrix_xp], training_config, inverse='all')
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


##────────────────────────────────────────────────────────────────────────────}}}

subset = single_uorfs + [uorf_dict[i] for i in TRAINING_SETS[training_set]]

dman = dman_full.make_subset(subset)

if wandb_project is not None:
    loggers = train.setup_wandb_logging(wandb_project, dman, training_config, compute_config)
else:
    loggers = [
        (1, train.console_log),
        (100, partial(train.local_save, save_dir=local_save_dir)),
    ]

train.start(dman, training_config, compute_config, loggers)
##

a = jnp.arange(10)


def get_slice(a, start, end):
    offset = start // a.shape[0]
    start = start % a.shape[0]
    end = end - offset * a.shape[0]
    if end > a.shape[0]:
        return jnp.concatenate([a[start:], get_slice(a, 0, end - a.shape[0])])
    else:
        return a[start:end]

get_slice(a, 0, 10)
get_slice(a, 0, 14)
get_slice(a, 10, 22)
