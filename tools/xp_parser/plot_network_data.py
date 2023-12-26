### {{{                          --     imports     --
import sys

import openpyxl
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple

from biocomp import utils as ut
import json
import biocomp.datautils as du
import biocomp.plotutils as pu
import biocomp.utils as ut
import biocomp.train as train
import biocomp.compute as cmp
import biocomp.parameters as pm
import biocomp as bc
import time
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json5

# pretty print from rich
from rich import print as rprint
import argparse
import json
from pathlib import Path
from rich import print as rprint

import common as cm
from common import (
    DEFAULT_CALIB_PATHS,
    DEFAULT_CALIB_NAMES,
    DEFAULT_XP_PATH,
    DEFAULT_RECIPE_PATH,
    DEFAULT_XP_CACHE_DIR,
    DEFAULT_DATA_CONFIG,
    DEFAULT_DATA_CONFIG_PATH,
)

##────────────────────────────────────────────────────────────────────────────}}}

prog = cm.CLIProgram()

"""
Utils for plotting data in various ways, from the network representation of an experiment.
It can build this network from scratch given a recipe, library and data file, or it can
use a database file to load the network and plot it.
"""

### {{{                --     arg declaration and parsing     --

# for database mode we need a database file
prog.add_argument('--database', type=str, help='path to database file')
# and a network id (or list of ids, or 'all')
prog.add_argument(
    '--network_id', help='network id to plot: int, list of ints, or "all"', default='all'
)

# for recipe mode we need a recipe file and a data file
prog.add_argument('--recipe_file', type=str, help='path to recipe file')
prog.add_argument('--data_file', type=str, help='path to data file')

prog.add_argument('--data_config', type=str, default=DEFAULT_DATA_CONFIG_PATH)

##────────────────────────────────────────────────────────────────────────────}}}

prog.parse_args(['--database', 'devtmp/database.xlsx'])

##

if (
    prog.args.database is not None
    and (prog.args.recipe_file is not None or prog.args.data_file is not None)
) or (prog.args.database is None and prog.args.recipe_file is None and prog.args.data_file is None):
    raise ValueError('You must provide EITHER a database file or a recipe file and a data file')

if prog.args.recipe_file is not None or prog.args.data_file is not None:
    raise NotImplementedError('Direct recipe + data not implemented yet, please provide a database')

if prog.args.database is not None:
    netdf = cm.load_database_table(prog.args.database, 'network')
    xpdf = cm.load_database_table(prog.args.database, 'experiment')

if prog.args.network_id == 'all':
    net_ids = netdf['id'].tolist()
else:
    net_ids = int(prog.args.network_id)

net_ids = 20

def get_recipe_and_data_filepaths(netdf, nid, path_prefix=''):
    if nid not in netdf['id'].tolist():
        raise ValueError(f'Network id {nid} not found in database')
    if len(netdf[netdf['id'] == nid]) > 1:
        raise ValueError(f'Network id {nid} is not unique in database')

    # check data file present
    data_file = netdf[netdf['id'] == nid]['data_file'].tolist()[0]
    if pd.isna(data_file):
        raise ValueError(f'Data file information for network id {nid} is missing')
    data_file = Path(path_prefix) / data_file
    data_file = Path(data_file).resolve()
    if not Path(data_file).exists():
        raise ValueError(f'Data file {data_file} not found')

    # check recipe file present
    recipe_file = netdf[netdf['id'] == nid]['recipe_file'].tolist()[0]
    if pd.isna(recipe_file):
        raise ValueError(f'Recipe file information for network id {nid} is missing')
    recipe_file = Path(path_prefix) / recipe_file
    recipe_file = Path(recipe_file).resolve()
    if not Path(recipe_file).exists():
        raise ValueError(f'Recipe file {recipe_file} not found')

    return recipe_file, data_file



# check data_file that are not empty
net_with_data = netdf[netdf['data_file'].notna()]
net_with_data['id']

##
# I need to be able to load the network as a dictionnary, directly from the database
