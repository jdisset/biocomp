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


##────────────────────────────────────────────────────────────────────────────}}}
prog = cm.CLIProgram()
### {{{                --     arg declaration and parsing     --

prog.add_argument('--database', type=str, required=True)
prog.add_argument('--mode', type=str, default='update_from_filesystem')
prog.add_argument('--create', action='store_true', default=False) # create database if it doesn't exist

DEFAULT_XP_PATH = ut.DEFAULT_XP_PATH
prog.add_argument('--xp_path', type=str, default=DEFAULT_XP_PATH)

DEFAULT_RECIPE_PATH = ut.DEFAULT_RECIPE_PATH
prog.add_argument('--recipe_paths', type=str, nargs='+', default=DEFAULT_RECIPE_PATH)

prog.parse_args(['--database', 'devtmp/database.xlsx', '--create'])
##────────────────────────────────────────────────────────────────────────────}}}
### {{{                    --     arg postprocessing     --
# get the database path
database_path = Path(prog.database)
if not database_path.exists():
    if not prog.create:
        raise ValueError(f'database file {database_path} does not exist')
    else:
        wb = cm.create_database_file(database_path, ['experiment'])

# check extensiion (it should be an excel file)
if database_path.suffix != '.xlsx':
    raise ValueError(f'database file {database_path} must be an excel file')

prog.xp_path = Path(prog.xp_path)
prog.recipe_paths = [Path(p) for p in prog.recipe_paths]
prog.lib = ut.load_lib()


##────────────────────────────────────────────────────────────────────────────}}}


### {{{  --     create dmans dictionary of datamanagers (where available)    --
dmans = {}
all_recipes = []
all_networks = []
xp_dmans = {}


for xpname, xp in list(xp_objs.items())[:]:
    print(f'loading {xpname}')
    is_ok = True
    if xp.data_files:
        print(f'loading {xpname}')
        xp.load_raw_data()
        networks, samples = xp.build_networks(ignore_errors=True, inverse='all')
        X, Y = xp.get_XY(networks, samples, ignore_errors=True)
        if xp.network_building_errors:
            is_ok = False
            print(f'{xp.network_building_errors}')
        if xp.data_loading_errors:
            is_ok = False
            print(f'{xp.data_loading_errors}')
        xp_entries[xpname]['network_building_errors'] = xp.network_building_errors
        xp_entries[xpname]['data_loading_errors'] = xp.data_loading_errors
        assert len(networks) == len(X) == len(Y)
        for i, net_entry in enumerate(networks):
            if net_entry:
                net_entry = {
                    'xp': xpname,
                    'network': net_entry,
                    'sample': samples[i],
                }
                all_networks.append(net_entry)
        if is_ok:
            for x, y, net_entry in zip(X, Y, networks):
                if x.size == 0 or y.size == 0:
                    is_ok = False
                    xp_entries[xpname][
                        'data_loading_errors'
                    ] += f'empty data for network {net_entry.name}\n\n'

        if is_ok:
            xp_dmans[xpname] = du.DataManager(X, Y, networks, data_cfg=training_config)


print('done')
networks

##────────────────────────────────────────────────────────────────────────────}}}##


# if prog.xp_cache_dir is not None:
    # prog.xp_cache_dir = Path(prog.xp_cache_dir)
    # prog.xp_cache_dir.mkdir(parents=True, exist_ok=True)
    # # remove all the files in the cache dir
    # for f in prog.xp_cache_dir.iterdir():
        # if f.is_file():
            # f.unlink()
    # for xp_name, xp in xp_objs.items():
        # xp_path = prog.xp_cache_dir / f'{xp_name}.pkl'
        # ut.save(xp, xp_path)
        # xp_entries[xp_name]['xp_obj_cache'] = xp_path
