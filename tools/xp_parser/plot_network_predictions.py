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
# arguments:
# --database: path to the database file (mandatory)
prog.add_argument('--database', type=str, required=True)

DEFAULT_CALIB_PATHS = [
    './data/calibrated_data_v3',
    './data/calibrated_data_v2',
    './data/calibrated_data',
]
prog.add_argument('--calib_paths', type=str, nargs='+', default=DEFAULT_CALIB_PATHS)

DEFAULT_CALIB_NAMES = ['v3', 'v2', 'old']
prog.add_argument('--calib_names', type=str, nargs='+', default=DEFAULT_CALIB_NAMES)

DEFAULT_XP_PATH = ut.DEFAULT_XP_PATH
prog.add_argument('--xp_path', type=str, default=DEFAULT_XP_PATH)

DEFAULT_RECIPE_PATH = ut.DEFAULT_RECIPE_PATH
prog.add_argument('--recipe_paths', type=str, nargs='+', default=DEFAULT_RECIPE_PATH)

DEFAULT_XP_CACHE_DIR = './devtmp/cache/xp_objs'
prog.add_argument('--xp_cache_dir', type=str, default=DEFAULT_XP_CACHE_DIR)


DEFAULT_DATA_CONFIG = {
    'network_cache_location': './__cache/network',
    'training_cache_location': './__cache/training',
    'densities_cache_location': './__cache/densities',
    'data_min_value': 500,
    'data_max_value': 100000000.0,
    'data_log_offset': 3000.0,
    'data_log_factor': 100,
    'data_log_poly_threshold': 300,
    'data_log_poly_compression': 0.4,
    'data_sampling_kde_bw_method': 0.02,
    'data_sampling_max_density_samples': 4000,
    'data_sampling_density_quantile_threshold': 0.025,
    'data_sampling_coords_for_density_threshold': 0.15,
}
DEFAULT_DATA_CONFIG_PATH = None
prog.add_argument('--data_config', type=str, default=DEFAULT_DATA_CONFIG_PATH)

# verbosity level
prog.add_argument('--verbose', type=int, default=0)

prog.parse_args(['--database', 'devtmp/database.xlsx', '--create'])
##────────────────────────────────────────────────────────────────────────────}}}
### {{{                    --     arg postprocessing     --

# get the database path
database_path = Path(prog.database)
if not database_path.exists():
    if not prog.create:
        raise ValueError(f'database file {database_path} does not exist')
    else:
        wb = cm.create_database_file(database_path, ['experiment', 'network'])

# check extensiion (it should be an excel file)
if database_path.suffix != '.xlsx':
    raise ValueError(f'database file {database_path} must be an excel file')

prog.xp_path = Path(prog.xp_path)
prog.recipe_paths = [Path(p) for p in prog.recipe_paths]
prog.lib = ut.load_lib()
if prog.data_config is None:
    prog.data_config = DEFAULT_DATA_CONFIG
else:
    import json5

    prog.data_config = json5.load(open(prog.data_config, 'r'))

assert len(prog.calib_paths) == len(prog.calib_names)

import logging

# loggers = [logging.getLogger(name) for name in sorted(logging.root.manager.loggerDict)]
logging.getLogger('jax').setLevel(logging.WARNING)
# completely silence biocomp's logger (including warning and error messages)
logging.getLogger('biocomp').setLevel(logging.CRITICAL)


# rich console
from rich.console import Console

prog.console = Console()


##────────────────────────────────────────────────────────────────────────────}}}
logger = logging.getLogger('build_xp_table')

### {{{                         --     plot data     --

dataplot_savedir = Path('~/ResearchMisc/biocomp/dataplots').expanduser()
dataplot_savedir.mkdir(parents=True, exist_ok=True)

xp_to_plot =  ['2023-03-26_MatrixCsy4']

for xp_name, dman in list(xp_dmans.items())[:]:
    if xp_name not in xp_to_plot:
        continue
    print(f'Plotting {xp_name}')
    for i, net in list(enumerate(dman.get_networks()))[:1]:
        print(f'Plotting {i} / {len(dman.get_networks())} - {net.name}')
        # try:
        netname = f'{net.metadata["from_xp"]}.{net.name}'
        filename = f'{netname}.png'
        # if Path(dataplot_savedir / filename).exists():
            # print(f'Skipping {filename} (already exists)')
            # continue
        outputs = net.get_output_proteins()
        inputs = net.get_inverted_input_proteins()
        print(f'outputs: {outputs}')
        params = dict(
            vmin=0,
            vmax=0.7,
            xmin=0,
            xmax=0.85,
            slices=[0.1, 0.3, 0.5],
            knn_method='quantile',
            qu=0.5,
            method='scatter'
        )
        if len(outputs) <= 3:
            fig, ax = pu.mkfig(1, 1, (4, 4), dpi=300)
            pu.network_plot(dman, i, ax=ax, **params)
        else:
            # find the protein in output but not in input
            actual_outputs = [o for o in outputs if o not in inputs]
            assert len(actual_outputs) == 1
            print(f'actual_outputs: {actual_outputs}')
            fig, allaxes = pu.mkfig(3, 3, (3, 3), dpi=300)
            input_order = [0, 1, 2]
            axes = allaxes[0]
            pu.network_plot(dman, i, axes=axes, ax=None, input_order=input_order, **params)
            input_order = [0, 2, 1]
            axes = allaxes[1]
            pu.network_plot(dman, i, axes=axes, ax=None, input_order=input_order, **params)
            input_order = [2, 1, 0]
            axes = allaxes[2]
            pu.network_plot(dman, i, axes=axes, ax=None, input_order=input_order, **params)

        fig.tight_layout()
        fig.savefig(dataplot_savedir / f'{netname}.png', bbox_inches='tight')
        plt.show()
        plt.close(fig)
        plt.close('all')
        # except Exception as e:
            # print(f'Error plotting {net.name}')
            # print(e)
            # continue


##────────────────────────────────────────────────────────────────────────────}}}

