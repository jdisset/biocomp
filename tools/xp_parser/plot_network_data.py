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
prog.add_argument('--network_id', help='network id to plot: int, list of ints, or "all"', default='all')

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

if prog.args.database is not None:
    netdf = cm.load_database_table(prog.args.database, 'network')
    xpdf = cm.load_database_table(prog.args.database, 'experiment')

netdf.columns
xpdf.columns
