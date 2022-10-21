## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                 --     import and init     --
# ···············································································
from jax.tree_util import Partial as partial
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, value_and_grad
import biocomp as bc
import biocomp.utils as bu
import scriptutils as ut
import datautils as du
from pathlib import Path
import json5
import json
import sqlite3
from tqdm import tqdm
import pandas as pd
import numpy as np

lib = ut.getLibFromGoogleSheet()

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

base_path = Path("/Users/jeandisset/Dropbox (MIT)/Biocomp/")
xp_path = base_path / "Experiments"
recipe_path = base_path / "Recipes"
experiments = [x.name for x in xp_path.iterdir() if x.is_dir()]
# ['20220501-GW-l1vsl2', 'E20221012A_massCtrls']

xps = {}

for x in tqdm(experiments):
    xps[x] = bc.XP(x, xp_path, recipe_path, lib)


