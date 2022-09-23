## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     import and init     --
# ···············································································
%load_ext autoreload
%autoreload 2

import streamlit as st
st.set_page_config(layout='wide')

import pandas as pd
import numpy as np
import jax.numpy as jnp
import sqlite3
import os

from collections import defaultdict
import jax
from jax import jit, vmap, grad
import scriptutils as ut
import biocomp.utils as bu
from functools import partial
import biocomp as bc
import json
from rich import print

l = ut.load("../biocomp/test_data/all_sheets.pickle")
lib = bc.PartsLibrary(l.parts, l.L0s, l.L1s, l.L2s, l.categories, l.sequestrons, l.sequestron_types)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

# create network from test_data/recipe00.json5:
