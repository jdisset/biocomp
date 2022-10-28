## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                 --     import and init     --
# ···············································································
import streamlit as st

st.set_page_config(layout='wide')

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
import logging


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import wandb as wb

api = wb.Api()
entity, project = 'jdisset', 'georg_data_3'
all_runs = api.runs(f'{entity}/{project}')
run_dict = {run.name: run.id for run in all_runs}
run_name = 'wise-mountain-18'

run = api.run(f'{entity}/{project}/{run_dict[run_name]}')

# get the config
config = run.config

# get the metrics
metrics = run.history()
metrics
# plot loss over time
import matplotlib.pyplot as plt
plt.plot(metrics['loss'])
plt.show()

# now plot each param over time, on the same plot
# a param is just a column in metrics that has no Nan
# and is not loss
params = [col for col in metrics.columns if col != 'loss' and not metrics[col].isna().any()]

fig, ax = plt.subplots(figsize=(10, 10))
for param in params:
    ax.plot(metrics[param], label=param)
# use log scale for all
ax.set_yscale('log')
ax.legend()
plt.show()

# now plot only affinity params. they end with [num]::affinity. we want to rename it just [num]
affinity_params = sorted([col for col in params if col.endswith('::affinity')])
affinity_intensity = list(range(len(affinity_params)))
affinity_dict = dict(zip(affinity_params, affinity_intensity))
affinity_dict
fig, ax = plt.subplots(figsize=(10, 10))
for param in affinity_params:
    ax.plot(metrics[param], label=affinity_dict[param])
# use log scale for all
ax.set_yscale('log')
# order the legend by final value
order = np.argsort(metrics[affinity_params].iloc[-1].values)[::-1]
ax.legend([ax.get_lines()[i] for i in order], [affinity_dict[affinity_params[i]] for i in order])
fig.suptitle('affinity params')
plt.show()
