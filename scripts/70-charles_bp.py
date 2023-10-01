### {{{                          --     imports     --
import matplotlib

import biocomp as bc
from biocomp import datautils as du
from functools import partial
from datetime import datetime
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import jax
from jax import jit, vmap, value_and_grad
import jax.numpy as jnp

from biocomp import utils as ut
import scriptutils as su
import biocomp.datautils as du
from biocomp import train
from biocomp import compute as cmp

# matplotlib.use('agg')

##────────────────────────────────────────────────────────────────────────────}}}
### {{{                      --     load parameters     --
training_archive = du.load('../__results/training_archives/20230923_fulltrain_v0.pkl')
shared_parameters = training_archive['parameters']
compute_config = training_archive['compute_config']
training_config = training_archive['training_config']
compute_config.set_impl('bias', bc.nodes.bias)
##────────────────────────────────────────────────────────────────────────────}}}

XP = {'BPattempt': '2023-05-10_ColorControls_BPV2'}
xpname = 'BPattempt'
with ut.timer(f'Loading data and building networks for {XP[xpname]}'):
    lib = su.load_lib()
    bp_xp = su.load_xp(XP[xpname], lib, data_path='./data/calibrated_data_v3')
    dman_full = du.DataManager.from_xps([bp_xp], training_config, inverse='all')

##

# su.plot_networks([dman_full.get_networks()[0]], W=2000, H=4000)
n = dman_full.get_networks()[0]
n.get_inverted_input_proteins()


from matplotlib import pyplot as plt

savedir = Path('~/Desktop/bp_attempt_charles/v2/').expanduser()
savedir.mkdir(exist_ok=True)
for i in range(len(dman_full.get_networks()))[:]:
    fig, ax = du.mkfig(1,1, (14,14), dpi=200)
    du.network_plot(dman_full, i, ax=ax, input_order=[0,1,2])
    fig.savefig(savedir / f'network_{i}.pdf')
    plt.show()
    plt.close(fig)
    print(f'Saved network {i}')



