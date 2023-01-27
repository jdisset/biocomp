
### {{{                          --     imports     --
import datetime
import biocomp as bc
import matplotlib.pyplot as plt
import numpy as np
import time
from functools import partial
import biocomp.utils as bu
import scriptutils as ut
import jax
from jax import jit, vmap, grad, value_and_grad
import jax.numpy as jnp
import biocomp.datautils as du
import optax
from pathlib import Path
from tqdm import tqdm
import biocomp.nodes as bn
import biocomp.compute as bcc
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [7.0, 7.0]
plt.rcParams['figure.dpi'] = 200

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                         --     load xps     --
lib = ut.load_lib()

xp = ut.load_xp('2023-01-22_CasE_ALLuORFs', lib)
dman = du.DataManager.from_xps([xp])

mnames = [m.node_namespace for m in dman.get_models()]
# ut.plot_networks([m.network for m in mass_dman.get_models()])


##────────────────────────────────────────────────────────────────────────────}}}
