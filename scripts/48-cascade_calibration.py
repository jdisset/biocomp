### {{{                          --     imports     --
import biocomp as bc
from biocomp import datautils as du
from jax.scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import scriptutils as ut
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import gaussian_kde
import jax
import optax
from jax import jit, vmap, value_and_grad
from tqdm import tqdm
from calibry import Calibration
##────────────────────────────────────────────────────────────────────────────}}}

# get xp_name from --xp argument

xp_name = '2023-03-19_CascadesV2'


xp_path = ut.DEFAULT_XP_PATH / xp_name
raw_path = xp_path / 'data/raw_data_gated'

# create the control dictionnary:
# key = name of the control, value = path to the csv or fcs file
# key follows the following convention:
# single color control : the name of the protein. 'EBFP2', 'EYFP', ...
# all color control: 'ALL'
# blank: 'CNTL' or 'EMPTY'
control_files = list(raw_path.glob('color_controls/*.csv'))
controls = {c.stem.split('.')[0]: c for c in control_files}

beads = list(raw_path.glob('beads/*.fcs'))[0]

cal = Calibration(controls, beads, reference_protein='MKATE', use_channels=['FITC', 'PACIFIC_BLUE', 'PE_TEXAS_RED', 'APC_ALEXA_700'])
cal.fit()
cal.plot_beads_diagnostics()
cal.plot_color_mapping_diagnostics()
cal.plot_bleedthrough_diagnostics()

ctrls = cal._Calibration__controls_values

cmasks = cal._Calibration__controls_masks
all_masks = cmasks.sum(axis=1) > 1
fluo_X = ctrls[all_masks]
fluo_X

log_scale = cal._Calibration__log_scale_factor
log_scale

bleedthrough = cal._Calibration__bleedthrough_matrix
bleedthrough
# plot the bleedthrough matrix

fig, ax = plt.subplots()
logbt = np.log10(bleedthrough)
im = ax.imshow(logbt, cmap='viridis')
pnames = cal._Calibration__fluo_proteins
cnames = cal._Calibration__channel_order
ax.set_xticks(np.arange(len(cnames)))
ax.set_yticks(np.arange(len(pnames)))
ax.set_xticklabels(cnames)
ax.set_yticklabels(pnames)
bleedthrough



def logtransform(x, scale, offset=0):
    x = jnp.clip(x, 1e-9, None)
    return jnp.log10(x + offset) / scale

logX = logtransform(controls[:,0], log_scale)
logX
