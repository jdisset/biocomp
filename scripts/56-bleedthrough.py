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

plt.rcParams['figure.dpi'] = 200
##────────────────────────────────────────────────────────────────────────────}}}

xp_name = '2023-04-18_Constraints_PguCascades'
xp_name = '2023-04-03_Constraints_Pgu_Bleedthrough'
xp_path = ut.DEFAULT_XP_PATH / xp_name
raw_path = xp_path / 'data/raw_data_gated'

control_files = list(raw_path.glob('color_controls/*.csv'))
controls = {c.stem.split('.')[0]: c for c in control_files}
beads = list(raw_path.glob('beads/*.fcs'))[0]

cal = Calibration(
    controls,
    beads,
    reference_protein='MKATE',
    # use_channels=['FITC', 'PACIFIC_BLUE', 'PE_TEXAS_RED', 'APC_ALEXA_700', 'APC', 'PE','PerCP_CY5_5', 'AMCYAN', 'APC_CY7' ],
    use_channels=['FITC', 'PACIFIC_BLUE', 'PE_TEXAS_RED', 'APC_ALEXA_700', 'APC', 'PE'],
    # use_channels=['FITC', 'PACIFIC_BLUE', 'PE_TEXAS_RED', 'APC_ALEXA_700'],
    remove_after_fluo_value = 250000,
)

cal.fit()
cal.plot_beads_diagnostics()
cal.plot_color_mapping_diagnostics()
cal.plot_bleedthrough_diagnostics()


channel_order = cal._Calibration__channel_order
pnames = cal._Calibration__fluo_proteins
cv = cal._Calibration__controls_values
cm = cal._Calibration__controls_masks.astype(bool)

autofluo = cal._Calibration__autofluorescence
allctrl = np.all(cm, axis=1)
noctrl = np.all(~cm, axis=1)


def get_single_ctrl(pname):
    ctrl = np.zeros(len(pnames), dtype=bool)
    ctrl[pnames.index(pname)] = True
    mask = np.all(cm == ctrl, axis=1)
    return cv[mask]


du.fluo_scatter(get_single_ctrl('MKATE'), channel_order, title='MKATE ctrl')
du.fluo_scatter(get_single_ctrl('IRFP720'), channel_order, title='IRFP720 ctrl')
du.fluo_densities(get_single_ctrl('TAGBFP'), channel_order, title='tagbfp ctrl')
du.fluo_densities(get_single_ctrl('IRFP720'), channel_order, title='IRFP720 ctrl')

du.fluo_scatter(cv[allctrl], channel_order, title='all ctrl')
du.fluo_scatter(cv[noctrl], channel_order, title='empty ctrl')


##
BW = 0.03
xmin = 2
xmax = 8


# F = du.fluo_densities
F = du.fluo_scatter
raw_mkate = get_single_ctrl('MKATE')

highapc_mask = raw_mkate[:, channel_order.index('APC')] > 10**4
highapc = raw_mkate[highapc_mask]

lowapc_mask = raw_mkate[:, channel_order.index('APC')] < 10**4
lowapc = raw_mkate[lowapc_mask]


mediumapc_mask = (raw_mkate[:, channel_order.index('APC')] > 10**3) & (raw_mkate[:, channel_order.index('APC')] < 10**4)
mediumapc = raw_mkate[mediumapc_mask]

corrected_mkate = cal.apply_to_array(highapc)
F(corrected_mkate, pnames, xmin=xmin, xmax=xmax, title='corrected highAPC MKATE ctrl', bw_method=BW, alpha=1)

corrected_mkate = cal.apply_to_array(lowapc)
F(corrected_mkate, pnames, xmin=xmin, xmax=xmax, title='corrected lowAPC MKATE ctrl', bw_method=BW, alpha=1)

corrected_mkate = cal.apply_to_array(mediumapc)
F(corrected_mkate, pnames, xmin=xmin, xmax=xmax, title='corrected mediumAPC MKATE ctrl', bw_method=BW, alpha=1)

fullmkate = cal.apply_to_array(raw_mkate)
F(fullmkate, pnames, xmin=xmin, xmax=xmax, title='corrected MKATE ctrl', bw_method=BW, alpha=1)

F(raw_mkate, channel_order, xmin=0, xmax=6, title='MKATE ctrl raw', bw_method=BW, alpha=0.5)

corrected_irfp = cal.apply_to_array(get_single_ctrl('IRFP720'))
F(corrected_irfp, pnames, xmin, xmax, title='corrected IRFP720 ctrl', bw_method=BW, alpha=1)

corrected_mneon = cal.apply_to_array(get_single_ctrl('MNEONGREEN'))
F(corrected_mneon, pnames, xmin, xmax, title='corrected mNeonGreen ctrl', bw_method=BW)

corrected_all = cal.apply_to_array(cv[allctrl])
F(corrected_all, pnames, xmin, xmax, title='corrected all ctrl', bw_method=BW)

corrected_empty = cal.apply_to_array(cv[noctrl])
F(corrected_empty, pnames, xmin, xmax, title='corrected empty ctrl', bw_method=BW)
