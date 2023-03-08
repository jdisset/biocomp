### {{{                          --     imports     --
import biocomp as bc
from biocomp import datautils as du
from jax.scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib as mpl
from biocomp.calibration import Calibration
import scriptutils as ut
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import gaussian_kde
import jax
import optax
from jax import jit, vmap, value_and_grad
from tqdm import tqdm

plt.rcParams['figure.dpi'] = 200
##────────────────────────────────────────────────────────────────────────────}}}

xp_name = '2023-03-03_CascadesV1'
xp_path = ut.DEFAULT_XP_PATH / xp_name
raw_path = xp_path / 'data/raw_data_gated'

control_files = list(raw_path.glob('color_controls/*.csv'))
controls = {c.stem.split('.')[0]: c for c in control_files}

beads = list(raw_path.glob('beads/*.fcs'))[0]

controls

cal = Calibration(controls, beads, reference_protein='mKate', use_channels=['FITC', 'PACIFIC_BLUE', 'PE_TEXAS_RED', 'APC_ALEXA_700'])

cal.fit_TASBE()
cal.plot_beads_diagnostics()

##

Y_all = cal.controls[('ALL',)].values
X_corrected = cal.apply_to_array(Y_all)
pnames = cal._Calibration__fluo_proteins

du.fluo_densities(X_corrected, pnames)


##
Y = Y_all
Y = Y - cal._Calibration__autofluorescence  # remove autofluorescence
X = Y @ jnp.linalg.pinv(cal._Calibration__bleedthrough_matrix)  # apply bleedthrough
X += cal._Calibration__offset  # add offset

X = X[jnp.all(X > 1, axis=1)]
logX = bc.calibration.logtransform(X, cal._Calibration__log_scale_factor, 0)
logX = logX[jnp.all(logX > cal.clamp_values[0], axis=1)]
logX = logX[jnp.all(logX < cal.clamp_values[1], axis=1)]

du.fluo_scatter(logX, pnames, logscale=False)
cmapX = cal.cmap_transform(logX)
du.fluo_scatter(cmapX, pnames, logscale=False)

refprotid = cal._Calibration__fluo_proteins.index(cal.reference_protein)
refchanid = jnp.argmax(cal._Calibration__bleedthrough_matrix[refprotid])
jnp.argmax(cal._Calibration__bleedthrough_matrix[refprotid])
refchanid
calibratedX = jnp.array([cal.beads_transform[refchanid](cmapX[:, i]) for i in range(cmapX.shape[1])]).T
refchanid = cal._Calibration__channel_order.index(cal.reference_channel)


x_order = np.argsort(X, axis=0)
Xx, Yx = np.take_along_axis(X, x_order, axis=0), Y[x_order]
Xx[-6].shape

## plot transform
pnames = cal._Calibration__fluo_proteins
target_prot = 'mkate'
CF=100
X = logX * CF
Y = X[:, refprotid]
xx = np.linspace(0, 1, 500)
all_xx = np.tile(xx, (len(pnames), 1)).T
yy_all = cal.cmap_transform(all_xx)
xx_all = cal.cmap_inv_transform(yy_all)
xrange = (cal.clamp_values[0].min()*0.9, cal.clamp_values[1].max()*1.05)
for i in range(0, 4):
    yy = yy_all[:, i]
    selected = (xx > cal.clamp_values[0][i]*0.8) & (xx < cal.clamp_values[1][i]*1.2)
    plt.scatter(X[:, i]/CF, Y/CF, s=5, alpha=0.075, marker='.', linewidths=0)
    plt.plot(xx[selected], yy[selected], alpha=1, lw=2.5, color='w')
    plt.plot(xx[selected], yy[selected], label=pnames[i], alpha=1, lw=1, color='C%d' % i)

plt.ylim(xrange)
plt.xlim(xrange)
plt.legend()
plt.xlabel('source')
plt.ylabel(f'target ({target_prot})')
plt.title(f'Color mapping to {target_prot}')


