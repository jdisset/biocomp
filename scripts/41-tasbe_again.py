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

### {{{                    --     first call to calib     --

data_dir = ut.DEFAULT_XP_PATH / '2023-01-22_CasE_ALLuORFs/data/raw_data_gated/'

blank = data_dir / 'color_controls/CNTL.2023-01-22_CasE_ALLuORFs.csv'

controls = {
    'blank': data_dir / 'color_controls/CNTL.2023-01-22_CasE_ALLuORFs.csv',
    'eYFP': data_dir / 'color_controls/eYFP.2023-01-22_CasE_ALLuORFs.csv',
    'eBFP2': data_dir / 'color_controls/EBFP2.2023-01-22_CasE_ALLuORFs.csv',
    'mKate': data_dir / 'color_controls/mKate.2023-01-22_CasE_ALLuORFs.csv',
    'all': data_dir / 'color_controls/ALL.2023-01-22_CasE_ALLuORFs.csv',
}

beads = Path(data_dir / 'beads/2023-01-22_CasE_ALLuORFs_BEADS_AL01_017.fcs')

cal = Calibration(
    color_controls_files=controls,
    beads_file=beads,
    reference_protein='EBFP2',
    beads_mef_values=bc.calibration.SPHEROTECH_RCP_30_5a,
    use_channels=['FITC-A', 'PACIFIC_BLUE_A', 'PE_TEXAS_RED_A'],
)

cal.fit_TASBE()
S = cal._Calibration__bleedthrough_matrix
autofluo = cal._Calibration__autofluorescence
pnames = cal._Calibration__fluo_proteins
cnames = cal._Calibration__channel_order
bchan = cal._Calibration__beads_channel_order
densities = cal._Calibration__beads_densities
log_scale_factor = cal._Calibration__log_scale_factor



##────────────────────────────────────────────────────────────────────────────}}}

Y_all = cal.controls[('ALL',)].values - autofluo
Y_all.shape
Y = Y_all

order = np.argsort(Y, axis=0)
Ys = np.take_along_axis(Y, order, axis=0)
np.all(np.diff(Ys[:, 0]) >= 0)

# (Y_all < 0).sum() / Y_all.size
# du.fluo_distributions(Y_all, cnames, title='ALL - Y')


# cal.plot_beads_diagnostics()

# splines = cal.splines
# transform = lambda X: np.array([s(x) for s, x in zip(splines, X.T)]).T
# transform(beads_peaks)

# ### {{{                         --     beads fit     --
# from scipy.interpolate import UnivariateSpline

# spline_order = 1
# smooth_percent = 20
# CF = 100
# X = beads_peaks * CF  # (NBEADS, NCHAN)
# Y = logcalib * CF
# NBEADS, NCHAN = X.shape


# MIN_PERCENT_DIST = 5

# dist_fwd = np.abs(X[1:] - X[:-1])
# # make sure X and Y are ordered
# assert np.all(dist_fwd >= 0)
# ydiff = np.diff(Y, axis=0)
# assert np.all(ydiff >= 0)

# dist_fwd = jnp.concatenate([dist_fwd[0:1], dist_fwd], axis=0)
# dist_bwd = np.abs(X[:-1] - X[1:])
# # missing element is duplicated
# dist_bwd = jnp.concatenate([dist_bwd, dist_bwd[-1:]], axis=0)
# avg_dist_to_neigh = (dist_fwd + dist_bwd) / 2
# w = np.clip(avg_dist_to_neigh / MIN_PERCENT_DIST, 0, 1)
# w

# splines = [
# UnivariateSpline(
# X[:, c], Y[:, c], k=spline_order, s=smooth_percent, check_finite=True, w=w[:, c]
# )
# for c in range(0, NCHAN)
# ]
# inv_splines = [
# UnivariateSpline(
# Y[:, c], X[:, c], k=spline_order, s=smooth_percent, check_finite=True, w=w[:, c]
# )
# for c in range(0, NCHAN)
# ]
# xdiff[:, 7]
# X[:, 7]
# Y[:, 7]
# w[:, 4]
# w
# splines[7].get_knots()
# splines[7].get_coeffs()


##────────────────────────────────────────────────────────────────────────────}}}

from scipy.interpolate import UnivariateSpline
from scipy.interpolate import LSQUnivariateSpline

offset = 150
threshold = 1
refprotid = 0
spline_order = 3
smooth_percent = 100000
N_KNOTS = 1
Y_all = cal.controls[('ALL',)].values - autofluo
X_all = Y_all @ jnp.linalg.pinv(S) + offset
X_all = X_all[jnp.all(X_all > threshold, axis=1)]
logX = np.log10(X_all) / log_scale_factor
nresample = min(100000, logX.shape[0])
which = np.random.choice(logX.shape[0], nresample, replace=False)
CF = 100
X = logX[which] * CF
Y = X[:, refprotid]
x_order = np.argsort(X, axis=0)
Xx, Yx = np.take_along_axis(X, x_order, axis=0), Y[x_order]
w = 0.1 * np.ones_like(Y)
knots = np.linspace(np.quantile(X, 0.001, axis=0), np.quantile(X, 0.999, axis=0), N_KNOTS).T
splines = [
    LSQUnivariateSpline(Xx[:, c], Yx[:, c], k=spline_order, t=knots[c], w=w)
    for c in tqdm(range(X.shape[1]))
]
y_order = np.argsort(Y, axis=0)
Xy, Yy = X[y_order], Y[y_order]
yknots = np.linspace(Yy.min() + 1, Yy.max() - 1, N_KNOTS)
inv_splines = [
    LSQUnivariateSpline(Yy, Xy[:, c], k=spline_order, t=yknots, w=w) for c in range(X.shape[1])
]
transform = lambda X: np.array([s(x * CF) / CF for s, x in zip(splines, X.T)]).T
inv_transform = lambda Y: np.array([s(y * CF) / CF for s, y in zip(inv_splines, Y.T)]).T

# plot transform
xx = np.linspace(0, 1, 500)
xx_3 = np.array([xx, xx, xx]).T
# yy = transform(xx_3)
yrange = (xx.min(), xx.max())
yy_all = transform(xx_3)
xx_all = inv_transform(yy_all)
for i in range(0, 3):
    # yy = splines[i](xx * CF) / CF
    yy = yy_all[:, i]
    xx_inv = xx_all[:, i]
    plt.scatter(X[:, i]/CF, Y/CF, s=5, alpha=0.05, marker='.', linewidths=0)
    plt.plot(xx, yy, alpha=1, lw=2.5, color='w')
    plt.plot(xx, yy, label=pnames[i], alpha=1, lw=1, color='C%d' % i)
    plt.plot(xx_inv, yy, alpha=1, lw=2.5, color='w')
    plt.plot(xx_inv, yy, label=pnames[i], alpha=1, lw=1, color='k')

plt.ylim(yrange)
plt.xlim(yrange)
plt.legend()
plt.xlabel('source')
plt.ylabel('target (EBFP)')
plt.title(
    f'Color mapping to EBFP\n offset={offset}, threshold={threshold}, degree={spline_order}'
)


##

th=0
offset = 200

Y_all = cal.controls[('ALL',)].values - autofluo
X_all = Y_all @ jnp.linalg.pinv(S) 
YY=X_all + offset
YY = YY[jnp.all(YY > th, axis=1)]
du.fluo_scatter(YY,pnames)

##
Y_all = cal.controls[('ALL',)].values
X_corrected = cal.apply_to_array(Y_all)
du.fluo_scatter(X_corrected, pnames, fname=Path('~/Desktop/fluos_corrected.png').expanduser()


