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
    beads_reference_values=bc.calibration.SPHEROTECH_RCP_30_5a,
    use_channels=['FITC-A', 'PACIFIC_BLUE_A', 'PE_TEXAS_RED_A'],
)
bparams = cal._Calibration__beads_params

cal.fit_TASBE()
S = cal._Calibration__bleedthrough_matrix
autofluo = cal._Calibration__autofluorescence
pnames = cal._Calibration__fluo_proteins
cnames = cal._Calibration__channel_order
beads_peaks = cal._Calibration__beads_peaks
logcalib = cal._Calibration__logcalib
logbeads = cal._Calibration__logbeads
bchan = cal._Calibration__beads_channel_order
densities = cal._Calibration__beads_densities

d = densities[:,4,:]
# d = d / d.max(axis=1)[:,None]

# for each bead, compute the max density at each point that is NOT from the bead itself
masks = 1.0-jnp.eye(d.shape[0])
max_other = vmap(lambda x, m: jnp.max(m[:,None]*x, axis=0), in_axes=(None,0))(d, masks)
non_overlap = jnp.maximum(0, d - max_other)
max_other.shape
# normalize per total density for each bead
# non_overlap = non_overlap / jnp.sum(d, axis=1)[:,None]
# non_overlap /= jnp.max(non_overlap, axis=1)[:,None]
# confidence is the proportion of each bead density that is not overlapped
norm_d = d / jnp.max(d, axis=1)[:,None]
confidence = jnp.sum(non_overlap, axis=1) / jnp.sum(d, axis=1)
confidence

# find the position of peaks for each bead using the average of the beads 
# observations weighted by the density
TH = 0.25 # threshold for density to be considered in the average
norm_densities = densities / densities.max(axis=2)[:,:,None]
w.shape
x = jnp.arange(0, densities.shape[2])
w = jnp.where(norm_densities > TH, norm_densities-TH, 0)
xx = jnp.tile(x, (densities.shape[0], densities.shape[1], 1))
avg_pos = jnp.average(xx, axis=2, weights=w)
avg_pos
# avg_pos = vmap(jnp.average, in_axes=(None,None,0))(jnp.arange(0, d.shape[1]), 0, w)

# plot densities
fig, ax = du.mkfig(1, 1, (15,7))
for i in range(0, d.shape[0]):
    ax.plot(non_overlap[i,:], color=mpl.cm.tab10(i), label=f'bead {i}')
    ax.plot(norm_d[i,:], color=mpl.cm.tab10(i), label=f'bead {i}')
    # ax.plot(avg_pos[i], d[i,int(avg_pos[i])], 'x', color=mpl.cm.tab10(i))
    # vertical line at average position
    ax.axvline(avg_pos[i], color=mpl.cm.tab10(i), linestyle='--')
ax.set_title('density of peaks for channel 4')
ax.set_xlabel('bead index')
ax.set_ylabel('density')
ax.legend()



# for each bead, compute the max density at each point that is NOT from the bead itself
masks = 1.0-jnp.eye(d.shape[0])
norm_densities = densities / densities.max(axis=2)[:,:,None]
max_other = vmap(lambda x, m: jnp.max(m[:,None]*x, axis=0), in_axes=(None,0))(d, masks)
non_overlap = jnp.maximum(0, d - max_other)
norm_d = d / jnp.max(d, axis=1)[:,None]
confidence = jnp.sum(non_overlap, axis=1) / jnp.sum(d, axis=1)
confidence

from scipy.interpolate import UnivariateSpline
from scipy.interpolate import LSQUnivariateSpline
cf = 100
X = beads_peaks*cf
beads_peaks.shape
beads_peaks
Y = logcalib*cf
# Y = beads_peaks
# X = logcalib
NCHAN = logcalib.shape[1]
order = jnp.argsort(X, axis=0)
spls = []

# if using higher order than 1, first add some extrapolation from k=1
K = 2
for i in range(0,NCHAN):
    x = np.maximum(X[order[:,i],i], 1e-3)
    y = np.maximum(Y[order[:,i],i], 1e-3)
    # w = [1,1,1,1,1,0.1,0.1,0,0]
    w = cf/y
    spl = UnivariateSpline(x, y, k=K, s=1, w=w, check_finite=True)
    spls.append(spl)
transform = lambda X: np.array([s(x) for s,x in zip(spls,X.T)]).T
fig, ax = du.mkfig(1, 1)
channels_to_plot = [5,8,4]
for i in channels_to_plot:
    color = mpl.cm.tab10(i)
    ax.plot(X[:,i], Y[:,i], 'o', label=f'channel {i}', color=color)
    xx = np.linspace(X[:,i].min()-10, Y[:,i].max()+20, 500)
    yy = spls[i](xx)
    print(f'knots: {spls[i].get_knots()}')
    ax.plot(xx, yy, color=color)
    newpeaks = spls[i](X[:,i])
    ax.plot(newpeaks, Y[:,i], 'x', color=color)
    minmax = np.array([[X[:,i],Y[:,i]], [X[:,i],Y[:,i]]])
    minmax = minmax.min(), minmax.max()
    # plot knots:
    # for k in spls[i].get_knots():
        # ax.plot([k,k], minmax, color='k', alpha=0.25, linestyle='--', lw=1)
    ax.set_xlim(minmax + np.array([-1,1])*10)
    ax.set_ylim(minmax + np.array([-1,1])*10)
ax.legend()
ax.plot(minmax, minmax, label='identity', color='k', alpha=0.25, linestyle='--', lw=1)
# TODO: maybe add a 10th point that is just the linear interp at max X value to avoid weird extrapolation

transform_cf = lambda X: np.array([s(x*cf)/cf for s,x in zip(spls,X.T)]).T
bc.calibration.plot_beads_dists(transform, beads_peaks, logcalib, logbeads, bchan, cal.channel_to_unit)

beadsraw = 10 ** (logbeads*4)
du.fluo_densities(beadsraw, bchan)
beadscorrected = 10 ** (transform(logbeads)*4)
du.fluo_densities(beadscorrected, bchan)

spl[0].get_knots()

jnp.argsort(beads_peaks, axis=0)

fig, ax = du.mkfig(1, 1)
bc.calibration.plot_spectral_sig(cal._Calibration__bleedthrough_matrix, ax)

cal.plot_beads_diagnostics()

beads = cal.beads.values
du.fluo_scatter(beads, cal.beads.columns)
du.fluo_densities(beads, cal.beads.columns)



##────────────────────────────────────────────────────────────────────────────}}}

Y_all = cal.controls[('ALL',)].values - autofluo
(Y_all < 0).sum() / Y_all.size
du.fluo_distributions(Y_all, cnames, title='ALL - Y')
Y_all[:,0].max()

X_all = cal.apply_to_array(Y_all)

@jit
def loglog(x):
    return jnp.where(x > 1, jnp.log10(x), jnp.where(x < -1, -jnp.log10(-x), 0))
@jit
def inv_loglog(x):
    return jnp.where(x > 0, 10 ** x, jnp.where(x < 0, -10 ** -x, 0))


yloglog = loglog(Y_all + 100)
a = 1.0
b = 0
yloglog = yloglog*a + b
du.fluo_distributions(inv_loglog(yloglog), cnames, title='ALL - Y')


du.fluo_distributions(X_all, pnames, title='ALL - X')

cal.offset
Y_all.min()
cal.beads.values.min()

##
Y_bfp = cal.controls[('EBFP2',)].values - autofluo
Y_bfp.mean(axis=0)
du.fluo_distributions(Y_bfp, cnames, title='EBFP2 - Y')
X_bfp = cal.apply_to_array(Y_bfp)
du.fluo_distributions(X_bfp, pnames, title='EBFP2 - X')
(Y_bfp < 0).sum() / Y_bfp.size


Y_yfp = cal.controls[('EYFP',)].values - autofluo
du.fluo_distributions(Y_yfp, cnames, title='EYFP - Y')
X_yfp = cal.apply_to_array(Y_yfp)
du.fluo_distributions(X_yfp, pnames, title='EYFP - X')
(Y_yfp < 0).sum() / Y_yfp.size
(Y_all[:,2]<150).sum() / Y_all[:,0].size


##
offset = 1e3
Y_blank = cal.controls[('BLANK',)].values - autofluo
du.fluo_distributions(Y_blank, cnames, title='ALL - Y')
X_blank = cal.apply_to_array(Y_blank)
du.fluo_distributions(X_blank, pnames, title='ALL - X')

(Y_blank < 0).sum() / Y_blank.size

##
# One solution:
# 1 - when capturing, force the reference channel to have perfect visibility of all peaks
# 2 - non linear (spline?) regression on the beads data to linearize the channel 
# 3 - non linear regression for the other colors...



### {{{                           --     plots     --


def remove_axis_and_spines(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_fluo_distribution(ax, data, res=1000, log10max=6, axtitle=None):
    logdata = jnp.log10(1 + jnp.maximum(0, data)) / log10max
    xrange = 1.0
    XX = jnp.linspace(-0.05, xrange, res)
    kde = gaussian_kde(logdata.T, bw_method=0.01)
    smoothkde = gaussian_kde(logdata.T, bw_method=0.1)
    densities = kde(XX.T)
    densities = densities * 0.01
    ax.fill_between(XX, 0, densities, color='k', alpha=0.25, lw=0)
    ax.plot(XX, densities, color='k', alpha=1, lw=0.5)
    ax.set_yscale("log")
    # ax.set_aspect("equal")
    ax.set_xlim(-0.05, xrange)
    ax.set_ylim(0.001, 1)
    # use real data for y ticks
    ax.set_xticks(XX[:: (res // 5)])
    ax.set_xticklabels([f'{10**x:.1e}' for x in (XX * log10max)[:: (res // 5)]])
    # remove_axis_and_spines(ax)
    if axtitle is not None:
        ax.set_title(axtitle)


def fluo_histogram(ax, data, axtitle=None, minx=-1e5, maxx=1e6, nbins=100):
    bins = np.linspace(minx, maxx, nbins)
    ax.hist(data, bins=bins, color='k', alpha=0.5)
    # ax.set_yscale('log')
    ax.set_xlim(minx, maxx)
    # ax.set_ylim(1, 1e5)
    if axtitle is not None:
        ax.set_title(axtitle)


cal._Calibration__beads_params
fluonames = cal._Calibration__fluo_proteins
channels = cal._Calibration__channel_order
refchanid = cal._Calibration__channel_order.index(cal.reference_channel)
refprotid = cal._Calibration__fluo_proteins.index(cal.reference_protein)

cal.controls.keys()
blank_df = cal.controls[tuple()]
bfp_df = cal.controls[('EBFP2',)]
yfp_df = cal.controls[('EYFP',)]
mkate_df = cal.controls[('MKATE',)]
all_df = cal.controls[('ALL')]
compg = all_df
compg[compg['FITC']<0]
43/76

# df = bfp_df
# df = mkate_df
# df = yfp_df

# Y = df.values - cal._Calibration__autofluorescence
# S = cal._Calibration__bleedthrough_matrix
# S = jnp.where(S < 0.5, 0, S) # no bleedthrough
# X = Y @ jnp.linalg.pinv(S)

# X.min()
# X = (X+200)*10

# X = X[jnp.all(X > 0, axis=1), :]

# # (X<0).sum(axis=0) / X.shape[0]
# # (Y<0).sum(axis=0) / Y.shape[0]

# params = bc.calibration.color_mapping_gd(X, refprotid, num_iter=2000, learning_rate=0.1)
# a,b = bc.calibration.affine_opt(X, refprotid)
# a,b
# params

Y = compg.values
X = cal.apply(Y)


fig, axes = du.mkfig(4, 1, (8, 3))
for i, f in enumerate(channels):
    plot_fluo_distribution(axes[i], Y[:, i], axtitle=f)
    # fluo_histogram(axes[i], Y[:, i], axtitle=f)
fig.suptitle('Raw data')
fig.tight_layout()
fig, axes = du.mkfig(3, 1, (8, 3))
for i, f in enumerate(fluonames):
    plot_fluo_distribution(axes[i], X[:, i], axtitle=f)
    # fluo_histogram(axes[i], X[:,i], axtitle=f)
fig.suptitle('Calibrated protein counts')
fig.tight_layout()


##
import itertools

Y_orig = cal._Calibration__controls_values
# Y_orig = cal._Calibration__to_MEF(Y_orig)
masks = cal._Calibration__controls_masks
fluoprots = cal._Calibration__fluo_proteins
channels = cal._Calibration__channel_order

fluo_to_mask = {fluoprots[i]: jax.nn.one_hot(i, len(fluoprots)) for i in range(len(fluoprots))}
channelpairs = list(itertools.combinations(range(len(channels)), 2))

fluoprots
channels

fig, axes = du.mkfig(len(channelpairs), 2)
Y_orig
masks


# cal._Calibration__bleedthrough_matrix = S.at[S < 0].set(0)
Y_corrected = cal.apply(Y_orig)
for i, (c1, c2) in enumerate(channelpairs):
    size = 0.5
    ch1, ch2 = channels[c1], channels[c2]
    for j in range(2):
        Y = Y_orig if j == 0 else Y_corrected
        for f in fluoprots:
            pick = jnp.all(masks == fluo_to_mask[f], axis=1)
            axes[i][j].scatter(Y[pick, c1], Y[pick, c2], s=size, label=f)
        pick = jnp.all(masks == jnp.array([1, 1, 1]), axis=1)
        axes[i][j].scatter(Y[pick, c1], Y[pick, c2], s=size, label='all')
        axes[i][j].set_xlabel(ch1)
        axes[i][j].set_ylabel(ch2)
        axes[i][j].set_title('Raw' if j == 0 else 'Corrected')
        axes[i][j].legend()
        # log log
        axes[i][j].set_xscale('log')
        axes[i][j].set_yscale('log')
        axes[i][j].set_xlim(1e-1, 1e8)
        axes[i][j].set_ylim(1e-1, 1e8)
fig.tight_layout()


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                         --     toy dist     --

# generate a random gamma distribution of X
keys = jax.random.split(jax.random.PRNGKey(0), 4)
X = jax.random.gamma(keys[0], 3, (100, 2))
S = jnp.array([[1, 0.2, 0.1], [0.25, 1, 0.05]])
M = jax.random.randint(keys[1], (100,), 0, 2)
M = jax.nn.one_hot(M, 2)
Y = X * M @ S
# add some noise
Y = Y + jax.random.normal(keys[2], Y.shape) * jnp.linalg.norm(Y) * 0.01
q = jnp.quantile(Y, 0.1)
# Y = Y.at[Y < q].set(jax.random.uniform(keys[1], (Y[Y < q].shape)))

S_h = jax.random.uniform(keys[3], (2, 3), minval=0.1, maxval=1)

X_h = Y @ jnp.linalg.pinv(S_h)
S_h = jnp.linalg.pinv(X_h * M) @ Y
S_h = S_h / jnp.max(S_h, axis=1)[:, None]
X_h = jnp.maximum(X_h, 0)
S_h


##
y0 = Y[:, 0]
y1 = Y[:, 1]
y2 = Y[:, 2]

# y1 = x * y0
# find x that minimizes the lsq error
# use np.linalg.lstsq
x1 = np.linalg.lstsq(y0[:, None], y1[:, None])[0]
x1
x2 = np.linalg.lstsq(y0[:, None], y2[:, None])[0]
# or with pinv
px1 = np.linalg.pinv(y0[:, None]) @ y1[:, None]
px2 = np.linalg.pinv(y0[:, None]) @ y2[:, None]

# all in one go:
x = np.linalg.lstsq(y0[:, None], Y[:, 1:])[0]
# or with pinv
px = np.linalg.pinv(y0[:, None]) @ Y
px

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                            --     OT     --
import numpy as np
from scipy import stats

S = np.random.normal(loc=0.0, scale=1.0, size=1000)
# T = np.random.normal(loc=0.0, scale=1.0, size=1000)

T = S * 0.7 - 1 + np.random.normal(loc=0.0, scale=0.5, size=1000) * S * 0.1

fig, ax = du.mkfig(2, 1)
fluo_histogram(ax[0], S, minx=-4, maxx=4, nbins=50)
fluo_histogram(ax[1], T, minx=-4, maxx=4, nbins=50)
fig.tight_layout()

##
plt.scatter(S, np.zeros_like(S), s=50, alpha=0.02, lw=0)
plt.scatter(T, np.ones_like(T), s=50, alpha=0.02, lw=0)
plt.show()

slope, intercept, r_value, p_value, std_err = stats.linregress(S, T)
slope, intercept


def affine_transform(X, Y):
    # Compute the means of X and Y
    mean_x = np.mean(X)
    mean_y = np.mean(Y)

    # Compute the centered versions of X and Y
    X_centered = X - mean_x
    Y_centered = Y - mean_y

    # Compute the covariance matrix
    cov = np.sum(X_centered * Y_centered) / len(X)

    # Compute the variance of X
    var_x = np.sum((X - mean_x) ** 2) / len(X)

    # Compute the optimal slope a and intercept b
    a = cov / var_x
    b = mean_y - a * mean_x

    return a, b


a, b = affine_transform(S, T)
a
b
intercept
slope
##
from ott.geometry.pointcloud import PointCloud
from ott.problems.linear.linear_problem import LinearProblem
from ott.solvers.linear.sinkhorn import Sinkhorn


@jit
def ot_mat(s, t):
    return Sinkhorn()(LinearProblem(PointCloud(s[:, None], t[:, None])))


m = ot_mat(S, T)

m.matrix

# d = d / jnp.sum(d)
d = np.exp(d)
##
plt.scatter(S, np.zeros_like(S), s=50, alpha=0.02, lw=0)
plt.scatter(T, np.ones_like(T), s=50, alpha=0.02, lw=0)
plt.scatter(d, np.ones_like(T) * 0.5, s=50, alpha=0.02, lw=0)
plt.show()

##────────────────────────────────────────────────────────────────────────────}}}
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

rng = np.random.default_rng()
x = np.linspace(-3, 3, 50)
y = np.exp(-x**2) + 0.1 * rng.standard_normal(50)
plt.plot(x, y, 'ro', ms=5)

spl = UnivariateSpline(x, y)
xs = np.linspace(-3, 3, 1000)
plt.plot(xs, spl(xs), 'g', lw=3)

spl.set_smoothing_factor(3)
plt.plot(xs, spl(xs), 'b', lw=3)
plt.show()
