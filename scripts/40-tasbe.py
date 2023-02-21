### {{{                          --     imports     --
import biocomp as bc
from biocomp import datautils as du
from jax.scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
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

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                    --     first call to calib     --

data_dir = ut.DEFAULT_XP_PATH / '2023-01-22_CasE_ALLuORFs/data/raw_data_gated/'

blank = data_dir / 'color_controls/CNTL.2023-01-22_CasE_ALLuORFs.csv'

controls = {
    'eYFP': data_dir / 'color_controls/eYFP.2023-01-22_CasE_ALLuORFs.csv',
    'eBFP2': data_dir / 'color_controls/EBFP2.2023-01-22_CasE_ALLuORFs.csv',
    'mKate': data_dir / 'color_controls/mKate.2023-01-22_CasE_ALLuORFs.csv',
    ('eYFP', 'eBFP2', 'mKate'): data_dir / 'color_controls/ALL.2023-01-22_CasE_ALLuORFs.csv',
}

beads = Path(data_dir / 'beads/2023-01-22_CasE_ALLuORFs_BEADS_AL01_017.fcs')

cal = Calibration(
    blanks_file=blank,
    color_controls_files=controls,
    beads_file=beads,
    reference_protein='EBFP2',
    reference_channel='PACIFIC_BLUE_A',
    beads_reference_values=bc.calibration.SPHEROTECH_RCP_30_5a,
    use_channels=['FITC-A', 'PACIFIC_BLUE_A', 'PE_TEXAS_RED_A'],
)

cal.fit_TASBE()

# bleedthrough computation
fig, ax = du.mkfig(1, 1)
bc.calibration.plot_spectral_sig(cal._Calibration__bleedthrough_matrix, ax)

S = cal._Calibration__bleedthrough_matrix
bp = cal._Calibration__beads_params 

##────────────────────────────────────────────────────────────────────────────}}}

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
all_df = cal.controls[('EYFP', 'EBFP2', 'MKATE')]
df = all_df

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

Y = df.values
X = cal.apply(Y)


fig, axes = du.mkfig(4, 1, (8, 3))
for i, f in enumerate(channels):
    plot_fluo_distribution(axes[i], Y[:,i], axtitle=f)
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
        pick = jnp.all(masks == jnp.array([1,1,1]), axis=1)
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
plt.scatter(d,np.ones_like(T)*0.5, s=50, alpha=0.02, lw=0)
plt.show()

##────────────────────────────────────────────────────────────────────────────}}}
