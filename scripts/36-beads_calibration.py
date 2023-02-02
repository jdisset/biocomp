
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
from jax.scipy.stats import gaussian_kde
import biocomp.datautils as du
import optax
from pathlib import Path
from tqdm import tqdm
import biocomp.nodes as bn
import biocomp.compute as bcc
from mpl_toolkits.axes_grid1 import make_axes_locatable
import flowio
import matplotlib.pyplot as plt
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
import ott

plt.rcParams['figure.figsize'] = [7.0, 7.0]
plt.rcParams['figure.dpi'] = 300

# ────────────────────────────────────────────────────────────────────────────}}}


xpname = '2023-01-22_CasE_ALLuORFs'
beadfcs = ut.DEFAULT_XP_PATH / xpname / 'data/FCS_FILES/2023-01-22_CasE_ALLuORFs_BEADS_AJ01_018.fcs'
fcs_data = flowio.FlowData(beadfcs.as_posix())
channels = [fcs_data.channels[str(i + 1)]['PnN'] for i in range(fcs_data.channel_count)]
original_data = np.reshape(fcs_data.events, (-1, fcs_data.channel_count))

### {{{                         --     plot kdes     --
fig, axes = du.mkfig(3, 5, (5, 3))
axes = axes.flatten()
for i, ax in tqdm(enumerate(axes), total=len(axes)):
    x = np.linspace(0, 6, 1000)
    logdata = data[:, i + 1]
    logdata = jnp.log10(logdata[logdata > 0])
    kde = gaussian_kde(logdata, bw_method=0.025)
    y = kde(x)
    y /= y.max()
    ax.plot(x, y)

    ax.set_title(fcs_data.channels[str(i + 2)]['PnN'])
    ax.set_xlabel('log10')
    ax.set_ylabel('density')
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1.1)

# increase margins between subplots
fig.subplots_adjust(hspace=0.5, wspace=0.25)
fig.suptitle('FCS bead data (with peak amplification)', fontsize=10, y=0.95, x=0.45)

##────────────────────────────────────────────────────────────────────────────}}}##


# THE PLAN

# 1. get the calibrated values and generate points from them
# 2. define the cost function (OT cost)
# 3. define the gradient for a and b
# 4. update loop
# 5. plot the results

### {{{                     --     calibrated values     --
calibrated_beads_values = {
    'MEFL': [4648.0, 14631.0, 42313.0, 128924.0, 381106.0, 1006897.0, 2957538.0, 7435549.0],
    'MEPE': [2800.0, 8770.0, 25174.0, 74335.0, 219816.0, 548646.0, 1600005.0, 4255375.0],
    'MEPTR': [17518.0, 53950.0, 153641.0, 450901.0, 1283877.0, 3254513.0, 9431807.0, 24840372.0],
    'MEPerCP': [4354.0, 10032.0, 22473.0, 51739.0, 115599.0, 256091.0, 562684.0, 1201350.0],
    'MEPCY5.5': [2088.0, 6705.0, 20441.0, 66215.0, 211174.0, 645020.0, 2478405.0, 10603147.0],
    'MEeF710': [2259.0, 6862.0, 20129.0, 63316.0, 196680.0, 609247.0, 2451473.0, 11687960.0],
    'MEPCY7': [534.0, 1555.0, 4600.0, 14826.0, 47575.0, 161926.0, 706536.0, 3262715.0],
    'MEPCY5': [4354.0, 10032.0, 22473.0, 51739.0, 115599.0, 256091.0, 562684.0, 1201350.0],
    'MEPCY5.5 #2': [1999.0, 6228.0, 20393.0, 69124.0, 220232.0, 777840.0, 2521966.0, 8948283.0],
    'MEAX700': [6625.0, 17113.0, 58590.0, 199825.0, 629666.0, 2289301.0, 6504723.0, 17637305.0],
    'MEPCY7 #2': [457.0, 1334.0, 4666.0, 17500.0, 58774.0, 230324.0, 724800.0, 2057002.0],
    'MEAPC': [1170.0, 1970.0, 4669.0, 13757.0, 36757.0, 119744.0, 293242.0, 638909.0],
    'MEAX680': [6844.0, 17166.0, 56676.0, 195246.0, 622426.0, 333985.0, 6617776.0, 17561028.0],
    'MEAPCCY7': [1385.0, 3804.0, 13066.0, 47512.0, 151404.0, 542987.0, 1305924.0, 2540123.0],
    'PacBlue': [4450.0, 8342.0, 17587.0, 38906.0, 89281.0, 179989.0, 408481.0, 822214.0],
    'MEAMCY': [5974.0, 10513.0, 21623.0, 46727.0, 105630.0, 213273.0, 494395.0, 1072308.0],
    'MEPO': [391.0, 753.0, 1797.0, 4766.0, 13937.0, 39280.0, 156244.0, 652221.0],
    'MEQ605': [3133.0, 4774.0, 8471.0, 16359.0, 34465.0, 71375.0, 189535.0, 517591.0],
    'MEQ655': [1859.0, 2858.0, 5598.0, 11928.0, 27542.0, 66084.0, 202508.0, 650000.0],
    'MEQ705': [1695.0, 2858.0, 5598.0, 11928.0, 27542.0, 66084.0, 202508.0, 650000.0],
    'MEBV711': [1564.0, 3234.0, 5516.0, 12249.0, 29651.0, 71051.0, 197915.0, 596714.0],
    'MEQ800': [1358.0, 2085.0, 4301.0, 10037.0, 23446.0, 64511.0, 186279.0, 644779.0],
}
calib_names = list(calibrated_beads_values.keys())
calib_values = jnp.array([calibrated_beads_values[name] for name in calib_names]).T

# # from datasheet we know:
# units:
# MEFL: FITC
# MEPE: PE
# MEPTR: PE-TR
# MEPerCP: PerCP
# MEPCY5.5: PerCP-Cy5.5
# MEeF710: PerCP-eFluor
# MEPCY7: PE-Cy7
# MEPTR: PE-TR
# MEPCY5: PE-Cy5
# MEPCY5.5: PE-Cy5.5
# MEAX700: PE-Alexa 700
# MEPCY7: PE-Cy7
# MEAPC: APC
# MEAX680: Alexa 680
# MEAX700: Alexa 700
# MEAPCCY7: APC-CY7
# MEPB: PacBlue
# MEAMCY: AmCyan
# MEPO: Pacific Orange
# MEQ605: Qdot 605
# MEQ655: Qdot 655
# MEQ705: Qdot 705
# MEBV711: BV711
# MEQ800: Qdot 800

# and the channels we have for our data:
# 'Pacific Blue-A',
# 'AmCyan-A',
# 'FITC-A',
# 'PerCP-Cy5-5-A',
# 'PE-A',
# 'PE-Texas Red-A',
# 'APC-A',
# 'APC-Alexa 700-A',
# 'APC-Cy7-A'


# now we need to match the channels to the units:
channels_to_units = {
    'Pacific Blue-A': 'PacBlue',
    'AmCyan-A': 'MEAMCY',
    'FITC-A': 'MEFL',
    'PerCP-Cy5-5-A': 'MEPCY5.5',
    'PE-A': 'MEPE',
    'PE-Texas Red-A': 'MEPTR',
    'APC-A': 'MEAPC',
    'APC-Alexa 700-A': 'MEAX700',
    'APC-Cy7-A': 'MEAPCCY7',
}

# now let's just get the values for the channels we have:
color_channels = list(channels_to_units.keys())
calib_values = jnp.array(
    [calibrated_beads_values[channels_to_units[channel]] for channel in color_channels]
).T
data = original_data[:, [channels.index(channel) for channel in color_channels]]

logdata = jnp.log10(data - jnp.min(data, axis=0) + 1)

##────────────────────────────────────────────────────────────────────────────}}}



### {{{                    --     plot original data     --

# make scatter plots for every channel pair:
fig, axes = du.mkfig(9, 9, (2, 2))

for i, channel1 in tqdm(list(enumerate(color_channels))):
    for j, channel2 in enumerate(color_channels):
        ax = axes[j, i]
        ax.scatter(sample[:, i], sample[:, j], s=0.1, alpha=0.4)
        # also plot the calib points:
        ax.set_xlabel(channel1)
        ax.set_ylabel(channel2)
        ax.scatter(calib_values[:, i], calib_values[:, j], s=25, alpha=1, marker='x', color='red')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1, 1e8)
        ax.set_ylim(1, 1e8)
        ax.set_aspect('equal')

fig.tight_layout()
# increase margin between subplots:
fig.subplots_adjust(hspace=0.5, wspace=0.5)

##────────────────────────────────────────────────────────────────────────────}}}


### {{{                        --     testing ott     --


def reg_ot_cost(geom):
    out = sinkhorn.Sinkhorn()(linear_problem.LinearProblem(geom))
    return out.reg_ot_cost, out


target = calib_values
x = logsample / 7
y = jnp.log10(target) / 7


num_iter = 1500
dump_every = 10
learning_rate = 0.1

reg_ot_cost_vg = jax.jit(jax.value_and_grad(reg_ot_cost, has_aux=True))
# Run a naive, fixed stepsize, gradient descent on locations `x`.
ots = []
X = []


# scatter of X
fig, ax = du.mkfig(1, 1)
for i in range(len(X)):
    color = plt.cm.viridis(i / len(X))
    ax.scatter(X[i][:, 0], X[i][:, 1], s=0.1, alpha=1, color=color)
    ax.scatter(y[:, 0], y[:, 1], s=25, alpha=1, marker='x', color='red')


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                  --     random explorations...     --

choice = jax.random.choice(key, jnp.arange(data.shape[0]), shape=(1000,), replace=False)
logsample = logdata[choice]
source = logsample / 7
target = jnp.log10(calib_values) / 7
key = jax.random.PRNGKey(0)

num_iter = 10000
dump_every = 1000
learning_rate = 1e-3

target.shape


def loss_function(params):
    xx = params['a'] * source + params['b']
    geom = pointcloud.PointCloud(xx, target)
    out = sinkhorn.Sinkhorn()(linear_problem.LinearProblem(geom))
    return out.reg_ot_cost


def loss_function2(params):
    xx = params['a'] * source + params['b']
    geom = pointcloud.PointCloud(xx, target)
    out = sinkhorn.Sinkhorn()(linear_problem.LinearProblem(geom))
    return out.reg_ot_cost


lossf = jit(jax.value_and_grad(loss_function))

optimizer = optax.adam(learning_rate=learning_rate)
params = {
    'a': jax.random.uniform(key, shape=(x.shape[1],)),
    'b': jax.random.uniform(key, shape=(x.shape[1],)),
}
opt_state = optimizer.init(params)


def update(params, opt_state):
    loss, grad = lossf(params)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


all_params = []
for i in tqdm(list(range(0, num_iter + 1))):
    params, opt_state, loss = update(params, opt_state)
    if i % dump_every == 0:
        print(loss)
        all_params.append(params)

##
fig, ax = du.mkfig(1, 1)
xx = all_params[-1]['a'] * source + all_params[-1]['b']
ax.scatter(xx[:, 0], xx[:, 3], s=0.1, alpha=1, color='b')
ax.scatter(target[:, 2], target[:, 3], s=25, alpha=1, marker='x', color='red')


##
from jax import jit, vmap
from evosax import CMA_ES


def flatten_params(params):
    leaves, treedef = jax.tree_util.tree_flatten(params)
    flat_leaves = [l.flatten() for l in leaves]
    shapes = [l.shape for l in leaves]
    flat_params = np.concatenate(flat_leaves)
    return flat_params, (shapes, treedef)


def unflatten_params(flat_params, pdef):
    shapes, treedef = pdef
    splits = np.cumsum([np.prod(s) for s in shapes], dtype=np.int32)
    leaves = []
    start = 0
    for sp, sh in zip(splits, shapes):
        leaves.append(flat_params[start:sp].reshape(sh))
        start = sp
    params = jax.tree_util.tree_unflatten(treedef, leaves)
    return params


def fitness(flat_params):
    p = unflatten_params(flat_params, pdef)
    l = loss_function(p)
    return l


vm_fitness = jax.jit(jax.vmap(fitness))

params = {
    'a': jax.random.uniform(key, shape=(x.shape[1],)),
    'b': jax.random.uniform(key, shape=(x.shape[1],)),
}
flat_params, pdef = flatten_params(params)


rng = jax.random.PRNGKey(3)
strategy = CMA_ES(popsize=250, num_dims=flat_params.shape[0])
es_params = strategy.default_params.replace(init_min=0, init_max=1)
state = strategy.initialize(rng, es_params)

num_generations = 40
fitnesses = []

for t in tqdm(list(range(num_generations))):
    rng, rng_gen, rng_eval = jax.random.split(rng, 3)
    ps, state = strategy.ask(rng_gen, state, es_params)
    fitness = vm_fitness(ps)
    state = strategy.tell(ps, fitness, state, es_params)
    fitnesses.append(fitness)

best_params = unflatten_params(state.best_member, pdef)

##
newparams = best_params.copy()
jit(loss_function)(newparams)

##

# newparams['a'] = newparams['a'].at[0].set(0.88)
# newparams['b'] = newparams['b'].at[0].set(0.32)
# newparams['a'] = newparams['a'].at[1].set(0.64)
# newparams['b'] = newparams['b'].at[1].set(0.25)


fig, ax = du.mkfig(1, 1)
xx = newparams['a'] * source + newparams['b']
ax.scatter(xx[:, 0], xx[:, 1], s=0.1, alpha=1, color='b')
ax.scatter(target[:, 0], target[:, 1], s=25, alpha=1, marker='x', color='red')


##

choice = jax.random.choice(key, jnp.arange(data.shape[0]), shape=(10000,), replace=False)
logsample = logdata[choice]
source = logsample / 7
# source = newparams['a'] * source + newparams['b']
target = jnp.log10(calib_values) / 7


geom = pointcloud.PointCloud(source, target)
# Define a linear problem with that cost structure.
ot_prob = linear_problem.LinearProblem(geom)
# Create a Sinkhorn solver
solver = sinkhorn.Sinkhorn()
# Solve OT problem
ot = solver(ot_prob)
# The out object contains many things, among which the regularized OT cost
print(
    " Sinkhorn has converged: ",
    ot.converged,
    "\n",
    "Error upon last iteration: ",
    ot.errors[(ot.errors > -1)][-1],
    "\n",
    "Sinkhorn required ",
    jnp.sum(ot.errors > -1),
    " iterations to converge. \n",
    "Entropy regularized OT cost: ",
    ot.reg_ot_cost,
    "\n",
    "OT cost (without entropy): ",
    jnp.sum(ot.matrix * ot.geom.cost_matrix),
)

##

##
# cluster kmeans on source (use scipy
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=target.shape[0], random_state=0).fit(source)
source_assignments = kmeans.labels_
# scatter
fig, ax = du.mkfig(1, 1)
# scatter source with color based on assignment
# first we group by assignment
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
for i in range(P.shape[1]):
    ax.scatter(
        source[source_assignments == i, 0],
        source[source_assignments == i, 2],
        s=0.1,
        alpha=1,
        color=colors[i],
    )

# scatter target, same color (from assignment
for i in range(P.shape[1]):
    ax.scatter(target[i, 6], target[i, 0], s=25, alpha=1, marker='x', color=colors[i])

# count where any feature is below zero:
# sample 1000 random points:
key = jax.random.PRNGKey(0)
choice = jax.random.choice(key, jnp.arange(data.shape[0]), shape=(25,), replace=False)

logsample = logdata[choice]
source = logsample / 7
# source = newparams['a'] * source + newparams['b']
target = jnp.log10(calib_values) / 7


geom = pointcloud.PointCloud(source, target)
# Define a linear problem with that cost structure.
ot_prob = linear_problem.LinearProblem(geom)
# Create a Sinkhorn solver
solver = sinkhorn.Sinkhorn()
# Solve OT problem
ot = solver(ot_prob)
# The out object contains many things, among which the regularized OT cost

print(
    " Sinkhorn has converged: ",
    ot.converged,
    "\n",
    "Error upon last iteration: ",
    ot.errors[(ot.errors > -1)][-1],
    "\n",
    "Sinkhorn required ",
    jnp.sum(ot.errors > -1),
    " iterations to converge. \n",
    "Entropy regularized OT cost: ",
    ot.reg_ot_cost,
    "\n",
    "OT cost (without entropy): ",
    jnp.sum(ot.matrix * ot.geom.cost_matrix),
)


P = ot.matrix
# find assignments per point
source_assignments = np.argmax(P, axis=1)
source_assignments
# plot P heatmap
fig, ax = du.mkfig(1, 1, (2, 50))
ax.imshow(P, cmap='viridis')

##

# source_assignment contains a digit from 0 to 9
fig, ax = du.mkfig(1, 1)
# scatter source with color based on assignment
# first we group by assignment
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
for i in range(P.shape[1]):
    ax.scatter(
        source[source_assignments == i, 0],
        source[source_assignments == i, 1],
        s=0.1,
        alpha=1,
        color=colors[i],
    )

# scatter target, same color (from assignment
for i in range(P.shape[1]):
    ax.scatter(target[i, 0], target[i, 1], s=25, alpha=1, marker='x', color=colors[i])


##


def w_function(x, a, b, a_steepness=10, b_steepness=100):
    y1 = 1 / (1 + jnp.exp(-a_steepness * (x - a)))
    y2 = 1 / (1 + jnp.exp(-b_steepness * (x - b)))
    return y1 - y2


thresholds = (2.3, 5.3)
# plot kde of logdata, per channel
fig, axes = du.mkfig(logdata.shape[1], 1, (7, 2))
for i in range(logdata.shape[1]):
    ax = axes[i]
    kde = gaussian_kde(logdata[:, i], bw_method=0.005)
    x = np.linspace(0, 8, 2000)
    densities = kde(x)
    densities = densities / np.max(densities)
    w = w_function(x, thresholds[0], thresholds[1])
    chname = color_channels[i]
    # vline at thresholds
    ax.plot(x, densities, label=chname, color='k', linewidth=1)
    ax.plot(x, w, label='w', color='r', linewidth=1)
    ax.axvline(thresholds[0], color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(thresholds[1], color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.legend()

##

# weight of each observation in each channel:

weights = vmap(w_function, in_axes=(0, None, None))(logsample, thresholds[0], thresholds[1])
fig, ax = du.mkfig(1, 1, (2, 20))
im = ax.imshow(weights, cmap='viridis')
ax.set_xlabel('channel')
ax.set_ylabel('sample')
ax.set_title('weights')
# aspect ratio
ax.set_aspect('auto')
# smaller colorbar, on the bottom, horizontal
# cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.05, pad=0.05)
# cbar.ax.set_ylabel('weight')
# fig.tight_layout()


##


# count where any feature is below zero:
# sample 1000 random points:
key = jax.random.PRNGKey(0)
choice = jax.random.choice(key, jnp.arange(data.shape[0]), shape=(50000,))

logsample = logdata[choice]
source = logsample / 7
# source = newparams['a'] * source + newparams['b']
target = jnp.log10(calib_values) / 7


def vote(source, target):
    s = source[:, None]
    t = target[:, None]
    geom = pointcloud.PointCloud(s, t)
    ot_prob = linear_problem.LinearProblem(geom)
    ot = sinkhorn.Sinkhorn()(ot_prob)
    return ot.matrix


votes = jit(vmap(vote, in_axes=(1, 1)))(source, target)
# dims = (CHANNEL, SAMPLE, BEAD)

# sum over channels
vmat = jnp.sum(votes, axis=0)


# fig, ax = du.mkfig(1, 1, (2,20))
# im = ax.imshow(vmat, cmap='viridis')
# ax.set_xlabel('bead')
# ax.set_ylabel('sample')
# ax.set_title('non-weighted vote')
# ax.set_aspect('auto')

weights = vmap(w_function, in_axes=(0, None, None))(logsample, thresholds[0], thresholds[1])
tweights = weights.T
votes.shape
tweights.shape
weighted_votes = votes * tweights[:, :, None]
wvmat = jnp.sum(weighted_votes, axis=0)

# fig, ax = du.mkfig(1, 1, (2,20))
# im = ax.imshow(wvmat, cmap='viridis')
# ax.set_xlabel('bead')
# ax.set_ylabel('sample')
# ax.set_title('weighted vote')
# ax.set_aspect('auto')


##
assignments = np.argmax(wvmat, axis=1)
# scatter source with color based on assignment
# first we group by assignment
fig, ax = du.mkfig(1, 1, (2, 2))
for i in range(vmat.shape[1]):
    ax.scatter(
        source[assignments == i, 0],
        source[assignments == i, 1],
        s=0.1,
        alpha=1,
        color=colors[i],
    )

# scatter target, same color (from assignment
for i in range(vmat.shape[1]):
    ax.scatter(target[i, 0], target[i, 1], s=25, alpha=1, marker='x', color=colors[i])

##

# now compute the weighted average for each bead
# we just need to use wvmat as the weights for the average coordinate of each point according to each bead

# wvmat dims = (SAMPLE, BEAD)
# source dims = (SAMPLE, CHANNEL)

# threshold the weighhts
wvmat_norm = wvmat / jnp.sum(wvmat, axis=1)[:, None]
confidence_threshold = jnp.quantile(wvmat_norm, 0.9)
wvmat_th = jnp.where(wvmat_norm > confidence_threshold, wvmat_norm, 0)

ax0 = 0
ax1 = 5

beadcentroids = vmap(jnp.average, in_axes=(None, None, 1))(source, 0, wvmat_th)
# scatter bead centroids and target
fig, ax = du.mkfig(1, 1, (5, 5))
# scater points in black
ax.scatter(source[:, ax0], source[:, ax1], s=1, alpha=0.05, color='k', linewidth=0)
colors = plt.cm.tab10(np.linspace(0, 1, vmat.shape[1]))

for i in range(vmat.shape[1]):
    ax.scatter(
        beadcentroids[i, ax0],
        beadcentroids[i, ax1],
        s=25,
        alpha=1,
        color=colors[i],
    )

# scatter target, same color (from assignment
for i in range(vmat.shape[1]):
    ax.scatter(target[i, ax0], target[i, ax1], s=25, alpha=1, marker='x', color=colors[i])

# add labels
ax.set_xlabel(color_channels[ax0])
ax.set_ylabel(color_channels[ax1])




##────────────────────────────────────────────────────────────────────────────}}}


data.shape

ax0 = 2
ax1 = 8

thresholds = (2.3, 5.3)
confidence_threshold_quantile = 0.9
key = jax.random.PRNGKey(0)
# choice = jax.random.choice(key, jnp.arange(data.shape[0]), shape=(25,), replace=False)
logsample = logdata
source = logsample / 7
target = jnp.log10(calib_values) / 7
newminval = jnp.min(target, axis=0) - 0.03
target = jnp.hstack((newminval[:,None], target.T)).T
target.shape



@jit
def compute_centroids(
    source,
    target,
    thresholds=(2.3, 5.3),
    confidence_threshold_quantile=0.9,
    confidence_threshold_absolute=0.1,
    left_steepness=10,
    right_steepness=100,
):
    def w_function(x, a, b):
        y1 = 1 / (1 + jnp.exp(-left_steepness * (x - a)))
        y2 = 1 / (1 + jnp.exp(-right_steepness * (x - b)))
        return y1 - y2

    # weight of each observation in each channel:
    weights = vmap(w_function, in_axes=(0, None, None))(logsample, thresholds[0], thresholds[1])

    def vote(source, target):
        s = source[:, None]
        t = target[:, None]
        geom = pointcloud.PointCloud(s, t)
        ot_prob = linear_problem.LinearProblem(geom)
        ot = sinkhorn.Sinkhorn()(ot_prob)
        return ot.matrix

    votes = vmap(vote, in_axes=(1, 1))(source, target)
    # votes = votes / jnp.sum(votes, axis=1)[:, None]

    weights = vmap(w_function, in_axes=(0, None, None))(logsample, thresholds[0], thresholds[1]).T

    weighted_votes = votes * weights[:, :, None]

    weighted_votes.shape

    wvmat = jnp.sum(weighted_votes, axis=0) / jnp.sum(weights, axis=0)[:, None]

    # threshold the weighhts
    wvmat_norm = wvmat / jnp.sum(wvmat, axis=1)[:, None]
    confidence_threshold = jnp.clip(jnp.quantile(wvmat_norm, confidence_threshold_quantile), confidence_threshold_absolute)
    wvmat_th = jnp.where(wvmat_norm > confidence_threshold, wvmat_norm, 0)

    beadcentroids = vmap(jnp.average, in_axes=(None, None, 1))(source, 0, wvmat_th)
    return beadcentroids


centroids = compute_centroids(
    source,
    target,
    thresholds=(3,5),
    confidence_threshold_quantile=0.9,
    confidence_threshold_absolute=0.3,
)

centroids.shape

# scatter bead centroids and target
fig, ax = du.mkfig(1, 1, (5, 5))
# scater points in black
ax.scatter(source[:, ax0], source[:, ax1], s=1, alpha=0.05, color='k', linewidth=0)
colors = plt.cm.tab10(np.linspace(0, 1, centroids.shape[0]))
for i in range(centroids.shape[0]):
    ax.scatter(
        centroids[i, ax0],
        centroids[i, ax1],
        s=20,
        alpha=1,
        color=colors[i],
    )

# scatter target, same color (from assignment
for i in range(centroids.shape[0]):
    ax.scatter(target[i, ax0], target[i, ax1], s=25, alpha=1, marker='x', color=colors[i])

# add labels
ax.set_xlabel(color_channels[ax0])
ax.set_ylabel(color_channels[ax1])
