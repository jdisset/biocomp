## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     imports     --
# ···············································································
import biocomp as bc
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import biocomp.utils as bu
import scriptutils as ut
import jax
from mpl_toolkits.axes_grid1 import make_axes_locatable
from jax import jit, vmap, grad, value_and_grad
import jax.numpy as jnp
import biocomp.datautils as du
import optax
from tqdm import tqdm
import biocomp.nodes as bn
import biocomp.compute as bcc
from jax.scipy.stats import gaussian_kde


import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10.0, 10.0]
plt.rcParams['figure.dpi'] = 200


def mkfig(rows, cols, size=(6, 6)):
    fig, ax = plt.subplots(rows, cols, figsize=(cols * size[0], rows * size[1]))
    return fig, ax


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                 --     data generation functions     --
# ···············································································


def mixture_gen(n_samples, n_blobs, seed, ndim=2):
    rng = jax.random.PRNGKey(seed)
    weights = jax.random.uniform(rng, (n_blobs,), minval=1, maxval=5)
    weights /= weights.sum()
    lens = list((weights * n_samples).astype(int))
    lens[-1] += n_samples - sum(lens)

    k0, k1, k2, k3 = jax.random.split(rng, 4)
    mean = jax.random.uniform(k0, (n_blobs, ndim), minval=-1, maxval=1) * 100
    sigma_1d = jax.random.uniform(k1, (n_blobs,), minval=5, maxval=30)
    sigma = jnp.expand_dims(sigma_1d, axis=-1) * jnp.ones((1, ndim))

    keys = jax.random.split(k2, n_blobs)

    def gen_blob(k, m, s, n):
        return jax.random.normal(k, (n, ndim)) * s + m

    X = jnp.vstack([gen_blob(k, m, s, n) for k, m, s, n in zip(keys, mean, sigma, lens)])
    X = jax.random.permutation(k3, X, axis=0) / 100.0
    return X


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                    --     plotting functions     --
# ···············································································


def mkfig(rows, cols, size=(6, 6)):
    fig, ax = plt.subplots(rows, cols, figsize=(cols * size[0], rows * size[1]))
    return fig, ax

def plot_over_1d(f, vmin=-2, vmax=2, resolution=200, title=None, ax=None):
    x = jnp.linspace(vmin, vmax, resolution)
    y = f(x)
    if ax is None:
        _, ax = mkfig(1, 1)
    ax.plot(x, y)
    if title is not None:
        ax.set_title(title)
    return ax


vmin = -2
vmax = 2
resolution = 100


def plot_over_2d(
    f, vmin=-2, vmax=2, resolution=200, title=None, ax=None, add_colorbar=True, value_range=None
):
    x = jnp.linspace(vmin, vmax, resolution)
    xy = jnp.vstack(map(jnp.ravel, jnp.meshgrid(x, x))).T
    z = f(xy.T).reshape((resolution, resolution))
    if ax is None:
        _, ax = mkfig(1, 1)
    if value_range is None:
        value_range = (z.min(), z.max())
    im = ax.pcolormesh(x, x, z, vmin=value_range[0], vmax=value_range[1], cmap='inferno')
    if add_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        cax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.0e}".format(x)))
    ax.set_aspect('equal')
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    if title is not None:
        ax.set_title(title)
    return ax


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     data generation     --
# ···············································································

n_samples = 10000
n_blobs = 2
seed = 2340
X = mixture_gen(n_samples, n_blobs, seed)
# X = jnp.hstack(
    # [mixture_gen(n_samples, n_blobs, seed, ndim=1), mixture_gen(n_samples, n_blobs, seed*32, ndim=1)]
# )
jpdf_target = gaussian_kde(X.T, 'silverman')
plot_over_2d(jpdf_target, title='KDE')


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                            --     cdfs     --
# ···············································································


def cdf(S, x):
    def lt(s, x):
        return jnp.all(s <= x)
    return jnp.mean(jax.vmap(lt, in_axes=(0, None))(S, x))



# compute actual invere cdf and try if it works??? to just aim for that LOL
SHARPNESS = 50

@jit
def heavyside(x, sharpness=SHARPNESS):
    return jax.nn.sigmoid(x * sharpness)


def cdf_d(S, x, sharpness=SHARPNESS):
    assert S.ndim == 1
    smin, smax = S.min(), S.max()
    S = (S - smin) / (smax - smin)
    x = (x - smin) / (smax - smin)
    def lt_diff(s, x):
        return heavyside(x - s, sharpness=sharpness)
    return jnp.mean(jax.vmap(lt_diff, in_axes=(0, None))(S, x))

def jcdf_d(S, x, sharpness=SHARPNESS):
    smin, smax = S.min(), S.max()
    S = (S - smin) / (smax - smin)
    x = (x - smin) / (smax - smin)
    def lt_diff(s, x):
        return jnp.prod(heavyside(x - s, sharpness=sharpness))

    return jnp.mean(jax.vmap(lt_diff, in_axes=(0, None))(S, x))

vcdf = jax.vmap(cdf, in_axes=(None, 0))
vcdf_d = jax.vmap(cdf_d, in_axes=(None, 0))

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                    --     projections     --
# ···············································································

pvecs = jnp.array([[1, 0], [0, 1]])
# pvecs = jnp.array([[1, 1], [-1, 1]])
# normalize all pvecs:
pvecs = pvecs / jnp.linalg.norm(pvecs, axis=-1, keepdims=True)

X_proj = jnp.dot(X, pvecs.T)
mpdf_target = [gaussian_kde(X_proj[:, i].T) for i in range(len(pvecs))]

all_cdfs = [partial(vcdf, X_proj[:, i]) for i in range(len(pvecs))]
all_cdfs_d = [partial(vcdf_d, X_proj[:, i]) for i in range(len(pvecs))]


def transpose_vjcdf(x):
    return vcdf(X, x.T)


def prod_cdf(x):
    cdfs = jnp.array([c(xx) for c, xx in zip(all_cdfs, x)])
    pr = jnp.prod(cdfs, axis=0)
    jc = transpose_vjcdf(x)
    diffs = jnp.abs(jc - pr)
    return diffs


def plot_proj_line(ax, pvec, sp=None, f=None, npoints=5000, scaling=0.5):
    ax.plot(
        [-pvec[0] * 1000, pvec[0] * 1000],
        [-pvec[1] * 1000, pvec[1] * 1000],
        'w--',
        linewidth=0.5,
        alpha=1,
    )
    if f is not None:
        pnormal = jnp.array([-pvec[1], pvec[0]])
        vx = jnp.linspace(vmin * 1.5, vmax * 1.5, npoints)
        vy = f(vx) * scaling
        pnormal = jnp.array([-pvec[1], pvec[0]])
        px = pvec[:, None] * vx
        py = pnormal[:, None] * vy + px
        ax.scatter(py[0, :], py[1, :], s=0.01)


fig, axes = mkfig(3, 1)
ax = axes[0]
plot_over_2d(jpdf_target, title='Marginal PDFs over joint PDF', ax=ax)
for pvec, sp, k in zip(pvecs, X_proj.T, mpdf_target):
    plot_proj_line(ax, pvec, sp=sp, f=k)

ax = axes[1]
plot_over_2d(transpose_vjcdf, title='Marginal CDFs over joint CDF', ax=ax)
for pvec, sp, c in zip(pvecs, X_proj.T, all_cdfs):
    plot_proj_line(ax, pvec, sp=sp, f=c, scaling=1.0)

ax = axes[2]
plot_over_2d(
    prod_cdf, title='Difference of jCDF with product of marginal CDFs', ax=ax, value_range=(0, 0.1)
)


plt.show()

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                           --     model    --
# ···············································································
H_SIZE = 64
N_LAYERS = 5


def get_param(name, init, params, shared=False):
    if name not in params:
        print(f'initializing {name}')
        params[name] = init()
    return params[name]


def dense_layer(input_values, output_size, get_param, key, name):
    input_size = 1 if input_values.shape == () else input_values.shape[0]
    w = get_param(f'{name}_w', init=bu.he_initializer(key, (input_size, output_size)), shared=True)
    b = get_param(f'{name}_b', init=lambda: jnp.zeros((output_size,)), shared=True)
    try:
        return jnp.dot(input_values, w) + b
    except ValueError as e:
        msg = f'input_values.shape: {input_values.shape}\n'
        msg += f'w.shape: {w.shape}\n'
        msg += f'b.shape: {b.shape}\n'
        raise ValueError(msg) from e


activation = jax.nn.leaky_relu


def model(params, z, key):
    k0, k1, k2, k3, kn = jax.random.split(key, 5)
    get_p = partial(get_param, params=params)
    # we go from z to H_size to output_shape
    x = dense_layer(z, H_SIZE, get_p, k0, 'm_dense0')
    x = activation(x)
    for i in range(N_LAYERS - 1):
        x = dense_layer(x, H_SIZE, get_p, k1, f'm_dense{i+1}')
        x = activation(x)
        # x = jnp.concatenate([x.flatten(), z])
    return dense_layer(x, Y.shape[1], get_p, k3, 'm_dense_out')
vm = vmap(model, in_axes=(None, 0, None))

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                           --     train     --
# ···············································································


Y = X
Y_proj = jnp.dot(Y, pvecs.T)
N = 1000  # batch size, basically
zsize = Y.shape[1]

params = {}
rng_key = jax.random.PRNGKey(1424)

model(params, jnp.zeros(zsize), rng_key)

opt = optax.amsgrad(learning_rate=5e-4)
opt_state = opt.init(params)

unit_hypercube_samples = jax.random.uniform(rng_key, (50000, Y.shape[1]), minval=0, maxval=1)
Hproj = jnp.dot(unit_hypercube_samples, pvecs.T)


def huber_loss(y, yhat, tau, delta=1.0):
    residual = y - yhat
    loss = np.where(np.abs(residual) <= delta, 
                   (1/2) * residual ** 2,
                   delta * (np.abs(residual) - (1/2) * delta))
    return tau * loss

def loss_fn(params, key):
    zs = jax.random.uniform(key, (N, Y.shape[1]), minval=0, maxval=1)
    yhat_m = vm(params, zs, key)
    all_quantiles = vmap(partial(jnp.quantile), in_axes=(1, 1))(Y_proj, zs)
    err = jnp.sqrt(jnp.sum((all_quantiles.T - yhat_m) ** 2, axis=1))

    # cdf_y(y) = cdf_y(z) = z
    # jcdf

    # zproj = jnp.dot(zs, pvecs.T)
    # yproj = jnp.dot(yhat, pvecs.T)
    # z_cdf = vmap(vcdf_d, in_axes=(1, 1))(Hproj, zproj).T
    # all_quantiles = vmap(partial(jnp.quantile, method='linear'), in_axes=(1, 1))(Y_proj, z_cdf)
    # cdfs = vmap(vcdf_d, in_axes=(1,1))(Y_proj, yhat).T
    # err = jnp.sqrt(jnp.sum((cdfs - zs) ** 2, axis=1))

    return jnp.mean(err)

# there should be a way to learn a function C (the copula) that maps from (u,v) to (x,y)

@jit
def update(params, opt_state, key):
    loss, grads = value_and_grad(loss_fn)(params, key)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


n_epochs = 1500

losses = []
for epoch in range(n_epochs):
    rng_key, subkey = jax.random.split(rng_key, 2)
    params, opt_state, loss = update(params, opt_state, rng_key)
    losses.append(loss)
    print(f'Epoch {epoch} loss: {loss}')

plt.semilogy(losses)

n_gen_samples = 5000
zs = jax.random.uniform(rng_key, (N, Y.shape[1]), minval=0, maxval=1)

X_gen = vm(params, zs, rng_key)
jnp.quantile(Y_proj[:, 0], zs[:, 0])
jpdf_gen = gaussian_kde(X_gen.T, bw_method='silverman')
mpdf_gen = [gaussian_kde(X_gen[:, i].T) for i in range(len(pvecs))]

fig, axes = mkfig(2, 1)
ax = axes[0]
plot_over_2d(jpdf_target, title='[TARGET] Marginal PDFs over joint PDF', ax=ax)
for pvec, sp, k in zip(pvecs, X_proj.T, mpdf_target):
    plot_proj_line(ax, pvec, sp=sp, f=k)

ax = axes[1]
plot_over_2d(jpdf_gen, title='[LEARNT] Marginal PDFs over joint PDF', ax=ax)
for pvec, sp, k in zip(pvecs, X_proj.T, mpdf_gen):
    plot_proj_line(ax, pvec, sp=sp, f=k)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

### {{{                          --     train2     --

def qmodel(params, x, z, key):
    k0, k1, k2, k3, kn = jax.random.split(key, 5)
    get_p = partial(get_param, params=params)
    # we go from z to H_size to output_shape
    x = dense_layer(z, H_SIZE, get_p, k0, 'm_dense0')
    x = activation(x)
    for i in range(N_LAYERS - 1):
        x = dense_layer(x, H_SIZE, get_p, k1, f'm_dense{i+1}')
        x = activation(x)
        # x = jnp.concatenate([x.flatten(), z])
    return dense_layer(x, Y.shape[1], get_p, k3, 'm_dense_out')
vmq = vmap(model, in_axes=(None, 0, None))

Y = X
N = 1000  # batch size, basically
zsize = Y.shape[1]
params = {}
rng_key = jax.random.PRNGKey(1424)
key = rng_key
model(params, jnp.zeros(zsize), rng_key)
opt = optax.amsgrad(learning_rate=5e-4)
opt_state = opt.init(params)


def quantile_loss(y_true, y_pred, q):
    e = y_true - y_pred
    return jnp.maximum(q * e, (q - 1.0) * e)

def loss_fn(params, y, key):
    z = jax.random.uniform(key, (N, ), minval=0, maxval=1)
    zs = jnp.hstack([y[:,0], z])

    yhat_m = vm(params, zs, key)
    
    # losses = vmap(quantile_loss,in_axes=(0, 0, None))(y, yhat_m, 0.5)
    losses = vmap(quantile_loss,in_axes=(0, 0, 0))(y[:, yhat_m, zs[:,0])

    return jnp.mean(losses) 


@jit
def update(params, opt_state, y, key):
    loss, grads = value_and_grad(loss_fn)(params, y, key)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

n_epochs = 1000

# random with replacement
y_batches = jax.random.choice(rng_key, Y, shape=(n_epochs, N))
y = y_batches[0]

losses = []
for yb in y_batches:
    rng_key, subkey = jax.random.split(rng_key, 2)
    params, opt_state, loss = update(params, opt_state, yb, rng_key)
    losses.append(loss)
    print(f'Loss: {loss}')

plt.semilogy(losses)

n_gen_samples = 5000
zs = jax.random.uniform(rng_key, (N, Y.shape[1]), minval=0, maxval=1)

X_gen = vm(params, zs, rng_key)
jnp.quantile(Y_proj[:, 0], zs[:, 0])
jpdf_gen = gaussian_kde(X_gen.T, bw_method='silverman')
mpdf_gen = [gaussian_kde(X_gen[:, i].T) for i in range(len(pvecs))]

fig, axes = mkfig(2, 1)
ax = axes[0]
plot_over_2d(jpdf_target, title='[TARGET] Marginal PDFs over joint PDF', ax=ax)
for pvec, sp, k in zip(pvecs, X_proj.T, mpdf_target):
    plot_proj_line(ax, pvec, sp=sp, f=k)

ax = axes[1]
plot_over_2d(jpdf_gen, title='[LEARNT] Marginal PDFs over joint PDF', ax=ax)
for pvec, sp, k in zip(pvecs, X_proj.T, mpdf_gen):
    plot_proj_line(ax, pvec, sp=sp, f=k)



##────────────────────────────────────────────────────────────────────────────}}}



## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                            --     BS     --
#···············································································

u = jax.random.uniform(rng_key, (1000, 1), minval=0, maxval=1)
z = jnp.hstack([u,u])
z = jax.random.uniform(rng_key, (10000, 2), minval=0, maxval=1)
quantiles = vmap(partial(jnp.quantile), in_axes=(1, 1))(Y_proj, z).T

# scatter plot of quantiles:
fig, ax = plt.subplots()
ax.scatter(quantiles[:, 0], quantiles[:, 1], s=2, alpha=0.5)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────



## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     random thoughs     --
# ···············································································

# dependant is really just a function of the independent variable
# maybe we could learn that?
# like, we learn the independant distributions, and we also learn
# to combine in a way that reproduces the jointcdf somehow?

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     old code     --
# ···············································································

# scatter plot of X:
plt.scatter(X[:, 0], X[:, 1], s=50, c='k', alpha=0.01, edgecolors='none')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.show()
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
