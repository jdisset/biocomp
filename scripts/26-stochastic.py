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
from jax import jit, vmap, grad, value_and_grad
import jax.numpy as jnp
import biocomp.datautils as du
import optax
from tqdm import tqdm
import biocomp.nodes as bn
import biocomp.compute as bcc


import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10.0, 10.0]
plt.rcParams['figure.dpi'] = 200

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                       --     generate data     --
# ···············································································
def generate_probability_distribution_jax(n_samples, n_blobs, rng, ndim=2):
    X = jnp.empty((0, ndim))
    weights = jax.random.uniform(rng, (n_blobs,), minval=1, maxval=4)
    weights /= weights.sum()
    lens = (weights * n_samples).astype(int)
    keys = jax.random.split(rng, n_blobs)

    for l, k in zip(lens, keys):
        k0, k1, k2 = jax.random.split(k, 3)
        mean = jax.random.uniform(k0, (ndim,), minval=-2, maxval=2) * 10
        covariance = jax.random.normal(k1, (ndim, ndim)) * 0.75 + 1.0
        covariance = jnp.eye(ndim) * (jax.random.uniform(k1) * 10.0 + jnp.abs(covariance))
        samples = jax.random.multivariate_normal(k2, mean, covariance, shape=(l,))
        X = jnp.concatenate((X, samples), axis=0)

    return X


# Set the number of samples to generate for each blob
n_samples = 5000

# Set the number of blobs to generate
n_blobs = 4

key = jax.random.PRNGKey(123921)
rng = key
# Generate the probability distribution
ndim = 1
S = generate_probability_distribution_jax(n_samples, n_blobs, key, ndim=ndim)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                       --     plot density     --
# ···············································································

from jax.scipy.stats import gaussian_kde
import scipy.stats

kde = gaussian_kde(S.T, bw_method='silverman')
# kde_sc = scipy.stats.gaussian_kde(S.T, bw_method='silverman')

def nd_digitize(x, grid):
    x = jnp.asarray(x)
    assert x.ndim == 1
    grid = jnp.asarray(grid)
    return jnp.stack([jnp.digitize(x[i], grid[i]) for i in range(x.shape[0])], axis=0)

SHARPNESS = 100

import ott
import ott.tools

# softranks = jax.jit(ott.tools.soft_sort.ranks)

def rotate2d(x, theta):
    c, s = jnp.cos(theta), jnp.sin(theta)
    R = jnp.array([[c, -s], [s, c]])
    return jnp.dot(x, R)

# theta = jnp.pi / 4
# S_rot = rotate2d(S, theta)

# S_back = rotate2d(S_rot, -theta)
# jnp.allclose(S_back, S, atol=1e-4)

@jit
def heavyside(x, sharpness=SHARPNESS):
    return jax.nn.sigmoid(x * sharpness)


def jcdf_m(x, S):
    def lt(s, x):
        return jnp.all(s <= x)

    return jnp.mean(jax.vmap(lt, in_axes=(0, None))(S, x))


# differentiable version. Returns the product
def jcdf(x, S, sharpness=SHARPNESS):
    smin, smax = S.min(axis=0), S.max(axis=0)
    S = (S - smin) / (smax - smin)
    x = (x - smin) / (smax - smin)

    def lt_diff(s, x):
        return jnp.prod(heavyside(x - s, sharpness=sharpness))

    return jnp.mean(jax.vmap(lt_diff, in_axes=(0, None))(S, x))


def jcdf_nd(x, S, sharpness=SHARPNESS):
    # normalize x so that sharpness has constant meaning
    smin, smax = S.min(axis=0), S.max(axis=0)
    S = (S - smin) / (smax - smin)
    x = (x - smin) / (smax - smin)

    def lt_diff(s, x):
        return heavyside(x - s, sharpness=sharpness)

    return jnp.mean(jax.vmap(lt_diff, in_axes=(0, None))(S, x), axis=0)


def jcdf_nd_ott(x, S):
    return softranks(S, x, axis=0) / S.shape[0]


def jcdf_ax(x, S, axis, sharpness=SHARPNESS):
    def lt_diff(s, x):
        return heavyside(x[axis] - s[axis], sharpness=sharpness)

    return jnp.mean(jax.vmap(lt_diff, in_axes=(0, None))(S, x))


vjcdf = jit(vmap(jcdf, in_axes=(0, None)))
vjcdf_ax = jit(vmap(jcdf_ax, in_axes=(0, None, None)))
vjcdf_nd = jit(vmap(jcdf_nd, in_axes=(0, None)))
vjcdf_nd_ott = jit(vmap(jcdf_nd_ott, in_axes=(0, None)))
vjcdf_m = jit(vmap(jcdf_m, in_axes=(0, None)))
grad_vjcdf = jit(vmap(grad(jcdf), in_axes=(0, None)))


# vjcdf_nd_ott(x, S)

kde = gaussian_kde(S.T, bw_method='silverman')

if ndim == 2:
    # x = jnp.linspace(-1, 1, 128)
    x = jnp.linspace(S.min() - 1, S.max() + 1, 128)
    y = x
    Xgrid, Ygrid = np.meshgrid(x, y)
    xy = np.vstack([Xgrid.ravel(), Ygrid.ravel()]).T

    kde_eval = kde.evaluate(xy.T).reshape(Xgrid.shape)

    cdf_m = vjcdf_m(xy, S).reshape(Xgrid.shape)
    cdf = vjcdf(xy, S).reshape(Xgrid.shape)

    fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    ax[0,0].imshow(
        cdf_m,
        origin='lower',
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap='inferno',
        vmin=0,
        vmax=1,
    )
    ax[0,0].set_title('true ecdf')

    ax[0,1].imshow(
        cdf,
        origin='lower',
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap='inferno',
        vmin=0,
        vmax=1,
    )
    ax[0,1].set_title('differentiable ecdf')

    im = ax[0,2].imshow(
        kde_eval, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], cmap='inferno'
    )
    ax[0,2].set_title('kde')

    cdf_grad_x = grad_vjcdf(xy, S)[:, 0].reshape(Xgrid.shape)
    cdf_grad_y = grad_vjcdf(xy, S)[:, 1].reshape(Xgrid.shape)
    cdf_grad_u = grad_vjcdf(xy, S_rot)[:, 0].reshape(Xgrid.shape)
    ax[1,0].imshow(
        cdf_grad_x,
        origin='lower',
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap='inferno',
    )
    ax[1,0].set_title('cdf grad x')

    ax[1,1].imshow(
        cdf_grad_y,
        origin='lower',
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap='inferno',
    )
    ax[1,1].set_title('cdf grad y')

    ax[1,2].imshow(
        cdf_grad_u,
        origin='lower',
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap='inferno',
    )
    ax[1,2].set_title('cdf grad u')

    # pdf_x = np.diff(cdf_grad_x, axis=0, prepend=1.0)


    cdf_x = vjcdf_ax(xy, S, 0).reshape(Xgrid.shape)
    cdf_y = vjcdf_ax(xy, S, 1).reshape(Xgrid.shape)
    cdf_u = vjcdf_ax(xy, S_rot, 0).reshape(Xgrid.shape)
    ax[2,0].imshow(
        cdf_x,
        origin='lower',
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap='inferno',
        vmin=0,
    )
    ax[2,0].set_title('cdf x')

    ax[2,1].imshow(
        cdf_y,
        origin='lower',
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap='inferno',
        vmin=0,
    )
    ax[2,1].set_title('cdf y')

    ax[2,2].imshow(
        cdf_u,
        origin='lower',
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap='inferno',
        vmin=0,
    )
    ax[2,2].set_title('cdf u')

    plt.show()

else:

    x = jnp.linspace(S.min() - 1, S.max() + 1, 200)
    kde_eval = kde.evaluate(x)
    cdf = vjcdf_nd(x, S)
    diff = jnp.diff(cdf, axis=0)
    # cdf_grad = grad_vjcdf(x, S)
    fig, ax = plt.subplots(3, 1, figsize=(5, 15))
    ax[0].plot(x, kde_eval)
    ax[0].set_title('kde density')
    ax[1].plot(x, cdf)
    ax[1].set_title('ecdf')
    ax[2].plot(x[:-1], diff)
    ax[2].set_title('diff')
    plt.show()

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     quantiles     --
# ···············································································



# def Q(q):
    # # q is a quantile
    # # returns the value of S at q
    # return vmap(jnp.interp, in_axes=(1, None, 1))(q, ranks, S_sorted)


# Sx = S[:, 0]
# Sy = S[:, 1]
# Su = 2.0*Sx + 2.0*Sy
# # rotation matrix for theta
# Srotx = Srot[:, 0]
# Sroty = Srot[:, 1]


def Q(q, arr):
    # get quantile directly using jnp
    return jnp.quantile(arr, q)
    # q = jnp.asarray(q)
    # arr_sorted = jnp.sort(arr)
    # ranks = jnp.arange(arr.shape[0]) / arr.shape[0]
    # return jnp.interp(q, ranks, arr_sorted)

# qx = Q(0.5, Sx)
# qy = Q(0.5, Sy)
# qrx = Q(0.5, Srotx)

# qrx
# invrot = jnp.array([[jnp.cos(theta), jnp.sin(theta)], [-jnp.sin(theta), jnp.cos(theta)]])

# rotated_qx_qy = jnp.dot(jnp.array([qx, qy]), invrot)
# rotated_qx_qy

# 2*qx + 3*qy

# x = jnp.linspace(0, 1, 200)
# fig, ax = plt.subplots(4, 1, figsize=(5, 20))
# ax[0].plot(x, Q(x, Sx))
# ax[0].set_title('Qx')
# ax[1].plot(x, Q(x, Sy))
# ax[1].set_title('Qy')
# ax[2].plot(x, Q(x, Su))
# ax[2].set_title('Qu')
# ax[3].plot(x, 2*Q(x, Sx) + 3*Q(x, Sy))
# ax[3].set_title('2Qx + 3Qy')
# plt.show()





# x = jnp.linspace(0, 1, 200)
# Xgrid, Ygrid = np.meshgrid(x, x)
# xy = np.vstack([Xgrid.ravel(), Ygrid.ravel()]).T
# qq = Q(xy).T
# qq_x = qq[:, 0].reshape(Xgrid.shape)
# qq_y = qq[:, 1].reshape(Xgrid.shape)


# fig, ax = plt.subplots(3, 2, figsize=(8, 8))
# ax[0, 0].imshow(
    # qq_x,
    # origin='lower',
    # extent=[0, 1, 0, 1],
    # cmap='inferno',
    # vmin=0,
# )
# ax[0, 0].set_title('qq_x')
# ax[0, 1].imshow(
    # qq_y,
    # origin='lower',
    # extent=[0, 1, 0, 1],
    # cmap='inferno',
    # vmin=0,
# )
# ax[0, 1].set_title('qq_y')



# plt.show()


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                           --     train     --
# ···············································································


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


Y = S
X = jnp.zeros_like(Y)

H_SIZE = 256
N_LAYERS = 6


def uniform_cdf(x):
    return jnp.prod(jnp.maximum(0, jnp.minimum(1, x)), axis=0)


activation = jax.nn.leaky_relu

# def model(params, z, key):
# k0, k1, k2, k3, kn = jax.random.split(key, 5)
# get_p = partial(get_param, params=params)
# # we go from z to H_size to output_shape
# x = dense_layer(z, H_SIZE, get_p, k0, 'dense0')
# x = activation(x)
# for i in range(N_LAYERS - 1):
# x = dense_layer(x, H_SIZE, get_p, k1, f'dense{i+1}')
# x = activation(x)
# return dense_layer(x, Y.shape[1], get_p, k3, 'dense_out')


def model(params, z, key):
    k0, k1, k2, k3, kn = jax.random.split(key, 5)
    get_p = partial(get_param, params=params)
    # we go from z to H_size to output_shape
    x = dense_layer(z, H_SIZE, get_p, k0, 'dense0')
    x = activation(x)
    for i in range(N_LAYERS - 1):
        x = dense_layer(x, H_SIZE, get_p, k1, f'dense{i+1}')
        x = activation(x)
        # x = jnp.concatenate([x.flatten(), z])
    return dense_layer(x, Y.shape[1], get_p, k3, 'dense_out')



# opt = optax.amsgrad(learning_rate=3e-4)
opt = optax.adam(learning_rate=3e-4)

# schedule = optax.linear_schedule(1e-3, 1e-5, 2000)
# opt = optax.chain(
# optax.adam(learning_rate=schedule),
# )

n_rotations = 9
zsize = Y.shape[1] * n_rotations

angles = jnp.linspace(0, jnp.pi, n_rotations, endpoint=False)

vmodel = vmap(model, in_axes=(None, 0, 0))
vjcdf_nd_sh = vmap(jcdf_nd, in_axes=(0, None, None))
vjcdf_sh = jit(vmap(jcdf, in_axes=(0, None, None)))

sh = 10.0

params = {}
rng_key = jax.random.PRNGKey(1424)
model(params, jnp.zeros(zsize), key)

n_gen_samples = 10000
zs = jax.random.uniform(key, (n_gen_samples, zsize), minval=0, maxval=1)
# zs = jax.random.uniform(key, (n_gen_samples, 1), minval=0, maxval=1)
# zs = jnp.tile(zs, (1, zsize))

Yrotated = jnp.stack([rotate2d(Y, a) for a in angles])


def loss_fn(params, sh, y, key):

    zs = jax.random.uniform(key, (y.shape[0], zsize), minval=0, maxval=1)
    pred = vmodel(params, zs, jax.random.split(key, y.shape[0]))
    rotated_preds = jnp.stack([rotate2d(pred, a) for a in angles])

    # pred_rotated = jnp.dot(S, rotm)


    # maybe add more zs. Reproject?
    # Let's say:
    # u = 2x+3y
    # v = 5x-2y
    # I know F(x,y), the original cdf.
    # Solve the equations defining u and v for x and y in terms of u and v:
    # x = (2u - 3v)/9
    # y = (5v - 2u)/7
    # G(u,v) = F((2u - 3v)/9, (5v - 2u)/7)
    # x, y = zs


    # cdfnd_xy = vjcdf_nd_sh(pred, Y, sh)
    # cdfnd_r = vjcdf_nd_sh(pred_transformed, Yr, sh)

    cdf_all = jnp.hstack(vmap(vjcdf_nd_sh, in_axes=(0, 0, None))(rotated_preds, Yrotated, sh))
    nderror = jnp.sqrt(jnp.mean((cdf_all - zs) ** 2))

    # # cdf = vjcdf_sh(pred, Y, 500.0)
    # cdf = vjcdf(pred, Y)
    # zprod = jnp.prod(zs, axis=1)
    # cdf_error = jnp.mean((cdf - zprod) ** 2)

    # quantiles = Q(z1).T
    # q_error = jnp.sqrt(jnp.mean((quantiles - pred) ** 2)) * cdferror
    # quantiles_p = Q_rp(z2).T
    # q_error_p = jnp.sqrt(jnp.mean((quantiles_p - pred) ** 2)) * cdferror

    # return qerror + comp_cdf_error
    # return comp_cdf_error

    # return q_error * jnp.max(jnp.array([0.1, 1.0 - sh])) + cdf_error * sh + cdferror * sh
    return  nderror


opt_state = opt.init(params)


@jit
def update(params, opt_state, sh, y, key):
    loss, grads = value_and_grad(loss_fn)(params, sh, y, key)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


n_epochs = 5
batch_size = 64
n_batches = Y.shape[0] // batch_size
xbatches = jnp.split(X[: n_batches * batch_size], n_batches)
ybatches = jnp.split(Y[: n_batches * batch_size], n_batches)

y = ybatches[0]
x = xbatches[0]

# sharpness_range = (1, 1000.0)
sharpness_range = (0.0, 100.0)

losses = []
for epoch in range(n_epochs):
    sharpness = sharpness_range[0] + (sharpness_range[1] - sharpness_range[0]) * (epoch / n_epochs)
    rng_key, subkey = jax.random.split(rng_key, 2)
    batchkeys = jax.random.split(subkey, n_batches)
    for xbatch, ybatch, key in zip(xbatches, ybatches, batchkeys):
        params, opt_state, loss = update(params, opt_state, sharpness, ybatch, key)
        losses.append(loss)
    print(f'Epoch {epoch} loss: {loss}')

plt.semilogy(losses)


subkeys = jax.random.split(rng_key, n_gen_samples)
vm = vmap(model, in_axes=(None, 0, 0))
gen_samples = vm(params, zs, subkeys)
kde_gen = gaussian_kde(gen_samples.T, bw_method='silverman')

if ndim == 1:
    x = jnp.linspace(Y.min() - 1, Y.max() + 1, 500)
    Ztarget = kde.evaluate(x)
    Zlearned = kde_gen.evaluate(x)
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].plot(x, Ztarget)
    ax[0, 0].set_title('Target distribution')
    ax[0, 1].plot(x, Zlearned)
    ax[0, 1].set_title('Learned distribution')

    ax[1, 0].plot(x, vjcdf(x, Y))
    ax[1, 0].set_title('Target cdf')
    ax[1, 1].plot(x, vjcdf(x, gen_samples))
    ax[1, 1].set_title('Learned cdf')
    plt.show()

else:

    x = jnp.linspace(Y[:, 0].min(), Y[:, 0].max(), 200)
    y = jnp.linspace(Y[:, 1].min(), Y[:, 1].max(), 200)
    Xgrid, Ygrid = np.meshgrid(x, y)
    xy = np.vstack([Xgrid.ravel(), Ygrid.ravel()])

    diffmin = -0.05
    diffmax = 0.05

    fig, ax = plt.subplots(5, 3, figsize=(9, 15))

    Ztarget = kde.evaluate(xy).reshape(Xgrid.shape)
    Zlearned = kde_gen.evaluate(xy).reshape(Xgrid.shape)
    Zdiff = Zlearned - Ztarget
    ax[0, 0].pcolormesh(Xgrid, Ygrid, Ztarget, cmap='inferno')
    ax[0, 0].set_title('Target distribution')
    ax[0, 1].pcolormesh(Xgrid, Ygrid, Zlearned, cmap='inferno')
    ax[0, 1].set_title('Learned distribution')
    ax[0, 2].pcolormesh(Xgrid, Ygrid, Zdiff, cmap='jet')
    ax[0, 2].set_title('Difference')

    cdf_target_xy = vjcdf_nd(xy.T, Y)
    cdf_target_x = cdf_target_xy[:, 0].reshape(Xgrid.shape)
    cdf_target_y = cdf_target_xy[:, 1].reshape(Xgrid.shape)

    cdf_learned_xy = vjcdf_nd(xy.T, gen_samples)
    cdf_learned_x = cdf_learned_xy[:, 0].reshape(Xgrid.shape)
    cdf_learned_y = cdf_learned_xy[:, 1].reshape(Xgrid.shape)

    xdiff = cdf_learned_x - cdf_target_x
    ydiff = cdf_learned_y - cdf_target_y

    ax[1, 0].pcolormesh(Xgrid, Ygrid, cdf_target_x, cmap='inferno')
    ax[1, 0].set_title('Target cdf x')
    ax[1, 1].pcolormesh(Xgrid, Ygrid, cdf_learned_x, cmap='inferno')
    ax[1, 1].set_title('Learned cdf x')
    ax[1, 2].pcolormesh(Xgrid, Ygrid, xdiff, cmap='jet', vmin=diffmin, vmax=diffmax)
    ax[1, 2].set_title('Difference x')

    ax[2, 0].pcolormesh(Xgrid, Ygrid, cdf_target_y, cmap='inferno')
    ax[2, 0].set_title('Target cdf y')
    ax[2, 1].pcolormesh(Xgrid, Ygrid, cdf_learned_y, cmap='inferno')
    ax[2, 1].set_title('Learned cdf y')
    ax[2, 2].pcolormesh(Xgrid, Ygrid, ydiff, cmap='jet', vmin=diffmin, vmax=diffmax)
    ax[2, 2].set_title('Difference y')

    cdf_target = vjcdf(xy.T, Y).reshape(Xgrid.shape)
    cdf_learned = vjcdf(xy.T, gen_samples).reshape(Xgrid.shape)
    ax[3, 0].pcolormesh(Xgrid, Ygrid, cdf_target, cmap='inferno', vmin=0, vmax=1)
    ax[3, 0].set_title('Target cdf')
    ax[3, 1].pcolormesh(Xgrid, Ygrid, cdf_learned, cmap='inferno')
    ax[3, 1].set_title('Learned cdf')
    ax[3, 2].pcolormesh(
        Xgrid, Ygrid, cdf_learned - cdf_target, cmap='jet', vmin=diffmin, vmax=diffmax
    )
    ax[3, 2].set_title('Difference')

    recons_cdf_target = cdf_target_x * cdf_target_y
    recons_cdf_learned = cdf_learned_x * cdf_learned_y

    ax[4, 0].pcolormesh(Xgrid, Ygrid, recons_cdf_target, cmap='inferno', vmin=0, vmax=1)
    ax[4, 0].set_title('Target cdf')
    ax[4, 1].pcolormesh(Xgrid, Ygrid, recons_cdf_learned, cmap='inferno')
    ax[4, 1].set_title('Learned cdf')
    ax[4, 2].pcolormesh(
        Xgrid, Ygrid, recons_cdf_learned - recons_cdf_target, cmap='jet', vmin=diffmin, vmax=diffmax
    )
    ax[4, 2].set_title('Difference')

    plt.show()

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     quad func     --
# ···············································································
import scipy.integrate


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def quad(func, a, b, args=()):
    """Calculates the integral

    \int_a^b func(t, *args) dt
    """

    result, _ = scipy.integrate.quad(func, a, b, args)
    return result


def quad_fwd(func, a, b, args=()):
    result = quad(func, a, b, args)
    aux = (a, b, args)
    return result, aux


def quad_bwd(func, aux, grad):
    a, b, args = aux

    grad_a = -grad * func(a, *args)
    grad_b = grad * func(b, *args)

    grad_args = []
    for i in range(len(args)):

        def _vjp_func(_t, *_args):
            return jax.grad(func, i)(_t, *_args)

        grad_args.append(grad * quad(_vjp_func, a, b, args))
    grad_args = tuple(grad_args)

    return grad_a, grad_b, grad_args


quad.defvjp(quad_fwd, quad_bwd)

# grad_true = kde.evaluate(pred.T)
# def mod_squeezed(z, k):
# return model(params, z, k).squeeze()
# grad_pred = vmap(jax.grad(mod_squeezed), in_axes=(0, 0))(zs, subkeys)
# grad_error = jnp.mean((grad_pred - grad_true) ** 2)
# error += grad_error * (1.0 - sh/30.0)
# # also we want to maximize likelyhood of pred
# llhpred = 1.0 - kde.evaluate(pred.T)
# llhz = 1.0 - kde.evaluate((zs.T-0.5) * 5.0)
# error += jnp.mean((llhpred - llhz) ** 2)
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     mnist
# ···············································································

import mnist


data = mnist.train_images()
labels = mnist.train_labels()
data = data.astype(np.float32)
data = data / 255.0
# get all the zeros
select = 4
data = data[labels == select]

for d in data[:10]:
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(d, cmap='gray')
    plt.show()


# flatten all the images
data = data.reshape(data.shape[0], -1)
# get the first image
d = data[0]
d.shape
ndim = 784

# # generate samples from the learned distribution
# n_gen_samples = 10
# subkeys = jax.random.split(rng_key, n_gen_samples)
# vm = vmap(model, in_axes=(None, 0, 0))
# zs = jax.random.uniform(key, (n_gen_samples, Y.shape[1]), minval=0, maxval=1)
# zs = jax.random.uniform(key, (n_gen_samples, ), minval=0, maxval=1)
# zs = jnp.tile(zs, (Y.shape[1], 1)).T

# zs = jnp.linspace(0, 1, n_gen_samples)
# zs = jnp.tile(zs, (Y.shape[1], 1)).T

# gen_samples = vm(params, zs, subkeys)

# # they're mnists. Let's plot the samples:
# fig, axs = plt.subplots(1, n_gen_samples, figsize=(n_gen_samples, 1))
# for i in range(n_gen_samples):
# axs[i].imshow(gen_samples[i].reshape(28, 28), cmap='gray')
# axs[i].axis('off')

# plt.show()


diff1 = jnp.diff(cdf_target, axis=1)
    pdf_x_target = jnp.diff(diff1, axis=0)
    ax[4, 0].imshow(
        pdf_x_target,
        origin='lower',
        cmap='inferno',
        vmin=0,
    )
    ax[4, 0].set_title('target pdf reconstructed')

    diff1 = jnp.diff(cdf_learned, axis=1)
    pdf_x_learned = jnp.diff(diff1, axis=0)
    ax[4, 1].imshow(
        pdf_x_learned,
        origin='lower',
        cmap='inferno',
        vmin=0,
    )
    ax[4, 1].set_title('learned pdf reconstructed')

    ax[4, 2].imshow(
        pdf_x_learned - pdf_x_target, origin='lower', cmap='jet', vmin=diffmin, vmax=diffmax
    )
    ax[4, 2].set_title('Difference')

    # mirror first column
    # prepcdf = np.concatenate((cdf_grad_x[:, 0:1], cdf_grad_x), axis=1)
    diff1 = jnp.diff(cdf, axis=1)
    pdf_x = jnp.diff(diff1, axis=0)
    # pdf_y = np.diff(cdf_grad_y, axis=1, prepend=1.0)
    ax[5].imshow(
        pdf_x,
        origin='lower',
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap='inferno',
        vmin=0,
    )
    ax[5].set_title('pdf reconstructed')

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
