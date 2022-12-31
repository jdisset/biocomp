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
n_samples = 3000

# Set the number of blobs to generate
n_blobs = 30

key = jax.random.PRNGKey(9021)
rng = key
# Generate the probability distribution
ndim = 2
S = generate_probability_distribution_jax(n_samples, n_blobs, key, ndim=ndim) / 20.0

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


SHARPNESS = 20.0


@jit
def heavyside(x, sharpness=SHARPNESS):
    return jax.nn.sigmoid(x * sharpness)


def jcdf_m(x, S):
    def lt(s, x):
        return jnp.all(s <= x)

    return jnp.mean(jax.vmap(lt, in_axes=(0, None))(S, x))


# differentiable version. Returns the product
def jcdf(x, S, sharpness=SHARPNESS):
    def lt_diff(s, x):
        return jnp.prod(heavyside(x - s, sharpness=sharpness))

    return jnp.mean(jax.vmap(lt_diff, in_axes=(0, None))(S, x))


def jcdf_nd(x, S, sharpness=SHARPNESS):
    def lt_diff(s, x):
        return heavyside(x - s, sharpness=sharpness)

    return jnp.mean(jax.vmap(lt_diff, in_axes=(0, None))(S, x), axis=0)


def jcdf_ax(x, S, axis, sharpness=SHARPNESS):
    def lt_diff(s, x):
        return heavyside(x[axis] - s[axis], sharpness=sharpness)

    return jnp.mean(jax.vmap(lt_diff, in_axes=(0, None))(S, x))


vjcdf = jit(vmap(jcdf, in_axes=(0, None)))
vjcdf_ax = jit(vmap(jcdf_ax, in_axes=(0, None, None)))
vjcdf_nd = jit(vmap(jcdf_nd, in_axes=(0, None)))
vjcdf_m = jit(vmap(jcdf_m, in_axes=(0, None)))
grad_vjcdf = jit(vmap(grad(jcdf), in_axes=(0, None)))


kde = gaussian_kde(S.T, bw_method='silverman')

if ndim == 2:
    x = jnp.linspace(-1, 1, 128)
    y = x
    Xgrid, Ygrid = np.meshgrid(x, y)
    xy = np.vstack([Xgrid.ravel(), Ygrid.ravel()]).T

    kde_eval = kde.evaluate(xy.T).reshape(Xgrid.shape)

    cdf_m = vjcdf_m(xy, S).reshape(Xgrid.shape)
    cdf = vjcdf(xy, S).reshape(Xgrid.shape)

    fig, ax = plt.subplots(3, 3, figsize=(30, 30))
    ax = ax.flatten()
    ax[0].imshow(
        cdf_m,
        origin='lower',
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap='inferno',
        vmin=0,
        vmax=1,
    )
    ax[0].set_title('true ecdf')

    ax[1].imshow(
        cdf,
        origin='lower',
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap='inferno',
        vmin=0,
        vmax=1,
    )
    ax[1].set_title('differentiable ecdf')

    im = ax[2].imshow(
        kde_eval, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], cmap='inferno'
    )
    ax[2].set_title('kde')

    cdf_grad_x = grad_vjcdf(xy, S)[:, 0].reshape(Xgrid.shape)
    cdf_grad_y = grad_vjcdf(xy, S)[:, 1].reshape(Xgrid.shape)
    ax[3].imshow(
        cdf_grad_x,
        origin='lower',
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap='inferno',
        vmin=0,
        vmax=1,
    )
    ax[3].set_title('cdf grad x')

    ax[4].imshow(
        cdf_grad_y,
        origin='lower',
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap='inferno',
        vmin=0,
        vmax=1,
    )
    ax[4].set_title('cdf grad y')

    # pdf_x = np.diff(cdf_grad_x, axis=0, prepend=1.0)

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

    cdf_x = vjcdf_ax(xy, S, 0).reshape(Xgrid.shape)
    cdf_y = vjcdf_ax(xy, S, 1).reshape(Xgrid.shape)
    ax[6].imshow(
        cdf_x,
        origin='lower',
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap='inferno',
        vmin=0,
    )
    ax[6].set_title('cdf x')

    ax[7].imshow(
        cdf_y,
        origin='lower',
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap='inferno',
        vmin=0,
    )
    ax[7].set_title('cdf y')

    cdf_xy = vjcdf_nd(xy, S)[:, 1].reshape(Xgrid.shape)
    ax[8].imshow(
        cdf_xy,
        origin='lower',
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap='inferno',
        vmin=0,
    )

    plt.show()

else:

    x = jnp.linspace(S.min() - 1, S.max() + 1, 200)
    kde_eval = kde.evaluate(x)
    cdf = vjcdf(x, S)
    cdf_grad = grad_vjcdf(x, S)
    fig, ax = plt.subplots(3, 1, figsize=(5, 15))
    ax[0].plot(x, kde_eval)
    ax[0].set_title('kde density')
    ax[1].plot(x, cdf)
    ax[1].set_title('ecdf')
    ax[2].plot(x, cdf_grad)
    ax[2].set_title('grad')
    plt.show()

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                        --     other plot     --
# ···············································································

s = S.T
# Estimate the density using gaussian_kde
kde = gaussian_kde(s)

# Generate a fixed 100x100 grid of points between (-1,-1) and (1,1)
x, y = np.mgrid[-1:1:100j, -1:1:100j]
positions = np.vstack([x.ravel(), y.ravel()])

# Evaluate the density at each point on the grid
density = kde.evaluate(positions)

# Reshape the density array to the same shape as the grid
density = density.reshape(x.shape)

# Plot the estimated PDF
fig, ax = plt.subplots()
ax.pcolormesh(x, y, density)
ax.set_title("PDF estimate")

# Compute the CDF along each dimension using the cumulative sum function
cdf_x = np.cumsum(density, axis=0)
cdf_y = np.cumsum(density, axis=1)

cdf = np.cumsum(cdf_x, axis=1)

# Plot the CDF
fig, ax = plt.subplots()
ax.pcolormesh(x, y, cdf_x)
ax.set_title("CDF (x dimension)")

fig, ax = plt.subplots()
ax.pcolormesh(x, y, cdf_y)
ax.set_title("CDF (y dimension)")


fig, ax = plt.subplots()
ax.pcolormesh(x, y, cdf)
ax.set_title("CDF (x+y dimension)")


pdf_x = np.diff(cdf, axis=0, prepend=0)
pdf = np.diff(pdf_x, axis=1, prepend=0)

# Plot the final PDF
fig, ax = plt.subplots()
ax.pcolormesh(x, y, pdf)
ax.set_title("PDF (x+y dimension)")


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

H_SIZE = 64
N_LAYERS = 12


def uniform_cdf(x):
    return jnp.prod(jnp.maximum(0, jnp.minimum(1, x)), axis=0)


# activation = jax.nn.relu
activation = jax.nn.leaky_relu


def model(params, z, key):
    k0, k1, k2, k3, kn = jax.random.split(key, 5)
    get_p = partial(get_param, params=params)
    # we go from z to H_size to output_shape
    x = dense_layer(z, H_SIZE, get_p, k0, 'dense0')
    x = activation(x)
    for i in range(N_LAYERS - 1):
        x = dense_layer(x, H_SIZE, get_p, k1, f'dense{i+1}')
        x = activation(x)
    return dense_layer(x, S.shape[1], get_p, k3, 'dense_out')


params = {}
rng_key = jax.random.PRNGKey(121)
model(params, jnp.zeros(Y.shape[1]), key)

# opt = optax.amsgrad(learning_rate=1e-4)

schedule = optax.linear_schedule(1e-3, 1e-5, 5000)
opt = optax.chain(
    optax.adam(learning_rate=schedule),
)


vmodel = vmap(model, in_axes=(None, 0, 0))
vjcdf_sh = jit(vmap(jcdf, in_axes=(0, None, None)))
vjcdf_ax_sh = jit(vmap(jcdf_ax, in_axes=(0, None, None, None)))
vjcdf_nd_sh = jit(vmap(jcdf_nd, in_axes=(0, None, None)))


def softmax(x):
    return jnp.exp(x) / jnp.sum(jnp.exp(x), axis=0)


def softmin(x):
    return jnp.log(jnp.exp(x) / jnp.sum(jnp.exp(x), axis=0))


def loss_fn(params, sh, y, key):
    zs = jax.random.uniform(key, (y.shape[0], y.shape[1]), minval=0, maxval=1)
    pred = vmodel(params, zs, jax.random.split(key, y.shape[0]))

    cdf = vjcdf_nd_sh(pred, S, sh)
    # return jnp.mean(jnp.abs(cdf - zs))
    return jnp.mean((cdf - zs)**2)


opt_state = opt.init(params)


@jit
def update(params, opt_state, sh, y, key):
    loss, grads = value_and_grad(loss_fn)(params, sh, y, key)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


n_epochs = 100
batch_size = 64
n_batches = S.shape[0] // batch_size
# We need split to result in an equal division
xbatches = jnp.split(X[: n_batches * batch_size], n_batches)
ybatches = jnp.split(Y[: n_batches * batch_size], n_batches)
y = ybatches[0]
x = xbatches[0]

sharpness_range = (0, 100.0)
# sharpness_range = (10.0, 50.0)

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

# generate lots of samples from the learned distribution
n_gen_samples = 10000
subkeys = jax.random.split(rng_key, n_gen_samples)
vm = vmap(model, in_axes=(None, 0, 0))

zs = jax.random.uniform(key, (n_gen_samples, S.shape[1]), minval=0, maxval=1)
gen_samples = vm(params, zs, subkeys)

kde_gen = gaussian_kde(gen_samples.T, bw_method='silverman')
# we plot side by side:
# row 0: target distribution, learned distribution
# row 1: target cdf, learned cdf


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
    Ztarget = kde.evaluate(xy).reshape(Xgrid.shape)
    Zlearned = kde_gen.evaluate(xy).reshape(Xgrid.shape)
    fig, ax = plt.subplots(3, 2, figsize=(10, 15))
    ax[0, 0].pcolormesh(Xgrid, Ygrid, Ztarget, cmap='inferno')
    ax[0, 0].set_title('Target distribution')
    ax[0, 1].pcolormesh(Xgrid, Ygrid, Zlearned, cmap='inferno')
    ax[0, 1].set_title('Learned distribution')
    cdf_target = vjcdf(xy.T, Y).reshape(Xgrid.shape)
    cdf_learned = vjcdf(xy.T, gen_samples).reshape(Xgrid.shape)

    ax[1, 0].pcolormesh(Xgrid, Ygrid, cdf_target, cmap='inferno')
    ax[1, 0].set_title('Target cdf')
    ax[1, 1].pcolormesh(Xgrid, Ygrid, cdf_learned, cmap='inferno')
    ax[1, 1].set_title('Learned cdf')

    diff1 = jnp.diff(cdf_target, axis=1)
    pdf_x = jnp.diff(diff1, axis=0)
    ax[2, 0].imshow(
        pdf_x,
        origin='lower',
        cmap='inferno',
        vmin=0,
    )
    ax[2, 0].set_title('target pdf reconstructed')

    diff1 = jnp.diff(cdf_learned, axis=1)
    pdf_x = jnp.diff(diff1, axis=0)
    ax[2, 1].imshow(
        pdf_x,
        origin='lower',
        cmap='inferno',
        vmin=0,
    )
    ax[2, 1].set_title('learned pdf reconstructed')

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
