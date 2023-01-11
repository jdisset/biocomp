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
# {{{                       --     generate data     --
# ···············································································
def generate_probability_distribution_jax(n_samples, n_blobs, rng, ndim=2):
    X = jnp.empty((0, ndim))
    weights = jax.random.uniform(rng, (n_blobs,), minval=1, maxval=3)
    weights /= weights.sum()
    lens = (weights * n_samples).astype(int)
    keys = jax.random.split(rng, n_blobs)

    for l, k in zip(lens, keys):
        k, k0, k1, k2 = jax.random.split(k, 4)
        mean = jax.random.uniform(k0, (ndim,), minval=-2, maxval=2) * 40
        sigma = jax.random.uniform(k1, (ndim,), minval=-2, maxval=2) * 10
        # covariance = jax.random.normal(k1, (ndim, ndim)) * 0.75 + 1.0
        # covariance = jnp.eye(ndim) * (jax.random.uniform(k2, (ndim)) * 10.0 + jnp.abs(covariance))
        # samples = jax.random.multivariate_normal(k2, mean, covariance, shape=(l,))
        samples = jax.random.normal(k2, (l, ndim)) * sigma + mean
        X = jnp.concatenate((X, samples), axis=0)

    return X


# Set the number of samples to generate for each blob
n_samples = 1000
n_blobs = 8
seed = 12
key = jax.random.PRNGKey(seed)

ndim = 1
S = generate_probability_distribution_jax(n_samples, n_blobs, key, ndim=ndim) / 100.0

kde = gaussian_kde(S.T, bw_method='silverman')

resolution = 200
dmin, dmax = -2, 2

x = jnp.linspace(dmin, dmax, resolution)
if ndim == 2:
    xy = jnp.vstack(map(jnp.ravel, jnp.meshgrid(x, x))).T
    pdf = kde(xy.T).reshape(resolution, resolution)
    fig, ax = mkfig(1, 1)
    ax.pcolormesh(x, x, pdf, cmap='inferno')
    ax.set_title(f'KDE for S ({n_samples} samples, {n_blobs} blobs, seed={seed})')
    plt.show()
else:
    kde_eval = kde.evaluate(x)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(x, kde_eval)
    ax.set_title('kde density')
    plt.show()



#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                    --     random projections     --
# ···············································································

# generate random unit vectors to project onto
_, key = jax.random.split(key, 2)
n_proj = 1

def rotate2d(x, theta):
    c, s = jnp.cos(theta), jnp.sin(theta)
    R = jnp.array([[c, -s], [s, c]])
    return jnp.dot(x, R)

# pvecs = jnp.array(
    # [v / jnp.linalg.norm(v) for v in jax.random.normal(key, (n_proj, ndim))]
# )
# pvecs = jnp.array([v for v in jax.random.normal(key, (n_proj, ndim))])
# scales = jax.random.uniform(key, (n_proj, ndim), minval=0.1, maxval=10.0)
# pvecs = pvecs * scales

pvecs = vmap(rotate2d, in_axes=(None, 0))(jnp.array([0,1]), jnp.linspace(0, jnp.pi/2, n_proj, endpoint=True))

translations = jax.random.uniform(key, (n_proj,), minval=-10, maxval=10)
translations = 0

pvecs = jnp.array([[1]])

S_proj = jnp.dot(S, pvecs.T) + translations
all_kdes = [gaussian_kde(S_proj[:, i].T) for i in range(n_proj)]

fig, ax = mkfig(1, 1)
ax.pcolormesh(x, x, pdf, cmap='inferno')
for pvec, sp, k in zip(pvecs, S_proj.T, all_kdes):
    # plot the projection vector as an infinite line
    ax.plot([-pvec[0] * 1000, pvec[0] * 1000], [-pvec[1] * 1000, pvec[1] * 1000], 'w--', linewidth=0.5, alpha=1)
    vx = jnp.linspace(dmin*1.5, dmax*1.5, 5000)
    vy = k(vx) * 0.5
    pnormal = jnp.array([-pvec[1], pvec[0]])
    px = pvec[:, None] * vx
    py = pnormal[:, None] * vy + px
    ax.scatter(py[0,:], py[1,:], s=0.01)

ax.set_xlim(dmin, dmax)
ax.set_ylim(dmin, dmax)
ax.set_title(f'Projection of S onto {n_proj} random unit vectors')

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     quantiles     --
#···············································································


@jit
def Q(s, z):
    return jnp.quantile(s, z)

# all_quantiles = vmap(Q, in_axes=(1, 1))(S_proj, zs_proj)

# scatter the samples
fig, ax = mkfig(1, 1)
ax.scatter(unit_hypercube_samples[:, 0], unit_hypercube_samples[:, 1], s=0.5)
ax.set_ylim(-1,2)
ax.set_xlim(-1,2)
ax.set_title('Samples from the unit hypercube')

unit_hypercube_samples = jax.random.uniform(key, (50000, S.shape[1]), minval=-1, maxval=1)
Hproj = jnp.dot(unit_hypercube_samples, pvecs.T) + translations

def cdf(S, x):
    def lt(s, x):
        return jnp.all(s <= x)
    return jnp.mean(jax.vmap(lt, in_axes=(0, None))(S, x))

SHARPNESS = 100
@jit
def heavyside(x, sharpness=SHARPNESS):
    return jax.nn.sigmoid(x * sharpness)
def cdf_d(S, x, sharpness=SHARPNESS):
    assert (S.ndim == 1)
    smin, smax = S.min(), S.max()
    S = (S - smin) / (smax - smin)
    x = (x - smin) / (smax - smin)
    def lt_diff(s, x):
        return heavyside(x - s, sharpness=sharpness)
    return jnp.mean(jax.vmap(lt_diff, in_axes=(0, None))(S, x))



vcdf = jax.vmap(cdf, in_axes=(None, 0))
vcdf_d = jax.vmap(cdf_d, in_axes=(None, 0))


for pvec, h in zip(pvecs, Hproj.T):
    qx = jnp.linspace(0, 1, 500)
    q = Q(h, qx)
    fig, ax = mkfig(1, 3)
    ax[0].plot(qx, q)
    ax[0].set_title(f'Quantiles of S projected onto {pvec}')
    c = vcdf(h, qx)
    ax[1].plot(qx, c)
    ax[1].set_title(f'CDF of S projected onto {pvec}')
    c_d = vcdf_d(h, qx)
    ax[2].plot(qx, c_d)
    ax[2].set_title(f'CDF of S projected onto {pvec} (differentiable)')





#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                           --     model    --
#···············································································
H_SIZE = 400
N_LAYERS = 8

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
    x = dense_layer(z, H_SIZE, get_p, k0, 'dense0')
    x = activation(x)
    for i in range(N_LAYERS - 1):
        x = dense_layer(x, H_SIZE, get_p, k1, f'dense{i+1}')
        x = activation(x)
        x = jnp.concatenate([x.flatten(), z])
    return dense_layer(x, Y.shape[1], get_p, k3, 'dense_out')

vm = vmap(model, in_axes=(None, 0, None))

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                           --     train     --
#···············································································

Y=S
Y_proj = jnp.dot(Y, pvecs.T) + translations
N = 100 # batch size, basically

zsize = Y.shape[1]
zsize = pvecs.shape[0]

params = {}
rng_key = jax.random.PRNGKey(1424)

model(params, jnp.zeros(zsize), rng_key)

opt = optax.adam(learning_rate=1e-4)
opt_state = opt.init(params)


def loss_fn(params, key):
    zs = jax.random.uniform(key, (N, Y.shape[1]), minval=-1, maxval=1)
    zproj = jnp.dot(zs, pvecs.T) + translations
    # yhat = vm(params, zs, key)
    yhat = vm(params, zproj, key)
    yproj = jnp.dot(yhat, pvecs.T) + translations
    z_cdf = vmap(vcdf_d, in_axes=(1,1))(Hproj, zproj).T
    all_quantiles = vmap(Q, in_axes=(1, 1))(Y_proj, z_cdf)

    # # neg_log_likelihood = -jnp.log(kde.evaluate(yhat.T))
    # likelihood = kde.evaluate(yhat.T)
    # maxl = likelihood.max()
    # likelihood = likelihood / (maxl + 1e-12)

    err = jnp.sqrt(jnp.sum((all_quantiles.T - yproj)**2, axis=1))

    # return jnp.mean(err * ((-jnp.log(likelihood + 1e-9) + 1.0)*0.3))
    # return jnp.mean(err) + jnp.mean(-jnp.log(likelihood + 1e-9))*0.2
    return jnp.mean(err)

@jit
def update(params, opt_state, key):
    loss, grads = value_and_grad(loss_fn)(params, key)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

n_epochs = 100

losses = []
for epoch in range(n_epochs):
    rng_key, subkey = jax.random.split(rng_key, 2)
    for key in jax.random.split(subkey, Y.shape[0]//N):
        params, opt_state, loss = update(params, opt_state, key)
        losses.append(loss)
    print(f'Epoch {epoch} loss: {loss}')

plt.semilogy(losses)

##

n_gen_samples = 1000
zs = jax.random.uniform(key, (N, Y.shape[1]), minval=-1, maxval=1)
zproj = jnp.dot(zs, pvecs.T) + translations
gen_samples = vm(params, zproj, key)
kde_gen = gaussian_kde(gen_samples.T, bw_method='silverman')
x = jnp.linspace(dmin, dmax, resolution)

# xy = jnp.vstack(map(jnp.ravel, jnp.meshgrid(x, x))).T
# pdf_gen = kde_gen(xy.T).reshape(resolution, resolution)
# pdf_orig = kde(xy.T).reshape(resolution, resolution)
# fig, ax = mkfig(2, 1)
# ax[0].pcolormesh(x, x, pdf_orig, cmap='inferno')
# ax[0].set_title(f'KDE for target distribution')
# ax[1].pcolormesh(x, x, pdf_gen, cmap='inferno')
# ax[1].set_title(f'KDE for generated distribution')
# plt.show()


kde_eval = kde_gen.evaluate(x)
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(x, kde_eval)
ax.set_title('kde density')
plt.show()



#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                        --     uniform cdf     --
#···············································································

def uniform_CDF(Z, proj):
  zproj = jnp.dot(Z, proj)
  return jnp.prod(proj) * zproj
  return p[0] * p[1] * x + (1 - p[0] * p[1]) * y



uniform_CDF(jnp.array([1,1]), jnp.array([0,1]))

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     hypercube stuff     --
#···············································································
@jax.jit
def g1(x, m, n):
    return jnp.dot(n, x) - jnp.dot(m, n)

@jax.jit
def volume_of_unit_hypercube_clipped_by_hyperplane(m, n):
    # Generate coordinate arrays for all possible vertices of the unit hypercube
    coordinate_arrays = [jnp.array([0, 1]) for _ in range(len(m))]
    # Use meshgrid to generate a grid of all possible vertex coordinates
    vertex_grid = jnp.meshgrid(*coordinate_arrays)
    vertex_grid
    # Stack the vertex coordinates into a 2D array with shape (2^n, n)
    vertices = jnp.stack(vertex_grid, axis=-1)
    # Reshape the vertex array into a 1D array with shape (2^n * n,)
    vertices = vertices.reshape(-1, len(m))
    # Filter the vertices to include only those inside the half-space
    inside_vertices = vertices[jax.vmap(g1, in_axes=(0, None, None))(vertices, m, n) >= 0]
    volume = 0
    for v in inside_vertices:
        zeros = len(jnp.nonzero(v == 0)[0])
        volume += (-1) ** zeros * g1(v, m, n) ** len(m) / (jnp.math.factorial(len(m)) * jnp.prod(n))
    return volume

m = jnp.array([0,0])
n = jnp.array([1,0])

volume_of_unit_hypercube_clipped_by_hyperplane(m, n)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
