## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     imports     --
# ···············································································
from jax.tree_util import Partial as partial
from jax import tree_util as pytree
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad, jit, vmap, lax
from jax.scipy.signal import convolve
import numpy as np
import jax

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                     --     helper functions     --
# ···············································································


def gaussian_kernel(l=5, sig=1.0):
    ax = np.linspace(-(l - 1) / 2.0, (l - 1) / 2.0, l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def diffuse_kernel(l, maxL):
    return gaussian_kernel(int(np.round(l * maxL)), l * maxL / 5.0)


def unison_shuffled(key, a, b):
    # returns a and b in the same new random order
    p = jax.random.permutation(key, len(a))
    return a[p], b[p]


def neighbor_shuffle(k, ar, m):
    # this shuffles 2 neighboring cells only if one of them is empty (m[0] != m[1])
    # otherwise it returns the original array
    return lax.cond(m[0] == m[1], lambda *_: (ar, m), lambda *_: unison_shuffled(k, ar, m), None)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

S = 10

key = jax.random.PRNGKey(0)
w = jax.random.bernoulli(key, 0.1, shape=(S, S, 1))


plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'


def plotworld(w, title=None, cmap='gray'):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(w, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)
    plt.show()


k_attract = diffuse_kernel(0.75,S)
k_repel = diffuse_kernel(0.75, S)
attract_intensity = 0.5
repel_intensity = 0.5
attractions = convolve(w[:, :, 0], k_attract, mode='same') * attract_intensity
repulsions = convolve(w[:, :, 0], k_repel, mode='same') * repel_intensity

plotworld(w[:, :, 0], 'positions')
plotworld(attractions, 'attractions', cmap='Blues')
plotworld(repulsions, 'repulsions', cmap='Reds')

sobel_x = np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]])
sobel_y = sobel_x.transpose()

attract_gradient_x = convolve(attractions, sobel_x, mode='same')
attract_gradient_y = convolve(attractions, sobel_y, mode='same')
attract = np.stack([attract_gradient_x, attract_gradient_y], axis=2)

repel_gradient_x = convolve(repulsions, sobel_x, mode='same')
repel_gradient_y = convolve(repulsions, sobel_y, mode='same')
repel = np.stack([repel_gradient_x, repel_gradient_y], axis=2)

# we want to plot attract as rgb image but there are only 2 values
# we need to add the blue channel as zeros
# attract = np.dstack((attract, np.zeros_like(attract[:,:,0])))
# # scale attract to be between 0 and 1 (currently it can be negative)
# attract = attract - np.min(attract)
# attract = attract / np.max(attract)
# plotworld(attract, 'attract_rgb')

# # same for repel:
# repel = np.dstack((repel, np.zeros_like(repel[:,:,0])))
# repel = repel - np.min(repel)
# repel = repel / np.max(repel)
# plotworld(repel, 'repel_rgb')

# magic ipython to reload modules:
# %load_ext autoreload
# %autoreload 2


plotworld(w[:, :, 0] * attract_gradient_x, 'positions')


@jit
def pair_masked_shuffle(key, data, mask):
    # masked shuffle of the cell pairs
    assert data.shape == mask.shape
    assert data.shape[1] == 2
    keys = jax.random.split(key, data.shape[0])
    ds, ms = vmap(neighbor_shuffle)(keys, data, mask)
    return ds.reshape(data.shape), ms.reshape(mask.shape)


@jit
def randomwalk_to_empty_2d(key, data, mask):
    k = jax.random.split(key, 4)
    d, m = pair_masked_shuffle(k[0], data.reshape(-1, 2), mask.reshape(-1, 2))
    # d and m are now stacks of pairs of neihhboring cells, possibly suffled
    # we just reconstruct them to their original shape
    d = d.reshape(data.shape)
    m = m.reshape(mask.shape)
    return d, m


a, b = randomwalk_to_empty_2d(key, w[:, :, 0], w[:, :, 0])
plotworld(a, 'positions')
plotworld(w, 'prev_positions')

##


def neighbor_masked_gradient_descent(gradient, mask, data):
    # all of the inputs are 2d arrays of 2 columns (and same shape)
    # they correspond to pairs of neighboring cells in a 2d grid
    # if one of them is empty, i.e if mask[0] != mask[1],
    # then we move the non-empty cell to the empty one if the gradient is lower in the new cell
    return lax.cond(
        mask[0] == mask[1],
        lambda *_: (gradient, mask, data),
        lambda *_: paired_descent_swap(gradient, mask, data),
        None,
    )


def paired_descent_swap(gradient, mask, data):
    # gradient and all other arrays are arrays of pairs
    # we want to reorder the pairs so that lowest gradient is first
    gradient_sorted = jnp.argsort(gradient)
    return (gradient[gradient_sorted], mask[gradient_sorted], data[gradient_sorted])


def plot_positions_and_gradient(m, g, d, title=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(g, cmap='Reds')
    x, y = np.where(m > 0.0)
    values = [d[xx, yy] for xx, yy in zip(x, y)]
    for xx, yy, v in zip(x, y, values):
        # write in bold font, and centerd in the cell
        ax.text(xx, yy, str(v), fontsize=10, ha='center', va='center')
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)
    plt.show()


@partial(jit, static_argnums=(1, 2, 3))
def cut_array(arr, off, divisor, axis=0):
    # returns the array in 3 chunks: pre, mid, post
    # mid is the biggest slice that can be split
    # in chunks of size "divisor" along axis
    def arr_slice(start, end):
        starts = [0] * arr.ndim
        starts[axis] = start
        ends = list(arr.shape)
        ends[axis] = end - start
        return lax.dynamic_slice(arr, starts, ends)

    L = arr.shape[axis]
    pre = arr_slice(0, off)
    post_size = (L - off) % divisor
    mid = arr_slice(off, L - post_size)
    post = arr_slice(L - post_size, L)
    return pre, mid, post


def rebuild_array(pre, mid, post, axis):
    return jnp.concatenate([pre, mid, post], axis=axis)

@partial(jit, static_argnums=(3,4))
def masked_gradient_descent(gradient, mask, data, axis, offset):
    sliced_g= cut_array(gradient, offset, 2, axis=axis)
    sliced_m= cut_array(mask, offset,2, axis=axis)
    sliced_d= cut_array(data, offset, 2,axis=axis)
    g, m, d = vmap(neighbor_masked_gradient_descent)(sliced_g[1].reshape(-1, 2), sliced_m[1].reshape(-1, 2), sliced_d[1].reshape(-1, 2))
    g = rebuild_array(sliced_g[0], g.reshape(sliced_g[1].shape), sliced_g[2], axis=axis)
    m = rebuild_array(sliced_m[0], m.reshape(sliced_m[1].shape), sliced_m[2], axis=axis)
    d = rebuild_array(sliced_d[0], d.reshape(sliced_d[1].shape), sliced_d[2], axis=axis)
    return g, m, d


m = w[:, :, 0].astype(np.float32)
d = np.arange(m.size).reshape(m.shape)
g = attractions
plot_positions_and_gradient(m, g, d, title='init')
for i in range(2):
    _,m,d = masked_gradient_descent(g, m, d, axis=0, offset=0)
    plot_positions_and_gradient(m, g, d, title=f'iteration {i}.5')
    _,m,d = masked_gradient_descent(g, m, d, axis=0, offset=1)
    plot_positions_and_gradient(m, g, d, title=f'iteration {i+1}')


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     imports     --
#···············································································
import jax
from jax._src.api import block_until_ready
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from numpy.ma.core import default_fill_value
from rich import print
from jax import jit, vmap, lax
from jax import tree_util as pytree
import jax.numpy as jnp

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                   --     2d random walk thingy     --
#···············································································

plt.rcParams['figure.facecolor'] = 'white'

def unison_shuffled(key, a, b):
    p = jax.random.permutation(key, len(a))
    return a[p], b[p]

def neighbor_shuffle(k, ar, m):
    return lax.cond(m[0] == m[1], lambda *_: (ar, m), lambda *_: unison_shuffled(k, ar, m), None)

def neighbor_shuffle_nobranch(k, ar, m):
    ars, ms = unison_shuffled(k, ar,m)
    return ar * m + ars * (~ms | ~m), ms

sh = (2,3)
axis = 1
key = jax.random.PRNGKey(30)
data = jnp.arange(sh[0]*sh[1]) + 1
mask = jax.random.uniform(key, (sh[0]*sh[1],)) > 0.5
data = data.reshape(sh)
mask = mask.reshape(sh)

rows, cols = data.shape

s0 = data[:,0:cols-(cols%2)].reshape(-1,2)
res = s0.reshape(rows, cols-(cols%2))

if cols%2 == 1:
    end = jnp.expand_dims(data[:,-1], axis=axis)
    res = jnp.concatenate((res, end), axis=1)
assert(np.all(data == res))

s1 = data[:,1:cols-((cols+1)%2)].reshape(-1,2)
res = s1.reshape(rows, cols-1-((cols+1)%2))
beg = jnp.expand_dims(data[:,0], axis=1)

if cols%2 == 1:
    res = jnp.concatenate((beg, res), axis=1)
else:
    end = jnp.expand_dims(data[:,-1], axis=1)
    res = jnp.concatenate((beg, res, end), axis=1)
assert(np.all(data == res))


def rdmwlk_2D(key, data, mask, offset=0, axis=0):
    s0 = data[:,0:cols-(cols%2)].reshape(-1,2)
    res = s0.reshape(rows, cols-(cols%2))
    if cols%2 == 1:
        end = jnp.expand_dims(data[:,-1], axis=axis)
        res = jnp.concatenate((res, end), axis=1)

d = np.array(data)
offset = 0
axis = 1
l_axis = d.shape[axis]
start, stop = np.zeros(len(d.shape)), np.array(d.shape)
stop[axis] -= (l_axis%2)
s0 = d[:, 0:cols-(cols%2)].reshape(-1,2)
s1 = d[:, 0:cols-(cols%2)].reshape(-1,2)



##
def test(key, shp):
    dd = jax.random.uniform(key, shp)
    return dd.at[range(0,shp[0]//2)]

dd = jax.random.uniform(key, (100,100))
dd.at[range(0,)]

jit(test, static_argnums=1)(key, (1000,1000))

##

@jit
def pair_masked_shuffle(key, data, mask):
    assert(data.shape == mask.shape)
    assert(data.shape[1] == 2)
    keys = jax.random.split(key, d.shape[0])
    ds, ms = vmap(neighbor_shuffle)(keys, d, m)
    return ds.reshape(data.shape), ms.reshape(mask.shape)

@jit
def randomwalk_to_empty_2d(key, data, mask):
    k = jax.random.split(key, 4)
    d, m = pair_masked_shuffle(k[0], data.reshape(-1,2), mask.reshape(-1, 2))




#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


# example usage:

def init_state(params):


world = dc.channel_stack(channel('fluo', (3,)),channel('divide'), channel('morphogens', (2,)))
init_func_, update_func = dc.parallel(dc.sequential(perceive, model), dc.sequential(divide))

params = init_func(key)

init_state = world(world_shape, default_value=0).set('fluo', init_fig).set('alive')


# implementation

# Channels are just stored in a dict. The keys are the names of the channels.
# The values are the shapes of the channels.

# channel_stack returns a function that will instantiate the channels in the given world shape
# (to which we add each channel's own shape as extra dimensions)

def channel_stack(channels):
    def stack(world_shape, default_value=0):
        return {name: np.zeros(world_shape + shape) for name, shape in channels.items()}
    return stack

world = channel_stack({'fluo': (3,), 'divide': (1,), 'morphogens': (2,)})

# sequential accepts a sequence of transforms. Each transforms returns 2 functions: init and apply (aka update).
# An init function takes a key and returns a tuple of parameters. The parameters are passed to the apply functions.
# The return of Sequential is also a tuple of init and apply functions, which will dispatch to the transforms in the sequence.

def sequential(*transforms):

    ntransforms = len(transforms)
    init_funcs, apply_funcs = zip(*transforms)

    def init(key):
        _, keys = jax.random.split(key, num=ntransforms)
        return tuple(f(k) for f, k in zip(init_funcs, keys))

    def apply(params, key, world):
        _, keys = jax.random.split(key, num=ntransforms)
        # we feed the result of each transform to the next and return the final result
        for f, k in zip(apply_funcs, keys):
            world = f(params, k, world)
        return world

    return init, apply

def parallel(*transforms):

    ntransforms = len(transforms)
    init_funcs, apply_funcs = zip(*transforms)

    def init(key):
        _, keys = jax.random.split(key, num=ntransforms)
        return tuple(f(k) for f, k in zip(init_funcs, keys))

    def apply(params, key, world):
        _, keys = jax.random.split(key, num=ntransforms)
        # each transform is applied in parallel, we return a tuple of all the results
        return tuple(f(p, k, w) for f, p, k, w in zip(apply_funcs, params, keys, world))

    return init, apply

