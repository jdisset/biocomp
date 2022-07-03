## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     imports     --
# ···············································································
from jax.tree_util import Partial as partial
from jax import tree_util as pytree
import matplotlib.pyplot as plt
import matplotlib
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

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                  --     direct descent approach     --
# ···············································································
def paired_descent_swap(gradient, mask, data):
    # all arrays are arrays of pairs (shape [n, 2])
    # we want to reorder the pairs so that lowest gradient is first
    G = (gradient[1] - gradient[0]) > 0
    print(G)
    order = jnp.array([G != mask[0], G == mask[0]]).astype(int)
    return mask[order], data[order]


od = jnp.array([0, 1])
om = jnp.array([0, 1]).astype(bool)
m, d = paired_descent_swap(jnp.array([1.0, 2.0]), om, od)
assert m.shape == d.shape == (2,)
assert np.all(m == om[::-1])
assert np.all(d == od[::-1])

m, d = paired_descent_swap(jnp.array([2.0, 1.0]), om, od)
assert m.shape == d.shape == (2,)
assert np.all(m == om)
assert np.all(d == od)


# if one of the cell in the pair is empty, i.e if mask[0] != mask[1],
# then we move the non-empty cell to the empty one if the gradient is lower in the new cell


def neighbor_masked_gradient_descent_nobranch(gradient, mask, data):
    ms, ds = paired_descent_swap(gradient, mask, data)
    return ms, data * mask + ds * (~ms | ~mask)


def neighbor_masked_gradient_descent_cond(gradient, mask, data):
    return lax.cond(
        (mask[0] == mask[1]) | (gradient[0] == gradient[1]),
        lambda *_: (mask, data),
        lambda *_: paired_descent_swap(gradient, mask, data),
        None,
    )


od = jnp.array([0, 1])
om = jnp.array([0, 1]).astype(bool)
m, d = neighbor_masked_gradient_descent_cond(jnp.array([1.0, 2.0]), om, od)
assert m.shape == d.shape == (2,)
assert np.all(m == om[::-1])
assert np.all(d * m == od[::-1] * m)
m, d


m, d = neighbor_masked_gradient_descent_cond(jnp.array([2.0, 1.0]), om, od)
assert m.shape == d.shape == (2,)
assert np.all(m == om)
assert np.all(d * m == od * m)
m, d


om = jnp.array([1, 1]).astype(bool)
m, d = neighbor_masked_gradient_descent_cond(jnp.array([2.0, 1.0]), om, od)
assert m.shape == d.shape == (2,)
assert np.all(m == om)
assert np.all(d * m == od * om)
m, d

om = jnp.array([0, 0]).astype(bool)
m, d = neighbor_masked_gradient_descent_cond(jnp.array([2.0, 1.0]), om, od)
assert m.shape == d.shape == (2,)
assert np.all(m == om)
assert np.all(d * m == od * om)
m, d


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


@partial(jit, static_argnums=(3,))
def rebuild_array(pre, mid, post, axis):
    return jnp.concatenate([pre, mid, post], axis=axis)


a = jax.random.uniform(jax.random.PRNGKey(0), (10, 10))
pre, mid, post = cut_array(a, 0, 2, axis=0)
assert np.array(pre).size == 0
assert np.array(post).size == 0
assert np.array(mid).shape == (10, 10)
assert np.all(rebuild_array(pre, mid, post, axis=0) == a)


a = jax.random.uniform(jax.random.PRNGKey(0), (10, 10))
pre, mid, post = cut_array(a, 1, 2, axis=0)
assert np.array(pre).shape == (1, 10)
assert np.array(post).shape == (1, 10)
assert np.array(mid).shape == (8, 10)
assert np.all(rebuild_array(pre, mid, post, axis=0) == a)

a = jax.random.uniform(jax.random.PRNGKey(0), (10, 8))
pre, mid, post = cut_array(a, 0, 2, axis=1)
assert np.array(pre).size == 0
assert np.array(post).size == 0
assert np.array(mid).shape == (10, 8)
assert np.all(rebuild_array(pre, mid, post, axis=1) == a)

a = jax.random.uniform(jax.random.PRNGKey(0), (10, 8))
pre, mid, post = cut_array(a, 1, 2, axis=1)
assert np.array(pre).shape == (10, 1)
assert np.array(post).shape == (10, 1)
assert np.array(mid).shape == (10, 6)
assert np.all(rebuild_array(pre, mid, post, axis=1) == a)

a = jax.random.uniform(jax.random.PRNGKey(0), (10, 7))
pre, mid, post = cut_array(a, 1, 2, axis=1)
assert np.array(pre).shape == (10, 1)
assert np.array(post).size == 0
assert np.array(mid).shape == (10, 6)
assert np.all(rebuild_array(pre, mid, post, axis=1) == a)


def plot_positions_and_gradient(m, g, d, title=None, fsize=(10, 10)):
    fig, ax = plt.subplots(figsize=fsize)
    ax.imshow(g, cmap='Reds', alpha=0.5)
    y, x = np.where(m > 0.0)
    # scatter with a cross symbol
    ax.scatter(x, y, c='black', s=20)
    # values = [d[xx, yy] for xx, yy in zip(x, y)]
    # for xx, yy, v in zip(x, y, values):
    # ax.text(xx, yy, str(v), fontsize=10, ha='center', va='center')
    # ax.set_xticks([])
    # ax.set_yticks([])
    if title is not None:
        ax.set_title(title)
    plt.show()


@partial(jit, static_argnums=(3, 4))
def masked_gradient_descent(gradient, mask, data, axis, offset):
    sliced_g = cut_array(gradient, offset, 2, axis=axis)
    sliced_m = cut_array(mask, offset, 2, axis=axis)
    sliced_d = cut_array(data, offset, 2, axis=axis)
    m, d = vmap(neighbor_masked_gradient_descent_cond)(
        sliced_g[1].reshape(-1, 2), sliced_m[1].reshape(-1, 2), sliced_d[1].reshape(-1, 2)
    )
    m, d = vmap(neighbor_masked_gradient_descent_cond)(
        sliced_g[1].reshape(-1, 2), sliced_m[1].reshape(-1, 2), sliced_d[1].reshape(-1, 2)
    )
    m = rebuild_array(sliced_m[0], m.reshape(sliced_m[1].shape), sliced_m[2], axis=axis)
    d = rebuild_array(sliced_d[0], d.reshape(sliced_d[1].shape), sliced_d[2], axis=axis)
    return m, d


##
# make default matplotlib background white:
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'


WORLD_SIZE = 100
m = jax.random.bernoulli(jax.random.PRNGKey(0), 0.1, (WORLD_SIZE, WORLD_SIZE)).astype(np.float32)
d = np.arange(m.size).reshape(m.shape)
k_attract = diffuse_kernel(0.75, WORLD_SIZE)
k_repel = diffuse_kernel(0.5, WORLD_SIZE)
attract_intensity = 0.5
repel_intensity = 0.25
repulsions = convolve(m, k_repel, mode='same') * repel_intensity
g = repulsions


def gr(m, g, d):
    mm, dd = masked_gradient_descent(g, m, d, axis=0, offset=0)
    return jnp.sum(mm * dd)


plot_positions_and_gradient(m, grad(gr)(m, g, d), d)


plot_positions_and_gradient(m, g, d, title='init')
for i in range(60):
    attractions = convolve(m, k_attract, mode='same') * attract_intensity
    repulsions = convolve(m, k_repel, mode='same') * repel_intensity
    m, d = masked_gradient_descent(g, m, d, axis=0, offset=0)
    m, d = masked_gradient_descent(g, m, d, axis=1, offset=0)
    m, d = masked_gradient_descent(g, m, d, axis=0, offset=1)
    m, d = masked_gradient_descent(g, m, d, axis=1, offset=1)
    plot_positions_and_gradient(m, g, d, title=f'iteration {i+1}')


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

# let's now explore a different way.
# We'll try first with the 1d case.
# We keep an array of the (float) positions of the cells.
# 1 cell per element of the array, i.e no overlap.
# We also use a mask to keep track of which cells are alive.
# The int part of the postion of the cell is its index in the array.
# When we update positions, we need to move cells whose position is in a new index.


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                           --     print     --
# ···············································································
def prnt(pos, mask=None):
    # print horizintal line:
    print()
    print('|', end='')
    for i, x in enumerate(pos):
        if mask is not None and mask[i]:
            print(f'\033[1m\033[32m  {x:.1f} \033[0m |', end='')
        else:
            print(f'  {x:.1f}  |', end='')
    print()


#   if mask is not None:
# print(' ' + '-' * ((8 * len(pos)) - 1))
# print('|', end='')
# for m in mask:
# if m:
# print(f'\033[32m   {int(m)}  \033[0m |', end='')
# else:
# print('       |', end='')
# print()
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

# default matplotlib background white:
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'




# a live cell that doesn't want to move has the priority over a neighbor that wants to move in the same index.
# we also need to handle the case where there was multiple neighbors that wanted to move in the same index.
# to do so we can check which neighbor has been selected to move (through from_neighbor)


@jit
def compute_reorder_1d(desired, alive):
    desired = desired.astype(np.int32)
    natural = jnp.arange(len(desired))
    from_left = jnp.where(desired[natural[:-1]] == natural[1:], natural[:-1], -1)
    from_left = jnp.pad(from_left, (1, 0), 'constant', constant_values=-1)
    from_right = jnp.where(desired[natural[1:]] == natural[:-1], natural[1:], -1)
    from_right = jnp.pad(from_right, (0, 1), 'constant', constant_values=-1)
    from_neighbor = jnp.stack([from_left, from_right], axis=0)
    from_neighbor = jnp.max(from_neighbor, axis=0)
    reorder = jnp.where(
        alive,
        jnp.where(
            (desired != natural) & (~alive[desired]) & (from_neighbor[desired] == natural),
            desired,
            natural,
        ),
        jnp.where(from_neighbor >= 0, from_neighbor, natural),
    )
    return reorder


WS = 100
pos = jax.random.uniform(jax.random.PRNGKey(0), (WS,), maxval=0.99) + jnp.arange(WS)
alive = jax.random.bernoulli(jax.random.PRNGKey(0), 0.3, (WS,))
impulses = jax.random.uniform(jax.random.PRNGKey(0), (WS,), minval=-0.99, maxval=0.99)
pos += impulses * alive
pos = jnp.clip(pos, 0, WS - 1)
desired = jnp.floor(pos)
reorder = compute_reorder_1d(desired, alive)

# todo: clip the ones that havent't moved
prnt(pos, alive)
prnt(pos[reorder], alive[reorder])

## 2d
def plot_2d(pos, mask, title=None, fsize=1.25):
    figsize = (np.array(pos.shape[:2][::-1])*fsize)
    fig, ax = plt.subplots(figsize=figsize)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'my_colormap', ['#333333', '#FCDD80'], N=2
    )
    ax.pcolormesh(mask, edgecolors='k', alpha=1.0, cmap=cmap)
    r, c = np.where(mask)
    ax.set_ylim(ax.get_ylim()[::-1])
    # at all coordinates including the ones that are not alive
    for rr, cc in zip(*np.where(mask.astype(int)> -1)):
        ax.text(cc + 0.5, rr + 0.85, f'{rr},{cc}', ha='center', va='center', fontsize=10, color='#000000')
    for rr, cc in zip(r, c):
        color = 'k' if int(rr) == int(pos[rr,cc,0]) and int(cc) == int(pos[rr,cc,1]) else '#BF1719'
        ax.text(cc + 0.5, rr + 0.4, f'{int(pos[rr, cc, 0])},{int(pos[rr,cc,1])}', ha='center', va='center', fontsize=14, fontweight='bold', color=color)
    if title is not None:
        ax.set_title(title)
    plt.show()

WS2D = (3,4)
alive = jax.random.bernoulli(jax.random.PRNGKey(0), 0.3, WS2D)
natural = jnp.stack(jnp.meshgrid(jnp.arange(WS2D[0]), jnp.arange(WS2D[1]),indexing='ij'), axis=2)
pos = natural + jax.random.uniform(jax.random.PRNGKey(0), natural.shape, maxval=0.99)
impulses = (
    jax.random.uniform(jax.random.PRNGKey(0), pos.shape, minval=-0.2, maxval=0.2)
    * alive[..., None]
)
pos += impulses
pos = pos.at[:, :, 0].set(jnp.clip(pos[:, :, 0], 0, WS2D[0] - 0.01))
pos = pos.at[:, :, 1].set(jnp.clip(pos[:, :, 1], 0, WS2D[1] - 0.01))
np.indices(WS2D)
desired = jnp.floor(pos)
plot_2d(desired, alive)

# instead of just from_left and from_righht, from_neighbor should be filled
# with candidates from the whole 3x3 neighborhood (not including the center)
# So we are finding all neighbors that want to swich positions by checking 
# where the array of desired positions, when shifted by 1 in every direction, 
# is the same as the array of natural positions (i.e a neighbor wants to move here)


# let's start manually:

# -1, -1 (we check that if each top left neighbor wants to move)
eq = jnp.all(desired[1:,1:, :] == natural[:-1,:-1, :], axis=2)
eq = jnp.pad(eq, ((0, 1), (0, 1)), 'constant', constant_values=False)
n0 = jnp.where(eq[:,:,None], natural, -1)

# 0, -1 (center left)
eq1 = jnp.all(desired[:,1:, :] == natural[:,:-1, :], axis=2)
eq1 = jnp.pad(eq1, ((0, 0), (0, 1)), 'constant', constant_values=False)
n1 = jnp.where(eq1[:,:,None], natural, -1)

# 1, -1 (bottom left)
eq = jnp.all(desired[:-1,1:, :] == natural[1:,:-1, :], axis=2)
eq = jnp.pad(eq, ((1, 0), (0, 1)), 'constant', constant_values=False)
n2 = jnp.where(eq[:,:,None], natural, -1)

# 0, 1 (center right)
eq3 = jnp.all(desired[:,:-1, :] == natural[:,1:, :], axis=2)
eq3 = jnp.pad(eq3, ((0, 0), (1, 0)), 'constant', constant_values=False)
n3 = jnp.where(eq3[:,:,None], natural, -1)


# and now as a generalized version of the above
def get_n(desired, natural, i, j):
    def start_end(i,j):
        start = (max(i, 0), max(j, 0),  0)
        end = (desired.shape[0] - max(-i, 0), desired.shape[1] - max(-j, 0), desired.shape[2])
        return start, end
    shift = start_end(i,j)
    anti_shift = start_end(-i,-j)
    eq = jnp.all(lax.slice(desired, *anti_shift) == lax.slice(natural, *shift), axis=2)
    n = jnp.where(eq[:,:,None], lax.slice(natural, *anti_shift), -1)
    return jnp.pad(n, ((max(i, 0), max(-i, 0)), (max(j, 0), max(-j, 0)), (0,0)), 'constant', constant_values=-1)

plot_2d(desired, alive)

# now we can use the above function to get all 3x3 neighbors (except center)
neighbors = [get_n(desired, natural, i, j) for i in range(-1,2) for j in range(-1,2) if i != 0 or j != 0]
from_neighbor = jnp.stack(neighbors, axis=2)

# and now we just need to grap 1 from each neighboorhood (one that wants to move if available)
s = jnp.sum(from_neighbor, axis=3)
a = jnp.argmax(s, axis=2)
# a gives us the index of the neighbor that wants to move, (id of the neighbor along the 3rd axis of from_neighbor)
from_neighbor[..., a].shape # not working. Should use unravel_index maybe?

