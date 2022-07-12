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
# default matplotlib background white:
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

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

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{            --     compute reorder 1d experiment/proto     --
#···············································································

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


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     plot 2d     --
#···············································································
def plot_2d(pos, mask, title=None, fsize=1.):
    figsize = np.array(pos.shape[:2][::-1]) * fsize
    fig, ax = plt.subplots(figsize=figsize)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'my_colormap', ['#333333', '#FCDD80'], N=2
    )
    ax.pcolormesh(mask, edgecolors='k', alpha=1.0, cmap=cmap)
    r, c = np.where(mask)
    ax.set_ylim(ax.get_ylim()[::-1])
    # at all coordinates including the ones that are not alive
    for rr, cc in zip(*np.where(mask.astype(int) > -1)):
        ax.text(
            cc + 0.5,
            rr + 0.85,
            f'{rr},{cc}',
            ha='center',
            va='center',
            fontsize=10,
            color='#000000',
        )
    for rr, cc in zip(r, c):
        color = (
            'k' if int(rr) == int(pos[rr, cc, 0]) and int(cc) == int(pos[rr, cc, 1]) else '#BF1719'
        )
        ax.text(
            cc + 0.5,
            rr + 0.4,
            f'{int(pos[rr, cc, 0])},{int(pos[rr,cc,1])}',
            ha='center',
            va='center',
            fontsize=14,
            fontweight='bold',
            color=color,
        )
    if title is not None:
        ax.set_title(title)
    plt.show()

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

# now let's experiment with a 2d version.
# first with random impulses
WORLD_SIZE = 25
WS2D = (WORLD_SIZE, WORLD_SIZE)
alive = jax.random.bernoulli(jax.random.PRNGKey(0), 0.3, WS2D)
natural = jnp.stack(jnp.meshgrid(jnp.arange(WS2D[0]), jnp.arange(WS2D[1]), indexing='ij'), axis=2)
pos = natural + jax.random.uniform(jax.random.PRNGKey(0), natural.shape, maxval=0.99)
impulses = (
    jax.random.uniform(jax.random.PRNGKey(0), pos.shape, minval=-0.2, maxval=0.2) * alive[..., None]
)
pos += impulses
pos = pos.at[:, :, 0].set(jnp.clip(pos[:, :, 0], 0, WS2D[0] - 0.01))
pos = pos.at[:, :, 1].set(jnp.clip(pos[:, :, 1], 0, WS2D[1] - 0.01))
np.indices(WS2D)
desired = jnp.floor(pos).astype(int)

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{          --     prototyping code for compute_reorder_2d      --
#···············································································

# instead of just from_left and from_righht, from_neighbor should be filled
# with candidates from the whole 3x3 neighborhood (not including the center)
# So we are finding all neighbors that want to swich positions by checking
# where the array of desired positions, when shifted by 1 in every direction,
# is the same as the array of natural positions (i.e a neighbor wants to move here)


# let's start manually:

# -1, -1 (we check that if each top left neighbor wants to move)
eq = jnp.all(desired[1:, 1:, :] == natural[:-1, :-1, :], axis=2)
eq = jnp.pad(eq, ((0, 1), (0, 1)), 'constant', constant_values=False)
n0 = jnp.where(eq[:, :, None], natural, -1)

# 0, -1 (center left)
eq1 = jnp.all(desired[:, 1:, :] == natural[:, :-1, :], axis=2)
eq1 = jnp.pad(eq1, ((0, 0), (0, 1)), 'constant', constant_values=False)
n1 = jnp.where(eq1[:, :, None], natural, -1)

# 1, -1 (bottom left)
eq = jnp.all(desired[:-1, 1:, :] == natural[1:, :-1, :], axis=2)
eq = jnp.pad(eq, ((1, 0), (0, 1)), 'constant', constant_values=False)
n2 = jnp.where(eq[:, :, None], natural, -1)

# 0, 1 (center right)
eq3 = jnp.all(desired[:, :-1, :] == natural[:, 1:, :], axis=2)
eq3 = jnp.pad(eq3, ((0, 0), (1, 0)), 'constant', constant_values=False)
n3 = jnp.where(eq3[:, :, None], natural, -1)


# and now as a generalized version of the above
@partial(jit, static_argnums=(1, 2))
def get_n(desired, i, j):
    # todo: try using roll instead of pad and slices
    natural = jnp.stack(
        jnp.meshgrid(jnp.arange(desired.shape[0]), jnp.arange(desired.shape[1]), indexing='ij'),
        axis=2,
    )

    def start_end(i, j):
        start = (max(i, 0), max(j, 0), 0)
        end = (desired.shape[0] - max(-i, 0), desired.shape[1] - max(-j, 0), desired.shape[2])
        return start, end

    shift = start_end(i, j)
    anti_shift = start_end(-i, -j)
    eq = jnp.all(lax.slice(desired, *anti_shift) == lax.slice(natural, *shift), axis=2)
    n = jnp.where(eq[:, :, None], lax.slice(natural, *anti_shift), -1)
    return jnp.pad(
        n,
        ((max(i, 0), max(-i, 0)), (max(j, 0), max(-j, 0)), (0, 0)),
        'constant',
        constant_values=-1,
    )



# now we can use the above function to get all 3x3 neighbors (except center)
neighbors = [get_n(desired, i, j) for i in range(-1, 2) for j in range(-1, 2) if i != 0 or j != 0]
stacked_neighbor = jnp.stack(neighbors, axis=2)


# and now we just need to grab 1 from each neighboorhood (one that wants to move if available)
s = jnp.expand_dims(jnp.sum(stacked_neighbor, axis=3), axis=3)
a = jnp.expand_dims(jnp.argmax(s, axis=2), axis=2)
from_neighbor = jnp.take_along_axis(stacked_neighbor, a, axis=2).squeeze()

# yay we have from_neighbor, we can compute reorder now.
# again, from_neighbor now contains either a pair of -1, or the coordinates of the neighbor that wants to move
# at this location
# reorder will contain the new indexing that allows to move elements in the array to the right location when they
# desire it and when it's not in conflict with other elements that are alive.
# (And when multiple want to move at the same location, just one has been picked by from_neighbor).


going = (natural != desired).any(axis=2)  # indicates that a cell wants to move

# oh no! a cell is alive at a desired position. It's not possible to move there.
# could also be an alive cell that just wants to stay where it is though, in which case ok!
alive_at_desired = alive[desired[:, :, 0], desired[:, :, 1]]
neighbor_desired = from_neighbor[desired[:, :, 0], desired[:, :, 1]]

# selected to move indicates for a cell that wants to move, that indeed it is expected as its new location.
# this is a useful check in case several cells want to move at the same location (only one is selected)
selected_to_move = (neighbor_desired == natural).all(axis=2)
# I think ~alive_at_desired is not necessary since selected_to_move should be false for alive cells
# but I'm really not sure yet so we'll keep it for now.

reorder = jnp.where(
    # TODO: simplify. Only 2 where are needed.
    alive[:, :, None],  # where there were alive cells
    jnp.where(
        (
            (going)  # where they want to move
            & (~alive_at_desired)  # and no alive cell is already at the desired position
            & (selected_to_move)  # and this cell has been selected to move
        )[:, :, None],
        desired,  # then we want to pick the element at the desired position
        natural,  # else we don't change this element
    ),
    jnp.where(
        from_neighbor >= 0,  # wherever there was no alive cell, and we have a neighbor coming
        from_neighbor,  # we welcome it
        natural,
    ),
)
reorder = tuple(reorder[:, :, i] for i in range(2))

plot_2d(desired, alive, fsize=0.8)
plot_2d(desired[reorder], alive[reorder], fsize=0.8)



#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                --     compute reorder 2d function     --
#···············································································
# This function will return the new indexing that allows to move the alive elements in desired
# to the neighboring index the point to (max distance (1,1)), as long as there is no alive cell
# at the desired position. If multiple cells want to move at the same position, only one will;
# currently the one with the "max" coordinates, i.e it favors lower-right-most indices.

def compute_reorder_2d(desired, alive):
    natural = jnp.stack(
            jnp.meshgrid(jnp.arange(desired.shape[0]), jnp.arange(desired.shape[1]), indexing='ij'),
            axis=2,
        )
    def get_n(i, j):
        # TODO[optim]: try using roll instead of pad and slices
        def start_end(i, j):
            start = (max(i, 0), max(j, 0), 0)
            end = (desired.shape[0] - max(-i, 0), desired.shape[1] - max(-j, 0), desired.shape[2])
            return start, end

        shift = start_end(i, j)
        anti_shift = start_end(-i, -j)
        eq = jnp.all(lax.slice(desired, *anti_shift) == lax.slice(natural, *shift), axis=2)
        n = jnp.where(eq[:, :, None], lax.slice(natural, *anti_shift), -1)
        return jnp.pad(
            n,
            ((max(i, 0), max(-i, 0)), (max(j, 0), max(-j, 0)), (0, 0)),
            'constant',
            constant_values=-1,
        )
    neighbors = [get_n(i, j) for i in range(-1, 2) for j in range(-1, 2) if i != 0 or j != 0]
    stacked_neighbor = jnp.stack(neighbors, axis=2)
    s = jnp.expand_dims(jnp.sum(stacked_neighbor, axis=3), axis=3)
    a = jnp.expand_dims(jnp.argmax(s, axis=2), axis=2)
    from_neighbor = jnp.take_along_axis(stacked_neighbor, a, axis=2).squeeze()

    going = (natural != desired).any(axis=2)  # indicates that a cell wants to move

    # oh no! a cell is alive at a desired position. It's not possible to move there.
    # could also be an alive cell that just wants to stay where it is though, in which case ok!
    alive_at_desired = alive[desired[:, :, 0], desired[:, :, 1]]
    neighbor_desired = from_neighbor[desired[:, :, 0], desired[:, :, 1]]

    # selected to move indicates for a cell that wants to move, that indeed it is expected as its new location.
    # this is a useful check in case several cells want to move at the same location (only one is selected)
    selected_to_move = (neighbor_desired == natural).all(axis=2)
    # I think ~alive_at_desired is not necessary since selected_to_move should be false for alive cells
    # but I'm really not sure yet so we'll keep it for now.
    big_cond = ( (going)  # where they want to move
                & (~alive_at_desired)  # and no alive cell is already at the desired position
                & (selected_to_move)  # and this cell has been selected to move
            )

    reorder = jnp.where(
        # TODO[optim]: simplify. Only 2 where are needed.
        alive[:, :, None],  # where there were alive cells
        jnp.where(
            big_cond[:, :, None],
            desired,  # then we want to pick the element at the desired position
            natural,  # else we don't change this element
        ),
        jnp.where(
            from_neighbor >= 0,  # wherever there was no alive cell, and we have a neighbor coming
            from_neighbor,  # we welcome it
            natural,
        ),
    )

    big_cond_and_alive = (big_cond & alive)

    # reorder = jnp.where(
            # (~alive[:,:,None] & from_neighbor < 0) | (alive & ~big_cond)[:,:,None],
            # natural,
        # jnp.where(big_cond_and_alive[:,:,None], desired, from_neighbor,),
        # natural
    # )

    # natural where from_neighbor < 0 & not alive or alive & not big cond
    return (reorder[:, :, 0], reorder[:, :, 1])


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

reorder = compute_reorder_2d(desired, alive)
plot_2d(desired, alive, fsize=0.8)

##
def plot_positions_and_gradient(m, g, title=None, fsize=(10, 10), csize=5):
    fig, ax = plt.subplots(figsize=fsize)
    ax.imshow(g, cmap='Reds', alpha=0.5)
    masked_data = np.ma.masked_where(~m, m)
    ax.imshow(masked_data, cmap='Greys_r', interpolation='none')
    if title is not None:
        ax.set_title(title)
    plt.show()

# now let's experiment with a 2d version.
# first with random impulses
WORLD_SIZE = 350
attr_intensity = 10
rep_intensity = 1000
attr_radius = 31.0/WORLD_SIZE
rep_radius = 3.0/WORLD_SIZE
WS2D = (WORLD_SIZE, WORLD_SIZE)
alive = jax.random.bernoulli(jax.random.PRNGKey(0), 0.03, WS2D)
natural = jnp.stack(jnp.meshgrid(jnp.arange(WS2D[0]), jnp.arange(WS2D[1]), indexing='ij'), axis=2)
k_attract = diffuse_kernel(attr_radius, WORLD_SIZE)
pos = natural.astype(jnp.float32)
k_repel = diffuse_kernel(rep_radius, WORLD_SIZE)
attractions = convolve(alive, k_attract, mode='same')
repulsions = convolve(alive, k_repel, mode='same')
plot_positions_and_gradient(alive, attractions)

def step(alive_pos, _):
    alive, pos = alive_pos
    attractions = convolve(alive, k_attract, mode='same')
    repulsions = convolve(alive, k_repel, mode='same')
    # apply sobel to get attraction gradient in x and y:
    attr_g = jnp.clip(jnp.stack(jnp.gradient(attractions), axis=2) * attr_intensity, -1, 1)
    rep_g = jnp.clip(jnp.stack(jnp.gradient(repulsions), axis=2) * rep_intensity, -1, 1)
    pos = pos + attr_g - rep_g
    pos = pos.at[:, :, 0].set(jnp.clip(pos[:, :, 0], 0, WS2D[0] - 0.01))
    pos = pos.at[:, :, 1].set(jnp.clip(pos[:, :, 1], 0, WS2D[1] - 0.01))
    desired = jnp.floor(pos).astype(int)
    reorder = compute_reorder_2d(desired, alive)
    alive = alive[reorder]
    pos = pos[reorder]
    pos = jnp.clip(pos, natural, natural + 0.99999)
    return (alive, pos), None

@partial(jit, static_argnums=(2,))
def n_steps(alive, pos, n):
    final_state, _ = jax.lax.scan(step, (alive, pos), None, length=n)
    return final_state

# for i in range(1000):
    # alive, pos, attr, rep = step(alive, pos)
    # if i % 50 == 0:
        # plot_positions_and_gradient(alive, attr, title=f'{i}')
        # print(alive.sum())

for i in range(20):
    alive, pos = n_steps(alive, pos, 50)
    plot_positions_and_gradient(alive, convolve(alive, k_attract, mode='same'), title=f'{i}')
    print(alive.sum())


%timeit jit(step)((alive, pos), None)[0][0].block_until_ready()
%timeit n_steps(alive, pos, 100)[0].block_until_ready()


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                 --     version without indexing     --
#···············································································

# doing things through reindexing is porbably slower than it could be. 
# here I want to try to do everything without indexing at all.
# Since a desired index can only be 1 away (in every dimensions), 
# the key is to treat one shift after another. 

WS2D = (60,100)
alive = jax.random.bernoulli(jax.random.PRNGKey(0), 0.35, WS2D)
natural = jnp.stack(jnp.meshgrid(jnp.arange(WS2D[0]), jnp.arange(WS2D[1]), indexing='ij'), axis=2)
pos = natural + jax.random.uniform(jax.random.PRNGKey(0), natural.shape, maxval=0.99)
impulses = (
    jax.random.uniform(jax.random.PRNGKey(0), pos.shape, minval=-0.6, maxval=0.6) * alive[..., None]
)
pos += impulses
pos = pos.at[:, :, 0].set(jnp.clip(pos[:, :, 0], 0, WS2D[0] - 0.01))
pos = pos.at[:, :, 1].set(jnp.clip(pos[:, :, 1], 0, WS2D[1] - 0.01))
np.indices(WS2D)
desired = jnp.floor(pos).astype(int)


i,j = -1,1
data = pos

def get_n(i, j):

        def start_end(i, j):
            start = (max(i, 0), max(j, 0), 0)
            end = (desired.shape[0] - max(-i, 0), desired.shape[1] - max(-j, 0), desired.shape[2])
            return start, end
        shift = start_end(i, j)
        anti_shift = start_end(-i, -j)

        eq = jnp.all(lax.slice(desired, *anti_shift) == lax.slice(natural, *shift), axis=2)
        n = jnp.where(eq[:, :, None], lax.slice(natural, *anti_shift), -1)
        return jnp.pad(
            n,
            ((max(i, 0), max(-i, 0)), (max(j, 0), max(-j, 0)), (0, 0)),
            'constant',
            constant_values=-1,
        )


def update_data_noind(alive, data, desired):

    movement = desired - natural


    def with_offset(alive, data, i, j):

        def start_end(i, j):
            start = (max(i, 0), max(j, 0))
            end = (alive.shape[0] - max(-i, 0), alive.shape[1] - max(-j, 0))
            return start, end

        offset=start_end(i,j)
        counter_offset = start_end(-i,-j)



        going = alive & (movement == jnp.array((i,j))[None, None, :]).all(axis=2)  # indicates that a cell wants to move in the current offset's direction

        # offsets with the padding based technique
        receiving = lax.slice(going, *offset)
        offset_data = lax.slice(data,  offset[0]+(0,), offset[1]+(data.shape[2],))
        target_alive = lax.slice(alive, *counter_offset)
        offset_going = lax.slice(going, *counter_offset)
        offset_alive = lax.slice(alive, *offset)
        counter_data= lax.slice(data, counter_offset[0]+(0,), counter_offset[1]+(data.shape[2],))
        new_alive = (offset_alive & (~offset_going | target_alive)) | (~offset_alive & receiving)
        new_data = jnp.where((receiving & ~offset_alive)[:,:,None], offset_data, counter_data)
        new_alive = jnp.pad(new_alive, ((max(i, 0), max(-i, 0)), (max(j, 0), max(-j, 0))), 'constant', constant_values=False)
        new_data = jnp.pad(new_data, ((max(i, 0), max(-i, 0)), (max(j, 0), max(-j, 0)), (0, 0)), 'constant', constant_values=0)

        # with rolling:

        # offset=(i,j)
        # counter_offset = (-i,-j)
        # receiving = jnp.roll(going, offset, axis=(0,1))
        # offset_data = jnp.roll(data, offset, axis=(0,1))
        # target_alive = jnp.roll(alive, counter_offset, axis=(0,1))
        # new_alive = (alive & (~going | target_alive)) | (~alive & receiving)
        # new_data = jnp.where((receiving & ~alive)[:,:,None], offset_data, data)

        return new_alive, new_data

    a, d = alive, data
    a, d = with_offset(a, d, -1,-1)
    a, d = with_offset(a, d, -1, 0)
    a, d = with_offset(a, d, -1, 1)
    a, d = with_offset(a, d,  0,-1)
    a, d = with_offset(a, d,  0, 1)
    a, d = with_offset(a, d,  1,-1)
    a, d = with_offset(a, d,  1, 0)
    a, d = with_offset(a, d,  1, 1)

    # with lax scan:
    # possible_offsets = [(i,j) for i in range(-1,2) for j in range(-1,2) if i != 0 or j != 0]

    return a, d

# plot_2d(pos, alive)
a, p = update_data_noind(alive, pos, desired)
# plot_2d(p, a)
print(alive.sum())
print(a.sum())

##

import jax.scipy as jsp

WORLD_SIZE = 512
attr_intensity = 10
rep_intensity = 1000
attr_radius = 3.0/WORLD_SIZE
rep_radius = 3.0/WORLD_SIZE
WS2D = (WORLD_SIZE, WORLD_SIZE)
alive = jax.random.bernoulli(jax.random.PRNGKey(0), 0.03, WS2D)
natural = jnp.stack(jnp.meshgrid(jnp.arange(WS2D[0]), jnp.arange(WS2D[1]), indexing='ij'), axis=2)
k_attract = diffuse_kernel(attr_radius, WORLD_SIZE)
pos = natural.astype(jnp.float32)
k_repel = diffuse_kernel(rep_radius, WORLD_SIZE)
attractions = convolve(alive, k_attract, mode='same')
repulsions = convolve(alive, k_repel, mode='same')
plot_positions_and_gradient(alive, attractions)


def sob_gradient(arr):
    sobel_x = jnp.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]])
    return jnp.stack((jsp.signal.convolve(arr, sobel_x.transpose(), mode='same'), jsp.signal.convolve(arr, sobel_x, mode='same')), axis=2)

def jnp_grad(arr):
    return jnp.stack(jnp.gradient(arr), axis=2)

def dumb_grad(arr):
    r = arr[:-1,:] - arr[1:,:]
    c = arr[:,:-1] - arr[:,1:]
    # pad with zeros
    r = jnp.pad(r, ((0,1),(0,0)), mode='constant')
    c = jnp.pad(c, ((0,0),(0,1)), mode='constant')
    return jnp.stack((r, c), axis=2)

# jit(sob_gradient)(attractions).block_until_ready()
# %timeit jit(sob_gradient)(attractions).block_until_ready()

# jit(jnp_grad)(attractions).block_until_ready()
# %timeit jit(jnp_grad)(attractions).block_until_ready()

# jit(dumb_grad)(attractions).block_until_ready()
# %timeit jit(dumb_grad)(attractions).block_until_ready()

##
def step(alive_pos, _):
    alive, pos = alive_pos
    # attractions = convolve(alive, k_attract, mode='same')
    repulsions = convolve(alive, k_repel, mode='same')
    # attr_g = jnp.clip(jnp.stack(jnp.gradient(attractions), axis=2) * attr_intensity, -1, 1)
    rep_g = jnp.clip(jnp.stack(jnp.gradient(repulsions), axis=2) * rep_intensity, -1, 1)
    # pos = pos + attr_g - rep_g
    pos = pos - rep_g
    pos = pos.at[:, :, 0].set(jnp.clip(pos[:, :, 0], 0, WS2D[0] - 0.01))
    pos = pos.at[:, :, 1].set(jnp.clip(pos[:, :, 1], 0, WS2D[1] - 0.01))
    desired = jnp.floor(pos).astype(int)
    a, p = update_data_noind(alive, pos, desired)
    p = jnp.clip(p, natural, natural + 0.99999)
    return (a, p), None

@partial(jit, static_argnums=(2,))
def n_steps(alive, pos, n):
    final_state, _ = jax.lax.scan(step, (alive, pos), None, length=n)
    return final_state

n_steps(alive, pos, 100)[0].block_until_ready()
%timeit n_steps(alive, pos, 100)[0].block_until_ready()
##
# for i in range(1000):
    # alive, pos, attr, rep = step(alive, pos)
    # if i % 50 == 0:
        # plot_positions_and_gradient(alive, attr, title=f'{i}')
        # print(alive.sum())

for i in range(20):
    alive, pos = n_steps(alive, pos, 50)
    plot_positions_and_gradient(alive, convolve(alive, k_attract, mode='same'), title=f'{i}')
    print(alive.sum())

# (a, p), _ = step((alive, pos), None)
# plot_positions_and_gradient(a, convolve(alive, k_attract, mode='same'), title=f'{i}')
# print(alive.sum())
# print(a.sum())

def update_data_ind(alive, data, desired):
    reorder = compute_reorder_2d(desired, alive)
    a = alive[reorder]
    d = data[reorder]
    return a,d

desired = jnp.floor(pos).astype(int)
jit(update_data_ind)(alive, pos, desired)[0].block_until_ready()
jit(update_data_noind)(alive, pos, desired)[0].block_until_ready()

%timeit jit(update_data_noind)(alive, pos, desired)[0].block_until_ready()
%timeit jit(update_data_ind)(alive, pos, desired)[0].block_until_ready()

jit(step)((alive, pos), None)[0][0].block_until_ready()
%timeit jit(step)((alive, pos), None)[0][0].block_until_ready()





#                                                                            }}}
## ──────────────────────────────────────────────────────────────────────────data


