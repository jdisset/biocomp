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

