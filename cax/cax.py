## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     imports     --
#···············································································
import jax
from jax._src.api import block_until_ready
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from rich import print
from jax import jit, vmap, lax
from jax import tree_util as pytree
import jax.numpy as jnp

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     core functions     --
#···············································································
def make_world(state_layers, shape):


def set_perception(percept_layers):
    pass

def update(state_layers, perception_stack, n_steps):
    pass


def channel(name, shape=(1,), dtype=float):
    def set(val):
        return {name:val.astype(dtype)}

    def zeros(world_shape):
        return set(jnp.zeros(world_shape + shape).squeeze())

    return zeros

def channel_stack(*channels):
    def build(world_shape):
        out = {}
        for c in channels:
            out.update(c(world_shape))
        return out
    return build

def channel_tree():
    pass

# w.set_perception({'m0':cax.diffusion('m0', 0.2), 'm1':cax.diffusion('m1', 0.5)})

c = channel_stack(channel('fluo', (3,)), channel('d'), channel('morphogens', (2,)))



def model(perception):
    ## fluo[0] = perception['morphogens'][0]
    
    out = dosomething(jax.tree_leaves())

    return apply(perception)



#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


##

plt.rcParams['figure.facecolor'] = 'white'


def plot(ar):
    data = ar if len(ar.shape) > 1 else np.array([ar])
    heatmap = plt.pcolor(data, vmin=0, vmax=1)
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            plt.text(x + 0.5, y + 0.5, '%i' % data[y, x],
                     horizontalalignment='center',
                     verticalalignment='center',
                      )
    plt.show()



def unison_shuffled(key, a, b):
    p = jax.random.permutation(key, len(a))
    return a[p], b[p]

def neighbor_shuffle(k, ar, m):
    return lax.cond(m[0] == m[1], lambda *_: (ar, m), lambda *_: unison_shuffled(k, ar, m), None)

def neighbor_shuffle_nobranch(k, ar, m):
    ars, ms = unison_shuffled(k, ar,m)
    return ar * m + ars * (~ms | ~m), ms


sh = (1,3)
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
    d, m = pair_masked_shuffle(k[1], data.reshape(-1,2), mask.reshape(-1, 2))


def randomwalk_to_empty(key, data, mask):

d0, m0 = data, mask
for k in jax.random.split(key, 10):
    plot(data*mask)
    data, mask = randomwalk_to_empty(k, data, mask)
    print(np.mean((m0.astype(int) - mask.astype(int))**2))


