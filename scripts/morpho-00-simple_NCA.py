## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     imports     --
# ···············································································

import matplotlib.pyplot as plt
from jax import tree_util as pytree
import matplotlib.image as mpimg
import jax
from jax import vmap, jit
from rich import print
from jax.tree_util import Partial as partial
import jax.numpy as jnp
import jax.scipy as jsp
import scriptutils as ut
import numpy as np
from pathlib import Path
import optax
import biocomp as bc
import biocomp.utils as bu
from time import time

from jax.example_libraries import stax
from jax.example_libraries.stax import BatchNorm, Conv, Dense, Flatten, Relu, LogSoftmax

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     perceive     --
# ···············································································

# outputs a N_CHANNEL * 2 vector:
# the concatenation of the cell's current state and the conv2d of the neighbors'


@jit
def perceive(state_grid):
    sobel_x = jnp.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]])
    sobel_y = sobel_x.transpose()

    def conv(im, f):
        return jsp.signal.convolve(im, f, mode='same')

    # Convolve sobel filters with states in x, y and channel dimension.
    grad_x = vmap(partial(conv, f=sobel_x), in_axes=2, out_axes=2)(state_grid)
    grad_y = vmap(partial(conv, f=sobel_y), in_axes=2, out_axes=2)(state_grid)

    perception_grid = jnp.concatenate((state_grid, grad_x, grad_y), axis=2)
    return perception_grid


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                       --      load data     --
# ···············································································

N_CHANNEL = 8

path = Path('../data/morpho/liverlobule')

init = mpimg.imread(path / 'init.png')
init = (init > 0.5) * 1.0

target = mpimg.imread(path / 'target.png')
target = (target > 0.5) * 1.0

plt.imshow(target, origin='lower')
plt.title('target')
plt.show()
plt.imshow(init, origin='lower')
plt.title('initial state')
plt.show()

*target.shape[:2], 1

missing_channels = max(N_CHANNEL - target.shape[2], 0)
if missing_channels > 0:
    init = jnp.concatenate((init, jnp.ones((*init.shape[:2], missing_channels))), axis=2)
    target = jnp.concatenate((target, jnp.ones((*target.shape[:2], missing_channels))), axis=2)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                    --     update  and train     --
# ···············································································




# TODO: stochastic update with masking
# TODO: the weights of the final convolutional layer should be zero to default to no-op
# TODO: in the originam paper, update computes by how much we should update a cell's state, given its neighbors
# no final Relu as we need to be able to decrease state's variables... why not output directly the state???

GRID_W, GRID_H, CHANNELS = target.shape
input_shape = (CHANNELS * 3,)
assert init.shape == target.shape

init_fun, model = stax.serial(Dense(128), Relu, Dense(CHANNELS))
model = jit(model)




def alive_masking(state_grid):
    # Take the alpha channel as the measure of “life”.
    alive = state_grid[:, :, 3] > 0.1
    return vmap(partial(jnp.multiply, alive), in_axes=2, out_axes=2)(state_grid)

def grow(params, prev_state, _):
    # perceptions = perceive(alive_masking(prev_state))
    perceptions = perceive(prev_state)
    next_state = vmap(vmap(partial(model, params)))(perceptions)
    return next_state, None


def run(params, init_grid):
    final_state, _ = jax.lax.scan(
        partial(grow, params), init_grid, xs=None, length=n_simulation_steps
    )
    return final_state


key = jax.random.PRNGKey(1)
optimizer = optax.adam(learning_rate=0.01)

def loss(params, init_grid, target):
    y_pred = run(params, init_grid)
    l = optax.l2_loss(y_pred[:,:,:4], target[:,:,:4])
    return l.mean()


n_training_steps = 30
n_init = 3
n_simulation_steps = 30

initialization_keys = jax.random.split(key, n_init)

def training_step(params, opt_state, init_grid, target):
    loss_value, grads = jax.value_and_grad(loss)(params, init_grid, target)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value

@jit
def train_one(init_grid, target, key, input_shape=(CHANNELS * 3,)):
    _, initial_params = init_fun(key, input_shape)
    initial_params = pytree.tree_map(lambda x:x*0.000001, initial_params)
    initial_state = optimizer.init(initial_params)

    @bu.progress_scan(n_training_steps, bu.TQDMProgress, 'Training model')
    def scannable_step(params_and_state, iter_num):
        params, opt_state = params_and_state
        new_params, new_state, loss = training_step(params, opt_state, init_grid, target)
        return (new_params, new_state), (loss, params)

    _, losses_and_params_history = jax.lax.scan(
        scannable_step, (initial_params, initial_state), np.arange(n_training_steps)
    )
    return losses_and_params_history

# losses, stacked_params = train_one(init, target, key)

# actual training "loop"
start = time()
train_all = vmap(train_one)
all_losses, all_params = train_all(initialization_keys)
end = time()
print('Trained in', end - start)

ut.save(all_losses, './all_losses.pickle')
ut.save(all_params, './all_params.pickle')

all_losses

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
