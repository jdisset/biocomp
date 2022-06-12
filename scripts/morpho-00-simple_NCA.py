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

def plotCAState(state, title='', outfile=None):
    fig, a = plt.subplots(1, 1, figsize=(5, 5))
    a.imshow(state[:,:,:4])
    fig.suptitle(title)
    if outfile is not None:
        fig.savefig(outfile, dpi=100)
        plt.close()
    else:
        plt.show()
    return fig
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

N_CHANNEL = 16

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


def run(params, init_grid, n_steps):
    final_state, _ = jax.lax.scan(
        partial(grow, params), init_grid, xs=None, length=n_steps
    )
    return final_state


key = jax.random.PRNGKey(1)
optimizer = optax.adamw(learning_rate=0.0001)

def loss(params, init_grid, target, n_steps):
    y_pred = run(params, init_grid, n_steps)
    l = (y_pred[:,:,:4] - target[:,:,:4])**2
    return l.mean()


n_init = 5
n_training_steps = 30000
n_simulation_steps = 64

initialization_keys = jax.random.split(key, n_init)

def training_step(params, opt_state, init_grid, target):
    loss_value, grads = jax.value_and_grad(loss)(params, init_grid, target, n_simulation_steps)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value

@jit
def train_one(init_grid, target, key, input_shape=input_shape):
    _, initial_params = init_fun(key, input_shape)
    initial_params = pytree.tree_map(lambda x:x*0.00001, initial_params)
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


train_all_vmaped = jit(vmap(partial(train_one, init, target)))


for i,k in enumerate(initialization_keys):
    start = time()
    runloss, params = train_one(init, target, k)
    end = time()
    print('Trained in', end - start)
    params_history = bu.param_unstack(params, len(runloss) + 1)
    best_epoch = np.argmin(np.array(runloss))
    best_loss = f'{runloss[best_epoch]:.4f}'
    ut.plotBestLoss(runloss, [], title=f'run {i}\nbest = {best_loss} at epoch {best_epoch}', outfile=f'lossplot_{i}_{best_loss}.png')
    best_params = params_history[best_epoch]
    best_state = run(best_params, init, n_simulation_steps)
    plotCAState(best_state, outfile=f'best_state_{i}_{best_loss}.png')
    title = f'Epoch {i}, after {n_simulation_steps} steps'
    ut.save(runloss, f'./losses_{i}.pickle', overwrite=True)
    ut.save(best_params, f'./params_{i}.pickle', overwrite=True)

# best_run = np.argmin(np.array(losses)[:, -1])
# best_loss = losses[best_run]
# params_history = bu.param_unstack(all_params[best_run], len(best_loss) + 1)
# losses = ut.load('./all_losses.pickle')
# best_run = np.argmin(np.array(losses)[:, -1])
# best_loss = losses[best_run]
# params = ut.load('./best_params.pickle')
# best_state = run(params, init, n_simulation_steps)
# plotCAState(best_state, outfile='best_state.png')
# ut.plotBestLoss(best_loss, losses, outfile='lossplot.png', vmax=0.1)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
