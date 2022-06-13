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
from jax.example_libraries.stax import BatchNorm, Conv, Dense, Flatten, Relu, LogSoftmax, Sigmoid


def plotCAState(state, title='', outfile=None):
    fig, a = plt.subplots(1, 1, figsize=(5, 5))
    a.imshow(state[:, :, :4])
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
# {{{                          --     config     --
# ···············································································

import wandb as wb

wb.init(project="morpholiver", entity="jdisset")

wb.config = {
    "learning_rate": 0.001,
    "epochs": 100,
    "channels": 16,
    "grid_size": (70, 70),
    "nn_size": 128,
    "steps_per_run": 64,
    "rng_key": 42,
    "initial_param_scaling": 0.00001,
    "activation": "Sigmoid",
}

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                       --      load data     --
# ···············································································

from PIL import Image

path = Path('../data/morpho/liverlobule')


def readimg(p, threshold=None, size=None):
    im = Image.open(p)
    if size is not None:
        im = im.resize(size)
    im = jnp.array(im) / 255
    if threshold:
        im = (im > threshold) * 1.0
    return im


init = readimg(path / 'init.png', threshold=0.5, size=wb.config['grid_size'])
target = readimg(path / 'target.png', threshold=0.5, size=wb.config['grid_size'])

wb.log(
    {
        'init': wb.Image(np.array(init), caption="initial state"),
        'target': wb.Image(np.array(target), caption="target state"),
    },
    step=0,
)

missing_channels = max(wb.config['channels'] - target.shape[2], 0)
if missing_channels > 0:
    init = jnp.concatenate((init, jnp.ones((*init.shape[:2], missing_channels))), axis=2)
    target = jnp.concatenate((target, jnp.ones((*target.shape[:2], missing_channels))), axis=2)


GRID_W, GRID_H, CHANNELS = target.shape
input_shape = (CHANNELS * 3,)
assert init.shape == target.shape
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                    --     model, loss, updates     --
# ···············································································


# TODO: stochastic update with masking
# TODO: the weights of the final convolutional layer should be zero to default to no-op
# TODO: in the originam paper, update computes by how much we should update a cell's state, given its neighbors
# no final Relu as we need to be able to decrease state's variables... why not output directly the state???


act = {'Sigmoid': Sigmoid, 'Relu': Relu}[wb.config['activation']]

init_fun, model = stax.serial(Dense(wb.config['nn_size']), act, Dense(CHANNELS), act)
model = jit(model)


@jit
def perceive(state_grid):
    # outputs a N_CHANNEL * 2 vector:
    # the concatenation of the cell's current state and the conv2d of the neighbors'
    sobel_x = jnp.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]])
    sobel_y = sobel_x.transpose()

    def conv(im, f):
        return jsp.signal.convolve(im, f, mode='same')

    # Convolve sobel filters with states in x, y and channel dimension.
    grad_x = vmap(partial(conv, f=sobel_x), in_axes=2, out_axes=2)(state_grid)
    grad_y = vmap(partial(conv, f=sobel_y), in_axes=2, out_axes=2)(state_grid)

    perception_grid = jnp.concatenate((state_grid, grad_x, grad_y), axis=2)
    return perception_grid


def alive_masking(state_grid):
    # Take the alpha channel as the measure of “life”.
    alive = state_grid[:, :, 3] > 0.1
    return vmap(partial(jnp.multiply, alive), in_axes=2, out_axes=2)(state_grid)


def grow(params, prev_state, _):
    perceptions = perceive(alive_masking(prev_state))
    next_state = vmap(vmap(partial(model, params)))(perceptions)
    return next_state, None


def run(params, init_grid, n_steps):
    final_state, _ = jax.lax.scan(partial(grow, params), init_grid, xs=None, length=n_steps)
    return final_state


def loss(params, init_grid, target, n_steps):
    y_pred = run(params, init_grid, n_steps)
    l = (y_pred[:, :, :4] - target[:, :, :4]) ** 2
    return l.mean()


key = jax.random.PRNGKey(wb.config['rng_key'])
optimizer = optax.adamw(learning_rate=wb.config['learning_rate'])


def training_step(params, opt_state, init_grid, target):
    loss_value, grads = jax.value_and_grad(loss)(
        params, init_grid, target, wb.config['steps_per_run']
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, grads, loss_value


@jit
def train_one(init_grid, target, key, input_shape=input_shape):
    _, initial_params = init_fun(key, input_shape)
    initial_params = pytree.tree_map(
        lambda x: x * wb.config['initial_param_scaling'], initial_params
    )
    initial_state = optimizer.init(initial_params)

    def wandb_hook(acc, iter_num):
        loss, params, state, grads = acc
        fgrad = np.concatenate([f.flatten() for g in grads for f in g])
        wb.log({'loss': loss, 'params': params, 'state': state, 'grads':fgrad}, step=iter_num)
        print(f'{iter_num}: {loss:.4f}')

    @ut.hooked_scan(wb.config['epochs'], wandb_hook, call_rate=10)
    def scannable_step(params_and_state, iter_num):
        params, opt_state = params_and_state
        new_params, new_state, grads, loss = training_step(params, opt_state, init_grid, target)
        return (new_params, new_state), (loss, params, opt_state, grads)

    _, (losses, params, states, grads) = jax.lax.scan(
        scannable_step, (initial_params, initial_state), np.arange(wb.config['epochs'])
    )

    return (losses, params, states, grads)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                    --     update  and train     --
# ···············································································

start = time()
runloss, params, states, grads = train_one(init, target, key)
end = time()
print('Trained in', end - start)
params_history = bu.param_unstack(params, len(runloss) + 1)
best_epoch = np.argmin(np.array(runloss))
best_loss = f'{runloss[best_epoch]:.4f}'

g = bu.param_unstack(grads, len(runloss) + 1)



# i = 0
# ut.plotBestLoss(
# runloss,
# [],
# title=f'run {i}\nbest = {best_loss} at epoch {best_epoch}',
# outfile=f'lossplot_{i}_{best_loss}.png',
# )
# best_params = params_history[best_epoch]
# best_state = run(best_params, init, n_simulation_steps)
# plotCAState(best_state, outfile=f'best_state_{i}_{best_loss}.png')
# title = f'Epoch {i}, after {n_simulation_steps} steps'
# ut.save(runloss, f'./losses_{i}.pickle', overwrite=True)
# ut.save(best_params, f'./params_{i}.pickle', overwrite=True)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
