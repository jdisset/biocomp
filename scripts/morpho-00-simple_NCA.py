## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     imports     --
# ···············································································

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import jax
from jax import vmap, jit, lax
from jax import tree_util as pytree
from jax.tree_util import Partial as partial
from rich import print
import jax.numpy as jnp
import jax.scipy as jsp
import scriptutils as ut
import numpy as np
from pathlib import Path
import optax
import biocomp as bc
import biocomp.utils as bu
from time import time
from scriptutils import ddict

from jax.example_libraries import stax
from jax.example_libraries.stax import BatchNorm, Conv, Dense, Flatten, Relu, LogSoftmax, Sigmoid


plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'


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


cpu = jax.devices('cpu')[0]
gpu = jax.devices('gpu')[0]

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     config     --
# ···············································································

import wandb as wb

cfg = ddict(
    {
        # training
        "learning_rate": 0.001,
        "adam_w_decay": 0.01,
        "clipping": 0.001,
        "initial_param_scaling": 0.00001,
        "epochs": 200000,
        "log_rate": 100,
        "rng_key": 1,
        # CA sim
        "grid_size": (32, 32),
        "update_prob": 0.5,
        "alive_threshold": 0.2,
        "steps_per_run": 64,
        # model
        "nn_size": 32,
        "channels": 8,
        "neighbors_sobel": False,
        "neighbors_states": True,
        "activation": "Sigmoid",
    }
)

wb.init(config=cfg, project="morpholiver", entity="jdisset", reinit=True)

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


init = readimg(path / 'init.png', threshold=0.5, size=cfg.grid_size)
target = readimg(path / 'target.png', threshold=0.5, size=cfg.grid_size)

wb.log(
    {
        'init': wb.Image(np.array(init), caption="initial state"),
        'target': wb.Image(np.array(target), caption="target state"),
    },
    step=0,
)

missing_channels = max(cfg.channels - target.shape[2], 0)
if missing_channels > 0:
    init = jnp.concatenate((init, jnp.zeros((*init.shape[:2], missing_channels))), axis=2)
    target = jnp.concatenate((target, jnp.zeros((*target.shape[:2], missing_channels))), axis=2)


GRID_W, GRID_H, CHANNELS = target.shape

n_inputs = CHANNELS
if cfg.neighbors_sobel:
    n_inputs += CHANNELS * 2
if cfg.neighbors_states:
    n_inputs += CHANNELS

input_shape = (n_inputs,)
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

act = {'Sigmoid': Sigmoid, 'Relu': Relu}[cfg.activation]
init_fun, model = stax.serial(Dense(cfg.nn_size), act, Dense(CHANNELS*2), act, Dense(CHANNELS), act)
model = jit(model)


def perceive(state_grid):
    def conv(im, f):
        return jsp.signal.convolve(im, f, mode='same')

    def vconvol(im, f):
        return vmap(partial(conv, f=f), in_axes=2, out_axes=2)(im)

    perception = [state_grid]

    if cfg.neighbors_states:
        sq2 = jnp.sqrt(2)
        filter = jnp.array([[sq2, 1, sq2], [1, 0, 1], [sq2, 1, sq2]])
        states = vconvol(state_grid, filter)
        perception.append(states)

    if cfg.neighbors_sobel:
        sobel_x = jnp.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]])
        sobel_y = sobel_x.transpose()
        perception += [vconvol(state_grid, sobel_x), vconvol(state_grid, sobel_y)]

    perception_grid = jnp.concatenate(perception, axis=2)
    return perception_grid

##


def maxpool(x, dims, strides=(1, 1), border_mode='same'):
    return lax.reduce_window(x, -np.inf, lax.max, dims, strides, border_mode)


def alive_masking(state_grid):
    # Take the alpha channel as the measure of “life”.
    alive = maxpool(state_grid[:, :, 3], (3, 3)) > cfg.alive_threshold
    return vmap(partial(jnp.multiply, alive), in_axes=2, out_axes=2)(state_grid)


key = jax.random.PRNGKey(2)
test = jax.random.uniform(key, (500, 500, 60)).astype(float)


##

def grow(params, prev_state, key):
    alive = alive_masking(prev_state)
    perceptions = perceive(alive)
    next_state = vmap(vmap(partial(model, params)))(perceptions)

    # random masking for simulated asynchronicity
    mask = jax.random.uniform(key, prev_state.shape[:2]) > cfg.update_prob
    mask = jnp.repeat(jnp.expand_dims(mask, axis=2), prev_state.shape[2], axis=2)
    inv_mask = jnp.invert(mask)
    next_state = next_state * mask.astype(float) + alive * inv_mask.astype(float)

    return next_state, None


def run(params, init_grid, n_steps, key):
    final_state, _ = jax.lax.scan(partial(grow, params), init_grid, jax.random.split(key, n_steps))
    return final_state


def loss(params, init_grid, target, n_steps, key):
    y_pred = run(params, init_grid, n_steps, key)
    l = (y_pred[:, :, :4] - target[:, :, :4]) ** 2
    return l.mean()


key = jax.random.PRNGKey(cfg.rng_key)
optimizer = optax.chain(
    optax.adaptive_grad_clip(cfg.clipping),
    optax.adamw(learning_rate=cfg.learning_rate, weight_decay=cfg.adam_w_decay),
)


@jit
def training_step(params, opt_state, init_grid, target, key):
    loss_value, grads = jax.value_and_grad(loss)(params, init_grid, target, cfg.steps_per_run, key)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, grads, loss_value

from rich.progress import track

import jax.profiler

evalrate = 10


evalnum = 0


def wandb_update(loss, params, grads, iter_num):
    global evalnum
    fgrad = np.concatenate([f.flatten() for g in grads for f in g])
    fpar = np.concatenate([a.flatten() for a in pytree.tree_leaves(params)]).flatten()
    if evalnum % evalrate == 0:
        res = run(params, init, cfg.steps_per_run, key)
        wb.log({'eval': wb.Image(np.array(res[:, :, :4]), caption="current state")}, step=iter_num)
        wb.log({'ch4': wb.Image(np.array(res[:, :, 4]))}, step=iter_num)
        wb.log({'ch5': wb.Image(np.array(res[:, :, 5]))}, step=iter_num)
        wb.log({'ch6': wb.Image(np.array(res[:, :, 6]))}, step=iter_num)
        wb.log({'ch7': wb.Image(np.array(res[:, :, 7]))}, step=iter_num)
    wb.log(
        {'loss': loss, 'gradients': wb.Histogram(fgrad), 'parameters': wb.Histogram(fpar)},
        step=iter_num,
    )
    evalnum += 1


def train(init_grid, target, key, input_shape=input_shape):
    _, initial_params = init_fun(key, input_shape)
    params = pytree.tree_map(lambda x: x * cfg.initial_param_scaling, initial_params)
    opt_state = optimizer.init(initial_params)

    params_history = []
    loss_history = []
    for i in track(range(cfg.epochs), description='Training...'):
        params, opt_state, grads, loss = training_step(params, opt_state, init_grid, target, key)
        loss_history.append(loss)
        params_history.append(params)
        if i == cfg.epochs or i % cfg.log_rate == 0 or i == 0:
            wandb_update(loss, params, grads, i)

    return params_history, loss_history


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                    --     update  and train     --
# ···············································································

start = time()
params_history, runloss = train(init, target, key)
end = time()

print()
print('Trained in', end - start)

# params_history = bu.param_unstack(params, len(runloss) + 1)
best_epoch = np.argmin(np.array(runloss))
best_loss = f'{runloss[best_epoch]:.4f}'
best_params = params_history[best_epoch]
ut.save(runloss, f'./losses.pickle', overwrite=True)
ut.save(best_params, f'./params.pickle', overwrite=True)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     archives     --
# ···············································································
# @jit
# def train_one_scan(init_grid, target, key, input_shape=input_shape):
# _, initial_params = init_fun(key, input_shape)
# initial_params = pytree.tree_map(lambda x: x * cfg.initial_param_scaling, initial_params)
# initial_state = optimizer.init(initial_params)

# progress = Progress()
# task = progress.add_task("Training...", total=float(cfg.epochs))

# def wandb_hook(acc, iter_num):
# progress.update(task, advance=iter_num / cfg.epochs)
# # loss, params, grads = acc
# # fgrad = np.concatenate([f.flatten() for g in grads for f in g])
# # fpar = np.concatenate([a.flatten() for a in pytree.tree_leaves(params)]).flatten()
# # res = jit(run, device=cpu)(params, init, cfg.steps_per_run)
# # wb.log({'eval': wb.Image(np.array(res[:, :, :4]), caption="current state")}, step=iter_num)
# # wb.log(
# # {'loss': loss, 'gradients': wb.Histogram(fgrad), 'parameters': wb.Histogram(fpar)},
# # step=iter_num,
# # )
# # print(f'{iter_num}: {loss:.4f}')

# @ut.hooked_scan(cfg.epochs, wandb_hook, call_rate=100)
# def scannable_step(params_and_state, iter_num):
# params, opt_state = params_and_state
# new_params, new_state, grads, loss = training_step(params, opt_state, init_grid, target)
# return (new_params, new_state), (loss, params, grads)

# _, (losses, params, grads) = jax.lax.scan(
# scannable_step, (initial_params, initial_state), np.arange(cfg.epochs)
# )

# return (losses, params, grads)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
