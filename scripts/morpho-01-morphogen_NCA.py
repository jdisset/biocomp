## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     imports     --
# ···············································································

import matplotlib.pyplot as plt

import jax
from jax import vmap, jit, lax
from jax import tree_util as pytree
from jax.tree_util import Partial as partial
import jax.numpy as jnp
import jax.scipy as jsp

from rich import print
import scriptutils as ut
import numpy as np
from pathlib import Path
import optax
from time import time
from scriptutils import ddict

from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu, Sigmoid


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


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     config     --
# ···············································································


cfg = ddict(
    {
        # training
        "learning_rate": 0.001,
        "adam_w_decay": 0.01,
        "clipping": 0.001,
        "initial_param_scaling": 0.0001,
        "epochs": 150000,
        "log_rate": 50,
        "rng_key": 11,
        "target_steps_per_run": (70, 100),
        "pool_size": 8,
        # CA sim
        "grid_size": (64, 64),
        "update_prob": 0.65,
        "alive_threshold": 0.2,
        # model
        # - i/o
        "fluo_channels": 3,
        "switch_channels": 1,  # channels that have a probability to switch from 0 to 1
        "morphos": [0.8, 0.8, 0, 0],  #  npixel for kernel = max(1, r*grid_size)
        # - controler
        "nn_size": 128,
        "activation": "Sigmoid",
    }
)

# Without a varied random target time, not only is it not robust, but it probably also cheats a lot...

INDEX_FLUO = 0
INDEX_ALIVE = 3
INDEX_DIVIDE = 4
INDEX_MORPHO = 5
N_MORPHO = len(cfg.morphos)
INDEX_SWITCH = INDEX_MORPHO + N_MORPHO - 1

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                       --      load data     --
# ···············································································

path = Path('../data/morpho/liverlobule')
init = ut.readimg(path / 'init.png', threshold=0.5, size=cfg.grid_size)
target = ut.readimg(path / 'target.png', threshold=0.5, size=cfg.grid_size)


extra_output_channels = 1 + N_MORPHO

if extra_output_channels > 0:
    extra = np.zeros((*init.shape[:2], extra_output_channels))
    init = np.concatenate((init, extra), axis=2)
    target = np.concatenate((target, extra), axis=2)

OUTPUT_SIZE = 4 + extra_output_channels


def sweep(op, A, B, axis=2):
    return vmap(partial(op, B), in_axes=axis, out_axes=axis)(A)


@partial(jit, static_argnums=2)
def subtract_sweep(A, B, axis=2):
    return sweep(jnp.subtract, A, B, axis)


from scipy.ndimage.interpolation import rotate

# generate rotated versions of the target to make the loss rotation invariant
t = target[:, :, :4]
targets = []
for a in np.linspace(0, 360 / 12, 10):
    rotated = rotate(t, angle=a, reshape=False)
    rotated[:, :, 3] = (rotated[:, :, 3] >= 0.5) * 1.0
    targets.append(rotated)
targets = np.array(targets)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                    --     model, loss, updates     --
# ···············································································


act = {'Sigmoid': Sigmoid, 'Relu': Relu}[cfg.activation]
init_fun, model = stax.serial(Dense(cfg.nn_size), act, Dense(OUTPUT_SIZE), act)
model = jit(model)


def maxpool(x, dims, strides=(1, 1), border_mode='same'):
    return lax.reduce_window(x, -np.inf, lax.max, dims, strides, border_mode)


def gaussian_kernel(l=5, sig=1.0):
    ax = np.linspace(-(l - 1) / 2.0, (l - 1) / 2.0, l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def morpho_kernel(l):
    if l == 0.0:
        sq2 = np.sqrt(2)
        filter = np.array([[sq2, 1, sq2], [1, 0, 1], [sq2, 1, sq2]])
        return filter / filter.sum()
    W = max(cfg.grid_size)
    return gaussian_kernel(int(np.round(l * W)), l * W / 5.0)


morpho_kernels = [morpho_kernel(l) for l in cfg.morphos]

# a = jax.random.uniform(key, (5, 5, 2))
# b = jax.random.uniform(key, (5, 5, 6))
# c = jnp.concatenate((a, b), axis=2)


def perceive(state_grid, kernels):

    return jnp.concatenate(
        (
            jnp.stack(
                [
                    jsp.signal.convolve(state_grid[:, :, INDEX_MORPHO + i], kernels[i], mode='same')
                    for i in range(len(kernels))
                ],
                axis=2,
            ),
            state_grid[:, :, INDEX_SWITCH:],
        ),
        axis=2,
    )


@partial(jit, static_argnums=2)
def multiply_axis(A, B, axis=2):
    return vmap(partial(jnp.multiply, B), in_axes=axis, out_axes=axis)(A)


def cell_division(state, key):
    div_prob = jnp.maximum(state[:, :, INDEX_ALIVE], maxpool(state[:, :, INDEX_DIVIDE], (3, 3)))
    div_draw = jax.random.uniform(key, state.shape[:2])
    divide = div_draw <= div_prob  # mask with the old and new alive cells
    return state.at[:, :, INDEX_ALIVE].set(divide.astype(float))


@jit
def grow(params, state_grid, key):
    alive = multiply_axis(state_grid, state_grid[:, :, INDEX_ALIVE])
    prev_state = cell_division(alive, key)
    alive_mask = prev_state[:, :, INDEX_ALIVE]

    # random switch channel
    prev_switch_channels = jnp.asarray(prev_state[:, :, INDEX_SWITCH:], bool)
    sw = vmap(jnp.less, in_axes=(2, 0), out_axes=2)(
        jax.random.uniform(key, shape=prev_switch_channels.shape), params['switch_prob']
    )

    new_switch_channels = jnp.asarray(
        (prev_switch_channels & alive_mask[:, :, None].astype(bool)) | sw, float
    )
    prev_state = prev_state.at[:, :, INDEX_SWITCH:].set(new_switch_channels)

    perceptions = perceive(prev_state, morpho_kernels)
    next_state = vmap(vmap(partial(model, params['model'])))(perceptions)

    # random masking for simulated asynchronicity
    mask = jax.random.uniform(key, prev_state.shape[:2]) > cfg.update_prob
    next_state = next_state * mask[:, :, None] + prev_state * jnp.invert(mask)[:, :, None]

    return next_state.at[:, :, INDEX_ALIVE].set(alive_mask), (state_grid, perceptions)


def run(params, n_steps, key):
    keys = jax.random.split(key, n_steps)
    final_state, _ = jax.lax.scan(partial(grow, params), init, keys)
    return final_state


@partial(jit, static_argnums=(1,))
def run_acc(params, n_steps, key):
    final_state, state_history = jax.lax.scan(
        partial(grow, params), init, jax.random.split(key, n_steps)
    )
    return final_state, state_history


def loss(params, n_steps, key):
    y_pred = run(params, n_steps, key)
    l = subtract_sweep(targets, y_pred[:, :, :4], axis=0) ** 2
    return l.mean(axis=(1, 2, 3)).min()


# pool loss creates n_indiv individuals and average the loss over them
# def pool_loss(params, n_steps_arr, key):
    # keys = jax.random.split(key, len(n_steps_arr))
    # losses = vmap(partial(loss, params, n_steps_arr[0]))(keys)
    # return losses.mean()


key = jax.random.PRNGKey(cfg.rng_key)
optimizer = optax.chain(
    optax.adaptive_grad_clip(cfg.clipping),
    optax.adamw(learning_rate=cfg.learning_rate, weight_decay=cfg.adam_w_decay),
)


@partial(jit, static_argnums=(2,))
def training_step(params, opt_state, n_steps, key):
    loss_value, grads = jax.value_and_grad(loss)( params, n_steps, key)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, grads, loss_value


from rich.progress import track

import jax.profiler

evalrate = 10
evalnum = 0


def wandb_update(loss, params, grads, iter_num):
    global evalnum

    fgrad = np.concatenate([f.flatten() for g in grads['model'] for f in g])
    fpar = np.concatenate([a.flatten() for a in pytree.tree_leaves(params['model'])]).flatten()
    if evalnum % evalrate == 0:
        res = run(params, cfg.target_steps_per_run[1]-5, key)
        wb.log({'eval': wb.Image(np.array(res[:, :, :4]), caption="current state")}, step=iter_num)
        fig, axs = plt.subplots(1, OUTPUT_SIZE + 1, figsize=(OUTPUT_SIZE * 4, 4))
        axs[0].imshow(np.array(res[:, :, :4]))
        for i in range(OUTPUT_SIZE):
            axs[i + 1].imshow(np.array(res[:, :, i]))
        wb.log({'channels': fig}, step=iter_num)
        plt.close()
    wb.log(
        {'loss': loss, 'gradients': wb.Histogram(fgrad), 'parameters': wb.Histogram(fpar), 'switch_prob': params['switch_prob'][0]},
        step=iter_num,
    )
    evalnum += 1


def train(key, input_shape=(N_MORPHO + cfg.switch_channels,)):
    _, initial_model_params = init_fun(key, input_shape)
    switch_rates = jax.random.uniform(key, (cfg.switch_channels,))
    model_params = pytree.tree_map(lambda x: x * cfg.initial_param_scaling, initial_model_params)
    params = {'model': model_params, 'switch_prob': switch_rates}
    opt_state = optimizer.init(params)

    params_history = []
    loss_history = []
    n_steps_arr = np.array(jax.random.randint(key, (cfg.epochs,), * cfg.target_steps_per_run))
    for i,n_steps in track(enumerate(n_steps_arr), description='Training...', total=cfg.epochs):
        params, opt_state, grads, loss = training_step(params, opt_state, n_steps, key)
        loss_history.append(loss)
        params_history.append(params)
        if i == cfg.epochs or i % cfg.log_rate == 0 or i == 0:
            wandb_update(loss, params, grads, i)

    return params_history, loss_history


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                    --     train     --
# ···············································································
import wandb as wb

wb.init(config=cfg, project="morpholiver_01", entity="jdisset", reinit=True)

wb.log(
    {
        'init': wb.Image(np.array(init[:, :, :4]), caption="initial state"),
        'target': wb.Image(np.array(target[:, :, :4]), caption="target state"),
    },
    step=0,
)


start = time()
params_history, runloss = train(key)
end = time()

print('------------')
print('Trained in', end - start)

# params_history = bu.param_unstack(params, len(runloss) + 1)
best_epoch = np.argmin(np.array(runloss))
best_loss = f'{runloss[best_epoch]:.4f}'
best_params = params_history[best_epoch]
ut.save(runloss, f'./losses.pickle', overwrite=True)
ut.save(best_params, f'./params.pickle', overwrite=True)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
