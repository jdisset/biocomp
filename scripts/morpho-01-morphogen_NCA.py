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
        "initial_param_scaling": 0.0001,
        "epochs": 500000,
        "log_rate": 50,
        "rng_key": 11,
        # CA sim
        "grid_size": (64, 64),
        "update_prob": 0.65,
        "alive_threshold": 0.2,
        "steps_per_run": 64,
        # model
        # - i/o
        "fluo_channels": 3,
        "morphos": [0.75, 0.5, 0, 0],  #  npixel for kernel = max(1, r*grid_size)
        # - controler
        "nn_size": 128,
        "activation": "Sigmoid",
    }
)

INDEX_FLUO = 0
INDEX_ALIVE = 3
INDEX_DIVIDE = 4
INDEX_MORPHO = 5

N_MORPHO = len(cfg.morphos)

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
t = target[:,:,:4]
targets = []
for a in np.linspace(0,360/12,10):
    rotated = rotate(t, angle=a, reshape=False)
    rotated[:,:,3] = (rotated[:,:,3] >= 0.5) * 1.0
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
        return filter
    W = max(cfg.grid_size)
    return gaussian_kernel(int(np.round(l * W)), l * W/2.0)


morpho_kernels = [morpho_kernel(l) for l in cfg.morphos]


def perceive(state_grid, kernels):
    return jnp.stack(
        [
            jsp.signal.convolve(state_grid[:, :, INDEX_MORPHO+i], kernels[i], mode='same')
            for i in range(len(kernels))
        ],
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

    perceptions = perceive(prev_state, morpho_kernels)
    next_state = vmap(vmap(partial(model, params)))(perceptions)

    # random masking for simulated asynchronicity
    mask = jax.random.uniform(key, prev_state.shape[:2]) > cfg.update_prob
    inv_mask = jnp.invert(mask)
    # mask = jnp.repeat(jnp.expand_dims(mask, axis=2), prev_state.shape[2], axis=2)

    next_state = multiply_axis(next_state, mask.astype(float)) + multiply_axis(
        prev_state, inv_mask.astype(float)
    )

    return next_state.at[:,:,INDEX_ALIVE].set(alive_mask), None


def run(params, n_steps, key):
    final_state, _ = jax.lax.scan(partial(grow, params), init, jax.random.split(key, n_steps))
    return final_state


def loss(params, n_steps, key):
    y_pred = run(params, n_steps, key)
    l = (subtract_sweep(targets, y_pred[:,:,:4], axis=0) ** 2)
    return l.mean(axis=(1,2,3)).min()

key = jax.random.PRNGKey(cfg.rng_key)
optimizer = optax.chain(
    optax.adaptive_grad_clip(cfg.clipping),
    optax.adamw(learning_rate=cfg.learning_rate, weight_decay=cfg.adam_w_decay),
)


@jit
def training_step(params, opt_state, key):
    loss_value, grads = jax.value_and_grad(loss)(params, cfg.steps_per_run, key)
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
        res = run(params, cfg.steps_per_run, key)
        wb.log({'eval': wb.Image(np.array(res[:, :, :4]), caption="current state")}, step=iter_num)
        fig, axs = plt.subplots(1,OUTPUT_SIZE+1, figsize=(OUTPUT_SIZE*4,4))
        axs[0].imshow(np.array(res[:, :, :4]))
        for i in range(OUTPUT_SIZE):
            axs[i+1].imshow(np.array(res[:, :, i]))
        wb.log({'channels': fig}, step=iter_num)
        plt.close()
    wb.log(
        {'loss': loss, 'gradients': wb.Histogram(fgrad), 'parameters': wb.Histogram(fpar)},
        step=iter_num,
    )
    evalnum += 1


def train(key, input_shape=(N_MORPHO,)):
    _, initial_params = init_fun(key, input_shape)
    params = pytree.tree_map(lambda x: x * cfg.initial_param_scaling, initial_params)
    opt_state = optimizer.init(initial_params)

    params_history = []
    loss_history = []
    for i in track(range(cfg.epochs), description='Training...'):
        params, opt_state, grads, loss = training_step(params, opt_state, key)
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
