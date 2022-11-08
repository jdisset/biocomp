## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     imports     --
# ···············································································
import biocomp as bc
import biocomp.compute as bcc
import numpy as np
from functools import partial
import time
import biocomp.utils as bu
import scriptutils as ut
import jax
from jax import jit, vmap, grad, value_and_grad
import jax.numpy as jnp
import random
import biocomp.datautils as du
import optax

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                           --     init     --
# ···············································································

random.seed()
lib = ut.load_lib()
xp = ut.load_xp('20221012A_massCtrls', lib)
models = xp.get_models()
X, Y = xp.get_XY(models)

cfg = {
    "node_remap": {},
    "optimizer": "sgd",
    "learning_rate": 0.001,
    "adam_w_decay": 0.0001,
    "loss_function": "mse",
    "rng_key": 42,
    "epochs": 10000,
    "n_replicates": 1,
    "compile_training": True,
    "n_batches": 32,
    "norm_factor": 1e6,
    "balance_bin_resolution": 0.5,
    "balance_threshold_quantile": 0.4,
    "balance_threshold_min": 40,
    "log_rate": 1,
    "plot_rate": 100,
    "node_remap": {
        "sequestron_ERN": "ERN_with_affinity",
        "transcription": "transcription_nn",
        "inv_transcription": "inverse_transcription_nn",
        "translation": "translation_nn",
        "inv_translation": "inverse_translation_nn",
    },
    "save_rate": 100,
}
optimizer = optax.sgd(learning_rate=cfg['learning_rate'])

key = jax.random.PRNGKey(cfg['rng_key'])
ikeys = jax.random.split(key, len(models))

params = {}
constraints = {}

for s, m, k in zip(models.keys(), models.values(), ikeys):
    params, constraints = m.init(k, pre_params=params, pre_constraints=constraints)

opt_state = optimizer.init(params)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                       --     training step     --
# ···············································································

# def loss_f(y, y_hat):
# return jnp.mean((y - y_hat) ** 2)

# nmodels = len(models)
# @jit
# def sum_pytrees(trees):
# tr = [t['shared'] for t in trees]
# res_sh = jax.tree_map(lambda *x: jnp.sum(jnp.stack(x), axis=0), *tr)
# # we  use the shared sum, and then all the rest from the first model
# res = {'shared': res_sh}
# # update with everything that is not "shared" from the first model
# res.update({k: v for k, v in trees[0].items() if k != 'shared'})
# return res
# print(f'Compiling sum pytrees...')
# t0 = time.time()
# lowered = sum_pytrees.lower([params]*nmodels)
# print(f'Lowered...')
# compiled_sum = lowered.compile()
# t1 = time.time()
# print(f'compilation time: {t1-t0:.2f}s')

# def loss_func_1(params, x, y, rng_key):

# def apply_model(params, x, key, name):
# m = partial(models[name], params, rng_key=key)
# return vmap(m)(x[name]).squeeze()

# nmodels = len(models)
# ikeys = jax.random.split(rng_key, nmodels)
# names = list(x.keys())
# res = loss_f(apply_model(params, x, ikeys[0], names[0]), y[names[0]]).mean()
# # res = jnp.array(
# # [
# # loss_f(apply_model(params, x, k, sample), y[sample])
# # for sample, k in zip(models.keys(), ikeys)
# # ]
# # ).mean()
# return res

# def get_all_f(x, y, ikeys):
# allx = list(x.values())
# ally = list(y.values())
# allmodels = [partial(lambda i, p: loss_f(vmap(partial(v, p, rng_key=ikeys[i]))(allx[i]), ally[i]).mean(), i) for i, (k, v) in enumerate(models.items())]
# return allmodels

# def loss_func_2(params, x, y, rng_key):
# nmodels = len(models)
# ikeys = jax.random.split(rng_key, nmodels)

# allmodels = get_all_f(x, y, ikeys)

# # res = jnp.array([m(params) for m in allmodels]).mean()
# res = [value_and_grad(m)(params) for m in allmodels]

# res, grads = zip(*res)

# # now sum all the gradients
# # grads = jax.tree_map(lambda *x: jnp.sum(jnp.stack(x), axis=0), *grads)
# grads = compiled_sum(grads)

# return jnp.array(res).mean(), grads


# # res = vmap(lambda i: jax.lax.switch(i, allmodels))(jnp.arange(nmodels)).mean()
# # res, grads = zip(*res)
# # return res


# def training_step(params, opt_state, key, x, y):
# # loss, grads = jax.value_and_grad(loss_func_2)(params, x, y, key)
# loss, grads = loss_func_2(params, x, y, key)
# # grads = jax.tree_map(lambda *x: jnp.sum(jnp.stack(x), axis=0), *grads)
# updates, opt_state = optimizer.update(grads, opt_state, params)
# updates['node'] = jax.tree_map(lambda x: jnp.zeros_like(x), updates['node'])
# params = optax.apply_updates(params, updates)
# params = bu.apply_constraints(params, constraints)
# return params, opt_state, grads, loss


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     compile     --
# ···············································································

# step = jit(training_step)
# print(f'Compiling training step...')
# t0 = time.time()
# lowered = step.lower(params, opt_state, k, X, Y)
# print(f'Lowered...')
# compiled = lowered.compile()
# t1 = time.time()
# print(f'compilation time: {t1-t0:.2f}s')
# step = compiled

# loss_func= jit(loss_func)
# print(f'Compiling training step...')
# t0 = time.time()
# lowered = loss_func.lower(params, X, Y, k)
# print(f'Lowered...')
# compiled = lowered.compile()
# t1 = time.time()
# print(f'compilation time: {t1-t0:.2f}s')
# loss_func = compiled


# %timeit step(params, opt_state, k, X, Y)
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

# nmodels = len(models)
# @jit
# def sum_pytrees(trees):
# tr = [t['shared'] for t in trees]
# res_sh = jax.tree_map(lambda *x: jnp.sum(jnp.stack(x), axis=0), *tr)
# # we  use the shared sum, and then all the rest from the first model
# res = {'shared': res_sh}
# # update with everything that is not "shared" from the first model
# res.update({k: v for k, v in trees[0].items() if k != 'shared'})
# return res
# print(f'Compiling sum pytrees...')
# t0 = time.time()
# lowered = sum_pytrees.lower([params]*nmodels)
# print(f'Lowered...')
# compiled_sum = lowered.compile()
# t1 = time.time()
# print(f'compilation time: {t1-t0:.2f}s')
# %timeit compiled_sum([params]*nmodels)


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                     --     synthetic example     --
# ···············································································
# mshape = (20, 20)
# nf = 30
# ndata = 1000

# params = {i: np.random.uniform(0,1,mshape) for i in range(50)}

# def gen_f():
# keys = np.random.choice(list(params.keys()), 30)
# f = lambda p, x: sum([p[k] * x for k in keys])
# return f

# F = [gen_f() for _ in range(nf)]
# XX = np.random.uniform(0, 1, (nf, ndata, *mshape))

# def loss_f(params, X):
# return sum([vmap(partial(f, params))(x) for f, x in zip(F, X)]).mean()

# opt_state = optimizer.init(params)

# def step(params, X, opt_state):
# loss, grad = value_and_grad(loss_f)(params, X)
# updates, opt_state = optimizer.update(grad, opt_state, params)
# params = optax.apply_updates(params, updates)
# return params, opt_state, grad, loss


# step = jit(step)
# print(f'Compiling training step...')
# t0 = time.time()
# lowered = step.lower(params, XX, opt_state)
# print(f'Lowered...')
# compiled = lowered.compile()
# t1 = time.time()
# print(f'compilation time: {t1-t0:.2f}s')
# step = compiled

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

# Note: splitting params into shared and node params reduces the compilation time
# by ~ 20% which is not bas but not enough... (we need something like 10x faster)

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                     --     synthetic example     --
# ···············································································


params = {}
constraints = {}
for s, m, k in zip(models.keys(), models.values(), ikeys):
    params, constraints = m.init(k, pre_params=params, pre_constraints=constraints)

models = xp.get_models()
N = 17
F = [partial(m, rng_key=jax.random.PRNGKey(0)) for s, m in models.items()][:N]
# F = [partial(m.recursive_eval, rng_key=jax.random.PRNGKey(0)) for s, m in models.items()][:N]

XX = list(X.values())[:N]
# tX is X where eah entry is truncated to the length of the shortest sequence
shortest = min([len(x) for x in X.values()])
shortest = 32

tX = [x[:shortest] for x in X.values()][:N]
XX = jnp.array(jnp.concatenate(tX, axis=1))
# n_inputs = [m.n_inputs for m in models.values()]
n_inputs = [m.n_inputs for m in models.values()][:N]
input_indices = [np.arange(n)+i for i, n in zip(np.cumsum([0]+n_inputs), n_inputs)]
XX = XX[:, input_indices]
XX = np.transpose(XX, (1, 0, 2))
index = jnp.arange(N)


def loss_f(params, X):
    return jnp.array([vmap(partial(f, params))(x).mean() for f, x in zip(F, X)]).mean()

def loss_f_agg_col(params, X):
    def inner(f, inputs_col):
        return vmap(partial(f, params))(inputs_col).mean()
    return jnp.array([inner(f, X[:,i]) for i, f in enumerate(F)]).mean()

def inner_c(f, par, inputs_col):
    return vmap(partial(f,par))(inputs_col).mean()

funcs_c = [partial(inner_c, f) for f in F]
allx = jnp.transpose(XX, (1, 0, 2))

def loss_f_agg_col_sw(params, X):
    vmap_functions = vmap(lambda i, p, x: jax.lax.switch(i, funcs_c, p, x), in_axes=(0, None, 0))
    return vmap_functions(index, params, X).mean()

def loss_f_agg_row(params, X):
    def inner(params, inputs_row):
        return jnp.array([f(params, r) for f, r in zip(F, inputs_row)]).mean()
    return vmap(partial(inner, params))(X).mean()

def loss_f_agg_row_sw(params, X):
    funcs = [partial(f, params) for f in F]
    vmap_functions = vmap(lambda i, x: jax.lax.switch(i, funcs, x))
    def inner(inputs_row):
        return vmap_functions(index, inputs_row).mean()
    return vmap(inner)(X).mean()


# loss_f_agg_col_sw(params, XX)

# loss_f_agg_row(params, XX)

# TODO: replace overwrite_with

step = jit(value_and_grad(loss_f_agg_col))

print(f'Compiling training step...')
t0 = time.time()
lowered = step.lower(params, allx)
print(f'Lowered...')
step_c = lowered.compile()
t1 = time.time()
print(f'compilation time: {t1-t0:.2f}s')

jaxpr = ut.get_jaxpr(step, params, allx)
nlines = len(jaxpr.__str__().splitlines())
print(f'jaxpr size: {nlines} lines')

%timeit step_c(params, allx)

print('res = ', step_c(params, allx)[0])


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     example 3     --
# ···············································································

# params = {}
# constraints = {}
# for s, m, k in zip(models.keys(), models.values(), ikeys):
# params, constraints = m.init(
# k, pre_params=params, pre_constraints=constraints)

# models = xp.get_models()

N = 17
F = [partial(m, rng_key=jax.random.PRNGKey(0)) for s, m in models.items()][:N]

XX = list(X.values())[:N]
shortest = min([len(x) for x in X.values()])
shortest = 3

tX = [x[:shortest] for x in X.values()]
tX
X_truncated = {s:x for s, x in zip(X.keys(), tX)}

XX = jnp.array(jnp.concatenate(tX, axis=1))
tX[0].shape

rng_key = jax.random.PRNGKey(0)
jax.random.choice(jax.random.PRNGKey(0), tX[0], (100,))

n_inputs = [m.n_inputs for m in models.values()]
input_indices = [np.arange(n) + i for i, n in zip(np.cumsum([0] + n_inputs), n_inputs)]
XX = XX[:, input_indices]
XX = np.transpose(XX, (1, 0, 2))

XX.shape

total_size = 30

ylist = [jax.random.choice(rng_key, x, (total_size,)) for x in Y.values()]
n_outputs = max([y.shape[1] for y in ylist])
n_outputs = 4

# add 0 padding to the end of the arrays to make them all the same size
ylist_p = jnp.array([np.pad(y, ((0, 0), (0, n_outputs - y.shape[1]))) for y in ylist])

batch_size = 6
n_batches = total_size // batch_size

ylist_p.shape

ylist_batches = jnp.array(jnp.split(ylist_p[:, :n_batches * batch_size], n_batches, axis=1))
ylist_batches.shape

ylist_batches.shape

def loss_f(params, X):
    return jnp.array([vmap(partial(f, params))(x).mean() for f, x in zip(F, X)]).mean()


def loss_func(params, x):
    def apply_model(params, x, name):
        m = partial(models[name], params, rng_key=jax.random.PRNGKey(0))
        return vmap(m)(x[name])
    res = jnp.array([apply_model(params, x, sample).mean() for sample in models.keys()][:N]).mean()
    return res


# loss_func(params, X_truncated)
# loss_f(params, XX)

XX = X_truncated
step = jit(value_and_grad(loss_func))

print(f'Compiling training step...')
t0 = time.time()
lowered = step.lower(params, XX)
print(f'Lowered...')
step_c = lowered.compile()
t1 = time.time()
print(f'compilation time: {t1-t0:.2f}s')

jaxpr = ut.get_jaxpr(step, params, XX)
nlines = len(jaxpr.__str__().splitlines())
print(f'jaxpr size: {nlines} lines')

%timeit step_c(params, XX)

print('res = ', step_c(params, XX))


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

import rich

from rich import status
s = rich.status.Status("Hello, [bold magenta]World[/bold magenta]!", spinner="dots")
s.start()
s.update("Loading...")
s.update("Loading [bold green]done[/bold green]!")
s.stop()


a = np.arange(120).reshape(5, 12, 2)
a.shape # (5, 12, 2)

nb = 3
b = np.array(np.split(a, nb, axis=1))
b.shape

# the equivalent of the above using reshape is
c = np.reshape(a, (nb, a.shape[0], a.shape[1] // nb, a.shape[2]))
c
b
c.shape
b.shape


np.all(c == b)
