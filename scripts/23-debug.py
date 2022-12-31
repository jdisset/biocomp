## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     imports     --
# ···············································································
import biocomp as bc
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import biocomp.utils as bu
import scriptutils as ut
import jax
from jax import jit, vmap, grad, value_and_grad
import jax.numpy as jnp
import biocomp.datautils as du
import optax
from tqdm import tqdm
import biocomp.nodes as bn
import biocomp.compute as bcc


import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10.0, 10.0]
plt.rcParams['figure.dpi'] = 200

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     config     --
# ···············································································

T_SIZE = 32
T_DEPTH = 3
I_SIZE = 32
I_DEPTH = 2
I_OUT = 4
ERN_SIZE = 64
ERN_DEPTH = 3
MEFL_SIZE = 64
MEFL_DEPTH = 3
node_impl = dict(
    bc.nodes.DEFAULT_COMPUTE_NODES_DICT,
    **{
        'output': partial(bc.nn.output, wsize=MEFL_SIZE, depth=MEFL_DEPTH),
        'transcription': partial(
            bc.nn.transcription,
            outer_wsize=T_SIZE,
            outer_depth=T_DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
            inner_out=I_OUT,
        ),
        'translation': partial(
            bc.nn.translation,
            outer_wsize=T_SIZE,
            outer_depth=T_DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
            inner_out=I_OUT,
        ),
        'inv_transcription': partial(
            bc.nn.inv_transcription,
            outer_wsize=T_SIZE,
            outer_depth=T_DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
            inner_out=I_OUT,
        ),
        'inv_translation': partial(
            bc.nn.inv_translation,
            outer_wsize=T_SIZE,
            outer_depth=T_DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
            inner_out=I_OUT,
        ),
        'sequestron_ERN': partial(bc.nn.ERN5p, wsize=ERN_SIZE, depth=ERN_DEPTH),
        'sequestron_ERN3p': partial(bc.nn.ERN3p, wsize=ERN_SIZE, depth=ERN_DEPTH),
    },
)
cfg = {
    "optimizer": "adam",
    "learning_rate": 0.0001,
    "adam_w_decay": 0.0001,
    "rng_key": np.random.randint(0, 2**32),
    # "rng_key": 11325,
    "epochs": 200,
    "compile_training": True,
    "batch_size": 8,
    "norm_factor": 1e7,
    "balance_bin_resolution": 0.5,
    "balance_threshold_quantile": 0.4,
    "balance_threshold_min": 40,
    "node_impl": node_impl,
    "nmodels": 28,
}

lib = ut.load_lib()
rng = jax.random.PRNGKey(cfg['rng_key'])

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     setup     --
# ···············································································
xp = ut.load_xp('E20221124A_ERNbandpassV2', lib)

# def pick_n_random_models(n, models):
# # select n models from models (which is a dict of modelname -> model)
# keys = list(models.keys())
# np.random.shuffle(keys)
# return {k: models[k] for k in keys[:n]}
# models = pick_n_random_models(cfg['nmodels'], all_models)
# print(f'Picked models: {list(models.keys())}')


# X, Y = bc.train.preprocess_data(models, xp.get_Y(models), cfg)
# batch_size = cfg['batch_size']
# x_batches, y_batches = du.make_batches_uniform_sampling(
# Y.values(), batch_size, rng, models.values()
# )


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                    --     generate    --
# ···············································································
# let's now generate synthetic data using the default nodes with random parameters
# and see if we can find them back easily

rng = jax.random.PRNGKey(cfg['rng_key'])
# cfg['node_impl'] = bc.nodes.DEFAULT_COMPUTE_NODES_DICT

models = xp.get_models(node_impl=cfg['node_impl'])
fwd_models = xp.get_models(node_impl=bc.nodes.DEFAULT_COMPUTE_NODES_DICT, inverse=False, numeric_inputs=True)

zerorng = jax.random.PRNGKey(0)
ikeys = jax.random.split(zerorng, len(models))
params, constraints = {}, {}
for (s, m), r in zip(fwd_models.items(), ikeys):
    params, constraints = m.init(r, pre_params=params, pre_constraints=constraints)


generator_params = params

nsamples = 100000
Xsynth = {}
Ysynth = {}
for (s, m), r in tqdm(zip(fwd_models.items(), ikeys)):
    Xsynth[s] = jax.random.uniform(r, (nsamples, m.n_inputs), minval=0, maxval=10)
    # Xsynth[s] = jax.random.normal(r, (nsamples, m.n_inputs)) * 0.1 + 0.5
    # use a lognormal distribution
    # Xsynth[s] = jax.random.lognormal(r, (nsamples, m.n_inputs), 0, 0.1)
    # Xsynth[s] = np.random.lognormal(0, 0.5, (nsamples, m.n_inputs))
    vmapped = jit(jax.vmap(m, in_axes=(None, 0, None)))
    Ysynth[s] = vmapped(generator_params, Xsynth[s], r)

cfg["norm_factor"] = 1
cfg["balance_bin_resolution"] = 0.2
X, Y = bc.train.preprocess_data(models, Ysynth, cfg)
batch_size = 12

model_values = []
Y_values = []
for k, m in models.items():
    model_values.append(m)
    Y_values.append(Y[k])

x_batches, y_batches = du.make_batches_uniform_sampling(Y_values, batch_size, rng, model_values)

# s,m = list(models.items())[2]
# du.model_heatmap(m, Ysynth[s], inner_resolution=0.2, lims=(1e-4,1e2))

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                    --     some safety checks     --
# ···············································································

# let's first verify that all inverted inputs are correctly mapped
# let's do that in 2 ways:
# - first from data and annotation of input/output only
# - second let's build the models with the default nodes, which are perfectly invertible,
# and then check that the inverted inputs are correctly mapped


def check_model(m, x, y):
    outp = m.get_output_proteins()  # name of output proteins
    inp = m.get_inverted_input_proteins()  # name of input proteins
    in_pos = m.get_inverted_input_positions()
    # in_pos contains input_pos -> output_pos
    assert len(inp) == len(in_pos)
    assert len(inp) == len(set(inp))
    for iname in inp:
        assert iname in outp
    for ipos, outpos in in_pos.items():
        assert inp[ipos] == outp[outpos]
        assert np.all(x[:, ipos] == y[:, outpos])
    mdef = bc.ComputeGraphModel(m.network)
    mdef.build(bc.nodes.DEFAULT_COMPUTE_NODES_DICT)
    zerorng = jax.random.PRNGKey(0)
    p, _ = mdef.init(zerorng)
    vmapped = jit(jax.vmap(mdef, in_axes=(None, 0, None)))
    ydef = vmapped(p, x, zerorng)
    # ut.plot_networks([mdef.network])
    for ipos, outpos in in_pos.items():
        assert np.allclose(x[:, ipos], ydef[:, outpos])


for k, m in tqdm(models.items()):
    check_model(m, X[k], Y[k])

print('data ok')

for i, (k, m) in tqdm(enumerate(models.items())):
    for xb, yb in tqdm(list(zip(x_batches, y_batches))[:2]):
        check_model(m, xb[i, :, : m.n_inputs], yb[i, :, : m.n_outputs])

print('batches ok')
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                           --     train     --
# ···············································································

epochs = []


def console_log(epoch, cfg, epoch_history=None, **_):
    if epoch_history is not None:
        loss = np.array(epoch_history['loss'])
        avg = np.mean(loss)
        std = np.std(loss)
        lmin, lmax = jnp.min(loss), jnp.max(loss)
        print(
            f'[{epoch}/{cfg["epochs"]}] loss: {avg:.3f} ± {std:.3f} [min {lmin:.3f}, max {lmax:.3f}]'
        )
        epochs.append(epoch_history)


loggers = [
    (1, console_log),
]

cfg['epochs'] = 10
train_history = bc.train.train_models(model_values, x_batches, y_batches, cfg, loggers)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                        --     plot tools     --
# ···············································································

alllosses = np.concatenate([np.array(e['loss']) for e in epochs[:2]])
smoothwin = 100
smoothed = jnp.convolve(alllosses, jnp.ones(smoothwin) / smoothwin, mode='valid')
windowed_std = (
    jnp.convolve(alllosses**2, jnp.ones(smoothwin) / smoothwin, mode='valid') - smoothed**2
)
windowed_std = jnp.sqrt(windowed_std)
plt.figure()
plt.plot(smoothed)
plt.fill_between(
    np.arange(len(smoothed)),
    smoothed - windowed_std,
    smoothed + windowed_std,
    alpha=0.25,
)
plt.show()

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

best_params = bu.get_pytree(epochs[-1]['params'], len(x_batches) - 1)
generator_params['shared']
best_params['shared']


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     a bug??     --
# ···············································································

loss_f = bc.train.mse_loss


def loss_func(params, X, Y, rng_key):
    nmodels = len(models)
    assert len(X) == nmodels, f"Expected {nmodels} models, got {X.shape}"
    assert len(Y) == nmodels
    K = jax.random.split(rng_key, nmodels)
    res = jnp.array(
        [
            loss_f(vmap(partial(m, params, rng_key=k))(x[:, : m.n_inputs]), y, m.n_outputs)
            for m, x, y, k in zip(models.values(), X, Y, K)
        ]
    ).mean()
    return res


loss_func(best_params, x_batches[0], y_batches[0], jax.random.PRNGKey(0))

##

# BIG ERROR RIGHT THERE!?
loss_func(
    generator_params, x_batches[0], y_batches[0], jax.random.PRNGKey(0)
)  # WHY IS THIS NOT ZERO???????????????????????????????

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


params = generator_params
params = best_params


def mse(y, yhat):
    return jnp.mean((y - yhat) ** 2)


k, m = list(models.items())[10]

ut.plot_node('transcription', generator_params, m)
ut.plot_node('transcription', best_params, m)
ut.plot_node('translation', generator_params, m)
ut.plot_node('translation', best_params, m)

best_params['shared']
generator_params['shared']

#hmmmm the node plots don't look right. 
# Actually I think it's normal that the transcription nodes don't have
# to have the same slope. The translation nodes should be the same though.
# since the ERN is a relu, you can scale both inputs by whtever 

ut.plot_networks([m.network])
# extra = m.network.compute_graph[m.network.compute_graph.type == 'sequestron_ERN'].extra.to_list()
ut.plot_node('sequestron_ERN', best_params, m, xlim=(-10, 100), n_inputs=2, mode='3d')
best_params['node'][k]
generator_params['node'][k]

X[k][0]
# get index of first element of Y[k][:, 0] that is > 1.0
ex_id = jnp.where(Y[k][:, 0] > 1.0)[0][0]
Y[k][ex_id]
xx = X[k][ex_id]

_,rb = m.collect_all_results(best_params, xx, rng_key=jax.random.PRNGKey(0))
ut.plot_networks([m.network], outputs=[rb], W=1000, H=2000)

_,rg = m.collect_all_results(generator_params, xx, rng_key=jax.random.PRNGKey(0))
ut.plot_networks([m.network], outputs=[rg], W=1000, H=2000)

Xsynth[k].shape

total_loss = 0
for k, m in models.items():
    yhat = vmap(m, in_axes=(None, 0, None))(params, X[k], jax.random.PRNGKey(0))
    l = mse(Y[k], yhat)
    sqerr = (Y[k] - yhat) ** 2
    maxerr = jnp.max(sqerr, axis=0)
    print(f'{k}: {l:.5f} (maxerr: {maxerr})')
    total_loss += l
print(f'total loss: {total_loss:.5f}, avg: {total_loss/len(models):.5f}')

##

yvalues = list(Y.values())
mvalues = models.values()
mnames = [m.network.name for m in mvalues]
yfromname = [Y[k] for k in mnames]
for i, name in enumerate(mnames):
    print(f'{name}: ynameshape = {yfromname[i].shape}, yshape = {yvalues[i].shape}')
    assert yfromname[i] == yvalues[i]
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# O M G, I am so dumb... When I generate batches using Y.values() and models.values(), the order
# is not preserved...


total_loss = 0
# params = best_params
params = generator_params
# now with batches:

mid = 10
k, m = list(models.items())[mid]
k
xmid = list(X.values())[mid]


yhat = vmap(m, in_axes=(None, 0, None))(
    params, x_batches[0][mid, :, : m.n_inputs], jax.random.PRNGKey(0)
)
y = y_batches[0][mid, :, : m.n_outputs]
yhat_raw = vmap(m, in_axes=(None, 0, None))(params, X[k], jax.random.PRNGKey(0))
y_raw = Y[k]
l = mse(y, yhat)
l_raw = mse(y_raw, yhat_raw)
sqerr = (y - yhat) ** 2
maxerr = jnp.max(sqerr, axis=0)
print(f'{k}: {l:.5f} (maxerr: {maxerr})')
total_loss += l
