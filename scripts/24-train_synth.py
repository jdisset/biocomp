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
I_DEPTH = 3
I_OUT = 4
ERN_SIZE = 32
ERN_DEPTH = 4
MEFL_SIZE = 32
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

# node_impl = bc.nodes.DEFAULT_COMPUTE_NODES_DICT

cfg = {
    # "optimizer": "adam",
    "optimizer": "amsgrad",
    "learning_rate": 3e-4,
    "rng_key": np.random.randint(0, 2**32),
    # "rng_key": 11325,
    "epochs": 500,
    "compile_training": True,
    "batch_size": 4,
    # "norm_factor": 1,
    "norm_factor": 1e7,
    "balance_bin_resolution": 0.25,
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
# {{{                    --     generate    --
# ···············································································
xp = ut.load_xp('E20221124A_ERNbandpassV2', lib)

rng = jax.random.PRNGKey(cfg['rng_key'])

models = xp.get_models(node_impl=cfg['node_impl'])
# fwd_models = xp.get_models(node_impl=bc.nodes.DEFAULT_COMPUTE_NODES_DICT, inverse=False, numeric_inputs=True)

# zerorng = jax.random.PRNGKey(0)
# ikeys = jax.random.split(rng, len(models))
# params, constraints = {}, {}
# for (s, m), r in zip(fwd_models.items(), ikeys):
    # params, constraints = m.init(r, pre_params=params, pre_constraints=constraints)

# generator_params = params

# print(f'generating synthetic data with shared params: {generator_params["shared"]}')

# nsamples = 30000
# Xsynth = {}
# Ysynth = {}
# for (s, m), r in tqdm(zip(fwd_models.items(), ikeys)):
    # Xsynth[s] = jax.random.uniform(r, (nsamples, m.n_inputs), minval=0, maxval=10)
    # vmapped = jit(jax.vmap(m, in_axes=(None, 0, None)))
    # Ysynth[s] = vmapped(generator_params, Xsynth[s], r)

# cfg["norm_factor"] = 1
# cfg["balance_bin_resolution"] = 0.2
# X, Y = bc.train.preprocess_data(models, Ysynth, cfg)

X, Y = xp.get_XY(models)
X, Y = bc.train.preprocess_data(models, Y, cfg)

models_list= []
Y_list = []
for k, m in models.items():
    models_list.append(m)
    Y_list.append(Y[k])

x_batches, y_batches = du.make_batches_uniform_sampling(Y_list, cfg['batch_size'], rng, models_list)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                        --     stat tools     --
# ···············································································
def models_data_fig(models, Y):
    for sample, model in models.items():
        out_proteins = model.get_output_proteins()
        in_proteins = model.get_inverted_input_proteins()
        z_prot = set(out_proteins) - set(in_proteins)
        if len(out_proteins) >= 4 and len(out_proteins) <= 5:
            fig, ax = du.model_heatmap(model, Y[sample])
        else:
            fig, ax = du.model_parallel_coords(model, Y[sample])
        yield sample, model, fig, ax


@partial(jit, static_argnums=(1,))
def compstats(v, smooth_win=1):
    medians = vmap(jnp.median)(v)
    mins = vmap(jnp.min)(v)
    maxs = vmap(jnp.max)(v)
    p20s = vmap(lambda x: jnp.percentile(x, 20))(v)
    p80s = vmap(lambda x: jnp.percentile(x, 80))(v)
    if smooth_win > 1:
        medians = jnp.convolve(medians, jnp.ones(smooth_win) / smooth_win, mode='same')
        p80s = jnp.convolve(p80s, jnp.ones(smooth_win) / smooth_win, mode='same')
        p20s = jnp.convolve(p20s, jnp.ones(smooth_win) / smooth_win, mode='same')
        maxs = jnp.convolve(maxs, jnp.ones(smooth_win) / smooth_win, mode='same')
        mins = jnp.convolve(mins, jnp.ones(smooth_win) / smooth_win, mode='same')
    return medians, p20s, p80s, mins, maxs


def get_epoch_stats(epoch_data, smooth_win=1):
    stats = {'grad': {}, 'params': {}}
    for k, v in epoch_data['grad']['shared'].items():
        stats['grad'][k] = compstats(v, smooth_win=smooth_win)
    for k, v in epoch_data['params']['shared'].items():
        stats['params'][k] = compstats(v, smooth_win=smooth_win)
    return stats


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     training     --
# ···············································································
import wandb as wb

# project = 'train_synth_00'
project = 'train_bandpassgeorg_00'
log_grads_and_params_to_wandb = False

if project is not None:
    wb.init(config=cfg, project=project, entity="jdisset", reinit=True)

print(f"About to train {len(models)} models.")

zero_rng = jax.random.PRNGKey(0)

jitted_models = {
    s: jit(jax.vmap(partial(m, rng_key=zero_rng), in_axes=(None, 0))) for s, m in models.items()
}

import time

save_dir = '../__out' if project is None else f'../__out/{wb.run.name}'


def local_save(epoch, cfg, epoch_history=None, **_):
    if epoch_history is None:
        return
    t0 = time.time()
    print(f"Saving epoch {epoch} to disk")
    if epoch <= 2:
        du.save(epoch_history, f'{save_dir}/epoch_{epoch}_full.pkl')
        print(f"Done")
    stats = get_epoch_stats(epoch_history)
    du.save(stats, f'{save_dir}/epoch_{epoch}_stats.pkl')
    # du.save(generator_params, f'{save_dir}/generator_params.pkl')
    print(f"Done in {time.time() - t0:.2f}s")


def wandb_plot_pred(epoch, cfg, models, X, Y, epoch_history=None, nbatches=len(x_batches), **_):
    if epoch == 0:
        gtruth = []
        for sample, model, fig, ax in models_data_fig(models, Y):
            gtruth.append(wb.Image(fig, caption=f'{model.network.name} ground truth'))
            plt.close(fig)
        wb.log({'ground truth': gtruth})

    if epoch_history is None:
        return

    t0 = time.time()
    print(f'Logging predictions for epoch {epoch}')
    params = bu.get_pytree(epoch_history['params'], nbatches - 1)
    Y_pred = {s: jitted_models[s](params, X[s]) for s in models}
    pred = []
    try:
        for sample, model, fig, ax in models_data_fig(models, Y_pred):
            pred.append(wb.Image(fig, caption=f'{model.network.name} predicted'))
            plt.close(fig)
    except Exception as e:
        print(e)
        print("Failed to plot predictions")

    wb.log({'prediction': pred})
    print(f'Done logging predictions for epoch {epoch} in {time.time() - t0:.2f}s')


def wandb_log_epoch(epoch, cfg, epoch_history=None, nbatches=len(x_batches), **_):
    if epoch == 0:
        wb.log({'config': cfg})
        # wb.log({'generator_params': generator_params})

    if epoch_history is None:
        return

    print(f"Logging epoch {epoch} to wandb")
    # measure time now:
    t0 = time.time()
    losses = np.array(epoch_history['loss'])
    for loss in losses:
        wb.log({'loss': loss})
    del losses
    if log_grads_and_params_to_wandb:
        stats = du.load(f'{save_dir}/epoch_{epoch}_stats.pkl')
        for k, v in stats['grad'].items():
            wb.log({f'grad/{k}': v})
        for k, v in stats['params'].items():
            wb.log({f'params/{k}': v})
    print(f"Logging epoch {epoch} to wandb took {time.time() - t0:.2f}s")


def console_log(epoch, cfg, epoch_history=None, **_):
    if epoch_history is not None:
        loss = np.array(epoch_history['loss'])
        avg = np.mean(loss)
        std = np.std(loss)
        lmin, lmax = jnp.min(loss), jnp.max(loss)
        print(
            f'[{epoch}/{cfg["epochs"]}] loss: {avg:.3f} ± {std:.3f} [min {lmin:.3f}, max {lmax:.3f}]'
        )


if project is not None:
    loggers = [
        (1, console_log),
        (1, local_save),
        (1, wandb_log_epoch),
        (10, partial(wandb_plot_pred, models=models, X=X, Y=Y)),
    ]
else:
    loggers = [
        (1, console_log),
    ]

train_history = bc.train.train_models(models_list, x_batches, y_batches, cfg, loggers)

print('done')

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
