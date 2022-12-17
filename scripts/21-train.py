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

T_SIZE = 64
T_DEPTH = 3
I_SIZE = 64
I_DEPTH = 2
I_OUT = 16
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
    "epochs": 5,
    "compile_training": True,
    "batch_size": 8,
    "norm_factor": 1e7,
    "balance_bin_resolution": 0.5,
    "balance_threshold_quantile": 0.4,
    "balance_threshold_min": 40,
    "node_impl": node_impl,
}

lib = ut.load_lib()
rng = jax.random.PRNGKey(cfg['rng_key'])

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     setup     --
# ···············································································
xp = ut.load_xp('E20221124A_ERNbandpassV2', lib)

rng = jax.random.PRNGKey(cfg['rng_key'])
models = xp.get_models(node_impl=cfg['node_impl'])

# NMODELS = 1
# models = {k: v for k, v in list(models.items())[:NMODELS]}

X, Y = bc.train.preprocess_data(models, xp.get_Y(models), cfg)
batch_size = cfg['batch_size']
x_batches, y_batches = du.make_batches_uniform_sampling(
    Y.values(), batch_size, rng, models.values()
)

# x_batches = x_batches[:3]
# y_batches = y_batches[:3]


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     training     --
# ···············································································
import wandb as wb

project = 'bp_train_00'

wb.init(config=cfg, project=project, entity="jdisset", reinit=True)
print(f"About to train {len(models)} models. wb run {wb.run.name}")

def models_data_fig(models, Y):
    for sample, model in models.items():
        out_proteins = model.get_output_proteins()
        in_proteins = model.get_inverted_input_proteins()
        z_prot = set(out_proteins) - set(in_proteins)
        # print(f'{sample}: {in_proteins} -> {out_proteins} [diff:{z_prot}]')
        if len(z_prot) == 1 and len(in_proteins) == 2:
            fig, ax = du.model_heatmap(model, Y[sample])
            plt.show()
        else:
            fig, ax = du.model_parallel_coords(model, Y[sample])
            plt.show()
        yield sample, model, fig, ax


# we take random samples to plot
nsubsamples = 1000
# X_subsamples = {k: jax.random.choice(rng, v, (nsubsamples,)) for k, v in X.items()}
# let's make it so that we can also take the same samples for Y:

zero_rng = jax.random.PRNGKey(0)
indices = {k: jax.random.choice(zero_rng, v.shape[0], (nsubsamples,)) for k, v in X.items()}
X_samples = {k: v[indices[k]] for k, v in X.items()}
Y_samples = {k: v[indices[k]] for k, v in Y.items()}
# maxvals = {k: v.max() for k, v in X.items()}
# minvals = {k: v.min() for k, v in X.items()}

jitted_models = {
    s: jit(jax.vmap(partial(m, rng_key=zero_rng), in_axes=(None, 0)))
    for s, m in models.items()
}

for sample, model, fig, ax in models_data_fig(models, Y):
    pass

def wandb_plot_pred(epoch, cfg, models, X, Y, epoch_history=None, nbatches=len(x_batches), **_):
    if epoch == 0:
        gtruth = []
        for sample, model, fig, ax in models_data_fig(models, Y):
            gtruth.append(wb.Image(fig, caption=f'{model.network.name} ground truth'))
            plt.close(fig)
        wb.log({'ground truth': gtruth})

    if epoch_history is None:
        return

    print(f'Logging predictions for epoch {epoch}')
    params = bu.get_pytree(epoch_history['params'], nbatches-1)
    Y_pred = {s: jitted_models[s](params, X[s]) for s in models}
    pred = []
    for sample, model, fig, ax in models_data_fig(models, Y_pred):
        pred.append(wb.Image(fig, caption=f'{model.network.name} predicted'))
        plt.close(fig)

    wb.log({'prediction': pred})
    print('Done logging predictions')

def wandb_log_epoch(epoch, cfg, epoch_history=None, nbatches=len(x_batches), **_):
    if epoch_history is not None:
        print(f"Logging epoch {epoch}")
        losses = np.array(epoch_history['loss'])
        param_list = bu.param_unstack(epoch_history['params'],len(x_batches))
        for loss, params in zip(losses, param_list):
            wb.log({'loss': loss})
            wb.log({'shared_params': params['shared']})
            wb.log({'params': params})
        del param_list
        del losses
        print(f"Done")

def console_log(epoch, cfg, epoch_history=None, **_):
    if epoch_history is not None:
        loss = np.array(epoch_history['loss'])
        avg = np.mean(loss)
        std = np.std(loss)
        lmin, lmax = jnp.min(loss), jnp.max(loss)
        print(f'[{epoch}/{cfg["epochs"]}] loss: {avg:.3f} ± {std:.3f} [min {lmin:.3f}, max {lmax:.3f}]')

loggers = [
    (1, console_log),
    (1, wandb_log_epoch),
    (10, partial(wandb_plot_pred, models=models, X=X_samples, Y=Y_samples)),
]

train_history = bc.train.train_models(models.values(), x_batches, y_batches, cfg, loggers)

print('done')

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
