# ─────────────────────────────────────────────────────────────────────────────
#                                     SETUP
# ───────────────────────────────────── ▼ ─────────────────────────────────────
### {{{                          --     imports     --
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
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10.0, 10.0]
plt.rcParams['figure.dpi'] = 300

##────────────────────────────────────────────────────────────────────────────}}}
### {{{                          --     config     --
T_SIZE = 64
T_DEPTH = 3
I_SIZE = 64
I_DEPTH = 2
I_OUT = 8
ERN_SIZE = 128
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
    "learning_rate": 1e-4,
    "rng_key": np.random.randint(0, 2**32),
    # "rng_key": 11325,
    "epochs": 10,
    "compile_training": True,
    "batch_size": 5000,
    "n_batches": 1,
    "norm_factor": 1e3,
    "kde_bw_method": 0.1,
    "density_quantile_threshold": 0.1,
    "node_impl": node_impl,
}
cfg["batch_size"] = 5000
# cfg["n_batches"] = 1
cfg["kde_bw_method"] = 0.05
cfg["log_factor"] = 5e2
cfg["max_value"] = 1e7
cfg["density_quantile_threshold"] = 0.1

##────────────────────────────────────────────────────────────────────────────}}}

### {{{ --     create a composite training dataset from uOrfs + ERN data     --

lib = ut.load_lib()

uorf_xp = ut.load_xp('2022-11-10_uORFs_and_company', lib)
uorf_models, uorf_samples = uorf_xp.build_models(node_impl=cfg['node_impl'], inverse='all')
uorf_X, uorf_Y = uorf_xp.get_XY(uorf_models, uorf_samples)

ern_xp = ut.load_xp('20220501-GW-l1vsl2', lib)
ern_models, ern_samples = ern_xp.build_models(node_impl=cfg['node_impl'], inverse='all')
ern_X, ern_Y = ern_xp.get_XY(ern_models, ern_samples)
# ernmodel = ern_models[ern_samples.index('L2all_pGW42+10')]

raw_X, raw_Y = uorf_X + ern_X, uorf_Y + ern_Y
models = uorf_models + ern_models

dman = du.DataManager(raw_X, raw_Y, models, cfg)

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                 --     show data for each model     --
rng = jax.random.PRNGKey(0)

from pathlib import Path

save_dir = Path('~/Desktop/32-predict_things').expanduser()

figs, axes = zip(*[du.mkfig(1, 2) for _ in range(len(dman.kdes))])

ut.plot_networks(
    [model.network for model in dman.models], axes=[ax[0] for ax in axes], show_title=False
)

for i, (model, X, Y, kde, fig, ax) in tqdm(
    list(enumerate(zip(dman.models, dman.X, dman.Y, dman.kdes, figs, axes)))
):
    subsample = du.optimal_density_subsample(X, kde, rng, quantile_threshold=0.1)
    x, y = X[subsample], Y[subsample]
    if x.shape[1] == 1:
        du.smooth_1d(x, y, model, dman.rescale, ax[1])
    else:
        du.smooth_2d(x, y, model, dman.rescale, ax[1])
    fig.suptitle(f'{model.network.name} \n(after density-based resampling of {x.shape[0]} points)')
    # save to desktop
    sdir = save_dir / 'data2' / f'{model.network.name}_{i}.png'
    sdir.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(sdir)
    plt.close(fig)


##────────────────────────────────────────────────────────────────────────────}}}
