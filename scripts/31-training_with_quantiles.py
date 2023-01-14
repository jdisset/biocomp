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
    "batch_size": 32,
    "n_batches": 128,
    "norm_factor": 1e6,
    "density_quantile_threshold": 0.1,
    "node_impl": node_impl,
}
##────────────────────────────────────────────────────────────────────────────}}}

### {{{                         --     load data     --
lib = ut.load_lib()
# xp = ut.load_xp('E20221124A_ERNbandpassV2', lib)
xp = ut.load_xp('20220501-GW-l1vsl2', lib)
rng = jax.random.PRNGKey(cfg['rng_key'])
models = xp.get_models(node_impl=cfg['node_impl'])
X_raw, Y_raw = xp.get_XY(models)
##────────────────────────────────────────────────────────────────────────────}}}

### {{{                      --     display models     --

ut.plot_networks([m.network for m in models.values()])

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                 --     select a model to work on     --
# modelname = '(1:5; 100+73)+100i'
modelname = 'L2all_pGW42+10'
model = models[modelname]
X_r, Y_r = X_raw[modelname], Y_raw[modelname]

ut.plot_networks([model.network], ['/Users/jeandisset/Desktop/model_l2.pdf'])

##────────────────────────────────────────────────────────────────────────────}}}

### {{{            --     rescale and move to log space     --


def rescale(X, norm_factor):
    return np.log10(1.0 + (X / norm_factor))


cfg['norm_factor'] = 1e3

X = rescale(X_r, cfg['norm_factor'])
Y = rescale(Y_r, cfg['norm_factor'])

##────────────────────────────────────────────────────────────────────────────}}}
### {{{            --     estimate density of X using kde     --

x_prots = model.get_inverted_input_proteins()
y_prots = model.get_output_proteins()

from jax.scipy.stats import gaussian_kde

kde = gaussian_kde(X.T)

plot_over_2d(
    kde.evaluate,
    vmin=0,
    vmax=3,
    resolution=200,
    title=f'kde of X ({len(X)} points)',
    value_range=(0, 2.5),
)

##────────────────────────────────────────────────────────────────────────────}}}
### {{{                         --     rebalance     --
densities = kde(X.T) + 1e-19
threshold = np.quantile(densities, 0.07)
diceroll = jax.random.uniform(rng, shape=(len(densities),))
selected = np.where((densities < threshold) | (diceroll < (threshold / (densities * 1.2))))[0]

Xrebalanced = X[selected]
Yrebalanced = Y[selected]
kde_balanced = gaussian_kde(Xrebalanced.T)
plot_over_2d(
    kde_balanced,
    vmin=0,
    vmax=3,
    resolution=200,
    title=f'kde of X after rebalancing ({len(Xrebalanced)} points)',
    value_range=(0, 0.25),
)

##────────────────────────────────────────────────────────────────────────────}}}

dman = du.DataManager(X_raw, Y_raw, models)

xbatches, ybatches = dman.get_batches(cfg, rng)

