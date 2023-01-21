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

### {{{                     --     load ERN data     --
lib = ut.load_lib()
# xp = ut.load_xp('E20221124A_ERNbandpassV2', lib)
xp = ut.load_xp('20220501-GW-l1vsl2', lib)
rng = jax.random.PRNGKey(cfg['rng_key'])
ERN_models = xp.get_models(node_impl=cfg['node_impl'])
X_raw_ERN, Y_raw_ERN = xp.get_XY(ERN_models)
[jnp.log10(1 + x).max() for x in X_raw_ERN.values()]
##────────────────────────────────────────────────────────────────────────────}}}
### {{{                     --     generate batches     --
dman_ERN = du.DataManager(X_raw_ERN, Y_raw_ERN, ERN_models, cfg)
x_batches, y_batches = dman_ERN.get_batches(rng)
z_batches = jax.random.uniform(rng, y_batches.shape)
##────────────────────────────────────────────────────────────────────────────}}}

### {{{                       --     load uORFs xp     --
lib = ut.load_lib()
xpname = '2022-11-10_uORFs_and_company'
xp = ut.load_xp(xpname, lib)
uorfs_models = xp.get_models(node_impl=cfg['node_impl'])
X_raw_uorfs, Y_raw_uorfs = xp.get_XY(uorfs_models)
[max(jnp.log10(1 + x)) for x in X_raw_uorfs.values()]
[(x / 1e6).max() for x in X_raw_uorfs.values()]
##────────────────────────────────────────────────────────────────────────────}}}
### {{{                      --     display models     --
from pathlib import Path

save_to = Path('~/Desktop/models_uorfs').expanduser()
save_to.mkdir(parents=True, exist_ok=True)
ut.plot_networks(
    [m.network for m in uorfs_models.values()],
    [str(save_to / f'{n}.pdf') for n in uorfs_models.keys()],
    W=400,
    H=1200,
)

##────────────────────────────────────────────────────────────────────────────}}}
### {{{               --     plotting resampling of uORFS     --
cfg["batch_size"] = 5000
# cfg["n_batches"] = 1
cfg["kde_bw_method"] = 0.05
cfg["log_factor"] = 1
cfg["max_value"] = 1e7
cfg["density_quantile_threshold"] = 0.1

# all_log_factors = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]

save_to = Path('~/Desktop/models_uorfs/resampling').expanduser()
save_to.mkdir(parents=True, exist_ok=True)

cfg["batch_size"] = 8000
# cfg["n_batches"] = 1
cfg["kde_bw_method"] = 0.05
cfg["log_factor"] = 1
cfg["max_value"] = 1e7
cfg["density_quantile_threshold"] = 0.08

all_log_factors = jnp.geomspace(1, 1000, 50)
# all_log_factors = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]

for i, log_factor in enumerate(all_log_factors):
    cfg["log_factor"] = log_factor
    dman = du.DataManager(X_raw_uorfs, Y_raw_uorfs, uorfs_models, cfg)
    rng = jax.random.PRNGKey(cfg['rng_key'])
    x_batches, y_batches = dman.get_batches(rng)
    x_batches.shape
    x_batches.max()
    xmax = 1

    from scipy.spatial import cKDTree

    for logX, logY, mname in list(zip(x_batches[0], y_batches[0], dman.key_order))[:1]:
        fig, axes = du.mkfig(1, 4)
        mod = uorfs_models[mname]
        input_name = mod.get_inverted_input_proteins()[0]
        output_names = mod.get_output_proteins()
        which_output = int(not (output_names.index(input_name)))
        originalX = dman.rescale(X_raw_uorfs[mname])
        originalY = dman.rescale(Y_raw_uorfs[mname])

        ax = axes[0]
        ax.scatter(originalX, originalY[:, which_output], s=1, c='k', alpha=0.1)
        ax.set_title('Original scatter')
        ax.set_xlabel(input_name)
        ax.set_ylabel(output_names[which_output])
        ax.set_xlim(-0.1, xmax)
        ax.set_ylim(-0.1, xmax)
        unscaled_ticks = jnp.geomspace(1, cfg['max_value'], 10)
        ticks = dman.rescale(unscaled_ticks)
        tlabels = [du.scformat.format("{:m}", x) for x in unscaled_ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(tlabels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(tlabels)

        ax = axes[1]
        ax.scatter(logX, logY[:, which_output], s=1, c='k', alpha=0.4)
        ax.set_title('Resampled scatter')
        ax.set_xlabel(input_name)
        ax.set_ylabel(output_names[which_output])
        ax.set_xlim(-0.1, xmax)
        ax.set_ylim(-0.1, xmax)
        unscaled_ticks = jnp.geomspace(1, cfg['max_value'], 10)
        ticks = dman.rescale(unscaled_ticks)
        tlabels = [du.scformat.format("{:m}", x) for x in unscaled_ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(tlabels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(tlabels)

        ax = axes[2]
        tree = cKDTree(logX)
        res = 500
        x = jnp.linspace(-0.1, xmax, res).reshape(-1, 1)
        Z = du.get_knn_mean(x, logY[:, which_output], tree=tree, knn=200)
        Zq1 = du.get_knn_quantile(x, logY[:, which_output], qu=0.1, tree=tree, knn=200)
        Zq9 = du.get_knn_quantile(x, logY[:, which_output], qu=0.9, tree=tree, knn=200)
        ax.plot(x, Z, c='k')
        ax.fill_between(x[:, 0], Zq1, Zq9, alpha=0.25, color='k')
        ax.set_title('Smoothed knn mean and [0.1 - 0.9] quantile')
        ax.set_xlabel(input_name)
        ax.set_ylabel(output_names[which_output])
        ax.set_xlim(-0.1, xmax)
        ax.set_ylim(-0.1, xmax)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tlabels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(tlabels)

        ax = axes[3]
        kde = dman.kdes[mname]
        x = jnp.linspace(0, xmax, res).reshape(-1, 1)
        y = kde(x.T)
        kde_new = du.gaussian_kde(logX.T, bw_method=0.1)
        ynew = kde_new(x.T)
        ax.plot(x, y / y.max(), c='#888888', ls='--')
        ax.plot(x, ynew / ynew.max(), c='k')
        ax.legend(['original', 'resampled'])

        # draw a horizontal line at density_quantile_threshold
        ax.axhline(cfg['density_quantile_threshold'], c='#AAAAAA', ls=':')
        # write "threshold" on the line
        ax.text(
            0.5,
            cfg['density_quantile_threshold'],
            'density threshold',
            ha='center',
            va='bottom',
            c='#AAAAAA',
        )
        ax.set_title('Input density estimation')

        ax.set_xticks(ticks)
        ax.set_xticklabels(tlabels)

        ax.set_xlabel(input_name)
        ax.set_ylabel('density')

        fig.suptitle(f'Data for {mname}\nrescaling with log_factor of {cfg["log_factor"]:.1f}')

        # save in save_to/dataplot_mname.png
        fig.savefig(save_to / f'dataplot_{mname}_{i}.png', dpi=300)
        plt.show()


##────────────────────────────────────────────────────────────────────────────}}}

### {{{ --     create a composite training dataset from uOrfs + ERN data     --

# first we can create all the inverse versions of each model
# (in the case of the uOrf data, since it's a 1:1 coTx, both branches are invertible
# and it'll nake sure that the uORFs effects are learnt in both directions)

uorf_xp = ut.load_xp('2022-11-10_uORFs_and_company', lib)
uorf_samples, uorf_models = uorf_xp.build_models(node_impl=cfg['node_impl'], inverse='shortest')

ernmodel_name = 'L2all_pGW42+10'
combined_models = {**uorfs_models, ernmodel_name: ERN_models[ernmodel_name]}
combined_X = {**X_raw_uorfs, ernmodel_name: X_raw_ERN[ernmodel_name]}
combined_Y = {**Y_raw_uorfs, ernmodel_name: Y_raw_ERN[ernmodel_name]}
dman = du.DataManager(combined_X, combined_Y, combined_models, cfg)

ut.plot_networks([m.network for m in combined_models.values()])

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                       --     training loop     --

##────────────────────────────────────────────────────────────────────────────}}}
