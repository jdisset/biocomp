### {{{                          --     imports     --
import biocomp as bc
from biocomp import datautils as du
import matplotlib.pyplot as plt
import scriptutils as ut
from pathlib import Path
import json5
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import gaussian_kde
import jax
import optax
from jax import jit, vmap, value_and_grad
from jax.tree_util import Partial as partial
from tqdm import tqdm
import biocomp.defaults as bdf
import pandas as pd
import copy

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                        --     config     --
T_SIZE = 64
T_DEPTH = 4
I_SIZE = 64
I_DEPTH = 3
I_OUT = 8
ERN_SIZE = 128
ERN_DEPTH = 4
MEFL_SIZE = 64
MEFL_DEPTH = 4

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

config = {
    **bdf.DEFAULT_CONFIG,
    **{
        'node_impl': node_impl,
        'rng_key': np.random.randint(0, 2**32),
        "batch_size": 16,
        "n_batches": 4,
        "epochs": 12,
    },
}

##────────────────────────────────────────────────────────────────────────────}}}

lib = ut.load_lib()
lib.disable_all_sequestrons()
lib.set_enabled_sequestrons(['ERN'])


cascade_xp = ut.load_xp('2023-03-19_CascadesV2', lib, data_path='./data/calibrated_data')
dman = du.DataManager.from_xps([cascade_xp], config, inverse='all')
names = [m.node_namespace for m in dman.get_models()]


## saving data and model plots

base_dir = Path('~/Desktop/cascade_v2_renamed/bfp_x/').expanduser()
# escape names so that they can be used as filenames (they have : and /)
enames = [n.replace(':', '_').replace('/', '-') for n in names]


mdir = [base_dir / n for n in enames]
for m in mdir:
    m.mkdir(parents=True, exist_ok=True)

model_paths = [str(m / 'network.pdf') for m in mdir]

##
model = dman.get_models()[4]
model.get_inverted_input_proteins()

# ut.plot_networks([m.network for m in dman.get_models()], model_paths)

min_points = 30
radius = 0.1
knn = 2000
m
for i, n in tqdm(list(enumerate(enames))[3:]):
    model = dman.get_models()[i]
    ninputs = model.n_inputs
    if ninputs == 3:
        fig, axes = du.mkfig(1, 5)
        axes = axes.flatten()
        du.model_plot(
            dman,
            i,
            ax=None,
            axes=axes,
            min_points=min_points,
            radius=radius,
            knn=knn,
            slices=np.linspace(0, 0.8, 5),
            method='mean',
            input_order=[0,1,2],
            # qu=0.5,
            # vmax=1000,
            # density_as_alpha=True,
            # density_threshold = 100,
            # density_plot=True,
        )
    else:
        fig, ax = du.mkfig(1, 1)
        du.model_plot(
            dman,
            i,
            ax,
            min_points=min_points,
            radius=radius,
            knn=knn,
            method='mean',
            # qu=0.5,
            # density_as_opacity=True,
        )
    fig.tight_layout()
    # add 1 inch border:
    fig.set_size_inches(fig.get_size_inches() + 1)
    fig.savefig(mdir[i] / f'data_{n}.pdf', format='pdf')
print('done')

##

fig, ax = du.mkfig(1, 1)
du.model_plot(dman, 0, ax, min_points=min_points, radius=0.025, knn=knn)


fig, ax = du.mkfig(1, 1)
du.model_plot(dman, 1, ax, min_points=min_points, radius=0.03, knn=knn)


fig, ax = du.mkfig(1, 1)
du.model_plot(dman, 2, ax, min_points=min_points, radius=0.03, knn=knn)


m = dman.get_models()[0]

m.network.get_inverted_input_proteins()

##

from scipy.spatial import cKDTree

model_id = 1
model = dman.get_models()[model_id]
rescaler = dman.rescale
unscaler = dman.unscale
input_order, input_names, output_pos, output_name, ticks, tlabels = du.model_ticks_and_labels(
    model, rescaler
)
x, y = dman.get_X()[model_id], dman.get_Y()[model_id]
kde = dman.get_kdes()[model_id]
rng = jax.random.PRNGKey(0)
subsample = du.optimal_density_subsample(x, kde, rng, quantile_threshold=0.1)
x, y = x[subsample], y[subsample][:, output_pos]

fig, ax = du.mkfig(1, 1)
tree = cKDTree(x)

slice_at = np.linspace(0.2, 0.8, 15)
which_slice = 1

res = 200
xx = jnp.linspace(0, 1, res).reshape(-1, 1)
radius = 0.125
min_points = 200
import matplotlib as mpl
color_map = mpl.cm.get_cmap('viridis')
for i, sl in enumerate(slice_at):
    color = color_map(i / len(slice_at))
    # xquery = jnp.concatenate([xx, jnp.full((res, 1), sl)], axis=1)
    # depending on which_slice we append or prepend
    if which_slice == 0:
        xquery = jnp.concatenate([xx, jnp.full((res, 1), sl)], axis=1)
    else:
        xquery = jnp.concatenate([jnp.full((res, 1), sl), xx], axis=1)
    z, p = du.get_knn_mean(xquery, y, tree, min_points=min_points, radius=radius, knn=10000)
    rescaled_sl = unscaler(np.array([sl]))[0]
    ax.plot(xx, z, label=f'{input_names[which_slice]}={rescaled_sl:.0e}', color=color)
    # zql, p = du.get_knn_quantile(
        # xquery, y, qu=0.2, tree=tree, min_points=min_points, radius=radius, knn=5000
    # )
    # zqh, p = du.get_knn_quantile(
        # xquery, y, qu=0.8, tree=tree, min_points=min_points, radius=radius, knn=5000
    # )
    # ax.fill_between(xx.flatten(), zql.flatten(), zqh.flatten(), alpha=0.2)


ax.legend()
ax.set_xlabel(input_names[1 - which_slice])
ax.set_ylabel(output_name)
