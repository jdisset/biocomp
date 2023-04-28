### {{{                          --     imports     --
import matplotlib

import biocomp as bc
from biocomp import datautils as du
from functools import partial
from datetime import datetime
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import jax
from jax import jit, vmap, value_and_grad
import jax.numpy as jnp

from biocomp import utils as ut
import scriptutils as su
import biocomp.datautils as du
from biocomp import train
from biocomp import compute as cmp

# matplotlib.use('agg')

##────────────────────────────────────────────────────────────────────────────}}}

xpname = 'csy4'

MAX_UORF = 80
TRAINING_SETS = {
    '1corner': [(0, 0)],
    '2corners_recog': [(0, 0), (0, MAX_UORF)],
    '2corners_ern': [(0, 0), (MAX_UORF, 0)],
    '2corners_diag': [(0, 0), (MAX_UORF, MAX_UORF)],
    '3corners': [(0, 0), (0, MAX_UORF), (MAX_UORF, 0)],
    '4corners': [(0, 0), (0, MAX_UORF), (MAX_UORF, 0), (MAX_UORF, MAX_UORF)],
    'all': None,
}

### {{{                      --     loading matrix xp     --
training_config = train.DEFAULT_TRAINING_CONFIG
compute_config = cmp.DEFAULT_COMPUTE_CONFIG

XP = {'case': '2023-02-16_Matrix', 'csy4': '2023-03-26_MatrixCsy4'}
with ut.timer(f'Loading data and building networks for {XP[xpname]}'):
    lib = su.load_lib()
    matrix_xp = su.load_xp(XP[xpname], lib, data_path='./data/calibrated_data')
    dman_full = du.DataManager.from_xps([matrix_xp], training_config, inverse='all')

key = jax.random.PRNGKey(0)
stack = dman_full.build_compute_stack(compute_config)
with ut.timer('Stack initialization'):
    base_params = stack.init(key)

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                      --     quantify uorfs     --


def get_uorf_value(param):
    if 'tl_rate' in param:
        u = param['tl_rate'][0].split('_')[0]
        try:
            v = int(u[:-1]) * 10
        except ValueError:
            v = 0
        if u[-1] == 'w':
            v = v - 5
        return v
    else:
        return 0


def get_uorf_values(network):
    cdg = network.central_dogma_graph
    ERN_inputs = network.compute_graph[network.compute_graph['type'] == 'sequestron_ERN'][
        'cdg_input'
    ].values[0]
    cdgin = cdg.loc[ERN_inputs]
    ern_side = cdg.loc[cdgin.iloc[0].predecessor[0]]
    recog_side = cdgin.iloc[1]
    values = (get_uorf_value(ern_side.params), get_uorf_value(recog_side.params))
    return values


def get_max_uorf(network):
    cdg = network.central_dogma_graph
    params = cdg.params.values
    uorfs = [get_uorf_value(p) for p in params]
    return max(uorfs)


uorf_dict = {}
for i, n in enumerate(dman_full.get_networks()):
    has_ERN_node = n.compute_graph['type'] == 'sequestron_ERN'
    if has_ERN_node.any():
        uorf_dict[get_uorf_values(n)] = i
    # else:
    # uorf_dict[(get_max_uorf(n),)] = i

uorf_dict
single_uorfs = [i for i in range(len(dman_full.get_networks())) if i not in uorf_dict.values()]

TRAINING_SETS['all'] = list(uorf_dict.keys())

# single_names = [n.name for i, n in enumerate(dman_full.get_networks()) if i in single_uorfs]


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                        --     plot matrix     --
def plot_matrix(f, uorf_dict=None, square=None):

    from matplotlib import patches

    n_unique_ern_side = len(np.unique([v[0] for v in uorf_dict.keys()]))
    n_unique_recog_side = len(np.unique([v[1] for v in uorf_dict.keys()]))

    ER_to_rowcol = {
        (E, R): (r, c)
        for c, E in enumerate(np.unique([v[0] for v in uorf_dict.keys()]))
        for r, R in enumerate(np.unique([v[1] for v in uorf_dict.keys()]))
    }

    fig, axes = du.mkfig(n_unique_ern_side, n_unique_recog_side, (2, 2))
    for (E, R), m in tqdm(list(uorf_dict.items())[:]):
        r, c = ER_to_rowcol[(E, R)]
        ax = axes[r, c]
        title = ''
        contours = np.linspace(0, 1, 7)
        f(
            m,
            ax=ax,
            radius=0.15,
            knn=400,
            min_points=20,
            colorbar=False,
            title=title,
            contours=contours,
            res=100,
        )
        # remove left ticks except if j == 0
        if r != n_unique_recog_side - 1:
            ax.set_xticks([])
            ax.set_xlabel('')
        if c != 0:
            ax.set_yticks([])
            ax.set_ylabel('')

        # add E and R labels for the whole grid
        if r == 0:  # first row
            ax.text(0.5, 1.1, f'E: {E/10:.1f}x', transform=ax.transAxes, ha='center', va='bottom')

        # last column, write to the right
        if c == n_unique_ern_side - 1:
            ax.text(
                1.1,
                0.5,
                f'R: {R/10:.1f}x',
                transform=ax.transAxes,
                ha='left',
                va='center',
                rotation=90,
            )

    if square:
        for E, R in square:
            r, c = ER_to_rowcol[(E, R)]
            ax = axes[r, c]
            # add a star symbol to the top left corner
            # to show it was used for training
            ax.text(
                0.05,
                0.95,
                '*',
                transform=ax.transAxes,
                ha='left',
                va='top',
                fontsize=20,
                color='red',
            )

    fig.tight_layout()
    return fig


##────────────────────────────────────────────────────────────────────────────}}})

import wandb
import pickle

wandb.login()


REPLOT = False

# ground truth

# savepath = Path(f'~/Desktop/predictions/lvl1_matrix/{xpname}').expanduser()
# savepath.mkdir(parents=True, exist_ok=True)
# figpath = savepath / 'ground_truth.png'

# if not figpath.exists() or REPLOT:
# print('plotting ground truth')
# f = partial(du.network_plot, dman_full)
# fig = plot_matrix(f, uorf_dict)
# fig.suptitle(f'Ground truth for {xpname}', y=1.02)
# fig.savefig(figpath, dpi=200)


##

for tset in TRAINING_SETS.keys():
    print(f'PLOTTING PREDS FOR {tset}')

    try:
        project_name = f'matrix_{xpname}_{tset}'
        entity = 'jdisset'

        api = wandb.Api()
        project_path = f"{entity}/{project_name}" if entity else project_name
        runs = api.runs(project_path)
        bestrun = None
        bestloss = np.inf
        for run in runs:
            if 'loss' in run.summary and run.summary['loss'] is not None:
                if run.summary['loss'] < bestloss:
                    bestrun = run
                    bestloss = run.summary['loss']
        print(f'Best run is {bestrun.name} with loss {bestloss}')

        tmp_dir = Path(f'/tmp/{project_name}')
        param_file = bestrun.file('latest_params.pkl').download(replace=True, root=tmp_dir)
        with open(param_file.name, 'rb') as f:
            trained_params = pickle.load(f)

        best_params = stack.use_shared_params(base_params, trained_params)

        train_subset = single_uorfs + [uorf_dict[i] for i in TRAINING_SETS[tset]]

        savepath.mkdir(parents=True, exist_ok=True)
        figpath = savepath / f'predictions_{tset}.png'
        if not figpath.exists() or REPLOT:
            print('plotting predictions for', tset, '...')
            f = partial(du.plot_model_at_x, best_params, dman_full)
            fig = plot_matrix(f, uorf_dict, square=TRAINING_SETS[tset])
            fig.suptitle(f'Predictions for {xpname} with {tset}', y=1.02)
            fig.savefig(figpath, dpi=200)

    except Exception as e:
        print(f'ERROR: {e}')


##
tset = '3corners'
project_name = f'matrix_{xpname}_{tset}'
entity = 'jdisset'

api = wandb.Api()
project_path = f"{entity}/{project_name}" if entity else project_name
runs = api.runs(project_path)
bestrun = None
bestloss = np.inf
for run in runs:
    if 'loss' in run.summary and run.summary['loss'] is not None:
        if run.summary['loss'] < bestloss:
            bestrun = run
            bestloss = run.summary['loss']
print(f'Best run is {bestrun.name} with loss {bestloss}')

tmp_dir = Path(f'/tmp/{project_name}')
param_file = bestrun.file('latest_params.pkl').download(replace=True, root=tmp_dir)
with open(param_file.name, 'rb') as f:
    trained_params = pickle.load(f)

best_params = stack.use_shared_params(base_params, trained_params)

##
mid = uorf_dict[(80,80)]
contours = np.linspace(0, 1, 7)

with ut.timer('plot 1'):
    fig, ax = du.mkfig(1, 1)
    du.plot_model_at_x(
        best_params, dman_full, mid, ax=ax, radius=0.15, knn=400, min_points=20, colorbar=False, res=50, contours=contours
    )

##
mid = uorf_dict[(80,80)]
mid = uorf_dict[(0,80)]
# mid = uorf_dict[(0,0)]
with ut.timer('ground truth'):
    fig, ax = du.mkfig(1, 1, (15,15))
    du.network_plot(dman_full, mid, ax=ax,contours=contours, method='scatter', kde=False, size=10)

savepath = Path(f'~/Desktop/predictions/lvl1_matrix').expanduser()
savepath.mkdir(parents=True, exist_ok=True)
fig.savefig(savepath/f'ern_{mid}_nokde.pdf', dpi=200)

with ut.timer('ground truth'):
    fig, ax = du.mkfig(1, 1, (15,15))
    du.network_plot(dman_full, mid, ax=ax,contours=contours, method='scatter', size=10)

savepath = Path(f'~/Desktop/predictions/lvl1_matrix').expanduser()
savepath.mkdir(parents=True, exist_ok=True)
fig.savefig(savepath/f'ern_{mid}_kde.pdf', dpi=200)


# It's the clls with 0 ERN plasmids!!!!!

# su.plot_networks([dman_full.get_networks()[mid]])
##
with ut.timer('plot 2'):
    fig, ax = du.mkfig(1, 1)
    du.eval_model_grid( best_params, dman_full, mid, ax=ax, n_repeats=100, res=100, contours=contours)


