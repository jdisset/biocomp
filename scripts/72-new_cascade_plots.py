### {{{                          --     imports     --
from biocomp import utils as ut
import scriptutils as su
import biocomp.datautils as du
import biocomp.plotutils as pu
import time
import biocomp.train as train
import biocomp.compute as cmp
import biocomp.parameters as pm
import biocomp as bc
from biocomp.parameters import ParameterTree
from jax.tree_util import Partial as partial
import jax.tree_util as jtu
from pathlib import Path
import jax.numpy as jnp
from copy import deepcopy
import optax
from tqdm import tqdm
import numpy as np
import jax
from jax import jit, grad, vmap, random, value_and_grad
from jax import numpy as jnp
from matplotlib import pyplot as plt

##────────────────────────────────────────────────────────────────────────────}}}
### {{{                      --     load parameters     --
training_archive = du.load('../__results/training_archives/20230923_fulltrain_v0.pkl')
shared_parameters = training_archive['parameters']
compute_config = training_archive['compute_config']
training_config = training_archive['training_config']
compute_config.set_impl('bias', bc.nodes.bias)
##────────────────────────────────────────────────────────────────────────────}}}
XP = {'BPattempt': '2023-10-01_Cascades_CCv4'}
xpname = 'BPattempt'
with ut.timer(f'Loading data and building networks for {XP[xpname]}'):
    lib = su.load_lib()
    bp_xp = su.load_xp(
        XP[xpname],
        lib,
        data_path='./data/calibrated_data_v3',
        recipe_path=su.DEFAULT_DATA_PATH / 'Experiments' / XP[xpname] / 'recipes',
    )
    dman_full = du.DataManager.from_xps([bp_xp], training_config, inverse='all')

##

# su.plot_networks([dman_full.get_networks()[0]], W=2000, H=4000)
n = dman_full.get_networks()[0]
n.get_inverted_input_proteins()


from matplotlib import pyplot as plt

savedir = Path('~/Desktop/newcascades/').expanduser()
savedir.mkdir(exist_ok=True, parents=True)

##
# plot rescaler
x = dman_full.get_raw_X()[0]
x.shape
xprime = dman_full.rescale([x])[0]
xback = dman_full.unscale([xprime])[0]
np.allclose(x, xback)

##

for i in range(len(dman_full.get_networks()))[:1]:
    # fig, axes = du.mkfig(1,4, (4,4), dpi=200)
    # fig, ax = du.mkfig(1,1, (10,10), dpi=200)
    # if not isinstance(axes, list):
    # axes = [axes]
    # du.network_plot(dman_full, i, axes=axes, input_order=[0,1,2], slices=[[0.1,0.3,0.5],[0.45]], method='smooth_lines', xmax=0.6)
    # fig.tight_layout()

    fig, axes = pu.mkfig(1, 2, (5, 5), dpi=200)

    fig, ax = pu.mkfig(1, 1, (7, 5), dpi=200)
    # pu.network_plot(
        # dman_full,
        # i,
        # # axes=axes,
        # ax=ax,
        # input_order=[0, 1, 2],
        # slices=[[0.1, 0.3, 0.5], [0.1, 0.5]],
        # method='smooth_lines',
        # min_points=50,
        # radius=0.125,
        # knn_method='mean',
    # )

    pu.network_plot(dman_full, i, ax=ax, input_order=[0,1,2], method='smooth')
    fig.tight_layout()

    # fig.savefig(savedir / f'cascade_{i}.pdf')
    # plt.show()
    # plt.close(fig)
    # print(f'Saved network {i}')
