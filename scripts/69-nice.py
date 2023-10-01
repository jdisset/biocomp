### {{{                          --     imports     --
from biocomp import utils as ut
import scriptutils as su
import biocomp.datautils as du
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

### {{{                   --     plot translation node     --
import matplotlib as mpl

from labellines import labelLine, labelLines
def plot_node(node_name, qid, ax):
    tl = compute_config.get_impl(node_name)

    L = tl(input_shapes=[(1,)], n_outputs=1, stack=None, layer_id=0)

    class FakeNode(cmp.VirtualNode):
        def get_compute_node(self, _):
            return None

        def get_inverse_node(self, _):
            return None

        def get_layer_and_local_id(self, _):
            return 0, 0


    key = jax.random.PRNGKey(0)

    p = pm.ParameterTree()
    L.prepare(p, [FakeNode()], key)
    p.tag('local', 'local')
    local, _ = p.filter_by_tag('local')

    qname = None
    qnames = []
    if node_name in ('translation', 'transcription', 'inv_transcription', 'inv_translation'):
        qmaskleaf = None
        for l, v in local.data.iter_leaves():
            if str(l).endswith('quantization_mask'):
                qmaskleaf = l
                break
        qnames = compute_config.config['functions'][node_name]['parameters']['quantization_names']
        base_mask = np.zeros((len(qnames),), dtype=np.bool).reshape(1, 1, -1)
        base_mask[:, :, qid] = True
        local[qmaskleaf] = base_mask
        qname = qnames[qid]

    pmerged = pm.ParameterTree.merge(shared_parameters, local)

    @jax.jit
    def vapply(xvals, qs, params):
        f = lambda x, q: L.apply(x, quantiles=q, node_id=0, params=params, key=key)
        return jax.vmap(f)(xvals, qs)


    n_evals = 500
    x = np.linspace(0, 1, n_evals).reshape(-1, 1)
    medianq = np.ones_like(x) * 0.5

    ymedian = vapply(x, medianq, pmerged).flatten()

    n_random = 20000
    randomx = np.random.uniform(0, 1, n_random).reshape(-1, 1)
    randomq = np.random.uniform(0, 1, n_random).reshape(-1, 1)
    yrandom = vapply(randomx, randomq, pmerged).flatten()

    # from qid
    color = mpl.cm.get_cmap('YlGnBu_r')(qid / 12)

    ax.scatter(randomx, yrandom, s=2, c=color, alpha=0.05, linewidth=0)
    ax.plot(x, ymedian, label=qname if qname is not None else '', c=color, ls='--', lw=2)

# fig, axes = du.mkfig(1, 2, (8, 8), dpi=300)
fig = plt.figure(constrained_layout=False, figsize=(10, 10), dpi=300)
gs = fig.add_gridspec(100, 100)
ax = fig.add_subplot(gs[:, :])

node_name = 'translation'

plot_node('translation', 0, ax)
plot_node('transcription', 0, ax)
# plot_node('translation', 1, ax)
# plot_node('translation', 2, ax)
# plot_node('translation', 3, ax)
# plot_node('translation', 4, ax)
# plot_node('translation', 5, ax)
# plot_node('translation', 6, ax)
# plot_node('translation', 7, ax)
# plot_node('translation', 8, ax)

ax.set_xlabel('mRNA (A.U.)')
ax.set_ylabel('Protein (A.U.)')
ax.set_title('Translation Node (post training)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_aspect('equal', 'box')
labelLines(ax.get_lines(), zorder=2.5)

ax.text(0.1, 0.8, 'Steady-state concentrations\n(median line + random samples)', transform=ax.transAxes, ha='left', va='bottom', fontsize=10)
# ax.legend()
ax = fig.add_subplot(gs[50:85, 40:95])
names = compute_config.config['functions'][node_name]['parameters']['quantization_names'][:9]
values = shared_parameters['shared/quantization/tl_rate_values'].squeeze()[:len(names)]
values
# color = mpl.cm.get_cmap('YlGnBu_r')(qid / 11)
# bar plot of values using the same colormap
ax.bar(np.arange(len(names))*1.1, values[::-1]+0.2, color=mpl.cm.get_cmap('YlGnBu_r')(np.linspace(8/12, 0, len(names))), bottom=-0.2)

# ax.set_xlabel('quantization name')
# rotate xticks
ax.set_xticks(np.arange(len(names))*1.1)
ax.set_xticklabels(names[::-1], rotation=45, ha='center')
# hide ticks
ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
# remove frame
ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# transparent background
ax.patch.set_alpha(0)
# y on the right
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
# ax.set_ylabel('translation rate')
# instead of a title, annotate with text
ax.text(0.12, 0.4, 'Learned quantization values\nfor the translation rates\ngiven a 5\' part:', transform=ax.transAxes, ha='left', va='bottom', fontsize=10)
# ax.set_title('learned quantization values')

# fig.tight_layout()
# fig.savefig(Path('~/Desktop/translation_node_after_training.png').expanduser(), dpi=300)


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

XP = {'case': '2023-02-16_Matrix', 'csy4': '2023-03-26_MatrixCsy4'}
with ut.timer(f'Loading data and building networks for {XP[xpname]}'):
    lib = su.load_lib()
    matrix_xp = su.load_xp(XP[xpname], lib, data_path='./data/calibrated_data')
    dman_full = du.DataManager.from_xps([matrix_xp], training_config, inverse='all')

##
key = jax.random.PRNGKey(0)
stack = dman_full.build_compute_stack(compute_config)
stack

def init_stack(stack, rng):
    local_params, _ = stack.init(rng).filter_by_tag('local')
    local_params.data.check()
    full_params = ParameterTree.merge(local_params, shared_parameters)
    return full_params

full_params = init_stack(stack, key)

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


