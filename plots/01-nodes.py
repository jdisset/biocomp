### {{{                          --     imports     --
import sys

sys.path.append('../scripts')

from labellines import labelLines, labelLine
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
from matplotlib import pyplot as plt
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

import cProfile


class profiler:
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self.profiler = cProfile.Profile()
        self.profiler.enable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.disable()
        self.profiler.dump_stats(self.filename)


from common import plotdir, onedrive, cmaps

##────────────────────────────────────────────────────────────────────────────}}}
### {{{                      --     load parameters     --
training_archive = du.load('../__results/training_archives/20230923_fulltrain_v0.pkl')
shared_parameters = training_archive['parameters']
compute_config = training_archive['compute_config']
training_config = training_archive['training_config']
compute_config.set_impl('bias', bc.nodes.bias)
empty_dman = pu.DataManager([], [], [], training_config)
rescaler = pu.DataRescaler.from_data_manager(empty_dman)
##────────────────────────────────────────────────────────────────────────────}}}

fnames = compute_config.config['functions']
fnames.keys()

### {{{                   --     base functions --


class FakeNode(cmp.VirtualNode):
    def __init__(self, attributes=None):
        self.attributes = attributes

    def get_compute_node(self, colname):
        if self.attributes is None:
            return None
        return self.attributes.get(colname, None)

    def get_inverse_node(self, _):
        return None

    def get_layer_and_local_id(self, _):
        return 0, 0


def node_func(
    node_name,
    shared_parameters,
    compute_config,
    quantized_param_id=0,
    qnamedict=None,
):

    key = jax.random.PRNGKey(0)
    impl = compute_config.get_impl(node_name)
    layer_instance = impl(input_shapes=[(1,)], n_outputs=1, stack=None, layer_id=0)
    p = pm.ParameterTree()
    layer_instance.prepare(p, [FakeNode()], key)
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
        base_mask[:, :, quantized_param_id] = True
        local[qmaskleaf] = base_mask
        qname = qnames[quantized_param_id]
        if qnamedict is not None:
            qname = qnamedict.get(qname, qname)

    pmerged = pm.ParameterTree.merge(shared_parameters, local)
    @jax.jit
    def vapply(xvals, qs):
        f = lambda x, q: layer_instance.apply(x, quantiles=q, node_id=0, params=pmerged, key=key)
        return jax.vmap(f)(xvals, qs)

    return vapply, qname


# randomx = np.random.uniform(0, 1, n_random_evals).reshape(-1, 1)
# randomq = np.random.uniform(0, 1, n_random_evals).reshape(-1, 1)
# yrandom = f(randomx, randomq).flatten()
# ax.scatter(randomx, yrandom, s=2, c=color, alpha=0.05, linewidth=0)

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                        --     transcription    --
cmap = cmaps['blues']

basedir = plotdir / 'Nodes'
basedir.mkdir(exist_ok=True, parents=True)

node_name = 'transcription'
savedir = basedir / node_name

savedir.mkdir(exist_ok=True, parents=True)

fig, ax = pu.mkfig(1, 1, (3, 3), dpi=300)

uorf_dict = {'00_empty_tc': 'no uORF'}

xlims = (0, 1)
resolution = 500

f, _ = node_func(
    node_name,
    shared_parameters,
    compute_config,
)

x = np.linspace(-0.01, 1.01, resolution).reshape(-1, 1)
medianq = np.ones_like(x) * 0.5
ymedian = f(x, medianq).flatten()

color = cmap(1 - (7 / 20))
ax.plot(x, ymedian, c=color, lw=1)

ax.set_xticks(xlims)
ax.set_yticks(xlims)

ax.set_xlabel('DNA (A.U., log-like latent space)')
ax.set_ylabel('mRNA (A.U., log-like latent space)')
ax.set_title(f'Transcription node')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_aspect('equal', 'box')

fig.tight_layout()
fig.savefig(savedir / f'{node_name}_node.pdf', dpi=300)


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                        --     translation     --
cmap = cmaps['blues']

basedir = plotdir / 'Nodes'
basedir.mkdir(exist_ok=True, parents=True)

node_name = 'translation'
savedir = basedir / node_name

savedir.mkdir(exist_ok=True, parents=True)

fig, ax = pu.mkfig(1, 1, (3, 3), dpi=300)

uorf_dict = {'00_empty_tc': 'no uORF'}

xlims = (0, 1)
resolution = 500

for i in range(9):
    f, qname = node_func(
        node_name,
        shared_parameters,
        compute_config,
        quantized_param_id=i,
        qnamedict=uorf_dict,
    )

    x = np.linspace(*xlims, resolution).reshape(-1, 1)
    medianq = np.ones_like(x) * 0.5
    ymedian = f(x, medianq).flatten()

    color = cmap(1 - ((i + 7) / 20))
    ax.plot(x, ymedian, label=qname.replace('_', ' ') if qname is not None else '', c=color, lw=1)


# show line labels
ax.legend(loc='lower right', bbox_to_anchor=(1, 0.05), frameon=False, fontsize=5)

# only tick at 0 and 1, tick inside
ax.set_xticks(xlims)
ax.set_yticks(xlims)
# ax.tick_params(axis='both', which='both', length=5, pad=2, direction='in')

ax.set_xlabel('mRNA (A.U., log-like latent space)')
ax.set_ylabel('Protein (A.U., log-like latent space)')
ax.set_title(f'Translation node')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_aspect('equal', 'box')
# labelLines(ax.get_lines(), zorder=2.5)

fig.tight_layout()
fig.savefig(savedir / 'translation_node.pdf', dpi=300)


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                        --     output     --
cmap = cmaps['blues']

basedir = plotdir / 'Nodes'
basedir.mkdir(exist_ok=True, parents=True)

node_name = 'output'
savedir = basedir / node_name

savedir.mkdir(exist_ok=True, parents=True)

fig, ax = pu.mkfig(1, 1, (3, 3), dpi=300)

xlims = (0, 1)
resolution = 500

f, _ = node_func(
    node_name,
    shared_parameters,
    compute_config,
)

x = np.linspace(-0.01, 1.01, resolution).reshape(-1, 1)
medianq = np.ones_like(x) * 0.5
ymedian = f(x, medianq).flatten()

color = cmap(1 - (7 / 20))
ax.plot(x, ymedian, c=color, lw=1)

ax.set_xticks(xlims)
ax.set_yticks(xlims)

ax.set_xlabel('Protein (A.U., log-like latent space)')
ax.set_title(f'Output Fluorescence node')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_aspect('equal', 'box')


pu.setup_transformed_yaxis(ax, (rescaler(1e3 - 1), rescaler(1e8 + 1)), rescaler)
ax.set_ylabel('Fluorescence intensity (MEF)')

fig.tight_layout()
fig.savefig(savedir / f'{node_name}_node.pdf', dpi=300)


##────────────────────────────────────────────────────────────────────────────}}}

shared_parameters.data

### {{{                        --     ERN    --

affinities = compute_config.config['functions']['sequestron_ERN']['parameters']['affinity_names']
cmap = cmaps['blues']
basedir = plotdir / 'Nodes'
basedir.mkdir(exist_ok=True, parents=True)
node_name = 'sequestron_ERN'
savedir = basedir / node_name
savedir.mkdir(exist_ok=True, parents=True)


key = jax.random.PRNGKey(0)
impl = compute_config.get_impl(node_name)
layer_instance = impl(input_shapes=[(1,), (1,)], n_outputs=1, stack=None, layer_id=0)

xlims = (0, 1)
resolution = 500
xy = pu.make_xy_grid(*xlims, xres=resolution)
fig, axes = pu.mkfig(1, len(affinities), (3, 3), dpi=300)

for i, ax in enumerate(axes):
    ernname = affinities[i].split('::')[-1].split('#')[0]
    seqname = affinities[i]
    p = pm.ParameterTree()
    layer_instance.prepare(
        p, [FakeNode({'extra': {'seq_name': seqname, 'quantile_variable_id': 0}})], key
    )

    p.tag('local', 'local')
    local, _ = p.filter_by_tag('local')
    pmerged = pm.ParameterTree.merge(shared_parameters, local)

    @jax.jit
    def f(xvals, qs):
        l = lambda x, q: layer_instance.apply(x[1],x[0], quantiles=q, node_id=0, params=pmerged, key=key)
        return jax.vmap(l)(xvals, qs)

    output_values = f(xy, np.ones((xy.shape[0], 1)) * 0.5).flatten()

    pu.heatmap_new(ax, xy, output_values, rescaler, cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(xlims, minor=False)
    ax.set_xticks([], minor=True)
    ax.set_xticklabels([0, 1])
    ax.set_xlabel('mRNA with target site (A.U.)')

    ax.set_yticks(xlims)
    ax.set_yticks([], minor=True)
    ax.set_yticklabels([0, 1])
    ax.set_ylabel('ERN protein (A.U.)')

    ax.set_title(f'{ernname}')

    # find the colorbar, remove it, and make a new one
    cbar = ax.get_images()[0].colorbar
    if i == len(affinities) - 1:
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels([0, 1])
        cbar.set_ticks([], minor=True)
        cbar.set_label('Surviving mRNA (A.U.)')
        # cbar.ax.yaxis.set_label_position('left')
    else:
        cbar.remove()

fig.tight_layout()

fig.savefig(savedir / 'ern_node.pdf', dpi=300)


##────────────────────────────────────────────────────────────────────────────}}}

##────────────────────────────────────────────────────────────────────────────}}}


### {{{                    --     dna to mef path     --

cmap = cmaps['blues']

basedir = plotdir / 'Nodes'
basedir.mkdir(exist_ok=True, parents=True)
savedir = basedir / 'dna_to_mef'

savedir.mkdir(exist_ok=True, parents=True)

fig, ax = pu.mkfig(1, 1, (3, 3), dpi=300)

uorf_dict = {'00_empty_tc': 'no uORF'}

xlims = (0, 1)
resolution = 500


# for i in range(1):
for i in [3]:

    tx, _ = node_func(
        'transcription',
        shared_parameters,
        compute_config,
    )

    tl, qname = node_func(
        'translation',
        shared_parameters,
        compute_config,
        quantized_param_id=i,
        qnamedict=uorf_dict,
    )

    out, _ = node_func(
        'output',
        shared_parameters,
        compute_config,
    )

    f = lambda x, q: out(tl(tx(x, q), q), q)

    x = np.linspace(*xlims, resolution).reshape(-1, 1)
    medianq = np.ones_like(x) * 0.5
    ymedian = f(x, medianq).flatten()

    color = cmap(1 - ((i + 7) / 20))
    ax.plot(x, ymedian, label=qname.replace('_', ' ') if qname is not None else '', c=color, lw=1)

    n_random_evals = 50000
    randomx = np.random.uniform(0, 1, n_random_evals).reshape(-1, 1)
    randomq = np.random.uniform(0, 1, n_random_evals).reshape(-1, 1)
    yrandom = f(randomx, randomq).flatten()
    ax.scatter(randomx, yrandom, s=2, c=color, alpha=0.05, linewidth=0)


# show line labels
ax.legend(loc='lower right', bbox_to_anchor=(1, 0.05), frameon=False, fontsize=5)

# only tick at 0 and 1, tick inside
ax.set_xticks(xlims)
ax.set_yticks(xlims)
# ax.tick_params(axis='both', which='both', length=5, pad=2, direction='in')

ax.set_xlabel('DNA (A.U., log-like latent space)')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_aspect('equal', 'box')
# labelLines(ax.get_lines(), zorder=2.5)
pu.setup_transformed_yaxis(ax, (rescaler(1e3 - 1), rescaler(1e8 + 1)), rescaler)
ax.set_ylabel('Fluorescence intensity (MEF)')

fig.tight_layout()
fig.savefig(savedir / 'dna_to_mef.pdf', dpi=300)


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                    --     rna to mef path     --

cmap = cmaps['blues']

basedir = plotdir / 'Nodes'
basedir.mkdir(exist_ok=True, parents=True)
savedir = basedir / 'rna_to_mef'

savedir.mkdir(exist_ok=True, parents=True)

fig, ax = pu.mkfig(1, 1, (3, 3), dpi=300)

uorf_dict = {'00_empty_tc': 'no uORF'}

xlims = (0, 1)
resolution = 500

jax.clear_caches()

for i in range(9):

    tx, _ = node_func(
        'transcription',
        shared_parameters,
        compute_config,
    )

    tl, qname = node_func(
        'translation',
        shared_parameters,
        compute_config,
        quantized_param_id=i,
        qnamedict=uorf_dict,
    )

    out, _ = node_func(
        'output',
        shared_parameters,
        compute_config,
    )

    f = lambda x, q: out(tl(x, q), q)

    x = np.linspace(*xlims, resolution).reshape(-1, 1)
    medianq = np.ones_like(x) * 0.5
    ymedian = f(x, medianq).flatten()

    color = cmap(1 - ((i + 7) / 20))
    # color = 'k'
    ax.plot(x, ymedian, label=qname.replace('_', ' ') if qname is not None else '', c=color, lw=1)


# show line labels
ax.legend(loc='lower right', bbox_to_anchor=(1, 0.05), frameon=False, fontsize=5)

# only tick at 0 and 1, tick inside
ax.set_xticks(xlims)
ax.set_yticks(xlims)
# ax.tick_params(axis='both', which='both', length=5, pad=2, direction='in')

ax.set_xlabel('RNA (A.U., log-like latent space)')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_aspect('equal', 'box')
# labelLines(ax.get_lines(), zorder=2.5)
pu.setup_transformed_yaxis(ax, (rescaler(1e3 - 1), rescaler(1e8 + 1)), rescaler)
ax.set_ylabel('Fluorescence intensity (MEF)')

fig.tight_layout()
fig.savefig(savedir / 'rna_to_mef.pdf', dpi=300)


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                    --     mef to dna path     --

cmap = cmaps['blues']

basedir = plotdir / 'Nodes'
basedir.mkdir(exist_ok=True, parents=True)
savedir = basedir / 'mef_to_dna'

savedir.mkdir(exist_ok=True, parents=True)

fig, ax = pu.mkfig(1, 1, (3, 3), dpi=300)

uorf_dict = {'00_empty_tc': 'no uORF'}

xlims = (0, 1)
resolution = 500


for i in range(9):

    itx, _ = node_func(
        'inv_transcription',
        shared_parameters,
        compute_config,
    )

    itl, qname = node_func(
        'inv_translation',
        shared_parameters,
        compute_config,
        # quantized_param_id=i,
        # qnamedict=uorf_dict,
    )

    f = lambda x, q: itx(itl(x, q), q)

    x = np.linspace(*xlims, resolution).reshape(-1, 1)
    medianq = np.ones_like(x) * 0.5
    ymedian = f(x, medianq).flatten()

    color = cmap(1 - ((i + 7) / 20))
    ax.plot(x, ymedian, label=qname.replace('_', ' ') if qname is not None else '', c=color, lw=1)


# show line labels
ax.legend(loc='lower right', bbox_to_anchor=(1, 0.05), frameon=False, fontsize=5)

# only tick at 0 and 1, tick inside
ax.set_xticks(xlims)
ax.set_yticks(xlims)
# ax.tick_params(axis='both', which='both', length=5, pad=2, direction='in')

ax.set_xlabel('DNA (A.U., log-like latent space)')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_aspect('equal', 'box')
# labelLines(ax.get_lines(), zorder=2.5)
pu.setup_transformed_yaxis(ax, (rescaler(1e3 - 1), rescaler(1e8 + 1)), rescaler)
ax.set_ylabel('Fluorescence intensity (MEF)')

fig.tight_layout()
fig.savefig(savedir / 'dna_to_mef.pdf', dpi=300)


##────────────────────────────────────────────────────────────────────────────}}}

