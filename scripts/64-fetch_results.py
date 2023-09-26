from biocomp import utils as ut
import scriptutils as su
import biocomp.datautils as du
import biocomp.train as train
import biocomp.compute as cmp
import biocomp.parameters as pm
from pathlib import Path
import jax.numpy as jnp
import numpy as np
import jax
from copy import deepcopy


### {{{                  --     fetch best training run     --

project_name = 'fulltrain_v0'
runs, losses = du.retrieve_wandb_results(project_name, per_page=1000)
best_run_id = du.losses_plot(losses[50:150], runs=runs[50:150])
best_run_id
runs[best_run_id].name
trained_params, compute_config, training_config, local_params = du.get_wandb_trained_params(runs[50+best_run_id])

trained_params.set_read_only(True)

##
training_archive = {'parameters': trained_params, 'compute_config': compute_config, 'training_config': training_config}
import time
timestr = time.strftime("%Y%m%d")
savepath = Path(f'../__results/training_archives/{timestr}_{project_name}.pkl')
savepath.parent.mkdir(parents=True, exist_ok=True)
du.save(training_archive, savepath)

tarxv = du.load(savepath)
assert tarxv['parameters'] == training_archive['parameters']
assert tarxv['training_config'] == training_archive['training_config']
assert tarxv['compute_config'].config == training_archive['compute_config'].config

##────────────────────────────────────────────────────────────────────────────}}}

trained_params['shared/quantization/tl_rate_values']
compute_config.config['functions'][node_name]['parameters']['quantization_names']
node_name = 'translation'
qid = 0

ptree = str(local_params.data)
# save
with open(f'./local_params_{node_name}_{qid}.txt', 'w') as f:
    f.write(ptree)

def plot_node(node_name, qid, ax):
    tl = compute_config.get_impl(node_name)

    prepare, apply, _ = tl(input_shapes=[(1,)], n_outputs=1, stack=None, layer_id=0)


    class FakeNode(cmp.VirtualNode):
        def get_compute_node(self, _):
            return None

        def get_inverse_node(self, _):
            return None

        def get_layer_and_local_id(self, _):
            return 0, 0


    key = jax.random.PRNGKey(0)

    p = pm.ParameterTree()
    prepare(p, [FakeNode()], key)
    p.tag('local', 'local')
    local, _ = p.filter_by_tag('local')

    qname = None
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


    pmerged = pm.ParameterTree.merge(trained_params, local)


    @jax.jit
    def vapply(xvals, qs, params):
        f = lambda x, q: apply(x, quantiles=q, node_id=0, params=params, key=key)
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
    import matplotlib as mpl
    color = mpl.cm.get_cmap('tab10')(qid / 10)

    ax.scatter(randomx, yrandom, s=4, c=color, alpha=0.05, linewidth=0)
    ax.plot(x, ymedian, label=qname if qname is not None else '', c=color, ls='--', lw=2)

fig, ax = du.mkfig(1, 1, (10, 10), dpi=300)

plot_node('translation', 0, ax)
plot_node('translation', 1, ax)
plot_node('translation', 2, ax)
# plot_node('translation', 3, ax)
plot_node('translation', 4, ax)
# plot_node('translation', 5, ax)
plot_node('translation', 6, ax)
# plot_node('translation', 7, ax)
plot_node('translation', 8, ax)

ax.set_xlabel('mRNA')
ax.set_ylabel('PRT')
ax.set_title(node_name)
ax.legend()
ax.set_aspect('equal', 'box')
ax.grid(linestyle='--', linewidth=0.5)


### {{{                          --     archive     --


class TracerParameterTree(pm.ParameterTree):
    def __init__(self, paramtree):
        # copy constructor
        self.__dict__ = deepcopy(paramtree.__dict__)
        self.success_trace = []
        self.fail_trace = []

    def __getitem__(self, key):
        try:
            r = pm.ParameterTree.__getitem__(self, key)
        except KeyError:
            r = None
            self.fail_trace.append(key)
        else:
            self.success_trace.append(key)
        return r


traced_params.success_trace
traced_params.fail_trace

##────────────────────────────────────────────────────────────────────────────}}}

