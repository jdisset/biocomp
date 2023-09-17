from biocomp import utils as ut
import numpy as np
import scriptutils as su
import biocomp.datautils as du
import biocomp.train as train
import biocomp.parameters as pm
import biocomp.nodes as nd
import biocomp.compute as cmp
import jax
import jax.tree_util as jtu
import cProfile

prog = train.TrainingProgram()
prog.parse_args()

ut.set_loglevel('info')

### {{{                      --     loading xp     --

XP = {
    'bt': '2023-04-03_Constraints_Pgu_Bleedthrough',
    'cascades': '2023-04-18_Constraints_PguCascades',
    'csy4matrix': '2023-03-26_MatrixCsy4',
    'casematrix': '2023-02-16_Matrix',
}
# xpnames = ['bt', 'cascades', 'csy4matrix', 'casematrix']
xpnames = XP.keys()


with ut.timer(f'Loading data and building networks for {xpnames}'):
    # 7.6s
    # profiler = cProfile.Profile()
    # profiler.enable()

    lib = su.load_lib()
    loadedxp = {
        xpname: su.load_xp(XP[xpname], lib, data_path='./data/calibrated_data_v2')
        for xpname in xpnames
    }

    dman_full = du.DataManager.from_xps(loadedxp.values(), prog.training_config, inverse='all')
    # profiler.disable()
    # profiler.dump_stats("/tmp/dmanbuild.prof")

# all_networks = dman_full.get_networks()
# net_xp = [n.metadata['from_xp'] for n in all_networks]
# net_name = [n.name for n in all_networks]

##────────────────────────────────────────────────────────────────────────────}}}

### {{{               --     training and validation sets     --

# list net names that have cascade in the name:
inert_nets = {n: i for i, n in enumerate(net_name) if 'inert' in n.lower()}
cascade_nets = {
    n: i for i, n in enumerate(net_name) if 'cascade' in n.lower() and 'inert' not in n.lower()
}

# training set is all networks except the ones in inert or cascade
training_set = [
    i
    for i, _ in enumerate(net_name)
    if i not in inert_nets.values() and i not in cascade_nets.values()
]

validation_set = [
    i for i, _ in enumerate(net_name) if i not in inert_nets.values() and i in cascade_nets.values()
]

n_outputs = [n.get_nb_outputs() for n in all_networks]

validation = dman_full.make_subset(validation_set)
training = dman_full.make_subset(training_set)



# prog.start_training(dman_full.make_subset(training_set), validation)

##────────────────────────────────────────────────────────────────────────────}}}

compute_config = cmp.DEFAULT_COMPUTE_CONFIG
key = jax.random.PRNGKey(0)

with ut.timer('Stack building'):
    # 7.5s -> 5.3s
    stack = training.build_compute_stack(compute_config)
# profiler.disable()
# profiler.dump_stats("/tmp/stackbuild2.prof")

##

# profiler = cProfile.Profile()
# profiler.enable()
with ut.timer('Stack initialization'):
    # 1.4s
    params = stack.init(key)
    # params = stack.init(key)
    # params = stack.init(key)
# profiler.disable()
# profiler.dump_stats("/tmp/stackinit3.prof")

# params
##

from copy import deepcopy
paramsold = deepcopy(params)
params == paramsold
params

l, s = jtu.tree_flatten(params)

reconstructed = jtu.tree_unflatten(s, l)

reconstructed

##

assert(reconstructed == params)
reconstructed.data == params.data

# p = 'local/l1 inverse_tl (101)/tl_rate'
p = 'local/l15 ERN_5p (16)/affinity'
base = 'local/l15 ERN_5p (16)'
params[base]
params.data.get_at(p, follow_ref=False)


p in params.data
params[base] == reconstructed[base]
params.data[base] == reconstructed.data[base]
params[base]
reconstructed[base]
pm.is_equal(params[p],reconstructed[p])
params[p]
reconstructed[p]

np.all(params[p] == reconstructed[p])
pm.is_equal(params[p],reconstructed[p])
pm.is_equal(params.data[p],reconstructed.data[p])


diff = pm.ParameterTree.datadiff(params, reconstructed)
diff


nj, j = params.filter_by_tag('non_jit')
ng, g = params.filter_by_tag('non_grad')
nj

lp, sp = jtu.tree_flatten(params, is_leaf=lambda x: x is None)
lr, sr = jtu.tree_flatten(reconstructed, is_leaf=lambda x: x is None)

lp == lr
str(sp) == str(sr)
sp
sr

pm.flatten_PTree(params.data)


##

# def sorted_flatten(params):
    # keys, values  = zip(*list(params.data.iter_leaves(path_as_str=True)))
    # order = np.argsort(keys)
    # sorted_keys = np.array(keys)[order].tolist()
    # sorted_values = np.array(values, dtype=object)[order].tolist()
    # return sorted_keys, sorted_values

def sorted_flatten(params):
    # without numpy to sort:
    keys, values  = zip(*list(params.data.iter_leaves(path_as_str=False)))
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    sorted_keys = [keys[i] for i in order]
    sorted_values = [values[i] for i in order]
    return sorted_keys, sorted_values

# timeit:
import timeit
t = timeit.timeit(lambda: sorted_flatten(params), number=100)
print(f'{t*10:.3f} ms')

t2 = timeit.timeit(lambda: jtu.tree_flatten(params), number=100)
print(f'{t2*1000:.3f} ms')

##

p = pm.PTree()
p['a'] = np.arange(10).reshape(2,5)
p['b'] = np.arange(12).reshape(4,3)
p['c'] = np.arange(15).reshape(3,5)
p

ref1 = pm.ArrayRef(p)
ref1.push_back('a', 0)
ref1.push_back('c', (1,))
ref1

ref2 = pm.ArrayRef(p)
ref2.push_back('a', (0,1))
ref2.push_back('b', (1,2))
ref2.push_back('c', (1,2))
ref2

p['ref2'] = ref2
p






