from biocomp import utils as ut
import numpy as np
import scriptutils as su
import biocomp.datautils as du
import biocomp.train as train
import biocomp.parameters as pm
import biocomp.nodes as nd
import biocomp.compute as cmp
import jax

prog = train.TrainingProgram()
prog.parse_args()

ut.set_loglevel('info')

### {{{                      --     loading xp     --

XP = {
    # 'bt': '2023-04-03_Constraints_Pgu_Bleedthrough',
    'cascades': '2023-04-18_Constraints_PguCascades',
    # 'csy4matrix': '2023-03-26_MatrixCsy4',
    # 'casematrix': '2023-02-16_Matrix',
}
# xpnames = ['bt', 'cascades', 'csy4matrix', 'casematrix']
xpnames = XP.keys()

with ut.timer(f'Loading data and building networks for {xpnames}'):
    lib = su.load_lib()
    loadedxp = {
        xpname: su.load_xp(XP[xpname], lib, data_path='./data/calibrated_data_v2')
        for xpname in xpnames
    }
    dman_full = du.DataManager.from_xps(loadedxp.values(), prog.training_config, inverse='all')

all_networks = dman_full.get_networks()
net_xp = [n.metadata['from_xp'] for n in all_networks]
net_name = [n.name for n in all_networks]

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
stack = training.build_compute_stack(compute_config)

with ut.timer('Stack initialization'):
    params = stack.init(key)


params
