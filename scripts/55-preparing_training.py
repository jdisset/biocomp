from biocomp import utils as ut
from pathlib import Path
import jax.numpy as jnp
import jax
import numpy as np
import scriptutils as su
import biocomp.datautils as du
import biocomp.train as train

prog = train.TrainingProgram()
prog.add_argument('--subset', type=str, help=f'nets to train on')
# prog.parse_args(['--config', 'epochs=3'])
prog.parse_args()

# nans with [281 228  33 126 307 207 287  73 127 125 282   1   3 165   6]
# and [ 34 205 177 107  50   55 238 161 170 130 308  78 207  46  73  38]
# intersection:
# s0 = {281, 228, 33, 126, 307, 207, 287, 73, 127, 125, 282, 1, 3, 165, 6}
# s1 = {34, 205, 177, 107, 50, 55, 238, 161, 170, 130, 308, 78, 207, 46, 73, 38}
# intersection = s0.intersection(s1)

ut.logger.debug(f'Using {prog.device} device')

### {{{                      --     loading xp     --

XP = {
    'bt': '2023-04-03_Constraints_Pgu_Bleedthrough',
    'cascades': '2023-04-18_Constraints_PguCascades',
    'csy4matrix': '2023-03-26_MatrixCsy4',
    'casematrix': '2023-02-16_Matrix',
}
xpnames = ['bt', 'cascades', 'csy4matrix', 'casematrix']

with ut.timer(f'Loading data and building networks for {xpnames}'):
    lib = su.load_lib()
    loadedxp = {xpname: su.load_xp(XP[xpname], lib, data_path=prog.data_path) for xpname in xpnames}
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

test_set = [0, 10, len(all_networks) - 1, 50, 60, 20, 40, 44, 42, 120, 250, 280, 300]

##────────────────────────────────────────────────────────────────────────────}}}

if prog.args.subset == 'random':
    nsub = 15  # we pick these from all the networks that are not inert
    pool = [i for i in range(len(net_name)) if i not in inert_nets.values()]
    training_set = np.random.choice(pool, nsub, replace=False)
    ut.logger.info(f'Randomly selected {nsub} networks for training: {training_set}')

#  or it could be a list (that we need to parse), e.g.: "[1, 2, 3]"
elif prog.args.subset is not None and prog.args.subset.startswith('['):
    training_set = eval(prog.args.subset)
    ut.logger.info(f'Using training set {training_set}')


validation = dman_full.make_subset(validation_set)
training = dman_full.make_subset(training_set)
prog.start_training(dman_full.make_subset(training_set), validation)

##


# net_name
# net_name[73]
# su.plot_networks([all_networks[73]], [(path/f'{net_name[73]}.pdf').as_posix()])

##

# BADNET_ID = 73
# badnet = all_networks[BADNET_ID]
# badnet.compute_graph
# badman = dman_full.make_subset([BADNET_ID])
# badman.build_compute_stack(prog.compute_config)
# badstack = badman.get_compute_stack()
# params = badstack.init(jax.random.PRNGKey(0))
# badnet.compute_graph.loc[16].extra

# from rich import print as pprint
# for k in params.keys():
    # if 'agg' in k:
        # pprint(k, params[k])

# # def tree_has_nan(tree):
    # # for v in tree.values():
        # # if np.isnan(v).any():
            # # return True
    # # return False

# # tree_has_nan(params)

# n_inputs = badnet.get_nb_inputs()
# n_inputs
# n_outputs = badnet.get_nb_outputs()
# n_outputs
# badnet.get_output_proteins()
# badnet.get_inverted_input_proteins()

# # apply(params, inputs, quantiles, key):

# params

# inputs = jnp.zeros(n_inputs)
# quantiles = jnp.zeros(n_outputs)
# badstack.apply(params, inputs, quantiles, jax.random.PRNGKey(0))


# path = Path('~/Desktop/').expanduser()
# su.plot_networks([all_networks[73]], [(path/f'{net_name[73]}.pdf').as_posix()])
# all_networks[73].compute_graph
# all_networks[73].central_dogma_graph

# key = jax.random.PRNGKey(0)
# vstack = validation.build_compute_stack(prog.compute_config)
# base_params = vstack.init(key)

# ##
# testing = dman_full.make_subset(test_set)
# tstack = testing.build_compute_stack(prog.compute_config)
# t_params = tstack.init(key)

# ##
# vstack.shared_store
# tstack.shared_store
# from rich import print as rprint
# for k, v in t_params.items():
# if 'qvals' in k:
# rprint(f'{k}:{v}')
# ##

# testing = dman_full.make_subset(test_set)
# stack = testing.build_compute_stack(prog.compute_config, max_t=1)
# key = jax.random.PRNGKey(0)
# with ut.timer('Stack initialization'):
# params = stack.init(key)


##

# params
# # for mid, n in enumerate(validation.get_networks()[2]):

# mid = 0
# n = validation.get_networks()[mid]
# n.get_inverted_input_proteins()


# fig, axes = du.mkfig(1, 4)
# contours = np.linspace(0, 0.8, 5)
# fig = du.network_plot(
# validation,
# mid,
# n_views=1,
# ax=None,
# axes=axes,
# # method='scatter',
# contours=contours,
# # kde=False,
# size=3,
# lw=0.001,
# radius=0.15,
# knn=1000,
# input_order=[0, 1, 2],
# slices=np.linspace(0.1, 0.8, 4),
# )


# savepath = Path(f'~/Desktop/cascade_v3/{n.name}_3d.pdf').expanduser()
# if not savepath.parent.exists():
# savepath.parent.mkdir()
# fig.savefig(savepath, bbox_inches='tight')

##


##

# TODO:
# "Can we guess what a cascade looks like without seeing one?"
# [ ] make training set with [Csy4, Case] matrices + single uOrfs +
#       rows: 2, 3, 4, 6:16 -> USE EVERYTHING that's not detrimental (or contains cascade)
# Validation set: 22, 23, 24, 25
# [ ] Plot and quantify accuracy
# [ ] Add all of the validation set BUT ONE XP to the training set ; what's the accuracy on the remaining one?
# [ ] Plot distribution of null-transfected cells to see the size of the zero band
# [ ] plot validation with new x also (the ones that are actually computed)

##
