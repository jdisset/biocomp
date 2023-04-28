from biocomp import utils as ut
from pathlib import Path
import numpy as np
import scriptutils as su
import biocomp.datautils as du
import biocomp.train as train

prog = train.TrainingProgram()
prog.parse_args(['--config', 'epochs=2'])

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
net_name

net_name[184]
# su.plot_networks(all_networks[184:185])

##────────────────────────────────────────────────────────────────────────────}}}

### {{{               --     training and validation sets     --

# list net names that have cascade in the name:
inert_nets = {n: i for i, n in enumerate(net_name) if 'inert' in n.lower()}
cascade_nets = {
    n: i for i, n in enumerate(net_name) if 'cascade' in n.lower() and 'inert' not in n.lower()
}

inert_nets

cn = [all_networks[i] for i in cascade_nets.values()]

# training set is all networks except the ones in inert or cascade
training_set = [
    i
    for i, _ in enumerate(net_name)
    if i not in inert_nets.values() and i not in cascade_nets.values()
]

validation_set = [
    i for i, _ in enumerate(net_name) if i not in inert_nets.values() and i in cascade_nets.values()
]

cascade_nets
##────────────────────────────────────────────────────────────────────────────}}}
validation = dman_full.make_subset(validation_set)
training = dman_full.make_subset(training_set)
# prog.start_training(dman_full.make_subset(training_set), validation)

##

stack = training.build_compute_stack(prog.compute_config, max_t=1)

##
key = jax.random.PRNGKey(0)
with ut.timer('Stack initialization'):
    params = stack.init(key)



# for mid, n in enumerate(validation.get_networks()[:1]):
    # # fig, axes = du.mkfig(1, 4)
    # contours = np.linspace(0, 0.8, 5)
    # fig = du.network_plot(
        # validation,
        # mid,
        # n_views=1,
        # method='scatter',
        # contours=contours,
        # kde=False,
        # size=3,
        # lw=0.001,
        # radius=0.15,
        # knn=1000,
        # input_order=[0, 1, 2],
        # slices=np.linspace(0, 0.8, 4),
    # )

    # fig.show()

    # savepath = Path(f'~/Desktop/cascade_v3/{n.name}_3d.pdf').expanduser()
    # if not savepath.parent.exists():
    # savepath.parent.mkdir()
    # fig.savefig(savepath, bbox_inches='tight')

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

