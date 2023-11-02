from biocomp import utils as ut
import scriptutils as su
import biocomp.datautils as du
import biocomp.train as train

XP = {
    'bt': '2023-04-03_Constraints_Pgu_Bleedthrough',
    'cascades': '2023-04-18_Constraints_PguCascades',
    'csy4matrix': '2023-03-26_MatrixCsy4',
    'casematrix': '2023-02-16_Matrix',
    'uorfs': '2022-11-10_uORFs_and_company',
}

prog = train.TrainingProgram()
# prog.parse_args('--config learning_rate=3e-4'.split())
prog.parse_args()
##

prog.training_config

##

with ut.timer(f'Loading data and building networks for {XP.keys()}'):
    loadedxp = {
        xpname: su.load_xp(xppath, su.load_lib(), data_path='./data/calibrated_data_v2')
        for xpname, xppath in XP.items()
    }
    dman = du.DataManager.from_xps(loadedxp.values(), prog.training_config, inverse='all')

nets = dman.get_networks()

su.plot_networks(nets[:3])


##

training_set = [i for i, n in enumerate(dman.get_networks()) if 'inert' not in n.name.lower()]

training = dman.make_subset(training_set)


##
# prog.start_training(training)

