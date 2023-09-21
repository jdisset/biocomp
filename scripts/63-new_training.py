from biocomp import utils as ut
import scriptutils as su
import biocomp.datautils as du
import biocomp.train as train

prog = train.TrainingProgram()
prog.parse_args()

XP = {
    'bt': '2023-04-03_Constraints_Pgu_Bleedthrough',
    'cascades': '2023-04-18_Constraints_PguCascades',
    'csy4matrix': '2023-03-26_MatrixCsy4',
    'casematrix': '2023-02-16_Matrix',
    'uorfs': '2022-11-10_uORFs_and_company',
}

with ut.timer(f'Loading data and building networks for {XP.keys()}'):
    loadedxp = {
        xpname: su.load_xp(xppath, su.load_lib(), data_path='./data/calibrated_data_v2')
        for xpname, xppath in XP.items()
    }
    dman_full = du.DataManager.from_xps(loadedxp.values(), prog.training_config, inverse='all')

training_set = [i for i, n in enumerate(dman_full.get_networks()) if 'inert' not in n.name.lower()]

validation = dman_full.make_subset(training_set[::10])
training = dman_full.make_subset(training_set)

prog.start_training(training, validation)
