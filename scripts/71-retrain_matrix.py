from biocomp import utils as ut
import scriptutils as su
import biocomp.datautils as du
import biocomp.train as train
import numpy as np

prog = train.TrainingProgram()
prog.parse_args('--config learning_rate=3e-4'.split())

prog.training_config

##

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


def get_all_uorf_values(network):
    cdg = network.central_dogma_graph
    ERN_inputs = network.compute_graph[network.compute_graph['type'] == 'sequestron_ERN'][
        'cdg_input'
    ].values
    values = []
    for inp in ERN_inputs:
        cdgin = cdg.loc[inp]
        ern_side = cdg.loc[cdgin.iloc[0].predecessor[0]]
        recog_side = cdgin.iloc[1]
        values.append((get_uorf_value(ern_side.params), get_uorf_value(recog_side.params)))
    return tuple(values)


##────────────────────────────────────────────────────────────────────────────}}}

from collections import defaultdict

uorf_dict = defaultdict(list)
multi_uorf = defaultdict(list)
for i, n in enumerate(dman_full.get_networks()):
    has_ERN_node = n.compute_graph['type'] == 'sequestron_ERN'
    number_of_ERN_nodes = has_ERN_node.sum()
    if number_of_ERN_nodes == 1:
        uorf_dict[get_uorf_values(n)].append(i)
    if number_of_ERN_nodes > 1:
        multi_uorf[get_all_uorf_values(n)].append(i)

uorf_dict
corners = [(0, 0), (0, 80), (80, 0), (80, 80)]

non_corner_ids = []
for coords, ids in uorf_dict.items():
    if coords not in corners:
        non_corner_ids.extend(ids)

for coords, ids in multi_uorf.items():
    sumuorf = np.array(coords).sum()
    if sumuorf > 0:
        non_corner_ids.extend(ids)
        print(coords, ids)

training_set = [
    i
    for i, n in enumerate(dman_full.get_networks())
    if 'inert' not in n.name.lower() and i not in non_corner_ids
]
training = dman_full.make_subset(training_set)

len(training_set)

##

prog.start_training(training)
