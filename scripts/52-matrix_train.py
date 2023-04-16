from biocomp import utils as ut
import scriptutils as su
import biocomp.datautils as du
import biocomp.train as train

MAX_UORF = 80
TRAINING_SETS = {
    '1_corner': [(0, 0)],
    '2_corners_recog': [(0, 0), (0, MAX_UORF)],
    '2_corners_ern': [(0, 0), (MAX_UORF, 0)],
    '2_corners_diag': [(0, 0), (MAX_UORF, MAX_UORF)],
    '3_corners': [(0, 0), (0, MAX_UORF), (MAX_UORF, 0)],
    '4_corners': [(0, 0), (0, MAX_UORF), (MAX_UORF, 0), (MAX_UORF, MAX_UORF)],
    'all': None,
}

XP = {'case': '2023-02-16_Matrix', 'csy4': '2023-03-26_MatrixCsy4'}
prog = train.TrainingProgram()
prog.add_argument('xp', type=str, help=f'xp to train on from {list(XP.keys())}')
prog.add_argument(
    'training_set',
    type=str,
    help=f'name of training set to use from {list(TRAINING_SETS.keys())}',
)
prog.parse_args()

### {{{                      --     loading matrix xp     --
with ut.timer(f'Loading data and building networks for {XP[prog.xp.lower()]}'):
    lib = su.load_lib()
    matrix_xp = su.load_xp(XP[prog.xp.lower()], lib, data_path='./data/calibrated_data')
    dman_full = du.DataManager.from_xps([matrix_xp], prog.training_config, inverse='all')
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


subset = single_uorfs + [uorf_dict[i] for i in TRAINING_SETS[prog.training_set]]

prog.start_training(dman_full.make_subset(subset))
