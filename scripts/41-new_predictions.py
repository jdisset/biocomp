### {{{                          --     imports     --
import biocomp as bc
from biocomp import datautils as du
from jax.scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from biocomp.calibration import Calibration
import scriptutils as ut
from pathlib import Path
import json5
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import gaussian_kde
import jax
import optax
from jax import jit, vmap, value_and_grad
from jax.tree_util import Partial as partial
from tqdm import tqdm
import biocomp.defaults as bdf
import pandas as pd

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                        --     node config     --
T_SIZE = 64
T_DEPTH = 4
I_SIZE = 64
I_DEPTH = 3
I_OUT = 8
ERN_SIZE = 128
ERN_DEPTH = 4
MEFL_SIZE = 64
MEFL_DEPTH = 4

node_impl = dict(
    bc.nodes.DEFAULT_COMPUTE_NODES_DICT,
    **{
        'output': partial(bc.nn.output, wsize=MEFL_SIZE, depth=MEFL_DEPTH),
        'transcription': partial(
            bc.nn.transcription,
            outer_wsize=T_SIZE,
            outer_depth=T_DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
            inner_out=I_OUT,
        ),
        'translation': partial(
            bc.nn.translation,
            outer_wsize=T_SIZE,
            outer_depth=T_DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
            inner_out=I_OUT,
        ),
        'inv_transcription': partial(
            bc.nn.inv_transcription,
            outer_wsize=T_SIZE,
            outer_depth=T_DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
            inner_out=I_OUT,
        ),
        'inv_translation': partial(
            bc.nn.inv_translation,
            outer_wsize=T_SIZE,
            outer_depth=T_DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
            inner_out=I_OUT,
        ),
        'sequestron_ERN': partial(bc.nn.ERN5p, wsize=ERN_SIZE, depth=ERN_DEPTH),
        'sequestron_ERN3p': partial(bc.nn.ERN3p, wsize=ERN_SIZE, depth=ERN_DEPTH),
    },
)

config = {
    **bdf.DEFAULT_CONFIG,
    **{
        'node_impl': node_impl,
        'rng_key': np.random.randint(0, 2**32),
    },
}

##────────────────────────────────────────────────────────────────────────────}}}

lib = ut.load_lib()
matrix_xp = ut.load_xp('2023-02-16_Matrix', lib, data_path='./data/calibrated_data')

matrix_xp

# dman = du.DataManager.from_xps([uorf_xp, ern_xp], config, inverse='all')
# loggers = bc.train.setup_wandb_logging('quantile_v2', dman, config)
# bc.train.start(dman, config, loggers)

##

xp_path = ut.DEFAULT_XP_PATH / '2023-02-16_Matrix'
# load al csv files in xp_path/data/raw_data_gated
raw_path = xp_path / 'data/raw_data_gated'
datafiles = list(raw_path.glob('*.csv'))

control_files = list(raw_path.glob('color_controls/*.csv'))
control_color = [c.stem.split('.')[0] for c in control_files]
controls = {c: cpath for c, cpath in zip(control_color, control_files)}
beads = list(raw_path.glob('beads/*.fcs'))[0]

cal = Calibration(controls, beads, reference_protein='EBFP2', use_channels=['FITC', 'PACIFIC_BLUE', 'PE_TEXAS_RED'])

cal.fit_TASBE()

##
calibrated_path = xp_path / 'data/calibrated_data'
calibrated_path.mkdir(exist_ok=True)

for f in tqdm(datafiles):
    calibrated = cal.apply(pd.read_csv(f))
    calibrated.to_csv(calibrated_path / f.name, index=False)



