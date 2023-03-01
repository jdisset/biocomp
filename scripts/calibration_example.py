from biocomp.calibration import Calibration
import scriptutils as ut
from pathlib import Path
from tqdm import tqdm
import pandas as pd

xp_path = ut.DEFAULT_XP_PATH / '2023-02-16_Matrix'
raw_path = xp_path / 'data/raw_data_gated'

# create the control dictionnary:
# key = name of the control, value = path to the csv or fcs file
# key follows the following convention:
# single color control : the name of the protein. 'EBFP2', 'EYFP', ...
# all color control: 
control_files = list(raw_path.glob('color_controls/*.csv'))
controls = {c.stem.split('.')[0]: c for c in control_files}

beads = list(raw_path.glob('beads/*.fcs'))[0]

cal = Calibration(controls, beads, reference_protein='EYFP', use_channels=['FITC', 'PACIFIC_BLUE', 'PE_TEXAS_RED'])
cal.fit_TASBE()

datafiles = list(raw_path.glob('*.csv'))

calibrated_path = xp_path / 'data/calibrated_data'
calibrated_path.mkdir(exist_ok=True)

for f in tqdm(datafiles):
    calibrated = cal.apply(pd.read_csv(f))
    calibrated.to_csv(calibrated_path / f.name, index=False)

