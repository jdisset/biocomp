# Calibry

Calibry is a Python package for calibrating fluorescence values in flow cytometry data. 
It uses color controls, blank controls, and beads with known fluorescence values for calibration.

## Installation

To install the Calibry package, follow these steps:

1. Clone the Calibry repository

```bash
git clone <calibry_repository_url>
```

2. Install the package

After cloning the repository, navigate to the Calibry directory and
run the following command to install the package and its dependencies:
```
pip install .
```

3. Update the package

To update the Calibry package to the latest version, first pull the latest changes from the GitHub repository:

```
git pull
```

Then, reinstall the package:

```
pip install --upgrade .
```

## Usage

Here's an example of how to use the Calibration class from the Calibry package:

```python

from calibry import Calibration
import pandas as pd

# Your control and beads files
beads = "path/to/beads/file.fcs"

# create the control dictionnary:
# key = name of the control, value = path to the csv or fcs file
# key follows the following convention:
# single color control : the name of the protein. 'EBFP2', 'EYFP', ...
# all color control: 'ALL'
# blank: 'CNTL' or 'EMPTY'
control_files = list(raw_path.glob('color_controls/*.csv'))
controls = {c.stem.split('.')[0]: c for c in control_files}


# Initialize the Calibration instance
cal = Calibration(controls, beads, reference_protein='MKATE', use_channels=['FITC', 'PACIFIC_BLUE', 'PE_TEXAS_RED', 'APC_ALEXA_700'])

# Fit the calibration model
cal.fit()

# Apply the calibration to your data
data = pd.read_csv("path/to/data/file.csv")
calibrated = cal.apply(data)
```

## Calibration Class

The Calibration class is the main class of the Calibry package. It handles the calibration of the fluorescence values in a given experiment and can export to a calibrated pandas dataframe.

### Constructor Arguments

    * `color_controls_files`: A dictionary with keys as control protein names (e.g., 'eYFP') and values as paths to their corresponding FCS or CSV files.

    * `beads_file`: A string representing the path to the beads control FCS file.

    * `reference_protein`: (Optional) A string representing the name of the reference protein used to align other protein quantities. The default value is 'mkate'.

    * `reference_channel`: (Optional) A string representing the name of the reference channel used to convert to standard MEF units. The default value is 'PE_TEXAS_RED'.

    * `beads_mef_values`: (Optional) A dictionary associating the unit name with the list of reference values for the beads in this unit. The default value is SPHEROTECH_RCP_30_5a.

    * `channel_to_unit`: (Optional) A dictionary associating the channel name with the unit name. The default value is FORTESSA_CHANNELS.

    * `random_seed`: (Optional) An integer representing the random seed for any resampling that might be done. The default value is 42.

    * `use_channels`: (Optional) A list of channel names to use when computing protein quantity from channel values (i.e., bleedthrough correction). The default value is None, which means all available channels will be used.

    * `max_value`: (Optional) An integer representing the maximum fluorescence value used to detect saturation. The default value is FORTESSA_MAX_V.

    * `offset`: (Optional) An integer representing the offset to add to the fluorescence values to avoid negative values. The default value is FORTESSA_OFFSET.


Please refer to the comments and documentation in the calibry.py file for more details.
