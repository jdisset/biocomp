import pandas as pd
from .Tube import Tube
from .flowgeo import *
import json

class Experiment:
    '''
    Main class containing multiple tubes.
    '''

    ExperimentTube = Tube

    def __init__(self, 
        name=None,
        date=None,
        data=None,
        tubes=None,
        calibration_tubes=None,
        channels=None):
        
        self.name = name
        self.date = date
        self.tubes = {} if tubes is None else tubes
        self.calibration_tubes = [] if calibration_tubes is None else calibration_tubes
        self.channels = [] if channels is None else channels
        self.data = pd.DataFrame if data is None else data

    """
    def __iter__(self):
        '''
        iterate through tubes `for tube in experiment`
        '''
        return self.tubes
    """
    
    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)
    
    def populate_data(self, dataframe, singlets_conditions=None, ALLOWED_COLORS=None):
        '''
        
        initialise an Experiment object with data from cytoflow

        Parameters
        ----------
        dataframe
            Pandas dataframe
            You get this after gating etc from cytoflow (ex_N.data)
        
        singlets_conditions
            unwrappable tuple
            After gating, mark cells as cell singlets in Cytoflow.
            Pass conditions to `populate_data` to load the gates
                e.g. (('Cells', 'Cells_3'), ('SingletsSSC', True))
            If singlets_conditions=None, no gating will be applied in loading.

        ALLOWED_CHANNELS (optional)
            List of strings
            check color channels in dataframe to add color data to tubes

        returns
        -------
        Updated experiment object
        '''

        ALLOWED_COLORS_default = [
            'Pacific_Blue_A',
            'PE_Texas_Red_A',
            'FITC_A',
            'APC_A',
            'APC_Alexa_700_A',
            'AmCyan_A',
            'PE_A',
            'PerCP_Cy5_5_A',
        ]

        ALLOWED_CHANNELS = [
            'FITC_A',
            'FSC_A',
            'FSC_H',
            'FSC_W',
            'SSC_A',
            'SSC_H',
            'SSC_W',
            'Time',
        ]

        ALLOWED_COLORS = ALLOWED_COLORS_default if ALLOWED_COLORS is None else ALLOWED_COLORS

        if singlets_conditions:
            ex_data = subset(dataframe, *singlets_conditions)
        else:
            ex_data = dataframe
        
        self.data = ex_data.to_dict()

        tube_names = list(ex_data['TUBE_NAME'].unique())

        for tube_name in tube_names:
            tube = self.ExperimentTube(name=tube_name, 
                data=subset(ex_data, ('TUBE_NAME', tube_name)).to_dict())
            self.tubes[tube_name] = tube

