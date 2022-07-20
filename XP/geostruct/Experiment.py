import json
from .Tube import Tube

class Experiment:
    '''
    Main class containing multiple tubes.
    '''

    ExperimentTube = Tube

    def __init__(self, 
        name=None,
        date=None,
        tubes=None):
        
        self.name = name
        self.date = date
        self.tubes = {} if tubes is None else tubes
    
    def __iter__(self):
        '''
        iterate through tubes `for tube in experiment`
        '''
        return self.tubes