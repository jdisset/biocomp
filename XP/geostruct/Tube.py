import pandas as pd


class Tube:
    '''
    Main class containing actual data.
    '''

    def __init__(self,
        name=None,
        data=None,
        aggregates=None):

        self.name = name
        self.data = pd.DataFrame if data is None else data
        aggregates = {} if aggregates is None else aggregates
