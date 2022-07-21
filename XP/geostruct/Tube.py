import pandas as pd
import json

class Tube:
    '''
    Main class containing actual data.
    '''

    def __init__(self,
        name=None,
        data=None,
        aggregates=None):

        self.name = name
        self.data = {} if data is None else data
        self.aggregates = {} if aggregates is None else aggregates

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)
