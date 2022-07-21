import json

def __init__(self, name=None, plasmids=None):
    self.name = name
    self.plasmids = plasmids

    self.plasmids = [] if plasmids is None else plasmids

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)


if __name__ == '__main__':
    exampleDict = {
        'agg1': {
            'plasmidObj',
            'plasmidObj2'
        },
        'agg2': {},
    }