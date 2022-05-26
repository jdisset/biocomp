import pandas as pd
from . import utils as ut

class PartsLibrary:
    def __init__(self, parts, categories, sequestrons, sequestron_types):
        self.parts = parts
        self.categories = categories
        self.sequestrons = sequestrons
        self.sequestron_types = sequestron_types
        self.pc = pd.merge(parts, categories, left_on='category', right_index=True, how='left')
        self.seqs = self.sequestrons.merge(self.sequestron_types, left_on='type', right_index=True)
        self.seqs = ut.decode_json(self.seqs, ['output_part', 'output_category'])

    def getRna(self, dna):
        d = self.pc.loc[dna]
        return tuple(d[d.transcripted == 1].index)

    def getPrt(self, dna):
        d = self.pc.loc[dna]
        return tuple(d[d.translated == 1].index)

    def __repr__(self):
        return str({"Parts & categories": self.pc, "Sequestrons":self.seqs})

