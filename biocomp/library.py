import pandas as pd
from . import utils as ut


class PartsLibrary:
    def __init__(self, parts, L0s, L1s, L2s, categories, sequestrons, sequestron_types):
        self.parts = parts
        self.L0s = L0s.loc[L0s.index != '']
        self.L1s = L1s.loc[L1s.index != '']
        self.L2s = L2s.loc[L2s.index != '']
        self.categories = categories
        self.sequestrons = sequestrons
        self.sequestron_types = sequestron_types
        self.pc = pd.merge(parts, categories, left_on='category', right_index=True, how='left')
        self.seqs = self.sequestrons.merge(self.sequestron_types, left_on='type', right_index=True)
        self.seqs = ut.decode_json(self.seqs, ['output_part', 'output_category'])

    def addPart(self, part, category):
        self.parts.loc[part] = {'category': category}
        self.pc = pd.merge(
            self.parts, self.categories, left_on='category', right_index=True, how='left'
        )

    def addSequestron(self, dic):
        self.sequestrons = self.sequestrons.append(dic, ignore_index=True)
        self.seqs = self.sequestrons.merge(self.sequestron_types, left_on='type', right_index=True)
        self.seqs = ut.decode_json(self.seqs, ['output_part', 'output_category'])

    def getRna(self, dna):
        d = self.pc.loc[dna]
        return tuple(d[d.transcripted == 1].index)

    def getPrt(self, dna):
        d = self.pc.loc[dna]
        return tuple(d[d.translated == 1].index)

    def __repr__(self):
        return f"""
        Parts & categories: {self.pc},
        ------------------------------------------
        Sequestrons: \n{self.seqs}\n
        ------------------------------------------
        L0s: \n{self.L0s}\n
        ------------------------------------------
        L1s: \n{self.L1s}\n
        ------------------------------------------
        L2s: \n{self.L2s}\n
        """
