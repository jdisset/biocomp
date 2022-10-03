import biocomp as bc
import pandas as pd
import scriptutils as ut

l = ut.getAllGoogleSheets()
ut.save(l, 'all_sheets.pickle')

l2 = ut.load('all_sheets.pickle')

