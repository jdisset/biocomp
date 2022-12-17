import biocomp as bc
import pandas as pd
import scriptutils as ut
import biocomp.datautils as du

l = ut.getAllGoogleSheets()
du.save(l, 'all_sheets.pickle')

l2 = du.load('all_sheets.pickle')

