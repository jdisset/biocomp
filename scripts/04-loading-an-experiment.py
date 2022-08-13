# %load_ext autoreload
# %autoreload 2

import pandas as pd
import numpy as np

import scriptutils as ut
import biocomp.utils as bu
from functools import partial
import biocomp as bc
import json
from rich import print

l = ut.load("all_sheets.pickle")
lib = bc.PartsLibrary(l.parts, l.L0s, l.L1s, l.L2s, l.categories, l.sequestrons, l.sequestron_types)


xp = json.load(open("example_xpfile.json"))

# remarks for the wet team
# - validate json
# - no caps for any field name (all lowercase, camel case)
# - no named aggregation, just a list of lists in a field named "aggregations"
# - can we make tubes XP instead? Tubes are a very bio-tied concept. From the software side, it makes sense to call a tube an XP, since it's the biggest independant unit. I get that a collection of XP can be related IRL, but maybe there's a better word? Or if we keep XP as a collection of "tubes", maybe there's a better word than tube that makes sense in both the soft and the wet worlds! Could a tube be a "run" or an "xp"? An XP an "assay"? I don't think these terms are perfect..

tubes = xp['tubes']
print(tubes[2])

XP = bc.Run(tubes[2], lib)

print(XP)

# TODO:
# [ ] build central dogma from XP
# [ ] build compute graph
# [ ] add aggregations to the compute graph

