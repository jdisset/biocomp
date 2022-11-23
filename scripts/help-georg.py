import scriptutils as ut

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                             --    constructs      --
# ···············································································
constructs = [
    """
[1:1] -> (Csy4) ; (Csy4_5’ + CasE)
[x] -> (Pgu)
[x] -> (CasE_5’ + YFP + Pgu_3’)
""",
    """
[1:1] -> (Csy4) ; (Csy4_5’ + CasE)
[x] -> (Pgu)
[x] -> (Pgu_5’ + YFP + CasE_3’)
""",
    """
[1:1] -> (CasE) ; (CasE_5’ + Pgu)
[x] -> (Csy4)
[x] -> (Csy4_5’ + YFP + Pgu_3’)
""",
    """
[4:1] -> (Csy4) ; (Csy4_5’ + CasE)
[x] -> (Pgu)
[x] -> (CasE_5’ + YFP + Pgu_3’)
""",
    """
[4:1] -> (Csy4) ; (Csy4_5’ + CasE)
[x] -> (Pgu)
[x] -> (Pgu_5’ + YFP + CasE_3’)
""",
    """
[4:1] -> (CasE) ; (CasE_5’ + Pgu)
[x] -> (Csy4)
[x] -> (Csy4_5’ + YFP + Pgu_3’)
""",
    """
[1:4] -> (Csy4) ; (Csy4_5’ + CasE)
[x] -> (Pgu)
[x] -> (CasE_5’ + YFP + Pgu_3’)
""",
    """
[1:4] -> (Csy4) ; (Csy4_5’ + CasE)
[x] -> (Pgu)
[x] -> (Pgu_5’ + YFP + CasE_3’)
""",
    """
[1:4] -> (CasE) ; (CasE_5’ + Pgu)
[x] -> (Csy4)
[x] -> (Csy4_5’ + YFP + Pgu_3’)
""",
    """
[x] -> (Csy4)
[x] -> (Csy4_5’ + CasE)
[x] -> (CasE_5’ + YFP)
""",
    """
[x] -> (Pgu)
[x] -> (Pgu_5’ + CasE)
[x] -> (CasE_5’ + YFP)
""",
    """
[x] -> (Pgu)
[x] -> (Pgu_5’ + Csy4)
[x] -> (Csy4_5’ + YFP)
""",
    """
[x] -> (Csy4)
[x] -> (Csy4_5’ + CasE)
[x] -> (CasE_3’ + YFP)
""",
    """
[x] -> (Pgu)
[x] -> (Pgu_5’ + CasE)
[x] -> (CasE_3’ + YFP)
""",
    """
[x] -> (Pgu)
[x] -> (Pgu_5’ + Csy4)
[x] -> (Csy4_3’ + YFP)
""",
    """
[x] -> (CasE)
[x] -> (Csy4)
[x] -> (Csy4_5’ + YFP + CasE_3’)
""",
    """
[x] -> (CasE)
[x] -> (Pgu)
[x] -> (CasE_5’ + YFP + Pgu_3’)
""",
    """
[x] -> (CasE)
[x] -> (Pgu)
[x] -> (Pgu_5’ + YFP + CasE_3’)
""",
    """
[x] -> (YFP)
[x] -> (YFP)
[x] -> (YFP)
""",
    """
[x] -> (CasE)
[x] -> (CasE_5’ + YFP)
[x] -> (YFP)
""",
    """
[x] -> (CasE)
[x] -> (YFP + CasE_3’)
[x] -> (YFP)
""",
    """
[x] -> (CasE)
[x] -> (CasE)
[x] -> (CasE_5’ + YFP)
""",
    """
[x] -> (CasE)
[x] -> (CasE_5’ + YFP)
[x] -> (CasE_5’+ YFP)
""",
    """
[x] -> (CasE)
[x] -> (CasE_5’ + YFP)
[x] -> (YFP + CasE_3’)
""",
    """
[x] -> (Csy4)
[x] -> (Csy4_5’ + YFP)
[x] -> (YFP)
""",
    """
[x] -> (Csy4)
[x] -> (YFP + Csy4_3’)
[x] -> (YFP)
""",
    """
[x] -> (Csy4)
[x] -> (Csy4)
[x] -> (Csy4_5’ + YFP)
""",
    """
[x] -> (Csy4)
[x] -> (Csy4_5’ + YFP)
[x] -> (Csy4_5’ + YFP)
""",
    """
[x] -> (Csy4)
[x] -> (Csy4_5’ + YFP)
[x] -> (YFP + Csy4_3’)
""",
    """
[x] -> (Pgu)
[x] -> (Pgu_5’ + YFP)
[x] -> (YFP)
""",
    """
[x] -> (Pgu)
[x] -> (YFP + Pgu_3’)
[x] -> (YFP)
""",
    """
[x] -> (Pgu)
[x] -> (Pgu)
[x] -> (Pgu_5’ + YFP)
""",
    """
[x] -> (Pgu)
[x] -> (Pgu_5’ + YFP)
[x] -> (Pgu_5’ + YFP)
""",
    """
[x] -> (Pgu)
[x] -> (Pgu_5’ + YFP)
[x] -> (YFP + Pgu_3’)
""",
    """
[1:5] -> (CasE) ; (CasE_5’ + YFP)
[x] -> (Csy4)
[x] -> (Csy4_5’ + YFP)
""",
    """
[x] -> (CasE)
[x] -> (CasE_5’ + YFP)
[x] -> (CasE_3’ + YFP)
""",
    """
[x] -> (Csy4)
[x] -> (Csy4_5’ + YFP)
[x] -> (Csy4_3’ + YFP)
""",
    """
[x] -> (Pgu)
[x] -> (Pgu_5’ + YFP)
[x] -> (Pgu_3’ + YFP)
""",
    """
[1:9] -> (YFP) ; (RFP)
""",
]
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

len(constructs)

con = [[l for l in cc.splitlines() if l.strip()] for cc in constructs]

# parse format
# a line is: {ratios : [a:b:...]} -> (part) ; (part_5' + part_2' + part_3') ; ...
# where ratios are the ratios of the parts in the construct
# we want to parse each line in a dict

def parse_line(line):
    ratios, _, parts = line.partition('->')
    ratios = ratios.strip()
    parts = parts.strip()
    # print(line)
    # print(f'ratios: {ratios}')
    # print(f'parts: {parts}')
    # strip the brackets
    ratios = ratios[1:-1]

    if ratios:
        ratios = ratios.split(':')
        if len(ratios) == 1:
            ratios = [1]
        else:
            ratios = [int(r) for r in ratios]

    parts = parts.split(';')
    parts = [p.strip()[1:-1].split('+') for p in parts]
    parts = [[pp.strip() for pp in p] for p in parts if p != ['']]



    return {'ratios': ratios, 'parts': parts}


con = [[parse_line(l) for l in cc] for cc in con]

# from con parts to csv part names
part_to_csv= {'CasE': 'G.CasE', 'Csy4': 'G.Csy4', 'Pgu': 'G.Pgu', 'YFP': 'G.eYFPG5A',
            'CasE_5’': '5.CasE.rec', 'CasE_3’':'3.CasE-ON', 'Csy4_5’': '5.Csy4.rec',
            'Csy4_3’': '3.Csy4-ON', 'Pgu_5’': '5.Pgu.rec', 'Pgu_3’': '3.Pgu-ON'}
# dual:
csv_to_part = {v.lower(): k for k, v in part_to_csv.items()}

# open csv file for parts
import csv
with open('/Users/jeandisset/Downloads/dna_designs (2).csv') as f:
    reader = csv.reader(f)
    parts = list(reader)
    # remove empty lines
    parts = [p for p in parts if p[0]]
    parts = parts[1:]
    # replace any matching column name with the part name, case insensitive
    for i, p in enumerate(parts):
        for j, c in enumerate(p):
            if c.lower() in csv_to_part:
                parts[i][j] = csv_to_part[c.lower()]



# first column of parts is the name of the plasmid. 
# We want to assign a plasmid name to each construct

# remember each construct in con is a list of dicts (one dict per cotx)
# and in a cotx, there can be multiple transcription_units, which themselves are list of parts

# we want to assign a plasmid to each transcription_unit

# we do that by counting the number of parts in common between the transcription_unit and the plasmid


def match_score(tu, plasmid):
    s = sum([1 for p in tu if p in plasmid])
    # and we subtract the number of parts that are in the plasmid but not in the tu
    s -= sum([1 for p in plasmid if p not in tu])
    return s

def assign_plasmid(tu, plasmids):
    best_match = []
    best_match_score= -100
    for p in plasmids:
        s = match_score(tu, p)
        if s > best_match_score:
            best_match_score = s
            best_match = [p[0]]
        elif s == best_match_score:
            best_match.append(p[0])

    print(f'best match for {tu} is {best_match} with score {best_match_score}')
    if len(best_match) == 1:
        return best_match[0]
    else:
        return 'NO MATCH'


for cc in con:
    for cotx in cc:
        plasmids = []
        for tu in cotx['parts']:
            plasmid = assign_plasmid(tu, parts)
            plasmids.append(plasmid)
        cotx['plasmids'] = plasmids

con

def pretty_print_con(con):
    for cc in con:
        print('-' * 80)
        for cotx in cc:
            print(' -' * 10)
            print(f"ratios: {cotx['ratios']}")
            print(f"parts: {cotx['parts']}")
            print(f"plasmids: {cotx['plasmids']}")

pretty_print_con(con)

def as_csv(con):
    for cc in con:
        for cotx in cc:
            print(','.join(cotx['plasmids']))

