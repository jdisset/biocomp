import networkx as nx
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import matplotlib.pyplot as plt
import collections.abc
import pandas as pd
import json
import sys

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     draw GRN   --
#···············································································

def drawGRN(S):
    bgcol = "#ffffff"
    nodeColor = { 'dna' : "#699da3" ,  
            'rna' : "#5a708a",
            'prt':"#1e384e"}
    edgeColor = { 'transcription': [0.155, 0.144, 0.209],
            'translation' :[0.155, 0.144, 0.209],
            'erncut': [0.844, 0.1, 0.111]}

    fig, ax = plt.subplots(1, 1, figsize=(12,12),facecolor=bgcol)
    MIN_MARGIN = 30
    NODE_SIZE = 2500
    startpos = np.array([0,0])
    shiftX = 1000
    # pos = nx.spring_layout(S, pos = fixed_pos, k = 0.5, iterations = 50, fixed = fixed_pos.keys())
    startpos = np.array([0,0])
    shiftX = 100
    shiftY = -300
    pos = {n:startpos + i * np.array([shiftX, 0]) for i,n in enumerate(gdf[gdf.type=='dna'].index)}
    pos.update({n:startpos + np.array([0,shiftY]) + i * np.array([shiftX, 0]) for i,n in enumerate(gdf[gdf.type=='rna'].index)})
    pos.update({n:startpos + np.array([0,2*shiftY]) + i * np.array([shiftX, 0]) for i,n in enumerate(gdf[gdf.type=='prt'].index)})
    tr_edges = [[i,j] for i,j,d in S.edges(data=True) if d['type']=='transcription']
    tl_edges = [[i,j] for i,j,d in S.edges(data=True) if d['type']=='translation']
    cut_edges = [[i,j] for i,j,d in S.edges(data=True) if d['type']=='erncut']
    n_colors = [nodeColor[d['type']] for _,d in S.nodes(data=True)]
    n_labels = {i:str(i) for i in S.nodes()}

    nx.draw_networkx_nodes(S, pos, ax=ax, node_color=n_colors, node_size=NODE_SIZE, margins=0.25) 
    nx.draw_networkx_labels(S, pos, ax=ax, font_color='white', font_weight=800,labels = n_labels)
    nx.draw_networkx_edges(S, pos, ax=ax, edgelist=tr_edges, edge_color=edgeColor['transcription'],
                            width=3, min_source_margin=MIN_MARGIN, min_target_margin=MIN_MARGIN)
    nx.draw_networkx_edges(S, pos, ax=ax, edgelist=tl_edges, edge_color=edgeColor['translation'],
                            width=3, min_source_margin=MIN_MARGIN, min_target_margin=MIN_MARGIN)
    nx.draw_networkx_edges(S, pos, ax=ax, edgelist=cut_edges, edge_color=edgeColor['erncut'],
                            width=3, arrowstyle='-[', min_source_margin=MIN_MARGIN, min_target_margin=MIN_MARGIN) 
    ax.set_facecolor(bgcol)
    plt.show()
    print('')


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


def flatten(t):
    return [item for sublist in t for item in sublist]

# build a networkx graph from its pandas description
def pandasGraphToNx(gdf):
    G = nx.MultiDiGraph()
    for i,n in gdf.iterrows():
        G.add_node(i, type=n.type)
    for i,n in gdf.iterrows():
        if n.successor:
            G.add_edge(i,n.successor,type='transcription' if n.type == 'dna' else 'translation')
    return G

def updated_dict(d1, d2):
    res = {}
    for key, val in d1.items():
        if type(val) == dict:
            if key in d2 and type(d2[key] == dict):
                res[key] = updated_dict(d1[key], d2[key])
        else:
            if key in d2:
                res[key] = d2[key]
            else:
                res[key] = d1[key]
    for key, val in d2.items():
        if not key in d1:
            res[key] = val
    return res

def decode_json(df, cols):
    for col in cols:
        df[col] = df[col].apply(lambda x: json.loads(str(x)))
    return df

def isSubset(l1, l2):
    for e in l1:
        if e not in l2:
            return False
    return True


GOOGLE_APP_CREDENTIALS = '/Users/jeandisset/.google/biocomp/key.json'
# This function grabs the content of a google sheet and returns a pandas dataframe:
def getGoogleSheet(key, sheet_name, credentials=GOOGLE_APP_CREDENTIALS):
    gspread_client = gspread.service_account(filename=credentials)
    workbook = gspread_client.open_by_key(key)
    sheet = workbook.worksheet(sheet_name)
    data = sheet.get_all_values()
    headers = data.pop(0)
    df = pd.DataFrame(data, columns=headers)
    df = df.set_index(df.columns[0])
    return df

def getAllGoogleSheets(key, credentials=GOOGLE_APP_CREDENTIALS):
    gspread_client = gspread.service_account(filename=credentials)
    workbook = gspread_client.open_by_key(key)
    sheets = workbook.worksheets()
    sheets_dict = {}
    for sheet in sheets:
        df = pd.DataFrame(sheet.get_all_records())
        df.set_index(df.columns[0], inplace=True)
        sheets_dict[sheet.title] = df
    return sheets_dict

def listGoogleSpreadsheets(credentials=GOOGLE_APP_CREDENTIALS):
    gspread_client = gspread.service_account(filename=credentials)
    spreadsheets = gspread_client.openall()
    if spreadsheets:
        print("Available spreadsheet workbooks:")
        for spreadsheet in spreadsheets:
            print("Title:", spreadsheet.title, "URL:", spreadsheet.url)
    else:
        print("No spreadsheets available")
        print("Please share the spreadsheet with Service Account email")


