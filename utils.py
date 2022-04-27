import networkx as nx
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import matplotlib.pyplot as plt
import collections.abc
import pandas as pd

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

def dicupdate(dict1, dict2):
    res = {}
    for key, val in dict1.items():
        if type(val) == dict:
            if key in dict2 and type(dict2[key] == dict):
                dicupdate(dict1[key], dict2[key])
        else:
            if key in dict2:
                dict1[key] = dict2[key]
    for key, val in dict2.items():
        if not key in dict1:
            dict1[key] = val
    return dict1

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


