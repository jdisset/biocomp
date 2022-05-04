import networkx as nx
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import matplotlib.pyplot as plt
import collections.abc
import pandas as pd
import json
import sys


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

class DotDict(dict):
    def __getattr__(*args):
        val = dict.__getitem__(*args)
        return DotDict(val) if type(val) is dict else val
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

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


