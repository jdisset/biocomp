from pathlib import Path
import argparse
import pickle
import os
from types import SimpleNamespace
import biocomp as bc

### {{{                   --     google sheet helpers     --
import gspread
from rich.progress import track
import pandas as pd

GOOGLE_APP_CREDENTIALS = '/Users/jeandisset/.google/biocomp/key.json'
SHEET_KEY = '1K_2bt90E-Wk-A9PYGXGbKDJy-olojKtksy1jxCQAzME'


def getAllGoogleSheets(key=SHEET_KEY, credentials=GOOGLE_APP_CREDENTIALS):
    gspread_client = gspread.service_account(filename=credentials)
    workbook = gspread_client.open_by_key(key)
    sheets = workbook.worksheets()
    sheets_dict = {}
    for sheet in track(sheets, description='Loading library sheets'):
        df = pd.DataFrame(sheet.get_all_records())
        col0 = df.columns[0]
        previous_len = len(df)
        df = df[df[col0] != ""] # remove empty rows
        print(f"Ignoring {previous_len - len(df)} empty rows from {sheet.title}")
        df.set_index(col0, inplace=True)
        sheets_dict[sheet.title] = df
    lib = SimpleNamespace(**sheets_dict)
    return lib


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


def getLibFromGoogleSheet(key=SHEET_KEY, credentials=GOOGLE_APP_CREDENTIALS):
    l = getAllGoogleSheets(key, credentials)
    lib = bc.PartsLibrary(
        l.parts, l.L0s, l.L1s, l.L2s, l.categories, l.sequestrons, l.sequestron_types
    )
    return lib

def main(libpath):
    libpath = Path(libpath)
    libpath.parent.mkdir(parents=True, exist_ok=True)
    print("Updating biocomp lib from google sheets...")
    lib = getLibFromGoogleSheet()
    with open(libpath, "wb") as f:
        pickle.dump(lib, f)
    print("Done.")

###                                                                            }}}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="where to save the biocomp lib")
    args = parser.parse_args()
    # if no path is given, try to get the BIOCOMP_LIB_PATH environment variable
    if args.path is None:
        args.path = os.getenv("BIOCOMP_LIB_PATH")
    main(args.path)

