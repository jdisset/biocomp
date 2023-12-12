### {{{                          --     imports     --
import sys

from dataclasses import dataclass
from typing import List, Tuple

from biocomp import utils as ut
import json
import biocomp.datautils as du
import biocomp.plotutils as pu
import biocomp.utils as ut
import biocomp.train as train
import biocomp.compute as cmp
import biocomp.parameters as pm
import biocomp as bc
import time
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json5

# pretty print from rich
from rich import print as rprint

##────────────────────────────────────────────────────────────────────────────}}}
### {{{                      --     helpers    --
import sqlite3
import pandas as pd


def find_all_keys_recursively(d, key):
    if isinstance(d, dict):
        if key in d:
            yield d[key]
        for k, v in d.items():
            if isinstance(v, dict):
                for result in find_all_keys_recursively(v, key):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in find_all_keys_recursively(d, key):
                        yield result


def get_parts(l1id):
    l0_cols = ["insulator", "promoter", "5'UTR", "gene", "3'UTR", "terminator"]
    l0_cols = ["promoter", "5'UTR", "gene", "3'UTR"]
    # l0_cols = ["5'UTR", "gene", "3'UTR"]
    L0s = lib.L1s.loc[l1id][l0_cols].tolist()
    part_cols = [f'part_{i}' for i in range(1, 7)]
    parts = []
    for l in L0s:
        parts += [p for p in lib.L0s.loc[l][part_cols].tolist() if p]
    return parts


def has_sample_data(xp_path, xp_name, sample_name):
    data_path = xp_path / 'data'
    if not data_path.exists():
        return False
    return (data_path / f"{sample_name}.{xp_name}.csv").exists()


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                      --     load parameters     --
training_archive = ut.load('../../__results/training_archives/20230923_fulltrain_v0.pkl')
shared_parameters = training_archive['parameters']
compute_config = training_archive['compute_config']
training_config = training_archive['training_config']
compute_config.set_impl('bias', bc.nodes.bias)
lib = ut.load_lib()
##────────────────────────────────────────────────────────────────────────────}}}
### {{{                  --     list all xps in experiment folder    --
xp_path = ut.DEFAULT_XP_PATH
recipe_paths = ut.DEFAULT_RECIPE_PATH
calib_paths = ['./data/calibrated_data_v3', './data/calibrated_data_v2', './data/calibrated_data']

# list all folders
xp_folders = sorted([f for f in xp_path.iterdir() if f.is_dir()])

xp_entries = {}
xp_objs = {}
for xp_dir in xp_folders[:]:
    warning_msg = ''
    subfolders = sorted([f for f in xp_dir.iterdir() if f.is_dir()])
    # check if there is {xp_dir.name}.xp.json5
    xp_json = xp_dir / f'{xp_dir.name}.xp.json5'
    if not xp_json.exists():
        print(f'xp.json5 not found in {xp_dir.name}')
        continue

    base_xp_path = xp_dir.parent

    xp = bc.XP(
        xp_dir.name,
        base_xp_path,
        recipe_path=recipe_paths,
        lib=lib,
        # db_path=dbpath,
        data_path=calib_paths,
        load_data=False,
        ignore_errors=True,
    )

    recipe_loading_errors = xp.recipe_loading_errors
    xp_entries[xp.name] = {
        'name': xp.name,
        'transfection_date': xp.transfection_date,
        'path': xp_dir,
        'recipe_errors': xp.recipe_loading_errors,
    }
    xp_objs[xp.name] = xp


##────────────────────────────────────────────────────────────────────────────}}}
### {{{            --     initial xpdf with calibration info     --
def calibration_info(xppath):
    calib_folders = list(xppath.glob('data/calibrated_data*'))
    calib_type = 'no'
    calib_plot = False
    calib_path = None

    if len(calib_folders) > 0:
        if any([f.name == 'calibrated_data_v3' for f in calib_folders]):
            calib_type = 'v3'
            calib_path = Path('./data/calibrated_data_v3')
        elif any([f.name == 'calibrated_data_v2' for f in calib_folders]):
            calib_type = 'v2'
            calib_path = Path('./data/calibrated_data_v2')
        else:
            calib_type = 'old'
            calib_path = Path(f'./data/{calib_folders[0].name}')

    # check if there is a calibration plot
    calib_diag_path = xppath / 'data' / 'unmixing_diagnostics'
    if calib_diag_path.exists():
        calib_plot = True

    return calib_type, calib_plot, calib_path


for xp in xp_entries.values():
    calib_type, calib_plot, _ = calibration_info(xp['path'])
    xp['calibration_version'] = calib_type
    xp['calibration_diagnostics'] = calib_plot

##────────────────────────────────────────────────────────────────────────────}}}
### {{{  --     create dmans dictionary of datamanagers (where available)    --
dmans = {}
all_recipes = []
all_networks = []
xp_dmans = {}

print('Creating datamanagers')

for xpname, xp in list(xp_objs.items())[:]:
    print(f'loading {xpname}')
    is_ok = True
    if xp.data_files:
        print(f'loading {xpname}')
        xp.load_raw_data()
        networks, samples = xp.build_networks(ignore_errors=True, inverse='all')
        X, Y = xp.get_XY(networks, samples, ignore_errors=True)
        if xp.network_building_errors:
            is_ok = False
            print(f'{xp.network_building_errors}')
        if xp.data_loading_errors:
            is_ok = False
            print(f'{xp.data_loading_errors}')
        xp_entries[xpname]['network_building_errors'] = xp.network_building_errors
        xp_entries[xpname]['data_loading_errors'] = xp.data_loading_errors
        assert len(networks) == len(X) == len(Y)
        for i, net_entry in enumerate(networks):
            if net_entry:
                net_entry = {
                    'xp': xpname,
                    'network': net_entry,
                    'sample': samples[i],
                }
                all_networks.append(net_entry)
        if is_ok:
            for x, y, net_entry in zip(X, Y, networks):
                if x.size == 0 or y.size == 0:
                    is_ok = False
                    xp_entries[xpname][
                        'data_loading_errors'
                    ] += f'empty data for network {net_entry.name}\n\n'

        if is_ok:
            xp_dmans[xpname] = du.DataManager(X, Y, networks, data_cfg=training_config)


print('done')
networks

##────────────────────────────────────────────────────────────────────────────}}}##
### {{{                      --     topology analysis helpers     --


def get_uorf_value(param):
    if 'tl_rate' in param:
        u = param['tl_rate'][0].split('_')[0]
        try:
            v = int(u[:-1]) * 10
        except ValueError:
            v = 0
        if u[-1] == 'w':
            v = v - 5
        return v
    else:
        return 0


UORF_DICT = {
    0: 'No uORF',
    5: 'weak uORF',
    10: '1x uORF',
    20: '2x uORF',
    30: '3x uORF',
    40: '4x uORF',
    50: '5x uORF',
    60: '6x uORF',
    70: '7x uORF',
    80: '8x uORF',
}


def get_all_ERN_ids(network):
    ERN_ids = network.compute_graph[network.compute_graph['type'] == 'sequestron_ERN'].index.values
    return network.sort_nodes_by_upstream(ERN_ids)


def get_all_ERNs_names(network):
    ERNs = network.compute_graph.loc[get_all_ERN_ids(network)]
    ERN_extras = ERNs['extra'].values
    ERN_names = [e['seq_name'].split('#')[0].split('::')[-1] for e in ERN_extras]
    return ERN_names


def get_uorf_names(uorf_values, ern_names):
    uorf_names = []
    for uorf, ern_name in zip(uorf_values, ern_names):
        ERN_uorf, REC_uorf = uorf
        ERN_uorf = UORF_DICT[ERN_uorf]
        REC_uorf = UORF_DICT[REC_uorf]
        uorf_names.append((f'{ern_name} ERN: {ERN_uorf}', f'{ern_name} REC: {REC_uorf}'))
    return uorf_names


def get_all_uorf_values(network):
    cdg = network.central_dogma_graph
    ERNs = network.compute_graph.loc[get_all_ERN_ids(network)]
    ERN_names = get_all_ERNs_names(network)
    ERN_inputs = ERNs['cdg_input'].values
    values = []
    for inp in ERN_inputs:
        cdgin = cdg.loc[inp]
        ern_side = cdg.loc[cdgin.iloc[0].predecessor[0]]
        recog_side = cdgin.iloc[1]
        uvals = (get_uorf_value(ern_side.params), get_uorf_value(recog_side.params))
        values.append(uvals)
    names = get_uorf_names(values, ERN_names)
    return tuple(values), tuple(names)


##────────────────────────────────────────────────────────────────────────────}}}
### {{{        --     add architecture family, sequestron type, ...     --

from typing import List, Callable


def get_ERN_ids(network):
    return network.compute_graph[network.compute_graph['type'] == 'sequestron_ERN'].index.values


def get_RCB_ids(network):
    return network.compute_graph[
        network.compute_graph['type'].str.startswith('sequestron_R')
    ].index.values


def get_sequestron_ids(network):
    return network.compute_graph[
        network.compute_graph['type'].str.startswith('sequestron_')
    ].index.values


def make_is_upstream(network):
    def is_upstream(i, j):
        return network.compute_node_is_upstream_of(i, j)

    return is_upstream


def topological_sort(
    node_list: List[int], is_upstream: Callable[[int, int], bool]
) -> List[List[int]]:

    visited = set()
    batches = []
    while len(visited) < len(node_list):
        independent = [
            i
            for i in node_list
            if i not in visited
            and all([j in visited for j in node_list if j != i and is_upstream(j, i)])
        ]

        if not independent:
            raise ValueError('Cycle detected in graph')
        visited.update(independent)
        batches.append(independent)
    return batches


def get_network_family(network):
    erns = get_ERN_ids(network)
    rcbs = get_RCB_ids(network)
    seqs = get_sequestron_ids(network)
    ts = topological_sort(seqs, make_is_upstream(network))

    seqtype = 'none'
    family = 'unknown'
    match (len(erns) > 0, len(rcbs) > 0):
        case (True, True):
            seqtype = 'hybrid'
        case (True, False):
            seqtype = 'ERN'
        case (False, True):
            seqtype = 'RCB'

    match (len(seqs), len(ts)):
        case (0, 0):
            family = 'no device'
        case (1, 1):
            family = 'single'
        case (2, 2):
            family = 'cascade'
        case (2, 1):
            family = 'dual region'
        case (3, 2):
            family = 'bandpass'

    return family, seqtype


def flatten(l):
    return [item for sublist in l for item in sublist]


local_savedir = Path('~/ResearchMisc/biocomp/').expanduser()
local_savedir.mkdir(parents=True, exist_ok=True)
url_base = 'https://jdisset.com/biocomp'


for net_entry in all_networks:
    net = net_entry['network']
    arch, seqtype = get_network_family(net)
    uorf_vals, uorf_names = get_all_uorf_values(net)
    cdg = net.central_dogma_graph
    genes = flatten(cdg[cdg.type == 'PRT']['content'].tolist())
    full_netname = f'{net.metadata["from_xp"]}.{net.name}'
    geneplotfile = f'jeanplots/{full_netname}.png'
    dataplotfile = f'dataplots/{full_netname}.png'
    new_entry = {
        'xp': net.metadata['from_xp'],
        'name': net.name,
        'sequestron_type': seqtype,
        'architecture': arch,
        'ERN_names': ', '.join(get_all_ERNs_names(net)),
        'uorf_values': ', '.join([str(v) for v in uorf_vals]),
        'uorf_names': ', '.join(flatten(uorf_names)),
        'genes': ', '.join(genes),
        'markers': ', '.join(net.get_inverted_input_proteins()),
        'output_proteins': ', '.join(net.get_output_proteins()),
        'recipe_file': net.metadata['recipe_file'],
        'geneplot': f'=image("{url_base}/{geneplotfile}")',
        'dataplot': f'=image("{url_base}/{dataplotfile}")',
    }
    net_entry.update(new_entry)


##────────────────────────────────────────────────────────────────────────────}}}##

### {{{   --     turn xp_entries and all_networks into xpdf and netdf     --
xpdf = pd.DataFrame(xp_entries).T
# replace all the *_errors types to string
for col in xpdf.columns:
    if '_errors' in col:
        xpdf[col] = xpdf[col].astype(str)
        xpdf[col] = xpdf[col].apply(lambda x: x.replace('nan', ''))


netdf = pd.DataFrame(all_networks)
# netdf = netdf.drop(columns=['network', 'X', 'Y', 'jeanplot'])
netdf = netdf.drop(columns=['network', 'X', 'Y'])
# put name column in front
cols = netdf.columns.tolist()
col_order_beginning = ['name', 'xp', 'architecture']
col_order_end = ['geneplot', 'dataplot']
col_order = (
    col_order_beginning
    + [c for c in cols if c not in col_order_beginning + col_order_end]
    + col_order_end
)
netdf = netdf[col_order]

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                          --     traindf     --
import json

this_training = [
    {
        'training_date': None,
        'training_duration': None,
        'final_loss': None,
        'compute_config': compute_config.dumps(),
        'training_config': json.dumps(training_config, indent=4),
        'training_subset_id': None,
    }
]


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                          --     add ids     --
# add numeric row id
# puts the id column in front
# first we sort by xp name
xpdf = xpdf.sort_values(by=['name'])
xpdf['id'] = np.arange(len(xpdf))
xpdf = xpdf[['id'] + [c for c in xpdf.columns if c != 'id']]

netdf['xp_id'] = netdf['xp'].apply(lambda x: xpdf[xpdf['name'] == x]['id'].values[0])
# first we group by xp_id, then for each group we sort by net.name
# the net.id will be the index of each net in the group (cumcount)
# net are sorted by name first:
netdf = netdf.sort_values(by=['name'])
netdf['id_in_xp'] = netdf.groupby('xp_id').cumcount()
# for nets, their id will be {xp_id}/{i}
netdf['id'] = netdf.apply(lambda x: f'{x["xp_id"]}/{x["id_in_xp"]}', axis=1)
netdf = netdf.drop(columns=['xp_id', 'id_in_xp'])
front_cols = ['id']
netdf = netdf[front_cols + [c for c in netdf.columns if c not in front_cols]]

traindf = pd.DataFrame(this_training)
traindf['id'] = np.arange(len(traindf))
##────────────────────────────────────────────────────────────────────────────}}}

### {{{     --     read existing excel table to keep using same ids     --
import shutil
from openpyxl import Workbook

onedrive = Path(
    ''.join(
        [
            '~/Library/CloudStorage/',
            'OneDrive-SharedLibraries-MassachusettsInstituteofTechnology/',
            'Neuromorphic Biocompiler - Documents',
        ]
    ),
).expanduser()

tablefile= 'BigTable/bigtable.xlsx'
xlpath = onedrive / tablefile
xlpath.parent.mkdir(parents=True, exist_ok=True)


if xlpath.exists():
    shutil.copy(xlpath, xlpath.parent / f'{xlpath.stem}_backup.xlsx')
else:

    # create file if not exists
    with pd.ExcelWriter(xlpath, engine='openpyxl') as writer:
        xpdf.to_excel(writer, sheet_name='experiments', index=False)
        netdf.to_excel(writer, sheet_name='networks', index=False)

# Open the file and load the specified sheets
og_netdf= pd.read_excel(xlpath, sheet_name='networks')
og_xpdf = pd.read_excel(xlpath, sheet_name='experiments')

# we merge on xp name and keep the id column from the original table but
# everything else from the new table
new_xpdf = pd.merge(xpdf, og_xpdf, on='name', how='left', suffixes=('','_old'))
new_xpdf['id'] = new_xpdf['id_old'].fillna(new_xpdf['id'])
new_xpdf = new_xpdf[[c for c in new_xpdf.columns if not c.endswith('_old')]]

# for networks:
# we merge on the combination of xp name and net name
new_netdf = pd.merge(
    netdf,
    og_netdf,
    left_on=['xp', 'name'],
    right_on=['xp', 'name'],
    how='left',
    suffixes=('','_old'),
)
mask = new_netdf['id_old'] == ''
new_netdf.loc[mask, 'id_old'] = new_netdf.loc[mask, 'id']
new_netdf['id'] = new_netdf['id_old'].fillna(new_netdf['id'])
new_netdf = new_netdf[[c for c in new_netdf.columns if not c.endswith('_old')]]

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                --     manual write to excel sheet     --
# Dump the 'netdf' dataframe into the "network" sheet
with pd.ExcelWriter(xlpath, engine='openpyxl', mode='a') as writer:
    book = writer.book
    # Remove the existing "network" sheet if it exists
    if 'network' in book.sheetnames:
        del book['network']
    # Write the 'netdf' dataframe to a new "network" sheet
    new_netdf.to_excel(writer, sheet_name='network', index=False)
##────────────────────────────────────────────────────────────────────────────}}}

netdf.columns
xpdf.columns

### {{{                          --     to sql     --

sql_schema = """
CREATE TABLE Experiment (
    id INTEGER PRIMARY KEY,
    name TEXT,
    transfection_date TEXT,
    path TEXT,
    network_building_errors INTEGER,
    calibration_version INTEGER,
    calibration_diagnostics BOOLEAN,
    data_loading_errors INTEGER,
    recipe_errors INTEGER
);

CREATE TABLE Recipe (
    id INTEGER PRIMARY KEY,
    name TEXT,
    file_path TEXT,
    design_training_run_id INTEGER,
    design_target_id INTEGER,
    design_loss TEXT
);

CREATE TABLE Network (
    id INTEGER PRIMARY KEY,
    sample_name TEXT,
    sequestron_type TEXT,
    architecture TEXT,
    uorf_values TEXT,
    ERN_names TEXT,
    genes TEXT,
    markers TEXT,
    output_proteins TEXT,
    data_plot_path TEXT,
    gene_plot_path TEXT,
    recipe_id INTEGER,
    FOREIGN KEY (recipe_id) REFERENCES Recipe (id)
);

CREATE TABLE Training_Set (
    id INTEGER PRIMARY KEY,
    name TEXT,
    description TEXT
);

CREATE TABLE Training_Run (
    id INTEGER PRIMARY KEY,
    training_set_id INTEGER,
    FOREIGN KEY (training_set_id) REFERENCES Training_Set (id)
);

CREATE TABLE Prediction (
    id INTEGER PRIMARY KEY,
    error REAL,
    plot_path TEXT,
    network_id INTEGER,
    training_run_id INTEGER,
    FOREIGN KEY (network_id) REFERENCES Network (id),
    FOREIGN KEY (training_run_id) REFERENCES Training_Run (id)
);

CREATE TABLE Recipe_in_Experiment (
    recipe_id INTEGER,
    experiment_id INTEGER,
    FOREIGN KEY (recipe_id) REFERENCES Recipe (id),
    FOREIGN KEY (experiment_id) REFERENCES Experiment (id)
);

CREATE TABLE Network_in_Training_Set (
    network_id INTEGER,
    training_set_id INTEGER,
    FOREIGN KEY (network_id) REFERENCES Network (id),
    FOREIGN KEY (training_set_id) REFERENCES Training_Set (id)
);

CREATE TABLE Target (
    id INTEGER PRIMARY KEY,
    image_path TEXT,
    constraints TEXT
);
"""

# Create a new SQLite in-memory database
conn = sqlite3.connect(':memory:')
c = conn.cursor()
c.executescript(sql_schema)


##────────────────────────────────────────────────────────────────────────────}}}

tablefile2 = 'BigTable/bigtable_v2.xlsx'
xlpath = onedrive / tablefile2

import pandas as pd
from pandas import ExcelWriter
import sqlite3
from typing import List

def load_sheets_as_sqlite(workbook: str, sheet_names: List[str]) -> sqlite3.Connection:
    """
    Load specified sheets from an Excel workbook into an SQLite database in memory.
    Args:
    workbook (str): Path to the Excel workbook.
    sheet_names (List[str]): List of sheet names to load into SQLite.
    Returns:
    sqlite3.Connection: Connection object to the SQLite in-memory database.
    """
    # Create a new SQLite in-memory database
    conn = sqlite3.connect(':memory:')
    # Load each specified sheet into a pandas DataFrame and then into SQLite
    for sheet in sheet_names:
        df = pd.read_excel(workbook, sheet_name=sheet)
        df.to_sql(sheet, conn, if_exists='replace', index=False)
    return conn


def save_sqlite_as_sheets(db_connection: sqlite3.Connection, table_names: List[str], workbook: str, sheet_names: List[str]):
    """
    Save specified SQLite tables into an Excel workbook as sheets.
    Args:
    db_connection (sqlite3.Connection): Connection object to the SQLite database.
    table_names (List[str]): List of table names in SQLite to save to Excel.
    workbook (str): Path to the Excel workbook where sheets will be saved.
    sheet_names (List[str]): List of sheet names for the Excel workbook.
    """
    # Initialize an Excel writer object with the openpyxl engine
    with ExcelWriter(workbook, engine='openpyxl') as writer:
        for table, sheet in zip(table_names, sheet_names):
            df = pd.read_sql_query(f"SELECT * FROM {table}", db_connection)
            df.to_excel(writer, sheet_name=sheet, index=False)
        # Save the workbook

all_tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
all_tables = [t[0] for t in all_tables]

save_sqlite_as_sheets(conn, all_tables, xlpath, all_tables)


