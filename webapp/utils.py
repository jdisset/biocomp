import networkx as nx
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import matplotlib.pyplot as plt
import collections.abc
import pandas as pd
import json
import sys

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                    --     random misc stuff     --
#···············································································

def flatten(t):
    return [item for sublist in t for item in sublist]

def is_interactive():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return True # Terminal running IPython
        if not hasattr(sys, 'ps1'):
            return True;
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter



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


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                   --     google sheet helpers     --
#···············································································

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


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                        --     JAX helpers     --
#···············································································

from jax.experimental import host_callback
from tqdm import tqdm
from jax import grad, jit, vmap, random, lax, tree_map
from jax import tree_util as pytree
import jax.numpy as jnp

# --- tqdm progress bar for jax scan ---
# This code is from this blog post: https://www.jeremiecoullon.com/2021/01/29/jax_progress_bar/
def tqdm_scan(num_samples, message=None):
    "Progress bar for a JAX scan"
    if message is None:
            message = ""
            # message = f"Running for {num_samples:,} iterations"
    tqdm_bars = {}

    if num_samples > 20:
        print_rate = int(num_samples / 100)
    else:
        print_rate = 1
    remainder = num_samples % print_rate

    def _define_tqdm(arg, transform):
        tqdm_bars[0] = tqdm(range(num_samples))
        tqdm_bars[0].set_description(message, refresh=False)

    def _update_tqdm(arg, transform):
        tqdm_bars[0].update(arg)

    def _update_progress_bar(iter_num):
        "Updates tqdm progress bar of a JAX scan or loop"
        _ = lax.cond(
            iter_num == 0,
            lambda _: host_callback.id_tap(_define_tqdm, None, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = lax.cond(
            # update tqdm every multiple of `print_rate` except at the end
            (iter_num % print_rate == 0) & (iter_num != num_samples-remainder),
            lambda _: host_callback.id_tap(_update_tqdm, print_rate, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = lax.cond(
            # update tqdm by `remainder`
            iter_num == num_samples-remainder,
            lambda _: host_callback.id_tap(_update_tqdm, remainder, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

    def _close_tqdm(arg, transform):
        tqdm_bars[0].close()

    def close_tqdm(result, iter_num):
        return lax.cond(
            iter_num == num_samples-1,
            lambda _: host_callback.id_tap(_close_tqdm, None, result=result),
            lambda _: result,
            operand=None,
        )

    def _progress_bar_scan(func):
        """Decorator that adds a progress bar to `body_fun` used in `lax.scan`.
        Note that `body_fun` must either be looping over `np.arange(num_samples)`,
        or be looping over a tuple who's first element is `np.arange(num_samples)`
        This means that `iter_num` is the current iteration number
        """

        def wrapper_progress_bar(carry, x):
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x   
            _update_progress_bar(iter_num)
            result = func(carry, x)
            return close_tqdm(result, iter_num)

        return wrapper_progress_bar

    return _progress_bar_scan


def get_pytree(t, i):
    return tree_map(lambda x: x[i], t)

def param_unstack(t, N):
    return [tree_map(lambda x: x[i], t) for i in range(N)]

def tree_stack(trees):
    """Takes a list of trees and stacks every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = pytree.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def tree_unstack(tree):
    """Takes a tree and turns it into a list of trees. Inverse of tree_stack.
    For example, given a tree ((a, b), c), where a, b, and c all have first
    dimension k, will make k trees
    [((a[0], b[0]), c[0]), ..., ((a[k], b[k]), c[k])]
    Useful for turning the output of a vmapped function into normal objects.
    """
    leaves, treedef = pytree.tree_flatten(tree)
    n_trees = leaves[0].shape[0]
    new_leaves = [[] for _ in range(n_trees)]
    for leaf in leaves:
        for i in range(n_trees):
            new_leaves[i].append(leaf[i])
    new_trees = [treedef.unflatten(l) for l in new_leaves]
    return new_trees


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
