### {{{                          --     imports     --
import sys

from dataclasses import dataclass
from typing import List, Tuple

from biocomp import utils as ut
import json
import biocomp.datautils as du
import biocomp.utils as ut
import biocomp.plotutils as pu
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
from dataclasses import dataclass
from typing import Any
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Arrow, Rectangle
import numpy as np
from svgpath2mpl import parse_path
import xml.etree.ElementTree as etree
import matplotlib as mpl
import re
from io import StringIO

# pretty print from rich
from rich import print as rprint


##────────────────────────────────────────────────────────────────────────────}}}

import logging as log
# log = ut.get_logger(__name__)


### {{{                     --     get tu inf proto     --
network = networks[22]
network.compute_graph
network.central_dogma_graph.loc[2]
network.central_dogma_graph.loc[3]


layout = get_tu_grid_layout(network)
tudf = get_tu_informations(network)



##

tus = []
cotx_prots = network.get_inverted_input_proteins()
sources = network.compute_graph[network.compute_graph['type'] == 'source']
for s, src in sources.iterrows():
    plasmid_name = '_'.join(src['source_id'].split('_')[:-1])
    tu_cdgs = network.central_dogma_graph.loc[src['cdg_output']]
    tus_from_plasmid = []
    is_in_l2 = len(tu_cdgs) > 1

    for p, (_, tu_row) in enumerate(tu_cdgs.iterrows()):
        assert len(tu_row['tu_id']) == 1
        tu_id = tu_row['tu_id'][0]
        tu_name = '_'.join(tu_id.split('_')[:-1])
        content = tu_row['content']
        cotx = [g for g in content if g in cotx_prots]
        if len(cotx) > 1:
            raise ValueError(f'{tu_name} contains more than one marker??')
        ismarker = len(cotx) == 1
        if ismarker:
            cotx = cotx[0]
        tus_from_plasmid.append({
            'tu_name': tu_name,
            'tu_id': tu_id,
            'cotx_marker': cotx if ismarker else None,
            'is_marker': ismarker,
            'plasmid_name': plasmid_name,
            'in_l2': is_in_l2,
            'position_in_plasmid': p,
        })

    # now check if the plasmid is an aggregation
    upstream_node_id = src['input_from'][0][0]
    upstream_node_type = network.compute_graph.at[upstream_node_id, 'type']
    if upstream_node_type == 'aggregation':
        # if it is, we need to check the ratios
        ratios = network.compute_graph.at[upstream_node_id, 'extra']['ratios']
        for i, tu in enumerate(tus_from_plasmid):
            tu['aggregation_ratio'] = ratios[i]
            tu['aggregation_node_id'] = upstream_node_id
    tus.extend(tus_from_plasmid)

tudf = pd.DataFrame(tus)
tudf['aggregation_node_id'] = tudf['aggregation_node_id'].fillna(-1)
tudf['aggregation_node_id'] = tudf['aggregation_node_id'].astype(int)
tudf['in_aggregation'] = tudf['aggregation_node_id'] != -1
# ratios are in fraction, but we want them normalized to lowest value (per aggregation)
tudf['aggregation_ratio_norm'] = tudf.groupby('aggregation_node_id')['aggregation_ratio'].transform(
    lambda x: x / x.min()
)
marker_rows = tudf[tudf['is_marker']]
tudf.loc[marker_rows.index, 'marker_ratio'] = marker_rows['aggregation_ratio_norm'].values
tudf['marker_ratio'] = tudf.groupby('aggregation_node_id')['marker_ratio'].transform('max')
tudf['aggregation_ratio_label'] = tudf.apply(
    lambda r: f'{r.aggregation_ratio_norm:.0f}:{r.marker_ratio:.0f}',
    axis=1,
)
tudf.loc[~tudf['in_aggregation'], 'aggregation_ratio_label'] = ''
# propagate cotx_marker to all row with the same plasmid_name
for _, row in tudf[tudf['is_marker']].iterrows():
    other_in_plasmid = tudf[(tudf['plasmid_name'] == row['plasmid_name']) & (tudf['tu_id'] != row['tu_id'])]
    assert other_in_plasmid['cotx_marker'].isna().all()
    tudf.loc[other_in_plasmid.index, 'cotx_marker'] = row['cotx_marker']
    # and now same for aggregation nodes
    if row['in_aggregation']:
        other_in_agg = tudf[(tudf['aggregation_node_id'] == row['aggregation_node_id']) & (tudf['tu_id'] != row['tu_id'])]
        assert other_in_agg['cotx_marker'].isna().all()
        tudf.loc[other_in_agg.index, 'cotx_marker'] = row['cotx_marker']

tudf


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                          --     config     --
lib = ut.load_lib()

RESOURCES_PATH = Path('./resources').resolve()
PARTS_PATH = RESOURCES_PATH / 'parts'


# relative part units: 0 to 1, starting from bottom left to top right, relative to width and height of part
#   (0, 0) is bottom left, (1, 1) is top right

# origin (x, y) in relative part units
# offset: (x, y) in relative part units
# label_position: (x, y) in relative part units

# font_size will be scaled with ax (in data units)

# mpl don't use latex:
mpl.rcParams['text.usetex'] = False

BASE_FONT_SIZE = 7
SMALL_FONT_SIZE = 6
LARGE_FONT_SIZE = 9


DEFAULT_PART_PARAMS = {
    'default': {
        'main_color': '#EEEEEE',
        'secondary_color': '#EEEEEE',
        'anchor_points': [(0.5, 1), (0.5, -1.5)],
        'edgecolor': 'k',
        'linewidth': 0,
        'origin': (0, 0.5),  # center left
        'label_properties': {
            'fontname': 'Roboto',
            'fontsize': BASE_FONT_SIZE,
            'ha': 'center',
            'va': 'center',
            'linespacing': 1.5,
        },
    },
    'promoter': {'origin': (-0.5, 0)},
    'terminator': {'origin': (0, 0)},
    'fluo_marker': {
        'label_position': (0.45, 0.0),
    },
    'ERN': {
        'label_text': 'ERN',
        'label_position': (0.2, 0.0),
        'label_properties': {'ha': 'left', 'va': 'center'},
        'anchor_points': [(0.78, 0.75), (0.78, -0.8)],
    },
    'ERN_recog_site_5p': {
        'origin': (0, 0.1),
        'label_text': 'ERN\nTS',
        'label_position': (0.5, -0.175),
        'label_properties': {
            'fontsize': SMALL_FONT_SIZE,
            'ha': 'center',
            'va': 'top',
        },
    },
    'uORF_group': {
        'label_position': (0.5, -0.5),
        'label_properties': {
            'fontsize': SMALL_FONT_SIZE,
            'ha': 'center',
            'va': 'top',
        },
    },
}

uorfparts = lib.parts[lib.parts['category'] == 'uORF_group'].index.tolist()
UORF_GROUPS_PARAMS = {}
for u in uorfparts:
    label = u.replace('_', '\n')
    UORF_GROUPS_PARAMS[f'uORF_group.{u}'] = {'label_text': label}
DEFAULT_PART_PARAMS = ut.updated_dict(DEFAULT_PART_PARAMS, UORF_GROUPS_PARAMS)

ERN_COLORS = {
    'Csy4': '#AAAAAA',
    'CasE': '#CCCCCC',
    'PgU': '#EEEEEE',
}

ERN_PART_PARAMS = {}
for e, c in ERN_COLORS.items():
    ERN_PART_PARAMS[f'ERN.{e}'] = {'main_color': c, 'label_text': e}
    ERN_PART_PARAMS[f'ERN_recog_site_5p.{e}_rec'] = {'main_color': c, 'label_text': f'{e}\nTS'}
DEFAULT_PART_PARAMS = ut.updated_dict(DEFAULT_PART_PARAMS, ERN_PART_PARAMS)


lib.parts[lib.parts['category'] == 'fluo_marker'].index.tolist()


BASE_FLUO_COLORS = {
    'red': {'base': '#ef957d', 'light': '#ffe5de', 'dark': '#840137'},
    'green': {'base': '#6CCB83', 'light': '#EFFFDD', 'dark': '#0E633A'},
    'blue': {'base': '#6cafc3', 'light': '#F3FAFD', 'dark': '#006394'},
    'yellow': {'base': '#FAD26D', 'light': '#FFF8B8', 'dark': '#9B6600'},
    'ir': {'base': '#df9ae4', 'light': '#ffe7f4', 'dark': '#6c1772'},
    'maroon': {'base': '#D3A888', 'light': '#F4DECD', 'dark': '#734727'},
}


MARKER_COLORS = {
    'NeonGreen': BASE_FLUO_COLORS['green'],
    'eYFP': BASE_FLUO_COLORS['yellow'],
    'eBFP': BASE_FLUO_COLORS['blue'],
    'mKate': BASE_FLUO_COLORS['red'],
    'iRFP720': BASE_FLUO_COLORS['ir'],
    '1xiRFP720': BASE_FLUO_COLORS['ir'],
    'L0.G_mNeonGreen': BASE_FLUO_COLORS['green'],
    'L0.G_iRFPmystery': BASE_FLUO_COLORS['ir'],
    'eYFPG5A': BASE_FLUO_COLORS['yellow'],
    'tagBFP': BASE_FLUO_COLORS['blue'],
    'tdTomato': BASE_FLUO_COLORS['red'],
    'mKO2': BASE_FLUO_COLORS['red'],
    'mMaroon1': BASE_FLUO_COLORS['maroon'],
}

THEME = {
    'colors': {
        'markers': MARKER_COLORS,
    },
    'fontname': 'Roboto',
}


def get_color_family(protein):
    return MARKER_COLORS.get(protein, 'k')


MARKER_ALIAS = {
    'NeonGreen': 'mNeonGreen',
    'L0.G_mNeonGreen': 'mNeonGreen',
    'L0.G_iRFPmystery': 'iRFPmystery',
    'eBFP': 'eBFP2',
}

FLUO_PART_PARAMS = {}
for m, c in MARKER_COLORS.items():
    FLUO_PART_PARAMS[f'fluo_marker.{m}'] = {
        'main_color': c['light'],
        'label_text': MARKER_ALIAS.get(m, m),
    }
DEFAULT_PART_PARAMS = ut.updated_dict(DEFAULT_PART_PARAMS, FLUO_PART_PARAMS)


DEFAULT_ENABLED_TU_SLOTS = ["promoter", "5'UTR", "gene", "3'UTR", "terminator"]
DEFAULT_TU_PARAMS = {
    # all widths in ax units
    'default': {
        'enabled_slots': DEFAULT_ENABLED_TU_SLOTS,
        'slot_widths': {
            'promoter': 22,
            '5\'UTR': 40,
            'gene': 60,
            '3\'UTR': 1,
            'terminator': 15,
        },
        'display_empty_slots': True,
        'rescale_parts': 'never',  # 'never', 'always', 'if_too_big'
        'parts_spacing': 12,
        'default_part_width': 3,
        'tu_linewidth': 1.2,
        'padding': (0.1, 0.1),
        'zorder': -1,
    },
}


DEFAULT_CONFIG = {
    'Theme': THEME,
    'Part': DEFAULT_PART_PARAMS,
    'TU': DEFAULT_TU_PARAMS,
    'Wrapper': {
        'default': {
            'display_border': True,
            'border_properties': {
                'ec': '$$Theme/colors/markers/%%marker_color%%/dark$$',
                'boxstyle': 'round,pad=0.0,rounding_size=5',
                'fc': 'none',
                'linewidth': 2.5,
                'zorder': 1,
                'alpha': 0.5,
                'linewidth': 0.3,
                'linestyle': (0, (7, 7)),
            },
        },
    },
    'Label': {
        'default': {
            'origin': (1.0, 0.5),
            'size': (84, 12),
            'relative_position': (0.9, 0.0),
            'logo_offset': (-67, 0.5),
            'logo_max_size': (10, 10),
            'text_offset': (-62, 0),
            'text_properties': {
                'ha': 'left',
                'va': 'center',
                'color': '#555',
                'fontsize': BASE_FONT_SIZE,
            },
            'shape_properties': {
                'ec': '#BBB',
                'fc': '#EEE',
                'linewidth': 0.5,
                'zorder': 1,
                'boxstyle': 'round,pad=0.0,rounding_size=6.5',
                'alpha': 1,
            },
        },
        'aggregation': {
            'logo_svg': 'symbols/aggregation.svg',
            'logo_min_size': (7, 7),
            'logo_main_color': '$$Theme/colors/markers/%%marker_color%%/dark$$',
            'text_properties': {
                'color': '$$Theme/colors/markers/%%marker_color%%/dark$$',
            },
            'shape_properties': {
                'ec': '$$Theme/colors/markers/%%marker_color%%/base$$',
                'fc': '$$Theme/colors/markers/%%marker_color%%/light$$',
            },
        },

        'l2': {
            'relative_position': (0.15, 0.0),
            'origin': (0.0, 0.5),
            'logo_offset': (2, 0.0),
            'logo_max_size': (10, 10),
            'logo_min_size': (7, 7),
            'text_offset': (15, 0),
            'logo_svg': 'symbols/l2.svg',
            'logo_min_size': (7, 7),
            'logo_main_color': '$$Theme/colors/markers/%%marker_color%%/dark$$',
            'text_properties': {
                'color': '$$Theme/colors/markers/%%marker_color%%/dark$$',
            },
            'shape_properties': {
                'ec': '$$Theme/colors/markers/%%marker_color%%/base$$',
                'fc': '$$Theme/colors/markers/%%marker_color%%/light$$',
            },
        },

    },
    'NetworkScene': {
        'row_spacing': 20,
        'col_spacing': 20,
        'show_cotx_tu': False,
        'cell_size': (210, 80),
    },
}

DEFAULT_CONFIG


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                       --     Configurable     --
class Configurable:
    def __init__(self, section_name, params=None, priorities=None, params_context=None, **_):
        self.priorities = priorities
        self.section_name = section_name
        self.update_params(params, params_context=params_context, **_)

    def update_params(self, params, params_context=None, **kwargs):
        self.all_params = params or {}
        params = resolve_references(params, context=params_context)
        section_params = params.get(self.section_name, {})
        if self.priorities is None:
            self.local_params = section_params.copy()
        else:
            self.local_params = section_params.get(self.priorities[0], {}).copy()
            for pname in self.priorities[1:]:
                self.local_params = ut.updated_dict(
                    self.local_params, section_params.get(pname, {})
                )
        self.local_params = ut.updated_dict(self.local_params, kwargs)


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                       --     Positionable     --


class Positionable:
    def __init__(
        self,
        position=(0, 0),
        origin=(0, 0),
        size=(1, 1),
        scale=1.0,
        relative_position=None,
        margin=0,
        **_,
    ):
        self.position = np.asarray(position)
        self.size = np.asarray(size)
        self.width = size[0]
        self.height = size[1]
        self.scale = scale
        self.origin = origin
        self.relative_position = relative_position
        self.margin = margin
        if isinstance(margin, (int, float)):
            self.padding = (margin, margin)

    def set_transform(self, position, scale=1.0, size=None):
        self.position = np.asarray(position)
        self.scale = scale
        if size is not None:
            self.size = np.asarray(size)
            self.width = size[0]
            self.height = size[1]
        self.on_transform_update()

    def get_bottom_left_position(self):
        return self.position - self.size * self.scale * self.origin

    def get_center_position(self):
        return self.position - self.size * self.scale * self.origin + self.size * self.scale / 2

    def get_relative_position(self, position):
        return self.get_bottom_left_position() + np.asarray(position) * self.size * self.scale

    def get_pos_with_offset(self, offset):
        return self.position + np.asarray(offset) * self.scale

    def on_transform_update(self):
        raise NotImplementedError


class Padded:
    def __init__(self, padding=(0, 0), **_):
        self.padding = padding
        if isinstance(padding, (int, float)):
            self.padding = (padding, padding)


##────────────────────────────────────────────────────────────────────────────}}}
### {{{           --     network topology and content helpers     --
def get_tu_grid_layout(network, node_type='translation'):
    """return a list of lists of TUs, ordered by topological order of the nodes
    of type node_type (default: translation)"""
    cnodes = network.compute_graph[network.compute_graph['type'] == node_type]

    translation_node_to_tu_id = {}
    for i, t in cnodes['cdg_input'].items():
        tu_ids = [network.central_dogma_graph.loc[tu]['tu_id'][0] for tu in t]
        translation_node_to_tu_id[i] = tu_ids

    topo_order = network.topological_order(cnodes.index.tolist())

    # now that we have the columns we also need to order the rows:
    # we will do that by starting from the last column and going upstream,
    # making sure we order col n-1 by "upstream-ness" to col n
    # (network.is_upstream_of(node1, node2) returns True if node1 is upstream of node2)
    fully_ordered = [[] for _ in range(len(topo_order))]
    fully_ordered[-1] = topo_order[-1]
    for col in range(len(topo_order) - 2, -1, -1):
        nextcol = topo_order[col + 1]
        fully_ordered[col] = []
        for nxt in nextcol:
            for node in topo_order[col]:
                if network.compute_node_is_upstream_of(node, nxt):
                    fully_ordered[col].append(node)
        remaining = [n for n in topo_order[col] if n not in fully_ordered[col]]
        fully_ordered[col].extend(remaining)

    final = [[translation_node_to_tu_id[n] for n in col] for col in fully_ordered]
    # flatten inner lists
    final = [[tu for tu_list in col for tu in tu_list] for col in final]
    return final


import pandas as pd

def get_tu_informations(network):
    tus = []
    cotx_prots = network.get_inverted_input_proteins()
    sources = network.compute_graph[network.compute_graph['type'] == 'source']
    for s, src in sources.iterrows():
        plasmid_name = '_'.join(src['source_id'].split('_')[:-1])
        tu_cdgs = network.central_dogma_graph.loc[src['cdg_output']]
        tus_from_plasmid = []
        is_in_l2 = len(tu_cdgs) > 1

        for p, (_, tu_row) in enumerate(tu_cdgs.iterrows()):
            assert len(tu_row['tu_id']) == 1
            tu_id = tu_row['tu_id'][0]
            tu_name = '_'.join(tu_id.split('_')[:-1])
            content = tu_row['content']
            cotx = [g for g in content if g in cotx_prots]
            if len(cotx) > 1:
                raise ValueError(f'{tu_name} contains more than one marker??')
            ismarker = len(cotx) == 1
            if ismarker:
                cotx = cotx[0]
            tus_from_plasmid.append({
                'tu_name': tu_name,
                'tu_id': tu_id,
                'cotx_marker': cotx if ismarker else None,
                'is_marker': ismarker,
                'plasmid_name': plasmid_name,
                'in_l2': is_in_l2,
                'position_in_plasmid': p,
                'number_of_tu_in_plasmid': len(tu_cdgs),
            })

        # now check if the plasmid is an aggregation
        upstream_node_id = src['input_from'][0][0]
        upstream_node_type = network.compute_graph.at[upstream_node_id, 'type']
        if upstream_node_type == 'aggregation':
            # if it is, we need to check the ratios
            ratios = network.compute_graph.at[upstream_node_id, 'extra']['ratios']
            for i, tu in enumerate(tus_from_plasmid):
                tu['aggregation_ratio'] = ratios[i]
                tu['aggregation_node_id'] = upstream_node_id
        tus.extend(tus_from_plasmid)
    tudf = pd.DataFrame(tus)
    tudf['aggregation_node_id'] = tudf['aggregation_node_id'].fillna(-1)
    tudf['aggregation_node_id'] = tudf['aggregation_node_id'].astype(int)
    tudf['in_aggregation'] = tudf['aggregation_node_id'] != -1
    # ratios are in fraction, but we want them normalized to lowest value (per aggregation)
    tudf['aggregation_ratio_norm'] = tudf.groupby('aggregation_node_id')['aggregation_ratio'].transform(
        lambda x: x / x.min()
    )
    marker_rows = tudf[tudf['is_marker']]
    tudf.loc[marker_rows.index, 'marker_ratio'] = marker_rows['aggregation_ratio_norm'].values
    tudf['marker_ratio'] = tudf.groupby('aggregation_node_id')['marker_ratio'].transform('max')
    tudf['aggregation_ratio_label'] = tudf.apply(
        lambda r: f'{r.aggregation_ratio_norm:.0f}:{r.marker_ratio:.0f}',
        axis=1,
    )
    tudf.loc[~tudf['in_aggregation'], 'aggregation_ratio_label'] = ''
    # propagate cotx_marker to all row with the same plasmid_name
    tudf['marker_in_l2'] = tudf['in_l2'] & tudf['is_marker']
    for _, row in tudf[tudf['is_marker']].iterrows():
        other_in_plasmid = tudf[(tudf['plasmid_name'] == row['plasmid_name']) & (tudf['tu_id'] != row['tu_id'])]
        assert other_in_plasmid['cotx_marker'].isna().all()
        tudf.loc[other_in_plasmid.index, 'cotx_marker'] = row['cotx_marker']
        tudf.loc[other_in_plasmid.index, 'marker_in_l2'] = row['marker_in_l2']
        # and now same for aggregation nodes
        if row['in_aggregation']:
            other_in_agg = tudf[(tudf['aggregation_node_id'] == row['aggregation_node_id']) & (tudf['tu_id'] != row['tu_id'])]
            assert other_in_agg['cotx_marker'].isna().all()
            tudf.loc[other_in_agg.index, 'cotx_marker'] = row['cotx_marker']
    return tudf

# def get_tu_informations(network):
    # aggs = []
    # cotx_prots = network.get_inverted_input_proteins()
    # aggregations = network.compute_graph[network.compute_graph['type'] == 'aggregation']
    # agg_to_cotx = {}
    # for a, agg in aggregations.iterrows():
        # sources_id = [n for n, _ in agg['output_to']]
        # ratios = agg['extra']['ratios']
        # sources = network.compute_graph.loc[sources_id]
        # for i, (s, src) in enumerate(sources.iterrows()):
            # plasmid_name = '_'.join(src['source_id'].split('_')[:-1])
            # plamid_ratio = ratios[i]
            # tu_cdgs = network.central_dogma_graph.loc[src['cdg_output']]
            # for _, tu_row in tu_cdgs.iterrows():
                # assert len(tu_row['tu_id']) == 1
                # tu_id = tu_row['tu_id'][0]
                # tu_name = '_'.join(tu_id.split('_')[:-1])
                # content = tu_row['content']
                # cotx = [g for g in content if g in cotx_prots]
                # if len(cotx) > 1:
                    # raise ValueError(f'{tu_name} contains more than one marker??')
                # ismarker = len(cotx) == 1
                # if ismarker:
                    # agg_to_cotx[a] = cotx[0]
                # aggs.append(
                    # {
                        # 'tu_name': tu_name,
                        # 'tu_id': tu_id,
                        # 'cotx_marker': None,
                        # 'is_marker': ismarker,
                        # 'plasmid_name': plasmid_name,
                        # 'plasmid_ratio': plamid_ratio,
                        # 'source_node_id': s,
                        # 'aggregation_node_id': a,
                    # }
                # )
    # aggdf = pd.DataFrame(aggs)
    # aggdf['cotx_marker'] = aggdf['aggregation_node_id'].apply(lambda a: agg_to_cotx.get(a, None))
    # # ratios are in fraction, but we want them normalized to lowest value (per aggregation)
    # aggdf['plasmid_ratio_norm'] = aggdf.groupby('aggregation_node_id')['plasmid_ratio'].transform(
        # lambda x: x / x.min()
    # )
    # marker_rows = aggdf[aggdf['is_marker']]
    # aggdf.loc[marker_rows.index, 'marker_ratio'] = marker_rows['plasmid_ratio_norm'].values
    # aggdf['marker_ratio'] = aggdf.groupby('aggregation_node_id')['marker_ratio'].transform('max')
    # aggdf['plasmid_ratio_label'] = aggdf.apply(
        # lambda r: f'{r.plasmid_ratio_norm:.0f}:{r.marker_ratio:.0f}',
        # axis=1,
    # )
    # return aggdf


def get_ERN_interactions(network):
    ERNs = network.compute_graph[network.compute_graph['type'] == 'sequestron_ERN']
    ERN_interactions = []
    for i, e in ERNs.iterrows():
        inputs = network.central_dogma_graph.loc[e['cdg_input']]
        assert len(inputs) == 2
        ern_tu_ids = inputs.iloc[0]['tu_id']
        ern_part_name = inputs.iloc[0]['content'][0]
        for src_tu_id in ern_tu_ids:
            rec_tu_ids = inputs.iloc[1]['tu_id']
            tgt_parts = inputs.iloc[1]['content']
            tgt_part_name = [p for p in tgt_parts if ern_part_name in p][0]
            for rec_tu_id in rec_tu_ids:
                ERN_interactions.append(
                    {
                        'src_tu_id': src_tu_id,
                        'src_part_name': ern_part_name,
                        'tgt_tu_id': rec_tu_id,
                        'tgt_part_name': tgt_part_name,
                    }
                )

    ERN_interactions = pd.DataFrame(ERN_interactions)
    return ERN_interactions


def get_interactions(network):
    erns = get_ERN_interactions(network)
    erns['type'] = 'ERN'
    return erns

# get_interactions(network)

# aggdf = get_tu_informations(net)
# layout = get_tu_grid_layout(net)

# add a cotx_id column to aggdf
# we need to find the one row for each aggregation that has is_marker to true
#: it's the row we will use to populate the cotx_id column (with its tu_id)

# add a plasmid_ratio_label that writes ratio_norm : {ratio_norm of the plasmid of this aggregation that is_marker)


##────────────────────────────────────────────────────────────────────────────}}}
### {{{               --     small tools, helpers     --


def remove_empty_strings(l):
    if not isinstance(l, list):
        return l
    res = []
    for e in l:
        if isinstance(e, str) and e == '':
            continue
        res.append(remove_empty_strings(e))
    return res


def get_parts_from_tu(lib, tu, enabled_tu_slots):
    l0_names = lib.L1s.loc[tu][enabled_tu_slots].tolist()
    L0_PART_SLOT_NAMES = [f'part_{i}' for i in range(1, 7)]
    part_names = lib.L0s.loc[l0_names][L0_PART_SLOT_NAMES].values.tolist()
    return remove_empty_strings(part_names)


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                   --     data unit primitives     --
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text


def to_display_units(x, ax):
    """Convert x from data units to display units"""
    ppd = 72.0 / ax.figure.dpi
    trans = ax.transData.transform
    return ((trans((1, x)) - trans((0, 0))) * ppd)[1]


class LineDataUnits(Line2D):
    def __init__(self, *args, **kwargs):
        _lw_data = kwargs.pop("linewidth", 1)
        super().__init__(*args, **kwargs)
        self._lw_data = _lw_data

    def _get_lw(self):
        if self.axes is not None:
            return to_display_units(self._lw_data, self.axes)
        return 1

    def _set_lw(self, lw):
        self._lw_data = lw

    _linewidth = property(_get_lw, _set_lw)


class TextDataUnits(Text):
    def __init__(self, *args, **kwargs):
        _fontsize_data = kwargs.pop("fontsize", 10)
        super().__init__(*args, **kwargs)
        self.set_fontsize(_fontsize_data)

    def _update_fontsize_display(self):
        if self.axes:
            super().set_fontsize(to_display_units(self._fontsize_data, self.axes))

    def set_fontsize(self, fontsize):
        self._fontsize_data = fontsize

    def draw(self, renderer):
        self._update_fontsize_display()
        super().draw(renderer)


class FancyBboxPatchDataUnits(patches.FancyBboxPatch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# class FancyBboxPatchDataUnits(patches.FancyBboxPatch):
# def __init__(self, *args, **kwargs):
# self._lw_data = kwargs.pop("linewidth", 1)
# super().__init__(*args, **kwargs)
# self.set_linewidth(self._lw_data)

# def set_linewidth(self, lw):
# self._lw_data = lw

# def get_linewidth(self):
# if self.axes is not None:
# return to_display_units(self._lw_data, self.axes)
# return 1


# def draw(self, renderer):
# super().draw(renderer)

##────────────────────────────────────────────────────────────────────────────}}}
### {{{                  --     Config Reference system     --

# reference system in the config:
# resolve references by looking into nested dicts.
# if a value is a string, we need to check if it contains local or global references
# global references are easy: replace the string section between $$ by $$path/to/var$$
# local references are a bit more tricky:
# when we find one at a given nesting level, we need to go back level by level until we find
# a dict with a key that matches the reference. then we replace the reference by the value

# Syntax:
# $$path/to/var$$ (global ref)
# %%local_var_name%% (local ref, has priority over global ones by default)

DEFAULT_MARKERS = {
    'local': ('%%', '%%'),
    'global': ('$$', '$$'),
}
DEFAULT_FILTER_RULES = {
    'local': r'(%%[^\%]+%%)',
    'global': r'(\$\$[^\$]+\$\$)',
}
DEFAULT_REFERENCE_RESOLVE_SEQUENCE = ('local', 'global')


def get_ref_type(string, detect_types, markers=DEFAULT_MARKERS):
    for mtype in detect_types:
        mark = markers[mtype]
        if string.startswith(mark) and string.endswith(mark) and len(string) > len(mark) * 2:
            return mtype, string[2:-2]
    else:
        return None, string


def get_dict_at(path, d):
    pathsequence = filter(None, path.strip('/').split('/'))
    for p in pathsequence:
        d = d.get(p, None)
        if d is None:
            return None
    return d


def replace_by_ref(refstr, path, global_params, ref_type=None, context=None):
    if context is None:
        context = {}
    if ref_type == 'global':
        return get_dict_at(refstr, global_params)
    elif ref_type == 'local':
        subpaths = path.strip('/').split('/')
        if refstr in context:
            return context[refstr]
        for i in range(len(subpaths), -1, -1):
            testpath = '/'.join(subpaths[:i])
            d = get_dict_at(testpath, global_params)
            if d is not None:
                if refstr in d:
                    return d[refstr]
            else:
                raise ValueError(f'Could not find {testpath} in {global_params}')
    else:
        return refstr


def resolve_reference(
    string,
    path,
    global_params,
    ref_type=None,
    resolve_type='local',
    filter_rules=DEFAULT_FILTER_RULES,
    context=None,
):
    if context is None:
        context = {}
    chunks = filter(None, re.split(filter_rules[resolve_type], string))
    type_innerchunk_pairs = [get_ref_type(c, detect_types=(resolve_type,)) for c in chunks]
    nrefs = sum([1 for c in type_innerchunk_pairs if c[0] == resolve_type])
    if nrefs == 0:
        if ref_type is None or resolve_type != ref_type:
            return string, False
        resolved = replace_by_ref(string, path, global_params, ref_type=ref_type, context=context)
        if resolved is None:
            # raise ValueError(f'Could not resolve {string} at {path}, as a {ref_type} reference')
            return string, False
        return resolved, True
    else:
        resolved_chunks = [
            resolve_reference(
                c, path, global_params, ref_type=t, resolve_type=resolve_type, context=context
            )
            for t, c in type_innerchunk_pairs
        ]
        newstr = ''.join([c[0] for c in resolved_chunks])
        had_ref = any([c[1] for c in resolved_chunks])
        return newstr, had_ref


def resolve_references(
    current_params,
    path='',
    global_params=None,
    context=None,
    max_recursion_depth=5,
    resolve_type_sequence=DEFAULT_REFERENCE_RESOLVE_SEQUENCE,
):
    if global_params is None:
        global_params = current_params
    if context is None:
        context = {}

    def _impl(
        current_params, path=path, resolve_type=None, max_recursion_depth=max_recursion_depth
    ):
        resolved_params = {}
        had_ref = False
        for key, value in current_params.items():
            if isinstance(value, str):
                result, had_ref = resolve_reference(
                    value, path, global_params, resolve_type=resolve_type, context=context
                )
                resolved_params[key] = result
            elif isinstance(value, dict):
                resolved_params[key] = _impl(
                    value,
                    path=f'{path}/{key}',
                    resolve_type=resolve_type,
                )
            else:
                resolved_params[key] = value

        if max_recursion_depth <= 0 or not had_ref:
            return resolved_params
        else:
            return _impl(
                resolved_params,
                path=path,
                max_recursion_depth=max_recursion_depth - 1,
                resolve_type=resolve_type,
            )

    resolved_params = current_params
    for resolve_type in resolve_type_sequence:
        resolved_params = _impl(
            resolved_params,
            resolve_type=resolve_type,
        )
    return resolved_params


SimpleExampleConf = {
    'theme': {
        'fontname': 'Roboto',
        'colors': {
            'neutral_grey': '#EEEEEE',
            'red': '#FF0000',
            'green': '#00FF00',
            'blue': '#0000FF',
            'colfont': '%%fontname%%',
        },
    },
    'random_param': '$$theme/colors/colfont$$',
    'default': {
        'base_color': 'neutral_grey',
        'facecolor': 'a_$$theme/colors/%%base_color%%$$',
        'edgecolor': '%%facecolor%%+++%%random_param%%!!!!',
        'linewidth': 0.5,
    },
    'red': {
        'base_color': 'red',
    },
}

resolved_config = resolve_references(SimpleExampleConf)
resolved_config

##────────────────────────────────────────────────────────────────────────────}}}
### {{{                        --     GraphicsResource    --


class GraphicsResource:
    """
    A class that loads an img (currently only from an svg file) and can
    generate a matplotlib collection from it, with the required scaling and
    colors.

    Parts can have the following tunable parameters passed to get_collection:
    - main color
    - secondary color
    - edge color
    - line width

    They also define a width and height and [TODO] maybe a text position
    defined by a text marker in the svg file?
    """

    def __init__(self, svgpath, ppi=1.0):
        ppi = float(ppi)
        self.width = 0
        self.height = 0
        self.paths = []
        self.linewidths = []
        self.facecolors = []
        self.edgecolors = []
        self.maincolor_ids = []
        self.secondarycolor_ids = []

        # check if svg file exists
        if not svgpath.exists():
            return

        try:
            svgfile = open(svgpath, 'r')
            tree = etree.parse(StringIO(svgfile.read()))
        except:
            raise ValueError(f'Could not open svg file {svgpath}')

        root = tree.getroot()
        self.width = int(re.match(r'\d+', root.attrib['width']).group())
        self.height = int(re.match(r'\d+', root.attrib['height']).group())
        self.width = self.width / ppi
        self.height = self.height / ppi
        path_elems = root.findall('.//{http://www.w3.org/2000/svg}path')
        self.paths, self.linewidths = [], []
        for elem in path_elems:
            p = parse_path(elem.attrib['d'])
            p.vertices /= ppi
            # flip y axis
            p.vertices[:, 1] = self.height - p.vertices[:, 1]
            self.paths.append(p)
            self.linewidths.append(elem.attrib.get('stroke_width', 1))

        self.facecolors = [elem.attrib.get('fill', 'none') for elem in path_elems]
        self.edgecolors = [elem.attrib.get('stroke', 'none') for elem in path_elems]
        self.maincolor_ids = [i for i, c in enumerate(self.facecolors) if c == '#0000FF']
        self.secondarycolor_ids = [i for i, c in enumerate(self.facecolors) if c == '#00FF00']
        self.size = np.asarray((self.width, self.height))

    def get_collection(self, main_color='k', secondary_color='k', edgecolor=None, linewidth=None):
        facecolors = [
            main_color
            if i in self.maincolor_ids
            else secondary_color
            if i in self.secondarycolor_ids
            else c
            for i, c in enumerate(self.facecolors)
        ]
        edgecolors = [edgecolor if edgecolor else c for c in self.edgecolors]
        linewidths = [linewidth if linewidth else 0 for l in self.linewidths]

        collection = mpl.collections.PathCollection(
            self.paths,
            edgecolors=edgecolors,
            facecolors=facecolors,
            capstyle='round',
            linewidths=linewidths,
        )

        return collection

    def __repr__(self):
        return f'GraphicsResource({self.width}, {self.height})'


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                            --     SVGArtist     --


class SVGArtist(Positionable):
    def __init__(self, svgpath, ppi=1.0, **kwargs):
        Positionable.__init__(self, **kwargs)
        self.resource = GraphicsResource(svgpath, ppi=ppi)

    def on_transform_update(self):
        self.size = np.asarray((self.resource.width, self.resource.height))

    def draw(
        self,
        ax,
        main_color='k',
        secondary_color='k',
        edgecolor=None,
        linewidth=None,
        rotation=0,
        origin=(0, 0),
        zorder=2,
        **_,
    ):
        collection = self.resource.get_collection(main_color, secondary_color, edgecolor, linewidth)

        pos = self.get_bottom_left_position()
        pos -= self.size * self.scale * origin

        collection.set_transform(
            mpl.transforms.Affine2D().rotate_deg(rotation).scale(self.scale).translate(*pos)
            + ax.transData
        )
        collection.set_zorder(zorder)
        ax.add_artist(collection)

    def __repr__(self):
        return f'SVGArtist({self.resource})'


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                           --     Part    --
class Part(SVGArtist, Configurable):
    def __init__(self, lib, part_name, **kwargs):

        self.lib = lib
        self.part_name = part_name
        self.part_row = lib.parts.loc[part_name]
        self.part_category = self.part_row['category']
        # check if we have an svg. there could be a generic one "$category.svg"
        # or a specific one "$category.$partname.svg"
        # priority is given to the specific one
        self.svgpath = PARTS_PATH / f'{self.part_category}.{self.part_name}.svg'
        if not self.svgpath.exists():
            self.svgpath = PARTS_PATH / f'{self.part_category}.svg'

        SVGArtist.__init__(self, self.svgpath, **kwargs)
        priorities = ['default', self.part_category, f'{self.part_category}.{self.part_name}']
        Configurable.__init__(self, 'Part', priorities=priorities, **kwargs)
        self.prepare(**self.local_params)

    def prepare(self, anchor_points=None, **_):
        anchor_points = anchor_points or [(0.5, 0.5)]
        relative_anchor_points = [np.asarray(p) for p in anchor_points]
        self.anchor_points = [self.get_relative_position(p) for p in relative_anchor_points]

    def on_transform_update(self):
        SVGArtist.on_transform_update(self)
        self.prepare(**self.local_params)

    def get_anchor_positions(self):
        return self.anchor_points

    def draw(self, ax):
        SVGArtist.draw(self, ax, **self.local_params)
        self._draw_impl(ax, **self.local_params)

    def _draw_impl(
        self, ax, label_text=None, label_position=(0.5, 0.5), label_properties=None, **_
    ):
        label_properties = label_properties or {}
        label_position = np.asarray(label_position)

        if label_text:
            labelpos = self.position + self.size * self.scale * label_position
            text = TextDataUnits(*labelpos, label_text, **label_properties)
            ax.add_artist(text)

    def __repr__(self):
        return f'Part({self.part_name}, {self.svgpath})'


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                            --     TU     --
class TU(Positionable, Configurable):
    def __init__(self, lib, tu_name: str, params: dict, **kwargs):

        if not isinstance(tu_name, str):
            raise ValueError(f'TU name must be a string, got {tu_name} instead')

        self.lib = lib
        self.tu_name = tu_name
        self.tu_row = lib.L1s.loc[tu_name]

        Configurable.__init__(self, 'TU', params, priorities=['default', tu_name])
        Positionable.__init__(self, **ut.updated_dict(kwargs, self.local_params))

        self.enabled_tu_slots = self.local_params.get('enabled_slots', DEFAULT_ENABLED_TU_SLOTS)
        self.part_names = get_parts_from_tu(lib, tu_name, enabled_tu_slots=self.enabled_tu_slots)
        self.parts = [[Part(lib, p, params=self.all_params) for p in ps] for ps in self.part_names]
        self.parts_dict = {p.part_name: p for ps in self.parts for p in ps}

        self.prepare(**self.local_params)

    def on_transform_update(self):
        self.prepare(**self.local_params)

    def prepare(
        self,
        slot_widths=None,
        display_empty_slots=None,
        rescale_parts=None,
        parts_spacing=None,
        default_part_width=1,
        padding=(0, 0),
        **_,
    ):
        # TODO: should use a wrapper here

        slot_widths = slot_widths or {}
        xpos = self.position[0] + padding[0]
        ypos = self.position[1] + padding[1]
        for slot_name, parts in zip(self.enabled_tu_slots, self.parts):
            slot_target_width = slot_widths.get(slot_name, default_part_width)
            if len(parts) == 0:
                if display_empty_slots:
                    xpos += slot_target_width + parts_spacing
                continue
            total_part_width = sum([p.width for p in parts]) + parts_spacing * (len(parts) - 1)
            if rescale_parts == 'always' or (
                rescale_parts == 'if_too_big' and total_part_width > slot_target_width
            ):
                part_scale = slot_target_width / total_part_width
            else:
                part_scale = 1
            actual_width = max(total_part_width * part_scale, slot_widths.get(slot_name, 0))
            prev_xpos = xpos
            for part in parts:
                part.set_transform((xpos, ypos), scale=part_scale)
                xpos += part.width * part_scale + parts_spacing
            xpos = prev_xpos + actual_width
        self.size = np.array((xpos - self.position[0], ypos - self.position[1]))

    def draw(
        self,
        ax,
    ):
        self._draw_impl(ax, **self.local_params)

    def _draw_impl(
        self,
        ax,
        tu_linewidth=1,
        tu_linecolor='k',
        zorder=None,
        **_,
    ):

        for slot_name, parts in zip(self.enabled_tu_slots, self.parts):
            for part in parts:
                part.draw(ax)

        ax.add_line(
            LineDataUnits(
                [self.position[0], self.position[0] + self.size[0]],
                [self.position[1], self.position[1]],
                linewidth=tu_linewidth,
                color=tu_linecolor,
                zorder=zorder,
            )
        )

    def __repr__(self):
        return f'TU({self.tu_name}, {self.part_names})'


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                          --     Wrapper     --
class Wrapper(Positionable, Padded, Configurable):
    def __init__(
        self,
        elements=None,
        **kwargs,
    ):
        Configurable.__init__(self, 'Wrapper', priorities=['default'], **kwargs)
        Positionable.__init__(self, **ut.updated_dict(kwargs, self.local_params))
        Padded.__init__(self, **ut.updated_dict(kwargs, self.local_params))
        self.elements = elements or []
        if not isinstance(self.elements, list):
            self.elements = [self.elements]
        self.prepare(**self.local_params)

    def on_transform_update(self):
        self.prepare(**self.local_params)

    def prepare(self, content_align=('center', 'center'), **_):
        self.inner_position = np.asarray(self.position) + np.asarray(self.padding)
        self.inner_size = np.asarray(self.size) - 2 * np.asarray(self.padding)
        self.content_align = content_align

        for elt in self.elements:
            # assumes elt origin is left center (true for TUs)
            elt_width, elt_height = elt.size

            # check if elt has a relative_position property
            if hasattr(elt, 'relative_position') and elt.relative_position is not None:
                xelt, yelt = elt.relative_position
                xelt = self.inner_position[0] + self.inner_size[0] * xelt
                yelt = self.inner_position[1] + self.inner_size[1] * yelt
                elt.set_transform((xelt, yelt), scale=1.0)
                continue

            if self.content_align[1] == 'center':
                yelt = self.inner_position[1] + self.inner_size[1] / 2
            else:
                raise ValueError(f'Unknown vertical alignment {self.content_align[1]}')

            if self.content_align[0] == 'left':
                xelt = self.inner_position[0]
            elif self.content_align[0] == 'right':
                xelt = self.inner_position[0] + self.inner_size[0] - elt_width
            elif self.content_align[0] == 'center':
                xelt = self.inner_position[0] + self.inner_size[0] / 2 - elt_width / 2
            else:
                raise ValueError(f'Unknown horizontal alignment {self.content_align[0]}')
            elt.set_transform((xelt, yelt), scale=1.0)

    def add_elements(self, elements):
        if not isinstance(elements, list):
            elements = [elements]
        self.elements.extend(elements)
        self.prepare(**self.local_params)

    def draw(self, ax):
        self._draw_impl(ax, **self.local_params)

    def _draw_impl(
        self,
        ax,
        display_border=True,
        border_properties=None,
        **_,
    ):
        if self.position is None or self.size is None:
            raise ValueError('Wrapper position and size must be defined to draw it')

        if display_border:
            border_properties = border_properties or {}
            lw = border_properties.pop('lw', 0.5)
            lw = to_display_units(border_properties.pop('linewidth', lw), ax)
            linestyle = border_properties.pop('linestyle', '-')

            if isinstance(linestyle, tuple):
                spacing, (on, off) = linestyle

                # linestyle = (
                # to_display_units(spacing, ax),
                # (to_display_units(on, ax), to_display_units(off, ax)),
                # )

            b = FancyBboxPatchDataUnits(
                self.get_bottom_left_position(),
                self.size[0],
                self.size[1],
                linewidth=lw,
                linestyle=linestyle,
                transform=ax.transData,
                **border_properties,
            )
            ax.add_patch(b)

        for elt in self.elements:
            elt.draw(ax)


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                           --     Label     --
class Label(Positionable, Configurable, Padded):
    def __init__(self, label_type, text, **kwargs):
        if isinstance(label_type, str):
            label_type = [label_type]
        priorities = ['default'] + label_type
        Configurable.__init__(self, 'Label', priorities=priorities, **kwargs)
        Padded.__init__(self, **ut.updated_dict(kwargs, self.local_params))
        Positionable.__init__(self, **ut.updated_dict(kwargs, self.local_params))
        self.text = text

    def on_transform_update(self):
        pass

    def draw(self, ax):
        self._draw_impl(ax, **self.local_params)

    def _draw_impl(
        self,
        ax,
        logo_max_size=(15, 15),
        logo_min_size=(10, 10),
        logo_svg=None,
        logo_offset=(0, 0),
        logo_main_color='k',
        text_offset=(0, 0),
        text_properties=None,
        shape_properties=None,
        **_,
    ):
        if shape_properties is not None:
            linewidth = to_display_units(shape_properties.pop('linewidth', 0.25), ax)
            self.bbox = FancyBboxPatchDataUnits(
                self.get_bottom_left_position(),
                width=self.size[0],
                height=self.size[1],
                linewidth=linewidth,
                **shape_properties,
            )
            p = ax.add_patch(self.bbox)
            # add shadow patch


        tpos = self.get_pos_with_offset(text_offset)
        self.textartist = TextDataUnits(*tpos, self.text, **text_properties)
        ax.add_artist(self.textartist)

        if logo_svg is not None:
            svgpath = RESOURCES_PATH / logo_svg
            self.logo = SVGArtist(svgpath, **self.local_params)
            logo_size = self.logo.resource.size
            logo_scale = min(logo_max_size[0] / logo_size[0], logo_max_size[1] / logo_size[1])
            logo_scale = max(logo_scale, logo_min_size[0] / logo_size[0], logo_min_size[1] / logo_size[1])
            self.logo.set_transform(self.get_pos_with_offset(logo_offset), scale=logo_scale)
            self.logo.draw(ax, zorder=3, main_color=logo_main_color)


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                        --     GridLayout     --
class GridSpecItem:
    def __init__(self, grid_spec, row_slice, col_slice):
        self.grid_spec = grid_spec
        self.row_slice = row_slice
        self.col_slice = col_slice

    def get_position_and_size(self):
        # Convert integers to slices for uniform handling
        row_slice = (
            self.row_slice
            if isinstance(self.row_slice, slice)
            else slice(self.row_slice, self.row_slice + 1)
        )
        col_slice = (
            self.col_slice
            if isinstance(self.col_slice, slice)
            else slice(self.col_slice, self.col_slice + 1)
        )

        row_start, row_end = row_slice.start or 0, row_slice.stop or self.grid_spec.nrows
        col_start, col_end = col_slice.start or 0, col_slice.stop or self.grid_spec.ncols

        bottom_left, _ = self.grid_spec.get_cell_corner_positions(row_start, col_start)
        _, top_right = self.grid_spec.get_cell_corner_positions(row_end - 1, col_end - 1)

        size = top_right - bottom_left
        return bottom_left, size

    def __repr__(self):
        return f'GridSpecItem({self.row_slice}, {self.col_slice})'


class GridSpec(Positionable):
    def __init__(self, nrows, ncols, cell_size=(1, 1), hspace=0, wspace=0, **kwargs):
        Positionable.__init__(self, **kwargs)
        self.nrows = nrows
        self.ncols = ncols
        self.cell_size = cell_size
        self.width = cell_size[0] * ncols + wspace * (ncols - 1)
        self.height = cell_size[1] * nrows + hspace * (nrows - 1)
        self.hspace = hspace
        self.wspace = wspace
        self.items = []

    def __getitem__(self, item):
        # Handle the slicing and single index access to return a GridSpecItem
        if isinstance(item, tuple):
            row_slice, col_slice = item
            if not isinstance(row_slice, slice):
                row_slice = slice(row_slice, row_slice + 1)
            if not isinstance(col_slice, slice):
                col_slice = slice(col_slice, col_slice + 1)
        else:
            # Single index, convert to a slice
            row_slice = slice(item, item + 1)
            col_slice = slice(0, self.ncols)  # Assuming entire row

        return GridSpecItem(self, row_slice, col_slice)

    def __setitem__(self, item, value):
        self.items.append(value)
        gspecitem = self[item]
        position, size = gspecitem.get_position_and_size()
        value.set_transform(position=position, size=size)

    def get_cell_at(self, position):
        row = int(position[1] // (self.cell_size[1] + self.hspace))
        col = int(position[0] // (self.cell_size[0] + self.wspace))
        return self[row, col]

    def get_cell_corner_positions(self, row, col):
        bottom_left = self.position + np.asarray((col, row)) * (
            self.cell_size + np.asarray((self.wspace, self.hspace))
        )
        top_right = bottom_left + self.cell_size
        return bottom_left, top_right

    def draw(self, ax):
        # rescale ax if needed
        current_xlim = ax.get_xlim()
        current_ylim = ax.get_ylim()
        ax.set_xlim(
            min(current_xlim[0], self.position[0]),
            max(current_xlim[1], self.position[0] + self.width),
        )
        ax.set_ylim(
            min(current_ylim[0], self.position[1]),
            max(current_ylim[1], self.position[1] + self.height),
        )
        for item in self.items:
            item.draw(ax)


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                     --     InteractionLine     --

def rounded_angle_path(path, codes, radius=1, mode='relative'):
    # add intermediate points to make the path rounded
    newpath = []
    newcodes = []
    CURVE4 = mpl.path.Path.CURVE4
    for i in range(0, len(path) - 2, 1):
        print(f'Processing {i}')
        p0, p1, p2 = path[i], path[i + 1], path[i + 2]
        c0, c1, c2 = codes[i], codes[i + 1], codes[i + 2]
        v0 = p1 - p0
        v1 = p2 - p1
        d0 = np.linalg.norm(v0)
        d1 = np.linalg.norm(v1)
        n0 = v0 / np.maximum(d0, 1e-6)
        n1 = v1 / np.maximum(d1, 1e-6)
        r0, r1 = radius * n0, radius * n1
        if mode=='relative':
            r0*=d0
            r1*=d1
        c0 = p1 - n0 * (np.minimum(r0, d0))
        c1 = p1 + n1 * (np.minimum(r1, d1))
        newpath += [p0, c0, c1, p2]
        newcodes += [c0, CURVE4, CURVE4, CURVE4]
        print(f'New path: {newpath}')

    # newpath += path[-2:]
    # newcodes += codes[-2:]
    newpath = np.array(newpath).tolist()
    return newpath, newcodes



##


def quantize_angle(v, possible_angles, round_to=3):
    possible_vectors = np.asarray([np.cos(possible_angles), np.sin(possible_angles)]).T
    distances = np.dot(possible_vectors, v)
    best = possible_vectors[np.argmax(distances)]
    return np.round(best, round_to)


def get_intersection(origin, direction, AB):
    # origin is a 2d vector
    # direction is a 2d vector
    # AB is a 2x2 matrix with the coordinates of the two points defining the line
    # returns the intersection point between the line and the ray, or None if no intersection

    dirnorm = np.linalg.norm(direction)
    if dirnorm == 0:
        return None
    direction = direction / dirnorm
    point1 = AB[0]
    point2 = AB[1]
    v1 = origin - point1
    v2 = point2 - point1
    v3 = np.array([-direction[1], direction[0]])
    t1 = np.cross(v2, v1) / np.dot(v2, v3)
    t2 = np.dot(v1, v3) / np.dot(v2, v3)
    if t1 >= 0.0 and t2 >= 0.0 and t2 <= 1.0:
        return origin + t1 * direction
    else:
        return None


def get_cell_corners(bottom_left, cell_size, wspacing, hspacing):
    return np.asarray(
        [
            bottom_left - np.asarray((wspacing / 2, hspacing / 2)),  # BL
            bottom_left + np.array([-wspacing / 2, cell_size[1] + hspacing / 2]),  # TL
            bottom_left
            + np.array([cell_size[0] + wspacing / 2, cell_size[1] + hspacing / 2]),  # TR
            bottom_left + np.array([cell_size[0] + wspacing / 2, -hspacing / 2]),  # BR
        ]
    )


class InteractionLine(Configurable):
    def __init__(self, interaction_type, source, target, grid_spec=None, **kwargs):
        Configurable.__init__(
            self, 'InteractionLine', priorities=['default', interaction_type], **kwargs
        )
        self.src = source
        self.tgt = target
        self.grid_spec = grid_spec

    def prepare(self, quantize_angles_to=None, **_):
        if quantize_angles_to is None:
            # only up and down by default
            quantize_angles_to = [np.pi / 2, -np.pi / 2]

        src_anchors = np.atleast_2d(np.asarray(self.src.get_anchor_positions()))
        tgt_anchors = np.atleast_2d(np.asarray(self.tgt.get_anchor_positions()))


        min_dist = np.inf
        src_anchor_id = 0
        tgt_anchor_id = 0
        for i, src_anchor in enumerate(src_anchors):
            for j, tgt_anchor in enumerate(tgt_anchors):
                dist = np.linalg.norm(src_anchor - tgt_anchor)
                if dist < min_dist:
                    min_dist = dist
                    src_anchor_id = i
                    tgt_anchor_id = j



        self.src_anchor = src_anchors[src_anchor_id]
        self.tgt_anchor = tgt_anchors[tgt_anchor_id]

        # find orientation of the line (to the center of the object
        self.src_orientation = self.src_anchor - self.src.get_center_position()
        self.tgt_orientation = self.tgt_anchor - self.tgt.get_center_position()

        # quantize to the given angles
        self.src_orientation = quantize_angle(self.src_orientation, quantize_angles_to)
        self.tgt_orientation = quantize_angle(self.tgt_orientation, quantize_angles_to)

        if self.grid_spec is not None:
            src_cell_pos, src_cell_size = self.grid_spec.get_cell_at(
                self.src.position
            ).get_position_and_size()
            tgt_cell_pos, tgt_cell_size = self.grid_spec.get_cell_at(
                self.tgt.position
            ).get_position_and_size()
            # now we need to find the point of intersection between src_orientation
            # and any of the 4 sides of the cell (+ 0.5hspace and + 0.5wspace)
            src_corners = get_cell_corners(
                src_cell_pos, src_cell_size, self.grid_spec.wspace, self.grid_spec.hspace
            )
            tgt_corners = get_cell_corners(
                tgt_cell_pos, tgt_cell_size, self.grid_spec.wspace, self.grid_spec.hspace
            )
            src_intersections = [
                get_intersection(self.src_anchor, self.src_orientation, np.asarray([c1, c2]))
                for c1, c2 in zip(src_corners, src_corners[1:] + [src_corners[0]])
            ]
            tgt_intersections = [
                get_intersection(self.tgt_anchor, self.tgt_orientation, np.asarray([c1, c2]))
                for c1, c2 in zip(tgt_corners, tgt_corners[1:] + [tgt_corners[0]])
            ]

            # pick the first intersection found
            src_intersection = next(filter(None, src_intersections), None)
            tgt_intersection = next(filter(None, tgt_intersections), None)

            assert src_intersection is not None, f'No intersection found for {self.src}'
            assert tgt_intersection is not None, f'No intersection found for {self.tgt}'

            # for now we'll simply create a path that goes:
            # src_anchor -> src_intersection -> tgt_intersection -> tgt_anchor

            self.path = mpl.path.Path(
                [
                    self.src_anchor,
                    src_intersection,
                    tgt_intersection,
                    self.tgt_anchor,
                ],
                codes=[1, 2, 2, 2],
            )
        else:
            self.path = mpl.path.Path(
                [
                    self.src_anchor,
                    self.src_anchor + self.src_orientation * 10,
                    self.tgt_anchor + self.tgt_orientation * 10,
                    self.tgt_anchor,
                ],
                codes=[1, 2, 2, 2],
            )
            # print(f'Old path: {self.path}')
            # path, codes = self.path.vertices, self.path.codes
            # rpath, rcodes = rounded_angle_path(path, codes, radius=1, mode='absolute')
            # self.path = mpl.path.Path(rpath, codes=rcodes)
            # print(f'New path: {self.path}')

    def draw(self, ax):
        self._draw_impl(ax, **self.local_params)

    def _draw_impl(
        self,
        ax,
        color='#777',
        linewidth=0.5,
        zorder=0,
        **_,
    ):
        self.prepare(**self.local_params)

        line = mpl.patches.FancyArrowPatch(
            path=self.path,
            arrowstyle=mpl.patches.ArrowStyle(
                '|-|', widthA=0.0, angleA=None, widthB=3, angleB=None
            ),
            linewidth=linewidth,
            color=color,
            zorder=zorder,
        )

        ax.add_patch(line)



##────────────────────────────────────────────────────────────────────────────}}}
### {{{                     --     Network Gene Plot     --
class NetworkScene(Positionable, Configurable):
    def __init__(self, network: bc.Network, params: dict, **kwargs):
        self.log = log.getLogger('NetworkScene')
        Configurable.__init__(self, 'NetworkScene', params=params, **kwargs)
        Positionable.__init__(self, **ut.updated_dict(kwargs, self.local_params))
        self.network = network
        self.full_net_name = f'{self.network.metadata["from_xp"]}/{self.network.name}'
        self.tus = {}
        self.parts = {}

        self.interactions = []

        self.smart_grid(**self.local_params)
        self.size = (self.grid.width, self.grid.height)



    def smart_grid(
        self, row_spacing=20, col_spacing=20, show_cotx_tu=False, cell_size=(200, 70), min_rows=3, min_cols=3, **_
    ):
        layout = get_tu_grid_layout(self.network)
        tudf = get_tu_informations(self.network)
        interactions = get_interactions(self.network)
        if not show_cotx_tu:
            markers_id = tudf[tudf['is_marker']]['tu_id'].tolist()
            tudf = tudf[tudf['is_marker'] == False]
            layout = [[tu for tu in col if tu not in markers_id] for col in layout]
            layout = [col for col in layout if len(col) > 0]

        self.grid = GridSpec(
            nrows=max(max([len(col) for col in layout]), min_rows),
            ncols=max(len(layout), min_cols),
            cell_size=cell_size,
            hspace=row_spacing,
            wspace=col_spacing,
            padding=0,
            position=self.position,
        )
        for i, col in enumerate(layout):
            for j, tu_id in enumerate(col):
                tu_row = tudf[tudf['tu_id'] == tu_id]
                assert len(tu_row) == 1, f'When plotting {self.full_net_name}: tu_id {tu_id} not found.'
                # get it as dict
                tu_row = tu_row.to_dict(orient='records')[0]
                tu_name = tu_row['tu_name']
                cotx_text = tu_row['cotx_marker']
                context = {'marker_color': cotx_text}
                cotx_text = MARKER_ALIAS.get(cotx_text, cotx_text)
                tu = TU(self.network.lib, tu_name, params=self.all_params, params_context=context)
                self.tus[tu_id] = tu
                self.parts[tu_id] = tu.parts_dict

                lbls = []

                if tu_row['in_aggregation']:
                    ratio_text = tu_row['aggregation_ratio_label']
                    lbls += [Label(
                        ['aggregation'],
                        f'{ratio_text} {cotx_text}',
                        position=(0.5, -0.2),
                        params=self.all_params,
                        params_context=context,
                    )]
                if tu_row['in_l2']:
                    plsmd_text = cotx_text if tu_row['marker_in_l2'] else tu_row['plasmid_name']
                    lbls += [Label(
                        ['l2'],
                        f'[{tu_row["position_in_plasmid"]}/{tu_row["number_of_tu_in_plasmid"]}] of {plsmd_text}',
                        position=(0.5, -0.2),
                        params=self.all_params,
                        params_context=context,
                    )]

                r, c = self.grid.nrows - j - 1, i
                self.grid[r, c] = Wrapper(
                    elements=[tu, *lbls], params=self.all_params, params_context=context
                )

        for _, row in interactions.iterrows():
            # interactions contain src_tu_id, src_part_name, tgt_tu_id, tgt_part_name
            if row['src_tu_id'] not in self.parts:
                self.log.error(f'When plotting {self.full_net_name}: interaction src TU {row["src_tu_id"]} not found.')
                continue

            if row['tgt_tu_id'] not in self.parts:
                self.log.error(f'When plotting {self.full_net_name}: interaction tgt TU {row["tgt_tu_id"]} not found.')
                continue

            src_part = self.parts[row['src_tu_id']][row['src_part_name']]
            tgt_part = self.parts[row['tgt_tu_id']][row['tgt_part_name']]

            il = InteractionLine(
                row['type'],
                src_part,
                tgt_part,
                grid_spec=None,
                params=self.all_params,
            )
            self.interactions.append(il)

    def one_row_per_plasmid(self, row_spacing=10, col_spacing=10, **_):
        sources = self.network.compute_graph[self.network.compute_graph['type'] == 'source']
        tus = self.network.compute_graph[self.network.compute_graph['type'] == 'tu']
        self.tu_names = []
        self.plasmid_names = []
        for i, src in sources.iterrows():
            tu_names = [
                '_'.join(self.network.central_dogma_graph.loc[t]['tu_id'][0].split('_')[:-1])
                for t in src['cdg_output']
            ]
            self.tu_names.append(tu_names)
            self.plasmid_names.append('_'.join(src['source_id'].split('_')[:-1]))

        max_tus = max([len(tus) for tus in self.tu_names])
        self.grid = GridSpec(
            nrows=len(self.plasmid_names),
            ncols=max_tus,
            cell_size=(150, 50),
            hspace=row_spacing,
            wspace=col_spacing,
            padding=0,
            position=self.position,
        )

        cotx_prots = self.net.get_inverted_input_proteins()
        for i, (pname, tus) in enumerate(zip(self.plasmid_names, self.tu_names)):
            for j, tu_name in enumerate(tus):
                tu = TU(self.network.lib, tu_name, params=self.all_params)
                if self.params.get('label_by_cotx_marker', False):
                    if tu_name in cotx_prots:
                        tu.label_text = cotx_prots[tu_name]
                self.grid[i, j] = Wrapper(elements=[tu], params=self.all_params)

    def draw(self, ax):
        self._draw_impl(ax, **self.local_params)

    def _draw_impl(self, ax, ax_min_padding=50, **_):
        ax.axis('off')
        ax.set_aspect('equal')
        current_xlim = ax.get_xlim()
        current_ylim = ax.get_ylim()
        ax.set_xlim(
            min(current_xlim[0], self.position[0] - ax_min_padding),
            max(current_xlim[1], self.position[0] + self.size[0] + ax_min_padding),
        )
        ax.set_ylim(
            min(current_ylim[0], self.position[1] - ax_min_padding),
            max(current_ylim[1], self.position[1] + self.size[1] + ax_min_padding),
        )
        self.grid.draw(ax)
        for il in self.interactions:
            il.draw(ax)


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                    --     load some networks     --
xpnames = [
    '2023-10-31_ConstraintsV2_4',
    '2023-04-03_Constraints_Pgu_Bleedthrough',
    '2023-03-26_MatrixCsy4',
]

xps = [ut.load_xp(
    xpname,
    lib,
    data_path=['./data/calibrated_data_v3','./data/calibrated_data'],
) for xpname in xpnames]

networks = ut.flatten([list(xp.build_networks()[0]) for xp in xps])

##────────────────────────────────────────────────────────────────────────────}}}
##


fig, ax = plt.subplots(dpi=300, figsize=(10, 10))
netscene = NetworkScene(network, position=(0, 0), params=DEFAULT_CONFIG)
netscene.draw(ax)


##


onedrive = Path(
    '~/Library/CloudStorage/OneDrive-SharedLibraries-MassachusettsInstituteofTechnology/'
).expanduser()
savedir = onedrive / 'Neuromorphic Biocompiler - Documents/Plots/jeanplots/'
savedir.mkdir(parents=True, exist_ok=True)

for i, net in list(enumerate(networks))[22:23]:
    print(f'Plotting {i} - {net.name}')
    fig, ax = plt.subplots(dpi=300, figsize=(10, 10))
    netscene = NetworkScene(net, position=(0, 0), params=DEFAULT_CONFIG)
    netscene.draw(ax)
    netname = f'{net.metadata["from_xp"]}.{net.name}'
    fig.savefig(savedir / f'{netname}.pdf', bbox_inches='tight')
    # close the figure to avoid memory leaks
    plt.show()
    plt.close(fig)
    plt.close('all')



