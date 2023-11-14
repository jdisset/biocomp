### {{{                          --     imports     --
import sys

from dataclasses import dataclass
from typing import List, Tuple

sys.path.append('../../scripts')
from biocomp import utils as ut
import scriptutils as su
import json
import biocomp.datautils as du
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
### {{{                    --     load some networks     --
xpname = '2023-10-10_Cascades_CCv4_2'
lib = su.load_lib()
xp = su.load_xp(
    xpname,
    lib,
    data_path='./data/calibrated_data_v3',
    recipe_path='./recipes',
)

networks, sample_names = xp.build_networks()

recipe_names = [n.name for n in networks]
recipe_names
##────────────────────────────────────────────────────────────────────────────}}}
### {{{         --     get some plasmids from an exmaple network     --

net: bc.Network = networks[-1]
cg = net.compute_graph
cdg = net.central_dogma_graph
sources = cg[cg['type'] == 'source']  # source nodes can be L1 or L2

TU_names = []
plasmid_names = []
for i, src in sources.iterrows():
    tu_names = ['_'.join(cdg.loc[t]['tu_id'][0].split('_')[:-1]) for t in src['cdg_output']]
    TU_names.append(tu_names)
    plasmid_names.append('_'.join(src['source_id'].split('_')[:-1]))

TU_names
plasmid_names


##────────────────────────────────────────────────────────────────────────────}}}

# elements can define a "relative_position" which will override the wrapper's normal positioning system
# and will instead position the element relative to the wrapper's position and size

### {{{                          --     config     --

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
        'label_position': (0.5, -0.2),
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
    'red': '#ffe5de',
    'green': '#efffdd',
    'blue': '#ddf5ff',
    'yellow': '#fffcdc',
    'ir': '#ffe7f4',
    'marroon': '#E4CAB7',
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
    'mMaroon1': BASE_FLUO_COLORS['marroon'],
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
    FLUO_PART_PARAMS[f'fluo_marker.{m}'] = {'main_color': c, 'label_text': MARKER_ALIAS.get(m, m)}
DEFAULT_PART_PARAMS = ut.updated_dict(DEFAULT_PART_PARAMS, FLUO_PART_PARAMS)


DEFAULT_ENABLED_TU_SLOTS = ["promoter", "5'UTR", "gene", "3'UTR", "terminator"]
DEFAULT_TU_PARAMS = {
    # all widths in ax units
    'default': {
        'enabled_slots': DEFAULT_ENABLED_TU_SLOTS,
        'slot_widths': {
            'promoter': 25,
            '5\'UTR': 30,
            'gene': 60,
            '3\'UTR': 1,
            'terminator': 15,
        },
        'display_empty_slots': True,
        'rescale_parts': 'never',  # 'never', 'always', 'if_too_big'
        'parts_spacing': 0.1,
        'default_part_width': 3,
        'tu_linewidth': 1.2,
        'padding': (0.1, 0.1),
        'zorder': -1,
    },
}

DEFAULT_WRAPPER_BORDER_PARAMS = {
    'ec': 'k',
    'boxstyle': 'round,pad=0.0,rounding_size=5',
    'fc': 'none',
    'lw': 0.5,
    'zorder': 1,
    'linestyle': (0, (5, 5)),
    'alpha': 1,
}

DEFAULT_NETWORK_SCENE_PARAMS = {
    'Part': DEFAULT_PART_PARAMS,
    'TU': DEFAULT_TU_PARAMS,
    'Wrapper': {
        'default': {'border': True, 'border_params': DEFAULT_WRAPPER_BORDER_PARAMS},
    },
    'Label_Aggregation': {
        'default': {
            'shape_params': {
                'ec': 'k',
                'fc': 'k',
                'lw': 0.5,
                'zorder': 1,
                'boxstyle': 'round,pad=0.0,rounding_size=5',
                'alpha': 0.1,
            },
        },
    },
}

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
        print(f'ordering col {col}')
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
    aggs = []
    cotx_prots = net.get_inverted_input_proteins()
    aggregations = network.compute_graph[network.compute_graph['type'] == 'aggregation']
    agg_to_cotx = {}
    for a, agg in aggregations.iterrows():
        sources_id = [n for n, _ in agg['output_to']]
        sources = network.compute_graph.loc[sources_id]
        for s, src in sources.iterrows():
            plasmid_name = '_'.join(src['source_id'].split('_')[:-1])
            tu_cdgs = network.central_dogma_graph.loc[src['cdg_output']]
            for _, tu_row in tu_cdgs.iterrows():
                assert len(tu_row['tu_id']) == 1
                tu_id = tu_row['tu_id'][0]
                tu_name = '_'.join(tu_id.split('_')[:-1])
                content = tu_row['content']
                cotx = [g for g in content if g in cotx_prots]
                if len(cotx) > 1:
                    raise ValueError(f'{tu_name} contains more than one marker??')
                ismarker = len(cotx) == 1
                if ismarker:
                    agg_to_cotx[a] = cotx[0]
                aggs.append(
                    {
                        'tu_name': tu_name,
                        'tu_id': tu_id,
                        'cotx_marker': None,
                        'is_marker': ismarker,
                        'plasmid_name': plasmid_name,
                        'source_node_id': s,
                        'aggrefation_node_id': a,
                    }
                )
    aggdf = pd.DataFrame(aggs)
    aggdf['cotx_marker'] = aggdf['aggrefation_node_id'].apply(lambda a: agg_to_cotx.get(a, None))
    return aggdf


aggdf = get_tu_informations(net)
layout = get_tu_grid_layout(net)


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
    'local': '%%',
    'global': '$$',
}
DEFAULT_FILTER_RULES = {
    'local': r'(%%[^\%]+%%)',
    'global': r'(\$\$[^\$]+\$\$)',
    'any': r'(%%[^\%]+%%|\$\$[^\$]+\$\$)',
}
DEFAULT_REFERENCE_RESOLVE_SEQUENCE = ('local', 'global')

def get_dict_at(path, d):
    pathsequence = filter(None, path.strip('/').split('/'))
    for p in pathsequence:
        d = d.get(p, None)
        if d is None:
            return None
    return d

def replace_by_ref(refstr, path, global_params, ref_type=None):
    if ref_type == 'global':
        return get_dict_at(refstr, global_params)
    elif ref_type == 'local':
        subpaths = path.strip('/').split('/')
        print(f'        Trying to resolve local reference {refstr} at {path}')
        for i in range(len(subpaths), -1, -1):
            testpath = '/'.join(subpaths[:i])
            d = get_dict_at(testpath, global_params)
            if d is not None:
                if refstr in d:
                    # print(f'found {refstr} in {d}: {d[refstr]}')
                    return d[refstr]
            else:
                raise ValueError(f'Could not find {testpath} in {global_params}')
    else:
        return refstr


def get_ref_type(string, detect_types, markers=DEFAULT_MARKERS):
    for mtype in detect_types:
        mark = markers[mtype]
        if string.startswith(mark) and string.endswith(mark) and len(string) > len(mark) * 2:
            return mtype, string[2:-2]
    else:
        return None, string


def resolve_reference(
    string,
    path,
    global_params,
    ref_type=None,
    resolve_type='local',
    filter_rules=DEFAULT_FILTER_RULES,
):
    chunks = filter(None, re.split(filter_rules[resolve_type], string))
    type_innerchunk_pairs = [get_ref_type(c, detect_types=(resolve_type,)) for c in chunks]
    nrefs = sum([1 for c in type_innerchunk_pairs if c[0] == resolve_type])
    if nrefs > 0 or ref_type is not None:
        print(f'found {nrefs} subrefs in ({ref_type}, "{string}")')
        print(f'    subrefs are: {list(type_innerchunk_pairs)}')
    if nrefs == 0:
        if ref_type is None or resolve_type != ref_type:
            return string, False
        print(f'    no refs found in {string}, resolve as {ref_type}')
        resolved = replace_by_ref(string, path, global_params, ref_type=ref_type)
        if resolved is None:
            raise ValueError(f'Could not resolve {string} at {path}, as a {ref_type} reference')
        print(f'    resolved {ref_type} reference "{string}" to {resolved}')
        return resolved, True
    else:
        resolved_chunks = [
            resolve_reference(c, path, global_params, ref_type=t, resolve_type=resolve_type)
            for t, c in type_innerchunk_pairs
        ]
        newstr = ''.join([c[0] for c in resolved_chunks])
        had_ref = any([c[1] for c in resolved_chunks])
        return newstr, had_ref


def resolve_references(
    current_params,
    path='',
    global_params=None,
    max_recursion_depth=5,
    resolve_type_sequence=DEFAULT_REFERENCE_RESOLVE_SEQUENCE,
):
    def _impl(
        current_params, path='', global_params=None, max_recursion_depth=5, resolve_type=None
    ):


        if global_params is None:
            global_params = current_params
        resolved_params = {}
        had_ref = False
        for key, value in current_params.items():
            if isinstance(value, str):
                result, had_ref = resolve_reference(
                    value, path, global_params, resolve_type=resolve_type
                )
                resolved_params[key] = result
                print()
            elif isinstance(value, dict):
                resolved_params[key] = _impl(
                    value,
                    path=f'{path}/{key}',
                    global_params=global_params,
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
                global_params=global_params,
                max_recursion_depth=max_recursion_depth - 1,
                resolve_type=resolve_type,
            )

    resolved_params = current_params
    for resolve_type in resolve_type_sequence:
        resolved_params = _impl(
            resolved_params,
            path=path,
            global_params=global_params,
            max_recursion_depth=max_recursion_depth,
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
        'edgecolor': '%%facecolor%%_%%random_param%%',
        'linewidth': 0.5,
    },
    'red': {
        'base_color': 'red',
    },
}

resolved_config = resolve_references(SimpleExampleConf)
resolved_config

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                       --     Configurable     --
##


class Configurable:
    def __init__(self, section_name, params=None, priorities=None, **_):
        self.priorities = priorities
        self.section_name = section_name
        self.update_params(params)

    def update_params(self, params, **kwargs):
        self.all_params = params or {}
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
        self.local_params = self.resolve_references(self.local_params)


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

    def on_transform_update(self):
        raise NotImplementedError


class Padded:
    def __init__(self, padding=(0, 0), **_):
        self.padding = padding
        if isinstance(padding, (int, float)):
            self.padding = (padding, padding)


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

    def get_collection(self, main_color='w', secondary_color='w', edgecolor=None, linewidth=None):
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
        main_color='w',
        secondary_color='w',
        edgecolor=None,
        linewidth=None,
        rotation=0,
        origin=(0, 0),
        zorder=0,
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

### {{{                           --     Label     --
class Label(Positionable, Configurable, Padded):
    def __init__(self, label_type, text, **kwargs):
        Configurable.__init__(self, label_type, **kwargs)
        Padded.__init__(self, **ut.updated_dict(kwargs, self.local_params))
        Positionable.__init__(self, **ut.updated_dict(kwargs, self.local_params))
        self.text = text

    def on_transform_update(self):
        pass

    def draw(self, ax):
        self._draw_impl(ax, **self.local_params)

    def _draw_impl(self, ax, shape_params=None, **_):
        # add a (fancy) rectangle
        if shape_params is not None:
            shape_params = ut.updated_dict(shape_params, self.local_params)
            r = patches.FancyBboxPatch(
                self.position,
                self.size[0],
                self.size[1],
                **shape_params,
            )
            ax.add_patch(r)

        text = TextDataUnits(*self.position, self.text, **self.local_params)
        ax.add_artist(text)


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
### {{{                          --     Wrapper     --
class Wrapper(Positionable, Padded, Configurable):
    def __init__(
        self,
        elements=None,
        **kwargs,
    ):
        Configurable.__init__(self, 'Wrapper', prioriries=['default'], **kwargs)
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
        border=True,
        border_params=DEFAULT_WRAPPER_BORDER_PARAMS,
        **_,
    ):
        if self.position is None or self.size is None:
            raise ValueError('Wrapper position and size must be defined to draw it')

        if border:
            b = patches.FancyBboxPatch(self.position, self.size[0], self.size[1], **border_params)
            ax.add_patch(b)

        for elt in self.elements:
            elt.draw(ax)


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                     --     Network Gene Plot     --
class NetworkScene(Positionable, Configurable):
    def __init__(self, network: bc.Network, params: dict, **kwargs):
        self.network = network
        self.sources = network.compute_graph[network.compute_graph['type'] == 'source']
        self.tus = network.compute_graph[network.compute_graph['type'] == 'tu']

        Configurable.__init__(self, 'NetworkScene', params=params, **kwargs)
        Positionable.__init__(self, **ut.updated_dict(kwargs, self.local_params))

        self.smart_grid(**self.local_params)
        self.size = (self.grid.width, self.grid.height)

    def smart_grid(
        self, row_spacing=20, col_spacing=20, show_cotx_tu=False, cell_size=(200, 70), **_
    ):
        layout = get_tu_grid_layout(self.network)
        aggdf = get_tu_informations(self.network)
        if not show_cotx_tu:
            markers_id = aggdf[aggdf['is_marker'] == True]['tu_id'].tolist()
            aggdf = aggdf[aggdf['is_marker'] == False]
            layout = [[tu for tu in col if tu not in markers_id] for col in layout]
            layout = [col for col in layout if len(col) > 0]

        self.grid = GridSpec(
            nrows=max([len(col) for col in layout]),
            ncols=len(layout),
            cell_size=cell_size,
            hspace=row_spacing,
            wspace=col_spacing,
            padding=0,
            position=self.position,
        )
        for i, col in enumerate(layout):
            for j, tu_id in enumerate(col):
                tu_name = aggdf[aggdf['tu_id'] == tu_id]['tu_name'].tolist()[0]
                tu = TU(self.network.lib, tu_name, params=self.all_params)
                lbl_text = aggdf[aggdf['tu_id'] == tu_id]['cotx_marker'].tolist()[0]
                main_color = get_color_family(lbl_text)
                lbl = Label(
                    ['aggregation', main_color],
                    lbl_text,
                    position=(0.5, -0.2),
                    params=self.all_params,
                )
                self.grid[j, i] = Wrapper(elements=[tu, lbl], params=self.all_params)

    def one_row_per_plasmid(self, row_spacing=10, col_spacing=10, **_):
        self.tu_names = []
        self.plasmid_names = []
        for i, src in self.sources.iterrows():
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


fig, ax = plt.subplots(dpi=300, figsize=(10, 10))
netscene = NetworkScene(net, position=(0, 0), params=DEFAULT_NETWORK_SCENE_PARAMS)
netscene.draw(ax)


##────────────────────────────────────────────────────────────────────────────}}}
