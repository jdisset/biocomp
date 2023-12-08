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

net = networks[-1]
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
    'promoter': {'origin': (0, 0), 'offset': (0.5, 0)},
    'terminator': {'origin': (0, 0)},
    'fluo_marker': {
        'label_position': (0.45, 0.5),
    },
    'ERN': {
        'label_text': 'ERN',
        'label_position': (0.2, 0.5),
        'label_properties': {'ha': 'left', 'va': 'center'},
    },
    'ERN_recog_site_5p': {
        'origin': (0, 0),
        'offset': (0, -0.1),
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


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                       --     Positionable     --
class Positionable:
    def __init__(self, position=(0, 0), size=(1,1), scale=1.0, relative_position=None, margin=0):
        self.position = np.asarray(position)
        self.size = np.asarray(size)
        self.scale = scale
        self.relative_position = relative_position
        self.margin = margin
        if isinstance(margin, (int, float)):
            self.padding = (margin, margin)
    def set_transform(self, position, scale=1.0):
        self.position = np.asarray(position)
        self.scale = scale
    def draw(self, _):
        raise NotImplementedError
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
### {{{                        --     PartLoader   --
class PartResource:
    """
    A class that loads a part (currently only from an svg file) and can
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

    def get_collection(
        self,
        main_color='w',
        secondary_color='w',
        edgecolor=None,
        linewidth=None,
    ):

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
### {{{                           --     Part     --
class PartArtist:
    def __init__(
        self, lib, part_name, part_params=DEFAULT_PART_PARAMS, position=(0, 0), scale=1.0, **kwargs
    ):
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
        # if not self.svgpath.exists():
        # raise ValueError(
        # f'No svg file for part {self.part_name} in category {self.part_category}'
        # )
        self.resource = PartResource(self.svgpath, **kwargs)
        part_params = part_params or {}
        self.part_params = part_params.get('default', {}).copy()
        self.part_params = ut.updated_dict(
            self.part_params, part_params.get(self.part_category, {})
        )
        self.part_params = ut.updated_dict(
            self.part_params, part_params.get(f'{self.part_category}.{self.part_name}', {})
        )
        self.part_params = ut.updated_dict(self.part_params, kwargs)
        self.kwargs = kwargs
        self.position = np.asarray(position)
        self.scale = scale

    def set_position(self, position, scale=1.0):
        self.position = np.asarray(position)
        self.scale = scale

    def draw(self, ax, **kwargs):
        params = ut.updated_dict(self.part_params, kwargs)
        self._draw_impl(ax, **params)

    def _draw_impl(
        self,
        ax,
        main_color,
        secondary_color,
        edgecolor,
        linewidth,
        origin=(0, 0),
        offset=(0, 0),
        rotation=0.0,
        zorder=1,
        label_text=None,
        label_position=(0.5, 0.5),
        label_properties=None,
        **kwargs,
    ):
        x, y = self.position
        label_properties = label_properties or {}
        collection = self.resource.get_collection(main_color, secondary_color, edgecolor, linewidth)
        x += self.resource.width * self.scale * (offset[0] - origin[0])
        y += self.resource.height * self.scale * (offset[1] - origin[1])
        collection.set_transform(
            mpl.transforms.Affine2D().rotate_deg(rotation).scale(self.scale).translate(x, y)
            + ax.transData
        )
        collection.set_zorder(zorder)
        ax.add_artist(collection)
        if label_text:
            text = TextDataUnits(
                x + self.resource.width * self.scale * label_position[0],
                y + self.resource.height * self.scale * label_position[1],
                label_text,
                **label_properties,
            )
            ax.add_artist(text)

    def __repr__(self):
        return f'Part({self.part_name}, {self.svgpath})'


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                            --     TU     --
class TU(Positionable):
    def __init__(self, lib, tu_name, tu_params=DEFAULT_TU_PARAMS, **kwargs):
        super().__init__(**kwargs)

        self.lib = lib
        self.tu_name = tu_name
        self.tu_row = lib.L1s.loc[tu_name]
        # grab enabled_slots from tu_params

        tu_params = tu_params or {}
        self.tu_params = tu_params.get('default', {}).copy()
        self.tu_params = ut.updated_dict(self.tu_params, tu_params.get(tu_name, {}))

        self.enabled_tu_slots = self.tu_params.get('enabled_slots', DEFAULT_ENABLED_TU_SLOTS)
        self.part_names = get_parts_from_tu(lib, tu_name, enabled_tu_slots=self.enabled_tu_slots)

        self.parts = [[PartArtist(lib, p) for p in ps] for ps in self.part_names]
        self.size = np.array([0, 0])

        self.prepare(**self.tu_params)

    def set_transform(self, position, scale=1.0):
        self.position = np.asarray(position)
        self.scale = scale
        self.prepare(**self.tu_params)

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
        slot_widths = slot_widths or {}
        xpos = self.position[0] + padding[0]
        ypos = self.position[1] + padding[1]
        for slot_name, parts in zip(self.enabled_tu_slots, self.parts):
            slot_target_width = slot_widths.get(slot_name, default_part_width)
            if len(parts) == 0:
                if display_empty_slots:
                    xpos += slot_target_width + parts_spacing
                continue
            total_part_width = sum([p.resource.width for p in parts]) + parts_spacing * (
                len(parts) - 1
            )
            if rescale_parts == 'always' or (
                rescale_parts == 'if_too_big' and total_part_width > slot_target_width
            ):
                part_scale = slot_target_width / total_part_width
            else:
                part_scale = 1

            actual_width = max(total_part_width * part_scale, slot_widths.get(slot_name, 0))

            prev_xpos = xpos
            for part in parts:
                part.set_position((xpos, ypos), scale=part_scale)
                xpos += part.resource.width * part_scale + parts_spacing
            xpos = prev_xpos + actual_width

        self.size = np.array((xpos - self.position[0], ypos - self.position[1]))

    def draw(
        self,
        ax,
        **kwargs,
    ):
        # for now scale is ignored
        self._draw_impl(ax, **self.tu_params, **kwargs)

    def _draw_impl(
        self,
        ax,
        tu_linewidth=1,
        tu_linecolor='k',
        zorder=None,
        **kwargs,
    ):

        for slot_name, parts in zip(self.enabled_tu_slots, self.parts):
            for part in parts:
                part.draw(ax, **kwargs)

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

        # Calculate the position and size for the merged cells
        x0, y0, single_width, single_height = self.grid_spec.get_subplot_position(
            row_start, col_start
        )
        x1, _, _, _ = self.grid_spec.get_subplot_position(row_end - 1, col_end - 1)

        total_width = x1 - x0 + single_width
        total_height = single_height * (row_end - row_start) + self.grid_spec.hspace * (
            row_end - row_start - 1
        )

        return x0, y0, total_width, total_height


class GridSpec:
    def __init__(
        self, nrows, ncols, origin=(0, 0), padding=0, cell_size=(1, 1), hspace=0, wspace=0
    ):
        self.nrows = nrows
        self.ncols = ncols
        self.origin = origin
        self.width = cell_size[0] * ncols + wspace * (ncols - 1)
        self.height = cell_size[1] * nrows + hspace * (nrows - 1)
        self.hspace = hspace
        self.wspace = wspace
        self.padding = padding
        if isinstance(padding, (int, float)):
            self.padding = (padding, padding)
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

    def get_subplot_position(self, row, col):
        # Calculate the total space occupied by the horizontal and vertical spaces
        total_hspace = (self.ncols - 1) * self.wspace
        total_vspace = (self.nrows - 1) * self.hspace

        # Calculate the width and height of each subplot
        subplot_width = (self.width - total_hspace) / self.ncols
        subplot_height = (self.height - total_vspace) / self.nrows

        # Calculate the position of the subplot
        x0 = self.origin[0] + (subplot_width + self.wspace) * col
        y0 = self.origin[1] + (subplot_height + self.hspace) * (self.nrows - row - 1)

        # Apply padding
        x0 += self.padding[0]
        y0 += self.padding[1]
        subplot_width -= 2 * self.padding[0]
        subplot_height -= 2 * self.padding[1]

        return x0, y0, subplot_width, subplot_height

    def draw(self, ax):
        for item in self.items:
            item.draw(ax)


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                          --     Wrapper     --

class Div(Positionable):
    def __init__(
        self,
        grid_cells=None,
        padding=(1, 1),
        border=True,
        align=('center', 'center'),
        elements=None,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.grid_cells = grid_cells
        if grid_cells:
            grid_cells.grid_spec.items.append(self)
            x0, y0, width, height = self.grid_cells.get_position_and_size()
            self.position = (x0, y0)
            self.size = (width, height)

        self.padding = padding
        if isinstance(padding, (int, float)):
            self.padding = (padding, padding)

        self.border = border
        self.align = align
        self.elements = elements or []

    def set_transform(self, position, scale=1.0):
        self.position = np.asarray(position)
        self.scale = scale
        self.update_elt_positions()

    def add_elements(self, elements):
        if not isinstance(elements, list):
            elements = [elements]
        self.elements.extend(elements)
        self.update_elt_positions()

    def update_elt_positions(self):
        self.inner_position = np.asarray(self.position) + np.asarray(self.padding)
        self.inner_size = np.asarray(self.size) - 2 * np.asarray(self.padding)

        for elt in self.elements:
            # assumes elt origin is left center (true for TUs)
            elt_width, elt_height = elt.size

            # check if elt has a relative_position property
            if hasattr(elt, 'relative_position'):
                xelt, yelt = elt.relative_position
                xelt = self.inner_position[0] + self.inner_size[0] * xelt
                yelt = self.inner_position[1] + self.inner_size[1] * yelt
                elt.set_transform((xelt, yelt), scale=1.0)
                continue

            if self.align[1] == 'center':
                yelt = self.inner_position[1] + self.inner_size[1] / 2
            else:
                raise ValueError(f'Unknown vertical alignment {self.align[1]}')

            if self.align[0] == 'left':
                xelt = self.inner_position[0]
            elif self.align[0] == 'right':
                xelt = self.inner_position[0] + self.inner_size[0] - elt_width
            elif self.align[0] == 'center':
                xelt = self.inner_position[0] + self.inner_size[0] / 2 - elt_width / 2
            else:
                raise ValueError(f'Unknown horizontal alignment {self.align[0]}')

            elt.set_transform((xelt, yelt), scale=1.0)

    def draw(self, ax):
        if self.position is None or self.size is None:
            raise ValueError('Wrapper position and size must be defined to draw it')
        width, height = self.size
        if self.border:
            border_rect = patches.FancyBboxPatch(
                self.position,
                width,
                height,
                ec="k",
                boxstyle="round,pad=0.0,rounding_size=5",
                fc="none",
                lw=0.5,
                zorder=1,
                linestyle=(0, (5, 5)),
                alpha=1,
            )
            ax.add_patch(border_rect)

        for elt in self.elements:
            elt.draw(ax)


fig, ax = plt.subplots(dpi=300, figsize=(10, 10))
ax.set_aspect('equal')
ax.axis('off')
ax.set_xlim(-10, 110)
ax.set_ylim(-10, 110)

out_w = Div(position=(0, 0), size=(100, 100), padding=5, border=True)
in_w = Div(size=(50, 10), relative_position=(0.5, 0.5), padding=5, border=True)
out_w.add_elements(in_w)

out_w.draw(ax)


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                  --     grid and wrapper system     --

DEFAULT_PART_PARAMS

fig, ax = plt.subplots(dpi=300, figsize=(10, 10))
ax.set_aspect('equal')
ax.axis('off')

ntus = len(TU_names)

gs = GridSpec(nrows=ntus, ncols=1, origin=(0, 0), cell_size=(150, 50), hspace=10, wspace=10)

ax.set_ylim(-10, gs.height + 10)
ax.set_xlim(-10, gs.width + 10)

for i, plasmid in enumerate(TU_names):
    for j, tu_name in enumerate(plasmid):
        tu = TU(lib, tu_name, tu_params=DEFAULT_TU_PARAMS)
        wrapper = Div(gs[i, :], padding=5, border=True)
        wrapper.add_elements(tu)

gs.draw(ax)


##


##────────────────────────────────────────────────────────────────────────────}}}


### {{{                    --     new general system     --
# Base classes:
# - Positionable

# elements:
# - Div
# - SVG
# - Grid

# TUs inherit from Div
# Parts inherit from SVG



##────────────────────────────────────────────────────────────────────────────}}}

alltus = lib.L1s.index.tolist()[:]
fig, ax = plt.subplots(dpi=200, figsize=(10, 10))
ax.set_aspect('equal')
ax.axis('off')

tu_h_spacing = 300
tu_v_spacing = 60

# n_parts_per_cols = 30
# n_cols = len(alltus) // n_parts_per_cols + 1
# n_rows = n_parts_per_cols
# ax.set_xlim(-200, n_cols * tu_h_spacing)
# ax.set_ylim(-150, n_rows * tu_v_spacing)
# for i, tu_name in enumerate(alltus):
# x = i % n_cols
# y = i // n_cols
# tu = TU(lib, tu_name, tu_params=DEFAULT_TU_PARAMS)
# tu.draw(ax, (x * tu_h_spacing, y * tu_v_spacing))
# ax.text(
# x * tu_h_spacing - 10,
# y * tu_v_spacing,
# tu_name,
# fontsize=lw_from_data_units(8, ax),
# ha='right',
# va='center',
# fontname='Roboto',
# )


fig.tight_layout()
# save as pdf
# fig.savefig('all_L1s.pdf')
