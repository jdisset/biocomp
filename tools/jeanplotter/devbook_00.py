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

# get an example network (a bandpass)
net = networks[-1]

cg = net.compute_graph
cdg = net.central_dogma_graph

src = cg[cg['type'] == 'source']  # source nodes can be L1 or L2
TUs = src['cdg_output']  # a list of TUs for each source


TU_names = []
for i, tu in TUs.items():
    tu_names = ['_'.join(cdg.loc[t]['tu_id'][0].split('_')[:-1]) for t in tu]
    print(f'Source {i}: {tu}')
    print(f'  TUs: {tu_names}')
    TU_names.append(tu_names)

##

### {{{               --     small tools, constants, and helpers     --

ERN_COLORS = {
    'Csy4': '#AAAAAA',
    'CasE': '#CCCCCC',
    'PgU': '#EEEEEE',

RESOURCES_PATH = Path('./resources').resolve()
PARTS_PATH = RESOURCES_PATH / 'parts'

ENABLED_TU_SLOTS = ["promoter", "5'UTR", "gene", "3'UTR", "terminator"]

DEFAULT_PART_PARAMS = {
    'default': {
        'main_color': '#EEEEEE',
        'secondary_color': '#EEEEEE',
        'edgecolor': 'k',
        'linewidth': 0.5,
        'origin': ('center', 'left'),
    },
    'promoter': {'origin': ('bottom', 'left')},
    'terminator': {'origin': ('bottom', 'left')},
    'ERN_recog_site_5p': {'origin': ('bottom', 'left')},
    'ERN.Csy4': {'main_color': '#CCCCCC'},
    'ERN.CasE': {'main_color': '#DDDDDD'},
    'ERN.PgU': {'main_color': '#EEEEEE'},
    'ERN_recog_site_5p': {'origin': ('bottom', 'left')},
}

DEFAULT_SLOT_WIDTHS = {
    'promoter': 2,
    '5\'UTR': 2,
    'gene': 6,
    '3\'UTR': 0.1,
    'terminator': 1,
}


def lw_from_data_units(lw, ax):
    length = ax.get_figure().bbox_inches.height * ax.get_position().height
    value_range = np.diff(ax.get_ylim())
    return lw * (length * 72 / value_range)


def remove_empty_strings(l):
    if not isinstance(l, list):
        return l
    res = []
    for e in l:
        if isinstance(e, str) and e == '':
            continue
        res.append(remove_empty_strings(e))
    return res


def get_parts_from_tu(lib, tu, enabled_tu_slots=ENABLED_TU_SLOTS):
    l0_names = lib.L1s.loc[tu][enabled_tu_slots].tolist()
    L0_PART_SLOT_NAMES = [f'part_{i}' for i in range(1, 7)]
    part_names = lib.L0s.loc[l0_names][L0_PART_SLOT_NAMES].values.tolist()
    return remove_empty_strings(part_names)


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                        --     PartArtist     --


class PartArtist:
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

    def __init__(self, svgpath, ppi=10.0):
        ppi = float(ppi)

        tree = etree.parse(StringIO(open(svgpath, 'r').read()))
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
        linewidths = [linewidth if linewidth else l for l in self.linewidths]

        collection = mpl.collections.PathCollection(
            self.paths, edgecolors=edgecolors, linewidths=0, facecolors=facecolors, capstyle='round'
        )

        return collection


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                           --     Part     --


class Part:
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
        if not self.svgpath.exists():
            raise ValueError(
                f'No svg file for part {self.part_name} in category {self.part_category}'
            )
        self.artist = PartArtist(self.svgpath, **kwargs)

    def draw(self, ax, x, y, part_params=DEFAULT_PART_PARAMS, **kwargs):
        """
        Draw a part at a given position and scale, with a given rotation.
        """
        part_params = part_params or {}

        params = part_params.get('default', {}).copy()
        params.update(part_params.get(self.part_category, {}))
        params.update(part_params.get(f'{self.part_category}.{self.part_name}', {}))
        params.update(kwargs)

        print(f'Drawing part {self.part_name} with params {params} (category {self.part_category})')

        self._draw_impl(ax, x, y, **params)

    def _draw_impl(
        self,
        ax,
        x,
        y,
        main_color,
        secondary_color,
        edgecolor,
        linewidth,
        origin,
        scale=1.0,
        rotation=0.0,
    ):
        collection = self.artist.get_collection(main_color, secondary_color, edgecolor, linewidth)

        # origin (vertical, horizontal)

        if origin[0] == 'top':
            pass
        elif origin[0] == 'center':
            y -= (self.artist.height / 2.0) * scale
        elif origin[0] == 'bottom':
            y -= self.artist.height * scale
        else:
            raise ValueError(f'Unknown vertical origin {origin[0]}')

        if origin[1] == 'left':
            pass
        elif origin[1] == 'center':
            x -= (self.artist.width / 2.0) * scale
        elif origin[1] == 'right':
            x -= self.artist.width * scale
        else:
            raise ValueError(f'Unknown horizontal origin {origin[1]}')

        collection.set_transform(
            mpl.transforms.Affine2D().rotate_deg(rotation).scale(scale).translate(x, y)
            + ax.transData
        )
        ax.add_artist(collection)

    def __repr__(self):
        return f'Part({self.part_name}, {self.svgpath})'


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                            --     TU     --
class TU:
    def __init__(self, lib, tu_name, enabled_tu_slots, **_):
        self.lib = lib
        self.tu_name = tu_name
        self.tu_row = lib.L1s.loc[tu_name]
        self.enabled_tu_slots = enabled_tu_slots
        self.part_names = get_parts_from_tu(lib, tu_name, enabled_tu_slots=enabled_tu_slots)
        self.parts = [[Part(lib, p) for p in ps] for ps in self.part_names]

    def draw(
        self,
        ax,
        x,
        y,
        slot_widths=None,
        display_empty_slots=True,
        rescale_parts='never',  # 'never', 'always', 'if_too_big'
        parts_spacing=0.1,
        default_part_width=3,
        tu_linewidth=0.5,
        padding=1,
        **kwargs,
    ):
        slot_widths = slot_widths or {}
        ypos = y
        xpos = x + padding
        for slot_name, parts in zip(self.enabled_tu_slots, self.parts):
            print(slot_name, parts)
            target_width = slot_widths.get(slot_name, default_part_width)
            if len(parts) == 0:
                if display_empty_slots:
                    xpos += target_width + parts_spacing
                continue
            total_part_width = sum([p.artist.width for p in parts]) + parts_spacing * (
                len(parts) - 1
            )
            if rescale_parts == 'always' or (
                rescale_parts == 'if_too_big' and total_part_width > target_width
            ):
                scale = target_width / total_part_width
            else:
                scale = 1.0

            actual_width = max(total_part_width * scale, slot_widths.get(slot_name, 0))

            prev_xpos = xpos
            for part in parts:
                part.draw(ax, xpos, ypos, scale=scale, **kwargs)
                xpos += part.artist.width * scale + parts_spacing
            xpos = prev_xpos + actual_width

        # draw a line to indicate the TU from x to xpos
        ax.plot([x, xpos], [y, y], color='k', linewidth=tu_linewidth, zorder=0)

    def __repr__(self):
        return f'TU({self.tu_name}, {self.part_names})'


##────────────────────────────────────────────────────────────────────────────}}}

tu_name = TU_names[2][0]

pname = get_parts_from_tu(lib, tu_name, enabled_tu_slots=ENABLED_TU_SLOTS)
fig, ax = plt.subplots(dpi=300)
ax.set_aspect('equal')
ax.axis('off')


# tu = TU(lib, tu_name, ENABLED_TU_SLOTS)
# tu.draw(ax, 0, 0, rescale_parts='never', slot_widths=DEFAULT_SLOT_WIDTHS)

for i, plasmid in enumerate(TU_names):
    for j, tu_name in enumerate(plasmid):
        tu = TU(lib, tu_name, ENABLED_TU_SLOTS)
        tu.draw(
            ax, 0, i * 5, rescale_parts='never', slot_widths=DEFAULT_SLOT_WIDTHS, parts_spacing=0.0
        )


ax.set_xlim(-3, 40)
ax.set_ylim(-3, 40)
ax.invert_yaxis()


### {{{                          --     archive     --

tu_name = TU_names[2][0]
TU_names
tu_name
pname = get_parts_from_tu(lib, tu_name, enabled_tu_slots=ENABLED_TU_SLOTS)

tu = TU(lib, tu_name, ENABLED_TU_SLOTS)
fig, ax = plt.subplots(dpi=300)
ax.set_aspect('equal')
ax.axis('off')
tu.parts

term = tu.parts[-1][0].artist.paths[0]
uorfs = tu.parts[1][0].artist.paths[2]
tu.parts[1][0].artist.paths

# tu.parts[0][0].draw(ax, 0, 0, linewidth=1)
prom = tu.parts[0][0].artist.paths[0]
# gn = tu.parts[2][0].artist.paths[0]
gn = tu.parts[2][0].artist.paths[2]


# gn.vertices *= 50
# plot vertices scatter plot
len(gn.codes)
# long markers

# p = term
# p = gn
# p = prom
p = uorfs

ax.scatter(
    p.vertices[:, 0],
    p.vertices[:, 1],
    c=p.codes,
    s=30,
    marker='x',
    linewidth=0.5,
    cmap='tab10',
    alpha=1,
)
ax.add_patch(patches.PathPatch(p, facecolor='g', edgecolor='k', linewidth=0, capstyle='round'))


# draw a simple vline
# ax.plot([0, 0], [0, 5], color='k', linewidth=1)


##────────────────────────────────────────────────────────────────────────────}}}
