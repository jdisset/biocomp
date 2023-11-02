### {{{                           --     cmaps     --
import matplotlib.colors as mcolors

cmap = 'Blues'
cmap = 'GnBu'
cmap = 'BuPu'

blues = [
    '#F9F7F5',
    '#EEECEA',
    '#B0CCD6',
    '#6CAFC3',
    '#2974A4',
    '#3B4B90',
    '#3D1277',
    '#22044B',
]

greens = [
    '#F9F7F5',
    '#E2EADA',
    '#CBE4BB',
    '#9DDDAA',
    '#4CCDAB',
    '#30A78F',
    '#1F7D73',
    '#0C5558',
]

reds = [
    '#F5F5F5',
    '#F1E6E5',
    '#F3CFBC',
    '#EF957D',
    '#D3494B',
    '#B00031',
    '#840137',
    '#560140',
]


cmaps = {
    'blues': mcolors.LinearSegmentedColormap.from_list('cm', blues, N=256),
    'greens': mcolors.LinearSegmentedColormap.from_list('cm', greens, N=256),
    'reds': mcolors.LinearSegmentedColormap.from_list('cm', reds, N=256),
}


##────────────────────────────────────────────────────────────────────────────}}}

from pathlib import Path

onedrive = Path(
    '~/Library/CloudStorage/OneDrive-SharedLibraries-MassachusettsInstituteofTechnology/'
).expanduser()
plotdir = onedrive / 'Neuromorphic Biocompiler - Documents/Plots'
