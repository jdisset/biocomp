### {{{                          --     imports     --
import datetime
import biocomp as bc
from rich import print as pprint
import matplotlib.pyplot as plt
import numpy as np
import time
from functools import partial
import biocomp.utils as bu
import scriptutils as ut
import jax
from jax import jit, vmap, grad, value_and_grad
import jax.numpy as jnp
import biocomp.datautils as du
import optax
from pathlib import Path
from tqdm import tqdm
import biocomp.nodes as bn
import biocomp.compute as bcc
from mpl_toolkits.axes_grid1 import make_axes_locatable

import biocomp.defaults as bdf

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [7.0, 7.0]
plt.rcParams['figure.dpi'] = 200

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                  --     creating some networks     --

models = train_dman.get_models()
mparams = {}
m = models[47]
m.node_namespace = None
m.init(key, mparams)
full_params = params
full_params['node'] = mparams['node']


def any_uorf(lib, *_, **__):
    all_uORFs = lib.pc[lib.pc.category == 'uORF_group'].index.tolist()
    return [all_uORFs]


def P(name):
    return bc.Slot(lib, name)


uorfs = any_uorf(lib)[0]
any_uorf(lib)
uorfs = uorfs[:8]

import itertools

rec_uorfs = ['empty_tl'] + uorfs
ern_uorfs = ['empty_tl'] + uorfs
all_uorfs = itertools.product(rec_uorfs, ern_uorfs)
all_uorfs = list(all_uorfs)

NROWS = len(rec_uorfs)

invns = []
for rec_uorf, ern_uorf in tqdm(all_uorfs):
    tus = {
        'CasE': bc.TranscriptionUnit([P('hEF1a'), P(ern_uorf), P('CasE')]),
        'CasE_marker': bc.TranscriptionUnit([P('hEF1a'), P('mKate')]),
        'rec+eYFP': bc.TranscriptionUnit([P('hEF1a'), P(rec_uorf), P('CasE_rec'), P('eYFP')]),
        'rec_marker': bc.TranscriptionUnit([P('hEF1a'), P('eBFP')]),
    }
    aggregations = [['CasE', 'CasE_marker'], ['rec+eYFP', 'rec_marker']]
    sources = {tu_name: [tu_name] for tu_name, tu in tus.items()}
    n = bc.Network.from_dict(lib, 'v1', tus, sources, aggregations)
    invn = bc.inverted_network(n)[0]
    invns.append(invn)

# ut.plot_networks(invns)


# ideal syntax

nm = NetDesigner(lib, promoter='hEF1a')
nn.add_tu('CasE', ['CasE'])

all_variations = nn.build_all()
##────────────────────────────────────────────────────────────────────────────}}}

# distances, indices = tree.query(x, k=knn, distance_upper_bound=radius)

# writing a grid partitionning in jax for much faster plotting

##
### {{{                  --     jax compatible wrapper     --

import jax
import jax.numpy as np
from functools import wraps

class JAXCompatible:
    def __init__(self, static_attrs):
        self.static_attrs = static_attrs
        self.dynamic_attrs = [attr for attr in self.__dict__ if attr not in static_attrs]


    def split(self):
        dynamic = {attr: getattr(self, attr) for attr in self.__dict__ if attr not in self.static_attrs}
        static = {attr: getattr(self, attr) for attr in self.static_attrs}
        return dynamic, static


    @classmethod
    def reassemble(cls, dynamic, static):
        instance = cls.__new__(cls)
        for attr, value in dynamic.items():
            setattr(instance, attr, value)
        for attr, value in static.items():
            setattr(instance, attr, value)
        return instance

def jaxcls(static_attrs):
    def decorator(cls):
        @wraps(cls)
        def wrapper(*args, **kwargs):
            instance = cls(*args, **kwargs)
            jax_instance = JAXCompatible(static_attrs)
            jax_instance.__dict__.update(instance.__dict__)
            return jax_instance
        return wrapper
    return decorator


@jaxcls(static_attrs=['static_attr'])
class MyClass:
    def __init__(self, dynamic_attr, static_attr):
        self.dynamic_attr = dynamic_attr
        self.static_attr = static_attr


my_instance1 = MyClass(np.array([1.0, 2.0, 3.0]), 2.0)
my_instance2 = MyClass(np.array([4.0, 5.0, 6.0]), 3.0)
other_param = np.array([7.0, 8.0, 9.0])
other_static_param = 4.0

result = my_function(my_instance1, other_param, my_instance2, other_static_param)
print(result)

##────────────────────────────────────────────────────────────────────────────}}}

import jax.numpy as jnp

EPSILON = 1e-9


import jax
import jax.numpy as jnp
import numpy as np

from typing import NamedTuple

class GridPartition(NamedTuple):
    elements: jnp.array
    cell_size: np.array
    shape: np.array
    min_corner: jnp.array
    sorted_elements_ids: jnp.array
    cell_slices: jnp.array  # (N_CELLS, 2) array of (start, end) indices for each cell


@partial(jax.jit, static_argnames=('shape'))
def make_grid(elements, cell_size, min_corner, shape):
    N_CELLS = np.prod(shape)
    N_ELMTS = len(elements)
    def get_cell_id(pos):
        loc = ((pos - min_corner) // cell_size).astype(jnp.int32)
        return jnp.ravel_multi_index(loc, shape, mode='clip')
    cell_assignment = jax.vmap(get_cell_id)(elements)
    sorted_elements_ids = jnp.argsort(cell_assignment)
    cell_assignment_sorted = cell_assignment[sorted_elements_ids]
    cell_occupancy = jnp.bincount(cell_assignment_sorted, length=N_CELLS)
    transitions = jnp.diff(
        jnp.concatenate(
            [
                jnp.array([-1]),
                cell_assignment_sorted,
            ]
        )
    )
    which_cells = jnp.where(transitions > 0, cell_assignment_sorted, -1)
    cells_start_at = (
        jnp.zeros(N_CELLS + 1, dtype=jnp.int32).at[which_cells].set(jnp.arange(N_ELMTS))[:-1]
    )
    slices = jnp.stack([cells_start_at, cell_occupancy], axis=-1)
    return GridPartition(
        elements=elements,
        cell_size=cell_size,
        shape=shape,
        min_corner=min_corner,
        sorted_elements_ids=sorted_elements_ids,
        cell_slices=slices,
    )


def make_best_fitting_grid(elements, cell_size):
    cell_size = np.array(cell_size.astype(np.float32))
    margin = 0.1 * cell_size
    min_corner = np.min(elements, axis=0) - margin
    shape = np.max(elements, axis=0) - min_corner / cell_size
    shape = np.ceil(shape).astype(np.int32)
    shape = tuple(shape)
    print(shape)
    print(min_corner)
    # turn into a tuple of int
    return make_grid(elements, cell_size, min_corner, shape)

tuple(np.arange(3))

def get_element_mask_at_cells(grid, cell_ids):
    def cell_mask(slice):
        start, n = slice
        valid_range = jnp.arange(len(grid.elements))
        return jnp.logical_and(valid_range >= start, valid_range < start + n)

    cmasks = jax.vmap(cell_mask)(grid.cell_slices[cell_ids])
    combined_mask = jnp.any(cmasks, axis=0)
    indices = jnp.where(combined_mask, grid.sorted_elements_ids, -1)
    out = jnp.zeros(len(grid.sorted_elements_ids) + 1, dtype=jnp.bool_)
    return out.at[indices].set(True)[:-1]


# def get_element_ids_in_cell(grid, coord):
# cell_loc = ((coord - self.min_corner) // self.cell_size).astype(jnp.int32)
# cell_id = jnp.ravel_multi_index(cell_loc, self.grid_shape, mode='clip')
# start, n = self.slices[cell_id]
# return self.sorted_elements_ids[start : start + n]


# -------- jit-able static methods:


@partial(jax.jit, static_argnums=(2, 3))
def get_within(grid, center, radius, aabb_only=False):
    NDIMS = len(grid.shape)
    MAX_CELL = np.array(grid.shape) - 1

    n_cells_in_aabb = np.ceil(2 * radius / grid.cell_size).astype(np.int32) + 2
    dim_vectors = [np.arange(n) for n in n_cells_in_aabb]
    all_locs = np.array(np.meshgrid(*dim_vectors, indexing='ij')).reshape(NDIMS, -1).T

    center_loc = (center - grid.min_corner) / grid.cell_size

    all_locs = all_locs + center_loc - n_cells_in_aabb // 2
    all_locs = np.clip(all_locs, 0, MAX_CELL).astype(np.int32)

    all_cell_ids = jnp.ravel_multi_index(all_locs.T, grid.shape, mode='clip')
    element_mask = grid.get_element_mask_at_cells(
        all_cell_ids, grid.sorted_elements_ids, grid.slices
    )
    if aabb_only:
        return element_mask
    else:
        r2 = radius * radius
        # well... that kinda defeats the whole purpose of using a grid...
        return jnp.logical_and(element_mask, jnp.sum((grid.elements - center) ** 2, axis=-1) <= r2)


# testing

key = jax.random.PRNGKey(0)
elements = jax.random.uniform(key, (5000, 2), minval=-0.8, maxval=1.8)
g = make_best_fitting_grid(elements, cell_size=np.array([0.1, 0.1]))

radius = 0.62
center = np.array([0.70, 0.82])

inside = get_within(g, center, radius)
g.shape
g.cell_size

##

# scatter plot of the elements:
import matplotlib.pyplot as plt

plt.scatter(elements[inside, 0], elements[inside, 1], s=1, c='g')
plt.scatter(elements[~inside, 0], elements[~inside, 1], s=0.25, c='k', alpha=0.5)
plt.scatter(center[0], center[1], c='r', marker='x', lw=1, s=30)
plt.gca().add_artist(plt.Circle(center, radius, color='r', fill=False))
# plot the grid
for i in range(g.shape[0] + 1):
    plt.axvline(g.min_corner[0] + i * g.cell_size[0], color='k', alpha=0.2, lw=0.5)
for i in range(g.shape[1] + 1):
    plt.axhline(g.min_corner[1] + i * g.cell_size[1], color='k', alpha=0.2, lw=0.5)

plt.axis('equal')
plt.xlim(-1, 2)
plt.ylim(-1, 2)
plt.show()


# TODO:
# "Can we guess what a cascade looks like without seeing one?"
# [ ] make training set with [Csy4, Case] matrices + single uOrfs + 
#       rows: 2, 3, 4, 6:16 -> USE EVERYTHING that's not detrimental (or contains cascade)
# Validation set: 22, 23, 24, 25
# [ ] Plot and quantify accuracy
# [ ] Add all of the validation set BUT ONE XP to the training set ; what's the accuracy on the remaining one?


# jax.lax.dynamic_slice(jnp.arange(10), (1,), (12,))

### {{{                        --     manual grid     --
cell_assignment = jax.vmap(_get_cell_id)(elements)
cell_assignment
sorted_elements_ids = jnp.argsort(cell_assignment)
cell_assignment_sorted = cell_assignment[sorted_elements_ids]
cell_assignment_sorted
cell_occupancy = jnp.bincount(cell_assignment_sorted, length=N_CELLS)
transitions = jnp.diff(
    jnp.concatenate(
        [
            jnp.array([-1]),
            cell_assignment_sorted,
        ]
    )
)
transitions
which_cells = jnp.where(transitions > 0, cell_assignment_sorted, -1)
cells_start_at = (
    jnp.zeros(N_CELLS + 1, dtype=jnp.int32).at[which_cells].set(jnp.arange(N_ELMTS))[:-1]
)
ranges = jnp.stack([cells_start_at, cells_start_at + cell_occupancy], axis=-1)
##────────────────────────────────────────────────────────────────────────────}}}
