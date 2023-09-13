### {{{                          --     imports     --
import matplotlib

from dataclasses import dataclass
from jax import make_jaxpr
import jax.tree_util as jtu
from jax.tree_util import register_pytree_node_class
from collections.abc import Mapping
from copy import deepcopy
import biocomp as bc
from biocomp import datautils as du
from jax.tree_util import Partial as partial
from datetime import datetime
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import jax
from jax import jit, vmap, value_and_grad
import jax.numpy as jnp
import pickle

from biocomp import utils as ut
import scriptutils as su
import biocomp.datautils as du
from biocomp import train
from biocomp import compute as cmp
from evosax import CMA_ES
from evosax.utils import ESLog, FitnessShaper
import os
import joblib
import datetime

from matplotlib import pyplot as plt

dirname = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# matplotlib.use('agg')
matplotlib.rcParams['figure.dpi'] = 200

##────────────────────────────────────────────────────────────────────────────}}}
### {{{                     --     generate networks     --

lib = su.load_lib()


def sequestron_ERN3p(get_param, get_quantized, **_):
    def apply(rna, ern, **_):
        # return rna * (1.0 - jnp.exp(-ern))
        return jnp.relu(ern - rna)

    return apply


def any_uorf(lib, *_, **__):
    all_uORFs = lib.pc[lib.pc.category == 'uORF_group'].index.tolist()
    return [all_uORFs]


def P(name):
    return bc.Slot(lib, name)


def TU(*parts):
    partlist = [P('hEF1a')] + list(parts)
    return bc.TranscriptionUnit(partlist)


uorfs = P(any_uorf(lib)[0][:8])
# 'Csy4+uOrfs': bc.TranscriptionUnit([promoter, P('Csy4'), P(any_uorf(lib)[0])]),

ERNs = ['CasE', 'Csy4', 'PgU']
ern = [P(ern) for ern in ERNs]
rec = [P(ern + '_rec') for ern in ERNs]
colors = [P('mKate'), P('eBFP'), P('NeonGreen'), P('iRFP720')]

tus_bp = {
    # node A
    'A_pos_0': TU(rec[0], uorfs, ern[2]),
    'A_pos_1': TU(rec[0], uorfs, ern[2]),
    'A_pos_2': TU(rec[0], uorfs, ern[2]),
    'A_neg_0': TU(ern[0]),
    'A_neg_1': TU(ern[0]),
    'A_neg_2': TU(ern[0]),
    # node B
    'B_pos_0': TU(rec[1], uorfs, ern[2]),
    'B_pos_1': TU(rec[1], uorfs, ern[2]),
    'B_pos_2': TU(rec[1], uorfs, ern[2]),
    'B_neg_0': TU(ern[1]),
    'B_neg_1': TU(ern[1]),
    'B_neg_2': TU(ern[1]),
    # colors
    'x0color': TU(colors[0]),
    'x1color': TU(colors[1]),
    'biascolor': TU(colors[2]),
    # output node
    'C_pos': TU(rec[2], colors[3]),
    'C_neg': TU(ern[2]),
}


# everything everywhere all at once:
aggregations_bp = [
    ['A_pos_0', 'A_neg_0', 'B_pos_0', 'B_neg_0', 'x0color'],  # x0
    ['A_pos_1', 'A_neg_1', 'B_pos_1', 'B_neg_1', 'x1color'],  # x1
    ['A_pos_2', 'A_neg_2', 'B_pos_2', 'B_neg_2', 'C_pos', 'C_neg', 'biascolor'],  # biases
]


sources_bp = {
    tu_name: [tu_name] for tu_name, tu in tus_bp.items() if tu_name in ut.flatten(aggregations_bp)
}
used_tus_bp = {
    tu_name: tu for tu_name, tu in tus_bp.items() if tu_name in ut.flatten(aggregations_bp)
}

n_bp = bc.Network.from_dict(lib, 'bp_attempt', used_tus_bp, sources_bp, aggregations_bp)
bp_net = bc.inverted_network(n_bp)[0]


tus_single = {
    'A_pos': TU(rec[0], colors[3]),
    'A_neg': TU(ern[0]),
    'x0color': TU(colors[0]),
    'x1color': TU(colors[1]),
}
aggregations_single = [
    ['A_pos', 'x0color'],  # x0
    ['A_neg', 'x1color'],  # x1
]
sources_single = {tu_name: [tu_name] for tu_name, tu in tus_single.items()}
n_single = bc.Network.from_dict(lib, 'single_ERN', tus_single, sources_single, aggregations_single)
single_net = bc.inverted_network(n_single)[0]

networks = [bp_net]
su.plot_networks(networks, W=4500, H=4000, show=True, figsize=(22, 20))
NETWORK = networks[0]


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                     --     old school params     --
def indirect_param_at(
    params,
    name,
    node_id=0,
    base_path=ut.NODE_PATH,
    init=None,
    overwrite_with=None,
    read_only=True,
    number_of_nodes_at_least=1,
    **_,
):

    """
    Retrieves or sets a parameter from the given params dictionary.
    Vectorizable across the node_id axis.
    If the parameter is not found, it is created and added to the params dict. (unless read_only is True)
    - params: the dictionary of parameters
    - name: the name of the parameter
    - node_id: the id of the node that owns this parameter
    - base_path: the path to the node in the params dict, which acts as a namespace ("node", "shared", "static", ...)
    - init: the initialization function to use if the parameter is not found
    - overwrite_with: if not None, the parameter will be overwritten with this value wether it exists or not
    - read_only: if True, the parameter will not be created if it is not found (and not overwritten)
    """

    # We can't jit/vectorize a dictionnary lookup. i.e we can't do:
    # res = params[node_id] as this requires branching
    # Indexing an array is fine though, so we could simply create
    # an array of params for each node that is as big as the largest
    # node_id, and then index it with the node_id, which is exactly what we do in
    # direct_param_at.
    # However, this is wasteful for params that have large shapes
    # but are only used by a few nodes (most params are like this).

    # So instead I add one layer of indirection to have a sparse array of params:
    # we save a key_vec which will contain -1 for all nodes that don't use
    # the given parameter, and an actual parameter_id for the nodes that do.
    # This way we can use the key_vec to index a parameter array that contains
    # only the parameters that are actually used by the network.

    # I think in theory we can also use node_id with base_path = shared
    # to vectorize tl vs tx by accessing different weights!

    assert isinstance(params, dict), f'params must be a dict, not {type(params)}'

    dpath = base_path / name

    nparams = ut.at_path(params, dpath, None)
    nparams = nparams.shape[0] if nparams is not None else 0

    keys_path = ut.KEYS_PATH + dpath
    key_vec = ut.at_path(params, keys_path, None)  # key_vec is an integer vector (n_nodes,)

    if not read_only:  # non-jittable path (only used for initialization)
        N_NODES = max(node_id, number_of_nodes_at_least - 1) + 1
        if key_vec is None or key_vec.shape[0] <= N_NODES:
            # extend key_vec to fit node_id
            v = key_vec if key_vec is not None else jnp.zeros((0,), dtype=jnp.int32)
            key_vec = jnp.concatenate(
                [v, jnp.full((N_NODES - v.shape[0] + 1,), -1, dtype=jnp.int32)]
            )
        if int(key_vec[node_id]) == -1:  # param doesn't exist yet
            try:
                new_param_value = overwrite_with if overwrite_with is not None else init()
                p = ut.at_path(params, dpath)  # get existing parameter array
                if p is None:  # first param ever for this path
                    p = jnp.expand_dims(new_param_value, axis=0)
                else:  # add new param to existing array
                    p = jnp.concatenate([p, jnp.expand_dims(new_param_value, axis=0)])
                ut.at_path(params, dpath, p)  # update params
                # update and save key_vec:
                key_vec = ut.at_path(params, keys_path, key_vec.at[node_id].set(nparams))
            except Exception as e:
                msg = f'Error initializing param "{name}" from node {node_id}: {e}'
                raise RuntimeError(msg) from e

    param_id = key_vec[node_id]

    if overwrite_with is not None and not read_only:  # also non-jittable
        allp = ut.at_path(params, dpath).at[param_id].set(overwrite_with)
        ut.at_path(params, dpath, allp)

    res = ut.at_path(params, dpath)[param_id]

    return res


def direct_param_at(
    params,
    name,
    node_id=0,
    base_path=ut.NODE_PATH,
    init=None,
    overwrite_with=None,
    read_only=True,
    number_of_nodes_at_least=1,
    **_,
):

    """
    Similar to indirect_param_at, but doesn't use key_vec: it's a dense param array
    instead of a sparse one. Potentially VERY wasteful, but faster to access.
    """

    if not isinstance(params, dict):
        raise TypeError(f'params must be a dict, not {type(params)}')

    dpath = base_path / name
    p_array = ut.at_path(params, dpath, None)  # p_array is the parameter array (n_params, *shape)

    if not read_only:  # non-jittable path (only used for initialization)
        # first we will check if the param is already initialized
        IS_INIT_PATH = ut.STATIC_PATH / 'is_init'
        # we store a boolean indicating if a param is initialized
        is_init_array = ut.at_path(params, IS_INIT_PATH / dpath, None)
        if is_init_array is None or is_init_array.shape[0] <= node_id:
            # extend is_init_array to fit node_id
            v = is_init_array if is_init_array is not None else np.zeros((0,), dtype=np.bool_)
            is_init_array = np.concatenate(
                [v, np.full((node_id - v.shape[0] + 1,), False, dtype=np.bool_)]
            )
            ut.at_path(params, IS_INIT_PATH / dpath, is_init_array)
        param_is_init = is_init_array[node_id]

        if not param_is_init or overwrite_with is not None:
            new_value = overwrite_with if overwrite_with is not None else init()
            if p_array is not None and p_array.shape[1:] != new_value.shape:
                raise ValueError(
                    f'Param "{name}" has shape {p_array.shape[1:]}, but '
                    f'new value has shape {new_value.shape}.'
                )
            # then let's make sure the param array is big enough
            REQUIRED_LENGTH = max(node_id, number_of_nodes_at_least - 1) + 1
            if p_array is None:
                p_array = np.zeros((REQUIRED_LENGTH,) + new_value.shape, dtype=new_value.dtype)
            elif p_array.shape[0] < REQUIRED_LENGTH:
                p_array = np.concatenate(
                    [p_array, np.zeros((REQUIRED_LENGTH - p_array.shape[0],) + new_value.shape)]
                ).astype(new_value.dtype)

            # finally we can set the param
            p_array[node_id] = new_value
            p_array = ut.at_path(params, dpath, p_array)
            p = p_array[node_id]
            # and mark the param as initialized
            is_init_array[node_id] = True
            ut.at_path(params, IS_INIT_PATH / dpath, is_init_array)

    dtype = p_array.dtype
    p = p_array[node_id].astype(dtype)
    return p


PARAM_AT = direct_param_at


def set_param(params, name, value, node_id=0, base_path=ut.NODE_PATH, **kw):
    return PARAM_AT(
        params, name, node_id, base_path, overwrite_with=np.asarray(value), read_only=False, **kw
    )


def get_param(params, name, node_id=0, base_path=ut.NODE_PATH, **_):
    return PARAM_AT(params, name, node_id, base_path, read_only=True)


def init_param_if_needed(params, name, init, node_id=0, base_path=ut.NODE_PATH, **kw):
    return PARAM_AT(params, name, node_id, base_path, init=init, read_only=False, **kw)


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                        --     Ptree     --

from typing import Callable, Optional, Union, Tuple, Any, Dict, List

# TODO:
# to make inits much faster, I should implement something like this:
# set_row(params, "inv_aggregation:original_output_slot", outslots)
# or, if I find the time to implement a proper paramtree
# params.set_row("inv_aggregation_original_output_slot", outslots)
#
# About the paramtree, I think I should implement it as a class
# and allow arbitrary namespaces - static being the only default.
# and maybe a tag system to allow for easy filtering:
# params.set_param("local", "inv_aggregation_original_output_slot", tags=[nograd])
#
# nogradparams = params.filter(tags=["nograd"])
# localparams = params.filter(namespaces=["local"])
# nogradlocalparams = params.filter(tags=["nograd"], namespaces=["local"])
# inverse logic (select all params that are not tagged "nograd"):
# gradparams = params.filter(tags=["nograd"], inverse=True)


@jax.jit
def tree_transpose(list_of_trees):
    """Convert a list of trees of identical structure into a single tree of lists."""
    return jax.tree_map(lambda *xs: jnp.array(xs), *list_of_trees)


class ParamPath:
    def __init__(self, path=None):
        if isinstance(path, str):
            path = path.strip("/").split("/")
        self.path = path or []

    def __truediv__(self, key):
        if isinstance(key, str):
            key = key.strip("/").split("/")
        elif isinstance(key, ParamPath):
            key = key.path
        return ParamPath(self.path + key)

    def __add__(self, key):
        return self.__truediv__(key)

    def __repr__(self):
        return "/".join(self.path)

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, key):
        return self.path[key]

    def __len__(self):
        return len(self.path)

    def __iter__(self):
        return iter(self.path)

    def __eq__(self, other):
        if isinstance(other, str):
            other = other.strip("/").split("/")
        elif isinstance(other, ParamPath):
            other = other.path
        return self.path == other

    def __lt__(self, other):
        if isinstance(other, str):
            other = other.strip("/").split("/")
        elif isinstance(other, ParamPath):
            other = other.path
        return self.path < other

    def __gt__(self, other):
        if isinstance(other, str):
            other = other.strip("/").split("/")
        elif isinstance(other, ParamPath):
            other = other.path
        return self.path > other

    def __hash__(self):
        return hash(tuple(self.path))

    def __contains__(self, key):
        if isinstance(key, str):
            key = key.strip("/").split("/")
        elif isinstance(key, ParamPath):
            key = key.path
        return key in self.path


class PTree:
    def __init__(self, value=None, read_only=False):
        self.value = value
        self.set_read_only(read_only)

    def is_leaf(self):
        if not isinstance(self.value, dict):
            return True
        if len(self.value) == 0:
            return True
        if any(not isinstance(v, PTree) for v in self.value.values()):
            return True
        return False

    def __contains__(self, key):
        return key in self.value

    def pretty(self, levels=None, key=None):
        s = ""
        if levels == None:
            s += f"\n ▼"
            s += self.pretty([]) + "\n\n"
            if self.value is None:
                return " ∅\n"
        else:
            other_branches = [' │  ' if l else '    ' for l in levels]
            lineheader = f"\n{''.join(other_branches)}"
            if self.is_leaf():
                keylen = len(key) if key is not None else 0
                valstr = str(self.value) if self.value is not None else "∅"
                valstr = valstr.replace("\n", f'{lineheader}{" " * keylen}     ')
                s += f" ⟶ {valstr}"
            else:
                nitems = len(self.value.items())
                for i, (k, v) in enumerate(self.value.items()):
                    this_branch_char = " └─ " if i == nitems - 1 else " ├─ "
                    s += f"{lineheader}{this_branch_char}'{k}'"
                    s += v.pretty(levels + [i < nitems - 1], k)
        return s

    def __str__(self):
        return self.pretty()

    def __repr__(self):
        return self.pretty()

    def __getitem__(self, path):
        if not isinstance(path, ParamPath):
            path = ParamPath(path)
        if len(path) == 0:
            if self.is_leaf():
                return self.value
            else:
                return self
        if self.value is None:
            raise KeyError(f"ParamTree is empty, cannot get {path}")
        return self.value[path[0]][path[1:]]

    def __setitem__(self, path, value):
        if self.read_only:
            raise RuntimeError("Cannot set value on read-only ParamTree")
        if not isinstance(path, ParamPath):
            path = ParamPath(path)
        # we'll create the path if it doesn't exist
        if len(path) == 0:
            self.value = value
            return
        if self.value is None:
            self.value = {}
            self.tags = {}
        else:
            if self.is_leaf():
                raise KeyError(f"Trying to create path {path} on leaf node")
        if path[0] not in self.value:
            self.value[path[0]] = PTree(read_only=self.read_only)
        self.value[path[0]][path[1:]] = value

    def at(self, path, value=None, overwrite=False):
        if self.read_only or value is None:
            return self[path]
        else:
            try:
                self[path]
            except KeyError:
                overwrite = True
            if overwrite:
                self[path] = value
            return self[path]

    def has_leaf(self, path):
        return not isinstance(self[path], PTree)

    def is_empty(self):
        return self.value is None

    def get_read_only_copy(self):
        from copy import deepcopy

        cop = deepcopy(self)
        cop.set_read_only()
        return cop

    def set_read_only(self, ro=True):
        self.read_only = ro
        if not self.is_leaf():
            for v in self.value.values():
                v.set_read_only(ro)

    def all_leaves_are_none(self):
        return jax.tree_util.tree_all(jax.tree_util.tree_map(lambda x: x is None, self))

    def remove_empty_leaves(self):
        newvals = {}
        if self.is_leaf():
            return self
        for k, v in self.value.items():
            if not v.all_leaves_are_none():
                newvals[k] = v.remove_empty_leaves()
        return PTree(value=newvals, read_only=self.read_only)

    def iter_leaves(self, path=ParamPath()):
        if self.is_leaf():
            yield path, self.value
        else:
            for k, v in self.value.items():
                yield from v.iter_leaves(path / k)

    def __eq__(self, other):
        if not isinstance(other, PTree):
            return False
        if not type(self.value) == type(other.value):
            return False
        if isinstance(self.value, (np.ndarray, jnp.ndarray)):
            return np.all(self.value == other.value)
        return self.value == other.value


class TreeRef:
    # a reference to a subtree
    def __init__(self, path, tree):
        self.path = path
        self.tree = tree

    def __repr__(self):
        return f"TreeRef* ({self.path}): {self.tree[self.path]}"

    def get(self):
        return self.tree[self.path]


class ArrayRef:
    # an array of references to some other arrays values
    # aka a view, but over potentially several different arrays

    # I don't think we can have vmappable pointers so we need
    # to be a bit more hacky here

    # Can we maintain a list of reference to all the needed arrays
    # (avoiding duplicates so need another layer of indirection probably)
    # then we concatenate everything together and just maintain a 1d array of offsets?
    # Question is: will grad propagate through this? I think it should
    # Actually if that works, no need to concat everything, just the relevant portions
    # of each arrays is enough.

    # Much uglier way if we can't concat array references is to
    # maintain a tree-wide flatened array of everything then use
    # offsets and shapes but that is very ugly....

    # Other way around: whenever we create an array ref, we actually take
    # ownership of all the referenced arrays, i.e create one big array that contains
    # all the information of the referenced arrays, and turned the referenced arrays
    # into simple views to the big array.
    # Problem is we can only have one reference in this case

    def __init__(self, tree, paths=None, indices=None):
        self.tree = tree
        self.indices = indices or ()  # tuple of (array_num, index) coordinates
        self.paths = paths or ()  # tuple of paths to the referenced arrays
        self._pathdict = {p: i for i, p in enumerate(self.paths)}
        self.make_map()

    def __repr__(self):
        return f"RefArray[]: {self.get()}"

    def push_back(self, array_path, id):
        if array_path not in self.paths:
            self.paths += (array_path,)
            self._pathdict[array_path] = len(self.paths) - 1
        self.indices += ((self._pathdict[array_path], id),)
        self.make_map()

    def make_map(self):
        idx = np.asarray(self.indices)
        self.arrays = tuple([self.tree[p] for p in self.paths])
        self.shape = self.tree[self.paths[0]].shape
        self.map = ()
        for a in np.unique(idx[:, 0]):
            positions = np.where(idx[:, 0] == a)[0]
            ids_in_array = idx[positions, 1]
            self.map += ((a, positions, ids_in_array),)


    # @partial(jax.jit, static_argnums=(0,))
    def get(self):
        # how expensive is this?
        # is it actually doing that every time?
        # are the gradients correctly propagated?

        N = len(self.indices)
        if N == 0:
            return jnp.array([])

        conc = jnp.zeros((N, *self.shape[1:]))
        for a,p,i in self.map:
            conc = conc.at[p].set(self.arrays[a][i])

        return conc



@dataclass
class RefPath:
    actual_path: ParamPath
    points_to: ParamPath


@dataclass
class ArrayRefPath:
    actual_path: ParamPath
    paths: List[ParamPath]
    indices: List[Tuple[int, int]]


# jax.tree_util.register_pytree_node(ParamRef, flatten_paramref, unflatten_paramref)

# if we are to use reference, I think I have to write my own manual flatten...


def flatten_PTree_manual(ptree):
    keys, values = [], []
    for k, v in ptree.iter_leaves():
        if isinstance(v, TreeRef):
            values.append(None)
            keys.append(RefPath(k, v.path))  # all the information to reconstruct the reference
        elif isinstance(v, ArrayRef):
            values.append(None)
            keys.append(ArrayRefPath(k, v.paths, v.indices))
        else:
            values.append(v)
            keys.append(k)

    aux_data = (keys, ptree.read_only)
    return (values, aux_data)


def unflatten_PTree_manual(aux_data, content):
    keys = aux_data[0]
    read_only = aux_data[1]
    ptree = PTree(read_only=read_only)
    for k, v in zip(keys, content):
        if isinstance(k, RefPath):
            ptree[k.actual_path] = TreeRef(k.points_to, ptree)
        elif isinstance(k, ArrayRefPath):
            a = ArrayRef(ptree, k.paths, k.indices)
        else:
            ptree[k] = v
    ptree.set_read_only(read_only)
    return ptree


jax.tree_util.register_pytree_node(PTree, flatten_PTree_manual, unflatten_PTree_manual)

# def flatten_PTree(ptree):
# flat_contents = (ptree.value,)
# aux_data = (ptree.read_only,)
# return (flat_contents, aux_data)


# def unflatten_PTree(aux_data, flat_contents):
# value = flat_contents[0]
# read_only = aux_data[0]
# return PTree(value=value, read_only=read_only)


# jax.tree_util.register_pytree_node(PTree, flatten_PTree, unflatten_PTree)

ptree = PTree()
ptree['a/c'] = 2
ptree['a/b0'] = np.array([1, 2, 3])
ptree['a/b1'] = np.array([4, 5])
ptree.at('a/b/yo', np.array([5, 6, 7]))
ptree['a/r'] = TreeRef('a/b', ptree)
ptree['a/r'].get()['newbranch'] = 3

ptree

ptree['arr/ref'] = ArrayRef(ptree, ['a/b0', 'a/b1'], [(0, 1), (1, 0)])

ptree['arr/ref'].get()

make_jaxpr(ArrayRef.get, static_argnums=(0,))(ptree['arr/ref'])
print(jit(ArrayRef.get, static_argnums=(0,)).lower(ptree['arr/ref']).compile().as_text())


jax.tree_util.tree_structure(ptree)
jax.tree_util.tree_leaves(ptree)

jax.tree_util.tree_flatten(ptree)


# l, s = jtu.tree_flatten(ptree)

# ptree_reconstruct = jtu.tree_unflatten(s, l)
# ptree

# ptree_reconstruct


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                        --     ParameterTree     --

# or, rather, there should be a core PTree without tags,
# and a TaggedTree that contains a Tree of params and a Tree of tags
# self.tags = tags
# self.tagnames = tagnames or []


class ParameterTree:
    def __init__(
        self, data: PTree = None, tags: PTree = None, tagnames: list = None, read_only=False
    ):
        self.data = data or PTree()
        self.tags = tags or PTree()
        self.tagnames = tagnames or []
        self.__tagdict = {name: i for i, name in enumerate(self.tagnames)}
        self.read_only = read_only
        self.data.set_read_only(read_only)
        if tags is not None:
            self.tags.set_read_only(read_only)

    def __setitem__(self, path, value):
        if self.read_only:
            raise RuntimeError("Cannot set value on read-only ParameterTree")
        self.data[path] = value
        if len(self.tagnames) > 0:
            self.tags[path] = np.zeros(len(self.tagnames), dtype=bool)

    def __getitem__(self, path):
        return self.data[path]

    def at(self, path, value=None, tags=None, overwrite=False):
        """
        overwrite = True -> overwrite existing value
        overwrite = False -> return existing value
        overwrite = None -> raise KeyError if path exists
        """
        if self.read_only or value is None:
            return self.data[path]
        else:
            exists = True
            try:
                self.data[path]
            except KeyError:
                exists = False
            if overwrite is None and exists:
                raise KeyError(f"Path {path} already exists, cant overwrite without overwrite=True")
            if overwrite or not exists:
                self.data[path] = value
                if tags is not None:
                    self.create_tags_if_required(tags)
                    tag_ids = [self.__tagdict[tag] for tag in tags]
                    tag_flags = np.zeros(len(self.tagnames), dtype=bool)
                    tag_flags[tag_ids] = True
                    self.tags[path] = tag_flags
            return self.data[path]

    def __repr__(self):
        def with_box(title, data_splitlines, lw):
            reconstructed_lines = [
                f"│ {line}{' ' * (lw - len(line))}│\n" for line in data_splitlines[:-1]
            ]
            s = f"\n╭── {title} {('─' * (lw - len(title) - 3))}╮\n"
            s += "".join(reconstructed_lines)
            s += f"╰─{'─' * lw}╯\n"
            return s

        data_splitlines = self.data.__repr__().split("\n")
        tag_splitlines = self.tags.__repr__().split("\n")
        tags_str = ", ".join(self.tagnames)
        lw = max([len(line) for line in data_splitlines]) + 1
        lw = max(lw, max([len(line) for line in tag_splitlines]) + 1)
        lw = max(lw, len(tags_str) + 30)

        content = (
            [' ']
            + ["data"]
            + data_splitlines[1:]
            + ["╴" * (lw - 1)]
            + [' ']
            + [f"tags [{tags_str}]"]
            + tag_splitlines[1:]
        )
        s = with_box(f"Parameter Tree ({'RO' if self.read_only else 'RW'})", content, lw)
        return s

    def create_tags_if_required(self, tags):
        if isinstance(tags, str):
            tags = [tags]
        for tag in tags:
            if tag not in self.tagnames:
                self.add_new_tag(tag)

    def add_new_tag(self, tag):
        self.tagnames = sorted(self.tagnames + [tag])
        self.__tagdict = {name: i for i, name in enumerate(self.tagnames)}

        if self.tags is None or self.tags.is_empty():
            self.tags = jax.tree_util.tree_map(lambda _: np.array([False]), self.data)
        else:
            insert_idx = self.__tagdict[tag]
            self.tags = jax.tree_util.tree_map(lambda t: np.insert(t, insert_idx, False), self.tags)

    def tag(self, path, tags, overwrite=False):
        if self.read_only:
            raise RuntimeError("Cannot add tag read-only ParameterTree")
        self.create_tags_if_required(tags)
        if not self.data.has_leaf(path):
            raise KeyError(f"Trying to tag non-leaf node {path}")

        if isinstance(tags, str):
            tags = [tags]

        self.create_tags_if_required(tags)

        assert self.tags[path].shape == (len(self.tagnames),)

        tag_ids = [self.__tagdict[tag] for tag in tags]
        tag_flags = np.zeros(len(self.tagnames), dtype=bool)
        tag_flags[tag_ids] = True
        if overwrite:
            self.tags[path] = tag_flags
        else:
            self.tags[path] = self.tags[path] | tag_flags

    def get_read_only_copy(self):
        from copy import deepcopy

        return ParameterTree(
            data=self.data.get_read_only_copy(),
            tags=self.tags.get_read_only_copy(),
            tagnames=deepcopy(self.tagnames),
            read_only=True,
        )

    def filter_by_tag(self, tags):
        if isinstance(tags, str):
            tags = [tags]
        for t in tags:
            if t not in self.tagnames:
                raise KeyError(f"Tag {t} not found in ParameterTree")

        tag_ids = [self.__tagdict[tag] for tag in tags]
        is_valid = jax.tree_util.tree_map(lambda x: np.all(x[tag_ids]), self.tags)

        left_data_tree = jax.tree_util.tree_map(
            lambda mask, x: x if mask else None, is_valid, self.data
        )
        right_data_tree = jax.tree_util.tree_map(
            lambda mask, x: x if not mask else None, is_valid, self.data
        )

        left_tag_tree = jax.tree_util.tree_map(
            lambda mask, x: x if mask else None, is_valid, self.tags
        )
        right_tag_tree = jax.tree_util.tree_map(
            lambda mask, x: x if not mask else None, is_valid, self.tags
        )

        left_param_tree = ParameterTree(
            data=left_data_tree,
            tags=left_tag_tree,
            tagnames=self.tagnames,
            read_only=self.read_only,
        )
        right_param_tree = ParameterTree(
            data=right_data_tree,
            tags=right_tag_tree,
            tagnames=self.tagnames,
            read_only=self.read_only,
        )

        return left_param_tree, right_param_tree

    def remove_empty_leaves(self):
        return ParameterTree(
            data=self.data.remove_empty_leaves(),
            tags=self.tags.remove_empty_leaves(),
            tagnames=self.tagnames,
            read_only=self.read_only,
        )

    def get_tags(self, path):
        if self.tags is None:
            return []
        else:
            return [self.tagnames[i] for i in np.where(self.tags[path])[0]]

    @classmethod
    def merge(cls, left, right):
        is_leaf = lambda x: x is None
        left_struct = jtu.tree_structure(left, is_leaf=is_leaf)
        right_struct = jtu.tree_structure(right, is_leaf=is_leaf)
        if left_struct == right_struct:
            return jtu.tree_map(lambda l, r: l if r is None else r, left, right, is_leaf=is_leaf)
        else:
            return cls.sparse_merge(left, right)

    @classmethod
    def sparse_merge(cls, left, right):
        merged = deepcopy(left)
        for right_tag_name in right.tagnames:
            if right_tag_name not in merged.tagnames:
                merged.add_new_tag(right_tag_name)

        for k, v in right.data.iter_leaves():
            vtags = right.get_tags(k)
            merged.at(k, value=v, tags=vtags, overwrite=None)

        return merged

    def __eq__(self, other):
        if not isinstance(other, ParameterTree):
            return False
        else:
            return (
                self.data == other.data
                and self.tags == other.tags
                and self.tagnames == other.tagnames
            )


# let's try to add references


params = ParameterTree()
params['a/x'] = 2
params['a/b/c/d'] = 3
params.at('a/b/yo', np.array([1, 2, 4]))
# ex usage:
# params['a/b/c'] = ParamRef('a/b/d')


def flatten_ParameterTree(ptree):
    flat_contents = (ptree.data, ptree.tags)
    aux_data = (ptree.read_only, ptree.tagnames)
    return (flat_contents, aux_data)


def unflatten_ParameterTree(aux_data, flat_contents):
    data, tags = flat_contents
    read_only, tagnames = aux_data
    return ParameterTree(data=data, read_only=read_only, tags=tags, tagnames=tagnames)


jax.tree_util.register_pytree_node(ParameterTree, flatten_ParameterTree, unflatten_ParameterTree)


jax.tree_util.tree_leaves(params)

jax.tree_util.tree_flatten_with_path(params)

params.tag('a/b/c/d', ['test', 'b'])
params.tag('a/x', ['othertag', 'aha'])
params.tag('a/b/yo', ['b'])

a, b = params.filter_by_tag('aha')

a_noempty = a.remove_empty_leaves()

m = ParameterTree.merge(a, b)
m_noempty = ParameterTree.merge(a.remove_empty_leaves(), b.remove_empty_leaves())

# let's try to add references


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                         --     ParamRow     --


class ParamRow:
    def __init__(self, init_f=None):
        self.init_f = init_f
        self.data = None
        self.n_elems = 0

    def generate_all(self):
        self.data = vmap(self.init_f)(np.arange(self.n_elems))

    def __getitem__(self, node_id):
        if self.data is None:
            self.n_elems = max(node_id + 1, self.n_elems)
        else:
            return self.data[node_id]

    def __setitem__(self, node_id, value):
        assert self.init_f is None
        self.n_elems = max(node_id + 1, self.n_elems)
        if self.data is None:
            self.data = np.zeros(self.n_elems)
        if self.data.shape[0] < self.n_elems:
            self.data = np.concatenate([self.data, np.zeros(self.n_elems - self.data.shape[0])])
        self.data[node_id] = value


def flatten_ParamRow(prow):
    return (prow.data, (prow.init_f, prow.n_elems))


def unflatten_ParamRow(aux_data, flat_contents):
    init_f, n_elems = aux_data
    prow = ParamRow(init_f)
    prow.n_elems = n_elems
    prow.data = flat_contents
    return prow


jtu.register_pytree_node(ParamRow, flatten_ParamRow, unflatten_ParamRow)

##────────────────────────────────────────────────────────────────────────────}}}


### {{{                           --     tests     --

# ALSO TODO:
# what will a node look like when paramtrees are used?


def init_param_if_needed(params, name, init, node_id=0, base_path=ut.NODE_PATH, **kw):
    return PARAM_AT(params, name, node_id, base_path, init=init, read_only=False, **kw)


param_f = (partial(init_param_if_needed, params),)
rates = param_f(
    individual_rate_name,
    init=ut.continuous_initializer(key, rshape),
    node_id=node_id,
    base_path=local_path,
)

init_param_if_needed(params, rate_name, init=init, base_path=ut.QVALS_PATH, node_id=0)

rates = params.at(local_path / rate_name / l_id, ParamRow(init_f(shape)))[node_id]


##


tree = PTree()


def test_apply(params, x):
    v = params.at('x0', np.ones(1) * x, overwrite=False)
    v2 = params.at('other/path/x1', np.zeros(2), overwrite=False)
    y = params.at('x0', v + 1, overwrite=True)
    # params.at('static', 'oh no, a string!!', overwrite=False)
    params.at('static', True, overwrite=False)
    return y + x


def make_trees(xlist):
    for x in xlist:
        tree = PTree()
        test_apply(tree, x)
        yield tree


trees = tree_transpose(list(make_trees(np.array([1, 2, 3]))))

trees

# vf = jit(vmap(partial(test_apply, trees.get_read_only_tree())))
vf = jit(vmap(test_apply))
vf(trees.get_read_only_tree(), np.array([1, 2, 1]))

##────────────────────────────────────────────────────────────────────────────}}}
