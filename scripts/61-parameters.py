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

from io import StringIO
import sys


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


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

    @classmethod
    def is_leaf(cls, tree, count_none_as_leaf=True):
        if tree.value is None:
            return count_none_as_leaf
        if not isinstance(tree.value, dict):
            return True
        if len(tree.value) == 0:
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
            if PTree.is_leaf(self):
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

    def get(self):
        if PTree.is_leaf(self, count_none_as_leaf=False):
            # does this value itself have a get method? (e.g. TreeReferences do...)
            if hasattr(self.value, "get"):
                return self.value.get()  # alows to follow references
            return self.value
        return self

    def __getitem__(self, path):
        if not isinstance(path, ParamPath):
            path = ParamPath(path)
        # print(f"PTree {id(self)} GET at path {path}")
        if self.value is None:
            raise KeyError(f"PTree is empty, cannot get {path}")
        if len(path) == 0:
            raise KeyError(f"PTree getitem called with empty path")
        if PTree.is_leaf(self):
            raise KeyError(f"PTree is a leaf, cannot get {path}")

        p, rest = path[0], path[1:]
        if p not in self.value:
            raise KeyError(f"Path {path} not found in ParamTree")

        if len(rest) == 0:
            return self.value[p].get()

        return self.value[p].get()[rest]

    def __setitem__(self, path, value):
        if self.read_only:
            raise RuntimeError("Cannot set value on read-only ParamTree")
        if not isinstance(path, ParamPath):
            path = ParamPath(path)
        # print(f"PTree SET item called with path {path} and value {value}")
        if len(path) == 0:
            raise KeyError(f"Path is empty")

        p, rest = path[0], path[1:]
        if self.value is None:
            self.value = {}
        if p not in self.value:
            self.value[p] = PTree(read_only=self.read_only)
        if len(rest) == 0:
            # if not PTree.is_leaf(self.value[p], count_none_as_leaf=True):
                # raise KeyError(f"Trying to assign value on non-leaf node!")
            self.value[p].value = value
        else:
            self.value[p].get()[rest] = value

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
        if not PTree.is_leaf(self):
            for v in self.value.values():
                v.set_read_only(ro)

    def all_leaves_are_none(self):
        # return jtu.tree_all(jtu.tree_map(lambda x: x is None, self))
        for _, v in self.iter_leaves():
            if v is not None:
                return False
        return True

    def remove_empty_leaves(self):
        newvals = {}

        if PTree.is_leaf(self, count_none_as_leaf=False):
            return self
        if self.value is not None:
            for k, v in self.value.items():
                print(f"PTree remove_empty_leaves: {k} {v}")
                if not v.all_leaves_are_none():
                    newvals[k] = v.remove_empty_leaves()

        return PTree(value=newvals, read_only=self.read_only)

    def iter_leaves(self, path=ParamPath()):
        if PTree.is_leaf(self):
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
        # print(f"TreeRef {id(self)} __init__ with path {path}")
        self.path = path  # the parampath pointing to the subtree
        self.tree = tree

    def __repr__(self):
        return f"TreeRef* ({self.path}): {self.tree[self.path]}"

    def get(self):
        # print(f"TreeRef {id(self)} *-> {id(self.tree)}[{self.path}] :: GET()")
        return self.tree[self.path]

    def __getitem__(self, path):
        # print(f"TreeRef {id(self)} GET for path:{path} (self.path={self.path})")
        subtree = self.tree[self.path]
        return subtree.__getitem__(path)

    def __setitem__(self, path, value):
        # print(f"TreeRef {id(self)} SET for path:{path} (self.path={self.path})")
        self.tree[self.path / path] = value

    def __eq__(self, other):
        if not isinstance(other, TreeRef):
            return False
        return self.path == other.path and self.tree[self.path] == other.tree[other.path]


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
        if len(self.indices) == 0:
            self.arrays = ()
            self.shape = ()
            self.map = ()
            return
        idx = np.asarray(self.indices)
        self.arrays = tuple([self.tree[p] for p in self.paths])
        self.shape = self.tree[self.paths[0]].shape
        self.map = ()
        for a in np.unique(idx[:, 0]):
            positions = np.where(idx[:, 0] == a)[0]
            ids_in_array = idx[positions, 1]
            self.map += ((a, positions, ids_in_array),)

    def get(self):
        N = len(self.indices)
        if N == 0:
            return jnp.array([])

        conc = jnp.zeros((N, *self.shape[1:]))
        for a, p, i in self.map:
            conc = conc.at[p].set(self.arrays[a][i])

        return conc

    def __eq__(self, other):
        if not isinstance(other, ArrayRef):
            return False
        return np.all(self.get() == other.get())

    def __hash__(self):
        return hash(self.get().tobytes())


@dataclass
class RefPath:
    actual_path: ParamPath
    points_to: ParamPath


@dataclass
class ArrayRefPath:
    actual_path: ParamPath
    paths: List[ParamPath]
    indices: List[Tuple[int, int]]


# I have to write my own manual (non-recursive) flatten to handle the references
def flatten_PTree(ptree):
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


def unflatten_PTree(aux_data, content):
    keys = aux_data[0]
    read_only = aux_data[1]
    ptree = PTree(read_only=read_only)
    for k, v in zip(keys, content):
        if isinstance(k, RefPath):
            ptree[k.actual_path] = TreeRef(k.points_to, ptree)
        elif isinstance(k, ArrayRefPath):
            ptree[k.actual_path] = ArrayRef(ptree, k.paths, k.indices)
        else:
            ptree[k] = v
    ptree.set_read_only(read_only)
    return ptree


jtu.register_pytree_node(PTree, flatten_PTree, unflatten_PTree)

# jtu.tree_leaves(ptree)
# make_jaxpr(ArrayRef.get, static_argnums=(0,))(ptree['arr/ref'])
# print(jit(ArrayRef.get, static_argnums=(0,)).lower(ptree['arr/ref']).compile().as_text())

ptree = PTree()
ptree['a/c'] = 2.0
ptree['a/b0'] = np.array([1, 2, 3], dtype=np.float32)
ptree['a/b1'] = np.array([4, 5], dtype=np.float32)
ptree.at('a/b/yo', np.array([5, 6, 7], np.float32))
assert ptree['a/b/yo'][1] == 6
assert ptree['a/c'] == 2.0
assert ptree['a']['c'] == 2.0

ptree['a/r'] = TreeRef('a/b', ptree)
ptree['a/r'].get()['newbranch'] = 3.0
ptree['a/r/newbranch'] = 2.2
assert ptree['a/r']['newbranch'] == 2.2
assert ptree['a/b']['newbranch'] == 2.2
assert ptree['a/b/newbranch'] == 2.2

ptree['arr/ref'] = ArrayRef(ptree, ['a/b0', 'a/b1'], [(0, 1), (1, 0)])
assert np.all(ptree['arr/ref'] == np.array([2, 4], dtype=np.float32))
ptree['a/b0'][1] = 42
assert np.all(ptree['arr/ref'] == np.array([42, 4], dtype=np.float32))


l, s = jtu.tree_flatten(ptree)
reconstructed = jtu.tree_unflatten(s, l)
assert reconstructed == ptree
assert np.all(reconstructed['arr/ref'] == np.array([42, 4], dtype=np.float32))

assert np.all(ptree['a/r']['yo'] == ptree['a/b/yo'])


def test_f(params, x):
    arr = params['arr/ref']
    return jnp.prod(arr) * x


g = jax.jit(jax.grad(test_f))(ptree, 2.0)

assert g['arr/ref'][0] > 0
assert g['arr/ref'][0] == g['a/b0'][1]
assert g['arr/ref'][1] == g['a/b1'][0]

ref = ArrayRef(ptree)
ptree['other/arrayref'] = ref
ref.push_back('a/b0', 2)
ptree
ref.push_back('a/b1', 1)

print('Passed!')

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

    def createReference(self, path, to):
        if self.read_only:
            raise RuntimeError("Cannot set value on read-only ParameterTree")
        self.data[path] = TreeRef(to, self.data)
        self.tags[path] = TreeRef(to, self.tags)

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
            self.tags = jtu.tree_map(lambda _: np.array([False]), self.data)
        else:
            insert_idx = self.__tagdict[tag]
            self.tags = jtu.tree_map(lambda t: np.insert(t, insert_idx, False), self.tags)

    def tag(self, path, tags, overwrite=False):
        if self.read_only:
            raise RuntimeError("Cannot tag read-only ParameterTree")
        self.create_tags_if_required(tags)
        if not self.data.has_leaf(path):
            raise KeyError(f"Trying to tag non-leaf node {path}")
        if isinstance(tags, str):
            tags = [tags]

        self.create_tags_if_required(tags)

        assert self.tags[path].shape == (len(self.tagnames),)
        print(f"tagging {path} with {tags}")
        print(f"current tags: {self.tags}")

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
        is_valid = jtu.tree_map(lambda x: np.all(x[tag_ids]), self.tags)

        left_data_tree = jtu.tree_map(lambda mask, x: x if mask else None, is_valid, self.data)
        right_data_tree = jtu.tree_map(lambda mask, x: x if not mask else None, is_valid, self.data)

        left_tag_tree = jtu.tree_map(lambda mask, x: x if mask else None, is_valid, self.tags)
        right_tag_tree = jtu.tree_map(lambda mask, x: x if not mask else None, is_valid, self.tags)

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
        jax_leaf = lambda x: x is None
        left_struct = jtu.tree_structure(left, is_leaf=jax_leaf)
        right_struct = jtu.tree_structure(right, is_leaf=jax_leaf)
        if left_struct == right_struct:
            return jtu.tree_map(lambda l, r: l if r is None else r, left, right, is_leaf=jax_leaf)
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


def flatten_ParameterTree(ptree):
    flat_contents = (ptree.data, ptree.tags)
    aux_data = (ptree.read_only, ptree.tagnames)
    return (flat_contents, aux_data)


def unflatten_ParameterTree(aux_data, flat_contents):
    data, tags = flat_contents
    read_only, tagnames = aux_data
    return ParameterTree(data=data, read_only=read_only, tags=tags, tagnames=tagnames)


jtu.register_pytree_node(ParameterTree, flatten_ParameterTree, unflatten_ParameterTree)


params = ParameterTree()
params['a/x'] = 2
params['a/b/c/d'] = 3
params.at('a/b/yo', np.array([1, 2, 4]))
assert params['a/b/c/d'] == 3
assert params['a/b/yo'][2] == 4

params.tag('a/b/c/d', ['test', 'b'])
assert np.all(params.tags['a/b/c/d'] == np.array([True, True]))
assert params.tagnames == ['b', 'test']

params.tag('a/x', ['othertag', 'aha'])
params.tag('a/b/yo', ['b'])
params

jtu.tree_leaves(params)
l, s = jtu.tree_flatten(params)
reconstructed = jtu.tree_unflatten(s, l)

a, b = params.filter_by_tag('aha')
a
b
assert a != b
b.remove_empty_leaves()
a.remove_empty_leaves()

m = ParameterTree.merge(a, b)
m_noempty = ParameterTree.merge(a.remove_empty_leaves(), b.remove_empty_leaves())
m
m_noempty
assert m_noempty == m

# let's try to add references

params.createReference('a/ref', 'a/b')
params['a/ref/c/d']
params.tag('a/ref/c/d', ['ref'])

assert params.tagnames == ['aha', 'b', 'othertag', 'ref', 'test']

assert np.all(params.tags['a/b/c/d'] == np.array([False, True, False, True, True]))
params.tags['a/b/c/d'] = np.array([False, False, False, True, True])
assert np.all(params.tags['a/ref/c/d/'] == np.array([False, False, False, True, True]))

print(params)
print('ParameterTree passed all assertions!')


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                         --     ParamRow     --


class ParamRow:
    def __init__(self, init_f=None):
        self.init_f = init_f
        self.data = None
        self.n_elems = 0

    def generate_all(self):
        self.data = vmap(self.init_f)(np.arange(self.n_elems))

    def get(self):
        return self.data

    def __getitem__(self, node_id):
        return self.data[node_id]

    def __setitem__(self, node_id, value):
        assert self.init_f is None
        self.n_elems = max(node_id + 1, self.n_elems)
        if self.data is None:
            self.data = np.zeros(self.n_elems)
        if self.data.shape[0] < self.n_elems:
            self.data = np.concatenate([self.data, np.zeros(self.n_elems - self.data.shape[0])])
        self.data[node_id] = value

    def __repr__(self):
        return f'ParamRow({self.data})'


def flatten_ParamRow(prow):
    return (prow.data, (prow.init_f, prow.n_elems))


def unflatten_ParamRow(aux_data, flat_contents):
    init_f, n_elems = aux_data
    prow = ParamRow(init_f)
    prow.n_elems = n_elems
    prow.data = flat_contents
    return prow


jtu.register_pytree_node(ParamRow, flatten_ParamRow, unflatten_ParamRow)


params['paramrow'] = ParamRow()
params['paramrow']

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



##


params = ParameterTree()

# set_param(
    # params,
    # "numeric:value",
    # init_val,
    # node_id=vnode.node_id,
    # number_of_nodes_at_least=maxid + 1,
# )

# def set_param(params, name, value, node_id=0, base_path=ut.NODE_PATH, **kw):
    # return PARAM_AT(
        # params, name, node_id, base_path, overwrite_with=np.asarray(value), read_only=False, **kw
    # )


params

params['numeric:value'] = 1

