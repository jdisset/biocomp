from typing import Callable, Optional, Union, Tuple, Any, Dict, List, Sequence, Iterable

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import numpy as np

from dataclasses import dataclass
from jax.tree_util import register_pytree_node_class
from copy import deepcopy
from . import utils as ut

import re


### {{{                           --     utils     --


def is_equal(a, b):
    if not type(a) == type(b):
        return False
    if isinstance(a, (np.ndarray, jnp.ndarray)):
        return np.all(a == b)
    return a == b


def pretty_str(x):
    msg = ''
    if isinstance(x, str):
        msg = x
    elif isinstance(x, (np.ndarray, jnp.ndarray)):
        if x.size <= 15:
            with np.printoptions(precision=3):
                msg = str(x)
        else:
            with np.printoptions(precision=3, edgeitems=2, threshold=5):
                typestr = "jax" if isinstance(x, jnp.ndarray) else "numpy"
                msg = f"{x.shape} {x.dtype} {typestr} array:\n{np.asarray(x)}"

    elif isinstance(x, (list, tuple)):
        chars = '()' if isinstance(x, tuple) else '[]'
        max_lines = 5
        max_elem_per_line = 3
        # xspl = x[:::max_elem_per_line]
        # make chunks of max_elem_per_line
        xspl = [x[i : i + max_elem_per_line] for i in range(0, len(x), max_elem_per_line)]
        msg = chars[0] + '\n'.join([', '.join(x) for x in xspl[:max_lines]])
        if len(xspl) > max_lines:
            msg += f"\n... {len(xspl) - max_lines} more elements"
        msg += chars[1]

    else:
        msg = str(x)

    return msg + "\n"


##────────────────────────────────────────────────────────────────────────────}}}


### {{{                         --     ParamPath     --


class ParamPath:
    @staticmethod
    def psplit(key):
        if isinstance(key, str):
            return key.strip("/").split("/")
        return key

    @staticmethod
    def tostr(key):
        if isinstance(key, str):
            return key
        return "/".join(key)

    def __init__(self, path=None):
        self.path = ParamPath.psplit(path) or []
        self._str = ParamPath.tostr(self.path)

    def __truediv__(self, key):
        if isinstance(key, str):
            key = ParamPath.psplit(key)
        elif isinstance(key, ParamPath):
            key = key.path
        return ParamPath(self.path + key)

    def __add__(self, key):
        return self.__truediv__(key)

    def __repr__(self):
        return self._str

    def __str__(self):
        return self._str

    def __getitem__(self, key):
        return self.path[key]

    def __len__(self):
        return len(self.path)

    def __iter__(self):
        return iter(self.path)

    def __eq__(self, other):
        return str(self) == str(other)

    def __lt__(self, other):
        return str(self) < str(other)

    def __gt__(self, other):
        return str(self) > str(other)

    def __hash__(self):
        return hash(str(self))

    def __contains__(self, key):
        if isinstance(key, str):
            key = ParamPath.psplit(key)
        elif isinstance(key, ParamPath):
            key = key.path
        return key in self.path


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                        --     Ptree     --
class PTree:
    def __init__(self, value=None, read_only=False):
        self.value = value
        self.set_read_only(read_only)

    @staticmethod
    def is_leaf(tree, count_none_as_leaf=True):
        if not isinstance(tree, PTree):
            return True
        if tree.value is None:
            return count_none_as_leaf
        if not isinstance(tree.value, dict):
            return True
        if len(tree.value) == 0:
            return True
        return False

    def __contains__(self, key):
        if PTree.is_leaf(self):
            return False
        try:
            self[key]
            return True
        except KeyError:
            return False

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
                valstr = pretty_str(self.value) if self.value is not None else "∅"
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

    def get(self, follow_ref=True):
        if PTree.is_leaf(self, count_none_as_leaf=False):
            # does this value itself have a get method? (e.g. TreeReferences do...)
            if follow_ref and hasattr(self.value, "get"):
                return self.value.get()  # alows to follow references
            return self.value
        return self

    def get_at(self, path, follow_ref=True):
        if not isinstance(path, ParamPath):
            path = ParamPath(path)
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
            return self.value[p].get(follow_ref)

        return self.value[p].get_at(rest, follow_ref)

    def __getitem__(self, path):
        return self.get_at(path, follow_ref=True)

    def __setitem__(self, path, value):
        if self.read_only:
            raise RuntimeError("Cannot set value on read-only ParamTree")
        if not isinstance(path, ParamPath):
            path = ParamPath(path)
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

    def at(self, path, value=None, overwrite=False, follow_ref=True):
        if self.read_only or value is None:
            return self.get_at(path, follow_ref)
        else:
            try:
                self[path]
            except KeyError:
                overwrite = True
            if overwrite:
                self[path] = value
            return self.get_at(path, follow_ref)

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
                if not v.all_leaves_are_none():
                    newvals[k] = v.remove_empty_leaves()

        return PTree(value=newvals, read_only=self.read_only)

    def iter_leaves(self, path=ParamPath(), path_as_str=False):
        if PTree.is_leaf(self):
            if path_as_str:
                yield str(path), self.value
            else:
                yield path, self.value
        else:
            for k, v in self.value.items():
                yield from v.iter_leaves(path / k)

    def __eq__(self, other):
        if not isinstance(other, PTree):
            return False
        k1, k2 = set(self.value.keys()), set(other.value.keys())
        if k1 != k2:
            return False
        for k in k1:
            if not is_equal(self.get_at(k, follow_ref=False), other.get_at(k, follow_ref=False)):
                return False
        return True


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                          --     TreeRef     --
class TreeRef:
    """A reference to a subtree of a ParamTree"""

    def __init__(self, path, tree):
        self.path = path  # the parampath pointing to the subtree
        self.tree = tree

    def __repr__(self):
        return f"TreeRef* ({self.path}): {self.tree[self.path]}"

    def get(self):
        return self.tree[self.path]

    def __getitem__(self, path):
        subtree = self.tree[self.path]
        return subtree.__getitem__(path)

    def __setitem__(self, path, value):
        self.tree[self.path / path] = value

    def __eq__(self, other):
        if not isinstance(other, TreeRef):
            return False
        return self.path == other.path and self.tree[self.path] == other.tree[other.path]


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                         --     ArrayRef     --


class ArrayRef:

    """An array of references to some other arrays values
    aka a view, but over potentially several different arrays
    """

    def __init__(self, tree, paths=None, indices=None):
        self.tree = tree
        self.indices = indices or ()  # tuple of (array_num, index0, index1, ...) coordinates
        self.paths = paths or ()  # tuple of paths to the referenced arrays
        self._pathdict = {p: i for i, p in enumerate(self.paths)}
        self.make_map()

    def __repr__(self):
        return f"RefArray from {len(self.paths)} pointed arrays:\n{pretty_str(self.get())}"

    def push_back(self, array_path, id):

        if not isinstance(id, (list, tuple)):
            id = (id,)

        if array_path not in self.paths:
            self.paths += (array_path,)
            self._pathdict[array_path] = len(self.paths) - 1

        self.indices += ((self._pathdict[array_path], *id),)
        self.make_map()

    def get(self):
        N = len(self.indices)
        if N == 0:
            return jnp.array([])

        arrays = tuple([self.tree[p] for p in self.paths])

        a0, _, i0 = self.map[0]
        shape = arrays[a0][i0][0].shape

        conc = jnp.zeros((N, *shape), dtype=arrays[0].dtype)

        for a, p, i in self.map:
            # print(f'a shape: {arrays[a][i].shape}')
            # print(f'conc shape: {conc.shape}')
            # print(f'p: {p}')
            # print(f'i: {i}')
            # a is the array number
            # p is the position in the concatenated array
            # i are the coordinates of the value in the array)
            conc = conc.at[p].set(arrays[a][i])

        return conc

    def make_map(self):
        if len(self.indices) == 0:
            self.map = ()
            return
        idx = np.asarray(self.indices)
        self.map = ()
        for a in np.unique(idx[:, 0]):
            positions = np.where(idx[:, 0] == a)[0]
            ids_in_array = idx[positions, 1:]
            idtup = tuple(ids_in_array.T) # transpose to get a tuple of arrays of indices
            self.map += ((a, positions, idtup),)


    def __eq__(self, other):
        if not isinstance(other, ArrayRef):
            return False
        return self.indices == other.indices and self.paths == other.paths

    def __hash__(self):
        return hash(self.get().tobytes())


##────────────────────────────────────────────────────────────────────────────}}}


def isRef(x):
    return isinstance(x, (TreeRef, ArrayRef))


### {{{                 --     jax [un]flattening of Ptrees     --


@dataclass
class RefPath:
    actual_path: ParamPath
    points_to: ParamPath

    def __eq__(self, other):
        if not isinstance(other, RefPath):
            return False
        return self.actual_path == other.actual_path and self.points_to == other.points_to

    def __lt__(self, other):
        if isinstance(other, ParamPath):
            return self.actual_path < other
        return self.actual_path < other.actual_path

    def __gt__(self, other):
        if isinstance(other, ParamPath):
            return self.actual_path > other
        return self.actual_path > other.actual_path


@dataclass
class ArrayRefPath:
    actual_path: ParamPath
    paths: List[ParamPath]
    indices: List[Tuple[int, int]]

    def __eq__(self, other):
        if not isinstance(other, ArrayRefPath):
            return False
        return (
            self.actual_path == other.actual_path
            and self.paths == other.paths
            and self.indices == other.indices
        )

    def __lt__(self, other):
        if isinstance(other, ParamPath):
            return self.actual_path < other
        return self.actual_path < other.actual_path

    def __gt__(self, other):
        print("gt", self.actual_path, other)
        if isinstance(other, ParamPath):
            return self.actual_path > other
        return self.actual_path > other.actual_path


# I have to write my own manual (non-recursive) flatten function to handle the references
# As a result, this is VERY SLOW compared to a normal jax tree flatten
# (which calls the c++ xla implementation)
# one way around this would be to wrap the flatten and
# flag the references in the tree as leaves before postprocessing
# but for now I'll just use this slow version
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

    order = sorted(range(len(keys)), key=lambda i: keys[i])
    sorted_keys = [keys[i] for i in order]
    sorted_values = [values[i] for i in order]

    aux_data = (sorted_keys, ptree.read_only)
    return (sorted_values, aux_data)


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

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                        --     ParameterTree     --


class ParameterTree:
    """A tree of parameters, with a separate tree of tags + some convenience partitionning methods"""

    def __init__(
        self, data: PTree = None, tags: PTree = None, tagnames: list = None, read_only=False
    ):
        self.data = data or PTree()
        self.tags = tags or PTree()
        self.tagnames = tagnames or []
        self.__tagdict = {name: i for i, name in enumerate(self.tagnames)}
        self.set_read_only(read_only)

    def set_read_only(self, read_only):
        self.read_only = read_only
        self.data.set_read_only(read_only)
        if self.tags is not None:
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
                self[path] = value
                self.tag(path, tags, True)
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
        is_leaf = lambda x: PTree.is_leaf(x) and not isinstance(x, ParameterTree)

        if self.tags is None or self.tags.is_empty():
            self.tags = jtu.tree_map(lambda _: np.array([False]), self.data, is_leaf=is_leaf)
        else:
            insert_idx = self.__tagdict[tag]
            self.tags = jtu.tree_map(
                lambda t: np.insert(t, insert_idx, False), self.tags, is_leaf=is_leaf
            )

    def get_tag_flags(self, tags):
        if isinstance(tags, str):
            tags = [tags]
        tag_ids = [self.__tagdict[tag] for tag in tags]
        tag_flags = np.zeros(len(self.tagnames), dtype=bool)
        tag_flags[tag_ids] = True
        return tag_flags

    def tag(self, path, tags, overwrite=False):
        if tags is None:
            return
        if self.read_only:
            raise RuntimeError("Cannot tag read-only ParameterTree")
        self.create_tags_if_required(tags)
        if not self.data.has_leaf(path):
            raise KeyError(f"Trying to tag non-leaf node {path}")
        if isinstance(tags, str):
            tags = [tags]

        self.create_tags_if_required(tags)

        assert self.tags[path].shape == (len(self.tagnames),)

        tag_flags = self.get_tag_flags(tags)

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

    # def filter_by_tag(self, tags):
    # if isinstance(tags, str):
    # tags = [tags]
    # for t in tags:
    # if t not in self.tagnames:
    # raise KeyError(f"Tag {t} not found in ParameterTree")
    # tag_ids = [self.__tagdict[tag] for tag in tags]
    # is_valid = jtu.tree_map(lambda x: np.all(x[tag_ids]), self.tags)
    # left_data_tree = jtu.tree_map(lambda mask, x: x if mask else None, is_valid, self.data)
    # right_data_tree = jtu.tree_map(lambda mask, x: x if not mask else None, is_valid, self.data)
    # left_tag_tree = jtu.tree_map(lambda mask, x: x if mask else None, is_valid, self.tags)
    # right_tag_tree = jtu.tree_map(lambda mask, x: x if not mask else None, is_valid, self.tags)
    # left_param_tree = ParameterTree(
    # data=left_data_tree,
    # tags=left_tag_tree,
    # tagnames=self.tagnames,
    # read_only=self.read_only,
    # )
    # right_param_tree = ParameterTree(
    # data=right_data_tree,
    # tags=right_tag_tree,
    # tagnames=self.tagnames,
    # read_only=self.read_only,
    # )
    # return left_param_tree, right_param_tree

    # @classmethod
    # def merge(cls, left, right):
    # jax_leaf = lambda x: x is None
    # left_struct = jtu.tree_structure(left, is_leaf=jax_leaf)
    # right_struct = jtu.tree_structure(right, is_leaf=jax_leaf)
    # if left_struct == right_struct:
    # return jtu.tree_map(lambda l, r: l if r is None else r, left, right, is_leaf=jax_leaf)
    # else:
    # return cls.sparse_merge(left, right)

    def filter_by_tag(self, tags):
        if isinstance(tags, str):
            tags = [tags]
        for t in tags:
            if t not in self.tagnames:
                raise KeyError(f"Tag {t} not found in ParameterTree")
        tag_ids = [self.__tagdict[tag] for tag in tags]
        left_param_tree = ParameterTree(
            tagnames=self.tagnames,
            read_only=False,
        )
        right_param_tree = ParameterTree(
            tagnames=self.tagnames,
            read_only=False,
        )

        for path, data in self.data.iter_leaves():
            tag_flags = self.tags[path]
            if np.all(tag_flags[tag_ids]):
                left_param_tree.data[path] = data
                left_param_tree.tags[path] = tag_flags
            else:
                right_param_tree.data[path] = data
                right_param_tree.tags[path] = tag_flags

        left_param_tree.set_read_only(self.read_only)
        right_param_tree.set_read_only(self.read_only)

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

    @staticmethod
    def datadiff(left, right):
        paths = set()
        for path, left_data in left.data.iter_leaves():
            if path not in right.data:
                paths.add(path)
                continue
            right_data = right.data.get_at(path, follow_ref=False)
            if not is_equal(left_data, right_data):
                print('diff')
                print(f"left: {left_data}")
                print(f"right: {right_data}")
                paths.add(path)
        for path, right_data in right.data.iter_leaves():
            if path not in left.data:
                paths.add(path)
                continue
        return paths

    @staticmethod
    def merge(left, right):
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

##────────────────────────────────────────────────────────────────────────────}}}


def init_if_needed(params, path, init_f, base_path=''):
    try:
        return params[f'{base_path}/{path}']
    except KeyError:
        params[f'{base_path}/{path}'] = init_f()
        return params[f'{base_path}/{path}']


def get_param(params, path, base_path='', **_):
    return params[f'{base_path}/{path}']


def make_view(
    params: ParameterTree,
    at_path: ParamPath,
    from_paths: Sequence[ParamPath],
    from_ids: Sequence[int],
    leaves: Sequence[ParamPath],
):
    for leaf in leaves:
        leafpath = ParamPath(at_path) / leaf
        ref = ArrayRef(params.data)
        for from_path, from_id in zip(from_paths, from_ids):
            ref.push_back(f'{from_path}/{leaf}', from_id)
        params[leafpath] = ref
