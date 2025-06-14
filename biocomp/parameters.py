from typing import (
    Callable,
    Optional,
    Union,
    Tuple,
    Any,
    Dict,
    List,
    Sequence,
    Iterable,
    Self,
    TypeVar,
    Type,
)
import traceback
import jax
import jax.numpy as jnp
import h5py
import jax.tree_util as jtu
from jaxlib.xla_extension import ArrayImpl

import numpy as np

from dataclasses import dataclass
from jax.tree_util import register_pytree_node_class
from copy import deepcopy
from . import utils as ut

import base64
import re

logger = ut.get_logger(__name__)

### {{{                           --     utils     --


def isArrayRef(x):
    return str(type(x)) == str(ArrayRef)


def is_equal(a, b):
    if id(a) == id(b):
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, (np.ndarray, jnp.ndarray)):
        return np.all(a == b)
    if isinstance(a, PTree):
        return a.direct_compare(b)
    return a == b


def pretty_str(x):
    msg = ""
    if isinstance(x, str):
        msg = x
    elif isinstance(x, (np.ndarray, jnp.ndarray)):
        if x.size <= 15:
            with np.printoptions(precision=3):
                msg = str(x)
        else:
            with np.printoptions(precision=3, edgeitems=4, threshold=8):
                typestr = "jax" if isinstance(x, jnp.ndarray) else "numpy"
                msg = f"{x.shape} {x.dtype} {typestr} array:\n{np.asarray(x)}"

    elif isinstance(x, (list, tuple)):
        chars = "()" if isinstance(x, tuple) else "[]"
        max_lines = 5
        max_elem_per_line = 5
        # xspl = x[:::max_elem_per_line]
        # make chunks of max_elem_per_line
        xspl = [x[i : i + max_elem_per_line] for i in range(0, len(x), max_elem_per_line)]
        msg = chars[0] + "\n".join([", ".join(str(x)) for x in xspl[:max_lines]])
        if len(xspl) > max_lines:
            msg += f"\n... {len(xspl) - max_lines} more elements"
        msg += chars[1]

    else:
        msg = str(x)

    return msg + "\n"


##────────────────────────────────────────────────────────────────────────────}}}


### {{{                       --     serialization     --

# maintain a dict of type to a serialize or deserialize functions

serializers = {}
deserializers = {}

jnparr = ArrayImpl


def register_serializer(cls, func):
    if isinstance(cls, str):
        serializers[cls] = func
    else:
        serializers[cls.__name__] = func


def register_deserializer(cls, func):
    if isinstance(cls, str):
        deserializers[cls] = func
    else:
        deserializers[cls.__name__] = func


def serializer(types: Union[Sequence, type]):
    if isinstance(types, type):
        types = (types,)

    def decorator(function):
        for t in types:
            register_serializer(t, function)
        return function

    return decorator


def deserializer(types: Union[Sequence, type]):
    if isinstance(types, type):
        types = (types,)

    def decorator(function):
        for t in types:
            register_deserializer(t, function)
        return function

    return decorator


def serialize(x):
    if type(x).__name__ in serializers:
        return serializers[type(x).__name__](x)
    else:
        raise ValueError(f"Cannot serialize type {type(x)}")


# ArrayLike
@serializer((np.ndarray, jnparr))
def serialize_arraylike(x):
    x = np.asarray(x)
    b64data = base64.b64encode(x.tobytes())
    return {
        "type": type(x).__name__,
        "dtype": str(x.dtype),
        "shape": x.shape,
        "data": b64data.decode("utf-8"),
    }


@deserializer(np.ndarray)
def deserialize_arraylike(x):
    dtype = np.dtype(x["dtype"])
    shape = tuple(x["shape"])
    b64data = x["data"].encode("utf-8")
    data = np.frombuffer(base64.decodebytes(b64data), dtype=dtype)
    return data.reshape(shape)


# ListLike
@serializer((list, tuple))
def serialize_listlike(x):
    return {"type": type(x).__name__, "data": tuple(serialize(xi) for xi in x)}


@deserializer((list, tuple))
def deserialize_listlike(x):
    return type(x["type"])(deserialize(xi) for xi in x["data"])


# Dict
@serializer(dict)
def serialize_dict(x):
    return {"type": type(x).__name__, "data": {serialize(k): serialize(v) for k, v in x.items()}}


def serialize_PTree(x):
    return jtu.tree_map(serialize, x)


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                         --     ParamPath     --


class ParamPath:
    @staticmethod
    def psplit(key):
        if key is None:
            return []
        if isinstance(key, str):
            return key.strip("/").split("/")
        elif isinstance(key, ParamPath):
            return key.path
        elif isinstance(key, (list, tuple)):
            return key
        else:
            raise ValueError(f"Invalid key type: {type(key)}")

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

    def __lt__(self, other):
        if isinstance(other, ParamPath):
            for i in range(min(len(self), len(other))):
                if self[i] == other[i]:
                    continue
                return self[i] < other[i]
            return len(self) < len(other)
        if isinstance(other, str):
            return self < ParamPath(other)
        return other > self  # delegate to __gt__ of the other type

    def __gt__(self, other):
        if isinstance(other, ParamPath):
            for i in range(min(len(self), len(other))):
                if self[i] == other[i]:
                    continue
                return self[i] > other[i]
            return len(self) > len(other)
        if isinstance(other, str):
            return self > ParamPath(other)
        return other < self  # delegate to __lt__ of the other type

    def __eq__(self, other):
        if isinstance(other, ParamPath):
            return self.path == other.path
        if isinstance(other, str):
            return self.path == ParamPath.psplit(other)
        return other == self  # delegate to __eq__ of the other type

    def __hash__(self):
        return hash(str(self))

    def __contains__(self, key):
        if not isinstance(key, ParamPath):
            key = ParamPath(key)
        n = len(key)
        for i in range(len(self.path)):
            if self.path[i : i + n] == key.path:
                return True
        return False


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                        --     Ptree     --


class PTreeBranch(dict):
    pass


class PTree:
    # a PTree is a tree of trees, aka, a tree.
    # The children of a tree are stored in the self.value member variable.
    # If the tree is a leaf, then self.value is the value of the leaf
    # If the tree is a branch, then self.value is a PTreeBranch, which is a dict
    # (using a different type name than dict to avoid confusion with a possible
    # leaf value that would be a dict, which is allowed).
    # None is a special value for self.value, which is used to represent an empty tree.
    #
    # The goal of PTrees is to be fully compatible with jax.tree_util functions, i.e a PTree is a pytree.
    # During flattening/unflattening PTree are "aware" of a special leaf type: ArrayRef, which is a composite
    # view of multiple arrays.

    def __init__(self, value=None, read_only=False):
        self.value = value
        self.set_read_only(read_only)

    def is_empty(self):
        return self.value is None

    def is_leaf(self, count_empty_as_leaf=True):
        if self.is_empty():
            return count_empty_as_leaf
        return not isinstance(self.value, PTreeBranch)

    def is_leaf_at(self, path, count_empty_as_leaf=True):
        branch = self.get_at(path, get_leaf_value=False)
        return branch.is_leaf(count_empty_as_leaf)

    def visualize_tree_structure(self, seen=None, depth=0, prefix=""):
        if seen is None:
            seen = {}

        result = []
        indent = "  " * depth
        node_id = id(self)

        # Check for loops
        if node_id in seen:
            return [f"{indent}↩ Loop back to #{node_id} (at {seen[node_id]})"]

        seen[node_id] = prefix or "root"

        if self.is_leaf():
            val = self.value
            if isinstance(val, PTree):
                result.append(f"{indent}● #{node_id} [Leaf -> PTree]")
                result.extend(
                    val.visualize_tree_structure(seen, depth + 1, prefix=prefix + "/value")
                )
            elif isinstance(val, (np.ndarray, jnp.ndarray)):
                result.append(f"{indent}● #{node_id} [Leaf Array shape={val.shape}]")
            else:
                result.append(f"{indent}● #{node_id} [Leaf {type(val).__name__}]")
        else:
            result.append(f"{indent}○ #{node_id} [Branch]")
            if self.value:
                for key in sorted(self.value.keys()):
                    result.append(f"{indent}├─{key}")
                    child_lines = self.value[key].visualize_tree_structure(
                        seen, depth + 1, prefix=f"{prefix}/{key}"
                    )
                    for i, line in enumerate(child_lines):
                        if i < len(child_lines) - 1:
                            result.append(f"{indent}│ {line}")
                        else:
                            result.append(f"{indent}└ {line}")

        return result

    def print_structure(self):
        """Print the tree structure"""
        print("Tree Structure:")
        print("Legend:")
        print("  ○ Branch node")
        print("  ● Leaf node")
        print("  ↩ Loop reference")
        print("  # Object ID")
        print("-" * 50)
        print("\n".join(self.visualize_tree_structure()))

    def direct_compare(self, other, depth=0, seen=None, path=None):
        """Direct value comparison without using is_equal to avoid recursion"""
        if seen is None:
            seen = {}  # Change to dict to store paths
        if path is None:
            path = []

        pair_id = (id(self), id(other))
        if pair_id in seen:
            logger.error(f"Loop detected when comparing PTree({id(self)}) with PTree({id(other)})")
            logger.error("Path to loop:")
            prev_path = seen[pair_id]
            full_path = prev_path + [f"-> loop back to PTree({id(self)})"]
            for step in full_path:
                logger.error(f"  {step}")
            logger.error("Tree structure:")
            logger.error("\nSelf:")
            logger.error("\n".join(self.visualize_tree_structure()))
            logger.error("\nOther:")
            logger.error("\n".join(other.visualize_tree_structure()))
            return False

        seen[pair_id] = path[:]

        if not isinstance(other, PTree):
            return False

        this_is_leaf, other_is_leaf = self.is_leaf(), other.is_leaf()
        if this_is_leaf != other_is_leaf:
            return False

        if this_is_leaf:
            if type(self.value) is not type(other.value):
                return False

            if isinstance(self.value, (np.ndarray, jnp.ndarray)):
                return np.all(self.value == other.value)

            if isinstance(self.value, PTree):
                path.append(f"leaf value PTree({id(self.value)})")
                return self.value.direct_compare(other.value, depth + 1, seen, path)

            return self.value == other.value

        # Branch comparison
        k1, k2 = set(self.value.keys()), set(other.value.keys())

        if k1 != k2:
            return False

        for k in k1:
            a = self.get_at(k, get_leaf_value=False)
            b = other.get_at(k, get_leaf_value=False)
            path.append(f"branch '{k}' -> PTree({id(a)})")
            if not a.direct_compare(b, depth + 1, seen, path):
                return False
            path.pop()

        return True

    def get_at(self, path, get_leaf_value=True):
        if not isinstance(path, ParamPath):
            path = ParamPath(path)
        if len(path) == 0:
            raise KeyError(f"PTree get_at called with empty path")
        if self.is_empty():
            raise KeyError(f"PTree is empty, cannot get {path}")
        if self.is_leaf(self):
            raise KeyError(f"PTree is a leaf, cannot get {path}")
        assert isinstance(self.value, PTreeBranch), f"self.value is not a PTreeBranch: {self.value}"

        p, rest = path[0], path[1:]
        if p not in self.value:
            raise KeyError(f"Path {path} not found in ParamTree")

        if len(rest) == 0:
            return self.value[p].get(get_leaf_value=get_leaf_value)

        return self.value[p].get_at(rest, get_leaf_value)

    def get(self, get_leaf_value=True):
        """Return the value of the leaf, or the value of the branch if it is not a leaf.
        If the leaf is an ArrayRef and get_leaf_value is True, return the view of the array.
        """
        if self.is_leaf() and get_leaf_value:
            if isArrayRef(self.value):
                return self.value.view()
            return self.value
        return self

    def set_at(self, path, value):
        """if set_leaf_value is true, set the .value of the leaf node to value,
        otherwise, set the branch node to value
        """
        if self.read_only:
            raise RuntimeError("Cannot set value on read-only ParamTree")
        if not isinstance(path, ParamPath):
            path = ParamPath(path)
        if len(path) == 0:
            raise KeyError(f"Path is empty")

        p, rest = path[0], path[1:]
        if self.value is None:
            self.value = PTreeBranch()
        if p not in self.value:
            self.value[p] = PTree(read_only=self.read_only)
        if len(rest) == 0:
            self.value[p].value = value
        else:
            if self.is_leaf_at(p, count_empty_as_leaf=False):
                raise KeyError(
                    f"Trying to expand leaf node into branch is not allowed, delete leaf first"
                )
            self.value[p].get(False).set_at(rest, value)

    def at(self, path, value=None, overwrite=False, leaf_value=True):
        if self.read_only or value is None:
            return self.get_at(path, get_leaf_value=leaf_value)
        else:
            try:
                self[path]
            except KeyError:
                overwrite = True
            if overwrite:
                self.set_at(path, value)
            return self.get_at(path, get_leaf_value=leaf_value)

    def __getitem__(self, path):
        return self.get_at(path, get_leaf_value=True)

    def __setitem__(self, path, value):
        self.set_at(path, value)

    def __delitem__(self, path):
        if self.read_only:
            raise RuntimeError("Cannot delete value on read-only ParamTree")
        if not isinstance(path, ParamPath):
            path = ParamPath(path)
        if len(path) == 0:
            raise KeyError(f"Path is empty")
        p, rest = path[0], path[1:]
        if self.value is None:
            raise KeyError(f"Path {path} not found in ParamTree")
        if p not in self.value:
            raise KeyError(f"Path {path} not found in ParamTree")
        if len(rest) == 0:
            del self.value[p]
            if len(self.value) == 0:
                self.value = None
        else:
            del self.value[p][rest]

    def __len__(self):
        return len(list(self.iter_leaves()))

    def __contains__(self, key):
        if self.is_leaf(count_empty_as_leaf=True):
            return False
        try:
            self.get_at(key)
        except KeyError:
            return False
        return True

    def getpretty(self, levels=None, key=None):
        s = ""
        if levels == None:
            if self.is_leaf(count_empty_as_leaf=False):
                return f"{self.get(get_leaf_value=True)}\n"
            if self.is_empty():
                return " ∅\n"
            s += f"\n ▼"
            s += self.getpretty([]) + "\n\n"
        else:
            other_branches = [" │  " if l else "    " for l in levels]
            lineheader = f"\n{''.join(other_branches)}"
            if self.is_leaf():
                keylen = len(key) if key is not None else 0
                valstr = pretty_str(self.value) if self.value is not None else "∅"
                valstr = valstr.replace("\n", f"{lineheader}{' ' * keylen}     ")
                s += f" ⟶ {valstr}"
            else:
                nitems = len(self.value.items())
                for i, (k, v) in enumerate(self.value.items()):
                    this_branch_char = " └─ " if i == nitems - 1 else " ├─ "
                    s += f"{lineheader}{this_branch_char}'{k}'"
                    s += v.getpretty(levels + [i < nitems - 1], k)
        return s

    def __str__(self):
        return self.getpretty()

    def __repr__(self):
        return self.getpretty()

    def get_read_only_copy(self):
        from copy import deepcopy

        cop = deepcopy(self)
        cop.set_read_only(True)
        return cop

    def set_read_only(self, ro=True):
        self.read_only = ro
        if not self.is_leaf():
            for v in self.value.values():
                v.set_read_only(ro)

    def all_leaves_are_none(self):
        for _, v in self.iter_leaves():
            if v is not None:
                return False
        return True

    def iter_leaves(self, path=ParamPath(), path_as_str=False, get_leaf_value=True):
        if self.is_empty():
            return
        if self.is_leaf(count_empty_as_leaf=False):
            retval = self.value if get_leaf_value else self
            p = str(path) if path_as_str else path
            yield p, retval
        else:
            for k, v in self.value.items():
                yield from v.iter_leaves(path / k, path_as_str, get_leaf_value)

    def __eq__(self, other):
        return is_equal(self, other)

    def diff(self, other):
        diffs = set()
        for k, v in self.iter_leaves(get_leaf_value=False):
            assert isinstance(v, PTree), f"this branch at {k} is {type(v)}"
            if k not in other:
                diffs.add(k)
            else:
                otherb = other.get_at(k, get_leaf_value=False)
                assert isinstance(otherb, PTree), f"other branch at {k} is {type(otherb)}"
                if not is_equal(v.value, otherb.value):
                    diffs.add(k)

        for k, v in other.iter_leaves(get_leaf_value=False):
            assert isinstance(v, PTree), f"other branch at {k} is {type(v)}"
            if k not in self:
                diffs.add(k)

        return diffs

    def check(self):
        for k, nd in self.iter_leaves(get_leaf_value=False):
            assert isinstance(nd, PTree), f"branch at {k} is {type(nd)}"
            if isArrayRef(nd.value):
                assert nd.value.tree is self, (
                    f"branch at {k} has wrong tree: {id(nd.value.tree)} != {id(self)}"
                )


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                         --     ArrayRef     --


class ArrayRef:
    """An array of references to some other arrays values
    aka a view, but over potentially several different arrays
    """

    def __init__(self, tree: PTree, paths=None, indices=None):
        assert isinstance(tree, PTree), f"tree must be a PTree, not {type(tree)}"
        self.tree = tree
        self.indices = indices or ()  # tuple of (array_num, index0, index1, ...) coordinates
        self.paths = paths or ()  # tuple of paths to the referenced arrays
        self._pathdict = {p: i for i, p in enumerate(self.paths)}
        self.make_map()

    def __repr__(self):
        r = f"RefArray from {len(self.paths)} pointed arrays:\n"
        for a, p, i in self.map:
            r += f"* {self.paths[a]}: ({len(i[0])} elmts)\n"
        return r

    def push_back(self, array_path, id):
        if not isinstance(id, (list, tuple)):
            id = (id,)

        if array_path not in self.paths:
            self.paths += (array_path,)
            self._pathdict[array_path] = len(self.paths) - 1

        self.indices += ((self._pathdict[array_path], *id),)
        self.make_map()

    def view(self):
        N = len(self.indices)
        if N == 0:
            return jnp.array([])

        arrays = tuple([self.tree[p] for p in self.paths])

        a0, _, i0 = self.map[0]
        shape = arrays[a0][i0][0].shape

        conc = jnp.zeros((N, *shape), dtype=arrays[0].dtype)

        for a, p, i in self.map:
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
            idtup = tuple(ids_in_array.T)  # transpose to get a tuple of arrays of indices
            self.map += ((a, positions, idtup),)

    def __eq__(self, other):
        if not isArrayRef(other):
            return False
        return self.indices == other.indices and self.paths == other.paths

    def __hash__(self):
        return hash(self.get().tobytes())


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                 --     jax [un]flattening of Ptrees     --


@dataclass(order=False)
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
        if isArrayRef(v):
            values.append(None)
            keys.append(ArrayRefPath(ParamPath(k), v.paths, v.indices))
        else:
            values.append(v)
            keys.append(ParamPath(k))
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    sorted_keys = tuple(keys[i] for i in order)
    sorted_values = tuple(values[i] for i in order)
    aux_data = (sorted_keys, ptree.read_only)
    return (sorted_values, aux_data)


def unflatten_PTree(aux_data, content):
    keys = aux_data[0]
    read_only = aux_data[1]
    ptree = PTree(read_only=False)
    for k, v in zip(keys, content):
        if isinstance(k, ArrayRefPath):
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
            exists = path in self.data
            if overwrite is None and exists:
                raise KeyError(f"Path {path} already exists, cant overwrite without overwrite=True")
            if overwrite or not exists:
                self[path] = value
                self.tag(path, tags, True)
            return self.data[path]

    def visualize_tree_structure(self):
        return "\n".join(self.data.visualize_tree_structure())

    def get_subtree(self, path):
        return ParameterTree(
            data=self.data[path],
            tags=self.tags[path],
            tagnames=self.tagnames,
            read_only=self.read_only,
        )

    def set_subtree_at(self, path: str, subtree: PTree, overwrite=True):
        """Set all leaf values from subtree at the given path, preserving tree structure."""
        if self.read_only:
            raise RuntimeError("Cannot set value on read-only ParameterTree")

        base_path = ParamPath(path)
        if base_path in self.data:
            del self.data[base_path]

        for leaf_path, value in subtree.iter_leaves():
            full_path = base_path / leaf_path
            self.data[full_path] = value
            # Preserve tags if they exist
            if path in self.tags:
                old_tags = self.tags[path]
                self.tags[full_path] = old_tags

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
            [" "]
            + ["data"]
            + data_splitlines[1:]
            + ["╴" * (lw - 1)]
            + [" "]
            + [f"tags [{tags_str}]"]
            + tag_splitlines[1:]
        )
        # s = with_box(
        content_str = "\n".join(content)
        inner_s = f"Parameter Tree ({'RO' if self.read_only else 'RW'})\n{content_str}\n"
        return inner_s

    def create_tags_if_required(self, tags):
        if isinstance(tags, str):
            tags = [tags]
        for tag in tags:
            if tag not in self.tagnames:
                self.add_new_tag(tag)

    def add_new_tag(self, tag):
        if self.read_only:
            raise RuntimeError("Cannot add new tag on read-only ParameterTree")
        if tag in self.__tagdict:
            return
        self.tagnames = sorted(self.tagnames + [tag])
        self.__tagdict = {name: i for i, name in enumerate(self.tagnames)}
        is_leaf = lambda x: PTree.is_leaf(x) and not isinstance(x, ParameterTree)

        if self.tags is None or self.tags.is_empty():
            for p, _ in self.data.iter_leaves():
                self.tags[p] = np.zeros(len(self.tagnames), dtype=bool)
        else:
            insert_idx = self.__tagdict[tag]
            for p, _ in self.data.iter_leaves():
                self.tags[p] = np.insert(self.tags[p], insert_idx, False)

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

        if isinstance(tags, str):
            tags = [tags]

        self.create_tags_if_required(tags)
        tag_flags = self.get_tag_flags(tags)

        if self.data.is_leaf_at(path):
            if overwrite:
                self.tags[path] = tag_flags
            else:
                self.tags[path] = self.tags[path] | tag_flags

        else:
            # it's a branch, tag all leaves
            for p, _ in self.data[path].iter_leaves():
                pp = ParamPath(path) / p
                if overwrite:
                    self.tags[pp] = tag_flags
                else:
                    self.tags[pp] = self.tags[pp] | tag_flags

    def get_read_only_copy(self):
        from copy import deepcopy

        return ParameterTree(
            data=self.data.get_read_only_copy(),
            tags=self.tags.get_read_only_copy(),
            tagnames=deepcopy(self.tagnames),
            read_only=True,
        )

    def filter_by_tag(self, tags, mode="any"):
        # modes: 'any', 'all', 'exact'

        if isinstance(tags, str):
            tags = [tags]
        for t in tags:
            if t not in self.tagnames:
                # raise KeyError(f"Tag {t} not found in ParameterTree")
                logger.warning(f"Tag {t} not found in ParameterTree")
                return ParameterTree(), self

        tag_ids = [self.__tagdict[tag] for tag in tags]
        left_param_tree = ParameterTree(
            tagnames=self.tagnames,
            read_only=False,
        )
        right_param_tree = ParameterTree(
            tagnames=self.tagnames,
            read_only=False,
        )

        match_f = lambda x: np.any(x[tag_ids])
        if mode == "all":
            match_f = lambda x: np.all(x[tag_ids])
        elif mode == "exact":
            target_tag_flags = np.zeros(len(self.tagnames), dtype=bool)
            target_tag_flags[tag_ids] = True
            match_f = lambda x: np.all(x == target_tag_flags)

        def setval(tree, path, value, tags):
            if isArrayRef(value):
                newref = ArrayRef(tree.data, value.paths, value.indices)
                tree.data[path] = newref
            else:
                tree.data[path] = value
            tree.tags[path] = tags

        for path, data in self.data.iter_leaves():
            tag_flags = self.tags[path]
            if match_f(tag_flags):
                setval(left_param_tree, path, data, tag_flags)
            else:
                setval(right_param_tree, path, data, tag_flags)

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
    def merge(left: Self, right: Self, which: str = "left"):
        merged = ParameterTree()

        for left_tag_name in left.tagnames:
            merged.add_new_tag(left_tag_name)
        for right_tag_name in right.tagnames:
            merged.add_new_tag(right_tag_name)

        def setval(path, data, tags):
            if isinstance(data, ArrayRef):
                newref = ArrayRef(merged.data, data.paths, data.indices)
                merged.data[path] = newref
            else:
                merged.data[path] = data
            merged.tags[path] = tags

        for path, left_data in left.data.iter_leaves():
            if path in right.data:
                if not which:
                    raise ValueError(f"Path {path} found in both trees, specify which arg to merge")
                if which == "left":
                    setval(path, left_data, left.tags[path])
                elif which == "right":
                    setval(path, right.data[path], right.tags[path])
                else:
                    raise ValueError(
                        f"Unknown 'which' arg {which}. Allowed values: 'left', 'right'"
                    )
            else:
                setval(path, left_data, left.tags[path])

        for path, right_data in right.data.iter_leaves():
            if path not in left.data:
                setval(path, right_data, right.tags[path])

        merged.set_read_only(left.read_only and right.read_only)

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

    def tree_set_at(self, path, set_to):
        """return a copy of the tree, with the value at path set to set_to"""
        # naive version:
        tcopy = deepcopy(self)
        tcopy.set_read_only(False)
        tcopy[path] = set_to
        tcopy.tags[path] = self.tags[path]
        tcopy.set_read_only(self.read_only)
        tcopy2 = jtu.tree_map(lambda x: x, tcopy)
        return tcopy2


def flatten_ParameterTree(ptree):
    flat_contents = (ptree.data,)
    aux_data = (ptree.read_only, ptree.tagnames, ptree.tags)
    return (flat_contents, aux_data)


def unflatten_ParameterTree(aux_data, flat_contents):
    data = flat_contents[0]
    read_only, tagnames, tags = aux_data
    return ParameterTree(data=data, read_only=read_only, tags=tags, tagnames=tagnames)


jtu.register_pytree_node(ParameterTree, flatten_ParameterTree, unflatten_ParameterTree)

##────────────────────────────────────────────────────────────────────────────}}}


def init_if_needed(params, path, init_f, base_path=""):
    try:
        return params[f"{base_path}/{path}"]
    except KeyError:
        params[f"{base_path}/{path}"] = init_f()
        return params[f"{base_path}/{path}"]


def get_param(params, path, base_path="", **_):
    return params[f"{base_path}/{path}"]


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
            ref.push_back(f"{from_path}/{leaf}", from_id)
        params[leafpath] = ref


def save_ptree_to_hdf5_group(ptree: PTree, h5_group: h5py.Group):
    """Recursively saves a PTree to an HDF5 group."""
    for path, leaf_value in ptree.iter_leaves(get_leaf_value=True):
        current_group = h5_group
        path_parts = path.path
        for part in path_parts[:-1]:
            current_group = current_group.require_group(part)

        leaf_name = path_parts[-1]

        if isArrayRef(leaf_value):
            ref_group = current_group.require_group(leaf_name)
            ref_group.attrs["__type__"] = "ArrayRef"
            ref_group.attrs["paths"] = [str(p) for p in leaf_value.paths]
            ref_group.attrs["indices"] = np.array(
                leaf_value.indices, dtype=object if not leaf_value.indices else None
            )

        elif isinstance(leaf_value, (np.ndarray, jnp.ndarray)):
            current_group.create_dataset(leaf_name, data=np.asarray(leaf_value))

        elif leaf_value is None:
            dset = current_group.create_dataset(leaf_name, data=h5py.Empty("f"))
            dset.attrs["__type__"] = "None"

        else:  # Handle other scalar types
            dset = current_group.create_dataset(leaf_name, data=leaf_value)
            dset.attrs["__type__"] = type(leaf_value).__name__


def load_ptree_from_hdf5_group(h5_group: h5py.Group, target_ptree: PTree):
    """Recursively loads an HDF5 group into a PTree, reconstructing ArrayRefs."""
    refs_to_reconstruct = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            path = ParamPath(name)
            if obj.attrs.get("__type__") == "None":
                target_ptree[path] = None
            else:
                target_ptree[path] = obj[()]  # obj[()] reads the data
        elif isinstance(obj, h5py.Group):
            if obj.attrs.get("__type__") == "ArrayRef":
                # ArrayRef. Defer creation.
                path = ParamPath(name)
                paths = [ParamPath(p) for p in obj.attrs["paths"]]
                indices = tuple(map(tuple, obj.attrs["indices"]))
                refs_to_reconstruct.append((path, paths, indices))

    h5_group.visititems(visitor)

    # now we can construct the references
    for path, paths, indices in refs_to_reconstruct:
        target_ptree[path] = ArrayRef(target_ptree, paths, indices)


def save_parameter_tree(pt: ParameterTree, filename: str):
    """Saves a ParameterTree to an HDF5 file."""
    with h5py.File(filename, "w") as f:
        f.attrs["tagnames"] = pt.tagnames
        f.attrs["read_only"] = pt.read_only
        data_group = f.create_group("data")
        save_ptree_to_hdf5_group(pt.data, data_group)
        tags_group = f.create_group("tags")
        save_ptree_to_hdf5_group(pt.tags, tags_group)
    logger.info(f"Saved ParameterTree to {filename}")


def load_parameter_tree(filename: str) -> ParameterTree:
    """Loads a ParameterTree from an HDF5 file."""
    with h5py.File(filename, "r") as f:
        tagnames = list(f.attrs.get("tagnames", []))
        read_only = bool(f.attrs.get("read_only", False))

        data_tree = PTree()
        if "data" in f:
            load_ptree_from_hdf5_group(f["data"], data_tree)

        tags_tree = PTree()
        if "tags" in f:
            load_ptree_from_hdf5_group(f["tags"], tags_tree)

        pt = ParameterTree(data=data_tree, tags=tags_tree, tagnames=tagnames)
        pt.set_read_only(read_only)

    logger.info(f"Loaded ParameterTree from {filename}")
    return pt
