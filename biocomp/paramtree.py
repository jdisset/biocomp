from jax.tree_util import register_pytree_node_class
from collections.abc import Mapping
from copy import deepcopy


@register_pytree_node_class
class ParamTree:
    def __init__(self, data=None):
        self.data = data or {}

    def __repr__(self):
        return f"ParamTree({self.data})"

    def __getitem__(self, path):
        if not isinstance(path, ParamPath):
            raise ValueError("Path must be an instance of ParamPath")
        data = self.data
        for p in path.path:
            data = data[p]
        return data

    def __setitem__(self, path, value):
        if not isinstance(path, ParamPath):
            raise ValueError("Path must be an instance of ParamPath")
        data = self.data
        for p in path.path[:-1]:
            data = data.setdefault(p, {})
        data[path.path[-1]] = value

    def at(self, path, val=None, create_path=False):
        if not isinstance(path, ParamPath):
            raise ValueError("Path must be an instance of ParamPath")
        try:
            return self[path]
        except KeyError:
            if create_path:
                self[path] = val
                return val
            return None

    def tree_flatten(self):
        children = (self.data,)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0])

    def delete(self, path):
        if not isinstance(path, ParamPath):
            raise ValueError("Path must be an instance of ParamPath")
        data = self.data
        for p in path.path[:-1]:
            data = data[p]
        del data[path.path[-1]]

    @staticmethod
    def split(tree, paths):
        if not isinstance(tree, ParamTree):
            raise ValueError("Tree must be an instance of ParamTree")
        if isinstance(paths, ParamPath):
            paths = [paths]

        tree1_data = deepcopy(tree.data)
        tree2_data = {}

        for path in paths:
            data1 = tree1_data
            data2 = tree2_data
            for p in path.path[:-1]:
                data1 = data1.setdefault(p, {})
                data2 = data2.setdefault(p, {})
            data2[path.path[-1]] = data1.pop(path.path[-1])

        tree1 = ParamTree(tree1_data)
        tree2 = ParamTree(tree2_data)
        return tree1, tree2

    @staticmethod
    def merge(tree1, tree2):
        if not isinstance(tree1, ParamTree) or not isinstance(tree2, ParamTree):
            raise ValueError("Both inputs must be instances of ParamTree")

        def merge_dicts(d1, d2):
            for key, value in d2.items():
                if key in d1:
                    if isinstance(d1[key], Mapping) and isinstance(value, Mapping):
                        d1[key] = merge_dicts(d1[key], value)
                    else:
                        raise ValueError(f"Conflicting keys '{key}' found in both trees.")
                else:
                    d1[key] = deepcopy(value)
            return d1

        merged_data = merge_dicts(deepcopy(tree1.data), deepcopy(tree2.data))
        merged_tree = ParamTree(merged_data)
        return merged_tree
