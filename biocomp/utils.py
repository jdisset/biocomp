import json
import copy

from jax.experimental import host_callback
from pathlib import Path
from tqdm import tqdm
import jax
from jax import jit, vmap, lax
from jax import tree_util as pytree
import jax.numpy as jnp
import pickle
import json5
import numpy as np
import logging

logger = logging.getLogger('biocomp')

def warn(*args, **kwargs):
    logger.warning(*args, **kwargs)

def info(*args, **kwargs):
    logger.info(*args, **kwargs)

def debug(*args, **kwargs):
    logger.debug(*args, **kwargs)


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                    --     random misc stuff     --
# ···············································································


def at_path(d: dict, path, val=None, defaultinit=lambda: None):
    for key in path[:-1]:
        try:
            d = d.setdefault(key, dict())
        except AttributeError as e:
            msg = (
                f'Cannot set "{key}" at path {path}: {e}. Did you pass something other than a dict?'
            )
            raise AttributeError(msg)
    if val is not None:
        d[path[-1]] = val
        d = d[path[-1]]
    else:
        d = d.setdefault(path[-1], defaultinit())
    return d

def delete_path(d, path):
    for key in path[:-1]:
        d = d[key]
    del d[path[-1]]

def apply_constraints(par, cons):
    newpar = par.copy()
    F = {'clip': jnp.clip}
    for ctype in cons.keys():
        assert ctype in F.keys(), f'constraint type {ctype} not implemented'
        f = F[ctype]
        for c in cons[ctype]:
            x = at_path(newpar, c[0])
            assert x is not None, f'path {c[0]} not found in parameters'
            at_path(newpar, c[0], f(x, *c[1]))
    return newpar


def uniqueIdGenerator(start=0):
    unique_id = int(start)

    def uniqueId():
        nonlocal unique_id
        unique_id += 1
        return unique_id - 1

    return uniqueId


def flatten(t):
    return [item for sublist in t for item in sublist]


def updated_dict(d1, d2):
    res = {}
    for key, val in d1.items():
        if type(val) == dict:
            if key in d2 and type(d2[key] == dict):
                res[key] = updated_dict(d1[key], d2[key])
            else:
                res[key] = copy.deepcopy(d1[key])
        else:
            if key in d2:
                res[key] = copy.deepcopy(d2[key])
            else:
                res[key] = copy.deepcopy(d1[key])
    for key, val in d2.items():
        if not key in d1:
            res[key] = copy.deepcopy(val)
    return res


def decode_json(df, cols):
    for col in cols:
        df[col] = df[col].apply(lambda x: json.loads(str(x)))
    return df


def isSubset(l1, l2):
    for e in l1:
        if e not in l2:
            return False
    return True


class DotDict(dict):
    def __getattr__(*args):
        val = dict.__getitem__(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# make sure that a list has at least i elements and then assign val to the ith element
def set_list_item(lst, i, val):
    if len(lst) <= i:
        lst.extend([None] * (i - len(lst) + 1))
    lst[i] = val


def load(path, suffix='.pickle'):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Not a file: {path}')
    if path.suffix != suffix:
        raise ValueError(f'Not a {suffix} file: {path}')
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def load_json5(path):
    with open(path) as f:
        return json5.load(f)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                        --     JAX helpers     --
# ···············································································


class TQDMProgress:
    def __init__(self, num_samples, message):
        self.bar = tqdm(range(num_samples))
        self.bar.set_description(message, refresh=False)

    def update(self, count):
        self.bar.update(count)

    def close(self):
        self.bar.close()
        pass


# --- tqdm progress bar for jax scan ---
# This code is from this blog post: https://www.jeremiecoullon.com/2021/01/29/jax_progress_bar/
def progress_scan(num_samples, progress_type=TQDMProgress, message=None, print_rate=None):
    "Progress bar for a JAX scan"
    if message is None:
        message = ""
        # message = f"Running for {num_samples:,} iterations"
    bars = {}

    if print_rate is None:
        print_rate = max(1, int(num_samples / 100))

    remainder = num_samples % print_rate

    def create(arg, transform):
        bars[0] = progress_type(num_samples, message)
        pass

    def update(arg, transform):
        bars[0].update(arg)

    def _update_progress_bar(iter_num):
        "Updates tqdm progress bar of a JAX scan or loop"
        _ = lax.cond(
            iter_num == 0,
            lambda _: host_callback.id_tap(create, None, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = lax.cond(
            # update every multiple of `print_rate` except at the end
            (iter_num % print_rate == 0) & (iter_num != num_samples - remainder),
            lambda _: host_callback.id_tap(update, print_rate, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = lax.cond(
            # update by `remainder`
            iter_num == num_samples - remainder,
            lambda _: host_callback.id_tap(update, remainder, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

    def close(arg, transform):
        bars[0].close()

    def _close_progress_bar(result, iter_num):
        return lax.cond(
            iter_num == num_samples - 1,
            lambda _: host_callback.id_tap(close, None, result=result),
            lambda _: result,
            operand=None,
        )

    def _progress_bar_scan(func):
        """Decorator that adds a progress bar to `body_fun` used in `lax.scan`.
        Note that `body_fun` must either be looping over `np.arange(num_samples)`,
        or be looping over a tuple who's first element is `np.arange(num_samples)`
        This means that `iter_num` is the current iteration number
        """

        @jit
        def wrapper_progress_bar(carry, x):
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x
            _update_progress_bar(iter_num)
            result = func(carry, x)
            return _close_progress_bar(result, iter_num)

        return wrapper_progress_bar

    return _progress_bar_scan


@jit
def tree_shape(t):
    return pytree.tree_map(lambda x: x.shape, t)


@jit
def tree_append(t, e):
    fa, tt = pytree.tree_flatten(t)
    fb, te = pytree.tree_flatten(e)
    assert te == tt
    return pytree.tree_unflatten(tt, [jnp.concatenate([a, jnp.array([b])]) for a, b in zip(fa, fb)])


def get_pytree(t, i):
    return pytree.tree_map(lambda x: x[i], t)


@pytree.Partial(jit, static_argnums=2)
def get_pytree2(t, i, ts):
    pp, _ = pytree.tree_flatten(t)
    l = [p[i] for p in pp]
    print(l)
    return pytree.tree_unflatten(l, ts)


def param_unstack(t, N):
    return [get_pytree(t, i) for i in range(N)]


def get_params(param_tree, i):
    return [jit(get_pytree, static_argnums=(1,))(t, i) for t in tqdm(param_tree)]


def param_unstack2(t, N):
    return [get_pytree2(t, i) for i in range(N)]


@jit
def tree_stack(trees):
    """Takes a list of trees and stacks every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = pytree.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


@jit
def tree_unstack(tree):
    """Takes a tree and turns it into a list of trees. Inverse of tree_stack.
    For example, given a tree ((a, b), c), where a, b, and c all have first
    dimension k, will make k trees
    [((a[0], b[0]), c[0]), ..., ((a[k], b[k]), c[k])]
    Useful for turning the output of a vmapped function into normal objects.
    """
    leaves, treedef = pytree.tree_flatten(tree)
    n_trees = leaves[0].shape[0]
    new_leaves = [[] for _ in range(n_trees)]
    for leaf in leaves:
        for i in range(n_trees):
            new_leaves[i].append(leaf[i])
    new_trees = [treedef.unflatten(l) for l in new_leaves]
    return new_trees


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                     --     parameters utils     --
#···············································································

def split_params(params, static_paths):
    """Split params into static and dynamic parts."""
    # any path that is not in static_paths is dynamic
    dynamic = params.copy()
    static = {}
    for path in static_paths:
        at_path(static, path, at_path(dynamic, path))
        delete_path(dynamic, path)
    return dynamic, static


def assemble_params(dynamic, static):
    """Assemble params from static and dynamic parts."""
    res = updated_dict(dynamic, static)
    return res

def flatten_params(params):
    """Flatten params into a single vector, 
    and also returns a descriptor that can be used
    to unflatten them."""
    leaves, treedef = jax.tree_util.tree_flatten(params)
    flat_leaves = [l.flatten() for l in leaves]
    shapes = [l.shape for l in leaves]
    flat_params = jnp.concatenate(flat_leaves)
    descriptor = (shapes, treedef)
    return flat_params, descriptor

def unflatten_params(flat_params, pdescriptor):
    """Unflatten params from a single vector and a descriptor."""
    shapes, treedef = pdescriptor
    # splits = jnp.cumsum(jnp.array([jnp.prod(jnp.array(s)) for s in shapes]), dtype=jnp.int32)
    splits = np.cumsum([np.prod(s) for s in shapes], dtype=np.int32)
    leaves = []
    start = 0
    for sp, sh in zip(splits, shapes):
        leaves.append(flat_params[start:sp].reshape(sh))
        start = sp
    params = jax.tree_util.tree_unflatten(treedef, leaves)
    return params



#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────



import time
from rich import status
from rich.live import Live

s = status.Status("Hello, [bold magenta]World[/bold magenta]!", spinner="dots")
s.start()
s.update("Loading...")
s.update("Loading [bold green]done[/bold green]!")
s.stop()


class Timer:
    def __init__(self, name, console=None):
        self.console = console
        self.name = name
        self.laps = []
        self.end_time = []

    def start(self, with_spinner=False):
        self.start_time = time.time()
        if with_spinner:
            assert(self.console is not None)
            stat = f"{self.name}..." if not isinstance(with_spinner, str) else with_spinner
            self.spinner = self.console.status(stat, spinner="dots")
            self.spinner.start()

    def lap(self):
        self.end_time.append(time.time() - self.start_time)

    def stop(self):
        self.lap()
        if hasattr(self, "spinner"):
            self.spinner.stop()

    def stop_print(self):
        self.stop()
        msg = f"{self.name} took {self.end_time[-1]:.2f} seconds"
        if self.console is not None:
            self.console.print(msg)
        else:
            print(msg)


class TimeStore:
    def __init__(self, console=None):
        self.console = console
        self.timers = {}

    def start(self, name, with_spinner=False):
        if name not in self.timers:
            self.timers[name] = Timer(name, self.console)
        self.timers[name].start(with_spinner=with_spinner)
        return self.timers[name]

    def lap(self, name):
        self.timers[name].lap()

    def stop_print(self, name):
        if isinstance(name, str):
            assert(name in self.timers)
            self.timers[name].stop_print()
        else:
            assert(isinstance(name, Timer))
            name.stop_print()

    def stop_all(self):
        for name in self.starts:
            self.lap(name)

    def print_all(self):
        for name in self.times:
            print(f"{name}: {self.times[name][-1]}")
