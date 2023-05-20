import json
import copy

import time
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
from rich.logging import RichHandler
from jax.tree_util import Partial as partial
from contextlib import contextmanager
import rich

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                       --     logging utils     --
# ···············································································

import logging
from rich.logging import RichHandler


def setup_logger(lname=None, level=logging.INFO):
    root_logger = logging.getLogger(lname)
    root_logger.setLevel(level)
    ch = RichHandler(rich_tracebacks=True)
    ch.setFormatter(logging.Formatter("%(message)s"))
    ch.setLevel(level)
    root_logger.handlers = [ch]
    root_logger.propagate = False
    return root_logger


setup_logger()
setup_logger('jax')
logger = setup_logger('biocomp')


def set_loglevel(level: str):
    global logger
    level = level.upper()
    logger.setLevel(level)
    logger.info(f"Log level set to {level}")
    return level


@contextmanager
def timer(name=None, use_logger=True):
    from time import perf_counter

    if use_logger:
        printf = logger.info
    else:
        printf = rich.print

    t = perf_counter()
    if name is not None:
        printf(f"\n{name}...")
    yield
    if name is not None:
        printf(f"\n{name} done in {perf_counter() - t:.2f} seconds")
    else:
        printf(f"\nElapsed time: {perf_counter() - t:.2f} seconds")


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                    --     random misc stuff     --
# ···············································································


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


def flatten_single(t):
    """Flattens a single level of a nested list"""
    return [item for sublist in t for item in sublist]


def flatten(x):
    if isinstance(x, list):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


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


def flat_concat(*arrays):
    return jnp.concatenate([jnp.asarray(a).ravel() for a in arrays])


def flatten_list(x):
    """Flatten nested lists of lists."""
    if isinstance(x, list):
        return [a for i in x for a in flatten_list(i)]
    else:
        return [x]


def str_to_int_array(s):
    return np.array([ord(c) for c in s], dtype=np.int32)


def int_array_to_str(a):
    return ''.join([chr(int(c)) for c in a])


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                        --     JAX helpers     --
# ···············································································


def get_looped_slice(a, start, end):
    """Get a slice of an array that loops around the end of the array."""
    offset = start // a.shape[0]
    start = start % a.shape[0]
    end = end - offset * a.shape[0]
    if end > a.shape[0]:  # loop around
        return jnp.concatenate([a[start:], get_looped_slice(a, 0, end - a.shape[0])])
    else:
        return a[start:end]


def value_and_jacfwd(f, x):
    pushfwd = partial(jax.jvp, f, (x,))
    basis = jnp.eye(x.size, dtype=x.dtype)
    y, jac = jax.vmap(pushfwd, out_axes=(None, 1))((basis,))
    return y, jac


def value_and_jacrev(f, x):
    y, pullback = jax.vjp(f, x)
    basis = jnp.eye(y.size, dtype=y.dtype)
    jac = jax.vmap(pullback)(basis)
    return y, jac


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

def freeze(struct):
    # converts dict to frozendict, list to tuple and recursively
    # freezes all nested dicts, lists, tuples, and sets.
    import frozendict
    if isinstance(struct, dict):
        return frozendict.frozendict({k: freeze(v) for k, v in struct.items()})
    elif isinstance(struct, list):
        return tuple([freeze(v) for v in struct])
    elif isinstance(struct, tuple):
        return tuple([freeze(v) for v in struct])
    elif isinstance(struct, set):
        return frozenset([freeze(v) for v in struct])
    else:
        return struct

def tree_shape(t):
    return pytree.tree_map(lambda x: x.shape, t)


@jit
def tree_append(t, e):
    fa, tt = pytree.tree_flatten(t)
    fb, te = pytree.tree_flatten(e)
    assert te == tt
    return pytree.tree_unflatten(tt, [jnp.concatenate([a, jnp.array([b])]) for a, b in zip(fa, fb)])


def tree_get(t, i):
    return pytree.tree_map(lambda x: x[i], t)


@jax.jit
def tree_unstack(t):
    """Unstack a tree of arrays into a list of trees of arrays"""
    N = jax.tree_util.tree_leaves(t)[0].shape[0]
    return [tree_get(t, i) for i in range(N)]


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                     --     parameters utils     --
# ···············································································
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

    def __repr__(self):
        return "/".join(self.path)


NODE_PATH = ParamPath('node')
SHARED_PATH = ParamPath('shared')
STATIC_PATH = ParamPath('__static__')
QVALS_PATH = SHARED_PATH / 'qvals'
KEYS_PATH = STATIC_PATH / '__keys__'
MASK_PATH = STATIC_PATH / 'qmasks'
NAMED_VALUES = STATIC_PATH / 'named_values'
QNAME_PATH = NAMED_VALUES / 'qnames'


def at_path_nested(d: dict, path, val=None, defaultinit=lambda: None):
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


def at_path(d: dict, path: ParamPath, val=None, defaultinit=lambda: None):
    return at_path_nested(d, path.path, val, defaultinit)


def at_path_flat(d: dict, path: str, val=None, defaultinit=lambda: None):
    if val is None:
        return d.setdefault(path, defaultinit())
    else:
        d[path] = val
        return val


def delete_path_nested(d, path):
    for key in path[:-1]:
        d = d[key]
    del d[path[-1]]


def delete_path_flat(d, path):
    del d[path]


def delete_path(d: dict, path: ParamPath):
    return delete_path_nested(d, path.path)


def split_params_nested(params, static_paths):
    """Split params into static and dynamic parts."""
    # any path that is not in static_paths is dynamic
    dynamic = params.copy()
    static = {}
    for path in static_paths:
        at_path(static, path, at_path(dynamic, path))
        delete_path(dynamic, path)
    return dynamic, static


def split_params_flat(params, static_paths):
    """Split params into static and dynamic parts."""
    static = {}
    dynamic = {}
    for k, v in params.items():
        # if k starts with any of the static_paths, it is static
        if any(k.startswith(p) for p in static_paths):
            static[k] = v
        else:
            dynamic[k] = v
    return dynamic, static


def split_params(params: dict, static_paths: list[ParamPath]):
    return split_params_nested(params, static_paths)


DEFAULT_MIN_RATE = 0.0
DEFAULT_MAX_RATE = 1.0


def continuous_initializer(rng, shape=(), minval=DEFAULT_MIN_RATE, maxval=DEFAULT_MAX_RATE):
    def init():
        return jax.random.uniform(
            key=rng, shape=shape, minval=minval, maxval=maxval, dtype=jnp.float32
        )

    return init


def glorot_initializer(rng, shape):
    def init():
        return jax.nn.initializers.glorot_normal()(rng, shape)

    return init


def he_initializer(rng, shape):
    def init():
        return jax.nn.initializers.he_uniform()(rng, shape)

    return init


def path_contains_flat(params, path):
    """returns the params with a path that contains the given path"""
    contains, doesnt_contain = {}, {}
    for k, v in params.items():
        if path in k:
            contains[k] = v
        else:
            doesnt_contain[k] = v
    return contains, doesnt_contain


def merge_dicts(*dicts):
    res = {}
    for d in dicts:
        res.update(d)
    return res


def assemble_params_flat(dynamic, static):
    """Assemble params from static and dynamic parts."""
    res = dynamic.copy()
    res.update(static)
    return res


def assemble_params_nested(dynamic, static):
    """Assemble params from static and dynamic parts."""
    res = updated_dict(dynamic, static)
    return res


# def assemble_params(dynamic, static):
# return assemble_params_nested(dynamic, static)


def assemble_params(*p):
    res = p[0]
    for d in p[1:]:
        res = assemble_params_nested(res, d)
    return res


def flatten_params(params):
    # TODO: switch to jax.flattten_util.ravel_pytree
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
    # TODO: switch to jax.flattten_util.ravel_pytree
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


@jax.jit
def get_params(param_tree, i):
    return [jit(tree_get, static_argnums=(1,))(t, i) for t in tqdm(param_tree)]


def params_to_numpy(params):
    # use tree_map to convert all the jax arrays to numpy arrays
    return jax.tree_map(lambda x: x if isinstance(x, float) else np.array(x), params)


def params_to_jax(params):
    return jax.tree_map(lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, params)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                        --     time utils     --
# ···············································································


class Timer:
    def __init__(self, name, console=None):
        self.console = console
        self.name = name
        self.laps = []
        self.end_time = []

    def start(self, with_spinner=False):
        self.start_time = time.time()
        if with_spinner:
            assert self.console is not None
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
            assert name in self.timers
            self.timers[name].stop_print()
        else:
            assert isinstance(name, Timer)
            name.stop_print()

    def stop_all(self):
        for name in self.starts:
            self.lap(name)

    def print_all(self):
        for name in self.times:
            print(f"{name}: {self.times[name][-1]}")


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


# at_path = at_path_flat
# delete_path = delete_path_flat
# split_params = split_params_flat
# assemble_params = assemble_params_flat
# path_contains = path_contains_flat

# checks
enable_checks = False


def set_enable_checks(value: bool):
    global enable_checks
    enable_checks = value


from jax.experimental import checkify


def check(*args, **kwargs):
    global enable_checks
    if enable_checks:
        checkify.check(*args, **kwargs)
    else:
        # replace by an assert of the same thing
        assert args[0](*args[1:], **kwargs)


from jax.experimental.checkify import Error


def checkwrap(func, errors=(checkify.user_checks | checkify.index_checks | checkify.float_checks)):
    global enable_checks
    if enable_checks:
        logger.info(f"checkwrap enabled for {func}")
        return jit(checkify.checkify(func, errors=errors))
    else:

        def wrapped_function(*args, **kwargs):
            result = func(*args, **kwargs)
            return Error({}, {}, {}, {}), result

        return wrapped_function
