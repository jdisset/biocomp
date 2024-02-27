### {{{                          --     imports     --
import json
import copy
import xxhash
import time
from pathlib import Path
from tqdm import tqdm
import jax
from jax.experimental import host_callback
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
from pkg_resources import get_distribution, resource_filename
from .recipe import XP
import rich
import subprocess
import os

import cProfile

from typing import Union, Tuple, List, Dict, Any, Optional, Callable, Sequence, Iterable

##────────────────────────────────────────────────────────────────────────────}}}

PathLike = Union[str, Path]


class profiler:
    def __init__(self, filename):
        self.filename = filename
        # mkdir if it doesn't exist
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        self.profiler = cProfile.Profile()
        self.profiler.enable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.disable()
        self.profiler.dump_stats(self.filename)


def get_git_commit_hash():
    bcpath = Path(__file__).parent
    bcpath = bcpath.resolve()
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=bcpath).decode('ascii').strip()


def get_biocomp_version():
    return get_distribution('biocomp').version


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                       --     logging utils     --
# ···············································································

import logging
from rich.logging import RichHandler


# def setup_logger(lname=None, level=logging.INFO):
# root_logger = logging.getLogger(lname)
# root_logger.setLevel(level)
# ch = RichHandler(rich_tracebacks=True)
# ch.setFormatter(logging.Formatter("%(message)s"))
# ch.setLevel(level)
# root_logger.handlers = [ch]
# root_logger.propagate = False
# return root_logger


def setup_logger(lname=None, level=logging.INFO):
    log = logging.getLogger(lname)
    if log.hasHandlers():
        log.handlers.clear()
    logging_handler = RichHandler()
    logging_handler.setFormatter(logging.Formatter(datefmt="%Y-%m-%dT%H:%M:%S%z "))
    logging_handler._log_render.show_path = False
    log.addHandler(logging_handler)
    log.setLevel(level)
    return log


setup_logger()
setup_logger('jax')
logger = setup_logger('biocomp')
logger.propagate = False


def set_loglevel(level: str):
    global logger
    level = level.upper()
    logger.propagate = False
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

### {{{               --     loading constants and config     --

from omegaconf import OmegaConf
from pathlib import Path


def load_config(*config_files):  # in order of priority, the last one wins
    config = OmegaConf.create()
    for config_file in config_files:
        if not Path(config_file).exists():
            logger.warning(f'Config file {config_file} not found.')
            continue
        config = OmegaConf.merge(config, OmegaConf.load(config_file))
    OmegaConf.resolve(config)
    return config


BIOCOMP_BASE_CONFIG = load_config(resource_filename('biocomp', 'config/base.yaml'))

# TODO: switch the following to use the config file
# we check if there is a file named ~/.biocomp.json
# if so, we load it and use the paths defined there
# otherwise, we use the default paths defined above
GLOBAL_CONFIG_PATH = Path.home() / '.biocomp.json'
if GLOBAL_CONFIG_PATH.exists():
    with open(GLOBAL_CONFIG_PATH) as f:
        config = json.load(f)
        DEFAULT_XP_PATH = Path(config.get('xp_path', '')).expanduser()
        DEFAULT_RECIPE_PATH = config.get('recipe_path', '')
        if isinstance(DEFAULT_RECIPE_PATH, str):
            DEFAULT_RECIPE_PATH = Path(DEFAULT_RECIPE_PATH).expanduser()
        elif isinstance(DEFAULT_RECIPE_PATH, list):
            DEFAULT_RECIPE_PATH = [Path(p).expanduser() for p in DEFAULT_RECIPE_PATH]
        DEFAULT_LIB_PATH = Path(config.get('lib_path', '')).expanduser()

# we also check the environment variables to see if they define the paths
# if so, we use them in priority

if 'BIOCOMP_XP_PATH' in os.environ:
    DEFAULT_XP_PATH = Path(os.environ['BIOCOMP_XP_PATH']).expanduser()
if 'BIOCOMP_RECIPE_PATH' in os.environ:
    DEFAULT_RECIPE_PATH = Path(os.environ['BIOCOMP_RECIPE_PATH']).expanduser()
if 'BIOCOMP_LIB_PATH' in os.environ:
    DEFAULT_LIB_PATH = Path(os.environ['BIOCOMP_LIB_PATH']).expanduser()

##────────────────────────────────────────────────────────────────────────────}}}
# {{{                      --     data load/save     --
# ···············································································
import pickle


def save(data, path, overwrite=False, rename_if_exists=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        elif rename_if_exists:
            path = path.with_name(path.stem + '_' + path.suffix)
        else:
            raise RuntimeError(f'File {path} already exists.')
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def load(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Not a file: {path}')
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


#                                                                            }}}
### {{{               --     convenience loading functions     --
def load_lib(lib_path=DEFAULT_LIB_PATH):
    return load(lib_path)


# convenience loading functions with default paths
def load_xp(xpname, lib, xp_path=DEFAULT_XP_PATH, recipe_path=DEFAULT_RECIPE_PATH, **kwargs):
    xp = XP(xpname, xp_path, recipe_path, lib, **kwargs)
    return xp


##────────────────────────────────────────────────────────────────────────────}}}


### {{{                           --     cache     --
def get_cache(
    gen_f: Callable, signature: str, cache_location: Optional[PathLike], create_dir: bool = True
):
    """
    Get a cached value or generate it if it doesn't exist.
    Args:
    gen_f: function to generate the value
    signature: unique signature for the value
    cache_location: path to the cache directory
    create_dir: whether to create the cache directory if it doesn't exist
    Returns:
    the cached value or the generated value
    """
    if cache_location is not None:
        sighash = xxhash.xxh128(signature).hexdigest()
        if isinstance(cache_location, str):
            cache_location = Path(cache_location)
        try:
            if create_dir:
                cache_location.mkdir(parents=True, exist_ok=True)
            elif not cache_location.exists():
                raise FileNotFoundError(
                    f'Path {cache_location} doesn\'t exist and create_dir is False'
                )
            cachepath = cache_location / sighash
            cachepath = cachepath.resolve()
        except Exception as e:
            logger.error(f'Error creating cache directory: {e}')
            logger.error(f'Not using cache.')
            return gen_f()
        if cachepath.exists():
            logger.debug(f'Loading {sighash} from cache.')
            with open(cachepath, 'rb') as file:
                data = pickle.load(file)
        else:
            logger.debug(f'No such signature in cache: {signature}')
            logger.debug(f'Generating {sighash} and saving to cache.')
            data = gen_f()
            try:
                with open(cachepath, 'wb') as file:
                    pickle.dump(data, file)
            except Exception as e:
                logger.error(f'Error generating {sighash}: {e}')
    else:
        # no cache location = caching is disabled
        data = gen_f()
    return data


##────────────────────────────────────────────────────────────────────────────}}}

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                    --     random misc stuff     --
# ···············································································


def np_converter(obj):
    import jax.numpy as jnp

    if isinstance(obj, (np.integer, jnp.integer)):
        return int(obj)
    elif isinstance(obj, (np.floating, jnp.floating)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, jnp.ndarray)):
        return obj.tolist()
    elif isinstance(obj, np.bool_) or isinstance(obj, jnp.bool_):
        return bool(obj)
    elif np.isnan(obj) or jnp.isnan(obj):
        return None


# parse_float=lambda x: round(float(x), 3)
def make_json_compatible(o, converter=np_converter, float_precision=None):
    if float_precision is not None:
        return json.loads(
            json.dumps(o, default=converter), parse_float=lambda x: round(float(x), float_precision)
        )
    else:
        return json.loads(json.dumps(o, default=converter))


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
    if d1 is None:
        return copy.deepcopy(d2)
    if d2 is None:
        return copy.deepcopy(d1)
    if not isinstance(d1, dict):
        return copy.deepcopy(d2)
    if not isinstance(d2, dict):
        return copy.deepcopy(d2)
    res = {}
    for key, val in d1.items():
        if isinstance(val, dict):
            if key in d2 and isinstance(d2[key], dict):
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

from omegaconf import OmegaConf

def nested_resolve(d, known=None, throw_if_non_dict=True):
    known = copy.deepcopy(known) or {}
    if not isinstance(d, dict):
        if throw_if_non_dict:
            raise ValueError(f'Expected a dict, got {type(d)}')
        return d
    for k, v in d.items():
        if isinstance(v, dict):
            known[k] = updated_dict(known.get(k, {}), v)
        else:
            known[k] = copy.deepcopy(v)
    return {k: nested_resolve(known[k], known, False) for k in d.keys()}


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


def load_json5(path: PathLike):
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


def tree_to_jax(params):
    return jax.tree_map(lambda x: jnp.asarray(x), params)


def tree_to_np(params):
    return jax.tree_map(lambda x: np.asarray(x), params)


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

### {{{                   --     log-poly-log transform     --


def logb(x, base=10):
    """Compute log of x in base b."""
    return np.log(x) / np.log(base)


def cubic_exp_fwd(x, threshold, base, scale=1):
    """
    cubic polynomial that goes through (0,0) and has same first
    and second derivative as the log function at the threshold
    In other works, a spline that is log-like near the threshold
    scale is a parameter to squeeze or stretch the function
    """
    # assert base > 1 and scale > 0, 'Base must be > 1 and scale > 0'
    # assert (
    # 6 * logb(threshold, base) * scale > 5
    # ), 'Threshold too small for given scale (or vice versa)'

    logthresh = np.log(threshold)
    logbase = np.log(base)
    a = -0.5 * (3 - 2 * scale * logthresh) / (threshold**3 * logbase)
    b = -(-4 + 3 * scale * logthresh) / (threshold**2 * logbase)
    c = -0.5 * (5 - 6 * scale * logthresh) / (threshold * logbase)
    return a * x**3 + b * x**2 + c * x


def cubic_exp_inv(y, threshold, base, scale):
    """
    inverse of cubic_exp_fwd (on [0,threshold])
    """
    # used wolfram to solve the analytical inverse
    lT, lB, cb2 = np.log(threshold), np.log(base), np.cbrt(2)
    T, T2, T3 = threshold, threshold**2, threshold**3
    A = T3 * (
        56
        + y * lB * (486 - 648 * scale * lT + 216 * scale**2 * lT**2)
        - 522 * scale * lT
        + 648 * scale**2 * lT**2
        - 216 * scale**3 * lT**3
    )
    B = np.sqrt(4 * (-19 * T2 + 12 * scale * T2 * lT) ** 3 + A**2)
    C = np.cbrt(A + B)
    D = -9 + 6 * scale * lT
    E = 2 * T * (-4 + 3 * scale * lT) / D
    F = cb2 * (-19 * T2 + 12 * scale * T2 * lT)
    return E - (F / (D * C)) + (C / (cb2 * D))


def log_poly_log(x, threshold=100, base=10, compression=0.5):
    """
    bi-logarithm function with smooth transition to cubic polynomial between [-threshold, threshold]
    """
    x = np.asarray(x)
    sign = np.sign(x)
    x = np.abs(x)
    diff = logb(threshold, base) * (1.0 - compression)
    x = np.where(
        x > threshold,
        logb(x, base) - diff,
        cubic_exp_fwd(x, threshold, base=base, scale=compression),
    )
    return x * sign


def inverse_log_poly_log(y, threshold=100, base=10, compression=0.5):
    y = np.asarray(y)
    sign = np.sign(y)
    y = np.abs(y)
    diff = logb(threshold, base) * (1.0 - compression)
    transformed_threshold = cubic_exp_fwd(threshold, threshold, base=base, scale=compression)
    y = np.where(
        y > transformed_threshold,
        base ** (y + diff),
        cubic_exp_inv(y, threshold, base=base, scale=compression),
    )
    return y * sign


### {{{                        --     jax version     --


def jlogb(x, base=10):
    """Compute log of x in base b."""
    return jnp.log(x) / jnp.log(base)


def jcubic_exp_fwd(x, threshold, base, scale=1):
    """
    cubic polynomial that goes through (0,0) and has same first
    and second derivative as the log function at the threshold
    In other works, a spline that is log-like near the threshold
    scale is a parameter to squeeze or stretch the function
    """
    # assert base > 1 and scale > 0, 'Base must be > 1 and scale > 0'
    # assert (
    # 6 * logb(threshold, base) * scale > 5
    # ), 'Threshold too small for given scale (or vice versa)'

    logthresh = jnp.log(threshold)
    logbase = jnp.log(base)
    a = -0.5 * (3 - 2 * scale * logthresh) / (threshold**3 * logbase)
    b = -(-4 + 3 * scale * logthresh) / (threshold**2 * logbase)
    c = -0.5 * (5 - 6 * scale * logthresh) / (threshold * logbase)
    return a * x**3 + b * x**2 + c * x


def jcubic_exp_inv(y, threshold, base, scale):
    """
    inverse of cubic_exp_fwd (on [0,threshold])
    """
    # used wolfram to solve the analytical inverse
    lT, lB, cb2 = jnp.log(threshold), jnp.log(base), jnp.cbrt(2)
    T, T2, T3 = threshold, threshold**2, threshold**3
    A = T3 * (
        56
        + y * lB * (486 - 648 * scale * lT + 216 * scale**2 * lT**2)
        - 522 * scale * lT
        + 648 * scale**2 * lT**2
        - 216 * scale**3 * lT**3
    )
    B = jnp.sqrt(4 * (-19 * T2 + 12 * scale * T2 * lT) ** 3 + A**2)
    C = jnp.cbrt(A + B)
    D = -9 + 6 * scale * lT
    E = 2 * T * (-4 + 3 * scale * lT) / D
    F = cb2 * (-19 * T2 + 12 * scale * T2 * lT)
    return E - (F / (D * C)) + (C / (cb2 * D))


@jit
def jax_log_poly_log(x, threshold=100, base=10, compression=0.5):
    """
    bi-logarithm function with smooth transition to cubic polynomial between [-threshold, threshold]
    """
    x = jnp.asarray(x)
    sign = jnp.sign(x)
    x = jnp.abs(x)
    diff = jlogb(threshold, base) * (1.0 - compression)
    x = jnp.where(
        x > threshold,
        jlogb(x, base) - diff,
        jcubic_exp_fwd(x, threshold, base=base, scale=compression),
    )
    return x * sign


@jit
def jax_inverse_log_poly_log(y, threshold=100, base=10, compression=0.5):
    y = jnp.asarray(y)
    sign = jnp.sign(y)
    y = jnp.abs(y)
    diff = jlogb(threshold, base) * (1.0 - compression)
    transformed_threshold = jcubic_exp_fwd(threshold, threshold, base=base, scale=compression)
    y = jnp.where(
        y > transformed_threshold,
        base ** (y + diff),
        jcubic_exp_inv(y, threshold, base=base, scale=compression),
    )
    return y * sign


##────────────────────────────────────────────────────────────────────────────}}}


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                         --     jaxutils     --
from jax.experimental import host_callback

enable_checks = False


def grid_map(F, xrange, yrange, meshres):
    import jax

    XX, YY = np.meshgrid(
        np.linspace(xrange[0], xrange[1], meshres[0]),
        np.linspace(yrange[0], yrange[1], meshres[1]),
        indexing='xy',
    )
    coords = np.column_stack((XX.ravel(), YY.ravel()))
    ZZ = jax.vmap(F)(coords).reshape(XX.shape)
    return XX, YY, ZZ


def hooked_scan(num_samples, on_update, call_rate=1):
    import jax
    from jax import lax
    from jax.experimental import host_callback

    def update(args, transform):
        result, iternum = args
        carry, acc = result
        on_update(acc, iternum)

    def _update_(result, iter_num):
        return lax.cond(
            (iter_num % call_rate == 0) | (iter_num == num_samples - 1),
            lambda _: host_callback.id_tap(update, (result, iter_num), result=result),
            lambda _: result,
            operand=None,
        )

    def _hooked_scan(func):
        @jax.jit
        def wrapper(carry, x):
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x
            result = func(carry, x)
            return _update_(result, iter_num)

        return wrapper

    return _hooked_scan


def get_jaxpr(fun, *args, **kwargs):
    import jax

    return jax.make_jaxpr(fun)(*args, **kwargs)


def print_jaxpr(fun, *args, **kwargs):
    get_jaxpr(fun, *args, **kwargs).pretty_print()


def get_xla(fun, *args, static_argnums=(), **kwargs):
    import jax
    import jaxlib.xla_extension as xla_ext

    console = Console(highlighter=rich.highlighter.ReprHighlighter())
    c = jax.xla_computation(fun, static_argnums=static_argnums)(*args, **kwargs)
    backend = jax.lib.xla_bridge.get_backend()
    e = backend.compile(c)
    option = xla_ext.HloPrintOptions.short_parsable()
    out = e.hlo_modules()[0].to_string(option)
    return out


def print_xla(fun, *args, static_argnums=(), **kwargs):
    print(get_xla(fun, *args, **kwargs))


def get_looped_slice(a, start, end, axis=0):
    """Get a slice of an array that loops around the end of the array if end > a.shape[axis]"""
    offset = start // a.shape[axis]
    start = start % a.shape[axis]
    end = end - offset * a.shape[axis]
    if end > a.shape[axis]:  # loop around
        idx = [slice(None)] * a.ndim
        idx[axis] = slice(start, None)
        s1 = a[tuple(idx)]
        idx[axis] = slice(0, end - a.shape[axis])
        s2 = get_looped_slice(a, 0, end - a.shape[axis], axis)
        return np.concatenate([s1, s2], axis=axis)
    else:
        idx = [slice(None)] * a.ndim
        idx[axis] = slice(start, end)
        return a[tuple(idx)]


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


##────────────────────────────────────────────────────────────────────────────}}}

# at_path = at_path_flat
# delete_path = delete_path_flat
# split_params = split_params_flat
# assemble_params = assemble_params_flat
# path_contains = path_contains_flat

### {{{                      --     topology analysis helpers     --


def get_uorf_value(param):
    if 'tl_rate' in param:
        u = param['tl_rate'][0].split('_')[0]
        try:
            v = int(u[:-1]) * 10
        except ValueError:
            v = 0
        if u[-1] == 'w':
            v = v - 5
        return v
    else:
        return 0


UORF_DICT = {
    0: 'No uORF',
    5: 'weak uORF',
    10: '1x uORF',
    20: '2x uORF',
    30: '3x uORF',
    40: '4x uORF',
    50: '5x uORF',
    60: '6x uORF',
    70: '7x uORF',
    80: '8x uORF',
}


def get_all_ERN_ids(network):
    ERN_ids = network.compute_graph[network.compute_graph['type'] == 'sequestron_ERN'].index.values
    return network.sort_nodes_by_upstream(ERN_ids)


def get_all_ERNs_names(network):
    ERNs = network.compute_graph.loc[get_all_ERN_ids(network)]
    ERN_extras = ERNs['extra'].values
    ERN_names = [e['seq_name'].split('#')[0].split('::')[-1] for e in ERN_extras]
    return ERN_names


def get_uorf_names(uorf_values, ern_names):
    uorf_names = []
    for uorf, ern_name in zip(uorf_values, ern_names):
        ERN_uorf, REC_uorf = uorf
        ERN_uorf = UORF_DICT[ERN_uorf]
        REC_uorf = UORF_DICT[REC_uorf]
        uorf_names.append((f'{ern_name} ERN: {ERN_uorf}', f'{ern_name} REC: {REC_uorf}'))
    return uorf_names


def get_all_uorf_values(network):
    cdg = network.central_dogma_graph
    ERNs = network.compute_graph.loc[get_all_ERN_ids(network)]
    ERN_names = get_all_ERNs_names(network)
    ERN_inputs = ERNs['cdg_input'].values
    values = []
    for inp in ERN_inputs:
        cdgin = cdg.loc[inp]
        ern_side = cdg.loc[cdgin.iloc[0].predecessor[0]]
        recog_side = cdgin.iloc[1]
        uvals = (get_uorf_value(ern_side.params), get_uorf_value(recog_side.params))
        values.append(uvals)
    names = get_uorf_names(values, ERN_names)
    return tuple(values), tuple(names)


from typing import List, Callable


def get_ERN_ids(network):
    return network.compute_graph[network.compute_graph['type'] == 'sequestron_ERN'].index.values


def get_RCB_ids(network):
    return network.compute_graph[
        network.compute_graph['type'].str.startswith('sequestron_R')
    ].index.values


def get_sequestron_ids(network):
    return network.compute_graph[
        network.compute_graph['type'].str.startswith('sequestron_')
    ].index.values


def make_is_upstream(network):
    def is_upstream(i, j):
        return network.compute_node_is_upstream_of(i, j)

    return is_upstream


def topological_sort(
    node_list: List[int], is_upstream: Callable[[int, int], bool]
) -> List[List[int]]:
    visited = set()
    batches = []
    while len(visited) < len(node_list):
        independent = [
            i
            for i in node_list
            if i not in visited
            and all([j in visited for j in node_list if j != i and is_upstream(j, i)])
        ]

        if not independent:
            raise ValueError('Cycle detected in graph')
        visited.update(independent)
        batches.append(independent)
    return batches


def get_network_family(network):
    erns = get_ERN_ids(network)
    rcbs = get_RCB_ids(network)
    seqs = get_sequestron_ids(network)
    ts = topological_sort(seqs, make_is_upstream(network))

    seqtype = 'none'
    family = 'unknown'
    match (len(erns) > 0, len(rcbs) > 0):
        case (True, True):
            seqtype = 'hybrid'
        case (True, False):
            seqtype = 'ERN'
        case (False, True):
            seqtype = 'RCB'

    match (len(seqs), len(ts)):
        case (0, 0):
            family = 'no device'
        case (1, 1):
            family = 'single'
        case (2, 2):
            family = 'cascade'
        case (2, 1):
            family = 'dual region'
        case (3, 2):
            family = 'bandpass'

    return family, seqtype


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                  --     function serialization     --
import inspect
import sys


def unwrap_partial_function(implementation):
    if hasattr(implementation, 'func') and hasattr(implementation, 'keywords'):
        partial_args = implementation.keywords
        implementation = implementation.func
    else:
        partial_args = {}
    return implementation, partial_args


def serialize_function(implementation: Callable, **kwargs):
    implementation, partial_args = unwrap_partial_function(implementation)
    kwargs.update(partial_args)

    # detect if it's in a module
    module_name = implementation.__module__
    if module_name == '__main__':
        module_name = None

    signature = inspect.signature(implementation)
    parameters = {}
    for name, param in signature.parameters.items():
        if name in kwargs:
            parameters[name] = kwargs[name]
        elif param.default != inspect.Parameter.empty:
            parameters[name] = param.default

    res = {
        'implementation': implementation.__name__,
        'parameters': parameters,
    }
    if module_name is not None:
        res['module_name'] = module_name
    return res


def deserialize_function(func_data: dict, module_names: Optional[list[str]] = None):

    if module_names is None:
        if 'module_name' in func_data:
            module_names = [func_data['module_name']]
        else:  # use the global namespace
            module_names = ['__main__']
            print('No module name provided, using __main__')

    for module_name in module_names:
        if module_name in sys.modules:
            module = sys.modules[module_name]
            if hasattr(module, func_data['implementation']):
                implementation = getattr(module, func_data['implementation'])
                return partial(implementation, **func_data['parameters'])

    raise ValueError(f'No function named {func_data["implementation"]}')


##────────────────────────────────────────────────────────────────────────────}}}
