### {{{                          --     imports     --
from biocomp.logging_config import setup_logging, get_logger
import yaml
import json
import logging
from rich.logging import RichHandler
from rich import print as rprint
import inspect
import sys
import copy
from copy import deepcopy
import xxhash
import time
from pathlib import Path
from tqdm import tqdm
import jax
from omegaconf import DictConfig, ListConfig
from jax.experimental import host_callback
from jax import jit, vmap, lax
from jax import tree_util as pytree
import jax.numpy as jnp
import pickle
import json5
import numpy as np
from jax.tree_util import Partial as partial
from contextlib import contextmanager
from pkg_resources import get_distribution, resource_filename
import rich
import subprocess
import os

import cProfile

from biocomp.models import buildLibFromDatabase

from pydantic import BaseModel, BeforeValidator, ConfigDict

##────────────────────────────────────────────────────────────────────────────}}}
## {{{                           --     types     --
from typing import (
    Union,
    List,
    Dict,
    Any,
    Optional,
    Callable,
    Sequence,
    TypeVar,
    Generic,
    Annotated,
    Type,
)

T = TypeVar("T")
R = TypeVar("R")
PathLike = Union[str, Path]
DictLike = Union[Dict, DictConfig]


class ArbitraryModel(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
    )

    # class Config:
    # arbitrary_types_allowed = True
    # extra = "forbid"
    # validate_default = True

    def model_dump(self, **kwargs) -> dict[str, Any]:
        return super().model_dump(serialize_as_any=True, **kwargs)

    def model_dump_json(self, **kwargs) -> str:
        return super().model_dump_json(serialize_as_any=True, **kwargs)

    def model_dump_yaml(self, **kwargs) -> str:
        dict_repr = self.model_dump()
        return yaml_dump(dict_repr, **kwargs)

    def __str__(self):
        return self.__repr__()


##────────────────────────────────────────────────────────────────────────────}}}
# {{{                       --     logging utils     --
# ···············································································


logger = get_logger(__name__)


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

# ╭─────────────────────────────────────────────╮
# │               loading, saving               │
# ╰───────────────────── ⟱ ─────────────────────╯

# {{{                      --     data load/save     --
# ···············································································
import pickle


def save(data: Any, path: Union[str, Path], overwrite: bool = False, rename_if_exists: bool = True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        elif rename_if_exists:
            path = path.with_name(path.stem + "_" + path.suffix)
        else:
            raise RuntimeError(f"File {path} already exists.")
    with open(path, "wb") as file:
        pickle.dump(data, file)


def load(path: Union[str, Path]) -> Any:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data


#                                                                            }}}
## {{{                           --     cache     --


def get_cache(
    gen_f: Callable[[], T],
    signature: str,
    cache_location: Optional[PathLike],
    create_dir: bool = True,
) -> T:
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
                    f"Path {cache_location} doesn't exist and create_dir is False"
                )
            cachepath = cache_location / sighash
            cachepath = cachepath.resolve()
        except Exception as e:
            logger.error(f"Error creating cache directory: {e}")
            logger.error(f"Not using cache.")
            return gen_f()
        if cachepath.exists():
            logger.debug(f"Loading {sighash} from cache.")
            with open(cachepath, "rb") as file:
                data = pickle.load(file)
        else:
            logger.debug(f"No such signature in cache: {signature}")
            logger.debug(f"Generating {sighash} and saving to cache.")
            data = gen_f()
            try:
                with open(cachepath, "wb") as file:
                    pickle.dump(data, file)
            except Exception as e:
                logger.error(f"Error generating {sighash}: {e}")
    else:
        # no cache location = caching is disabled
        data = gen_f()
    return data


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                  --     function serialization     --


class PartialFunction(ArbitraryModel, Generic[T, R]):
    """
    A partial function that can be serialized and deserialized
    """

    func: Union[str, Callable]  # 'module.fname' or 'fname' or function
    args: list = []
    kwargs: dict = {}
    modules: list = []

    def model_post_init(self, *a, **kw):
        super().model_post_init(*a, **kw)
        self._func = None

        print("Post init function: ", self.model_dump())

    def set_missing_kwargs(self, new_kwargs: dict):
        """only overwrite the kwargs that are not already set, and are in the signature"""
        import inspect

        if isinstance(self.func, str):
            func = decode_type(self.func, self.modules)
        else:
            func = self.func

        assert callable(func), "The resolved function is not callable"

        sig = inspect.signature(func)
        has_param = False
        for k in new_kwargs:
            if k in self.kwargs:
                continue
            if k in sig.parameters:
                has_param = True
            # or, detect if the method has a ** like parameter
            if not has_param:
                for p in sig.parameters.values():
                    if p.kind == p.VAR_KEYWORD:
                        has_param = True
                        break
            if has_param:
                self.kwargs[k] = new_kwargs[k]

    def get_impl(
        self,
        extra_module_names: Optional[List[str]] = None,
        force_refresh=False,
    ) -> Callable:
        if self._func is not None and not force_refresh:
            return self._func

        if extra_module_names is None:
            extra_module_names = []

        implem = self.func
        if isinstance(implem, str):
            implem = decode_type(implem, self.modules + extra_module_names)

        assert callable(implem), f"{implem} is not a function"
        args, kwargs = self.parsed_args()
        self._func = partial(implem, *args, **kwargs)

        return self._func

    def parsed_args(self):
        args = [arg() if isinstance(arg, PartialFunctionResult) else arg for arg in self.args]
        kwargs = {
            k: v() if isinstance(v, PartialFunctionResult) else v for k, v in self.kwargs.items()
        }
        return args, kwargs

    def __call__(self, *args, **kw):
        return self.get_impl()(*args, **kw)

    def __repr__(self):
        return f"PartialFunction({self.func=}, {self.args=}, {self.kwargs=})"

    def get_name(self) -> str:
        if isinstance(self.func, str):
            spl = self.func.rsplit(".", 1)
            if len(spl) == 2:
                return spl[1]
            return self.func
        return self.func.__name__


T = TypeVar("T")
R = TypeVar("R")


class PartialFunctionResult(PartialFunction[T, R]):
    """
    Meant to signal to PartialFunction that the function should be called
    when it's used as argument (and the argument therefore is the result of the function)
    """

    pass


class ExecuteFunction(PartialFunction[T, R]):
    """
    Execute on instantiation (and return result)
    """

    def __new__(cls, **data):
        instance = super().__new__(cls)
        instance.__init__(**data)
        return instance()


def unwrap_partial_function(implementation):
    if hasattr(implementation, "func") and hasattr(implementation, "keywords"):
        partial_args = implementation.keywords
        implementation = implementation.func
    else:
        partial_args = {}
    return implementation, partial_args


def get_fname(func: Callable) -> str:
    module_name = func.__module__
    if module_name == "__main__":
        module_name = None
    fname = func.__name__
    if module_name is not None:
        fname = f"{module_name}.{fname}"
    return fname


def encode_function(func: Callable, **kwargs) -> PartialFunction:
    if isinstance(func, (PartialFunction, PartialFunctionResult)):
        new_pf = func
        new_pf.kwargs.update(kwargs)
        if not isinstance(new_pf.func, str):
            new_pf.func = get_fname(new_pf.func)
        return new_pf

    else:
        func, partial_args = unwrap_partial_function(func)

    kwargs.update(partial_args)

    signature = inspect.signature(func)
    f_kwargs = {}
    for name, param in signature.parameters.items():
        if name in kwargs:
            f_kwargs[name] = kwargs[name]
        elif param.default != inspect.Parameter.empty:
            f_kwargs[name] = param.default

    fname = get_fname(func)

    return PartialFunction(func=fname, kwargs=f_kwargs)


EncodedPartialFunction = Annotated[PartialFunction, BeforeValidator(encode_function)]


def decode_type(
    type_str: str,
    available_module_names: Optional[List[str]] = None,
) -> Type:
    """
    If it exists, returns the first type found
    with the given name in the given modules
    format: 'module.type' or 'type'
    """

    if available_module_names is None:
        available_module_names = ["__main__"]

    spl = type_str.rsplit(".", 1)
    if len(spl) == 2:
        module_name, type_name = spl
        available_module_names = [module_name] + available_module_names
    else:
        type_name = type_str

    for module_name in available_module_names:
        if module_name in sys.modules:
            module = sys.modules[module_name]
        else:
            # load module
            module = __import__(module_name, fromlist=[type_name])
        if hasattr(module, type_name):
            return getattr(module, type_name)

    raise ValueError(f"No type named {type_str} in modules {available_module_names}")


def build_if_has_target(
    value: Union[DictLike, T],
    available_module_names: Optional[List[str]] = None,
    enforce_type: Optional[Type[T]] = None,
) -> Union[T, DictLike]:
    """
    If the dictionary has a key '_target_', will instantiate
    an object of this type with the remaining keys as arguments.
    """

    if not dict_like(value):
        if enforce_type is not None and not isinstance(value, enforce_type):
            raise ValueError(f"Value {value} is not an instance of {enforce_type}")
        return value

    assert isinstance(value, DictLike)

    if not "_target_" in value:  # type: ignore
        return value

    target_type = decode_type(value["_target_"], available_module_names)
    if enforce_type is not None and not issubclass(target_type, enforce_type):
        raise ValueError(f"Target type {target_type} not a subclass of {enforce_type}")
    ctor_dict = {str(k): value[k] for k in value if k != "_target_"}
    return target_type(**ctor_dict)


##────────────────────────────────────────────────────────────────────────────}}}

# ╭─────────────────────────────────────────────╮
# │             configuration utils             │
# ╰───────────────────── ⟱ ─────────────────────╯

## {{{                       --     config utils     --


"""
A nested configuration is a dictionary that contains the configuration
for each function in a call stack (i.e. the kwargs for each function).
To know that a function calls another one and needs a nested config,
we need 2 things:
     - the calling function needs to have a parameter named
        {called_function_name}{function_config_suffix}, e.g. 'makeplot_params'

     - the called function needs to have been declared as configurable
        (can use the @configurable decorator for that)

Because some modules (such as plotting) rely heavily on nested configurations,
and because different paths of nested calls might require different configurations,
(for example heatmap might need a different configuration when called from a
2d pipeline vs for a 3d slice in a 3d pipeline), we need a way to declare
configurations that can be progressively specialized depending on the call stack.

The idea is that we can then make configuration files that inherit from the closest base
configuration (which can also inherit from an upstream declaration, etc...),
and only override the parameters we want to change.

For example, if we have a function makeplot that calls a function plotdata,
we can have a configuration file like this:

```
    f1_params:
     ... (base parameters for f1)

    f2_params:
     ... (base parameters for f2)

    f3_params:
     ... (base parameters for f3)

    f2_params:
        f3_params:
            ... (arguments for f3 when called from f2)

    f1_params:
        f3_params:
            ... (arguments for f3 when called from f1)
        f2_params:
            ... (arguments for f2 when called from f1)
            f3_params:
                ... (arguments for "f3 in f2 in f1")
                    final f3_params will be the result of merging,
                    from least to most specific:
                     - base f3_params
                     - f3 in f1
                     - f3 in f2
                     - f3 in f2 in f1
```

since, for that to work, we need to know the full path of the function in the call stack,
which would be annoying to write by hand in the config file, we can use the
generate_base_nested_config function to generate a template for the config file
and merge it with the (potentially sparse) user's config file to get the full nested config.

"""

import inspect


_CONFIGURABLE_FUNCTIONS = {}


def get_configurable_functions(namespace: str = "default") -> Dict:
    if namespace not in _CONFIGURABLE_FUNCTIONS:
        _CONFIGURABLE_FUNCTIONS[namespace] = {}
    return _CONFIGURABLE_FUNCTIONS[namespace]


def configurable(func: Callable, namespace: str = "default") -> Callable:
    """Decorator to add a function and its arguments to the list of configurable functions."""
    local_conf_functions = get_configurable_functions(namespace)
    sig = inspect.signature(func)
    # fkwargs = list(sig.parameters.keys())

    local_conf_functions[func.__name__] = {
        k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty
    }

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def configurable_decorator(namespace: str = "default") -> Callable:
    return partial(configurable, namespace=namespace)


def generate_base_nested_config(
    available_functions: Optional[Dict] = None,
    function_config_suffix: str = "_params",
    add_defaults: bool = False,
    namespace: str = "default",
):
    if available_functions is None:
        available_functions = get_configurable_functions(namespace)

    # essentially a template for the full config, that shows
    # its structure (in terms of nested functions).
    # for each available function, we can check if it needs a nested config
    # if it does, we can generate an empty config for it

    emptyconf = {}

    def generate_empty_func_conf(func_name, func_args):
        subconf = {}
        for arg in func_args.keys():
            if isinstance(arg, str) and arg.endswith(function_config_suffix):
                fname = arg[: -len(function_config_suffix)]
                if fname in available_functions:
                    subconf[arg] = generate_empty_func_conf(fname, available_functions[fname])
                else:
                    logger.debug(
                        f"{func_name} has a nested config {arg} but {fname} is not a known function"
                    )
        return subconf

    for func_name, func_args in available_functions.items():
        argname = f"{func_name}{function_config_suffix}"
        emptyconf[argname] = generate_empty_func_conf(func_name, func_args)

    if add_defaults:
        for func_name, func_args in available_functions.items():
            fdict = emptyconf[func_name + function_config_suffix]
            for arg, default_val in func_args.items():
                if not arg.endswith(function_config_suffix):
                    fdict[arg] = default_val

    return emptyconf


def resolve_if_ends_with(key: Any, suffix: str) -> bool:
    return isinstance(key, str) and key.endswith(suffix)


def dict_like(obj) -> bool:
    return (
        hasattr(obj, "keys")
        and hasattr(obj, "get")
        and hasattr(obj, "__getitem__")
        and hasattr(obj, "__contains__")
        and hasattr(obj, "__iter__")
        and hasattr(obj, "items")
    )


# define valid enum of merge modes: extend, replace, auto
from typing import Type, Union


def replace(d1, d2):
    return deepcopy(d2)


def extend(d1, d2):
    if d1 is None:
        return deepcopy(d2)
    if d2 is None:
        return deepcopy(d1)
    return deepcopy(d2) + deepcopy(d1)


DEFAULT_MERGE_MODES = {"replace": replace, "extend": extend, "auto": "auto"}


def maybecopy(obj, deep: bool = True):
    return deepcopy(obj) if deep else obj


def updated_dict(
    d1,
    d2,
    merge_mode: Optional[Dict[Union[Type, str], Union[str, Callable]]] = None,
    deep: bool = True,
) -> Dict:
    if merge_mode is None:
        merge_mode = {}

    t1, t2 = type(d1), type(d2)
    st1, st2 = str(t1.__name__), str(t2.__name__)

    mmode = "auto"

    if t1 in merge_mode or st1 in merge_mode:
        mmode = merge_mode.get(t1, merge_mode.get(st1))
        if mmode in DEFAULT_MERGE_MODES:
            mmode = DEFAULT_MERGE_MODES[mmode]
        if callable(mmode):
            return mmode(d1, d2)

    if t2 in merge_mode or st2 in merge_mode:
        mmode = merge_mode.get(t2, merge_mode.get(st2))
        if mmode in DEFAULT_MERGE_MODES:
            mmode = DEFAULT_MERGE_MODES[mmode]
        if callable(mmode):
            return mmode(d1, d2)

    if mmode == "auto":
        if not dict_like(d1):
            return maybecopy(d2, deep) if d2 is not None else maybecopy(d1, deep)
        if not dict_like(d2):
            return maybecopy(d1, deep) if d1 is not None else maybecopy(d2, deep)
    else:
        raise NotImplementedError(f"Cannot merge {t1} and {t2}")

    assert mmode == "auto", f"Invalid merge mode {mmode}"
    # they're both dicts:
    res = {}
    for key, val in d1.items():
        if key in d2:
            res[key] = updated_dict(d1[key], d2[key], merge_mode)
        else:
            res[key] = maybecopy(d1[key], deep)
    for key, val in d2.items():
        if not key in d1:
            res[key] = maybecopy(val, deep)
    return res


def nested_resolve(
    input_dict: Any,
    already_seen: Dict = {},
    resolve_key: Callable[[str], bool] = partial(resolve_if_ends_with, suffix="_params"),
) -> Dict:
    if not isinstance(input_dict, dict):
        return deepcopy(input_dict)

    new_seen = deepcopy(already_seen)
    for k, v in input_dict.items():
        new_seen[k] = updated_dict(already_seen.get(k, None), v) if resolve_key(k) else deepcopy(v)

    new_dict = {
        k: nested_resolve(deepcopy(new_seen[k]), deepcopy(new_seen), resolve_key)
        for k in input_dict.keys()
    }

    return new_dict


def generate_full_nested_config(
    user_config: Optional[Dict] = None,
    empty_config: Optional[Dict] = None,
    namespace: str = "default",
    **kw,
):
    if empty_config is None:
        empty_config = generate_base_nested_config(namespace=namespace, **kw)
    if user_config is None:
        return empty_config
    merged = nested_resolve(updated_dict(user_config, empty_config))
    return merged


class BiocompYamlDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(BiocompYamlDumper, self).increase_indent(flow, False)

    def ignore_aliases(self, data):
        return True

    def represent_sequence(self, tag, sequence, flow_style=None):
        return super(BiocompYamlDumper, self).represent_sequence(tag, sequence, flow_style=True)


def delete_empty(d: Any):
    if isinstance(d, dict):
        newd = {}
        for k, v in d.items():
            if isinstance(v, dict):
                if len(v) > 0:
                    newv = delete_empty(v)
                    if len(newv) > 0:
                        newd[k] = newv
            else:
                newd[k] = copy.deepcopy(v)
    else:
        newd = copy.deepcopy(d)
    return newd


def yaml_dump(data, **kw):
    return yaml.dump(data, Dumper=BiocompYamlDumper, **kw)


def dump_default_config(namespace: str = "default"):
    baseconf = generate_base_nested_config(add_defaults=True, namespace=namespace)
    # baseconf = delete_empty(baseconf)
    return yaml_dump(baseconf)
    # dump using omegaconf
    # conf = OmegaConf.create(baseconf)
    # return OmegaConf.to_yaml(conf)


##────────────────────────────────────────────────────────────────────────────}}}
## {{{               --     loading constants and config     --

from pathlib import Path


BIOCOMP_ROOT_PATH = os.getenv("BIOCOMP_ROOT")
if BIOCOMP_ROOT_PATH is None:
    logger.warning("BIOCOMP_ROOT not defined. Using default paths.")
    BIOCOMP_ROOT_PATH = "~/Dropbox (MIT)/Biocomp/"

DEFAULT_LIB_PATH = Path(BIOCOMP_ROOT_PATH).expanduser() / "partsdb.sqlite"
DEFAULT_LIB_PATH = f"sqlite:///{DEFAULT_LIB_PATH}"
if "BIOCOMP_PARTS_DB" in os.environ:
    DEFAULT_LIB_PATH = Path(os.environ["BIOCOMP_PARTS_DB"]).expanduser().resolve()


def load_lib(lib_path=DEFAULT_LIB_PATH):
    lib = buildLibFromDatabase(lib_path)
    return lib


##────────────────────────────────────────────────────────────────────────────}}}

# ╭─────────────────────────────────────────────╮
# │            general purpose utils            │
# ╰───────────────────── ⟱ ─────────────────────╯

## {{{                        --     list utils     --

ListLike = Union[list, tuple, ListConfig]


def list_like(obj) -> bool:
    return isinstance(obj, (list, tuple, ListConfig))


def as_list(obj: Any) -> Union[list, tuple]:
    """Put obj in a list if it's not already a list or tuple"""
    return [obj] if not isinstance(obj, (list, tuple)) else obj


def flatten_single(t) -> list:
    """Flatten a single level of a nested list"""
    return [item for sublist in t for item in sublist]


def flatten(x) -> list:
    """Flatten nested lists of lists. (always returns a list)"""
    return [a for i in x for a in flatten(i)] if list_like(x) else [x]


def set_list_item(lst: list, i: int, val: Any):
    """make sure that a list has at least i elements and then assign val to the ith element"""
    if len(lst) <= i:
        lst.extend([None] * (i - len(lst) + 1))
    lst[i] = val


def isSubset(l1: Sequence, l2: Sequence) -> bool:
    """Check if all elements of l1 are in l2"""
    for e in l1:
        if e not in l2:
            return False
    return True


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                        --     dict utils     --


def remove_keys(d: Dict, keys: Sequence):
    # ignored_keys = [k for k in keys if k in d]
    new_dict = {k: v for k, v in d.items() if k not in keys}
    return new_dict


def dict_print(d: dict, indent=4):
    res = ""

    def dict_print_impl(d, indentlvl):
        nonlocal res
        for k, v in d.items():
            if dict_like(v):
                res += " " * indentlvl + f"{k}:\n"
                dict_print_impl(v, indentlvl + indent)
            else:
                res += " " * indentlvl + f"{k}: {v}\n"

    dict_print_impl(d, 0)
    rprint(res)


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                     --     profiler context     --

# with profiler('profile.prof'):
#     do_stuff()


@contextmanager
def profiler(filename="profile.prof"):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    profiler.enable()
    yield
    profiler.disable()
    profiler.dump_stats(filename)


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                        --     json utils     --


def load_json5(path: PathLike):
    with open(path) as f:
        return json5.load(f)


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


def make_json_compatible(o, converter=np_converter, float_precision=None):
    if float_precision is not None:
        return json.loads(
            json.dumps(o, default=converter), parse_float=lambda x: round(float(x), float_precision)
        )
    else:
        return json.loads(json.dumps(o, default=converter))


def decode_json(df, cols):
    for col in cols:
        df[col] = df[col].apply(lambda x: json.loads(str(x)))
    return df


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                       --     timing utils     --


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


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                  --     parameter initializers     --


def continuous_initializer(rng, shape=(), minval=0, maxval=1):
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


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                   --     log-poly-log transform     --


def logb(x, base=10):
    """Compute log of x in base b."""
    return np.log(x) / np.log(base)


def cubic_exp_fwd(x, threshold, base, scale: float = 1):
    """
    cubic polynomial that goes through (0,0) and has same first
    and second derivative as the log (in given base) at the threshold.
    In other words, a spline that is log-like near the threshold.

    Args:
    - x: input
    - threshold: the value at which the function should be log-like
    - base: the base of the logarithm
    - scale: a parameter to squeeze (<1) or stretch (>1) the function
    """
    logthresh = np.log(threshold)
    logbase = np.log(base)
    a = -0.5 * (3 - 2 * scale * logthresh) / (threshold**3 * logbase)
    b = -(-4 + 3 * scale * logthresh) / (threshold**2 * logbase)
    c = -0.5 * (5 - 6 * scale * logthresh) / (threshold * logbase)
    return a * x**3 + b * x**2 + c * x


def cubic_exp_inv(y, threshold, base, scale: float):
    """
    inverse of cubic_exp_fwd (on [0,threshold])
    """
    # used wolfram to solve for the analytical inverse
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


def log_poly_log(x, threshold: float = 100, base: int = 10, compression: float = 0.5):
    """
    bi-logarithm function with smooth transition to cubic polynomial between [-threshold, threshold]

    Args:
    - x: input
    - threshold: when the function should transition between log and spline
    - base: the base of the logarithm
    - compression: a parameter to squeeze (<1) or stretch (>1) the function in the spline region
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


def inverse_log_poly_log(y, threshold: float = 100, base: int = 10, compression: float = 0.5):
    """
    inverse of log_poly_log
    """
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
## {{{                         --     jaxutils     --
from jax.experimental import host_callback

enable_checks = False


def grid_map(F, xrange, yrange, meshres):
    import jax

    XX, YY = np.meshgrid(
        np.linspace(xrange[0], xrange[1], meshres[0]),
        np.linspace(yrange[0], yrange[1], meshres[1]),
        indexing="xy",
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


def flat_concat(*arrays):
    return jnp.concatenate([jnp.asarray(a).ravel() for a in arrays])


def str_to_int_array(s):
    return np.array([ord(c) for c in s], dtype=np.int32)


def int_array_to_str(a):
    return "".join([chr(int(c)) for c in a])


def tree_to_jax(params):
    return jax.tree_map(lambda x: jnp.asarray(x), params)


def tree_to_np(params):
    return jax.tree_map(lambda x: np.asarray(x), params)


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                      --     topology analysis helpers     --


def get_uorf_value(param):
    if "tl_rate" in param:
        u = param["tl_rate"][0].split("_")[0]
        try:
            v = int(u[:-1]) * 10
        except ValueError:
            v = 0
        if u[-1] == "w":
            v = v - 5
        return v
    else:
        return 0


UORF_DICT = {
    0: "No uORF",
    5: "weak uORF",
    10: "1x uORF",
    20: "2x uORF",
    30: "3x uORF",
    40: "4x uORF",
    50: "5x uORF",
    60: "6x uORF",
    70: "7x uORF",
    80: "8x uORF",
}


def get_all_ERN_ids(network):
    ERN_ids = network.compute_graph[network.compute_graph["type"] == "sequestron_ERN"].index.values
    return network.sort_nodes_by_upstream(ERN_ids)


def get_all_ERNs_names(network):
    ERNs = network.compute_graph.loc[get_all_ERN_ids(network)]
    ERN_extras = ERNs["extra"].values
    ERN_names = [e["seq_name"].split("#")[0].split("::")[-1] for e in ERN_extras]
    return ERN_names


def get_uorf_names(uorf_values, ern_names):
    uorf_names = []
    for uorf, ern_name in zip(uorf_values, ern_names):
        ERN_uorf, REC_uorf = uorf
        ERN_uorf = UORF_DICT[ERN_uorf]
        REC_uorf = UORF_DICT[REC_uorf]
        uorf_names.append((f"{ern_name} ERN: {ERN_uorf}", f"{ern_name} REC: {REC_uorf}"))
    return uorf_names


def get_all_uorf_values(network):
    cdg = network.central_dogma_graph
    ERNs = network.compute_graph.loc[get_all_ERN_ids(network)]
    ERN_names = get_all_ERNs_names(network)
    ERN_inputs = ERNs["cdg_input"].values
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
    return network.compute_graph[network.compute_graph["type"] == "sequestron_ERN"].index.values


def get_RCB_ids(network):
    return network.compute_graph[
        network.compute_graph["type"].str.startswith("sequestron_R")
    ].index.values


def get_sequestron_ids(network):
    return network.compute_graph[
        network.compute_graph["type"].str.startswith("sequestron_")
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
            raise ValueError("Cycle detected in graph")
        visited.update(independent)
        batches.append(independent)
    return batches


def get_network_family(network):
    erns = get_ERN_ids(network)
    rcbs = get_RCB_ids(network)
    seqs = get_sequestron_ids(network)
    ts = topological_sort(seqs, make_is_upstream(network))

    seqtype = "none"
    family = "unknown"
    match (len(erns) > 0, len(rcbs) > 0):
        case (True, True):
            seqtype = "hybrid"
        case (True, False):
            seqtype = "ERN"
        case (False, True):
            seqtype = "RCB"

    match (len(seqs), len(ts)):
        case (0, 0):
            family = "no device"
        case (1, 1):
            family = "single"
        case (2, 2):
            family = "cascade"
        case (2, 1):
            family = "dual region"
        case (3, 2):
            family = "bandpass"

    return family, seqtype


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                        --     misc utils     --


def get_git_commit_hash():
    bcpath = Path(__file__).parent
    bcpath = bcpath.resolve()
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=bcpath).decode("ascii").strip()


def get_biocomp_version():
    return get_distribution("biocomp").version


def uniqueIdGenerator(start=0):
    unique_id = int(start)

    def uniqueId():
        nonlocal unique_id
        unique_id += 1
        return unique_id - 1

    return uniqueId


##────────────────────────────────────────────────────────────────────────────}}}#                                                                            }}}
