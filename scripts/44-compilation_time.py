### {{{                          --     imports     --
from contextlib import contextmanager
from time import sleep
import biocomp as bc
from biocomp import datautils as du
from jax.scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from biocomp.calibration import Calibration
import scriptutils as sut
from biocomp import utils as ut
from pathlib import Path
import json5
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import gaussian_kde
import jax
import optax
from jax import jit, vmap, value_and_grad
from jax.tree_util import Partial as partial
from tqdm import tqdm
import biocomp.defaults as bdf
import pandas as pd

@contextmanager
def timer():
    from time import perf_counter
    t = perf_counter()
    yield
    print(f"\nElapsed time: {perf_counter() - t:.2f} seconds")
with timer():
    sleep(1)
##────────────────────────────────────────────────────────────────────────────}}}


a = jnp.arange(10.0)

def select(a, i, axis=0):
    m = jax.nn.one_hot(i, a.shape[0])
    c = (a.swapaxes(axis, -1) * m).swapaxes(axis, -1)
    return jnp.sum(c, axis=axis)


jax.grad(select)(a, 5.0)

name = 'a'
node_id = 4
def get_param(
    params,
    name,
    init=None,
    overwrite_with=None,
    shared=False,
    node_id=0,
    node_namespace=None,
    read_only=True,
    **kwargs,
):
    assert isinstance(params, dict), f'params must be a dict, not {type(params)}'
    dpath = []
    if not shared:
        dpath.append('node')
        if node_namespace is not None:
            dpath.append(node_namespace)
    else:
        dpath.append('shared')

    dpath.append(name)

    keys_path = ['keys'] + dpath

    assert (init is None) != (overwrite_with is None)

    # we save a key_vec which will contain -1 for all nodes that don't have
    # a param for this path, and the param_id for the node that does

    nparams = ut.at_path(params, dpath, None)
    nparams = nparams.shape[0] if nparams is not None else 0

    key_vec = ut.at_path(params, keys_path, None)
    if key_vec is None or node_id >= key_vec.shape[0]:
        assert read_only is False, f'Cannot initialize a read_only param for path {dpath}'
        v = key_vec if key_vec is not None else jnp.zeros((0,), dtype=jnp.int32)
        v = jnp.concatenate([v, jnp.full((node_id - v.shape[0] + 1,), -1, dtype=jnp.int32)])
        key_vec = v

    param_id = select(key_vec, node_id)

    if param_id == -1:
        assert read_only is False, f'Cannot initialize a read_only param for path {dpath}'
        try:
            param_id = nparams
            p = ut.at_path(params, dpath)
            obj = init()
            if p is None:
                p = jnp.expand_dims(obj, axis=0)
            else:
                p = jnp.concatenate([p, jnp.expand_dims(obj, axis=0)])
            ut.at_path(params, dpath, p)
            key_vec = key_vec.at[node_id].set(param_id)
            ut.at_path(params, keys_path, key_vec)
        except Exception as e:
            msg = f'Error initializing param "{name}" from node {node_id}: {e}'
            raise RuntimeError(msg)

    if overwrite_with is not None:
        assert read_only is False, f'Cannot overwrite a read_only param for path {dpath}'
        allp = ut.at_path(params, dpath)
        allp = allp.at[param_id].set(overwrite_with)
        ut.at_path(params, dpath, allp)

    allp = ut.at_path(params, dpath)
    return select(allp, param_id)



params = {}
key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)
get_param(params, 'a', lambda: jax.random.normal(key, (3, 3)), read_only=False, shared=True)
get_param(params, 'n_a', lambda: jax.random.normal(k1, (3, 2)), read_only=False, node_id=4)
get_param(params, 'n_a', lambda: jax.random.normal(k2, (3, 2)), read_only=False, node_id=5)
params
##

a1 = jnp.arange(6).reshape((2, 3))
a2 = jnp.arange(6,12).reshape((2, 3))

ac = jnp.concatenate([jnp.expand_dims(a1, axis=0), jnp.expand_dims(a2, axis=0)], axis=0)
ac.shape

mask = jnp.array([0.0, 1.0])
# multiply by mask along axis 0. Need to broadcast automatically (to any shape from ac)
def mul(a, m, axis=0):
    # we vmap over the first axis of a, and broadcast m to the same shape
    c = (a.swapaxes(axis, -1) * m).swapaxes(axis, -1)
    return c.sum(axis=0)

mul(ac, jnp.array([1.0, 0.0]))

