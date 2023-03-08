### {{{                          --     imports     --
from contextlib import contextmanager
from time import sleep
import biocomp as bc
from biocomp import datautils as du
from jax.scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from biocomp.calibration import Calibration
import scriptutils as su
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
def timer(name=None):
    from time import perf_counter
    t = perf_counter()
    yield
    if name is not None:
        print(f"\n{name}: {perf_counter() - t:.2f} seconds")
    else:
        print(f"\nElapsed time: {perf_counter() - t:.2f} seconds")
##────────────────────────────────────────────────────────────────────────────}}}

### {{{                   --     vectorized get_param     --

a = jnp.arange(10.0)
name = 'a'
node_id = 4


# def select(a, i, axis=0):
    # """Reduces an array to its ith element along axis"""
    # mask = jax.nn.one_hot(i, a.shape[0])
    # # replace 0 with nan so that we don't return ambiguous zeros
    # # when i is out of bounds
    # mask = jnp.where(mask == 0, jnp.nan, mask)
    # c = (a.swapaxes(axis, -1) * mask).swapaxes(axis, -1)
    # # return jnp.sum(c, axis=axis)
    # return jnp.nanmean(c, axis=axis)


def get_param_vect(
    params,
    name,
    node_id=0,
    init=None,
    overwrite_with=None,
    shared=False,
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


    # The slight complication here is that we can't jit/vectorize a
    # dictionnary lookup. i.e we can't do:
    # res = params[node_id] as this requires branching
    # Indexing an array is fine though, so we could simply create
    # an array of params for each node that is as big as the largest
    # node_id, and then index it. However, this would be wasteful
    # for params that have large shapes.

    # So instead I add one layer of indirection:
    # we save a key_vec which will contain -1 for all nodes that don't have
    # a param for this path, and the actual param_id for the nodes that do.
    # This way we can jit the lookup, and only need to store the params
    # that are actually used. The extra '-1' entries are certainly not
    # a problem. (And easier to deal with than sparse arrays)
    

    nparams = ut.at_path(params, dpath, None)
    nparams = nparams.shape[0] if nparams is not None else 0

    keys_path = ['keys'] + dpath
    key_vec = ut.at_path(params, keys_path, None)

    if not read_only: # non-jittable path (only used for initialization)
        assert (init is None) != (overwrite_with is None)
        if key_vec is None or node_id >= key_vec.shape[0]:
            v = key_vec if key_vec is not None else jnp.zeros((0,), dtype=jnp.int32)
            v = jnp.concatenate([v, jnp.full((node_id - v.shape[0] + 1,), -1, dtype=jnp.int32)])
            key_vec = v

        param_id = key_vec[node_id]
        if param_id == -1: # param doesn't exist yet
            try:
                new_param_value = init()
                p = ut.at_path(params, dpath)
                if p is None: # first param ever for this path
                    p = jnp.expand_dims(new_param_value, axis=0)
                else: # add new param to existing array
                    p = jnp.concatenate([p, jnp.expand_dims(new_param_value, axis=0)])
                ut.at_path(params, dpath, p) # update params
                # update and save key_vec:
                ut.at_path(params, keys_path, key_vec.at[node_id].set(nparams)) 
            except Exception as e:
                msg = f'Error initializing param "{name}" from node {node_id}: {e}'
                raise RuntimeError(msg) from e

    param_id = key_vec[node_id]

    assert param_id != -1, f'Param "{name}" not found for node {node_id}'
    # right now when jitted, param_id will be clamped to the last or first index
    # we could pad the array with nans to make sure things stand out, but
    # I don't think it's worth the hassle + memory cost.

    if overwrite_with is not None: # also non-jittable
        assert read_only is False, f'Cannot overwrite a read_only param for path {dpath}'
        allp = ut.at_path(params, dpath)
        allp = allp.at[param_id].set(overwrite_with)
        ut.at_path(params, dpath, allp)

    return ut.at_path(params, dpath)[param_id]



params = {}
key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)
get_param_vect(params, 'a', init=lambda: jax.random.normal(key, (3, 3)), read_only=False, shared=True)
get_param_vect(params, 'n_a', init=lambda: jax.random.normal(k1, (3,2 )), read_only=False, node_id=7)
get_param_vect(params, 'n_a', init=lambda: jax.random.normal(k2, (3,2 )), read_only=False, node_id=4)
get_param_vect(params, 'n_a', init=lambda: jax.random.normal(k3, (3,2 )), read_only=False, node_id=0)
params

gp = partial(get_param_vect, params, 'n_a', init=None)

jit(vmap(gp))(jnp.arange(8))


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                        --     node config     --
T_SIZE = 64
T_DEPTH = 4
I_SIZE = 64
I_DEPTH = 3
I_OUT = 8
ERN_SIZE = 128
ERN_DEPTH = 4
MEFL_SIZE = 64
MEFL_DEPTH = 4

node_impl = dict(
    bc.nodes.DEFAULT_COMPUTE_NODES_DICT,
    **{
        'output': partial(bc.nn.output, wsize=MEFL_SIZE, depth=MEFL_DEPTH),
        'transcription': partial(
            bc.nn.transcription,
            outer_wsize=T_SIZE,
            outer_depth=T_DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
            inner_out=I_OUT,
        ),
        'translation': partial(
            bc.nn.translation,
            outer_wsize=T_SIZE,
            outer_depth=T_DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
            inner_out=I_OUT,
        ),
        'inv_transcription': partial(
            bc.nn.inv_transcription,
            outer_wsize=T_SIZE,
            outer_depth=T_DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
            inner_out=I_OUT,
        ),
        'inv_translation': partial(
            bc.nn.inv_translation,
            outer_wsize=T_SIZE,
            outer_depth=T_DEPTH,
            inner_wsize=I_SIZE,
            inner_depth=I_DEPTH,
            inner_out=I_OUT,
        ),
        'sequestron_ERN': partial(bc.nn.ERN5p, wsize=ERN_SIZE, depth=ERN_DEPTH),
        'sequestron_ERN3p': partial(bc.nn.ERN3p, wsize=ERN_SIZE, depth=ERN_DEPTH),
    },
)

config = {
    **bdf.DEFAULT_CONFIG,
    **{
        'node_impl': node_impl,
        'rng_key': np.random.randint(0, 2**32),
        "batch_size": 16,
        "n_batches": 2048,
        "epochs": 3,
        "log_factor": 2e4,
    },
}

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                     --     load data     --
lib = su.load_lib()
matrix_xp = su.load_xp('2023-02-16_Matrix', lib, data_path='./data/calibrated_data')
dman = du.DataManager.from_xps([matrix_xp], config, inverse='all')
names = [m.node_namespace for m in dman.get_models()]
key = jax.random.PRNGKey(0)
##────────────────────────────────────────────────────────────────────────────}}}

subset = list(range(3))
dman.set_subset(subset)

models = dman.get_models()

params = {}
for m, k in tqdm(zip(models, jax.random.split(key, len(models)))):
    params, _ = m.init(k, pre_params=params)

### {{{                          --     batches     --

with timer():
    batches = dman.get_batches(key)

# ## plot each kdes over a range from 0 to 1
# res = 300
# xx = np.linspace(0, 1, res)
# xygrid = np.meshgrid(xx, xx)
# xygrid = np.stack(xygrid, axis=-1)
# xygrid = xygrid.reshape(-1, 2)
# for kde in dman.get_kdes():
    # z = kde(xygrid.T)
    # z = z / z.max()
    # z = z.reshape(res, res)
    # logz = np.log10(z+1)
    # plt.imshow(logz)
    # plt.show()

Y = dman.get_Y()
xb, yb = batches
yb.shape
xb.shape
# flatten first 2 dimensions of yb
ybf = yb.reshape(-1, yb.shape[-1])

du.fluo_densities(ybf[:,:3]*10, ['bfp','yfp','mkate'], logscale=False, bw_method=0.1)
du.fluo_densities(Y[0]*10, ['bfp','yfp','mkate'], logscale=False, bw_method=0.1)
# for y in Y:


##────────────────────────────────────────────────────────────────────────────}}}


### {{{                 --     vectorized get_quantized     --

def get_possible_parts(param_name, cdg_node_id, cdg):
    # should return the name of possible parts for a given cdg node, slot and param name
    # example: get_possible_values('transcription_rate', ...) -> ['hEF1a', 'hEF1b', 'hEF1c']
    #          get_possible_values('translation_rate', ...) -> [None, '1xuORF', '2xuORF', ...]
    # params are stored in the params column of the cdg as a dict {param_name:[possiblevaluees]}
    available_params = cdg.loc[cdg_node_id, 'params']
    if param_name not in available_params:
        raise ValueError(
            f'Param {param_name} not available for cdg node {cdg_node_id}. Available: {available_params}'
        )
    return available_params[param_name]

def get_quantized(
    get_param,
    param_name,
    values,
    node_id,
    cdf,
    cdg,
    quantize_fun,
    rng_key,
    mode='input_edges',
    logto=None,
):
    """Return a *quantized* version of the parameter, conditioned on the
    relevant species (either input or output cdg nodes). mode can be 'input' or 'output'."""
    # We are selecting possible values for a parameter depending on which path, or edge, they come from.
    # The edges of the compute graph are basically nodes in the central dogma graph,
    # i.e they're *more or less*
    # dual to each other, but not exactly since the compute graph adds interactions and extra nodes).
    # Anyway I think it's reasonable to consider that the type of nodes that will call getQuantized
    # are the types for which it's ok to consider the CDG as the dual of the COMPG.

    # check that mode is either input_edges or output_edges or inner
    if mode not in ['input_edges', 'output_edges', 'inner']:
        raise ValueError(f'Invalid mode {mode} for get_quantized')

    cdg_ids = (
        cdf.loc[node_id]['cdg_input'] if mode == 'input_edges' else cdf.loc[node_id]['cdg_output']
    )
    if cdg_ids is None:
        raise ValueError(f'Node {node_id} has no {mode} CDG node')
    if not isinstance(cdg_ids, list):
        cdg_ids = [cdg_ids]

    possible_parts = [get_possible_parts(param_name, cdg_id, cdg) for cdg_id in cdg_ids]

    # concat part names with param_name
    possible_names = [[f'{part}::{param_name}' for part in parts] for parts in possible_parts]
    assert len(possible_names) == len(
        values
    ), f'len(possible_names)={len(possible_names)} != len(values)={len(values)}'

    possible_values = [
        jnp.array(
            [
                get_param(n, ut.continuous_initializer(k, val.shape), shared=True)
                for n, k in zip(names, jax.random.split(kk, len(names)))
            ]
        )
        for names, val, kk in zip(possible_names, values, jax.random.split(rng_key, len(values)))
    ]

    if logto is not None:
        if node_id in logto:
            logto[node_id].add(param_name)
        else:
            logto[node_id] = {param_name}

    res = jnp.array([quantize_fun(v, p) for v, p in zip(values, possible_values)])
    return res



##────────────────────────────────────────────────────────────────────────────}}}


### {{{                   --     loss and compilation     --

zb = jax.random.uniform(key, yb.shape)
b = 0
xxb = xb[b]
yyb = yb[b]
zzb = zb[b]

nmodels = len(models)
x_start = np.cumsum([m.n_inputs for m in models])[:-1]
y_start = np.cumsum([m.n_outputs for m in models])[:-1]

def huber_quantile_loss(e, q, delta=0.1):
    return jnp.where(
        jnp.abs(e) <= delta, 0.5 * e**2, delta * (jnp.abs(e) - 0.5 * delta)
    ) * jnp.where(e < 0, q, (1.0 - q))

def apply_models_and_grad(params, x, z, key):
    k1, k2 = jax.random.split(key)
    keys = jax.random.split(k1, nmodels)
    xs = jnp.split(x, x_start)
    zs = jnp.split(z, y_start)
    res = [
        m.apply_and_negative_grad(params, xx, zz, k)
        for m, xx, zz, k in zip(models, xs, zs, keys)
    ]
    keys = jax.random.split(k2, nmodels)
    just_for_grads = [
        m.apply_and_negative_grad(
            params, xx, zz, k, override_w_uniform=['transcription', 'translation', 'output']
        )
        for m, xx, zz, k in zip(models, xs, zs, keys)
    ]
    yhat, negative_grads_x = zip(*res)
    _, negative_grads_rdm = zip(*just_for_grads)
    return jnp.concatenate(yhat, axis=0), jnp.sum(
        ut.flat_concat(*[negative_grads_x, negative_grads_rdm])
    )

def loss_func(params, X, Y, Z, key):
    assert X.ndim == Y.ndim == Z.ndim == 2
    assert X.shape[0] == Y.shape[0] == Z.shape[0]
    assert X.shape[1] == sum([m.n_inputs for m in models])
    assert Y.shape[1] == Z.shape[1] == sum([m.n_outputs for m in models])

    keys = jax.random.split(key, X.shape[0])
    yhat, grads = vmap(apply_models_and_grad, in_axes=(None, 0, 0, 0))(params, X, Z, keys)
    assert yhat.shape == Y.shape
    assert grads.shape == (X.shape[0],)

    error = yhat - Y
    qantile_loss = jnp.mean(
        huber_quantile_loss(error, Z, delta=config['huber_quantile_loss_delta'])
    )

    negative_grad_penalty = config['negative_grad_penalty'] * jnp.mean(
        jnp.where(grads < 0, -grads, 0)
    )

    return qantile_loss + negative_grad_penalty



def step(params, x, y, z, key):
    loss, grads = value_and_grad(loss_func)(params, x, y, z, key)
    return loss, grads


def compilation_analysis(f, *args):
    print('Compiling...')
    with timer('Total compilation time'):
        with timer('lowering'):
            lowered = jit(f).lower(*args)
        with timer('compiling'):
            compiled = lowered.compile()
    lowered_text = lowered.as_text()
    lowered_nlines = lowered_text.count('\n')
    compiled_text = compiled.as_text()
    compiled_nlines = compiled_text.count('\n')
    print(f'Lowered: {lowered_nlines} lines')
    print(f'Compiled: {compiled_nlines} lines')
    print(f'Compiled cost analysis:{compiled.cost_analysis()}')
    print(f'Compiled memory analysis:{compiled.memory_analysis()}')
    return compiled


compiled = compilation_analysis(step, params, xxb, yyb, zzb, key)

xxb.shape
xx = xxb[0]
zz = zzb[0]

jit(models[0])(params,xx[2:4],zz[:3],key)

##────────────────────────────────────────────────────────────────────────────}}}



def test(a, i):
    return a[i]

a = jnp.arange(10).astype(jnp.float32)
a_dict = {i: float(a[i]) for i in range(10)}

jit(test)(a_dict, 2)
