### {{{                          --     imports     --
from contextlib import contextmanager
from time import sleep
import biocomp as bc
from biocomp import datautils as du
from biocomp import nodes as nd
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


from rich import print as pprint

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
    base_path=ut.NODE_PATH,
    init=None,
    overwrite_with=None,
    read_only=True,
    **_,
):
    """
    Retrieves a parameter from the given params dictionary.
    """
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

    # I think we can use node_id with base_path = shared to vectorize tl vs tx by accessing different weights!

    assert isinstance(params, dict), f'params must be a dict, not {type(params)}'

    dpath = base_path + [name]

    nparams = ut.at_path(params, dpath, None)
    nparams = nparams.shape[0] if nparams is not None else 0

    keys_path = ut.KEYS_PATH + dpath
    key_vec = ut.at_path(params, keys_path, None)

    if not read_only:  # non-jittable path (only used for initialization)
        assert (init is None) != (overwrite_with is None)
        if key_vec is None or node_id >= key_vec.shape[0]:
            v = key_vec if key_vec is not None else jnp.zeros((0,), dtype=jnp.int32)
            key_vec = jnp.concatenate(
                [v, jnp.full((node_id - v.shape[0] + 1,), -1, dtype=jnp.int32)]
            )

        param_id = key_vec[node_id]
        if param_id == -1:  # param doesn't exist yet
            try:
                new_param_value = overwrite_with if overwrite_with is not None else init()
                p = ut.at_path(params, dpath)
                if p is None:  # first param ever for this path
                    p = jnp.expand_dims(new_param_value, axis=0)
                else:  # add new param to existing array
                    p = jnp.concatenate([p, jnp.expand_dims(new_param_value, axis=0)])
                ut.at_path(params, dpath, p)  # update params
                # update and save key_vec:
                key_vec = ut.at_path(params, keys_path, key_vec.at[node_id].set(nparams))
            except Exception as e:
                msg = f'Error initializing param "{name}" from node {node_id}: {e}'
                raise RuntimeError(msg) from e

    param_id = key_vec[node_id]

    if overwrite_with is not None and not read_only:  # also non-jittable
        allp = ut.at_path(params, dpath).at[param_id].set(overwrite_with)
        ut.at_path(params, dpath, allp)

    return ut.at_path(params, dpath)[param_id]


def get_quantized_vect(
    values_to_quantize,
    params,
    param_name,
    node_id,
):
    # initialization of both keys and values is done upstream. We assume both are already initialized
    possible_values = get_param_vect(params, param_name, base_path=ut.SHARED_PATH, read_only=True)
    masks = get_param_vect(
        params, param_name, node_id=node_id, base_path=ut.MASK_PATH, read_only=True
    )
    # masks is a 2D array of shape (max_n_masks_per_node, n_qvalues)
    masks = masks[: values_to_quantize.shape[0]]
    return vmap(nd.quantize_masked, in_axes=(0, None, 0), out_axes=0)(
        values_to_quantize, possible_values, masks
    )


params = {}
key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)
get_param_vect(
    params, 'a', init=lambda: jax.random.normal(key, (3, 3)), read_only=False, base_path=['shared']
)
get_param_vect(
    params, 'n_a', init=lambda: jax.random.normal(k1, (3, 2)), read_only=False, node_id=7
)
get_param_vect(
    params, 'n_a', init=lambda: jax.random.normal(k2, (3, 2)), read_only=False, node_id=4
)
get_param_vect(
    params, 'n_a', init=lambda: jax.random.normal(k3, (3, 2)), read_only=False, node_id=0
)

gp = partial(get_param_vect, params, 'n_a', init=None)
jit(vmap(gp))(jnp.arange(8))

jnp.full((2,), np.nan)


def initialize_quantization_values(params, pname, qnames, init):
    qname_path = ut.QNAME_PATH + [pname]
    qnames = sorted(qnames)
    assert len(qnames) == len(
        set(qnames)
    ), f'quantization names for {pname} must be unique, got {qnames}'
    already = ut.at_path(params, qname_path)
    if already is None:
        ut.at_path(params, qname_path, qnames)
        possible_values = jnp.array([init() for _ in qnames])
        ut.at_path(params, ut.QVALS_PATH + [pname], possible_values)
    else:
        assert (
            qnames == already
        ), f'qnames for {pname} already initialized to {already}, cannot change to {qnames}'


qnames = ['q1', 'q2', 'q3']
pname = 'quantized_param_1'

initialize_quantization_values(params, pname, qnames, lambda: jax.random.uniform(key))


def save_quantization_mask(params, pname, mask, node_id):
    mask = jnp.array(mask)
    assert mask.dtype == jnp.bool_, f'mask must be boolean, got {mask.dtype}'
    assert (
        mask.shape[0] == ut.at_path(params, ut.QVALS_PATH + [pname]).shape[0]
    ), f'mask must have same length as possible values for {pname}, got {mask.shape[0]} and {ut.at_path(params, ut.QVALS_PATH + [pname]).shape[0]}'
    mask_path = ut.MASK_PATH
    get_param_vect(
        params, pname, node_id=node_id, base_path=mask_path, overwrite_with=mask, read_only=False
    )


pprint(params)

save_quantization_mask(params, pname, [True, False, True], 0)

pprint(params)

# Define some input data for the tests
values_to_quantize = np.array([1.2, 3.4, 5.6])
possible_values = np.array([1.5, 2, 3, 4, 5, 6])
mask = np.array([1, 0, 1, 0, 0, 1])


def get_available_quantizations(param_name, cdg_node_id, cdg):
    # returns the name of possible parts for a given cdg node, slot and param name
    # example: get_possible_values('transcription_rate', ...) -> ['hEF1a', 'hEF1b', 'hEF1c']
    #          get_possible_values('translation_rate', ...) -> [None, '1xuORF', '2xuORF', ...]
    # params are stored in the params column of the cdg as a dict {param_name:[possiblevaluees]}
    available_params = cdg.loc[cdg_node_id, 'params']
    if param_name not in available_params:
        raise ValueError(
            f'Param {param_name} not available for cdg node {cdg_node_id}. Available: {available_params}'
        )
    return available_params[param_name]


def generate_quantization_masks(params, pname, node_id, n_masks_per_node, cdf, cdg):
    qnames = ut.at_path(params, ut.QNAME_PATH + [pname])
    assert qnames is not None, f'quantization names for {pname} not initialized'

    cdg_ids = cdf.loc[node_id]['cdg_input']
    assert cdg_ids is not None, f'Node {node_id} has no input CDG node'
    cdg_ids = [cdg_ids] if not isinstance(cdg_ids, list) else cdg_ids

    this_node_qnames = [get_available_quantizations(pname, cid, cdg) for cid in cdg_ids]
    # we have one mask per CDG input, and we need the same mask shape for all nodes
    assert len(this_node_qnames) <= n_masks_per_node, (
        f'Node {node_id} has {len(this_node_qnames)} CDG inputs, '
        f'but only a max of {n_masks_per_node} masks are available'
    )

    # now create the mask array
    mask = np.zeros((n_masks_per_node, len(qnames)), dtype=bool)
    for i in range(len(this_node_qnames)):
        mask[i, [qnames.index(q) for q in this_node_qnames[i]]] = True

    save_quantization_mask(params, pname, mask, node_id)


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

du.fluo_densities(ybf[:, :3] * 10, ['bfp', 'yfp', 'mkate'], logscale=False, bw_method=0.1)
du.fluo_densities(Y[0] * 10, ['bfp', 'yfp', 'mkate'], logscale=False, bw_method=0.1)
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
        m.apply_and_negative_grad(params, xx, zz, k) for m, xx, zz, k in zip(models, xs, zs, keys)
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

jit(models[0])(params, xx[2:4], zz[:3], key)

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                           --     tools     --
def get_batch_sequence_of_nodes(network):
    """Returns a list of lists of compute nodes from the network,
    where each node of a sublist can be computed independently of the others,
    but each sublist must be computed in order."""
    visited = set()
    batches = []
    while len(visited) < len(network.compute_graph):
        independent = [
            i
            for i, row in network.compute_graph.iterrows()
            if (not row['input_from'] or all([x[0] in visited for x in row['input_from']]))
            and i not in visited
        ]
        if not independent:
            msg = f'Invalid compute graph, no independent nodes found. Remaining nodes: {set(network.compute_graph.index) - visited}. visited={visited}'
            raise ValueError(msg)
        visited.update(independent)
        batches.append(independent)
    return batches

##────────────────────────────────────────────────────────────────────────────}}}

network = models[0].network.copy()
cg = network.compute_graph

node_impl = nd.DEFAULT_COMPUTE_NODES_DICT
node_namespace = None


batches = get_batch_sequence_of_nodes()
# each sublist in batches contains independent nodes that can be safely called in parallel
# however we are not parallelizing the calls, so we can flatten the list and be sure
# that the dependency order is respected (upstream nodes will always be called before downstream ones)
flat_batches = [item for sublist in batches for item in sublist]
call_dicts = []
for nid in flat_batches:
    call_d = {}
    node_row = cg.loc[nid]

    # if it's an inverse node, we need to get the id of the node that it's inverting
    nodeid_for_getters = nid
    if node_row.extra is not None and 'is_inverse_of' in node_row.extra:
        nodeid_for_getters = node_row.extra['is_inverse_of']

    # a node needs a method to access the parameters.
    # we simply preset the general purpose get_param on the correct nodeid
    get_p = partial(
        get_param_,
        node_id=nodeid_for_getters,
    )
    # same with get_quantized
    # we preset the node id, the central dogma graph and the compute graph
    # as well as the actual quantize function
    get_q = partial(
        get_quantized,
        node_id=nodeid_for_getters,
        cdf=cg,
        cdg=self.network.central_dogma_graph,
        quantize_fun=nd.quantize,
    )

    # extra_params will be passed to the node function
    # n_outputs and n_inputs are always passed
    # + any specific thing that the network builder has added
    extra_params = {
        'n_outputs': len(cg.loc[nid]['output_to']),
        'n_inputs': len(cg.loc[nid]['input_from']),
    }
    if node_row.extra is not None:
        extra_params.update(node_row.extra)

    call_d['get_p'] = get_p  # a node needs a method to access the parameters
    call_d['get_q'] = get_q  # and a method for quantization
    call_d['extra_params'] = extra_params  # these will be appended to the node call
    call_d['type'] = node_row.type
    call_d['input_from'] = node_row.input_from  # upstream node(s) we are taking input from
    call_d['fun'] = None  # the actual function to call (set later)
    call_d['nid'] = nid  # this node's id, important to store the output

    if node_row.type not in ('input'):
        assert node_row.type in self.node_impl, f'Unimplemented node type {node_row.type}'
        call_d['fun'] = self.node_impl[node_row.type]  # the actual function to call

    call_dicts.append(call_d)


# {{{                  --     ComputeGraphModel class     --
# ···············································································

from typing import List, Dict, Tuple, Union, Optional, Callable, Any
from jax.tree_util import Partial as partial


class ComputeGraphModel_v2:
    def __init__(self, network):
        self.network = network
        cg = self.network.compute_graph
        assert len(cg[cg['type'] == 'output']) == 1, 'The graph must have exactly one output node'
        self.n_inputs = len(cg[cg['type'] == 'input'])
        self.n_outputs = len(cg[cg['type'] == 'output'].input_from[0])
        self.built = False

    def build(
        self,
        node_impl: Dict[str, Callable] = nd.DEFAULT_COMPUTE_NODES_DICT,
        node_namespace: Optional[str] = None,
    ):
        """Builds the model, i.e. creates a list of functions to be called in sequence
        (together with the necesary information to call them) according to the compute graph"""

        self.node_namespace = node_namespace
        assert self.network is not None
        assert self.network.is_built()
        assert isinstance(node_impl, dict)
        self.node_impl = node_impl

        cg = self.network.compute_graph

        # we'll build the list of call_dicts
        # a call dict is a dictionary that contains the relevant information in
        # order to actually call a node function

        batches = self.__get_batch_sequence_of_nodes()
        # each sublist in batches contains independent nodes that can be safely called in parallel
        # however we are not parallelizing the calls, so we can flatten the list and be sure
        # that the dependency order is respected (upstream nodes will always be called before downstream ones)
        flat_batches = [item for sublist in batches for item in sublist]
        call_dicts = []
        for nid in flat_batches:
            call_d = {}
            node_row = cg.loc[nid]

            # if it's an inverse node, we need to get the id of the node that it's inverting
            nodeid_for_getters = nid
            if node_row.extra is not None and 'is_inverse_of' in node_row.extra:
                nodeid_for_getters = node_row.extra['is_inverse_of']

            # a node needs a method to access the parameters.
            # we simply preset the general purpose get_param on the correct nodeid
            get_p = partial(
                get_param_,
                node_id=nodeid_for_getters,
            )
            # same with get_quantized
            # we preset the node id, the central dogma graph and the compute graph
            # as well as the actual quantize function
            get_q = partial(
                get_quantized,
                node_id=nodeid_for_getters,
                cdf=cg,
                cdg=self.network.central_dogma_graph,
                quantize_fun=nd.quantize,
            )

            # extra_params will be passed to the node function
            # n_outputs and n_inputs are always passed
            # + any specific thing that the network builder has added
            extra_params = {
                'n_outputs': len(cg.loc[nid]['output_to']),
                'n_inputs': len(cg.loc[nid]['input_from']),
            }
            if node_row.extra is not None:
                extra_params.update(node_row.extra)

            call_d['get_p'] = get_p  # a node needs a method to access the parameters
            call_d['get_q'] = get_q  # and a method for quantization
            call_d['extra_params'] = extra_params  # these will be appended to the node call
            call_d['type'] = node_row.type
            call_d['input_from'] = node_row.input_from  # upstream node(s) we are taking input from
            call_d['fun'] = None  # the actual function to call (set later)
            call_d['nid'] = nid  # this node's id, important to store the output

            if node_row.type not in ('input'):
                assert node_row.type in self.node_impl, f'Unimplemented node type {node_row.type}'
                call_d['fun'] = self.node_impl[node_row.type]  # the actual function to call

            call_dicts.append(call_d)

        def collect_all_results(
            params: dict,
            inputs: jnp.ndarray,
            quantiles: jnp.ndarray,
            rng_key,
            read_only=True,
            constraints=None,
            with_grad: Optional[list[str]] = None,
            override_w_uniform: Optional[list[str]] = None,
        ):
            """
            params: the parameters
            inputs: the inputs to the network
            quantiles: array of quantiles we want to estimate (1 per output)
            rng_key: the rng key
            read_only: will make sure that the parameters are not modified by the node functions

            Executes and collects all the results of the nodes in the compute graph.
            This method basically just calls the nodes in call_dicts, in order, populating
            the result dictionnary with each node's output and feeding the output of each
            upstream node as input to the next downstream one.


            Special parameters (mostly useful during training):
            --------------------------------------------

            constraints: a list of constraints to be applied to the output of the nodes (like clamping)
            with_grad: if True, will also return the gradient (the full jacobian) of the output
            with respect to the input of the specified node types. Useful for monotonicity constraints.

            override_w_uniform: if not None, will override the inputs to the specified
            node types with uniform values. This is useful for collecting the behavior of
            some nodes across their entire input space so that they can't learn sneaky tricks
            like being twice increasing in two different regions of the input space where the "junction"
            is never visited by the training data (and thus would never show up in the gradients)

            """
            assert (
                len(inputs) >= self.n_inputs
            ), f'len(inputs)={len(inputs)} < n_inputs={self.n_inputs}'

            keys = jax.random.split(rng_key, len(flat_batches))

            results = {}
            grads = []

            for (n, key) in zip(call_dicts, keys):
                k1, k2, k3 = jax.random.split(key, 3)

                extra_params = n['extra_params']

                get_grad = with_grad and n['type'] in with_grad

                nid = n['nid']

                if n['type'] == 'input':
                    results[nid] = inputs[n['extra_params']['input_position']]
                    continue

                upstream_results = []

                if override_w_uniform is not None and n['type'] in override_w_uniform:
                    subkeys = jax.random.split(k1, extra_params['n_inputs'])
                    for inp, k in zip(n['input_from'], subkeys):
                        if len(results[inp[0]].shape) == 0:
                            upstream_results.append(jax.random.uniform(k))
                        else:
                            upstream_results.append(
                                jax.random.uniform(k, shape=results[inp[0]][inp[1]].shape)
                            )
                else:
                    for inp in n['input_from']:
                        if len(results[inp[0]].shape) == 0:
                            upstream_results.append(results[inp[0]])
                        else:
                            upstream_results.append(results[inp[0]][inp[1]])

                get_p = partial(
                    n['get_p'],
                    params,
                    node_namespace=self.node_namespace,
                    constraints=constraints,
                    read_only=read_only,
                )

                qtl = None
                if quantiles is not None:
                    pick_quantile = n['extra_params'].get('quantile_variable_id', [])
                    if len(pick_quantile) > 0 and all([x is not None for x in pick_quantile]):
                        qtl = quantiles[jnp.array(pick_quantile)]

                get_q = partial(n['get_q'], get_p, rng_key=k2)
                comp_node = n['fun'](get_p, get_q, **extra_params)

                f = partial(comp_node, quantile=qtl, rng_key=k3)

                res = f(*upstream_results)

                if get_grad:
                    n_upstream = len(upstream_results)
                    grad = jax.jacfwd(f, argnums=list(range(n_upstream)))(*upstream_results)
                    grads.append(grad)

                results[nid] = res

                if n['type'] == 'output':
                    if with_grad:
                        return res, grads, results
                    return res, results

            raise ValueError('Invalid compute graph, no output node found')

        def apply(*args, **kwargs):
            """Executes the model. It simply calls collect_all_results and only returns the final output"""
            return collect_all_results(*args, with_grad=False, **kwargs)[0]

        def apply_and_negative_grad(
            *args, with_grad=['transcription', 'translation', 'output'], **kwargs
        ):
            allres = collect_all_results(*args, with_grad=with_grad, **kwargs)
            grads = jnp.zeros(1)
            if len(allres[1]) > 0:
                grads = ut.flat_concat(*allres[1])
            negative_grad = jnp.mean(jnp.where(grads < 0, grads, 0))
            return allres[0], negative_grad

        def init(rng_key, pre_params=None, pre_constraints=None):
            params = {} if pre_params is None else pre_params
            constraints = {} if pre_constraints is None else pre_constraints
            apply(
                params,
                jnp.zeros(self.n_inputs),
                jnp.zeros(self.n_outputs),
                rng_key,
                constraints=constraints,
                read_only=False,
            )
            return params, constraints

        self.apply = apply
        self.apply_and_negative_grad = apply_and_negative_grad
        self.collect_all_results = collect_all_results
        self.init = init
        self.flat_batches = flat_batches
        self.built = True

    def __repr__(self):
        def list_network_inputs():
            return [self.network.compute_graph[self.network.compute_graph['type'] == 'input']]

        def list_network_outputs():
            return [self.network.compute_graph[self.network.compute_graph['type'] == 'output']]

        return f'ComputeGraphModel({list_network_inputs()} -> {list_network_outputs()})'

    def __call__(self, *args, **kwargs):
        assert self.built
        return self.apply(*args, **kwargs)

    def __get_batch_sequence_of_nodes(self):
        """Returns a list of lists of compute nodes from the network,
        where each node of a sublist can be computed independently of the others,
        but each sublist must be computed in order."""
        visited = set()
        batches = []
        while len(visited) < len(self.network.compute_graph):
            independent = [
                i
                for i, row in self.network.compute_graph.iterrows()
                if (not row['input_from'] or all([x[0] in visited for x in row['input_from']]))
                and i not in visited
            ]
            if not independent:
                msg = f'Invalid compute graph, no independent nodes found. Remaining nodes: {set(self.network.compute_graph.index) - visited}. visited={visited}'
                raise ValueError(msg)
            visited.update(independent)
            batches.append(independent)
        return batches

    def get_output_proteins(self):
        return self.network.get_output_proteins()

    def get_input_from_output(self, output_arr):
        return self.network.get_input_from_output(output_arr)

    def get_inverted_input_proteins(self):
        return self.network.get_inverted_input_proteins()

    def get_inverted_input_positions(self):
        return self.network.get_inverted_input_positions()

    def get_quantized_parameters_per_node_type(self, params):
        """Returns a dictionary with node types as keys and a list of quantized parameters used by that node type as values.
        When a parameter is quantized, the key to each value is the name of the quantized value (i.e '1x_uORF').
        When a parameter is not quantized, we record all of the values for this param in use in the network, and the key
        is the id of the node that uses this parameter with the specific given value.
        """
        node_dict = self.network.get_compute_types()
        params_per_type = {k: dict() for k in node_dict.keys()}
        pnode = params['node']
        if self.node_namespace is not None:
            pnode = params['node'][self.node_namespace]
        for node_type, node_ids in node_dict.items():
            for node_id in node_ids:
                if node_id in pnode.keys():
                    for pname, v in pnode[node_id].items():
                        if pname in params_per_type[node_type].keys():
                            params_per_type[node_type][pname].update({node_id: v})
                        else:
                            params_per_type[node_type][pname] = {node_id: v}
        res = {}
        shared = params['shared']
        for ntype, pdict in params_per_type.items():  # pdict is dict(str,dict(int,np.ndarray))
            res[ntype] = {}
            # check if we have some quantization values for the parameters
            # a quantization value has a key of the form vname::param_name
            # we want to collect all the quantization values for the parameters of this node type
            for pname in pdict:
                matching = {k: v for k, v in shared.items() if k.endswith(f'::{pname}')}
                if len(matching) > 0:
                    res[ntype][pname] = matching
                else:
                    res[ntype][pname] = params_per_type[ntype][pname]
        res
        return res


#                                                                            }}}
