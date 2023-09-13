from .library import PartsLibrary as PartsLibrary
import jax
import inspect
import json
import importlib
from jax import vmap, jit, grad
from jax.tree_util import Partial as partial
import jax.numpy as jnp
# import jax.numpy as n
import numpy as np
from . import utils as ut

from .utils import check

from tqdm import tqdm


### {{{                  --     parameter manipulations     --

def indirect_param_at(
    params,
    name,
    node_id=0,
    base_path=ut.NODE_PATH,
    init=None,
    overwrite_with=None,
    read_only=True,
    number_of_nodes_at_least=1,
    **_,
):
    """
    Retrieves or sets a parameter from the given params dictionary.
    Vectorizable across the node_id axis.
    If the parameter is not found, it is created and added to the params dict. (unless read_only is True)
    - params: the dictionary of parameters
    - name: the name of the parameter
    - node_id: the id of the node that owns this parameter
    - base_path: the path to the node in the params dict, which acts as a namespace ("node", "shared", "static", ...)
    - init: the initialization function to use if the parameter is not found
    - overwrite_with: if not None, the parameter will be overwritten with this value wether it exists or not
    - read_only: if True, the parameter will not be created if it is not found (and not overwritten)
    """
    # We can't jit/vectorize a dictionnary lookup. i.e we can't do:
    # res = params[node_id] as this requires branching
    # Indexing an array is fine though, so we could simply create
    # an array of params for each node that is as big as the largest
    # node_id, and then index it with the node_id, which is exactly what we do in
    # direct_param_at.
    # However, this is wasteful for params that have large shapes
    # but are only used by a few nodes (most params are like this).

    # So instead I add one layer of indirection to have a sparse array of params:
    # we save a key_vec which will contain -1 for all nodes that don't use
    # the given parameter, and an actual parameter_id for the nodes that do.
    # This way we can use the key_vec to index a parameter array that contains
    # only the parameters that are actually used by the network.

    # I think in theory we can also use node_id with base_path = shared
    # to vectorize tl vs tx by accessing different weights!

    assert isinstance(params, dict), f'params must be a dict, not {type(params)}'

    dpath = base_path / name

    nparams = ut.at_path(params, dpath, None)
    nparams = nparams.shape[0] if nparams is not None else 0

    keys_path = ut.KEYS_PATH + dpath
    key_vec = ut.at_path(params, keys_path, None)  # key_vec is an integer vector (n_nodes,)

    if not read_only:  # non-jittable path (only used for initialization)
        N_NODES = max(node_id, number_of_nodes_at_least - 1) + 1
        if key_vec is None or key_vec.shape[0] <= N_NODES:
            # extend key_vec to fit node_id
            v = key_vec if key_vec is not None else jnp.zeros((0,), dtype=jnp.int32)
            key_vec = jnp.concatenate(
                [v, jnp.full((N_NODES - v.shape[0] + 1,), -1, dtype=jnp.int32)]
            )

        if int(key_vec[node_id]) == -1:  # param doesn't exist yet
            try:
                new_param_value = overwrite_with if overwrite_with is not None else init()
                p = ut.at_path(params, dpath)  # get existing parameter array
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

    res = ut.at_path(params, dpath)[param_id]

    return res


def direct_param_at(
    params,
    name,
    node_id=0,
    base_path=ut.NODE_PATH,
    init=None,
    overwrite_with=None,
    read_only=True,
    number_of_nodes_at_least=1,
    **_,
):

    """
    Similar to indirect_param_at, but doesn't use key_vec: it's a dense param array
    instead of a sparse one. Potentially VERY wasteful, but faster to access.
    """

    if not isinstance(params, dict):
        raise TypeError(f'params must be a dict, not {type(params)}')

    dpath = base_path / name
    p_array = ut.at_path(params, dpath, None)  # p_array is the parameter array (n_params, *shape)

    if not read_only:  # non-jittable path (only used for initialization)
        # first we will check if the param is already initialized
        IS_INIT_PATH = ut.STATIC_PATH / 'is_init'
        # we store a boolean indicating if a param is initialized
        is_init_array = ut.at_path(params, IS_INIT_PATH / dpath, None)
        if is_init_array is None or is_init_array.shape[0] <= node_id:
            # extend is_init_array to fit node_id
            v = is_init_array if is_init_array is not None else np.zeros((0,), dtype=np.bool_)
            is_init_array = np.concatenate(
                [v, np.full((node_id - v.shape[0] + 1,), False, dtype=np.bool_)]
            )
            ut.at_path(params, IS_INIT_PATH / dpath, is_init_array)
        param_is_init = is_init_array[node_id]

        if not param_is_init or overwrite_with is not None:
            new_value = overwrite_with if overwrite_with is not None else init()
            if p_array is not None and p_array.shape[1:] != new_value.shape:
                raise ValueError(
                    f'Param "{name}" has shape {p_array.shape[1:]}, but '
                    f'new value has shape {new_value.shape}.'
                )
            # then let's make sure the param array is big enough
            REQUIRED_LENGTH = max(node_id, number_of_nodes_at_least - 1) + 1
            if p_array is None:
                p_array = np.zeros((REQUIRED_LENGTH,) + new_value.shape, dtype=new_value.dtype)
            elif p_array.shape[0] < REQUIRED_LENGTH:
                p_array = np.concatenate(
                    [p_array, np.zeros((REQUIRED_LENGTH - p_array.shape[0],) + new_value.shape)]
                ).astype(new_value.dtype)

            # finally we can set the param
            p_array[node_id] = new_value
            p_array = ut.at_path(params, dpath, p_array)
            p = p_array[node_id]
            # and mark the param as initialized
            is_init_array[node_id] = True
            ut.at_path(params, IS_INIT_PATH / dpath, is_init_array)

    dtype = p_array.dtype
    p = p_array[node_id].astype(dtype)
    return p


PARAM_AT = direct_param_at


def set_param(params, name, value, node_id=0, base_path=ut.NODE_PATH, **kw):
    return PARAM_AT(
        params, name, node_id, base_path, overwrite_with=np.asarray(value), read_only=False, **kw
    )

def get_param(params, name, node_id=0, base_path=ut.NODE_PATH, **_):
    return PARAM_AT(params, name, node_id, base_path, read_only=True)

def init_param_if_needed(params, name, init, node_id=0, base_path=ut.NODE_PATH, **kw):
    return PARAM_AT(params, name, node_id, base_path, init=init, read_only=False, **kw)

##────────────────────────────────────────────────────────────────────────────}}}# ------------ quantization

### {{{                       --     quantization     --
def quantize(x, possible_values):
    if len(possible_values) == 0:
        return x
    if len(possible_values) == 1:
        return possible_values[0]
    else:
        return quantize_impl(x, possible_values)


def quantize_masked(x, possible_values, mask):
    if len(possible_values) == 0:
        return x
    if len(possible_values) == 1:
        return possible_values[0]
    else:
        return quantize_masked_impl(x, possible_values, mask)


def quantize_impl(x, arr):
    zero = x - jax.lax.stop_gradient(x)  # for straight-through gradient
    return zero + jax.lax.stop_gradient(arr[jnp.argmin(jnp.abs(arr - x))])


@jit
def quantize_masked_impl(x, qvalues, mask):
    """Quantize x to the nearest element in qvalues, but only if the corresponding
    element in mask is True.
    """
    zero = x - jax.lax.stop_gradient(x)  # for straight-through gradient
    dist = jnp.where(mask, jnp.abs(qvalues - x), jnp.inf)
    amin = jnp.argmin(dist)
    res = zero + jax.lax.stop_gradient(qvalues[amin])
    return res


@jax.custom_jvp
def round_to_int(x):
    zero = x - jax.lax.stop_gradient(x)  # for straight-through gradient
    return zero + jax.lax.stop_gradient(jnp.round(x))


def get_quantized(
    values_to_quantize,
    node_id,
    params, # params dict
    param_name, # i.e "tl_rate", "tc_rate", ...
):
    """Quantize the given values using the quantization values stored in params."""
    # initialization of both keys and values is done upstream. We assume both are already initialized
    # i.e there is a param called param_name in params, which is a vector (n_qvalues, ...)
    # of all the possible quantization values for this parameter.
    possible_values = get_param(params, param_name, base_path=ut.QVALS_PATH)
    possible_values = jnp.atleast_1d(possible_values.squeeze())
    # possible_values is a 1D array of shape (n_qvalues,) that contains all the possible
    # quantization values for this parameter. Possible values for the whole stack,
    # but then will be masked to only use the ones available for this node
    masks = get_param(params, param_name, node_id=node_id, base_path=ut.MASK_PATH)
    assert len(values_to_quantize) <= len(masks), (
        f'Number of inputs ({len(values_to_quantize)}) is larger than the number of masks '
        f'({len(masks)}) for node {node_id} and parameter {param_name}.'
    )
    assert len(possible_values) == len(masks[0]), (
        f'Number of possible values ({len(possible_values)}) is different from the number of '
        f'masks ({len(masks[0])}) for node {node_id} and parameter {param_name}.'
    )
    # masks is a 2D array of shape (max_n_masks_per_node, n_qvalues) that tells us which
    # quantization values are allowed for this node.
    # max_n_masks_per_node is the maximum number of quantization values that can be used for
    # this node. Remember that a node can have several inputs, coming from different nodes,
    # and each input can have a different set of possible quantization values.
    masks = masks[
        : values_to_quantize.shape[0]
    ]  # trim masks to the specific number of inputs of this node
    return vmap(quantize_masked, in_axes=(0, None, 0), out_axes=0)(
        values_to_quantize, possible_values, masks
    )


def get_all_possible_quantization_params(network) -> dict[str, list[str]]:
    # returns a dictionary of all possible parameters
    # they can be found at each row of the central_dogma_graph, in the params column
    # which is a dict[str, list[str]] itself. We just want the exhaustive list of keys
    # and all possible values for each key
    # example: {'tl_rate': ['1xuORF', '2xuORF']}
    all_params = {}
    for _, row in network.central_dogma_graph.iterrows():
        for k, v in row.params.items():
            if k not in all_params:
                all_params[k] = set()
            all_params[k].update(v)
    return {k: list(v) for k, v in all_params.items()}


def get_available_quantizations(param_name, cdg_node_id, cdg):
    """
    returns the name of possible parts for a given cdg node, slot and param name
    example: get_possible_values('transcription_rate', ...) -> ['hEF1a', 'hEF1b', 'hEF1c']
              get_possible_values('translation_rate', ...) -> [None, '1xuORF', '2xuORF', ...]
    params are stored in the params column of the cdg as a dict {param_name:[possiblevaluees]}
    """
    available_params = cdg.loc[cdg_node_id, 'params']
    if param_name not in available_params:
        raise ValueError(
            f'Param {param_name} not available for cdg node {cdg_node_id}. Available: {available_params}'
        )
    return available_params[param_name]


def generate_quantization_masks(
    qnames, params, pname, vnode, maximum_required_masks_per_node=4, **kwargs
):
    """
    generate the quantization masks for a given vnode and parameter. One mask per input.
    - qnames: the list of quantization names for this parameter (e.g. ['hEF1a', ...])
    - params: the parameters dictionnary, where only arrays can be used (because it'll be jitted)
    - pname: the name of the parameter we want to quantize (e.g. 'tl_rate')
    - vnode: the node we want to quantize
    - maximum_required_masks_per_node: basically the max number of inputs a node can have
    """
    # example: generate_quantization_masks(['hEF1a', 'hEF1b', 'hEF1c'], params, 'tc_rate', vnode)
    compute_node_id = vnode.compute_node_id
    stack_node_id = vnode.node_id
    network = vnode.network
    if network is None:
        # pure virtual node, no network, no masks!
        ut.logger.warning(f'Node {vnode.node_id} has no network, no quantization mask generated')
        mask = np.ones((1, len(qnames)), dtype=bool)
        set_param(params, pname, mask, node_id=stack_node_id, base_path=ut.MASK_PATH, **kwargs)
        return

    cdf = network.compute_graph
    cdg = network.central_dogma_graph

    cdg_ids = cdf.loc[compute_node_id]['cdg_input']
    assert cdg_ids is not None, f'Node {compute_node_id} has no input CDG node'
    cdg_ids = [cdg_ids] if not isinstance(cdg_ids, list) else cdg_ids

    this_node_qnames = [get_available_quantizations(pname, cid, cdg) for cid in cdg_ids]
    # we have one mask per CDG input, and we need the same mask shape for all nodes
    assert len(this_node_qnames) <= maximum_required_masks_per_node, (
        f'Node {compute_node_id} has {len(this_node_qnames)} CDG inputs, '
        f'but only a max of {maximum_required_masks_per_node} masks are available'
    )
    # check that this_node_qnames is a subset of qnames
    should_be_in = [qq for q in this_node_qnames for qq in q if qq not in qnames]
    if len(should_be_in) > 0:
        raise ValueError(
            f'Node {compute_node_id} has unknown quantization names {should_be_in} '
            f'for parameter {pname} (available: {qnames})'
        )

    # now create the mask array
    mask = np.zeros((maximum_required_masks_per_node, len(qnames)), dtype=bool)
    for i in range(len(this_node_qnames)):
        mask[i, [qnames.index(q) for q in this_node_qnames[i]]] = True

    # now we store the mask in the params dict, under the mask namespace,
    set_param(params, pname, mask, node_id=stack_node_id, base_path=ut.MASK_PATH, **kwargs)

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                 --     quantile variable helpers     --
def register_quantile_variable_ids(params, vnode, stack):
    # a node may use more than one quantile variable so we'll just pad with -1 grapb qids:
    comp_node = vnode.get_compute_node()
    if comp_node is not None:
        assert 'quantile_variable_id' in comp_node.extra
        qid = np.array(comp_node.extra['quantile_variable_id']).astype(int)
        # turn qids from network to stack scope
    else:
        # it's a purely virtual node, we're probably just in the tracer or analyzer...
        qid = np.zeros(1, dtype=int)

    if stack is not None:
        max_qsize = stack.max_nb_of_outputs_per_network
        n_nodes = stack.number_of_nodes
        qid_stack = np.array(
            [stack.get_network_global_output_id(vnode.network_id, q) for q in qid]
        ).astype(int)
        assert qid_stack.ndim == 1
    else:
        max_qsize = 1
        n_nodes = 1
        qid_stack = qid

    assert qid_stack.shape[0] <= max_qsize
    qid_stack = np.pad(qid_stack, (0, max_qsize - qid_stack.shape[0]), constant_values=-1)
    set_param(
        params,
        'quantile_variable_id',
        qid_stack,
        node_id=vnode.node_id,
        base_path=ut.STATIC_PATH,
        number_of_nodes_at_least=n_nodes,
    )


def get_quantile_variables(params, node_id, quantiles, n):
    qid = get_param(
        params, 'quantile_variable_id', node_id=node_id, base_path=ut.STATIC_PATH
    ).astype(int)
    assert qid.ndim == 1
    q = jnp.where(qid == -1, 0, quantiles[qid])[:n]

    return q.squeeze()

##────────────────────────────────────────────────────────────────────────────}}}
