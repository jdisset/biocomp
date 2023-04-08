from .library import PartsLibrary as PartsLibrary
import jax
import inspect
import json
import importlib
from jax import vmap, jit, grad
from jax.tree_util import Partial as partial
import jax.numpy as jnp
import numpy as np
from . import utils as ut

from tqdm import tqdm

### {{{                  --     params and quantization     --


# general param function (for initializing, setting or getting)
def param_at(
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
    # node_id, and then index it with the node_id. However, this would be wasteful
    # for params that have large shapes but are only used by a few nodes.

    # So instead I add one layer of indirection:
    # we save a key_vec which will contain -1 for all nodes that don't use
    # the given parameter, and an actual parameter_id for the nodes that do.
    # This way we can use the key_vec to index a parameter array that contains
    # only the parameters that are actually used by the network.

    # I think in theory we can also use node_id with base_path = shared
    # to vectorize tl vs tx by accessing different weights!

    node_id = jnp.asarray(node_id).astype(jnp.int32)

    assert isinstance(params, dict), f'params must be a dict, not {type(params)}'

    dpath = base_path + [name]

    nparams = ut.at_path(params, dpath, None)
    nparams = nparams.shape[0] if nparams is not None else 0

    keys_path = ut.KEYS_PATH + dpath
    key_vec = ut.at_path(params, keys_path, None)  # key_vec is an integer vector (n_nodes,)

    if not read_only:  # non-jittable path (only used for initialization)
        N_NODES = max(node_id, number_of_nodes_at_least - 1) + 1
        if key_vec is None or key_vec.shape[0] <= N_NODES:
            # extend key_vec to fit node_id
            v = key_vec if key_vec is not None else jnp.zeros((0,), dtype=jnp.float32)
            key_vec = jnp.concatenate(
                [v, jnp.full((N_NODES - v.shape[0] + 1,), -1, dtype=jnp.float32)]
            )

        if int(key_vec[node_id]) == -1:  # param doesn't exist yet
            try:
                new_param_value = (
                    overwrite_with.astype(jnp.float32)
                    if overwrite_with is not None
                    else init().astype(jnp.float32)
                )
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

    param_id = key_vec[node_id].astype(jnp.int32)

    if overwrite_with is not None and not read_only:  # also non-jittable
        allp = ut.at_path(params, dpath).at[param_id].set(overwrite_with.astype(jnp.float32))
        ut.at_path(params, dpath, allp)

    res = ut.at_path(params, dpath)[param_id]

    # if param_is is not valid, it's -1, and jax just returns the first element[:10]
    # however I want to return nans instead so I can at least see that something is wrong.
    # it won't work if the param is not a float, but that's better than nothing
    # using lax cond (not where but lax.cond directly) is faster than using jnp.where:

    res = jnp.where(param_id == -1, jnp.full_like(res, np.nan), res)

    return res


def set_param(params, name, value, node_id=0, base_path=ut.NODE_PATH, **_):
    return param_at(params, name, node_id, base_path, overwrite_with=jnp.asarray(value), read_only=False)


def get_param(params, name, node_id=0, base_path=ut.NODE_PATH, **_):
    return param_at(params, name, node_id, base_path)


def init_param_if_needed(params, name, init, node_id=0, base_path=ut.NODE_PATH, **_):
    return param_at(params, name, node_id, base_path, init=init, read_only=False)


def save_to_params(all_params, node_id, node_params):
    for param_name, param_value in node_params.items():
        # Retrieve the current param value for this node, if it exists
        current_param_value = get_param(
            all_params,
            param_name,
            node_id,
        )

        if current_param_value is None or np.any(current_param_value.shape != param_value.shape):
            # Resize all existing params if the new param_value has a different shape
            existing_params = all_params[param_name]
            max_shape = tuple(np.maximum(existing_params.shape[1:], param_value.shape))
            resized_params = []

            for i in range(existing_params.shape[0]):
                resized_param = np.full(max_shape, np.nan)
                resized_param[: existing_params[i].shape[0], ...] = existing_params[i]
                resized_params.append(resized_param)

            all_params[param_name] = np.stack(resized_params, axis=0)

        # Save the new param_value for the given node_id
        set_param(
            all_params,
            param_name,
            param_value,
            node_id,
        )


# ------------ quantization
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


def quantize_masked_impl(x, arr, mask):
    zero = x - jax.lax.stop_gradient(x)  # for straight-through gradient
    dist = jnp.where(mask, jnp.abs(arr - x), jnp.inf)
    return zero + jax.lax.stop_gradient(arr[jnp.argmin(dist)])


@jax.custom_jvp
def round_to_int(x):
    zero = x - jax.lax.stop_gradient(x)  # for straight-through gradient
    return zero + jax.lax.stop_gradient(jnp.round(x))


def get_quantized(
    values_to_quantize,
    node_id,
    params,
    param_name,
):
    """Quantize the given values using the quantization values stored in params."""
    # initialization of both keys and values is done upstream. We assume both are already initialized
    # i.e there is a param called param_name in params, which is a vector (n_qvalues, ...)
    # of all the possible quantization values for this parameter.

    possible_values = get_param(params, param_name, base_path=ut.QVALS_PATH)
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


def set_quantization_values(params, pname, qnames, qvals):
    """Initialize all the available quantization values for a given parameter."""
    qnames = sorted(qnames)
    assert len(qnames) == len(
        set(qnames)
    ), f'quantization names for {pname} must be unique, got {qnames}'
    qname_path = ut.QNAME_PATH + [pname]
    already = ut.at_path(params, qname_path)
    if already is None:
        assert len(qvals) == len(qnames)
        ut.at_path(params, qname_path, qnames)
        set_param(params, pname, qvals, base_path=ut.QVALS_PATH)
    else:
        assert (
            qnames == already
        ), f'qnames for {pname} already initialized to {already}, cannot change to {qnames}'


def get_all_possible_quantization_params(network) -> dict[str, list[str]]:
    # returns a dictionary of all possible parameters
    # they can be found at each row of the central_dogma_graph, in the params column
    # which is a dict[str, list[str]] itself. We just want the exhaustive list of keys
    # and all possible values for each key
    all_params = {}
    for _, row in network.central_dogma_graph.iterrows():
        for k, v in row.params.items():
            if k not in all_params:
                all_params[k] = set()
            all_params[k].update(v)
    return {k: list(v) for k, v in all_params.items()}


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


def generate_quantization_masks(
    qnames, params, pname, vnode, maximum_required_masks_per_node=4, **kwargs
):
    """
    generate the quantization masks for a given node and parameter. One mask per input.
    - qnames: the ordered list of quantization names for this parameter
    - params: the parameters dictionnary, where only arrays can be used (because it'll be jitted)
    - pname: the name of the parameter we want to quantize (e.g. 'tl_rate')
    - vnode: the node we want to quantize
    - maximum_required_masks_per_node: basically the max number of inputs a node can have
    """

    network = vnode.network
    cdf = network.compute_graph
    cdg = network.central_dogma_graph
    compute_node_id = vnode.compute_node_id
    stack_node_id = vnode.node_id

    cdg_ids = cdf.loc[compute_node_id]['cdg_input']
    assert cdg_ids is not None, f'Node {compute_node_id} has no input CDG node'
    cdg_ids = [cdg_ids] if not isinstance(cdg_ids, list) else cdg_ids

    this_node_qnames = [get_available_quantizations(pname, cid, cdg) for cid in cdg_ids]
    # we have one mask per CDG input, and we need the same mask shape for all nodes
    assert len(this_node_qnames) <= maximum_required_masks_per_node, (
        f'Node {compute_node_id} has {len(this_node_qnames)} CDG inputs, '
        f'but only a max of {maximum_required_masks_per_node} masks are available'
    )

    # now create the mask array
    mask = np.zeros((maximum_required_masks_per_node, len(qnames)), dtype=bool)
    for i in range(len(this_node_qnames)):
        mask[i, [qnames.index(q) for q in this_node_qnames[i]]] = True

    # now we store the mask in the params dict, under the mask namespace,
    set_param(params, pname, mask, node_id=stack_node_id, base_path=ut.MASK_PATH, **kwargs)


def register_quantile_variable_ids(params, vnode, stack):
    # problem is that a node may use more than one quantile variable so we'll just pad with -1
    # grapb qids:
    comp_node = vnode.get_compute_node()
    assert 'quantile_variable_id' in comp_node.extra
    qid = np.array(comp_node.extra['quantile_variable_id']).astype(int)
    # turn qids from network to stack scope
    qid_stack = np.array(
        [stack.get_network_global_output_id(vnode.network_id, q) for q in qid]
    ).astype(int)
    assert qid_stack.ndim == 1
    max_qsize = stack.max_nb_of_outputs_per_network
    assert qid_stack.shape[0] <= max_qsize
    qid_stack = np.pad(qid_stack, (0, max_qsize - qid_stack.shape[0]), constant_values=-1)
    set_param(params, 'quantile_variable_id', qid_stack, node_id=vnode.node_id)


def get_quantile_variables(params, node_id, quantiles, n):
    qid = get_param(params, 'quantile_variable_id', node_id=node_id).astype(int)
    assert qid.ndim == 1
    q = jnp.where(qid == -1, 0, quantiles[qid])[:n]

    return q.squeeze()


##────────────────────────────────────────────────────────────────────────────}}}


### {{{                   --     general compute implementations    --

# When we create a compute layer, we pass the shape of all inputs as a list of tuples.
# Indeed, a node can have several inputs, and each input can have a different shape.
# The node constructor must then return the apply function, and the shape of the outputs.
# There can also be multiple outputs, each of which can have a different shape.
# we also pass the numper of outputs, which is useful for the source node.

# one question is whether we shouls allow for multiple outputs with different shapes.
# I don't think it's necessary for now, but at the same time it's not a big deal to allow it.
# So I guess yes, we should allow it. that means that we output a tuple of arrays


# Signatures:
# prepare (params, vnode, key)
# apply (values, quantiles, params, node_id, key)


def empty_prepare(*_, **__):
    pass


# input_shapes is a list of shape tuples, one for each input
def single_passthrough(input_shapes, *_, **__):

    assert len(input_shapes) == 1, f'Passthrough expects 1 input, got {len(input_shapes)}'

    def apply(value, **___):
        return value

    output_shapes = input_shapes

    return empty_prepare, apply, output_shapes


# source node is just an L2 plasmid, i.e an aggregation that has a fixed ratio of 1:1
# we make it a multi-output node so that it's compatible with the aggregation node but
# really we're just duplicating the input so we could also just use a passthrough node
# or skip the node altogether (for a future version with an optimizer)
def source(input_shapes, n_outputs, **_):
    assert len(input_shapes) == 1, f'A source node should have 1 input, got {len(input_shapes)}'

    def apply(value, *_, **__):
        return jnp.repeat(value, n_outputs, axis=0)

    output_shapes = input_shapes * n_outputs

    return empty_prepare, apply, output_shapes


# inverse of source is just a passthrough, as it's only inverted when only one output and one input
def inv_source(*args, **kwargs):
    return single_passthrough(*args, **kwargs)


def numeric(input_shapes, shape, **__):

    assert len(input_shapes) == 0

    def prepare(params, vnodelist, key, **_):
        for vnode in vnodelist:
            init_val = jax.random.uniform(key, shape=shape, minval=0.0, maxval=1.0)
            set_param(params, "numeric:value", init_val, node_id=vnode.node_id)

    def apply(v, q, params, node_id, k):
        return get_param(params, "numeric:value", node_id=node_id)

    output_shapes = [shape]

    return prepare, apply, output_shapes


# inverse of numeric is just a pass-through
def inv_numeric(*args, **kwargs):
    return single_passthrough(*args, **kwargs)


def aggregation(input_shapes, n_outputs, normalize=False, **_):

    assert len(input_shapes) == 1, f'Aggregation expects 1 input, got {len(input_shapes)}'

    def prepare(params, vnodelist, key, **_):
        for vnode in vnodelist:
            extra = vnode.get_compute_node().extra
            if 'ratios' in extra:
                ratio_v = jnp.array(extra['ratios'], dtype=jnp.float32)
            else:
                ratio_v = jax.random.uniform(key, (n_outputs,))
            assert ratio_v.shape == (n_outputs,), f'Invalid ratio shape {ratio_v.shape}'
            set_param(params, "aggregation:ratios", ratio_v, node_id=vnode.node_id)
            ratios = get_param(params, "aggregation:ratios", node_id=vnode.node_id)
            ratios = get_param(params, "aggregation:ratios", node_id=141)

    def apply(input, quantiles, params, node_id, key):
        assert input.shape == input_shapes[0], f'Invalid input shape {input.shape}'
        ratios = get_param(params, "aggregation:ratios", node_id)
        if normalize:
            ratios = ratios / jnp.maximum(jnp.sum(ratios), 1e-12)
        return jnp.array(ratios) * input

    output_shape = input_shapes * n_outputs

    return prepare, apply, output_shape


def inv_aggregation(input_shapes, n_outputs, stack, normalize=False, **_):

    # an inverse aggregation node always has 1 input and 1 output
    assert len(input_shapes) == 1, f'inverse_Aggregation expects 1 input, got {len(input_shapes)}'
    assert n_outputs == 1, f'inverse_Aggregation expects 1 output, got {n_outputs}'

    def prepare(params, vnodelist, **_):
        for vnode in vnodelist:
            inv_vnode = vnode.get_inverse_vnode(stack)
            cnode = vnode.get_compute_node()

            extra = cnode.extra
            assert 'original_output_len' in extra
            assert 'original_output_slot' in extra
            assert extra['original_output_len'] > 0
            assert extra['original_output_slot'] < extra['original_output_len']

            set_param(
                params,
                "inv_aggregation:original_output_slot",
                jnp.asarray(extra['original_output_slot']),
                node_id=vnode.node_id,
            )
            set_param(
                params, "inv_aggregation:inv_node_id", jnp.asarray(inv_vnode.node_id), node_id=vnode.node_id
            )

    def apply(inp, quantiles, params, node_id, key):
        inv_id = get_param(params, "inv_aggregation:inv_node_id", node_id).astype(jnp.int32)
        original_output_slot = get_param(
            params, "inv_aggregation:original_output_slot", node_id
        ).astype(jnp.int32)
        ratios = get_param(params, "aggregation:ratios", inv_id)
        if normalize:
            ratios = ratios / jnp.maximum(jnp.sum(ratios), 1e-12)

        return inp / ratios[original_output_slot]

    output_shape = input_shapes
    return prepare, apply, output_shape


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                    --     neural utils     --


def leaky_relu(x, alpha=0.2):
    return jnp.where(x > 0, x, alpha * x)


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


ACTIVATION_FUNCTIONS = {
    'leaky_relu': leaky_relu,
    'sigmoid': sigmoid,
    'none': lambda x: x,
}

DEFAULT_ACTIVATION = 'leaky_relu'
DEFAULT_OUT_ACTIVATION = 'sigmoid'


def dense_layer(input_values, output_size, param_f, key, name):
    assert len(input_values.shape) == 1, f"In {name}: input_values should be a 1D array."
    input_size = 1 if input_values.shape == () else input_values.shape[0]

    w = param_f(f'{name}_w', init=ut.he_initializer(key, (input_size, output_size)))
    b = param_f(f'{name}_b', init=lambda: jnp.zeros((output_size,)))

    assert input_values.shape == (
        input_size,
    ), f'In {name}: {input_values.shape} != {(input_size,)}'
    assert w.shape == (
        input_size,
        output_size,
    ), f'In {name}: {w.shape} != {(input_size, output_size)}'
    assert b.shape == (output_size,), f'In {name}: {b.shape} != {(output_size,)}'

    assert w.shape == (
        input_size,
        output_size,
    ), f'In {name}: {w.shape} != {(input_size, output_size)}'

    res = jnp.dot(input_values, w) + b
    assert res.shape == (output_size,), f'In {name}: {res.shape} != {(output_size,)}'
    return res


def dense_multilevel(
    input_values,
    hidden_s,
    output_s,
    depth,
    param_f,
    key,
    name,
    activation,
):
    assert len(input_values.shape) == 1, f"In {name}: input_values should be a 1D array."
    assert (
        isinstance(depth, int) and depth >= 1
    ), f"In {name}: depth should be an integer greater than or equal to 1."
    assert (
        isinstance(hidden_s, int) and hidden_s > 0
    ), f"In {name}: hidden_s should be a positive integer."
    assert (
        isinstance(output_s, int) and output_s > 0
    ), f"In {name}: output_s should be a positive integer."

    res = input_values
    keys = jax.random.split(key, depth)
    for i in range(depth - 1):
        res = activation(dense_layer(res, hidden_s, param_f, keys[i], f'{name}_{i}'))
        assert res.shape == (hidden_s,), f'In {name}: {res.shape} != {(hidden_s,)}'

    res = dense_layer(res, output_s, param_f, keys[-1], f'{name}_{depth - 1}')
    assert res.shape == (output_s,), f'In {name}: {res.shape} != {(output_s,)}'
    return res


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                       --     neural nodes     --


def transform_nn(
    input_shapes,
    n_outputs,
    stack,
    transform_name,
    outer_wsize=64,
    outer_depth=2,
    inner_wsize=32,
    inner_depth=2,
    inner_outsize=4,
    rate_dim=1,
    tr_namespace='',
    inner_activation_name=DEFAULT_ACTIVATION,
    outer_activation_name=DEFAULT_OUT_ACTIVATION,
    **_,
):
    inner_activation = ACTIVATION_FUNCTIONS[inner_activation_name]
    outer_activation = ACTIVATION_FUNCTIONS[outer_activation_name]

    assert n_outputs == 1, f'NN transform only supports 1 output, got {n_outputs}'
    rate_name = f'{transform_name}_rate'

    # we separate between node_impl and shared_impl for performance during prepare
    # only node_impl has to be called for each node_id

    def __node_impl(*values, key, param_f, params, node_id):
        val = jnp.array(values)
        rshape = (val.shape[0], rate_dim)
        # first grab the continuous values for the rates
        rates = param_f(rate_name, init=ut.continuous_initializer(key, rshape), node_id=node_id)
        # then quantize them
        rates = get_quantized(rates, node_id=node_id, params=params, param_name=rate_name)
        return val, rates

    def __shared_impl(val, rates, quantile, key, param_f):

        assert val.shape[0] == rates.shape[0]
        k1, k2 = jax.random.split(key, 2)

        def inner(value, rate_embeding, key):
            """For a single source, computes a latent output from the concatenation of
            the rate embedding and the source value.
            All of these outputs will then be summed up and passed through a final layer.
            """
            # TODO idea: to give more flexibility, we could add the index of the
            # value as this might allow clever padding of the sum
            # we'd then need to make sure that the index is unique for each
            # while, probably, being random (to avoid any "preferred" order)

            if value.ndim == 0:
                value = value.reshape((1,))
            if rate_embeding.ndim == 0:
                rate_embeding = rate_embeding.reshape((1,))

            assert value.ndim == 1, f'In {transform_name}: {value.ndim} != 1: {value}'
            assert rate_embeding.ndim == 1

            inputs = ut.flat_concat(value, rate_embeding, quantile)

            out = inner_activation(
                dense_multilevel(
                    inputs,
                    inner_wsize,
                    inner_outsize,
                    depth=inner_depth,
                    param_f=partial(param_f, node_id=0, base_path=ut.SHARED_PATH),
                    key=key,
                    name=f'{tr_namespace}{transform_name}_inner',
                    activation=inner_activation,
                )
            )

            assert out.shape == (inner_outsize,)

            return out

        # first we apply the inner stack to all inputs and sum them:

        inner_keys = jax.random.split(k1, val.shape[0])
        inner_out = sum(inner(v, r, k) for v, r, k in zip(val, rates, inner_keys))
        inner_out = ut.flat_concat(inner_out, quantile)

        assert inner_out.shape == (inner_outsize + 1,)

        # then we apply a final outer layer to the summed output:
        return outer_activation(
            dense_multilevel(
                inner_out,
                outer_wsize,
                1,
                depth=outer_depth,
                param_f=partial(param_f, node_id=0, base_path=ut.SHARED_PATH),
                key=k2,
                name=f'{tr_namespace}{transform_name}_outer',
                activation=inner_activation,
            )
        )

    def prepare(params, vnodelist, key):
        # during prepare, we call _impl with dummy inputs + a param_function that
        # creates the parameters on the fly if they don't exist yet

        qnames = ut.at_path(stack.shared_store, ut.QNAME_PATH + [rate_name])
        assert qnames is not None, f'quantization names for {rate_name} not initialized'

        init = ut.continuous_initializer(key, (len(qnames), rate_dim))
        init_param_if_needed(params, rate_name, init=init, base_path=ut.QVALS_PATH, node_id=0)

        for vnode in vnodelist:
            register_quantile_variable_ids(params, vnode, stack)
            generate_quantization_masks(
                qnames, params, rate_name, vnode, number_of_nodes_at_least=stack.number_of_nodes
            )
            key, _ = jax.random.split(key)
            val, rates = __node_impl(
                *[np.zeros(shape) for shape in input_shapes],
                key=key,
                param_f=partial(
                    init_param_if_needed, params, number_of_nodes_at_least=stack.number_of_nodes
                ),
                params=params,
                node_id=vnode.node_id,
            )

        __shared_impl(
            val,
            rates,
            quantile=0,
            key=key,
            param_f=partial(
                init_param_if_needed, params, number_of_nodes_at_least=stack.number_of_nodes
            ),
        )

    def apply(*values, quantiles, params, node_id, key):
        assert len(values) == len(input_shapes)
        param_f = partial(get_param, params)  # read-only
        val, rates = __node_impl(*values, key=key, param_f=param_f, params=params, node_id=node_id)
        quantile = get_quantile_variables(params, node_id, quantiles, 1)
        return __shared_impl(val, rates, quantile, key, param_f)

    output_shape = [(1,)]

    return prepare, apply, output_shape


def property_id(prop_dict, prop_name, prop_value):
    """Returns the id of the property value in the dict, or creates it if it doesn't exist"""
    pvals = ut.at_path(prop_dict, ut.PROPERTIES_PATH + [prop_name], defaultinit=lambda: [])
    if prop_value not in pvals:
        pvals.append(prop_value)
    return pvals.index(prop_value)


def sequestron_ERN(
    input_shapes,
    n_outputs,
    stack,
    affinity_dim=1,
    wsize=128,
    depth=4,
    out_dim=1,
    subtype='5p',
    inner_activation_name=DEFAULT_ACTIVATION,
    outer_activation_name=DEFAULT_OUT_ACTIVATION,
    **_,
):

    inner_activation = ACTIVATION_FUNCTIONS[inner_activation_name]
    outer_activation = ACTIVATION_FUNCTIONS[outer_activation_name]

    # ERN have 2 inputs of same size
    assert len(input_shapes) == 2
    assert (
        input_shapes[0] == input_shapes[1]
    ), f'ERN inputs must have same shape, got {input_shapes}'
    assert n_outputs == 1, f'ERN only supports 1 output, got {n_outputs}'

    ERN_AFFINITY_ID_NAME = f'ERN{subtype}_affinity_id'
    ERN_AFFINITY_VALUE_PARAM = f'ERN{subtype}_affinity_value'

    def __impl(neg, pos, quantile, rng_key, param_f, affinity_id):
        affinity = param_f(
            ERN_AFFINITY_VALUE_PARAM,
            init=ut.continuous_initializer(rng_key, (affinity_dim,)),
            base_path=ut.SHARED_PATH,
            node_id=affinity_id,
        )

        res = dense_multilevel(
            ut.flat_concat(neg, pos, affinity, quantile),
            wsize,
            out_dim,
            depth,
            param_f=partial(param_f, node_id=0, base_path=ut.SHARED_PATH),
            key=rng_key,
            name=f'ERN_{subtype}',
            activation=inner_activation,
        )

        return outer_activation(jnp.squeeze(res))

    def prepare(params, vnodelist, key):
        for vnode in vnodelist:
            register_quantile_variable_ids(params, vnode, stack)
            # we need to know which affinity value to use for this node
            assert 'seq_name' in vnode.get_compute_node().extra
            seq_name = vnode.get_compute_node().extra['seq_name']  # ex: 'CasE5p'
            prop_name = f'ERN_affinity_{subtype}'
            affinity_id = property_id(stack.shared_store['properties'], prop_name, seq_name)
            # affinity_id is the index at which the affinity value is stored
            # in the array of all affinity values. We cNone an store this index so that
            # we can retrieve the correct value during apply (vectorized on all node_ids)
            set_param(
                params,
                ERN_AFFINITY_ID_NAME,
                affinity_id,
                node_id=vnode.node_id,
                number_of_nodes_at_least=stack.number_of_nodes,
            )
            affinity_id = get_param(params, ERN_AFFINITY_ID_NAME, node_id=vnode.node_id).astype(jnp.int32)

        __impl(
            *[np.zeros(shape) for shape in input_shapes],
            quantile=0,
            rng_key=key,
            param_f=partial(
                init_param_if_needed, params, number_of_nodes_at_least=stack.number_of_nodes
            ),
            affinity_id=affinity_id,
        )

    def apply(*values, quantiles, params, node_id, key):
        assert len(values) == len(input_shapes)
        affinity_id = get_param(params, ERN_AFFINITY_ID_NAME, node_id=node_id)
        quantile = get_quantile_variables(params, node_id, quantiles, 1)
        return __impl(
            *values,
            quantile=quantile,
            rng_key=key,
            param_f=partial(get_param, params),
            affinity_id=affinity_id,
        )

    output_shape = [(1,)]

    return prepare, apply, output_shape


def grouped_output(
    input_shapes,
    n_outputs,
    stack,
    wsize=64,
    depth=3,
    inner_activation_name=DEFAULT_ACTIVATION,
    outer_activation_name=DEFAULT_OUT_ACTIVATION,
    **_,
):

    inner_activation = ACTIVATION_FUNCTIONS[inner_activation_name]
    outer_activation = ACTIVATION_FUNCTIONS[outer_activation_name]

    def __impl(*inputs, quantiles, rng_key, param_f, **_):
        assert quantiles.shape == (len(inputs),)
        res = vmap(
            lambda x: dense_multilevel(
                ut.flat_concat(x, quantiles),
                wsize,
                1,
                depth,
                param_f=partial(param_f, node_id=0, base_path=ut.SHARED_PATH),
                key=rng_key,
                name='grouped_output',
                activation=inner_activation,
            )
        )(jnp.asarray(inputs))
        return outer_activation(res)

    def prepare(params, vnodelist, key):
        for vnode in vnodelist:
            register_quantile_variable_ids(params, vnode, stack)

        __impl(
            *[np.zeros(shape) for shape in input_shapes],
            quantiles=np.zeros((len(input_shapes),)),
            rng_key=key,
            param_f=partial(
                init_param_if_needed, params, number_of_nodes_at_least=stack.number_of_nodes
            ),
        )

    def apply(*inputs, quantiles, params, node_id, key):
        q = get_quantile_variables(params, node_id, quantiles, len(inputs))
        return __impl(*inputs, quantiles=q, rng_key=key, param_f=partial(get_param, params))

    output_shape = [(1,)] * len(input_shapes)

    return prepare, apply, output_shape


transcription = partial(transform_nn, transform_name='tc')
translation = partial(transform_nn, transform_name='tl')
inv_transcription = partial(transform_nn, transform_name='tc', tr_namespace='inv_')
inv_translation = partial(transform_nn, transform_name='tl', tr_namespace='inv_')

ERN5p = partial(sequestron_ERN, subtype='5p')
ERN3p = partial(sequestron_ERN, subtype='3p')


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                      --     config manager     --


def unwrap_partial_function(implementation):
    if hasattr(implementation, 'func') and hasattr(implementation, 'keywords'):
        partial_args = implementation.keywords
        implementation = implementation.func
    else:
        partial_args = {}
    return implementation, partial_args


class ConfigManager:
    def __init__(self, module_name=None):
        self.config = {'functions': {}}
        self.module_name = module_name

    def set(self, key, implementation, **kwargs):
        implementation, partial_args = unwrap_partial_function(implementation)
        kwargs.update(partial_args)

        signature = inspect.signature(implementation)
        parameters = {}
        for name, param in signature.parameters.items():
            if name in kwargs:
                parameters[name] = kwargs[name]
            elif param.default != inspect.Parameter.empty:
                parameters[name] = param.default

        sc = parameters.pop('stack', None)
        assert sc is None, 'stack is a reserved parameter name'

        self.config['functions'][key] = {
            'implementation': implementation.__name__,
            'parameters': parameters,
        }

    def get(self, key):
        assert key in self.config['functions'], f'No function named {key}'
        func_data = self.config['functions'][key]
        if self.module_name:
            module = importlib.import_module(self.module_name)
        else:
            module = importlib.import_module(__name__)
        implementation = getattr(module, func_data['implementation'])
        params = func_data['parameters']
        return partial(implementation, **params)

    def export(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.config, f)

    def load(self, filename):
        with open(filename, 'r') as f:
            self.config = json.load(f)


DEFAULT_NODE_CONFIG = ConfigManager()
DEFAULT_NODE_CONFIG.set('transcription', transcription)
DEFAULT_NODE_CONFIG.set('translation', translation)
DEFAULT_NODE_CONFIG.set('inv_transcription', inv_transcription)
DEFAULT_NODE_CONFIG.set('inv_translation', inv_translation)
DEFAULT_NODE_CONFIG.set('sequestron_ERN', ERN5p)
DEFAULT_NODE_CONFIG.set('sequestron_ERN3p', ERN3p)
DEFAULT_NODE_CONFIG.set('source', source)
DEFAULT_NODE_CONFIG.set('inv_source', inv_source)
DEFAULT_NODE_CONFIG.set('numeric', numeric)
DEFAULT_NODE_CONFIG.set('inv_numeric', inv_numeric)
DEFAULT_NODE_CONFIG.set('aggregation', aggregation)
DEFAULT_NODE_CONFIG.set('inv_aggregation', inv_aggregation)
DEFAULT_NODE_CONFIG.set('output', grouped_output)
DEFAULT_NODE_CONFIG.set('deadend', single_passthrough)


##────────────────────────────────────────────────────────────────────────────}}}
