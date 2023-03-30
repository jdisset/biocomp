from .library import PartsLibrary as PartsLibrary
import jax
from jax import vmap, jit, grad
from jax.tree_util import Partial as partial
import jax.numpy as jnp
import numpy as np
from . import utils as ut


### {{{                  --     params and quantization     --


def get_param(
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

    assert isinstance(params, dict), f'params must be a dict, not {type(params)}'

    dpath = base_path + [name]

    nparams = ut.at_path(params, dpath, None)
    nparams = nparams.shape[0] if nparams is not None else 0

    keys_path = ut.KEYS_PATH + dpath
    key_vec = ut.at_path(params, keys_path, None)  # key_vec is an integer vector (n_nodes,)

    if not read_only:  # non-jittable path (only used for initialization)
        if key_vec is None or node_id >= key_vec.shape[0]:  # key_vec is too small
            # extend key_vec to fit node_id
            v = key_vec if key_vec is not None else jnp.zeros((0,), dtype=jnp.int32)
            key_vec = jnp.concatenate(
                [v, jnp.full((node_id - v.shape[0] + 1,), -1, dtype=jnp.int32)]
            )

        if key_vec[node_id] == -1:  # param doesn't exist yet
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
    # if param_is is not valid, it's -1, and jax just returns the first element
    # however I want to return nans instead so I can at least see that something is wrong.
    # it won't work if the param is not a float, but that's better than nothing
    res = jnp.where(param_id == -1, jnp.full_like(res, np.nan), res)
    return res


def save_to_params(all_params, node_id, node_params):
    for param_name, param_value in node_params.items():
        # Retrieve the current param value for this node, if it exists
        current_param_value = get_param(
            all_params,
            param_name,
            node_id,
            read_only=True,
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
        get_param(
            all_params,
            param_name,
            node_id,
            init=None,
            overwrite_with=param_value,
            read_only=False,
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
    possible_values = get_param(params, param_name, base_path=ut.QVALS_PATH, read_only=True)
    masks = get_param(params, param_name, node_id=node_id, base_path=ut.MASK_PATH, read_only=True)
    assert len(possible_values) <= len(masks)

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



def initialize_quantization_values(params, pname, qnames, init):
    """Initialize all the available quantization values for a given parameter."""
    qnames = sorted(qnames)
    assert len(qnames) == len(
        set(qnames)
    ), f'quantization names for {pname} must be unique, got {qnames}'
    qname_path = ut.QNAME_PATH + [pname]
    already = ut.at_path(params, qname_path)
    if already is None:
        ut.at_path(params, qname_path, qnames)
        qvals = get_param(params, pname, base_path=ut.QVALS_PATH, init=init, read_only=False)
        assert len(qvals) == len(qnames)
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


def generate_quantization_masks(params, pname, node_id, network, maximum_required_masks_per_node):
    """generate the quantization masks for a given node and parameter. One mask per input."""
    cdf = network.compute_graph
    cdg = network.central_dogma_graph

    qnames = ut.at_path(params, ut.QNAME_PATH + [pname])
    assert qnames is not None, f'quantization names for {pname} not initialized'

    cdg_ids = cdf.loc[node_id]['cdg_input']
    assert cdg_ids is not None, f'Node {node_id} has no input CDG node'
    cdg_ids = [cdg_ids] if not isinstance(cdg_ids, list) else cdg_ids

    this_node_qnames = [get_available_quantizations(pname, cid, cdg) for cid in cdg_ids]
    # we have one mask per CDG input, and we need the same mask shape for all nodes
    assert len(this_node_qnames) <= maximum_required_masks_per_node, (
        f'Node {node_id} has {len(this_node_qnames)} CDG inputs, '
        f'but only a max of {maximum_required_masks_per_node} masks are available'
    )

    # now create the mask array
    mask = np.zeros((maximum_required_masks_per_node, len(qnames)), dtype=bool)
    for i in range(len(this_node_qnames)):
        mask[i, [qnames.index(q) for q in this_node_qnames[i]]] = True

    # now we store the mask in the params dict, under the mask namespace,
    get_param(
        params, pname, node_id=node_id, base_path=ut.MASK_PATH, overwrite_with=mask, read_only=False
    )


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                   --     general simple nodes     --

DEFAULT_COMPUTE_NODES_DICT = {}


def compnode(f):
    DEFAULT_COMPUTE_NODES_DICT[f.__name__] = f
    return f


@compnode
def deadend(*_, **__):
    def apply(value, *_, **__):
        return value

    return apply


@compnode
def output(*_, **__):
    def apply(*value, **__):
        return jnp.array(value)

    return apply


@compnode
def source(get_param, get_quantized, n_outputs, **_):
    def apply(value, *_, **__):
        return jnp.ones(n_outputs) * value

    return apply


@compnode
def inv_source(*_, **__):
    # inverse of source is just a pass-through
    def apply(value, *_, **__):
        return value

    return apply


@compnode
def numeric(get_param, get_quantized, **_):
    def apply(rng_key):
        res = get_param("value", init=ut.continuous_initializer(rng_key))
        return res

    return apply


@compnode
def inv_numeric(*_, **__):
    # inverse of numeric is just a pass-through
    def apply(value, *_, **__):
        return value

    return apply


# # aggregations split a single input in ratios (defined by parameters)
# @compnode
# def aggregation(get_param, get_quantized, n_outputs, normalize=False, **kwargs):
# def apply(inp, quantile, rng_key):
# if 'ratios' in kwargs:
# ratios = get_param(
# "ratios", overwrite_with=jnp.array(kwargs['ratios'], dtype=jnp.float32)
# )
# else:
# ratios = get_param("ratios", init=ut.continuous_initializer(rng_key, (n_outputs,)))
# assert ratios.shape == (n_outputs,)
# # ratios = ratios / jnp.maximum(jnp.sum(ratios), 1e-12)
# return jnp.array(ratios) * inp
# return apply

# aggregations split a single input in ratios (defined by parameters)
@compnode
def aggregation(n_outputs, normalize=False, **kwargs):
    def apply(inp, quantile, rng_key):
        if 'ratios' in kwargs:
            ratios = get_param(
                "ratios", overwrite_with=jnp.array(kwargs['ratios'], dtype=jnp.float32)
            )
        else:
            ratios = get_param("ratios", init=ut.continuous_initializer(rng_key, (n_outputs,)))
        assert ratios.shape == (n_outputs,)
        # ratios = ratios / jnp.maximum(jnp.sum(ratios), 1e-12)
        return jnp.array(ratios) * inp

    return apply


@compnode
def inv_aggregation(get_param, get_quantized, original_output_len, original_output_slot, **_):
    assert original_output_len > 0
    assert original_output_slot < original_output_len

    def apply(inp, quantile, rng_key):
        ratios = get_param(
            "ratios", init=ut.continuous_initializer(rng_key, (original_output_len,))
        )
        # ratios = ratios / jnp.maximum(jnp.sum(ratios), 1e-12)
        return inp / ratios[original_output_slot]

    return apply


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                    --     neural utils     --

DEFAULT_ACTIVATION = jax.nn.leaky_relu
DEFAULT_OUT_ACTIVATION = jax.nn.sigmoid


def dense_layer(input_values, output_size, get_param, key, name):
    input_size = 1 if input_values.shape == () else input_values.shape[0]
    w = get_param(f'{name}_w', init=ut.he_initializer(key, (input_size, output_size)), shared=True)
    b = get_param(f'{name}_b', init=lambda: jnp.zeros((output_size,)), shared=True)

    assert input_values.shape == (
        input_size,
    ), f'In {name}: {input_values.shape} != {(input_size,)}'
    assert w.shape == (
        input_size,
        output_size,
    ), f'In {name}: {w.shape} != {(input_size, output_size)}'
    assert b.shape == (output_size,), f'In {name}: {b.shape} != {(output_size,)}'

    res = jnp.dot(input_values, w) + b
    return res.squeeze()


def dense_multilevel(
    input_values,
    hidden_s,
    output_s,
    depth,
    get_param,
    key,
    name,
    activation,
):
    res = input_values
    keys = jax.random.split(key, depth)
    for i in range(depth - 1):
        res = activation(dense_layer(res, hidden_s, get_param, keys[i], f'{name}_{i}'))

    return dense_layer(res, output_s, get_param, keys[-1], f'{name}_{depth - 1}')


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                       --     neural nodes     --


def transform_nn(
    get_param,
    get_quantized,
    transform_name,
    outer_wsize=64,
    outer_depth=2,
    inner_wsize=32,
    inner_depth=2,
    inner_out=4,
    rate_dim=1,
    tr_namespace='',
    inner_activation=DEFAULT_ACTIVATION,
    outer_activation=DEFAULT_OUT_ACTIVATION,
    **_,
):
    def inner(value, rate_embeding, quantile, key):
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

        assert quantile.ndim == 1
        assert value.ndim == 1
        assert rate_embeding.ndim == 1

        inputs = ut.flat_concat(value, rate_embeding, quantile)

        out = inner_activation(
            dense_multilevel(
                inputs,
                inner_wsize,
                inner_out,
                depth=inner_depth,
                get_param=get_param,
                key=key,
                name=f'{tr_namespace}{transform_name}_inner',
                activation=inner_activation,
            )
        )

        assert out.shape == (inner_out,)

        return out

    def apply(*values, quantile, rng_key):

        k0, k1, k2 = jax.random.split(rng_key, 3)
        val = jnp.array(values)

        rate_name = f'{transform_name}_rate'
        rate_shape = (val.shape[0], rate_dim)
        rates = get_quantized(
            rate_name,
            get_param(rate_name, init=ut.continuous_initializer(k0, rate_shape)),
            mode='input_edges',
        )

        assert val.shape[0] == rates.shape[0]

        # first we apply a simple inner layer to all inputs and sum them:
        inner_out = jnp.sum(
            jax.vmap(inner, in_axes=(0, 0, None, None))(val, rates, quantile, k1), axis=0
        )

        inner_out = ut.flat_concat(inner_out, quantile)

        # then we apply a final outer layer to the summed output:
        return outer_activation(
            dense_multilevel(
                inner_out,
                outer_wsize,
                1,
                depth=outer_depth,
                get_param=get_param,
                key=k2,
                name=f'{tr_namespace}{transform_name}_outer',
                activation=inner_activation,
            )
        )

    return apply


def sequestron_ERN(
    get_param,
    get_quantized,
    seq_name,
    affinity_dim=1,
    wsize=128,
    depth=3,
    out_dim=1,
    subtype='5p',
    inner_activation=DEFAULT_ACTIVATION,
    outer_activation=DEFAULT_OUT_ACTIVATION,
    **_,
):
    def apply(neg, pos, quantile, rng_key, **_):
        param_name = f'{seq_name}::affinity_{subtype}'
        affinity = get_param(
            param_name, init=ut.continuous_initializer(rng_key, (affinity_dim,)), shared=True
        )
        res = dense_multilevel(
            ut.flat_concat(neg, pos, affinity, quantile),
            wsize,
            out_dim,
            depth,
            get_param,
            rng_key,
            f'ERN_{subtype}',
            activation=inner_activation,
        )
        return outer_activation(jnp.squeeze(res))

    return apply


def output(
    get_param,
    get_quantized,
    wsize=64,
    depth=3,
    inner_activation=DEFAULT_ACTIVATION,
    outer_activation=DEFAULT_OUT_ACTIVATION,
    **_,
):
    def apply(*value, quantile, rng_key, **_):

        value = jnp.array(value)
        assert value.shape[0] == quantile.shape[0]
        assert quantile.ndim == 1

        res = jnp.array(
            [
                dense_multilevel(
                    ut.flat_concat(x, q),
                    wsize,
                    1,
                    depth,
                    get_param,
                    rng_key,
                    'out',
                    activation=inner_activation,
                )
                for x, q in zip(value, quantile)
            ]
        )
        return outer_activation(res)

    return apply


transcription = partial(transform_nn, transform_name='tc')
translation = partial(transform_nn, transform_name='tl')
inv_transcription = partial(transform_nn, transform_name='tc', tr_namespace='inv_')
inv_translation = partial(transform_nn, transform_name='tl', tr_namespace='inv_')

ERN5p = partial(sequestron_ERN, subtype='5p')
ERN3p = partial(sequestron_ERN, subtype='3p')


##────────────────────────────────────────────────────────────────────────────}}}

T_SIZE = 64
T_DEPTH = 4
I_SIZE = 64
I_DEPTH = 3
I_OUT = 8
ERN_SIZE = 128
ERN_DEPTH = 4
MEFL_SIZE = 64
MEFL_DEPTH = 4


class ComputeNodeFactory:
    type_to_fn = {}
    options_per_type = {}

    def __init__(self, node_type_to_fn=None, node_type_to_options=None):
        self.type_to_fn = node_type_to_fn or {}
        self.options_per_type = node_type_to_options or {}

    def get_impl(self, node_type):
        assert node_type in self.type_to_fn
        return partial(self.type_to_fn[node_type], **self.options_per_type[node_type])

    def register(self, node_type, impl, options=None):
        self.type_to_fn[node_type] = impl
        if options is not None:
            self.options_per_type[node_type] = options

    def overwrite_options(self, node_type, options):
        self.options_per_type[node_type] = options

    def update_options(self, node_type, options):
        self.options_per_type[node_type].update(options)
