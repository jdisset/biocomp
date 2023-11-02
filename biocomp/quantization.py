from .library import PartsLibrary as PartsLibrary
import jax
from jax import vmap, jit
import jax.numpy as jnp
import numpy as np
from . import utils as ut
from .parameters import ParameterTree

### {{{                       --     actual quantization functions    --


def quantize_masked(x, possible_values, mask):
    if len(possible_values) == 0:
        return x
    if len(possible_values) == 1:
        return possible_values[0]
    else:
        return quantize_masked_impl(x, possible_values, mask)[0]


def quantize_masked_impl(x, qvalues, mask):
    """Quantize x to the nearest element in qvalues, but only if the corresponding
    element in mask is True. Passthrough for x gradient.
    """
    zero = x - jax.lax.stop_gradient(x)  # for straight-through gradient
    dist = jnp.where(mask, jnp.abs(qvalues - x), jnp.inf)
    amin = jnp.argmin(dist)
    res = zero + qvalues[amin]
    return res, amin


##────────────────────────────────────────────────────────────────────────────}}}

### {{{               --     quantized parameters helpers     --

def get_quantized(
    values_to_quantize,
    params: ParameterTree,
    quantization_values_path,
    quantization_mask_path,
    node_id,
):
    """Quantize the given values using the quantization values stored in params."""
    # initialization of both keys and values is done upstream. We assume both are already initialized
    # i.e there is a param called param_name in params, which is a vector (n_qvalues, ...)
    # of all the possible quantization values for this parameter.
    possible_values = jnp.atleast_1d(params[quantization_values_path].squeeze())

    # possible_values is a 1D array of shape (n_qvalues,) that contains all the possible
    # quantization values for this parameter. Possible values for the whole stack,
    # but then will be masked to only use the ones available for this node
    masks = params[quantization_mask_path][node_id]

    assert masks.shape == (values_to_quantize.shape[0], len(possible_values))

    # masks is a 2D array of shape (max_n_masks_per_node, n_qvalues) that tells us which
    # quantization values are allowed for this node.
    # Remember that a node can have several inputs, coming from different nodes,
    # and each input can have a different set of possible quantization values.
    return vmap(quantize_masked, in_axes=(0, None, 0), out_axes=0)(
        values_to_quantize, possible_values, masks
    )

def get_quantized_rate_names(values_to_quantize, params, qnames, quantization_values_path, quantization_mask_path, node_id):
    possible_values = jnp.atleast_1d(params[quantization_values_path].squeeze())
    masks = params[quantization_mask_path][node_id]
    assert masks.shape == (values_to_quantize.shape[0], len(possible_values))
    names = []
    for v, m in zip(values_to_quantize, masks):
        _, i = quantize_masked_impl(v, possible_values, m)
        names.append(qnames[i])
    return names



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
    available_params = cdg.at[cdg_node_id, 'params']
    if param_name not in available_params:
        raise ValueError(
            f'Param {param_name} not available for cdg node {cdg_node_id}. Available: {available_params}'
        )
    return available_params[param_name]


def get_quantization_mask(qnames, pname, vnode, masks_per_node=1, **kwargs):
    """
    generate the quantization masks for a given vnode and parameter. One mask per input.
    - qnames: the list of quantization names for this parameter (e.g. ['hEF1a', ...])
    - params: the parameters dictionnary, where only arrays can be used (because it'll be jitted)
    - pname: the name of the parameter we want to quantize (e.g. 'tl_rate')
    - vnode: the node we want to quantize
    - masks_per_node: basically the max number of inputs a node can have
    """
    # example: generate_quantization_masks(['hEF1a', 'hEF1b', 'hEF1c'], params, 'tc_rate', vnode)
    compute_node_id = vnode.compute_node_id
    network = vnode.network
    if network is None:
        # pure virtual node, no network, no masks!
        ut.logger.warning(f'Node {vnode.node_id} has no network, no quantization mask generated')
        mask = np.ones((1, len(qnames)), dtype=bool)
        return mask

    cdf = network.compute_graph
    cdg = network.central_dogma_graph

    cdg_ids = cdf.at[compute_node_id, 'cdg_input']
    assert cdg_ids is not None, f'Node {compute_node_id} has no input CDG node'
    cdg_ids = [cdg_ids] if not isinstance(cdg_ids, list) else cdg_ids

    this_node_qnames = [get_available_quantizations(pname, cid, cdg) for cid in cdg_ids]
    # we have one mask per CDG input, and we need the same mask shape for all nodes
    assert len(this_node_qnames) <= masks_per_node, (
        f'Node {compute_node_id} has {len(this_node_qnames)} CDG inputs, '
        f'but only a max of {masks_per_node} masks are available'
    )
    # check that this_node_qnames is a subset of qnames
    should_be_in = [qq for q in this_node_qnames for qq in q if qq not in qnames]
    if len(should_be_in) > 0:
        raise ValueError(
            f'Node {compute_node_id} has unknown quantization names {should_be_in} '
            f'for parameter {pname} (available: {qnames})'
        )

    # now create the mask array
    mask = np.zeros((masks_per_node, len(qnames)), dtype=bool)
    for i in range(len(this_node_qnames)):
        mask[i, [qnames.index(q) for q in this_node_qnames[i]]] = True

    # now we store the mask in the params dict, under the mask namespace,
    return mask


def collapse_quantized_parameter(vnode, param_name, value):
    """
    collapse a quantized parameter into a single value
    - vnode: the node we want to quantize
    - param_name: the name of the parameter we want to quantize (e.g. 'tl_rate')
    - value: the value of the parameter as a list of names (1 per input)
    """

    compute_node_id = vnode.compute_node_id
    network = vnode.network
    if network is None:
        return
    cdf = network.compute_graph
    cdg = network.central_dogma_graph

    cdg_ids = cdf.at[compute_node_id, 'cdg_input']
    assert cdg_ids is not None, f'Node {compute_node_id} has no input CDG node'
    cdg_ids = [cdg_ids] if not isinstance(cdg_ids, list) else cdg_ids

    assert len(value) == len(cdg_ids), (
        f'Node {compute_node_id} has {len(cdg_ids)} CDG inputs, '
        f'but only {len(value)} values were provided'
    )

    for cid, val in zip(cdg_ids, value):
        if isinstance(val, (list, tuple)):
            assert len(val) == 1
        else:
            val = [val]
        current_params = cdg.at[cid, 'params']
        assert param_name in current_params, f'Param {param_name} not available for cdg node {cid}'
        current_params[param_name] = val
        cdg.at[cid, 'params'] = current_params


##────────────────────────────────────────────────────────────────────────────}}}
