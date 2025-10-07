from .library import PartsLibrary as PartsLibrary
import jax
from jax import vmap, jit
import jax.numpy as jnp
import numpy as np
from .parameters import ParameterTree
from jax import random as random
from biocomp.logging_config import get_logger

logger = get_logger(__name__)

### {{{                       --     actual quantization functions    --


def get_nearest_masked_id(x, qvalues, mask):
    """Quantize a single x continuous value, of shape (embedding_dim,),
    to the nearest quantization value in qvalues that has a corresponding True in mask.
    """
    assert mask.ndim == 1, (
        f"Quantization mask must be 1D, got {mask.ndim}D array with shape {mask.shape}"
    )
    assert x.ndim == 1, f"Input value x must be 1D, got {x.ndim}D array with shape {x.shape}"
    assert qvalues.shape == (mask.shape[0], x.shape[0]), (
        f"Quantization values shape {qvalues.shape} does not match input value shape {x.shape} and mask shape {mask.shape}"
    )

    # Compute the distance to each quantization value
    distances = jnp.linalg.norm(qvalues - x, axis=-1)
    masked_dist = jnp.where(mask, distances, jnp.inf)
    assert masked_dist.shape == mask.shape, (
        f"Masked distances shape {masked_dist.shape} does not match mask shape {mask.shape}"
    )
    closest_idx = jnp.argmin(masked_dist)
    return closest_idx


def straight_through(x_quant, x):
    # forward: x_quant   |  backward: identity grad wrt x & x_quant
    zero = x - jax.lax.stop_gradient(x)
    return x_quant + zero


def quantize_value_to_nearest_masked_embedding(x, qvalues, mask):
    """Quantize a single x continuous value, of shape (embedding_dim,),
    to the nearest quantization value in qvalues that has a corresponding True in mask.
    """

    closest_idx = get_nearest_masked_id(x, qvalues, mask)
    return straight_through(qvalues[closest_idx], x)


def quantize_all_values_to_nearest_masked_embeddings(
    values_to_quantize,  # values to quantize, shape (nvalues, embedding_dim)
    embedding_values,  # possible quantization values, shape (n_qvalues, embedding_dim)
    masks,  # quantization mask, shape (max_n_masks_per_node, n_qvalues)
):
    check_multiple_quantization_shapes(values_to_quantize, embedding_values, masks)
    mask_in_axes = None if masks.ndim == 1 else 0
    return vmap(
        quantize_value_to_nearest_masked_embedding, in_axes=(0, None, mask_in_axes), out_axes=0
    )(values_to_quantize, embedding_values, masks)


def check_multiple_quantization_shapes(
    values_to_quantize,  # values to quantize, shape (nvalues, embedding_dim)
    embedding_values,  # possible quantization values, shape (n_qvalues, embedding_dim)
    masks,  # quantization mask, shape (max_n_masks_per_node, n_qvalues)
):
    assert values_to_quantize.ndim == 2, (
        f"Values to quantize must be 2D, got {values_to_quantize.ndim}D array with shape {values_to_quantize.shape} ({values_to_quantize=})"
    )
    assert embedding_values.ndim == 2, (
        f"Embedding values must be 2D, got {embedding_values.ndim}D array with shape {embedding_values.shape} ({embedding_values=})"
    )
    assert values_to_quantize.shape[1] == embedding_values.shape[1], (
        f"Values to quantize shape {values_to_quantize.shape} does not match embedding values shape {embedding_values.shape}"
    )


def get_quantized(
    values_to_quantize,  # values to quantize, shape (nvalues, embedding_dim)
    params: ParameterTree,
    quantization_values_path,  # path to the quantization values in params, e.g. 'quantization_values/tl_rate'
    quantization_mask_path,  # path to the quantization mask in params, e.g. 'quantization_mask/tl_rate'
    node_id,  # masks vary per node, so we need to know which node we are quantizing for
):
    """Quantize the given values using the quantization values stored in params."""

    # initialization of both keys and values is done upstream. We assume both are already initialized
    # i.e there is a param called param_name in params, which is a vector (n_qvalues, ...)
    # of all the possible quantization values for this parameter.

    # embedding_values an array of shape (nvalues, ndims) that contains all the possible
    # quantization values (aka "code book embeddings") for this parameter.
    # Shared for the whole stack, but then will be masked to only use the ones available for this node

    # masks is a 2D array of shape (max_n_masks_per_node, n_qvalues) that tells us which
    # quantization values are allowed for this node.
    # 1 mask per value to quantize.
    # Remember that a node can have several inputs, coming from different nodes,
    # and each input can have a different set of possible quantization values.

    embedding_values = params[quantization_values_path]  # aka possible values
    masks = params[quantization_mask_path][node_id]

    assert masks.shape == (values_to_quantize.shape[0], embedding_values.shape[0]), (
        f"Quantization mask shape {masks.shape} does not match values to quantize shape {values_to_quantize.shape} "
        f"and embedding values shape {embedding_values.shape}"
    )

    return quantize_all_values_to_nearest_masked_embeddings(
        values_to_quantize, embedding_values, masks
    ), {
        "q_masks": masks,
        "q_node_id": node_id,
    }


def dot_gather(operand, indices):
    """
    gather (take) operation using a one-hot dot product.
    workaround for cases where direct indexing makes checkify unhappy.
    """
    num_classes = operand.shape[0]
    one_hot = jax.nn.one_hot(indices, num_classes=num_classes)
    return jnp.dot(one_hot, operand)


def get_variational_quantized(
    values_to_quantize,
    params: ParameterTree,
    quantization_values_path,
    quantization_mask_path,
    logstdevs_path,
    node_id,
    key,
    min_logstdev=-10.0,
    max_logstdev=5.0,
):
    """
    Quantize the given values using the quantization values ("code book" of embeddings) stored in params,
    and add noise based on the log standard deviations stored in params.
    This function is used for variational quantization, where we add noise to the quantized values.
    """

    masks = jnp.asarray(params[quantization_mask_path])
    mask_for_node = jnp.take(masks, node_id, axis=0)

    embedding_means = params[quantization_values_path]  # aka possible values
    embeddings_logstds = params[logstdevs_path]
    assert embedding_means.shape == embeddings_logstds.shape, (
        f"Embedding means shape {embedding_means.shape} does not match embeddings logstds shape {embeddings_logstds.shape}"
    )
    check_multiple_quantization_shapes(values_to_quantize, embedding_means, mask_for_node)

    assert mask_for_node.shape == (values_to_quantize.shape[0], embedding_means.shape[0]), (
        f"Quantization mask shape {mask_for_node.shape} does not match values to quantize shape {values_to_quantize.shape} "
        f"and embedding values shape {embedding_means.shape}"
    )

    closest_ids = vmap(get_nearest_masked_id, in_axes=(0, None, 0), out_axes=0)(
        values_to_quantize, embedding_means, mask_for_node
    )

    assert len(closest_ids) == values_to_quantize.shape[0]

    values_logstds = dot_gather(embeddings_logstds, closest_ids)
    values_logstds = jnp.clip(values_logstds, min_logstdev, max_logstdev)

    assert values_logstds.shape == values_to_quantize.shape, (
        f"Values logstds shape {values_logstds.shape} does not match values to quantize shape {values_to_quantize.shape}"
    )

    means = straight_through(embedding_means[closest_ids], values_to_quantize)
    assert means.shape == values_to_quantize.shape, (
        f"Means shape {means.shape} does not match values to quantize shape {values_to_quantize.shape}"
    )
    noise = random.normal(key, means.shape) * jnp.exp(values_logstds)
    assert noise.shape == means.shape, (
        f"Noise shape {noise.shape} does not match means shape {means.shape}"
    )
    vq = means + noise
    assert vq.shape == values_to_quantize.shape, (
        f"Quantized values shape {vq.shape} does not match values to quantize shape {values_to_quantize.shape}"
    )

    return vq, {
        "q_masks": mask_for_node,
        "q_embedding_logstdevs": embeddings_logstds,
        "q_node_id": node_id,
        "q_logstdevs": values_logstds,
        "q_means": means,
        "q_noise": noise,
    }


##────────────────────────────────────────────────────────────────────────────}}}

### {{{               --     quantized parameters helpers     --


def get_quantized_rate_names(
    values_to_quantize, params, qnames, quantization_values_path, quantization_mask_path, node_id
):
    """
    Get closest part names for the given continuous values.
    """
    possible_values = params[quantization_values_path]  # shape: (n_qvalues, rate_dim)
    # ensure it's at least 2D for quantize_value_to_nearest_masked_embedding
    if possible_values.ndim == 1:
        possible_values = possible_values.reshape(-1, 1)

    masks = params[quantization_mask_path][node_id]
    assert masks.shape == (values_to_quantize.shape[0], possible_values.shape[0]), (
        f"Mask shape {masks.shape} doesn't match expected "
        f"({values_to_quantize.shape[0]}, {possible_values.shape[0]})"
    )

    names = []
    for v, m in zip(values_to_quantize, masks):
        # v has shape (rate_dim,), needs to match possible_values dim
        if v.ndim == 0:
            v = v.reshape(1)
        idx = get_nearest_masked_id(v, possible_values, m)
        names.append(qnames[idx])
    return names


def get_available_quantizations(param_name, cdg_node_id, cdg):
    """
    returns the name of possible parts for a given cdg node, slot and param name
    example: get_possible_values('transcription_rate', ...) -> ['hEF1a', 'hEF1b', 'hEF1c']
              get_possible_values('translation_rate', ...) -> [None, '1xuORF', '2xuORF', ...]
    params are stored in the params column of the cdg as a dict {param_name:[possiblevaluees]}
    """
    available_params = cdg.at[cdg_node_id, "params"]
    if param_name not in available_params:
        raise ValueError(
            f"Param {param_name} not available for cdg node {cdg_node_id}. Available: {available_params}"
        )
    return available_params[param_name]


def get_quantization_mask(qnames, pname, vnode, masks_per_node=1, **kwargs):
    """
    generate the quantization masks for a given vnode and parameter. One mask per input.
    - qnames: the list of quantization names for this parameter (e.g. ['hEF1a', ...])
    - params: the parameters dictionnary, where only arrays can be used (because it'll be jitted)
    - pname: the name of the parameter we want to quantize (e.g. 'tl_rate')
    - vnode: the node we want to quantize
    - masks_per_node: basically the max number of inputs a node can have (extras will be ignored)
    """
    # example: generate_quantization_masks(['hEF1a', 'hEF1b', 'hEF1c'], params, 'tc_rate', vnode)
    compute_node_id = vnode.compute_node_id
    network = vnode.network
    if network is None:
        # pure virtual node, no network, no masks!
        logger.warning(f"Node {vnode.node_id} has no network, no quantization mask generated")
        mask = np.ones((1, len(qnames)), dtype=bool)
        return mask

    cdf = network.compute_graph
    cdg = network.central_dogma_graph

    cdg_ids = cdf.at[compute_node_id, "cdg_input"]
    assert cdg_ids is not None, f"Node {compute_node_id} has no input CDG node"
    cdg_ids = [cdg_ids] if not isinstance(cdg_ids, list) else cdg_ids

    this_node_qnames = [get_available_quantizations(pname, cid, cdg) for cid in cdg_ids]
    # we have one mask per CDG input, and we need the same mask shape for all nodes
    assert len(this_node_qnames) <= masks_per_node, (
        f"Node {compute_node_id} has {len(this_node_qnames)} CDG inputs, "
        f"but only a max of {masks_per_node} masks are available"
    )
    # check that this_node_qnames is a subset of qnames
    should_be_in = [qq for q in this_node_qnames for qq in q if qq not in qnames]
    if len(should_be_in) > 0:
        raise ValueError(
            f"Node {compute_node_id} has unknown quantization names {should_be_in} "
            f"for parameter {pname} (available: {qnames})"
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

    cdg_ids = cdf.at[compute_node_id, "cdg_input"]
    assert cdg_ids is not None, f"Node {compute_node_id} has no input CDG node"
    cdg_ids = [cdg_ids] if not isinstance(cdg_ids, list) else cdg_ids

    assert len(value) == len(cdg_ids), (
        f"Node {compute_node_id} has {len(cdg_ids)} CDG inputs, "
        f"but only {len(value)} values were provided"
    )

    # Update CDG params
    for cid, val in zip(cdg_ids, value):
        if isinstance(val, (list, tuple)):
            assert len(val) == 1
        else:
            val = [val]
        current_params = cdg.at[cid, "params"]
        assert param_name in current_params, f"Param {param_name} not available for cdg node {cid}"
        current_params[param_name] = val
        cdg.at[cid, "params"] = current_params

    # Also update the TranscriptionUnit slots to reflect the quantized values
    if network.transcription_units is not None:
        for cdg_id, resolved_name in zip(cdg_ids, value):
            if cdg_id in cdg.index:
                tu_ids = cdg.at[cdg_id, "tu_id"]
                if tu_ids:
                    tu_id = tu_ids[0] if isinstance(tu_ids, list) else tu_ids
                    if tu_id in network.transcription_units:
                        tu = network.transcription_units[tu_id]
                        
                        # Update the TU params to have the single quantized value
                        if param_name in tu.params:
                            tu.params[param_name] = [resolved_name]
                        
                        # Update the slots that map to this parameter
                        for slot in tu.slots:
                            if slot.maps_to_parameter == param_name:
                                # Set the slot's part to the single quantized value
                                # Keep it as a list to maintain consistency
                                slot.part = [resolved_name]


##────────────────────────────────────────────────────────────────────────────}}}
