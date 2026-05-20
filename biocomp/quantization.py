# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from .library import PartsLibrary as PartsLibrary
import jax
from jax import vmap
import jax.numpy as jnp
import numpy as np
from .parameters import ParameterTree
from jax import random as random
from biocomp.logging_config import get_logger
from biocomp.jaxutils import check as jax_check
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from biocomp.compute import StackNode

logger = get_logger(__name__)


def quantization_mask_str(names, mask) -> str:
    col_width = max(len(name) for name in names)
    result = " " * 5
    for _i, name in enumerate(names):
        result += f"{name:^{col_width}} "
    result += "\n"
    for i, row in enumerate(mask):
        result += f"{i:<4}|"
        for val in row[0]:
            result += f"{'X' if val else ' ':^{col_width}}|"
        result += "\n"
    return result


### {{{                       --     actual quantization functions    --


def get_nearest_masked_id(x, qvalues, mask):
    """Quantize a single x continuous value, of shape (embedding_dim,),
    to the nearest quantization value in qvalues that has a corresponding True in mask.

    Raises AssertionError (or checkify error if enabled) if mask is all False.
    """
    assert mask.ndim == 1, (
        f"Quantization mask must be 1D, got {mask.ndim}D array with shape {mask.shape}"
    )
    assert x.ndim == 1, f"Input value x must be 1D, got {x.ndim}D array with shape {x.shape}"
    assert qvalues.shape == (mask.shape[0], x.shape[0]), (
        f"Quantization values shape {qvalues.shape} does not match input value shape {x.shape} and mask shape {mask.shape}"
    )

    distances = jnp.linalg.norm(qvalues - x, axis=-1)
    masked_dist = jnp.where(mask, distances, jnp.inf)
    assert masked_dist.shape == mask.shape, (
        f"Masked distances shape {masked_dist.shape} does not match mask shape {mask.shape}"
    )

    # CRITICAL: Check for empty mask (all False) - design is impossible if no valid options
    # This check works with JAX checkify when enabled, catching the error at runtime in JIT code
    has_valid_option = jnp.any(mask)
    jax_check(has_valid_option, "Quantization mask has no valid options (all False). Design is impossible.")

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
    disable_variational=False,  # if True, no noise is added
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

    if not disable_variational:
        noise = random.normal(key, means.shape) * jnp.exp(values_logstds)
        assert noise.shape == means.shape, (
            f"Noise shape {noise.shape} does not match means shape {means.shape}"
        )
    else:
        noise = jnp.zeros_like(means)

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

### {{{               --     quantized parameters (parts embeddings) helpers     --


def get_quantized_part_names(
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
    for v, m in zip(values_to_quantize, masks, strict=False):
        # v has shape (rate_dim,), needs to match possible_values dim
        if v.ndim == 0:
            v = v.reshape(1)
        idx = get_nearest_masked_id(v, possible_values, m)
        names.append(qnames[idx])
    return names


def get_quantization_mask(all_qnames, pname, node: "StackNode", stack):
    """
    generate the quantization masks for a given node and parameter. One mask per input edge.
    - all_qnames: the list of quantization names, aka embedding names for this parameter (e.g. ['hEF1a', ...])
    - pname: the name of the parameter we want to quantize (e.g. 'tl_rate')
    - node: the StackNode we want to quantize
    - stack: the compute stack
    """
    node_qnames = [e.content_embedding_names[pname] for e in node.get_incoming_edges(stack)]
    mask = np.zeros((len(node_qnames), len(all_qnames)), dtype=bool)
    for i in range(len(node_qnames)):
        mask[i, [all_qnames.index(q) for q in node_qnames[i]]] = True

    return mask


##────────────────────────────────────────────────────────────────────────────}}}
