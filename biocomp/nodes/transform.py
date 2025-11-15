from biocomp.jaxutils import flat_concat
from biocomp.compute import StackNode, ComputeStack
import jax
from jax.tree_util import Partial as partial
from jax.typing import ArrayLike
import jax.numpy as jnp
from jax import vmap
import numpy as np
from biocomp.parameters import ArrayRef, ParameterTree, init_if_needed, make_view, get_param
from biocomp.nodeutils import (
    LayerInstance,
    add_random_var_ids,
    NON_GRAD_TAG,
    get_prev_num_random_vars,
    reference_forward_random_var_ids,
)
from biocomp.utils import get_logger

from biocomp.neuralutils import (
    ACTIVATION_FUNCTIONS,
    INITIALIZERS,
    DEFAULT_ACTIVATION,
    DEFAULT_OUT_ACTIVATION,
    DEFAULT_INITIALIZER,
    dense_mlp,
    dummy_mlp,
)

import biocomp.quantization as qz


PRNGKey = ArrayLike
NDArray = np.ndarray | jnp.ndarray

logger = get_logger(__name__)


def identity(x):
    return x


def transform_nn(
    input_shapes: list[tuple[int]],
    n_outputs: int,
    stack: ComputeStack,
    namespace: str,
    transform_name: str,
    quantization_names: list[str],  # ordered list. ex: ['1xuorf', '2xuorf', ...]
    outer_wsize: int = 64,
    outer_depth: int = 4,
    inner_wsize: int = 64,
    inner_depth: int = 3,
    inner_outsize: int = 8,
    rate_dim: int = 1,
    is_inverse: bool = False,
    inner_activation_name: str = DEFAULT_ACTIVATION,
    outer_activation_name: str = DEFAULT_OUT_ACTIVATION,
    initializer_name: str = DEFAULT_INITIALIZER,
    bias_offset: float = 0.0,
    alpha_init: float = 0.5,
    beta_init: float = 0.5,
    dummy: bool = False,  # disable neural + residual, for testing
):
    # TODO: make sure incoming edges order is deterministic

    assert n_outputs == 1, f"NN transform only supports 1 output, got {n_outputs}"
    if is_inverse and len(input_shapes) != 1:
        raise ValueError(f"Inverse {transform_name} should have 1 input, got {len(input_shapes)}")

    if not all(s == input_shapes[0] for s in input_shapes):
        raise ValueError(
            f"All inputs of a transformation should have the same shape, got {input_shapes}"
        )

    inner_activation = ACTIVATION_FUNCTIONS[inner_activation_name]
    outer_activation = ACTIVATION_FUNCTIONS[outer_activation_name]
    initializer = INITIALIZERS[initializer_name]

    layer_name = namespace.split("/")[-1]  # extract layer name from namespace

    rate_shape = (len(input_shapes), rate_dim)
    rate_name = f"{transform_name}_rate"  # _x{len(input_shapes)}'
    shared_layer_name = f"{'inv' if is_inverse else 'fwd'}_{transform_name}"

    quantization_values_path = f"shared/quantization/values/{rate_name}"
    mask_name = f"{rate_name}_quantization_mask"
    quantization_mask_path = f"{namespace}/{mask_name}"

    logstdevs_path = f"shared/quantization/logstdevs/{rate_name}"
    count_array_path = f"shared/quantization/counts/{rate_name}"

    mlp = dummy_mlp if dummy else dense_mlp
    i_activation = identity if dummy else inner_activation
    o_activation = identity if dummy else outer_activation

    def inner(params, value: NDArray, random_var, rate_embedding: NDArray, key: PRNGKey):
        """For a single source, computes a latent output from the concatenation of
        the rate embedding and the source value.
        All of these outputs will then be summed up and passed through a final layer.
        """

        if value.ndim == 0:
            value = value.reshape((1,))
        if rate_embedding.ndim == 0:
            rate_embedding = rate_embedding.reshape((1,))

        assert value.ndim == 1, f"In {transform_name}: {value.ndim} != 1: {value}"
        assert rate_embedding.ndim == 1

        inputs = flat_concat(value, rate_embedding, random_var)

        out = i_activation(
            mlp(
                inputs,
                inner_wsize,
                inner_outsize,
                depth=inner_depth,
                activation=inner_activation,
                initializer=initializer,
                bias_offset=bias_offset,
                key=key,
                param_f=partial(init_if_needed, params, base_path="shared"),
                name=f"NN/{shared_layer_name}/inner",
            )
        )
        if dummy and is_inverse:
            out = out - (inner_outsize + 1) * (rate_embedding + random_var)

        assert out.shape == (inner_outsize,)

        return out

    def prepare(params: ParameterTree, nodelist: list[StackNode], key: PRNGKey):
        key0, key1 = jax.random.split(key, 2)
        n_nodes = len(nodelist)

        # --------- quantization
        # First, initializing quantization values for the rates (if not already done)
        # qnames is a list of names for the rate values available in this stack (1xuORf, ...)
        try:
            qvalues = params[quantization_values_path]
        except KeyError:
            if rate_dim <= 1:
                qvalues = jnp.linspace(-1, 1, len(quantization_names) * rate_dim).reshape(
                    (len(quantization_names), rate_dim)
                )
            else:
                qvalues = jax.random.normal(key0, (len(quantization_names), rate_dim))
            params[quantization_values_path] = qvalues
        # Now initialize logstdevs in the same way
        try:
            logstdevs = params[logstdevs_path]
        except KeyError:
            logstdevs = jnp.zeros((len(quantization_names), rate_dim)) - 3
            params[logstdevs_path] = logstdevs

        assert qvalues.shape == (len(quantization_names), rate_dim)

        if not dummy:
            init_if_needed(
                params,
                f"shared/{shared_layer_name}/residual_alpha",
                init_f=lambda: jnp.array(alpha_init),
            )
            init_if_needed(
                params,
                f"shared/{shared_layer_name}/residual_beta",
                init_f=lambda: jnp.array(beta_init),
            )

        if not is_inverse:  # forward node
            # We initialize quantization masks for these nodes.
            # Quantization masks are used to select which qvalues are accessible to each node.
            qmasks = [
                qz.get_quantization_mask(quantization_names, rate_name, node, stack)
                for node in nodelist
            ]
            for m in qmasks:
                assert m.shape == (len(input_shapes), len(quantization_names)), (
                    f"Invalid quantization mask shape {m.shape} for node in layer {layer_name}, expected {(len(input_shapes), len(quantization_names))}"
                )

            params.at(f"{quantization_mask_path}", np.array(qmasks), tags=[NON_GRAD_TAG])
            logger.debug(
                f"quantization mask for {layer_name}:\n{qz.quantization_mask_str(quantization_names, qmasks)}"
            )
            try:
                params.at(
                    count_array_path,
                    np.array(qmasks).sum(axis=(0, 1)) + params.at(count_array_path),
                    overwrite=True,
                    tags=[NON_GRAD_TAG],
                )
            except KeyError:
                params.at(
                    count_array_path,
                    np.array(qmasks).sum(axis=(0, 1)),
                    tags=[NON_GRAD_TAG],
                )

            # And we also initialize the quantized rates
            params[f"{namespace}/{rate_name}"] = jax.random.uniform(key1, (n_nodes, *rate_shape))

        else:
            # For inverse nodes, we will use a view (a subtree of ArrayRef that mirrors the original subtree)
            # of both the quantized rates and the quantization masks of the corresponding forward nodes,
            # since they should be shared between the forward and inverse nodes.
            def get_fwd(node):
                fwd_node = node.get_forward_stacknode(stack)
                fwd_namespace = stack.get_layer_namespace(fwd_node.layer_number)
                return fwd_namespace, fwd_node.node_position_in_layer

            fwd_paths, fwd_loc = zip(*[get_fwd(node) for node in nodelist])

            # make view will create 2 subtrees of ArrayRef, one for the rates and one for the masks
            # that point to the same underlying data as the forward nodes
            make_view(params, namespace, fwd_paths, fwd_loc, leaves=[rate_name, mask_name])
            params.tag(f"{namespace}/{mask_name}", [NON_GRAD_TAG])

        # --------- random_var var
        if is_inverse:
            reference_forward_random_var_ids(stack, params, nodelist, namespace)
        else:
            add_random_var_ids(params, len(nodelist), len(input_shapes) + 1, namespace)

        fake_vals = [np.zeros(s) for s in input_shapes]

        apply(
            *fake_vals,
            random_vars=np.zeros(get_prev_num_random_vars(params) + 1),
            params=params,
            node_id=0,
            key=key1,
        )

    def outer(inner_out: ArrayLike, params, key: PRNGKey):
        if dummy and is_inverse:
            out = jnp.array([(inner_out[0] - inner_out[-1]) / inner_outsize])
        else:
            out = o_activation(
                mlp(
                    inner_out,
                    outer_wsize,
                    1,
                    depth=outer_depth,
                    param_f=partial(init_if_needed, params, base_path="shared"),
                    initializer=initializer,
                    bias_offset=bias_offset,
                    key=key,
                    name=f"NN/{shared_layer_name}/outer",
                    activation=inner_activation,
                )
            )
        assert out.shape == (1,), f"Invalid outer output shape {out.shape}"
        return out

    def apply(
        *values: ArrayLike,
        random_vars: ArrayLike,
        params: ParameterTree,
        node_id: ArrayLike,
        key: PRNGKey,
    ) -> tuple[ArrayLike, dict]:
        k1, k2, k3 = jax.random.split(key, 3)

        rvid = params[f"{namespace}/random_variable_id"][node_id]
        random_var = random_vars[rvid]

        val = jnp.array(values)

        rates = params[f"{namespace}/{rate_name}"][node_id]

        assert val.shape == (len(input_shapes), *input_shapes[0])
        assert rates.shape == (len(input_shapes), rate_dim)
        assert random_var.shape == (len(input_shapes) + 1,)

        qrates, qaux = qz.get_variational_quantized(
            rates,
            params,
            quantization_values_path,
            quantization_mask_path,
            logstdevs_path,
            node_id,
            k3,
            disable_variational=dummy,
        )

        # first we apply the inner head to all inputs and sum them:
        inner_keys = jax.random.split(k1, val.shape[0])

        inner_out = sum(
            inner(params, value=v, random_var=random_var[i], rate_embedding=r, key=k)
            for i, (v, r, k) in enumerate(zip(val, qrates, inner_keys))
        )

        inner_out = flat_concat(inner_out, random_var[len(input_shapes)])

        assert inner_out.shape == (inner_outsize + 1,)

        # then we apply a final outer layer to the summed output:
        ans = outer(inner_out, params, k2)

        # residual connection
        input_sum = jnp.sum(val, axis=0)
        if not dummy:
            alpha = params[f"shared/{shared_layer_name}/residual_alpha"]
            beta = params[f"shared/{shared_layer_name}/residual_beta"]
            # apply softmax normalization to alpha and beta
            alpha_norm = jnp.exp(alpha) / (jnp.exp(alpha) + jnp.exp(beta))
            beta_norm = jnp.exp(beta) / (jnp.exp(alpha) + jnp.exp(beta))
            final_output = alpha_norm * input_sum + beta_norm * ans
        else:
            final_output = ans
            alpha_norm = jnp.array(0.0)
            beta_norm = jnp.array(0.0)

        return final_output, {
            "random_var": random_var,
            "rates": rates,
            "quantized_rates": qrates,
            "inner_output": inner_out,
            "outer_output": ans,
            "input_sum": input_sum,
            "alpha_norm": alpha_norm,
            "beta_norm": beta_norm,
            "is_inverse": is_inverse,
            "n_inputs": len(input_shapes),
            **qaux,
        }

    def commit(params: ParameterTree, nodelist: list[StackNode], stack: ComputeStack, **_):
        for node_id, node in enumerate(nodelist):
            rates = params[f"{namespace}/{rate_name}"][node_id]
            resolved_parameter_names = qz.get_quantized_part_names(
                rates,
                params,
                quantization_names,
                quantization_values_path,
                quantization_mask_path,
                node_id,
            )
            i_edges = node.get_incoming_edges(stack)
            assert len(i_edges) == len(resolved_parameter_names), (
                f"Number of incoming edges {len(i_edges)} does not match number of resolved rate names {len(resolved_parameter_names)}"
                f" for node {node} in namespace {namespace}"
            )
            for e, pname in zip(i_edges, resolved_parameter_names):
                e.content_embedding_names[rate_name] = (pname,)

    output_shape = [(1,)]

    return LayerInstance(prepare, apply, output_shape, commit=commit)


from biocomp.part_embeddings import EMBEDDINGS_BY_NAME

transcription = partial(
    transform_nn,
    transform_name="tc",
    quantization_names=EMBEDDINGS_BY_NAME["tc_rate"].available_parts,
)
translation = partial(
    transform_nn,
    transform_name="tl",
    quantization_names=EMBEDDINGS_BY_NAME["tl_rate"].available_parts,
)

inv_transcription = partial(
    transform_nn,
    transform_name="tc",
    is_inverse=True,
    quantization_names=EMBEDDINGS_BY_NAME["tc_rate"].available_parts,
)
inv_translation = partial(
    transform_nn,
    transform_name="tl",
    is_inverse=True,
    quantization_names=EMBEDDINGS_BY_NAME["tl_rate"].available_parts,
)

simple_transcription = partial(
    transform_nn,
    transform_name="tc",
    quantization_names=EMBEDDINGS_BY_NAME["tc_rate"].available_parts,
    dummy=True,
)
simple_translation = partial(
    transform_nn,
    transform_name="tl",
    quantization_names=EMBEDDINGS_BY_NAME["tl_rate"].available_parts,
    dummy=True,
)

simple_inv_transcription = partial(
    transform_nn,
    transform_name="tc",
    is_inverse=True,
    quantization_names=EMBEDDINGS_BY_NAME["tc_rate"].available_parts,
    dummy=True,
)
simple_inv_translation = partial(
    transform_nn,
    transform_name="tl",
    is_inverse=True,
    quantization_names=EMBEDDINGS_BY_NAME["tl_rate"].available_parts,
    dummy=True,
)
