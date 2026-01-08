from biocomp.jaxutils import flat_concat
from biocomp.compute import StackNode, ComputeStack
import jax
from jax.tree_util import Partial as partial
from jax.typing import ArrayLike
import jax.numpy as jnp
import numpy as np
from biocomp.parameters import ArrayRef, ParameterTree, init_if_needed, get_param
from biocomp.nodeutils import (
    LayerInstance,
    add_random_var_ids,
    add_tu_input_mapping,
    add_node_network_ids,
    NON_GRAD_TAG,
)
from biocomp.utils import get_logger
from typing import Optional
from biocomp.neuralutils import (
    ACTIVATION_FUNCTIONS,
    INITIALIZERS,
    DEFAULT_ACTIVATION,
    DEFAULT_OUT_ACTIVATION,
    DEFAULT_INITIALIZER,
    dense_mlp,
    dummy_mlp,
    uniform_initializer,
)
from typing import Callable


PRNGKey = ArrayLike
NDArray = np.ndarray | jnp.ndarray

logger = get_logger(__name__)


def identity(x):
    return x


def sequestron_ERN(
    input_shapes: list[tuple[int, ...]],
    n_outputs: int,
    stack: ComputeStack,
    namespace: str,
    affinity_names: list[str],  # ordered list of available affinity names (case, csy4, etc..)
    affinity_dim: int = 1,
    wsize: int = 128,
    depth: int = 4,
    out_dim: int = 1,
    subtype: str = "5p",
    inner_activation_name: str = DEFAULT_ACTIVATION,
    outer_activation_name: str = DEFAULT_OUT_ACTIVATION,
    initializer_name: str = DEFAULT_INITIALIZER,
    bias_offset: float = 0.0,
    use_ern_layer_id: bool = False,
    max_ern_layers: int = 4,  # for one-hot encoding size
    alpha_init: float = 0.5,  # initial value for input residual
    beta_init: float = 0.5,  # initial value for network output
    dummy: bool = False,  # disable neural + residual, for testing
) -> LayerInstance:
    inner_activation = ACTIVATION_FUNCTIONS[inner_activation_name]
    outer_activation = ACTIVATION_FUNCTIONS[outer_activation_name]

    i_activation = identity if dummy else inner_activation
    o_activation = identity if dummy else outer_activation

    initializer = INITIALIZERS[initializer_name]

    # ERN have 2 inputs of same size
    assert len(input_shapes) == 2
    assert input_shapes[0] == input_shapes[1], (
        f"ERN inputs must have same shape, got {input_shapes}"
    )
    assert n_outputs == 1, f"ERN only supports 1 output, got {n_outputs}"

    shared_layer_name = f"ERN_{subtype}"
    local_layer_name = namespace.split("/")[-1]  # extract layer name from namespace

    def MLP(
        neg: ArrayLike,
        pos: ArrayLike,
        affinity: ArrayLike,
        random_var: ArrayLike,
        param_f: Callable,
        key: PRNGKey,
        layer_id_onehot: NDArray = np.empty((0,)),
    ):
        if use_ern_layer_id:
            input_values = flat_concat(neg, pos, affinity, layer_id_onehot, random_var)
            assert layer_id_onehot.shape == (max_ern_layers,), (
                f"ERN layer_id_onehot should be of size {max_ern_layers}, got {len(layer_id_onehot)}"
            )
        else:
            input_values = flat_concat(neg, pos, affinity, random_var)

        mlp = dummy_mlp if dummy else dense_mlp

        res = mlp(
            input_values,
            wsize,
            out_dim,
            depth,
            param_f=param_f,
            initializer=initializer,
            bias_offset=bias_offset,
            key=key,
            name=f"NN/ERN_{subtype}",
            activation=i_activation,
        )

        # add residual connections
        neg_sum = jnp.sum(neg)
        pos_sum = jnp.sum(pos)
        if not dummy:
            alpha = param_f(
                f"{shared_layer_name}/residual_alpha", init_f=lambda: jnp.array(alpha_init)
            )
            beta = param_f(
                f"{shared_layer_name}/residual_beta", init_f=lambda: jnp.array(beta_init)
            )
            # apply softmax normalization to alpha and beta
            alpha = jnp.exp(alpha) / (jnp.exp(alpha) + jnp.exp(beta))
            beta = jnp.exp(beta) / (jnp.exp(alpha) + jnp.exp(beta))
            out = alpha * (pos_sum - neg_sum) + beta * res
        else:
            out = res
        return out

    def prepare(params: ParameterTree, nodelist: list[StackNode], key: PRNGKey):
        # --------- random_var var
        add_random_var_ids(params, len(nodelist), 1, namespace)
        add_tu_input_mapping(params, stack, nodelist, namespace)
        add_node_network_ids(params, nodelist, namespace)

        init_if_needed(
            params,
            f"shared/{shared_layer_name}/affinities",
            init_f=uniform_initializer(key, (len(affinity_names), affinity_dim)),
        )

        # for now the ERN node does'nt use the more complex quantization,
        # we just have one affinity value per ERN type (case, csy4, etc..)
        # and store one reference to the affinity value per node.

        # very important to use ArrayRef so that we don't copy the data which
        # would be catastrophic as it would create one new affinity value per node
        ref = ArrayRef(params.data)

        # store node layer ids if enabled
        seq_layer_ids = []

        for node in nodelist:
            # handle affinity value for this node
            comp_node = node.get(stack)
            seq_name = comp_node.extra["seq_name"]
            if seq_name not in affinity_names:
                raise ValueError(f"Unknown affinity name {seq_name}. Available: {affinity_names}")
            affinity_id = affinity_names.index(seq_name)
            ref.push_back(f"shared/{shared_layer_name}/affinities", affinity_id)

            # collect node layer ids if enabled
            if use_ern_layer_id:
                assert "layer_id" in comp_node.extra, (
                    f"ERN layer_id enabled but no layer_id found in extra dict of node {node}"
                )
                node_layer_id = comp_node.extra["layer_id"]
                assert 0 <= node_layer_id < max_ern_layers, (
                    f"Invalid ERN layer_id {node_layer_id} for node {node}, should be in [0, {max_ern_layers})"
                )
                seq_layer_ids.append(node_layer_id)

        params.at(f"{namespace}/affinity", ref, overwrite=None)

        # store node layer ids as a param array with non_grad tag if enabled
        if use_ern_layer_id:
            seqlayerid_arr = jnp.array(seq_layer_ids)
            assert seqlayerid_arr.shape == (len(nodelist),), (
                f"ERN node layer IDs should have shape ({(len(nodelist),)}), got {seqlayerid_arr.shape}"
            )
            params.at(
                f"{namespace}/node_layer_ids",
                seqlayerid_arr,
                tags=[NON_GRAD_TAG],
            )
            logger.debug(f"Node layer IDs for {local_layer_name}:\n{seqlayerid_arr}")

        # initialize MLP with dummy inputs
        # include dummy one-hot layer id if needed
        layer_id_onehot = jnp.zeros(max_ern_layers) if use_ern_layer_id else np.empty((0,))

        MLP(
            *[np.zeros(shape) for shape in input_shapes],
            affinity=np.zeros((affinity_dim,)),
            random_var=0,
            param_f=partial(init_if_needed, params, base_path="shared"),
            key=key,
            layer_id_onehot=layer_id_onehot,
        )

    def apply(
        *values: ArrayLike,
        random_vars: NDArray,
        params: ParameterTree,
        node_id: ArrayLike,
        key,
        tu_enabled_random_vars: Optional[ArrayLike] = None,
        network_id: Optional[ArrayLike] = None,
        **_kwargs,
    ) -> tuple[ArrayLike, dict]:
        assert len(values) == len(input_shapes)

        affinity = params[f"{namespace}/affinity"][node_id]
        assert affinity.shape == (affinity_dim,)

        qid = params[f"{namespace}/random_variable_id"][node_id]

        # create one-hot encoded layer_id if enabled
        layer_id_onehot = jnp.empty((0,))  # default empty if not using layer_id
        node_layer_id = 0
        if use_ern_layer_id:
            node_layer_id = params[f"{namespace}/node_layer_ids"][node_id]
            layer_id_onehot = jax.nn.one_hot(node_layer_id, max_ern_layers)

        input_tu_indices_path = f"{namespace}/input_tu_indices"
        if input_tu_indices_path in params:
            from biocomp.tumasking import get_tu_masks

            tu_indices = params[input_tu_indices_path][node_id]
            input_masks = get_tu_masks(
                params, tu_indices, tu_enabled_random_vars, network_id, is_multi_tu=True
            )
        else:
            input_masks = jnp.ones(2)

        neg_val, pos_val = values
        neg_enabled, pos_enabled = input_masks[0], input_masks[1]

        def normal_ern():
            return o_activation(
                MLP(
                    neg_val,
                    pos_val,
                    affinity=affinity,
                    random_var=random_vars[qid],
                    param_f=partial(get_param, params, base_path="shared"),
                    key=key,
                    layer_id_onehot=layer_id_onehot,
                )
            )

        def passthrough_pos():
            return jnp.sum(pos_val).reshape(1)

        def zero_output():
            return jnp.zeros(1)

        result = jnp.where(
            pos_enabled > 0.5,
            jnp.where(neg_enabled > 0.5, normal_ern(), passthrough_pos()),
            zero_output(),
        )

        input_diff = jnp.sum(pos_val) - jnp.sum(neg_val)

        aux_dict = {
            "affinity": affinity,
            "random_var": random_vars[qid],
            "node_layer_id": node_layer_id if use_ern_layer_id else None,
            "layer_id_onehot": layer_id_onehot,
            "neg_input": neg_val,
            "pos_input": pos_val,
            "input_diff": input_diff,
            "input_masks": input_masks,
            "neg_enabled": neg_enabled,
            "pos_enabled": pos_enabled,
        }

        if use_ern_layer_id:
            aux_dict["node_layer_id"] = params[f"{namespace}/node_layer_ids"][node_id]

        return result, aux_dict

    output_shape = [(1,)]

    def introspect(
        params: ParameterTree,
        nodelist: list[StackNode],
        stack: ComputeStack,
        network_id: int,
        local_only: bool = True,
    ) -> list:
        from biocomp.paramintrospect import (
            NodeParamInfo,
            ParamValue,
            ParamKind,
            InputSlot,
            get_tu_prob,
            is_tu_enabled,
        )

        result = []
        for node_idx, node in enumerate(nodelist):
            if node.network_id != network_id:
                continue

            comp_node = node.get(stack)
            extra = comp_node.extra
            seq_name = extra.get("seq_name", "unknown")
            node_name = seq_name.split("::")[-1] if "::" in seq_name else seq_name

            affinity_ref = params[f"{namespace}/affinity"]
            if hasattr(affinity_ref, "view"):
                affinity_val = float(np.mean(np.asarray(affinity_ref[node_idx])))
            else:
                affinity_val = float(np.mean(np.asarray(affinity_ref[node_idx])))

            ungrouped = [
                ParamValue(
                    name="affinity",
                    kind=ParamKind.AFFINITY,
                    value=affinity_val,
                )
            ]

            if use_ern_layer_id:
                lid_path = f"{namespace}/node_layer_ids"
                if lid_path in params:
                    lid_val = int(params[lid_path][node_idx])
                    ungrouped.append(
                        ParamValue(
                            name="layer_id",
                            kind=ParamKind.OTHER,
                            value=lid_val,
                        )
                    )

            edges = node.get_incoming_edges(stack)
            edges_sorted = sorted(edges, key=lambda e: e.to_input_slot)

            input_tu_path = f"{namespace}/input_tu_indices"
            tu_indices = None
            if input_tu_path in params:
                tu_arr = params[input_tu_path]
                tu_indices = np.asarray(tu_arr.view() if hasattr(tu_arr, "view") else tu_arr)
                if tu_indices.ndim >= 1 and node_idx < tu_indices.shape[0]:
                    tu_indices = tu_indices[node_idx]

            ungrouped_inputs = []
            input_labels = ["neg", "pos"]
            for slot_idx, edge in enumerate(edges_sorted):
                source = stack.get_node(node.network_id, edge.source_id)
                source_name = source.extra.get("name", f"node_{edge.source_id}") if source else None

                is_masked = False
                if tu_indices is not None:
                    if tu_indices.ndim == 2 and slot_idx < tu_indices.shape[0]:
                        slot_tu_indices = tu_indices[slot_idx]
                        for tidx in slot_tu_indices:
                            tidx = int(tidx)
                            if tidx >= 0:
                                prob = get_tu_prob(params, network_id, tidx)
                                if not is_tu_enabled(prob):
                                    is_masked = True
                                break
                    elif tu_indices.ndim == 1 and slot_idx < len(tu_indices):
                        tidx = int(tu_indices[slot_idx])
                        if tidx >= 0:
                            prob = get_tu_prob(params, network_id, tidx)
                            is_masked = not is_tu_enabled(prob)

                tu_ids_on_edge = edge.extra.get("tu_id", []) if edge.extra else []
                tu_id = tu_ids_on_edge[0] if tu_ids_on_edge else None
                label = (
                    input_labels[slot_idx] if slot_idx < len(input_labels) else f"input_{slot_idx}"
                )

                ungrouped_inputs.append(
                    InputSlot(
                        slot_idx=slot_idx,
                        tu_id=tu_id,
                        is_masked=is_masked,
                        source_node=f"{label}:{source_name}" if source_name else label,
                    )
                )

            result.append(
                NodeParamInfo(
                    node_type="sequestron_ERN",
                    node_name=node_name,
                    network_id=network_id,
                    ungrouped=ungrouped,
                    ungrouped_inputs=ungrouped_inputs,
                )
            )

        return result

    return LayerInstance(prepare, apply, output_shape, introspect=introspect)


ERN_DEFAULT_NEG_PARTS = ["CasE", "Csy4", "PgU"]
ERN_DEFAULT_POS_PARTS = [["CasE_rec"], ["Csy4_rec"], ["PgU_rec"]]
DEFAULT_AVAILABLE_5P_AFFINITIES = []
for i, positive_part in enumerate(ERN_DEFAULT_NEG_PARTS):
    for negative_part in ERN_DEFAULT_POS_PARTS[i]:
        DEFAULT_AVAILABLE_5P_AFFINITIES.append(f"ERN::{positive_part}#{negative_part}")
ERN5p = partial(sequestron_ERN, subtype="5p", affinity_names=DEFAULT_AVAILABLE_5P_AFFINITIES)
