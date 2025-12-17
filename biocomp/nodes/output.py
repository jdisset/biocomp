from biocomp.compute import StackNode, ComputeStack
from jax.tree_util import Partial as partial
from jax.typing import ArrayLike
import jax.numpy as jnp
from jax import vmap
import numpy as np
from biocomp.parameters import ParameterTree, init_if_needed
from biocomp.nodeutils import LayerInstance, add_tu_input_mapping
from biocomp.tumasking import TU_LOG_ALPHA_PATH
from typing import Optional

from biocomp.neuralutils import (
    ACTIVATION_FUNCTIONS,
    INITIALIZERS,
    DEFAULT_ACTIVATION,
    DEFAULT_OUT_ACTIVATION,
    DEFAULT_INITIALIZER,
    dense_mlp,
    dummy_mlp,
)

PRNGKey = ArrayLike
NDArray = np.ndarray | jnp.ndarray

### {{{                    --     output (fluorescence) node     --


def grouped_output(
    input_shapes: list[tuple[int, ...]],
    n_outputs: int,  # unused
    stack: ComputeStack,  # unused
    namespace: str,
    wsize: int = 64,
    depth: int = 4,
    bias_offset: float = 0.0,
    inner_activation_name: str = DEFAULT_ACTIVATION,
    outer_activation_name: str = DEFAULT_OUT_ACTIVATION,
    initializer_name: str = DEFAULT_INITIALIZER,
    dummy: bool = False,  # disable neural + residual, for testing
    **_,
):
    del n_outputs
    # stack is used for TU mapping in prepare()

    assert all(shape == input_shapes[0] for shape in input_shapes)
    inner_activation = ACTIVATION_FUNCTIONS[inner_activation_name]
    outer_activation = ACTIVATION_FUNCTIONS[outer_activation_name]
    initializer = INITIALIZERS[initializer_name]

    mlp = dummy_mlp if dummy else dense_mlp

    def MLP_head(x, rng_key, params):
        return mlp(
            x,
            wsize,
            1,
            depth,
            param_f=partial(init_if_needed, params, base_path="shared"),
            initializer=initializer,
            bias_offset=bias_offset,
            key=rng_key,
            name="NN/grouped_output",
            activation=inner_activation,
        )

    def prepare(params: ParameterTree, nodelist: list[StackNode], key: PRNGKey):
        MLP_head(x=np.zeros(input_shapes[0]), rng_key=key, params=params)
        add_tu_input_mapping(params, stack, nodelist, namespace)

    def apply(
        *inputs: ArrayLike,
        random_vars: NDArray,
        params: ParameterTree,
        node_id: ArrayLike,
        key,
        tu_enabled_random_vars: Optional[ArrayLike] = None,
        network_id: Optional[ArrayLike] = None,
        **_kwargs,
    ) -> tuple[ArrayLike, dict]:
        inputs_arr = jnp.array(inputs)
        assert len(inputs_arr) == len(input_shapes)

        input_tu_indices_path = f"{namespace}/input_tu_indices"
        if input_tu_indices_path in params:
            from biocomp.tumasking import compute_input_masks

            tu_indices = params[input_tu_indices_path][node_id]
            tu_log_alpha_full = params[TU_LOG_ALPHA_PATH] if TU_LOG_ALPHA_PATH in params else None
            tu_log_alpha = None
            if tu_log_alpha_full is not None:
                assert tu_log_alpha_full.ndim == 2, (
                    f"tu_log_alpha must be 2D (n_networks, n_tus), got {tu_log_alpha_full.ndim}D"
                )
                assert network_id is not None, "network_id required for per-network TU masking"
                tu_log_alpha = tu_log_alpha_full[network_id]
            input_masks = compute_input_masks(tu_indices, tu_enabled_random_vars, tu_log_alpha)
        else:
            input_masks = jnp.ones(len(input_shapes))

        res = vmap(partial(MLP_head, rng_key=key, params=params))(inputs_arr)

        masked_inputs = inputs_arr * input_masks.reshape(-1, *([1] * len(input_shapes[0])))
        masked_res = res * input_masks.reshape(-1, 1)

        if dummy:
            pre = masked_res
            output = masked_res
        else:
            pre = 0.5 * masked_res + 0.5 * masked_inputs
            output = outer_activation(pre)

        return output, {
            "mlp_outputs": res,
            "pre_activation": pre,
            "n_inputs": len(inputs_arr),
            "input_values": inputs_arr,
            "input_masks": input_masks,
        }

    output_shape = [(1,)] * len(input_shapes)

    return LayerInstance(prepare, apply, output_shape)


##────────────────────────────────────────────────────────────────────────────}}}##
