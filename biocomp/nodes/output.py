from biocomp.jaxutils import flat_concat
from biocomp.compute import StackNode, ComputeStack
from jax.tree_util import Partial as partial
from jax.typing import ArrayLike
import jax.numpy as jnp
from jax import vmap
import numpy as np
from biocomp.parameters import ArrayRef, ParameterTree, init_if_needed, make_view, get_param
from biocomp.nodeutils import LayerInstance, add_random_var_ids
from biocomp.neuralutils import (
    ACTIVATION_FUNCTIONS,
    INITIALIZERS,
    DEFAULT_ACTIVATION,
    DEFAULT_OUT_ACTIVATION,
    DEFAULT_INITIALIZER,
    dense_mlp,
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
    **_,
):
    del n_outputs
    del stack

    assert all(shape == input_shapes[0] for shape in input_shapes)
    inner_activation = ACTIVATION_FUNCTIONS[inner_activation_name]
    outer_activation = ACTIVATION_FUNCTIONS[outer_activation_name]
    initializer = INITIALIZERS[initializer_name]

    layer_name = namespace.split("/")[-1]  # extract layer name from namespace

    def MLP_head(x, rng_key, params):
        return dense_mlp(
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

    def apply(
        *inputs: ArrayLike,
        random_vars: NDArray,
        params: ParameterTree,
        node_id: ArrayLike,
        key,
    ) -> tuple[ArrayLike, dict]:
        inputs_arr = jnp.array(inputs)
        assert len(inputs_arr) == len(input_shapes)
        res = vmap(partial(MLP_head, rng_key=key, params=params))(inputs_arr)

        # simple residual connection
        pre = 0.5 * res + 0.5 * inputs_arr
        output = outer_activation(pre)

        return output, {
            "mlp_outputs": res,
            "pre_activation": pre,
            "n_inputs": len(inputs_arr),
            "input_values": inputs_arr,
        }

    output_shape = [(1,)] * len(input_shapes)

    return LayerInstance(prepare, apply, output_shape)


##────────────────────────────────────────────────────────────────────────────}}}##
