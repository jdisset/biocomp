### {{{                          --     imports     --
from . import datautils as du
from . import utils as ut
from . import nodes as nodes
from .parameters import ParameterTree
from biocomp.logging_config import get_logger

logger = get_logger(__name__)

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                     --     helper functions     --


def init_stack(
    compute_config,
    datamanager: du.DataManager,
    n_replicates: int,
    key,
):
    import jax

    stack = datamanager.build_compute_stack(compute_config)
    assert stack.init is not None
    from jax import vmap

    with ut.timer("Stack initialization", logger):
        params = vmap(stack.init)(jax.random.split(key, n_replicates))
    return stack, params


def generate_batches(
    datamanager: du.DataManager,
    n_replicates: int,
    n_batches: int,
    batch_size: int,
    key,
):
    total_n_batches = n_replicates * n_batches

    with ut.timer("Generating batches", logger):
        xbatches, ybatches = datamanager.get_batches(total_n_batches, batch_size, key)
    # current shape is (R*B,N,F), final shape should be (R,B,N,F)
    # R: replicates, B: batches, N: data, F: features
    xbatches = xbatches.reshape(n_replicates, n_batches, *xbatches.shape[1:])
    ybatches = ybatches.reshape(n_replicates, n_batches, *ybatches.shape[1:])

    assert xbatches.shape[:-1] == (
        n_replicates,
        n_batches,
        batch_size,
    )
    assert ybatches.shape[:-1] == (
        n_replicates,
        n_batches,
        batch_size,
    )

    return xbatches, ybatches


def make_training_step(loss_func, optimizer, fields_to_keep_in_history=("loss",), scannable=True):
    from jax import value_and_grad
    import optax

    def base_training_step(params, opt_state, x, y, z, key):
        static, dynamic = params.filter_by_tag(["non_grad", "local"])

        (loss, aux), grads = value_and_grad(loss_func, has_aux=True)(
            dynamic, static, x, y, z, key, opt_state[0].count
        )

        updates, opt_state = optimizer.update(grads, opt_state, dynamic)
        dynamic = optax.apply_updates(dynamic, updates)
        params = ParameterTree.merge(static, dynamic)
        res = {
            "params": params,
            "loss": loss,
            "grad": grads,
            "opt": opt_state,
            "x": x,
            "y": y,
            "z": z,
            "key": key,
            **aux,
        }
        return res

    training_step = base_training_step

    if scannable:

        def scannable_training_step(carry, i_x_y_z_k):
            params, opt_state = carry
            i, x, y, z, k = i_x_y_z_k
            updt = base_training_step(params, opt_state, x, y, z, k)
            params, opt_state = updt["params"], updt["opt"]
            history = {k: updt[k] for k in fields_to_keep_in_history}
            return (params, opt_state), history

        training_step = scannable_training_step

    return training_step


##────────────────────────────────────────────────────────────────────────────}}}
