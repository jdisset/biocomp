### {{{                          --     imports     --
import random
import numpy as np
from . import datautils as du
from biocomp.compute import ComputeStack, ComputeConfig
from biocomp.utils import (
    EncodedPartialFunction,
    PartialFunction,
    ArbitraryModel,
    PartialFunctionResult,
)
from biocomp.network import Network, CoTransfection, Unit, Slot
from biocomp.train import create_counter
from assertpy import assert_that
from . import nodes as nodes
from .parameters import ParameterTree, ParamPath
from . import utils as ut
import time
from typing import List, Tuple, Callable, Optional, NamedTuple, Union, Literal
from pydantic import Field, BaseModel, ConfigDict
from biocomp.logging_config import get_logger
import optax
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jax import vmap, jit
from .jaxutils import get_looped_slice
import os
from jax.experimental import checkify

from biocomptools.modelmodel import BiocompModel
from biocomp.designutils import sample_from_svg
from biocomp.optimutils import (
    make_training_step,
    per_replicate_step,
    per_replicate_step_nonscan,
    optimize,
    as_schedule,
    OptimConfig,
    DEFAULT_OPTIMIZER,
)

from pathlib import Path
from jax.typing import ArrayLike

logger = get_logger(__name__)

##────────────────────────────────────────────────────────────────────────────}}}


def assert_eq(a, b, msg=None):
    assert a == b, f"Assertion failed: {a} != {b}" if msg is None else msg


## {{{                      --     loss functions     --


def single_sinkhorn_loss(yhat, target_y, epsilon=0.1):
    from ott.geometry import pointcloud
    from ott.solvers import linear

    assert yhat.shape == target_y.shape, f"Shape mismatch: {yhat.shape} != {target_y.shape}"
    geom = pointcloud.PointCloud(jnp.atleast_2d(yhat), jnp.atleast_2d(target_y), epsilon=epsilon)
    ot = linear.solve(geom)
    return ot.reg_ot_cost


def compute_all_losses(yhat, target_y, lossfunc):
    # shape is (batch_size, n_targets, n_outputs)
    assert_that(yhat.shape).is_equal_to(target_y.shape)
    per_network = lossfunc
    per_target = vmap(per_network, in_axes=(-1, -1))
    losses = vmap(per_target, in_axes=(1, 1))(yhat, target_y)
    assert_that(losses.shape).is_equal_to((yhat.shape[1], yhat.shape[2]))
    return losses


def sinkhorn_loss(stack, dconf, dmanager, num_z, epsilon=0.1):
    from ott.geometry import pointcloud
    from ott.solvers import linear

    n_targets = dmanager.n_targets
    n_inputs = stack.get_nb_inputs()
    n_outputs = stack.get_nb_outputs()
    n_networks = stack.get_nb_networks()
    dep_mask = stack.get_dependent_output_mask()
    nb_dep = np.sum(dep_mask)

    assert n_inputs == 2 * n_networks, (
        f"Expected {2 * n_networks} inputs, got {n_inputs}. Not optimizing a 2D design?"
    )

    def per_batch_apply(params, X, Z, keys):
        assert_that(X.shape).is_equal_to((dconf.batch_size, stack.get_nb_inputs()))
        assert_that(Z.shape).is_equal_to((dconf.batch_size, num_z[-1]))
        assert_that(keys.shape).is_equal_to((dconf.batch_size, 2))
        return vmap(stack.apply, in_axes=(None, 0, 0, 0))(params, X, Z, keys)

    def per_target_apply(params, X, Z, keys):
        assert_that(X.shape).is_equal_to((dconf.batch_size, n_targets, stack.get_nb_inputs()))
        assert_that(Z.shape).is_equal_to((dconf.batch_size, n_targets, num_z[-1]))
        assert_that(keys.shape).is_equal_to((dconf.batch_size, n_targets, 2))
        return vmap(per_batch_apply, in_axes=(0, 1, 1, 1), out_axes=1)(params, X, Z, keys)

    def all_losses_func(yhat, target_y, epsilon):
        assert_that(yhat.shape).is_equal_to(target_y.shape)
        assert_that(yhat.ndim).is_equal_to(3)
        assert_that(target_y.ndim).is_equal_to(3)
        all_losses = compute_all_losses(
            yhat, target_y, lambda yhat, target_y: single_sinkhorn_loss(yhat, target_y, epsilon)
        )
        assert_that(all_losses.shape).is_equal_to((n_targets, n_networks))
        return all_losses

    def loss_func(dynamic, static, X, Y, Z, key, step):
        params = ParameterTree.merge(dynamic, static)

        # shape of X: (batch_size, n_targets, n_infeatures)
        # yhat should be of shape (batch_size, n_targets, n_outfeatures)

        assert_that(X.shape).is_equal_to((dconf.batch_size, dmanager.n_targets, n_inputs))
        assert_that(Y.shape).is_equal_to((dconf.batch_size, dmanager.n_targets, n_networks))
        assert_that(Z.shape).is_equal_to((dconf.batch_size, dmanager.n_targets, num_z[-1]))

        keys = jax.random.split(key, (X.shape[0], X.shape[1]))
        yhat, (apply_aux, full_output) = per_target_apply(params, X, Z, keys)
        assert_that(yhat.shape).is_equal_to((*X.shape[:2], n_outputs))

        yhatdep = jnp.compress(dep_mask, yhat, axis=-1, size=nb_dep)
        assert_that(yhatdep.shape).is_equal_to(Y.shape)

        all_losses = all_losses_func(yhatdep, Y, as_schedule(epsilon)(step))

        avgloss = all_losses.mean()

        aux = {"apply_aux": apply_aux, "all_losses": all_losses}

        return avgloss, aux

    return loss_func


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                     --     helper functions     --


def get_ind_params(params, target_id, ind_id):
    return jax.tree.map(lambda x: x[target_id, ind_id], params)


def plot_prediction(
    design_config,
    params,
    target_id,
    ind_id,
    net_id,
    target_x,
    target_y,
    key,
    stack,
    num_z,
    dep_output_mask,
    max_evals=30000,
):
    """Plot the prediction for a given target and individual."""
    import matplotlib.pyplot as plt

    params_ind = get_ind_params(params, target_id, ind_id)
    t_x = target_x[:, target_id]
    t_x = t_x.reshape(-1, t_x.shape[-1])[:max_evals]
    t_y = target_y[:, target_id]
    t_y = t_y.reshape(-1, t_y.shape[-1])[:max_evals]

    z = jax.random.uniform(key, (*t_x.shape[:-1], num_z))

    t_yhat = design_config.forward(params_ind, t_x, z, key, stack)
    t_yhatdep = t_yhat[..., dep_output_mask]
    assert_that(t_yhatdep.shape).is_equal_to(t_y.shape)

    loss_value = single_l2loss(t_yhatdep[:, net_id], t_y[:, net_id])

    t_x_net = t_x[:, 2 * net_id : 2 * net_id + 2]

    # 2 subplots (ground truth and prediction)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    scatter1 = ax1.scatter(
        t_x_net[:, 0], t_x_net[:, 1], c=t_y[:, net_id], s=1, cmap="viridis", vmin=0, vmax=1
    )
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Ground Truth")
    ax1.set_aspect("equal")
    plt.colorbar(scatter1, ax=ax1)
    scatter2 = ax2.scatter(
        t_x_net[:, 0], t_x_net[:, 1], c=t_yhatdep[:, net_id], s=1, cmap="viridis", vmin=0, vmax=1
    )
    ax2.set_title("Prediction")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_aspect("equal")
    plt.colorbar(scatter2, ax=ax2)
    ax2.set_title(f"Prediction (Loss: {loss_value:.4f})")
    plt.tight_layout()


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                           --     top k     --


def get_top_k_indices(losses, n_targets, popsize, num_nets, k=1):
    """
    Get the indices of the top k losses for each target, returning two arrays:
    - indices: shape (n_targets, k, 2), each row is (i, j)
    - values: shape (n_targets, k, 1), each row is the corresponding loss value
    """
    assert_that(losses.shape).is_equal_to((n_targets, popsize, num_nets))

    indices_per_target = []
    values_per_target = []
    for t in range(n_targets):
        flat = losses[t].reshape(-1)  # flatten the 2D (i,j) losses
        if k >= flat.size:
            topk_flat = jnp.argsort(flat)  # if k >= total pairs, just sort everything
        else:
            topk_flat = jnp.argpartition(flat, k - 1)[:k]
            topk_flat = topk_flat[jnp.argsort(flat[topk_flat])]

        # recover (i, j) from flat indices
        indices = []
        values = []
        for idx in topk_flat:
            i, j = jnp.unravel_index(idx, (popsize, num_nets))
            indices.append((i, j))
            values.append([losses[t, i, j]])
        indices_per_target.append(indices)
        values_per_target.append(values)

    indices_arr = jnp.array(indices_per_target)
    values_arr = jnp.array(values_per_target)

    assert_that(indices_arr.shape).is_equal_to((n_targets, k, 2))
    assert_that(values_arr.shape).is_equal_to((n_targets, k, 1))

    return indices_arr, values_arr


# usage: topk_ids, topk_losses = get_top_k_indices( all_losses, design_config.n_targets, design_config.population_size, NUM_NETS, k=3)
# target_id, ind_id, net_id = 0, *topk_ids[0, 0]


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                          --     design manager   --
DEFAULT_RESCALE_TARGET = {
    "x": (0.0, 0.6),
    "y": (0.0, 0.6),
    "out": (0.0, 0.6),
}


class Target(NamedTuple):
    path: Union[str, Path]
    name: Optional[str] = None
    rescale_to: dict = DEFAULT_RESCALE_TARGET
    xlim: tuple[float, float] = (0.0, 1.0)
    ylim: tuple[float, float] = (0.0, 1.0)
    outlim: tuple[float, float] = (0.0, 1.0)
    transform_to_log_space: bool = False
    max_is_black: bool = True


class DesignManager(BaseModel):
    """Handles loading and sampling of 2d design target data."""

    targets: list[Target]
    networks: List[Network]

    def get_samples(
        self,
        samples: int | tuple[int, ...],
        seed: Optional[int | ArrayLike] = None,
    ) -> tuple[jax.Array, jax.Array]:
        # returns (x, y) where x and y are arrays of shape (n_samples, n_targets, n_features)
        xsamples = []
        ysamples = []

        if isinstance(samples, int):
            samples = (samples,)

        requested_shape = samples

        n = np.prod(requested_shape)
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        elif isinstance(seed, ArrayLike):  # it's a JAX PRNG key
            seed = int(jax.random.randint(seed, (), 0, jnp.iinfo(jnp.int32).max))

        for target in self.targets:
            xsample, ysample = sample_from_svg(
                target.path,
                n=n,
                seed=seed,
                log=target.transform_to_log_space,
                xlim=target.xlim,
                ylim=target.ylim,
                outlim=target.outlim,
                rescale_to=target.rescale_to,
                max_is_black=target.max_is_black,
            )

            xsamples.append(xsample)
            ysamples.append(ysample)

        xsamples = jnp.stack(xsamples, axis=1)
        ysamples = jnp.stack(ysamples, axis=1)

        assert_that(xsamples.shape).is_equal_to((n, len(self.targets), 2))  # 2d inputs
        assert_that(ysamples.shape).is_equal_to((n, len(self.targets), 1))  # 1d outputs

        xsamples = xsamples.reshape(*requested_shape, len(self.targets), -1)
        ysamples = ysamples.reshape(*requested_shape, len(self.targets), -1)

        return xsamples, ysamples

    def build_stack(self, model: BiocompModel):
        stack = ComputeStack(networks=self.networks)
        stack.build(model.compute_config)
        return stack

    @property
    def n_targets(self):
        return len(self.targets)


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                   --     param initialization     --


def initialize_params(stack, n_replicates, n_targets, shared_params, key):
    # could be faster if we stacked copies of the shared parameters and did the merge on the whole stack...
    # good enough for now
    def init_single(k):
        params = stack.init(k)
        _, nonshared = params.filter_by_tag(["shared"])
        return ParameterTree.merge(shared_params, nonshared)

    def init_target_params(k):
        params = vmap(init_single)(jax.random.split(k, n_targets))
        return params

    return vmap(init_target_params)(jax.random.split(key, n_replicates))


class DesignConfig(OptimConfig):
    loss_function: EncodedPartialFunction = Field(default=sinkhorn_loss)
    n_replicates: int = 4


##────────────────────────────────────────────────────────────────────────────}}}


def assert_tree_shape(tree, expected_shape, only_first_dims=True):
    """Assert that the shape of each leaf in the tree matches the expected shape."""

    N_DIMS = len(expected_shape)

    def check_shape(x):
        if isinstance(x, jax.Array):
            assert_that(x.shape[:N_DIMS] if only_first_dims else x.shape).is_equal_to(
                expected_shape
            )
        return x

    jax.tree.map(check_shape, tree)


### {{{                       --     main design function     --


def start(
    dmanager: DesignManager,
    dconf: DesignConfig,
    model: BiocompModel,
    loggers: Optional[List[Tuple[int, Callable]]] = None,
    async_handler=None,
):
    pkey, bkey, loop_key = jax.random.split(dconf.seed_key, 3)

    # -- initializations --
    stack = dmanager.build_stack(model)
    initial_params = initialize_params(
        stack, dconf.n_replicates, dmanager.n_targets, model.shared_params, pkey
    )
    assert_tree_shape(initial_params, (dconf.n_replicates, dmanager.n_targets))
    static, dynamic = initial_params.filter_by_tag(["non_grad", "shared"])
    initial_optimizer_state = vmap(vmap(dconf.optimizer.init))(dynamic)

    # -- get data --
    num_z = static["global/number_of_quantile_variables"]
    assert_that(num_z.shape[0]).is_equal_to(dconf.n_replicates)
    assert_that(jnp.all(num_z == num_z[0])).is_true()
    print(f"Number of quantile variables: {num_z}")
    num_z = (dmanager.n_targets, int(num_z.ravel()[0].squeeze()))

    n_batches_per_epoch = min(
        int((dconf.n_batches_per_epoch / dconf.batches_per_step) * dconf.batches_per_step), 1
    )
    steps_per_epoch = max(1, n_batches_per_epoch // dconf.batches_per_step)
    total_steps = int(dconf.n_epochs * steps_per_epoch)
    n_networks = stack.get_nb_networks()
    xbatches, ybatches = dmanager.get_samples(
        (n_networks, steps_per_epoch, dconf.batches_per_step, dconf.n_replicates, dconf.batch_size),
        bkey,
    )

    # glue all networks along the last dimension
    xbatches = jnp.concatenate(xbatches, axis=-1)
    ybatches = jnp.concatenate(ybatches, axis=-1)

    n_inputs = stack.get_nb_inputs()
    n_outputs = stack.get_nb_outputs()

    assert_that(xbatches.shape).is_equal_to(
        (
            steps_per_epoch,
            dconf.batches_per_step,
            dconf.n_replicates,
            dconf.batch_size,
            dmanager.n_targets,
            n_inputs,
        )
    )

    # -- step function --
    loss_func = dconf.loss_function.get_impl()(stack, dconf, dmanager, num_z=num_z)
    step_fn = make_training_step(
        loss_func,
        dconf.optimizer,
        dconf.keep_in_history,
        scannable=True,
        static_tags=["non_grad", "shared"],
    )

    def step(params: ParameterTree, opt_state: optax.OptState, step_key, xs, ys):
        keys = jax.random.split(step_key, dconf.n_replicates)
        assert_that(xs.shape).is_equal_to(
            (
                dconf.n_replicates,
                dconf.batches_per_step,
                dconf.batch_size,
                dmanager.n_targets,
                n_inputs,
            )
        )
        assert_that(ys.shape).is_equal_to(
            (
                dconf.n_replicates,
                dconf.batches_per_step,
                dconf.batch_size,
                dmanager.n_targets,
                n_networks,  # it' only dependent outputs, so one per network
            )
        )
        assert_tree_shape(params, (dconf.n_replicates, dmanager.n_targets))
        assert_tree_shape(opt_state, (dconf.n_replicates, dmanager.n_targets))

        return jax.vmap(
            Partial(per_replicate_step, num_z=num_z, training_config=dconf, scannable_step=step_fn)
        )(params, opt_state, keys, xs, ys)

    return optimize(
        step,
        initial_params,
        initial_optimizer_state,
        xbatches=xbatches,
        ybatches=ybatches,
        config=dconf,
        n_total_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
        key=loop_key,
        stack=stack,
        loggers=loggers,
        async_handler=async_handler,
    )


##────────────────────────────────────────────────────────────────────────────}}}
