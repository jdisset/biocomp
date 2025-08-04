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
import biocomp.utils
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


## {{{                      --     loss functions     --


def bw_sliced_wasserstein_loss(
    x: jnp.ndarray,
    y: jnp.ndarray,
    yhat: jnp.ndarray,
    *,
    n_proj: int = 128,
    p: int = 2,
    rng: "jax.Array | None" = None,  # e.g. jax.random.key(0) for reproducibility
):
    """
    Sliced-Wasserstein distance between target and prediction on the same support x.
    Faster, lower-variance gradients than full OT for many settings.
    """
    import jax.numpy as jnp
    from ott.tools import sliced

    # intensities -> non-negative weights, sum to 1 (pred vs target)
    a = jnp.clip(yhat, 0.0)
    a /= jnp.maximum(a.sum(), 1e-12)
    b = jnp.clip(y, 0.0)
    b /= jnp.maximum(b.sum(), 1e-12)

    # Same coordinates for both measures; only weights differ
    # Pass number of projections and optional rng via kwargs
    loss, _ = sliced.sliced_wasserstein(
        x=x,
        y=x,
        a=a,
        b=b,
        # kwargs go to the default projector: random directions on the sphere
        n_proj=n_proj,
        rng=rng,
        p=p,
    )
    return loss


def bw_unbalanced_sinkhorn_div_loss(
    x: jnp.ndarray,
    y: jnp.ndarray,
    yhat: jnp.ndarray,
    *,
    epsilon: float = 1e-2,
    tau: float = 0.995,  # 1.0 => balanced; <1.0 => unbalanced
    **solver_kwargs,
):
    """
    Debiased (Sinkhorn-divergence-style) loss with unbalanced marginals controlled by `tau`.
    """

    import jax.numpy as jnp
    from ott.geometry import pointcloud
    from ott.problems.linear import linear_problem
    from ott.solvers.linear import sinkhorn

    # weights
    a = jnp.clip(yhat, 0.0)
    a /= jnp.maximum(a.sum(), 1e-12)
    b = jnp.clip(y, 0.0)
    b /= jnp.maximum(b.sum(), 1e-12)

    # Single geometry since supports coincide
    geom = pointcloud.PointCloud(x, x, epsilon=epsilon)
    solver = sinkhorn.Sinkhorn(**solver_kwargs)

    # cross term OT_ε(μ, ν; tau)
    prob_xy = linear_problem.LinearProblem(geom, a=a, b=b, tau_a=tau, tau_b=tau)
    ot_xy = solver(prob_xy).reg_ot_cost

    # self terms OT_ε(μ, μ; tau) and OT_ε(ν, ν; tau)
    prob_aa = linear_problem.LinearProblem(geom, a=a, b=a, tau_a=tau, tau_b=tau)
    prob_bb = linear_problem.LinearProblem(geom, a=b, b=b, tau_a=tau, tau_b=tau)
    ot_aa = solver(prob_aa).reg_ot_cost
    ot_bb = solver(prob_bb).reg_ot_cost

    # debias (Sinkhorn divergence)
    return ot_xy - 0.5 * (ot_aa + ot_bb)


def bw_energy_distance(
    x: jnp.ndarray,
    y: jnp.ndarray,
    yhat: jnp.ndarray,
    **kwargs,
):
    """Weighted energy distance between target and prediction."""
    # normalised non-negative weights
    a = jnp.clip(y, 0.0)
    b = jnp.clip(yhat, 0.0)
    a /= jnp.maximum(a.sum(), 1e-12)
    b /= jnp.maximum(b.sum(), 1e-12)

    # pairwise distances
    d_xy = jnp.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)

    # cross term (target vs prediction)
    cross = jnp.sum(a[:, None] * b[None, :] * d_xy)

    # self terms
    self_a = jnp.sum(a[:, None] * a[None, :] * d_xy)
    self_b = jnp.sum(b[:, None] * b[None, :] * d_xy)

    return 2 * cross - self_a - self_b


def bw_mse_loss(
    x: jnp.ndarray,
    y: jnp.ndarray,
    yhat: jnp.ndarray,
    **kwargs,
):
    diff = yhat - y
    return jnp.mean(diff**2)


def bw_sinkhorn_loss(
    x: jnp.ndarray,  # (n, 2) sample coordinates
    y: jnp.ndarray,  # (n,)   target intensities
    yhat: jnp.ndarray,  # (n,)   predicted intensities
    epsilon: float = 0.01,
    **sink_kwargs,
):
    """
    Debiased Sinkhorn divergence between a black-and-white target and prediction.
    """

    from ott.geometry import pointcloud
    from ott.tools import sinkhorn_divergence as sd

    assert_that(x).has_shape((y.shape[0], 2))
    assert_that(y).has_same_shape(yhat)
    assert_that(y.ndim).is_equal_to(1)

    # intensities -> non-negative weights, sum to 1
    a = jnp.clip(yhat, 0.0)
    b = jnp.clip(y, 0.0)
    a /= jnp.maximum(a.sum(), 1e-12)
    b /= jnp.maximum(b.sum(), 1e-12)

    # using same support (x). Only the weights change.
    divergence, _ = sd.sinkhorn_divergence(
        pointcloud.PointCloud,
        x=x,
        y=x,
        a=a,  # input weights aka intensities
        b=b,  # target weights
        epsilon=epsilon,
        **sink_kwargs,
    )

    return divergence


@Partial(jax.jit, static_argnames=["lossfunc", "n_inputs_per_network"])
def compute_all_losses(x, y, yhatdep, lossfunc: Callable, n_inputs_per_network=2):
    n_networks = int(x.shape[-1] / n_inputs_per_network)
    n_inputs = n_networks * n_inputs_per_network

    batch_size = y.shape[0]
    n_targets = y.shape[1]

    # shape is (batch_size, n_targets, n_outputs)
    assert_that(x).has_shape((y.shape[0], y.shape[1], n_inputs))
    assert_that(yhatdep).has_shape((batch_size, n_targets, n_networks))
    assert_that(y).has_same_shape(yhatdep)

    # now we need to split the inputs into the different networks
    xsplit = jnp.reshape(x, (batch_size, n_targets, n_networks, n_inputs_per_network))

    per_target = vmap(lossfunc, in_axes=(1, 1, 1))
    losses = vmap(per_target, in_axes=(1, 1, 1))(xsplit, yhatdep, y)
    assert_that(losses).has_shape((n_targets, n_networks))

    return losses


def per_batch_apply(params, X, Z, keys, stack):
    return vmap(stack.apply, in_axes=(None, 0, 0, 0))(params, X, Z, keys)


def per_target_apply(params, X, Z, keys, stack):
    return vmap(Partial(per_batch_apply, stack=stack), in_axes=(0, 1, 1, 1), out_axes=1)(
        params, X, Z, keys
    )


@Partial(jit, static_argnames=["stack"])
def per_replicate_apply(params, X, Z, keys, stack):
    return vmap(Partial(per_target_apply, stack=stack))(params, X, Z, keys)


def distance_loss(stack, dconf, dmanager, num_z, epsilon=0.01, distance_func=bw_sinkhorn_loss):
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

    def all_losses_func(x, target_y, yhat, epsilon):
        assert_that(x.shape).is_equal_to((dconf.batch_size, n_targets, n_inputs))
        assert_that(target_y).has_same_shape(yhat)
        assert_that(x.ndim).is_equal_to(3)
        all_losses = compute_all_losses(
            x,
            target_y,
            yhat,
            Partial(distance_func, epsilon=epsilon),
        )
        assert_that(all_losses).has_shape((n_targets, n_networks))
        return all_losses

    def loss_func(dynamic, static, X, Y, Z, key, step):
        params = ParameterTree.merge(dynamic, static)

        # shape of X: (batch_size, n_targets, n_infeatures)
        # yhat should be of shape (batch_size, n_targets, n_outfeatures)

        assert_that(X.shape).is_equal_to((dconf.batch_size, n_targets, n_inputs))
        assert_that(Y.shape).is_equal_to((dconf.batch_size, n_targets, n_networks))
        assert_that(Z.shape).is_equal_to((dconf.batch_size, n_targets, num_z[-1]))

        keys = jax.random.split(key, (X.shape[0], X.shape[1]))
        yhat, (apply_aux, full_output) = per_target_apply(params, X, Z, keys, stack)
        assert_that(yhat.shape).is_equal_to((*X.shape[:2], n_outputs))

        yhatdep = jnp.compress(dep_mask, yhat, axis=-1, size=nb_dep)
        assert_that(yhatdep.shape).is_equal_to(Y.shape)

        all_losses = all_losses_func(X, Y, yhatdep, as_schedule(epsilon)(step))
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


## {{{                          --     design manager   --
DEFAULT_RESCALE_TARGET = {
    "x": (0.0, 0.6),
    "y": (0.0, 0.6),
    "out": (0.0, 0.5),
}


class Target(BaseModel):
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
    loss_function: EncodedPartialFunction = Field(default=distance_loss)
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


## {{{                   --     evaluation and analysis     --


def sample_for_evaluation(
    dmanager: DesignManager,
    dconf: DesignConfig,
    final_params: ParameterTree,
    n_eval_samples: int = 5000,
    key: Optional[ArrayLike] = None,
) -> Tuple[jax.Array, jax.Array]:
    """Sample data for evaluation of trained design parameters.

    Returns:
        xraw: shape (n_networks, n_replicates, n_eval_samples, n_targets, 2)
        yraw: shape (n_networks, n_replicates, n_eval_samples, n_targets, 1)
    """
    if key is None:
        key = jax.random.key(0)

    n_networks = len(dmanager.networks)

    # sample evaluation data
    xraw, yraw = dmanager.get_samples((n_networks, dconf.n_replicates, n_eval_samples), key)

    # assertions
    assert_that(xraw).has_shape(
        (n_networks, dconf.n_replicates, n_eval_samples, dmanager.n_targets, 2)
    )
    assert_that(yraw).has_shape(
        (n_networks, dconf.n_replicates, n_eval_samples, dmanager.n_targets, 1)
    )

    return xraw, yraw


def evaluate_design(
    dmanager: DesignManager,
    dconf: DesignConfig,
    model: BiocompModel,
    final_params: ParameterTree,
    xraw: jax.Array,
    yraw: jax.Array,
    key: Optional[ArrayLike] = None,
    max_eval_size: int = 1000,
) -> Tuple[jax.Array, jax.Array]:
    """Evaluate design performance on sampled data.

    Args:
        dmanager: Design manager with networks and targets
        dconf: Design configuration
        model: Trained biocomp model
        final_params: Final optimized parameters
        xraw: Input samples shape (n_networks, n_replicates, n_eval_samples, n_targets, 2)
        yraw: Target samples shape (n_networks, n_replicates, n_eval_samples, n_targets, 1)
        key: JAX random key
        max_eval_size: Maximum number of samples to process at once (for memory efficiency)

    Returns:
        yhatdep: Predictions shape (n_replicates, n_eval_samples, n_targets, n_networks)
        losses: Loss values shape (n_replicates, n_targets, n_networks)
    """
    if key is None:
        key = jax.random.key(0)

    n_networks = len(dmanager.networks)
    n_eval_samples = xraw.shape[2]

    # reshape inputs for batch processing
    X = jnp.concatenate(xraw, axis=-1)
    Y = jnp.concatenate(yraw, axis=-1)
    assert_that(X).has_shape(
        (dconf.n_replicates, n_eval_samples, dmanager.n_targets, 2 * n_networks)
    )
    assert_that(Y).has_shape((dconf.n_replicates, n_eval_samples, dmanager.n_targets, n_networks))

    # get quantile variable size
    num_z = int(final_params["global/number_of_quantile_variables"].ravel()[0])

    # build stack once
    stack = dmanager.build_stack(model)
    n_outputs = stack.get_nb_outputs()
    dep_mask = stack.get_dependent_output_mask()

    # determine chunk size
    chunk_size = min(max_eval_size, n_eval_samples)
    n_chunks = (n_eval_samples + chunk_size - 1) // chunk_size  # ceiling division

    # process in chunks if needed
    if n_chunks > 1:
        # we'll accumulate predictions in chunks
        yhatdep_chunks = []

        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, n_eval_samples)
            actual_chunk_size = end_idx - start_idx

            # slice the data for this chunk
            X_chunk = X[:, start_idx:end_idx]
            Y_chunk = Y[:, start_idx:end_idx]

            # generate random quantile variables for this chunk
            z_shape = (dconf.n_replicates, actual_chunk_size, dmanager.n_targets, num_z)
            Z_chunk = jax.random.uniform(jax.random.fold_in(key, chunk_idx), z_shape)

            # apply model on chunk
            chunk_keys = jax.random.split(
                jax.random.fold_in(key, chunk_idx + 1000), X_chunk.shape[:-1]
            )
            YHAT_chunk, _ = per_replicate_apply(final_params, X_chunk, Z_chunk, chunk_keys, stack)
            assert_that(YHAT_chunk).has_shape(
                (dconf.n_replicates, actual_chunk_size, dmanager.n_targets, n_outputs)
            )

            # extract dependent outputs for chunk
            yhatdep_chunk = jnp.compress(dep_mask, YHAT_chunk, axis=-1, size=sum(dep_mask))
            assert_that(yhatdep_chunk).has_shape(
                (dconf.n_replicates, actual_chunk_size, dmanager.n_targets, n_networks)
            )

            yhatdep_chunks.append(yhatdep_chunk)

        # concatenate all chunks
        yhatdep = jnp.concatenate(yhatdep_chunks, axis=1)
        assert_that(yhatdep).has_shape(
            (dconf.n_replicates, n_eval_samples, dmanager.n_targets, n_networks)
        )

    else:
        # process all at once (original implementation)
        z_shape = (dconf.n_replicates, n_eval_samples, dmanager.n_targets, num_z)
        Z = jax.random.uniform(key, z_shape)

        YHAT, _ = per_replicate_apply(
            final_params, X, Z, jax.random.split(key, X.shape[:-1]), stack
        )
        assert_that(YHAT).has_shape(
            (dconf.n_replicates, n_eval_samples, dmanager.n_targets, n_outputs)
        )

        yhatdep = jnp.compress(dep_mask, YHAT, axis=-1, size=sum(dep_mask))
        assert_that(yhatdep).has_shape(
            (dconf.n_replicates, n_eval_samples, dmanager.n_targets, n_networks)
        )

    # compute losses (this should be relatively lightweight)
    loss_func = dconf.loss_function.kwargs.get("distance_func", bw_sinkhorn_loss)
    losses = vmap(Partial(compute_all_losses, lossfunc=loss_func))(X, Y, yhatdep)
    assert_that(losses).has_shape((dconf.n_replicates, dmanager.n_targets, n_networks))

    return yhatdep, losses


def get_topk_replicate_network_pairs(
    losses: jax.Array,
    dmanager: DesignManager,
    dconf: DesignConfig,
    k: int = 1,
) -> List[List[Tuple[int, int, float]]]:
    """Find top-k replicate/network pairs with lowest loss for each target.

    Args:
        losses: Loss values shape (n_replicates, n_targets, n_networks)
        dmanager: Design manager with networks and targets
        dconf: Design configuration
        k: Number of top pairs to return per target

    Returns:
        List of lists, one per target, each containing k tuples of (replicate_id, network_id, loss_value)
    """
    n_replicates, n_targets, n_networks = losses.shape
    assert_that(n_replicates).is_equal_to(dconf.n_replicates)
    assert_that(n_targets).is_equal_to(dmanager.n_targets)
    assert_that(n_networks).is_equal_to(len(dmanager.networks))
    assert_that(k).is_less_than_or_equal_to(n_replicates * n_networks)

    best_per_target = []
    for tid in range(n_targets):
        tlosses = losses[:, tid, :]  # shape: (n_replicates, n_networks)
        flat_tlosses = tlosses.reshape((-1,))  # shape: (n_replicates * n_networks,)
        topk_flat_idx = jnp.argsort(flat_tlosses)[:k]

        # convert flat indices back to (replicate_id, network_id)
        rep_ids, net_ids = jnp.unravel_index(topk_flat_idx, (n_replicates, n_networks))
        topk_pairs = [
            (int(rep_ids[j]), int(net_ids[j]), float(flat_tlosses[topk_flat_idx[j]]))
            for j in range(k)
        ]
        best_per_target.append(topk_pairs)

    return best_per_target


def plot_design_results(
    dmanager: DesignManager,
    dconf: DesignConfig,
    xraw: jax.Array,
    yraw: jax.Array,
    yhatdep: jax.Array,
    topk: List[List[Tuple[int, int, float]]],
    n_eval_samples: Optional[int] = None,
    save_dir: Optional[Path] = None,
    show_difference: bool = False,
) -> None:
    """Plot design results for each target showing best replicate/network combination.

    Args:
        dmanager: Design manager with networks and targets
        dconf: Design configuration
        xraw: Input samples shape (n_networks, n_replicates, n_eval_samples, n_targets, 2)
        yraw: Target samples shape (n_networks, n_replicates, n_eval_samples, n_targets, 1)
        yhatdep: Predictions shape (n_replicates, n_eval_samples, n_targets, n_networks)
        topk: Top-k results from get_topk_replicate_network_pairs
        n_eval_samples: Maximum number of samples to plot (for performance)
        save_dir: Directory to save figures (if None, just display)
    """
    import matplotlib.pyplot as plt
    from biocomp.plotting.plotting_core import DEFAULT_CMAP_NAME

    if n_eval_samples is None:
        n_eval_samples = xraw.shape[2]
    else:
        n_eval_samples = min(n_eval_samples, xraw.shape[2])

    # validate shapes
    n_networks = len(dmanager.networks)
    assert_that(xraw).has_shape(
        (n_networks, dconf.n_replicates, xraw.shape[2], dmanager.n_targets, 2)
    )
    assert_that(yraw).has_shape(
        (n_networks, dconf.n_replicates, yraw.shape[2], dmanager.n_targets, 1)
    )
    assert_that(yhatdep).has_shape(
        (dconf.n_replicates, yhatdep.shape[1], dmanager.n_targets, n_networks)
    )

    for tid, target in enumerate(dmanager.targets):
        rep_id, net_id, loss_val = topk[tid][0]  # best for this target

        # get data for this specific target/network/replicate combo
        x_target = xraw[net_id, rep_id, :n_eval_samples, tid]  # shape: (n_samples, 2)
        y_target = yraw[net_id, rep_id, :n_eval_samples, tid, 0]  # squeeze last dim
        yhat_target = yhatdep[rep_id, :n_eval_samples, tid, net_id]  # shape: (n_samples,)

        # assertions
        assert_that(x_target).has_shape((n_eval_samples, 2))
        assert_that(y_target).has_shape((n_eval_samples,))
        assert_that(yhat_target).has_shape((n_eval_samples,))

        # create figure
        nax = 3 if show_difference else 2
        fig, axes = plt.subplots(1, nax, figsize=(nax * 5, 5), dpi=100)

        # ground truth
        sc1 = axes[0].scatter(
            x_target[:, 0], x_target[:, 1], c=y_target, cmap=DEFAULT_CMAP_NAME, s=5, alpha=0.7
        )
        axes[0].set_title("Target")
        axes[0].set_aspect("equal")
        plt.colorbar(sc1, ax=axes[0])

        # prediction
        sc2 = axes[1].scatter(
            x_target[:, 0], x_target[:, 1], c=yhat_target, cmap=DEFAULT_CMAP_NAME, s=5, alpha=0.7
        )
        axes[1].set_title(f"Prediction (loss={loss_val:.4f})")
        axes[1].set_aspect("equal")
        plt.colorbar(sc2, ax=axes[1])

        # difference
        if show_difference:
            diff = yhat_target - y_target
            assert_that(diff).has_shape((n_eval_samples,))
            vmax = jnp.abs(diff).max()
            sc3 = axes[2].scatter(
                x_target[:, 0],
                x_target[:, 1],
                c=diff,
                cmap="RdBu_r",
                s=5,
                alpha=0.7,
                vmin=-vmax,
                vmax=vmax,
            )
            axes[2].set_title(f"Difference (net: {dmanager.networks[net_id].name})")
            axes[2].set_aspect("equal")
            plt.colorbar(sc3, ax=axes[2])

        plt.suptitle(f"Target: {target.name} | Best: net {dmanager.networks[net_id].name})")
        plt.tight_layout()

        if save_dir:
            save_path = Path(save_dir) / f"design_result_{target.name}_rep{rep_id}_net{net_id}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved figure to {save_path}")

        plt.show()


##────────────────────────────────────────────────────────────────────────────}}}

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
    num_z = (dmanager.n_targets, int(num_z.ravel()[0].squeeze()))

    steps_per_epoch = max(1, dconf.n_batches_per_epoch // dconf.batches_per_step)
    total_steps = int(dconf.n_epochs * steps_per_epoch)

    print(
        f"Total steps: {total_steps}, Steps per epoch: {steps_per_epoch}, Epochs: {dconf.n_epochs}"
    )

    n_networks = stack.get_nb_networks()
    xbatches, ybatches = dmanager.get_samples(
        (n_networks, steps_per_epoch, dconf.n_replicates, dconf.batches_per_step, dconf.batch_size),
        bkey,
    )

    # glue all networks along the last dimension
    xbatches = jnp.concatenate(xbatches, axis=-1)
    ybatches = jnp.concatenate(ybatches, axis=-1)

    n_inputs = stack.get_nb_inputs()
    n_outputs = stack.get_nb_outputs()

    assert_that(xbatches).has_shape(
        (
            steps_per_epoch,
            dconf.n_replicates,
            dconf.batches_per_step,
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
        updates_need_vmap=True,
        static_tags=["non_grad", "shared"],
    )

    def step(params: ParameterTree, opt_state: optax.OptState, step_key, xs, ys):
        keys = jax.random.split(step_key, dconf.n_replicates)
        assert_that(xs).has_shape(
            (
                dconf.n_replicates,
                dconf.batches_per_step,
                dconf.batch_size,
                dmanager.n_targets,
                n_inputs,
            )
        )
        assert_that(ys).has_shape(
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
        verbose=True,
    )


##────────────────────────────────────────────────────────────────────────────}}}
