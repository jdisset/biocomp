### {{{                          --     imports     --
from assertpy import assert_that
from . import datautils as du
import optax
import random

from biocomp.utils import (
    EncodedPartialFunction,
    PartialFunction,
    ArbitraryModel,
    PartialFunctionResult,
)
import os
from . import nodes as nodes
from .parameters import ParameterTree, ParamPath
from . import utils as ut
import time
from typing import List, Tuple, Callable, Optional, NamedTuple
from pydantic import Field
from biocomp.logging_config import get_logger
from biocomp.compute import ComputeStack


import jax
from jax import jit, vmap, lax
from jax import numpy as jnp
from jax.typing import ArrayLike
from jax.experimental import checkify

logger = get_logger(__name__)

##────────────────────────────────────────────────────────────────────────────}}}

## {{{                --     config and optimizer stack     --
DEFAULT_OPTIMIZER = [
    PartialFunction(
        func="optax.clip_by_global_norm",
        # func="optax.transforms._clipping.clip_by_global_norm",
        kwargs={"max_norm": 1.0},
    ),
    PartialFunction(
        func="optax.adamw",
        # func="optax._src.alias.adamw",
        kwargs={
            "learning_rate": PartialFunctionResult(
                func="optax.warmup_cosine_decay_schedule",
                kwargs={
                    "init_value": 1e-7,
                    "peak_value": 1e-3,
                    "warmup_steps": 15,
                    "decay_steps": 130,
                    "end_value": 1e-5,
                },
            )
        },
    ),
]


class CounterState(NamedTuple):
    count: jnp.ndarray  # type: ignore


def create_counter():
    """Creates a no-op gradient transformation that just counts steps."""
    import jax.numpy as jnp
    import optax

    def init_fn(params):
        return CounterState(count=jnp.zeros([], jnp.int32))

    def update_fn(updates, state, params=None):
        return updates, CounterState(count=state.count + 1)

    return optax.GradientTransformation(init_fn, update_fn)


class OptimConfig(ArbitraryModel):
    loss_function: EncodedPartialFunction
    optimizer_stack: list[EncodedPartialFunction] = DEFAULT_OPTIMIZER

    seed: Optional[int] = None
    batches_per_step: int = 4  # how many batches to process in one scan step
    batch_size: int = 32
    n_epochs: float = 3
    n_batches_per_epoch: int = 128  # number of batches per epoch to draw from the data source
    n_replicates: int = 16  # aka population size
    keep_in_history: List[str] = ["loss"]

    reshuffle_batches: bool = True  # whether to reshuffle batches at the end of each epoch

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        if self.seed is None:
            self.seed = random.randint(0, 2**32 - 1)

    @property
    def seed_key(self):
        if self.seed is None:
            self.seed = random.randint(0, 2**32 - 1)
        return jax.random.PRNGKey(self.seed)

    @property
    def optimizer(self):
        main_chain = [comp() for comp in self.optimizer_stack]
        return optax.chain(create_counter(), *main_chain)

    def create_optimizer_with_lr_injection(self):
        """Create optimizer with learning rate injection for debugging purposes."""
        import optax

        main_chain = []

        for comp in self.optimizer_stack:
            # check if this component has a learning_rate parameter
            if hasattr(comp, "kwargs") and "learning_rate" in comp.kwargs:
                if hasattr(comp, "func"):
                    original_func = comp.func
                elif hasattr(comp, "_func"):
                    original_func = comp._func
                else:
                    # Fallback to regular instantiation
                    main_chain.append(comp())
                    continue

                # handle string function references
                if isinstance(original_func, str):
                    import importlib

                    try:
                        module_name, func_name = original_func.rsplit(".", 1)
                        module = importlib.import_module(module_name)
                        func = getattr(module, func_name)
                    except (ValueError, ImportError, AttributeError):
                        # Fallback if we can't resolve the string
                        main_chain.append(comp())
                        continue
                else:
                    func = original_func

                try:
                    # for learning rate injection to work, we need to resolve PartialFunctionResult first
                    lr_value = comp.kwargs["learning_rate"]
                    if hasattr(lr_value, "get_impl"):
                        # it's a PartialFunctionResult, resolve it to get the actual schedule
                        lr_schedule = lr_value.get_impl()()
                    else:
                        lr_schedule = lr_value

                    wrapped_func = optax.inject_hyperparams(func)

                    other_kwargs = {k: v for k, v in comp.kwargs.items() if k != "learning_rate"}

                    optimizer_instance = wrapped_func(learning_rate=lr_schedule, **other_kwargs)
                    main_chain.append(optimizer_instance)
                except Exception:
                    logger.warning(
                        f"Failed to inject learning rate for {comp.func}. "
                        "Falling back to original component."
                    )
                    main_chain.append(comp())
            else:
                main_chain.append(comp())

        return optax.chain(create_counter(), *main_chain)


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                  --     training step functions     --


def extract_learning_rate(opt_state):
    """Extract learning rate from the optimizer state."""
    learning_rate = None
    try:
        # check for hyperparams in individual state components
        if isinstance(opt_state, tuple):
            for state_component in opt_state:
                if (
                    hasattr(state_component, "hyperparams")
                    and "learning_rate" in state_component.hyperparams
                ):
                    learning_rate = state_component.hyperparams["learning_rate"]
                    break

        # check direct hyperparams access
        if (
            learning_rate is None
            and hasattr(opt_state, "hyperparams")
            and "learning_rate" in opt_state.hyperparams
        ):
            learning_rate = opt_state.hyperparams["learning_rate"]

        # try tree_get
        if learning_rate is None:
            try:
                learning_rate = optax.tree_utils.tree_get(
                    opt_state,
                    "learning_rate",
                    default=None,
                    filtering=lambda path, value: isinstance(value, (float, int))
                    or (hasattr(value, "shape") and hasattr(value, "dtype")),
                )
            except (KeyError, ValueError):
                pass

    except Exception:
        pass

    return learning_rate


def make_training_step(
    loss_func,
    optimizer,
    fields_to_keep_in_history=("loss",),
    scannable=True,
    updates_need_vmap=False,
    static_tags=None,
    post_update_hook: Optional[Callable] = None,
):
    from jax import value_and_grad
    import optax
    import jax.numpy as jnp

    if static_tags is None:
        static_tags = ["non_grad", "local"]

    opt_updt = optimizer.update
    if updates_need_vmap:
        opt_updt = vmap(optimizer.update)

    def base_training_step(params, opt_state, x, y, z, key):
        static, dynamic = params.filter_by_tag(static_tags)

        (loss, aux), grads = value_and_grad(loss_func, has_aux=True)(
            dynamic, static, x, y, z, key, opt_state[0].count
        )

        updates, opt_state = opt_updt(grads, opt_state, dynamic)
        dynamic = optax.apply_updates(dynamic, updates)

        if static.data:
            params = ParameterTree.merge(static, dynamic)
        else:
            params = dynamic

        if post_update_hook:
            params = post_update_hook(params)

        res = {
            "params": params,
            "loss": loss,
            "grad": grads,
            "opt": opt_state,
            "x": x,
            "y": y,
            "z": z,
            "key": key,
            "learning_rate": extract_learning_rate(opt_state),
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


def per_replicate_step_nonscan(
    start_params,
    start_opt_state,
    key,
    xbatches,
    ybatches,
    num_z,
    training_config,
    non_scannable_step,
):
    """Non-scannable version of per-replicate training step.

    Args:
        start_params: Initial parameters
        start_opt_state: Initial optimizer state
        key: JAX random key
        xbatches: Input batches of shape (batches_per_step, batch_size, features)
        ybatches: Target batches of shape (batches_per_step, batch_size, outputs)
        num_z: Number of quantile variables
        training_config: Training configuration containing batches_per_step and batch_size
        non_scannable_step: Non-scannable training step function
    """
    import jax
    import jax.numpy as jnp

    assert xbatches.shape[:-1] == (
        training_config.batches_per_step,
        training_config.batch_size,
    )
    assert ybatches.shape[:-1] == (
        training_config.batches_per_step,
        training_config.batch_size,
    )
    if isinstance(num_z, int):
        num_z = (num_z,)
    zbatches = jax.random.uniform(
        key, (training_config.batches_per_step, training_config.batch_size, *num_z)
    )
    batch_keys = jax.random.split(key, training_config.batches_per_step)
    xs = (
        jnp.arange(training_config.batches_per_step),
        xbatches,
        ybatches,
        zbatches,
        batch_keys,
    )
    history = {"loss": []}
    params, opt_state = (start_params, start_opt_state)
    for i, x, y, z, k in zip(*xs):
        updt = non_scannable_step(params, opt_state, x, y, z, k)
        params, opt_state = updt["params"], updt["opt"]
        history["loss"].append(updt["loss"])
    return params, opt_state, history


def reshuffle_batches_jax(xbatches, ybatches, key, axes=(0, 1, 2, 3)):
    """Full permutation of the given axes of xbatches and ybatches, after flattening them."""
    assert xbatches.shape[:-1] == ybatches.shape[:-1], (
        "xbatches and ybatches must have the same shape"
    )

    axes = tuple(sorted(axes))
    batch_shape = xbatches.shape[:-1]
    feature_dim_x = xbatches.shape[-1]
    feature_dim_y = ybatches.shape[-1]

    # dimensions of the axes to flatten
    axes_sizes = jnp.asarray([batch_shape[i] for i in axes], dtype=jnp.int32)
    flattened_size = int(jnp.prod(axes_sizes))

    remaining_axes = [i for i in range(len(batch_shape)) if i not in axes]
    remaining_sizes = [batch_shape[i] for i in remaining_axes]

    # permute so that flatten-axes come first, then remaining, then features
    permute_order = jnp.asarray(list(axes) + remaining_axes + [len(batch_shape)], dtype=jnp.int32)
    xb = xbatches.transpose(permute_order)
    yb = ybatches.transpose(permute_order)

    xb = xb.reshape((flattened_size, *remaining_sizes, feature_dim_x))
    yb = yb.reshape((flattened_size, *remaining_sizes, feature_dim_y))

    # shuffle along the flattened axis
    idx = jax.random.permutation(key, jnp.arange(flattened_size))
    xb = xb[idx]
    yb = yb[idx]

    # restore original grouped axes shape
    xb = xb.reshape((*axes_sizes, *remaining_sizes, feature_dim_x))
    yb = yb.reshape((*axes_sizes, *remaining_sizes, feature_dim_y))

    # invert the initial permutation to return to original order
    inv_permute = jnp.argsort(permute_order)
    xb = xb.transpose(inv_permute)
    yb = yb.transpose(inv_permute)

    return xb, yb


def per_replicate_step(
    start_params,
    start_opt_state,
    key,
    xbatches,
    ybatches,
    num_z,
    training_config,
    scannable_step,
    enable_jax_tqdm=False,
):
    """Scannable version of per-replicate training step.

    Args:
        start_params: Initial parameters
        start_opt_state: Initial optimizer state
        key: JAX random key
        xbatches: Input batches of shape (batches_per_step, batch_size, features)
        ybatches: Target batches of shape (batches_per_step, batch_size, outputs)
        num_z: Number of quantile variables
        training_config: Training configuration containing batches_per_step and batch_size
        scannable_step: Scannable training step function
        enable_jax_tqdm: Whether to enable jax_tqdm progress bar
    """
    import jax
    import jax.numpy as jnp

    actual_batches_per_step, actual_batch_size = xbatches.shape[:2]
    assert_that(actual_batches_per_step).is_equal_to(training_config.batches_per_step)
    assert_that(xbatches.shape[0]).is_equal_to(ybatches.shape[0])
    assert_that(xbatches.shape[1]).is_equal_to(ybatches.shape[1])

    if isinstance(num_z, int):
        num_z = (num_z,)
    zbatches = jax.random.uniform(
        key, (actual_batches_per_step, actual_batch_size, *num_z)
    )

    batch_keys = jax.random.split(key, training_config.batches_per_step)
    if enable_jax_tqdm:
        from jax_tqdm import scan_tqdm

        sstep = scan_tqdm(training_config.batches_per_step)(scannable_step)  # type: ignore
    else:
        sstep = scannable_step
    carry = (start_params, start_opt_state)
    xs = (
        jnp.arange(training_config.batches_per_step),
        xbatches,
        ybatches,
        zbatches,
        batch_keys,
    )
    (final_params, final_opt_state), step_history = jax.lax.scan(
        sstep,
        carry,
        xs,
    )
    return final_params, final_opt_state, step_history


##────────────────────────────────────────────────────────────────────────────}}}


def as_schedule(value_or_callable):
    """Convert a value or callable to an optax-like schedule function."""

    if callable(value_or_callable):
        return value_or_callable

    def f(step):
        return jnp.asarray(value_or_callable)

    return f


def optimize(
    step: Callable,
    params,
    opt_state,
    xbatches: jax.Array,
    ybatches: jax.Array,
    config: OptimConfig,
    n_total_steps: int,
    steps_per_epoch: int,
    key: ArrayLike,
    stack: ComputeStack,
    loggers: Optional[List[Tuple[int, Callable]]] = None,
    async_handler=None,
    verbose=False,
):
    loggers = loggers or []

    assert_that(xbatches.shape[:3]).is_equal_to(
        (steps_per_epoch, config.n_replicates, config.batches_per_step)
    )
    assert_that(ybatches.shape[:3]).is_equal_to(
        (steps_per_epoch, config.n_replicates, config.batches_per_step)
    )

    def prnt(msg):
        logger.info(msg)

    BIOCOMP_CHECKIFY = os.environ.get("BIOCOMP_CHECKIFY", "").lower() in ("true", "1", "yes", "on")

    xb, yb = xbatches[0], ybatches[0]
    logger.info("Compiling training step...")
    if not BIOCOMP_CHECKIFY:
        t0 = time.time()
        prnt("Compiling training step...")
        lowered = jax.jit(step).lower(params, opt_state, key, xb, yb)
        compiled_step = lowered.compile()
        logger.info(f"Compiled training step in {time.time() - t0:.2f}s")
    else:
        ckf = jax.jit(checkify.checkify(step, errors=checkify.all_checks))

        def checkified_step(params, opt_state, step_key, xs, ys):
            err, data = ckf(params, opt_state, step_key, xs, ys)
            err.throw()
            return data

        compiled_step = checkified_step

    # call start-of-training loggers (period=0)
    if async_handler:
        async_handler.process_start_loggers(config, stack)
    else:
        for period, callback in loggers:
            if period == 0:
                try:
                    callback(0, config, step_history={}, stack=stack)
                except Exception as e:
                    logger.error(f"Start logger callback failed: {e}")
                    logger.exception(e)

    step_history, loss_history = {}, []
    epoch = -1
    assert xbatches.shape[0] == steps_per_epoch
    assert ybatches.shape[0] == steps_per_epoch

    prnt(f"Starting training for {config.n_epochs} epochs with {n_total_steps} total steps.")

    last_log_time = time.time()
    for i, step_key in enumerate(jax.random.split(key, n_total_steps), 1):
        if i % (steps_per_epoch) == 0:
            epoch += 1
            b_key = jax.random.fold_in(step_key, epoch)
            if config.reshuffle_batches and i > 0:
                logger.debug(f"Reshuffling batches at epoch {epoch + 1}")
                xbatches, ybatches = reshuffle_batches_jax(xbatches, ybatches, b_key)
            current_loss = loss_history[-1] if loss_history else float('nan')
            if hasattr(current_loss, 'mean'):
                current_loss = float(current_loss.mean())
            logger.info(f"Epoch {epoch + 1}/{config.n_epochs} | Step {i}/{n_total_steps} | Loss: {current_loss:.4f}")

        t0 = time.time()

        xb, yb = xbatches[i % steps_per_epoch], ybatches[i % steps_per_epoch]
        params, opt_state, step_history = compiled_step(params, opt_state, step_key, xb, yb)

        step_history["step_time"] = time.time() - t0
        step_history["latest_params"] = params
        step_history["opt_state"] = opt_state

        # prnt(f"Step {i} completed in {step_history['step_time']:.2f} seconds")
        # prnt(
        #     f"Loss: {step_history.get('loss', 'N/A')} \n"
        #     f"Learning Rate: {step_history.get('learning_rate', 'N/A')}"
        # )

        if "loss" in step_history:
            loss_history.append(step_history["loss"])

        # call logger callbacks at their specified periods
        for period, callback in loggers:
            if period > 0 and i % period == 0:
                try:
                    callback(i, config, step_history=step_history, stack=stack)
                except Exception as e:
                    logger.error(f"Logger callback failed at step {i}: {e}")
                    logger.exception(e)

    # call end-of-training loggers (period=None or -1)
    if not async_handler:  # end loggers handled separately for async mode
        for period, callback in loggers:
            if period is None or period == -1:
                try:
                    callback(n_total_steps, config, step_history=step_history, stack=stack)
                except Exception as e:
                    logger.error(f"End logger callback failed: {e}")
                    logger.exception(e)

    logger.info(f"End of training for {config.n_epochs} epochs")
    prnt("Training completed.")

    return params, loss_history, step_history
