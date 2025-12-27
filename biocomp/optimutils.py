### {{{                          --     imports     --
from assertpy import assert_that
import optax
import random
import os
import time
from typing import List, Literal, Tuple, Callable, Optional, NamedTuple, Union

from biocomp.utils import (
    EncodedPartialFunction,
    PartialFunction,
    ArbitraryModel,
    PartialFunctionResult,
)
from .parameters import ParameterTree
from biocomp.logging_config import get_logger
from biocomp.compute import ComputeStack

import jax
from jax import vmap
from jax import numpy as jnp
from jax.typing import ArrayLike
from jax.experimental import checkify

logger = get_logger(__name__)

##────────────────────────────────────────────────────────────────────────────}}}

## {{{                --     config and optimizer stack     --
DEFAULT_OPTIMIZER = [
    PartialFunction(
        func="optax.clip_by_global_norm",
        kwargs={"max_norm": 1.0},
    ),
    PartialFunction(
        func="optax.adamw",
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
    """No-op gradient transformation that counts steps."""

    def init_fn(params):
        return CounterState(count=jnp.zeros([], jnp.int32))

    def update_fn(updates, state, params=None):
        return updates, CounterState(count=state.count + 1)

    return optax.GradientTransformation(init_fn, update_fn)


def build_optimizer_chain(optimizer_stack: list, with_lr_injection: bool = False):
    import importlib

    main_chain = []
    for comp in optimizer_stack:
        if not with_lr_injection or not (
            hasattr(comp, "kwargs") and "learning_rate" in comp.kwargs
        ):
            main_chain.append(comp())
            continue

        original_func = getattr(comp, "func", getattr(comp, "_func", None))
        if original_func is None:
            main_chain.append(comp())
            continue

        if isinstance(original_func, str):
            try:
                module_name, func_name = original_func.rsplit(".", 1)
                func = getattr(importlib.import_module(module_name), func_name)
            except (ValueError, ImportError, AttributeError):
                main_chain.append(comp())
                continue
        else:
            func = original_func

        try:
            lr_value = comp.kwargs["learning_rate"]
            lr_schedule = lr_value.get_impl()() if hasattr(lr_value, "get_impl") else lr_value
            wrapped_func = optax.inject_hyperparams(func)
            other_kwargs = {k: v for k, v in comp.kwargs.items() if k != "learning_rate"}
            main_chain.append(wrapped_func(learning_rate=lr_schedule, **other_kwargs))
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Failed to inject learning rate for {comp.func}: {e}. Falling back.")
            main_chain.append(comp())

    return optax.chain(create_counter(), *main_chain)


class OptimConfig(ArbitraryModel):
    optimizer_stack: list[EncodedPartialFunction] = DEFAULT_OPTIMIZER
    seed: Optional[int] = None
    batches_per_step: int = 4
    batch_size: int = 32
    n_epochs: float = 3
    n_replicates: int = 16
    keep_in_history: Union[List[str], Literal["all"]] = ["loss"]

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
        return build_optimizer_chain(self.optimizer_stack, with_lr_injection=False)

    def create_optimizer_with_lr_injection(self):
        return build_optimizer_chain(self.optimizer_stack, with_lr_injection=True)


# Design-specific config (adds loss_function, design-specific fields)
class DesignOptimConfig(OptimConfig):
    loss_function: EncodedPartialFunction
    n_batches_per_epoch: int = 128
    reshuffle_batches: bool = True


def get_checkify_enabled():
    return os.environ.get("BIOCOMP_CHECKIFY", "").lower() in ("true", "1", "yes", "on")


def compile_step(step_fn, sample_args, use_checkify=None):
    if use_checkify is None:
        use_checkify = get_checkify_enabled()

    if not use_checkify:
        lowered = jax.jit(step_fn).lower(*sample_args)
        return lowered.compile()

    ckf = jax.jit(checkify.checkify(step_fn, errors=checkify.all_checks))

    def checkified_step(*args):
        err, data = ckf(*args)
        err.throw()
        return data

    return checkified_step


def run_logger_callbacks(loggers, step, config, step_history, stack, period_filter):
    for period, callback in loggers:
        if period_filter(period, step):
            try:
                callback(step, config, step_history=step_history, stack=stack)
            except Exception as e:
                logger.error(f"Logger callback failed at step {step}: {e}")
                logger.exception(e)
                raise


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                  --     training step functions     --


def extract_learning_rate(opt_state):
    learning_rate = None
    try:
        if isinstance(opt_state, tuple):
            for state_component in opt_state:
                if (
                    hasattr(state_component, "hyperparams")
                    and "learning_rate" in state_component.hyperparams
                ):
                    learning_rate = state_component.hyperparams["learning_rate"]
                    break

        if (
            learning_rate is None
            and hasattr(opt_state, "hyperparams")
            and "learning_rate" in opt_state.hyperparams
        ):
            learning_rate = opt_state.hyperparams["learning_rate"]

        if learning_rate is None:
            try:
                learning_rate = optax.tree_utils.tree_get(
                    opt_state,
                    "learning_rate",
                    default=None,
                    filtering=lambda path, value: isinstance(value, (float, int))
                    or (hasattr(value, "shape") and hasattr(value, "dtype")),
                )
            except (KeyError, ValueError, TypeError):
                pass

    except (AttributeError, TypeError, KeyError):
        pass

    return learning_rate


def sanitize_gradients(grads):
    return jax.tree.map(lambda g: jnp.where(jnp.isfinite(g), g, 0.0) if g is not None else g, grads)


def make_training_step(
    loss_func,
    optimizer,
    fields_to_keep_in_history=("loss",),
    scannable=True,
    updates_need_vmap=False,
    static_tags=None,
    post_update_hook: Optional[Callable] = None,
    sanitize_grads: bool = False,
):
    from jax import value_and_grad

    if static_tags is None:
        static_tags = ["non_grad", "local"]

    opt_updt = optimizer.update
    if updates_need_vmap:
        opt_updt = vmap(optimizer.update)

    def base_training_step(params, opt_state, x, y, z, key):
        static, dynamic = params.filter_by_tag(static_tags)
        # extract scalar step (opt_state may be vmapped, giving count shape (n_targets,))
        step_count = opt_state[0].count
        step = step_count.ravel()[0] if jnp.ndim(step_count) > 0 else step_count

        (loss, aux), grads = value_and_grad(loss_func, has_aux=True)(
            dynamic, static, x, y, z, key, step
        )

        if sanitize_grads:
            grads = sanitize_gradients(grads)

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
        # Keys to exclude from history (handled separately or too large)
        exclude_from_all = {"opt"}

        def scannable_training_step(carry, i_x_y_z_k):
            params, opt_state = carry
            i, x, y, z, k = i_x_y_z_k
            updt = base_training_step(params, opt_state, x, y, z, k)
            params, opt_state = updt["params"], updt["opt"]
            if fields_to_keep_in_history == "all":
                history = {k: v for k, v in updt.items() if k not in exclude_from_all}
            else:
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
    assert xbatches.shape[:-1] == (
        training_config.batches_per_step,
        training_config.batch_size,
    )
    assert ybatches.shape[:-1] == (
        training_config.batches_per_step,
        training_config.batch_size,
    )
    num_z = (num_z,) if isinstance(num_z, int) else num_z
    zbatches = jax.random.uniform(
        key, (training_config.batches_per_step, training_config.batch_size, *num_z)
    )
    batch_keys = jax.random.split(key, training_config.batches_per_step)
    xs = (jnp.arange(training_config.batches_per_step), xbatches, ybatches, zbatches, batch_keys)
    history = {"loss": []}
    params, opt_state = start_params, start_opt_state
    for i, x, y, z, k in zip(*xs):
        updt = non_scannable_step(params, opt_state, x, y, z, k)
        params, opt_state = updt["params"], updt["opt"]
        history["loss"].append(updt["loss"])
    return params, opt_state, history


def reshuffle_batches_jax(xbatches, ybatches, key, axes=(0, 1, 2, 3)):
    assert xbatches.shape[:-1] == ybatches.shape[:-1], (
        "xbatches and ybatches must have the same shape"
    )

    axes = tuple(sorted(axes))
    batch_shape, feat_x, feat_y = xbatches.shape[:-1], xbatches.shape[-1], ybatches.shape[-1]
    axes_sizes = jnp.asarray([batch_shape[i] for i in axes], dtype=jnp.int32)
    flat_size = int(jnp.prod(axes_sizes))
    remaining = [i for i in range(len(batch_shape)) if i not in axes]
    remaining_sizes = [batch_shape[i] for i in remaining]

    perm = jnp.asarray(list(axes) + remaining + [len(batch_shape)], dtype=jnp.int32)
    xb, yb = xbatches.transpose(perm), ybatches.transpose(perm)
    xb = xb.reshape((flat_size, *remaining_sizes, feat_x))
    yb = yb.reshape((flat_size, *remaining_sizes, feat_y))

    idx = jax.random.permutation(key, jnp.arange(flat_size))
    xb, yb = xb[idx], yb[idx]

    xb = xb.reshape((*axes_sizes, *remaining_sizes, feat_x))
    yb = yb.reshape((*axes_sizes, *remaining_sizes, feat_y))
    inv = jnp.argsort(perm)
    return xb.transpose(inv), yb.transpose(inv)


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
    actual_batches_per_step, actual_batch_size = xbatches.shape[:2]
    assert_that(actual_batches_per_step).is_equal_to(training_config.batches_per_step)
    assert_that(xbatches.shape[0]).is_equal_to(ybatches.shape[0])
    assert_that(xbatches.shape[1]).is_equal_to(ybatches.shape[1])

    num_z = (num_z,) if isinstance(num_z, int) else num_z
    zbatches = jax.random.uniform(key, (actual_batches_per_step, actual_batch_size, *num_z))
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
    if callable(value_or_callable):
        return value_or_callable
    return lambda step: jnp.asarray(value_or_callable)


def three_phase_schedule(
    total_steps: int,
    phase1_frac: float,
    phase2_frac: float,
    phase1_value: float,
    phase2_end_value: float,
    phase3_end_value: float,
    phase2_power: float = 1.0,
    phase3_power: float = 2.0,
) -> optax.Schedule:
    """Create a three-phase schedule for TU masking temperature or L0 penalty.

    Phases:
        1. Constant at phase1_value (exploration)
        2. Polynomial decay from phase1_value to phase2_end_value (mask formation)
        3. Polynomial decay from phase2_end_value to phase3_end_value (commitment)

    The power parameter controls decay shape: 1=linear, 2=quadratic (slower start, faster end).
    """
    assert 0 < phase1_frac < phase2_frac < 1, (
        f"phase fractions must be 0 < phase1 < phase2 < 1, got {phase1_frac}, {phase2_frac}"
    )

    phase1_steps = int(phase1_frac * total_steps)
    phase2_steps = int(phase2_frac * total_steps)
    phase2_duration = phase2_steps - phase1_steps
    phase3_duration = total_steps - phase2_steps

    return optax.join_schedules(
        schedules=[
            optax.constant_schedule(phase1_value),
            optax.polynomial_schedule(
                init_value=phase1_value,
                end_value=phase2_end_value,
                power=phase2_power,
                transition_steps=phase2_duration,
            ),
            optax.polynomial_schedule(
                init_value=phase2_end_value,
                end_value=phase3_end_value,
                power=phase3_power,
                transition_steps=phase3_duration,
            ),
        ],
        boundaries=[phase1_steps, phase2_steps],
    )


def five_phase_schedule(
    total_steps: int,
    phase1_frac: float,
    phase2_frac: float,
    phase3_frac: float,
    phase4_frac: float,
    phase1_value: float,
    phase2_end_value: float,
    phase3_end_value: float,
    phase4_start_value: float,
    phase5_end_value: float,
    phase2_power: float = 1.0,
    phase3_power: float = 2.0,
    phase5_power: float = 2.0,
) -> optax.Schedule:
    """Create a five-phase schedule with wake-up (re-softening) phase.

    This schedule extends three_phase_schedule with a "wake-up" phase that
    JUMPS back to a softer value, allowing recovery from over-pruning.
    Inspired by simulated annealing with reheating.

    Phases:
        1. Constant at phase1_value (warm exploration)
        2. Polynomial decay to phase2_end_value (initial pruning)
        3. Polynomial decay to phase3_end_value (initial commitment)
        4. JUMP to phase4_start_value (wake-up - allows TU recovery)
        5. Polynomial decay to phase5_end_value (final commitment)

    The key innovation is Phase 4: a discontinuous jump to a softer value,
    allowing pruned TUs to potentially recover if gradients favor them.

    Args:
        total_steps: Total number of optimization steps
        phase1_frac: Fraction of steps for phase 1 (e.g., 0.25)
        phase2_frac: Fraction of steps at end of phase 2 (e.g., 0.50)
        phase3_frac: Fraction of steps at end of phase 3 (e.g., 0.65)
        phase4_frac: Fraction of steps at end of phase 4 (e.g., 0.85)
        phase1_value: Constant value during exploration
        phase2_end_value: Value at end of initial pruning
        phase3_end_value: Value at end of initial commitment
        phase4_start_value: JUMP to this value at wake-up start
        phase5_end_value: Final value at end of training
        phase2_power: Polynomial power for phase 2 decay
        phase3_power: Polynomial power for phase 3 decay
        phase5_power: Polynomial power for phase 5 decay

    Returns:
        optax.Schedule that can be called with step number
    """
    assert 0 < phase1_frac < phase2_frac < phase3_frac < phase4_frac < 1, (
        f"phase fractions must be 0 < p1 < p2 < p3 < p4 < 1, got "
        f"{phase1_frac}, {phase2_frac}, {phase3_frac}, {phase4_frac}"
    )

    phase1_steps = int(phase1_frac * total_steps)
    phase2_steps = int(phase2_frac * total_steps)
    phase3_steps = int(phase3_frac * total_steps)
    phase4_steps = int(phase4_frac * total_steps)

    phase2_duration = phase2_steps - phase1_steps
    phase3_duration = phase3_steps - phase2_steps
    phase5_duration = total_steps - phase4_steps

    return optax.join_schedules(
        schedules=[
            # Phase 1: constant warm exploration
            optax.constant_schedule(phase1_value),
            # Phase 2: polynomial decay (initial pruning)
            optax.polynomial_schedule(
                init_value=phase1_value,
                end_value=phase2_end_value,
                power=phase2_power,
                transition_steps=phase2_duration,
            ),
            # Phase 3: polynomial decay (initial commitment)
            optax.polynomial_schedule(
                init_value=phase2_end_value,
                end_value=phase3_end_value,
                power=phase3_power,
                transition_steps=phase3_duration,
            ),
            # Phase 4: JUMP to wake-up value (discontinuous!)
            optax.constant_schedule(phase4_start_value),
            # Phase 5: polynomial decay (final commitment)
            optax.polynomial_schedule(
                init_value=phase4_start_value,
                end_value=phase5_end_value,
                power=phase5_power,
                transition_steps=phase5_duration,
            ),
        ],
        boundaries=[phase1_steps, phase2_steps, phase3_steps, phase4_steps],
    )


def optimize(
    step: Callable,
    params,
    opt_state,
    xbatches: jax.Array,
    ybatches: jax.Array,
    config: DesignOptimConfig,
    n_total_steps: int,
    steps_per_epoch: int,
    key: ArrayLike,
    stack: ComputeStack,
    loggers: Optional[List[Tuple[int, Callable]]] = None,
    async_handler=None,
    verbose=False,
    defer_sync: bool = True,
    sync_every: int = 0,
):
    loggers = loggers or []

    assert_that(xbatches.shape[:3]).is_equal_to(
        (steps_per_epoch, config.n_replicates, config.batches_per_step)
    )
    assert_that(ybatches.shape[:3]).is_equal_to(
        (steps_per_epoch, config.n_replicates, config.batches_per_step)
    )

    xb, yb = xbatches[0], ybatches[0]
    logger.info("[COMPILE] Compiling training step (AOT)...")
    t_compile = time.perf_counter()
    compiled_step = compile_step(step, (params, opt_state, key, xb, yb))
    compile_time = time.perf_counter() - t_compile
    if not get_checkify_enabled():
        logger.info(f"[COMPILE] Step compiled in {compile_time:.2f}s")

    if async_handler:
        async_handler.process_start_loggers(config, stack)
    else:
        run_logger_callbacks(loggers, 0, config, {}, stack, lambda p, s: p == 0)

    step_history, loss_history = {}, []
    epoch = -1
    pending_losses = []  # collect losses without forcing sync

    # sync_every=0 means sync at epoch boundaries only
    effective_sync_every = sync_every if sync_every > 0 else steps_per_epoch

    # Progress reporting frequency
    progress_every = max(1, n_total_steps // 20)  # ~5% progress updates

    logger.info(f"[OPTIMIZE] Starting {config.n_epochs} epochs, {n_total_steps} total steps")
    logger.info(f"[OPTIMIZE] Config: {steps_per_epoch} steps/epoch, defer_sync={defer_sync}")

    t_loop_start = time.perf_counter()
    epoch_start_time = t_loop_start
    epoch_step_count = 0

    for i, step_key in enumerate(jax.random.split(key, n_total_steps), 1):
        is_epoch_boundary = i % steps_per_epoch == 0
        should_sync = not defer_sync or (i % effective_sync_every == 0) or is_epoch_boundary

        if is_epoch_boundary:
            # End of epoch timing
            epoch_time = time.perf_counter() - epoch_start_time
            epoch += 1
            b_key = jax.random.fold_in(step_key, epoch)
            if config.reshuffle_batches and i > 0:
                xbatches, ybatches = reshuffle_batches_jax(xbatches, ybatches, b_key)

            # sync and report loss at epoch boundaries
            if pending_losses:
                jax.block_until_ready(params)
                loss_history.extend([float(jnp.mean(l)) for l in pending_losses])
                pending_losses = []

            current_loss = loss_history[-1] if loss_history else float("nan")
            if hasattr(current_loss, "mean"):
                current_loss = float(current_loss.mean())
            elif hasattr(current_loss, "__float__"):
                current_loss = float(current_loss)

            steps_per_sec = steps_per_epoch / epoch_time if epoch_time > 0 else 0
            logger.info(
                f"[EPOCH {epoch + 1}/{int(config.n_epochs)}] "
                f"Step {i}/{n_total_steps} | Loss: {current_loss:.4f} | "
                f"{epoch_time:.1f}s ({steps_per_sec:.1f} steps/s)"
            )

            # Reset for next epoch
            epoch_start_time = time.perf_counter()
            epoch_step_count = 0

        # Progress update mid-epoch
        elif i % progress_every == 0:
            elapsed = time.perf_counter() - t_loop_start
            pct = i / n_total_steps * 100
            eta = (elapsed / i) * (n_total_steps - i) if i > 0 else 0
            logger.info(
                f"[PROGRESS] Step {i}/{n_total_steps} ({pct:.0f}%) | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s"
            )

        xb, yb = xbatches[i % steps_per_epoch], ybatches[i % steps_per_epoch]

        if defer_sync:
            params, opt_state, step_history = compiled_step(params, opt_state, step_key, xb, yb)
            if "loss" in step_history:
                pending_losses.append(step_history["loss"])
            # only populate full step_history at sync points
            if should_sync:
                jax.block_until_ready(params)
                step_history["latest_params"] = params
                step_history["opt_state"] = opt_state
        else:
            t0 = time.perf_counter()
            params, opt_state, step_history = compiled_step(params, opt_state, step_key, xb, yb)
            step_history["step_time"] = time.perf_counter() - t0
            step_history["latest_params"] = params
            step_history["opt_state"] = opt_state
            if "loss" in step_history:
                loss_history.append(step_history["loss"])

        epoch_step_count += 1

        run_logger_callbacks(
            loggers, i, config, step_history, stack, lambda p, s: p > 0 and s % p == 0
        )

    t_sync = time.perf_counter()
    jax.block_until_ready(params)
    sync_time = time.perf_counter() - t_sync

    # flush any remaining pending losses
    if pending_losses:
        loss_history.extend([float(jnp.mean(l)) for l in pending_losses])
        pending_losses = []

    total_loop_time = time.perf_counter() - t_loop_start

    if not async_handler:
        run_logger_callbacks(
            loggers, n_total_steps, config, step_history, stack, lambda p, s: p is None or p == -1
        )

    # Final summary
    logger.info("=" * 60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info(f"  Compilation:    {compile_time:.2f}s")
    logger.info(f"  Loop time:      {total_loop_time:.2f}s ({n_total_steps} steps)")
    logger.info(f"  Final sync:     {sync_time:.2f}s")
    logger.info(f"  Avg step time:  {total_loop_time / n_total_steps * 1000:.2f}ms")
    if loss_history:
        logger.info(f"  Final loss:     {loss_history[-1]:.4f}")
    logger.info("=" * 60)

    return params, loss_history, step_history
