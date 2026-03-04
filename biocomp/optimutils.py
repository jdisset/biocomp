### {{{                          --     imports     --
from assertpy import assert_that
import optax
import random
import os
import time
from typing import List, Literal, Callable, Optional, NamedTuple, Union

from biocomp.utils import (
    EncodedPartialFunction,
    PartialFunction,
    ArbitraryModel,
    PartialFunctionResult,
)
from .parameters import ParameterTree
from biocomp.logging_config import get_logger
from biocomp.compute import ComputeStack
from biocomp.logger_dispatch import LoggerDispatch, NullDispatch
from biocomp.step_history import StepHistorySnapshot

import jax
from jax import vmap
from jax import numpy as jnp
from jax.typing import ArrayLike
from jax.experimental import checkify

logger = get_logger(__name__)

##────────────────────────────────────────────────────────────────────────────}}}

## {{{                --     config and optimizer stack     --

DEFAULT_OPTIMIZER_SIMPLE = [
    PartialFunction(func="optax.clip_by_global_norm", kwargs={"max_norm": 1.0}),
    PartialFunction(func="optax.adam", kwargs={"learning_rate": 0.02}),
]

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


def set_optimizer_state_step(opt_state, step: int):
    """Set all optimizer `count` fields to a specific step value."""
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    if step == 0:
        return opt_state

    step_value = int(step)

    def _map(path, leaf):
        if not path:
            return leaf
        key = path[-1]
        name = getattr(key, "name", None)
        if name != "count":
            return leaf
        if not hasattr(leaf, "dtype"):
            return leaf
        try:
            if not jnp.issubdtype(leaf.dtype, jnp.integer):
                return leaf
        except TypeError:
            return leaf
        return jnp.full_like(leaf, step_value)

    return jax.tree_util.tree_map_with_path(_map, opt_state)


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


def build_two_timescale_optimizer(
    optimizer_stack: list,
    tu_mask_lr_scale: float = 0.1,
) -> optax.GradientTransformation:
    """Two-timescale optimizer: slower learning rate for TU mask parameters.

    TU mask params (log_alpha) update slower to prevent them from racing ahead
    of continuous params and making premature enable/disable decisions.
    """
    base_opt = build_optimizer_chain(optimizer_stack, with_lr_injection=False)
    scaled_opt = optax.chain(optax.scale(tu_mask_lr_scale), base_opt)

    def label_fn(path, _):
        path_str = "/".join(str(p) for p in path) if isinstance(path, tuple) else str(path)
        return "tu_mask" if "tu_log_alpha" in path_str else "default"

    return optax.multi_transform({"tu_mask": scaled_opt, "default": base_opt}, label_fn)


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


def create_gd_step_fn(optimizer: optax.GradientTransformation, sanitize_grads: bool = True):
    """Create a reusable GD step function for inner optimization loops."""
    def gd_step(params, opt_state, loss_fn):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        if sanitize_grads:
            grads = sanitize_gradients(grads)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, loss
    return gd_step


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
    for _i, x, y, z, k in zip(*xs, strict=False):
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


def linear_schedule(
    total_steps: int,
    start_value: float,
    end_value: float,
) -> optax.Schedule:
    """Simple linear interpolation schedule from start to end over total_steps."""
    return optax.polynomial_schedule(
        init_value=start_value,
        end_value=end_value,
        power=1.0,
        transition_steps=total_steps,
    )


def constant_or_linear_schedule(
    total_steps: int,
    start_value: float,
    end_value: float | None = None,
) -> optax.Schedule:
    """If end_value is None or equals start_value, return constant; else linear."""
    if end_value is None or start_value == end_value:
        return optax.constant_schedule(start_value)
    return linear_schedule(total_steps, start_value, end_value)


def jax_three_phase_schedule(
    step: ArrayLike,
    total_steps: int,
    phase1_frac: ArrayLike,
    phase2_frac: ArrayLike,
    phase1_value: ArrayLike,
    phase2_end_value: ArrayLike,
    phase3_end_value: ArrayLike,
    phase2_power: float = 1.0,
    phase3_power: float = 2.0,
) -> jnp.ndarray:
    """Pure JAX three-phase schedule evaluation for dynamic hyperparameter injection.

    Unlike optax schedules, this function takes schedule parameters as JAX arrays,
    allowing them to be changed without recompiling the JIT-traced function.

    Uses the same polynomial interpolation formula as optax.polynomial_schedule:
        decay = (1 - progress)^power
        value = end_value + (init_value - end_value) * decay

    Args:
        step: Current step (JAX traced value)
        total_steps: Total optimization steps (static, known at trace time)
        phase1_frac: Fraction of steps for phase 1 (exploration)
        phase2_frac: Fraction of steps at end of phase 2 (pruning)
        phase1_value: Constant value during phase 1
        phase2_end_value: Value at end of phase 2
        phase3_end_value: Value at end of phase 3
        phase2_power: Polynomial power for phase 2 decay (default 1.0 = linear)
        phase3_power: Polynomial power for phase 3 decay (default 2.0 = quadratic)

    Returns:
        Interpolated value at the given step
    """
    step = jnp.asarray(step, dtype=jnp.float32)
    phase1_steps = phase1_frac * total_steps
    phase2_steps = phase2_frac * total_steps

    # Phase 1: constant at phase1_value
    in_phase1 = step < phase1_steps

    # Phase 2: polynomial decay from phase1_value to phase2_end_value
    # Using optax formula: decay = (1 - progress)^power, value = end + (init - end) * decay
    phase2_duration = jnp.maximum(phase2_steps - phase1_steps, 1e-8)
    phase2_progress = jnp.clip((step - phase1_steps) / phase2_duration, 0.0, 1.0)
    phase2_decay = (1.0 - phase2_progress) ** phase2_power
    phase2_value = phase2_end_value + (phase1_value - phase2_end_value) * phase2_decay
    in_phase2 = (step >= phase1_steps) & (step < phase2_steps)

    # Phase 3: polynomial decay from phase2_end_value to phase3_end_value
    phase3_duration = jnp.maximum(total_steps - phase2_steps, 1e-8)
    phase3_progress = jnp.clip((step - phase2_steps) / phase3_duration, 0.0, 1.0)
    phase3_decay = (1.0 - phase3_progress) ** phase3_power
    phase3_value = phase3_end_value + (phase2_end_value - phase3_end_value) * phase3_decay

    return jnp.where(in_phase1, phase1_value, jnp.where(in_phase2, phase2_value, phase3_value))


def jax_linear_schedule(
    step: ArrayLike,
    total_steps: int,
    start_value: ArrayLike,
    end_value: ArrayLike,
) -> jnp.ndarray:
    """Pure JAX linear schedule evaluation for dynamic hyperparameter injection.

    Args:
        step: Current step (JAX traced value)
        total_steps: Total optimization steps (static)
        start_value: Value at step 0
        end_value: Value at final step

    Returns:
        Linearly interpolated value at the given step
    """
    step = jnp.asarray(step, dtype=jnp.float32)
    progress = jnp.clip(step / (total_steps + 1e-8), 0.0, 1.0)
    return start_value + (end_value - start_value) * progress


def jax_smooth_three_phase_schedule(
    step: ArrayLike,
    total_steps: int,
    phase1_frac: ArrayLike,
    phase2_frac: ArrayLike,
    phase1_value: ArrayLike,
    phase2_end_value: ArrayLike,
    phase3_end_value: ArrayLike,
    transition_sharpness: float = 10.0,
) -> jnp.ndarray:
    """Three-phase schedule with sigmoid-blended transitions.

    Unlike jax_three_phase_schedule which has hard phase boundaries,
    this uses sigmoid blending for smoother transitions that don't
    abruptly kill promising TUs at phase boundaries.

    Args:
        transition_sharpness: Higher = sharper transitions (10.0 is reasonably smooth)
    """
    step = jnp.asarray(step, dtype=jnp.float32)
    phase1_steps = phase1_frac * total_steps
    phase2_steps = phase2_frac * total_steps

    transition_width = 0.05 * total_steps

    blend_1_to_2 = jax.nn.sigmoid(transition_sharpness * (step - phase1_steps) / transition_width)
    blend_2_to_3 = jax.nn.sigmoid(transition_sharpness * (step - phase2_steps) / transition_width)

    phase2_val = phase1_value + (phase2_end_value - phase1_value) * blend_1_to_2
    return phase2_val + (phase3_end_value - phase2_end_value) * blend_2_to_3


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
    dispatch: LoggerDispatch | None = None,
    verbose=False,
    defer_sync: bool = True,
    sync_every: int = 0,
    precompiled: bool = False,
    skip_lifecycle: bool = False,
    select_best_synced_params: bool = False,
    best_synced_score_fn: Callable[[ParameterTree, dict, int], float | None] | None = None,
    best_synced_initial_score: float | None = None,
    step_offset: int = 0,
):
    dispatch = dispatch or NullDispatch()

    assert_that(xbatches.shape[:3]).is_equal_to(
        (steps_per_epoch, config.n_replicates, config.batches_per_step)
    )
    assert_that(ybatches.shape[:3]).is_equal_to(
        (steps_per_epoch, config.n_replicates, config.batches_per_step)
    )

    xb, yb = xbatches[0], ybatches[0]
    if precompiled:
        compiled_step = step
        compile_time = 0.0
        logger.info("[COMPILE] Using pre-compiled step")
    else:
        logger.info("[COMPILE] Compiling training step (AOT)...")
        t_compile = time.perf_counter()
        compiled_step = compile_step(step, (params, opt_state, key, xb, yb))
        compile_time = time.perf_counter() - t_compile
        if not get_checkify_enabled():
            logger.info(f"[COMPILE] Step compiled in {compile_time:.2f}s")

    if not skip_lifecycle:
        dispatch.on_start(config, stack)

    step_history, loss_history = {}, []
    best_params = params if select_best_synced_params else None
    best_metric = (
        float(best_synced_initial_score)
        if (select_best_synced_params and best_synced_initial_score is not None)
        else float("inf")
    )
    best_step = 0
    epoch = -1
    pending_losses = []  # collect losses without forcing sync

    # sync_every=0 means sync at epoch boundaries only
    effective_sync_every = sync_every if sync_every > 0 else steps_per_epoch

    # Progress reporting frequency
    progress_every = max(1, n_total_steps // 20)  # ~5% progress updates

    logger.info(f"[OPTIMIZE] Starting {config.n_epochs} epochs, {n_total_steps} total steps")
    logger.info(f"[OPTIMIZE] Config: {steps_per_epoch} steps/epoch, defer_sync={defer_sync}")
    logger.info(f"[OPTIMIZE] Dispatch: {type(dispatch).__name__}")

    t_loop_start = time.perf_counter()
    epoch_start_time = t_loop_start
    epoch_step_count = 0

    for i, step_key in enumerate(jax.random.split(key, n_total_steps), 1):
        is_epoch_boundary = i % steps_per_epoch == 0
        global_step = step_offset + i
        logger_needs_sync = dispatch.needs_params_sync(global_step)
        should_sync = (
            not defer_sync
            or (i % effective_sync_every == 0)
            or is_epoch_boundary
            or logger_needs_sync
        )

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

        if select_best_synced_params and (should_sync or not defer_sync):
            metric: float | None = None
            if best_synced_score_fn is not None:
                try:
                    metric = best_synced_score_fn(params, step_history, i)
                except Exception as exc:
                    logger.warning(
                        "[OPTIMIZE] Best-synced score function failed at step %d: %s",
                        i,
                        exc,
                    )
            elif "loss" in step_history:
                metric = float(jnp.mean(step_history["loss"]))

            if metric is not None and metric < best_metric:
                best_metric = metric
                best_step = i
                best_params = params

        epoch_step_count += 1

        dispatch.on_step(global_step, config, step_history, stack)

    t_sync = time.perf_counter()
    jax.block_until_ready(params)
    sync_time = time.perf_counter() - t_sync

    # flush any remaining pending losses
    if pending_losses:
        loss_history.extend([float(jnp.mean(l)) for l in pending_losses])
        pending_losses = []

    total_loop_time = time.perf_counter() - t_loop_start

    if not skip_lifecycle:
        dispatch.on_end(step_offset + n_total_steps, config, step_history, stack)

    # Final summary
    logger.info("=" * 60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info(f"  Compilation:    {compile_time:.2f}s")
    logger.info(f"  Loop time:      {total_loop_time:.2f}s ({n_total_steps} steps)")
    logger.info(f"  Final sync:     {sync_time:.2f}s")
    logger.info(f"  Avg step time:  {total_loop_time / n_total_steps * 1000:.2f}ms")
    if loss_history:
        logger.info(f"  Final loss:     {loss_history[-1]:.4f}")
    if select_best_synced_params and best_params is not None:
        has_candidate = best_step > 0 or best_synced_initial_score is not None
        if has_candidate:
            params = best_params
            if best_step > 0:
                metric_name = "score" if best_synced_score_fn is not None else "loss"
                logger.info(
                    f"  Best synced:    step {best_step}/{n_total_steps}, "
                    f"{metric_name}={best_metric:.4f} (restored)"
                )
            else:
                logger.info(
                    f"  Best synced:    baseline score={best_metric:.4f} (restored initial params)"
                )
    # Keep final step-history params aligned with returned params (important when
    # best-synced restoration replaces the terminal optimizer state).
    if isinstance(step_history, dict):
        step_history["latest_params"] = params
        step_history["params"] = params
    logger.info("=" * 60)

    return params, loss_history, StepHistorySnapshot.from_raw(step_history)
