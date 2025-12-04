### {{{                          --     imports     --
from . import datautils as du
from biocomp.utils import (
    EncodedPartialFunction,
    PartialFunction,
    ArbitraryModel,
    PartialFunctionResult,
)
from . import nodes as nodes
from .parameters import ParameterTree, ParamPath
from . import utils as ut
import time
from typing import List, Tuple, Callable, Optional, NamedTuple
from pydantic import Field
from biocomp.logging_config import get_logger
from biocomp.optimutils import make_training_step, per_replicate_step, per_replicate_step_nonscan
##────────────────────────────────────────────────────────────────────────────}}}

### {{{                     --     helper functions     --

logger = get_logger(__name__)


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


class CompiledTrainingStep(NamedTuple):
    """Pre-compiled training step for reuse across trials with different weights.

    When optimizing dataset weights in hyperopt, the ComputeStack structure is
    identical across trials - only the weights change. By caching the compiled
    step and stack, we avoid expensive JIT recompilation each trial.
    """

    compiled_step: Callable
    stack: "ComputeStack"  # noqa: F821
    optimizer: "optax.GradientTransformation"  # noqa: F821
    num_z: int


def compile_training_step(
    dman: du.DataManager,
    training_config: "TrainingConfig",
    compute_config,
    enable_jax_tqdm: bool = False,
) -> CompiledTrainingStep:
    """Compile training step for reuse across trials.

    This function extracts the expensive JIT compilation into a cacheable unit.
    The resulting CompiledTrainingStep can be reused across hyperopt trials
    when only the weights (stored in params) change.

    Args:
        dman: DataManager (used to build the stack)
        training_config: Training configuration
        compute_config: Compute configuration
        enable_jax_tqdm: Whether to enable tqdm in training

    Returns:
        CompiledTrainingStep containing the compiled step and associated objects
    """
    import jax
    import jax.numpy as jnp
    from jax.tree_util import Partial
    import os
    from jax.experimental import checkify
    from .jaxutils import get_looped_slice

    BIOCOMP_CHECKIFY = os.environ.get("BIOCOMP_CHECKIFY", "").lower() in ("true", "1", "yes", "on")

    # build stack (this is the key thing we want to cache)
    stack = dman.build_compute_stack(compute_config)

    # Use learning rate injection if learning_rate is requested in history
    if "learning_rate" in training_config.keep_in_history:
        optimizer = training_config.create_optimizer_with_lr_injection()
    else:
        optimizer = training_config.optimizer

    # create loss function (captures stack in closure)
    loss_func_generator = training_config.loss_function.get_impl()
    loss_func = loss_func_generator(stack, training_config)
    assert callable(loss_func)

    scannable_step = make_training_step(
        loss_func,
        optimizer,
        fields_to_keep_in_history=training_config.keep_in_history,
        scannable=True,
    )

    # init params and batches for compilation
    key = jax.random.PRNGKey(training_config.seed or 42)
    key, init_key, batch_key = jax.random.split(key, 3)

    with ut.timer("Stack initialization (for compilation)", logger):
        sample_params = jax.vmap(stack.init)(jax.random.split(init_key, training_config.n_replicates))

    # generate sample batch
    xbatches, ybatches = generate_batches(
        dman,
        training_config.n_replicates,
        training_config.n_batches,
        training_config.batch_size,
        batch_key,
    )
    sample_xb = get_looped_slice(xbatches, 0, training_config.batches_per_step, axis=1)
    sample_yb = get_looped_slice(ybatches, 0, training_config.batches_per_step, axis=1)

    # init optimizer state (must use same optimizer as the step for pytree matching)
    static, dynamic = sample_params.filter_by_tag(["non_grad", "local"])
    sample_opt_state = jax.vmap(optimizer.init)(dynamic)

    # get num_z from sample params
    num_z = static["global/number_of_random_variables"]
    assert num_z.shape == (training_config.n_replicates,)
    assert jnp.all(num_z == num_z[0]), "All replicates must have the same number of quantile variables"
    num_z = int(num_z[0])

    # add per_output_weights to sample_params for pytree structure matching
    # (start() adds this, so we need it during compilation for caching to work)
    per_output_weights = expand_weights_to_outputs(dman.get_weights(), stack.networks)
    weights_arr = jnp.asarray(per_output_weights)
    weights_replicated = jnp.broadcast_to(weights_arr, (training_config.n_replicates, len(per_output_weights)))
    sample_params.at("global/per_output_weights", weights_replicated, tags=["non_grad", "local"], overwrite=True)

    def step(params: ParameterTree, opt_state, step_key, xs, ys, num_z):
        keys = jax.random.split(step_key, training_config.n_replicates)
        return jax.vmap(
            Partial(
                per_replicate_step,
                num_z=num_z,
                training_config=training_config,
                scannable_step=scannable_step,
                enable_jax_tqdm=enable_jax_tqdm,
            )
        )(params, opt_state, keys, xs, ys)

    logger.info("Compiling training step (for caching)...")
    t0 = time.time()
    jitable_base = Partial(step, num_z=num_z)
    if not BIOCOMP_CHECKIFY:
        lowered = jax.jit(jitable_base).lower(sample_params, sample_opt_state, key, sample_xb, sample_yb)
        compiled_step = lowered.compile()
        logger.info(f"Compiled training step in {time.time() - t0:.2f} seconds")
    else:
        ckf = jax.jit(checkify.checkify(jitable_base, errors=checkify.all_checks))

        def checkified_step(params, opt_state, step_key, xs, ys):
            err, data = ckf(params, opt_state, step_key, xs, ys)
            err.throw()
            return data

        compiled_step = checkified_step

    return CompiledTrainingStep(
        compiled_step=compiled_step,
        stack=stack,
        optimizer=optimizer,
        num_z=num_z,
    )


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                      --     loss functions     --


def expand_weights_to_outputs(weights: list[float], networks: list) -> list[float]:
    """Expand per-network weights to per-output weights."""
    return [w for w, n in zip(weights, networks) for _ in range(n.nb_outputs)]


def check_XYZ(X, Y, Z, stack):
    nb_inputs = sum([n.nb_inputs for n in stack.networks])
    nb_outputs = sum([n.nb_outputs for n in stack.networks])
    assert X.ndim == Y.ndim == Z.ndim == 2, "X, Y, and Z must have 2 dimensions"
    assert X.shape[0] == Y.shape[0] == Z.shape[0], "X, Y, and Z must have the same number of rows"
    assert X.shape[1] == nb_inputs, (
        "X must have as many columns as the total number of inputs in the stack"
    )
    assert Y.shape[1] == Z.shape[1] == nb_outputs, (
        "Y and Z must have as many columns as the total number of outputs in the stack"
    )


def check_XYZ_new(X, Y, Z, stack):
    nb_inputs = sum([n.nb_inputs for n in stack.networks])
    nb_outputs = sum([n.nb_outputs for n in stack.networks])
    nb_nodes = len(stack.node_map)
    assert X.ndim == Y.ndim == Z.ndim == 2, "X, Y, and Z must have 2 dimensions"
    assert X.shape[0] == Y.shape[0] == Z.shape[0], "X, Y, and Z must have the same number of rows"
    assert X.shape[1] == nb_inputs, (
        "X must have as many columns as the total number of inputs in the stack"
    )
    assert Y.shape[1] == nb_outputs, (
        "Y must have as many columns as the total number of outputs in the stack"
    )


def as_schedule(value_or_callable):
    import jax.numpy as jnp

    if callable(value_or_callable):
        return value_or_callable

    def f(step):
        return jnp.asarray(value_or_callable)

    return f


def lerp(a, b, t):
    # when t=0 return a, when t=1 return b
    return a + t * (b - a)


def stable_sigma(logstd, *, min_std=1e-3):
    """Forward σ ≡ exp(logσ); backward dσ/dlogσ ≡ sigmoid(logσ)."""
    import jax.numpy as jnp
    import jax

    sigma_fwd = jnp.exp(logstd)  # keeps identical activations
    sigma_grad = min_std + jax.nn.softplus(logstd)  # nice, ≥0.25 derivative
    # swap in the softplus derivative, keep forward value
    return sigma_fwd + jax.lax.stop_gradient(sigma_grad - sigma_fwd)


def sorting_loss(
    stack,
    training_config,
    negative_grad_penalty=1.0,
    kl_weight=0.1,
    sorting_mse_weight=0.1,
    percent_batch_used=1.0,
    out_vs_in_mse_weight=0.5,
    out_vs_in_sortmse_weight=1,
    use_same_key=False,
    per_output_weights=None,  # deprecated, use global/per_output_weights in params
):
    import jax
    import jax.numpy as jnp
    from jax.tree_util import tree_leaves
    from .jaxutils import flat_concat, robust_sort

    # sorting loss attempts to make the model learn the distribution rather than just the mean
    # it tries to learn the quantile function - sort of...
    # does so by feeding a random variable to each node, and compute the loss as the mse between
    # the sorted model outputs and the sorted targets. Which, ultimately, sort of leads to learning the quantile function.

    batch_apply = jax.vmap(stack.apply, in_axes=(None, 0, 0, 0))

    def loss_func(dynamic, static, X, Y, Z, key, step):
        check_XYZ_new(X, Y, Z, stack)
        params = ParameterTree.merge(dynamic, static)

        if use_same_key:
            keys = jnp.array([key] * X.shape[0])
        else:
            keys = jax.random.split(key, X.shape[0])

        yhat, (apply_aux, full_output) = batch_apply(params, X, Z, keys)

        grads_wrt_inputs = apply_aux["grads_wrt_inputs"]
        aux = {"yhat": yhat, "grads_wrt_inputs": grads_wrt_inputs, "full_output": full_output}

        assert isinstance(yhat, jnp.ndarray)
        assert isinstance(Y, jnp.ndarray)
        assert yhat.shape == Y.shape, (
            f"yhat and Y must have the same shape, got {yhat.shape} and {Y.shape}"
        )

        # kl divergence for smooth embeddings
        klw = as_schedule(kl_weight)(step)
        qvalues = flat_concat(*tree_leaves(params["shared/quantization/values"]))
        logstds = flat_concat(*tree_leaves(params["shared/quantization/logstdevs"]))
        counts = flat_concat(*tree_leaves(params["shared/quantization/counts"]))

        std = stable_sigma(logstds, min_std=1e-3)
        # Check for division by zero in KL loss
        counts_sum = counts.sum()
        kl_loss = (counts * (qvalues**2 + std**2 - 1 - 2 * jnp.log(std))).sum() / counts_sum * klw

        # negative grads, used to penalize "inverted" functions
        negative_grads = jnp.mean(jnp.clip(-grads_wrt_inputs, 0, None))
        ngp = as_schedule(negative_grad_penalty)(step)
        ng_loss = negative_grads * ngp

        # weigh the dependent outputs more than the independent variables (aka inputs)
        dep_mask = params["global/dependent_output_mask"]
        indep_mask = ~dep_mask
        assert dep_mask.shape == (stack.total_nb_of_outputs,) == (Y.shape[1],), (
            f"dep_out must have the same shape as Y features, got {dep_mask.shape} and {Y.shape}"
        )

        # per-network weights (expanded to per-output) - read from params for JIT caching
        if "global/per_output_weights" in params:
            output_weights = params["global/per_output_weights"]
        elif per_output_weights is not None:
            # fallback for backwards compatibility
            output_weights = jnp.asarray(per_output_weights)
        else:
            output_weights = jnp.ones(Y.shape[1])
        assert output_weights.shape == (Y.shape[1],)

        # only use a percentage of the batch (allows to vary batch size without recompiling)
        pct = as_schedule(percent_batch_used)(step)
        selected = (jnp.linspace(0, 1, X.shape[0]) <= pct)[:, None]
        eff_batch_size = jnp.maximum(selected.sum(), 1)

        # compute the mse loss (with per-network weights)
        sqdiff = (yhat - Y) ** 2 * selected * output_weights[None, :]
        dep_mask_sum = (dep_mask * output_weights).sum()
        mse_dependent = (sqdiff * dep_mask[None, :]).sum() / (eff_batch_size * dep_mask_sum)
        indep_mask_sum = (indep_mask * output_weights).sum()
        mse_independent = (sqdiff * indep_mask[None, :]).sum() / (eff_batch_size * indep_mask_sum)
        out_v_in_mse = as_schedule(out_vs_in_mse_weight)(step)
        mse = lerp(mse_independent, mse_dependent, out_v_in_mse)

        # sorting loss with pushing masked out values to the end
        MAXFLOAT = jnp.finfo(Y.dtype).max
        sorted_yhat = robust_sort(jnp.where(selected, yhat, MAXFLOAT), axis=0)
        sorted_y = robust_sort(jnp.where(selected, Y, MAXFLOAT), axis=0)
        sort_sqdiff = (sorted_yhat - sorted_y) ** 2 * output_weights[None, :]
        out_v_in_sortmse = as_schedule(out_vs_in_sortmse_weight)(step)
        sorting_mse_dependent = (sort_sqdiff * dep_mask[None, :]).sum() / (
            eff_batch_size * dep_mask_sum
        )
        sorting_mse_independent = (sort_sqdiff * indep_mask[None, :]).sum() / (
            eff_batch_size * indep_mask_sum
        )
        sorting_mse = lerp(sorting_mse_independent, sorting_mse_dependent, out_v_in_sortmse)

        # mix the two losses
        smw = as_schedule(sorting_mse_weight)(step)
        main_loss = lerp(mse, sorting_mse, smw)

        aux["sublosses"] = {
            "mse": mse,
            "sorting_mse": sorting_mse,
            "kl_loss": kl_loss,
            "main_loss": main_loss,
        }

        aux["debug"] = {
            "negative_grads": negative_grads,
            "ng_loss": ng_loss,
            "effective_batch_size": eff_batch_size,
            "std": std,
            "selected": selected,
            "full_mse": ((yhat - Y) ** 2).mean(),
            "full_rmse": jnp.sqrt(((yhat - Y) ** 2).mean()),
            "sorted_yhat": sorted_yhat,
            "sorted_y": sorted_y,
            "pct": pct,
            "kl_weight": klw,
            "negative_grad_penalty": ngp,
            "sqdiff": sqdiff,
            "sort_sqdiff": sort_sqdiff,
            "mse_dependent": mse_dependent,
            "mse_independent": mse_independent,
            "sorting_mse_dependent": sorting_mse_dependent,
            "sorting_mse_independent": sorting_mse_independent,
            "out_v_in_mse": out_v_in_mse,
            "out_v_in_sortmse": out_v_in_sortmse,
            "qvalues": qvalues,
            "logstds": logstds,
            "counts": counts,
            "step": step,
            "output_weights": output_weights,
        }
        aux["apply_aux"] = apply_aux

        return main_loss + kl_loss + ng_loss, aux

    return loss_func


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                  --     training config     --


def create_counter():
    """Creates a no-op gradient transformation that just counts steps."""
    import jax.numpy as jnp
    import optax

    class CounterState(NamedTuple):
        count: jnp.ndarray  # type: ignore

    def init_fn(params):
        return CounterState(count=jnp.zeros([], jnp.int32))

    def update_fn(updates, state, params=None):
        return updates, CounterState(count=state.count + 1)

    return optax.GradientTransformation(init_fn, update_fn)


DEFAULT_OPTIMIZER = [
    PartialFunction(
        # func=optax.clip_by_global_norm,
        func="optax.transforms._clipping.clip_by_global_norm",
        kwargs={"max_norm": 1.0},
    ),
    PartialFunction(
        # func=optax.adamw,
        func="optax._src.alias.adamw",
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


class TrainingConfig(ArbitraryModel):
    # training parameters
    optimizer_stack: list[EncodedPartialFunction] = DEFAULT_OPTIMIZER
    loss_function: EncodedPartialFunction = Field(default=sorting_loss)

    seed: Optional[int] = None
    batches_per_step: int = 128
    batch_size: int = 32
    n_epochs: float = 3
    n_batches: int = 2048  # can't really have "real" epochs because each network has a different qtty of data points
    n_replicates: int = 1
    keep_in_history: List[str] = ["loss"]

    # memory optimization: generate batches on-demand instead of pre-generating all
    # reduces GPU memory usage significantly, enabling parallel training processes
    streaming_batches: bool = False

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        if self.seed is None:
            import random

            self.seed = random.randint(0, 2**32 - 1)

    @property
    def optimizer(self):
        import optax

        main_chain = [comp() for comp in self.optimizer_stack]
        return optax.chain(create_counter(), *main_chain)

    def create_optimizer_with_lr_injection(self):
        """Create optimizer with learning rate injection for debugging purposes."""
        import optax

        # Try to detect and inject learning rates for better tracking
        main_chain = []

        for comp in self.optimizer_stack:
            # Check if this component has a learning_rate parameter
            if hasattr(comp, "kwargs") and "learning_rate" in comp.kwargs:
                # Get the original function
                if hasattr(comp, "func"):
                    original_func = comp.func
                elif hasattr(comp, "_func"):
                    original_func = comp._func
                else:
                    # Fallback to regular instantiation
                    main_chain.append(comp())
                    continue

                # Handle string function references
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
                    # For learning rate injection to work, we need to resolve PartialFunctionResult first
                    lr_value = comp.kwargs["learning_rate"]
                    if hasattr(lr_value, "get_impl"):
                        # This is a PartialFunctionResult, resolve it to get the actual schedule
                        lr_schedule = lr_value.get_impl()()  # Call the schedule function
                    else:
                        lr_schedule = lr_value

                    # Create wrapped version with inject_hyperparams
                    wrapped_func = optax.inject_hyperparams(func)

                    # Get all other kwargs (excluding learning_rate)
                    other_kwargs = {k: v for k, v in comp.kwargs.items() if k != "learning_rate"}

                    # Create the optimizer instance with injected learning rate
                    optimizer_instance = wrapped_func(learning_rate=lr_schedule, **other_kwargs)
                    main_chain.append(optimizer_instance)
                except Exception:
                    # Fallback to regular instantiation if injection fails
                    main_chain.append(comp())
            else:
                main_chain.append(comp())

        return optax.chain(create_counter(), *main_chain)


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                       --     main training function     --


def start(
    dman: du.DataManager,
    training_config: TrainingConfig,
    compute_config,
    loggers: Optional[List[Tuple[int, Callable]]] = None,
    xy_batches: Optional[Tuple] = None,
    async_handler=None,
    enable_jax_tqdm: bool = False,
    init_params: Optional[ParameterTree] = None,
    cached_step: Optional[CompiledTrainingStep] = None,
):
    import optax
    import jax

    if enable_jax_tqdm:
        from jax_tqdm import scan_tqdm
    import jax.numpy as jnp
    from jax.tree_util import Partial
    from jax import vmap, jit
    from .jaxutils import get_looped_slice
    import os
    from jax.experimental import checkify

    BIOCOMP_CHECKIFY = os.environ.get("BIOCOMP_CHECKIFY", "").lower() in ("true", "1", "yes", "on")

    logger.debug(f"Training config: {training_config}")
    logger.debug(f"Compute config: {compute_config}")

    # --- init & batches generation
    assert training_config.seed is not None, "Seed must be set"
    key = jax.random.PRNGKey(training_config.seed)  # {{{}}}
    key, init_key, batch_key, loop_key = jax.random.split(key, 4)

    total_steps = int(
        training_config.n_epochs * training_config.n_batches / training_config.batches_per_step
    )

    # use cached stack if provided (for hyperopt with cached compilation)
    if cached_step is not None:
        stack = cached_step.stack
        if init_params is not None:
            params = init_params
        else:
            # init fresh params using the cached stack
            with ut.timer("Stack initialization", logger):
                params = jax.vmap(stack.init)(jax.random.split(init_key, training_config.n_replicates))
    elif init_params is None:
        stack, params = init_stack(compute_config, dman, training_config.n_replicates, init_key)
    else:
        stack = dman.build_compute_stack(compute_config)
        params = init_params

        n_reps = training_config.n_replicates
        num_z_check = params["global/number_of_random_variables"]
        num_z_arr = jnp.asarray(num_z_check)

        if num_z_arr.shape != (n_reps,):
            logger.info(f"Replicating provided init_params {n_reps} times to match n_replicates.")

            def replicate_leaf(x):
                if x is None:
                    return None
                x_arr = jnp.asarray(x)
                if x_arr.ndim == 0:
                    x_arr = x_arr[None]
                return jnp.repeat(x_arr[None, ...], n_reps, axis=0)

            params = jax.tree.map(replicate_leaf, params)
        else:
            logger.info(
                f"Using provided init_params as is (assumed to be already replicated with size {n_reps})."
            )

    def get_new_batches(rng_key=batch_key):
        xbatches, ybatches = generate_batches(
            dman,
            training_config.n_replicates,
            training_config.n_batches,
            training_config.batch_size,
            rng_key,
        )
        assert xbatches.shape == (
            training_config.n_replicates,
            training_config.n_batches,
            training_config.batch_size,
            stack.total_nb_of_inputs,
        ), (
            f"xbatches shape mismatch: {xbatches.shape} != ({training_config.n_replicates}, {training_config.n_batches}, {training_config.batch_size}, {stack.total_nb_of_inputs}"
        )
        assert ybatches.shape == (
            training_config.n_replicates,
            training_config.n_batches,
            training_config.batch_size,
            stack.total_nb_of_outputs,
        ), (
            f"ybatches shape mismatch: {ybatches.shape} != ({training_config.n_replicates}, {training_config.n_batches}, {training_config.batch_size}, {stack.total_nb_of_outputs})"
        )

        xbatches_arr = jnp.asarray(xbatches)
        ybatches_arr = jnp.asarray(ybatches)

        return xbatches_arr, ybatches_arr

    def get_step_batches(rng_key):
        """Generate batches for a single step only (streaming mode)."""
        xbatches, ybatches = generate_batches(
            dman,
            training_config.n_replicates,
            training_config.batches_per_step,  # only generate what we need for one step
            training_config.batch_size,
            rng_key,
        )
        return jnp.asarray(xbatches), jnp.asarray(ybatches)

    streaming_mode = training_config.streaming_batches
    if streaming_mode:
        logger.info("Using streaming batch generation (lower GPU memory, slightly slower)")
        # generate a small batch just for shape inference during compilation
        xbatches, ybatches = get_step_batches(batch_key)
    elif xy_batches is not None:
        xbatches, ybatches = xy_batches
    else:
        xbatches, ybatches = get_new_batches()

    static, dynamic = params.filter_by_tag(["non_grad", "local"])

    # use cached optimizer if available, otherwise create new one
    if cached_step is not None:
        optimizer = cached_step.optimizer
    elif "learning_rate" in training_config.keep_in_history:
        optimizer = training_config.create_optimizer_with_lr_injection()
    else:
        optimizer = training_config.optimizer
    opt_state = jax.vmap(optimizer.init)(dynamic)

    logger.info(
        f"""Done initializing optimizer,
        n_replicates: {training_config.n_replicates}
        batches: {xbatches.shape[1]}
        batch per step: {training_config.batches_per_step}
        random seed: {training_config.seed}"""
    )

    # --- loss & update functions
    # store per_output_weights in params for JIT caching (weights can change without recompilation)
    per_output_weights = expand_weights_to_outputs(dman.get_weights(), stack.networks)
    # add replicate dimension to match params structure (will be sliced by vmap in loss func)
    weights_arr = jnp.asarray(per_output_weights)
    weights_replicated = jnp.broadcast_to(weights_arr, (training_config.n_replicates, len(per_output_weights)))
    params.at("global/per_output_weights", weights_replicated, tags=["non_grad", "local"], overwrite=True)

    # use cached compiled step if available, otherwise build and compile
    if cached_step is not None:
        compiled_step = cached_step.compiled_step
        num_z = cached_step.num_z
        logger.info("Using cached compiled training step (no recompilation)")
    else:
        loss_func_generator = training_config.loss_function.get_impl()
        loss_func = loss_func_generator(stack, training_config)  # weights read from params now
        assert callable(loss_func)
        scannable_step = make_training_step(
            loss_func,
            optimizer,
            fields_to_keep_in_history=training_config.keep_in_history,
            scannable=True,
        )

        non_scannable_step = make_training_step(
            loss_func,
            optimizer,
            fields_to_keep_in_history=training_config.keep_in_history,
            scannable=False,
        )

        def step(params: ParameterTree, opt_state: optax.OptState, step_key, xs, ys, num_z):
            keys = jax.random.split(step_key, training_config.n_replicates)
            assert (
                xs.shape[:-1]
                == ys.shape[:-1]
                == (
                    training_config.n_replicates,
                    training_config.batches_per_step,
                    training_config.batch_size,
                )
            )
            return jax.vmap(
                Partial(
                    per_replicate_step,
                    num_z=num_z,
                    training_config=training_config,
                    scannable_step=scannable_step,
                    enable_jax_tqdm=enable_jax_tqdm,
                )
            )(params, opt_state, keys, xs, ys)

        xb = get_looped_slice(xbatches, 0, training_config.batches_per_step, axis=1)
        yb = get_looped_slice(ybatches, 0, training_config.batches_per_step, axis=1)

        num_z = static["global/number_of_random_variables"]
        assert num_z.shape == (training_config.n_replicates,)
        assert jnp.all(num_z == num_z[0]), (
            "All replicates must have the same number of quantile variables"
        )
        num_z = int(num_z[0])

        logger.info("Compiling training step...")
        t0 = time.time()
        jitable_base = Partial(step, num_z=num_z)
        if not BIOCOMP_CHECKIFY:
            lowered = jax.jit(jitable_base).lower(params, opt_state, key, xb, yb)
            compiled_step = lowered.compile()
            logger.info(f"Compiled training step in {time.time() - t0:.2f} seconds")
        else:
            ckf = jax.jit(checkify.checkify(jitable_base, errors=checkify.all_checks))

            def checkified_step(params, opt_state, step_key, xs, ys):
                err, data = ckf(params, opt_state, step_key, xs, ys)
                err.throw()
                return data

            compiled_step = checkified_step

    # --- main training loop
    loggers = loggers or []

    # call start-of-training loggers (period=0)
    if async_handler:
        async_handler.process_start_loggers(training_config, stack)
    else:
        for period, callback in loggers:
            if period == 0:
                try:
                    callback(0, training_config, step_history={}, stack=stack)
                except Exception as e:
                    logger.error(f"Start logger callback failed: {e}")
                    logger.exception(e)

    logger.info(f"Running for {total_steps} iterations")

    step_history, loss_history = {}, []
    epoch = -1

    step_per_epoch = training_config.n_batches // training_config.batches_per_step

    for i, step_key in enumerate(jax.random.split(loop_key, total_steps), 1):
        if i % max(1, total_steps // 20) == 0:  # Log every 5% progress
            logger.info(f"Training progress: [{i}/{total_steps}] ({i / total_steps * 100:.1f}%)")

        t0 = time.time()

        if streaming_mode:
            # streaming: generate fresh batches for this step only
            b_key = jax.random.fold_in(step_key, i)
            xb, yb = get_step_batches(b_key)
        else:
            # pre-generated: slice from full batch array, regenerate each epoch
            if i % step_per_epoch == 0:
                epoch += 1
                logger.info(f"Starting epoch {epoch}")
                b_key = jax.random.fold_in(step_key, epoch)
                xbatches, ybatches = get_new_batches(b_key)

            xb = get_looped_slice(
                xbatches,
                i * training_config.batches_per_step,
                (i + 1) * training_config.batches_per_step,
                axis=1,
            )
            yb = get_looped_slice(
                ybatches,
                i * training_config.batches_per_step,
                (i + 1) * training_config.batches_per_step,
                axis=1,
            )

        params, opt_state, step_history = compiled_step(params, opt_state, step_key, xb, yb)

        step_history["step_time"] = time.time() - t0
        step_history["latest_params"] = params
        step_history["opt_state"] = opt_state

        if "loss" in step_history:
            loss_history.append(step_history["loss"])

        # call logger callbacks at their specified periods
        for period, callback in loggers:
            if period > 0 and i % period == 0:
                try:
                    callback(i, training_config, step_history=step_history, stack=stack)
                except Exception as e:
                    logger.error(f"Logger callback failed at step {i}: {e}")
                    logger.exception(e)

    # call end-of-training loggers (period=None or -1)
    if not async_handler:  # end loggers handled separately for async mode
        for period, callback in loggers:
            if period is None or period == -1:
                try:
                    callback(total_steps, training_config, step_history=step_history, stack=stack)
                except Exception as e:
                    logger.error(f"End logger callback failed: {e}")
                    logger.exception(e)

    logger.info(f"End of training for {training_config.n_epochs} epochs")

    return params, loss_history, step_history


##────────────────────────────────────────────────────────────────────────────}}}
