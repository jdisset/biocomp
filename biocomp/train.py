### {{{                          --     imports     --
from . import datautils as du
from biocomp.utils import EncodedPartialFunction
from .parameters import ParameterTree
from . import utils as ut
import time
from typing import List, Tuple, Callable, Optional, NamedTuple
from pydantic import Field
from biocomp.logging_config import get_logger
from biocomp.optimutils import (
    make_training_step,
    per_replicate_step,
    as_schedule,
    OptimConfig,
    compile_step,
    run_logger_callbacks,
    get_checkify_enabled,
)
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
    """Compile training step for reuse across trials (e.g., hyperopt with cached compilation)."""
    import jax
    import jax.numpy as jnp
    from jax.tree_util import Partial
    from .jaxutils import get_looped_slice

    stack = dman.build_compute_stack(compute_config)
    optimizer = (
        training_config.create_optimizer_with_lr_injection()
        if "learning_rate" in training_config.keep_in_history
        else training_config.optimizer
    )

    loss_func = training_config.loss_function.get_impl()(stack, training_config)
    scannable_step = make_training_step(
        loss_func,
        optimizer,
        fields_to_keep_in_history=training_config.keep_in_history,
        scannable=True,
    )

    key = jax.random.PRNGKey(training_config.seed or 42)
    key, init_key, batch_key = jax.random.split(key, 3)

    with ut.timer("Stack initialization (for compilation)", logger):
        sample_params = jax.vmap(stack.init)(
            jax.random.split(init_key, training_config.n_replicates)
        )

    xbatches, ybatches = generate_batches(
        dman,
        training_config.n_replicates,
        training_config.n_batches,
        training_config.batch_size,
        batch_key,
    )
    sample_xb = get_looped_slice(xbatches, 0, training_config.batches_per_step, axis=1)
    sample_yb = get_looped_slice(ybatches, 0, training_config.batches_per_step, axis=1)

    per_output_weights = expand_weights_to_outputs(dman.get_weights(), stack.networks)
    weights_replicated = jnp.broadcast_to(
        jnp.asarray(per_output_weights), (training_config.n_replicates, len(per_output_weights))
    )
    sample_params.at(
        "global/per_output_weights", weights_replicated, tags=["non_grad", "local"], overwrite=True
    )

    static, dynamic = sample_params.filter_by_tag(["non_grad", "local"])
    sample_opt_state = jax.vmap(optimizer.init)(dynamic)

    num_z = static["global/number_of_random_variables"]
    assert num_z.shape == (training_config.n_replicates,) and jnp.all(num_z == num_z[0])
    num_z = int(num_z[0])

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
    compiled = compile_step(
        jitable_base, (sample_params, sample_opt_state, key, sample_xb, sample_yb)
    )
    if not get_checkify_enabled():
        logger.info(f"Compiled training step in {time.time() - t0:.2f} seconds")

    return CompiledTrainingStep(
        compiled_step=compiled, stack=stack, optimizer=optimizer, num_z=num_z
    )


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                      --     loss functions     --


def expand_weights_to_outputs(weights: list[float], networks: list) -> list[float]:
    """Expand per-network weights to per-output weights."""
    return [w for w, n in zip(weights, networks) for _ in range(n.nb_outputs)]


def check_XYZ(X, Y, Z, stack):
    nb_inputs = sum(n.nb_inputs for n in stack.networks)
    nb_outputs = sum(n.nb_outputs for n in stack.networks)
    assert X.ndim == Y.ndim == Z.ndim == 2
    assert X.shape[0] == Y.shape[0] == Z.shape[0]
    assert X.shape[1] == nb_inputs
    assert Y.shape[1] == nb_outputs


def lerp(a, b, t):
    return a + t * (b - a)


def stable_sigma(logstd, *, min_std=1e-3):
    """Forward σ = exp(logσ); backward dσ/dlogσ = softplus(logσ) for stable gradients."""
    import jax
    import jax.numpy as jnp

    sigma_fwd = jnp.exp(logstd)
    sigma_grad = min_std + jax.nn.softplus(logstd)
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
    per_output_weights=None,
):
    """MSE loss with sorted outputs to learn distribution (quantile-like)."""
    import jax
    import jax.numpy as jnp
    from jax.tree_util import tree_leaves
    from .jaxutils import flat_concat, robust_sort

    batch_apply = jax.vmap(stack.apply, in_axes=(None, 0, 0, 0))

    def loss_func(dynamic, static, X, Y, Z, key, step):
        check_XYZ(X, Y, Z, stack)
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
        counts_sum = counts.sum()
        kl_loss = (counts * (qvalues**2 + std**2 - 1 - 2 * jnp.log(std))).sum() / counts_sum * klw

        negative_grads = jnp.mean(jnp.clip(-grads_wrt_inputs, 0, None))
        ngp = as_schedule(negative_grad_penalty)(step)
        ng_loss = negative_grads * ngp

        dep_mask = params["global/dependent_output_mask"]
        indep_mask = ~dep_mask
        assert dep_mask.shape == (stack.total_nb_of_outputs,) == (Y.shape[1],)

        if "global/per_output_weights" in params:
            output_weights = params["global/per_output_weights"]
        elif per_output_weights is not None:
            output_weights = jnp.asarray(per_output_weights)
        else:
            output_weights = jnp.ones(Y.shape[1])
        assert output_weights.shape == (Y.shape[1],)

        pct = as_schedule(percent_batch_used)(step)
        selected = (jnp.linspace(0, 1, X.shape[0]) <= pct)[:, None]
        eff_batch_size = jnp.maximum(selected.sum(), 1)

        sqdiff = (yhat - Y) ** 2 * selected * output_weights[None, :]
        dep_mask_sum = (dep_mask * output_weights).sum()
        mse_dependent = (sqdiff * dep_mask[None, :]).sum() / (eff_batch_size * dep_mask_sum)
        indep_mask_sum = (indep_mask * output_weights).sum()
        mse_independent = (sqdiff * indep_mask[None, :]).sum() / (eff_batch_size * indep_mask_sum)
        out_v_in_mse = as_schedule(out_vs_in_mse_weight)(step)
        mse = lerp(mse_independent, mse_dependent, out_v_in_mse)

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


class TrainingConfig(OptimConfig):
    """Training-specific config extending OptimConfig with training-specific fields."""

    loss_function: EncodedPartialFunction = Field(default=sorting_loss)
    batches_per_step: int = 128  # override default
    n_replicates: int = 1  # override default
    n_batches: int = 2048
    streaming_batches: bool = False  # generate batches on-demand to reduce memory
    clear_source_data: bool = True  # clear DataManager data after batch generation


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
    import jax
    import jax.numpy as jnp
    from jax.tree_util import Partial
    from .jaxutils import get_looped_slice

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
                params = jax.vmap(stack.init)(
                    jax.random.split(init_key, training_config.n_replicates)
                )
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

    # store per_output_weights in params BEFORE filter_by_tag (filter creates copies)
    per_output_weights = expand_weights_to_outputs(dman.get_weights(), stack.networks)
    weights_arr = jnp.asarray(per_output_weights)
    weights_replicated = jnp.broadcast_to(
        weights_arr, (training_config.n_replicates, len(per_output_weights))
    )
    params.at(
        "global/per_output_weights", weights_replicated, tags=["non_grad", "local"], overwrite=True
    )

    if training_config.clear_source_data and not streaming_mode:
        dman.clear_source_data()

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
    if cached_step is not None:
        compiled_step, num_z = cached_step.compiled_step, cached_step.num_z
        logger.info("Using cached compiled training step (no recompilation)")
    else:
        loss_func = training_config.loss_function.get_impl()(stack, training_config)
        scannable_step = make_training_step(
            loss_func,
            optimizer,
            fields_to_keep_in_history=training_config.keep_in_history,
            scannable=True,
        )

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

        xb = get_looped_slice(xbatches, 0, training_config.batches_per_step, axis=1)
        yb = get_looped_slice(ybatches, 0, training_config.batches_per_step, axis=1)

        num_z = static["global/number_of_random_variables"]
        assert num_z.shape == (training_config.n_replicates,) and jnp.all(num_z == num_z[0])
        num_z = int(num_z[0])

        logger.info("Compiling training step...")
        t0 = time.time()
        compiled_step = compile_step(Partial(step, num_z=num_z), (params, opt_state, key, xb, yb))
        if not get_checkify_enabled():
            logger.info(f"Compiled training step in {time.time() - t0:.2f} seconds")

    # --- main training loop
    loggers = loggers or []

    if async_handler:
        async_handler.process_start_loggers(training_config, stack)
    else:
        run_logger_callbacks(loggers, 0, training_config, {}, stack, lambda p, s: p == 0)

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

        run_logger_callbacks(
            loggers, i, training_config, step_history, stack, lambda p, s: p > 0 and s % p == 0
        )

    if not async_handler:
        run_logger_callbacks(
            loggers,
            total_steps,
            training_config,
            step_history,
            stack,
            lambda p, s: p is None or p == -1,
        )

    logger.info(f"End of training for {training_config.n_epochs} epochs")

    return params, loss_history, step_history


##────────────────────────────────────────────────────────────────────────────}}}
