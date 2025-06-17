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
import asyncio
from .async_logger import AsyncLoggerManager
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
        
        # Handle empty static parameters properly
        if static.data:  # If static has data
            params = ParameterTree.merge(static, dynamic)
        else:  # If static is empty, just use dynamic
            params = dynamic
        
        # Try to extract learning rate from optimizer state (best effort)
        learning_rate = None
        try:
            # Method 1: Check for hyperparams in individual state components
            # opt_state is typically a tuple of states from chained optimizers
            if isinstance(opt_state, tuple):
                for state_component in opt_state:
                    if hasattr(state_component, 'hyperparams') and 'learning_rate' in state_component.hyperparams:
                        learning_rate = state_component.hyperparams['learning_rate']
                        break
            
            # Method 2: Check direct hyperparams access
            if learning_rate is None and hasattr(opt_state, 'hyperparams') and 'learning_rate' in opt_state.hyperparams:
                learning_rate = opt_state.hyperparams['learning_rate']
            
            # Method 3: Try tree_get as fallback (might fail with multiple matches)
            if learning_rate is None:
                try:
                    learning_rate = optax.tree_utils.tree_get(
                        opt_state, 'learning_rate',
                        default=None,
                        filtering=lambda path, value: isinstance(value, (float, int)) or (hasattr(value, 'shape') and hasattr(value, 'dtype'))
                    )
                except (KeyError, ValueError):
                    # Multiple learning_rate entries or other tree_get issues
                    pass
                    
        except Exception:
            # If all methods fail, learning_rate remains None
            pass
        
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
        
        # Add learning rate to result if available
        if learning_rate is not None:
            res["learning_rate"] = learning_rate
            
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

### {{{                      --     loss functions     --


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


def l2_loss(stack, training_config, negative_grad_penalty=1.0, kl_weight=1):
    import jax
    import jax.numpy as jnp

    batch_apply = jax.vmap(stack.apply, in_axes=(None, 0, 0, 0))

    def loss_func(dynamic, static, X, Y, Z, key, step):
        check_XYZ_new(X, Y, Z, stack)
        params = ParameterTree.merge(dynamic, static)
        keys = jax.random.split(key, X.shape[0])
        # jax.debug.print("X {}", X)
        # jax.debug.print("Z {}", Z)
        yhat, (grads_wrt_inputs, full_output) = batch_apply(params, X, Z, keys)
        assert yhat.shape == Y.shape, "yhat and Y must have the same shape"
        aux = {"yhat": yhat, "grads_wrt_inputs": grads_wrt_inputs, "full_output": full_output}

        qvalues_dir = ParamPath("shared/quantization/values")
        logstd_dir = ParamPath("shared/quantization/logstdevs")
        count_dir = ParamPath("shared/quantization/counts")
        klw = as_schedule(kl_weight)(step)
        qvalues, logstds, counts = map(
            lambda path: jnp.concatenate(
                tuple(map(lambda t: t[1], params[path].iter_leaves()))
            ).flatten(),
            (qvalues_dir, logstd_dir, count_dir),
        )
        kl_loss = (
            counts * (qvalues**2 + jnp.exp(2 * logstds) / 2 - logstds - 0.5)
        ).sum() / counts.sum()

        mse = ((yhat - Y) ** 2).mean()
        negative_grads = jnp.mean(jnp.clip(-grads_wrt_inputs, 0, None))

        ngp = as_schedule(negative_grad_penalty)(step)

        loss = mse + ngp * negative_grads + klw * kl_loss

        return loss, aux

    return loss_func


def lerp(a, b, t):
    # when t=0 return a, when t=1 return b
    return a + t * (b - a)


def sorting_loss(
    stack,
    training_config,
    negative_grad_penalty=1.0,  # favor monotonicity
    kl_weight=0.1,
    sorting_mse_weight=0.1,
    percent_batch_used=1.0,
    use_same_key=False,
):
    import jax
    import jax.numpy as jnp

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

        yhat, (grads_wrt_inputs, full_output) = batch_apply(params, X, Z, keys)
        aux = {"yhat": yhat, "grads_wrt_inputs": grads_wrt_inputs, "full_output": full_output}

        assert isinstance(yhat, jnp.ndarray)
        assert isinstance(Y, jnp.ndarray)
        assert yhat.shape == Y.shape, (
            f"yhat and Y must have the same shape, got {yhat.shape} and {Y.shape}"
        )

        # kl
        qvalues_dir = ParamPath("shared/quantization/values")
        logstd_dir = ParamPath("shared/quantization/logstdevs")
        count_dir = ParamPath("shared/quantization/counts")
        klw = as_schedule(kl_weight)(step)
        qvalues, logstds, counts = map(
            lambda path: jnp.concatenate(
                tuple(map(lambda t: t[1], params[path].iter_leaves()))
            ).flatten(),
            (qvalues_dir, logstd_dir, count_dir),
        )
        kl_loss = (
            (counts * (qvalues**2 + jnp.exp(2 * logstds) / 2 - logstds - 0.5)).sum()
            / counts.sum()
            * klw
        )

        # negative grads, used to penalize "inverted" functions
        negative_grads = jnp.mean(jnp.clip(-grads_wrt_inputs, 0, None))
        ngp = as_schedule(negative_grad_penalty)(step)
        ng_loss = negative_grads * ngp

        # only use a percentage of the batch (allows to vary batch size without recompiling)
        pct = as_schedule(percent_batch_used)(step)
        selected = (jnp.linspace(0, 1, X.shape[0]) <= pct)[:, None]
        count = jnp.maximum(selected.sum(), 1)

        mse = ((yhat - Y) ** 2 * selected).sum() / count

        # sorting loss with pushing masked out values to the end
        MAXFLOAT = jnp.finfo(Y.dtype).max

        sorting_mse = (
            jnp.sort(jnp.where(selected, yhat, MAXFLOAT), axis=0)
            - jnp.sort(jnp.where(selected, Y, MAXFLOAT), axis=0)
        ) ** 2
        sorting_mse = sorting_mse.sum() / count

        # mix the two losses
        smw = as_schedule(sorting_mse_weight)(step)
        main_loss = lerp(mse, sorting_mse, smw)

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
    # PartialFunction(
    #     # func=optax.clip_by_global_norm,
    #     func="optax.transforms._clipping.clip_by_global_norm",
    #     kwargs={"max_norm": 1.0},
    # ),
    # PartialFunction(
    #     # func=optax.adamw,
    #     func="optax._src.alias.adamw",
    #     kwargs={
    #         "learning_rate": PartialFunctionResult(
    #             func="optax.warmup_cosine_decay_schedule",
    #             kwargs={
    #                 "init_value": 1e-7,
    #                 "peak_value": 1e-3,
    #                 "warmup_steps": 15,
    #                 "decay_steps": 130,
    #                 "end_value": 1e-5,
    #             },
    #         )
    #     },
    # ),
]


class TrainingConfig(ArbitraryModel):
    # training parameters
    optimizer_stack: list[EncodedPartialFunction] = DEFAULT_OPTIMIZER
    loss_function: EncodedPartialFunction = Field(default=l2_loss)

    seed: Optional[int] = None
    batches_per_step: int = 128
    batch_size: int = 32
    n_epochs: float = 3
    n_batches: int = 2048  # can't really have "real" epochs because each network has a different qtty of data points
    n_replicates: int = 1
    keep_in_history: List[str] = ["loss"]

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
            if hasattr(comp, 'kwargs') and 'learning_rate' in comp.kwargs:
                # Get the original function
                if hasattr(comp, 'func'):
                    original_func = comp.func
                elif hasattr(comp, '_func'):
                    original_func = comp._func
                else:
                    # Fallback to regular instantiation
                    main_chain.append(comp())
                    continue
                
                # Handle string function references
                if isinstance(original_func, str):
                    import importlib
                    try:
                        module_name, func_name = original_func.rsplit('.', 1)
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
                    lr_value = comp.kwargs['learning_rate']
                    if hasattr(lr_value, 'get_impl'):
                        # This is a PartialFunctionResult, resolve it to get the actual schedule
                        lr_schedule = lr_value.get_impl()()  # Call the schedule function
                    else:
                        lr_schedule = lr_value
                    
                    # Create wrapped version with inject_hyperparams
                    wrapped_func = optax.inject_hyperparams(func)
                    
                    # Get all other kwargs (excluding learning_rate)
                    other_kwargs = {k: v for k, v in comp.kwargs.items() if k != 'learning_rate'}
                    
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


async def start(
    dman: du.DataManager,
    training_config: TrainingConfig,
    compute_config,
    loggers: Optional[List[Tuple[int, Callable]]] = None,
    xy_batches: Optional[Tuple] = None,
):
    import optax
    import jax
    from jax_tqdm import scan_tqdm
    import jax.numpy as jnp
    from jax.tree_util import Partial
    from jax import vmap, jit
    from .jaxutils import get_looped_slice

    logger.debug(f"Training config: {training_config}")
    logger.debug(f"Compute config: {compute_config}")

    # --- init & batches generation
    assert training_config.seed is not None, "Seed must be set"
    key = jax.random.PRNGKey(training_config.seed)
    key, init_key, batch_key, loop_key = jax.random.split(key, 4)

    total_steps = int(
        training_config.n_epochs * training_config.n_batches / training_config.batches_per_step
    )

    stack, params = init_stack(compute_config, dman, training_config.n_replicates, init_key)

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
            stack.nb_inputs,
        ), (
            f"xbatches shape mismatch: {xbatches.shape} != ({training_config.n_replicates}, {training_config.n_batches}, {training_config.batch_size}, {stack.nb_inputs})"
        )
        assert ybatches.shape == (
            training_config.n_replicates,
            training_config.n_batches,
            training_config.batch_size,
            stack.nb_outputs,
        ), (
            f"ybatches shape mismatch: {ybatches.shape} != ({training_config.n_replicates}, {training_config.n_batches}, {training_config.batch_size}, {stack.nb_outputs})"
        )

        return jnp.asarray(xbatches), jnp.asarray(ybatches)

    if xy_batches is not None:
        xbatches, ybatches = xy_batches
    else:
        xbatches, ybatches = get_new_batches()

    static, dynamic = params.filter_by_tag(["non_grad", "local"])

    # Use learning rate injection if learning_rate is requested in history
    if "learning_rate" in training_config.keep_in_history:
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

    loss_func_generator = training_config.loss_function.get_impl()
    loss_func = loss_func_generator(stack, training_config)
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

    def per_replicate_step_nonscan(start_params, start_opt_state, key, xbatches, ybatches, num_z):
        assert xbatches.shape[:-1] == (
            training_config.batches_per_step,
            training_config.batch_size,
        )
        assert ybatches.shape[:-1] == (
            training_config.batches_per_step,
            training_config.batch_size,
        )

        zbatches = jax.random.uniform(
            key, (training_config.batches_per_step, training_config.batch_size, num_z)
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

    def per_replicate_step(start_params, start_opt_state, key, xbatches, ybatches, num_z):
        assert xbatches.shape[:-1] == (
            training_config.batches_per_step,
            training_config.batch_size,
        )
        assert ybatches.shape[:-1] == (
            training_config.batches_per_step,
            training_config.batch_size,
        )
        zbatches = jax.random.uniform(
            key, (training_config.batches_per_step, training_config.batch_size, num_z)
        )

        batch_keys = jax.random.split(key, training_config.batches_per_step)
        sstep = scan_tqdm(training_config.batches_per_step)(scannable_step)
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
        return jax.vmap(Partial(per_replicate_step, num_z=num_z))(params, opt_state, keys, xs, ys)

    xb = get_looped_slice(xbatches, 0, training_config.batches_per_step, axis=1)
    yb = get_looped_slice(ybatches, 0, training_config.batches_per_step, axis=1)

    num_z = static["global/number_of_quantile_variables"]
    assert num_z.shape == (training_config.n_replicates,)
    assert jnp.all(num_z == num_z[0]), (
        "All replicates must have the same number of quantile variables"
    )
    num_z = int(num_z[0])

    logger.info("Compiling training step...")
    t0 = time.time()
    lowered = jax.jit(Partial(step, num_z=num_z)).lower(params, opt_state, key, xb, yb)
    compiled_step = lowered.compile()
    logger.info(f"Compiled training step in {time.time() - t0:.2f} seconds")

    # --- main training loop
    loggers = loggers or []
    
    # initialize async logger manager
    async with AsyncLoggerManager() as logger_manager:
        # submit start-of-training loggers (period=0)
        start_tasks = await logger_manager.submit_logger_batch(
            step=0,
            logger_callbacks=loggers,
            training_config=training_config,
            step_history={},
            xbatches=None,
            ybatches=None,
            stack=stack
        )
        # wait for start loggers to complete before training begins
        if start_tasks:
            await asyncio.gather(*start_tasks, return_exceptions=True)

        logger.info(f"Begin training for {total_steps} steps")

        step_history, loss_history = {}, []

        epoch = -1
        step_per_epoch = training_config.n_batches // training_config.batches_per_step

        for i, step_key in enumerate(jax.random.split(loop_key, total_steps), 1):
            # wait for previous step's loggers to complete before starting new step
            if i > 1:  # no previous loggers for first step
                await logger_manager.wait_for_previous_loggers()
            
            if i % (step_per_epoch) == 0:
                epoch += 1
                logger.info(f"Starting epoch {epoch}")
                b_key = jax.random.fold_in(step_key, epoch)
                xbatches, ybatches = get_new_batches(b_key)

            t0 = time.time()
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

            qvalues_dir = ParamPath("shared/quantization/values")
            qvalues = tuple(map(lambda t: t[1], params[qvalues_dir].iter_leaves()))

            # submit loggers for current step asynchronously
            logger_manager.pending_tasks = await logger_manager.submit_logger_batch(
                step=i,
                logger_callbacks=loggers,
                training_config=training_config,
                step_history=step_history,
                xbatches=xbatches,
                ybatches=ybatches,
                stack=stack
            )

        # wait for final step's loggers to complete
        await logger_manager.wait_for_previous_loggers()
        
        # handle end-of-training loggers
        await logger_manager.submit_end_loggers(
            step=total_steps,
            logger_callbacks=loggers,
            training_config=training_config,
            step_history=step_history,
            xbatches=xbatches,
            ybatches=ybatches,
            stack=stack
        )

    logger.info(f"End of training for {training_config.n_epochs} epochs")

    return params, loss_history, step_history


##────────────────────────────────────────────────────────────────────────────}}}
