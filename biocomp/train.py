### {{{                          --     imports     --
from . import datautils as du
from . import trainutils as tu
from biocomp.utils import (
    EncodedPartialFunction,
    PartialFunction,
    ArbitraryModel,
    PartialFunctionResult,
)
from . import nodes as nodes
from .parameters import ParameterTree, ParamPath

import time

from typing import List, Tuple, Callable, Optional, NamedTuple
from pydantic import Field
from biocomp.logging_config import get_logger

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                      --     loss functions     --


def check_XYZ(X, Y, Z, stack):
    nb_inputs = sum([n.nb_inputs for n in stack.networks])
    nb_outputs = sum([n.nb_outputs for n in stack.networks])
    assert X.ndim == Y.ndim == Z.ndim == 2, "X, Y, and Z must have 2 dimensions"
    assert X.shape[0] == Y.shape[0] == Z.shape[0], "X, Y, and Z must have the same number of rows"
    assert (
        X.shape[1] == nb_inputs
    ), "X must have as many columns as the total number of inputs in the stack"
    assert (
        Y.shape[1] == Z.shape[1] == nb_outputs
    ), "Y and Z must have as many columns as the total number of outputs in the stack"


def check_XYZ_new(X, Y, Z, stack):
    nb_inputs = sum([n.nb_inputs for n in stack.networks])
    nb_outputs = sum([n.nb_outputs for n in stack.networks])
    nb_nodes = len(stack.node_map)
    assert X.ndim == Y.ndim == Z.ndim == 2, "X, Y, and Z must have 2 dimensions"
    assert X.shape[0] == Y.shape[0] == Z.shape[0], "X, Y, and Z must have the same number of rows"
    assert (
        X.shape[1] == nb_inputs
    ), "X must have as many columns as the total number of inputs in the stack"
    assert (
        Y.shape[1] == nb_outputs
    ), "Y must have as many columns as the total number of outputs in the stack"


def as_schedule(value_or_callable):
    if callable(value_or_callable):
        return value_or_callable

    def f(step):
        return value_or_callable

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

        # jax.debug.print("yhat {}", yhat)
        # jax.debug.print("Y {}", Y)

        mse = ((yhat - Y) ** 2).mean()
        # jax.debug.print("mse {}", mse)
        negative_grads = jnp.mean(jnp.clip(-grads_wrt_inputs, 0, None))

        ngp = as_schedule(negative_grad_penalty)(step)

        loss = mse + ngp * negative_grads + klw * kl_loss
        # jax.debug.print("loss {}", loss)

        return loss, aux

    return loss_func


def sorting_loss(
    stack,
    training_config,
    negative_grad_penalty=1.0,
    kl_weight=0.1,
    sorting_mse_weight=0.1,
    percent_batch_used=1.0,
    qvalues_coeff=1000,
    use_same_key=False,
):
    import jax
    import jax.numpy as jnp

    batch_apply = jax.vmap(stack.apply, in_axes=(None, 0, 0, 0))

    def loss_func(dynamic, static, X, Y, Z, key, step):
        check_XYZ_new(X, Y, Z, stack)
        params = ParameterTree.merge(dynamic, static)

        if use_same_key:
            keys = jnp.array([key] * X.shape[0])
        else:
            keys = jax.random.split(key, X.shape[0])

        yhat, (grads_wrt_inputs, full_output) = batch_apply(params, X, Z, keys)
        assert yhat.shape == Y.shape, "yhat and Y must have the same shape"
        aux = {"yhat": yhat, "grads_wrt_inputs": grads_wrt_inputs, "full_output": full_output}

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
            (
                counts * ((qvalues * qvalues_coeff) ** 2 + jnp.exp(2 * logstds) / 2 - logstds - 0.5)
            ).sum()
            / counts.sum()
            * klw
        )

        # negative grads
        negative_grads = jnp.mean(jnp.clip(-grads_wrt_inputs, 0, None))
        ngp = as_schedule(negative_grad_penalty)(step)
        ng_loss = negative_grads * ngp

        pct = as_schedule(percent_batch_used)(step)
        select = jnp.linspace(0, 1, X.shape[0]) > pct
        Y = jnp.where(
            select[:, None],
            jnp.zeros(Y.shape),
            Y,
        )
        yhat = jnp.where(
            select[:, None],
            jnp.zeros(Y.shape),
            yhat,
        )

        # mse and sorted mse
        mse = ((yhat - Y) ** 2).mean()
        sorting_mse = ((yhat.sort(axis=0) - Y.sort(axis=0)) ** 2).mean() / pct
        smw = as_schedule(sorting_mse_weight)(step)
        sorting_loss = sorting_mse * smw + mse * (1 - smw)

        return sorting_loss + kl_loss + ng_loss, aux

    return loss_func


##────────────────────────────────────────────────────────────────────────────}}}

logger = get_logger(__name__)

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


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                       --     main training function     --


def start(
    dman: du.DataManager,
    training_config: TrainingConfig,
    compute_config,
    loggers: Optional[List[Tuple[int, Callable]]] = None,
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

    total_steps = int(
        training_config.n_epochs * training_config.n_batches / training_config.batches_per_step
    )

    stack, params = tu.init_stack(compute_config, dman, training_config.n_replicates, key)

    xbatches, ybatches = tu.generate_batches(
        dman,
        training_config.n_replicates,
        training_config.n_batches,
        training_config.batch_size,
        key,
    )

    static, dynamic = params.filter_by_tag(["non_grad", "local"])

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
    scannable_step = tu.make_training_step(
        loss_func,
        optimizer,
        fields_to_keep_in_history=training_config.keep_in_history,
        scannable=True,
    )
    non_scannable_step = tu.make_training_step(
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
    assert jnp.all(
        num_z == num_z[0]
    ), "All replicates must have the same number of quantile variables"
    num_z = int(num_z[0])

    logger.info("Compiling training step...")
    t0 = time.time()
    lowered = jax.jit(Partial(step, num_z=num_z)).lower(params, opt_state, key, xb, yb)
    compiled_step = lowered.compile()
    logger.info(f"Compiled training step in {time.time() - t0:.2f} seconds")

    # --- main training loop
    loggers = loggers or []

    for t, l in loggers:
        # call loggers at the beginning of the training if they have a period of 0
        if t == 0:
            l(step=0, training_config=training_config)

    logger.info(f"Begin training for {total_steps} steps")

    step_history, loss_history = {}, []

    epoch = -1
    step_per_epoch = training_config.n_batches // training_config.batches_per_step

    def reshuffle_batches(xbatches, ybatches, key):
        # shape is (n_replicates, n_batches, batch_size, n_inputs)
        # so make it (nrepl, n_batches * batch_size, n_inputs)
        # then shuffle, then reshape back
        reshaped_x = xbatches.reshape(xbatches.shape[0], -1, xbatches.shape[-1])
        reshaped_y = ybatches.reshape(ybatches.shape[0], -1, ybatches.shape[-1])
        perm = jax.random.permutation(key, reshaped_x.shape[1])
        return reshaped_x[:, perm, :].reshape(xbatches.shape), reshaped_y[:, perm, :].reshape(
            ybatches.shape
        )

    for i, step_key in enumerate(jax.random.split(key, total_steps), 1):
        if i % step_per_epoch == 0:
            epoch += 1
            logger.info(f"Starting epoch {epoch}")
            xbatches, ybatches = reshuffle_batches(xbatches, ybatches, step_key)

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

        if "loss" in step_history:
            loss_history.append(step_history["loss"])

        qvalues_dir = ParamPath("shared/quantization/values")
        qvalues = tuple(map(lambda t: t[1], params[qvalues_dir].iter_leaves()))

        for t, l in loggers:
            if t is not None:
                if t == 0 or (i % t == 0 and t > 0):
                    logger.debug(f"Calling logger {l} at step {i}")
                    l(
                        step=i,
                        training_config=training_config,
                        step_history=step_history,
                    )

    for t, l in loggers:
        if t is None or t == -1:
            logger.debug(f"Calling logger {l} at the end of training")
            l(
                step=total_steps,
                training_config=training_config,
                step_history=step_history,
            )

    logger.info(f"End of training for {training_config.n_epochs} epochs")

    return params, loss_history, step_history


##────────────────────────────────────────────────────────────────────────────}}}
