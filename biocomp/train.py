### {{{                          --     imports     --
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, NamedTuple, Tuple
import time

from . import datautils as du
from . import utils as ut
from .parameters import ParameterTree
from biocomp.utils import EncodedPartialFunction

if TYPE_CHECKING:
    import optax
    from .compute import ComputeStack
from pydantic import Field
from biocomp.logging_config import get_logger
from biocomp.optimutils import (
    make_training_step,
    per_replicate_step,
    as_schedule,
    OptimConfig,
    compile_step,
    get_checkify_enabled,
)
from biocomp.logger_dispatch import LoggerDispatch, NullDispatch
from biocomp.step_history import StepHistorySnapshot
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


def _aligned_quantization_kl_inputs(params):
    """Extract quantization values, logstdevs, counts with aligned shapes.

    Counts are per-embedding-name (1D per type), but values/logstdevs are
    per-embedding-dimension ((n, rate_dim) per type). Repeat each count
    across its embedding's dimensions so broadcasting works.

    Returns (qvalues, logstds, aligned_counts, original_counts_sum) where
    original_counts_sum is the sum of counts *before* repeating across
    dimensions, for correct KL normalization.
    """
    import jax.numpy as jnp
    from jax.tree_util import tree_leaves
    from .jaxutils import flat_concat

    value_leaves = tree_leaves(params["shared/quantization/values"])
    logstd_leaves = tree_leaves(params["shared/quantization/logstdevs"])
    count_leaves = tree_leaves(params["shared/quantization/counts"])

    original_counts_sum = jnp.array(0.0)
    aligned_counts = []
    for count_leaf, value_leaf in zip(count_leaves, value_leaves, strict=True):
        c = jnp.asarray(count_leaf).ravel()
        original_counts_sum = original_counts_sum + c.sum()
        v = jnp.asarray(value_leaf)
        if v.ndim == 2 and v.shape[1] > 1:
            c = jnp.repeat(c, v.shape[1])
        aligned_counts.append(c)

    qvalues = flat_concat(*value_leaves)
    logstds = flat_concat(*logstd_leaves)
    counts = jnp.concatenate(aligned_counts)
    return qvalues, logstds, counts, original_counts_sum


def _quantization_kl_loss(params, kl_weight, step):
    """KL(q || N(0,1)) for variational codebook embeddings, weighted by usage counts.

    Normalization uses the original (pre-repeat) counts sum so the formula is a
    weighted average over *embeddings* of the full multi-dim KL.  For rate_dim=1
    this is identical to the old behavior; for rate_dim>1 each dimension now gets
    full KL pressure instead of diluted-by-rate_dim pressure.
    """
    klw = as_schedule(kl_weight)(step)
    qvalues, logstds, counts, original_counts_sum = _aligned_quantization_kl_inputs(params)
    std = stable_sigma(logstds, min_std=1e-3)
    kl = 0.5 * (counts * (qvalues**2 + std**2 - 1 - 2 * logstds)).sum() / original_counts_sum * klw
    return kl, klw, qvalues, logstds, counts, std


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

    logger.info("Compiling training step...")
    t0 = time.time()
    jitable_base = Partial(step, num_z=num_z)
    sample_args = (sample_params, sample_opt_state, key, sample_xb, sample_yb)

    if not get_checkify_enabled():
        from biocomp.compilation_cache import (
            CompilationSignature,
            cached_compile,
            loss_function_source_hash,
            extract_arg_shapes,
            training_config_compilation_dump,
        )

        stack_sig = CompilationSignature.for_stack(stack)
        training_sig = CompilationSignature.for_training_step(
            stack_sig=stack_sig,
            training_config_dump=training_config_compilation_dump(training_config),
            loss_source_hash=loss_function_source_hash(training_config),
            arg_shapes=extract_arg_shapes(*sample_args),
        )

        def _do_compile():
            lowered = jax.jit(jitable_base).lower(*sample_args)
            return lowered.compile()

        compiled = cached_compile(_do_compile, signature=training_sig)
        logger.info(f"Training step ready in {time.time() - t0:.2f}s (cached or compiled)")
    else:
        compiled = compile_step(jitable_base, sample_args)

    return CompiledTrainingStep(
        compiled_step=compiled, stack=stack, optimizer=optimizer, num_z=num_z
    )


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                      --     loss functions     --


def expand_weights_to_outputs(weights: list[float], networks: list) -> list[float]:
    """Expand per-network weights to per-output weights."""
    return [w for w, n in zip(weights, networks, strict=False) for _ in range(n.nb_outputs)]


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


class InversePairSpec(NamedTuple):
    pair_type: str  # kept for logging only
    network_id: int
    fwd_layer_id: int
    fwd_node_pos: int
    inv_layer_id: int
    inv_node_pos: int
    output_slot: int
    n_fwd_inputs: int
    fwd_input_shape: tuple[int, ...]
    rate_path: Optional[str] = None
    is_multi_input: bool = False
    is_slotted_output: bool = False


def _discover_inverse_pair_specs(stack) -> tuple[list[InversePairSpec], dict[str, int]]:
    from .graphengine import is_inverse_node_type

    counts: dict[str, int] = {}
    pair_specs: list[InversePairSpec] = []

    layers = getattr(stack, "layers", None)
    each_node = getattr(stack, "each_node", None)
    if layers is None or each_node is None:
        return pair_specs, counts

    for inv_stacknode in each_node():
        inv_node = inv_stacknode.get(stack)
        if inv_node.is_inverse_of is None:
            continue
        if not is_inverse_node_type(inv_node.node_type):
            continue

        fwd_stacknode = inv_stacknode.get_forward_stacknode(stack)
        if fwd_stacknode is None:
            continue
        if fwd_stacknode.layer_number is None or inv_stacknode.layer_number is None:
            continue

        fwd_layer = layers[fwd_stacknode.layer_number]
        if fwd_layer.f_input_shapes is None or len(fwd_layer.f_input_shapes) == 0:
            continue

        fwd_node = fwd_stacknode.get(stack)
        pair_type = fwd_node.node_type

        # rate_path: derive from edge content_embedding_names (SSOT)
        rate_path = None
        fwd_edges = fwd_stacknode.get_incoming_edges(stack)
        if fwd_edges:
            emb_keys = list(fwd_edges[0].content_embedding_names.keys())
            if emb_keys:
                rate_path = f"{stack.get_layer_namespace(fwd_stacknode.layer_number)}/{emb_keys[0]}"

        n_fwd_inputs = len(fwd_layer.f_input_shapes)
        is_slotted = inv_node.is_inverse_of.output_len > 1

        pair_specs.append(
            InversePairSpec(
                pair_type=pair_type,
                network_id=inv_stacknode.network_id,
                fwd_layer_id=fwd_stacknode.layer_number,
                fwd_node_pos=int(fwd_stacknode.node_position_in_layer),
                inv_layer_id=inv_stacknode.layer_number,
                inv_node_pos=int(inv_stacknode.node_position_in_layer),
                output_slot=int(inv_node.is_inverse_of.output_slot),
                n_fwd_inputs=n_fwd_inputs,
                fwd_input_shape=tuple(int(d) for d in fwd_layer.f_input_shapes[0]),
                rate_path=rate_path,
                is_multi_input=n_fwd_inputs > 1,
                is_slotted_output=is_slotted,
            )
        )
        counts[pair_type] = counts.get(pair_type, 0) + 1

    return pair_specs, counts


def _apply_pair_node(
    apply_f,
    *inputs,
    random_vars,
    params,
    node_id,
    key,
    network_id,
    rate_override=None,
):
    kwargs = {
        "random_vars": random_vars,
        "params": params,
        "node_id": node_id,
        "key": key,
        "network_id": network_id,
    }
    if rate_override is not None:
        kwargs["rate_override"] = rate_override
    return apply_f(*inputs, **kwargs)


def _single_inverse_cycle_sample(
    params,
    stack,
    spec: InversePairSpec,
    x_scalar,
    random_vars,
    sample_key,
    rate_override,
):
    import jax.numpy as jnp

    assert stack.layers is not None, "Stack layers are required for inverse consistency loss"

    fwd_layer = stack.layers[spec.fwd_layer_id]
    inv_layer = stack.layers[spec.inv_layer_id]
    assert fwd_layer.f_apply is not None and inv_layer.f_apply is not None

    dtype = random_vars.dtype
    x0 = jnp.asarray(x_scalar, dtype=dtype).reshape(())
    x_target = jnp.array([x0], dtype=dtype)
    fwd_input = jnp.broadcast_to(x0, spec.fwd_input_shape)

    fwd_node_id = jnp.asarray(spec.fwd_node_pos, dtype=jnp.int32)
    inv_node_id = jnp.asarray(spec.inv_node_pos, dtype=jnp.int32)
    network_id = jnp.asarray(spec.network_id, dtype=jnp.int32)

    # --- Input construction (multi-input vs single-input) ---
    if spec.is_multi_input:
        zeros = jnp.zeros((spec.n_fwd_inputs, *spec.fwd_input_shape), dtype=dtype)
        stacked_inputs = zeros.at[spec.output_slot].set(fwd_input)
        fwd_inputs = tuple(stacked_inputs[i] for i in range(spec.n_fwd_inputs))
    else:
        fwd_inputs = (fwd_input,)

    # --- Forward pass ---
    fwd_out, _ = _apply_pair_node(
        fwd_layer.f_apply,
        *fwd_inputs,
        random_vars=random_vars,
        params=params,
        node_id=fwd_node_id,
        key=sample_key,
        network_id=network_id,
        rate_override=rate_override,
    )

    # --- Output extraction (slotted tuple vs scalar) ---
    if spec.is_slotted_output:
        inv_input = jnp.ravel(fwd_out[spec.output_slot])[:1]
    else:
        assert inv_layer.f_input_shapes is not None and len(inv_layer.f_input_shapes) == 1
        inv_input = jnp.asarray(fwd_out).reshape(inv_layer.f_input_shapes[0])

    inv_out, _ = _apply_pair_node(
        inv_layer.f_apply,
        inv_input,
        random_vars=random_vars,
        params=params,
        node_id=inv_node_id,
        key=sample_key,
        network_id=network_id,
        rate_override=rate_override,
    )
    recon = jnp.ravel(inv_out)[:1]
    return jnp.mean((recon - x_target) ** 2)


def _inverse_pair_cycle_loss(
    params,
    stack,
    spec: InversePairSpec,
    pair_key,
    *,
    num_random_vars: int,
    batch_size: int,
    dtype,
    sample_embeddings: bool,
    embedding_low: float,
    embedding_high: float,
):
    import jax
    import jax.numpy as jnp

    kx, krv, ksample, kemb = jax.random.split(pair_key, 4)
    x_samples = jax.random.uniform(kx, (batch_size,), minval=0.0, maxval=1.0, dtype=dtype)
    rv_samples = jax.random.uniform(
        krv, (batch_size, num_random_vars), minval=0.0, maxval=1.0, dtype=dtype
    )
    sample_keys = jax.random.split(ksample, batch_size)

    rate_samples = None
    if sample_embeddings and spec.rate_path is not None:
        base_rate = jnp.asarray(params[spec.rate_path][spec.fwd_node_pos])
        rate_samples = jax.random.uniform(
            kemb,
            (batch_size, *base_rate.shape),
            minval=embedding_low,
            maxval=embedding_high,
            dtype=base_rate.dtype,
        )

    if rate_samples is None:
        losses = jax.vmap(
            lambda x, rv, sk: _single_inverse_cycle_sample(
                params=params,
                stack=stack,
                spec=spec,
                x_scalar=x,
                random_vars=rv,
                sample_key=sk,
                rate_override=None,
            )
        )(x_samples, rv_samples, sample_keys)
    else:
        losses = jax.vmap(
            lambda x, rv, sk, rs: _single_inverse_cycle_sample(
                params=params,
                stack=stack,
                spec=spec,
                x_scalar=x,
                random_vars=rv,
                sample_key=sk,
                rate_override=rs,
            )
        )(x_samples, rv_samples, sample_keys, rate_samples)
    return jnp.mean(losses)


def _compute_inverse_consistency_loss(
    params,
    stack,
    pair_specs: list[InversePairSpec],
    pair_counts: dict[str, int],
    key,
    *,
    num_random_vars: int,
    batch_size: int,
    dtype,
    sample_embeddings: bool,
    embedding_low: float,
    embedding_high: float,
):
    import jax
    import jax.numpy as jnp

    zero = jnp.asarray(0.0, dtype=dtype)
    if not pair_specs:
        return zero, {
            "inverse_consistency_n_pairs": 0,
            "inverse_consistency_batch_size": batch_size,
            "inverse_consistency_n_source_pairs": 0,
            "inverse_consistency_n_transcription_pairs": 0,
            "inverse_consistency_n_translation_pairs": 0,
            "inverse_consistency_n_output_pairs": 0,
        }

    pair_keys = jax.random.split(key, len(pair_specs))
    pair_losses = [
        _inverse_pair_cycle_loss(
            params=params,
            stack=stack,
            spec=spec,
            pair_key=pk,
            num_random_vars=num_random_vars,
            batch_size=batch_size,
            dtype=dtype,
            sample_embeddings=sample_embeddings,
            embedding_low=embedding_low,
            embedding_high=embedding_high,
        )
        for spec, pk in zip(pair_specs, pair_keys, strict=False)
    ]
    pair_losses_arr = jnp.stack(pair_losses)
    return jnp.mean(pair_losses_arr), {
        "inverse_consistency_n_pairs": len(pair_specs),
        "inverse_consistency_batch_size": batch_size,
        "inverse_consistency_n_source_pairs": pair_counts.get("source", 0),
        "inverse_consistency_n_transcription_pairs": pair_counts.get("transcription", 0),
        "inverse_consistency_n_translation_pairs": pair_counts.get("translation", 0),
        "inverse_consistency_n_output_pairs": pair_counts.get("output", 0),
    }


def _prepare_inverse_consistency_context(
    *,
    stack,
    batch_size,
    sample_embeddings,
    embedding_low,
    embedding_high,
):
    pair_specs, pair_counts = _discover_inverse_pair_specs(stack)
    inverse_batch_size = int(batch_size)
    assert inverse_batch_size >= 1, "inverse_consistency_batch_size must be >= 1"
    inverse_embed_low = float(embedding_low)
    inverse_embed_high = float(embedding_high)
    assert inverse_embed_low < inverse_embed_high, (
        "inverse_consistency_embedding_low must be < inverse_consistency_embedding_high"
    )
    inverse_sample_embeddings = bool(sample_embeddings)
    return (
        pair_specs,
        pair_counts,
        inverse_batch_size,
        inverse_sample_embeddings,
        inverse_embed_low,
        inverse_embed_high,
    )


def sorting_loss(
    stack,
    training_config,
    negative_grad_penalty=1.0,
    kl_weight=0.2,
    sorting_mse_weight=0.1,
    z_sorting_mse_weight=0.0,
    pinball_weight=0.0,
    pinball_use_z_order=False,
    pinball_use_z_tau=False,
    pinball_rank_clip_low=0.0,
    pinball_rank_clip_high=1.0,
    cdf_calibration_weight=0.0,
    cdf_calibration_temperature=0.15,
    percent_batch_used=1.0,
    out_vs_in_mse_weight=0.5,
    out_vs_in_sortmse_weight=1,
    use_same_key=False,
    per_output_weights=None,
    inverse_consistency_weight=0.0,
    inverse_consistency_batch_size=64,
    inverse_consistency_sample_embeddings=True,
    inverse_consistency_embedding_low=-1.0,
    inverse_consistency_embedding_high=1.0,
    **_unused_kwargs,
):
    """MSE loss with sorted outputs to learn distribution (quantile-like)."""
    import jax
    import jax.numpy as jnp
    from .jaxutils import robust_sort

    batch_apply = jax.vmap(stack.apply, in_axes=(None, 0, 0, 0))
    (
        inverse_pair_specs,
        inverse_pair_counts,
        inverse_batch_size,
        inverse_sample_embeddings,
        inverse_embed_low,
        inverse_embed_high,
    ) = _prepare_inverse_consistency_context(
        stack=stack,
        batch_size=inverse_consistency_batch_size,
        sample_embeddings=inverse_consistency_sample_embeddings,
        embedding_low=inverse_consistency_embedding_low,
        embedding_high=inverse_consistency_embedding_high,
    )

    def loss_func(dynamic, static, X, Y, Z, key, step):
        check_XYZ(X, Y, Z, stack)
        assert 0.0 <= pinball_rank_clip_low <= pinball_rank_clip_high <= 1.0, (
            "pinball_rank_clip_low/high must satisfy 0 <= low <= high <= 1"
        )
        assert cdf_calibration_temperature > 0.0, "cdf_calibration_temperature must be > 0"
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

        kl_loss, klw, qvalues, logstds, counts, std = _quantization_kl_loss(params, kl_weight, step)

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
        z_scalar = jnp.asarray(Z[:, 0] if Z.ndim > 1 else Z, dtype=Y.dtype)
        z_sort_key = jnp.where(selected[:, 0], z_scalar, jnp.finfo(z_scalar.dtype).max)
        z_order = jnp.argsort(z_sort_key)
        z_sorted_yhat = yhat[z_order]
        sort_sqdiff = (sorted_yhat - sorted_y) ** 2 * output_weights[None, :]
        out_v_in_sortmse = as_schedule(out_vs_in_sortmse_weight)(step)
        sorting_mse_dependent = (sort_sqdiff * dep_mask[None, :]).sum() / (
            eff_batch_size * dep_mask_sum
        )
        sorting_mse_independent = (sort_sqdiff * indep_mask[None, :]).sum() / (
            eff_batch_size * indep_mask_sum
        )
        sorting_mse = lerp(sorting_mse_independent, sorting_mse_dependent, out_v_in_sortmse)

        rank_valid = (sorted_y < (MAXFLOAT * 0.5)).astype(Y.dtype)
        valid_mask = rank_valid > 0
        z_sort_err = jnp.where(valid_mask, z_sorted_yhat - sorted_y, 0.0)
        z_sort_sqdiff = (z_sort_err**2) * output_weights[None, :]
        z_sorting_mse_dependent = (z_sort_sqdiff * dep_mask[None, :]).sum() / (
            eff_batch_size * dep_mask_sum
        )
        z_sorting_mse_independent = (z_sort_sqdiff * indep_mask[None, :]).sum() / (
            eff_batch_size * indep_mask_sum
        )
        z_sorting_mse = lerp(z_sorting_mse_independent, z_sorting_mse_dependent, out_v_in_sortmse)

        # Pinball objective variants:
        # - rank-based on sorted values (legacy/default)
        # - z-conditioned on raw predictions (quantile-regression style)
        if pinball_use_z_tau:
            tau = jnp.clip(z_scalar, pinball_rank_clip_low, pinball_rank_clip_high)[:, None]
            pinball_err = Y - yhat
            pinball = jnp.maximum(tau * pinball_err, (tau - 1.0) * pinball_err)
            pinball_weighted = pinball * selected * output_weights[None, :]
            pinball_dependent = (pinball_weighted * dep_mask[None, :]).sum() / (
                eff_batch_size * dep_mask_sum
            )
            pinball_independent = (pinball_weighted * indep_mask[None, :]).sum() / (
                eff_batch_size * indep_mask_sum
            )
        else:
            rank_pos = (jnp.arange(X.shape[0], dtype=Y.dtype) + 0.5) / X.shape[0]
            tau = jnp.clip(rank_pos, pinball_rank_clip_low, pinball_rank_clip_high)[:, None]
            pinball_base = z_sorted_yhat if pinball_use_z_order else sorted_yhat
            safe_sorted_y = jnp.where(valid_mask, sorted_y, pinball_base)
            pinball_err = safe_sorted_y - pinball_base
            pinball = jnp.maximum(tau * pinball_err, (tau - 1.0) * pinball_err)
            pinball_weighted = pinball * rank_valid * output_weights[None, :]
            rank_count = jnp.maximum(rank_valid.sum(), 1.0)
            pinball_dependent = (pinball_weighted * dep_mask[None, :]).sum() / (
                rank_count * dep_mask_sum
            )
            pinball_independent = (pinball_weighted * indep_mask[None, :]).sum() / (
                rank_count * indep_mask_sum
            )
        pinball_loss = lerp(pinball_independent, pinball_dependent, out_v_in_sortmse)

        # CDF calibration objective:
        # Encourage calibrated quantiles: P(Y <= q_tau(X)) ~= tau, where tau comes from z.
        tau_z = jnp.clip(z_scalar, pinball_rank_clip_low, pinball_rank_clip_high)[:, None]
        selected_count_per_out = jnp.maximum(selected.sum(axis=0), 1.0)
        selected_y = selected * Y
        selected_mean = selected_y.sum(axis=0) / selected_count_per_out
        selected_var = (selected * (Y - selected_mean[None, :]) ** 2).sum(axis=0) / (
            selected_count_per_out
        )
        # Normalize by per-output spread so one temperature works across wide output scales.
        y_scale = jnp.sqrt(jnp.maximum(selected_var, 1e-12))
        calib_logits = (yhat - Y) / (cdf_calibration_temperature * y_scale[None, :] + 1e-6)
        cdf_prob = jax.nn.sigmoid(calib_logits)
        cdf_calibration_sqerr = (cdf_prob - tau_z) ** 2 * selected * output_weights[None, :]
        cdf_calibration_dependent = (cdf_calibration_sqerr * dep_mask[None, :]).sum() / (
            eff_batch_size * dep_mask_sum
        )
        cdf_calibration_independent = (cdf_calibration_sqerr * indep_mask[None, :]).sum() / (
            eff_batch_size * indep_mask_sum
        )
        cdf_calibration_loss = lerp(
            cdf_calibration_independent, cdf_calibration_dependent, out_v_in_sortmse
        )

        smw = as_schedule(sorting_mse_weight)(step)
        main_loss = lerp(mse, sorting_mse, smw)
        zsmw = as_schedule(z_sorting_mse_weight)(step)
        main_loss = lerp(main_loss, z_sorting_mse, zsmw)
        pbw = as_schedule(pinball_weight)(step)
        main_loss = lerp(main_loss, pinball_loss, pbw)
        cdfw = as_schedule(cdf_calibration_weight)(step)
        main_loss = lerp(main_loss, cdf_calibration_loss, cdfw)
        icw = as_schedule(inverse_consistency_weight)(step)
        inverse_consistency_loss, inverse_dbg = _compute_inverse_consistency_loss(
            params=params,
            stack=stack,
            pair_specs=inverse_pair_specs,
            pair_counts=inverse_pair_counts,
            key=jax.random.fold_in(key, 33),
            num_random_vars=Z.shape[1],
            batch_size=inverse_batch_size,
            dtype=Y.dtype,
            sample_embeddings=inverse_sample_embeddings,
            embedding_low=inverse_embed_low,
            embedding_high=inverse_embed_high,
        )
        main_loss = main_loss + icw * inverse_consistency_loss

        aux["sublosses"] = {
            "mse": mse,
            "sorting_mse": sorting_mse,
            "z_sorting_mse": z_sorting_mse,
            "pinball_loss": pinball_loss,
            "cdf_calibration_loss": cdf_calibration_loss,
            "inverse_consistency_loss": inverse_consistency_loss,
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
            "z_scalar": z_scalar,
            "z_order": z_order,
            "z_sorted_yhat": z_sorted_yhat,
            "pct": pct,
            "kl_weight": klw,
            "negative_grad_penalty": ngp,
            "sqdiff": sqdiff,
            "sort_sqdiff": sort_sqdiff,
            "z_sort_sqdiff": z_sort_sqdiff,
            "mse_dependent": mse_dependent,
            "mse_independent": mse_independent,
            "sorting_mse_dependent": sorting_mse_dependent,
            "sorting_mse_independent": sorting_mse_independent,
            "z_sorting_mse_dependent": z_sorting_mse_dependent,
            "z_sorting_mse_independent": z_sorting_mse_independent,
            "z_sorting_mse_weight": zsmw,
            "pinball_dependent": pinball_dependent,
            "pinball_independent": pinball_independent,
            "pinball_weight": pbw,
            "pinball_use_z_tau": pinball_use_z_tau,
            "cdf_calibration_dependent": cdf_calibration_dependent,
            "cdf_calibration_independent": cdf_calibration_independent,
            "cdf_calibration_weight": cdfw,
            "cdf_calibration_temperature": cdf_calibration_temperature,
            "y_scale": y_scale,
            "out_v_in_mse": out_v_in_mse,
            "out_v_in_sortmse": out_v_in_sortmse,
            "qvalues": qvalues,
            "logstds": logstds,
            "counts": counts,
            "step": step,
            "output_weights": output_weights,
            "inverse_consistency_weight": icw,
            **inverse_dbg,
        }
        aux["apply_aux"] = apply_aux

        return main_loss + kl_loss + ng_loss, aux

    return loss_func


def energy_sampling_loss(
    stack,
    training_config,
    negative_grad_penalty=1.0,
    kl_weight=0.2,
    percent_batch_used=1.0,
    energy_n_samples=8,
    energy_outputs_independent=True,
    energy_include_input_z=True,
    energy_z_distribution="uniform",
    energy_z_normal_mean=0.5,
    energy_z_normal_std=0.2,
    energy_z_normal_clip=True,
    energy_weight=1.0,
    energy_pairwise_weight=0.5,
    out_vs_in_energy_weight=1.0,
    coverage_calibration_weight=0.0,
    coverage_interval_low=0.1,
    coverage_interval_high=0.9,
    coverage_temperature=0.15,
    tail_pinball_weight=0.0,
    tail_tau_low=0.03,
    tail_tau_high=0.97,
    use_same_key=False,
    per_output_weights=None,
    inverse_consistency_weight=0.0,
    inverse_consistency_batch_size=64,
    inverse_consistency_sample_embeddings=True,
    inverse_consistency_embedding_low=-1.0,
    inverse_consistency_embedding_high=1.0,
    **_unused_kwargs,
):
    """Sample-based distributional loss (CRPS / energy score).

    This loss supervises only final outputs while treating internal latent variables
    as nuisance variables. For each training (X, Y), we draw K latent samples and
    optimize a proper scoring rule:
      score = E||Yhat - Y|| - 0.5 * E||Yhat - Yhat'||

    `energy_outputs_independent=True` computes univariate energy per output
    (equivalent to averaged CRPS). `False` uses the joint vector norm.
    """
    import math
    import jax
    import jax.numpy as jnp

    batch_apply = jax.vmap(stack.apply, in_axes=(None, 0, 0, 0))
    (
        inverse_pair_specs,
        inverse_pair_counts,
        inverse_batch_size,
        inverse_sample_embeddings,
        inverse_embed_low,
        inverse_embed_high,
    ) = _prepare_inverse_consistency_context(
        stack=stack,
        batch_size=inverse_consistency_batch_size,
        sample_embeddings=inverse_consistency_sample_embeddings,
        embedding_low=inverse_consistency_embedding_low,
        embedding_high=inverse_consistency_embedding_high,
    )
    n_energy_samples = int(energy_n_samples)
    assert n_energy_samples >= 1, "energy_n_samples must be >= 1"
    z_dist = str(energy_z_distribution).lower()
    assert z_dist in {"uniform", "normal"}, (
        f"energy_z_distribution must be 'uniform' or 'normal', got {energy_z_distribution!r}"
    )
    z_normal_mean = float(energy_z_normal_mean)
    z_normal_std = float(energy_z_normal_std)
    assert z_normal_std > 0, "energy_z_normal_std must be > 0"
    z_normal_clip = bool(energy_z_normal_clip)
    qlow = float(coverage_interval_low)
    qhigh = float(coverage_interval_high)
    assert 0.0 <= qlow < qhigh <= 1.0, "coverage interval must satisfy 0 <= low < high <= 1"
    target_coverage = qhigh - qlow
    temp_cov = float(coverage_temperature)
    assert temp_cov > 0, "coverage_temperature must be > 0"
    tail_low = float(tail_tau_low)
    tail_high = float(tail_tau_high)
    assert 0.0 < tail_low < tail_high < 1.0, "tail taus must satisfy 0 < low < high < 1"

    def sample_z_latents(key, shape, dtype):
        if z_dist == "uniform":
            return jax.random.uniform(key, shape, dtype=dtype)
        z = z_normal_mean + z_normal_std * jax.random.normal(key, shape, dtype=dtype)
        if z_normal_clip:
            z = jnp.clip(z, 0.0, 1.0)
        return z

    def sample_quantile(yhat_samples_block, q):
        """Linear-interpolated sample quantile along sample axis."""
        sorted_samples = jnp.sort(yhat_samples_block, axis=0)
        pos = float(q) * float(n_energy_samples - 1)
        low_idx = int(math.floor(pos))
        high_idx = int(math.ceil(pos))
        alpha = pos - low_idx
        low_val = sorted_samples[low_idx]
        high_val = sorted_samples[high_idx]
        return (1.0 - alpha) * low_val + alpha * high_val

    def block_energy_score(
        yhat_samples_block,
        y_block,
        block_weights,
        selected_1d,
        eff_batch_size,
        independent,
        pairwise_weight,
    ):
        k = yhat_samples_block.shape[0]
        active_weight = block_weights.sum()
        wsum = jnp.maximum(active_weight, 1e-12)

        if independent:
            abs_err = (
                jnp.abs(yhat_samples_block - y_block[None, :, :]) * block_weights[None, None, :]
            )
            term_a = (abs_err * selected_1d[None, :, None]).sum() / (k * eff_batch_size * wsum)

            pair_abs = (
                jnp.abs(yhat_samples_block[:, None, :, :] - yhat_samples_block[None, :, :, :])
                * block_weights[None, None, None, :]
            )
            term_b = (pair_abs * selected_1d[None, None, :, None]).sum() / (
                k * k * eff_batch_size * wsum
            )
        else:
            scale = jnp.sqrt(jnp.maximum(block_weights, 0.0))
            diff = (yhat_samples_block - y_block[None, :, :]) * scale[None, None, :]
            dist = jnp.sqrt((diff**2).sum(axis=-1) + 1e-12)
            term_a = (dist * selected_1d[None, :]).sum() / (k * eff_batch_size)

            pair_diff = (
                yhat_samples_block[:, None, :, :] - yhat_samples_block[None, :, :, :]
            ) * scale[None, None, None, :]
            pair_dist = jnp.sqrt((pair_diff**2).sum(axis=-1) + 1e-12)
            term_b = (pair_dist * selected_1d[None, None, :]).sum() / (k * k * eff_batch_size)

        gate = (active_weight > 0).astype(y_block.dtype)
        energy = gate * (term_a - pairwise_weight * term_b)
        return energy, {"energy_term_a": term_a, "energy_term_b": term_b}

    def block_coverage_calibration(
        yhat_samples_block, y_block, block_weights, selected_1d, eff_batch_size
    ):
        active_weight = block_weights.sum()
        wsum = jnp.maximum(active_weight, 1e-12)
        gate = (active_weight > 0).astype(y_block.dtype)

        lower = sample_quantile(yhat_samples_block, qlow)
        upper = sample_quantile(yhat_samples_block, qhigh)
        # Smooth interval-membership surrogate for differentiability.
        in_low = jax.nn.sigmoid((y_block - lower) / temp_cov)
        in_high = jax.nn.sigmoid((upper - y_block) / temp_cov)
        inside = in_low * in_high

        weighted_inside = inside * selected_1d[:, None] * block_weights[None, :]
        coverage = weighted_inside.sum() / (eff_batch_size * wsum)
        cov_loss = gate * (coverage - target_coverage) ** 2
        return cov_loss, {"coverage": coverage, "target_coverage": target_coverage}

    def block_tail_pinball(yhat_samples_block, y_block, block_weights, selected_1d, eff_batch_size):
        active_weight = block_weights.sum()
        wsum = jnp.maximum(active_weight, 1e-12)
        gate = (active_weight > 0).astype(y_block.dtype)

        qlo = sample_quantile(yhat_samples_block, tail_low)
        qhi = sample_quantile(yhat_samples_block, tail_high)
        lo_err = y_block - qlo
        hi_err = y_block - qhi
        lo_pb = jnp.maximum(tail_low * lo_err, (tail_low - 1.0) * lo_err)
        hi_pb = jnp.maximum(tail_high * hi_err, (tail_high - 1.0) * hi_err)
        tail_pb = 0.5 * (lo_pb + hi_pb)

        weighted_pb = tail_pb * selected_1d[:, None] * block_weights[None, :]
        loss = gate * (weighted_pb.sum() / (eff_batch_size * wsum))
        return loss

    def loss_func(dynamic, static, X, Y, Z, key, step):
        check_XYZ(X, Y, Z, stack)
        params = ParameterTree.merge(dynamic, static)

        z_shape = Z.shape[1:]
        z_key, apply_key = jax.random.split(key)
        z_samples = sample_z_latents(
            z_key,
            (n_energy_samples, X.shape[0], *z_shape),
            Y.dtype,
        )
        if energy_include_input_z:
            z_samples = z_samples.at[0].set(Z)

        if use_same_key:
            one_batch_key = jnp.broadcast_to(key, (X.shape[0], *key.shape))
            sample_keys = jnp.broadcast_to(
                one_batch_key, (n_energy_samples, X.shape[0], *key.shape)
            )
        else:
            sample_keys = jax.random.split(apply_key, n_energy_samples * X.shape[0]).reshape(
                n_energy_samples, X.shape[0], -1
            )

        def predict_one(zb, kb):
            return batch_apply(params, X, zb, kb)

        yhat_samples, (apply_aux_samples, full_output_samples) = jax.vmap(
            predict_one, in_axes=(0, 0)
        )(z_samples, sample_keys)
        yhat = yhat_samples[0]
        apply_aux = jax.tree_util.tree_map(lambda v: v[0], apply_aux_samples)
        full_output = full_output_samples[0]

        grads_wrt_inputs = apply_aux["grads_wrt_inputs"]
        aux = {"yhat": yhat, "grads_wrt_inputs": grads_wrt_inputs, "full_output": full_output}

        kl_loss, klw, qvalues, logstds, counts, std = _quantization_kl_loss(params, kl_weight, step)

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
            output_weights = jnp.ones(Y.shape[1], dtype=Y.dtype)
        assert output_weights.shape == (Y.shape[1],)

        pct = as_schedule(percent_batch_used)(step)
        selected = (jnp.linspace(0, 1, X.shape[0]) <= pct)[:, None]
        selected_f = selected.astype(Y.dtype)
        selected_1d = selected_f[:, 0]
        eff_batch_size = jnp.maximum(selected_1d.sum(), 1.0)

        sqdiff = (yhat - Y) ** 2 * selected_f * output_weights[None, :]
        dep_mask_sum = jnp.maximum((dep_mask * output_weights).sum(), 1e-12)
        indep_mask_sum = jnp.maximum((indep_mask * output_weights).sum(), 1e-12)
        mse_dependent = (sqdiff * dep_mask[None, :]).sum() / (eff_batch_size * dep_mask_sum)
        mse_independent = (sqdiff * indep_mask[None, :]).sum() / (eff_batch_size * indep_mask_sum)

        dep_weights = output_weights * dep_mask.astype(Y.dtype)
        dep_energy, dep_dbg = block_energy_score(
            yhat_samples,
            Y,
            dep_weights,
            selected_1d,
            eff_batch_size,
            independent=bool(energy_outputs_independent),
            pairwise_weight=as_schedule(energy_pairwise_weight)(step),
        )

        indep_weights = output_weights * indep_mask.astype(Y.dtype)
        indep_energy, indep_dbg = block_energy_score(
            yhat_samples,
            Y,
            indep_weights,
            selected_1d,
            eff_batch_size,
            independent=bool(energy_outputs_independent),
            pairwise_weight=as_schedule(energy_pairwise_weight)(step),
        )

        ovie = as_schedule(out_vs_in_energy_weight)(step)
        energy_loss = lerp(indep_energy, dep_energy, ovie)
        mse = lerp(mse_independent, mse_dependent, ovie)
        ew = as_schedule(energy_weight)(step)
        cdep_loss, cdep_dbg = block_coverage_calibration(
            yhat_samples, Y, dep_weights, selected_1d, eff_batch_size
        )
        cindep_loss, cindep_dbg = block_coverage_calibration(
            yhat_samples, Y, indep_weights, selected_1d, eff_batch_size
        )
        coverage_loss = lerp(cindep_loss, cdep_loss, ovie)
        ccw = as_schedule(coverage_calibration_weight)(step)
        tdep_loss = block_tail_pinball(yhat_samples, Y, dep_weights, selected_1d, eff_batch_size)
        tindep_loss = block_tail_pinball(
            yhat_samples, Y, indep_weights, selected_1d, eff_batch_size
        )
        tail_loss = lerp(tindep_loss, tdep_loss, ovie)
        tpw = as_schedule(tail_pinball_weight)(step)
        main_loss = lerp(mse, energy_loss, ew) + ccw * coverage_loss + tpw * tail_loss
        icw = as_schedule(inverse_consistency_weight)(step)
        inverse_consistency_loss, inverse_dbg = _compute_inverse_consistency_loss(
            params=params,
            stack=stack,
            pair_specs=inverse_pair_specs,
            pair_counts=inverse_pair_counts,
            key=jax.random.fold_in(key, 33),
            num_random_vars=Z.shape[1],
            batch_size=inverse_batch_size,
            dtype=Y.dtype,
            sample_embeddings=inverse_sample_embeddings,
            embedding_low=inverse_embed_low,
            embedding_high=inverse_embed_high,
        )
        main_loss = main_loss + icw * inverse_consistency_loss

        aux["sublosses"] = {
            "mse": mse,
            "energy_loss": energy_loss,
            "coverage_loss": coverage_loss,
            "tail_pinball_loss": tail_loss,
            "inverse_consistency_loss": inverse_consistency_loss,
            "kl_loss": kl_loss,
            "main_loss": main_loss,
        }
        aux["debug"] = {
            "negative_grads": negative_grads,
            "ng_loss": ng_loss,
            "effective_batch_size": eff_batch_size,
            "selected": selected,
            "pct": pct,
            "std": std,
            "qvalues": qvalues,
            "logstds": logstds,
            "counts": counts,
            "step": step,
            "output_weights": output_weights,
            "mse_dependent": mse_dependent,
            "mse_independent": mse_independent,
            "dep_energy": dep_energy,
            "indep_energy": indep_energy,
            "out_vs_in_energy_weight": ovie,
            "energy_weight": ew,
            "energy_pairwise_weight": as_schedule(energy_pairwise_weight)(step),
            "energy_n_samples": n_energy_samples,
            "energy_outputs_independent": bool(energy_outputs_independent),
            "energy_z_distribution": z_dist,
            "energy_z_normal_mean": z_normal_mean,
            "energy_z_normal_std": z_normal_std,
            "energy_z_normal_clip": z_normal_clip,
            "dep_energy_term_a": dep_dbg.get("energy_term_a", 0.0),
            "dep_energy_term_b": dep_dbg.get("energy_term_b", 0.0),
            "indep_energy_term_a": indep_dbg.get("energy_term_a", 0.0),
            "indep_energy_term_b": indep_dbg.get("energy_term_b", 0.0),
            "coverage_calibration_weight": ccw,
            "coverage_interval_low": qlow,
            "coverage_interval_high": qhigh,
            "coverage_temperature": temp_cov,
            "dep_coverage": cdep_dbg.get("coverage", 0.0),
            "indep_coverage": cindep_dbg.get("coverage", 0.0),
            "dep_target_coverage": cdep_dbg.get("target_coverage", target_coverage),
            "indep_target_coverage": cindep_dbg.get("target_coverage", target_coverage),
            "tail_pinball_weight": tpw,
            "tail_tau_low": tail_low,
            "tail_tau_high": tail_high,
            "dep_tail_pinball_loss": tdep_loss,
            "indep_tail_pinball_loss": tindep_loss,
            "inverse_consistency_weight": icw,
            **inverse_dbg,
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
    dispatch: LoggerDispatch | None = None,
    xy_batches: Optional[Tuple] = None,
    enable_jax_tqdm: bool = False,
    init_params: Optional[ParameterTree] = None,
    cached_step: Optional[CompiledTrainingStep] = None,
    skip_weight_init: bool = False,
    skip_loss_history: bool = False,
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
    t_batch = time.time()
    if streaming_mode:
        logger.info("Using streaming batch generation (lower GPU memory, slightly slower)")
        # generate a small batch just for shape inference during compilation
        xbatches, ybatches = get_step_batches(batch_key)
    elif xy_batches is not None:
        xbatches, ybatches = xy_batches
    else:
        xbatches, ybatches = get_new_batches()
    batch_time = time.time() - t_batch
    if batch_time > 1.0:
        logger.info(f"Batch generation took {batch_time:.1f}s")

    # store per_output_weights in params BEFORE filter_by_tag (filter creates copies)
    if not skip_weight_init:
        per_output_weights = expand_weights_to_outputs(dman.get_weights(), stack.networks)
        weights_arr = jnp.asarray(per_output_weights)
        weights_replicated = jnp.broadcast_to(
            weights_arr, (training_config.n_replicates, len(per_output_weights))
        )
        params.at(
            "global/per_output_weights",
            weights_replicated,
            tags=["non_grad", "local"],
            overwrite=True,
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
        f"Done initializing optimizer, n_replicates: {training_config.n_replicates}, "
        f"batches: {xbatches.shape[1]}, batch_per_step: {training_config.batches_per_step}"
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
    dispatch = dispatch or NullDispatch()

    dispatch.on_start(training_config, stack)

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
            if i > 1 and (i - 1) % step_per_epoch == 0:
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

        if "loss" in step_history and not skip_loss_history:
            loss_history.append(step_history["loss"])

        dispatch.on_step(i, training_config, step_history, stack)

    logger.info("Training loop finished, starting cleanup...")

    t_sync = time.time()
    jax.block_until_ready(params)
    logger.info(f"GPU sync (params) took {time.time() - t_sync:.2f}s")

    if loss_history:
        t_sync = time.time()
        jax.block_until_ready(loss_history)
        logger.info(f"GPU sync (loss_history) took {time.time() - t_sync:.2f}s")

    t_callbacks = time.time()
    dispatch.on_end(total_steps, training_config, step_history, stack)
    logger.info(f"End callbacks took {time.time() - t_callbacks:.2f}s")

    logger.info(f"End of training for {training_config.n_epochs} epochs")

    return params, loss_history, StepHistorySnapshot.from_raw(step_history)


##────────────────────────────────────────────────────────────────────────────}}}
