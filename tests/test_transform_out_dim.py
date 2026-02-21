import jax
import jax.numpy as jnp
import numpy as np

import biocomp.biorules as br
from biocomp.compute import ComputeStack
from biocomp.config import SIMPLE_NODES_COMPUTE_CONFIG
from biocomp.library import LibraryContext
from biocomp.network import recipe_to_networks

pytest_plugins = ["test_declarative_recipes"]

FORWARD_TRANSFORM_NODES = ("transcription", "translation", "sequestron_ERN")


def _build_stack(recipe, lib, config):
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True)
        stack = ComputeStack([networks[0]])
        stack.build(config=config)
    return stack


def _explicit_transform_out_dim_config(out_dim: int):
    config = SIMPLE_NODES_COMPUTE_CONFIG.model_copy(deep=True)
    config.extra = {}
    assert config.node_functions is not None
    for node_name in FORWARD_TRANSFORM_NODES:
        config.node_functions[node_name].kwargs["out_dim"] = out_dim
    return config


def test_transform_out_dim_default_one_parity(lib, simple_single_ern):
    config_ssot = SIMPLE_NODES_COMPUTE_CONFIG.model_copy(deep=True)
    config_ssot.extra = {"transform_out_dim": 1}
    config_explicit = _explicit_transform_out_dim_config(out_dim=1)

    stack_ssot = _build_stack(simple_single_ern, lib, config_ssot)
    stack_explicit = _build_stack(simple_single_ern, lib, config_explicit)

    assert stack_ssot.layers is not None
    for layer in stack_ssot.layers:
        if layer.f_type in FORWARD_TRANSFORM_NODES:
            assert layer.f_out_shapes == [(1,)]

    init_key = jax.random.PRNGKey(123)
    params_ssot = stack_ssot.init(init_key)
    params_explicit = stack_explicit.init(init_key)

    nrv_ssot = int(params_ssot["global/number_of_random_variables"])
    nrv_explicit = int(params_explicit["global/number_of_random_variables"])
    assert nrv_ssot == nrv_explicit

    input_value = jnp.array([0.37], dtype=jnp.float32)
    for apply_key in jax.random.split(jax.random.PRNGKey(999), 3):
        random_vars = jax.random.normal(apply_key, (nrv_ssot,))
        y_ssot, _ = stack_ssot.apply(params_ssot, input_value, random_vars, apply_key)
        y_explicit, _ = stack_explicit.apply(params_explicit, input_value, random_vars, apply_key)
        np.testing.assert_allclose(np.asarray(y_ssot), np.asarray(y_explicit), rtol=1e-6, atol=1e-6)

    committed_ssot = stack_ssot.commit(params_ssot)
    committed_explicit = stack_explicit.commit(params_explicit)
    assert committed_ssot == committed_explicit


def test_transform_out_dim_nd_forward_only(lib, simple_single_ern):
    out_dim = 8
    config = SIMPLE_NODES_COMPUTE_CONFIG.model_copy(deep=True)
    config.extra = {"transform_out_dim": out_dim}

    stack = _build_stack(simple_single_ern, lib, config)
    assert stack.layers is not None
    assert stack.config is not None
    assert stack.config.node_functions is not None

    assert stack.config.node_functions["transcription"].kwargs.get("out_dim") == out_dim
    assert stack.config.node_functions["translation"].kwargs.get("out_dim") == out_dim
    assert stack.config.node_functions["sequestron_ERN"].kwargs.get("out_dim") == out_dim
    assert "out_dim" not in stack.config.node_functions["inv_transcription"].kwargs
    assert "out_dim" not in stack.config.node_functions["inv_translation"].kwargs

    for layer in stack.layers:
        if layer.f_type in FORWARD_TRANSFORM_NODES:
            assert layer.f_out_shapes == [(out_dim,)]
        if layer.f_type.startswith("inv_"):
            assert all(tuple(shape) == (1,) for shape in layer.f_out_shapes)
        if layer.f_type == "output":
            assert all(shape == (1,) for shape in layer.f_out_shapes)

    init_key = jax.random.PRNGKey(7)
    params = stack.init(init_key)
    nrv = int(params["global/number_of_random_variables"])
    random_vars = jax.random.normal(jax.random.PRNGKey(8), (nrv,))
    yhat, _ = stack.apply(params, jnp.array([0.42], dtype=jnp.float32), random_vars, init_key)

    assert yhat.ndim == 1
    assert yhat.shape[0] == stack.get_nb_outputs()

    committed = stack.commit(params)
    assert len(committed) == 1
