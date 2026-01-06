"""Tests for bias node parameter initialization and gradient flow."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from pytest import approx
from biocomp.nodes.bias import bias
from biocomp.parameters import ParameterTree


class MockNode:
    def __init__(self, extra):
        self._extra = extra

    @property
    def extra(self):
        return self._extra


class MockStackNode:
    def __init__(self, extra):
        self._extra = extra
        self.network_id = 0
        self.node_id = 0

    def get(self, stack):
        return MockNode(self._extra)


@pytest.fixture
def fluo_bias_range_spec():
    return {
        "tu_id": 0,
        "value": {"min": 0.0, "max": 0.5},
        "protein": "mMaroon1",
        "units": "Rescaled AU",
    }


@pytest.fixture
def fluo_bias_scalar_spec():
    return {
        "tu_id": 0,
        "value": 0.3,
        "protein": "mMaroon1",
        "units": "Rescaled AU",
    }


def test_bias_reads_fluo_bias_key_with_range(fluo_bias_range_spec):
    """Bias node reads 'fluo_bias' key and correctly parses min/max range.

    This is a regression test for the bug where bias nodes looked for
    'fluo_bias_data' key but biorules.py stored the data under 'fluo_bias'.
    """
    mock_node = MockStackNode(extra={"fluo_bias": fluo_bias_range_spec})
    nodelist = [mock_node]

    layer_instance = bias(
        input_shapes=[],
        n_outputs=1,
        stack=None,
        namespace="local/bias_test",
        shape=(1,),
    )

    params = ParameterTree()
    key = jax.random.PRNGKey(42)

    layer_instance.prepare(params, nodelist, key)

    min_value = float(np.asarray(params["local/bias_test/min_value"]).flatten()[0])
    max_value = float(np.asarray(params["local/bias_test/max_value"]).flatten()[0])

    assert min_value == 0.0
    assert max_value == 0.5
    assert min_value != max_value, "min != max required for gradient flow during design"


def test_bias_reads_legacy_fluo_bias_data_key(fluo_bias_range_spec):
    """Bias node falls back to 'fluo_bias_data' key for backward compatibility."""
    mock_node = MockStackNode(extra={"fluo_bias_data": fluo_bias_range_spec})
    nodelist = [mock_node]

    layer_instance = bias(
        input_shapes=[],
        n_outputs=1,
        stack=None,
        namespace="local/bias_test",
        shape=(1,),
    )

    params = ParameterTree()
    key = jax.random.PRNGKey(42)

    layer_instance.prepare(params, nodelist, key)

    min_value = float(np.asarray(params["local/bias_test/min_value"]).flatten()[0])
    max_value = float(np.asarray(params["local/bias_test/max_value"]).flatten()[0])

    assert min_value == 0.0
    assert max_value == 0.5


def test_bias_prefers_fluo_bias_over_fluo_bias_data():
    """When both keys exist, 'fluo_bias' takes precedence."""
    spec_new = {"tu_id": 0, "value": {"min": 0.1, "max": 0.9}, "protein": "A", "units": "AU"}
    spec_old = {"tu_id": 0, "value": {"min": 0.2, "max": 0.8}, "protein": "B", "units": "AU"}

    mock_node = MockStackNode(extra={"fluo_bias": spec_new, "fluo_bias_data": spec_old})
    nodelist = [mock_node]

    layer_instance = bias(
        input_shapes=[],
        n_outputs=1,
        stack=None,
        namespace="local/bias_test",
        shape=(1,),
    )

    params = ParameterTree()
    layer_instance.prepare(params, nodelist, jax.random.PRNGKey(42))

    min_value = float(np.asarray(params["local/bias_test/min_value"]).flatten()[0])
    max_value = float(np.asarray(params["local/bias_test/max_value"]).flatten()[0])

    assert min_value == approx(0.1), "Should use fluo_bias (0.1), not fluo_bias_data (0.2)"
    assert max_value == approx(0.9), "Should use fluo_bias (0.9), not fluo_bias_data (0.8)"


def test_bias_with_scalar_value_is_locked(fluo_bias_scalar_spec):
    """Scalar value is locked: min == max == value (consistent with hard_bias behavior)."""
    mock_node = MockStackNode(extra={"fluo_bias": fluo_bias_scalar_spec})
    nodelist = [mock_node]

    layer_instance = bias(
        input_shapes=[],
        n_outputs=1,
        stack=None,
        namespace="local/bias_test",
        shape=(1,),
    )

    params = ParameterTree()
    layer_instance.prepare(params, nodelist, jax.random.PRNGKey(42))

    min_value = float(np.asarray(params["local/bias_test/min_value"]).flatten()[0])
    max_value = float(np.asarray(params["local/bias_test/max_value"]).flatten()[0])

    assert min_value == approx(0.3), "Scalar value should lock min to the value"
    assert max_value == approx(0.3), "Scalar value should lock max to the value"

    output, _ = layer_instance.apply(params=params, node_id=jnp.array(0))
    output_value = float(output.flatten()[0])
    assert output_value == approx(0.3, rel=0.01), "Output should equal the locked scalar value"


def test_bias_default_when_no_spec():
    """Without fluo_bias spec, uses locked default value 0.5 (min == max == 0.5)."""
    mock_node = MockStackNode(extra={})
    nodelist = [mock_node]

    layer_instance = bias(
        input_shapes=[],
        n_outputs=1,
        stack=None,
        namespace="local/bias_test",
        shape=(1,),
    )

    params = ParameterTree()
    layer_instance.prepare(params, nodelist, jax.random.PRNGKey(42))

    min_value = float(np.asarray(params["local/bias_test/min_value"]).flatten()[0])
    max_value = float(np.asarray(params["local/bias_test/max_value"]).flatten()[0])

    assert min_value == approx(0.5), "Default should be locked at 0.5 (min)"
    assert max_value == approx(0.5), "Default should be locked at 0.5 (max)"

    output, _ = layer_instance.apply(params=params, node_id=jnp.array(0))
    output_value = float(output.flatten()[0])
    assert output_value == approx(0.5, rel=0.01), "Default output should be 0.5"


def test_bias_output_varies_with_raw_value_when_unlocked():
    """Verify output changes with raw_value when min != max (unlocked)."""
    spec = {"tu_id": 0, "value": {"min": 0.0, "max": 1.0}, "protein": "X", "units": "AU"}
    mock_node = MockStackNode(extra={"fluo_bias": spec})
    nodelist = [mock_node]

    layer_instance = bias(
        input_shapes=[],
        n_outputs=1,
        stack=None,
        namespace="local/bias_test",
        shape=(1,),
    )

    params = ParameterTree()
    layer_instance.prepare(params, nodelist, jax.random.PRNGKey(42))

    out1, _ = layer_instance.apply(params=params, node_id=jnp.array(0))

    raw_val_orig = params["local/bias_test/raw_value"]
    params["local/bias_test/raw_value"] = raw_val_orig + 0.5

    out2, _ = layer_instance.apply(params=params, node_id=jnp.array(0))

    params["local/bias_test/raw_value"] = raw_val_orig

    assert not np.allclose(out1, out2), (
        "Output should change when raw_value changes (unlocked bias)"
    )


def test_bias_output_constant_when_locked():
    """Verify output stays constant when min == max (locked bias via explicit range)."""
    spec = {"tu_id": 0, "value": {"min": 0.5, "max": 0.5}, "protein": "X", "units": "AU"}
    mock_node = MockStackNode(extra={"fluo_bias": spec})
    nodelist = [mock_node]

    layer_instance = bias(
        input_shapes=[],
        n_outputs=1,
        stack=None,
        namespace="local/bias_test",
        shape=(1,),
    )

    params = ParameterTree()
    layer_instance.prepare(params, nodelist, jax.random.PRNGKey(42))

    out1, _ = layer_instance.apply(params=params, node_id=jnp.array(0))

    raw_val_orig = params["local/bias_test/raw_value"]
    scale_orig = params["local/bias_test/scale"]

    params["local/bias_test/raw_value"] = raw_val_orig + 0.5
    params["local/bias_test/scale"] = scale_orig + 1.0

    out2, _ = layer_instance.apply(params=params, node_id=jnp.array(0))

    assert np.allclose(out1, out2), "Output should be constant when bias is locked (min == max)"
    assert float(out1.flatten()[0]) == approx(0.5), "Locked output should equal min_value"
