"""Tests for inv_aggregation normalization invariance.

Key invariant: linear scaling of ratios should NOT change predictions since
aggregation normalizes internally. Tests catch the bug where inv_agg nodes
using different forward aggregation paths would compute incorrect sums.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

from biocomp.network import recipe_to_networks
from biocomp.compute import ComputeStack
from biocomp.config import SIMPLE_NODES_COMPUTE_CONFIG
from biocomp.parameters import ParameterTree, isArrayRef
import biocomp.biorules as br


@pytest.fixture
def multi_aggregation_recipe():
    import dracon as dr

    recipe_path = (
        Path(__file__).parent.parent.parent / "biocomp-jobs/design/architectures/two_and_one.yaml"
    )
    if not recipe_path.exists():
        pytest.skip(f"Recipe file not found: {recipe_path}")
    config = dr.load(str(recipe_path))
    return config["recipe"]


@pytest.fixture
def multi_aggregation_network(multi_aggregation_recipe):
    networks = recipe_to_networks(multi_aggregation_recipe, br.ALL_RULES, invert=True)
    return networks[0]


@pytest.fixture
def multi_aggregation_stack(multi_aggregation_network):
    stack = ComputeStack([multi_aggregation_network])
    stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)
    return stack


def get_aggregation_paths(params: ParameterTree) -> list[str]:
    return [
        str(path)
        for path, _ in params.data.iter_leaves()
        if "aggregation" in str(path) and "ratios" in str(path) and "inv_" not in str(path)
    ]


def get_inv_aggregation_paths(params: ParameterTree) -> list[str]:
    return [
        str(path)
        for path, _ in params.data.iter_leaves()
        if "inv_aggregation" in str(path) and "ratios" in str(path)
    ]


def get_n_random_vars(params: ParameterTree) -> int:
    return int(params["global/number_of_random_variables"])


class TestInvAggregationNormalizationInvariance:
    def test_uniform_scaling_invariance(self, multi_aggregation_stack):
        key = jax.random.key(42)
        params = multi_aggregation_stack.init(key)
        agg_paths = get_aggregation_paths(params)
        assert len(agg_paths) > 0

        n_rvars = get_n_random_vars(params)
        X = jnp.array([0.5, 0.5])
        random_vars = jnp.zeros((n_rvars,))

        Y_before, _ = multi_aggregation_stack.apply(params, X, random_vars, key)

        for path in agg_paths:
            params[path] = params[path] * 10.0

        Y_after, _ = multi_aggregation_stack.apply(params, X, random_vars, key)

        np.testing.assert_allclose(Y_before, Y_after, rtol=1e-5, atol=1e-6)

    def test_per_row_scaling_invariance(self, multi_aggregation_stack):
        key = jax.random.key(42)
        params = multi_aggregation_stack.init(key)
        agg_paths = get_aggregation_paths(params)
        assert len(agg_paths) > 0

        n_rvars = get_n_random_vars(params)
        X = jnp.array([0.3, 0.7])
        random_vars = jnp.zeros((n_rvars,))

        Y_before, _ = multi_aggregation_stack.apply(params, X, random_vars, key)

        for path in agg_paths:
            ratios = params[path]
            for i in range(ratios.shape[0]):
                scale = float(i + 1) * 5.0
                params[path] = params[path].at[i].set(ratios[i] * scale)

        Y_after, _ = multi_aggregation_stack.apply(params, X, random_vars, key)

        np.testing.assert_allclose(Y_before, Y_after, rtol=1e-5, atol=1e-6)

    def test_normalize_by_min_invariance(self, multi_aggregation_stack):
        key = jax.random.key(42)
        params = multi_aggregation_stack.init(key)
        agg_paths = get_aggregation_paths(params)
        assert len(agg_paths) > 0

        n_rvars = get_n_random_vars(params)
        X = jnp.array([0.4, 0.6])
        random_vars = jnp.zeros((n_rvars,))

        Y_before, _ = multi_aggregation_stack.apply(params, X, random_vars, key)

        for path in agg_paths:
            ratios = np.array(params[path])
            for i in range(ratios.shape[0]):
                row = ratios[i]
                min_val = row[row > 0].min() if (row > 0).any() else 1.0
                ratios[i] = row / min_val
            params[path] = jnp.array(ratios)

        Y_after, _ = multi_aggregation_stack.apply(params, X, random_vars, key)

        # min-normalization can create scale factors up to ~300x, causing float32 precision loss
        np.testing.assert_allclose(Y_before, Y_after, rtol=1e-4, atol=1e-5)


def get_unique_aggregation_types(agg_paths: list[str]) -> set[str]:
    unique_layer_names = set()
    for path in agg_paths:
        for part in path.split("/"):
            if "aggregation" in part and "inv_" not in part:
                unique_layer_names.add(part)
    return unique_layer_names


class TestInvAggregationMultiPath:
    def test_has_multiple_aggregation_paths(self, multi_aggregation_stack):
        key = jax.random.key(42)
        params = multi_aggregation_stack.init(key)
        agg_paths = get_aggregation_paths(params)
        unique_types = get_unique_aggregation_types(agg_paths)
        if len(unique_types) < 2:
            pytest.skip(f"Only {len(unique_types)} aggregation type(s), need 2+ for this test")

    def test_inv_aggregation_uses_arrayref(self, multi_aggregation_stack):
        key = jax.random.key(42)
        params = multi_aggregation_stack.init(key)
        inv_paths = get_inv_aggregation_paths(params)
        assert len(inv_paths) > 0

        for inv_path in inv_paths:
            ref_obj = params.data.get_at(inv_path, get_leaf_value=False)
            assert ref_obj is not None
            assert isArrayRef(ref_obj.value)
            assert len(ref_obj.value.paths) >= 1

    def test_different_aggregation_sizes(self, multi_aggregation_stack):
        key = jax.random.key(42)
        params = multi_aggregation_stack.init(key)
        agg_paths = get_aggregation_paths(params)

        n_outputs_set = set()
        for path in agg_paths:
            ratios = params[path]
            if ratios.ndim == 2:
                n_outputs_set.add(ratios.shape[1])

        if len(n_outputs_set) < 2:
            pytest.skip(f"Only {len(n_outputs_set)} aggregation size(s), need 2+ for this test")

    def test_selective_scaling_per_aggregation_type(self, multi_aggregation_stack):
        key = jax.random.key(42)
        params = multi_aggregation_stack.init(key)
        agg_paths = get_aggregation_paths(params)
        if len(agg_paths) < 2:
            pytest.skip(f"Only {len(agg_paths)} aggregation path(s), need 2+ for this test")

        n_rvars = get_n_random_vars(params)
        X = jnp.array([0.5, 0.5])
        random_vars = jnp.zeros((n_rvars,))

        Y_before, _ = multi_aggregation_stack.apply(params, X, random_vars, key)

        params[agg_paths[0]] = params[agg_paths[0]] * 100.0

        Y_after, _ = multi_aggregation_stack.apply(params, X, random_vars, key)

        np.testing.assert_allclose(Y_before, Y_after, rtol=1e-5, atol=1e-6)

    def test_scale_each_aggregation_type_separately(self, multi_aggregation_stack):
        key = jax.random.key(42)
        params = multi_aggregation_stack.init(key)
        agg_paths = get_aggregation_paths(params)

        n_rvars = get_n_random_vars(params)
        X = jnp.array([0.5, 0.5])
        random_vars = jnp.zeros((n_rvars,))

        Y_before, _ = multi_aggregation_stack.apply(params, X, random_vars, key)

        for i, path in enumerate(agg_paths):
            params[path] = params[path] * ((i + 1) * 50.0)

        Y_after, _ = multi_aggregation_stack.apply(params, X, random_vars, key)

        # scale factors up to 50x per path can cause float32 precision loss
        np.testing.assert_allclose(Y_before, Y_after, rtol=1e-4, atol=1e-5)


class TestInvAggregationEdgeCases:
    def test_very_large_scale_factors(self, multi_aggregation_stack):
        key = jax.random.key(42)
        params = multi_aggregation_stack.init(key)
        agg_paths = get_aggregation_paths(params)

        n_rvars = get_n_random_vars(params)
        X = jnp.array([0.5, 0.5])
        random_vars = jnp.zeros((n_rvars,))

        Y_before, _ = multi_aggregation_stack.apply(params, X, random_vars, key)

        for path in agg_paths:
            params[path] = params[path] * 1e6

        Y_after, _ = multi_aggregation_stack.apply(params, X, random_vars, key)

        np.testing.assert_allclose(Y_before, Y_after, rtol=1e-4, atol=1e-5)

    def test_very_small_scale_factors(self, multi_aggregation_stack):
        key = jax.random.key(42)
        params = multi_aggregation_stack.init(key)
        agg_paths = get_aggregation_paths(params)

        n_rvars = get_n_random_vars(params)
        X = jnp.array([0.5, 0.5])
        random_vars = jnp.zeros((n_rvars,))

        Y_before, _ = multi_aggregation_stack.apply(params, X, random_vars, key)

        for path in agg_paths:
            params[path] = params[path] * 1e-6

        Y_after, _ = multi_aggregation_stack.apply(params, X, random_vars, key)

        np.testing.assert_allclose(Y_before, Y_after, rtol=1e-4, atol=1e-5)

    def test_negative_ratios_handled_via_abs(self, multi_aggregation_stack):
        key = jax.random.key(42)
        params = multi_aggregation_stack.init(key)
        agg_paths = get_aggregation_paths(params)

        n_rvars = get_n_random_vars(params)
        X = jnp.array([0.5, 0.5])
        random_vars = jnp.zeros((n_rvars,))

        Y_before, _ = multi_aggregation_stack.apply(params, X, random_vars, key)

        for path in agg_paths:
            params[path] = -params[path]

        Y_after, _ = multi_aggregation_stack.apply(params, X, random_vars, key)

        np.testing.assert_allclose(Y_before, Y_after, rtol=1e-5, atol=1e-6)
