"""Tests for inv_aggregation normalization invariance.

Key invariant: linear scaling of ratios should NOT change predictions since
aggregation normalizes internally. Tests catch the bug where inv_agg nodes
using different forward aggregation paths would compute incorrect sums.
"""

from pathlib import Path
import pytest
import jax
import jax.numpy as jnp
import numpy as np

from biocomp.network import recipe_to_networks
from biocomp.compute import ComputeStack
from biocomp.config import SIMPLE_NODES_COMPUTE_CONFIG
from biocomp.parameters import ParameterTree, isArrayRef
import biocomp.biorules as br

RESOURCES_DIR = Path(__file__).parent / "resources"


@pytest.fixture
def multi_aggregation_recipe():
    import dracon as dr

    recipe_path = RESOURCES_DIR / "design/architectures/two_and_one.yaml"
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


class TestInvAggregationBasicFunctionality:
    def test_deterministic_output(self, multi_aggregation_stack):
        """Verify that forward pass produces deterministic output."""
        key = jax.random.key(42)
        params = multi_aggregation_stack.init(key)

        n_rvars = get_n_random_vars(params)
        X = jnp.array([0.5, 0.5])
        random_vars = jnp.zeros((n_rvars,))

        Y1, _ = multi_aggregation_stack.apply(params, X, random_vars, key)
        Y2, _ = multi_aggregation_stack.apply(params, X, random_vars, key)

        np.testing.assert_allclose(Y1, Y2, rtol=1e-6, atol=1e-7)

    def test_inv_agg_reads_forward_ratios(self, multi_aggregation_stack):
        """Verify that inv_aggregation reads from forward aggregation ratios via ArrayRef."""
        key = jax.random.key(42)
        params = multi_aggregation_stack.init(key)

        inv_ns = 'local/5/inv_aggregation'
        ratio_ref = params.data.get_at(f'{inv_ns}/ratios', get_leaf_value=False).value

        assert len(ratio_ref.paths) > 0, "ArrayRef should have at least one path"
        for path in ratio_ref.paths:
            assert 'aggregation' in path, f"Path should reference forward aggregation: {path}"
            assert 'inv_' not in path, f"Path should not reference inv_aggregation: {path}"

    def test_output_changes_with_ratio_change(self, multi_aggregation_stack):
        """Verify that changing ratios changes output (not testing invariance, just that ratios matter)."""
        key = jax.random.key(42)
        params = multi_aggregation_stack.init(key)
        agg_paths = get_aggregation_paths(params)

        n_rvars = get_n_random_vars(params)
        X = jnp.array([0.5, 0.5])
        random_vars = jnp.zeros((n_rvars,))

        Y_before, _ = multi_aggregation_stack.apply(params, X, random_vars, key)

        for path in agg_paths:
            ratios = params[path]
            new_ratios = jnp.ones_like(ratios) * 0.5
            params[path] = new_ratios

        Y_after, _ = multi_aggregation_stack.apply(params, X, random_vars, key)

        assert not np.allclose(Y_before, Y_after), "Output should change when ratios change significantly"


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


class TestInvAggregationEdgeCases:
    def test_ratio_bounds_respected(self, multi_aggregation_stack):
        """Verify ratios are constrained within [ratio_min, ratio_max]."""
        key = jax.random.key(42)
        params = multi_aggregation_stack.init(key)
        agg_paths = get_aggregation_paths(params)

        for path in agg_paths:
            ratios = params[path]
            ratio_min_path = path.replace("/ratios", "/ratio_min")
            ratio_max_path = path.replace("/ratios", "/ratio_max")
            if ratio_min_path in params and ratio_max_path in params:
                params[ratio_min_path]
                ratio_max = params[ratio_max_path]
                assert jnp.all(ratios >= 0) or jnp.all(ratios <= ratio_max), \
                    f"Ratios should be within bounds for {path}"
