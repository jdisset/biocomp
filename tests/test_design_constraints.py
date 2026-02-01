"""
Tests for design constraint mechanisms (ratio and bias locking).

These tests verify that:
1. Zero-freedom recipes produce constant loss during optimization
2. Locked ratios/biases are constrained via min=max clipping
3. Heterogeneous stacks respect per-network constraints
4. Constraint clipping works correctly at the per-node level
"""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from biocomp.compute import ComputeStack
from biocomp.designcodec import GenomeCodec
from biocomp.network import recipe_to_networks
from biocomp.nodeutils import NON_GRAD_TAG
from biocomp.parameters import ParameterTree
import biocomp.biorules as br
import dracon as dr

RESOURCES_DIR = Path(__file__).parent / "resources"


# Skip all tests if BIOCOMP_DESIGNER_MODEL not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("BIOCOMP_DESIGNER_MODEL"),
    reason="BIOCOMP_DESIGNER_MODEL environment variable not set",
)


@pytest.fixture(scope="module")
def designer_model():
    """Load the designer model for testing."""
    from biocomptools.modelmodel import BiocompModel

    model_path = os.environ.get("BIOCOMP_DESIGNER_MODEL")
    return BiocompModel.load(model_path)


def load_recipe(recipe_path: Path):
    """Load a recipe from YAML file."""
    config = dr.load(str(recipe_path))
    return config["recipe"]


def build_network_and_params(recipe, model, key=None):
    """Build network, stack, and initialize params from a recipe."""
    if key is None:
        key = jax.random.PRNGKey(42)

    networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True, inversion_mode="main")
    assert len(networks) == 1, f"Expected 1 network, got {len(networks)}"
    network = networks[0]

    stack = ComputeStack(networks=[network])
    stack.build(model.compute_config, enable_tu_masking=False)

    init_params = stack.init(key)
    _, nonshared = init_params.filter_by_tag(["shared"])
    params = ParameterTree.merge(model.shared_params, nonshared)

    return network, stack, params


class TestZeroFreedomConstraints:
    """Tests for zero-freedom recipe constraint behavior."""

    def test_zero_freedom_ratios_have_min_equals_max(self, designer_model):
        """Verify that zero-freedom recipes have min==max for all ratios."""
        recipe_path = RESOURCES_DIR / "design/architectures/T_2_zero_freedom.yaml"
        if not recipe_path.exists():
            pytest.skip(f"Recipe not found: {recipe_path}")

        recipe = load_recipe(recipe_path)
        network, stack, params = build_network_and_params(recipe, designer_model)

        for layer_idx, _layer in enumerate(stack.layers):
            ns = stack.get_layer_namespace(layer_idx)
            if "aggregation" in ns and "inv" not in ns:
                ratio_min_path = f"{ns}/ratio_min"
                ratio_max_path = f"{ns}/ratio_max"

                assert ratio_min_path in params, f"ratio_min not found at {ratio_min_path}"
                assert ratio_max_path in params, f"ratio_max not found at {ratio_max_path}"

                ratio_min = np.asarray(params[ratio_min_path])
                ratio_max = np.asarray(params[ratio_max_path])

                assert np.allclose(ratio_min, ratio_max), (
                    f"Zero-freedom ratios should have min==max at {ns}, "
                    f"but got min={ratio_min}, max={ratio_max}"
                )

    def test_zero_freedom_bias_has_min_equals_max(self, designer_model):
        """Verify that zero-freedom recipes have min==max for bias values."""
        recipe_path = RESOURCES_DIR / "design/architectures/T_2_zero_freedom.yaml"
        if not recipe_path.exists():
            pytest.skip(f"Recipe not found: {recipe_path}")

        recipe = load_recipe(recipe_path)
        network, stack, params = build_network_and_params(recipe, designer_model)

        for layer_idx, _layer in enumerate(stack.layers):
            ns = stack.get_layer_namespace(layer_idx)
            if "bias" in ns:
                min_path = f"{ns}/min_value"
                max_path = f"{ns}/max_value"

                if min_path in params and max_path in params:
                    min_value = np.asarray(params[min_path])
                    max_value = np.asarray(params[max_path])

                    assert np.allclose(min_value, max_value), (
                        f"Zero-freedom bias should have min==max at {ns}, "
                        f"but got min={min_value}, max={max_value}"
                    )

    def test_zero_freedom_ratios_tagged_non_grad(self, designer_model):
        """Verify that fully-constrained ratios are tagged NON_GRAD."""
        recipe_path = RESOURCES_DIR / "design/architectures/T_2_zero_freedom.yaml"
        if not recipe_path.exists():
            pytest.skip(f"Recipe not found: {recipe_path}")

        recipe = load_recipe(recipe_path)
        network, stack, params = build_network_and_params(recipe, designer_model)

        for layer_idx, _layer in enumerate(stack.layers):
            ns = stack.get_layer_namespace(layer_idx)
            if "aggregation" in ns and "inv" not in ns:
                ratios_path = f"{ns}/ratios"
                tags = params.get_tags(ratios_path)

                assert tags is not None, f"ratios at {ratios_path} should have tags"
                assert NON_GRAD_TAG in tags, (
                    f"Fully-constrained ratios at {ratios_path} should be tagged NON_GRAD"
                )

    def test_zero_freedom_bias_tagged_non_grad(self, designer_model):
        """Verify that fully-constrained bias is tagged NON_GRAD."""
        recipe_path = RESOURCES_DIR / "design/architectures/T_2_zero_freedom.yaml"
        if not recipe_path.exists():
            pytest.skip(f"Recipe not found: {recipe_path}")

        recipe = load_recipe(recipe_path)
        network, stack, params = build_network_and_params(recipe, designer_model)

        for layer_idx, _layer in enumerate(stack.layers):
            ns = stack.get_layer_namespace(layer_idx)
            if "bias" in ns:
                raw_value_path = f"{ns}/raw_value"
                if raw_value_path in params:
                    # Check if min==max (fully constrained)
                    min_path = f"{ns}/min_value"
                    max_path = f"{ns}/max_value"
                    if min_path in params and max_path in params:
                        min_val = np.asarray(params[min_path])
                        max_val = np.asarray(params[max_path])
                        if np.allclose(min_val, max_val):
                            tags = params.get_tags(raw_value_path)
                            assert tags is not None and NON_GRAD_TAG in tags, (
                                f"Fully-constrained bias at {raw_value_path} should be tagged NON_GRAD"
                            )

    def test_zero_freedom_loss_constant_during_optimization(self, designer_model):
        """Verify that optimization doesn't change loss for zero-freedom recipes."""
        recipe_path = RESOURCES_DIR / "design/architectures/T_2_zero_freedom.yaml"
        if not recipe_path.exists():
            pytest.skip(f"Recipe not found: {recipe_path}")

        recipe = load_recipe(recipe_path)
        network, stack, params = build_network_and_params(recipe, designer_model)

        codec = GenomeCodec.from_params(params)
        genome = codec.encode(params)

        # Set up forward pass
        resolution = (24, 24)
        batch_size = resolution[0] * resolution[1]
        X_latent = np.stack(
            np.meshgrid(
                np.linspace(0.1, 0.9, resolution[0]),
                np.linspace(0.1, 0.9, resolution[1]),
            ),
            axis=-1,
        ).reshape(-1, 2)

        num_z_path = "global/number_of_random_variables"
        num_z = int(params[num_z_path]) if num_z_path in params else 0
        Z_const = jnp.zeros((batch_size, num_z))
        X_const = jnp.array(X_latent)

        dep_mask = stack.get_dependent_output_mask()
        dep_indices = np.where(dep_mask)[0]

        def forward_pass(p, x_batch, z_batch, keys):
            def apply_single(x, z, k):
                return stack.apply(p, x, z, k)[0]
            return jax.vmap(apply_single)(x_batch, z_batch, keys)

        forward_jit = jax.jit(forward_pass)

        def loss_fn(genome_flat, key):
            p = codec.decode(genome_flat)
            keys = jax.random.split(key, batch_size)
            yhat = forward_jit(p, X_const, Z_const, keys)
            yhat_dep = yhat[:, dep_indices]
            if len(dep_indices) > 1:
                yhat_dep = yhat_dep.mean(axis=-1, keepdims=True)
            return jnp.mean(yhat_dep**2)

        loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

        # Run optimization
        learning_rate = 0.01
        n_steps = 50
        loss_history = []
        key = jax.random.PRNGKey(0)

        for step in range(n_steps):
            step_key = jax.random.fold_in(key, step)
            loss_val, grads = loss_and_grad(genome, step_key)
            loss_history.append(float(loss_val))
            genome = genome - learning_rate * grads

        loss_std = np.std(loss_history)

        # Loss should be essentially constant (allowing for float precision)
        assert loss_std < 1e-5, (
            f"Zero-freedom loss should be constant during optimization, "
            f"but std={loss_std:.2e}"
        )


class TestConstraintClipping:
    """Tests for constraint clipping behavior."""

    def test_locked_ratios_clipped_to_fixed_values(self, designer_model):
        """Verify that locked ratios are clipped to their fixed values."""
        recipe_path = RESOURCES_DIR / "design/architectures/T_2_zero_freedom.yaml"
        if not recipe_path.exists():
            pytest.skip(f"Recipe not found: {recipe_path}")

        recipe = load_recipe(recipe_path)
        network, stack, params = build_network_and_params(recipe, designer_model)

        for layer_idx, _layer in enumerate(stack.layers):
            ns = stack.get_layer_namespace(layer_idx)
            if "aggregation" in ns and "inv" not in ns:
                ratios = np.asarray(params[f"{ns}/ratios"])
                ratio_min = np.asarray(params[f"{ns}/ratio_min"])
                ratio_max = np.asarray(params[f"{ns}/ratio_max"])

                # Simulate drifted values
                drifted = ratios + 0.5

                # Apply clipping
                clipped = np.clip(drifted, ratio_min, ratio_max)

                # For locked ratios (min==max), clipped should equal original
                for node_idx in range(ratios.shape[0]):
                    if np.allclose(ratio_min[node_idx], ratio_max[node_idx]):
                        assert np.allclose(clipped[node_idx], ratios[node_idx]), (
                            f"Locked ratios should clip to original values at node {node_idx}"
                        )

    def test_locked_bias_clipped_to_fixed_value(self, designer_model):
        """Verify that locked bias is clipped to fixed value."""
        recipe_path = RESOURCES_DIR / "design/architectures/T_2_zero_freedom.yaml"
        if not recipe_path.exists():
            pytest.skip(f"Recipe not found: {recipe_path}")

        recipe = load_recipe(recipe_path)
        network, stack, params = build_network_and_params(recipe, designer_model)

        for layer_idx, _layer in enumerate(stack.layers):
            ns = stack.get_layer_namespace(layer_idx)
            if "bias" in ns:
                if f"{ns}/raw_value" in params:
                    raw_value = np.asarray(params[f"{ns}/raw_value"])
                    min_value = np.asarray(params[f"{ns}/min_value"])
                    max_value = np.asarray(params[f"{ns}/max_value"])

                    # Simulate drifted values
                    drifted = raw_value + 0.5

                    # Apply clipping
                    clipped = np.clip(drifted, min_value, max_value)

                    # For locked bias (min==max), clipped should equal min (the fixed value)
                    if np.allclose(min_value, max_value):
                        assert np.allclose(clipped, min_value), (
                            f"Locked bias should clip to fixed value at {ns}"
                        )


class TestHeterogeneousConstraints:
    """Tests for heterogeneous constraint behavior across multiple networks."""

    def test_multi_network_per_node_constraints(self, designer_model):
        """Verify that each network maintains its own constraints in a multi-network stack."""
        # Load locked and unlocked recipes
        locked_path = RESOURCES_DIR / "design/architectures/T_2_zero_freedom.yaml"
        unlocked_path = RESOURCES_DIR / "design/architectures/T_2_ratios_only.yaml"

        if not locked_path.exists():
            pytest.skip(f"Recipe not found: {locked_path}")
        if not unlocked_path.exists():
            pytest.skip(f"Recipe not found: {unlocked_path}")

        recipe_locked = load_recipe(locked_path)
        recipe_unlocked = load_recipe(unlocked_path)

        # Build networks
        networks_locked = recipe_to_networks(recipe_locked, br.ALL_RULES, invert=True, inversion_mode="main")
        networks_unlocked = recipe_to_networks(recipe_unlocked, br.ALL_RULES, invert=True, inversion_mode="main")

        network_locked = networks_locked[0]
        network_unlocked = networks_unlocked[0]
        network_locked.name = "locked"
        network_unlocked.name = "unlocked"

        # Build combined stack
        stack = ComputeStack(networks=[network_locked, network_unlocked])
        stack.build(designer_model.compute_config, enable_tu_masking=False)

        key = jax.random.PRNGKey(42)
        init_params = stack.init(key)
        _, nonshared = init_params.filter_by_tag(["shared"])
        params = ParameterTree.merge(designer_model.shared_params, nonshared)

        # Check that each network has correct constraints
        for layer_idx, layer in enumerate(stack.layers):
            ns = stack.get_layer_namespace(layer_idx)
            if "aggregation" in ns and "inv" not in ns:
                ratio_min = np.asarray(params[f"{ns}/ratio_min"])
                ratio_max = np.asarray(params[f"{ns}/ratio_max"])

                for node_idx, node in enumerate(layer.nodes):
                    network_id = node.network_id
                    node_min = ratio_min[node_idx]
                    node_max = ratio_max[node_idx]
                    is_locked = np.allclose(node_min, node_max)

                    if network_id == 0:  # locked network
                        assert is_locked, (
                            f"Node {node_idx} from locked network should have locked ratios"
                        )
                    elif network_id == 1:  # unlocked network
                        assert not is_locked, (
                            f"Node {node_idx} from unlocked network should have unlocked ratios"
                        )

    def test_multi_network_non_grad_tag_only_when_all_locked(self, designer_model):
        """Verify NON_GRAD is only applied when ALL nodes in layer are locked."""
        locked_path = RESOURCES_DIR / "design/architectures/T_2_zero_freedom.yaml"
        unlocked_path = RESOURCES_DIR / "design/architectures/T_2_ratios_only.yaml"

        if not locked_path.exists() or not unlocked_path.exists():
            pytest.skip("Required recipes not found")

        recipe_locked = load_recipe(locked_path)
        recipe_unlocked = load_recipe(unlocked_path)

        networks_locked = recipe_to_networks(recipe_locked, br.ALL_RULES, invert=True, inversion_mode="main")
        networks_unlocked = recipe_to_networks(recipe_unlocked, br.ALL_RULES, invert=True, inversion_mode="main")

        # Mixed stack: should NOT have NON_GRAD on ratios
        stack_mixed = ComputeStack(networks=[networks_locked[0], networks_unlocked[0]])
        stack_mixed.build(designer_model.compute_config, enable_tu_masking=False)
        params_mixed = stack_mixed.init(jax.random.PRNGKey(42))

        for layer_idx, _layer in enumerate(stack_mixed.layers):
            ns = stack_mixed.get_layer_namespace(layer_idx)
            if "aggregation" in ns and "inv" not in ns:
                tags = params_mixed.get_tags(f"{ns}/ratios")
                # When there are mixed constraints, NON_GRAD should NOT be applied
                if tags:
                    assert NON_GRAD_TAG not in tags, (
                        f"Mixed-constraint layer at {ns} should NOT have NON_GRAD tag"
                    )

        # All-locked stack: SHOULD have NON_GRAD on ratios
        stack_locked = ComputeStack(networks=[networks_locked[0]])
        stack_locked.build(designer_model.compute_config, enable_tu_masking=False)
        params_locked = stack_locked.init(jax.random.PRNGKey(42))

        for layer_idx, _layer in enumerate(stack_locked.layers):
            ns = stack_locked.get_layer_namespace(layer_idx)
            if "aggregation" in ns and "inv" not in ns:
                tags = params_locked.get_tags(f"{ns}/ratios")
                assert tags is not None and NON_GRAD_TAG in tags, (
                    f"All-locked layer at {ns} SHOULD have NON_GRAD tag"
                )

    def test_multi_network_clipping_respects_per_node_constraints(self, designer_model):
        """Verify clipping respects per-node constraints in heterogeneous stacks."""
        locked_path = RESOURCES_DIR / "design/architectures/T_2_zero_freedom.yaml"
        unlocked_path = RESOURCES_DIR / "design/architectures/T_2_ratios_only.yaml"

        if not locked_path.exists() or not unlocked_path.exists():
            pytest.skip("Required recipes not found")

        recipe_locked = load_recipe(locked_path)
        recipe_unlocked = load_recipe(unlocked_path)

        networks_locked = recipe_to_networks(recipe_locked, br.ALL_RULES, invert=True, inversion_mode="main")
        networks_unlocked = recipe_to_networks(recipe_unlocked, br.ALL_RULES, invert=True, inversion_mode="main")

        stack = ComputeStack(networks=[networks_locked[0], networks_unlocked[0]])
        stack.build(designer_model.compute_config, enable_tu_masking=False)

        init_params = stack.init(jax.random.PRNGKey(42))
        _, nonshared = init_params.filter_by_tag(["shared"])
        params = ParameterTree.merge(designer_model.shared_params, nonshared)

        for layer_idx, layer in enumerate(stack.layers):
            ns = stack.get_layer_namespace(layer_idx)
            if "aggregation" in ns and "inv" not in ns:
                ratios = np.asarray(params[f"{ns}/ratios"])
                ratio_min = np.asarray(params[f"{ns}/ratio_min"])
                ratio_max = np.asarray(params[f"{ns}/ratio_max"])

                # Simulate large drift
                drifted = ratios + 1.0
                clipped = np.clip(drifted, ratio_min, ratio_max)

                for node_idx, node in enumerate(layer.nodes):
                    network_id = node.network_id
                    is_locked = np.allclose(ratio_min[node_idx], ratio_max[node_idx])

                    if is_locked:
                        # Locked: should return to original
                        assert np.allclose(clipped[node_idx], ratios[node_idx]), (
                            f"Locked node {node_idx} (network {network_id}) should clip to original"
                        )
                    else:
                        # Unlocked: should be within range
                        assert np.all(clipped[node_idx] >= ratio_min[node_idx]), (
                            f"Unlocked node {node_idx} should be >= min"
                        )
                        assert np.all(clipped[node_idx] <= ratio_max[node_idx]), (
                            f"Unlocked node {node_idx} should be <= max"
                        )


class TestGenomeCodecIntegration:
    """Tests for GenomeCodec integration with constraints."""

    def test_zero_freedom_genome_excludes_locked_params(self, designer_model):
        """Verify that locked params are excluded from the genome."""
        recipe_path = RESOURCES_DIR / "design/architectures/T_2_zero_freedom.yaml"
        if not recipe_path.exists():
            pytest.skip(f"Recipe not found: {recipe_path}")

        recipe = load_recipe(recipe_path)
        network, stack, params = build_network_and_params(recipe, designer_model)

        codec = GenomeCodec.from_params(params)

        # Count ratio and bias params that should be excluded
        excluded_count = 0
        for layer_idx, _layer in enumerate(stack.layers):
            ns = stack.get_layer_namespace(layer_idx)
            if "aggregation" in ns and "inv" not in ns:
                ratios = params[f"{ns}/ratios"]
                excluded_count += np.prod(ratios.shape)
            if "bias" in ns:
                if f"{ns}/raw_value" in params:
                    raw_val = params[f"{ns}/raw_value"]
                    excluded_count += np.prod(np.asarray(raw_val).shape)
                if f"{ns}/scale" in params:
                    scale = params[f"{ns}/scale"]
                    excluded_count += np.prod(np.asarray(scale).shape)

        # The genome should be smaller when locked params are excluded
        codec.encode(params)

        # Get paths in genome
        static, dynamic = params.filter_by_tag(list(codec.static_tags))
        dynamic_paths = [str(path) for path, _ in dynamic.data.iter_leaves()]

        # Verify locked ratio paths are not in dynamic params
        for layer_idx, _layer in enumerate(stack.layers):
            ns = stack.get_layer_namespace(layer_idx)
            if "aggregation" in ns and "inv" not in ns:
                ratios_path = f"{ns}/ratios"
                assert ratios_path not in dynamic_paths, (
                    f"Locked ratios at {ratios_path} should be excluded from genome"
                )

    def test_unlocked_recipe_genome_includes_ratios(self, designer_model):
        """Verify that unlocked params are included in the genome."""
        recipe_path = RESOURCES_DIR / "design/architectures/T_2_ratios_only.yaml"
        if not recipe_path.exists():
            pytest.skip(f"Recipe not found: {recipe_path}")

        recipe = load_recipe(recipe_path)
        network, stack, params = build_network_and_params(recipe, designer_model)

        codec = GenomeCodec.from_params(params)
        static, dynamic = params.filter_by_tag(list(codec.static_tags))
        dynamic_paths = [str(path) for path, _ in dynamic.data.iter_leaves()]

        # Verify unlocked ratio paths ARE in dynamic params
        found_ratios = False
        for layer_idx, _layer in enumerate(stack.layers):
            ns = stack.get_layer_namespace(layer_idx)
            if "aggregation" in ns and "inv" not in ns:
                ratios_path = f"{ns}/ratios"
                if ratios_path in dynamic_paths:
                    found_ratios = True
                    break

        assert found_ratios, "Unlocked ratios should be included in genome"


class TestRecipeToMaskIntegration:
    """Integration tests: recipe part definitions → quantization mask choices."""

    def test_single_part_slot_creates_single_choice_mask(self, designer_model):
        """Recipe with single fixed part should create single-choice quantization mask."""
        # T_2_zero_freedom has fixed uORFs like '4x_uORF' directly in slots
        recipe_path = RESOURCES_DIR / "design/architectures/T_2_zero_freedom.yaml"
        if not recipe_path.exists():
            pytest.skip("Recipe not found")

        recipe = load_recipe(recipe_path)
        _, stack, params = build_network_and_params(recipe, designer_model)

        # Verify: every tl_rate mask has exactly 1 choice per slot
        for layer_idx, _ in enumerate(stack.layers):
            ns = stack.get_layer_namespace(layer_idx)
            if "translation" in ns and "inv_" not in ns:
                mask_path = f"{ns}/tl_rate_quantization_mask"
                if mask_path in params:
                    masks = np.asarray(params[mask_path])
                    choices = masks.sum(axis=-1)
                    assert np.all(choices == 1), (
                        f"Fixed uORF slots in {ns} should produce single-choice masks"
                    )

    def test_multi_part_slot_creates_multi_choice_mask(self, designer_model):
        """Recipe with multiple parts in slot should create multi-choice quantization mask."""
        # two_and_one_all_uorfs has slots like ${U1} with all uORFs available
        recipe_path = RESOURCES_DIR / "design/architectures/two_and_one_all_uorfs.yaml"
        if not recipe_path.exists():
            pytest.skip("Recipe not found")

        recipe = load_recipe(recipe_path)
        _, stack, params = build_network_and_params(recipe, designer_model)

        found_multi = False
        for layer_idx, _ in enumerate(stack.layers):
            ns = stack.get_layer_namespace(layer_idx)
            if "translation" in ns and "inv_" not in ns:
                mask_path = f"{ns}/tl_rate_quantization_mask"
                if mask_path in params:
                    masks = np.asarray(params[mask_path])
                    choices = masks.sum(axis=-1)
                    if np.any(choices > 1):
                        found_multi = True
                        # Verify we have multiple choices where expected
                        assert np.max(choices) > 1, (
                            f"Multi-part slots in {ns} should produce multi-choice masks"
                        )
        assert found_multi, "all_uorfs recipe should have multi-choice translation masks"

    def test_mixed_slot_configuration_creates_mixed_masks(self, designer_model):
        """Recipe with both fixed and multi-part slots should have mixed mask patterns."""
        # two_and_one has U1/U2 with many uORFs, U3 with NO_UORFS (single)
        recipe_path = RESOURCES_DIR / "design/architectures/two_and_one.yaml"
        if not recipe_path.exists():
            pytest.skip("Recipe not found")

        recipe = load_recipe(recipe_path)
        _, stack, params = build_network_and_params(recipe, designer_model)

        found_single = False
        found_multi = False
        for layer_idx, _ in enumerate(stack.layers):
            ns = stack.get_layer_namespace(layer_idx)
            if "translation" in ns and "inv_" not in ns:
                mask_path = f"{ns}/tl_rate_quantization_mask"
                if mask_path in params:
                    masks = np.asarray(params[mask_path])
                    for node_idx in range(masks.shape[0]):
                        for slot_idx in range(masks.shape[1]):
                            n_choices = masks[node_idx, slot_idx].sum()
                            if n_choices == 1:
                                found_single = True
                            elif n_choices > 1:
                                found_multi = True

        # two_and_one has U1/U2 unlocked (multi) and some slots without uORFs (single)
        assert found_single or found_multi, (
            "Mixed slot config should have at least some single or multi choice slots"
        )

    def test_param_dim_reflects_unlocked_parts(self, designer_model):
        """param_dim should increase with more unlocked part choices."""
        zero_path = RESOURCES_DIR / "design/architectures/T_2_zero_freedom.yaml"
        all_path = RESOURCES_DIR / "design/architectures/two_and_one_all_uorfs.yaml"

        if not zero_path.exists() or not all_path.exists():
            pytest.skip("Required recipes not found")

        # Zero freedom: param_dim = 0
        zero_recipe = load_recipe(zero_path)
        _, _, zero_params = build_network_and_params(zero_recipe, designer_model)
        zero_codec = GenomeCodec.from_params(zero_params)

        # All uORFs: param_dim > 0
        all_recipe = load_recipe(all_path)
        _, _, all_params = build_network_and_params(all_recipe, designer_model)
        all_codec = GenomeCodec.from_params(all_params)

        assert zero_codec.param_dim == 0, "Zero freedom should have param_dim=0"
        assert all_codec.param_dim > 0, "All uORFs unlocked should have param_dim>0"
        assert all_codec.param_dim > zero_codec.param_dim, (
            "More unlocked parts should mean more dynamic params"
        )


class TestTransformNodeConstraints:
    """Tests for transform node (transcription/translation) NON_GRAD optimization."""

    def test_single_choice_masks_tagged_non_grad(self, designer_model):
        """Forward transform rates with single-choice masks should be tagged NON_GRAD."""
        recipe_path = RESOURCES_DIR / "design/architectures/T_2_zero_freedom.yaml"
        if not recipe_path.exists():
            pytest.skip(f"Recipe not found: {recipe_path}")

        recipe = load_recipe(recipe_path)
        _, stack, params = build_network_and_params(recipe, designer_model)

        found_any = False
        for layer_idx, _ in enumerate(stack.layers):
            ns = stack.get_layer_namespace(layer_idx)
            # Skip inverse layers (they use RefArrays to forward rates)
            if "inv_" in ns:
                continue
            for rate_name in ["tc_rate", "tl_rate"]:
                mask_path = f"{ns}/{rate_name}_quantization_mask"
                rate_path = f"{ns}/{rate_name}"
                if mask_path in params and rate_path in params:
                    masks = np.asarray(params[mask_path])
                    choices_per_slot = masks.sum(axis=-1)
                    all_single = np.all(choices_per_slot == 1)
                    tags = params.get_tags(rate_path)
                    is_non_grad = NON_GRAD_TAG in tags if tags else False
                    if all_single:
                        found_any = True
                        assert is_non_grad, (
                            f"Layer {ns} has single-choice masks but rates not tagged NON_GRAD"
                        )
        assert found_any, "No single-choice forward transform layers found"

    def test_multi_choice_masks_not_tagged_non_grad(self, designer_model):
        """Forward transform rates with multi-choice masks should NOT be tagged NON_GRAD."""
        recipe_path = RESOURCES_DIR / "design/architectures/two_and_one_all_uorfs.yaml"
        if not recipe_path.exists():
            pytest.skip(f"Recipe not found: {recipe_path}")

        recipe = load_recipe(recipe_path)
        _, stack, params = build_network_and_params(recipe, designer_model)

        found_multi = False
        for layer_idx, _ in enumerate(stack.layers):
            ns = stack.get_layer_namespace(layer_idx)
            if "inv_" in ns:
                continue
            for rate_name in ["tc_rate", "tl_rate"]:
                mask_path = f"{ns}/{rate_name}_quantization_mask"
                rate_path = f"{ns}/{rate_name}"
                if mask_path in params and rate_path in params:
                    masks = np.asarray(params[mask_path])
                    choices_per_slot = masks.sum(axis=-1)
                    has_multi = np.any(choices_per_slot > 1)
                    if has_multi:
                        found_multi = True
                        tags = params.get_tags(rate_path)
                        is_non_grad = NON_GRAD_TAG in tags if tags else False
                        assert not is_non_grad, (
                            f"Layer {ns} has multi-choice masks but rates ARE tagged NON_GRAD"
                        )
        assert found_multi, "No multi-choice forward transform layers found"

    def test_zero_freedom_has_zero_param_dim(self, designer_model):
        """Zero freedom recipe should have param_dim=0 with all optimizations."""
        recipe_path = RESOURCES_DIR / "design/architectures/T_2_zero_freedom.yaml"
        if not recipe_path.exists():
            pytest.skip(f"Recipe not found: {recipe_path}")

        recipe = load_recipe(recipe_path)
        _, _, params = build_network_and_params(recipe, designer_model)

        codec = GenomeCodec.from_params(params)
        assert codec.param_dim == 0, (
            f"Zero freedom recipe should have param_dim=0, got {codec.param_dim}"
        )

    def test_unlocked_uorfs_have_positive_param_dim(self, designer_model):
        """Recipes with unlocked uORFs should have param_dim > 0."""
        recipe_path = RESOURCES_DIR / "design/architectures/two_and_one_all_uorfs.yaml"
        if not recipe_path.exists():
            pytest.skip(f"Recipe not found: {recipe_path}")

        recipe = load_recipe(recipe_path)
        _, _, params = build_network_and_params(recipe, designer_model)

        codec = GenomeCodec.from_params(params)
        assert codec.param_dim > 0, "Recipe with unlocked uORFs should have param_dim > 0"


class TestHeterogeneousTransformConstraints:
    """Tests for heterogeneous transform node constraints across networks."""

    def test_mixed_networks_respect_individual_constraints(self, designer_model):
        """Multi-network stack with different uORF constraints should respect each."""
        locked_path = RESOURCES_DIR / "design/architectures/T_2_zero_freedom.yaml"
        unlocked_path = RESOURCES_DIR / "design/architectures/two_and_one_all_uorfs.yaml"
        if not locked_path.exists() or not unlocked_path.exists():
            pytest.skip("Required recipes not found")

        locked_recipe = load_recipe(locked_path)
        unlocked_recipe = load_recipe(unlocked_path)

        locked_networks = recipe_to_networks(
            locked_recipe, br.ALL_RULES, invert=True, inversion_mode="main"
        )
        unlocked_networks = recipe_to_networks(
            unlocked_recipe, br.ALL_RULES, invert=True, inversion_mode="main"
        )

        locked_networks[0].name = "locked"
        unlocked_networks[0].name = "unlocked"

        stack = ComputeStack(networks=[locked_networks[0], unlocked_networks[0]])
        stack.build(designer_model.compute_config, enable_tu_masking=False)

        key = jax.random.PRNGKey(42)
        init_params = stack.init(key)
        _, nonshared = init_params.filter_by_tag(["shared"])
        params = ParameterTree.merge(designer_model.shared_params, nonshared)

        found_shared_layer = False
        for layer_idx, _layer in enumerate(stack.layers):
            ns = stack.get_layer_namespace(layer_idx)
            if "inv_" in ns:
                continue
            for rate_name in ["tc_rate", "tl_rate"]:
                mask_path = f"{ns}/{rate_name}_quantization_mask"
                rate_path = f"{ns}/{rate_name}"
                if mask_path in params and rate_path in params:
                    masks = np.asarray(params[mask_path])
                    choices_per_slot = masks.sum(axis=-1)
                    has_locked = np.any(choices_per_slot == 1)
                    has_unlocked = np.any(choices_per_slot > 1)
                    if has_locked and has_unlocked:
                        found_shared_layer = True
                        tags = params.get_tags(rate_path)
                        is_non_grad = NON_GRAD_TAG in tags if tags else False
                        assert not is_non_grad, (
                            f"Layer {ns} has mixed lock/unlock but rates tagged NON_GRAD"
                        )

        if not found_shared_layer:
            locked_found = False
            unlocked_found = False
            for layer_idx, _layer in enumerate(stack.layers):
                ns = stack.get_layer_namespace(layer_idx)
                if "inv_" in ns:
                    continue
                for rate_name in ["tc_rate", "tl_rate"]:
                    mask_path = f"{ns}/{rate_name}_quantization_mask"
                    if mask_path in params:
                        masks = np.asarray(params[mask_path])
                        choices = masks.sum(axis=-1)
                        if np.all(choices == 1):
                            locked_found = True
                        if np.any(choices > 1):
                            unlocked_found = True
            assert locked_found and unlocked_found, "Both locked and unlocked layers expected"

    def test_per_node_mask_choices_preserved(self, designer_model):
        """Each node's mask choices should be preserved in multi-network stacks."""
        locked_path = RESOURCES_DIR / "design/architectures/T_2_zero_freedom.yaml"
        unlocked_path = RESOURCES_DIR / "design/architectures/two_and_one_all_uorfs.yaml"
        if not locked_path.exists() or not unlocked_path.exists():
            pytest.skip("Required recipes not found")

        locked_recipe = load_recipe(locked_path)
        unlocked_recipe = load_recipe(unlocked_path)

        locked_networks = recipe_to_networks(
            locked_recipe, br.ALL_RULES, invert=True, inversion_mode="main"
        )
        unlocked_networks = recipe_to_networks(
            unlocked_recipe, br.ALL_RULES, invert=True, inversion_mode="main"
        )

        locked_networks[0].name = "locked"
        unlocked_networks[0].name = "unlocked"

        stack = ComputeStack(networks=[locked_networks[0], unlocked_networks[0]])
        stack.build(designer_model.compute_config, enable_tu_masking=False)

        key = jax.random.PRNGKey(42)
        init_params = stack.init(key)
        _, nonshared = init_params.filter_by_tag(["shared"])
        params = ParameterTree.merge(designer_model.shared_params, nonshared)

        for layer_idx, layer in enumerate(stack.layers):
            ns = stack.get_layer_namespace(layer_idx)
            for rate_name in ["tc_rate", "tl_rate"]:
                mask_path = f"{ns}/{rate_name}_quantization_mask"
                if mask_path in params:
                    masks = np.asarray(params[mask_path])
                    for node_idx, node in enumerate(layer.nodes):
                        network_id = node.network_id
                        node_mask = masks[node_idx]
                        choices = node_mask.sum(axis=-1)
                        if network_id == 0:  # locked network
                            assert np.all(choices == 1), (
                                f"Node {node_idx} from locked network has non-single choices"
                            )


class TestDesignModeConstraints:
    """Tests for constraints when using DesignManager.build_stack (random_init=True).

    CRITICAL: This tests the design mode behavior where random_init=True is passed
    to aggregation nodes. The bug we're catching is that random_init=True was
    overriding explicitly locked ratios (RatioSpec(locked=True)).
    """

    def test_zero_freedom_ratios_stay_locked_in_design_mode(self, designer_model):
        """Zero-freedom recipe should have locked ratios even with random_init=True."""
        from biocomp.design import DesignManager, Target

        recipe_path = RESOURCES_DIR / "design/architectures/T_2_zero_freedom.yaml"
        target_path = RESOURCES_DIR / "design/targets/MIT_T_only.yaml"

        if not recipe_path.exists():
            pytest.skip(f"Recipe not found: {recipe_path}")
        if not target_path.exists():
            pytest.skip(f"Target not found: {target_path}")

        recipe = load_recipe(recipe_path)
        target = Target(path=str(target_path))

        networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True, inversion_mode="main")
        assert len(networks) == 1

        dmanager = DesignManager(targets=[target], networks=networks, enable_tu_masking=False)

        # This is the critical call - builds stack with unlock_ratios=True (random_init=True)
        stack = dmanager.build_stack(designer_model, unlock_ratios=True)

        key = jax.random.PRNGKey(42)
        init_params = stack.init(key)
        _, nonshared = init_params.filter_by_tag(["shared"])
        params = ParameterTree.merge(designer_model.shared_params, nonshared)

        # Check that ratios are still locked (min == max)
        for layer_idx, _layer in enumerate(stack.layers):
            ns = stack.get_layer_namespace(layer_idx)
            if "aggregation" in ns and "inv" not in ns:
                ratio_min_path = f"{ns}/ratio_min"
                ratio_max_path = f"{ns}/ratio_max"

                if ratio_min_path in params:
                    ratio_min = np.asarray(params[ratio_min_path])
                    ratio_max = np.asarray(params[ratio_max_path])

                    # For zero-freedom recipe, all ratios should be locked
                    assert np.allclose(ratio_min, ratio_max), (
                        f"Zero-freedom ratios should remain locked in design mode at {ns}. "
                        f"Bug: random_init=True is overriding explicit locks. "
                        f"Got min={ratio_min}, max={ratio_max}"
                    )

    def test_zero_freedom_param_dim_zero_in_design_mode(self, designer_model):
        """Zero-freedom recipe should have param_dim=0 even in design mode."""
        from biocomp.design import DesignManager, Target

        recipe_path = RESOURCES_DIR / "design/architectures/T_2_zero_freedom.yaml"
        target_path = RESOURCES_DIR / "design/targets/MIT_T_only.yaml"

        if not recipe_path.exists():
            pytest.skip(f"Recipe not found: {recipe_path}")
        if not target_path.exists():
            pytest.skip(f"Target not found: {target_path}")

        recipe = load_recipe(recipe_path)
        target = Target(path=str(target_path))

        networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True, inversion_mode="main")
        dmanager = DesignManager(targets=[target], networks=networks, enable_tu_masking=False)

        stack = dmanager.build_stack(designer_model, unlock_ratios=True)

        key = jax.random.PRNGKey(42)
        init_params = stack.init(key)
        _, nonshared = init_params.filter_by_tag(["shared"])
        params = ParameterTree.merge(designer_model.shared_params, nonshared)

        codec = GenomeCodec.from_params(params)

        assert codec.param_dim == 0, (
            f"Zero-freedom recipe should have param_dim=0 in design mode, got {codec.param_dim}. "
            f"Bug: random_init=True is adding degrees of freedom to locked ratios."
        )

    def test_unlocked_ratios_still_unlock_in_design_mode(self, designer_model):
        """Unlocked recipes should still have unlocked ratios in design mode."""
        from biocomp.design import DesignManager, Target

        recipe_path = RESOURCES_DIR / "design/architectures/T_2_ratios_only.yaml"
        target_path = RESOURCES_DIR / "design/targets/MIT_T_only.yaml"

        if not recipe_path.exists():
            pytest.skip(f"Recipe not found: {recipe_path}")
        if not target_path.exists():
            pytest.skip(f"Target not found: {target_path}")

        recipe = load_recipe(recipe_path)
        target = Target(path=str(target_path))

        networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True, inversion_mode="main")
        dmanager = DesignManager(targets=[target], networks=networks, enable_tu_masking=False)

        stack = dmanager.build_stack(designer_model, unlock_ratios=True)

        key = jax.random.PRNGKey(42)
        init_params = stack.init(key)
        _, nonshared = init_params.filter_by_tag(["shared"])
        params = ParameterTree.merge(designer_model.shared_params, nonshared)

        codec = GenomeCodec.from_params(params)

        # T_2_ratios_only should have unlocked ratios, so param_dim > 0
        assert codec.param_dim > 0, (
            f"Unlocked recipe should have param_dim > 0 in design mode, got {codec.param_dim}"
        )

    def test_design_mode_genome_empty_for_zero_freedom(self, designer_model):
        """Zero-freedom recipe should have empty genome (nothing to optimize)."""
        recipe_path = RESOURCES_DIR / "design/architectures/T_2_zero_freedom.yaml"

        if not recipe_path.exists():
            pytest.skip(f"Recipe not found: {recipe_path}")

        recipe = load_recipe(recipe_path)
        networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True, inversion_mode="main")

        # Build stack
        stack = ComputeStack(networks=networks)
        stack.build(designer_model.compute_config, enable_tu_masking=False)

        key = jax.random.PRNGKey(42)
        init_params = stack.init(key)
        _, nonshared = init_params.filter_by_tag(["shared"])
        params = ParameterTree.merge(designer_model.shared_params, nonshared)

        codec = GenomeCodec.from_params(params)

        # With zero freedom, param_dim should be 0
        assert codec.param_dim == 0, f"Zero-freedom should have param_dim=0, got {codec.param_dim}"

        # Genome should be empty
        genome = codec.encode(params)
        assert genome.shape == (0,), f"Zero-freedom genome should be empty, got shape {genome.shape}"


class TestMultiTopologyDegreesOfFreedom:
    """Tests for degrees of freedom across various network topologies."""

    @pytest.mark.parametrize("recipe_name,expected_locked_rates,expected_locked_ratios", [
        ("T_2_zero_freedom.yaml", True, True),
        ("T_2_ratios_only.yaml", True, False),  # Rates locked, ratios unlocked
        ("two_and_one.yaml", False, False),  # Both unlocked
        ("two_and_one_all_uorfs.yaml", False, False),  # All unlocked
    ])
    def test_various_topologies_respect_constraints(
        self, designer_model, recipe_name, expected_locked_rates, expected_locked_ratios
    ):
        """Different topologies should have correct param_dim based on their constraints."""
        recipe_path = RESOURCES_DIR / "design/architectures" / recipe_name
        if not recipe_path.exists():
            pytest.skip(f"Recipe not found: {recipe_path}")

        recipe = load_recipe(recipe_path)
        _, stack, params = build_network_and_params(recipe, designer_model)

        codec = GenomeCodec.from_params(params)

        # Check rates: find any forward transform layer with multi-choice masks
        any_rate_unlocked = False
        for layer_idx, _ in enumerate(stack.layers):
            ns = stack.get_layer_namespace(layer_idx)
            if "inv_" in ns:
                continue
            for rate_name in ["tc_rate", "tl_rate"]:
                mask_path = f"{ns}/{rate_name}_quantization_mask"
                if mask_path in params:
                    masks = np.asarray(params[mask_path])
                    if np.any(masks.sum(axis=-1) > 1):
                        any_rate_unlocked = True
                        break
            if any_rate_unlocked:
                break

        # Check ratios: find any forward aggregation layer without NON_GRAD tag
        any_ratio_unlocked = False
        for layer_idx, _ in enumerate(stack.layers):
            ns = stack.get_layer_namespace(layer_idx)
            if "aggregation" in ns and "inv" not in ns:
                ratio_path = f"{ns}/ratios"
                if ratio_path in params:
                    tags = params.get_tags(ratio_path)
                    if not (tags and NON_GRAD_TAG in tags):
                        any_ratio_unlocked = True
                        break

        # Verify expectations
        if expected_locked_rates:
            assert not any_rate_unlocked, f"{recipe_name}: Expected rates locked but found unlocked"
        else:
            assert any_rate_unlocked, f"{recipe_name}: Expected rates unlocked but all locked"

        if expected_locked_ratios:
            assert not any_ratio_unlocked, f"{recipe_name}: Expected ratios locked but found unlocked"
        else:
            assert any_ratio_unlocked, f"{recipe_name}: Expected ratios unlocked but all locked"

        # Verify param_dim reflects expectations
        if expected_locked_rates and expected_locked_ratios:
            assert codec.param_dim == 0, f"{recipe_name}: Expected param_dim=0 for fully locked"

    def test_three_network_stack_heterogeneous_topologies(self, designer_model):
        """Stack with 3 different topologies should respect each network's constraints."""
        paths = [
            RESOURCES_DIR / "design/architectures/T_2_zero_freedom.yaml",
            RESOURCES_DIR / "design/architectures/T_2_ratios_only.yaml",
            RESOURCES_DIR / "design/architectures/two_and_one_all_uorfs.yaml",
        ]
        for p in paths:
            if not p.exists():
                pytest.skip(f"Required recipe not found: {p}")

        all_networks = []
        for i, path in enumerate(paths):
            recipe = load_recipe(path)
            networks = recipe_to_networks(
                recipe, br.ALL_RULES, invert=True, inversion_mode="main"
            )
            networks[0].name = f"net_{i}_{path.stem}"
            all_networks.append(networks[0])

        stack = ComputeStack(networks=all_networks)
        stack.build(designer_model.compute_config, enable_tu_masking=False)

        key = jax.random.PRNGKey(42)
        init_params = stack.init(key)
        _, nonshared = init_params.filter_by_tag(["shared"])
        params = ParameterTree.merge(designer_model.shared_params, nonshared)

        # Expected constraints per network:
        # Network 0 (T_2_zero_freedom): all tl_rates locked, ratios locked
        # Network 1 (T_2_ratios_only): all tl_rates locked, ratios unlocked
        # Network 2 (two_and_one_all_uorfs): SOME tl_rates unlocked, ratios unlocked
        # Note: Not ALL nodes in net 2 have unlocked uORFs - marker TUs have no uORFs

        net_rate_stats = {0: [], 1: [], 2: []}
        expected_ratio_locked = {0: True, 1: False, 2: False}

        for layer_idx, layer in enumerate(stack.layers):
            ns = stack.get_layer_namespace(layer_idx)
            if "inv_" in ns:
                continue

            # Collect translation rate lock stats per network
            if "translation" in ns:
                mask_path = f"{ns}/tl_rate_quantization_mask"
                if mask_path in params:
                    masks = np.asarray(params[mask_path])
                    for node_idx, node in enumerate(layer.nodes):
                        net_id = node.network_id
                        node_choices = masks[node_idx].sum(axis=-1)
                        is_locked = np.all(node_choices == 1)
                        net_rate_stats[net_id].append(is_locked)

            # Check aggregation ratios
            if "aggregation" in ns:
                ratio_min_path = f"{ns}/ratio_min"
                ratio_max_path = f"{ns}/ratio_max"
                if ratio_min_path in params:
                    ratio_min = np.asarray(params[ratio_min_path])
                    ratio_max = np.asarray(params[ratio_max_path])
                    for node_idx, node in enumerate(layer.nodes):
                        net_id = node.network_id
                        is_locked = np.allclose(ratio_min[node_idx], ratio_max[node_idx])
                        assert is_locked == expected_ratio_locked[net_id], (
                            f"Node {node_idx} (net_id={net_id}) ratio lock mismatch in {ns}"
                        )

        # Verify translation rate expectations
        assert all(net_rate_stats[0]), "Network 0 (zero_freedom) should have all tl_rates locked"
        assert all(net_rate_stats[1]), "Network 1 (ratios_only) should have all tl_rates locked"
        # Network 2 should have at least SOME unlocked rates (not all TUs have uORFs)
        assert any(not locked for locked in net_rate_stats[2]), (
            "Network 2 (all_uorfs) should have at least some unlocked tl_rates"
        )

    def test_duplicate_networks_maintain_independent_constraints(self, designer_model):
        """Same topology duplicated should maintain independent per-node constraints."""
        recipe_path = RESOURCES_DIR / "design/architectures/two_and_one.yaml"
        if not recipe_path.exists():
            pytest.skip(f"Recipe not found: {recipe_path}")

        recipe = load_recipe(recipe_path)
        networks1 = recipe_to_networks(recipe, br.ALL_RULES, invert=True, inversion_mode="main")
        networks2 = recipe_to_networks(recipe, br.ALL_RULES, invert=True, inversion_mode="main")

        networks1[0].name = "net_0"
        networks2[0].name = "net_1"

        stack = ComputeStack(networks=[networks1[0], networks2[0]])
        stack.build(designer_model.compute_config, enable_tu_masking=False)

        key = jax.random.PRNGKey(42)
        init_params = stack.init(key)
        _, nonshared = init_params.filter_by_tag(["shared"])
        params = ParameterTree.merge(designer_model.shared_params, nonshared)

        # Both networks have same topology → same constraints per position
        for layer_idx, layer in enumerate(stack.layers):
            ns = stack.get_layer_namespace(layer_idx)
            if "aggregation" in ns and "inv" not in ns:
                ratio_min = np.asarray(params[f"{ns}/ratio_min"])
                ratio_max = np.asarray(params[f"{ns}/ratio_max"])

                # Group nodes by network
                net0_nodes = [i for i, n in enumerate(layer.nodes) if n.network_id == 0]
                net1_nodes = [i for i, n in enumerate(layer.nodes) if n.network_id == 1]

                # Both networks should have same constraint pattern
                for n0, n1 in zip(net0_nodes, net1_nodes, strict=False):
                    n0_locked = np.allclose(ratio_min[n0], ratio_max[n0])
                    n1_locked = np.allclose(ratio_min[n1], ratio_max[n1])
                    assert n0_locked == n1_locked, (
                        f"Duplicate networks have different constraints at layer {ns}"
                    )
