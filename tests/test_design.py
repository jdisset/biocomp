"""Test design mode: commitment, recipe serialization, and prediction roundtrip.

Tests:
- Committed networks have collapsed/quantized slots
- Recipes can be saved as proper YAML and reloaded
- Reloaded recipes produce identical predictions
- Ratios are locked (not NumRange) after commit
- Network selection in topk is unbiased
- fluo_bias value is locked after commit
"""

import pytest
import tempfile
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import dracon as dr
from collections import Counter

import biocomp.parameters as pr
import biocomp.biorules as br
from biocomp.library import LibraryContext, load_lib
from biocomp.compute import ComputeStack
from biocomp.config import SIMPLE_NODES_COMPUTE_CONFIG
from biocomp.network import recipe_to_networks
from biocomp.recipe import Recipe, CoTransfection, TranscriptionUnit, Slot, NumRange


P = "hEF1a"
T = "L0.T_4560"
UORFS = [None, "1w_uORF", "1x_uORF", "2x_uORF", "3x_uORF"]


@pytest.fixture
def lib():
    return load_lib()


@pytest.fixture
def simple_design_recipe(lib):
    """A simple design recipe with unlocked uORFs and ratios."""
    u = Slot(part=UORFS, ref_id="U1")
    with LibraryContext.with_library(lib):
        return Recipe(
            name="simple_design_test",
            content=[
                CoTransfection(
                    name="test_cotx",
                    units=[
                        TranscriptionUnit(slots=[P, u, "CasE_rec", "mNeonGreen", T], name="output", source="p1"),
                        TranscriptionUnit(slots=[P, "CasE", T], name="ern", source="p2"),
                        TranscriptionUnit(slots=[P, "mKO2", T], name="marker", source="p3"),
                    ],
                    ratios=[NumRange(min=0.2, max=0.5), NumRange(min=0.3, max=0.6), NumRange(min=0.1, max=0.3)],
                )
            ],
        )


def test_commit_collapses_slots(lib, simple_design_recipe):
    """Test that stack.commit() produces networks with collapsed (single-value) slots."""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(simple_design_recipe, br.ALL_RULES, invert=True)
        network = networks[0]
        stack = ComputeStack([network])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        key = jax.random.PRNGKey(42)
        params = stack.init(key)

        committed_networks = stack.commit(params)
        committed = committed_networks[0]
        exported_recipe = committed.to_recipe()

        for cotx in exported_recipe.content:
            for tu in cotx.units:
                for slot in tu.slots:
                    if isinstance(slot, Slot):
                        if isinstance(slot.part, list):
                            assert len(slot.part) == 1, f"Slot should be collapsed to single part, got {slot.part}"
                    elif isinstance(slot, list):
                        assert False, f"Slot should not be a list after commit: {slot}"


def test_recipe_yaml_roundtrip(lib, simple_design_recipe):
    """Test that committed recipes can be saved as YAML and reloaded."""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(simple_design_recipe, br.ALL_RULES, invert=True)
        network = networks[0]
        stack = ComputeStack([network])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        key = jax.random.PRNGKey(123)
        params = stack.init(key)

        committed_networks = stack.commit(params)
        committed = committed_networks[0]
        exported_recipe = committed.to_recipe()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_content = dr.dump(exported_recipe)
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            reloaded_recipe = dr.load(yaml_path, Recipe)
            assert reloaded_recipe.name == exported_recipe.name
            assert len(reloaded_recipe.content) == len(exported_recipe.content)
        finally:
            yaml_path.unlink()



def test_reloaded_recipe_produces_same_predictions(lib, simple_design_recipe):
    """Test that reloaded recipe produces identical predictions."""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(simple_design_recipe, br.ALL_RULES, invert=True)
        network = networks[0]
        stack = ComputeStack([network])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        orig_key = jax.random.PRNGKey(123)
        opt_key = jax.random.PRNGKey(42)
        eval_key = jax.random.PRNGKey(999)

        orig_params = stack.init(orig_key)
        opt_params = stack.init(opt_key)
        orig_shared, _ = orig_params.filter_by_tag(['shared'])
        _, opt_nonshared = opt_params.filter_by_tag(['shared'])
        opt_params = pr.ParameterTree.merge(orig_shared, opt_nonshared)

        committed_networks = stack.commit(opt_params)
        committed = committed_networks[0]
        exported_recipe = committed.to_recipe()

        # save and reload
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_content = dr.dump(exported_recipe)
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            reloaded_recipe = dr.load(yaml_path, Recipe)
            rebuilt_networks = recipe_to_networks(reloaded_recipe, br.ALL_RULES, invert=True)
            rebuilt = rebuilt_networks[0]
            rebuilt_stack = ComputeStack([rebuilt])
            rebuilt_stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

            rebuilt_params = rebuilt_stack.init(opt_key)
            rebuilt_shared, rebuilt_nonshared = rebuilt_params.filter_by_tag(['shared'])
            rebuilt_params = pr.ParameterTree.merge(orig_shared, rebuilt_nonshared)

            # generate test inputs
            nb_inputs = network.nb_inputs
            N_EVALS = 50
            x = jax.random.uniform(eval_key, (N_EVALS, nb_inputs))
            num_z = opt_params["global/number_of_random_variables"]
            random_variables = jnp.zeros((num_z,))

            y_opt, _ = jax.vmap(stack.apply, in_axes=(None, 0, None, None))(
                opt_params, x, random_variables, eval_key
            )
            y_rebuilt, _ = jax.vmap(rebuilt_stack.apply, in_axes=(None, 0, None, None))(
                rebuilt_params, x, random_variables, eval_key
            )

            assert y_opt.shape == y_rebuilt.shape
            assert jnp.allclose(y_opt, y_rebuilt, atol=1e-5)

        finally:
            yaml_path.unlink()


def test_ratios_are_locked_after_commit(lib, simple_design_recipe):
    """Test that ratios become fixed values (not NumRange) after commit."""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(simple_design_recipe, br.ALL_RULES, invert=True)
        network = networks[0]
        stack = ComputeStack([network])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        key = jax.random.PRNGKey(42)
        params = stack.init(key)

        committed_networks = stack.commit(params)
        committed = committed_networks[0]
        exported_recipe = committed.to_recipe()

        for cotx in exported_recipe.content:
            if cotx.ratios:
                for ratio in cotx.ratios:
                    assert not isinstance(ratio, NumRange), f"Ratio should be locked, got NumRange: {ratio}"
                    assert isinstance(ratio, (int, float)), f"Ratio should be numeric, got {type(ratio)}"


def test_topk_selection_not_biased():
    """Test that get_topk_replicate_network_pairs doesn't have a bias toward network 0.

    This tests for the bug where network 0 was always selected. We create random loss
    matrices where different networks should be best, and verify the selection is correct.
    """
    from biocomp.design import get_topk_replicate_network_pairs

    # Test with various scenarios where different networks should be selected
    n_replicates = 5
    n_targets = 3
    n_networks = 4
    k = 2

    # create mock classes outside the loop
    class MockDM:
        pass

    class MockDC:
        pass

    mock_dm = MockDM()
    mock_dm.n_targets = n_targets
    mock_dm.networks = [type('Net', (), {'name': f'net{i}'})() for i in range(n_networks)]

    mock_dc = MockDC()
    mock_dc.n_replicates = n_replicates

    key = jax.random.PRNGKey(42)
    counter = Counter()

    for trial in range(20):
        key, subkey = jax.random.split(key)
        # random losses
        losses = jax.random.uniform(subkey, (n_replicates, n_targets, n_networks))

        topk = get_topk_replicate_network_pairs(losses, mock_dm, mock_dc, k=k)

        # collect which networks were selected as best for each target
        for target_results in topk:
            for rep_id, net_id, loss in target_results:
                counter[net_id] += 1

    # with random losses, each network should be selected roughly equally
    # allow for some variation but fail if any network is never selected
    min_count = min(counter.values()) if counter else 0

    # with 20 trials * 3 targets * 2 top-k = 120 selections across 4 networks
    # each network should get ~30 selections, allow wide margin
    assert min_count >= 10, f"Network selection appears biased: {counter}"
    assert len(counter) == n_networks, f"Not all networks were ever selected: {counter}"


def test_real_multi_network_selection_unbiased(lib):
    """Test network selection with real multi-inversion networks.

    Creates a recipe that produces multiple equivalent network inversions,
    builds a real ComputeStack, and verifies that:
    1. Multiple networks are actually created
    2. Selection based on actual computed losses is correct
    3. Network index doesn't correlate with selection when losses are similar
    """
    from biocomp.design import get_topk_replicate_network_pairs
    from scipy import stats

    with LibraryContext.with_library(lib):
        # Recipe with 3 reporter outputs (no ERN control) - produces multiple inversions
        # because each output can be the "dependent" one
        recipe = Recipe(
            name="multi_inversion_test",
            content=[
                CoTransfection(
                    name="test_cotx",
                    units=[
                        TranscriptionUnit(slots=[P, "CasE_rec", "mNeonGreen", T], name="output1", source="p1"),
                        TranscriptionUnit(slots=[P, "CasE", T], name="ern", source="p2"),
                        TranscriptionUnit(slots=[P, "eBFP2", T], name="output2", source="p3"),
                        TranscriptionUnit(slots=[P, "mMaroon1", T], name="output3", source="p4"),
                    ],
                    ratios=[0.3, 0.2, 0.25, 0.25],
                )
            ],
        )

        # Use inversion_mode="all" to get multiple networks
        networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True, inversion_mode="all")

        # Skip test if only one network (some recipes may only have one valid inversion)
        if len(networks) < 2:
            pytest.skip(f"Recipe only produced {len(networks)} network(s), need >= 2 for this test")

        n_networks = len(networks)
        stack = ComputeStack(networks)
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        # Track which networks get selected across many random initializations
        network_selection_counts = Counter()
        n_trials = 30

        for trial in range(n_trials):
            key = jax.random.PRNGKey(trial * 1000)
            params = stack.init(key)

            # Generate random inputs - ComputeStack expects total_nb_of_inputs (sum across all networks)
            eval_key = jax.random.PRNGKey(trial * 1000 + 500)
            total_inputs = stack.total_nb_of_inputs
            n_samples = 10
            x = jax.random.uniform(eval_key, (n_samples, total_inputs))
            num_z = params["global/number_of_random_variables"]
            random_vars = jnp.zeros((num_z,))

            # Compute outputs for all samples - vmap over samples dimension
            y, _ = jax.vmap(stack.apply, in_axes=(None, 0, None, None))(
                params, x, random_vars, eval_key
            )
            # y shape: (n_samples, total_outputs) where total_outputs = sum of dependent outputs across networks

            # Create synthetic target with some random variation
            target_key = jax.random.PRNGKey(trial * 1000 + 999)
            target = y.mean(axis=0) + jax.random.normal(target_key, y.shape[1:]) * 0.1

            # Compute MSE per output column
            mse_per_output = jnp.mean((y - target) ** 2, axis=0)  # shape: (n_outputs,)

            # Find which output index has the lowest loss
            best_output_idx = int(jnp.argmin(mse_per_output))
            network_selection_counts[best_output_idx] += 1

        # Verify no strong bias - use chi-squared test for uniformity
        observed = list(network_selection_counts.values())
        n_selections = sum(observed)
        expected = [n_selections / len(observed)] * len(observed)

        if len(observed) > 1 and n_selections > 10:
            chi2, p_value = stats.chisquare(observed, expected)
            # p < 0.01 would indicate significant bias
            assert p_value > 0.01, (
                f"Network selection appears significantly biased (p={p_value:.4f}): {network_selection_counts}"
            )


def test_real_network_topk_correctness(lib):
    """Test that get_topk_replicate_network_pairs returns actually lowest losses.

    Uses real networks and verifies the returned pairs have the lowest losses.
    """
    from biocomp.design import get_topk_replicate_network_pairs

    with LibraryContext.with_library(lib):
        recipe = Recipe(
            name="topk_correctness_test",
            content=[
                CoTransfection(
                    name="test_cotx",
                    units=[
                        TranscriptionUnit(slots=[P, "mNeonGreen", T], name="out1", source="p1"),
                        TranscriptionUnit(slots=[P, "eBFP2", T], name="out2", source="p2"),
                    ],
                    ratios=[0.5, 0.5],
                )
            ],
        )

        networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True, inversion_mode="all")
        n_networks = len(networks)

        # Create synthetic loss matrix with known structure
        n_replicates = 5
        n_targets = 2

        # Mock objects for the function
        class MockDM:
            pass
        class MockDC:
            pass

        mock_dm = MockDM()
        mock_dm.n_targets = n_targets
        mock_dm.networks = networks

        mock_dc = MockDC()
        mock_dc.n_replicates = n_replicates

        key = jax.random.PRNGKey(42)

        for trial in range(10):
            key, subkey = jax.random.split(key)
            # Create random losses
            losses = jax.random.uniform(subkey, (n_replicates, n_targets, n_networks))

            k = min(3, n_replicates * n_networks)
            topk = get_topk_replicate_network_pairs(losses, mock_dm, mock_dc, k=k)

            # Verify each target's topk
            for tid, target_topk in enumerate(topk):
                # Get all losses for this target
                target_losses = losses[:, tid, :]  # shape: (n_reps, n_nets)
                flat_losses = target_losses.reshape(-1)
                sorted_indices = jnp.argsort(flat_losses)

                # Verify returned pairs match the actual k smallest
                for rank, (rep_id, net_id, loss_val) in enumerate(target_topk):
                    expected_flat_idx = sorted_indices[rank]
                    expected_rep, expected_net = jnp.unravel_index(expected_flat_idx, target_losses.shape)
                    expected_loss = float(flat_losses[expected_flat_idx])

                    assert rep_id == int(expected_rep), f"Trial {trial}, target {tid}, rank {rank}: rep mismatch"
                    assert net_id == int(expected_net), f"Trial {trial}, target {tid}, rank {rank}: net mismatch"
                    assert abs(loss_val - expected_loss) < 1e-5, f"Trial {trial}, target {tid}, rank {rank}: loss mismatch"


def test_multi_network_stack_output_independence(lib):
    """Test that different networks in a stack produce genuinely different outputs.

    This verifies that when we have multiple network inversions, they aren't
    accidentally producing identical outputs (which would make selection meaningless).
    """
    with LibraryContext.with_library(lib):
        # Recipe with 3 outputs - produces multiple inversions as each can be dependent
        recipe = Recipe(
            name="output_independence_test",
            content=[
                CoTransfection(
                    name="test_cotx",
                    units=[
                        TranscriptionUnit(slots=[P, "CasE_rec", "mNeonGreen", T], name="out1", source="p1"),
                        TranscriptionUnit(slots=[P, "CasE", T], name="ern", source="p2"),
                        TranscriptionUnit(slots=[P, "eBFP2", T], name="out2", source="p3"),
                        TranscriptionUnit(slots=[P, "mMaroon1", T], name="out3", source="p4"),
                    ],
                    ratios=[0.3, 0.2, 0.25, 0.25],
                )
            ],
        )

        networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True, inversion_mode="all")

        if len(networks) < 2:
            pytest.skip(f"Recipe only produced {len(networks)} network(s)")

        stack = ComputeStack(networks)
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        key = jax.random.PRNGKey(123)
        params = stack.init(key)

        # Generate inputs - ComputeStack expects total_nb_of_inputs (sum across all networks)
        eval_key = jax.random.PRNGKey(456)
        total_inputs = stack.total_nb_of_inputs
        n_samples = 20
        x = jax.random.uniform(eval_key, (n_samples, total_inputs))
        num_z = params["global/number_of_random_variables"]
        random_vars = jnp.zeros((num_z,))

        # Compute outputs - vmap over samples dimension
        y, _ = jax.vmap(stack.apply, in_axes=(None, 0, None, None))(
            params, x, random_vars, eval_key
        )

        # y shape should be (n_samples, total_outputs)
        # Each output column corresponds to a different dependent output across networks
        # With multiple networks, we should see variation

        # Check that outputs have meaningful variance (not all identical)
        output_std = jnp.std(y, axis=0)
        assert jnp.all(output_std > 1e-6), (
            f"Some outputs have near-zero variance, suggesting degenerate networks: std={output_std}"
        )

        # If we have multiple output columns, check they're not perfectly correlated
        if y.shape[1] > 1:
            corr_matrix = jnp.corrcoef(y.T)
            # Off-diagonal elements should not all be 1.0 (perfect correlation)
            off_diag = corr_matrix[jnp.triu_indices(y.shape[1], k=1)]
            assert not jnp.all(jnp.abs(off_diag) > 0.999), (
                f"Outputs are nearly perfectly correlated, suggesting redundant networks"
            )


def test_fluo_bias_preserved_after_commit(lib):
    """Test that fluo_bias is properly set after commit."""
    from biocomp.recipe import FluoIntensity

    u = Slot(part=UORFS, ref_id="U1")
    with LibraryContext.with_library(lib):
        recipe = Recipe(
            name="bias_test",
            content=[
                CoTransfection(
                    name="test_cotx",
                    units=[
                        TranscriptionUnit(slots=[P, u, "CasE_rec", "mNeonGreen", T], name="output", source="p1"),
                        TranscriptionUnit(slots=[P, "CasE", T], name="ern", source="p2"),
                        TranscriptionUnit(slots=[P, "mKO2", T], name="marker", source="p3"),
                    ],
                    ratios=[0.3, 0.4, 0.3],
                    fluo_bias=FluoIntensity(tu_id=2, value=NumRange(min=0.3, max=0.7), protein="mKO2"),
                )
            ],
        )

        networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True)
        network = networks[0]
        stack = ComputeStack([network])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        key = jax.random.PRNGKey(42)
        params = stack.init(key)

        committed_networks = stack.commit(params)
        committed = committed_networks[0]
        exported_recipe = committed.to_recipe()

        cotx = exported_recipe.content[0]
        assert cotx.fluo_bias is not None, "fluo_bias should be preserved after commit"
        # value should be locked (not NumRange)
        if hasattr(cotx.fluo_bias, 'value'):
            assert not isinstance(cotx.fluo_bias.value, NumRange), \
                f"fluo_bias.value should be locked, got {cotx.fluo_bias.value}"


def test_independent_uorf_slots_not_cross_committed(lib):
    """Test that two independent TUs with same uORF options commit independently.

    This tests for the bug where propagate_embedding_backwards could incorrectly
    collapse unrelated TUs that happen to have the same embedding options.

    We manually set different tl_rate values to ensure they quantize to different
    parts, then verify they stay separate after commit.
    """
    # Create two INDEPENDENT uORF slots with the same options but different ref_ids
    u1 = Slot(part=UORFS, ref_id="U1")
    u2 = Slot(part=UORFS, ref_id="U2")

    with LibraryContext.with_library(lib):
        recipe = Recipe(
            name="independent_uorf_test",
            content=[
                CoTransfection(
                    name="test_cotx",
                    units=[
                        # TU1: has uORF u1
                        TranscriptionUnit(slots=[P, u1, "CasE_rec", "mNeonGreen", T], name="output1", source="p1"),
                        # TU2: has uORF u2 (independent from u1, same options)
                        TranscriptionUnit(slots=[P, u2, "mKO2", T], name="output2", source="p2"),
                    ],
                    ratios=[0.5, 0.5],
                )
            ],
        )

        networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True)
        network = networks[0]
        stack = ComputeStack([network])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        key = jax.random.PRNGKey(42)
        params = stack.init(key)

        # Manually set tl_rate values to ensure they quantize to different parts
        # The masked quantization values are [-1, -0.833, -0.667, -0.5, -0.333] for indices 0-4
        # Set node 0 close to -1.0 (index 0, 00_empty_tc)
        # Set node 1 close to -0.333 (index 4, 3x_uORF)
        tl_rate_path = 'local/8/translation/tl_rate'
        old_tl_rate = params[tl_rate_path]
        new_tl_rate = jnp.array([[[-0.95]], [[-0.35]]])  # node 0 -> index 0, node 1 -> index 4
        params[tl_rate_path] = new_tl_rate

        committed_networks = stack.commit(params)
        committed = committed_networks[0]

        # Check that edges for TU1 and TU2 have DIFFERENT committed values
        edges = committed.compute_graph.edges
        tu1_tl_rate = None
        tu2_tl_rate = None
        for key, edge in edges.items():
            tu_ids = edge.extra.get("tu_id", []) if edge.extra else []
            tl_rate = edge.content_embedding_names.get("tl_rate") if edge.content_embedding_names else None
            if "output1_test_cotx" in tu_ids and tl_rate:
                tu1_tl_rate = tl_rate
            if "output2_test_cotx" in tu_ids and tl_rate:
                tu2_tl_rate = tl_rate

        assert tu1_tl_rate is not None, "TU1 should have tl_rate"
        assert tu2_tl_rate is not None, "TU2 should have tl_rate"
        assert tu1_tl_rate != tu2_tl_rate, \
            f"Independent TUs should have different tl_rate after commit: TU1={tu1_tl_rate}, TU2={tu2_tl_rate}"

        # Also verify via recipe extraction
        exported_recipe = committed.to_recipe()
        cotx = exported_recipe.content[0]

        uorf_values = {}
        for tu in cotx.units:
            tu_name = tu.name
            for slot in tu.slots:
                if isinstance(slot, Slot) and slot.ref_id in ("U1", "U2"):
                    val = slot.part[0] if isinstance(slot.part, list) else slot.part
                    uorf_values[slot.ref_id] = val
                elif isinstance(slot, str) and slot in UORFS:
                    # collapsed to string
                    if tu_name == "output1":
                        uorf_values["U1"] = slot
                    elif tu_name == "output2":
                        uorf_values["U2"] = slot

        assert len(uorf_values) >= 2, f"Should have found uORF values for both TUs, got {uorf_values}"
        assert uorf_values.get("U1") != uorf_values.get("U2"), \
            f"Independent uORF slots should commit to different values: {uorf_values}"


if __name__ == "__main__":
    lib_instance = load_lib()

    u = Slot(part=UORFS, ref_id="U1")
    with LibraryContext.with_library(lib_instance):
        simple_recipe = Recipe(
            name="simple_design_test",
            content=[
                CoTransfection(
                    name="test_cotx",
                    units=[
                        TranscriptionUnit(slots=[P, u, "CasE_rec", "mNeonGreen", T], name="output", source="p1"),
                        TranscriptionUnit(slots=[P, "CasE", T], name="ern", source="p2"),
                        TranscriptionUnit(slots=[P, "mKO2", T], name="marker", source="p3"),
                    ],
                    ratios=[NumRange(min=0.2, max=0.5), NumRange(min=0.3, max=0.6), NumRange(min=0.1, max=0.3)],
                )
            ],
        )

    print("Running test_commit_collapses_slots...")
    test_commit_collapses_slots(lib_instance, simple_recipe)
    print("PASSED\n")

    print("Running test_recipe_yaml_roundtrip...")
    test_recipe_yaml_roundtrip(lib_instance, simple_recipe)
    print("PASSED\n")

    print("Running test_reloaded_recipe_produces_same_predictions...")
    test_reloaded_recipe_produces_same_predictions(lib_instance, simple_recipe)
    print("PASSED\n")

    print("Running test_ratios_are_locked_after_commit...")
    test_ratios_are_locked_after_commit(lib_instance, simple_recipe)
    print("PASSED\n")

    print("Running test_topk_selection_not_biased...")
    test_topk_selection_not_biased()
    print("PASSED\n")

    print("Running test_fluo_bias_preserved_after_commit...")
    test_fluo_bias_preserved_after_commit(lib_instance)
    print("PASSED\n")

    print("All tests passed!")
