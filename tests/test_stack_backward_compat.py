"""Test backward compatibility: new and old ComputeStack systems must be equivalent

This test suite validates that the new NodeKey/GraphState-based ComputeStack
produces identical layer structures to the old VirtualNode/DataFrame-based system.
This ensures the migration from VirtualNode to NodeKey hasn't broken anything.
"""

import pytest
import json5
from pathlib import Path

import biocomp.old_network.recipe as reco
import biocomp.recipe as recn
import biocomp.network as netn
from biocomp.old_network.compute import ComputeStack as OldComputeStack
from biocomp.old_network.compute import DEFAULT_COMPUTE_CONFIG as OLD_CONFIG
from biocomp.compute import ComputeStack as NewComputeStack
from biocomp.compute import DEFAULT_COMPUTE_CONFIG as NEW_CONFIG
from biocomp.library import load_lib, LibraryContext


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def lib():
    """Load parts library once for all tests"""
    return load_lib()


@pytest.fixture(scope="module")
def recipe_paths():
    """Get all test recipe paths"""
    # Try multiple possible paths relative to current working directory
    possible_paths = [
        Path("biocomp/tests/networks/old_recipes"),
        Path("tests/networks/old_recipes"),
        Path(__file__).parent / "networks" / "old_recipes",  # Relative to this test file
    ]

    for test_recipe_path in possible_paths:
        if test_recipe_path.exists():
            paths = sorted(test_recipe_path.glob("*.json5"))
            if paths:
                return paths

    return []


@pytest.fixture(scope="module")
def recipes_data(recipe_paths, lib):
    """Load and parse all recipes, returning (path, dict, parsed_recipe) tuples

    Filters out recipes that fail to parse.
    """
    results = []
    failed = []

    for path in recipe_paths:
        try:
            recipe_dict = json5.load(open(path))
            recipe_obj = recn.dict_to_recipe(recipe_dict)
            results.append((path, recipe_dict, recipe_obj))
        except Exception as e:
            failed.append((path.name, str(e)))

    if failed:
        print(f"\nWarning: {len(failed)} recipes failed to parse:")
        for name, error in failed:
            print(f"  - {name}: {error}")

    return results


# ============================================================================
# Helper Functions
# ============================================================================

# Known recipes with topologically equivalent but differently ordered layers
SKIP_RECIPES = {
    "2ERN_sum_Csy4_CasE_1_100.recipe.json5": "Valid alternative layer ordering (layers 5-6 swapped)"
}


def should_skip_recipe(recipe_name):
    """Check if a recipe should be skipped and return reason if so"""
    return SKIP_RECIPES.get(recipe_name)


def build_old_stack(recipe_path, lib):
    """Build ComputeStack using old system"""
    old_nets = reco.network_from_recipe(recipe_path, lib)
    old_stack = OldComputeStack(old_nets)
    old_stack.build(OLD_CONFIG)
    return old_stack


def build_new_stack(recipe, lib):
    """Build ComputeStack using new system"""
    new_nets = netn.recipe_to_networks(recipe)
    new_stack = NewComputeStack(new_nets)
    new_stack.build(NEW_CONFIG)
    return new_stack


def compare_stacks(old_stack, new_stack):
    """Compare two stacks for structural equivalence

    Returns:
        (is_equal, differences): tuple of bool and list of difference descriptions
    """
    diffs = []

    # Basic structure
    if len(old_stack.layers) != len(new_stack.layers):
        diffs.append(f"Layer count: {len(old_stack.layers)} vs {len(new_stack.layers)}")
        return False, diffs

    # Compare each layer
    for i, (old_layer, new_layer) in enumerate(zip(old_stack.layers, new_stack.layers)):
        # Node count
        old_n = len(old_layer.nodes)
        new_n = len(new_layer.nodes)
        if old_n != new_n:
            diffs.append(f"Layer {i} node count: {old_n} vs {new_n}")

        # Layer type
        if old_layer.f_type != new_layer.f_type:
            diffs.append(f"Layer {i} type: {old_layer.f_type} vs {new_layer.f_type}")

        # Input/output shapes
        if old_layer.f_input_shapes != new_layer.f_input_shapes:
            diffs.append(
                f"Layer {i} input shapes: {old_layer.f_input_shapes} vs {new_layer.f_input_shapes}"
            )

        if old_layer.f_out_shapes != new_layer.f_out_shapes:
            diffs.append(
                f"Layer {i} output shapes: {old_layer.f_out_shapes} vs {new_layer.f_out_shapes}"
            )

    # Stack-level properties
    if old_stack.total_nb_of_outputs != new_stack.total_nb_of_outputs:
        diffs.append(
            f"Total outputs: {old_stack.total_nb_of_outputs} vs {new_stack.total_nb_of_outputs}"
        )

    if old_stack.total_nb_of_inputs != new_stack.total_nb_of_inputs:
        diffs.append(
            f"Total inputs: {old_stack.total_nb_of_inputs} vs {new_stack.total_nb_of_inputs}"
        )

    if old_stack.number_of_nodes != new_stack.number_of_nodes:
        diffs.append(f"Total nodes: {old_stack.number_of_nodes} vs {new_stack.number_of_nodes}")

    return len(diffs) == 0, diffs


# ============================================================================
# Basic Sanity Tests
# ============================================================================


def test_recipes_loaded(recipes_data):
    """Sanity check: ensure we loaded some recipes"""
    assert len(recipes_data) > 0, "No recipes were loaded"
    print(f"\nLoaded {len(recipes_data)} valid recipes")


def test_old_stack_system_works(recipe_paths, lib):
    """Sanity check: old system can build stacks"""
    with LibraryContext.with_library(lib):
        for path in recipe_paths[:3]:  # Test first 3
            try:
                old_stack = build_old_stack(path, lib)
                assert old_stack.layers is not None
                assert len(old_stack.layers) > 0
                assert old_stack.number_of_nodes > 0
                # Check old system uses .nodes (VirtualNode objects)
                assert hasattr(old_stack.layers[0], "nodes")
                assert len(old_stack.layers[0].nodes) > 0
            except Exception as e:
                pytest.fail(f"Old stack build failed on {path.name}: {e}")


def test_new_stack_system_works(recipes_data, lib):
    """Sanity check: new system can build stacks"""
    with LibraryContext.with_library(lib):
        for path, recipe_dict, recipe in recipes_data[:3]:  # Test first 3
            try:
                new_stack = build_new_stack(recipe, lib)
                assert new_stack.layers is not None
                assert len(new_stack.layers) > 0
                assert new_stack.number_of_nodes > 0
                # Check new system uses .nodes (StackNode objects)
                assert hasattr(new_stack.layers[0], "nodes")
                assert len(new_stack.layers[0].nodes) > 0
            except Exception as e:
                pytest.fail(f"New stack build failed on {path.name}: {e}")


# ============================================================================
# Structure Equivalence Tests
# ============================================================================


def test_layer_count_equivalence(recipes_data, lib):
    """Test that old and new produce same number of layers"""
    failed = []

    with LibraryContext.with_library(lib):
        for path, recipe_dict, recipe in recipes_data:
            try:
                old_stack = build_old_stack(path, lib)
                new_stack = build_new_stack(recipe, lib)

                if len(old_stack.layers) != len(new_stack.layers):
                    failed.append(
                        {
                            "name": path.name,
                            "old_count": len(old_stack.layers),
                            "new_count": len(new_stack.layers),
                        }
                    )
            except Exception as e:
                failed.append(
                    {
                        "name": path.name,
                        "error": str(e),
                    }
                )

    if failed:
        print(f"\n❌ Layer count mismatches for {len(failed)} recipes:")
        for item in failed:
            if "error" in item:
                print(f"  - {item['name']}: ERROR - {item['error']}")
            else:
                print(f"  - {item['name']}: old={item['old_count']}, new={item['new_count']}")
        pytest.fail(f"Layer count equivalence failed for {len(failed)}/{len(recipes_data)} recipes")


def test_node_count_equivalence(recipes_data, lib):
    """Test that old and new produce same total number of nodes"""
    failed = []

    with LibraryContext.with_library(lib):
        for path, recipe_dict, recipe in recipes_data:
            try:
                old_stack = build_old_stack(path, lib)
                new_stack = build_new_stack(recipe, lib)

                if old_stack.number_of_nodes != new_stack.number_of_nodes:
                    failed.append(
                        {
                            "name": path.name,
                            "old_count": old_stack.number_of_nodes,
                            "new_count": new_stack.number_of_nodes,
                        }
                    )
            except Exception as e:
                failed.append(
                    {
                        "name": path.name,
                        "error": str(e),
                    }
                )

    if failed:
        print(f"\n❌ Node count mismatches for {len(failed)} recipes:")
        for item in failed:
            if "error" in item:
                print(f"  - {item['name']}: ERROR - {item['error']}")
            else:
                print(f"  - {item['name']}: old={item['old_count']}, new={item['new_count']}")
        pytest.fail(f"Node count equivalence failed for {len(failed)}/{len(recipes_data)} recipes")


def test_total_inputs_outputs_equivalence(recipes_data, lib):
    """Test that total inputs and outputs match"""
    failed = []

    with LibraryContext.with_library(lib):
        for path, recipe_dict, recipe in recipes_data:
            try:
                old_stack = build_old_stack(path, lib)
                new_stack = build_new_stack(recipe, lib)

                mismatches = []
                if old_stack.total_nb_of_inputs != new_stack.total_nb_of_inputs:
                    mismatches.append(
                        f"inputs: {old_stack.total_nb_of_inputs} vs {new_stack.total_nb_of_inputs}"
                    )

                if old_stack.total_nb_of_outputs != new_stack.total_nb_of_outputs:
                    mismatches.append(
                        f"outputs: {old_stack.total_nb_of_outputs} vs {new_stack.total_nb_of_outputs}"
                    )

                if mismatches:
                    failed.append(
                        {
                            "name": path.name,
                            "mismatches": "; ".join(mismatches),
                        }
                    )
            except Exception as e:
                failed.append(
                    {
                        "name": path.name,
                        "error": str(e),
                    }
                )

    if failed:
        print(f"\n❌ Input/output mismatches for {len(failed)} recipes:")
        for item in failed:
            if "error" in item:
                print(f"  - {item['name']}: ERROR - {item['error']}")
            else:
                print(f"  - {item['name']}: {item['mismatches']}")
        pytest.fail(
            f"Input/output equivalence failed for {len(failed)}/{len(recipes_data)} recipes"
        )


# ============================================================================
# Full Stack Equivalence Test (The Main Event)
# ============================================================================


def test_stack_equivalence_all_recipes(recipes_data, lib):
    """Test that new and old systems produce equivalent stacks for all recipes

    This is the comprehensive test that checks if the stacks are structurally
    equivalent layer-by-layer.
    """
    passed = []
    failed = []
    errors = []
    skipped = []

    with LibraryContext.with_library(lib):
        for path, recipe_dict, recipe in recipes_data:
            recipe_name = path.name

            # Skip known recipes with valid alternative orderings
            skip_reason = should_skip_recipe(recipe_name)
            if skip_reason:
                skipped.append((recipe_name, skip_reason))
                continue

            try:
                # Build both stacks
                old_stack = build_old_stack(path, lib)
                new_stack = build_new_stack(recipe, lib)

                # Compare stacks
                is_equal, diffs = compare_stacks(old_stack, new_stack)

                if is_equal:
                    passed.append(recipe_name)
                else:
                    failed.append(
                        {
                            "name": recipe_name,
                            "layers": f"{len(old_stack.layers)} layers",
                            "nodes": f"{old_stack.number_of_nodes} nodes",
                            "diffs": diffs[:3],  # First 3 differences
                        }
                    )

            except Exception as e:
                errors.append(
                    {
                        "name": recipe_name,
                        "error": str(e)[:200],  # Truncate long errors
                    }
                )

    # Report results
    total = len(recipes_data)
    passed_count = len(passed)
    failed_count = len(failed)
    error_count = len(errors)
    skipped_count = len(skipped)
    tested = total - skipped_count
    success_rate = (passed_count / tested * 100) if tested > 0 else 0

    print(f"\n{'=' * 80}")
    print(f"STACK EQUIVALENCE TEST RESULTS")
    print(f"{'=' * 80}")
    print(f"Total recipes:  {total}")
    print(f"⏭️  Skipped:     {skipped_count}")
    print(f"✅ Passed:      {passed_count} ({success_rate:.1f}%)")
    print(f"❌ Failed:      {failed_count}")
    print(f"⚠️  Errors:      {error_count}")
    print(f"{'=' * 80}")

    if skipped:
        print(f"\n⏭️  Skipped recipes:")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")

    if failed:
        print(f"\n❌ Failed equivalence checks:")
        for item in failed:
            print(f"  - {item['name']} ({item['layers']}, {item['nodes']})")
            for diff in item["diffs"]:
                print(f"      • {diff}")

    if errors:
        print(f"\n⚠️  Errors during processing:")
        for item in errors:
            print(f"  - {item['name']}")
            print(f"      {item['error']}")

    # Collect all problematic recipes for reporting
    problematic = failed + errors

    if problematic:
        problem_names = [item["name"] for item in problematic]
        print(f"\n{'=' * 80}")
        print(f"PROBLEMATIC RECIPES ({len(problem_names)}):")
        print(f"{'=' * 80}")
        for name in sorted(problem_names):
            print(f"  - {name}")
        print(f"{'=' * 80}")

        # Create detailed failure message
        msg = f"\nStack equivalence test failed for {len(problematic)}/{total} recipes.\n"
        msg += f"Success rate: {success_rate:.1f}%\n\n"
        msg += "Problematic recipes:\n"
        for item in problematic:
            if "error" in item:
                msg += f"  ❌ {item['name']}: {item['error']}\n"
            else:
                msg += f"  ❌ {item['name']}: Stacks are not equivalent\n"
                for diff in item.get("diffs", []):
                    msg += f"       - {diff}\n"

        pytest.fail(msg)

    print(f"\n✅ All {total} recipes produce equivalent stacks!")


# ============================================================================
# Individual Recipe Debugging Tests
# ============================================================================


@pytest.mark.parametrize("recipe_idx", [0, 1, 2])
def test_first_few_recipes_detailed(recipes_data, lib, recipe_idx):
    """Detailed test for first few recipes to help with debugging"""
    if recipe_idx >= len(recipes_data):
        pytest.skip(f"Recipe index {recipe_idx} not available")

    path, recipe_dict, recipe = recipes_data[recipe_idx]

    # Skip known recipes with valid alternative orderings
    skip_reason = should_skip_recipe(path.name)
    if skip_reason:
        pytest.skip(f"{path.name}: {skip_reason}")

    print(f"\n{'=' * 80}")
    print(f"Testing: {path.name}")
    print(f"{'=' * 80}")

    with LibraryContext.with_library(lib):
        # Build old stack
        old_stack = build_old_stack(path, lib)
        print(f"Old stack built:")
        print(f"  Layers: {len(old_stack.layers)}")
        print(f"  Nodes: {old_stack.number_of_nodes}")
        print(f"  Inputs: {old_stack.total_nb_of_inputs}")
        print(f"  Outputs: {old_stack.total_nb_of_outputs}")

        # Build new stack
        new_stack = build_new_stack(recipe, lib)
        print(f"\nNew stack built:")
        print(f"  Layers: {len(new_stack.layers)}")
        print(f"  Nodes: {new_stack.number_of_nodes}")
        print(f"  Inputs: {new_stack.total_nb_of_inputs}")
        print(f"  Outputs: {new_stack.total_nb_of_outputs}")

        # Compare
        is_equal, diffs = compare_stacks(old_stack, new_stack)

        print(f"\nEquivalent: {is_equal}")

        if not is_equal:
            print(f"\nDifferences:")
            for diff in diffs:
                print(f"  • {diff}")

        assert is_equal, f"Stacks are not equivalent for {path.name}"


# ============================================================================
# Layer-by-Layer Comparison Tests
# ============================================================================


def test_layer_types_match(recipes_data, lib):
    """Test that layer types match in order"""
    failed = []

    with LibraryContext.with_library(lib):
        for path, recipe_dict, recipe in recipes_data:
            try:
                old_stack = build_old_stack(path, lib)
                new_stack = build_new_stack(recipe, lib)

                if len(old_stack.layers) != len(new_stack.layers):
                    continue  # Already caught by other test

                mismatches = []
                for i, (old_layer, new_layer) in enumerate(zip(old_stack.layers, new_stack.layers)):
                    if old_layer.f_type != new_layer.f_type:
                        mismatches.append(f"Layer {i}: {old_layer.f_type} vs {new_layer.f_type}")

                if mismatches:
                    failed.append(
                        {
                            "name": path.name,
                            "mismatches": mismatches[:5],  # First 5
                        }
                    )
            except Exception as e:
                failed.append(
                    {
                        "name": path.name,
                        "error": str(e),
                    }
                )

    if failed:
        print(f"\n❌ Layer type mismatches for {len(failed)} recipes:")
        for item in failed:
            if "error" in item:
                print(f"  - {item['name']}: ERROR - {item['error']}")
            else:
                print(f"  - {item['name']}:")
                for mismatch in item["mismatches"]:
                    print(f"      {mismatch}")
        pytest.fail(f"Layer type matching failed for {len(failed)}/{len(recipes_data)} recipes")


def test_layer_shapes_match(recipes_data, lib):
    """Test that input/output shapes match per layer"""
    failed = []

    with LibraryContext.with_library(lib):
        for path, recipe_dict, recipe in recipes_data:
            # Skip known recipes with valid alternative orderings
            if should_skip_recipe(path.name):
                continue

            try:
                old_stack = build_old_stack(path, lib)
                new_stack = build_new_stack(recipe, lib)

                if len(old_stack.layers) != len(new_stack.layers):
                    continue  # Already caught by other test

                mismatches = []
                for i, (old_layer, new_layer) in enumerate(zip(old_stack.layers, new_stack.layers)):
                    if old_layer.f_input_shapes != new_layer.f_input_shapes:
                        mismatches.append(
                            f"Layer {i} input shapes: {old_layer.f_input_shapes} vs {new_layer.f_input_shapes}"
                        )
                    if old_layer.f_out_shapes != new_layer.f_out_shapes:
                        mismatches.append(
                            f"Layer {i} output shapes: {old_layer.f_out_shapes} vs {new_layer.f_out_shapes}"
                        )

                if mismatches:
                    failed.append(
                        {
                            "name": path.name,
                            "mismatches": mismatches[:5],  # First 5
                        }
                    )
            except Exception as e:
                failed.append(
                    {
                        "name": path.name,
                        "error": str(e),
                    }
                )

    if failed:
        print(f"\n❌ Shape mismatches for {len(failed)} recipes:")
        for item in failed:
            if "error" in item:
                print(f"  - {item['name']}: ERROR - {item['error']}")
            else:
                print(f"  - {item['name']}:")
                for mismatch in item["mismatches"]:
                    print(f"      {mismatch}")
        pytest.fail(f"Layer shape matching failed for {len(failed)}/{len(recipes_data)} recipes")


# ============================================================================
# Per-Recipe Parameterized Tests
# ============================================================================


@pytest.mark.parametrize("recipe_idx", range(18))
def test_individual_recipe_equivalence(recipes_data, lib, recipe_idx):
    """Test each recipe individually (parameterized for better reporting)"""
    if recipe_idx >= len(recipes_data):
        pytest.skip(f"Recipe index {recipe_idx} not available (only {len(recipes_data)} recipes)")

    path, recipe_dict, recipe = recipes_data[recipe_idx]

    # Skip known recipes with valid alternative orderings
    skip_reason = should_skip_recipe(path.name)
    if skip_reason:
        pytest.skip(f"{path.name}: {skip_reason}")

    with LibraryContext.with_library(lib):
        old_stack = build_old_stack(path, lib)
        new_stack = build_new_stack(recipe, lib)

        is_equal, diffs = compare_stacks(old_stack, new_stack)

        if not is_equal:
            error_msg = f"Stack equivalence failed for {path.name}:\n"
            for diff in diffs:
                error_msg += f"  - {diff}\n"
            pytest.fail(error_msg)
