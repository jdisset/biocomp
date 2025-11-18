"""Test backward compatibility: new and old network systems must generate isomorphic graphs"""

import pytest
import json5
from pathlib import Path
from collections import Counter

import biocomp.old_network.recipe as reco
import biocomp.recipe as recn
import biocomp.network as netn
import biocomp.biorules as br
from biocomp.graphengine import apply_rule_sequence, graphs_are_isomorphic
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
# Helper Functions (from roadmap.md)
# ============================================================================


def build_old_network(recipe_path, lib):
    """Build network using old imperative system"""

    def error_handler(msg):
        print(f"Old system error: {msg}")
        return False

    old_net = reco.network_from_recipe(
        recipe_path, lib, inverse="all", error_handler=error_handler
    )[0]
    old_net.build()
    return old_net


def build_new_network_cdg(recipe, lib):
    """Build CDG using new declarative system"""
    cdg = netn.build_central_dogma_graph_direct(recipe.content, lib, dual=True)
    return cdg


def build_new_network_compg(recipe, lib):
    """Build compute graph using new declarative system with rules"""
    cdg = build_new_network_cdg(recipe, lib)
    compg = apply_rule_sequence(br.ALL_RULES, cdg)[0]
    compg = br.sort_output_edges(compg)
    return compg


def old_network_to_graphstate(old_net):
    """Convert old network's compute graph to GraphState"""
    return netn.old_network_compg_to_graphstate(old_net)


# ============================================================================
# Basic Sanity Tests
# ============================================================================


def test_recipes_loaded(recipes_data):
    """Sanity check: ensure we loaded some recipes"""
    assert len(recipes_data) > 0, "No recipes were loaded"
    print(f"\nLoaded {len(recipes_data)} valid recipes")


def test_old_system_still_works(recipe_paths, lib):
    """Sanity check: old system can still build networks"""
    with LibraryContext.with_library(lib):
        for path in recipe_paths[:3]:  # Test first 3
            try:
                old_net = build_old_network(path, lib)
                assert old_net.is_built()
                assert old_net.compute_graph is not None
            except Exception as e:
                pytest.fail(f"Old system failed on {path.name}: {e}")


def test_new_system_builds_cdg(recipes_data, lib):
    """Sanity check: new system can build CDGs"""
    with LibraryContext.with_library(lib):
        for path, recipe_dict, recipe in recipes_data[:3]:  # Test first 3
            try:
                cdg = build_new_network_cdg(recipe, lib)
                assert len(cdg.nodes) > 0
                assert len(cdg.edges) > 0
            except Exception as e:
                pytest.fail(f"New CDG build failed on {path.name}: {e}")


def test_new_system_applies_rules(recipes_data, lib):
    """Sanity check: rules can be applied to CDGs"""
    with LibraryContext.with_library(lib):
        for path, recipe_dict, recipe in recipes_data[:3]:  # Test first 3
            try:
                compg = build_new_network_compg(recipe, lib)
                assert compg is not None
                assert len(compg.nodes) > 0
            except Exception as e:
                pytest.fail(f"Rule application failed on {path.name}: {e}")


# ============================================================================
# Structure Equivalence Tests
# ============================================================================


def test_node_count_equivalence(recipes_data, lib):
    """Test that old and new produce same number of nodes"""
    failed = []

    with LibraryContext.with_library(lib):
        for path, recipe_dict, recipe in recipes_data:
            try:
                old_net = build_old_network(path, lib)
                new_compg = build_new_network_compg(recipe, lib)
                old_compg = old_network_to_graphstate(old_net)

                if len(new_compg.nodes) != len(old_compg.nodes):
                    failed.append(
                        {
                            "name": path.name,
                            "new_count": len(new_compg.nodes),
                            "old_count": len(old_compg.nodes),
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
                print(f"  - {item['name']}: new={item['new_count']}, old={item['old_count']}")
        pytest.fail(f"Node count equivalence failed for {len(failed)}/{len(recipes_data)} recipes")


def test_edge_count_equivalence(recipes_data, lib):
    """Test that old and new produce same number of edges"""
    failed = []

    with LibraryContext.with_library(lib):
        for path, recipe_dict, recipe in recipes_data:
            try:
                old_net = build_old_network(path, lib)
                new_compg = build_new_network_compg(recipe, lib)
                old_compg = old_network_to_graphstate(old_net)

                if len(new_compg.edges) != len(old_compg.edges):
                    failed.append(
                        {
                            "name": path.name,
                            "new_count": len(new_compg.edges),
                            "old_count": len(old_compg.edges),
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
        print(f"\n❌ Edge count mismatches for {len(failed)} recipes:")
        for item in failed:
            if "error" in item:
                print(f"  - {item['name']}: ERROR - {item['error']}")
            else:
                print(f"  - {item['name']}: new={item['new_count']}, old={item['old_count']}")
        pytest.fail(f"Edge count equivalence failed for {len(failed)}/{len(recipes_data)} recipes")


def test_node_type_distribution(recipes_data, lib):
    """Test that node type distributions match"""
    failed = []

    with LibraryContext.with_library(lib):
        for path, recipe_dict, recipe in recipes_data:
            try:
                old_net = build_old_network(path, lib)
                new_compg = build_new_network_compg(recipe, lib)
                old_compg = old_network_to_graphstate(old_net)

                new_types = Counter(n.node_type for n in new_compg.nodes.values())
                old_types = Counter(n.node_type for n in old_compg.nodes.values())

                if new_types != old_types:
                    failed.append(
                        {
                            "name": path.name,
                            "new_types": dict(new_types),
                            "old_types": dict(old_types),
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
        print(f"\n❌ Node type distribution mismatches for {len(failed)} recipes:")
        for item in failed:
            if "error" in item:
                print(f"  - {item['name']}: ERROR - {item['error']}")
            else:
                print(f"  - {item['name']}:")
                print(f"      New: {item['new_types']}")
                print(f"      Old: {item['old_types']}")
        pytest.fail(f"Node type distribution failed for {len(failed)}/{len(recipes_data)} recipes")


# ============================================================================
# Graph Isomorphism Test (The Main Event)
# ============================================================================


def test_graph_isomorphism_all_recipes(recipes_data, lib):
    """Test that new and old systems produce isomorphic graphs for all recipes

    This is the comprehensive test based on roadmap.md that checks if the
    graphs are structurally equivalent (isomorphic).
    """
    passed = []
    failed = []
    errors = []

    with LibraryContext.with_library(lib):
        for path, recipe_dict, recipe in recipes_data:
            recipe_name = path.name

            try:
                # Build both networks (from roadmap.md)
                old_net = build_old_network(path, lib)
                new_compg = build_new_network_compg(recipe, lib)
                old_compg = old_network_to_graphstate(old_net)

                # Test isomorphism (from roadmap.md)
                iso = graphs_are_isomorphic(
                    new_compg,
                    old_compg,
                    unordered_outgoing_types={"aggregation"},
                    unordered_incoming_types={"output", "translation", "transcription"},
                )

                if iso:
                    passed.append(recipe_name)
                else:
                    failed.append(
                        {
                            "name": recipe_name,
                            "nodes": f"new={len(new_compg.nodes)}, old={len(old_compg.nodes)}",
                            "edges": f"new={len(new_compg.edges)}, old={len(old_compg.edges)}",
                        }
                    )

            except Exception as e:
                errors.append(
                    {
                        "name": recipe_name,
                        "error": str(e),
                    }
                )

    # Report results
    total = len(recipes_data)
    passed_count = len(passed)
    failed_count = len(failed)
    error_count = len(errors)
    success_rate = (passed_count / total * 100) if total > 0 else 0

    print(f"\n{'=' * 80}")
    print(f"GRAPH ISOMORPHISM TEST RESULTS")
    print(f"{'=' * 80}")
    print(f"Total recipes:  {total}")
    print(f"✅ Passed:      {passed_count} ({success_rate:.1f}%)")
    print(f"❌ Failed:      {failed_count}")
    print(f"⚠️  Errors:      {error_count}")
    print(f"{'=' * 80}")

    if failed:
        print(f"\n❌ Failed isomorphism checks:")
        for item in failed:
            print(f"  - {item['name']}")
            print(f"      {item['nodes']}, {item['edges']}")

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
        msg = f"\nGraph isomorphism test failed for {len(problematic)}/{total} recipes.\n"
        msg += f"Success rate: {success_rate:.1f}%\n\n"
        msg += "Problematic recipes:\n"
        for item in problematic:
            if "error" in item:
                msg += f"  ❌ {item['name']}: {item['error']}\n"
            else:
                msg += f"  ❌ {item['name']}: Graphs are not isomorphic\n"

        pytest.fail(msg)

    print(f"\n✅ All {total} recipes produce isomorphic graphs!")


# ============================================================================
# Individual Recipe Debugging Tests
# ============================================================================


@pytest.mark.parametrize("recipe_idx", [0, 1, 2])
def test_first_few_recipes_detailed(recipes_data, lib, recipe_idx):
    """Detailed test for first few recipes to help with debugging"""
    if recipe_idx >= len(recipes_data):
        pytest.skip(f"Recipe index {recipe_idx} not available")

    path, recipe_dict, recipe = recipes_data[recipe_idx]

    print(f"\n{'=' * 80}")
    print(f"Testing: {path.name}")
    print(f"{'=' * 80}")

    with LibraryContext.with_library(lib):
        # Build old network
        old_net = build_old_network(path, lib)
        print(f"Old network built: {old_net.name}")
        print(f"  TUs: {len(old_net.transcription_units)}")
        print(f"  Aggregations: {len(old_net.aggregations)}")

        # Build new CDG
        cdg = build_new_network_cdg(recipe, lib)
        print(f"\nNew CDG built:")
        print(f"  Nodes: {len(cdg.nodes)}")
        print(f"  Edges: {len(cdg.edges)}")

        # Apply rules
        compg = apply_rule_sequence(br.ALL_RULES, cdg)[0]
        print(f"\nAfter applying rules:")
        print(f"  Nodes: {len(compg.nodes)}")
        print(f"  Edges: {len(compg.edges)}")

        # Convert old
        old_compg = old_network_to_graphstate(old_net)
        print(f"\nOld compute graph:")
        print(f"  Nodes: {len(old_compg.nodes)}")
        print(f"  Edges: {len(old_compg.edges)}")

        # Check isomorphism
        iso = graphs_are_isomorphic(
            compg,
            old_compg,
            unordered_outgoing_types={"aggregation"},
            unordered_incoming_types={"output", "translation", "transcription"},
        )

        print(f"\nIsomorphic: {iso}")

        if not iso:
            # Show node type distribution
            new_types = Counter(n.node_type for n in compg.nodes.values())
            old_types = Counter(n.node_type for n in old_compg.nodes.values())
            print(f"\nNode type distribution:")
            print(f"  New: {dict(new_types)}")
            print(f"  Old: {dict(old_types)}")

        assert iso, f"Graphs are not isomorphic for {path.name}"


# ============================================================================
# Network Helper Methods Equivalence Tests
# ============================================================================


def test_get_output_compute_node_equivalence(recipes_data, lib):
    """Test that get_output_compute_node returns equivalent output node"""
    with LibraryContext.with_library(lib):
        for path, recipe_dict, recipe in recipes_data:
            old_net = build_old_network(path, lib)
            new_compg = build_new_network_compg(recipe, lib)
            new_net = netn.Network(name="test", compute_graph=new_compg)

            old_output_node = old_net.get_output_compute_node()
            new_output_node = new_net.get_output_compute_node()

            assert old_output_node["type"] == new_output_node.node_type == "output"


def test_nb_outputs_equivalence(recipes_data, lib):
    """Test that nb_outputs returns same value"""
    with LibraryContext.with_library(lib):
        for path, recipe_dict, recipe in recipes_data:
            old_net = build_old_network(path, lib)
            new_compg = build_new_network_compg(recipe, lib)
            new_net = netn.Network(name="test", compute_graph=new_compg)

            assert old_net.nb_outputs == new_net.nb_outputs, (
                f"{path.name}: old={old_net.nb_outputs}, new={new_net.nb_outputs}"
            )


def test_nb_inputs_equivalence(recipes_data, lib):
    """Test that nb_inputs returns same value"""
    with LibraryContext.with_library(lib):
        for path, recipe_dict, recipe in recipes_data:
            old_net = build_old_network(path, lib)
            new_compg = build_new_network_compg(recipe, lib)
            new_net = netn.Network(name="test", compute_graph=new_compg)

            # nb_inputs only makes sense for inverted networks
            if old_net.nb_inputs > 0:
                assert old_net.nb_inputs == new_net.nb_inputs, (
                    f"{path.name}: old={old_net.nb_inputs}, new={new_net.nb_inputs}"
                )


def test_get_output_proteins_equivalence(recipes_data, lib):
    """Test that get_output_proteins returns proteins in alphabetical order"""
    with LibraryContext.with_library(lib):
        for path, recipe_dict, recipe in recipes_data:
            old_net = build_old_network(path, lib)
            new_compg = build_new_network_compg(recipe, lib)
            new_net = netn.Network(name="test", compute_graph=new_compg)

            old_proteins = old_net.get_output_proteins()
            new_proteins = new_net.get_output_proteins()

            # Same proteins, and new system should be in alphabetical order
            assert set(old_proteins) == set(new_proteins), (
                f"{path.name}: old={old_proteins}, new={new_proteins}"
            )
            assert len(old_proteins) == len(new_proteins), (
                f"{path.name}: old has {len(old_proteins)} proteins, new has {len(new_proteins)}"
            )
            # New system should be alphabetically sorted
            assert new_proteins == sorted(new_proteins), (
                f"{path.name}: new proteins not alphabetically sorted: {new_proteins}"
            )


def test_get_inverted_input_positions_equivalence(recipes_data, lib):
    """Test that get_inverted_input_positions returns same mapping"""
    with LibraryContext.with_library(lib):
        for path, recipe_dict, recipe in recipes_data:
            old_net = build_old_network(path, lib)
            new_compg = build_new_network_compg(recipe, lib)
            new_net = netn.Network(name="test", compute_graph=new_compg)

            if old_net.nb_inputs > 0:
                old_mapping = old_net.get_inverted_input_positions()
                new_mapping = new_net.get_inverted_input_positions()

                assert old_mapping == new_mapping, (
                    f"{path.name}: old={old_mapping}, new={new_mapping}"
                )


def test_get_inverted_input_proteins_equivalence(recipes_data, lib):
    """Test that get_inverted_input_proteins returns same proteins"""
    with LibraryContext.with_library(lib):
        for path, recipe_dict, recipe in recipes_data:
            old_net = build_old_network(path, lib)
            new_compg = build_new_network_compg(recipe, lib)
            new_net = netn.Network(name="test", compute_graph=new_compg)

            if old_net.nb_inputs > 0:
                old_proteins = old_net.get_inverted_input_proteins()
                new_proteins = new_net.get_inverted_input_proteins()

                assert old_proteins == new_proteins, (
                    f"{path.name}: old={old_proteins}, new={new_proteins}"
                )


def test_get_dependent_output_proteins_equivalence(recipes_data, lib):
    """Test that get_dependent_output_proteins returns same proteins"""
    with LibraryContext.with_library(lib):
        for path, recipe_dict, recipe in recipes_data:
            old_net = build_old_network(path, lib)
            new_compg = build_new_network_compg(recipe, lib)
            new_net = netn.Network(name="test", compute_graph=new_compg)

            if old_net.nb_inputs > 0:
                old_proteins = old_net.get_dependent_output_proteins()
                new_proteins = new_net.get_dependent_output_proteins()

                assert old_proteins == new_proteins, (
                    f"{path.name}: old={old_proteins}, new={new_proteins}"
                )


def test_get_dependent_output_mask_equivalence(recipes_data, lib):
    """Test that get_dependent_output_mask returns same mask"""
    with LibraryContext.with_library(lib):
        for path, recipe_dict, recipe in recipes_data:
            old_net = build_old_network(path, lib)
            new_compg = build_new_network_compg(recipe, lib)
            new_net = netn.Network(name="test", compute_graph=new_compg)

            if old_net.nb_inputs > 0:
                old_mask = old_net.get_dependent_output_mask()
                new_mask = new_net.get_dependent_output_mask()

                assert (old_mask == new_mask).all(), f"{path.name}: masks differ"


def test_topological_order_equivalence(recipes_data, lib):
    """Test that topological_order returns compatible ordering"""
    with LibraryContext.with_library(lib):
        for path, recipe_dict, recipe in recipes_data:
            old_net = build_old_network(path, lib)
            new_compg = build_new_network_compg(recipe, lib)
            new_net = netn.Network(name="test", compute_graph=new_compg)

            old_topo = old_net.topological_order()
            new_topo = new_net.topological_order()

            # Same number of layers
            assert len(old_topo) == len(new_topo), (
                f"{path.name}: old={len(old_topo)} layers, new={len(new_topo)} layers"
            )

            # Same total nodes
            old_total = sum(len(batch) for batch in old_topo)
            new_total = sum(len(batch) for batch in new_topo)
            assert old_total == new_total, (
                f"{path.name}: old={old_total} nodes, new={new_total} nodes"
            )


def test_network_info_equivalence(recipes_data, lib):
    """Test that generate_network_info returns compatible information"""

    def strip_tu_suffix(tu_name):
        """Remove _1, _2, etc. suffixes from TU names"""
        import re
        return re.sub(r'_\d+$', '', tu_name)

    def compare_all_parts(old_parts, new_parts):
        """Compare all_parts ignoring TU name suffixes"""
        # Strip suffixes from old keys
        old_stripped = {strip_tu_suffix(k): v for k, v in old_parts.items()}
        # New system shouldn't have suffixes but strip just in case
        new_stripped = {strip_tu_suffix(k): v for k, v in new_parts.items()}
        return set(old_stripped.keys()) == set(new_stripped.keys())

    def compare_uorf_ern_values(old_uorfs, new_uorfs, old_erns, new_erns):
        """Compare uORF values using ERN names as keys to handle ordering differences"""
        if len(old_uorfs) != len(new_uorfs) or len(old_erns) != len(new_erns):
            return False, "Different number of ERNs or uORF values"

        # Build dictionaries mapping ERN name to uORF values
        old_dict = dict(zip(old_erns, old_uorfs))
        new_dict = dict(zip(new_erns, new_uorfs))

        # Check that both have the same ERN names
        if set(old_dict.keys()) != set(new_dict.keys()):
            return False, f"Different ERN names: old={set(old_dict.keys())}, new={set(new_dict.keys())}"

        # Check that uORF values match for each ERN
        for ern_name in old_dict:
            if old_dict[ern_name] != new_dict[ern_name]:
                return False, f"uORF mismatch for {ern_name}: old={old_dict[ern_name]}, new={new_dict[ern_name]}"

        return True, None

    with LibraryContext.with_library(lib):
        failed = []
        for path, recipe_dict, recipe in recipes_data:
            try:
                old_net = build_old_network(path, lib)
                new_compg = build_new_network_compg(recipe, lib)
                new_net = netn.Network(name="test", compute_graph=new_compg)

                # Get network info from both systems
                old_info = old_net.generate_network_info()
                new_info = new_net.generate_network_info()

                # Check key fields match
                if old_info["sequestron_type"] != new_info["sequestron_type"]:
                    failed.append(f"{path.name}: sequestron_type mismatch - old={old_info['sequestron_type']}, new={new_info['sequestron_type']}")

                if old_info["architecture"] != new_info["architecture"]:
                    failed.append(f"{path.name}: architecture mismatch - old={old_info['architecture']}, new={new_info['architecture']}")

                # Check ERN names (order doesn't matter)
                if sorted(old_info["ern_names"]) != sorted(new_info["ern_names"]):
                    failed.append(f"{path.name}: ern_names mismatch - old={sorted(old_info['ern_names'])}, new={sorted(new_info['ern_names'])}")

                # Compare uORF values using ERN-based dictionary comparison
                if old_info["uorf_values"] or new_info["uorf_values"]:  # Only if there are uORFs
                    match, msg = compare_uorf_ern_values(
                        old_info["uorf_values"],
                        new_info["uorf_values"],
                        old_info["ern_names"],
                        new_info["ern_names"]
                    )
                    if not match:
                        failed.append(f"{path.name}: {msg}")

                if sorted(old_info["markers"]) != sorted(new_info["markers"]):
                    failed.append(f"{path.name}: markers mismatch - old={old_info['markers']}, new={new_info['markers']}")

                if sorted(old_info["output_proteins"]) != sorted(new_info["output_proteins"]):
                    failed.append(f"{path.name}: output_proteins mismatch - old={old_info['output_proteins']}, new={new_info['output_proteins']}")

                if sorted(old_info["dependent_outputs"]) != sorted(new_info["dependent_outputs"]):
                    failed.append(f"{path.name}: dependent_outputs mismatch - old={old_info['dependent_outputs']}, new={new_info['dependent_outputs']}")

                # Check all_parts has same structure (ignoring TU suffixes)
                if not compare_all_parts(old_info["all_parts"], new_info["all_parts"]):
                    old_stripped = {strip_tu_suffix(k) for k in old_info["all_parts"].keys()}
                    new_stripped = {strip_tu_suffix(k) for k in new_info["all_parts"].keys()}
                    failed.append(f"{path.name}: all_parts keys mismatch - old_stripped={old_stripped}, new_stripped={new_stripped}")

            except Exception as e:
                failed.append(f"{path.name}: ERROR - {str(e)}")

        if failed:
            print(f"\n❌ Network info mismatches for {len(failed)} issues:")
            for issue in failed:
                print(f"  - {issue}")
            pytest.fail(f"Network info compatibility test failed with {len(failed)} issues")
