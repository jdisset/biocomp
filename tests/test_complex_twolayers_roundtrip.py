"""Test commit-export-rebuild roundtrip for complex_twolayers network

Validates:
- Ratio padding when fewer ratios than units
- Ratio range constraints with NumRange
- Network export to recipe
- Recipe rebuild with correct structure
- Parameter preservation through roundtrip
- Output consistency after rebuild
"""

import pytest
import jax
import jax.numpy as jnp
import biocomp.parameters as pr
from biocomp.network import recipe_to_networks
from biocomp.library import LibraryContext, load_lib
import biocomp.biorules as br
from biocomp.compute import ComputeStack
from biocomp.config import SIMPLE_NODES_COMPUTE_CONFIG
from biocomp.recipe import Recipe, CoTransfection, TranscriptionUnit, Slot, FluoIntensity, NumRange


P = "hEF1a"
T = "L0.T_4560"
ERNS = ["CasE", "Csy4", "PgU"]
UORFS = [None, "1w_uORF", "1x_uORF", "2x_uORF", "3x_uORF", "4x_uORF", "5x_uORF", "6x_uORF", "8x_uORF"]
COLORS = {"x1": "mKO2", "x2": "eBFP2", "b": "mMaroon1", "y": "mNeonGreen"}
BIAS_FLUO = FluoIntensity(tu_id=0, value=NumRange(min=0.3, max=0.6), protein=COLORS["b"], units="Rescaled AU")


def make_units(tu_name, erns):
    recs = [f"{ern}_rec" for ern in erns]
    u1 = Slot(part=UORFS[0], ref_id="U1")
    u2 = Slot(part=UORFS, ref_id="U2")
    u3 = Slot(part=UORFS[1:], ref_id="U3")
    return [
        TranscriptionUnit(slots=[P, COLORS[tu_name], T], name=f"{tu_name}_marker", source="themarker"),
        TranscriptionUnit(slots=[P, u1, recs[0], erns[2], T], name=f"{tu_name}_a+", source="03"),
        TranscriptionUnit(slots=[P, erns[0], T], name=f"{tu_name}_a-", source="45"),
        TranscriptionUnit(slots=[P, u2, recs[1], erns[2], T], name=f"{tu_name}_b+", source="haha12"),
        TranscriptionUnit(slots=[P, erns[1], T], name=f"{tu_name}_b-", source="wrong order 78"),
        TranscriptionUnit(slots=[P, u3, recs[2], COLORS["y"], T], name=f"{tu_name}_c+", source="a random id"),
        TranscriptionUnit(slots=[P, erns[2], T], name=f"{tu_name}_c-", source="00aaa"),
        TranscriptionUnit(slots=[P, COLORS["y"], T], name=f"{tu_name}_direct_out", source="direct"),
    ]


@pytest.fixture
def lib():
    return load_lib()


@pytest.fixture
def design_recipe_with_ranges(lib):
    with LibraryContext.with_library(lib):
        recipe = Recipe(
            name="two_and_one",
            content=[
                CoTransfection(name="x1", units=make_units("x1", ERNS), ratios=[NumRange(min=1, max=10) for _ in range(8)]),
                CoTransfection(name="x2", units=make_units("x2", ERNS), ratios=[NumRange(min=2, max=100) for _ in range(8)]),
                CoTransfection(name="b", units=make_units("b", ERNS), fluo_bias=BIAS_FLUO, ratios=[NumRange(min=1, max=2) for _ in range(8)]),
            ],
        )
    return recipe


def test_ratio_count_validation(lib):
    with LibraryContext.with_library(lib):
        recipe = Recipe(
            name="mismatched",
            content=[
                CoTransfection(name="x1", units=make_units("x1", ERNS), ratios=[1.0, 2.0, 3.0])
            ],
        )
        with pytest.raises(ValueError, match="ratios count .* must match units count"):
            recipe_to_networks(recipe, br.ALL_RULES, invert=True)


def test_ratio_range_bounds(lib, design_recipe_with_ranges):
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(design_recipe_with_ranges, br.ALL_RULES, invert=True)
        network = networks[0]
        stack = ComputeStack([network])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        orig_key = jax.random.PRNGKey(123)
        params = stack.init(orig_key)

        agglayer = stack.layers[6]
        assert len(agglayer.nodes) == 3

        for i, stack_node in enumerate(agglayer.nodes):
            cotxnode = network.compute_graph.get_node(stack_node.node_id)
            grpname = cotxnode.extra['cotx_group']
            ratios = params[f'{agglayer.namespace}/ratios'][i]
            ratio_range = ratios.max() / ratios.min()

            maxrange_expected = {"x1": 10.0, "x2": 50.0, "b": 2.0}[grpname]
            assert ratio_range <= maxrange_expected + 1e-6


def test_commit_and_export(lib, design_recipe_with_ranges):
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(design_recipe_with_ranges, br.ALL_RULES, invert=True)
        network = networks[0]
        stack = ComputeStack([network])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        key = jax.random.PRNGKey(42)
        params = stack.init(key)

        committed_network = stack.commit(params)[0]
        exported_recipe = committed_network.to_recipe()

        assert len(exported_recipe.content) == 3
        for cotx in exported_recipe.content:
            assert len(cotx.units) == 8
            assert len(cotx.ratios) == 8


def test_rebuild_structure(lib, design_recipe_with_ranges):
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(design_recipe_with_ranges, br.ALL_RULES, invert=True)
        network = networks[0]
        stack = ComputeStack([network])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        key = jax.random.PRNGKey(42)
        params = stack.init(key)

        committed_network = stack.commit(params)[0]
        exported_recipe = committed_network.to_recipe()

        rebuilt_network = recipe_to_networks(exported_recipe, br.ALL_RULES, invert=True)[0]
        rebuilt_stack = ComputeStack([rebuilt_network])
        rebuilt_stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        assert len(rebuilt_stack.layers) == len(stack.layers)


def test_ratios_preserved(lib, design_recipe_with_ranges):
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(design_recipe_with_ranges, br.ALL_RULES, invert=True)
        network = networks[0]
        stack = ComputeStack([network])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        orig_key = jax.random.PRNGKey(123)
        opt_key = jax.random.PRNGKey(42)

        orig_params = stack.init(orig_key)
        opt_params = stack.init(opt_key)
        orig_shared, _ = orig_params.filter_by_tag(['shared'])
        _, opt_nonshared = opt_params.filter_by_tag(['shared'])
        opt_params = pr.ParameterTree.merge(orig_shared, opt_nonshared)

        committed_network = stack.commit(opt_params)[0]
        exported_recipe = committed_network.to_recipe()

        rebuilt_network = recipe_to_networks(exported_recipe, br.ALL_RULES, invert=True)[0]
        rebuilt_stack = ComputeStack([rebuilt_network])
        rebuilt_stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)
        rebuilt_params = rebuilt_stack.init(opt_key)
        rebuilt_shared, rebuilt_nonshared = rebuilt_params.filter_by_tag(['shared'])
        rebuilt_params = pr.ParameterTree.merge(orig_shared, rebuilt_nonshared)

        for layer in stack.layers:
            if any(network.compute_graph.get_node(n.node_id).node_type == "aggregation" for n in layer.nodes):
                ratio_path = f"{layer.namespace}/ratios"
                opt_ratios = opt_params[ratio_path]
                reb_ratios = rebuilt_params[ratio_path]
                assert jnp.allclose(opt_ratios, reb_ratios)


def test_outputs_match(lib, design_recipe_with_ranges):
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(design_recipe_with_ranges, br.ALL_RULES, invert=True)
        network = networks[0]
        stack = ComputeStack([network])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        orig_key = jax.random.PRNGKey(123)
        opt_key = jax.random.PRNGKey(42)
        eval_key = jax.random.PRNGKey(1234)

        orig_params = stack.init(orig_key)
        opt_params = stack.init(opt_key)
        orig_shared, _ = orig_params.filter_by_tag(['shared'])
        _, opt_nonshared = opt_params.filter_by_tag(['shared'])
        opt_params = pr.ParameterTree.merge(orig_shared, opt_nonshared)

        committed_network = stack.commit(opt_params)[0]
        exported_recipe = committed_network.to_recipe()

        rebuilt_network = recipe_to_networks(exported_recipe, br.ALL_RULES, invert=True)[0]
        rebuilt_stack = ComputeStack([rebuilt_network])
        rebuilt_stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)
        rebuilt_params = rebuilt_stack.init(opt_key)
        rebuilt_shared, rebuilt_nonshared = rebuilt_params.filter_by_tag(['shared'])
        rebuilt_params = pr.ParameterTree.merge(orig_shared, rebuilt_nonshared)

        N_EVALS = 100
        x = jax.random.uniform(eval_key, (N_EVALS, 2))
        num_z = opt_params["global/number_of_random_variables"]
        random_variables = jnp.zeros((num_z,))

        y_opt, _ = jax.vmap(stack.apply, in_axes=(None, 0, None, None))(opt_params, x, random_variables, eval_key)
        y_rebuilt, _ = jax.vmap(rebuilt_stack.apply, in_axes=(None, 0, None, None))(rebuilt_params, x, random_variables, eval_key)

        assert y_opt.shape == (N_EVALS, 4)
        assert y_rebuilt.shape == (N_EVALS, 4)
        assert jnp.allclose(y_opt, y_rebuilt)


def test_source_output_slots(lib):
    with LibraryContext.with_library(lib):
        recipe = Recipe(
            name="multi_tu_source",
            content=[
                CoTransfection(
                    name="x1",
                    units=[
                        TranscriptionUnit(slots=[P, COLORS["x1"], T], name="tu1", source="shared_src"),
                        TranscriptionUnit(slots=[P, COLORS["x1"], T], name="tu2", source="shared_src"),
                        TranscriptionUnit(slots=[P, COLORS["x1"], T], name="tu3", source="unique_src"),
                    ],
                    ratios=[1.0, 2.0, 3.0],
                )
            ],
        )

        networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True)
        network = networks[0]

        sources = network.compute_graph.get_nodes_by_type("source")
        shared_src = [s for s in sources if s.extra.get("source_id") == "shared_src"]

        assert len(shared_src) == 1
        outgoing = network.compute_graph.get_outgoing_edges(shared_src[0].node_id)
        slots = [e.from_output_slot for e in outgoing]
        assert sorted(slots) == [0, 1]


if __name__ == "__main__":
    lib_instance = load_lib()

    with LibraryContext.with_library(lib_instance):
        recipe_instance = Recipe(
            name="two_and_one",
            content=[
                CoTransfection(name="x1", units=make_units("x1", ERNS), ratios=[NumRange(min=1, max=10) for _ in range(8)]),
                CoTransfection(name="x2", units=make_units("x2", ERNS), ratios=[NumRange(min=2, max=100) for _ in range(8)]),
                CoTransfection(name="b", units=make_units("b", ERNS), fluo_bias=BIAS_FLUO, ratios=[NumRange(min=1, max=2) for _ in range(8)]),
            ],
        )

    print("Running ratio count validation test...")
    test_ratio_count_validation(lib_instance)
    print("✓ Ratio count validation test passed\n")

    print("Running ratio range bounds test...")
    test_ratio_range_bounds(lib_instance, recipe_instance)
    print("✓ Ratio range bounds test passed\n")

    print("Running commit and export test...")
    test_commit_and_export(lib_instance, recipe_instance)
    print("✓ Commit and export test passed\n")

    print("Running rebuild structure test...")
    test_rebuild_structure(lib_instance, recipe_instance)
    print("✓ Rebuild structure test passed\n")

    print("Running ratios preserved test...")
    test_ratios_preserved(lib_instance, recipe_instance)
    print("✓ Ratios preserved test passed\n")

    print("Running outputs match test...")
    test_outputs_match(lib_instance, recipe_instance)
    print("✓ Outputs match test passed\n")

    print("Running source output slots test...")
    test_source_output_slots(lib_instance)
    print("✓ Source output slots test passed\n")

    print("All roundtrip tests passed!")
