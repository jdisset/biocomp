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
import numpy as np
import biocomp.parameters as pr
from biocomp.network import recipe_to_networks
from biocomp.library import LibraryContext, load_lib
from biocomp.ratio_schema import get_slot_entries
import biocomp.biorules as br
from biocomp.compute import ComputeStack
from biocomp.config import SIMPLE_NODES_COMPUTE_CONFIG
from biocomp.recipe import Recipe, CoTransfection, TranscriptionUnit, Slot, FluoIntensity, NumRange, RATIO_PRECISION


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


@pytest.fixture(params=[
    "complex_twolayers", "complex_twolayers_locked", "complex_twolayers_mixed",
    "simple_two_reporters", "simple_two_reporters_unlocked",
    "unlocked_ratios",
    "shared_source", "shared_source_unlocked", "shared_source_mixed"
])
def roundtrip_recipe(request, lib):
    with LibraryContext.with_library(lib):
        if request.param == "complex_twolayers":
            return Recipe(
                name="two_and_one",
                content=[
                    CoTransfection(name="x1", units=make_units("x1", ERNS), ratios=[NumRange(min=1, max=10) for _ in range(8)]),
                    CoTransfection(name="x2", units=make_units("x2", ERNS), ratios=[NumRange(min=2, max=100) for _ in range(8)]),
                    CoTransfection(name="b", units=make_units("b", ERNS), fluo_bias=BIAS_FLUO, ratios=[NumRange(min=1, max=2) for _ in range(8)]),
                ],
            )
        elif request.param == "complex_twolayers_locked":
            return Recipe(
                name="two_and_one_locked",
                content=[
                    CoTransfection(name="x1", units=make_units("x1", ERNS), ratios=[1.0 + i * 0.5 for i in range(8)]),
                    CoTransfection(name="x2", units=make_units("x2", ERNS), ratios=[2.0 + i * 2.0 for i in range(8)]),
                    CoTransfection(name="b", units=make_units("b", ERNS), fluo_bias=BIAS_FLUO, ratios=[1.0 + i * 0.1 for i in range(8)]),
                ],
            )
        elif request.param == "complex_twolayers_mixed":
            return Recipe(
                name="two_and_one_mixed",
                content=[
                    CoTransfection(name="x1", units=make_units("x1", ERNS), ratios=[NumRange(min=1, max=10) if i % 2 == 0 else 2.0 + i for i in range(8)]),
                    CoTransfection(name="x2", units=make_units("x2", ERNS), ratios=[5.0, NumRange(min=2, max=50), 10.0, NumRange(min=5, max=100), 15.0, 20.0, NumRange(min=10, max=80), 25.0]),
                    CoTransfection(name="b", units=make_units("b", ERNS), fluo_bias=BIAS_FLUO, ratios=[1.0, 1.2, NumRange(min=1, max=2), 1.4, 1.5, NumRange(min=1, max=1.8), 1.6, 1.7]),
                ],
            )
        elif request.param == "simple_two_reporters":
            return Recipe(
                name="simple_two_reporters",
                content=[
                    CoTransfection(
                        units=[
                            TranscriptionUnit(slots=["cHS4", "hEF1a", "eBFP2", "L0.T_4560"]),
                            TranscriptionUnit(slots=["cHS4", "hEF1a", "mMaroon1", "L0.T_4560"]),
                        ],
                        ratios=[0.833, 0.167],
                    )
                ],
            )
        elif request.param == "simple_two_reporters_unlocked":
            return Recipe(
                name="simple_two_reporters_unlocked",
                content=[
                    CoTransfection(
                        units=[
                            TranscriptionUnit(slots=["cHS4", "hEF1a", "eBFP2", "L0.T_4560"]),
                            TranscriptionUnit(slots=["cHS4", "hEF1a", "mMaroon1", "L0.T_4560"]),
                        ],
                        ratios=[NumRange(min=0.5, max=0.9), NumRange(min=0.1, max=0.5)],
                    )
                ],
            )
        elif request.param == "unlocked_ratios":
            return Recipe(
                name="unlocked_ratios",
                content=[
                    CoTransfection(
                        units=[
                            TranscriptionUnit(slots=["cHS4", "hEF1a", "CasE_rec", "eBFP2", "L0.T_4560"], source="p1"),
                            TranscriptionUnit(slots=["cHS4", "hEF1a", "CasE", "L0.T_4560"], source="p2"),
                            TranscriptionUnit(slots=["cHS4", "hEF1a", "mNeonGreen", "L0.T_4560"], source="p3"),
                        ],
                        ratios=[NumRange(min=0.2, max=0.5), 0.3, NumRange(min=0.1, max=0.3)],
                    )
                ],
            )
        elif request.param == "shared_source":
            u1 = Slot(part=["1w_uORF", "2x_uORF"], ref_id="U1")
            u2 = Slot(part=[None, "4x_uORF", "3x_uORF"], ref_id="U2")
            return Recipe(
                name="shared_source",
                content=[
                    CoTransfection(
                        units=[
                            TranscriptionUnit(slots=["cHS4", "hEF1a", u1, "CasE_rec", "eBFP2", "L0.T_4560"], source="plsmd1"),
                            TranscriptionUnit(slots=["cHS4", "hEF1a", "CasE", "L0.T_4560"], source="plsmd1"),
                            TranscriptionUnit(slots=["cHS4", "hEF1a", u2, "Csy4_rec", "eYFP", "L0.T_4560"], source="out2_plsmd"),
                            TranscriptionUnit(slots=["cHS4", "hEF1a", "Csy4", "L0.T_4560"], source="ern2_plsmd"),
                            TranscriptionUnit(slots=["mKO2"], source="mrkr_plsmd"),
                        ],
                        ratios=[i + 1 for i in range(4)],
                    )
                ],
            )
        elif request.param == "shared_source_unlocked":
            u1 = Slot(part=["1w_uORF", "2x_uORF"], ref_id="U1")
            u2 = Slot(part=[None, "4x_uORF", "3x_uORF"], ref_id="U2")
            return Recipe(
                name="shared_source_unlocked",
                content=[
                    CoTransfection(
                        units=[
                            TranscriptionUnit(slots=["cHS4", "hEF1a", u1, "CasE_rec", "eBFP2", "L0.T_4560"], source="plsmd1"),
                            TranscriptionUnit(slots=["cHS4", "hEF1a", "CasE", "L0.T_4560"], source="plsmd1"),
                            TranscriptionUnit(slots=["cHS4", "hEF1a", u2, "Csy4_rec", "eYFP", "L0.T_4560"], source="out2_plsmd"),
                            TranscriptionUnit(slots=["cHS4", "hEF1a", "Csy4", "L0.T_4560"], source="ern2_plsmd"),
                            TranscriptionUnit(slots=["mKO2"], source="mrkr_plsmd"),
                        ],
                        ratios=[NumRange(min=1, max=5), NumRange(min=2, max=8), NumRange(min=1, max=4), NumRange(min=1, max=3)],
                    )
                ],
            )
        elif request.param == "shared_source_mixed":
            u1 = Slot(part=["1w_uORF", "2x_uORF"], ref_id="U1")
            u2 = Slot(part=[None, "4x_uORF", "3x_uORF"], ref_id="U2")
            return Recipe(
                name="shared_source_mixed",
                content=[
                    CoTransfection(
                        units=[
                            TranscriptionUnit(slots=["cHS4", "hEF1a", u1, "CasE_rec", "eBFP2", "L0.T_4560"], source="plsmd1"),
                            TranscriptionUnit(slots=["cHS4", "hEF1a", "CasE", "L0.T_4560"], source="plsmd1"),
                            TranscriptionUnit(slots=["cHS4", "hEF1a", u2, "Csy4_rec", "eYFP", "L0.T_4560"], source="out2_plsmd"),
                            TranscriptionUnit(slots=["cHS4", "hEF1a", "Csy4", "L0.T_4560"], source="ern2_plsmd"),
                            TranscriptionUnit(slots=["mKO2"], source="mrkr_plsmd"),
                        ],
                        ratios=[NumRange(min=1, max=5), 2.5, NumRange(min=1, max=4), 1.8],
                    )
                ],
            )


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
        with pytest.raises(ValueError, match="ratios count .* must match number of unique sources"):
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


def test_commit_and_export(roundtrip_recipe):
    with LibraryContext.with_library(load_lib()):
        networks = recipe_to_networks(roundtrip_recipe, br.ALL_RULES, invert=True)
        network = networks[0]
        stack = ComputeStack([network])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        key = jax.random.PRNGKey(42)
        params = stack.init(key)

        committed_network = stack.commit(params)[0]
        exported_recipe = committed_network.to_recipe()

        assert len(exported_recipe.content) == len(roundtrip_recipe.content)
        for cotx_orig, cotx_exp in zip(roundtrip_recipe.content, exported_recipe.content, strict=False):
            assert len(cotx_exp.ratios) == len(set(tu.source for tu in cotx_orig.units))


def test_rebuild_structure(roundtrip_recipe):
    with LibraryContext.with_library(load_lib()):
        networks = recipe_to_networks(roundtrip_recipe, br.ALL_RULES, invert=True)
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


def test_ratios_preserved(roundtrip_recipe):
    with LibraryContext.with_library(load_lib()):
        networks = recipe_to_networks(roundtrip_recipe, br.ALL_RULES, invert=True)
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
                ratio_tol = 10 ** (-RATIO_PRECISION)
                opt_normalized = opt_ratios / jnp.sum(opt_ratios, axis=-1, keepdims=True)
                reb_normalized = reb_ratios / jnp.sum(reb_ratios, axis=-1, keepdims=True)
                assert jnp.allclose(opt_normalized, reb_normalized, rtol=ratio_tol, atol=ratio_tol)


def test_outputs_match(roundtrip_recipe):
    with LibraryContext.with_library(load_lib()):
        networks = recipe_to_networks(roundtrip_recipe, br.ALL_RULES, invert=True)
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

        nb_inputs = network.nb_inputs
        N_EVALS = 100
        x = jax.random.uniform(eval_key, (N_EVALS, nb_inputs))
        num_z = opt_params["global/number_of_random_variables"]
        random_variables = jnp.zeros((num_z,))

        y_opt, _ = jax.vmap(stack.apply, in_axes=(None, 0, None, None))(opt_params, x, random_variables, eval_key)
        y_rebuilt, _ = jax.vmap(rebuilt_stack.apply, in_axes=(None, 0, None, None))(rebuilt_params, x, random_variables, eval_key)

        assert y_opt.shape == y_rebuilt.shape
        output_tol = 5 * 10 ** (-RATIO_PRECISION + 1)
        assert jnp.allclose(y_opt, y_rebuilt, rtol=output_tol, atol=output_tol)

        # Also verify that rebuilding with original key produces original outputs
        # Only valid for recipes without unlocked ratios or unlocked slots (commit locks structure)
        from biocomp.recipe import NumRange, Slot
        has_unlocked_ratios = any(
            isinstance(r, NumRange) for cotx in roundtrip_recipe.content
            for r in (cotx.ratios or [])
        )
        # Check for unlocked slots (Slots with multiple part options)
        has_unlocked_slots = any(
            isinstance(slot, Slot) and isinstance(slot.part, list) and len(slot.part) > 1
            for cotx in roundtrip_recipe.content
            for tu in cotx.units
            for slot in tu.slots
        )

        if not has_unlocked_ratios and not has_unlocked_slots:
            orig_y, _ = jax.vmap(stack.apply, in_axes=(None, 0, None, None))(orig_params, x, random_variables, eval_key)
            rebuilt_orig_params = rebuilt_stack.init(orig_key)
            rebuilt_orig_shared, rebuilt_orig_nonshared = rebuilt_orig_params.filter_by_tag(['shared'])
            rebuilt_orig_params = pr.ParameterTree.merge(orig_shared, rebuilt_orig_nonshared)
            y_rebuilt_orig, _ = jax.vmap(rebuilt_stack.apply, in_axes=(None, 0, None, None))(rebuilt_orig_params, x, random_variables, eval_key)
            # Use rtol=1e-4 for large magnitude outputs, atol for near-zero
            assert jnp.allclose(orig_y, y_rebuilt_orig, rtol=1e-4, atol=1e-4)


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
                    ratios=[1.0, 2.0],
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


def test_shared_source_network_structure(lib):
    with LibraryContext.with_library(lib):
        u1 = Slot(part=["1w_uORF", "2x_uORF"], ref_id="U1")
        u2 = Slot(part=[None, "4x_uORF", "3x_uORF"], ref_id="U2")

        recipe = Recipe(
            name="shared_source",
            content=[
                CoTransfection(
                    units=[
                        TranscriptionUnit(name="output1", slots=["cHS4", "hEF1a", u1, "CasE_rec", "eBFP2", "L0.T_4560"], source="plsmd1"),
                        TranscriptionUnit(name="ern", slots=["cHS4", "hEF1a", "CasE", "L0.T_4560"], source="plsmd1"),
                        TranscriptionUnit(name="output2", slots=["cHS4", "hEF1a", u2, "Csy4_rec", "eYFP", "L0.T_4560"], source="out2_plsmd"),
                        TranscriptionUnit(name="ern2", slots=["cHS4", "hEF1a", "Csy4", "L0.T_4560"], source="ern2_plsmd"),
                        TranscriptionUnit(name="marker", slots=["mKO2"], source="mrkr_plsmd"),
                    ],
                    ratios=[i + 1 for i in range(4)],
                )
            ],
        )

        networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True)
        network = networks[0]
        stack = ComputeStack([network])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        assert network.nb_inputs == 1
        assert network.get_inverted_input_proteins() == ["mKO2"]
        assert network.get_output_proteins() == ["eBFP2", "eYFP", "mKO2"]
        assert np.all(network.get_dependent_output_mask() == [True, True, False])

        agglayer = stack.layers[5]
        assert len(agglayer.nodes) == 1
        aggnode = network.compute_graph.get_node(agglayer.nodes[0].node_id)

        slot_entries = get_slot_entries(aggnode.extra)
        members_to_ratio = {entry["source_id"]: entry["ratio"] for entry in slot_entries}
        assert members_to_ratio == {'plsmd1': 0.1, 'out2_plsmd': 0.2, 'ern2_plsmd': 0.3, 'mrkr_plsmd': 0.4}

        sorted_members = [entry["source_id"] for entry in slot_entries]
        plsmd1_idx = sorted_members.index('plsmd1')
        plsmd1_nodes = network.compute_graph.get_downstream_nodes_by_output_slot(aggnode.node_id, plsmd1_idx)
        assert len(plsmd1_nodes) == 1

        plsmd1_node = network.compute_graph.get_node(plsmd1_nodes[0])
        downstream = network.compute_graph.get_downstream_nodes(plsmd1_node.node_id)
        assert len(downstream) == 2

        for i, (dn, de) in enumerate(downstream):
            assert dn.node_type == 'transcription'
            if i == 0:
                assert de.content_embedding_names['tl_rate'] == ('1w_uORF', '2x_uORF')
            else:
                assert de.content_embedding_names['tl_rate'] == ('00_empty_tc',)


if __name__ == "__main__":

    lib_instance = load_lib()

    print("Running ratio count validation test...")
    test_ratio_count_validation(lib_instance)
    print("✓ Ratio count validation test passed\n")

    with LibraryContext.with_library(lib_instance):
        recipe_instance = Recipe(
            name="two_and_one",
            content=[
                CoTransfection(name="x1", units=make_units("x1", ERNS), ratios=[NumRange(min=1, max=10) for _ in range(8)]),
                CoTransfection(name="x2", units=make_units("x2", ERNS), ratios=[NumRange(min=2, max=100) for _ in range(8)]),
                CoTransfection(name="b", units=make_units("b", ERNS), fluo_bias=BIAS_FLUO, ratios=[NumRange(min=1, max=2) for _ in range(8)]),
            ],
        )

    print("Running ratio range bounds test...")
    test_ratio_range_bounds(lib_instance, recipe_instance)
    print("✓ Ratio range bounds test passed\n")

    print("Running source output slots test...")
    test_source_output_slots(lib_instance)
    print("✓ Source output slots test passed\n")

    print("Running shared source network structure test...")
    test_shared_source_network_structure(lib_instance)
    print("✓ Shared source network structure test passed\n")

    # Run parametrized roundtrip tests on all recipes
    print("Running roundtrip tests on all recipe variants...\n")

    with LibraryContext.with_library(lib_instance):
        u1 = Slot(part=["1w_uORF", "2x_uORF"], ref_id="U1")
        u2 = Slot(part=[None, "4x_uORF", "3x_uORF"], ref_id="U2")
        recipes = {
            "complex_twolayers": Recipe(
                name="two_and_one",
                content=[
                    CoTransfection(name="x1", units=make_units("x1", ERNS), ratios=[NumRange(min=1, max=10) for _ in range(8)]),
                    CoTransfection(name="x2", units=make_units("x2", ERNS), ratios=[NumRange(min=2, max=100) for _ in range(8)]),
                    CoTransfection(name="b", units=make_units("b", ERNS), fluo_bias=BIAS_FLUO, ratios=[NumRange(min=1, max=2) for _ in range(8)]),
                ],
            ),
            "complex_twolayers_locked": Recipe(
                name="two_and_one_locked",
                content=[
                    CoTransfection(name="x1", units=make_units("x1", ERNS), ratios=[1.0 + i * 0.5 for i in range(8)]),
                    CoTransfection(name="x2", units=make_units("x2", ERNS), ratios=[2.0 + i * 2.0 for i in range(8)]),
                    CoTransfection(name="b", units=make_units("b", ERNS), fluo_bias=BIAS_FLUO, ratios=[1.0 + i * 0.1 for i in range(8)]),
                ],
            ),
            "complex_twolayers_mixed": Recipe(
                name="two_and_one_mixed",
                content=[
                    CoTransfection(name="x1", units=make_units("x1", ERNS), ratios=[NumRange(min=1, max=10) if i % 2 == 0 else 2.0 + i for i in range(8)]),
                    CoTransfection(name="x2", units=make_units("x2", ERNS), ratios=[5.0, NumRange(min=2, max=50), 10.0, NumRange(min=5, max=100), 15.0, 20.0, NumRange(min=10, max=80), 25.0]),
                    CoTransfection(name="b", units=make_units("b", ERNS), fluo_bias=BIAS_FLUO, ratios=[1.0, 1.2, NumRange(min=1, max=2), 1.4, 1.5, NumRange(min=1, max=1.8), 1.6, 1.7]),
                ],
            ),
            "simple_two_reporters": Recipe(
                name="simple_two_reporters",
                content=[
                    CoTransfection(
                        units=[
                            TranscriptionUnit(slots=["cHS4", "hEF1a", "eBFP2", "L0.T_4560"]),
                            TranscriptionUnit(slots=["cHS4", "hEF1a", "mMaroon1", "L0.T_4560"]),
                        ],
                        ratios=[0.833, 0.167],
                    )
                ],
            ),
            "simple_two_reporters_unlocked": Recipe(
                name="simple_two_reporters_unlocked",
                content=[
                    CoTransfection(
                        units=[
                            TranscriptionUnit(slots=["cHS4", "hEF1a", "eBFP2", "L0.T_4560"]),
                            TranscriptionUnit(slots=["cHS4", "hEF1a", "mMaroon1", "L0.T_4560"]),
                        ],
                        ratios=[NumRange(min=0.5, max=0.9), NumRange(min=0.1, max=0.5)],
                    )
                ],
            ),
            "unlocked_ratios": Recipe(
                name="unlocked_ratios",
                content=[
                    CoTransfection(
                        units=[
                            TranscriptionUnit(slots=["cHS4", "hEF1a", "CasE_rec", "eBFP2", "L0.T_4560"], source="p1"),
                            TranscriptionUnit(slots=["cHS4", "hEF1a", "CasE", "L0.T_4560"], source="p2"),
                            TranscriptionUnit(slots=["cHS4", "hEF1a", "mNeonGreen", "L0.T_4560"], source="p3"),
                        ],
                        ratios=[NumRange(min=0.2, max=0.5), 0.3, NumRange(min=0.1, max=0.3)],
                    )
                ],
            ),
            "shared_source": Recipe(
                name="shared_source",
                content=[
                    CoTransfection(
                        units=[
                            TranscriptionUnit(slots=["cHS4", "hEF1a", u1, "CasE_rec", "eBFP2", "L0.T_4560"], source="plsmd1"),
                            TranscriptionUnit(slots=["cHS4", "hEF1a", "CasE", "L0.T_4560"], source="plsmd1"),
                            TranscriptionUnit(slots=["cHS4", "hEF1a", u2, "Csy4_rec", "eYFP", "L0.T_4560"], source="out2_plsmd"),
                            TranscriptionUnit(slots=["cHS4", "hEF1a", "Csy4", "L0.T_4560"], source="ern2_plsmd"),
                            TranscriptionUnit(slots=["mKO2"], source="mrkr_plsmd"),
                        ],
                        ratios=[i + 1 for i in range(4)],
                    )
                ],
            ),
            "shared_source_unlocked": Recipe(
                name="shared_source_unlocked",
                content=[
                    CoTransfection(
                        units=[
                            TranscriptionUnit(slots=["cHS4", "hEF1a", u1, "CasE_rec", "eBFP2", "L0.T_4560"], source="plsmd1"),
                            TranscriptionUnit(slots=["cHS4", "hEF1a", "CasE", "L0.T_4560"], source="plsmd1"),
                            TranscriptionUnit(slots=["cHS4", "hEF1a", u2, "Csy4_rec", "eYFP", "L0.T_4560"], source="out2_plsmd"),
                            TranscriptionUnit(slots=["cHS4", "hEF1a", "Csy4", "L0.T_4560"], source="ern2_plsmd"),
                            TranscriptionUnit(slots=["mKO2"], source="mrkr_plsmd"),
                        ],
                        ratios=[NumRange(min=1, max=5), NumRange(min=2, max=8), NumRange(min=1, max=4), NumRange(min=1, max=3)],
                    )
                ],
            ),
            "shared_source_mixed": Recipe(
                name="shared_source_mixed",
                content=[
                    CoTransfection(
                        units=[
                            TranscriptionUnit(slots=["cHS4", "hEF1a", u1, "CasE_rec", "eBFP2", "L0.T_4560"], source="plsmd1"),
                            TranscriptionUnit(slots=["cHS4", "hEF1a", "CasE", "L0.T_4560"], source="plsmd1"),
                            TranscriptionUnit(slots=["cHS4", "hEF1a", u2, "Csy4_rec", "eYFP", "L0.T_4560"], source="out2_plsmd"),
                            TranscriptionUnit(slots=["cHS4", "hEF1a", "Csy4", "L0.T_4560"], source="ern2_plsmd"),
                            TranscriptionUnit(slots=["mKO2"], source="mrkr_plsmd"),
                        ],
                        ratios=[NumRange(min=1, max=5), 2.5, NumRange(min=1, max=4), 1.8],
                    )
                ],
            ),
        }

    for recipe_name, recipe in recipes.items():
        print(f"Running roundtrip tests for {recipe_name}...")
        test_commit_and_export(recipe)
        test_rebuild_structure(recipe)
        test_ratios_preserved(recipe)
        test_outputs_match(recipe)
        print(f"✓ All roundtrip tests passed for {recipe_name}\n")

    print("All tests passed!")
