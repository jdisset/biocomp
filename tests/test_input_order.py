"""Tests for input_order functionality in Recipe and Network

input_order allows explicit control over the order of input proteins
in the network, ensuring proper topological ordering regardless of
the natural order determined by graph traversal.
"""

import pytest
import numpy as np
from biocomp.recipe import CoTransfection, TranscriptionUnit, Slot, Recipe
from biocomp.library import load_lib, LibraryContext
from biocomp.network import recipe_to_networks


@pytest.fixture
def lib():
    return load_lib()


@pytest.fixture
def two_input_recipe(lib):
    """Recipe with two input cotx groups that become two input proteins after inversion"""
    with LibraryContext.with_library(lib):
        return Recipe(
            name="two_inputs",
            content=[
                CoTransfection(
                    name="cotx1",
                    units=[
                        TranscriptionUnit(
                            name="eBFP2_reporter",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="eBFP2"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                    ],
                ),
                CoTransfection(
                    name="cotx2",
                    units=[
                        TranscriptionUnit(
                            name="mKO2_reporter",
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="mKO2"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                    ],
                ),
            ],
        )


@pytest.fixture
def three_input_recipe(lib):
    """Recipe with three input cotx groups"""
    with LibraryContext.with_library(lib):
        return Recipe(
            name="three_inputs",
            content=[
                CoTransfection(
                    name="cotx1",
                    units=[
                        TranscriptionUnit(
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="eBFP2"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                    ],
                ),
                CoTransfection(
                    name="cotx2",
                    units=[
                        TranscriptionUnit(
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="mKO2"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                    ],
                ),
                CoTransfection(
                    name="cotx3",
                    units=[
                        TranscriptionUnit(
                            slots=[
                                Slot(part="cHS4"),
                                Slot(part="hEF1a"),
                                Slot(part="mNeonGreen"),
                                Slot(part="L0.T_4560"),
                            ],
                        ),
                    ],
                ),
            ],
        )


# =============================================================================
# Basic input_order tests
# =============================================================================


def test_input_order_reverses_two_inputs(lib, two_input_recipe):
    """Test that input_order reverses the input order"""
    with LibraryContext.with_library(lib):
        # first build without input_order to see natural order
        networks_natural = recipe_to_networks(two_input_recipe, invert=True)
        net_natural = networks_natural[0]
        natural_order = net_natural.get_inverted_input_proteins()
        assert len(natural_order) == 2

        # now build with reversed input_order
        reversed_order = list(reversed(natural_order))
        two_input_recipe.input_order = reversed_order
        networks_reordered = recipe_to_networks(two_input_recipe, invert=True)
        net_reordered = networks_reordered[0]

        actual_order = net_reordered.get_inverted_input_proteins()
        assert actual_order == reversed_order
        assert actual_order != natural_order


def test_input_order_three_inputs(lib, three_input_recipe):
    """Test input_order with three inputs"""
    with LibraryContext.with_library(lib):
        # build without input_order to see natural order
        networks = recipe_to_networks(three_input_recipe, invert=True)
        net = networks[0]
        natural_order = net.get_inverted_input_proteins()
        assert len(natural_order) == 3

        # specify explicit order: [2, 0, 1]
        desired_order = [natural_order[2], natural_order[0], natural_order[1]]
        three_input_recipe.input_order = desired_order
        networks_reordered = recipe_to_networks(three_input_recipe, invert=True)
        net_reordered = networks_reordered[0]

        actual_order = net_reordered.get_inverted_input_proteins()
        assert actual_order == desired_order


def test_input_order_stored_in_metadata(lib, two_input_recipe):
    """Test that input_order is stored in network metadata"""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(two_input_recipe, invert=True)
        net = networks[0]
        natural_order = net.get_inverted_input_proteins()

        two_input_recipe.input_order = list(reversed(natural_order))
        networks_reordered = recipe_to_networks(two_input_recipe, invert=True)
        net_reordered = networks_reordered[0]

        assert net_reordered.get_input_order() == two_input_recipe.input_order


def test_input_order_no_effect_without_inversion(lib, two_input_recipe):
    """Test that input_order has no effect when invert=False"""
    with LibraryContext.with_library(lib):
        two_input_recipe.input_order = ["should", "not", "matter"]
        # this should not raise, input_order is ignored when invert=False
        networks = recipe_to_networks(two_input_recipe, invert=False)
        assert len(networks) == 1


# =============================================================================
# Validation tests
# =============================================================================


def test_input_order_validation_duplicates():
    """Test that duplicate proteins in input_order are rejected"""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="duplicates"):
        Recipe(
            name="test",
            content=[],
            input_order=["protein_A", "protein_A", "protein_B"],
        )


def test_input_order_validation_missing_protein(lib, two_input_recipe):
    """Test that missing proteins in input_order raise an error"""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(two_input_recipe, invert=True)
        net = networks[0]
        natural_order = net.get_inverted_input_proteins()

        # only include one of the two input proteins
        two_input_recipe.input_order = [natural_order[0]]

        with pytest.raises(AssertionError, match="missing"):
            recipe_to_networks(two_input_recipe, invert=True)


def test_input_order_validation_extra_protein(lib, two_input_recipe):
    """Test that extra proteins in input_order raise an error"""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(two_input_recipe, invert=True)
        net = networks[0]
        natural_order = net.get_inverted_input_proteins()

        # add an extra protein that's not an input
        two_input_recipe.input_order = natural_order + ["FAKE_PROTEIN"]

        with pytest.raises(AssertionError, match="extra"):
            recipe_to_networks(two_input_recipe, invert=True)


# =============================================================================
# get_input_from_output tests
# =============================================================================


def test_input_order_affects_get_input_from_output(lib, two_input_recipe):
    """Test that input_order affects get_input_from_output correctly"""
    with LibraryContext.with_library(lib):
        # build without input_order
        networks_natural = recipe_to_networks(two_input_recipe, invert=True)
        net_natural = networks_natural[0]
        natural_order = net_natural.get_inverted_input_proteins()

        # create dummy output array where each column has unique values
        n_outputs = net_natural.nb_outputs
        output_arr = np.arange(n_outputs).reshape(1, n_outputs).astype(float)

        # get input columns in natural order
        input_natural = net_natural.get_input_from_output(output_arr)
        assert input_natural.shape == (1, 2)

        # now with reversed input_order
        reversed_order = list(reversed(natural_order))
        two_input_recipe.input_order = reversed_order
        networks_reordered = recipe_to_networks(two_input_recipe, invert=True)
        net_reordered = networks_reordered[0]

        input_reordered = net_reordered.get_input_from_output(output_arr)

        # the reordered input columns should be reversed from natural
        assert np.allclose(input_reordered[0, 0], input_natural[0, 1])
        assert np.allclose(input_reordered[0, 1], input_natural[0, 0])


# =============================================================================
# Network.apply_input_order direct tests
# =============================================================================


def test_apply_input_order_directly(lib, two_input_recipe):
    """Test calling apply_input_order directly on a network"""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(two_input_recipe, invert=True)
        net = networks[0]
        natural_order = net.get_inverted_input_proteins()

        # apply reversed order directly
        reversed_order = list(reversed(natural_order))
        net.apply_input_order(reversed_order)

        assert net.get_inverted_input_proteins() == reversed_order
        assert net.get_input_order() == reversed_order


def test_apply_input_order_idempotent(lib, two_input_recipe):
    """Test that applying the same order twice gives same result"""
    with LibraryContext.with_library(lib):
        networks = recipe_to_networks(two_input_recipe, invert=True)
        net = networks[0]
        natural_order = net.get_inverted_input_proteins()

        # apply same order twice
        net.apply_input_order(natural_order)
        order_after_first = net.get_inverted_input_proteins()

        net.apply_input_order(natural_order)
        order_after_second = net.get_inverted_input_proteins()

        assert order_after_first == order_after_second == natural_order
