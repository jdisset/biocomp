"""Test for get_reordered_protein_names behavior with bias inputs"""

import pytest
from biocomp.network import Network, CoTransfection, Unit
from biocomp.plotutils import get_reordered_protein_names


def test_get_reordered_protein_names_with_bias():
    """Test that get_reordered_protein_names correctly handles bias inputs and only_dependent_outputs"""
    
    # Setup constants
    P = "hEF1a"
    T = "L0.T_4560"
    ERNS = ['CasE', 'Csy4', 'PgU']
    
    COLORS = {
        'x1': 'mKO2',
        'x2': 'eBFP2', 
        'x3': 'iRFP720',
        'b': 'mMaroon1',
        'y': 'mNeonGreen',
    }
    
    erns = ERNS
    recs = [f"{ern}_rec" for ern in erns]
    
    # Create network
    n1 = Network(
        cotx=[
            CoTransfection(
                name="TNFa",
                units=[
                    Unit(slots=[P, COLORS['x1'], T], name="x1_marker"),
                    Unit(slots=[P, recs[0], COLORS['y'], T], name="x1_a+"),
                ],
            ),
            CoTransfection(
                name="INFab",
                units=[
                    Unit(slots=[P, COLORS['x2'], T], name="x2_marker"),
                    Unit(slots=[P, erns[0], T], name="x2_a-"),
                ],
            ),
            CoTransfection(
                name="B",
                units=[
                    Unit(slots=[P, COLORS['b'], T], name="ba_marker"),
                    Unit(slots=[P, erns[0], T], name="b_a-"),
                ],
            ),
        ],
        invert_on_build=True,
    )
    
    # Build the network
    n1.build()
    
    # Test 1: Before setting bias, with only_dependent_outputs=True (default)
    input_order, output_pos, input_names, output_name = get_reordered_protein_names(n1)
    
    # Debug prints
    print(f"All outputs: {n1.get_output_proteins(only_dependent_outputs=False)}")
    print(f"Dependent outputs: {n1.get_output_proteins(only_dependent_outputs=True)}")
    print(f"Inverted inputs: {n1.get_inverted_input_proteins()}")
    print(f"Input names from get_reordered: {input_names}")
    print(f"Output name from get_reordered: {output_name}")
    
    # With invert_on_build=True, mKO2, eBFP2, mMaroon1 are inverted inputs
    # Only mNeonGreen is a dependent output
    assert len(input_names) == 3
    assert 'mKO2' in input_names
    assert 'eBFP2' in input_names
    assert 'mMaroon1' in input_names
    assert output_name == 'mNeonGreen'  # Single output returned as string
    
    # Test 2: Before setting bias, with only_dependent_outputs=False
    input_order_all, output_pos_all, input_names_all, output_names_all = get_reordered_protein_names(
        n1, only_dependent_outputs=False
    )
    
    # Should have 3 inputs, but now all 4 proteins are in the outputs
    assert len(input_names_all) == 3
    # With 4 outputs, it should return a list
    assert isinstance(output_names_all, list)
    assert len(output_names_all) == 4
    assert 'mNeonGreen' in output_names_all
    assert 'mMaroon1' in output_names_all
    assert 'mKO2' in output_names_all
    assert 'eBFP2' in output_names_all
    
    # Test 3: Set mMaroon1 as bias
    n1.set_input_as_bias('mMaroon1')
    
    # Test 4: After setting bias, with only_dependent_outputs=True (default)
    input_order_bias, output_pos_bias, input_names_bias, output_name_bias = get_reordered_protein_names(n1)
    
    print(f"\nAfter setting bias:")
    print(f"Inverted inputs (no bias): {n1.get_inverted_input_proteins(include_biases=False)}")
    print(f"Inverted inputs (with bias): {n1.get_inverted_input_proteins(include_biases=True)}")
    print(f"Input names from get_reordered: {input_names_bias}")
    
    # Now only 2 inputs (mKO2, eBFP2) since mMaroon1 is a bias
    # Still 1 dependent output (mNeonGreen)
    assert len(input_names_bias) == 2
    assert 'mKO2' in input_names_bias
    assert 'eBFP2' in input_names_bias
    assert 'mMaroon1' not in input_names_bias  # No longer an input
    assert output_name_bias == 'mNeonGreen'
    
    # Test 5: After setting bias, with only_dependent_outputs=False
    input_order_all_bias, output_pos_all_bias, input_names_all_bias, output_names_all_bias = get_reordered_protein_names(
        n1, only_dependent_outputs=False
    )
    
    # Should have 2 inputs (bias excluded) and 4 outputs (all proteins)
    assert len(input_names_all_bias) == 2
    assert 'mMaroon1' not in input_names_all_bias
    assert isinstance(output_names_all_bias, list)
    assert len(output_names_all_bias) == 4
    assert 'mNeonGreen' in output_names_all_bias
    assert 'mMaroon1' in output_names_all_bias  # Still in outputs
    
    # Test 6: Verify output positions are correct
    all_outputs = n1.get_output_proteins(only_dependent_outputs=False)
    dependent_outputs = n1.get_output_proteins(only_dependent_outputs=True)
    
    # Verify mNeonGreen is the only dependent output
    assert dependent_outputs == ['mNeonGreen']
    
    # Check that mNeonGreen position matches
    mneongreen_pos_in_all = all_outputs.index('mNeonGreen')
    assert output_pos_bias == mneongreen_pos_in_all
    
    # Test 7: Verify that bias affects inverted inputs but not outputs
    assert 'mMaroon1' in n1.get_inverted_input_proteins(include_biases=True)
    assert 'mMaroon1' not in n1.get_inverted_input_proteins(include_biases=False)
    assert 'mMaroon1' in n1.get_output_proteins(only_dependent_outputs=False)


if __name__ == "__main__":
    test_get_reordered_protein_names_with_bias()
    print("All tests passed!")