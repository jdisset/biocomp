"""Generate golden file for test_complex_twolayers_builds_and_runs

Run this script to create or update the golden reference file.
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

from biocomp.library import LibraryContext, load_lib
from biocomp.network import recipe_to_networks
from biocomp.compute import ComputeStack
from biocomp.recipe import Recipe, CoTransfection, FluoIntensity, NumRange
from biocomp.config import SIMPLE_NODES_COMPUTE_CONFIG
import biocomp.biorules as br

# Import the make_units function and other dependencies from test file
import sys

sys.path.insert(0, str(Path(__file__).parent))
from test_complex_twolayers_computation import make_units, COLORS, ERNS


def main():
    lib = load_lib()

    # Build the recipe directly
    erns = ERNS
    ern_names = ", ".join(erns)
    with LibraryContext.with_library(lib):
        recipe = Recipe(
            name=f"two_and_one ({ern_names})",
            content=[
                CoTransfection(name="x1", units=make_units("x1", erns=erns)),
                CoTransfection(name="x2", units=make_units("x2", erns=erns)),
                CoTransfection(
                    name="b",
                    units=make_units("b", erns=erns),
                    fluo_bias=FluoIntensity(
                        tu_id=0,
                        value=NumRange(min=0.3, max=0.6),
                        protein=COLORS["b"],
                        units="Rescaled AU",
                    ),
                ),
            ],
        )

        networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True)
        stack = ComputeStack([networks[0]])
        stack.build(config=SIMPLE_NODES_COMPUTE_CONFIG)

        # Use fixed seed for reproducibility
        fixed_key = jax.random.PRNGKey(12345)
        params = stack.init(fixed_key)

        # Create fixed inputs (size based on actual number of input nodes, not bias nodes)
        nb_inputs = networks[0].nb_inputs
        inputs = jnp.ones((nb_inputs,)) if nb_inputs > 0 else jnp.array([])
        n_random_vars = params["global/number_of_random_variables"]
        random_vars = jax.random.normal(fixed_key, (n_random_vars,))

        # Run forward pass
        stack_result, aux = stack.apply(params, inputs, random_vars, fixed_key)

        # Save to golden file
        golden_dir = Path(__file__).parent / "golden_files"
        golden_dir.mkdir(exist_ok=True)
        golden_path = golden_dir / "complex_twolayers_output.npz"

        np.savez(
            golden_path,
            stack_result=np.array(stack_result),
            aux_loss=np.array(aux.get("loss", 0.0)) if aux and "loss" in aux else np.array(0.0),
            nb_inputs=networks[0].nb_inputs,
            output_shape=np.array(stack_result.shape),
        )

        print(f"Golden file saved to: {golden_path}")
        print(f"  stack_result: {stack_result}")
        print(f"  output shape: {stack_result.shape}")
        print(f"  nb_inputs: {networks[0].nb_inputs}")


if __name__ == "__main__":
    main()
