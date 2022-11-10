## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     imports     --
# ···············································································
import biocomp as bc
import biocomp.compute as bcc
import scriptutils as ut
import jax
import random
import jax.numpy as jnp

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

lib = ut.load_lib()

recipe = {
    'name': "justbfp",
    'content': [
        {'sources': [{'plasmid': "pAK0022"}]},
        # { 'sources': [{ 'plasmid': "pGW0010" }] }
    ],
}


n = bc.recipe.network_from_recipe(recipe, lib)
inv_n = bc.network.inverted_network(n)

bc.network.fuse_consecutive(inv_n.compute_graph, ("inv_translation", "inv_transcription"), "inv_fused")

inv_n.compute_graph
n.compute_graph

n.set_numeric_as_input()
model = bc.ComputeGraphModel(n)
model.build()

inv_model = bc.ComputeGraphModel(inv_n)
inv_model.build()




rng_key = jax.random.PRNGKey(0)
params, constraints = model.init(rng_key)

model(params, jnp.array([1.0]), rng_key=rng_key)

inv_model.collect_all_results(params, jnp.array([1.0]), rng_key=rng_key)


def inv_fused_nn(get_param, get_quantized, transform_name, wsize=64, depth=2, **_):
    def apply(value, rng_key):
            k0, k1, k2 = jax.random.split(rng_key, 3)
            rate_name = f'{transform_name}_rate'
            deg_param_name = f'{transform_name}_deg'
    return apply



from functools import partial

partial(inv_fused_nn, wsize=64, depth=2) # is what we'd like to use as a remap function.
# but we need to be able to write this as a config line.

##
config = {
    'node_impl': {
        'inv_fused': partial(inv_fused_nn, wsize=64, depth=2),
        'inv_fused2': inv_fused_nn,
    }
}

# to serialize this, we need to be able to serialize partials.
# we can do this by using the __name__ attribute of the function
# and the __dict__ attribute of the partial.

def serialize_partial(part):
    return {
        'function': part.func.__name__,
        'kwargs': part.keywords,
    }

def serialize_function(func):
    return {
        'function': func.__name__,
    }

serialize_partial(partial(inv_fused_nn, wsize=64, depth=2))

# print the whole config object, using serialize_partial and serialize_function
# to do so, we need to select the right function for each node_impl.

def serialize_partial_or_function(field):
    if isinstance(field, partial):
        return serialize_partial(field)
    return serialize_function(field)


import json
json.dumps(config, 







