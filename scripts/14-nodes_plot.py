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


# the goal is to plot what's happening inside the nodes.
# I think stacking collect_all_results with vmap is the way to go.
# This way we can have a scatter plot of the functions in the ranges
# where they are actually being used.
# Then later I guess it can be nice to also plot the function inbetween
# the training points, and probably a bit outside, to get a sense of how it
# generalizes.

# We'll work with Georg's current data and try to learn a full nn with fused inverter
xp = ut.load_xp('20221012A_massCtrls', lib)


cfg = {
    "learning_rate": 0.01,
    "compile_training": True,
    "node_remap": {
        "sequestron_ERN": "ERN_nn_multi",
        "transcription": "transcription_nn",
        "inv_transcription": "inverse_transcription_nn",
        "translation": "translation_nn",
        "inv_translation": "inverse_translation_nn",
        "output": "output_nn",
    },
    "balance_bin_resolution": 0.5,
    "balance_threshold_quantile": 0.4,
    "balance_threshold_min": 40,
    "batch_size": 1024,
    "rng_key": random.randint(0, 1e12),
}
