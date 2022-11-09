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


# I guess we can distinguish 2 generation mode:
# 1. PLasmid level: we parse a library of available plasmids to generate a recipe (a set of coTx plasmids).
#    in which case the optimisation is done only on ratios and copy numbers, and the exploration is just trying different subsets.
# 2. L0 + constraint level: we have a set of L0 + some generation rules (aka constraints) to generate valid plasmids.
#    We can further refine this mode in 2 sub-approaches:
#       a. Resolved TUs, where we generate a set of valid plasmids, then switch to mode 1 to optimise the ratios and copy numbers.
#       b. Full optim with unresolved TUs, where we deal with differentiable quantizable parameters
#         that can vary without modifying the topology of a network directly

# Let's focus on mode 1 for now, then we'll see about mode 2.a (and 2.b later, probably).
# Most of the complexity and code for all modes has already been written so nothing should be too insane.

# WELL, jk. The main problem is that generating graphs from TUs is super slow and not jittable.
# So, 2.b is the way because by parameterizing the non-topology related parts (uORfs, promoters) we can explore so much so quickly.


def any_promoter(lib, *_, **__):
    all_promoters = lib.pc[lib.pc.category == 'promoter'].index.tolist()
    return [all_promoters]


def any_uorf(lib, *_, **__):
    all_uORFs = lib.pc[lib.pc.category == 'uORF_group'].index.tolist()
    return [all_uORFs + [None]]


def any_ERN(lib, others, **_):
    erns = lib.sequestrons[lib.sequestrons.type == 'ERN']
    already_in_tu = []
    for s in others:
        if s.part in erns['positive_part'].values.tolist():
            already_in_tu.append(s.part)

    # we also filter out the ones with a positive part that end with a digit (all of Georg's modified affinity stuff):
    possible_ern = list(
        set(
            erns[
                ~erns['positive_part'].isin(already_in_tu)
                & ~erns['positive_part'].str.endswith(tuple('0123456789'))
            ]['negative_part'].values.tolist()
        )
    )

    return possible_ern


def any_ERN_rec(lib, others, **_):
    all_sequestrons = lib.sequestrons[lib.sequestrons.type == 'ERN']
    already_in_tu = []

    for s in others:
        if s.part in all_sequestrons['negative_part'].values.tolist():
            already_in_tu.append(s.part)

    possible_rec = all_sequestrons[~all_sequestrons['negative_part'].isin(already_in_tu)][
        'positive_part'
    ].values.tolist()

    p = set(possible_rec)
    # remove the ones that end with a number (all of Georg's modified affinity stuff)
    p = [x for x in p if not x[-1].isdigit()]
    return list(p)


def random_seed():
    return random.randint(0, 2**32)


def part(name):
    return lambda *_, **__: [name]


any_promoter(lib)

ERN_template = bc.TranscriptionUnitGenerator(
    [
        any_promoter,
        any_uorf,
        any_ERN_rec,
        any_ERN,
        part('NeonGreen'),
    ]
)

from tqdm import tqdm
from jax import jit
from jax import random

# Let's try to generate a TU with a specific ERN

available_TUs = list(ERN_template.generate_all(lib))
available_TUs
