import numpy as np

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{               --     template + generator for TUs     --
# ···············································································

# usage:
# ERN_template = bc.TranscriptionUnit(
# [
# bc.Slot(any_promoter),
# bc.Slot(any_uorf),
# bc.Slot(random_ERN_rec),
# bc.Slot(random_ERN),
# bc.Part('NeonGreen'),
# ]
# )
# ERN_template.resolve_all_slots(lib, random_seed=3)


def any_promoter(lib, **_):
    all_promoters = lib.pc[lib.pc.category == "promoter"].index.tolist()
    return all_promoters


def any_uorf(lib, **_):
    all_uORFs = lib.pc[lib.pc.category == "uORF"].index.tolist()
    return all_uORFs + [None]


# picks a randmo ern_rec, and ensure that it is not for an ERN that's already in the L1
def random_ERN_rec(lib, rdm_key, l1, **_):
    all_sequestrons = lib.sequestrons[lib.sequestrons.type == "ERN"]
    already_in_l1 = []
    for s in l1.slots:
        if s.is_resolved and s.part in all_sequestrons["negative_part"].values:
            already_in_l1.append(s.part)
    possible_recog = all_sequestrons[~all_sequestrons["negative_part"].isin(already_in_l1)][
        "positive_part"
    ].values.tolist()

    if already_in_l1:
        possible_recog = possible_recog + [None]

    return possible_recog[np.random.randint(0, len(possible_recog))]


# picks a random ern, and ensure that it is not for an ERN_rec that's already in the L1
def random_ERN(lib, rdm_key, l1, **_):
    all_sequestrons = lib.sequestrons[lib.sequestrons.type == "ERN"]
    already_in_l1 = []
    for s in l1.slots:
        if s.is_resolved and s.part in all_sequestrons["positive_part"].values:
            already_in_l1.append(s.part)
    possible_ern = all_sequestrons[~all_sequestrons["positive_part"].isin(already_in_l1)][
        "negative_part"
    ].values.tolist()

    if already_in_l1:
        possible_ern = possible_ern + [None]

    return possible_ern[np.random.randint(0, len(possible_ern))]


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
