## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     import and init     --
# ···············································································
from jax.tree_util import Partial as partial
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, value_and_grad
import biocomp as bc
import scriptutils as ut
from pathlib import Path
import json5
import json
import sqlite3
from tqdm import tqdm
import pandas as pd
import numpy as np

lib = ut.getLibFromGoogleSheet()

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

base_path = Path("/Users/jeandisset/Dropbox (MIT)/Biocomp/")
xp_path = base_path / "Experiments"
recipe_path = base_path / "Recipes"
experiments = [x.name for x in xp_path.iterdir() if x.is_dir()]

xp = bc.XP(experiments[0], xp_path, recipe_path, lib)
models = xp.get_models(inverse=True)
X, Y = xp.get_XY(models)


cfg = {
    "learning_rate": 0.001,
    "adam_w_decay": 0.01,
    "clipping": 0.001,
    "n_replicates": 10,
    "initial_param_scaling": 0.01,
    "normalize_data": False,
    "epochs": 100,
    "log_rate": 50,
    "rng_key": 1,
}

sample = 'CoTX-All'
model = models[sample]
x, y = X[sample], Y[sample]

params_hist, loss_hist = bc.train_single_model(model, x, y, cfg)


##
# plot all losses. They are stored in a list where each element is a point in history for each replicate
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(loss_hist)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
# limits to top 10% of losses
ax.set_ylim(np.min(loss_hist), np.percentile(loss_hist, 90))
plt.show()

min_losses_per_replicate = np.array(loss_hist).min(axis=0)
min_losses_per_replicate

