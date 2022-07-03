## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     imports     --
# ···············································································
from jax.tree_util import Partial as partial
from jax import tree_util as pytree
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad, jit, vmap, lax
from jax.scipy.signal import convolve
import numpy as np
import jax
import dcell as dc

# magic ipython to reload modules
# %load_ext autoreload
# %autoreload 2

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


