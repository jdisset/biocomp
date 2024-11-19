# {{{                          --     imports     --
# ···············································································
import jax
import jax.numpy as jnp
from matplotlib import scale as mscale
from functools import partial
from scipy.spatial import cKDTree
from jax import jit, vmap
import numpy as np
from biocomp import utils as ut
from biocomp import datautils as du
from biocomp import compute as cmp
from biocomp.datautils import DataManager
import matplotlib.pyplot as plt
from jax.scipy.stats import gaussian_kde
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import difflib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import string
from labellines import labelLine, labelLines
from jax.typing import ArrayLike
from typing import Tuple
import os
from typing import Union, Sequence, List, Tuple, Dict, Any, Optional, Callable
from matplotlib.ticker import ScalarFormatter, NullFormatter, MaxNLocator
from matplotlib import colors as mcolors
from pkg_resources import resource_filename
from . import plotting_core as pc
from .plotting_core import (
    DEFAULT_CMAP_NAME,
    setup_transformed_axis,
    get_reordered_protein_names,
    network_ticks_and_labels,
    make_xy_grid,
    knn_avg,
    get_knn_quantile,
    format_powers,
    apply_style,
    heatmap,
)

NdArray = Union[np.ndarray, jnp.ndarray]
configurable = pc.configurable
##────────────────────────────────────────────────────────────────────────────}}}
# ---- density histograms

