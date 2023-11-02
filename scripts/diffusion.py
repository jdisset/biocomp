## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     imports     --
# ···············································································

from scipy.interpolate import interp1d
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.transforms as mtransforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from scriptutils import ddict
import sys


import jax
from jax import vmap, jit, lax
from jax import tree_util as pytree
from jax.tree_util import Partial as partial
import jax.numpy as jnp
import jax.scipy as jsp


#                                                                            }}}
## ════════════════════════════════════════════════════════════════════════════

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                   --     Parameters & config     --
# ···············································································
# units : mm, minutes

fixedTime = bool(int(sys.argv[2]))
well = sys.argv[1]

# morphogen declaration
morpho = ddict()
morpho.dx = 0.01
morpho.dy = morpho.dx
morpho.diffusion_rate = 1.0
morpho.decay_rate = 0.01


dx2, dy2 = morpho.dx * morpho.dx, morpho.dy * morpho.dy
Dmax = morpho.diffusion_rate
dt = np.min(dx2 * dy2 / (2.0 * Dmax * (dx2 + dy2)))

nsteps = int(tend / dt) + 1

#                                                                            }}}
## ════════════════════════════════════════════════════════════════════════════



#                                                                            }}}
## ════════════════════════════════════════════════════════════════════════════

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                  --     Building cell matrices     --
# ···············································································
filterIntensity = 0.01


compg = pd.read_csv('../data/cell_data.csv')
xbins = np.linspace(0, params['plate_w'], nx + 1, endpoint=True)
ybins = np.linspace(0, params['plate_h'], ny + 1, endpoint=True)
scaling = (
    params['plate_w'] / np.max(compg['cell_x'] + 1),
    params['plate_h'] / np.max(compg['cell_y'] + 1),
)


def getHist2d(time, well, color):
    h = compg[
        (compg['intensity'] > filterIntensity)
        & (compg['well'] == well)
        & (compg['color'] == color)
        & (compg['time'] == time)
    ]
    allpos = np.array([h['cell_x'] * scaling[0], h['cell_y'] * scaling[1]])
    R, _, _ = np.histogram2d(x=allpos[0, :], y=allpos[1, :], bins=(xbins, ybins))
    return R.transpose()


# we interpolate the position matrices over time


MAX_TIME = max(compg['time'])

allR = np.array([getHist2d(t, well, 'Green').flatten() for t in range(8)])
linfitR = interp1d(list(range(MAX_TIME + 1)), allR, 'cubic', axis=0)

allS = np.array([getHist2d(t, well, 'Red').flatten() for t in range(8)])
linfitS = interp1d(list(range(MAX_TIME + 1)), allS, 'cubic', axis=0)


def getReceiverCells(time):
    t = MAX_TIME * time / tend
    return linfitR(t).reshape(nx, ny)


def getSenderCells(time):
    t = MAX_TIME * time / tend
    return linfitS(t).reshape(nx, ny)


#                                                                            }}}
## ════════════════════════════════════════════════════════════════════════════

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                     --     stepping functions   --
# ···············································································


def diffuse(U, D, xx, yy, dt):
    U_next = U.copy()
    U_next[1:-1, 1:-1] = U[1:-1, 1:-1] + D * dt * (
        (U[2:, 1:-1] - 2 * U[1:-1, 1:-1] + U[:-2, 1:-1]) / xx
        + (U[1:-1, 2:] - 2 * U[1:-1, 1:-1] + U[1:-1, :-2]) / yy
    )

    # boundary conditions:
    U_next[0, :] = U[1, :]
    U_next[-1, :] = U[-2, :]
    U_next[:, 0] = U[:, 1]
    U_next[:, -1] = U[:, -2]
    return U_next


#                                                                            }}}
## ════════════════════════════════════════════════════════════════════════════

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     Simulation     --
# ···············································································

# plt.imshow(S, cmap='Reds')
# plt.title('initital senders positions in well ' + well)
# plt.show()
# plt.imshow(R, cmap='Greens')
# plt.title('initital receivers positions in well ' + well)
# plt.show()


P0 = np.zeros((nx, ny))  # initial phloretin concentration
Y0 = np.zeros((nx, ny))  # initial YFP concentration
A = np.ones((nx, ny))  # initial Phloretic Acid concentration

P = [P0]
Y = [Y0]

save_every_n = max(1, int(nsteps / 1000))


S = getSenderCells(0)
R = getReceiverCells(0)

if fixedTime:
    S = getSenderCells(tend)
    R = getReceiverCells(tend)

Spos = [S]
Rpos = [R]
for i in tqdm(range(nsteps)):
    # Phloretin
    Pprev = P[-1]
    Pnext = diffuse(Pprev, params['PDiffusion'], dx2, dy2, dt)  # diffusion
    AN1 = A ** params['np']
    Pnext += np.multiply(S, np.divide(params['Bp'] * AN1, params['Kp'] + AN1)) * dt
    Pnext -= params['PDecay'] * Pprev * dt  # decay
    # YFP
    Yprev = Y[-1]
    Ynext = Yprev.copy()
    PN2 = Pprev ** params['ny']
    Ynext = (
        np.multiply(
            R,
            np.divide(params['By'] * PN2, params['Ky'] + PN2)
            - params['YDecay'] * Yprev
            + params['YLeak'],
        )
        * dt
    )

    if i % save_every_n == 0:
        if not fixedTime:
            S = getSenderCells(i * dt)
            R = getReceiverCells(i * dt)
        Spos.append(S)
        Rpos.append(R)
        P.append(Pnext)
        Y.append(Ynext)
    else:
        P[-1] = Pnext
        Y[-1] = Ynext

#                                                                            }}}
## ════════════════════════════════════════════════════════════════════════════

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     Plot Functions    --
# ···············································································
# FP colormaps


def addTranspCurve(cmap, name='', baseTransp=0.3):
    N = 300
    transpCurve = 1.0 - np.geomspace(1, baseTransp, N) + baseTransp  # alpha
    c = cmap(np.linspace(0, 1, N, endpoint=True))
    c[:, 3] = transpCurve
    return LinearSegmentedColormap.from_list(name, c)


GFP = addTranspCurve(plt.get_cmap('Greens'), 'GFP')
RFP = addTranspCurve(plt.get_cmap('Reds'), 'RFP')
BFP = addTranspCurve(plt.get_cmap('Blues'), 'BFP')


def do_plot(ax, Z, t, cmap, vmax, transform, connector=True):
    im = ax.imshow(Z, interpolation='none', cmap=cmap, aspect=1, vmin=0, vmax=vmax)
    trans_data = transform + ax.transData
    im.set_transform(trans_data)
    x1, x2, y1, y2 = im.get_extent()
    w = x2 - x1
    h = y1 - y2
    textPad = 0.1 * h
    if connector:
        textPad = 0.4 * h
        ax.plot(
            [x1, x2, x2, x1, x1],
            [y1, y1, y2, y2, y1],
            "-",
            color='grey',
            transform=trans_data,
            linewidth=0.3,
        )
        ax.plot(
            [w * 0.5, w * 0.5],
            [y1 + 0.05 * h, y1 + 0.3 * h],
            ":",
            color='grey',
            transform=trans_data,
            linewidth=1,
        )
    ax.text(w * 0.5, y1 + textPad, t, horizontalalignment='center', transform=trans_data)
    return im


def setup_fig(title):
    fig, ax = plt.subplots(1, 1)
    fig.patch.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.suptitle(title)
    return fig, ax


def timelapse_persp(Q, title, interv, cmap, outputfile=None, show=True):
    overlap = 0.2
    vmax = np.max(Q)
    ids = np.floor(np.array(interv) * (len(P) - 1)).astype(int)
    slices = np.array(Q)[ids]
    fig, ax = setup_fig(title)
    w, _ = slices[0].shape
    fig.set_size_inches(len(slices) * 2.3, 6)
    for ii, s in enumerate(reversed(slices)):
        i = len(slices) - ii - 1
        im = do_plot(
            ax,
            s,
            '{:.0f} hours'.format(interv[i] * tend),
            cmap,
            vmax,
            mtransforms.Affine2D()
            .scale(1, 1.5)
            .skew_deg(0, 20)
            .translate(w * (1 - overlap) * i, 0),
        )
    ax.set_xlim(-0.5 * w, (len(slices) - overlap) * w)
    cbar_ax = fig.add_axes([0.85, 0.28, 0.015, 0.5])
    cb = fig.colorbar(im, cax=cbar_ax, drawedges=False)
    cb.outline.set_linewidth(0)
    cb.ax.locator_params(nbins=5)
    if outputfile is not None:
        plt.savefig(outputfile, dpi=150)
    if show:
        plt.show()
    plt.close()


def timelapse_movie(Q, title, cmap, nframes, outputdir='./outputplots'):
    outputdir.rstrip('/')
    Path(outputdir).mkdir(parents=True, exist_ok=True)
    interv = np.linspace(0, 1, nframes, endpoint=True)
    vmax = np.max(Q)
    ids = np.floor(np.array(interv) * (len(P) - 1)).astype(int)
    slices = np.array(Q)[ids]
    for i, s in tqdm(enumerate(slices)):
        fig, ax = setup_fig(title)
        fig.set_size_inches(7, 5)
        im = do_plot(
            ax,
            s,
            '{:.0f} hours'.format(interv[i] * tend),
            cmap,
            vmax,
            mtransforms.Affine2D().scale(1, 1),
            connector=False,
        )
        cbar_ax = fig.add_axes([0.85, 0.28, 0.015, 0.5])
        cb = fig.colorbar(im, cax=cbar_ax, drawedges=False)
        cb.outline.set_linewidth(0)
        cb.ax.locator_params(nbins=5)
        fig.savefig(outputdir + '/' + str(i) + '.png', dpi=150)
        plt.close()


#                                                                            }}}
## ════════════════════════════════════════════════════════════════════════════
