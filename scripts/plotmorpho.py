from tqdm import tqdm
import matplotlib.transforms as mtransforms
from pathlib import Path

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                     --     load and evaluate     --
# ···············································································
bestparams = ut.load(f'../__out/morpho/liverlobule/params.pickle')

key = jax.random.PRNGKey(cfg.rng_key)
finalstate, (statehist, percepthist) = run_acc(bestparams, cfg.steps_per_run, key)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


def do_plot(ax, Z, title, cmap, vmax, transform, connector=True):
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
    ax.text(w * 0.5, y1 + textPad, title, horizontalalignment='center', transform=trans_data)
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


def persp_stack(title, titles, slices, cmap, outputfile=None, show=True):
    overlap = 0.2
    fig, ax = setup_fig(title)
    w = slices[0].shape[0]
    fig.set_size_inches(len(slices) * 2.3, 6)
    # mtransforms.Affine2D().scale(1, 1)
    for ii, (s, t) in enumerate(zip(reversed(slices), reversed(titles))):
        i = len(slices) - ii - 1
        im = do_plot(
            ax,
            s,
            t,
            cmap,
            None,
            mtransforms.Affine2D()
            .scale(1, 1.5)
            .skew_deg(0, 20)
            .translate(w * (1 - overlap) * i, 0),
        )
    ax.set_xlim(-0.5 * w, (len(slices) - overlap) * w)
    if outputfile is not None:
        plt.savefig(outputfile, dpi=150)
    if show:
        plt.show()
    plt.close()


# plot state history
slices = [
    [statehist[frame, :, :, :4]]
    + [statehist[frame, :, :, 3] * statehist[frame, :, :, i] for i in range(3, statehist.shape[3])]
    for frame in range(64)
]
titles = ['Fluo (R,G,B)', 'Alive', 'Divide'] + [
    f'Morpho {i-5}' for i in range(5, statehist.shape[3])
]
outpath = Path('../__out/morpho/liverlobule/2022-06-20/')
outpath.mkdir(parents=True, exist_ok=True)
[
    persp_stack('Channels', titles, slices[i], 'inferno', outpath / f'{i}')
    for i in range(len(statehist))
]

## plot perception history

perceptions = [
    [percepthist[frame, :, :, i] for i in range(percepthist.shape[3])] for frame in range(64)
]
percept_titles = [f'Morpho {i}' for i in range(percepthist.shape[3])]
outpath = Path('../__out/morpho/liverlobule/2022-06-20/perceptions_contact')
outpath.mkdir(parents=True, exist_ok=True)
[
    persp_stack('Perception layers', percept_titles, perceptions[i], 'inferno', outpath / f'{i}')
    for i in range(len(perceptions))
]

print('\nDone')

