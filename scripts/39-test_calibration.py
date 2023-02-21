### {{{                          --     imports     --
import biocomp as bc
from biocomp import datautils as du
from jax.scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from biocomp.calibration import Calibration
import scriptutils as ut
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import gaussian_kde
import jax
import optax
from jax import jit, vmap, value_and_grad
from tqdm import tqdm
##────────────────────────────────────────────────────────────────────────────}}}

### {{{                    --     first call to calib     --

data_dir = ut.DEFAULT_XP_PATH / '2023-01-22_CasE_ALLuORFs/data/raw_data_gated/'

blank = data_dir / 'color_controls/CNTL.2023-01-22_CasE_ALLuORFs.csv'

controls = {
    'eYFP': data_dir / 'color_controls/eYFP.2023-01-22_CasE_ALLuORFs.csv',
    'eBFP2': data_dir / 'color_controls/EBFP2.2023-01-22_CasE_ALLuORFs.csv',
    'mKate': data_dir / 'color_controls/mKate.2023-01-22_CasE_ALLuORFs.csv',
    ('eYFP', 'eBFP2', 'mKate'): data_dir / 'color_controls/ALL.2023-01-22_CasE_ALLuORFs.csv',
}

beads = Path(data_dir / 'beads/2023-01-22_CasE_ALLuORFs_BEADS_AL01_017.fcs')

cal = Calibration(
    blanks_file=blank,
    color_controls_files=controls,
    beads_file=beads,
    reference_protein='EYFP',
    reference_channel='FITC-A',
    beads_reference_values = bc.calibration.SPHEROTECH_RCP_30_5a,
    use_channels=['FITC-A', 'PACIFIC_BLUE_A', 'PE_TEXAS_RED_A'],
)

cal.channel_to_unit
cal.unit_to_channel
cal._Calibration__fluo_proteins
cal.controls.keys()
cal._Calibration__channel_order
cal.beads_reference_values

cal.fit()

# cal.plot_diagnostics()

cal._Calibration__spectral_signature_matrix

cal.plot_spectral_diagnostics()
# 7.188e-2
# 7.18762e-2
loss = cal._Calibration__losses
fig, ax = plt.subplots()
plt.plot(loss)
plt.yscale('log')

cal._Calibration__controls_masks
cal._Calibration__controls_values


##────────────────────────────────────────────────────────────────────────────}}}

### {{{             --     experimenting with log correction     --

def spectral_signature_estimation_gd_log(
    Y,
    M,
    normalize_to_prot_chan,
    max_iterations=5,
    max_n=10000,
    max_learning_rate=0.5,
    jax_seed=0,
):

    # same as above but using gradient descent, which allows to use much bigger datasets
    # and add fancy regularization and weightings. We still use the closed form version for init.
    # TODO: weighted Alternating Least Square

    if len(Y) > max_n:  # resample Y and M to get only max_n
        choice = jax.random.choice(jax.random.PRNGKey(jax_seed), len(Y), (min(max_n, len(Y)),))
        Y, M = Y[choice], M[choice]

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=max_learning_rate,
        warmup_steps=0.25 * max_iterations,
        decay_steps=0.85 * max_iterations,
        end_value=1e-3,
    )

    S = jax.random.uniform(jax.random.PRNGKey(0), (M.shape[1], Y.shape[1]), minval=0.1, maxval=1)
    S /= S[normalize_to_prot_chan]  # normalize S to the desired protein and channel

    rescale_factor = jnp.quantile(Y, 0.95)
    Y = Y / rescale_factor

    K = Y @ jnp.linalg.pinv(S)
    K = jnp.average(K, axis=1, weights=M+1e-6)
    K = jnp.ones(Y.shape[0])
    K = jnp.ones((Y.shape[0], M.shape[1]))

    A = jnp.zeros(Y.shape[1])  # additive term
    other_params = {'A': A, 'b': 0.0, 'a': 1.1}

    # hmmm... there has to be a better way than initializing 3 optimizers hahaha
    optS = optax.adam(learning_rate=schedule)
    optK = optax.adam(learning_rate=schedule)
    optO = optax.adam(learning_rate=schedule)
    stateS, stateK, state = optS.init(S), optK.init(K), optO.init(other_params)

    EPS = 1e-6
    P,C = normalize_to_prot_chan
    def loss_single_row(yi, ki, mi, S, A, b, a):
        ki = (ki / ki[P]) * yi[C]
        gamma = (ki * mi) @ S + A
        yhat = a * jnp.log10(jnp.clip(gamma,EPS)) + b
        yi - jnp.log10(jnp.clip(yi,EPS))
        return jnp.mean((yhat - yi)**2)


    def lS(S, K, other):
        A, b, a = other['A'], jnp.maximum(other['b'],0.1), jnp.maximum(other['a'],0.1)
        # s = S / S[normalize_to_prot_chan]
        s = S
        err = vmap(loss_single_row, in_axes=(0, 0, 0, None, None, None, None))(Y, K, M, s, A, b, a)
        return jnp.mean(err)

    def lK(K, S, other):
        return lS(S, K, other)

    def lO(other_params, S, K):
        return lS(S, K, other_params)

    def half_update(a, b, c, lossf, opt, state):
        loss, g = jax.value_and_grad(lossf)(a, b, c)
        update, state = opt.update(g, state, a)
        a = optax.apply_updates(a, update)
        return a, state, loss

    i=0
    yi, ki, mi = Y[i], K[i], M[i]
    A, b, a = other_params['A'], jnp.maximum(other_params['b'],0), jnp.maximum(other_params['a'],0)
    loss_single_row(yi, ki, mi, S, A, b, a)
    jax.value_and_grad(lS)(S, K, other_params)
    jax.value_and_grad(lK)(K, S, other_params)
    jax.value_and_grad(lO)(other_params, S, K)
    print(f'Params: {other_params}, S={S}, K={K[:5]}')

    @jit
    def update(s, k, other, stateS, stateK, stateO):
        s, stateS, lossS = half_update(s, k, other, lS, optS, stateS)
        k, stateK, lossK = half_update(k, s, other, lK, optK, stateK)
        other, stateO, lossother = half_update(other, s, k, lO, optO, stateO)
        return s, k, other, stateS, stateK, stateO, lossS + lossK + lossother

    losses = []
    pbar = tqdm(range(max_iterations), desc='Spectral signature estimation')
    SPrev = S / S[normalize_to_prot_chan]
    for i in pbar:
        S, K, other_params, stateS, stateK, state, loss = update(S, K, other_params, stateS, stateK, state)
        Snormalized = S / S[normalize_to_prot_chan]
        sdist = jnp.linalg.norm(Snormalized - SPrev) / jnp.linalg.norm(Snormalized)
        losses.append(loss)
        pbar.set_description(f'Spectral signature estimation (loss={loss:.5e}, update={sdist:.2e})')
        SPrev = Snormalized

    # S = S / S[normalize_to_prot_chan]

    final_params = {'A': other_params['A'], 'b': other_params['b'], 'a': other_params['a'], 'K': K, 'S': S}

    return final_params, losses

from jax.config import config
config.update("jax_debug_nans", False)

refprot_id = cal._Calibration__fluo_proteins.index(cal.reference_protein)
refchan_id = cal._Calibration__channel_order.index(cal.reference_channel)
Y = cal._Calibration__controls_values
M = cal._Calibration__controls_masks
normalize_to_prot_chan=(refprot_id, refchan_id)
params, losses = spectral_signature_estimation_gd_log(Y, M, normalize_to_prot_chan, max_iterations=1000)
params
params

plt.figure()
plt.plot(losses)
plt.xlabel('iteration')
plt.yscale('log')
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
bc.calibration.plot_spectral_sig(params['S'], ax, cal._Calibration__fluo_proteins, cal._Calibration__channel_order)

def get_xi(yi, a, b, A, S):
    EPS = 1e-6
    logyi = jnp.log10(jnp.maximum(EPS, yi))
    log_gamma = (logyi - b) / a
    gamma = 10**log_gamma
    sinv = jnp.linalg.pinv(S)
    print(f'gamma={gamma}, A={A}, S={S}), sinv={sinv}, log_gamma={log_gamma}')
    xi = (gamma - A) @ sinv
    return xi

xi = get_xi(Y[0], params['a'], params['b'], params['A'], params['S'])
xi

X = vmap(get_xi, in_axes=(0, None, None, None, None))(Y, params['a'], params['b'], params['A'], params['S'])
X



##────────────────────────────────────────────────────────────────────────────}}}

### {{{                           --     plots     --

def remove_axis_and_spines(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

def plot_fluo_distribution(ax, data, res=1000, log10max = 6, axtitle=None):
    logdata = jnp.log10(1+jnp.maximum(0, data)) / log10max
    xrange = 1.0
    XX = jnp.linspace(-0.05, xrange, res)
    kde = gaussian_kde(logdata.T, bw_method=0.01)
    smoothkde = gaussian_kde(logdata.T, bw_method=0.1)
    densities = kde(XX.T)
    densities = densities * 0.01
    ax.fill_between(XX, 0, densities, color='k', alpha=0.25, lw=0)
    ax.plot(XX, densities, color='k', alpha=1, lw=0.5)
    ax.set_yscale("log")
    # ax.set_aspect("equal")
    ax.set_xlim(-0.05, xrange)
    ax.set_ylim(0.001, 1)
    # use real data for y ticks
    ax.set_xticks(XX[::(res//5)])
    ax.set_xticklabels([f'{10**x:.1e}' for x in (XX * log10max)[::(res//5)]])
    # remove_axis_and_spines(ax)
    if axtitle is not None:
        ax.set_title(axtitle)

def fluo_histogram(ax, data, axtitle=None):
    MIN = -1e5
    MAX = 1e6
    bins = np.linspace(MIN, MAX, 100)
    ax.hist(data, bins=bins, color='k', alpha=0.5)
    ax.set_yscale('log')
    ax.set_xlim(MIN, MAX)
    ax.set_ylim(1, 1e5)
    if axtitle is not None:
        ax.set_title(axtitle)



cal.controls.keys()
blank_df = cal.controls[tuple()]
bfp_df = cal.controls[('EBFP2',)]
yfp_df = cal.controls[('EYFP',)]
mkate_df = cal.controls[('MKATE',)]
all_df = cal.controls[('EYFP', 'EBFP2', 'MKATE')]
len(all_df)
len(bfp_df)
len(yfp_df)
len(mkate_df)
len(blank_df)
np.median(cal.blanks.values, axis=0)
# Pdf = cal.controls[('EYFP', 'EBFP2', 'MKATE')]
df = all_df
# df = bfp_df
# df = mkate_df
# df = yfp_df

Y = df.values
Y = cal.to_MEF(Y)
channels = df.columns
X = cal.apply(Y)

fluonames = cal._Calibration__fluo_proteins
# cal._Calibration__autofluorescence


fig, axes = du.mkfig(3, 1, (8,3))
for i, f in enumerate(channels):
    plot_fluo_distribution(axes[i], Y[:,i], axtitle=f)
    # fluo_histogram(axes[i], Y[:,i], axtitle=f)
fig.suptitle('Raw data')
fig.tight_layout()


fig, axes = du.mkfig(3, 1, (8,3))
for i, f in enumerate(fluonames):
    plot_fluo_distribution(axes[i], X[:,i], axtitle=f)
    #fluo_histogram(axes[i], X[:,i], axtitle=f)

fig.suptitle('Calibrated protein counts')
fig.tight_layout()

##────────────────────────────────────────────────────────────────────────────}}}


# OK well things are not working as expected because I'm not sure why 
# the mapping between channel has to be in log space... in the original model
# I assumed each channel is a linear combination of the protein spectra
# however color mapping requires a log space ax+b transform. 
# Why is that? Is there some kind of log-space op-amp in the machine?
# Time is running out as Charles is going to get some xp results soon
# So I'll just use the simple tasbe protocol with my beads mapping for now
# and come back to this later
