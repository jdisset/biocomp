import jax
import optax
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad
from jax.tree_util import Partial as partial
from jax.scipy.stats import gaussian_kde
from ott.geometry.pointcloud import PointCloud
from ott.problems.linear.linear_problem import LinearProblem
from ott.solvers.linear.sinkhorn import Sinkhorn

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import flowio


def spectral_signature_estimation(
    Y,
    M,
    normalize_to_prot_chan=None,
    max_iterations=2,
    max_n=10000,
    jax_seed=0,
    progress_bar=False,
):
    # Solves with Alternating Least Square using closed form solutions (pinv)

    # model: Y = KM.S + A
    # Y: observations
    # M: masks (from controls: single color = (1,0,...))
    # S: spectral signature
    # K: some latent "proxy" for the true quantity of protein.
    # A: intercept + noise

    key = jax.random.PRNGKey(jax_seed)

    if normalize_to_prot_chan is not None:
        normalize = lambda x: x / x[normalize_to_prot_chan]
    else:
        normalize = lambda x: x / (jnp.maximum(jnp.max(x, axis=1)[:, None], 1e-19))

    if len(Y) > max_n:  # resample Y and M to get only max_n
        choice = jax.random.choice(jax.random.PRNGKey(jax_seed), len(Y), (min(max_n, len(Y)),))
        Y, M = Y[choice], M[choice]

    K = jax.random.uniform(key, (M.shape[0],), minval=0.1, maxval=1)
    A = jnp.zeros((Y.shape[1],))

    @jit
    def alsq(K, A):  # one iteration of alternating least squares
        S = normalize(jnp.linalg.pinv(K[:, None] * M) @ (Y - A))  # find S from K
        K = (Y - A) @ jnp.linalg.pinv(S)  # find K from S
        K = jnp.average(K, axis=1, weights=M)
        K = jnp.nan_to_num(K, nan=0)
        K = jnp.maximum(K, 0)  # K should be positive
        A = Y - (K[:,None] * M) @ S  # find A from K and S
        A = jnp.average(A, axis=0)
        return S, K, A

    if progress_bar:
        pbar = tqdm(range(max_iterations), desc='Spectral signature estimation')
    else:
        pbar = range(max_iterations)

    Sprev = 0
    for _ in pbar:
        S, K, A = alsq(K, A)
        Y_norm = jnp.linalg.norm(Y)
        error = (((Y - A) - (K[:, None] * M) @ S) ** 2).mean() / Y_norm
        print(f'error: {error:.2e}')
        print(f'update: {jnp.mean((Sprev - S)**2):.2e}')
        Sprev = S

    return S, K, A


def affine_gd(Y, ref_channel, num_iter=100, learning_rate=0.1, max_N=500000):

    NCHAN = Y.shape[1]
    if len(Y) > max_N:
        choice = jax.random.choice(jax.random.PRNGKey(0), len(Y), (min(max_N, len(Y)),))
        Y = Y[choice]

    params = {
        'a': jnp.ones((NCHAN,)),
        'b': jnp.zeros((NCHAN,)),
    }

    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    Yref = Y[:, ref_channel]
    # only take the 90% middle quantile
    lowq, highq = jnp.quantile(Y, 0.1, axis=0), jnp.quantile(Y, 0.9, axis=0)
    weights = jnp.where((Y > lowq) & (Y < highq), 1, 0)

    @jax.value_and_grad
    def lossf(params):
        yhat = params['a'] * Y + params['b']
        err = (yhat - Yref[:, None])**2
        err = err * weights
        return jnp.mean(err ** 2)

    @jit
    def update(params, opt_state):
        loss, grad = lossf(params)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    losses = []
    for i in range(0, num_iter):
        params, opt_state, loss = update(params, opt_state)
        losses.append(loss)
        print(f'iter {i}: loss = {loss:.2e}')

    return params




### {{{                     --     calibration functions     --


def spectral_signature_estimation(
    Y,
    M,
    normalize_to_prot_chan,
    max_iterations=10,
    max_n=10000,
    jax_seed=0,
    progress_bar=False,
):
    # Solves with Alternating Least Square using closed form solutions (pinv)

    # model: Y = KMS
    # Y: observations
    # M: masks (from controls: single color = (1,0,...), all colors = (1,1,...))
    # S: spectral signature
    # K: some latent "proxy" for the true quantity of protein.
    # K could be diagonal in a perfect world but the matrix factorization algorithm
    # will certainly exploit every indices. In any case we don't really care about K
    # as long as it allows us to estimate S, which is what we are really interested in.

    if len(Y) > max_n:  # resample Y and M to get only max_n
        choice = jax.random.choice(jax.random.PRNGKey(jax_seed), len(Y), (min(max_n, len(Y)),))
        Y, M = Y[choice], M[choice]

    # Initialize S with random positive values and K as identity
    S = jax.random.uniform(jax.random.PRNGKey(0), (M.shape[1], Y.shape[1]), minval=0.1, maxval=1)
    S /= S[normalize_to_prot_chan]  # normalize S to the desired protein and channel
    K = jnp.identity(
        Y.shape[0]
    )  # Identity is a good start for K (spoiler: it won't end as a diagonal matrix)

    @jit
    def alsq(S, K):  # one iteration of alternating least squares
        S = jnp.linalg.pinv(K @ M) @ Y  # find S from K
        S /= S[normalize_to_prot_chan]  # normalize S to the desired protein and channel
        K = Y @ jnp.linalg.pinv(M @ S)  # find K from S
        return S, K

    if progress_bar:
        pbar = tqdm(range(max_iterations), desc='Spectral signature estimation')
    else:
        pbar = range(max_iterations)

    for _ in pbar:
        S, K = alsq(S, K)
        # error = ((Y - K @ M @ S) ** 2).mean() / Y_norm

    return S, K


def spectral_signature_estimation_gd_log(
    Y,
    M,
    normalize_to_prot_chan,
    max_iterations=2500,
    max_n=100000,
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

    # initialize with a quick OLS (way faster!)
    S, _ = spectral_signature_estimation(
        Y,
        M,
        normalize_to_prot_chan=normalize_to_prot_chan,
        max_iterations=10,
        max_n=5000,
        jax_seed=jax_seed,
    )

    S = jax.random.uniform(jax.random.PRNGKey(0), (M.shape[1], Y.shape[1]), minval=0.1, maxval=1)
    S /= S[normalize_to_prot_chan]  # normalize S to the desired protein and channel

    rescale_factor = jnp.quantile(Y, 0.95)
    Y = Y / rescale_factor

    K = Y @ jnp.linalg.pinv(S)
    K = jnp.average(K, axis=1, weights=M)

    A = jnp.zeros(Y.shape[1])  # additive term
    other_params = {'A': A, 'b': 0.0, 'a': 1.0}

    # hmmm... there has to be a better way than initializing 3 optimizers hahaha
    optS = optax.adam(learning_rate=schedule)
    optK = optax.adam(learning_rate=schedule)
    optO = optax.adam(learning_rate=schedule)
    stateS, stateK, state = optS.init(S), optK.init(K), optO.init(other_params)

    def loss_single_row(yi, ki, mi, S, A, b, a):
        return jnp.mean(((ki * mi @ S) ** a - jnp.divide((yi - A), 10**b)) ** 2)

    def lS(S, K, other):
        A, b, a = other['A'], jnp.maximum(other['b'], 0), jnp.maximum(other['a'], 0)
        s = S / S[normalize_to_prot_chan]
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
        S, K, other_params, stateS, stateK, state, loss = update(
            S, K, other_params, stateS, stateK, state
        )
        Snormalized = S / S[normalize_to_prot_chan]
        sdist = jnp.linalg.norm(Snormalized - SPrev) / jnp.linalg.norm(Snormalized)
        losses.append(loss)
        pbar.set_description(f'Spectral signature estimation (loss={loss:.5e}, update={sdist:.2e})')
        SPrev = Snormalized

    S = S / S[normalize_to_prot_chan]
    # A = A * sigma + mu
    A = A * rescale_factor

    final_params = {'A': A, 'b': other_params['b'], 'a': other_params['a'], 'K': K, 'S': S}

    return final_params, losses


def spectral_signature_estimation_gd_w_a(
    Y,
    M,
    normalize_to_prot_chan,
    max_iterations=1500,
    max_n=100000,
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
        end_value=1e-4,
    )

    # initialize with a quick OLS (way faster!)
    S, _ = spectral_signature_estimation(
        Y,
        M,
        normalize_to_prot_chan=normalize_to_prot_chan,
        max_iterations=10,
        max_n=5000,
        jax_seed=jax_seed,
    )

    # S = jax.random.uniform(jax.random.PRNGKey(0), (M.shape[1], Y.shape[1]), minval=0.1, maxval=1)
    S /= S[normalize_to_prot_chan]  # normalize S to the desired protein and channel

    A = jnp.zeros(Y.shape[1])  # additive term

    rescale_factor = jnp.quantile(Y, 0.95)
    Y = Y / rescale_factor

    # mu, sigma = jnp.mean(Y), jnp.std(Y)
    # Y = (Y - mu) / sigma

    K = Y @ jnp.linalg.pinv(S)
    K = jnp.average(K, axis=1, weights=M)

    # hmmm... there has to be a better way than initializing 3 optimizers hahaha
    optS = optax.adam(learning_rate=schedule)
    optK = optax.adam(learning_rate=schedule)
    optA = optax.adam(learning_rate=schedule)
    stateS, stateK, stateA = optS.init(S), optK.init(K), optA.init(A)

    def loss_single_row(yi, ki, mi, A, S):
        return jnp.mean((ki * mi @ S - (yi - A)) ** 2)

    def lS(S, K, A):
        s = S / S[normalize_to_prot_chan]
        err = vmap(loss_single_row, in_axes=(0, 0, 0, None, None))(Y, K, M, A, s)
        return jnp.mean(err)

    def lK(K, S, A):
        return lS(S, K, A)

    def lA(A, S, K):
        return lS(S, K, A)

    def half_update(a, b, c, lossf, opt, state):
        loss, g = jax.value_and_grad(lossf)(a, b, c)
        update, state = opt.update(g, state, a)
        a = optax.apply_updates(a, update)
        return a, state, loss

    @jit
    def update(s, k, a, stateS, stateK, stateA):
        s, stateS, lossS = half_update(s, k, a, lS, optS, stateS)
        k, stateK, lossK = half_update(k, s, a, lK, optK, stateK)
        a, stateA, lossA = half_update(a, s, k, lA, optA, stateA)
        return s, k, a, stateS, stateK, stateA, lossS + lossK + lossA

    losses = []
    pbar = tqdm(range(max_iterations), desc='Spectral signature estimation')
    SPrev = S / S[normalize_to_prot_chan]
    for i in pbar:
        S, K, A, stateS, stateK, stateA, loss = update(S, K, A, stateS, stateK, stateA)
        Snormalized = S / S[normalize_to_prot_chan]
        sdist = jnp.linalg.norm(Snormalized - SPrev) / jnp.linalg.norm(Snormalized)
        losses.append(loss)
        pbar.set_description(f'Spectral signature estimation (loss={loss:.5e}, update={sdist:.2e})')
        SPrev = Snormalized

    S = S / S[normalize_to_prot_chan]
    # A = A * sigma + mu
    A = A * rescale_factor

    return S, A, losses


def spectral_signature_estimation_gd(
    Y,
    M,
    normalize_to_prot_chan,
    max_iterations=1500,
    max_n=100000,
    max_learning_rate=1,
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
        warmup_steps=0.15 * max_iterations,
        decay_steps=0.85 * max_iterations,
        end_value=1e-5,
    )

    # initialize with a quick OLS (way faster!)
    S, _ = spectral_signature_estimation(
        Y,
        M,
        normalize_to_prot_chan=normalize_to_prot_chan,
        max_iterations=10,
        max_n=5000,
        jax_seed=jax_seed,
    )

    Y = (Y - jnp.mean(Y)) / jnp.std(Y)

    K = Y @ jnp.linalg.pinv(S)
    K = jnp.average(K, axis=1, weights=M)

    optS = optax.adam(learning_rate=schedule)
    optK = optax.adam(learning_rate=schedule)
    stateS, stateK = optS.init(S), optK.init(K)

    def loss_single_row(yi, ki, mi, S):
        return jnp.mean((ki * mi @ S - yi) ** 2)

    def lS(S, K):
        s = S / S[normalize_to_prot_chan]
        err = vmap(loss_single_row, in_axes=(0, 0, 0, None))(Y, K, M, s)
        return jnp.mean(err)

    def lK(K, S):
        return lS(S, K)

    def half_update(a, b, lossf, opt, state):
        loss, g = jax.value_and_grad(lossf)(a, b)
        update, state = opt.update(g, state, a)
        a = optax.apply_updates(a, update)
        return a, state, loss

    @jit
    def update(s, k, stateS, stateK):
        s, stateS, lossS = half_update(s, k, lS, optS, stateS)
        k, stateK, lossK = half_update(k, s, lK, optK, stateK)
        return s, k, stateS, stateK, lossS + lossK

    def has_converged(losses, NWIN=100):
        if len(losses) < NWIN:
            return False
        mean = jnp.mean(losses[-NWIN:])
        return jnp.mean(jnp.abs(jnp.array(losses[-NWIN:]) - mean)) / mean < 1e-3

    losses = []
    pbar = tqdm(range(max_iterations), desc='Spectral signature estimation')
    SPrev = S / S[normalize_to_prot_chan]
    for i in pbar:
        S, K, stateS, stateK, loss = update(S, K, stateS, stateK)
        Snormalized = S / S[normalize_to_prot_chan]
        sdist = jnp.linalg.norm(Snormalized - SPrev) / jnp.linalg.norm(Snormalized)
        losses.append(loss)
        pbar.set_description(f'Spectral signature estimation (loss={loss:.5e}, update={sdist:.2e})')
        SPrev = Snormalized

    S = S / S[normalize_to_prot_chan]

    return S, K, losses


##────────────────────────────────────────────────────────────────────────────}}}

def fit(self, **kwargs):
    """
    Fit the calibration parameters to the controls.
    """
    # print('Estimating autofluorescence...')
    # self.__autofluorescence = np.median(self.blanks.values, axis=0)

    # first we find the parameters of the log10 transform
    maxv = max(jnp.max(self.__controls_values), jnp.max(self.beads.values)) * 1.01
    self.__log_scale_factor = jnp.log10(maxv)

    print('Computing peaks assignment...')
    calib_values = jnp.array(
        [self.beads_reference_values[self.channel_to_unit[c]] for c in self.__channel_order]
    ).T
    logcalib = logtransform(calib_values, self.__log_scale_factor)
    self.__logbeads = logtransform(jnp.clip(self.beads.values, 1), self.__log_scale_factor)
    self.__beads_peaks, (self.__beads_densities, self.__beads_vmat) = compute_peaks(
        self.__logbeads, logcalib
    )

    print('Computing channel calibration...')
    self.__beads_params, self.__beads_loss = beads_fit(self.__beads_peaks, logcalib)

    print('Estimating spectral signature...')
    refprot_id = self.__fluo_proteins.index(self.reference_protein)
    refchan_id = self.__channel_order.index(self.reference_channel)

    Y = self.to_MEF(self.__controls_values)
    # Y = self.__controls_values

    (
        self.__spectral_signature_matrix,
        self.__autofluorescence,
        self.__losses,
    ) = spectral_signature_estimation_gd_w_a(
        Y, self.__controls_masks, normalize_to_prot_chan=(refprot_id, refchan_id)
    )

    self.__fitted = True

def apply(self, Y):
    X = (self.to_MEF(Y) - self.__autofluorescence) @ jnp.linalg.pinv(
        self.__spectral_signature_matrix
    )
    return X


def plot_spectral_diagnostics(self):
    if self.__spectral_signature_matrix is None:
        raise ValueError('You must fit the calibration first')
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    plot_spectral_sig(
        self.__spectral_signature_matrix, ax, self.__fluo_proteins, self.__channel_order
    )
