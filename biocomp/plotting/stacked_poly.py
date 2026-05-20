# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Stacked polynomial fitting (from calibrie.utils).

Canonical source: calibrie/calibrie/utils.py
Copied here to avoid calibrie's heavy __init__ import chain (dearpygui).
"""

from functools import partial

import jax.numpy as jnp
from jax import jit, vmap


def _gaussian_kernel(x, mean, std, scaler=1e12):
    return scaler * jnp.exp(-((x - mean) ** 2) / (2 * std**2))


@partial(jit, static_argnames=("degree",))
def _fit_coeffs(x, y, w, mticks, stds, degree=1):
    weights = vmap(_gaussian_kernel, in_axes=(None, 0, 0))(x, mticks, stds)
    weights = weights * w[None, :]

    def pfit(x, y, w):
        ww = jnp.where(w.sum() > 0, w, jnp.ones_like(w) * 1e-12)
        return jnp.polyfit(x, y, deg=degree, w=ww)

    return vmap(pfit, in_axes=(None, None, 0))(x, y, weights)


@jit
def evaluate_stacked_poly(x, params):
    coeffs, mticks, stds = params
    evals = vmap(jnp.polyval, in_axes=(0, None))(coeffs, x)
    eval_weights = vmap(_gaussian_kernel, in_axes=(None, 0, 0))(x, mticks, stds)
    EPS = 1e-9
    eval_weights = eval_weights.at[0].set(
        jnp.where(x < mticks[0], jnp.clip(eval_weights[0], EPS, None), eval_weights[0])
    )
    eval_weights = eval_weights.at[-1].set(
        jnp.where(x > mticks[-1], jnp.clip(eval_weights[-1], EPS, None), eval_weights[-1])
    )
    return jnp.average(evals, weights=eval_weights, axis=0)


@partial(jit, static_argnames=("degree",))
def fit_stacked_poly_at_quantiles(x, y, w, quantiles, degree=1):
    mticks = jnp.quantile(x, quantiles)
    diff = jnp.pad(jnp.diff(mticks), (1, 1), mode="edge")
    stds = (diff[:-1] + diff[1:]) / 2
    coeffs = _fit_coeffs(x, y, w, mticks, stds, degree=degree)
    return (coeffs, mticks, stds)
