## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     import and init     --
# ···············································································
from jax.tree_util import Partial as partial
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, value_and_grad
from pathlib import Path
import json5
import sqlite3
from tqdm import tqdm
import pandas as pd
import optax

import numpy as np
from .recipe import XP, import_recipes_to_sql
from .network import Network, inverted_network
import wandb as wb
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────



def mse_loss(y, y_hat):
    return jnp.mean((y - y_hat)**2)

def wandb_update(loss, params, iter_num):
    wb.log({'loss': loss}, step=iter_num)
    wb.log({'params': params}, step=iter_num)


def train_single_model(model, X, Y, cfg, loss_f=mse_loss, wandb=None):
    optimizer = optax.chain(
        optax.adaptive_grad_clip(cfg['clipping']),
        optax.adamw(learning_rate=cfg['learning_rate'], weight_decay=cfg['adam_w_decay']),
    )


    def loss_func(params, x, y, rng_key):
        m = partial(model, params, rng_key=rng_key)
        y_hat = vmap(m)(x).squeeze()
        return loss_f(y, y_hat)

    def logger_update(loss, params, iter_num):
        if wandb:
            wandb_update(loss, params, iter_num)
        else:
            print(f'[{iter_num}/{cfg["epochs"]}] loss: {loss}')


    def training_step(params, opt_state, key, x, y):
        loss, grads = jax.value_and_grad(loss_func)(params, x, y, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, grads, loss


    key = jax.random.PRNGKey(cfg['rng_key'])
    initkeys = jax.random.split(key, cfg['n_replicates'])

    params = vmap(model.init)(initkeys)
    opt_states = vmap(optimizer.init)(params)

    step = jit(vmap(partial(training_step, x=X, y=Y)))

    if wandb:
        wb.init(config=cfg, project=wandb, entity="jdisset", reinit=True)

    params_history = []
    loss_history = []

    for i, k in enumerate(jax.random.split(key, cfg['epochs'])):
        keys = jax.random.split(k, cfg['n_replicates'])
        params, opt_state, grads, loss = step(params, opt_states, keys)
        loss_history.append(loss)
        params_history.append(params)
        if i == cfg['epochs'] or i % cfg['log_rate'] == 0 or i == 0:
            logger_update(loss, params, i)

    return params_history, np.array(loss_history)
