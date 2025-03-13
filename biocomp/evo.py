from . import jaxutils as ut
import jax
import jax.numpy as jnp
from jax import jit, vmap
from evosax import CMA_ES
from tqdm import tqdm


DEFAULT_CFG = {
    "rng_key": 42,
    "generations": 200,
    "popsize": 200,
    "init_min": 0,
    "init_max": 1,
    "regularize_strength": 0.001,
    "static_params": [["shared"]],
}


def optimize_model(model, params, x, y, config=DEFAULT_CFG, loggers=None):
    cfg = {**DEFAULT_CFG, **config}
    loggers = loggers or {}
    rng = jax.random.PRNGKey(cfg["rng_key"])

    dynamic, static_params = ut.split_params(params, cfg["static_params"])
    flat_dyn, dyn_descriptor = ut.flatten_params(dynamic)

    strategy = CMA_ES(popsize=cfg["popsize"], num_dims=flat_dyn.shape[0])

    es_params = strategy.default_params
    es_params.replace(init_min=cfg["init_min"], init_max=cfg["init_max"])
    state = strategy.initialize(rng, es_params)

    history = {
        "fitnesses": [],
    }

    vm_model = jax.vmap(model, in_axes=(None, 0, None))

    def fitness_fn(dyn_flat_params):
        # reconstruct the full parameters:
        dyn_params = ut.unflatten_params(dyn_flat_params, dyn_descriptor)
        params = ut.assemble_params(dyn_params, static_params)
        # compute loss:
        yhat = vm_model(params, x, jax.random.PRNGKey(0))
        loss = jnp.mean((yhat - y) ** 2)
        # regularize so params stay close to 1:
        penalty = jnp.mean((dyn_flat_params - 1) ** 2)
        return loss + cfg["regularize_strength"] * penalty, params

    vm_fitness = jit(vmap(fitness_fn))

    best_fitness = jnp.inf
    best_params = None

    for g in tqdm(list(range(cfg["generations"])), desc="generations"):
        rng, rng_gen, rng_eval = jax.random.split(rng, 3)
        samples, state = strategy.ask(rng_gen, state, es_params)
        fitnesses, params = vm_fitness(samples)
        state = strategy.tell(samples, fitnesses, state, es_params)

        history["fitnesses"].append(fitnesses)

        f_argmin = jnp.argmin(fitnesses)
        if fitnesses[f_argmin] < best_fitness:
            best_fitness = fitnesses[f_argmin]
            best_params = ut.get_pytree(params, f_argmin)

        for logger in loggers.values():
            logger.log(g, history)

    return best_params, history
