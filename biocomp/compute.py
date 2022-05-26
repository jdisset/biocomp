import pandas as pd
from . import utils as ut
import numpy as np
from .library import PartsLibrary as PartsLibrary
from jax.example_libraries.optimizers import adam
import jax
from jax import jit, vmap, value_and_grad
from jax import tree_util as pytree
from jax.tree_util import Partial as partial
import jax.numpy as jnp
from time import time
from copy import deepcopy

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     Model building     --
# ···············································································

DEFAULT_RNA_DEG_RATE = 1.0
DEFAULT_PRT_DEG_RATE = 1.0

DEFAULT_MIN_RATE = 0.0
DEFAULT_MAX_RATE = 1.0

DEFAULT_MIN_COPY_N = 0.0
DEFAULT_MAX_COPY_N = 1.0


def rate_init_continuous(rng, n, minval=DEFAULT_MIN_RATE, maxval=DEFAULT_MAX_RATE):
    return jax.random.uniform(key=rng, shape=(n,), minval=minval, maxval=maxval, dtype=jnp.float32)


def copy_n_init(rng, minval=DEFAULT_MIN_COPY_N, maxval=DEFAULT_MAX_COPY_N):
    return jax.random.uniform(key=rng, minval=minval, maxval=maxval, dtype=jnp.float32)


# each node type is a function that returns 2 other functions:
# - init(rng) -> returns the parameters (this node, others)
# - apply(params, X) -> returns the value of the compute node
#                       X, the inputs, is only useful for the input leaves
# - collect(params, dic) -> collect the param values to the dict[key] node
# - constrain(params) -> constrain params to certain ranges or values


CNODE = {}


def compnode(f):
    CNODE[f.__name__] = f
    return f


def init_upstream(rng, init_funs):
    nbranches = len(init_funs)
    rngs = jax.random.split(rng, nbranches)
    return [init(rng) for init, rng in zip(init_funs, rngs)]


def apply_upstream(params, apply_funs, inputs, **kwargs):
    nbranches = len(apply_funs)
    rng = kwargs.pop(
        'rng', None
    )  # we transmit rngs upstream as some apply functions might need randomness
    rngs = jax.random.split(rng, nbranches) if rng is not None else (None,) * nbranches
    return jnp.array([f(p, inputs, rng=r, **kwargs) for f, p, r in zip(apply_funs, params, rngs)])


def collect_upstream(params, collect_funs):
    res = []
    for f, p in zip(collect_funs, params):
        res += f(p)
    return res


def constrain_upstream(params, constrain_funs):
    return [constrain(p) for constrain, p in zip(constrain_funs, params)]


@compnode
def transcription(*branches, deg_rate=DEFAULT_RNA_DEG_RATE, nid=None):
    nbranches = len(branches)
    init_funs, constrain_funs, apply_funs, collect_funs = zip(*branches)

    def init(rng):
        return (rate_init_continuous(rng, nbranches), init_upstream(rng, init_funs))

    def collect(params):
        t_rates, others = params
        return [(nid, {'tr_rates': np.array(t_rates)})] + collect_upstream(others, collect_funs)

    def constrain(params):
        t_rates, others = params
        t_rates = jnp.clip(t_rates, 0.0, 1.0)
        return (t_rates, constrain_upstream(others, constrain_funs))

    def apply(params, inputs, **kwargs):
        t_rates, others = params
        return jnp.dot(apply_upstream(others, apply_funs, inputs, **kwargs), t_rates) / deg_rate

    return init, constrain, apply, collect


@compnode
def translation(*branches, deg_rate=DEFAULT_PRT_DEG_RATE, **kwargs):
    return transcription(*branches, deg_rate=deg_rate, **kwargs)


@compnode
def sequestron_ERN(neg, pos, nid=None):

    ini, con, app, ass = zip(neg, pos)

    def init(rng):
        return init_upstream(rng, ini)

    def collect(params):
        return collect_upstream(params, ass)

    def constrain(params):
        return constrain_upstream(params, con)

    def apply(params, inputs, **kwargs):
        res = apply_upstream(params, app, inputs, **kwargs)
        return jnp.maximum(0, res[1] - res[0])

    return init, constrain, apply, collect


@compnode
def sequestron_RECOMBINASE(neg, pos, **kwargs):
    return sequestron_ERN(neg, pos, **kwargs)


@compnode
def bias(*_, nid=None):
    def init(rng):
        return copy_n_init(rng)

    def constrain(copy_n):
        copy_n = jnp.clip(copy_n, 0.0, 1.0)
        return copy_n

    def apply(copy_n, inputs, **kwargs):
        return copy_n

    def collect(copy_n):
        return [(nid, {'copy_number': float(copy_n)})]

    return init, constrain, apply, collect


@compnode
def input(id, nid=None):
    def init(rng):
        return copy_n_init(rng)

    def constrain(copy_n):
        copy_n = jnp.clip(copy_n, 0.0, 1.0)
        return copy_n

    def apply(copy_n, inputs, **kwargs):
        return inputs[id] * copy_n

    def collect(copy_n):
        return [(nid, {'copy_number': float(copy_n)})]

    return init, constrain, apply, collect


@compnode
def output(*branches, nid=None):  # simply returns the vector of results from all branches
    init_funs, constrain_funs, apply_funs, collect_funs = zip(*branches)

    def init(rng):
        return init_upstream(rng, init_funs)

    def constrain(params):
        return constrain_upstream(params, constrain_funs)

    def apply(params, inputs, **kwargs):
        return apply_upstream(params, apply_funs, inputs, **kwargs)

    def collect(params):
        return collect_upstream(params, collect_funs)

    return init, constrain, apply, collect


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                  --     ComputeGraphModel class     --
# ···············································································
def base_step(i, state, dlossfunc, get_params, update, model, x, y_true):
    params = get_params(state)
    loss, g = dlossfunc(params, model.apply, x, y_true)
    return (update(i, g, state), loss)


def mseloss(params, apply_f, x, y_true):
    y_preds = vmap(pytree.Partial(apply_f, params))(x)
    return jnp.mean(jnp.power(y_preds - y_true, 2))


dmseloss = value_and_grad(mseloss)


def make_training_start(params_initializer, state_initializer, stepfunc, n_steps):
    @ut.tqdm_scan(n_steps, 'Training model')
    def scannable_step(previous_state, iteration):
        new_state, loss = stepfunc(iteration, previous_state)
        return new_state, (loss, previous_state)

    def train_one_start(key):
        params = params_initializer(key)
        initial_state = state_initializer(params)
        final_state, states_and_losses_history = jax.lax.scan(
            scannable_step, initial_state, np.arange(n_steps)
        )
        losses, sthists = states_and_losses_history
        return (losses, ut.tree_append(sthists, final_state))

    return train_one_start


# training parameters
N_INITIALIZATIONS = 10
N_TRAINING_STEPS = 100
LEARNING_RATE = 1e-2


# # builds the model from the compute graph representation
# # cdf: compute graph dataframe
# def buildModel(cdf):
# outNode = cdf[cdf.type == 'output'].iloc[0]
# def buildImpl(node):
# if node.input_from:  # recursive case: any non-input node
# branches = cdf.loc[node.input_from]
# return CNODE[node.type](*[buildImpl(b) for _, b in branches.iterrows()], nid=node.name)
# return CNODE[node.type](node.is_input, nid=node.name)  # terminal node
# return ComputeGraphModel(*buildImpl(outNode))


class ComputeGraphModel:
    def __init__(self, init, constrain, apply, collect, compg=None):
        self.init = init
        self.constrain = constrain
        self.apply = jit(apply)
        self.collect = collect
        self.compg = compg

    @classmethod
    def fromDataframe(cls, cdf):
        # builds the model from the compute graph representation
        # cdf: compute graph dataframe
        outNode = cdf[cdf.type == 'output'].iloc[0]

        def buildImpl(node):
            if node.input_from:  # recursive case: any non-input node
                branches = cdf.loc[node.input_from]
                return CNODE[node.type](
                    *[buildImpl(b) for _, b in branches.iterrows()], nid=node.name
                )
            return CNODE[node.type](node.is_input, nid=node.name)  # terminal node

        return cls(*(buildImpl(outNode) + (cdf,)))

    def asTuple(self):
        return (self.init, self.constrain, self.apply, self.collect)

    def collectParamsToDataframe(self, params, df):
        c = self.collect(params)
        rowids, p = zip(*c)
        df['parameters'] = None
        df.loc[rowids, 'parameters'] = p

    def toDataframe(self, params):
        if self.compg is None:
            raise Exception(
                """
                Can't use toDataframe as this model doesn't hold any reference to an original dataframe.
                Try using collectParamsToDataframe instead, or build with fromDataframe()
                """
            )
        else:
            df =self.compg.copy(deep=True)
            self.collectParamsToDataframe(params, df)
            return df

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def train(
        self,
        key,
        X,
        y_true,
        learning_rate=LEARNING_RATE,
        n_init=N_INITIALIZATIONS,
        n_steps=N_TRAINING_STEPS,
    ):
        initialization_keys = jax.random.split(key, n_init)

        # compiled training functions
        opt_init, update, get_params = adam(step_size=learning_rate)  # optimizer
        step = jit(
            partial(
                base_step,
                get_params=get_params,
                dlossfunc=dmseloss,
                update=update,
                model=self,
                x=X,
                y_true=y_true,
            )
        )
        train_fun = make_training_start(self.init, opt_init, step, n_steps)

        # actual training "loop"
        start = time()
        loss_state_histories = vmap(train_fun)(initialization_keys)
        end = time()
        print('Trained in', end - start)

        losses, stacked_states = loss_state_histories
        return (get_params(stacked_states), losses)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
