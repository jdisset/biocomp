import pandas as pd
from scipy.signal.filter_design import EPSILON
from . import utils as ut
import numpy as np
from .library import PartsLibrary as PartsLibrary
import jax
from jax import jit, vmap
from jax import tree_util as pytree
from jax.tree_util import Partial as partial
import jax.numpy as jnp
from time import time
import optax

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     Compute Nodes     --
# ···············································································

DEFAULT_RNA_DEG_RATE = 1.0
DEFAULT_PRT_DEG_RATE = 1.0

DEFAULT_MIN_RATE = 0.0
DEFAULT_MAX_RATE = 1.0

DEFAULT_MIN_COPY_N = 0.0
DEFAULT_MAX_COPY_N = 50.0

POSSIBLE_TL_RATES = jnp.array([1.0 / 2**n for n in range(5)] + [0.75, 0.9])
POSSIBLE_TX_RATES = jnp.linspace(0.0, 1.0, num=21)


def rate_init_continuous(rng, n, minval=DEFAULT_MIN_RATE, maxval=DEFAULT_MAX_RATE):
    return jax.random.uniform(key=rng, shape=(n,), minval=minval, maxval=maxval, dtype=jnp.float32)


def copy_n_init(rng, minval=0.0, maxval=2.0):
    return jax.random.uniform(key=rng, minval=minval, maxval=maxval, dtype=jnp.float32)


# each node type is a function that returns 2 other functions:
# - init(rng) -> returns the parameters (this node, others)
# - apply(params, X) -> returns the value of the compute node
#                       X, the inputs, is only useful for the input leaves
# - collect(params, dic) -> collect the param values to the dict[key] node
# - constrain(params) -> constrain params to certain ranges or values


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def quantize(x, arr):
    if len(arr) == 0:
        return x
    return arr[jnp.argmin(jnp.abs(arr - x))]


# we define the derivative of the quantize function as if it was just the identity function (x -> x)
@quantize.defjvp
def quantize_jvp(_, x, x_tang):
    (x,) = x
    (x_dot,) = x_tang
    return x, x_dot


@jax.custom_jvp
def round_to_int(x):
    return jnp.round(x)


# we define the derivative of the quantize function as if it was just the identity function (x -> x)
@round_to_int.defjvp
def round_to_int_jvp(x, x_tang):
    (x,) = x
    (x_dot,) = x_tang
    return x, x_dot


CNODE = {}


def compnode(f):
    CNODE[f.__name__] = f
    return f


# TODO: init should definitely initialize to a possible (quantized) value.
# This way when we load a fully determined circuit from XP results, we can just have the right value
# and they won't move (since they should be marked as non-trainable)


def init_upstream(rng, init_funs):
    nbranches = len(init_funs)
    rngs = jax.random.split(rng, nbranches)
    res = []
    for init, r in zip(init_funs, rngs):
        res += init(r)
    return res


def collect_upstream(params, collect_funs):
    res = []
    for c in collect_funs:
        res += c(params)
    return res


def apply_upstream(params, apply_funs, inputs, **kwargs):
    nbranches = len(apply_funs)
    rng = kwargs.pop('rng', None)
    rngs = jax.random.split(rng, nbranches) if rng is not None else (None,) * nbranches
    return jnp.array([f(p, inputs, rng=r, **kwargs) for f, p, r in zip(apply_funs, params, rngs)])


def linear(nid, quantized_rates_ids, rate_name, deg_name, *branches):
    # quantized_rates_ids is a list of keys for the rates we can use in this node
    nbranches = len(branches)
    print(f"nbranches: {nbranches}, branches: {branches}")
    init_funs, apply_funs, collect_funs = zip(*branches)

    def init(rng):
        return [(nid, {rate_name: rate_init_continuous(rng, nbranches)})] + init_upstream(
            rng, init_funs
        )

    def quantized_rates(params):
        possible_rates = params['shared']['quantized'][rate_name][quantized_rates_ids]
        t_rates = vmap(partial(quantize, arr=possible_rates))(params['local'][nid][rate_name])
        return t_rates


    def quantized_rates(params):
        possible_rates = [l1_qfs["tx_rate"](params['shared']['quantized']) for l1_qfs in list_of_l1_quantize_functions]
        t_rates = vmap(quantize)(params['local'][nid][rate_name], possible_rates)
        return t_rates

    def apply(params, inputs, **kwargs):
        t_rates = quantized_rates(params)
        return (
            jnp.dot(apply_upstream(params, apply_funs, inputs, **kwargs), t_rates)
            / params['shared'][deg_name]
        )

    def collect(params):
        return [(nid, {rate_name: quantized_rates(params)})] + collect_upstream(
            params, collect_funs
        )

    return init, apply, collect


@compnode
def transcription(nid, quantized_rates_ids, *branches):
    return linear(nid, quantized_rates_ids, 'tx_rate', 'rna_deg_rate', *branches)


@compnode
def translation(nid, quantized_rates_ids, *branches):
    return linear(nid, quantized_rates_ids, 'tl_rate', 'prt_deg_rate', *branches)


@compnode
def sequestron_ERN(nid, pos, neg):

    ini, app, col = zip(neg, pos)

    def init(rng):
        return init_upstream(rng, ini)

    def apply(params, inputs, **kwargs):
        res = apply_upstream(params, app, inputs, **kwargs)
        return jnp.maximum(0, res[1] - res[0])

    def collect(params):
        return collect_upstream(params, col)

    return init, apply, collect


@compnode
def sequestron_RECOMBINASE(nid, neg, pos):

    DIV_EPSILON = 1e-9

    _, app, _ = zip(neg, pos)

    init, _, collect = sequestron_ERN(nid, neg, pos)

    def apply(params, inputs, **kwargs):
        res = apply_upstream(params, app, inputs, **kwargs)
        return res[1] / (res[1] + res[0] + DIV_EPSILON)

    return init, apply, collect


@compnode
def bias(nid, MAX_COPY_N=DEFAULT_MAX_COPY_N):
    def init(rng):
        return [(nid, {'copy_number': copy_n_init(rng)})]

    def apply(params, *_, **__):
        return params['local'][nid]['copy_number']

    def collect(copy_n):
        return [(nid, {'copy_number': float(copy_n)})]

    return init, apply, collect


@compnode
def input(nid, id, MAX_COPY_N=DEFAULT_MAX_COPY_N):

    init, _, collect = bias(nid, MAX_COPY_N)

    def apply(params, inputs, **kwargs):
        return inputs[id] * params['local'][nid]['copy_number']

    return init, apply, collect


@compnode
def output(nid, *branches):  # simply returns the vector of results from all branches
    init_funs, apply_funs, collect_funs = zip(*branches)

    def init(rng):
        return init_upstream(rng, init_funs)

    def apply(params, inputs, **kwargs):
        return apply_upstream(params, apply_funs, inputs, **kwargs)

    def collect(params):
        return collect_upstream(params, collect_funs)

    return init, apply, collect


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                  --     ComputeGraphModel class     --
# ···············································································


# training parameters
N_INITIALIZATIONS = 10
N_TRAINING_STEPS = 100
LEARNING_RATE = 1e-2


class ComputeGraphModel:
    def __init__(self, init, apply, collect, compg=None):
        self.init = init
        self.apply = jit(apply)
        self.collect = collect
        self.compg = compg

    @classmethod
    def fromDataframe(cls, cdf):
        # builds the model from the compute graph representation
        # cdf: compute graph dataframe
        outNode = cdf[cdf.type == 'output'].iloc[0]

        def buildImpl(node):
            print(f'\n---\nbuilding node {node.name}, of type {node.type}')

            # if node requires a quantier

            if node.type == 'transcription' or node.type == 'translation':
                quantized_rates_ids = node.quantized_rates_ids.split(',')
                quantized_rates_ids = [int(x) for x in quantized_rates_ids]
                return linear(node.name, quantized_rates_ids, node.rate_name, node.deg_name, *[buildImpl(n) for n in node.branches])

            if node.input_from:  # recursive case: any non-input node
                branches = cdf.loc[node.input_from]
                return CNODE[node.type](node.name, *[buildImpl(b) for _, b in branches.iterrows()])
            return CNODE[node.type](node.name, node.is_input)  # terminal node

        return cls(*(buildImpl(outNode) + (cdf,)))

    def asTuple(self):
        return (self.init, self.apply, self.collect)

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
            df = self.compg.copy(deep=True)
            self.collectParamsToDataframe(params, df)
            return df

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def train(
        self,
        key,
        X,
        y_true,
        n_init=N_INITIALIZATIONS,
        n_steps=N_TRAINING_STEPS,
        learning_rate=LEARNING_RATE,
        compile_train_loop=False,
        progress_type=ut.TQDMProgress,
    ):
        optimizer = optax.adam(learning_rate=learning_rate)
        initialization_keys = jax.random.split(key, n_init)

        @jit
        def loss(params, x: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
            y_pred = vmap(partial(self.apply, params))(x)
            l = optax.l2_loss(y_pred, y_true)
            return l.mean()

        def step(params, opt_state):
            loss_value, grads = jax.value_and_grad(loss)(params, X, y_true)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            # params = self.constrain(params)
            return params, opt_state, loss_value

        def train_one(key):
            initial_params = self.init(key)
            initial_state = optimizer.init(initial_params)

            @ut.progress_scan(n_steps, progress_type, 'Training model')
            def scannable_step(params_and_state, iter_num):
                params, state = params_and_state
                new_params, new_state, loss = step(params, state)
                return (new_params, new_state), (loss, params)

            _, losses_and_params_history = jax.lax.scan(
                scannable_step, (initial_params, initial_state), np.arange(n_steps)
            )
            return losses_and_params_history

        # actual training "loop"
        start = time()
        train_all = vmap(train_one)
        if compile_train_loop:
            train_all = jax.jit(train_all)
        all_losses, all_params = train_all(initialization_keys)
        end = time()
        print('Trained in', end - start)

        return all_losses, all_params


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     Archive     --
# ···············································································
# # def init_upstream(rng, init_funs):
# nbranches = len(init_funs)
# rngs = jax.random.split(rng, nbranches)
# return [init(rng) for init, rng in zip(init_funs, rngs)]


# def apply_upstream(params, apply_funs, inputs, **kwargs):
# nbranches = len(apply_funs)
# rng = kwargs.pop(
# 'rng', None
# )  # we transmit rngs upstream as some apply functions might need randomness
# rngs = jax.random.split(rng, nbranches) if rng is not None else (None,) * nbranches
# return jnp.array([f(p, inputs, rng=r, **kwargs) for f, p, r in zip(apply_funs, params, rngs)])


# def collect_upstream(params, collect_funs):
# res = []
# for f, p in zip(collect_funs, params):
# res += f(p)
# return res

# def constrain_upstream(params, constrain_funs):
# return [constrain(p) for constrain, p in zip(constrain_funs, params)]


# @compnode
# def transcription(
# *branches, deg_rate=DEFAULT_RNA_DEG_RATE, nid=None, possible_rates=POSSIBLE_TX_RATES
# ):
# nbranches = len(branches)
# init_funs, constrain_funs, apply_funs, collect_funs = zip(*branches)

# def init(rng):
# return (rate_init_continuous(rng, nbranches), init_upstream(rng, init_funs))

# def collect(params):
# t_rates, others = params
# return [(nid, {'tr_rates': np.array(t_rates)})] + collect_upstream(others, collect_funs)

# def constrain(params):
# t_rates, others = params
# t_rates = jnp.clip(t_rates, 0.0, 1.0)
# t_rates = vmap(quantize)(t_rates, jnp.tile(possible_rates, (len(t_rates), 1)))
# return (t_rates, constrain_upstream(others, constrain_funs))

# def apply(params, inputs, **kwargs):
# t_rates, others = params
# t_rates = vmap(quantize)(t_rates, jnp.tile(possible_rates, (len(t_rates), 1)))
# return jnp.dot(apply_upstream(others, apply_funs, inputs, **kwargs), t_rates) / deg_rate

# return init, constrain, apply, collect


# @compnode
# def translation(*branches, deg_rate=DEFAULT_PRT_DEG_RATE, possible_rates=POSSIBLE_TL_RATES, **kwargs):
# return transcription(*branches, deg_rate=deg_rate, possible_rates=possible_rates, **kwargs)


# @compnode
# def sequestron_ERN(neg, pos, nid=None):

# ini, con, app, ass = zip(neg, pos)

# def init(rng):
# return init_upstream(rng, ini)

# def collect(params):
# return collect_upstream(params, ass)

# def constrain(params):
# return constrain_upstream(params, con)

# def apply(params, inputs, **kwargs):
# res = apply_upstream(params, app, inputs, **kwargs)
# return jnp.maximum(0, res[1] - res[0])

# return init, constrain, apply, collect


# @compnode
# def sequestron_RECOMBINASE(neg, pos, **kwargs):
# return sequestron_ERN(neg, pos, **kwargs)


# @compnode
# def bias(*_, nid=None, MAX_COPY_N=DEFAULT_MAX_COPY_N):
# def init(rng):
# return copy_n_init(rng)

# def constrain(copy_n):
# return jnp.clip(copy_n, 0.0, MAX_COPY_N)

# def apply(copy_n, inputs, **kwargs):
# return copy_n

# def collect(copy_n):
# return [(nid, {'copy_number': float(copy_n)})]

# return init, constrain, apply, collect


# @compnode
# def input(id, nid=None, MAX_COPY_N=DEFAULT_MAX_COPY_N):
# def init(rng):
# return copy_n_init(rng)

# def constrain(copy_n):
# return jnp.clip(copy_n, 0.0, MAX_COPY_N)

# def apply(copy_n, inputs, **kwargs):
# return inputs[id] * copy_n

# def collect(copy_n):
# return [(nid, {'copy_number': float(copy_n)})]

# return init, constrain, apply, collect


# @compnode
# def output(*branches, nid=None):  # simply returns the vector of results from all branches
# init_funs, constrain_funs, apply_funs, collect_funs = zip(*branches)

# def init(rng):
# return init_upstream(rng, init_funs)

# def constrain(params):
# return constrain_upstream(params, constrain_funs)

# def apply(params, inputs, **kwargs):
# return apply_upstream(params, apply_funs, inputs, **kwargs)

# def collect(params):
# return collect_upstream(params, collect_funs)

# return init, constrain, apply, collect


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
