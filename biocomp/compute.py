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


def continuous_initializer(rng, n, minval=DEFAULT_MIN_RATE, maxval=DEFAULT_MAX_RATE):
    def init():
        return jax.random.uniform(
            key=rng, shape=(n,), minval=minval, maxval=maxval, dtype=jnp.float32
        )

    return init

def quantize(x, possible_values):
    if len(possible_values) == 0:
        return x
    if len(possible_values) == 1:
        return possible_values[0]
    else :
        return quantize_impl(x, possible_values)

@partial(jax.custom_jvp, nondiff_argnums=(1,))
def quantize_impl(x, arr):
    return arr[jnp.argmin(jnp.abs(arr - x))]


# TODO reverse diff
# we define the derivative of the quantize function as if it was just the identity function (x -> x)
@quantize_impl.defjvp
def quantize_impl_jvp(_, x, x_tang):
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


# nodes to write:
# translation, transcription, sequestron_ERN, sequestron_RCB, source, numeric, aggregation


@compnode
def transcription(get_param, get_quantized, **_):
    def apply(*values, rng_key):
        k0, k1 = jax.random.split(rng_key, 2)
        t_rates = get_quantized(
            "tc_rate",
            get_param("tc_rate", init=continuous_initializer(k0, len(values))),
            mode='input_edges',
        )
        assert len(t_rates) == len(values)
        return jnp.dot(t_rates, jnp.array(values)) / get_param(
            "rna_deg_rate", init=continuous_initializer(k1, 1), shared=True
        )

    return apply


@compnode
def translation(get_param, get_quantized, **_):
    def apply(*values, rng_key):
        k0, k1 = jax.random.split(rng_key, 2)
        tl_rates = get_quantized(
            "tl_rate",
            get_param("tl_rate", init=continuous_initializer(k0, len(values))),
            mode='input_edges',
        )
        assert len(tl_rates) == len(values)
        return jnp.dot(tl_rates, jnp.array(values)) / get_param(
            "prt_deg_rate", init=continuous_initializer(k1, 1), shared=True
        )

    return apply


@compnode
def sequestron_ERN(get_param, get_quantized, **_):
    def apply(neg, pos, **_):
        return jnp.maximum(pos - neg, 0.0)

    return apply


@compnode
def sequestron_RCB(get_param, get_quantized, **_):
    EPSILON = 1e-12

    def apply(neg, pos, **_):
        return pos / (neg + pos + EPSILON)

    return apply


@compnode
def source(get_param, get_quantized, n_outputs, **_):
    def apply(inp, **_):
        return jnp.ones(n_outputs) * inp

    return apply


@compnode
def numeric(get_param, get_quantized, **_):
    def apply(rng_key):
        return get_param("value", init=continuous_initializer(rng_key, 1))

    return apply


# aggregations split a single input in ratios (defined by parameters)
@compnode
def aggregation(get_param, get_quantized, n_outputs, ratios=None, **_):

    initializer = (
        continuous_initializer if ratios is None else lambda rng, n: lambda: jnp.array(ratios)
    )

    def apply(inp, rng_key):
        ratios = get_param("ratios", init=initializer(rng_key, n_outputs))
        assert len(ratios) == n_outputs
        ratios = jnp.maximum(ratios, 0.0)
        ratios += 1e-12
        ratios /= jnp.sum(ratios)
        return jnp.array(ratios) * inp

    return apply


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                  --     ComputeGraphModel class     --
# ···············································································


def get_param(params, name, init, shared=False, nodeid=None):
    if not shared:
        params.setdefault('node', {})
        pardict = params['node'].setdefault(nodeid, {})
    else:
        params.setdefault('shared', {})
        pardict = params['shared']
    if name not in pardict:
        try:
            pardict[name] = init()
        except Exception as e:
            print(f'Error initializing param "{name}" from node {nodeid}: {e}')
            raise e
    return pardict[name]


def get_possible_parts(param_name, cdg_node_id, cdg):
    # should return the name of possible parts for a given cdg node, slot and param name
    # example: get_possible_values('transcription_rate', ...) -> ['hEF1a', 'hEF1b', 'hEF1c']
    #          get_possible_values('translation_rate', ...) -> [None, '1xuORF', '2xuORF', ...]
    # params are stored in the params column of the cdg as a dict {param_name:[possiblevaluees]}
    available_params = cdg.loc[cdg_node_id, 'params']
    if param_name not in available_params:
        raise ValueError(
            f'Param {param_name} not available for cdg node {cdg_node_id}. Available: {available_params}'
        )
    return available_params[param_name]


def get_quantized(params, param_name, values, node_id, cdf, cdg, quantize_fun, mode='input_edges'):
    """Return a quantized version of the parameter, conditioned on the
    relevant species (either input or output cdg nodes). mode can be 'input' or 'output'."""
    # We are selecting possible values for a parameter depending on which path, or edge, they come from.
    # The edges of the compute graph are basically nodes in the central dogma graph,
    # i.e they're *more or less*
    # dual to each other, but not exactly since the compute graph adds interactions and extra nodes).
    # Anyway I think it's reasonable to consider that the type of nodes that will call getQuantized
    # are the types for which it's ok to consider the CDG as the dual of the COMPG.

    # check that mode is either input_edges or output_edges or inner
    if mode not in ['input_edges', 'output_edges', 'inner']:
        raise ValueError(f'Invalid mode {mode} for get_quantized')

    cdg_ids = (
        cdf.loc[node_id]['cdg_input'] if mode == 'input_edges' else cdf.loc[node_id]['cdg_output']
    )
    if cdg_ids is None:
        raise ValueError(f'Node {node_id} has no {mode} CDG node')
    if not isinstance(cdg_ids, list):
        cdg_ids = [cdg_ids]

    try:
        possible_parts = [get_possible_parts(param_name, cdg_id, cdg) for cdg_id in cdg_ids]
    except ValueError as e:
        raise ValueError(
            f"""get_quantized: Error getting possible parts for node {node_id} with cdg_ids {cdg_ids} and param_name {param_name}, mode {mode}:
            \n-> {e}.
            \n compute graph:
            \n{cdf}
            \n node:
            \n{cdf.loc[node_id]}
            \n central dogma graph:
            \n{cdg}"""
        )
    # concat part names with param_name
    possible_names = [[f'{part}::{param_name}' for part in parts] for parts in possible_parts]
    assert len(possible_names) == len(
        values
    ), f'len(possible_names)={len(possible_names)} != len(values)={len(values)}'

    possible_values = [
        jnp.array([get_param(params, n, lambda: val, shared=True) for n in names])
        for names, val in zip(possible_names, values)
    ]
    return jnp.array([quantize_fun(v, p) for v, p in zip(values, possible_values)])


class ComputeGraphModel:
    def __init__(self, network):
        self.network = network
        self.built = False


    def build(self):
        assert self.network is not None
        assert self.network.is_built()

        batches = self.__get_batch_sequence_of_nodes()
        flat_batches = [item for sublist in batches for item in sublist]

        def apply(params, inputs, rng_key):
            assert len(inputs) == len(
                self.network.compute_graph[self.network.compute_graph['type'] == 'input']
            )
            results = {}
            # split keys for each node
            keys = jax.random.split(rng_key, len(flat_batches))
            for nid, key in zip(flat_batches, keys):
                node_row = self.network.compute_graph.loc[nid]
                upstream_results = [results[inp[0]][inp[1]] for inp in node_row.input_from]
                if node_row.type == 'input':
                    results[nid] = inputs[node_row.extra['input_position']]
                    break
                if node_row.type == 'output':
                    return jnp.array(upstream_results)
                assert node_row.type in CNODE, f'Invalid node type {node_row.type}'
                get_p = partial(get_param, params, nodeid=nid)
                get_q = partial(
                    get_quantized,
                    params,
                    node_id=nid,
                    cdf=self.network.compute_graph,
                    cdg=self.network.central_dogma_graph,
                    quantize_fun=quantize,
                )
                extra_params = {
                    'n_outputs': len(self.network.compute_graph.loc[nid]['output_to']),
                    'n_inputs': len(self.network.compute_graph.loc[nid]['input_from']),
                }
                if node_row.extra is not None:
                    extra_params.update(node_row.extra)
                comp_node = CNODE[node_row.type](get_p, get_q, **extra_params)
                res = comp_node(*upstream_results, rng_key=key)
                if extra_params['n_outputs'] == 1:
                    results[nid] = jnp.array([res])
                else:
                    results[nid] = res

            # should never reach this point
            raise ValueError('Invalid compute graph, no output node found')

        def init(rng_key):
            params = {}
            n_inputs = len(
                self.network.compute_graph[self.network.compute_graph['type'] == 'input']
            )
            apply(params, [jnp.array([1.0])] * n_inputs, rng_key)
            return params

        self.apply = apply
        self.init = init
        self.built = True

    def __call__(self, params, inputs, rng_key):
        assert self.built
        return self.apply(params, inputs, rng_key)

    def __get_batch_sequence_of_nodes(self):
        """Return a list of lists of compute nodes from the network,
        where each node of a sublist can be computed independently of the others,
        but each sublist must be computed in order."""
        visited = set()
        batches = []
        while len(visited) < len(self.network.compute_graph):
            independent = [
                i
                for i, row in self.network.compute_graph.iterrows()
                if (not row['input_from'] or all([x[0] in visited for x in row['input_from']]))
                and i not in visited
            ]
            if not independent:
                raise ValueError('Compute graph is not acyclic')
            visited.update(independent)
            batches.append(independent)
        return batches


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
