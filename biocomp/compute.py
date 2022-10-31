from .library import PartsLibrary as PartsLibrary
import jax
from jax.tree_util import Partial as partial
import jax.numpy as jnp


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


def continuous_initializer(rng, shape=(), minval=DEFAULT_MIN_RATE, maxval=DEFAULT_MAX_RATE):
    def init():
        res = jax.random.uniform(
            key=rng, shape=shape, minval=minval, maxval=maxval, dtype=jnp.float32
        )
        return res

    return init

def glorot_initializer(rng, shape):
    def init():
        return jax.nn.initializers.glorot_normal()(rng, shape)
    return init


def quantize(x, possible_values):
    if len(possible_values) == 0:
        return x
    if len(possible_values) == 1:
        return possible_values[0]
    else:
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


BC_EPSILON = 1e-12

COMPUTE_NODES_DICT = {}
INVERSE_NODES_DICT = {}


def compnode(f):
    COMPUTE_NODES_DICT[f.__name__] = f
    return f


def inv_compnode(fwd_name):
    def inv(f):
        COMPUTE_NODES_DICT[f.__name__] = f
        INVERSE_NODES_DICT[fwd_name] = f.__name__
        return f

    return inv



# translation and transcription are the same, except for the parameters
def _transform(get_param, get_quantized, transform_name, deg_param_name, **_):
    def apply(*values, rng_key):
        val = jnp.array(values)
        k0, k1 = jax.random.split(rng_key, 2)
        rate_name = f'{transform_name}_rate'
        rates = get_quantized(
            rate_name,
            get_param(rate_name, init=continuous_initializer(k0, val.shape)),
            mode='input_edges',
        )
        deg_rate = get_param(deg_param_name, init=continuous_initializer(k1), shared=True)
        res = (jnp.dot(rates, val) / deg_rate)
        # print(f'values: {values}')
        # print(f'val: {val}')
        # print(f'rates: {rates}')
        # print(f'deg_rate: {deg_rate}')
        # print(f'res: {res}')
        return res

    return apply


def _inverse_transform(get_param, get_quantized, transform_name, deg_param_name, **_):
    def apply(value, rng_key):
        # inverse can only work if there's only one input edge
        assert (value.shape == ())
        k0, k1 = jax.random.split(rng_key, 2)

        rate_name = f'{transform_name}_rate'
        rate = get_quantized(
            rate_name,
            get_param(rate_name, init=continuous_initializer(k0, (1,))),
            mode='input_edges',
        )[0]
        deg = get_param(deg_param_name, init=continuous_initializer(k1), shared=True)

        res = (value * deg / rate)
        return res

    return apply




@compnode
def transcription(get_param, get_quantized, **_):
    return _transform(get_param, get_quantized, 'tc', 'rna_deg_rate')


@inv_compnode(fwd_name='transcription')
def inv_transcription(get_param, get_quantized, **_):
    return _inverse_transform(get_param, get_quantized, 'tc', 'rna_deg_rate')

@compnode
def translation(get_param, get_quantized, **_):
    return _transform(get_param, get_quantized, 'tl', 'prt_deg_rate')


@inv_compnode(fwd_name='translation')
def inv_translation(get_param, get_quantized, **_):
    return _inverse_transform(get_param, get_quantized, 'tl', 'prt_deg_rate')


@compnode
def sequestron_ERN(get_param, get_quantized, **_):
    def apply(neg, pos, **_):
        return jnp.maximum(pos - neg, 0.0)

    return apply


@compnode
def ERN_with_affinity(get_param, get_quantized, seq_name, **_):
    def apply(neg, pos, rng_key, **_):
        param_name = f'{seq_name}::affinity'
        affinity = get_param(
            param_name, init=continuous_initializer(rng_key), shared=True
        )
        return jnp.maximum(pos - neg * affinity, 0.0)

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


# inverse of source is just a pass-through
@inv_compnode(fwd_name='source')
def inv_source(*_, **__):
    def apply(value, **_):
        return value

    return apply


@compnode
def numeric(get_param, get_quantized, **_):
    def apply(rng_key):
        res = get_param("value", init=continuous_initializer(rng_key))
        return res

    return apply


# inverse of numeric is just a pass-through
@inv_compnode(fwd_name='numeric')
def inv_numeric(*_, **__):
    def apply(value, **_):
        return value

    return apply


# aggregations split a single input in ratios (defined by parameters)
@compnode
def aggregation(get_param, get_quantized, n_outputs, **kwargs):
    def apply(inp, rng_key):

        if 'ratios' in kwargs:
            ratios = jnp.maximum(get_param("ratios", overwrite_with=jnp.array(kwargs['ratios'])), BC_EPSILON)
        else:
            ratios = jnp.maximum(get_param("ratios", init=continuous_initializer(rng_key, (n_outputs,))), BC_EPSILON)

        assert ratios.shape == (n_outputs,)

        ratios /= jnp.sum(ratios)
        return jnp.array(ratios) * inp

    return apply


@inv_compnode(fwd_name='aggregation')
def inv_aggregation(get_param, get_quantized, original_output_len, original_output_slot, **_):
    assert original_output_len > 0
    assert original_output_slot < original_output_len

    def apply(inp, rng_key):
        ratios = jnp.maximum(
            get_param("ratios", init=continuous_initializer(rng_key, (original_output_len,))),
            BC_EPSILON,
        )
        ratios /= jnp.sum(ratios)
        return inp / ratios[original_output_slot]

    return apply


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                  --     ComputeGraphModel class     --
# ···············································································


def get_param(params, name, init=None, overwrite_with=None, shared=False, node_id=None, node_namespace=None):
    if not shared:
        assert node_id is not None
        params.setdefault('node', {})
        if node_namespace is not None:
            params['node'].setdefault(node_namespace, {})
            pardict = params['node'][node_namespace].setdefault(node_id, {})
        else:
            pardict = params['node'].setdefault(node_id, {})
    else:
        params.setdefault('shared', {})
        pardict = params['shared']

    assert (init is None) != (overwrite_with is None)

    if overwrite_with is not None:
        pardict[name] = overwrite_with

    if name not in pardict:
        try:
            pardict[name] = init()
        except Exception as e:
            print(f'Error initializing param "{name}" from node {node_id}: {e}')
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
    res = jnp.array([quantize_fun(v, p) for v, p in zip(values, possible_values)])
    return res


class ComputeGraphModel:
    def __init__(self, network):
        self.network = network
        self.built = False

    def build(self, node_remap=dict()):
        assert self.network is not None
        assert self.network.is_built()

        # node_remap is a dictionnary that maps "vanilla" node types to
        # new names. Useful to try different node implementations
        # (e.g translation -> custom_translation_v2)

        batches = self.__get_batch_sequence_of_nodes()
        flat_batches = [item for sublist in batches for item in sublist]

        def collect_all_results(params, inputs, rng_key, node_namespace=None):
            assert len(inputs) == len(
                self.network.compute_graph[self.network.compute_graph['type'] == 'input']
            )
            results = {}
            # split keys for each node
            keys = jax.random.split(rng_key, len(flat_batches))
            for nid, key in zip(flat_batches, keys):
                node_row = self.network.compute_graph.loc[nid]
                # upstream_results = [results[inp[0]][inp[1]] for inp in node_row.input_from]
                upstream_results = []
                for inp in node_row.input_from:
                    if results[inp[0]].shape == ():
                        assert inp[1] == 0
                        upstream_results.append(results[inp[0]])
                    else:
                        upstream_results.append(results[inp[0]][inp[1]])

                if node_row.type == 'input':
                    # results[nid] = jnp.array([inputs[node_row.extra['input_position']]])
                    results[nid] = inputs[node_row.extra['input_position']]
                    continue
                if node_row.type == 'output':
                    return jnp.array(upstream_results), results
                assert node_row.type in COMPUTE_NODES_DICT, f'Invalid node type {node_row.type}'

                # if it's an inverse node:
                nodeid_for_getters = nid
                if node_row.extra is not None and 'is_inverse_of' in node_row.extra:
                    nodeid_for_getters = node_row.extra['is_inverse_of']

                get_p = partial(
                    get_param,
                    params,
                    node_id=nodeid_for_getters,
                    node_namespace=node_namespace,
                )

                get_q = partial(
                    get_quantized,
                    params,
                    node_id=nodeid_for_getters,
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

                fun_name = node_remap.get(node_row.type, node_row.type)
                assert fun_name in COMPUTE_NODES_DICT, f'Invalid node type {fun_name}'

                comp_node = COMPUTE_NODES_DICT[fun_name](get_p, get_q, **extra_params)
                res = comp_node(*upstream_results, rng_key=key)
                results[nid] = res

                # if extra_params['n_outputs'] == 1:
                    # results[nid] = jnp.array([res])
                # else:
                    # results[nid] = res

            raise ValueError('Invalid compute graph, no output node found')

        def apply(*args, **kwargs):
            return collect_all_results(*args, **kwargs)[0]

        def init(rng_key, pre_params=None, node_namespace=None):
            params = {}
            if pre_params is not None:
                params = pre_params
            n_inputs = len(
                self.network.compute_graph[self.network.compute_graph['type'] == 'input']
            )
            apply(params, [jnp.array([1.0])] * n_inputs, rng_key, node_namespace=node_namespace)
            return params

        self.apply = apply
        self.collect_all_results = collect_all_results
        self.init = init
        self.flat_batches = flat_batches
        self.built = True

    def __repr__(self):
        def list_network_inputs():
            return [self.network.compute_graph[self.network.compute_graph['type'] == 'input']]

        def list_network_outputs():
            return [self.network.compute_graph[self.network.compute_graph['type'] == 'output']]

        return f'ComputeGraphModel({list_network_inputs()} -> {list_network_outputs()})'

    def __call__(self, *args, **kwargs):
        assert self.built
        return self.apply(*args, **kwargs)

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

    def get_output_proteins(self):
        onode = self.network.compute_graph[self.network.compute_graph['type'] == 'output']
        assert len(onode) == 1, f'Invalid number of output nodes: {len(onode)}'
        # get onode.cdg_input, match it with the id in network.central_dogma_graph, and get the content
        # (for each cdg_input)
        return [
            self.network.central_dogma_graph.loc[cdg_id]['content'][0]
            for cdg_id in onode.iloc[0]['cdg_input']
        ]

    def get_input_from_output(self, output_arr):
        # each input node has, in its extra, 'input_from_output' and 'input_position'
        # we want to transform output_arr by reordering the columns
        mapping = self.get_inverted_input_positions()
        return output_arr[:, [mapping[i] for i in range(len(mapping))]]

    def get_inverted_input_proteins(self):
        mapping = self.get_inverted_input_positions()
        output_proteins = self.get_output_proteins()
        assert len(mapping) <= len(output_proteins)
        return [output_proteins[mapping[i]] for i in range(len(mapping))]

    def get_inverted_input_positions(self):
        mapping = {}
        for _, row in self.network.compute_graph[
            self.network.compute_graph['type'] == 'input'
        ].iterrows():
            mapping[row.extra['input_position']] = row.extra['input_from_output']
        assert set(mapping.keys()) == set(range(len(mapping.keys())))
        assert len(mapping.keys()) == len(set(mapping.values()))

        return mapping


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


def test_inverse():
    import numpy as np

    params = {}
    get_p = partial(get_param, params)

    def no_quantize(_, values, **__):
        return values

    x = jnp.array([0.01, 0.2, 1.0, 100.0, 1000.0, 100000.0])

    # transcription
    rng_key = jax.random.PRNGKey(0)
    tl = translation(partial(get_p, nodeid=1), no_quantize)
    inv_tl = inv_translation(partial(get_p, nodeid=1), no_quantize)
    y = np.array([tl(xx, rng_key=rng_key) for xx in x]).squeeze()
    y_inv = np.array([inv_tl(yy, rng_key=rng_key) for yy in y]).squeeze()
    assert np.allclose(x, y_inv)

    # translation
    rng_key = jax.random.PRNGKey(0)
    tl = transcription(partial(get_p, nodeid=2), no_quantize)
    inv_tl = inv_transcription(partial(get_p, nodeid=2), no_quantize)
    y = np.array([tl(xx, rng_key=rng_key) for xx in x]).squeeze()
    y_inv = np.array([inv_tl(yy, rng_key=rng_key) for yy in y]).squeeze()
    assert np.allclose(x, y_inv)

    # aggregation
    rng_key = jax.random.PRNGKey(0)
    tl = aggregation(partial(get_p, nodeid=3), no_quantize, n_outputs=2)
    y = np.array([tl(xx, rng_key=rng_key) for xx in x]).squeeze()
    inv_tl_0 = inv_aggregation(
        partial(get_p, nodeid=3), no_quantize, original_output_len=2, original_output_slot=0
    )
    y_inv_0 = np.array([inv_tl_0(yy[0], rng_key=rng_key) for yy in y]).squeeze()
    assert np.allclose(x, y_inv_0)
    inv_tl_1 = inv_aggregation(
        partial(get_p, nodeid=3), no_quantize, original_output_len=2, original_output_slot=1
    )
    y_inv_1 = np.array([inv_tl_1(yy[1], rng_key=rng_key) for yy in y]).squeeze()
    assert np.allclose(x, y_inv_1)
