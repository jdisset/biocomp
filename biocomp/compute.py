from .library import PartsLibrary as PartsLibrary
import jax
from jax.tree_util import Partial as partial
import jax.numpy as jnp
from . import utils as ut

from . import nodes as nd


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                  --     Params and quantization     --
# ···············································································
def get_param(
    params,
    name,
    init=None,
    overwrite_with=None,
    shared=False,
    node_id=None,
    node_namespace=None,
    clip_to=None,
    constraints=None,
    read_only=True,
):
    assert isinstance(params, dict), f'params must be a dict, not {type(params)}'
    assert constraints is None or isinstance(
        constraints, dict
    ), f'constraints must be a dict, not {type(constraints)}'

    dpath = []
    if not shared:
        assert node_id is not None
        dpath.append('node')
        if node_namespace is not None:
            dpath.append(node_namespace)
        dpath.append(node_id)
    else:
        dpath.append('shared')

    dpath.append(name)

    assert (init is None) != (overwrite_with is None)

    if overwrite_with is not None and read_only is False:
        ut.at_path(params, dpath, overwrite_with)
        return overwrite_with

    if ut.at_path(params, dpath) is None:
        assert read_only is False
        try:
            r = ut.at_path(params, dpath, init())
            if clip_to is not None:
                assert (
                    constraints is not None
                ), 'clip_to requires to pass a constraints dict to get_param'
                r = jnp.clip(r, *clip_to)
                ut.at_path(params, dpath, r)
                c = ut.at_path(constraints, ['clip'], defaultinit=list)
                c.append((tuple(dpath), clip_to))
                ut.at_path(constraints, ['clip'], c)

        except Exception as e:
            print(f'Error initializing param "{name}" from node {node_id}: {e}')
            raise e
    return ut.at_path(params, dpath)


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


def get_quantized(
    get_param, param_name, values, node_id, cdf, cdg, quantize_fun, mode='input_edges'
):
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
        jnp.array([get_param(n, lambda: val, shared=True) for n in names])
        for names, val in zip(possible_names, values)
    ]
    res = jnp.array([quantize_fun(v, p) for v, p in zip(values, possible_values)])
    return res


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                  --     ComputeGraphModel class     --
# ···············································································


class ComputeGraphModel:
    def __init__(self, network):
        self.network = network
        self.n_inputs = len(
            self.network.compute_graph[self.network.compute_graph['type'] == 'input']
        )

        self.built = False

    def build(self, node_remap=dict(), node_namespace=None):
        self.node_namespace = node_namespace
        assert self.network is not None
        assert self.network.is_built()

        # node_remap is a dictionnary that maps "vanilla" node types to
        # new names. Useful to try different node implementations
        # (e.g translation -> custom_translation_v2)

        # let's do everything we can before collect_all_results to avoid long compile times
        batches = self.__get_batch_sequence_of_nodes()
        flat_batches = [item for sublist in batches for item in sublist]
        call_dicts = []
        output_node = None
        nid_to_call_dict = dict()
        for i, nid in enumerate(flat_batches):
            call_d = {}
            node_row = self.network.compute_graph.loc[nid]
            # if it's an inverse node:
            nodeid_for_getters = nid
            if node_row.extra is not None and 'is_inverse_of' in node_row.extra:
                nodeid_for_getters = node_row.extra['is_inverse_of']
            get_p = partial(
                get_param,
                node_id=nodeid_for_getters,
            )
            get_q = partial(
                get_quantized,
                node_id=nodeid_for_getters,
                cdf=self.network.compute_graph,
                cdg=self.network.central_dogma_graph,
                quantize_fun=nd.quantize,
            )
            extra_params = {
                'n_outputs': len(self.network.compute_graph.loc[nid]['output_to']),
                'n_inputs': len(self.network.compute_graph.loc[nid]['input_from']),
            }
            if node_row.extra is not None:
                extra_params.update(node_row.extra)

            call_d['get_p'] = get_p
            call_d['get_q'] = get_q
            call_d['extra_params'] = extra_params
            call_d['type'] = node_row.type
            call_d['input_from'] = node_row.input_from
            call_d['fun'] = None
            call_d['nid'] = nid
            nid_to_call_dict[nid] = i

            fun_name = node_remap.get(node_row.type, node_row.type)
            if node_row.type not in ('input'):
                assert fun_name in nd.COMPUTE_NODES_DICT, f'Unimplemented node type {fun_name}'
                call_d['fun'] = nd.COMPUTE_NODES_DICT[fun_name]
            if node_row.type == 'output':
                output_node = nid

            call_dicts.append(call_d)

        def recursive_eval(params, inputs, rng_key, read_only=True, constraints=None):
            def evalnode(n, key):
                if n['type'] == 'input':
                    return inputs[n['extra_params']['input_position']]

                get_p = partial(
                    n['get_p'],
                    params,
                    node_namespace=self.node_namespace,
                    constraints=constraints,
                    read_only=read_only,
                )
                get_q = partial(n['get_q'], get_p)

                assert callable(n['fun'])
                comp = n['fun'](get_p, get_q, **n['extra_params'])

                upstream_results = []
                n_inp = len(n['input_from'])
                keys = jax.random.split(key, n_inp)
                for inp, k in zip(n['input_from'], keys):
                    res = evalnode(call_dicts[nid_to_call_dict[inp[0]]], k)
                    if len(res.shape) == 0:
                        upstream_results.append(res)
                    else:
                        upstream_results.append(res[inp[1]])

                return comp(*upstream_results, rng_key=key)

            assert output_node is not None
            return evalnode(call_dicts[nid_to_call_dict[output_node]], rng_key)

        def collect_all_results(params, inputs, rng_key, read_only=True, constraints=None):
            assert (
                len(inputs) == self.n_inputs
            ), f'len(inputs)={len(inputs)} != n_inputs={self.n_inputs}'

            keys = jax.random.split(rng_key, len(flat_batches))

            results = {}

            for (n, key) in zip(call_dicts, keys):
                nid = n['nid']

                if n['type'] == 'input':
                    results[nid] = inputs[n['extra_params']['input_position']]
                    continue

                upstream_results = []
                for inp in n['input_from']:
                    if len(results[inp[0]].shape) == 0:
                        upstream_results.append(results[inp[0]])
                    else:
                        upstream_results.append(results[inp[0]][inp[1]])

                get_p = partial(
                    n['get_p'],
                    params,
                    node_namespace=self.node_namespace,
                    constraints=constraints,
                    read_only=read_only,
                )
                get_q = partial(n['get_q'], get_p)
                comp_node = n['fun'](get_p, get_q, **n['extra_params'])
                res = comp_node(*upstream_results, rng_key=key)
                results[nid] = res

                if n['type'] == 'output':
                    return res, results

            raise ValueError('Invalid compute graph, no output node found')

        def apply(*args, **kwargs):
            return collect_all_results(*args, **kwargs)[0]

        def init(rng_key, pre_params=None, pre_constraints=None):
            params = {} if pre_params is None else pre_params
            constraints = {} if pre_constraints is None else pre_constraints
            apply(
                params,
                [jnp.array([1.0])] * self.n_inputs,
                rng_key,
                constraints=constraints,
                read_only=False,
            )
            return params, constraints

        self.apply = apply
        self.collect_all_results = collect_all_results
        self.recursive_eval = recursive_eval
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
                msg = f'Invalid compute graph, no independent nodes found. Remaining nodes: {set(self.network.compute_graph.index) - visited}. visited={visited}'
                raise ValueError(msg)
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
