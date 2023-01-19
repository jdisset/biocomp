from .library import PartsLibrary as PartsLibrary
import jax
from jax.tree_util import Partial as partial
import jax.numpy as jnp
from . import utils as ut

from . import nodes as nd
from typing import List, Dict, Tuple, Union, Optional, Callable, Any


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
            msg = f'Error initializing param "{name}" from node {node_id}: {e}'
            raise RuntimeError(msg)
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
    get_param,
    param_name,
    values,
    node_id,
    cdf,
    cdg,
    quantize_fun,
    rng_key,
    mode='input_edges',
    logto=None,
):
    """Return a *quantized* version of the parameter, conditioned on the
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
        jnp.array(
            [
                get_param(n, ut.continuous_initializer(k, val.shape), shared=True)
                for n, k in zip(names, jax.random.split(kk, len(names)))
            ]
        )
        for names, val, kk in zip(possible_names, values, jax.random.split(rng_key, len(values)))
    ]

    if logto is not None:
        if node_id in logto:
            logto[node_id].add(param_name)
        else:
            logto[node_id] = {param_name}

    res = jnp.array([quantize_fun(v, p) for v, p in zip(values, possible_values)])
    return res


#                                                                            }}}

# {{{                  --     ComputeGraphModel class     --
# ···············································································


class ComputeGraphModel:
    def __init__(self, network):
        self.network = network
        cg = self.network.compute_graph
        assert len(cg[cg['type'] == 'output']) == 1, 'The graph must have exactly one output node'
        self.n_inputs = len(cg[cg['type'] == 'input'])
        self.n_outputs = len(cg[cg['type'] == 'output'].input_from[0])
        self.built = False

    def build(
        self,
        node_impl: Dict[str, Callable] = nd.DEFAULT_COMPUTE_NODES_DICT,
        node_namespace: Optional[str] = None,
    ):
        """Builds the model, i.e. creates a list of functions to be called in sequence
        (together with the necesary information to call them) according to the compute graph"""

        self.node_namespace = node_namespace
        assert self.network is not None
        assert self.network.is_built()
        assert isinstance(node_impl, dict)
        self.node_impl = node_impl

        cg = self.network.compute_graph

        # we'll build the list of call_dicts
        # a call dict is a dictionary that contains the relevant information in
        # order to actually call a node function

        batches = self.__get_batch_sequence_of_nodes()
        # each sublist in batches contains independent nodes that can be safely called in parallel
        # however we are not parallelizing the calls, so we can flatten the list and be sure
        # that the dependency order is respected (upstream nodes will always be called before downstream ones)
        flat_batches = [item for sublist in batches for item in sublist]
        call_dicts = []
        for nid in flat_batches:
            call_d = {}
            node_row = cg.loc[nid]

            # if it's an inverse node, we need to get the id of the node that it's inverting
            nodeid_for_getters = nid
            if node_row.extra is not None and 'is_inverse_of' in node_row.extra:
                nodeid_for_getters = node_row.extra['is_inverse_of']

            # a node needs a method to access the parameters.
            # we simply preset the general purpose get_param on the correct nodeid
            get_p = partial(
                get_param,
                node_id=nodeid_for_getters,
            )
            # same with get_quantized
            # we preset the node id, the central dogma graph and the compute graph
            # as well as the actual quantize function
            get_q = partial(
                get_quantized,
                node_id=nodeid_for_getters,
                cdf=cg,
                cdg=self.network.central_dogma_graph,
                quantize_fun=nd.quantize,
            )

            # extra_params will be passed to the node function
            # n_outputs and n_inputs are always passed
            # + any specific thing that the network builder has added
            extra_params = {
                'n_outputs': len(cg.loc[nid]['output_to']),
                'n_inputs': len(cg.loc[nid]['input_from']),
            }
            if node_row.extra is not None:
                extra_params.update(node_row.extra)

            call_d['get_p'] = get_p  # a node needs a method to access the parameters
            call_d['get_q'] = get_q  # and a method for quantization
            call_d['extra_params'] = extra_params  # these will be appended to the node call
            call_d['type'] = node_row.type
            call_d['input_from'] = node_row.input_from  # upstream node(s) we are taking input from
            call_d['fun'] = None  # the actual function to call (set later)
            call_d['nid'] = nid  # this node's id, important to store the output

            if node_row.type not in ('input'):
                assert node_row.type in self.node_impl, f'Unimplemented node type {node_row.type}'
                call_d['fun'] = self.node_impl[node_row.type]  # the actual function to call

            call_dicts.append(call_d)

        def collect_all_results(
            params: dict,
            inputs: jnp.ndarray,
            rng_key,
            quantiles=None,
            read_only=True,
            constraints=None,
        ) -> tuple[jnp.ndarray, dict[int, jnp.ndarray]]:
            """
            params: the parameters
            inputs: the inputs to the network
            quantiles: array of quantiles we want to estimate (1 per output)

            Executes and collects all the results of the nodes in the compute graph.
            This method basically just calls the nodes in call_dicts, in order, populating
            the result dictionnary with each node's output and feeding the output of each
            upstream node as input to the next downstream one."""
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

                extra_params = n['extra_params']
                qtl = None
                if quantiles is not None:
                    pick_quantile = n['extra_params'].get('quantile_variable_id', [])
                    if len(pick_quantile) > 0:
                        extra_params['quantile'] = quantiles[jnp.array(pick_quantile)]

                get_q = partial(n['get_q'], get_p, rng_key=key)
                comp_node = n['fun'](get_p, get_q, z=qtl, **extra_params)
                res = comp_node(*upstream_results, rng_key=key)
                results[nid] = res

                if n['type'] == 'output':
                    return res, results

            raise ValueError('Invalid compute graph, no output node found')

        def apply(*args, **kwargs):
            """Executes the model. It simply calls collect_all_results and only returns the final output"""
            return collect_all_results(*args, **kwargs)[0]

        def init(rng_key, pre_params=None, pre_constraints=None):
            params = {} if pre_params is None else pre_params
            constraints = {} if pre_constraints is None else pre_constraints
            apply(
                params,
                [jnp.array([1.0])] * self.n_inputs,
                rng_key,
                quantiles=[jnp.array([1.0])] * self.n_outputs,
                constraints=constraints,
                read_only=False,
            )
            return params, constraints

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
        """Returns a list of lists of compute nodes from the network,
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
        return self.network.get_output_proteins()

    def get_input_from_output(self, output_arr):
        return self.network.get_input_from_output(output_arr)

    def get_inverted_input_proteins(self):
        return self.network.get_inverted_input_proteins()

    def get_inverted_input_positions(self):
        return self.network.get_inverted_input_positions()

    def get_quantized_parameters_per_node_type(self, params):
        """Returns a dictionary with node types as keys and a list of quantized parameters used by that node type as values.
        When a parameter is quantized, the key to each value is the name of the quantized value (i.e '1x_uORF').
        When a parameter is not quantized, we record all of the values for this param in use in the network, and the key
        is the id of the node that uses this parameter with the specific given value.
        """
        node_dict = self.network.get_compute_types()
        params_per_type = {k: dict() for k in node_dict.keys()}
        pnode = params['node']
        if self.node_namespace is not None:
            pnode = pnode[self.node_namespace]
        for node_type, node_ids in node_dict.items():
            for node_id in node_ids:
                if node_id in pnode.keys():
                    for pname, v in pnode[node_id].items():
                        if pname in params_per_type[node_type].keys():
                            params_per_type[node_type][pname].update({node_id: v})
                        else:
                            params_per_type[node_type][pname] = {node_id: v}
        res = {}
        shared = params['shared']
        for ntype, pdict in params_per_type.items():  # pdict is dict(str,dict(int,np.ndarray))
            res[ntype] = {}
            # check if we have some quantization values for the parameters
            # a quantization value has a key of the form vname::param_name
            # we want to collect all the quantization values for the parameters of this node type
            for pname in pdict:
                matching = {k: v for k, v in shared.items() if k.endswith(f'::{pname}')}
                if len(matching) > 0:
                    res[ntype][pname] = matching
                else:
                    res[ntype][pname] = params_per_type[ntype][pname]
        res
        return res


#                                                                            }}}
