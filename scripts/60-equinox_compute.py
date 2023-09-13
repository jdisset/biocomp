### {{{                          --     imports     --
import matplotlib

import biocomp as bc
from biocomp import datautils as du
from jax.tree_util import Partial as partial
from datetime import datetime
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import jax
from jax import jit, vmap, value_and_grad
import jax.numpy as jnp
import pickle

from biocomp import utils as ut
import scriptutils as su
import biocomp.datautils as du
from biocomp import train
from biocomp import compute as cmp
from biocomp import nodes as nd
from biocomp.nodes import translation
from evosax import CMA_ES
from evosax.utils import ESLog, FitnessShaper
import os
import joblib
import datetime

from matplotlib import pyplot as plt

dirname = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# matplotlib.use('agg')
matplotlib.rcParams['figure.dpi'] = 200

##────────────────────────────────────────────────────────────────────────────}}}

import equinox as eqx


class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size, out_size, key):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (out_size, in_size))
        self.bias = jax.random.normal(bkey, (out_size,))

    def __call__(self, x):
        return self.weight @ x + self.bias


class MultiSpeciesTransform(eqx.Module):
    """
    A transform (i.e transcription, translation, ...) that takes
    one or multiple quantities of a species as input and outputs a single quantity of the next species.
    The output is a sum of the outputs of a neural network applied to each species.
    """

    rates: jax.Array

    def __init__(self, in_shape, out_shape, rate_dim, qmasks, key):
        self.rates = jax.random.normal(key, (in_shape[0], rate_dim))


class TranslationLayer(eqx.Module):

    def __init__():
        pass

    def __call__(
        self,
        *values,
        quantile,
        key,
        quantization_masks,
        possible_rate_values,
        inner_f: eqx.nn.MLP,
        outer_f: eqx.nn.MLP,
    ):
        qrates = nd.quantize_masked(self.rates, possible_rate_values, quantization_masks)
        inner_keys = jax.random.split(key, len(values))
        val = jnp.array(values)
        inner_out = sum(inner_f(v, r, k) for v, r, k in zip(val, qrates, inner_keys))
        inner_out = ut.flat_concat(inner_out, quantile)
        return outer_f(inner_out)



def setup(self, config: ComputeConfigManager, stack):

    self.check()

    first_node = self.nodes[0].get_compute_node()
    self.f_type = first_node.type

    if self.f_type == 'input':
        self.f_out_shapes = [(1,)]
        self.f_input_shapes = [(1,)]
        self.is_built = True
        return

    # get the shapes of the inputs. We'll collect all the inputs for each node
    # to make sure they are all the same
    node_inputs = []  # list of list of (net_id, compute_node_id, slot_id)
    for n in self.nodes:
        ninp = n.get_compute_node().input_from
        node_inputs.append([(n.network_id, *i) for i in ninp])

    # get the shapes of the inputs
    all_input_shapes = []  # list of list of shapes
    for n_inp in node_inputs:
        input_shapes = []
        for input_net_id, input_compute_node_id, input_slot_id in n_inp:
            input_layer_id, _ = stack.node_map[(input_net_id, input_compute_node_id)]
            assert input_layer_id < self.layer_id, 'Input node is in a later layer'
            assert stack.layers[input_layer_id].is_built, 'Input layer is not built'
            input_layer_output_shapes = stack.layers[input_layer_id].f_out_shapes
            assert input_slot_id < len(
                input_layer_output_shapes
            ), f'Input slot {input_slot_id} is out of range'
            input_shapes.append(input_layer_output_shapes[input_slot_id])
        all_input_shapes.append(tuple(input_shapes))
    # they should all be the same
    assert len(set(all_input_shapes)) == 1
    self.f_input_shapes = all_input_shapes[0]

    n_outputs = len(first_node.output_to)

    impl = config.get_impl(self.f_type)(
        input_shapes=self.f_input_shapes, n_outputs=n_outputs, stack=stack
    )
    self.f_prepare, self.f_apply, self.f_out_shapes = impl
    self.is_built = True


def transform_nn(
    input_shapes,
    n_outputs,
    stack,
    transform_name,
    outer_wsize=64,
    outer_depth=4,
    inner_wsize=64,
    inner_depth=3,
    inner_outsize=8,
    rate_dim=1,
    tr_namespace='',
    quantization_names: list[str] = None,  # ordered list. ex: ['1xuorf', '2xuorf', ...]
    inner_activation_name=DEFAULT_ACTIVATION,
    outer_activation_name=DEFAULT_OUT_ACTIVATION,
    **_,
):

    assert quantization_names is not None, 'quantization_names should be provided'

    inner_activation = ACTIVATION_FUNCTIONS[inner_activation_name]
    outer_activation = ACTIVATION_FUNCTIONS[outer_activation_name]

    assert n_outputs == 1, f'NN transform only supports 1 output, got {n_outputs}'
    rate_name = f'{transform_name}_rate'

    # we separate between node_impl and shared_impl for performance during prepare
    # only node_impl has to be called for each node_id

    def __node_impl(*values, key, param_f, params, node_id):
        val = jnp.array(values)
        rshape = (val.shape[0], rate_dim)
        # first grab the continuous values for the rates, specific to this node
        individual_rate_name = f'{rate_name}_x{rshape[0]}'
        rates = param_f(
            individual_rate_name, init=ut.continuous_initializer(key, rshape), node_id=node_id
        )
        # then quantize them
        rates = get_quantized(rates, node_id=node_id, params=params, param_name=rate_name)
        return val, rates

    def __shared_impl(val, rates, quantile, key, param_f):

        assert val.shape[0] == rates.shape[0]
        k1, k2 = jax.random.split(key, 2)

        def inner(value, rate_embeding, key):
            """For a single source, computes a latent output from the concatenation of
            the rate embedding and the source value.
            All of these outputs will then be summed up and passed through a final layer.
            """
            # TODO idea: to give more flexibility, we could add the index of the
            # value as this might allow clever padding of the sum
            # we'd then need to make sure that the index is unique for each
            # while, probably, being random (to avoid any "preferred" order)

            if value.ndim == 0:
                value = value.reshape((1,))
            if rate_embeding.ndim == 0:
                rate_embeding = rate_embeding.reshape((1,))

            assert value.ndim == 1, f'In {transform_name}: {value.ndim} != 1: {value}'
            assert rate_embeding.ndim == 1

            inputs = ut.flat_concat(value, rate_embeding, quantile)

            out = inner_activation(
                dense_multilevel(
                    inputs,
                    inner_wsize,
                    inner_outsize,
                    depth=inner_depth,
                    param_f=partial(param_f, node_id=0, base_path=ut.SHARED_PATH),
                    key=key,
                    name=f'{tr_namespace}{transform_name}_inner',
                    activation=inner_activation,
                )
            )

            assert out.shape == (inner_outsize,)

            return out

        # first we apply the inner stack to all inputs and sum them:

        inner_keys = jax.random.split(k1, val.shape[0])
        inner_out = sum(inner(v, r, k) for v, r, k in zip(val, rates, inner_keys))
        inner_out = ut.flat_concat(inner_out, quantile)

        assert inner_out.shape == (inner_outsize + 1,)

        # then we apply a final outer layer to the summed output:
        return outer_activation(
            dense_multilevel(
                inner_out,
                outer_wsize,
                1,
                depth=outer_depth,
                param_f=partial(param_f, node_id=0, base_path=ut.SHARED_PATH),
                key=k2,
                name=f'{tr_namespace}{transform_name}_outer',
                activation=inner_activation,
            )
        )

    def prepare(params, vnodelist, key):
        # during prepare, we call _impl with dummy inputs + a param_function that
        # creates the parameters on the fly if they don't exist yet

        # qnames is a list of names for the rate values available in this stack (1xuORf, ...)
        # they all get an initial value that the rates will be quantized to
        init = ut.continuous_initializer(key, (len(quantization_names), rate_dim))
        init_param_if_needed(params, rate_name, init=init, base_path=ut.QVALS_PATH, node_id=0)

        maxid = max([vnode.node_id for vnode in vnodelist])

        for vnode in vnodelist:
            register_quantile_variable_ids(params, vnode, stack)
            generate_quantization_masks(
                quantization_names,
                params,
                rate_name,
                vnode,
                number_of_nodes_at_least=maxid + 1,
            )
            key, _ = jax.random.split(key)
            val, rates = __node_impl(
                *[np.zeros(shape) for shape in input_shapes],
                key=key,
                param_f=partial(init_param_if_needed, params, number_of_nodes_at_least=maxid + 1),
                params=params,
                node_id=vnode.node_id,
            )

        __shared_impl(
            val,
            rates,
            quantile=0,
            key=key,
            param_f=partial(init_param_if_needed, params),
        )

    def apply(*values, quantiles, params, node_id, key):
        assert len(values) == len(input_shapes)
        param_f = partial(get_param, params)  # read-only
        val, rates = __node_impl(*values, key=key, param_f=param_f, params=params, node_id=node_id)
        quantile = get_quantile_variables(params, node_id, quantiles, 1)
        return __shared_impl(val, rates, quantile, key, param_f)

    output_shape = [(1,)]

    return prepare, apply, output_shape
