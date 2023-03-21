from .library import PartsLibrary as PartsLibrary
import jax
from jax.tree_util import Partial as partial
import jax.numpy as jnp
from . import utils as ut
from . import nodes as nd
from typing import List, Dict, Tuple, Union, Optional, Callable, Any
from .network import Network


class ComputeGraphModel:
    def __init__(self, networks: List[Network]):
        self.networks = networks

    def build(
        self,
        node_impl: Dict[str, Callable] = nd.DEFAULT_COMPUTE_NODES_DICT,
        node_namespace: Optional[str] = None,
    ):
        # build a "meta" network that contains all the nodes
        # they are linked through their output node

        # I have list of independent networks. Each network is composed of computation nodes, which correspond
        # to a function. There are only a few node types. I can execute nodes of a same type in a vectorized fashion,
        # using jax.vmap, which is very advantageous. I want to write a function that will take this list of networks,
        # and give me the order in which I should execute the nodes, maximizing the amount of vectorization.
        # Most nodes except input nodes (in a same network) require as input the output of other upstream nodes, so there is a dependency order to be respected. The input to this big function is a tree represented as a pandas dataframe where each node has a node_id, a type field, an input_from column and an output_to column. Input_from and output_to are stored as a list of (node_id,slot_number) tuples, i.e {'node_id':3, 'input_from':[(2,0),(5,1)], 'output_to':[(1,0)]} represents a node (with id 2) that takes 2 inputs: one from the first output of node number 3 and the other one from the second output of node number 5. It outputs to the first input slot of node number 1.

        pass
