from typing import (
    Any,
    Optional,
    Literal,
    Annotated,
    Union,
)
from pydantic import BaseModel, Field, model_validator
from biocomp.graphrules import GraphRewritingRule

# comp_g
# Index(['type', 'cdg_input', 'cdg_output', 'input_from', 'output_to',
#        'is_inverse_of', 'extra', 'source_id'],
#       dtype='object')
# cdg
# Index(['tu_id', 'type', 'predecessor', 'successor', 'content', 'content_type',
#        'params', 'is_output', 'is_input'],
#       dtype='object')


NodeType = Literal[
    "output",
    "sequestron_ERN",
    "translation",
    "transcription",
    "source",
    "aggregation",
    "inv_aggregation",
    "inv_source",
    "inv_transcription",
    "inv_translation",
    "input",
]


class Part(BaseModel):
    name: str
    category: str


class GraphEdge(BaseModel):
    source_id: int
    target_id: int
    output_slot: int
    input_slot: int
    content: tuple[Part, ...]
    content_type: Optional[Literal["DNA", "RNA", "PRT"]] = None
    content_embedding_names: dict[str, tuple[str]] = {}  # 'tl_rate: ('0xUORF', '1xUORF')}


class InverseSpec(BaseModel):
    # inverse nodes always have only one input and one output
    # but we need to store the original output slot id so that we can use it
    # when converting aggregation nodes for example
    node_id: int
    output_slot: int
    output_len: int


class GraphNode(BaseModel):
    node_id: int
    node_type: NodeType
    is_inverse_of: Optional[InverseSpec] = None
    extra: dict = {}


class GraphState(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


def apply_rule(rule: GraphRewritingRule, graph: GraphState, **kw) -> list[GraphState]: ...
