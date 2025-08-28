from typing import (
    Any,
    Optional,
    Literal,
    Annotated,
    Union,
)
from pydantic import BaseModel, Field, model_validator


class PropertyConstraint(BaseModel):
    """
    property-based constraints for a single node variable.
    Example: {"type": "PRT", "is_output": True}
    """

    properties: dict[str, Any]


class EdgeConstraint(BaseModel):
    """
    Defines a required (or forbidden) edge between two node variables.
    Example: source_var="rna", target_var="protein"
    """

    source_var: str
    target_var: str
    properties: dict[str, Any] = {}  # optional constraints on edge properties


class MatchQuery(BaseModel):
    """
    Declarative query to find and bind variables to subgraphs.
    Result is a list of "match dictionaries" that map variable names to node IDs.
    e.g., to find ERN nodes -> [{"negative": 10, "positive": 5}, {"negative": 12, "positive": 7}, ...]
    """

    # bind variables to nodes that satisfy property constraints
    bind: dict[str, PropertyConstraint]

    # the required topology of the matched subgraph
    where_connected: list[EdgeConstraint] = []
    # define topology that must NOT exist for a match to be valid
    where_not_connected: list[EdgeConstraint] = []

    # For complex, non-structural logic.
    # will call a registered Python function with the match dictionary.
    # e.g., "are_params_compatible(protein, rna)"
    where_filter_function: Optional[str] = None

    @model_validator(mode="after")
    def check_variable_consistency(self) -> "MatchQuery":
        """ensures all variables used in constraints are defined in `bind`."""
        bound_vars = set(self.bind.keys())
        for edge in self.where_connected:
            if edge.source_var not in bound_vars:
                raise ValueError(
                    f"Variable '{edge.source_var}' in `where_connected` is not defined in `bind`."
                )
            if edge.target_var not in bound_vars:
                raise ValueError(
                    f"Variable '{edge.target_var}' in `where_connected` is not defined in `bind`."
                )

        return self


class ActionBase(BaseModel):  # define the transformations to apply for each match.
    pass


class AddNode(ActionBase):
    action_type: Literal["add_node"] = "add_node"
    # the local name used to refer to this new node in subsequent actions.
    local_name: str
    # properties for the new node. Can use Jinja2-style templates to reference
    # properties of matched nodes, e.g., {"type": "translation", "tu_id": "{{protein.tu_id}}"}.
    properties: dict[str, Any]


class AddEdge(ActionBase):
    action_type: Literal["add_edge"] = "add_edge"
    # Source and target can be names of variables from the MatchQuery
    # or local_names from a preceding AddNode action.
    source: str
    target: str
    properties: dict[str, Any] = {}


class SetProperties(ActionBase):
    action_type: Literal["set_properties"] = "set_properties"
    node_var: str  # name of the variable (from match or a new node) to modify.
    properties: dict[str, Any]  # dictionary of properties to set. Also supports templating.


class DeleteNode(ActionBase):
    # also delete the edges connected to this node
    action_type: Literal["delete_node"] = "delete_node"
    node_var: str  # the variable name of the node to delete from the match.


class DeleteEdge(ActionBase):
    action_type: Literal["delete_edge"] = "delete_edge"
    source_var: str
    target_var: str


# discriminated union of all possible action types.
# allows Pydantic to automatically parse based on the `action_type` field.
AnyAction = Annotated[
    Union[AddNode, AddEdge, SetProperties, DeleteNode, DeleteEdge],
    Field(discriminator="action_type"),
]


class GraphRewritingRule(BaseModel):
    name: str
    query: MatchQuery
    actions: list[AnyAction]
    run_until_stable: bool = False
