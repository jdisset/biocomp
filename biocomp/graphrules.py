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

    Either source_var or target_var (or both) can be None to match edges
    based only on properties or one endpoint.

    The contains property can specify a list of part names that must be present
    in the edge's content (as a subset check).

    When used in bind_edges with bind_endpoints=True (default), automatically creates
    node bindings for the edge endpoints as "{edge_name}_source" and "{edge_name}_target".
    """

    source_var: Optional[str] = None
    target_var: Optional[str] = None
    properties: dict[str, Any] = {}
    contains: Optional[list[str]] = None  # list of part names that must be in edge content
    bind_endpoints: bool = True  # automatically bind source and target nodes


class MatchQuery(BaseModel):
    """
    Declarative query to find and bind variables to subgraphs.
    Result is a list of "match dictionaries" that map variable names to nodes/edges.
    e.g., to find ERN nodes -> [{"negative": 10, "positive": 5}, {"rna_edge": edge_obj}, ...]
    """

    # bind variables to nodes that satisfy property constraints
    bind: dict[str, PropertyConstraint] = {}

    # bind variables to edges that satisfy property constraints
    bind_edges: dict[str, EdgeConstraint] = {}

    # the required topology of the matched subgraph
    where_connected: list[EdgeConstraint] = []
    # define topology that must NOT exist for a match to be valid
    where_not_connected: list[EdgeConstraint] = []

    # For complex, non-structural logic.
    # eval'd with the context of each match dict.
    where_filter_function: Optional[str] = None

    @model_validator(mode="after")
    def check_variable_consistency(self) -> "MatchQuery":
        """ensures all variables used in constraints are defined in `bind` or `bind_edges`."""
        bound_vars = set(self.bind.keys())
        bound_vars.add("any")  # special case for "any" node matching

        for edge in self.where_connected:
            if edge.source_var is not None and edge.source_var not in bound_vars:
                raise ValueError(
                    f"Variable '{edge.source_var}' in `where_connected` is not defined in `bind`."
                )
            if edge.target_var is not None and edge.target_var not in bound_vars:
                raise ValueError(
                    f"Variable '{edge.target_var}' in `where_connected` is not defined in `bind`."
                )

        for edge in self.where_not_connected:
            if edge.source_var is not None and edge.source_var not in bound_vars:
                raise ValueError(
                    f"Variable '{edge.source_var}' in `where_not_connected` is not defined in `bind`."
                )
            if edge.target_var is not None and edge.target_var not in bound_vars:
                raise ValueError(
                    f"Variable '{edge.target_var}' in `where_not_connected` is not defined in `bind`."
                )

        # Validate bind_edges - edge constraints need source_var and target_var to be in bind
        for edge_var, edge_constraint in self.bind_edges.items():
            if (
                edge_constraint.source_var is not None
                and edge_constraint.source_var not in bound_vars
            ):
                raise ValueError(
                    f"Edge '{edge_var}' source_var '{edge_constraint.source_var}' not defined in `bind`."
                )
            if (
                edge_constraint.target_var is not None
                and edge_constraint.target_var not in bound_vars
            ):
                raise ValueError(
                    f"Edge '{edge_var}' target_var '{edge_constraint.target_var}' not defined in `bind`."
                )

        # Check for conflicts between auto-generated endpoint names and manually bound nodes
        for edge_var, edge_constraint in self.bind_edges.items():
            if edge_constraint.bind_endpoints:
                auto_source_name = f"{edge_var}_source"
                auto_target_name = f"{edge_var}_target"
                if auto_source_name in bound_vars:
                    raise ValueError(
                        f"Auto-generated node binding '{auto_source_name}' conflicts with manually bound node. "
                        f"Either rename your node binding or set bind_endpoints=False for edge '{edge_var}'."
                    )
                if auto_target_name in bound_vars:
                    raise ValueError(
                        f"Auto-generated node binding '{auto_target_name}' conflicts with manually bound node. "
                        f"Either rename your node binding or set bind_endpoints=False for edge '{edge_var}'."
                    )

        return self


class ActionBase(BaseModel):  # define the transformations to apply for each match.
    pass


class AddNode(ActionBase):
    action_type: Literal["add_node"] = "add_node"
    # the local name used to refer to this new node in subsequent actions.
    local_name: str
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


class DeleteProperties(ActionBase):
    action_type: Literal["delete_properties"] = "delete_properties"
    node_var: str  # name of the variable (from match or a new node) to modify.
    property_keys: list[str]  # list of property keys to delete.


class DeleteNode(ActionBase):
    action_type: Literal["delete_node"] = "delete_node"
    node_var: str  # the variable name of the node to delete from the match.


class DeleteEdge(ActionBase):
    action_type: Literal["delete_edge"] = "delete_edge"
    source_var: str
    target_var: str


class RewireEdgesFrom(ActionBase):
    action_type: Literal["rewire_edges_from"] = "rewire_edges_from"
    old_source_var: str
    new_source_var: str


class RewireEdgesTo(ActionBase):
    action_type: Literal["rewire_edges_to"] = "rewire_edges_to"
    old_target_var: str
    new_target_var: str


class EditEdge(ActionBase):
    action_type: Literal["edit_edge"] = "edit_edge"
    edge_var: str  # Name of the bound edge variable to modify
    source_var: Optional[str] = None  # New source node (if changing)
    target_var: Optional[str] = None  # New target node (if changing)
    properties: Optional[dict[str, Any]] = None  # New properties to set
    content: Optional[list[str]] = None  # New part names for content


class CopyEdge(ActionBase):
    action_type: Literal["copy_edge"] = "copy_edge"
    source_edge_var: str  # Name of the bound edge variable to copy from
    source_var: str  # New source node variable name
    target_var: str  # New target node variable name
    properties: Optional[dict[str, Any]] = None  # Additional/override properties
    content: Optional[list[str]] = None  # Override content (if None, copies original content)
    content_type: Optional[str] = None  # Override content_type (if None, copies original)


AnyAction = Annotated[
    Union[
        AddNode,
        AddEdge,
        SetProperties,
        DeleteNode,
        DeleteEdge,
        RewireEdgesFrom,
        RewireEdgesTo,
        EditEdge,
        CopyEdge,
        DeleteProperties,
    ],
    Field(discriminator="action_type"),
]


class GraphRewritingRule(BaseModel):
    name: str
    query: MatchQuery
    actions: list[AnyAction]
    run_until_stable: bool = False
    yield_strategy: Literal["batched", "per_match", "cartesian_product_by_key"] = "batched"
    # For cartesian_product_by_key: which variable to group by (e.g., "numeric")
    cartesian_product_key: Optional[str] = None
