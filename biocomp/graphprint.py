"""Interactive graph visualization for GraphState objects using Dash Cytoscape."""

import dash
from dash import html, dcc, Input, Output
import dash_cytoscape as cyto
import webbrowser
import threading
from .graphengine import GraphState, GraphNode, GraphEdge

NODE_COLORS = {
    "promoter": "#FF6B6B",
    "gene": "#4ECDC4",
    "terminator": "#45B7D1",
    "DNA": "#E8B4CB",
    "RNA": "#96CEB4",
    "PRT": "#FFEAA7",
    "transcription": "#DDA0DD",
    "translation": "#F0E68C",
    "aggregation": "#FFB347",
    "sequestron_ERN": "#FF69B4",
    "input": "#90EE90",
    "output": "#FFB6C1",
    "source": "#98FB98",
    "inv_transcription": "#E6E6FA",
    "inv_translation": "#F5F5DC",
    "inv_aggregation": "#FFDAB9",
    "inv_source": "#F0FFF0",
}

LAYOUTS = {
    "cose": {"name": "cose", "animate": True, "idealEdgeLength": 100, "nodeOverlap": 20},
    "circle": {"name": "circle", "animate": True, "radius": 200},
    "grid": {"name": "grid", "animate": True, "condense": True},
    "breadthfirst": {
        "name": "breadthfirst",
        "animate": True,
        "directed": True,
        "spacingFactor": 1.5,
        "avoidOverlap": True,
    },
    "horizontal": {"name": "preset", "animate": True},  # Uses preset with computed positions
    "concentric": {
        "name": "concentric",
        "animate": True,
        "minNodeSpacing": 100,
        "avoidOverlap": True,
    },
}


def _node_label(node: GraphNode) -> str:
    parts = [f"ID:{node.node_id}", node.node_type]
    for key in ["name", "tu_id", "protein_name"]:
        if node.extra.get(key):
            parts.append(f"{key}:{node.extra[key]}")
    return "\n".join(parts)


def _edge_label(edge: GraphEdge) -> str:
    parts = []
    if edge.content:
        parts.append(f"[{', '.join(p.name for p in edge.content)}]")
    if edge.content_type:
        parts.append(edge.content_type)
    if edge.output_slot or edge.input_slot:
        parts.append(f"{edge.output_slot}→{edge.input_slot}")
    return " ".join(parts)


def _compute_horizontal_positions(graph: GraphState) -> dict:
    """Compute horizontal left-to-right positions for nodes based on graph topology."""
    if not graph.nodes:
        return {}

    # Build adjacency list for forward edges (outgoing)
    adjacency = {node.node_id: [] for node in graph.nodes.values()}
    in_degree = {node.node_id: 0 for node in graph.nodes.values()}

    for edge in graph.edges.values():
        adjacency[edge.source_id].append(edge.target_id)
        in_degree[edge.target_id] += 1

    # Find root nodes (no incoming edges)
    roots = [node_id for node_id, degree in in_degree.items() if degree == 0]
    if not roots:
        roots = [next(iter(graph.nodes.keys()))]  # fallback to first node

    # BFS to assign levels (x-coordinates)
    levels = {}
    queue = [(root_id, 0) for root_id in roots]
    visited = set()

    while queue:
        node_id, level = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)
        levels[node_id] = level

        for neighbor in adjacency[node_id]:
            if neighbor not in visited:
                queue.append((neighbor, level + 1))

    # Assign positions: x based on level, y distributed within level
    positions = {}
    level_groups = {}
    for node_id, level in levels.items():
        if level not in level_groups:
            level_groups[level] = []
        level_groups[level].append(node_id)

    for level, node_ids in level_groups.items():
        y_step = 150 if len(node_ids) > 1 else 0
        y_start = -y_step * (len(node_ids) - 1) / 2
        for i, node_id in enumerate(node_ids):
            positions[node_id] = {
                "x": level * 200,  # 200px between levels
                "y": y_start + i * y_step,
            }

    return positions


def _create_elements(graph: GraphState, use_horizontal=False) -> list:
    elements = []

    # Compute horizontal positions if requested
    horizontal_positions = _compute_horizontal_positions(graph) if use_horizontal else {}

    # Add nodes
    for node in graph.nodes.values():
        element = {
            "data": {
                "id": str(node.node_id),
                "label": _node_label(node),
                "node_type": node.node_type,
            },
            "classes": f"node-{node.node_type}",
        }

        # Add position for horizontal layout
        if use_horizontal and node.node_id in horizontal_positions:
            element["position"] = horizontal_positions[node.node_id]

        elements.append(element)

    # Add edges
    for edge in graph.edges.values():
        elements.append(
            {
                "data": {
                    "id": f"edge-{edge.source_id}-{edge.target_id}",
                    "source": str(edge.source_id),
                    "target": str(edge.target_id),
                    "label": _edge_label(edge),
                }
            }
        )

    return elements


def _create_stylesheet() -> list:
    stylesheet = [
        {
            "selector": "node",
            "style": {
                "width": "80px",
                "height": "80px",
                "shape": "ellipse",
                "background-color": "#CCC",
                "border-width": "1px",
                "border-color": "#444",
                "font-size": "10px",
                "text-wrap": "wrap",
                "text-max-width": "70px",
                "text-valign": "center",
                "text-halign": "center",
                "label": "data(label)",
            },
        },
        {
            "selector": "edge",
            "style": {
                "curve-style": "bezier",
                "target-arrow-shape": "triangle",
                "arrow-scale": 1.5,
                "line-color": "#666",
                "target-arrow-color": "#666",
                "width": "2px",
                "font-size": "8px",
                "label": "data(label)",
                "text-rotation": "autorotate",
                "text-margin-y": "-10px",
                "text-background-color": "white",
                "text-background-opacity": 0.8,
                "text-background-padding": "2px",
            },
        },
        {"selector": "node:selected", "style": {"border-width": "3px", "border-color": "#ff7f0e"}},
        {
            "selector": "edge:selected",
            "style": {"line-color": "#ff7f0e", "target-arrow-color": "#ff7f0e", "width": "3px"},
        },
    ]

    # Add color rules for each node type
    for node_type, color in NODE_COLORS.items():
        stylesheet.append(
            {"selector": f"[node_type = '{node_type}']", "style": {"background-color": color}}
        )

    return stylesheet


def show_graph(
    graph: GraphState,
    title: str = "biocomp graph",
    port: int = 8050,
    auto_open: bool = True,
    layout_name: str = "horizontal",
) -> None:
    """Show interactive graph visualization using Dash Cytoscape."""

    # Create elements with horizontal positioning if needed
    use_horizontal = layout_name == "horizontal"
    elements = _create_elements(graph, use_horizontal=use_horizontal)
    stylesheet = _create_stylesheet()

    app = dash.Dash(__name__, title=title)

    app.layout = html.Div(
        [
            cyto.Cytoscape(
                id="cytoscape-graph",
                elements=elements,
                stylesheet=stylesheet,
                layout=LAYOUTS[layout_name],
                style={"width": "100%", "height": "90vh"},
            ),
            html.Div(
                [
                    html.Label("Layout: ", style={"marginRight": "10px", "color": "#666"}),
                    dcc.Dropdown(
                        id="layout-dropdown",
                        options=[
                            {"label": k.replace("-", " ").title(), "value": k}
                            for k in LAYOUTS.keys()
                        ],
                        value=layout_name,
                        style={"width": "140px", "fontSize": "14px"},
                        clearable=False,
                    ),
                ],
                style={
                    "padding": "10px 20px",
                    "display": "flex",
                    "alignItems": "center",
                    "backgroundColor": "#f8f9fa",
                    "borderTop": "1px solid #eee",
                },
            ),
            html.Div(
                id="selection-info",
                style={
                    "position": "absolute",
                    "top": "20px",
                    "left": "20px",
                    "background": "rgba(255,255,255,0.96)",
                    "border": "1px solid #ddd",
                    "borderRadius": "8px",
                    "padding": "16px",
                    "fontSize": "13px",
                    "maxWidth": "350px",
                    "maxHeight": "70vh",
                    "overflow": "auto",
                    "boxShadow": "0 4px 16px rgba(0,0,0,0.2)",
                    "zIndex": 1000,
                    "fontFamily": "system-ui, -apple-system, sans-serif",
                    "lineHeight": "1.4",
                },
                children="Click a node or edge to see details",
            ),
        ],
        style={
            "height": "100vh",
            "fontFamily": "system-ui, -apple-system, sans-serif",
            "position": "relative",
        },
    )

    @app.callback(
        [Output("cytoscape-graph", "layout"), Output("cytoscape-graph", "elements")],
        Input("layout-dropdown", "value"),
    )
    def update_layout(layout_name):
        # Recreate elements with horizontal positioning if needed
        use_horizontal = layout_name == "horizontal"
        new_elements = _create_elements(graph, use_horizontal=use_horizontal)
        return LAYOUTS.get(layout_name, LAYOUTS["cose"]), new_elements

    @app.callback(
        Output("selection-info", "children"),
        [
            Input("cytoscape-graph", "selectedNodeData"),
            Input("cytoscape-graph", "selectedEdgeData"),
        ],
    )
    def show_selection_info(selected_nodes, selected_edges):
        def format_node(node):
            content = [
                html.Div(
                    [
                        html.Span("🔵 ", style={"fontSize": "16px"}),
                        html.Span(
                            f"NODE {node.node_id}",
                            style={"fontWeight": "bold", "color": "#2c3e50", "fontSize": "15px"},
                        ),
                    ],
                    style={"marginBottom": "8px"},
                ),
                html.Div(
                    [
                        html.Span("Type: ", style={"fontWeight": "600", "color": "#34495e"}),
                        html.Span(
                            node.node_type,
                            style={
                                "color": "#e74c3c",
                                "fontWeight": "500",
                                "fontFamily": "monospace",
                            },
                        ),
                    ],
                    style={"marginBottom": "6px"},
                ),
            ]

            if node.is_inverse_of:
                content.append(
                    html.Div(
                        [
                            html.Span(
                                "Inverse of: ", style={"fontWeight": "600", "color": "#34495e"}
                            ),
                            html.Span(
                                f"Node {node.is_inverse_of.node_id} ", style={"color": "#3498db"}
                            ),
                            html.Span(
                                f"(slot {node.is_inverse_of.output_slot}, len {node.is_inverse_of.output_len})",
                                style={"color": "#7f8c8d", "fontSize": "12px"},
                            ),
                        ],
                        style={"marginBottom": "6px"},
                    )
                )

            if node.extra:
                content.append(
                    html.Div(
                        [
                            html.Span("📝 ", style={"fontSize": "14px"}),
                            html.Span(
                                "Extra Properties", style={"fontWeight": "600", "color": "#8e44ad"}
                            ),
                        ],
                        style={"marginTop": "10px", "marginBottom": "6px"},
                    )
                )

                for key, value in node.extra.items():
                    content.append(
                        html.Div(
                            [
                                html.Span(
                                    f"  {key}: ", style={"fontWeight": "600", "color": "#2c3e50"}
                                ),
                                html.Span(
                                    str(value),
                                    style={"color": "#27ae60", "fontFamily": "monospace"},
                                ),
                            ],
                            style={"marginLeft": "10px", "marginBottom": "4px"},
                        )
                    )

            return html.Div(content)

        def format_edge(edge):
            content = [
                html.Div(
                    [
                        html.Span("🔗 ", style={"fontSize": "16px"}),
                        html.Span(
                            f"EDGE {edge.source_id} → {edge.target_id}",
                            style={"fontWeight": "bold", "color": "#2c3e50", "fontSize": "15px"},
                        ),
                    ],
                    style={"marginBottom": "8px"},
                )
            ]

            if edge.output_slot != 0 or edge.input_slot != 0:
                content.append(
                    html.Div(
                        [
                            html.Span("Slots: ", style={"fontWeight": "600", "color": "#34495e"}),
                            html.Span(
                                f"{edge.output_slot} → {edge.input_slot}",
                                style={"color": "#e67e22", "fontFamily": "monospace"},
                            ),
                        ],
                        style={"marginBottom": "6px"},
                    )
                )

            if edge.content_type:
                content.append(
                    html.Div(
                        [
                            html.Span(
                                "Content Type: ", style={"fontWeight": "600", "color": "#34495e"}
                            ),
                            html.Span(
                                edge.content_type, style={"color": "#9b59b6", "fontWeight": "500"}
                            ),
                        ],
                        style={"marginBottom": "6px"},
                    )
                )

            if edge.content:
                content.append(
                    html.Div(
                        [
                            html.Span("🧬 ", style={"fontSize": "14px"}),
                            html.Span(
                                f"Content ({len(edge.content)} parts)",
                                style={"fontWeight": "600", "color": "#16a085"},
                            ),
                        ],
                        style={"marginTop": "10px", "marginBottom": "6px"},
                    )
                )

                for part in edge.content:
                    content.append(
                        html.Div(
                            [
                                html.Span("  • ", style={"color": "#95a5a6"}),
                                html.Span(
                                    part.name, style={"fontWeight": "500", "color": "#2c3e50"}
                                ),
                                html.Span(
                                    f" ({part.category})",
                                    style={"color": "#7f8c8d", "fontSize": "12px"},
                                ),
                            ],
                            style={"marginLeft": "10px", "marginBottom": "3px"},
                        )
                    )

            if edge.content_embedding_names:
                content.append(
                    html.Div(
                        [
                            html.Span("🏷️ ", style={"fontSize": "14px"}),
                            html.Span(
                                "Embeddings", style={"fontWeight": "600", "color": "#d35400"}
                            ),
                        ],
                        style={"marginTop": "10px", "marginBottom": "6px"},
                    )
                )

                for key, value in edge.content_embedding_names.items():
                    val_str = (
                        value[0] if isinstance(value, tuple) and len(value) == 1 else str(value)
                    )
                    content.append(
                        html.Div(
                            [
                                html.Span(
                                    f"  {key}: ", style={"fontWeight": "600", "color": "#2c3e50"}
                                ),
                                html.Span(
                                    val_str, style={"color": "#f39c12", "fontFamily": "monospace"}
                                ),
                            ],
                            style={"marginLeft": "10px", "marginBottom": "4px"},
                        )
                    )

            return html.Div(content)

        if selected_nodes:
            node_id = int(selected_nodes[0]["id"])
            node = next((n for n in graph.nodes.values() if n.node_id == node_id), None)
            return format_node(node) if node else f"NODE {node_id} not found"
        elif selected_edges:
            edge_data = selected_edges[0]
            source_id = int(edge_data["source"])
            target_id = int(edge_data["target"])
            edge = next(
                (
                    e
                    for e in graph.edges.values()
                    if e.source_id == source_id and e.target_id == target_id
                ),
                None,
            )
            return format_edge(edge) if edge else f"EDGE {source_id} → {target_id} not found"
        else:
            return html.Div(
                "Click a node or edge to see details",
                style={"color": "#7f8c8d", "fontStyle": "italic"},
            )

    if auto_open:
        threading.Timer(1, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()

    print(f"Graph visualization at http://127.0.0.1:{port}")
    app.run(debug=False, port=port, host="127.0.0.1")


plot_graph = show_graph
