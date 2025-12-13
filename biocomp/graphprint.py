"""Graph visualization for GraphState objects.

Interactive HTML visualization using Dash Cytoscape and text-based ASCII visualization
for console debugging.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from .network import Network

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
    if edge.from_output_slot or edge.to_input_slot:
        parts.append(f"{edge.from_output_slot}→{edge.to_input_slot}")
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
    import webbrowser
    import threading

    import dash
    from dash import html, dcc, Input, Output
    import dash_cytoscape as cyto

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

            if edge.from_output_slot != 0 or edge.to_input_slot != 0:
                content.append(
                    html.Div(
                        [
                            html.Span("Slots: ", style={"fontWeight": "600", "color": "#34495e"}),
                            html.Span(
                                f"{edge.from_output_slot} → {edge.to_input_slot}",
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


# ════════════════════════════════════════════════════════════════════════════════
# TEXT-BASED GRAPH PRINTING (ASCII visualization for console/logging)
# ════════════════════════════════════════════════════════════════════════════════

TYPE_SHORTNAMES = {
    "source": "src",
    "transcription": "tc",
    "translation": "tl",
    "sequestron_ERN": "ern",
    "aggregation": "agg",
    "output": "out",
    "bias": "bias",
    "numeric": "num",
    "input": "inp",
    "inv_source": "i_src",
    "inv_transcription": "i_tc",
    "inv_translation": "i_tl",
    "inv_aggregation": "i_agg",
}


def _get_type_short(node_type: str) -> str:
    return TYPE_SHORTNAMES.get(node_type, node_type[:4])


def _get_tu_name(node: GraphNode, graph: GraphState) -> str:
    """Generate TU name: {cotx_name}::{tu_name}::{position}"""
    cotx = node.extra.get("cotx_group", "")
    tu = node.extra.get("tu_name") or node.extra.get("name", "")
    if not cotx and not tu:
        return ""
    name = f"{cotx}::{tu}" if cotx else tu
    return name


def _build_layer_info(network: "Network") -> tuple[list[dict], dict[int, int]]:
    """Build layer information from a single network using topological ordering.

    Returns:
        (layers, node_to_layer): layers is a list of dicts with 'type', 'nodes'
        node_to_layer maps node_id -> layer_index
    """
    from .stack_builder import topological_order

    graph = network.compute_graph
    if graph is None:
        return [], {}

    batches = topological_order(graph)

    # group nodes by type within each batch, then merge same-type groups across batches
    layers = []
    node_to_layer = {}

    for batch in batches:
        # group by node type
        type_groups: dict[str, list[int]] = defaultdict(list)
        for node_id in batch:
            node = graph.nodes[node_id]
            type_groups[node.node_type].append(node_id)

        # each type group becomes a layer
        for node_type, node_ids in type_groups.items():
            layer_idx = len(layers)
            layers.append({"type": node_type, "nodes": sorted(node_ids)})
            for nid in node_ids:
                node_to_layer[nid] = layer_idx

    return layers, node_to_layer


def _format_embedding_value(value) -> str:
    """Format an embedding value for display."""
    if isinstance(value, tuple):
        if len(value) == 1:
            return str(value[0])
        return f"({len(value)} opts)"
    return str(value)


def _is_unlocked(value) -> bool:
    """Check if an embedding value is unlocked (multiple options)."""
    return isinstance(value, (list, tuple)) and len(value) > 1


class GraphPrinter:
    """ASCII graph printer for Network objects."""

    def __init__(self, network: "Network"):
        self.network = network
        self.graph = network.compute_graph
        self.layers, self.node_to_layer = _build_layer_info(network)

    def format_header(self) -> str:
        """Format the header box."""
        if self.graph is None:
            return "Empty graph"

        name = self.network.name or "unnamed"
        n_nodes = len(self.graph.nodes)
        n_edges = len(self.graph.edges)

        inner = f" COMPUTE GRAPH: {name}"
        stats = f"{n_nodes} nodes │ {n_edges} edges "
        width = max(80, len(inner) + len(stats) + 4)

        lines = [
            "┏" + "━" * (width - 2) + "┓",
            f"┃{inner}{' ' * (width - len(inner) - len(stats) - 2)}{stats}┃",
            "┗" + "━" * (width - 2) + "┛",
        ]
        return "\n".join(lines)

    def format_column_headers(self) -> str:
        """Format the layer column headers."""
        if not self.layers:
            return ""

        headers = []
        for i, layer in enumerate(self.layers):
            short = _get_type_short(layer["type"])
            headers.append(f"L{i}:{short}")

        # compute column width
        col_width = max(self.min_col_width, max(len(h) for h in headers) + 2)

        label_width = max(len(lbl) for lbl in self.row_labels) if self.row_labels else 0
        label_width = max(label_width, 20)

        header_line = " " * label_width + "  "
        underline = " " * label_width + "  "

        for h in headers:
            header_line += h.center(col_width)
            underline += "─" * len(h).center(col_width) if len(h) < col_width else "─" * col_width

        return header_line + "\n" + underline

    def format_node_table(self) -> str:
        """Format the node table."""
        if self.graph is None:
            return ""

        lines = [
            f"\nNODES ({len(self.graph.nodes)} total)",
            "─" * 100,
            f" {'ID':>3} │ {'Type':<15} │ {'Name':<25} │ {'Layer':^5} │ {'Cotx':<8} │ Extra",
            "─" * 100,
        ]

        # Sort by layer index first, then by node_id
        sorted_nodes = sorted(
            self.graph.nodes.values(),
            key=lambda n: (self.node_to_layer.get(n.node_id, 999), n.node_id),
        )

        for node in sorted_nodes:
            # Get name based on node type
            name = node.extra.get("name", "")
            if not name and node.extra.get("seq_name"):
                name = node.extra["seq_name"]
            name = name[:25]

            layer_idx = self.node_to_layer.get(node.node_id, -1)
            layer_str = f"L{layer_idx}" if layer_idx >= 0 else "-"
            cotx = node.extra.get("cotx_group", "-")[:8]

            # Build extra info based on node type
            extra_parts = []
            if node.node_type == "source":
                # Show ratio with range if unlocked
                ratio_range = node.extra.get("ratio_range")
                if ratio_range:
                    rmin = ratio_range.get("min", "?")
                    rmax = ratio_range.get("max", "?")
                    extra_parts.append(f"ratio=[{rmin}-{rmax}]")
                elif "ratio" in node.extra:
                    extra_parts.append(f"ratio={node.extra['ratio']:.2f}")
                if "source_id" in node.extra:
                    extra_parts.append(f"src={node.extra['source_id']}")
            elif node.node_type == "aggregation":
                # Show ratios with unlocked indicator
                ratios = node.extra.get("ratios", [])
                ratio_ranges = node.extra.get("ratio_ranges", [])
                if ratios:
                    ratio_strs = []
                    for i, r in enumerate(ratios):
                        rng = ratio_ranges[i] if i < len(ratio_ranges) else None
                        if rng:
                            ratio_strs.append(f"[{rng.get('min', '?')}-{rng.get('max', '?')}]")
                        else:
                            ratio_strs.append(f"{r:.2f}")
                    extra_parts.append(f"ratios=[{','.join(ratio_strs)}]")
                # Show bias if present
                bias = node.extra.get("fluo_bias")
                if bias is not None:
                    if isinstance(bias, dict):
                        if "min" in bias or "max" in bias:
                            extra_parts.append(f"bias=[{bias.get('min', '?')}-{bias.get('max', '?')}]")
                        elif "value" in bias:
                            extra_parts.append(f"bias={bias['value']:.2f}")
                    else:
                        extra_parts.append(f"bias={bias}")
            elif node.node_type == "sequestron_ERN":
                if "layer_id" in node.extra:
                    extra_parts.append(f"layer={node.extra['layer_id']}")
            elif node.node_type == "numeric":
                if "role" in node.extra:
                    extra_parts.append(f"role={node.extra['role']}")

            extra_str = ", ".join(extra_parts)[:45]

            lines.append(
                f" {node.node_id:>3} │ {node.node_type:<15} │ {name:<25} │ {layer_str:^5} │ {cotx:<8} │ {extra_str}"
            )

        lines.append("─" * 100)
        return "\n".join(lines)

    def format_edge_table(
        self,
        embedding: Optional[str] = None,
        unlocked_only: bool = False,
    ) -> str:
        """Format the edge table with optional filtering."""
        if self.graph is None:
            return ""

        edges = list(self.graph.edges.values())

        # filter by embedding if specified
        if embedding:
            edges = [e for e in edges if embedding in e.content_embedding_names]

        # filter unlocked only
        if unlocked_only:
            edges = [
                e for e in edges if any(_is_unlocked(v) for v in e.content_embedding_names.values())
            ]

        if not edges:
            return "\nNo edges match the filter criteria."

        lines = [
            f"\nEDGES ({len(edges)} total)",
            "─" * 110,
            f" {'From → To':<20} │ {'Embeddings':<50} │ tu_id",
            "─" * 110,
        ]

        for edge in sorted(edges, key=lambda e: (e.source_id, e.target_id)):
            src_node = self.graph.nodes.get(edge.source_id)
            tgt_node = self.graph.nodes.get(edge.target_id)
            src_short = _get_type_short(src_node.node_type) if src_node else "?"
            tgt_short = _get_type_short(tgt_node.node_type) if tgt_node else "?"
            from_to = f"{src_short}:{edge.source_id} → {tgt_short}:{edge.target_id}"

            # Format all embeddings
            emb_parts = []
            for emb_name, emb_value in edge.content_embedding_names.items():
                emb_parts.append(f"{emb_name}={_format_embedding_value(emb_value)}")
            emb_str = ", ".join(emb_parts) if emb_parts else "-"

            # Get tu_id
            tu_id = edge.extra.get("tu_id", "-")
            if isinstance(tu_id, list):
                tu_id = tu_id[0] if tu_id else "-"

            lines.append(f" {from_to:<20} │ {emb_str:<50} │ {tu_id}")

        lines.append("─" * 110)
        return "\n".join(lines)

    def format_full(
        self,
        show_nodes: bool = True,
        show_edges: bool = True,
    ) -> str:
        """Format the complete graph output."""
        parts = [self.format_header()]

        if show_nodes:
            parts.append(self.format_node_table())

        if show_edges:
            parts.append(self.format_edge_table())

        return "\n".join(parts)


def print_graph(
    network: "Network",
    show_nodes: bool = True,
    show_edges: bool = True,
    output: Optional[Path] = None,
    return_string: bool = False,
) -> Optional[str]:
    """Print node and edge tables for a network's compute graph.

    Args:
        network: Network to visualize
        show_nodes: Show node table
        show_edges: Show edge table
        output: Write to file instead of stdout
        return_string: Return string instead of printing

    Returns:
        String if return_string=True, else None
    """
    printer = GraphPrinter(network)
    result = printer.format_full(show_nodes=show_nodes, show_edges=show_edges)

    if return_string:
        return result

    if output:
        output.write_text(result)
        print(f"Graph written to {output}")
    else:
        print(result)

    return None


def print_paths(
    network: "Network",
    to_node: Optional[int] = None,
    from_node: Optional[int] = None,
    max_length: int = 10,
    show_edge_details: bool = True,
    return_string: bool = False,
) -> Optional[str]:
    """Print all paths to or from a specific node.

    Args:
        network: Network to analyze
        to_node: Find all paths TO this node
        from_node: Find all paths FROM this node
        max_length: Maximum path length to search
        show_edge_details: Show edge info on paths
        return_string: Return string instead of printing
    """
    graph = network.compute_graph
    if graph is None:
        msg = "Network has no compute graph"
        return msg if return_string else print(msg)

    if to_node is None and from_node is None:
        msg = "Must specify either to_node or from_node"
        return msg if return_string else print(msg)

    lines = []

    if to_node is not None:
        target = graph.nodes.get(to_node)
        if target is None:
            msg = f"Node {to_node} not found"
            return msg if return_string else print(msg)

        lines.append(f"PATHS TO NODE [{target.node_type}:{to_node}]")
        lines.append("═" * 40)
        lines.append("")

        # find all paths using BFS with path tracking
        paths = []
        queue = [([to_node], set([to_node]))]

        while queue:
            path, visited = queue.pop(0)
            if len(path) > max_length:
                continue

            current = path[-1]
            incoming = graph.get_incoming_edges(current)

            if not incoming:
                # reached a root, save path (reversed)
                paths.append(list(reversed(path)))
            else:
                for edge in incoming:
                    if edge.source_id not in visited:
                        new_path = path + [edge.source_id]
                        new_visited = visited | {edge.source_id}
                        queue.append((new_path, new_visited))

        if not paths:
            lines.append("No paths found.")
        else:
            for i, path in enumerate(paths, 1):
                via = ""
                if any(
                    graph.nodes.get(nid).node_type == "sequestron_ERN"
                    for nid in path
                    if graph.nodes.get(nid)
                ):
                    via = ", via ERN"

                lines.append(f"Path {i} (length {len(path)}{via}):")

                for j, node_id in enumerate(path):
                    node = graph.nodes[node_id]
                    short = _get_type_short(node.node_type)
                    tu = _get_tu_name(node, graph)
                    extra = ""
                    if node.node_type == "sequestron_ERN" and node.extra.get("ern_type"):
                        extra = f" {node.extra['ern_type']}"

                    lines.append(f"  [{short}:{node_id}]{extra} {tu}")

                    if j < len(path) - 1 and show_edge_details:
                        # find edge to next node
                        next_id = path[j + 1]
                        edge = None
                        for e in graph.get_outgoing_edges(node_id):
                            if e.target_id == next_id:
                                edge = e
                                break

                        if edge:
                            edge_info = f"      │ edge ({edge.source_id},{edge.target_id},{edge.from_output_slot},{edge.to_input_slot})"
                            if edge.content_embedding_names:
                                emb_parts = []
                                for k, v in edge.content_embedding_names.items():
                                    emb_parts.append(f"{k}={_format_embedding_value(v)}")
                                edge_info += f": {', '.join(emb_parts)}"
                            lines.append(edge_info)
                            lines.append("      ▼")

                lines.append("")

            lines.append(f"{len(paths)} path(s) found.")

    elif from_node is not None:
        source = graph.nodes.get(from_node)
        if source is None:
            msg = f"Node {from_node} not found"
            return msg if return_string else print(msg)

        lines.append(f"PATHS FROM NODE [{source.node_type}:{from_node}]")
        lines.append("═" * 40)
        lines.append("")

        # find all paths using BFS
        paths = []
        queue = [([from_node], set([from_node]))]

        while queue:
            path, visited = queue.pop(0)
            if len(path) > max_length:
                continue

            current = path[-1]
            outgoing = graph.get_outgoing_edges(current)

            if not outgoing:
                # reached a leaf, save path
                paths.append(path)
            else:
                for edge in outgoing:
                    if edge.target_id not in visited:
                        new_path = path + [edge.target_id]
                        new_visited = visited | {edge.target_id}
                        queue.append((new_path, new_visited))

        if not paths:
            lines.append("No paths found.")
        else:
            for i, path in enumerate(paths, 1):
                lines.append(f"Path {i} (length {len(path)}):")

                for j, node_id in enumerate(path):
                    node = graph.nodes[node_id]
                    short = _get_type_short(node.node_type)
                    tu = _get_tu_name(node, graph)
                    lines.append(f"  [{short}:{node_id}] {tu}")

                    if j < len(path) - 1:
                        lines.append("      ▼")

                lines.append("")

            lines.append(f"{len(paths)} path(s) found.")

    result = "\n".join(lines)

    if return_string:
        return result
    print(result)
    return None


def print_edges(
    network: "Network",
    embedding: Optional[str] = None,
    unlocked_only: bool = False,
    tu_filter: Optional[list[int]] = None,
    return_string: bool = False,
) -> Optional[str]:
    """Print edge information with optional filtering.

    Args:
        network: Network to analyze
        embedding: Only edges with this embedding name
        unlocked_only: Only show edges with unlocked (multi-value) embeddings
        tu_filter: Only these tu_ids
        return_string: Return string instead of printing
    """
    printer = GraphPrinter(network)
    result = printer.format_edge_table(embedding=embedding, unlocked_only=unlocked_only)

    if return_string:
        return result
    print(result)
    return None
