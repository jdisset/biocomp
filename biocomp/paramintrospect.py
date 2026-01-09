"""Parameter introspection for BNN nodes - single source of truth for param display."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable
import numpy as np

if TYPE_CHECKING:
    from biocomp.parameters import ParameterTree
    from biocomp.compute import ComputeStack, StackNode

TU_THRESHOLD = 0.5


class ParamKind(Enum):
    RATIO = "ratio"
    RATE = "rate"
    AFFINITY = "affinity"
    BIAS = "bias"
    OTHER = "other"


@dataclass
class ParamValue:
    name: str
    kind: ParamKind
    value: float | list[float]
    bounds: tuple[float, float] | None = None
    quantized_to: str | None = None


@dataclass
class InputSlot:
    slot_idx: int
    tu_id: str | None
    is_masked: bool
    source_node: str | None = None


@dataclass
class TUParamGroup:
    tu_id: str
    is_enabled: bool
    prob: float
    cotx_group: str = ""
    no_masking: bool = False
    params: list[ParamValue] = field(default_factory=list)
    inputs: list[InputSlot] = field(default_factory=list)


@dataclass
class NodeParamInfo:
    node_type: str
    node_name: str
    network_id: int
    tu_groups: list[TUParamGroup] = field(default_factory=list)
    ungrouped: list[ParamValue] = field(default_factory=list)
    ungrouped_inputs: list[InputSlot] = field(default_factory=list)


IntrospectFn = Callable[
    ["ParameterTree", list["StackNode"], "ComputeStack", int, bool],
    list[NodeParamInfo],
]


def get_tu_prob(params: "ParameterTree", network_id: int, tu_idx: int) -> float:
    """Get TU activation probability from params. Returns 1.0 if TU masking not enabled."""
    from biocomp.tumasking import TU_ALWAYS_ENABLED, get_log_alpha_from_params

    if tu_idx == TU_ALWAYS_ENABLED or tu_idx < 0:
        return 1.0

    try:
        log_alpha = get_log_alpha_from_params(params, network_id)
        if tu_idx >= len(log_alpha):
            return 1.0
        return float(1 / (1 + np.exp(-float(log_alpha[tu_idx]))))
    except (KeyError, ValueError):
        return 1.0


def is_tu_enabled(prob: float, threshold: float = TU_THRESHOLD) -> bool:
    return prob >= threshold


def get_input_mask_status(
    params: "ParameterTree",
    stack: "ComputeStack",
    node: "StackNode",
    namespace: str,
    node_idx: int,
) -> list[InputSlot]:
    """Get mask status for all inputs of a node."""
    from biocomp.tumasking import TU_ALWAYS_ENABLED

    result = []
    edges = node.get_incoming_edges(stack)
    edges_sorted = sorted(edges, key=lambda e: e.to_input_slot)

    input_tu_path = f"{namespace}/input_tu_indices"
    tu_indices = None
    if input_tu_path in params:
        arr = params[input_tu_path]
        tu_indices = np.asarray(arr.view() if hasattr(arr, "view") else arr)
        if tu_indices.ndim >= 1 and node_idx < tu_indices.shape[0]:
            tu_indices = tu_indices[node_idx]

    for slot_idx, edge in enumerate(edges_sorted):
        source = stack.get_node(node.network_id, edge.source_id)
        source_name = source.extra.get("name", f"node_{edge.source_id}") if source else None

        tu_ids_on_edge = edge.extra.get("tu_id", []) if edge.extra else []
        tu_id = tu_ids_on_edge[0] if tu_ids_on_edge else None

        is_masked = False
        if tu_indices is not None and stack.tu_id_to_idx:
            if tu_indices.ndim == 1:
                idx_val = (
                    int(tu_indices[slot_idx]) if slot_idx < len(tu_indices) else TU_ALWAYS_ENABLED
                )
            elif tu_indices.ndim == 2 and slot_idx < tu_indices.shape[0]:
                slot_tu_indices = tu_indices[slot_idx]
                for tidx in slot_tu_indices:
                    tidx = int(tidx)
                    if tidx >= 0:
                        prob = get_tu_prob(params, node.network_id, tidx)
                        if not is_tu_enabled(prob):
                            is_masked = True
                            break
                idx_val = TU_ALWAYS_ENABLED
            else:
                idx_val = TU_ALWAYS_ENABLED

            if tu_indices.ndim == 1 and idx_val >= 0:
                prob = get_tu_prob(params, node.network_id, idx_val)
                is_masked = not is_tu_enabled(prob)

        result.append(
            InputSlot(
                slot_idx=slot_idx,
                tu_id=tu_id,
                is_masked=is_masked,
                source_node=source_name,
            )
        )

    return result


def _build_tu_id_mapping(
    stack: "ComputeStack", network_id: int
) -> tuple[dict[str, str], dict[str, str]]:
    """Build mappings from tu_id to display name and cotx group.

    Maps both full format (name_cotx) and display name format to support
    different introspection sources (transform uses full, aggregation uses display).
    """
    name_mapping: dict[str, str] = {}
    cotx_mapping: dict[str, str] = {}
    graph = stack.networks[network_id].compute_graph
    if graph is None or graph.nodes is None:
        return name_mapping, cotx_mapping
    for node in graph.nodes.values():
        if node.node_type == "source":
            tu_name = node.extra.get("name", "")
            cotx_group = node.extra.get("cotx_group", "")
            if tu_name:
                if cotx_group:
                    full_tu_id = f"{tu_name}_{cotx_group}"
                    name_mapping[full_tu_id] = tu_name
                    cotx_mapping[full_tu_id] = cotx_group
                cotx_mapping[tu_name] = cotx_group
    return name_mapping, cotx_mapping


def _normalize_tu_ids(
    infos: list[NodeParamInfo],
    name_mapping: dict[str, str],
    cotx_mapping: dict[str, str],
    no_masking_tu_ids: set[str],
) -> None:
    """Normalize tu_ids, set cotx_group and no_masking flags in place."""
    for info in infos:
        for tg in info.tu_groups:
            original_id = tg.tu_id
            tg.cotx_group = cotx_mapping.get(original_id, "")
            tg.no_masking = original_id in no_masking_tu_ids
            tg.tu_id = name_mapping.get(original_id, original_id)
            for inp in tg.inputs:
                if inp.tu_id:
                    inp.tu_id = name_mapping.get(inp.tu_id, inp.tu_id)


def introspect_stack(
    stack: "ComputeStack",
    params: "ParameterTree",
    network_id: int,
    local_only: bool = True,
) -> list[NodeParamInfo]:
    """Collect parameter info from all layers for a given network."""
    assert stack.is_built, "Stack must be built before introspection"
    assert 0 <= network_id < len(stack.networks), (
        f"network_id {network_id} out of range [0, {len(stack.networks)})"
    )

    result: list[NodeParamInfo] = []
    if stack.layers:
        for layer in stack.layers:
            if layer.f_introspect is None:
                continue
            layer_infos = layer.f_introspect(params, layer.nodes, stack, network_id, local_only)
            result.extend(layer_infos)

    name_mapping, cotx_mapping = _build_tu_id_mapping(stack, network_id)
    no_masking_tu_ids = stack.no_masking_tu_ids or set()
    _normalize_tu_ids(result, name_mapping, cotx_mapping, no_masking_tu_ids)

    return result


def aggregate_by_tu(infos: list[NodeParamInfo]) -> dict[str, list[tuple[str, TUParamGroup]]]:
    """Aggregate node param infos by TU ID for consolidated display."""
    tu_data: dict[str, list[tuple[str, TUParamGroup]]] = {}

    for info in infos:
        for tg in info.tu_groups:
            if tg.tu_id not in tu_data:
                tu_data[tg.tu_id] = []
            tu_data[tg.tu_id].append((info.node_type, tg))

    return tu_data


def _fmt_value(v: float | list[float], precision: int = 3) -> str:
    if isinstance(v, list):
        if len(v) <= 6:
            return " | ".join(f"{x:.{precision}f}" for x in v)
        return " | ".join(f"{x:.{precision}f}" for x in v[:5]) + f" ... ({len(v)} total)"
    return f"{v:.{precision}f}"


def _fmt_inputs(inputs: list[InputSlot]) -> str:
    if not inputs:
        return ""
    parts = []
    for inp in inputs:
        status = "MASKED" if inp.is_masked else "ON"
        src = inp.source_node or f"slot_{inp.slot_idx}"
        parts.append(f"{src}: {status}")
    return "[" + ", ".join(parts) + "]"


def format_network_params(
    stack: "ComputeStack",
    params: "ParameterTree",
    network_id: int,
    local_only: bool = True,
) -> str:
    """Human-readable parameter summary grouped by TU."""
    infos = introspect_stack(stack, params, network_id, local_only)
    if not infos:
        return f"Network {network_id}: no introspectable parameters"

    net_name = stack.networks[network_id].name if stack.networks else f"network_{network_id}"
    lines = [f"Network: {net_name}", "=" * 50]

    tu_data = aggregate_by_tu(infos)
    ungrouped_by_type: dict[str, list[tuple[NodeParamInfo, list[ParamValue]]]] = {}

    for info in infos:
        if info.ungrouped:
            if info.node_type not in ungrouped_by_type:
                ungrouped_by_type[info.node_type] = []
            ungrouped_by_type[info.node_type].append((info, info.ungrouped))

    for tu_id in sorted(tu_data.keys()):
        entries = tu_data[tu_id]
        first_tg = entries[0][1]
        status = "ON" if first_tg.is_enabled else "OFF"
        lines.append(f"\nTU: {tu_id} [{status}] p={first_tg.prob:.2f}")
        lines.append("-" * 40)

        for node_type, tg in entries:
            for pv in tg.params:
                val_str = _fmt_value(pv.value)
                line = f"  {node_type}/{pv.name}: {val_str}"
                if pv.quantized_to:
                    line += f" -> {pv.quantized_to}"
                if pv.bounds:
                    line += f" [{pv.bounds[0]:.2f}-{pv.bounds[1]:.2f}]"
                if not tg.is_enabled:
                    line += " MASKED"
                lines.append(line)

            if tg.inputs:
                lines.append(f"    inputs: {_fmt_inputs(tg.inputs)}")

    for node_type, entries in ungrouped_by_type.items():
        lines.append(f"\n{node_type.upper()}:")
        for info, params_list in entries:
            for pv in params_list:
                val_str = _fmt_value(pv.value)
                line = f"  {info.node_name}: {pv.name}={val_str}"
                if pv.bounds:
                    line += f" [{pv.bounds[0]:.2f}-{pv.bounds[1]:.2f}]"
                lines.append(line)

            if info.ungrouped_inputs:
                lines.append(f"    inputs: {_fmt_inputs(info.ungrouped_inputs)}")

    return "\n".join(lines)


def get_network_param_dict(
    stack: "ComputeStack",
    params: "ParameterTree",
    network_id: int,
    local_only: bool = True,
) -> dict:
    """Dictionary representation of parameters for logging/serialization."""
    infos = introspect_stack(stack, params, network_id, local_only)

    def param_to_dict(pv: ParamValue) -> dict:
        d = {"name": pv.name, "kind": pv.kind.value, "value": pv.value}
        if pv.bounds:
            d["bounds"] = pv.bounds
        if pv.quantized_to:
            d["quantized_to"] = pv.quantized_to
        return d

    def input_to_dict(inp: InputSlot) -> dict:
        return {
            "slot": inp.slot_idx,
            "tu_id": inp.tu_id,
            "masked": inp.is_masked,
            "source": inp.source_node,
        }

    def tg_to_dict(tg: TUParamGroup) -> dict:
        return {
            "tu_id": tg.tu_id,
            "cotx_group": tg.cotx_group,
            "no_masking": tg.no_masking,
            "enabled": tg.is_enabled,
            "prob": tg.prob,
            "params": [param_to_dict(p) for p in tg.params],
            "inputs": [input_to_dict(i) for i in tg.inputs],
        }

    return {
        "network_id": network_id,
        "nodes": [
            {
                "node_type": info.node_type,
                "node_name": info.node_name,
                "tu_groups": [tg_to_dict(tg) for tg in info.tu_groups],
                "ungrouped": [param_to_dict(p) for p in info.ungrouped],
                "ungrouped_inputs": [input_to_dict(i) for i in info.ungrouped_inputs],
            }
            for info in infos
        ],
    }


def _fmt_inputs_rich(inputs: list[InputSlot]) -> str:
    if not inputs:
        return ""
    parts = []
    for inp in inputs:
        src = inp.source_node or f"slot_{inp.slot_idx}"
        if inp.is_masked:
            parts.append(f"[red]{src}: MASKED[/red]")
        else:
            parts.append(f"[green]{src}: ON[/green]")
    return "[" + ", ".join(parts) + "]"


def _fmt_input_compact(inputs: list[InputSlot]) -> str:
    if not inputs:
        return ""
    parts = []
    for inp in inputs:
        src = inp.source_node or f"slot_{inp.slot_idx}"
        sym = "✗" if inp.is_masked else "✓"
        color = "red" if inp.is_masked else "green"
        parts.append(f"[{color}]{src} {sym}[/{color}]")
    return "← " + ", ".join(parts)


def _render_tu_table(console, tu_data: dict) -> None:
    """Render unified TU table with all parameters, grouped by CoTransfection."""
    from rich.table import Table

    cotx_groups: dict[str, list[str]] = {}
    for tu_id, entries in tu_data.items():
        cotx = entries[0][1].cotx_group if entries else ""
        if cotx not in cotx_groups:
            cotx_groups[cotx] = []
        cotx_groups[cotx].append(tu_id)

    total_enabled = sum(
        1 for entries in tu_data.values() if any(tg.is_enabled for _, tg in entries)
    )
    table = Table(title=f"Transcription Units ({total_enabled}/{len(tu_data)} ON)")
    table.add_column("TU")
    table.add_column("", justify="center")
    table.add_column("p", justify="right")
    table.add_column("Ratio", justify="right")
    table.add_column("tc_rate", justify="left")
    table.add_column("tl_rate", justify="left")

    for cotx in sorted(cotx_groups.keys()):
        tu_ids = sorted(cotx_groups[cotx])
        cotx_enabled = sum(1 for tid in tu_ids if any(tg.is_enabled for _, tg in tu_data[tid]))
        table.add_row(
            f"[bold cyan]{cotx}[/bold cyan]" if cotx else "[dim]ungrouped[/dim]",
            f"[cyan]{cotx_enabled}/{len(tu_ids)}[/cyan]",
            "",
            "",
            "",
            "",
            style="on grey15",
        )

        for tu_id in tu_ids:
            entries = tu_data[tu_id]

            ratio_str = tc_str = tl_str = "-"
            ratio_range = ""
            is_enabled = False
            prob = 0.0
            no_masking = False

            for _, tg in entries:
                if tg.is_enabled:
                    is_enabled = True
                prob = max(prob, tg.prob)
                no_masking = no_masking or tg.no_masking

                for pv in tg.params:
                    val_fmt = f"{pv.value:.3f}"
                    if pv.kind == ParamKind.RATIO:
                        ratio_str = val_fmt
                        if pv.bounds:
                            ratio_range = f" [{pv.bounds[0]:.1f}-{pv.bounds[1]:.1f}]"
                    elif pv.kind == ParamKind.RATE:
                        if pv.quantized_to:
                            val_fmt += f" ({pv.quantized_to})"
                        if "tc" in pv.name:
                            tc_str = val_fmt
                        elif "tl" in pv.name:
                            tl_str = val_fmt

            lock = "🔒" if no_masking else ""
            status = "✓" if is_enabled else "✗"
            status_styled = f"[green]{status}[/green]" if is_enabled else f"[red]{status}[/red]"
            ratio_display = ratio_str + ratio_range if ratio_str != "-" else "-"

            if is_enabled:
                table.add_row(
                    f"  {tu_id} {lock}".rstrip(),
                    status_styled,
                    f"{prob:.2f}",
                    ratio_display,
                    tc_str,
                    tl_str,
                )
            else:
                d = "grey50"
                table.add_row(
                    f"[{d}]  {tu_id} {lock}[/{d}]".rstrip(),
                    f"[{d}]{status}[/{d}]",
                    f"[{d}]{prob:.2f}[/{d}]",
                    f"[{d}]{ratio_display}[/{d}]",
                    f"[{d}]{tc_str}[/{d}]",
                    f"[{d}]{tl_str}[/{d}]",
                )

    console.print(table)


def _render_bio_tus(console, tu_data: dict) -> None:
    if tu_data:
        _render_tu_table(console, tu_data)


def _render_ungrouped_rich(console, infos: list[NodeParamInfo]) -> None:
    ungrouped_by_type: dict[str, list[tuple[NodeParamInfo, list[ParamValue]]]] = {}
    for info in infos:
        if info.ungrouped:
            if info.node_type not in ungrouped_by_type:
                ungrouped_by_type[info.node_type] = []
            ungrouped_by_type[info.node_type].append((info, info.ungrouped))

    for node_type, entries in ungrouped_by_type.items():
        console.print(f"\n[bold]{node_type.upper()}:[/bold]")
        for info, params_list in entries:
            param_strs = []
            for pv in params_list:
                val_str = _fmt_value(pv.value)
                p_str = f"{pv.name}={val_str}"
                if pv.bounds:
                    p_str += f" [{pv.bounds[0]:.2f}-{pv.bounds[1]:.2f}]"
                param_strs.append(p_str)

            line = f"  {info.node_name}: {', '.join(param_strs)}"
            if info.ungrouped_inputs:
                line += f" {_fmt_input_compact(info.ungrouped_inputs)}"
            console.print(line)


def format_network_params_rich(
    stack: "ComputeStack",
    params: "ParameterTree",
    network_id: int,
    console=None,
) -> None:
    from rich.console import Console
    from rich.panel import Panel

    if console is None:
        console = Console()

    infos = introspect_stack(stack, params, network_id)
    if not infos:
        console.print(f"[dim]Network {network_id}: no introspectable parameters[/dim]")
        return

    net_name = stack.networks[network_id].name if stack.networks else f"network_{network_id}"
    console.print(Panel(f"[bold]{net_name}[/bold]", expand=False))

    tu_data = aggregate_by_tu(infos)

    if tu_data:
        _render_bio_tus(console, tu_data)

    _render_ungrouped_rich(console, infos)


def format_committed_network_tus(network, console=None) -> None:
    """Display TUs directly from a committed network's compute_graph.

    This inspects the network structure itself (not params), showing what TUs
    actually exist after commit. All displayed TUs are enabled by definition
    since disabled ones should have been removed during commit.

    Uses the same rendering as format_network_params_rich for consistency.
    """
    from rich.console import Console
    from rich.panel import Panel

    if console is None:
        console = Console()

    if network.compute_graph is None:
        console.print("[dim]Network has no compute_graph[/dim]")
        return

    net_name = network.name or "committed_network"
    console.print(Panel(f"[bold]{net_name}[/bold] (committed)", expand=False))

    tu_data: dict[str, list[tuple[str, TUParamGroup]]] = {}
    for node in network.compute_graph.nodes.values():
        if node.node_type != "source":
            continue

        tu_name = node.extra.get("name", "")
        cotx = node.extra.get("cotx_group", "")
        if not tu_name:
            continue

        tg = TUParamGroup(
            tu_id=tu_name,
            is_enabled=True,
            prob=1.0,
            cotx_group=cotx,
            no_masking=False,
        )
        if tu_name not in tu_data:
            tu_data[tu_name] = []
        tu_data[tu_name].append(("source", tg))

    if not tu_data:
        console.print("[dim]No TUs found in committed network[/dim]")
        return

    _render_tu_table(console, tu_data)
