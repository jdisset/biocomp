from typing import (
    Any,
    Optional,
    Tuple,
    Sequence,
)
from pydantic import BaseModel, ConfigDict
import pandas as pd
import numpy as np

from biocomp.library import LibraryContext, PartsLibrary
from biocomp.recipe import Recipe, TranscriptionUnit, CoTxList, Slot, CoTransfection
from biocomp.logging_config import get_logger
from biocomp.graphengine import (
    GraphState,
    GraphNode,
    GraphEdge,
    Part,
    InverseSpec,
    apply_rule_sequence,
)
from biocomp.graphrules import GraphRewritingRule
import biocomp.biorules as br

# TODO:
# - [ ] the network stat tools
# - [ ] layer annotations for ERN

logger = get_logger(__name__)

# Canonical biological ordering of part categories for slot reconstruction
CATEGORY_ORDER = {
    "insulator": 0,
    "promoter": 10,
    "uORF_group": 15,
    "ERN_recog_site_5p": 17,
    "fluo_marker": 20,
    "ERN": 20,
    "ERN_recog_site_3p": 25,
    "terminator": 30,
}

# Default/implicit values that represent "empty" embeddings
IMPLICIT_EMPTY = {"tl_rate": "00_empty_tc"}


class Network(BaseModel):
    """Pure data container for network definitions"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Optional[str] = None
    metadata: dict[str, Any] = {}
    compute_graph: Optional[GraphState] = None

    def get_output_compute_node(self) -> GraphNode:
        assert self.compute_graph is not None
        output_nodes = [n for n in self.compute_graph.nodes.values() if n.node_type == "output"]
        assert len(output_nodes) == 1, f"Invalid number of output nodes: {len(output_nodes)}"
        return output_nodes[0]

    @property
    def nb_outputs(self) -> int:
        return len(self.get_output_proteins())

    @property
    def nb_inputs(self):
        return len(self.get_inverted_input_proteins())

    def get_nb_outputs(self) -> int:
        """Compatibility method for old code expecting get_nb_outputs()"""
        return self.nb_outputs

    def get_nb_inputs(self) -> int:
        """Compatibility method for old code expecting get_nb_inputs()"""
        return self.nb_inputs

    def get_input_from_output(self, output_arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Given an array of output values, returns the columns that are inputs of the inverted network,
        properly ordered by input number"""
        assert self.compute_graph is not None
        if output_arr is None:
            return None
        mapping = self.get_inverted_input_positions()
        return output_arr[:, [mapping[i] for i in range(len(mapping))]]

    def get_inverted_input_proteins(self, include_biases: bool = False) -> list[str]:
        assert self.compute_graph is not None
        mapping = self.get_inverted_input_positions(include_biases)
        output_proteins = self.get_output_proteins()
        assert len(mapping) <= len(output_proteins), f"Invalid mapping: {mapping}"
        return [output_proteins[mapping[i]] for i in range(len(mapping))]

    def get_inverted_input_positions(self, include_biases: bool = False) -> dict[int, int]:
        """Returns a mapping from input position to output position

        Args:
            include_biases: If True, also includes bias nodes in the mapping
        """
        assert self.compute_graph is not None
        mapping = {}
        # Input nodes have input_position/input_from_output (from inversion)
        mask_types = ["input"]

        # If including biases, add bias nodes to the types
        if include_biases:
            mask_types.append("bias")

        inputs = [n for n in self.compute_graph.nodes.values() if n.node_type in mask_types]
        for node in inputs:
            # Bias nodes don't have input_position, they map directly via input_from_output
            if node.node_type == "bias":
                # Bias nodes only have input_from_output in inverted networks
                if "input_from_output" not in node.extra:
                    # Skip bias nodes without input_from_output (non-inverted networks)
                    continue
                # For bias nodes, use a special position (after regular inputs)
                # Count existing inputs to determine position
                bias_position = len(
                    [n for n in inputs if n.node_type == "input" and n.node_id < node.node_id]
                )
                # Add number of regular inputs to get the bias position
                regular_input_count = len([n for n in inputs if n.node_type == "input"])
                bias_input_pos = regular_input_count + len(
                    [
                        n
                        for n in inputs
                        if n.node_type == "bias"
                        and n.node_id < node.node_id
                        and "input_from_output" in n.extra
                    ]
                )
                mapping[bias_input_pos] = node.extra["input_from_output"]
            else:
                assert "input_position" in node.extra, f"input_position not in {node.extra}"
                assert "input_from_output" in node.extra, f"input_from_output not in {node.extra}"
                assert node.extra["input_position"] not in mapping
                mapping[node.extra["input_position"]] = node.extra["input_from_output"]

        assert set(mapping.keys()) == set(range(len(mapping.keys()))), f"Invalid mapping: {mapping}"
        assert len(mapping.keys()) == len(set(mapping.values())), f"Invalid mapping: {mapping}"
        return mapping

    def get_bias_proteins(self) -> list[str]:
        """Returns the names of proteins that are bias inputs (fluo_bias nodes)"""
        assert self.compute_graph is not None
        bias_nodes = self.compute_graph.get_nodes_by_type("bias")
        bias_proteins = []
        for bias_node in bias_nodes:
            fluo_bias = bias_node.extra.get("fluo_bias")
            if fluo_bias and isinstance(fluo_bias, dict):
                protein = fluo_bias.get("protein")
                if protein:
                    bias_proteins.append(protein)
        return bias_proteins

    def get_output_proteins(self, only_dependent_outputs: bool = False) -> list[str]:
        """Returns the names of all proteins that are outputs of the network"""
        assert self.compute_graph is not None
        if only_dependent_outputs:
            return self.get_dependent_output_proteins()

        onode = self.get_output_compute_node()

        incoming_edges = self.compute_graph.get_incoming_edges(onode.node_id)
        output_proteins = []
        for edge in sorted(incoming_edges, key=lambda e: e.to_input_slot):
            if edge.content and edge.content_type == "PRT":
                output_proteins.append(edge.content[0].name)

        return output_proteins

    def get_dependent_output_proteins(self) -> list[str]:
        """Returns the names of the proteins that are outputs of the network and are not inverted inputs"""
        assert self.compute_graph is not None
        all_outputs = self.get_output_proteins()
        input_proteins = self.get_inverted_input_proteins(include_biases=True)
        return [p for p in all_outputs if p not in input_proteins]

    def get_dependent_output_mask(self) -> np.ndarray:
        """Returns a boolean mask of the output proteins that are dependent on the inputs"""
        assert self.compute_graph is not None
        n_outputs = self.nb_outputs
        input_positions = self.get_inverted_input_positions(include_biases=True).values()
        dependent_outputs = [i for i in range(n_outputs) if i not in input_positions]
        mask = np.zeros(n_outputs, dtype=bool)
        mask[dependent_outputs] = True
        return mask

    def set_input_as_bias(self, input_protein_name: Sequence[str]) -> None:
        """Sets this input protein as a bias node (instead of an input one)"""
        original_mapping = self.get_inverted_input_positions()
        output_proteins = self.get_output_proteins()
        assert input_protein_name in output_proteins, (
            f"Invalid input protein name: {input_protein_name}"
        )
        output_position = output_proteins.index(input_protein_name)
        assert output_position in original_mapping.values()

        inputs = self.compute_graph.get_nodes_by_type("input")
        found = False
        for node in inputs:
            assert "input_position" in node.extra, f"input_position not in {node.extra}"
            assert "input_from_output" in node.extra, f"input_from_output not in {node.extra}"
            if node.extra["input_from_output"] == output_position:
                # Modify node type to bias
                self.compute_graph.nodes[node.node_id] = GraphNode(
                    node_id=node.node_id,
                    node_type="bias",
                    is_inverse_of=node.is_inverse_of,
                    extra=node.extra,
                )
                found = True
                break
        assert found, f"Could not find input protein {input_protein_name} in compute graph"

        self._renumber_input_positions()

        new_mapping = self.get_inverted_input_positions()
        assert len(new_mapping) == len(original_mapping) - 1, f"Invalid mapping: {new_mapping}"
        assert output_position not in new_mapping.values()
        assert len(self.get_inverted_input_proteins()) == len(new_mapping)

    def set_bias_as_input(self, bias_protein_name: Sequence[str]) -> None:
        """Sets this bias protein back as an input node (instead of a bias one)"""
        assert self.compute_graph is not None
        original_mapping = self.get_inverted_input_positions()
        output_proteins = self.get_output_proteins()
        assert bias_protein_name in output_proteins, (
            f"Invalid bias protein name: {bias_protein_name}"
        )
        output_position = output_proteins.index(bias_protein_name)

        assert output_position not in original_mapping.values(), (
            f"Protein {bias_protein_name} is already an input, not a bias"
        )

        biases = self.compute_graph.get_nodes_by_type("bias")
        found = False
        for node in biases:
            assert "input_from_output" in node.extra, f"input_from_output not in {node.extra}"
            if node.extra["input_from_output"] == output_position:
                new_extra = dict(node.extra)
                if "input_position" not in new_extra:
                    next_input_pos = len(original_mapping) if original_mapping else 0
                    new_extra["input_position"] = next_input_pos

                self.compute_graph.nodes[node.node_id] = GraphNode(
                    node_id=node.node_id,
                    node_type="input",
                    is_inverse_of=node.is_inverse_of,
                    extra=new_extra,
                )
                found = True
                break
        assert found, f"Could not find bias protein {bias_protein_name} in compute graph"

        self._renumber_input_positions()

        new_mapping = self.get_inverted_input_positions()
        assert len(new_mapping) == len(original_mapping) + 1, f"Invalid mapping: {new_mapping}"
        assert output_position in new_mapping.values()
        assert len(self.get_inverted_input_proteins()) == len(new_mapping)

    def _renumber_input_positions(self) -> None:
        """Renumbers input positions to be consecutive starting from 0"""
        assert self.compute_graph is not None
        inputs = self.compute_graph.get_nodes_by_type("input")

        input_output_pairs = []
        for node in inputs:
            assert "input_from_output" in node.extra, f"input_from_output not in {node.extra}"
            input_output_pairs.append((node.node_id, node.extra["input_from_output"]))

        input_output_pairs.sort(key=lambda x: x[1])

        for new_pos, (node_id, _) in enumerate(input_output_pairs):
            node = self.compute_graph.nodes[node_id]
            new_extra = dict(node.extra)
            new_extra["input_position"] = new_pos
            self.compute_graph.nodes[node_id] = GraphNode(
                node_id=node.node_id,
                node_type=node.node_type,
                is_inverse_of=node.is_inverse_of,
                extra=new_extra,
            )

    def to_recipe(self) -> Recipe:
        """Converts the network back to a Recipe object"""

        cotx_groups = self._extract_cotx_groups()
        tus_and_ratios_by_cotx = self._build_transcription_units(cotx_groups)
        bias_by_cotx = self._extract_bias_nodes()

        # Sort by cotx_index to preserve original order
        sorted_group_ids = sorted(cotx_groups.keys(), key=lambda g: cotx_groups[g]["cotx_index"])

        content = []
        for group_id in sorted_group_ids:
            tus, reordered_ratios = tus_and_ratios_by_cotx[group_id]
            content.append(
                CoTransfection(
                    name=group_id if group_id != "cotx_1" or len(cotx_groups) > 1 else None,
                    units=tus,
                    ratios=reordered_ratios if len(reordered_ratios) > 1 else None,
                    fluo_bias=bias_by_cotx.get(group_id),
                )
            )

        metadata_dict = {k: v for k, v in self.metadata.items() if k not in ["name", "description"]}

        return Recipe(
            name=self.name or self.metadata.get("name"),
            description=self.metadata.get("description"),
            metadata=metadata_dict if metadata_dict else None,
            content=content,
        )

    def _extract_cotx_groups(self) -> dict[str, dict]:
        from biocomp.recipe import NumRange

        cotx_groups = {}
        assert self.compute_graph is not None

        source_cotx_indices = {}
        for node in self.compute_graph.get_nodes_by_type("source"):
            group_id = node.extra.get("cotx_group")
            cotx_index = node.extra.get("cotx_index", 0)
            if group_id not in source_cotx_indices:
                source_cotx_indices[group_id] = cotx_index

        for node in self.compute_graph.get_nodes_by_type("aggregation"):
            group_id = node.extra["cotx_group"]
            ratios = []
            if "ratio_ranges" in node.extra:
                ratio_ranges = node.extra["ratio_ranges"]
                base_ratios = node.extra["ratios"]
                for base_ratio, ratio_range in zip(base_ratios, ratio_ranges):
                    if ratio_range is not None and isinstance(ratio_range, dict):
                        ratios.append(
                            NumRange(min=ratio_range.get("min"), max=ratio_range.get("max"))
                        )
                    else:
                        ratios.append(base_ratio)
            else:
                ratios = node.extra["ratios"]
            cotx_groups[group_id] = {
                "ratios": ratios,
                "source_ids": node.extra["members"],
                "cotx_index": source_cotx_indices.get(group_id, 0),
            }

        for node in self.compute_graph.get_nodes_by_type("source"):
            group_id = node.extra.get("cotx_group")
            source_id = node.extra.get("source_id")
            cotx_index = node.extra.get("cotx_index", 0)
            if group_id not in cotx_groups:
                cotx_groups[group_id] = {
                    "ratios": [node.extra.get("ratio", 1.0)],
                    "source_ids": [source_id],
                    "cotx_index": cotx_index,
                }

        return cotx_groups

    def _build_transcription_units(
        self, cotx_groups: dict
    ) -> dict[str, tuple[list[TranscriptionUnit], list]]:
        tus_and_ratios_by_cotx = {}
        for group_id, info in cotx_groups.items():
            tus = []
            # Create mapping from source_id to ratio (both are in aggregation order, e.g. alphabetically sorted)
            source_id_to_ratio = dict(zip(info["source_ids"], info["ratios"]))

            # Collect source node with their output slots (one TU per output slot)
            tu_specs = []  # (position, source_id, source_node, output_slot)
            assert self.compute_graph is not None
            for node in self.compute_graph.get_nodes_by_type("source"):
                if node.extra.get("cotx_group") == group_id:
                    source_id = node.extra.get("source_id")
                    # Count output slots from outgoing edges
                    outgoing = self.compute_graph.get_outgoing_edges(node.node_id)
                    output_slots = sorted(set(e.from_output_slot for e in outgoing))

                    for output_slot in output_slots:
                        position = node.extra.get("position_in_source", 0) + output_slot
                        tu_specs.append((position, source_id, node, output_slot))

            # Sort by position to restore original TU order
            tu_specs.sort(key=lambda x: x[0])

            # Build ratios list (one per unique source in TU order)
            seen_sources = set()
            reordered_ratios = []
            for position, source_id, source_node, output_slot in tu_specs:
                if source_id not in seen_sources:
                    reordered_ratios.append(source_id_to_ratio.get(source_id, 1.0))
                    seen_sources.add(source_id)

            for position, source_id, source_node, output_slot in tu_specs:

                param_ref_ids = source_node.extra.get("param_ref_ids", {})
                slots = self._extract_slots_from_source(source_node, param_ref_ids, output_slot)

                if param_ref_ids:
                    for slot in slots:
                        if slot.maps_to_parameter and slot.maps_to_parameter in param_ref_ids:
                            slot.ref_id = param_ref_ids[slot.maps_to_parameter]

                tu = TranscriptionUnit(
                    name=source_node.extra.get("name", ""),
                    slots=slots,
                    source=source_id,
                    position_in_source=position,
                )

                if param_ref_ids:
                    tu.param_ref_ids = dict(param_ref_ids)

                tus.append(tu)
            tus_and_ratios_by_cotx[group_id] = (tus, reordered_ratios)
        return tus_and_ratios_by_cotx

    def _find_source_node(self, source_id: str, cotx_group: str):
        assert self.compute_graph is not None
        for node in self.compute_graph.nodes.values():
            if (
                node.node_type == "source"
                and node.extra.get("source_id") == source_id
                and node.extra.get("cotx_group") == cotx_group
            ):
                return node
        return None

    def _get_dna_edge(self, source_node, output_slot: int = 0):
        """Get DNA edge from source node for specific output slot, or None if not found"""
        assert self.compute_graph is not None
        outgoing = self.compute_graph.get_outgoing_edges(source_node.node_id)
        dna_edges = [e for e in outgoing if e.content_type == "DNA" and e.from_output_slot == output_slot]
        return dna_edges[0] if dna_edges else None

    def _should_include_embedding(
        self, emb_name: str, part_names: tuple, param_ref_ids: dict
    ) -> bool:
        """Check if embedding should be included (has real parts or explicit ref_id)"""
        implicit_empty = IMPLICIT_EMPTY.get(emb_name)
        has_real_parts = any(p != implicit_empty for p in part_names) if part_names else False
        has_ref_id = emb_name in param_ref_ids and param_ref_ids[emb_name] is not None
        return has_real_parts or has_ref_id

    def _extract_slots_from_source(self, source_node, param_ref_ids: dict = None, output_slot: int = 0) -> list[Slot]:
        """Reconstruct slots by sorting all parts by their biological category"""
        param_ref_ids = param_ref_ids or {}
        lib = LibraryContext.get_library()

        dna_edge = self._get_dna_edge(source_node, output_slot)
        if not dna_edge:
            return []

        embeddings = dna_edge.content_embedding_names or {}

        # Collect all parts with (category, name, embedding_name)
        parts = []

        # DNA parts (non-embeddings) - categories already stored in edge
        for part_obj in dna_edge.content:
            parts.append((part_obj.category, part_obj.name, None))

        # Embedding parts - look up categories from library
        for emb_name, part_names in embeddings.items():
            if self._should_include_embedding(emb_name, part_names, param_ref_ids):
                implicit_empty = IMPLICIT_EMPTY.get(emb_name)
                # Clean up implicit empties
                cleaned_parts = [p if p != implicit_empty else None for p in part_names]
                # Determine category from first non-None part
                cat = None
                for pname in cleaned_parts:
                    if pname is not None and pname in lib.pc.index:
                        cat = lib.pc.loc[pname, "category"]
                        break
                # Add as single entry (will become one slot with potentially multiple parts)
                parts.append((cat, cleaned_parts, emb_name))

        # Sort by category order (stable sort preserves relative order within category)
        parts.sort(key=lambda x: CATEGORY_ORDER.get(x[0], 999) if x[0] else 998)

        # Create slots
        slots = []
        for category, name, emb in parts:
            slot = Slot(part=name)
            if emb:
                slot.maps_to_parameter = emb
            slots.append(slot)

        return slots

    def _parse_value_to_numrange_or_float(self, value_raw):
        """Parse a value (string/dict/numeric) into NumRange or float"""
        from biocomp.recipe import NumRange
        import ast

        if not value_raw or value_raw == "":
            return None

        # parse string to dict if needed
        if isinstance(value_raw, str):
            try:
                value_raw = ast.literal_eval(value_raw)
            except:
                try:
                    return float(value_raw)
                except:
                    return None

        # convert dict to NumRange, otherwise return as float
        if isinstance(value_raw, dict):
            return NumRange(min=value_raw.get("min"), max=value_raw.get("max"))
        elif isinstance(value_raw, (int, float)):
            return float(value_raw)
        return None

    def _find_cotx_group_for_bias(self, node) -> str:
        """Find cotx group by traversing edges from bias node"""
        assert self.compute_graph is not None

        # Traverse forward through inv_ nodes to find aggregation/source
        current_id = node.node_id
        visited = set()
        while current_id not in visited:
            visited.add(current_id)
            current_node = self.compute_graph.nodes.get(current_id)
            if not current_node:
                break

            # Check if we reached aggregation or source
            if current_node.node_type in ["aggregation", "source"]:
                return current_node.extra.get("cotx_group", "cotx_1")

            # Check if we reached inv_aggregation (which has cotx_group from original)
            if current_node.node_type == "inv_aggregation":
                # Get the original aggregation node it inverts
                if current_node.is_inverse_of:
                    orig_node = self.compute_graph.nodes.get(current_node.is_inverse_of.node_id)
                    if orig_node:
                        return orig_node.extra.get("cotx_group", "cotx_1")
                return current_node.extra.get("cotx_group", "cotx_1")

            # Follow outgoing edges
            outgoing = list(self.compute_graph.get_outgoing_edges(current_id))
            if not outgoing:
                break
            current_id = outgoing[0].target_id

        return "cotx_1"

    def _extract_bias_from_aggregation(self, node):
        """Extract bias info from connected aggregation node as fallback"""
        import ast

        assert self.compute_graph is not None
        for edge in self.compute_graph.get_outgoing_edges(node.node_id):
            target = self.compute_graph.nodes.get(edge.target_id)
            if target and target.node_type == "aggregation" and "fluo_bias" in target.extra:
                fb = target.extra["fluo_bias"]

                # parse string representation if needed
                if isinstance(fb, str):
                    try:
                        fb = ast.literal_eval(fb)
                    except:
                        continue

                if isinstance(fb, dict):
                    value = self._parse_value_to_numrange_or_float(fb.get("value"))
                    return {
                        "tu_id": fb.get("tu_id", 0),
                        "value": value if value is not None else 100.0,
                        "protein": fb.get("protein"),
                        "units": fb.get("units", "AU"),
                    }
        return None

    def _parse_fluo_bias_data(self, data):
        if not data:
            return None
        if isinstance(data, dict):
            return data
        import ast

        try:
            return ast.literal_eval(data) if isinstance(data, str) else None
        except (ValueError, SyntaxError):
            return None

    def _create_fluo_intensity_from_dict(self, data: dict):
        """Create FluoIntensity from parsed dict data"""
        from biocomp.recipe import FluoIntensity

        tu_id = data.get("tu_id", 0)
        value = self._parse_value_to_numrange_or_float(data.get("value"))
        protein = data.get("protein")
        if protein in ["", "None", None]:
            protein = None
        units = data.get("units", "AU") or "AU"

        return FluoIntensity(
            tu_id=tu_id,
            value=value if value is not None else 100.0,
            protein=protein,
            units=units,
        )

    def _extract_bias_nodes(self) -> dict:
        """Extract bias nodes and reconstruct FluoIntensity objects for each cotx group"""
        from biocomp.recipe import FluoIntensity

        bias_by_cotx = {}
        assert self.compute_graph is not None

        for node in self.compute_graph.nodes.values():
            if node.node_type != "bias" or node.extra.get("role") != "fluo_bias":
                continue

            cotx_group = self._find_cotx_group_for_bias(node)

            # Try to get fluo_bias dict (current format)
            fluo_bias = node.extra.get("fluo_bias")
            if fluo_bias and isinstance(fluo_bias, dict):
                # Parse value from fluo_bias dict
                value = self._parse_value_to_numrange_or_float(fluo_bias.get("value"))
                protein = fluo_bias.get("protein")
                if protein in ["", "None"]:
                    protein = None
                units = fluo_bias.get("units", "AU") or "AU"
                tu_id = fluo_bias.get("tu_id", 0)

                bias_by_cotx[cotx_group] = FluoIntensity(
                    tu_id=tu_id,
                    value=value if value is not None else 100.0,
                    protein=protein,
                    units=units,
                )
                continue

            # Legacy format: try fluo_bias_data dict
            fluo_bias_data = self._parse_fluo_bias_data(node.extra.get("fluo_bias_data"))
            if fluo_bias_data:
                bias_by_cotx[cotx_group] = self._create_fluo_intensity_from_dict(fluo_bias_data)
                continue

            # Fallback: try to get fields directly from node.extra (oldest format)
            value = self._parse_value_to_numrange_or_float(node.extra.get("value"))
            protein = node.extra.get("protein")
            if protein in ["", "None"]:
                protein = None
            units = node.extra.get("units", "AU") or "AU"
            tu_id = node.extra.get("tu_id", 0)

            bias_by_cotx[cotx_group] = FluoIntensity(
                tu_id=tu_id,
                value=value if value is not None else 100.0,
                protein=protein,
                units=units,
            )

        return bias_by_cotx

    def compute_dependency_map(self) -> dict[int, set[int]]:
        """Returns {node id -> set of upstream node ids}"""
        assert self.compute_graph is not None
        return self.compute_graph.compute_dependency_map()

    def topological_order(self, nodes=None, dependency_map=None):
        assert self.compute_graph is not None
        return self.compute_graph.topological_order(nodes, dependency_map)

    def generate_network_info(self):
        return generate_network_info(self)

    @property
    def network_info(self):
        return generate_network_info(self)


def assign_ern_layer_ids(graph: GraphState) -> GraphState:
    ern_nodes = {n.node_id: n for n in graph.nodes.values() if n.node_type == "sequestron_ERN"}
    if not ern_nodes:
        return graph

    topo_layers = graph.topological_order(nodes=ern_nodes.keys())
    for layer_id, layer_nodes in enumerate(topo_layers):
        for node_id in layer_nodes:
            ern_nodes[node_id].extra["layer_id"] = layer_id

    return graph


def recipe_to_networks(
    recipe: Recipe,
    rules: Optional[list[GraphRewritingRule]] = None,
    invert=True,
    lib: Optional[PartsLibrary] = None,
) -> list[Network]:
    from biocomp.inversion import invert_all_paths

    rules = rules or br.ALL_RULES
    lib = lib or LibraryContext.get_library()
    assert lib is not None, "PartsLibrary must be provided or set in LibraryContext"

    cdg = build_central_dogma_graph_direct(recipe.content, lib)
    compg = apply_rule_sequence(rules, cdg)
    assert len(compg) == 1, "Multiple computation graphs generated before inversion"
    compg = compg[0]
    compg = br.sort_output_edges(compg)
    compg = assign_ern_layer_ids(compg)
    graphs = invert_all_paths(compg) if invert else [compg]

    result = []

    for graph in graphs:
        graph = assign_ern_layer_ids(graph)
        net = Network(compute_graph=graph)
        dependent_outputs_names = "_".join(net.get_dependent_output_proteins())
        base_name = recipe.name or "network"
        net.name = f"{base_name}_{dependent_outputs_names}"
        result.append(net)

    return result


class NetworkConstructionError(Exception):
    """Exception for errors during network construction"""

    pass


def graphstate_to_cdg_df(graph: GraphState) -> pd.DataFrame:
    """Converts a GraphState object to a Central Dogma Graph DataFrame."""
    node_data = {
        node.node_id: {
            "type": node.node_type,
            **node.extra,
            "predecessor": [],
            "successor": [],
        }
        for node in graph.nodes.values()
    }

    for edge in graph.edges.values():
        if edge.source_id in node_data and edge.target_id in node_data:
            node_data[edge.source_id]["successor"].append(edge.target_id)
            node_data[edge.target_id]["predecessor"].append(edge.source_id)

    for nid in node_data:
        if not node_data[nid]["predecessor"]:
            node_data[nid]["predecessor"] = None
        if not node_data[nid]["successor"]:
            node_data[nid]["successor"] = None

    return pd.DataFrame.from_dict(node_data, orient="index")


def _get_dna(tu: TranscriptionUnit, lib: PartsLibrary) -> Tuple[list[str], dict[str, list[str]]]:
    content = [s.part for s in tu.slots if s.maps_to_parameter is None and s.part is not None]
    return content, tu.params


def _get_downstream(tu: TranscriptionUnit, transform: str, lib: PartsLibrary):
    dna_content, dna_params = _get_dna(tu, lib)
    if not dna_content:
        return (), {}

    d = lib.pc.loc[dna_content]
    content = tuple(d[d[transform] == 1].index)
    params = {}
    for param_name, parts in dna_params.items():
        non_none_parts = [p for p in parts if p is not None]
        if non_none_parts:
            p = lib.pc.loc[non_none_parts]
            if p[transform].sum() > 0:
                params[param_name] = list(p[p[transform] == 1].index)
    return content, params


def _get_rna(tu: TranscriptionUnit, lib: PartsLibrary):
    return _get_downstream(tu, "transcripted", lib)


def _get_prt(tu: TranscriptionUnit, lib: PartsLibrary):
    return _get_downstream(tu, "translated", lib)


def _make_hashable(params, tu_obj):
    hashable_params = {}
    for param_name, parts in params.items():
        ref_id = tu_obj.param_ref_ids.get(param_name)
        if ref_id is not None:
            hashable_params[param_name] = (f"ref:{ref_id}",)
        else:
            hashable_params[param_name] = tuple(parts) if isinstance(parts, list) else (parts,)
    return tuple(sorted((k, v) for k, v in hashable_params.items()))


def preprocess_network_tus(recipe: CoTxList, lib: PartsLibrary) -> dict[str, Any]:
    """
    Parses the recipe and returns a dictionary with all necessary
    pre-calculated information for building either the primal or dual graph.
    """
    if not recipe:
        return {}

    tu_map = {}
    tu_to_cotx_map = {}
    global_tu_index = 0

    for cotx_index, cotx_group in enumerate(recipe):
        group_name = cotx_group.name or f"cotx_{cotx_index + 1}"
        for unit_index, unit in enumerate(cotx_group.units):
            # Create unique TUID that includes cotx group info for duplicate plasmids
            base_name = unit.name or f"TU_{global_tu_index}"
            tuid = f"{base_name}_{group_name}"
            tu_map[tuid] = unit
            tu_to_cotx_map[tuid] = group_name
            global_tu_index += 1

    tu_info = {}
    for tuid, tu in tu_map.items():
        dna_content, dna_params = _get_dna(tu, lib)
        rna_content, rna_params = _get_rna(tu, lib)
        prt_content, prt_params = _get_prt(tu, lib)
        tu_info[tuid] = {
            "tu": tu,
            "cotx_group": tu_to_cotx_map[tuid],
            "DNA_content": dna_content,
            "DNA_params": dna_params,
            "RNA_content": rna_content,
            "RNA_params": rna_params,
            "PRT_content": prt_content,
            "PRT_params": prt_params,
            "RNA_params_hashable": _make_hashable(rna_params, tu),
            "PRT_params_hashable": _make_hashable(prt_params, tu),
        }

    def has_multi_value_params(tu):
        return any(isinstance(p, list) and len(p) > 1 for p in tu.params.values())

    is_committed = any(has_multi_value_params(tu) for tu in tu_map.values()) and any(
        not has_multi_value_params(tu) for tu in tu_map.values()
    )

    return {"recipe": recipe, "tu_map": tu_map, "tu_info": tu_info, "is_committed": is_committed}


def _build_cdg_primal_from_preprocessed(
    preprocessed_data: dict[str, Any], lib: PartsLibrary, custom_outputs_parts=None
) -> GraphState:
    """Builds the primal CDG where nodes are biological entities (DNA, RNA, PRT)."""
    tu_info = preprocessed_data["tu_info"]
    is_committed = preprocessed_data["is_committed"]

    nodes, next_node_id = [], 0
    group_to_node_id, tu_to_node_id = {}, {}

    for tuid, info in tu_info.items():
        dna_group_key = ("DNA", tuid)
        rna_group_key = (
            "RNA",
            info["RNA_content"]
            if is_committed
            else (info["RNA_content"], info["RNA_params_hashable"]),
        )
        prt_group_key = (
            "PRT",
            info["PRT_content"]
            if is_committed
            else (info["PRT_content"], info["PRT_params_hashable"]),
        )

        for key in [dna_group_key, rna_group_key, prt_group_key]:
            if key not in group_to_node_id:
                group_to_node_id[key] = next_node_id
                next_node_id += 1
            tu_to_node_id[(key[0], tuid)] = group_to_node_id[key]

    node_id_to_info = {nid: {"type": key[0], "tu_ids": []} for key, nid in group_to_node_id.items()}
    for (type, tuid), nid in tu_to_node_id.items():
        node_id_to_info[nid]["tu_ids"].append(tuid)

    outputs = (custom_outputs_parts or []) + lib.parts[
        lib.parts.category == "fluo_marker"
    ].index.tolist()

    for nid, info in node_id_to_info.items():
        rep_tuid = info["tu_ids"][0]
        content = tu_info[rep_tuid][f"{info['type']}_content"]
        params = tu_info[rep_tuid][f"{info['type']}_params"]
        is_output = info["type"] == "PRT" and any(o in content for o in outputs)
        extra = {
            "tu_id": info["tu_ids"],
            "content": content,
            "params": params,
            "is_output": is_output,
            "is_input": None,
            "content_type": tuple(lib.parts.loc[p].iloc[0] for p in content) if content else (),
        }
        nodes.append(GraphNode(node_id=nid, node_type=info["type"], extra=extra))

    edges = []
    for tuid in tu_info:
        edges.append(
            GraphEdge(
                source_id=tu_to_node_id[("DNA", tuid)],
                target_id=tu_to_node_id[("RNA", tuid)],
                from_output_slot=0,
                to_input_slot=0,
                content=(),
            )
        )

    rna_nodes = [n for n in nodes if n.node_type == "RNA"]
    prt_nodes = [n for n in nodes if n.node_type == "PRT"]
    for rna_node in rna_nodes:
        rna_tu_set = set(rna_node.extra["tu_id"])
        for prt_node in prt_nodes:
            if rna_tu_set.issubset(set(prt_node.extra["tu_id"])):
                edges.append(
                    GraphEdge(
                        source_id=rna_node.node_id,
                        target_id=prt_node.node_id,
                        from_output_slot=0,
                        to_input_slot=0,
                        content=(),
                    )
                )

    unique_edges = {(e.source_id, e.target_id): e for e in edges}.values()
    nodes_dict = {n.node_id: n for n in nodes}
    edges_dict = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e for e in unique_edges
    }
    return GraphState(nodes=nodes_dict, edges=edges_dict)


def _build_cdg_dual_from_preprocessed(
    preprocessed_data: dict[str, Any], lib: PartsLibrary, custom_outputs_parts=None
) -> GraphState:
    """Builds the dual CDG where nodes are transformations."""
    recipe = preprocessed_data["recipe"]
    tu_info = preprocessed_data["tu_info"]
    is_committed = preprocessed_data["is_committed"]
    outputs_list = (custom_outputs_parts or []) + lib.parts[
        lib.parts.category == "fluo_marker"
    ].index.tolist()

    nodes, edges = [], []
    next_node_id = 0

    # Create a mapping from (source_id, cotx_group) to normalized ratio and range info
    from biocomp.recipe import NumRange, FluoIntensity

    source_cotx_to_ratio_map: dict[tuple[str | None, str], tuple[float, Optional[NumRange]]] = {}
    cotx_to_fluo_bias: dict[str, FluoIntensity] = {}  # cotx_group -> FluoIntensity

    for i, cotx in enumerate(recipe or []):
        group_name = cotx.name or f"cotx_{i + 1}"

        # Build list of unique sources in order of first appearance
        unique_sources_ordered = []
        seen = set()
        for tu in cotx.units:
            if tu.source not in seen:
                unique_sources_ordered.append(tu.source)
                seen.add(tu.source)

        if cotx.ratios and len(cotx.ratios) != len(unique_sources_ordered):
            raise ValueError(
                f"CoTransfection '{group_name}': ratios count ({len(cotx.ratios)}) "
                f"must match number of unique sources ({len(unique_sources_ordered)})"
            )
        raw_ratios = cotx.ratios or [1.0] * len(unique_sources_ordered)

        # Track fluo_bias info for this cotx
        if cotx.fluo_bias is not None:
            cotx_to_fluo_bias[group_name] = cotx.fluo_bias

        # Extract numeric values for normalization (use midpoint for ranges)
        numeric_ratios = []
        for r in raw_ratios:
            if isinstance(r, NumRange):
                # use midpoint or 1.0 if unbounded
                min_v = r.min if r.min is not None else 0.0
                max_v = r.max if r.max is not None else 1.0
                numeric_ratios.append((min_v + max_v) / 2.0)
            else:
                numeric_ratios.append(float(r))

        ratio_sum = sum(numeric_ratios)
        # Normalize ratios within each cotx group to sum to 1.0
        normalized_ratios = (
            [r / ratio_sum for r in numeric_ratios]
            if ratio_sum > 0
            else [1.0 / len(unique_sources_ordered)] * len(unique_sources_ordered)
        )

        # Map sources to their ratios
        source_to_norm_ratio_map = {}
        for source, norm_ratio, orig_ratio in zip(
            unique_sources_ordered, normalized_ratios, raw_ratios
        ):
            source_to_norm_ratio_map[source] = (norm_ratio, orig_ratio)

        for unit_idx, unit in enumerate(cotx.units):
            norm_ratio, orig_ratio = source_to_norm_ratio_map[unit.source]
            # Store both normalized value and original range info (if NumRange)
            range_info = orig_ratio if isinstance(orig_ratio, NumRange) else None
            # Also store param_ref_ids, TU name, position, and cotx_index for roundtrip preservation
            # Use position_in_source if explicitly set (non-zero), otherwise use unit_idx
            position = (
                unit.position_in_source
                if (hasattr(unit, "position_in_source") and unit.position_in_source != 0)
                else unit_idx
            )
            source_cotx_to_ratio_map[(unit.source, group_name)] = (
                float(norm_ratio),
                range_info,
                dict(unit.param_ref_ids),  # copy to avoid mutation
                unit.name,  # TU name for roundtrip
                position,  # position in cotx for ordering
                i,  # cotx index for ordering CoTransfections
            )

    source_nodes, tx_nodes, tl_nodes = {}, {}, {}
    output_node, dead_end_nodes = None, {}

    for (source_id, cotx_group), (
        ratio,
        range_info,
        param_ref_ids,
        tu_name,
        position,
        cotx_index,
    ) in source_cotx_to_ratio_map.items():
        source_key = (source_id, cotx_group)
        if source_key not in source_nodes:
            source_nodes[source_key] = next_node_id
            next_node_id += 1

            source_extra = {
                "source_id": source_id,
                "cotx_group": cotx_group,
                "ratio": ratio,
                "param_ref_ids": param_ref_ids,  # store for roundtrip preservation
                "name": tu_name,  # store TU name for roundtrip
                "position_in_source": position,  # store position for ordering units
                "cotx_index": cotx_index,  # store index for ordering CoTransfections
            }
            # Add range info if ratio is unlocked
            if range_info is not None:
                source_extra["ratio_range"] = {
                    "min": range_info.min,
                    "max": range_info.max,
                }
            # Add fluo_bias info if this cotx has a bias
            if cotx_group in cotx_to_fluo_bias:
                fluo_bias = cotx_to_fluo_bias[cotx_group]
                source_extra["fluo_bias"] = {
                    "tu_id": fluo_bias.tu_id,
                    "value": (
                        fluo_bias.value
                        if isinstance(fluo_bias.value, (int, float))
                        else {"min": fluo_bias.value.min, "max": fluo_bias.value.max}
                    ),
                    "protein": fluo_bias.protein,
                    "units": fluo_bias.units,
                }
            nodes.append(
                GraphNode(node_id=source_nodes[source_key], node_type="source", extra=source_extra)
            )

    for tuid, info in tu_info.items():
        tu = info["tu"]

        rna_key = (
            info["RNA_content"]
            if is_committed
            else (info["RNA_content"], info["RNA_params_hashable"])
        )
        if rna_key not in tx_nodes:
            tx_nodes[rna_key] = next_node_id
            next_node_id += 1
            nodes.append(GraphNode(node_id=tx_nodes[rna_key], node_type="transcription", extra={}))

        prt_key = (
            info["PRT_content"]
            if is_committed
            else (info["PRT_content"], info["PRT_params_hashable"])
        )
        if prt_key not in tl_nodes:
            tl_nodes[prt_key] = next_node_id
            next_node_id += 1
            nodes.append(GraphNode(node_id=tl_nodes[prt_key], node_type="translation", extra={}))

    # track output slot per source to handle multiple TUs on same plasmid
    source_output_slot_counters = {}
    output_slot_counter = 0

    for tuid, info in tu_info.items():
        tu = info["tu"]

        cotx_group = info["cotx_group"]
        source_key = (tu.source, cotx_group)
        src_id = source_nodes[source_key]
        rna_key = (
            info["RNA_content"]
            if is_committed
            else (info["RNA_content"], info["RNA_params_hashable"])
        )
        tx_id = tx_nodes[rna_key]

        # assign sequential output slots for TUs sharing same source
        if source_key not in source_output_slot_counters:
            source_output_slot_counters[source_key] = 0
        source_output_slot = source_output_slot_counters[source_key]
        source_output_slot_counters[source_key] += 1

        edges.append(
            GraphEdge(
                source_id=src_id,
                target_id=tx_id,
                from_output_slot=source_output_slot,
                to_input_slot=0,
                content_type="DNA",
                content=tuple(
                    Part(name=p, category=lib.pc.loc[p, "category"] if p in lib.pc.index else "DNA")
                    for p in info["DNA_content"]
                ),
                content_embedding_names={k: tuple(v) for k, v in info["DNA_params"].items()},
                extra={"tu_id": [tuid]},
            )
        )

        prt_key = (
            info["PRT_content"]
            if is_committed
            else (info["PRT_content"], info["PRT_params_hashable"])
        )
        tl_id = tl_nodes[prt_key]
        edges.append(
            GraphEdge(
                source_id=tx_id,
                target_id=tl_id,
                from_output_slot=0,
                to_input_slot=0,
                content_type="RNA",
                content=tuple(Part(name=p, category="RNA") for p in info["RNA_content"]),
                content_embedding_names={k: tuple(v) for k, v in info["RNA_params"].items()},
            )
        )

        is_output = any(o in info["PRT_content"] for o in outputs_list)
        target_node_id = -1
        input_slot = 0

        if is_output:
            if output_node is None:
                output_node = next_node_id
                next_node_id += 1
                nodes.append(GraphNode(node_id=output_node, node_type="output", extra={}))
            target_node_id = output_node
            input_slot = output_slot_counter
            output_slot_counter += 1
        else:
            prt_content_tuple = tuple(sorted(info["PRT_content"]))
            if prt_content_tuple not in dead_end_nodes:
                dead_end_nodes[prt_content_tuple] = next_node_id
                next_node_id += 1
                nodes.append(
                    GraphNode(
                        node_id=dead_end_nodes[prt_content_tuple], node_type="deadend", extra={}
                    )
                )
            target_node_id = dead_end_nodes[prt_content_tuple]

        edges.append(
            GraphEdge(
                source_id=tl_id,
                target_id=target_node_id,
                from_output_slot=0,
                to_input_slot=input_slot,
                content_type="PRT",
                content=tuple(Part(name=p, category="PRT") for p in info["PRT_content"]),
                content_embedding_names={k: tuple(v) for k, v in info["PRT_params"].items()},
            )
        )

    unique_edges_dict = {}
    for e in edges:
        if e.content_type == "DNA":
            key = (e.source_id, e.from_output_slot)
        else:
            key = (e.source_id, e.target_id, e.content_type)
        if key not in unique_edges_dict:
            unique_edges_dict[key] = e

    nodes_dict = {n.node_id: n for n in nodes}
    edges_dict = {
        (e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e
        for e in unique_edges_dict.values()
    }
    return GraphState(nodes=nodes_dict, edges=edges_dict)


def build_central_dogma_graph_direct(
    recipe: CoTxList, lib: PartsLibrary, custom_outputs_parts=None, dual: bool = True
) -> GraphState:
    """
    Builds a central dogma graph directly from a recipe into a GraphState.
    Args:
        recipe: The CoTxList defining the recipe.
        lib: The parts library.
        custom_outputs_parts: Optional list of part names to be considered outputs.
        dual: If False (default), builds the primal graph where nodes are biological
              entities (DNA, RNA, PRT). If True, builds the dual graph where nodes
              are transformations (Source, Transcription, Translation).

    """
    preprocessed_data = preprocess_network_tus(recipe, lib)
    if not preprocessed_data:
        return GraphState(nodes={}, edges={})

    if dual:
        return _build_cdg_dual_from_preprocessed(preprocessed_data, lib, custom_outputs_parts)
    else:
        return _build_cdg_primal_from_preprocessed(preprocessed_data, lib, custom_outputs_parts)


##────────────────────────────────────────────────────────────────────────────}}}


def graphstate_to_compute_df(graph: GraphState) -> pd.DataFrame:
    """Converts a GraphState object to a Compute Graph DataFrame."""
    node_data = {
        node.node_id: {
            "type": node.node_type,
            "is_inverse_of": node.is_inverse_of,
            **node.extra,
            "input_from": [],
            "output_to": [],
        }
        for node in graph.nodes.values()
    }

    edges_by_source = {nid: [] for nid in node_data}
    edges_by_target = {nid: [] for nid in node_data}
    for edge in graph.edges.values():
        if edge.source_id in edges_by_source:
            edges_by_source[edge.source_id].append(edge)
        if edge.target_id in edges_by_target:
            edges_by_target[edge.target_id].append(edge)

    for nid, data in node_data.items():
        outgoing = sorted(edges_by_source.get(nid, []), key=lambda e: e.from_output_slot)
        data["output_to"] = [(e.target_id, e.to_input_slot) for e in outgoing]

        incoming = sorted(edges_by_target.get(nid, []), key=lambda e: e.to_input_slot)
        data["input_from"] = [(e.source_id, e.from_output_slot) for e in incoming]

    return pd.DataFrame.from_dict(node_data, orient="index")


def compute_df_to_graphstate(compute_df: pd.DataFrame) -> GraphState:
    """Converts a Compute Graph DataFrame to a GraphState object."""
    nodes: list[GraphNode] = []
    property_columns = ["cdg_input", "cdg_output", "extra", "source_id"]

    for idx, row in compute_df.iterrows():
        extra_properties = {
            col: row[col] for col in property_columns if col in row and pd.notna(row[col])
        }
        inverse_spec = row.get("is_inverse_of")
        if isinstance(inverse_spec, dict):
            inverse_spec = InverseSpec(**inverse_spec)

        node = GraphNode(
            node_id=int(idx),
            node_type=row["type"],
            is_inverse_of=inverse_spec,
            extra=extra_properties,
        )
        nodes.append(node)

    edges: list[GraphEdge] = []
    for idx, row in compute_df.iterrows():
        outputs = row.get("output_to")
        if outputs and pd.notna(outputs).all():
            for output_slot, (target_id, input_slot) in enumerate(outputs):
                edge = GraphEdge(
                    source_id=int(idx),
                    target_id=int(target_id),
                    from_output_slot=int(output_slot),
                    to_input_slot=int(input_slot),
                    content=(),
                )
                edges.append(edge)

    nodes_dict = {n.node_id: n for n in nodes}
    edges_dict = {(e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e for e in edges}
    return GraphState(nodes=nodes_dict, edges=edges_dict)


##────────────────────────────────────────────────────────────────────────────}}}


def old_network_compg_to_graphstate(old_network) -> GraphState:
    """
    Convert an old-style network.compute_graph (DataFrame) into a GraphState.

    - Nodes: reuse old compute_graph row index as node_id and row['type'] as node_type.
      Store all other compute_graph columns under node.extra with 'cg_' prefix to preserve info.
    - Edges: built from 'output_to' with slots. Biological edges (DNA/RNA/PRT) are enriched
      with content and content_embedding_names looked up from central_dogma_graph via cdg_input/cdg_output.
    """
    cg = getattr(old_network, "compute_graph", None)
    cdg = getattr(old_network, "central_dogma_graph", None)
    if cg is None or len(cg) == 0:
        return GraphState(nodes={}, edges={})

    def to_list(v):
        if v is None:
            return []
        if isinstance(v, (list, tuple)):
            return list(v)
        return [v]

    def parts_from_cdg_row(row: pd.Series, kind: str) -> tuple[Part, ...]:
        items = to_list(row.get("content"))
        lib = LibraryContext.get_library()
        return tuple(
            Part(name=str(p), category=lib.pc.loc[p, "category"] if p in lib.pc.index else kind)
            for p in items
        )

    def embeddings_from_cdg_row(row: pd.Series) -> dict[str, tuple[str, ...]]:
        p = row.get("params") or {}
        return {k: tuple(str(x) for x in to_list(v)) for k, v in p.items()}

    nodes: list[GraphNode] = []
    for nid, row in cg.iterrows():
        extra = {}
        for col, val in row.items():
            if col == "type":
                continue
            extra[f"cg_{col}"] = val

        node_type = str(row.get("type"))
        nodes.append(GraphNode(node_id=int(nid), node_type=node_type, extra=extra))

    def desired_content_type(src_type: str, dst_type: str) -> str | None:
        if src_type == "source" and dst_type == "transcription":
            return "DNA"
        if src_type == "transcription" and dst_type in ("translation", "sequestron_ERN"):
            return "RNA"
        if src_type == "translation" and dst_type in ("output", "sequestron_ERN"):
            return "PRT"
        if src_type == "sequestron_ERN" and dst_type == "translation":
            return "RNA"  # Sequestron ERN outputs RNA to translation
        return None

    def pick_cdg_row(
        src_row: pd.Series, dst_row: pd.Series, ctype: str, out_slot: int = 0
    ) -> pd.Series | None:
        if cdg is None or len(cdg) == 0:
            return None

        # First try to use the output slot to select from source's cdg_output
        src_outputs = to_list(src_row.get("cdg_output"))
        if src_outputs and out_slot < len(src_outputs):
            try:
                cid = int(src_outputs[out_slot])
                crow = cdg.loc[cid]
                if str(crow.get("type")) == ctype:
                    return crow
            except Exception:
                pass

        # Fallback to original logic
        candidate_ids: list[int] = []
        for key, r in (("cdg_output", src_row), ("cdg_input", dst_row)):
            for x in to_list(r.get(key)):
                try:
                    candidate_ids.append(int(x))
                except Exception:
                    pass
        for cid in candidate_ids:
            try:
                crow = cdg.loc[cid]
            except Exception:
                continue
            if str(crow.get("type")) == ctype:
                return crow
        return None

    edges: list[GraphEdge] = []
    for src_id, src_row in cg.iterrows():
        outputs = src_row.get("output_to")
        if not isinstance(outputs, list):
            continue
        for out_slot, pair in enumerate(outputs):
            try:
                dst_id, in_slot = pair
            except Exception:
                dst_id, in_slot = pair[0], 0
            dst_row = cg.loc[dst_id]
            ctype = desired_content_type(str(src_row.get("type")), str(dst_row.get("type")))

            kwargs = dict(
                source_id=int(src_id),
                target_id=int(dst_id),
                from_output_slot=int(out_slot),
                to_input_slot=int(in_slot),
                content=(),
            )
            if ctype is not None:
                crow = pick_cdg_row(src_row, dst_row, ctype, out_slot)
                if crow is not None:
                    kwargs["content"] = parts_from_cdg_row(crow, ctype)
                    kwargs["content_type"] = ctype
                    kwargs["content_embedding_names"] = embeddings_from_cdg_row(crow)
            elif str(src_row.get("type")) != "numeric":
                # When ctype is None and source is NOT numeric, try to find content
                # This handles edges to dead-end nodes (like output nodes)
                # Numeric edges should never have content (they're control flow edges)
                for possible_ctype in ("DNA", "RNA", "PRT"):
                    crow = pick_cdg_row(src_row, dst_row, possible_ctype, out_slot)
                    if crow is not None:
                        kwargs["content"] = parts_from_cdg_row(crow, possible_ctype)
                        kwargs["content_type"] = possible_ctype
                        kwargs["content_embedding_names"] = embeddings_from_cdg_row(crow)
                        break
            edges.append(GraphEdge(**kwargs))

    nodes_dict = {n.node_id: n for n in nodes}
    edges_dict = {(e.source_id, e.target_id, e.from_output_slot, e.to_input_slot): e for e in edges}
    return GraphState(nodes=nodes_dict, edges=edges_dict)


##                      --     network info generation     --

def get_uorf_value(params):
    if "tl_rate" in params:
        u = params["tl_rate"][0].split("_")[0] if isinstance(params["tl_rate"], (list, tuple)) else params["tl_rate"].split("_")[0]
        try:
            v = int(u[:-1]) * 10
        except ValueError:
            v = 0
        if u[-1] == "w":
            v = v - 5
        return v
    else:
        return 0

UORF_DICT = {
    0: "No uORF",
    5: "weak uORF",
    10: "1x uORF",
    20: "2x uORF",
    30: "3x uORF",
    40: "4x uORF",
    50: "5x uORF",
    60: "6x uORF",
    70: "7x uORF",
    80: "8x uORF",
}

def get_all_ERN_ids(network):
    assert network.compute_graph is not None
    ERN_ids = [n.node_id for n in network.compute_graph.nodes.values() if n.node_type == "sequestron_ERN"]
    return ERN_ids

def get_all_ERNs_names(network):
    ERN_ids = get_all_ERN_ids(network)
    ERN_names = []
    for ern_id in ERN_ids:
        node = network.compute_graph.nodes[ern_id]
        if node.extra and "seq_name" in node.extra:
            name = node.extra["seq_name"].split("#")[0].split("::")[-1]
            ERN_names.append(name)
    return ERN_names

def get_uorf_names(uorf_values, ern_names):
    uorf_names = []
    for uorf, ern_name in zip(uorf_values, ern_names):
        ERN_uorf, REC_uorf = uorf
        ERN_uorf = UORF_DICT[ERN_uorf]
        REC_uorf = UORF_DICT[REC_uorf]
        uorf_names.append((f"{ern_name} ERN: {ERN_uorf}", f"{ern_name} REC: {REC_uorf}"))
    return uorf_names

def get_all_uorf_values(network):
    assert network.compute_graph is not None
    ERN_ids = get_all_ERN_ids(network)
    ERN_names = get_all_ERNs_names(network)
    values = []

    for ern_id in ERN_ids:
        node = network.compute_graph.nodes[ern_id]
        incoming_edges = [e for e in network.compute_graph.edges.values() if e.target_id == ern_id]
        incoming_edges = sorted(incoming_edges, key=lambda e: e.to_input_slot)

        if len(incoming_edges) >= 2:
            edge0 = incoming_edges[0]
            edge1 = incoming_edges[1]

            # Try to get embedding names from edge, or trace back to translation node
            val0 = _get_uorf_value_from_edge_or_source(network.compute_graph, edge0)
            val1 = _get_uorf_value_from_edge_or_source(network.compute_graph, edge1)
            values.append((val0, val1))
        else:
            values.append((0, 0))

    names = get_uorf_names(values, ERN_names)
    return tuple(values), tuple(names)

def _get_uorf_value_from_edge_or_source(graph, edge):
    """Get uORF value from edge's content_embedding_names, or trace back to find it."""
    # First try direct embedding names on the edge
    if hasattr(edge, 'content_embedding_names') and edge.content_embedding_names:
        return get_uorf_value(edge.content_embedding_names)

    # If edge has no embedding info, trace back through the source node
    source_node = graph.nodes.get(edge.source_id)
    if source_node and source_node.node_type == 'translation':
        # Find the incoming edge to this translation node
        incoming_to_tl = [e for e in graph.edges.values() if e.target_id == source_node.node_id]
        if incoming_to_tl:
            # Get the first incoming edge (should be from transcription)
            tl_input_edge = incoming_to_tl[0]
            if hasattr(tl_input_edge, 'content_embedding_names') and tl_input_edge.content_embedding_names:
                return get_uorf_value(tl_input_edge.content_embedding_names)

    return 0

def get_ERN_ids(network):
    return get_all_ERN_ids(network)

def get_RCB_ids(network):
    assert network.compute_graph is not None
    return [n.node_id for n in network.compute_graph.nodes.values()
            if n.node_type and n.node_type.startswith("sequestron_R")]

def get_sequestron_ids(network):
    assert network.compute_graph is not None
    return [n.node_id for n in network.compute_graph.nodes.values()
            if n.node_type and n.node_type.startswith("sequestron_")]

def get_network_family(network):
    erns = get_ERN_ids(network)
    rcbs = get_RCB_ids(network)
    all_seqs = get_sequestron_ids(network)

    layers = network.compute_graph.topological_order(all_seqs) if all_seqs else []

    seqtype = "none"
    family = "unknown"
    match (len(erns) > 0, len(rcbs) > 0):
        case (True, True):
            seqtype = "hybrid"
        case (True, False):
            seqtype = "ERN"
        case (False, True):
            seqtype = "RCB"

    match (len(all_seqs), len(layers)):
        case (0, 0):
            family = ""
        case (1, 1):
            family = "single"
        case (2, 2):
            family = "cascade"
        case (2, 1):
            family = "dual region"
        case (3, 1):
            family = "triple region"
        case (3, 2):
            family = "bandpass"
        case _:
            family = f"complex ({len(all_seqs)} seqs, {len(layers)} layers)"

    return family, seqtype

def get_ratio(agg_node, network):
    assert network.compute_graph is not None
    if agg_node.extra and "ratios" in agg_node.extra:
        ratios = np.array(agg_node.extra["ratios"])
    else:
        ratios = np.array([1.0])

    min_ratio = np.maximum(ratios.min(), 1e-6)
    normed_ratios = np.round(ratios / min_ratio, 2)

    def is_round(x):
        return x == int(x)

    normed_ratios = [str(int(r)) if is_round(r) else str(r) for r in normed_ratios]

    incoming_edges = [e for e in network.compute_graph.edges.values() if e.target_id == agg_node.node_id]
    incoming_edges = sorted(incoming_edges, key=lambda e: e.to_input_slot)

    tu_names = []
    for edge in incoming_edges:
        source_node = network.compute_graph.nodes[edge.source_id]
        if source_node.extra and "tu_name" in source_node.extra:
            tu_names.append(source_node.extra["tu_name"])
        elif edge.content:
            tu_names.append(str(edge.content[0]) if edge.content else "unknown")
        else:
            tu_names.append("unknown")

    sorted_pairs = sorted(zip(tu_names, normed_ratios[:len(tu_names)]))
    sorted_tu_names, sorted_ratios = zip(*sorted_pairs) if sorted_pairs else ([], [])

    return (tuple(sorted_tu_names), tuple(sorted_ratios))

def get_ratios(network):
    assert network.compute_graph is not None
    agg_nodes = [n for n in network.compute_graph.nodes.values() if n.node_type == "aggregation"]
    all_ratios = [get_ratio(a, network) for a in agg_nodes]
    return all_ratios

def cotx_ratios_str(cotx):
    lines = []
    for tus, ratios in cotx:
        lines.append(":".join(tus) + " -> " + ":".join(ratios))
    return "\n".join(lines)

def get_parts_categories(parts, lib):
    res = {}
    for part in parts:
        if part in lib.parts.index:
            res[part] = lib.parts.loc[part].category
        else:
            res[part] = "unknown"
    return res

def get_tu_parts(tu, lib):
    from biocomp.library import load_lib
    if lib is None:
        lib = load_lib()
    parts = []
    for slot in tu.slots:
        if isinstance(slot.part, str):
            parts.append(slot.part)
        elif isinstance(slot.part, list) and len(slot.part) == 1:
            parts.append(slot.part[0])
    return get_parts_categories(parts, lib)

def get_all_parts(network, lib=None):
    from biocomp.library import load_lib
    if lib is None:
        lib = load_lib()

    # Try to get transcription_units if they exist (old system compatibility)
    if hasattr(network, 'transcription_units') and network.transcription_units:
        return {tname: get_tu_parts(t, lib) for tname, t in network.transcription_units.items()}

    # Otherwise, extract from the graph (new system)
    result = {}
    if network.compute_graph:
        # Find source nodes which represent transcription units (node_type is "source" in new system)
        source_nodes = [n for n in network.compute_graph.nodes.values() if n.node_type == "source"]
        for node in source_nodes:
            # Look for parts in outgoing edges - each edge may represent a different TU
            edges = [e for e in network.compute_graph.edges.values() if e.source_id == node.node_id]
            for edge in edges:
                # Get TU name from edge's extra.tu_id if available, otherwise from source node
                tu_name = None
                if hasattr(edge, 'extra') and edge.extra and 'tu_id' in edge.extra:
                    # Extract TU name from tu_id (e.g., 'L1-CasER1w_eYFP_cotx2' -> 'L1-CasER1w_eYFP')
                    tu_id_full = edge.extra['tu_id'][0] if isinstance(edge.extra['tu_id'], list) else edge.extra['tu_id']
                    # Remove the _cotxN suffix
                    tu_name = tu_id_full.rsplit('_cotx', 1)[0]
                elif node.extra and "name" in node.extra:
                    tu_name = node.extra["name"]

                if tu_name:
                    parts = {}
                    # Check for Part objects in content
                    if hasattr(edge, 'content') and edge.content:
                        for item in edge.content:
                            if hasattr(item, 'name'):
                                # It's a Part object
                                part_name = item.name
                                if part_name in lib.parts.index:
                                    parts[part_name] = lib.parts.loc[part_name].category
                            elif isinstance(item, str) and item in lib.parts.index:
                                # It's a string part name
                                parts[item] = lib.parts.loc[item].category

                    if parts:
                        # Add a suffix to distinguish multiple TUs from same source
                        # Count how many TUs we've already seen with this base name
                        base_name = tu_name
                        counter = 1
                        final_name = f"{base_name}_{counter}"
                        while final_name in result:
                            counter += 1
                            final_name = f"{base_name}_{counter}"
                        result[final_name] = parts

    return result

def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, (list, tuple)):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

def generate_network_info(network, lib=None):
    from biocomp.library import load_lib
    if lib is None:
        lib = load_lib()

    arch, seqtype = get_network_family(network)
    uorf_vals, uorf_names = get_all_uorf_values(network)

    genes = []
    if network.compute_graph:
        for node in network.compute_graph.nodes.values():
            if node.node_type == "translation" and node.extra and "protein" in node.extra:
                genes.append(node.extra["protein"])

    markers = tuple(sorted(network.get_inverted_input_proteins()))
    all_outputs = tuple(sorted(network.get_output_proteins()))
    dependent_outputs = tuple(sorted(list(set(all_outputs) - set(markers))))
    ern_names = get_all_ERNs_names(network)
    cotx = get_ratios(network)

    net_info = {
        "sequestron_type": seqtype,
        "architecture": arch,
        "ern_names": ern_names,
        "uorf_values": uorf_vals,
        "uorf_names": flatten(uorf_names),
        "genes": genes,
        "markers": markers,
        "output_proteins": all_outputs,
        "dependent_outputs": dependent_outputs,
        "cotx": cotx,
        "cotx_str": cotx_ratios_str(cotx),
        "ern_names_str": ", ".join(ern_names),
        "all_parts": get_all_parts(network, lib),
    }
    return net_info


##────────────────────────────────────────────────────────────────────────────}}
