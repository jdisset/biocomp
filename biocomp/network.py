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

        Note: Bias nodes are not included because they are direct inputs (not inverted outputs).
        They don't have input_position or input_from_output fields.
        """
        assert self.compute_graph is not None
        mapping = {}
        # Only input nodes have input_position/input_from_output (from inversion)
        # Bias nodes are direct inputs like numeric nodes, so we don't include them
        mask_types = ["input"]

        inputs = [n for n in self.compute_graph.nodes.values() if n.node_type in mask_types]
        for node in inputs:
            assert "input_position" in node.extra, f"input_position not in {node.extra}"
            assert "input_from_output" in node.extra, f"input_from_output not in {node.extra}"
            assert node.extra["input_position"] not in mapping
            mapping[node.extra["input_position"]] = node.extra["input_from_output"]

        assert set(mapping.keys()) == set(range(len(mapping.keys()))), f"Invalid mapping: {mapping}"
        assert len(mapping.keys()) == len(set(mapping.values())), f"Invalid mapping: {mapping}"
        return mapping

    def get_output_proteins(self, only_dependent_outputs: bool = False) -> list[str]:
        """Returns the names of all proteins that are outputs of the network"""
        assert self.compute_graph is not None
        if only_dependent_outputs:
            return self.get_dependent_output_proteins()

        onode = self.get_output_compute_node()

        incoming_edges = self.compute_graph.get_incoming_edges(onode.node_id)
        output_proteins = []
        for edge in sorted(incoming_edges, key=lambda e: e.input_slot):
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

        inputs = [n for n in self.compute_graph.nodes.values() if n.node_type == "input"]
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
        original_mapping = self.get_inverted_input_positions()
        output_proteins = self.get_output_proteins()
        assert bias_protein_name in output_proteins, (
            f"Invalid bias protein name: {bias_protein_name}"
        )
        output_position = output_proteins.index(bias_protein_name)

        assert output_position not in original_mapping.values(), (
            f"Protein {bias_protein_name} is already an input, not a bias"
        )

        biases = [n for n in self.compute_graph.nodes.values() if n.node_type == "bias"]
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
        inputs = [n for n in self.compute_graph.nodes.values() if n.node_type == "input"]

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
        tus_by_cotx = self._build_transcription_units(cotx_groups)
        bias_by_cotx = self._extract_bias_nodes()

        # Sort by cotx_index to preserve original order
        sorted_group_ids = sorted(cotx_groups.keys(), key=lambda g: cotx_groups[g]["cotx_index"])

        content = []
        for group_id in sorted_group_ids:
            info = cotx_groups[group_id]
            content.append(
                CoTransfection(
                    name=group_id if group_id != "cotx_1" or len(cotx_groups) > 1 else None,
                    units=tus_by_cotx[group_id],
                    ratios=info["ratios"] if len(info["ratios"]) > 1 else None,
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
        for node in self.compute_graph.nodes.values():
            if node.node_type == "source":
                group_id = node.extra.get("cotx_group")
                cotx_index = node.extra.get("cotx_index", 0)
                if group_id not in source_cotx_indices:
                    source_cotx_indices[group_id] = cotx_index

        for node in self.compute_graph.nodes.values():
            if node.node_type == "aggregation":
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

        for node in self.compute_graph.nodes.values():
            if node.node_type == "source":
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

    def _build_transcription_units(self, cotx_groups: dict) -> dict[str, list[TranscriptionUnit]]:
        tus_by_cotx = {}
        for group_id, info in cotx_groups.items():
            tus = []
            source_nodes_with_pos = []
            for source_id in info["source_ids"]:
                source_node = self._find_source_node(source_id, group_id)
                if source_node:
                    position = source_node.extra.get("position_in_source", 0)
                    source_nodes_with_pos.append((position, source_id, source_node))

            source_nodes_with_pos.sort(key=lambda x: x[0])

            for position, source_id, source_node in source_nodes_with_pos:
                param_ref_ids = source_node.extra.get("param_ref_ids", {})
                slots = self._extract_slots_from_source(source_node, param_ref_ids)

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
            tus_by_cotx[group_id] = tus
        return tus_by_cotx

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

    def _get_dna_edge(self, source_node):
        """Get DNA edge from source node, or None if not found"""
        assert self.compute_graph is not None
        outgoing = self.compute_graph.get_outgoing_edges(source_node.node_id)
        dna_edges = [e for e in outgoing if e.content_type == "DNA"]
        return dna_edges[0] if dna_edges else None

    def _should_include_embedding(
        self, emb_name: str, part_names: tuple, param_ref_ids: dict
    ) -> bool:
        """Check if embedding should be included (has real parts or explicit ref_id)"""
        implicit_empty = IMPLICIT_EMPTY.get(emb_name)
        has_real_parts = any(p != implicit_empty for p in part_names) if part_names else False
        has_ref_id = emb_name in param_ref_ids and param_ref_ids[emb_name] is not None
        return has_real_parts or has_ref_id

    def _extract_slots_from_source(self, source_node, param_ref_ids: dict = None) -> list[Slot]:
        """Reconstruct slots by sorting all parts by their biological category"""
        param_ref_ids = param_ref_ids or {}
        lib = LibraryContext.get_library()

        dna_edge = self._get_dna_edge(source_node)
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

    def _parse_fluo_bias_data(self, data_str):
        """Parse fluo_bias_data from string to dict"""
        import ast
        if not data_str:
            return None
        if isinstance(data_str, dict):
            return data_str
        try:
            return ast.literal_eval(data_str)
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

            # Try new format first (fluo_bias_data dict)
            fluo_bias_data = self._parse_fluo_bias_data(node.extra.get("fluo_bias_data"))
            if fluo_bias_data:
                bias_by_cotx[cotx_group] = self._create_fluo_intensity_from_dict(fluo_bias_data)
                continue

            # Fallback to old format (individual fields)
            tu_id = 0
            if tu_id_str := node.extra.get("tu_id", ""):
                try:
                    tu_id = int(tu_id_str)
                except (ValueError, TypeError):
                    pass

            value = self._parse_value_to_numrange_or_float(node.extra.get("value"))
            protein = node.extra.get("protein")
            if protein in ["", "None"]:
                protein = None
            units = node.extra.get("units", "AU") or "AU"

            # Fallback to aggregation node if values are empty
            if not tu_id_str and not node.extra.get("value"):
                if fallback := self._extract_bias_from_aggregation(node):
                    tu_id = fallback["tu_id"]
                    value = fallback["value"]
                    protein = fallback["protein"]
                    units = fallback["units"]

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
        dependency_map = {}
        for node_id, node in self.compute_graph.nodes.items():
            incoming = self.compute_graph.get_incoming_edges(node_id)
            if incoming:
                dependency_map[node_id] = set(e.source_id for e in incoming)
            else:
                dependency_map[node_id] = set()
        return dependency_map

    def topological_order(self, nodes=None, dependency_map=None):
        """Returns a list of lists of compute nodes from the network,
        where each node of a sublist can be computed independently of the others,
        but each sublist must be computed in order."""
        all_nodes = set(self.compute_graph.nodes.keys())
        nodes_set = set(nodes) if nodes is not None else all_nodes
        dependency_map = dependency_map or self.compute_dependency_map()

        visited = set()
        batches = []
        remaining = all_nodes.copy()

        while remaining:
            independent = [node for node in remaining if dependency_map[node].issubset(visited)]
            if not independent:
                msg = f"No independent node. Remaining:{set(self.compute_graph.nodes.keys()) - visited}. Visited:{visited}"
                raise ValueError(msg)
            visited.update(independent)
            remaining.difference_update(independent)
            batch = [node for node in independent if node in nodes_set]
            if batch:
                batches.append(batch)

        return batches


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
    graphs = invert_all_paths(compg) if invert else [compg]

    result = []

    for graph in graphs:
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
                output_slot=0,
                input_slot=0,
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
                        output_slot=0,
                        input_slot=0,
                        content=(),
                    )
                )

    unique_edges = {(e.source_id, e.target_id): e for e in edges}.values()
    nodes_dict = {n.node_id: n for n in nodes}
    edges_dict = {(e.source_id, e.target_id, e.output_slot, e.input_slot): e for e in unique_edges}
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
        raw_ratios = cotx.ratios or [1.0] * len(cotx.units)

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
            else [1.0 / len(cotx.units)] * len(cotx.units)
        )

        for unit_idx, (unit, norm_ratio, orig_ratio) in enumerate(
            zip(cotx.units, normalized_ratios, raw_ratios)
        ):
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

        source_output_slot = getattr(tu, "position_in_source", 0) or 0

        edges.append(
            GraphEdge(
                source_id=src_id,
                target_id=tx_id,
                output_slot=source_output_slot,
                input_slot=0,
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
                output_slot=0,
                input_slot=0,
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
                output_slot=0,
                input_slot=input_slot,
                content_type="PRT",
                content=tuple(Part(name=p, category="PRT") for p in info["PRT_content"]),
                content_embedding_names={k: tuple(v) for k, v in info["PRT_params"].items()},
            )
        )

    unique_edges_dict = {}
    for e in edges:
        if e.content_type == "DNA":
            key = (e.source_id, e.output_slot)
        else:
            key = (e.source_id, e.target_id, e.content_type)
        if key not in unique_edges_dict:
            unique_edges_dict[key] = e

    nodes_dict = {n.node_id: n for n in nodes}
    edges_dict = {
        (e.source_id, e.target_id, e.output_slot, e.input_slot): e
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


def build_central_dogma_graph(
    network: Network, lib: PartsLibrary, custom_outputs_parts=None
) -> pd.DataFrame:
    """
    Builds the primal central dogma graph and returns it as a pandas DataFrame
    for backward compatibility with the old API.
    """
    graph_state = build_central_dogma_graph_direct(network, lib, custom_outputs_parts, dual=False)
    return graphstate_to_cdg_df(graph_state)


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
        outgoing = sorted(edges_by_source.get(nid, []), key=lambda e: e.output_slot)
        data["output_to"] = [(e.target_id, e.input_slot) for e in outgoing]

        incoming = sorted(edges_by_target.get(nid, []), key=lambda e: e.input_slot)
        data["input_from"] = [(e.source_id, e.output_slot) for e in incoming]

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
                    output_slot=int(output_slot),
                    input_slot=int(input_slot),
                    content=(),
                )
                edges.append(edge)

    nodes_dict = {n.node_id: n for n in nodes}
    edges_dict = {(e.source_id, e.target_id, e.output_slot, e.input_slot): e for e in edges}
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
                output_slot=int(out_slot),
                input_slot=int(in_slot),
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
    edges_dict = {(e.source_id, e.target_id, e.output_slot, e.input_slot): e for e in edges}
    return GraphState(nodes=nodes_dict, edges=edges_dict)


##────────────────────────────────────────────────────────────────────────────}}
