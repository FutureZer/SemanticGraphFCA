from os import remove
from typing import List, Dict, Any, Optional, Union

class Node:
    """
    Base class representing a node in a semantic graph.

    For frameworks like UCCA, PTG, EDS, the 'label' is expected to be
    derived from text anchors *before* creating the Node instance.
    For AMR, DRG, the 'label' comes directly from the source data.
    """
    def __init__(self, node_id: Union[int, str], label: Optional[str]):
        """
        Initializes the base Node.

        Args:
            node_id: The unique identifier for the node.
            label: The primary label for the node (either from source or derived from anchors).
                   Can be None if no label is applicable or derivable.
        """
        self.id: Union[int, str] = node_id
        self.label: Optional[str] = label

    def __repr__(self) -> str:
        """Provides a string representation of the Node."""
        return f"{self.__class__.__name__}(id={self.id}, label='{self.label}')"


class RichNode(Node):
    """
    Extended node class for frameworks like PTG and EDS, which include
    additional properties, values, and a distinct concept label.

    Inherits 'id' and 'label' (derived from anchors) from the base Node class.
    """
    def __init__(self,
                 node_id: Union[int, str],
                 label: Optional[str],
                 concept_label: Optional[str],
                 properties: Optional[List[str]],
                 values: Optional[List[str]]):
        """
        Initializes the RichNode.

        Args:
            node_id: The unique identifier for the node.
            label: The primary label derived from text anchors (same as in base Node).
            concept_label: The original 'label' field from the PTG/EDS JSON,
                           representing the underlying concept or type.
            properties: A list of property keys associated with the node (e.g., 'sempos', 'frame').
            values: A list of property values corresponding to the keys in 'properties'.
        """
        super().__init__(node_id=node_id, label=label)
        self.concept_label: Optional[str] = concept_label

        self.properties: List[str] = properties if properties is not None else []
        self.values: List[str] = values if values is not None else []

        self.attributes: Dict[str, str] = {}
        if self.properties and self.values and len(self.properties) == len(self.values):
            self.attributes = dict(zip(self.properties, self.values))

    def __repr__(self) -> str:
        """Provides a string representation of the RichNode."""
        base_repr = super().__repr__()
        base_part = base_repr[:-1]
        parts = [
            f"concept_label='{self.concept_label}'",
            f"properties={self.properties}",
            f"values={self.values}"
        ]
        return f"{base_part}, {', '.join(parts)})"


class Edge:
    """
    Represents a single edge (relation) in a semantic graph.
    """
    def __init__(self, source: Union[int, str], target: Union[int, str], label: str,
                 attributes: Optional[Dict[str, Any]] = None, raw_data: Optional[Dict[str, Any]] = None):
        """
        Initializes an Edge object.

        Args:
            source: The ID of the source node.
            target: The ID of the target node.
            label: The label describing the relationship between source and target.
            attributes: Additional attributes for the edge (e.g., 'remote' in UCCA).
            raw_data: The original dictionary representation of the edge (optional, for reference).
        """
        self.source: Union[int, str] = source
        self.target: Union[int, str] = target
        self.label: str = label
        self.attributes: Dict[str, Any] = attributes if attributes is not None else {}
        self._raw_data = raw_data # Store original data if needed later

    def __repr__(self) -> str:
        parts = [f"source={self.source}", f"target={self.target}", f"label='{self.label}'"]
        if self.attributes:
            parts.append(f"attributes={self.attributes}")
        return f"Edge({', '.join(parts)})"


class SemanticGraph:
    """
    Base class for representing various semantic graph notations.
    It holds the common elements found across different graph formats.
    """
    def __init__(self, graph_id: str, input_text: str, nodes: List[Node],
                 edges: List[Edge], tops: List[Union[int, str]], framework: str,
                 timestamp: Optional[str] = None):
        """
        Initializes a SemanticGraph object.

        Args:
            graph_id: The identifier for the graph/sentence.
            input_text: The original input sentence.
            nodes: A list of Node objects.
            edges: A list of Edge objects.
            tops: A list of IDs of the top/root nodes in the graph.
            framework: The name of the semantic framework (e.g., "amr", "ucca").
            timestamp: The timestamp associated with the graph generation (optional).
        """
        self.id: str = graph_id
        self.input_text: str = input_text
        self.nodes: List[Node] = nodes
        self.edges: List[Edge] = edges
        self.tops: List[Union[int, str]] = tops
        self.framework: str = framework
        self.time: Optional[str] = timestamp

        # Optional: Create maps for faster node lookup if needed
        self._nodes_by_id: Dict[Union[int, str], Node] = {node.id: node for node in nodes}
        self.remove_isolated_nodes()

    def get_node_by_id(self, node_id: Union[int, str]) -> Optional[Node]:
        """Efficiently retrieves a node by its ID."""
        return self._nodes_by_id.get(node_id)

    def get_outgoing_edges(self, node_id: Union[int, str]) -> List[Edge]:
        """Retrieves all edges originating from the given node ID."""
        return [edge for edge in self.edges if edge.source == node_id]

    def get_incoming_edges(self, node_id: Union[int, str]) -> List[Edge]:
        """Retrieves all edges pointing to the given node ID."""
        return [edge for edge in self.edges if edge.target == node_id]

    def remove_isolated_nodes(self) -> None:
        """Removes nodes that have no incoming or outgoing edges."""
        connected_node_ids = set()
        for edge in self.edges:
            connected_node_ids.add(edge.source)
            connected_node_ids.add(edge.target)

        new_nodes = [node for node in self.nodes if node.id in connected_node_ids]
        self.nodes = new_nodes
        self._nodes_by_id = {node.id: node for node in new_nodes}
        self.tops = [top_id for top_id in self.tops if top_id in self._nodes_by_id]

    def __repr__(self) -> str:
        return (f"<{self.__class__.__name__}(id={self.id}, framework='{self.framework}', "
                f"nodes={len(self.nodes)}, edges={len(self.edges)})>")

    def __str__(self) -> str:
        """Provides a more detailed string representation."""
        nl = "\n"
        return (
            f"--- {self.framework.upper()} Graph (ID: {self.id}) ---\n"
            f"Input: {self.input_text}\n"
            f"Tops: {self.tops}\n"
            f"Nodes:\n{nl.join(f'  {node}' for node in self.nodes)}\n"
            f"Edges:\n{nl.join(f'  {edge}' for edge in self.edges)}\n"
            f"Timestamp: {self.time or 'N/A'}"
        )
