from graph_model import Node, RichNode, Edge, SemanticGraph
from typing import List, Dict, Any, Optional

from graph_preprocessing import preprocess_ucca_tokens, process_empty_nodes


def _get_text_from_anchors(sentence: str, anchors: Optional[List[Dict[str, int]]]) -> Optional[str]:
    """Extracts the text segment based on the provided anchor information."""
    if not sentence or not anchors:
        return None

    # Assuming anchors can be non-contiguous, we'll concatenate them
    segments = []
    for anchor in anchors:
        start = anchor.get('from')
        end = anchor.get('to')
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(sentence):
            segments.append(sentence[start:end])

    return " ".join(segments) if segments else None


def _parse_nodes(node_data_list: List[Dict[str, Any]], sentence: str, framework: str) -> List[Node]:
    """Helper method to parse node data from the input dictionary."""
    nodes = []
    for node_dict in node_data_list:
        node_id = node_dict.get('id')
        if node_id is None:
            print(f"Warning: Node data missing 'id': {node_dict}")
            continue

        label = node_dict.get('label')
        properties = node_dict.get('properties')
        values = node_dict.get('values')
        anchors = node_dict.get('anchors')

        # Try to derive label from anchors if present and label is None
        if framework in ['ucca']:
            label = _get_text_from_anchors(sentence, anchors)

        # Handle RichNode for PTG and EDS based on the presence of 'concept_label'
        if framework in ['ptg', 'eds']:
            nodes.append(RichNode(node_id=node_id, label=_get_text_from_anchors(sentence, anchors),
                                  concept_label=label, properties=properties, values=values))
        else:
            nodes.append(Node(node_id=node_id, label=label))
    return nodes


def _parse_edges(edge_data_list: List[Dict[str, Any]]) -> List[Edge]:
    """Helper method to parse edge data from the input dictionary."""
    edges = []
    for edge_dict in edge_data_list:
        try:
            source = edge_dict['source']
            target = edge_dict['target']
            label = edge_dict['label'] if 'label' in edge_dict.keys() else None
        except KeyError as e:
            # print(f"Warning: Edge data missing mandatory key '{e}': {edge_dict}")
            continue

        attributes = edge_dict.get('attributes')
        edges.append(Edge(source=source, target=target, label=label, attributes=attributes))
    return edges


def from_dict(data: Dict[str, Any]) -> 'SemanticGraph':
    """
    Factory method to create a SemanticGraph instance from a dictionary.
    """
    graph_id = data.get('id', 'unknown_id')
    input_text = data.get('input', '')
    framework = data.get('framework', 'unknown_framework')
    timestamp = data.get('time')
    tops = data.get('tops', [])

    nodes = _parse_nodes(data.get('nodes', []), input_text, framework)
    edges = _parse_edges(data.get('edges', []))

    graph = SemanticGraph(graph_id=graph_id, input_text=input_text, nodes=nodes,
                          edges=edges, tops=tops, framework=framework, timestamp=timestamp)

    graph = preprocess_ucca_tokens(graph)

    return graph
