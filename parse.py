import networkx as nx

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
            nodes.append(RichNode(node_id=node_id, label=label,
                                  anchors=anchors,
                                  concept_label=_get_text_from_anchors(sentence, anchors),
                                  properties=properties, values=values))
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


def semantic_graph_to_networkx(sem_graph: SemanticGraph) -> nx.MultiDiGraph:
    """
    Преобразует объект SemanticGraph в объект networkx.MultiDiGraph.

    Узлы и ребра из SemanticGraph будут добавлены в MultiDiGraph.
    Атрибуты узлов будут включать 'label', 'concept_label', 'properties', 'values' (для RichNode).
    Атрибуты ребер будут включать 'label' и 'attributes'.
    """
    G = nx.MultiDiGraph()

    node_id_dict = {}
    # Добавляем узлы
    for node in sem_graph.nodes:
        node_id_dict[node.id] = 'EMP' if node.label is None else node.label
        # node_attributes = {"label": "EMT" if node.label is None else node.label}

        # if isinstance(node, RichNode):
        #     node_attributes["concept_label"] = node.concept_label
        #     node_attributes["properties"] = node.properties
        #     node_attributes["values"] = node.values
        #     node_attributes.update(node.attributes) # Добавляем атрибуты из словаря attributes

        G.add_node(node_id_dict[node.id])

    # Добавляем ребра
    for edge in sem_graph.edges:
        G.add_edge(node_id_dict[edge.source], node_id_dict[edge.target], edge.label, label=edge.label)

    # Добавляем информацию о "топах" (корневых узлах) как атрибут графа, если они есть
    # if sem_graph.tops:
    #     G.graph["tops"] = sem_graph.tops
    # G.graph["id"] = sem_graph.id
    # G.graph["input_text"] = sem_graph.input_text
    # G.graph["framework"] = sem_graph.framework
    # G.graph["time"] = sem_graph.time

    return G
