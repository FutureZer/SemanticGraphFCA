import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from typing import Dict, Union
from graph_model import Node, Edge, SemanticGraph


def preprocess_ucca_tokens(graph: SemanticGraph) -> SemanticGraph:
    """
    Preprocesses node labels for UCCA graphs:
    - Converts labels to lowercase.
    - Removes punctuation and special characters.
    Args:
        graph: A SemanticGraph object representing a UCCA graph.
    Returns:
        The modified SemanticGraph object with preprocessed node labels.
    """
    if graph.framework.lower() == "ucca":
        lemmatizer = WordNetLemmatizer()
        for node in graph.nodes:
            if node.label:
                processed_label = node.label.lower()
                # processed_label = re.sub(r'\d+', 'NUM', processed_label)
                processed_label = processed_label.translate(str.maketrans('', '', string.punctuation))
                processed_label = re.sub(r'[^a-zA-Z0-9\s]', '', processed_label)
                if processed_label:
                    lemmatized_tokens = [lemmatizer.lemmatize(token, 'n') for token in processed_label.split()]
                    node.label = " ".join(lemmatized_tokens)
                else:
                    node.label = None
    return graph


def process_empty_nodes(graph: SemanticGraph) -> SemanticGraph:
    """
    Processes empty (label=None) nodes in UCCA graphs:
    1. Removes empty leaf nodes, redirecting incoming edges to their parents.
    2. If an empty node has exactly one outgoing edge with label 'c',
       it inherits the label of the target node.
    For other cases or non-UCCA graphs, the graph is returned unchanged.

    Args:
        graph: A SemanticGraph object.

    Returns:
        The modified SemanticGraph object.
    """
    if graph.framework.lower() == "ucca":

        # First pass: Identify empty leaf nodes
        nodes_to_remove = set()
        for node in graph.nodes:
            if not node.label and not any(edge.source == node.id for edge in graph.edges):
                nodes_to_remove.add(node.id)

        # Process nodes, potentially inheriting labels
        processed_nodes = {}
        for node in graph.nodes:
            if node.id not in nodes_to_remove:

                # Removes all functional words (articles, "to be")
                func_word_edges = [edge for edge in graph.edges if edge.source == node.id and edge.label == 'f']
                for func_edge in func_word_edges:
                    child_node = graph.get_node_by_id(func_edge.target)
                    nodes_to_remove.add(child_node.id)

                if node.label is None:
                    # Set parent's label equals to core's element label
                    core_edges = [edge for edge in graph.edges if edge.source == node.id and edge.label == 'c']
                    if len(core_edges) == 1:
                        child_node = graph.get_node_by_id(core_edges[0].target)
                        if child_node and child_node.label:
                            processed_nodes[node.id] = Node(node_id=node.id, label=child_node.label)
                            nodes_to_remove.add(child_node.id)
                            continue

                processed_nodes[node.id] = node

        new_nodes = list(processed_nodes.values())
        new_node_ids = {node.id for node in new_nodes}
        new_edges = [
            Edge(source=edge.source, target=edge.target, label=edge.label, attributes=edge.attributes, raw_data=edge._raw_data)
            for edge in graph.edges
            if edge.source in new_node_ids and edge.target in new_node_ids and edge.source not in nodes_to_remove and edge.target not in nodes_to_remove
        ]

        new_tops = [top_id for top_id in graph.tops if top_id in new_node_ids and top_id not in nodes_to_remove]

        modified_graph = SemanticGraph(
            graph_id=graph.id,
            input_text=graph.input_text,
            nodes=new_nodes,
            edges=new_edges,
            tops=new_tops,
            framework=graph.framework,
            timestamp=graph.time
        )
        modified_graph.remove_isolated_nodes()
        return modified_graph
    else:
        return graph