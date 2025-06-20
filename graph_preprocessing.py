import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from typing import Dict, Union, List, Set
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


def preprocess_graph_ptg(graph: SemanticGraph) -> None:
    """
    Preprocesses a single SemanticGraph by:
    1. Removing special coreference nodes (labels starting with '#') and their incident edges.
    2. Merging nodes with identical labels into a single canonical node,
       re-pointing all associated edges to the canonical node.
    3. Removing any isolated nodes that result from these operations.
    4. Removing duplicate edges that lead from the same source to the same target.

    Args:
        graph: The SemanticGraph object to be preprocessed (modified in-place).
    """

    for top_node_id in graph.tops:
        node = graph.get_node_by_id(top_node_id)
        if node and (node.label is None or node.label == ""):
            node.label = f"ROOT_{graph.id}"

    # --- Step 1: Remove special coreference nodes and their incident edges ---
    nodes_to_remove_ids = set()
    for node in graph.nodes:
        # Check if node.label exists and starts with '#'
        if node.label and isinstance(node.label, str) and node.label.startswith('#'):
            nodes_to_remove_ids.add(node.id)

    if nodes_to_remove_ids:
        # Filter out edges connected to the nodes to be removed
        new_edges = []
        for edge in graph.edges:
            if edge.source not in nodes_to_remove_ids and edge.target not in nodes_to_remove_ids:
                new_edges.append(edge)
        graph.edges = new_edges

        # Filter out the nodes themselves
        graph.nodes = [node for node in graph.nodes if node.id not in nodes_to_remove_ids]
        graph._nodes_by_id = {node.id: node for node in graph.nodes}

    # --- Step 2: Merge nodes with identical labels ---
    labels_to_nodes_map: Dict[str, List[Node]] = {}
    for node in graph.nodes:
        if node.label is not None: # Only consider nodes with a label for merging
            if node.label not in labels_to_nodes_map:
                labels_to_nodes_map[node.label] = []
            labels_to_nodes_map[node.label].append(node)

    nodes_to_keep: List[Node] = []
    id_mapping: Dict[Union[int, str], Union[int, str]] = {} # Map old_id -> new_id (canonical_id)

    for label, nodes_with_label in labels_to_nodes_map.items():
        if len(nodes_with_label) > 1:
            # Pick the first node as the canonical one (can be improved with better heuristics)
            canonical_node = nodes_with_label[0]
            nodes_to_keep.append(canonical_node)

            for duplicate_node in nodes_with_label[1:]: # All other nodes are duplicates
                id_mapping[duplicate_node.id] = canonical_node.id
        else:
            # If only one node with this label, just keep it
            nodes_to_keep.append(nodes_with_label[0])
            id_mapping[nodes_with_label[0].id] = nodes_with_label[0].id # Map to itself

    # Apply ID re-mapping to edges
    for edge in graph.edges:
        if edge.source in id_mapping:
            edge.source = id_mapping[edge.source]
        if edge.target in id_mapping:
            edge.target = id_mapping[edge.target]

    # Apply ID re-mapping to tops
    new_tops = []
    for top_id in graph.tops:
        if top_id in id_mapping:
            new_tops.append(id_mapping[top_id])
        else:
            new_tops.append(top_id)
    # Remove potential duplicate top_ids after mapping and convert back to list
    graph.tops = list(set(new_tops))

    # Update the graph's node list and internal ID map
    graph.nodes = nodes_to_keep
    graph._nodes_by_id = {node.id: node for node in nodes_to_keep}

    # --- Step 3: Remove any newly isolated nodes (after all changes) ---
    # This call also updates graph.nodes and graph._nodes_by_id
    graph.remove_isolated_nodes()

    # --- Step 4: Remove duplicate edges (source, target, label) ---
    unique_edges: Set[Edge] = set()
    new_edges_list: List[Edge] = []
    for edge in graph.edges:
        # We need a hashable representation for the edge to put into a set.
        # A tuple (source, target, label) works well.
        edge_tuple = (edge.source, edge.target, edge.label)
        if edge.source == edge.target:
            continue
        if edge_tuple not in unique_edges:
            unique_edges.add(edge_tuple)
            new_edges_list.append(edge)
    graph.edges = new_edges_list