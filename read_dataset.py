import json
import os
import logging
import random
from typing import Dict, List, Any, Optional, Tuple

import networkx as nx

from graph_model import SemanticGraph
from parse import from_dict, semantic_graph_to_networkx

# Logging configuration
LOGS_DIR = 'logs'
LOG_FILE = os.path.join(LOGS_DIR, 'parsing.log')

if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _read_graphs_from_file(file_path: str, file_idx: int) -> List[SemanticGraph]:
    """
    Reads and parses semantic graphs from a single JSON file.
    Handles cases where multiple JSON objects are concatenated in the file.

    Args:
        file_path: The path to the JSON file.

    Returns:
        A list of SemanticGraph objects parsed from the file.
    """
    semantic_graphs: List[SemanticGraph] = []
    buffer = ""
    brace_balance = 0
    start_index = -1
    sentence_index = 1
    with open(file_path, 'r', encoding='utf-8') as f:
        for char in f.read():
            buffer += char
            if char == '{':
                brace_balance += 1
                if start_index == -1:
                    start_index = len(buffer) - 1
            elif char == '}':
                brace_balance -= 1

            if brace_balance == 0 and start_index != -1:
                json_string = buffer[start_index:len(buffer)]
                start_index = -1  # Reset for the next JSON object
                try:
                    graph_data = json.loads(json_string)
                    graph_data['id'] = f"{graph_data['framework']}_{file_idx}_{sentence_index}"
                    semantic_graphs.append(from_dict(graph_data))
                    sentence_index += 1
                    buffer = ""  # Reset the buffer after successful parsing
                except json.JSONDecodeError as e:
                    logging.error(f"JSONDecodeError in {file_path}: {e}\nSegment: {json_string}")
                except Exception as e:
                    logging.error(f"Error processing JSON segment in {file_path}: {e}\nSegment: {json_string}")

    return semantic_graphs


def load_graphs(dataset_path: str, notation: str, split: float, random_state: Optional[int] = None) -> Tuple[Dict[str, List[SemanticGraph]], Dict[str, List[SemanticGraph]]]:
    """
    Loads semantic graph data for a specific notation from a given dataset directory.
    The dataset directory is expected to have a structure like:
    dataset_path/
        class_label_1/
            notation/
                sentence_1.json
                sentence_2.json
                ...
        class_label_2/
            notation/
                sentence_1.json
                sentence_2.json
                ...
        ...

    Each .json file is processed to extract one or more semantic graph objects.

    Args:
        dataset_path: The path to the root directory of the dataset.
        notation: The semantic graph notation to load (e.g., "amr", "drg", "eds", "ptg", "ucca").

    Returns:
        A dictionary where keys are the class labels and values are lists of
        SemanticGraph objects for that class and notation.
    """
    if random_state is not None:
        random.seed(random_state)

    training_data: Dict[str, List[SemanticGraph]] = {}
    testing_data: Dict[str, List[SemanticGraph]] = {}

    for class_label in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_label)
        if os.path.isdir(class_path):
            notation_path = os.path.join(class_path, notation)
            if os.path.isdir(notation_path):
                logging.info(f"Processing class directory: {notation_path}")
                all_files_in_dir = [f for f in os.listdir(notation_path) if f.endswith(".json")]
                random.shuffle(all_files_in_dir)  # Shuffle files to ensure random split

                num_test_files = int(len(all_files_in_dir) * split)
                test_files = all_files_in_dir[:num_test_files]
                train_files = all_files_in_dir[num_test_files:]

                training_data[class_label] = []
                testing_data[class_label] = []

                # Process training files
                train_successful_parses = 0
                for filename in train_files:
                    file_path = os.path.join(notation_path, filename)
                    graphs_from_file = _read_graphs_from_file(file_path,
                                                              len(training_data[class_label]) + 1)  # Pass a unique ID
                    training_data[class_label].extend(graphs_from_file)
                    train_successful_parses += len(graphs_from_file)

                logging.info(f"Successfully parsed {train_successful_parses} training graphs from: {notation_path}")

                # Process testing files
                test_successful_parses = 0
                for filename in test_files:
                    file_path = os.path.join(notation_path, filename)
                    graphs_from_file = _read_graphs_from_file(file_path,
                                                              len(testing_data[class_label]) + 1)  # Pass a unique ID
                    testing_data[class_label].extend(graphs_from_file)
                    test_successful_parses += len(graphs_from_file)

                logging.info(f"Successfully parsed {test_successful_parses} testing graphs from: {notation_path}")
            else:
                logging.warning(f"Notation '{notation}' not found in class '{class_label}'.")

    return training_data, testing_data


def load_graphs_as_networkx(dataset_path: str, notation: str, split: float, random_state: Optional[int] = None) -> Tuple[Dict[str, List[nx.MultiDiGraph]], Dict[str, List[nx.MultiDiGraph]]]:
    """
    Loads semantic graph data for a specific notation from a given dataset directory
    and converts them into networkx.MultiDiGraph objects.

    Args:
        dataset_path: The path to the root directory of the dataset.
        notation: The semantic graph notation to load (e.g., "amr", "drg", "eds", "ptg", "ucca").
        split: The proportion of data to be allocated for the testing set (e.g., 0.2 for 20%).
        random_state: Seed for the random number generator for reproducibility.

    Returns:
        A tuple of two dictionaries:
        - training_data: Keys are class labels, values are lists of networkx.MultiDiGraph objects
                         for the training set.
        - testing_data: Keys are class labels, values are lists of networkx.MultiDiGraph objects
                        for the testing set.
    """
    if random_state is not None:
        random.seed(random_state)

    training_data_nx: Dict[str, List[nx.MultiDiGraph]] = {}
    testing_data_nx: Dict[str, List[nx.MultiDiGraph]] = {}

    for class_label in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_label)
        if os.path.isdir(class_path):
            notation_path = os.path.join(class_path, notation)
            if os.path.isdir(notation_path):
                logging.info(f"Processing class directory: {notation_path}")
                all_files_in_dir = [f for f in os.listdir(notation_path) if f.endswith(".json")]
                random.shuffle(all_files_in_dir)  # Shuffle files to ensure random split

                num_test_files = int(len(all_files_in_dir) * split)
                test_files = all_files_in_dir[:num_test_files]
                train_files = all_files_in_dir[num_test_files:]

                training_data_nx[class_label] = []
                testing_data_nx[class_label] = []

                # Process training files
                train_successful_parses = 0
                for filename in train_files:
                    file_path = os.path.join(notation_path, filename)
                    semantic_graphs_from_file = _read_graphs_from_file(file_path,
                                                                       len(training_data_nx[class_label]) + 1)
                    for sem_graph in semantic_graphs_from_file:
                        try:
                            nx_graph = semantic_graph_to_networkx(sem_graph)
                            training_data_nx[class_label].append(nx_graph)
                            train_successful_parses += 1
                        except Exception as e:
                            logging.error(f"Failed to convert SemanticGraph to NetworkX for graph ID {sem_graph.id} in {file_path}: {e}")


                logging.info(f"Successfully converted {train_successful_parses} training graphs from: {notation_path}")

                # Process testing files
                test_successful_parses = 0
                for filename in test_files:
                    file_path = os.path.join(notation_path, filename)
                    semantic_graphs_from_file = _read_graphs_from_file(file_path,
                                                                       len(testing_data_nx[class_label]) + 1)
                    for sem_graph in semantic_graphs_from_file:
                        try:
                            nx_graph = semantic_graph_to_networkx(sem_graph)
                            testing_data_nx[class_label].append(nx_graph)
                            test_successful_parses += 1
                        except Exception as e:
                            logging.error(f"Failed to convert SemanticGraph to NetworkX for graph ID {sem_graph.id} in {file_path}: {e}")

                logging.info(f"Successfully converted {test_successful_parses} testing graphs from: {notation_path}")
            else:
                logging.warning(f"Notation '{notation}' not found in class '{class_label}'.")

    return training_data_nx, testing_data_nx


def semantic_graph_to_networkx_with_combined_edge_labels(sem_graph: SemanticGraph) -> nx.MultiDiGraph:
    """
    Converts a SemanticGraph object to a networkx.MultiDiGraph.
    Node labels remain their original labels.
    Edge labels are encoded as 'source_label::original_edge_label::target_label'.
    """
    nx_graph = nx.MultiDiGraph()

    # Create a mapping from SemanticGraph node IDs to their original labels
    node_id_to_label = {node.id: node.label for node in sem_graph.nodes}

    # Add nodes to the MultiDiGraph with their original labels
    for node_data in sem_graph.nodes:
        nx_graph.add_node(node_data.id, label=node_data.label)

    # Add edges with combined labels
    for edge_data in sem_graph.edges:
        source_id = edge_data.source
        target_id = edge_data.target
        original_edge_label = edge_data.label

        # Get original node labels for combined edge label
        source_node_original_label = sem_graph.get_node_by_id(source_id)
        target_node_original_label = sem_graph.get_node_by_id(target_id)

        # Construct the new edge label
        combined_edge_label = f"{source_node_original_label}::{original_edge_label}::{target_node_original_label}"

        # Add the edge to the MultiDiGraph
        # 'key' parameter allows multiple edges between the same two nodes
        # 'label' attribute stores the combined string
        nx_graph.add_edge(source_id, target_id, key=original_edge_label, label=combined_edge_label)

    return nx_graph


def convert_semantic_graphs_to_networkx(
    semantic_graphs_dict: Dict[str, List[SemanticGraph]]
) -> Dict[str, List[nx.MultiDiGraph]]:
    """
    Converts a dictionary of lists of SemanticGraph objects into a dictionary
    of lists of networkx.MultiDiGraph objects. Node labels retain their original form.
    Edge labels are encoded as 'source_label::original_edge_label::target_label'.

    Args:
        semantic_graphs_dict: A dictionary where keys are class labels and values
                              are lists of SemanticGraph objects.

    Returns:
        A dictionary where keys are class labels and values are lists of
        networkx.MultiDiGraph objects.
    """
    converted_graphs_nx: Dict[str, List[nx.MultiDiGraph]] = {}

    for class_label, sem_graphs_list in semantic_graphs_dict.items():
        converted_graphs_nx[class_label] = []
        logging.info(f"Converting graphs for class: {class_label}")

        for i, sem_graph in enumerate(sem_graphs_list):
            try:
                # Use the new conversion function with combined edge labels
                nx_graph = semantic_graph_to_networkx_with_combined_edge_labels(sem_graph)
                converted_graphs_nx[class_label].append(nx_graph)
            except Exception as e:
                logging.error(f"Failed to convert SemanticGraph (ID: {sem_graph.id}) "
                              f"to NetworkX for class '{class_label}', graph {i+1}: {e}")

    logging.info("Finished converting all SemanticGraphs to NetworkX MultiDiGraphs.")
    return converted_graphs_nx
