import json
import os
import logging
from typing import Dict, List, Any

from graph_model import SemanticGraph
from parse import from_dict

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
    with open(file_path, 'r') as f:
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


def load_graphs(dataset_path: str, notation: str) -> Dict[str, List[SemanticGraph]]:
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
    loaded_data: Dict[str, List[SemanticGraph]] = {}

    for class_label in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_label)
        if os.path.isdir(class_path):
            notation_path = os.path.join(class_path, notation)
            if os.path.isdir(notation_path):
                logging.info(f"Processing class directory: {notation_path}")
                loaded_data[class_label] = []
                successful_parses = 0
                file_idx = 1
                for filename in os.listdir(notation_path):
                    if filename.endswith(".json"):
                        file_path = os.path.join(notation_path, filename)
                        graphs_from_file = _read_graphs_from_file(file_path, file_idx)
                        loaded_data[class_label].extend(graphs_from_file)
                        successful_parses += len(graphs_from_file)
                        file_idx += 1
                logging.info(f"Successfully parsed {successful_parses} graphs from: {notation_path}")
            else:
                logging.warning(f"Notation '{notation}' not found in class '{class_label}'.")

    return loaded_data
