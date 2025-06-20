import os
import re
from collections import defaultdict

import yaml
from typing import List, Dict
import pickle
import shutil
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from merge_graphs import combine_from_sentence_graphs
from results_aggregator import construct_results_dataframe
from graph_pattern_reconstruction import graph_pattern_reconstruction_iterator
from graph_pattern_weighting import graph_pattern_weighting_iterator
from graph_pattern_scoring import graph_pattern_scoring_iterator
from graph_pattern_visualize import graph_pattern_visualize_iterator
from parse_to_gsofia import semantic_graphs_to_gsofia_format, LabelEncoder
from graph_model import SemanticGraph
from graph_preprocessing import preprocess_graph_ptg
from read_dataset import load_graphs_as_networkx


def get_project_root():
    return Path(__file__).parent

def do_operations(root, config, dataset='all', mode='frequent_subgraphs', notation='ptg', weighting='yes', graph_pattern_reconstruction='yes', visualization='no'):

    if dataset == 'all':
        data_root_bbc = os.path.join(str(root), config['bbcsport_data_prefix'])
        gsofia_root_bbc = os.path.join(str(root), 'gsofia', config['bbcsport_data_prefix'])
        training_bbc, testing_bbc = load_graphs_as_networkx(data_root_bbc, notation, 0.12, 42)
        # _, testing_bbc = load_graphs(data_root_bbc, notation, 0.12, 42)
        # testing_bbc = all_raw_graphs_actions(testing_bbc)
        # testing_bbc = convert_semantic_graphs_to_networkx(testing_bbc)

        if graph_pattern_reconstruction == 'yes':
            print("Reconstructing graph patterns...")
            graph_pattern_reconstruction_iterator(config['bbcsport_classes'], data_root_bbc, gsofia_root_bbc, notation, mode)
            # graph_pattern_reconstruction_iterator(config['ten_newsgroups_classes'], data_root_news, gsofia_root_news, notation, mode)

        if weighting == 'yes':
            print("Weighting graph patterns...")
            graph_pattern_weighting_iterator('bbcsport', config['bbcsport_classes'], data_root_bbc, training_bbc, notation, mode)
            # graph_pattern_weighting_iterator('newsgroup', config['bbcsport_classes'], gsofia_root_news, notation, mode)

        # ten_newsgroups_results = graph_pattern_scoring_iterator('ten_newsgroups', config['ten_newsgroups_classes'], str(root) + '/' + config['ten_newsgroups_data_prefix'], mode)
        bbcsport_results = graph_pattern_scoring_iterator('bbcsport', config['bbcsport_classes'], os.path.join(str(root), config['bbcsport_data_prefix']), testing_bbc, notation, mode)
        print(bbcsport_results)

        # ten_newsgroups_results_file_name = 'ten_newsgroups_results.pickle'
        bbcsport_results_file_name = f'bbcsport_results_{notation}.pickle'

        # with open(ten_newsgroups_results_file_name, 'wb') as handle:
        #    pickle.dump(ten_newsgroups_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # ten_newsgroups_results_path = os.path.join(str(root), config['ten_newsgroups_data_prefix'], ten_newsgroups_results_file_name)
        # shutil.move(ten_newsgroups_results_file_name, ten_newsgroups_results_path)

        with open(bbcsport_results_file_name, 'wb') as handle:
            pickle.dump(bbcsport_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        bbcsport_results_path = os.path.join(str(root), config['bbcsport_data_prefix'], bbcsport_results_file_name)
        shutil.move(bbcsport_results_file_name, bbcsport_results_path)

        construct_results_dataframe(str(root), notation, config)

    if visualization == 'yes':
        graph_pattern_reconstruction_iterator(config['bbcsport'][:10], os.path.join(str(root), config['example_data_prefix']), mode)
        graph_pattern_visualize_iterator(config['bbcsport'][:10], os.path.join(str(root), config['example_data_prefix']), mode)


def make_gsofia_dir(base_output_path: str):
    output_dir = os.path.join("gsofia", base_output_path)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def create_gsofia_input(loaded_graphs: Dict[str, List[SemanticGraph]], output_dir: str, notation: str):
    """
    Создает входные файлы gSOFIA для каждого класса графов и для всех графов вместе,
    используя отдельные LabelEncoder'ы для каждого класса и один для всех графов.
    Все используемые энкодеры сохраняются в один файл pickle.
    """
    # Словарь для хранения энкодеров по категориям
    all_encoders: Dict[str, LabelEncoder] = {}

    try:
        with open('gsofia/data/bbcsport/ptg-train/labels_enc.pkl', 'wb') as f:
            all_encoders = pickle.load(f)
    except Exception:
        print('Unable to load encodings')


    output_dir_full = os.path.join(output_dir, notation)
    os.makedirs(output_dir_full, exist_ok=True)

    all_graphs: List[SemanticGraph] = []

    for category, graphs in loaded_graphs.items():
        # Создаем новый энкодер для текущей категории
        category_encoder = LabelEncoder()
        all_encoders[category] = category_encoder  # Сохраняем энкодер в словарь

        output_file_path = os.path.join(output_dir_full, f"gsofia_input_{category}.txt")
        semantic_graphs_to_gsofia_format(graphs, output_file_path, category_encoder)
        all_graphs.extend(graphs)
        print(f"Graphs '{category}' saved in: {output_file_path}")
        print(
            f"Encoder for '{category}' has {len(category_encoder.get_node_encodings())} node labels and {len(category_encoder.get_edge_encodings())} edge labels.")

    encoders_filepath = os.path.join(output_dir_full, "labels_enc.pkl")
    with open(encoders_filepath, 'wb') as f:
        pickle.dump(all_encoders, f)
    print(f"\nAll encoders saved to: {encoders_filepath}")

    return output_dir_full


def all_raw_graphs_actions(sentence_graphs, output_dir="merged"):
    all_processed_document_graphs = {}
    total_skipped = 0

    # Ensure the base output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for class_label in sentence_graphs:
        class_output_path = os.path.join(output_dir, str(class_label))  # Use str() for class_label for path safety

        # Check if the pickled result for this class_label already exists
        if os.path.exists(class_output_path + ".pkl"):
            print(f"Found pre-processed data for class '{class_label}'. Loading from pickle...")
            with open(class_output_path + ".pkl", 'rb') as f:
                all_processed_document_graphs[class_label] = pickle.load(f)
            continue  # Skip processing this class and move to the next one

        all_processed_document_graphs[class_label] = []

        # Create a directory for the current class_label if it doesn't exist
        os.makedirs(class_output_path, exist_ok=True)

        # Группируем графы по ID документа
        document_graphs_by_id: Dict[str, List[SemanticGraph]] = defaultdict(list)
        for graph in sentence_graphs[class_label]:
            # Извлекаем ID документа из ID графа (например, 'ptg_0_0' -> '0')
            match = re.match(r"ptg_(\d+)_(\d+)", graph.id)
            if match:
                doc_id = match.group(1)
                document_graphs_by_id[doc_id].append(graph)

        for doc_id, sentence_graphs_for_doc in document_graphs_by_id.items():
            try:
                sentence_graphs_for_doc.sort(key=lambda g: int(g.id.split('_')[-1]))
                combined_doc_graph = combine_from_sentence_graphs(sentence_graphs_for_doc, stanza_verbose_load=False)
                combined_doc_graph.graph_id = f"ptg_{doc_id}"
                preprocess_graph_ptg(combined_doc_graph)
                all_processed_document_graphs[class_label].append(combined_doc_graph)
                print(f"Graphs of {doc_id} for {class_label} are preprocessed and merged")
            except Exception as e:
                total_skipped += 1
                print(f"Graphs of {doc_id} for {class_label} are preprocessed is skipped. Error: {e}")

        # Save the processed data for the current class_label
        try:
            with open(class_output_path + ".pkl", 'wb') as f:
                pickle.dump(all_processed_document_graphs[class_label], f)
            print(f"Successfully saved processed data for class '{class_label}' to {class_output_path}.pkl")
        except Exception as e:
            print(f"Error saving processed data for class '{class_label}': {e}")

    print(f"Total skipped {total_skipped}")
    return all_processed_document_graphs

if __name__ == '__main__':
    root = get_project_root()

    parser = ArgumentParser(description='Main script', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', type=Path, default=root/'config/config.yaml',
                        help='Enter the config file path.')

    parser.add_argument('--dataset', type=str, default='all', choices=['all', 'ten_newsgroups', 'bbcsport'],
                        help='Choose the dataset.')

    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'concepts', 'frequent_subgraphs', 'equivalence_classes'],
                        help='Choose the operation mode.')

    parser.add_argument('--weighting', type=str, default='yes',
                        choices=['yes', 'no'],
                        help='Choose whether to weight graph patterns.')

    parser.add_argument('--graph_pattern_reconstruction', type=str, default='yes',
                        choices=['yes', 'no'],
                        help='Choose whether to reconstruct graph patterns.')

    parser.add_argument('--visualization', type=str, default='no',
                        choices=['yes', 'no'],
                        help='Choose whether to visualize graph patterns reconstruction using a toy example.')

    args, unknown = parser.parse_known_args()

    with args.config.open() as y:
        config = yaml.load(y, Loader=yaml.FullLoader)

    do_operations(root=root,
                  config=config,
                  dataset=args.dataset,
                  mode='concepts',
                  weighting='no',
                  graph_pattern_reconstruction=args.graph_pattern_reconstruction,
                  visualization=args.visualization)