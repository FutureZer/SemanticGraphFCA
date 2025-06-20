import os
import re
import copy
import pickle
import pandas as pd
import networkx as nx
import base_functions as bf

from itertools import chain
from sklearn.metrics import classification_report


def edge_penalty_calculation(subgraphs, edge_penalties):
    edge_penalty = 0

    for subgraph in subgraphs:
        for node1, node2, data in subgraph.edges(data=True):
            edge_penalty += edge_penalties[(node1, node2, data['label'])]

    return edge_penalty


def flatten_chain(matrix):
    return list(chain.from_iterable(matrix))


def graph_pattern_scoring(weighted_class_graph_patterns, edge_penalties, classes, prefix, test_graphs, penalty):

    column = []

    for class_name in classes:

        graphs = test_graphs[class_name]

        for graph in graphs:
            score = 0
            weighted_class_graph_pattern_counter = 0

            for weighted_class_graph_pattern in weighted_class_graph_patterns:
                weighted_class_graph_pattern_counter += 1

                multiple_subsumption = bf.multiple_subsumption_check(graph, weighted_class_graph_pattern['subgraphs'])

                if multiple_subsumption == 1:
                    weighted_class_graph_pattern_weight = bf.find_graph_pattern_weight(weighted_class_graph_pattern)

                    if penalty == 'edge_penalty':
                        weighted_class_graph_pattern_edge_penalty = edge_penalty_calculation(weighted_class_graph_pattern['subgraphs'], edge_penalties)

                        if weighted_class_graph_pattern_edge_penalty == 0:
                            score += (weighted_class_graph_pattern_weight / 1)

                        else:
                            score += (weighted_class_graph_pattern_weight / weighted_class_graph_pattern_edge_penalty)

                    else:

                        if weighted_class_graph_pattern[f'{penalty}'] == 0:
                            score += (weighted_class_graph_pattern_weight / 1)

                        else:
                            score += (weighted_class_graph_pattern_weight / weighted_class_graph_pattern[f'{penalty}'])

            column.append(copy.deepcopy(score / weighted_class_graph_pattern_counter))
    return column


def graph_pattern_classification(dataset, classes, prefix, test_graphs, notation, mode):
    results = {'dataset': dataset, 'mode': mode}

    penalties = ['baseline_penalty', 'penalty_1', 'penalty_2', 'penalty_3', 'edge_penalty', 'penalty_5', 'penalty_6']

    edge_penalties_file_name = dataset + '_' + mode + '_edge_penalties.pickle'
    edge_penalties_path = prefix + '/' + '/' + edge_penalties_file_name

    with open(edge_penalties_path, 'rb') as f:
        edge_penalties = pickle.load(f)

    for penalty in penalties:

        d = {}
        y_true = []
        index_list = []

        for class_name in classes:
            weighted_class_graph_patterns_file_name = prefix + '/' + class_name + '/' + notation + '/' + class_name + '_weighted_' + mode + '.pickle'

            with open(weighted_class_graph_patterns_file_name, 'rb') as f:
                weighted_class_graph_patterns = pickle.load(f)

            d[class_name] = graph_pattern_scoring(weighted_class_graph_patterns, edge_penalties, classes, prefix, test_graphs, penalty)

        for class_name in classes:
            for j in range(len(test_graphs[class_name])):
                index_list.append(class_name + str(j))
                y_true.append(copy.deepcopy(class_name))

        classification_df = pd.DataFrame(data=d, index=index_list)

        max_value_index = classification_df.idxmax(axis="columns")

        y_pred = max_value_index.to_list()

        cr = classification_report(y_true=y_true, y_pred=y_pred, target_names=classes, output_dict=True)
        print(cr)
        results[f'{penalty}_macro-averaged_f1-score'] = cr['macro avg']['f1-score']

    return results


def graph_pattern_scoring_iterator(dataset, classes, prefix, testing_graphs, notation, mode):

    if mode == 'all':
        concepts_results = graph_pattern_classification(dataset, classes, prefix, testing_graphs, notation, 'concepts')
        print(f"Scoring for {dataset} for concepts done.")
        print(concepts_results)

        equivalence_classes_results = graph_pattern_classification(dataset, classes, prefix, testing_graphs, notation,'equivalence_classes')
        print(f"Scoring for {dataset} for equivalence classes done.")
        print(equivalence_classes_results)

        frequent_subgraphs_results = graph_pattern_classification(dataset, classes, prefix, testing_graphs, notation,'frequent_subgraphs')
        print(f"Scoring for {dataset} for frequent subgraphs done.")
        print(frequent_subgraphs_results)

        results = [concepts_results, equivalence_classes_results, frequent_subgraphs_results]

    else:
        results = [graph_pattern_classification(dataset, classes, prefix, testing_graphs, notation, mode)]

    return results
