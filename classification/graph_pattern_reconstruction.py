import os
import json
import copy
import pickle
import shutil
import networkx as nx
import base_functions as bf
from itertools import groupby

from read_dataset import load_graphs


def frequent_subgraphs_builder(class_name, data_pref, gsofia_pref, notation):
    frequent_subgraphs = []
    frequent_subgraph_ele = {}
    vertex_positions = {}

    frequent_subgraphs_file_name = os.path.join(gsofia_pref, f'{notation}-train', 'out', f'gsofia_{class_name}.OUT')
    label_edge_enc = os.path.join(gsofia_pref, f'{notation}-train', 'labels_enc.pkl')
    with open(label_edge_enc, 'rb') as f:
        label_encoder = pickle.load(f)

    vertex_labels = {v: k for k, v in label_encoder[class_name].node_label_to_id.items()}
    edge_labels = {e: k for k, e in label_encoder[class_name].edge_label_to_id.items()}

    with open(frequent_subgraphs_file_name) as f:
        lines = f.read().splitlines()

    g = nx.MultiDiGraph()

    frequent_subgraph_id_counter = 0

    for line in lines:
        words = line.split(" ")
        if words[0] == "#":
            frequent_subgraph_ele['supports'] = copy.deepcopy([int(words[1])])

        if words[0] == "v":
            vertex_positions[int(words[1])] = copy.deepcopy(vertex_labels[int(words[2])])

        if words[0] == "e":
            g.add_edge("EMP" if vertex_positions[int(words[1])] is None else vertex_positions[int(words[1])],
                       "EMP" if vertex_positions[int(words[2])] is None else vertex_positions[int(words[2])],
                       edge_labels[int(words[3])], label=edge_labels[int(words[3])])

        if words[0] == "x":
            frequent_subgraph_ele['subgraphs'] = [copy.deepcopy(g)]
            frequent_subgraph_ele['extent'] = words[1:]
            frequent_subgraph_ele['id'] = class_name + '_frequent_subgraph_' + str(frequent_subgraph_id_counter)

            frequent_subgraphs.append(copy.deepcopy(frequent_subgraph_ele))

            g = nx.MultiDiGraph()
            frequent_subgraph_ele = {}
            vertex_positions = {}

            frequent_subgraph_id_counter += 1

    frequent_subgraphs_file_name = class_name + '_frequent_subgraphs.pickle'

    with open(frequent_subgraphs_file_name, 'wb') as handle:
        pickle.dump(frequent_subgraphs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    frequent_subgraphs_path = os.path.join(data_pref, class_name, notation, frequent_subgraphs_file_name)
    shutil.move(frequent_subgraphs_file_name, frequent_subgraphs_path)


def find_maximal_subgraph_index(indices, frequent_subgraphs):
    maximal_subgraph_node_count = 0
    maximal_subgraph_edge_count = 0
    maximal_subgraph_index = 0

    for i in indices:
        if len(frequent_subgraphs[i]['subgraphs'][0].nodes) > maximal_subgraph_node_count:
            maximal_subgraph_node_count = len(frequent_subgraphs[i]['subgraphs'][0].nodes)
            maximal_subgraph_index = i

        elif len(frequent_subgraphs[i]['subgraphs'][0].nodes) == maximal_subgraph_node_count:
            singular_subsumption = bf.singular_subsumption_check(frequent_subgraphs[maximal_subgraph_index]['subgraphs'][0],
                                                                 frequent_subgraphs[i]['subgraphs'][0])

            if singular_subsumption == 0 and len(frequent_subgraphs[i]['subgraphs'][0].edges) > maximal_subgraph_edge_count:
                maximal_subgraph_node_count = len(frequent_subgraphs[i]['subgraphs'][0].nodes)
                maximal_subgraph_edge_count = len(frequent_subgraphs[i]['subgraphs'][0].edges)
                maximal_subgraph_index = i

    return maximal_subgraph_index


def filter_frequent_subgraphs(indices, frequent_subgraphs):
    filtered_indices = []

    while len(indices) > 0:
        maximal_subgraph_index = find_maximal_subgraph_index(indices, frequent_subgraphs)
        filtered_indices.append(maximal_subgraph_index)

        indices.remove(maximal_subgraph_index)

        bad_indices = []

        for i in indices:
            singular_subsumption = bf.singular_subsumption_check(frequent_subgraphs[maximal_subgraph_index]['subgraphs'][0],
                                                                 frequent_subgraphs[i]['subgraphs'][0])

            if singular_subsumption == 1:
                bad_indices.append(i)

        indices = [index for index in indices if index not in bad_indices]

    return filtered_indices


def concepts_builder(class_name, data_pref, gsofia_pref, notation):

    frequent_subgraphs_file_name = os.path.join(data_pref, class_name, notation, f'{class_name}_frequent_subgraphs.pickle')

    with open(frequent_subgraphs_file_name, 'rb') as f:
        frequent_subgraphs = pickle.load(f)

    concepts = []
    concept_ele = {}

    lattice_file_name = os.path.join(gsofia_pref, f'{notation}-train', 'out', f'gsofia_{class_name}.json')

    with open(lattice_file_name) as f:
        lattice = json.load(f)

    for ext_index, ext in enumerate(lattice[1]['Nodes']):
        matching_indices = [f_index for f_index, frequent_subgraph_ele in enumerate(frequent_subgraphs)
                            if set(ext['Ext']['Inds']).issubset(list(map(int, frequent_subgraph_ele['extent'])))]

        concept_ele['subgraphs'] = []
        concept_ele['supports'] = []

        filtered_indices = filter_frequent_subgraphs(matching_indices, frequent_subgraphs)

        for m in filtered_indices:
            concept_ele['subgraphs'].append(copy.deepcopy(frequent_subgraphs[m]['subgraphs'][0]))
            concept_ele['supports'].append(copy.deepcopy(frequent_subgraphs[m]['supports'][0]))

        concept_ele['extent'] = ext['Ext']['Inds']
        concept_ele['id'] = class_name + '_concept_' + str(ext_index)
        concepts.append(copy.deepcopy(concept_ele))
        concept_ele = {}

    concepts_file_name = class_name + '_concepts.pickle'

    with open(concepts_file_name, 'wb') as handle:
        pickle.dump(concepts, handle, protocol=pickle.HIGHEST_PROTOCOL)

    concepts_path = os.path.join(data_pref, class_name, notation, concepts_file_name)
    shutil.move(concepts_file_name, concepts_path)


def equivalence_classes_builder(class_name,  data_pref, notation):
    frequent_subgraphs_file_name = os.path.join(data_pref, class_name, notation, f'{class_name}_frequent_subgraphs.pickle')

    with open(frequent_subgraphs_file_name, 'rb') as f:
        frequent_subgraphs = pickle.load(f)

    equivalence_classes = []
    equivalence_class_ele = {}

    frequent_subgraphs.sort(key=lambda x: x['extent'])

    for i, (k, v) in enumerate(groupby(frequent_subgraphs, key=lambda x: x['extent'])):
        equivalence_class_ele['id'] = class_name + '_equivalence_class_' + str(i)

        v = list(v)
        equivalence_class_ele['extent'] = v[0]['extent']

        equivalence_class_ele['subgraphs'] = []
        equivalence_class_ele['supports'] = []

        for ele in v:
            equivalence_class_ele['subgraphs'].append(copy.deepcopy(ele['subgraphs'][0]))
            equivalence_class_ele['supports'].append(copy.deepcopy(ele['supports'][0]))

        equivalence_classes.append(copy.deepcopy(equivalence_class_ele))

        equivalence_class_ele = {}

    equivalence_classes_file_name = class_name + '_equivalence_classes.pickle'

    with open(equivalence_classes_file_name, 'wb') as handle:
        pickle.dump(equivalence_classes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    equivalence_classes_path = os.path.join(data_pref, class_name, notation, equivalence_classes_file_name)

    shutil.move(equivalence_classes_file_name, equivalence_classes_path)


def graph_pattern_reconstruction_iterator(classes, data_pref, gsofia_pref, notation, mode):

    # train_data, test_data = load_graphs(data_pref, notation, 0.16, 42)

    for class_name in classes:

        if mode == 'frequent_subgraphs':
            frequent_subgraphs_builder(class_name, data_pref, gsofia_pref, notation)

        elif mode == 'concepts':
            concepts_builder(class_name, data_pref, gsofia_pref, notation)

        elif mode == 'equivalence_classes':
            equivalence_classes_builder(class_name, data_pref, notation)

        elif mode == 'all':
            frequent_subgraphs_builder(class_name, data_pref, gsofia_pref, notation)
            concepts_builder(class_name, data_pref, gsofia_pref, notation)
            equivalence_classes_builder(class_name, data_pref, notation)
