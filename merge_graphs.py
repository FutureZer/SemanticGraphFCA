from datetime import datetime
from typing import List, Dict, Any, Tuple, Union, Optional
import stanza

from graph_model import SemanticGraph, Node, RichNode, Edge  # Assuming these are defined elsewhere as provided
from graph_preprocessing import preprocess_graph_ptg
from graph_visualize import visualize_graph_list  # Assuming this is defined elsewhere as provided
from parse import from_dict  # Assuming this is defined elsewhere as provided

def add_node_for_document_graph(label: Optional[str],  # Now this 'label' is the final desired label (NER or original)
                                concept_label: Optional[str],
                                properties: Optional[List[str]],
                                values: Optional[List[str]],
                                anchors: Optional[List[Dict[str, int]]],
                                current_max_id: int) -> Tuple[RichNode, int]:
    """
    Adds a new RichNode to the document graph with a new unique ID.
    Returns the new node and the updated max ID.
    """
    new_id = current_max_id + 1
    new_node = RichNode(node_id=new_id,
                        label=label,  # Use the passed 'label' directly
                        concept_label=concept_label,
                        properties=properties,
                        values=values,
                        anchors=anchors)
    return new_node, new_id


_nlp_stanza_instance = None


def _initialize_stanza_pipeline(verbose: bool = False):
    """
    Initializes and returns a Stanza NLP pipeline.
    Downloads models if not present.
    """
    global _nlp_stanza_instance
    if _nlp_stanza_instance is None:
        try:
            if verbose:
                print("Initializing Stanza pipeline (this may take time)...")
            stanza.download('en', processors='tokenize,mwt,pos,lemma,depparse,ner,coref', verbose=False)
            _nlp_stanza_instance = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse,ner,coref',
                                                   verbose=False)
            if verbose:
                print("Stanza pipeline initialized.")
        except Exception as e:
            print(f"ERROR: Failed to initialize Stanza: {e}")
            print("Please ensure 'stanza' and its core dependencies are installed.")
            print("Try: pip install stanza")
            raise
    return _nlp_stanza_instance


def _get_ner_type_for_span_or_word(
        start_char: int, end_char: int,
        ner_entities: List[Any], allowed_ner_types: List[str]
) -> Optional[str]:
    """
    Finds a matching NER type for a given character span.
    Prioritizes entities that completely contain the span, or have significant overlap.
    """
    best_match_type = None
    best_overlap = -1

    for ent in ner_entities:
        if ent.type not in allowed_ner_types:
            continue

        ent_start = ent.start_char
        ent_end = ent.end_char

        overlap_start = max(start_char, ent_start)
        overlap_end = min(end_char, ent_end)
        overlap = max(0, overlap_end - overlap_start)

        if overlap > 0:
            span_length = end_char - start_char
            ent_length = ent_end - ent_start

            if (ent_start <= start_char and ent_end >= end_char) or \
                    (start_char <= ent_start and end_char >= ent_end) or \
                    (min(span_length, ent_length) > 0 and (overlap / min(span_length, ent_length)) >= 0.7):
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match_type = ent.type
    return best_match_type


def combine_from_sentence_graphs(sentence_graphs: List['SemanticGraph'],
                                 stanza_verbose_load: bool = False) -> 'SemanticGraph':
    """
    Combines a list of sentence-level SemanticGraph objects into a single
    document-level SemanticGraph using Stanza for coreference resolution and NER.

    Args:
        sentence_graphs: A list of SemanticGraph objects, each representing a sentence.
        stanza_verbose_load: If True, prints verbose output during Stanza pipeline initialization.

    Returns:
        A new SemanticGraph object representing the combined document.
    """
    if not sentence_graphs:
        print("WARNING: No sentence graphs provided. Returning empty document graph.")
        return SemanticGraph(graph_id="empty_doc", input_text="", nodes=[], edges=[], tops=[], framework="ptg_merged")

    full_document_text = "\n".join([g.input_text for g in sentence_graphs])

    # Initialize Stanza
    nlp_stanza = _initialize_stanza_pipeline(verbose=stanza_verbose_load)

    doc_stanza = nlp_stanza(full_document_text)

    # --- Process Stanza NER Results for selected types ---
    # allowed_ner_types = ["PERSON", "ORG", "GPE", "EVENT"]
    allowed_ner_types = []
    filtered_ner_entities = [ent for ent in doc_stanza.entities if ent.type in allowed_ner_types]

    # --- Stanza Coreference Clusters ---
    coreference_clusters_stanza = []
    if hasattr(doc_stanza, 'coref') and doc_stanza.coref:
        for cluster in doc_stanza.coref:
            mentions = []
            for mention_span in cluster.mentions:
                sentence_index_in_doc = mention_span.sentence
                sentence_obj = doc_stanza.sentences[sentence_index_in_doc]

                start_char = sentence_obj.words[mention_span.start_word].start_char
                end_char = sentence_obj.words[mention_span.end_word - 1].end_char

                mention_text = full_document_text[start_char:end_char]

                mention_ner_label = _get_ner_type_for_span_or_word(
                    start_char, end_char, filtered_ner_entities, allowed_ner_types
                )

                mentions.append({
                    "text": mention_text,
                    "start_char": start_char,
                    "end_char": end_char,
                    "ner_label": mention_ner_label
                })
            coreference_clusters_stanza.append({"mentions": mentions})

    # --- Mapping Stanza coref clusters to PTG nodes ---
    ptg_anchors_map: Dict[
        Tuple[int, int], List[Tuple[int, RichNode]]] = {}
    current_offset = 0
    for s_idx, s_graph in enumerate(sentence_graphs):
        for node in s_graph.nodes:
            if isinstance(node, RichNode) and node.anchors:
                for anchor in node.anchors:
                    abs_start = current_offset + anchor['from']
                    abs_end = current_offset + anchor['to']
                    if (abs_start, abs_end) not in ptg_anchors_map:
                        ptg_anchors_map[(abs_start, abs_end)] = []
                    ptg_anchors_map[(abs_start, abs_end)].append((s_idx, node))
        current_offset += len(s_graph.input_text) + 1

    ptg_coreference_clusters: List[List[Tuple[int, Union[int, str]]]] = []
    ptg_node_in_coref_cluster: Dict[Tuple[int, Union[int, str]], bool] = {}

    for cluster_stanza in coreference_clusters_stanza:
        current_ptg_cluster_nodes = []
        for mention in cluster_stanza['mentions']:
            mention_start_char = mention['start_char']
            mention_end_char = mention['end_char']

            for (abs_start, abs_end), nodes_list in ptg_anchors_map.items():
                if (max(abs_start, mention_start_char) < min(abs_end, mention_end_char) or
                        (abs(abs_start - mention_start_char) <= 3 and abs(abs_end - mention_end_char) <= 3)):

                    for s_idx, ptg_node_obj in nodes_list:
                        full_ptg_id = (s_idx, ptg_node_obj.id)
                        if full_ptg_id not in ptg_node_in_coref_cluster:
                            current_ptg_cluster_nodes.append(full_ptg_id)
                            ptg_node_in_coref_cluster[full_ptg_id] = True

        if current_ptg_cluster_nodes:
            ptg_coreference_clusters.append(current_ptg_cluster_nodes)

    # --- Building the final document graph ---
    new_nodes_for_doc_graph: List[RichNode] = []
    old_to_new_node_obj_map: Dict[
        Tuple[int, Union[int, str]], RichNode] = {}
    current_max_doc_node_id: int = -1

    for cluster_idx, cluster in enumerate(ptg_coreference_clusters):
        canonical_ptg_node_candidate: Optional[RichNode] = None
        all_cluster_anchors: List[Dict[str, int]] = []
        cluster_ner_label: Optional[str] = None

        # Find a suitable canonical node and collect all anchors
        for s_idx, node_id in cluster:
            node_obj = sentence_graphs[s_idx].get_node_by_id(node_id)
            if isinstance(node_obj, RichNode):
                if node_obj.anchors:
                    sentence_offset = sum(len(sg.input_text) + 1 for sg in sentence_graphs[:s_idx])
                    for anchor in node_obj.anchors:
                        abs_anchor_from = sentence_offset + anchor['from']
                        abs_anchor_to = sentence_offset + anchor['to']
                        all_cluster_anchors.append({'from': abs_anchor_from, 'to': abs_anchor_to})

                        if cluster_ner_label is None:
                            found_ner_for_anchor = _get_ner_type_for_span_or_word(
                                abs_anchor_from, abs_anchor_to, filtered_ner_entities, allowed_ner_types
                            )
                            if found_ner_for_anchor:
                                cluster_ner_label = found_ner_for_anchor

                if node_obj.concept_label not in ["#perspron", "#cor"] and node_obj.label:
                    if canonical_ptg_node_candidate is None:
                        canonical_ptg_node_candidate = node_obj

        if canonical_ptg_node_candidate is None and cluster:
            first_node_id = cluster[0]
            canonical_ptg_node_candidate = sentence_graphs[first_node_id[0]].get_node_by_id(first_node_id[1])

        if canonical_ptg_node_candidate:
            new_doc_node, updated_max_id = add_node_for_document_graph(
                label=cluster_ner_label if cluster_ner_label is not None else canonical_ptg_node_candidate.label,
                concept_label=canonical_ptg_node_candidate.concept_label,
                properties=list(canonical_ptg_node_candidate.properties),
                values=list(canonical_ptg_node_candidate.values),
                anchors=all_cluster_anchors,
                current_max_id=current_max_doc_node_id
            )
            new_nodes_for_doc_graph.append(new_doc_node)
            current_max_doc_node_id = updated_max_id

            for s_idx, node_id in cluster:
                old_to_new_node_obj_map[(s_idx, node_id)] = new_doc_node
        else:
            print(
                f"WARNING: Could not find a suitable canonical node for Stanza coref cluster {cluster_idx + 1}. Skipping.")

    # --- Adding remaining nodes (not part of coreference clusters) ---
    for s_idx, s_graph in enumerate(sentence_graphs):
        for node in s_graph.nodes:
            if (s_idx, node.id) not in old_to_new_node_obj_map:
                node_ner_label: Optional[str] = None
                abs_anchors_for_node = []

                if isinstance(node, RichNode):
                    if node.anchors:
                        sentence_offset = sum(len(sg.input_text) + 1 for sg in sentence_graphs[:s_idx])
                        for anchor in node.anchors:
                            abs_anchor_from = sentence_offset + anchor['from']
                            abs_anchor_to = sentence_offset + anchor['to']
                            abs_anchors_for_node.append({
                                'from': abs_anchor_from,
                                'to': abs_anchor_to
                            })
                            if node_ner_label is None:
                                found_ner = _get_ner_type_for_span_or_word(
                                    abs_anchor_from, abs_anchor_to, filtered_ner_entities, allowed_ner_types
                                )
                                if found_ner:
                                    node_ner_label = found_ner

                    new_doc_node, updated_max_id = add_node_for_document_graph(
                        label=node_ner_label if node_ner_label is not None else node.label,
                        concept_label=node.concept_label,
                        properties=list(node.properties),
                        values=list(node.values),
                        anchors=abs_anchors_for_node,
                        current_max_id=current_max_doc_node_id
                    )
                    new_nodes_for_doc_graph.append(new_doc_node)
                    old_to_new_node_obj_map[(s_idx, node.id)] = new_doc_node
                    current_max_doc_node_id = updated_max_id
                else:
                    new_doc_node, updated_max_id = add_node_for_document_graph(
                        label=node.label,
                        concept_label=None,
                        properties=[],
                        values=[],
                        anchors=[],
                        current_max_id=current_max_doc_node_id
                    )
                    new_nodes_for_doc_graph.append(new_doc_node)
                    old_to_new_node_obj_map[(s_idx, node.id)] = new_doc_node
                    current_max_doc_node_id = updated_max_id

    # --- Collecting and remapping edges for the document graph ---
    new_edges_for_doc_graph: List[Edge] = []
    unique_edges_set = set()

    for s_idx, s_graph in enumerate(sentence_graphs):
        for edge in s_graph.edges:
            original_source_id = edge.source
            original_target_id = edge.target

            new_source_node_obj = old_to_new_node_obj_map.get((s_idx, original_source_id))
            new_target_node_obj = old_to_new_node_obj_map.get((s_idx, original_target_id))

            if new_source_node_obj and new_target_node_obj:
                if edge.label in ["coref.text", "coref.gram"]:
                    continue

                new_source_id = new_source_node_obj.id
                new_target_id = new_target_node_obj.id

                edge_tuple = (new_source_id, new_target_id, edge.label)
                if edge_tuple not in unique_edges_set:
                    new_edges_for_doc_graph.append(Edge(
                        source=new_source_id,
                        target=new_target_id,
                        label=edge.label,
                        attributes=edge.attributes,
                        raw_data=edge._raw_data
                    ))
                    unique_edges_set.add(edge_tuple)

    # --- Recalculating tops for the document graph ---
    combined_tops: List[Union[int, str]] = []
    for s_idx, s_graph in enumerate(sentence_graphs):
        for top_id in s_graph.tops:
            mapped_node = old_to_new_node_obj_map.get((s_idx, top_id))
            if mapped_node and mapped_node.id not in combined_tops:
                combined_tops.append(mapped_node.id)
    if not combined_tops and new_nodes_for_doc_graph:
        combined_tops.append(new_nodes_for_doc_graph[0].id)

    # --- Creating final SemanticGraph object ---
    document_graph = SemanticGraph(
        graph_id="combined_doc",
        input_text=full_document_text,
        nodes=new_nodes_for_doc_graph,
        edges=new_edges_for_doc_graph,
        tops=combined_tops,
        framework=sentence_graphs[0].framework,
        timestamp=str(datetime.now())
    )

    # --- Removing isolated nodes from document graph ---
    document_graph.remove_isolated_nodes()

    return document_graph


# --- 0. Загрузка PTG данных (предполагаем, что у вас есть эти данные в виде списка словарей) ---
# Ваши данные PTG для первых двух предложений
ptg_data_s0 = {
    "id": "ptg_0_0",
    "input": "Manchester United secured a crucial victory over Arsenal with a late goal.",
    "nodes": [
        {"id": 0, "label": "manchester united", "anchors": [{"from": 0, "to": 17}], "properties": ["sempos"], "values": ["n.denot"], "node_type": "entity"},
        {"id": 1, "label": "secure", "anchors": [{"from": 18, "to": 25}], "properties": ["frame", "sempos"], "values": ["v"], "node_type": "predicate"},
        {"id": 2, "label": "victory", "anchors": [{"from": 36, "to": 43}], "properties": ["sempos"], "values": ["n.denot"], "node_type": "entity"},
        {"id": 3, "label": "arsenal", "anchors": [{"from": 49, "to": 56}], "properties": ["sempos"], "values": ["n.denot"], "node_type": "entity"},
        {"id": 4, "label": "goal", "anchors": [{"from": 70, "to": 74}], "properties": ["sempos"], "values": ["n.denot"], "node_type": "entity"}
    ],
    "edges": [
        {"source": 1, "target": 0, "label": "agent"},
        {"source": 1, "target": 2, "label": "pat"},
        {"source": 1, "target": 3, "label": "loc"}, # Упрощено для "over Arsenal"
        {"source": 1, "target": 4, "label": "instr"} # Упрощено для "with a late goal"
    ],
    "tops": [1],
    "framework": "ptg",
    "time": "2025-06-12"
}

# Документ 0, Предложение 1 (обратите внимание на label "victory" для "win")
ptg_data_s1 = {
    "id": "ptg_0_1",
    "input": "The win boosts their hopes of finishing in the top four this season.",
    "nodes": [
        {"id": 0, "label": "victory", "anchors": [{"from": 4, "to": 7}], "properties": ["sempos"], "values": ["n.denot"], "node_type": "entity"}, # "The win" кореферентен к "victory"
        {"id": 1, "label": "boost", "anchors": [{"from": 8, "to": 14}], "properties": ["frame", "sempos"], "values": ["v"], "node_type": "predicate"},
        {"id": 2, "label": "hopes", "anchors": [{"from": 21, "to": 26}], "properties": ["sempos"], "values": ["n.denot"], "node_type": "entity"},
        {"id": 3, "label": "top four", "anchors": [{"from": 42, "to": 50}], "properties": ["sempos"], "values": ["n.denot"], "node_type": "entity"},
        {"id": 4, "label": "season", "anchors": [{"from": 60, "to": 66}], "properties": ["sempos"], "values": ["n.denot"], "node_type": "entity"}
    ],
    "edges": [
        {"source": 1, "target": 0, "label": "agent"},
        {"source": 1, "target": 2, "label": "pat"},
        {"source": 2, "target": 3, "label": "of_rel"}, # Отношение "hopes of top four"
        {"source": 1, "target": 4, "label": "twhen"} # Упрощенное временное отношение
    ],
    "tops": [1],
    "framework": "ptg",
    "time": "2025-06-12"
}

ptg_data = [
    {"id": "9", "input": "Williams was also far from content.", "nodes": [
        {"id": 0, "label": "williams", "anchors": [{"from": 0, "to": 8}], "properties": ["sempos"],
         "values": ["n.denot"]}, {"id": 1, "label": "be", "anchors": [{"from": 9, "to": 12}, {"from": 34, "to": 35}],
                                  "properties": ["frame", "sempos", "sentmod"],
                                  "values": ["en-v#ev-w218f7_u_nobody", "v", "enunc"]},
        {"id": 2, "label": "also", "anchors": [{"from": 13, "to": 17}], "properties": ["sempos"], "values": ["x"]},
        {"id": 3, "label": "far", "anchors": [{"from": 18, "to": 21}], "properties": ["sempos"],
         "values": ["adv.denot.grad.neg"]},
        {"id": 4, "label": "content", "anchors": [{"from": 22, "to": 26}, {"from": 27, "to": 34}],
         "properties": ["sempos"], "values": ["n.denot"]}, {"id": 5}],
     "edges": [{"source": 5, "target": 1, "label": "pred"}, {"source": 1, "target": 0, "label": "act"},
               {"source": 1, "target": 2, "label": "rhem"}, {"source": 1, "target": 3, "label": "loc"},
               {"source": 3, "target": 4, "label": "dir1"}], "tops": [5], "framework": "ptg", "time": "2025-04-12"},
    {"id": "8", "input": "\"I started well and finished well, but played some so-so games in the middle,\" she said.",
     "nodes": [{"id": 0, "label": "#perspron", "anchors": [], "properties": ["sempos"], "values": ["???"]},
               {"id": 1, "label": "#perspron", "anchors": [{"from": 1, "to": 2}], "properties": ["sempos"],
                "values": ["n.pron.def.pers"]},
               {"id": 2, "label": "start", "anchors": [{"from": 3, "to": 10}], "properties": ["frame", "sempos"],
                "values": ["en-v#ev-w3148f1", "v"]},
               {"id": 3, "label": "well", "anchors": [{"from": 11, "to": 15}], "properties": ["sempos"],
                "values": ["adv.denot.grad.neg"]},
               {"id": 4, "label": "and", "anchors": [{"from": 16, "to": 19}], "properties": ["sempos"],
                "values": ["x"]}, {"id": 5, "label": "#gen", "anchors": [], "properties": ["sempos"], "values": ["x"]},
               {"id": 6, "label": "finish", "anchors": [{"from": 20, "to": 28}], "properties": ["frame", "sempos"],
                "values": ["en-v#ev-w1331f1", "v"]},
               {"id": 7, "label": "well", "anchors": [{"from": 29, "to": 33}], "properties": ["sempos"],
                "values": ["adv.denot.grad.neg"]},
               {"id": 8, "label": "but", "anchors": [{"from": 35, "to": 38}], "properties": ["sempos"],
                "values": ["x"]},
               {"id": 9, "label": "play", "anchors": [{"from": 39, "to": 45}], "properties": ["sempos"],
                "values": ["v"]},
               {"id": 10, "label": "some", "anchors": [{"from": 46, "to": 50}], "properties": ["sempos"],
                "values": ["n.pron.indef"]},
               {"id": 11, "label": "so-so", "anchors": [{"from": 51, "to": 56}], "properties": ["sempos"],
                "values": ["adj.denot"]},
               {"id": 12, "label": "game", "anchors": [{"from": 57, "to": 62}], "properties": ["sempos"],
                "values": ["n.denot"]}, {"id": 13, "label": "middle",
                                         "anchors": [{"from": 63, "to": 65}, {"from": 66, "to": 69},
                                                     {"from": 70, "to": 76}], "properties": ["sempos"],
                                         "values": ["n.denot"]},
               {"id": 14, "label": "#perspron", "anchors": [{"from": 79, "to": 82}], "properties": ["sempos"],
                "values": ["n.pron.def.pers"]},
               {"id": 15, "label": "#gen", "anchors": [], "properties": ["sempos"], "values": ["x"]},
               {"id": 16, "label": "say.", "anchors": [{"from": 83, "to": 88}], "properties": ["sempos", "sentmod"],
                "values": ["v", "enunc"]}, {"id": 17}],
     "edges": [{"source": 16, "target": 15, "label": "addr"}, {"source": 17, "target": 16, "label": "pred"},
               {"source": 16, "target": 14, "label": "act"},
               {"source": 16, "target": 9, "label": "eff", "attributes": ["effective"], "values": ["true"]},
               {"source": 16, "target": 8, "label": "advs"},
               {"source": 16, "target": 2, "label": "eff", "attributes": ["effective"], "values": ["true"]},
               {"source": 12, "target": 11, "label": "rstr"},
               {"source": 6, "target": 1, "label": "act", "attributes": ["effective"], "values": [True]},
               {"source": 2, "target": 1, "label": "act", "attributes": ["effective"], "values": [True]},
               {"source": 9, "target": 12, "label": "pat"},
               {"source": 16, "target": 6, "label": "eff", "attributes": ["effective"], "values": [True]},
               {"source": 12, "target": 10, "label": "rstr"},
               {"source": 4, "target": 6, "label": "eff", "attributes": ["member"], "values": [True]},
               {"source": 8, "target": 9, "label": "eff", "attributes": ["member"], "values": [True]},
               {"source": 4, "target": 2, "label": "eff", "attributes": ["member"], "values": [True]},
               {"source": 4, "target": 1, "label": "act"}, {"source": 9, "target": 13, "label": "twhen"},
               {"source": 8, "target": 1, "label": "act"},
               {"source": 9, "target": 1, "label": "act", "attributes": ["effective"], "values": [True]},
               {"source": 8, "target": 4, "label": "conj", "attributes": ["member"], "values": [True]},
               {"source": 16, "target": 4, "label": "conj"}, {"source": 6, "target": 7, "label": "mann"},
               {"source": 2, "target": 0, "label": "act"}, {"source": 2, "target": 3, "label": "mann"},
               {"source": 6, "target": 5, "label": "pat"},
               {"source": 6, "target": 0, "label": "act", "attributes": ["effective"], "values": [True]}], "tops": [17],
     "framework": "ptg", "time": "2025-04-12"}
]


if __name__ == '__main__':
    sentence_graphs = [from_dict(d) for d in ptg_data]
    visualize_graph_list(sentence_graphs, 'smt')
    merged = combine_from_sentence_graphs(sentence_graphs, stanza_verbose_load=False)
    # preprocess_graph_ptg(merged)
    visualize_graph_list([merged], 'smt')

    print("\n--- Details of Merged Document Graph Nodes (with conditional NER as label) ---")
    print(f"Total Nodes in Merged Graph: {len(merged.nodes)}")
    for node in merged.nodes:
        if isinstance(node, RichNode):
            print(
                f"  Node ID: {node.id}, Label: '{node.label}', Concept: '{node.concept_label}', Anchors: {node.anchors}")
        else:
            print(f"  Node ID: {node.id}, Label: '{node.label}'")