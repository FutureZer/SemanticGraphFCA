import graphviz
import os
from typing import List

from graph_model import SemanticGraph, RichNode

# Directory for storing visualization files
VIS_DIR = 'visualizations'

if not os.path.exists(VIS_DIR):
    os.makedirs(VIS_DIR)

def visualize_semantic_graph(semantic_graph: SemanticGraph, output_path: str) -> None:
    """
    Visualizes a single SemanticGraph using Graphviz and saves it to the specified output path.

    Args:
        semantic_graph: The SemanticGraph object to visualize.
        output_path: The path to save the visualization (without extension, e.g., 'amr_graph').
                     Graphviz will add the appropriate extension (e.g., '.png').
    """
    dot = graphviz.Digraph(comment=f'Semantic Graph ID: {semantic_graph.id}, Framework: {semantic_graph.framework}')

    for node in semantic_graph.nodes:
        label_text = ""
        if node.label:
            label_text += node.label
        if isinstance(node, RichNode):
            if node.concept_label:
                label_text += f"\n{node.concept_label}"
            for prop, value in node.attributes.items():
                label_text += f"\n|{prop} {value}|"

        attrs = {'label': label_text}
        if not label_text:
            attrs['shape'] = 'circle'
            attrs['style'] = 'filled'
            attrs['fillcolor'] = 'black'
            attrs['fontcolor'] = 'white'
            attrs['width'] = '0.3'
            attrs['height'] = '0.3'

        dot.node(str(node.id), **attrs)

    for edge in semantic_graph.edges:
        attrs = {'label': edge.label} if edge.label else {}
        dot.edge(str(edge.source), str(edge.target), **attrs)

    try:
        dot.render(output_path, format='png', cleanup=True)
        print(f"Visualization saved to {output_path}.png")
    except graphviz.ExecutableNotFound as e:
        print(f"Error: Graphviz executable not found. Please ensure Graphviz is installed and in your system's PATH. ({e})")
    except Exception as e:
        print(f"An error occurred during Graphviz rendering: {e}")


def visualize_graph_list(semantic_graphs: List[SemanticGraph]) -> None:
    """
    Visualizes a list of SemanticGraph objects and saves each visualization
    to a temporary directory organized by framework.

    Args:
        semantic_graphs: A list of SemanticGraph objects to visualize.
    """

    for graph in semantic_graphs:
        framework_dir = os.path.join(VIS_DIR, graph.framework)
        os.makedirs(framework_dir, exist_ok=True)
        output_filename_base = os.path.join(framework_dir, str(graph.id))
        visualize_semantic_graph(graph, output_filename_base)
